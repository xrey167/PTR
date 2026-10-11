//! Arm C: `RaftEngineLedger` over the vendored raft-engine, crashed at the engine's own
//! failpoints (`file_pipe_log::append`, `log_file::write::err`, `log_fd::sync::err`).
//!
//! The engine preallocates its log file (it reads 2 MiB long while it holds a few hundred
//! bytes) and truncates it to the written length when it closes, so the harness learns the
//! logical length of the log after each record by closing the engine after every append and
//! measuring the file. The observing run therefore reopens the engine for each event; the
//! child processes run the whole workload in one session.
//!
//! **Stated limit.** `RaftEngineLedger` has no hash chain and no protected anchor, so it cannot
//! tell that it lost an acknowledged record: its default recovery mode drops a torn tail and
//! reports success. Images that honour the fsync model are judged like every other arm's (the
//! recovered history must be the acknowledged prefix plus at most the record in flight). The
//! fsync-lie control is **descriptive only**: `undetected_rollback` counts the lies after which
//! the engine opened without error on fewer records than were acknowledged, and
//! `engine_lie_stale_accepts` the stale accepts that followed. No product work is attempted on
//! it (owner decision of 2026-10-09).

use super::abort_arm;
use super::check::{
    add, counted, counted_append, failpoints_blocked, guarded, hang_limit_reached, judge,
    Expectation, Recovery,
};
use super::file_arm::RunDir;
use super::image::{check_in_model, fsync_lie, power_loss, Image, Policy};
use super::model::{Ev, Workload};
use super::params;
use super::probe;
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;
use fail::FailScenario;
use ptr_ledger::RaftEngineLedger;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const ARM: &str = "engine";

/// How much of each file a capture keeps. The workloads write a few kilobytes; the rest of the
/// preallocated file is zeros.
const CAPTURE_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug)]
struct Capture {
    idx: usize,
    point: &'static str,
    files: Vec<(String, Vec<u8>)>,
}

/// An observing run.
#[derive(Debug, Default)]
pub struct Observation {
    captures: Vec<Capture>,
    /// `boundaries[k]`: the logical length of the log after `k` acknowledged records.
    boundaries: Vec<usize>,
    pub hits: Vec<(&'static str, i64)>,
    pub clean_checked: i64,
    pub unsynced_acks: i64,
    pub reopen_mismatches: i64,
    pub append_failures: i64,
    final_log: Vec<u8>,
}

#[derive(Default)]
struct Shared {
    obs: Observation,
    idx: usize,
    /// Set while an append runs: opening the engine writes too, and those writes are no
    /// failpoint of an append.
    armed: bool,
}

fn files_of(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut found: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_file())
                .filter_map(|entry| {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let mut bytes = std::fs::read(entry.path()).ok()?;
                    bytes.truncate(CAPTURE_BYTES);
                    Some((name, bytes))
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

fn raftlog_in(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "raftlog"))
}

fn logical_len(dir: &Path) -> usize {
    raftlog_in(dir)
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|meta| meta.len() as usize)
        .unwrap_or(0)
}

/// Run `workload` on the engine, closing and reopening it after every append, with a callback
/// on each of the engine's three failpoints.
pub fn observe(workload: &Workload, dir: &Path, probing: bool) -> Result<Observation, String> {
    let scenario = FailScenario::setup();
    let shared = Arc::new(Mutex::new(Shared::default()));
    let owner = std::thread::current().id();
    for point in params::ENGINE_POINTS {
        let shared = Arc::clone(&shared);
        let dir = dir.to_path_buf();
        fail::cfg_callback(point, move || {
            if std::thread::current().id() != owner {
                return;
            }
            let mut state = shared.lock().expect("observation state");
            if !state.armed {
                return;
            }
            match state.obs.hits.iter_mut().find(|(name, _)| *name == point) {
                Some((_, count)) => *count += 1,
                None => state.obs.hits.push((point, 1)),
            }
            // Only the clean reading is taken inside a run: asking the kernel about a dirty
            // file starts its writeback, so a dirty reading would clean what the next one is
            // to find. That the probe sees a dirty file is shown once, by `probe::calibrate`.
            if probing && point == "file_pipe_log::append" {
                let log = raftlog_in(&dir);
                match log.as_deref().and_then(probe::has_unsynced_pages) {
                    Some(false) => state.obs.clean_checked += 1,
                    Some(true) => state.obs.unsynced_acks += 1,
                    None => {}
                }
            }
            let capture = Capture {
                idx: state.idx,
                point,
                files: files_of(&dir),
            };
            state.obs.captures.push(capture);
        })
        .map_err(|reason| format!("cannot register an engine failpoint: {reason}"))?;
    }
    // Opening the engine on a directory it wrote itself must work: a refusal is the product's
    // failure to read its own log and is counted, and the run of this workload ends there.
    let finish_failed = |shared: &Arc<Mutex<Shared>>| {
        let mut state = shared.lock().expect("state");
        state.obs.append_failures += 1;
        std::mem::take(&mut state.obs)
    };
    match RaftEngineLedger::open(dir) {
        Ok(ledger) => drop(ledger),
        Err(_) => {
            drop(scenario);
            return Ok(finish_failed(&shared));
        }
    }
    shared
        .lock()
        .expect("state")
        .obs
        .boundaries
        .push(logical_len(dir));
    for (i, event) in workload.events.iter().enumerate() {
        let Ok(mut ledger) = RaftEngineLedger::open(dir) else {
            shared.lock().expect("state").obs.append_failures += 1;
            break;
        };
        let mut state = shared.lock().expect("state");
        if ledger.events().len() != i {
            state.obs.reopen_mismatches += 1;
        }
        state.idx = i;
        state.armed = true;
        drop(state);
        let appended = ledger.append_durable(event.to_ledger());
        let mut state = shared.lock().expect("state");
        state.armed = false;
        if appended.is_err() {
            state.obs.append_failures += 1;
            break;
        }
        if probing {
            // The engine answered: its files must hold no unsynced page.
            for (name, _) in files_of(dir) {
                match probe::has_unsynced_pages(&dir.join(&name)) {
                    Some(false) => state.obs.clean_checked += 1,
                    Some(true) => state.obs.unsynced_acks += 1,
                    None => {}
                }
            }
        }
        drop(state);
        drop(ledger);
        let length = logical_len(dir);
        shared.lock().expect("state").obs.boundaries.push(length);
    }
    drop(scenario);
    let mut state = shared.lock().expect("state");
    let mut obs = std::mem::take(&mut state.obs);
    obs.final_log = raftlog_in(dir)
        .and_then(|path| std::fs::read(path).ok())
        .unwrap_or_default();
    Ok(obs)
}

/// One image: the files of a directory.
struct Prepared {
    files: Vec<(String, Vec<u8>)>,
    expect: Expectation,
    /// A control outside the honest model; judged descriptively.
    lie: bool,
}

fn engine_expect(
    acked: usize,
    attempted: usize,
    damage: Option<super::image::Shape>,
    must_refuse: bool,
) -> Expectation {
    Expectation {
        acked,
        attempted,
        must_refuse,
        damage,
        class: None,
        witness: 0,
    }
}

fn with_log(files: &[(String, Vec<u8>)], log: Vec<u8>) -> Vec<(String, Vec<u8>)> {
    files
        .iter()
        .map(|(name, bytes)| {
            if name.ends_with(".raftlog") {
                (name.clone(), log.clone())
            } else {
                (name.clone(), bytes.clone())
            }
        })
        .collect()
}

fn images(
    workload: &Workload,
    obs: &Observation,
    rng: &mut Rng,
    tally: &mut Tally,
) -> Result<Vec<Prepared>, String> {
    let n = workload.events.len();
    // Both checks in this function are preconditions of the image model and exit 2 (harness
    // integrity), not product findings: the boundaries are lengths the harness measured itself,
    // and the model builds an image from the bytes up to the logical end of the log on the
    // assumption that a fresh engine log is zero after it (the file is preallocated). Where
    // that does not hold (a recycled file, a changed layout) the images would not be what a
    // crash leaves, and a verdict on them would be about the harness. They never count against
    // the product; the preregistration (section 1) says so.
    if obs.boundaries.len() != n + 1 {
        return Err(format!(
            "the engine run measured {} boundaries for {n} events",
            obs.boundaries.len()
        ));
    }
    let mut out = Vec::new();
    for capture in &obs.captures {
        let (durable, end) = match capture.point {
            "file_pipe_log::append" => (obs.boundaries[capture.idx], obs.boundaries[capture.idx]),
            _ => (obs.boundaries[capture.idx], obs.boundaries[capture.idx + 1]),
        };
        let raftlog = capture
            .files
            .iter()
            .find(|(name, _)| name.ends_with(".raftlog"))
            .map(|(_, bytes)| bytes)
            .ok_or("an engine capture holds no log file")?;
        if raftlog.len() < end || raftlog[end..].iter().any(|byte| *byte != 0) {
            return Err(format!(
                "the engine log at {} is not zeros after its logical end {end}",
                capture.point
            ));
        }
        let written = &raftlog[..end];
        let mut seen: Vec<Vec<u8>> = Vec::new();
        for policy in Policy::HONEST {
            let image: Image = power_loss(written, durable, policy, rng);
            check_in_model(&image, written, durable)?;
            if seen.contains(&image.bytes) {
                continue;
            }
            seen.push(image.bytes.clone());
            if image.lost_bytes {
                add(tally, ARM, "images_lost_bytes", 1);
            }
            if image.damaged {
                add(tally, ARM, "images_damaged", 1);
            }
            let damage = super::image::damage_shape(&image, written, durable);
            out.push(Prepared {
                files: with_log(&capture.files, image.bytes),
                expect: engine_expect(capture.idx, capture.idx + 1, damage, false),
                lie: false,
            });
        }
    }
    // The fsync lie: acknowledged bytes are gone.
    let template = obs
        .captures
        .last()
        .map(|capture| capture.files.clone())
        .unwrap_or_default();
    for acked in 1..=n {
        let Some(revoked_at) = workload.events[..acked].iter().rposition(Ev::is_revocation) else {
            continue;
        };
        if acked != revoked_at + 1 && acked != n {
            continue;
        }
        let post_ack = &obs.final_log[..obs.boundaries[acked].min(obs.final_log.len())];
        for keep in [
            obs.boundaries[revoked_at],
            obs.boundaries[revoked_at]
                + (obs.boundaries[revoked_at + 1] - obs.boundaries[revoked_at]) / 2,
        ] {
            add(tally, ARM, "lie_images", 1);
            out.push(Prepared {
                files: with_log(&template, fsync_lie(post_ack, keep)),
                expect: engine_expect(acked, acked, None, true),
                lie: true,
            });
        }
    }
    Ok(out)
}

/// Recover the engine in `dir`.
fn recover(tally: &mut Tally, dir: &Path) -> Option<Recovery> {
    let dir = dir.to_path_buf();
    let outcome = guarded(
        std::time::Duration::from_millis(params::HANG_WATCHDOG_MS),
        move || match RaftEngineLedger::open(&dir) {
            Ok(ledger) => Recovery::Recovered {
                events: ledger.events().to_vec(),
                class: None,
            },
            Err(_) => Recovery::Refused { class: None },
        },
    );
    counted(tally, ARM, outcome)
}

/// Recover a directory the engine was crashed in and judge it with the hard counters.
pub fn judge_dir(tally: &mut Tally, workload: &Arc<Workload>, dir: &Path, expect: &Expectation) {
    if let Some(recovery) = recover(tally, dir) {
        judge(tally, ARM, workload, expect, recovery);
    }
}

/// Everything arm C does with one workload: the baseline, the power-loss images, the lies and
/// the child processes.
pub fn run_workload(
    tally: &mut Tally,
    runs: &mut RunDir,
    workload: &Arc<Workload>,
    workload_seed: u64,
    rng: &mut Rng,
    probing: bool,
    children: bool,
) -> Result<Observation, String> {
    if failpoints_blocked(tally) || hang_limit_reached(tally) {
        return Ok(Observation::default());
    }
    let dir = runs.fresh();
    let (wl, observed_dir) = ((**workload).clone(), dir.clone());
    let outcome = guarded(
        std::time::Duration::from_millis(params::HANG_WATCHDOG_MS),
        move || observe(&wl, &observed_dir, probing),
    );
    let obs = match counted_append(tally, ARM, outcome) {
        Some(observed) => observed?,
        None => return Ok(Observation::default()),
    };
    if obs.append_failures > 0 {
        // The valid workload failed on the engine itself; counted by the caller.
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(obs);
    }
    // Every append that returned ran through the pipe log's append, its write and the sync of
    // the log file, once each. An append that answered without reaching the sync did not
    // synchronize its record, whatever the probe says of the pages; one that never reached the
    // append or the write was not the engine's.
    let appended = obs.boundaries.len().saturating_sub(1);
    let hits_of = |point: &str| -> usize {
        obs.hits
            .iter()
            .find(|(name, _)| *name == point)
            .map_or(0, |(_, count)| *count as usize)
    };
    add(
        tally,
        ARM,
        "unsynced_acks",
        appended.saturating_sub(hits_of("log_fd::sync::err")) as i64,
    );
    let unrouted = appended
        .saturating_sub(hits_of("file_pipe_log::append"))
        .max(appended.saturating_sub(hits_of("log_file::write::err")));
    add(tally, ARM, "reference_mismatches", unrouted as i64);
    // B0: the whole run, reopened.
    if hang_limit_reached(tally) {
        return Ok(obs);
    }
    let full = engine_expect(workload.events.len(), workload.events.len(), None, false);
    let before = super::metrics::hard_sum(tally);
    judge_dir(tally, workload, &dir, &full);
    if super::metrics::hard_sum(tally) > before {
        add(tally, ARM, "reference_mismatches", 1);
    }
    let _ = std::fs::remove_dir_all(&dir);
    for prepared in images(workload, &obs, rng, tally)? {
        if hang_limit_reached(tally) {
            break;
        }
        let image_dir = runs.fresh();
        for (name, bytes) in &prepared.files {
            std::fs::write(image_dir.join(name), bytes).map_err(|error| error.to_string())?;
        }
        if prepared.lie {
            lie_control(tally, workload, &image_dir, &prepared.expect);
        } else {
            judge_dir(tally, workload, &image_dir, &prepared.expect);
        }
        let _ = std::fs::remove_dir_all(&image_dir);
    }
    if children {
        for point in params::ENGINE_POINTS {
            // A point the engine did not reach on every append of the run cannot be crashed at;
            // it has been counted above.
            if hits_of(point) < appended {
                continue;
            }
            let picked =
                abort_arm::sample_covering(rng, workload, params::ENGINE_CHILD_IDX_PER_POINT);
            for idx in picked {
                let child_dir = runs.fresh();
                if abort_arm::child_case(tally, ARM, &child_dir, workload_seed, point, idx, idx)? {
                    if workload.events[idx].is_revocation() {
                        add(tally, ARM, "crash_points_at_revocation", 1);
                    }
                    judge_dir(
                        tally,
                        workload,
                        &child_dir,
                        &engine_expect(idx, idx + 1, None, false),
                    );
                }
                let _ = std::fs::remove_dir_all(&child_dir);
            }
        }
    }
    Ok(obs)
}

/// The lie control of the engine: descriptive counters only, never a falsifier.
fn lie_control(tally: &mut Tally, workload: &Arc<Workload>, dir: &Path, expect: &Expectation) {
    let mut scratch = Tally::new(super::metrics::LIST);
    match recover(&mut scratch, dir) {
        Some(Recovery::Recovered { events, .. }) => {
            if events.len() < expect.acked {
                add(tally, ARM, "undetected_rollback", 1);
                let mut stale = Tally::new(super::metrics::LIST);
                super::check::judge_history(&mut stale, ARM, workload, expect, &events);
                add(
                    tally,
                    ARM,
                    "engine_lie_stale_accepts",
                    stale.get("stale_accepts"),
                );
            }
        }
        Some(Recovery::Refused { .. }) => add(tally, ARM, "engine_lie_refused", 1),
        None => add(tally, ARM, "engine_lie_unfinished", 1),
    }
}
