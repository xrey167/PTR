//! Arms A and B: `FileLedger` alone, and `AcknowledgedLedger` (the log and its protected
//! anchor), crashed at every failpoint of every append of a seeded workload.
//!
//! Each workload is run once with a callback at every failpoint, which records the files as
//! they stand at that point and what the durability probe says about them (the *observing*
//! run); from each record the synthesizer builds the images a crash there can leave (the
//! power-loss class). A second set of runs panics at a chosen point and recovers from the real
//! files the unwinding left (the in-process process-abort class); `abort_arm` does the same
//! with child processes that call `abort()`.
//!
//! The anchored product is the code under test. Baseline B0 is the same workload without a
//! fault; baseline B1 recovers the log alone by trimming its tail and trusting what is left.

use super::check::{
    add, bump, counted, guarded, hang_limit_reached, judge, Cls, Expectation, Recovery,
};
use super::image::{check_in_model, damage_shape, fsync_lie, power_loss, Image, Policy};
use super::model::{Ev, Workload};
use super::params;
use super::probe;
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;
use fail::FailScenario;
use ptr_ledger::acknowledged::{AcknowledgedLedger, TailPolicy};
use ptr_ledger::anchor::{AnchorKey, ChainOrigin, ProtectedAnchor};
use ptr_ledger::compaction::LogPaths;
use ptr_ledger::integrity::{decode_log, encode_log, LogAnchor};
use ptr_ledger::{CommittedEvent, FileLedger};
use ptr_types::CommitIndex;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

/// Which ledger a run drives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// `FileLedger` alone, recovered against an anchor the host holds.
    File,
    /// `AcknowledgedLedger`: the log and the protected anchor that covers it.
    Ack,
}

impl Kind {
    pub fn arm(self) -> &'static str {
        match self {
            Kind::File => "file",
            Kind::Ack => "ack",
        }
    }

    pub fn points(self) -> Vec<&'static str> {
        match self {
            Kind::File => params::LOG_POINTS.to_vec(),
            Kind::Ack => params::LOG_POINTS
                .iter()
                .chain(params::ANCHOR_POINTS.iter())
                .copied()
                .collect(),
        }
    }
}

/// The scratch directory of a run, removed when it goes out of scope.
pub struct RunDir {
    base: PathBuf,
    next: u64,
}

impl RunDir {
    pub fn new(seed: u64, label: &str) -> Self {
        let base =
            std::env::temp_dir().join(format!("ptr-l001v2-{}-{seed}-{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create the scratch directory");
        Self { base, next: 0 }
    }

    pub fn path(&self) -> &Path {
        &self.base
    }

    /// A new empty directory under the base.
    pub fn fresh(&mut self) -> PathBuf {
        self.next += 1;
        let dir = self.base.join(format!("case-{}", self.next));
        std::fs::create_dir_all(&dir).expect("create a case directory");
        dir
    }
}

impl Drop for RunDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

/// The host's key for every anchor of the experiment.
pub fn key() -> AnchorKey {
    AnchorKey::from_bytes([0x71; 32])
}

pub fn paths_in(dir: &Path) -> LogPaths {
    LogPaths::new(dir, "journal")
}

pub fn log_path(dir: &Path) -> PathBuf {
    paths_in(dir).log_path(CommitIndex(0))
}

fn committed(events: &[Ev]) -> Vec<CommittedEvent> {
    events
        .iter()
        .enumerate()
        .map(|(i, event)| CommittedEvent {
            index: CommitIndex(i as u64 + 1),
            event: event.to_ledger(),
        })
        .collect()
}

/// The commitment the host holds for the first `count` events: computed from the events the
/// harness sent, with the codec, never from the file under recovery.
pub fn host_anchor(workload: &Workload, count: usize) -> LogAnchor {
    let bytes = encode_log(&committed(workload.prefix(count))).expect("encode the host's prefix");
    decode_log(&bytes)
        .expect("decode the host's prefix")
        .anchor()
}

/// The anchors the host would accept after `acked` events were acknowledged and one more was in
/// flight: the longer first, since a recovery that kept the in-flight record must match it.
fn candidates(workload: &Workload, acked: usize) -> Vec<LogAnchor> {
    let mut counts = Vec::new();
    if acked < workload.events.len() {
        counts.push(acked + 1);
    }
    counts.push(acked);
    counts
        .into_iter()
        .map(|count| host_anchor(workload, count))
        .collect()
}

/// What the harness saw at one failpoint of an observing run.
#[derive(Clone, Debug)]
pub struct Capture {
    /// Events acknowledged before this append began.
    pub idx: usize,
    pub point: &'static str,
    pub log: Vec<u8>,
    pub log_durable: usize,
    pub anchor: Vec<u8>,
    /// Every other file of the directory (the anchor's scratch file), by name.
    pub extras: Vec<(String, Vec<u8>)>,
    /// The anchor epoch the host retained before this append.
    pub witness: u64,
}

/// An observing run: the captures and the state after each acknowledgement.
#[derive(Debug, Default)]
pub struct Observation {
    pub captures: Vec<Capture>,
    /// `boundaries[k]` is the length of the log after `k` acknowledged records.
    pub boundaries: Vec<usize>,
    /// `anchors[k]` is the anchor file after `k` acknowledgements (empty for `Kind::File`).
    pub anchors: Vec<Vec<u8>>,
    pub hits: Vec<(&'static str, i64)>,
    pub clean_checked: i64,
    pub unsynced_acks: i64,
    pub append_failures: i64,
    /// The log after the whole workload.
    pub final_log: Vec<u8>,
    /// Directory synchronizations the run made, creation included.
    pub sync_parent_hits: i64,
}

#[derive(Default)]
struct Shared {
    obs: Observation,
    idx: usize,
    /// Set once the set is created: creating it publishes the first anchor through the same
    /// failpoints, and that is no append.
    armed: bool,
    boundary: usize,
    anchor_prev: Vec<u8>,
    witness: u64,
}

fn extras_of(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let log = log_path(dir);
    let anchor = paths_in(dir).anchor_path();
    let mut found: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path != &log && path != &anchor && path.is_file())
                .filter_map(|path| {
                    let name = path.file_name()?.to_string_lossy().into_owned();
                    Some((name, std::fs::read(&path).ok()?))
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

/// Run `workload` once on `kind`, recording the files at every failpoint.
pub fn observe(kind: Kind, workload: &Workload, dir: &Path, probing: bool) -> Observation {
    let scenario = FailScenario::setup();
    let shared = Arc::new(Mutex::new(Shared::default()));
    let owner: ThreadId = std::thread::current().id();
    let log = log_path(dir);
    let anchor_path = paths_in(dir).anchor_path();
    for point in kind.points() {
        let shared = Arc::clone(&shared);
        let (log, anchor_path, dir) = (log.clone(), anchor_path.clone(), dir.to_path_buf());
        fail::cfg_callback(point, move || {
            if std::thread::current().id() != owner {
                return;
            }
            let mut state = shared.lock().expect("observation state");
            if !state.armed {
                return;
            }
            let hit = state.obs.hits.iter_mut().find(|(name, _)| *name == point);
            match hit {
                Some((_, count)) => *count += 1,
                None => state.obs.hits.push((point, 1)),
            }
            let bytes = std::fs::read(&log).unwrap_or_default();
            let extras = extras_of(&dir);
            if probing {
                observe_probe(&mut state, point, &log, &dir, &extras);
            }
            let log_durable = if matches!(
                point,
                "ledger.before_record_write"
                    | "ledger.after_length_before_payload"
                    | "ledger.after_payload_before_sync"
            ) {
                state.boundary
            } else {
                bytes.len()
            };
            let capture = Capture {
                idx: state.idx,
                point,
                log: bytes,
                log_durable,
                anchor: std::fs::read(&anchor_path).unwrap_or_default(),
                extras,
                witness: state.witness,
            };
            state.obs.captures.push(capture);
        })
        .expect("register a failpoint callback");
    }
    let dir_syncs = Arc::new(std::sync::atomic::AtomicI64::new(0));
    {
        // Counted from the first call: creating the set synchronizes its directory too.
        let dir_syncs = Arc::clone(&dir_syncs);
        fail::cfg_callback(params::SYNC_PARENT_POINT, move || {
            if std::thread::current().id() == owner {
                dir_syncs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        })
        .expect("register the directory-sync callback");
    }
    match kind {
        Kind::File => run_file(workload, &shared, &log),
        Kind::Ack => run_ack(workload, dir, &shared, &log, &anchor_path),
    }
    drop(scenario);
    let mut state = shared.lock().expect("observation state");
    let mut obs = std::mem::take(&mut state.obs);
    obs.final_log = std::fs::read(&log).unwrap_or_default();
    obs.sync_parent_hits = dir_syncs.load(std::sync::atomic::Ordering::SeqCst);
    obs
}

/// The probe reads at one failpoint. Where the product claims a record durable, or is about
/// to write the next one, the files must hold no unsynced page. Only clean readings are taken
/// inside a run: asking the kernel about a dirty file starts its writeback, so a dirty reading
/// would clean what the next one is to find. That the probe sees a dirty file is shown once, by
/// `probe::calibrate`.
fn observe_probe(
    state: &mut Shared,
    point: &str,
    log: &Path,
    dir: &Path,
    extras: &[(String, Vec<u8>)],
) {
    let scratch: Vec<PathBuf> = extras.iter().map(|(name, _)| dir.join(name)).collect();
    let mut expect_clean = |paths: &[&Path]| {
        for path in paths {
            match probe::has_unsynced_pages(path) {
                Some(false) => state.obs.clean_checked += 1,
                Some(true) => state.obs.unsynced_acks += 1,
                None => {}
            }
        }
    };
    let scratch_refs: Vec<&Path> = scratch.iter().map(PathBuf::as_path).collect();
    match point {
        "ledger.before_record_write" | "ledger.after_sync_before_memory" => expect_clean(&[log]),
        // The log is durable before the anchor moves.
        "anchor.after_scratch_write_before_sync" => expect_clean(&[log]),
        // The scratch file is synced before it is renamed over the anchor.
        "anchor.after_scratch_sync_before_rename" => {
            expect_clean(&[log]);
            expect_clean(&scratch_refs);
        }
        _ => {}
    }
}

fn run_file(workload: &Workload, shared: &Arc<Mutex<Shared>>, log: &Path) {
    let mut ledger = FileLedger::create_new(log).expect("create the log");
    {
        let mut state = shared.lock().expect("state");
        state.armed = true;
        state.boundary = std::fs::metadata(log)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let boundary = state.boundary;
        state.obs.boundaries.push(boundary);
        state.obs.anchors.push(Vec::new());
    }
    for (i, event) in workload.events.iter().enumerate() {
        shared.lock().expect("state").idx = i;
        if ledger.append_durable(event.to_ledger()).is_err() {
            shared.lock().expect("state").obs.append_failures += 1;
            break;
        }
        let mut state = shared.lock().expect("state");
        state.boundary = std::fs::metadata(log)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let boundary = state.boundary;
        state.obs.boundaries.push(boundary);
        state.obs.anchors.push(Vec::new());
    }
}

fn run_ack(
    workload: &Workload,
    dir: &Path,
    shared: &Arc<Mutex<Shared>>,
    log: &Path,
    anchor_path: &Path,
) {
    let paths = paths_in(dir);
    let mut ledger = AcknowledgedLedger::create(&paths, key()).expect("create the set");
    {
        let mut state = shared.lock().expect("state");
        state.boundary = std::fs::metadata(log)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let boundary = state.boundary;
        state.obs.boundaries.push(boundary);
        state.anchor_prev = std::fs::read(anchor_path).unwrap_or_default();
        let prev = state.anchor_prev.clone();
        state.obs.anchors.push(prev);
        state.witness = ledger.epoch();
        state.armed = true;
    }
    for (i, event) in workload.events.iter().enumerate() {
        shared.lock().expect("state").idx = i;
        if ledger.append_acknowledged(event.to_ledger()).is_err() {
            shared.lock().expect("state").obs.append_failures += 1;
            break;
        }
        let mut state = shared.lock().expect("state");
        state.boundary = std::fs::metadata(log)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let boundary = state.boundary;
        state.obs.boundaries.push(boundary);
        let anchor = std::fs::read(anchor_path).unwrap_or_default();
        state.anchor_prev = anchor.clone();
        state.obs.anchors.push(anchor);
        state.witness = ledger.epoch();
    }
}

/// The class the table gives a log of `complete` records (and a partial frame after them, if
/// `partial`) under an anchor that covers `covers` records.
pub fn expected_class(complete: usize, partial: bool, covers: usize) -> Cls {
    if covers > complete {
        Cls::LostSuffix
    } else if covers == complete && !partial {
        Cls::Aligned
    } else {
        Cls::Unacknowledged {
            records: complete - covers,
            partial,
        }
    }
}

/// How many whole records a log of `len` bytes holds, and whether bytes of another follow.
fn shape_of(boundaries: &[usize], len: usize) -> (usize, bool) {
    let complete = boundaries.iter().rposition(|b| *b <= len).unwrap_or(0);
    (complete, len > boundaries[complete])
}

/// Write an image's files into a fresh directory.
fn materialize(dir: &Path, log: &[u8], anchor: Option<&[u8]>, extras: &[(String, Vec<u8>)]) {
    std::fs::write(log_path(dir), log).expect("write the log image");
    if let Some(anchor) = anchor.filter(|bytes| !bytes.is_empty()) {
        std::fs::write(paths_in(dir).anchor_path(), anchor).expect("write the anchor image");
    }
    for (name, bytes) in extras {
        std::fs::write(dir.join(name), bytes).expect("write an extra image");
    }
}

/// Recover the directory `dir` as `kind` does and report what came back.
fn recover(
    tally: &mut Tally,
    kind: Kind,
    workload: &Arc<Workload>,
    dir: &Path,
    expect: &Expectation,
) -> Option<Recovery> {
    let watchdog = std::time::Duration::from_millis(params::HANG_WATCHDOG_MS);
    let dir = dir.to_path_buf();
    let outcome = match kind {
        Kind::File => {
            let anchors = candidates(workload, expect.acked);
            guarded(watchdog, move || {
                let path = log_path(&dir);
                for trusted in anchors {
                    if let Ok(ledger) = FileLedger::recover_unacknowledged_tail(&path, trusted) {
                        return Recovery::Recovered {
                            events: ledger.events().to_vec(),
                            class: None,
                        };
                    }
                }
                Recovery::Refused { class: None }
            })
        }
        Kind::Ack => {
            let witness = expect.witness;
            guarded(watchdog, move || {
                let paths = paths_in(&dir);
                let inspected = AcknowledgedLedger::inspect(&paths, key(), witness);
                let found = inspected.as_ref().ok().map(|(split, _)| Cls::of(*split));
                match AcknowledgedLedger::open(&paths, key(), witness, TailPolicy::Acknowledge) {
                    Ok((ledger, split)) => Recovery::Recovered {
                        events: ledger.events().to_vec(),
                        class: Some(Cls::of(split)),
                    },
                    Err(ptr_ledger::AcknowledgedError::Unrecoverable(split)) => Recovery::Refused {
                        class: Some(Cls::of(split)),
                    },
                    Err(_) => Recovery::Refused { class: found },
                }
            })
        }
    };
    counted(tally, kind.arm(), outcome)
}

/// One image, ready to recover: the files it holds and what the table says of them.
pub struct Prepared {
    pub log: Vec<u8>,
    pub anchor: Option<Vec<u8>>,
    pub extras: Vec<(String, Vec<u8>)>,
    pub expect: Expectation,
}

/// Recover a prepared image in a fresh directory and judge it.
pub fn run_prepared(
    tally: &mut Tally,
    runs: &mut RunDir,
    kind: Kind,
    workload: &Arc<Workload>,
    prepared: &Prepared,
) {
    if hang_limit_reached(tally) {
        return;
    }
    let dir = runs.fresh();
    materialize(
        &dir,
        &prepared.log,
        prepared.anchor.as_deref(),
        &prepared.extras,
    );
    let expect = &prepared.expect;
    if expect.must_refuse {
        naive_control(tally, runs, kind, workload, prepared);
    }
    judge_dir(tally, kind, workload, &dir, expect);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Recover the files in `dir` as `kind` does, and judge what came back.
pub fn judge_dir(
    tally: &mut Tally,
    kind: Kind,
    workload: &Arc<Workload>,
    dir: &Path,
    expect: &Expectation,
) {
    if let Some(recovery) = recover(tally, kind, workload, dir, expect) {
        if kind == Kind::Ack && matches!(recovery, Recovery::Recovered { .. }) {
            follow_on(tally, workload, dir, &recovery);
        }
        judge(tally, kind.arm(), workload, expect, recovery);
    }
}

/// After a recovery of the set, the next event of the workload appends, is acknowledged and
/// survives a reopen: recovery must leave a ledger that can go on.
fn follow_on(tally: &mut Tally, workload: &Arc<Workload>, dir: &Path, recovery: &Recovery) {
    let Recovery::Recovered { events, .. } = recovery else {
        return;
    };
    let Some(next) = workload.events.get(events.len()) else {
        return;
    };
    if events.len() > workload.events.len()
        || events
            .iter()
            .zip(&workload.events)
            .any(|(got, want)| got.event != want.to_ledger())
    {
        return;
    }
    let next = next.to_ledger();
    let expected = events.len() + 1;
    let dir = dir.to_path_buf();
    let outcome = guarded(
        std::time::Duration::from_millis(params::HANG_WATCHDOG_MS),
        move || {
            let paths = paths_in(&dir);
            let (mut ledger, _) =
                AcknowledgedLedger::open(&paths, key(), 0, TailPolicy::Acknowledge).ok()?;
            ledger.append_acknowledged(next).ok()?;
            drop(ledger);
            let (reopened, split) =
                AcknowledgedLedger::open(&paths, key(), 0, TailPolicy::Acknowledge).ok()?;
            Some((reopened.events().len(), Cls::of(split)))
        },
    );
    match counted(tally, "ack", outcome) {
        Some(Some((len, Cls::Aligned))) if len == expected => {
            bump(tally, "ack", "follow_on_appends")
        }
        Some(_) => bump(tally, "ack", "unjustified_refusals"),
        None => {}
    }
}

/// The images a crash at each captured point can leave, with the table's reading of each.
pub fn images(
    kind: Kind,
    workload: &Workload,
    obs: &Observation,
    rng: &mut Rng,
    tally: &mut Tally,
) -> Result<Vec<Prepared>, String> {
    let mut prepared: Vec<Prepared> = Vec::new();
    let mut seen: BTreeSet<(usize, Vec<u8>, Vec<u8>, bool)> = BTreeSet::new();
    let arm = kind.arm();
    for capture in &obs.captures {
        let covers_old = capture.idx;
        let is_anchor_point = capture.point.starts_with("anchor.");
        let log_images: Vec<Image> = if is_anchor_point {
            // The log is synced before the anchor moves: nothing of it is unsynced.
            vec![power_loss(
                &capture.log,
                capture.log.len(),
                Policy::All,
                rng,
            )]
        } else {
            Policy::HONEST
                .iter()
                .map(|policy| power_loss(&capture.log, capture.log_durable, *policy, rng))
                .collect()
        };
        for image in &log_images {
            check_in_model(
                image,
                &capture.log,
                capture.log_durable.min(capture.log.len()),
            )?;
        }
        // The anchor and scratch states a crash here can leave.
        let states = anchor_states(kind, capture, obs, rng, tally)?;
        for image in &log_images {
            if image.lost_bytes {
                add(tally, arm, "images_lost_bytes", 1);
            }
            if image.damaged {
                add(tally, arm, "images_damaged", 1);
            }
            let (complete, partial) = shape_of(&obs.boundaries, image.bytes.len());
            // The shape of the zeroed sectors after the synced length: a tail of zeros alone
            // must recover, any other shape may be refused (see `Expectation::damage`).
            let damage = damage_shape(image, &capture.log, capture.log_durable);
            for state in &states {
                let key = (
                    capture.idx,
                    image.bytes.clone(),
                    [
                        state.anchor.clone().unwrap_or_default(),
                        flat(&state.extras),
                    ]
                    .concat(),
                    state.new_anchor,
                );
                if !seen.insert(key) {
                    continue;
                }
                let covers = if state.new_anchor {
                    covers_old + 1
                } else {
                    covers_old
                };
                let class = (kind == Kind::Ack && !image.damaged)
                    .then(|| expected_class(complete, partial, covers));
                prepared.push(Prepared {
                    log: image.bytes.clone(),
                    anchor: state.anchor.clone(),
                    extras: state.extras.clone(),
                    expect: Expectation {
                        acked: capture.idx,
                        attempted: capture.idx + 1,
                        must_refuse: false,
                        damage,
                        class,
                        witness: capture.witness,
                    },
                });
            }
        }
    }
    prepared.extend(lies(kind, workload, obs, tally));
    Ok(prepared)
}

fn flat(extras: &[(String, Vec<u8>)]) -> Vec<u8> {
    extras
        .iter()
        .flat_map(|(name, bytes)| name.bytes().chain(bytes.iter().copied()))
        .collect()
}

/// One anchor-and-scratch state of a capture.
struct AnchorState {
    anchor: Option<Vec<u8>>,
    extras: Vec<(String, Vec<u8>)>,
    /// The anchor on disk covers the record being appended.
    new_anchor: bool,
}

fn anchor_states(
    kind: Kind,
    capture: &Capture,
    obs: &Observation,
    rng: &mut Rng,
    tally: &mut Tally,
) -> Result<Vec<AnchorState>, String> {
    if kind == Kind::File {
        return Ok(vec![AnchorState {
            anchor: None,
            extras: Vec::new(),
            new_anchor: false,
        }]);
    }
    let old = obs.anchors[capture.idx].clone();
    let scratch_name = capture.extras.first().map(|(name, _)| name.clone());
    let scratch_bytes = capture.extras.first().map(|(_, bytes)| bytes.clone());
    let with_scratch = |bytes: Vec<u8>| -> Vec<(String, Vec<u8>)> {
        scratch_name
            .clone()
            .map(|name| vec![(name, bytes)])
            .unwrap_or_default()
    };
    let state = |anchor: &[u8], extras, new_anchor| AnchorState {
        anchor: Some(anchor.to_vec()),
        extras,
        new_anchor,
    };
    Ok(match capture.point {
        "anchor.after_scratch_write_before_sync" => {
            // The scratch file is written and unsynced, the anchor is the old one.
            let written = scratch_bytes.unwrap_or_default();
            let mut states = vec![state(&old, Vec::new(), false)];
            for policy in Policy::HONEST {
                let image = power_loss(&written, 0, policy, rng);
                check_in_model(&image, &written, 0)?;
                states.push(state(&old, with_scratch(image.bytes), false));
            }
            states
        }
        "anchor.after_scratch_sync_before_rename" => vec![
            state(&old, with_scratch(scratch_bytes.unwrap_or_default()), false),
            state(&old, Vec::new(), false),
        ],
        "anchor.after_rename_before_dir_sync" => {
            // The rename happened but its directory entry is not synced: either name may
            // be the one that survives. The pair of states is the point of the exercise.
            add(tally, "ack", "rename_image_pairs", 1);
            vec![
                state(&capture.anchor, Vec::new(), true),
                state(&old, with_scratch(capture.anchor.clone()), false),
            ]
        }
        "anchor.after_dir_sync" => vec![state(&capture.anchor, Vec::new(), true)],
        _ => vec![state(&old, Vec::new(), false)],
    })
}

/// The controls outside the honest model: acknowledged bytes lost, or the anchor file rolled
/// back below the witness. Each must be refused; the naive recovery is run on them too.
fn lies(kind: Kind, workload: &Workload, obs: &Observation, tally: &mut Tally) -> Vec<Prepared> {
    let mut out = Vec::new();
    let n = obs.boundaries.len() - 1;
    for acked in 1..=n {
        let Some(revoked_at) = workload.events[..acked].iter().rposition(Ev::is_revocation) else {
            continue;
        };
        // Only the acknowledgement of the revocation itself and the end of the run, so a
        // workload contributes a handful of lies and not one per event.
        if acked != revoked_at + 1 && acked != n {
            continue;
        }
        let post_ack: Vec<u8> = obs.final_log[..obs.boundaries[acked]].to_vec();
        for keep in [
            obs.boundaries[revoked_at],
            obs.boundaries[revoked_at]
                + (obs.boundaries[revoked_at + 1] - obs.boundaries[revoked_at]) / 2,
        ] {
            let log = fsync_lie(&post_ack, keep);
            let (complete, partial) = shape_of(&obs.boundaries, log.len());
            add(tally, kind.arm(), "lie_images", 1);
            let anchor = (kind == Kind::Ack).then(|| obs.anchors[acked].clone());
            out.push(Prepared {
                log,
                anchor,
                extras: Vec::new(),
                expect: Expectation {
                    acked,
                    attempted: acked,
                    must_refuse: true,
                    damage: None,
                    class: (kind == Kind::Ack).then(|| expected_class(complete, partial, acked)),
                    witness: obs.epoch_after(acked),
                },
            });
        }
        if kind == Kind::Ack && acked >= 2 {
            // The anchor file of an earlier epoch comes back: authentic, and older than the
            // witness the host retained.
            add(tally, "ack", "lie_images", 1);
            out.push(Prepared {
                log: post_ack.clone(),
                anchor: Some(obs.anchors[acked - 1].clone()),
                extras: Vec::new(),
                expect: Expectation {
                    acked,
                    attempted: acked,
                    must_refuse: true,
                    damage: None,
                    class: None,
                    witness: obs.epoch_after(acked),
                },
            });
        }
    }
    out
}

impl Observation {
    /// The anchor epoch after `acked` acknowledgements: the initial epoch is 1 and every
    /// acknowledgement advances it by one.
    fn epoch_after(&self, acked: usize) -> u64 {
        1 + acked as u64
    }
}

/// Baseline B1: the log alone, its tail trimmed, trusting whatever remains.
pub fn naive_recover(log: &Path) -> Option<Vec<CommittedEvent>> {
    let recoverable = FileLedger::lock_for_recovery(log, LogAnchor::empty()).ok()?;
    let anchor = ProtectedAnchor {
        epoch: 1,
        log: recoverable.tail(),
        origin: ChainOrigin::UNSET,
        base: LogAnchor::empty(),
    };
    let (ledger, _) = recoverable.repair(anchor, TailPolicy::Acknowledge).ok()?;
    Some(ledger.events().to_vec())
}

/// The control: the same lie image recovered by baseline B1, and the stale accepts it yields.
fn naive_control(
    tally: &mut Tally,
    runs: &mut RunDir,
    kind: Kind,
    workload: &Arc<Workload>,
    prepared: &Prepared,
) {
    let dir = runs.fresh();
    materialize(
        &dir,
        &prepared.log,
        prepared.anchor.as_deref(),
        &prepared.extras,
    );
    let path = log_path(&dir);
    let events = guarded(
        std::time::Duration::from_millis(params::HANG_WATCHDOG_MS),
        move || naive_recover(&path),
    );
    if let Some(Some(events)) = counted(tally, kind.arm(), events) {
        let mut scratch = Tally::new(super::metrics::LIST);
        super::check::judge_history(&mut scratch, "b1", workload, &prepared.expect, &events);
        add(tally, kind.arm(), "b1_recoveries", 1);
        add(
            tally,
            kind.arm(),
            "b1_stale_accepts",
            scratch.get("stale_accepts"),
        );
        if scratch.get("stale_accepts") > 0 {
            add(tally, kind.arm(), "b1_lie_images_accepted_stale", 1);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Baseline B0: the observing run was a workload without a fault, so recovering what it left
/// must give back every acknowledged event, and a recovery that trips any falsifier is the
/// product's defect.
pub fn reference(
    tally: &mut Tally,
    kind: Kind,
    workload: &Arc<Workload>,
    dir: &Path,
    obs: &Observation,
) {
    let acked = obs.boundaries.len().saturating_sub(1);
    let expect = Expectation {
        acked,
        attempted: acked,
        must_refuse: false,
        damage: None,
        class: (kind == Kind::Ack).then_some(Cls::Aligned),
        witness: 1 + acked as u64,
    };
    let before = super::metrics::hard_sum(tally);
    judge_dir(tally, kind, workload, dir, &expect);
    if super::metrics::hard_sum(tally) > before {
        bump(tally, kind.arm(), "reference_mismatches");
    }
}
