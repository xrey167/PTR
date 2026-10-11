//! The process-abort class: the process stops and the page cache keeps every byte it wrote.
//!
//! Two forms, both on the real files they leave and not on synthesized ones. The in-process
//! form panics at a failpoint, lets the unwinding drop the ledger (as `tests/failpoints.rs`
//! does) and recovers from what is on disk. The child form runs the workload in a child
//! process that calls `abort()` at the failpoint, so nothing is dropped, flushed or unlocked
//! by the process itself; the parent reads the acknowledgements the child printed (one line
//! per append that returned) and recovers from the directory it left.
//!
//! The table of [`state_at`] says what a process that stops at each point leaves behind, in
//! terms of complete records, a partial frame and the records the anchor covers. It is the
//! harness's own reading of the append protocol and is compared with the product's
//! classification.

use super::check::{
    add, bump, counted_append, failpoints_blocked, guarded, hang_limit_reached, Expectation,
    InjectedPanic,
};
use super::file_arm::{expected_class, judge_dir, key, log_path, paths_in, Kind, RunDir};
use super::model::Workload;
use super::params;
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;
use fail::FailScenario;
use ptr_ledger::acknowledged::AcknowledgedLedger;
use ptr_ledger::FileLedger;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What a process that stops at `point` during the append of event `idx` leaves: complete
/// records, a partial frame after them, and the records the anchor covers.
pub fn state_at(point: &str, idx: usize) -> (usize, bool, usize) {
    match point {
        "ledger.before_record_write" => (idx, false, idx),
        "ledger.after_length_before_payload" => (idx, true, idx),
        "ledger.after_payload_before_sync" | "ledger.after_sync_before_memory" => {
            (idx + 1, false, idx)
        }
        "anchor.after_scratch_write_before_sync" | "anchor.after_scratch_sync_before_rename" => {
            (idx + 1, false, idx)
        }
        // The rename is in the page cache's directory entry from here on.
        _ => (idx + 1, false, idx + 1),
    }
}

/// Which hit of `point` is the one during the append of event `idx`: every append hits every
/// log point once, and creating the set publishes the anchor once before the first append.
pub fn hit_of(kind: Kind, point: &str, idx: usize) -> usize {
    if kind == Kind::Ack && point.starts_with("anchor.") {
        idx + 1
    } else {
        idx
    }
}

/// The expectation of a directory a process left after stopping at `point` during event `idx`.
pub fn expectation(kind: Kind, point: &str, idx: usize) -> Expectation {
    let (complete, partial, covers) = state_at(point, idx);
    Expectation {
        acked: idx,
        attempted: idx + 1,
        must_refuse: false,
        damage: None,
        class: (kind == Kind::Ack).then(|| expected_class(complete, partial, covers)),
        witness: 1 + idx as u64,
    }
}

/// `count` distinct indices of the workload's events, seeded, and always one of its
/// revocations: the crashes that matter most for the claim must land on a revocation append.
pub fn sample_covering(rng: &mut Rng, workload: &Workload, count: usize) -> Vec<usize> {
    let n = workload.events.len();
    let revocations: Vec<usize> = (0..n)
        .filter(|i| workload.events[*i].is_revocation())
        .collect();
    let mut picked: Vec<usize> = Vec::new();
    if let Some(first) = revocations.get(rng.index(revocations.len().max(1))) {
        picked.push(*first);
    }
    while picked.len() < count.min(n) {
        let idx = rng.index(n);
        if !picked.contains(&idx) {
            picked.push(idx);
        }
    }
    picked.sort_unstable();
    picked
}

/// What one in-process crash run saw.
struct UnwindRun {
    /// The injected panic fired.
    injected: bool,
    /// Some other panic came out of the product.
    product_panic: bool,
    acked: usize,
    /// A create or an append returned an error on a valid workload.
    error: bool,
}

/// Run the workload on `kind` in `dir` and panic at hit `target` of `point`. The whole run,
/// failpoint scenario included, is one call so that it can be put under a watchdog.
fn unwind_case(
    kind: Kind,
    events: Vec<ptr_ledger::LedgerEvent>,
    dir: std::path::PathBuf,
    point: &'static str,
    target: usize,
) -> UnwindRun {
    let acked = Arc::new(AtomicUsize::new(0));
    let errored = Arc::new(AtomicBool::new(false));
    let hits = Arc::new(AtomicUsize::new(0));
    let owner = std::thread::current().id();
    let scenario = FailScenario::setup();
    {
        let hits = Arc::clone(&hits);
        fail::cfg_callback(point, move || {
            if std::thread::current().id() == owner && hits.fetch_add(1, Ordering::SeqCst) == target
            {
                std::panic::panic_any(InjectedPanic);
            }
        })
        .expect("register the panic callback");
    }
    let result = {
        let (acked, errored) = (Arc::clone(&acked), Arc::clone(&errored));
        catch_unwind(AssertUnwindSafe(move || match kind {
            Kind::File => {
                let Ok(mut ledger) = FileLedger::create_new(log_path(&dir)) else {
                    errored.store(true, Ordering::SeqCst);
                    return;
                };
                for event in events {
                    if ledger.append_durable(event).is_err() {
                        errored.store(true, Ordering::SeqCst);
                        return;
                    }
                    acked.fetch_add(1, Ordering::SeqCst);
                }
            }
            Kind::Ack => {
                let Ok(mut ledger) = AcknowledgedLedger::create(&paths_in(&dir), key()) else {
                    errored.store(true, Ordering::SeqCst);
                    return;
                };
                for event in events {
                    if ledger.append_acknowledged(event).is_err() {
                        errored.store(true, Ordering::SeqCst);
                        return;
                    }
                    acked.fetch_add(1, Ordering::SeqCst);
                }
            }
        }))
    };
    drop(scenario);
    UnwindRun {
        injected: matches!(&result, Err(payload) if payload.downcast_ref::<InjectedPanic>().is_some()),
        product_panic: matches!(&result, Err(payload) if payload.downcast_ref::<InjectedPanic>().is_none()),
        acked: acked.load(Ordering::SeqCst),
        error: errored.load(Ordering::SeqCst),
    }
}

/// The in-process form: panic at `point` during event `idx`, unwind, recover. A product that
/// fails an append of the valid workload, panics by itself or hangs is counted and the run goes
/// on; a failpoint the product never reaches is counted as a reference mismatch, as in a child.
pub fn unwind_runs(
    tally: &mut Tally,
    runs: &mut RunDir,
    kind: Kind,
    workload: &Arc<Workload>,
    rng: &mut Rng,
) -> Result<(), String> {
    let arm = kind.arm();
    for point in kind.points() {
        for idx in sample_covering(rng, workload, params::UNWIND_IDX_PER_POINT) {
            if failpoints_blocked(tally) || hang_limit_reached(tally) {
                return Ok(());
            }
            let dir = runs.fresh();
            let target = hit_of(kind, point, idx);
            let events: Vec<_> = workload.events.iter().map(|e| e.to_ledger()).collect();
            let call_dir = dir.clone();
            let outcome = guarded(Duration::from_millis(params::HANG_WATCHDOG_MS), move || {
                unwind_case(kind, events, call_dir, point, target)
            });
            let Some(run) = counted_append(tally, arm, outcome) else {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            };
            if run.product_panic {
                bump(tally, arm, "product_panics");
            } else if run.error {
                // The valid workload failed to append before the crash point.
                bump(tally, arm, "reference_mismatches");
            } else if !run.injected {
                // The valid workload ran to its end without reaching the failpoint: the product
                // no longer passes through it, as for a child (SkippedFailpoint).
                bump(tally, arm, "reference_mismatches");
            } else if run.acked != idx {
                bump(tally, arm, "reference_mismatches");
            } else {
                add(tally, arm, "unwind_panics_observed", 1);
                if workload.events[idx].is_revocation() {
                    add(tally, arm, "crash_points_at_revocation", 1);
                }
                judge_dir(tally, kind, workload, &dir, &expectation(kind, point, idx));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}

/// How a child process ended.
#[derive(Debug, Eq, PartialEq)]
pub enum ChildEnd {
    /// It died of SIGABRT at its failpoint, after this many acknowledgement lines.
    Aborted { acked: usize },
    /// A product call panicked before the failpoint (status 101).
    ProductPanic,
    /// A product call returned an error on a valid workload (status 4).
    ProductError,
    /// It ran to the end without reaching the failpoint (status 3).
    SkippedFailpoint,
    /// Another signal or status: the product crashed or exited by itself.
    Abnormal(String),
    /// It outlived its watchdog and was killed.
    Hung,
}

/// Run the child of `kind` on `dir` and read how it ended. Only a malformed command line of the
/// child (status 2) or an acknowledgement line out of sequence is a protocol error of the
/// harness; every other end is a [`ChildEnd`] that [`child_case`] counts.
pub fn run_child(
    kind: &str,
    dir: &std::path::Path,
    workload_seed: u64,
    point: &str,
    hit: usize,
) -> Result<ChildEnd, String> {
    use std::io::Read;
    use std::os::unix::process::ExitStatusExt;
    let exe = std::env::current_exe().map_err(|error| format!("no executable: {error}"))?;
    let mut child = std::process::Command::new(exe)
        .arg("revocation-crash-child")
        .arg(kind)
        .arg(dir)
        .arg(workload_seed.to_string())
        .arg(point)
        .arg(hit.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|error| format!("cannot spawn the child: {error}"))?;
    let deadline = Instant::now() + Duration::from_millis(params::CHILD_WATCHDOG_MS);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(ChildEnd::Hung);
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(error) => return Err(format!("cannot wait for the child: {error}")),
        }
    };
    let mut text = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut text);
    }
    match (status.signal(), status.code()) {
        (None, Some(101)) => return Ok(ChildEnd::ProductPanic),
        (None, Some(4)) => return Ok(ChildEnd::ProductError),
        (None, Some(3)) => return Ok(ChildEnd::SkippedFailpoint),
        (None, Some(2)) => {
            return Err(format!(
                "the {kind} child at {point} hit {hit} refused its command line"
            ))
        }
        (Some(6), _) => {}
        (signal, code) => {
            return Ok(ChildEnd::Abnormal(format!(
                "status {code:?}, signal {signal:?}"
            )))
        }
    }
    let mut acked = 0usize;
    for line in text.lines() {
        let expected = format!("ACK {}", acked + 1);
        if line != expected {
            return Err(format!(
                "the {kind} child printed {line:?} where {expected:?} was due"
            ));
        }
        acked += 1;
    }
    Ok(ChildEnd::Aborted { acked })
}

/// Run the child of arm `arm` (its name is its kind) and say whether it aborted where it should,
/// so that the directory it left is to be judged. A child that met a defect of the product on the
/// way (a panic, an error, a hang, another crash, a skipped failpoint, another count of
/// acknowledgements than the run without a fault gave) is counted against the product.
pub fn child_case(
    tally: &mut Tally,
    arm: &str,
    dir: &std::path::Path,
    workload_seed: u64,
    point: &str,
    idx: usize,
    hit: usize,
) -> Result<bool, String> {
    if hang_limit_reached(tally) {
        return Ok(false);
    }
    match run_child(arm, dir, workload_seed, point, hit)? {
        ChildEnd::Aborted { acked } if acked == idx => {
            add(tally, arm, "child_aborts", 1);
            Ok(true)
        }
        ChildEnd::Aborted { .. } | ChildEnd::SkippedFailpoint | ChildEnd::ProductError => {
            bump(tally, arm, "reference_mismatches");
            Ok(false)
        }
        ChildEnd::ProductPanic | ChildEnd::Abnormal(_) => {
            bump(tally, arm, "product_panics");
            Ok(false)
        }
        ChildEnd::Hung => {
            bump(tally, arm, "product_hangs");
            Ok(false)
        }
    }
}

/// The child form for `kind` ("file" or "ack"): abort at `point` during event `idx`, recover.
pub fn child_runs(
    tally: &mut Tally,
    runs: &mut RunDir,
    kind: Kind,
    workload: &Arc<Workload>,
    workload_seed: u64,
    rng: &mut Rng,
) -> Result<(), String> {
    let arm = kind.arm();
    for point in kind.points() {
        for idx in sample_covering(rng, workload, params::CHILD_IDX_PER_POINT) {
            let dir = runs.fresh();
            let hit = hit_of(kind, point, idx);
            if child_case(tally, arm, &dir, workload_seed, point, idx, hit)? {
                if workload.events[idx].is_revocation() {
                    add(tally, arm, "crash_points_at_revocation", 1);
                }
                judge_dir(tally, kind, workload, &dir, &expectation(kind, point, idx));
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_reads_the_append_protocol() {
        // Before any byte, and after the header only: the anchor covers what was acknowledged.
        assert_eq!(state_at("ledger.before_record_write", 3), (3, false, 3));
        assert_eq!(
            state_at("ledger.after_length_before_payload", 3),
            (3, true, 3)
        );
        // The record is whole, the anchor is not yet there.
        assert_eq!(
            state_at("ledger.after_payload_before_sync", 3),
            (4, false, 3)
        );
        assert_eq!(
            state_at("anchor.after_scratch_sync_before_rename", 3),
            (4, false, 3)
        );
        // The rename happened.
        assert_eq!(
            state_at("anchor.after_rename_before_dir_sync", 3),
            (4, false, 4)
        );
        assert_eq!(state_at("anchor.after_dir_sync", 3), (4, false, 4));
    }

    #[test]
    fn anchor_hits_are_counted_after_the_publication_at_creation() {
        assert_eq!(hit_of(Kind::Ack, "anchor.after_dir_sync", 0), 1);
        assert_eq!(hit_of(Kind::Ack, "ledger.before_record_write", 4), 4);
        assert_eq!(hit_of(Kind::File, "ledger.before_record_write", 4), 4);
    }

    #[test]
    fn a_sample_is_distinct_sorted_bounded_and_holds_a_revocation() {
        for seed in 0..50 {
            let workload = crate::experiments::l001::model::generate(&mut Rng::new(seed), 6, 14);
            let mut rng = Rng::new(seed);
            let picked = sample_covering(&mut rng, &workload, 2);
            assert_eq!(picked.len(), 2);
            assert!(picked.windows(2).all(|w| w[0] < w[1]));
            assert!(picked.iter().any(|i| workload.events[*i].is_revocation()));
            assert_eq!(sample_covering(&mut rng, &workload, 1).len(), 1);
        }
    }
}
