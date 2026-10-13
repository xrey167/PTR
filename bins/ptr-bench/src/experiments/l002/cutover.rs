//! Arm K: the compaction cutover of an anchored log, crashed at every position of its protocol.
//!
//! `AcknowledgedLedger::compact` builds a new log file named for the new floor, then advances
//! the protected anchor, which is the only thing that says which file is live. L001-v2 never
//! compacted. Here a seeded workload is acknowledged up to `k` events, a compacted snapshot of
//! the runtime at `k` is exported (the host's retained description of the discarded prefix), a
//! floor `f <= k` is planned, and `compact` is crashed at each of its nine failpoints: by an
//! in-process panic (the process-abort class), and by fsync-honest power-loss images of the new
//! log and of the anchor's publication. After every crash the harness recovers as a host does
//! (`AcknowledgedLedger::open` with the epoch it retained), and holds the result to the table
//! of this file: **the live floor is the old one for a crash before the anchor was renamed and
//! the new one after, the retained records are exactly the acknowledged ones above that floor,
//! the runtime restored from the snapshot and those records answers every (subject, generation)
//! as the oracle does for the whole history, orphans are reclaimed without touching the live
//! log, and the cutover can be retried and the log goes on.** Appends above the new floor are
//! crashed at the log and anchor failpoints as well.
//!
//! Controls outside the honest model (`lie_images`): the anchor of the epoch before the
//! compaction restored over the live one, and the live log cut before an acknowledged
//! revocation. Baseline B1 trusts the anchor file without a witness (and the log alone), and
//! must accept a stale revocation on them or the run is invalid.

use super::super::l001::check::{
    add, bump, counted, counted_append, failpoints_blocked, guarded, InjectedPanic,
};
use super::super::l001::image::{check_in_model, power_loss, Policy};
use super::super::l001::model::{Ev, Workload};
use super::super::l001::probe;
use super::{params, view};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;
use fail::FailScenario;
use ptr_config::PtrConfig;
use ptr_ledger::acknowledged::{AcknowledgedLedger, Split, TailPolicy};
use ptr_ledger::anchor::{AnchorKey, ChainOrigin, ProtectedAnchor};
use ptr_ledger::compaction::{CompactionDecision, CompactionPlan, LogPaths, RetentionPolicy};
use ptr_ledger::integrity::{encode_log_from, LogAnchor};
use ptr_ledger::{CommittedEvent, CompactionBarrier, FileLedger};
use ptr_runtime::compacted::CompactedAnchor;
use ptr_runtime::PtrRuntime;
use ptr_types::CommitIndex;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const ARM: &str = "cutover";

/// The scratch directory of a run, removed when it goes out of scope.
pub struct RunDir {
    base: PathBuf,
    next: u64,
}

impl RunDir {
    pub fn new(seed: u64, label: &str) -> Self {
        let base =
            std::env::temp_dir().join(format!("ptr-l002-{}-{seed}-{label}", std::process::id()));
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
    AnchorKey::from_bytes([0x72; 32])
}

pub fn paths_in(dir: &Path) -> LogPaths {
    LogPaths::new(dir, "journal")
}

fn watchdog() -> Duration {
    Duration::from_millis(params::HANG_WATCHDOG_MS)
}

/// The cutover the plan runs on a workload: `k` events acknowledged, `keep` records kept above
/// the new floor `floor = k - keep`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cut {
    pub k: usize,
    pub keep: u64,
}

impl Cut {
    pub fn floor(self) -> usize {
        self.k - self.keep as usize
    }
}

/// The cutovers of one workload: the first discards a prefix that holds a revocation (when the
/// history allows it), the second is drawn from the whole range.
pub fn cuts_for(workload: &Workload, rng: &mut Rng) -> Vec<Cut> {
    let n = workload.events.len();
    let first_revocation = workload
        .events
        .iter()
        .position(Ev::is_revocation)
        .expect("every workload revokes");
    let mut cuts = Vec::new();
    for slot in 0..params::CUTS_PER_WORKLOAD {
        let keep = params::KEEP_RECORDS[rng.index(params::KEEP_RECORDS.len())];
        let low = keep as usize + 1;
        let high = n - 1;
        let k = if slot == 0 {
            (first_revocation + keep as usize + 1 + rng.index(3)).clamp(low, high)
        } else {
            rng.range(low as u64, high as u64) as usize
        };
        cuts.push(Cut { k, keep });
    }
    cuts
}

/// Whether the history discarded by the cutover holds a revocation.
pub fn discards_revocation(workload: &Workload, cut: Cut) -> bool {
    workload.events[..cut.floor()].iter().any(Ev::is_revocation)
}

/// Whether a revocation is appended after the cutover (so that an older anchor loses it).
pub fn revokes_after(workload: &Workload, cut: Cut) -> bool {
    workload.events[cut.k..].iter().any(Ev::is_revocation)
}

/// What the table says of the live floor after a crash at `point` of `compact`: the anchor
/// names the new log from its rename on, and a process abort keeps the rename in the page cache.
pub fn new_floor_at(point: &str) -> bool {
    matches!(
        point,
        "anchor.after_rename_before_dir_sync"
            | "anchor.after_dir_sync"
            | "compaction.after_anchor_before_install"
    )
}

/// A set ready to be compacted, with what the host retained.
struct Prepared {
    ledger: AcknowledgedLedger,
    snapshot: Vec<u8>,
    anchor: CompactedAnchor,
    plan: CompactionPlan,
    epoch_before: u64,
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

/// Create the set in `dir`, acknowledge the first `k` events, export the runtime's snapshot at
/// `k` and plan the cutover. An error is a refusal of a valid step, the product's to answer for.
fn prepare(dir: &Path, workload: &Workload, cut: Cut) -> Result<Prepared, String> {
    let paths = paths_in(dir);
    let mut ledger = AcknowledgedLedger::create(&paths, key()).map_err(|e| format!("{e:?}"))?;
    for event in &workload.events[..cut.k] {
        ledger
            .append_acknowledged(event.to_ledger())
            .map_err(|e| format!("append: {e:?}"))?;
    }
    let runtime = PtrRuntime::replay(PtrConfig::default(), ledger.events())
        .map_err(|e| format!("replay: {e:?}"))?;
    let exported = runtime
        .export_compacted_snapshot()
        .map_err(|e| format!("export: {e:?}"))?;
    let barrier = CompactionBarrier {
        snapshot_covers: exported.covers(),
        all_consumers_caught_up: true,
        unresolved_revocations: 0,
    };
    let decision = ledger
        .plan_compaction(RetentionPolicy::keeping(cut.keep), barrier)
        .map_err(|e| format!("plan: {e:?}"))?;
    let CompactionDecision::Compact(plan) = decision else {
        return Err(format!("the plan did not compact: {decision:?}"));
    };
    if plan.base.index.0 != cut.floor() as u64 {
        return Err(format!(
            "the plan's floor is {}, not {}",
            plan.base.index.0,
            cut.floor()
        ));
    }
    let epoch_before = ledger.epoch();
    Ok(Prepared {
        ledger,
        snapshot: exported.bytes().to_vec(),
        anchor: exported.anchor(),
        plan,
        epoch_before,
    })
}

/// The files of a directory at one moment, by name.
fn dir_files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter_map(|path| {
                    let name = path.file_name()?.to_string_lossy().into_owned();
                    Some((name, std::fs::read(&path).ok()?))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The facts a cutover case is judged against.
#[derive(Clone)]
pub struct Case {
    pub workload: Arc<Workload>,
    pub cut: Cut,
    pub snapshot: Vec<u8>,
    pub anchor: CompactedAnchor,
    pub epoch_before: u64,
}

/// The observing run: the directory at each failpoint of the cutover, and the states around it.
#[derive(Default)]
pub struct Observation {
    pub captures: Vec<(&'static str, BTreeMap<String, Vec<u8>>)>,
    pub hits: Vec<(&'static str, i64)>,
    pub sync_parent_hits: i64,
    pub clean_checked: i64,
    pub unsynced_acks: i64,
    /// A step of the valid run was refused.
    pub failure: Option<String>,
    /// The directory just before the cutover.
    pub before: BTreeMap<String, Vec<u8>>,
    /// The directory after the whole workload was acknowledged.
    pub final_files: BTreeMap<String, Vec<u8>>,
    pub epoch_final: u64,
    pub case: Option<Case>,
}

#[derive(Default)]
struct Shared {
    obs: Observation,
    armed: bool,
}

/// Run the cutover of `cut` without a fault and then acknowledge the rest of the workload,
/// recording the directory at every failpoint of the cutover.
pub fn observe(workload: &Arc<Workload>, cut: Cut, dir: &Path, probing: bool) -> Observation {
    let scenario = FailScenario::setup();
    let mut prepared = match prepare(dir, workload, cut) {
        Ok(prepared) => prepared,
        Err(failure) => {
            drop(scenario);
            return Observation {
                failure: Some(failure),
                ..Observation::default()
            };
        }
    };
    let shared = Arc::new(Mutex::new(Shared::default()));
    let owner = std::thread::current().id();
    let paths = paths_in(dir);
    let new_log = paths.log_path(prepared.plan.base.index);
    let old_log = paths.log_path(CommitIndex(0));
    for point in params::CUTOVER_POINTS {
        let shared = Arc::clone(&shared);
        let (dir, new_log, old_log) = (dir.to_path_buf(), new_log.clone(), old_log.clone());
        let paths = paths.clone();
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
            if probing {
                // The new log is durable before the anchor may name it, and the anchor's scratch
                // file before it is renamed.
                let mut expect_clean = |path: &Path| match probe::has_unsynced_pages(path) {
                    Some(false) => state.obs.clean_checked += 1,
                    Some(true) => state.obs.unsynced_acks += 1,
                    None => {}
                };
                match point {
                    "ledger.create_after_sync_before_dir_sync"
                    | "compaction.after_log_built_before_anchor"
                    | "anchor.after_scratch_write_before_sync" => {
                        expect_clean(&old_log);
                        if new_log.exists() {
                            expect_clean(&new_log);
                        }
                    }
                    "anchor.after_scratch_sync_before_rename" => {
                        expect_clean(&new_log);
                        let mut scratch = paths.anchor_path().into_os_string();
                        scratch.push(".publishing");
                        expect_clean(&PathBuf::from(scratch));
                    }
                    _ => {}
                }
            }
            let files = dir_files(&dir);
            state.obs.captures.push((point, files));
        })
        .expect("register a failpoint callback");
    }
    let dir_syncs = Arc::new(AtomicI64::new(0));
    {
        let dir_syncs = Arc::clone(&dir_syncs);
        let shared = Arc::clone(&shared);
        fail::cfg_callback(params::SYNC_PARENT_POINT, move || {
            if std::thread::current().id() == owner && shared.lock().expect("state").armed {
                dir_syncs.fetch_add(1, Ordering::SeqCst);
            }
        })
        .expect("register the directory-sync callback");
    }
    {
        let mut state = shared.lock().expect("state");
        state.obs.before = dir_files(dir);
        state.armed = true;
    }
    let outcome = prepared.ledger.compact(prepared.plan);
    shared.lock().expect("state").armed = false;
    let mut failure = None;
    if let Err(error) = outcome {
        failure = Some(format!("compact: {error:?}"));
    } else {
        for event in &workload.events[cut.k..] {
            if let Err(error) = prepared.ledger.append_acknowledged(event.to_ledger()) {
                failure = Some(format!("append above the floor: {error:?}"));
                break;
            }
        }
    }
    let epoch_final = prepared.ledger.epoch();
    drop(prepared.ledger);
    drop(scenario);
    let mut state = shared.lock().expect("observation state");
    let mut obs = std::mem::take(&mut state.obs);
    obs.sync_parent_hits = dir_syncs.load(Ordering::SeqCst);
    obs.failure = failure;
    obs.final_files = dir_files(dir);
    obs.epoch_final = epoch_final;
    obs.case = Some(Case {
        workload: Arc::clone(workload),
        cut,
        snapshot: prepared.snapshot,
        anchor: prepared.anchor,
        epoch_before: prepared.epoch_before,
    });
    obs
}

/// What recovering a directory and going on from it produced, as plain data.
#[derive(Default)]
struct Outcome {
    open_error: Option<String>,
    aligned: bool,
    split: String,
    floor: u64,
    events: Vec<CommittedEvent>,
    orphans: Vec<PathBuf>,
    removed: Vec<PathBuf>,
    logs_after: Vec<String>,
    live_missing: bool,
    live_changed: bool,
    retry: Option<Result<(), String>>,
    floor_after_retry: u64,
    append_error: Option<String>,
    final_open_error: Option<String>,
    final_floor: u64,
    final_events: Vec<CommittedEvent>,
}

fn logs_in(dir: &Path) -> Vec<String> {
    let mut logs: Vec<String> = dir_files(dir)
        .into_keys()
        .filter(|name| name.ends_with(".log"))
        .collect();
    logs.sort();
    logs
}

/// Recover `dir` as a host does, reclaim the orphans, retry the cutover if it did not happen,
/// acknowledge the rest of the workload and open again.
fn recover_and_continue(dir: PathBuf, case: Case, witness: u64) -> Outcome {
    let mut out = Outcome::default();
    let paths = paths_in(&dir);
    let opened = AcknowledgedLedger::open(&paths, key(), witness, TailPolicy::Acknowledge);
    let (mut ledger, split) = match opened {
        Ok(opened) => opened,
        Err(error) => {
            out.open_error = Some(format!("{error:?}"));
            return out;
        }
    };
    out.aligned = split == Split::Aligned;
    out.split = format!("{split:?}");
    out.floor = ledger.anchor().base.index.0;
    out.events = ledger.events().to_vec();
    let live_log = paths.log_path(ledger.anchor().base.index);
    let live_before = std::fs::read(&live_log).unwrap_or_default();
    out.orphans = paths
        .orphans(ledger.anchor().base.index)
        .unwrap_or_default();
    out.removed = ledger.reclaim_orphans().unwrap_or_default();
    out.logs_after = logs_in(&dir);
    match std::fs::read(&live_log) {
        Ok(after) => out.live_changed = after != live_before,
        Err(_) => out.live_missing = true,
    }
    let cut = case.cut;
    if out.floor == 0 {
        // The cutover did not happen: it can be planned again and run, now that nothing
        // stands in its way.
        let barrier = CompactionBarrier {
            snapshot_covers: case.anchor.floor.index,
            all_consumers_caught_up: true,
            unresolved_revocations: 0,
        };
        let retried = ledger
            .plan_compaction(RetentionPolicy::keeping(cut.keep), barrier)
            .map_err(|e| format!("plan: {e:?}"))
            .and_then(|decision| match decision {
                CompactionDecision::Compact(plan) => ledger
                    .compact(plan)
                    .map(|_| ())
                    .map_err(|e| format!("compact: {e:?}")),
                other => Err(format!("no plan: {other:?}")),
            });
        out.floor_after_retry = ledger.anchor().base.index.0;
        out.retry = Some(retried);
    } else {
        out.floor_after_retry = out.floor;
    }
    let have = ledger.anchor().log.index.0 as usize;
    for event in &case.workload.events[have.min(case.workload.events.len())..] {
        if let Err(error) = ledger.append_acknowledged(event.to_ledger()) {
            out.append_error = Some(format!("{error:?}"));
            return out;
        }
    }
    let epoch = ledger.epoch();
    drop(ledger);
    match AcknowledgedLedger::open(&paths, key(), epoch, TailPolicy::Acknowledge) {
        Ok((reopened, _)) => {
            out.final_floor = reopened.anchor().base.index.0;
            out.final_events = reopened.events().to_vec();
        }
        Err(error) => out.final_open_error = Some(format!("{error:?}")),
    }
    out
}

/// What the table expects of a recovery.
struct Expect {
    /// The anchor on disk names the new log.
    new_floor: bool,
    /// Events acknowledged before the crash.
    acked: usize,
    /// Events whose append had begun.
    attempted: usize,
    /// The anchor must cover the log's tail exactly (a crash of the cutover moves no tail).
    aligned: bool,
}

/// Judge a recovered directory against the table. `outcome` comes from
/// [`recover_and_continue`].
fn judge_outcome(tally: &mut Tally, case: &Case, expect: &Expect, outcome: Outcome) {
    bump(tally, ARM, "recoveries");
    let workload = &*case.workload;
    let n = workload.events.len();
    if let Some(error) = &outcome.open_error {
        explain("the product refused a recoverable set", &error.clone());
        bump(tally, ARM, "refusals");
        bump(tally, ARM, "unjustified_refusals");
        return;
    }
    bump(tally, ARM, "recovered");
    let want_floor = if expect.new_floor {
        case.cut.floor() as u64
    } else {
        0
    };
    if outcome.floor != want_floor || (expect.aligned && !outcome.aligned) {
        explain(
            "the live floor or the classification differs from the table",
            &format!(
                "floor {} (want {want_floor}), split {} (new_floor {})",
                outcome.floor, outcome.split, expect.new_floor
            ),
        );
        bump(tally, ARM, "unclassified_cutover");
    }
    let floor = (outcome.floor as usize).min(workload.events.len());
    let logical = view::logical(&workload.events[..floor], &outcome.events);
    view::judge_logical(
        tally,
        ARM,
        workload,
        expect.acked,
        expect.attempted,
        &logical,
    );
    view::judge_restore(
        tally,
        ARM,
        workload,
        &case.snapshot,
        case.anchor,
        &outcome.events,
        logical.len().min(n),
        expect.acked,
    );
    // Orphans: exactly the log files the anchor does not name, all removed, the live one intact.
    let live_name = file_name_of(&paths_in(Path::new("x")).log_path(CommitIndex(outcome.floor)));
    let mut orphan_names: Vec<String> = outcome.orphans.iter().map(|p| file_name_of(p)).collect();
    orphan_names.sort();
    let mut removed_names: Vec<String> = outcome.removed.iter().map(|p| file_name_of(p)).collect();
    removed_names.sort();
    if outcome.live_missing
        || outcome.live_changed
        || orphan_names != removed_names
        || outcome.logs_after != vec![live_name.clone()]
        || orphan_names.contains(&live_name)
    {
        explain(
            "the reclaim of orphans touched the live log or left another",
            &format!(
                "orphans {orphan_names:?}, removed {removed_names:?}, logs after {:?}, live {live_name}, missing {}, changed {}",
                outcome.logs_after, outcome.live_missing, outcome.live_changed
            ),
        );
        bump(tally, ARM, "orphan_hazard");
    }
    if let Some(retry) = &outcome.retry {
        bump(tally, ARM, "retried_cutovers");
        match retry {
            Ok(()) if outcome.floor_after_retry == case.cut.floor() as u64 => {}
            Ok(()) => bump(tally, ARM, "unclassified_cutover"),
            Err(error) => {
                explain("the cutover could not be retried", error);
                bump(tally, ARM, "unjustified_refusals");
            }
        }
    }
    if let Some(error) = &outcome.append_error {
        explain("the recovered ledger could not go on", error);
        bump(tally, ARM, "unjustified_refusals");
        return;
    }
    if let Some(error) = &outcome.final_open_error {
        explain("the ledger could not be opened again", error);
        bump(tally, ARM, "unjustified_refusals");
        return;
    }
    // The end of the story: every event acknowledged, the floor where the cutover left it.
    let final_floor = (outcome.final_floor as usize).min(n);
    let end = view::logical(&workload.events[..final_floor], &outcome.final_events);
    view::judge_logical(tally, ARM, workload, n, n, &end);
    view::judge_restore(
        tally,
        ARM,
        workload,
        &case.snapshot,
        case.anchor,
        &outcome.final_events,
        end.len().min(n),
        n,
    );
    if outcome.final_floor != outcome.floor_after_retry {
        bump(tally, ARM, "unclassified_cutover");
    }
    bump(tally, ARM, "follow_on_appends");
}

/// Say on stderr why a recovery was counted against the product, when `L002_EXPLAIN` is set. A
/// diagnostic for whoever reads a failing run; nothing reads it.
pub fn explain(what: &str, detail: &str) {
    if std::env::var_os("L002_EXPLAIN").is_some() {
        let label = LABEL.with(|label| label.borrow().clone());
        eprintln!("L002 [{label}] {what}: {detail}");
    }
}

/// The text a panic carried, for [`explain`].
pub fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no text".into())
}

thread_local! {
    /// What the running case is, for the diagnostics of [`explain`].
    static LABEL: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Name the running case for the diagnostics of [`explain`].
pub fn label(text: String) {
    LABEL.with(|label| *label.borrow_mut() = text);
}

/// Recover a directory under the watchdog and judge it.
fn run_dir(tally: &mut Tally, case: &Case, dir: &Path, expect: &Expect, witness: u64) {
    let (call_dir, call_case) = (dir.to_path_buf(), case.clone());
    let outcome = guarded(watchdog(), move || {
        recover_and_continue(call_dir, call_case, witness)
    });
    if let Some(outcome) = counted(tally, ARM, outcome) {
        judge_outcome(tally, case, expect, outcome);
    }
}

fn materialize(dir: &Path, files: &BTreeMap<String, Vec<u8>>) {
    for (name, bytes) in files {
        std::fs::write(dir.join(name), bytes).expect("write an image file");
    }
}

/// Absorb the observing run's counters and judge the unfaulted baseline B0: the directory the
/// run left must recover to the whole workload.
pub fn absorb_and_reference(
    tally: &mut Tally,
    runs: &mut RunDir,
    obs: &Observation,
) -> Option<Case> {
    for (point, hits) in &obs.hits {
        tally.add(
            &format!("failpoint_hits.{}", params::hit_name(point)),
            *hits,
        );
    }
    tally.add(
        &format!(
            "failpoint_hits.{}",
            params::hit_name(params::SYNC_PARENT_POINT)
        ),
        obs.sync_parent_hits,
    );
    tally.add("probe_clean_checked", obs.clean_checked);
    add(tally, ARM, "unsynced_acks", obs.unsynced_acks);
    // The cutover synchronizes the directory twice: the new log, and the anchor's publication.
    add(
        tally,
        ARM,
        "unsynced_acks",
        (2 - obs.sync_parent_hits).max(0),
    );
    if let Some(failure) = &obs.failure {
        explain("a step of the valid run was refused", failure);
        bump(tally, ARM, "reference_mismatches");
        return None;
    }
    let case = obs.case.clone()?;
    bump(tally, ARM, "cuts");
    if discards_revocation(&case.workload, case.cut) {
        bump(tally, ARM, "cuts_over_revocation");
    }
    let dir = runs.fresh();
    materialize(&dir, &obs.final_files);
    let before = super::metrics::hard_sum(tally);
    let n = case.workload.events.len();
    // B0: the whole story, recovered through the same path and held to the same checks.
    let (call_dir, call_case) = (dir.clone(), case.clone());
    let witness = obs.epoch_final;
    let outcome = guarded(watchdog(), move || {
        recover_and_continue(call_dir, call_case, witness)
    });
    if let Some(outcome) = counted(tally, ARM, outcome) {
        judge_outcome(
            tally,
            &case,
            &Expect {
                new_floor: true,
                acked: n,
                attempted: n,
                aligned: true,
            },
            outcome,
        );
    }
    if super::metrics::hard_sum(tally) > before {
        bump(tally, ARM, "reference_mismatches");
    }
    let _ = std::fs::remove_dir_all(&dir);
    Some(case)
}

/// The in-process crashes: panic at each failpoint of `compact`, recover, judge.
pub fn unwind_runs(tally: &mut Tally, runs: &mut RunDir, case: &Case) {
    for point in params::CUTOVER_POINTS {
        if failpoints_blocked(tally) || tally.get("product_hangs") >= params::HANG_LIMIT {
            return;
        }
        let dir = runs.fresh();
        let (call_dir, call_case) = (dir.clone(), case.clone());
        let outcome = guarded(watchdog(), move || {
            crash_cutover(call_dir, call_case, point)
        });
        let Some(crash) = counted_append(tally, ARM, outcome) else {
            let _ = std::fs::remove_dir_all(&dir);
            continue;
        };
        match crash {
            Crashed::Injected => {
                add(tally, ARM, "unwind_panics_observed", 1);
                if case.cut.k > 0 && discards_revocation(&case.workload, case.cut) {
                    bump(tally, ARM, "crash_points_over_revocation");
                }
                run_dir(
                    tally,
                    case,
                    &dir,
                    &Expect {
                        new_floor: new_floor_at(point),
                        acked: case.cut.k,
                        attempted: case.cut.k,
                        aligned: true,
                    },
                    case.epoch_before,
                );
            }
            Crashed::ProductPanic => bump(tally, ARM, "product_panics"),
            Crashed::Refused(error) => {
                explain("a step of the crash run was refused", &error);
                bump(tally, ARM, "reference_mismatches");
            }
            Crashed::NotReached => {
                explain("the failpoint was never reached", point);
                bump(tally, ARM, "reference_mismatches");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

enum Crashed {
    Injected,
    ProductPanic,
    Refused(String),
    NotReached,
}

/// Prepare the set and panic at the first hit of `point` during `compact`.
fn crash_cutover(dir: PathBuf, case: Case, point: &'static str) -> Crashed {
    let scenario = FailScenario::setup();
    let mut prepared = match prepare(&dir, &case.workload, case.cut) {
        Ok(prepared) => prepared,
        Err(error) => {
            drop(scenario);
            return Crashed::Refused(error);
        }
    };
    let owner = std::thread::current().id();
    let hits = Arc::new(AtomicUsize::new(0));
    {
        let hits = Arc::clone(&hits);
        fail::cfg_callback(point, move || {
            if std::thread::current().id() == owner && hits.fetch_add(1, Ordering::SeqCst) == 0 {
                std::panic::panic_any(InjectedPanic);
            }
        })
        .expect("register the panic callback");
    }
    let plan = prepared.plan;
    let result = catch_unwind(AssertUnwindSafe(|| prepared.ledger.compact(plan)));
    drop(prepared);
    drop(scenario);
    match result {
        Err(payload) if payload.downcast_ref::<InjectedPanic>().is_some() => Crashed::Injected,
        Err(payload) => {
            explain("the product panicked", &panic_text(payload.as_ref()));
            Crashed::ProductPanic
        }
        Ok(Err(error)) => Crashed::Refused(format!("{error:?}")),
        Ok(Ok(_)) => Crashed::NotReached,
    }
}

/// The crashes of appends above the new floor: after a completed cutover, panic at each log and
/// anchor failpoint of the append of event `k + idx`.
pub fn above_floor_runs(tally: &mut Tally, runs: &mut RunDir, case: &Case, rng: &mut Rng) {
    let n = case.workload.events.len();
    let available = n - case.cut.k;
    for point in params::ABOVE_FLOOR_POINTS {
        for _ in 0..params::ABOVE_FLOOR_IDX_PER_POINT {
            if failpoints_blocked(tally) || tally.get("product_hangs") >= params::HANG_LIMIT {
                return;
            }
            let idx = rng.index(available);
            let dir = runs.fresh();
            let (call_dir, call_case) = (dir.clone(), case.clone());
            let outcome = guarded(watchdog(), move || {
                crash_above_floor(call_dir, call_case, point, idx)
            });
            let Some(crash) = counted_append(tally, ARM, outcome) else {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            };
            match crash {
                (Crashed::Injected, witness) => {
                    add(tally, ARM, "unwind_panics_observed", 1);
                    if case.workload.events[case.cut.k + idx].is_revocation() {
                        bump(tally, ARM, "crash_points_at_revocation");
                    }
                    run_dir(
                        tally,
                        case,
                        &dir,
                        &Expect {
                            new_floor: true,
                            acked: case.cut.k + idx,
                            attempted: case.cut.k + idx + 1,
                            aligned: false,
                        },
                        witness,
                    );
                }
                (Crashed::ProductPanic, _) => bump(tally, ARM, "product_panics"),
                (Crashed::Refused(error), _) => {
                    explain("a step of the above-floor run was refused", &error);
                    bump(tally, ARM, "reference_mismatches");
                }
                (Crashed::NotReached, _) => {
                    explain("the failpoint was never reached", point);
                    bump(tally, ARM, "reference_mismatches");
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// Compact without a fault, then append; panic at the `idx`-th hit of `point` after that. The
/// second member is the epoch the host last saw acknowledged.
fn crash_above_floor(dir: PathBuf, case: Case, point: &'static str, idx: usize) -> (Crashed, u64) {
    let scenario = FailScenario::setup();
    let mut prepared = match prepare(&dir, &case.workload, case.cut) {
        Ok(prepared) => prepared,
        Err(error) => {
            drop(scenario);
            return (Crashed::Refused(error), 0);
        }
    };
    let plan = prepared.plan;
    if let Err(error) = prepared.ledger.compact(plan) {
        drop(scenario);
        return (Crashed::Refused(format!("compact: {error:?}")), 0);
    }
    let owner = std::thread::current().id();
    let hits = Arc::new(AtomicUsize::new(0));
    {
        let hits = Arc::clone(&hits);
        fail::cfg_callback(point, move || {
            if std::thread::current().id() == owner && hits.fetch_add(1, Ordering::SeqCst) == idx {
                std::panic::panic_any(InjectedPanic);
            }
        })
        .expect("register the panic callback");
    }
    let witness = Arc::new(AtomicI64::new(prepared.ledger.epoch() as i64));
    let events: Vec<_> = case.workload.events[case.cut.k..]
        .iter()
        .map(Ev::to_ledger)
        .collect();
    let seen = Arc::clone(&witness);
    let mut ledger = prepared.ledger;
    let result = catch_unwind(AssertUnwindSafe(move || {
        for event in events {
            ledger
                .append_acknowledged(event)
                .map_err(|e| format!("{e:?}"))?;
            seen.store(ledger.epoch() as i64, Ordering::SeqCst);
        }
        Ok::<(), String>(())
    }));
    drop(scenario);
    let witness = witness.load(Ordering::SeqCst) as u64;
    let crashed = match result {
        Err(payload) if payload.downcast_ref::<InjectedPanic>().is_some() => Crashed::Injected,
        Err(payload) => {
            explain("the product panicked", &panic_text(payload.as_ref()));
            Crashed::ProductPanic
        }
        Ok(Err(error)) => Crashed::Refused(error),
        Ok(Ok(())) => Crashed::NotReached,
    };
    (crashed, witness)
}

/// The power-loss images and the controls built from one observing run.
pub fn images(
    tally: &mut Tally,
    runs: &mut RunDir,
    case: &Case,
    obs: &Observation,
    rng: &mut Rng,
) -> Result<(), String> {
    let paths = paths_in(Path::new("x"));
    let new_log_name = file_name_of(&paths.log_path(CommitIndex(case.cut.floor() as u64)));
    for (point, files) in &obs.captures {
        match *point {
            // The new log is written and not synced: the anchor is the old one, whatever
            // survives of the new file.
            "ledger.create_after_write_before_sync" => {
                let Some(written) = files.get(&new_log_name) else {
                    continue;
                };
                for policy in Policy::HONEST {
                    let image = power_loss(written, 0, policy, rng);
                    check_in_model(&image, written, 0)?;
                    if image.lost_bytes {
                        add(
                            tally,
                            ARM,
                            "images_lost_bytes",
                            (written.len() - image.bytes.len()).max(1) as i64,
                        );
                    }
                    if image.damaged {
                        add(tally, ARM, "images_damaged", 1);
                    }
                    let mut state = files.clone();
                    state.insert(new_log_name.clone(), image.bytes);
                    image_case(tally, runs, case, &state, false);
                }
            }
            // The new log is synced and its directory entry is not: the entry may be missing.
            "ledger.create_after_sync_before_dir_sync" => {
                let mut state = files.clone();
                if let Some(lost) = state.remove(&new_log_name) {
                    add(tally, ARM, "images_lost_bytes", lost.len() as i64);
                    image_case(tally, runs, case, &state, false);
                }
                image_case(tally, runs, case, files, false);
            }
            // The rename of the anchor happened and its directory entry did not: either name
            // may be the one that survives, as a pair.
            "anchor.after_rename_before_dir_sync" => {
                add(tally, ARM, "rename_image_pairs", 1);
                image_case(tally, runs, case, files, true);
                let mut old = files.clone();
                let anchor_name = file_name_of(&paths.anchor_path());
                let mut scratch_name = anchor_name.clone();
                scratch_name.push_str(".publishing");
                if let Some(old_anchor) = obs.before.get(&anchor_name) {
                    if let Some(new_anchor) = old.insert(anchor_name.clone(), old_anchor.clone()) {
                        old.insert(scratch_name, new_anchor);
                    }
                    image_case(tally, runs, case, &old, false);
                }
            }
            // The scratch file is written and not synced: the anchor is the old one.
            "anchor.after_scratch_write_before_sync" => {
                let anchor_name = file_name_of(&paths.anchor_path());
                let mut scratch_name = anchor_name.clone();
                scratch_name.push_str(".publishing");
                let Some(written) = files.get(&scratch_name) else {
                    continue;
                };
                for policy in Policy::HONEST {
                    let image = power_loss(written, 0, policy, rng);
                    check_in_model(&image, written, 0)?;
                    if image.lost_bytes {
                        add(
                            tally,
                            ARM,
                            "images_lost_bytes",
                            (written.len() - image.bytes.len()).max(1) as i64,
                        );
                    }
                    let mut state = files.clone();
                    state.insert(scratch_name.clone(), image.bytes);
                    image_case(tally, runs, case, &state, false);
                }
            }
            _ => {}
        }
    }
    lies(tally, runs, case, obs);
    Ok(())
}

/// Recover one image. The image's anchor says which floor it names: the table follows it.
fn image_case(
    tally: &mut Tally,
    runs: &mut RunDir,
    case: &Case,
    files: &BTreeMap<String, Vec<u8>>,
    new_floor: bool,
) {
    if tally.get("product_hangs") >= params::HANG_LIMIT {
        return;
    }
    let dir = runs.fresh();
    materialize(&dir, files);
    run_dir(
        tally,
        case,
        &dir,
        &Expect {
            new_floor,
            acked: case.cut.k,
            attempted: case.cut.k,
            aligned: true,
        },
        case.epoch_before,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The controls outside the honest model, each held to a refusal by the anchored product and
/// run through baseline B1.
fn lies(tally: &mut Tally, runs: &mut RunDir, case: &Case, obs: &Observation) {
    if !revokes_after(&case.workload, case.cut) {
        return;
    }
    let paths = paths_in(Path::new("x"));
    let anchor_name = file_name_of(&paths.anchor_path());
    let n = case.workload.events.len();
    // (a) The anchor of the epoch before the cutover comes back, authentic and older than the
    // witness. The old log still stands beside the new one.
    if let Some(old_anchor) = obs.before.get(&anchor_name) {
        let mut files = obs.final_files.clone();
        files.insert(anchor_name.clone(), old_anchor.clone());
        add(tally, ARM, "lie_images", 1);
        let dir = runs.fresh();
        materialize(&dir, &files);
        let witness = obs.epoch_final;
        let call_dir = dir.clone();
        let anchored = guarded(watchdog(), move || {
            AcknowledgedLedger::open(
                &paths_in(&call_dir),
                key(),
                witness,
                TailPolicy::Acknowledge,
            )
            .map(|(ledger, _)| ledger.events().len())
        });
        match counted(tally, ARM, anchored) {
            Some(Ok(_)) => bump(tally, ARM, "anchored_lie_accepted"),
            Some(Err(_)) => bump(tally, ARM, "refusals_lost_ack"),
            None => {}
        }
        // B1: the anchor file is trusted without a witness.
        let call_dir = dir.clone();
        let naive = guarded(watchdog(), move || {
            AcknowledgedLedger::open(&paths_in(&call_dir), key(), 0, TailPolicy::Acknowledge)
                .map(|(ledger, _)| (ledger.anchor().base.index.0, ledger.events().to_vec()))
                .ok()
        });
        if let Some(Some((floor, events))) = counted(tally, ARM, naive) {
            naive_judgement(tally, case, floor as usize, &events, n);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    // (b) The live log loses its last acknowledged revocation (cut at the record boundary and in
    // the middle of the record), the anchor being the current one.
    let floor = case.cut.floor();
    let Some(last_revocation) = (case.cut.k..n)
        .rev()
        .find(|i| case.workload.events[*i].is_revocation())
    else {
        return;
    };
    let live_name = file_name_of(&paths.log_path(CommitIndex(floor as u64)));
    let Some(live) = obs.final_files.get(&live_name) else {
        return;
    };
    let base = anchor_of_floor(&case.workload, floor);
    let retained_before = committed_range(&case.workload.events, floor, last_revocation);
    let Ok(prefix_bytes) = encode_log_from(&retained_before, base) else {
        return;
    };
    let next = committed_range(&case.workload.events, floor, last_revocation + 1);
    let Ok(through_bytes) = encode_log_from(&next, base) else {
        return;
    };
    for keep in [
        prefix_bytes.len(),
        prefix_bytes.len() + (through_bytes.len() - prefix_bytes.len()) / 2,
    ] {
        if keep >= live.len() {
            continue;
        }
        let mut files = obs.final_files.clone();
        files.insert(live_name.clone(), live[..keep].to_vec());
        add(tally, ARM, "lie_images", 1);
        let dir = runs.fresh();
        materialize(&dir, &files);
        let witness = obs.epoch_final;
        let call_dir = dir.clone();
        let anchored = guarded(watchdog(), move || {
            AcknowledgedLedger::open(
                &paths_in(&call_dir),
                key(),
                witness,
                TailPolicy::Acknowledge,
            )
            .map(|(ledger, _)| ledger.events().len())
        });
        match counted(tally, ARM, anchored) {
            Some(Ok(_)) => bump(tally, ARM, "anchored_lie_accepted"),
            Some(Err(_)) => bump(tally, ARM, "refusals_lost_ack"),
            None => {}
        }
        // B1: the log alone, its tail trimmed, whatever is left trusted.
        let path = paths.log_path(CommitIndex(floor as u64));
        let call_path = dir.join(path.file_name().unwrap_or_default());
        let naive = guarded(watchdog(), move || naive_recover(&call_path, base));
        if let Some(Some(events)) = counted(tally, ARM, naive) {
            naive_judgement(tally, case, floor, &events, n);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn committed_range(events: &[Ev], from: usize, to: usize) -> Vec<CommittedEvent> {
    events[from..to]
        .iter()
        .enumerate()
        .map(|(i, event)| CommittedEvent {
            index: CommitIndex((from + i) as u64 + 1),
            event: event.to_ledger(),
        })
        .collect()
}

/// The commitment of the first `count` events, from the codec and not from any file.
fn anchor_of_floor(workload: &Workload, count: usize) -> LogAnchor {
    if count == 0 {
        return LogAnchor::empty();
    }
    let bytes = ptr_ledger::integrity::encode_log(&committed(&workload.events[..count]))
        .expect("encode the host's prefix");
    ptr_ledger::integrity::decode_log(&bytes)
        .expect("decode the host's prefix")
        .anchor()
}

/// Baseline B1 on a compacted log: the log alone above `base`, its tail trimmed, trusted.
fn naive_recover(path: &Path, base: LogAnchor) -> Option<Vec<CommittedEvent>> {
    let recoverable = FileLedger::lock_for_recovery(path, base).ok()?;
    let anchor = ProtectedAnchor {
        epoch: 1,
        log: recoverable.tail(),
        origin: ChainOrigin::UNSET,
        base,
    };
    let (ledger, _) = recoverable.repair(anchor, TailPolicy::Acknowledge).ok()?;
    Some(ledger.events().to_vec())
}

/// What the naive recovery accepted: judged as a history, its stale accepts are counted, and a
/// lie it accepted with a stale revocation is the control tripping.
fn naive_judgement(
    tally: &mut Tally,
    case: &Case,
    floor: usize,
    events: &[CommittedEvent],
    acked: usize,
) {
    let mut scratch = Tally::new(super::metrics::LIST);
    let logical = view::logical(
        &case.workload.events[..floor.min(case.workload.events.len())],
        events,
    );
    view::judge_logical(&mut scratch, "b1", &case.workload, acked, acked, &logical);
    add(tally, ARM, "b1_recoveries", 1);
    add(tally, ARM, "b1_stale_accepts", scratch.get("stale_accepts"));
    if scratch.get("stale_accepts") > 0 {
        add(tally, ARM, "b1_lie_images_accepted_stale", 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::l001::model::generate;

    #[test]
    fn the_table_puts_the_new_floor_after_the_rename_and_not_before() {
        for point in params::CUTOVER_POINTS {
            let expected = matches!(
                point,
                "anchor.after_rename_before_dir_sync"
                    | "anchor.after_dir_sync"
                    | "compaction.after_anchor_before_install"
            );
            assert_eq!(new_floor_at(point), expected, "{point}");
        }
    }

    #[test]
    fn every_cut_leaves_records_to_discard_and_records_to_append() {
        for seed in 0..40 {
            let mut rng = Rng::new(seed);
            let workload = generate(&mut rng, params::EVENTS_MIN, params::EVENTS_MAX);
            for cut in cuts_for(&workload, &mut rng) {
                assert!(cut.floor() >= 1, "{cut:?}");
                assert!(cut.k < workload.events.len(), "{cut:?}");
                assert!(cut.keep >= 1, "{cut:?}");
            }
        }
    }

    #[test]
    fn a_cutover_runs_without_a_fault_and_recovers_to_the_whole_workload() {
        let workload = Arc::new(generate(
            &mut Rng::new(1001),
            params::EVENTS_MIN,
            params::EVENTS_MAX,
        ));
        let cut = Cut { k: 6, keep: 2 };
        let mut runs = RunDir::new(1001, "cutover-unit");
        let dir = runs.fresh();
        let obs = observe(&workload, cut, &dir, false);
        assert_eq!(obs.failure, None);
        assert_eq!(obs.captures.len(), params::CUTOVER_POINTS.len());
        let mut tally = Tally::new(super::super::metrics::LIST);
        let case = absorb_and_reference(&mut tally, &mut runs, &obs).expect("a case");
        assert_eq!(case.cut, cut);
        assert_eq!(
            super::super::metrics::hard_sum(&tally),
            0,
            "{:?}",
            tally.internal().render()
        );
        assert_eq!(tally.get("cuts"), 1);
        assert_eq!(tally.get("arms.cutover.follow_on_appends"), 1);
    }
}
