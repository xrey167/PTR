//! L004 — is a PostgreSQL projection always equal to the reference, and does it
//! refuse every foreign or rolled-back history?
//!
//! Each case generates a random ledger log that the runtime itself accepts
//! (every event kind, generations and supersessions that pass the runtime's
//! lifecycle validation, real encoded semantic deltas, effect attempts that are
//! settled or reconciled exactly once, awkward strings). The oracle is
//! `PtrRuntime::replay` of that log: its materialized state, its
//! `generation_validity` for every target and generation, its revisions. The
//! PostgreSQL projection receives the log record by record with injected
//! crashes (the client dropping an in-flight transaction, the server killing
//! the session), redeliveries and gaps, and is compared with the oracle in
//! full. Forked histories that diverge behind, at and ahead of the projection
//! must be refused; a legitimate catch-up from an older backup must not be.
//! Finally the projection is rebuilt from the log and compared again, and a
//! record PostgreSQL text cannot hold must be refused without moving the
//! watermark. With the `turso-oracle` feature the Turso projection is a third
//! implementation compared entry by entry.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use ptr_config::PtrConfig;
use ptr_ledger::integrity::{chain_anchors, sha256, LogAnchor};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_pg::{PgError, PgSubstrate, ProjectionApply};
use ptr_runtime::PtrRuntime;
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticValue};
use ptr_state::{projection_entries, ApplyOutcome, MaterializedState};
use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, Generation, ProjectId, Validity,
    VerificationLevel,
};
use tokio_postgres::Client;

use super::pg::{self, Instance};
use super::rng::Rng;

/// Names with quotes, separators, whitespace, non-ASCII and emoji: whatever a
/// capsule or subject may legitimately be called.
const NAMES: [&str; 14] = [
    "alpha",
    "beta",
    "delta-1",
    "Ωmega",
    "quote'd",
    "dq\"x",
    "sp ace",
    "semi;colon",
    "back\\slash",
    "per%cent",
    "emoji🙂",
    "colon:inside",
    "naïve",
    "tab\there",
];
const PROJECTS: [&str; 3] = ["atlas", "borealis", "ç-project"];
const CONSTRAINTS: [&str; 3] = ["budget", "latency slo", "naïve"];
const PROCEDURES: [&str; 3] = ["deploy", "roll back", "ünïcode"];

#[derive(Default)]
struct Metrics {
    records: u64,
    invalid_logs: u64,
    state_divergences: u64,
    extra_or_missing_keys: u64,
    lifecycle_checks: u64,
    lifecycle_divergences: u64,
    revision_divergences: u64,
    event_log_divergences: u64,
    watermark_divergences: u64,
    crash_injections: u64,
    client_aborts: u64,
    server_kills: u64,
    /// Crashes after which the record was found committed, and after which
    /// it had rolled back and was re-sent: both must occur for the crash test
    /// to have probed the commit boundary from both sides.
    crashes_committed: u64,
    crashes_rolled_back: u64,
    crash_recovery_failures: u64,
    redeliveries: u64,
    redelivery_failures: u64,
    gap_probes: u64,
    gap_failures: u64,
    foreign_probes: u64,
    foreign_histories_accepted: u64,
    false_refusals: u64,
    backup_catchups: u64,
    catchup_divergences: u64,
    rebuild_divergences: u64,
    rebuild_ns: u128,
    rebuild_commits: u64,
    apply_ns: u128,
    apply_commits: u64,
    /// One long ledger per run, rebuilt and replayed: the cost per commit a
    /// rebuild of a large ledger multiplies, without the fixed cost of
    /// dropping and migrating that dominates a short log.
    long_replay_commits: u64,
    long_replay_ns: u128,
    /// Two projector sessions applying the same record at once, as after a
    /// failover that left the old projector running: exactly one may apply it.
    duel_probes: u64,
    duel_failures: u64,
    turso_divergences: u64,
    /// Reads of an applied projection that failed: a projection that cannot
    /// answer at the ledger head is as wrong as one that answers wrongly.
    read_failures: u64,
    nul_probes: u64,
    nul_failures: u64,
}

impl Metrics {
    fn hard_failures(&self) -> u64 {
        self.invalid_logs
            + self.state_divergences
            + self.extra_or_missing_keys
            + self.lifecycle_divergences
            + self.revision_divergences
            + self.event_log_divergences
            + self.watermark_divergences
            + self.crash_recovery_failures
            + self.redelivery_failures
            + self.gap_failures
            + self.foreign_histories_accepted
            + self.false_refusals
            + self.catchup_divergences
            + self.rebuild_divergences
            + self.duel_failures
            + self.turso_divergences
            + self.read_failures
            + self.nul_failures
    }

    fn fields(&self) -> Vec<(&'static str, u128)> {
        vec![
            ("records", self.records.into()),
            ("invalid_logs", self.invalid_logs.into()),
            ("state_divergences", self.state_divergences.into()),
            ("extra_or_missing_keys", self.extra_or_missing_keys.into()),
            ("lifecycle_checks", self.lifecycle_checks.into()),
            ("lifecycle_divergences", self.lifecycle_divergences.into()),
            ("revision_divergences", self.revision_divergences.into()),
            ("event_log_divergences", self.event_log_divergences.into()),
            ("watermark_divergences", self.watermark_divergences.into()),
            ("crash_injections", self.crash_injections.into()),
            ("client_aborts", self.client_aborts.into()),
            ("server_kills", self.server_kills.into()),
            ("crashes_committed", self.crashes_committed.into()),
            ("crashes_rolled_back", self.crashes_rolled_back.into()),
            (
                "crash_recovery_failures",
                self.crash_recovery_failures.into(),
            ),
            ("redeliveries", self.redeliveries.into()),
            ("redelivery_failures", self.redelivery_failures.into()),
            ("gap_probes", self.gap_probes.into()),
            ("gap_failures", self.gap_failures.into()),
            ("duel_probes", self.duel_probes.into()),
            ("duel_failures", self.duel_failures.into()),
            ("foreign_probes", self.foreign_probes.into()),
            (
                "foreign_histories_accepted",
                self.foreign_histories_accepted.into(),
            ),
            ("false_refusals", self.false_refusals.into()),
            ("backup_catchups", self.backup_catchups.into()),
            ("catchup_divergences", self.catchup_divergences.into()),
            ("rebuild_divergences", self.rebuild_divergences.into()),
            ("rebuild_ns", self.rebuild_ns),
            ("rebuild_commits", self.rebuild_commits.into()),
            ("apply_ns", self.apply_ns),
            ("apply_commits", self.apply_commits.into()),
            ("long_replay_ns", self.long_replay_ns),
            ("long_replay_commits", self.long_replay_commits.into()),
            ("turso_divergences", self.turso_divergences.into()),
            ("read_failures", self.read_failures.into()),
            ("nul_probes", self.nul_probes.into()),
            ("nul_failures", self.nul_failures.into()),
        ]
    }
}

/// Report a divergence on stderr (captured in the run record) and count it.
fn diverged(counter: &mut u64, message: String) {
    if *counter < 20 {
        eprintln!("L004 divergence: {message}");
    }
    *counter += 1;
}

pub fn run(iterations: usize, seed: u64) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let (metrics, server, elapsed) = runtime.block_on(async {
        let raw = pg::raw_client().await;
        let server = pg::server_version(&raw).await;
        let started = Instant::now();
        let mut metrics = Metrics::default();
        let mut rng = Rng::new(seed);
        for case in 0..iterations {
            let mut case_rng = rng.fork(case as u64);
            run_case(&raw, seed, case, &mut case_rng, &mut metrics).await;
        }
        let mut long_rng = rng.fork(u64::MAX);
        long_replay(&raw, seed, &mut long_rng, &mut metrics).await;
        (metrics, server, started.elapsed())
    });
    let hard = metrics.hard_failures();
    let mut line = format!(
        "{{\"benchmark\":\"projection-equivalence\",\"iterations\":{iterations},\"seed\":{seed},\
         \"server\":{},\"turso_oracle\":{},\"elapsed_ns\":{}",
        pg::json_string(&server),
        cfg!(feature = "turso-oracle"),
        elapsed.as_nanos()
    );
    for (name, value) in metrics.fields() {
        line.push_str(&format!(",\"{name}\":{value}"));
    }
    line.push_str(&format!(",\"hard_failures\":{hard}}}"));
    println!("{line}");
    if hard > 0 {
        std::process::exit(1);
    }
}

/// Records in the long ledger rebuilt once per run.
const LONG_REPLAY_COMMITS: usize = 2_000;

/// Rebuild a long generated ledger onto a fresh instance, timed, and compare
/// the result with the runtime replay. A rebuild applies one transaction per
/// commit, so this is the cost a rebuild of a large ledger multiplies.
async fn long_replay(raw: &Client, seed: u64, rng: &mut Rng, metrics: &mut Metrics) {
    let log = generate_log(rng, LONG_REPLAY_COMMITS);
    let anchors = chain_anchors(&log, LogAnchor::empty()).expect("the long log chains");
    let oracle = match PtrRuntime::replay(PtrConfig::default(), &log) {
        Ok(oracle) => oracle,
        Err(error) => {
            diverged(
                &mut metrics.invalid_logs,
                format!("seed {seed}: the runtime refused the long log: {error:?}"),
            );
            return;
        }
    };
    let instance = Instance::new("l004r", seed, 0);
    let mut substrate = instance.create(raw).await;
    let started = Instant::now();
    let rebuilt = substrate.rebuild_projection().await;
    let applied = substrate.replay(&log).await;
    metrics.long_replay_ns += started.elapsed().as_nanos();
    metrics.long_replay_commits += log.len() as u64;
    let entries = substrate.state_entries(CommitIndex(log.len() as u64)).await;
    let watermark = substrate.watermark().await;
    if rebuilt.is_err()
        || !matches!(applied, Ok(count) if count == log.len() as u64)
        || !matches!(&entries, Ok(entries) if *entries == oracle.materialized_state().values)
        || !matches!(watermark, Ok(anchor) if anchor == anchors[anchors.len() - 1])
    {
        diverged(
            &mut metrics.rebuild_divergences,
            format!("seed {seed}: the long rebuild differs from the reference ({applied:?})"),
        );
    }
    substrate
        .drop_all()
        .await
        .expect("drop the long-replay instance");
}

async fn run_case(raw: &Client, seed: u64, case: usize, rng: &mut Rng, metrics: &mut Metrics) {
    // The ledger the case projects, and the records it grows by later for the
    // catch-up probe: generated together, so the extension carries every event
    // kind the generator produces, lifecycle changes included.
    let length = rng.range(40, 160) as usize;
    let extension = rng.range(3, 12) as usize;
    let extended = generate_log(rng, length + extension);
    let log = extended[..length].to_vec();
    let anchors = chain_anchors(&log, LogAnchor::empty()).expect("the generated log chains");
    metrics.records += log.len() as u64;

    // The oracle: the runtime's own replay of the log.
    let oracle = match PtrRuntime::replay(PtrConfig::default(), &log) {
        Ok(runtime) => runtime,
        Err(error) => {
            diverged(
                &mut metrics.invalid_logs,
                format!(
                    "seed {seed} case {case}: the runtime refused the generated log: {error:?}"
                ),
            );
            return;
        }
    };
    let mut reference = MaterializedState::default();
    for record in &log {
        reference.apply(record);
    }
    // A self-check of the harness, not of the projection: the runtime builds
    // its state through the same reference.
    if reference.values != oracle.materialized_state().values {
        diverged(
            &mut metrics.invalid_logs,
            format!("seed {seed} case {case}: the reference and the runtime disagree"),
        );
    }
    let extended_oracle = match PtrRuntime::replay(PtrConfig::default(), &extended) {
        Ok(runtime) => runtime,
        Err(error) => {
            diverged(
                &mut metrics.invalid_logs,
                format!("seed {seed} case {case}: the runtime refused the extension: {error:?}"),
            );
            return;
        }
    };

    let instance = Instance::new("l004", seed, case);
    let mut substrate = instance.create(raw).await;

    // Feed the log with crashes, redeliveries and gaps. Every per-record
    // decision is drawn before the feed, so a case's random stream is fixed by
    // the seed; only whether a crashed transaction had committed depends on
    // timing, and a rolled-back record is re-sent without another crash.
    let plan: Vec<RecordPlan> = (0..log.len()).map(|_| RecordPlan::draw(rng)).collect();
    let started = Instant::now();
    let mut next = 0usize;
    let mut crashed_at = None;
    while next < log.len() {
        let record = &log[next];
        let anchor = anchors[next];
        let step = &plan[next];
        if step.crash && crashed_at != Some(next) {
            crashed_at = Some(next);
            metrics.crash_injections += 1;
            let (fresh, returned) = crash_during_apply(
                &instance, raw, substrate, record, anchor, step.kill, step.delay, metrics,
            )
            .await;
            substrate = fresh;
            let returned = match returned {
                Ok(returned) => returned,
                Err(error) => {
                    diverged(
                        &mut metrics.crash_recovery_failures,
                        format!("seed {seed} case {case}: {error}"),
                    );
                    break;
                }
            };
            let watermark = match substrate.watermark().await {
                Ok(watermark) => watermark,
                Err(error) => {
                    diverged(
                        &mut metrics.read_failures,
                        format!("seed {seed} case {case}: the watermark after a crash is unreadable: {error:?}"),
                    );
                    break;
                }
            };
            let index = record.index.0;
            // What the projector told its caller before the crash reached it:
            // a record reported applied must be found committed.
            let acknowledged = matches!(
                &returned,
                Some(Ok(report)) if report.outcome == ApplyOutcome::Applied
            );
            if watermark.index.0 == index && watermark == anchor {
                metrics.crashes_committed += 1;
            } else if watermark.index.0 + 1 == index
                && (index == 1 || watermark == anchors[next - 1])
            {
                if acknowledged {
                    diverged(
                        &mut metrics.crash_recovery_failures,
                        format!(
                            "seed {seed} case {case}: record {index} was reported applied but rolled back"
                        ),
                    );
                    break;
                }
                // Rolled back means nothing of the record may be visible: the
                // projection must still be the reference at the prefix.
                let problems =
                    check_prefix(&mut substrate, raw, &instance.prefix, &log[..next]).await;
                if !problems.is_empty() {
                    diverged(
                        &mut metrics.crash_recovery_failures,
                        format!(
                            "seed {seed} case {case}: after a rolled-back crash at {index}: {}",
                            problems.join("; ")
                        ),
                    );
                    break;
                }
                metrics.crashes_rolled_back += 1;
                continue;
            } else {
                diverged(
                    &mut metrics.crash_recovery_failures,
                    format!(
                        "seed {seed} case {case}: after a crash at {index} the watermark is {}",
                        watermark.index.0
                    ),
                );
                break;
            }
        } else if step.duel {
            // A second projector session applies the same record at the same
            // moment: the watermark row lock must let exactly one apply it and
            // show the other a verified duplicate.
            metrics.duel_probes += 1;
            let mut rival = instance.connect().await;
            let (first, second) = tokio::join!(
                substrate.apply_committed(record, anchor),
                rival.apply_committed(record, anchor),
            );
            drop(rival);
            let outcomes = [
                first.map(|report| report.outcome),
                second.map(|report| report.outcome),
            ];
            let count = |wanted: ApplyOutcome| {
                outcomes
                    .iter()
                    .filter(|outcome| matches!(outcome, Ok(found) if *found == wanted))
                    .count()
            };
            let applied = count(ApplyOutcome::Applied);
            if applied != 1 || count(ApplyOutcome::Duplicate) != 1 {
                diverged(
                    &mut metrics.duel_failures,
                    format!(
                        "seed {seed} case {case}: two projectors applying {} gave {outcomes:?}",
                        record.index.0
                    ),
                );
                if applied == 0 {
                    break;
                }
            }
        } else {
            match substrate.apply_committed(record, anchor).await {
                Ok(report) if report.outcome == ApplyOutcome::Applied => {}
                other => {
                    diverged(
                        &mut metrics.crash_recovery_failures,
                        format!(
                            "seed {seed} case {case}: record {} was not applied: {other:?}",
                            record.index.0
                        ),
                    );
                    break;
                }
            }
        }
        next += 1;
        if step.redeliver {
            // Redeliver a record already applied: a duplicate or out of order.
            metrics.redeliveries += 1;
            let earlier = (step.earlier % next as u64) as usize;
            let expected = if earlier + 1 == next {
                ApplyOutcome::Duplicate
            } else {
                ApplyOutcome::OutOfOrder
            };
            match substrate
                .apply_committed(&log[earlier], anchors[earlier])
                .await
            {
                Ok(report) if report.outcome == expected => {}
                other => diverged(
                    &mut metrics.redelivery_failures,
                    format!(
                        "seed {seed} case {case}: redelivery of {} gave {other:?}",
                        earlier + 1
                    ),
                ),
            }
        }
        if step.gap && next + 1 < log.len() {
            metrics.gap_probes += 1;
            match substrate
                .apply_committed(&log[next + 1], anchors[next + 1])
                .await
            {
                Ok(report) if report.outcome == ApplyOutcome::Gap => {}
                other => diverged(
                    &mut metrics.gap_failures,
                    format!(
                        "seed {seed} case {case}: a gap at {} gave {other:?}",
                        next + 2
                    ),
                ),
            }
        }
    }
    metrics.apply_ns += started.elapsed().as_nanos();
    metrics.apply_commits += log.len() as u64;
    if next < log.len() {
        // The failure is counted; every later phase assumes a projection at the
        // head and would only report consequences of it.
        let _ = substrate.drop_all().await;
        return;
    }

    let fence = CommitIndex(log.len() as u64);
    compare_with_oracle(
        &mut substrate,
        raw,
        &instance.prefix,
        &oracle,
        &log,
        anchors[anchors.len() - 1],
        fence,
        seed,
        case,
        metrics,
    )
    .await;

    #[cfg(feature = "turso-oracle")]
    compare_turso(&log, &oracle, seed, case, metrics).await;

    // Forked histories and the legitimate catch-up.
    probe_foreign_histories(&mut substrate, &log, &anchors, rng, seed, case, metrics).await;
    probe_catch_up(
        raw,
        &instance,
        &mut substrate,
        &extended,
        log.len(),
        &extended_oracle,
        rng,
        seed,
        case,
        metrics,
    )
    .await;

    // Rebuild onto an unrelated log first: replaying the same log would rewrite
    // the same keys and hide whatever a rebuild failed to clear, so the rebuilt
    // projection of another history must equal that history's oracle in full.
    let unrelated_length = rng.range(20, 60) as usize;
    let unrelated = generate_log(rng, unrelated_length);
    let unrelated_anchors =
        chain_anchors(&unrelated, LogAnchor::empty()).expect("the unrelated log chains");
    match PtrRuntime::replay(PtrConfig::default(), &unrelated) {
        Ok(unrelated_oracle) => {
            if let Err(error) = substrate.rebuild_projection().await {
                diverged(
                    &mut metrics.rebuild_divergences,
                    format!("seed {seed} case {case}: the rebuild failed: {error:?}"),
                );
            }
            match substrate.replay(&unrelated).await {
                Ok(applied) if applied == unrelated.len() as u64 => {
                    compare_with_oracle(
                        &mut substrate,
                        raw,
                        &instance.prefix,
                        &unrelated_oracle,
                        &unrelated,
                        unrelated_anchors[unrelated_anchors.len() - 1],
                        CommitIndex(unrelated.len() as u64),
                        seed,
                        case,
                        metrics,
                    )
                    .await;
                }
                other => diverged(
                    &mut metrics.rebuild_divergences,
                    format!("seed {seed} case {case}: replay onto a rebuild gave {other:?}"),
                ),
            }
        }
        Err(error) => diverged(
            &mut metrics.invalid_logs,
            format!("seed {seed} case {case}: the runtime refused the unrelated log: {error:?}"),
        ),
    }

    // Rebuild and replay the true ledger; then a record PostgreSQL text cannot
    // hold.
    let extended_anchors = chain_anchors(&extended, LogAnchor::empty()).expect("extension chains");
    let started = Instant::now();
    let rebuilt = substrate.rebuild_projection().await;
    let applied = substrate.replay(&extended).await;
    metrics.rebuild_ns += started.elapsed().as_nanos();
    metrics.rebuild_commits += extended.len() as u64;
    let expected_head = extended_anchors[extended_anchors.len() - 1];
    if rebuilt.is_err() || !matches!(applied, Ok(count) if count == extended.len() as u64) {
        diverged(
            &mut metrics.rebuild_divergences,
            format!(
                "seed {seed} case {case}: the rebuild of the true ledger failed (replay {applied:?})"
            ),
        );
    }
    compare_with_oracle(
        &mut substrate,
        raw,
        &instance.prefix,
        &extended_oracle,
        &extended,
        expected_head,
        CommitIndex(extended.len() as u64),
        seed,
        case,
        metrics,
    )
    .await;
    let watermark = match substrate.watermark().await {
        Ok(watermark) => watermark,
        Err(error) => {
            diverged(
                &mut metrics.read_failures,
                format!("seed {seed} case {case}: the watermark after the rebuild is unreadable: {error:?}"),
            );
            let _ = substrate.drop_all().await;
            return;
        }
    };

    metrics.nul_probes += 1;
    let nul_record = CommittedEvent {
        index: CommitIndex(extended.len() as u64 + 1),
        event: LedgerEvent::VerifierAttested {
            subject: "nul\0subject".into(),
            passed: true,
        },
    };
    let nul_anchor = chain_anchors(
        std::slice::from_ref(&nul_record),
        extended_anchors[extended_anchors.len() - 1],
    )
    .expect("the NUL record chains")[0];
    match substrate.apply_committed(&nul_record, nul_anchor).await {
        Err(PgError::InvalidRecord { .. }) if matches!(substrate.watermark().await, Ok(after) if after == watermark) =>
            {}
        other => diverged(
            &mut metrics.nul_failures,
            format!("seed {seed} case {case}: a NUL record gave {other:?}"),
        ),
    }

    substrate
        .drop_all()
        .await
        .expect("drop experiment instance");
}

/// What happens around one record of the feed.
struct RecordPlan {
    crash: bool,
    kill: bool,
    delay: Duration,
    redeliver: bool,
    earlier: u64,
    gap: bool,
    duel: bool,
}

impl RecordPlan {
    fn draw(rng: &mut Rng) -> Self {
        Self {
            crash: rng.chance(0.06),
            kill: rng.chance(0.5),
            delay: Duration::from_micros(rng.below(4000)),
            redeliver: rng.chance(0.05),
            earlier: rng.next_u64(),
            gap: rng.chance(0.03),
            duel: rng.chance(0.04),
        }
    }
}

/// Apply one record in a task and crash it: drop the task mid-flight (a client
/// crash) or terminate the server session (a server-side kill). Returns a
/// fresh session, as a restarted process would open, and what the projector
/// returned if it finished before the crash reached it.
#[allow(clippy::too_many_arguments)]
async fn crash_during_apply(
    instance: &Instance,
    raw: &Client,
    mut substrate: PgSubstrate,
    record: &CommittedEvent,
    anchor: LogAnchor,
    kill: bool,
    delay: Duration,
    metrics: &mut Metrics,
) -> (
    PgSubstrate,
    Result<Option<Result<ProjectionApply, PgError>>, String>,
) {
    let record = record.clone();
    let task = tokio::spawn(async move { substrate.apply_committed(&record, anchor).await });
    if kill {
        metrics.server_kills += 1;
    } else {
        metrics.client_aborts += 1;
    }
    let returned = pg::crash_task(instance, raw, task, kill, delay).await;
    (instance.connect().await, returned)
}

/// The lifecycle catalog a log implies, folded here from the events
/// themselves rather than through `ptr_pg`'s mapping: the live generation and
/// project of every target, and the tombstone set.
type Catalog = (
    BTreeMap<String, (u64, Option<String>)>,
    BTreeSet<(String, u64)>,
);

fn expected_catalog(log: &[CommittedEvent]) -> Catalog {
    let mut live: BTreeMap<String, (u64, Option<String>)> = BTreeMap::new();
    let mut tombstones = BTreeSet::new();
    let set_live = |live: &mut BTreeMap<String, (u64, Option<String>)>,
                    target: String,
                    generation: u64,
                    project: Option<String>| {
        // A commit that names no project keeps the one the target has.
        let project = project.or_else(|| live.get(&target).and_then(|(_, kept)| kept.clone()));
        live.insert(target, (generation, project));
    };
    for record in log {
        match &record.event {
            LedgerEvent::CapsuleCommitted {
                project,
                capsule,
                generation,
            } => set_live(
                &mut live,
                capsule.0.clone(),
                generation.0,
                Some(project.0.clone()),
            ),
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                set_live(&mut live, capsule.0.clone(), new.0, None)
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                set_live(&mut live, format!("constraint:{key}"), generation.0, None)
            }
            LedgerEvent::ProcedurePromoted { id, generation } => {
                set_live(&mut live, format!("procedure:{id}"), generation.0, None)
            }
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                tombstones.insert((subject.clone(), generation.0));
            }
            LedgerEvent::ProcedureRevoked { id, generation } => {
                tombstones.insert((format!("procedure:{id}"), generation.0));
            }
            _ => {}
        }
    }
    (live, tombstones)
}

/// The lifecycle catalog as stored, read row by row, plus the number of
/// revisions recorded.
async fn stored_catalog(raw: &Client, prefix: &str) -> Result<(Catalog, i64), String> {
    let live_rows = raw
        .query(
            &format!("SELECT target, generation, project FROM {prefix}_projection.live_generation"),
            &[],
        )
        .await
        .map_err(|error| error.to_string())?;
    let tombstone_rows = raw
        .query(
            &format!("SELECT subject, generation FROM {prefix}_projection.tombstone"),
            &[],
        )
        .await
        .map_err(|error| error.to_string())?;
    let revisions: i64 = raw
        .query_one(
            &format!("SELECT count(*) FROM {prefix}_projection.semantic_revision"),
            &[],
        )
        .await
        .map_err(|error| error.to_string())?
        .get(0);
    let live = live_rows
        .iter()
        .map(|row| {
            (
                row.get::<_, String>(0),
                (row.get::<_, i64>(1) as u64, row.get::<_, Option<String>>(2)),
            )
        })
        .collect();
    let tombstones = tombstone_rows
        .iter()
        .map(|row| (row.get::<_, String>(0), row.get::<_, i64>(1) as u64))
        .collect();
    Ok(((live, tombstones), revisions))
}

fn revision_count(log: &[CommittedEvent]) -> i64 {
    log.iter()
        .filter(|record| matches!(record.event, LedgerEvent::SemanticDeltaCommitted { .. }))
        .count() as i64
}

/// Everything a rolled-back record could have left behind: the projection
/// must be exactly the reference of `prefix` — its state, its event count,
/// its lifecycle catalog and its revisions. Returns what differs.
async fn check_prefix(
    substrate: &mut PgSubstrate,
    raw: &Client,
    prefix: &str,
    applied: &[CommittedEvent],
) -> Vec<String> {
    let mut problems = Vec::new();
    let fence = CommitIndex(applied.len() as u64);
    let mut reference = MaterializedState::default();
    for record in applied {
        reference.apply(record);
    }
    match substrate.state_entries(fence).await {
        Ok(entries) if entries == reference.values => {}
        Ok(_) => problems.push("the state differs from the prefix".into()),
        Err(error) => problems.push(format!("the state is unreadable: {error:?}")),
    }
    match substrate
        .events_after(CommitIndex(0), applied.len() as u32 + 10)
        .await
    {
        Ok(events) if events.len() == applied.len() => {}
        Ok(events) => problems.push(format!(
            "{} event rows for {} commits",
            events.len(),
            applied.len()
        )),
        Err(error) => problems.push(format!("the event log is unreadable: {error:?}")),
    }
    match stored_catalog(raw, prefix).await {
        Ok((catalog, revisions)) => {
            if catalog != expected_catalog(applied) {
                problems.push("the lifecycle catalog differs from the prefix".into());
            }
            if revisions != revision_count(applied) {
                problems.push(format!("{revisions} revisions recorded"));
            }
        }
        Err(error) => problems.push(format!("the catalog is unreadable: {error}")),
    }
    problems
}

/// The topic each event kind is published under, written out here so the
/// harness does not check `ptr_pg`'s mapping against itself.
fn expected_topic(event: &LedgerEvent) -> &'static str {
    match event {
        LedgerEvent::SemanticDeltaCommitted { .. } => "semantic.delta_committed",
        LedgerEvent::CapsuleCommitted { .. } => "capsule.committed",
        LedgerEvent::CapsuleSuperseded { .. } => "capsule.superseded",
        LedgerEvent::Revoked { .. } => "lifecycle.revoked",
        LedgerEvent::HardConstraintCommitted { .. } => "constraint.committed",
        LedgerEvent::VerifierAttested { .. } => "verifier.attested",
        LedgerEvent::ProcedurePromoted { .. } => "procedure.promoted",
        LedgerEvent::ProcedureRevoked { .. } => "procedure.revoked",
        LedgerEvent::SnapshotCommitted { .. } => "snapshot.committed",
        LedgerEvent::EffectAttempted { .. } => "effect.attempted",
        LedgerEvent::EffectSettled { .. } => "effect.settled",
        LedgerEvent::EffectReconciled { .. } => "effect.reconciled",
    }
}

/// The subject a consumer partitions by, as doc 35 defines it: the lifecycle
/// target, the attested subject, the revision, the snapshot, or the effect
/// attempt's commit index.
fn expected_subject(record: &CommittedEvent) -> String {
    match &record.event {
        LedgerEvent::CapsuleCommitted { capsule, .. }
        | LedgerEvent::CapsuleSuperseded { capsule, .. } => capsule.0.clone(),
        LedgerEvent::Revoked { subject, .. } | LedgerEvent::VerifierAttested { subject, .. } => {
            subject.clone()
        }
        LedgerEvent::HardConstraintCommitted { key, .. } => format!("constraint:{key}"),
        LedgerEvent::ProcedurePromoted { id, .. } | LedgerEvent::ProcedureRevoked { id, .. } => {
            format!("procedure:{id}")
        }
        LedgerEvent::SemanticDeltaCommitted { revision, .. } => format!("revision:{}", revision.0),
        LedgerEvent::SnapshotCommitted { revision, .. } => format!("snapshot:{revision}"),
        LedgerEvent::EffectAttempted { .. } => format!("effect:{}", record.index.0),
        LedgerEvent::EffectSettled { attempt, .. }
        | LedgerEvent::EffectReconciled { attempt, .. } => format!("effect:{}", attempt.0),
    }
}

/// An event row's payload must carry the commit index and exactly the entries
/// the reference projects for the record.
fn payload_matches(row: &ptr_pg::ProjectionEventRow, record: &CommittedEvent) -> bool {
    let expected: BTreeMap<String, String> = projection_entries(record).into_iter().collect();
    let Some(entries) = row.payload["entries"].as_object() else {
        return false;
    };
    row.payload["commit_index"].as_u64() == Some(record.index.0)
        && entries.len() == expected.len()
        && entries
            .iter()
            .all(|(key, value)| value.as_str() == expected.get(key).map(String::as_str))
}

#[allow(clippy::too_many_arguments)]
async fn compare_with_oracle(
    substrate: &mut PgSubstrate,
    raw: &Client,
    prefix: &str,
    oracle: &PtrRuntime,
    log: &[CommittedEvent],
    head: LogAnchor,
    fence: CommitIndex,
    seed: u64,
    case: usize,
    metrics: &mut Metrics,
) {
    // State: every entry, and no extra or missing key.
    let expected = &oracle.materialized_state().values;
    let actual = match substrate.state_entries(fence).await {
        Ok(actual) => actual,
        Err(error) => {
            diverged(
                &mut metrics.read_failures,
                format!("seed {seed} case {case}: the state at the head is unreadable: {error:?}"),
            );
            return;
        }
    };
    for (key, value) in expected {
        match actual.get(key) {
            Some(found) if found == value => {}
            Some(found) => diverged(
                &mut metrics.state_divergences,
                format!("seed {seed} case {case}: {key:?} is {found:?}, expected {value:?}"),
            ),
            None => diverged(
                &mut metrics.extra_or_missing_keys,
                format!("seed {seed} case {case}: {key:?} is missing"),
            ),
        }
    }
    for key in actual.keys().filter(|key| !expected.contains_key(*key)) {
        diverged(
            &mut metrics.extra_or_missing_keys,
            format!("seed {seed} case {case}: {key:?} is extra"),
        );
    }

    // Lifecycle: every target the log names, every generation up to one past
    // the largest, plus a target it never names.
    let mut targets: BTreeSet<String> = BTreeSet::new();
    let mut max_generation = 0u64;
    for record in log {
        match &record.event {
            LedgerEvent::CapsuleCommitted {
                capsule,
                generation,
                ..
            } => {
                targets.insert(capsule.to_string());
                max_generation = max_generation.max(generation.0);
            }
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                targets.insert(capsule.to_string());
                max_generation = max_generation.max(new.0);
            }
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                targets.insert(subject.clone());
                max_generation = max_generation.max(generation.0);
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                targets.insert(format!("constraint:{key}"));
                max_generation = max_generation.max(generation.0);
            }
            LedgerEvent::ProcedurePromoted { id, generation }
            | LedgerEvent::ProcedureRevoked { id, generation } => {
                targets.insert(format!("procedure:{id}"));
                max_generation = max_generation.max(generation.0);
            }
            _ => {}
        }
    }
    targets.insert("never-named".into());
    for target in &targets {
        let live = substrate.live_generation(target, fence).await;
        if !matches!(&live, Ok(live) if *live == oracle.live_generation(target)) {
            diverged(
                &mut metrics.lifecycle_divergences,
                format!(
                    "seed {seed} case {case}: live generation of {target:?} is {live:?}, the runtime says {:?}",
                    oracle.live_generation(target)
                ),
            );
        }
        for generation in 0..=max_generation + 1 {
            metrics.lifecycle_checks += 1;
            let admissible = substrate
                .is_admissible(target, Generation(generation), fence)
                .await;
            let expected =
                oracle.generation_validity(target, Generation(generation)) == Some(Validity::Live);
            if !matches!(&admissible, Ok(admissible) if *admissible == expected) {
                diverged(
                    &mut metrics.lifecycle_divergences,
                    format!(
                        "seed {seed} case {case}: ({target:?}, {generation}) admissible {admissible:?}, the runtime says {expected}"
                    ),
                );
            }
        }
    }

    // The whole lifecycle catalog as stored — live generations with their
    // projects, tombstones, the number of revisions — against the one the log
    // implies: this also shows rows left over for targets the log never names.
    match stored_catalog(raw, prefix).await {
        Ok((catalog, revisions)) => {
            let expected = expected_catalog(log);
            for (target, (generation, _)) in &expected.0 {
                if oracle.live_generation(target) != Some(Generation(*generation)) {
                    diverged(
                        &mut metrics.invalid_logs,
                        format!("seed {seed} case {case}: the harness's catalog disagrees with the runtime on {target:?}"),
                    );
                }
            }
            if catalog.0 != expected.0 {
                diverged(
                    &mut metrics.lifecycle_divergences,
                    format!(
                        "seed {seed} case {case}: {} live-generation rows, {} expected, or a generation or project differs",
                        catalog.0.len(),
                        expected.0.len()
                    ),
                );
            }
            if catalog.1 != expected.1 {
                diverged(
                    &mut metrics.lifecycle_divergences,
                    format!(
                        "seed {seed} case {case}: {} tombstones, {} expected",
                        catalog.1.len(),
                        expected.1.len()
                    ),
                );
            }
            if revisions != revision_count(log) {
                diverged(
                    &mut metrics.revision_divergences,
                    format!(
                        "seed {seed} case {case}: {revisions} revisions recorded, {} expected",
                        revision_count(log)
                    ),
                );
            }
        }
        Err(error) => diverged(
            &mut metrics.read_failures,
            format!("seed {seed} case {case}: the lifecycle catalog is unreadable: {error}"),
        ),
    }

    // Revisions: which commit published each one.
    for record in log {
        if let LedgerEvent::SemanticDeltaCommitted { revision, .. } = &record.event {
            let found = substrate.revision_commit(*revision).await;
            if !matches!(&found, Ok(found) if *found == Some(record.index)) {
                diverged(
                    &mut metrics.revision_divergences,
                    format!(
                        "seed {seed} case {case}: revision {} maps to {found:?}, expected {}",
                        revision.0, record.index.0
                    ),
                );
            }
        }
    }

    // The event log: one row per commit, in commit order, with the topic and
    // subject written out above and the entries the reference projects.
    let events = match substrate
        .events_after(CommitIndex(0), log.len() as u32 + 10)
        .await
    {
        Ok(events) => events,
        Err(error) => {
            diverged(
                &mut metrics.read_failures,
                format!("seed {seed} case {case}: the event log is unreadable: {error:?}"),
            );
            Vec::new()
        }
    };
    if events.len() != log.len() {
        diverged(
            &mut metrics.event_log_divergences,
            format!(
                "seed {seed} case {case}: {} event rows for {} commits",
                events.len(),
                log.len()
            ),
        );
    }
    for (row, record) in events.iter().zip(log) {
        if row.commit_index != record.index
            || row.topic != expected_topic(&record.event)
            || row.subject != expected_subject(record)
            || !payload_matches(row, record)
        {
            diverged(
                &mut metrics.event_log_divergences,
                format!(
                    "seed {seed} case {case}: event row {} differs from the commit",
                    record.index.0
                ),
            );
        }
    }

    match substrate.watermark().await {
        Ok(watermark) if watermark == head => {}
        other => diverged(
            &mut metrics.watermark_divergences,
            format!(
                "seed {seed} case {case}: watermark {other:?} differs from the ledger head {}",
                head.index.0
            ),
        ),
    }
}

/// Histories that share a prefix with the projection and then diverge. None of
/// them may be accepted: a ledger that is behind, a ledger at the same index
/// with another record, a ledger ahead whose next record chains from another
/// history, a different record redelivered at an applied index, and a record
/// handed over with another index's anchor.
#[allow(clippy::too_many_arguments)]
async fn probe_foreign_histories(
    substrate: &mut PgSubstrate,
    log: &[CommittedEvent],
    anchors: &[LogAnchor],
    rng: &mut Rng,
    seed: u64,
    case: usize,
    metrics: &mut Metrics,
) {
    let head = log.len();
    let fork_at = rng.range(1, head as u64) as usize; // 1-based first divergent index
    let fork = |length: usize, rng: &mut Rng| -> Vec<CommittedEvent> {
        let mut forked: Vec<CommittedEvent> = log[..fork_at - 1].to_vec();
        for index in fork_at..=length {
            forked.push(CommittedEvent {
                index: CommitIndex(index as u64),
                event: LedgerEvent::VerifierAttested {
                    subject: format!("fork-{index}-{}", rng.below(1000)),
                    passed: rng.chance(0.5),
                },
            });
        }
        forked
    };
    let mut refused = |outcome: Result<bool, String>, what: &str| {
        metrics.foreign_probes += 1;
        match outcome {
            Ok(true) => {}
            Ok(false) => diverged(
                &mut metrics.foreign_histories_accepted,
                format!("seed {seed} case {case}: {what} was accepted"),
            ),
            Err(error) => diverged(
                &mut metrics.foreign_histories_accepted,
                format!("seed {seed} case {case}: {what} gave {error}"),
            ),
        }
    };

    // A ledger behind the projection (a discarded tail).
    if fork_at < head {
        let length = rng.range(fork_at as u64, head as u64 - 1) as usize;
        let behind = fork(length, rng);
        let behind_anchors = chain_anchors(&behind, LogAnchor::empty()).expect("fork chains");
        let result = substrate
            .check_against_ledger(behind_anchors[length - 1])
            .await;
        refused(
            Ok(matches!(result, Err(PgError::ProjectionAhead { .. }))),
            "a ledger behind the projection",
        );
    }

    // A ledger at the same index with another history.
    let beside = fork(head, rng);
    let beside_anchors = chain_anchors(&beside, LogAnchor::empty()).expect("fork chains");
    let result = substrate
        .check_against_ledger(beside_anchors[head - 1])
        .await;
    refused(
        Ok(matches!(result, Err(PgError::ForeignHistory { .. }))),
        "a ledger beside the projection",
    );

    // A ledger ahead whose next record chains from another history.
    let ahead = fork(head + rng.range(1, 3) as usize, rng);
    let ahead_anchors = chain_anchors(&ahead, LogAnchor::empty()).expect("fork chains");
    let result = substrate
        .apply_committed(&ahead[head], ahead_anchors[head])
        .await;
    refused(
        Ok(matches!(result, Err(PgError::ForeignHistory { .. }))),
        "the next record of a diverged ledger",
    );

    // A different record redelivered at an applied index.
    let redelivered = rng.range(fork_at as u64, head as u64) as usize - 1;
    let result = substrate
        .apply_committed(&beside[redelivered], beside_anchors[redelivered])
        .await;
    refused(
        Ok(matches!(result, Err(PgError::ForeignHistory { .. }))),
        "a different record at an applied index",
    );

    // The forked record handed over with the true anchor of its index, and
    // the true record with the forked anchor: the anchor alone must not make
    // a different record a duplicate.
    let mixed = rng.range(fork_at as u64, head as u64) as usize - 1;
    let result = substrate
        .apply_committed(&beside[mixed], anchors[mixed])
        .await;
    refused(
        Ok(matches!(
            result,
            Err(PgError::ForeignHistory { .. } | PgError::InvalidRecord { .. })
        )),
        "a different record with the true anchor at an applied index",
    );
    let result = substrate
        .apply_committed(&log[mixed], beside_anchors[mixed])
        .await;
    refused(
        Ok(matches!(
            result,
            Err(PgError::ForeignHistory { .. } | PgError::InvalidRecord { .. })
        )),
        "the true record with a forked anchor at an applied index",
    );

    // A record handed over with another index's anchor.
    let result = substrate.apply_committed(&log[head - 1], anchors[0]).await;
    refused(
        Ok(head == 1 || matches!(result, Err(PgError::InvalidRecord { .. }))),
        "a record with another index's anchor",
    );

    // None of the refusals may have moved the projection.
    let watermark = substrate.watermark().await;
    refused(
        Ok(matches!(watermark, Ok(watermark) if watermark == anchors[head - 1])),
        "a refusal that moved the watermark",
    );
}

/// The negative control: the ledger grows by records of every kind, and both
/// the projection and an instance restored from an older backup catch up with
/// it. Nothing here may be refused, and both must then equal the runtime's
/// replay of the grown ledger in full.
#[allow(clippy::too_many_arguments)]
async fn probe_catch_up(
    raw: &Client,
    instance: &Instance,
    substrate: &mut PgSubstrate,
    extended: &[CommittedEvent],
    applied: usize,
    oracle: &PtrRuntime,
    rng: &mut Rng,
    seed: u64,
    case: usize,
    metrics: &mut Metrics,
) {
    let anchors = chain_anchors(extended, LogAnchor::empty()).expect("extension chains");
    let head = anchors[anchors.len() - 1];
    let fence = CommitIndex(extended.len() as u64);

    // The main instance is simply behind the grown ledger.
    match substrate.check_against_ledger(head).await {
        Ok(behind) if behind == (extended.len() - applied) as u64 => {}
        other => diverged(
            &mut metrics.false_refusals,
            format!("seed {seed} case {case}: a legitimate catch-up check gave {other:?}"),
        ),
    }
    for (record, anchor) in extended[applied..].iter().zip(&anchors[applied..]) {
        match substrate.apply_committed(record, *anchor).await {
            Ok(report) if report.outcome == ApplyOutcome::Applied => {}
            other => diverged(
                &mut metrics.false_refusals,
                format!(
                    "seed {seed} case {case}: legitimate record {} gave {other:?}",
                    record.index.0
                ),
            ),
        }
    }
    compare_with_oracle(
        substrate,
        raw,
        &instance.prefix,
        oracle,
        extended,
        head,
        fence,
        seed,
        case,
        metrics,
    )
    .await;

    // A second instance restored from a backup taken at a random earlier
    // index catches up from there.
    metrics.backup_catchups += 1;
    let backup = Instance::new("l004b", seed, case);
    let mut restored = backup.create(raw).await;
    let taken_at = rng.range(0, applied as u64) as usize;
    if let Err(error) = restored.replay(&extended[..taken_at]).await {
        diverged(
            &mut metrics.false_refusals,
            format!("seed {seed} case {case}: the backup prefix was refused: {error:?}"),
        );
    }
    match restored.check_against_ledger(head).await {
        Ok(behind) if behind == (extended.len() - taken_at) as u64 => {}
        other => diverged(
            &mut metrics.false_refusals,
            format!("seed {seed} case {case}: a restored backup's check gave {other:?}"),
        ),
    }
    let mut caught_up = true;
    for (record, anchor) in extended[taken_at..].iter().zip(&anchors[taken_at..]) {
        if !matches!(
            restored.apply_committed(record, *anchor).await,
            Ok(report) if report.outcome == ApplyOutcome::Applied
        ) {
            diverged(
                &mut metrics.false_refusals,
                format!(
                    "seed {seed} case {case}: the restored backup refused {}",
                    record.index.0
                ),
            );
            caught_up = false;
            break;
        }
    }
    if caught_up {
        let before = metrics.hard_failures();
        compare_with_oracle(
            &mut restored,
            raw,
            &backup.prefix,
            oracle,
            extended,
            head,
            fence,
            seed,
            case,
            metrics,
        )
        .await;
        if metrics.hard_failures() > before {
            diverged(
                &mut metrics.catchup_divergences,
                format!("seed {seed} case {case}: the caught-up backup differs from the reference"),
            );
        }
    }
    restored.drop_all().await.expect("drop backup instance");
}

#[cfg(feature = "turso-oracle")]
async fn compare_turso(
    log: &[CommittedEvent],
    oracle: &PtrRuntime,
    seed: u64,
    case: usize,
    metrics: &mut Metrics,
) {
    let path = std::env::temp_dir().join(format!(
        "ptr-l004-turso-{}-{seed}-{case}.db",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let mut turso = ptr_state::TursoMaterializedState::open(path.to_str().expect("utf-8 path"))
        .await
        .expect("open Turso projection");
    for record in log {
        match turso.try_apply(record).await {
            Ok(ApplyOutcome::Applied) => {}
            other => {
                diverged(
                    &mut metrics.turso_divergences,
                    format!(
                        "seed {seed} case {case}: Turso refused {}: {other:?}",
                        record.index.0
                    ),
                );
                return;
            }
        }
    }
    for (key, value) in &oracle.materialized_state().values {
        let found = turso.get(key).await.expect("Turso read");
        if found.as_ref() != Some(value) {
            diverged(
                &mut metrics.turso_divergences,
                format!("seed {seed} case {case}: Turso has {found:?} for {key:?}"),
            );
        }
    }
    if turso.last_applied() != log.len() as u64 {
        diverged(
            &mut metrics.turso_divergences,
            format!(
                "seed {seed} case {case}: Turso stopped at {}",
                turso.last_applied()
            ),
        );
    }
    drop(turso);
    let _ = std::fs::remove_file(&path);
}

/// A random log the runtime accepts: generations respect activation rules,
/// supersessions name the current generation, semantic deltas are real encoded
/// deltas on the revision chain, and effect attempts are settled or reconciled
/// exactly once with keys never reused while in flight.
fn generate_log(rng: &mut Rng, length: usize) -> Vec<CommittedEvent> {
    let mut generator = Generator {
        events: Vec::with_capacity(length),
        semdb: SemanticHost::default(),
        capsules: BTreeMap::new(),
        live: BTreeMap::new(),
        tombstones: BTreeSet::new(),
        unsettled: Vec::new(),
        next_effect_key: 0,
    };
    while generator.events.len() < length {
        generator.step(rng);
    }
    generator.events
}

struct Generator {
    events: Vec<CommittedEvent>,
    semdb: SemanticHost,
    /// Capsule id to its project.
    capsules: BTreeMap<String, String>,
    /// Lifecycle target to its live generation, keyed as the runtime keys it.
    live: BTreeMap<String, u64>,
    tombstones: BTreeSet<(String, u64)>,
    /// Unsettled effect attempts: commit index and at-most-once key.
    unsettled: Vec<(u64, Option<String>)>,
    next_effect_key: u64,
}

impl Generator {
    fn push(&mut self, event: LedgerEvent) {
        let index = CommitIndex(self.events.len() as u64 + 1);
        self.events.push(CommittedEvent { index, event });
    }

    fn name(rng: &mut Rng) -> String {
        format!("{}-{}", rng.pick(&NAMES), rng.below(40))
    }

    /// A generation the runtime would accept for `target`: at least the live
    /// one and not tombstoned.
    fn activation(&self, rng: &mut Rng, target: &str) -> Option<u64> {
        for _ in 0..8 {
            let generation = match self.live.get(target) {
                Some(current) => current + rng.below(3),
                None => rng.below(4),
            };
            if !self.tombstones.contains(&(target.to_owned(), generation)) {
                return Some(generation);
            }
        }
        None
    }

    fn step(&mut self, rng: &mut Rng) {
        match rng.below(100) {
            0..=17 => self.commit_capsule(rng),
            18..=29 => self.supersede(rng),
            30..=41 => self.revoke(rng),
            42..=49 => {
                let key = rng.pick(&CONSTRAINTS).to_string();
                let target = format!("constraint:{key}");
                if let Some(generation) = self.activation(rng, &target) {
                    self.live.insert(target, generation);
                    self.push(LedgerEvent::HardConstraintCommitted {
                        key,
                        generation: Generation(generation),
                    });
                }
            }
            50..=55 => {
                let id = rng.pick(&PROCEDURES).to_string();
                let target = format!("procedure:{id}");
                if let Some(generation) = self.activation(rng, &target) {
                    self.live.insert(target, generation);
                    self.push(LedgerEvent::ProcedurePromoted {
                        id,
                        generation: Generation(generation),
                    });
                }
            }
            56..=60 => {
                let id = rng.pick(&PROCEDURES).to_string();
                let target = format!("procedure:{id}");
                let generation = match self.live.get(&target) {
                    Some(current) if rng.chance(0.6) => *current,
                    _ => rng.below(4),
                };
                self.tombstones.insert((target, generation));
                self.push(LedgerEvent::ProcedureRevoked {
                    id,
                    generation: Generation(generation),
                });
            }
            61..=66 => {
                let subject = Self::name(rng);
                self.push(LedgerEvent::VerifierAttested {
                    subject,
                    passed: rng.chance(0.5),
                });
            }
            67..=76 => self.semantic_delta(rng),
            77..=80 => {
                let covers = CommitIndex(rng.range(0, self.events.len() as u64));
                self.push(LedgerEvent::SnapshotCommitted {
                    revision: self.semdb.revision().0,
                    covers,
                });
            }
            81..=88 => self.attempt(rng),
            89..=94 => self.settle(rng),
            _ => self.reconcile(rng),
        }
    }

    fn commit_capsule(&mut self, rng: &mut Rng) {
        let existing: Vec<String> = self.capsules.keys().cloned().collect();
        let capsule = if !existing.is_empty() && rng.chance(0.5) {
            rng.pick(&existing).clone()
        } else {
            Self::name(rng)
        };
        let project = self
            .capsules
            .get(&capsule)
            .cloned()
            .unwrap_or_else(|| rng.pick(&PROJECTS).to_string());
        let Some(generation) = self.activation(rng, &capsule) else {
            return;
        };
        self.capsules.insert(capsule.clone(), project.clone());
        self.live.insert(capsule.clone(), generation);
        self.push(LedgerEvent::CapsuleCommitted {
            project: ProjectId(project),
            capsule: CapsuleId(capsule),
            generation: Generation(generation),
        });
    }

    fn supersede(&mut self, rng: &mut Rng) {
        let live: Vec<(String, u64)> = self
            .capsules
            .keys()
            .filter_map(|capsule| self.live.get(capsule).map(|g| (capsule.clone(), *g)))
            .collect();
        if live.is_empty() {
            return self.commit_capsule(rng);
        }
        let (capsule, old) = rng.pick(&live).clone();
        for _ in 0..4 {
            let new = old + rng.range(1, 3);
            if !self.tombstones.contains(&(capsule.clone(), new)) {
                self.live.insert(capsule.clone(), new);
                self.push(LedgerEvent::CapsuleSuperseded {
                    capsule: CapsuleId(capsule),
                    old: Generation(old),
                    new: Generation(new),
                });
                return;
            }
        }
    }

    fn revoke(&mut self, rng: &mut Rng) {
        let known: Vec<String> = self.live.keys().cloned().collect();
        let subject = if !known.is_empty() && rng.chance(0.75) {
            rng.pick(&known).clone()
        } else {
            Self::name(rng)
        };
        let generation = match self.live.get(&subject) {
            Some(current) => match rng.below(10) {
                0..=4 => *current,
                5..=6 => current.saturating_sub(rng.range(1, 2)),
                _ => current + rng.range(1, 2),
            },
            None => rng.below(4),
        };
        self.tombstones.insert((subject.clone(), generation));
        self.push(LedgerEvent::Revoked {
            subject,
            generation: Generation(generation),
        });
    }

    fn semantic_delta(&mut self, rng: &mut Rng) {
        let mut delta = SemanticDelta::default();
        let index = self.events.len() + 1;
        for _ in 0..rng.range(1, 3) {
            delta.upserts.insert(
                format!("k{}", rng.below(20)),
                SemanticValue::Text(format!("v{index}-{}", rng.below(1000))),
            );
        }
        let encoded = delta.encode().expect("a generated delta encodes");
        let base = self.semdb.revision();
        let (revision, _) = self
            .semdb
            .apply_delta(delta)
            .expect("a generated delta applies");
        self.push(LedgerEvent::SemanticDeltaCommitted {
            base_revision: base,
            revision,
            encoded_delta: encoded,
        });
    }

    fn attempt(&mut self, rng: &mut Rng) {
        let index = self.events.len() as u64 + 1;
        let key = rng.chance(0.5).then(|| {
            self.next_effect_key += 1;
            format!("effect-key-{}", self.next_effect_key)
        });
        self.unsettled.push((index, key.clone()));
        self.push(LedgerEvent::EffectAttempted {
            key,
            project: ProjectId(rng.pick(&PROJECTS).to_string()),
            principal: Self::name(rng),
            target: Self::name(rng),
            operation: rng.pick(&["write", "delete row", "send ✉"]).to_string(),
            capability: CapabilityId(format!("cap.{}", rng.below(5))),
            effect: *rng.pick(&[
                Effect::Pure,
                Effect::Read,
                Effect::Mutation,
                Effect::External,
                Effect::Irreversible,
            ]),
            generation: Generation(rng.below(5)),
            revision: self.semdb.revision(),
            verification: *rng.pick(&[
                VerificationLevel::FullSemantic,
                VerificationLevel::Deterministic,
            ]),
            action_digest: rng.digest(),
        });
    }

    fn settle(&mut self, rng: &mut Rng) {
        if self.unsettled.is_empty() {
            return self.attempt(rng);
        }
        let (attempt, _) = self.unsettled.remove(rng.index(self.unsettled.len()));
        let (response, response_digest) = if rng.chance(0.6) {
            let length = rng.index(64);
            let bytes = rng.bytes(length);
            let digest = sha256(&bytes);
            (Some(bytes), digest)
        } else {
            (None, rng.digest())
        };
        self.push(LedgerEvent::EffectSettled {
            attempt: CommitIndex(attempt),
            response,
            response_digest,
        });
    }

    fn reconcile(&mut self, rng: &mut Rng) {
        if self.unsettled.is_empty() {
            return self.attempt(rng);
        }
        let (attempt, _) = self.unsettled.remove(rng.index(self.unsettled.len()));
        self.push(LedgerEvent::EffectReconciled {
            attempt: CommitIndex(attempt),
            applied: rng.chance(0.5),
            evidence: format!("operator note {} — checked", rng.below(100)),
        });
    }
}
