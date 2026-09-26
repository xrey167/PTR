//! L003 — can a revoked input ever influence a fast-memory readout after
//! revocation, replay, restore or crash?
//!
//! Each case runs one to three fast memories (random shapes, checkpoint
//! intervals and decay forms) whose journals and checkpoints live in
//! PostgreSQL. Writes come from lifecycle targets that a ledger commits,
//! supersedes and revokes through the real projector, which deletes the
//! journal writes and checkpoints a change makes inadmissible; the process
//! holding the memories refolds them in memory when it sees the change.
//!
//! The oracle is independent of the database and of the revocation path: the
//! runtime's replay of the ledger decides which source generations are
//! admissible, and the harness's own record of every acknowledged write, as
//! composed, folded from scratch without the inadmissible ones, is the
//! never-saw-it fold. That fold uses `ptr-fastmem`'s own arithmetic (the
//! delta rule is checked against its closed form by the crate's unit tests);
//! what it is independent of is the refold, its checkpoints, the journal store
//! and the projector's cascade, which are what this experiment tests. After
//! every record the refolded memory, the journal PostgreSQL kept, every
//! stored checkpoint and the newest one handed out are compared with that
//! fold bit for bit, and a read in the window between the change's commit and
//! the refold must be denied. Processes crash mid-append, mid-revocation and
//! between operations, and an operation reported done must be found durable;
//! appends and checkpoints race revocations on separate sessions.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use ptr_config::PtrConfig;
use ptr_fastmem::{
    binding_digest_of, decode_state, encode_state, Decay, FastMemory, FastMemoryConfig,
    FastWeightState, Query, SourceRef, WriteRequest, WriteSeq,
};
use ptr_ledger::integrity::{chain_anchors, sha256, LogAnchor};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_pg::{lifecycle_change, FastMemoryRecord, LifecycleChange, PgError, PgSubstrate};
use ptr_runtime::PtrRuntime;
use ptr_state::ApplyOutcome;
use ptr_types::{CapsuleId, CommitIndex, Generation, PrincipalId, ProjectId, Validity};
use tokio_postgres::Client;

use super::pg::{self, Instance};
use super::rng::Rng;

const CAPSULES: [&str; 8] = [
    "alpha",
    "Ωmega",
    "quote'd",
    "sp ace",
    "emoji🙂",
    "colon:inside",
    "naïve",
    "back\\slash",
];
const PROJECTS: [&str; 3] = ["atlas", "borealis", "ç-project"];
const CONSTRAINTS: [&str; 3] = ["budget", "latency slo", "naïve"];
const PROCEDURES: [&str; 3] = ["deploy", "roll back", "ünïcode"];

#[derive(Default)]
struct Metrics {
    ledger_records: u64,
    lifecycle_records: u64,
    memories: u64,
    writes_acknowledged: u64,
    /// Writes a crash interrupted before they were durable; a restored
    /// memory must not contain them.
    writes_lost_in_crashes: u64,
    /// Journal writes the lifecycle changes made inadmissible, by the model.
    writes_removed: u64,
    revocations_with_removal: u64,
    /// Writes the in-memory refolds replayed, and what folding every remaining
    /// write from scratch would have replayed instead.
    writes_replayed: u64,
    full_refold_writes: u64,
    max_replayed: u64,
    /// The longest journal a check compared.
    max_journal_len: u64,
    bit_identity_checks: u64,
    readout_checks: u64,
    window_reads: u64,
    window_denials: u64,
    inadmissible_probes: u64,
    checkpoints_put: u64,
    checkpoints_verified: u64,
    stale_checkpoints_refused: u64,
    /// Checkpoints of an unrefolded state planted directly in storage, as a
    /// restored backup or a faulty writer could leave them; the newest
    /// checkpoint handed out must never be one.
    planted_stale_checkpoints: u64,
    process_crashes: u64,
    append_crashes: u64,
    append_crashes_committed: u64,
    revocation_crashes: u64,
    revocation_crashes_committed: u64,
    revocation_crashes_rolled_back: u64,
    server_kills: u64,
    client_aborts: u64,
    restores: u64,
    append_races: u64,
    append_races_append_first: u64,
    append_races_revocation_first: u64,
    checkpoint_races: u64,
    checkpoint_races_stored: u64,
    checkpoint_races_refused: u64,
    journal_checks: u64,
    // Hard failures.
    invalid_logs: u64,
    bit_identity_failures: u64,
    readout_failures: u64,
    resurrected_reads: u64,
    false_denials: u64,
    admission_divergences: u64,
    journal_mismatches: u64,
    removed_count_mismatches: u64,
    inadmissible_appends_accepted: u64,
    append_false_refusals: u64,
    checkpoint_false_refusals: u64,
    stale_checkpoints_accepted: u64,
    checkpoint_violations: u64,
    stale_checkpoint_rows: u64,
    /// Checkpoints stored and still folding the journal that disappeared.
    checkpoints_lost: u64,
    atomicity_failures: u64,
    crash_recovery_failures: u64,
    race_violations: u64,
    read_failures: u64,
}

impl Metrics {
    fn hard_failures(&self) -> u64 {
        self.invalid_logs
            + self.bit_identity_failures
            + self.readout_failures
            + self.resurrected_reads
            + self.false_denials
            + self.admission_divergences
            + self.journal_mismatches
            + self.removed_count_mismatches
            + self.inadmissible_appends_accepted
            + self.append_false_refusals
            + self.checkpoint_false_refusals
            + self.stale_checkpoints_accepted
            + self.checkpoint_violations
            + self.stale_checkpoint_rows
            + self.checkpoints_lost
            + self.atomicity_failures
            + self.crash_recovery_failures
            + self.race_violations
            + self.read_failures
    }

    fn fields(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("ledger_records", self.ledger_records),
            ("lifecycle_records", self.lifecycle_records),
            ("memories", self.memories),
            ("writes_acknowledged", self.writes_acknowledged),
            ("writes_lost_in_crashes", self.writes_lost_in_crashes),
            ("writes_removed", self.writes_removed),
            ("revocations_with_removal", self.revocations_with_removal),
            ("writes_replayed", self.writes_replayed),
            ("full_refold_writes", self.full_refold_writes),
            ("max_replayed", self.max_replayed),
            ("max_journal_len", self.max_journal_len),
            ("bit_identity_checks", self.bit_identity_checks),
            ("readout_checks", self.readout_checks),
            ("window_reads", self.window_reads),
            ("window_denials", self.window_denials),
            ("inadmissible_probes", self.inadmissible_probes),
            ("checkpoints_put", self.checkpoints_put),
            ("checkpoints_verified", self.checkpoints_verified),
            ("stale_checkpoints_refused", self.stale_checkpoints_refused),
            ("planted_stale_checkpoints", self.planted_stale_checkpoints),
            ("process_crashes", self.process_crashes),
            ("append_crashes", self.append_crashes),
            ("append_crashes_committed", self.append_crashes_committed),
            ("revocation_crashes", self.revocation_crashes),
            (
                "revocation_crashes_committed",
                self.revocation_crashes_committed,
            ),
            (
                "revocation_crashes_rolled_back",
                self.revocation_crashes_rolled_back,
            ),
            ("server_kills", self.server_kills),
            ("client_aborts", self.client_aborts),
            ("restores", self.restores),
            ("append_races", self.append_races),
            ("append_races_append_first", self.append_races_append_first),
            (
                "append_races_revocation_first",
                self.append_races_revocation_first,
            ),
            ("checkpoint_races", self.checkpoint_races),
            ("checkpoint_races_stored", self.checkpoint_races_stored),
            ("checkpoint_races_refused", self.checkpoint_races_refused),
            ("journal_checks", self.journal_checks),
            ("invalid_logs", self.invalid_logs),
            ("bit_identity_failures", self.bit_identity_failures),
            ("readout_failures", self.readout_failures),
            ("resurrected_reads", self.resurrected_reads),
            ("false_denials", self.false_denials),
            ("admission_divergences", self.admission_divergences),
            ("journal_mismatches", self.journal_mismatches),
            ("removed_count_mismatches", self.removed_count_mismatches),
            (
                "inadmissible_appends_accepted",
                self.inadmissible_appends_accepted,
            ),
            ("append_false_refusals", self.append_false_refusals),
            ("checkpoint_false_refusals", self.checkpoint_false_refusals),
            (
                "stale_checkpoints_accepted",
                self.stale_checkpoints_accepted,
            ),
            ("checkpoint_violations", self.checkpoint_violations),
            ("stale_checkpoint_rows", self.stale_checkpoint_rows),
            ("checkpoints_lost", self.checkpoints_lost),
            ("atomicity_failures", self.atomicity_failures),
            ("crash_recovery_failures", self.crash_recovery_failures),
            ("race_violations", self.race_violations),
            ("read_failures", self.read_failures),
        ]
    }
}

/// Report a divergence on stderr (captured in the run record) and count it.
fn diverged(counter: &mut u64, message: String) {
    if *counter < 20 {
        eprintln!("L003 divergence: {message}");
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
        (metrics, server, started.elapsed())
    });
    let hard = metrics.hard_failures();
    let mut line = format!(
        "{{\"benchmark\":\"fastmem-revocation\",\"iterations\":{iterations},\"seed\":{seed},\
         \"server\":{},\"elapsed_ns\":{}",
        pg::json_string(&server),
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

async fn run_case(raw: &Client, seed: u64, case: usize, rng: &mut Rng, metrics: &mut Metrics) {
    let instance = Instance::new("l003", seed, case);
    let writer = instance.create(raw).await;
    let projector = instance.connect().await;
    let mut run = Case {
        raw,
        instance,
        sessions: Sessions {
            writer: Some(writer),
            projector: Some(projector),
        },
        ledger: Ledger::new(),
        memories: Vec::new(),
        metrics,
        label: format!("seed {seed} case {case}"),
        broken: false,
    };
    for index in 0..rng.range(1, 3) as usize {
        run.register(index, rng).await;
    }
    for _ in 0..rng.range(3, 6) {
        run.activate(rng).await;
    }
    // Writes outnumber revocations several times over, so memories grow to
    // tens of writes and a refold has checkpoints to restart from.
    for _ in 0..rng.range(150, 300) {
        if run.broken {
            break;
        }
        match rng.below(100) {
            0..=43 => run.write(rng).await,
            44 => {
                // A burst grows the journals to a hundred writes and more.
                for _ in 0..rng.range(40, 150) {
                    run.write(rng).await;
                }
            }
            45..=47 => run.inadmissible_write(rng).await,
            48..=57 => run.activate(rng).await,
            58..=64 => run.revoke(rng, Mode::Plain).await,
            65..=67 => run.revoke(rng, Mode::AppendRace).await,
            68..=70 => run.revoke(rng, Mode::CheckpointRace).await,
            71..=73 => run.revoke(rng, Mode::Crash).await,
            74..=76 => run.crash_append(rng).await,
            77..=86 => run.checkpoint(rng).await,
            87..=89 => run.process_crash(rng).await,
            90..=96 => run.read(rng).await,
            _ => run.attest(rng).await,
        }
    }
    if !run.broken {
        run.final_check(rng).await;
    }
    run.close().await;
}

/// How a lifecycle change reaches the projector.
#[derive(Clone, Copy)]
enum Mode {
    /// Applied on its own.
    Plain,
    /// Applied while an append from the revoked source races it.
    AppendRace,
    /// Applied while a checkpoint folding a revoked write races it.
    CheckpointRace,
    /// Applied by a process that crashes while the transaction is in flight.
    Crash,
}

/// The digest of the input a source generation consumed. Fixed per source:
/// input edits at an unchanged generation are outside this experiment.
fn input_digest(key: &str, generation: u64) -> [u8; 32] {
    sha256(format!("{key}\u{0}{generation}").as_bytes())
}

fn source(key: &str, generation: u64) -> SourceRef {
    SourceRef {
        key: key.to_owned(),
        generation: Generation(generation),
        input_digest: input_digest(key, generation),
    }
}

/// The oracle's admission: the runtime says the generation is live and the
/// write consumed the input that generation holds.
fn admissible_in(oracle: &PtrRuntime, source: &SourceRef) -> bool {
    oracle.generation_validity(&source.key, source.generation) == Some(Validity::Live)
        && source.input_digest == input_digest(&source.key, source.generation.0)
}

/// Which writes a lifecycle change makes inadmissible: the rule the projector
/// deletes journal rows by, and the one the process refolds its memories by.
fn removed_by(change: &LifecycleChange, source: &SourceRef) -> bool {
    match change {
        LifecycleChange::SetLive {
            target, generation, ..
        } => source.key == *target && source.generation != *generation,
        LifecycleChange::Tombstone {
            subject,
            generation,
        } => source.key == *subject && source.generation == *generation,
        LifecycleChange::Revision { .. } | LifecycleChange::None => false,
    }
}

/// The ledger, the generator's view of its lifecycle, and the runtime oracle
/// replayed from it.
struct Ledger {
    log: Vec<CommittedEvent>,
    head: LogAnchor,
    oracle: PtrRuntime,
    live: BTreeMap<String, u64>,
    tombstones: BTreeSet<(String, u64)>,
    projects: BTreeMap<String, String>,
}

impl Ledger {
    fn new() -> Self {
        Self {
            log: Vec::new(),
            head: LogAnchor::empty(),
            oracle: PtrRuntime::replay(PtrConfig::default(), &[]).expect("an empty log replays"),
            live: BTreeMap::new(),
            tombstones: BTreeSet::new(),
            projects: BTreeMap::new(),
        }
    }

    fn fence(&self) -> CommitIndex {
        CommitIndex(self.log.len() as u64)
    }

    fn prepare(&self, event: LedgerEvent) -> (CommittedEvent, LogAnchor) {
        let record = CommittedEvent {
            index: CommitIndex(self.log.len() as u64 + 1),
            event,
        };
        let anchor =
            chain_anchors(std::slice::from_ref(&record), self.head).expect("a record chains")[0];
        (record, anchor)
    }

    /// The runtime's view after `record`, which it must accept.
    fn preview(&self, record: &CommittedEvent) -> Result<PtrRuntime, String> {
        let mut log = self.log.clone();
        log.push(record.clone());
        PtrRuntime::replay(PtrConfig::default(), &log).map_err(|error| format!("{error:?}"))
    }

    fn commit(&mut self, record: CommittedEvent, anchor: LogAnchor, oracle: PtrRuntime) {
        match &record.event {
            LedgerEvent::CapsuleCommitted {
                project,
                capsule,
                generation,
            } => {
                self.projects.insert(capsule.0.clone(), project.0.clone());
                self.live.insert(capsule.0.clone(), generation.0);
            }
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                self.live.insert(capsule.0.clone(), new.0);
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                self.live.insert(format!("constraint:{key}"), generation.0);
            }
            LedgerEvent::ProcedurePromoted { id, generation } => {
                self.live.insert(format!("procedure:{id}"), generation.0);
            }
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                self.tombstones.insert((subject.clone(), generation.0));
            }
            LedgerEvent::ProcedureRevoked { id, generation } => {
                self.tombstones
                    .insert((format!("procedure:{id}"), generation.0));
            }
            _ => {}
        }
        self.log.push(record);
        self.head = anchor;
        self.oracle = oracle;
    }

    fn admissible_sources(&self) -> Vec<SourceRef> {
        self.live
            .iter()
            .map(|(target, generation)| source(target, *generation))
            .filter(|source| admissible_in(&self.oracle, source))
            .collect()
    }

    /// A source the lifecycle authority does not admit: a revoked or
    /// superseded generation, one ahead of the live one, or a target never
    /// committed.
    fn inadmissible_source(&self, rng: &mut Rng) -> SourceRef {
        let tombstones: Vec<&(String, u64)> = self.tombstones.iter().collect();
        let live: Vec<(&String, &u64)> = self.live.iter().collect();
        match rng.below(4) {
            0 if !tombstones.is_empty() => {
                let (key, generation) = rng.pick(&tombstones);
                source(key, *generation)
            }
            1 if !live.is_empty() => {
                let (key, generation) = rng.pick(&live);
                source(key, *generation + rng.range(1, 2))
            }
            2 if live.iter().any(|(_, generation)| **generation > 0) => {
                let older: Vec<&(&String, &u64)> = live
                    .iter()
                    .filter(|(_, generation)| **generation > 0)
                    .collect();
                let (key, generation) = rng.pick(&older);
                source(key, rng.below(**generation))
            }
            _ => source(&format!("never-committed-{}", rng.below(5)), rng.below(3)),
        }
    }

    /// A generation the runtime would activate for `target`: at least the
    /// live one and never tombstoned.
    fn activation_generation(
        &self,
        rng: &mut Rng,
        target: &str,
        above: Option<u64>,
    ) -> Option<u64> {
        for _ in 0..8 {
            let generation = match (above, self.live.get(target)) {
                (Some(floor), _) => floor + rng.range(1, 3),
                (None, Some(current)) => current + rng.below(3),
                (None, None) => rng.below(4),
            };
            if !self.tombstones.contains(&(target.to_owned(), generation)) {
                return Some(generation);
            }
        }
        None
    }

    fn activation(&self, rng: &mut Rng) -> Option<LedgerEvent> {
        match rng.below(10) {
            0..=5 => {
                let existing: Vec<&String> = self.projects.keys().collect();
                let capsule = if !existing.is_empty() && rng.chance(0.3) {
                    (*rng.pick(&existing)).clone()
                } else {
                    format!("{}-{}", rng.pick(&CAPSULES), rng.below(6))
                };
                let generation = self.activation_generation(rng, &capsule, None)?;
                let project = self
                    .projects
                    .get(&capsule)
                    .cloned()
                    .unwrap_or_else(|| rng.pick(&PROJECTS).to_string());
                Some(LedgerEvent::CapsuleCommitted {
                    project: ProjectId(project),
                    capsule: CapsuleId(capsule),
                    generation: Generation(generation),
                })
            }
            6..=7 => {
                let key = rng.pick(&CONSTRAINTS).to_string();
                let generation =
                    self.activation_generation(rng, &format!("constraint:{key}"), None)?;
                Some(LedgerEvent::HardConstraintCommitted {
                    key,
                    generation: Generation(generation),
                })
            }
            _ => {
                let id = rng.pick(&PROCEDURES).to_string();
                let generation =
                    self.activation_generation(rng, &format!("procedure:{id}"), None)?;
                Some(LedgerEvent::ProcedurePromoted {
                    id,
                    generation: Generation(generation),
                })
            }
        }
    }

    /// A change that makes the live, admissible `source` inadmissible: a
    /// revocation of its generation, a supersession, or an activation of a
    /// later generation.
    fn revocation(&self, rng: &mut Rng, source: &SourceRef) -> Option<LedgerEvent> {
        let target = source.key.as_str();
        let generation = source.generation.0;
        if let Some(key) = target.strip_prefix("constraint:") {
            return if rng.chance(0.5) {
                Some(LedgerEvent::Revoked {
                    subject: target.to_owned(),
                    generation: source.generation,
                })
            } else {
                let later = self.activation_generation(rng, target, Some(generation))?;
                Some(LedgerEvent::HardConstraintCommitted {
                    key: key.to_owned(),
                    generation: Generation(later),
                })
            };
        }
        if let Some(id) = target.strip_prefix("procedure:") {
            return if rng.chance(0.5) {
                Some(LedgerEvent::ProcedureRevoked {
                    id: id.to_owned(),
                    generation: source.generation,
                })
            } else {
                let later = self.activation_generation(rng, target, Some(generation))?;
                Some(LedgerEvent::ProcedurePromoted {
                    id: id.to_owned(),
                    generation: Generation(later),
                })
            };
        }
        match rng.below(3) {
            0 => Some(LedgerEvent::Revoked {
                subject: target.to_owned(),
                generation: source.generation,
            }),
            1 => {
                let later = self.activation_generation(rng, target, Some(generation))?;
                Some(LedgerEvent::CapsuleSuperseded {
                    capsule: CapsuleId(target.to_owned()),
                    old: source.generation,
                    new: Generation(later),
                })
            }
            _ => {
                let later = self.activation_generation(rng, target, Some(generation))?;
                Some(LedgerEvent::CapsuleCommitted {
                    project: ProjectId(self.projects.get(target)?.clone()),
                    capsule: CapsuleId(target.to_owned()),
                    generation: Generation(later),
                })
            }
        }
    }

    /// A revocation of another generation of `source`'s target, which must
    /// remove nothing that is still admissible.
    fn off_target_revocation(&self, rng: &mut Rng, source: &SourceRef) -> Option<LedgerEvent> {
        let live = source.generation.0;
        let generation = if live > 0 && rng.chance(0.5) {
            rng.below(live)
        } else {
            live + rng.range(1, 2)
        };
        Some(match source.key.strip_prefix("procedure:") {
            Some(id) => LedgerEvent::ProcedureRevoked {
                id: id.to_owned(),
                generation: Generation(generation),
            },
            None => LedgerEvent::Revoked {
                subject: source.key.clone(),
                generation: Generation(generation),
            },
        })
    }
}

/// One memory, the process's live copy of it, and every write the journal
/// acknowledged, as composed.
struct Tracked {
    id: String,
    config: FastMemoryConfig,
    memory: FastMemory,
    acknowledged: Vec<(WriteSeq, WriteRequest)>,
    /// Checkpoints the store accepted, by applied sequence number, with the
    /// journal prefix each folded. One stays valid while the journal keeps
    /// exactly that prefix; once it is not, it is dropped for good.
    stored: BTreeMap<WriteSeq, Vec<(WriteSeq, SourceRef)>>,
}

impl Tracked {
    /// The journal the memory must have: every acknowledged write the oracle
    /// still admits, in sequence order.
    fn expected(&self, oracle: &PtrRuntime) -> Vec<(WriteSeq, WriteRequest)> {
        self.acknowledged
            .iter()
            .filter(|(_, request)| admissible_in(oracle, &request.source))
            .cloned()
            .collect()
    }

    /// Record a checkpoint the store accepted, folding `journal` up to
    /// `applied`.
    fn remember(&mut self, applied: WriteSeq, journal: &[(WriteSeq, WriteRequest)]) {
        self.stored
            .insert(applied, prefix_sources(journal, applied));
    }

    /// Drop every remembered checkpoint that no longer folds a prefix of
    /// `journal`, and return the valid ones.
    fn valid_checkpoints(&mut self, journal: &[(WriteSeq, WriteRequest)]) -> BTreeSet<WriteSeq> {
        self.stored
            .retain(|applied, folded| prefix_sources(journal, *applied) == *folded);
        self.stored.keys().copied().collect()
    }
}

fn prefix_sources(
    journal: &[(WriteSeq, WriteRequest)],
    applied: WriteSeq,
) -> Vec<(WriteSeq, SourceRef)> {
    journal
        .iter()
        .filter(|(seq, _)| *seq <= applied)
        .map(|(seq, request)| (*seq, request.source.clone()))
        .collect()
}

/// The process's two sessions: one appends and checkpoints, one projects.
struct Sessions {
    writer: Option<PgSubstrate>,
    projector: Option<PgSubstrate>,
}

impl Sessions {
    async fn open(instance: &Instance) -> Self {
        Self {
            writer: Some(instance.connect().await),
            projector: Some(instance.connect().await),
        }
    }

    fn writer(&mut self) -> &mut PgSubstrate {
        self.writer.as_mut().expect("the writer session is open")
    }

    fn projector(&mut self) -> &mut PgSubstrate {
        self.projector
            .as_mut()
            .expect("the projector session is open")
    }

    fn both(&mut self) -> (&mut PgSubstrate, &mut PgSubstrate) {
        (
            self.writer.as_mut().expect("the writer session is open"),
            self.projector
                .as_mut()
                .expect("the projector session is open"),
        )
    }
}

struct Case<'a> {
    raw: &'a Client,
    instance: Instance,
    sessions: Sessions,
    ledger: Ledger,
    memories: Vec<Tracked>,
    metrics: &'a mut Metrics,
    /// `seed … case …`, the prefix of every divergence message.
    label: String,
    /// The projection and the ledger no longer agree; the failure is counted
    /// and the case stops rather than report its consequences.
    broken: bool,
}

impl Case<'_> {
    fn expected_all(&self, oracle: &PtrRuntime) -> Vec<Vec<(WriteSeq, WriteRequest)>> {
        self.memories
            .iter()
            .map(|tracked| tracked.expected(oracle))
            .collect()
    }

    async fn register(&mut self, index: usize, rng: &mut Rng) {
        let config = FastMemoryConfig {
            heads: rng.range(1, 4) as usize,
            key_dim: rng.range(1, 12) as usize,
            value_dim: rng.range(1, 12) as usize,
            checkpoint_interval: rng.range(1, 6) as u32,
            max_writes: 65_536,
        };
        let record = FastMemoryRecord {
            id: format!("memory {index} — ü"),
            principal: PrincipalId(format!("agent-{index}")),
            thread: format!("thread '{index}'"),
            config,
            projection_digest: rng.digest(),
            codebook_seed: rng.next_u64(),
        };
        self.sessions
            .writer()
            .create_memory(&record)
            .await
            .expect("register a memory");
        self.metrics.memories += 1;
        self.memories.push(Tracked {
            id: record.id,
            config,
            memory: FastMemory::new(config).expect("a supported shape"),
            acknowledged: Vec::new(),
            stored: BTreeMap::new(),
        });
    }

    async fn activate(&mut self, rng: &mut Rng) {
        if let Some(event) = self.ledger.activation(rng) {
            self.apply(event, rng).await;
        }
    }

    async fn attest(&mut self, rng: &mut Rng) {
        let event = LedgerEvent::VerifierAttested {
            subject: format!("attestation {}", rng.below(100)),
            passed: rng.chance(0.5),
        };
        self.apply(event, rng).await;
    }

    async fn apply(&mut self, event: LedgerEvent, rng: &mut Rng) {
        let (record, anchor) = self.ledger.prepare(event);
        match self.ledger.preview(&record) {
            Ok(after) => self.apply_prepared(record, anchor, after, rng).await,
            Err(error) => diverged(
                &mut self.metrics.invalid_logs,
                format!(
                    "{}: the runtime refused a generated record: {error}",
                    self.label
                ),
            ),
        }
    }

    async fn apply_prepared(
        &mut self,
        record: CommittedEvent,
        anchor: LogAnchor,
        after: PtrRuntime,
        rng: &mut Rng,
    ) {
        let pre = self.expected_all(&self.ledger.oracle);
        let result = self
            .sessions
            .projector()
            .apply_committed(&record, anchor)
            .await;
        let removed = match result {
            Ok(report) if report.outcome == ApplyOutcome::Applied => report.removed_writes,
            other => {
                diverged(
                    &mut self.metrics.crash_recovery_failures,
                    format!(
                        "{}: the projector did not apply {}: {other:?}",
                        self.label, record.index.0
                    ),
                );
                self.broken = true;
                return;
            }
        };
        self.ledger.commit(record.clone(), anchor, after);
        self.after_commit(&record, pre, Some(removed), true, rng)
            .await;
    }

    /// Everything that must hold once a record is committed: the journals
    /// PostgreSQL kept, the checkpoints it still holds, denial in the window
    /// before the refold, and the refold itself.
    async fn after_commit(
        &mut self,
        record: &CommittedEvent,
        pre: Vec<Vec<(WriteSeq, WriteRequest)>>,
        removed_by_projector: Option<u64>,
        window: bool,
        rng: &mut Rng,
    ) {
        self.metrics.ledger_records += 1;
        let change = lifecycle_change(record);
        let post = self.expected_all(&self.ledger.oracle);
        let removed: u64 = pre
            .iter()
            .zip(&post)
            .map(|(before, after)| before.len().saturating_sub(after.len()) as u64)
            .sum();
        self.metrics.writes_removed += removed;
        if let Some(reported) = removed_by_projector {
            if reported != removed {
                diverged(
                    &mut self.metrics.removed_count_mismatches,
                    format!(
                        "{}: record {} removed {reported} writes, the model says {removed}",
                        self.label, record.index.0
                    ),
                );
            }
        }
        // Every record is checked, not only those the projector's mapping calls
        // lifecycle changes: a change mapped to nothing must still show here.
        if matches!(
            change,
            LifecycleChange::SetLive { .. } | LifecycleChange::Tombstone { .. }
        ) {
            self.metrics.lifecycle_records += 1;
        }
        for (index, expected) in post.iter().enumerate() {
            self.check_journal(index, expected).await;
            self.check_checkpoint_rows(index, expected).await;
            if window {
                self.window_probe(index, expected, rng).await;
            }
            // The process is handed the committed record, as a consumer of the
            // projection's change feed would be, and refolds without every
            // write the change made inadmissible.
            let report = self.memories[index]
                .memory
                .revoke(|source| removed_by(&change, source));
            if report.removed > 0 {
                let remaining = self.memories[index].memory.writes().len() as u64;
                self.metrics.revocations_with_removal += 1;
                self.metrics.writes_replayed += report.replayed as u64;
                self.metrics.full_refold_writes += remaining;
                self.metrics.max_replayed = self.metrics.max_replayed.max(report.replayed as u64);
            }
            self.check_fold(index, expected, rng).await;
            if rng.chance(0.25) {
                self.check_latest_checkpoint(index, expected).await;
            }
        }
    }

    /// The journal PostgreSQL holds, bit for bit.
    async fn check_journal(&mut self, index: usize, expected: &[(WriteSeq, WriteRequest)]) {
        self.metrics.journal_checks += 1;
        self.metrics.max_journal_len = self.metrics.max_journal_len.max(expected.len() as u64);
        let id = self.memories[index].id.clone();
        match self.sessions.writer().load_journal(&id).await {
            Ok(journal) if same_journal(&journal, expected) => {}
            Ok(journal) => diverged(
                &mut self.metrics.journal_mismatches,
                format!(
                    "{}: the journal of {id:?} holds {} writes, the model {}",
                    self.label,
                    journal.len(),
                    expected.len()
                ),
            ),
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!(
                    "{}: the journal of {id:?} is unreadable: {error:?}",
                    self.label
                ),
            ),
        }
    }

    /// Every stored checkpoint must still fold exactly the journal prefix it
    /// names — its binding and, decoded, its cells — because a revocation
    /// deletes the ones that folded a removed write, even one committed while
    /// it waited. And every checkpoint the store accepted that still folds the
    /// journal must still be there: the cascade deletes nothing it need not.
    async fn check_checkpoint_rows(&mut self, index: usize, expected: &[(WriteSeq, WriteRequest)]) {
        let id = self.memories[index].id.clone();
        let config = self.memories[index].config;
        let rows = match self
            .raw
            .query(
                &format!(
                    "SELECT applied_seq, binding_digest, state FROM {}_work.fastmem_checkpoint \
                     WHERE memory = $1",
                    self.instance.prefix
                ),
                &[&id],
            )
            .await
        {
            Ok(rows) => rows,
            Err(error) => {
                diverged(
                    &mut self.metrics.read_failures,
                    format!(
                        "{}: the checkpoint rows are unreadable: {error}",
                        self.label
                    ),
                );
                return;
            }
        };
        let mut present = BTreeSet::new();
        for row in rows {
            let applied = WriteSeq(row.get::<_, i64>(0) as u64);
            present.insert(applied);
            let digest: Vec<u8> = row.get(1);
            let bytes: Vec<u8> = row.get(2);
            let prefix: Vec<(WriteSeq, WriteRequest)> = expected
                .iter()
                .filter(|(seq, _)| *seq <= applied)
                .cloned()
                .collect();
            let binding =
                binding_digest_of(prefix.iter().map(|(seq, request)| (*seq, &request.source)));
            let current =
                prefix.last().map(|(seq, _)| *seq) == Some(applied) && binding[..] == digest[..];
            if !current {
                diverged(
                    &mut self.metrics.stale_checkpoint_rows,
                    format!(
                        "{}: a checkpoint of {id:?} at {} no longer folds the journal",
                        self.label, applied.0
                    ),
                );
                continue;
            }
            let folded = FastMemory::restore(config, prefix)
                .ok()
                .zip(decode_state(&bytes).ok())
                .is_some_and(|(fold, state)| same_state(fold.state(), &state));
            if !folded {
                diverged(
                    &mut self.metrics.checkpoint_violations,
                    format!(
                        "{}: the stored state of {id:?} at {} is not the fold of its prefix",
                        self.label, applied.0
                    ),
                );
            }
        }
        for applied in self.memories[index].valid_checkpoints(expected) {
            if !present.contains(&applied) {
                diverged(
                    &mut self.metrics.checkpoints_lost,
                    format!(
                        "{}: the checkpoint of {id:?} at {} still folds the journal but is gone",
                        self.label, applied.0
                    ),
                );
            }
        }
    }

    /// Ask the projection which of `sources` it admits at the ledger head, and
    /// compare every answer with the oracle's.
    async fn projection_admits(&mut self, sources: &BTreeSet<SourceRef>) -> BTreeSet<SourceRef> {
        let fence = self.ledger.fence();
        let mut admitted = BTreeSet::new();
        for source in sources {
            match self
                .sessions
                .projector()
                .is_admissible(&source.key, source.generation, fence)
                .await
            {
                Ok(answer) => {
                    let expected = self
                        .ledger
                        .oracle
                        .generation_validity(&source.key, source.generation)
                        == Some(Validity::Live);
                    if answer != expected {
                        diverged(
                            &mut self.metrics.admission_divergences,
                            format!(
                                "{}: the projection admits ({:?}, {}) {answer}, the runtime {expected}",
                                self.label,
                                source.key,
                                source.generation.0
                            ),
                        );
                    }
                    if answer
                        && source.input_digest == input_digest(&source.key, source.generation.0)
                    {
                        admitted.insert(source.clone());
                    }
                }
                Err(error) => diverged(
                    &mut self.metrics.read_failures,
                    format!("{}: admission is unreadable: {error:?}", self.label),
                ),
            }
        }
        admitted
    }

    /// Between a change's commit and the refold, a read of a memory that
    /// still folds an inadmissible write must be denied, and a checkpoint of
    /// that state refused.
    async fn window_probe(
        &mut self,
        index: usize,
        expected: &[(WriteSeq, WriteRequest)],
        rng: &mut Rng,
    ) {
        let sources: BTreeSet<SourceRef> = self.memories[index]
            .memory
            .sources()
            .into_iter()
            .cloned()
            .collect();
        let admitted = self.projection_admits(&sources).await;
        let depends = self.memories[index]
            .memory
            .writes()
            .iter()
            .any(|write| !admissible_in(&self.ledger.oracle, write.source()));
        let Some(query) = random_query(rng, &self.memories[index].config) else {
            return;
        };
        self.metrics.window_reads += 1;
        let read = self.memories[index]
            .memory
            .read_admitted(&query, |source| admitted.contains(source));
        match read {
            Ok(_) if depends => diverged(
                &mut self.metrics.resurrected_reads,
                format!(
                    "{}: a read of {:?} was admitted while it folds a revoked write",
                    self.label, self.memories[index].id
                ),
            ),
            Ok(_) => {}
            Err(_) if depends => self.metrics.window_denials += 1,
            Err(error) => diverged(
                &mut self.metrics.false_denials,
                format!(
                    "{}: a read with only admissible writes gave {error}",
                    self.label
                ),
            ),
        }
        if depends && rng.chance(0.25) {
            self.plant_stale_checkpoint(index, expected).await;
        } else if depends && rng.chance(0.3) {
            let tracked = &self.memories[index];
            let (id, binding, state) = (
                tracked.id.clone(),
                tracked.memory.binding_digest(),
                tracked.memory.state().clone(),
            );
            match self
                .sessions
                .writer()
                .put_checkpoint(&id, binding, &state)
                .await
            {
                Err(PgError::InvalidCheckpoint { .. }) => {
                    self.metrics.stale_checkpoints_refused += 1;
                }
                Ok(()) => diverged(
                    &mut self.metrics.stale_checkpoints_accepted,
                    format!(
                        "{}: a checkpoint folding a revoked write was stored",
                        self.label
                    ),
                ),
                Err(error) => diverged(
                    &mut self.metrics.read_failures,
                    format!("{}: a stale checkpoint gave {error:?}", self.label),
                ),
            }
        }
    }

    /// Plant a checkpoint of the unrefolded state straight into storage,
    /// bypassing every check a writer passes, and require the newest
    /// checkpoint handed out to be a fold of the journal all the same.
    async fn plant_stale_checkpoint(
        &mut self,
        index: usize,
        expected: &[(WriteSeq, WriteRequest)],
    ) {
        let tracked = &self.memories[index];
        let id = tracked.id.clone();
        let applied = tracked.memory.state().applied().0 as i64;
        let planted = self
            .raw
            .execute(
                &format!(
                    "INSERT INTO {}_work.fastmem_checkpoint \
                     (memory, applied_seq, binding_digest, state) VALUES ($1, $2, $3, $4) \
                     ON CONFLICT (memory, applied_seq) DO NOTHING",
                    self.instance.prefix
                ),
                &[
                    &id,
                    &applied,
                    &tracked.memory.binding_digest().to_vec(),
                    &encode_state(tracked.memory.state()),
                ],
            )
            .await
            .expect("plant a stale checkpoint");
        if planted == 0 {
            return;
        }
        self.metrics.planted_stale_checkpoints += 1;
        self.check_latest_checkpoint(index, expected).await;
        self.raw
            .execute(
                &format!(
                    "DELETE FROM {}_work.fastmem_checkpoint WHERE memory = $1 AND applied_seq = $2",
                    self.instance.prefix
                ),
                &[&id, &applied],
            )
            .await
            .expect("remove the planted checkpoint");
    }

    /// The live memory must be the never-saw-it fold bit for bit, and a read
    /// the projection admits must return that fold's readout bit for bit.
    async fn check_fold(
        &mut self,
        index: usize,
        expected: &[(WriteSeq, WriteRequest)],
        rng: &mut Rng,
    ) {
        self.metrics.bit_identity_checks += 1;
        let config = self.memories[index].config;
        let never = match FastMemory::restore(config, expected.iter().cloned()) {
            Ok(never) => never,
            Err(error) => {
                diverged(
                    &mut self.metrics.invalid_logs,
                    format!("{}: the never-saw-it fold was refused: {error}", self.label),
                );
                return;
            }
        };
        let memory = &self.memories[index].memory;
        if !same_state(memory.state(), never.state())
            || memory.binding_digest() != never.binding_digest()
        {
            diverged(
                &mut self.metrics.bit_identity_failures,
                format!(
                    "{}: {:?} differs from the never-saw-it fold ({} writes, {} expected)",
                    self.label,
                    self.memories[index].id,
                    memory.writes().len(),
                    expected.len()
                ),
            );
        }
        let Some(query) = random_query(rng, &config) else {
            return;
        };
        let sources: BTreeSet<SourceRef> = self.memories[index]
            .memory
            .sources()
            .into_iter()
            .cloned()
            .collect();
        let admitted = self.projection_admits(&sources).await;
        self.metrics.readout_checks += 1;
        let reference = never
            .read_admitted(&query, |_| true)
            .expect("the never-saw-it fold is readable");
        match self.memories[index]
            .memory
            .read_admitted(&query, |source| admitted.contains(source))
        {
            Ok(readout)
                if readout.as_of == reference.as_of
                    && same_bits(&readout.values, &reference.values) => {}
            Ok(_) => diverged(
                &mut self.metrics.readout_failures,
                format!(
                    "{}: a readout differs from the never-saw-it fold",
                    self.label
                ),
            ),
            Err(error) => diverged(
                &mut self.metrics.false_denials,
                format!("{}: a refolded memory's read gave {error}", self.label),
            ),
        }
    }

    /// The newest checkpoint PostgreSQL hands out must be the fold of the
    /// journal prefix it names, bound to that prefix.
    async fn check_latest_checkpoint(
        &mut self,
        index: usize,
        expected: &[(WriteSeq, WriteRequest)],
    ) {
        let id = self.memories[index].id.clone();
        let config = self.memories[index].config;
        let newest = self.memories[index]
            .valid_checkpoints(expected)
            .last()
            .copied();
        match self.sessions.writer().latest_checkpoint(&id).await {
            Ok(None) if newest.is_none() => {}
            Ok(None) => diverged(
                &mut self.metrics.checkpoint_violations,
                format!(
                    "{}: no checkpoint of {id:?} was handed out, the newest valid is at {}",
                    self.label,
                    newest.map_or(0, |applied| applied.0)
                ),
            ),
            Ok(Some(checkpoint)) => {
                self.metrics.checkpoints_verified += 1;
                if Some(checkpoint.applied) != newest {
                    diverged(
                        &mut self.metrics.checkpoint_violations,
                        format!(
                            "{}: the checkpoint of {id:?} handed out is at {}, the newest valid at {:?}",
                            self.label, checkpoint.applied.0, newest
                        ),
                    );
                }
                let prefix: Vec<(WriteSeq, WriteRequest)> = expected
                    .iter()
                    .filter(|(seq, _)| *seq <= checkpoint.applied)
                    .cloned()
                    .collect();
                let binding =
                    binding_digest_of(prefix.iter().map(|(seq, request)| (*seq, &request.source)));
                let folded = prefix.last().map(|(seq, _)| *seq) == Some(checkpoint.applied)
                    && binding == checkpoint.binding_digest
                    && FastMemory::restore(config, prefix)
                        .is_ok_and(|fold| same_state(fold.state(), &checkpoint.state));
                if !folded {
                    diverged(
                        &mut self.metrics.checkpoint_violations,
                        format!(
                            "{}: the checkpoint of {id:?} at {} is not the fold of the journal",
                            self.label, checkpoint.applied.0
                        ),
                    );
                }
            }
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!(
                    "{}: the latest checkpoint is unreadable: {error:?}",
                    self.label
                ),
            ),
        }
    }

    /// Replace the live memory with one restored from the stored journal, as
    /// a restarted process does.
    async fn restore_memory(&mut self, index: usize) {
        let id = self.memories[index].id.clone();
        let config = self.memories[index].config;
        match self.sessions.writer().load_journal(&id).await {
            Ok(journal) => match FastMemory::restore(config, journal) {
                Ok(memory) => {
                    self.memories[index].memory = memory;
                    self.metrics.restores += 1;
                }
                Err(error) => {
                    diverged(
                        &mut self.metrics.journal_mismatches,
                        format!(
                            "{}: the journal of {id:?} does not restore: {error}",
                            self.label
                        ),
                    );
                    self.broken = true;
                }
            },
            Err(error) => {
                diverged(
                    &mut self.metrics.read_failures,
                    format!(
                        "{}: the journal of {id:?} is unreadable: {error:?}",
                        self.label
                    ),
                );
                self.broken = true;
            }
        }
    }

    /// Write from an admissible source: fold in the process, then append.
    async fn write(&mut self, rng: &mut Rng) {
        let sources = self.ledger.admissible_sources();
        if sources.is_empty() {
            return self.activate(rng).await;
        }
        let source = rng.pick(&sources).clone();
        let index = rng.index(self.memories.len());
        let request = compose(rng, &self.memories[index].config, source);
        let Ok(receipt) = self.memories[index].memory.write(request.clone()) else {
            // A zero key or an underflowing head: refused before anything
            // was folded or journaled.
            return;
        };
        let id = self.memories[index].id.clone();
        match self
            .sessions
            .writer()
            .append_write(&id, receipt.seq, &request)
            .await
        {
            Ok(()) => {
                self.metrics.writes_acknowledged += 1;
                self.memories[index]
                    .acknowledged
                    .push((receipt.seq, request));
            }
            Err(error) => {
                diverged(
                    &mut self.metrics.append_false_refusals,
                    format!("{}: an admissible append gave {error:?}", self.label),
                );
                self.restore_memory(index).await;
            }
        }
    }

    /// Write from an inadmissible source: the append must be refused, and the
    /// process drops the write it had folded.
    async fn inadmissible_write(&mut self, rng: &mut Rng) {
        let source = self.ledger.inadmissible_source(rng);
        let index = rng.index(self.memories.len());
        let request = compose(rng, &self.memories[index].config, source.clone());
        let Ok(receipt) = self.memories[index].memory.write(request.clone()) else {
            return;
        };
        self.metrics.inadmissible_probes += 1;
        let id = self.memories[index].id.clone();
        match self
            .sessions
            .writer()
            .append_write(&id, receipt.seq, &request)
            .await
        {
            Err(PgError::NotLive { .. }) => {}
            Ok(()) => diverged(
                &mut self.metrics.inadmissible_appends_accepted,
                format!(
                    "{}: an append from ({:?}, {}) was accepted",
                    self.label, source.key, source.generation.0
                ),
            ),
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!("{}: an inadmissible append gave {error:?}", self.label),
            ),
        }
        let _ = self.memories[index]
            .memory
            .revoke(|written| *written == source);
        let expected = self.memories[index].expected(&self.ledger.oracle);
        self.check_fold(index, &expected, rng).await;
    }

    async fn checkpoint(&mut self, rng: &mut Rng) {
        let index = rng.index(self.memories.len());
        let applied = self.memories[index].memory.state().applied();
        if applied.0 == 0 || self.checkpoint_exists(index, applied).await {
            return;
        }
        let tracked = &self.memories[index];
        let (id, binding, state) = (
            tracked.id.clone(),
            tracked.memory.binding_digest(),
            tracked.memory.state().clone(),
        );
        match self
            .sessions
            .writer()
            .put_checkpoint(&id, binding, &state)
            .await
        {
            Ok(()) => {
                self.metrics.checkpoints_put += 1;
                let journal = self.memories[index].expected(&self.ledger.oracle);
                self.memories[index].remember(state.applied(), &journal);
            }
            Err(error) => diverged(
                &mut self.metrics.checkpoint_false_refusals,
                format!(
                    "{}: a current checkpoint was refused: {error:?}",
                    self.label
                ),
            ),
        }
        if rng.chance(0.5) {
            let expected = self.memories[index].expected(&self.ledger.oracle);
            self.check_latest_checkpoint(index, &expected).await;
        }
    }

    async fn checkpoint_exists(&self, index: usize, applied: WriteSeq) -> bool {
        self.raw
            .query_one(
                &format!(
                    "SELECT EXISTS (SELECT 1 FROM {}_work.fastmem_checkpoint \
                     WHERE memory = $1 AND applied_seq = $2)",
                    self.instance.prefix
                ),
                &[&self.memories[index].id, &(applied.0 as i64)],
            )
            .await
            .expect("look up a checkpoint")
            .get(0)
    }

    async fn read(&mut self, rng: &mut Rng) {
        let index = rng.index(self.memories.len());
        let expected = self.memories[index].expected(&self.ledger.oracle);
        self.check_fold(index, &expected, rng).await;
    }

    /// The process dies between operations: every live memory is lost and
    /// restored from its journal.
    async fn process_crash(&mut self, rng: &mut Rng) {
        self.metrics.process_crashes += 1;
        self.sessions = Sessions::open(&self.instance).await;
        for index in 0..self.memories.len() {
            self.restore_memory(index).await;
            let expected = self.memories[index].expected(&self.ledger.oracle);
            self.check_fold(index, &expected, rng).await;
        }
    }

    async fn revoke(&mut self, rng: &mut Rng, mode: Mode) {
        // Prefer a source some memory folds, so the change has work to do.
        let mut candidates: Vec<SourceRef> = self
            .memories
            .iter()
            .flat_map(|tracked| tracked.memory.sources().into_iter().cloned())
            .filter(|source| admissible_in(&self.ledger.oracle, source))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if candidates.is_empty() || rng.chance(0.5) {
            candidates = self.ledger.admissible_sources();
        }
        if candidates.is_empty() {
            return self.activate(rng).await;
        }
        let source = rng.pick(&candidates).clone();
        let event = match mode {
            Mode::Plain if rng.chance(0.12) => self.ledger.off_target_revocation(rng, &source),
            _ => self.ledger.revocation(rng, &source),
        };
        let Some(event) = event else {
            return;
        };
        match mode {
            Mode::Plain => self.apply(event, rng).await,
            Mode::AppendRace => self.append_race(event, source, rng).await,
            Mode::CheckpointRace => self.checkpoint_race(event, source, rng).await,
            Mode::Crash => self.crash_revocation(event, rng).await,
        }
    }

    /// An append from `source` and the change revoking it, on two sessions at
    /// once. Either the append is refused or the change deletes it; the
    /// journal must never keep it.
    async fn append_race(&mut self, event: LedgerEvent, source: SourceRef, rng: &mut Rng) {
        let (record, anchor) = self.ledger.prepare(event);
        let after = match self.ledger.preview(&record) {
            Ok(after) => after,
            Err(error) => {
                diverged(
                    &mut self.metrics.invalid_logs,
                    format!(
                        "{}: the runtime refused a generated record: {error}",
                        self.label
                    ),
                );
                return;
            }
        };
        let index = rng.index(self.memories.len());
        let request = compose(rng, &self.memories[index].config, source.clone());
        let Ok(receipt) = self.memories[index].memory.write(request.clone()) else {
            return self.apply_prepared(record, anchor, after, rng).await;
        };
        self.metrics.append_races += 1;
        let id = self.memories[index].id.clone();
        // The change reaches the target's row after a few statements, the
        // append after one: a wider delay on the append lets either side win.
        let (append_delay, apply_delay) = (
            Duration::from_micros(rng.below(4000)),
            Duration::from_micros(rng.below(1000)),
        );
        let (writer, projector) = self.sessions.both();
        let (appended, applied) = tokio::join!(
            async {
                tokio::time::sleep(append_delay).await;
                writer.append_write(&id, receipt.seq, &request).await
            },
            async {
                tokio::time::sleep(apply_delay).await;
                projector.apply_committed(&record, anchor).await
            },
        );
        match appended {
            Ok(()) => {
                self.metrics.append_races_append_first += 1;
                self.metrics.writes_acknowledged += 1;
                self.memories[index]
                    .acknowledged
                    .push((receipt.seq, request));
            }
            Err(PgError::NotLive { .. }) => self.metrics.append_races_revocation_first += 1,
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!("{}: a racing append gave {error:?}", self.label),
            ),
        }
        let removed = match applied {
            Ok(report) if report.outcome == ApplyOutcome::Applied => report.removed_writes,
            other => {
                diverged(
                    &mut self.metrics.crash_recovery_failures,
                    format!("{}: a racing change was not applied: {other:?}", self.label),
                );
                self.broken = true;
                return;
            }
        };
        let pre = self.expected_all(&self.ledger.oracle);
        self.ledger.commit(record.clone(), anchor, after);
        match self.sessions.writer().load_journal(&id).await {
            Ok(journal) if journal.iter().any(|(_, kept)| kept.source == source) => diverged(
                &mut self.metrics.race_violations,
                format!(
                    "{}: a write from a revoked source survived its race",
                    self.label
                ),
            ),
            Ok(_) => {}
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!("{}: the journal is unreadable: {error:?}", self.label),
            ),
        }
        self.after_commit(&record, pre, Some(removed), true, rng)
            .await;
    }

    /// A checkpoint that folds a write from `source` and the change revoking
    /// it, on two sessions at once. Either the checkpoint is refused or the
    /// change deletes it; none may survive that folds a revoked write.
    async fn checkpoint_race(&mut self, event: LedgerEvent, source: SourceRef, rng: &mut Rng) {
        let (record, anchor) = self.ledger.prepare(event);
        let after = match self.ledger.preview(&record) {
            Ok(after) => after,
            Err(error) => {
                diverged(
                    &mut self.metrics.invalid_logs,
                    format!(
                        "{}: the runtime refused a generated record: {error}",
                        self.label
                    ),
                );
                return;
            }
        };
        let folding: Vec<usize> = (0..self.memories.len())
            .filter(|index| self.memories[*index].memory.sources().contains(&source))
            .collect();
        let Some(&index) = folding.first() else {
            return self.apply_prepared(record, anchor, after, rng).await;
        };
        let applied = self.memories[index].memory.state().applied();
        if applied.0 == 0 || self.checkpoint_exists(index, applied).await {
            return self.apply_prepared(record, anchor, after, rng).await;
        }
        self.metrics.checkpoint_races += 1;
        let tracked = &self.memories[index];
        let (id, binding, state) = (
            tracked.id.clone(),
            tracked.memory.binding_digest(),
            tracked.memory.state().clone(),
        );
        let pre = self.expected_all(&self.ledger.oracle);
        let (put_delay, apply_delay) = (
            Duration::from_micros(rng.below(4000)),
            Duration::from_micros(rng.below(1000)),
        );
        let (writer, projector) = self.sessions.both();
        let (stored, applied) = tokio::join!(
            async {
                tokio::time::sleep(put_delay).await;
                writer.put_checkpoint(&id, binding, &state).await
            },
            async {
                tokio::time::sleep(apply_delay).await;
                projector.apply_committed(&record, anchor).await
            },
        );
        match stored {
            Ok(()) => {
                // Stored before the change: it folded the journal as it was.
                self.metrics.checkpoint_races_stored += 1;
                self.memories[index].remember(state.applied(), &pre[index]);
            }
            Err(PgError::InvalidCheckpoint { .. }) => self.metrics.checkpoint_races_refused += 1,
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!("{}: a racing checkpoint gave {error:?}", self.label),
            ),
        }
        let removed = match applied {
            Ok(report) if report.outcome == ApplyOutcome::Applied => report.removed_writes,
            other => {
                diverged(
                    &mut self.metrics.crash_recovery_failures,
                    format!("{}: a racing change was not applied: {other:?}", self.label),
                );
                self.broken = true;
                return;
            }
        };
        self.ledger.commit(record.clone(), anchor, after);
        self.after_commit(&record, pre, Some(removed), true, rng)
            .await;
        let expected = self.memories[index].expected(&self.ledger.oracle);
        self.check_latest_checkpoint(index, &expected).await;
    }

    /// The process crashes while its projector applies a change. Every
    /// journal must be entirely before or entirely after the change, as the
    /// watermark says; the restarted process restores its memories and
    /// re-sends the change if it rolled back.
    async fn crash_revocation(&mut self, event: LedgerEvent, rng: &mut Rng) {
        let (record, anchor) = self.ledger.prepare(event);
        let after = match self.ledger.preview(&record) {
            Ok(after) => after,
            Err(error) => {
                diverged(
                    &mut self.metrics.invalid_logs,
                    format!(
                        "{}: the runtime refused a generated record: {error}",
                        self.label
                    ),
                );
                return;
            }
        };
        let pre = self.expected_all(&self.ledger.oracle);
        let post = self.expected_all(&after);
        let kill = rng.chance(0.5);
        let delay = Duration::from_micros(rng.below(4000));
        self.metrics.revocation_crashes += 1;
        if kill {
            self.metrics.server_kills += 1;
        } else {
            self.metrics.client_aborts += 1;
        }
        drop(self.sessions.writer.take());
        let mut projector = self
            .sessions
            .projector
            .take()
            .expect("the projector session is open");
        let in_flight = record.clone();
        // The session ends with the task, as it would with the process.
        let task = tokio::spawn(async move { projector.apply_committed(&in_flight, anchor).await });
        let returned = pg::crash_task(&self.instance, self.raw, task, kill, delay).await;
        self.sessions = Sessions::open(&self.instance).await;
        let acknowledged = match returned {
            Ok(Some(Ok(report))) => report.outcome == ApplyOutcome::Applied,
            Ok(_) => false,
            Err(error) => {
                diverged(
                    &mut self.metrics.crash_recovery_failures,
                    format!("{}: {error}", self.label),
                );
                self.broken = true;
                return;
            }
        };

        let watermark = match self.sessions.projector().watermark().await {
            Ok(watermark) => watermark,
            Err(error) => {
                diverged(
                    &mut self.metrics.read_failures,
                    format!("{}: the watermark is unreadable: {error:?}", self.label),
                );
                self.broken = true;
                return;
            }
        };
        let committed = if watermark == anchor {
            true
        } else if watermark == self.ledger.head && acknowledged {
            diverged(
                &mut self.metrics.crash_recovery_failures,
                format!(
                    "{}: record {} was reported applied but rolled back",
                    self.label, record.index.0
                ),
            );
            self.broken = true;
            return;
        } else if watermark == self.ledger.head {
            false
        } else {
            diverged(
                &mut self.metrics.crash_recovery_failures,
                format!(
                    "{}: after a crash at {} the watermark is {}",
                    self.label, record.index.0, watermark.index.0
                ),
            );
            self.broken = true;
            return;
        };
        for index in 0..self.memories.len() {
            let expected = if committed { &post[index] } else { &pre[index] };
            let id = self.memories[index].id.clone();
            match self.sessions.writer().load_journal(&id).await {
                Ok(journal) if same_journal(&journal, expected) => {}
                Ok(_) => diverged(
                    &mut self.metrics.atomicity_failures,
                    format!(
                        "{}: after a crash the journal of {id:?} is neither before nor after \
                         record {}",
                        self.label, record.index.0
                    ),
                ),
                Err(error) => diverged(
                    &mut self.metrics.read_failures,
                    format!("{}: the journal is unreadable: {error:?}", self.label),
                ),
            }
            self.restore_memory(index).await;
        }
        if committed {
            self.metrics.revocation_crashes_committed += 1;
            self.ledger.commit(record.clone(), anchor, after);
            self.after_commit(&record, pre, None, false, rng).await;
        } else {
            self.metrics.revocation_crashes_rolled_back += 1;
            self.apply_prepared(record, anchor, after, rng).await;
        }
    }

    /// The process crashes while it appends: the journal holds the write in
    /// full or not at all, and the restarted process restores every memory.
    async fn crash_append(&mut self, rng: &mut Rng) {
        let sources = self.ledger.admissible_sources();
        if sources.is_empty() {
            return;
        }
        let source = rng.pick(&sources).clone();
        let index = rng.index(self.memories.len());
        let request = compose(rng, &self.memories[index].config, source);
        let Ok(receipt) = self.memories[index].memory.write(request.clone()) else {
            return;
        };
        let kill = rng.chance(0.5);
        let delay = Duration::from_micros(rng.below(3000));
        self.metrics.append_crashes += 1;
        if kill {
            self.metrics.server_kills += 1;
        } else {
            self.metrics.client_aborts += 1;
        }
        drop(self.sessions.projector.take());
        let mut writer = self
            .sessions
            .writer
            .take()
            .expect("the writer session is open");
        let (id, in_flight) = (self.memories[index].id.clone(), request.clone());
        let seq = receipt.seq;
        // The session ends with the task, as it would with the process.
        let task = tokio::spawn(async move { writer.append_write(&id, seq, &in_flight).await });
        let returned = pg::crash_task(&self.instance, self.raw, task, kill, delay).await;
        self.sessions = Sessions::open(&self.instance).await;
        let acknowledged = match returned {
            Ok(Some(result)) => result.is_ok(),
            Ok(None) => false,
            Err(error) => {
                diverged(
                    &mut self.metrics.crash_recovery_failures,
                    format!("{}: {error}", self.label),
                );
                self.broken = true;
                return;
            }
        };

        let expected = self.memories[index].expected(&self.ledger.oracle);
        let mut with_write = expected.clone();
        with_write.push((seq, request.clone()));
        let id = self.memories[index].id.clone();
        match self.sessions.writer().load_journal(&id).await {
            Ok(journal) if same_journal(&journal, &with_write) => {
                self.metrics.append_crashes_committed += 1;
                self.metrics.writes_acknowledged += 1;
                self.memories[index].acknowledged.push((seq, request));
            }
            Ok(journal) if same_journal(&journal, &expected) && acknowledged => diverged(
                &mut self.metrics.atomicity_failures,
                format!(
                    "{}: an append reported done was not in the journal after the crash",
                    self.label
                ),
            ),
            Ok(journal) if same_journal(&journal, &expected) => {
                self.metrics.writes_lost_in_crashes += 1;
            }
            Ok(_) => diverged(
                &mut self.metrics.atomicity_failures,
                format!("{}: a crashed append left a partial journal", self.label),
            ),
            Err(error) => diverged(
                &mut self.metrics.read_failures,
                format!("{}: the journal is unreadable: {error:?}", self.label),
            ),
        }
        for index in 0..self.memories.len() {
            self.restore_memory(index).await;
            let expected = self.memories[index].expected(&self.ledger.oracle);
            self.check_fold(index, &expected, rng).await;
        }
    }

    async fn final_check(&mut self, rng: &mut Rng) {
        for index in 0..self.memories.len() {
            let expected = self.memories[index].expected(&self.ledger.oracle);
            self.check_journal(index, &expected).await;
            self.check_checkpoint_rows(index, &expected).await;
            self.check_fold(index, &expected, rng).await;
            self.restore_memory(index).await;
            self.check_fold(index, &expected, rng).await;
            self.check_latest_checkpoint(index, &expected).await;
        }
    }

    async fn close(mut self) {
        drop(self.sessions.projector.take());
        let writer = match self.sessions.writer.take() {
            Some(writer) => writer,
            None => self.instance.connect().await,
        };
        writer.drop_all().await.expect("drop experiment instance");
    }
}

/// A write as a caller composes it, with the awkward floats a real input can
/// carry: signed zeros, subnormals, tiny and large magnitudes.
fn compose(rng: &mut Rng, config: &FastMemoryConfig, source: SourceRef) -> WriteRequest {
    let key = (0..config.key_len()).map(|_| awkward(rng)).collect();
    let value = (0..config.value_len()).map(|_| awkward(rng)).collect();
    let beta = if rng.chance(0.1) {
        1.0
    } else {
        rng.f32_between(0.01, 1.0)
    };
    let decay = match rng.below(10) {
        0..=2 => Decay::None,
        3..=7 => Decay::Scalar(if rng.chance(0.1) {
            1.0
        } else {
            rng.f32_between(0.5, 1.0)
        }),
        _ => Decay::PerChannel(
            (0..config.key_len())
                .map(|_| rng.f32_between(0.25, 1.0))
                .collect(),
        ),
    };
    WriteRequest {
        source,
        key,
        value,
        beta,
        decay,
    }
}

fn awkward(rng: &mut Rng) -> f32 {
    match rng.below(20) {
        0 => -0.0,
        1 => 0.0,
        2 => f32::MIN_POSITIVE / 8.0,
        3 => -1.0e-30,
        4 => rng.f32_between(-1000.0, 1000.0),
        _ => rng.f32_between(-1.0, 1.0),
    }
}

fn random_query(rng: &mut Rng, config: &FastMemoryConfig) -> Option<Query> {
    let raw = (0..config.key_len())
        .map(|_| rng.f32_between(-1.0, 1.0))
        .collect();
    Query::new(config, raw).ok()
}

fn same_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn same_state(left: &FastWeightState, right: &FastWeightState) -> bool {
    left.applied() == right.applied() && same_bits(left.cells(), right.cells())
}

fn same_decay(left: &Decay, right: &Decay) -> bool {
    match (left, right) {
        (Decay::None, Decay::None) => true,
        (Decay::Scalar(left), Decay::Scalar(right)) => left.to_bits() == right.to_bits(),
        (Decay::PerChannel(left), Decay::PerChannel(right)) => same_bits(left, right),
        _ => false,
    }
}

fn same_request(left: &WriteRequest, right: &WriteRequest) -> bool {
    left.source == right.source
        && same_bits(&left.key, &right.key)
        && same_bits(&left.value, &right.value)
        && left.beta.to_bits() == right.beta.to_bits()
        && same_decay(&left.decay, &right.decay)
}

fn same_journal(left: &[(WriteSeq, WriteRequest)], right: &[(WriteSeq, WriteRequest)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((left_seq, left), (right_seq, right))| {
                left_seq == right_seq && same_request(left, right)
            })
}
