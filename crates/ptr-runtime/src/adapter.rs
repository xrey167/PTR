//! Adapter promotion, adapter revocation and the improvement-cycle budget as committed history.
//!
//! **Promotion (Invariants 3, 7, 14).** Which adapter generation may serve is decided by two
//! ledger events, [`LedgerEvent::AdapterPromoted`] and [`LedgerEvent::AdapterRevoked`], and by
//! nothing else: `ptr-lineage`'s registry is a working model and does not decide. A promotion
//! starts from a [`PromotionProposal`] (a passing, baseline-bearing gate report bound to the
//! adapter, its weights digest and its base model) and is committed only after the runtime
//!
//! * re-checks that the report passed and holds no [`GateViolation::MissingBaseline`],
//! * recomputes the SHA-256 of the weights it was handed and compares it with the digest the
//!   gate report was bound to (the lineage records that digest as the trainer declared it and
//!   never reads it), and
//! * compares the frozen evaluation splits the caller names with the split lock the gate was
//!   evaluated under.
//!
//! The serving state is a projection of the committed events: the live generation and the
//! revocation tombstones of the key `ptr-adapter:<project>/<adapter>` in the runtime's lifecycle
//! maps, plus the binding (base, weights digest, report digest) of the promoted generation.
//! Keys are scoped by project, so one project's adapter name never reads another's. A use is
//! checked against the tombstones every time ([`PtrRuntime::check_adapter_use`]), not only when
//! the adapter was admitted, so a revocation reaches a holder of an earlier answer, a cache or
//! a replayed runtime alike.
//!
//! **Compaction.** The binding is not part of the compacted snapshot layouts. Rather than let
//! a snapshot silently drop it, [`PtrRuntime::export_compacted_snapshot`] refuses while any
//! adapter or improvement-cycle state exists ([`AdapterError::CompactionUnsupported`]).
//!
//! **Improvement cycles (Invariants 8, 12).** A self-improvement cycle is admitted by
//! [`PtrRuntime::begin_improvement_cycle`] against an [`ImprovementCycleBudget`], on the day the
//! runtime's [`CycleClock`] reports (the caller does not choose it), and ends in
//! one [`LedgerEvent::ImprovementCycleRecorded`], completed or aborted. The count of a day is
//! the number of events committed for it plus the cycles admitted and not yet recorded, so an
//! aborted cycle and a cycle still running spend budget like a completed one, and the count is
//! rebuilt from the ledger on replay. A ticket that is dropped without being recorded stays
//! counted until the runtime restarts. A crash between admission and recording is the one way a
//! cycle goes uncounted; the limit is stated in the component's `missing` list.

use crate::{PtrRuntime, RuntimeError};
use ptr_feedback::{BudgetExhausted, CycleOutcomeKind, CycleRecord, ImprovementCycleBudget};
use ptr_ledger::{ImprovementCycleOutcome, LedgerEvent};
use ptr_lineage::{GateViolation, PromotionProposal, SplitLock};
use ptr_types::{CommitIndex, Generation, ProjectId, Validity};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The most `(project, adapter)` bindings the runtime keeps.
pub const MAX_PROMOTED_ADAPTERS: usize = 4096;

/// The most projects whose improvement-cycle day is tracked, and the most (project, day) pairs
/// with an admitted cycle that is not recorded yet.
pub const MAX_TRACKED_CYCLE_PROJECTS: usize = 1024;

/// Why an adapter or improvement-cycle request was refused. The digests are the expected and
/// the actual value of the comparison that failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterError {
    /// The proposal's gate report did not pass or was evaluated without a serving baseline.
    GateNotPassed,
    /// The weights handed to the promotion are empty.
    EmptyArtifact,
    /// The SHA-256 of the weights is not the digest the gate report was bound to.
    ArtifactDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// The frozen splits named are not the ones the gate was evaluated under.
    SplitLockMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// The runtime already keeps [`MAX_PROMOTED_ADAPTERS`] adapters.
    TooManyAdapters { limit: usize },
    /// No generation of this adapter has been promoted.
    UnknownAdapter { project: String, adapter: String },
    /// The adapter generation may not be used now: `validity` is the lifecycle answer
    /// (`None` when the authority knows no such generation, for instance a generation ahead of
    /// the promoted one).
    NotServing {
        project: String,
        adapter: String,
        generation: Generation,
        validity: Option<Validity>,
    },
    /// Adapter or improvement-cycle state exists that no compacted snapshot layout carries.
    CompactionUnsupported,
    /// No improvement budget was configured: no cycle is admitted.
    NoImprovementBudget,
    /// The day's budget is spent.
    BudgetExhausted(BudgetExhausted),
    /// The clock's day is earlier than a day the project already has a cycle admitted or
    /// recorded for (its high-water day). A clock that went back, or a history from the
    /// future, must not hand out a fresh budget.
    ClockRegressed { high_water_day: u64, today: u64 },
    /// The runtime's cycle clock cannot say what day it is, or none was configured.
    ClockUnavailable,
    /// The proposal was evaluated for another project than the one the promotion names.
    ProjectMismatch { expected: String, actual: String },
    /// The runtime already tracks [`MAX_TRACKED_CYCLE_PROJECTS`] projects.
    TooManyCycleProjects { limit: usize },
    /// The policy digest of a cycle is all zero.
    ZeroPolicyDigest,
    /// The ticket was not admitted by this runtime, or its admission has expired or was
    /// already used. Nothing was committed.
    TicketNotAdmitted,
}

/// A promotion of one adapter generation.
///
/// `artifact` is the exact weights the proposal's digest names. `generation` must be above the
/// adapter's current one.
pub struct PromoteAdapter<'a> {
    pub project: ProjectId,
    pub proposal: PromotionProposal,
    /// The frozen evaluation splits the caller says the gate was measured on.
    pub frozen_splits: SplitLock,
    pub generation: Generation,
    pub artifact: &'a [u8],
}

// Not derived: the weights are large and a request is not something to print whole.
impl fmt::Debug for PromoteAdapter<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PromoteAdapter")
            .field("project", &self.project)
            .field("adapter", self.proposal.adapter())
            .field("generation", &self.generation)
            .field("artifact_len", &self.artifact.len())
            .finish_non_exhaustive()
    }
}

/// A revocation of one promoted adapter generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RevokeAdapter {
    pub project: ProjectId,
    pub adapter: String,
    pub generation: Generation,
    pub reason: String,
}

/// What a promoted adapter generation is bound to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromotedAdapter {
    generation: Generation,
    base: [u8; 32],
    artifact_sha256: [u8; 32],
    gate_report_digest: [u8; 32],
}

impl PromotedAdapter {
    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn base(&self) -> [u8; 32] {
        self.base
    }

    pub fn artifact_sha256(&self) -> [u8; 32] {
        self.artifact_sha256
    }

    pub fn gate_report_digest(&self) -> [u8; 32] {
        self.gate_report_digest
    }
}

/// The admission of one improvement cycle. It must be handed back to
/// [`PtrRuntime::record_improvement_cycle`]; the fields are private so only
/// [`PtrRuntime::begin_improvement_cycle`] builds one.
#[derive(Debug, Eq, PartialEq)]
#[must_use = "an admitted cycle must be recorded, or it stays counted as running"]
pub struct ImprovementCycleTicket {
    /// The runtime instance that admitted the cycle, and the admission's serial in it.
    issuer: u64,
    serial: u64,
    project: ProjectId,
    day: u64,
    policy_digest: [u8; 32],
}

/// What day it is, as the runtime's improvement budget counts days: whole days since the Unix
/// epoch. The caller of a cycle does not choose the day; the runtime reads this clock, so a
/// caller cannot count upwards to get a fresh budget or go back to an old one. `None` means the
/// clock cannot be read, and then no cycle is admitted.
pub trait CycleClock: Send + Sync + fmt::Debug {
    fn today(&self) -> Option<u64>;
}

/// Days since the Unix epoch on the system clock; `None` when the clock is before the epoch.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemCycleClock;

impl CycleClock for SystemCycleClock {
    fn today(&self) -> Option<u64> {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        Some(elapsed.as_secs() / 86_400)
    }
}

/// A request to admit one improvement cycle. The day is the runtime clock's.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeginImprovementCycle {
    pub project: ProjectId,
    /// Digest of the improvement policy the cycle runs under.
    pub policy_digest: [u8; 32],
}

/// The latest cycle day of a project and the cycles committed for it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CycleDay {
    day: u64,
    recorded: u64,
}

/// How many days an admitted cycle may stay unrecorded before its admission expires and the
/// ticket can no longer be recorded.
pub const OPEN_CYCLE_RETENTION_DAYS: u64 = 7;

static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

/// The adapter state projected from committed events.
#[derive(Clone, Debug, Default)]
pub(crate) struct AdapterProjection {
    /// Distinguishes this runtime's admissions from another's: a ticket names its issuer.
    instance: u64,
    next_serial: u64,
    promoted: BTreeMap<String, PromotedAdapter>,
    cycle_days: BTreeMap<String, CycleDay>,
    /// Admitted and not yet recorded cycles: serial -> (project, day). Memory only.
    open: BTreeMap<u64, (String, u64)>,
    /// The latest day a project has a cycle admitted or recorded for. A cycle is refused on any
    /// earlier day, so the open counts of earlier days can be dropped without opening a way back.
    high_water: BTreeMap<String, u64>,
    budget: Option<ImprovementCycleBudget>,
    clock: Option<Arc<dyn CycleClock>>,
}

impl AdapterProjection {
    pub(crate) fn new() -> Self {
        Self {
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            ..Self::default()
        }
    }

    /// Whether any adapter or cycle state exists that no snapshot layout carries: promoted
    /// adapters, recorded cycles, high-water days and admitted-but-unrecorded cycles alike.
    pub(crate) fn is_empty(&self) -> bool {
        self.promoted.is_empty()
            && self.cycle_days.is_empty()
            && self.high_water.is_empty()
            && self.open.is_empty()
    }

    fn used(&self, project: &str, day: u64) -> impl Iterator<Item = CycleRecord> {
        let recorded = self
            .cycle_days
            .get(project)
            .filter(|latest| latest.day == day)
            .map_or(0, |latest| latest.recorded);
        let open = self
            .open
            .values()
            .filter(|(open_project, open_day)| open_project == project && *open_day == day)
            .count();
        let completed = CycleRecord {
            day,
            outcome: CycleOutcomeKind::Completed,
        };
        let running = CycleRecord {
            day,
            outcome: CycleOutcomeKind::InFlight,
        };
        (0..recorded)
            .map(move |_| completed)
            .chain((0..open).map(move |_| running))
    }
}

/// The prefix of an adapter's lifecycle key; no new capsule may take it.
pub(crate) const ADAPTER_KEY_PREFIX: &str = "ptr-adapter:";

/// The lifecycle key of an adapter: the project is part of it, and the ledger refuses a `/` in
/// either name, so two pairs never share a key.
pub(crate) fn adapter_key(project: &ProjectId, adapter: &str) -> String {
    format!("{ADAPTER_KEY_PREFIX}{}/{}", project.0, adapter)
}

impl PtrRuntime {
    /// Commit the promotion of an adapter generation.
    ///
    /// # Errors
    /// [`AdapterError::GateNotPassed`] for a report that did not pass or holds a missing
    /// baseline, [`AdapterError::EmptyArtifact`], [`AdapterError::ArtifactDigestMismatch`] when
    /// the weights do not hash to the digest the gate was bound to,
    /// [`AdapterError::SplitLockMismatch`], [`RuntimeError::InvalidLifecycleTransition`] for a
    /// generation that does not advance the adapter or one that is revoked,
    /// [`AdapterError::TooManyAdapters`], and the runtime's own errors when the append is
    /// refused. A refused promotion appends nothing.
    pub fn promote_adapter(
        &mut self,
        request: PromoteAdapter<'_>,
    ) -> Result<CommitIndex, RuntimeError> {
        let PromoteAdapter {
            project,
            proposal,
            frozen_splits,
            generation,
            artifact,
        } = request;
        if proposal.project() != project.0 {
            return Err(AdapterError::ProjectMismatch {
                expected: proposal.project().to_owned(),
                actual: project.0,
            }
            .into());
        }
        let report = proposal.report();
        if !report.passed()
            || report
                .report()
                .violations()
                .iter()
                .any(|violation| matches!(violation, GateViolation::MissingBaseline))
        {
            return Err(AdapterError::GateNotPassed.into());
        }
        if artifact.is_empty() {
            return Err(AdapterError::EmptyArtifact.into());
        }
        let actual: [u8; 32] = Sha256::digest(artifact).into();
        if actual != proposal.artifact_sha256() {
            return Err(AdapterError::ArtifactDigestMismatch {
                expected: proposal.artifact_sha256(),
                actual,
            }
            .into());
        }
        if frozen_splits != proposal.split_lock() {
            return Err(AdapterError::SplitLockMismatch {
                expected: proposal.split_lock().sha256(),
                actual: frozen_splits.sha256(),
            }
            .into());
        }
        self.commit_gated(LedgerEvent::AdapterPromoted {
            project,
            adapter: proposal.adapter().0.clone(),
            generation,
            base: proposal.base_digest(),
            artifact_sha256: actual,
            gate_report_digest: proposal.gate_report_digest(),
        })
    }

    /// Commit the revocation of a promoted adapter generation. The tombstone is permanent.
    ///
    /// # Errors
    /// [`AdapterError::UnknownAdapter`] when no generation of the adapter was promoted,
    /// [`RuntimeError::InvalidLifecycleTransition`] for a generation above the promoted one or
    /// one already revoked, [`RuntimeError::Ledger`] for a malformed name or reason, and the
    /// runtime's own errors when the append is refused.
    pub fn revoke_adapter(&mut self, request: RevokeAdapter) -> Result<CommitIndex, RuntimeError> {
        let RevokeAdapter {
            project,
            adapter,
            generation,
            reason,
        } = request;
        if self
            .live_generation(&adapter_key(&project, &adapter))
            .is_none()
        {
            return Err(AdapterError::UnknownAdapter {
                project: project.0,
                adapter,
            }
            .into());
        }
        self.commit_gated(LedgerEvent::AdapterRevoked {
            project,
            adapter,
            generation,
            reason,
        })
    }

    /// The promoted generation of `adapter` in `project`, if it may serve now: it is promoted
    /// and neither revoked nor superseded. Answered from the committed lifecycle on every
    /// call.
    pub fn serving_adapter(&self, project: &ProjectId, adapter: &str) -> Option<&PromotedAdapter> {
        let key = adapter_key(project, adapter);
        let binding = self.adapters.promoted.get(&key)?;
        (self.generation_validity(&key, binding.generation) == Some(Validity::Live))
            .then_some(binding)
    }

    /// Check one use of `adapter` at `generation` and return its binding. Call it for every
    /// use, not once at admission: a revocation committed since is seen here.
    ///
    /// # Errors
    /// [`AdapterError::UnknownAdapter`] and [`AdapterError::NotServing`] (revoked, superseded
    /// or never promoted at that generation).
    pub fn check_adapter_use(
        &self,
        project: &ProjectId,
        adapter: &str,
        generation: Generation,
    ) -> Result<&PromotedAdapter, AdapterError> {
        let key = adapter_key(project, adapter);
        let Some(binding) = self.adapters.promoted.get(&key) else {
            return Err(AdapterError::UnknownAdapter {
                project: project.0.clone(),
                adapter: adapter.to_owned(),
            });
        };
        let validity = self.generation_validity(&key, generation);
        if validity != Some(Validity::Live) || binding.generation != generation {
            return Err(AdapterError::NotServing {
                project: project.0.clone(),
                adapter: adapter.to_owned(),
                generation,
                validity,
            });
        }
        Ok(binding)
    }

    /// Set the number of improvement cycles admitted per day and the clock that says what day it
    /// is. Host configuration, not history: a runtime rebuilt from its log has neither, and
    /// without them no cycle is admitted.
    pub fn set_improvement_budget(
        &mut self,
        budget: ImprovementCycleBudget,
        clock: Arc<dyn CycleClock>,
    ) {
        self.adapters.budget = Some(budget);
        self.adapters.clock = Some(clock);
    }

    /// Cycles counted against `day` of `project`: committed ones plus admitted ones not yet
    /// recorded.
    pub fn improvement_cycles_used(&self, project: &ProjectId, day: u64) -> u64 {
        u64::try_from(self.adapters.used(&project.0, day).count()).unwrap_or(u64::MAX)
    }

    /// Admit one improvement cycle if the day's budget has room. The returned ticket counts
    /// against the budget until it is recorded.
    ///
    /// # Errors
    /// [`AdapterError::NoImprovementBudget`], [`AdapterError::ZeroPolicyDigest`],
    /// [`AdapterError::ClockRegressed`], [`AdapterError::TooManyCycleProjects`] and
    /// [`AdapterError::BudgetExhausted`]. A refusal admits nothing.
    pub fn begin_improvement_cycle(
        &mut self,
        request: BeginImprovementCycle,
    ) -> Result<ImprovementCycleTicket, RuntimeError> {
        let BeginImprovementCycle {
            project,
            policy_digest,
        } = request;
        let (Some(budget), Some(clock)) = (self.adapters.budget, self.adapters.clock.clone())
        else {
            return Err(AdapterError::NoImprovementBudget.into());
        };
        if policy_digest == [0; 32] {
            return Err(AdapterError::ZeroPolicyDigest.into());
        }
        let Some(day) = clock.today() else {
            return Err(AdapterError::ClockUnavailable.into());
        };
        match self.adapters.high_water.get(&project.0) {
            Some(high_water) if day < *high_water => {
                return Err(AdapterError::ClockRegressed {
                    high_water_day: *high_water,
                    today: day,
                }
                .into());
            }
            None if self.adapters.high_water.len() >= MAX_TRACKED_CYCLE_PROJECTS => {
                return Err(AdapterError::TooManyCycleProjects {
                    limit: MAX_TRACKED_CYCLE_PROJECTS,
                }
                .into());
            }
            _ => {}
        }
        budget
            .check_budget(day, self.adapters.used(&project.0, day))
            .map_err(AdapterError::BudgetExhausted)?;
        // Admissions of this project that stayed unrecorded for longer than the retention expire
        // (they stay counted for the days they belong to, but can no longer be recorded).
        self.adapters.open.retain(|_, (open_project, open_day)| {
            open_project != &project.0 || open_day.saturating_add(OPEN_CYCLE_RETENTION_DAYS) >= day
        });
        // The open table is bounded like the recorded one, and before it grows.
        if self.adapters.open.len() >= MAX_TRACKED_CYCLE_PROJECTS {
            return Err(AdapterError::TooManyCycleProjects {
                limit: MAX_TRACKED_CYCLE_PROJECTS,
            }
            .into());
        }
        self.adapters.high_water.insert(project.0.clone(), day);
        let serial = self.adapters.next_serial;
        self.adapters.next_serial = serial.saturating_add(1);
        self.adapters.open.insert(serial, (project.0.clone(), day));
        Ok(ImprovementCycleTicket {
            issuer: self.adapters.instance,
            serial,
            project,
            day,
            policy_digest,
        })
    }

    /// Record how an admitted cycle ended. A failed append leaves the cycle counted as
    /// running.
    ///
    /// # Errors
    /// [`RuntimeError::Ledger`] for a malformed abort reason and the runtime's own errors when
    /// the append is refused.
    pub fn record_improvement_cycle(
        &mut self,
        ticket: ImprovementCycleTicket,
        outcome: ImprovementCycleOutcome,
    ) -> Result<CommitIndex, RuntimeError> {
        let ImprovementCycleTicket {
            issuer,
            serial,
            project,
            day,
            policy_digest,
        } = ticket;
        // Only this runtime's own, still open admission may be recorded: a ticket of another
        // runtime, an expired one or one already used commits nothing.
        let admitted = issuer == self.adapters.instance
            && self
                .adapters
                .open
                .get(&serial)
                .is_some_and(|(open_project, open_day)| {
                    open_project == &project.0 && *open_day == day
                });
        // An admission left open longer than the retention is expired: judged against the
        // runtime clock here, not only when a later admission happens to sweep the table.
        let expired = self
            .adapters
            .clock
            .as_ref()
            .and_then(|clock| clock.today())
            .is_some_and(|today| day.saturating_add(OPEN_CYCLE_RETENTION_DAYS) < today);
        if expired {
            self.adapters.open.remove(&serial);
        }
        if !admitted || expired {
            return Err(AdapterError::TicketNotAdmitted.into());
        }
        let index = self.commit_gated(LedgerEvent::ImprovementCycleRecorded {
            project,
            day,
            outcome,
            policy_digest,
        })?;
        self.adapters.open.remove(&serial);
        Ok(index)
    }

    /// The replay-time check of an adapter or cycle event, shared by commit and replay.
    pub(crate) fn validate_adapter_event(&self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        ptr_ledger::check_encodable(event)
            .map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        match event {
            LedgerEvent::AdapterPromoted {
                project,
                adapter,
                generation,
                ..
            } => {
                let key = adapter_key(project, adapter);
                if self.capsule_projects.contains_key(&key) {
                    // A capsule of an older ledger owns this key; an adapter never shares it.
                    return Err(RuntimeError::ReservedTargetNamespace { target: key });
                }
                let current = self.live_generation(&key);
                if current.is_none() && self.adapters.promoted.len() >= MAX_PROMOTED_ADAPTERS {
                    return Err(AdapterError::TooManyAdapters {
                        limit: MAX_PROMOTED_ADAPTERS,
                    }
                    .into());
                }
                // A promotion creates a new generation: equal or lower never advances.
                if current.is_some_and(|live| *generation <= live)
                    || self
                        .revoked_generations
                        .contains(&(key.clone(), *generation))
                {
                    return Err(RuntimeError::InvalidLifecycleTransition {
                        target: key,
                        current,
                        requested: *generation,
                    });
                }
                Ok(())
            }
            LedgerEvent::AdapterRevoked {
                project,
                adapter,
                generation,
                ..
            } => {
                let key = adapter_key(project, adapter);
                if self.capsule_projects.contains_key(&key) {
                    return Err(RuntimeError::ReservedTargetNamespace { target: key });
                }
                let current = self.live_generation(&key);
                if current.is_none_or(|live| *generation > live)
                    || self
                        .revoked_generations
                        .contains(&(key.clone(), *generation))
                {
                    return Err(RuntimeError::InvalidLifecycleTransition {
                        target: key,
                        current,
                        requested: *generation,
                    });
                }
                Ok(())
            }
            LedgerEvent::ImprovementCycleRecorded { project, .. } => {
                // The bound of the live admission holds on replay too: a history that names
                // more projects than the runtime tracks is refused, not grown into.
                if !self.adapters.high_water.contains_key(&project.0)
                    && self.adapters.high_water.len() >= MAX_TRACKED_CYCLE_PROJECTS
                {
                    return Err(AdapterError::TooManyCycleProjects {
                        limit: MAX_TRACKED_CYCLE_PROJECTS,
                    }
                    .into());
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Fold a committed adapter or cycle event into the projection.
    pub(crate) fn apply_adapter_event(&mut self, event: &LedgerEvent) {
        match event {
            LedgerEvent::AdapterPromoted {
                project,
                adapter,
                generation,
                base,
                artifact_sha256,
                gate_report_digest,
            } => {
                let key = adapter_key(project, adapter);
                self.set_live_generation(key.clone(), *generation);
                self.adapters.promoted.insert(
                    key,
                    PromotedAdapter {
                        generation: *generation,
                        base: *base,
                        artifact_sha256: *artifact_sha256,
                        gate_report_digest: *gate_report_digest,
                    },
                );
            }
            LedgerEvent::AdapterRevoked {
                project,
                adapter,
                generation,
                ..
            } => {
                self.revoked_generations
                    .insert((adapter_key(project, adapter), *generation));
            }
            LedgerEvent::ImprovementCycleRecorded { project, day, .. } => {
                let high_water = self
                    .adapters
                    .high_water
                    .entry(project.0.clone())
                    .or_insert(0);
                *high_water = (*high_water).max(*day);
                let latest = self
                    .adapters
                    .cycle_days
                    .entry(project.0.clone())
                    .or_default();
                if *day > latest.day {
                    *latest = CycleDay {
                        day: *day,
                        recorded: 1,
                    };
                } else if *day == latest.day {
                    latest.recorded = latest.recorded.saturating_add(1);
                }
            }
            _ => {}
        }
    }
}

impl From<AdapterError> for RuntimeError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(error)
    }
}
