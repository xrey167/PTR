//! The supervisor side of the CUDA staging lifecycle (doc 42 rules 3, 3a and 3b): the guardian
//! process, the staging pool over it, the export registry, and the order in which they and the
//! workers of a [`PodHost`] are ended.
//!
//! `ptr-fabric` holds the mechanisms (`StagingPool`, `ExportRegistry`, `GuardianHost`) and knows
//! no quota ledger; the [`PodHost`] knows workers and quotas; this module is where both are
//! visible, so it holds the rules that join them:
//!
//! * **Workers first, the guardian last.** [`StagingHost::shutdown`] reaps every worker domain,
//!   telling the pool and the registry as each one is gone, and only then ends the guardian,
//!   because a staging buffer freed while a worker may still import it is undefined behaviour.
//!   A worker that cannot be reaped stops the shutdown with the guardian still alive; calling it
//!   again resumes. Each reap is logged with the time it completed and whether the guardian was
//!   still alive.
//! * **A guardian loss ends every worker it backed, and survives a failed reap.**
//!   [`StagingHost::watch`] polls the guardian. A guardian that exited, closed its channel, or
//!   stopped answering (the stub is broken after a timeout: the process is killed first) is
//!   lost. The loss is *stored* (its cause and the workers still to end), because the supervisor
//!   reports a loss only once: every worker that is a producer or consumer of a live buffer is
//!   terminated and reaped through the [`PodHost`] (so its quota charges are released the usual
//!   way), one at a time; a reap that fails returns the error and the next `watch` resumes with
//!   the workers that are left. Only when all of them are gone are the pool and the registry
//!   emptied (the buffers died with the guardian, and no export record may outlive them) and the
//!   guardian's remains reaped, last.
//! * **A release the guardian did not confirm keeps its charges.** When a buffer's release fails
//!   (the guardian does not answer) the pool keeps the charge, since the memory may still be
//!   held; they are released with the rest when the loss path runs.
//! * **Reaps feed the pool.** [`StagingHost::reap_worker`] reaps through the host and then marks
//!   the session reaped in the pool (which may free or release buffers for reuse) and in the
//!   registry (which drops the session's records).
//! * **Recovery is the caller's to start, ours to run.** A producer quota refusal comes with
//!   [`RecoveryPlan`] from the pool; [`StagingHost::recycle`] reaps the producer and the listed
//!   consumers, oldest export first. Respawning replacements is the caller's.
//! * **The caller ends the host.** Dropping a `StagingHost` that still backs workers does not
//!   reap the guardian (a loud message goes to standard error): call [`StagingHost::shutdown`]
//!   first. The guardian's own supervisor still ends its process group when the host is dropped,
//!   which cannot be prevented without leaking it, so drop the [`PodHost`] (and with it the
//!   workers) before the staging host.

use std::time::Instant;

use ptr_fabric::{
    AcquireRequest, AdmittedExport, ExportId, ExportRegistry, FootprintSink, GuardianHost,
    GuardianHostConfig, ImportClaim, Lease, LostCause, PoolLimits, PoolRefusal, ReapReport,
    RecoveryPlan, RegistryRefusal, RemoteBackend, SlotId, StagingPool, SupervisorError,
};
use ptr_runtime::WorkerSession;

use crate::host::{HostError, PodHost, ReapOutcome};

/// What a [`StagingHost`] is built from.
#[derive(Clone, Debug)]
pub struct StagingHostConfig {
    /// The guardian process to start.
    pub guardian: GuardianHostConfig,
    /// The bounds of the staging pool.
    pub limits: PoolLimits,
    /// Most export records live at once.
    pub max_export_records: usize,
    /// Refuse to run with a guardian that the supervisor did not observe in a cgroup and a
    /// network namespace of its own (a host without them). When false the gap is logged as a
    /// [`StagingEvent::GuardianIsolationMissing`] and on standard error.
    pub require_isolation: bool,
}

/// Why the guardian counts as lost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardianLoss {
    /// The process exited or closed its channel.
    Process(LostCause),
    /// It stopped answering within the request timeout; its process was killed.
    Unresponsive,
}

/// What the staging host did, in order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StagingEvent {
    /// The guardian runs without a cgroup and/or network namespace of its own.
    GuardianIsolationMissing {
        cgroup_v2: bool,
        net_namespace: bool,
    },
    /// A worker's domain was reaped and the pool and the registry were told.
    WorkerReaped {
        session: u64,
        /// The guardian process still existed when the worker was reaped.
        guardian_alive: bool,
        /// Buffers the pool freed or released for reuse because of it.
        freed: usize,
        reusable: usize,
    },
    /// The guardian was lost; `terminated` are the workers it backed, all ended.
    GuardianLost {
        cause: GuardianLoss,
        terminated: Vec<u64>,
    },
    /// The buffers and export records that died with the guardian were dropped.
    BuffersAbandoned {
        buffers: usize,
        bytes: u64,
        records: usize,
    },
    /// The guardian's domain was reaped. Always after every worker event of the same
    /// shutdown or loss.
    GuardianReaped,
}

/// An event with the moment it completed.
#[derive(Clone, Debug)]
pub struct LoggedEvent {
    pub at: Instant,
    pub event: StagingEvent,
}

/// Why a staging host call failed.
#[derive(Debug)]
pub enum StagingError {
    Guardian(SupervisorError),
    Pool(PoolRefusal),
    Registry(RegistryRefusal),
    Host(HostError),
    /// The guardian was lost (its loss is being handled or was) or ended; no further staging is
    /// possible.
    GuardianGone,
    /// `require_isolation` was set and the guardian runs without a cgroup or network namespace.
    GuardianNotIsolated {
        cgroup_v2: bool,
        net_namespace: bool,
    },
}

impl std::fmt::Display for StagingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StagingError {}

impl From<SupervisorError> for StagingError {
    fn from(error: SupervisorError) -> Self {
        Self::Guardian(error)
    }
}

impl From<PoolRefusal> for StagingError {
    fn from(error: PoolRefusal) -> Self {
        Self::Pool(error)
    }
}

impl From<RegistryRefusal> for StagingError {
    fn from(error: RegistryRefusal) -> Self {
        Self::Registry(error)
    }
}

impl From<HostError> for StagingError {
    fn from(error: HostError) -> Self {
        Self::Host(error)
    }
}

/// A buffer leased for one handle and the record issued for it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IssuedExport {
    pub lease: Lease,
    pub export: ExportId,
}

/// A guardian loss being handled: what is left to do, kept between calls of `watch`.
struct PendingLoss {
    cause: GuardianLoss,
    /// Backed workers not yet reaped, in the order they are ended.
    remaining: Vec<u64>,
    terminated: Vec<u64>,
    /// The pool and the registry were emptied and the loss logged.
    abandoned: bool,
}

/// The guardian, the pool over it and the export registry of one host.
pub struct StagingHost {
    guardian: GuardianHost,
    pool: StagingPool<RemoteBackend>,
    registry: ExportRegistry,
    events: Vec<LoggedEvent>,
    pending: Option<PendingLoss>,
    ended: bool,
    /// Test hook: fail this many of the next worker reaps (see `inject_reap_failures`).
    reap_faults: u32,
}

fn process_exists(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

impl StagingHost {
    /// Start the guardian and build the pool over it, charging `sink`.
    pub fn spawn(
        config: &StagingHostConfig,
        sink: Box<dyn FootprintSink>,
    ) -> Result<Self, StagingError> {
        let (guardian, backend) = GuardianHost::spawn(&config.guardian)?;
        let mut backend = backend;
        // The guardian answers before anything depends on it.
        backend
            .ping()
            .map_err(|error| StagingError::Pool(PoolRefusal::Backend(error)))?;
        let observation = guardian.spawned().observation.clone();
        let mut host = Self {
            guardian,
            pool: StagingPool::new(backend, sink, config.limits.clone()),
            registry: ExportRegistry::new(config.max_export_records),
            events: Vec::new(),
            pending: None,
            ended: false,
            reap_faults: 0,
        };
        if !observation.cgroup_v2 || !observation.net_namespace {
            eprintln!(
                "ptr-worker: WARNING: the guardian runs without a cgroup of its own ({}) or a \
                 network namespace ({}); its isolation is weaker than configured",
                observation.cgroup_v2, observation.net_namespace
            );
            host.log(StagingEvent::GuardianIsolationMissing {
                cgroup_v2: observation.cgroup_v2,
                net_namespace: observation.net_namespace,
            });
            if config.require_isolation {
                // Dropping `host` ends the guardian: nothing depends on it yet.
                return Err(StagingError::GuardianNotIsolated {
                    cgroup_v2: observation.cgroup_v2,
                    net_namespace: observation.net_namespace,
                });
            }
        }
        Ok(host)
    }

    pub fn guardian(&self) -> &GuardianHost {
        &self.guardian
    }

    pub fn guardian_mut(&mut self) -> &mut GuardianHost {
        &mut self.guardian
    }

    pub fn pool(&self) -> &StagingPool<RemoteBackend> {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut StagingPool<RemoteBackend> {
        &mut self.pool
    }

    pub fn registry(&self) -> &ExportRegistry {
        &self.registry
    }

    /// What happened, oldest first.
    pub fn events(&self) -> &[LoggedEvent] {
        &self.events
    }

    /// Test hook: make the next `count` worker reaps through this host fail with a typed error
    /// before they reach the supervisor, to test that a half-finished loss or shutdown resumes.
    pub fn inject_reap_failures(&mut self, count: u32) {
        self.reap_faults = count;
    }

    fn log(&mut self, event: StagingEvent) {
        self.events.push(LoggedEvent {
            at: Instant::now(),
            event,
        });
    }

    fn reap_through(
        &mut self,
        host: &mut PodHost,
        session: WorkerSession,
    ) -> Result<ReapOutcome, StagingError> {
        if self.reap_faults > 0 {
            self.reap_faults -= 1;
            return Err(StagingError::Host(HostError::Unexpected(
                "injected reap failure",
            )));
        }
        Ok(host.reap(session)?)
    }

    /// Lease a buffer for one handle and issue its export record, valid until
    /// `expires_at_ms` (the operation deadline plus skew). Nothing is leased when the registry
    /// is full or the pool refuses.
    pub fn issue_export(
        &mut self,
        request: &AcquireRequest,
        expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<IssuedExport, StagingError> {
        if self.ended || self.pending.is_some() {
            return Err(StagingError::GuardianGone);
        }
        if !self.registry.has_room(now_ms) {
            return Err(RegistryRefusal::Full {
                max: self.registry.max_records(),
            }
            .into());
        }
        let lease = self.pool.acquire(request)?;
        let export = self
            .registry
            .issue(lease.record, lease.slot, expires_at_ms, now_ms)?;
        Ok(IssuedExport { lease, export })
    }

    /// Admission of a locator: consumes the one live record that matches the claim, or refuses.
    pub fn admit_import(
        &mut self,
        claim: &ImportClaim,
        now_ms: u64,
    ) -> Result<AdmittedExport, StagingError> {
        Ok(self.registry.admit(claim, now_ms)?)
    }

    /// Tell the pool and the registry that `session` was reaped.
    pub fn worker_reaped(&mut self, session: u64) -> ReapReport {
        let report = self.pool.on_reaped(session);
        self.registry.on_reaped(session);
        let guardian_alive = !self.guardian.is_reaped() && process_exists(self.guardian.pid());
        self.log(StagingEvent::WorkerReaped {
            session,
            guardian_alive,
            freed: report.freed.len(),
            reusable: report.became_reusable.len(),
        });
        report
    }

    /// Reap `session` through the host (kill the domain, drain the channel, release quota), then
    /// tell the pool and the registry.
    pub fn reap_worker(
        &mut self,
        host: &mut PodHost,
        session: WorkerSession,
    ) -> Result<ReapOutcome, StagingError> {
        let outcome = self.reap_through(host, session)?;
        self.worker_reaped(session.0);
        Ok(outcome)
    }

    /// Execute a producer-quota recovery plan: reap the consumers it lists, oldest export
    /// first, and then the producer. The caller respawns replacements.
    pub fn recycle(
        &mut self,
        host: &mut PodHost,
        plan: &RecoveryPlan,
    ) -> Result<Vec<ReapOutcome>, StagingError> {
        let mut outcomes = Vec::new();
        for consumer in &plan.consumers {
            outcomes.push(self.reap_worker(host, WorkerSession(*consumer))?);
        }
        outcomes.push(self.reap_worker(host, WorkerSession(plan.producer_session))?);
        Ok(outcomes)
    }

    /// Whether a guardian loss is being handled (a `watch` call left it unfinished).
    pub fn loss_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Poll the guardian and carry a loss forward. A guardian that exited, closed its channel or
    /// stopped answering is lost; this ends every worker it backed, one at a time, then drops
    /// what died with it and reaps its remains. If a step fails the error is returned and the
    /// loss stays pending: call `watch` again and it resumes with what is left. Returns the
    /// events of this call.
    pub fn watch(&mut self, host: &mut PodHost) -> Result<Vec<StagingEvent>, StagingError> {
        if self.ended {
            return Ok(Vec::new());
        }
        let start = self.events.len();
        if self.pending.is_none() {
            let cause = if let Some(cause) = self.guardian.poll() {
                Some(GuardianLoss::Process(cause))
            } else if self.pool.backend().is_broken() {
                // A guardian that does not answer holds buffers nobody can free: end it.
                let _ = self.guardian.kill_process();
                Some(GuardianLoss::Unresponsive)
            } else {
                None
            };
            let Some(cause) = cause else {
                return Ok(Vec::new());
            };
            let backed = self.pool.backed_sessions();
            let remaining = host
                .workers()
                .into_iter()
                .map(|session| session.0)
                .filter(|session| backed.contains(session))
                .collect();
            self.pending = Some(PendingLoss {
                cause,
                remaining,
                terminated: Vec::new(),
                abandoned: false,
            });
        }
        // Every backed worker first; a failure leaves the rest for the next call.
        while let Some(next) = self
            .pending
            .as_ref()
            .and_then(|pending| pending.remaining.first().copied())
        {
            self.reap_through(host, WorkerSession(next))?;
            if let Some(pending) = self.pending.as_mut() {
                pending.remaining.remove(0);
                pending.terminated.push(next);
            }
        }
        // The workers are gone; now the buffers and records that died with the guardian.
        let announce = self.pending.as_ref().and_then(|pending| {
            (!pending.abandoned).then(|| (pending.cause, pending.terminated.clone()))
        });
        if let Some((cause, terminated)) = announce {
            let (buffers, bytes) = self.pool.abandon_all();
            let records = self.registry.clear();
            self.log(StagingEvent::GuardianLost { cause, terminated });
            self.log(StagingEvent::BuffersAbandoned {
                buffers,
                bytes,
                records,
            });
            if let Some(pending) = self.pending.as_mut() {
                pending.abandoned = true;
            }
        }
        self.guardian.reap()?;
        self.pending = None;
        self.ended = true;
        self.log(StagingEvent::GuardianReaped);
        Ok(self.events[start..]
            .iter()
            .map(|logged| logged.event.clone())
            .collect())
    }

    /// End everything: reap every worker of the host (the pool and the registry follow), then
    /// the guardian, last. A worker that cannot be reaped stops the shutdown with the guardian
    /// still alive; call it again to resume. A host whose guardian was lost already only reaps
    /// the workers.
    pub fn shutdown(&mut self, host: &mut PodHost) -> Result<(), StagingError> {
        if self.pending.is_some() {
            self.watch(host)?;
        }
        for session in host.workers() {
            self.reap_worker(host, session)?;
        }
        if !self.ended {
            // Every worker domain is gone, so no import of any buffer remains; what the pool
            // still holds (buffers of sessions the host never knew) goes with the guardian.
            self.pool.abandon_all();
            self.registry.clear();
            self.guardian.reap()?;
            self.ended = true;
            self.log(StagingEvent::GuardianReaped);
        }
        Ok(())
    }

    /// Whether the guardian has been ended.
    pub fn is_ended(&self) -> bool {
        self.ended
    }

    /// The pool's buffer behind `slot`, for the caller that delivers the descriptors.
    pub fn export_descriptor(
        &mut self,
        slot: SlotId,
        record: &ptr_fabric::ExportRecord,
    ) -> Result<std::os::fd::OwnedFd, StagingError> {
        Ok(self.pool.export(slot, record)?)
    }
}

impl Drop for StagingHost {
    /// Dropping without `shutdown` does not reap the guardian while workers may still import
    /// its buffers: it says so loudly and leaves that to the guardian's own supervisor, which
    /// ends the process group when its field is dropped right after this (drop the [`PodHost`]
    /// first so the workers are gone by then). With nothing backed, the guardian is reaped here.
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        let backed = self.pool.backed_sessions().len();
        if backed > 0 {
            eprintln!(
                "ptr-worker: WARNING: a StagingHost was dropped without shutdown while {backed} \
                 session(s) still back staging buffers; the guardian is not reaped here"
            );
            return;
        }
        let _ = self.guardian.reap();
    }
}
