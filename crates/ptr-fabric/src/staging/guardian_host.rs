//! The supervisor's handle on the guardian process (doc 42 rule 3a). Feature
//! `worker-supervisor`, Linux only.
//!
//! The guardian is started by a supervisor of its own, so it gets its own cgroup-v2 domain, and
//! it is **not** seccomp-confined: it is trusted code the supervisor starts from the same image,
//! it holds the device handles that the staging buffers are, and it needs the driver's ioctls
//! and memory-mapping calls that the worker allowlist exists to forbid. What it does not get is
//! worker input: its only descriptor is the channel to the supervisor, and the protocol (see
//! [`guardian`](super::guardian)) has no request that takes one. Capabilities are dropped and
//! `PR_SET_NO_NEW_PRIVS` is set like for a worker, and the default policy gives it a network
//! namespace with no interface; it gets no uid of its own by default, because the device nodes
//! of the GPU belong to the supervisor's user.
//!
//! Lifecycle rules this type carries (the host glue applies them to the worker side):
//!
//! * the guardian is reaped **last**: [`GuardianHost::reap`] is for the moment every worker
//!   domain has been reaped, because freeing a staging buffer while a worker may still import it
//!   is undefined behaviour, and killing the guardian frees all of them;
//! * a guardian that dies on its own is reported by [`GuardianHost::poll`] as lost; the host then
//!   terminates and reaps every worker whose import it backed (a dead guardian's buffers are
//!   gone, and an import of one is dangling), then reaps the guardian's remains.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

use super::guardian::RemoteBackend;
use crate::{
    ConfinePolicy, LostCause, ReapedWorker, SessionId, SpawnRequest, SpawnedWorker, Step,
    Supervisor, SupervisorConfig, SupervisorError, SupervisorEvent,
};

/// What to start as the guardian.
#[derive(Clone, Debug)]
pub struct GuardianHostConfig {
    /// The guardian program (`ptr-worker`, built with the `data-plane` feature).
    pub program: PathBuf,
    /// The arguments after the program name and before the channel flag, for example
    /// `guardian --backend fake`.
    pub args: Vec<OsString>,
    /// The guardian's own supervisor. Its `max_workers` is forced to 1; its uid range must be
    /// disjoint from the worker supervisor's (the default policy takes no uid, so the range is
    /// then unused).
    pub supervisor: SupervisorConfig,
    /// How long a request waits for the guardian's answer.
    pub request_timeout: Duration,
}

impl GuardianHostConfig {
    /// The guardian of `program` as `program guardian`, with the trusted-process policy: its
    /// own cgroup and network namespace where the host has them, no uid of its own.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        let supervisor = SupervisorConfig {
            max_workers: 1,
            policy: ConfinePolicy {
                own_uid: Step::Skip,
                cgroup_v2: Step::IfAvailable,
                net_namespace: Step::IfAvailable,
            },
            ..SupervisorConfig::default()
        };
        Self {
            program: program.into(),
            args: vec![OsString::from("guardian")],
            supervisor,
            request_timeout: Duration::from_secs(10),
        }
    }

    /// Append an argument for the guardian program (`--backend fake`).
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }
}

/// The running guardian process.
pub struct GuardianHost {
    supervisor: Supervisor,
    spawned: SpawnedWorker,
    reaped: bool,
}

impl GuardianHost {
    /// Start the guardian and return it with the stub the staging pool is built over.
    pub fn spawn(config: &GuardianHostConfig) -> Result<(Self, RemoteBackend), SupervisorError> {
        let mut supervisor_config = config.supervisor.clone();
        supervisor_config.max_workers = 1;
        let mut supervisor = Supervisor::new(supervisor_config)?;
        let mut request = SpawnRequest::new(&config.program);
        for arg in &config.args {
            request = request.arg(arg.clone());
        }
        let spawned = supervisor.spawn(&request)?;
        // A second handle on the same socket for the stub: the supervisor keeps its own, for
        // loss detection and for the drain at reap.
        let duplicate = supervisor
            .channel(spawned.session)?
            .try_clone()
            .map_err(SupervisorError::from)?;
        let backend = RemoteBackend::new(duplicate, config.request_timeout);
        Ok((
            Self {
                supervisor,
                spawned,
                reaped: false,
            },
            backend,
        ))
    }

    /// The guardian's session id.
    pub fn session(&self) -> SessionId {
        self.spawned.session
    }

    /// What the supervisor observed when it started the guardian.
    pub fn spawned(&self) -> &SpawnedWorker {
        &self.spawned
    }

    pub fn pid(&self) -> u32 {
        self.spawned.pid
    }

    /// Has the guardian process ended or closed its channel? `Some` once, then `None`.
    pub fn poll(&mut self) -> Option<LostCause> {
        self.supervisor
            .poll()
            .into_iter()
            .map(|SupervisorEvent::Lost { cause, .. }| cause)
            .next()
    }

    /// `SIGKILL` the guardian process only, as a crash (tests; nothing is cleaned up).
    pub fn kill_process(&mut self) -> Result<(), SupervisorError> {
        self.supervisor.kill_process(self.spawned.session)
    }

    /// Terminate the guardian's whole domain and drain its channel. Call it after every worker
    /// domain was reaped, or after the workers a lost guardian backed were. A guardian already
    /// reaped answers `Ok(None)`.
    pub fn reap(&mut self) -> Result<Option<ReapedWorker>, SupervisorError> {
        if self.reaped {
            return Ok(None);
        }
        let reaped = self.supervisor.reap(self.spawned.session)?;
        self.reaped = true;
        Ok(Some(reaped))
    }

    /// Whether the guardian was reaped.
    pub fn is_reaped(&self) -> bool {
        self.reaped
    }
}

impl std::fmt::Debug for GuardianHost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GuardianHost")
            .field("session", &self.spawned.session)
            .field("pid", &self.spawned.pid)
            .field("reaped", &self.reaped)
            .finish()
    }
}
