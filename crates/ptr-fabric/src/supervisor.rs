//! The worker supervisor (doc 42 stage S1): the backend mechanics of spawning, confining,
//! watching and reaping Pod worker processes. Linux only, feature `worker-supervisor`.
//!
//! This module knows processes, cgroups and channels. It knows nothing about handles, ledgers
//! or quotas: the code that releases a quota charge when a worker is reaped lives where both
//! this crate and the runtime are visible (`bins/ptr-worker`), which consumes the
//! [`SupervisorEvent`] and [`ReapedWorker`] values defined here.
//!
//! # Spawn
//!
//! [`Supervisor::spawn`] starts a worker child with one end of a `SOCK_SEQPACKET`
//! [`Channel`] inherited at a fixed descriptor number (3 by default, passed to the program as
//! `--channel-fd 3`). In the forked child, before `exec`, in this order:
//!
//! 1. move into the fresh cgroup-v2 group the supervisor created for this worker (write `0` to
//!    its `cgroup.procs`), so nothing the worker does later escapes the group;
//! 2. `unshare(CLONE_NEWNET)`: a network namespace with no interface but loopback;
//! 3. make the channel end descriptor 3 (`dup2`) and mark every other descriptor from 3 up
//!    close-on-exec (`close_range`), so the worker inherits exactly the channel (standard
//!    input and output are `/dev/null`);
//! 4. drop the bounding capability set and clear the ambient set;
//! 5. `setgroups(0)`, then `setresgid` and `setresuid` to a user and group id of the worker's
//!    own (drawn from a configured range, never shared by two live workers);
//! 6. `capset` to empty: no capability in any set, `CAP_NET_ADMIN` included;
//! 7. `PR_SET_NO_NEW_PRIVS`.
//!
//! The fourth mechanism of the confinement, the seccomp allowlist, is installed by the worker
//! itself as its first action after `exec` ([`install_worker_seccomp`](crate::install_worker_seccomp)):
//! a filter that denies `execve` cannot be installed before the exec. The supervisor does not
//! take the worker's word for any of it: after `spawn` it reads `/proc/<pid>/status`, the
//! group's `cgroup.procs` and the network namespace link and reports the
//! [`Confinement`] it **observed**, `Full` only when all four hold
//! ([`Supervisor::confirm_seccomp`] for the filter, once the worker has had time to install it).
//! A step the policy asks for ([`Step::IfAvailable`], [`Step::Require`]) that the observation
//! does not show makes the spawn fail and the child is killed.
//!
//! # Reap
//!
//! [`Supervisor::reap`] kills the whole cgroup (`cgroup.kill` where the kernel has it, else
//! `SIGKILL` to every pid in `cgroup.procs` until the list is empty), waits for the child,
//! verifies the group is empty, removes it, and only then drains the old channel end
//! ([`drain`]): a worker can send a descriptor just before it dies, and that reference would
//! otherwise survive outside its domain. The returned [`ReapedWorker`] is the supervisor's
//! proof that all of that happened. If any step cannot be completed the worker stays
//! registered and the error is typed; the caller must not treat the worker as reaped.
//!
//! # Bounds and sessions
//!
//! At most [`SupervisorConfig::max_workers`] workers are registered at once. Session ids are a
//! bijective scramble of a counter offset by a random salt drawn at start: within one
//! supervisor no id is issued twice, and a respawned worker always has a new one.
//!
//! Where a mechanism is not available (no delegated cgroup v2, no `CAP_SETUID`, no network
//! namespace) the worker still starts under [`Step::IfAvailable`] and the report says what is
//! missing; [`carrier_permitted`](crate::carrier_permitted) then keeps the shared carriers
//! away from it. A descendant that escapes the group on a host without cgroup v2 cannot be
//! found, which is one more reason such a worker gets no shared carrier.

use std::collections::BTreeMap;
use std::ffi::{c_int, c_ulong, CString, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::os::fd::{AsFd, AsRawFd, RawFd};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use crate::channel::{drain, Channel, ChannelError, DrainEnd, DrainReport};
use crate::confine::{
    own_cgroup_directory, AppliedMechanisms, Confinement, ConfinementProbe, ForbiddenCall,
    Mechanism,
};

/// The descriptor number the worker inherits its channel end at.
pub const DEFAULT_CHANNEL_FD: i32 = 3;

/// The identity of one spawn of a worker, drawn by the supervisor and never reused.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SessionId(pub u64);

impl fmt::Display for SessionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016x}", self.0)
    }
}

/// What the supervisor does about one confinement mechanism it applies itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Step {
    /// Do not apply it.
    Skip,
    /// Apply it where the host can; the report shows whether it was.
    IfAvailable,
    /// Apply it, or refuse to spawn.
    Require,
}

/// Which of the three mechanisms the supervisor applies are wanted. Capability dropping and
/// `PR_SET_NO_NEW_PRIVS` are always applied; seccomp is the worker's own step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfinePolicy {
    pub own_uid: Step,
    pub cgroup_v2: Step,
    pub net_namespace: Step,
}

impl ConfinePolicy {
    /// Every mechanism where the host has it.
    pub const BEST_EFFORT: Self = Self {
        own_uid: Step::IfAvailable,
        cgroup_v2: Step::IfAvailable,
        net_namespace: Step::IfAvailable,
    };

    /// Every mechanism or no worker.
    pub const REQUIRE_ALL: Self = Self {
        own_uid: Step::Require,
        cgroup_v2: Step::Require,
        net_namespace: Step::Require,
    };

    /// No mechanism of the supervisor's (capabilities are still dropped).
    pub const NONE: Self = Self {
        own_uid: Step::Skip,
        cgroup_v2: Step::Skip,
        net_namespace: Step::Skip,
    };
}

/// The bounds and placement of a [`Supervisor`].
#[derive(Clone, Debug)]
pub struct SupervisorConfig {
    /// Workers registered at once, at most; a spawn beyond it is refused.
    pub max_workers: usize,
    pub policy: ConfinePolicy,
    /// The cgroup-v2 directory to create worker groups in. `None`: the supervisor's own group.
    pub cgroup_parent: Option<PathBuf>,
    /// The first user and group id handed to workers; each worker gets `uid_base + n` for a
    /// counter `n` below `uid_span`, starting at a random offset and skipping ids that a live
    /// worker of this supervisor holds. [`Supervisor::new`] rejects a range that contains 0,
    /// the supervisor's own effective uid or `u32::MAX`, that overflows, or that is smaller than
    /// `max_workers`. The supervisor cannot see other supervisors or the host's user database:
    /// every supervisor on a host needs a range disjoint from every other one's.
    pub uid_base: u32,
    pub uid_span: u32,
    /// How long a reap waits for the child to exit and the group to empty.
    pub reap_timeout: Duration,
    /// Let a worker's standard error go to the supervisor's instead of `/dev/null`.
    pub inherit_stderr: bool,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            max_workers: 16,
            policy: ConfinePolicy::BEST_EFFORT,
            cgroup_parent: None,
            uid_base: 2_000_000_000,
            uid_span: 1_000_000,
            reap_timeout: Duration::from_secs(5),
            inherit_stderr: false,
        }
    }
}

/// What to start.
#[derive(Clone, Debug)]
pub struct SpawnRequest {
    pub program: PathBuf,
    /// The arguments before the channel flag.
    pub args: Vec<OsString>,
    /// The worker's whole environment: the supervisor clears the inherited one.
    pub env: Vec<(OsString, OsString)>,
    /// The flag the supervisor appends, followed by the descriptor number. `None` appends
    /// nothing (a program that takes the number from its environment, or a test helper).
    pub channel_fd_flag: Option<OsString>,
    /// The descriptor number the worker inherits the channel at.
    pub channel_fd: i32,
}

impl SpawnRequest {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            channel_fd_flag: Some(OsString::from("--channel-fd")),
            channel_fd: DEFAULT_CHANNEL_FD,
        }
    }

    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }
}

/// What the supervisor found out about a worker process from the kernel, not from the worker.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerObservation {
    /// All four user and group ids equal the id the supervisor assigned.
    pub own_uid: bool,
    /// The pid is in the group's `cgroup.procs`.
    pub cgroup_v2: bool,
    /// The network namespace differs from the supervisor's.
    pub net_namespace: bool,
    /// A seccomp filter is installed (`Seccomp: 2`).
    pub seccomp_filter: bool,
    pub no_new_privs: bool,
    /// The permitted, effective, inheritable and ambient sets are empty. Required.
    pub capabilities_dropped: bool,
    /// The bounding set is empty too. Informational: a supervisor without `CAP_SETPCAP` cannot
    /// empty it, and with an empty permitted set and `no_new_privs` the worker cannot gain a
    /// capability through it either.
    pub bounding_set_empty: bool,
}

/// A worker the supervisor started.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpawnedWorker {
    pub session: SessionId,
    pub pid: u32,
    /// The user and group id assigned, when the own-uid mechanism was applied.
    pub uid: Option<u32>,
    pub cgroup: Option<PathBuf>,
    pub observation: WorkerObservation,
    /// The confinement observed now. Seccomp counts only after
    /// [`Supervisor::confirm_seccomp`].
    pub confinement: Confinement,
}

/// Why a worker was reported lost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LostCause {
    /// The child process exited (or was killed).
    ChildExited(ExitInfo),
    /// The worker's end of the channel is closed.
    ChannelHungUp,
}

/// How a child ended: its exit code, or the signal that killed it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitInfo {
    pub code: Option<i32>,
    pub signal: Option<i32>,
}

impl From<ExitStatus> for ExitInfo {
    fn from(status: ExitStatus) -> Self {
        Self {
            code: status.code(),
            signal: status.signal(),
        }
    }
}

/// Proof that a worker's isolation domain is gone and its old channel end is drained. Only
/// [`Supervisor::reap`] makes one; the ledger glue turns it into a verified teardown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReapedWorker {
    session: SessionId,
    exit: ExitInfo,
    drain_report: DrainReport,
    cgroup_removed: bool,
}

impl ReapedWorker {
    pub(crate) fn new(
        session: SessionId,
        exit: ExitInfo,
        drain_report: DrainReport,
        cgroup_removed: bool,
    ) -> Self {
        Self {
            session,
            exit,
            drain_report,
            cgroup_removed,
        }
    }

    pub fn session(&self) -> SessionId {
        self.session
    }

    pub fn exit(&self) -> ExitInfo {
        self.exit
    }

    /// What the drain of the old channel end received and closed, over all rounds; its `end`
    /// is always [`DrainEnd::PeerClosed`]: only that proves the old end is gone.
    pub fn drain_report(&self) -> DrainReport {
        self.drain_report
    }

    /// The group was empty and removed (`false` for a worker without a cgroup).
    pub fn cgroup_removed(&self) -> bool {
        self.cgroup_removed
    }
}

/// What [`Supervisor::poll`] reports. A completed reap is not an event: [`Supervisor::reap`]
/// returns its [`ReapedWorker`] to the caller that asked for it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SupervisorEvent {
    /// A worker is gone or unreachable. It is still registered, its cgroup and channel are
    /// still there: the caller fences it and then calls [`Supervisor::reap`].
    Lost {
        session: SessionId,
        cause: LostCause,
    },
}

/// Why a supervisor call failed. Nothing panics; every refusal is typed.
#[derive(Debug)]
pub enum SupervisorError {
    /// [`SupervisorConfig::max_workers`] workers are registered.
    TooManyWorkers {
        max: usize,
    },
    UnknownSession(SessionId),
    /// A mechanism the policy requires is not available on this host.
    Unavailable {
        mechanism: Mechanism,
        reason: String,
    },
    /// A mechanism was applied by the child but the kernel does not show it; the worker was
    /// killed.
    NotApplied {
        mechanism: Mechanism,
        detail: String,
    },
    Channel(ChannelError),
    /// The program could not be started.
    Spawn(String),
    /// A cgroup operation failed.
    Cgroup(String),
    /// No random numbers could be drawn.
    Random(String),
    /// A reap step could not be completed; the worker stays registered.
    Reap(ReapFailure),
    /// `/proc/<pid>` could not be read.
    Observe(String),
    /// The configuration or a request is invalid (uid range, descriptor number).
    Config(String),
}

/// Which reap step failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReapFailure {
    /// The child did not exit within the reap timeout.
    ChildNotExited,
    /// The group still held processes after the timeout.
    CgroupNotEmpty { remaining: usize },
    /// The empty group could not be removed.
    CgroupRemove(String),
    /// The drain failed with an operating-system error.
    Drain(ChannelError),
    /// The drain reached its cap with the peer still sending: the queue is not emptied.
    DrainIncomplete(DrainReport),
    /// The queue was emptied but the old end never showed as closed within the timeout (a
    /// process outside the group still holds the worker's end): the end is not proven gone.
    DrainNotClosed(DrainReport),
}

impl fmt::Display for SupervisorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyWorkers { max } => write!(formatter, "already {max} workers"),
            Self::UnknownSession(session) => write!(formatter, "no worker with session {session}"),
            Self::Unavailable { mechanism, reason } => {
                write!(
                    formatter,
                    "{mechanism:?} is required but unavailable: {reason}"
                )
            }
            Self::NotApplied { mechanism, detail } => {
                write!(formatter, "{mechanism:?} was not applied: {detail}")
            }
            Self::Channel(error) => write!(formatter, "channel: {error}"),
            Self::Spawn(message) => write!(formatter, "spawn failed: {message}"),
            Self::Cgroup(message) => write!(formatter, "cgroup: {message}"),
            Self::Random(message) => write!(formatter, "random: {message}"),
            Self::Reap(failure) => write!(formatter, "reap incomplete: {failure:?}"),
            Self::Observe(message) => write!(formatter, "observe: {message}"),
            Self::Config(message) => write!(formatter, "configuration: {message}"),
        }
    }
}

impl std::error::Error for SupervisorError {}

impl From<ChannelError> for SupervisorError {
    fn from(error: ChannelError) -> Self {
        Self::Channel(error)
    }
}

// --- randomness ---------------------------------------------------------------------------

/// Fill `buffer` from the kernel's random number generator (`getrandom`), for keys, nonces and
/// session ids.
pub fn fill_random(buffer: &mut [u8]) -> Result<(), SupervisorError> {
    let mut filled = 0usize;
    while filled < buffer.len() {
        // SAFETY: the pointer and length describe the unfilled tail of `buffer`, which is
        // valid for writes and outlives the call.
        let got = unsafe {
            libc::getrandom(
                buffer[filled..].as_mut_ptr().cast(),
                buffer.len() - filled,
                0,
            )
        };
        if got < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(SupervisorError::Random(error.to_string()));
        }
        filled += got as usize;
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

// --- the supervisor -----------------------------------------------------------------------

struct WorkerRecord {
    child: Child,
    channel: Channel,
    /// The group, until it has been removed.
    cgroup: Option<PathBuf>,
    /// The group existed and has been removed by a reap.
    cgroup_removed: bool,
    uid: Option<u32>,
    applied: AppliedMechanisms,
    lost_reported: bool,
}

/// Spawns, watches and reaps worker processes. See the module documentation.
pub struct Supervisor {
    config: SupervisorConfig,
    probe: ConfinementProbe,
    workers: BTreeMap<SessionId, WorkerRecord>,
    salt: u64,
    counter: u64,
    uid_counter: u32,
}

impl Supervisor {
    /// A supervisor with `config`; probes the host once.
    pub fn new(config: SupervisorConfig) -> Result<Self, SupervisorError> {
        // SAFETY: `geteuid` takes no arguments and cannot fail.
        let euid = unsafe { libc::geteuid() };
        validate_config(&config, euid)?;
        let mut random = [0u8; 12];
        fill_random(&mut random)?;
        let mut salt = [0u8; 8];
        salt.copy_from_slice(&random[..8]);
        let mut offset = [0u8; 4];
        offset.copy_from_slice(&random[8..]);
        Ok(Self {
            config,
            probe: ConfinementProbe::detect(),
            workers: BTreeMap::new(),
            salt: u64::from_le_bytes(salt),
            counter: 0,
            // A random start, so two runs do not hand the same ids to different workers.
            uid_counter: u32::from_le_bytes(offset),
        })
    }

    /// What the host can provide, as probed at start.
    pub fn probe(&self) -> &ConfinementProbe {
        &self.probe
    }

    pub fn config(&self) -> &SupervisorConfig {
        &self.config
    }

    /// The registered workers, lost ones included, until they are reaped.
    pub fn sessions(&self) -> Vec<SessionId> {
        self.workers.keys().copied().collect()
    }

    pub fn is_registered(&self, session: SessionId) -> bool {
        self.workers.contains_key(&session)
    }

    /// The supervisor's end of the worker's channel.
    pub fn channel(&self, session: SessionId) -> Result<&Channel, SupervisorError> {
        self.workers
            .get(&session)
            .map(|record| &record.channel)
            .ok_or(SupervisorError::UnknownSession(session))
    }

    pub fn pid(&self, session: SessionId) -> Result<u32, SupervisorError> {
        self.workers
            .get(&session)
            .map(|record| record.child.id())
            .ok_or(SupervisorError::UnknownSession(session))
    }

    /// The user and group id assigned to the worker, when it has one of its own.
    pub fn uid_of(&self, session: SessionId) -> Result<Option<u32>, SupervisorError> {
        self.workers
            .get(&session)
            .map(|record| record.uid)
            .ok_or(SupervisorError::UnknownSession(session))
    }

    /// The cgroup directory of the worker, if it has one.
    pub fn cgroup_of(&self, session: SessionId) -> Result<Option<&Path>, SupervisorError> {
        self.workers
            .get(&session)
            .map(|record| record.cgroup.as_deref())
            .ok_or(SupervisorError::UnknownSession(session))
    }

    /// The confinement observed for the worker.
    pub fn confinement(&self, session: SessionId) -> Result<Confinement, SupervisorError> {
        self.workers
            .get(&session)
            .map(|record| Confinement::from_applied(record.applied))
            .ok_or(SupervisorError::UnknownSession(session))
    }

    /// Send `SIGKILL` to the worker's process only (no wait, no cleanup): a worker that
    /// crashes. [`poll`](Self::poll) then reports it lost and [`reap`](Self::reap) cleans up.
    pub fn kill_process(&mut self, session: SessionId) -> Result<(), SupervisorError> {
        let record = self
            .workers
            .get_mut(&session)
            .ok_or(SupervisorError::UnknownSession(session))?;
        match record.child.kill() {
            Ok(()) => Ok(()),
            // Already gone.
            Err(error) if error.kind() == io::ErrorKind::InvalidInput => Ok(()),
            Err(error) => Err(SupervisorError::Spawn(format!("kill: {error}"))),
        }
    }

    fn next_session(&mut self) -> SessionId {
        self.counter = self.counter.wrapping_add(1);
        SessionId(splitmix64(self.salt.wrapping_add(self.counter)))
    }

    /// The next id of the range that no live worker of this supervisor holds.
    fn next_uid(&mut self) -> Option<u32> {
        let span = self.config.uid_span.max(1);
        for _ in 0..span.min(1 << 20) {
            let candidate = self.config.uid_base.saturating_add(self.uid_counter % span);
            self.uid_counter = self.uid_counter.wrapping_add(1);
            if !self
                .workers
                .values()
                .any(|record| record.uid == Some(candidate))
            {
                return Some(candidate);
            }
        }
        None
    }

    /// Start a worker. See the module documentation for what the child does before `exec`.
    pub fn spawn(&mut self, request: &SpawnRequest) -> Result<SpawnedWorker, SupervisorError> {
        if request.channel_fd < 3 {
            return Err(SupervisorError::Config(format!(
                "the channel descriptor must be 3 or higher, not {}",
                request.channel_fd
            )));
        }
        if self.workers.len() >= self.config.max_workers {
            return Err(SupervisorError::TooManyWorkers {
                max: self.config.max_workers,
            });
        }
        let attempt_uid = decide(
            self.config.policy.own_uid,
            Mechanism::OwnUid,
            &self.probe.own_uid,
        )?;
        let attempt_cgroup = decide(
            self.config.policy.cgroup_v2,
            Mechanism::CgroupV2,
            &self.probe.cgroup_v2,
        )?;
        let attempt_net = decide(
            self.config.policy.net_namespace,
            Mechanism::NetNamespace,
            &self.probe.net_namespace,
        )?;
        let session = self.next_session();
        let (ours, theirs) = Channel::pair()?;
        let uid = if attempt_uid {
            Some(self.next_uid().ok_or(SupervisorError::TooManyWorkers {
                max: self.config.max_workers,
            })?)
        } else {
            None
        };
        let cgroup = if attempt_cgroup {
            Some(self.create_cgroup(session)?)
        } else {
            None
        };
        let cap_last = fs::read_to_string("/proc/sys/kernel/cap_last_cap")
            .ok()
            .and_then(|text| text.trim().parse::<c_int>().ok())
            .unwrap_or(40);
        let cgroup_procs = match &cgroup {
            Some(directory) => {
                match CString::new(
                    directory
                        .join("cgroup.procs")
                        .as_os_str()
                        .as_encoded_bytes(),
                ) {
                    Ok(path) => Some(path),
                    Err(_) => {
                        remove_cgroup_quietly(cgroup.as_deref());
                        return Err(SupervisorError::Cgroup("path holds a NUL".to_owned()));
                    }
                }
            }
            None => None,
        };
        let plan = ChildPlan {
            channel_fd: theirs.as_fd().as_raw_fd(),
            target_fd: request.channel_fd,
            cgroup_procs,
            net_namespace: attempt_net,
            ids: uid.map(|id| (id, id)),
            cap_last,
        };
        let mut command = Command::new(&request.program);
        command.args(&request.args);
        if let Some(flag) = &request.channel_fd_flag {
            command.arg(flag).arg(request.channel_fd.to_string());
        }
        command
            .env_clear()
            .envs(request.env.iter().map(|(key, value)| (key, value)))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(if self.config.inherit_stderr {
                Stdio::inherit()
            } else {
                Stdio::null()
            });
        // SAFETY: the closure runs in the forked child before `exec` and calls only
        // async-signal-safe functions of `libc` on data prepared before the fork (no
        // allocation, no locks); see `child_setup`.
        unsafe {
            command.pre_exec(move || child_setup(&plan));
        }
        let spawned = command.spawn();
        // The child has its own copy now (or the spawn failed); the supervisor never keeps the
        // worker's end.
        drop(theirs);
        let mut child = match spawned {
            Ok(child) => child,
            Err(error) => {
                remove_cgroup_quietly(cgroup.as_deref());
                return Err(SupervisorError::Spawn(error.to_string()));
            }
        };
        let observation = match observe(child.id(), cgroup.as_deref(), uid) {
            Ok(observation) => observation,
            Err(error) => {
                abort_spawn(&mut child, cgroup.as_deref());
                return Err(error);
            }
        };
        let missing = [
            (attempt_uid, observation.own_uid, Mechanism::OwnUid),
            (attempt_cgroup, observation.cgroup_v2, Mechanism::CgroupV2),
            (
                attempt_net,
                observation.net_namespace,
                Mechanism::NetNamespace,
            ),
        ]
        .into_iter()
        .find(|(attempted, seen, _)| *attempted && !*seen);
        if let Some((_, _, mechanism)) = missing {
            abort_spawn(&mut child, cgroup.as_deref());
            return Err(SupervisorError::NotApplied {
                mechanism,
                detail: format!("{observation:?}"),
            });
        }
        if !(observation.no_new_privs && observation.capabilities_dropped) {
            abort_spawn(&mut child, cgroup.as_deref());
            return Err(SupervisorError::NotApplied {
                mechanism: Mechanism::OwnUid,
                detail: format!("capabilities or no_new_privs not in force: {observation:?}"),
            });
        }
        let applied = AppliedMechanisms {
            seccomp: observation.seccomp_filter,
            own_uid: observation.own_uid,
            cgroup_v2: observation.cgroup_v2,
            net_namespace: observation.net_namespace,
        };
        let pid = child.id();
        self.workers.insert(
            session,
            WorkerRecord {
                child,
                channel: ours,
                cgroup: cgroup.clone(),
                cgroup_removed: false,
                uid,
                applied,
                lost_reported: false,
            },
        );
        Ok(SpawnedWorker {
            session,
            pid,
            uid,
            cgroup,
            observation,
            confinement: Confinement::from_applied(applied),
        })
    }

    fn create_cgroup(&self, session: SessionId) -> Result<PathBuf, SupervisorError> {
        let parent = match &self.config.cgroup_parent {
            Some(parent) => parent.clone(),
            None => own_cgroup_directory().ok_or_else(|| {
                SupervisorError::Cgroup("no cgroup v2 directory for this process".to_owned())
            })?,
        };
        let directory = parent.join(format!("ptr-worker-{session}"));
        fs::create_dir(&directory).map_err(|error| {
            SupervisorError::Cgroup(format!("{}: {error}", directory.display()))
        })?;
        Ok(directory)
    }

    /// Record whether the worker's seccomp allowlist is in force, and return the confinement
    /// observed now. Three things must hold: the kernel shows a filter on the process
    /// (`Seccomp: 2`, read from `/proc/<pid>/status`), `NoNewPrivs` is set, and `report` (the
    /// worker's own [`probe_forbidden_calls`](crate::probe_forbidden_calls), run after it
    /// installed the filter) shows `EPERM` for every call in [`REQUIRED_REFUSED_CALLS`]. The
    /// status line alone would accept any filter, an allow-all one included. The report is a
    /// self-report of a binary the supervisor started itself (a trusted image), not a kernel
    /// proof: it catches a wrong or empty filter, not a malicious worker. The worker installs
    /// the filter as its first action, so call this after it answered its first frame.
    pub fn confirm_seccomp(
        &mut self,
        session: SessionId,
        report: &[ForbiddenCall],
    ) -> Result<Confinement, SupervisorError> {
        let record = self
            .workers
            .get_mut(&session)
            .ok_or(SupervisorError::UnknownSession(session))?;
        let status = read_status(record.child.id())?;
        record.applied.seccomp = status_field(&status, "Seccomp:") == Some("2")
            && status_field(&status, "NoNewPrivs:") == Some("1")
            && report_proves_filter(report);
        Ok(Confinement::from_applied(record.applied))
    }

    /// Check every registered worker once: a child that exited, or a channel the worker closed,
    /// is reported lost (once). Nothing is reaped here.
    pub fn poll(&mut self) -> Vec<SupervisorEvent> {
        let mut events = Vec::new();
        for (session, record) in &mut self.workers {
            if record.lost_reported {
                continue;
            }
            let cause = match record.child.try_wait() {
                Ok(Some(status)) => Some(LostCause::ChildExited(status.into())),
                Ok(None) if record.channel.peer_hung_up() => Some(LostCause::ChannelHungUp),
                Ok(None) | Err(_) => None,
            };
            if let Some(cause) = cause {
                record.lost_reported = true;
                events.push(SupervisorEvent::Lost {
                    session: *session,
                    cause,
                });
            }
        }
        events
    }

    /// Terminate the worker's whole domain, wait for it, verify it is gone, remove its group,
    /// and drain the old channel end, in that order. The returned [`ReapedWorker`] is the proof
    /// that the shared bytes of this worker can be released. On failure the worker stays
    /// registered (the caller may retry) and must not be treated as reaped.
    pub fn reap(&mut self, session: SessionId) -> Result<ReapedWorker, SupervisorError> {
        let timeout = self.config.reap_timeout;
        let record = self
            .workers
            .get_mut(&session)
            .ok_or(SupervisorError::UnknownSession(session))?;
        let deadline = Instant::now() + timeout;
        if let Some(directory) = &record.cgroup {
            kill_cgroup(directory);
        }
        // The group does not cover a worker started without one, so the process itself is
        // always signalled too (an error means it is already gone).
        let _ = record.child.kill();
        let status = loop {
            match record.child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL_STEP),
                Ok(None) => return Err(SupervisorError::Reap(ReapFailure::ChildNotExited)),
                Err(error) => return Err(SupervisorError::Spawn(format!("wait: {error}"))),
            }
        };
        if let Some(directory) = record.cgroup.clone() {
            loop {
                // A group that is already gone holds nothing.
                let remaining = match cgroup_pids(&directory) {
                    Ok(pids) => pids,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
                    Err(error) => {
                        return Err(SupervisorError::Cgroup(format!(
                            "{}: {error}",
                            directory.display()
                        )))
                    }
                };
                if remaining.is_empty() {
                    break;
                }
                if Instant::now() >= deadline {
                    return Err(SupervisorError::Reap(ReapFailure::CgroupNotEmpty {
                        remaining: remaining.len(),
                    }));
                }
                for pid in remaining {
                    kill_pid(pid);
                }
                std::thread::sleep(POLL_STEP);
            }
            remove_cgroup_until(&directory, deadline).map_err(|error| {
                SupervisorError::Reap(ReapFailure::CgroupRemove(error.to_string()))
            })?;
            // Done for good: a retry of a later step must not look at the group again.
            record.cgroup = None;
            record.cgroup_removed = true;
        }
        // The old end is proven gone only when the queue is empty AND the end shows closed.
        // `Empty` means the peer still holds its end, so wait for it until the deadline.
        let mut drained = DrainReport::default();
        loop {
            let report = drain(&record.channel)
                .map_err(|error| SupervisorError::Reap(ReapFailure::Drain(error)))?;
            drained.frames += report.frames;
            drained.empty_frames += report.empty_frames;
            drained.descriptors += report.descriptors;
            drained.end = report.end;
            match report.end {
                DrainEnd::PeerClosed => break,
                DrainEnd::Capped => {
                    return Err(SupervisorError::Reap(ReapFailure::DrainIncomplete(drained)))
                }
                DrainEnd::Empty if Instant::now() >= deadline => {
                    return Err(SupervisorError::Reap(ReapFailure::DrainNotClosed(drained)))
                }
                DrainEnd::Empty => std::thread::sleep(POLL_STEP),
            }
        }
        let cgroup_removed = record.cgroup_removed;
        self.workers.remove(&session);
        Ok(ReapedWorker::new(
            session,
            status.into(),
            drained,
            cgroup_removed,
        ))
    }
}

impl Drop for Supervisor {
    /// A supervisor that goes away takes its workers with it: kill each group and process,
    /// wait, remove the groups. The queues are not drained (there is no one left to release
    /// the charges for).
    fn drop(&mut self) {
        for record in self.workers.values_mut() {
            if let Some(directory) = &record.cgroup {
                kill_cgroup(directory);
            }
            let _ = record.child.kill();
            let _ = record.child.wait();
            if let Some(directory) = &record.cgroup {
                let deadline = Instant::now() + Duration::from_secs(2);
                while !cgroup_pids(directory)
                    .map(|pids| pids.is_empty())
                    .unwrap_or(true)
                    && Instant::now() < deadline
                {
                    kill_cgroup(directory);
                    std::thread::sleep(POLL_STEP);
                }
                let _ = remove_cgroup_until(directory, deadline);
            }
        }
    }
}

const POLL_STEP: Duration = Duration::from_millis(2);

/// The calls a worker's self-test must show refused with `EPERM` before its seccomp filter
/// counts: a filter that lets any of them through is not the allowlist.
pub const REQUIRED_REFUSED_CALLS: [&str; 7] = [
    "socket",
    "socketpair",
    "fork",
    "open",
    "io_uring_setup",
    "process_vm_readv",
    "pidfd_open",
];

/// Does `report` show `EPERM` for every call in [`REQUIRED_REFUSED_CALLS`]?
fn report_proves_filter(report: &[ForbiddenCall]) -> bool {
    REQUIRED_REFUSED_CALLS.iter().all(|name| {
        report
            .iter()
            .any(|call| call.name == *name && call.errno == libc::EPERM)
    })
}

fn validate_config(config: &SupervisorConfig, euid: u32) -> Result<(), SupervisorError> {
    let bad = |message: String| Err(SupervisorError::Config(message));
    if config.max_workers == 0 {
        return bad("max_workers must be at least 1".to_owned());
    }
    if config.uid_base == 0 {
        return bad("the uid range must not contain 0 (root)".to_owned());
    }
    if config.uid_span == 0 || (config.uid_span as usize) < config.max_workers {
        return bad(format!(
            "uid_span {} is smaller than max_workers {}",
            config.uid_span, config.max_workers
        ));
    }
    let Some(last) = config.uid_base.checked_add(config.uid_span - 1) else {
        return bad("the uid range overflows".to_owned());
    };
    // `(uid_t) -1` means "no change" to setresuid, so it is never an id.
    if last == u32::MAX {
        return bad("the uid range must not contain u32::MAX".to_owned());
    }
    if (config.uid_base..=last).contains(&euid) {
        return bad(format!(
            "the uid range {}..={last} contains the supervisor's own uid {euid}",
            config.uid_base
        ));
    }
    Ok(())
}

fn decide(
    step: Step,
    mechanism: Mechanism,
    availability: &crate::confine::Availability,
) -> Result<bool, SupervisorError> {
    match (step, availability) {
        (Step::Skip, _) => Ok(false),
        (Step::IfAvailable, availability) => Ok(availability.is_available()),
        (Step::Require, crate::confine::Availability::Available) => Ok(true),
        (Step::Require, crate::confine::Availability::Unavailable(reason)) => {
            Err(SupervisorError::Unavailable {
                mechanism,
                reason: reason.clone(),
            })
        }
    }
}

// --- the child ----------------------------------------------------------------------------

/// Everything the forked child needs, prepared before the fork so the child allocates nothing.
struct ChildPlan {
    channel_fd: RawFd,
    target_fd: RawFd,
    cgroup_procs: Option<CString>,
    net_namespace: bool,
    ids: Option<(u32, u32)>,
    cap_last: c_int,
}

const PR_CAP_AMBIENT: c_int = 47;
const PR_CAP_AMBIENT_CLEAR_ALL: c_ulong = 4;
const LINUX_CAPABILITY_VERSION_3: u32 = 0x2008_0522;
const CLOSE_RANGE_CLOEXEC: c_ulong = 4;

#[repr(C)]
struct CapHeader {
    version: u32,
    pid: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CapData {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}

fn last_os_error() -> io::Error {
    io::Error::last_os_error()
}

fn check(result: c_int) -> io::Result<()> {
    if result == -1 {
        Err(last_os_error())
    } else {
        Ok(())
    }
}

/// Mark descriptors `first..=last` close-on-exec, with `close_range` where the kernel has it
/// and one `fcntl` per descriptor otherwise.
///
/// # Safety
///
/// Async-signal-safe: raw system calls only.
unsafe fn cloexec_range(first: c_ulong, last: c_ulong) {
    if first > last {
        return;
    }
    // SAFETY: `close_range` with `CLOSE_RANGE_CLOEXEC` only sets a flag on descriptors.
    let result = unsafe { libc::syscall(libc::SYS_close_range, first, last, CLOSE_RANGE_CLOEXEC) };
    if result == 0 {
        return;
    }
    // Older kernel: walk a bounded range.
    let top = last.min(4096);
    let mut fd = first;
    while fd <= top {
        // SAFETY: `fcntl(F_GETFD/F_SETFD)` on a number that may not be open fails harmlessly.
        unsafe {
            let flags = libc::fcntl(fd as c_int, libc::F_GETFD);
            if flags >= 0 {
                libc::fcntl(fd as c_int, libc::F_SETFD, flags | libc::FD_CLOEXEC);
            }
        }
        fd += 1;
    }
}

/// The child's side of the spawn, in the order the module documentation lists.
fn child_setup(plan: &ChildPlan) -> io::Result<()> {
    // SAFETY: this runs between `fork` and `exec` in the child, so it may call only
    // async-signal-safe functions and must not allocate; every call below is a plain system
    // call on integers or on the NUL-terminated path prepared before the fork, and the only
    // error value built is `io::Error::from_raw_os_error` (no allocation).
    unsafe {
        // 1. The cgroup, first: from here on everything the worker does is inside the group.
        if let Some(path) = &plan.cgroup_procs {
            let fd = libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC);
            check(fd)?;
            let written = libc::write(fd, b"0".as_ptr().cast(), 1);
            let outcome = if written == 1 {
                Ok(())
            } else {
                Err(last_os_error())
            };
            libc::close(fd);
            outcome?;
        }
        // 2. A network namespace of its own. The second `unshare` is the fallback for a
        //    supervisor that may not create a plain network namespace. It fails closed: in the
        //    new user namespace this process has no uid map, so when an own uid is asked for,
        //    `setresuid` below fails with EINVAL, the spawn fails, and nothing is accepted;
        //    where no own uid is asked for, the namespace is real and `observe` still checks
        //    that the network namespace differs from the supervisor's before it is reported.
        if plan.net_namespace
            && libc::unshare(libc::CLONE_NEWNET) != 0
            && libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNET) != 0
        {
            return Err(last_os_error());
        }
        // 3. The channel at the agreed number; everything else closes at exec.
        if plan.channel_fd == plan.target_fd {
            check(libc::fcntl(plan.target_fd, libc::F_SETFD, 0))?;
        } else {
            check(libc::dup2(plan.channel_fd, plan.target_fd))?;
        }
        let target = plan.target_fd as c_ulong;
        cloexec_range(3, target.saturating_sub(1));
        cloexec_range(target + 1, c_ulong::from(u32::MAX));
        // 4. The bounding set, then the ambient set. A cap the process cannot drop (no
        //    `CAP_SETPCAP`) is dropped by the capset below or was never held.
        let mut cap = 0;
        while cap <= plan.cap_last {
            libc::prctl(
                libc::PR_CAPBSET_DROP,
                cap as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
                0 as c_ulong,
            );
            cap += 1;
        }
        libc::prctl(
            PR_CAP_AMBIENT,
            PR_CAP_AMBIENT_CLEAR_ALL,
            0 as c_ulong,
            0 as c_ulong,
            0 as c_ulong,
        );
        // 5. Own user and group id; no supplementary groups.
        if let Some((uid, gid)) = plan.ids {
            check(libc::setgroups(0, std::ptr::null()))?;
            check(libc::setresgid(gid, gid, gid))?;
            check(libc::setresuid(uid, uid, uid))?;
        }
        // 6. No capability in any set.
        let header = CapHeader {
            version: LINUX_CAPABILITY_VERSION_3,
            pid: 0,
        };
        let data = [CapData {
            effective: 0,
            permitted: 0,
            inheritable: 0,
        }; 2];
        let result = libc::syscall(libc::SYS_capset, &header as *const CapHeader, data.as_ptr());
        if result != 0 {
            return Err(last_os_error());
        }
        // 7. Never gain privilege through an exec.
        check(libc::prctl(
            libc::PR_SET_NO_NEW_PRIVS,
            1 as c_ulong,
            0 as c_ulong,
            0 as c_ulong,
            0 as c_ulong,
        ))?;
    }
    Ok(())
}

// --- observation --------------------------------------------------------------------------

fn read_status(pid: u32) -> Result<String, SupervisorError> {
    fs::read_to_string(format!("/proc/{pid}/status"))
        .map_err(|error| SupervisorError::Observe(format!("/proc/{pid}/status: {error}")))
}

fn status_field<'a>(status: &'a str, name: &str) -> Option<&'a str> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(name))
        .map(str::trim)
}

fn all_ids_equal(field: Option<&str>, expected: u32) -> bool {
    field.is_some_and(|text| {
        let ids: Vec<&str> = text.split_whitespace().collect();
        ids.len() == 4
            && ids
                .iter()
                .all(|id| id.parse::<u32>().ok() == Some(expected))
    })
}

fn all_zero(field: Option<&str>) -> bool {
    field.is_some_and(|text| u64::from_str_radix(text.trim(), 16) == Ok(0))
}

fn observe(
    pid: u32,
    cgroup: Option<&Path>,
    uid: Option<u32>,
) -> Result<WorkerObservation, SupervisorError> {
    let status = read_status(pid)?;
    let own_uid = uid.is_some_and(|id| {
        all_ids_equal(status_field(&status, "Uid:"), id)
            && all_ids_equal(status_field(&status, "Gid:"), id)
    });
    let in_cgroup = cgroup.is_some_and(|directory| {
        cgroup_pids(directory)
            .map(|pids| pids.contains(&(pid as i32)))
            .unwrap_or(false)
    });
    let ours = fs::read_link("/proc/self/ns/net");
    let theirs = fs::read_link(format!("/proc/{pid}/ns/net"));
    let net_namespace = matches!((&ours, &theirs), (Ok(a), Ok(b)) if a != b);
    let capabilities_dropped = ["CapInh:", "CapPrm:", "CapEff:", "CapAmb:"]
        .into_iter()
        .all(|name| all_zero(status_field(&status, name)));
    Ok(WorkerObservation {
        own_uid,
        cgroup_v2: in_cgroup,
        net_namespace,
        seccomp_filter: status_field(&status, "Seccomp:") == Some("2"),
        no_new_privs: status_field(&status, "NoNewPrivs:") == Some("1"),
        capabilities_dropped,
        bounding_set_empty: all_zero(status_field(&status, "CapBnd:")),
    })
}

// --- cgroup helpers -----------------------------------------------------------------------

fn cgroup_pids(directory: &Path) -> io::Result<Vec<i32>> {
    let text = fs::read_to_string(directory.join("cgroup.procs"))?;
    Ok(text
        .lines()
        .filter_map(|line| line.trim().parse::<i32>().ok())
        .collect())
}

fn kill_pid(pid: i32) {
    // SAFETY: `kill` with a pid and a signal number takes plain integers.
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}

/// Kill every process of the group: `cgroup.kill` where the kernel has it (5.14), else
/// `SIGKILL` to each listed pid (the caller repeats until the list is empty).
fn kill_cgroup(directory: &Path) {
    if fs::write(directory.join("cgroup.kill"), "1").is_ok() {
        return;
    }
    if let Ok(pids) = cgroup_pids(directory) {
        for pid in pids {
            kill_pid(pid);
        }
    }
}

/// Remove the (empty) group, retrying while the kernel still counts a just-exited task, until
/// `deadline`. A group that is already gone is removed.
fn remove_cgroup_until(directory: &Path, deadline: Instant) -> io::Result<()> {
    loop {
        match fs::remove_dir(directory) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(_) if Instant::now() < deadline => std::thread::sleep(POLL_STEP),
            Err(error) => return Err(error),
        }
    }
}

fn remove_cgroup_quietly(directory: Option<&Path>) {
    if let Some(directory) = directory {
        let _ = remove_cgroup_until(directory, Instant::now() + Duration::from_secs(2));
    }
}

/// Undo a spawn whose confinement did not check out: kill the group and the child, wait, and
/// remove the group.
fn abort_spawn(child: &mut Child, cgroup: Option<&Path>) {
    if let Some(directory) = cgroup {
        kill_cgroup(directory);
    }
    let _ = child.kill();
    let _ = child.wait();
    remove_cgroup_quietly(cgroup);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_distinct_and_follow_the_salt() {
        let mut first = Supervisor::new(SupervisorConfig::default()).expect("supervisor");
        let ids: std::collections::BTreeSet<u64> =
            (0..10_000).map(|_| first.next_session().0).collect();
        assert_eq!(ids.len(), 10_000);
        let second = Supervisor::new(SupervisorConfig::default()).expect("supervisor");
        assert_ne!(first.salt, second.salt);
    }

    #[test]
    fn the_status_fields_are_read() {
        let status = "Uid:\t7\t7\t7\t7\nCapEff:\t0000000000000000\nSeccomp:\t2\n";
        assert!(all_ids_equal(status_field(status, "Uid:"), 7));
        assert!(!all_ids_equal(status_field(status, "Uid:"), 8));
        assert!(all_zero(status_field(status, "CapEff:")));
        assert_eq!(status_field(status, "Seccomp:"), Some("2"));
        assert!(!all_ids_equal(None, 0));
        assert!(!all_ids_equal(Some("7 7 7"), 7));
    }

    #[test]
    fn uids_stay_in_the_configured_range() {
        let mut supervisor = Supervisor::new(SupervisorConfig {
            uid_base: 5000,
            uid_span: 3,
            max_workers: 3,
            ..SupervisorConfig::default()
        })
        .expect("supervisor");
        let uids: Vec<u32> = (0..7)
            .map(|_| supervisor.next_uid().expect("a free id"))
            .collect();
        assert!(
            uids.iter().all(|uid| (5000..5003).contains(uid)),
            "{uids:?}"
        );
        // Consecutive draws walk the range: each is the previous plus one, wrapping.
        for pair in uids.windows(2) {
            assert_eq!((pair[0] - 5000 + 1) % 3, pair[1] - 5000, "{uids:?}");
        }
    }

    #[test]
    fn the_uid_range_is_validated() {
        let config = |base, span, workers| SupervisorConfig {
            uid_base: base,
            uid_span: span,
            max_workers: workers,
            ..SupervisorConfig::default()
        };
        let refused = |config: SupervisorConfig, euid| {
            matches!(
                validate_config(&config, euid),
                Err(SupervisorError::Config(_))
            )
        };
        assert!(!refused(config(1000, 16, 16), 5));
        assert!(refused(config(0, 16, 16), 5), "root");
        assert!(refused(config(1000, 8, 16), 5), "smaller than max_workers");
        assert!(refused(config(1000, 0, 16), 5), "empty");
        assert!(refused(config(u32::MAX - 3, 16, 16), 5), "overflow");
        assert!(
            refused(config(u32::MAX - 15, 16, 16), 5),
            "contains u32::MAX"
        );
        assert!(refused(config(1000, 16, 16), 1005), "contains the euid");
        assert!(
            !refused(config(u32::MAX - 16, 16, 16), 5),
            "ends at u32::MAX - 1"
        );
        assert!(refused(config(1000, 16, 0), 5), "no workers");
    }

    #[test]
    fn a_live_workers_uid_is_skipped() {
        let mut supervisor = Supervisor::new(SupervisorConfig {
            uid_base: 7000,
            uid_span: 2,
            max_workers: 2,
            ..SupervisorConfig::default()
        })
        .expect("supervisor");
        let first = supervisor.next_uid().expect("first");
        supervisor.workers.insert(
            SessionId(1),
            WorkerRecord {
                child: Command::new("/bin/sleep").arg("30").spawn().expect("child"),
                channel: Channel::pair().expect("pair").0,
                cgroup: None,
                cgroup_removed: false,
                uid: Some(first),
                applied: AppliedMechanisms::default(),
                lost_reported: false,
            },
        );
        for _ in 0..6 {
            assert_ne!(supervisor.next_uid(), Some(first));
        }
    }

    #[test]
    fn only_an_all_eperm_report_proves_the_filter() {
        let all = |errno| -> Vec<ForbiddenCall> {
            REQUIRED_REFUSED_CALLS
                .iter()
                .map(|name| ForbiddenCall { name, errno })
                .collect()
        };
        assert!(report_proves_filter(&all(libc::EPERM)));
        assert!(!report_proves_filter(&all(0)), "an allow-all filter");
        assert!(!report_proves_filter(&all(libc::EACCES)));
        assert!(!report_proves_filter(&[]), "no report");
        let mut partial = all(libc::EPERM);
        partial.pop();
        assert!(!report_proves_filter(&partial));
    }

    #[test]
    fn a_required_mechanism_that_is_missing_refuses_the_spawn() {
        use crate::confine::Availability;
        let missing = Availability::Unavailable("none".to_owned());
        assert!(matches!(
            decide(Step::Require, Mechanism::CgroupV2, &missing),
            Err(SupervisorError::Unavailable { .. })
        ));
        assert_eq!(
            decide(Step::IfAvailable, Mechanism::CgroupV2, &missing).ok(),
            Some(false)
        );
        assert_eq!(
            decide(Step::Skip, Mechanism::CgroupV2, &Availability::Available).ok(),
            Some(false)
        );
    }
}
