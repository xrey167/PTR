//! The supervisor side of the Pod data plane (doc 42 stage S1): the glue between the worker
//! processes of `ptr_fabric::Supervisor` and the admission and quota of `ptr-runtime`.
//!
//! `ptr-fabric` supervises processes and knows nothing about handles; `ptr-runtime` decides
//! admission and counts bytes and knows nothing about processes. This module is where both are
//! visible, so it holds the rules that join them:
//!
//! * **Identity comes from the host.** [`Placement`] says which stage a worker serves; the host
//!   generates the Ed25519 key pair, sends the secret as the first frame of the inherited
//!   channel, then challenges the worker (a random nonce signed over `domain | session | nonce`)
//!   and registers the producer with every [`HandleTable`] only after the signature verifies. A
//!   bad signature, a missing answer, or a seccomp filter the kernel does not show makes the
//!   spawn fail and reaps the worker.
//! * **Shared carriers only under full confinement.** [`PodHost::carrier_between`] answers
//!   `HostShm` only if [`carrier_permitted_for`] holds for both the producer and the consumer
//!   (the confinement *and* the seccomp profile the host observed), and
//!   `Bytes` otherwise; [`PodHost::admit`] refuses a shared handle of a worker that does not
//!   qualify.
//! * **The seccomp profile is observed, not believed.** A worker is asked for the CPU profile
//!   unless its [`Placement`] says [`SeccompProfile::Gpu`] (then `--gpu-profile <device>`, which
//!   needs `seccomp` and a device). After the handshake the host classifies the worker's own probe
//!   report ([`observed_profile`]) and requires it to equal the requested profile: a report that
//!   proves neither ([`HostError::ProfileUnproven`]) or the other one
//!   ([`HostError::ProfileMismatch`]) reaps the worker. A worker that cannot start in its
//!   profile (no driver) refuses the challenge with a code ([`HostError::WorkerStartup`]). The
//!   observed profile is kept per worker ([`PodHost::profile`]) and decides the carriers; what a
//!   reduced profile gives up is [`PodHost::profile_weakening`]. The classification is a
//!   self-report of the image the host started, as strong as the forbidden-call check.
//! * **Admission before access.** [`PodHost::admit`] inspects the received `memfd` (all four
//!   seals, size) to take the real footprint, then runs [`HandleTable::admit`]. Only
//!   [`PodHost::consume`] forwards the descriptor, and only to the addressed, admitted consumer
//!   worker, after [`HandleTable::consume`] revalidated.
//! * **`CudaIpc` only through an attached staging host.** The host chooses `CudaIpc` only when
//!   a [`StagingHost`] is passed to the staged calls ([`PodHost::start_produce_staged`],
//!   [`PodHost::admit_staged`], [`PodHost::consume_staged`]), is usable, both workers serve the
//!   guardian's staging import, the placement puts both on one device, and
//!   [`carrier_permitted_for`] holds for both; otherwise the carrier is the one of
//!   [`PodHost::carrier_between`]. The worker never chooses. The host leases the buffer and
//!   issues the export record, hands the producer a descriptor of the buffer (and the tensor in a
//!   sealed `memfd`, so the 64 KiB fill limit of the guardian does not apply), requires the
//!   signed locator to equal its template, claims the record in [`PodHost::admit_staged`]
//!   *before* the table sees the handle, and forwards a fresh descriptor and the record to the
//!   addressed consumer only. The unstaged [`PodHost::admit`] and [`PodHost::consume`] refuse a
//!   `CudaIpc` handle ([`HostError::StagingRequired`]).
//! * **`Consumed` frees nothing.** [`HandleTable::finish`] returns no notice for a shared
//!   carrier; the charge stays until the teardown below.
//! * **Teardown.** A lost worker is fenced ([`HandleTable::unregister_producer`], so its open
//!   handles can no longer be consumed) and later reaped by the supervisor. Only when **both**
//!   workers of a charged pair have been reaped, each with its old channel end drained, does
//!   [`PodHost::reap`] call [`HandleTable::release_after_teardown`] and
//!   [`SharedHostQuota::release_pair`]. [`PodHost::recycle`] reaps the counterparts of a pair so a
//!   pair that holds charges can be released at once.
//!
//! Single-threaded and synchronous: every wait has a deadline and every queue is bounded.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::OsString;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use ptr_fabric::{
    carrier_permitted_for, export_bytes, fill_random, inspect_region, observed_profile,
    AcquireRequest, Channel, ChannelError, Confinement, ExportRecord, FabricError, ForbiddenCall,
    Frame, ImportClaim, Lease, LostCause, Mechanism, ObservedProfile, ReapedWorker, SeccompProfile,
    SessionId, ShmExport, SpawnRequest, Supervisor, SupervisorConfig, SupervisorError,
    SupervisorEvent, WorkerConfinement, REQUIRED_REFUSED_CALLS,
};
use ptr_protocol::control::{decode, encode, CodecError, ControlFrame, IdempotencyKey};
use ptr_protocol::session::{ProtocolSession, SessionConfig, SessionError, SessionEvent};
use ptr_protocol::{
    content_digest, Carrier, CarrierLocator, CudaExport, TensorHandle, TypedPayload,
};
use ptr_runtime::{
    AdmittedHandle, ConsumeRequest, HandleAdmissionRequest, HandlePolicy, HandlePolicyViolation,
    HandleRejection, HandleSource, HandleTable, HostQuotaPolicy, HostQuotaPolicyViolation,
    OperationLiveness, ReclaimNotice, RegistrationRefused, SharedHostQuota, StageLiveness,
    Teardown, VerifiedTeardown, WorkerSession,
};
use ptr_types::ProjectId;

use crate::data_wire::{
    challenge_message, decode_handshake, encode_challenge, wipe, DataReply, DataRequest,
    DataWireError, Handshake, KeyFrame, WireExportRecord, NONCE_BYTES,
};
use crate::guardian::BackendChoice;
use crate::serve::worker_offer;
use crate::staging_host::{IssuedExport, StagingError, StagingHost};
use crate::wire::{WireError, WorkerCall, WorkerReply};

/// Node id the host announces in its Hello.
pub const HOST_NODE_ID: &str = "ptr-pod-host";

/// Most replies buffered for calls other than the one being waited for, per worker.
const MAX_BUFFERED_REPLIES: usize = 16;
/// Most principals of reaped workers remembered for the sweep of their pairs.
const MAX_RETIRED_PRINCIPALS: usize = 4096;

/// What the host is built from.
#[derive(Clone, Debug)]
pub struct HostConfig {
    /// The worker program (`ptr-worker`, built with the `data-plane` feature).
    pub program: PathBuf,
    pub supervisor: SupervisorConfig,
    /// The bounds of every consumer stage's [`HandleTable`].
    pub handle_policy: HandlePolicy,
    /// The host-wide byte ledger shared by all tables.
    pub quota: HostQuotaPolicy,
    /// How long a worker has to answer the key challenge.
    pub key_timeout: Duration,
    /// How long a call waits for its reply.
    pub call_timeout: Duration,
}

impl HostConfig {
    /// Defaults that suit a small host: 64 handles, 64 MiB in flight, 16 MiB per worker.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            supervisor: SupervisorConfig::default(),
            handle_policy: HandlePolicy {
                max_open_handles: 64,
                max_open_per_principal: 32,
                max_tombstones_per_principal: 64,
                max_inflight_bytes_per_principal: 1 << 26,
                max_handle_lifetime_ms: 60_000,
                max_clock_skew_ms: 1_000,
            },
            quota: HostQuotaPolicy::new(1 << 24, 1 << 24, 1 << 26, 256),
            key_timeout: Duration::from_secs(10),
            call_timeout: Duration::from_secs(10),
        }
    }
}

/// One worker to place, from the host's own placement and policy, never from the worker.
#[derive(Clone, Debug)]
pub struct Placement {
    /// The authenticated principal the worker's handles are admitted under.
    pub principal: String,
    /// The one stage identity the worker may serve.
    pub stage_identity: String,
    /// Ask the worker to install the seccomp allowlist (`--confined`).
    pub seccomp: bool,
    /// Further arguments for the worker program (for example `--fault silent` in tests).
    pub extra_args: Vec<OsString>,
    /// The device ordinal the worker's tensors live on. Two workers with the same `Some` device
    /// may exchange a `CudaIpc` handle; `None` (no GPU) never does.
    pub device: Option<u32>,
    /// The staging import the worker serves (`--staging`), which must be the one of the
    /// guardian ([`StagingHost::import`]) for the worker to take part in a `CudaIpc` handle.
    pub staging: Option<BackendChoice>,
    /// The seccomp profile the worker is asked to run: [`SeccompProfile::Cpu`] (the default of
    /// every worker) or the weaker GPU profile, which needs `seccomp` and a `device`. The host
    /// checks it against what the worker's probe shows and records the observed one.
    pub profile: SeccompProfile,
}

/// Why a host call failed. Nothing panics; every refusal is typed.
#[derive(Debug)]
pub enum HostError {
    Supervisor(SupervisorError),
    Channel(ChannelError),
    Wire(DataWireError),
    Pod(WireError),
    Codec(CodecError),
    Session(SessionError),
    Region(FabricError),
    Policy(HandlePolicyViolation),
    QuotaPolicy(HostQuotaPolicyViolation),
    Registration(RegistrationRefused),
    /// The worker did not answer the key challenge in time.
    KeyChallengeTimeout,
    /// The challenge signature does not verify under the key the host generated.
    KeyChallengeBadSignature,
    /// The worker was asked to confine itself and the kernel does not show a filter.
    SeccompNotInForce,
    /// The worker's probe report shows a filter of the other profile than the one requested.
    ProfileMismatch {
        requested: SeccompProfile,
        observed: SeccompProfile,
    },
    /// The worker's probe report shows a filter of neither profile (a forbidden call not
    /// refused, or no profile entry).
    ProfileUnproven {
        requested: SeccompProfile,
    },
    /// The worker could not start as asked and said why instead of signing the challenge, for
    /// example `gpu-no-driver` or `gpu-not-built`.
    WorkerStartup(String),
    /// The placement asks for a profile it cannot get (for example the GPU profile without a
    /// device, or without `seccomp`).
    ProfileMisplaced(&'static str),
    UnknownWorker(WorkerSession),
    /// The worker was lost or fenced.
    WorkerFenced(WorkerSession),
    UnknownStage(String),
    /// The carrier is not permitted for this worker's confinement.
    CarrierNotPermitted {
        carrier: Carrier,
        worker: WorkerSession,
    },
    Rejected(HandleRejection),
    /// A wait passed its deadline.
    Timeout,
    /// The worker refused the call with this code.
    Worker(String),
    /// A reply did not carry the descriptor it should.
    MissingDescriptor,
    /// The staging host refused (the pool, the export registry, the guardian). A `CudaIpc` locator
    /// that finds no live export record for its producer, consumer, operation, device and size is
    /// `Staging(Registry(NoRecord))`.
    Staging(Box<StagingError>),
    /// A `CudaIpc` handle reached a call that has no staging host attached.
    StagingRequired,
    /// The locator of a `CudaIpc` handle is not the VMM form the staging path serves.
    StagedLocator,
    /// A `CudaIpc` produce came without the lease the host made for it.
    MissingStagedExport,
    /// The record the registry handed out belongs to another lease than the handle's: two live
    /// leases of one producer, consumer, operation, device and size were admitted out of order.
    /// The record is spent; the buffers stay leased until the reaps.
    StagedExportMismatch,
    /// A worker failed its handshake or its seccomp/profile check (`cause`) and could not be
    /// reaped even on a retry (`reap`): it is still registered in the supervisor. After a failed
    /// handshake it has no host entry; after a failed check it keeps a fenced one, so nothing
    /// is admitted from or to it and [`PodHost::reap`] can be called again.
    StrandedWorker {
        session: WorkerSession,
        cause: String,
        reap: String,
    },
    /// The worker broke the protocol (malformed or unexpected reply, a carrier or field it may
    /// not choose, a bad region, a lying `Consumed`): it was fenced and reaped.
    Violation {
        worker: WorkerSession,
        what: &'static str,
    },
    Unexpected(&'static str),
}

impl HostError {
    /// Does this error show the worker, not the host or the network, at fault?
    fn is_worker_violation(&self) -> bool {
        matches!(
            self,
            Self::Codec(_)
                | Self::Session(_)
                | Self::Channel(
                    ChannelError::BodyTruncated
                        | ChannelError::ControlTruncated
                        | ChannelError::TooManyDescriptors { .. }
                        | ChannelError::EmptyFrame
                        | ChannelError::FrameTooLarge { .. }
                )
                | Self::Unexpected("too many unclaimed replies")
        )
    }
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for HostError {}

impl From<SupervisorError> for HostError {
    fn from(error: SupervisorError) -> Self {
        Self::Supervisor(error)
    }
}

impl From<ChannelError> for HostError {
    fn from(error: ChannelError) -> Self {
        Self::Channel(error)
    }
}

impl From<CodecError> for HostError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

impl From<SessionError> for HostError {
    fn from(error: SessionError) -> Self {
        Self::Session(error)
    }
}

impl From<StagingError> for HostError {
    fn from(error: StagingError) -> Self {
        match error {
            StagingError::Host(inner) => inner,
            other => Self::Staging(Box::new(other)),
        }
    }
}

impl From<HandleRejection> for HostError {
    fn from(error: HandleRejection) -> Self {
        Self::Rejected(error)
    }
}

/// A worker's place in the host.
struct HostWorker {
    principal: String,
    stage_identity: String,
    key: VerifyingKey,
    device: Option<u32>,
    staging: Option<BackendChoice>,
    /// The seccomp profile the worker's probe showed (CPU until the probe has run).
    profile: SeccompProfile,
    proto: ProtocolSession,
    /// Replies read while waiting for another call's.
    replies: BTreeMap<u64, (TypedPayload, Vec<OwnedFd>)>,
    /// Lost or fenced: no handle of it is admitted or consumed any more.
    fenced: bool,
}

/// A call sent and not yet answered.
#[derive(Debug)]
pub struct PendingCall {
    session: WorkerSession,
    seq: u64,
}

/// A produce call sent and not yet answered.
#[derive(Debug)]
pub struct PendingProduce {
    call: PendingCall,
    carrier: Carrier,
    /// What was sent: the signed handle must equal it in everything but the signature.
    template: TensorHandle,
    /// The lease and export record of a `CudaIpc` produce.
    staged: Option<IssuedExport>,
}

impl PendingProduce {
    /// The carrier the host chose for this tensor.
    pub fn carrier(&self) -> Carrier {
        self.carrier
    }
}

/// What a producer worker is asked to make: the unsigned handle (everything the authority
/// decides: chain, epoch, deadline, nonce, operation, consumer stage, payload type, shape) and
/// the tensor bytes. The host sets the carrier and locator; the worker fills in the digest and
/// signs.
#[derive(Clone, Debug)]
pub struct ProduceSpec {
    pub template: TensorHandle,
    pub tensor: Vec<u8>,
}

/// A signed handle the producer made, with the region that came with it.
#[derive(Debug)]
pub struct Produced {
    pub producer: WorkerSession,
    /// The carrier the host chose (never the worker); the handle was checked to carry it.
    pub carrier: Carrier,
    pub handle: TensorHandle,
    pub encoded: Vec<u8>,
    /// The sealed `memfd` of a `HostShm` handle, held by the host until it is forwarded.
    pub region: Option<OwnedFd>,
    /// The lease and export record of a `CudaIpc` handle: the buffer is the supervisor's, the
    /// worker holds no descriptor to forward, and nothing is released until the pool frees it.
    pub staged: Option<IssuedExport>,
}

/// What the staged calls need besides the call itself: the staging host and the host's clock.
pub struct StagedContext<'a> {
    pub staging: &'a mut StagingHost,
    /// The host's clock, for the expiry of the export record.
    pub now_ms: u64,
}

/// What [`PodHost::admit`] needs besides the handle.
pub struct AdmitRequest<'a> {
    pub produced: Produced,
    /// The consumer worker the handle is addressed to.
    pub consumer: WorkerSession,
    /// The consumer stage, which names the [`HandleTable`].
    pub consumer_stage: &'a str,
    pub now_ms: u64,
    pub operation: u64,
    pub project: &'a ProjectId,
    pub live: &'a dyn StageLiveness,
    pub operations: &'a dyn OperationLiveness,
}

/// An admitted handle waiting to be consumed, with the region that has not been forwarded yet.
#[derive(Debug)]
pub struct Transfer {
    admitted: AdmittedHandle,
    encoded: Vec<u8>,
    region: Option<OwnedFd>,
    carrier: Carrier,
    consumer_stage: String,
    consumer: WorkerSession,
    producer: WorkerSession,
    footprint_bytes: u64,
    /// The leased buffer of a `CudaIpc` handle, whose record admission consumed.
    staged: Option<Lease>,
}

impl Transfer {
    pub fn admitted(&self) -> &AdmittedHandle {
        &self.admitted
    }

    /// The footprint charged to the quota: measured on the region, never read from the handle.
    pub fn footprint_bytes(&self) -> u64 {
        self.footprint_bytes
    }

    pub fn producer(&self) -> WorkerSession {
        self.producer
    }

    pub fn consumer(&self) -> WorkerSession {
        self.consumer
    }
}

/// What the consumer worker answered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsumeOutcome {
    /// The digest of the bytes the consumer mapped and verified.
    pub digest: [u8; 32],
    pub mapped_len: u64,
    /// The notice of a carrier that is not shared: freed at once. Always `None` for a shared
    /// carrier, whose bytes stay charged until the teardown.
    pub reclaimed: Option<ReclaimNotice>,
}

/// What the host noticed on its own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostEvent {
    /// A worker is gone or unreachable and has been fenced; it still has to be reaped.
    Lost {
        session: WorkerSession,
        cause: LostCause,
    },
}

/// What a reap released.
#[derive(Debug)]
pub struct ReapOutcome {
    /// The supervisor's proof; `None` when the supervisor had already reaped the session.
    pub reaped: Option<ReapedWorker>,
    /// The shared buffers the tables released because both workers of their pair are reaped.
    pub notices: Vec<ReclaimNotice>,
    /// Session pairs whose charges were handed back to the ledger.
    pub released_pairs: usize,
    /// Bytes handed back to the ledger, notices included.
    pub released_bytes: u64,
}

/// The supervisor, the handle tables and the shared ledger of one host.
pub struct PodHost {
    config: HostConfig,
    supervisor: Supervisor,
    quota: SharedHostQuota,
    tables: BTreeMap<String, HandleTable>,
    workers: BTreeMap<WorkerSession, HostWorker>,
    /// The principal of each reaped worker, for the sweep of pairs it was part of.
    retired: BTreeMap<WorkerSession, String>,
    retired_order: VecDeque<WorkerSession>,
    call_counter: u64,
    /// Losses noticed by an internal refresh and not yet handed out by [`PodHost::poll`].
    events: Vec<HostEvent>,
}

impl PodHost {
    pub fn new(config: HostConfig) -> Result<Self, HostError> {
        let quota = SharedHostQuota::new(config.quota).map_err(HostError::QuotaPolicy)?;
        config.handle_policy.validate().map_err(HostError::Policy)?;
        let supervisor = Supervisor::new(config.supervisor.clone())?;
        Ok(Self {
            config,
            supervisor,
            quota,
            tables: BTreeMap::new(),
            workers: BTreeMap::new(),
            retired: BTreeMap::new(),
            retired_order: VecDeque::new(),
            call_counter: 0,
            events: Vec::new(),
        })
    }

    pub fn quota(&self) -> &SharedHostQuota {
        &self.quota
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    pub fn supervisor_mut(&mut self) -> &mut Supervisor {
        &mut self.supervisor
    }

    pub fn table(&self, consumer_stage: &str) -> Option<&HandleTable> {
        self.tables.get(consumer_stage)
    }

    /// The workers registered (lost ones until they are reaped).
    pub fn workers(&self) -> Vec<WorkerSession> {
        self.workers.keys().copied().collect()
    }

    /// Whether the host still has the worker: it is dropped only after the supervisor reaped it
    /// and drained its channel.
    pub fn is_registered(&self, session: WorkerSession) -> bool {
        self.workers.contains_key(&session)
    }

    /// Whether the host reaped the worker and still remembers it: registered no more, and in
    /// the bounded list of retired sessions. A session that fell out of that list (or never was
    /// the host's) answers `false`: whoever asks must then free late, never early.
    pub fn is_reaped(&self, session: WorkerSession) -> bool {
        !self.workers.contains_key(&session) && self.retired.contains_key(&session)
    }

    pub fn is_fenced(&self, session: WorkerSession) -> bool {
        self.workers
            .get(&session)
            .is_some_and(|worker| worker.fenced)
    }

    /// The confinement the supervisor observed for the worker.
    /// The seccomp profile the worker was observed to run (not the one requested).
    pub fn profile(&self, session: WorkerSession) -> Result<SeccompProfile, HostError> {
        self.workers
            .get(&session)
            .map(|worker| worker.profile)
            .ok_or(HostError::UnknownWorker(session))
    }

    /// What the observed profile of the worker gives up against the CPU profile, or `None` for a
    /// CPU worker. A host that runs a reduced profile records this where it records its
    /// confinement.
    pub fn profile_weakening(
        &self,
        session: WorkerSession,
    ) -> Result<Option<&'static str>, HostError> {
        Ok(self.profile(session)?.weakening())
    }

    /// The confinement the supervisor observed together with the profile the probe showed: what
    /// [`carrier_permitted_for`] decides on.
    pub fn worker_confinement(
        &self,
        session: WorkerSession,
    ) -> Result<WorkerConfinement, HostError> {
        let profile = self.profile(session)?;
        Ok(WorkerConfinement::new(self.confinement(session)?, profile))
    }

    pub fn confinement(&self, session: WorkerSession) -> Result<Confinement, HostError> {
        Ok(self.supervisor.confinement(SessionId(session.0))?)
    }

    pub fn pid(&self, session: WorkerSession) -> Result<u32, HostError> {
        Ok(self.supervisor.pid(SessionId(session.0))?)
    }

    /// Add the table for handles addressed to `consumer_stage`, sharing the host ledger, and
    /// register every worker already running as a producer in it.
    pub fn add_consumer_stage(&mut self, consumer_stage: &str) -> Result<(), HostError> {
        let mut table = HandleTable::with_host_quota(
            self.config.handle_policy,
            consumer_stage,
            self.quota.clone(),
        )
        .map_err(HostError::Policy)?;
        for (session, worker) in &self.workers {
            if worker.fenced {
                continue;
            }
            let _ = table
                .register_producer(
                    &worker.principal,
                    worker.key,
                    &worker.stage_identity,
                    *session,
                )
                .map_err(HostError::Registration)?;
        }
        self.tables.insert(consumer_stage.to_owned(), table);
        Ok(())
    }

    /// Spawn a worker for `placement`, give it its key, challenge it, check its filter, and
    /// register it with every table. On any failure the worker is reaped, nothing stays
    /// registered, and the error is returned.
    pub fn spawn_worker(&mut self, placement: &Placement) -> Result<WorkerSession, HostError> {
        let mut secret = [0u8; 32];
        fill_random(&mut secret)?;
        let key = SigningKey::from_bytes(&secret);
        let verifying = key.verifying_key();
        let mut request = SpawnRequest::new(&self.config.program).arg("serve");
        if placement.seccomp {
            request = request.arg("--confined");
        }
        match (placement.profile, placement.device, placement.seccomp) {
            (SeccompProfile::Cpu, _, _) => {}
            (SeccompProfile::Gpu, Some(device), true) => {
                request = request.arg("--gpu-profile").arg(device.to_string());
            }
            (SeccompProfile::Gpu, None, _) => {
                wipe(&mut secret);
                return Err(HostError::ProfileMisplaced(
                    "the GPU profile needs a device",
                ));
            }
            (SeccompProfile::Gpu, Some(_), false) => {
                wipe(&mut secret);
                return Err(HostError::ProfileMisplaced(
                    "the GPU profile is a seccomp filter: it needs seccomp",
                ));
            }
        }
        if let Some(staging) = placement.staging {
            request = request.arg("--staging").arg(staging.name());
        }
        request.args.extend(placement.extra_args.iter().cloned());
        let spawned = match self.supervisor.spawn(&request) {
            Ok(spawned) => spawned,
            Err(error) => {
                wipe(&mut secret);
                return Err(error.into());
            }
        };
        let session = WorkerSession(spawned.session.0);
        let established = self.establish(spawned.session, placement, &secret, &verifying);
        wipe(&mut secret);
        let proto = match established {
            Ok(proto) => proto,
            Err(error) => {
                // A worker that failed its handshake is never trusted with anything: kill it
                // and clean up. A reap that fails is retried once; if it fails again the worker
                // is stranded in the supervisor (no host entry), and the error says so.
                return Err(self.reap_stranded(spawned.session, error));
            }
        };
        self.workers.insert(
            session,
            HostWorker {
                principal: placement.principal.clone(),
                stage_identity: placement.stage_identity.clone(),
                key: verifying,
                device: placement.device,
                staging: placement.staging,
                profile: SeccompProfile::Cpu,
                proto,
                replies: BTreeMap::new(),
                fenced: false,
            },
        );
        if placement.seccomp {
            if let Err(error) = self.verify_seccomp(session, placement.profile) {
                // Like a failed handshake: the reap is retried once, and a worker that still
                // cannot be reaped is named in the error instead of being dropped silently.
                let reaped = match self.reap(session) {
                    Ok(_) => Ok(()),
                    Err(_) => self.reap(session).map(|_| ()),
                };
                return Err(match reaped {
                    Ok(()) => error,
                    Err(reap) => HostError::StrandedWorker {
                        session,
                        cause: format!("{error:?}"),
                        reap: format!("{reap:?}"),
                    },
                });
            }
        }
        let stages: Vec<String> = self.tables.keys().cloned().collect();
        let mut registered: Vec<&String> = Vec::new();
        for stage in &stages {
            let Some(table) = self.tables.get_mut(stage) else {
                continue;
            };
            match table.register_producer(
                &placement.principal,
                verifying,
                &placement.stage_identity,
                session,
            ) {
                Ok(_released) => registered.push(stage),
                Err(error) => {
                    // Undo the tables already done, then the worker.
                    for done in registered {
                        if let Some(table) = self.tables.get_mut(done) {
                            let _ =
                                table.unregister_producer_session(&placement.principal, session);
                        }
                    }
                    let _ = self.reap(session);
                    return Err(HostError::Registration(error));
                }
            }
        }
        Ok(session)
    }

    /// Reap a worker that has no host entry (it failed its handshake), retrying once. Returns
    /// `cause` if the worker is gone, else an error naming the session that is still stranded.
    fn reap_stranded(&mut self, session: SessionId, cause: HostError) -> HostError {
        let mut last = self.supervisor.reap(session);
        if matches!(&last, Err(error) if !matches!(error, SupervisorError::UnknownSession(_))) {
            last = self.supervisor.reap(session);
        }
        match last {
            Ok(_) | Err(SupervisorError::UnknownSession(_)) => cause,
            Err(reap) => HostError::StrandedWorker {
                session: WorkerSession(session.0),
                cause: format!("{cause:?}"),
                reap: format!("{reap}"),
            },
        }
    }

    /// After the handshake: ask the worker to run its self-test and let the supervisor accept
    /// the seccomp filter only if the kernel shows one and the report shows every forbidden
    /// call refused. The report is a self-report of the worker image the host started.
    fn verify_seccomp(
        &mut self,
        session: WorkerSession,
        requested: SeccompProfile,
    ) -> Result<(), HostError> {
        let calls = self.probe(session)?;
        let observed = observed_profile(calls.iter().map(|(name, errno)| (name.as_str(), *errno)));
        let report: Vec<ForbiddenCall> = calls
            .iter()
            .filter_map(|(name, errno)| {
                REQUIRED_REFUSED_CALLS
                    .iter()
                    .find(|known| **known == name.as_str())
                    .map(|known| ForbiddenCall {
                        name: known,
                        errno: *errno,
                    })
            })
            .collect();
        let confinement = self
            .supervisor
            .confirm_seccomp(SessionId(session.0), &report)?;
        if confinement.missing().contains(&Mechanism::Seccomp) {
            return Err(HostError::SeccompNotInForce);
        }
        // The profile is what the probe shows, and it must be what was asked for.
        let observed = match observed {
            ObservedProfile::Cpu => SeccompProfile::Cpu,
            ObservedProfile::Gpu => SeccompProfile::Gpu,
            ObservedProfile::Unproven => return Err(HostError::ProfileUnproven { requested }),
        };
        if observed != requested {
            return Err(HostError::ProfileMismatch {
                requested,
                observed,
            });
        }
        if let Some(worker) = self.workers.get_mut(&session) {
            worker.profile = observed;
        }
        Ok(())
    }

    /// The key and the challenge, then the control handshake.
    fn establish(
        &mut self,
        session: SessionId,
        placement: &Placement,
        secret: &[u8; 32],
        verifying: &VerifyingKey,
    ) -> Result<ProtocolSession, HostError> {
        let key_timeout = self.config.key_timeout;
        let call_timeout = self.config.call_timeout;
        {
            let channel = self.supervisor.channel(session)?;
            let key_frame = KeyFrame {
                session: session.0,
                stage_identity: placement.stage_identity.clone(),
                secret: *secret,
            };
            let mut bytes = key_frame.encode().map_err(HostError::Wire)?;
            let sent = channel.try_send(&bytes, &[]);
            wipe(&mut bytes);
            sent?;
            let mut nonce = [0u8; NONCE_BYTES];
            fill_random(&mut nonce)?;
            channel.try_send(&encode_challenge(&nonce), &[])?;
            let frame =
                recv_before(channel, Instant::now() + key_timeout).map_err(
                    |error| match error {
                        HostError::Timeout => HostError::KeyChallengeTimeout,
                        other => other,
                    },
                )?;
            let signature = match decode_handshake(&frame.body).map_err(HostError::Wire)? {
                Handshake::Signed(signature) => signature,
                Handshake::Refused(code) => return Err(HostError::WorkerStartup(code)),
            };
            verifying
                .verify_strict(
                    &challenge_message(session.0, &placement.stage_identity, &nonce),
                    &Signature::from_bytes(&signature),
                )
                .map_err(|_| HostError::KeyChallengeBadSignature)?;
        }
        // The control session, initiated from this side.
        let nonce = session.0 ^ 0x5054_525F_484F_5354;
        let mut proto =
            ProtocolSession::new(SessionConfig::new(HOST_NODE_ID, worker_offer(), nonce))?;
        let channel = self.supervisor.channel(session)?;
        channel.try_send(&encode(&proto.start()?)?, &[])?;
        let deadline = Instant::now() + call_timeout;
        let mut negotiated = false;
        let mut declared = false;
        while !(negotiated && declared) {
            let frame = recv_before(channel, deadline)?;
            let handled = proto.on_frame(decode(&frame.body)?)?;
            for out in &handled.send {
                channel.try_send(&encode(out)?, &[])?;
            }
            match handled.event {
                SessionEvent::Negotiated(_) => negotiated = true,
                SessionEvent::PeerDeclared { .. } => declared = true,
                _ => {}
            }
        }
        Ok(proto)
    }

    /// `HostShm` if the confinement of both workers permits a shared carrier, else `Bytes`.
    pub fn carrier_between(
        &self,
        producer: WorkerSession,
        consumer: WorkerSession,
    ) -> Result<Carrier, HostError> {
        let producer_confinement = self.worker_confinement(producer)?;
        let consumer_confinement = self.worker_confinement(consumer)?;
        if carrier_permitted_for(Carrier::HostShm, &producer_confinement)
            && carrier_permitted_for(Carrier::HostShm, &consumer_confinement)
        {
            Ok(Carrier::HostShm)
        } else {
            Ok(Carrier::Bytes)
        }
    }

    /// The device both workers share when a `CudaIpc` handle may travel between them through
    /// `staging`: it is usable, both workers serve its import, the placement puts both on one
    /// device, and the confinement of both permits the carrier.
    fn staged_device(
        &self,
        producer: WorkerSession,
        consumer: WorkerSession,
        staging: &StagingHost,
    ) -> Result<Option<u32>, HostError> {
        let Some(import) = staging.import() else {
            return Ok(None);
        };
        // REMOVE WHEN the CUDA producer can write through an imported VMM descriptor
        // (`ptr-fabric` has only the guardian's fill today): a `--staging cuda` producer refuses
        // every staged produce with `cuda-producer-write-not-built`, and each attempt would
        // leave a device lease behind. Until then a CUDA import is never chosen.
        if import == BackendChoice::Cuda {
            return Ok(None);
        }
        if !staging.is_usable() {
            return Ok(None);
        }
        let mut shared: Option<u32> = None;
        for (index, session) in [producer, consumer].into_iter().enumerate() {
            let worker = self
                .workers
                .get(&session)
                .ok_or(HostError::UnknownWorker(session))?;
            if worker.fenced || worker.staging != Some(import) {
                return Ok(None);
            }
            let Some(device) = worker.device else {
                return Ok(None);
            };
            if index == 1 && shared != Some(device) {
                return Ok(None);
            }
            shared = Some(device);
            if !carrier_permitted_for(Carrier::CudaIpc, &self.worker_confinement(session)?) {
                return Ok(None);
            }
        }
        Ok(shared)
    }

    /// Like [`Self::carrier_between`] for a host that has a staging host attached: `CudaIpc`
    /// when the placement, the confinement and the staging host allow it, else exactly what
    /// `carrier_between` answers.
    pub fn carrier_between_staged(
        &self,
        producer: WorkerSession,
        consumer: WorkerSession,
        staging: &StagingHost,
    ) -> Result<Carrier, HostError> {
        if self.staged_device(producer, consumer, staging)?.is_some() {
            return Ok(Carrier::CudaIpc);
        }
        self.carrier_between(producer, consumer)
    }

    /// Run a built-in Pod on the worker.
    pub fn call_pod(
        &mut self,
        session: WorkerSession,
        pod: &str,
        input: &[u8],
    ) -> Result<WorkerReply, HostError> {
        let payload = WorkerCall {
            pod: pod.to_owned(),
            input: input.to_vec(),
        }
        .to_payload()
        .map_err(HostError::Pod)?;
        let pending = self.send_call(session, payload, &[])?;
        let (reply, _) = self.wait_reply(&pending)?;
        WorkerReply::from_payload(&reply).map_err(HostError::Pod)
    }

    /// Ask the worker what its filter refuses (see `ptr_fabric::probe_forbidden_calls`).
    pub fn probe(&mut self, session: WorkerSession) -> Result<Vec<(String, i32)>, HostError> {
        let pending = self.send_call(session, DataRequest::Probe.to_payload(), &[])?;
        let (reply, _) = self.wait_reply(&pending)?;
        match self.data_reply(session, &reply)? {
            DataReply::Probed { calls } => Ok(calls),
            DataReply::Refused { code } => Err(HostError::Worker(code)),
            _ => Err(self.refuse(session, "unexpected-reply")),
        }
    }

    /// Send a produce call: the carrier is chosen by [`Self::carrier_between`] (never `CudaIpc`;
    /// that needs [`Self::start_produce_staged`]).
    pub fn start_produce(
        &mut self,
        producer: WorkerSession,
        consumer: WorkerSession,
        spec: ProduceSpec,
    ) -> Result<PendingProduce, HostError> {
        self.start_produce_with(None, producer, consumer, spec)
    }

    /// Like [`Self::start_produce`] with a staging host attached: the carrier is chosen by
    /// [`Self::carrier_between_staged`], so a `CudaIpc` handle is produced when the placement,
    /// the confinement and the staging host allow it, and everything else is as without it.
    pub fn start_produce_staged(
        &mut self,
        context: StagedContext<'_>,
        producer: WorkerSession,
        consumer: WorkerSession,
        spec: ProduceSpec,
    ) -> Result<PendingProduce, HostError> {
        self.start_produce_with(Some(context), producer, consumer, spec)
    }

    fn start_produce_with(
        &mut self,
        staged: Option<StagedContext<'_>>,
        producer: WorkerSession,
        consumer: WorkerSession,
        spec: ProduceSpec,
    ) -> Result<PendingProduce, HostError> {
        if let Some(context) = staged {
            context.staging.sync_reaped(self);
            if let Some(device) = self.staged_device(producer, consumer, context.staging)? {
                return self.start_produce_cuda(context, device, producer, consumer, spec);
            }
        }
        let carrier = self.carrier_between(producer, consumer)?;
        let ProduceSpec {
            mut template,
            tensor,
        } = spec;
        template.carrier = carrier;
        template.locator = match carrier {
            Carrier::HostShm => CarrierLocator::HostShm {
                fd_slot: 0,
                offset: 0,
            },
            _ => export_bytes(&tensor).map_err(HostError::Region)?,
        };
        // The worker recomputes the digest from the bytes it holds; a valid placeholder lets the
        // template encode (an inline locator is checked against it).
        template.content_digest = content_digest(&tensor);
        template.signature = [0; 64];
        let encoded = template
            .encode()
            .map_err(|_| HostError::Unexpected("the template is not a valid handle"))?;
        let template = TensorHandle::decode(&encoded)
            .map_err(|_| HostError::Unexpected("the template does not decode"))?;
        let payload = DataRequest::Produce {
            template: encoded,
            tensor,
        }
        .to_payload();
        let call = self.send_call(producer, payload, &[])?;
        Ok(PendingProduce {
            call,
            carrier,
            template,
            staged: None,
        })
    }

    /// The `CudaIpc` produce (doc 42 rules 3, 3a, 3b): validate the template, lease a staging
    /// buffer and issue its export record, put the real locator (device and the leased
    /// footprint as the allocation size) in the template, and hand the producer two descriptors:
    /// a descriptor of the staging buffer to write through, and a sealed `memfd` holding the
    /// tensor. The tensor therefore is not bounded by the frame size (nor by the guardian's
    /// fill limit); the staging buffer is written by the producer and is never filled by the
    /// guardian.
    ///
    /// A lease that cannot be used after it was made (the descriptor cannot be exported, the
    /// call cannot be sent) stays leased: the pool frees a buffer only when both of its workers
    /// are reaped, and it is bounded by the pool's limits.
    fn start_produce_cuda(
        &mut self,
        context: StagedContext<'_>,
        device: u32,
        producer: WorkerSession,
        consumer: WorkerSession,
        spec: ProduceSpec,
    ) -> Result<PendingProduce, HostError> {
        let StagedContext { staging, now_ms } = context;
        let ProduceSpec {
            mut template,
            tensor,
        } = spec;
        let len = tensor.len() as u64;
        template.carrier = Carrier::CudaIpc;
        template.content_digest = content_digest(&tensor);
        template.signature = [0; 64];
        // Nothing is leased for a template that cannot be a handle: check it with a provisional
        // allocation size (the real one is the leased footprint).
        template.locator = cuda_locator(device, len);
        template
            .encode()
            .map_err(|_| HostError::Unexpected("the template is not a valid handle"))?;
        let source = ShmExport::create(&tensor).map_err(HostError::Region)?;
        let lane = self
            .workers
            .get(&consumer)
            .map(|worker| lane_of(&worker.stage_identity))
            .ok_or(HostError::UnknownWorker(consumer))?;
        let request = AcquireRequest {
            producer_session: producer.0,
            consumer_session: consumer.0,
            consumer_lane: lane,
            operation: template.parent_operation_id,
            device,
            len,
        };
        let expires_at_ms = template
            .signed_absolute_deadline_ms
            .saturating_add(self.config.handle_policy.max_clock_skew_ms);
        let issued = staging.issue_export_for(self, &request, expires_at_ms, now_ms)?;
        template.locator = cuda_locator(device, issued.lease.footprint);
        let encoded = template
            .encode()
            .map_err(|_| HostError::Unexpected("the template is not a valid handle"))?;
        let template = TensorHandle::decode(&encoded)
            .map_err(|_| HostError::Unexpected("the template does not decode"))?;
        let buffer = staging.export_descriptor(issued.lease.slot, &issued.lease.record)?;
        let payload = DataRequest::ProduceStaged { template: encoded }.to_payload();
        let call = self.send_call(producer, payload, &[buffer.as_fd(), source.fd()])?;
        // The host keeps no descriptor of the buffer once the producer has its own.
        drop(buffer);
        drop(source);
        Ok(PendingProduce {
            call,
            carrier: Carrier::CudaIpc,
            template,
            staged: Some(issued),
        })
    }

    /// Wait for the signed handle (and its region) and check what the worker may not decide:
    /// the handle must carry exactly the carrier the host chose, equal the template in every
    /// field but the signature (the worker sets only the signature; the digest and locator it
    /// computes must be the ones the host expects), come with exactly one descriptor for
    /// `HostShm` and none otherwise, and that descriptor must be a sealed `memfd` of exactly the
    /// tensor's size. Anything else is a violation: the descriptor is closed, the worker is
    /// fenced and reaped.
    pub fn finish_produce(&mut self, pending: PendingProduce) -> Result<Produced, HostError> {
        let producer = pending.call.session;
        let (reply, mut descriptors) = self.wait_reply(&pending.call)?;
        let encoded = match self.data_reply(producer, &reply)? {
            DataReply::Produced { handle } => handle,
            DataReply::Refused { code } => return Err(HostError::Worker(code)),
            _ => {
                drop(descriptors);
                return Err(self.refuse(producer, "unexpected-reply"));
            }
        };
        let Ok(handle) = TensorHandle::decode(&encoded) else {
            drop(descriptors);
            return Err(self.refuse(producer, "malformed-handle"));
        };
        let expected_descriptors = usize::from(pending.carrier == Carrier::HostShm);
        if descriptors.len() != expected_descriptors {
            drop(descriptors);
            return Err(self.refuse(producer, "wrong-descriptor-count"));
        }
        if handle.carrier != pending.carrier {
            drop(descriptors);
            return Err(self.refuse(producer, "carrier-swapped"));
        }
        let mut unsigned = handle.clone();
        unsigned.signature = pending.template.signature;
        if unsigned != pending.template {
            drop(descriptors);
            return Err(self.refuse(producer, "template-tampered"));
        }
        let region = if pending.carrier == Carrier::HostShm {
            let descriptor = descriptors.pop();
            match &descriptor {
                Some(fd) => match inspect_region(fd.as_fd()) {
                    Ok(info) if info.byte_len == handle.byte_len => descriptor,
                    _ => {
                        drop(descriptor);
                        return Err(self.refuse(producer, "bad-region"));
                    }
                },
                None => return Err(self.refuse(producer, "wrong-descriptor-count")),
            }
        } else {
            None
        };
        Ok(Produced {
            producer,
            carrier: pending.carrier,
            handle,
            encoded,
            region,
            staged: pending.staged,
        })
    }

    /// Admit a produced handle into the consumer stage's table (doc 42 rule 1). A shared handle
    /// is admitted only if both workers are confined for it, and its footprint is measured on
    /// the region the host holds. A `CudaIpc` handle is refused here
    /// ([`HostError::StagingRequired`]): see [`Self::admit_staged`].
    pub fn admit(&mut self, request: AdmitRequest<'_>) -> Result<Transfer, HostError> {
        self.admit_with(None, request)
    }

    /// Like [`Self::admit`] with the staging host attached. For a `CudaIpc` handle the host
    /// claims the export record first (producer session, consumer session, the handle's
    /// `parent_operation_id`, and the device and allocation size of its locator) and only then
    /// lets the table see the handle, with the record's size as the footprint: a locator that
    /// matches no live record (another consumer, another producer, another size or device, a
    /// replay, an expired or reaped record) is refused as `Staging(Registry(NoRecord))` and the
    /// table is never reached. The record is consumed by the claim; if the table then refuses
    /// the handle (a bad signature, the quota, a replay of the nonce) the leased buffer stays
    /// held until its workers are reaped, which is the pool's rule and bounds it.
    ///
    /// The footprint is charged to the host-wide ledger here, once, by the table; the pool's
    /// own book is separate (see the module documentation of `staging_host`).
    pub fn admit_staged(
        &mut self,
        staging: &mut StagingHost,
        request: AdmitRequest<'_>,
    ) -> Result<Transfer, HostError> {
        self.admit_with(Some(staging), request)
    }

    fn admit_with(
        &mut self,
        staging: Option<&mut StagingHost>,
        request: AdmitRequest<'_>,
    ) -> Result<Transfer, HostError> {
        let AdmitRequest {
            produced,
            consumer,
            consumer_stage,
            now_ms,
            operation,
            project,
            live,
            operations,
        } = request;
        // A worker that died since the last look is fenced now, not when somebody polls.
        self.refresh();
        let mut staging = staging;
        if let Some(staging) = staging.as_deref_mut() {
            staging.sync_reaped(self);
        }
        let producer = produced.producer;
        // The carrier is the host's choice, recorded in `Produced`; the handle must agree, and
        // a region exists only for a shared carrier.
        if produced.handle.carrier != produced.carrier
            || (produced.carrier != Carrier::HostShm && produced.region.is_some())
        {
            return Err(HostError::Unexpected(
                "the produced handle does not match the carrier the host chose",
            ));
        }
        for session in [producer, consumer] {
            match self.workers.get(&session) {
                None => return Err(HostError::UnknownWorker(session)),
                Some(worker) if worker.fenced => return Err(HostError::WorkerFenced(session)),
                Some(_) => {}
            }
            if !carrier_permitted_for(produced.carrier, &self.worker_confinement(session)?) {
                return Err(HostError::CarrierNotPermitted {
                    carrier: produced.carrier,
                    worker: session,
                });
            }
        }
        let mut staged_lease: Option<Lease> = None;
        let footprint_bytes = match (&produced.carrier, &produced.region) {
            (Carrier::HostShm, Some(region)) => {
                let info = inspect_region(region.as_fd()).map_err(HostError::Region)?;
                if info.byte_len != produced.handle.byte_len {
                    return Err(HostError::Region(FabricError::LengthMismatch {
                        expected: produced.handle.byte_len,
                        found: info.byte_len,
                    }));
                }
                info.footprint_bytes
            }
            (Carrier::HostShm, None) => return Err(HostError::MissingDescriptor),
            (Carrier::CudaIpc, _) => {
                let Some(staging) = staging else {
                    return Err(HostError::StagingRequired);
                };
                let issued = produced.staged.ok_or(HostError::MissingStagedExport)?;
                let claim = import_claim(&produced, consumer)?;
                // The claim comes before the table: a locator without a live record never
                // reaches the signature check, the quota or the nonce.
                let admitted = staging.admit_import(&claim, now_ms)?;
                // Records of one key are interchangeable to the registry, which hands out the
                // oldest; the buffer behind this handle must be the one it handed out, so two
                // live leases of one key are admitted in lease order or not at all.
                if admitted.slot != issued.lease.slot || admitted.record != issued.lease.record {
                    return Err(HostError::StagedExportMismatch);
                }
                staged_lease = Some(issued.lease);
                admitted.record.size
            }
            _ => produced.handle.byte_len,
        };
        let principal = self
            .workers
            .get(&producer)
            .map(|worker| worker.principal.clone())
            .ok_or(HostError::UnknownWorker(producer))?;
        let table = self
            .tables
            .get_mut(consumer_stage)
            .ok_or_else(|| HostError::UnknownStage(consumer_stage.to_owned()))?;
        let admitted = table.admit(HandleAdmissionRequest {
            now_ms,
            principal: &principal,
            expected_operation: operation,
            handle: HandleSource::Encoded(&produced.encoded),
            live,
            operations,
            project,
            consumer_session: consumer,
            footprint_bytes,
        })?;
        Ok(Transfer {
            admitted,
            encoded: produced.encoded,
            region: produced.region,
            carrier: produced.carrier,
            consumer_stage: consumer_stage.to_owned(),
            consumer,
            producer,
            footprint_bytes,
            staged: staged_lease,
        })
    }

    /// Use an admitted handle: revalidate and consume it in the table, forward the descriptor to
    /// the addressed consumer worker (and only to it, and only for `HostShm`), wait for its
    /// `Consumed`, check it against the handle's digest and length, finish.
    ///
    /// Workers that died since the last look are fenced first, so a handle of a lost producer
    /// is refused without anybody having to poll. A handle that cannot be used (the consumer is
    /// gone, a fault) is given up or left to [`Self::expire`]; none stays open forever.
    pub fn consume(
        &mut self,
        transfer: Transfer,
        now_ms: u64,
        live: &dyn StageLiveness,
        operations: &dyn OperationLiveness,
    ) -> Result<ConsumeOutcome, HostError> {
        self.consume_inner(None, transfer, now_ms, live, operations, false)
    }

    /// Like [`Self::consume`] with the staging host attached, which a `CudaIpc` transfer needs:
    /// a fresh descriptor of the leased buffer and the export record go to the addressed,
    /// admitted consumer only, which copies the tensor into memory of its own and verifies the
    /// digest on that copy. `Consumed` frees nothing: the buffer is released by the pool after
    /// the reaps of both workers.
    pub fn consume_staged(
        &mut self,
        staging: &mut StagingHost,
        transfer: Transfer,
        now_ms: u64,
        live: &dyn StageLiveness,
        operations: &dyn OperationLiveness,
    ) -> Result<ConsumeOutcome, HostError> {
        self.consume_inner(Some(staging), transfer, now_ms, live, operations, false)
    }

    /// Like [`Self::consume`], but the consumer keeps its mapping after it answers: a consumer
    /// that lies about having let go, to show that `Consumed` frees nothing for a shared
    /// carrier. Only a `fault-injection` worker honours it.
    #[cfg(feature = "fault-injection")]
    pub fn consume_retaining(
        &mut self,
        transfer: Transfer,
        now_ms: u64,
        live: &dyn StageLiveness,
        operations: &dyn OperationLiveness,
    ) -> Result<ConsumeOutcome, HostError> {
        self.consume_inner(None, transfer, now_ms, live, operations, true)
    }

    fn consume_inner(
        &mut self,
        staging: Option<&mut StagingHost>,
        transfer: Transfer,
        now_ms: u64,
        live: &dyn StageLiveness,
        operations: &dyn OperationLiveness,
        retain: bool,
    ) -> Result<ConsumeOutcome, HostError> {
        self.refresh();
        let mut staging = staging;
        if let Some(staging) = staging.as_deref_mut() {
            staging.sync_reaped(self);
        }
        let Transfer {
            admitted,
            encoded,
            region,
            carrier,
            consumer_stage,
            consumer,
            staged,
            ..
        } = transfer;
        let consumer_usable = self
            .workers
            .get(&consumer)
            .is_some_and(|worker| !worker.fenced);
        let table = self
            .tables
            .get_mut(&consumer_stage)
            .ok_or_else(|| HostError::UnknownStage(consumer_stage.clone()))?;
        if !consumer_usable {
            // The consumer is gone: give the handle up (the nonce is tombstoned, the claim
            // stays until the teardown or the expiry).
            let _ = table.abandon(admitted);
            return Err(HostError::WorkerFenced(consumer));
        }
        if carrier == Carrier::CudaIpc && (staging.is_none() || staged.is_none()) {
            let _ = table.abandon(admitted);
            return Err(HostError::StagingRequired);
        }
        // The descriptor comes first: if the pool cannot export it (the consumer's session was
        // reaped) the handle is given up instead of being left consumed and unanswered.
        let mut buffer: Option<OwnedFd> = None;
        if let (Carrier::CudaIpc, Some(staging), Some(lease)) = (carrier, staging, staged) {
            match staging.export_descriptor(lease.slot, &lease.record) {
                Ok(fd) => buffer = Some(fd),
                Err(error) => {
                    let _ = table.abandon(admitted);
                    return Err(error.into());
                }
            }
        }
        let consumed = table.consume(ConsumeRequest {
            now_ms,
            admitted,
            live,
            operations,
        })?;
        let expected_digest = consumed.handle().content_digest;
        let expected_len = consumed.handle().byte_len;
        // A region goes to the consumer only for the shared carrier it belongs to.
        let borrowed: Vec<BorrowedFd<'_>> = match carrier {
            Carrier::HostShm => region.iter().map(AsFd::as_fd).collect(),
            _ => Vec::new(),
        };
        let pending = match (carrier, buffer.as_ref(), staged) {
            (Carrier::CudaIpc, Some(buffer), Some(lease)) => {
                // A fresh descriptor, from the pool, for the session the record names.
                let payload = DataRequest::ConsumeStaged {
                    handle: encoded,
                    record: wire_record(&lease.record),
                }
                .to_payload();
                self.send_call(consumer, payload, &[buffer.as_fd()])?
            }
            _ => self.send_call(
                consumer,
                DataRequest::Consume {
                    handle: encoded,
                    retain,
                }
                .to_payload(),
                &borrowed,
            )?,
        };
        // The host keeps no copy of the descriptor once it is forwarded.
        drop(borrowed);
        drop(region);
        let (reply, _) = self.wait_reply(&pending)?;
        match self.data_reply(consumer, &reply)? {
            DataReply::Consumed { digest, mapped_len } => {
                if digest != expected_digest || mapped_len != expected_len {
                    return Err(self.refuse(consumer, "consumed-does-not-match-the-handle"));
                }
                let reclaimed = self
                    .tables
                    .get_mut(&consumer_stage)
                    .ok_or_else(|| HostError::UnknownStage(consumer_stage.clone()))?
                    .finish(consumed)?;
                Ok(ConsumeOutcome {
                    digest,
                    mapped_len,
                    reclaimed,
                })
            }
            DataReply::Refused { code } => Err(HostError::Worker(code)),
            _ => Err(self.refuse(consumer, "unexpected-reply")),
        }
    }

    /// Hand every table's expired handles to their producers: open or abandoned ones whose
    /// deadline plus skew passed, and consumed ones past their retention. Shared buffers are
    /// not reported here; they leave only with their teardown. Call it with the host's clock
    /// now and then, or abandoned handles pile up against the table bounds.
    pub fn expire(&mut self, now_ms: u64) -> Vec<ReclaimNotice> {
        self.tables
            .values_mut()
            .flat_map(|table| table.expire(now_ms))
            .collect()
    }

    /// Stop trusting a worker: unregister its key from every table (its open handles can no
    /// longer be consumed) and refuse every further admission and call. Nothing is released:
    /// shared bytes stay charged until the worker is reaped.
    pub fn fence(&mut self, session: WorkerSession) {
        let Some(worker) = self.workers.get_mut(&session) else {
            return;
        };
        worker.fenced = true;
        for table in self.tables.values_mut() {
            // Only if the principal is still registered under this very session: after a
            // respawn under a newer session, fencing the old one must leave the new one be.
            // Non-shared buffers come back as notices; a lost worker's tensors travelled as
            // bytes through the host, so there is nobody to hand them to.
            let _notices = table.unregister_producer_session(&worker.principal, session);
        }
    }

    /// Look at the supervisor: fence every worker that has been lost since the last look and
    /// remember it for [`Self::poll`].
    fn refresh(&mut self) {
        let lost: Vec<(WorkerSession, LostCause)> = self
            .supervisor
            .poll()
            .into_iter()
            .map(|event| {
                let SupervisorEvent::Lost { session, cause } = event;
                (WorkerSession(session.0), cause)
            })
            .collect();
        for (session, cause) in lost {
            self.fence(session);
            self.events.push(HostEvent::Lost { session, cause });
        }
    }

    /// Check the supervisor once: every lost worker is fenced and reported (once).
    /// [`Self::admit`] and [`Self::consume`] look too, so fencing never waits for the caller.
    pub fn poll(&mut self) -> Vec<HostEvent> {
        self.refresh();
        std::mem::take(&mut self.events)
    }

    /// A worker broke the protocol: fence it, reap it (best effort; a failed reap leaves it
    /// registered and fenced), and return the violation.
    fn refuse(&mut self, worker: WorkerSession, what: &'static str) -> HostError {
        self.fence(worker);
        let _ = self.reap(worker);
        HostError::Violation { worker, what }
    }

    /// Decode a data reply; a payload that does not decode is a violation by the worker.
    fn data_reply(
        &mut self,
        worker: WorkerSession,
        payload: &TypedPayload,
    ) -> Result<DataReply, HostError> {
        match DataReply::from_payload(payload) {
            Ok(reply) => Ok(reply),
            Err(_) => Err(self.refuse(worker, "malformed-reply")),
        }
    }

    /// Fence, reap (the whole cgroup, the child, the group removed, the old channel drained),
    /// and then, for every charged pair whose two workers are both gone, release the shared
    /// buffers and the pair's charges. A pair whose other worker is still running stays
    /// charged: use [`Self::recycle`] to reap the counterparts too.
    pub fn reap(&mut self, session: WorkerSession) -> Result<ReapOutcome, HostError> {
        self.fence(session);
        // A session the supervisor no longer knows is already reaped: retire the host entry
        // all the same. Any other failure leaves the worker registered, as before.
        let reaped = match self.supervisor.reap(SessionId(session.0)) {
            Ok(reaped) => Some(reaped),
            Err(SupervisorError::UnknownSession(_)) => None,
            Err(error) => return Err(error.into()),
        };
        if let Some(worker) = self.workers.remove(&session) {
            self.retire(session, worker.principal);
        }
        let (notices, released_pairs, released_bytes) = self.sweep();
        Ok(ReapOutcome {
            reaped,
            notices,
            released_pairs,
            released_bytes,
        })
    }

    /// Reap `session` (if it is still registered) and every worker that shares a charged pair
    /// with it, so the pair's charges are released at once (doc 42 rule 3b: the supervisor
    /// recycles the pair). Replacing the workers is the caller's.
    pub fn recycle(&mut self, session: WorkerSession) -> Result<Vec<ReapOutcome>, HostError> {
        let counterparts: BTreeSet<WorkerSession> = self
            .quota
            .tracked_pairs()
            .into_iter()
            .filter_map(|pair| {
                if pair.producer_session == session {
                    Some(pair.consumer_session)
                } else if pair.consumer_session == session {
                    Some(pair.producer_session)
                } else {
                    None
                }
            })
            .filter(|other| *other != session && self.workers.contains_key(other))
            .collect();
        let mut outcomes = Vec::new();
        if self.workers.contains_key(&session) {
            outcomes.push(self.reap(session)?);
        }
        for other in counterparts {
            outcomes.push(self.reap(other)?);
        }
        Ok(outcomes)
    }

    fn retire(&mut self, session: WorkerSession, principal: String) {
        if self.retired.insert(session, principal).is_none() {
            self.retired_order.push_back(session);
        }
        while self.retired_order.len() > MAX_RETIRED_PRINCIPALS {
            if let Some(oldest) = self.retired_order.pop_front() {
                self.retired.remove(&oldest);
            }
        }
    }

    /// Release every charged pair whose two workers are both gone (reaped, channel drained).
    fn sweep(&mut self) -> (Vec<ReclaimNotice>, usize, u64) {
        let mut notices = Vec::new();
        let mut released_pairs = 0usize;
        let mut released_bytes = 0u64;
        for pair in self.quota.tracked_pairs() {
            let (producer, consumer) = (pair.producer_session, pair.consumer_session);
            if self.workers.contains_key(&producer) || self.workers.contains_key(&consumer) {
                continue;
            }
            let Some(principal) = self.retired.get(&producer).cloned() else {
                continue;
            };
            for table in self.tables.values_mut() {
                for notice in table.release_after_teardown(Teardown {
                    principal: &principal,
                    producer_session: producer,
                    consumer_session: consumer,
                }) {
                    released_bytes = released_bytes.saturating_add(notice.charged_bytes);
                    notices.push(notice);
                }
            }
            // Whatever the tables did not hold (a table that was dropped, a charge of a refused
            // slot) goes with the proof of the teardown.
            released_bytes = released_bytes.saturating_add(
                self.quota
                    .release_pair(VerifiedTeardown::reaped_and_drained(producer, consumer)),
            );
            released_pairs += 1;
        }
        (notices, released_pairs, released_bytes)
    }

    // --- calls ----------------------------------------------------------------------------

    fn send_call(
        &mut self,
        session: WorkerSession,
        payload: TypedPayload,
        descriptors: &[BorrowedFd<'_>],
    ) -> Result<PendingCall, HostError> {
        self.call_counter += 1;
        let key = IdempotencyKey::new(format!("host-call-{}", self.call_counter))
            .map_err(|_| HostError::Unexpected("bad idempotency key"))?;
        let Self {
            supervisor,
            workers,
            ..
        } = self;
        let worker = workers
            .get_mut(&session)
            .ok_or(HostError::UnknownWorker(session))?;
        if worker.fenced {
            return Err(HostError::WorkerFenced(session));
        }
        let frame = worker.proto.call(key, payload)?;
        let ControlFrame::Call { seq, .. } = &frame else {
            return Err(HostError::Unexpected(
                "the session built a frame that is no call",
            ));
        };
        let seq = *seq;
        supervisor
            .channel(SessionId(session.0))?
            .try_send(&encode(&frame)?, descriptors)?;
        Ok(PendingCall { session, seq })
    }

    /// Wait for the reply to `pending`. A reply that is malformed, oversized or against the
    /// session rules fences and reaps the worker.
    fn wait_reply(
        &mut self,
        pending: &PendingCall,
    ) -> Result<(TypedPayload, Vec<OwnedFd>), HostError> {
        let result = self.wait_reply_inner(pending);
        if let Err(error) = &result {
            if error.is_worker_violation() {
                self.fence(pending.session);
                let _ = self.reap(pending.session);
            }
        }
        result
    }

    fn wait_reply_inner(
        &mut self,
        pending: &PendingCall,
    ) -> Result<(TypedPayload, Vec<OwnedFd>), HostError> {
        let deadline = Instant::now() + self.config.call_timeout;
        let Self {
            supervisor,
            workers,
            ..
        } = self;
        let worker = workers
            .get_mut(&pending.session)
            .ok_or(HostError::UnknownWorker(pending.session))?;
        let channel = supervisor.channel(SessionId(pending.session.0))?;
        loop {
            if let Some(reply) = worker.replies.remove(&pending.seq) {
                return Ok(reply);
            }
            let Frame { body, descriptors } = recv_before(channel, deadline)?;
            let handled = worker.proto.on_frame(decode(&body)?)?;
            for out in &handled.send {
                channel.try_send(&encode(out)?, &[])?;
            }
            if let SessionEvent::Replied { seq, payload } = handled.event {
                if worker.replies.len() >= MAX_BUFFERED_REPLIES {
                    return Err(HostError::Unexpected("too many unclaimed replies"));
                }
                worker.replies.insert(seq, (payload, descriptors));
            }
        }
    }

    /// Whether a frame from the worker is queued (the next read will not block). Used by tests
    /// that need a reply to be in flight before they act.
    pub fn reply_waiting(
        &self,
        session: WorkerSession,
        timeout: Duration,
    ) -> Result<bool, HostError> {
        Ok(self
            .supervisor
            .channel(SessionId(session.0))?
            .wait_readable(Some(timeout))?)
    }
}

/// The locator of a staged handle: the VMM form with the leased allocation size, the tensor at
/// offset 0 of descriptor slot 0, and no event (nothing produces one yet).
fn cuda_locator(device: u32, alloc_size: u64) -> CarrierLocator {
    CarrierLocator::CudaIpc {
        device,
        export: CudaExport::Vmm {
            fd_slot: 0,
            alloc_size,
        },
        event_handle: [0; 64],
    }
}

/// The stable identity of a consumer stage as the pool's lane: a consumer recycled onto the same
/// stage keeps it, while its session changes.
fn lane_of(stage_identity: &str) -> u64 {
    let digest = content_digest(stage_identity.as_bytes());
    let mut lane = [0u8; 8];
    lane.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(lane)
}

/// What the claim on the export registry says, from facts the host knows: the session the handle
/// came from, the session it is presented by, the operation the handle is signed for, and the
/// device and allocation size of its locator.
fn import_claim(produced: &Produced, consumer: WorkerSession) -> Result<ImportClaim, HostError> {
    let CarrierLocator::CudaIpc {
        device,
        export: CudaExport::Vmm { alloc_size, .. },
        ..
    } = &produced.handle.locator
    else {
        return Err(HostError::StagedLocator);
    };
    Ok(ImportClaim {
        producer_session: produced.producer.0,
        consumer_session: consumer.0,
        operation: produced.handle.parent_operation_id,
        device: *device,
        size: *alloc_size,
    })
}

fn wire_record(record: &ExportRecord) -> WireExportRecord {
    WireExportRecord {
        producer_session: record.producer_session,
        consumer_session: record.consumer_session,
        operation: record.operation,
        device: record.device,
        size: record.size,
    }
}

/// Receive one frame by `deadline`. A closed channel is [`ChannelError::PeerClosed`].
fn recv_before(channel: &Channel, deadline: Instant) -> Result<Frame, HostError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if !channel.wait_readable(Some(remaining))? {
        return Err(HostError::Timeout);
    }
    Ok(channel.try_recv()?)
}
