//! A Pod gateway and a requester, both over `ALPN_PODWIRE`.
//!
//! Neither is a daemon. `serve_once` accepts one connection and answers it;
//! `request` sends one request and checks one answer. A caller decides when either
//! happens, which keeps the tests free of sleeps and keeps "how long do we wait"
//! and "how often do we accept" where they can be decided deliberately.
use crate::access::{answer, answer_bound, PodAccessPolicy};
use crate::frame::{
    decode_answer, decode_answer_v2, decode_request, decode_request_v2, encode_answer,
    encode_answer_v2, encode_request, encode_request_v2, request_digest, FrameError,
    PodAddressBinding, PodAdmissionBinding, PodAnswer, PodAnswerV2, PodOutcome, PodRequest,
    PodRequestV2, RefusalCode, MAX_FRAME_BYTES,
};
use ptr_net::{EndpointAddr, IrohSession, IrohTransport, NodeIdentity, PeerAddress, ALPN_PODWIRE};
use ptr_pods::ProtocolBinding;
use ptr_pods::{ExecutionManifest, PodManifest, PodRegistry, RegisteredPodBinding};
use ptr_protocol::TypedPayload;
use ptr_runtime::RuntimeExecutionManifestResolver;
use ptr_types::{
    ArtifactId, Digest, Generation, MeshEndpointBinding, NodeId, PodId, ProjectId, Revision,
    ScopeId, SessionId, StatefulRequestRecovery, Timestamp, UncertainRequest,
};
use ptr_verifier::Verifier;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// How long one exchange waits for an answer before giving up.
///
/// A default rather than a rule: the right deadline depends on the deployment, and
/// what matters is that there is one. Without it a host that stopped answering
/// holds a requester open for as long as the transport allows.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// The wall clock in milliseconds since the Unix epoch, the unit a V4 deadline is
/// written in. A deadline is absolute and crosses hosts, so it can only be as
/// accurate as the hosts' clocks; a clock that reads before the epoch reads as
/// zero, which expires every deadline rather than extending one.
fn now_millis() -> Timestamp {
    crate::frame::unix_millis_now()
}

/// How long one V2 exchange may wait: the configured timeout, shortened to the
/// time left before the request's absolute deadline. A deadline already passed is
/// refused before anything is sent, so nothing is left uncertain.
fn request_budget(
    configured: Duration,
    address: Option<&PodAddressBinding>,
) -> Result<(Duration, Option<Timestamp>), PodWireError> {
    budget_at(configured, address, now_millis())
}

/// [`request_budget`] at a given clock reading.
fn budget_at(
    configured: Duration,
    address: Option<&PodAddressBinding>,
    now: Timestamp,
) -> Result<(Duration, Option<Timestamp>), PodWireError> {
    let Some(deadline) = address.and_then(|address| address.deadline) else {
        return Ok((configured, None));
    };
    if now >= deadline {
        return Err(PodWireError::DeadlineExceeded {
            deadline,
            now,
            sent: false,
        });
    }
    let remaining = Duration::from_millis(deadline.0 - now.0);
    // Equality is deadline-bound: a wait that ends exactly at the deadline is the deadline's.
    if remaining <= configured {
        Ok((remaining, Some(deadline)))
    } else {
        Ok((configured, None))
    }
}

/// The error for a timed-out exchange: the deadline's own when the deadline set
/// the wait, the transport's otherwise.
fn timeout_error(bound_by: Option<Timestamp>) -> PodWireError {
    match bound_by {
        Some(deadline) => PodWireError::DeadlineExceeded {
            deadline,
            now: now_millis(),
            sent: true,
        },
        None => PodWireError::Transport("the host did not answer in time".to_owned()),
    }
}

/// Resolves an execution-manifest digest from the authoritative runtime or
/// catalog. The transport never treats a digest supplied by a caller as proof
/// that the corresponding lineage exists.
pub trait ExecutionManifestResolver: Send + Sync {
    type Error;

    fn resolve(&self, digest: &Digest) -> Result<ExecutionManifest, Self::Error>;
}

impl ExecutionManifestResolver for ExecutionManifest {
    type Error = ();

    fn resolve(&self, digest: &Digest) -> Result<ExecutionManifest, Self::Error> {
        (self.manifest_digest == *digest)
            .then_some(self.clone())
            .ok_or(())
    }
}

/// The validity check a runtime-backed binding carries into every request it serves. Compared
/// by identity, which is all a binding's equality needs.
#[derive(Clone)]
struct ManifestValidityHandle(Arc<dyn ptr_runtime::ManifestValidity>);

impl std::fmt::Debug for ManifestValidityHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ManifestValidity")
    }
}

impl PartialEq for ManifestValidityHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ManifestValidityHandle {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodWireV2Binding {
    artifact_id: ArtifactId,
    registered: RegisteredPodBinding,
    generation: Generation,
    revision: Revision,
    placement_epoch: Option<u64>,
    fencing_token: Option<u128>,
    semantic_revision: Option<Digest>,
    execution_manifest: Option<ExecutionManifest>,
    protocol: Option<ProtocolBinding>,
    admission: Option<PodAdmissionBinding>,
    mesh: Option<MeshEndpointBinding>,
    /// Set by [`Self::with_runtime_execution_manifest`]: the check that the manifest the
    /// binding was taken from the runtime for still holds, asked on every request the
    /// binding serves whether or not the host has a manifest registry attached.
    runtime_validity: Option<ManifestValidityHandle>,
}

impl PodWireV2Binding {
    /// The per-request half of a runtime-backed binding: its manifest must still hold now.
    /// A binding that did not come from the runtime asks nothing.
    fn check_runtime_validity(&self) -> Result<(), PodWireError> {
        let Some(validity) = &self.runtime_validity else {
            return Ok(());
        };
        let digest = self
            .execution_manifest
            .as_ref()
            .map(|manifest| manifest.manifest_digest)
            .ok_or(PodWireError::V2BindingUnavailable)?;
        validity
            .0
            .ensure_valid_at(&digest, now_millis())
            .map_err(PodWireError::ManifestNoLongerValid)
    }

    pub fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    pub fn manifest(&self) -> &PodManifest {
        self.registered.manifest()
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn revision(&self) -> Revision {
        self.revision
    }

    pub fn placement_epoch(&self) -> Option<u64> {
        self.placement_epoch
    }

    pub fn fencing_token(&self) -> Option<u128> {
        self.fencing_token
    }

    pub fn mesh(&self) -> Option<&MeshEndpointBinding> {
        self.mesh.as_ref()
    }

    pub fn execution_manifest(&self) -> Option<&ExecutionManifest> {
        self.execution_manifest.as_ref()
    }

    pub fn with_execution_manifest(
        mut self,
        manifest: ExecutionManifest,
    ) -> Result<Self, PodWireError> {
        manifest
            .validate()
            .map_err(|_| PodWireError::V2BindingUnavailable)?;
        if manifest.generation != self.generation || manifest.policy_revision.0 == 0 {
            return Err(PodWireError::V2BindingUnavailable);
        }
        self.execution_manifest = Some(manifest);
        Ok(self)
    }

    pub fn with_resolved_execution_manifest<R>(
        self,
        resolver: &R,
        digest: &Digest,
    ) -> Result<Self, PodWireError>
    where
        R: ExecutionManifestResolver,
    {
        let manifest = resolver
            .resolve(digest)
            .map_err(|_| PodWireError::V2BindingUnavailable)?;
        if manifest.manifest_digest != *digest {
            return Err(PodWireError::V2BindingUnavailable);
        }
        self.with_execution_manifest(manifest)
    }

    /// Build the binding from the runtime's admitted, lineage-validated
    /// registry. PodWire does not accept a caller-created manifest as proof of
    /// runtime admission; the runtime must resolve the exact digest first.
    pub fn with_runtime_execution_manifest(
        self,
        runtime: &ptr_runtime::PtrRuntime,
        validity: Arc<dyn ptr_runtime::ManifestValidity>,
        digest: &Digest,
    ) -> Result<Self, PodWireError> {
        let validated = runtime
            .resolve_execution_manifest(digest)
            .map_err(|_| PodWireError::V2BindingUnavailable)?;
        validity
            .ensure_valid_at(digest, now_millis())
            .map_err(PodWireError::ManifestNoLongerValid)?;
        let mut bound = self.with_execution_manifest(validated.manifest)?;
        bound.runtime_validity = Some(ManifestValidityHandle(validity));
        Ok(bound)
    }

    pub fn with_mesh(mut self, mesh: MeshEndpointBinding) -> Result<Self, PodWireError> {
        mesh.validate()
            .map_err(|_| PodWireError::V2BindingUnavailable)?;
        self.mesh = Some(mesh);
        Ok(self)
    }

    fn accepts(&self, request: &PodRequestV2, now: Timestamp) -> bool {
        (match (&request.address, self.semantic_revision) {
            (None, None) => true,
            (Some(address), Some(expected_revision)) => {
                address.validate().is_ok()
                    && address.hop_limit > 0
                    && address.deadline.is_none_or(|deadline| now < deadline)
                    && address.destination.semantic_revision == expected_revision
                    && address.destination.address.project == self.manifest().project
                    && address.destination.address.pod_id == self.manifest().id
                    && self.registered.namespace() == Some(&address.destination.address.namespace)
                    && address.destination.generation == request.generation
                    && address.protocol == self.protocol
                    && address.mesh == self.mesh
            }
            _ => false,
        }) && request.artifact_id == self.artifact_id
            && request.execution_manifest
                == self
                    .execution_manifest
                    .as_ref()
                    .map(|manifest| manifest.manifest_digest)
            && request.manifest_hash == self.manifest().digest()
            && request.generation == self.generation
            && request.revision == self.revision
            && request.placement_epoch == self.placement_epoch
            && request.fencing_token == self.fencing_token
            && self.admission.as_ref().is_none_or(|admission| {
                request.identity_digest == admission.identity_digest
                    && request.session_id == admission.session_id
                    && request.policy_revision == admission.policy_revision
            })
            && self.manifest().capabilities.contains(&request.capability)
            && self.manifest().protocol_version == request.expect_protocol
            && self.manifest().accepts.contains(&request.payload.type_id)
    }
}

/// Why a pod-wire operation was refused, at the resolution the *local* side gets.
///
/// Richer than what crosses the wire, in one direction only: a host's operator is
/// entitled to know which check refused a request, and the peer is not.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodWireError {
    /// The transport refused, failed, or did not answer in time.
    Transport(String),
    /// The manifest the request names is admitted but no longer holds: a knowledge object it
    /// binds has expired, was invalidated or is gone. The Pod was not invoked.
    ManifestNoLongerValid(ptr_runtime::RuntimeManifestError),
    /// The frame was malformed.
    Frame(FrameError),
    /// The request named another endpoint's key.
    Misaddressed {
        addressed_to: String,
        expected: String,
    },
    /// The access decision refused. Which refusal it was stays here *and* crosses
    /// the wire, because every code this protocol sends is already a fact about the
    /// requester or its own scope — the three that would be an oracle were collapsed
    /// into one before they ever reached this type.
    Refused(RefusalCode),
    /// An answer whose author is not the peer the connection authenticated.
    ///
    /// The case this layer exists for on the requesting side: an answer can be
    /// stored and forwarded, so one that names somebody else is refused rather than
    /// read.
    ForgedAnswer {
        authenticated: String,
        claimed: String,
    },
    /// An answer to a different request id.
    WrongRequest {
        expected: u64,
        answered: u64,
    },
    WrongGeneration {
        expected: Generation,
        answered: Generation,
    },
    WrongRevision {
        expected: Revision,
        answered: Revision,
    },
    StateBindingMismatch {
        expected_epoch: Option<u64>,
        answered_epoch: Option<u64>,
        expected_token: Option<u128>,
        answered_token: Option<u128>,
    },
    /// An answer whose digest is not the request that was sent. A well-formed reply
    /// to a question nobody asked here.
    UnboundAnswer,
    /// A session already completed this request id with different bytes.
    DuplicateRequest {
        request_id: u64,
    },
    /// The session has reached its configured number of concurrent streams.
    Backpressure,
    /// The session was closed or could not open another stream.
    SessionClosed,
    /// The request was sent but the outcome could not be established. It must
    /// not be retried on this session without an explicit reconciliation.
    RequestUncertain {
        request_id: u64,
    },
    V2BindingUnavailable,
    /// The request's absolute deadline passed. `sent` says whether the request had been sent:
    /// `false` means it expired before anything left (nothing ran, a retry is safe with a new
    /// deadline), `true` means it expired while waiting for the answer, so the host may still
    /// have run it (a session marks the request uncertain; do not retry blindly). A host that
    /// refuses an expired request reports it with `sent: true`.
    DeadlineExceeded {
        deadline: Timestamp,
        now: Timestamp,
        sent: bool,
    },
}

impl PodWireError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Transport(_) => "PTR_PODW_TRANSPORT",
            Self::Frame(_) => "PTR_PODW_FRAME",
            Self::Misaddressed { .. } => "PTR_PODW_MISADDRESSED",
            Self::Refused(code) => code.code(),
            Self::ForgedAnswer { .. } => "PTR_PODW_FORGED_ANSWER",
            Self::WrongRequest { .. } => "PTR_PODW_WRONG_REQUEST",
            Self::WrongGeneration { .. } => "PTR_PODW_WRONG_GENERATION",
            Self::WrongRevision { .. } => "PTR_PODW_WRONG_REVISION",
            Self::StateBindingMismatch { .. } => "PTR_PODW_STATE_BINDING_MISMATCH",
            Self::UnboundAnswer => "PTR_PODW_UNBOUND_ANSWER",
            Self::DuplicateRequest { .. } => "PTR_PODW_DUPLICATE_REQUEST",
            Self::Backpressure => "PTR_PODW_BACKPRESSURE",
            Self::SessionClosed => "PTR_PODW_SESSION_CLOSED",
            Self::RequestUncertain { .. } => "PTR_PODW_REQUEST_UNCERTAIN",
            Self::V2BindingUnavailable => "PTR_PODW_V2_BINDING_UNAVAILABLE",
            Self::DeadlineExceeded { .. } => "PTR_PODW_DEADLINE_EXCEEDED",
            Self::ManifestNoLongerValid(_) => "PTR_PODW_MANIFEST_NO_LONGER_VALID",
        }
    }
}

impl From<FrameError> for PodWireError {
    /// Carry a framing refusal through unchanged.
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl std::fmt::Display for PodWireError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for PodWireError {}

/// What one served request did, for the host's own caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Served {
    /// The peer the connection authenticated. Not a value from the payload — there
    /// is no such value.
    pub peer: NodeIdentity,
    /// The request id, when the frame decoded far enough to have one.
    pub request_id: Option<u64>,
    /// What the answer told the peer.
    pub reported: PodOutcome,
    /// The local reason behind a refusal.
    pub refused: Option<PodWireError>,
}

/// Pods reachable over `ALPN_PODWIRE`.
///
/// The host owns the registry, the policy and the verifier. What it does **not**
/// own is a runtime or a ledger, and that absence is the protocol's strongest
/// claim: a Pod invoked through this endpoint cannot write to the host's history,
/// because this endpoint has no history to write to. The in-process loop promotes
/// a Pod's output into the semantic state of the model run that asked for it;
/// there is no such run behind a request that arrived over a wire, and inventing
/// one would let a peer write to a host's semantic store through a Read-only door.
pub struct PodHost {
    registry: PodRegistry,
    policy: PodAccessPolicy,
    verifier: Box<dyn Verifier<TypedPayload> + Send + Sync>,
    transport: IrohTransport,
    runtime_manifests: Option<RuntimeManifests>,
}

/// The live manifest registry a host checks requests against, and the validity check that
/// goes with it. The two are attached together so a registry cannot be served from without
/// the time check.
struct RuntimeManifests {
    registry: Arc<RwLock<ptr_runtime::ExecutionManifestRegistry>>,
    validity: Arc<dyn ptr_runtime::ManifestValidity>,
}

impl PodHost {
    /// Bind an endpoint for `ALPN_PODWIRE` around a registry, a policy and the
    /// verifier every answer must pass before it leaves.
    pub async fn bind<V>(
        registry: PodRegistry,
        policy: PodAccessPolicy,
        verifier: V,
    ) -> Result<Self, PodWireError>
    where
        V: Verifier<TypedPayload> + Send + Sync + 'static,
    {
        let transport = IrohTransport::bind(&[ALPN_PODWIRE])
            .await
            .map_err(PodWireError::Transport)?;
        Ok(Self {
            registry,
            policy,
            verifier: Box::new(verifier),
            transport,
            runtime_manifests: None,
        })
    }

    /// Attach the live runtime manifest registry together with the check that its manifests
    /// still hold. The registry is shared rather than copied so revocation is observed by
    /// already-created bindings before their next invocation; the validity check runs with
    /// the host's clock (milliseconds since the epoch) on every request, after the registry
    /// resolved the digest, and a failure refuses the request before the Pod is invoked.
    pub fn with_runtime_manifest_registry(
        mut self,
        registry: Arc<RwLock<ptr_runtime::ExecutionManifestRegistry>>,
        validity: Arc<dyn ptr_runtime::ManifestValidity>,
    ) -> Self {
        self.runtime_manifests = Some(RuntimeManifests { registry, validity });
        self
    }

    /// The runtime-manifest part of admitting a request: with a registry attached the request
    /// must name an admitted, unrevoked manifest that is still valid now. Without one, nothing
    /// is asked.
    fn check_runtime_manifest(&self, request: &PodRequestV2) -> Result<(), PodWireError> {
        let Some(manifests) = &self.runtime_manifests else {
            return Ok(());
        };
        let digest = request
            .execution_manifest
            .ok_or(PodWireError::V2BindingUnavailable)?;
        let current = manifests
            .registry
            .read()
            .ok()
            .and_then(|registry| registry.resolve(&digest).ok())
            .ok_or(PodWireError::V2BindingUnavailable)?;
        if current.manifest.manifest_digest != digest {
            return Err(PodWireError::V2BindingUnavailable);
        }
        manifests
            .validity
            .ensure_valid_at(&digest, now_millis())
            .map_err(PodWireError::ManifestNoLongerValid)
    }

    pub fn bind_v2(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let registered = self
            .registry
            .bind_registered(project, pod)
            .map_err(|_| PodWireError::V2BindingUnavailable)?;
        Ok(PodWireV2Binding {
            artifact_id,
            registered,
            generation,
            revision,
            placement_epoch: None,
            fencing_token: None,
            semantic_revision: None,
            execution_manifest: None,
            protocol: None,
            admission: None,
            mesh: None,
            runtime_validity: None,
        })
    }

    pub fn bind_v2_addressed(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let mut binding = self.bind_v2(project, pod, artifact_id, generation, revision)?;
        // An address names a lifecycle namespace; the same pod id, manifest and generation can
        // be registered for several. An entry that names no namespace cannot be told apart, so
        // it is not addressable (fail closed).
        if binding.registered.namespace().is_none() {
            return Err(PodWireError::V2BindingUnavailable);
        }
        // The addressed revision is the admitted manifest digest.  Callers must
        // not be able to mint a second, untracked semantic identity for the same
        // registry entry.
        binding.semantic_revision = Some(binding.registered.manifest().digest());
        Ok(binding)
    }

    pub fn bind_v2_stateful(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
        placement_epoch: u64,
        fencing_token: u128,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let mut binding = self.bind_v2(project, pod, artifact_id, generation, revision)?;
        binding.placement_epoch = Some(placement_epoch);
        binding.fencing_token = Some(fencing_token);
        Ok(binding)
    }

    pub fn bind_v2_addressed_with_protocol(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
        protocol: ProtocolBinding,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let mut binding =
            self.bind_v2_addressed(project, pod, artifact_id, generation, revision)?;
        binding.protocol = Some(protocol);
        Ok(binding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_v2_stateful_addressed(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
        placement_epoch: u64,
        fencing_token: u128,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let mut binding =
            self.bind_v2_addressed(project, pod, artifact_id, generation, revision)?;
        binding.placement_epoch = Some(placement_epoch);
        binding.fencing_token = Some(fencing_token);
        Ok(binding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_v2_admitted(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
        placement_epoch: u64,
        fencing_token: u128,
        admission: PodAdmissionBinding,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let mut binding = self.bind_v2_stateful(
            project,
            pod,
            artifact_id,
            generation,
            revision,
            placement_epoch,
            fencing_token,
        )?;
        binding.admission = Some(admission);
        Ok(binding)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_v2_runtime_admitted(
        &self,
        runtime: &ptr_runtime::PtrRuntime,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
        placement_epoch: u64,
        fencing_token: u128,
        admission: PodAdmissionBinding,
        validity: Arc<dyn ptr_runtime::ManifestValidity>,
        execution_manifest: &Digest,
    ) -> Result<PodWireV2Binding, PodWireError> {
        self.bind_v2_admitted(
            project,
            pod,
            artifact_id,
            generation,
            revision,
            placement_epoch,
            fencing_token,
            admission,
        )?
        .with_runtime_execution_manifest(runtime, validity, execution_manifest)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_v2_admitted_addressed(
        &self,
        project: &ProjectId,
        pod: &PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        revision: Revision,
        placement_epoch: u64,
        fencing_token: u128,
        admission: PodAdmissionBinding,
    ) -> Result<PodWireV2Binding, PodWireError> {
        let mut binding = self.bind_v2_stateful_addressed(
            project,
            pod,
            artifact_id,
            generation,
            revision,
            placement_epoch,
            fencing_token,
        )?;
        binding.admission = Some(admission);
        Ok(binding)
    }

    /// The identity a requester must address.
    pub fn identity(&self) -> NodeIdentity {
        self.transport.identity()
    }

    /// Where requesters should send.
    pub fn address(&self) -> EndpointAddr {
        self.transport.direct_addr()
    }

    /// The policy, for the host's own trusted use — admitting and withdrawing
    /// peers between requests.
    pub fn policy_mut(&mut self) -> &mut PodAccessPolicy {
        &mut self.policy
    }

    /// Accept one connection, serve what it carries, and answer.
    ///
    /// Every request is answered, including every refused one: leaving a peer
    /// hanging makes this side's refusal the peer's problem. The answer is bound to
    /// the bytes that actually arrived, so even a request this build cannot parse
    /// gets a reply its sender can check against what it sent.
    ///
    /// The order is the protocol:
    ///
    /// 1. The peer is the **authenticated connection**. Nothing in the payload
    ///    names a sender, and nothing here reads one.
    /// 2. The frame must decode.
    /// 3. The request must be addressed to this endpoint's key. A request
    ///    legitimate at another host is not legitimate here.
    /// 4. [`answer`] decides the rest, from the policy and the registry.
    pub async fn serve_once(&self) -> Result<Served, PodWireError> {
        let incoming = self
            .transport
            .accept_once(MAX_FRAME_BYTES)
            .await
            .map_err(PodWireError::Transport)?;

        // (1) The sender is the connection. There is no payload field to prefer
        // over it, which is the shape of the format rather than a check made here.
        let peer = incoming.peer.clone();
        let digest = request_digest(&incoming.payload);
        let (request_id, outcome, mut refused) = self.decide(&peer, &incoming.payload);

        let mut answered = PodAnswer {
            responder: self.identity().public_key,
            request_id: request_id.unwrap_or(0),
            request_digest: digest,
            outcome,
        };

        // An answer that will not fit in a frame is still an answer. Pre-checking
        // one field would have missed the others — a Pod's output *type* is bounded
        // too — and every miss leaves a peer holding an open connection because this
        // side could not phrase its reply. So the bound is enforced where it is
        // known: by trying, and refusing when the attempt fails.
        //
        // Nothing applied, so there is nothing for the requester to reconcile; it is
        // told the answer exists and does not fit, which it can act on.
        let frame = match encode_answer(&answered) {
            Ok(frame) => frame,
            Err(error) => {
                answered.outcome = PodOutcome::Refused {
                    code: RefusalCode::AnswerTooLarge,
                };
                refused = Some(PodWireError::Frame(error));
                encode_answer(&answered)?
            }
        };
        incoming
            .respond(&frame)
            .await
            .map_err(PodWireError::Transport)?;

        Ok(Served {
            peer,
            request_id,
            reported: answered.outcome,
            refused,
        })
    }

    /// Serve one generation-bound V2 request using the supplied admitted binding.
    pub async fn serve_once_v2(&self, binding: &PodWireV2Binding) -> Result<Served, PodWireError> {
        let incoming = self
            .transport
            .accept_once(MAX_FRAME_BYTES)
            .await
            .map_err(PodWireError::Transport)?;
        let peer = incoming.peer.clone();
        let digest = request_digest(&incoming.payload);
        let request = decode_request_v2(&incoming.payload);
        let request_execution_manifest = request
            .as_ref()
            .ok()
            .and_then(|request| request.execution_manifest);
        let answer_address = request
            .as_ref()
            .ok()
            .and_then(|request| request.address.clone());
        let state_binding = request
            .as_ref()
            .ok()
            .map(|request| (request.placement_epoch, request.fencing_token));
        let (
            request_id,
            generation,
            revision,
            identity_digest,
            session_id,
            policy_revision,
            outcome,
            refused,
        ) = match request {
            Ok(request) => {
                let request_id = Some(request.request_id);
                if request.addressed_to != self.identity().public_key {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Misaddressed,
                        },
                        Some(PodWireError::Misaddressed {
                            addressed_to: request.addressed_to,
                            expected: self.identity().public_key,
                        }),
                    )
                } else if let Some(error) = expired_deadline(&request) {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(error),
                    )
                } else if !binding.accepts(&request, now_millis()) {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(PodWireError::V2BindingUnavailable),
                    )
                } else if let Err(error) = self
                    .check_runtime_manifest(&request)
                    .and_then(|()| binding.check_runtime_validity())
                {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(error),
                    )
                } else if let Some(error) = expired_deadline(&request) {
                    // The checks above take time; the deadline is asked again with a fresh
                    // clock right before the Pod would run.
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(error),
                    )
                } else {
                    self.answer_v2_bound_request(&peer, binding, request, request_id)
                }
            }
            Err(error) => (
                None,
                Generation(0),
                Revision(0),
                [0; 32],
                SessionId::from("malformed"),
                Revision(0),
                PodOutcome::Refused {
                    code: RefusalCode::Malformed,
                },
                Some(PodWireError::Frame(error)),
            ),
        };
        let answer = PodAnswerV2 {
            address: answer_address,
            execution_manifest: request_execution_manifest,
            responder: self.identity().public_key,
            request_id: request_id.unwrap_or(0),
            request_digest: digest,
            generation,
            revision,
            identity_digest,
            session_id,
            policy_revision,
            placement_epoch: state_binding.and_then(|binding| binding.0),
            fencing_token: state_binding.and_then(|binding| binding.1),
            outcome,
        };
        let frame = encode_answer_v2(&answer)?;
        incoming
            .respond(&frame)
            .await
            .map_err(PodWireError::Transport)?;
        Ok(Served {
            peer,
            request_id,
            reported: answer.outcome,
            refused,
        })
    }

    /// Accept one authenticated QUIC connection and serve a bounded number of
    /// V2 streams on it. A bound is required so a peer cannot keep a host's
    /// accept loop alive forever; callers can create another session explicitly.
    pub async fn serve_session_v2(
        &self,
        binding: &PodWireV2Binding,
        max_requests: usize,
    ) -> Result<Vec<Served>, PodWireError> {
        if max_requests == 0 {
            return Err(PodWireError::SessionClosed);
        }
        let session = self
            .transport
            .accept_session()
            .await
            .map_err(PodWireError::Transport)?;
        let mut served = Vec::with_capacity(max_requests);
        for _ in 0..max_requests {
            let incoming = session
                .accept_request(MAX_FRAME_BYTES)
                .await
                .map_err(PodWireError::Transport)?;
            served.push(self.serve_v2_incoming(binding, incoming).await?);
        }
        session.wait_closed().await;
        Ok(served)
    }

    fn answer_v2_bound_request(
        &self,
        peer: &NodeIdentity,
        binding: &PodWireV2Binding,
        request: PodRequestV2,
        request_id: Option<u64>,
    ) -> (
        Option<u64>,
        Generation,
        Revision,
        Digest,
        SessionId,
        Revision,
        PodOutcome,
        Option<PodWireError>,
    ) {
        match answer_bound(
            &self.policy,
            &self.registry,
            self.verifier.as_ref(),
            &NodeId(peer.public_key.clone()),
            binding.registered.pod_id(),
            binding.manifest(),
            &request.capability,
            request.expect_protocol,
            request.payload,
        ) {
            Ok(output) if binding.manifest().produces.contains(&output.type_id) => (
                request_id,
                request.generation,
                request.revision,
                request.identity_digest,
                request.session_id,
                request.policy_revision,
                PodOutcome::Answered { output },
                None,
            ),
            Ok(_) => (
                request_id,
                request.generation,
                request.revision,
                request.identity_digest,
                request.session_id,
                request.policy_revision,
                PodOutcome::Refused {
                    code: RefusalCode::Unverified,
                },
                Some(PodWireError::Refused(RefusalCode::Unverified)),
            ),
            Err(code) => (
                request_id,
                request.generation,
                request.revision,
                request.identity_digest,
                request.session_id,
                request.policy_revision,
                PodOutcome::Refused { code },
                Some(PodWireError::Refused(code)),
            ),
        }
    }

    async fn serve_v2_incoming(
        &self,
        binding: &PodWireV2Binding,
        incoming: ptr_net::IrohIncoming,
    ) -> Result<Served, PodWireError> {
        let peer = incoming.peer.clone();
        let digest = request_digest(&incoming.payload);
        let request = decode_request_v2(&incoming.payload);
        let request_execution_manifest = request
            .as_ref()
            .ok()
            .and_then(|request| request.execution_manifest);
        let answer_address = request
            .as_ref()
            .ok()
            .and_then(|request| request.address.clone());
        let state_binding = request
            .as_ref()
            .ok()
            .map(|request| (request.placement_epoch, request.fencing_token));
        let (
            request_id,
            generation,
            revision,
            identity_digest,
            session_id,
            policy_revision,
            outcome,
            refused,
        ) = match request {
            Ok(request) => {
                let request_id = Some(request.request_id);
                if request.addressed_to != self.identity().public_key {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Misaddressed,
                        },
                        Some(PodWireError::Misaddressed {
                            addressed_to: request.addressed_to,
                            expected: self.identity().public_key,
                        }),
                    )
                } else if let Some(error) = expired_deadline(&request) {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(error),
                    )
                } else if !binding.accepts(&request, now_millis()) {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(PodWireError::V2BindingUnavailable),
                    )
                } else if let Err(error) = self
                    .check_runtime_manifest(&request)
                    .and_then(|()| binding.check_runtime_validity())
                {
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(error),
                    )
                } else if let Some(error) = expired_deadline(&request) {
                    // The checks above take time; the deadline is asked again with a fresh
                    // clock right before the Pod would run.
                    (
                        request_id,
                        request.generation,
                        request.revision,
                        request.identity_digest,
                        request.session_id.clone(),
                        request.policy_revision,
                        PodOutcome::Refused {
                            code: RefusalCode::Unavailable,
                        },
                        Some(error),
                    )
                } else {
                    self.answer_v2_bound_request(&peer, binding, request, request_id)
                }
            }
            Err(error) => (
                None,
                Generation(0),
                Revision(0),
                [0; 32],
                SessionId::from("malformed"),
                Revision(0),
                PodOutcome::Refused {
                    code: RefusalCode::Malformed,
                },
                Some(PodWireError::Frame(error)),
            ),
        };
        let answer = PodAnswerV2 {
            address: answer_address,
            execution_manifest: request_execution_manifest,
            responder: self.identity().public_key,
            request_id: request_id.unwrap_or(0),
            request_digest: digest,
            generation,
            revision,
            identity_digest,
            session_id,
            policy_revision,
            placement_epoch: state_binding.and_then(|binding| binding.0),
            fencing_token: state_binding.and_then(|binding| binding.1),
            outcome,
        };
        let frame = encode_answer_v2(&answer)?;
        incoming
            .respond(&frame)
            .await
            .map_err(PodWireError::Transport)?;
        Ok(Served {
            peer,
            request_id,
            reported: answer.outcome,
            refused,
        })
    }

    /// Everything between the bytes arriving and the answer going out.
    ///
    /// Split out so that answering the peer is unconditional: a refusal decided
    /// here still becomes an answer rather than a dropped connection.
    fn decide(
        &self,
        peer: &NodeIdentity,
        payload: &[u8],
    ) -> (Option<u64>, PodOutcome, Option<PodWireError>) {
        // (2) The frame must decode.
        let request = match decode_request(payload) {
            Ok(request) => request,
            Err(error) => {
                return (
                    None,
                    PodOutcome::Refused {
                        code: RefusalCode::Malformed,
                    },
                    Some(PodWireError::Frame(error)),
                )
            }
        };
        let request_id = Some(request.request_id);

        // (3) Addressed here, or nowhere.
        let expected = self.identity().public_key;
        if request.addressed_to != expected {
            return (
                request_id,
                PodOutcome::Refused {
                    code: RefusalCode::Misaddressed,
                },
                Some(PodWireError::Misaddressed {
                    addressed_to: request.addressed_to,
                    expected,
                }),
            );
        }

        // (4) The access decision. The peer is the connection's, never the frame's.
        match answer(
            &self.policy,
            &self.registry,
            self.verifier.as_ref(),
            &NodeId(peer.public_key.clone()),
            &request.capability,
            request.expect_protocol,
            request.payload,
        ) {
            Ok(output) => (request_id, PodOutcome::Answered { output }, None),
            Err(code) => (
                request_id,
                PodOutcome::Refused { code },
                Some(PodWireError::Refused(code)),
            ),
        }
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}

/// A requester with an authenticated endpoint.
pub struct PodClient {
    transport: IrohTransport,
    request_timeout: Duration,
}

/// A requester-side session sharing one authenticated Iroh connection across
/// multiple PodWire streams. The wire contract and answer validation are exactly
/// the same as `PodClient`; only connection lifetime and bounded concurrency differ.
pub struct PodSession {
    transport: IrohSession,
    request_timeout: Duration,
    v1_cache: Mutex<BTreeMap<u64, ([u8; 32], PodAnswer)>>,
    v2_cache: Mutex<BTreeMap<u64, ([u8; 32], PodAnswerV2)>>,
    v1_inflight: Mutex<BTreeSet<u64>>,
    v2_inflight: Mutex<BTreeSet<u64>>,
    uncertain_v1: Mutex<BTreeSet<u64>>,
    uncertain_v2: Mutex<BTreeSet<u64>>,
}

struct RequestReservation<'a> {
    ids: &'a Mutex<BTreeSet<u64>>,
    request_id: u64,
}

impl Drop for RequestReservation<'_> {
    fn drop(&mut self) {
        self.ids
            .lock()
            .expect("pod session in-flight mutex poisoned")
            .remove(&self.request_id);
    }
}

impl PodSession {
    pub fn peer(&self) -> &NodeIdentity {
        self.transport.peer()
    }

    pub fn with_request_timeout(&mut self, timeout: Duration) {
        self.request_timeout = timeout;
    }

    pub async fn request(&self, request: &PodRequest) -> Result<PodAnswer, PodWireError> {
        let frame = encode_request(request)?;
        let digest = request_digest(&frame);
        if self
            .uncertain_v1
            .lock()
            .expect("pod session uncertain mutex poisoned")
            .contains(&request.request_id)
        {
            return Err(PodWireError::RequestUncertain {
                request_id: request.request_id,
            });
        }
        if let Some((cached_digest, answer)) = self
            .v1_cache
            .lock()
            .expect("pod session cache mutex poisoned")
            .get(&request.request_id)
            .cloned()
        {
            if cached_digest == digest {
                return Ok(answer);
            }
            return Err(PodWireError::DuplicateRequest {
                request_id: request.request_id,
            });
        }
        let _reservation = self.reserve(&self.v1_inflight, request.request_id)?;
        let response = tokio::time::timeout(
            self.request_timeout,
            self.transport.request(&frame, MAX_FRAME_BYTES),
        )
        .await
        .map_err(|_| {
            self.mark_uncertain_v1(request.request_id);
            PodWireError::Transport("the host did not answer in time".to_owned())
        })?
        .map_err(|error| {
            self.mark_uncertain_v1(request.request_id);
            session_error(error)
        })?;
        let answered = decode_answer(&response).map_err(|error| {
            self.mark_uncertain_v1(request.request_id);
            PodWireError::Frame(error)
        })?;
        let answer = validate_answer(
            &answered,
            request.request_id,
            digest,
            &self.peer().public_key,
        )
        .inspect_err(|_| {
            self.mark_uncertain_v1(request.request_id);
        })?;
        self.remember_v1(request.request_id, digest, answer.clone());
        Ok(answer)
    }

    pub async fn request_v2(&self, request: &PodRequestV2) -> Result<PodAnswerV2, PodWireError> {
        let frame = encode_request_v2(request)?;
        let digest = request_digest(&frame);
        if self
            .uncertain_v2
            .lock()
            .expect("pod session uncertain mutex poisoned")
            .contains(&request.request_id)
        {
            return Err(PodWireError::RequestUncertain {
                request_id: request.request_id,
            });
        }
        if let Some((cached_digest, answer)) = self
            .v2_cache
            .lock()
            .expect("pod session cache mutex poisoned")
            .get(&request.request_id)
            .cloned()
        {
            if cached_digest == digest {
                // A replay is still a call: past its deadline it is refused as the first
                // send would be, nothing having left.
                request_budget(self.request_timeout, request.address.as_ref())?;
                return Ok(answer);
            }
            return Err(PodWireError::DuplicateRequest {
                request_id: request.request_id,
            });
        }
        let (budget, bound_by) = request_budget(self.request_timeout, request.address.as_ref())?;
        let _reservation = self.reserve(&self.v2_inflight, request.request_id)?;
        let response =
            tokio::time::timeout(budget, self.transport.request(&frame, MAX_FRAME_BYTES))
                .await
                .map_err(|_| {
                    self.mark_uncertain_v2(request.request_id);
                    timeout_error(bound_by)
                })?
                .map_err(|error| {
                    self.mark_uncertain_v2(request.request_id);
                    session_error(error)
                })?;
        let answered = decode_answer_v2(&response).map_err(|error| {
            self.mark_uncertain_v2(request.request_id);
            PodWireError::Frame(error)
        })?;
        let answer = validate_answer_v2(&answered, request, digest, &self.peer().public_key)
            .inspect_err(|_| {
                self.mark_uncertain_v2(request.request_id);
            })?;
        self.remember_v2(request.request_id, digest, answer.clone());
        Ok(answer)
    }

    fn reserve<'a>(
        &self,
        ids: &'a Mutex<BTreeSet<u64>>,
        request_id: u64,
    ) -> Result<RequestReservation<'a>, PodWireError> {
        let mut guard = ids.lock().map_err(|_| {
            PodWireError::Transport("pod session in-flight mutex poisoned".to_owned())
        })?;
        if !guard.insert(request_id) {
            return Err(PodWireError::DuplicateRequest { request_id });
        }
        drop(guard);
        Ok(RequestReservation { ids, request_id })
    }

    fn remember_v1(&self, request_id: u64, digest: [u8; 32], answer: PodAnswer) {
        let mut cache = self
            .v1_cache
            .lock()
            .expect("pod session cache mutex poisoned");
        if cache.len() >= 256 {
            if let Some(oldest) = cache.keys().next().copied() {
                cache.remove(&oldest);
            }
        }
        cache.insert(request_id, (digest, answer));
    }

    fn remember_v2(&self, request_id: u64, digest: [u8; 32], answer: PodAnswerV2) {
        let mut cache = self
            .v2_cache
            .lock()
            .expect("pod session cache mutex poisoned");
        if cache.len() >= 256 {
            if let Some(oldest) = cache.keys().next().copied() {
                cache.remove(&oldest);
            }
        }
        cache.insert(request_id, (digest, answer));
    }

    fn mark_uncertain_v1(&self, request_id: u64) {
        self.uncertain_v1
            .lock()
            .expect("pod session uncertain mutex poisoned")
            .insert(request_id);
    }

    fn mark_uncertain_v2(&self, request_id: u64) {
        self.uncertain_v2
            .lock()
            .expect("pod session uncertain mutex poisoned")
            .insert(request_id);
    }

    pub fn uncertain_requests(&self) -> (Vec<u64>, Vec<u64>) {
        let v1 = self
            .uncertain_v1
            .lock()
            .expect("pod session uncertain mutex poisoned")
            .iter()
            .copied()
            .collect();
        let v2 = self
            .uncertain_v2
            .lock()
            .expect("pod session uncertain mutex poisoned")
            .iter()
            .copied()
            .collect();
        (v1, v2)
    }

    /// Forward an outcome-uncertain stateful request to the runtime-side
    /// recovery adapter. The adapter owns scope fencing, lease cleanup,
    /// journal append, and recompute; this transport crate only reports the
    /// typed boundary crossing.
    pub fn recover_uncertain_request<R: StatefulRequestRecovery>(
        &self,
        request_id: u64,
        scope_id: ScopeId,
        recovery: &mut R,
    ) -> Result<(), R::Error> {
        recovery.recover_uncertain(UncertainRequest {
            request_id,
            scope_id,
        })
    }

    pub async fn close(&self) {
        self.transport.close().await;
    }
}

fn session_error(error: String) -> PodWireError {
    if error.contains("backpressure") {
        PodWireError::Backpressure
    } else if error.contains("closed") || error.contains("connection lost") {
        PodWireError::SessionClosed
    } else {
        PodWireError::Transport(error)
    }
}

fn validate_answer(
    answered: &PodAnswer,
    expected_request: u64,
    digest: [u8; 32],
    authenticated: &str,
) -> Result<PodAnswer, PodWireError> {
    if answered.responder != authenticated {
        return Err(PodWireError::ForgedAnswer {
            authenticated: authenticated.to_owned(),
            claimed: answered.responder.clone(),
        });
    }
    if answered.request_id != expected_request {
        return Err(PodWireError::WrongRequest {
            expected: expected_request,
            answered: answered.request_id,
        });
    }
    if answered.request_digest != digest {
        return Err(PodWireError::UnboundAnswer);
    }
    Ok(answered.clone())
}

/// The refusal for a request whose absolute deadline has already passed on this
/// host. It crosses the wire as `Unavailable`: the peer learns nothing a clock
/// would not tell it, and the host's operator sees the real reason.
fn expired_deadline(request: &PodRequestV2) -> Option<PodWireError> {
    let deadline = request.address.as_ref()?.deadline?;
    let now = now_millis();
    (now >= deadline).then_some(PodWireError::DeadlineExceeded {
        deadline,
        now,
        sent: true,
    })
}

fn validate_answer_v2(
    answered: &PodAnswerV2,
    request: &PodRequestV2,
    digest: [u8; 32],
    authenticated: &str,
) -> Result<PodAnswerV2, PodWireError> {
    if answered.responder != authenticated {
        return Err(PodWireError::ForgedAnswer {
            authenticated: authenticated.to_owned(),
            claimed: answered.responder.clone(),
        });
    }
    if answered.request_id != request.request_id {
        return Err(PodWireError::WrongRequest {
            expected: request.request_id,
            answered: answered.request_id,
        });
    }
    if answered.address != request.address {
        return Err(PodWireError::V2BindingUnavailable);
    }
    if answered.execution_manifest != request.execution_manifest {
        return Err(PodWireError::V2BindingUnavailable);
    }
    if answered.generation != request.generation {
        return Err(PodWireError::WrongGeneration {
            expected: request.generation,
            answered: answered.generation,
        });
    }
    if answered.revision != request.revision {
        return Err(PodWireError::WrongRevision {
            expected: request.revision,
            answered: answered.revision,
        });
    }
    if answered.placement_epoch != request.placement_epoch
        || answered.fencing_token != request.fencing_token
    {
        return Err(PodWireError::StateBindingMismatch {
            expected_epoch: request.placement_epoch,
            answered_epoch: answered.placement_epoch,
            expected_token: request.fencing_token,
            answered_token: answered.fencing_token,
        });
    }
    if answered.identity_digest != request.identity_digest
        || answered.session_id != request.session_id
        || answered.policy_revision != request.policy_revision
    {
        return Err(PodWireError::V2BindingUnavailable);
    }
    if answered.request_digest != digest {
        return Err(PodWireError::UnboundAnswer);
    }
    Ok(answered.clone())
}

impl PodClient {
    /// Bind an endpoint for `ALPN_PODWIRE`.
    pub async fn bind() -> Result<Self, PodWireError> {
        let transport = IrohTransport::bind(&[ALPN_PODWIRE])
            .await
            .map_err(PodWireError::Transport)?;
        Ok(Self {
            transport,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        })
    }

    /// The identity a host will authenticate for this requester.
    pub fn identity(&self) -> NodeIdentity {
        self.transport.identity()
    }

    /// How long to wait for a host before giving up.
    pub fn with_request_timeout(&mut self, timeout: Duration) {
        self.request_timeout = timeout;
    }

    /// Establish one authenticated connection and reuse it for multiple
    /// requests. `max_in_flight` is a hard bound; exceeding it returns
    /// `PodWireError::Backpressure` immediately.
    pub async fn connect_session(
        &self,
        host: PeerAddress,
        max_in_flight: usize,
    ) -> Result<PodSession, PodWireError> {
        let host: EndpointAddr = host.into_address();
        let transport = self
            .transport
            .connect_session(host, ALPN_PODWIRE, max_in_flight)
            .await
            .map_err(session_error)?;
        Ok(PodSession {
            transport,
            request_timeout: self.request_timeout,
            v1_cache: Mutex::new(BTreeMap::new()),
            v2_cache: Mutex::new(BTreeMap::new()),
            v1_inflight: Mutex::new(BTreeSet::new()),
            v2_inflight: Mutex::new(BTreeSet::new()),
            uncertain_v1: Mutex::new(BTreeSet::new()),
            uncertain_v2: Mutex::new(BTreeSet::new()),
        })
    }

    /// Send one request and return the answer, or refuse it.
    ///
    /// The host is a [`PeerAddress`], which can only have come from a
    /// [`ptr_net::PeerBook`] the deployment installed. A bare address does not
    /// satisfy this signature, so a requester cannot dial one it was handed by a
    /// peer or read out of a payload — and an unrecorded peer is refused by the book
    /// before any connection is attempted.
    ///
    /// ```compile_fail
    /// # async fn dial(client: &ptr_podwire::PodClient, request: &ptr_podwire::PodRequest) {
    /// let bare: ptr_net::EndpointAddr = unimplemented!();
    /// let _ = client.request(bare, request).await;
    /// # }
    /// ```
    ///
    /// There is deliberately **no** second check here that the request names the peer
    /// being dialled. It would refuse a local bug one round trip earlier and buy no
    /// property — the host checks the name it was sent, and must, since a requester is
    /// not trusted about it.
    ///
    /// Then three checks on the answer, in this order. The **author** first: an
    /// answer from an endpoint other than the one the connection authenticated is
    /// refused before anything in it is read, because nothing in it is worth
    /// reading. Then the request id, then the digest of the exact bytes that were
    /// sent.
    ///
    /// What this does **not** establish: an answer is not signed. Inside this call
    /// the connection vouches for its author; once the bytes are stored or
    /// forwarded, nothing does. And the book cannot tell a *wrong* address for the
    /// right peer from a right one — that dial fails on the authenticated key rather
    /// than reaching the wrong host.
    pub async fn request(
        &self,
        host: PeerAddress,
        request: &PodRequest,
    ) -> Result<PodAnswer, PodWireError> {
        let host: EndpointAddr = host.into_address();
        let authenticated = host.id.to_string();
        let frame = encode_request(request)?;
        let digest = request_digest(&frame);

        let response = tokio::time::timeout(
            self.request_timeout,
            self.transport
                .request(host, ALPN_PODWIRE, &frame, MAX_FRAME_BYTES),
        )
        .await
        .map_err(|_| PodWireError::Transport("the host did not answer in time".to_owned()))?
        .map_err(PodWireError::Transport)?;

        let answered = decode_answer(&response)?;
        if answered.responder != authenticated {
            return Err(PodWireError::ForgedAnswer {
                authenticated,
                claimed: answered.responder,
            });
        }
        if answered.request_id != request.request_id {
            return Err(PodWireError::WrongRequest {
                expected: request.request_id,
                answered: answered.request_id,
            });
        }
        if answered.request_digest != digest {
            return Err(PodWireError::UnboundAnswer);
        }
        Ok(answered)
    }

    pub async fn request_v2(
        &self,
        host: PeerAddress,
        request: &PodRequestV2,
    ) -> Result<PodAnswerV2, PodWireError> {
        let host: EndpointAddr = host.into_address();
        let authenticated = host.id.to_string();
        let frame = encode_request_v2(request)?;
        let digest = request_digest(&frame);
        let (budget, bound_by) = request_budget(self.request_timeout, request.address.as_ref())?;
        let response = tokio::time::timeout(
            budget,
            self.transport
                .request(host, ALPN_PODWIRE, &frame, MAX_FRAME_BYTES),
        )
        .await
        .map_err(|_| timeout_error(bound_by))?
        .map_err(PodWireError::Transport)?;
        let answered = decode_answer_v2(&response)?;
        validate_answer_v2(&answered, request, digest, &authenticated)?;
        Ok(answered)
    }

    /// Close the endpoint.
    pub async fn close(&self) {
        self.transport.close().await;
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn bound_to(deadline: u64) -> PodAddressBinding {
        PodAddressBinding {
            source: None,
            destination: ptr_types::PodRevisionAddress {
                address: ptr_types::PodAddress::new("p".into(), "n".into(), "pod".into()).unwrap(),
                semantic_revision: [1; 32],
                generation: Generation(1),
            },
            trace_id: "budget".into(),
            hop_limit: 4,
            visited: Vec::new(),
            protocol: None,
            mesh: None,
            deadline: Some(Timestamp(deadline)),
            origin_ts: Some(Timestamp(1)),
        }
    }

    #[test]
    fn a_wait_that_ends_exactly_at_the_deadline_is_the_deadlines_own() {
        let configured = Duration::from_millis(500);
        let binding = bound_to(10_500);
        // Remaining equals the configured timeout: the deadline binds the wait.
        let (budget, bound_by) = budget_at(configured, Some(&binding), Timestamp(10_000)).unwrap();
        assert_eq!((budget, bound_by), (configured, Some(Timestamp(10_500))));
        assert!(matches!(
            timeout_error(bound_by),
            PodWireError::DeadlineExceeded { sent: true, .. }
        ));
        // More time left than configured: the transport timeout binds.
        let (budget, bound_by) = budget_at(configured, Some(&binding), Timestamp(9_999)).unwrap();
        assert_eq!((budget, bound_by), (configured, None));
        // Less time left: the deadline binds.
        let (budget, bound_by) = budget_at(configured, Some(&binding), Timestamp(10_001)).unwrap();
        assert_eq!(
            (budget, bound_by),
            (Duration::from_millis(499), Some(Timestamp(10_500)))
        );
        // At or after the deadline nothing is sent.
        assert!(matches!(
            budget_at(configured, Some(&binding), Timestamp(10_500)),
            Err(PodWireError::DeadlineExceeded { sent: false, .. })
        ));
    }
}
