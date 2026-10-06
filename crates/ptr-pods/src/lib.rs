pub use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, TypeId};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

mod adapter;
mod attention;
mod cache;
#[cfg(feature = "candle-cuda")]
mod candle;
mod evidence;
mod expert;
mod hypothesis;
mod kv;
mod manifest;
mod native;
mod neural;
mod output;
mod paged_kv;
mod protocol;
mod qwen;
mod qwen_stack;
mod runtime;
mod semantic;
mod tier_storage;
mod transport;
mod turns;
pub use adapter::{AdapterError, NeuralPodAdapter};
pub use attention::{
    CausalAttentionInput, CausalAttentionOutput, CausalAttentionWeights, KvContinuationExecutor,
    ReferenceCausalAttentionExecutor,
};
pub use cache::{PodCache, PodCacheKey};
#[cfg(feature = "candle-cuda")]
pub use candle::{
    CandleDenseExecutor, CandleExecutorError, CandleGpuTierBackend, CandleKvCache,
    CandleKvSnapshotLease, CandleKvTensorBackend, CandlePagedKvBackend, CandlePagedKvCache,
};
pub use evidence::{
    Ed25519EvidenceSigner, Ed25519EvidenceVerifier, EvidenceError, EvidenceSigner,
    EvidenceVerifier, PodEvidenceBundle, ReplayedEvidence,
};
pub use expert::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertEvidence,
    ExpertIdentityError, ExpertLevel, ExpertOperationRequest, ExpertOutcome, ExpertPodSpec,
    ExpertProposal, ExpertProposalError, ExpertState, ExpertTarget, ExpertTargetInit,
    ManifestDigest, OpKind, OperandCount, OperationId, OutcomeKind, ProposalId, ProposalParts,
    ProposerRef, RationaleDigest, RequestedOp, ResultIntent, RetiredReason, StructuralOp,
    ZeroDigest, MAX_EXPERT_COMMIT_RECORD_BYTES,
};
pub use hypothesis::{merge_hypotheses, HypothesisError, MergedPodResult, PodHypothesis};
pub use kv::{
    InMemoryKvCache, InMemoryKvTensorBackend, KvBackendError, KvCacheLayout, KvCacheTier,
    KvKeyEncoding, KvLayerSnapshot, KvPageTable, KvTensorBackend, KvTensorDType, KvTensorSchema,
    KvTensorSnapshot, TensorRef,
};
pub use manifest::{
    ArtifactLifecycle, ExecutionManifest, LifecycleGate, LineageBinding,
    ManifestError as ExecutionManifestError,
};
pub use native::{TcpProtocolExecutor, UdpProtocolExecutor, MAX_NATIVE_FRAME_BYTES};
pub use neural::{
    DescriptorError, DeviceLease, DeviceLeaseError, DeviceLeaseState, LeaseState,
    NeuralPodDescriptor, NeuralPodError, NeuralPodExecutor, NeuralPodLease, NeuralPodType,
    PodLifecycle, ReferenceNeuralExecutor, ResourceRequirements, TensorContract, TensorDType,
};
pub use output::{OutputError, PodOutput, PodOutputKind};
pub use paged_kv::{
    InMemoryKvSnapshotLease, InMemoryPagedKvBackend, InMemoryPagedKvCache, KvForkAuthority,
    KvForkWitness, KvPageBinding, KvPageEntry, KvPageId, KvPageMetrics, KvPageSize, KvPageSnapshot,
    KvPageState, KvPagedSnapshot, KvPrefixKey, KvRuntimeLeaseBinding, KvSharedPrefix,
    KvWitnessRejection, KvWitnessRevoker, KvWitnessState, PagedKvTensorBackend,
};
pub use protocol::{
    ConnectionScope, DeliveryMode, EgressPolicy, MessagePattern, NativeProtocolExecutor,
    NativeProtocolRequest, NativeProtocolResponse, NetworkEndpoint, PodLink, PodLinkError,
    ProtocolBinding, ProtocolError, ProtocolProfile,
};
pub use qwen::{
    Qwen2Activation, Qwen2DecoderInput, Qwen2DecoderOutput, Qwen2DecoderWeights, Qwen2LayerConfig,
    Qwen2LinearWeights, Qwen2PagedDecoder, Qwen2RopeScaling,
};
pub use qwen_stack::{
    Qwen2PagedStack, Qwen2QkNorm, Qwen2StackConfig, Qwen2StackError, Qwen2StackInput,
    Qwen2StackLayerWeights, Qwen2StackOutput, Qwen2StackWeights,
};
pub use runtime::{
    ArtifactCatalog, ArtifactError, ExecutorError, ExecutorFactory, HealthStatus,
    InMemoryArtifactCatalog, InMemoryResourceGovernor, ReferenceExecutorFactory, ResourceError,
    ResourceGovernor, ResourceLease,
};
pub use semantic::{
    ManifestError, ModelVariant, PodKind, PodResourceProfile, PodSemanticManifest,
    SemanticPodLifecycle,
};
pub use tier_storage::{
    ArtifactTierAdapter, ArtifactTierBinding, KvTierAdapter, KvTierBinding, ModelWeightTierAdapter,
    ModelWeightTierBinding, PagedKvTierAdapter, PodTierError,
};
pub use transport::{PodWireRequest, PodWireResponse, TransportError};
pub use turns::{DuplexSession, PodTurnEvent, PodTurnKind, TurnError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodManifest {
    /// The one project this Pod serves. A Pod that served every project would
    /// make the project boundary advisory, since resolution is the only thing
    /// standing between a request and a Pod's data.
    pub project: ProjectId,
    pub id: PodId,
    pub capabilities: Vec<CapabilityId>,
    pub accepts: Vec<TypeId>,
    pub produces: Vec<TypeId>,
    pub effects: Vec<Effect>,
    pub protocol_version: u32,
}

impl PodManifest {
    /// Canonical digest used to bind descriptors and transport requests to the
    /// exact invocation contract, including project and effect boundaries.
    pub fn digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        macro_rules! push {
            ($value:expr) => {{
                let value = $value.to_string();
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value.as_bytes());
            }};
        }
        push!(self.project.0);
        push!(self.id.0);
        for value in &self.capabilities {
            push!(value.0);
        }
        for value in &self.accepts {
            push!(value.0);
        }
        for value in &self.produces {
            push!(value.0);
        }
        for value in &self.effects {
            push!(format!("{:?}", value));
        }
        push!(self.protocol_version);
        Sha256::digest(bytes).into()
    }
}

pub trait Pod {
    type Input;
    type Output;
    type Error;
    fn manifest(&self) -> &PodManifest;
    fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
}

pub trait DynPod: Send + Sync {
    fn manifest(&self) -> &PodManifest;
    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisteredPodBinding {
    project: ProjectId,
    pod_id: PodId,
    manifest: PodManifest,
}

/// A non-replacing registration found the pod id already registered.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PodRegistrationError {
    AlreadyRegistered {
        project: ProjectId,
        pod: PodId,
    },
    /// A compare-and-replace found no entry for the key.
    NotRegistered {
        project: ProjectId,
        pod: PodId,
    },
    /// A compare-and-replace found an entry other than the expected one.
    NotCurrent {
        project: ProjectId,
        pod: PodId,
    },
    /// A compare-and-replace named a new pod whose `(project, id)` differs from the expected
    /// pod's.
    IdentityMismatch {
        expected: (ProjectId, PodId),
        found: (ProjectId, PodId),
    },
}

impl std::fmt::Display for PodRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRegistered { project, pod } => {
                write!(f, "pod {pod} is already registered in project {project}")
            }
            Self::NotRegistered { project, pod } => {
                write!(f, "pod {pod} is not registered in project {project}")
            }
            Self::NotCurrent { project, pod } => write!(
                f,
                "registered pod {pod} in project {project} is not the expected entry"
            ),
            Self::IdentityMismatch { expected, found } => write!(
                f,
                "replacement is pod {} in project {}, expected pod {} in project {}",
                found.1, found.0, expected.1, expected.0
            ),
        }
    }
}

impl std::error::Error for PodRegistrationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodBindingError {
    NotFound { project: ProjectId, pod: PodId },
}

impl RegisteredPodBinding {
    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn pod_id(&self) -> &PodId {
        &self.pod_id
    }

    pub fn manifest(&self) -> &PodManifest {
        &self.manifest
    }
}

/// Pods addressed by project *and* id.
///
/// The project is part of the key rather than a filter applied afterwards, so
/// two projects may each register an `echo` Pod without one shadowing the
/// other, and no lookup path exists that forgets to check it.
#[derive(Default)]
pub struct PodRegistry {
    pods: BTreeMap<(ProjectId, PodId), Arc<dyn DynPod>>,
}

impl PodRegistry {
    /// Insert or REPLACE the pod with this `(project, id)`, returning the previous entry.
    ///
    /// Structural-admission paths (a birth, merge, split or reactivation result becoming a
    /// registered pod) must use [`PodRegistry::register_new`] instead: replacing a live pod
    /// silently would swap an expert without any lifecycle transition.
    pub fn register(&mut self, pod: Arc<dyn DynPod>) -> Option<Arc<dyn DynPod>> {
        let manifest = pod.manifest();
        let key = (manifest.project.clone(), manifest.id.clone());
        self.pods.insert(key, pod)
    }

    /// Insert a pod without replacing: fails with
    /// [`PodRegistrationError::AlreadyRegistered`] if its `(project, id)` is taken, leaving the
    /// existing entry unchanged. Structural admission of a fresh id (birth, merge or split
    /// result) must use this, not [`PodRegistry::register`]; a takeover of an operand of the
    /// same operation (a retired-target reactivation puts generation g+1 on the same pod id)
    /// uses [`PodRegistry::replace_if_current`]. The registry tracks no token-expert `(layer, slot)`, so slot
    /// ownership is not checked here.
    pub fn register_new(&mut self, pod: Arc<dyn DynPod>) -> Result<(), PodRegistrationError> {
        let manifest = pod.manifest();
        let key = (manifest.project.clone(), manifest.id.clone());
        match self.pods.entry(key) {
            std::collections::btree_map::Entry::Occupied(entry) => {
                let (project, pod) = entry.key().clone();
                Err(PodRegistrationError::AlreadyRegistered { project, pod })
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(pod);
                Ok(())
            }
        }
    }

    /// Compare-and-replace: swap in `new` only if the entry for its `(project, id)` is exactly
    /// `expected` (compared with [`Arc::ptr_eq`]), returning the old entry. This is the atomic
    /// takeover for structural admission of an operand of the same operation; `register` is not
    /// for structural admission.
    ///
    /// Fails, leaving the registry unchanged, with
    /// [`PodRegistrationError::IdentityMismatch`] if `new` and `expected` differ in project or
    /// id, [`PodRegistrationError::NotRegistered`] if the key is absent, or
    /// [`PodRegistrationError::NotCurrent`] if another entry replaced `expected` meanwhile.
    pub fn replace_if_current(
        &mut self,
        expected: &Arc<dyn DynPod>,
        new: Arc<dyn DynPod>,
    ) -> Result<Arc<dyn DynPod>, PodRegistrationError> {
        let key_of = |pod: &Arc<dyn DynPod>| {
            let manifest = pod.manifest();
            (manifest.project.clone(), manifest.id.clone())
        };
        let expected_key = key_of(expected);
        let new_key = key_of(&new);
        if expected_key != new_key {
            return Err(PodRegistrationError::IdentityMismatch {
                expected: expected_key,
                found: new_key,
            });
        }
        let (project, pod) = expected_key;
        match self.pods.get_mut(&(project.clone(), pod.clone())) {
            None => Err(PodRegistrationError::NotRegistered { project, pod }),
            Some(current) if Arc::ptr_eq(current, expected) => Ok(std::mem::replace(current, new)),
            Some(_) => Err(PodRegistrationError::NotCurrent { project, pod }),
        }
    }

    pub fn get(&self, project: &ProjectId, id: &PodId) -> Option<Arc<dyn DynPod>> {
        self.pods.get(&(project.clone(), id.clone())).cloned()
    }

    pub fn bind_registered(
        &self,
        project: &ProjectId,
        pod: &PodId,
    ) -> Result<RegisteredPodBinding, PodBindingError> {
        let resolved = self
            .get(project, pod)
            .ok_or_else(|| PodBindingError::NotFound {
                project: project.clone(),
                pod: pod.clone(),
            })?;
        Ok(RegisteredPodBinding {
            project: project.clone(),
            pod_id: pod.clone(),
            manifest: resolved.manifest().clone(),
        })
    }

    pub fn resolve_bound(&self, binding: &RegisteredPodBinding) -> Option<Arc<dyn DynPod>> {
        let pod = self.get(&binding.project, &binding.pod_id)?;
        (pod.manifest() == &binding.manifest).then_some(pod)
    }

    /// Resolve within one project only.
    ///
    /// A Pod belonging to another project is not a worse match here, it is not a
    /// match at all: the caller learns nothing about whether it exists.
    pub fn resolve(
        &self,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Option<Arc<dyn DynPod>> {
        self.pods
            .iter()
            .find(|((pod_project, _), pod)| {
                let manifest = pod.manifest();
                pod_project == project
                    && manifest.capabilities.contains(capability)
                    && manifest.accepts.contains(input_type)
            })
            .map(|(_, pod)| pod.clone())
    }

    /// Return only project-owned candidates, in canonical PodId order.
    pub fn candidates(
        &self,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Vec<PodId> {
        self.pods
            .iter()
            .filter(|((pod_project, _), pod)| {
                let manifest = pod.manifest();
                pod_project == project
                    && manifest.capabilities.contains(capability)
                    && manifest.accepts.contains(input_type)
            })
            .map(|((_, pod_id), _)| pod_id.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.pods.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pods.is_empty()
    }
}

pub struct Ready;
pub struct Revoked;
pub struct PodLease<S> {
    pub pod_id: PodId,
    _state: PhantomData<S>,
}

impl PodLease<Ready> {
    pub fn new(pod_id: PodId) -> Self {
        Self {
            pod_id,
            _state: PhantomData,
        }
    }

    pub fn revoke(self) -> PodLease<Revoked> {
        PodLease {
            pod_id: self.pod_id,
            _state: PhantomData,
        }
    }

    pub fn can_invoke(&self) -> bool {
        true
    }
}

impl PodLease<Revoked> {
    pub fn can_invoke(&self) -> bool {
        false
    }
}

#[derive(Debug, PartialEq)]
pub enum LeaseInvokeError<E> {
    WrongPod,
    Pod(E),
}

/// Invoke a Pod only with a lease whose typestate is Ready.
///
/// A revoked lease cannot be passed to this function.
///
/// ~~~compile_fail
/// use ptr_pods::{invoke_with_lease, Pod, PodLease, PodManifest, Ready};
/// use ptr_types::{PodId, ProjectId};
///
/// struct Echo { manifest: PodManifest }
/// impl Pod for Echo {
///     type Input = String;
///     type Output = String;
///     type Error = ();
///     fn manifest(&self) -> &PodManifest { &self.manifest }
///     fn invoke(&self, input: String) -> Result<String, ()> { Ok(input) }
/// }
///
/// # let manifest = PodManifest {
/// #   project: ProjectId::from("p"), id: PodId::from("echo"),
/// #   capabilities: vec![], accepts: vec![],
/// #   produces: vec![], effects: vec![], protocol_version: 1
/// # };
/// let pod = Echo { manifest };
/// let ready: PodLease<Ready> = PodLease::new(PodId::from("echo"));
/// let revoked = ready.revoke();
/// let _ = invoke_with_lease(&pod, &revoked, "x".to_string());
/// ~~~
pub fn invoke_with_lease<P: Pod>(
    pod: &P,
    lease: &PodLease<Ready>,
    input: P::Input,
) -> Result<P::Output, LeaseInvokeError<P::Error>> {
    if lease.pod_id != pod.manifest().id {
        return Err(LeaseInvokeError::WrongPod);
    }
    pod.invoke(input).map_err(LeaseInvokeError::Pod)
}
