//! The service behind `POST /v1/vcw/tasks`: construction from the `[vcw]` configuration and the
//! one-request pipeline.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use ptr_config::{
    VcwBackend, VcwHintMode, VcwLearning, VcwServiceConfig, VcwTaskKind, MAX_VCW_REQUEST_MS,
};
use ptr_ledger::LedgerEvent;
use ptr_pods::{
    ExecutionManifest, LineageBinding, PodOutput, PodOutputKind, PodRegistrationError, PodRegistry,
};
use ptr_protocol::TypedPayload;
use ptr_router::{
    AttentionSchema, Exploration, ExplorationError, NonceSequence, PodRef, RepairRoleAssignments,
    RouterConfig, SamplingKey, SchemaLimits, SelfModel, SelfModelLimits, UniformScorer,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::semantic::request_raw_key;
use ptr_runtime::vcw::experts::{
    ExpertCall, ExpertRole, LlmExpertPods, Recoded, SimulatedExperts, SimulationConfig,
    SimulationError, TextCodec,
};
use ptr_runtime::vcw::{
    admit_verified_answer, BudgetError, EscalateCall, HintMode, ProposeCall, RepairCall,
    RequestBudget, RolePods, Roster, RuntimeStanding, VcwAdmitError, VcwConfig, VcwError, VcwLoop,
    VcwOutcome, VcwParts, VcwRequest,
};
use ptr_runtime::{
    CleanupError, ExecutionScope, FencingToken, PlacementEpoch, PodOutputAdmissionRequest,
    PtrRuntime, ScopeCleanupCoordinator, ScopeState,
};
use ptr_types::{
    CapsuleId, Digest, EvidenceId, Generation, PodId, PrincipalId, ProjectId, ProvenanceRef,
    RequestId, Revision, ScopeId, ScopeLeaseBinding, SessionId, Timestamp, VerificationLevel,
};
use ptr_verifier::{code_spec, RepairRole};
use sha2::{Digest as _, Sha256};

use super::job::{answer_type, Job};
use super::pods::ServedPod;
use super::report::{LedgerSummary, TraceStore};
use super::shadow::{LearningReport, Shadow};
use super::sim::{plantable, ServedDefects, SimCodec};
use super::wire::{
    VcwApiError, VcwAuthor, VcwCommitted, VcwResponse, VcwSpent, VcwTask, VcwTaskRequest,
    VcwUnanswered, VcwVerified,
};

/// The secret and the nonce space of the routers of the path. The key is the router's secret and
/// must not be the key of any other router; `nonce_instance` must be fresh for every process
/// incarnation sharing the key.
pub struct VcwServiceParts {
    key: [u8; 32],
    nonce_instance: [u8; 8],
}

impl VcwServiceParts {
    #[must_use]
    pub fn new(key: [u8; 32], nonce_instance: [u8; 8]) -> Self {
        Self {
            key,
            nonce_instance,
        }
    }
}

impl fmt::Debug for VcwServiceParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VcwServiceParts").finish_non_exhaustive()
    }
}

/// Why the service could not be built or installed.
#[derive(Debug)]
#[non_exhaustive]
pub enum VcwBuildError {
    /// The configuration is invalid or not servable.
    Config(String),
    Budget(BudgetError),
    Exploration(ExplorationError),
    Simulation(SimulationError),
    /// A configured finding code is not one the path can plant or repair.
    UnknownCode {
        field: &'static str,
        code: String,
    },
    Pods(String),
    Registry(PodRegistrationError),
    Loop(VcwError),
    /// The `qwen-cpu` backend was selected in a build without the `qwen-cpu` feature.
    QwenNotCompiled,
    /// The model did not load (a digest mismatch is one reason).
    Qwen(String),
    /// Putting the Pods or the scope into the runtime failed.
    Install(String),
    /// A configured Pod id is already a capsule of another project in the lifecycle authority.
    PodIdCollision {
        pod: String,
        project: String,
        existing_project: String,
    },
    /// A configured generation differs from the live generation of a Pod the lifecycle authority
    /// already knows.
    GenerationMismatch {
        pod: String,
        configured: u64,
        live: u64,
    },
    /// The scope left by an earlier incarnation holds a lease binding or children, so it cannot
    /// be recovered as one that holds nothing.
    ScopeNotEmpty {
        scope: String,
    },
}

impl fmt::Display for VcwBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(why) => write!(f, "vcw configuration: {why}"),
            Self::Budget(error) => write!(f, "vcw budget: {error:?}"),
            Self::Exploration(error) => write!(f, "vcw exploration: {error}"),
            Self::Simulation(error) => write!(f, "vcw simulation: {error}"),
            Self::UnknownCode { field, code } => {
                write!(f, "{field} names {code}, which the vcw path cannot use")
            }
            Self::Pods(why) => write!(f, "vcw Pods: {why}"),
            Self::Registry(error) => write!(f, "vcw Pod registry: {error:?}"),
            Self::Loop(error) => write!(f, "vcw loop: {error}"),
            Self::QwenNotCompiled => f.write_str(
                "vcw.experts.backend is qwen-cpu but this build lacks the qwen-cpu feature",
            ),
            Self::Qwen(why) => write!(f, "vcw model: {why}"),
            Self::Install(why) => write!(f, "vcw install: {why}"),
            Self::PodIdCollision {
                pod,
                project,
                existing_project,
            } => write!(
                f,
                "vcw install: Pod id {pod} is already a capsule of project {existing_project}, \
                 not of the vcw project {project}; choose another id under [vcw.pods]"
            ),
            Self::GenerationMismatch {
                pod,
                configured,
                live,
            } => write!(
                f,
                "vcw install: [vcw.pods] generation is {configured} but Pod {pod} is live at \
                 generation {live} in the lifecycle authority; set generation = {live} (the \
                 service would otherwise find no eligible Pod for any request)"
            ),
            Self::ScopeNotEmpty { scope } => write!(
                f,
                "vcw install: scope {scope} of an earlier incarnation holds a lease binding or \
                 child scopes; it was not made by this service and is not recovered with a no-op \
                 cleanup"
            ),
        }
    }
}

impl std::error::Error for VcwBuildError {}

pub(super) enum Experts {
    Simulated(SimulatedExperts<ServedDefects>),
    /// Built only by the `qwen-cpu` backend.
    #[cfg_attr(not(feature = "qwen-cpu"), allow(dead_code))]
    Llm(LlmExpertPods),
}

struct KindState {
    vcw: VcwLoop,
    experts: Experts,
    /// Present only with `[vcw] learning = "shadow"`.
    shadow: Option<Shadow>,
}

/// The verifier-closed loop as a service: one loop and one expert set per served task kind, the
/// Pods in a registry of their own, and the scope the answers are admitted under.
pub struct VcwService {
    project: ProjectId,
    /// The scope of this incarnation, set by [`VcwService::install`].
    scope_id: OnceLock<ScopeId>,
    session_id: SessionId,
    backend: VcwBackend,
    /// What the manifests of the admissions say about the experts' origin.
    backend_origins: Vec<String>,
    budget: RequestBudget,
    request_deadline: Duration,
    pods: PodRegistry,
    base_pods: Vec<(PodId, Generation)>,
    kinds: BTreeMap<VcwTaskKind, Mutex<KindState>>,
    /// The traces of the last requests, for `GET /v1/vcw/traces/{id}`.
    traces: Mutex<TraceStore>,
}

/// The scope holds no lease, session or queue outside the runtime, so cleanup has nothing to do.
struct NoExternalResources;

impl ScopeCleanupCoordinator for NoExternalResources {
    fn stop_intake(&mut self, _: &ExecutionScope) -> Result<(), CleanupError> {
        Ok(())
    }
    fn cancel_children(&mut self, _: &ExecutionScope) -> Result<(), CleanupError> {
        Ok(())
    }
    fn drain_queues(&mut self, _: &ExecutionScope) -> Result<(), CleanupError> {
        Ok(())
    }
    fn release_pod_lease(&mut self, _: &ExecutionScope) -> Result<(), CleanupError> {
        Ok(())
    }
    fn release_resource_lease(&mut self, _: &ExecutionScope) -> Result<(), CleanupError> {
        Ok(())
    }
    fn close_session(&mut self, _: &ExecutionScope) -> Result<(), CleanupError> {
        Ok(())
    }
}

impl fmt::Debug for VcwService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VcwService")
            .field("project", &self.project)
            .field("backend", &self.backend)
            .field("kinds", &self.kinds())
            .field("pods", &self.base_pods)
            .finish_non_exhaustive()
    }
}

fn task_kind(task: &VcwTask) -> VcwTaskKind {
    match task {
        VcwTask::Arithmetic { .. } => VcwTaskKind::Arithmetic,
        VcwTask::JsonObject { .. } => VcwTaskKind::JsonObject,
        VcwTask::Code { .. } => VcwTaskKind::Code,
        VcwTask::Fact { .. } => VcwTaskKind::Fact,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256(parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// The name of a backend in manifests, capabilities and responses.
fn backend_tag(backend: VcwBackend) -> &'static str {
    match backend {
        VcwBackend::Simulated => "simulated",
        VcwBackend::QwenCpu => "qwen-cpu",
    }
}

/// What every admission's manifest says about where the candidates come from: the backend and,
/// for the model, the digests of its four files (verified when the model was loaded).
fn backend_origins(config: &VcwServiceConfig, backend: VcwBackend) -> Vec<String> {
    let mut origins = vec![format!("vcw:backend:{}", backend_tag(backend))];
    if backend == VcwBackend::QwenCpu {
        for (field, digest) in config.experts.qwen_cpu.digests() {
            let name = field.rsplit('.').next().unwrap_or(field);
            origins.push(format!("vcw:model:{name}:{digest}"));
        }
    }
    origins
}

fn simulation_config(
    config: &VcwServiceConfig,
    kind: VcwTaskKind,
) -> Result<SimulationConfig, VcwBuildError> {
    let sim = &config.experts.simulated;
    let hard = |field: &'static str, code: &str| {
        if code_spec(code).is_some_and(|spec| spec.hard) {
            Ok(())
        } else {
            Err(VcwBuildError::UnknownCode {
                field,
                code: code.to_owned(),
            })
        }
    };
    let every_plantable: Vec<&str> = VcwTaskKind::ALL
        .iter()
        .flat_map(|kind| plantable(*kind).iter().copied())
        .collect();
    let mut out = SimulationConfig::new(u64::from(sim.seed));
    for (code, rate) in &sim.error_rates {
        if !every_plantable.contains(&code.as_str()) {
            return Err(VcwBuildError::UnknownCode {
                field: "vcw.experts.simulated.error_rates",
                code: code.clone(),
            });
        }
        // A code of another task kind is validated above and planted only in its own kind.
        if plantable(kind).contains(&code.as_str()) {
            out.error_rates.insert(code.clone(), *rate);
        }
    }
    for (code, rate) in &sim.repair_success {
        hard("vcw.experts.simulated.repair_success", code)?;
        out.repair_success.insert(code.clone(), *rate);
    }
    for (pod, rates) in &sim.pod_repair_success {
        for (code, rate) in rates {
            hard("vcw.experts.simulated.pod_repair_success", code)?;
            out.pod_repair_success
                .insert((pod.clone(), code.clone()), *rate);
        }
    }
    out.default_repair_success = sim.default_repair_success;
    out.decline_on_failure = sim.decline_on_failure;
    Ok(out)
}

/// The repair role named in `[vcw.pods.extra_repairers]`; the names are those of
/// `ptr_config::VCW_REPAIR_ROLES`, which `validate` enforces. The two lists are in the same order
/// (checked by a unit test), so the position is the mapping and nothing can fall through.
fn repair_role(name: &str) -> Result<RepairRole, VcwBuildError> {
    ptr_config::VCW_REPAIR_ROLES
        .iter()
        .position(|candidate| *candidate == name)
        .map(|index| RepairRole::ALL[index])
        .ok_or_else(|| {
            VcwBuildError::Config(format!(
                "vcw.pods.extra_repairers names the unknown role {name:?}"
            ))
        })
}

impl VcwService {
    /// Build the service from the `[vcw]` section.
    ///
    /// # Errors
    /// [`VcwBuildError`] for an invalid or unservable configuration, a rate or code the path
    /// cannot use, or a model that does not load (a digest mismatch included).
    pub fn from_config(
        config: &VcwServiceConfig,
        parts: &VcwServiceParts,
    ) -> Result<Self, VcwBuildError> {
        config.validate().map_err(VcwBuildError::Config)?;
        if !config.enabled {
            return Err(VcwBuildError::Config("vcw is not enabled".into()));
        }
        let project = ProjectId::from(config.project.as_str());
        let backend = config.experts.backend.ok_or_else(|| {
            VcwBuildError::Config("vcw.experts.backend must be set explicitly".into())
        })?;
        let budget = RequestBudget::new(config.max_verifier_calls, config.max_rounds)
            .map_err(VcwBuildError::Budget)?;
        let exploration =
            Exploration::new(config.exploration_floor).map_err(VcwBuildError::Exploration)?;
        let generation = Generation(config.pods.generation);

        let proposer = PodId::from(config.pods.proposer.as_str());
        let assignments = [
            (RepairRole::Format, &config.pods.format_repair),
            (RepairRole::Value, &config.pods.value_repair),
            (RepairRole::Schema, &config.pods.schema_repair),
            (RepairRole::Code, &config.pods.code_repair),
            (RepairRole::Resource, &config.pods.resource_repair),
            (RepairRole::Fact, &config.pods.fact_repair),
        ]
        .into_iter()
        .fold(RepairRoleAssignments::new(), |roles, (role, pod)| {
            roles.assign(role, PodId::from(pod.as_str()))
        });
        // Further repairers by role name; a Pod under several roles covers all their codes.
        let mut assignments = assignments;
        for (role, pods) in &config.pods.extra_repairers {
            let role = repair_role(role)?;
            for pod in pods {
                assignments = assignments.assign(role, PodId::from(pod.as_str()));
            }
        }
        let repairers = assignments.repairers();
        let repairer_codes: Vec<(PodId, Vec<String>)> = repairers
            .iter()
            .map(|(pod, codes)| {
                (
                    pod.clone(),
                    codes.iter().map(|c| c.as_str().to_owned()).collect(),
                )
            })
            .collect();

        let kinds_served: Vec<VcwTaskKind> = VcwTaskKind::ALL
            .iter()
            .copied()
            .filter(|kind| config.serves(*kind))
            .collect();
        let types: Vec<_> = kinds_served.iter().map(|kind| answer_type(*kind)).collect();

        // The experts of each served kind, and the model identity (if any) for the manifests.
        let mut experts: BTreeMap<VcwTaskKind, Experts> = BTreeMap::new();
        let mut identity = None;
        match backend {
            VcwBackend::Simulated => {
                for kind in &kinds_served {
                    let mut sim = SimulatedExperts::<ServedDefects>::new(
                        simulation_config(config, *kind)?,
                        "{}",
                    )
                    .map_err(VcwBuildError::Simulation)?;
                    sim.add_proposer(proposer.clone())
                        .map_err(|error| VcwBuildError::Pods(error.to_string()))?;
                    for (pod, codes) in &repairer_codes {
                        let codes: Vec<&str> = codes.iter().map(String::as_str).collect();
                        sim.add_repairer(pod.clone(), &codes)
                            .map_err(|error| VcwBuildError::Pods(error.to_string()))?;
                    }
                    sim.validate_roster().map_err(VcwBuildError::Simulation)?;
                    experts.insert(*kind, Experts::Simulated(sim));
                }
            }
            VcwBackend::QwenCpu => {
                let (built, model) =
                    llm_experts(config, &kinds_served, &project, &proposer, &repairer_codes)?;
                experts = built;
                identity = model;
            }
        }

        // The Pod registry: one entry per Pod, accepting and producing every served type.
        let mut pods = PodRegistry::default();
        let mut base_pods = Vec::new();
        let roles = std::iter::once((proposer.clone(), ExpertRole::Proposer)).chain(
            repairer_codes
                .iter()
                .map(|(pod, _)| (pod.clone(), ExpertRole::Repairer)),
        );
        for (pod, role) in roles {
            let entry = ServedPod::new(
                &project,
                &pod,
                role,
                &types,
                identity.as_ref(),
                backend_tag(backend),
            );
            pods.register_new(entry.into_dyn())
                .map_err(VcwBuildError::Registry)?;
            base_pods.push((pod, generation));
        }

        // One loop per served kind, each with its own nonce space under the shared key.
        let mut kinds = BTreeMap::new();
        for (index, kind) in kinds_served.iter().enumerate() {
            let experts = experts
                .remove(kind)
                .ok_or_else(|| VcwBuildError::Pods("no experts for a served kind".into()))?;
            let roster = match &experts {
                Experts::Simulated(sim) => sim.roster().map_err(VcwBuildError::Simulation)?,
                Experts::Llm(llm) => llm.roster(),
            };
            let vcw = build_loop(config, *kind, index, roster, exploration, parts)?;
            let (vcw, shadow) = match config.learning {
                VcwLearning::Off => (vcw, None),
                VcwLearning::Shadow => {
                    if exploration.epsilon() <= 0.0 {
                        return Err(VcwBuildError::Config(
                            "learning = \"shadow\" needs exploration_floor above 0".into(),
                        ));
                    }
                    let seed = sha256(&[b"vcw-shadow", &parts.key, kind.as_str().as_bytes()]);
                    let mut head = [0_u8; 8];
                    head.copy_from_slice(&seed[..8]);
                    let shadow = Shadow::new(*kind, u64::from_le_bytes(head), exploration)
                        .map_err(VcwBuildError::Config)?;
                    let vcw = vcw
                        .with_learner(shadow.learner())
                        .map_err(VcwBuildError::Loop)?;
                    (vcw, Some(shadow))
                }
            };
            kinds.insert(
                *kind,
                Mutex::new(KindState {
                    vcw,
                    experts,
                    shadow,
                }),
            );
        }

        Ok(Self {
            scope_id: OnceLock::new(),
            session_id: SessionId::from("vcw"),
            project,
            backend,
            backend_origins: backend_origins(config, backend),
            budget,
            request_deadline: Duration::from_millis(u64::from(config.max_request_ms)),
            pods,
            base_pods,
            kinds,
            traces: Mutex::new(TraceStore::default()),
        })
    }

    /// Replace the wall-clock deadline of one request (at most the configured maximum). For
    /// hosts and tests that need a tighter bound than `max_request_ms` can express.
    #[must_use]
    pub fn with_request_deadline(mut self, deadline: Duration) -> Self {
        self.request_deadline = deadline.min(Duration::from_millis(u64::from(MAX_VCW_REQUEST_MS)));
        self
    }

    /// The capabilities in the manifest of a served Pod (`propose` or `repair`, and
    /// `backend:<name>`, and for a model one `model:<file>:sha256:<hex>` per file), if it is one.
    #[must_use]
    pub fn pod_capabilities(&self, pod: &PodId) -> Option<Vec<String>> {
        let entry = self.pods.get(&self.project, pod)?;
        Some(
            entry
                .manifest()
                .capabilities
                .iter()
                .map(|capability| capability.0.clone())
                .collect(),
        )
    }

    /// The backend of the experts.
    #[must_use]
    pub fn backend(&self) -> VcwBackend {
        self.backend
    }

    /// Make the Pods live in the runtime's lifecycle authority (only those it does not know yet:
    /// a revoked or superseded Pod stays so across restarts) and open the answers' scope (see
    /// `open_scope` for restarts).
    ///
    /// A Pod id that is a capsule of another project is refused
    /// ([`VcwBuildError::PodIdCollision`]), and so is a configured generation that differs from
    /// the live generation of a Pod the lifecycle authority knows
    /// ([`VcwBuildError::GenerationMismatch`]; a Pod revoked at the configured generation passes
    /// this check and stays revoked).
    ///
    /// # Errors
    /// [`VcwBuildError`]: a collision, a generation mismatch, a scope that cannot be recovered,
    /// or a commit the runtime refuses.
    pub fn install(&self, runtime: &mut PtrRuntime) -> Result<(), VcwBuildError> {
        for (pod, generation) in &self.base_pods {
            if let Some(existing) = runtime.capsule_project(&pod.0) {
                if existing != self.project.0 {
                    return Err(VcwBuildError::PodIdCollision {
                        pod: pod.0.clone(),
                        project: self.project.0.clone(),
                        existing_project: existing.to_owned(),
                    });
                }
            }
            if let Some(live) = runtime.live_generation(&pod.0) {
                if live != *generation {
                    return Err(VcwBuildError::GenerationMismatch {
                        pod: pod.0.clone(),
                        configured: generation.0,
                        live: live.0,
                    });
                }
            }
            let known = runtime.generation_validity(&pod.0, *generation).is_some()
                || runtime.live_generation(&pod.0).is_some();
            if !known {
                runtime
                    .commit(LedgerEvent::CapsuleCommitted {
                        project: self.project.clone(),
                        capsule: CapsuleId::from(pod.0.as_str()),
                        generation: *generation,
                    })
                    .map_err(|error| VcwBuildError::Install(format!("{error:?}")))?;
            }
        }
        let scope_id = self.open_scope(runtime)?;
        self.scope_id
            .set(scope_id)
            .map_err(|_| VcwBuildError::Install("the vcw service is installed twice".into()))?;
        Ok(())
    }

    /// The scope answers are admitted under. A runtime that restarts marks every scope that was
    /// not released as needing recovery, and such a scope refuses work, so a scope left by an
    /// earlier incarnation is recovered (it holds no external resource, so there is nothing to
    /// release) and a new one, numbered after it, is created.
    fn open_scope(&self, runtime: &mut PtrRuntime) -> Result<ScopeId, VcwBuildError> {
        let install = |error: &dyn fmt::Debug| VcwBuildError::Install(format!("{error:?}"));
        for incarnation in 0_u32..100_000 {
            let name = if incarnation == 0 {
                format!("vcw-scope-{}", self.project.0)
            } else {
                format!("vcw-scope-{}-{incarnation}", self.project.0)
            };
            let id = ScopeId::from(name.as_str());
            match runtime.scopes().get(&id) {
                Err(_) => {
                    runtime
                        .create_scope(
                            ExecutionScope {
                                id: id.clone(),
                                parent: None,
                                session: self.session_id.clone(),
                                project: self.project.clone(),
                                created_at: Timestamp(1),
                                deadline: None,
                                state: ScopeState::Created,
                                cancellation_requested: false,
                            },
                            ScopeLeaseBinding::default(),
                        )
                        .map_err(|error| install(&error))?;
                    return Ok(id);
                }
                Ok(scope) if scope.state == ScopeState::Released => {}
                Ok(_) if runtime.scopes().recovery_required(&id) => {
                    let lease = runtime
                        .scopes()
                        .lease(&id)
                        .map_err(|error| install(&error))?
                        .clone();
                    // The no-op coordinator is right only for a scope that holds nothing: this
                    // service attaches no lease and opens no child. Anything else was not made by
                    // this service, and declaring its cleanup done would skip real cleanup.
                    if lease != ScopeLeaseBinding::default()
                        || !runtime.scopes().children(&id).is_empty()
                    {
                        return Err(VcwBuildError::ScopeNotEmpty { scope: name });
                    }
                    runtime
                        .recover_scope(&id, &mut NoExternalResources, lease)
                        .map_err(|error| install(&error))?;
                }
                Ok(scope) if scope.state.is_terminal() => {}
                Ok(_) => return Ok(id),
            }
        }
        Err(VcwBuildError::Install(
            "no unused scope name for the vcw service".into(),
        ))
    }

    /// The Pods of the path with the generation each was declared at.
    #[must_use]
    pub fn pods(&self) -> &[(PodId, Generation)] {
        &self.base_pods
    }

    /// The project of the scope, the Pods and the committed answers.
    #[must_use]
    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    /// The task kinds served.
    #[must_use]
    pub fn kinds(&self) -> Vec<VcwTaskKind> {
        self.kinds.keys().copied().collect()
    }

    /// What each kind has learned in shadow mode (empty with `learning = "off"`). The learned
    /// candidates never route; this is the only place they are visible.
    #[must_use]
    pub fn learning_report(&self) -> Vec<LearningReport> {
        self.kinds
            .values()
            .filter_map(|state| {
                let state = state.lock().ok()?;
                state.shadow.as_ref().map(Shadow::report)
            })
            .collect()
    }

    /// The calls the expert Pods answered or refused, oldest first within each kind (at most
    /// `MAX_LOGGED_CALLS` per kind), for audit and tests.
    #[must_use]
    pub fn expert_calls(&self) -> Vec<ExpertCall> {
        let mut out = Vec::new();
        for state in self.kinds.values() {
            let Ok(state) = state.lock() else { continue };
            match &state.experts {
                Experts::Simulated(sim) => out.extend(sim.calls().iter().cloned()),
                Experts::Llm(llm) => out.extend(llm.calls().iter().cloned()),
            }
        }
        out
    }

    /// Run one task to a typed outcome, holding the runtime for the duration.
    ///
    /// A request id is consumed once its raw text is committed: whatever happens afterwards (an
    /// answer, no answer, a runtime or internal error) the id answers 409 on a retry, so a
    /// client that wants another attempt sends a new id. Every request that got that far records
    /// exactly one `RuntimeEvent::VcwFinished`.
    ///
    /// The request is bounded in wall-clock time by the configured deadline, checked before each
    /// Pod call: after it the loop is given no further candidate and the request ends
    /// `Exhausted` with reason `Deadline`. The runtime lock is therefore held for at most the
    /// deadline plus the step in flight (one model call of at most 128 new tokens per Pod called
    /// in the step, or one verifier run, which for code is at most 32 tests of 100 000 fuel).
    ///
    /// # Errors
    /// [`VcwApiError`]: invalid input, a reused request id, a runtime refusal, or an answer the
    /// runtime did not admit. An `Exhausted` or `Escalated` request is an `Ok` outcome.
    pub fn run_task(
        &self,
        runtime: &mut PtrRuntime,
        input: VcwTaskRequest,
    ) -> Result<VcwResponse, VcwApiError> {
        let started = Instant::now();
        let VcwTaskRequest { id, task } = input;
        if id.trim().is_empty() || id.chars().any(char::is_control) {
            return Err(VcwApiError::BadRequest(
                "id must be non-empty without control characters".into(),
            ));
        }
        if id.len() > crate::MAX_REQUEST_ID_BYTES {
            return Err(VcwApiError::TooLarge(format!(
                "id is longer than {} bytes",
                crate::MAX_REQUEST_ID_BYTES
            )));
        }
        let kind = task_kind(&task);
        let Some(slot) = self.kinds.get(&kind) else {
            return Err(VcwApiError::BadRequest(format!(
                "task kind {} is not served",
                kind.as_str()
            )));
        };
        let job = Job::from_request(&task, self.backend)?;
        let request_id = RequestId::from(id.as_str());
        if runtime
            .snapshot()
            .value(&request_raw_key(&request_id))
            .is_some()
        {
            return Err(VcwApiError::Conflict(format!(
                "request id {id} was used before"
            )));
        }
        let first_event = runtime.committed_events().len();
        // The raw task text is evidence and is kept whatever the outcome; an answer depends on it.
        runtime
            .ingest_text(request_id.clone(), job.statement.clone())
            .map_err(VcwApiError::Runtime)?;
        // From here the id is consumed, and the request ends in exactly one VcwFinished.
        let (result, outcome, digest) = self.run_ingested(
            runtime,
            slot,
            kind,
            &id,
            &request_id,
            &job,
            first_event,
            started + self.request_deadline,
        );
        runtime.record_vcw_finished(&request_id, outcome, digest);
        // The ledger range of the request: its raw text and, for an accepted answer, the admission.
        // Only the bounded summary is computed here, under the runtime lock (at most 64 canonical
        // encodings and digests); the store's own lock is taken for a move.
        let ledger =
            LedgerSummary::of(runtime.committed_events().get(first_event..).unwrap_or(&[]));
        if let Ok(mut traces) = self.traces.lock() {
            traces.finish(&id, outcome, ledger);
        }
        result
    }

    /// The trace of a finished request and a summary of the ledger events it committed, as JSON, or `None`
    /// if the request is unknown or no longer retained (the last
    /// [`MAX_RETAINED_TRACES`](super::report::MAX_RETAINED_TRACES) are kept). Read-only.
    #[must_use]
    pub fn trace_report(&self, id: &str) -> Option<serde_json::Value> {
        self.traces.lock().ok()?.report(id)
    }

    /// The part of a request after its raw text is committed. Returns the result with the
    /// outcome name and trace digest that the caller records; records nothing itself.
    #[allow(clippy::too_many_arguments)]
    fn run_ingested(
        &self,
        runtime: &mut PtrRuntime,
        slot: &Mutex<KindState>,
        kind: VcwTaskKind,
        id: &str,
        request_id: &RequestId,
        job: &Job,
        first_event: usize,
        deadline: Instant,
    ) -> (Result<VcwResponse, VcwApiError>, &'static str, Digest) {
        let failed = |why: String| (Err(VcwApiError::Internal(why)), "failed", [0; 32]);
        let judge = job.judge(runtime);
        let vcw_request = VcwRequest {
            request_id: request_id.clone(),
            project: self.project.clone(),
            request_digest: sha256(&[
                id.as_bytes(),
                job.statement.as_bytes(),
                kind.as_str().as_bytes(),
                &job.criteria,
            ]),
            required: VerificationLevel::Deterministic,
            budget: self.budget,
        };
        let sim_task = serde_json::to_string(&job.sim).unwrap_or_default();

        let Ok(mut state) = slot.lock() else {
            return failed("vcw loop mutex poisoned".into());
        };
        let KindState {
            vcw,
            experts,
            shadow,
        } = &mut *state;
        let (ran, deadline_hit) = {
            let mut standing = RuntimeStanding::new(runtime, &self.pods, &self.project);
            for (pod, generation) in &self.base_pods {
                standing = standing.with_base_pod(pod.clone(), *generation);
            }
            match experts {
                Experts::Simulated(sim) => {
                    sim.set_task(&sim_task);
                    let mut guarded = sim.guarded(&standing);
                    let mut recoded = Recoded::new(&mut guarded, SimCodec::default());
                    let mut pods = Deadlined::new(&mut recoded, deadline);
                    let ran = vcw.run(&vcw_request, &standing, &judge, &mut pods);
                    (ran, pods.hit)
                }
                Experts::Llm(llm) => {
                    let task_kind = match llm_kind(kind) {
                        Ok(task_kind) => task_kind,
                        Err(error) => return (Err(error), "failed", [0; 32]),
                    };
                    llm.set_task(ptr_runtime::vcw::experts::ExpertTask::new(
                        task_kind,
                        job.statement.clone(),
                    ));
                    let mut guarded = llm.guarded(&standing);
                    let mut recoded = Recoded::new(&mut guarded, TextCodec);
                    let mut pods = Deadlined::new(&mut recoded, deadline);
                    let ran = vcw.run(&vcw_request, &standing, &judge, &mut pods);
                    (ran, pods.hit)
                }
            }
        };
        let run = match ran {
            Ok(run) => run,
            Err(error) => {
                let digest = vcw.failed_trace().map_or([0; 32], |trace| *trace.digest());
                return (
                    Err(VcwApiError::Internal(format!("vcw loop failed: {error}"))),
                    "failed",
                    digest,
                );
            }
        };
        if let Some(shadow) = shadow {
            shadow.tick(runtime);
        }
        drop(state);

        let spent = run.trace.spent();
        let spent = VcwSpent {
            rounds: spent.rounds,
            verifier_calls: spent.verifier_calls,
            pod_calls: spent.pod_calls,
        };
        let trace_digest = *run.trace.digest();
        let digest_hex = hex(&trace_digest);
        let backend = backend_tag(self.backend).to_owned();
        if let Ok(mut traces) = self.traces.lock() {
            traces.insert(id, &backend, &run.trace);
        }
        let unanswered = |reason: String| VcwUnanswered {
            id: id.to_owned(),
            reason,
            spent,
            trace_digest: digest_hex.clone(),
            backend: backend.clone(),
        };
        match run.outcome {
            // The deadline took the Pods away: whatever ended the loop afterwards, the request
            // ran out of time. (A candidate that verified is still an answer.)
            VcwOutcome::Exhausted { .. } | VcwOutcome::Escalated { .. } if deadline_hit => (
                Ok(VcwResponse::Exhausted(unanswered("Deadline".into()))),
                "exhausted",
                trace_digest,
            ),
            VcwOutcome::Exhausted { reason } => (
                Ok(VcwResponse::Exhausted(unanswered(format!("{reason:?}")))),
                "exhausted",
                trace_digest,
            ),
            VcwOutcome::Escalated { reason } => (
                Ok(VcwResponse::Escalated(unanswered(format!("{reason:?}")))),
                "escalated",
                trace_digest,
            ),
            VcwOutcome::Verified(answer) => {
                let authors: Vec<VcwAuthor> = answer
                    .authors()
                    .iter()
                    .map(|author| VcwAuthor {
                        pod: author.pod.0.clone(),
                        generation: author.generation.0,
                    })
                    .collect();
                let payload = answer.payload().clone();
                let admission = match self.admission_request(
                    runtime,
                    request_id,
                    job,
                    answer.author().pod.clone(),
                    answer.author().generation,
                    payload.clone(),
                    trace_digest,
                ) {
                    Ok(admission) => admission,
                    Err(error) => return (Err(error), "failed", trace_digest),
                };
                // The runtime re-verifies the typed payload with the task's own verifier, built
                // over the state as it is now.
                let judge = job.judge(runtime);
                let admitted = admit_verified_answer(
                    runtime,
                    &self.pods,
                    &self.base_pods,
                    answer,
                    admission,
                    &judge,
                );
                if let Err(error) = admitted {
                    let error = match error {
                        VcwAdmitError::Runtime(error) => VcwApiError::Runtime(error),
                        other => VcwApiError::NotAdmitted(other.to_string()),
                    };
                    return (Err(error), "not_admitted", trace_digest);
                }
                let events = runtime.committed_events();
                let first_index = events
                    .get(first_event)
                    .map_or(first_event as u64, |event| event.index.0);
                let committed = VcwCommitted {
                    first_index,
                    count: (events.len() - first_event) as u64,
                };
                (
                    Ok(VcwResponse::Verified(VcwVerified {
                        id: id.to_owned(),
                        answer: String::from_utf8_lossy(&payload.bytes).into_owned(),
                        answer_type: payload.type_id.0.clone(),
                        authors,
                        spent,
                        trace_digest: digest_hex,
                        backend,
                        committed,
                    })),
                    "verified",
                    trace_digest,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn admission_request(
        &self,
        runtime: &PtrRuntime,
        request_id: &RequestId,
        job: &Job,
        pod: PodId,
        generation: Generation,
        payload: TypedPayload,
        trace_digest: Digest,
    ) -> Result<PodOutputAdmissionRequest, VcwApiError> {
        let internal = |why: &str| VcwApiError::Internal(why.to_owned());
        let entry = self
            .pods
            .get(&self.project, &pod)
            .ok_or_else(|| internal("the author is not in the Pod registry"))?;
        let pod_digest = entry.manifest().digest();
        let raw_key = request_raw_key(request_id);
        let revision: Revision = runtime.revision();
        let trace_hex = hex(&trace_digest);
        let manifest = ExecutionManifest::build(
            generation,
            vec![LineageBinding {
                key: raw_key.clone(),
                generation: Generation(1),
                digest: sha256(&[job.statement.as_bytes()]),
            }],
            vec![LineageBinding {
                key: format!("pod:{}", pod.0),
                generation,
                digest: pod_digest,
            }],
            // The trace digest is part of the manifest, so the manifest digest the ledger records
            // with the admission commits to it.
            {
                // Which experts produced the candidate is part of what the digest commits to.
                let mut origins = vec![
                    format!("vcw:request:{}", request_id.0),
                    format!("vcw:trace:{trace_hex}"),
                ];
                origins.extend(self.backend_origins.iter().cloned());
                origins
            },
            None,
            revision,
            sha256(&[request_id.0.as_bytes(), &trace_digest]),
            PrincipalId::from("ptr-server"),
            Revision(1),
        )
        .map_err(|error| VcwApiError::Internal(format!("manifest: {error:?}")))?;
        let manifest_digest = manifest.manifest_digest;
        let expected_type = payload.type_id.clone();
        Ok(PodOutputAdmissionRequest {
            request_id: request_id.clone(),
            manifest_digest,
            manifest,
            pod_id: pod,
            output: PodOutput {
                kind: PodOutputKind::Candidate,
                payload,
                generation,
                manifest_digest,
                artifact_digest: pod_digest,
                provenance: vec![ProvenanceRef {
                    source: EvidenceId::from(raw_key.as_str()),
                    note: Some(format!("vcw trace {trace_hex}")),
                }],
                dependencies: Vec::new(),
                revision,
                verified: false,
            },
            expected_type,
            session_id: self.session_id.clone(),
            scope_id: self
                .scope_id
                .get()
                .cloned()
                .ok_or_else(|| internal("the vcw service is not installed"))?,
            placement_epoch: PlacementEpoch(1),
            fencing_token: FencingToken(1),
            kv_taint: ptr_model_api::KvReuseTaint::Exact,
        })
    }
}

/// A [`RolePods`] that stops calling its Pods once the request's deadline has passed: every call
/// after it produces nothing, so the loop spends no more time on Pods or verifier runs and ends
/// within its remaining round budget.
struct Deadlined<'a, P> {
    inner: &'a mut dyn RolePods<P>,
    deadline: Instant,
    hit: bool,
}

impl<'a, P> Deadlined<'a, P> {
    fn new(inner: &'a mut dyn RolePods<P>, deadline: Instant) -> Self {
        Self {
            inner,
            deadline,
            hit: false,
        }
    }

    fn expired(&mut self) -> bool {
        if self.hit || Instant::now() >= self.deadline {
            self.hit = true;
        }
        self.hit
    }
}

impl<P> RolePods<P> for Deadlined<'_, P> {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<P> {
        if self.expired() {
            return None;
        }
        self.inner.propose(call)
    }

    fn repair(&mut self, call: RepairCall<'_, P>) -> Option<P> {
        if self.expired() {
            return None;
        }
        self.inner.repair(call)
    }

    fn escalate(&mut self, call: EscalateCall<'_, P>) -> Option<(PodRef, P)> {
        if self.expired() {
            return None;
        }
        self.inner.escalate(call)
    }
}

fn llm_kind(kind: VcwTaskKind) -> Result<ptr_runtime::vcw::experts::TaskKind, VcwApiError> {
    use ptr_runtime::vcw::experts::TaskKind;
    match kind {
        VcwTaskKind::Arithmetic => Ok(TaskKind::Arithmetic),
        VcwTaskKind::JsonObject => Ok(TaskKind::JsonObject),
        VcwTaskKind::Code | VcwTaskKind::Fact => Err(VcwApiError::BadRequest(format!(
            "the model backend has no prompts for {}",
            kind.as_str()
        ))),
    }
}

fn build_loop(
    config: &VcwServiceConfig,
    kind: VcwTaskKind,
    index: usize,
    roster: Roster,
    exploration: Exploration,
    parts: &VcwServiceParts,
) -> Result<VcwLoop, VcwBuildError> {
    let name = format!("vcw-{}", kind.as_str());
    let config_digest = sha256(&[
        name.as_bytes(),
        &config.max_verifier_calls.to_le_bytes(),
        &config.max_rounds.to_le_bytes(),
        &config.exploration_floor.to_le_bytes(),
    ]);
    let mut loop_config = VcwConfig::new(&name, config_digest);
    let schema = match config.hint_mode {
        VcwHintMode::TraceOnly => {
            loop_config.hint_mode = HintMode::TraceOnly;
            None
        }
        VcwHintMode::ApplyToSchema => {
            loop_config.hint_mode = HintMode::ApplyToSchema;
            let schema = AttentionSchema::new(SchemaLimits::default())
                .map_err(|error| VcwBuildError::Config(format!("attention schema: {error:?}")))?;
            Some(Arc::new(schema))
        }
    };
    // Each kind has its own nonce space under the shared key.
    let mut instance = parts.nonce_instance;
    instance[7] ^= u8::try_from(index + 1).unwrap_or(u8::MAX);
    VcwLoop::new(VcwParts {
        config: loop_config,
        roster,
        router: RouterConfig {
            exploration,
            ..RouterConfig::default()
        },
        scorer: UniformScorer,
        key: SamplingKey::from_bytes(parts.key),
        nonces: NonceSequence::starting_at(instance, 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema,
        structure: StructurePolicy::new(StructurePolicyConfig::default())
            .map_err(|error| VcwBuildError::Config(format!("structure policy: {error:?}")))?,
    })
    .map_err(VcwBuildError::Loop)
}

#[cfg(feature = "qwen-cpu")]
fn llm_experts(
    config: &VcwServiceConfig,
    kinds: &[VcwTaskKind],
    project: &ProjectId,
    proposer: &PodId,
    repairers: &[(PodId, Vec<String>)],
) -> Result<
    (
        BTreeMap<VcwTaskKind, Experts>,
        Option<ptr_pods::ModelIdentity>,
    ),
    VcwBuildError,
> {
    super::qwen::llm_experts(config, kinds, project, proposer, repairers)
}

#[cfg(not(feature = "qwen-cpu"))]
fn llm_experts(
    _config: &VcwServiceConfig,
    _kinds: &[VcwTaskKind],
    _project: &ProjectId,
    _proposer: &PodId,
    _repairers: &[(PodId, Vec<String>)],
) -> Result<
    (
        BTreeMap<VcwTaskKind, Experts>,
        Option<ptr_pods::ModelIdentity>,
    ),
    VcwBuildError,
> {
    Err(VcwBuildError::QwenNotCompiled)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_configurable_role_name_maps_to_its_own_role() {
        for (name, role) in ptr_config::VCW_REPAIR_ROLES.iter().zip(RepairRole::ALL) {
            assert_eq!(repair_role(name).unwrap(), role);
            // The configuration names are the verifier's role names without the suffix.
            assert_eq!(role.as_str(), format!("{name}-repair"));
        }
        assert!(repair_role("poetry").is_err());
    }

    use super::*;
    use ptr_config::PtrConfig;

    fn simulated() -> VcwServiceConfig {
        let mut config = VcwServiceConfig {
            enabled: true,
            ..VcwServiceConfig::default()
        };
        config.experts.backend = Some(VcwBackend::Simulated);
        config
    }

    #[test]
    fn an_enabled_service_needs_an_explicit_backend() {
        let mut config = simulated();
        config.experts.backend = None;
        let parts = VcwServiceParts::new([1; 32], [2; 8]);
        assert!(matches!(
            VcwService::from_config(&config, &parts),
            Err(VcwBuildError::Config(_))
        ));
    }

    #[test]
    fn the_manifest_of_every_admission_names_the_backend() {
        let service =
            VcwService::from_config(&simulated(), &VcwServiceParts::new([1; 32], [2; 8])).unwrap();
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        service.install(&mut runtime).unwrap();
        let task = VcwTask::Arithmetic {
            expression: "1 + 1".into(),
            reference: Some("2".into()),
        };
        let job = Job::from_request(&task, VcwBackend::Simulated).unwrap();
        let request = service
            .admission_request(
                &runtime,
                &RequestId::from("r"),
                &job,
                PodId::from("vcw-proposer"),
                Generation(1),
                TypedPayload {
                    type_id: answer_type(VcwTaskKind::Arithmetic),
                    bytes: b"2".to_vec(),
                },
                [7; 32],
            )
            .unwrap();
        assert!(request
            .manifest
            .origins
            .contains(&"vcw:backend:simulated".to_owned()));
        // The trace digest stays bound too, next to the backend.
        assert!(request
            .manifest
            .origins
            .iter()
            .any(|origin| origin.starts_with("vcw:trace:")));
        for pod in service.pods() {
            let capabilities = service.pod_capabilities(&pod.0).unwrap();
            assert!(capabilities.contains(&"backend:simulated".to_owned()));
        }
    }

    #[test]
    fn a_model_backend_is_marked_with_its_file_digests() {
        let mut config = simulated();
        config.experts.backend = Some(VcwBackend::QwenCpu);
        let digest = "ab".repeat(32);
        config.experts.qwen_cpu.weights_sha256 = digest.clone();
        config.experts.qwen_cpu.config_sha256 = digest.clone();
        config.experts.qwen_cpu.tokenizer_sha256 = digest.clone();
        config.experts.qwen_cpu.tokenizer_config_sha256 = digest.clone();
        let origins = backend_origins(&config, VcwBackend::QwenCpu);
        assert_eq!(origins[0], "vcw:backend:qwen-cpu");
        assert_eq!(origins.len(), 5);
        assert!(origins[1..]
            .iter()
            .all(|origin| origin.starts_with("vcw:model:") && origin.ends_with(&digest)));
    }
}
