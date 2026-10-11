//! One lane of the loop: a task kind with its own `VcwLoop`, Pods, router state and, if the arm
//! has it, a runtime that L3 changes the roster through.
//!
//! The lane is the unit every arm is built from, so that the arms differ only in the parts the
//! experiment names: the registry (which Pods a code may go to), the scorer, whether the
//! self-model's hints can steer, whether the broadcast reaches the Pods and whether L3 is live.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Instant;

use ed25519_dalek::SigningKey;
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_lineage::ForgettingGate;
use ptr_pods::{BaseModelDigest, BaseModelId, BaseModelRef, PodRegistry, ProposerRef};
use ptr_router::{
    ActiveScorer, AdmissibleScorer, AttentionSchema, ControlArm, ControlConfig, ControlledScorer,
    Exploration, NonceSequence, RouteClass, RouteError, RouterConfig, RoutingLearner, SamplingKey,
    SchemaLimits, ScoringContext, SelfModel, SelfModelLimits,
};
use ptr_runtime::execution::AdmissionPolicy;
use ptr_runtime::expert::{
    BirthSite, EnabledOps, ExpertAdmissionGrant, ExpertGateRequirement, ExpertGrantBinding,
    ExpertGrantKind, StructurePolicy, StructurePolicyConfig,
};
use ptr_runtime::vcw::experts::{SimulatedExperts, SimulationConfig};
use ptr_runtime::vcw::{
    HintMode, RequestBudget, RuntimeStanding, VcwConfig, VcwLoop, VcwOutcome, VcwParts, VcwRequest,
    VcwTrace,
};
use ptr_runtime::vcw_host::{
    DriverHost, HeldOutSuite, SimulatedHeldOut, SimulatedTrainer, StructureDriver, StructureResult,
    TrainedSpecialist,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{
    AuthenticationLevel, CapsuleId, Generation, IdentityContext, NodeId, PodId, ProjectId,
    RequestId, SessionId, Timestamp, TypeId, VerificationLevel,
};

use super::learn::GateScorer;
use super::params;
use super::task::{with_task, Judge, Kind, Served, Task, VerifyEvent};
use super::world::{
    loop_roster, repairer_roster, Dispatch, PodBank, RegistryMode, WorldId, GAP_CODE,
};

/// The scorer a lane routes with. One type, so that every arm is the same `VcwLoop`.
#[derive(Clone)]
pub enum RouteScorer {
    /// Uniform until a host promotes a learned policy into the book this handle reads.
    Active(ActiveScorer),
    /// The softmax gate baseline.
    Gate(Arc<GateScorer>),
}

impl AdmissibleScorer for RouteScorer {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        match self {
            Self::Active(scorer) => scorer.weights(context),
            Self::Gate(gate) => gate.weights(context),
        }
    }

    fn bound_exploration(&self) -> Option<Exploration> {
        match self {
            Self::Active(scorer) => scorer.bound_exploration(),
            Self::Gate(gate) => gate.bound_exploration(),
        }
    }
}

pub type LaneScorer = ControlledScorer<RouteScorer>;

/// How a lane is built.
#[derive(Clone)]
pub struct LaneConfig {
    pub world: WorldId,
    pub kind: Kind,
    pub registry: RegistryMode,
    pub scorer: RouteScorer,
    /// Whether the self-model's hints may boost a Pod inside the admissible set.
    pub hints_steer: bool,
    /// Whether the attention schema observes (a lane without a self-model has none).
    pub schema: bool,
    pub l3: bool,
    pub broadcast: bool,
    pub sim_seed: u64,
    pub router_seed: u8,
    pub learner: Option<Arc<RoutingLearner>>,
    /// The verifier-call budget of a request.
    pub calls: u32,
    /// Whether round 0 is the shared first draft of every arm (evaluation) or is routed freely
    /// (the training streams, on which the learned parts are fitted).
    pub shared_first_draft: bool,
}

// ---- the host side of L3 (the grant, the gate and the project), as the L3 tests set it up ----

const LAYER: u16 = 3;
const KEY_ID: &str = "policy-key";
const PEER: &str = "expert-host-peer";
const SESSION: &str = "session-1";
const NOW: Timestamp = Timestamp(1_000);

fn project_for(kind: Kind) -> ProjectId {
    ProjectId::from(format!("f007-{}", kind.name()).as_str())
}

fn identity() -> IdentityContext {
    let mut identity = IdentityContext {
        subject: "host".into(),
        issuer: "issuer".into(),
        groups: vec!["operators".into()],
        authentication_level: AuthenticationLevel::MultiFactor,
        session_id: SessionId(SESSION.into()),
        issued_at: Timestamp(100),
        expires_at: Timestamp(10_000),
        identity_digest: [0; 32],
    };
    identity.identity_digest = identity.canonical_digest();
    identity
}

fn prime(rt: &mut PtrRuntime) {
    rt.install_policy_key(KEY_ID, SigningKey::from_bytes(&[7; 32]).verifying_key());
    rt.admit_identity_context(&identity(), NOW)
        .expect("identity admitted");
    let mut policy = AdmissionPolicy::new();
    policy
        .admit(
            NodeId::from(PEER),
            "host",
            std::time::Duration::from_secs(60),
            Vec::new,
        )
        .expect("peer admitted");
    rt.install_admission_policy(policy);
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.1,
        max_task_forgetting: 0.2,
        min_backward_transfer: -0.1,
        max_public_regression: 0.05,
    }
}

fn base_model() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").expect("id"),
        BaseModelDigest::new([9; 32]).expect("digest"),
        1,
    )
    .expect("base model")
}

fn proposer_ref(project: &ProjectId) -> ProposerRef {
    ProposerRef::new(project.clone(), PodId::from("meta"), Generation(1)).expect("proposer")
}

fn grant_for(project: &ProjectId, manifest: [u8; 32], rt: &PtrRuntime) -> ExpertAdmissionGrant {
    let requirement = ExpertGateRequirement {
        gate_epoch: 7,
        config_digest: gate().config_digest().expect("digest"),
        eval_manifest_digest: manifest,
        require_inputs: false,
    };
    let binding = ExpertGrantBinding::new(
        identity(),
        NodeId::from(PEER),
        NOW,
        KEY_ID,
        rt.expert_grant_generation(ExpertGrantKind::Admission),
    );
    ExpertAdmissionGrant::new(project.clone(), proposer_ref(project), requirement, binding)
}

/// The runtime, registry, driver and trainer of a lane with L3 live.
struct L3Env {
    rt: PtrRuntime,
    pods: PodRegistry,
    driver: StructureDriver,
    trainer: SimulatedTrainer,
    suite: SimulatedHeldOut,
    /// The digest of the held-out suite, which the admission grant expects in the subject.
    manifest: [u8; 32],
    aimer: BTreeMap<PodId, f64>,
    project: ProjectId,
    admitted: Vec<PodId>,
    refused: u64,
}

impl L3Env {
    fn new(world: WorldId, kind: Kind, bank: &PodBank, seed: u64) -> Self {
        let project = project_for(kind);
        let mut rt = PtrRuntime::new(PtrConfig::default()).expect("runtime");
        prime(&mut rt);
        let mut pods = PodRegistry::default();
        for entry in bank.registry_entries(&project, kind) {
            pods.register_new(entry).expect("fresh Pod");
        }
        for pod in PodBank::base_pods() {
            rt.commit(LedgerEvent::CapsuleCommitted {
                project: project.clone(),
                capsule: CapsuleId::from(pod.0.as_str()),
                generation: Generation(1),
            })
            .expect("capsule committed");
        }
        // The held-out suite measures the gate's before/after clear rates on defect documents
        // with the same per-Pod skills.
        let mut sim = SimulationConfig::new(seed ^ 0x51);
        for pod in super::world::REPAIRERS {
            for code in kind.codes() {
                sim = sim.with_pod_repair_success(
                    pod.id,
                    code,
                    super::world::skill_permille(world, &pod, code) as f64 / 1000.0,
                );
            }
        }
        let held_out: Vec<&str> = kind.codes().to_vec();
        let suite = SimulatedHeldOut::new(
            sim,
            project.clone(),
            &held_out,
            params::HELD_OUT_TASKS_PER_CODE as u32,
            seed,
        )
        .expect("held-out suite");
        let driver = StructureDriver::new(
            project.clone(),
            proposer_ref(&project),
            base_model(),
            BirthSite::Token {
                layer: LAYER,
                slots: 16,
            },
            gate(),
        );
        let manifest = suite.manifest_digest();
        let born = params::BORN_SKILL_PERMILLE as f64 / 1000.0;
        let mut trainer = SimulatedTrainer::new(
            project.clone(),
            TypeId::from(kind.answer_type()),
            seed,
            born,
        );
        if world.gap() {
            trainer = trainer.with_skill(GAP_CODE, born);
        }
        Self {
            rt,
            pods,
            driver,
            trainer,
            suite,
            manifest,
            aimer: BTreeMap::new(),
            project,
            admitted: Vec::new(),
            refused: 0,
        }
    }
}

/// How one request ended, for the audit.
pub enum EndTag {
    Verified,
    Exhausted,
    Escalated,
    /// The loop returned an error (a product defect).
    Error,
    /// The loop panicked (a product defect, caught).
    Panic,
}

/// Everything the audit needs to know about one request.
pub struct RequestRecord {
    pub tag: EndTag,
    pub accepted: Option<Served>,
    pub events: Vec<VerifyEvent>,
    pub trace: Option<VcwTrace>,
    pub task: Arc<Task>,
    pub elapsed_ns: u128,
}

pub struct Lane {
    pub kind: Kind,
    vcw: VcwLoop<LaneScorer>,
    pub bank: PodBank,
    snapshot: ptr_router::EligibilitySnapshot,
    l3: Option<L3Env>,
    /// The independent admissible set per code: the registry as the arm declares it, plus the
    /// codes of an expert L3 has admitted.
    admissible: BTreeMap<String, BTreeSet<PodId>>,
    proposers: BTreeSet<PodId>,
    calls: u32,
    pub births: u64,
    pub l3_refusals: u64,
    pub unadmitted: u64,
    pub requests: u64,
}

impl Lane {
    pub fn new(cfg: LaneConfig) -> Self {
        let kind = cfg.kind;
        let cfg_calls = cfg.calls;
        let roster = loop_roster(cfg.world, kind, cfg.registry);
        let mut admissible: BTreeMap<String, BTreeSet<PodId>> = BTreeMap::new();
        for (pod, codes) in repairer_roster(cfg.world, kind, cfg.registry) {
            for code in codes {
                admissible.entry(code).or_default().insert(pod.clone());
            }
        }
        let proposers: BTreeSet<PodId> = roster.proposers.iter().cloned().collect();
        let mut config = VcwConfig::new("f007", [7; 32]);
        config.window_requests = params::WINDOW_REQUESTS;
        config.hint_mode = if cfg.hints_steer {
            HintMode::ApplyToSchema
        } else {
            HintMode::TraceOnly
        };
        let schema = Arc::new(AttentionSchema::new(SchemaLimits::default()).expect("schema"));
        let arm = if cfg.hints_steer {
            ControlArm::HintOnly
        } else {
            ControlArm::InnerOnly
        };
        let scorer = ControlledScorer::new(
            cfg.scorer.clone(),
            Arc::clone(&schema),
            arm,
            ControlConfig::default(),
        );
        let structure = StructurePolicyConfig {
            ops: EnabledOps {
                birth: true,
                merge: false,
                prune: false,
                freeze: false,
                reactivate: false,
            },
            min_unroutable: params::BIRTH_MIN_UNROUTABLE,
            birth_windows: params::BIRTH_WINDOWS as usize,
            ..StructurePolicyConfig::default()
        };
        let key = cfg.router_seed;
        let vcw = VcwLoop::new(VcwParts {
            config,
            roster,
            router: RouterConfig::default(),
            scorer,
            key: SamplingKey::from_bytes([key; 32]),
            nonces: NonceSequence::starting_at([key; 8], 0),
            self_model: SelfModel::new(SelfModelLimits::default()),
            schema: cfg.schema.then_some(schema),
            structure: StructurePolicy::new(structure).expect("structure policy"),
        })
        .expect("loop");
        let vcw = match &cfg.learner {
            Some(learner) => vcw.with_learner(Arc::clone(learner)).expect("learner"),
            None => vcw,
        };
        let mut bank = PodBank::new(cfg.world, kind, cfg.sim_seed, cfg.broadcast);
        bank.shared_first_draft = cfg.shared_first_draft;
        let mut snapshot = ptr_router::EligibilitySnapshot::new();
        for pod in PodBank::base_pods() {
            snapshot.set(pod, Generation(1));
        }
        let l3 = cfg
            .l3
            .then(|| L3Env::new(cfg.world, kind, &bank, cfg.sim_seed));
        Self {
            kind,
            vcw,
            bank,
            snapshot,
            l3,
            admissible,
            proposers,
            calls: cfg_calls,
            births: 0,
            l3_refusals: 0,
            unadmitted: 0,
            requests: 0,
        }
    }

    pub fn broadcast_reads(&self) -> u64 {
        self.bank.reads
    }

    pub fn counters(&self) -> ptr_runtime::vcw::VcwCounters {
        self.vcw.counters()
    }

    /// Whether `pod` may be asked for `class` according to the arm's declared registry.
    pub fn is_admissible(&self, class: &RouteClass, pod: &PodId) -> bool {
        match class {
            RouteClass::Repair(code) => self
                .admissible
                .get(code.as_str())
                .is_some_and(|pods| pods.contains(pod)),
            RouteClass::Regenerate(_) => self.proposers.contains(pod),
        }
    }

    /// Serve one request. A panic of the loop is caught and reported as such.
    pub fn serve(&mut self, request_id: &str, task: &Arc<Task>) -> RequestRecord {
        self.requests += 1;
        let started = Instant::now();
        let judge = Judge::new(task);
        let digest = ptr_ledger::integrity::sha256(request_id.as_bytes());
        let request = VcwRequest {
            request_id: RequestId::from(request_id),
            project: project_for(self.kind),
            request_digest: digest,
            required: VerificationLevel::Deterministic,
            budget: RequestBudget::new(self.calls, params::ROUND_BUDGET).expect("budget"),
        };
        let Self {
            vcw,
            bank,
            snapshot,
            l3,
            ..
        } = self;
        let result = catch_unwind(AssertUnwindSafe(|| {
            with_task(task, || match l3 {
                None => {
                    let mut pods = Dispatch {
                        bank,
                        standing: snapshot,
                    };
                    vcw.run(&request, snapshot, &judge, &mut pods)
                }
                Some(env) => {
                    let mut standing = RuntimeStanding::new(&env.rt, &env.pods, &env.project);
                    for pod in PodBank::base_pods() {
                        standing = standing.with_base_pod(pod, Generation(1));
                    }
                    let mut pods = Dispatch {
                        bank,
                        standing: &standing,
                    };
                    vcw.run(&request, &standing, &judge, &mut pods)
                }
            })
        }));
        let events = judge.take_events();
        let elapsed_ns = started.elapsed().as_nanos();
        let (tag, accepted, trace) = match result {
            Err(_) => (EndTag::Panic, None, None),
            Ok(Err(_)) => (EndTag::Error, None, None),
            Ok(Ok(run)) => {
                let trace = Some(run.trace);
                match run.outcome {
                    VcwOutcome::Verified(answer) => {
                        (EndTag::Verified, Some(answer.into_payload()), trace)
                    }
                    VcwOutcome::Exhausted { .. } => (EndTag::Exhausted, None, trace),
                    VcwOutcome::Escalated { .. } => (EndTag::Escalated, None, trace),
                }
            }
        };
        let record = RequestRecord {
            tag,
            accepted,
            events,
            trace,
            task: Arc::clone(task),
            elapsed_ns,
        };
        self.step_l3();
        record
    }

    /// Let the structure policy look at the windows that closed, through admission.
    fn step_l3(&mut self) {
        let Self { vcw, l3, bank, .. } = self;
        let Some(env) = l3 else { return };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let project = env.project.clone();
            let manifest = env.manifest;
            let grant = move |rt: &PtrRuntime| grant_for(&project, manifest, rt);
            let L3Env {
                rt,
                pods,
                driver,
                trainer,
                suite,
                aimer,
                ..
            } = env;
            driver.step(
                vcw,
                DriverHost {
                    runtime: rt,
                    pods,
                    grant: &grant,
                    trainer,
                    suite,
                    aimer,
                },
            )
        }));
        let Ok(Ok(report)) = outcome else { return };
        for entry in report.outcomes {
            match entry.result {
                StructureResult::Admitted {
                    specialist: Some(specialist),
                    ..
                } => {
                    apply_specialist(&mut bank.rep, &specialist);
                    env.admitted.push(specialist.pod.clone());
                    for code in &specialist.codes {
                        self.admissible
                            .entry(code.as_str().to_owned())
                            .or_default()
                            .insert(specialist.pod.clone());
                    }
                    self.births += 1;
                }
                StructureResult::Admitted { .. } => {}
                StructureResult::Refused(_) => {
                    env.refused += 1;
                    self.l3_refusals += 1;
                }
                // Committed in the ledger but not installed: not an admission here, so the
                // end-of-run audit counts the structural event as unbacked (a product defect).
                StructureResult::CommittedNotInstalled { .. } => {}
            }
        }
    }

    /// The end-of-run audit of L3: every expert in the roster is backed by exactly one
    /// committed structural event, and every structural event by an admitted proposal.
    pub fn audit_l3(&mut self) -> u64 {
        let Some(env) = &self.l3 else { return 0 };
        let committed = env
            .rt
            .committed_events()
            .iter()
            .filter(|event| matches!(event.event, LedgerEvent::ExpertStructuralCommitted { .. }))
            .count() as u64;
        let admitted = env.admitted.len() as u64;
        let mut bad = committed.abs_diff(admitted);
        let known: BTreeSet<PodId> = PodBank::base_pods()
            .into_iter()
            .chain(env.admitted.iter().cloned())
            .collect();
        for pod in self.vcw.repairers().keys() {
            if !known.contains(pod) {
                bad += 1;
            }
        }
        for pod in self
            .bank
            .rep
            .roster()
            .map(|r| r.repairers)
            .unwrap_or_default()
        {
            if !known.contains(&pod.0) {
                bad += 1;
            }
        }
        self.unadmitted = bad;
        bad
    }
}

fn apply_specialist(experts: &mut SimulatedExperts<Served>, specialist: &TrainedSpecialist) {
    let codes: Vec<&str> = specialist.codes.iter().map(|c| c.as_str()).collect();
    experts
        .add_repairer(specialist.pod.clone(), &codes)
        .expect("a fresh expert");
    for (code, p) in &specialist.skill {
        experts
            .set_pod_repair_success(&specialist.pod, code, *p)
            .expect("a skill for a code the expert repairs");
    }
    for collateral in &specialist.collateral {
        let _ = experts.set_pod_repair_success(
            &collateral.pod,
            &collateral.code,
            collateral.success_after,
        );
    }
}
