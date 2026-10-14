//! L3 live through the ledger: windows of real outcomes become structural requests, the host
//! completes them with a (simulated) specialist, a held-out suite of verified tasks feeds the
//! forgetting gate, and the change reaches the roster only through `admit_expert_proposal` and
//! `sync_roster`. Experts are the seeded programmatic ones: a specialist's skill is a
//! configuration, its "training" is a seed, and the collateral damage of a bad one is a
//! configured fact of the simulation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertInputRef, ExpertInputs, ExpertLevel,
    ExpertPodSpec, ExpertProposal, ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest,
    OpKind, PodRegistry, ProposerRef, RationaleDigest, RetiredReason, StructuralOp,
};
use ptr_router::LearnerConfig;
use ptr_router::{
    ActiveScorer, Exploration, FindingCode, NonceSequence, PolicyBook, PromotionGate,
    PromotionRule, RouteClass, RouterConfig, RoutingLearner, SamplingKey, SelfModel,
    SelfModelLimits, Verdict,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_artifact_digest_with_inputs,
    expert_evidence_base_model_digest, BirthContext, BirthSite, EnabledOps, ExpertAdmissionError,
    ExpertAdmissionGrant, ExpertFenceGrant, ExpertGrantKind, InputInvalidation, StructurePolicy,
    StructurePolicyConfig,
};
use ptr_runtime::vcw::experts::{DefectDoc, DefectDocVerifier, SimulatedExperts, SimulationConfig};
use ptr_runtime::vcw::{
    RequestBudget, Roster, RuntimeStanding, StructureHost, VcwConfig, VcwLoop, VcwOutcome,
    VcwParts, VcwRequest,
};
use ptr_runtime::vcw_host::{
    rebuild_roster, AbandonReason, DriverError, DriverHost, DriverReport, HeldOutSuite,
    MeasurementError, Refusal, RosterRebuildError, SimulatedHeldOut, SimulatedTrainer,
    SpecialistTrainer, StructureDriver, StructureResult, TrainedSpecialist, TrainingRequest,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    CapsuleId, CommitIndex, Generation, PodId, ProjectId, RequestId, VerificationLevel,
};
use sha2::{Digest as _, Sha256};

mod expert_support;

use expert_support::{current_binding, prime, requirement, EVAL_MANIFEST};

const LAYER: u16 = 3;
const TRAIN: &str = "dataset:train";
const D1: [u8; 32] = [0x11; 32];

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn pod(name: &str) -> PodId {
    PodId::from(name)
}

fn code(name: &str) -> FindingCode {
    FindingCode::new(name)
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap()
}

fn proposer() -> ProposerRef {
    ProposerRef::new(project(), pod("meta"), Generation(1)).unwrap()
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.1,
        max_task_forgetting: 0.2,
        min_backward_transfer: -0.1,
        max_public_regression: 0.05,
    }
}

/// The grant of the set-up (the seeds), which expects the test hosts' fixed evaluation manifest.
fn grant_for(rt: &PtrRuntime) -> ExpertAdmissionGrant {
    grant_expecting(rt, EVAL_MANIFEST)
}

/// A grant that expects `eval_manifest`: for the driver, the digest of its held-out suite.
fn grant_expecting(rt: &PtrRuntime, eval_manifest: [u8; 32]) -> ExpertAdmissionGrant {
    let mut requirement = requirement(7, gate().config_digest().unwrap());
    requirement.eval_manifest_digest = eval_manifest;
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement,
        current_binding(rt, ExpertGrantKind::Admission),
    )
}

/// A passing report for the set-up of the world (the seeds), bound to the proposal.
fn seed_report(proposal: &ExpertProposal, inputs: Option<&ExpertInputs>) -> BoundGateReport {
    let artifact = match inputs {
        Some(inputs) => expert_evidence_artifact_digest_with_inputs(proposal.op(), inputs),
        None => expert_evidence_artifact_digest(proposal.op()),
    };
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        artifact,
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate()
        .evaluate(
            &AdapterId::from("seed"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject)
}

fn dataset_inputs() -> ExpertInputs {
    ExpertInputs::new(
        vec![ExpertInputRef::new(TRAIN, Generation(1), D1).unwrap()],
        Vec::new(),
    )
    .unwrap()
}

struct Seed {
    name: &'static str,
    codes: &'static [&'static str],
    skill: f64,
    with_inputs: bool,
}

struct Opts {
    error_rates: Vec<(&'static str, f64)>,
    fixer_a_skill: f64,
    seeds: Vec<Seed>,
    structure: StructurePolicyConfig,
    window_requests: u32,
    held_out_codes: Vec<&'static str>,
    trainer: SimulatedTrainer,
    learn: bool,
    /// Extra base repairers `fill0..` (each repairing the code of its own name), to fill the
    /// loop's roster. They are measured by the held-out suite like every other code.
    filler: usize,
}

fn birth_only() -> StructurePolicyConfig {
    StructurePolicyConfig {
        ops: EnabledOps {
            birth: true,
            merge: false,
            prune: false,
            freeze: false,
            reactivate: false,
        },
        min_unroutable: 4,
        birth_windows: 2,
        ..StructurePolicyConfig::default()
    }
}

fn trainer(default_skill: f64) -> SimulatedTrainer {
    SimulatedTrainer::new(project(), DefectDoc::clean_type(), 99, default_skill)
}

struct World {
    rt: PtrRuntime,
    pods: PodRegistry,
    vcw: VcwLoop<ActiveScorer>,
    experts: SimulatedExperts<DefectDoc>,
    book: PolicyBook,
    learner: Option<Arc<RoutingLearner>>,
    driver: StructureDriver,
    trainer: SimulatedTrainer,
    suite: SimulatedHeldOut,
    aimer: BTreeMap<PodId, f64>,
    base_roster: Roster,
}

#[derive(Debug, Default)]
struct Served {
    verified: u32,
    requests: u32,
    routed: BTreeMap<PodId, u64>,
}

impl Served {
    fn routed_to(&self, name: &str) -> u64 {
        self.routed.get(&pod(name)).copied().unwrap_or(0)
    }
}

fn apply_specialist(experts: &mut SimulatedExperts<DefectDoc>, specialist: &TrainedSpecialist) {
    let codes: Vec<&str> = specialist.codes.iter().map(FindingCode::as_str).collect();
    experts
        .add_repairer(specialist.pod.clone(), &codes)
        .unwrap();
    for (skill_code, p) in &specialist.skill {
        experts
            .set_pod_repair_success(&specialist.pod, skill_code, *p)
            .unwrap();
    }
    for collateral in &specialist.collateral {
        let _ = experts.set_pod_repair_success(
            &collateral.pod,
            &collateral.code,
            collateral.success_after,
        );
    }
}

fn seed_expert(
    rt: &mut PtrRuntime,
    pods: &mut PodRegistry,
    trainer: &mut SimulatedTrainer,
    seed: &Seed,
    slot: u16,
) -> TrainedSpecialist {
    seed_expert_declaring(rt, pods, trainer, seed, slot, true)
}

/// As `seed_expert`; with `declare_codes` false the committed manifest names no code (no
/// `repairs:` capability), like a base Pod that was later registered as an expert.
fn seed_expert_declaring(
    rt: &mut PtrRuntime,
    pods: &mut PodRegistry,
    trainer: &mut SimulatedTrainer,
    seed: &Seed,
    slot: u16,
    declare_codes: bool,
) -> TrainedSpecialist {
    let mut trained = trainer
        .train(&TrainingRequest {
            kind: OpKind::Birth,
            pod: pod(seed.name),
            codes: seed.codes.iter().map(|c| code(c)).collect(),
            sources: Vec::new(),
        })
        .unwrap();
    if !declare_codes {
        trained
            .manifest
            .capabilities
            .retain(|c| !c.0.starts_with(ptr_runtime::vcw_host::REPAIRS_PREFIX));
        trained.manifest_digest = ptr_pods::ManifestDigest::new(trained.manifest.digest()).unwrap();
    }
    if seed.with_inputs {
        trained.inputs = Some(dataset_inputs());
    }
    let spec = ExpertPodSpec::new(
        project(),
        pod(seed.name),
        trained.manifest_digest,
        ExpertLevel::Token {
            layer: LAYER,
            index: slot,
        },
        base(),
        Generation(1),
    )
    .unwrap();
    let proposal = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Birth { result: spec },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    let report = seed_report(&proposal, trained.inputs.as_ref());
    let grant = grant_for(rt);
    match trained.inputs.clone() {
        Some(inputs) => rt.admit_expert_proposal_with_inputs(&proposal, &report, &grant, inputs),
        None => rt.admit_expert_proposal(&proposal, &report, &grant),
    }
    .unwrap();
    pods.register_new(trained.executable()).unwrap();
    trained
}

impl World {
    fn new(opts: Opts) -> Self {
        let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
        prime(&mut rt);
        for name in ["gen", "fixer_a"] {
            rt.commit(LedgerEvent::CapsuleCommitted {
                project: project(),
                capsule: CapsuleId::from(name),
                generation: Generation(1),
            })
            .unwrap();
        }
        let mut sim =
            SimulationConfig::new(31).with_pod_repair_success("fixer_a", "a", opts.fixer_a_skill);
        for (c, rate) in &opts.error_rates {
            sim = sim.with_error_rate(c, *rate);
        }
        for seed in &opts.seeds {
            for c in seed.codes {
                sim = sim.with_pod_repair_success(seed.name, c, seed.skill);
            }
        }
        let mut experts = SimulatedExperts::new(sim.clone(), "task").unwrap();
        experts.add_proposer(pod("gen")).unwrap();
        experts.add_repairer(pod("fixer_a"), &["a"]).unwrap();
        let fillers: Vec<String> = (0..opts.filler).map(|i| format!("fill{i}")).collect();
        for name in &fillers {
            experts.add_repairer(pod(name), &[name.as_str()]).unwrap();
        }
        let mut pods = PodRegistry::default();
        for registered in experts.pods(&project(), &DefectDoc::clean_type()) {
            pods.register_new(registered).unwrap();
        }
        let mut seeding = trainer(0.9);
        let mut roster = Roster {
            proposers: vec![pod("gen")],
            repairers: vec![(pod("fixer_a"), vec![code("a")])],
        };
        for name in &fillers {
            roster.repairers.push((pod(name), vec![code(name)]));
        }
        for (slot, seed) in opts.seeds.iter().enumerate() {
            let trained = seed_expert(
                &mut rt,
                &mut pods,
                &mut seeding,
                seed,
                100 + u16::try_from(slot).unwrap(),
            );
            experts.add_repairer(pod(seed.name), seed.codes).unwrap();
            roster
                .repairers
                .push((pod(seed.name), trained.codes.iter().cloned().collect()));
        }
        let book = PolicyBook::new("defect-doc", Exploration::default()).unwrap();
        let learner = opts
            .learn
            .then(|| Arc::new(RoutingLearner::new(LearnerConfig::new("defect-doc", 5)).unwrap()));
        let mut config = VcwConfig::new("structure-test", [4; 32]);
        config.window_requests = opts.window_requests;
        let vcw = VcwLoop::new(VcwParts {
            config,
            roster: roster.clone(),
            router: RouterConfig::default(),
            scorer: book.scorer(),
            key: SamplingKey::from_bytes([2; 32]),
            nonces: NonceSequence::starting_at([2; 8], 0),
            self_model: SelfModel::new(SelfModelLimits::default()),
            schema: None,
            structure: StructurePolicy::new(opts.structure).unwrap(),
        })
        .unwrap();
        let vcw = match &learner {
            Some(learner) => vcw.with_learner(Arc::clone(learner)).unwrap(),
            None => vcw,
        };
        let mut held_out: Vec<&str> = opts.held_out_codes.clone();
        held_out.extend(fillers.iter().map(String::as_str));
        let suite = SimulatedHeldOut::new(sim, project(), &held_out, 30, 7)
            .unwrap()
            .with_world_of(&opts.trainer);
        let driver = StructureDriver::new(
            project(),
            proposer(),
            base(),
            BirthSite::Token {
                layer: LAYER,
                slots: 16,
            },
            gate(),
        );
        Self {
            rt,
            pods,
            vcw,
            experts,
            book,
            learner,
            driver,
            trainer: opts.trainer,
            suite,
            aimer: BTreeMap::new(),
            base_roster: roster,
        }
    }

    fn serve(&mut self, prefix: &str, n: u32) -> Served {
        let mut served = Served::default();
        let proj = project();
        let standing = RuntimeStanding::new(&self.rt, &self.pods, &proj)
            .with_base_pod(pod("gen"), Generation(1))
            .with_base_pod(pod("fixer_a"), Generation(1));
        for i in 0..n {
            let id = format!("{prefix}-{i}");
            let request = VcwRequest {
                request_id: RequestId::from(id.as_str()),
                project: project(),
                request_digest: Sha256::digest(id.as_bytes()).into(),
                required: VerificationLevel::SampleVerified,
                budget: RequestBudget::new(8, 6).unwrap(),
            };
            let run = self
                .vcw
                .run(
                    &request,
                    &standing,
                    &DefectDocVerifier,
                    &mut self.experts.guarded(&standing),
                )
                .unwrap();
            served.requests += 1;
            served.verified += u32::from(matches!(run.outcome, VcwOutcome::Verified(_)));
            for record in run.trace.routes() {
                if matches!(record.class(), RouteClass::Repair(_)) {
                    *served.routed.entry(record.pod().clone()).or_default() += 1;
                }
            }
        }
        served
    }

    fn l3(&mut self) -> DriverReport {
        let suite = self.suite.clone();
        let expected = suite.manifest_digest();
        self.l3_with(&suite, expected)
    }

    /// One driver step measured by `suite`, under a grant that expects `expected`.
    fn l3_with(&mut self, suite: &dyn HeldOutSuite, expected: [u8; 32]) -> DriverReport {
        let grant = move |rt: &PtrRuntime| grant_expecting(rt, expected);
        let report = self
            .driver
            .step(
                &mut self.vcw,
                DriverHost {
                    runtime: &mut self.rt,
                    pods: &mut self.pods,
                    grant: &grant,
                    trainer: &mut self.trainer,
                    suite,
                    aimer: &self.aimer,
                },
            )
            .unwrap();
        for outcome in &report.outcomes {
            if let StructureResult::Admitted {
                specialist: Some(specialist),
                ..
            } = &outcome.result
            {
                apply_specialist(&mut self.experts, specialist);
            }
        }
        report
    }

    /// Retire `name` the way a host does: a Prune proposal with a passing seed report.
    fn prune_via_admission(&mut self, name: &str) {
        self.retire_with(name, None);
    }

    /// Retire `name` through a `Retire` proposal with `reason`.
    fn retire_via_admission(&mut self, name: &str, reason: RetiredReason) {
        self.retire_with(name, Some(reason));
    }

    fn retire_with(&mut self, name: &str, reason: Option<RetiredReason>) {
        let entry = self
            .rt
            .expert_registry()
            .get(&project(), &pod(name))
            .unwrap()
            .clone();
        let target = ExpertTarget::new(ExpertTargetInit {
            project: project(),
            pod: pod(name),
            generation: entry.generation,
            level: entry.level,
            base_model: entry.base_model,
            manifest_digest: entry.manifest_digest,
            state: entry.state,
            state_version: entry.state_version,
        })
        .unwrap();
        let op = match reason {
            None => StructuralOp::Prune {
                targets: vec![target],
            },
            Some(reason) => StructuralOp::Retire { target, reason },
        };
        let prune = ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(1),
            op,
            RationaleDigest::new([8; 32]).unwrap(),
        )
        .unwrap();
        let evidence = seed_report(&prune, None);
        let grant = grant_for(&self.rt);
        self.rt
            .admit_expert_proposal(&prune, &evidence, &grant)
            .unwrap();
    }

    fn structural_commits(&self) -> usize {
        self.rt
            .committed_events()
            .iter()
            .filter(|e| matches!(e.event, LedgerEvent::ExpertStructuralCommitted { .. }))
            .count()
    }
}

fn newcode_opts(trainer: SimulatedTrainer) -> Opts {
    Opts {
        error_rates: vec![("a", 0.5), ("newcode", 1.0)],
        fixer_a_skill: 0.95,
        seeds: Vec::new(),
        structure: birth_only(),
        window_requests: 4,
        held_out_codes: vec!["a", "newcode"],
        trainer,
        learn: false,
        filler: 0,
    }
}

fn only_admitted_birth(report: &DriverReport) -> &TrainedSpecialist {
    assert_eq!(report.outcomes.len(), 1, "{report:?}");
    match &report.outcomes[0].result {
        StructureResult::Admitted {
            specialist: Some(specialist),
            ..
        } => specialist,
        other => panic!("expected an admitted birth, got {other:?}"),
    }
}

// --- (a) a code with no repair Pod gets one -------------------------------------------------

#[test]
fn an_unroutable_code_is_born_through_admission_and_the_next_requests_are_repaired() {
    let mut world = World::new(newcode_opts(trainer(0.95)));

    // No Pod repairs `newcode`: every request that carries it ends Exhausted, and the router
    // counts the code as unroutable in every window.
    let before = world.serve("before", 24);
    assert_eq!(before.verified, 0, "{before:?}");
    assert!(world.vcw.windows_closed() >= 6);
    let unroutable: u64 = world
        .vcw
        .closed_windows()
        .map(|w| w.codes.get(&code("newcode")).map_or(0, |s| s.unroutable))
        .sum();
    assert!(unroutable >= 24, "{unroutable}");
    let commits = world.rt.committed_events().len();
    assert_eq!(world.structural_commits(), 0);

    // The driver closes the windows into a Birth request, trains, measures, gates, admits.
    let report = world.l3();
    assert!(report.ran_policy);
    let born = only_admitted_birth(&report).clone();
    assert_eq!(report.outcomes[0].kind, OpKind::Birth);
    assert_eq!(born.provenance, SimulatedTrainer::PROVENANCE);
    assert_eq!(world.trainer.trained(), 1);

    // Exactly one new record, and it is the structural one.
    assert_eq!(world.rt.committed_events().len(), commits + 1);
    assert_eq!(world.structural_commits(), 1);
    assert!(matches!(
        world.rt.committed_events().last().unwrap().event,
        LedgerEvent::ExpertStructuralCommitted { .. }
    ));
    let entry = world
        .rt
        .expert_registry()
        .get(&project(), &born.pod)
        .expect("the expert is in the committed registry");
    assert_eq!(entry.state, ExpertState::Serving);
    assert_eq!(entry.manifest_digest, born.manifest_digest);
    // The roster was enrolled by sync_roster, for the code the request named only.
    assert_eq!(
        world.vcw.repairers()[&born.pod],
        BTreeSet::from([code("newcode")])
    );
    assert_eq!(
        world.vcw.router().registry().repair_pods(&code("newcode")),
        vec![born.pod.clone()]
    );
    assert_eq!(world.vcw.outstanding_requests(), 0);

    // The next requests route the code to the born Pod and are Verified.
    let after = world.serve("after", 24);
    assert!(after.verified >= 22, "{after:?}");
    assert!(after.routed_to(&born.pod.0) >= 20, "{after:?}");
    // Nothing else in the structure changed.
    assert_eq!(world.structural_commits(), 1);
}

/// Review of #141 (round 5): a Birth whose result id is already taken by an unrelated Pod in
/// the PodRegistry is refused before admission, with nothing appended to the ledger and no
/// install left pending (it would otherwise be committed and never installable).
#[test]
fn a_birth_colliding_with_a_registered_pod_is_refused_before_admission() {
    // The id the policy picks for the born Pod, from an identical world.
    let born = {
        let mut probe = World::new(newcode_opts(trainer(0.95)));
        probe.serve("before", 24);
        only_admitted_birth(&probe.l3()).pod.clone()
    };
    let mut world = World::new(newcode_opts(trainer(0.95)));
    let intruder = SimulatedTrainer::new(project(), DefectDoc::clean_type(), 4_242, 0.5)
        .train(&TrainingRequest {
            kind: OpKind::Birth,
            pod: born.clone(),
            codes: BTreeSet::from([code("other")]),
            sources: Vec::new(),
        })
        .unwrap();
    world.pods.register_new(intruder.executable()).unwrap();
    world.serve("before", 24);
    let commits = world.rt.committed_events().len();
    let report = world.l3();
    assert_eq!(report.outcomes.len(), 1, "{report:?}");
    assert!(
        matches!(
            &report.outcomes[0].result,
            StructureResult::Refused(Refusal::PodIdTaken { pod }) if pod == &born
        ),
        "{report:?}"
    );
    assert_eq!(
        world.rt.committed_events().len(),
        commits,
        "nothing appended"
    );
    assert_eq!(world.structural_commits(), 0);
    assert!(world.driver.pending_installs().is_empty());
    assert_eq!(
        world
            .pods
            .get(&project(), &born)
            .unwrap()
            .manifest()
            .digest(),
        intruder.manifest.digest(),
        "the registered Pod is untouched"
    );
}

// --- (c) a failing gate report commits nothing ---------------------------------------------

#[test]
fn a_specialist_that_hurts_another_code_fails_the_gate_and_nothing_is_committed() {
    // Training the specialist for `newcode` is configured to wreck `fixer_a`'s repair of `a`.
    let harmful = trainer(0.95).with_collateral("fixer_a", "a", 0.0);
    let mut world = World::new(newcode_opts(harmful));
    world.serve("before", 24);
    let commits = world.rt.committed_events().len();
    let registry = world.rt.expert_registry().clone();
    let roster = world.vcw.repairers().clone();

    let report = world.l3();
    assert_eq!(report.outcomes.len(), 1, "{report:?}");
    match &report.outcomes[0].result {
        StructureResult::Refused(Refusal::Admission { gate_failed, .. }) => {
            assert!(gate_failed, "the runtime refused for another reason");
        }
        other => panic!("expected the runtime to refuse the report, got {other:?}"),
    }
    // Nothing committed, nothing enrolled, nothing registered, nothing outstanding.
    assert_eq!(world.rt.committed_events().len(), commits);
    assert_eq!(world.rt.expert_registry(), &registry);
    assert_eq!(world.vcw.repairers(), &roster);
    assert_eq!(world.vcw.outstanding_requests(), 0);
    let still = world.serve("still", 8);
    assert_eq!(still.verified, 0);

    // The refusal is not sticky: a specialist that costs nothing passes the same gate later.
    world.trainer = trainer(0.95);
    world.suite = world.suite.clone().with_world_of(&world.trainer);
    world.serve("more", 8);
    let retry = world.l3();
    let born = only_admitted_birth(&retry).clone();
    assert_eq!(world.structural_commits(), 1);
    assert!(world
        .rt
        .expert_registry()
        .get(&project(), &born.pod)
        .is_some());
}

/// A suite that returns the same clear rate for every code it was given, before and after.
struct Canned(BTreeMap<FindingCode, f64>);

impl HeldOutSuite for Canned {
    fn manifest_digest(&self) -> [u8; 32] {
        [0xC4; 32]
    }

    fn clear_rates(
        &self,
        _roster: &ptr_runtime::vcw_host::RosterView,
        _change: &ptr_runtime::vcw_host::RosterChange<'_>,
    ) -> Result<BTreeMap<FindingCode, f64>, ptr_runtime::vcw_host::SuiteError> {
        Ok(self.0.clone())
    }
}

// --- the suite is part of the evidence -----------------------------------------------------

#[test]
fn a_suite_that_does_not_measure_every_code_of_the_roster_and_the_candidate_is_refused() {
    // The serving roster repairs `a`; the suite measures only `newcode`. Whatever the Birth does
    // to `a` would be measured as "nothing lost", so nothing may be bound or committed.
    let mut opts = newcode_opts(trainer(0.95));
    opts.held_out_codes = vec!["newcode"];
    let mut world = World::new(opts);
    world.serve("before", 24);
    let commits = world.rt.committed_events().len();
    let report = world.l3();
    assert_eq!(report.outcomes.len(), 1, "{report:?}");
    match &report.outcomes[0].result {
        StructureResult::Refused(Refusal::Measurement(MeasurementError::Coverage { missing })) => {
            assert_eq!(missing, &BTreeSet::from([code("a")]));
        }
        other => panic!("expected a coverage refusal, got {other:?}"),
    }
    assert_eq!(world.rt.committed_events().len(), commits);
    assert_eq!(world.vcw.outstanding_requests(), 0);

    // A suite that misses the candidate's own code is refused the same way.
    let mut opts = newcode_opts(trainer(0.95));
    opts.held_out_codes = vec!["a"];
    let mut world = World::new(opts);
    world.serve("before", 24);
    let report = world.l3();
    assert!(matches!(
        &report.outcomes[0].result,
        StructureResult::Refused(Refusal::Measurement(MeasurementError::Coverage { missing }))
            if missing == &BTreeSet::from([code("newcode")])
    ));
    assert_eq!(world.structural_commits(), 0);
}

#[test]
fn the_bound_subject_carries_the_digest_of_the_suite_that_measured() {
    let mut world = World::new(newcode_opts(trainer(0.95)));
    world.serve("before", 24);
    let measured_by = world.suite.clone();
    // A different suite (another seed): a different digest.
    let other = SimulatedHeldOut::new(
        SimulationConfig::new(31)
            .with_error_rate("a", 0.5)
            .with_error_rate("newcode", 1.0)
            .with_pod_repair_success("fixer_a", "a", 0.95),
        project(),
        &["a", "newcode"],
        30,
        8,
    )
    .unwrap()
    .with_world_of(&world.trainer);
    assert_ne!(other.manifest_digest(), measured_by.manifest_digest());
    // The host's grant expects the digest of the first suite. A report measured by the other
    // one is bound to the other one's digest and admission refuses it as a different
    // evaluation manifest, not for its numbers.
    let report = world.l3_with(&other, measured_by.manifest_digest());
    match &report.outcomes[0].result {
        StructureResult::Refused(Refusal::Admission {
            gate_failed,
            message,
        }) => {
            assert!(!gate_failed);
            assert!(
                message.contains("EvidenceEvalManifestMismatch"),
                "{message}"
            );
        }
        other => panic!("expected an evaluation-manifest refusal, got {other:?}"),
    }
    assert_eq!(world.structural_commits(), 0);
    assert_eq!(world.vcw.outstanding_requests(), 0);
    // The suite that the grant expects is admitted (same world, same step).
    world.serve("more", 8);
    let admitted = world.l3_with(&measured_by, measured_by.manifest_digest());
    only_admitted_birth(&admitted);
    assert_eq!(world.structural_commits(), 1);
}

#[test]
fn what_the_trainer_claims_about_its_specialist_is_not_what_the_suite_measures() {
    // The trainer claims a harmless specialist (no collateral); in the simulated world training
    // it wrecks `fixer_a`. The suite measures the world, so the gate fails.
    let wrecks = trainer(0.95).with_collateral("fixer_a", "a", 0.0);
    let mut world = World::new(newcode_opts(trainer(0.95)));
    world.suite = world.suite.clone().with_world_of(&wrecks);
    world.serve("before", 24);
    let report = world.l3();
    match &report.outcomes[0].result {
        StructureResult::Refused(Refusal::Admission { gate_failed, .. }) => assert!(gate_failed),
        other => panic!("expected the gate to fail on the measured collateral, got {other:?}"),
    }
    assert_eq!(world.structural_commits(), 0);
}

#[test]
fn a_committed_expert_that_cannot_be_installed_is_reported_kept_and_retried() {
    // The loop's roster is full: fixer_a, the expert e_old and 254 fillers are 256 repairers.
    // The Birth is admitted (the ledger has room) but the loop cannot enrol one more.
    let mut world = World::new(Opts {
        error_rates: vec![("a", 0.5), ("newcode", 1.0)],
        fixer_a_skill: 0.95,
        seeds: vec![Seed {
            name: "e_old",
            codes: &["old"],
            skill: 0.9,
            with_inputs: false,
        }],
        structure: birth_only(),
        window_requests: 4,
        held_out_codes: vec!["a", "newcode", "old"],
        trainer: trainer(0.95),
        learn: false,
        filler: 254,
    });
    assert_eq!(world.vcw.repairers().len(), 256);
    let seeded = world.structural_commits();
    world.serve("before", 24);
    // A suite with canned rates: the simulated one cannot build the 257-repairer loop either,
    // which is its own refusal. Every roster code and the new one are measured, nothing lost.
    let mut rates: BTreeMap<FindingCode, f64> = world
        .vcw
        .repairers()
        .values()
        .flatten()
        .map(|c| (c.clone(), 0.9))
        .collect();
    rates.insert(code("newcode"), 0.9);
    let canned = Canned(rates);
    let report = world.l3_with(&canned, canned.manifest_digest());
    assert_eq!(report.outcomes.len(), 1, "{report:?}");
    assert!(report.outcomes[0].committed_not_installed(), "{report:?}");
    assert_eq!(report.pending_install(), 1);
    assert_eq!(report.refused(), 0, "a committed change is not a refusal");
    let message = match &report.outcomes[0].result {
        StructureResult::CommittedNotInstalled { message, .. } => message.clone(),
        other => panic!("{other:?}"),
    };
    assert!(message.contains("sync_roster"), "{message}");

    // The change is in the ledger; the expert is not routable; the state is observable.
    assert_eq!(world.structural_commits(), seeded + 1);
    let pending = world.driver.pending_installs();
    assert_eq!(pending.len(), 1);
    let born = pending[0].pods()[0].clone();
    assert_eq!(
        world
            .rt
            .expert_registry()
            .get(&project(), &born)
            .unwrap()
            .state,
        ExpertState::Serving
    );
    assert!(!world.vcw.repairers().contains_key(&born));
    assert_eq!(world.vcw.outstanding_requests(), 1, "the request must stay");
    assert!(world
        .vcw
        .router()
        .registry()
        .repair_pods(&code("newcode"))
        .is_empty());

    // A retry while the roster is still full fails again and keeps the entry.
    let again = world.l3_with(&canned, canned.manifest_digest());
    assert!(again.installed_late.is_empty());
    assert_eq!(world.driver.pending_installs().len(), 1);

    // The host frees a slot by pruning e_old through admission; the next step completes the
    // install on its own.
    world.prune_via_admission("e_old");
    let healed = world.l3_with(&canned, canned.manifest_digest());
    assert_eq!(healed.installed_late, vec![born.clone()]);
    assert!(world.driver.pending_installs().is_empty());
    assert!(world.vcw.repairers().contains_key(&born));
    assert!(!world.vcw.repairers().contains_key(&pod("e_old")));
    assert_eq!(world.vcw.outstanding_requests(), 0);
    assert_eq!(
        world.vcw.router().registry().repair_pods(&code("newcode")),
        vec![born]
    );
}

/// A world whose loop roster is full (256 repairers), so every Birth commits and cannot be
/// enrolled, with a suite that returns a clear rate for every code.
fn full_roster_world(codes: &[&'static str]) -> (World, Canned) {
    let mut error_rates = vec![("a", 0.5)];
    error_rates.extend(codes.iter().map(|c| (*c, 1.0)));
    let mut held_out_codes = vec!["a", "old"];
    held_out_codes.extend(codes.iter().copied());
    let world = World::new(Opts {
        error_rates,
        fixer_a_skill: 0.95,
        seeds: vec![Seed {
            name: "e_old",
            codes: &["old"],
            skill: 0.9,
            with_inputs: false,
        }],
        structure: birth_only(),
        window_requests: 4,
        held_out_codes,
        trainer: trainer(0.95),
        learn: false,
        filler: 254,
    });
    assert_eq!(world.vcw.repairers().len(), 256);
    let mut rates: BTreeMap<FindingCode, f64> = world
        .vcw
        .repairers()
        .values()
        .flatten()
        .map(|c| (c.clone(), 0.9))
        .collect();
    for c in codes {
        rates.insert(code(c), 0.9);
    }
    (world, Canned(rates))
}

#[test]
fn a_pending_install_is_abandoned_when_it_outlives_the_ttl_and_never_retried_forever() {
    let (mut world, canned) = full_roster_world(&["newcode"]);
    world.serve("before", 24);
    let digest = canned.manifest_digest();
    let first = world.l3_with(&canned, digest);
    assert!(first.outcomes[0].committed_not_installed(), "{first:?}");
    let born = world.driver.pending_installs()[0].pods()[0].clone();
    let commits = world.structural_commits();
    assert_eq!(world.vcw.outstanding_requests(), 1);
    assert!(first.abandoned_installs.is_empty());

    // Past outstanding_ttl_windows (16 windows of 4 requests) the roster is still full. The
    // driver retries first, fails, and gives up; the step itself still runs.
    world.serve("later", 80);
    assert!(world.vcw.windows_closed() >= 6 + u64::from(world.vcw.outstanding_ttl_windows()));
    let report = world.l3_with(&canned, digest);
    assert_eq!(report.abandoned_installs.len(), 1, "{report:?}");
    let abandoned = &report.abandoned_installs[0];
    assert_eq!(abandoned.pods, vec![born.clone()]);
    assert_eq!(abandoned.codes, BTreeSet::from([code("newcode")]));
    assert!(
        matches!(abandoned.reason, AbandonReason::Expired { windows } if windows >= 16),
        "{abandoned:?}"
    );
    assert!(abandoned.message.contains("sync_roster"), "{abandoned:?}");
    assert!(world.driver.pending_installs().is_empty());
    assert_eq!(world.vcw.outstanding_requests(), 0);
    // The committed expert is neither routable nor forgotten: the loop holds its code, so the
    // policy does not birth a second expert for it, now or in later windows.
    assert!(world.vcw.stranded_experts().contains_key(&born));
    assert!(!world.vcw.repairers().contains_key(&born));
    assert!(report.outcomes.is_empty(), "{report:?}");
    world.serve("again", 24);
    let next = world.l3_with(&canned, digest);
    assert!(next.ran_policy && next.outcomes.is_empty(), "{next:?}");
    assert!(next.abandoned_installs.is_empty());
    assert_eq!(world.structural_commits(), commits, "no second Birth");

    // The host retires the stranded expert through admission; the next step releases the hold.
    world.prune_via_admission(born.0.as_str());
    world.serve("last", 8);
    world.l3_with(&canned, digest);
    assert!(world.vcw.stranded_experts().is_empty());
}

#[test]
fn a_code_with_a_pending_install_is_not_born_again_while_the_install_is_pending() {
    let (mut world, canned) = full_roster_world(&["newcode"]);
    world.serve("before", 24);
    let digest = canned.manifest_digest();
    let first = world.l3_with(&canned, digest);
    assert!(first.outcomes[0].committed_not_installed(), "{first:?}");
    let commits = world.structural_commits();
    // Windows keep closing (well inside the TTL) and `newcode` stays unroutable.
    for round in 0..3 {
        world.serve(&format!("more{round}"), 8);
        let report = world.l3_with(&canned, digest);
        assert!(report.ran_policy, "{report:?}");
        assert!(report.outcomes.is_empty(), "{report:?}");
        assert_eq!(world.driver.pending_installs().len(), 1);
    }
    assert_eq!(world.structural_commits(), commits);
}

/// A committed-looking Birth proposal, only to name an operation for `strand_committed`.
fn any_birth_proposal(name: &str) -> ExpertProposal {
    let spec = ExpertPodSpec::new(
        project(),
        pod(name),
        ManifestDigest::new([3; 32]).unwrap(),
        ExpertLevel::Token {
            layer: LAYER,
            index: 400,
        },
        base(),
        Generation(1),
    )
    .unwrap();
    ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Birth { result: spec },
        RationaleDigest::new([6; 32]).unwrap(),
    )
    .unwrap()
}

fn hold_world() -> World {
    World::new(Opts {
        error_rates: vec![("a", 0.5)],
        fixer_a_skill: 0.95,
        seeds: vec![Seed {
            name: "e_old",
            codes: &["old"],
            skill: 0.9,
            with_inputs: false,
        }],
        structure: birth_only(),
        window_requests: 4,
        held_out_codes: vec!["a", "old"],
        trainer: trainer(0.95),
        learn: false,
        filler: 0,
    })
}

#[test]
fn a_stranded_hold_on_an_expert_the_projection_does_not_know_is_released_at_the_next_step() {
    let mut world = hold_world();
    world
        .vcw
        .strand_committed(
            &any_birth_proposal("ghost"),
            &[pod("ghost")],
            &BTreeSet::from([code("ghost")]),
        )
        .unwrap();
    assert!(world.vcw.stranded_experts().contains_key(&pod("ghost")));
    world.serve("w", 8);
    world.l3();
    assert!(world.vcw.stranded_experts().is_empty());
}

#[test]
fn a_stranded_hold_ends_when_the_expert_is_superseded_and_not_before() {
    let mut world = hold_world();
    world
        .vcw
        .strand_committed(
            &any_birth_proposal("e_old"),
            &[pod("e_old")],
            &BTreeSet::from([code("old")]),
        )
        .unwrap();
    // Serving: the hold stays through a step (the control for the release below).
    world.serve("w1", 8);
    world.l3();
    assert!(world.vcw.stranded_experts().contains_key(&pod("e_old")));

    world.retire_via_admission("e_old", RetiredReason::Superseded);
    assert_eq!(
        world
            .rt
            .expert_registry()
            .get(&project(), &pod("e_old"))
            .unwrap()
            .state,
        ExpertState::Retired(RetiredReason::Superseded)
    );
    world.serve("w2", 8);
    world.l3();
    assert!(world.vcw.stranded_experts().is_empty());
}

#[test]
fn a_step_error_after_an_abandon_still_reports_the_abandoned_install() {
    let (mut world, canned) = full_roster_world(&["newcode"]);
    world.serve("before", 24);
    let digest = canned.manifest_digest();
    let first = world.l3_with(&canned, digest);
    assert!(first.outcomes[0].committed_not_installed(), "{first:?}");
    let born = world.driver.pending_installs()[0].pods()[0].clone();
    world.serve("later", 80);
    // A non-finite AIMER score makes the loop refuse the step itself, after the driver has
    // already abandoned the install that outlived the TTL.
    world.aimer = BTreeMap::from([(pod("e_old"), f64::NAN)]);
    let grant = move |rt: &PtrRuntime| grant_expecting(rt, digest);
    let error = world
        .driver
        .step(
            &mut world.vcw,
            DriverHost {
                runtime: &mut world.rt,
                pods: &mut world.pods,
                grant: &grant,
                trainer: &mut world.trainer,
                suite: &canned,
                aimer: &world.aimer,
            },
        )
        .expect_err("a non-finite signal must refuse the step");
    let DriverError::Loop {
        error,
        abandoned_installs,
        ..
    } = error
    else {
        panic!("unexpected error shape");
    };
    assert!(
        matches!(*error, ptr_runtime::vcw::VcwError::Structure(_)),
        "{error:?}"
    );
    assert_eq!(
        abandoned_installs.len(),
        1,
        "the notification must not be lost"
    );
    assert_eq!(abandoned_installs[0].pods, vec![born.clone()]);
    assert!(matches!(
        abandoned_installs[0].reason,
        AbandonReason::Expired { .. }
    ));
    // The effect is real: the hold is recorded and the install is no longer tracked, so the
    // error is the only place the notification survives.
    assert!(world.vcw.stranded_experts().contains_key(&born));
    assert!(world.driver.pending_installs().is_empty());
}

#[test]
fn at_the_pending_cap_new_structural_steps_are_refused_and_nothing_is_dropped() {
    let (mut world, canned) = full_roster_world(&["newcode", "newcode2"]);
    world.driver = StructureDriver::new(
        project(),
        proposer(),
        base(),
        BirthSite::Token {
            layer: LAYER,
            slots: 16,
        },
        gate(),
    )
    .with_max_pending(1);
    world.serve("before", 24);
    let seeded = world.structural_commits();
    let report = world.l3_with(&canned, canned.manifest_digest());
    assert_eq!(report.outcomes.len(), 2, "{report:?}");
    let pending = report
        .outcomes
        .iter()
        .filter(|o| o.committed_not_installed())
        .count();
    assert_eq!(pending, 1, "{report:?}");
    let refused: Vec<_> = report
        .outcomes
        .iter()
        .filter_map(|o| match &o.result {
            StructureResult::Refused(refusal) => Some(refusal.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        refused,
        vec![Refusal::PendingInstallsFull { pending: 1 }],
        "{report:?}"
    );
    // Only the first was committed, and it is still tracked; the refused request is not held.
    assert_eq!(world.structural_commits(), seeded + 1);
    assert_eq!(world.driver.pending_installs().len(), 1);
    assert_eq!(world.vcw.outstanding_requests(), 1);
}

// --- (d) no commit without admission --------------------------------------------------------

#[test]
fn neither_the_loop_nor_a_forged_record_nor_a_foreign_report_can_change_the_structure() {
    let mut world = World::new(newcode_opts(trainer(0.95)));
    world.serve("before", 24);
    let commits = world.rt.committed_events().len();
    let registry = world.rt.expert_registry().clone();

    // 1. The loop's own structure step returns inert requests and writes nothing.
    let report = world
        .vcw
        .structure_step(StructureHost {
            project: project(),
            proposer: proposer(),
            observed_at: CommitIndex(1),
            registry: world.rt.expert_registry(),
            birth: Some(BirthContext {
                base_model: base(),
                site: BirthSite::Token {
                    layer: LAYER,
                    slots: 16,
                },
            }),
            aimer: BTreeMap::new(),
            overlaps: Vec::new(),
        })
        .unwrap();
    assert_eq!(report.proposals().len(), 1);
    assert!(report.proposals()[0].needs_training());
    assert!(
        report.proposals()[0].ready().is_none(),
        "an untrained request is not a proposal"
    );
    assert_eq!(world.rt.committed_events().len(), commits);

    // 2. A structural record handed to the public commit is refused.
    let forged = LedgerEvent::ExpertStructuralCommitted {
        project: project(),
        op_kind: ptr_ledger::ExpertOpKind::Birth,
        proposal_id: [1; 32],
        operation_id: [2; 32],
        evidence_digest: [3; 32],
        record: b"PTREXR01-forged".to_vec(),
    };
    assert_eq!(
        world.rt.commit(forged).err(),
        Some(RuntimeError::StructuralRecordOutsideAdmissionPath)
    );

    // 3. A completed proposal with a gate report bound to another proposal is refused.
    let trained = trainer(0.9)
        .train(&TrainingRequest {
            kind: OpKind::Birth,
            pod: pod("intruder"),
            codes: BTreeSet::from([code("newcode")]),
            sources: Vec::new(),
        })
        .unwrap();
    let spec = ExpertPodSpec::new(
        project(),
        pod("intruder"),
        trained.manifest_digest,
        ExpertLevel::Token {
            layer: LAYER,
            index: 5,
        },
        base(),
        Generation(1),
    )
    .unwrap();
    let make = |digest: u8| {
        ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(1),
            StructuralOp::Birth {
                result: spec.clone(),
            },
            RationaleDigest::new([digest; 32]).unwrap(),
        )
        .unwrap()
    };
    let (mine, other) = (make(7), make(8));
    let foreign = seed_report(&other, None);
    let grant = grant_for(&world.rt);
    assert_eq!(
        world
            .rt
            .admit_expert_proposal(&mine, &foreign, &grant)
            .err(),
        Some(RuntimeError::Expert(
            ExpertAdmissionError::EvidenceProposalMismatch
        ))
    );
    // 4. A report that fails the gate is refused by the runtime.
    let failing_matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.1, 0.9]]).unwrap();
    let subject = GateSubject::new(
        *mine.proposal_id().as_bytes(),
        *mine.operation_id().as_bytes(),
        expert_evidence_artifact_digest(mine.op()),
        expert_evidence_base_model_digest(mine.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let failing = gate()
        .evaluate(
            &AdapterId::from("bad"),
            &failing_matrix,
            PublicSuite {
                serving: 0.9,
                candidate: 0.9,
            },
        )
        .unwrap()
        .bind(subject);
    assert!(!failing.passed());
    assert_eq!(
        world
            .rt
            .admit_expert_proposal(&mine, &failing, &grant)
            .err(),
        Some(RuntimeError::Expert(ExpertAdmissionError::GateFailed))
    );
    assert_eq!(world.rt.committed_events().len(), commits);
    assert_eq!(world.rt.expert_registry(), &registry);

    // 5. A driver step before any new window closed asks nothing of the policy.
    world.l3();
    let idle = world.l3();
    assert!(!idle.ran_policy);
    assert!(idle.outcomes.is_empty());
}

#[test]
fn a_trainer_that_cannot_train_leaves_the_structure_untouched() {
    // The trainer refuses a skill outside 0..=1: the request is withdrawn, nothing is committed.
    let mut world = World::new(newcode_opts(trainer(0.95).with_skill("newcode", 1.5)));
    world.serve("before", 24);
    let commits = world.rt.committed_events().len();
    let report = world.l3();
    assert_eq!(report.outcomes.len(), 1);
    assert!(matches!(
        report.outcomes[0].result,
        StructureResult::Refused(Refusal::Training(_))
    ));
    assert_eq!(world.rt.committed_events().len(), commits);
    assert_eq!(world.vcw.outstanding_requests(), 0);
}

// --- (e) replay reproduces the roster ------------------------------------------------------

#[test]
fn replaying_the_ledger_reproduces_the_expert_registry_and_the_roster() {
    let mut world = World::new(newcode_opts(trainer(0.95)));
    world.serve("before", 24);
    let born = only_admitted_birth(&world.l3()).clone();
    assert_eq!(world.structural_commits(), 1);

    let events = world.rt.committed_events().to_vec();
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    assert_eq!(replayed.expert_registry(), world.rt.expert_registry());

    // The roster: base Pods plus every routable expert, with the codes its manifest declares.
    let rebuilt = rebuild_roster(
        world.base_roster.clone(),
        replayed.expert_registry(),
        &world.pods,
        &project(),
    )
    .unwrap();
    let as_map: BTreeMap<PodId, BTreeSet<FindingCode>> = rebuilt
        .repairers
        .iter()
        .map(|(p, c)| (p.clone(), c.iter().cloned().collect()))
        .collect();
    assert_eq!(&as_map, world.vcw.repairers());
    assert!(as_map.contains_key(&born.pod));

    // A fresh loop built from the rebuilt roster, on the replayed runtime, serves the code.
    let mut vcw = VcwLoop::new(VcwParts {
        config: VcwConfig::new("replayed", [4; 32]),
        roster: rebuilt,
        router: RouterConfig::default(),
        scorer: PolicyBook::new("defect-doc", Exploration::default())
            .unwrap()
            .scorer(),
        key: SamplingKey::from_bytes([8; 32]),
        nonces: NonceSequence::starting_at([8; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(birth_only()).unwrap(),
    })
    .unwrap();
    let proj = project();
    let standing = RuntimeStanding::new(&replayed, &world.pods, &proj)
        .with_base_pod(pod("gen"), Generation(1))
        .with_base_pod(pod("fixer_a"), Generation(1));
    let mut verified = 0;
    for i in 0..10 {
        let id = format!("replayed-{i}");
        let request = VcwRequest {
            request_id: RequestId::from(id.as_str()),
            project: project(),
            request_digest: Sha256::digest(id.as_bytes()).into(),
            required: VerificationLevel::SampleVerified,
            budget: RequestBudget::new(8, 6).unwrap(),
        };
        let run = vcw
            .run(
                &request,
                &standing,
                &DefectDocVerifier,
                &mut world.experts.guarded(&standing),
            )
            .unwrap();
        if matches!(run.outcome, VcwOutcome::Verified(_)) {
            verified += 1;
        }
    }
    assert!(verified >= 9, "{verified}");

    // A manifest swapped after the commit is detected by its digest; a missing Pod is refused.
    let mut tampered = PodRegistry::default();
    for registered in world.experts.pods(&project(), &DefectDoc::clean_type()) {
        if registered.manifest().id != born.pod {
            tampered.register_new(registered).unwrap();
        }
    }
    assert_eq!(
        rebuild_roster(
            world.base_roster.clone(),
            replayed.expert_registry(),
            &tampered,
            &project()
        )
        .unwrap_err(),
        RosterRebuildError::MissingPod {
            pod: born.pod.clone()
        }
    );
    let other = trainer(0.95)
        .train(&TrainingRequest {
            kind: OpKind::Birth,
            pod: born.pod.clone(),
            codes: BTreeSet::from([code("newcode")]),
            sources: Vec::new(),
        })
        .unwrap();
    let mut other_seed = SimulatedTrainer::new(project(), DefectDoc::clean_type(), 100, 0.95);
    let swapped = other_seed
        .train(&TrainingRequest {
            kind: OpKind::Birth,
            pod: born.pod.clone(),
            codes: BTreeSet::from([code("newcode")]),
            sources: Vec::new(),
        })
        .unwrap();
    assert_eq!(other.manifest_digest, born.manifest_digest);
    tampered.register_new(swapped.executable()).unwrap();
    assert_eq!(
        rebuild_roster(
            world.base_roster.clone(),
            replayed.expert_registry(),
            &tampered,
            &project()
        )
        .unwrap_err(),
        RosterRebuildError::ManifestDigestMismatch { pod: born.pod }
    );
}

#[test]
fn a_base_pod_whose_manifest_declares_no_code_keeps_its_base_codes_on_rebuild() {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    let mut pods = PodRegistry::default();
    let mut seeding = trainer(0.9);
    let seed = Seed {
        name: "e_base",
        codes: &["x"],
        skill: 0.9,
        with_inputs: false,
    };
    seed_expert_declaring(&mut rt, &mut pods, &mut seeding, &seed, 100, false);
    let base = Roster {
        proposers: vec![pod("gen")],
        repairers: vec![(pod("e_base"), vec![code("x")])],
    };
    // The expert is routable in the ledger and its manifest names no code: the base roster's
    // codes must survive the rebuild, or the Pod would serve nothing after a restart.
    let rebuilt = rebuild_roster(base, rt.expert_registry(), &pods, &project()).unwrap();
    assert_eq!(rebuilt.repairers, vec![(pod("e_base"), vec![code("x")])]);
}

// --- (b) learning starves a bad Pod, L3 prunes it ---------------------------------------------

#[test]
fn a_pod_the_learned_router_starves_is_pruned_through_admission_and_never_routed_again() {
    let structure = StructurePolicyConfig {
        ops: EnabledOps {
            birth: false,
            merge: false,
            prune: true,
            freeze: false,
            reactivate: false,
        },
        prune_usage_floor: 5,
        freeze_min_usage: 5,
        prune_windows: 2,
        prune_min_age_commits: 0,
        backup_min_cleared: 1,
        backup_clear_windows: 2,
        ..StructurePolicyConfig::default()
    };
    let mut world = World::new(Opts {
        error_rates: vec![("x", 1.0)],
        fixer_a_skill: 0.9,
        seeds: vec![
            Seed {
                name: "e_good",
                codes: &["x"],
                skill: 0.9,
                with_inputs: false,
            },
            Seed {
                name: "e_bad",
                codes: &["x"],
                skill: 0.05,
                with_inputs: false,
            },
        ],
        structure,
        window_requests: 20,
        held_out_codes: vec!["x", "a"],
        trainer: trainer(0.9),
        learn: true,
        filler: 0,
    });
    world.aimer = BTreeMap::from([(pod("e_bad"), 0.9), (pod("e_good"), 0.1)]);

    // Phase 1, uniform routing: both experts are used, so there is nothing to prune.
    let uniform = world.serve("uniform", 400);
    assert!(uniform.routed_to("e_bad") > 100, "{uniform:?}");
    let idle = world.l3();
    assert!(idle.ran_policy);
    assert!(idle.outcomes.is_empty(), "{idle:?}");

    // Phase 2: the learner has seen the verdicts; the host promotes on held-out evidence.
    let learner = world.learner.clone().unwrap();
    let candidate = learner.snapshot();
    world.serve("held-out", 900);
    let assessment = PromotionGate::new(PromotionRule::new(0.05, 500, 100.0))
        .unwrap()
        .assess(&learner, &candidate, &world.book.scorer())
        .unwrap();
    let Verdict::Promote(evidence) = assessment.verdict.clone() else {
        panic!("{assessment:?}");
    };
    world
        .book
        .promote(candidate, evidence, &mut world.rt)
        .unwrap();

    // Phase 3: the bad expert is reachable only through the router's floor, so its usage in
    // every window stays under the policy's floor.
    let learned = world.serve("learned", 60);
    assert!(learned.routed_to("e_good") > 40, "{learned:?}");
    assert!(learned.routed_to("e_bad") < 15, "{learned:?}");
    let commits = world.structural_commits();
    let report = world.l3();
    assert_eq!(report.outcomes.len(), 1, "{report:?}");
    assert_eq!(report.outcomes[0].kind, OpKind::Prune);
    match &report.outcomes[0].result {
        StructureResult::Admitted {
            removed,
            specialist,
            ..
        } => {
            assert_eq!(removed, &vec![pod("e_bad")]);
            assert!(specialist.is_none());
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(world.structural_commits(), commits + 1);
    assert_eq!(
        world
            .rt
            .expert_registry()
            .get(&project(), &pod("e_bad"))
            .unwrap()
            .state,
        ExpertState::Retired(RetiredReason::Pruned)
    );
    assert!(!world.vcw.repairers().contains_key(&pod("e_bad")));
    assert_eq!(
        world.vcw.router().registry().repair_pods(&code("x")),
        vec![pod("e_good")]
    );

    // Never routed again; the good expert carries the code.
    let skipped = world.vcw.router().ineligible_skipped();
    let after = world.serve("after", 100);
    assert_eq!(after.routed_to("e_bad"), 0, "{after:?}");
    assert!(after.verified >= 95, "{after:?}");
    assert_eq!(world.vcw.router().ineligible_skipped(), skipped);
    // And the good expert, the last demonstrated clearer of `x`, is never proposed for pruning.
    world.serve("later", 80);
    let later = world.l3();
    assert!(later.outcomes.is_empty(), "{later:?}");
}

// --- revocation ------------------------------------------------------------------------------

#[test]
fn revoking_a_dataset_a_born_pod_rests_on_removes_it_from_routing_with_its_dependents() {
    let mut world = World::new(Opts {
        error_rates: vec![("newcode", 1.0), ("y", 0.5)],
        fixer_a_skill: 0.9,
        seeds: vec![
            Seed {
                name: "e_dep",
                codes: &["y"],
                skill: 0.9,
                with_inputs: true,
            },
            Seed {
                name: "e_free",
                codes: &["z"],
                skill: 0.9,
                with_inputs: false,
            },
        ],
        structure: birth_only(),
        window_requests: 4,
        held_out_codes: vec!["a", "y", "z", "newcode"],
        trainer: trainer(0.95).with_inputs(dataset_inputs()),
        learn: false,
        filler: 0,
    });
    world.serve("before", 24);
    let born = only_admitted_birth(&world.l3()).clone();
    let served = world.serve("serving", 12);
    assert!(
        served.verified >= 11 && served.routed_to(&born.pod.0) > 0,
        "{served:?}"
    );

    // The dataset is revoked: one committed event fences every expert that rests on it.
    let grant = ExpertFenceGrant::global_revocation(current_binding(
        &world.rt,
        ExpertGrantKind::GlobalRevocation,
    ));
    let outcome = world
        .rt
        .invalidate_input(&grant, InputInvalidation::revoked(TRAIN, Generation(1), D1))
        .unwrap();
    let fenced: BTreeSet<PodId> = outcome.fenced().iter().map(|m| m.pod.clone()).collect();
    assert_eq!(fenced, BTreeSet::from([pod("e_dep"), born.pod.clone()]));
    let state = |name: &PodId| {
        world
            .rt
            .expert_registry()
            .get(&project(), name)
            .unwrap()
            .state
    };
    assert_eq!(state(&born.pod), ExpertState::Fenced);
    assert_eq!(state(&pod("e_dep")), ExpertState::Fenced);
    assert_eq!(state(&pod("e_free")), ExpertState::Serving);

    // Routing no longer reaches the born Pod: its code is unroutable again.
    let skipped = world.vcw.router().ineligible_skipped();
    let after = world.serve("after", 12);
    assert_eq!(after.routed_to(&born.pod.0), 0, "{after:?}");
    assert_eq!(after.verified, 0, "{after:?}");
    assert!(world.vcw.router().ineligible_skipped() > skipped);
    let proj = project();
    let standing = RuntimeStanding::new(&world.rt, &world.pods, &proj)
        .with_base_pod(pod("gen"), Generation(1));
    assert_eq!(
        ptr_router::PodEligibility::current_generation(&standing, &born.pod),
        None
    );

    // Replay re-derives the same fence set.
    let replayed = PtrRuntime::replay(PtrConfig::default(), world.rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), world.rt.expert_registry());
}
