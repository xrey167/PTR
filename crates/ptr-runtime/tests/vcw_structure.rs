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
    ExpertPodSpec, ExpertProposal, ExpertState, OpKind, PodRegistry, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_router::LearnerConfig;
use ptr_router::{
    ActiveScorer, FindingCode, NonceSequence, PolicyBook, PromotionGate, PromotionRule, RouteClass,
    RouterConfig, RoutingLearner, SamplingKey, SelfModel, SelfModelLimits, Verdict,
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
    rebuild_roster, DriverHost, DriverReport, Refusal, RosterRebuildError, SimulatedHeldOut,
    SimulatedTrainer, SpecialistTrainer, StructureDriver, StructureResult, TrainedSpecialist,
    TrainingRequest,
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

fn grant_for(rt: &PtrRuntime) -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
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
    let mut trained = trainer
        .train(&TrainingRequest {
            kind: OpKind::Birth,
            pod: pod(seed.name),
            codes: seed.codes.iter().map(|c| code(c)).collect(),
            sources: Vec::new(),
        })
        .unwrap();
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
        let mut pods = PodRegistry::default();
        for registered in experts.pods(&project(), &DefectDoc::clean_type()) {
            pods.register_new(registered).unwrap();
        }
        let mut seeding = trainer(0.9);
        let mut roster = Roster {
            proposers: vec![pod("gen")],
            repairers: vec![(pod("fixer_a"), vec![code("a")])],
        };
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
        let book = PolicyBook::new("defect-doc");
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
        let suite = SimulatedHeldOut::new(sim, project(), &opts.held_out_codes, 30, 7).unwrap();
        let driver = StructureDriver::new(
            project(),
            proposer(),
            base(),
            BirthSite::Token {
                layer: LAYER,
                slots: 16,
            },
            gate(),
            EVAL_MANIFEST,
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
        let grant = grant_for;
        let report = self
            .driver
            .step(
                &mut self.vcw,
                DriverHost {
                    runtime: &mut self.rt,
                    pods: &mut self.pods,
                    grant: &grant,
                    trainer: &mut self.trainer,
                    suite: &self.suite,
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
        scorer: PolicyBook::new("defect-doc").scorer(),
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
    let log = learner.log_from(candidate.cutoff());
    let assessment = PromotionGate::new(PromotionRule::new(0.05, 500, 100.0))
        .unwrap()
        .assess(
            &log,
            &candidate,
            &world.book.scorer(),
            world.book.active_digest(),
        )
        .unwrap();
    let Verdict::Promote(evidence) = assessment.verdict.clone() else {
        panic!("{assessment:?}");
    };
    world.book.promote(candidate, evidence).unwrap();

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
        held_out_codes: vec!["y", "z", "newcode"],
        trainer: trainer(0.95).with_inputs(dataset_inputs()),
        learn: false,
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
