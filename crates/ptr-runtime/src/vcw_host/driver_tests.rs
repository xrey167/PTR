//! Abandoned installs of Merge, Reactivate and Birth: what the driver does to the loop's roster
//! and its holds. The driver cannot reach a failing Merge or Reactivate install through its
//! public API (it passes no subspace overlaps, and a Merge frees roster slots before it enrols),
//! so these tests build the `PendingInstall` by hand around a change that the runtime really
//! committed.

#[path = "../../tests/expert_support/mod.rs"]
mod expert_support;

use std::collections::BTreeSet;

use ptr_config::PtrConfig;
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, OpKind, PodRegistry, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_router::{
    FindingCode, NonceSequence, RouterConfig, SamplingKey, SelfModel, SelfModelLimits,
    UniformScorer,
};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

use super::*;
use crate::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, ExpertAdmissionGrant,
    ExpertGrantKind, StructurePolicy, StructurePolicyConfig,
};
use crate::vcw::experts::DefectDoc;
use crate::vcw::{Roster, VcwConfig, VcwParts};
use crate::vcw_host::{SimulatedTrainer, SpecialistTrainer, TrainedSpecialist, TrainingRequest};
use expert_support::{current_binding, prime, requirement, EVAL_MANIFEST};

const LAYER: u16 = 3;

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

fn grant(rt: &PtrRuntime) -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        current_binding(rt, ExpertGrantKind::Admission),
    )
}

fn report(proposal: &ExpertProposal) -> BoundGateReport {
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate()
        .evaluate(
            &AdapterId::from("unit"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject)
}

/// What the driver tests operate on: a runtime with committed experts, the Pod registry, a loop
/// whose roster was set up by the test, and a driver.
struct Fixture {
    rt: PtrRuntime,
    pods: PodRegistry,
    vcw: VcwLoop,
    driver: StructureDriver,
    trainer: SimulatedTrainer,
    slots: u16,
}

impl Fixture {
    fn new(roster: &[(&str, &[&str])]) -> Self {
        let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
        prime(&mut rt);
        let vcw = VcwLoop::new(VcwParts {
            config: VcwConfig::new("driver-unit", [4; 32]),
            roster: Roster {
                proposers: vec![pod("gen")],
                repairers: roster
                    .iter()
                    .map(|(p, codes)| (pod(p), codes.iter().map(|c| code(c)).collect()))
                    .collect(),
            },
            router: RouterConfig::default(),
            scorer: UniformScorer,
            key: SamplingKey::from_bytes([5; 32]),
            nonces: NonceSequence::starting_at([4; 8], 0),
            self_model: SelfModel::new(SelfModelLimits::default()),
            schema: None,
            structure: StructurePolicy::new(StructurePolicyConfig::default()).unwrap(),
        })
        .unwrap();
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
            pods: PodRegistry::default(),
            vcw,
            driver,
            trainer: SimulatedTrainer::new(project(), DefectDoc::clean_type(), 99, 0.9),
            slots: 0,
        }
    }

    fn train(&mut self, name: &str, codes: &[&str]) -> TrainedSpecialist {
        self.trainer
            .train(&TrainingRequest {
                kind: OpKind::Birth,
                pod: pod(name),
                codes: codes.iter().map(|c| code(c)).collect(),
                sources: Vec::new(),
            })
            .unwrap()
    }

    fn spec(&self, trained: &TrainedSpecialist, slot: u16, generation: u64) -> ExpertPodSpec {
        ExpertPodSpec::new(
            project(),
            trained.pod.clone(),
            trained.manifest_digest,
            ExpertLevel::Token {
                layer: LAYER,
                index: slot,
            },
            base(),
            Generation(generation),
        )
        .unwrap()
    }

    fn proposal(op: StructuralOp) -> ExpertProposal {
        ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(1),
            op,
            RationaleDigest::new([7; 32]).unwrap(),
        )
        .unwrap()
    }

    fn admit(&mut self, op: StructuralOp) -> (ExpertProposal, CommitIndex) {
        let proposal = Self::proposal(op);
        let evidence = report(&proposal);
        let grant = grant(&self.rt);
        let commit = self
            .rt
            .admit_expert_proposal(&proposal, &evidence, &grant)
            .unwrap();
        (proposal, commit)
    }

    /// Commit a Birth of `name` for `codes` (not enrolled in the loop).
    fn birth(&mut self, name: &str, codes: &[&str]) -> TrainedSpecialist {
        let trained = self.train(name, codes);
        self.slots += 1;
        let result = self.spec(&trained, 100 + self.slots, 1);
        self.admit(StructuralOp::Birth { result });
        trained
    }

    fn target(&self, name: &str) -> ExpertTarget {
        let entry = self
            .rt
            .expert_registry()
            .get(&project(), &pod(name))
            .unwrap()
            .clone();
        ExpertTarget::new(ExpertTargetInit {
            project: project(),
            pod: pod(name),
            generation: entry.generation,
            level: entry.level,
            base_model: entry.base_model,
            manifest_digest: entry.manifest_digest,
            state: entry.state,
            state_version: entry.state_version,
        })
        .unwrap()
    }

    fn pend(
        &mut self,
        commit: CommitIndex,
        proposal: ExpertProposal,
        specialist: TrainedSpecialist,
        removed: Vec<PodId>,
        codes: &[&str],
    ) {
        self.driver.pending.push(PendingInstall {
            commit,
            proposal,
            specialist: Some(specialist),
            removed,
            codes: codes.iter().map(|c| code(c)).collect(),
            since_window: 0,
            message: "earlier failure".into(),
        });
    }

    fn retry(&mut self) -> InstallRetry {
        self.driver
            .retry_installs(&mut self.vcw, &self.rt, &mut self.pods)
    }
}

fn roster_pods(vcw: &VcwLoop) -> BTreeSet<PodId> {
    vcw.repairers().keys().cloned().collect()
}

#[test]
fn an_abandoned_merge_takes_its_superseded_sources_out_of_the_roster() {
    let mut f = Fixture::new(&[("a", &["x"]), ("b", &["x"])]);
    f.birth("a", &["x"]);
    f.birth("b", &["x"]);
    let merged = f.train("m", &["x"]);
    let result = f.spec(&merged, 200, 1);
    let (proposal, commit) = f.admit(StructuralOp::Merge {
        sources: vec![f.target("a"), f.target("b")],
        result,
    });
    assert_eq!(
        f.rt.expert_registry()
            .get(&project(), &pod("a"))
            .unwrap()
            .state,
        ExpertState::Superseded
    );
    // The loop lost the request: the install can never succeed.
    f.pend(commit, proposal, merged, vec![pod("a"), pod("b")], &["x"]);
    assert_eq!(roster_pods(&f.vcw), BTreeSet::from([pod("a"), pod("b")]));

    let retry = f.retry();
    assert_eq!(retry.abandoned.len(), 1, "{retry:?}");
    let abandoned = &retry.abandoned[0];
    assert_eq!(abandoned.reason, AbandonReason::RequestLost);
    assert_eq!(abandoned.pods, vec![pod("m")]);
    assert_eq!(abandoned.retire_error, None);
    // The superseded sources no longer occupy roster slots or routing.
    assert!(roster_pods(&f.vcw).is_empty(), "{:?}", roster_pods(&f.vcw));
    assert!(f.vcw.router().registry().repair_pods(&code("x")).is_empty());
    // The merged expert is stranded, not forgotten.
    assert!(f.vcw.stranded_experts().contains_key(&pod("m")));
    assert!(f.driver.pending_installs().is_empty());
}

#[test]
fn an_abandoned_reactivate_takes_dead_experts_out_of_the_roster() {
    // `dead` is Pruned and still in the loop's roster (a prune whose retire step never ran).
    let mut f = Fixture::new(&[("dead", &["y"])]);
    f.birth("dead", &["y"]);
    f.birth("a", &["x"]);
    let trained_a = f.train("a", &["x"]);
    let prune = StructuralOp::Prune {
        targets: vec![f.target("dead")],
    };
    f.admit(prune);
    let retire = StructuralOp::Retire {
        target: f.target("a"),
        reason: RetiredReason::GateFailed,
    };
    f.admit(retire);
    let target = f.target("a");
    let next = ExpertPodSpec::new(
        project(),
        pod("a"),
        trained_a.manifest_digest,
        target.level(),
        base(),
        Generation(target.generation().0 + 1),
    )
    .unwrap();
    let (proposal, commit) = f.admit(StructuralOp::Reactivate {
        target,
        result: Some(next),
    });
    f.pend(commit, proposal, trained_a, Vec::new(), &["x"]);
    assert!(roster_pods(&f.vcw).contains(&pod("dead")));

    let retry = f.retry();
    assert_eq!(retry.abandoned.len(), 1, "{retry:?}");
    assert_eq!(retry.abandoned[0].reason, AbandonReason::RequestLost);
    assert_eq!(retry.abandoned[0].retire_error, None);
    assert!(!roster_pods(&f.vcw).contains(&pod("dead")));
    assert!(f.vcw.stranded_experts().contains_key(&pod("a")));
}

#[test]
fn a_lost_birth_is_abandoned_as_request_lost_and_held() {
    let mut f = Fixture::new(&[]);
    let born = f.train("n", &["x"]);
    let result = f.spec(&born, 150, 1);
    let (proposal, commit) = f.admit(StructuralOp::Birth { result });
    f.pend(commit, proposal, born, Vec::new(), &["x"]);
    let retry = f.retry();
    assert!(retry.installed.is_empty());
    assert_eq!(retry.abandoned.len(), 1);
    assert_eq!(retry.abandoned[0].reason, AbandonReason::RequestLost);
    assert_eq!(retry.abandoned[0].retire_error, None);
    assert_eq!(
        f.vcw.stranded_experts().get(&pod("n")),
        Some(&BTreeSet::from([code("x")]))
    );
    // Idempotent: nothing left to retry.
    let again = f.retry();
    assert!(again.installed.is_empty() && again.abandoned.is_empty());
}

#[test]
fn at_the_stranded_cap_the_install_stays_tracked_and_nothing_is_dropped() {
    let mut f = Fixture::new(&[]);
    let born = f.train("n", &["x"]);
    let result = f.spec(&born, 150, 1);
    let (proposal, commit) = f.admit(StructuralOp::Birth { result });
    for i in 0..256 {
        f.vcw
            .strand_committed(&proposal, &[pod(&format!("s{i}"))], &BTreeSet::new())
            .unwrap();
    }
    assert_eq!(f.vcw.stranded_experts().len(), 256);
    f.pend(commit, proposal, born, Vec::new(), &["x"]);

    let retry = f.retry();
    assert!(retry.abandoned.is_empty(), "{retry:?}");
    assert_eq!(
        f.driver.pending_installs().len(),
        1,
        "the install is still tracked"
    );
    assert!(!f.vcw.stranded_experts().contains_key(&pod("n")));
    assert_eq!(f.vcw.stranded_experts().len(), 256);

    // The attempt is repeated on every retry, so a freed hold lets the abandonment through.
    let again = f.retry();
    assert!(again.abandoned.is_empty());
    assert_eq!(f.driver.pending_installs().len(), 1);
}

/// Issue #141: `install` takes over an existing PodRegistry entry only for a reactivation. A
/// Birth, Merge or Split result whose id is already registered with another manifest is
/// refused and the registered Pod stays; the same manifest is a retry and is left alone.
#[test]
fn install_replaces_an_existing_entry_only_for_a_reactivation() {
    let specialist = |seed: u64| {
        SimulatedTrainer::new(project(), DefectDoc::clean_type(), seed, 0.9)
            .train(&TrainingRequest {
                kind: OpKind::Birth,
                pod: pod("taken"),
                codes: [code("x")].into_iter().collect(),
                sources: Vec::new(),
            })
            .unwrap()
    };
    let incumbent = specialist(1);
    let newcomer = specialist(2);
    assert_ne!(
        incumbent.manifest.digest(),
        newcomer.manifest.digest(),
        "the two specialists must differ"
    );
    let registered = |pods: &PodRegistry| {
        pods.get(&project(), &pod("taken"))
            .unwrap()
            .manifest()
            .digest()
    };
    for kind in [OpKind::Birth, OpKind::Merge, OpKind::Split] {
        let mut pods = PodRegistry::default();
        pods.register_new(incumbent.executable()).unwrap();
        let refused = install(&mut pods, kind, &newcomer);
        assert!(
            matches!(refused, Err(PodRegistrationError::AlreadyRegistered { .. })),
            "{kind:?}: {refused:?}"
        );
        assert_eq!(registered(&pods), incumbent.manifest.digest(), "{kind:?}");
        // A retry of the install that already happened is not a conflict.
        install(&mut pods, kind, &incumbent).unwrap();
        assert_eq!(registered(&pods), incumbent.manifest.digest(), "{kind:?}");
        // A fresh id registers.
        let mut empty = PodRegistry::default();
        install(&mut empty, kind, &newcomer).unwrap();
        assert_eq!(registered(&empty), newcomer.manifest.digest(), "{kind:?}");
    }
    let mut pods = PodRegistry::default();
    pods.register_new(incumbent.executable()).unwrap();
    install(&mut pods, OpKind::Reactivate, &newcomer).unwrap();
    assert_eq!(registered(&pods), newcomer.manifest.digest());
}
