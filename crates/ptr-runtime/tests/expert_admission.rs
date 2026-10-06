//! Admission of expert structural changes: positive controls for each operation, every
//! compare-and-set and evidence refusal, replay of the admitted history and tampered histories.

use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, ExpertOpKind, FileLedger, LedgerEvent};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, OpKind, ProposerRef,
    RationaleDigest, RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    ledger_op_kind, ExpertAdmissionError, ExpertAdmissionGrant, ExpertEntry, ExpertEventField,
    ExpertStaleField,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};
use std::path::PathBuf;

const LAYER: u16 = 3;

fn project() -> ProjectId {
    ProjectId::from("proj")
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
    ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap()
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn other_gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.9,
        ..gate()
    }
}

fn passing() -> PublicSuite {
    PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    }
}

fn failing() -> PublicSuite {
    PublicSuite {
        serving: 0.9,
        candidate: 0.1,
    }
}

fn subject_of(proposal: &ExpertProposal) -> GateSubject {
    GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        [3; 32],
        [4; 32],
        [5; 32],
    )
}

fn report(
    proposal: &ExpertProposal,
    gate: &ForgettingGate,
    public: PublicSuite,
) -> BoundGateReport {
    bind(subject_of(proposal), gate, public)
}

fn bind(subject: GateSubject, gate: &ForgettingGate, public: PublicSuite) -> BoundGateReport {
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate.evaluate(&AdapterId::from("candidate"), &matrix, public)
        .unwrap()
        .bind(subject)
}

fn grant() -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(project(), proposer(), 7, gate().config_digest().unwrap())
}

/// A slot index that follows the pod name, so distinct pods get distinct slots.
fn slot_of(pod: &str) -> u16 {
    u16::from(pod.as_bytes()[0])
}

fn level_of(pod: &str) -> ExpertLevel {
    ExpertLevel::Token {
        layer: LAYER,
        index: slot_of(pod),
    }
}

fn spec_at(pod: &str, generation: u64, level: ExpertLevel) -> ExpertPodSpec {
    let mut digest = [1u8; 32];
    digest[0] = pod.as_bytes()[0];
    digest[1] = generation as u8;
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new(digest).unwrap(),
        level,
        base(),
        Generation(generation),
    )
    .unwrap()
}

fn spec(pod: &str) -> ExpertPodSpec {
    spec_at(pod, 1, level_of(pod))
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

fn admit(rt: &mut PtrRuntime, op: StructuralOp) -> Result<CommitIndex, RuntimeError> {
    let proposal = proposal(op);
    let report = report(&proposal, &gate(), passing());
    rt.admit_expert_proposal(&proposal, &report, &grant())
}

fn expert_error(error: ExpertAdmissionError) -> RuntimeError {
    RuntimeError::Expert(error)
}

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).unwrap()
}

fn entry(rt: &PtrRuntime, pod: &str) -> ExpertEntry {
    rt.expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap_or_else(|| panic!("no expert {pod}"))
        .clone()
}

fn init_of(rt: &PtrRuntime, pod: &str) -> ExpertTargetInit {
    let entry = entry(rt, pod);
    ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    }
}

type Tweak<'a> = &'a dyn Fn(&str, ExpertTargetInit) -> ExpertTargetInit;

fn exact(_: &str, init: ExpertTargetInit) -> ExpertTargetInit {
    init
}

/// The target of `pod` as the registry has it now, after `tweak`.
fn target(rt: &PtrRuntime, pod: &str, tweak: Tweak<'_>) -> ExpertTarget {
    ExpertTarget::new(tweak(pod, init_of(rt, pod))).unwrap()
}

fn birth(rt: &mut PtrRuntime, pod: &str) {
    admit(rt, StructuralOp::Birth { result: spec(pod) }).unwrap();
}

fn freeze(rt: &mut PtrRuntime, pod: &str) {
    let op = StructuralOp::Freeze {
        target: target(rt, pod, &exact),
    };
    admit(rt, op).unwrap();
}

fn retire(rt: &mut PtrRuntime, pod: &str, reason: RetiredReason) {
    let op = StructuralOp::Retire {
        target: target(rt, pod, &exact),
        reason,
    };
    admit(rt, op).unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Merge,
    Split,
    Freeze,
    ReactivateFrozen,
    ReactivateRetired,
    Prune,
    Retire,
}

const SCENARIOS: [Scenario; 7] = [
    Scenario::Merge,
    Scenario::Split,
    Scenario::Freeze,
    Scenario::ReactivateFrozen,
    Scenario::ReactivateRetired,
    Scenario::Prune,
    Scenario::Retire,
];

/// A runtime whose registry holds what `scenario` operates on.
fn seeded(scenario: Scenario) -> PtrRuntime {
    let mut rt = runtime();
    match scenario {
        Scenario::Merge | Scenario::Prune => {
            birth(&mut rt, "a");
            birth(&mut rt, "b");
        }
        Scenario::Split | Scenario::Freeze | Scenario::Retire => birth(&mut rt, "a"),
        Scenario::ReactivateFrozen => {
            birth(&mut rt, "a");
            freeze(&mut rt, "a");
        }
        Scenario::ReactivateRetired => {
            birth(&mut rt, "a");
            retire(&mut rt, "a", RetiredReason::GateFailed);
        }
    }
    rt
}

/// The operation of `scenario` over the registry, every operand passed through `tweak`.
fn operation(scenario: Scenario, rt: &PtrRuntime, tweak: Tweak<'_>) -> StructuralOp {
    match scenario {
        Scenario::Merge => StructuralOp::Merge {
            sources: vec![target(rt, "a", tweak), target(rt, "b", tweak)],
            result: spec("m"),
        },
        Scenario::Split => StructuralOp::Split {
            source: target(rt, "a", tweak),
            results: vec![spec("x"), spec("y")],
        },
        Scenario::Freeze => StructuralOp::Freeze {
            target: target(rt, "a", tweak),
        },
        Scenario::ReactivateFrozen => StructuralOp::Reactivate {
            target: target(rt, "a", tweak),
            result: None,
        },
        Scenario::ReactivateRetired => {
            let target = target(rt, "a", tweak);
            let next = spec_at("a", target.generation().0 + 1, target.level());
            StructuralOp::Reactivate {
                target,
                result: Some(next),
            }
        }
        Scenario::Prune => StructuralOp::Prune {
            targets: vec![target(rt, "a", tweak), target(rt, "b", tweak)],
        },
        Scenario::Retire => StructuralOp::Retire {
            target: target(rt, "a", tweak),
            reason: RetiredReason::Superseded,
        },
    }
}

/// Admit `scenario`'s operation as the registry currently describes its operands.
fn run(rt: &mut PtrRuntime, scenario: Scenario) -> Result<CommitIndex, RuntimeError> {
    let op = operation(scenario, rt, &exact);
    admit(rt, op)
}

fn kind_of(scenario: Scenario) -> OpKind {
    match scenario {
        Scenario::Merge => OpKind::Merge,
        Scenario::Split => OpKind::Split,
        Scenario::Freeze => OpKind::Freeze,
        Scenario::ReactivateFrozen | Scenario::ReactivateRetired => OpKind::Reactivate,
        Scenario::Prune => OpKind::Prune,
        Scenario::Retire => OpKind::Retire,
    }
}

// ---- positive controls -------------------------------------------------------------------

#[test]
fn birth_registers_a_serving_expert_and_commits_event_ids_derived_from_the_record() {
    let mut rt = runtime();
    let proposal = proposal(StructuralOp::Birth { result: spec("a") });
    let report = report(&proposal, &gate(), passing());
    let index = rt
        .admit_expert_proposal(&proposal, &report, &grant())
        .unwrap();
    assert_eq!(index, CommitIndex(1));
    let a = entry(&rt, "a");
    assert_eq!(a.state, ExpertState::Serving);
    assert_eq!(a.state_version, 1);
    assert_eq!(a.generation, Generation(1));
    assert_eq!(a.level, level_of("a"));
    let [committed] = rt.committed_events() else {
        panic!("one event");
    };
    let LedgerEvent::ExpertStructuralCommitted {
        project: event_project,
        op_kind,
        proposal_id,
        operation_id,
        evidence_digest,
        record,
    } = &committed.event
    else {
        panic!("structural event");
    };
    assert_eq!(event_project, &project());
    assert_eq!(*op_kind, ExpertOpKind::Birth);
    assert_eq!(proposal_id, proposal.proposal_id().as_bytes());
    assert_eq!(operation_id, proposal.operation_id().as_bytes());
    assert_eq!(*evidence_digest, report.digest());
    let decoded = ptr_pods::ExpertCommitRecord::decode_canonical(record).unwrap();
    assert_eq!(decoded.proposal(), &proposal);
    assert_eq!(decoded.gate_epoch(), 7);
    assert_eq!(decoded.evidence_digest(), &report.digest());
}

#[test]
fn merge_supersedes_every_source_and_registers_one_serving_result() {
    let mut rt = seeded(Scenario::Merge);
    run(&mut rt, Scenario::Merge).unwrap();
    for pod in ["a", "b"] {
        let e = entry(&rt, pod);
        assert_eq!((e.state, e.state_version), (ExpertState::Superseded, 2));
    }
    let m = entry(&rt, "m");
    assert_eq!((m.state, m.state_version), (ExpertState::Serving, 1));
}

#[test]
fn split_supersedes_the_source_and_registers_every_result() {
    let mut rt = seeded(Scenario::Split);
    run(&mut rt, Scenario::Split).unwrap();
    assert_eq!(entry(&rt, "a").state, ExpertState::Superseded);
    for pod in ["x", "y"] {
        assert_eq!(entry(&rt, pod).state, ExpertState::Serving);
    }
}

#[test]
fn freeze_then_unfreeze_keeps_the_generation_and_counts_each_transition() {
    let mut rt = seeded(Scenario::Freeze);
    run(&mut rt, Scenario::Freeze).unwrap();
    let frozen = entry(&rt, "a");
    assert_eq!(
        (frozen.state, frozen.state_version),
        (ExpertState::Frozen, 2)
    );
    run(&mut rt, Scenario::ReactivateFrozen).unwrap();
    let serving = entry(&rt, "a");
    assert_eq!(serving.state, ExpertState::Serving);
    assert_eq!(serving.state_version, 3);
    assert_eq!(serving.generation, Generation(1));
    assert_eq!(serving.manifest_digest, frozen.manifest_digest);
}

#[test]
fn reactivating_a_gate_failed_expert_takes_over_its_pod_and_slot_as_the_next_generation() {
    let mut rt = seeded(Scenario::ReactivateRetired);
    assert_eq!(
        entry(&rt, "a").state,
        ExpertState::Retired(RetiredReason::GateFailed)
    );
    run(&mut rt, Scenario::ReactivateRetired).unwrap();
    let a = entry(&rt, "a");
    assert_eq!(a.generation, Generation(2));
    assert_eq!((a.state, a.state_version), (ExpertState::Serving, 1));
    assert_eq!(a.level, level_of("a"));
}

#[test]
fn prune_retires_every_target_as_pruned() {
    let mut rt = seeded(Scenario::Prune);
    run(&mut rt, Scenario::Prune).unwrap();
    for pod in ["a", "b"] {
        let e = entry(&rt, pod);
        assert_eq!(e.state, ExpertState::Retired(RetiredReason::Pruned));
        assert_eq!(e.state_version, 2);
    }
}

#[test]
fn retire_records_the_reason() {
    let mut rt = seeded(Scenario::Retire);
    run(&mut rt, Scenario::Retire).unwrap();
    assert_eq!(
        entry(&rt, "a").state,
        ExpertState::Retired(RetiredReason::Superseded)
    );
}

#[test]
fn a_merge_result_may_take_over_a_source_slot_but_not_an_unrelated_one() {
    let mut rt = seeded(Scenario::Merge);
    birth(&mut rt, "c");
    let sources = |rt: &PtrRuntime| vec![target(rt, "a", &exact), target(rt, "b", &exact)];
    let onto_c = StructuralOp::Merge {
        sources: sources(&rt),
        result: spec_at("m", 1, level_of("c")),
    };
    assert_eq!(
        admit(&mut rt, onto_c),
        Err(expert_error(ExpertAdmissionError::ResultSlotOccupied {
            layer: LAYER,
            index: slot_of("c"),
        }))
    );
    let onto_a = StructuralOp::Merge {
        sources: sources(&rt),
        result: spec_at("m", 1, level_of("a")),
    };
    admit(&mut rt, onto_a).unwrap();
    assert_eq!(entry(&rt, "m").level, level_of("a"));
}

// ---- compare-and-set ---------------------------------------------------------------------

type StaleCase = (
    &'static str,
    ExpertStaleField,
    fn(ExpertTargetInit) -> ExpertTargetInit,
);

#[test]
fn a_stale_operand_is_refused_per_operation_and_field_and_changes_nothing() {
    let tweaks: [StaleCase; 3] = [
        ("generation", ExpertStaleField::Generation, |mut i| {
            i.generation = Generation(i.generation.0 + 1);
            i
        }),
        ("state_version", ExpertStaleField::StateVersion, |mut i| {
            i.state_version += 1;
            i
        }),
        (
            "manifest digest",
            ExpertStaleField::ManifestDigest,
            |mut i| {
                i.manifest_digest = ManifestDigest::new([0xEE; 32]).unwrap();
                i
            },
        ),
    ];
    for scenario in SCENARIOS {
        for (name, field, change) in tweaks {
            // The first operand, and for the two-operand operations the second one alone.
            let stale_pods: &[&str] = match scenario {
                Scenario::Merge | Scenario::Prune => &["a", "b"],
                _ => &["a"],
            };
            for stale in stale_pods {
                let mut rt = seeded(scenario);
                let registry = rt.expert_registry().clone();
                let events = rt.committed_events().len();
                let tweak = |pod: &str, init: ExpertTargetInit| {
                    if pod == *stale {
                        change(init)
                    } else {
                        init
                    }
                };
                let op = operation(scenario, &rt, &tweak);
                assert_eq!(
                    admit(&mut rt, op),
                    Err(expert_error(ExpertAdmissionError::StaleOperand {
                        pod: (*stale).to_owned(),
                        field,
                    })),
                    "{scenario:?} stale {name} on {stale}"
                );
                assert_eq!(rt.expert_registry(), &registry);
                assert_eq!(rt.committed_events().len(), events);
            }
        }
    }
}

#[test]
fn an_observed_state_that_the_registry_does_not_hold_is_refused() {
    for scenario in [Scenario::Split, Scenario::Retire] {
        let mut rt = seeded(scenario);
        let op = operation(scenario, &rt, &|_, mut init| {
            init.state = ExpertState::Frozen;
            init
        });
        assert_eq!(
            admit(&mut rt, op),
            Err(expert_error(ExpertAdmissionError::StaleOperand {
                pod: "a".to_owned(),
                field: ExpertStaleField::State,
            })),
            "{scenario:?}"
        );
    }
}

#[test]
fn an_operand_that_was_never_admitted_is_unknown() {
    let mut rt = runtime();
    let ghost = ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from("ghost"),
        generation: Generation(1),
        level: level_of("g"),
        base_model: base(),
        manifest_digest: ManifestDigest::new([1; 32]).unwrap(),
        state: ExpertState::Serving,
        state_version: 1,
    })
    .unwrap();
    assert_eq!(
        admit(&mut rt, StructuralOp::Freeze { target: ghost }),
        Err(expert_error(ExpertAdmissionError::UnknownOperand {
            pod: "ghost".to_owned()
        }))
    );
}

#[test]
fn a_result_pod_or_slot_that_is_taken_is_refused() {
    let mut rt = runtime();
    birth(&mut rt, "a");
    assert_eq!(
        admit(&mut rt, StructuralOp::Birth { result: spec("a") }),
        Err(expert_error(ExpertAdmissionError::ResultPodOccupied {
            pod: "a".to_owned()
        }))
    );
    let same_slot = spec_at("a2", 1, level_of("a"));
    assert_eq!(
        admit(&mut rt, StructuralOp::Birth { result: same_slot }),
        Err(expert_error(ExpertAdmissionError::ResultSlotOccupied {
            layer: LAYER,
            index: slot_of("a"),
        }))
    );
    assert_eq!(rt.committed_events().len(), 1);
}

#[test]
fn a_released_slot_can_be_reused_by_a_new_pod() {
    let mut rt = runtime();
    birth(&mut rt, "a");
    retire(&mut rt, "a", RetiredReason::Superseded);
    admit(
        &mut rt,
        StructuralOp::Birth {
            result: spec_at("a2", 1, level_of("a")),
        },
    )
    .unwrap();
    assert_eq!(entry(&rt, "a2").state, ExpertState::Serving);
}

// ---- evidence and authority --------------------------------------------------------------

/// A runtime with one serving expert `a` and a freeze of it to propose.
fn freezable() -> (PtrRuntime, ExpertProposal) {
    let rt = seeded(Scenario::Freeze);
    let proposal = proposal(operation(Scenario::Freeze, &rt, &exact));
    (rt, proposal)
}

fn assert_refused(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
    report: &BoundGateReport,
    grant: &ExpertAdmissionGrant,
    error: ExpertAdmissionError,
) {
    let registry = rt.expert_registry().clone();
    let events = rt.committed_events().len();
    assert_eq!(
        rt.admit_expert_proposal(proposal, report, grant),
        Err(expert_error(error))
    );
    assert_eq!(rt.expert_registry(), &registry);
    assert_eq!(rt.committed_events().len(), events);
}

#[test]
fn a_report_bound_to_another_proposal_is_refused() {
    let (mut rt, freeze) = freezable();
    let retire = proposal(StructuralOp::Retire {
        target: target(&rt, "a", &exact),
        reason: RetiredReason::Superseded,
    });
    let for_retire = report(&retire, &gate(), passing());
    assert_refused(
        &mut rt,
        &freeze,
        &for_retire,
        &grant(),
        ExpertAdmissionError::EvidenceProposalMismatch,
    );
}

#[test]
fn a_report_bound_to_another_operation_is_refused() {
    let (mut rt, freeze) = freezable();
    let subject = GateSubject {
        operation_id: [9; 32],
        ..subject_of(&freeze)
    };
    assert_refused(
        &mut rt,
        &freeze,
        &bind(subject, &gate(), passing()),
        &grant(),
        ExpertAdmissionError::EvidenceOperationMismatch,
    );
}

#[test]
fn a_report_from_another_gate_configuration_is_refused() {
    let (mut rt, freeze) = freezable();
    assert_refused(
        &mut rt,
        &freeze,
        &report(&freeze, &other_gate(), passing()),
        &grant(),
        ExpertAdmissionError::GateConfigMismatch,
    );
}

#[test]
fn a_failing_report_is_refused() {
    let (mut rt, freeze) = freezable();
    let failed = report(&freeze, &gate(), failing());
    assert!(!failed.passed());
    assert_refused(
        &mut rt,
        &freeze,
        &failed,
        &grant(),
        ExpertAdmissionError::GateFailed,
    );
}

#[test]
fn a_proposer_or_project_other_than_the_authenticated_one_is_refused_before_the_evidence() {
    let (mut rt, freeze) = freezable();
    let good = report(&freeze, &gate(), passing());
    let digest = gate().config_digest().unwrap();
    for other in [
        ProposerRef::new(project(), PodId::from("meta2"), Generation(1)).unwrap(),
        ProposerRef::new(project(), PodId::from("meta"), Generation(2)).unwrap(),
    ] {
        let grant = ExpertAdmissionGrant::new(project(), other, 7, digest);
        assert_refused(
            &mut rt,
            &freeze,
            &good,
            &grant,
            ExpertAdmissionError::ProposerMismatch,
        );
    }
    let elsewhere = ProjectId::from("elsewhere");
    let foreign = ExpertAdmissionGrant::new(
        elsewhere.clone(),
        ProposerRef::new(elsewhere, PodId::from("meta"), Generation(1)).unwrap(),
        7,
        digest,
    );
    // Even with a failing report the identity is the first thing refused.
    let failed = report(&freeze, &gate(), failing());
    assert_refused(
        &mut rt,
        &freeze,
        &failed,
        &foreign,
        ExpertAdmissionError::ProjectMismatch,
    );
}

#[test]
fn the_public_commit_path_still_refuses_a_structural_record() {
    let mut admitted = runtime();
    birth(&mut admitted, "a");
    let event = admitted.committed_events()[0].event.clone();
    let mut fresh = runtime();
    assert_eq!(
        fresh.commit(event),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    );
    assert!(fresh.committed_events().is_empty());
    assert!(fresh.expert_registry().is_empty());
}

// ---- replay ------------------------------------------------------------------------------

/// Births, merge, split, freeze, unfreeze, retirement, reactivation and prune in one history.
fn full_history() -> PtrRuntime {
    let mut rt = runtime();
    for pod in ["a", "b", "c", "d"] {
        birth(&mut rt, pod);
    }
    let merge = StructuralOp::Merge {
        sources: vec![target(&rt, "a", &exact), target(&rt, "b", &exact)],
        result: spec("m"),
    };
    admit(&mut rt, merge).unwrap();
    let split = StructuralOp::Split {
        source: target(&rt, "c", &exact),
        results: vec![spec("x"), spec("y")],
    };
    admit(&mut rt, split).unwrap();
    freeze(&mut rt, "m");
    let unfreeze = StructuralOp::Reactivate {
        target: target(&rt, "m", &exact),
        result: None,
    };
    admit(&mut rt, unfreeze).unwrap();
    retire(&mut rt, "x", RetiredReason::GateFailed);
    let revive = StructuralOp::Reactivate {
        target: target(&rt, "x", &exact),
        result: Some(spec_at("x", 2, level_of("x"))),
    };
    admit(&mut rt, revive).unwrap();
    let prune = StructuralOp::Prune {
        targets: vec![target(&rt, "y", &exact), target(&rt, "d", &exact)],
    };
    admit(&mut rt, prune).unwrap();
    rt
}

fn history(rt: &PtrRuntime) -> Vec<CommittedEvent> {
    rt.committed_events().to_vec()
}

/// `events` renumbered from 1, as a log read back after dropping or moving records would be.
fn renumbered(events: Vec<CommittedEvent>) -> Vec<CommittedEvent> {
    events
        .into_iter()
        .enumerate()
        .map(|(i, committed)| CommittedEvent {
            index: CommitIndex(i as u64 + 1),
            event: committed.event,
        })
        .collect()
}

fn replay_error(events: &[CommittedEvent]) -> RuntimeError {
    match PtrRuntime::replay(PtrConfig::default(), events) {
        Ok(_) => panic!("replay must fail"),
        Err(error) => error,
    }
}

#[test]
fn replaying_the_admitted_history_reproduces_the_registry_exactly() {
    let rt = full_history();
    let events = history(&rt);
    assert_eq!(events.len(), 11);
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_eq!(replayed.committed_events(), rt.committed_events());
    let x = entry(&replayed, "x");
    assert_eq!(
        (x.generation, x.state),
        (Generation(2), ExpertState::Serving)
    );
    assert_eq!(
        entry(&replayed, "y").state,
        ExpertState::Retired(RetiredReason::Pruned)
    );
}

fn temp_log() -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "ptr-expert-admission-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("ledger");
    (dir, path)
}

#[test]
fn a_durable_runtime_reopens_with_the_same_registry() {
    let (dir, path) = temp_log();
    let mut rt = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    birth(&mut rt, "a");
    birth(&mut rt, "b");
    freeze(&mut rt, "a");
    let registry = rt.expert_registry().clone();
    drop(rt);
    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(reopened.expert_registry(), &registry);
    assert_eq!(entry(&reopened, "a").state, ExpertState::Frozen);
    drop(reopened);
    // A log whose record disagrees with its event does not open and is left unchanged.
    let first = history_event(&path, 0);
    let mut log = FileLedger::open(&path).unwrap();
    let LedgerEvent::ExpertStructuralCommitted {
        project,
        op_kind,
        proposal_id,
        operation_id,
        record,
        ..
    } = first
    else {
        panic!("structural");
    };
    log.append_durable(LedgerEvent::ExpertStructuralCommitted {
        project,
        op_kind,
        proposal_id,
        operation_id,
        evidence_digest: [8; 32],
        record,
    })
    .unwrap();
    drop(log);
    let before = std::fs::read(&path).unwrap();
    assert!(matches!(
        PtrRuntime::open_durable(PtrConfig::default(), &path),
        Err(RuntimeError::Expert(_))
    ));
    assert_eq!(before, std::fs::read(&path).unwrap());
    std::fs::remove_dir_all(&dir).unwrap();
}

fn history_event(path: &std::path::Path, at: usize) -> LedgerEvent {
    FileLedger::open(path).unwrap().events()[at].event.clone()
}

/// The fields of a structural event, in declaration order.
type Fields = (
    ProjectId,
    ExpertOpKind,
    [u8; 32],
    [u8; 32],
    [u8; 32],
    Vec<u8>,
);

fn structural_fields(event: &LedgerEvent) -> Fields {
    let LedgerEvent::ExpertStructuralCommitted {
        project,
        op_kind,
        proposal_id,
        operation_id,
        evidence_digest,
        record,
    } = event
    else {
        panic!("structural event");
    };
    (
        project.clone(),
        *op_kind,
        *proposal_id,
        *operation_id,
        *evidence_digest,
        record.clone(),
    )
}

fn rebuilt(
    event: &LedgerEvent,
    change: impl FnOnce(
        &mut (
            ProjectId,
            ExpertOpKind,
            [u8; 32],
            [u8; 32],
            [u8; 32],
            Vec<u8>,
        ),
    ),
) -> LedgerEvent {
    let mut fields = structural_fields(event);
    change(&mut fields);
    let (project, op_kind, proposal_id, operation_id, evidence_digest, record) = fields;
    LedgerEvent::ExpertStructuralCommitted {
        project,
        op_kind,
        proposal_id,
        operation_id,
        evidence_digest,
        record,
    }
}

type EventCase = (ExpertEventField, fn(&mut Fields));

#[test]
fn replay_refuses_every_event_field_that_disagrees_with_its_record() {
    let rt = full_history();
    let events = history(&rt);
    let cases: [EventCase; 5] = [
        (ExpertEventField::Project, |f| {
            f.0 = ProjectId::from("proj2")
        }),
        (ExpertEventField::OpKind, |f| f.1 = ExpertOpKind::Merge),
        (ExpertEventField::ProposalId, |f| f.2 = [9; 32]),
        (ExpertEventField::OperationId, |f| f.3 = [9; 32]),
        (ExpertEventField::EvidenceDigest, |f| f.4 = [9; 32]),
    ];
    for (field, change) in cases {
        let mut tampered = events.clone();
        tampered[0].event = rebuilt(&events[0].event, change);
        assert_eq!(
            replay_error(&tampered),
            expert_error(ExpertAdmissionError::EventMismatch { field }),
            "{field:?}"
        );
    }
}

#[test]
fn replay_refuses_a_record_whose_evidence_digest_was_changed_in_the_record_itself() {
    let rt = full_history();
    let mut tampered = history(&rt);
    // The first occurrence of the digest in the record is its own evidence digest field, ahead
    // of the embedded report (whose last 32 bytes repeat it).
    tampered[0].event = rebuilt(&tampered[0].event, |f| {
        let at = f.5.windows(32).position(|w| w == f.4).unwrap();
        f.5[at] ^= 0xFF;
    });
    assert_eq!(
        replay_error(&tampered),
        expert_error(ExpertAdmissionError::EventMismatch {
            field: ExpertEventField::EvidenceDigest
        })
    );
}

#[test]
fn replay_refuses_a_record_that_does_not_decode() {
    let rt = full_history();
    let mut tampered = history(&rt);
    tampered[0].event = rebuilt(&tampered[0].event, |f| f.5.truncate(f.5.len() - 1));
    assert!(matches!(
        replay_error(&tampered),
        RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(_))
    ));
    tampered[0].event = rebuilt(&history(&rt)[0].event, |f| {
        f.5 = b"PTREXR01-opaque".to_vec()
    });
    assert!(matches!(
        replay_error(&tampered),
        RuntimeError::Expert(ExpertAdmissionError::InvalidRecord(_))
    ));
}

#[test]
fn replay_refuses_a_reordered_dropped_or_repeated_history() {
    let rt = full_history();
    let events = history(&rt);
    // The merge (index 5) moved in front of the births it consumes.
    let mut reordered = events.clone();
    let merge = reordered.remove(4);
    reordered.insert(0, merge);
    assert_eq!(
        replay_error(&renumbered(reordered)),
        expert_error(ExpertAdmissionError::UnknownOperand {
            pod: "a".to_owned()
        })
    );
    // The first birth dropped.
    let mut dropped = events.clone();
    dropped.remove(0);
    assert!(matches!(
        replay_error(&renumbered(dropped)),
        RuntimeError::Expert(ExpertAdmissionError::UnknownOperand { .. })
    ));
    // A birth repeated.
    let mut repeated = events.clone();
    repeated.insert(1, events[0].clone());
    assert_eq!(
        replay_error(&renumbered(repeated)),
        expert_error(ExpertAdmissionError::ResultPodOccupied {
            pod: "a".to_owned()
        })
    );
    // A freeze applied twice: the second observed a state version that has moved on.
    let freeze_at = 6;
    assert!(matches!(
        structural_fields(&events[freeze_at].event).1,
        ExpertOpKind::Freeze
    ));
    let mut twice = events.clone();
    twice.insert(freeze_at + 1, events[freeze_at].clone());
    assert_eq!(
        replay_error(&renumbered(twice)),
        expert_error(ExpertAdmissionError::StaleOperand {
            pod: "m".to_owned(),
            field: ExpertStaleField::State,
        })
    );
}

// ---- projection and codes ----------------------------------------------------------------

#[test]
fn the_state_projection_agrees_with_the_registry() {
    let rt = full_history();
    let mut state = MaterializedState::default();
    for committed in rt.committed_events() {
        state.apply(committed);
    }
    let mut experts = 0;
    for (project, pod, entry) in rt.expert_registry().iter() {
        let key = format!("expert:{project}/{pod}");
        let expected = format!(
            "{}:{}:{}",
            entry.generation.0,
            ptr_state::expert_state_name(entry.state),
            entry.state_version
        );
        assert_eq!(state.values.get(&key), Some(&expected), "{key}");
        experts += 1;
    }
    assert_eq!(experts, 7);
    assert_eq!(
        state.values.get("expert:proj/y").map(String::as_str),
        Some("1:retired-pruned:2")
    );
    assert_eq!(
        state.values.get("expert:proj/x").map(String::as_str),
        Some("2:serving:1")
    );
}

#[test]
fn ledger_codes_agree_with_the_proposal_kinds() {
    let pairs = [
        (OpKind::Birth, ExpertOpKind::Birth, 0),
        (OpKind::Merge, ExpertOpKind::Merge, 1),
        (OpKind::Split, ExpertOpKind::Split, 2),
        (OpKind::Freeze, ExpertOpKind::Freeze, 3),
        (OpKind::Reactivate, ExpertOpKind::Reactivate, 4),
        (OpKind::Prune, ExpertOpKind::Prune, 5),
        (OpKind::Retire, ExpertOpKind::Retire, 6),
    ];
    for (kind, ledger, code) in pairs {
        assert_eq!(ledger_op_kind(kind), ledger);
        assert_eq!(ledger.code(), code);
        assert_eq!(ExpertOpKind::from_code(code), Some(ledger));
    }
    // Every admitted event carries the code of the operation it records.
    let rt = full_history();
    let kinds: Vec<ExpertOpKind> = rt
        .committed_events()
        .iter()
        .map(|committed| structural_fields(&committed.event).1)
        .collect();
    assert_eq!(
        kinds.iter().filter(|k| **k == ExpertOpKind::Birth).count(),
        4
    );
    for scenario in SCENARIOS {
        let mut rt = seeded(scenario);
        let before = rt.committed_events().len();
        run(&mut rt, scenario).unwrap();
        let committed = &rt.committed_events()[before];
        assert_eq!(
            structural_fields(&committed.event).1,
            ledger_op_kind(kind_of(scenario))
        );
    }
}
