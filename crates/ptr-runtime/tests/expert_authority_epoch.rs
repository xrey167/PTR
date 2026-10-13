//! The evidence-authority epoch (R011 C12): admission evidence must come from the current gate
//! configuration epoch and the current evaluator, a revocation or tightening rotation fences
//! exactly the experts admitted under the old epoch (a non-tightening rotation fences nothing),
//! readmission needs evidence under the current epoch, and the epochs and the fences survive
//! replay, `open_durable` and compaction.

use ptr_config::PtrConfig;
use ptr_ledger::{
    CommittedEvent, ExpertAuthorityClass, ExpertAuthorityKind, ExpertFenceCause, LedgerEvent,
};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, InMemoryPagedKvBackend, KvPageBinding, KvPageSize,
    KvTensorBackend, KvTensorDType, KvTensorSchema, ManifestDigest, PagedKvTensorBackend,
    ProposerRef, RationaleDigest, RetiredReason, StructuralOp, TensorRef,
};
use ptr_runtime::compacted::CompactedError;
use ptr_runtime::expert::{
    expert_authority_fence_set_digest, expert_evidence_artifact_digest,
    expert_evidence_base_model_digest, expert_readmission_subject_ids, ExpertAdmissionError,
    ExpertAdmissionGrant, ExpertAuthorityChange, ExpertAuthorityOutcome, ExpertFenceAuthority,
    ExpertFenceGrant, ExpertFenceRequest, ExpertFenceSetMember, RouteRefusal, StaleEvidenceReason,
};
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvChainId, KvDependency, KvInsert, KvLivenessOracle, KvLookup,
    KvPrefixCatalog, KvRequestPositions, KvSalt, PtrRuntime, ReferenceKvError, ReferenceKvLimits,
    ReferenceKvReuseBackend, ReferenceKvScope, RuntimeError,
};
use ptr_types::{
    AdapterVersion, CommitIndex, DeviceId, Generation, ModelVersion, PodId, PrincipalId, ProjectId,
    RequestId, Revision, SemanticContext,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

mod expert_support;

use expert_support::{binding, prime, requirement, BASE_MODEL_DIGEST, EVAL_MANIFEST};

const GATE: ExpertAuthorityKind = ExpertAuthorityKind::GateConfig;
const EVALUATOR: ExpertAuthorityKind = ExpertAuthorityKind::Evaluator;
const REVOKE: ExpertAuthorityClass = ExpertAuthorityClass::Revoke;
const TIGHTEN: ExpertAuthorityClass = ExpertAuthorityClass::TighteningRotation;
const LOOSE: ExpertAuthorityClass = ExpertAuthorityClass::NonTightening;

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

/// The gate configuration of version `version`: each version has its own configuration digest.
fn gate_v(version: u8) -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5 - 0.02 * f64::from(version),
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn cfg(version: u8) -> [u8; 32] {
    gate_v(version).config_digest().unwrap()
}

fn evaluator(version: u8) -> [u8; 32] {
    [0xE0 + version; 32]
}

/// What a grant claims about the authority its evidence comes from.
#[derive(Clone, Copy)]
struct Setup {
    epoch: u64,
    version: u8,
    evaluator: Option<u8>,
}

/// The setup of the current standing used throughout: gate epoch `epoch` with configuration
/// version `epoch`, evaluator version `epoch`.
fn current(epoch: u64) -> Setup {
    Setup {
        epoch,
        version: epoch as u8,
        evaluator: Some(epoch as u8),
    }
}

/// Before any standing exists nothing is enforced; this is what a pre-regime host sends.
const PRE: Setup = Setup {
    epoch: 1,
    version: 1,
    evaluator: None,
};

fn level_of(pod: &str) -> ExpertLevel {
    ExpertLevel::Token {
        layer: 3,
        index: u16::from(pod.as_bytes()[0]),
    }
}

fn spec_at(pod: &str, generation: u64) -> ExpertPodSpec {
    let mut digest = [pod.as_bytes()[0]; 32];
    digest[1] = generation as u8;
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new(digest).unwrap(),
        level_of(pod),
        base(),
        Generation(generation),
    )
    .unwrap()
}

fn spec(pod: &str) -> ExpertPodSpec {
    spec_at(pod, 1)
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

fn matrix() -> AccuracyMatrix {
    AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap()
}

fn public() -> PublicSuite {
    PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    }
}

fn grant_for(setup: Setup) -> ExpertAdmissionGrant {
    let grant = ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(setup.epoch, cfg(setup.version)),
        binding(),
    );
    match setup.evaluator {
        Some(version) => grant.with_evaluator(evaluator(version)),
        None => grant,
    }
}

fn admit(rt: &mut PtrRuntime, op: StructuralOp, setup: Setup) -> Result<CommitIndex, RuntimeError> {
    let proposal = proposal(op);
    let report = gate_v(setup.version)
        .evaluate(&AdapterId::from("candidate"), &matrix(), public())
        .unwrap()
        .bind(GateSubject::new(
            *proposal.proposal_id().as_bytes(),
            *proposal.operation_id().as_bytes(),
            expert_evidence_artifact_digest(proposal.op()),
            expert_evidence_base_model_digest(proposal.op()).unwrap(),
            EVAL_MANIFEST,
        ));
    rt.admit_expert_proposal(&proposal, &report, &grant_for(setup))
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

fn birth(rt: &mut PtrRuntime, pod: &str, setup: Setup) {
    admit(rt, StructuralOp::Birth { result: spec(pod) }, setup).unwrap();
}

fn init_of(rt: &PtrRuntime, pod: &str) -> ExpertTargetInit {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap_or_else(|| panic!("no expert {pod}"))
        .clone();
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

fn target(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
    ExpertTarget::new(init_of(rt, pod)).unwrap()
}

fn freeze(rt: &mut PtrRuntime, pod: &str, setup: Setup) {
    let op = StructuralOp::Freeze {
        target: target(rt, pod),
    };
    admit(rt, op, setup).unwrap();
}

fn expert_error<T: std::fmt::Debug>(result: Result<T, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {other:?}"),
    }
}

fn revocation_grant() -> ExpertFenceGrant {
    ExpertFenceGrant::revocation(project(), binding())
}

/// The runtime-wide revocation authority an authority change needs.
fn global_grant() -> ExpertFenceGrant {
    ExpertFenceGrant::global_revocation(binding())
}

fn change(
    rt: &mut PtrRuntime,
    kind: ExpertAuthorityKind,
    epoch: u64,
    class: ExpertAuthorityClass,
) -> Result<ExpertAuthorityOutcome, RuntimeError> {
    let id = match kind {
        GATE => cfg(epoch as u8),
        EVALUATOR => evaluator(epoch as u8),
    };
    rt.change_expert_authority(
        &global_grant(),
        ExpertAuthorityChange::new(kind, id, epoch, class),
    )
}

fn state_of(rt: &PtrRuntime, pod: &str) -> (ExpertState, u64) {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap_or_else(|| panic!("no expert {pod}"));
    (entry.state, entry.state_version)
}

fn route(rt: &PtrRuntime, pod: &str) -> Result<(), RouteRefusal> {
    rt.expert_route_admission(&project(), &PodId::from(pod), Generation(1))
}

fn applied(rt: &PtrRuntime) -> usize {
    rt.committed_events().len()
}

fn history(rt: &PtrRuntime) -> Vec<CommittedEvent> {
    rt.committed_events().to_vec()
}

// ---- stale evidence for every gate-driven operation -----------------------------------------

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Birth,
    Merge,
    Split,
    Freeze,
    ReactivateFrozen,
    ReactivateRetired,
    Prune,
    Retire,
}

const SCENARIOS: [Scenario; 8] = [
    Scenario::Birth,
    Scenario::Merge,
    Scenario::Split,
    Scenario::Freeze,
    Scenario::ReactivateFrozen,
    Scenario::ReactivateRetired,
    Scenario::Prune,
    Scenario::Retire,
];

/// A pre-regime runtime whose registry holds what `scenario` operates on.
fn seeded(scenario: Scenario) -> PtrRuntime {
    let mut rt = runtime();
    match scenario {
        Scenario::Birth => {}
        Scenario::Merge | Scenario::Prune => {
            birth(&mut rt, "a", PRE);
            birth(&mut rt, "b", PRE);
        }
        Scenario::Split | Scenario::Freeze | Scenario::Retire => birth(&mut rt, "a", PRE),
        Scenario::ReactivateFrozen => {
            birth(&mut rt, "a", PRE);
            freeze(&mut rt, "a", PRE);
        }
        Scenario::ReactivateRetired => {
            birth(&mut rt, "a", PRE);
            let op = StructuralOp::Retire {
                target: target(&rt, "a"),
                reason: RetiredReason::GateFailed,
            };
            admit(&mut rt, op, PRE).unwrap();
        }
    }
    rt
}

fn operation(scenario: Scenario, rt: &PtrRuntime) -> StructuralOp {
    match scenario {
        Scenario::Birth => StructuralOp::Birth { result: spec("n") },
        Scenario::Merge => StructuralOp::Merge {
            sources: vec![target(rt, "a"), target(rt, "b")],
            result: spec("m"),
        },
        Scenario::Split => StructuralOp::Split {
            source: target(rt, "a"),
            results: vec![spec("x"), spec("y")],
        },
        Scenario::Freeze => StructuralOp::Freeze {
            target: target(rt, "a"),
        },
        Scenario::ReactivateFrozen => StructuralOp::Reactivate {
            target: target(rt, "a"),
            result: None,
        },
        Scenario::ReactivateRetired => StructuralOp::Reactivate {
            target: target(rt, "a"),
            result: Some(spec_at("a", 2)),
        },
        Scenario::Prune => StructuralOp::Prune {
            targets: vec![target(rt, "a"), target(rt, "b")],
        },
        Scenario::Retire => StructuralOp::Retire {
            target: target(rt, "a"),
            reason: RetiredReason::Superseded,
        },
    }
}

#[test]
fn stale_epoch_or_evaluator_evidence_is_refused_for_every_operation_and_a_current_grant_passes() {
    for scenario in SCENARIOS {
        let mut rt = seeded(scenario);
        // The regime starts: both kinds rotate without tightening, which fences nothing.
        let gate = change(&mut rt, GATE, 2, LOOSE).unwrap();
        let eval = change(&mut rt, EVALUATOR, 2, LOOSE).unwrap();
        assert_eq!(
            (gate.fence_count(), eval.fence_count()),
            (0, 0),
            "{scenario:?}"
        );

        let stale = |setup: Setup, kind, reason| {
            (
                setup,
                ExpertAdmissionError::StalePolicyEvidence { kind, reason },
            )
        };
        let cases = [
            // The gate epoch is the old one (the configuration is the current one).
            stale(
                Setup {
                    epoch: 1,
                    ..current(2)
                },
                GATE,
                StaleEvidenceReason::EpochMismatch {
                    current: 2,
                    evidence: 1,
                },
            ),
            // The epoch is current but the gate configuration is the old one.
            stale(
                Setup {
                    version: 1,
                    ..current(2)
                },
                GATE,
                StaleEvidenceReason::IdentityMismatch,
            ),
            // The evaluator is the old one.
            stale(
                Setup {
                    evaluator: Some(1),
                    ..current(2)
                },
                EVALUATOR,
                StaleEvidenceReason::IdentityMismatch,
            ),
            // The grant names no evaluator at all.
            stale(
                Setup {
                    evaluator: None,
                    ..current(2)
                },
                EVALUATOR,
                StaleEvidenceReason::IdentityNotNamed,
            ),
        ];
        for (setup, expected) in cases {
            let (before, registry) = (applied(&rt), rt.expert_registry().clone());
            let op = operation(scenario, &rt);
            assert_eq!(
                expert_error(admit(&mut rt, op, setup)),
                expected,
                "{scenario:?}"
            );
            assert_eq!(applied(&rt), before, "{scenario:?}: nothing was appended");
            assert_eq!(rt.expert_registry(), &registry, "{scenario:?}");
        }
        // The positive control: the same operation under the current epoch and evaluator.
        let op = operation(scenario, &rt);
        admit(&mut rt, op, current(2)).unwrap_or_else(|e| panic!("{scenario:?}: {e:?}"));
    }
}

#[test]
fn a_history_without_a_standing_enforces_nothing() {
    let mut rt = runtime();
    // Any gate epoch, configuration and evaluator are accepted while no event moved a kind.
    birth(
        &mut rt,
        "a",
        Setup {
            epoch: 7,
            version: 3,
            evaluator: Some(9),
        },
    );
    birth(&mut rt, "b", PRE);
    assert!(rt.expert_authority(GATE).is_none());
    assert!(rt.expert_authority(EVALUATOR).is_none());
    // One kind's standing does not enforce the other.
    change(&mut rt, GATE, 2, LOOSE).unwrap();
    assert!(rt.expert_authority(EVALUATOR).is_none());
    birth(
        &mut rt,
        "c",
        Setup {
            evaluator: None,
            ..current(2)
        },
    );
    assert_eq!(
        expert_error(admit(
            &mut rt,
            StructuralOp::Birth { result: spec("d") },
            PRE
        )),
        ExpertAdmissionError::StalePolicyEvidence {
            kind: GATE,
            reason: StaleEvidenceReason::EpochMismatch {
                current: 2,
                evidence: 1
            },
        }
    );
}

// ---- revocation and tightening fence the experts admitted under the old epoch ---------------

/// A runtime in the state used by the fence tests, with the facts the test knows independently:
/// pre-regime `p`; gate standing epoch 2 and evaluator epoch 1; `a`, `c` serving, `b` frozen,
/// `d` retired and `e` already quarantined, all admitted under gate epoch 2 and evaluator 1.
fn regime() -> PtrRuntime {
    let mut rt = runtime();
    birth(&mut rt, "p", PRE);
    assert_eq!(change(&mut rt, GATE, 2, LOOSE).unwrap().fence_count(), 0);
    assert_eq!(
        change(&mut rt, EVALUATOR, 1, LOOSE).unwrap().fence_count(),
        0
    );
    let setup = Setup {
        epoch: 2,
        version: 2,
        evaluator: Some(1),
    };
    for pod in ["a", "b", "c", "d", "e"] {
        birth(&mut rt, pod, setup);
    }
    freeze(&mut rt, "b", setup);
    let op = StructuralOp::Retire {
        target: target(&rt, "d"),
        reason: RetiredReason::GateFailed,
    };
    admit(&mut rt, op, setup).unwrap();
    rt.fence_expert(&quarantine_request("e", 1), &revocation_grant())
        .unwrap();
    rt
}

fn quarantine_request(pod: &str, version: u64) -> ExpertFenceRequest {
    ExpertFenceRequest {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        cause: ExpertFenceCause::AuthorityQuarantine,
        revocation: None,
    }
}

fn member(pod: &str, version: u64, resume: ptr_ledger::ExpertFenceResume) -> ExpertFenceSetMember {
    ExpertFenceSetMember {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        action: ptr_ledger::ExpertFenceAction::Fence,
        resume,
    }
}

fn tag_31(rt: &PtrRuntime) -> (u32, [u8; 32]) {
    let LedgerEvent::ExpertAuthorityChanged {
        fence_count,
        fence_digest,
        ..
    } = &rt.committed_events().last().unwrap().event
    else {
        panic!("the last event is not an authority change");
    };
    (*fence_count, *fence_digest)
}

#[test]
fn a_tightening_rotation_fences_exactly_the_experts_admitted_under_the_old_epoch() {
    use ptr_ledger::ExpertFenceResume::{Frozen, Serving};
    let mut rt = regime();
    // The facts the test knows: p, a and c are serving (state version 1), b is frozen (2). d is
    // retired and e is already quarantined, so neither is a member.
    assert_eq!(state_of(&rt, "p"), (ExpertState::Serving, 1));
    assert_eq!(state_of(&rt, "b"), (ExpertState::Frozen, 2));
    assert_eq!(state_of(&rt, "e"), (ExpertState::Fenced, 2));
    let admitted = |pod| {
        rt.expert_registry()
            .admission_epochs(&project(), &PodId::from(pod))
            .unwrap()
    };
    assert_eq!(
        (admitted("p").epoch(GATE), admitted("p").epoch(EVALUATOR)),
        (0, 0)
    );
    assert_eq!(
        (admitted("a").epoch(GATE), admitted("a").epoch(EVALUATOR)),
        (2, 1)
    );
    let expected = [
        member("a", 1, Serving),
        member("b", 2, Frozen),
        member("c", 1, Serving),
        member("p", 1, Serving),
    ];
    let expected_digest = expert_authority_fence_set_digest(GATE, 3, TIGHTEN, &expected);

    let before = applied(&rt);
    let outcome = change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    assert_eq!(applied(&rt), before + 1, "one event");
    assert_eq!(outcome.fence_count(), 4);
    assert_eq!(outcome.fence_digest(), &expected_digest);
    assert_eq!(outcome.fenced(), &expected);
    assert_eq!(
        tag_31(&rt),
        (4, expected_digest),
        "the event carries count and digest"
    );
    assert_eq!(rt.expert_authority(GATE).unwrap().epoch(), 3);
    assert_eq!(rt.expert_authority(GATE).unwrap().id_digest(), &cfg(3));
    // The evaluator standing did not move.
    assert_eq!(rt.expert_authority(EVALUATOR).unwrap().epoch(), 1);

    for (pod, version) in [("a", 2), ("b", 3), ("c", 2), ("p", 2)] {
        assert_eq!(state_of(&rt, pod), (ExpertState::Fenced, version), "{pod}");
        assert_eq!(
            rt.expert_registry()
                .fence_cause(&project(), &PodId::from(pod)),
            Some(ExpertFenceCause::AuthorityQuarantine),
            "{pod}"
        );
        assert_eq!(route(&rt, pod), Err(RouteRefusal::Fenced), "{pod}");
    }
    assert_eq!(
        rt.expert_registry()
            .fence_resume(&project(), &PodId::from("b")),
        Some(Frozen)
    );
    // The non-members are untouched.
    assert_eq!(
        state_of(&rt, "d"),
        (ExpertState::Retired(RetiredReason::GateFailed), 2)
    );
    assert_eq!(state_of(&rt, "e"), (ExpertState::Fenced, 2));
    assert_eq!(outcome.fence_epochs().len(), 4);

    // The fences, the standings and the admission epochs survive a replay.
    let replayed = PtrRuntime::replay(PtrConfig::default(), &history(&rt)).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
}

#[test]
fn a_revoke_fences_the_same_way_and_leaves_an_evaluator_regime_alone() {
    use ptr_ledger::ExpertFenceResume::{Frozen, Serving};
    let mut rt = regime();
    // An evaluator revocation fences by the evaluator epoch: p (0) and a, b, c (1) are below 2.
    let expected = [
        member("a", 1, Serving),
        member("b", 2, Frozen),
        member("c", 1, Serving),
        member("p", 1, Serving),
    ];
    let outcome = change(&mut rt, EVALUATOR, 2, REVOKE).unwrap();
    assert_eq!(
        outcome.fence_digest(),
        &expert_authority_fence_set_digest(EVALUATOR, 2, REVOKE, &expected)
    );
    assert_eq!(outcome.fenced(), &expected);
    for pod in ["a", "b", "c", "p"] {
        assert_eq!(route(&rt, pod), Err(RouteRefusal::Fenced), "{pod}");
    }
    assert_eq!(
        rt.expert_authority(GATE).unwrap().epoch(),
        2,
        "gate unchanged"
    );
    assert_eq!(
        rt.expert_authority(EVALUATOR).unwrap().id_digest(),
        &evaluator(2)
    );
    // The set is not the one of another class or kind.
    assert_ne!(
        outcome.fence_digest(),
        &expert_authority_fence_set_digest(EVALUATOR, 2, TIGHTEN, &expected)
    );
    assert_ne!(
        outcome.fence_digest(),
        &expert_authority_fence_set_digest(GATE, 2, REVOKE, &expected)
    );
}

#[test]
fn a_non_tightening_rotation_fences_nothing_and_the_experts_stay_routable() {
    let mut rt = regime();
    for pod in ["a", "b", "c", "p"] {
        assert_eq!(route(&rt, pod), Ok(()), "{pod}");
    }
    let before = rt.expert_registry().clone();
    let outcome = change(&mut rt, GATE, 3, LOOSE).unwrap();
    assert_eq!(outcome.fence_count(), 0);
    assert!(outcome.fenced().is_empty());
    assert_eq!(
        outcome.fence_digest(),
        &expert_authority_fence_set_digest(GATE, 3, LOOSE, &[])
    );
    for pod in ["a", "b", "c", "p"] {
        assert_eq!(route(&rt, pod), Ok(()), "{pod}");
    }
    for pod in ["a", "b", "c", "d", "e", "p"] {
        assert_eq!(state_of(&rt, pod), {
            let entry = before.get(&project(), &PodId::from(pod)).unwrap();
            (entry.state, entry.state_version)
        });
    }
    assert_eq!(rt.expert_authority(GATE).unwrap().epoch(), 3);
    // The old evidence is stale for a new admission all the same.
    assert_eq!(
        expert_error(admit(
            &mut rt,
            StructuralOp::Birth { result: spec("q") },
            current(2)
        )),
        ExpertAdmissionError::StalePolicyEvidence {
            kind: GATE,
            reason: StaleEvidenceReason::EpochMismatch {
                current: 3,
                evidence: 2
            },
        }
    );
    // The event cannot claim a fence for a non-tightening class, nor leave one out of a
    // tightening one.
    let mut history = history(&rt);
    if let LedgerEvent::ExpertAuthorityChanged { fence_count, .. } =
        &mut history.last_mut().unwrap().event
    {
        *fence_count = 1;
    }
    assert!(PtrRuntime::replay(PtrConfig::default(), &history).is_err());
}

// ---- readmission needs evidence under the current epoch -------------------------------------

fn readmission_report(request: &ExpertFenceRequest, version: u8) -> BoundGateReport {
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        &request.project,
        &request.pod,
        request.generation,
        request.expected_state_version,
        request.expected_state_version + 1,
        request.cause,
    );
    gate_v(version)
        .evaluate(&AdapterId::from("candidate"), &matrix(), public())
        .unwrap()
        .bind(GateSubject::new(
            proposal_id,
            operation_id,
            *init_digest(request).as_bytes(),
            BASE_MODEL_DIGEST,
            EVAL_MANIFEST,
        ))
}

/// The manifest digest of `spec(pod)`, which a readmission report binds to.
fn init_digest(request: &ExpertFenceRequest) -> ManifestDigest {
    *spec(&request.pod.0).manifest_digest()
}

fn gate_grant(setup: Setup) -> ExpertFenceGrant {
    let grant = ExpertFenceGrant::gate(
        project(),
        requirement(setup.epoch, cfg(setup.version)),
        binding(),
    );
    match setup.evaluator {
        Some(version) => grant.with_evaluator(evaluator(version)),
        None => grant,
    }
}

fn readmit(
    rt: &mut PtrRuntime,
    pod: &str,
    version: u64,
    setup: Setup,
) -> Result<CommitIndex, RuntimeError> {
    let request = quarantine_request(pod, version);
    rt.readmit_expert(
        &request,
        &readmission_report(&request, setup.version),
        &gate_grant(setup),
    )
}

#[test]
fn readmission_under_the_old_epoch_is_refused_and_under_the_current_epoch_accepted() {
    let mut rt = regime();
    change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    let (before, registry) = (applied(&rt), rt.expert_registry().clone());

    // Evidence under the old gate epoch and configuration.
    assert_eq!(
        expert_error(readmit(&mut rt, "a", 2, current(2))),
        ExpertAdmissionError::StalePolicyEvidence {
            kind: GATE,
            reason: StaleEvidenceReason::EpochMismatch {
                current: 3,
                evidence: 2
            },
        }
    );
    // The current epoch but the old configuration.
    assert_eq!(
        expert_error(readmit(
            &mut rt,
            "a",
            2,
            Setup {
                version: 2,
                ..current(3)
            }
        )),
        ExpertAdmissionError::StalePolicyEvidence {
            kind: GATE,
            reason: StaleEvidenceReason::IdentityMismatch,
        }
    );
    // Everything current but the evaluator (standing: version 1).
    assert_eq!(
        expert_error(readmit(
            &mut rt,
            "a",
            2,
            Setup {
                evaluator: None,
                ..current(3)
            }
        )),
        ExpertAdmissionError::StalePolicyEvidence {
            kind: EVALUATOR,
            reason: StaleEvidenceReason::IdentityNotNamed,
        }
    );
    assert_eq!(
        expert_error(readmit(
            &mut rt,
            "a",
            2,
            Setup {
                evaluator: Some(2),
                ..current(3)
            }
        )),
        ExpertAdmissionError::StalePolicyEvidence {
            kind: EVALUATOR,
            reason: StaleEvidenceReason::IdentityMismatch,
        }
    );
    assert_eq!(applied(&rt), before, "nothing was appended");
    assert_eq!(rt.expert_registry(), &registry);
    assert_eq!(route(&rt, "a"), Err(RouteRefusal::Fenced));

    // The control: current gate epoch and configuration, and the evaluator of the standing.
    let current_evidence = Setup {
        epoch: 3,
        version: 3,
        evaluator: Some(1),
    };
    readmit(&mut rt, "a", 2, current_evidence).unwrap();
    assert_eq!(state_of(&rt, "a"), (ExpertState::Serving, 3));
    assert_eq!(route(&rt, "a"), Ok(()));
    let epochs = rt
        .expert_registry()
        .admission_epochs(&project(), &PodId::from("a"))
        .unwrap();
    assert_eq!((epochs.epoch(GATE), epochs.epoch(EVALUATOR)), (3, 1));
    // Readmitted evidence is current: a non-tightening rotation spares it, a revocation fences it.
    change(&mut rt, GATE, 4, LOOSE).unwrap();
    assert_eq!(route(&rt, "a"), Ok(()));
    let outcome = change(&mut rt, GATE, 5, REVOKE).unwrap();
    assert_eq!(outcome.fence_count(), 1);
    assert_eq!(route(&rt, "a"), Err(RouteRefusal::Fenced));
}

// ---- monotone epochs, identity, grant ----------------------------------------------------------

#[test]
fn epochs_must_strictly_increase_per_kind_and_a_revoked_identity_cannot_stay_current() {
    let mut rt = runtime();
    change(&mut rt, GATE, 5, LOOSE).unwrap();
    let before = applied(&rt);
    for epoch in [5, 4, 1] {
        assert_eq!(
            expert_error(change(&mut rt, GATE, epoch, LOOSE)),
            ExpertAdmissionError::AuthorityEpochNotIncreasing {
                kind: GATE,
                current: 5,
                found: epoch,
            }
        );
    }
    // Another kind has its own epoch line.
    change(&mut rt, EVALUATOR, 1, LOOSE).unwrap();
    // A revocation that keeps the revoked identity current is refused; a rotation may keep it.
    assert_eq!(
        expert_error(rt.change_expert_authority(
            &global_grant(),
            ExpertAuthorityChange::new(GATE, cfg(5), 6, REVOKE)
        )),
        ExpertAdmissionError::AuthorityIdentityUnchanged { kind: GATE }
    );
    assert_eq!(
        applied(&rt),
        before + 1,
        "only the evaluator event was appended"
    );
    // Epoch zero and an all-zero identity are not events at all.
    assert!(rt
        .change_expert_authority(
            &global_grant(),
            ExpertAuthorityChange::new(EVALUATOR, evaluator(3), 0, LOOSE)
        )
        .is_err());
    assert!(rt
        .change_expert_authority(
            &global_grant(),
            ExpertAuthorityChange::new(EVALUATOR, [0; 32], 9, LOOSE)
        )
        .is_err());
    assert_eq!(applied(&rt), before + 1);

    // Replay refuses a history whose epochs do not rise.
    let mut forged = history(&rt);
    forged.push(CommittedEvent {
        index: CommitIndex(forged.len() as u64 + 1),
        event: LedgerEvent::ExpertAuthorityChanged {
            kind: GATE,
            id_digest: cfg(2),
            epoch: 5,
            class: LOOSE,
            fence_count: 0,
            fence_digest: expert_authority_fence_set_digest(GATE, 5, LOOSE, &[]),
        },
    });
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &forged),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::AuthorityEpochNotIncreasing { .. }
        ))
    ));
}

#[test]
fn a_gate_grant_cannot_issue_an_authority_change_and_the_public_commit_refuses_the_event() {
    let mut rt = regime();
    let before = (applied(&rt), rt.expert_registry().clone());
    let gate = gate_grant(current(2));
    assert!(matches!(
        expert_error(
            rt.change_expert_authority(&gate, ExpertAuthorityChange::new(GATE, cfg(3), 3, TIGHTEN))
        ),
        ExpertAdmissionError::FenceAuthorityMismatch { .. }
    ));
    let members = [member("a", 1, ptr_ledger::ExpertFenceResume::Serving)];
    let event = LedgerEvent::ExpertAuthorityChanged {
        kind: GATE,
        id_digest: cfg(3),
        epoch: 3,
        class: TIGHTEN,
        fence_count: 1,
        fence_digest: expert_authority_fence_set_digest(GATE, 3, TIGHTEN, &members),
    };
    assert!(matches!(
        rt.commit(event),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
    assert_eq!((applied(&rt), rt.expert_registry().clone()), before);
}

#[test]
fn a_project_scoped_revocation_grant_cannot_issue_an_authority_change() {
    // An authority change fences the experts of every project, so the project-scoped revocation
    // grant (the one that fences a single project's experts) is refused with a typed error and
    // nothing is appended; only the runtime-wide grant passes.
    let mut rt = regime();
    let before = (applied(&rt), rt.expert_registry().clone());
    assert_eq!(
        expert_error(rt.change_expert_authority(
            &revocation_grant(),
            ExpertAuthorityChange::new(GATE, cfg(3), 3, TIGHTEN)
        )),
        ExpertAdmissionError::FenceAuthorityMismatch {
            action: ptr_ledger::ExpertFenceAction::Fence,
            required: ExpertFenceAuthority::GlobalRevocation,
        }
    );
    assert_eq!((applied(&rt), rt.expert_registry().clone()), before);
    // A global grant may not fence one expert either.
    assert!(matches!(
        expert_error(rt.fence_expert(&quarantine_request("a", 1), &global_grant())),
        ExpertAdmissionError::FenceAuthorityMismatch { .. }
    ));
    change(&mut rt, GATE, 3, TIGHTEN).unwrap();
}

#[test]
fn a_superseded_global_grant_cannot_issue_an_authority_change() {
    use ptr_runtime::expert::ExpertGrantKind;
    let mut rt = regime();
    let stale = global_grant();
    rt.supersede_expert_grant(ExpertGrantKind::GlobalRevocation)
        .unwrap();
    let before = applied(&rt);
    assert!(matches!(
        expert_error(
            rt.change_expert_authority(
                &stale,
                ExpertAuthorityChange::new(GATE, cfg(3), 3, TIGHTEN)
            )
        ),
        ExpertAdmissionError::GrantGenerationStale { .. }
    ));
    assert_eq!(applied(&rt), before);
    assert_eq!(rt.expert_authority(GATE).unwrap().epoch(), 2);
}

// ---- replay of a tampered set ------------------------------------------------------------------

#[test]
fn a_tampered_fence_set_is_refused_on_replay() {
    let mut rt = regime();
    change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    let genuine = history(&rt);
    assert!(PtrRuntime::replay(PtrConfig::default(), &genuine).is_ok());
    let last = genuine.len() - 1;
    let tamper = |edit: &dyn Fn(&mut u32, &mut [u8; 32])| {
        let mut forged = genuine.clone();
        if let LedgerEvent::ExpertAuthorityChanged {
            fence_count,
            fence_digest,
            ..
        } = &mut forged[last].event
        {
            edit(fence_count, fence_digest);
        }
        PtrRuntime::replay(PtrConfig::default(), &forged)
    };
    // A smaller, a larger and a differently composed set, and a flipped digest.
    assert!(matches!(
        tamper(&|count, _| *count = 3),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceSetMismatch {
                derived_count: 4,
                event_count: 3,
                digest_matches: true,
            }
        ))
    ));
    assert!(matches!(
        tamper(&|count, _| *count = 5),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceSetMismatch { .. }
        ))
    ));
    assert!(matches!(
        tamper(&|_, digest| digest[0] ^= 1),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceSetMismatch {
                digest_matches: false,
                ..
            }
        ))
    ));
    // A forged event that fences nobody while the registry has members to fence is the same
    // mismatch.
    assert!(matches!(
        tamper(&|count, digest| {
            *count = 0;
            *digest = expert_authority_fence_set_digest(GATE, 3, TIGHTEN, &[]);
        }),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceSetMismatch { .. }
        ))
    ));
    // A tail event handed to a compacted restore is held to the same rule.
    let mut rt = regime();
    let floor = rt.export_compacted_snapshot().unwrap();
    change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    let mut tail = history(&rt)[history(&rt).len() - 1..].to_vec();
    if let LedgerEvent::ExpertAuthorityChanged { fence_count, .. } = &mut tail[0].event {
        *fence_count = 1;
    }
    assert!(matches!(
        PtrRuntime::restore_compacted(PtrConfig::default(), floor.bytes(), floor.anchor(), &tail),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceSetMismatch { .. }
        ))
    ));
}

#[test]
fn replay_re_checks_the_gate_epoch_of_a_record_against_the_standing() {
    let mut rt = runtime();
    change(&mut rt, GATE, 2, LOOSE).unwrap();
    birth(
        &mut rt,
        "a",
        Setup {
            evaluator: None,
            ..current(2)
        },
    );
    let mut forged = history(&rt);
    // The record claims the old gate epoch: rebuild the structural event with a stale grant by
    // admitting under a runtime that had no standing, then splice it after the change.
    let mut pre = runtime();
    birth(
        &mut pre,
        "a",
        Setup {
            evaluator: None,
            ..PRE
        },
    );
    let pre_event = history(&pre).remove(0);
    forged[1].event = pre_event.event;
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &forged),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::StalePolicyEvidence { kind: GATE, .. }
        ))
    ));
}

// ---- persistence -------------------------------------------------------------------------------

fn temp_log() -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "ptr-expert-authority-{}-{}",
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

fn assert_regime_after_tightening(rt: &PtrRuntime) {
    for pod in ["a", "b", "c", "p"] {
        assert_eq!(route(rt, pod), Err(RouteRefusal::Fenced), "{pod}");
        assert_eq!(
            rt.expert_registry()
                .fence_cause(&project(), &PodId::from(pod)),
            Some(ExpertFenceCause::AuthorityQuarantine),
            "{pod}"
        );
    }
    assert_eq!(rt.expert_authority(GATE).unwrap().epoch(), 3);
    assert_eq!(rt.expert_authority(GATE).unwrap().id_digest(), &cfg(3));
    assert_eq!(rt.expert_authority(EVALUATOR).unwrap().epoch(), 1);
    let epochs = rt
        .expert_registry()
        .admission_epochs(&project(), &PodId::from("a"))
        .unwrap();
    assert_eq!((epochs.epoch(GATE), epochs.epoch(EVALUATOR)), (2, 1));
}

#[test]
fn the_epochs_and_the_fences_survive_open_durable() {
    let (dir, path) = temp_log();
    let mut rt = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    prime(&mut rt);
    birth(&mut rt, "p", PRE);
    change(&mut rt, GATE, 2, LOOSE).unwrap();
    change(&mut rt, EVALUATOR, 1, LOOSE).unwrap();
    let setup = Setup {
        epoch: 2,
        version: 2,
        evaluator: Some(1),
    };
    for pod in ["a", "c"] {
        birth(&mut rt, pod, setup);
    }
    birth(&mut rt, "b", setup);
    freeze(&mut rt, "b", setup);
    change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    let registry = rt.expert_registry().clone();
    assert_regime_after_tightening(&rt);
    drop(rt);

    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(reopened.expert_registry(), &registry);
    assert_regime_after_tightening(&reopened);
    // The reopened runtime still enforces the standing and the monotone epoch.
    prime(&mut reopened);
    assert!(matches!(
        expert_error(admit(
            &mut reopened,
            StructuralOp::Birth { result: spec("q") },
            current(2)
        )),
        ExpertAdmissionError::StalePolicyEvidence { kind: GATE, .. }
    ));
    assert!(matches!(
        expert_error(change(&mut reopened, GATE, 3, LOOSE)),
        ExpertAdmissionError::AuthorityEpochNotIncreasing { .. }
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_epochs_and_the_fences_survive_compaction_with_and_without_a_tail() {
    // A history without a standing keeps the earlier layout, byte for byte.
    let mut plain = runtime();
    birth(&mut plain, "a", PRE);
    assert_eq!(
        &plain.export_compacted_snapshot().unwrap().bytes()[..8],
        b"PTRCS006"
    );

    let mut rt = regime();
    let floor = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&floor.bytes()[..8], b"PTRCS009");
    change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    let tail = history(&rt)[history(&rt).len() - 1..].to_vec();

    // Restore at the floor and replay the tightening above it.
    let resumed =
        PtrRuntime::restore_compacted(PtrConfig::default(), floor.bytes(), floor.anchor(), &tail)
            .unwrap();
    assert_eq!(resumed.expert_registry(), rt.expert_registry());
    assert_regime_after_tightening(&resumed);

    // A snapshot after the tightening carries the standings and the fences.
    let after = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&after.bytes()[..8], b"PTRCS009");
    let mut restored =
        PtrRuntime::restore_compacted(PtrConfig::default(), after.bytes(), after.anchor(), &[])
            .unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_regime_after_tightening(&restored);
    prime(&mut restored);
    assert!(matches!(
        expert_error(admit(
            &mut restored,
            StructuralOp::Birth { result: spec("q") },
            current(2)
        )),
        ExpertAdmissionError::StalePolicyEvidence { kind: GATE, .. }
    ));
    // The admission epochs crossed the floor: a later revocation of the evaluator fences the one
    // expert readmitted under the new epoch, which only the persisted epochs can say.
    let request = quarantine_request("a", 2);
    restored
        .readmit_expert(
            &request,
            &readmission_report(&request, 3),
            &gate_grant(Setup {
                epoch: 3,
                version: 3,
                evaluator: Some(1),
            }),
        )
        .unwrap();
    assert_eq!(route(&restored, "a"), Ok(()));
    let outcome = change(&mut restored, EVALUATOR, 2, REVOKE).unwrap();
    assert_eq!(outcome.fence_count(), 1);
    assert_eq!(outcome.fenced()[0].pod, PodId::from("a"));

    // A restored runtime refuses to apply a change that does not rise.
    assert!(matches!(
        expert_error(change(&mut restored, GATE, 3, LOOSE)),
        ExpertAdmissionError::AuthorityEpochNotIncreasing { .. }
    ));
}

#[test]
fn a_relabelled_or_tampered_authority_snapshot_is_refused() {
    let rt = regime();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    // Naming the snapshot as the previous layout leaves its lifecycle section mislabelled.
    let mut older = snapshot.bytes().to_vec();
    older[..8].copy_from_slice(b"PTRCS008");
    let digest = ptr_ledger::integrity::sha256(&older[..older.len() - 32]);
    let end = older.len() - 32;
    older[end..].copy_from_slice(&digest);
    let anchor = ptr_runtime::compacted::CompactedAnchor {
        digest,
        ..snapshot.anchor()
    };
    assert!(matches!(
        PtrRuntime::restore_compacted(PtrConfig::default(), &older, anchor, &[]),
        Err(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
    ));
    // A flipped byte fails the digest.
    let mut flipped = snapshot.bytes().to_vec();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 1;
    assert!(
        PtrRuntime::restore_compacted(PtrConfig::default(), &flipped, snapshot.anchor(), &[])
            .is_err()
    );
}

// ---- the KV layers -------------------------------------------------------------------------------

const CHAIN: KvChainId = KvChainId([1; 32]);

fn expert_dep(pod: &str, epoch: u64) -> KvDependency {
    KvDependency::expert_fenced(&project(), &PodId::from(pod), Generation(1), [0; 32], epoch)
}

fn kv_schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn kv_salt() -> KvSalt {
    KvSalt::derive(
        &PrincipalId::from("alice"),
        &project(),
        &ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    )
}

fn kv_request() -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        vec![PromptSegment::raw_from_text(&ByteTokenizer::new(), &"ab".repeat(16)).unwrap()],
    )
    .unwrap()
}

struct Rig {
    backend: InMemoryPagedKvBackend,
}

impl Rig {
    fn new(rt: &mut PtrRuntime) -> Self {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
        let catalog = KvPrefixCatalog::new(
            backend.take_fork_authority().unwrap(),
            CatalogLimits::default(),
        )
        .unwrap()
        .with_retirer(Arc::new(backend.clone()));
        rt.enable_kv_prefix_catalog(catalog).unwrap();
        Self { backend }
    }

    fn insert(&self, rt: &mut PtrRuntime, deps: Vec<KvDependency>) -> Result<(), KvCatalogError> {
        let req = kv_request();
        let mut cache = self
            .backend
            .allocate_bound(
                kv_schema(),
                64,
                KvPageBinding {
                    model: ModelVersion::from("base-v1"),
                    adapter: AdapterVersion::from("none"),
                    generation: Generation(1),
                    execution_manifest: [3; 32],
                    principal: PrincipalId::from("alice"),
                    device: DeviceId::from("cpu"),
                    dtype: KvTensorDType::F32,
                    page_tokens: KvPageSize::DEFAULT,
                },
            )
            .unwrap();
        let rows = |seed: f32| TensorRef {
            layer: 0,
            shape: vec![32, 4],
            values: (0..128).map(|i| seed + i as f32).collect(),
        };
        self.backend
            .append(&mut cache, &[rows(1.0)], &[rows(9.0)])
            .unwrap();
        let prefix = self.backend.publish_prefix(&mut cache).unwrap();
        let bytes = self.backend.page_metrics(&cache).resident_bytes;
        rt.kv_prefix_insert(KvInsert {
            salt: &kv_salt(),
            chain: &CHAIN,
            request: &req,
            prefix,
            page_bytes: bytes,
            extra_dependencies: deps,
        })
        .unwrap()
    }
}

fn hits(rt: &mut PtrRuntime) -> bool {
    let req = kv_request();
    let hit = rt
        .kv_prefix_lookup(KvLookup {
            salt: &kv_salt(),
            chain: &CHAIN,
            request: &req,
            positions: &KvRequestPositions::Canonical,
        })
        .unwrap();
    match hit {
        Some(hit) => {
            let pin = hit.pin().clone();
            rt.kv_prefix_release(&pin);
            true
        }
        None => false,
    }
}

#[test]
fn a_tightening_kills_the_kv_entries_of_the_fenced_experts_and_a_loosening_does_not() {
    let mut rt = regime();
    let rig = Rig::new(&mut rt);
    rig.insert(&mut rt, vec![expert_dep("a", 1)]).unwrap();
    assert!(hits(&mut rt), "an entry on a is live before");
    // A non-tightening rotation changes no KV liveness.
    change(&mut rt, GATE, 3, LOOSE).unwrap();
    assert!(hits(&mut rt));
    change(&mut rt, GATE, 4, TIGHTEN).unwrap();
    assert!(
        !hits(&mut rt),
        "the entry recorded under a's epoch 1 is dead"
    );
    for epoch in [0, 1, 2] {
        assert_eq!(
            rig.insert(&mut rt, vec![expert_dep("a", epoch)]),
            Err(KvCatalogError::DependencyNotLive),
            "epoch {epoch}"
        );
    }
    // The same mapping applies on replay of the log.
    let mut replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    let rig = Rig::new(&mut replayed);
    assert_eq!(
        rig.insert(&mut replayed, vec![expert_dep("c", 1)]),
        Err(KvCatalogError::DependencyNotLive)
    );
    // Neither does the retired `d` nor the already quarantined `e` gain anything: their epochs
    // stay what they were.
    assert_eq!(
        rig.insert(&mut replayed, vec![expert_dep("e", 1)]),
        Err(KvCatalogError::DependencyNotLive)
    );
}

#[test]
fn a_standalone_kv_backend_is_handed_the_fence_set_the_event_does_not_carry() {
    let mut rt = regime();
    let outcome = change(&mut rt, GATE, 3, TIGHTEN).unwrap();
    let event = rt.committed_events().last().unwrap().event.clone();
    let fences = outcome.fence_epochs();
    assert_eq!(fences.get("expert:proj/a"), Some(&2));
    assert_eq!(fences.get("expert:proj/b"), Some(&3));

    let scope = ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: project(),
    };
    let backend = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope).unwrap();
    // The event names no experts: the plain hook refuses it rather than leave them live.
    assert!(matches!(
        backend.apply_committed_event(&event),
        Err(ReferenceKvError::AuthorityChangeNeedsFences)
    ));
    assert!(matches!(
        backend.apply_authority_change(&rt.committed_events()[0].event, &fences),
        Err(ReferenceKvError::NotAnAuthorityChange)
    ));
    let live = backend.own_liveness();
    assert!(live.is_live(&expert_dep("a", 1)));
    backend.apply_authority_change(&event, &fences).unwrap();
    assert!(!live.is_live(&expert_dep("a", 1)));
    assert!(live.is_live(&expert_dep("a", 2)));
    assert!(
        live.is_live(&expert_dep("d", 1)),
        "a non-member keeps its entries"
    );
    let _: BTreeMap<String, u64> = fences;
}
