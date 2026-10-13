//! The compound input invalidation (`InputInvalidated`, ledger tag 30), the expert dependency
//! index behind it and the input checks of a version 3 structural record, through the public
//! API: one event revokes or supersedes an input and fences every dependent, transitively; the
//! fence set is committed by count and digest and re-derived on replay; stale input
//! generations and digests are refused at admission; and the whole thing survives a durable
//! reopen and compaction.

use std::collections::BTreeMap;
use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_ledger::{
    CommittedEvent, ExpertFenceAction, ExpertFenceCause, ExpertFenceResume, InputInvalidationKind,
    LedgerEvent,
};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertInputRef, ExpertInputs, ExpertLevel,
    ExpertPodSpec, ExpertProposal, ExpertState, ExpertTarget, ExpertTargetInit,
    InMemoryPagedKvBackend, KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType,
    KvTensorSchema, ManifestDigest, PagedKvTensorBackend, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp, TensorRef,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_artifact_digest_with_inputs,
    expert_evidence_base_model_digest, expert_fence_set_digest, ExpertAdmissionError,
    ExpertAdmissionGrant, ExpertFenceAuthority, ExpertFenceGrant, ExpertFenceRequest,
    ExpertGrantKind, ExpertInputRole, InputInvalidation, RouteRefusal,
};
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvChainId, KvDependency, KvInsert, KvLivenessOracle, KvLookup,
    KvPrefixCatalog, KvRequestPositions, KvSalt, PtrRuntime, ReferenceKvError, ReferenceKvLimits,
    ReferenceKvReuseBackend, ReferenceKvScope, RuntimeError,
};
use ptr_types::{
    AdapterVersion, CapsuleId, CommitIndex, DeviceId, Generation, ModelVersion, PodId, PrincipalId,
    ProjectId, RequestId, Revision, SemanticContext,
};

mod expert_support;

use expert_support::{binding, current_binding, prime, requirement, EVAL_MANIFEST};

const T: &str = "dataset:train";
const E: &str = "dataset:eval";
const U: &str = "dataset:other";
const D1: [u8; 32] = [0x11; 32];
const D2: [u8; 32] = [0x22; 32];
const D3: [u8; 32] = [0x33; 32];
const REPLACEMENT: [u8; 32] = [0x77; 32];

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

fn propose(op: StructuralOp) -> ExpertProposal {
    ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap()
}

fn report_for(proposal: &ExpertProposal, inputs: Option<&ExpertInputs>) -> BoundGateReport {
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
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject)
}

fn admit(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
    inputs: Option<ExpertInputs>,
) -> Result<CommitIndex, RuntimeError> {
    prime(rt);
    let grant = ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        current_binding(rt, ExpertGrantKind::Admission),
    );
    let report = report_for(proposal, inputs.as_ref());
    match inputs {
        Some(inputs) => rt.admit_expert_proposal_with_inputs(proposal, &report, &grant, inputs),
        None => rt.admit_expert_proposal(proposal, &report, &grant),
    }
}

fn input(identity: &str, generation: u64, digest: [u8; 32]) -> ExpertInputRef {
    ExpertInputRef::new(identity, Generation(generation), digest).unwrap()
}

fn inputs(
    training: &[(&str, u64, [u8; 32])],
    evaluation: &[(&str, u64, [u8; 32])],
) -> ExpertInputs {
    ExpertInputs::new(
        training.iter().map(|i| input(i.0, i.1, i.2)).collect(),
        evaluation.iter().map(|i| input(i.0, i.1, i.2)).collect(),
    )
    .unwrap()
}

/// A Segment-level expert is not bound to a routing slot, so many may be born.
fn segment_spec(pod: &str, generation: u64) -> ExpertPodSpec {
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new([pod.as_bytes()[0]; 32]).unwrap(),
        ExpertLevel::Segment,
        base(),
        Generation(generation),
    )
    .unwrap()
}

fn token_spec(pod: &str, generation: u64) -> ExpertPodSpec {
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new([pod.as_bytes()[0]; 32]).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(pod.as_bytes()[0]),
        },
        base(),
        Generation(generation),
    )
    .unwrap()
}

fn birth_with(rt: &mut PtrRuntime, spec: ExpertPodSpec, inputs: Option<ExpertInputs>) {
    admit(rt, &propose(StructuralOp::Birth { result: spec }), inputs).unwrap();
}

/// A token-level expert named `pod` resting on `inputs`.
fn born(rt: &mut PtrRuntime, pod: &str, inputs: ExpertInputs) {
    birth_with(rt, token_spec(pod, 1), Some(inputs));
}

fn target_of(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap()
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .unwrap()
}

fn state_of(rt: &PtrRuntime, pod: &str) -> (ExpertState, u64) {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap();
    (entry.state, entry.state_version)
}

fn route(rt: &PtrRuntime, pod: &str, generation: u64) -> Result<(), RouteRefusal> {
    rt.expert_route_admission(&project(), &PodId::from(pod), Generation(generation))
}

fn revocation_grant(rt: &PtrRuntime) -> ExpertFenceGrant {
    ExpertFenceGrant::revocation(project(), current_binding(rt, ExpertGrantKind::Fence))
}

/// The runtime-wide revocation authority an input invalidation needs.
fn global_grant(rt: &PtrRuntime) -> ExpertFenceGrant {
    ExpertFenceGrant::global_revocation(current_binding(rt, ExpertGrantKind::GlobalRevocation))
}

fn invalidate(
    rt: &mut PtrRuntime,
    request: InputInvalidation,
) -> Result<ptr_runtime::expert::InputInvalidationOutcome, RuntimeError> {
    prime(rt);
    let grant = global_grant(rt);
    rt.invalidate_input(&grant, request)
}

fn revoke(identity: &str, generation: u64, digest: [u8; 32]) -> InputInvalidation {
    InputInvalidation::revoked(identity, Generation(generation), digest)
}

fn supersede(
    identity: &str,
    old: u64,
    old_digest: [u8; 32],
    new: u64,
    new_digest: [u8; 32],
) -> InputInvalidation {
    InputInvalidation::superseded(
        identity,
        Generation(old),
        old_digest,
        Generation(new),
        new_digest,
        REPLACEMENT,
    )
}

fn expert_error<T: std::fmt::Debug>(result: Result<T, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {other:?}"),
    }
}

fn applied(rt: &PtrRuntime) -> u64 {
    rt.materialized_state().last_applied
}

fn history(rt: &PtrRuntime) -> Vec<CommittedEvent> {
    rt.committed_events().to_vec()
}

fn pods(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| (*n).to_owned()).collect()
}

fn fenced_pods(outcome: &ptr_runtime::expert::InputInvalidationOutcome) -> Vec<String> {
    outcome
        .fenced()
        .iter()
        .map(|member| member.pod.0.clone())
        .collect()
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

/// x rests on T (training); y on T and E (evaluation); z on U only; w names no inputs (a
/// version 2 record).
fn populated() -> PtrRuntime {
    let mut rt = runtime();
    born(&mut rt, "x", inputs(&[(T, 1, D1)], &[]));
    born(&mut rt, "y", inputs(&[(T, 1, D1)], &[(E, 1, D2)]));
    born(&mut rt, "z", inputs(&[(U, 1, D3)], &[]));
    birth_with(&mut rt, token_spec("w", 1), None);
    rt
}

fn temp_ledger(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "ptr-input-invalidation-{tag}-{}-{}",
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

fn assert_dependents_fenced_others_serving(rt: &PtrRuntime) {
    for pod in ["x", "y"] {
        assert_eq!(state_of(rt, pod), (ExpertState::Fenced, 2), "{pod}");
        assert_eq!(route(rt, pod, 1), Err(RouteRefusal::Fenced), "{pod}");
        assert_eq!(
            rt.expert_registry()
                .fence_cause(&project(), &PodId::from(pod)),
            Some(ExpertFenceCause::InputRevoked)
        );
        assert_eq!(
            rt.materialized_state()
                .values
                .get(&format!("expert:proj/{pod}")),
            Some(&"1:fenced:2".to_owned())
        );
    }
    for pod in ["z", "w"] {
        assert_eq!(state_of(rt, pod), (ExpertState::Serving, 1), "{pod}");
        assert_eq!(route(rt, pod, 1), Ok(()), "{pod}");
    }
}

// ---- revocation ---------------------------------------------------------------------------

#[test]
fn revoking_a_training_input_fences_every_dependent_in_one_event() {
    let mut rt = populated();
    for pod in ["x", "y", "z", "w"] {
        assert_eq!(route(&rt, pod, 1), Ok(()));
    }
    let before = rt.committed_events().len();
    let outcome = invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    // Exactly one event, tag 30, carrying the count and digest, no list.
    assert_eq!(rt.committed_events().len(), before + 1);
    assert_eq!(fenced_pods(&outcome), pods(&["x", "y"]));
    assert_eq!(outcome.fence_count(), 2);
    let LedgerEvent::InputInvalidated {
        input,
        generation,
        binding_digest,
        kind,
        fence_count,
        fence_digest,
    } = &rt.committed_events().last().unwrap().event
    else {
        panic!("not an input invalidation");
    };
    assert_eq!(
        (input.as_str(), *generation, *binding_digest),
        (T, Generation(1), D1)
    );
    assert_eq!(*kind, InputInvalidationKind::Revoked);
    assert_eq!(*fence_count, 2);
    // The digest is the one an independent computation over the explicit set yields.
    let members: Vec<_> = ["x", "y"]
        .iter()
        .map(|pod| ptr_runtime::expert::ExpertFenceSetMember {
            project: project(),
            pod: PodId::from(*pod),
            generation: Generation(1),
            expected_state_version: 1,
            action: ExpertFenceAction::Fence,
            resume: ExpertFenceResume::Serving,
        })
        .collect();
    assert_eq!(
        *fence_digest,
        expert_fence_set_digest(T, Generation(1), &members)
    );
    assert_eq!(outcome.fence_digest(), fence_digest);
    assert_eq!(
        outcome.fence_epochs(),
        BTreeMap::from([
            ("expert:proj/x".to_owned(), 2),
            ("expert:proj/y".to_owned(), 2)
        ])
    );
    assert_dependents_fenced_others_serving(&rt);
    // The input is revoked for the lifecycle authority too.
    assert_eq!(
        rt.generation_validity(T, Generation(1)),
        Some(ptr_types::Validity::Revoked)
    );
    assert!(rt
        .expert_registry()
        .dependencies()
        .is_revoked(T, Generation(1)));
    assert_eq!(
        rt.materialized_state().values.get("revoked:dataset:train"),
        Some(&"1".to_owned())
    );
}

#[test]
fn an_evaluation_input_is_invalidated_like_a_training_input() {
    let mut rt = populated();
    let outcome = invalidate(&mut rt, revoke(E, 1, D2)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["y"]));
    assert_eq!(state_of(&rt, "y"), (ExpertState::Fenced, 2));
    assert_eq!(route(&rt, "y", 1), Err(RouteRefusal::Fenced));
    for pod in ["x", "z", "w"] {
        assert_eq!(route(&rt, pod, 1), Ok(()), "{pod}");
    }
    // A record naming the revoked evaluation input is refused, and says it is an evaluation one.
    let applied_before = applied(&rt);
    let error = expert_error(admit(
        &mut rt,
        &propose(StructuralOp::Birth {
            result: token_spec("n", 1),
        }),
        Some(inputs(&[], &[(E, 1, D2)])),
    ));
    assert_eq!(
        error,
        ExpertAdmissionError::InputRevoked {
            role: ExpertInputRole::Evaluation,
            identity: E.into(),
            generation: Generation(1),
        }
    );
    assert_eq!(applied(&rt), applied_before);
}

#[test]
fn a_revoked_or_stale_input_refuses_admission_for_both_roles_and_appends_nothing() {
    let mut rt = populated();
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    let before = history(&rt);
    let registry = rt.expert_registry().clone();
    let attempt = |rt: &mut PtrRuntime,
                   training: &[(&str, u64, [u8; 32])],
                   evaluation: &[(&str, u64, [u8; 32])]| {
        expert_error(admit(
            rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(training, evaluation)),
        ))
    };
    assert_eq!(
        attempt(&mut rt, &[(T, 1, D1)], &[]),
        ExpertAdmissionError::InputRevoked {
            role: ExpertInputRole::Training,
            identity: T.into(),
            generation: Generation(1),
        }
    );
    assert_eq!(
        attempt(&mut rt, &[], &[(T, 1, D1)]),
        ExpertAdmissionError::InputRevoked {
            role: ExpertInputRole::Evaluation,
            identity: T.into(),
            generation: Generation(1),
        }
    );
    // Positive control: another generation of the same identity is not revoked.
    birth_with(
        &mut rt,
        token_spec("n", 1),
        Some(inputs(&[(T, 2, D3)], &[])),
    );
    assert_eq!(state_of(&rt, "n"), (ExpertState::Serving, 1));
    assert_eq!(history(&rt)[..before.len()], before[..]);
    assert_ne!(rt.expert_registry(), &registry);
}

#[test]
fn a_plain_revocation_refuses_admission_and_routing_before_any_fence() {
    let mut rt = populated();
    rt.commit(LedgerEvent::Revoked {
        subject: T.to_owned(),
        generation: Generation(1),
    })
    .unwrap();
    // The revoke-before-fence window: nothing fenced x, yet it is not routable.
    assert_eq!(state_of(&rt, "x"), (ExpertState::Serving, 1));
    for pod in ["x", "y"] {
        assert_eq!(
            route(&rt, pod, 1),
            Err(RouteRefusal::DependencyNotLive {
                identity: T.into(),
                generation: Generation(1),
            }),
            "{pod}"
        );
    }
    // An expert whose record named no inputs is outside this check until it is fenced.
    assert_eq!(route(&rt, "w", 1), Ok(()));
    assert_eq!(route(&rt, "z", 1), Ok(()));
    assert_eq!(
        expert_error(admit(
            &mut rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[(T, 1, D1)], &[])),
        )),
        ExpertAdmissionError::InputRevoked {
            role: ExpertInputRole::Training,
            identity: T.into(),
            generation: Generation(1),
        }
    );
    // The old pair keeps working: fence per expert with the committed revocation.
    let grant = revocation_grant(&rt);
    for pod in ["x", "y"] {
        rt.fence_expert(
            &ExpertFenceRequest {
                project: project(),
                pod: PodId::from(pod),
                generation: Generation(1),
                expected_state_version: 1,
                cause: ExpertFenceCause::InputRevoked,
                revocation: Some(ptr_ledger::ExpertFenceRevocation {
                    subject: T.to_owned(),
                    generation: Generation(1),
                }),
            },
            &grant,
        )
        .unwrap();
    }
    assert_eq!(state_of(&rt, "x"), (ExpertState::Fenced, 2));
    assert_eq!(route(&rt, "x", 1), Err(RouteRefusal::Fenced));
    // The compound event after the plain revocation sweeps what is left and is still accepted.
    let outcome = invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    assert_eq!(outcome.fence_count(), 0);
}

#[test]
fn a_lifecycle_supersession_makes_a_recorded_input_stale_without_any_fence() {
    let mut rt = runtime();
    for generation in [1, 2] {
        rt.commit(LedgerEvent::CapsuleCommitted {
            project: project(),
            capsule: CapsuleId::from("cap"),
            generation: Generation(generation),
        })
        .unwrap();
    }
    // Generation 1 of the capsule is below the live generation: a stale input generation.
    assert_eq!(
        expert_error(admit(
            &mut rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[("cap", 1, D1)], &[])),
        )),
        ExpertAdmissionError::InputSuperseded {
            role: ExpertInputRole::Training,
            identity: "cap".into(),
            generation: Generation(1),
            floor: Generation(2),
        }
    );
    born(&mut rt, "n", inputs(&[("cap", 2, D1)], &[]));
    assert_eq!(route(&rt, "n", 1), Ok(()));
    rt.commit(LedgerEvent::CapsuleSuperseded {
        capsule: CapsuleId::from("cap"),
        old: Generation(2),
        new: Generation(3),
    })
    .unwrap();
    assert_eq!(
        route(&rt, "n", 1),
        Err(RouteRefusal::DependencyNotLive {
            identity: "cap".into(),
            generation: Generation(2),
        })
    );
}

// ---- supersession and the digest registry ---------------------------------------------------

#[test]
fn a_supersession_advances_the_digest_and_refuses_the_old_one() {
    let mut rt = populated();
    let before = rt.committed_events().len();
    let outcome = invalidate(&mut rt, supersede(T, 1, D1, 2, D3)).unwrap();
    assert_eq!(rt.committed_events().len(), before + 1);
    assert_eq!(fenced_pods(&outcome), pods(&["x", "y"]));
    assert_dependents_fenced_others_serving(&rt);
    assert!(matches!(
        rt.committed_events().last().unwrap().event,
        LedgerEvent::InputInvalidated {
            kind: InputInvalidationKind::Superseded { .. },
            fence_count: 2,
            ..
        }
    ));
    let index = rt.expert_registry().dependencies();
    assert_eq!(index.supersession_floor(T), Some(Generation(2)));
    assert_eq!(index.registered_digest(T, Generation(2)), Some(D3));
    assert_eq!(index.registered_digest(T, Generation(1)), Some(D1));
    // A supersession is not a revocation of the lifecycle authority.
    assert_eq!(rt.generation_validity(T, Generation(1)), None);
    assert!(!index.is_revoked(T, Generation(1)));

    let attempt = |rt: &mut PtrRuntime, generation: u64, digest: [u8; 32]| {
        expert_error(admit(
            rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[(T, generation, digest)], &[])),
        ))
    };
    let before = history(&rt);
    // Stale generation, whatever the digest.
    for digest in [D1, D3] {
        assert_eq!(
            attempt(&mut rt, 1, digest),
            ExpertAdmissionError::InputSuperseded {
                role: ExpertInputRole::Training,
                identity: T.into(),
                generation: Generation(1),
                floor: Generation(2),
            }
        );
    }
    // The new generation with the old digest is a stale digest.
    assert_eq!(
        attempt(&mut rt, 2, D1),
        ExpertAdmissionError::InputDigestStale {
            role: ExpertInputRole::Training,
            identity: T.into(),
            generation: Generation(2),
        }
    );
    assert_eq!(history(&rt), before);
    // Positive control: the new generation with the new digest is accepted.
    born(&mut rt, "n", inputs(&[(T, 2, D3)], &[]));
    assert_eq!(route(&rt, "n", 1), Ok(()));
}

#[test]
fn an_evaluation_input_supersession_works_the_same() {
    let mut rt = populated();
    let outcome = invalidate(&mut rt, supersede(E, 1, D2, 2, D3)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["y"]));
    let attempt = |rt: &mut PtrRuntime, generation: u64, digest: [u8; 32]| {
        expert_error(admit(
            rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[], &[(E, generation, digest)])),
        ))
    };
    assert!(matches!(
        attempt(&mut rt, 1, D2),
        ExpertAdmissionError::InputSuperseded {
            role: ExpertInputRole::Evaluation,
            ..
        }
    ));
    assert_eq!(
        attempt(&mut rt, 2, D2),
        ExpertAdmissionError::InputDigestStale {
            role: ExpertInputRole::Evaluation,
            identity: E.into(),
            generation: Generation(2),
        }
    );
    birth_with(
        &mut rt,
        token_spec("n", 1),
        Some(inputs(&[], &[(E, 2, D3)])),
    );
}

#[test]
fn the_first_committed_use_fixes_a_digest_and_nothing_else_moves_it() {
    let mut rt = runtime();
    born(&mut rt, "x", inputs(&[(T, 1, D1)], &[]));
    let before = history(&rt);
    // Training and evaluation roles are held to the same digest.
    for (training, evaluation, role) in [
        (&[(T, 1, D2)][..], &[][..], ExpertInputRole::Training),
        (&[][..], &[(T, 1, D2)][..], ExpertInputRole::Evaluation),
    ] {
        assert_eq!(
            expert_error(admit(
                &mut rt,
                &propose(StructuralOp::Birth {
                    result: token_spec("n", 1),
                }),
                Some(inputs(training, evaluation)),
            )),
            ExpertAdmissionError::InputDigestStale {
                role,
                identity: T.into(),
                generation: Generation(1),
            }
        );
    }
    assert_eq!(history(&rt), before);
    // The same digest is accepted again.
    born(&mut rt, "m", inputs(&[(T, 1, D1)], &[]));
    // The base model is a dependency with its own registered digest: an explicit input that
    // collides with the derived pair under another digest is a stale digest.
    assert_eq!(
        expert_error(admit(
            &mut rt,
            &propose(StructuralOp::Birth {
                result: token_spec("k", 1),
            }),
            Some(inputs(&[("base-model:base-a", 1, D1)], &[])),
        )),
        ExpertAdmissionError::InputDigestStale {
            role: ExpertInputRole::Training,
            identity: "base-model:base-a".into(),
            generation: Generation(1),
        }
    );
}

// ---- wrong-generation and malformed invalidations ---------------------------------------------

#[test]
fn an_invalidation_naming_the_wrong_generation_is_refused_and_appends_nothing() {
    let mut rt = populated();
    rt.commit(LedgerEvent::CapsuleCommitted {
        project: project(),
        capsule: CapsuleId::from("cap"),
        generation: Generation(1),
    })
    .unwrap();
    born(&mut rt, "c", inputs(&[("cap", 1, D1)], &[]));
    let before = history(&rt);
    let registry = rt.expert_registry().clone();
    let refuse = |rt: &mut PtrRuntime, request: InputInvalidation| {
        let error = expert_error(invalidate(rt, request));
        assert_eq!(history(rt), before);
        assert_eq!(rt.expert_registry(), &registry);
        error
    };
    // g+1 is not committed in the lifecycle authority yet.
    assert_eq!(
        refuse(&mut rt, supersede("cap", 1, D1, 2, D2)),
        ExpertAdmissionError::InputGenerationMismatch {
            identity: "cap".into(),
            generation: Generation(2),
            live: Generation(1),
        }
    );
    // A revocation ahead of the live generation.
    assert_eq!(
        refuse(&mut rt, revoke("cap", 5, D1)),
        ExpertAdmissionError::InputGenerationMismatch {
            identity: "cap".into(),
            generation: Generation(5),
            live: Generation(1),
        }
    );
    // A supersession whose binding digest is other than the one the first use fixed. (A
    // revocation does not depend on the asserted digest; see
    // `a_revocation_does_not_depend_on_the_digest_a_proposer_asserted`.)
    assert_eq!(
        refuse(&mut rt, supersede(T, 1, D3, 2, D2)),
        ExpertAdmissionError::InputBindingMismatch {
            identity: T.into(),
            generation: Generation(1),
        }
    );
    // A supersession whose new generation would break the digest already fixed for it.
    born(&mut rt, "later", inputs(&[(T, 3, D3)], &[]));
    let before_later = history(&rt);
    let error = expert_error(invalidate(&mut rt, supersede(T, 1, D1, 3, D2)));
    assert_eq!(
        error,
        ExpertAdmissionError::InputBindingMismatch {
            identity: T.into(),
            generation: Generation(3),
        }
    );
    assert_eq!(history(&rt), before_later);
    // Positive control: with the generation committed, the supersession goes through.
    rt.commit(LedgerEvent::CapsuleSuperseded {
        capsule: CapsuleId::from("cap"),
        old: Generation(1),
        new: Generation(2),
    })
    .unwrap();
    let outcome = invalidate(&mut rt, supersede("cap", 1, D1, 2, D2)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["c"]));
}

#[test]
fn a_repeated_or_backward_invalidation_is_refused() {
    let mut rt = populated();
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    assert_eq!(
        expert_error(invalidate(&mut rt, revoke(T, 1, D1))),
        ExpertAdmissionError::InputAlreadyInvalidated {
            identity: T.into(),
            generation: Generation(1),
        }
    );
    invalidate(&mut rt, supersede(U, 1, D3, 4, D1)).unwrap();
    for new in [4, 3, 2] {
        assert_eq!(
            expert_error(invalidate(&mut rt, supersede(U, 1, D3, new, D2))),
            ExpertAdmissionError::StaleSupersession {
                identity: U.into(),
                floor: Generation(4),
                new_generation: Generation(new),
            }
        );
    }
    // Forward is fine.
    invalidate(&mut rt, supersede(U, 4, D1, 5, D2)).unwrap();
}

#[test]
fn the_grant_is_checked_like_any_runtime_fence() {
    let mut rt = populated();
    let before = history(&rt);
    // A gate grant never invalidates.
    let gate_grant = ExpertFenceGrant::gate(
        project(),
        requirement(7, gate().config_digest().unwrap()),
        binding(),
    );
    assert_eq!(
        expert_error(rt.invalidate_input(&gate_grant, revoke(T, 1, D1))),
        ExpertAdmissionError::FenceAuthorityMismatch {
            action: ExpertFenceAction::Fence,
            required: ExpertFenceAuthority::GlobalRevocation,
        }
    );
    // A grant minted before its generation was superseded is stale.
    let stale = global_grant(&rt);
    rt.supersede_expert_grant(ExpertGrantKind::GlobalRevocation)
        .unwrap();
    assert!(matches!(
        expert_error(rt.invalidate_input(&stale, revoke(T, 1, D1))),
        ExpertAdmissionError::GrantGenerationStale {
            kind: ExpertGrantKind::GlobalRevocation,
            ..
        }
    ));
    assert_eq!(history(&rt), before);
    // The current grant works.
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
}

#[test]
fn the_public_commit_refuses_the_event() {
    let mut rt = populated();
    let before = history(&rt);
    let event = LedgerEvent::InputInvalidated {
        input: T.to_owned(),
        generation: Generation(1),
        binding_digest: D1,
        kind: InputInvalidationKind::Revoked,
        fence_count: 2,
        fence_digest: [5; 32],
    };
    assert!(matches!(
        rt.commit(event),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
    assert_eq!(history(&rt), before);
}

// ---- transitive closure -----------------------------------------------------------------------

/// X names A; C names nothing. B = Merge(A, C) (a version 2 record, no inputs); D1, D2 =
/// Split(B). Only the descendants are live.
fn lineage() -> PtrRuntime {
    let mut rt = runtime();
    born(&mut rt, "a", inputs(&[("data:x", 1, D1)], &[]));
    birth_with(&mut rt, token_spec("c", 1), None);
    birth_with(&mut rt, token_spec("u", 1), None);
    let merge = propose(StructuralOp::Merge {
        sources: vec![target_of(&rt, "a"), target_of(&rt, "c")],
        result: token_spec("b", 1),
    });
    admit(&mut rt, &merge, None).unwrap();
    let split = propose(StructuralOp::Split {
        source: target_of(&rt, "b"),
        results: vec![token_spec("d", 1), token_spec("e", 1)],
    });
    admit(&mut rt, &split, None).unwrap();
    rt
}

#[test]
fn descendants_of_a_dependent_are_fenced_transitively() {
    let mut rt = lineage();
    for pod in ["d", "e", "u"] {
        assert_eq!(route(&rt, pod, 1), Ok(()));
    }
    // The consumed experts are not live; the Merge/Split results and nothing else are fenced.
    assert_eq!(state_of(&rt, "a").0, ExpertState::Superseded);
    let before = rt.committed_events().len();
    let outcome = invalidate(&mut rt, revoke("data:x", 1, D1)).unwrap();
    assert_eq!(rt.committed_events().len(), before + 1);
    assert_eq!(fenced_pods(&outcome), pods(&["d", "e"]));
    for pod in ["d", "e"] {
        assert_eq!(route(&rt, pod, 1), Err(RouteRefusal::Fenced), "{pod}");
    }
    assert_eq!(route(&rt, "u", 1), Ok(()));
    assert_eq!(state_of(&rt, "a").0, ExpertState::Superseded);
    // A supersession reaches the same descendants.
    let mut rt2 = lineage();
    let outcome = invalidate(&mut rt2, supersede("data:x", 1, D1, 2, D2)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["d", "e"]));
}

#[test]
fn a_reactivated_generation_is_a_descendant_of_its_seed() {
    let mut rt = runtime();
    born(&mut rt, "a", inputs(&[("data:x", 1, D1)], &[]));
    let retire = propose(StructuralOp::Retire {
        target: target_of(&rt, "a"),
        reason: RetiredReason::GateFailed,
    });
    admit(&mut rt, &retire, None).unwrap();
    let reactivate = propose(StructuralOp::Reactivate {
        target: target_of(&rt, "a"),
        result: Some(token_spec("a", 2)),
    });
    admit(&mut rt, &reactivate, None).unwrap();
    assert_eq!(route(&rt, "a", 2), Ok(()));
    // Generation 1 is retired and cannot be fenced; generation 2 inherits its input.
    let outcome = invalidate(&mut rt, revoke("data:x", 1, D1)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["a"]));
    assert_eq!(outcome.fenced()[0].generation, Generation(2));
    assert_eq!(route(&rt, "a", 2), Err(RouteRefusal::Fenced));
}

#[test]
fn a_successor_retrained_on_the_replacement_is_not_fenced_by_its_own_supersession() {
    // A names X@1. It fails its gate and is reactivated as generation 2, retrained on X@2.
    let build = || {
        let mut rt = runtime();
        born(&mut rt, "a", inputs(&[("data:x", 1, D1)], &[]));
        let retire = propose(StructuralOp::Retire {
            target: target_of(&rt, "a"),
            reason: RetiredReason::GateFailed,
        });
        admit(&mut rt, &retire, None).unwrap();
        let reactivate = propose(StructuralOp::Reactivate {
            target: target_of(&rt, "a"),
            result: Some(token_spec("a", 2)),
        });
        admit(
            &mut rt,
            &reactivate,
            Some(inputs(&[("data:x", 2, D2)], &[])),
        )
        .unwrap();
        rt
    };
    // Superseding X from 1 to 2: the successor names X@2 itself, so it is not fenced.
    let mut rt = build();
    let outcome = invalidate(&mut rt, supersede("data:x", 1, D1, 2, D2)).unwrap();
    assert_eq!(outcome.fence_count(), 0);
    assert_eq!(route(&rt, "a", 2), Ok(()));
    // Revoking X@1 fences it: a revocation has no such exception.
    let mut rt = build();
    let outcome = invalidate(&mut rt, revoke("data:x", 1, D1)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["a"]));
}

// ---- base model --------------------------------------------------------------------------------

#[test]
fn the_base_model_is_a_dependency_of_every_version_3_expert() {
    let mut rt = populated();
    // z..x rest on base-a through their records; w (version 2) does not.
    let outcome = invalidate(&mut rt, revoke("base-model:base-a", 1, [9; 32])).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["x", "y", "z"]));
    assert_eq!(route(&rt, "w", 1), Ok(()));
    assert_eq!(
        expert_error(admit(
            &mut rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[], &[])),
        )),
        ExpertAdmissionError::InputRevoked {
            role: ExpertInputRole::BaseModel,
            identity: "base-model:base-a".into(),
            generation: Generation(1),
        }
    );
    // A supersession whose binding digest is other than the base model's is refused.
    let mut other = populated();
    assert_eq!(
        expert_error(invalidate(
            &mut other,
            supersede("base-model:base-a", 1, D1, 2, D2)
        )),
        ExpertAdmissionError::InputBindingMismatch {
            identity: "base-model:base-a".into(),
            generation: Generation(1),
        }
    );
}

// ---- readmission -------------------------------------------------------------------------------

#[test]
fn a_quarantined_dependent_is_escalated_and_never_readmitted() {
    let mut rt = populated();
    let grant = revocation_grant(&rt);
    rt.fence_expert(
        &ExpertFenceRequest {
            project: project(),
            pod: PodId::from("x"),
            generation: Generation(1),
            expected_state_version: 1,
            cause: ExpertFenceCause::AuthorityQuarantine,
            revocation: None,
        },
        &grant,
    )
    .unwrap();
    assert_eq!(state_of(&rt, "x"), (ExpertState::Fenced, 2));
    let outcome = invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    // x is escalated (still fenced, new epoch, final cause), y is fenced.
    assert_eq!(fenced_pods(&outcome), pods(&["x", "y"]));
    assert_eq!(outcome.fenced()[0].action, ExpertFenceAction::Escalate);
    assert_eq!(outcome.fenced()[1].action, ExpertFenceAction::Fence);
    assert_eq!(state_of(&rt, "x"), (ExpertState::Fenced, 3));
    assert_eq!(
        rt.expert_registry()
            .fence_cause(&project(), &PodId::from("x")),
        Some(ExpertFenceCause::InputRevoked)
    );
    // A second invalidation does not touch a final fence.
    let again = invalidate(&mut rt, revoke(E, 1, D2)).unwrap();
    assert_eq!(again.fence_count(), 0);
}

#[test]
fn readmission_is_refused_while_a_recorded_input_is_not_live() {
    let mut rt = populated();
    let grant = revocation_grant(&rt);
    rt.fence_expert(
        &ExpertFenceRequest {
            project: project(),
            pod: PodId::from("z"),
            generation: Generation(1),
            expected_state_version: 1,
            cause: ExpertFenceCause::AuthorityQuarantine,
            revocation: None,
        },
        &grant,
    )
    .unwrap();
    // U is revoked by a plain record while z is quarantined.
    rt.commit(LedgerEvent::Revoked {
        subject: U.to_owned(),
        generation: Generation(1),
    })
    .unwrap();
    let report = {
        let (proposal_id, operation_id) = ptr_runtime::expert::expert_readmission_subject_ids(
            &project(),
            &PodId::from("z"),
            Generation(1),
            2,
            3,
            ExpertFenceCause::AuthorityQuarantine,
        );
        let entry = rt
            .expert_registry()
            .get(&project(), &PodId::from("z"))
            .unwrap()
            .clone();
        let subject = GateSubject::new(
            proposal_id,
            operation_id,
            *entry.manifest_digest.as_bytes(),
            *entry.base_model.digest().as_bytes(),
            EVAL_MANIFEST,
        );
        let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
        gate()
            .evaluate(
                &AdapterId::from("candidate"),
                &matrix,
                PublicSuite {
                    serving: 0.8,
                    candidate: 0.8,
                },
            )
            .unwrap()
            .bind(subject)
    };
    let gate_grant = ExpertFenceGrant::gate(
        project(),
        requirement(7, gate().config_digest().unwrap()),
        current_binding(&rt, ExpertGrantKind::Gate),
    );
    let before = history(&rt);
    assert_eq!(
        expert_error(rt.readmit_expert(
            &ExpertFenceRequest {
                project: project(),
                pod: PodId::from("z"),
                generation: Generation(1),
                expected_state_version: 2,
                cause: ExpertFenceCause::AuthorityQuarantine,
                revocation: None,
            },
            &report,
            &gate_grant,
        )),
        ExpertAdmissionError::DependencyNotLive {
            pod: "z".into(),
            identity: U.into(),
            generation: Generation(1),
        }
    );
    assert_eq!(history(&rt), before);
}

// ---- fan-out, replay, durability ---------------------------------------------------------------

#[test]
fn a_large_fan_out_is_one_event_with_a_count_and_a_digest() {
    const N: usize = 2_000;
    let mut rt = runtime();
    for i in 0..N {
        birth_with(
            &mut rt,
            segment_spec(&format!("e{i:05}"), 1),
            Some(inputs(&[("data:big", 1, D1)], &[])),
        );
    }
    birth_with(
        &mut rt,
        segment_spec("bystander", 1),
        Some(inputs(&[("data:other", 1, D2)], &[])),
    );
    let before = rt.committed_events().len();
    let outcome = invalidate(&mut rt, revoke("data:big", 1, D1)).unwrap();
    assert_eq!(rt.committed_events().len(), before + 1);
    assert_eq!(outcome.fence_count() as usize, N);
    let LedgerEvent::InputInvalidated {
        fence_count,
        fence_digest,
        ..
    } = &rt.committed_events().last().unwrap().event
    else {
        panic!("not an input invalidation");
    };
    assert_eq!(*fence_count as usize, N);
    assert_eq!(
        *fence_digest,
        expert_fence_set_digest("data:big", Generation(1), outcome.fenced())
    );
    // The set is in canonical order and every member is fenced and unroutable.
    let names = fenced_pods(&outcome);
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    for pod in &names {
        assert_eq!(state_of(&rt, pod), (ExpertState::Fenced, 2));
    }
    assert_eq!(route(&rt, "e00000", 1), Err(RouteRefusal::Fenced));
    assert_eq!(route(&rt, "bystander", 1), Ok(()));
    // Replay re-derives the same set from the index.
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_eq!(route(&replayed, "e01999", 1), Err(RouteRefusal::Fenced));
}

#[test]
fn replay_refuses_a_tampered_fence_count_or_digest() {
    let mut rt = populated();
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    let events = history(&rt);
    let last = events.len() - 1;
    // The honest history replays.
    let replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_dependents_fenced_others_serving(&replayed);
    let tamper = |edit: &dyn Fn(&mut u32, &mut [u8; 32])| {
        let mut events = events.clone();
        let LedgerEvent::InputInvalidated {
            fence_count,
            fence_digest,
            ..
        } = &mut events[last].event
        else {
            panic!("not an input invalidation");
        };
        edit(fence_count, fence_digest);
        PtrRuntime::replay(PtrConfig::default(), &events)
    };
    for (edit, derived, found, digest_matches) in [
        (
            (&|count: &mut u32, _: &mut [u8; 32]| *count += 1) as &dyn Fn(&mut u32, &mut [u8; 32]),
            2,
            3,
            true,
        ),
        (&|count: &mut u32, _: &mut [u8; 32]| *count = 0, 2, 0, true),
        (
            &|_: &mut u32, digest: &mut [u8; 32]| digest[0] ^= 1,
            2,
            2,
            false,
        ),
    ] {
        match tamper(edit) {
            Err(RuntimeError::Expert(ExpertAdmissionError::FenceSetMismatch {
                derived_count,
                event_count,
                digest_matches: matches,
            })) => {
                assert_eq!(
                    (derived_count, event_count, matches),
                    (derived, found, digest_matches)
                );
            }
            Err(other) => panic!("wrong refusal {other:?}"),
            Ok(_) => panic!("a tampered fence set replayed"),
        }
    }
    // An event that claims a set where the index derives none (an invalidation of an input
    // nothing rests on) is refused too.
    let mut forged = events.clone();
    if let LedgerEvent::InputInvalidated {
        input, fence_count, ..
    } = &mut forged[last].event
    {
        *input = "dataset:nobody".to_owned();
        assert_eq!(*fence_count, 2);
    }
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &forged),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::InputBindingMismatch { .. }
                | ExpertAdmissionError::FenceSetMismatch { .. }
        ))
    ));
}

#[test]
fn the_invalidation_survives_a_durable_reopen_and_no_dependent_is_routable_after_a_crash() {
    let (dir, path) = temp_ledger("durable");
    {
        let mut rt = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        prime(&mut rt);
        born(&mut rt, "x", inputs(&[(T, 1, D1)], &[]));
        born(&mut rt, "y", inputs(&[(T, 1, D1)], &[(E, 1, D2)]));
        born(&mut rt, "z", inputs(&[(U, 1, D3)], &[]));
        birth_with(&mut rt, token_spec("w", 1), None);
        invalidate(&mut rt, supersede(T, 1, D1, 2, D3)).unwrap();
        assert_dependents_fenced_others_serving(&rt);
        // Dropped here: the event is durable, the process "crashes" before anything else.
    }
    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_dependents_fenced_others_serving(&reopened);
    let index = reopened.expert_registry().dependencies();
    assert_eq!(index.supersession_floor(T), Some(Generation(2)));
    assert_eq!(index.registered_digest(T, Generation(2)), Some(D3));
    // The refusals survive too.
    assert!(matches!(
        expert_error(admit(
            &mut reopened,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[(T, 1, D1)], &[])),
        )),
        ExpertAdmissionError::InputSuperseded { .. }
    ));
    drop(reopened);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_empty_fence_set_is_a_valid_event() {
    let mut rt = populated();
    let outcome = invalidate(&mut rt, revoke("dataset:nobody", 1, D1)).unwrap();
    assert_eq!(outcome.fence_count(), 0);
    let LedgerEvent::InputInvalidated {
        fence_count,
        fence_digest,
        ..
    } = &rt.committed_events().last().unwrap().event
    else {
        panic!("not an input invalidation");
    };
    assert_eq!(*fence_count, 0);
    assert_eq!(
        *fence_digest,
        expert_fence_set_digest("dataset:nobody", Generation(1), &[])
    );
    // Nothing was fenced, but the input is revoked from here on.
    for pod in ["x", "y", "z", "w"] {
        assert_eq!(route(&rt, pod, 1), Ok(()));
    }
    assert!(matches!(
        expert_error(admit(
            &mut rt,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[("dataset:nobody", 1, D1)], &[])),
        )),
        ExpertAdmissionError::InputRevoked { .. }
    ));
}

#[test]
fn inputs_on_an_operation_without_a_result_are_refused() {
    let mut rt = populated();
    let freeze = propose(StructuralOp::Freeze {
        target: target_of(&rt, "z"),
    });
    let before = history(&rt);
    assert!(matches!(
        expert_error(admit(&mut rt, &freeze, Some(inputs(&[(U, 1, D3)], &[])))),
        ExpertAdmissionError::InvalidRecord(_)
    ));
    assert_eq!(history(&rt), before);
}

// ---- the KV catalog and the compaction snapshot ----------------------------------------------------

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
fn a_compound_invalidation_kills_the_kv_entries_of_the_fenced_experts_only() {
    for supersede_instead in [false, true] {
        let mut rt = populated();
        let rig = Rig::new(&mut rt);
        rig.insert(&mut rt, vec![expert_dep("x", 1)]).unwrap();
        assert!(hits(&mut rt), "an entry on x is live before");
        let request = if supersede_instead {
            supersede(T, 1, D1, 2, D3)
        } else {
            revoke(T, 1, D1)
        };
        invalidate(&mut rt, request).unwrap();
        assert!(
            !hits(&mut rt),
            "the entry recorded under x's epoch 1 is dead"
        );
        // No dependency is live at any epoch up to and including the fence.
        for epoch in [0, 1, 2] {
            assert_eq!(
                rig.insert(&mut rt, vec![expert_dep("x", epoch)]),
                Err(KvCatalogError::DependencyNotLive),
                "epoch {epoch}"
            );
        }
        // A bystander (z rests on another input) keeps its entries live.
        rig.insert(&mut rt, vec![expert_dep("z", 1)]).unwrap();
        assert!(hits(&mut rt));
        // The same mapping applies on replay of the log.
        let mut replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
        let rig = Rig::new(&mut replayed);
        assert_eq!(
            rig.insert(&mut replayed, vec![expert_dep("y", 1)]),
            Err(KvCatalogError::DependencyNotLive)
        );
        rig.insert(&mut replayed, vec![expert_dep("w", 1)]).unwrap();
    }
}

#[test]
fn a_standalone_kv_backend_is_handed_the_fence_set_the_event_does_not_carry() {
    let mut rt = populated();
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    let outcome_event = rt.committed_events().last().unwrap().event.clone();
    let fences = BTreeMap::from([
        ("expert:proj/x".to_owned(), 2),
        ("expert:proj/y".to_owned(), 2),
    ]);

    let scope = ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: project(),
    };
    let backend = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope).unwrap();
    // The event names no experts: the plain hook refuses it rather than leave them live.
    assert!(matches!(
        backend.apply_committed_event(&outcome_event),
        Err(ReferenceKvError::InputInvalidationNeedsFences)
    ));
    assert!(matches!(
        backend.apply_input_invalidation(&rt.committed_events()[0].event, &fences),
        Err(ReferenceKvError::NotAnInputInvalidation)
    ));
    let live = backend.own_liveness();
    let input_dep = KvDependency::base_model(T, Generation(1), [0; 32]);
    assert!(live.is_live(&input_dep));
    assert!(live.is_live(&expert_dep("x", 1)));
    backend
        .apply_input_invalidation(&outcome_event, &fences)
        .unwrap();
    assert!(!live.is_live(&input_dep), "the revoked pair is dead");
    assert!(!live.is_live(&expert_dep("x", 1)));
    assert!(live.is_live(&expert_dep("x", 2)));
    assert!(live.is_live(&expert_dep("z", 1)));

    // A supersession raises the floor of the identity.
    let mut rt = populated();
    invalidate(&mut rt, supersede(T, 1, D1, 2, D3)).unwrap();
    let event = rt.committed_events().last().unwrap().event.clone();
    let scope = ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: project(),
    };
    let backend = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope).unwrap();
    backend.apply_input_invalidation(&event, &fences).unwrap();
    let live = backend.own_liveness();
    assert!(!live.is_live(&KvDependency::base_model(T, Generation(1), [0; 32])));
    assert!(live.is_live(&KvDependency::base_model(T, Generation(2), [0; 32])));
}

#[test]
fn compaction_carries_the_index_the_digests_and_the_invalidations() {
    let mut rt = lineage();
    born(&mut rt, "s", inputs(&[("data:s", 1, D2)], &[]));
    invalidate(&mut rt, supersede("data:s", 1, D2, 2, D3)).unwrap();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS008");
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_eq!(
        restored.expert_registry().dependencies(),
        rt.expert_registry().dependencies()
    );
    assert_eq!(route(&restored, "s", 1), Err(RouteRefusal::Fenced));
    // The digest registry survived: a stale digest and a stale generation are still refused.
    assert_eq!(
        expert_error(admit(
            &mut restored,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[("data:s", 2, D2)], &[])),
        )),
        ExpertAdmissionError::InputDigestStale {
            role: ExpertInputRole::Training,
            identity: "data:s".into(),
            generation: Generation(2),
        }
    );
    assert!(matches!(
        expert_error(admit(
            &mut restored,
            &propose(StructuralOp::Birth {
                result: token_spec("n", 1),
            }),
            Some(inputs(&[("data:s", 1, D2)], &[])),
        )),
        ExpertAdmissionError::InputSuperseded { .. }
    ));
    // The source edges survived: the transitive closure still reaches the Split results.
    let outcome = invalidate(&mut restored, revoke("data:x", 1, D1)).unwrap();
    assert_eq!(fenced_pods(&outcome), pods(&["d", "e"]));
    assert_eq!(route(&restored, "u", 1), Ok(()));
}

#[test]
fn compaction_keeps_a_revocation_and_replay_above_the_floor_works() {
    let mut rt = populated();
    let floor = rt.export_compacted_snapshot().unwrap();
    // The events above the floor: the compound revocation, then more history.
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    let tail = rt.committed_events()[rt.committed_events().len() - 1..].to_vec();
    let resumed =
        PtrRuntime::restore_compacted(PtrConfig::default(), floor.bytes(), floor.anchor(), &tail)
            .unwrap();
    assert_eq!(resumed.expert_registry(), rt.expert_registry());
    assert_dependents_fenced_others_serving(&resumed);
    // A snapshot after the revocation carries it.
    let after = rt.export_compacted_snapshot().unwrap();
    let mut restored =
        PtrRuntime::restore_compacted(PtrConfig::default(), after.bytes(), after.anchor(), &[])
            .unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_dependents_fenced_others_serving(&restored);
    assert_eq!(
        restored.generation_validity(T, Generation(1)),
        Some(ptr_types::Validity::Revoked)
    );
    assert_eq!(
        expert_error(invalidate(&mut restored, revoke(T, 1, D1))),
        ExpertAdmissionError::InputAlreadyInvalidated {
            identity: T.into(),
            generation: Generation(1),
        }
    );
    // A tampered tail event is refused on top of the restore too.
    let mut forged = tail;
    if let LedgerEvent::InputInvalidated { fence_count, .. } = &mut forged[0].event {
        *fence_count = 7;
    }
    assert!(matches!(
        PtrRuntime::restore_compacted(PtrConfig::default(), floor.bytes(), floor.anchor(), &forged),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceSetMismatch { .. }
        ))
    ));
}

#[test]
fn a_history_without_inputs_keeps_the_earlier_snapshot_layout() {
    let mut rt = runtime();
    birth_with(&mut rt, token_spec("w", 1), None);
    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS006");
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert!(restored.expert_registry().dependencies().is_empty());
    assert_eq!(restored.expert_registry(), rt.expert_registry());
}
