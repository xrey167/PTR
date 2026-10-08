//! Revocation-driven retirement (fence action `Retire`): a `Fenced` expert of cause
//! `InputRevoked` is retired for good under the runtime revocation grant and a committed
//! revocation, never by a gate. The retired state is terminal, survives replay, a durable reopen
//! and compaction, is dead for the KV catalog, and every other path to it is refused typed with
//! nothing appended.

use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_ledger::{
    CommittedEvent, ExpertFenceAction, ExpertFenceCause, ExpertFenceEvidence, ExpertFenceResume,
    ExpertFenceRevocation, LedgerEvent,
};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertProposalError, ExpertState, ExpertTarget, ExpertTargetInit, InMemoryPagedKvBackend,
    KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType, KvTensorSchema, ManifestDigest,
    PagedKvTensorBackend, ProposerRef, RationaleDigest, RetiredReason, StructuralOp, TensorRef,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, ExpertAdmissionError,
    ExpertAdmissionGrant, ExpertFenceAuthority, ExpertFenceGrant, ExpertFenceRequest,
    ExpertGrantKind, ExpertStaleField, RouteRefusal,
};
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvChainId, KvDependency, KvInsert, KvLookup, KvPrefixCatalog,
    KvRequestPositions, KvSalt, PtrRuntime, RuntimeError,
};
use ptr_types::{
    AdapterVersion, CommitIndex, DeviceId, Generation, ModelVersion, PodId, PrincipalId, ProjectId,
    RequestId, Revision, SemanticContext,
};

mod expert_support;

use expert_support::{binding, current_binding, prime, requirement, EVAL_MANIFEST};

const CHAIN: KvChainId = KvChainId([1; 32]);
const QUARANTINE: ExpertFenceCause = ExpertFenceCause::AuthorityQuarantine;
const REVOKED: ExpertFenceCause = ExpertFenceCause::InputRevoked;

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

fn propose(op: StructuralOp) -> Result<ExpertProposal, ExpertProposalError> {
    ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
}

/// Admit a proposal through the gate path with a passing, correctly bound report.
fn try_admit(rt: &mut PtrRuntime, proposal: &ExpertProposal) -> Result<CommitIndex, RuntimeError> {
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let report: BoundGateReport = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject);
    prime(rt);
    let grant = ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        current_binding(rt, ExpertGrantKind::Admission),
    );
    rt.admit_expert_proposal(proposal, &report, &grant)
}

fn birth(rt: &mut PtrRuntime, pod: &str) {
    let spec = ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new([pod.as_bytes()[0]; 32]).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(pod.as_bytes()[0]),
        },
        base(),
        Generation(1),
    )
    .unwrap();
    let proposal = propose(StructuralOp::Birth { result: spec }).unwrap();
    try_admit(rt, &proposal).unwrap();
}

fn entry_state(rt: &PtrRuntime, pod: &str) -> (ExpertState, u64) {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap();
    (entry.state, entry.state_version)
}

fn target_as(rt: &PtrRuntime, pod: &str, state: ExpertState, version: u64) -> ExpertTarget {
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
        state,
        state_version: version,
    })
    .unwrap()
}

fn committed_revocation() -> ExpertFenceRevocation {
    ExpertFenceRevocation {
        subject: "k".to_owned(),
        generation: Generation(1),
    }
}

fn request(pod: &str, version: u64, cause: ExpertFenceCause) -> ExpertFenceRequest {
    ExpertFenceRequest {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        cause,
        revocation: (cause == REVOKED).then(committed_revocation),
    }
}

fn revocation_grant(rt: &PtrRuntime) -> ExpertFenceGrant {
    ExpertFenceGrant::revocation(project(), current_binding(rt, ExpertGrantKind::Fence))
}

fn gate_grant() -> ExpertFenceGrant {
    ExpertFenceGrant::gate(
        project(),
        requirement(7, gate().config_digest().unwrap()),
        binding(),
    )
}

/// A runtime that holds the committed revocation of `("k", 1)`.
fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    revoke(&mut rt);
    rt
}

fn revoke(rt: &mut PtrRuntime) {
    let revocation = committed_revocation();
    rt.commit(LedgerEvent::Revoked {
        subject: revocation.subject,
        generation: revocation.generation,
    })
    .unwrap();
}

fn fence(rt: &mut PtrRuntime, pod: &str, version: u64, cause: ExpertFenceCause) {
    let grant = revocation_grant(rt);
    rt.fence_expert(&request(pod, version, cause), &grant)
        .unwrap();
}

fn retire(rt: &mut PtrRuntime, request: &ExpertFenceRequest) -> Result<CommitIndex, RuntimeError> {
    prime(rt);
    let grant = revocation_grant(rt);
    rt.retire_revoked_expert(request, &grant)
}

/// `x` born, fenced for a revoked input at epoch 2 and retired at epoch 3.
fn retired_runtime() -> PtrRuntime {
    let mut rt = runtime();
    birth(&mut rt, "x");
    fence(&mut rt, "x", 1, REVOKED);
    retire(&mut rt, &request("x", 2, REVOKED)).unwrap();
    rt
}

fn expert_error(result: Result<CommitIndex, RuntimeError>) -> ExpertAdmissionError {
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

fn retired() -> ExpertState {
    ExpertState::Retired(RetiredReason::InputRevoked)
}

fn assert_retired_and_unroutable(rt: &PtrRuntime) {
    assert_eq!(entry_state(rt, "x"), (retired(), 3));
    assert_eq!(
        rt.expert_route_admission(&project(), &PodId::from("x"), Generation(1)),
        Err(RouteRefusal::NotServing { state: retired() })
    );
    assert_eq!(
        rt.materialized_state().values.get("expert:proj/x"),
        Some(&"1:retired-input-revoked:3".to_owned())
    );
    // The fence is cleared: a retired expert is under no fence cause.
    assert_eq!(
        rt.expert_registry()
            .fence_cause(&project(), &PodId::from("x")),
        None
    );
}

#[test]
fn a_fenced_revoked_expert_is_retired_under_the_runtime_grant_and_stays_unroutable() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    assert_eq!(
        rt.expert_route_admission(&project(), &PodId::from("x"), Generation(1)),
        Ok(())
    );
    fence(&mut rt, "x", 1, REVOKED);
    assert_eq!(
        rt.expert_route_admission(&project(), &PodId::from("x"), Generation(1)),
        Err(RouteRefusal::Fenced)
    );
    let before = rt.committed_events().len();
    // Positive control: the runtime grant, the fence's cause and its committed revocation.
    retire(&mut rt, &request("x", 2, REVOKED)).unwrap();
    assert_eq!(rt.committed_events().len(), before + 1);
    assert!(matches!(
        rt.committed_events().last().unwrap().event,
        LedgerEvent::ExpertFenceCommitted {
            action: ExpertFenceAction::Retire,
            cause: ExpertFenceCause::InputRevoked,
            epoch: 3,
            evidence: None,
            ..
        }
    ));
    assert_retired_and_unroutable(&rt);
}

#[test]
fn a_frozen_expert_and_a_procedure_revocation_retire_too() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    let target = target_as(&rt, "x", ExpertState::Serving, 1);
    let freeze = propose(StructuralOp::Freeze { target }).unwrap();
    try_admit(&mut rt, &freeze).unwrap();
    rt.commit(LedgerEvent::ProcedureRevoked {
        id: "proc".to_owned(),
        generation: Generation(4),
    })
    .unwrap();
    let procedure = ExpertFenceRevocation {
        subject: "procedure:proc".to_owned(),
        generation: Generation(4),
    };
    let grant = revocation_grant(&rt);
    let fence_request = ExpertFenceRequest {
        revocation: Some(procedure.clone()),
        ..request("x", 2, REVOKED)
    };
    rt.fence_expert(&fence_request, &grant).unwrap();
    let retire_request = ExpertFenceRequest {
        expected_state_version: 3,
        ..fence_request
    };
    retire(&mut rt, &retire_request).unwrap();
    assert_eq!(entry_state(&rt, "x"), (retired(), 4));
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
}

#[test]
fn the_retirement_survives_replay_a_durable_reopen_and_compaction() {
    let rt = retired_runtime();
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_retired_and_unroutable(&replayed);

    // Durable reopen.
    let dir = std::env::temp_dir().join(format!(
        "ptr-expert-retire-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("ledger");
    let mut durable = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    prime(&mut durable);
    revoke(&mut durable);
    birth(&mut durable, "x");
    fence(&mut durable, "x", 1, REVOKED);
    retire(&mut durable, &request("x", 2, REVOKED)).unwrap();
    assert_retired_and_unroutable(&durable);
    drop(durable);
    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(reopened.expert_registry(), rt.expert_registry());
    assert_retired_and_unroutable(&reopened);
    drop(reopened);
    std::fs::remove_dir_all(&dir).unwrap();

    // Compaction: the snapshot carries the retired state, the fence memo and no fence cause.
    let snapshot = rt.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS007");
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_retired_and_unroutable(&restored);
    assert_eq!(
        restored
            .expert_registry()
            .highest_fenced_epoch("expert:proj/x"),
        Some(2)
    );

    // A retirement above a floor that holds the fence replays on top of the restore.
    let mut above = runtime();
    birth(&mut above, "x");
    fence(&mut above, "x", 1, REVOKED);
    let floor = above.export_compacted_snapshot().unwrap();
    retire(&mut above, &request("x", 2, REVOKED)).unwrap();
    let tail = above.committed_events()[above.committed_events().len() - 1..].to_vec();
    let resumed =
        PtrRuntime::restore_compacted(PtrConfig::default(), floor.bytes(), floor.anchor(), &tail)
            .unwrap();
    assert_eq!(resumed.expert_registry(), rt.expert_registry());
    assert_retired_and_unroutable(&resumed);
    // The restored runtime keeps the terminal rules.
    let mut resumed = resumed;
    assert_eq!(
        expert_error(retire(&mut resumed, &request("x", 3, REVOKED))),
        ExpertAdmissionError::NotFenced {
            pod: "x".into(),
            state: retired(),
        }
    );
}

#[test]
fn every_other_path_to_a_revocation_retirement_is_refused_and_appends_nothing() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    birth(&mut rt, "q");
    birth(&mut rt, "w");
    fence(&mut rt, "q", 1, QUARANTINE);
    fence(&mut rt, "w", 1, REVOKED);
    retire(&mut rt, &request("w", 2, REVOKED)).unwrap();
    let before = applied(&rt);
    let registry = rt.expert_registry().clone();
    let refuse = |rt: &mut PtrRuntime, request: &ExpertFenceRequest| {
        let error = expert_error(retire(rt, request));
        assert_eq!(applied(rt), before);
        assert_eq!(rt.expert_registry(), &registry);
        error
    };

    // Unfenced: Serving.
    assert_eq!(
        refuse(&mut rt, &request("x", 1, REVOKED)),
        ExpertAdmissionError::NotFenced {
            pod: "x".into(),
            state: ExpertState::Serving,
        }
    );
    // Fenced for AuthorityQuarantine: with either cause in the request.
    for cause in [REVOKED, QUARANTINE] {
        assert_eq!(
            refuse(&mut rt, &request("q", 2, cause)),
            ExpertAdmissionError::NotRetirable { pod: "q".into() }
        );
    }
    // Already retired.
    assert_eq!(
        refuse(&mut rt, &request("w", 3, REVOKED)),
        ExpertAdmissionError::NotFenced {
            pod: "w".into(),
            state: retired(),
        }
    );
    // Unknown expert.
    assert_eq!(
        refuse(&mut rt, &request("nope", 1, REVOKED)),
        ExpertAdmissionError::UnknownOperand { pod: "nope".into() }
    );
    // The revocation: none, another subject and another generation of a committed one.
    fence(&mut rt, "x", 1, REVOKED);
    let before = applied(&rt);
    let registry = rt.expert_registry().clone();
    let refuse = |rt: &mut PtrRuntime, request: &ExpertFenceRequest| {
        let error = expert_error(retire(rt, request));
        assert_eq!(applied(rt), before);
        assert_eq!(rt.expert_registry(), &registry);
        error
    };
    for revocation in [
        None,
        Some(ExpertFenceRevocation {
            subject: "other".to_owned(),
            generation: Generation(1),
        }),
        Some(ExpertFenceRevocation {
            subject: "k".to_owned(),
            generation: Generation(2),
        }),
    ] {
        assert_eq!(
            refuse(
                &mut rt,
                &ExpertFenceRequest {
                    revocation,
                    ..request("x", 2, REVOKED)
                }
            ),
            ExpertAdmissionError::FenceRevocationNotCommitted { pod: "x".into() }
        );
    }
    // Wrong generation and wrong state version.
    assert_eq!(
        refuse(
            &mut rt,
            &ExpertFenceRequest {
                generation: Generation(2),
                ..request("x", 2, REVOKED)
            }
        ),
        ExpertAdmissionError::StaleOperand {
            pod: "x".into(),
            field: ExpertStaleField::Generation,
        }
    );
    for version in [1, 3] {
        assert_eq!(
            refuse(&mut rt, &request("x", version, REVOKED)),
            ExpertAdmissionError::StaleOperand {
                pod: "x".into(),
                field: ExpertStaleField::StateVersion,
            }
        );
    }
    assert!(matches!(
        refuse(&mut rt, &request("x", u64::MAX, REVOKED)),
        ExpertAdmissionError::FenceEpochMismatch { .. }
    ));
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 2));

    // Authority: the gate grant and a grant of another project.
    prime(&mut rt);
    assert_eq!(
        expert_error(rt.retire_revoked_expert(&request("x", 2, REVOKED), &gate_grant())),
        ExpertAdmissionError::FenceAuthorityMismatch {
            action: ExpertFenceAction::Retire,
            required: ExpertFenceAuthority::RuntimeRevocation,
        }
    );
    let other = ExpertFenceGrant::revocation(
        ProjectId::from("other"),
        current_binding(&rt, ExpertGrantKind::Fence),
    );
    assert_eq!(
        expert_error(rt.retire_revoked_expert(&request("x", 2, REVOKED), &other)),
        ExpertAdmissionError::FenceProjectMismatch
    );
    // A runtime grant revoked between mint and commit.
    let minted = revocation_grant(&rt);
    rt.supersede_expert_grant(ExpertGrantKind::Fence).unwrap();
    assert!(matches!(
        expert_error(rt.retire_revoked_expert(&request("x", 2, REVOKED), &minted)),
        ExpertAdmissionError::GrantGenerationStale {
            kind: ExpertGrantKind::Fence,
            ..
        }
    ));
    assert_eq!(applied(&rt), before);
    assert_eq!(rt.expert_registry(), &registry);
    // And the control: the current grant retires it.
    let current = revocation_grant(&rt);
    rt.retire_revoked_expert(&request("x", 2, REVOKED), &current)
        .unwrap();
    assert_eq!(entry_state(&rt, "x"), (retired(), 3));
}

#[test]
fn the_retire_event_cannot_be_appended_directly_and_carries_no_gate_evidence() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    fence(&mut rt, "x", 1, REVOKED);
    let before = applied(&rt);
    let retire_event = |evidence: Option<ExpertFenceEvidence>| LedgerEvent::ExpertFenceCommitted {
        project: project(),
        pod: PodId::from("x"),
        action: ExpertFenceAction::Retire,
        cause: REVOKED,
        resume: ExpertFenceResume::Serving,
        generation: Generation(1),
        expected_state_version: 2,
        epoch: 3,
        revocation: Some(committed_revocation()),
        evidence,
    };
    // The public commit path refuses the event, the L3 and proposal paths included.
    assert!(matches!(
        rt.commit(retire_event(None)),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    ));
    // Gate evidence on a retirement cannot even be framed.
    let with_evidence = retire_event(Some(ExpertFenceEvidence {
        gate_epoch: 7,
        required_gate_config_digest: [6; 32],
        report: b"PTRGR-opaque".to_vec(),
    }));
    assert!(ptr_ledger::check_encodable(&with_evidence).is_err());
    assert!(rt.commit(with_evidence).is_err());
    assert_eq!(applied(&rt), before);
    assert_eq!(entry_state(&rt, "x"), (ExpertState::Fenced, 2));
}

#[test]
fn the_gate_path_cannot_retire_a_fenced_or_revoked_expert_and_nothing_leaves_the_retired_state() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    fence(&mut rt, "x", 1, REVOKED);
    // A gate-driven Retire proposal of the fenced operand is refused FencedTarget, whatever the
    // reason it names.
    for reason in [RetiredReason::GateFailed, RetiredReason::Superseded] {
        let target = target_as(&rt, "x", ExpertState::Fenced, 2);
        assert_eq!(
            propose(StructuralOp::Retire { target, reason }).unwrap_err(),
            ExpertProposalError::FencedTarget {
                op: ptr_pods::OpKind::Retire,
                pod: PodId::from("x"),
            }
        );
    }
    // A proposal built from the stale Serving view fails the compare-and-set and appends nothing.
    let stale = propose(StructuralOp::Retire {
        target: target_as(&rt, "x", ExpertState::Serving, 1),
        reason: RetiredReason::GateFailed,
    })
    .unwrap();
    let before = applied(&rt);
    assert!(matches!(
        try_admit(&mut rt, &stale),
        Err(RuntimeError::Expert(ExpertAdmissionError::StaleOperand {
            field: ExpertStaleField::State,
            ..
        }))
    ));
    // The reason InputRevoked is not a proposal at all: no gate-less retirement by proposal.
    assert_eq!(
        propose(StructuralOp::Retire {
            target: target_as(&rt, "x", ExpertState::Serving, 1),
            reason: RetiredReason::InputRevoked,
        })
        .unwrap_err(),
        ExpertProposalError::ReasonNotProposable {
            reason: RetiredReason::InputRevoked
        }
    );
    assert_eq!(applied(&rt), before);

    // Terminal: from Retired(InputRevoked) there is no way back.
    retire(&mut rt, &request("x", 2, REVOKED)).unwrap();
    let before = applied(&rt);
    let registry = rt.expert_registry().clone();
    // Reactivate: no seed, with or without a result spec.
    let seed = target_as(&rt, "x", retired(), 3);
    let spec = ExpertPodSpec::new(
        project(),
        PodId::from("x2"),
        ManifestDigest::new([77; 32]).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: 77,
        },
        base(),
        Generation(2),
    )
    .unwrap();
    for result in [None, Some(spec)] {
        assert_eq!(
            propose(StructuralOp::Reactivate {
                target: seed.clone(),
                result
            })
            .unwrap_err(),
            ExpertProposalError::SeedUnavailable {
                pod: PodId::from("x"),
                reason: RetiredReason::InputRevoked,
            }
        );
    }
    // Reactivate built from a stale Frozen view is a compare-and-set failure.
    let stale = propose(StructuralOp::Reactivate {
        target: target_as(&rt, "x", ExpertState::Frozen, 3),
        result: None,
    })
    .unwrap();
    assert!(matches!(
        try_admit(&mut rt, &stale),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::StaleOperand { .. }
        ))
    ));
    // Readmission, escalation, fencing and a second retirement.
    let gate_grant = gate_grant();
    prime(&mut rt);
    let state = retired();
    let request_v3 = request("x", 3, REVOKED);
    let readmit_request = ExpertFenceRequest {
        revocation: None,
        ..request("x", 3, QUARANTINE)
    };
    let report = {
        let (proposal_id, operation_id) = ptr_runtime::expert::expert_readmission_subject_ids(
            &project(),
            &PodId::from("x"),
            Generation(1),
            3,
            4,
            QUARANTINE,
        );
        let subject = GateSubject::new(
            proposal_id,
            operation_id,
            [b'x'; 32],
            [9; 32],
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
    assert_eq!(
        expert_error(rt.readmit_expert(&readmit_request, &report, &gate_grant)),
        ExpertAdmissionError::NotFenced {
            pod: "x".into(),
            state
        }
    );
    let grant = revocation_grant(&rt);
    assert_eq!(
        expert_error(rt.escalate_expert_fence(&request_v3, &grant)),
        ExpertAdmissionError::NotFenced {
            pod: "x".into(),
            state
        }
    );
    assert_eq!(
        expert_error(rt.fence_expert(&request_v3, &grant)),
        ExpertAdmissionError::NotFenceable {
            pod: "x".into(),
            state
        }
    );
    assert_eq!(
        expert_error(rt.retire_revoked_expert(&request_v3, &grant)),
        ExpertAdmissionError::NotFenced {
            pod: "x".into(),
            state
        }
    );
    assert_eq!(applied(&rt), before);
    assert_eq!(rt.expert_registry(), &registry);
    assert_retired_and_unroutable(&rt);
}

#[test]
fn replay_revalidates_cause_state_revocation_and_the_compare_and_set() {
    let rt = retired_runtime();
    let good = history(&rt);
    PtrRuntime::replay(PtrConfig::default(), &good).unwrap();
    let fence_at = good
        .iter()
        .position(|c| {
            matches!(
                c.event,
                LedgerEvent::ExpertFenceCommitted {
                    action: ExpertFenceAction::Fence,
                    ..
                }
            )
        })
        .unwrap();
    let retire_at = good.len() - 1;
    let change = |at: usize, edit: &dyn Fn(&mut LedgerEvent)| {
        let mut events = good.clone();
        edit(&mut events[at].event);
        PtrRuntime::replay(PtrConfig::default(), &events)
    };

    // A retirement naming a revocation the history does not hold.
    assert!(matches!(
        change(retire_at, &|event| {
            if let LedgerEvent::ExpertFenceCommitted { revocation, .. } = event {
                *revocation = Some(ExpertFenceRevocation {
                    subject: "other".to_owned(),
                    generation: Generation(1),
                });
            }
        }),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::FenceRevocationNotCommitted { .. }
        ))
    ));
    // A retirement of a fence that was an AuthorityQuarantine fence.
    assert!(matches!(
        change(fence_at, &|event| {
            if let LedgerEvent::ExpertFenceCommitted {
                cause, revocation, ..
            } = event
            {
                *cause = QUARANTINE;
                *revocation = None;
            }
        }),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::NotRetirable { .. }
        ))
    ));
    // A retirement with the wrong compare-and-set operands.
    for (generation, version) in [(2, 2), (1, 5)] {
        assert!(matches!(
            change(retire_at, &|event| {
                if let LedgerEvent::ExpertFenceCommitted {
                    generation: g,
                    expected_state_version,
                    epoch,
                    ..
                } = event
                {
                    *g = Generation(generation);
                    *expected_state_version = version;
                    *epoch = version + 1;
                }
            }),
            Err(RuntimeError::Expert(
                ExpertAdmissionError::StaleOperand { .. }
            ))
        ));
    }
    // A retirement of an expert that was never fenced (the fence record removed, the operands
    // pointed at the serving expert).
    let mut unfenced = good.clone();
    unfenced.remove(fence_at);
    if let LedgerEvent::ExpertFenceCommitted {
        expected_state_version,
        epoch,
        ..
    } = &mut unfenced.last_mut().unwrap().event
    {
        *expected_state_version = 1;
        *epoch = 2;
    }
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &unfenced),
        Err(RuntimeError::Expert(ExpertAdmissionError::NotFenced {
            state: ExpertState::Serving,
            ..
        }))
    ));
    // A second retirement of the same expert.
    let mut twice = good.clone();
    let mut again = good[retire_at].clone();
    again.index = CommitIndex(good[retire_at].index.0 + 1);
    if let LedgerEvent::ExpertFenceCommitted {
        expected_state_version,
        epoch,
        ..
    } = &mut again.event
    {
        *expected_state_version = 3;
        *epoch = 4;
    }
    twice.push(again);
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &twice),
        Err(RuntimeError::Expert(ExpertAdmissionError::NotFenced {
            state: ExpertState::Retired(RetiredReason::InputRevoked),
            ..
        }))
    ));
    // Retire evidence-free events decode only: a retirement carrying evidence is not a record.
    if let LedgerEvent::ExpertFenceCommitted { evidence, .. } = &mut twice[retire_at].event {
        *evidence = Some(ExpertFenceEvidence {
            gate_epoch: 1,
            required_gate_config_digest: [1; 32],
            report: vec![1],
        });
    }
    assert!(PtrRuntime::replay(PtrConfig::default(), &twice).is_err());
}

fn schema() -> KvTensorSchema {
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

fn salt() -> KvSalt {
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

fn kv_insert(
    backend: &InMemoryPagedKvBackend,
    rt: &mut PtrRuntime,
    deps: Vec<KvDependency>,
) -> Result<(), KvCatalogError> {
    let req = kv_request();
    let mut cache = backend
        .allocate_bound(
            schema(),
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
    backend
        .append(&mut cache, &[rows(1.0)], &[rows(9.0)])
        .unwrap();
    let prefix = backend.publish_prefix(&mut cache).unwrap();
    let bytes = backend.page_metrics(&cache).resident_bytes;
    rt.kv_prefix_insert(KvInsert {
        salt: &salt(),
        chain: &CHAIN,
        request: &req,
        prefix,
        page_bytes: bytes,
        extra_dependencies: deps,
    })
    .unwrap()
}

fn kv_hits(rt: &mut PtrRuntime) -> bool {
    let req = kv_request();
    let hit = rt
        .kv_prefix_lookup(KvLookup {
            salt: &salt(),
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

fn dep(pod: &str, epoch: u64) -> KvDependency {
    KvDependency::expert_fenced(&project(), &PodId::from(pod), Generation(1), [0; 32], epoch)
}

#[test]
fn a_retired_expert_is_dead_for_the_kv_catalog_and_no_entry_is_revived() {
    let mut rt = runtime();
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let catalog = KvPrefixCatalog::new(
        backend.take_fork_authority().unwrap(),
        CatalogLimits::default(),
    )
    .unwrap()
    .with_retirer(Arc::new(backend.clone()));
    rt.enable_kv_prefix_catalog(catalog).unwrap();
    birth(&mut rt, "x");
    kv_insert(&backend, &mut rt, vec![dep("x", 1)]).unwrap();
    assert!(kv_hits(&mut rt));
    fence(&mut rt, "x", 1, REVOKED);
    assert!(!kv_hits(&mut rt));
    retire(&mut rt, &request("x", 2, REVOKED)).unwrap();
    assert!(!kv_hits(&mut rt));
    // No epoch, neither the one before the fence, nor the fence's, nor the retirement's, nor a
    // later one makes a dependency on the retired expert live.
    for epoch in [0, 1, 2, 3, 4] {
        assert_eq!(
            kv_insert(&backend, &mut rt, vec![dep("x", epoch)]),
            Err(KvCatalogError::DependencyNotLive),
            "epoch {epoch}"
        );
    }
}
