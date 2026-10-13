//! The fixes to the R011 WP1-WP3 invariant review (WP3b), through the public API: a descendant
//! inherits what its ancestors rest on, the digest registry cannot be squatted into blocking a
//! revocation, an admission can be made to require its dependency block and the gate attests it,
//! a grant cannot outlive the host clock, input invalidation needs the runtime-wide authority,
//! a retirement cites a revocation the expert rests on (or the supersession that fenced it), and
//! the plain projection is not an authority for fenced experts.

use ptr_config::PtrConfig;
use ptr_ledger::{
    CommittedEvent, ExpertFenceAction, ExpertFenceCause, ExpertFenceRevocation, LedgerEvent,
};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertInputRef, ExpertInputs,
    ExpertLevel, ExpertPodSpec, ExpertProposal, ExpertState, ExpertTarget, ExpertTargetInit,
    ManifestDigest, ProposerRef, RationaleDigest, RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_artifact_digest_with_inputs,
    expert_evidence_base_model_digest, ledger_op_kind, ExpertAdmissionError, ExpertAdmissionGrant,
    ExpertFenceAuthority, ExpertFenceGrant, ExpertFenceRequest, ExpertGateRequirement,
    ExpertGrantKind, ExpertInputRole, InputInvalidation, InputInvalidationOutcome, RouteRefusal,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapsuleId, CommitIndex, Generation, PodId, ProjectId, Timestamp};

mod expert_support;

use expert_support::{
    current_binding, prime, requirement, requirement_requiring_inputs, EVAL_MANIFEST,
};

const T: &str = "dataset:train";
const U: &str = "dataset:other";
const D1: [u8; 32] = [0x11; 32];
const D2: [u8; 32] = [0x22; 32];
const D3: [u8; 32] = [0x33; 32];
const REPLACEMENT: [u8; 32] = [0x77; 32];
const EPOCH: u64 = 7;

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

fn bind(subject: GateSubject) -> BoundGateReport {
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

/// A report whose subject carries `artifact`.
fn report_with_artifact(proposal: &ExpertProposal, artifact: [u8; 32]) -> BoundGateReport {
    bind(GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        artifact,
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    ))
}

/// A report bound to `proposal` and, when given, to the dependency block `inputs`.
fn report_for(proposal: &ExpertProposal, inputs: Option<&ExpertInputs>) -> BoundGateReport {
    let artifact = match inputs {
        Some(inputs) => expert_evidence_artifact_digest_with_inputs(proposal.op(), inputs),
        None => expert_evidence_artifact_digest(proposal.op()),
    };
    report_with_artifact(proposal, artifact)
}

fn admission_grant(rt: &PtrRuntime, requirement: ExpertGateRequirement) -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement,
        current_binding(rt, ExpertGrantKind::Admission),
    )
}

fn plain_requirement() -> ExpertGateRequirement {
    requirement(EPOCH, gate().config_digest().unwrap())
}

fn admit(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
    inputs: Option<ExpertInputs>,
) -> Result<CommitIndex, RuntimeError> {
    prime(rt);
    let grant = admission_grant(rt, plain_requirement());
    let report = report_for(proposal, inputs.as_ref());
    match inputs {
        Some(inputs) => rt.admit_expert_proposal_with_inputs(proposal, &report, &grant, inputs),
        None => rt.admit_expert_proposal(proposal, &report, &grant),
    }
}

fn input(identity: &str, generation: u64, digest: [u8; 32]) -> ExpertInputRef {
    ExpertInputRef::new(identity, Generation(generation), digest).unwrap()
}

fn inputs(training: &[(&str, u64, [u8; 32])]) -> ExpertInputs {
    ExpertInputs::new(
        training.iter().map(|i| input(i.0, i.1, i.2)).collect(),
        vec![],
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

fn born(rt: &mut PtrRuntime, pod: &str, inputs: Option<ExpertInputs>) {
    admit(
        rt,
        &propose(StructuralOp::Birth {
            result: token_spec(pod, 1),
        }),
        inputs,
    )
    .unwrap();
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

fn global_grant(rt: &PtrRuntime) -> ExpertFenceGrant {
    ExpertFenceGrant::global_revocation(current_binding(rt, ExpertGrantKind::GlobalRevocation))
}

fn revocation_grant(rt: &PtrRuntime) -> ExpertFenceGrant {
    ExpertFenceGrant::revocation(project(), current_binding(rt, ExpertGrantKind::Fence))
}

fn invalidate(
    rt: &mut PtrRuntime,
    request: InputInvalidation,
) -> Result<InputInvalidationOutcome, RuntimeError> {
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

fn history(rt: &PtrRuntime) -> Vec<CommittedEvent> {
    rt.committed_events().to_vec()
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

fn retire_gate_failed(rt: &mut PtrRuntime, pod: &str) {
    let proposal = propose(StructuralOp::Retire {
        target: target_of(rt, pod),
        reason: RetiredReason::GateFailed,
    });
    admit(rt, &proposal, None).unwrap();
}

fn reactivation(rt: &PtrRuntime, pod: &str, generation: u64) -> ExpertProposal {
    propose(StructuralOp::Reactivate {
        target: target_of(rt, pod),
        result: Some(token_spec(pod, generation)),
    })
}

// ---- P1-1: a descendant inherits what its ancestors rest on --------------------------------------

/// E@1 rests on T@1, is retired for a failed gate and T@1 is then revoked: the retired expert
/// fences nothing, and the reactivated generation must not come back as a serving descendant of
/// revoked data.
#[test]
fn reactivating_after_an_ancestor_input_was_revoked_is_refused_with_and_without_inputs() {
    let mut rt = runtime();
    born(&mut rt, "e", Some(inputs(&[(T, 1, D1)])));
    retire_gate_failed(&mut rt, "e");
    let outcome = invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    assert_eq!(outcome.fence_count(), 0, "a retired expert is not fenced");
    let before = history(&rt);
    let revoked_ancestor = ExpertAdmissionError::InputRevoked {
        role: ExpertInputRole::Inherited,
        identity: T.into(),
        generation: Generation(1),
    };
    // Without a block of its own.
    let proposal = reactivation(&rt, "e", 2);
    assert_eq!(
        expert_error(admit(&mut rt, &proposal, None)),
        revoked_ancestor
    );
    // With a block that names live inputs only: the revoked one is the ancestor's.
    assert_eq!(
        expert_error(admit(&mut rt, &proposal, Some(inputs(&[(U, 1, D3)])))),
        revoked_ancestor
    );
    assert_eq!(history(&rt), before, "nothing was appended");
    assert!(route(&rt, "e", 2).is_err());
    assert_eq!(
        state_of(&rt, "e").0,
        ExpertState::Retired(RetiredReason::GateFailed)
    );
}

/// The same for a supersession, with the exemption for a successor retrained on the replacement.
#[test]
fn reactivating_after_a_superseded_ancestor_input_needs_the_replacement() {
    let build = || {
        let mut rt = runtime();
        born(&mut rt, "e", Some(inputs(&[(T, 1, D1)])));
        retire_gate_failed(&mut rt, "e");
        invalidate(&mut rt, supersede(T, 1, D1, 2, D2)).unwrap();
        rt
    };
    let mut rt = build();
    let proposal = reactivation(&rt, "e", 2);
    assert_eq!(
        expert_error(admit(&mut rt, &proposal, None)),
        ExpertAdmissionError::InputSuperseded {
            role: ExpertInputRole::Inherited,
            identity: T.into(),
            generation: Generation(1),
            floor: Generation(2),
        }
    );
    // Naming the replacement is the retraining the exemption allows.
    let mut rt = build();
    let proposal = reactivation(&rt, "e", 2);
    admit(&mut rt, &proposal, Some(inputs(&[(T, 2, D2)]))).unwrap();
    assert_eq!(route(&rt, "e", 2), Ok(()));
}

/// A history in which a descendant is admitted after an input of its ancestor was invalidated
/// does not replay, so an invalidation committed before the descendant exists cannot be skipped
/// by a writer that splices the records in that order.
#[test]
fn replay_refuses_a_descendant_admitted_after_its_ancestors_input_was_revoked() {
    let mut a = runtime();
    born(&mut a, "e", Some(inputs(&[(T, 1, D1)])));
    retire_gate_failed(&mut a, "e");
    let honest_prefix = history(&a);
    let proposal = reactivation(&a, "e", 2);
    admit(&mut a, &proposal, None).unwrap();
    let reactivate = a.committed_events().last().unwrap().event.clone();
    let mut b = runtime();
    born(&mut b, "e", Some(inputs(&[(T, 1, D1)])));
    retire_gate_failed(&mut b, "e");
    invalidate(&mut b, revoke(T, 1, D1)).unwrap();
    assert_eq!(history(&b)[..honest_prefix.len()], honest_prefix[..]);
    let mut forged = history(&b);
    forged.push(CommittedEvent {
        index: CommitIndex(forged.len() as u64 + 1),
        event: reactivate,
    });
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &forged),
        Err(RuntimeError::Expert(ExpertAdmissionError::InputRevoked {
            role: ExpertInputRole::Inherited,
            ..
        }))
    ));
    // The honest order replays.
    assert!(PtrRuntime::replay(PtrConfig::default(), a.committed_events()).is_ok());
}

/// A descendant that names no block is judged over its ancestors' inputs: a plain `Revoked`
/// record (no fence follows it) makes the descendant unroutable.
#[test]
fn a_descendant_without_a_block_is_unroutable_when_an_ancestor_input_is_revoked_plainly() {
    let mut rt = runtime();
    born(&mut rt, "a", Some(inputs(&[("data:x", 1, D1)])));
    born(&mut rt, "c", None);
    let merge = propose(StructuralOp::Merge {
        sources: vec![target_of(&rt, "a"), target_of(&rt, "c")],
        result: token_spec("b", 1),
    });
    admit(&mut rt, &merge, None).unwrap();
    assert_eq!(route(&rt, "b", 1), Ok(()));
    rt.commit(LedgerEvent::Revoked {
        subject: "data:x".to_owned(),
        generation: Generation(1),
    })
    .unwrap();
    assert_eq!(
        route(&rt, "b", 1),
        Err(RouteRefusal::DependencyNotLive {
            identity: "data:x".into(),
            generation: Generation(1),
        })
    );
}

// ---- P1-2: the digest registry cannot be squatted ------------------------------------------------

#[test]
fn a_record_naming_a_generation_ahead_of_the_live_one_is_refused() {
    let mut rt = runtime();
    rt.commit(LedgerEvent::CapsuleCommitted {
        project: project(),
        capsule: CapsuleId::from("cap"),
        generation: Generation(1),
    })
    .unwrap();
    let before = history(&rt);
    let proposal = propose(StructuralOp::Birth {
        result: token_spec("x", 1),
    });
    assert_eq!(
        expert_error(admit(&mut rt, &proposal, Some(inputs(&[("cap", 2, D3)])))),
        ExpertAdmissionError::InputGenerationMismatch {
            identity: "cap".into(),
            generation: Generation(2),
            live: Generation(1),
        }
    );
    assert_eq!(history(&rt), before);
    // The live generation is accepted.
    admit(&mut rt, &proposal, Some(inputs(&[("cap", 1, D1)]))).unwrap();
}

#[test]
fn a_revocation_does_not_depend_on_the_digest_a_proposer_asserted() {
    let mut rt = runtime();
    // The lifecycle authority knows nothing of this identity, so the first use fixes whatever
    // digest the proposer asserts: here a bogus one.
    born(&mut rt, "x", Some(inputs(&[("squat:data", 1, D3)])));
    // The honest producer, naming the true digest, is told its digest is stale (the registry is
    // proposer-asserted: the first use wins).
    let honest = propose(StructuralOp::Birth {
        result: token_spec("y", 1),
    });
    assert_eq!(
        expert_error(admit(
            &mut rt,
            &honest,
            Some(inputs(&[("squat:data", 1, D1)]))
        )),
        ExpertAdmissionError::InputDigestStale {
            role: ExpertInputRole::Training,
            identity: "squat:data".into(),
            generation: Generation(1),
        }
    );
    // The revocation with the true digest is not blocked: it commits and fences the dependent.
    let outcome = invalidate(&mut rt, revoke("squat:data", 1, D1)).unwrap();
    assert_eq!(outcome.fence_count(), 1);
    assert_eq!(state_of(&rt, "x"), (ExpertState::Fenced, 2));
    assert_eq!(route(&rt, "x", 1), Err(RouteRefusal::Fenced));
    // The event records the digest the revoker asserted.
    assert!(matches!(
        rt.committed_events().last().unwrap().event,
        LedgerEvent::InputInvalidated { binding_digest, .. } if binding_digest == D1
    ));
    // A supersession starts from the registered digest and is still held to it (a limit).
    let mut rt = runtime();
    born(&mut rt, "x", Some(inputs(&[("squat:data", 1, D3)])));
    assert_eq!(
        expert_error(invalidate(&mut rt, supersede("squat:data", 1, D1, 2, D2))),
        ExpertAdmissionError::InputBindingMismatch {
            identity: "squat:data".into(),
            generation: Generation(1),
        }
    );
}

// ---- P1-3: completeness is required by the host and attested by the gate -------------------------

#[test]
fn a_grant_requiring_inputs_refuses_a_record_without_a_block() {
    let mut rt = runtime();
    let grant = admission_grant(
        &rt,
        requirement_requiring_inputs(EPOCH, gate().config_digest().unwrap()),
    );
    assert!(grant.requires_inputs());
    let proposal = propose(StructuralOp::Birth {
        result: token_spec("x", 1),
    });
    let before = history(&rt);
    // The v2 path.
    assert_eq!(
        expert_error(rt.admit_expert_proposal(&proposal, &report_for(&proposal, None), &grant)),
        ExpertAdmissionError::InputsRequired
    );
    assert_eq!(history(&rt), before);
    // With a block (even an empty one) the same grant admits.
    let block = inputs(&[]);
    rt.admit_expert_proposal_with_inputs(
        &proposal,
        &report_for(&proposal, Some(&block)),
        &grant,
        block,
    )
    .unwrap();
    // A grant that does not require them still admits the v2 path.
    let other = propose(StructuralOp::Birth {
        result: token_spec("y", 1),
    });
    admit(&mut rt, &other, None).unwrap();
}

#[test]
fn evidence_for_a_different_inputs_block_is_refused() {
    let mut rt = runtime();
    let proposal = propose(StructuralOp::Birth {
        result: token_spec("x", 1),
    });
    let named = inputs(&[(T, 1, D1)]);
    let other = inputs(&[(T, 1, D2)]);
    prime(&mut rt);
    let grant = admission_grant(&rt, plain_requirement());
    let before = history(&rt);
    // A report produced for another block.
    assert_eq!(
        expert_error(rt.admit_expert_proposal_with_inputs(
            &proposal,
            &report_for(&proposal, Some(&other)),
            &grant,
            named.clone(),
        )),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    // A report produced for no block at all.
    assert_eq!(
        expert_error(rt.admit_expert_proposal_with_inputs(
            &proposal,
            &report_for(&proposal, None),
            &grant,
            named.clone(),
        )),
        ExpertAdmissionError::EvidenceArtifactMismatch
    );
    assert_eq!(history(&rt), before);
    // The report bound to the block passes.
    rt.admit_expert_proposal_with_inputs(
        &proposal,
        &report_for(&proposal, Some(&named)),
        &grant,
        named,
    )
    .unwrap();
}

#[test]
fn replay_re_checks_that_the_evidence_is_bound_to_the_inputs_block() {
    let rt = runtime();
    let proposal = propose(StructuralOp::Birth {
        result: token_spec("x", 1),
    });
    let block = inputs(&[(T, 1, D1)]);
    // A record whose gate report was produced for no block (the plain artifact digest).
    let report = report_for(&proposal, None);
    let record = ExpertCommitRecord::new_with_inputs(
        proposal.clone(),
        report.digest(),
        EPOCH,
        gate().config_digest().unwrap(),
        report.encode_canonical(),
        block.clone(),
    )
    .unwrap();
    let event = |record: &ExpertCommitRecord| LedgerEvent::ExpertStructuralCommitted {
        project: project(),
        op_kind: ledger_op_kind(proposal.op().kind()),
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest: *record.evidence_digest(),
        record: record.encode_canonical(),
    };
    let mut forged = history(&rt);
    forged.push(CommittedEvent {
        index: CommitIndex(forged.len() as u64 + 1),
        event: event(&record),
    });
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &forged),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::EvidenceArtifactMismatch
        ))
    ));
    // Positive control: the same record with evidence bound to the block replays.
    let bound = report_for(&proposal, Some(&block));
    let record = ExpertCommitRecord::new_with_inputs(
        proposal.clone(),
        bound.digest(),
        EPOCH,
        gate().config_digest().unwrap(),
        bound.encode_canonical(),
        block,
    )
    .unwrap();
    let mut honest = history(&rt);
    honest.push(CommittedEvent {
        index: CommitIndex(honest.len() as u64 + 1),
        event: event(&record),
    });
    assert!(PtrRuntime::replay(PtrConfig::default(), &honest).is_ok());
}

// ---- P1-4: a grant cannot outlive the host clock -------------------------------------------------

#[test]
fn a_grant_reused_after_the_host_advanced_time_past_expiry_is_refused() {
    let mut rt = runtime();
    let grant = admission_grant(&rt, plain_requirement());
    let first = propose(StructuralOp::Birth {
        result: token_spec("x", 1),
    });
    // The grant (stamped NOW = 1000, identity valid until 10_000) commits while the host clock
    // is below the expiry.
    rt.observe_host_time(Timestamp(5_000)).unwrap();
    rt.admit_expert_proposal(&first, &report_for(&first, None), &grant)
        .unwrap();
    // The host clock passes the expiry; the same grant object, still stamped NOW, is refused.
    rt.observe_host_time(Timestamp(20_000)).unwrap();
    let second = propose(StructuralOp::Birth {
        result: token_spec("y", 1),
    });
    let before = history(&rt);
    assert_eq!(
        expert_error(rt.admit_expert_proposal(&second, &report_for(&second, None), &grant)),
        ExpertAdmissionError::GrantIdentityInvalid
    );
    assert_eq!(history(&rt), before);
    // The floor is monotone.
    assert_eq!(
        rt.observe_host_time(Timestamp(100)),
        Err(ExpertAdmissionError::HostTimeRegressed {
            floor: Timestamp(20_000),
            found: Timestamp(100),
        })
    );
    assert_eq!(rt.host_time_floor(), Timestamp(20_000));
    // Re-observing the same reading is allowed.
    rt.observe_host_time(Timestamp(20_000)).unwrap();
}

// ---- P1-5: input invalidation needs the runtime-wide authority -----------------------------------

#[test]
fn a_project_scoped_revocation_grant_cannot_invalidate_an_input() {
    let mut rt = runtime();
    born(&mut rt, "x", Some(inputs(&[(T, 1, D1)])));
    prime(&mut rt);
    let scoped = revocation_grant(&rt);
    let before = history(&rt);
    assert_eq!(
        expert_error(rt.invalidate_input(&scoped, revoke(T, 1, D1))),
        ExpertAdmissionError::FenceAuthorityMismatch {
            action: ExpertFenceAction::Fence,
            required: ExpertFenceAuthority::GlobalRevocation,
        }
    );
    assert_eq!(history(&rt), before);
    assert_eq!(route(&rt, "x", 1), Ok(()));
    // The runtime-wide grant passes, and it may not fence one expert.
    let outcome = invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    assert_eq!(outcome.fence_count(), 1);
    let global = global_grant(&rt);
    assert!(global.project().is_none());
    assert!(matches!(
        expert_error(rt.fence_expert(&request("x", 2), &global)),
        ExpertAdmissionError::FenceAuthorityMismatch { .. }
    ));
}

// ---- P2-4 and P2-5: what a fence, an escalation and a retirement may cite -----------------------

fn request(pod: &str, version: u64) -> ExpertFenceRequest {
    ExpertFenceRequest {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        cause: ExpertFenceCause::InputRevoked,
        revocation: None,
    }
}

fn citing(pod: &str, version: u64, subject: &str, generation: u64) -> ExpertFenceRequest {
    ExpertFenceRequest {
        revocation: Some(ExpertFenceRevocation {
            subject: subject.to_owned(),
            generation: Generation(generation),
        }),
        ..request(pod, version)
    }
}

fn quarantine(pod: &str, version: u64) -> ExpertFenceRequest {
    ExpertFenceRequest {
        cause: ExpertFenceCause::AuthorityQuarantine,
        ..request(pod, version)
    }
}

fn plain_revoke(rt: &mut PtrRuntime, subject: &str) {
    rt.commit(LedgerEvent::Revoked {
        subject: subject.to_owned(),
        generation: Generation(1),
    })
    .unwrap();
}

#[test]
fn a_retirement_or_escalation_citing_an_unrelated_revocation_is_refused() {
    let mut rt = runtime();
    born(&mut rt, "x", Some(inputs(&[(T, 1, D1)])));
    born(&mut rt, "y", Some(inputs(&[(T, 1, D1)])));
    born(&mut rt, "w", None);
    plain_revoke(&mut rt, T);
    plain_revoke(&mut rt, U);
    prime(&mut rt);
    let grant = revocation_grant(&rt);
    // A plain fence of an expert that recorded its inputs must cite one of them.
    assert_eq!(
        expert_error(rt.fence_expert(&citing("x", 1, U, 1), &grant)),
        ExpertAdmissionError::RevocationNotADependency {
            pod: "x".into(),
            subject: U.into(),
            generation: Generation(1),
        }
    );
    rt.fence_expert(&citing("x", 1, T, 1), &grant).unwrap();
    // Its retirement may not cite the unrelated revocation of U either.
    let before = history(&rt);
    assert_eq!(
        expert_error(rt.retire_revoked_expert(&citing("x", 2, U, 1), &grant)),
        ExpertAdmissionError::RevocationNotADependency {
            pod: "x".into(),
            subject: U.into(),
            generation: Generation(1),
        }
    );
    assert_eq!(history(&rt), before);
    rt.retire_revoked_expert(&citing("x", 2, T, 1), &grant)
        .unwrap();
    // The same for an escalation of a quarantined expert.
    rt.fence_expert(&quarantine("y", 1), &grant).unwrap();
    assert_eq!(
        expert_error(rt.escalate_expert_fence(&citing("y", 2, U, 1), &grant)),
        ExpertAdmissionError::RevocationNotADependency {
            pod: "y".into(),
            subject: U.into(),
            generation: Generation(1),
        }
    );
    rt.escalate_expert_fence(&citing("y", 2, T, 1), &grant)
        .unwrap();
    // An expert that recorded no inputs has nothing to compare with: any committed revocation.
    rt.fence_expert(&citing("w", 1, U, 1), &grant).unwrap();
    // Replay repeats the check.
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
}

#[test]
fn an_expert_fenced_by_a_supersession_is_retired_citing_it() {
    let mut rt = runtime();
    born(&mut rt, "x", Some(inputs(&[(T, 1, D1)])));
    invalidate(&mut rt, supersede(T, 1, D1, 2, D2)).unwrap();
    assert_eq!(state_of(&rt, "x"), (ExpertState::Fenced, 2));
    prime(&mut rt);
    let grant = revocation_grant(&rt);
    let before = history(&rt);
    // The new generation is not superseded, and a plain fence does not accept a supersession.
    assert_eq!(
        expert_error(rt.retire_revoked_expert(&citing("x", 2, T, 2), &grant)),
        ExpertAdmissionError::FenceRevocationNotCommitted { pod: "x".into() }
    );
    assert_eq!(history(&rt), before);
    // Only a retirement accepts a supersession as the citation: a fence does not.
    born(&mut rt, "q", Some(inputs(&[(U, 1, D3)])));
    prime(&mut rt);
    let grant = revocation_grant(&rt);
    assert_eq!(
        expert_error(rt.fence_expert(&citing("q", 1, T, 1), &grant)),
        ExpertAdmissionError::FenceRevocationNotCommitted { pod: "q".into() }
    );
    // The committed supersession of the generation it rests on retires it.
    rt.retire_revoked_expert(&citing("x", 2, T, 1), &grant)
        .unwrap();
    assert_eq!(
        state_of(&rt, "x"),
        (ExpertState::Retired(RetiredReason::InputRevoked), 3)
    );
    assert_eq!(
        route(&rt, "x", 1),
        Err(RouteRefusal::NotServing {
            state: ExpertState::Retired(RetiredReason::InputRevoked)
        })
    );
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
}

// ---- P2-6: the plain projection is not an authority for fenced experts ---------------------------

#[test]
fn materialized_state_marks_the_fenced_experts_but_a_naive_projection_replay_does_not() {
    let mut rt = runtime();
    born(&mut rt, "x", Some(inputs(&[(T, 1, D1)])));
    born(&mut rt, "z", Some(inputs(&[(U, 1, D3)])));
    invalidate(&mut rt, revoke(T, 1, D1)).unwrap();
    // The runtime's materialized state carries the fence it derived from its dependency index.
    assert_eq!(
        rt.materialized_state().values.get("expert:proj/x"),
        Some(&"1:fenced:2".to_owned())
    );
    // Applying only `projection_entries` (what ptr-state's reference backend and ptr-pg do) shows
    // x as serving: the event names the fenced experts by count and digest only. Nobody may
    // route from that projection.
    let mut naive = ptr_state::MaterializedState::default();
    for committed in rt.committed_events() {
        naive.apply(committed);
    }
    assert_eq!(naive.last_applied, rt.materialized_state().last_applied);
    assert_eq!(
        naive.values.get("expert:proj/x"),
        Some(&"1:serving:1".to_owned())
    );
    assert_ne!(
        naive.values.get("expert:proj/x"),
        rt.materialized_state().values.get("expert:proj/x")
    );
    // The invalidation itself is in both.
    assert_eq!(
        naive.values.get(&ptr_state::expert_input_key(T, 1)),
        rt.materialized_state()
            .values
            .get(&ptr_state::expert_input_key(T, 1))
    );
    // Unfenced experts agree.
    assert_eq!(
        naive.values.get("expert:proj/z"),
        rt.materialized_state().values.get("expert:proj/z")
    );
}
