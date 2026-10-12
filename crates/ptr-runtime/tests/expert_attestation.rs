//! The minimal attestor mechanism (R013, Tier 2) through the public API: an attestation binds
//! one proposal, one artifact digest, one single-use run id and an attestor at an epoch; the
//! attestor and the run are inputs of the admitted expert's dependency block, so the attestor's
//! revocation is the compound input invalidation (terminal for the experts admitted under it)
//! and a run id is consumed by the first record that names it, replay included.

use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, ExpertFenceCause};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertInputRef, ExpertInputs, ExpertLevel,
    ExpertPodSpec, ExpertProposal, ManifestDigest, ProposerRef, RationaleDigest, StructuralOp,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_artifact_digest_with_inputs,
    expert_evidence_base_model_digest, ExpertAdmissionError, ExpertAdmissionGrant,
    ExpertAttestation, ExpertFenceAuthority, ExpertFenceGrant, ExpertGrantKind, ExpertInputRole,
    InputInvalidation,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

mod expert_support;

use expert_support::{current_binding, prime, requirement, EVAL_MANIFEST};

const A_DIGEST: [u8; 32] = [0xA1; 32];
const A2_DIGEST: [u8; 32] = [0xA2; 32];
const B_DIGEST: [u8; 32] = [0xB1; 32];

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

fn spec(pod: &str) -> ExpertPodSpec {
    ExpertPodSpec::new(
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
    .unwrap()
}

fn birth(pod: &str) -> ExpertProposal {
    ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Birth { result: spec(pod) },
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

fn grant(rt: &PtrRuntime) -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        current_binding(rt, ExpertGrantKind::Admission),
    )
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

fn attestation(
    proposal: &ExpertProposal,
    attestor: &str,
    epoch: u64,
    digest: [u8; 32],
    run: u8,
) -> ExpertAttestation {
    ExpertAttestation::new(
        attestor,
        Generation(epoch),
        digest,
        [run; 32],
        *proposal.proposal_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
    )
    .unwrap()
}

fn admit_attested(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
    attestation: &ExpertAttestation,
) -> Result<CommitIndex, RuntimeError> {
    let block = attestation
        .attested_inputs(&ExpertInputs::default())
        .unwrap();
    let report = report_for(proposal, Some(&block));
    let grant = grant(rt);
    rt.admit_attested_expert_proposal(
        proposal,
        &report,
        &grant,
        ExpertInputs::default(),
        attestation,
    )
}

fn route(rt: &PtrRuntime, pod: &str) -> Result<(), ptr_runtime::expert::RouteRefusal> {
    rt.expert_route_admission(&project(), &PodId::from(pod), Generation(1))
}

fn expert_error(result: Result<CommitIndex, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {other:?}"),
    }
}

fn revoke(rt: &mut PtrRuntime, attestor: &str, epoch: u64, digest: [u8; 32]) {
    let grant =
        ExpertFenceGrant::global_revocation(current_binding(rt, ExpertGrantKind::GlobalRevocation));
    assert_eq!(grant.authority(), ExpertFenceAuthority::GlobalRevocation);
    rt.invalidate_input(
        &grant,
        InputInvalidation::revoked(format!("attestor:{attestor}"), Generation(epoch), digest),
    )
    .unwrap();
}

#[test]
fn an_attested_birth_is_admitted_and_its_block_names_the_attestor_and_the_run() {
    let mut rt = runtime();
    let proposal = birth("e");
    let attested = attestation(&proposal, "alpha", 1, A_DIGEST, 1);
    admit_attested(&mut rt, &proposal, &attested).unwrap();
    assert_eq!(route(&rt, "e"), Ok(()));
    let named: Vec<(String, Generation)> = rt
        .expert_registry()
        .dependencies()
        .recorded_inputs("proj", "e", Generation(1))
        .expect("the record names its inputs")
        .map(|(identity, generation)| (identity.to_string(), generation))
        .collect();
    assert!(named.contains(&("attestor:alpha".to_string(), Generation(1))));
    assert!(named.contains(&(attested.run_identity(), Generation(1))));
}

#[test]
fn an_attestation_of_another_artifact_or_another_proposal_is_refused_and_appends_nothing() {
    let mut rt = runtime();
    let proposal = birth("e");
    let other = birth("f");
    let events = rt.committed_events().len();
    let wrong_artifact = ExpertAttestation::new(
        "alpha",
        Generation(1),
        A_DIGEST,
        [1; 32],
        *proposal.proposal_id().as_bytes(),
        expert_evidence_artifact_digest(other.op()),
    )
    .unwrap();
    assert_eq!(
        expert_error(admit_attested(&mut rt, &proposal, &wrong_artifact)),
        ExpertAdmissionError::AttestationArtifactMismatch
    );
    let wrong_proposal = ExpertAttestation::new(
        "alpha",
        Generation(1),
        A_DIGEST,
        [2; 32],
        *other.proposal_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
    )
    .unwrap();
    assert_eq!(
        expert_error(admit_attested(&mut rt, &proposal, &wrong_proposal)),
        ExpertAdmissionError::AttestationProposalMismatch
    );
    assert_eq!(rt.committed_events().len(), events);
    assert!(rt
        .expert_registry()
        .get(&project(), &PodId::from("e"))
        .is_none());
}

#[test]
fn a_grant_that_requires_attestation_refuses_an_admission_without_one() {
    let mut rt = runtime();
    let proposal = birth("e");
    let required = grant(&rt).with_required_attestation();
    let report = report_for(&proposal, None);
    let events = rt.committed_events().len();
    assert_eq!(
        expert_error(rt.admit_expert_proposal(&proposal, &report, &required)),
        ExpertAdmissionError::AttestationRequired
    );
    let with_block = ExpertInputs::default();
    let report = report_for(&proposal, Some(&with_block));
    assert_eq!(
        expert_error(
            rt.admit_expert_proposal_with_inputs(&proposal, &report, &required, with_block)
        ),
        ExpertAdmissionError::AttestationRequired
    );
    assert_eq!(rt.committed_events().len(), events);
    // Without the requirement the same admission goes through, and with it an attested one does.
    let plain = grant(&rt);
    rt.admit_expert_proposal(&proposal, &report_for(&proposal, None), &plain)
        .unwrap();
    let second = birth("g");
    let attested = attestation(&second, "alpha", 1, A_DIGEST, 3);
    let block = attested.attested_inputs(&ExpertInputs::default()).unwrap();
    let report = report_for(&second, Some(&block));
    let required = grant(&rt).with_required_attestation();
    rt.admit_attested_expert_proposal(
        &second,
        &report,
        &required,
        ExpertInputs::default(),
        &attested,
    )
    .unwrap();
    assert_eq!(route(&rt, "g"), Ok(()));
}

#[test]
fn a_run_id_is_single_use_across_attestors_the_plain_path_and_a_replay() {
    let mut rt = runtime();
    let first = birth("e");
    let attested = attestation(&first, "alpha", 1, A_DIGEST, 1);
    admit_attested(&mut rt, &first, &attested).unwrap();

    let reused = |error: ExpertAdmissionError| {
        assert!(
            matches!(error, ExpertAdmissionError::RunIdReused { .. }),
            "{error:?}"
        );
    };
    let second = birth("f");
    reused(expert_error(admit_attested(
        &mut rt,
        &second,
        &attestation(&second, "alpha", 1, A_DIGEST, 1),
    )));
    reused(expert_error(admit_attested(
        &mut rt,
        &second,
        &attestation(&second, "beta", 1, B_DIGEST, 1),
    )));
    // The plain path naming the run as an input of its own.
    let run = ExpertInputRef::new(attested.run_identity(), Generation(1), [1; 32]).unwrap();
    let block = ExpertInputs::new(vec![run], Vec::new()).unwrap();
    let report = report_for(&second, Some(&block));
    reused(expert_error(rt.admit_expert_proposal_with_inputs(
        &second,
        &report,
        &grant(&rt),
        block,
    )));
    assert!(rt
        .expert_registry()
        .get(&project(), &PodId::from("f"))
        .is_none());

    // The consumed run is derived from the history: a runtime rebuilt by replay still refuses it,
    // and a history that names it twice does not replay.
    let events: Vec<CommittedEvent> = rt.committed_events().to_vec();
    let mut replayed = PtrRuntime::replay(PtrConfig::default(), &events).unwrap();
    prime(&mut replayed);
    reused(expert_error(admit_attested(
        &mut replayed,
        &second,
        &attestation(&second, "alpha", 1, A_DIGEST, 1),
    )));
    // A fresh run of the same attestor is admitted.
    admit_attested(
        &mut replayed,
        &second,
        &attestation(&second, "alpha", 1, A_DIGEST, 2),
    )
    .unwrap();

    // A donor runtime admits another expert under the same run; its record cannot be replayed
    // after the first.
    let mut donor = runtime();
    let donated = birth("d");
    admit_attested(
        &mut donor,
        &donated,
        &attestation(&donated, "beta", 1, B_DIGEST, 1),
    )
    .unwrap();
    let record = donor
        .committed_events()
        .iter()
        .rev()
        .find(|committed| {
            matches!(
                committed.event,
                ptr_ledger::LedgerEvent::ExpertStructuralCommitted { .. }
            )
        })
        .expect("the donor's record")
        .event
        .clone();
    let mut forged = events.clone();
    forged.push(CommittedEvent {
        index: CommitIndex(events.last().map_or(1, |last| last.index.0 + 1)),
        event: record,
    });
    match PtrRuntime::replay(PtrConfig::default(), &forged) {
        Err(RuntimeError::Expert(ExpertAdmissionError::RunIdReused { .. })) => {}
        other => panic!(
            "a history naming a run twice replayed: {:?}",
            other.map(|_| ())
        ),
    }
}

#[test]
fn a_proposal_queued_across_the_revocation_of_its_attestor_is_refused_and_a_fresh_one_is_not() {
    let mut rt = runtime();
    let queued = birth("e");
    let attested = attestation(&queued, "alpha", 1, A_DIGEST, 1);
    let block = attested.attested_inputs(&ExpertInputs::default()).unwrap();
    let report = report_for(&queued, Some(&block));
    let grant = grant(&rt);
    revoke(&mut rt, "alpha", 1, A_DIGEST);
    let refused = rt.admit_attested_expert_proposal(
        &queued,
        &report,
        &grant,
        ExpertInputs::default(),
        &attested,
    );
    assert!(
        matches!(
            expert_error(refused),
            ExpertAdmissionError::InputRevoked {
                role: ExpertInputRole::Training,
                ..
            }
        ),
        "the revoked attestor is a revoked input"
    );
    let fresh = birth("g");
    admit_attested(
        &mut rt,
        &fresh,
        &attestation(&fresh, "beta", 1, B_DIGEST, 2),
    )
    .unwrap();
    assert_eq!(route(&rt, "g"), Ok(()));
}

#[test]
fn the_revocation_of_an_attestor_fences_what_it_attested_for_good_and_a_new_epoch_retrains() {
    let mut rt = runtime();
    let attested_pod = birth("e");
    admit_attested(
        &mut rt,
        &attested_pod,
        &attestation(&attested_pod, "alpha", 1, A_DIGEST, 1),
    )
    .unwrap();
    let other_pod = birth("w");
    admit_attested(
        &mut rt,
        &other_pod,
        &attestation(&other_pod, "beta", 1, B_DIGEST, 2),
    )
    .unwrap();
    revoke(&mut rt, "alpha", 1, A_DIGEST);

    assert_eq!(
        route(&rt, "e"),
        Err(ptr_runtime::expert::RouteRefusal::Fenced)
    );
    assert_eq!(route(&rt, "w"), Ok(()));
    assert_eq!(
        rt.expert_registry()
            .fence_cause(&project(), &PodId::from("e")),
        Some(ExpertFenceCause::InputRevoked),
        "an attestor revocation is terminal: the cause is InputRevoked, which is never readmitted"
    );

    // The new epoch of the attestor admits a retrained expert; the old epoch stays revoked.
    let retrained = birth("r");
    admit_attested(
        &mut rt,
        &retrained,
        &attestation(&retrained, "alpha", 2, A2_DIGEST, 3),
    )
    .unwrap();
    assert_eq!(route(&rt, "r"), Ok(()));
    let again = birth("s");
    assert!(matches!(
        expert_error(admit_attested(
            &mut rt,
            &again,
            &attestation(&again, "alpha", 1, A_DIGEST, 4)
        )),
        ExpertAdmissionError::InputRevoked { .. }
    ));
}

#[test]
fn an_attestation_names_no_attestor_or_run_of_the_block_it_extends() {
    let proposal = birth("e");
    let attested = attestation(&proposal, "alpha", 1, A_DIGEST, 1);
    let own = ExpertInputRef::new("attestor:smuggled", Generation(1), [3; 32]).unwrap();
    let block = ExpertInputs::new(vec![own], Vec::new()).unwrap();
    assert!(matches!(
        attested.attested_inputs(&block),
        Err(ExpertAdmissionError::InvalidRecord(_))
    ));
    assert!(
        ExpertAttestation::new("", Generation(1), A_DIGEST, [1; 32], [0; 32], [0; 32]).is_err()
    );
    assert!(
        ExpertAttestation::new("alpha", Generation(0), A_DIGEST, [1; 32], [0; 32], [0; 32])
            .is_err()
    );
    assert!(
        ExpertAttestation::new("alpha", Generation(1), [0; 32], [1; 32], [0; 32], [0; 32]).is_err()
    );
    assert!(
        ExpertAttestation::new("alpha", Generation(1), A_DIGEST, [0; 32], [0; 32], [0; 32])
            .is_err()
    );
}
