//! The signed attestor mechanism (R013, Tier 2) through the public API: an attestation is a
//! record signed by a registered attestor key. It binds one proposal, one artifact digest, the
//! base model, the result generations, one single-use run id and an attestor at an epoch; the
//! attestor, the run and the signature are inputs of the admitted expert's dependency block, so
//! replay re-verifies them, the attestor's revocation is the compound input invalidation
//! (terminal for the experts admitted under it), a run id is consumed by the first record that
//! names it, and a requirement for attestation is recorded with the admission.

use ed25519_dalek::SigningKey;
use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, ExpertFenceCause, LedgerEvent};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertInputRef, ExpertInputs,
    ExpertLevel, ExpertPodSpec, ExpertProposal, ExpertTarget, ExpertTargetInit, ManifestDigest,
    ProposerRef, RationaleDigest, RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    expert_attestor_key_id, expert_evidence_artifact_digest,
    expert_evidence_artifact_digest_with_inputs, expert_evidence_base_model_digest, ledger_op_kind,
    ExpertAdmissionError, ExpertAdmissionGrant, ExpertAttestation, ExpertAttestationClaim,
    ExpertAttestorRegistry, ExpertFenceAuthority, ExpertFenceGrant, ExpertGrantKind,
    ExpertInputRole, InputInvalidation, ATTESTATION_POLICY_IDENTITY,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

mod expert_support;

use expert_support::{current_binding, prime, requirement, EVAL_MANIFEST};

fn alpha() -> SigningKey {
    SigningKey::from_bytes(&[0xA1; 32])
}

fn alpha2() -> SigningKey {
    SigningKey::from_bytes(&[0xA2; 32])
}

fn beta() -> SigningKey {
    SigningKey::from_bytes(&[0xB1; 32])
}

/// A key nobody registered.
fn rogue() -> SigningKey {
    SigningKey::from_bytes(&[0xEE; 32])
}

fn key_id(key: &SigningKey) -> [u8; 32] {
    expert_attestor_key_id(&key.verifying_key())
}

/// The registry a host that trusts alpha (epochs 1 and 2) and beta builds.
fn registry() -> ExpertAttestorRegistry {
    let mut registry = ExpertAttestorRegistry::new();
    registry
        .register("alpha", Generation(1), alpha().verifying_key())
        .unwrap();
    registry
        .register("alpha", Generation(2), alpha2().verifying_key())
        .unwrap();
    registry
        .register("beta", Generation(1), beta().verifying_key())
        .unwrap();
    registry
}

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

fn gate_config() -> [u8; 32] {
    gate().config_digest().unwrap()
}

fn grant(rt: &PtrRuntime) -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate_config()),
        current_binding(rt, ExpertGrantKind::Admission),
    )
}

fn required(rt: &PtrRuntime) -> ExpertAdmissionGrant {
    grant(rt).with_required_attestation()
}

/// A runtime whose host trusts alpha (epochs 1 and 2) and beta: the trusted set is fixed at
/// construction.
fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new_with_attestors(PtrConfig::default(), registry()).unwrap();
    prime(&mut rt);
    rt
}

fn signed(
    proposal: &ExpertProposal,
    name: &str,
    epoch: u64,
    key: &SigningKey,
    run: u8,
) -> ExpertAttestation {
    ExpertAttestationClaim::for_proposal(name, Generation(epoch), [run; 32], proposal)
        .unwrap()
        .sign(key)
}

fn alpha_attests(proposal: &ExpertProposal, run: u8) -> ExpertAttestation {
    signed(proposal, "alpha", 1, &alpha(), run)
}

fn admit_with(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
    attestation: &ExpertAttestation,
    grant: &ExpertAdmissionGrant,
) -> Result<CommitIndex, RuntimeError> {
    let block = attestation
        .attested_inputs(&ExpertInputs::default(), grant)
        .unwrap();
    let report = report_for(proposal, Some(&block));
    rt.admit_attested_expert_proposal(
        proposal,
        &report,
        grant,
        ExpertInputs::default(),
        attestation,
    )
}

fn admit_attested(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
    attestation: &ExpertAttestation,
) -> Result<CommitIndex, RuntimeError> {
    let grant = grant(rt);
    admit_with(rt, proposal, attestation, &grant)
}

fn admit_plain(
    rt: &mut PtrRuntime,
    proposal: &ExpertProposal,
) -> Result<CommitIndex, RuntimeError> {
    let grant = grant(rt);
    rt.admit_expert_proposal(proposal, &report_for(proposal, None), &grant)
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

fn revoke(rt: &mut PtrRuntime, attestor: &str, epoch: u64, key: &SigningKey) {
    let grant =
        ExpertFenceGrant::global_revocation(current_binding(rt, ExpertGrantKind::GlobalRevocation));
    assert_eq!(grant.authority(), ExpertFenceAuthority::GlobalRevocation);
    rt.invalidate_input(
        &grant,
        InputInvalidation::revoked(
            format!("attestor:{attestor}"),
            Generation(epoch),
            key_id(key),
        ),
    )
    .unwrap();
}

fn replay(events: &[CommittedEvent]) -> Result<PtrRuntime, RuntimeError> {
    PtrRuntime::replay_with_attestors(PtrConfig::default(), events, registry())
}

/// The event a record of `proposal` over `block` (none: an unattested record) would be, as a
/// ledger writer that bypasses the public API could append it after `history`.
fn forged_history(
    history: &[CommittedEvent],
    proposal: &ExpertProposal,
    block: Option<ExpertInputs>,
) -> Vec<CommittedEvent> {
    let report = report_for(proposal, block.as_ref());
    let record = match block {
        Some(block) => ExpertCommitRecord::new_with_inputs(
            proposal.clone(),
            report.digest(),
            7,
            gate_config(),
            report.encode_canonical(),
            block,
        ),
        None => ExpertCommitRecord::new_with_evidence(
            proposal.clone(),
            report.digest(),
            7,
            gate_config(),
            report.encode_canonical(),
        ),
    }
    .unwrap();
    let event = LedgerEvent::ExpertStructuralCommitted {
        project: project(),
        op_kind: ledger_op_kind(proposal.op().kind()),
        proposal_id: *proposal.proposal_id().as_bytes(),
        operation_id: *proposal.operation_id().as_bytes(),
        evidence_digest: *record.evidence_digest(),
        record: record.encode_canonical(),
    };
    let mut forged = history.to_vec();
    forged.push(CommittedEvent {
        index: CommitIndex(history.last().map_or(1, |last| last.index.0 + 1)),
        event,
    });
    forged
}

fn replay_error(result: Result<PtrRuntime, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        Err(other) => panic!("expected an expert refusal, got {other:?}"),
        Ok(_) => panic!("a history that must not replay replayed"),
    }
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

fn other_base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-b").unwrap(),
        BaseModelDigest::new([8; 32]).unwrap(),
        1,
    )
    .unwrap()
}

#[test]
fn an_attested_birth_is_admitted_and_its_block_names_the_attestor_the_run_and_the_signature() {
    let mut rt = runtime();
    let proposal = birth("e");
    let attested = alpha_attests(&proposal, 1);
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
    assert!(named.contains(&(attested.signature_identity(), Generation(1))));
    assert!(
        !named
            .iter()
            .any(|(identity, _)| identity == ATTESTATION_POLICY_IDENTITY),
        "an optional attestation records no requirement"
    );
}

#[test]
fn an_attestation_that_binds_another_proposal_artifact_base_model_or_result_is_refused() {
    let mut rt = runtime();
    let proposal = birth("e");
    let other = birth("f");
    let events = rt.committed_events().len();
    let claim = |proposal_id, artifact, base: BaseModelRef, results| {
        ExpertAttestationClaim::new(
            "alpha",
            Generation(1),
            [1; 32],
            proposal_id,
            artifact,
            base,
            results,
        )
        .unwrap()
        .sign(&alpha())
    };
    let right_id = *proposal.proposal_id().as_bytes();
    let right_artifact = expert_evidence_artifact_digest(proposal.op());
    let right_results = vec![(PodId::from("e"), Generation(1))];
    // Each claim is validly signed and wrong in exactly one binding.
    assert_eq!(
        expert_error(admit_attested(
            &mut rt,
            &proposal,
            &claim(
                *other.proposal_id().as_bytes(),
                right_artifact,
                base(),
                right_results.clone()
            )
        )),
        ExpertAdmissionError::AttestationProposalMismatch
    );
    assert_eq!(
        expert_error(admit_attested(
            &mut rt,
            &proposal,
            &claim(
                right_id,
                expert_evidence_artifact_digest(other.op()),
                base(),
                right_results.clone()
            )
        )),
        ExpertAdmissionError::AttestationArtifactMismatch
    );
    assert_eq!(
        expert_error(admit_attested(
            &mut rt,
            &proposal,
            &claim(
                right_id,
                right_artifact,
                other_base(),
                right_results.clone()
            )
        )),
        ExpertAdmissionError::AttestationBaseModelMismatch
    );
    assert_eq!(
        expert_error(admit_attested(
            &mut rt,
            &proposal,
            &claim(
                right_id,
                right_artifact,
                base(),
                vec![(PodId::from("e"), Generation(2))]
            )
        )),
        ExpertAdmissionError::AttestationResultMismatch
    );
    assert_eq!(
        expert_error(admit_attested(
            &mut rt,
            &proposal,
            &claim(
                right_id,
                right_artifact,
                base(),
                vec![(PodId::from("f"), Generation(1))]
            )
        )),
        ExpertAdmissionError::AttestationResultMismatch
    );
    assert_eq!(rt.committed_events().len(), events);
    assert!(rt
        .expert_registry()
        .get(&project(), &PodId::from("e"))
        .is_none());
}

#[test]
fn a_self_invented_attestor_or_a_forged_signature_is_refused_with_a_typed_error() {
    let mut rt = runtime();
    let proposal = birth("e");
    let events = rt.committed_events().len();
    // An attestor nobody registered, an epoch nobody registered, and a registered name signed
    // with a key of the caller's own.
    for (attestation, expected) in [
        (
            signed(&proposal, "invented", 1, &rogue(), 1),
            ExpertAdmissionError::AttestorUnknown {
                attestor: "invented".to_owned(),
                epoch: Generation(1),
            },
        ),
        (
            signed(&proposal, "alpha", 3, &alpha(), 2),
            ExpertAdmissionError::AttestorUnknown {
                attestor: "alpha".to_owned(),
                epoch: Generation(3),
            },
        ),
        (
            signed(&proposal, "alpha", 1, &rogue(), 3),
            ExpertAdmissionError::AttestorKeyMismatch {
                attestor: "alpha".to_owned(),
                epoch: Generation(1),
            },
        ),
    ] {
        assert_eq!(
            expert_error(admit_attested(&mut rt, &proposal, &attestation)),
            expected
        );
    }
    // The right key id with a signature that is not the attestor's, and a flipped bit.
    let genuine = alpha_attests(&proposal, 4);
    let mut flipped = *genuine.signature();
    flipped[0] ^= 1;
    let tampered =
        ExpertAttestation::from_parts(genuine.claim().clone(), *genuine.key_id(), flipped);
    let forged_by_rogue = {
        let rogue_signed = signed(&proposal, "alpha", 1, &rogue(), 4);
        ExpertAttestation::from_parts(
            genuine.claim().clone(),
            key_id(&alpha()),
            *rogue_signed.signature(),
        )
    };
    for attestation in [tampered, forged_by_rogue] {
        assert_eq!(
            expert_error(admit_attested(&mut rt, &proposal, &attestation)),
            ExpertAdmissionError::AttestationSignatureInvalid
        );
    }
    assert_eq!(rt.committed_events().len(), events);
    // The genuine one is admitted.
    admit_attested(&mut rt, &proposal, &genuine).unwrap();
}

/// The attacker of the re-review: a caller who holds only an admission grant assembles a gate
/// fence grant from `binding()` (the types allow it) to make its own key trusted. There is no API
/// to try it with: the trusted set is fixed at construction (`PtrRuntime::expert_attestors` is a
/// shared reference, `ExpertAttestorRegistry::register` needs `&mut` of a registry the runtime
/// does not hand out), so the forged grant opens nothing and the self-signed attestation is
/// refused as an unknown attestor.
#[test]
fn a_caller_holding_an_admission_grant_cannot_make_its_own_attestor_key_trusted() {
    let mut rt = runtime();
    let admission = required(&rt);
    // The forged gate grant: same binding, other kind. It is constructible and carries gate
    // authority on its face ...
    let forged = ExpertFenceGrant::gate(
        project(),
        requirement(7, gate_config()),
        admission.binding().clone(),
    );
    assert_eq!(forged.authority(), ExpertFenceAuthority::Gate);
    // ... and nothing in the public API takes it to change the registry; the registry is only
    // readable.
    let proposal = birth("e");
    let events = rt.committed_events().len();
    let attestation = signed(&proposal, "mine", 1, &rogue(), 1);
    assert_eq!(
        expert_error(admit_with(&mut rt, &proposal, &attestation, &admission)),
        ExpertAdmissionError::AttestorUnknown {
            attestor: "mine".to_owned(),
            epoch: Generation(1)
        }
    );
    // The same for an attestor name the host trusts, signed by the caller's key.
    let impostor = signed(&proposal, "alpha", 1, &rogue(), 2);
    assert_eq!(
        expert_error(admit_with(&mut rt, &proposal, &impostor, &admission)),
        ExpertAdmissionError::AttestorKeyMismatch {
            attestor: "alpha".to_owned(),
            epoch: Generation(1)
        }
    );
    assert!(rt.expert_attestors().key("mine", Generation(1)).is_none());
    assert_eq!(
        rt.expert_attestors().key("alpha", Generation(1)),
        Some(&alpha().verifying_key())
    );
    assert_eq!(rt.committed_events().len(), events);
    // The host's own attestor is admitted.
    admit_with(&mut rt, &proposal, &alpha_attests(&proposal, 3), &admission).unwrap();
}

#[test]
fn the_registry_validates_names_and_never_changes_an_epochs_key() {
    let key = rogue().verifying_key();
    let mut registry = registry();
    assert!(matches!(
        registry.register("alpha", Generation(0), key),
        Err(ExpertAdmissionError::InvalidRecord(_))
    ));
    for bad in ["", "Alpha", "al:pha", "al pha", &"a".repeat(65)] {
        assert!(
            matches!(
                registry.register(bad, Generation(1), key),
                Err(ExpertAdmissionError::InvalidAttestorName { .. })
            ),
            "{bad:?}"
        );
        assert!(matches!(
            ExpertAttestationClaim::for_proposal(bad, Generation(1), [1; 32], &birth("e")),
            Err(ExpertAdmissionError::InvalidAttestorName { .. })
        ));
    }
    // An epoch's key never changes.
    assert!(matches!(
        registry.register("alpha", Generation(1), key),
        Err(ExpertAdmissionError::InvalidRecord(_))
    ));
    registry
        .register("ok-name_1.x", Generation(1), key)
        .unwrap();
    registry
        .register(&"a".repeat(64), Generation(1), key)
        .unwrap();
}

#[test]
fn a_grant_that_requires_attestation_refuses_an_unattested_artifact_producing_admission() {
    let mut rt = runtime();
    let proposal = birth("e");
    let required = required(&rt);
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
    // The attested one goes through, and records the requirement.
    admit_with(&mut rt, &proposal, &alpha_attests(&proposal, 3), &required).unwrap();
    assert_eq!(route(&rt, "e"), Ok(()));
    assert!(rt
        .expert_registry()
        .dependencies()
        .recorded_inputs("proj", "e", Generation(1))
        .unwrap()
        .any(|(identity, _)| identity == ATTESTATION_POLICY_IDENTITY));
}

#[test]
fn operations_that_produce_no_artifact_are_neither_attested_nor_blocked_by_a_required_grant() {
    let mut rt = runtime();
    for (pod, run) in [("e", 1), ("f", 2), ("g", 3)] {
        let proposal = birth(pod);
        let required = required(&rt);
        admit_with(
            &mut rt,
            &proposal,
            &alpha_attests(&proposal, run),
            &required,
        )
        .unwrap();
    }
    let events = rt.committed_events().len();
    let required = required(&rt);
    let admit = |rt: &mut PtrRuntime, op: StructuralOp| {
        let proposal = ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(1),
            op,
            RationaleDigest::new([7; 32]).unwrap(),
        )
        .unwrap();
        let report = report_for(&proposal, None);
        rt.admit_expert_proposal(&proposal, &report, &required)
    };
    let op = StructuralOp::Freeze {
        target: target_of(&rt, "e"),
    };
    admit(&mut rt, op).unwrap();
    let op = StructuralOp::Retire {
        target: target_of(&rt, "f"),
        reason: RetiredReason::Superseded,
    };
    admit(&mut rt, op).unwrap();
    let op = StructuralOp::Prune {
        targets: vec![target_of(&rt, "g")],
    };
    admit(&mut rt, op).unwrap();
    assert_eq!(rt.committed_events().len(), events + 3);
    // Nothing was consumed: attesting a Freeze is refused, not given a run id.
    let frozen = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Retire {
            target: target_of(&rt, "e"),
            reason: RetiredReason::Superseded,
        },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    assert!(
        ExpertAttestationClaim::for_proposal("alpha", Generation(1), [9; 32], &frozen).is_err()
    );
}

#[test]
fn the_recorded_requirement_survives_a_restart_with_a_looser_grant() {
    let dir = std::env::temp_dir().join(format!(
        "ptr-attestation-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("ledger");
    let mut rt =
        PtrRuntime::open_durable_with_attestors(PtrConfig::default(), &path, registry()).unwrap();
    prime(&mut rt);
    let first = birth("e");
    let required = required(&rt);
    admit_with(&mut rt, &first, &alpha_attests(&first, 1), &required).unwrap();
    drop(rt);

    // The restart rebuilds the grant without the requirement.
    let mut reopened =
        PtrRuntime::open_durable_with_attestors(PtrConfig::default(), &path, registry()).unwrap();
    prime(&mut reopened);
    assert_eq!(route(&reopened, "e"), Ok(()));
    let events = reopened.committed_events().len();
    let loose = grant(&reopened);
    assert!(!loose.requires_attestation());
    let second = birth("f");
    assert_eq!(
        expert_error(admit_plain(&mut reopened, &second)),
        ExpertAdmissionError::AttestationRequired
    );
    let with_block = ExpertInputs::default();
    let report = report_for(&second, Some(&with_block));
    assert_eq!(
        expert_error(
            reopened.admit_expert_proposal_with_inputs(&second, &report, &loose, with_block)
        ),
        ExpertAdmissionError::AttestationRequired
    );
    assert_eq!(reopened.committed_events().len(), events);
    // An attested admission under the loose grant still works.
    admit_with(&mut reopened, &second, &alpha_attests(&second, 2), &loose).unwrap();
    drop(reopened);

    // A ledger without the attestor registry does not open: it fails closed.
    assert!(matches!(
        PtrRuntime::open_durable(PtrConfig::default(), &path),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::AttestorUnknown { .. }
        ))
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unattested_artifact_producing_record_does_not_replay_once_the_ledger_requires_attestation() {
    let mut rt = runtime();
    let first = birth("e");
    let required = required(&rt);
    admit_with(&mut rt, &first, &alpha_attests(&first, 1), &required).unwrap();
    let history: Vec<CommittedEvent> = rt.committed_events().to_vec();
    replay(&history).unwrap();

    // A record appended by a writer that bypasses the API: no inputs, an empty block, and a
    // record that carries the marker but no attestation.
    let second = birth("f");
    assert_eq!(
        replay_error(replay(&forged_history(&history, &second, None))),
        ExpertAdmissionError::AttestationRequired
    );
    assert_eq!(
        replay_error(replay(&forged_history(
            &history,
            &second,
            Some(ExpertInputs::default())
        ))),
        ExpertAdmissionError::AttestationRequired
    );
    let marker = ExpertInputRef::new(ATTESTATION_POLICY_IDENTITY, Generation(1), {
        let genuine = alpha_attests(&second, 5)
            .attested_inputs(&ExpertInputs::default(), &required)
            .unwrap();
        *genuine
            .training()
            .iter()
            .find(|input| input.identity() == ATTESTATION_POLICY_IDENTITY)
            .unwrap()
            .digest()
    })
    .unwrap();
    assert_eq!(
        replay_error(replay(&forged_history(
            &history,
            &second,
            Some(ExpertInputs::new(vec![marker], Vec::new()).unwrap())
        ))),
        ExpertAdmissionError::AttestationRequired
    );
    // Before any record carries the requirement an unattested record replays: the requirement
    // is the ledger's, not a default.
    let mut loose = runtime();
    admit_plain(&mut loose, &first).unwrap();
    replay(loose.committed_events()).unwrap();
    // A non-producing record is not attested and replays under the requirement.
    let freeze = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Freeze {
            target: target_of(&rt, "e"),
        },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    replay(&forged_history(&history, &freeze, None)).unwrap();
}

#[test]
fn replay_re_verifies_the_signature_and_the_bindings_of_a_recorded_attestation() {
    let mut rt = runtime();
    let first = birth("e");
    admit_attested(&mut rt, &first, &alpha_attests(&first, 1)).unwrap();
    let history: Vec<CommittedEvent> = rt.committed_events().to_vec();
    replay(&history).unwrap();

    // A genuine attestation for `f` carried by a record of another artifact, another pod and
    // another base model: the signed payload no longer matches the record.
    let genuine_proposal = birth("f");
    let genuine = alpha_attests(&genuine_proposal, 2);
    let block = genuine
        .attested_inputs(&ExpertInputs::default(), &grant(&rt))
        .unwrap();
    replay(&forged_history(
        &history,
        &genuine_proposal,
        Some(block.clone()),
    ))
    .unwrap();
    let other_artifact = birth("g");
    assert_eq!(
        replay_error(replay(&forged_history(
            &history,
            &other_artifact,
            Some(block.clone())
        ))),
        ExpertAdmissionError::AttestationSignatureInvalid
    );
    let other_base_spec = ExpertPodSpec::new(
        project(),
        PodId::from("f"),
        ManifestDigest::new([b'f'; 32]).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(b'f'),
        },
        other_base(),
        Generation(1),
    )
    .unwrap();
    let other_base_proposal = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Birth {
            result: other_base_spec,
        },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        replay_error(replay(&forged_history(
            &history,
            &other_base_proposal,
            Some(block.clone())
        ))),
        ExpertAdmissionError::AttestationSignatureInvalid
    );
    // The same history replayed against a registry that holds another key for the epoch, or
    // none, is refused.
    let mut other_keys = ExpertAttestorRegistry::new();
    other_keys
        .register("alpha", Generation(1), rogue().verifying_key())
        .unwrap();
    assert!(matches!(
        PtrRuntime::replay_with_attestors(PtrConfig::default(), &history, other_keys),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::AttestorKeyMismatch { .. }
        ))
    ));
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &history),
        Err(RuntimeError::Expert(
            ExpertAdmissionError::AttestorUnknown { .. }
        ))
    ));
}

#[test]
fn a_run_id_is_single_use_across_roles_generations_attestors_the_plain_path_and_a_replay() {
    let mut rt = runtime();
    let first = birth("e");
    let attested = alpha_attests(&first, 1);
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
        &alpha_attests(&second, 1),
    )));
    reused(expert_error(admit_attested(
        &mut rt,
        &second,
        &signed(&second, "beta", 1, &beta(), 1),
    )));
    // The plain path naming the consumed run, in any role and at any generation.
    let grant_now = grant(&rt);
    let events = rt.committed_events().len();
    for (generation, in_training) in [(1, true), (1, false), (2, true), (2, false), (7, false)] {
        let run =
            ExpertInputRef::new(attested.run_identity(), Generation(generation), [1; 32]).unwrap();
        let block = if in_training {
            ExpertInputs::new(vec![run], Vec::new())
        } else {
            ExpertInputs::new(Vec::new(), vec![run])
        }
        .unwrap();
        let report = report_for(&second, Some(&block));
        reused(expert_error(rt.admit_expert_proposal_with_inputs(
            &second, &report, &grant_now, block,
        )));
    }
    assert_eq!(rt.committed_events().len(), events);
    assert!(rt
        .expert_registry()
        .get(&project(), &PodId::from("f"))
        .is_none());

    // The consumed run is derived from the history: a runtime rebuilt by replay still refuses it.
    let history: Vec<CommittedEvent> = rt.committed_events().to_vec();
    let mut replayed = replay(&history).unwrap();
    prime(&mut replayed);
    reused(expert_error(admit_attested(
        &mut replayed,
        &second,
        &alpha_attests(&second, 1),
    )));
    // A fresh run of the same attestor is admitted.
    admit_attested(&mut replayed, &second, &alpha_attests(&second, 2)).unwrap();

    // A donor runtime admits another expert under the same run; its record cannot be replayed
    // after the first.
    let mut donor = runtime();
    let donated = birth("d");
    admit_attested(
        &mut donor,
        &donated,
        &signed(&donated, "beta", 1, &beta(), 1),
    )
    .unwrap();
    let record = donor
        .committed_events()
        .iter()
        .rev()
        .find(|committed| {
            matches!(
                committed.event,
                LedgerEvent::ExpertStructuralCommitted { .. }
            )
        })
        .expect("the donor's record")
        .event
        .clone();
    let mut forged = history.clone();
    forged.push(CommittedEvent {
        index: CommitIndex(history.last().map_or(1, |last| last.index.0 + 1)),
        event: record,
    });
    reused(replay_error(replay(&forged)));
}

#[test]
fn the_reserved_identities_are_refused_on_the_plain_path_and_in_an_attested_block() {
    let mut rt = runtime();
    let proposal = birth("e");
    let events = rt.committed_events().len();
    let grant_now = grant(&rt);
    let plain = |rt: &mut PtrRuntime, identity: String, digest: u8| {
        let input = ExpertInputRef::new(identity, Generation(1), [digest; 32]).unwrap();
        let block = ExpertInputs::new(vec![input], Vec::new()).unwrap();
        let report = report_for(&proposal, Some(&block));
        expert_error(rt.admit_expert_proposal_with_inputs(&proposal, &report, &grant_now, block))
    };
    // A run nobody consumed is still reserved; so are an attestor and a signature of the
    // caller's own, and the marker.
    assert!(matches!(
        plain(&mut rt, format!("run:{}", "ab".repeat(32)), 0xab),
        ExpertAdmissionError::ReservedInputIdentity { .. }
    ));
    assert!(matches!(
        plain(&mut rt, "attestor:alpha".to_owned(), 3),
        ExpertAdmissionError::AttestationMalformed(_)
    ));
    assert!(matches!(
        plain(&mut rt, format!("attestation:{}", "cd".repeat(64)), 4),
        ExpertAdmissionError::AttestationMalformed(_)
    ));
    assert_eq!(
        plain(&mut rt, ATTESTATION_POLICY_IDENTITY.to_owned(), 5),
        ExpertAdmissionError::AttestationRequired
    );
    // In the evaluation role too.
    let input = ExpertInputRef::new("attestor:alpha", Generation(1), [3; 32]).unwrap();
    let block = ExpertInputs::new(Vec::new(), vec![input]).unwrap();
    let report = report_for(&proposal, Some(&block));
    assert!(matches!(
        expert_error(rt.admit_expert_proposal_with_inputs(&proposal, &report, &grant_now, block)),
        ExpertAdmissionError::ReservedInputIdentity { .. }
    ));
    // And an attested admission refuses a block that names one.
    let own = ExpertInputRef::new("attestor:smuggled", Generation(1), [3; 32]).unwrap();
    let block = ExpertInputs::new(vec![own], Vec::new()).unwrap();
    assert!(matches!(
        alpha_attests(&proposal, 1).attested_inputs(&block, &grant_now),
        Err(ExpertAdmissionError::ReservedInputIdentity { .. })
    ));
    assert_eq!(rt.committed_events().len(), events);
}

#[test]
fn a_proposal_queued_across_the_revocation_of_its_attestor_is_refused_and_a_fresh_one_is_not() {
    let mut rt = runtime();
    let queued = birth("e");
    let attested = alpha_attests(&queued, 1);
    let grant_now = grant(&rt);
    let block = attested
        .attested_inputs(&ExpertInputs::default(), &grant_now)
        .unwrap();
    let report = report_for(&queued, Some(&block));
    revoke(&mut rt, "alpha", 1, &alpha());
    let refused = rt.admit_attested_expert_proposal(
        &queued,
        &report,
        &grant_now,
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
    admit_attested(&mut rt, &fresh, &signed(&fresh, "beta", 1, &beta(), 2)).unwrap();
    assert_eq!(route(&rt, "g"), Ok(()));
}

#[test]
fn the_revocation_of_an_attestor_fences_what_it_attested_for_good_and_a_new_epoch_retrains() {
    let mut rt = runtime();
    let attested_pod = birth("e");
    admit_attested(&mut rt, &attested_pod, &alpha_attests(&attested_pod, 1)).unwrap();
    let other_pod = birth("w");
    admit_attested(
        &mut rt,
        &other_pod,
        &signed(&other_pod, "beta", 1, &beta(), 2),
    )
    .unwrap();
    revoke(&mut rt, "alpha", 1, &alpha());

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
        &signed(&retrained, "alpha", 2, &alpha2(), 3),
    )
    .unwrap();
    assert_eq!(route(&rt, "r"), Ok(()));
    let again = birth("s");
    assert!(matches!(
        expert_error(admit_attested(&mut rt, &again, &alpha_attests(&again, 4))),
        ExpertAdmissionError::InputRevoked { .. }
    ));
}

#[test]
fn a_descendant_of_an_attested_expert_is_fenced_for_good_when_its_attestor_is_revoked() {
    let mut rt = runtime();
    let first = birth("e");
    admit_attested(&mut rt, &first, &alpha_attests(&first, 1)).unwrap();
    let second = birth("w");
    admit_attested(&mut rt, &second, &signed(&second, "beta", 1, &beta(), 2)).unwrap();
    let bystander = birth("x");
    admit_attested(
        &mut rt,
        &bystander,
        &signed(&bystander, "beta", 1, &beta(), 3),
    )
    .unwrap();
    let merge = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        StructuralOp::Merge {
            sources: vec![target_of(&rt, "e"), target_of(&rt, "w")],
            result: spec("m"),
        },
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    admit_plain(&mut rt, &merge).unwrap();
    assert_eq!(route(&rt, "m"), Ok(()));

    revoke(&mut rt, "alpha", 1, &alpha());
    assert_eq!(
        route(&rt, "m"),
        Err(ptr_runtime::expert::RouteRefusal::Fenced),
        "the descendant rests on the revoked attestor through its source"
    );
    assert_eq!(
        rt.expert_registry()
            .fence_cause(&project(), &PodId::from("m")),
        Some(ExpertFenceCause::InputRevoked)
    );
    assert_eq!(
        route(&rt, "x"),
        Ok(()),
        "an expert of the other attestor is untouched"
    );
}

#[test]
fn a_registered_key_is_checked_strictly_and_the_attestation_payload_is_domain_separated() {
    // The key id of a key is a domain-separated digest, not the key bytes.
    assert_ne!(&key_id(&alpha()), alpha().verifying_key().as_bytes());
    assert_ne!(key_id(&alpha()), key_id(&beta()));
    // Claims with a zero epoch, a zero run id or no result are not claims.
    let proposal = birth("e");
    let claim = |epoch, run, results: Vec<(PodId, Generation)>| {
        ExpertAttestationClaim::new(
            "alpha",
            Generation(epoch),
            [run; 32],
            *proposal.proposal_id().as_bytes(),
            expert_evidence_artifact_digest(proposal.op()),
            base(),
            results,
        )
    };
    let one = vec![(PodId::from("e"), Generation(1))];
    assert!(claim(0, 1, one.clone()).is_err());
    assert!(claim(1, 0, one.clone()).is_err());
    assert!(claim(1, 1, Vec::new()).is_err());
    assert!(claim(1, 1, one).is_ok());
}

#[test]
fn the_recorded_requirement_survives_a_compacted_snapshot() {
    let mut rt = runtime();
    let first = birth("e");
    let required = required(&rt);
    admit_with(&mut rt, &first, &alpha_attests(&first, 1), &required).unwrap();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    prime(&mut restored);
    assert_eq!(route(&restored, "e"), Ok(()));
    // The snapshot holds the dependency index, marker included: the looser grant of the restored
    // host still cannot admit an unattested artifact-producing record.
    let second = birth("f");
    assert_eq!(
        expert_error(admit_plain(&mut restored, &second)),
        ExpertAdmissionError::AttestationRequired
    );
}
