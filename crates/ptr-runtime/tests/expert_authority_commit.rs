//! Commit-time authority: a grant is a snapshot, so the runtime re-reads the authority behind it
//! immediately before the append. A grant minted from a session, key, principal and grant
//! generation that are no longer current is refused with a typed error and nothing is appended,
//! for the proposer's admission grant, the gate's readmission grant and the runtime's fence grant.
//! Also the use-time route admission.

mod expert_support;

use ed25519_dalek::SigningKey;
use expert_support::{
    binding_at, current_binding, identity, identity_for, peer, prime, requirement,
    BASE_MODEL_DIGEST, EVAL_MANIFEST, EXPIRES_AT, ISSUED_AT, KEY_ID, NOW, SESSION,
};
use ptr_config::PtrConfig;
use ptr_ledger::ExpertFenceCause;
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest,
    expert_readmission_subject_ids, ExpertAdmissionError, ExpertAdmissionGrant, ExpertFenceGrant,
    ExpertFenceRequest, ExpertGrantBinding, ExpertGrantGeneration, ExpertGrantKind, RouteRefusal,
};
use ptr_runtime::{ManifestAuthorityRegistry, PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId, SessionId, Timestamp};

const EPOCH: u64 = 7;
const QUARANTINE: ExpertFenceCause = ExpertFenceCause::AuthorityQuarantine;

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new(BASE_MODEL_DIGEST).unwrap(),
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

fn gate_digest() -> [u8; 32] {
    gate().config_digest().unwrap()
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

fn spec(pod: &str) -> ExpertPodSpec {
    let mut digest = [1u8; 32];
    digest[0] = pod.as_bytes()[0];
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new(digest).unwrap(),
        ExpertLevel::Token {
            layer: 3,
            index: u16::from(pod.as_bytes()[0]),
        },
        base(),
        Generation(1),
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

fn admit_as(
    rt: &mut PtrRuntime,
    op: StructuralOp,
    binding: ExpertGrantBinding,
) -> Result<CommitIndex, RuntimeError> {
    let proposal = proposal(op);
    let report = bind(GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    ));
    let grant = ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(EPOCH, gate_digest()),
        binding,
    );
    rt.admit_expert_proposal(&proposal, &report, &grant)
}

fn birth(rt: &mut PtrRuntime, pod: &str) {
    let binding = current_binding(rt, ExpertGrantKind::Admission);
    admit_as(rt, StructuralOp::Birth { result: spec(pod) }, binding).unwrap();
}

fn request(pod: &str, version: u64) -> ExpertFenceRequest {
    ExpertFenceRequest {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(1),
        expected_state_version: version,
        cause: QUARANTINE,
        revocation: None,
    }
}

fn readmission_report(request: &ExpertFenceRequest, manifest: [u8; 32]) -> BoundGateReport {
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        &request.project,
        &request.pod,
        request.generation,
        request.expected_state_version,
        request.expected_state_version + 1,
        request.cause,
    );
    bind(GateSubject::new(
        proposal_id,
        operation_id,
        manifest,
        BASE_MODEL_DIGEST,
        EVAL_MANIFEST,
    ))
}

fn manifest_of(rt: &PtrRuntime, pod: &str) -> [u8; 32] {
    *rt.expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap()
        .manifest_digest
        .as_bytes()
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

fn expert_error(result: Result<CommitIndex, RuntimeError>) -> ExpertAdmissionError {
    match result {
        Err(RuntimeError::Expert(error)) => error,
        other => panic!("expected an expert refusal, got {other:?}"),
    }
}

// ---- the three commit paths ---------------------------------------------------------------

/// A grant kind and the commit it authorizes.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// The proposer's admission of a birth.
    Admission,
    /// The runtime's fence of `x`.
    Fence,
    /// The gate's readmission of the fenced `x`.
    Gate,
}

const PATHS: [Path; 3] = [Path::Admission, Path::Fence, Path::Gate];

impl Path {
    fn kind(self) -> ExpertGrantKind {
        match self {
            Self::Admission => ExpertGrantKind::Admission,
            Self::Fence => ExpertGrantKind::Fence,
            Self::Gate => ExpertGrantKind::Gate,
        }
    }

    /// A runtime in which this path's commit is one valid step away.
    fn setup(self) -> PtrRuntime {
        let mut rt = runtime();
        birth(&mut rt, "x");
        if matches!(self, Self::Gate) {
            let binding = current_binding(&rt, ExpertGrantKind::Fence);
            rt.fence_expert(
                &request("x", 1),
                &ExpertFenceGrant::revocation(project(), binding),
            )
            .unwrap();
        }
        rt
    }

    /// Commit this path's step on `rt` under a grant carrying `binding`.
    fn commit(
        self,
        rt: &mut PtrRuntime,
        binding: ExpertGrantBinding,
    ) -> Result<CommitIndex, RuntimeError> {
        match self {
            Self::Admission => admit_as(rt, StructuralOp::Birth { result: spec("n") }, binding),
            Self::Fence => rt.fence_expert(
                &request("x", 1),
                &ExpertFenceGrant::revocation(project(), binding),
            ),
            Self::Gate => {
                let req = request("x", 2);
                let report = readmission_report(&req, manifest_of(rt, "x"));
                rt.readmit_expert(
                    &req,
                    &report,
                    &ExpertFenceGrant::gate(project(), requirement(EPOCH, gate_digest()), binding),
                )
            }
        }
    }

    fn fresh_binding(self, rt: &PtrRuntime) -> ExpertGrantBinding {
        current_binding(rt, self.kind())
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    SessionRevoked,
    ContextExpired,
    ContextNotYetIssued,
    ContextDigestWrong,
    PolicyKeyRevoked,
    PolicyKeyNeverInstalled,
    PeerDeadmitted,
    TransportPeerWithdrawn,
    GenerationSuperseded,
}

const FAULTS: [Fault; 9] = [
    Fault::SessionRevoked,
    Fault::ContextExpired,
    Fault::ContextNotYetIssued,
    Fault::ContextDigestWrong,
    Fault::PolicyKeyRevoked,
    Fault::PolicyKeyNeverInstalled,
    Fault::PeerDeadmitted,
    Fault::TransportPeerWithdrawn,
    Fault::GenerationSuperseded,
];

impl Fault {
    /// Mint the grant's binding, then let the authority behind it change; the binding to commit
    /// with and the refusal it must meet.
    fn inflict(
        self,
        rt: &mut PtrRuntime,
        path: Path,
    ) -> (ExpertGrantBinding, ExpertAdmissionError) {
        let minted = path.fresh_binding(rt);
        match self {
            Self::SessionRevoked => {
                rt.revoke_identity_session(SessionId(SESSION.into()), "operator revoked")
                    .unwrap();
                (minted, ExpertAdmissionError::GrantSessionRevoked)
            }
            Self::ContextExpired => (
                minted.at(Timestamp(EXPIRES_AT)),
                ExpertAdmissionError::GrantIdentityInvalid,
            ),
            Self::ContextNotYetIssued => (
                minted.at(Timestamp(ISSUED_AT - 1)),
                ExpertAdmissionError::GrantIdentityInvalid,
            ),
            Self::ContextDigestWrong => {
                let mut forged = identity();
                forged.expires_at = Timestamp(EXPIRES_AT * 2);
                (
                    ExpertGrantBinding::new(forged, peer(), NOW, KEY_ID, minted.generation()),
                    ExpertAdmissionError::GrantIdentityInvalid,
                )
            }
            Self::PolicyKeyRevoked => {
                rt.revoke_policy_key(KEY_ID);
                (minted, ExpertAdmissionError::GrantPolicyKeyUntrusted)
            }
            Self::PolicyKeyNeverInstalled => (
                ExpertGrantBinding::new(
                    identity(),
                    peer(),
                    NOW,
                    "unknown-key",
                    minted.generation(),
                ),
                ExpertAdmissionError::GrantPolicyKeyUntrusted,
            ),
            Self::PeerDeadmitted => {
                // No principal is admitted any more.
                *rt.manifest_authority_registry().write().unwrap() =
                    ManifestAuthorityRegistry::default();
                (minted, ExpertAdmissionError::GrantPrincipalNotAdmitted)
            }
            Self::TransportPeerWithdrawn => {
                rt.withdraw_peer(&peer()).unwrap();
                (minted, ExpertAdmissionError::GrantPeerNotAdmitted)
            }
            Self::GenerationSuperseded => {
                let current = rt.supersede_expert_grant(path.kind()).unwrap();
                (
                    minted.clone(),
                    ExpertAdmissionError::GrantGenerationStale {
                        kind: path.kind(),
                        grant: minted.generation().get(),
                        current: current.get(),
                    },
                )
            }
        }
    }
}

#[test]
fn fresh_authority_commits_on_every_path() {
    for path in PATHS {
        let mut rt = path.setup();
        let before = rt.committed_events().len();
        let binding = path.fresh_binding(&rt);
        path.commit(&mut rt, binding)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
        assert_eq!(rt.committed_events().len(), before + 1, "{path:?}");
    }
}

#[test]
fn authority_that_changed_after_the_grant_was_minted_is_refused_and_appends_nothing() {
    for path in PATHS {
        for fault in FAULTS {
            let mut rt = path.setup();
            let before = rt.committed_events().len();
            let registry = rt.expert_registry().clone();
            let (binding, expected) = fault.inflict(&mut rt, path);
            // A session revocation is itself a committed record; count from after it.
            let before = if matches!(fault, Fault::SessionRevoked) {
                rt.committed_events().len()
            } else {
                before
            };
            assert_eq!(
                expert_error(path.commit(&mut rt, binding)),
                expected,
                "{path:?} {fault:?}"
            );
            assert_eq!(rt.committed_events().len(), before, "{path:?} {fault:?}");
            assert_eq!(rt.expert_registry(), &registry, "{path:?} {fault:?}");
        }
    }
}

#[test]
fn a_superseded_grant_generation_is_only_cured_by_minting_at_the_new_one() {
    for path in PATHS {
        let mut rt = path.setup();
        let old = path.fresh_binding(&rt);
        assert_eq!(old.generation(), ExpertGrantGeneration::INITIAL);
        let next = rt.supersede_expert_grant(path.kind()).unwrap();
        assert_eq!(next.get(), ExpertGrantGeneration::INITIAL.get() + 1);
        assert_eq!(rt.expert_grant_generation(path.kind()), next);
        assert!(matches!(
            expert_error(path.commit(&mut rt, old)),
            ExpertAdmissionError::GrantGenerationStale { .. }
        ));
        let renewed = path.fresh_binding(&rt);
        path.commit(&mut rt, renewed)
            .unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
    }
}

#[test]
fn grant_generations_are_per_kind() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    let admission = rt.expert_grant_generation(ExpertGrantKind::Admission);
    let fence = rt.expert_grant_generation(ExpertGrantKind::Fence);
    let gate = rt.expert_grant_generation(ExpertGrantKind::Gate);
    rt.supersede_expert_grant(ExpertGrantKind::Admission)
        .unwrap();
    assert_ne!(
        rt.expert_grant_generation(ExpertGrantKind::Admission),
        admission
    );
    assert_eq!(rt.expert_grant_generation(ExpertGrantKind::Fence), fence);
    assert_eq!(rt.expert_grant_generation(ExpertGrantKind::Gate), gate);
    // The fence grant minted before the admission grant was superseded is still current.
    rt.fence_expert(
        &request("x", 1),
        &ExpertFenceGrant::revocation(project(), binding_at(fence)),
    )
    .unwrap();
}

#[test]
fn grant_generations_are_host_memory_not_ledger_state() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    rt.supersede_expert_grant(ExpertGrantKind::Admission)
        .unwrap();
    let before = rt.committed_events().len();
    rt.supersede_expert_grant(ExpertGrantKind::Fence).unwrap();
    assert_eq!(rt.committed_events().len(), before, "nothing is appended");
    // A registry compared with a fresh replay of its ledger stays equal, and the replay stands at
    // the initial generations.
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_eq!(replayed.expert_registry(), rt.expert_registry());
    assert_eq!(
        replayed.expert_grant_generation(ExpertGrantKind::Admission),
        ExpertGrantGeneration::INITIAL
    );
}

#[test]
fn a_grant_for_another_principal_than_the_admitted_one_is_refused() {
    // A context for a subject the runtime never admitted has no standing, however valid it is.
    let mut rt = runtime();
    let mut stranger = identity_for("session-2", ISSUED_AT, EXPIRES_AT);
    stranger.subject = "stranger".into();
    stranger.identity_digest = stranger.canonical_digest();
    let binding = ExpertGrantBinding::new(
        stranger,
        peer(),
        NOW,
        KEY_ID,
        rt.expert_grant_generation(ExpertGrantKind::Admission),
    );
    let before = rt.committed_events().len();
    assert_eq!(
        expert_error(admit_as(
            &mut rt,
            StructuralOp::Birth { result: spec("a") },
            binding
        )),
        ExpertAdmissionError::GrantPrincipalNotAdmitted
    );
    assert_eq!(rt.committed_events().len(), before);
    // Admitting the context is what gives it standing.
    let mut other = identity_for("session-2", ISSUED_AT, EXPIRES_AT);
    other.subject = "second".into();
    other.identity_digest = other.canonical_digest();
    rt.admit_identity_context(&other, NOW).unwrap();
    let binding = ExpertGrantBinding::new(
        other,
        peer(),
        NOW,
        KEY_ID,
        rt.expert_grant_generation(ExpertGrantKind::Admission),
    );
    admit_as(&mut rt, StructuralOp::Birth { result: spec("a") }, binding).unwrap();
}

#[test]
fn a_policy_key_id_is_trusted_only_once_the_host_has_installed_it() {
    let mut rt = runtime();
    let binding = ExpertGrantBinding::new(
        identity(),
        peer(),
        NOW,
        "late-key",
        rt.expert_grant_generation(ExpertGrantKind::Admission),
    );
    assert_eq!(
        expert_error(admit_as(
            &mut rt,
            StructuralOp::Birth { result: spec("a") },
            binding.clone()
        )),
        ExpertAdmissionError::GrantPolicyKeyUntrusted
    );
    rt.install_policy_key("late-key", SigningKey::from_bytes(&[8; 32]).verifying_key());
    admit_as(&mut rt, StructuralOp::Birth { result: spec("a") }, binding).unwrap();
}

// ---- route admission -----------------------------------------------------------------------

fn target(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
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

fn admit_op(rt: &mut PtrRuntime, op: StructuralOp) {
    let binding = current_binding(rt, ExpertGrantKind::Admission);
    admit_as(rt, op, binding).unwrap();
}

fn fence(rt: &mut PtrRuntime, pod: &str, version: u64) {
    let binding = current_binding(rt, ExpertGrantKind::Fence);
    rt.fence_expert(
        &request(pod, version),
        &ExpertFenceGrant::revocation(project(), binding),
    )
    .unwrap();
}

fn route(rt: &PtrRuntime, pod: &str, generation: u64) -> Result<(), RouteRefusal> {
    rt.expert_route_admission(&project(), &PodId::from(pod), Generation(generation))
}

/// Serving `s`, frozen `f`, fenced `q`, retired `r`, superseded by the merge `a` and `b` into `m`.
fn populated(rt: &mut PtrRuntime) {
    for pod in ["s", "f", "q", "r", "a", "b"] {
        birth(rt, pod);
    }
    let freeze = StructuralOp::Freeze {
        target: target(rt, "f"),
    };
    admit_op(rt, freeze);
    fence(rt, "q", 1);
    let retire = StructuralOp::Retire {
        target: target(rt, "r"),
        reason: RetiredReason::Superseded,
    };
    admit_op(rt, retire);
    let merge = StructuralOp::Merge {
        sources: vec![target(rt, "a"), target(rt, "b")],
        result: spec("m"),
    };
    admit_op(rt, merge);
}

fn assert_route_answers(rt: &PtrRuntime) {
    assert_eq!(route(rt, "s", 1), Ok(()), "serving");
    assert_eq!(route(rt, "f", 1), Ok(()), "frozen");
    assert_eq!(route(rt, "m", 1), Ok(()), "the merge result");
    assert_eq!(route(rt, "q", 1), Err(RouteRefusal::Fenced), "fenced");
    assert_eq!(
        route(rt, "r", 1),
        Err(RouteRefusal::NotServing {
            state: ExpertState::Retired(RetiredReason::Superseded)
        }),
        "retired"
    );
    assert_eq!(
        route(rt, "a", 1),
        Err(RouteRefusal::NotServing {
            state: ExpertState::Superseded
        }),
        "superseded by a merge"
    );
    assert_eq!(
        route(rt, "s", 2),
        Err(RouteRefusal::GenerationMismatch {
            current: Generation(1)
        }),
        "a generation the expert is not at"
    );
    assert_eq!(
        route(rt, "q", 2),
        Err(RouteRefusal::GenerationMismatch {
            current: Generation(1)
        }),
        "the generation is checked before the state"
    );
    assert_eq!(route(rt, "nobody", 1), Err(RouteRefusal::UnknownExpert));
    assert_eq!(
        rt.expert_route_admission(
            &ProjectId::from("elsewhere"),
            &PodId::from("s"),
            Generation(1)
        ),
        Err(RouteRefusal::UnknownExpert),
        "another project's registry is not this one's"
    );
}

#[test]
fn a_route_is_admitted_only_to_a_serving_or_frozen_unfenced_expert_at_exactly_its_generation() {
    let mut rt = runtime();
    populated(&mut rt);
    assert_route_answers(&rt);
}

#[test]
fn a_readmitted_expert_is_routable_again_and_a_refenced_one_is_not() {
    let mut rt = runtime();
    birth(&mut rt, "x");
    assert_eq!(route(&rt, "x", 1), Ok(()));
    fence(&mut rt, "x", 1);
    assert_eq!(route(&rt, "x", 1), Err(RouteRefusal::Fenced));
    let req = request("x", 2);
    let report = readmission_report(&req, manifest_of(&rt, "x"));
    let binding = current_binding(&rt, ExpertGrantKind::Gate);
    rt.readmit_expert(
        &req,
        &report,
        &ExpertFenceGrant::gate(project(), requirement(EPOCH, gate_digest()), binding),
    )
    .unwrap();
    assert_eq!(route(&rt, "x", 1), Ok(()));
    fence(&mut rt, "x", 3);
    assert_eq!(route(&rt, "x", 1), Err(RouteRefusal::Fenced));
}

#[test]
fn route_admission_survives_a_replay_and_a_durable_reopen() {
    let mut rt = runtime();
    populated(&mut rt);
    let replayed = PtrRuntime::replay(PtrConfig::default(), rt.committed_events()).unwrap();
    assert_route_answers(&replayed);

    let dir = std::env::temp_dir().join(format!(
        "ptr-expert-route-{}-{}",
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
    populated(&mut durable);
    assert_route_answers(&durable);
    drop(durable);
    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_route_answers(&reopened);
    drop(reopened);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_grant_over_a_peer_the_policy_never_admitted_is_refused_and_a_readmitted_peer_commits() {
    let mut rt = runtime();
    let stranger = ptr_types::NodeId::from("never-admitted-peer");
    let binding = ExpertGrantBinding::new(
        identity(),
        stranger,
        NOW,
        KEY_ID,
        rt.expert_grant_generation(ExpertGrantKind::Admission),
    );
    let before = rt.committed_events().len();
    assert_eq!(
        expert_error(admit_as(
            &mut rt,
            StructuralOp::Birth { result: spec("a") },
            binding
        )),
        ExpertAdmissionError::GrantPeerNotAdmitted
    );
    assert_eq!(rt.committed_events().len(), before);
    // Withdrawn between mint and commit, then admitted again by a new policy: positive control.
    let minted = current_binding(&rt, ExpertGrantKind::Admission);
    rt.withdraw_peer(&peer()).unwrap();
    assert_eq!(
        expert_error(admit_as(
            &mut rt,
            StructuralOp::Birth { result: spec("a") },
            minted.clone()
        )),
        ExpertAdmissionError::GrantPeerNotAdmitted
    );
    prime(&mut rt);
    admit_as(&mut rt, StructuralOp::Birth { result: spec("a") }, minted).unwrap();
}
