//! MoE experts through the real admission path: a birth, a retirement and a reactivation are
//! structural proposals admitted by `PtrRuntime::admit_expert_proposal`, so the runtime's own
//! materialised `expert:{project}/{pod}` record is what answers an expert dependency's
//! liveness. Nothing here scripts an oracle.
//!
//! An expert `(layer, index)` is the pod `moe-l{layer}-e{index}` at token level
//! `Token { layer, index }`. Revoking it is a `Retire`; superseding it is a `Retire` followed
//! by a `Reactivate` of the retired generation, which commits generation + 1 of the same pod.

use super::lab::{fail, Res, PROJECT};
use ptr_ledger::{ExpertFenceCause, ExpertFenceRevocation};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_runtime::execution::AdmissionPolicy;
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest,
    expert_readmission_subject_ids, ExpertAdmissionGrant, ExpertFenceGrant, ExpertFenceRequest,
    ExpertGateRequirement, ExpertGrantBinding, ExpertGrantKind,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{
    AuthenticationLevel, CommitIndex, Generation, IdentityContext, NodeId, PodId, ProjectId,
    SessionId, Timestamp,
};

const EVAL_MANIFEST: [u8; 32] = [5; 32];
const KEY_ID: &str = "r010-policy-key";
const NOW: Timestamp = Timestamp(1_000);

/// The host authority behind a grant, installed on `runtime` (idempotent) and read back.
fn host_binding(runtime: &mut PtrRuntime, kind: ExpertGrantKind) -> Res<ExpertGrantBinding> {
    let mut identity = IdentityContext {
        subject: "r010-host".into(),
        issuer: "r010".into(),
        groups: Vec::new(),
        authentication_level: AuthenticationLevel::MultiFactor,
        session_id: SessionId("r010-session".into()),
        issued_at: Timestamp(1),
        expires_at: Timestamp(1_000_000),
        identity_digest: [0; 32],
    };
    identity.identity_digest = identity.canonical_digest();
    runtime.install_policy_key(
        KEY_ID,
        ed25519_dalek::SigningKey::from_bytes(&[7; 32]).verifying_key(),
    );
    runtime
        .admit_identity_context(&identity, NOW)
        .map_err(fail("admitting the host identity"))?;
    let peer = NodeId::from("r010-host-peer");
    let mut policy = AdmissionPolicy::new();
    policy
        .admit(
            peer.clone(),
            "r010-host",
            std::time::Duration::from_secs(60),
            Vec::new,
        )
        .map_err(fail("admitting the host peer"))?;
    runtime.install_admission_policy(policy);
    Ok(ExpertGrantBinding::new(
        identity,
        peer,
        NOW,
        KEY_ID,
        runtime.expert_grant_generation(kind),
    ))
}

fn requirement() -> Res<ExpertGateRequirement> {
    Ok(ExpertGateRequirement {
        gate_epoch: GATE_EPOCH,
        config_digest: gate()
            .config_digest()
            .map_err(fail("the gate configuration digest"))?,
        eval_manifest_digest: EVAL_MANIFEST,
    })
}

/// The gate epoch of the admission grant.
const GATE_EPOCH: u64 = 7;

fn project() -> ProjectId {
    ProjectId::from(PROJECT)
}

fn base() -> Res<BaseModelRef> {
    BaseModelRef::new(
        BaseModelId::new("layered-kv-base").map_err(fail("the base model id"))?,
        BaseModelDigest::new([9; 32]).map_err(fail("the base model digest"))?,
        1,
    )
    .map_err(fail("the base model ref"))
}

fn proposer() -> Res<ProposerRef> {
    ProposerRef::new(project(), PodId::from("meta"), Generation(1)).map_err(fail("the proposer"))
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn grant(runtime: &mut PtrRuntime) -> Res<ExpertAdmissionGrant> {
    Ok(ExpertAdmissionGrant::new(
        project(),
        proposer()?,
        requirement()?,
        host_binding(runtime, ExpertGrantKind::Admission)?,
    ))
}

/// The pod of expert `(layer, index)`.
pub fn pod_of(layer: usize, index: usize) -> PodId {
    PodId::from(format!("moe-l{layer}-e{index}").as_str())
}

/// The lifecycle identity of expert `(layer, index)`: `expert:{project}/{pod}`.
pub fn identity_of(layer: usize, index: usize) -> String {
    ptr_runtime::KvDependency::expert(&project(), &pod_of(layer, index), Generation(1), [0; 32])
        .identity
}

fn level_of(layer: usize, index: usize) -> Res<ExpertLevel> {
    Ok(ExpertLevel::Token {
        layer: u16::try_from(layer).map_err(fail("the layer"))?,
        index: u16::try_from(index).map_err(fail("the index"))?,
    })
}

fn spec_at(layer: usize, index: usize, generation: u64) -> Res<ExpertPodSpec> {
    let mut digest = [1u8; 32];
    digest[0] = layer as u8;
    digest[1] = index as u8;
    digest[2] = generation as u8;
    ExpertPodSpec::new(
        project(),
        pod_of(layer, index),
        ManifestDigest::new(digest).map_err(fail("the manifest digest"))?,
        level_of(layer, index)?,
        base()?,
        Generation(generation),
    )
    .map_err(fail("the expert pod spec"))
}

fn admit(runtime: &mut PtrRuntime, op: StructuralOp) -> Res<()> {
    let proposal = ExpertProposal::new(
        project(),
        proposer()?,
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).map_err(fail("the rationale digest"))?,
    )
    .map_err(fail("the proposal"))?;
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op())
            .ok_or_else(|| "the proposal names no expert".to_owned())?,
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]])
        .map_err(fail("the accuracy matrix"))?;
    let report: BoundGateReport = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .map_err(fail("the gate evaluation"))?
        .bind(subject);
    let grant = grant(runtime)?;
    runtime
        .admit_expert_proposal(&proposal, &report, &grant)
        .map(|_| ())
        .map_err(fail("admitting an expert proposal"))
}

/// The registry's current view of expert `(layer, index)` as an operand target.
fn target(runtime: &PtrRuntime, layer: usize, index: usize) -> Res<ExpertTarget> {
    let entry = runtime
        .expert_registry()
        .get(&project(), &pod_of(layer, index))
        .ok_or_else(|| format!("expert ({layer}, {index}) is not in the registry"))?
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: pod_of(layer, index),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .map_err(fail("the expert target"))
}

/// Birth expert `(layer, index)` at generation 1, `Serving`.
pub fn birth(runtime: &mut PtrRuntime, layer: usize, index: usize) -> Res<()> {
    admit(
        runtime,
        StructuralOp::Birth {
            result: spec_at(layer, index, 1)?,
        },
    )
}

/// Revoke expert `(layer, index)`: a committed `Retire(GateFailed)`.
pub fn retire(runtime: &mut PtrRuntime, layer: usize, index: usize) -> Res<()> {
    let op = StructuralOp::Retire {
        target: target(runtime, layer, index)?,
        reason: RetiredReason::GateFailed,
    };
    admit(runtime, op)
}

/// Supersede expert `(layer, index)`: a committed `Retire(GateFailed)` and then a
/// `Reactivate` of the retired generation, which commits generation + 1 of the same pod.
/// Returns the new generation.
pub fn supersede(runtime: &mut PtrRuntime, layer: usize, index: usize) -> Res<u64> {
    retire(runtime, layer, index)?;
    let retired = target(runtime, layer, index)?;
    let next = retired.generation().0 + 1;
    admit(
        runtime,
        StructuralOp::Reactivate {
            target: retired,
            result: Some(spec_at(layer, index, next)?),
        },
    )?;
    Ok(next)
}

/// The state, generation and state version the registry holds for expert `(layer, index)`.
pub fn state_of(
    runtime: &PtrRuntime,
    layer: usize,
    index: usize,
) -> Option<(ExpertState, u64, u64)> {
    runtime
        .expert_registry()
        .get(&project(), &pod_of(layer, index))
        .map(|entry| (entry.state, entry.generation.0, entry.state_version))
}

/// Fence expert `(layer, index)` through the real authority path: `fence_expert` under the
/// revocation authority's grant. The expert goes to `Fenced` at an unchanged generation and a
/// new fence epoch. An `InputRevoked` fence names `revocation`, which the ledger must already
/// hold committed.
pub fn fence(
    runtime: &mut PtrRuntime,
    layer: usize,
    index: usize,
    cause: ExpertFenceCause,
    revocation: Option<(&str, u64)>,
) -> Res<()> {
    let (_, generation, version) = state_of(runtime, layer, index)
        .ok_or_else(|| format!("expert ({layer}, {index}) is not in the registry"))?;
    let request = ExpertFenceRequest {
        project: project(),
        pod: pod_of(layer, index),
        generation: Generation(generation),
        expected_state_version: version,
        cause,
        revocation: revocation.map(|(subject, generation)| ExpertFenceRevocation {
            subject: subject.to_string(),
            generation: Generation(generation),
        }),
    };
    let binding = host_binding(runtime, ExpertGrantKind::Fence)?;
    runtime
        .fence_expert(&request, &ExpertFenceGrant::revocation(project(), binding))
        .map(|_| ())
        .map_err(fail("fencing an expert"))
}

/// Readmit the fenced expert `(layer, index)` through the gate authority path: a passed gate
/// report bound to this readmission. The expert returns to the state the fence left it in, at
/// the next fence epoch.
pub fn readmit(
    runtime: &mut PtrRuntime,
    layer: usize,
    index: usize,
    cause: ExpertFenceCause,
) -> Res<()> {
    let (_, generation, version) = state_of(runtime, layer, index)
        .ok_or_else(|| format!("expert ({layer}, {index}) is not in the registry"))?;
    let request = ExpertFenceRequest {
        project: project(),
        pod: pod_of(layer, index),
        generation: Generation(generation),
        expected_state_version: version,
        cause,
        revocation: None,
    };
    let (proposal_id, operation_id) = expert_readmission_subject_ids(
        &request.project,
        &request.pod,
        request.generation,
        version,
        version + 1,
        cause,
    );
    let entry = runtime
        .expert_registry()
        .get(&request.project, &request.pod)
        .ok_or_else(|| "readmitting an unknown expert".to_owned())?;
    let subject = GateSubject::new(
        proposal_id,
        operation_id,
        *entry.manifest_digest.as_bytes(),
        *entry.base_model.digest().as_bytes(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]])
        .map_err(fail("the accuracy matrix"))?;
    let report: BoundGateReport = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .map_err(fail("the gate evaluation"))?
        .bind(subject);
    let binding = host_binding(runtime, ExpertGrantKind::Gate)?;
    runtime
        .readmit_expert(
            &request,
            &report,
            &ExpertFenceGrant::gate(project(), requirement()?, binding),
        )
        .map(|_| ())
        .map_err(fail("readmitting an expert"))
}
