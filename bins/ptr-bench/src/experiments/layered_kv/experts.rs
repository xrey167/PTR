//! MoE experts through the real admission path: a birth, a retirement and a reactivation are
//! structural proposals admitted by `PtrRuntime::admit_expert_proposal`, so the runtime's own
//! materialised `expert:{project}/{pod}` record is what answers an expert dependency's
//! liveness. Nothing here scripts an oracle.
//!
//! An expert `(layer, index)` is the pod `moe-l{layer}-e{index}` at token level
//! `Token { layer, index }`. Revoking it is a `Retire`; superseding it is a `Retire` followed
//! by a `Reactivate` of the retired generation, which commits generation + 1 of the same pod.

use super::lab::{fail, Res, PROJECT};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_runtime::expert::ExpertAdmissionGrant;
use ptr_runtime::PtrRuntime;
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

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

fn grant() -> Res<ExpertAdmissionGrant> {
    Ok(ExpertAdmissionGrant::new(
        project(),
        proposer()?,
        GATE_EPOCH,
        gate()
            .config_digest()
            .map_err(fail("the gate configuration digest"))?,
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
        [3; 32],
        [4; 32],
        [5; 32],
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
    runtime
        .admit_expert_proposal(&proposal, &report, &grant()?)
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
