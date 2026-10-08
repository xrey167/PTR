//! The expert registry crosses a compaction floor: a runtime restored from a compacted snapshot
//! admits and refuses exactly what one that replayed the whole history would.

use ptr_config::PtrConfig;
use ptr_ledger::integrity;
use ptr_lineage::{AccuracyMatrix, AdapterId, ForgettingGate, GateSubject, PublicSuite};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    RetiredReason, StructuralOp,
};
use ptr_runtime::compacted::{CompactedAnchor, CompactedError};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, ExpertAdmissionError,
    ExpertAdmissionGrant,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

mod expert_support;

use expert_support::{binding, prime, requirement, EVAL_MANIFEST};

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

fn admit(rt: &mut PtrRuntime, op: StructuralOp) -> Result<CommitIndex, RuntimeError> {
    let proposal = ExpertProposal::new(
        project(),
        proposer(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let public = PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    };
    let report = gate()
        .evaluate(&AdapterId::from("candidate"), &matrix, public)
        .unwrap()
        .bind(GateSubject::new(
            *proposal.proposal_id().as_bytes(),
            *proposal.operation_id().as_bytes(),
            expert_evidence_artifact_digest(proposal.op()),
            expert_evidence_base_model_digest(proposal.op()).unwrap(),
            EVAL_MANIFEST,
        ));
    prime(rt);
    let grant = ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        binding(),
    );
    rt.admit_expert_proposal(&proposal, &report, &grant)
}

fn target(rt: &PtrRuntime, pod: &str) -> ExpertTarget {
    let e = rt
        .expert_registry()
        .get(&project(), &PodId::from(pod))
        .unwrap_or_else(|| panic!("no expert {pod}"))
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: e.generation,
        level: e.level,
        base_model: e.base_model,
        manifest_digest: e.manifest_digest,
        state: e.state,
        state_version: e.state_version,
    })
    .unwrap()
}

fn birth(rt: &mut PtrRuntime, pod: &str) {
    admit(rt, StructuralOp::Birth { result: spec(pod) }).unwrap();
}

fn prune(rt: &mut PtrRuntime, pods: &[&str]) {
    let targets = pods.iter().map(|p| target(rt, p)).collect();
    admit(rt, StructuralOp::Prune { targets }).unwrap();
}

fn restore(rt: &PtrRuntime) -> PtrRuntime {
    let snapshot = rt.export_compacted_snapshot().unwrap();
    PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap()
}

/// Birth a, b, c, d and prune a: a retired expert and live ones below the floor.
fn before_the_floor() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    for pod in ["a", "b", "c", "d"] {
        birth(&mut rt, pod);
    }
    prune(&mut rt, &["a"]);
    rt
}

#[test]
fn a_restored_runtime_carries_the_registry_and_the_materialized_keys() {
    let rt = before_the_floor();
    let restored = restore(&rt);
    assert_eq!(restored.expert_registry(), rt.expert_registry());
    assert_eq!(
        restored.materialized_state().values,
        rt.materialized_state().values
    );
    assert_eq!(
        restored
            .materialized_state()
            .values
            .get("expert:proj/a")
            .map(String::as_str),
        Some("1:retired-pruned:2")
    );
}

#[test]
fn a_pre_floor_pod_id_cannot_be_born_again_after_restore() {
    let mut restored = restore(&before_the_floor());
    for pod in ["a", "b"] {
        assert_eq!(
            admit(&mut restored, StructuralOp::Birth { result: spec(pod) }),
            Err(RuntimeError::Expert(
                ExpertAdmissionError::ResultPodOccupied {
                    pod: pod.to_owned()
                }
            ))
        );
    }
    assert_eq!(
        restored
            .materialized_state()
            .values
            .get("expert:proj/a")
            .map(String::as_str),
        Some("1:retired-pruned:2")
    );
}

#[test]
fn pre_floor_experts_can_be_operated_on_after_restore() {
    let mut restored = restore(&before_the_floor());
    let freeze = StructuralOp::Freeze {
        target: target(&restored, "b"),
    };
    admit(&mut restored, freeze).unwrap();
    let retire = StructuralOp::Retire {
        target: target(&restored, "c"),
        reason: RetiredReason::Superseded,
    };
    admit(&mut restored, retire).unwrap();
    // The observed state of d is what the restored registry holds.
    let stale = StructuralOp::Freeze {
        target: ExpertTarget::new(ExpertTargetInit {
            state: ExpertState::Serving,
            ..init(&restored, "d")
        })
        .unwrap(),
    };
    admit(&mut restored, stale).unwrap();
    let merge = StructuralOp::Merge {
        sources: vec![target(&restored, "b"), target(&restored, "d")],
        result: spec("m"),
    };
    // b and d are now Frozen, both: a legal merge.
    admit(&mut restored, merge).unwrap();
    assert_eq!(
        restored
            .expert_registry()
            .get(&project(), &PodId::from("m"))
            .unwrap()
            .state,
        ExpertState::Serving
    );
}

fn init(rt: &PtrRuntime, pod: &str) -> ExpertTargetInit {
    let e = target(rt, pod);
    ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: e.generation(),
        level: e.level(),
        base_model: e.base_model().clone(),
        manifest_digest: *e.manifest_digest(),
        state: e.state(),
        state_version: e.state_version(),
    }
}

#[test]
fn export_restore_continue_equals_a_full_replay() {
    let mut full = before_the_floor();
    let snapshot = full.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor.index;
    // Continue on the original, then restore from the snapshot with the records above it.
    let freeze = StructuralOp::Freeze {
        target: target(&full, "b"),
    };
    admit(&mut full, freeze).unwrap();
    let split = StructuralOp::Split {
        source: target(&full, "c"),
        results: vec![spec("x"), spec("y")],
    };
    admit(&mut full, split).unwrap();
    prune(&mut full, &["d"]);
    let above: Vec<_> = full
        .committed_events()
        .iter()
        .filter(|event| event.index > floor)
        .cloned()
        .collect();
    assert_eq!(above.len(), 3);
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &above,
    )
    .unwrap();
    let replayed = PtrRuntime::replay(PtrConfig::default(), full.committed_events()).unwrap();
    assert_eq!(restored.expert_registry(), replayed.expert_registry());
    assert_eq!(restored.expert_registry(), full.expert_registry());
    assert_eq!(
        restored.materialized_state().values,
        replayed.materialized_state().values
    );
}

fn registry_block(bytes: &[u8]) -> usize {
    bytes
        .windows(8)
        .rposition(|window| window == b"PTREG001")
        .expect("registry block")
}

fn resealed(mut bytes: Vec<u8>, trusted: CompactedAnchor) -> (Vec<u8>, CompactedAnchor) {
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    (bytes, CompactedAnchor { digest, ..trusted })
}

fn error_of(bytes: &[u8], anchor: CompactedAnchor) -> Option<RuntimeError> {
    PtrRuntime::restore_compacted(PtrConfig::default(), bytes, anchor, &[]).err()
}

/// A history with exactly one expert, `xx`, so the registry block has one known entry.
fn one_expert() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    birth(&mut rt, "xx");
    rt
}

#[test]
fn a_registry_section_that_disagrees_with_the_materialized_keys_is_refused() {
    let rt = one_expert();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    let block = registry_block(snapshot.bytes());
    // magic, count, "proj" (4+4), "xx" (4+2), then the generation, state_version, digest, tag.
    let generation = block + 8 + 4 + 8 + 6;
    let state_version = generation + 8;
    let tag = state_version + 8 + 32;
    let mismatched = |at: usize, value: u8| {
        let mut bytes = snapshot.bytes().to_vec();
        bytes[at] = value;
        let (bytes, anchor) = resealed(bytes, snapshot.anchor());
        error_of(&bytes, anchor)
    };
    let state_mismatch = Some(RuntimeError::Compacted(CompactedError::StateMismatch));
    assert_eq!(mismatched(generation, 2), state_mismatch);
    assert_eq!(mismatched(state_version, 5), state_mismatch);
    assert_eq!(mismatched(tag, 3), state_mismatch);
    // An unknown state tag is refused as such.
    assert_eq!(
        mismatched(tag, 99),
        Some(RuntimeError::Compacted(CompactedError::UnknownTag))
    );
    // An empty registry beside an `expert:` key.
    // The registry entries are cut out; the evidence block that follows them stays.
    let evidence = snapshot
        .bytes()
        .windows(8)
        .rposition(|window| window == b"PTREV001")
        .expect("evidence block");
    let mut emptied = snapshot.bytes()[..block + 8].to_vec();
    emptied.extend_from_slice(&0u32.to_le_bytes());
    emptied.extend_from_slice(&snapshot.bytes()[evidence..]);
    let old_len = u64::from_le_bytes(snapshot.bytes()[64..72].try_into().unwrap()) as usize;
    let removed = evidence - (block + 12);
    let new_len = old_len - removed;
    emptied[64..72].copy_from_slice(&(new_len as u64).to_le_bytes());
    let (bytes, anchor) = resealed(emptied, snapshot.anchor());
    assert_eq!(error_of(&bytes, anchor), state_mismatch);
}

#[test]
fn the_earlier_layout_restores_only_a_history_without_expert_records() {
    let rt = one_expert();
    let snapshot = rt.export_compacted_snapshot().unwrap();
    let lifecycle_start =
        80 + u64::from_le_bytes(snapshot.bytes()[56..64].try_into().unwrap()) as usize;
    let old_len = u64::from_le_bytes(snapshot.bytes()[64..72].try_into().unwrap()) as usize;
    let block = registry_block(snapshot.bytes());
    let lifecycle_end = lifecycle_start + old_len;
    // Strip the registry block and relabel as PTRCS004 with PTRLC002.
    let mut older = snapshot.bytes()[..block].to_vec();
    older.extend_from_slice(&snapshot.bytes()[lifecycle_end..]);
    older[..8].copy_from_slice(b"PTRCS004");
    older[lifecycle_start..lifecycle_start + 8].copy_from_slice(b"PTRLC002");
    older[64..72].copy_from_slice(&((block - lifecycle_start) as u64).to_le_bytes());
    let (bytes, anchor) = resealed(older, snapshot.anchor());
    assert_eq!(
        error_of(&bytes, anchor),
        Some(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
    );
    // Without an expert record the same relabelling restores (covered by the layout tests in
    // compacted_snapshot.rs); here only the refusal for a history that holds one matters.
}
