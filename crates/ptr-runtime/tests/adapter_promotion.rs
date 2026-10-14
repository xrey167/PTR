//! Adapter promotion and revocation as committed history: digest recomputation, the split lock,
//! generation order, project scoping, revocation at every use, replay and the closed write path.
mod adapter_support;

use adapter_support::{digest, proposal, proposal_with_baseline, SPLITS, WEIGHTS};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_lineage::{LineageError, SplitLock};
use ptr_runtime::adapter::{AdapterError, PromoteAdapter, RevokeAdapter};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapsuleId, Generation, ProjectId, Validity};

fn project(name: &str) -> ProjectId {
    ProjectId::from(name)
}

fn request<'a>(name: &str, generation: u64, weights: &'a [u8]) -> PromoteAdapter<'a> {
    PromoteAdapter {
        project: project("alpha"),
        proposal: proposal(name),
        frozen_splits: SplitLock::of_manifest(SPLITS),
        generation: Generation(generation),
        artifact: weights,
    }
}

fn revoke(name: &str, generation: u64) -> RevokeAdapter {
    RevokeAdapter {
        project: project("alpha"),
        adapter: name.into(),
        generation: Generation(generation),
        reason: "training input revoked".into(),
    }
}

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).expect("runtime")
}

#[test]
fn a_promotion_commits_the_recomputed_digest_and_makes_the_adapter_serve() {
    let mut runtime = runtime();
    let before = runtime.committed_events().len();
    runtime
        .promote_adapter(request("coder", 1, WEIGHTS))
        .expect("promotes");
    let bound = runtime
        .serving_adapter(&project("alpha"), "coder")
        .expect("serves");
    assert_eq!(bound.generation(), Generation(1));
    assert_eq!(bound.artifact_sha256(), digest(WEIGHTS));
    assert_eq!(
        bound.gate_report_digest(),
        proposal("coder").gate_report_digest()
    );
    assert_eq!(runtime.committed_events().len(), before + 1);
    assert!(matches!(
        runtime.committed_events().last().map(|committed| &committed.event),
        Some(LedgerEvent::AdapterPromoted { adapter, generation: Generation(1), .. })
            if adapter == "coder"
    ));
}

#[test]
fn weights_that_do_not_hash_to_the_bound_digest_are_refused_and_append_nothing() {
    let mut runtime = runtime();
    let tampered = b"adapter weights, rank 5";
    let error = runtime
        .promote_adapter(request("coder", 1, tampered))
        .expect_err("a different artifact");
    assert_eq!(
        error,
        RuntimeError::Adapter(AdapterError::ArtifactDigestMismatch {
            expected: digest(WEIGHTS),
            actual: digest(tampered),
        })
    );
    assert!(runtime.committed_events().is_empty());
    assert!(runtime
        .serving_adapter(&project("alpha"), "coder")
        .is_none());
    assert_eq!(
        runtime.promote_adapter(request("coder", 1, b"")),
        Err(RuntimeError::Adapter(AdapterError::EmptyArtifact))
    );
}

#[test]
fn frozen_splits_other_than_the_gates_are_refused() {
    let mut runtime = runtime();
    let mut other = request("coder", 1, WEIGHTS);
    other.frozen_splits = SplitLock::of_manifest(b"another split manifest");
    assert_eq!(
        runtime.promote_adapter(other),
        Err(RuntimeError::Adapter(AdapterError::SplitLockMismatch {
            expected: SplitLock::of_manifest(SPLITS).sha256(),
            actual: SplitLock::of_manifest(b"another split manifest").sha256(),
        }))
    );
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn a_gate_without_a_serving_baseline_yields_no_proposal_to_promote() {
    assert_eq!(
        proposal_with_baseline("coder", WEIGHTS, None),
        Err(LineageError::MissingBaseline { id: "coder".into() })
    );
}

#[test]
fn a_revoked_generation_stops_serving_at_the_next_use_and_stays_revoked_after_replay() {
    let mut runtime = runtime();
    runtime
        .promote_adapter(request("coder", 1, WEIGHTS))
        .expect("promotes");
    runtime
        .check_adapter_use(&project("alpha"), "coder", Generation(1))
        .expect("serves before revocation");
    runtime.revoke_adapter(revoke("coder", 1)).expect("revokes");

    assert!(runtime
        .serving_adapter(&project("alpha"), "coder")
        .is_none());
    assert_eq!(
        runtime.check_adapter_use(&project("alpha"), "coder", Generation(1)),
        Err(AdapterError::NotServing {
            project: "alpha".into(),
            adapter: "coder".into(),
            generation: Generation(1),
            validity: Some(Validity::Revoked),
        })
    );
    let history = runtime.committed_events().to_vec();
    let replayed = PtrRuntime::replay(PtrConfig::default(), &history).expect("replays");
    assert!(replayed
        .serving_adapter(&project("alpha"), "coder")
        .is_none());
    assert!(replayed
        .check_adapter_use(&project("alpha"), "coder", Generation(1))
        .is_err());
}

#[test]
fn a_replayed_runtime_serves_the_same_promoted_generation() {
    let mut runtime = runtime();
    runtime
        .promote_adapter(request("coder", 3, WEIGHTS))
        .expect("promotes");
    let history = runtime.committed_events().to_vec();
    let replayed = PtrRuntime::replay(PtrConfig::default(), &history).expect("replays");
    assert_eq!(
        replayed.serving_adapter(&project("alpha"), "coder"),
        runtime.serving_adapter(&project("alpha"), "coder")
    );
}

#[test]
fn a_revoked_generation_cannot_be_promoted_again_and_a_new_one_must_advance() {
    let mut runtime = runtime();
    runtime
        .promote_adapter(request("coder", 2, WEIGHTS))
        .expect("promotes");
    for generation in [1, 2] {
        assert!(matches!(
            runtime.promote_adapter(request("coder", generation, WEIGHTS)),
            Err(RuntimeError::InvalidLifecycleTransition { .. })
        ));
    }
    runtime.revoke_adapter(revoke("coder", 2)).expect("revokes");
    assert!(matches!(
        runtime.promote_adapter(request("coder", 2, WEIGHTS)),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    runtime
        .promote_adapter(request("coder", 3, WEIGHTS))
        .expect("a later generation serves");
    assert_eq!(
        runtime
            .serving_adapter(&project("alpha"), "coder")
            .map(|bound| bound.generation()),
        Some(Generation(3))
    );
    // The earlier generation is superseded, not live.
    assert!(runtime
        .check_adapter_use(&project("alpha"), "coder", Generation(2))
        .is_err());
}

#[test]
fn a_revocation_of_an_unknown_adapter_or_generation_is_refused() {
    let mut runtime = runtime();
    assert!(matches!(
        runtime.revoke_adapter(revoke("coder", 1)),
        Err(RuntimeError::Adapter(AdapterError::UnknownAdapter { .. }))
    ));
    runtime
        .promote_adapter(request("coder", 1, WEIGHTS))
        .expect("promotes");
    assert!(matches!(
        runtime.revoke_adapter(revoke("coder", 2)),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    runtime.revoke_adapter(revoke("coder", 1)).expect("revokes");
    assert!(matches!(
        runtime.revoke_adapter(revoke("coder", 1)),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    let mut empty_reason = revoke("coder", 1);
    empty_reason.reason = String::new();
    assert!(runtime.revoke_adapter(empty_reason).is_err());
}

#[test]
fn another_projects_adapter_of_the_same_name_is_neither_served_nor_revoked() {
    let mut runtime = runtime();
    runtime
        .promote_adapter(request("coder", 1, WEIGHTS))
        .expect("promotes in alpha");
    assert!(runtime.serving_adapter(&project("beta"), "coder").is_none());
    assert!(matches!(
        runtime.check_adapter_use(&project("beta"), "coder", Generation(1)),
        Err(AdapterError::UnknownAdapter { .. })
    ));
    assert!(matches!(
        runtime.revoke_adapter(RevokeAdapter {
            project: project("beta"),
            ..revoke("coder", 1)
        }),
        Err(RuntimeError::Adapter(AdapterError::UnknownAdapter { .. }))
    ));
    assert!(runtime
        .serving_adapter(&project("alpha"), "coder")
        .is_some());
}

#[test]
fn the_lifecycle_key_cannot_be_forged_through_a_name_with_a_slash() {
    let mut runtime = runtime();
    let mut forged = request("coder", 1, WEIGHTS);
    forged.project = project("alpha/coder");
    assert!(matches!(
        runtime.promote_adapter(forged),
        Err(RuntimeError::Ledger(_))
    ));
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn the_public_commit_refuses_adapter_events_and_capsules_cannot_take_the_namespace() {
    let mut runtime = runtime();
    let event = LedgerEvent::AdapterPromoted {
        project: project("alpha"),
        adapter: "coder".into(),
        generation: Generation(1),
        base: [1; 32],
        artifact_sha256: [2; 32],
        gate_report_digest: [3; 32],
    };
    assert_eq!(
        runtime.commit(event),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    );
    assert!(matches!(
        runtime.commit(LedgerEvent::CapsuleCommitted {
            project: project("alpha"),
            capsule: CapsuleId::from("adapter:alpha/coder"),
            generation: Generation(9),
        }),
        Err(RuntimeError::ReservedTargetNamespace { .. })
    ));
    assert!(runtime
        .serving_adapter(&project("alpha"), "coder")
        .is_none());
}

#[test]
fn a_compacted_snapshot_is_refused_while_adapter_state_exists() {
    let mut runtime = runtime();
    runtime
        .promote_adapter(request("coder", 1, WEIGHTS))
        .expect("promotes");
    assert!(matches!(
        runtime.export_compacted_snapshot(),
        Err(RuntimeError::Adapter(AdapterError::CompactionUnsupported))
    ));
}

#[test]
fn a_promotion_request_prints_no_weights() {
    let printed = format!("{:?}", request("coder", 1, WEIGHTS));
    assert!(printed.contains("artifact_len"));
    assert!(!printed.contains("adapter weights"));
}
