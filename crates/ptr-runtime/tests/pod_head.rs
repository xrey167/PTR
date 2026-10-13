//! The head of a stable pod identity moves only through a committed compare-and-swap record.
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{pod_lifecycle_target, AdvancePodHead, PodHeadError, PtrRuntime, RuntimeError};
use ptr_types::{
    Generation, NamespaceId, PodAddress, PodId, PodIdentity, PodRevisionAddress, ProjectId,
};

fn identity() -> PodIdentity {
    PodIdentity::from("pod:stable")
}

fn revision(pod: &str, generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from(pod),
        )
        .unwrap(),
        semantic_revision: [generation as u8 + 1; 32],
        generation: Generation(generation),
    }
}

fn advance(
    runtime: &mut PtrRuntime,
    expected: Option<PodRevisionAddress>,
    new: PodRevisionAddress,
) -> Result<(), RuntimeError> {
    runtime
        .advance_pod_head(AdvancePodHead {
            identity: identity(),
            expected,
            new,
        })
        .map(|_| ())
}

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).unwrap()
}

#[test]
fn the_first_head_commits_and_reads_back() {
    let mut runtime = runtime();
    assert_eq!(runtime.pod_head(&identity()), None);
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    assert_eq!(runtime.pod_head(&identity()), Some(&revision("reader", 1)));
    assert_eq!(runtime.committed_events().len(), 1);
}

#[test]
fn a_move_names_the_head_it_observed() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    )
    .unwrap();
    assert_eq!(runtime.pod_head(&identity()), Some(&revision("reader", 2)));
}

#[test]
fn a_stale_expected_head_is_refused_and_appends_nothing() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    )
    .unwrap();
    let committed = runtime.committed_events().len();

    // A writer that still believes generation 1 is the head.
    let stale = advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 3),
    );
    assert_eq!(
        stale,
        Err(RuntimeError::PodHead(Box::new(PodHeadError::Conflict {
            identity: identity(),
            expected: Some(revision("reader", 1)),
            actual: Some(revision("reader", 2)),
        })))
    );
    // A writer that believes there is no head yet.
    let first = advance(&mut runtime, None, revision("reader", 3));
    assert_eq!(
        first,
        Err(RuntimeError::PodHead(Box::new(PodHeadError::Conflict {
            identity: identity(),
            expected: None,
            actual: Some(revision("reader", 2)),
        })))
    );
    assert_eq!(runtime.committed_events().len(), committed);
    assert_eq!(runtime.pod_head(&identity()), Some(&revision("reader", 2)));
}

#[test]
fn the_generation_must_rise_strictly() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 2)).unwrap();
    for generation in [1, 2] {
        assert_eq!(
            advance(
                &mut runtime,
                Some(revision("reader", 2)),
                revision("reader", generation)
            ),
            Err(RuntimeError::PodHead(Box::new(
                PodHeadError::NotAdvancing {
                    identity: identity(),
                    current: Generation(2),
                    requested: Generation(generation),
                }
            )))
        );
    }
    assert_eq!(runtime.pod_head(&identity()), Some(&revision("reader", 2)));
}

#[test]
fn a_revoked_generation_cannot_become_the_head() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    let candidate = revision("reader", 2);
    let target = pod_lifecycle_target(&candidate);
    runtime
        .commit(LedgerEvent::Revoked {
            subject: target.clone(),
            generation: Generation(2),
        })
        .unwrap();
    assert_eq!(
        advance(&mut runtime, Some(revision("reader", 1)), candidate),
        Err(RuntimeError::PodHead(Box::new(
            PodHeadError::RevokedGeneration {
                identity: identity(),
                target,
                generation: Generation(2),
            }
        )))
    );
    assert_eq!(runtime.pod_head(&identity()), Some(&revision("reader", 1)));
}

#[test]
fn a_revoked_head_is_not_returned_and_does_not_fall_back() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    )
    .unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&revision("reader", 2)),
            generation: Generation(2),
        })
        .unwrap();
    assert_eq!(runtime.pod_head(&identity()), None);
    assert_eq!(runtime.pod_heads().count(), 0);
}

#[test]
fn revoking_one_pod_leaves_the_heads_of_other_identities_live() {
    let mut runtime = runtime();
    let other = PodIdentity::from("pod:other");
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    runtime
        .advance_pod_head(AdvancePodHead {
            identity: other.clone(),
            expected: None,
            new: revision("writer", 1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&revision("reader", 1)),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(runtime.pod_head(&identity()), None);
    assert_eq!(runtime.pod_head(&other), Some(&revision("writer", 1)));
}

#[test]
fn replay_reproduces_the_same_heads() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    )
    .unwrap();
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(replayed.pod_head(&identity()), Some(&revision("reader", 2)));
    assert_eq!(
        replayed.materialized_state().values,
        runtime.materialized_state().values
    );
}

#[test]
fn replay_refuses_a_log_whose_head_move_does_not_match_the_head() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    let mut forged = runtime.committed_events().to_vec();
    // The same first-head record twice: the second one names `None` while a head exists.
    let mut duplicate = forged[0].clone();
    duplicate.index = ptr_types::CommitIndex(2);
    forged.push(duplicate);
    let Err(RuntimeError::PodHead(error)) = PtrRuntime::replay(PtrConfig::default(), &forged)
    else {
        panic!("replay accepted a head move that names no head while one exists");
    };
    assert!(matches!(*error, PodHeadError::Conflict { .. }));
}

#[test]
fn heads_survive_a_compacted_snapshot_and_moves_above_the_floor() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor;
    advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    )
    .unwrap();
    let all = runtime.committed_events().to_vec();
    let above_floor = &all[floor.index.0 as usize..];

    let at_floor = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(at_floor.pod_head(&identity()), Some(&revision("reader", 1)));

    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        above_floor,
    )
    .unwrap();
    assert_eq!(restored.pod_head(&identity()), Some(&revision("reader", 2)));
}
