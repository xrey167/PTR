//! The head of a stable pod identity moves only through a committed compare-and-swap record.
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{
    pod_lifecycle_target, AdvancePodHead, PodHeadError, PodHeadStatus, PtrRuntime, RuntimeError,
};
use ptr_types::{
    CapsuleId, Generation, NamespaceId, PodAddress, PodId, PodIdentity, PodRevisionAddress,
    ProjectId,
};

fn identity() -> PodIdentity {
    PodIdentity::from("pod:stable")
}

fn project() -> ProjectId {
    ProjectId::from("project")
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
            project: project(),
            identity: identity(),
            expected,
            new,
        })
        .map(|_| ())
}

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).unwrap()
}

/// Makes `revision`'s generation the one the lifecycle authority holds live for its pod.
fn make_live(runtime: &mut PtrRuntime, revision: &PodRevisionAddress) {
    let target = pod_lifecycle_target(revision);
    let event = match runtime.live_generation(&target) {
        None => LedgerEvent::CapsuleCommitted {
            project: revision.address.project.0.as_str().into(),
            capsule: CapsuleId::from(target.as_str()),
            generation: revision.generation,
        },
        Some(old) => LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(target.as_str()),
            old,
            new: revision.generation,
        },
    };
    runtime.commit(event).unwrap();
}

/// Makes `new` live in the lifecycle, then moves the head to it.
fn go(runtime: &mut PtrRuntime, expected: Option<PodRevisionAddress>, new: PodRevisionAddress) {
    make_live(runtime, &new);
    advance(runtime, expected, new).unwrap();
}

#[test]
fn the_first_head_commits_and_reads_back() {
    let mut runtime = runtime();
    assert_eq!(runtime.pod_head(&project(), &identity()), None);
    go(&mut runtime, None, revision("reader", 1));
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 1))
    );
    // The lifecycle commit and the head move.
    assert_eq!(runtime.committed_events().len(), 2);
}

#[test]
fn a_move_names_the_head_it_observed() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    go(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    );
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
}

#[test]
fn a_stale_expected_head_is_refused_and_appends_nothing() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    go(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    );
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
            project: project(),
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
            project: project(),
            identity: identity(),
            expected: None,
            actual: Some(revision("reader", 2)),
        })))
    );
    assert_eq!(runtime.committed_events().len(), committed);
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
}

#[test]
fn the_generation_must_rise_strictly() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 2));
    for generation in [1, 2] {
        assert_eq!(
            advance(
                &mut runtime,
                Some(revision("reader", 2)),
                revision("reader", generation)
            ),
            Err(RuntimeError::PodHead(Box::new(
                PodHeadError::NotAdvancing {
                    project: project(),
                    identity: identity(),
                    current: Generation(2),
                    requested: Generation(generation),
                }
            )))
        );
    }
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
}

#[test]
fn a_revoked_generation_cannot_become_the_head() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
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
                project: project(),
                identity: identity(),
                target,
                generation: Generation(2),
            }
        )))
    );
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 1))
    );
}

#[test]
fn a_revoked_head_is_not_returned_and_does_not_fall_back() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    go(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    );
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&revision("reader", 2)),
            generation: Generation(2),
        })
        .unwrap();
    assert_eq!(runtime.pod_head(&project(), &identity()), None);
    assert_eq!(runtime.pod_heads().count(), 0);
}

#[test]
fn revoking_one_pod_leaves_the_heads_of_other_identities_live() {
    let mut runtime = runtime();
    let other = PodIdentity::from("pod:other");
    go(&mut runtime, None, revision("reader", 1));
    make_live(&mut runtime, &revision("writer", 1));
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
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
    assert_eq!(runtime.pod_head(&project(), &identity()), None);
    assert_eq!(
        runtime.pod_head(&project(), &other),
        Some(&revision("writer", 1))
    );
}

#[test]
fn replay_reproduces_the_same_heads() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    go(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    );
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(
        replayed.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
    assert_eq!(
        replayed.materialized_state().values,
        runtime.materialized_state().values
    );
}

#[test]
fn replay_refuses_a_log_whose_head_move_does_not_match_the_head() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    let mut forged = runtime.committed_events().to_vec();
    // The same first-head record twice: the second one names `None` while a head exists.
    let mut duplicate = forged[1].clone();
    duplicate.index = ptr_types::CommitIndex(3);
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
    go(&mut runtime, None, revision("reader", 1));
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor;
    go(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 2),
    );
    let all = runtime.committed_events().to_vec();
    let above_floor = &all[floor.index.0 as usize..];

    let at_floor = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(
        at_floor.pod_head(&project(), &identity()),
        Some(&revision("reader", 1))
    );

    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        above_floor,
    )
    .unwrap();
    assert_eq!(
        restored.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
}

#[test]
fn a_revoked_head_can_be_replaced_by_a_new_generation_naming_it_as_expected() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&revision("reader", 1)),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(runtime.pod_head(&project(), &identity()), None);
    let revoked = match runtime.pod_head_status(&project(), &identity()) {
        Some(PodHeadStatus::Revoked(head)) => head.clone(),
        other => panic!("expected the revoked head to be visible, got {other:?}"),
    };
    assert_eq!(revoked, revision("reader", 1));

    make_live(&mut runtime, &revision("reader", 2));
    // A writer that cannot see the revoked head believes there is none and is refused.
    assert!(advance(&mut runtime, None, revision("reader", 2)).is_err());
    // Naming the revoked head moves the identity on to a new, unrevoked generation.
    advance(&mut runtime, Some(revoked), revision("reader", 2)).unwrap();
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
    assert!(matches!(
        runtime.pod_head_status(&project(), &identity()),
        Some(PodHeadStatus::Live(head)) if head == &revision("reader", 2)
    ));
}

#[test]
fn a_head_whose_generation_was_superseded_is_hidden_until_the_head_moves() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 1))
    );

    // The lifecycle authority supersedes generation 1 before any PodHeadAdvanced follows.
    make_live(&mut runtime, &revision("reader", 2));
    assert_eq!(runtime.pod_head(&project(), &identity()), None);
    assert_eq!(runtime.pod_heads().count(), 0);
    let stale = match runtime.pod_head_status(&project(), &identity()) {
        Some(PodHeadStatus::Stale(head)) => head.clone(),
        other => panic!("expected the stale head to be visible, got {other:?}"),
    };
    assert_eq!(stale, revision("reader", 1));

    // Replay gives the same answer.
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(replayed.pod_head(&project(), &identity()), None);
    assert_eq!(replayed.pod_heads().count(), 0);

    // Naming the stale head moves the identity to the live generation.
    advance(&mut runtime, Some(stale), revision("reader", 2)).unwrap();
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(
        replayed.pod_head(&project(), &identity()),
        Some(&revision("reader", 2))
    );
}

#[test]
fn a_head_for_a_pod_the_lifecycle_does_not_hold_live_fails_closed() {
    let mut runtime = runtime();
    advance(&mut runtime, None, revision("reader", 1)).unwrap();
    assert_eq!(runtime.pod_head(&project(), &identity()), None);
    assert!(matches!(
        runtime.pod_head_status(&project(), &identity()),
        Some(PodHeadStatus::Stale(_))
    ));
    // Once the lifecycle makes exactly that generation live, the head answers.
    make_live(&mut runtime, &revision("reader", 1));
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 1))
    );
}

#[test]
fn an_advance_below_the_live_generation_is_refused_and_appends_nothing() {
    let mut runtime = runtime();
    go(&mut runtime, None, revision("reader", 1));
    make_live(&mut runtime, &revision("reader", 2));
    make_live(&mut runtime, &revision("reader", 3));
    let committed = runtime.committed_events().len();
    assert_eq!(
        advance(
            &mut runtime,
            Some(revision("reader", 1)),
            revision("reader", 2)
        ),
        Err(RuntimeError::PodHead(Box::new(
            PodHeadError::BelowLiveGeneration {
                project: project(),
                identity: identity(),
                live: Generation(3),
                requested: Generation(2),
            }
        )))
    );
    assert_eq!(runtime.committed_events().len(), committed);
    // Moving to the live generation itself is allowed.
    advance(
        &mut runtime,
        Some(revision("reader", 1)),
        revision("reader", 3),
    )
    .unwrap();
    assert_eq!(
        runtime.pod_head(&project(), &identity()),
        Some(&revision("reader", 3))
    );
}

// --- Two projects, one identity -----------------------------------------------------------------

fn revision_in(project: &str, pod: &str, generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from(project),
            NamespaceId::from("ns"),
            PodId::from(pod),
        )
        .unwrap(),
        semantic_revision: [generation as u8 + 1; 32],
        generation: Generation(generation),
    }
}

fn go_in(
    runtime: &mut PtrRuntime,
    project: &str,
    expected: Option<PodRevisionAddress>,
    generation: u64,
) {
    let new = revision_in(project, "reader", generation);
    make_live(runtime, &new);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: ProjectId::from(project),
            identity: identity(),
            expected,
            new,
        })
        .unwrap();
}

#[test]
fn two_projects_with_the_same_identity_keep_independent_heads() {
    let (a, b) = (ProjectId::from("alpha"), ProjectId::from("beta"));
    let mut runtime = runtime();
    // Both projects derive the same identity and each starts with no head.
    go_in(&mut runtime, "alpha", None, 1);
    assert_eq!(
        runtime.pod_head(&a, &identity()),
        Some(&revision_in("alpha", "reader", 1))
    );
    assert_eq!(runtime.pod_head(&b, &identity()), None);
    go_in(&mut runtime, "beta", None, 1);
    go_in(
        &mut runtime,
        "beta",
        Some(revision_in("beta", "reader", 1)),
        2,
    );
    // Moving beta's head left alpha's alone, and the two never share a compare-and-swap.
    assert_eq!(
        runtime.pod_head(&a, &identity()),
        Some(&revision_in("alpha", "reader", 1))
    );
    assert_eq!(
        runtime.pod_head(&b, &identity()),
        Some(&revision_in("beta", "reader", 2))
    );
    assert_eq!(runtime.pod_heads().count(), 2);
    // Revoking alpha's revision hides alpha's head only.
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&revision_in("alpha", "reader", 1)),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(runtime.pod_head(&a, &identity()), None);
    assert_eq!(
        runtime.pod_head(&b, &identity()),
        Some(&revision_in("beta", "reader", 2))
    );
}

#[test]
fn a_head_move_names_only_addresses_of_its_own_project() {
    let mut runtime = runtime();
    let foreign = revision_in("beta", "reader", 1);
    make_live(&mut runtime, &foreign);
    let error = runtime
        .advance_pod_head(AdvancePodHead {
            project: ProjectId::from("alpha"),
            identity: identity(),
            expected: None,
            new: foreign,
        })
        .unwrap_err();
    assert_eq!(
        error,
        RuntimeError::PodHead(Box::new(PodHeadError::ProjectMismatch {
            project: ProjectId::from("alpha"),
            identity: identity(),
            address_project: ProjectId::from("beta"),
        }))
    );
    assert_eq!(runtime.pod_heads().count(), 0);
}

#[test]
fn replay_and_a_compacted_restore_keep_the_heads_of_both_projects() {
    let mut runtime = runtime();
    go_in(&mut runtime, "alpha", None, 1);
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor;
    go_in(&mut runtime, "beta", None, 1);

    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    for (project, generation) in [("alpha", 1), ("beta", 1)] {
        assert_eq!(
            replayed.pod_head(&ProjectId::from(project), &identity()),
            Some(&revision_in(project, "reader", generation))
        );
    }
    assert_eq!(
        replayed.materialized_state().values,
        runtime.materialized_state().values
    );
    assert!(replayed
        .materialized_state()
        .values
        .contains_key("pod-head:alpha/pod:stable"));

    let all = runtime.committed_events().to_vec();
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &all[floor.index.0 as usize..],
    )
    .unwrap();
    assert_eq!(
        restored.pod_head(&ProjectId::from("beta"), &identity()),
        Some(&revision_in("beta", "reader", 1))
    );
    assert_eq!(
        restored.pod_head(&ProjectId::from("alpha"), &identity()),
        Some(&revision_in("alpha", "reader", 1))
    );
}

#[test]
fn a_pod_capsule_committed_under_another_project_is_refused_and_the_victim_stays_not_live() {
    use ptr_ledger::CommittedEvent;
    let mut runtime = runtime();
    let victim = revision_in("beta", "reader", 1);
    let target = pod_lifecycle_target(&victim);
    let forged = LedgerEvent::CapsuleCommitted {
        project: "alpha".into(),
        capsule: CapsuleId::from(target.as_str()),
        generation: Generation(1),
    };
    assert_eq!(
        runtime.commit(forged.clone()),
        Err(RuntimeError::PodCapsuleProjectMismatch {
            capsule: target.clone(),
            embedded: "beta".into(),
            project: "alpha".into(),
        })
    );
    assert_eq!(runtime.live_generation(&target), None);
    assert!(runtime.committed_events().is_empty());

    // beta's own head move is a valid record, but beta's pod was never made live by beta.
    let beta = ProjectId::from("beta");
    runtime
        .advance_pod_head(AdvancePodHead {
            project: beta.clone(),
            identity: identity(),
            expected: None,
            new: victim,
        })
        .unwrap();
    assert_eq!(runtime.pod_head(&beta, &identity()), None);
    assert_eq!(runtime.pod_heads().count(), 0);

    // A log that carries the forged record is refused on replay, not grandfathered.
    let log = [CommittedEvent {
        index: ptr_types::CommitIndex(1),
        event: forged,
    }];
    assert!(PtrRuntime::replay(PtrConfig::default(), &log).is_err());
}

#[test]
fn a_pod_capsule_committed_by_its_own_project_is_accepted_and_other_ids_are_unaffected() {
    let mut runtime = runtime();
    let own = revision_in("beta", "reader", 1);
    make_live(&mut runtime, &own);
    assert_eq!(
        runtime.live_generation(&pod_lifecycle_target(&own)),
        Some(Generation(1))
    );
    for capsule in [
        "note:alpha",
        "pod:alpha",
        "pod:alpha/ns",
        "pod:/ns/p",
        "pod:a/b/c/d",
    ] {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: "alpha".into(),
                capsule: CapsuleId::from(capsule),
                generation: Generation(1),
            })
            .unwrap();
    }
}

#[test]
fn generation_zero_is_refused_for_a_pod_capsule_and_for_a_head() {
    use ptr_ledger::CommittedEvent;
    let mut runtime = runtime();
    let zero = revision("reader", 0);
    let target = pod_lifecycle_target(&zero);
    let commit = LedgerEvent::CapsuleCommitted {
        project: "project".into(),
        capsule: CapsuleId::from(target.as_str()),
        generation: Generation(0),
    };
    assert_eq!(
        runtime.commit(commit.clone()),
        Err(RuntimeError::PodCapsuleGenerationZero {
            capsule: target.clone()
        })
    );
    assert_eq!(runtime.live_generation(&target), None);

    let refused = advance(&mut runtime, None, zero).unwrap_err();
    assert!(matches!(
        refused,
        RuntimeError::PodHead(ref error) if matches!(**error, PodHeadError::InvalidAddress { .. })
    ));
    assert!(runtime.committed_events().is_empty());
    assert_eq!(runtime.pod_heads().count(), 0);

    // A log carrying the generation-zero commit fails replay closed.
    let log = [CommittedEvent {
        index: ptr_types::CommitIndex(1),
        event: commit,
    }];
    assert!(PtrRuntime::replay(PtrConfig::default(), &log).is_err());
}
