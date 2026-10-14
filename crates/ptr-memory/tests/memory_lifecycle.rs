mod common;

use common::{derived, derived_record, new_record, observation, rid, TempJournal};
use ptr_memory::{
    DurableMemoryStore, MemoryClass, MemoryLimits, MemoryStoreError, MemoryStoreOptions,
    RecordState, RecordViolation, Support, SupportKind, TraversalLimit,
};
use ptr_types::Revision;

fn quiet() -> MemoryStoreOptions {
    MemoryStoreOptions {
        compact_after_records: None,
        ..MemoryStoreOptions::default()
    }
}

fn open(file: &TempJournal) -> DurableMemoryStore {
    DurableMemoryStore::open(&file.0, quiet()).unwrap()
}

fn state(store: &DurableMemoryStore, id: &str) -> RecordState {
    store.get(&rid(id)).unwrap().state
}

#[test]
fn all_four_classes_are_stored_and_survive_reopen() {
    let file = TempJournal::new("ms-classes");
    let mut store = open(&file);
    for (id, class) in [
        ("sem", MemoryClass::Semantic),
        ("epi", MemoryClass::Episodic),
        ("pro", MemoryClass::Procedural),
        ("epm", MemoryClass::Epistemic),
    ] {
        store
            .insert(new_record(id, class, vec![observation("src", 1)]))
            .unwrap();
    }
    store.advance_committed_revision(Revision(1)).unwrap();
    store
        .transition(&rid("sem"), RecordState::Authoritative)
        .unwrap();
    let snapshot: Vec<_> = store.records().cloned().collect();
    let (horizon, clock) = (store.committed_revision(), store.clock());
    drop(store);
    let store = open(&file);
    assert_eq!(store.records().cloned().collect::<Vec<_>>(), snapshot);
    assert_eq!(
        (store.committed_revision(), store.clock()),
        (horizon, clock)
    );
    for class in [
        MemoryClass::Semantic,
        MemoryClass::Episodic,
        MemoryClass::Procedural,
        MemoryClass::Epistemic,
    ] {
        assert_eq!(store.records_of(class).count(), 1);
    }
}

#[test]
fn only_allowed_transitions_succeed_with_typed_errors_otherwise() {
    let file = TempJournal::new("ms-sm");
    let mut store = open(&file);
    store
        .insert(new_record(
            "r",
            MemoryClass::Semantic,
            vec![observation("a", 1)],
        ))
        .unwrap();
    // Provisional -> NeedsRevalidation is not an edge.
    assert_eq!(
        store.transition(&rid("r"), RecordState::NeedsRevalidation),
        Err(MemoryStoreError::InvalidTransition {
            id: rid("r"),
            from: RecordState::Provisional,
            to: RecordState::NeedsRevalidation
        })
    );
    // Self transition refused.
    assert!(matches!(
        store.transition(&rid("r"), RecordState::Provisional),
        Err(MemoryStoreError::InvalidTransition { .. })
    ));
    store.advance_committed_revision(Revision(1)).unwrap();
    store
        .transition(&rid("r"), RecordState::Authoritative)
        .unwrap();
    store
        .transition(&rid("r"), RecordState::NeedsRevalidation)
        .unwrap();
    store
        .transition(&rid("r"), RecordState::Authoritative)
        .unwrap();
    store.transition(&rid("r"), RecordState::Retired).unwrap();
    assert!(matches!(
        store.transition(&rid("r"), RecordState::Authoritative),
        Err(MemoryStoreError::InvalidTransition {
            from: RecordState::Retired,
            ..
        })
    ));
    store.transition(&rid("r"), RecordState::Revoked).unwrap();
    for to in [
        RecordState::Provisional,
        RecordState::Authoritative,
        RecordState::NeedsRevalidation,
        RecordState::Retired,
        RecordState::Revoked,
    ] {
        assert!(
            matches!(
                store.transition(&rid("r"), to),
                Err(MemoryStoreError::InvalidTransition {
                    from: RecordState::Revoked,
                    ..
                })
            ),
            "Revoked must be terminal (to {to:?})"
        );
    }
}

#[test]
fn authority_requires_a_committed_observation_invariant_4() {
    let file = TempJournal::new("ms-inv4");
    let mut store = open(&file);
    store
        .insert(new_record(
            "r",
            MemoryClass::Semantic,
            vec![observation("a", 5)],
        ))
        .unwrap();
    // Revision 5 is not committed yet.
    assert_eq!(
        store.transition(&rid("r"), RecordState::Authoritative),
        Err(MemoryStoreError::NoCommittedSupport(rid("r")))
    );
    store.advance_committed_revision(Revision(4)).unwrap();
    assert!(store
        .transition(&rid("r"), RecordState::Authoritative)
        .is_err());
    store.advance_committed_revision(Revision(5)).unwrap();
    store
        .transition(&rid("r"), RecordState::Authoritative)
        .unwrap();
    assert_eq!(
        store.advance_committed_revision(Revision(2)),
        Err(MemoryStoreError::HorizonRegression {
            current: Revision(5),
            requested: Revision(2)
        })
    );
}

#[test]
fn derived_or_search_support_never_grants_authority_invariant_5() {
    let file = TempJournal::new("ms-inv5");
    let mut store = open(&file);
    store.advance_committed_revision(Revision(100)).unwrap();
    store
        .insert(new_record(
            "r",
            MemoryClass::Semantic,
            vec![
                derived("inference"),
                Support {
                    source: "idx".into(),
                    kind: SupportKind::SearchHit,
                },
            ],
        ))
        .unwrap();
    assert_eq!(
        store.transition(&rid("r"), RecordState::Authoritative),
        Err(MemoryStoreError::NoCommittedSupport(rid("r")))
    );
    // Adding more derived support does not help either.
    store.add_support(&rid("r"), derived("more")).unwrap();
    assert!(store
        .transition(&rid("r"), RecordState::Authoritative)
        .is_err());
    assert_eq!(state(&store, "r"), RecordState::Provisional);
}

#[test]
fn revocation_propagates_to_transitive_dependents_and_is_persisted() {
    let file = TempJournal::new("ms-revoke");
    let mut store = open(&file);
    store.advance_committed_revision(Revision(1)).unwrap();
    store
        .insert(new_record(
            "root",
            MemoryClass::Semantic,
            vec![observation("a", 1)],
        ))
        .unwrap();
    store
        .transition(&rid("root"), RecordState::Authoritative)
        .unwrap();
    // child1 <- root, child2 <- child1, unrelated stays untouched.
    let mut child1 = derived_record("child1", MemoryClass::Semantic, &["root"]);
    child1.supports = vec![observation("b", 1)];
    store.insert(child1).unwrap();
    store
        .transition(&rid("child1"), RecordState::Authoritative)
        .unwrap();
    store
        .insert(derived_record("child2", MemoryClass::Episodic, &["child1"]))
        .unwrap();
    store
        .insert(new_record(
            "other",
            MemoryClass::Procedural,
            vec![observation("c", 1)],
        ))
        .unwrap();

    let report = store.revoke(&rid("root")).unwrap();
    assert_eq!(
        report.marked_for_revalidation,
        vec![rid("child1"), rid("child2")]
    );
    assert_eq!(state(&store, "root"), RecordState::Revoked);
    assert_eq!(state(&store, "child1"), RecordState::NeedsRevalidation);
    assert_eq!(state(&store, "child2"), RecordState::NeedsRevalidation);
    assert_eq!(state(&store, "other"), RecordState::Provisional);
    assert_eq!(
        store.get(&rid("child2")).unwrap().revoked_ancestors,
        vec![rid("root")]
    );

    // Persisted through the journal.
    drop(store);
    let mut store = open(&file);
    assert_eq!(state(&store, "root"), RecordState::Revoked);
    assert_eq!(state(&store, "child1"), RecordState::NeedsRevalidation);
    // Revalidation is refused while the revoked ancestor stands (Invariant 3).
    assert_eq!(
        store.transition(&rid("child1"), RecordState::Authoritative),
        Err(MemoryStoreError::RevokedAncestor {
            id: rid("child1"),
            ancestor: rid("root")
        })
    );
    assert!(matches!(
        store.transition(&rid("child2"), RecordState::Authoritative),
        Err(MemoryStoreError::NoCommittedSupport(_))
            | Err(MemoryStoreError::RevokedAncestor { .. })
    ));
    // Revoking again is an idempotent no-op; new derivations from it are refused.
    assert!(store
        .revoke(&rid("root"))
        .unwrap()
        .marked_for_revalidation
        .is_empty());
    assert!(matches!(
        store.insert(derived_record("late", MemoryClass::Semantic, &["root"])),
        Err(MemoryStoreError::RevokedAncestor { .. })
    ));
}

#[test]
fn revoking_through_transition_propagates_identically() {
    let file = TempJournal::new("ms-revoke-tr");
    let mut store = open(&file);
    store
        .insert(new_record("a", MemoryClass::Semantic, vec![]))
        .unwrap();
    store
        .insert(derived_record("b", MemoryClass::Semantic, &["a"]))
        .unwrap();
    store.transition(&rid("a"), RecordState::Revoked).unwrap();
    assert_eq!(state(&store, "b"), RecordState::NeedsRevalidation);
}

#[test]
fn revocation_is_cycle_safe() {
    let file = TempJournal::new("ms-cycle");
    let mut store = open(&file);
    store
        .insert(new_record("a", MemoryClass::Semantic, vec![]))
        .unwrap();
    store
        .insert(derived_record("b", MemoryClass::Semantic, &["a"]))
        .unwrap();
    store
        .insert(derived_record("c", MemoryClass::Semantic, &["b"]))
        .unwrap();
    // Close the loop a <- c, so a -> b -> c -> a.
    store.add_derivation(&rid("a"), &rid("c")).unwrap();
    assert_eq!(
        store.add_derivation(&rid("a"), &rid("a")),
        Err(MemoryStoreError::SelfDerivation(rid("a")))
    );
    let report = store.revoke(&rid("b")).unwrap();
    assert_eq!(report.marked_for_revalidation, vec![rid("a"), rid("c")]);
    assert_eq!(report.visited, 2);
    assert_eq!(state(&store, "b"), RecordState::Revoked);
}

#[test]
fn revocation_depth_and_breadth_are_bounded_and_atomic() {
    let file = TempJournal::new("ms-bound");
    let options = MemoryStoreOptions {
        limits: MemoryLimits {
            max_revocation_depth: 3,
            ..MemoryLimits::default()
        },
        ..quiet()
    };
    let mut store = DurableMemoryStore::open(&file.0, options).unwrap();
    store
        .insert(new_record("n0", MemoryClass::Semantic, vec![]))
        .unwrap();
    for i in 1..=5 {
        let parent = format!("n{}", i - 1);
        store
            .insert(derived_record(
                &format!("n{i}"),
                MemoryClass::Semantic,
                &[parent.as_str()],
            ))
            .unwrap();
    }
    let clock = store.clock();
    assert_eq!(
        store.revoke(&rid("n0")),
        Err(MemoryStoreError::TraversalBoundExceeded(
            TraversalLimit::Depth(3)
        ))
    );
    // Nothing changed.
    assert_eq!(store.clock(), clock);
    assert_eq!(state(&store, "n0"), RecordState::Provisional);
    assert_eq!(state(&store, "n5"), RecordState::Provisional);
    // Revoking from deeper in the chain fits the bound.
    store.revoke(&rid("n2")).unwrap();
    assert_eq!(state(&store, "n5"), RecordState::NeedsRevalidation);

    let breadth = MemoryStoreOptions {
        limits: MemoryLimits {
            max_revocation_visited: 2,
            ..MemoryLimits::default()
        },
        ..quiet()
    };
    let file2 = TempJournal::new("ms-bound2");
    let mut store = DurableMemoryStore::open(&file2.0, breadth).unwrap();
    store
        .insert(new_record("root", MemoryClass::Semantic, vec![]))
        .unwrap();
    for i in 0..3 {
        store
            .insert(derived_record(
                &format!("d{i}"),
                MemoryClass::Semantic,
                &["root"],
            ))
            .unwrap();
    }
    assert_eq!(
        store.revoke(&rid("root")),
        Err(MemoryStoreError::TraversalBoundExceeded(
            TraversalLimit::Visited(2)
        ))
    );
    assert_eq!(state(&store, "root"), RecordState::Provisional);
}

#[test]
fn input_bounds_are_enforced() {
    let file = TempJournal::new("ms-limits");
    let limits = MemoryLimits {
        max_records_per_class: 1,
        max_body_bytes: 8,
        max_supports: 1,
        ..MemoryLimits::default()
    };
    let mut store =
        DurableMemoryStore::open(&file.0, MemoryStoreOptions { limits, ..quiet() }).unwrap();
    let mut ok = new_record("a", MemoryClass::Semantic, vec![]);
    ok.body = "short".into();
    store.insert(ok.clone()).unwrap();
    assert_eq!(
        store.insert(ok.clone()),
        Err(MemoryStoreError::DuplicateRecord(rid("a")))
    );
    let mut second = new_record("b", MemoryClass::Semantic, vec![]);
    second.body = "short".into();
    assert!(matches!(
        store.insert(second),
        Err(MemoryStoreError::CapacityExceeded { limit: 1, .. })
    ));
    let mut long = new_record("c", MemoryClass::Episodic, vec![]);
    long.body = "far too long".into();
    assert_eq!(
        store.insert(long),
        Err(MemoryStoreError::InvalidRecord(
            RecordViolation::BodyTooLarge
        ))
    );
    let mut empty = new_record("", MemoryClass::Episodic, vec![]);
    empty.body.clear();
    assert_eq!(
        store.insert(empty),
        Err(MemoryStoreError::InvalidRecord(RecordViolation::EmptyId))
    );
    store.add_support(&rid("a"), observation("s", 1)).unwrap();
    assert_eq!(
        store.add_support(&rid("a"), observation("t", 1)),
        Err(MemoryStoreError::InvalidRecord(
            RecordViolation::TooManySupports
        ))
    );
    assert_eq!(
        store.add_support(&rid("a"), observation("", 1)),
        Err(MemoryStoreError::InvalidRecord(
            RecordViolation::EmptySupportSource
        ))
    );
    assert!(matches!(
        store.transition(&rid("zzz"), RecordState::Retired),
        Err(MemoryStoreError::UnknownRecord(_))
    ));
}

#[test]
fn retired_and_revoked_records_are_immutable() {
    let file = TempJournal::new("ms-term");
    let mut store = open(&file);
    store
        .insert(new_record("a", MemoryClass::Episodic, vec![]))
        .unwrap();
    store.transition(&rid("a"), RecordState::Retired).unwrap();
    assert_eq!(
        store.add_support(&rid("a"), observation("s", 1)),
        Err(MemoryStoreError::TerminalRecord(rid("a")))
    );
}

#[test]
fn compaction_preserves_every_record_horizon_and_clock() {
    let file = TempJournal::new("ms-compact");
    let mut store = open(&file);
    for i in 0..40 {
        store
            .insert(new_record(
                &format!("r{i}"),
                MemoryClass::Episodic,
                vec![observation("s", 1)],
            ))
            .unwrap();
    }
    store.advance_committed_revision(Revision(1)).unwrap();
    store
        .transition(&rid("r3"), RecordState::Authoritative)
        .unwrap();
    store.revoke(&rid("r4")).unwrap();
    let snapshot: Vec<_> = store.records().cloned().collect();
    let (horizon, clock) = (store.committed_revision(), store.clock());
    let before = std::fs::metadata(&file.0).unwrap().len();
    store.compact().unwrap();
    assert!(std::fs::metadata(&file.0).unwrap().len() <= before);
    store
        .insert(new_record("after", MemoryClass::Semantic, vec![]))
        .unwrap();
    drop(store);
    let store = open(&file);
    assert_eq!(store.committed_revision(), horizon);
    assert_eq!(store.clock(), clock + 1);
    let restored: Vec<_> = store
        .records()
        .filter(|r| r.id != rid("after"))
        .cloned()
        .collect();
    assert_eq!(restored, snapshot);

    // An empty store compacts to just its horizon and clock.
    let file2 = TempJournal::new("ms-compact-empty");
    let mut empty = open(&file2);
    empty.advance_committed_revision(Revision(9)).unwrap();
    empty.compact().unwrap();
    drop(empty);
    assert_eq!(open(&file2).committed_revision(), Revision(9));
}

#[test]
fn automatic_compaction_does_not_change_observable_state() {
    let file = TempJournal::new("ms-auto");
    let options = MemoryStoreOptions {
        compact_after_records: Some(7),
        ..MemoryStoreOptions::default()
    };
    let mut store = DurableMemoryStore::open(&file.0, options).unwrap();
    for i in 0..50 {
        store
            .insert(new_record(&format!("r{i}"), MemoryClass::Semantic, vec![]))
            .unwrap();
    }
    let snapshot: Vec<_> = store.records().cloned().collect();
    drop(store);
    let store = DurableMemoryStore::open(&file.0, options).unwrap();
    assert_eq!(store.records().cloned().collect::<Vec<_>>(), snapshot);
}

#[test]
fn torn_tail_drops_only_the_last_mutation_of_a_memory_store() {
    let file = TempJournal::new("ms-torn");
    let mut store = open(&file);
    store
        .insert(new_record("a", MemoryClass::Semantic, vec![]))
        .unwrap();
    store
        .insert(new_record("b", MemoryClass::Semantic, vec![]))
        .unwrap();
    drop(store);
    let bytes = std::fs::read(&file.0).unwrap();
    std::fs::write(&file.0, &bytes[..bytes.len() - 5]).unwrap();
    let store = open(&file);
    assert!(store.recovered_truncated_bytes() > 0);
    assert!(store.get(&rid("a")).is_some());
    assert!(store.get(&rid("b")).is_none());
}

#[test]
fn undecodable_frame_is_a_decode_error() {
    let file = TempJournal::new("ms-decode");
    let (mut journal, _) = ptr_memory::MemoryJournal::open(
        &file.0,
        ptr_memory::JournalKind::Memory,
        ptr_memory::JournalOptions::default(),
    )
    .unwrap();
    journal.append(&[0x10, 1, 2]).unwrap();
    drop(journal);
    assert!(matches!(
        DurableMemoryStore::open(&file.0, quiet()),
        Err(MemoryStoreError::Decode {
            record_index: 0,
            ..
        })
    ));
}

#[test]
fn promotion_respects_the_semantic_class_capacity() {
    let file = TempJournal::new("ms-promote-cap");
    let limits = MemoryLimits {
        max_records_per_class: 1,
        ..MemoryLimits::default()
    };
    let mut store =
        DurableMemoryStore::open(&file.0, MemoryStoreOptions { limits, ..quiet() }).unwrap();
    store
        .insert(new_record("sem", MemoryClass::Semantic, vec![]))
        .unwrap();
    store
        .insert(new_record(
            "epi",
            MemoryClass::Episodic,
            vec![observation("a", 1)],
        ))
        .unwrap();
    store.advance_committed_revision(Revision(1)).unwrap();
    let policy = ptr_memory::ConsolidationPolicy {
        min_independent_sources: 1,
        episodic_ttl: 10,
    };
    let before = store.clock();
    let result = store.promote(
        &rid("epi"),
        &policy,
        &ptr_memory::PromotionContext::default(),
    );
    assert!(matches!(
        result,
        Err(MemoryStoreError::CapacityExceeded { limit: 1, .. })
    ));
    assert_eq!(store.get(&rid("epi")).unwrap().class, MemoryClass::Episodic);
    assert_eq!(store.records_of(MemoryClass::Semantic).count(), 1);
    assert_eq!(store.clock(), before, "nothing committed");
}

/// Blocks the compaction temp file with a directory so `rewrite` fails with an I/O error while
/// the journal itself stays intact and writable.
fn block_compaction(file: &TempJournal) -> std::path::PathBuf {
    let mut tmp = file.0.as_os_str().to_owned();
    tmp.push(".compact.tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::create_dir(&tmp).unwrap();
    tmp
}

#[test]
fn failed_automatic_compaction_does_not_fail_a_committed_mutation() {
    let file = TempJournal::new("ms-compact-fail");
    let options = MemoryStoreOptions {
        compact_after_records: Some(2),
        ..MemoryStoreOptions::default()
    };
    let mut store = DurableMemoryStore::open(&file.0, options).unwrap();
    let blocker = block_compaction(&file);
    store
        .insert(new_record("a", MemoryClass::Semantic, vec![]))
        .unwrap();
    // The second append reaches the threshold; compaction fails but the insert is durable.
    store
        .insert(new_record("b", MemoryClass::Semantic, vec![]))
        .expect("a durable mutation must not be reported as failed");
    assert_eq!(store.compaction_failures(), 1);
    assert!(store.last_compaction_error().is_some());
    assert!(store.get(&rid("b")).is_some());
    // Retrying the call would be wrong: it already took effect.
    assert!(matches!(
        store.insert(new_record("b", MemoryClass::Semantic, vec![])),
        Err(MemoryStoreError::DuplicateRecord(_))
    ));
    // Compaction is retried on a later commit once the fault clears.
    std::fs::remove_dir(&blocker).unwrap();
    store
        .insert(new_record("c", MemoryClass::Semantic, vec![]))
        .unwrap();
    assert!(store.last_compaction_error().is_none());
    drop(store);
    let reopened = DurableMemoryStore::open(&file.0, options).unwrap();
    assert_eq!(reopened.len(), 3);
}

fn strict(limits: MemoryLimits) -> MemoryStoreOptions {
    MemoryStoreOptions { limits, ..quiet() }
}

#[test]
fn reopening_with_a_stricter_class_capacity_is_rejected_and_keeps_the_journal() {
    let file = TempJournal::new("ms-strict-capacity");
    let mut store = open(&file);
    for id in ["a", "b"] {
        store
            .insert(new_record(
                id,
                MemoryClass::Semantic,
                vec![observation("s", 1)],
            ))
            .unwrap();
    }
    drop(store);
    let before = std::fs::read(&file.0).unwrap();
    let limits = MemoryLimits {
        max_records_per_class: 1,
        ..MemoryLimits::default()
    };
    let error = DurableMemoryStore::open(&file.0, strict(limits))
        .err()
        .unwrap();
    assert_eq!(
        error,
        MemoryStoreError::RecoveredCapacityExceeded {
            class: "semantic",
            count: 2,
            limit: 1
        }
    );
    assert_eq!(std::fs::read(&file.0).unwrap(), before);
    assert_eq!(open(&file).len(), 2);
}

#[test]
fn reopening_with_stricter_field_limits_is_rejected_per_violation() {
    let file = TempJournal::new("ms-strict-fields");
    let mut store = open(&file);
    let mut record = new_record(
        "rec-long-id",
        MemoryClass::Episodic,
        vec![observation("s1", 1), observation("s2", 1)],
    );
    record.body = "0123456789".to_string();
    store.insert(record).unwrap();
    store
        .insert(derived_record(
            "kid",
            MemoryClass::Episodic,
            &["rec-long-id"],
        ))
        .unwrap();
    drop(store);
    let cases = [
        (
            MemoryLimits {
                max_id_bytes: 3,
                ..MemoryLimits::default()
            },
            RecordViolation::IdTooLong,
        ),
        (
            MemoryLimits {
                max_body_bytes: 5,
                ..MemoryLimits::default()
            },
            RecordViolation::BodyTooLarge,
        ),
        (
            MemoryLimits {
                max_supports: 1,
                ..MemoryLimits::default()
            },
            RecordViolation::TooManySupports,
        ),
        (
            MemoryLimits {
                max_derived_from: 0,
                ..MemoryLimits::default()
            },
            RecordViolation::TooManyDerivations,
        ),
    ];
    for (limits, violation) in cases {
        let error = DurableMemoryStore::open(&file.0, strict(limits))
            .err()
            .unwrap();
        assert!(
            matches!(
                &error,
                MemoryStoreError::RecoveredRecordInvalid { violation: v, .. } if *v == violation
            ),
            "{violation:?}: {error:?}"
        );
    }
    assert_eq!(open(&file).len(), 2);
}
