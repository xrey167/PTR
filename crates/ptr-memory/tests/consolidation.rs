mod common;

use common::{derived, new_record, observation, rid, TempJournal};
use ptr_memory::{
    ConsolidationPolicy, Contradiction, DeferReason, DurableMemoryStore, MemoryClass,
    MemoryStoreOptions, PromotionContext, PromotionDecision, RecordState, RejectReason,
    RetentionVerdict, Support, SupportKind,
};
use ptr_types::{Probability, Revision};

fn open(file: &TempJournal) -> DurableMemoryStore {
    let options = MemoryStoreOptions {
        compact_after_records: None,
        ..MemoryStoreOptions::default()
    };
    DurableMemoryStore::open(&file.0, options).unwrap()
}

fn policy() -> ConsolidationPolicy {
    ConsolidationPolicy {
        min_independent_sources: 2,
        episodic_ttl: 10,
    }
}

fn contradiction(id: &str) -> Contradiction {
    Contradiction {
        record: rid(id),
        verifier: "unit-checker".into(),
        detail: "x != y".into(),
    }
}

#[test]
fn promotes_after_enough_independent_committed_sources() {
    let file = TempJournal::new("co-promote");
    let mut store = open(&file);
    store
        .insert(new_record(
            "e",
            MemoryClass::Episodic,
            vec![observation("s1", 1)],
        ))
        .unwrap();
    store.advance_committed_revision(Revision(1)).unwrap();
    let ctx = PromotionContext::default();
    // One source is not enough.
    assert_eq!(
        store.promote(&rid("e"), &policy(), &ctx).unwrap(),
        PromotionDecision::Defer(DeferReason::InsufficientIndependentSources { have: 1, need: 2 })
    );
    // The same source twice is still one independent source.
    store.add_support(&rid("e"), observation("s1", 1)).unwrap();
    assert!(matches!(
        store.promote(&rid("e"), &policy(), &ctx).unwrap(),
        PromotionDecision::Defer(DeferReason::InsufficientIndependentSources { have: 1, .. })
    ));
    assert_eq!(store.get(&rid("e")).unwrap().class, MemoryClass::Episodic);
    store.add_support(&rid("e"), observation("s2", 1)).unwrap();
    assert_eq!(
        store.promote(&rid("e"), &policy(), &ctx).unwrap(),
        PromotionDecision::Promote
    );
    let promoted = store.get(&rid("e")).unwrap();
    assert_eq!(promoted.class, MemoryClass::Semantic);
    assert_eq!(promoted.state, RecordState::Authoritative);
    // Class and state survive a restart.
    drop(store);
    let store = open(&file);
    assert_eq!(store.get(&rid("e")).unwrap().class, MemoryClass::Semantic);
}

#[test]
fn uncommitted_observations_defer_promotion_invariant_4() {
    let file = TempJournal::new("co-inv4");
    let mut store = open(&file);
    store
        .insert(new_record(
            "e",
            MemoryClass::Episodic,
            vec![observation("s1", 3), observation("s2", 3)],
        ))
        .unwrap();
    store.advance_committed_revision(Revision(2)).unwrap();
    assert_eq!(
        store
            .promote(&rid("e"), &policy(), &PromotionContext::default())
            .unwrap(),
        PromotionDecision::Defer(DeferReason::NoCommittedObservation)
    );
    assert_eq!(
        store.get(&rid("e")).unwrap().state,
        RecordState::Provisional
    );
    store.advance_committed_revision(Revision(3)).unwrap();
    assert_eq!(
        store
            .promote(&rid("e"), &policy(), &PromotionContext::default())
            .unwrap(),
        PromotionDecision::Promote
    );
}

#[test]
fn derived_or_search_only_support_is_rejected_invariant_5() {
    let file = TempJournal::new("co-inv5");
    let mut store = open(&file);
    store.advance_committed_revision(Revision(9)).unwrap();
    store
        .insert(new_record(
            "e",
            MemoryClass::Episodic,
            vec![
                derived("d1"),
                derived("d2"),
                derived("d3"),
                Support {
                    source: "idx".into(),
                    kind: SupportKind::SearchHit,
                },
            ],
        ))
        .unwrap();
    // Even with a perfect learned score and many "independent" sources.
    let ctx = PromotionContext {
        contradictions: vec![],
        learned_score: Probability::new(1.0),
    };
    assert_eq!(
        store.promote(&rid("e"), &policy(), &ctx).unwrap(),
        PromotionDecision::Reject(RejectReason::OnlyDerivedOrSearchSupport)
    );
    let record = store.get(&rid("e")).unwrap();
    assert_eq!(record.class, MemoryClass::Episodic);
    assert_eq!(record.state, RecordState::Provisional);
    // Mixed support: derived sources do not count toward the independence threshold.
    store
        .add_support(&rid("e"), observation("real", 1))
        .unwrap();
    assert_eq!(
        store.promote(&rid("e"), &policy(), &ctx).unwrap(),
        PromotionDecision::Defer(DeferReason::InsufficientIndependentSources { have: 1, need: 2 })
    );
}

#[test]
fn verifier_contradiction_blocks_promotion_whatever_the_score_invariant_11() {
    let file = TempJournal::new("co-inv11");
    let mut store = open(&file);
    store
        .insert(new_record(
            "e",
            MemoryClass::Episodic,
            vec![
                observation("s1", 1),
                observation("s2", 1),
                observation("s3", 1),
            ],
        ))
        .unwrap();
    store.advance_committed_revision(Revision(1)).unwrap();
    let blocked = PromotionContext {
        contradictions: vec![contradiction("e"), contradiction("unrelated")],
        learned_score: Probability::new(1.0),
    };
    assert_eq!(
        store.promote(&rid("e"), &policy(), &blocked).unwrap(),
        PromotionDecision::Reject(RejectReason::VerifierContradiction)
    );
    assert_eq!(store.get(&rid("e")).unwrap().class, MemoryClass::Episodic);
    // A contradiction about a different record does not block this one.
    let other = PromotionContext {
        contradictions: vec![contradiction("unrelated")],
        learned_score: Probability::new(0.0),
    };
    assert_eq!(
        store.promote(&rid("e"), &policy(), &other).unwrap(),
        PromotionDecision::Promote
    );
}

#[test]
fn promotion_is_limited_to_live_episodic_records() {
    let file = TempJournal::new("co-scope");
    let mut store = open(&file);
    store
        .insert(new_record(
            "sem",
            MemoryClass::Semantic,
            vec![observation("a", 1)],
        ))
        .unwrap();
    store
        .insert(new_record("gone", MemoryClass::Episodic, vec![]))
        .unwrap();
    store
        .transition(&rid("gone"), RecordState::Retired)
        .unwrap();
    store.advance_committed_revision(Revision(1)).unwrap();
    let ctx = PromotionContext::default();
    assert_eq!(
        store.promote(&rid("sem"), &policy(), &ctx).unwrap(),
        PromotionDecision::Reject(RejectReason::NotEpisodic)
    );
    assert_eq!(
        store.promote(&rid("gone"), &policy(), &ctx).unwrap(),
        PromotionDecision::Reject(RejectReason::TerminalState)
    );
    let empty = new_record("empty", MemoryClass::Episodic, vec![]);
    store.insert(empty).unwrap();
    assert_eq!(
        store.promote(&rid("empty"), &policy(), &ctx).unwrap(),
        PromotionDecision::Defer(DeferReason::NoSupport)
    );
}

#[test]
fn promotion_refuses_revoked_ancestry_even_when_policy_would_promote() {
    let file = TempJournal::new("co-ancestry");
    let mut store = open(&file);
    store
        .insert(new_record("parent", MemoryClass::Semantic, vec![]))
        .unwrap();
    let mut child = new_record(
        "child",
        MemoryClass::Episodic,
        vec![observation("a", 1), observation("b", 1)],
    );
    child.derived_from = vec![rid("parent")];
    store.insert(child).unwrap();
    store.advance_committed_revision(Revision(1)).unwrap();
    store.revoke(&rid("parent")).unwrap();
    assert!(store
        .promote(&rid("child"), &policy(), &PromotionContext::default())
        .is_err());
    assert_eq!(
        store.get(&rid("child")).unwrap().class,
        MemoryClass::Episodic
    );
}

#[test]
fn retention_forgets_stale_unpinned_episodes_and_demotes_contradicted_authority() {
    let file = TempJournal::new("co-retention");
    let mut store = open(&file);
    store
        .insert(new_record("old", MemoryClass::Episodic, vec![]))
        .unwrap();
    let mut pinned = new_record("pinned", MemoryClass::Episodic, vec![]);
    pinned.pinned = true;
    store.insert(pinned).unwrap();
    store
        .insert(new_record(
            "auth",
            MemoryClass::Semantic,
            vec![observation("a", 1)],
        ))
        .unwrap();
    store
        .insert(new_record("proc", MemoryClass::Procedural, vec![]))
        .unwrap();
    store.advance_committed_revision(Revision(1)).unwrap();
    store
        .transition(&rid("auth"), RecordState::Authoritative)
        .unwrap();

    // Young records are kept.
    assert!(store.apply_retention(&policy(), &[]).unwrap().is_empty());

    // Age the clock past the TTL with unrelated mutations.
    for i in 0..12 {
        store
            .insert(new_record(
                &format!("filler{i}"),
                MemoryClass::Semantic,
                vec![],
            ))
            .unwrap();
    }
    let applied = store
        .apply_retention(&policy(), &[contradiction("auth")])
        .unwrap();
    let verdicts: Vec<_> = applied
        .iter()
        .map(|a| (a.id.0.as_str(), a.verdict))
        .collect();
    assert!(verdicts.contains(&("old", RetentionVerdict::Forget)));
    assert!(verdicts.contains(&("auth", RetentionVerdict::Demote)));
    assert!(!verdicts
        .iter()
        .any(|(id, _)| *id == "pinned" || *id == "proc"));
    assert_eq!(store.get(&rid("old")).unwrap().state, RecordState::Retired);
    assert_eq!(
        store.get(&rid("auth")).unwrap().state,
        RecordState::NeedsRevalidation
    );
    assert_eq!(
        store.get(&rid("pinned")).unwrap().state,
        RecordState::Provisional
    );
    // Non-episodic provisional records are not forgotten by the episodic TTL.
    assert_eq!(
        store.get(&rid("proc")).unwrap().state,
        RecordState::Provisional
    );

    // Applying again is a no-op, and everything persists.
    assert!(store
        .apply_retention(&policy(), &[contradiction("auth")])
        .unwrap()
        .is_empty());
    drop(store);
    let store = open(&file);
    assert_eq!(store.get(&rid("old")).unwrap().state, RecordState::Retired);
}
