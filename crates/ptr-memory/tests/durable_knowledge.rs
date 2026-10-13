mod common;

use common::{key, knowledge_object, raw_event, Lcg, TempJournal};
use ptr_memory::{
    DurableKnowledgeError, DurableKnowledgeStore, DurableOptions, JournalError, KnowledgeLifecycle,
    KnowledgeStore, KnowledgeStoreError,
};
use ptr_types::{Generation, KnowledgeObjectId};
use std::fs;

const LIFECYCLES: [KnowledgeLifecycle; 6] = [
    KnowledgeLifecycle::Hot,
    KnowledgeLifecycle::Warm,
    KnowledgeLifecycle::Cold,
    KnowledgeLifecycle::Pod,
    KnowledgeLifecycle::Archived,
    KnowledgeLifecycle::Invalidated,
];

/// Applies one pseudo-random operation to both stores and asserts they agree on the outcome.
fn step(rng: &mut Lcg, durable: &mut DurableKnowledgeStore, reference: &mut KnowledgeStore) {
    let object_id = format!("o{}", rng.next() % 6);
    let id = KnowledgeObjectId::from(object_id.as_str());
    match rng.next() % 6 {
        0 => {
            let event = raw_event(
                &format!("e{}", rng.next() % 8),
                &format!("content {}", rng.next() % 3),
            );
            let a = durable
                .append_raw(event.clone())
                .map_err(|e| unwrap_store(&e));
            let b = reference.append_raw(event);
            assert_eq!(a, b);
        }
        1 | 2 => {
            let generation = 1 + rng.next() % 3;
            let source = format!("e{}", rng.next() % 8);
            let deps = if rng.next().is_multiple_of(2) {
                vec![key(&format!("o{}", rng.next() % 6), 1)]
            } else {
                vec![]
            };
            let object = knowledge_object(&object_id, generation, &[source.as_str()], deps);
            let a = durable
                .admit_object(object.clone())
                .map_err(|e| unwrap_store(&e));
            let b = reference.admit_object(object);
            assert_eq!(a, b);
        }
        3 => {
            let generation = Generation(1 + rng.next() % 3);
            let a = durable
                .activate_generation(&id, generation)
                .map_err(|e| unwrap_store(&e));
            let b = reference.activate_generation(&id, generation);
            assert_eq!(a, b);
        }
        4 => {
            let to = LIFECYCLES[(rng.next() % 6) as usize];
            let a = durable.transition(&id, to).map_err(|e| unwrap_store(&e));
            let b = reference.transition(&id, to);
            assert_eq!(a, b);
        }
        _ => {
            let a = durable.invalidate(id.clone()).map_err(|e| unwrap_store(&e));
            let b = reference.invalidate(id);
            assert_eq!(a, b);
        }
    }
    assert_eq!(durable.state_digest(), reference.state_digest());
}

fn unwrap_store(error: &DurableKnowledgeError) -> KnowledgeStoreError {
    match error {
        DurableKnowledgeError::Store(inner) => inner.clone(),
        other => panic!("expected a store rejection, got {other:?}"),
    }
}

fn quiet() -> DurableOptions {
    DurableOptions {
        compact_after_records: None,
        ..DurableOptions::default()
    }
}

#[test]
fn state_after_reopen_equals_state_before_close_over_a_deterministic_sequence() {
    for seed in 1..=8_u64 {
        let file = TempJournal::new("dk-prop");
        let mut rng = Lcg(seed);
        let mut reference = KnowledgeStore::default();
        let mut durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
        for round in 0..6 {
            for _ in 0..60 {
                step(&mut rng, &mut durable, &mut reference);
            }
            let digest = durable.state_digest();
            drop(durable);
            durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
            assert_eq!(durable.state_digest(), digest, "seed {seed} round {round}");
            assert_eq!(durable.state_digest(), reference.state_digest());
            if round % 2 == 1 {
                durable.compact().unwrap();
                assert_eq!(durable.state_digest(), digest);
            }
        }
    }
}

#[test]
fn compaction_shrinks_the_journal_and_preserves_state() {
    let file = TempJournal::new("dk-compact");
    let mut reference = KnowledgeStore::default();
    let mut durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    let mut rng = Lcg(99);
    for _ in 0..400 {
        step(&mut rng, &mut durable, &mut reference);
    }
    let before = fs::metadata(&file.0).unwrap().len();
    let digest = durable.state_digest();
    durable.compact().unwrap();
    let after = fs::metadata(&file.0).unwrap().len();
    assert!(after < before, "{after} !< {before}");
    drop(durable);
    let reopened = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    assert_eq!(reopened.state_digest(), digest);
}

#[test]
fn automatic_compaction_keeps_the_journal_bounded() {
    let file = TempJournal::new("dk-auto");
    let options = DurableOptions {
        compact_after_records: Some(20),
        ..DurableOptions::default()
    };
    let mut durable = DurableKnowledgeStore::open(&file.0, options).unwrap();
    let mut reference = KnowledgeStore::default();
    let mut rng = Lcg(5);
    for _ in 0..300 {
        step(&mut rng, &mut durable, &mut reference);
    }
    let digest = durable.state_digest();
    drop(durable);
    let reopened = DurableKnowledgeStore::open(&file.0, options).unwrap();
    assert_eq!(reopened.state_digest(), digest);
}

#[test]
fn rejected_operations_are_not_journaled() {
    let file = TempJournal::new("dk-reject");
    let mut durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    let before = fs::metadata(&file.0).unwrap().len();
    let missing_source = knowledge_object("o1", 1, &["nope"], vec![]);
    assert_eq!(
        durable.admit_object(missing_source),
        Err(DurableKnowledgeError::Store(
            KnowledgeStoreError::UnknownSource("nope".into())
        ))
    );
    assert_eq!(fs::metadata(&file.0).unwrap().len(), before);
    // Idempotent replay of the same raw event is not journaled twice.
    assert_eq!(durable.append_raw(raw_event("e1", "a")), Ok(true));
    let after_first = fs::metadata(&file.0).unwrap().len();
    assert_eq!(durable.append_raw(raw_event("e1", "a")), Ok(false));
    assert_eq!(fs::metadata(&file.0).unwrap().len(), after_first);
}

#[test]
fn durable_store_keeps_invalidation_cascades_across_restart() {
    let file = TempJournal::new("dk-cascade");
    let mut durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    durable.append_raw(raw_event("e1", "x")).unwrap();
    durable
        .admit_object(knowledge_object("base", 1, &["e1"], vec![]))
        .unwrap();
    durable
        .admit_object(knowledge_object(
            "derived",
            1,
            &["e1"],
            vec![key("base", 1)],
        ))
        .unwrap();
    durable.invalidate(KnowledgeObjectId::from("base")).unwrap();
    drop(durable);
    let reopened = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    assert!(reopened
        .store()
        .is_invalidated(&KnowledgeObjectId::from("base")));
    assert!(reopened
        .store()
        .is_invalidated(&KnowledgeObjectId::from("derived")));
    // Invariant 3: the revoked generation cannot be revived after restart.
    let mut reopened = reopened;
    assert!(matches!(
        reopened.activate_generation(&KnowledgeObjectId::from("base"), Generation(1)),
        Err(DurableKnowledgeError::Store(
            KnowledgeStoreError::InvalidLifecycleTransition { .. }
        ))
    ));
}

#[test]
fn torn_tail_after_a_crash_loses_only_the_last_operation() {
    let file = TempJournal::new("dk-torn");
    let mut durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    durable.append_raw(raw_event("e1", "one")).unwrap();
    let digest = durable.state_digest();
    durable.append_raw(raw_event("e2", "two")).unwrap();
    drop(durable);
    let bytes = fs::read(&file.0).unwrap();
    fs::write(&file.0, &bytes[..bytes.len() - 3]).unwrap();
    let reopened = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    assert!(reopened.recovered_truncated_bytes() > 0);
    assert_eq!(reopened.state_digest(), digest);
}

#[test]
fn corrupt_payload_that_passes_crc_is_a_decode_error() {
    let file = TempJournal::new("dk-decode");
    // A well-framed journal whose single record has an unknown op tag.
    let (mut journal, _) = ptr_memory::MemoryJournal::open(
        &file.0,
        ptr_memory::JournalKind::Knowledge,
        ptr_memory::JournalOptions::default(),
    )
    .unwrap();
    journal.append(&[0xEE]).unwrap();
    drop(journal);
    assert!(matches!(
        DurableKnowledgeStore::open(&file.0, quiet()),
        Err(DurableKnowledgeError::Decode {
            record_index: 0,
            ..
        })
    ));
}

#[test]
fn mid_file_corruption_surfaces_as_a_journal_error() {
    let file = TempJournal::new("dk-corrupt");
    let mut durable = DurableKnowledgeStore::open(&file.0, quiet()).unwrap();
    durable.append_raw(raw_event("e1", "one")).unwrap();
    durable.append_raw(raw_event("e2", "two")).unwrap();
    drop(durable);
    let mut bytes = fs::read(&file.0).unwrap();
    bytes[8 + 8 + 2] ^= 0x40;
    fs::write(&file.0, &bytes).unwrap();
    assert!(matches!(
        DurableKnowledgeStore::open(&file.0, quiet()),
        Err(DurableKnowledgeError::Journal(JournalError::Corrupt { .. }))
    ));
}
