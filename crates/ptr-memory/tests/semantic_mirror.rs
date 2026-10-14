mod common;

use common::{key, knowledge_object, new_record, observation, raw_event, rid, TempJournal};
use ptr_memory::{
    render_knowledge_mirror, render_memory_mirror, semantic_buckets_from_objects,
    semantic_objects_from_buckets, DurableKnowledgeStore, DurableMemoryStore, DurableOptions,
    MemoryClass, MemoryStoreOptions, RecordState, SemanticBuckets, SemanticCapsule,
    SemanticConversionError, SemanticObject, REDACTED,
};
use ptr_types::{CapsuleId, Generation, ProjectId, Revision, Validity};

fn capsule() -> SemanticCapsule {
    SemanticCapsule {
        id: CapsuleId::from("c"),
        project: ProjectId::from("p"),
        generation: Generation(1),
        revision_created: Revision(1),
        validity: Validity::Live,
        goals: vec!["ship".into()],
        hard_constraints: vec!["no network".into()],
        known: vec!["rust 1.99".into(), "linux".into()],
        hypotheses: vec!["fast enough".into()],
        unknowns: vec!["memory use".into()],
        relations: vec!["crate -[depends_on]-> sha2".into()],
        provenance: vec![],
    }
}

#[test]
fn capsule_buckets_convert_to_typed_objects_and_back_exactly() {
    let capsule = capsule();
    let objects = capsule.typed_objects().unwrap();
    assert_eq!(objects.len(), 7);
    assert_eq!(objects[0], SemanticObject::Goal("ship".into()));
    assert_eq!(objects[1], SemanticObject::Constraint("no network".into()));
    assert_eq!(
        objects[6],
        SemanticObject::Relation {
            subject: "crate".into(),
            predicate: "depends_on".into(),
            object: "sha2".into()
        }
    );
    let mut rebuilt = capsule.clone();
    rebuilt.goals.clear();
    rebuilt.known.clear();
    rebuilt.relations.clear();
    rebuilt.set_typed_objects(&objects).unwrap();
    assert_eq!(rebuilt, capsule);
}

#[test]
fn bucket_object_bucket_round_trip_is_identity() {
    let buckets = SemanticBuckets {
        goals: vec!["g1".into(), "g2".into()],
        hard_constraints: vec![],
        known: vec!["k".into()],
        hypotheses: vec!["h".into()],
        unknowns: vec!["u1".into(), "u2".into()],
        relations: vec!["a -[b]-> c -[d]-> e".into()],
    };
    let objects = semantic_objects_from_buckets(&buckets).unwrap();
    assert_eq!(semantic_buckets_from_objects(&objects).unwrap(), buckets);
}

#[test]
fn malformed_relation_leaves_the_capsule_unchanged() {
    let mut capsule = capsule();
    capsule.relations.push("not a relation".into());
    assert_eq!(
        capsule.typed_objects(),
        Err(SemanticConversionError::RelationFormat {
            raw: "not a relation".into()
        })
    );
    let original = capsule.clone();
    let bad = [
        SemanticObject::Goal("kept?".into()),
        SemanticObject::Relation {
            subject: String::new(),
            predicate: "p".into(),
            object: "o".into(),
        },
    ];
    assert_eq!(
        capsule.set_typed_objects(&bad),
        Err(SemanticConversionError::UnrepresentableRelation { field: "subject" })
    );
    assert_eq!(capsule, original);
}

fn memory_fixture(order: &[&str], file: &TempJournal) -> DurableMemoryStore {
    let options = MemoryStoreOptions {
        compact_after_records: None,
        ..MemoryStoreOptions::default()
    };
    let mut store = DurableMemoryStore::open(&file.0, options).unwrap();
    for id in order {
        let class = match id.chars().next() {
            Some('s') => MemoryClass::Semantic,
            Some('e') => MemoryClass::Episodic,
            Some('p') => MemoryClass::Procedural,
            _ => MemoryClass::Epistemic,
        };
        let mut record = new_record(
            id,
            class,
            vec![observation("src-b", 1), observation("src-a", 1)],
        );
        record.body = format!("line one\n# fake heading for {id}");
        if *id == "secret" {
            record.sensitive = true;
            record.body = "api-key-hunter2".into();
        }
        store.insert(record).unwrap();
    }
    store
}

#[test]
fn memory_mirror_is_deterministic_sorted_and_independent_of_insert_order() {
    let a = TempJournal::new("mir-a");
    let b = TempJournal::new("mir-b");
    let store_a = memory_fixture(&["s2", "e1", "p1", "x1", "s1"], &a);
    let store_b = memory_fixture(&["s1", "x1", "p1", "e1", "s2"], &b);
    let text_a = render_memory_mirror(&store_a);
    // Timestamps (logical clock ticks) differ with insert order, so compare the structure
    // that must be identical: ordering of headings and bodies.
    let headings: Vec<&str> = text_a.lines().filter(|l| l.starts_with("##")).collect();
    assert_eq!(
        headings,
        vec![
            "## semantic",
            "### s1",
            "### s2",
            "## episodic",
            "### e1",
            "## procedural",
            "### p1",
            "## epistemic",
            "### x1"
        ]
    );
    // Same store rendered twice, and after reopen, is byte-identical.
    assert_eq!(text_a, render_memory_mirror(&store_a));
    drop(store_a);
    let options = MemoryStoreOptions {
        compact_after_records: None,
        ..MemoryStoreOptions::default()
    };
    let reopened = DurableMemoryStore::open(&a.0, options).unwrap();
    assert_eq!(text_a, render_memory_mirror(&reopened));
    // Supports are sorted regardless of the order they were added in.
    assert!(text_a.contains("- supports: src-a (observation r1), src-b (observation r1)"));
    // Free text cannot forge headings.
    assert!(text_a.contains("- body: line one\\n# fake heading for s1"));
    assert!(!text_a.lines().any(|l| l.starts_with("# fake")));
    assert_eq!(
        render_memory_mirror(&store_b)
            .lines()
            .filter(|l| l.starts_with("###"))
            .count(),
        5
    );
}

#[test]
fn sensitive_bodies_are_never_written_to_the_mirror() {
    let file = TempJournal::new("mir-secret");
    let mut store = memory_fixture(&["s1", "secret"], &file);
    store.advance_committed_revision(Revision(1)).unwrap();
    store
        .transition(&rid("secret"), RecordState::Authoritative)
        .unwrap();
    let text = render_memory_mirror(&store);
    assert!(!text.contains("hunter2"));
    assert!(text.contains(&format!("- body: {REDACTED}")));
    assert!(!format!("{:?}", store.get(&rid("secret")).unwrap()).contains("hunter2"));
}

#[test]
fn knowledge_mirror_shows_digests_not_content_and_is_stable() {
    let file = TempJournal::new("mir-know");
    let options = DurableOptions {
        compact_after_records: None,
        ..DurableOptions::default()
    };
    let mut store = DurableKnowledgeStore::open(&file.0, options).unwrap();
    store
        .append_raw(raw_event("e1", "TOP-SECRET-PAYLOAD"))
        .unwrap();
    store
        .admit_object(knowledge_object("base", 1, &["e1"], vec![]))
        .unwrap();
    store
        .admit_object(knowledge_object("dep", 1, &["e1"], vec![key("base", 1)]))
        .unwrap();
    store.invalidate("base".into()).unwrap();
    let text = render_knowledge_mirror(store.store());
    assert!(!text.contains("TOP-SECRET-PAYLOAD"));
    assert!(text.contains("bytes=18"));
    assert!(text.contains("### base generation 1"));
    assert!(text.contains("- lifecycle: Invalidated"));
    assert_eq!(text, render_knowledge_mirror(store.store()));
    drop(store);
    let reopened = DurableKnowledgeStore::open(&file.0, options).unwrap();
    assert_eq!(text, render_knowledge_mirror(reopened.store()));
}

fn raw_event_in_session(id: &str, session: &str) -> ptr_memory::RawEvent {
    ptr_memory::RawEvent::new(
        ptr_types::RawEventId::from(id),
        ptr_types::SessionId::from(session),
        ptr_types::EventRole::User,
        b"payload".to_vec(),
        ptr_types::Timestamp(1),
    )
}

#[test]
fn session_ids_are_digested_not_written_to_the_knowledge_mirror() {
    let token = "sk-live-0123456789abcdef-SECRET";
    let file = TempJournal::new("mirror-session");
    let mut store = DurableKnowledgeStore::open(&file.0, DurableOptions::default()).unwrap();
    store.append_raw(raw_event_in_session("e1", token)).unwrap();
    store.append_raw(raw_event_in_session("e2", token)).unwrap();
    store
        .append_raw(raw_event_in_session("e3", "other-session"))
        .unwrap();
    let text = render_knowledge_mirror(store.store());
    assert!(!text.contains(token), "{text}");
    assert!(!text.contains("other-session"), "{text}");
    let digests: Vec<&str> = text
        .lines()
        .filter_map(|l| l.split(" session=").nth(1))
        .filter_map(|rest| rest.split(' ').next())
        .collect();
    assert_eq!(digests.len(), 3);
    assert!(digests[0].starts_with("session:sha256:") && digests[0].len() == 15 + 16);
    assert_eq!(digests[0], digests[1]);
    assert_ne!(digests[0], digests[2]);
}
