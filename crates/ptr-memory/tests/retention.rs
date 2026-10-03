use ptr_memory::{
    compile_structured, decide, BasicKnowledgeObjectBuilder, ConflictState,
    DeterministicEntityExtractor, DeterministicTokenizer, EntityExtractor, Freshness,
    InMemoryKvStateRuntime, KnowledgeKey, KnowledgeLifecycle, KnowledgeObject,
    KnowledgeObjectBuilder, KnowledgeStore, KvStateRuntime, NamespaceContext, NamespaceResolver,
    RetentionAction, RetentionCandidate, RetentionError, RetentionModel, RetentionReason,
    RetentionScore, RetentionState, SemanticTokenizer, StaticNamespaceResolver, TokenBudget,
};
use ptr_types::{
    AdapterVersion, EntityId, EventRole, EvidenceId, Generation, KnowledgeObjectId, ModelVersion,
    NamespaceId, PolicyVersion, Probability, ProvenanceRef, RawEventId, SessionId, Timestamp,
};

fn probability(value: f32) -> Probability {
    Probability::new(value).unwrap()
}

#[test]
fn retention_preserves_pinned_objects() {
    let candidate = RetentionCandidate {
        object_id: KnowledgeObjectId::from("k1"),
        source_events: vec![RawEventId::from("e1")],
        token_cost: 10,
        pinned: true,
    };
    let decision = decide(
        &candidate,
        RetentionScore {
            object_id: candidate.object_id.clone(),
            keep_call: probability(0.0),
            keep_result: probability(0.0),
        },
        probability(0.5),
        PolicyVersion::from("v1"),
    );
    let decision = decision.unwrap();
    assert_eq!(decision.action, RetentionAction::KeepVerbatim);
    assert_eq!(decision.reason, RetentionReason::Pinned);
}

#[test]
fn retention_drops_result_before_dropping_call() {
    let candidate = RetentionCandidate {
        object_id: KnowledgeObjectId::from("k1"),
        source_events: vec![],
        token_cost: 10,
        pinned: false,
    };
    let decision = decide(
        &candidate,
        RetentionScore {
            object_id: candidate.object_id.clone(),
            keep_call: probability(0.8),
            keep_result: probability(0.1),
        },
        probability(0.5),
        PolicyVersion::from("v1"),
    );
    assert_eq!(decision.unwrap().action, RetentionAction::DropResult);
}

#[test]
fn invalidation_is_not_archive() {
    assert_ne!(
        KnowledgeLifecycle::Invalidated,
        KnowledgeLifecycle::Archived
    );
}

fn object(
    id: &str,
    lifecycle: KnowledgeLifecycle,
    dependencies: Vec<KnowledgeKey>,
) -> KnowledgeObject {
    KnowledgeObject {
        id: KnowledgeObjectId::from(id),
        namespace: NamespaceId::from("ptr.test"),
        entities: vec![EntityId::from("entity")],
        relations: vec![],
        sources: vec![],
        provenance: vec![ProvenanceRef {
            source: EvidenceId::from("test"),
            note: None,
        }],
        generation: Generation(1),
        supersedes: None,
        dependencies,
        confidence: probability(1.0),
        relevance: probability(1.0),
        freshness: Freshness::Current,
        conflict: ConflictState::None,
        lifecycle,
        retrieval_keys: vec![],
        token_cost: 1,
    }
}

#[test]
fn raw_history_is_digest_checked_and_idempotent() {
    let event = ptr_memory::RawEvent::new(
        RawEventId::from("e1"),
        SessionId::from("s1"),
        EventRole::Tool,
        b"result".to_vec(),
        Timestamp(1),
    );
    let mut store = KnowledgeStore::default();
    assert!(store.append_raw(event.clone()).unwrap());
    assert!(!store.append_raw(event).unwrap());
}

#[test]
fn invalidation_propagates_to_dependents() {
    let mut store = KnowledgeStore::default();
    store
        .register_object(object("base", KnowledgeLifecycle::Hot, vec![]))
        .unwrap();
    store
        .register_object(object(
            "derived",
            KnowledgeLifecycle::Hot,
            vec![KnowledgeKey {
                logical_id: KnowledgeObjectId::from("base"),
                generation: Generation(1),
            }],
        ))
        .unwrap();
    store.invalidate(KnowledgeObjectId::from("base")).unwrap();
    assert!(store.is_invalidated(&KnowledgeObjectId::from("derived")));
}

#[test]
fn context_compiler_rejects_missing_dependencies_and_budget_overflow() {
    let derived = object(
        "derived",
        KnowledgeLifecycle::Warm,
        vec![KnowledgeKey {
            logical_id: KnowledgeObjectId::from("missing"),
            generation: Generation(1),
        }],
    );
    assert!(compile_structured(&[derived], TokenBudget(10)).is_err());
    let base = object("base", KnowledgeLifecycle::Warm, vec![]);
    assert!(compile_structured(&[base], TokenBudget(0)).is_err());
}

struct DuplicateScoreModel;

impl RetentionModel for DuplicateScoreModel {
    fn score(&self, state: &RetentionState) -> Result<Vec<RetentionScore>, String> {
        Ok(state
            .candidates
            .iter()
            .map(|_candidate| RetentionScore {
                object_id: state.candidates[0].object_id.clone(),
                keep_call: probability(0.5),
                keep_result: probability(0.5),
            })
            .collect())
    }
}

#[test]
fn retention_rejects_duplicate_scores() {
    let state = RetentionState {
        goal: "test".into(),
        candidates: vec![
            RetentionCandidate {
                object_id: KnowledgeObjectId::from("a"),
                source_events: vec![],
                token_cost: 1,
                pinned: false,
            },
            RetentionCandidate {
                object_id: KnowledgeObjectId::from("b"),
                source_events: vec![],
                token_cost: 1,
                pinned: false,
            },
        ],
    };
    assert!(matches!(
        ptr_memory::score_candidates(&DuplicateScoreModel, &state),
        Err(RetentionError::UnknownScore(_))
    ));
}

#[test]
fn raw_history_rejects_tampering_and_conflicting_reuse() {
    let event = ptr_memory::RawEvent::new(
        RawEventId::from("e2"),
        SessionId::from("s1"),
        EventRole::Tool,
        b"original".to_vec(),
        Timestamp(2),
    );
    let mut store = KnowledgeStore::default();
    store.append_raw(event.clone()).unwrap();
    let conflict = ptr_memory::RawEvent::new(
        RawEventId::from("e2"),
        SessionId::from("s1"),
        EventRole::Tool,
        b"changed".to_vec(),
        Timestamp(2),
    );
    assert!(matches!(
        store.append_raw(conflict),
        Err(ptr_memory::KnowledgeStoreError::ConflictingRawEvent(_))
    ));
    let mut tampered = event;
    tampered.content = b"tampered".to_vec();
    assert!(matches!(
        store.append_raw(tampered),
        Err(ptr_memory::KnowledgeStoreError::InvalidRawDigest(_))
    ));
}

#[test]
fn knowledge_generation_is_immutable_and_context_digest_tracks_content() {
    let original = object("fact", KnowledgeLifecycle::Hot, vec![]);
    let mut store = KnowledgeStore::default();
    store.register_object(original.clone()).unwrap();
    let mut changed = original.clone();
    changed.relevance = probability(0.5);
    assert!(matches!(
        store.register_object(changed),
        Err(ptr_memory::KnowledgeStoreError::ConflictingKnowledgeGeneration(_))
    ));

    let first = compile_structured(&[original], TokenBudget(10)).unwrap();
    let mut next = object("fact", KnowledgeLifecycle::Hot, vec![]);
    next.generation = Generation(2);
    next.relevance = probability(0.5);
    let second = compile_structured(&[next], TokenBudget(10)).unwrap();
    assert_ne!(first.digest, second.digest);
}

#[test]
fn lifecycle_transition_does_not_mutate_immutable_generation_digest() {
    let original = object("lifecycle", KnowledgeLifecycle::Hot, vec![]);
    let mut store = KnowledgeStore::default();
    store.register_object(original.clone()).unwrap();
    let before = store.object(&original.id).unwrap();
    let before_digest = before.content_digest();

    store
        .transition(&original.id, KnowledgeLifecycle::Warm)
        .unwrap();

    let after = store.object(&original.id).unwrap();
    assert_eq!(before_digest, after.content_digest());
    assert_eq!(after.lifecycle, KnowledgeLifecycle::Warm);
}

#[test]
fn dropped_dependency_cannot_satisfy_an_active_object() {
    let base = object("base", KnowledgeLifecycle::Warm, vec![]);
    let mut derived = object("derived", KnowledgeLifecycle::Warm, vec![]);
    derived.dependencies = vec![KnowledgeKey {
        logical_id: base.id.clone(),
        generation: base.generation,
    }];
    let candidates = vec![
        ptr_memory::ContextCandidate {
            object: base,
            action: RetentionAction::DropResult,
            verbatim: None,
            pinned: false,
        },
        ptr_memory::ContextCandidate {
            object: derived,
            action: RetentionAction::KeepStructured,
            verbatim: None,
            pinned: false,
        },
    ];
    assert_eq!(
        ptr_memory::compile_selected(&candidates, TokenBudget(10)),
        Err(ptr_memory::ContextCompileError::MissingDependency)
    );
}

#[test]
fn context_digest_is_independent_of_equal_priority_input_order() {
    let first = object("a", KnowledgeLifecycle::Warm, vec![]);
    let second = object("b", KnowledgeLifecycle::Warm, vec![]);
    let one =
        ptr_memory::compile_structured(&[first.clone(), second.clone()], TokenBudget(10)).unwrap();
    let two = ptr_memory::compile_structured(&[second, first], TokenBudget(10)).unwrap();
    assert_eq!(one.digest, two.digest);
    assert_eq!(one.entries, two.entries);
}

#[test]
fn kv_runtime_continues_and_requires_recompute_on_dependency_invalidation() {
    let context = compile_structured(
        &[object("base", KnowledgeLifecycle::Warm, vec![])],
        TokenBudget(10),
    )
    .unwrap();
    let mut runtime = InMemoryKvStateRuntime::new(
        SessionId::from("session"),
        ModelVersion::from("model-v1"),
        AdapterVersion::from("adapter-v1"),
    );
    let state = runtime.recompute(context.clone()).unwrap();
    let continued = runtime.continue_state(&state, context).unwrap();
    assert_eq!(
        runtime.metadata(&continued).unwrap().parent_state_id,
        Some(runtime.metadata(&state).unwrap().state_id)
    );
    let descendant_context = compile_structured(
        &[object("base", KnowledgeLifecycle::Warm, vec![])],
        TokenBudget(10),
    )
    .unwrap();
    let descendant = runtime
        .continue_state(&continued, descendant_context)
        .unwrap();
    let dependency = runtime.metadata(&continued).unwrap().dependency_digests[0];
    let invalidated = runtime.invalidate_dependencies(&[dependency]).unwrap();
    assert_eq!(invalidated.len(), 3);
    assert!(runtime
        .continue_state(
            &continued,
            compile_structured(
                &[object("base", KnowledgeLifecycle::Warm, vec![])],
                TokenBudget(10)
            )
            .unwrap()
        )
        .is_err());
    assert!(runtime
        .continue_state(
            &descendant,
            compile_structured(
                &[object("base", KnowledgeLifecycle::Warm, vec![])],
                TokenBudget(10),
            )
            .unwrap(),
        )
        .is_err());
}

#[test]
fn kv_handles_are_owned_by_their_runtime() {
    let context = compile_structured(
        &[object("foreign", KnowledgeLifecycle::Warm, vec![])],
        TokenBudget(10),
    )
    .unwrap();
    let mut first = InMemoryKvStateRuntime::new(
        SessionId::from("session"),
        ModelVersion::from("model-v1"),
        AdapterVersion::from("adapter-v1"),
    );
    let mut second = InMemoryKvStateRuntime::new(
        SessionId::from("session"),
        ModelVersion::from("model-v1"),
        AdapterVersion::from("adapter-v1"),
    );
    let handle = first.recompute(context.clone()).unwrap();
    assert_eq!(
        second.continue_state(&handle, context),
        Err(ptr_memory::KvError::ForeignHandle)
    );
}

#[test]
fn deterministic_ingestion_builds_typed_knowledge_object() {
    let event = ptr_memory::RawEvent::new(
        RawEventId::from("ingest-1"),
        SessionId::from("session"),
        EventRole::Tool,
        b"supplier delivery".to_vec(),
        Timestamp(2),
    );
    let tokenized = DeterministicTokenizer.tokenize(&event).unwrap();
    let entities = DeterministicEntityExtractor.extract(&tokenized).unwrap();
    let namespaces = StaticNamespaceResolver
        .resolve(
            &entities,
            &NamespaceContext {
                namespaces: vec![NamespaceId::from("ptr.supply")],
            },
        )
        .unwrap();
    let object = BasicKnowledgeObjectBuilder
        .build(&tokenized, &entities, &namespaces)
        .unwrap();
    assert_eq!(object.namespace, NamespaceId::from("ptr.supply"));
    assert_eq!(object.sources, vec![RawEventId::from("ingest-1")]);
    assert_eq!(object.generation, Generation(1));
}

fn key(id: &str, generation: u64) -> KnowledgeKey {
    KnowledgeKey {
        logical_id: KnowledgeObjectId::from(id),
        generation: Generation(generation),
    }
}

fn successor(
    id: &str,
    lifecycle: KnowledgeLifecycle,
    dependencies: Vec<KnowledgeKey>,
) -> KnowledgeObject {
    let mut next = object(id, lifecycle, dependencies);
    next.generation = Generation(2);
    next.supersedes = Some(key(id, 1));
    next
}

#[test]
fn transition_to_invalidated_cascades_like_invalidate() {
    let mut store = KnowledgeStore::default();
    store
        .register_object(object("base", KnowledgeLifecycle::Hot, vec![]))
        .unwrap();
    store
        .register_object(object(
            "derived",
            KnowledgeLifecycle::Hot,
            vec![key("base", 1)],
        ))
        .unwrap();
    store
        .transition(
            &KnowledgeObjectId::from("base"),
            KnowledgeLifecycle::Invalidated,
        )
        .unwrap();
    assert!(store.is_invalidated(&KnowledgeObjectId::from("derived")));
}

#[test]
fn reactivating_a_generation_keeps_its_recorded_demotion() {
    let mut store = KnowledgeStore::default();
    let id = KnowledgeObjectId::from("topic");
    store
        .register_object(object("topic", KnowledgeLifecycle::Hot, vec![]))
        .unwrap();
    store
        .register_object(successor("topic", KnowledgeLifecycle::Hot, vec![]))
        .unwrap();
    store.activate_generation(&id, Generation(2)).unwrap();
    store.transition(&id, KnowledgeLifecycle::Warm).unwrap();
    store.activate_generation(&id, Generation(2)).unwrap();
    assert_eq!(store.lifecycle(&id), Some(KnowledgeLifecycle::Warm));
}

#[test]
fn activation_that_would_invalidate_itself_is_refused_without_side_effects() {
    let mut store = KnowledgeStore::default();
    let id = KnowledgeObjectId::from("topic");
    store
        .register_object(object("topic", KnowledgeLifecycle::Hot, vec![]))
        .unwrap();
    store
        .register_object(successor(
            "topic",
            KnowledgeLifecycle::Hot,
            vec![key("topic", 1)],
        ))
        .unwrap();
    assert!(store.activate_generation(&id, Generation(2)).is_err());
    assert!(!store.is_generation_invalidated(&id, Generation(1)));
    assert!(!store.is_generation_invalidated(&id, Generation(2)));
    assert_eq!(store.object(&id).unwrap().generation, Generation(1));
}

#[test]
fn invalidation_transition_reaches_a_diamond_once_and_preserves_other_generations() {
    let mut store = KnowledgeStore::default();
    for item in [
        object("base", KnowledgeLifecycle::Hot, vec![]),
        successor("base", KnowledgeLifecycle::Cold, vec![]),
        object("left", KnowledgeLifecycle::Warm, vec![key("base", 1)]),
        object("right", KnowledgeLifecycle::Cold, vec![key("base", 1)]),
        object(
            "leaf",
            KnowledgeLifecycle::Hot,
            vec![key("left", 1), key("right", 1)],
        ),
        object("other", KnowledgeLifecycle::Hot, vec![key("base", 2)]),
    ] {
        store.register_object(item).unwrap();
    }
    let base = KnowledgeObjectId::from("base");
    store
        .transition(&base, KnowledgeLifecycle::Invalidated)
        .unwrap();
    for id in ["base", "left", "right", "leaf"] {
        assert!(
            store.is_generation_invalidated(&KnowledgeObjectId::from(id), Generation(1)),
            "{id}"
        );
    }
    assert_eq!(
        store.object_at(&key("base", 2)).unwrap().lifecycle,
        KnowledgeLifecycle::Cold
    );
    assert_eq!(
        store.lifecycle(&KnowledgeObjectId::from("other")),
        Some(KnowledgeLifecycle::Hot)
    );
    assert_eq!(
        store.transition(&base, KnowledgeLifecycle::Invalidated),
        Ok(())
    );
}

#[test]
fn transitive_self_invalidation_refuses_activation_without_changing_any_object() {
    let mut store = KnowledgeStore::default();
    for item in [
        object("topic", KnowledgeLifecycle::Warm, vec![]),
        object("middle", KnowledgeLifecycle::Cold, vec![key("topic", 1)]),
        object("sibling", KnowledgeLifecycle::Hot, vec![key("topic", 1)]),
        successor("topic", KnowledgeLifecycle::Hot, vec![key("middle", 1)]),
    ] {
        store.register_object(item).unwrap();
    }
    let keys = [
        key("topic", 1),
        key("topic", 2),
        key("middle", 1),
        key("sibling", 1),
    ];
    let before: Vec<_> = keys.iter().map(|key| store.object_at(key)).collect();
    assert_eq!(
        store.activate_generation(&KnowledgeObjectId::from("topic"), Generation(2)),
        Err(
            ptr_memory::KnowledgeStoreError::InvalidLifecycleTransition {
                from: KnowledgeLifecycle::Hot,
                to: KnowledgeLifecycle::Hot,
            }
        )
    );
    assert_eq!(
        keys.iter()
            .map(|key| store.object_at(key))
            .collect::<Vec<_>>(),
        before
    );
    assert_eq!(
        store
            .object(&KnowledgeObjectId::from("topic"))
            .unwrap()
            .generation,
        Generation(1)
    );
}

#[test]
fn successful_activation_invalidates_old_dependents_and_keeps_new_generation_lifecycle() {
    let mut store = KnowledgeStore::default();
    for item in [
        object("topic", KnowledgeLifecycle::Hot, vec![]),
        object("derived", KnowledgeLifecycle::Hot, vec![key("topic", 1)]),
        successor("topic", KnowledgeLifecycle::Cold, vec![]),
    ] {
        store.register_object(item).unwrap();
    }
    let id = KnowledgeObjectId::from("topic");
    store.activate_generation(&id, Generation(2)).unwrap();
    assert_eq!(store.object(&id).unwrap().generation, Generation(2));
    assert_eq!(store.lifecycle(&id), Some(KnowledgeLifecycle::Cold));
    assert!(store.is_generation_invalidated(&id, Generation(1)));
    assert!(store.is_invalidated(&KnowledgeObjectId::from("derived")));
    store.transition(&id, KnowledgeLifecycle::Pod).unwrap();
    store.transition(&id, KnowledgeLifecycle::Archived).unwrap();
    store.activate_generation(&id, Generation(2)).unwrap();
    assert_eq!(store.lifecycle(&id), Some(KnowledgeLifecycle::Archived));
}
