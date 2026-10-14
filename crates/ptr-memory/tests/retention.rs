use ptr_memory::{
    compile_selected_at, compile_structured, compile_structured_at, decide,
    BasicKnowledgeObjectBuilder, ConflictState, DeterministicEntityExtractor,
    DeterministicTokenizer, EntityExtractor, Freshness, InMemoryKvStateRuntime, KnowledgeKey,
    KnowledgeLifecycle, KnowledgeObject, KnowledgeObjectBuilder, KnowledgeStore, KvStateRuntime,
    NamespaceContext, NamespaceResolver, RetentionAction, RetentionCandidate, RetentionError,
    RetentionModel, RetentionReason, RetentionScore, RetentionState, SemanticTokenizer,
    StaticNamespaceResolver, TokenBudget,
};
use ptr_types::{
    AdapterVersion, EntityId, EventRole, EvidenceId, Generation, KnowledgeObjectId, ModelVersion,
    NamespaceId, PolicyVersion, Probability, ProvenanceRef, RawEventId, RetrievalKey, SessionId,
    Timestamp,
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
        validity: Default::default(),
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

#[test]
fn an_expired_object_is_left_out_of_a_context_compiled_for_an_instant() {
    use ptr_memory::ContextEntry;
    use ptr_types::KnowledgeValidity;
    let timed = |id: &str, from: Option<u64>, until: Option<u64>| {
        let mut item = object(id, KnowledgeLifecycle::Hot, vec![]);
        item.validity = KnowledgeValidity {
            valid_from: from.map(Timestamp),
            valid_until: until.map(Timestamp),
        };
        item
    };
    let objects = [
        timed("always", None, None),
        timed("current", Some(10), Some(20)),
        timed("expired", Some(1), Some(10)),
        timed("future", Some(20), None),
    ];
    let ids = |context: &ptr_memory::CompiledContext| -> Vec<String> {
        context
            .entries
            .iter()
            .filter_map(|entry| match entry {
                ContextEntry::Structured(object) => Some(object.id.0.clone()),
                ContextEntry::Verbatim(_) => None,
            })
            .collect()
    };
    let at_15 = compile_structured_at(&objects, TokenBudget(100), Timestamp(15)).unwrap();
    assert_eq!(ids(&at_15), ["always", "current"]);
    assert_eq!(at_15.omitted_expired, 2);
    // Half-open: the end is outside, the start inside.
    let at_10 = compile_structured_at(&objects, TokenBudget(100), Timestamp(10)).unwrap();
    assert_eq!(ids(&at_10), ["always", "current"]);
    let at_20 = compile_structured_at(&objects, TokenBudget(100), Timestamp(20)).unwrap();
    assert_eq!(ids(&at_20), ["always", "future"]);
    // The audit path has no time rule and omits nothing.
    let audit = compile_structured(&objects, TokenBudget(100)).unwrap();
    assert_eq!(audit.entries.len(), 4);
    assert_eq!(audit.omitted_expired, 0);
    // A dependent of an omitted object is refused, not compiled without its basis.
    let mut dependent = object(
        "dependent",
        KnowledgeLifecycle::Hot,
        vec![key("expired", 1)],
    );
    dependent.validity = KnowledgeValidity::ALWAYS;
    let candidates: Vec<ptr_memory::ContextCandidate> = [objects[2].clone(), dependent]
        .into_iter()
        .map(|object| ptr_memory::ContextCandidate {
            object,
            action: ptr_memory::RetentionAction::KeepStructured,
            verbatim: None,
            pinned: false,
        })
        .collect();
    assert_eq!(
        compile_selected_at(&candidates, TokenBudget(100), Timestamp(15)),
        Err(ptr_memory::ContextCompileError::MissingDependency)
    );
}

#[test]
fn an_objects_validity_is_half_open_and_part_of_its_digest_only_when_bounded() {
    use ptr_types::KnowledgeValidity;
    let unbounded = object("topic", KnowledgeLifecycle::Hot, vec![]);
    assert!(unbounded.is_valid_at(Timestamp(0)));
    assert!(unbounded.is_valid_at(Timestamp(u64::MAX)));

    let mut bounded = unbounded.clone();
    bounded.validity = KnowledgeValidity {
        valid_from: Some(Timestamp(10)),
        valid_until: Some(Timestamp(20)),
    };
    assert!(!bounded.is_valid_at(Timestamp(9)));
    assert!(bounded.is_valid_at(Timestamp(10)), "the start is inside");
    assert!(bounded.is_valid_at(Timestamp(19)));
    assert!(!bounded.is_valid_at(Timestamp(20)), "the end is outside");

    assert_ne!(
        bounded.content_digest(),
        unbounded.content_digest(),
        "two objects that hold at different times are not the same content"
    );
    let mut shifted = bounded.clone();
    shifted.validity.valid_until = Some(Timestamp(21));
    assert_ne!(shifted.content_digest(), bounded.content_digest());
    let mut unchanged = unbounded.clone();
    unchanged.validity = KnowledgeValidity::ALWAYS;
    assert_eq!(
        unchanged.content_digest(),
        unbounded.content_digest(),
        "the same validity gives the same digest"
    );
}

#[test]
fn text_that_looks_like_another_field_cannot_collide_with_it() {
    // A retrieval key holding a newline and a validity line must not hash like the same
    // object with that validity: every field is length-prefixed, the validity is binary.
    let mut forged = object("x", KnowledgeLifecycle::Hot, vec![]);
    forged.retrieval_keys = vec![RetrievalKey::from("k\nvalidity=1:2")];
    let mut honest = object("x", KnowledgeLifecycle::Hot, vec![]);
    honest.retrieval_keys = vec![RetrievalKey::from("k")];
    honest.validity = ptr_types::KnowledgeValidity {
        valid_from: Some(Timestamp(1)),
        valid_until: Some(Timestamp(2)),
    };
    assert_ne!(forged.content_digest(), honest.content_digest());
    assert_ne!(forged.digest(), honest.digest());
    // Moving a boundary between two list elements changes the digest too.
    let mut split = object("x", KnowledgeLifecycle::Hot, vec![]);
    split.retrieval_keys = vec![RetrievalKey::from("ab"), RetrievalKey::from("c")];
    let mut joined = object("x", KnowledgeLifecycle::Hot, vec![]);
    joined.retrieval_keys = vec![RetrievalKey::from("a"), RetrievalKey::from("bc")];
    assert_ne!(split.content_digest(), joined.content_digest());
    // An open start and a start at zero are different validities.
    let mut open = object("x", KnowledgeLifecycle::Hot, vec![]);
    open.validity.valid_until = Some(Timestamp(5));
    let mut zero = open.clone();
    zero.validity.valid_from = Some(Timestamp(0));
    assert_ne!(open.content_digest(), zero.content_digest());
}

#[test]
fn an_object_whose_validity_ends_no_later_than_it_starts_is_refused_at_admission() {
    use ptr_memory::KnowledgeStoreError;
    use ptr_types::KnowledgeValidity;
    for (from, until) in [(20, 20), (20, 10)] {
        let mut never = object("never", KnowledgeLifecycle::Hot, vec![]);
        never.validity = KnowledgeValidity {
            valid_from: Some(Timestamp(from)),
            valid_until: Some(Timestamp(until)),
        };
        let expected = KnowledgeStoreError::InvalidValidity {
            valid_from: Some(Timestamp(from)),
            valid_until: Some(Timestamp(until)),
        };
        let mut store = KnowledgeStore::default();
        assert_eq!(store.register_object(never.clone()), Err(expected.clone()));
        assert_eq!(store.admit_object(never), Err(expected));
        assert!(store.object(&KnowledgeObjectId::from("never")).is_none());
    }
    // Open-ended and bounded intervals are fine.
    let mut fine = object("fine", KnowledgeLifecycle::Hot, vec![]);
    fine.validity = KnowledgeValidity {
        valid_from: Some(Timestamp(10)),
        valid_until: None,
    };
    assert_eq!(KnowledgeStore::default().register_object(fine), Ok(()));
}
