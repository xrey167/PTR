use ptr_memory::{
    ContextCandidate, InMemoryKvStateRuntime, KvStateRuntime, RetentionAction, RetentionState,
    RuleRetentionModel, TokenBudget,
};
use ptr_runtime::memory::ContextMemoryController;
use ptr_types::{
    EventRole, NamespaceId, PolicyVersion, Probability, RawEventId, SessionId, Timestamp,
};

#[test]
fn memory_controller_ingests_and_compiles_through_runtime_owned_path() {
    let mut controller = ContextMemoryController::new(
        RuleRetentionModel,
        Probability::new(0.5).unwrap(),
        PolicyVersion::from("policy-v1"),
    );
    let object = controller
        .ingest(
            ptr_memory::RawEvent::new(
                RawEventId::from("event-1"),
                SessionId::from("session-1"),
                EventRole::Tool,
                b"delivery 14 days".to_vec(),
                Timestamp(1),
            ),
            NamespaceId::from("ptr.test"),
        )
        .unwrap();
    let context = controller
        .compile(std::slice::from_ref(&object.id), TokenBudget(10))
        .unwrap();
    assert_eq!(context.entries.len(), 1);
    assert_eq!(context.token_cost, 3);
    assert_eq!(
        controller
            .retain(&RetentionState {
                goal: "delivery".into(),
                candidates: vec![ptr_memory::RetentionCandidate {
                    object_id: object.id,
                    source_events: vec![RawEventId::from("event-1")],
                    token_cost: 3,
                    pinned: false,
                }],
            })
            .len(),
        1
    );
}

#[test]
fn generation_update_invalidates_old_context_and_kv_state() {
    let mut controller = ContextMemoryController::new(
        RuleRetentionModel,
        Probability::new(0.5).unwrap(),
        PolicyVersion::from("policy-v1"),
    );
    let first = controller
        .ingest(
            ptr_memory::RawEvent::new(
                RawEventId::from("event-generation-1"),
                SessionId::from("session-1"),
                EventRole::Tool,
                b"delivery 14 days".to_vec(),
                Timestamp(1),
            ),
            NamespaceId::from("ptr.test"),
        )
        .unwrap();
    let first_context = controller
        .compile(std::slice::from_ref(&first.id), TokenBudget(10))
        .unwrap();

    let mut second = first.clone();
    second.generation = ptr_types::Generation(2);
    second.supersedes = Some(ptr_memory::KnowledgeKey {
        logical_id: first.id.clone(),
        generation: ptr_types::Generation(1),
    });
    second.relevance = Probability::new(1.0).unwrap();
    second.token_cost = 4;
    controller.store.admit_object(second).unwrap();
    controller
        .store
        .activate_generation(&first.id, ptr_types::Generation(2))
        .unwrap();

    assert!(controller
        .store
        .is_generation_invalidated(&first.id, ptr_types::Generation(1)));
    let current = controller
        .compile(std::slice::from_ref(&first.id), TokenBudget(10))
        .unwrap();
    assert_ne!(first_context.digest, current.digest);
    assert_eq!(current.token_cost, 4);

    let mut kv = InMemoryKvStateRuntime::new(
        SessionId::from("session-1"),
        ptr_types::ModelVersion::from("model-v1"),
        ptr_types::AdapterVersion::from("adapter-v1"),
    );
    let old_state = kv.recompute(first_context).unwrap();
    let dependency = kv.metadata(&old_state).unwrap().dependency_digests[0];
    let invalidated = kv.invalidate_dependencies(&[dependency]).unwrap();
    assert_eq!(invalidated.len(), 1);
    assert!(kv.continue_state(&old_state, current).is_err());
}

#[test]
fn context_compiler_preserves_verbatim_and_skips_dropped_results() {
    let mut controller = ContextMemoryController::new(
        RuleRetentionModel,
        Probability::new(0.5).unwrap(),
        PolicyVersion::from("policy-v1"),
    );
    let object = controller
        .ingest(
            ptr_memory::RawEvent::new(
                RawEventId::from("event-verbatim"),
                SessionId::from("session-1"),
                EventRole::Tool,
                b"exact tool result".to_vec(),
                Timestamp(1),
            ),
            NamespaceId::from("ptr.test"),
        )
        .unwrap();
    let context = controller
        .compile_candidates(
            &[
                ContextCandidate {
                    object: object.clone(),
                    action: RetentionAction::KeepVerbatim,
                    verbatim: Some(b"exact tool result".to_vec()),
                    pinned: true,
                },
                ContextCandidate {
                    object,
                    action: RetentionAction::DropResult,
                    verbatim: None,
                    pinned: false,
                },
            ],
            TokenBudget(10),
        )
        .unwrap();
    assert_eq!(context.entries.len(), 1);
    assert_eq!(
        context.entries[0],
        ptr_memory::ContextEntry::Verbatim(b"exact tool result".to_vec())
    );
}

#[test]
fn raw_event_flows_through_retention_context_and_kv_recompute() {
    let mut controller = ContextMemoryController::new(
        RuleRetentionModel,
        Probability::new(0.5).unwrap(),
        PolicyVersion::from("policy-v1"),
    );
    let object = controller
        .ingest(
            ptr_memory::RawEvent::new(
                RawEventId::from("event-e2e"),
                SessionId::from("session-e2e"),
                EventRole::Tool,
                b"retained answer".to_vec(),
                Timestamp(1),
            ),
            NamespaceId::from("ptr.test"),
        )
        .unwrap();
    let candidate = ptr_memory::RetentionCandidate {
        object_id: object.id.clone(),
        source_events: vec![RawEventId::from("event-e2e")],
        token_cost: object.token_cost,
        pinned: false,
    };
    let decisions = controller.retain(&RetentionState {
        goal: "answer".into(),
        candidates: vec![candidate.clone()],
    });
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].action, RetentionAction::KeepVerbatim);

    let context = controller
        .compile_candidates(
            &[ContextCandidate {
                object,
                action: decisions[0].action,
                verbatim: Some(b"retained answer".to_vec()),
                pinned: candidate.pinned,
            }],
            TokenBudget(10),
        )
        .unwrap();
    let mut kv = InMemoryKvStateRuntime::new(
        SessionId::from("session-e2e"),
        ptr_types::ModelVersion::from("model-v1"),
        ptr_types::AdapterVersion::from("adapter-v1"),
    );
    let state = kv.recompute(context.clone()).unwrap();
    let metadata = kv.metadata(&state).unwrap();
    assert_eq!(metadata.dependency_digests, vec![context.digest]);
    assert_eq!(metadata.validity, ptr_memory::KvValidity::Valid);
}
