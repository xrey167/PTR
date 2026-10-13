//! Stepwise sessions of the reference KV-reuse backend: deterministic race, lag, restore
//! and saturation arms (R005 Exact half). Public API only. Not an R005 result.

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{ModelEvent, PromptSegment, SegmentSource, SegmentedModelRequest, Tokenizer};
use ptr_runtime::{
    CatalogLimits, IndexUpdate, KvCancelReason, KvCapacityResource, KvCatalogError, KvDependency,
    KvLivenessOracle, KvSession, KvSessionFinish, KvSessionPhase, KvSessionStep, PtrRuntime,
    ReferenceKvError, ReferenceKvFaults, ReferenceKvLimits, ReferenceKvOutcome,
    ReferenceKvReuseBackend, ReferenceKvScope,
};
use ptr_types::{
    CapsuleId, Generation, KnowledgeObjectId, PrincipalId, ProjectId, RequestId, Revision,
    SemanticContext,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("p"),
    }
}

fn backend_with(limits: ReferenceKvLimits) -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::new(limits, scope()).unwrap()
}

fn backend() -> ReferenceKvReuseBackend {
    backend_with(ReferenceKvLimits::default())
}

fn context() -> SemanticContext {
    SemanticContext {
        revision: Revision(1),
        entries: Vec::new(),
    }
}

/// `count` ASCII bytes derived from `seed`, never containing a repeated 16-byte chunk.
fn text(count: usize, seed: usize) -> String {
    (0..count)
        .map(|i| char::from(b'a' + ((i * 7 + seed * 13 + i / 16 * 5) % 26) as u8))
        .collect()
}

/// Like `text`, but every seed below 26 gives different chunks (`text` only depends on the
/// seed's parity).
fn varied(count: usize, seed: usize) -> String {
    (0..count)
        .map(|i| char::from(b'a' + ((i * 7 + seed * 3 + i / 16 * 5) % 26) as u8))
        .collect()
}

fn request_of(
    backend: &ReferenceKvReuseBackend,
    id: &str,
    segments: Vec<PromptSegment>,
) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from(id),
        context(),
        backend.tokenizer().identity().clone(),
        segments,
    )
    .unwrap()
}

fn raw(backend: &ReferenceKvReuseBackend, id: &str, body: &str) -> SegmentedModelRequest {
    let segment = PromptSegment::raw_from_text(backend.tokenizer(), body).unwrap();
    request_of(backend, id, vec![segment])
}

/// A bound semantic segment `doc:one` at `generation`, then a raw tail.
fn semantic(
    backend: &ReferenceKvReuseBackend,
    id: &str,
    generation: u64,
    bound: &str,
    tail: &str,
) -> SegmentedModelRequest {
    let head = PromptSegment::from_text(
        backend.tokenizer(),
        SegmentSource::Semantic {
            key: "doc:one".to_owned(),
            generation: Generation(generation),
        },
        bound,
    )
    .unwrap();
    let tail = PromptSegment::raw_from_text(backend.tokenizer(), tail).unwrap();
    request_of(backend, id, vec![head, tail])
}

fn dead_set(identity: &str, generation: u64) -> BTreeSet<(String, Generation)> {
    [(identity.to_owned(), Generation(generation))]
        .into_iter()
        .collect()
}

fn cold(request: &SegmentedModelRequest) -> ReferenceKvOutcome {
    let reference = backend();
    reference.infer_cold(&scope(), request).unwrap()
}

fn same_result(a: &ReferenceKvOutcome, b: &ReferenceKvOutcome) {
    assert_eq!(a.generated, b.generated);
    assert_eq!(a.logits_digest, b.logits_digest);
    assert_eq!(a.run.events, b.run.events);
    assert!(a
        .run
        .events
        .iter()
        .any(|event| matches!(event, ModelEvent::Token(text) if !text.is_empty())));
}

/// Liveness the test scripts, independent of the backend's own set.
#[derive(Default)]
struct Scripted {
    dead: Mutex<BTreeSet<(String, Generation)>>,
}

impl Scripted {
    fn kill(&self, identity: &str, generation: u64) {
        self.dead
            .lock()
            .unwrap()
            .insert((identity.to_owned(), Generation(generation)));
    }
}

impl KvLivenessOracle for Scripted {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        !self
            .dead
            .lock()
            .unwrap()
            .contains(&(dependency.identity.clone(), dependency.generation))
    }
}

fn prefill_all(session: &mut KvSession<'_>, oracle: &dyn KvLivenessOracle) {
    while session.phase() == KvSessionPhase::Prefilling {
        match session.prefill_chunk(oracle).unwrap() {
            KvSessionStep::Prefilled { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}

fn decode(session: &mut KvSession<'_>, oracle: &dyn KvLivenessOracle, steps: usize) {
    for _ in 0..steps {
        assert!(matches!(
            session.decode_step(oracle).unwrap(),
            KvSessionStep::Token(_)
        ));
    }
}

fn completed(finish: KvSessionFinish) -> ReferenceKvOutcome {
    match finish {
        KvSessionFinish::Completed(outcome) => *outcome,
        KvSessionFinish::Cancelled(cancelled) => panic!("cancelled: {cancelled:?}"),
    }
}

fn warm_semantic(backend: &ReferenceKvReuseBackend, id: &str, tail_seed: usize) {
    let outcome = backend
        .infer_scoped(
            &scope(),
            &semantic(backend, id, 1, &text(32, 7), &text(8, tail_seed)),
            None,
        )
        .unwrap();
    assert_eq!(
        outcome.inserted_chunks, 2,
        "boundaries 16 and 32 are cached"
    );
}

// ---- the one-shot path is a session ----------------------------------------------------

#[test]
fn a_stepwise_session_is_bitwise_equal_to_the_one_shot_run_and_to_cold() {
    let request = raw(&backend(), "a", &format!("{}{}", text(32, 1), text(10, 2)));
    let one_shot = backend();
    let expected = one_shot.infer_scoped(&scope(), &request, None).unwrap();

    let stepped = backend();
    let live = stepped.own_liveness();
    let mut session = stepped
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(session.phase(), KvSessionPhase::Prefilling);
    prefill_all(&mut session, &live);
    assert_eq!(session.phase(), KvSessionPhase::Decoding);
    decode(&mut session, &live, stepped.limits().max_new_tokens);
    assert_eq!(session.phase(), KvSessionPhase::Finished);
    let outcome = completed(session.finish(&live).unwrap());
    assert_eq!(session.phase(), KvSessionPhase::Closed);
    same_result(&expected, &outcome);
    same_result(&cold(&request), &outcome);
    assert_eq!(outcome.inserted_chunks, expected.inserted_chunks);
    let stats = stepped.session_stats().unwrap();
    assert_eq!(stats.sessions_begun, 1);
    assert_eq!(stats.sessions_completed, 1);
    assert!(stats.cancelled_requests.is_empty());
}

#[test]
fn two_sessions_interleaved_step_by_step_both_equal_cold() {
    let backend = backend();
    let live = backend.own_liveness();
    let first = raw(&backend, "a", &format!("{}{}", text(32, 1), text(5, 2)));
    let second = raw(&backend, "b", &format!("{}{}", text(32, 1), text(9, 3)));
    let mut a = backend
        .begin_session(&scope(), &first, None, &live)
        .unwrap();
    let mut b = backend
        .begin_session(&scope(), &second, None, &live)
        .unwrap();
    // Round robin, one step each, until both are done.
    loop {
        let mut progressed = false;
        for session in [&mut a, &mut b] {
            match session.phase() {
                KvSessionPhase::Prefilling => {
                    session.prefill_chunk(&live).unwrap();
                    progressed = true;
                }
                KvSessionPhase::Decoding => {
                    session.decode_step(&live).unwrap();
                    progressed = true;
                }
                _ => {}
            }
        }
        if !progressed {
            break;
        }
    }
    same_result(&cold(&first), &completed(a.finish(&live).unwrap()));
    same_result(&cold(&second), &completed(b.finish(&live).unwrap()));
}

// ---- inflight revocation race ------------------------------------------------------------

#[test]
fn a_revocation_between_decode_steps_emits_no_further_token_inserts_nothing_and_restarts_cold() {
    let backend = backend();
    warm_semantic(&backend, "warm", 8);
    let request = semantic(&backend, "race", 1, &text(32, 7), &text(8, 9));
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(
        session.hit_tokens(),
        32,
        "the fork is pinned for the whole session"
    );
    prefill_all(&mut session, &live);
    decode(&mut session, &live, 3);
    let entries_before = backend.catalog_stats().unwrap().entries;
    assert!(entries_before >= 1);

    // The revocation commits between two decode steps.
    backend.invalidate_inputs(&dead_set("doc:one", 1)).unwrap();

    let step = session.decode_step(&live).unwrap();
    let KvSessionStep::Cancelled(cancelled) = step else {
        panic!("expected a cancellation, got {step:?}");
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
    assert_eq!(cancelled.at_phase, KvSessionPhase::Decoding);
    assert_eq!(cancelled.tokens_emitted, 3);
    assert_eq!(
        session.generated().len(),
        3,
        "no token after the revocation point"
    );
    assert_eq!(session.phase(), KvSessionPhase::Cancelled);
    // Nothing was inserted after the revocation, the pinned tombstoned entry was reclaimed.
    let stats = backend.catalog_stats().unwrap();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.tombstoned_pinned_entries, 0);
    // No further step is possible, and the restart is the only way on.
    assert!(matches!(
        session.decode_step(&live),
        Err(ReferenceKvError::SessionPhase { .. })
    ));
    assert!(matches!(
        session.finish(&live),
        Err(ReferenceKvError::SessionPhase { .. })
    ));

    let restarted = session.restart().unwrap();
    same_result(&cold(&request), &restarted);
    assert_eq!(restarted.hit_tokens, 0);
    assert_eq!(restarted.prefilled_tokens, request.total_tokens());
    assert_eq!(restarted.inserted_chunks, 0);
    assert!(restarted.restarted_after_cancel);
    assert_eq!(
        backend.catalog_stats().unwrap().entries,
        0,
        "a cold restart inserts nothing"
    );
    assert_eq!(session.phase(), KvSessionPhase::Closed);

    let stats = backend.session_stats().unwrap();
    let expected: BTreeSet<RequestId> = [RequestId::from("race")].into_iter().collect();
    assert_eq!(stats.cancelled_requests, expected);
    assert_eq!(stats.restarted_requests, expected);
    assert_eq!(stats.sessions_cancelled, 1);
    assert_eq!(stats.sessions_restarted, 1);
    assert_eq!(stats.pinned_entry_evictions, 0);
}

#[test]
fn a_revocation_between_prefill_chunks_cancels_before_the_next_insert() {
    let backend = backend();
    let request = semantic(&backend, "race", 1, &text(32, 7), &text(20, 9));
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(
        session.prefill_chunk(&live).unwrap(),
        KvSessionStep::Prefilled { position: 16 }
    );
    assert_eq!(
        backend.catalog_stats().unwrap().entries,
        1,
        "boundary 16 was cached"
    );
    backend.invalidate_inputs(&dead_set("doc:one", 1)).unwrap();
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);
    let step = session.prefill_chunk(&live).unwrap();
    assert!(matches!(
        step,
        KvSessionStep::Cancelled(ref cancelled)
            if cancelled.reason == KvCancelReason::InputNotLive && cancelled.tokens_emitted == 0
    ));
    assert_eq!(
        backend.catalog_stats().unwrap().entries,
        0,
        "boundary 32 was never inserted"
    );
    assert_eq!(session.position(), 16);
    let restarted = session.restart().unwrap();
    same_result(&cold(&request), &restarted);
}

#[test]
fn a_revocation_the_session_learns_only_at_finish_still_cancels_the_outcome() {
    let backend = backend();
    let request = semantic(&backend, "race", 1, &text(32, 7), &text(8, 9));
    let scripted = Scripted::default();
    let mut session = backend
        .begin_session(&scope(), &request, None, &scripted)
        .unwrap();
    prefill_all(&mut session, &scripted);
    decode(&mut session, &scripted, backend.limits().max_new_tokens);
    scripted.kill("doc:one", 1);
    let KvSessionFinish::Cancelled(cancelled) = session.finish(&scripted).unwrap() else {
        panic!("finish must not hand out an outcome derived from a revoked input");
    };
    assert_eq!(cancelled.at_phase, KvSessionPhase::Finished);
    assert_eq!(cancelled.tokens_emitted, backend.limits().max_new_tokens);
    same_result(&cold(&request), &session.restart().unwrap());
}

#[test]
fn an_external_oracle_cancels_even_when_the_backend_was_never_told() {
    // The oracle (the ledger, in production) is the authority; the backend's own set and
    // its catalog index never see the revocation.
    let backend = backend();
    warm_semantic(&backend, "warm", 8);
    let request = semantic(&backend, "race", 1, &text(32, 7), &text(8, 9));
    let scripted = Scripted::default();
    let mut session = backend
        .begin_session(&scope(), &request, None, &scripted)
        .unwrap();
    assert_eq!(session.hit_tokens(), 32);
    prefill_all(&mut session, &scripted);
    decode(&mut session, &scripted, 2);
    scripted.kill("doc:one", 1);
    let KvSessionStep::Cancelled(cancelled) = session.decode_step(&scripted).unwrap() else {
        panic!("expected a cancellation");
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
    assert_eq!(session.generated().len(), 2);
    same_result(&cold(&request), &session.restart().unwrap());
}

#[test]
fn a_ledger_committed_revocation_cancels_through_the_runtime_oracle() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    let backend = backend();
    let memory = PromptSegment::from_text(
        backend.tokenizer(),
        SegmentSource::Memory {
            id: KnowledgeObjectId::from("capsule:a"),
            generation: Generation(1),
        },
        &text(32, 5),
    )
    .unwrap();
    let tail = PromptSegment::raw_from_text(backend.tokenizer(), &text(8, 6)).unwrap();
    let request = request_of(&backend, "ledger", vec![memory, tail]);

    let mut session = backend
        .begin_session(&scope(), &request, None, &runtime)
        .unwrap();
    prefill_all(&mut session, &runtime);
    decode(&mut session, &runtime, 2);
    // The revocation is committed to the ledger between two decode steps.
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    let KvSessionStep::Cancelled(cancelled) = session.decode_step(&runtime).unwrap() else {
        panic!("the committed revocation must cancel the session");
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
    assert_eq!(session.generated().len(), 2);
    same_result(&cold(&request), &session.restart().unwrap());
}

// ---- supersession race --------------------------------------------------------------------

#[test]
fn a_supersession_between_decode_steps_cancels_and_the_restart_equals_cold() {
    let backend = backend();
    warm_semantic(&backend, "warm", 8);
    let request = semantic(&backend, "super", 1, &text(32, 7), &text(8, 9));
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(session.hit_tokens(), 32);
    prefill_all(&mut session, &live);
    decode(&mut session, &live, 4);

    let floors: BTreeMap<String, Generation> = [("doc:one".to_owned(), Generation(2))]
        .into_iter()
        .collect();
    assert!(backend.supersede_inputs(&floors).unwrap() >= 1);

    let KvSessionStep::Cancelled(cancelled) = session.decode_step(&live).unwrap() else {
        panic!("a superseded generation must cancel");
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
    assert_eq!(
        session.generated().len(),
        4,
        "no token after the supersession point"
    );
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);

    let restarted = session.restart().unwrap();
    same_result(&cold(&request), &restarted);
    let stats = backend.session_stats().unwrap();
    assert!(stats.cancelled_requests.contains(&RequestId::from("super")));
    assert!(stats.restarted_requests.contains(&RequestId::from("super")));

    // The same content at the live generation is another lineage: computed fresh, equal to
    // cold, and cacheable again.
    let newer = semantic(&backend, "newer", 2, &text(32, 7), &text(8, 9));
    let fresh = backend.infer_scoped(&scope(), &newer, None).unwrap();
    assert_eq!(fresh.hit_tokens, 0);
    assert_eq!(fresh.inserted_chunks, 2);
    same_result(&cold(&newer), &fresh);
    // Generation 1 stays refused for good (the floor is never lowered).
    let again = backend.infer_scoped(&scope(), &request, None).unwrap();
    assert_eq!(again.hit_tokens, 0);
    assert_eq!(again.inserted_chunks, 0);
}

#[test]
fn a_request_that_starts_with_a_dead_input_runs_uncached_and_is_not_cancelled() {
    let backend = backend();
    backend.invalidate_inputs(&dead_set("doc:one", 1)).unwrap();
    let request = semantic(&backend, "dead", 1, &text(32, 7), &text(8, 9));
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    prefill_all(&mut session, &live);
    decode(&mut session, &live, backend.limits().max_new_tokens);
    let outcome = completed(session.finish(&live).unwrap());
    same_result(&cold(&request), &outcome);
    assert_eq!(outcome.inserted_chunks, 0);
    assert!(outcome
        .insert_refusal_reasons
        .iter()
        .all(|reason| *reason == KvCatalogError::DependencyNotLive));
    assert!(outcome.insert_refusals >= 1);
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);
}

// ---- lag arm -------------------------------------------------------------------------------

#[test]
fn with_the_index_update_suppressed_only_the_use_time_check_prevents_a_stale_hit() {
    let backend = backend();
    warm_semantic(&backend, "warm", 8);
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .unwrap();
    let before = backend.catalog_stats().unwrap();
    assert!(before.entries >= 1);

    assert_eq!(
        backend.invalidate_inputs(&dead_set("doc:one", 1)).unwrap(),
        0,
        "the suppressed update tombstones and retires nothing"
    );
    let stale = backend.catalog_stats().unwrap();
    assert_eq!(
        stale.entries, before.entries,
        "the index is stale: the entries are still there"
    );
    assert_eq!(stale.tombstoned_pinned_entries, 0);
    assert_eq!(backend.session_stats().unwrap().suppressed_index_updates, 1);

    // The one-shot path and a session both refuse the stale entry at use.
    let request = semantic(&backend, "after", 1, &text(32, 7), &text(8, 9));
    let one_shot = backend.infer_scoped(&scope(), &request, None).unwrap();
    assert_eq!(one_shot.hit_tokens, 0, "no stale hit");
    same_result(&cold(&request), &one_shot);
    let after = backend.catalog_stats().unwrap();
    assert_eq!(after.hits, before.hits, "the catalog served no hit");
    assert!(
        after.use_time_rejections >= 1,
        "the use-time check refused it"
    );

    let live = backend.own_liveness();
    let session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(session.hit_tokens(), 0);
    assert_eq!(backend.session_stats().unwrap().stale_hits, 0);
}

#[test]
fn a_session_holding_a_pin_is_cancelled_by_the_use_time_check_when_the_index_is_stale() {
    let backend = backend();
    warm_semantic(&backend, "warm", 8);
    let request = semantic(&backend, "lag", 1, &text(32, 7), &text(8, 9));
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(session.hit_tokens(), 32);
    prefill_all(&mut session, &live);
    decode(&mut session, &live, 2);

    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .unwrap();
    backend.invalidate_inputs(&dead_set("doc:one", 1)).unwrap();
    // The catalog never learned of the revocation: the pinned entry is not tombstoned.
    assert_eq!(
        backend.catalog_stats().unwrap().tombstoned_pinned_entries,
        0
    );

    let KvSessionStep::Cancelled(cancelled) = session.decode_step(&live).unwrap() else {
        panic!("the use-time check alone must cancel");
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
    assert_eq!(session.generated().len(), 2);
    same_result(&cold(&request), &session.restart().unwrap());
}

// ---- restore arm ---------------------------------------------------------------------------

#[test]
fn a_restored_entry_whose_input_was_revoked_at_rest_misses_at_use() {
    let source = backend();
    warm_semantic(&source, "warm", 8);
    let snapshot = source.export_catalog().unwrap();

    let restored = backend();
    restored.invalidate_inputs(&dead_set("doc:one", 1)).unwrap();
    assert!(restored.restore_catalog(&snapshot).unwrap().entries >= 1);
    let request = semantic(&restored, "after", 1, &text(32, 7), &text(8, 9));
    let outcome = restored.infer_scoped(&scope(), &request, None).unwrap();
    assert_eq!(
        outcome.hit_tokens, 0,
        "the restored pre-revocation entry is not served"
    );
    same_result(&cold(&request), &outcome);
    let stats = restored.catalog_stats().unwrap();
    assert_eq!(stats.hits, 0);
    assert!(stats.use_time_rejections >= 1);
    assert!(stats.restored_entries >= 1);
}

#[test]
fn a_restored_live_entry_without_its_pages_is_dropped_and_the_run_equals_cold() {
    let source = backend();
    warm_semantic(&source, "warm", 8);
    let snapshot = source.export_catalog().unwrap();

    let restored = backend();
    restored.restore_catalog(&snapshot).unwrap();
    let request = semantic(&restored, "after", 1, &text(32, 7), &text(8, 9));
    let outcome = restored.infer_scoped(&scope(), &request, None).unwrap();
    assert_eq!(outcome.hit_tokens, 0);
    same_result(&cold(&request), &outcome);
    assert!(restored.catalog_stats().unwrap().backend_missing_rejections >= 1);
    // A snapshot is not restorable twice, and garbage is a typed error.
    assert!(matches!(
        restored.restore_catalog(&snapshot),
        Err(ReferenceKvError::Snapshot(_))
    ));
    assert!(matches!(
        backend().restore_catalog(b"not a snapshot"),
        Err(ReferenceKvError::Snapshot(_))
    ));
}

// ---- saturation and pinned forks --------------------------------------------------------

fn two_node_backend() -> ReferenceKvReuseBackend {
    backend_with(ReferenceKvLimits {
        catalog: CatalogLimits {
            max_trie_nodes: 2,
            ..CatalogLimits::default()
        },
        ..ReferenceKvLimits::default()
    })
}

#[test]
fn saturation_with_pinned_forks_is_capacity_exhausted_and_the_uncached_fallback_equals_cold() {
    let backend = two_node_backend();
    let short = text(32, 12);
    backend
        .infer_scoped(&scope(), &raw(&backend, "warm", &short), None)
        .unwrap();
    assert_eq!(backend.catalog_stats().unwrap().nodes, 2);

    // An open session pins the whole catalog.
    let live = backend.own_liveness();
    let pinned_request = raw(&backend, "pinned", &format!("{short}{}", text(16, 13)));
    let mut pinned = backend
        .begin_session(&scope(), &pinned_request, None, &live)
        .unwrap();
    assert_eq!(pinned.hit_tokens(), 32);

    // A second request needs a new node, and the only candidate victim is pinned.
    let other_request = raw(&backend, "other", &text(40, 21));
    let mut other = backend
        .begin_session(&scope(), &other_request, None, &live)
        .unwrap();
    prefill_all(&mut other, &live);
    decode(&mut other, &live, backend.limits().max_new_tokens);
    let outcome = completed(other.finish(&live).unwrap());
    assert_eq!(outcome.inserted_chunks, 0, "no insert");
    assert!(outcome.insert_refusals >= 1);
    assert!(matches!(
        outcome.insert_refusal_reasons.first(),
        Some(KvCatalogError::CapacityExhausted(
            KvCapacityResource::TrieNodes
        ))
    ));
    same_result(&cold(&other_request), &outcome);
    assert_eq!(outcome.prefilled_tokens, 40, "an uncached full prefill");

    // The pinned fork was untouched and still finishes bitwise equal to cold.
    prefill_all(&mut pinned, &live);
    decode(&mut pinned, &live, backend.limits().max_new_tokens);
    same_result(
        &cold(&pinned_request),
        &completed(pinned.finish(&live).unwrap()),
    );

    let stats = backend.session_stats().unwrap();
    assert!(stats.capacity_exhausted_inserts >= 2);
    assert_eq!(stats.pinned_entry_evictions, 0);
    assert_eq!(backend.catalog_stats().unwrap().nodes, 2);
}

#[test]
fn eviction_never_takes_an_entry_an_active_fork_pins() {
    let backend = backend_with(ReferenceKvLimits {
        catalog: CatalogLimits {
            max_trie_nodes: 4,
            ..CatalogLimits::default()
        },
        ..ReferenceKvLimits::default()
    });
    let kept = varied(32, 1);
    let evictable = varied(32, 2);
    backend
        .infer_scoped(&scope(), &raw(&backend, "kept", &kept), None)
        .unwrap();
    backend
        .infer_scoped(&scope(), &raw(&backend, "evictable", &evictable), None)
        .unwrap();
    assert_eq!(backend.catalog_stats().unwrap().nodes, 4);

    let live = backend.own_liveness();
    let pinned_request = raw(&backend, "pinned", &format!("{kept}{}", text(8, 13)));
    let mut pinned = backend
        .begin_session(&scope(), &pinned_request, None, &live)
        .unwrap();
    assert_eq!(pinned.hit_tokens(), 32);

    // Unique prompts force evictions while the fork is pinned.
    for seed in 3..7 {
        backend
            .infer_scoped(
                &scope(),
                &raw(&backend, &format!("unique{seed}"), &varied(32, seed)),
                None,
            )
            .unwrap();
    }
    let stats = backend.session_stats().unwrap();
    assert!(
        stats.evictions_with_active_fork_pinned >= 1,
        "evictions happened while a fork was pinned"
    );
    assert_eq!(
        stats.pinned_entry_evictions, 0,
        "none of them took the pinned entry"
    );
    assert!(backend.catalog_stats().unwrap().nodes <= 4);

    prefill_all(&mut pinned, &live);
    decode(&mut pinned, &live, backend.limits().max_new_tokens);
    let outcome = completed(pinned.finish(&live).unwrap());
    assert_eq!(outcome.hit_tokens, 32);
    same_result(&cold(&pinned_request), &outcome);

    // After the release the entry is evictable again: flush takes everything.
    assert!(backend.flush().unwrap() >= 1);
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);
    assert_eq!(backend.session_stats().unwrap().pinned_entry_evictions, 0);
}

#[test]
fn the_number_of_open_sessions_is_bounded_with_a_typed_error() {
    let backend = backend_with(ReferenceKvLimits {
        max_sessions: 2,
        ..ReferenceKvLimits::default()
    });
    let live = backend.own_liveness();
    let request = raw(&backend, "a", &text(20, 1));
    let first = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    let second = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert!(matches!(
        backend.begin_session(&scope(), &request, None, &live),
        Err(ReferenceKvError::SessionLimit { maximum: 2 })
    ));
    assert_eq!(backend.session_stats().unwrap().session_limit_refusals, 1);
    // The one-shot path is not refused by the session bound.
    backend
        .infer_scoped(&scope(), &raw(&backend, "b", &text(20, 2)), None)
        .unwrap();
    // Dropping a session frees its slot.
    drop(first);
    let third = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    drop((second, third));

    let zero = ReferenceKvLimits {
        max_sessions: 0,
        ..ReferenceKvLimits::default()
    };
    assert!(matches!(
        ReferenceKvReuseBackend::new(zero, scope()),
        Err(ReferenceKvError::InvalidLimits)
    ));
}

#[test]
fn an_abandoned_session_releases_its_pin_so_the_entry_can_be_evicted() {
    let backend = backend();
    let prompt = text(32, 4);
    backend
        .infer_scoped(&scope(), &raw(&backend, "warm", &prompt), None)
        .unwrap();
    let live = backend.own_liveness();
    let request = raw(&backend, "abandoned", &format!("{prompt}{}", text(8, 5)));
    let session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert!(
        backend.flush().unwrap() < 2,
        "the pinned entry survives a flush"
    );
    drop(session);
    assert!(backend.flush().unwrap() >= 1);
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);
    assert_eq!(backend.session_stats().unwrap().pinned_entry_evictions, 0);
}

// ---- negatives ----------------------------------------------------------------------------

#[test]
fn steps_out_of_phase_are_typed_errors_and_change_nothing() {
    let backend = backend();
    let live = backend.own_liveness();
    let request = raw(&backend, "a", &text(20, 1));
    let mut session = backend
        .begin_session(&scope(), &request, None, &live)
        .unwrap();
    assert!(matches!(
        session.decode_step(&live),
        Err(ReferenceKvError::SessionPhase {
            operation: "decode_step",
            phase: KvSessionPhase::Prefilling
        })
    ));
    assert!(matches!(
        session.finish(&live),
        Err(ReferenceKvError::SessionPhase { .. })
    ));
    assert!(matches!(
        session.restart(),
        Err(ReferenceKvError::SessionPhase { .. })
    ));
    assert_eq!(session.phase(), KvSessionPhase::Prefilling);
    prefill_all(&mut session, &live);
    assert!(matches!(
        session.prefill_chunk(&live),
        Err(ReferenceKvError::SessionPhase { .. })
    ));
    decode(&mut session, &live, backend.limits().max_new_tokens);
    assert!(matches!(
        session.decode_step(&live),
        Err(ReferenceKvError::SessionPhase { .. })
    ));
    let outcome = completed(session.finish(&live).unwrap());
    assert!(matches!(
        session.finish(&live),
        Err(ReferenceKvError::SessionPhase { .. })
    ));
    same_result(&cold(&request), &outcome);
}

#[test]
fn begin_refuses_a_bad_scope_an_oversized_prompt_and_a_foreign_tokenizer() {
    let backend = backend();
    let live = backend.own_liveness();
    let blank = ReferenceKvScope {
        principal: PrincipalId::from(" "),
        project: ProjectId::from("p"),
    };
    let request = raw(&backend, "a", &text(20, 1));
    assert!(matches!(
        backend.begin_session(&blank, &request, None, &live),
        Err(ReferenceKvError::InvalidScope)
    ));
    let long = raw(&backend, "long", &text(300, 1));
    assert!(matches!(
        backend.begin_session(&scope(), &long, None, &live),
        Err(ReferenceKvError::PromptTooLong { .. })
    ));
    assert_eq!(backend.session_stats().unwrap().sessions_begun, 0);
}
