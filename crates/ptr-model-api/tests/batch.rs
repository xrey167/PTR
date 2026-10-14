//! The adaptive batcher through its public API: batching, key separation, the queue bound,
//! deadlines, cancellation, revocation at dispatch and failure containment.
mod common;

use common::{eventually, request, Latch, RecordingBackend};
use ptr_model_api::{
    AdaptiveBatcher, BatchAdmission, BatchClock, BatchError, BatchGate, BatchKey, BatchPolicy,
    CancelFlag, Continuation, InferenceBackend, ModelError, ModelEvent, ModelObservation,
    ModelRequest, ModelResumeRequest, SystemClock,
};
use ptr_types::{Generation, Revision, SemanticContext, Timestamp, TypeId};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const PATIENT: Duration = Duration::from_secs(10);

struct OpenGate;

impl BatchGate for OpenGate {
    fn admits(&self, _key: &BatchKey) -> bool {
        true
    }
}

/// A batcher whose gate admits every key, for tests that are not about the gate.
fn gated<B: ptr_model_api::BatchInferenceBackend + 'static>(
    backend: Arc<B>,
    policy: BatchPolicy,
) -> Result<AdaptiveBatcher<B>, BatchError> {
    AdaptiveBatcher::with_parts(
        backend,
        policy,
        Arc::new(SystemClock),
        Some(Arc::new(OpenGate) as Arc<dyn BatchGate>),
    )
}

fn policy(max_batch: usize, max_wait_millis: u64, queue_capacity: usize) -> BatchPolicy {
    BatchPolicy {
        max_batch,
        max_wait_millis,
        queue_capacity,
        deadline_margin_millis: 5,
    }
}

fn key(adapter: &str) -> BatchKey {
    BatchKey {
        project: Some("p".into()),
        adapter: Some(adapter.into()),
        principal: Some("alice".into()),
        generation: Generation(1),
    }
}

fn admission(adapter: &str) -> BatchAdmission {
    BatchAdmission {
        key: key(adapter),
        deadline: None,
        cancel: None,
    }
}

fn tokens(events: &[ModelEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::Token(token) => Some(token.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn concurrent_requests_share_one_model_call_when_the_batch_fills() {
    let backend = Arc::new(RecordingBackend::default());
    let batcher = gated(Arc::clone(&backend), policy(4, 5_000, 16)).expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let handles: Vec<_> = (0..4)
            .map(|n| {
                scope.spawn(move || batcher.infer_with(&request(&format!("r{n}")), admission("a")))
            })
            .collect();
        for (n, handle) in handles.into_iter().enumerate() {
            let events = handle.join().expect("thread").expect("served");
            assert_eq!(tokens(&events), vec![format!("r{n}")]);
        }
    });
    let stats = batcher.stats();
    assert_eq!(stats.batches, 1);
    assert_eq!(stats.largest_batch, 4);
    assert_eq!(stats.dispatched, 4);
    assert_eq!(backend.recorded().len(), 1);
}

#[test]
fn a_lone_request_is_served_once_max_wait_has_passed() {
    let backend = Arc::new(RecordingBackend::default());
    let batcher = gated(Arc::clone(&backend), policy(8, 20, 16)).expect("built");
    let events = batcher.infer(&request("alone")).expect("served");
    assert_eq!(tokens(&events), vec!["alone".to_owned()]);
    assert_eq!(batcher.stats().batches, 1);
}

#[test]
fn requests_under_different_keys_never_share_a_model_call() {
    let backend = Arc::new(RecordingBackend::default());
    let batcher = gated(Arc::clone(&backend), policy(8, 100, 16)).expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let handles: Vec<_> = ["a0", "b0", "a1", "b1"]
            .into_iter()
            .map(|id| {
                scope.spawn(move || {
                    batcher
                        .infer_with(&request(id), admission(&id[..1]))
                        .expect("served")
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread");
        }
    });
    for batch in backend.recorded() {
        let first = &batch[0][..1];
        assert!(
            batch.iter().all(|id| &id[..1] == first),
            "a batch mixed adapters: {batch:?}"
        );
    }
    assert_eq!(backend.calls.load(Ordering::SeqCst), 4);
}

#[test]
fn a_full_queue_answers_overloaded_at_once_and_nothing_is_lost() {
    let latch = Arc::new(Latch::default());
    let backend = Arc::new(RecordingBackend::blocking(Arc::clone(&latch)));
    let batcher = gated(Arc::clone(&backend), policy(1, 0, 2)).expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let first = scope.spawn(move || batcher.infer_with(&request("held"), admission("a")));
        assert!(
            eventually(PATIENT, || batcher.stats().dispatched == 1),
            "the first request reached the model"
        );
        let queued: Vec<_> = (0..2)
            .map(|n| {
                scope.spawn(move || batcher.infer_with(&request(&format!("q{n}")), admission("a")))
            })
            .collect();
        assert!(
            eventually(PATIENT, || batcher.queued() == 2),
            "the queue filled"
        );
        let refused = batcher.infer_with(&request("extra"), admission("a"));
        assert_eq!(refused, Err(ModelError::overloaded()));
        assert!(batcher.queued() <= 2, "the bound was never exceeded");
        assert_eq!(batcher.stats().overloaded, 1);
        latch.open();
        first.join().expect("thread").expect("held request served");
        for handle in queued {
            handle
                .join()
                .expect("thread")
                .expect("queued request served");
        }
    });
    assert_eq!(batcher.stats().dispatched, 3);
}

#[test]
fn a_deadline_that_has_passed_is_refused_before_the_model() {
    let backend = Arc::new(RecordingBackend::default());
    let batcher = gated(Arc::clone(&backend), policy(2, 10, 4)).expect("built");
    let past = BatchAdmission {
        key: key("a"),
        deadline: Some(Timestamp(1)),
        cancel: None,
    };
    assert_eq!(
        batcher.infer_with(&request("late"), past),
        Err(ModelError::deadline_exceeded())
    );
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    assert_eq!(batcher.stats().deadline_drops, 1);
}

#[test]
fn a_deadline_that_passes_in_the_queue_drops_the_request_unserved() {
    let latch = Arc::new(Latch::default());
    let backend = Arc::new(RecordingBackend::blocking(Arc::clone(&latch)));
    let batcher = gated(Arc::clone(&backend), policy(1, 0, 4)).expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let held = scope.spawn(move || batcher.infer_with(&request("held"), admission("a")));
        assert!(eventually(PATIENT, || batcher.stats().dispatched == 1));
        let soon = Timestamp(SystemClock.now().0 + 50);
        let waiting = scope.spawn(move || {
            batcher.infer_with(
                &request("doomed"),
                BatchAdmission {
                    key: key("a"),
                    deadline: Some(soon),
                    cancel: None,
                },
            )
        });
        assert!(eventually(PATIENT, || batcher.queued() == 1));
        std::thread::sleep(Duration::from_millis(120));
        latch.open();
        held.join().expect("thread").expect("held served");
        assert_eq!(
            waiting.join().expect("thread"),
            Err(ModelError::deadline_exceeded())
        );
    });
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1, "doomed never ran");
    assert_eq!(batcher.stats().deadline_drops, 1);
}

#[test]
fn a_cancelled_request_is_dropped_before_the_model() {
    let latch = Arc::new(Latch::default());
    let backend = Arc::new(RecordingBackend::blocking(Arc::clone(&latch)));
    let batcher = gated(Arc::clone(&backend), policy(1, 0, 4)).expect("built");
    let flag = CancelFlag::default();
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let held = scope.spawn(move || batcher.infer_with(&request("held"), admission("a")));
        assert!(eventually(PATIENT, || batcher.stats().dispatched == 1));
        let cancel = flag.clone();
        let waiting = scope.spawn(move || {
            batcher.infer_with(
                &request("withdrawn"),
                BatchAdmission {
                    key: key("a"),
                    deadline: None,
                    cancel: Some(cancel),
                },
            )
        });
        assert!(eventually(PATIENT, || batcher.queued() == 1));
        flag.cancel();
        latch.open();
        held.join().expect("thread").expect("held served");
        assert_eq!(
            waiting.join().expect("thread"),
            Err(ModelError::cancelled())
        );
    });
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert_eq!(batcher.stats().cancelled, 1);
}

struct SwitchGate(AtomicBool);

impl BatchGate for SwitchGate {
    fn admits(&self, _key: &BatchKey) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn a_generation_revoked_while_a_request_waited_is_refused_at_dispatch() {
    let latch = Arc::new(Latch::default());
    let backend = Arc::new(RecordingBackend::blocking(Arc::clone(&latch)));
    let gate = Arc::new(SwitchGate(AtomicBool::new(true)));
    let batcher = AdaptiveBatcher::with_parts(
        Arc::clone(&backend),
        policy(1, 0, 4),
        Arc::new(SystemClock),
        Some(Arc::clone(&gate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let held = scope.spawn(move || batcher.infer_with(&request("held"), admission("a")));
        assert!(eventually(PATIENT, || batcher.stats().dispatched == 1));
        let waiting = scope.spawn(move || batcher.infer_with(&request("stale"), admission("a")));
        assert!(eventually(PATIENT, || batcher.queued() == 1));
        // The request was admitted while the key was live; the revocation lands afterwards.
        gate.0.store(false, Ordering::SeqCst);
        latch.open();
        held.join()
            .expect("thread")
            .expect("already running, served");
        assert_eq!(waiting.join().expect("thread"), Err(ModelError::revoked()));
    });
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1, "stale never ran");
    assert_eq!(batcher.stats().revoked, 1);
}

#[test]
fn a_backend_panic_fails_its_batch_and_the_batcher_keeps_serving() {
    let backend = Arc::new(RecordingBackend::default());
    backend.panic_on_batch.store(true, Ordering::SeqCst);
    let batcher = gated(Arc::clone(&backend), policy(1, 0, 4)).expect("built");
    assert_eq!(
        batcher.infer(&request("boom")),
        Err(ModelError::backend_failure())
    );
    backend.panic_on_batch.store(false, Ordering::SeqCst);
    let events = batcher.infer(&request("after")).expect("serves again");
    assert_eq!(tokens(&events), vec!["after".to_owned()]);
    assert_eq!(batcher.stats().backend_errors, 1);
}

#[test]
fn a_backend_that_answers_the_wrong_count_fails_the_whole_batch() {
    let backend = Arc::new(RecordingBackend::default());
    backend.short_answer.store(true, Ordering::SeqCst);
    let batcher = gated(Arc::clone(&backend), policy(1, 0, 4)).expect("built");
    assert_eq!(
        batcher.infer(&request("short")),
        Err(ModelError::backend_failure())
    );
}

#[test]
fn a_policy_outside_its_ranges_is_refused_by_field() {
    let backend = Arc::new(RecordingBackend::default());
    let cases = [
        (policy(0, 1, 1), "max_batch"),
        (policy(1, 1, 0), "queue_capacity"),
        (policy(1, 10_001, 1), "max_wait_millis"),
        (
            BatchPolicy {
                deadline_margin_millis: 10_001,
                ..policy(1, 1, 1)
            },
            "deadline_margin_millis",
        ),
        (policy(1025, 1, 1), "max_batch"),
        (policy(1, 1, 65_537), "queue_capacity"),
    ];
    for (bad, field) in cases {
        let error = gated(Arc::clone(&backend), bad).expect_err("refused");
        assert!(
            matches!(error, BatchError::InvalidPolicy { field: named, .. } if named == field),
            "{field}: {error:?}"
        );
    }
}

#[test]
fn the_debug_form_names_the_backend_and_prints_no_prompt() {
    let backend = Arc::new(RecordingBackend::default());
    let batcher = gated(backend, policy(1, 0, 1)).expect("built");
    let printed = format!("{batcher:?}");
    assert!(printed.contains("recording"));
    assert!(!printed.contains("text of"));
}

fn keyed(project: &str, adapter: &str) -> BatchAdmission {
    BatchAdmission {
        key: BatchKey {
            project: Some(project.into()),
            adapter: Some(adapter.into()),
            principal: Some("alice".into()),
            generation: Generation(1),
        },
        deadline: None,
        cancel: None,
    }
}

/// Admits the keys of live projects only.
struct ProjectGate(&'static [&'static str]);

impl BatchGate for ProjectGate {
    fn admits(&self, key: &BatchKey) -> bool {
        key.project
            .as_deref()
            .is_some_and(|project| self.0.contains(&project))
    }
}

#[test]
fn one_projects_revoked_adapter_does_not_ride_with_another_projects_live_one() {
    let backend = Arc::new(RecordingBackend::default());
    let gate = Arc::new(ProjectGate(&["live"]));
    let batcher = AdaptiveBatcher::with_parts(
        Arc::clone(&backend),
        policy(8, 100, 16),
        Arc::new(SystemClock),
        Some(gate as Arc<dyn BatchGate>),
    )
    .expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let live =
            scope.spawn(move || batcher.infer_with(&request("live-1"), keyed("live", "coder")));
        let revoked =
            scope.spawn(move || batcher.infer_with(&request("dead-1"), keyed("dead", "coder")));
        assert!(live.join().expect("thread").is_ok());
        assert_eq!(revoked.join().expect("thread"), Err(ModelError::revoked()));
    });
    for batch in backend.recorded() {
        assert!(
            batch.iter().all(|id| id.starts_with("live")),
            "the revoked project's request reached the model: {batch:?}"
        );
    }
    assert_eq!(batcher.stats().revoked, 1);
}

#[test]
fn a_key_naming_an_adapter_is_refused_when_no_gate_vouches_for_it() {
    let backend = Arc::new(RecordingBackend::default());
    let batcher = AdaptiveBatcher::new(Arc::clone(&backend), policy(1, 0, 4)).expect("built");
    assert_eq!(
        batcher.infer_with(&request("a"), keyed("p", "coder")),
        Err(ModelError::revoked())
    );
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    // A key that names no adapter needs no gate.
    assert!(batcher.infer(&request("plain")).is_ok());
}

fn echo_batcher(gate: Arc<dyn BatchGate>) -> AdaptiveBatcher<ptr_model_api::ReferenceEchoBackend> {
    AdaptiveBatcher::with_parts(
        Arc::new(ptr_model_api::ReferenceEchoBackend),
        policy(1, 0, 4),
        Arc::new(SystemClock),
        Some(gate),
    )
    .expect("built")
}

/// A gate that holds the dispatching worker until the test lets it go.
struct HeldGate {
    entered: AtomicBool,
    latch: Latch,
}

impl BatchGate for HeldGate {
    fn admits(&self, _key: &BatchKey) -> bool {
        self.entered.store(true, Ordering::SeqCst);
        self.latch.wait();
        true
    }
}

#[test]
fn a_deadline_that_passes_while_the_gate_decides_keeps_the_request_from_the_backend() {
    let backend = Arc::new(RecordingBackend::default());
    let gate = Arc::new(HeldGate {
        entered: AtomicBool::new(false),
        latch: Latch::default(),
    });
    let batcher = AdaptiveBatcher::with_parts(
        Arc::clone(&backend),
        policy(1, 0, 4),
        Arc::new(SystemClock),
        Some(Arc::clone(&gate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let soon = Timestamp(SystemClock.now().0 + 100);
        let waiting = scope.spawn(move || {
            batcher.infer_with(
                &request("late"),
                BatchAdmission {
                    key: key("a"),
                    deadline: Some(soon),
                    cancel: None,
                },
            )
        });
        assert!(eventually(PATIENT, || gate.entered.load(Ordering::SeqCst)));
        std::thread::sleep(Duration::from_millis(200));
        gate.latch.open();
        assert_eq!(
            waiting.join().expect("thread"),
            Err(ModelError::deadline_exceeded())
        );
    });
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    assert_eq!(batcher.stats().deadline_drops, 1);
}

#[test]
fn a_cancellation_that_lands_while_the_gate_decides_keeps_the_request_from_the_backend() {
    let backend = Arc::new(RecordingBackend::default());
    let gate = Arc::new(HeldGate {
        entered: AtomicBool::new(false),
        latch: Latch::default(),
    });
    let batcher = AdaptiveBatcher::with_parts(
        Arc::clone(&backend),
        policy(1, 0, 4),
        Arc::new(SystemClock),
        Some(Arc::clone(&gate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    let flag = CancelFlag::default();
    std::thread::scope(|scope| {
        let batcher = &batcher;
        let cancel = flag.clone();
        let waiting = scope.spawn(move || {
            batcher.infer_with(
                &request("withdrawn"),
                BatchAdmission {
                    key: key("a"),
                    deadline: None,
                    cancel: Some(cancel),
                },
            )
        });
        assert!(eventually(PATIENT, || gate.entered.load(Ordering::SeqCst)));
        flag.cancel();
        gate.latch.open();
        assert_eq!(
            waiting.join().expect("thread"),
            Err(ModelError::cancelled())
        );
    });
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    assert_eq!(batcher.stats().cancelled, 1);
}

/// A backend whose resume panics.
struct PanickingResume;

impl ptr_model_api::InferenceBackend for PanickingResume {
    fn name(&self) -> &'static str {
        "panicking-resume"
    }

    fn infer(&self, _request: &ptr_model_api::ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

impl ptr_model_api::BatchInferenceBackend for PanickingResume {}

impl ptr_model_api::ResumableInferenceBackend for PanickingResume {
    fn resume(
        &self,
        _request: &ptr_model_api::ModelResumeRequest,
    ) -> Result<Vec<ModelEvent>, ModelError> {
        panic!("resume panics on purpose");
    }
}

#[test]
fn the_unscoped_resume_serves_only_a_batcher_without_a_gate() {
    use ptr_model_api::{ReferenceEchoBackend, ResumableInferenceBackend};
    let ungated =
        AdaptiveBatcher::new(Arc::new(ReferenceEchoBackend), policy(1, 0, 4)).expect("built");
    assert!(ungated.resume(&common::resume_request("r")).is_ok());
    let gated = echo_batcher(Arc::new(OpenGate));
    assert_eq!(
        gated.resume(&common::resume_request("r")),
        Err(ModelError::revoked())
    );
}

/// A request whose semantic context is at `revision`.
fn request_at(id: &str, revision: u64) -> ModelRequest {
    ModelRequest {
        semantic_context: SemanticContext {
            revision: Revision(revision),
            entries: vec![],
        },
        ..request(id)
    }
}

fn observation(revision: u64, payload: &[u8]) -> ModelObservation {
    ModelObservation {
        revision: Revision(revision),
        source: "test".into(),
        type_id: TypeId::from("test.v1"),
        payload: payload.to_vec(),
    }
}

fn admission_under(key: &BatchKey) -> BatchAdmission {
    BatchAdmission {
        key: key.clone(),
        deadline: None,
        cancel: None,
    }
}

/// Open a continuation for `id` under `key`.
fn open<B: ptr_model_api::BatchInferenceBackend + 'static>(
    batcher: &AdaptiveBatcher<B>,
    id: &str,
    key: &BatchKey,
) -> Continuation {
    batcher
        .infer_scoped(&request_at(id, 7), admission_under(key))
        .expect("served")
        .continuation
}

fn scope(project: &str, generation: u64) -> BatchKey {
    BatchKey {
        project: Some(project.into()),
        adapter: Some("coder".into()),
        principal: None,
        generation: Generation(generation),
    }
}

/// Admits every generation except the ones listed as revoked.
struct RevokedGenerations(std::sync::Mutex<Vec<u64>>);

impl BatchGate for RevokedGenerations {
    fn admits(&self, key: &BatchKey) -> bool {
        !self.0.lock().unwrap().contains(&key.generation.0)
    }
}

/// An echo backend that keeps the resume requests it was handed.
#[derive(Default)]
struct ContextRecorder {
    resumed: std::sync::Mutex<Vec<ModelResumeRequest>>,
}

impl InferenceBackend for ContextRecorder {
    fn name(&self) -> &'static str {
        "context-recorder"
    }

    fn infer(&self, _request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

impl ptr_model_api::BatchInferenceBackend for ContextRecorder {}

impl ptr_model_api::ResumableInferenceBackend for ContextRecorder {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.resumed.lock().unwrap().push(request.clone());
        Ok(vec![ModelEvent::Finished])
    }
}

#[test]
fn a_resume_runs_on_the_stored_context_and_round_and_never_on_the_callers() {
    let backend = Arc::new(ContextRecorder::default());
    let batcher = AdaptiveBatcher::with_parts(
        Arc::clone(&backend),
        policy(1, 0, 4),
        Arc::new(SystemClock),
        Some(Arc::new(OpenGate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    let key = key("coder");
    let first = open(&batcher, "r", &key);
    // The caller supplies an observation, which carries its own revision; the request the
    // backend sees takes id, semantic context and round from the table.
    let resumed = batcher
        .resume_continuation(first, &key, observation(99, b"payload"))
        .expect("resumes");
    let seen = backend.resumed.lock().unwrap()[0].clone();
    assert_eq!(seen.request_id, request("r").request_id);
    assert_eq!(seen.semantic_context.revision, Revision(7));
    assert_eq!(seen.round, 1);
    assert_eq!(seen.observation.revision, Revision(99));
    assert_eq!(seen.observation.payload, b"payload".to_vec());
    // The next round's continuation carries round 2 and the same context.
    let next = resumed.next.expect("room for a next round");
    batcher
        .resume_continuation(next, &key, observation(100, b"again"))
        .expect("resumes again");
    let seen = backend.resumed.lock().unwrap()[1].clone();
    assert_eq!(seen.round, 2);
    assert_eq!(seen.semantic_context.revision, Revision(7));
}

#[test]
fn a_continuation_from_a_revoked_scope_is_refused_and_each_is_single_use() {
    let gate = Arc::new(RevokedGenerations(std::sync::Mutex::new(Vec::new())));
    let batcher = echo_batcher(Arc::clone(&gate) as Arc<dyn BatchGate>);
    let gen2 = scope("p", 2);
    let gen3 = scope("p", 3);
    // One request id, identical outputs, a scope that will be revoked (A) and a live one (B).
    let a = open(&batcher, "same-id", &gen2);
    let a_again = open(&batcher, "same-id", &gen2);
    let a_wrong_key = open(&batcher, "same-id", &gen2);
    let b = open(&batcher, "same-id", &gen3);
    let b_second = open(&batcher, "same-id", &gen3);
    gate.0.lock().unwrap().push(2);
    // A's continuation reaches the gate under A's scope: revoked.
    assert_eq!(
        batcher
            .resume_continuation(a, &gen2, observation(1, b"x"))
            .err(),
        Some(ModelError::revoked())
    );
    // A's continuation claimed under B's live key is a scope mismatch.
    assert_eq!(
        batcher
            .resume_continuation(a_wrong_key, &gen3, observation(1, b"x"))
            .err(),
        Some(ModelError::scope_mismatch())
    );
    // B's continuations pass, each once; the handle is consumed by the call.
    assert!(batcher
        .resume_continuation(b, &gen3, observation(1, b"x"))
        .is_ok());
    assert!(batcher
        .resume_continuation(b_second, &gen3, observation(1, b"x"))
        .is_ok());
    // A continuation that was already consumed by a refused resume is gone: a second
    // continuation of A is still A's and still revoked.
    assert_eq!(
        batcher
            .resume_continuation(a_again, &gen2, observation(1, b"x"))
            .err(),
        Some(ModelError::revoked())
    );
}

#[test]
fn the_open_continuations_are_bounded_and_an_evicted_one_fails_closed() {
    let batcher = echo_batcher(Arc::new(OpenGate));
    let key = key("coder");
    let first = open(&batcher, "first", &key);
    for n in 0..ptr_model_api::MAX_OPEN_CONTINUATIONS {
        batcher
            .infer_scoped(&request(&format!("flood-{n}")), admission_under(&key))
            .expect("served");
    }
    assert_eq!(
        batcher
            .resume_continuation(first, &key, observation(1, b"x"))
            .err(),
        Some(ModelError::scope_mismatch())
    );
}

fn principal_scope(principal: &str) -> BatchKey {
    BatchKey {
        principal: Some(principal.into()),
        ..key("coder")
    }
}

#[test]
fn one_principal_cannot_push_another_principals_continuation_out() {
    let batcher = echo_batcher(Arc::new(OpenGate));
    let x = principal_scope("x");
    let y = principal_scope("y");
    let kept = open(&batcher, "y-run", &y);
    // 5000 continuations from X, far past both its quota and the global bound.
    let mut x_first = None;
    let mut x_last = None;
    for n in 0..5000 {
        let continuation = open(&batcher, &format!("x-{n}"), &x);
        if n == 0 {
            x_first = Some(continuation);
        } else if n == 4999 {
            x_last = Some(continuation);
        }
    }
    // Y's continuation is still there and resumes.
    assert!(batcher
        .resume_continuation(kept, &y, observation(1, b"x"))
        .is_ok());
    // X's own oldest failed closed once X passed its quota; its newest still resumes.
    assert_eq!(
        batcher
            .resume_continuation(x_first.unwrap(), &x, observation(1, b"x"))
            .err(),
        Some(ModelError::scope_mismatch())
    );
    assert!(batcher
        .resume_continuation(x_last.unwrap(), &x, observation(1, b"x"))
        .is_ok());
}

#[test]
fn many_principals_cannot_push_a_victim_out_and_a_full_table_refuses_the_next_open() {
    use ptr_model_api::MAX_OPEN_CONTINUATIONS;
    let batcher = echo_batcher(Arc::new(OpenGate));
    let victim_scope = principal_scope("victim");
    let victim = open(&batcher, "victim-run", &victim_scope);
    // 4095 more, each under its own principal, fill the table without any quota being hit.
    let mut held = Vec::new();
    for n in 0..MAX_OPEN_CONTINUATIONS - 1 {
        held.push(open(
            &batcher,
            &format!("sybil-{n}"),
            &principal_scope(&format!("sybil-{n}")),
        ));
    }
    // The next open is refused, and nothing was evicted for it.
    assert_eq!(
        batcher
            .infer_scoped(
                &request("one-too-many"),
                admission_under(&principal_scope("late"))
            )
            .err(),
        Some(ModelError::overloaded())
    );
    // The victim's continuation survived: it resumes, and its slot carries over to the next
    // round of the same run.
    let resumed = batcher
        .resume_continuation(victim, &victim_scope, observation(1, b"x"))
        .expect("the victim was not displaced");
    assert!(resumed.next.is_some());
    drop(held);
}

#[test]
fn an_expired_continuation_frees_its_slot_in_a_full_table() {
    use ptr_model_api::MAX_OPEN_CONTINUATIONS;
    let clock = Arc::new(SetClock(std::sync::atomic::AtomicU64::new(0)));
    let batcher = AdaptiveBatcher::with_parts(
        Arc::new(ptr_model_api::ReferenceEchoBackend),
        policy(1, 0, 4),
        Arc::clone(&clock) as Arc<dyn BatchClock>,
        Some(Arc::new(OpenGate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    for n in 0..MAX_OPEN_CONTINUATIONS {
        let scope = principal_scope(&format!("p{n}"));
        batcher
            .infer_scoped(
                &request(&format!("r{n}")),
                BatchAdmission {
                    key: scope,
                    deadline: Some(Timestamp(100)),
                    cancel: None,
                },
            )
            .expect("served");
    }
    let late = principal_scope("late");
    assert_eq!(
        batcher
            .infer_scoped(&request("late"), admission_under(&late))
            .err(),
        Some(ModelError::overloaded())
    );
    // The deadlines pass: the expired continuations are swept and the slot is free again.
    clock.0.store(100, Ordering::SeqCst);
    assert!(batcher
        .infer_scoped(&request("late"), admission_under(&late))
        .is_ok());
}

#[test]
fn plain_inference_opens_no_continuation() {
    let batcher = echo_batcher(Arc::new(OpenGate));
    let key = key("coder");
    let kept = open(&batcher, "kept", &key);
    for n in 0..5000 {
        batcher
            .infer_with(&request(&format!("plain-{n}")), admission_under(&key))
            .expect("served");
        batcher
            .infer(&request(&format!("unscoped-{n}")))
            .expect("served");
    }
    assert!(batcher
        .resume_continuation(kept, &key, observation(1, b"x"))
        .is_ok());
}

#[test]
fn a_resume_is_stopped_by_a_closed_gate() {
    let gate = Arc::new(SwitchGate(AtomicBool::new(true)));
    let batcher = echo_batcher(Arc::clone(&gate) as Arc<dyn BatchGate>);
    let scoped = key("coder");
    let continuation = open(&batcher, "r", &scoped);
    gate.0.store(false, Ordering::SeqCst);
    assert_eq!(
        batcher
            .resume_continuation(continuation, &scoped, observation(1, b"x"))
            .err(),
        Some(ModelError::revoked())
    );
}

#[test]
fn a_backend_panic_in_resume_fails_that_call_and_the_batcher_keeps_serving() {
    let batcher = AdaptiveBatcher::with_parts(
        Arc::new(PanickingResume),
        policy(1, 0, 4),
        Arc::new(SystemClock),
        Some(Arc::new(OpenGate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    let continuation = open(&batcher, "r", &BatchKey::unscoped());
    assert_eq!(
        batcher
            .resume_continuation(continuation, &BatchKey::unscoped(), observation(1, b"x"))
            .err(),
        Some(ModelError::backend_failure())
    );
    assert_eq!(batcher.stats().backend_errors, 1);
    assert!(batcher.infer(&request("after")).is_ok());
}

/// A gate that panics once armed.
struct PanickingGate(AtomicBool);

impl BatchGate for PanickingGate {
    fn admits(&self, _key: &BatchKey) -> bool {
        if self.0.load(Ordering::SeqCst) {
            panic!("gate panics on purpose");
        }
        true
    }
}

#[test]
fn a_gate_that_panics_refuses_the_batch_and_the_resume_and_the_worker_keeps_serving() {
    let gate = Arc::new(PanickingGate(AtomicBool::new(false)));
    let batcher = echo_batcher(Arc::clone(&gate) as Arc<dyn BatchGate>);
    let continuation = open(&batcher, "r", &key("coder"));
    gate.0.store(true, Ordering::SeqCst);
    for n in 0..3 {
        assert_eq!(
            batcher.infer_with(&request(&format!("r{n}")), keyed("p", "coder")),
            Err(ModelError::revoked())
        );
    }
    assert_eq!(batcher.stats().dispatched, 1);
    assert_eq!(batcher.stats().revoked, 3);
    assert_eq!(
        batcher
            .resume_continuation(continuation, &key("coder"), observation(1, b"x"))
            .err(),
        Some(ModelError::revoked())
    );
    gate.0.store(false, Ordering::SeqCst);
    assert!(batcher.infer(&request("after")).is_ok());
}

/// A clock that panics on the batcher's worker thread once armed.
struct WorkerKillingClock {
    armed: AtomicBool,
}

impl BatchClock for WorkerKillingClock {
    fn now(&self) -> Timestamp {
        if self.armed.load(Ordering::SeqCst) && std::thread::current().name() == Some("ptr-batcher")
        {
            panic!("the worker dies on purpose");
        }
        Timestamp(0)
    }
}

#[test]
fn a_dead_worker_closes_the_batcher_and_answers_every_waiting_caller() {
    let clock = Arc::new(WorkerKillingClock {
        armed: AtomicBool::new(false),
    });
    let batcher = AdaptiveBatcher::with_parts(
        Arc::new(RecordingBackend::default()),
        policy(2, 10_000, 8),
        Arc::clone(&clock) as Arc<dyn BatchClock>,
        Some(Arc::new(OpenGate) as Arc<dyn BatchGate>),
    )
    .expect("built");
    std::thread::scope(|scope| {
        let batcher = &batcher;
        // Waits for company that never comes (max_wait is long, the batch is not full).
        let waiting = scope.spawn(move || batcher.infer(&request("waiting")));
        assert!(eventually(PATIENT, || batcher.queued() == 1));
        clock.armed.store(true, Ordering::SeqCst);
        // Waking the worker makes it read the clock, which kills it.
        let second = scope.spawn(move || batcher.infer(&request("second")));
        assert_eq!(waiting.join().expect("thread"), Err(ModelError::closed()));
        assert_eq!(second.join().expect("thread"), Err(ModelError::closed()));
    });
    assert_eq!(batcher.infer(&request("after")), Err(ModelError::closed()));
}

/// A clock the test sets.
struct SetClock(std::sync::atomic::AtomicU64);

impl BatchClock for SetClock {
    fn now(&self) -> Timestamp {
        Timestamp(self.0.load(Ordering::SeqCst))
    }
}

fn recorder_batcher(
    backend: &Arc<ContextRecorder>,
    clock: Arc<dyn BatchClock>,
    gate: Option<Arc<dyn BatchGate>>,
) -> AdaptiveBatcher<ContextRecorder> {
    AdaptiveBatcher::with_parts(Arc::clone(backend), policy(1, 0, 4), clock, gate).expect("built")
}

#[test]
fn a_resume_after_the_deadline_or_a_cancellation_never_reaches_the_backend() {
    let backend = Arc::new(ContextRecorder::default());
    let clock = Arc::new(SetClock(std::sync::atomic::AtomicU64::new(0)));
    let batcher = recorder_batcher(
        &backend,
        Arc::clone(&clock) as Arc<dyn BatchClock>,
        Some(Arc::new(OpenGate) as Arc<dyn BatchGate>),
    );
    let key = key("coder");
    let cancel = CancelFlag::default();
    let admit = |deadline: Option<u64>, cancel: Option<CancelFlag>| BatchAdmission {
        key: key.clone(),
        deadline: deadline.map(Timestamp),
        cancel,
    };
    let in_time = batcher
        .infer_scoped(&request_at("a", 1), admit(Some(100), None))
        .expect("served")
        .continuation;
    let cancellable = batcher
        .infer_scoped(&request_at("b", 1), admit(None, Some(cancel.clone())))
        .expect("served")
        .continuation;
    let live = batcher
        .infer_scoped(&request_at("c", 1), admit(Some(100), None))
        .expect("served")
        .continuation;
    // Within the deadline and not cancelled: runs.
    assert!(batcher
        .resume_continuation(live, &key, observation(1, b"x"))
        .is_ok());
    let ran = backend.resumed.lock().unwrap().len();
    // The deadline passes: refused, and the backend gets nothing.
    clock.0.store(100, Ordering::SeqCst);
    assert_eq!(
        batcher
            .resume_continuation(in_time, &key, observation(1, b"x"))
            .err(),
        Some(ModelError::deadline_exceeded())
    );
    // A cancellation lands: refused, and the backend gets nothing.
    cancel.cancel();
    assert_eq!(
        batcher
            .resume_continuation(cancellable, &key, observation(1, b"x"))
            .err(),
        Some(ModelError::cancelled())
    );
    assert_eq!(backend.resumed.lock().unwrap().len(), ran);
}

#[test]
fn every_resume_path_is_refused_once_the_batcher_is_closed() {
    use ptr_model_api::ResumableInferenceBackend;
    for gated in [true, false] {
        let backend = Arc::new(ContextRecorder::default());
        let clock = Arc::new(WorkerKillingClock {
            armed: AtomicBool::new(false),
        });
        let gate = gated.then(|| Arc::new(OpenGate) as Arc<dyn BatchGate>);
        let batcher = recorder_batcher(&backend, Arc::clone(&clock) as Arc<dyn BatchClock>, gate);
        let key = if gated {
            key("coder")
        } else {
            BatchKey::unscoped()
        };
        let continuation = open(&batcher, "r", &key);
        // Kill the worker: it reads the clock when a request arrives and panics.
        clock.armed.store(true, Ordering::SeqCst);
        assert_eq!(batcher.infer(&request("dies")), Err(ModelError::closed()));
        assert_eq!(
            batcher
                .resume_continuation(continuation, &key, observation(1, b"x"))
                .err(),
            Some(ModelError::closed())
        );
        if !gated {
            assert_eq!(
                batcher.resume(&common::resume_request("r")),
                Err(ModelError::closed())
            );
        }
        assert!(backend.resumed.lock().unwrap().is_empty(), "gated: {gated}");
    }
}
