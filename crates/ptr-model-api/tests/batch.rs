//! The adaptive batcher through its public API: batching, key separation, the queue bound,
//! deadlines, cancellation, revocation at dispatch and failure containment.
mod common;

use common::{eventually, request, Latch, RecordingBackend};
use ptr_model_api::{
    AdaptiveBatcher, BatchAdmission, BatchClock, BatchError, BatchGate, BatchKey, BatchPolicy,
    CancelFlag, InferenceBackend, ModelError, ModelEvent, ResumeBinding, SystemClock,
};
use ptr_types::{Generation, Timestamp};
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

/// Answer `id` under `key`, record `payload` as the observation that resumes it, and return the
/// resume request together with the binding of the continuation.
fn prepared<B: ptr_model_api::BatchInferenceBackend + 'static>(
    batcher: &AdaptiveBatcher<B>,
    id: &str,
    key: &BatchKey,
    payload: &[u8],
) -> (ptr_model_api::ModelResumeRequest, ResumeBinding) {
    let answer = batcher
        .infer_scoped(
            &request(id),
            BatchAdmission {
                key: key.clone(),
                deadline: None,
                cancel: None,
            },
        )
        .expect("served");
    let mut resume = common::resume_request(id);
    resume.observation.payload = payload.to_vec();
    batcher
        .record_observation(&answer.binding, &resume.observation)
        .expect("recorded once");
    (resume, answer.binding)
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

#[test]
fn a_continuation_is_bound_to_the_scope_and_observation_it_was_recorded_under() {
    let gate = Arc::new(RevokedGenerations(std::sync::Mutex::new(Vec::new())));
    let batcher = echo_batcher(Arc::clone(&gate) as Arc<dyn BatchGate>);
    let gen2 = scope("p", 2);
    let gen3 = scope("p", 3);
    // The same request id and identical outputs under a scope that will be revoked (A) and a
    // live one (B): only the continuation id tells them apart.
    let (resume_a, binding_a) = prepared(&batcher, "same-id", &gen2, b"observation of A");
    let (resume_b, binding_b) = prepared(&batcher, "same-id", &gen3, b"observation of B");
    gate.0.lock().unwrap().push(2);
    // A's observation with B's binding and B's live key: the entry holds B's observation digest.
    assert_eq!(
        batcher.resume_with(&resume_a, &binding_b, &gen3),
        Err(ModelError::scope_mismatch())
    );
    // A's observation with A's binding but the live key: the entry holds A's scope.
    assert_eq!(
        batcher.resume_with(&resume_a, &binding_a, &gen3),
        Err(ModelError::scope_mismatch())
    );
    // Another project's key does not fit either.
    assert_eq!(
        batcher.resume_with(&resume_a, &binding_a, &scope("q", 2)),
        Err(ModelError::scope_mismatch())
    );
    // A's own pair reaches the gate under A's key: revoked, and the continuation is consumed.
    assert_eq!(
        batcher.resume_with(&resume_a, &binding_a, &gen2),
        Err(ModelError::revoked())
    );
    // B's own pair passes, once.
    assert!(batcher.resume_with(&resume_b, &binding_b, &gen3).is_ok());
    assert_eq!(
        batcher.resume_with(&resume_b, &binding_b, &gen3),
        Err(ModelError::scope_mismatch()),
        "a continuation is single-use"
    );
}

#[test]
fn a_continuation_without_a_recorded_observation_or_from_nowhere_is_refused() {
    let batcher = echo_batcher(Arc::new(OpenGate));
    let key = key("coder");
    let answer = batcher
        .infer_scoped(
            &request("r"),
            BatchAdmission {
                key: key.clone(),
                deadline: None,
                cancel: None,
            },
        )
        .expect("served");
    let resume = common::resume_request("r");
    assert_eq!(
        batcher.resume_with(&resume, &answer.binding, &key),
        Err(ModelError::scope_mismatch()),
        "no observation was recorded for it"
    );
    batcher
        .record_observation(&answer.binding, &resume.observation)
        .expect("recorded");
    assert_eq!(
        batcher.record_observation(&answer.binding, &resume.observation),
        Err(ModelError::scope_mismatch()),
        "the observation of a continuation is recorded once"
    );
    // A binding from another batcher is an id this one never issued.
    let other = echo_batcher(Arc::new(OpenGate));
    let foreign = other
        .infer_scoped(
            &request("r"),
            BatchAdmission {
                key: key.clone(),
                deadline: None,
                cancel: None,
            },
        )
        .expect("served")
        .binding;
    assert_eq!(
        batcher.resume_with(&resume, &foreign, &key),
        Err(ModelError::scope_mismatch())
    );
    assert!(batcher.resume_with(&resume, &answer.binding, &key).is_ok());
}

#[test]
fn the_open_continuations_are_bounded_and_an_evicted_one_fails_closed() {
    let batcher = echo_batcher(Arc::new(OpenGate));
    let key = key("coder");
    let (first_resume, first) = prepared(&batcher, "first", &key, b"first");
    for n in 0..ptr_model_api::MAX_OPEN_CONTINUATIONS {
        batcher
            .infer_scoped(
                &request(&format!("flood-{n}")),
                BatchAdmission {
                    key: key.clone(),
                    deadline: None,
                    cancel: None,
                },
            )
            .expect("served");
    }
    assert_eq!(
        batcher.resume_with(&first_resume, &first, &key),
        Err(ModelError::scope_mismatch())
    );
}

#[test]
fn a_resume_is_stopped_by_a_closed_gate() {
    let gate = Arc::new(SwitchGate(AtomicBool::new(true)));
    let batcher = echo_batcher(Arc::clone(&gate) as Arc<dyn BatchGate>);
    let scoped = key("coder");
    let (resume, binding) = prepared(&batcher, "r", &scoped, b"x");
    gate.0.store(false, Ordering::SeqCst);
    assert_eq!(
        batcher.resume_with(&resume, &binding, &scoped),
        Err(ModelError::revoked())
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
    let (resume, binding) = prepared(&batcher, "r", &BatchKey::unscoped(), b"x");
    assert_eq!(
        batcher.resume_with(&resume, &binding, &BatchKey::unscoped()),
        Err(ModelError::backend_failure())
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
    let (resume, binding) = prepared(&batcher, "r", &key("coder"), b"x");
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
        batcher.resume_with(&resume, &binding, &key("coder")),
        Err(ModelError::revoked())
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
