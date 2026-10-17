//! Backends and builders shared by the batcher and replica tests. No test lives here.
#![allow(dead_code)]

use ptr_model_api::{
    BatchInferenceBackend, InferenceBackend, ModelError, ModelEvent, ModelObservation,
    ModelRequest, ModelResumeRequest, ResumableInferenceBackend,
};
use ptr_types::{RequestId, Revision, SemanticContext, TypeId};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

pub fn request(id: &str) -> ModelRequest {
    ModelRequest {
        request_id: RequestId::from(id),
        semantic_context: SemanticContext {
            revision: Revision(1),
            entries: vec![],
        },
        raw_text: format!("text of {id}"),
    }
}

pub fn resume_request(id: &str) -> ModelResumeRequest {
    ModelResumeRequest {
        request_id: RequestId::from(id),
        semantic_context: SemanticContext {
            revision: Revision(1),
            entries: vec![],
        },
        round: 1,
        observation: ModelObservation {
            revision: Revision(1),
            source: "test".into(),
            type_id: TypeId::from("test.v1"),
            payload: vec![],
        },
    }
}

/// A latch the test opens to let blocked backend calls proceed.
#[derive(Default)]
pub struct Latch {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Latch {
    pub fn open(&self) {
        *self.open.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.changed.notify_all();
    }

    pub fn wait(&self) {
        let mut open = self.open.lock().unwrap_or_else(PoisonError::into_inner);
        while !*open {
            open = self
                .changed
                .wait(open)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Echoes the request id, records every batch it was given, and can block behind a latch.
#[derive(Default)]
pub struct RecordingBackend {
    pub batches: Mutex<Vec<Vec<String>>>,
    pub calls: AtomicUsize,
    pub block: Option<std::sync::Arc<Latch>>,
    pub panic_on_batch: AtomicBool,
    pub short_answer: AtomicBool,
}

impl RecordingBackend {
    pub fn blocking(latch: std::sync::Arc<Latch>) -> Self {
        Self {
            block: Some(latch),
            ..Self::default()
        }
    }

    pub fn recorded(&self) -> Vec<Vec<String>> {
        self.batches
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl InferenceBackend for RecordingBackend {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.infer_batch(&[request])
            .pop()
            .unwrap_or_else(|| Err(ModelError::backend_failure()))
    }
}

impl BatchInferenceBackend for RecordingBackend {
    fn infer_batch(&self, requests: &[&ModelRequest]) -> Vec<Result<Vec<ModelEvent>, ModelError>> {
        self.calls.fetch_add(requests.len(), Ordering::SeqCst);
        self.batches
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(requests.iter().map(|r| r.request_id.to_string()).collect());
        if let Some(latch) = &self.block {
            latch.wait();
        }
        assert!(
            !self.panic_on_batch.load(Ordering::SeqCst),
            "backend panics on purpose"
        );
        let mut results: Vec<_> = requests
            .iter()
            .map(|r| Ok(vec![ModelEvent::Token(r.request_id.to_string())]))
            .collect();
        if self.short_answer.load(Ordering::SeqCst) {
            results.pop();
        }
        results
    }
}

/// A replica that fails with a chosen error a chosen number of times, then succeeds.
pub struct FlakyReplica {
    pub label: &'static str,
    pub failures_left: Mutex<usize>,
    pub error: ModelError,
    pub calls: AtomicUsize,
}

impl FlakyReplica {
    pub fn new(label: &'static str, failures: usize, error: ModelError) -> Self {
        Self {
            label,
            failures_left: Mutex::new(failures),
            error,
            calls: AtomicUsize::new(0),
        }
    }

    fn answer(&self) -> Result<Vec<ModelEvent>, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let failing = {
            let mut left = self
                .failures_left
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let failing = *left > 0;
            *left = left.saturating_sub(1);
            failing
        };
        if failing {
            Err(self.error.clone())
        } else {
            Ok(vec![ModelEvent::Token(self.label.into())])
        }
    }
}

impl InferenceBackend for FlakyReplica {
    fn name(&self) -> &'static str {
        "flaky"
    }

    fn infer(&self, _request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.answer()
    }
}

impl ResumableInferenceBackend for FlakyReplica {
    fn resume(&self, _request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.answer()
    }
}

/// Poll `condition` until it holds or `within` passes; true when it held.
pub fn eventually(within: Duration, condition: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < within {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    condition()
}
