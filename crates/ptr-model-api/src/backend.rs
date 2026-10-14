use crate::{ModelEvent, ModelRequest, ModelResumeRequest};
use ptr_types::{ActionIr, CapabilityId, Effect, Generation, TypeId};
use std::fmt;

/// What kind of failure a [`ModelError`] is. Routing decisions (failover) read the kind and
/// never the message.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ModelErrorKind {
    /// A bounded queue was full; nothing was run.
    Overloaded,
    /// The request's deadline had passed before the model was called.
    DeadlineExceeded,
    /// The caller withdrew the request before the model was called.
    Cancelled,
    /// The adapter or principal generation the request was admitted under is no longer admitted,
    /// or no gate vouches for it.
    Revoked,
    /// The backend was shut down.
    Closed,
    /// The backend failed for a reason that is not the request's (a panic, a malformed batch
    /// answer, an unreachable replica).
    BackendFailure,
    /// A request id is already bound to another scope.
    DuplicateRequest,
    /// Any other failure, including a refusal of the request itself (validation, policy). The
    /// default of [`ModelError::new`].
    Other,
}

/// A failed model call: a kind, which decides what may be done about it, and a message for
/// people.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError {
    kind: ModelErrorKind,
    message: String,
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ModelError {}

impl ModelError {
    /// An error of kind [`ModelErrorKind::Other`]: it is not retried on another replica.
    pub fn new(message: impl Into<String>) -> Self {
        Self::with_kind(ModelErrorKind::Other, message)
    }

    pub fn with_kind(kind: ModelErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ModelErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn overloaded() -> Self {
        Self::with_kind(ModelErrorKind::Overloaded, "model backend overloaded")
    }

    pub fn deadline_exceeded() -> Self {
        Self::with_kind(
            ModelErrorKind::DeadlineExceeded,
            "request deadline exceeded",
        )
    }

    pub fn cancelled() -> Self {
        Self::with_kind(ModelErrorKind::Cancelled, "request cancelled")
    }

    pub fn revoked() -> Self {
        Self::with_kind(
            ModelErrorKind::Revoked,
            "adapter generation no longer admitted",
        )
    }

    pub fn closed() -> Self {
        Self::with_kind(ModelErrorKind::Closed, "model backend closed")
    }

    /// The request id is already bound to a different scope; the request was not admitted.
    pub fn duplicate_request() -> Self {
        Self::with_kind(
            ModelErrorKind::DuplicateRequest,
            "request id already bound to another scope",
        )
    }

    pub fn backend_failure() -> Self {
        Self::with_kind(ModelErrorKind::BackendFailure, "model backend failed")
    }

    pub fn is_overloaded(&self) -> bool {
        self.kind == ModelErrorKind::Overloaded
    }

    pub fn is_deadline_exceeded(&self) -> bool {
        self.kind == ModelErrorKind::DeadlineExceeded
    }

    pub fn is_cancelled(&self) -> bool {
        self.kind == ModelErrorKind::Cancelled
    }

    pub fn is_revoked(&self) -> bool {
        self.kind == ModelErrorKind::Revoked
    }

    pub fn is_closed(&self) -> bool {
        self.kind == ModelErrorKind::Closed
    }

    /// Whether another replica may be tried: only for overload and for a backend failure, the
    /// two kinds a different replica can change. Every other kind, a refusal of the request
    /// itself included, ends the request.
    pub fn allows_failover(&self) -> bool {
        matches!(
            self.kind,
            ModelErrorKind::Overloaded | ModelErrorKind::BackendFailure
        )
    }
}

pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError>;
}

pub trait ResumableInferenceBackend: InferenceBackend {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError>;
}

/// Deterministic plumbing backend used only for runtime conformance tests.
///
/// It is not an LLM baseline and must not be reported as model-quality evidence.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceEchoBackend;

impl InferenceBackend for ReferenceEchoBackend {
    fn name(&self) -> &'static str {
        "reference-echo"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![
            ModelEvent::Token(request.raw_text.clone()),
            ModelEvent::Finished,
        ])
    }
}

impl ResumableInferenceBackend for ReferenceEchoBackend {
    fn resume(&self, _request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

/// Deterministic demonstrator backend for the local HTTP/effect path.
/// It is plumbing evidence only, not a model-quality baseline.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScenarioBackend;

impl InferenceBackend for ScenarioBackend {
    fn name(&self) -> &'static str {
        "scenario"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![
            ModelEvent::Token(request.raw_text.clone()),
            ModelEvent::ActionReady(ActionIr {
                operation: "create".into(),
                target: "demo-note".into(),
                capability: CapabilityId::from("demo.local-note.create"),
                effect: Effect::Mutation,
                input_type: TypeId::from("ptr.demo-note.v1"),
                generation: Generation(1),
                revision: request.revision(),
                payload: request.raw_text.as_bytes().to_vec(),
            }),
        ])
    }
}

impl ResumableInferenceBackend for ScenarioBackend {
    fn resume(&self, _request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}
