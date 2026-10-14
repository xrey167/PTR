use crate::{ModelEvent, ModelRequest, ModelResumeRequest};
use ptr_types::{ActionIr, CapabilityId, Effect, Generation, TypeId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);

impl ModelError {
    const OVERLOADED: &'static str = "model backend overloaded";
    const DEADLINE_EXCEEDED: &'static str = "request deadline exceeded";
    const CANCELLED: &'static str = "request cancelled";
    const REVOKED: &'static str = "adapter generation no longer admitted";
    const CLOSED: &'static str = "model backend closed";
    const BACKEND_FAILURE: &'static str = "model backend failed";

    /// A bounded queue was full; nothing was run.
    pub fn overloaded() -> Self {
        Self(Self::OVERLOADED.into())
    }

    /// The request's deadline had passed before the model was called.
    pub fn deadline_exceeded() -> Self {
        Self(Self::DEADLINE_EXCEEDED.into())
    }

    /// The caller withdrew the request before the model was called.
    pub fn cancelled() -> Self {
        Self(Self::CANCELLED.into())
    }

    /// The adapter or principal generation the request was admitted under is no longer
    /// admitted.
    pub fn revoked() -> Self {
        Self(Self::REVOKED.into())
    }

    /// The backend was shut down.
    pub fn closed() -> Self {
        Self(Self::CLOSED.into())
    }

    /// The backend failed without a more specific reason (a panic or a malformed batch
    /// answer).
    pub fn backend_failure() -> Self {
        Self(Self::BACKEND_FAILURE.into())
    }

    pub fn is_overloaded(&self) -> bool {
        self.0 == Self::OVERLOADED
    }

    pub fn is_deadline_exceeded(&self) -> bool {
        self.0 == Self::DEADLINE_EXCEEDED
    }

    pub fn is_cancelled(&self) -> bool {
        self.0 == Self::CANCELLED
    }

    pub fn is_revoked(&self) -> bool {
        self.0 == Self::REVOKED
    }

    pub fn is_closed(&self) -> bool {
        self.0 == Self::CLOSED
    }

    /// Whether trying the request on another replica is pointless or wrong: its deadline
    /// passed, it was cancelled, its generation was revoked, or the backend is closed.
    pub fn refuses_failover(&self) -> bool {
        self.is_deadline_exceeded() || self.is_cancelled() || self.is_revoked() || self.is_closed()
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
