use crate::{ModelEvent, ModelRequest, ModelResumeRequest};
use ptr_types::{ActionIr, CapabilityId, Effect, Generation, TypeId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);

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
