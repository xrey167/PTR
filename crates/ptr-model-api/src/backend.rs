use crate::{ModelEvent, ModelRequest};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);

pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError>;
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
