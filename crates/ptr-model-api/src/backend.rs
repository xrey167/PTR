use crate::{ModelEvent, ModelRequest};
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError>;
}
