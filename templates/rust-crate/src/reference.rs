use crate::{Backend, BackendState, ServiceError};

#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceBackend;

impl Backend<String, String> for ReferenceBackend {
    fn name(&self) -> &str {
        "reference"
    }

    fn state(&self) -> BackendState {
        BackendState::Ready
    }

    fn execute(&self, input: String) -> Result<String, ServiceError> {
        Ok(input)
    }
}
