use crate::{BackendState, ServiceError};

pub trait Backend<TInput, TOutput>: Send + Sync {
    fn name(&self) -> &str;

    fn state(&self) -> BackendState;

    fn execute(&self, input: TInput) -> Result<TOutput, ServiceError>;
}
