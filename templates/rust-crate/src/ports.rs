use crate::{BackendState, ServiceError};

pub trait Backend<TInput, TOutput>: Send + Sync {
    fn name(&self) -> &str;

    fn state(&self) -> BackendState;

    fn execute(&self, input: TInput) -> Result<TOutput, ServiceError>;
}

pub fn execute_cloned<TInput, TOutput, TBackend>(
    backend: &TBackend,
    input: &TInput,
) -> Result<TOutput, ServiceError>
where
    TBackend: Backend<TInput, TOutput> + ?Sized,
    TInput: Clone,
{
    backend.execute(input.clone())
}
