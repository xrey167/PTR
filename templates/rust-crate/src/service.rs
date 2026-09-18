use std::sync::Arc;

use crate::types::BackendRegistry;
use crate::{Backend, BackendState, ExecuteRequest, ExecuteResult, ServiceError};

pub struct Service<TInput, TOutput> {
    backends: BackendRegistry<Arc<dyn Backend<TInput, TOutput>>>,
}

impl<TInput, TOutput> Default for Service<TInput, TOutput> {
    fn default() -> Self {
        Self {
            backends: BackendRegistry::default(),
        }
    }
}

impl<TInput, TOutput> Service<TInput, TOutput> {
    pub fn register(
        &mut self,
        backend: Arc<dyn Backend<TInput, TOutput>>,
    ) -> Option<Arc<dyn Backend<TInput, TOutput>>> {
        self.backends.insert(backend.name(), backend)
    }

    pub fn execute(
        &self,
        request: ExecuteRequest<TInput>,
    ) -> Result<ExecuteResult<TOutput>, ServiceError> {
        let backend_name = request
            .preferred_backend
            .as_deref()
            .ok_or_else(|| ServiceError::BackendUnavailable {
                backend: "no backend selected".to_owned(),
            })?;

        let backend = self.backends.get(backend_name).ok_or_else(|| {
            ServiceError::BackendUnavailable {
                backend: backend_name.to_owned(),
            }
        })?;

        match backend.state() {
            BackendState::Ready | BackendState::Degraded => {
                let value = backend.execute(request.input)?;
                Ok(ExecuteResult {
                    value,
                    backend: backend_name.to_owned(),
                })
            }
            BackendState::Unavailable => Err(ServiceError::BackendUnavailable {
                backend: backend_name.to_owned(),
            }),
        }
    }
}
