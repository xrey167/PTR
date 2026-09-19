use std::sync::Arc;

use crate::types::BackendRegistry;
use crate::{
    check_backend_name, validate_request, Backend, BackendState, ExecuteRequest, ExecuteResult,
    NoopTraceSink, ServiceError, TraceError, TraceEvent, TraceLevel, TraceSink,
};

pub struct Service<TInput, TOutput> {
    backends: BackendRegistry<Arc<dyn Backend<TInput, TOutput>>>,
    trace_sink: Arc<dyn TraceSink>,
}

impl<TInput, TOutput> Default for Service<TInput, TOutput> {
    fn default() -> Self {
        Self {
            backends: BackendRegistry::default(),
            trace_sink: Arc::new(NoopTraceSink),
        }
    }

}

impl<TInput, TOutput> Service<TInput, TOutput> {
    pub fn with_trace_sink(mut self, trace_sink: Arc<dyn TraceSink>) -> Self {
        self.trace_sink = trace_sink;
        self
    }

    pub fn register(
        &mut self,
        backend: Arc<dyn Backend<TInput, TOutput>>,
    ) -> Option<Arc<dyn Backend<TInput, TOutput>>> {
        let backend_name = backend.name().to_owned();
        self.backends.insert(backend_name, backend)
    }

    fn emit_trace(&self, event: TraceEvent) {
        if let Err(error) = self.trace_sink.emit(&event) {
            match error {
                TraceError::SinkUnavailable { .. } | TraceError::Export { .. } => {
                    // Explicit best-effort policy: telemetry cannot change semantic outcome.
                }
            }
        }
    }

    pub fn backend_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.backends.names()
    }

    pub fn matching_backend_names<'service, F>(
        &'service self,
        mut predicate: F,
    ) -> impl Iterator<Item = &'service str> + 'service
    where
        F: FnMut(&str) -> bool + 'service,
    {
        self.backend_names()
            .filter(move |name| predicate(name))
    }

    pub fn execute_all<I>(
        &self,
        requests: I,
    ) -> Result<Vec<ExecuteResult<TOutput>>, ServiceError>
    where
        I: IntoIterator<Item = ExecuteRequest<TInput>>,
    {
        requests
            .into_iter()
            .map(|request| self.execute(request))
            .collect()
    }

    pub fn execute(
        &self,
        request: ExecuteRequest<TInput>,
    ) -> Result<ExecuteResult<TOutput>, ServiceError> {
        validate_request(request.as_ref())?;
        let backend_name = check_backend_name(request.preferred_backend.as_deref())?;

        let backend = self.backends.get(backend_name).ok_or_else(|| {
            ServiceError::BackendUnavailable {
                backend: backend_name.to_owned(),
            }
        })?;

        self.emit_trace(
            TraceEvent::new(TraceLevel::Debug, "backend.execute")
                .with_field("backend", backend_name),
        );

        match backend.state() {
            BackendState::Ready | BackendState::Degraded => {
                let value = backend.execute(request.input)?;
                self.emit_trace(
                    TraceEvent::new(TraceLevel::Info, "backend.execute.completed")
                        .with_field("backend", backend_name),
                );
                Ok(ExecuteResult {
                    value,
                    backend: backend_name.to_owned(),
                })
            }
            actual @ BackendState::Unavailable => {
                let error = ServiceError::InvalidState {
                    expected: BackendState::Ready,
                    actual,
                };
                self.emit_trace(
                    TraceEvent::new(TraceLevel::Error, "backend.execute.rejected")
                        .with_field("backend", backend_name)
                        .with_field("error.code", error.code()),
                );
                Err(error)
            }
        }
    }
}
