use crate::BackendState;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceError {
    BackendUnavailable {
        backend: String,
    },
    InvalidState {
        expected: BackendState,
        actual: BackendState,
    },
    Backend {
        backend: String,
        message: String,
    },
}

impl ServiceError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::BackendUnavailable { .. } => "backend_unavailable",
            Self::InvalidState { .. } => "invalid_state",
            Self::Backend { .. } => "backend",
        }
    }
}
