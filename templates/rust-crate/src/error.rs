use std::fmt;

use crate::BackendState;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    MissingField {
        field: &'static str,
        message: &'static str,
    },
    InvalidValue {
        field: &'static str,
        value: String,
        message: &'static str,
    },
}

impl ValidationError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingField { .. } => "missing_field",
            Self::InvalidValue { .. } => "invalid_value",
        }
    }

    pub fn message(&self) -> &'static str {
        match self {
            Self::MissingField { message, .. } | Self::InvalidValue { message, .. } => message,
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField { field, message } => write!(formatter, "{field}: {message}"),
            Self::InvalidValue {
                field,
                value,
                message,
            } => write!(formatter, "{field}={value:?}: {message}"),
        }
    }
}

impl std::error::Error for ValidationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceError {
    Validation(ValidationError),
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
            Self::Validation(error) => error.code(),
            Self::BackendUnavailable { .. } => "backend_unavailable",
            Self::InvalidState { .. } => "invalid_state",
            Self::Backend { .. } => "backend",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Validation(error) => error.message(),
            Self::BackendUnavailable { .. } => "backend is unavailable",
            Self::InvalidState { .. } => "backend state does not satisfy the operation",
            Self::Backend { message, .. } => message,
        }
    }
}

impl From<ValidationError> for ServiceError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => error.fmt(formatter),
            Self::BackendUnavailable { backend } => {
                write!(formatter, "backend {backend:?} is unavailable")
            }
            Self::InvalidState { expected, actual } => {
                write!(
                    formatter,
                    "invalid backend state: expected {expected:?}, actual {actual:?}"
                )
            }
            Self::Backend { backend, message } => {
                write!(formatter, "backend {backend:?} failed: {message}")
            }
        }
    }
}

impl std::error::Error for ServiceError {}
