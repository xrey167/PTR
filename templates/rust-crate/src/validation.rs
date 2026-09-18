use crate::{ExecuteRequestRef, ValidationError};

pub fn check_backend_name<'request>(
    backend: Option<&'request str>,
) -> Result<&'request str, ValidationError> {
    let backend = backend.ok_or(ValidationError::MissingField {
        field: "preferred_backend",
        message: "preferred backend must be selected before execution",
    })?;

    if backend.trim().is_empty() {
        return Err(ValidationError::InvalidValue {
            field: "preferred_backend",
            value: backend.to_owned(),
            message: "preferred backend must not be empty",
        });
    }

    Ok(backend)
}

pub fn check_request_key(key: &str) -> Result<(), ValidationError> {
    if key.trim().is_empty() {
        return Err(ValidationError::MissingField {
            field: "key",
            message: "request key must not be empty",
        });
    }
    Ok(())
}

pub fn validate_request<T>(request: ExecuteRequestRef<'_, T>) -> Result<(), ValidationError> {
    check_request_key(request.key)?;
    check_backend_name(request.preferred_backend)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_backend_name_returns_named_invalid_value() {
        assert_eq!(
            check_backend_name(Some("   ")),
            Err(ValidationError::InvalidValue {
                field: "preferred_backend",
                value: "   ".into(),
                message: "preferred backend must not be empty",
            })
        );
    }

    #[test]
    fn empty_request_key_returns_named_missing_field() {
        assert_eq!(
            check_request_key(""),
            Err(ValidationError::MissingField {
                field: "key",
                message: "request key must not be empty",
            })
        );
    }
}
