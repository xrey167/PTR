use crate::{ExecuteRequestRef, ValidationError};

const MAX_BACKEND_NAME_LEN: usize = 64;
const MAX_REQUEST_KEY_LEN: usize = 128;

pub fn check_backend_name<'request>(
    backend: Option<&'request str>,
) -> Result<&'request str, ValidationError> {
    match backend {
        None => Err(ValidationError::MissingField {
            field: "preferred_backend",
            message: "preferred backend must be selected before execution",
        }),
        Some(name) if name.trim().is_empty() => Err(ValidationError::InvalidValue {
            field: "preferred_backend",
            value: name.to_owned(),
            message: "preferred backend must not be empty",
        }),
        Some(name) if name.len() > MAX_BACKEND_NAME_LEN => {
            Err(ValidationError::InvalidValue {
                field: "preferred_backend",
                value: name.to_owned(),
                message: "preferred backend name exceeds the supported length",
            })
        }
        Some(name) => Ok(name),
    }
}

pub fn check_request_key(key: &str) -> Result<(), ValidationError> {
    match key {
        value if value.trim().is_empty() => Err(ValidationError::MissingField {
            field: "key",
            message: "request key must not be empty",
        }),
        value if value.len() > MAX_REQUEST_KEY_LEN => Err(ValidationError::InvalidValue {
            field: "key",
            value: value.to_owned(),
            message: "request key exceeds the supported length",
        }),
        _ => Ok(()),
    }
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
    fn missing_backend_name_matches_missing_field_arm() {
        assert_eq!(
            check_backend_name(None),
            Err(ValidationError::MissingField {
                field: "preferred_backend",
                message: "preferred backend must be selected before execution",
            })
        );
    }

    #[test]
    fn empty_backend_name_matches_guarded_invalid_value_arm() {
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
    fn oversized_backend_name_matches_length_guard() {
        let value = "x".repeat(MAX_BACKEND_NAME_LEN + 1);
        assert_eq!(
            check_backend_name(Some(&value)),
            Err(ValidationError::InvalidValue {
                field: "preferred_backend",
                value: value.clone(),
                message: "preferred backend name exceeds the supported length",
            })
        );
    }


    #[test]
    fn empty_guard_precedes_length_guard_for_whitespace_only_backend() {
        let value = " ".repeat(MAX_BACKEND_NAME_LEN + 1);
        assert_eq!(
            check_backend_name(Some(&value)),
            Err(ValidationError::InvalidValue {
                field: "preferred_backend",
                value: value.clone(),
                message: "preferred backend must not be empty",
            })
        );
    }

    #[test]
    fn empty_request_key_matches_guarded_missing_field_arm() {
        assert_eq!(
            check_request_key(""),
            Err(ValidationError::MissingField {
                field: "key",
                message: "request key must not be empty",
            })
        );
    }

    #[test]
    fn oversized_request_key_matches_length_guard() {
        let value = "k".repeat(MAX_REQUEST_KEY_LEN + 1);
        assert_eq!(
            check_request_key(&value),
            Err(ValidationError::InvalidValue {
                field: "key",
                value,
                message: "request key exceeds the supported length",
            })
        );
    }
}
