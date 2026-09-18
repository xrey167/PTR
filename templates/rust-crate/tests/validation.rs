mod common;

use ptr_rust_crate_template::{ServiceError, ValidationError};

#[test]
fn missing_backend_returns_named_validation_error() {
    let service = common::reference_service();
    let mut request = common::valid_request("hello");
    request.preferred_backend = None;

    assert_eq!(
        service.execute(request),
        Err(ServiceError::Validation(ValidationError::MissingField {
            field: "preferred_backend",
            message: "preferred backend must be selected before execution",
        }))
    );
}

#[test]
fn empty_request_key_returns_named_validation_error() {
    let service = common::reference_service();
    let mut request = common::valid_request("hello");
    request.key.clear();

    assert_eq!(
        service.execute(request),
        Err(ServiceError::Validation(ValidationError::MissingField {
            field: "key",
            message: "request key must not be empty",
        }))
    );
}
