use std::sync::Arc;

use ptr_rust_crate_template::{ExecuteRequest, ReferenceBackend, Service};

pub fn reference_service() -> Service<String, String> {
    let mut service = Service::default();
    service.register(Arc::new(ReferenceBackend));
    service
}

pub fn valid_request(input: &str) -> ExecuteRequest<String> {
    ExecuteRequest {
        key: "request-1".into(),
        input: input.into(),
        preferred_backend: Some("reference".into()),
    }
}
