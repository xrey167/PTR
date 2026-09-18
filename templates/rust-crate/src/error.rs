#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceError {
    BackendUnavailable { backend: String },
    Backend(String),
}
