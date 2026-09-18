use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ptr_model_api::{ModelEvent, ReferenceEchoBackend};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::RequestId;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiRequest {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ApiResponse {
    pub id: String,
    pub revision: u64,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Clone)]
pub struct ServerState {
    runtime: Arc<Mutex<PtrRuntime>>,
}

impl ServerState {
    pub fn new(runtime: PtrRuntime) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
        }
    }
}

pub fn router(runtime: PtrRuntime) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/requests", post(request))
        .with_state(ServerState::new(runtime))
}

pub async fn serve(listener: tokio::net::TcpListener, runtime: PtrRuntime) -> std::io::Result<()> {
    axum::serve(listener, router(runtime)).await
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

async fn request(
    State(state): State<ServerState>,
    Json(input): Json<ApiRequest>,
) -> Result<Json<ApiResponse>, ApiError> {
    if input.id.trim().is_empty() || input.text.trim().is_empty() {
        return Err(ApiError::BadRequest("id and text must be non-empty".into()));
    }

    let mut runtime = state
        .runtime
        .lock()
        .map_err(|_| ApiError::Internal("runtime mutex poisoned".into()))?;
    let events = runtime
        .run_model_once(
            RequestId::from(input.id.as_str()),
            input.text,
            &ReferenceEchoBackend,
        )
        .map_err(ApiError::Runtime)?;

    let text = events
        .into_iter()
        .filter_map(|event| match event {
            ModelEvent::Token(token) => Some(token),
            _ => None,
        })
        .collect::<String>();

    Ok(Json(ApiResponse {
        id: input.id,
        revision: runtime.revision().0,
        text,
    }))
}

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    Runtime(RuntimeError),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::Runtime(RuntimeError::StaleRevision { .. })
            | Self::Runtime(RuntimeError::StaleGeneration { .. }) => {
                (StatusCode::CONFLICT, format!("{self:?}"))
            }
            Self::Runtime(error) => (StatusCode::UNPROCESSABLE_ENTITY, format!("{error:?}")),
            Self::Internal(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
