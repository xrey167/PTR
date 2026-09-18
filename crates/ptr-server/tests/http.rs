use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
};
use ptr_config::PtrConfig;
use ptr_runtime::PtrRuntime;
use ptr_server::router;
use serde_json::Value;
use tower::ServiceExt;

#[tokio::test]
async fn health_contract_matches_sdk() {
    let app = router(PtrRuntime::new(PtrConfig::default()).unwrap());
    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn request_contract_returns_revisioned_response() {
    let app = router(PtrRuntime::new(PtrConfig::default()).unwrap());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"r1","text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], "r1");
    assert_eq!(json["revision"], 1);
    assert_eq!(json["text"], "hello");
}

#[tokio::test]
async fn malformed_request_is_bad_request() {
    let app = router(PtrRuntime::new(PtrConfig::default()).unwrap());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"","text":""}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
