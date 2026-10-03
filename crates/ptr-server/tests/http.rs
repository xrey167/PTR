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
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
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

#[tokio::test]
async fn oversized_id_text_and_body_are_refused_before_the_runtime_sees_them() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let app =
        ptr_server::router(ptr_runtime::PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap());
    let post = |body: String| {
        Request::post("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    };
    let long_id = "i".repeat(ptr_server::MAX_REQUEST_ID_BYTES + 1);
    let long_text = "t".repeat(ptr_server::MAX_REQUEST_TEXT_BYTES + 1);
    let huge = "x".repeat(ptr_server::MAX_REQUEST_BODY_BYTES + 1);
    let cases = [
        (
            format!(r#"{{"id":"{long_id}","text":"hello"}}"#),
            StatusCode::BAD_REQUEST,
        ),
        (
            format!(r#"{{"id":"r1","text":"{long_text}"}}"#),
            StatusCode::BAD_REQUEST,
        ),
        (
            format!(r#"{{"id":"r1","text":"{huge}"}}"#),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
    ];
    for (body, expected) in cases {
        let response = app.clone().oneshot(post(body)).await.unwrap();
        assert_eq!(response.status(), expected);
    }
}
