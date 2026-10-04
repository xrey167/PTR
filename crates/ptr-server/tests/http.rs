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

#[tokio::test]
async fn a_shared_runtime_is_the_one_the_router_writes_to() {
    // `ptrd` hands the same runtime to the HTTP API and to its execution wire. The
    // property that makes that safe is that there is one runtime: a request through
    // the router is visible to the other holder, with nothing reopened.
    let shared = std::sync::Arc::new(std::sync::Mutex::new(
        PtrRuntime::new(PtrConfig::default()).unwrap(),
    ));
    let before = shared.lock().unwrap().committed_events().len();
    let app = ptr_server::router_with_shared_runtime(
        std::sync::Arc::clone(&shared),
        std::sync::Arc::new(ptr_model_api::ReferenceEchoBackend),
        ptr_router::PodRouter,
        std::sync::Arc::new(ptr_pods::PodRegistry::default()),
        ptr_server::default_verifier(),
        std::sync::Arc::new(Vec::new),
    );
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
    assert!(
        shared.lock().unwrap().committed_events().len() > before,
        "the write landed in the runtime the other holder has"
    );
}

#[tokio::test]
async fn serve_until_answers_until_the_shutdown_and_then_returns() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(ptr_server::serve_until(
        listener,
        router(PtrRuntime::new(PtrConfig::default()).unwrap()),
        async move {
            let _ = stopped.await;
        },
    ));

    // The control: before the shutdown it answers.
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"GET /health HTTP/1.1\r\nhost: ptr\r\nconnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    stop.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), server)
        .await
        .expect("the server returns once told to stop")
        .unwrap()
        .unwrap();
    assert!(
        tokio::net::TcpStream::connect(address).await.is_err(),
        "after it returns, nothing is listening"
    );
}
