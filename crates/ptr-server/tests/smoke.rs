use ptr_server::ApiRequest;

#[test]
fn api_request_keeps_id() {
    let request = ApiRequest {
        id: "r".into(),
        text: "hello".into(),
        idempotency_key: None,
    };
    assert_eq!(request.id, "r");
}

#[tokio::test]
async fn http_request_runs_against_injected_backend_and_returns_revision() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use ptr_config::PtrConfig;
    use ptr_runtime::PtrRuntime;
    use tower::ServiceExt;

    let app = ptr_server::router(PtrRuntime::new(PtrConfig::default()).unwrap());
    let response = app
        .oneshot(
            Request::post("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"http-1","text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["id"], "http-1");
    assert_eq!(json["revision"], 1);
    assert_eq!(json["text"], "hello");
}

#[tokio::test]
async fn scenario_backend_executes_demo_note_once_and_replays_receipt() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use ptr_config::PtrConfig;
    use ptr_ledger::LedgerEvent;
    use ptr_model_api::ScenarioBackend;
    use ptr_runtime::PtrRuntime;
    use ptr_types::{CapabilityId, CapsuleId, Generation};
    use std::sync::Arc;
    use tower::ServiceExt;

    let data_dir = std::env::temp_dir().join(format!(
        "ptr-demo-note-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "demo".into(),
            capsule: CapsuleId::from("demo-note"),
            generation: Generation(1),
        })
        .unwrap();
    runtime.permissions_mut().allow_mutation = true;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    let app = ptr_server::router_with_dependencies(
        runtime,
        Arc::new(ScenarioBackend),
        Default::default(),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        ptr_server::demo_effect_grants(&data_dir),
    );
    let request = || {
        Request::post("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"id":"effect-1","text":"hello","idempotency_key":"key-1"}"#,
            ))
            .unwrap()
    };

    let first = app.clone().oneshot(request()).await.unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let first_bytes = axum::body::to_bytes(first.into_body(), usize::MAX)
        .await
        .unwrap();
    let first_json: serde_json::Value = serde_json::from_slice(&first_bytes).unwrap();
    assert_eq!(first_json["effect"]["attempt"], 3);
    assert_eq!(first_json["effect"]["settlement"], 4);
    assert_eq!(
        std::fs::read(data_dir.join("effects/demo-note.txt")).unwrap(),
        b"hello"
    );

    let missing_key = app
        .clone()
        .oneshot(
            Request::post("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"id":"effect-3","text":"no key"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_key.status(), StatusCode::BAD_REQUEST);

    let second = app.clone().oneshot(request()).await.unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    let second_bytes = axum::body::to_bytes(second.into_body(), usize::MAX)
        .await
        .unwrap();
    let second_json: serde_json::Value = serde_json::from_slice(&second_bytes).unwrap();
    assert_eq!(second_json["effect"]["attempt"], 3);
    assert_eq!(second_json["effect"]["settlement"], 4);

    let conflict = app
        .oneshot(
            Request::post("/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"id":"effect-2","text":"different","idempotency_key":"key-1"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn durable_restart_fences_open_effect_until_explicit_reconciliation() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use ptr_config::PtrConfig;
    use ptr_ledger::LedgerEvent;
    use ptr_model_api::ScenarioBackend;
    use ptr_runtime::PtrRuntime;
    use ptr_types::{
        CapabilityId, CapsuleId, Effect, Generation, ProjectId, Revision, VerificationLevel,
    };
    use std::sync::Arc;
    use tower::ServiceExt;

    let root = std::env::temp_dir().join(format!(
        "ptr-effect-restart-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let log = root.join("ledger.log");
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &log).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "demo".into(),
            capsule: CapsuleId::from("demo-note"),
            generation: Generation(1),
        })
        .unwrap();
    runtime.permissions_mut().allow_mutation = true;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    let attempt = runtime
        .commit(LedgerEvent::EffectAttempted {
            key: Some("crash-key".into()),
            project: ProjectId::from("demo"),
            principal: "http".into(),
            target: "demo-note".into(),
            operation: "create".into(),
            capability: CapabilityId::from("demo.local-note.create"),
            effect: Effect::Mutation,
            generation: Generation(1),
            revision: Revision(0),
            verification: VerificationLevel::Deterministic,
            action_digest: [0; 32],
        })
        .unwrap();
    drop(runtime);

    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &log).unwrap();
    assert_eq!(reopened.unsettled_effects()[0].attempt, attempt);
    let app = ptr_server::router_with_dependencies(
        reopened,
        Arc::new(ScenarioBackend),
        Default::default(),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        ptr_server::demo_effect_grants(&root),
    );
    let request = || {
        Request::post("/v1/requests")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"id":"restart-1","text":"after restart","idempotency_key":"crash-key"}"#,
            ))
            .unwrap()
    };
    let fenced = app.clone().oneshot(request()).await.unwrap();
    let fenced_status = fenced.status();
    let fenced_body = axum::body::to_bytes(fenced.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        fenced_status,
        StatusCode::CONFLICT,
        "{}",
        String::from_utf8_lossy(&fenced_body)
    );
    drop(app);

    let mut reconciled = PtrRuntime::open_durable(PtrConfig::default(), &log).unwrap();
    reconciled
        .reconcile_effect(attempt, false, "crash window inspected; no file present")
        .unwrap();
    reconciled.permissions_mut().allow_mutation = true;
    reconciled
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    let app = ptr_server::router_with_dependencies(
        reconciled,
        Arc::new(ScenarioBackend),
        Default::default(),
        Arc::new(Default::default()),
        ptr_server::default_verifier(),
        ptr_server::demo_effect_grants(&root),
    );
    let settled = app.oneshot(request()).await.unwrap();
    let settled_status = settled.status();
    let settled_body = axum::body::to_bytes(settled.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        settled_status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&settled_body)
    );
    assert_eq!(
        std::fs::read(root.join("effects/demo-note.txt")).unwrap(),
        b"after restart"
    );
    let _ = std::fs::remove_dir_all(root);
}
