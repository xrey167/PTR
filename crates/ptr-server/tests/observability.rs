//! Opt-in request tracing and ingress diagnostics (default off, byte-identical responses).

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use ptr_config::PtrConfig;
use ptr_runtime::PtrRuntime;
use ptr_server::{
    router, router_from_state, IngressDiagnosticsLimits, IngressDiagnosticsMode,
    RequestTracingOptions, ServerOptions, ServerOptionsError, ServerState,
};
use serde_json::Value;
use std::path::PathBuf;
use tower::ServiceExt;

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).unwrap()
}

fn temp_file(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ptr-server-obs-{tag}-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn post(path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("authorization", "Bearer abcdefghijklmnop1234567890")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

async fn call(app: &Router, body: &str) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(post("/v1/requests", body))
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, bytes.to_vec())
}

fn state_with(options: ServerOptions) -> ServerState {
    ServerState::new(runtime()).with_options(options).unwrap()
}

#[tokio::test]
async fn default_options_leave_the_response_byte_identical() {
    let plain = router(runtime());
    let optioned = router_from_state(state_with(ServerOptions::default()));
    let body = r#"{"id":"r1","text":"please delete the file"}"#;
    let (status_a, bytes_a) = call(&plain, body).await;
    let (status_b, bytes_b) = call(&optioned, body).await;
    assert_eq!(status_a, StatusCode::OK);
    assert_eq!(status_a, status_b);
    assert_eq!(bytes_a, bytes_b);
    let json: Value = serde_json::from_slice(&bytes_a).unwrap();
    assert!(json.get("ingress_diagnostics").is_none());
    assert_eq!(
        String::from_utf8(bytes_a).unwrap(),
        r#"{"id":"r1","revision":1,"text":"please delete the file","verified_evidence":[],"effect":null}"#
    );
}

#[tokio::test]
async fn ingress_diagnostics_are_added_and_leave_the_other_fields_unchanged() {
    let plain = router(runtime());
    let diagnosed = router_from_state(state_with(ServerOptions {
        ingress_diagnostics: IngressDiagnosticsMode::Advisory,
        ..ServerOptions::default()
    }));
    let body = r#"{"id":"r1","text":"please delete the file"}"#;
    let (_, plain_bytes) = call(&plain, body).await;
    let (status, bytes) = call(&diagnosed, body).await;
    assert_eq!(status, StatusCode::OK);
    let mut json: Value = serde_json::from_slice(&bytes).unwrap();
    let diagnostics = json.as_object_mut().unwrap().remove("ingress_diagnostics");
    // Everything else equals the response without the option, and the runtime saw the raw text.
    assert_eq!(json, serde_json::from_slice::<Value>(&plain_bytes).unwrap());
    assert_eq!(json["text"], "please delete the file");
    let diagnostics = diagnostics.expect("field present");
    assert_eq!(diagnostics["advisory"], true);
    assert_eq!(diagnostics["kind"], "text");
    let labels = diagnostics["labels"].as_array().unwrap();
    assert!(
        labels
            .iter()
            .any(|l| l["axis"] == "effect" && l["label"] == "write"),
        "{labels:?}"
    );
    assert_eq!(diagnostics["verification"]["status"], "needs_full_check");
    assert!(diagnostics["verification"]["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r == "high_risk_effect"));
}

#[tokio::test]
async fn text_beyond_the_analysis_cap_is_truncated_in_the_report_only() {
    let diagnosed = router_from_state(state_with(ServerOptions {
        ingress_diagnostics: IngressDiagnosticsMode::Advisory,
        ..ServerOptions::default()
    }));
    // 200 KiB, above the 64 KiB default cap; the high-risk words sit beyond the cap.
    let text = format!("{}rm -rf everything", "a".repeat(200 * 1024));
    let (status, bytes) = call(&diagnosed, &format!(r#"{{"id":"big","text":"{text}"}}"#)).await;
    assert_eq!(status, StatusCode::OK);
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["text"].as_str().unwrap(),
        text,
        "the runtime saw the full text"
    );
    let diagnostics = &json["ingress_diagnostics"];
    assert_eq!(diagnostics["truncated"], true);
    assert_eq!(diagnostics["analyzed_bytes"], 64 * 1024);
    assert_eq!(diagnostics["advisory"], true);
}

#[tokio::test]
async fn the_analysis_cap_is_configurable_and_zero_limits_are_refused() {
    let state = state_with(ServerOptions {
        ingress_diagnostics: IngressDiagnosticsMode::Advisory,
        ..ServerOptions::default()
    })
    .with_ingress_diagnostics_limits(IngressDiagnosticsLimits {
        max_concurrent: 1,
        max_analyzed_bytes: 8,
    })
    .unwrap();
    let (status, bytes) = call(
        &router_from_state(state),
        r#"{"id":"r","text":"hello world, this is long"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["ingress_diagnostics"]["analyzed_bytes"], 8);
    assert_eq!(json["ingress_diagnostics"]["truncated"], true);
    for limits in [
        IngressDiagnosticsLimits {
            max_concurrent: 0,
            ..IngressDiagnosticsLimits::default()
        },
        IngressDiagnosticsLimits {
            max_analyzed_bytes: 0,
            ..IngressDiagnosticsLimits::default()
        },
    ] {
        let refused = ServerState::new(runtime())
            .with_options(ServerOptions {
                ingress_diagnostics: IngressDiagnosticsMode::Advisory,
                ..ServerOptions::default()
            })
            .unwrap()
            .with_ingress_diagnostics_limits(limits);
        assert!(matches!(
            refused,
            Err(ServerOptionsError::ZeroIngressConcurrency
                | ServerOptionsError::ZeroIngressAnalysisCap)
        ));
    }
}

#[tokio::test]
async fn a_high_risk_label_is_a_diagnostic_and_never_blocks_the_request() {
    let diagnosed = router_from_state(state_with(ServerOptions {
        ingress_diagnostics: IngressDiagnosticsMode::Advisory,
        ..ServerOptions::default()
    }));
    let text = "rm -rf everything and exec the network upload";
    let (status, bytes) = call(&diagnosed, &format!(r#"{{"id":"r2","text":"{text}"}}"#)).await;
    assert_eq!(status, StatusCode::OK);
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["text"], text,
        "the runtime receives the raw text unchanged"
    );
    assert_eq!(json["effect"], Value::Null);
    assert_eq!(
        json["ingress_diagnostics"]["verification"]["status"],
        "needs_full_check"
    );
}

#[tokio::test]
async fn a_refused_request_gets_no_diagnostics() {
    let diagnosed = router_from_state(state_with(ServerOptions {
        ingress_diagnostics: IngressDiagnosticsMode::Advisory,
        ..ServerOptions::default()
    }));
    let plain = router(runtime());
    let body = r#"{"id":"","text":""}"#;
    let (status_a, bytes_a) = call(&plain, body).await;
    let (status_b, bytes_b) = call(&diagnosed, body).await;
    assert_eq!(status_a, StatusCode::BAD_REQUEST);
    assert_eq!((status_a, bytes_a), (status_b, bytes_b));
}

#[tokio::test]
async fn tracing_exports_the_fields_and_never_a_secret_from_body_or_headers() {
    let path = temp_file("redact");
    let app = router_from_state(state_with(ServerOptions {
        request_tracing: Some(RequestTracingOptions::new(&path)),
        ..ServerOptions::default()
    }));
    // The request id itself carries secrets too; it must not reach the file either.
    let body = r#"{"id":"password=hunter2","text":"password=hunter2 postgres://user:pass@host/db Authorization: Bearer abcdefghijklmnop1234567890"}"#;
    let (status, _) = call(&app, body).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = call(&app, r#"{"id":"ok-1","text":"hello"}"#).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = call(&app, r#"{"id":"","text":""}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let exported = std::fs::read_to_string(&path).unwrap();
    for secret in [
        "hunter2",
        "user:pass",
        "abcdefghijklmnop",
        "postgres://",
        "Bearer",
    ] {
        assert!(!exported.contains(secret), "{secret} leaked: {exported}");
    }
    let lines: Vec<Value> = exported
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    let span = |index: usize| lines[index]["resourceSpans"][0]["scopeSpans"][0]["spans"][0].clone();
    let attribute = |span: &Value, key: &str| -> Value {
        span["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["key"] == key)
            .unwrap_or_else(|| panic!("no attribute {key}"))["value"]
            .clone()
    };
    let ok = span(1);
    // Every client id is hashed, even a short plain one.
    let ok_id = attribute(&ok, "ptr.request_id")["stringValue"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        ok_id.starts_with("sha256:") && !ok_id.contains("ok-1"),
        "{ok_id}"
    );
    assert_eq!(attribute(&ok, "http.route")["stringValue"], "/v1/requests");
    assert_eq!(attribute(&ok, "http.status")["intValue"], "200");
    assert_eq!(attribute(&ok, "ptr.outcome")["stringValue"], "ok");
    assert!(attribute(&ok, "ptr.latency_ns")["intValue"].is_string());
    assert_eq!(ok["status"]["code"], 1);
    // Only the five documented fields exist.
    assert_eq!(ok["attributes"].as_array().unwrap().len(), 5);
    let bad = span(2);
    assert_eq!(attribute(&bad, "http.status")["intValue"], "400");
    assert_eq!(
        attribute(&bad, "ptr.outcome")["stringValue"],
        "client_error"
    );
    // The unsafe id was replaced by a digest prefix.
    let first = span(0);
    assert!(attribute(&first, "ptr.request_id")["stringValue"]
        .as_str()
        .unwrap()
        .starts_with("sha256:"));
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn a_short_client_request_id_never_reaches_the_file_but_stays_correlatable() {
    let path = temp_file("idhash");
    let app = router_from_state(state_with(ServerOptions {
        request_tracing: Some(RequestTracingOptions::new(&path)),
        ..ServerOptions::default()
    }));
    // Short, alphanumeric and low entropy: a credential the redactor cannot recognise.
    for id in ["hunter2", "hunter2", "other-id"] {
        let (status, _) = call(&app, &format!(r#"{{"id":"{id}","text":"hello"}}"#)).await;
        assert_eq!(status, StatusCode::OK);
    }
    let exported = std::fs::read_to_string(&path).unwrap();
    assert!(!exported.contains("hunter2"), "raw id leaked: {exported}");
    assert!(!exported.contains("other-id"), "raw id leaked: {exported}");
    let ids: Vec<String> = exported
        .lines()
        .map(|line| {
            let line: Value = serde_json::from_str(line).unwrap();
            line["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["key"] == "ptr.request_id")
                .unwrap()["value"]["stringValue"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(ids.len(), 3);
    assert!(ids
        .iter()
        .all(|id| id.starts_with("sha256:") && id.len() == 7 + 16));
    assert_eq!(ids[0], ids[1], "the same id maps to the same digest");
    assert_ne!(ids[0], ids[2]);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn the_export_file_is_capped_and_drops_are_counted() {
    let path = temp_file("cap");
    let state = state_with(ServerOptions {
        request_tracing: Some(RequestTracingOptions {
            export_path: path.clone(),
            max_export_bytes: 1500,
        }),
        ..ServerOptions::default()
    });
    let app = router_from_state(state.clone());
    for index in 0..10 {
        let (status, _) = call(&app, &format!(r#"{{"id":"r{index}","text":"hello"}}"#)).await;
        assert_eq!(status, StatusCode::OK, "tracing never changes the response");
    }
    let size = std::fs::metadata(&path).unwrap().len();
    assert!(size > 0 && size <= 1500, "{size}");
    assert!(state.dropped_trace_exports() > 0);
    assert_eq!(state.trace_export_errors(), 0);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn tracing_covers_health_and_does_not_change_it() {
    let path = temp_file("health");
    let app = router_from_state(state_with(ServerOptions {
        request_tracing: Some(RequestTracingOptions::new(&path)),
        ..ServerOptions::default()
    }));
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
    let exported = std::fs::read_to_string(&path).unwrap();
    assert!(exported.contains("\"/health\""));
    assert!(exported.contains("\"none\""), "no request id on /health");
    let _ = std::fs::remove_file(path);
}

#[test]
fn unusable_tracing_options_are_typed_errors() {
    let zero = ServerState::new(runtime()).with_options(ServerOptions {
        request_tracing: Some(RequestTracingOptions {
            export_path: temp_file("zero"),
            max_export_bytes: 0,
        }),
        ..ServerOptions::default()
    });
    assert!(matches!(zero, Err(ServerOptionsError::ZeroTraceExportCap)));
    for max_export_bytes in [u64::MAX, ptr_server::MAX_TRACE_EXPORT_BYTES + 1] {
        let path = temp_file("huge");
        let too_large = ServerState::new(runtime()).with_options(ServerOptions {
            request_tracing: Some(RequestTracingOptions {
                export_path: path.clone(),
                max_export_bytes,
            }),
            ..ServerOptions::default()
        });
        assert!(matches!(
            too_large,
            Err(ServerOptionsError::TraceExportCapTooLarge { max })
                if max == ptr_server::MAX_TRACE_EXPORT_BYTES
        ));
        let _ = std::fs::remove_file(path);
    }
    let at_max = temp_file("at-max");
    assert!(ServerState::new(runtime())
        .with_options(ServerOptions {
            request_tracing: Some(RequestTracingOptions {
                export_path: at_max.clone(),
                max_export_bytes: ptr_server::MAX_TRACE_EXPORT_BYTES,
            }),
            ..ServerOptions::default()
        })
        .is_ok());
    let _ = std::fs::remove_file(at_max);
    let missing = ServerState::new(runtime()).with_options(ServerOptions {
        request_tracing: Some(RequestTracingOptions::new(
            std::env::temp_dir().join("ptr-server-no-such-dir-xyz/trace.jsonl"),
        )),
        ..ServerOptions::default()
    });
    assert!(matches!(
        missing,
        Err(ServerOptionsError::TraceExportUnavailable { .. })
    ));
}
