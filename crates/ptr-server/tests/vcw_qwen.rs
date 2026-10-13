//! The `qwen-cpu` backend through the server. Ignored by default: it needs a model directory in
//! `PTR_QWEN_MODEL_DIR` (with the pinned `SHA256SUMS` file next to the weights) and the
//! `qwen-cpu` feature, and takes minutes on a CPU.
//!
//! ```text
//! PTR_QWEN_MODEL_DIR=<dir> cargo test -p ptr-server --release --features qwen-cpu \
//!     --test vcw_qwen -- --ignored --nocapture
//! ```
//!
//! The assertions are the ones that do not depend on the model's skill: a `200` answer is the
//! verifier's (it parses to the exact result) and is in the ledger, any other outcome is `422`
//! with no candidate, and a digest that does not match refuses to load the model.
#![cfg(feature = "qwen-cpu")]

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use ptr_config::{PtrConfig, VcwBackend, VcwServiceConfig, VcwTaskKind};
use ptr_ledger::LedgerEvent;
use ptr_runtime::PtrRuntime;
use ptr_server::{router_from_state, ServerState, VcwBuildError, VcwService, VcwServiceParts};
use serde_json::{json, Value};
use tower::ServiceExt;

fn model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("PTR_QWEN_MODEL_DIR").expect("PTR_QWEN_MODEL_DIR"))
}

/// The digest of `file` from the `SHA256SUMS` next to the weights.
fn pinned(dir: &std::path::Path, file: &str) -> String {
    std::fs::read_to_string(dir.join("SHA256SUMS"))
        .expect("SHA256SUMS")
        .lines()
        .find_map(|line| {
            let (hash, name) = line.split_once("  ")?;
            (name == file).then(|| hash.to_owned())
        })
        .unwrap_or_else(|| panic!("{file} in SHA256SUMS"))
}

fn config(dir: &std::path::Path) -> VcwServiceConfig {
    let mut config = VcwServiceConfig {
        enabled: true,
        kinds: vec![VcwTaskKind::Arithmetic],
        ..VcwServiceConfig::default()
    };
    config.experts.backend = Some(VcwBackend::QwenCpu);
    let qwen = &mut config.experts.qwen_cpu;
    qwen.model_dir = dir.to_string_lossy().into_owned();
    qwen.weights_sha256 = pinned(dir, "model.safetensors");
    qwen.config_sha256 = pinned(dir, "config.json");
    qwen.tokenizer_sha256 = pinned(dir, "tokenizer.json");
    qwen.tokenizer_config_sha256 = pinned(dir, "tokenizer_config.json");
    config
}

#[test]
#[ignore = "needs PTR_QWEN_MODEL_DIR"]
fn a_digest_that_does_not_match_refuses_to_load_the_model() {
    let dir = model_dir();
    let mut config = config(&dir);
    config.experts.qwen_cpu.config_sha256 = "0".repeat(64);
    let result = VcwService::from_config(&config, &VcwServiceParts::new([1; 32], [2; 8]));
    assert!(matches!(result, Err(VcwBuildError::Qwen(_))), "{result:?}");
}

#[tokio::test]
#[ignore = "needs PTR_QWEN_MODEL_DIR and takes minutes"]
async fn a_real_model_answers_through_the_loop_and_every_answer_is_the_verifiers() {
    let dir = model_dir();
    let config = config(&dir);
    let service = VcwService::from_config(&config, &VcwServiceParts::new([5; 32], [6; 8]))
        .expect("the model loads with the pinned digests");
    let state = ServerState::new(PtrRuntime::new(PtrConfig::default()).unwrap())
        .with_vcw(service)
        .unwrap();
    let app = router_from_state(state.clone());

    let (mut verified, mut unanswered) = (0, 0);
    for (index, (expression, expected)) in [("17 + 25", 42), ("12 * 7", 84), ("91 - 36", 55)]
        .into_iter()
        .enumerate()
    {
        let id = format!("qwen-{index}");
        let body = json!({"id": id, "task": {"kind": "arithmetic", "expression": expression}});
        let request = Request::post("/v1/vcw/tasks")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        eprintln!("{expression} -> {status} {body}");
        match status {
            StatusCode::OK => {
                verified += 1;
                assert_eq!(body["outcome"], "verified");
                assert_eq!(
                    body["answer"]
                        .as_str()
                        .unwrap()
                        .trim()
                        .parse::<i64>()
                        .unwrap(),
                    expected
                );
                let handle = state.runtime_handle();
                let runtime = handle.lock().unwrap();
                assert!(runtime.committed_events().iter().any(|e| matches!(
                    &e.event, LedgerEvent::PodOutputAdmitted { request_id, .. } if request_id.0 == id
                )));
            }
            StatusCode::UNPROCESSABLE_ENTITY => {
                unanswered += 1;
                assert!(body.get("answer").is_none(), "{body}");
            }
            other => panic!("unexpected status {other}: {body}"),
        }
    }
    let calls = state.vcw_service().unwrap().expert_calls();
    assert!(
        calls.iter().all(|c| c.model.is_some()),
        "every call names its model files"
    );
    eprintln!(
        "verified {verified}, unanswered {unanswered}, expert calls {}",
        calls.len()
    );
}
