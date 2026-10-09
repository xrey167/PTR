//! Public-API integration test: a bench workspace's whole path through the kit
//! (generate, time, summarise, capture, write evidence, read it back).

use evalkit::clock::Utc;
use evalkit::envcap::capture;
use evalkit::evidence::{
    BenchSection, EvidenceRecord, GateResult, EXTENSION_FIELDS, RECORD_FIELDS,
};
use evalkit::gen::keys;
use evalkit::json::Value;
use evalkit::stats::{sample_latencies, Repetitions, Summary};
use evalkit::{rss, sha256};

#[test]
fn a_benchmark_writes_a_record_the_checker_can_read() {
    let data = keys(17, 64, 16);
    let samples = sample_latencies(10_000, |i| {
        std::hint::black_box(data[i % data.len()].len());
    });
    let summary = Summary::latency(&samples).expect("10k samples");
    assert!(summary.p50 <= summary.p95 && summary.p95 <= summary.p99 && summary.p99 <= summary.max);

    let medians = Repetitions::DEFAULT.run(|i| i as f64);
    assert_eq!(medians.len(), 5);

    let environment = capture(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
    let started = Utc::now();
    let record = EvidenceRecord {
        component: "allocator".into(),
        candidate: Value::obj([("id", Value::str("kit-self-test"))]),
        criteria: vec![("latency".into(), 1.0)],
        git_sha: environment.git_sha.clone(),
        platform: format!("Linux-{}", environment.kernel),
        candidate_manifest_sha256: None,
        cargo_lock_sha256: sha256::hex_of_file(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
        ),
        started_at: started.utc_stamp(),
        finished_at: Utc::now().isoformat(),
        command: vec!["kit-self-test".into()],
        exit_code: Some(0),
        stdout: String::new(),
        stderr: String::new(),
        launch_error: None,
        duration_ns: 1,
        protocol_sha256: sha256::hex(b"protocol text"),
        bench: BenchSection {
            candidate_version: "0.1.0".into(),
            gates: vec![GateResult {
                id: "g".into(),
                passed: true,
            }],
            benchmark_ran: true,
            warmup: 1,
            measured: 5,
            decision_grade: environment.durable_writes_decision_grade(),
            decision_note: String::new(),
            metrics: vec![
                ("p99_ns".into(), summary.p99 as f64),
                (
                    "peak_rss_kib".into(),
                    rss::peak_rss_kib().unwrap_or(0) as f64,
                ),
            ],
            proxies: Value::Null,
            environment: environment.to_value(),
        },
    };

    let dir = std::env::temp_dir().join(format!("evalkit-it-{}", std::process::id()));
    let path = record.write_exclusive(&dir).expect("write");
    let parsed = Value::parse(&std::fs::read_to_string(&path).unwrap()).expect("parse");
    for field in RECORD_FIELDS.iter().chain(EXTENSION_FIELDS.iter()) {
        assert!(parsed.get(field).is_some(), "missing {field}");
    }
    assert_eq!(parsed, record.to_value());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn latency_summaries_refuse_small_samples_through_the_public_api() {
    let few = vec![1u64; 9_999];
    assert!(Summary::latency(&few).is_err());
    assert!(Repetitions::new(1, 4).is_err());
}
