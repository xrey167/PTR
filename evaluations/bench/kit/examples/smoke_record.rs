//! Template for a batch benchmark's `main`: measure something trivial with the
//! kit and write a protocol-conformant evidence record.
//!
//! cargo run --manifest-path evaluations/bench/kit/Cargo.toml --locked \
//!   --example smoke_record -- COMPONENT CANDIDATE PROTOCOL_TOML OUT_DIR [GATE...]
//!
//! Set EVALKIT_HARDWARE_PROFILE to the protocol's hardware profile file; without it
//! the record is not decision grade.
//!
//! Each GATE is `id` (passed) or `!id` (failed). The record is a smoke test of
//! the plumbing; its metric is a timing of a no-op, not a comparison.

use evalkit::clock::Utc;
use evalkit::envcap::capture;
use evalkit::evidence::{BenchSection, EvidenceRecord, GateResult};
use evalkit::gen::keys;
use evalkit::json::Value;
use evalkit::stats::{sample_latencies, Repetitions, Summary};
use evalkit::{rss, sha256};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [component, candidate, protocol, out, gates @ ..] = args.as_slice() else {
        eprintln!("usage: smoke_record COMPONENT CANDIDATE PROTOCOL_TOML OUT_DIR [GATE...]");
        std::process::exit(2);
    };
    let started = Utc::now();
    let environment = capture(Path::new("."));
    let data = keys(17, 64, 16);
    let reps = Repetitions::DEFAULT;
    let medians: Vec<f64> = reps.run(|_| {
        let samples = sample_latencies(10_000, |i| {
            std::hint::black_box(data[i % data.len()].len());
        });
        Summary::latency(&samples).map_or(f64::NAN, |s| s.p99 as f64)
    });
    let record = EvidenceRecord {
        component: component.clone(),
        candidate: Value::obj([("id", Value::str(candidate))]),
        criteria: vec![("latency".into(), 1.0)],
        git_sha: environment.git_sha.clone(),
        platform: format!("Linux-{}", environment.kernel),
        candidate_manifest_sha256: None,
        cargo_lock_sha256: None,
        started_at: started.utc_stamp(),
        finished_at: Utc::now().isoformat(),
        command: std::env::args().collect(),
        exit_code: Some(0),
        stdout: String::new(),
        stderr: String::new(),
        launch_error: None,
        duration_ns: 0,
        protocol_sha256: sha256::hex_of_file(Path::new(protocol)).unwrap_or_default(),
        bench: BenchSection {
            candidate_version: "smoke".into(),
            gates: gates
                .iter()
                .map(|g| GateResult {
                    id: g.trim_start_matches('!').into(),
                    passed: !g.starts_with('!'),
                })
                .collect(),
            benchmark_ran: true,
            warmup: reps.warmup as u32,
            measured: reps.measured as u32,
            decision_grade: environment.decision_grade(
                &std::env::var("EVALKIT_HARDWARE_PROFILE")
                    .ok()
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .unwrap_or_default(),
            ),
            decision_note: "plumbing smoke test".into(),
            metrics: vec![
                (
                    "noop_p99_ns".into(),
                    evalkit::stats::median(&medians).unwrap_or(f64::NAN),
                ),
                (
                    "peak_rss_kib".into(),
                    rss::peak_rss_kib().unwrap_or(0) as f64,
                ),
            ],
            proxies: Value::Null,
            environment: environment.to_value(),
        },
    };
    match record.write_exclusive(Path::new(out)) {
        Ok(path) => println!("{}", path.display()),
        Err(error) => {
            eprintln!("cannot write evidence: {error}");
            std::process::exit(1);
        }
    }
}
