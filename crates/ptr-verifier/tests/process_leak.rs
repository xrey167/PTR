//! Repeated runs whose grandchild keeps the output pipe open must not accumulate blocked capture
//! threads. This is its own test binary so no other test's threads are counted (Linux only).
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::time::Duration;

use ptr_verifier::{CommandRequest, ProcessCommandVerifier, ProcessConfig, ProcessLimits};

fn threads() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("Threads:").map(|n| n.trim().parse().ok()))
        })
        .flatten()
        .unwrap_or(0)
}

#[test]
fn repeated_runs_with_a_pipe_holding_grandchild_do_not_accumulate_threads() {
    let sh = PathBuf::from("/bin/sh");
    if !sh.exists() {
        return;
    }
    let limits = ProcessLimits {
        timeout: Duration::from_secs(5),
        ..ProcessLimits::default()
    };
    let verifier = ProcessCommandVerifier::new(ProcessConfig::new(vec![sh], limits).unwrap());
    let request = CommandRequest::new(&["/bin/sh", "-c", "sleep 20 & exit 0"]);
    let before = threads();
    for _ in 0..10 {
        let run = verifier.run(&request).unwrap();
        assert!(run.output_truncated);
    }
    let leaked = threads().saturating_sub(before);
    assert!(
        leaked <= 16,
        "{leaked} capture threads are still blocked after 10 runs"
    );
}
