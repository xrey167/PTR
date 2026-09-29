use std::process::Command;

#[test]
fn benchmark_binary_emits_json_lines() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["all", "10"])
        .output()
        .expect("run ptr-bench");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(r#""benchmark":"semdb""#));
    assert!(stdout.contains(r#""benchmark":"mailbox""#));
}

#[test]
fn ledger_recovery_smoke_has_zero_false_accepts() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["ledger-recovery", "3", "17"])
        .output()
        .expect("run ledger recovery smoke");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(r#""false_accepts":0"#));
    assert!(stdout.contains(r#""recovery_errors":0"#));
    assert!(stdout.contains(r#""tail_trim_errors":0"#));
}

#[test]
fn certified_branches_runs_one_case_clean() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["certified-branches", "1", "17"])
        .output()
        .expect("run the S003 smoke");
    assert!(
        output.status.success(),
        "a hard counter is above zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "exactly one JSON line on stdout");
    assert!(stdout.starts_with(r#"{"benchmark":"certified-branches","iterations":1,"seed":17,"server":"in-memory","preregistration":"#));
    assert!(stdout.contains(r#""hard_failures":0"#));
    for name in [
        "lost_updates",
        "provenance_mismatches",
        "canary_misses",
        "nondeterminism",
        "harness_errors",
    ] {
        assert!(stdout.contains(&format!(r#""{name}":0"#)), "{name}");
    }
    // The probes, the canaries and the round trips ran.
    assert!(stdout.contains(r#""probe_p26_exercised":1"#));
    assert!(stdout.contains(r#""canaries_run":9"#));
    assert!(stdout.contains(r#""durable_roundtrips":"#));
    assert!(stdout.contains(r#""cases":[{"case":0,"level":0,"groups":4"#));
}
