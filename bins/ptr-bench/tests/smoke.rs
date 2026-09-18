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
