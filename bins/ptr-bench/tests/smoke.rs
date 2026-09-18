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
    assert!(stdout.contains(""benchmark":"semdb""));
    assert!(stdout.contains(""benchmark":"mailbox""));
}
