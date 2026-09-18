use std::path::PathBuf;
use std::process::Command;

#[test]
fn doctor_validates_repository_setup() {
    let exe = env!("CARGO_BIN_EXE_ptrctl");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let output = Command::new(exe)
        .arg("doctor")
        .arg(root)
        .output()
        .expect("run ptrctl doctor");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("PTR doctor: OK"));
    assert!(stdout.contains("config: OK"));
}
