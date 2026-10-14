//! Public-API tests of the prototype process verifier (Unix only; skipped when a standard tool
//! is not installed).
#![cfg(unix)]

use std::path::PathBuf;
use std::time::Duration;

use ptr_verifier::{
    code_spec, CommandRequest, NamedVerifier, ProcessCommandVerifier, ProcessConfig,
    ProcessConfigError, ProcessExit, ProcessLimits, VerificationStatus, Verifier, VerifierFabric,
};

fn tool(name: &str) -> Option<String> {
    ["/bin", "/usr/bin"]
        .iter()
        .map(|dir| format!("{dir}/{name}"))
        .find(|path| std::path::Path::new(path).exists())
}

fn verifier(names: &[&str], timeout_ms: u64) -> Option<ProcessCommandVerifier> {
    let allowed = names
        .iter()
        .map(|name| tool(name).map(PathBuf::from))
        .collect::<Option<Vec<_>>>()?;
    let limits = ProcessLimits {
        timeout: Duration::from_millis(timeout_ms),
        max_capture_bytes: 32,
        ..ProcessLimits::default()
    };
    Some(ProcessCommandVerifier::new(
        ProcessConfig::new(allowed, limits).ok()?,
    ))
}

fn request(program: &str, args: &[&str]) -> CommandRequest {
    let mut argv = vec![tool(program).unwrap()];
    argv.extend(args.iter().map(|a| (*a).to_owned()));
    CommandRequest { argv }
}

#[test]
fn exit_codes_map_to_pass_and_fail() {
    let Some(v) = verifier(&["true", "false"], 2000) else {
        return;
    };
    assert_eq!(
        v.verify(&request("true", &[])).status,
        VerificationStatus::Pass
    );
    let fail = v.verify(&request("false", &[]));
    assert_eq!(fail.status, VerificationStatus::Fail);
    assert_eq!(fail.findings[0].code, "process.exit_nonzero");
    assert_eq!(NamedVerifier::<CommandRequest>::name(&v), "process-command");
}

#[test]
fn output_is_captured_and_truncated() {
    let Some(v) = verifier(&["echo"], 2000) else {
        return;
    };
    let run = v.run(&request("echo", &["captured"])).unwrap();
    assert_eq!(run.stdout, b"captured\n");
    let run = v.run(&request("echo", &[&"y".repeat(100)])).unwrap();
    assert_eq!(run.stdout.len(), 32);
    assert!(run.output_truncated);
}

#[test]
fn timeout_kills_and_reaps_the_child() {
    let Some(v) = verifier(&["sleep"], 200) else {
        return;
    };
    let run = v.run(&request("sleep", &["5"])).unwrap();
    assert_eq!(run.exit, ProcessExit::TimedOut);
    assert!(run.elapsed < Duration::from_secs(4));
    #[cfg(target_os = "linux")]
    assert!(!std::path::Path::new(&format!("/proc/{}", run.pid.unwrap())).exists());
    let report = v.verify(&request("sleep", &["5"]));
    assert_eq!(report.findings[0].code, "process.timeout");
    assert_eq!(report.status, VerificationStatus::Fail);
}

#[test]
fn every_process_code_is_in_the_table_with_the_right_hardness() {
    let Some(v) = verifier(&["true", "false", "echo"], 2000) else {
        return;
    };
    let long = "z".repeat(100);
    let requests = [
        CommandRequest { argv: vec![] },
        CommandRequest::new(&["/bin/sh", "-c", "true"]),
        CommandRequest {
            argv: vec![tool("true").unwrap(); 40],
        },
        request("false", &[]),
        request("echo", &[&long]),
        CommandRequest::new(&["/nonexistent/ptr-ghost"]),
    ];
    for request in requests {
        for finding in v.verify(&request).findings {
            let spec = code_spec(&finding.code).unwrap_or_else(|| panic!("{}", finding.code));
            assert_eq!(spec.hard, finding.hard, "{}", finding.code);
        }
    }
    // The ghost executable is not allowlisted here; the allowlisted-but-missing case is soft.
    let ghost = PathBuf::from("/nonexistent/ptr-ghost");
    let g = ProcessCommandVerifier::new(
        ProcessConfig::new(vec![ghost.clone()], ProcessLimits::default()).unwrap(),
    );
    let report = g.verify(&CommandRequest {
        argv: vec![ghost.to_string_lossy().into_owned()],
    });
    assert_eq!(report.status, VerificationStatus::Unknown);
    assert_eq!(report.findings[0].code, "process.spawn");
}

#[test]
fn a_failing_command_dominates_a_passing_one_in_a_fabric() {
    let Some(pass) = verifier(&["true", "false"], 2000) else {
        return;
    };
    let mut fabric = VerifierFabric::<CommandRequest>::default();
    fabric.push(pass.clone());
    fabric.push(pass);
    assert_eq!(
        fabric.verify(&request("true", &[])).status,
        VerificationStatus::Pass
    );
    let report = fabric.verify(&request("false", &[]));
    assert_eq!(report.status, VerificationStatus::Fail);
    assert!(report.findings.iter().all(|f| f.hard));
}

#[test]
fn secret_like_environment_names_are_refused() {
    let config =
        ProcessConfig::new(vec![PathBuf::from("/bin/true")], ProcessLimits::default()).unwrap();
    assert_eq!(
        config.with_env("API_TOKEN", "x"),
        Err(ProcessConfigError::SecretLikeEnvName("API_TOKEN".into()))
    );
}
