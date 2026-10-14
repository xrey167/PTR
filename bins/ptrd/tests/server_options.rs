//! The opt-in `--trace-file` / `--trace-max-bytes` / `--ingress-diagnostics` flags of the real
//! `ptrd` binary. Without them nothing changes (covered by the other test files).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Daemon {
    child: Child,
    addr: String,
    dir: PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ptrd-opts-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn free_addr() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().to_string()
}

/// On unix `ptrd` creates its sampling keys itself; elsewhere the test provisions fixed,
/// non-secret test keys, as `reference_kv.rs` does.
#[cfg(not(unix))]
fn provision_keys(data_dir: &Path) {
    std::fs::create_dir_all(data_dir).unwrap();
    for (file, byte) in [
        ("router-sampling.key", 0x5a_u8),
        ("vcw-sampling.key", 0xa5_u8),
    ] {
        let path = data_dir.join(file);
        if !path.exists() {
            std::fs::write(&path, [byte; 32]).unwrap();
        }
    }
}

fn command(dir: &Path, addr: &str, extra: &[&str]) -> Command {
    let config = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/default.toml");
    #[cfg(not(unix))]
    provision_keys(dir);
    let mut command = Command::new(env!("CARGO_BIN_EXE_ptrd"));
    command
        .env("PTR_CONFIG", config)
        .args(["--bind", addr, "--backend", "reference-echo", "--data-dir"])
        .arg(dir)
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    command
}

fn http(addr: &str, method: &str, path: &str, body: &str) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(addr).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .ok()?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).ok()?;
    let (head, body) = raw.split_once("\r\n\r\n")?;
    Some((
        head.split_whitespace().nth(1)?.parse().ok()?,
        body.to_owned(),
    ))
}

fn start(tag: &str, extra: &[&str]) -> Daemon {
    let dir = temp_dir(tag);
    let addr = free_addr();
    let mut command = command(&dir, &addr, extra);
    command.stderr(Stdio::null());
    let child = command.spawn().unwrap();
    let daemon = Daemon { child, addr, dir };
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if http(&daemon.addr, "GET", "/health", "").is_some() {
            return daemon;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("ptrd did not come up");
}

fn refused(tag: &str, extra: &[&str]) -> String {
    let dir = temp_dir(tag);
    let output = command(&dir, &free_addr(), extra).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(output.status.code(), Some(2), "{extra:?} must exit 2");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn trace_file_receives_one_redacted_span_line_per_request() {
    let trace_dir = temp_dir("trace-out");
    let trace = trace_dir.join("trace.jsonl");
    let daemon = start("trace", &["--trace-file", trace.to_str().unwrap()]);
    let secret = "zebra-quartz-marker-7731";
    let (status, body) = http(
        &daemon.addr,
        "POST",
        "/v1/requests",
        &format!(r#"{{"id":"t-1","text":"{secret}"}}"#),
    )
    .unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(
        !body.contains("ingress_diagnostics"),
        "diagnostics stay off"
    );

    // The span is written after the response is built: wait for it, bounded.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut content = String::new();
    while Instant::now() < deadline {
        content = std::fs::read_to_string(&trace).unwrap_or_default();
        if content.contains("/v1/requests") {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = std::fs::remove_dir_all(&trace_dir);
    // Health probes made while waiting for startup are traced too; the request is one line.
    let lines: Vec<&str> = content
        .lines()
        .filter(|line| line.contains(r#""stringValue":"/v1/requests""#))
        .collect();
    assert_eq!(lines.len(), 1, "one span line for the request: {content:?}");
    assert!(
        lines[0].starts_with(r#"{"resourceSpans":["#),
        "OTLP/JSON-shaped: {content}"
    );
    assert!(
        !content.contains(secret),
        "request text must not be exported"
    );
}

#[test]
fn ingress_diagnostics_flag_adds_the_advisory_field() {
    let daemon = start("diag", &["--ingress-diagnostics"]);
    let (status, body) = http(
        &daemon.addr,
        "POST",
        "/v1/requests",
        r#"{"id":"d-1","text":"hello"}"#,
    )
    .unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("ingress_diagnostics"), "{body}");
}

#[test]
fn bad_option_values_exit_2_with_a_clear_message() {
    let cases: [&[&str]; 5] = [
        &["--trace-max-bytes", "100"],
        &["--trace-file", "x.jsonl", "--trace-max-bytes", "0"],
        &["--trace-file", "x.jsonl", "--trace-max-bytes", "many"],
        &["--trace-file"],
        // Unusable file: its parent directory does not exist.
        &["--trace-file", "/nonexistent-ptrd-dir-xyz/sub/trace.jsonl"],
    ];
    for (index, extra) in cases.iter().enumerate() {
        let stderr = refused(&format!("bad{index}"), extra);
        assert!(!stderr.is_empty(), "{extra:?} needs an error message");
        assert!(!stderr.contains("panicked"), "{extra:?}: {stderr}");
    }
}

#[test]
fn help_documents_the_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_ptrd"))
        .arg("--help")
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&output.stdout);
    for flag in ["--trace-file", "--trace-max-bytes", "--ingress-diagnostics"] {
        assert!(help.contains(flag), "--help must mention {flag}");
    }
}
