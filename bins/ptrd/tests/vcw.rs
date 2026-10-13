//! Starts the real `ptrd` binary with the `[vcw]` section and talks HTTP to it.
//!
//! The section is off by default (the endpoint then does not exist); enabled with simulated
//! experts it serves `POST /v1/vcw/tasks`, keeps its ledger across a restart, and a section it
//! cannot honour stops the daemon before it listens, and a revoked Pod stays revoked.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Daemon {
    child: Child,
    addr: String,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ptrd-vcw-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("data")).unwrap();
        Self(dir)
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
    }

    fn config(&self, text: &str) -> PathBuf {
        let path = self.0.join("ptr.toml");
        std::fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn free_addr() -> String {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .to_string()
}

/// On unix `ptrd` creates its sampling keys itself; elsewhere a missing key is a startup error
/// by design (see `load_or_create_key` in `src/main.rs`), so the test provisions fixed,
/// non-secret test keys the way an operator would before starting the daemon.
#[cfg(not(unix))]
fn provision_keys(data_dir: &std::path::Path) {
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

fn command(config: &Path, data: &Path, addr: &str) -> Command {
    #[cfg(not(unix))]
    provision_keys(data);
    let mut command = Command::new(env!("CARGO_BIN_EXE_ptrd"));
    command
        .env("PTR_CONFIG", config)
        .args(["--bind", addr, "--data-dir"])
        .arg(data)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

fn start(config: &Path, data: &Path) -> Daemon {
    let addr = free_addr();
    let child = command(config, data, &addr).spawn().unwrap();
    let daemon = Daemon { child, addr };
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if http(&daemon.addr, "GET", "/health", "").is_some() {
            return daemon;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("ptrd did not come up");
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
    let status = head.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, body.to_owned()))
}

const ENABLED: &str = r#"
[vcw]
enabled = true

[vcw.experts]
backend = "simulated"

[vcw.experts.simulated]
seed = 3

[vcw.experts.simulated.error_rates]
"arith.value" = 1.0
"#;

fn task(id: &str) -> String {
    format!(
        r#"{{"id":"{id}","task":{{"kind":"arithmetic","expression":"(3 + 4) * 5","reference":"35"}}}}"#
    )
}

#[test]
fn a_daemon_with_vcw_enabled_serves_the_endpoint_and_keeps_its_ledger_across_a_restart() {
    let scratch = Scratch::new("serve");
    let config = scratch.config(ENABLED);
    let first_digest;
    {
        let daemon = start(&config, &scratch.data());
        let (status, body) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("d1")).unwrap();
        assert_eq!(status, 200, "{body}");
        let json: serde_like::Fields = serde_like::Fields::of(&body);
        assert_eq!(json.get("outcome"), "verified");
        assert_eq!(json.get("answer"), "35");
        first_digest = json.get("trace_digest");
        assert_eq!(first_digest.len(), 64);

        // The existing endpoint and health are unchanged next to it.
        let (status, _) = http(&daemon.addr, "GET", "/health", "").unwrap();
        assert_eq!(status, 200);
        let (status, body) = http(
            &daemon.addr,
            "POST",
            "/v1/requests",
            r#"{"id":"plain","text":"hello"}"#,
        )
        .unwrap();
        assert_eq!(status, 200, "{body}");
    }

    // The same data directory again: the Pods are already live, the scope exists, and the
    // request id from the first run is remembered from the ledger.
    let daemon = start(&config, &scratch.data());
    let (status, _) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("d1")).unwrap();
    assert_eq!(status, 409);
    let (status, body) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("d2")).unwrap();
    assert_eq!(status, 200, "{body}");
}

#[test]
fn a_daemon_without_the_section_has_no_vcw_endpoint() {
    let scratch = Scratch::new("off");
    let config = scratch.config("");
    let daemon = start(&config, &scratch.data());
    let (status, _) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("x")).unwrap();
    assert_eq!(status, 404);
    let (status, _) = http(&daemon.addr, "GET", "/health", "").unwrap();
    assert_eq!(status, 200);
    // A section that is present but off is the same.
    let off = scratch.config("[vcw]\nenabled = false\n");
    drop(daemon);
    let daemon = start(&off, &scratch.data());
    let (status, _) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("x")).unwrap();
    assert_eq!(status, 404);
}

/// Runs the daemon on `config` and watches it until it exits: whether anything ever accepted a
/// connection on its address, its exit code, and what it printed.
fn run_to_exit(config: &Path, data: &Path) -> (Option<i32>, bool, String) {
    let addr = free_addr();
    let mut child = command(config, data, &addr)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut connected = false;
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if TcpStream::connect(&addr).is_ok() {
            connected = true;
        }
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "the daemon neither served nor exited"
        );
        std::thread::sleep(Duration::from_millis(2));
    };
    let mut printed = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut printed)
        .unwrap();
    (status.code(), connected, printed)
}

#[test]
fn an_unusable_section_stops_the_daemon_before_it_listens() {
    let scratch = Scratch::new("refuse");
    let cases = [
        // A misspelled key.
        "[vcw]\nenabeld = true\n",
        "[vcw.pods]\nproposr = \"x\"\n",
        // Valid TOML the section refuses.
        "[vcw]\nenabled = true\n[vcw.experts]\nbackend = \"simulated\"\nmax_rounds = 0\n",
        "[vcw]\nenabled = true\nmax_rounds = 0\n[vcw.experts]\nbackend = \"simulated\"\n",
        // No backend named: there is no default to fall back on.
        "[vcw]\nenabled = true\n",
        // A code the simulation cannot plant.
        "[vcw]\nenabled = true\n[vcw.experts]\nbackend = \"simulated\"\n\
         [vcw.experts.simulated.error_rates]\n\"arith.nonsense\" = 0.5\n",
        // The model backend with a model that is not there, whether or not this build has it.
        "[vcw]\nenabled = true\nkinds = [\"arithmetic\"]\n[vcw.experts]\nbackend = \"qwen-cpu\"\n\
         [vcw.experts.qwen_cpu]\nmodel_dir = \"/nonexistent-model\"\n\
         weights_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n\
         config_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n\
         tokenizer_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n\
         tokenizer_config_sha256 = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n",
    ];
    for (index, text) in cases.iter().enumerate() {
        let config = scratch.config(text);
        let (code, connected, printed) = run_to_exit(&config, &scratch.data());
        assert_eq!(code, Some(2), "case {index}");
        assert!(!connected, "case {index}: something accepted a connection");
        assert!(!printed.contains("listening"), "case {index}: {printed}");
    }
}

#[test]
fn a_build_error_in_the_service_surfaces_after_config_and_before_the_listener() {
    // A configuration that parses and validates but that the service build refuses (a per-Pod
    // override for a Pod that does not exist) must also stop the daemon before it listens.
    let scratch = Scratch::new("build");
    let config = scratch.config(
        "[vcw]\nenabled = true\n[vcw.experts]\nbackend = \"simulated\"\n\
         [vcw.experts.simulated.pod_repair_success.no-such-pod]\n\"arith.value\" = 0.5\n",
    );
    let (code, connected, printed) = run_to_exit(&config, &scratch.data());
    assert_eq!(code, Some(2));
    assert!(!connected);
    assert!(!printed.contains("listening"), "{printed}");
}

#[test]
fn a_revoked_pod_stays_revoked_after_the_daemon_restarts() {
    use ptr_config::PtrConfig;
    use ptr_ledger::LedgerEvent;
    use ptr_runtime::PtrRuntime;
    use ptr_types::Generation;

    let scratch = Scratch::new("revoke");
    let config = scratch.config(ENABLED);
    {
        let daemon = start(&config, &scratch.data());
        let (status, body) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("r1")).unwrap();
        assert_eq!(status, 200, "{body}");
    }
    // The operator revokes the proposer in the ledger while the daemon is down.
    {
        let mut runtime =
            PtrRuntime::open_durable(PtrConfig::default(), scratch.data().join("ledger.log"))
                .unwrap();
        runtime
            .commit(LedgerEvent::Revoked {
                subject: "vcw-proposer".to_owned(),
                generation: Generation(1),
            })
            .unwrap();
    }
    let daemon = start(&config, &scratch.data());
    let (status, body) = http(&daemon.addr, "POST", "/v1/vcw/tasks", &task("r2")).unwrap();
    assert_eq!(status, 422, "{body}");
    assert!(body.contains("\"escalated\""), "{body}");
    assert!(body.contains("NoProposer"), "{body}");
}

/// Just enough JSON field reading for flat string fields, without a JSON dependency.
mod serde_like {
    pub struct Fields<'a>(&'a str);

    impl<'a> Fields<'a> {
        pub fn of(body: &'a str) -> Self {
            Self(body)
        }

        pub fn get(&self, key: &str) -> String {
            let marker = format!("\"{key}\":\"");
            let start = self
                .0
                .find(&marker)
                .unwrap_or_else(|| panic!("{key} in {}", self.0))
                + marker.len();
            let rest = &self.0[start..];
            rest[..rest.find('"').unwrap_or(rest.len())].to_owned()
        }
    }
}
