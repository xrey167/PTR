//! Prototype process-isolated command verification.
//!
//! [`ProcessCommandVerifier`] runs a candidate command as a child process and maps how it ended
//! to a verdict. It is a PROTOTYPE of process isolation, not a sandbox: there are no namespaces,
//! no seccomp, no cgroup, no network or filesystem confinement and no process-group kill, so a
//! allowlisted program can still read what the OS user can read and a grandchild it spawns can
//! outlive the timeout. What it does enforce, with `std::process` only:
//!
//! * the executable must be an absolute path that is in the caller's allowlist, compared as the
//!   exact path (no `PATH` search, no shell: the argument vector goes to the program as given).
//!   Never allowlist an interpreter or a shell; it would run arbitrary code;
//! * the environment is cleared, and only the caller's explicit `name=value` pairs are set.
//!   Names that look like secrets (`KEY`, `TOKEN`, `SECRET`, `PASSWORD`, ...) are refused at
//!   configuration time, and the parent's environment is never inherited (invariant 13);
//! * the working directory is a fresh, empty directory under the system temp dir, removed
//!   afterwards;
//! * stdin is closed (`/dev/null`);
//! * stdout and stderr are captured up to a byte limit each; the rest is read and discarded so
//!   the child never blocks on a full pipe, and the capture is marked truncated;
//! * a wall-clock timeout is enforced by polling `try_wait`; on expiry the child is killed and
//!   reaped (`wait`) before the call returns;
//! * argument count, argument size and the number of allowlisted executables are bounded.
//!
//! The exit status maps as follows: exit 0 is `Pass`; an exit code in the configured fail codes
//! (default `[1]`) is `Fail` (`process.exit_nonzero`); a timeout is `Fail` (`process.timeout`);
//! any other exit code, death by signal, or a failure to start or wait is `Unknown` with a soft
//! finding (`process.exit_unmapped`, `process.signal`, `process.spawn`). A candidate that is
//! empty, oversize or not allowlisted is a hard finding (`process.malformed`,
//! `process.oversize`, `process.not_allowed`) and nothing is started. Truncated output adds the
//! soft `process.output_truncated`. Findings never quote the child's output (it may contain
//! anything); [`ProcessCommandVerifier::run`] returns the captured bytes to the caller.
//! See [`crate::codes`].

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ptr_types::{Probability, VerificationLevel};

use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// Most allowlisted executables.
pub const MAX_ALLOWED_EXECUTABLES: usize = 64;
/// Most explicit environment variables.
pub const MAX_ENV_VARS: usize = 16;
/// Longest environment variable name or value, in bytes.
pub const MAX_ENV_BYTES: usize = 256;
/// Longest timeout a configuration may ask for.
pub const MAX_TIMEOUT: Duration = Duration::from_secs(60);
/// Largest per-stream capture a configuration may ask for.
pub const MAX_CAPTURE_BYTES: usize = 1 << 20;
/// How long to wait for a reader thread after the child is gone (a grandchild may hold the pipe).
const DRAIN_GRACE: Duration = Duration::from_millis(500);
/// Interval of the `try_wait` poll.
const POLL: Duration = Duration::from_millis(5);

/// Substrings (upper case) that make an environment variable name look like a secret.
const SECRET_MARKERS: [&str; 9] = [
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
    "SESSION",
];

/// Why a [`ProcessConfig`] or its [`ProcessLimits`] are not usable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessConfigError {
    EmptyAllowlist,
    TooManyExecutables,
    /// The allowlist entry is not an absolute path (a bare name would be searched on `PATH`).
    RelativeExecutable(PathBuf),
    ZeroTimeout,
    TimeoutTooLong,
    ZeroCapture,
    CaptureTooLarge,
    ZeroArgs,
    ZeroArgBytes,
    TooManyEnvVars,
    /// The name is empty, too long, or not `[A-Za-z_][A-Za-z0-9_]*`.
    InvalidEnvName(String),
    /// The value is too long or contains a NUL byte. The value is not echoed.
    InvalidEnvValue(String),
    /// The name looks like a secret; secrets are never passed to a child.
    SecretLikeEnvName(String),
}

/// Bounds of one run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessLimits {
    /// Wall-clock budget; the child is killed when it expires.
    pub timeout: Duration,
    /// Bytes kept per stream; the rest is discarded.
    pub max_capture_bytes: usize,
    /// Most entries in the argument vector, the executable included.
    pub max_args: usize,
    /// Longest single argument, in bytes.
    pub max_arg_bytes: usize,
}

impl Default for ProcessLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            max_capture_bytes: 16 * 1024,
            max_args: 32,
            max_arg_bytes: 4096,
        }
    }
}

/// The caller's policy for [`ProcessCommandVerifier`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessConfig {
    allowed: Vec<PathBuf>,
    env: Vec<(String, String)>,
    limits: ProcessLimits,
    fail_codes: Vec<i32>,
}

fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name.len() <= MAX_ENV_BYTES
}

impl ProcessConfig {
    /// A policy allowing exactly `allowed` (absolute paths) under `limits`, with an empty
    /// environment and exit code 1 as the only `Fail` code.
    pub fn new(allowed: Vec<PathBuf>, limits: ProcessLimits) -> Result<Self, ProcessConfigError> {
        if allowed.is_empty() {
            return Err(ProcessConfigError::EmptyAllowlist);
        }
        if allowed.len() > MAX_ALLOWED_EXECUTABLES {
            return Err(ProcessConfigError::TooManyExecutables);
        }
        if let Some(relative) = allowed.iter().find(|path| !path.is_absolute()) {
            return Err(ProcessConfigError::RelativeExecutable(relative.clone()));
        }
        if limits.timeout.is_zero() {
            return Err(ProcessConfigError::ZeroTimeout);
        }
        if limits.timeout > MAX_TIMEOUT {
            return Err(ProcessConfigError::TimeoutTooLong);
        }
        if limits.max_capture_bytes == 0 {
            return Err(ProcessConfigError::ZeroCapture);
        }
        if limits.max_capture_bytes > MAX_CAPTURE_BYTES {
            return Err(ProcessConfigError::CaptureTooLarge);
        }
        if limits.max_args == 0 {
            return Err(ProcessConfigError::ZeroArgs);
        }
        if limits.max_arg_bytes == 0 {
            return Err(ProcessConfigError::ZeroArgBytes);
        }
        Ok(Self {
            allowed,
            env: Vec::new(),
            limits,
            fail_codes: vec![1],
        })
    }

    /// Adds one explicit environment variable. Nothing else is ever inherited.
    pub fn with_env(mut self, name: &str, value: &str) -> Result<Self, ProcessConfigError> {
        if self.env.len() >= MAX_ENV_VARS {
            return Err(ProcessConfigError::TooManyEnvVars);
        }
        if !valid_env_name(name) {
            return Err(ProcessConfigError::InvalidEnvName(name.to_owned()));
        }
        let upper = name.to_ascii_uppercase();
        if SECRET_MARKERS.iter().any(|marker| upper.contains(marker)) {
            return Err(ProcessConfigError::SecretLikeEnvName(name.to_owned()));
        }
        if value.len() > MAX_ENV_BYTES || value.contains('\0') {
            return Err(ProcessConfigError::InvalidEnvValue(name.to_owned()));
        }
        self.env.retain(|(existing, _)| existing != name);
        self.env.push((name.to_owned(), value.to_owned()));
        Ok(self)
    }

    /// The exit codes that mean `Fail`; exit 0 is always `Pass`, any other code `Unknown`.
    pub fn with_fail_codes(mut self, codes: &[i32]) -> Self {
        self.fail_codes = codes.iter().copied().filter(|code| *code != 0).collect();
        self
    }

    pub fn limits(&self) -> ProcessLimits {
        self.limits
    }
}

/// A command to run: `argv[0]` is the executable, the rest its arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandRequest {
    pub argv: Vec<String>,
}

impl CommandRequest {
    pub fn new(argv: &[&str]) -> Self {
        Self {
            argv: argv.iter().map(|arg| (*arg).to_owned()).collect(),
        }
    }
}

/// How a run ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessExit {
    /// The process exited with this code.
    Code(i32),
    /// The process was ended by a signal it did not choose (no exit code).
    Signal,
    /// The timeout expired; the child was killed and reaped.
    TimedOut,
    /// The scratch directory or the process could not be created.
    SpawnFailed(std::io::ErrorKind),
    /// Waiting on the process failed; it was killed and reaped as far as possible.
    WaitFailed(std::io::ErrorKind),
}

/// The observed result of one run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessRun {
    pub exit: ProcessExit,
    /// Up to `max_capture_bytes` of stdout.
    pub stdout: Vec<u8>,
    /// Up to `max_capture_bytes` of stderr.
    pub stderr: Vec<u8>,
    /// Output beyond the limit was discarded, or a stream could not be drained in time.
    pub output_truncated: bool,
    pub elapsed: Duration,
    /// The child's process id, if it was started. It is reaped when the run returns.
    pub pid: Option<u32>,
}

/// Why a request was refused before anything was started.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessRejection {
    EmptyArgv,
    TooManyArgs { got: usize, max: usize },
    ArgTooLong { index: usize, max: usize },
    NulByte { index: usize },
    NotAllowed { executable: String },
}

/// Runs allowlisted commands as isolated-by-convention child processes. See the module
/// documentation for exactly what is and is not enforced.
#[derive(Clone, Debug)]
pub struct ProcessCommandVerifier {
    config: ProcessConfig,
}

/// A fresh scratch directory, removed on drop.
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn create() -> std::io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "ptr-verifier-proc-{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

type Capture = mpsc::Receiver<(Vec<u8>, bool)>;

/// Reads `stream` to the end on a thread, keeping at most `limit` bytes.
fn capture<R: Read + Send + 'static>(stream: Option<R>, limit: usize) -> Option<Capture> {
    let mut stream = stream?;
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("ptr-verifier-capture".into())
        .spawn(move || {
            let mut kept = Vec::new();
            let mut truncated = false;
            let mut buffer = [0_u8; 8192];
            loop {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let room = limit.saturating_sub(kept.len());
                        kept.extend_from_slice(&buffer[..n.min(room)]);
                        truncated |= n > room;
                    }
                }
            }
            let _ = tx.send((kept, truncated));
        });
    spawned.ok().map(|_| rx)
}

fn collect(capture: Option<Capture>) -> (Vec<u8>, bool) {
    match capture.map(|rx| rx.recv_timeout(DRAIN_GRACE)) {
        Some(Ok(result)) => result,
        // No reader, or it did not finish: the output is incomplete.
        Some(Err(_)) | None => (Vec::new(), true),
    }
}

impl ProcessCommandVerifier {
    pub fn new(config: ProcessConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ProcessConfig {
        &self.config
    }

    fn check(&self, request: &CommandRequest) -> Result<(), ProcessRejection> {
        let limits = &self.config.limits;
        let Some(executable) = request.argv.first() else {
            return Err(ProcessRejection::EmptyArgv);
        };
        if request.argv.len() > limits.max_args {
            return Err(ProcessRejection::TooManyArgs {
                got: request.argv.len(),
                max: limits.max_args,
            });
        }
        for (index, arg) in request.argv.iter().enumerate() {
            if arg.len() > limits.max_arg_bytes {
                return Err(ProcessRejection::ArgTooLong {
                    index,
                    max: limits.max_arg_bytes,
                });
            }
            if arg.contains('\0') {
                return Err(ProcessRejection::NulByte { index });
            }
        }
        let path = PathBuf::from(executable);
        if !self.config.allowed.contains(&path) {
            return Err(ProcessRejection::NotAllowed {
                executable: executable.clone(),
            });
        }
        Ok(())
    }

    /// Runs `request` and returns what was observed. A refused request starts nothing.
    pub fn run(&self, request: &CommandRequest) -> Result<ProcessRun, ProcessRejection> {
        self.check(request)?;
        let started = Instant::now();
        let finish = |exit, pid, stdout, stderr, output_truncated| ProcessRun {
            exit,
            stdout,
            stderr,
            output_truncated,
            elapsed: started.elapsed(),
            pid,
        };
        let scratch = match ScratchDir::create() {
            Ok(scratch) => scratch,
            Err(error) => {
                return Ok(finish(
                    ProcessExit::SpawnFailed(error.kind()),
                    None,
                    Vec::new(),
                    Vec::new(),
                    false,
                ))
            }
        };
        let limits = self.config.limits;
        let mut child = match Command::new(&request.argv[0])
            .args(&request.argv[1..])
            .env_clear()
            .envs(self.config.env.iter().map(|(k, v)| (k, v)))
            .current_dir(&scratch.path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                return Ok(finish(
                    ProcessExit::SpawnFailed(error.kind()),
                    None,
                    Vec::new(),
                    Vec::new(),
                    false,
                ))
            }
        };
        let pid = child.id();
        let stdout = capture(child.stdout.take(), limits.max_capture_bytes);
        let stderr = capture(child.stderr.take(), limits.max_capture_bytes);
        let deadline = started + limits.timeout;
        let exit = loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    break status.code().map_or(ProcessExit::Signal, ProcessExit::Code);
                }
                Ok(None) => {
                    let now = Instant::now();
                    if now >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        break ProcessExit::TimedOut;
                    }
                    std::thread::sleep(POLL.min(deadline - now));
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break ProcessExit::WaitFailed(error.kind());
                }
            }
        };
        let (out, out_truncated) = collect(stdout);
        let (err, err_truncated) = collect(stderr);
        drop(scratch);
        Ok(finish(
            exit,
            Some(pid),
            out,
            err,
            out_truncated || err_truncated,
        ))
    }
}

fn report(status: VerificationStatus, findings: Vec<Finding>) -> VerificationReport {
    let score = if status == VerificationStatus::Pass {
        1.0
    } else {
        0.0
    };
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).expect("bounded score"),
        findings,
    }
}

fn rejection_finding(rejection: &ProcessRejection) -> Finding {
    match rejection {
        ProcessRejection::EmptyArgv => {
            finding("process.malformed", "the argument vector is empty".into())
        }
        ProcessRejection::NulByte { index } => finding(
            "process.malformed",
            format!("argument {index} contains a NUL byte"),
        ),
        ProcessRejection::TooManyArgs { got, max } => finding(
            "process.oversize",
            format!("{got} arguments, the bound is {max}"),
        ),
        ProcessRejection::ArgTooLong { index, max } => finding(
            "process.oversize",
            format!("argument {index} is longer than {max} bytes"),
        ),
        ProcessRejection::NotAllowed { executable } => finding(
            "process.not_allowed",
            format!("executable {executable:?} is not in the allowlist"),
        ),
    }
}

impl Verifier<CommandRequest> for ProcessCommandVerifier {
    fn verify(&self, request: &CommandRequest) -> VerificationReport {
        let run = match self.run(request) {
            Ok(run) => run,
            Err(rejection) => {
                return report(
                    VerificationStatus::Fail,
                    vec![rejection_finding(&rejection)],
                );
            }
        };
        let (status, mut findings) = match run.exit {
            ProcessExit::Code(0) => (VerificationStatus::Pass, Vec::new()),
            ProcessExit::Code(code) if self.config.fail_codes.contains(&code) => (
                VerificationStatus::Fail,
                vec![finding(
                    "process.exit_nonzero",
                    format!("the command exited with code {code}"),
                )],
            ),
            ProcessExit::Code(code) => (
                VerificationStatus::Unknown,
                vec![finding(
                    "process.exit_unmapped",
                    format!("the command exited with code {code}, which is not a verdict"),
                )],
            ),
            ProcessExit::Signal => (
                VerificationStatus::Unknown,
                vec![finding(
                    "process.signal",
                    "the command was ended by a signal".into(),
                )],
            ),
            ProcessExit::TimedOut => (
                VerificationStatus::Fail,
                vec![finding(
                    "process.timeout",
                    format!(
                        "the command did not finish within {} ms and was killed",
                        self.config.limits.timeout.as_millis()
                    ),
                )],
            ),
            ProcessExit::SpawnFailed(kind) => (
                VerificationStatus::Unknown,
                vec![finding(
                    "process.spawn",
                    format!("the command could not be started: {kind}"),
                )],
            ),
            ProcessExit::WaitFailed(kind) => (
                VerificationStatus::Unknown,
                vec![finding(
                    "process.spawn",
                    format!("waiting for the command failed: {kind}"),
                )],
            ),
        };
        if run.output_truncated {
            findings.push(finding(
                "process.output_truncated",
                format!(
                    "output beyond {} bytes per stream was discarded",
                    self.config.limits.max_capture_bytes
                ),
            ));
        }
        report(status, findings)
    }
}

impl NamedVerifier<CommandRequest> for ProcessCommandVerifier {
    fn name(&self) -> &'static str {
        "process-command"
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

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
            max_capture_bytes: 64,
            ..ProcessLimits::default()
        };
        Some(ProcessCommandVerifier::new(
            ProcessConfig::new(allowed, limits).ok()?,
        ))
    }

    fn argv(program: &str, args: &[&str]) -> CommandRequest {
        let mut vector = vec![tool(program).unwrap()];
        vector.extend(args.iter().map(|a| (*a).to_owned()));
        CommandRequest { argv: vector }
    }

    #[test]
    fn exit_zero_passes_and_exit_one_fails() {
        let Some(v) = verifier(&["true", "false"], 2000) else {
            return;
        };
        let pass = v.verify(&argv("true", &[]));
        assert_eq!(pass.status, VerificationStatus::Pass);
        assert_eq!(pass.score.get(), 1.0);
        assert!(pass.findings.is_empty());
        let fail = v.verify(&argv("false", &[]));
        assert_eq!(fail.status, VerificationStatus::Fail);
        assert_eq!(fail.findings[0].code, "process.exit_nonzero");
        assert!(fail.findings[0].hard);
    }

    #[test]
    fn captured_output_is_bounded_and_marked_truncated() {
        let Some(v) = verifier(&["echo"], 2000) else {
            return;
        };
        let run = v.run(&argv("echo", &["hello"])).unwrap();
        assert_eq!(run.exit, ProcessExit::Code(0));
        assert_eq!(run.stdout, b"hello\n");
        assert!(run.stderr.is_empty());
        assert!(!run.output_truncated);
        let long = "x".repeat(200);
        let run = v.run(&argv("echo", &[&long])).unwrap();
        assert_eq!(run.stdout.len(), 64);
        assert!(run.output_truncated);
        let report = v.verify(&argv("echo", &[&long]));
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].code, "process.output_truncated");
        assert!(!report.findings[0].hard);
    }

    #[test]
    fn a_hung_command_is_killed_and_reaped_at_the_timeout() {
        let Some(v) = verifier(&["sleep"], 200) else {
            return;
        };
        let run = v.run(&argv("sleep", &["5"])).unwrap();
        assert_eq!(run.exit, ProcessExit::TimedOut);
        assert!(run.elapsed >= Duration::from_millis(200));
        assert!(run.elapsed < Duration::from_secs(4), "{:?}", run.elapsed);
        #[cfg(target_os = "linux")]
        {
            let pid = run.pid.unwrap();
            assert!(
                !std::path::Path::new(&format!("/proc/{pid}")).exists(),
                "the killed child must be reaped, not left a zombie"
            );
        }
        let report = v.verify(&argv("sleep", &["5"]));
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(report.findings[0].code, "process.timeout");
        assert!(report.findings[0].hard);
    }

    #[test]
    fn the_environment_is_cleared_except_for_the_explicit_list() {
        let Some(env_tool) = tool("env") else {
            return;
        };
        let config = ProcessConfig::new(vec![PathBuf::from(&env_tool)], ProcessLimits::default())
            .unwrap()
            .with_env("PTR_ALLOWED", "yes")
            .unwrap();
        let v = ProcessCommandVerifier::new(config);
        let run = v
            .run(&CommandRequest {
                argv: vec![env_tool],
            })
            .unwrap();
        let text = String::from_utf8_lossy(&run.stdout).into_owned();
        assert!(text.contains("PTR_ALLOWED=yes"), "{text}");
        // Nothing of the parent's environment (which has PATH on any CI machine) is inherited.
        assert!(!text.contains("PATH="), "{text}");
        assert!(!text.contains("HOME="), "{text}");
        assert_eq!(text.lines().count(), 1, "{text}");
    }

    #[test]
    fn the_working_directory_is_a_fresh_temp_dir_that_is_removed() {
        let Some(pwd) = tool("pwd") else {
            return;
        };
        let config =
            ProcessConfig::new(vec![PathBuf::from(&pwd)], ProcessLimits::default()).unwrap();
        let v = ProcessCommandVerifier::new(config);
        let run = v.run(&CommandRequest { argv: vec![pwd] }).unwrap();
        let cwd = PathBuf::from(String::from_utf8_lossy(&run.stdout).trim().to_owned());
        assert!(cwd.to_string_lossy().contains("ptr-verifier-proc-"));
        assert!(!cwd.exists(), "the scratch directory is removed afterwards");
    }

    #[test]
    fn stdin_is_closed() {
        let Some(cat) = tool("cat") else {
            return;
        };
        let config =
            ProcessConfig::new(vec![PathBuf::from(&cat)], ProcessLimits::default()).unwrap();
        let v = ProcessCommandVerifier::new(config);
        // `cat` with no file reads stdin: with stdin at EOF it exits 0 at once with no output.
        let run = v.run(&CommandRequest { argv: vec![cat] }).unwrap();
        assert_eq!(run.exit, ProcessExit::Code(0));
        assert!(run.stdout.is_empty());
    }

    #[test]
    fn a_missing_executable_is_unknown_not_a_verdict() {
        // A start failure: the path is allowlisted but does not exist.
        let ghost = PathBuf::from("/nonexistent/ptr-verifier-ghost");
        let config = ProcessConfig::new(vec![ghost.clone()], ProcessLimits::default()).unwrap();
        let ghost_v = ProcessCommandVerifier::new(config);
        let report = ghost_v.verify(&CommandRequest {
            argv: vec![ghost.to_string_lossy().into_owned()],
        });
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(report.findings[0].code, "process.spawn");
        assert!(!report.findings[0].hard);
        assert_eq!(report.score.get(), 0.0);
    }

    #[test]
    fn exit_codes_outside_the_fail_set_and_signals_are_unknown() {
        let Some(sh_free) = tool("false") else {
            return;
        };
        // `false` exits 1; with fail codes [2] it is an unmapped code, not a verdict.
        let config = ProcessConfig::new(vec![PathBuf::from(&sh_free)], ProcessLimits::default())
            .unwrap()
            .with_fail_codes(&[2]);
        let v = ProcessCommandVerifier::new(config);
        let report = v.verify(&CommandRequest {
            argv: vec![sh_free.clone()],
        });
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(report.findings[0].code, "process.exit_unmapped");
        assert!(!report.findings[0].hard);
        // A fail-code list never turns exit 0 into a failure.
        let zero = ProcessConfig::new(vec![PathBuf::from(&sh_free)], ProcessLimits::default())
            .unwrap()
            .with_fail_codes(&[0, 1]);
        assert_eq!(zero.fail_codes, vec![1]);
    }

    #[test]
    fn refused_requests_start_nothing() {
        let Some(v) = verifier(&["true"], 2000) else {
            return;
        };
        let cases: [(CommandRequest, &str); 5] = [
            (CommandRequest { argv: vec![] }, "process.malformed"),
            (argv("true", &["a\0b"]), "process.malformed"),
            (argv("true", &[&"x".repeat(5000)]), "process.oversize"),
            (
                CommandRequest {
                    argv: vec![tool("true").unwrap(); 40],
                },
                "process.oversize",
            ),
            (
                CommandRequest::new(&["/bin/sh", "-c", "true"]),
                "process.not_allowed",
            ),
        ];
        for (request, code) in cases {
            assert!(v.run(&request).is_err(), "{code}");
            let report = v.verify(&request);
            assert_eq!(report.status, VerificationStatus::Fail, "{code}");
            assert_eq!(report.findings[0].code, code);
            assert!(report.findings[0].hard);
        }
        // A bare name is never searched on PATH.
        assert_eq!(
            v.run(&CommandRequest::new(&["true"])),
            Err(ProcessRejection::NotAllowed {
                executable: "true".into()
            })
        );
    }

    #[test]
    fn configuration_errors_are_typed() {
        let abs = vec![PathBuf::from("/bin/true")];
        let limits = ProcessLimits::default();
        assert_eq!(
            ProcessConfig::new(vec![], limits),
            Err(ProcessConfigError::EmptyAllowlist)
        );
        assert_eq!(
            ProcessConfig::new(vec![PathBuf::from("true")], limits),
            Err(ProcessConfigError::RelativeExecutable(PathBuf::from(
                "true"
            )))
        );
        assert_eq!(
            ProcessConfig::new(vec![PathBuf::from("/bin/true"); 65], limits),
            Err(ProcessConfigError::TooManyExecutables)
        );
        let with = |f: fn(&mut ProcessLimits)| {
            let mut l = limits;
            f(&mut l);
            ProcessConfig::new(abs.clone(), l)
        };
        assert_eq!(
            with(|l| l.timeout = Duration::ZERO),
            Err(ProcessConfigError::ZeroTimeout)
        );
        assert_eq!(
            with(|l| l.timeout = Duration::from_secs(61)),
            Err(ProcessConfigError::TimeoutTooLong)
        );
        assert_eq!(
            with(|l| l.max_capture_bytes = 0),
            Err(ProcessConfigError::ZeroCapture)
        );
        assert_eq!(
            with(|l| l.max_capture_bytes = (1 << 20) + 1),
            Err(ProcessConfigError::CaptureTooLarge)
        );
        assert_eq!(with(|l| l.max_args = 0), Err(ProcessConfigError::ZeroArgs));
        assert_eq!(
            with(|l| l.max_arg_bytes = 0),
            Err(ProcessConfigError::ZeroArgBytes)
        );
        let ok = ProcessConfig::new(abs, limits).unwrap();
        for secret in [
            "API_KEY",
            "github_token",
            "DB_PASSWORD",
            "AWS_SECRET_ACCESS",
            "AUTH_HEADER",
        ] {
            assert_eq!(
                ok.clone().with_env(secret, "v"),
                Err(ProcessConfigError::SecretLikeEnvName(secret.into())),
                "{secret}"
            );
        }
        for bad in ["", "1A", "A-B", "A B"] {
            assert_eq!(
                ok.clone().with_env(bad, "v"),
                Err(ProcessConfigError::InvalidEnvName(bad.into()))
            );
        }
        assert_eq!(
            ok.clone().with_env("LANG", "a\0b"),
            Err(ProcessConfigError::InvalidEnvValue("LANG".into()))
        );
        let mut full = ok.clone();
        for i in 0..MAX_ENV_VARS {
            full = full.with_env(&format!("V{i}"), "x").unwrap();
        }
        assert_eq!(
            full.clone().with_env("EXTRA", "x"),
            Err(ProcessConfigError::TooManyEnvVars)
        );
        // Redefining a name replaces it and does not count twice.
        assert!(ok
            .with_env("LANG", "C")
            .unwrap()
            .with_env("LANG", "D")
            .is_ok());
    }

    #[test]
    fn verifier_is_named() {
        let config =
            ProcessConfig::new(vec![PathBuf::from("/bin/true")], ProcessLimits::default()).unwrap();
        assert_eq!(
            NamedVerifier::<CommandRequest>::name(&ProcessCommandVerifier::new(config)),
            "process-command"
        );
    }
}
