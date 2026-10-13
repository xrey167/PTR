//! The evidence record.
//!
//! [`EvidenceRecord::to_value`] emits exactly the fields, in the order, that
//! `scripts/run_component_eval.py` stores for a run (`base_record` then the run
//! fields), see [`RECORD_FIELDS`]. Two further fields carry what the protocol
//! needs: `protocol_sha256` (the hash of the component's `protocol.toml`, which
//! `scripts/check_component_protocols.py` compares) and `bench` (see
//! [`BenchSection`]). A record written here is a normal `-run.json` file next to
//! the ones the Python runner writes.

use crate::json::Value;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The fields of a runner record, in the runner's order.
pub const RECORD_FIELDS: [&str; 18] = [
    "schema_version",
    "component",
    "candidate",
    "criteria",
    "git_sha",
    "platform",
    "python",
    "candidate_manifest_sha256",
    "cargo_lock_sha256",
    "status",
    "started_at",
    "command",
    "finished_at",
    "exit_code",
    "stdout",
    "stderr",
    "launch_error",
    "duration_ns",
];

/// The two extension fields the protocol checker reads.
pub const EXTENSION_FIELDS: [&str; 2] = ["protocol_sha256", "bench"];

/// The runner's `status` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Exit code 0.
    Completed,
    /// A non-zero exit code.
    Failed,
    /// The process could not be started.
    FailedToLaunch,
}

impl RunStatus {
    /// The runner's spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::FailedToLaunch => "failed-to-launch",
        }
    }

    /// The runner's rule: `completed` iff the exit code is 0, `failed-to-launch`
    /// when there was no exit code.
    pub fn from_exit_code(exit_code: Option<i32>) -> Self {
        match exit_code {
            Some(0) => Self::Completed,
            Some(_) => Self::Failed,
            None => Self::FailedToLaunch,
        }
    }
}

/// One gate and whether it passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateResult {
    /// The gate id from `protocol.toml`.
    pub id: String,
    /// Whether the candidate passed it.
    pub passed: bool,
}

/// The `bench` section of a record.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchSection {
    /// The candidate version actually measured (from `Cargo.lock` or the binary).
    pub candidate_version: String,
    /// Gate outcomes.
    pub gates: Vec<GateResult>,
    /// True only when the timed benchmark ran to completion.
    pub benchmark_ran: bool,
    /// Warmup repetitions executed.
    pub warmup: u32,
    /// Measured repetitions executed.
    pub measured: u32,
    /// False when the environment cannot support the claim (for example durable
    /// writes on tmpfs/overlay); a `preferred` verdict then needs a re-run.
    pub decision_grade: bool,
    /// Why `decision_grade` is false (or a short note when true).
    pub decision_note: String,
    /// Named metric medians over the measured repetitions.
    pub metrics: Vec<(String, f64)>,
    /// The measurable proxies (`license_fit`, `integration_cost`, `portability`)
    /// as the protocol defines them, or `null` when not measured.
    pub proxies: Value,
    /// Where the metrics ran.
    pub environment: Value,
}

impl BenchSection {
    /// True when every recorded gate passed and at least one gate was recorded.
    pub fn gates_passed(&self) -> bool {
        !self.gates.is_empty() && self.gates.iter().all(|g| g.passed)
    }

    /// Encode as a JSON object.
    pub fn to_value(&self) -> Value {
        Value::obj([
            ("candidate_version", Value::str(&self.candidate_version)),
            (
                "gates",
                Value::Obj(
                    self.gates
                        .iter()
                        .map(|g| (g.id.clone(), Value::Bool(g.passed)))
                        .collect(),
                ),
            ),
            ("benchmark_ran", Value::Bool(self.benchmark_ran)),
            (
                "repetitions",
                Value::obj([
                    ("warmup", Value::Int(i64::from(self.warmup))),
                    ("measured", Value::Int(i64::from(self.measured))),
                ]),
            ),
            ("decision_grade", Value::Bool(self.decision_grade)),
            ("decision_note", Value::str(&self.decision_note)),
            (
                "metrics",
                Value::Obj(
                    self.metrics
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::Float(*v)))
                        .collect(),
                ),
            ),
            ("proxies", self.proxies.clone()),
            ("environment", self.environment.clone()),
        ])
    }
}

/// A complete evidence record.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceRecord {
    /// Component directory name.
    pub component: String,
    /// The candidate's `candidates.toml` table as JSON (must contain `id`).
    pub candidate: Value,
    /// The component's `[criteria]` weights.
    pub criteria: Vec<(String, f64)>,
    /// Repository HEAD.
    pub git_sha: String,
    /// `<system>-<kernel>-<arch>`, the runner's `platform.platform()` analogue.
    pub platform: String,
    /// SHA-256 of `candidates.toml`.
    pub candidate_manifest_sha256: Option<String>,
    /// SHA-256 of the root `Cargo.lock` (the runner's field; the bench
    /// workspace's own lock is named in the protocol).
    pub cargo_lock_sha256: Option<String>,
    /// `started_at` stamp (`Utc::utc_stamp`).
    pub started_at: String,
    /// `finished_at` (`Utc::isoformat`).
    pub finished_at: String,
    /// The command that was run.
    pub command: Vec<String>,
    /// Exit code, `None` if it never launched.
    pub exit_code: Option<i32>,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
    /// Launch error text.
    pub launch_error: Option<String>,
    /// Wall time of the command.
    pub duration_ns: u64,
    /// Hash of the component's `protocol.toml`.
    pub protocol_sha256: String,
    /// The benchmark section.
    pub bench: BenchSection,
}

impl EvidenceRecord {
    /// The candidate id (`candidate.id`).
    pub fn candidate_id(&self) -> Option<&str> {
        self.candidate.get("id").and_then(Value::as_str)
    }

    /// Encode with exactly [`RECORD_FIELDS`] in the runner's order, then the
    /// [`EXTENSION_FIELDS`].
    pub fn to_value(&self) -> Value {
        let status = RunStatus::from_exit_code(self.exit_code);
        Value::obj([
            ("schema_version", Value::Int(1)),
            ("component", Value::str(&self.component)),
            ("candidate", self.candidate.clone()),
            (
                "criteria",
                Value::Obj(
                    self.criteria
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::Float(*v)))
                        .collect(),
                ),
            ),
            ("git_sha", Value::str(&self.git_sha)),
            ("platform", Value::str(&self.platform)),
            ("python", Value::str("n/a (rust bench kit)")),
            (
                "candidate_manifest_sha256",
                Value::opt_str(self.candidate_manifest_sha256.clone()),
            ),
            (
                "cargo_lock_sha256",
                Value::opt_str(self.cargo_lock_sha256.clone()),
            ),
            ("status", Value::str(status.as_str())),
            ("started_at", Value::str(&self.started_at)),
            (
                "command",
                Value::Arr(self.command.iter().map(Value::str).collect()),
            ),
            ("finished_at", Value::str(&self.finished_at)),
            (
                "exit_code",
                self.exit_code
                    .map_or(Value::Null, |c| Value::Int(i64::from(c))),
            ),
            ("stdout", Value::str(&self.stdout)),
            ("stderr", Value::str(&self.stderr)),
            ("launch_error", Value::opt_str(self.launch_error.clone())),
            (
                "duration_ns",
                Value::Int(i64::try_from(self.duration_ns).unwrap_or(i64::MAX)),
            ),
            ("protocol_sha256", Value::str(&self.protocol_sha256)),
            ("bench", self.bench.to_value()),
        ])
    }

    /// The runner's file name: `<started_at>-<candidate>-run.json`.
    pub fn file_name(&self) -> Option<String> {
        self.candidate_id()
            .map(|id| format!("{}-{id}-run.json", self.started_at))
    }

    /// Write the record into `dir` (created if needed) exclusively, as the runner
    /// does: an existing file of the same name is an error, never overwritten.
    pub fn write_exclusive(&self, dir: &Path) -> std::io::Result<PathBuf> {
        let name = self.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "candidate has no string `id`",
            )
        })?;
        if name.contains(['/', '\\']) || name.starts_with('.') {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "candidate id is not a plain file-name component",
            ));
        }
        std::fs::create_dir_all(dir)?;
        let path = dir.join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(self.to_value().to_pretty().as_bytes())?;
        file.write_all(b"\n")?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench(gates: &[(&str, bool)]) -> BenchSection {
        BenchSection {
            candidate_version: "0.5.6".into(),
            gates: gates
                .iter()
                .map(|(id, passed)| GateResult {
                    id: (*id).into(),
                    passed: *passed,
                })
                .collect(),
            benchmark_ran: true,
            warmup: 1,
            measured: 5,
            decision_grade: true,
            decision_note: String::new(),
            metrics: vec![("alloc_p99_ns".into(), 120.0)],
            proxies: Value::Null,
            environment: Value::obj([("fs_type", Value::str("ext4"))]),
        }
    }

    fn record() -> EvidenceRecord {
        EvidenceRecord {
            component: "allocator".into(),
            candidate: Value::obj([
                ("id", Value::str("mimalloc")),
                ("source", Value::str("external")),
            ]),
            criteria: vec![("correctness".into(), 0.3), ("latency".into(), 0.7)],
            git_sha: "deadbeef".into(),
            platform: "Linux-6.1-x86_64".into(),
            candidate_manifest_sha256: Some("aa".into()),
            cargo_lock_sha256: None,
            started_at: "20261009T120000.000001Z".into(),
            finished_at: "2026-10-09T12:00:01+00:00".into(),
            command: vec!["cargo".into(), "run".into()],
            exit_code: Some(0),
            stdout: "ok\n".into(),
            stderr: String::new(),
            launch_error: None,
            duration_ns: 1_000_000_000,
            protocol_sha256: "ff".repeat(32),
            bench: bench(&[("digest", true), ("rss", true)]),
        }
    }

    #[test]
    fn the_record_has_exactly_the_runner_fields_in_order_then_the_extensions() {
        let value = record().to_value();
        let Value::Obj(pairs) = &value else {
            panic!("not an object")
        };
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        let expected: Vec<&str> = RECORD_FIELDS
            .iter()
            .chain(EXTENSION_FIELDS.iter())
            .copied()
            .collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn the_runner_field_list_matches_run_component_eval_py() {
        // base_record(...) keys then the run keys, as written in the Python runner.
        let python = include_str!("../../../../scripts/run_component_eval.py");
        for field in RECORD_FIELDS {
            assert!(
                python.contains(&format!("\"{field}\"")),
                "run_component_eval.py never mentions {field}"
            );
        }
    }

    #[test]
    fn status_follows_the_exit_code_like_the_runner() {
        assert_eq!(RunStatus::from_exit_code(Some(0)), RunStatus::Completed);
        assert_eq!(RunStatus::from_exit_code(Some(1)), RunStatus::Failed);
        assert_eq!(RunStatus::from_exit_code(Some(-1)), RunStatus::Failed);
        assert_eq!(RunStatus::from_exit_code(None), RunStatus::FailedToLaunch);
        assert_eq!(RunStatus::Completed.as_str(), "completed");
        assert_eq!(RunStatus::Failed.as_str(), "failed");
        assert_eq!(RunStatus::FailedToLaunch.as_str(), "failed-to-launch");
        let mut failed = record();
        failed.exit_code = None;
        assert_eq!(
            failed.to_value().get("status"),
            Some(&Value::str("failed-to-launch"))
        );
        assert_eq!(failed.to_value().get("exit_code"), Some(&Value::Null));
    }

    #[test]
    fn the_record_round_trips_through_the_parser() {
        let value = record().to_value();
        assert_eq!(Value::parse(&value.to_pretty()).unwrap(), value);
        assert_eq!(value.get("duration_ns"), Some(&Value::Int(1_000_000_000)));
        assert_eq!(value.get("cargo_lock_sha256"), Some(&Value::Null));
        assert_eq!(
            value.get("protocol_sha256"),
            Some(&Value::str("ff".repeat(32)))
        );
    }

    #[test]
    fn bench_section_carries_gates_repetitions_and_grade() {
        let section = record().bench.to_value();
        assert_eq!(
            section.get("gates").and_then(|g| g.get("digest")),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            section.get("repetitions").and_then(|r| r.get("measured")),
            Some(&Value::Int(5))
        );
        assert_eq!(section.get("benchmark_ran"), Some(&Value::Bool(true)));
        assert_eq!(section.get("decision_grade"), Some(&Value::Bool(true)));
        assert_eq!(section.get("candidate_version"), Some(&Value::str("0.5.6")));
        assert_eq!(section.get("proxies"), Some(&Value::Null));
    }

    #[test]
    fn gates_passed_needs_at_least_one_gate_and_all_passing() {
        assert!(bench(&[("a", true), ("b", true)]).gates_passed());
        assert!(!bench(&[("a", true), ("b", false)]).gates_passed());
        assert!(!bench(&[]).gates_passed());
    }

    #[test]
    fn file_name_uses_stamp_and_candidate() {
        assert_eq!(
            record().file_name().as_deref(),
            Some("20261009T120000.000001Z-mimalloc-run.json")
        );
        let mut anonymous = record();
        anonymous.candidate = Value::obj::<&str>([]);
        assert_eq!(anonymous.file_name(), None);
        assert_eq!(anonymous.candidate_id(), None);
    }

    #[test]
    fn write_exclusive_creates_once_and_refuses_overwrite() {
        let dir = std::env::temp_dir().join(format!("evalkit-ev-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = record();
        let path = rec.write_exclusive(&dir.join("evidence")).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("}\n"));
        assert_eq!(Value::parse(&text).unwrap(), rec.to_value());
        let again = rec.write_exclusive(&dir.join("evidence")).unwrap_err();
        assert_eq!(again.kind(), std::io::ErrorKind::AlreadyExists);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_exclusive_refuses_ids_that_escape_the_directory() {
        let dir = std::env::temp_dir().join(format!("evalkit-ev-esc-{}", std::process::id()));
        let mut rec = record();
        rec.candidate = Value::obj([("id", Value::str("../evil"))]);
        let err = rec.write_exclusive(&dir).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        rec.candidate = Value::obj::<&str>([]);
        assert_eq!(
            rec.write_exclusive(&dir).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(!dir.exists());
    }
}
