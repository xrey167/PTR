use std::fs;
use std::path::PathBuf;

use ptr_feedback::{
    read_log, rotated_path, Candidate, Critique, Decision, Generator, LogConfig, LogError, Outcome,
    ReadMode, RecordError, RepairBudget, RepairLoop, RepairPolicy, Repairer, TrajectoryLog,
    TrajectoryRecord, MAX_TEXT_CHARS,
};
use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "ptr-feedback-persistence-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Fails any text containing "bad".
struct BadWord;
impl Verifier<String> for BadWord {
    fn verify(&self, value: &String) -> VerificationReport {
        let bad = value.contains("bad");
        VerificationReport {
            status: if bad {
                VerificationStatus::Fail
            } else {
                VerificationStatus::Pass
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(if bad { 0.1 } else { 0.9 }).unwrap(),
            findings: if bad {
                vec![Finding {
                    code: "bad-text".into(),
                    message: String::new(),
                    hard: true,
                }]
            } else {
                vec![]
            },
        }
    }
}

struct Script(Vec<String>);
impl Generator<String> for Script {
    fn generate(&mut self) -> String {
        self.0.remove(0)
    }
}

struct Fix;
impl Repairer<String> for Fix {
    fn repair(&mut self, previous: &Candidate<String>, _: &Critique) -> Option<String> {
        Some(previous.value.replace("bad", "ok"))
    }
}

fn run_record(id: &str, first: &str) -> TrajectoryRecord {
    let policy = RepairPolicy {
        required_level: VerificationLevel::Deterministic,
        budget: RepairBudget {
            max_verifier_calls: 4,
            max_rounds: 4,
        },
    };
    let run = RepairLoop::new(&BadWord, policy)
        .unwrap()
        .run(&mut Script(vec![first.to_owned()]), &mut Fix);
    assert!(matches!(run.outcome, Outcome::Verified(_)));
    assert_eq!(run.trajectory.steps[0].decision, Decision::Repair);
    TrajectoryRecord::from_run(id, &run).unwrap()
}

fn open(dir: &TempDir, max_bytes: u64, keep: u32) -> TrajectoryLog {
    TrajectoryLog::open(LogConfig {
        path: dir.0.join("t.jsonl"),
        max_bytes,
        keep,
    })
    .unwrap()
}

#[test]
fn a_real_run_survives_write_and_read_unchanged() {
    let dir = TempDir::new("roundtrip");
    let log = open(&dir, 1 << 20, 3);
    let first = run_record("r1", "bad text with \"quotes\" and\nnewlines é 😀");
    let second = run_record("r2", "another bad one");
    log.append(&first).unwrap();
    log.append(&second).unwrap();
    let read = read_log(&log.config().path, ReadMode::Strict).unwrap();
    assert_eq!(read.records, [first, second]);
}

#[test]
fn secrets_never_reach_the_file() {
    let dir = TempDir::new("secrets");
    let log = open(&dir, 1 << 20, 1);
    let secret =
        "bad password=hunter2 key sk-abcdefghijklmnopqrstuvwxyz Authorization: Bearer tok123456";
    log.append(&run_record("r1", secret)).unwrap();
    let raw = fs::read_to_string(&log.config().path).unwrap();
    for leaked in ["hunter2", "sk-abcdefghijklmnop", "tok123456"] {
        assert!(!raw.contains(leaked), "{leaked} leaked");
    }
    assert!(raw.contains("[REDACTED]"));
}

#[test]
fn long_text_is_cut_and_flagged_and_the_line_stays_bounded() {
    let record = run_record("r1", &format!("bad {}", "x".repeat(MAX_TEXT_CHARS * 2)));
    assert!(record.steps[0].text_truncated);
    assert_eq!(record.steps[0].text.chars().count(), MAX_TEXT_CHARS);
    assert!(record.encode().unwrap().len() < 2 * MAX_TEXT_CHARS * 4);
}

#[test]
fn rotation_by_size_keeps_every_record_in_exactly_one_file() {
    let dir = TempDir::new("rotate");
    let sample = run_record("r00", "bad a").encode().unwrap().len() as u64 + 1;
    let log = open(&dir, sample * 3, 5);
    for i in 0..10 {
        log.append(&run_record(&format!("r{i:02}"), "bad a"))
            .unwrap();
    }
    let ids_of = |path: &std::path::Path| -> Vec<String> {
        read_log(path, ReadMode::Strict)
            .unwrap()
            .records
            .into_iter()
            .map(|r| r.run_id)
            .collect()
    };
    let mut ids: Vec<String> = Vec::new();
    for n in (1..=5).rev() {
        let path = rotated_path(&log.config().path, n);
        if path.exists() {
            ids.extend(ids_of(&path));
        }
    }
    ids.extend(ids_of(&log.config().path));
    let expected: Vec<String> = (0..10).map(|i| format!("r{i:02}")).collect();
    assert_eq!(ids, expected);
}

#[test]
fn corruption_is_a_typed_error_or_skipped() {
    let dir = TempDir::new("corrupt");
    let path = dir.0.join("t.jsonl");
    let good = run_record("r1", "bad a").encode().unwrap();
    let wrong_version = good.replace("\"v\":1", "\"v\":9");
    fs::write(
        &path,
        format!("{good}\n{wrong_version}\nnot json\n{good}\n"),
    )
    .unwrap();
    assert_eq!(
        read_log(&path, ReadMode::Strict).unwrap_err(),
        LogError::Corrupt {
            line: 2,
            error: RecordError::UnsupportedVersion(9)
        }
    );
    let lenient = read_log(&path, ReadMode::SkipCorrupt).unwrap();
    assert_eq!(lenient.records.len(), 2);
    assert_eq!(lenient.skipped.len(), 2);
    assert_eq!(lenient.skipped[0], (2, RecordError::UnsupportedVersion(9)));
    assert!(matches!(
        lenient.skipped[1],
        (3, RecordError::Syntax { .. })
    ));
}
