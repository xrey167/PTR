//! Environment capture: what machine and tree produced a number.
//!
//! Every reader degrades to `None`/`"unknown"` instead of failing, and the
//! parsers are separate pure functions so they are tested on fixed text.

use crate::json::Value;
use std::path::Path;
use std::process::Command;

/// Facts about the machine and the tree a benchmark ran on.
#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    /// `git rev-parse HEAD` of the working directory, or `unknown`.
    pub git_sha: String,
    /// True when tracked files differ from HEAD (`git status --porcelain -uno`).
    pub git_dirty: Option<bool>,
    /// `rustc -V` output.
    pub rustc: String,
    /// CPU model name from `/proc/cpuinfo`.
    pub cpu_model: String,
    /// Logical CPUs visible to this process.
    pub cpu_threads: usize,
    /// `MemTotal` in KiB.
    pub mem_total_kib: Option<u64>,
    /// Kernel release (`/proc/sys/kernel/osrelease`).
    pub kernel: String,
    /// Filesystem type of the benchmark data directory (`stat -f -c %T`).
    pub fs_type: String,
    /// cpufreq governor of cpu0, when readable.
    pub cpu_governor: Option<String>,
}

/// First `model name` value in `/proc/cpuinfo` text.
pub fn parse_cpu_model(cpuinfo: &str) -> Option<String> {
    cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name")
            .then(|| value.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}

/// `MemTotal` in KiB from `/proc/meminfo` text.
pub fn parse_mem_total_kib(meminfo: &str) -> Option<u64> {
    crate::rss::parse_status_kib(meminfo, "MemTotal")
}

/// Clean the single-line output of `stat -f -c %T`: trimmed, never empty.
pub fn parse_fs_type(output: &str) -> Option<String> {
    let line = output.lines().next()?.trim();
    (!line.is_empty()).then(|| line.to_string())
}

/// Filesystem types for which durable-write numbers are not decision-grade:
/// memory-backed or layered filesystems whose `fsync` does not reach a device.
pub fn is_non_durable_fs(fs_type: &str) -> bool {
    matches!(
        fs_type.to_ascii_lowercase().as_str(),
        "tmpfs"
            | "overlay"
            | "overlayfs"
            | "ramfs"
            | "9p"
            | "virtiofs"
            | "fuseblk"
            | "fuse"
            | "unknown"
    )
}

/// The `decision_grade` flag of a hardware profile (`hardware/*.toml`), read from
/// its top-level `decision_grade = true|false` line. `None` when absent.
pub fn profile_decision_grade(profile_toml: &str) -> Option<bool> {
    profile_toml.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        if key.trim() != "decision_grade" {
            return None;
        }
        match value.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    })
}

fn run(program: &str, args: &[&str], dir: Option<&Path>) -> Option<String> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    let output = command.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Read the machine and tree facts. `data_dir` is where a benchmark writes its
/// data (its filesystem type is the one that matters for durability) and also
/// the directory `git` is asked about.
pub fn capture(data_dir: &Path) -> Environment {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let path = data_dir.to_string_lossy();
    Environment {
        git_sha: run("git", &["rev-parse", "HEAD"], Some(data_dir))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string()),
        git_dirty: run("git", &["status", "--porcelain", "-uno"], Some(data_dir))
            .map(|s| !s.is_empty()),
        rustc: run("rustc", &["-V"], None).unwrap_or_else(|| "unknown".to_string()),
        cpu_model: parse_cpu_model(&cpuinfo).unwrap_or_else(|| "unknown".to_string()),
        cpu_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        mem_total_kib: parse_mem_total_kib(&meminfo),
        kernel: std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| s.trim().to_string())
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string()),
        fs_type: run("stat", &["-f", "-c", "%T", &path], None)
            .and_then(|s| parse_fs_type(&s))
            .unwrap_or_else(|| "unknown".to_string()),
        cpu_governor: std::fs::read_to_string(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        )
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty()),
    }
}

impl Environment {
    /// The decision-grade question for durable-write benchmarks.
    pub fn durable_writes_decision_grade(&self) -> bool {
        !is_non_durable_fs(&self.fs_type)
    }

    /// Decision grade of a run: the filesystem must allow a durable claim and the
    /// hardware profile must not say otherwise (a missing flag counts as false).
    pub fn decision_grade(&self, profile_toml: &str) -> bool {
        self.durable_writes_decision_grade() && profile_decision_grade(profile_toml) == Some(true)
    }

    /// Encode as a JSON object (absent facts are `null`).
    pub fn to_value(&self) -> Value {
        Value::obj([
            ("git_sha", Value::str(&self.git_sha)),
            ("git_dirty", self.git_dirty.map_or(Value::Null, Value::Bool)),
            ("rustc", Value::str(&self.rustc)),
            ("cpu_model", Value::str(&self.cpu_model)),
            ("cpu_threads", Value::Int(self.cpu_threads as i64)),
            (
                "mem_total_kib",
                self.mem_total_kib
                    .map_or(Value::Null, |k| Value::Int(k as i64)),
            ),
            ("kernel", Value::str(&self.kernel)),
            ("fs_type", Value::str(&self.fs_type)),
            ("cpu_governor", Value::opt_str(self.cpu_governor.clone())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPUINFO: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Processor @ 2.80GHz\ncpu MHz\t\t: 2800.000\n\nprocessor\t: 1\nmodel name\t: Other\n";

    #[test]
    fn cpu_model_is_the_first_model_name() {
        assert_eq!(
            parse_cpu_model(CPUINFO).as_deref(),
            Some("Intel(R) Xeon(R) Processor @ 2.80GHz")
        );
        assert_eq!(parse_cpu_model("processor: 0\n"), None);
        assert_eq!(parse_cpu_model("model name\t:   \n"), None);
        assert_eq!(parse_cpu_model(""), None);
    }

    #[test]
    fn mem_total_parses_kib() {
        assert_eq!(
            parse_mem_total_kib("MemTotal:       16384000 kB\nMemFree: 1 kB\n"),
            Some(16_384_000)
        );
        assert_eq!(parse_mem_total_kib("MemFree: 1 kB\n"), None);
    }

    #[test]
    fn fs_type_takes_the_first_nonempty_line() {
        assert_eq!(parse_fs_type("ext2/ext3\n").as_deref(), Some("ext2/ext3"));
        assert_eq!(
            parse_fs_type("  overlayfs  \nextra\n").as_deref(),
            Some("overlayfs")
        );
        assert_eq!(parse_fs_type(""), None);
        assert_eq!(parse_fs_type("   \n"), None);
    }

    #[test]
    fn non_durable_filesystems_are_flagged() {
        for fs in [
            "tmpfs",
            "overlay",
            "overlayfs",
            "TMPFS",
            "ramfs",
            "9p",
            "unknown",
        ] {
            assert!(is_non_durable_fs(fs), "{fs}");
        }
        for fs in ["ext2/ext3", "ext4", "xfs", "btrfs", "zfs"] {
            assert!(!is_non_durable_fs(fs), "{fs}");
        }
    }

    fn sample(fs: &str) -> Environment {
        Environment {
            git_sha: "abc".into(),
            git_dirty: Some(false),
            rustc: "rustc 1.99.0".into(),
            cpu_model: "cpu".into(),
            cpu_threads: 4,
            mem_total_kib: Some(100),
            kernel: "6.1".into(),
            fs_type: fs.into(),
            cpu_governor: None,
        }
    }

    #[test]
    fn profile_flag_is_read_from_its_own_line_only() {
        assert_eq!(
            profile_decision_grade("id = \"x\"\ndecision_grade = false\n"),
            Some(false)
        );
        assert_eq!(profile_decision_grade("decision_grade=true"), Some(true));
        assert_eq!(
            profile_decision_grade("durable_writes_decision_grade = true\n"),
            None
        );
        assert_eq!(profile_decision_grade("# decision_grade = true\n"), None);
        assert_eq!(profile_decision_grade("decision_grade = \"yes\"\n"), None);
        assert_eq!(profile_decision_grade(""), None);
    }

    #[test]
    fn decision_grade_needs_fs_and_profile_to_agree() {
        let good = sample("xfs");
        assert!(good.decision_grade("decision_grade = true\n"));
        assert!(!good.decision_grade("decision_grade = false\n"));
        assert!(!good.decision_grade("id = \"no flag\"\n"));
        assert!(!sample("overlayfs").decision_grade("decision_grade = true\n"));
    }

    #[test]
    fn durability_grade_follows_the_filesystem() {
        assert!(sample("ext2/ext3").durable_writes_decision_grade());
        assert!(!sample("overlayfs").durable_writes_decision_grade());
    }

    #[test]
    fn to_value_has_every_field_and_nulls_for_absent_facts() {
        let v = sample("xfs").to_value();
        for key in [
            "git_sha",
            "git_dirty",
            "rustc",
            "cpu_model",
            "cpu_threads",
            "mem_total_kib",
            "kernel",
            "fs_type",
            "cpu_governor",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v.get("cpu_governor"), Some(&Value::Null));
        assert_eq!(v.get("cpu_threads"), Some(&Value::Int(4)));
        let mut none = sample("xfs");
        none.mem_total_kib = None;
        none.git_dirty = None;
        assert_eq!(none.to_value().get("mem_total_kib"), Some(&Value::Null));
        assert_eq!(none.to_value().get("git_dirty"), Some(&Value::Null));
    }

    #[test]
    fn capture_reads_this_machine() {
        let env = capture(Path::new(env!("CARGO_MANIFEST_DIR")));
        assert!(env.rustc.starts_with("rustc") || env.rustc == "unknown");
        assert!(env.cpu_threads >= 1);
        assert!(!env.git_sha.is_empty());
        assert!(!env.fs_type.is_empty());
        #[cfg(target_os = "linux")]
        {
            assert_ne!(env.kernel, "unknown");
            assert_ne!(env.cpu_model, "unknown");
            assert!(env.mem_total_kib.unwrap_or(0) > 0);
        }
    }

    #[test]
    fn capture_of_a_missing_directory_degrades_to_unknown() {
        let env = capture(Path::new("/nonexistent/evalkit/dir"));
        assert_eq!(env.git_sha, "unknown");
        assert_eq!(env.fs_type, "unknown");
        assert!(!env.durable_writes_decision_grade());
    }
}
