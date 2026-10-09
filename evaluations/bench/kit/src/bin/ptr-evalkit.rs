//! `ptr-evalkit`: small command-line front end of the measurement kit.
//!
//! * `ptr-evalkit env [DIR]`: print the captured environment as JSON.
//! * `ptr-evalkit hardware ID [DIR]`: print a measured hardware profile (TOML).
//! * `ptr-evalkit sha256 FILE`: print a file's SHA-256.
//! * `ptr-evalkit validate FILE`: parse a JSON evidence record and check that it
//!   carries every runner field.

use evalkit::envcap::{capture, Environment};
use evalkit::evidence::{EXTENSION_FIELDS, RECORD_FIELDS};
use evalkit::json::Value;
use evalkit::{rss, sha256};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn dir_arg(arg: Option<&String>) -> PathBuf {
    arg.map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// TOML profile text from a captured environment (strings quoted conservatively).
fn hardware_profile(id: &str, env: &Environment) -> String {
    let quote = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let ram_gib = env
        .mem_total_kib
        .map_or(0.0, |k| (k as f64 / 1024.0 / 1024.0 * 10.0).round() / 10.0);
    format!(
        "id = {}\nos = {}\ncpu = {}\ncpu_threads = {}\nram_gib = {}\nfilesystem = {}\ncpu_governor = {}\ndurable_writes_decision_grade = {}\ngpu = \"none\"\ngpu_memory_gib = 0\n",
        quote(id),
        quote(&format!("Linux {}", env.kernel)),
        quote(&env.cpu_model),
        env.cpu_threads,
        ram_gib,
        quote(&env.fs_type),
        quote(env.cpu_governor.as_deref().unwrap_or("not readable")),
        env.durable_writes_decision_grade(),
    )
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("env") => {
            println!("{}", capture(&dir_arg(args.get(1))).to_value().to_pretty());
            ExitCode::SUCCESS
        }
        Some("hardware") => {
            let Some(id) = args.get(1) else {
                eprintln!("usage: ptr-evalkit hardware ID [DIR]");
                return ExitCode::from(2);
            };
            print!("{}", hardware_profile(id, &capture(&dir_arg(args.get(2)))));
            ExitCode::SUCCESS
        }
        Some("sha256") => match args.get(1).and_then(|p| sha256::hex_of_file(Path::new(p))) {
            Some(hex) => {
                println!("{hex}");
                ExitCode::SUCCESS
            }
            None => {
                eprintln!("usage: ptr-evalkit sha256 FILE (readable)");
                ExitCode::from(2)
            }
        },
        Some("validate") => {
            let Some(text) = args.get(1).and_then(|p| std::fs::read_to_string(p).ok()) else {
                eprintln!("usage: ptr-evalkit validate FILE (readable)");
                return ExitCode::from(2);
            };
            match Value::parse(&text) {
                Ok(value) => {
                    let missing: Vec<&str> = RECORD_FIELDS
                        .iter()
                        .chain(EXTENSION_FIELDS.iter())
                        .copied()
                        .filter(|f| value.get(f).is_none())
                        .collect();
                    if missing.is_empty() {
                        println!("ok");
                        ExitCode::SUCCESS
                    } else {
                        eprintln!("missing fields: {missing:?}");
                        ExitCode::FAILURE
                    }
                }
                Err(error) => {
                    eprintln!("not JSON: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("rss") => {
            println!(
                "peak_kib={:?} current_kib={:?}",
                rss::peak_rss_kib(),
                rss::current_rss_kib()
            );
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: ptr-evalkit env|hardware|sha256|validate|rss ...");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_profile_is_valid_looking_toml_with_the_grade() {
        let env = Environment {
            git_sha: "x".into(),
            git_dirty: None,
            rustc: "r".into(),
            cpu_model: "Some \"CPU\"".into(),
            cpu_threads: 4,
            mem_total_kib: Some(16_384_000),
            kernel: "6.1".into(),
            fs_type: "overlayfs".into(),
            cpu_governor: None,
        };
        let toml = hardware_profile("cpu-4core-15gib", &env);
        assert!(toml.contains("id = \"cpu-4core-15gib\"\n"));
        assert!(toml.contains("cpu = \"Some \\\"CPU\\\"\"\n"));
        assert!(toml.contains("ram_gib = 15.6\n"));
        assert!(toml.contains("cpu_governor = \"not readable\"\n"));
        assert!(toml.contains("durable_writes_decision_grade = false\n"));
    }
}
