use ptr_config::PtrConfig;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "doctor".into());
    let code = match command.as_str() {
        "doctor" => {
            let root = args
                .next()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            doctor(&root)
        }
        "layout" => {
            println!("See docs/architecture, docs/components and README.md");
            0
        }
        _ => {
            eprintln!("usage: ptrctl [doctor [repo-root]|layout]");
            2
        }
    };
    std::process::exit(code);
}

fn doctor(root: &Path) -> i32 {
    let required = [
        "Cargo.toml",
        "Cargo.lock",
        "config/default.toml",
        "training/uv.lock",
        "experiments/registry.toml",
        "evaluations/registry.toml",
        "datasets/registry.toml",
        "docs/components/STATUS.md",
    ];
    let mut errors = Vec::new();

    for rel in required {
        if !root.join(rel).exists() {
            errors.push(format!("missing {rel}"));
        }
    }

    let config_path = root.join("config/default.toml");
    if config_path.exists() {
        match PtrConfig::from_path(&config_path).and_then(|config| {
            config.validate()?;
            Ok(config)
        }) {
            Ok(config) => {
                println!(
                    "config: OK mode={} mailbox={} max_parallel={}",
                    config.runtime.mode,
                    config.runtime.mailbox_capacity,
                    config.runtime.max_parallel_candidates
                );
            }
            Err(err) => errors.push(format!("invalid config/default.toml: {err}")),
        }
    }

    for (name, command, arg) in [
        ("git", "git", "--version"),
        ("rustc", "rustc", "--version"),
        ("python", "python", "--version"),
    ] {
        match Command::new(command).arg(arg).output() {
            Ok(output) if output.status.success() => {
                let text = if output.stdout.is_empty() {
                    String::from_utf8_lossy(&output.stderr)
                } else {
                    String::from_utf8_lossy(&output.stdout)
                };
                println!("{name}: {}", text.trim());
            }
            _ => println!("{name}: unavailable (optional for this doctor check)"),
        }
    }

    if errors.is_empty() {
        println!("PTR doctor: OK");
        0
    } else {
        for error in errors {
            eprintln!("PTR doctor: ERROR: {error}");
        }
        1
    }
}
