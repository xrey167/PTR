//! `ptr-worker` command line. Stdout is the protocol channel of `serve --stdio`; every
//! diagnostic goes to stderr and never contains call payloads.

use std::io::{self, Write};
use std::process::ExitCode;

use ptr_worker::pods::{builtin_registry, listing};
use ptr_worker::serve::{serve, ServeError, ServeLimits};
use ptr_worker::transport::StdioTransport;

const USAGE: &str = "\
ptr-worker: host for deterministic CPU reference Pods

usage:
  ptr-worker                 print this help
  ptr-worker pods            list the built-in Pods (name, contract summary)
  ptr-worker serve [--stdio] serve one control-protocol session on stdin/stdout

exit codes: 0 clean (or help/listing), 1 protocol, session or I/O failure,
2 usage error, 3 peer closed the stream without Bye
";

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Help,
    Pods,
    ServeStdio,
}

fn parse(args: &[String]) -> Result<Command, String> {
    match args {
        [] => Ok(Command::Help),
        [a] if matches!(a.as_str(), "help" | "-h" | "--help") => Ok(Command::Help),
        [a] if a == "pods" => Ok(Command::Pods),
        [a] if a == "serve" => Ok(Command::ServeStdio),
        [a, b] if a == "serve" && b == "--stdio" => Ok(Command::ServeStdio),
        [a, rest @ ..] if a == "serve" => Err(format!("unknown serve argument: {}", rest[0])),
        [a, ..] => Err(format!("unknown command: {a}")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Pods) => {
            print!("{}", listing());
            ExitCode::SUCCESS
        }
        Ok(Command::ServeStdio) => run_stdio(),
        Err(message) => {
            eprintln!("ptr-worker: {message}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run_stdio() -> ExitCode {
    let registry = match builtin_registry() {
        Ok(registry) => registry,
        Err(error) => {
            eprintln!("ptr-worker: cannot build the Pod registry: {error}");
            return ExitCode::from(1);
        }
    };
    let mut transport = StdioTransport::new(io::stdin().lock(), io::stdout().lock());
    let nonce = u64::from(std::process::id());
    let result = serve(&mut transport, &registry, ServeLimits::default(), nonce);
    let code = match result {
        Ok(summary) => {
            eprintln!(
                "ptr-worker: session closed; executed {} call(s), {} duplicate(s) from cache, \
                 {} cancelled",
                summary.calls_executed, summary.duplicates_served, summary.calls_cancelled
            );
            0
        }
        Err(ServeError::PeerVanished { during_handshake }) => {
            eprintln!(
                "ptr-worker: peer closed the stream without Bye (handshake: {during_handshake})"
            );
            3
        }
        Err(error) => {
            eprintln!("ptr-worker: session failed: {error}");
            1
        }
    };
    let _ = io::stderr().flush();
    ExitCode::from(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parses_commands() {
        assert_eq!(parse(&[]), Ok(Command::Help));
        assert_eq!(parse(&args(&["--help"])), Ok(Command::Help));
        assert_eq!(parse(&args(&["pods"])), Ok(Command::Pods));
        assert_eq!(parse(&args(&["serve"])), Ok(Command::ServeStdio));
        assert_eq!(parse(&args(&["serve", "--stdio"])), Ok(Command::ServeStdio));
        assert!(parse(&args(&["serve", "--tcp"])).is_err());
        assert!(parse(&args(&["bogus"])).is_err());
        assert!(parse(&args(&["pods", "extra"])).is_err());
    }
}
