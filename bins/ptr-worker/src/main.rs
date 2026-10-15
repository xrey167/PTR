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
  ptr-worker serve --channel-fd <n> [--confined]
                             serve one session over the inherited SOCK_SEQPACKET channel
                             (data-plane builds, Linux): --confined installs the seccomp
                             allowlist before anything else
  ptr-worker guardian --channel-fd <n> --backend fake|cuda
                             hold the staging buffers of the CUDA carriers for the supervisor
                             (data-plane builds, Linux); serves only that channel and is not
                             seccomp-confined

exit codes: 0 clean (or help/listing), 1 protocol, session or I/O failure,
2 usage error, 3 peer closed the stream without Bye
";

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Help,
    Pods,
    ServeStdio,
    /// `serve --channel-fd <n> [--confined] [--fault <name>]`.
    ServeChannel {
        fd: i32,
        confined: bool,
        fault: Option<String>,
    },
    /// `guardian --channel-fd <n> --backend <name>`.
    Guardian {
        fd: i32,
        backend: String,
    },
}

fn parse(args: &[String]) -> Result<Command, String> {
    match args {
        [] => Ok(Command::Help),
        [a] if matches!(a.as_str(), "help" | "-h" | "--help") => Ok(Command::Help),
        [a] if a == "pods" => Ok(Command::Pods),
        [a] if a == "serve" => Ok(Command::ServeStdio),
        [a, b] if a == "serve" && b == "--stdio" => Ok(Command::ServeStdio),
        [a, rest @ ..] if a == "serve" && rest.iter().any(|arg| arg == "--channel-fd") => {
            parse_channel(rest)
        }
        [a, rest @ ..] if a == "serve" => Err(format!("unknown serve argument: {}", rest[0])),
        [a, rest @ ..] if a == "guardian" => parse_guardian(rest),
        [a, ..] => Err(format!("unknown command: {a}")),
    }
}

fn parse_guardian(args: &[String]) -> Result<Command, String> {
    let mut fd = None;
    let mut backend = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--channel-fd" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--channel-fd needs a descriptor number".to_owned())?;
                let number: i32 = value
                    .parse()
                    .map_err(|_| format!("not a descriptor number: {value}"))?;
                if number < 3 {
                    return Err("the channel descriptor must be 3 or higher".to_owned());
                }
                fd = Some(number);
            }
            "--backend" => {
                backend = Some(
                    iter.next()
                        .ok_or_else(|| "--backend needs a name (fake or cuda)".to_owned())?
                        .clone(),
                );
            }
            other => return Err(format!("unknown guardian argument: {other}")),
        }
    }
    match (fd, backend) {
        (Some(fd), Some(backend)) => Ok(Command::Guardian { fd, backend }),
        (None, _) => Err("--channel-fd is required".to_owned()),
        (_, None) => Err("--backend is required (fake or cuda)".to_owned()),
    }
}

fn parse_channel(args: &[String]) -> Result<Command, String> {
    let mut fd = None;
    let mut confined = false;
    let mut fault = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--channel-fd" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--channel-fd needs a descriptor number".to_owned())?;
                let number: i32 = value
                    .parse()
                    .map_err(|_| format!("not a descriptor number: {value}"))?;
                if number < 3 {
                    return Err("the channel descriptor must be 3 or higher".to_owned());
                }
                fd = Some(number);
            }
            "--confined" => confined = true,
            "--fault" => {
                fault = Some(
                    iter.next()
                        .ok_or_else(|| "--fault needs a name".to_owned())?
                        .clone(),
                );
            }
            other => return Err(format!("unknown serve argument: {other}")),
        }
    }
    match fd {
        Some(fd) => Ok(Command::ServeChannel {
            fd,
            confined,
            fault,
        }),
        None => Err("--channel-fd is required".to_owned()),
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
        Ok(Command::ServeChannel {
            fd,
            confined,
            fault,
        }) => run_channel(fd, confined, fault.as_deref()),
        Ok(Command::Guardian { fd, backend }) => run_guardian_command(fd, &backend),
        Err(message) => {
            eprintln!("ptr-worker: {message}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(all(feature = "data-plane", target_os = "linux"))]
fn run_channel(fd: i32, confined: bool, fault: Option<&str>) -> ExitCode {
    use ptr_fabric::Channel;
    use ptr_worker::channel_serve::{serve_channel, ChannelServeConfig, ChannelServeError};

    #[cfg(feature = "fault-injection")]
    let fault = match fault.map(|name| (name, ptr_worker::channel_serve::Fault::parse(name))) {
        None => None,
        Some((_, Some(fault))) => Some(fault),
        Some((name, None)) => {
            eprintln!("ptr-worker: unknown fault: {name}");
            return ExitCode::from(2);
        }
    };
    #[cfg(not(feature = "fault-injection"))]
    if let Some(name) = fault {
        eprintln!("ptr-worker: this build has no fault injection (asked for {name})");
        return ExitCode::from(2);
    }
    // SAFETY: the supervisor started this process with the channel at descriptor `fd` and
    // nothing else in the process owns it; the channel takes ownership.
    let channel = match unsafe { Channel::from_inherited_fd(fd) } {
        Ok(channel) => channel,
        Err(error) => {
            eprintln!("ptr-worker: cannot use descriptor {fd} as the channel: {error}");
            return ExitCode::from(1);
        }
    };
    let config = ChannelServeConfig {
        confined,
        #[cfg(feature = "fault-injection")]
        fault,
        ..ChannelServeConfig::default()
    };
    let code = match serve_channel(channel, config) {
        Ok(summary) => {
            eprintln!(
                "ptr-worker: session closed; executed {} call(s), {} duplicate(s) from cache",
                summary.calls_executed, summary.duplicates_served
            );
            0
        }
        Err(ChannelServeError::PeerVanished { during_handshake }) => {
            eprintln!(
                "ptr-worker: supervisor closed the channel without Bye (handshake: {during_handshake})"
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

#[cfg(all(feature = "data-plane", target_os = "linux"))]
fn run_guardian_command(fd: i32, backend: &str) -> ExitCode {
    use ptr_worker::guardian::{run_guardian, BackendChoice};

    let Some(choice) = BackendChoice::parse(backend) else {
        eprintln!("ptr-worker: unknown guardian backend: {backend} (fake or cuda)");
        return ExitCode::from(2);
    };
    if choice == BackendChoice::Fake {
        eprintln!(
            "ptr-worker: WARNING: guardian started with --backend fake, the CPU test double; it \
             holds no device memory and is not for production"
        );
    }
    let code = match run_guardian(fd, choice) {
        Ok(summary) => {
            eprintln!(
                "ptr-worker: guardian closed; {} request(s), {} buffer(s) freed at exit",
                summary.requests, summary.freed_at_exit
            );
            0
        }
        Err(error) => {
            eprintln!("ptr-worker: guardian failed: {error}");
            1
        }
    };
    let _ = io::stderr().flush();
    ExitCode::from(code)
}

#[cfg(not(all(feature = "data-plane", target_os = "linux")))]
fn run_guardian_command(_fd: i32, _backend: &str) -> ExitCode {
    eprintln!("ptr-worker: this build has no data plane (Linux, feature data-plane)");
    ExitCode::from(2)
}

#[cfg(not(all(feature = "data-plane", target_os = "linux")))]
fn run_channel(_fd: i32, _confined: bool, _fault: Option<&str>) -> ExitCode {
    eprintln!("ptr-worker: this build has no data plane (Linux, feature data-plane)");
    ExitCode::from(2)
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
        assert_eq!(
            parse(&args(&["serve", "--confined", "--channel-fd", "3"])),
            Ok(Command::ServeChannel {
                fd: 3,
                confined: true,
                fault: None
            })
        );
        assert_eq!(
            parse(&args(&["serve", "--channel-fd", "7", "--fault", "silent"])),
            Ok(Command::ServeChannel {
                fd: 7,
                confined: false,
                fault: Some("silent".to_owned())
            })
        );
        assert!(parse(&args(&["serve", "--channel-fd"])).is_err());
        assert!(parse(&args(&["serve", "--channel-fd", "2"])).is_err());
        assert!(parse(&args(&["serve", "--channel-fd", "x"])).is_err());
        assert!(parse(&args(&["serve", "--channel-fd", "3", "--tcp"])).is_err());
        assert_eq!(
            parse(&args(&[
                "guardian",
                "--channel-fd",
                "3",
                "--backend",
                "fake"
            ])),
            Ok(Command::Guardian {
                fd: 3,
                backend: "fake".to_owned()
            })
        );
        assert!(parse(&args(&["guardian", "--channel-fd", "3"])).is_err());
        assert!(parse(&args(&["guardian", "--backend", "fake"])).is_err());
        assert!(parse(&args(&[
            "guardian",
            "--channel-fd",
            "2",
            "--backend",
            "fake"
        ]))
        .is_err());
        assert!(parse(&args(&["guardian", "--channel-fd", "3", "--tcp"])).is_err());
        assert!(parse(&args(&["bogus"])).is_err());
        assert!(parse(&args(&["pods", "extra"])).is_err());
    }
}
