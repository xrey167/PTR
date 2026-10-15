//! Opt-in `ptrd` flags that build `ptr_server::ServerOptions`.
//!
//! `ptr_config::CliOverrides` rejects flags it does not know, so these are split off the
//! argument list first and the rest is handed to it unchanged. With none of them present the
//! result is `ServerOptions::default()`, i.e. today's behaviour.

use ptr_server::{IngressDiagnosticsMode, RequestTracingOptions, ServerOptions};
use std::path::PathBuf;

const TRACE_FILE: &str = "--trace-file";
const TRACE_MAX_BYTES: &str = "--trace-max-bytes";
const INGRESS_DIAGNOSTICS: &str = "--ingress-diagnostics";

/// The `--help` lines for these flags.
pub(crate) const USAGE: &str = "[--trace-file PATH [--trace-max-bytes N]] [--ingress-diagnostics]";

/// The parsed opt-in flags; `Default` is everything off.
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct ServerFlags {
    trace_file: Option<PathBuf>,
    trace_max_bytes: Option<u64>,
    ingress_diagnostics: bool,
}

impl ServerFlags {
    /// Removes the opt-in flags from `args` and returns them with the remaining arguments in
    /// their original order. Values are never echoed back in errors.
    pub(crate) fn split(
        args: impl IntoIterator<Item = String>,
    ) -> Result<(Self, Vec<String>), String> {
        let mut flags = Self::default();
        let mut rest = Vec::new();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                TRACE_FILE => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{TRACE_FILE} requires a value"))?;
                    if value.is_empty() {
                        return Err(format!("{TRACE_FILE} must not be empty"));
                    }
                    if flags.trace_file.replace(PathBuf::from(value)).is_some() {
                        return Err(format!("{TRACE_FILE} given more than once"));
                    }
                }
                TRACE_MAX_BYTES => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{TRACE_MAX_BYTES} requires a value"))?;
                    let bytes = value
                        .parse::<u64>()
                        .map_err(|_| format!("{TRACE_MAX_BYTES} must be a non-negative integer"))?;
                    if flags.trace_max_bytes.replace(bytes).is_some() {
                        return Err(format!("{TRACE_MAX_BYTES} given more than once"));
                    }
                }
                INGRESS_DIAGNOSTICS => flags.ingress_diagnostics = true,
                _ => rest.push(arg),
            }
        }
        Ok((flags, rest))
    }

    /// Builds the options. A zero cap or a cap without a trace file is refused here; the file
    /// itself is checked by `ServerState::with_options`.
    pub(crate) fn into_options(self) -> Result<ServerOptions, String> {
        let request_tracing = match (self.trace_file, self.trace_max_bytes) {
            (None, None) => None,
            (None, Some(_)) => {
                return Err(format!("{TRACE_MAX_BYTES} requires {TRACE_FILE}"));
            }
            (Some(path), cap) => {
                let mut tracing = RequestTracingOptions::new(path);
                if let Some(cap) = cap {
                    if cap == 0 {
                        return Err(format!("{TRACE_MAX_BYTES} must be non-zero"));
                    }
                    tracing.max_export_bytes = cap;
                }
                Some(tracing)
            }
        };
        Ok(ServerOptions {
            request_tracing,
            ingress_diagnostics: if self.ingress_diagnostics {
                IngressDiagnosticsMode::Advisory
            } else {
                IngressDiagnosticsMode::Off
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    fn options(list: &[&str]) -> Result<ServerOptions, String> {
        ServerFlags::split(args(list)).and_then(|(flags, _)| flags.into_options())
    }

    #[test]
    fn no_flags_give_default_options_and_untouched_arguments() {
        let input = args(&["--bind", "127.0.0.1:1", "--backend", "scenario"]);
        let (flags, rest) = ServerFlags::split(input.clone()).unwrap();
        assert_eq!(rest, input);
        assert_eq!(flags.into_options().unwrap(), ServerOptions::default());
    }

    #[test]
    fn trace_file_uses_the_default_cap() {
        let built = options(&["--trace-file", "t.jsonl"]).unwrap();
        assert_eq!(
            built.request_tracing,
            Some(RequestTracingOptions::new("t.jsonl"))
        );
        assert_eq!(built.ingress_diagnostics, IngressDiagnosticsMode::Off);
    }

    #[test]
    fn trace_cap_and_diagnostics_are_applied_and_other_args_pass_through() {
        let (flags, rest) = ServerFlags::split(args(&[
            "--bind",
            "x",
            "--trace-file",
            "t.jsonl",
            "--ingress-diagnostics",
            "--trace-max-bytes",
            "4096",
            "--mode",
            "standalone",
        ]))
        .unwrap();
        assert_eq!(rest, args(&["--bind", "x", "--mode", "standalone"]));
        let built = flags.into_options().unwrap();
        let tracing = built.request_tracing.unwrap();
        assert_eq!(tracing.export_path, PathBuf::from("t.jsonl"));
        assert_eq!(tracing.max_export_bytes, 4096);
        assert_eq!(built.ingress_diagnostics, IngressDiagnosticsMode::Advisory);
    }

    #[test]
    fn bad_values_are_errors() {
        for bad in [
            &["--trace-file"][..],
            &["--trace-file", ""],
            &["--trace-file", "a", "--trace-file", "b"],
            &["--trace-max-bytes"],
            &["--trace-file", "a", "--trace-max-bytes", "abc"],
            &["--trace-file", "a", "--trace-max-bytes", "-1"],
            &["--trace-file", "a", "--trace-max-bytes", "0"],
            &["--trace-max-bytes", "10"],
            &[
                "--trace-file",
                "a",
                "--trace-max-bytes",
                "1",
                "--trace-max-bytes",
                "2",
            ],
        ] {
            assert!(options(bad).is_err(), "{bad:?} must be refused");
        }
    }
}
