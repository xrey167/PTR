//! Argument helpers shared by the opt-in subcommands (`ingest`, `events`, `trace`).
//!
//! Exit codes: [`EXIT_OK`] success, [`EXIT_INPUT`] invalid or corrupt input, [`EXIT_USAGE`]
//! usage or harness error.

use std::vec::IntoIter;

pub(crate) const EXIT_OK: i32 = 0;
pub(crate) const EXIT_INPUT: i32 = 1;
pub(crate) const EXIT_USAGE: i32 = 2;

pub(crate) const EXIT_CODES_HELP: &str = "exit codes:
  0  success
  1  invalid or corrupt input (unreadable, too large, damaged)
  2  usage or harness error";

/// Print a usage error to stderr and return [`EXIT_USAGE`].
pub(crate) fn usage_error(command: &str, message: &str, usage: &str) -> i32 {
    eprintln!("ptrctl {command}: {message}");
    eprintln!("{usage}");
    EXIT_USAGE
}

/// The value after `flag`, or a usage error code.
pub(crate) fn flag_value(
    rest: &mut IntoIter<String>,
    command: &str,
    flag: &str,
    usage: &str,
) -> Result<String, i32> {
    rest.next()
        .ok_or_else(|| usage_error(command, &format!("{flag} needs a value"), usage))
}

/// The value after `flag` parsed as an unsigned integer, or a usage error code.
pub(crate) fn flag_number<T: std::str::FromStr>(
    rest: &mut IntoIter<String>,
    command: &str,
    flag: &str,
    usage: &str,
) -> Result<T, i32> {
    let value = flag_value(rest, command, flag, usage)?;
    value.parse().map_err(|_| {
        usage_error(
            command,
            &format!("{flag} needs a non-negative integer, got {value:?}"),
            usage,
        )
    })
}

/// Whether `arg` asks for help.
pub(crate) fn is_help(arg: &str) -> bool {
    arg == "--help" || arg == "-h"
}
