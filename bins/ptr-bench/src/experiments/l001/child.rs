//! The child process of the process-abort arms: `ptr-bench revocation-crash-child <kind> <dir>
//! <workload-seed> <point> <hit>`.
//!
//! It rebuilds the workload from its seed, opens a ledger of `kind` ("file", "ack" or "engine")
//! in `dir`, appends the events one at a time and prints `ACK <n>` (flushed) after the n-th
//! append returned. A callback on the failpoint `point` counts its hits and calls
//! `std::process::abort()` at the `hit`-th (counting from 0): nothing is dropped, flushed or
//! unlocked. A child that gets to the end without aborting exits with status 3, which the
//! parent reads as a protocol error of the harness, as it does a malformed command line (status
//! 2). A product call that returns an error before the failpoint exits with status 4 and one
//! that panics ends the child with Rust's status 101; the parent counts both against the
//! product.

use super::file_arm::{key, log_path, paths_in};
use super::model::generate;
use super::params;
use crate::experiments::rng::Rng;
use ptr_ledger::acknowledged::AcknowledgedLedger;
use ptr_ledger::FileLedger;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn ack(count: usize) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "ACK {count}");
    let _ = out.flush();
}

/// A product call returned an error where the workload is valid: the product's defect.
fn product_error(reason: &str) -> ! {
    eprintln!("revocation-crash-child: {reason}");
    std::process::exit(4);
}

fn usage(reason: &str) -> ! {
    eprintln!("revocation-crash-child: {reason}");
    std::process::exit(2);
}

/// Run the child; never returns.
pub fn run(args: &[String]) -> ! {
    let [kind, dir, seed, point, hit] = args else {
        usage("usage: revocation-crash-child <kind> <dir> <workload-seed> <point> <hit>");
    };
    let dir = PathBuf::from(dir);
    let seed: u64 = seed
        .parse()
        .unwrap_or_else(|_| usage("the seed is not a number"));
    let hit: usize = hit
        .parse()
        .unwrap_or_else(|_| usage("the hit is not a number"));
    let workload = generate(&mut Rng::new(seed), params::EVENTS_MIN, params::EVENTS_MAX);
    // Armed once the ledger is open for the engine, whose own opening writes through the
    // same failpoints; the log and the set count from the first call, which `hit_of` knows.
    let arm = || {
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        fail::cfg_callback(point.clone(), move || {
            if hits.fetch_add(1, Ordering::SeqCst) == hit {
                std::process::abort();
            }
        })
        .unwrap_or_else(|reason| usage(&reason));
    };
    let events = workload.events.iter().map(|event| event.to_ledger());
    match kind.as_str() {
        "file" => {
            arm();
            let mut ledger = FileLedger::create_new(log_path(&dir))
                .unwrap_or_else(|error| product_error(&format!("cannot create the log: {error}")));
            for (i, event) in events.enumerate() {
                if ledger.append_durable(event).is_err() {
                    product_error("an append failed");
                }
                ack(i + 1);
            }
        }
        "ack" => {
            arm();
            let mut ledger = AcknowledgedLedger::create(&paths_in(&dir), key())
                .unwrap_or_else(|error| product_error(&format!("cannot create the set: {error}")));
            for (i, event) in events.enumerate() {
                if ledger.append_acknowledged(event).is_err() {
                    product_error("an append failed");
                }
                ack(i + 1);
            }
        }
        "engine" => {
            let mut ledger = ptr_ledger::RaftEngineLedger::open(&dir)
                .unwrap_or_else(|error| product_error(&format!("cannot open the engine: {error}")));
            arm();
            for (i, event) in events.enumerate() {
                if ledger.append_durable(event).is_err() {
                    product_error("an append failed");
                }
                ack(i + 1);
            }
        }
        other => usage(&format!("unknown kind {other:?}")),
    }
    // The failpoint was never reached.
    std::process::exit(3);
}
