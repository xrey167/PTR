//! Shared measurement kit for the component-evaluation benchmarks.
//!
//! Std only, on purpose: the kit must not add a candidate (or anything with an
//! advisory) to the graph of every benchmark workspace that depends on it. The
//! modules are small and independent:
//!
//! * [`rng`] and [`gen`]: seeded, library-version-independent generators.
//! * [`stats`]: nearest-rank quantiles, latency summaries (>= 10k samples),
//!   repetition protocol (warmup + at least 5 measured runs).
//! * [`rss`]: resident-set size from `/proc/self/status`.
//! * [`envcap`]: git sha, rustc, CPU, kernel, filesystem type, governor.
//! * [`evidence`]: the evidence record `scripts/run_component_eval.py` stores,
//!   plus the benchmark section the protocol checker reads.
//! * [`json`], [`sha256`], [`clock`]: the small parts evidence needs.

pub mod clock;
pub mod envcap;
pub mod evidence;
pub mod gen;
pub mod json;
pub mod rng;
pub mod rss;
pub mod sha256;
pub mod stats;
