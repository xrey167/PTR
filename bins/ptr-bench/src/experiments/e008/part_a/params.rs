//! The parameters of E008 Part A, compiled in. E008 is `planned`: nothing here is frozen, and
//! the head pins the thresholds in the preregistration before it leaves that status.

/// The experiment's manifest, read at compile time (the seeds come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../../experiments/system/E008-split-pod-extensions/experiment.toml");

pub const BENCHMARK: &str = "split-pod-extensions";

/// The invalidation paths of the matrix, in the order the cells print.
pub const PATHS: [&str; 3] = ["normal", "lag", "inflight"];

/// Every attempt and case counter must reach at least this many (a run below it is vacuous).
pub const MIN_CASES: i64 = 1;
/// A hard counter may not exceed this.
pub const MAX_VIOLATIONS: i64 = 0;
/// The cases of the seeded families (raw ancestry, modes, persistence) at the pinned scale; a
/// reduced run (`<cases> <seed>`) is no evidence.
pub const PINNED_CASES: u64 = 24;
/// The cells of the invalidation coverage matrix: the seven dependency kinds, each with a
/// revocation and a supersession, the expert with a fence besides (7 x 2 + 1 = 15 kind x type
/// pairs), times the three paths.
pub const MATRIX_CELLS: usize = 45;
/// Capacity of the downstream cache a case builds.
pub const CACHE_CAPACITY: usize = 8;
/// Width and rows of the synthetic activation frames.
pub const FRAME_ROWS: u32 = 4;
pub const FRAME_WIDTH: u32 = 8;

/// The seeds the manifest declares, read from its `seeds = [...]` line.
pub fn declared_seeds() -> Vec<u64> {
    EXPERIMENT_TOML
        .lines()
        .find(|line| line.trim_start().starts_with("seeds"))
        .and_then(|line| line.split_once('['))
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| {
            inner
                .split(',')
                .filter_map(|seed| seed.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}
