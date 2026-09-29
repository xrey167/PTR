//! Experiment harnesses. The PostgreSQL ones (L003, L004) are compiled only
//! with the `postgres-experiments` feature and read the server's connection
//! string from `PTR_PG_EXPERIMENT_DSN`; the seeded generator and the JSON
//! helper every harness shares are always built.

// Only the PostgreSQL harnesses use the two shared modules until another
// harness builds without the feature.
#[cfg_attr(not(feature = "postgres-experiments"), allow(dead_code))]
mod json;
#[cfg(feature = "postgres-experiments")]
pub mod l003;
#[cfg(feature = "postgres-experiments")]
pub mod l004;
#[cfg(feature = "postgres-experiments")]
mod pg;
#[cfg_attr(not(feature = "postgres-experiments"), allow(dead_code))]
mod rng;
