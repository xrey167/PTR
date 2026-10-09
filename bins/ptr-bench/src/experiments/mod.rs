//! Experiment harnesses. The PostgreSQL ones (L003, L004) are compiled only
//! with the `postgres-experiments` feature and read the server's connection
//! string from `PTR_PG_EXPERIMENT_DSN`; S003 and R005 run in memory, and they and the
//! seeded generator and the JSON helper every harness shares are always built.

pub mod f001;
// Only the PostgreSQL harnesses use the two shared modules until another
// harness builds without the feature.
pub mod f002;
pub mod f003;
#[cfg_attr(not(feature = "postgres-experiments"), allow(dead_code))]
mod json;
pub mod kv_reuse_smoke;
#[cfg(feature = "postgres-experiments")]
pub mod l003;
#[cfg(feature = "postgres-experiments")]
pub mod l004;
pub mod layered_kv;
pub mod m008;
#[cfg(feature = "postgres-experiments")]
mod pg;
pub mod q002;
pub mod r001;
pub mod r005;
pub mod r006;
pub mod r007;
pub mod r009;
pub mod r010;
pub mod r011;
pub mod r012;
pub mod rag_bridge;
#[cfg_attr(not(feature = "postgres-experiments"), allow(dead_code))]
mod rng;
pub mod s001;
pub mod s002;
pub mod s003;
mod vtime;
