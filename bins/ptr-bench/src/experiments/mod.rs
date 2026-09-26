//! Experiment harnesses that need a PostgreSQL server. They are compiled only
//! with the `postgres-experiments` feature and read the server's connection
//! string from `PTR_PG_EXPERIMENT_DSN`.

pub mod l003;
pub mod l004;
mod pg;
mod rng;
