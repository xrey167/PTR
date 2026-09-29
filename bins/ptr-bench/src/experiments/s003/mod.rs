//! Experiment S003, certified branches: do dependency-certified agent branches
//! avoid lost updates and phantoms while merging more concurrent work than
//! serial execution? The harness runs in memory, with no PostgreSQL, in every
//! build; its parameters are the preregistered table in the experiment's
//! `config.toml` (`params`).

// The harness is assembled module by module; until `run` lands, the default
// build has nothing that calls into it.
#![allow(dead_code)]

pub mod canary;
pub mod model;
pub mod oracle;
pub mod params;
pub mod program;
pub mod verifier;
pub mod workload;
