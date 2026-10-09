//! Candidates, critiques and the generate-verify-repair loop.
//!
//! The crate owns one acceptance rule, [`Candidate::passes`], and uses it everywhere a candidate
//! is accepted: [`RepairLoop`] returns a candidate only through it and
//! [`Population::best_verified`] selects only through it.

mod candidate;
mod repair;

pub use candidate::{Candidate, Critique, Population};
pub use repair::{
    Decision, ExhaustionReason, Generator, Origin, Outcome, RepairBudget, RepairError, RepairLoop,
    RepairPolicy, RepairRun, Repairer, RewriteReason, Step, Trajectory, Usage,
};
