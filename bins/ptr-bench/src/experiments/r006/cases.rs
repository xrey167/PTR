//! The cases C1 to C11 of R006's draft preregistration. **Not built yet.**
//!
//! The cases need the position-independent secondary index and the positional session of
//! `ptr-runtime` (work packages 1 to 3 of the R006 plan), which do not exist in this tree. Until
//! they do, every entry point here returns [`HarnessError::NotBuilt`]: the harness refuses to
//! run, and nothing produces a metric, so no result can be mistaken for evidence.

use std::fmt;

/// The part of the harness that is not built yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// The cases C1 to C8, C10 and C11 (shift parity, revocation, supersession, refusals,
    /// controls and pins).
    Cases,
    /// The 52-cell invalidation matrix of C9.
    Matrix,
}

impl Part {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cases => "cases C1 to C8, C10 and C11",
            Self::Matrix => "matrix C9",
        }
    }
}

/// A failure of the harness itself (never a metric): the code the cases need is missing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarnessError {
    /// The part cannot run until the secondary index and the positional session exist in
    /// `ptr-runtime`.
    NotBuilt { part: Part },
}

impl fmt::Display for HarnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBuilt { part } => write!(
                formatter,
                "the R006 harness is not built yet: {} need the secondary index and the \
                 positional session of ptr-runtime; no metric is produced",
                part.name()
            ),
        }
    }
}

impl std::error::Error for HarnessError {}

/// Run every case of `seed`. Not built yet: always [`HarnessError::NotBuilt`].
pub fn run_all(_seed: u64) -> Result<(), HarnessError> {
    Err(HarnessError::NotBuilt { part: Part::Cases })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_the_cases_is_a_typed_not_built_error_and_no_metric() {
        let error = run_all(17).unwrap_err();
        assert_eq!(error, HarnessError::NotBuilt { part: Part::Cases });
        assert!(error.to_string().contains("not built yet"));
        assert!(error.to_string().contains("no metric is produced"));
    }

    #[test]
    fn the_parts_are_named_apart() {
        assert_ne!(Part::Cases.name(), Part::Matrix.name());
    }
}
