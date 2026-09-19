//! Confidence has an explicit semantic target, never runtime authority.

use crate::{Probability, ReasoningOperator, SemanticRole, TypeId};
use std::fmt;

/// The question answered by a confidence estimate, within its owning record.
///
/// Role/type/operator variants identify the proposed alternative. `Proposition`
/// refers to the proposition in the owning record, not to all claims from a source.
/// The owner must retain that proposition, evidence, revision and generation;
/// this enum alone is not a complete persisted observation or training label.
/// These variants are semantic names, not stable neural codebook IDs.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ConfidenceTarget {
    SemanticRole(SemanticRole),
    ValueType(TypeId),
    Proposition,
    Operator(ReasoningOperator),
}

/// A bounded probability attached to an explicit semantic question.
///
/// The estimate is reported data, not proof of calibration, verification,
/// lifecycle validity, requirement strength or permission. It deliberately has
/// no `Default` or ordering implementation: missing confidence is represented
/// with `Option<ConfidenceEstimate>`, and different questions cannot be ranked
/// by accidentally comparing the wrapper itself.
///
/// ```
/// use ptr_types::{ConfidenceEstimate, ConfidenceTarget, Probability, SemanticRole};
///
/// let estimate = ConfidenceEstimate::new(
///     ConfidenceTarget::SemanticRole(SemanticRole::Constraint),
///     Probability::new(0.8).expect("bounded example"),
/// );
/// assert_eq!(estimate.probability().get(), 0.8);
/// ```
///
/// An estimate cannot be implicitly promoted into a verification result:
///
/// ```compile_fail
/// use ptr_types::{ConfidenceEstimate, VerificationLevel};
/// fn promote(estimate: ConfidenceEstimate) -> VerificationLevel {
///     estimate.into()
/// }
/// ```
///
/// Nor can it grant an effect classification or permission:
///
/// ```compile_fail
/// use ptr_types::{ConfidenceEstimate, Effect};
/// fn authorize(estimate: ConfidenceEstimate) -> Effect {
///     estimate.into()
/// }
/// ```
///
/// Cross-target ordering requires an explicit policy outside this value type:
///
/// ```compile_fail
/// use ptr_types::ConfidenceEstimate;
/// fn rank(left: ConfidenceEstimate, right: ConfidenceEstimate) -> bool {
///     left > right
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ConfidenceEstimate {
    target: ConfidenceTarget,
    probability: Probability,
}

impl ConfidenceEstimate {
    pub fn new(target: ConfidenceTarget, probability: Probability) -> Self {
        Self {
            target,
            probability,
        }
    }

    pub fn target(&self) -> &ConfidenceTarget {
        &self.target
    }

    pub fn probability(&self) -> Probability {
        self.probability
    }

    /// Read a probability only for the expected question in the same owning record.
    /// This checks target identity, not calibration, truth or source validity.
    pub fn probability_for(
        &self,
        expected: &ConfidenceTarget,
    ) -> Result<Probability, ConfidenceTargetMismatch> {
        if &self.target == expected {
            Ok(self.probability)
        } else {
            Err(ConfidenceTargetMismatch {
                expected: expected.clone(),
                actual: self.target.clone(),
            })
        }
    }
}

/// A confidence estimate answered a different question from the one requested.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfidenceTargetMismatch {
    pub expected: ConfidenceTarget,
    pub actual: ConfidenceTarget,
}

impl ConfidenceTargetMismatch {
    pub const CODE: &'static str = "PTR_CONFIDENCE_TARGET_MISMATCH";
}

impl fmt::Display for ConfidenceTargetMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: expected {:?}, actual {:?}",
            Self::CODE,
            self.expected,
            self.actual
        )
    }
}

impl std::error::Error for ConfidenceTargetMismatch {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_preserves_target_and_probability() {
        let probability = Probability::new(0.25).expect("bounded fixture");
        let estimate = ConfidenceEstimate::new(
            ConfidenceTarget::Operator(ReasoningOperator::Symbolic),
            probability,
        );
        assert_eq!(estimate.probability(), probability);
        assert_eq!(
            estimate.target(),
            &ConfidenceTarget::Operator(ReasoningOperator::Symbolic)
        );
    }

    #[test]
    fn equal_numbers_do_not_erase_different_targets() {
        let probability = Probability::new(0.9).expect("bounded fixture");
        let role = ConfidenceEstimate::new(
            ConfidenceTarget::SemanticRole(SemanticRole::Claim),
            probability,
        );
        let truth = ConfidenceEstimate::new(ConfidenceTarget::Proposition, probability);
        assert_ne!(role, truth);
    }
}
