use std::fmt;

/// Probability with which the routing policy sampled the action it took, given the same inputs.
/// Only the trusted routing path produces it, as the quantized probability (interval width over 2^53) of the sampled pod.
///
/// A propensity is logged next to every routing decision so a different policy can later be
/// evaluated off-policy by inverse-propensity weighting. It is the probability of the action
/// actually *taken*. That differs from `LoggedTriage::auto_propensity` in `ptr-branch`, which is
/// the probability of one fixed action (auto-propose) whatever was taken, may be 0 for an
/// ineligible record, and gives the other action's probability as its complement; the two must
/// not be exchanged. Weighting divides by this value, so the range is `(0, 1]`: zero is excluded
/// because a router can never have chosen, with probability zero, an action it actually took.
/// Values whose reciprocal is not finite (roughly everything below `1 / f64::MAX`, such as the
/// smallest subnormals) are rejected too, so the weight `1 / p` is always finite;
/// `f64::MIN_POSITIVE` stays accepted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Propensity(f64);

impl Propensity {
    /// The propensity of a deterministic router.
    pub const CERTAIN: Self = Self(1.0);

    /// Validates `value` as finite, in `(0, 1]`, and with a finite reciprocal.
    pub fn new(value: f64) -> Result<Self, PropensityError> {
        if value.is_finite() && value > 0.0 && value <= 1.0 && (1.0 / value).is_finite() {
            Ok(Self(value))
        } else {
            Err(PropensityError { value })
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

/// A rejected propensity value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropensityError {
    value: f64,
}

impl PropensityError {
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl fmt::Display for PropensityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "propensity {} is not a finite number in (0, 1] with a finite reciprocal",
            self.value
        )
    }
}

impl std::error::Error for PropensityError {}
