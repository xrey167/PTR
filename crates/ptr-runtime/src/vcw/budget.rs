//! Per-request budgets of the verifier-closed loop.
//!
//! A request carries two typed budgets: the number of verifier calls (one per candidate) and the
//! number of rounds (attempts to produce a candidate after the first). Both are positive and
//! bounded by a crate maximum, and a request that spends either without an accepted candidate
//! ends `Exhausted`; no failed candidate is returned in its place.

use std::fmt;
use std::num::NonZeroU32;

/// Largest verifier-call budget a request may carry.
pub const MAX_VERIFIER_CALLS: u32 = 256;
/// Largest round budget a request may carry.
pub const MAX_ROUNDS: u32 = 256;

/// Why a budget was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BudgetError {
    ZeroVerifierCalls,
    ZeroRounds,
    TooManyVerifierCalls { requested: u32, max: u32 },
    TooManyRounds { requested: u32, max: u32 },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVerifierCalls => f.write_str("the budget allows no verifier call"),
            Self::ZeroRounds => f.write_str("the budget allows no round"),
            Self::TooManyVerifierCalls { requested, max } => {
                write!(f, "{requested} verifier calls requested, at most {max}")
            }
            Self::TooManyRounds { requested, max } => {
                write!(f, "{requested} rounds requested, at most {max}")
            }
        }
    }
}

impl std::error::Error for BudgetError {}

/// The most verifier calls one request may spend. Each candidate is verified exactly once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifierCallBudget(NonZeroU32);

impl VerifierCallBudget {
    /// # Errors
    /// [`BudgetError`] for zero or more than [`MAX_VERIFIER_CALLS`].
    pub fn new(calls: u32) -> Result<Self, BudgetError> {
        let calls_nz = NonZeroU32::new(calls).ok_or(BudgetError::ZeroVerifierCalls)?;
        if calls > MAX_VERIFIER_CALLS {
            return Err(BudgetError::TooManyVerifierCalls {
                requested: calls,
                max: MAX_VERIFIER_CALLS,
            });
        }
        Ok(Self(calls_nz))
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// The most attempts after the first candidate one request may spend (repairs, regenerations
/// and escalations, a failed attempt included).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundBudget(NonZeroU32);

impl RoundBudget {
    /// # Errors
    /// [`BudgetError`] for zero or more than [`MAX_ROUNDS`].
    pub fn new(rounds: u32) -> Result<Self, BudgetError> {
        let rounds_nz = NonZeroU32::new(rounds).ok_or(BudgetError::ZeroRounds)?;
        if rounds > MAX_ROUNDS {
            return Err(BudgetError::TooManyRounds {
                requested: rounds,
                max: MAX_ROUNDS,
            });
        }
        Ok(Self(rounds_nz))
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// Both budgets of one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestBudget {
    pub verifier_calls: VerifierCallBudget,
    pub rounds: RoundBudget,
}

impl RequestBudget {
    /// # Errors
    /// [`BudgetError`] when either number is zero or above its maximum.
    pub fn new(verifier_calls: u32, rounds: u32) -> Result<Self, BudgetError> {
        Ok(Self {
            verifier_calls: VerifierCallBudget::new(verifier_calls)?,
            rounds: RoundBudget::new(rounds)?,
        })
    }
}

/// What a request has spent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetUse {
    /// Verifier calls, one per candidate.
    pub verifier_calls: u32,
    /// Attempts after the first, a failed one included.
    pub rounds: u32,
    /// Calls into role Pods (proposals, repairs, escalations).
    pub pod_calls: u32,
}

/// Which budget ended a request (the type of `ptr-feedback`'s loop).
pub use ptr_feedback::ExhaustionReason;

/// The meter of one request. The only way to call the verifier or to start another attempt is to
/// charge it first, so a spent budget cannot be exceeded by a code path that forgets to check.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BudgetMeter {
    budget: RequestBudget,
    used: BudgetUse,
}

impl BudgetMeter {
    pub(crate) fn new(budget: RequestBudget) -> Self {
        Self {
            budget,
            used: BudgetUse::default(),
        }
    }

    pub(crate) fn used(&self) -> BudgetUse {
        self.used
    }

    pub(crate) fn budget(&self) -> RequestBudget {
        self.budget
    }

    /// Verifier calls still available.
    pub(crate) fn calls_left(&self) -> u32 {
        self.budget
            .verifier_calls
            .get()
            .saturating_sub(self.used.verifier_calls)
    }

    /// Charge one verifier call.
    pub(crate) fn charge_verifier(&mut self) -> Result<(), ExhaustionReason> {
        if self.used.verifier_calls >= self.budget.verifier_calls.get() {
            return Err(ExhaustionReason::VerifierCalls);
        }
        self.used.verifier_calls += 1;
        Ok(())
    }

    /// Charge one attempt after the first.
    pub(crate) fn charge_round(&mut self) -> Result<(), ExhaustionReason> {
        if self.used.rounds >= self.budget.rounds.get() {
            return Err(ExhaustionReason::Rounds);
        }
        self.used.rounds += 1;
        Ok(())
    }

    pub(crate) fn count_pod_call(&mut self) {
        self.used.pod_calls = self.used.pod_calls.saturating_add(1);
    }

    /// The limit that ends a request after a rejected verdict, in the order of `ptr-feedback`'s
    /// loop: verifier calls first, then rounds.
    pub(crate) fn stop_after_verdict(&self) -> Option<ExhaustionReason> {
        if self.used.verifier_calls >= self.budget.verifier_calls.get() {
            Some(ExhaustionReason::VerifierCalls)
        } else if self.used.rounds >= self.budget.rounds.get() {
            Some(ExhaustionReason::Rounds)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_are_positive_and_bounded() {
        assert_eq!(
            VerifierCallBudget::new(0),
            Err(BudgetError::ZeroVerifierCalls)
        );
        assert_eq!(RoundBudget::new(0), Err(BudgetError::ZeroRounds));
        assert!(VerifierCallBudget::new(MAX_VERIFIER_CALLS).is_ok());
        assert!(matches!(
            VerifierCallBudget::new(MAX_VERIFIER_CALLS + 1),
            Err(BudgetError::TooManyVerifierCalls { .. })
        ));
        assert!(matches!(
            RoundBudget::new(MAX_ROUNDS + 1),
            Err(BudgetError::TooManyRounds { .. })
        ));
    }

    #[test]
    fn the_meter_refuses_a_charge_past_the_budget() {
        let mut meter = BudgetMeter::new(RequestBudget::new(2, 1).unwrap());
        assert_eq!(meter.charge_verifier(), Ok(()));
        assert_eq!(meter.stop_after_verdict(), None);
        assert_eq!(meter.charge_round(), Ok(()));
        assert_eq!(meter.stop_after_verdict(), Some(ExhaustionReason::Rounds));
        assert_eq!(meter.charge_round(), Err(ExhaustionReason::Rounds));
        assert_eq!(meter.charge_verifier(), Ok(()));
        assert_eq!(
            meter.stop_after_verdict(),
            Some(ExhaustionReason::VerifierCalls)
        );
        assert_eq!(
            meter.charge_verifier(),
            Err(ExhaustionReason::VerifierCalls)
        );
        assert_eq!(meter.used().verifier_calls, 2);
        assert_eq!(meter.calls_left(), 0);
    }
}
