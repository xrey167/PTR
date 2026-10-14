//! The budget of self-improvement cycles, counted from committed history.
//!
//! A self-improvement cycle proposes a change to the system's own adaptation (a new adapter, a
//! new policy) and measures it. How many may run per day is a safety property, not a tuning
//! knob, so the count has to come from something the cycle cannot rewrite. This module holds
//! the rule and nothing else: the runtime derives [`CycleRecord`]s from the ledger events it
//! committed (`ImprovementCycleRecorded`) and from the cycles it has admitted but not yet
//! recorded, and asks [`ImprovementCycleBudget::check_budget`]. This crate stays free of the
//! ledger.
//!
//! **An aborted cycle spends budget.** A cycle that is stopped early still used compute and
//! still produced evidence a proposer could mine, so counting only completed cycles would let
//! a proposer retry without limit by aborting. A cycle that is still running spends its slot
//! too, so concurrent starts cannot each see one free slot.

use std::fmt;

/// How a counted cycle ended, as far as the budget is concerned. Every variant spends one
/// unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CycleOutcomeKind {
    Completed,
    Aborted,
    /// Admitted and not yet recorded.
    InFlight,
}

/// One counted cycle: the day it belongs to (days since the Unix epoch) and how it ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CycleRecord {
    pub day: u64,
    pub outcome: CycleOutcomeKind,
}

/// The cycles that may be admitted per day.
///
/// A budget of zero admits none, which is the state of a runtime nobody has configured.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImprovementCycleBudget {
    per_day: u32,
}

/// The budget of `day` is spent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetExhausted {
    pub day: u64,
    /// Cycles already counted for `day`, saturating at `u64::MAX`.
    pub used: u64,
    pub per_day: u32,
}

impl fmt::Display for BudgetExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "improvement budget of day {} is spent: {} of {} cycles used",
            self.day, self.used, self.per_day
        )
    }
}

impl std::error::Error for BudgetExhausted {}

impl ImprovementCycleBudget {
    #[must_use]
    pub const fn new(per_day: u32) -> Self {
        Self { per_day }
    }

    #[must_use]
    pub const fn per_day(&self) -> u32 {
        self.per_day
    }

    /// Whether one more cycle may be admitted on `day`, given every cycle counted so far.
    /// Records of other days are ignored.
    ///
    /// # Errors
    /// [`BudgetExhausted`] when the cycles already counted for `day` reach the budget.
    pub fn check_budget(
        &self,
        day: u64,
        history: impl IntoIterator<Item = CycleRecord>,
    ) -> Result<(), BudgetExhausted> {
        let used = history
            .into_iter()
            .filter(|record| record.day == day)
            .fold(0u64, |used, _| used.saturating_add(1));
        if used >= u64::from(self.per_day) {
            return Err(BudgetExhausted {
                day,
                used,
                per_day: self.per_day,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(day: u64, outcome: CycleOutcomeKind) -> CycleRecord {
        CycleRecord { day, outcome }
    }

    #[test]
    fn a_zero_budget_admits_nothing() {
        assert_eq!(
            ImprovementCycleBudget::new(0).check_budget(5, []),
            Err(BudgetExhausted {
                day: 5,
                used: 0,
                per_day: 0
            })
        );
    }

    #[test]
    fn aborted_and_in_flight_cycles_spend_budget_like_completed_ones() {
        for outcome in [
            CycleOutcomeKind::Completed,
            CycleOutcomeKind::Aborted,
            CycleOutcomeKind::InFlight,
        ] {
            let budget = ImprovementCycleBudget::new(2);
            assert_eq!(budget.check_budget(7, [record(7, outcome)]), Ok(()));
            assert_eq!(
                budget.check_budget(7, [record(7, outcome), record(7, outcome)]),
                Err(BudgetExhausted {
                    day: 7,
                    used: 2,
                    per_day: 2
                })
            );
        }
    }

    #[test]
    fn other_days_do_not_spend_this_days_budget() {
        let budget = ImprovementCycleBudget::new(1);
        let history = [
            record(6, CycleOutcomeKind::Completed),
            record(8, CycleOutcomeKind::Aborted),
        ];
        assert_eq!(budget.check_budget(7, history), Ok(()));
    }
}
