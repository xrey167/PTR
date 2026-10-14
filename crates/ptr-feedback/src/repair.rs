//! The generate-verify-repair loop.
//!
//! A [`Generator`] proposes a value, the verifier judges it, and the loop returns it only when
//! [`Candidate::passes`] holds at the policy's required level. A rejected value is critiqued
//! from its report ([`Critique::from_report`]); a repairable critique goes to the [`Repairer`],
//! which may decline, and everything else is replaced by a fresh draw. Each candidate is
//! verified exactly once, the number of verifier calls and rounds is bounded by the
//! [`RepairBudget`], and a run that spends its budget without an accepted candidate returns
//! [`Outcome::Exhausted`], never a best-effort failure. The trajectory keeps every candidate,
//! rejected ones included, with its report, critique and the loop's decision, as evidence for
//! training and analysis.

use std::error::Error;
use std::fmt;

use ptr_types::{CandidateId, VerificationLevel};
use ptr_verifier::Verifier;

use crate::candidate::{Candidate, Critique};
use crate::record::MAX_STEPS;

/// Proposes a fresh value. The loop calls it for the first candidate and whenever a critique
/// cannot be repaired.
pub trait Generator<T> {
    fn generate(&mut self) -> T;
}

/// Edits a rejected candidate into a new value that addresses the critique's issues, or
/// returns `None` when it cannot, in which case the loop draws a fresh value instead.
pub trait Repairer<T> {
    fn repair(&mut self, previous: &Candidate<T>, critique: &Critique) -> Option<T>;
}

/// Most steps a run may retain; the same bound a [`crate::TrajectoryRecord`] accepts, so every
/// run of a valid budget can be recorded.
const MAX_RUN_STEPS: usize = MAX_STEPS;

/// The most a run may spend. `max_verifier_calls` bounds the verifier calls (one per
/// candidate); `max_rounds` bounds the replacements that follow the first candidate, so a run
/// holds at most `1 + max_rounds` candidates. Both must be positive, and the smaller of
/// `max_verifier_calls` and `1 + max_rounds` may not exceed [`MAX_STEPS`], the most steps a run
/// retains (every candidate is kept in the trajectory).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepairBudget {
    pub max_verifier_calls: u32,
    pub max_rounds: u32,
}

impl RepairBudget {
    pub fn validate(&self) -> Result<(), RepairError> {
        if self.max_verifier_calls == 0 {
            return Err(RepairError::NoVerifierCalls);
        }
        if self.max_rounds == 0 {
            return Err(RepairError::NoRounds);
        }
        // A run retains one step per verified candidate: at most the smaller of the two limits.
        let steps = u64::from(self.max_verifier_calls).min(u64::from(self.max_rounds) + 1);
        if steps > MAX_RUN_STEPS as u64 {
            return Err(RepairError::BudgetTooLarge {
                max_steps: MAX_RUN_STEPS as u32,
            });
        }
        Ok(())
    }
}

/// What the loop requires of an accepted candidate and what it may spend to find one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepairPolicy {
    pub required_level: VerificationLevel,
    pub budget: RepairBudget,
}

/// Why a loop could not be built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepairError {
    NoVerifierCalls,
    NoRounds,
    /// The run could retain more steps than a trajectory record holds.
    BudgetTooLarge {
        max_steps: u32,
    },
}

impl fmt::Display for RepairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVerifierCalls => f.write_str("the budget allows no verifier call"),
            Self::NoRounds => f.write_str("the budget allows no round"),
            Self::BudgetTooLarge { max_steps } => {
                write!(f, "the budget allows a run of more than {max_steps} steps")
            }
        }
    }
}

impl Error for RepairError {}

/// Why a rejected candidate was replaced by a fresh draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RewriteReason {
    /// The critique was not repairable (`Unknown`, `Disputed`, a contradictory `Pass`, a
    /// `Fail` without a hard finding, or a `Pass` below the required level).
    NotRepairable,
    /// The critique was repairable but the repairer declined.
    Declined,
}

/// How a candidate came to be.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Origin {
    Initial,
    Repaired {
        from: CandidateId,
    },
    Rewritten {
        from: CandidateId,
        reason: RewriteReason,
    },
}

/// Which limit ended a run without an accepted candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExhaustionReason {
    VerifierCalls,
    Rounds,
}

/// What the loop did with a verified candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Decision {
    Accepted,
    Repair,
    Rewrite(RewriteReason),
    Stop(ExhaustionReason),
}

/// One verified candidate of a run.
#[derive(Clone, Debug)]
pub struct Step<T> {
    pub candidate: Candidate<T>,
    pub origin: Origin,
    /// The critique of the candidate's report; `None` for the accepted candidate.
    pub critique: Option<Critique>,
    pub decision: Decision,
}

/// Every candidate of a run in order, rejected ones included.
#[derive(Clone, Debug)]
pub struct Trajectory<T> {
    pub steps: Vec<Step<T>>,
}

/// What a run claims to have spent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    pub verifier_calls: u32,
    /// Replacements after the first candidate (repaired or rewritten).
    pub rounds: u32,
    /// Calls of [`Generator::generate`], the first candidate's included.
    pub generated: u32,
    /// Repairs the repairer returned.
    pub repaired: u32,
    /// Repairs the repairer declined (each is followed by a fresh draw).
    pub declined: u32,
}

/// The end of a run.
#[derive(Clone, Debug)]
pub enum Outcome<T> {
    /// The candidate passed at the required level.
    Verified(Candidate<T>),
    /// The budget was spent without an accepted candidate; no failed candidate is returned.
    Exhausted { reason: ExhaustionReason },
}

/// A finished run.
#[derive(Clone, Debug)]
pub struct RepairRun<T> {
    pub outcome: Outcome<T>,
    pub trajectory: Trajectory<T>,
    pub usage: Usage,
}

/// The loop over one verifier and one policy.
pub struct RepairLoop<'v, T> {
    verifier: &'v dyn Verifier<T>,
    policy: RepairPolicy,
}

impl<'v, T: Clone> RepairLoop<'v, T> {
    /// # Errors
    /// Returns [`RepairError`] when the budget allows no verifier call or no round, or a run that
    /// could retain more than [`MAX_STEPS`] steps.
    pub fn new(verifier: &'v dyn Verifier<T>, policy: RepairPolicy) -> Result<Self, RepairError> {
        policy.budget.validate()?;
        Ok(Self { verifier, policy })
    }

    pub fn policy(&self) -> &RepairPolicy {
        &self.policy
    }

    /// Run the loop once. Terminates after at most `max_verifier_calls` verifier calls.
    pub fn run(
        &self,
        generator: &mut dyn Generator<T>,
        repairer: &mut dyn Repairer<T>,
    ) -> RepairRun<T> {
        let budget = self.policy.budget;
        let mut usage = Usage::default();
        let mut steps: Vec<Step<T>> = Vec::new();
        let mut value = generator.generate();
        usage.generated += 1;
        let mut origin = Origin::Initial;
        loop {
            let id = CandidateId::from(format!("c{}", usage.verifier_calls).as_str());
            let report = self.verifier.verify(&value);
            usage.verifier_calls += 1;
            let candidate = Candidate {
                id,
                value,
                verification: Some(report),
            };
            if candidate.passes(self.policy.required_level) {
                steps.push(Step {
                    candidate: candidate.clone(),
                    origin,
                    critique: None,
                    decision: Decision::Accepted,
                });
                return RepairRun {
                    outcome: Outcome::Verified(candidate),
                    trajectory: Trajectory { steps },
                    usage,
                };
            }
            let critique = candidate
                .critique()
                .expect("a verified candidate has a report");
            let stop = if usage.verifier_calls >= budget.max_verifier_calls {
                Some(ExhaustionReason::VerifierCalls)
            } else if usage.rounds >= budget.max_rounds {
                Some(ExhaustionReason::Rounds)
            } else {
                None
            };
            if let Some(reason) = stop {
                steps.push(Step {
                    candidate,
                    origin,
                    critique: Some(critique),
                    decision: Decision::Stop(reason),
                });
                return RepairRun {
                    outcome: Outcome::Exhausted { reason },
                    trajectory: Trajectory { steps },
                    usage,
                };
            }
            let repaired = if critique.repairable {
                repairer.repair(&candidate, &critique)
            } else {
                None
            };
            let (next, next_origin, decision) = match repaired {
                Some(next) => {
                    usage.repaired += 1;
                    let from = candidate.id.clone();
                    (next, Origin::Repaired { from }, Decision::Repair)
                }
                None => {
                    let reason = if critique.repairable {
                        usage.declined += 1;
                        RewriteReason::Declined
                    } else {
                        RewriteReason::NotRepairable
                    };
                    let fresh = generator.generate();
                    usage.generated += 1;
                    let from = candidate.id.clone();
                    (
                        fresh,
                        Origin::Rewritten { from, reason },
                        Decision::Rewrite(reason),
                    )
                }
            };
            usage.rounds += 1;
            steps.push(Step {
                candidate,
                origin,
                critique: Some(critique),
                decision,
            });
            value = next;
            origin = next_origin;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::Probability;
    use ptr_verifier::{Finding, VerificationReport, VerificationStatus};
    use std::cell::{Cell, RefCell};

    /// A value is a list of defect codes; the verifier reports each as a hard finding.
    type Value = Vec<&'static str>;

    #[derive(Default)]
    struct Judge {
        calls: Cell<u32>,
        seen: RefCell<Vec<Value>>,
        status_if_clean: Option<VerificationStatus>,
        level_if_clean: Option<VerificationLevel>,
    }

    impl Verifier<Value> for Judge {
        fn verify(&self, value: &Value) -> VerificationReport {
            self.calls.set(self.calls.get() + 1);
            self.seen.borrow_mut().push(value.clone());
            let findings: Vec<Finding> = value
                .iter()
                .map(|code| Finding {
                    code: (*code).into(),
                    message: String::new(),
                    hard: true,
                })
                .collect();
            let clean = findings.is_empty();
            VerificationReport {
                status: if !clean {
                    VerificationStatus::Fail
                } else {
                    self.status_if_clean.unwrap_or(VerificationStatus::Pass)
                },
                level: self
                    .level_if_clean
                    .unwrap_or(VerificationLevel::Deterministic),
                score: Probability::new(if clean { 1.0 } else { 0.0 }).unwrap(),
                findings,
            }
        }
    }

    /// Hands out the scripted values in order.
    struct Script(Vec<Value>);
    impl Generator<Value> for Script {
        fn generate(&mut self) -> Value {
            self.0.remove(0)
        }
    }

    /// Removes the named defects and records what it was asked.
    #[derive(Default)]
    struct Fixer {
        asked: Vec<Vec<String>>,
        decline: bool,
    }
    impl Repairer<Value> for Fixer {
        fn repair(&mut self, previous: &Candidate<Value>, critique: &Critique) -> Option<Value> {
            self.asked.push(critique.issues.clone());
            if self.decline {
                return None;
            }
            Some(
                previous
                    .value
                    .iter()
                    .copied()
                    .filter(|code| !critique.issues.iter().any(|issue| issue == code))
                    .collect(),
            )
        }
    }

    fn policy(calls: u32, rounds: u32) -> RepairPolicy {
        RepairPolicy {
            required_level: VerificationLevel::Deterministic,
            budget: RepairBudget {
                max_verifier_calls: calls,
                max_rounds: rounds,
            },
        }
    }

    fn run(
        judge: &Judge,
        policy: RepairPolicy,
        script: Vec<Value>,
        fixer: &mut Fixer,
    ) -> RepairRun<Value> {
        RepairLoop::new(judge, policy)
            .expect("a positive budget")
            .run(&mut Script(script), fixer)
    }

    #[test]
    fn a_clean_first_candidate_is_accepted_after_one_call() {
        let judge = Judge::default();
        let outcome = run(&judge, policy(4, 4), vec![vec![]], &mut Fixer::default());
        assert!(matches!(outcome.outcome, Outcome::Verified(ref c) if c.value.is_empty()));
        assert_eq!(judge.calls.get(), 1);
        assert_eq!(outcome.usage.verifier_calls, 1);
        assert_eq!(outcome.usage.rounds, 0);
        assert_eq!(outcome.trajectory.steps.len(), 1);
        assert_eq!(outcome.trajectory.steps[0].decision, Decision::Accepted);
        assert_eq!(outcome.trajectory.steps[0].origin, Origin::Initial);
    }

    #[test]
    fn a_repairable_critique_goes_to_the_repairer_and_the_repair_is_verified() {
        let judge = Judge::default();
        let mut fixer = Fixer::default();
        let run = run(&judge, policy(4, 4), vec![vec!["a", "b"]], &mut fixer);
        assert_eq!(fixer.asked, [["a".to_string(), "b".to_string()]]);
        let Outcome::Verified(winner) = &run.outcome else {
            panic!("repaired and verified")
        };
        assert_eq!(winner.id.to_string(), "c1");
        assert_eq!(judge.calls.get(), 2, "the repaired value is verified too");
        assert_eq!(
            *judge.seen.borrow(),
            [vec!["a", "b"], Vec::<&str>::new()],
            "each candidate is verified exactly once"
        );
        let steps = &run.trajectory.steps;
        assert_eq!(steps[0].decision, Decision::Repair);
        assert_eq!(steps[0].critique.as_ref().unwrap().issues, ["a", "b"]);
        assert_eq!(
            steps[1].origin,
            Origin::Repaired {
                from: CandidateId::from("c0")
            }
        );
        assert_eq!((run.usage.generated, run.usage.repaired), (1, 1));
    }

    #[test]
    fn an_unknown_or_disputed_report_is_rewritten_not_repaired() {
        for status in [VerificationStatus::Unknown, VerificationStatus::Disputed] {
            let judge = Judge {
                status_if_clean: Some(status),
                ..Judge::default()
            };
            let mut fixer = Fixer::default();
            let run = run(&judge, policy(2, 2), vec![vec![], vec![]], &mut fixer);
            assert!(fixer.asked.is_empty(), "{status:?} is not repairable");
            assert!(
                matches!(run.outcome, Outcome::Exhausted { .. }),
                "{status:?}"
            );
            assert_eq!(
                run.trajectory.steps[0].decision,
                Decision::Rewrite(RewriteReason::NotRepairable)
            );
            assert_eq!(run.usage.generated, 2);
        }
    }

    #[test]
    fn a_pass_below_the_required_level_is_not_accepted() {
        let judge = Judge {
            level_if_clean: Some(VerificationLevel::LatentAgreement),
            ..Judge::default()
        };
        let run = run(&judge, policy(3, 3), vec![vec![]; 3], &mut Fixer::default());
        assert!(matches!(run.outcome, Outcome::Exhausted { .. }));
        assert_eq!(judge.calls.get(), 3);
    }

    #[test]
    fn a_declined_repair_becomes_a_fresh_draw() {
        let judge = Judge::default();
        let mut fixer = Fixer {
            decline: true,
            ..Fixer::default()
        };
        let run = run(&judge, policy(4, 4), vec![vec!["a"], vec![]], &mut fixer);
        assert!(matches!(run.outcome, Outcome::Verified(_)));
        assert_eq!(
            run.trajectory.steps[1].origin,
            Origin::Rewritten {
                from: CandidateId::from("c0"),
                reason: RewriteReason::Declined
            }
        );
        assert_eq!((run.usage.generated, run.usage.declined), (2, 1));
    }

    #[test]
    fn the_verifier_call_budget_is_never_exceeded_and_exhaustion_returns_no_candidate() {
        for calls in 1..=5 {
            let judge = Judge::default();
            let script = vec![vec!["a"]; 8];
            let mut fixer = Fixer {
                decline: true,
                ..Fixer::default()
            };
            let run = run(&judge, policy(calls, 99), script, &mut fixer);
            let Outcome::Exhausted { reason } = run.outcome else {
                panic!("every candidate fails: exhausted")
            };
            assert_eq!(reason, ExhaustionReason::VerifierCalls);
            assert_eq!(judge.calls.get(), calls);
            assert_eq!(run.usage.verifier_calls, calls);
            assert_eq!(run.trajectory.steps.len() as u32, calls);
            assert_eq!(
                run.trajectory.steps.last().unwrap().decision,
                Decision::Stop(ExhaustionReason::VerifierCalls)
            );
        }
    }

    #[test]
    fn the_round_budget_bounds_the_replacements() {
        let judge = Judge::default();
        let mut fixer = Fixer {
            decline: true,
            ..Fixer::default()
        };
        let run = run(&judge, policy(99, 2), vec![vec!["a"]; 8], &mut fixer);
        assert!(matches!(
            run.outcome,
            Outcome::Exhausted {
                reason: ExhaustionReason::Rounds
            }
        ));
        assert_eq!(judge.calls.get(), 3, "the first candidate and two rounds");
        assert_eq!(run.usage.rounds, 2);
    }

    #[test]
    fn a_zero_budget_cannot_build_a_loop() {
        let judge = Judge::default();
        assert_eq!(
            RepairLoop::new(&judge, policy(0, 1)).err(),
            Some(RepairError::NoVerifierCalls)
        );
        assert_eq!(
            RepairLoop::new(&judge, policy(1, 0)).err(),
            Some(RepairError::NoRounds)
        );
        assert!(!RepairError::NoRounds.to_string().is_empty());
    }

    #[test]
    fn an_oversized_budget_is_refused_before_a_run_can_outgrow_a_record() {
        let judge = Judge::default();
        for (calls, rounds) in [
            (u32::MAX, u32::MAX),
            (300, 300),
            (257, 257),
            (257, u32::MAX),
        ] {
            assert!(
                RepairLoop::new(&judge, policy(calls, rounds)).is_err(),
                "{calls} calls / {rounds} rounds"
            );
        }
    }

    #[test]
    fn every_accepted_budget_yields_a_run_that_converts_to_a_record() {
        use crate::record::{TrajectoryRecord, MAX_STEPS};
        struct Never;
        impl Verifier<String> for Never {
            fn verify(&self, _: &String) -> VerificationReport {
                VerificationReport {
                    status: VerificationStatus::Fail,
                    level: VerificationLevel::Deterministic,
                    score: Probability::new(0.0).unwrap(),
                    findings: vec![Finding {
                        code: "x".into(),
                        message: String::new(),
                        hard: true,
                    }],
                }
            }
        }
        struct Always;
        impl Generator<String> for Always {
            fn generate(&mut self) -> String {
                "v".into()
            }
        }
        struct Keep;
        impl Repairer<String> for Keep {
            fn repair(&mut self, previous: &Candidate<String>, _: &Critique) -> Option<String> {
                Some(previous.value.clone())
            }
        }
        let judge = Judge::default();
        let steps = MAX_STEPS as u32;
        // The tightest accepted budgets: either bound alone caps the run at MAX_STEPS.
        for (calls, rounds) in [(steps, steps), (u32::MAX, steps - 1), (steps, u32::MAX)] {
            let looped = RepairLoop::new(
                &Never,
                RepairPolicy {
                    required_level: VerificationLevel::Deterministic,
                    budget: RepairBudget {
                        max_verifier_calls: calls,
                        max_rounds: rounds,
                    },
                },
            )
            .expect("within the cap");
            let run = looped.run(&mut Always, &mut Keep);
            assert_eq!(run.trajectory.steps.len(), MAX_STEPS, "{calls}/{rounds}");
            assert!(
                TrajectoryRecord::from_run("r", &run).is_ok(),
                "{calls}/{rounds}"
            );
        }
        assert_eq!(
            RepairLoop::new(&judge, policy(steps + 1, steps + 1)).err(),
            Some(RepairError::BudgetTooLarge { max_steps: steps })
        );
        assert!(!RepairError::BudgetTooLarge { max_steps: steps }
            .to_string()
            .is_empty());
    }

    #[test]
    fn a_report_that_contradicts_itself_is_refused_and_rewritten() {
        struct Forger;
        impl Verifier<Value> for Forger {
            fn verify(&self, value: &Value) -> VerificationReport {
                VerificationReport {
                    status: VerificationStatus::Pass,
                    level: VerificationLevel::Deterministic,
                    score: Probability::new(1.0).unwrap(),
                    findings: value
                        .iter()
                        .map(|code| Finding {
                            code: (*code).into(),
                            message: String::new(),
                            hard: true,
                        })
                        .collect(),
                }
            }
        }
        let mut fixer = Fixer::default();
        let run = RepairLoop::new(&Forger, policy(3, 3))
            .unwrap()
            .run(&mut Script(vec![vec!["a"], vec![]]), &mut fixer);
        assert!(
            fixer.asked.is_empty(),
            "a forged Pass names nothing to trust"
        );
        let Outcome::Verified(winner) = run.outcome else {
            panic!("the clean value passes")
        };
        assert!(winner.value.is_empty());
        assert_eq!(
            run.trajectory.steps[0].decision,
            Decision::Rewrite(RewriteReason::NotRepairable)
        );
    }
}
