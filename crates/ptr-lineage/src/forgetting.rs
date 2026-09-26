use crate::error::LineageError;
use crate::lineage::AdapterId;
use crate::scale;

/// `values[i][j]`: score on task `j` after training stage `i`, for `T` stages
/// and `T` tasks (task `j` is the one introduced at stage `j`).
#[derive(Clone, Debug, PartialEq)]
pub struct AccuracyMatrix {
    values: Vec<Vec<f64>>,
}

impl AccuracyMatrix {
    /// Build a square stage-by-task score matrix. Scores need only be finite;
    /// they are not restricted to probabilities.
    ///
    /// No summary overflows on the way to its value: means are computed
    /// without an overflowing running sum and differences as differences of
    /// halves, so the average accuracy is always finite, and backward
    /// transfer and forgetting are infinite only when their exact value
    /// exceeds `f64::MAX` (a score of `f64::MAX` forgotten down to
    /// `-f64::MAX`), never NaN.
    ///
    /// # Errors
    /// Rejects an empty matrix, rows of the wrong length, or nonfinite scores.
    pub fn new(values: Vec<Vec<f64>>) -> Result<Self, LineageError> {
        let tasks = values.len();
        if tasks == 0 {
            return Err(LineageError::Empty {
                field: "accuracy matrix",
            });
        }
        for row in &values {
            if row.len() != tasks {
                return Err(LineageError::ShapeMismatch {
                    field: "accuracy matrix row",
                    expected: tasks,
                    actual: row.len(),
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(LineageError::NonFinite {
                    field: "accuracy matrix",
                });
            }
        }
        Ok(Self { values })
    }

    fn last(&self) -> &[f64] {
        self.values.last().expect("non-empty matrix")
    }

    /// Mean final score over all tasks.
    pub fn average_accuracy(&self) -> f64 {
        scale::mean(self.last())
    }

    /// Backward transfer (Lopez-Paz and Ranzato): mean change on every earlier
    /// task between just after learning it and the end. Negative is
    /// forgetting. Zero for a single task.
    pub fn backward_transfer(&self) -> f64 {
        let tasks = self.values.len();
        if tasks < 2 {
            return 0.0;
        }
        let last = self.last();
        let halves: Vec<f64> = (0..tasks - 1)
            .map(|j| last[j] / 2.0 - self.values[j][j] / 2.0)
            .collect();
        2.0 * scale::mean(&halves)
    }

    /// Forgetting of each earlier task (Chaudhry et al.): its best score at any
    /// earlier stage minus its final score. Never negative.
    pub fn task_forgetting(&self) -> Vec<f64> {
        self.half_forgetting()
            .into_iter()
            .map(|half| 2.0 * half)
            .collect()
    }

    /// Mean forgetting over earlier tasks, or zero for a single task.
    pub fn average_forgetting(&self) -> f64 {
        let halves = self.half_forgetting();
        if halves.is_empty() {
            0.0
        } else {
            2.0 * scale::mean(&halves)
        }
    }

    /// Half of each earlier task's forgetting, `max(0, best / 2 - last / 2)`,
    /// which is finite for any finite scores; halving is exact unless a score
    /// is subnormal.
    fn half_forgetting(&self) -> Vec<f64> {
        let tasks = self.values.len();
        let last = self.last();
        (0..tasks.saturating_sub(1))
            .map(|j| {
                let best = (j..tasks - 1)
                    .map(|stage| self.values[stage][j])
                    .fold(f64::NEG_INFINITY, f64::max);
                (best / 2.0 - last[j] / 2.0).max(0.0)
            })
            .collect()
    }
}

/// Thresholds an adapter must meet before it may serve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForgettingGate {
    pub max_average_forgetting: f64,
    pub max_task_forgetting: f64,
    pub min_backward_transfer: f64,
    /// Largest allowed drop on the public regression suite, which catches the
    /// forgetting of general ability that task-local suites cannot see.
    pub max_public_regression: f64,
}

/// One threshold an adapter missed.
#[derive(Clone, Debug, PartialEq)]
pub enum GateViolation {
    AverageForgetting {
        observed: f64,
        limit: f64,
    },
    TaskForgetting {
        task: usize,
        observed: f64,
        limit: f64,
    },
    BackwardTransfer {
        observed: f64,
        limit: f64,
    },
    PublicRegression {
        observed: f64,
        limit: f64,
    },
}

/// The outcome of a gate. Only [`ForgettingGate::evaluate`] produces one.
#[derive(Clone, Debug, PartialEq)]
pub struct GateReport {
    adapter: AdapterId,
    violations: Vec<GateViolation>,
}

impl GateReport {
    /// The adapter whose evaluation produced this report.
    pub fn adapter(&self) -> &AdapterId {
        &self.adapter
    }

    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violations(&self) -> &[GateViolation] {
        &self.violations
    }
}

/// Public-suite scores of the serving model and of the candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublicSuite {
    pub serving: f64,
    pub candidate: f64,
}

impl ForgettingGate {
    fn check(&self) -> Result<(), LineageError> {
        let thresholds = [
            ("max_average_forgetting", self.max_average_forgetting),
            ("max_task_forgetting", self.max_task_forgetting),
            ("min_backward_transfer", self.min_backward_transfer),
            ("max_public_regression", self.max_public_regression),
        ];
        for (field, value) in thresholds {
            if !value.is_finite() {
                return Err(LineageError::InvalidParameter {
                    field,
                    message: "must be finite",
                });
            }
        }
        Ok(())
    }

    /// Report every exceeded forgetting or regression limit and every missed
    /// backward-transfer minimum. Equality passes each threshold. A drop in
    /// public score that overflows to positive infinity exceeds every limit
    /// and is reported as a public-regression violation.
    ///
    /// # Errors
    /// Returns `LineageError::InvalidParameter` for a nonfinite threshold and
    /// `LineageError::NonFinite` for a nonfinite public-suite score, before
    /// anything is evaluated: every comparison with NaN is false, so a NaN
    /// limit would otherwise report no violation and pass, and an infinite
    /// score would pass as an unbounded improvement.
    pub fn evaluate(
        &self,
        adapter: &AdapterId,
        matrix: &AccuracyMatrix,
        public: PublicSuite,
    ) -> Result<GateReport, LineageError> {
        self.check()?;
        if !(public.serving.is_finite() && public.candidate.is_finite()) {
            return Err(LineageError::NonFinite {
                field: "public suite score",
            });
        }
        let mut violations = Vec::new();
        let average = matrix.average_forgetting();
        if average > self.max_average_forgetting {
            violations.push(GateViolation::AverageForgetting {
                observed: average,
                limit: self.max_average_forgetting,
            });
        }
        for (task, observed) in matrix.task_forgetting().into_iter().enumerate() {
            if observed > self.max_task_forgetting {
                violations.push(GateViolation::TaskForgetting {
                    task,
                    observed,
                    limit: self.max_task_forgetting,
                });
            }
        }
        let transfer = matrix.backward_transfer();
        if transfer < self.min_backward_transfer {
            violations.push(GateViolation::BackwardTransfer {
                observed: transfer,
                limit: self.min_backward_transfer,
            });
        }
        let regression = public.serving - public.candidate;
        if regression.is_nan() || regression > self.max_public_regression {
            violations.push(GateViolation::PublicRegression {
                observed: regression,
                limit: self.max_public_regression,
            });
        }
        Ok(GateReport {
            adapter: adapter.clone(),
            violations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> AccuracyMatrix {
        AccuracyMatrix::new(vec![
            vec![0.9, 0.1, 0.1],
            vec![0.8, 0.85, 0.1],
            vec![0.7, 0.8, 0.9],
        ])
        .unwrap()
    }

    #[test]
    fn backward_transfer_and_forgetting_match_their_definitions() {
        let m = matrix();
        // BWT = ((0.7 - 0.9) + (0.8 - 0.85)) / 2 = -0.125
        assert!((m.backward_transfer() + 0.125).abs() < 1e-12);
        // f_0 = max(0.9, 0.8) - 0.7 = 0.2; f_1 = 0.85 - 0.8 = 0.05
        let forgetting = m.task_forgetting();
        assert!((forgetting[0] - 0.2).abs() < 1e-12);
        assert!((forgetting[1] - 0.05).abs() < 1e-12);
        assert!((m.average_accuracy() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_gate_reports_every_violated_threshold() {
        let gate = ForgettingGate {
            max_average_forgetting: 0.1,
            max_task_forgetting: 0.15,
            min_backward_transfer: -0.1,
            max_public_regression: 0.01,
        };
        let report = gate
            .evaluate(
                &AdapterId::from("candidate"),
                &matrix(),
                PublicSuite {
                    serving: 0.5,
                    candidate: 0.45,
                },
            )
            .unwrap();
        assert!(!report.passed());
        assert_eq!(report.violations().len(), 4);
    }

    fn lenient_gate() -> ForgettingGate {
        ForgettingGate {
            max_average_forgetting: 1.0,
            max_task_forgetting: 1.0,
            min_backward_transfer: -1.0,
            max_public_regression: 1.0,
        }
    }

    #[test]
    fn a_nonfinite_public_score_is_refused_rather_than_passed() {
        let scores = [
            (0.5, f64::NAN),
            (f64::NAN, 0.5),
            (0.5, f64::INFINITY),
            (f64::NEG_INFINITY, 0.5),
            (f64::INFINITY, f64::INFINITY),
        ];
        for (serving, candidate) in scores {
            assert_eq!(
                lenient_gate().evaluate(
                    &AdapterId::from("candidate"),
                    &matrix(),
                    PublicSuite { serving, candidate },
                ),
                Err(LineageError::NonFinite {
                    field: "public suite score"
                }),
                "serving {serving} candidate {candidate}"
            );
        }
    }

    #[test]
    fn a_nonfinite_threshold_is_refused_before_evaluation() {
        let public = PublicSuite {
            serving: 0.9,
            candidate: 0.0,
        };
        for limit in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let gates = [
                (
                    "max_average_forgetting",
                    ForgettingGate {
                        max_average_forgetting: limit,
                        ..lenient_gate()
                    },
                ),
                (
                    "max_task_forgetting",
                    ForgettingGate {
                        max_task_forgetting: limit,
                        ..lenient_gate()
                    },
                ),
                (
                    "min_backward_transfer",
                    ForgettingGate {
                        min_backward_transfer: limit,
                        ..lenient_gate()
                    },
                ),
                (
                    "max_public_regression",
                    ForgettingGate {
                        max_public_regression: limit,
                        ..lenient_gate()
                    },
                ),
            ];
            for (field, gate) in gates {
                assert_eq!(
                    gate.evaluate(&AdapterId::from("candidate"), &matrix(), public),
                    Err(LineageError::InvalidParameter {
                        field,
                        message: "must be finite",
                    }),
                    "{field} = {limit}"
                );
            }
        }
        // All-NaN limits compare false everywhere; they must not yield a
        // passing report for a candidate that regressed badly.
        let nan_gate = ForgettingGate {
            max_average_forgetting: f64::NAN,
            max_task_forgetting: f64::NAN,
            min_backward_transfer: f64::NAN,
            max_public_regression: f64::NAN,
        };
        assert!(nan_gate
            .evaluate(&AdapterId::from("candidate"), &matrix(), public)
            .is_err());
    }

    #[test]
    fn a_ragged_matrix_is_refused() {
        assert!(AccuracyMatrix::new(vec![vec![0.9, 0.1], vec![0.8]]).is_err());
    }
}
