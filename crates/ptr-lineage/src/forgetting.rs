use crate::error::LineageError;

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
        self.last().iter().sum::<f64>() / self.values.len() as f64
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
        (0..tasks - 1)
            .map(|j| last[j] - self.values[j][j])
            .sum::<f64>()
            / (tasks - 1) as f64
    }

    /// Forgetting of each earlier task (Chaudhry et al.): its best score at any
    /// earlier stage minus its final score. Never negative.
    pub fn task_forgetting(&self) -> Vec<f64> {
        let tasks = self.values.len();
        let last = self.last();
        (0..tasks.saturating_sub(1))
            .map(|j| {
                let best = (j..tasks - 1)
                    .map(|stage| self.values[stage][j])
                    .fold(f64::NEG_INFINITY, f64::max);
                (best - last[j]).max(0.0)
            })
            .collect()
    }

    /// Mean forgetting over earlier tasks, or zero for a single task.
    pub fn average_forgetting(&self) -> f64 {
        let forgetting = self.task_forgetting();
        if forgetting.is_empty() {
            0.0
        } else {
            forgetting.iter().sum::<f64>() / forgetting.len() as f64
        }
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
    violations: Vec<GateViolation>,
}

impl GateReport {
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
    /// Report every exceeded forgetting or regression limit and every missed
    /// backward-transfer minimum. Equality passes each threshold. A NaN public
    /// score difference is reported as a public-regression violation.
    pub fn evaluate(&self, matrix: &AccuracyMatrix, public: PublicSuite) -> GateReport {
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
        GateReport { violations }
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
        let report = gate.evaluate(
            &matrix(),
            PublicSuite {
                serving: 0.5,
                candidate: 0.45,
            },
        );
        assert!(!report.passed());
        assert_eq!(report.violations().len(), 4);
    }

    #[test]
    fn a_nan_public_score_fails_rather_than_passes() {
        let gate = ForgettingGate {
            max_average_forgetting: 1.0,
            max_task_forgetting: 1.0,
            min_backward_transfer: -1.0,
            max_public_regression: 1.0,
        };
        let report = gate.evaluate(
            &matrix(),
            PublicSuite {
                serving: 0.5,
                candidate: f64::NAN,
            },
        );
        assert!(!report.passed());
    }

    #[test]
    fn a_ragged_matrix_is_refused() {
        assert!(AccuracyMatrix::new(vec![vec![0.9, 0.1], vec![0.8]]).is_err());
    }
}
