use ptr_analytics::{
    brier_score, expected_calibration_error, wilson_interval, Binning, RateEstimate,
};

use crate::error::LabelingError;
use crate::votes::{Vote, VoteMatrix};

/// Who asserted a gold label.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GoldSource {
    /// A label known by construction in an explicitly labelled mechanism
    /// experiment. It must never be mixed with human labels in one evaluation.
    Oracle,
    /// A named person's adjudication.
    Human { annotator: String },
}

/// How the items of an evaluation set were chosen.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoldSampling {
    /// Uniformly at random from the labelled population. Calibration can be
    /// measured on it.
    Uniform,
    /// Chosen by an acquisition function (entropy, margin, disputes). A biased
    /// sample of hard items: good for accuracy on hard cases, invalid for
    /// calibration.
    Active,
}

/// A label that may serve as ground truth. There is deliberately no way to
/// build one from a model posterior: predicted labels can be evaluated against
/// gold, never become it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GoldLabel {
    pub item: usize,
    pub class: usize,
    pub source: GoldSource,
}

/// Gold labels of one kind of source and one sampling design.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationSet {
    sampling: GoldSampling,
    labels: Vec<GoldLabel>,
}

impl EvaluationSet {
    pub fn new(sampling: GoldSampling) -> Self {
        Self {
            sampling,
            labels: Vec::new(),
        }
    }

    /// Add a gold label. An oracle label and a human label may not share a set,
    /// because an evaluation that mixes them measures neither.
    pub fn push(&mut self, label: GoldLabel) -> Result<(), LabelingError> {
        if let Some(first) = self.labels.first() {
            let same_kind = matches!(
                (&first.source, &label.source),
                (GoldSource::Oracle, GoldSource::Oracle)
                    | (GoldSource::Human { .. }, GoldSource::Human { .. })
            );
            if !same_kind {
                return Err(LabelingError::InvalidParameter {
                    field: "gold source",
                    message: "oracle and human labels may not be mixed in one evaluation set",
                });
            }
        }
        self.labels.push(label);
        Ok(())
    }

    pub fn sampling(&self) -> GoldSampling {
        self.sampling
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// How a label model's posteriors score against an evaluation set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvaluationReport {
    pub accuracy: f64,
    /// Present only for a uniformly sampled set.
    pub calibration: Option<Calibration>,
}

/// Calibration of posteriors on a uniform gold sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub brier: f64,
    /// Expected calibration error over equal-mass bins.
    pub expected_calibration_error: f64,
}

/// How often one labeling function's class votes match uniform gold labels.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionAccuracy {
    pub function: String,
    /// The adapter a model function is attributed to, if any.
    pub adapter: Option<String>,
    /// Correct class votes over class votes cast on gold items, with a Wilson
    /// interval. `None` when the function cast no class vote on a gold item:
    /// a verifier never does, and an abstaining function tells nothing.
    pub estimate: Option<RateEstimate>,
}

/// Score each labeling function's class votes against gold labels: the share
/// of its class votes on gold items that name the gold class, with a Wilson
/// interval at normal quantile `z`. Abstentions and vetoes are not scored. A
/// function attributed to an adapter reports it, which is how labeling quality
/// is measured per adapter.
///
/// Only a uniformly sampled gold set gives an unbiased estimate: an actively
/// sampled set over-represents the items the model found hardest.
///
/// # Errors
/// Refuses an empty or actively sampled gold set, a gold item beyond the vote
/// matrix, and a `z` that is not finite and positive.
pub fn function_accuracy(
    matrix: &VoteMatrix,
    gold: &EvaluationSet,
    z: f64,
) -> Result<Vec<FunctionAccuracy>, LabelingError> {
    if gold.is_empty() {
        return Err(LabelingError::Empty {
            field: "evaluation set",
        });
    }
    if gold.sampling != GoldSampling::Uniform {
        return Err(LabelingError::InvalidParameter {
            field: "evaluation set",
            message: "per-function accuracy needs a uniformly sampled gold set",
        });
    }
    let mut counts = vec![(0u64, 0u64); matrix.functions().len()];
    for label in &gold.labels {
        if label.item >= matrix.items() {
            return Err(LabelingError::LengthMismatch {
                expected: label.item + 1,
                actual: matrix.items(),
            });
        }
        for (count, vote) in counts.iter_mut().zip(matrix.row(label.item)) {
            if let Vote::Class(class) = vote {
                count.1 += 1;
                if *class == label.class {
                    count.0 += 1;
                }
            }
        }
    }
    matrix
        .functions()
        .iter()
        .zip(counts)
        .map(|(function, (correct, voted))| {
            let estimate = if voted == 0 {
                None
            } else {
                Some(wilson_interval(correct, voted, z)?)
            };
            Ok(FunctionAccuracy {
                function: function.name.clone(),
                adapter: function.adapter.clone(),
                estimate,
            })
        })
        .collect()
}

/// Score posteriors against gold labels. Items without a gold label are not
/// scored. Calibration is reported only when the set was sampled uniformly.
///
/// `bins` controls equal-mass calibration bins for a uniform set and is
/// ignored for an active set.
///
/// # Errors
/// Rejects an empty gold set or a gold item without a posterior. For uniform
/// sets, also propagates statistics errors for invalid distributions, invalid
/// truth indices, or zero bins; active sets do not perform those checks.
pub fn evaluate(
    posteriors: &[Vec<f64>],
    gold: &EvaluationSet,
    bins: usize,
) -> Result<EvaluationReport, LabelingError> {
    if gold.is_empty() {
        return Err(LabelingError::Empty {
            field: "evaluation set",
        });
    }
    let mut predicted = Vec::with_capacity(gold.len());
    let mut truth = Vec::with_capacity(gold.len());
    for label in &gold.labels {
        let posterior = posteriors
            .get(label.item)
            .ok_or(LabelingError::LengthMismatch {
                expected: label.item + 1,
                actual: posteriors.len(),
            })?;
        predicted.push(posterior.clone());
        truth.push(label.class);
    }
    let correct = predicted
        .iter()
        .zip(&truth)
        .filter(|(p, &t)| {
            p.iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(class, _)| class)
                == Some(t)
        })
        .count();
    let calibration = match gold.sampling {
        GoldSampling::Uniform => Some(Calibration {
            brier: brier_score(&predicted, &truth)?,
            expected_calibration_error: expected_calibration_error(
                &predicted,
                &truth,
                Binning::EqualMass(bins),
            )?,
        }),
        GoldSampling::Active => None,
    };
    Ok(EvaluationReport {
        accuracy: correct as f64 / truth.len() as f64,
        calibration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_and_human_labels_cannot_share_a_set() {
        let mut set = EvaluationSet::new(GoldSampling::Uniform);
        set.push(GoldLabel {
            item: 0,
            class: 1,
            source: GoldSource::Oracle,
        })
        .unwrap();
        assert!(set
            .push(GoldLabel {
                item: 1,
                class: 0,
                source: GoldSource::Human {
                    annotator: "annotator-1".into()
                },
            })
            .is_err());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn an_actively_sampled_set_reports_no_calibration() {
        let mut set = EvaluationSet::new(GoldSampling::Active);
        set.push(GoldLabel {
            item: 0,
            class: 0,
            source: GoldSource::Oracle,
        })
        .unwrap();
        let report = evaluate(&[vec![0.9, 0.1]], &set, 5).unwrap();
        assert_eq!(report.accuracy, 1.0);
        assert_eq!(report.calibration, None);
    }
}
