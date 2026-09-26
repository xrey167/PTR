use std::collections::BTreeSet;

use crate::error::LabelingError;
use crate::votes::{FunctionKind, Vote, VoteMatrix};

/// Parameters of the Dawid-Skene fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DawidSkeneParams {
    pub max_iterations: usize,
    /// Stop when no posterior moves by more than this. It must lie in
    /// `[0, 1)`: a posterior is a probability, so no step moves it by more
    /// than one, and a tolerance of one or more would call every fit converged
    /// after its first iteration. Zero stops only at an exact fixed point,
    /// where no posterior moves at all; otherwise the fit runs to
    /// `max_iterations` and warns that it did not converge.
    pub tolerance: f64,
    /// Additive (Dirichlet) smoothing of priors and confusion matrices, which
    /// keeps every prior and confusion probability positive, so no vote
    /// multiplies a class's posterior by zero. It must be finite and positive
    /// and large enough that the smallest smoothed probability,
    /// `smoothing / (classes * smoothing + items)`, is a normal positive
    /// `f64`: below that, probabilities round to zero and the guarantee is
    /// lost. Any larger value is accepted: totals it makes overflow are
    /// normalized at a representable scale, and near `f64::MAX` every
    /// smoothed probability is uniform.
    pub smoothing: f64,
}

impl Default for DawidSkeneParams {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            smoothing: 1.0,
        }
    }
}

/// Conditions under which the fitted model should not be trusted as is.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelWarning {
    /// Dawid-Skene is identifiable (up to a permutation of classes) only with
    /// at least three conditionally independent functions.
    FewerThanThreeFunctions { count: usize },
    /// A probabilistic function never voted on an item another function also
    /// voted on, so its confusion matrix is estimated from the prior alone.
    NoOverlap { function: String },
    /// EM stopped at the iteration limit before converging.
    NotConverged { iterations: usize },
}

/// A fitted label model: class priors and one confusion matrix per
/// probabilistic function (`confusion[j][true][voted]`). Verifier functions
/// have none; they are not modelled as noisy voters.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelModel {
    pub priors: Vec<f64>,
    pub confusion: Vec<Option<Vec<Vec<f64>>>>,
    pub iterations: usize,
    pub warnings: Vec<ModelWarning>,
    /// Posterior class distribution of every item. An item no probabilistic
    /// function voted on keeps the prior here, and is never resolved from it.
    pub posteriors: Vec<Vec<f64>>,
}

/// Fit a Dawid-Skene model by expectation-maximisation.
///
/// E-step: `T_ic ∝ p_c * prod_{j: L_ij != abstain} pi_j[c, L_ij]`. M-step:
/// `pi_j[c, l] ∝ sum_i T_ic 1{L_ij = l}` over non-abstaining votes and
/// `p_c ∝ sum_i T_ic`, both with additive smoothing. Abstentions are treated as
/// missing at random: they contribute nothing to the likelihood, so a function
/// whose abstention depends on the true class biases the estimate. The fit
/// starts from the smoothed majority vote with a fixed iteration order, which
/// makes it replayable and anchors class identities; Dawid-Skene is otherwise
/// identifiable only up to a permutation of classes. Correlated functions
/// double-count evidence and make posteriors overconfident; the model cannot
/// see that, the calibration report can.
///
/// Every fitted prior, confusion probability and posterior is finite, and every
/// prior and confusion probability is positive, whatever accepted smoothing
/// is used.
///
/// # Errors
/// Refuses zero `max_iterations`, a `tolerance` outside `[0, 1)` (including a
/// nonfinite one), a `smoothing` that is not finite and positive, a matrix
/// without items, and a `smoothing` so small for the number of items that the
/// smallest smoothed probability is not a normal positive number.
pub fn fit_label_model(
    matrix: &VoteMatrix,
    params: DawidSkeneParams,
) -> Result<LabelModel, LabelingError> {
    if params.max_iterations == 0 {
        return Err(LabelingError::InvalidParameter {
            field: "max_iterations",
            message: "at least one iteration is required",
        });
    }
    if !(0.0..1.0).contains(&params.tolerance) {
        return Err(LabelingError::InvalidParameter {
            field: "tolerance",
            message: "must lie in [0, 1)",
        });
    }
    if !(params.smoothing.is_finite() && params.smoothing > 0.0) {
        return Err(LabelingError::InvalidParameter {
            field: "smoothing",
            message: "must be finite and positive",
        });
    }
    if matrix.items() == 0 {
        return Err(LabelingError::Empty { field: "items" });
    }
    let classes = matrix.schema().len();
    // The smallest smoothed probability, written so that neither a tiny nor a
    // huge smoothing overflows on the way: 1 / (classes + items / smoothing).
    let floor = 1.0 / (classes as f64 + matrix.items() as f64 / params.smoothing);
    if floor < f64::MIN_POSITIVE {
        return Err(LabelingError::InvalidParameter {
            field: "smoothing",
            message: "is too small for this many items to keep every smoothed probability positive",
        });
    }
    let modelled: Vec<bool> = matrix
        .functions()
        .iter()
        .map(|function| function.kind != FunctionKind::Verifier)
        .collect();

    let mut warnings = Vec::new();
    let count = modelled.iter().filter(|m| **m).count();
    if count < 3 {
        warnings.push(ModelWarning::FewerThanThreeFunctions { count });
    }
    for (j, function) in matrix.functions().iter().enumerate() {
        if !modelled[j] {
            continue;
        }
        let overlaps = (0..matrix.items()).any(|item| {
            let row = matrix.row(item);
            matches!(row[j], Vote::Class(_))
                && row
                    .iter()
                    .enumerate()
                    .any(|(k, vote)| k != j && modelled[k] && matches!(vote, Vote::Class(_)))
        });
        if !overlaps {
            warnings.push(ModelWarning::NoOverlap {
                function: function.name.clone(),
            });
        }
    }

    let mut posteriors: Vec<Vec<f64>> = (0..matrix.items())
        .map(|item| {
            let mut counts = vec![params.smoothing; classes];
            for (vote, &is_modelled) in matrix.row(item).iter().zip(&modelled) {
                if let (Vote::Class(class), true) = (vote, is_modelled) {
                    counts[*class] += 1.0;
                }
            }
            normalize(counts)
        })
        .collect();

    let mut priors = vec![1.0 / classes as f64; classes];
    let mut confusion: Vec<Option<Vec<Vec<f64>>>> = modelled
        .iter()
        .map(|&is_modelled| is_modelled.then(|| vec![vec![1.0 / classes as f64; classes]; classes]))
        .collect();
    let mut iterations = 0;
    let mut converged = false;

    while iterations < params.max_iterations {
        iterations += 1;
        let mut prior_counts = vec![params.smoothing; classes];
        for posterior in &posteriors {
            for (count, p) in prior_counts.iter_mut().zip(posterior) {
                *count += p;
            }
        }
        priors = normalize(prior_counts);
        for (function, slot) in confusion.iter_mut().enumerate() {
            let Some(table) = slot else { continue };
            let mut counts = vec![vec![params.smoothing; classes]; classes];
            for (item, posterior) in posteriors.iter().enumerate() {
                if let Vote::Class(voted) = matrix.row(item)[function] {
                    for (truth, p) in posterior.iter().enumerate() {
                        counts[truth][voted] += p;
                    }
                }
            }
            *table = counts.into_iter().map(normalize).collect();
        }

        let mut largest_change: f64 = 0.0;
        for (item, posterior) in posteriors.iter_mut().enumerate() {
            let mut logs: Vec<f64> = priors.iter().map(|p| p.ln()).collect();
            for (function, slot) in confusion.iter().enumerate() {
                if let (Some(table), Vote::Class(voted)) = (slot, matrix.row(item)[function]) {
                    for (truth, log) in logs.iter_mut().enumerate() {
                        *log += table[truth][voted].ln();
                    }
                }
            }
            let next = softmax(&logs);
            for (old, new) in posterior.iter().zip(&next) {
                largest_change = largest_change.max((old - new).abs());
            }
            *posterior = next;
        }
        if largest_change <= params.tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        warnings.push(ModelWarning::NotConverged { iterations });
    }

    Ok(LabelModel {
        priors,
        confusion,
        iterations,
        warnings,
        posteriors,
    })
}

/// The label an item receives, or why it receives none.
#[derive(Clone, Debug, PartialEq)]
pub enum LabelOutcome {
    /// Every other class was ruled out by verifiers: the class is determined
    /// by elimination, not estimated.
    Determined { class: usize },
    /// Estimated by the label model over the classes no verifier ruled out,
    /// with at least the required probability.
    Estimated { class: usize, probability: f64 },
    /// Verifiers ruled out every class. Nothing learned can settle that; a
    /// person must.
    Disputed { vetoed: BTreeSet<usize> },
    /// No class is determined and none reaches the required probability, or no
    /// probabilistic function voted at all. Unknown is a valid answer.
    Unknown,
}

/// Resolve every item: verifier vetoes first, then the label model.
///
/// Vetoed classes get probability zero whatever the model says; the posterior
/// is renormalized over the classes that remain. Verifiers ruling out all
/// classes yield `Disputed`; leaving exactly one yields `Determined`, even
/// without probabilistic votes. If multiple classes remain and no
/// probabilistic function voted, the item is `Unknown`.
///
/// When the model puts no probability on any class the verifiers left, no
/// class reaches the required probability and the item is `Unknown`.
///
/// # Errors
/// Returns an error if `min_probability` is not finite and in `(0, 1]`, if
/// the number of posteriors differs from the number of items, and
/// `LabelingError::InvalidPosterior` for the first posterior that is not a
/// probability distribution over the schema's classes, before any item is
/// resolved: a negative or out-of-range entry would resolve to a
/// "probability" above one, and an entry beyond the schema would drop mass
/// unseen.
pub fn resolve(
    matrix: &VoteMatrix,
    model: &LabelModel,
    min_probability: f64,
) -> Result<Vec<LabelOutcome>, LabelingError> {
    if !(min_probability.is_finite() && min_probability > 0.0 && min_probability <= 1.0) {
        return Err(LabelingError::InvalidParameter {
            field: "min_probability",
            message: "must lie in (0, 1]",
        });
    }
    if model.posteriors.len() != matrix.items() {
        return Err(LabelingError::LengthMismatch {
            expected: matrix.items(),
            actual: model.posteriors.len(),
        });
    }
    let classes = matrix.schema().len();
    for (item, posterior) in model.posteriors.iter().enumerate() {
        if posterior.len() != classes {
            return Err(LabelingError::InvalidPosterior { item });
        }
        check_posterior(item, posterior)?;
    }
    Ok((0..matrix.items())
        .map(|item| {
            let row = matrix.row(item);
            let vetoed: BTreeSet<usize> = row
                .iter()
                .filter_map(|vote| match vote {
                    Vote::Veto(class) => Some(*class),
                    Vote::Class(_) | Vote::Abstain => None,
                })
                .collect();
            let remaining: Vec<usize> = (0..classes).filter(|c| !vetoed.contains(c)).collect();
            match remaining.as_slice() {
                [] => return LabelOutcome::Disputed { vetoed },
                [only] => return LabelOutcome::Determined { class: *only },
                _ => {}
            }
            let voted = row.iter().any(|vote| matches!(vote, Vote::Class(_)));
            if !voted {
                return LabelOutcome::Unknown;
            }
            let posterior = &model.posteriors[item];
            // With no mass left every share is 0 / 0, a NaN, which reaches no
            // required probability: the item is Unknown.
            let mass: f64 = remaining.iter().map(|&c| posterior[c]).sum();
            let (class, probability) = remaining
                .iter()
                .map(|&c| (c, posterior[c] / mass))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("at least two classes remain");
            if probability >= min_probability {
                LabelOutcome::Estimated { class, probability }
            } else {
                LabelOutcome::Unknown
            }
        })
        .collect())
}

/// Tolerance on the total of a posterior, the one the statistics kernel's
/// calibration measures apply.
const DISTRIBUTION_TOLERANCE: f64 = 1e-6;

/// Refuse a posterior that is not a probability distribution: empty, with an
/// entry that is negative or not finite, or not summing to one within
/// [`DISTRIBUTION_TOLERANCE`].
pub(crate) fn check_posterior(item: usize, posterior: &[f64]) -> Result<(), LabelingError> {
    let total: f64 = posterior.iter().sum();
    let valid = !posterior.is_empty()
        && posterior.iter().all(|p| p.is_finite() && *p >= 0.0)
        && (total - 1.0).abs() < DISTRIBUTION_TOLERANCE;
    if valid {
        Ok(())
    } else {
        Err(LabelingError::InvalidPosterior { item })
    }
}

/// Divide positive finite values by their total. A total that overflows, as
/// smoothing near `f64::MAX` makes it, is taken over the values divided by the
/// largest instead, so every share stays finite; otherwise the direct total
/// is used.
fn normalize(values: Vec<f64>) -> Vec<f64> {
    let total: f64 = values.iter().sum();
    if total.is_finite() {
        return values.into_iter().map(|value| value / total).collect();
    }
    let largest = values.iter().copied().fold(0.0, f64::max);
    let total: f64 = values.iter().map(|value| value / largest).sum();
    values
        .into_iter()
        .map(|value| value / largest / total)
        .collect()
}

fn softmax(logs: &[f64]) -> Vec<f64> {
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    normalize(logs.iter().map(|log| (log - max).exp()).collect())
}
