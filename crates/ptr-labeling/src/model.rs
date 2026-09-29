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

/// A fitted label model: class priors, one confusion matrix per
/// probabilistic function (`confusion[j][true][voted]`) and the posterior
/// class distribution of every item. Verifier functions have no confusion
/// matrix; they are not modelled as noisy voters.
///
/// A model is bound to the vote matrix it was fitted on by that matrix's
/// [`VoteMatrix::digest`], which only [`fit_label_model`] sets: its
/// posteriors and confusion matrices describe those items and functions and
/// no others, and [`resolve`] refuses any other matrix.
///
/// Everything a model holds is set by [`fit_label_model`] and read only:
/// its fields are private, and [`Self::priors`], [`Self::confusion`],
/// [`Self::posteriors`], [`Self::iterations`] and [`Self::warnings`] lend
/// them out immutably. So the posteriors [`resolve`] reads are the ones the
/// fit computed for the matrix the digest names; they cannot be replaced by
/// another fit's, or by any other distributions, while the digest stays.
/// Nor can a warning be dropped or the iteration count changed. A caller may
/// copy the values out and use them as it likes, but cannot hand them back
/// as a model.
///
/// ```compile_fail
/// fn substitute(model: &mut ptr_labeling::LabelModel, other: Vec<Vec<f64>>) {
///     model.posteriors = other;
/// }
/// ```
///
/// ```compile_fail
/// fn nudge(model: &mut ptr_labeling::LabelModel) {
///     model.posteriors()[0][0] = 1.0;
/// }
/// ```
///
/// ```compile_fail
/// fn retune(model: &mut ptr_labeling::LabelModel, other: Vec<Option<Vec<Vec<f64>>>>) {
///     model.confusion = other;
/// }
/// ```
///
/// ```compile_fail
/// fn reweigh(model: &mut ptr_labeling::LabelModel, other: Vec<f64>) {
///     model.priors = other;
/// }
/// ```
///
/// ```compile_fail
/// fn hide(model: &mut ptr_labeling::LabelModel) {
///     model.warnings.clear();
/// }
/// ```
///
/// ```compile_fail
/// fn relabel(model: &mut ptr_labeling::LabelModel) {
///     model.iterations = 1;
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LabelModel {
    priors: Vec<f64>,
    confusion: Vec<Option<Vec<Vec<f64>>>>,
    iterations: usize,
    warnings: Vec<ModelWarning>,
    posteriors: Vec<Vec<f64>>,
    matrix: [u8; 32],
}

impl LabelModel {
    /// The fitted class priors, one per class of the schema.
    pub fn priors(&self) -> &[f64] {
        &self.priors
    }

    /// One entry per function of the matrix, in function order: the fitted
    /// confusion matrix `confusion[j][true][voted]` of a probabilistic
    /// function, `None` for a verifier.
    pub fn confusion(&self) -> &[Option<Vec<Vec<f64>>>] {
        &self.confusion
    }

    /// The posterior class distribution of every item, in item order, one
    /// entry per class of the schema. An item no probabilistic function voted
    /// on keeps the prior here (up to rounding), and is never resolved from
    /// it.
    pub fn posteriors(&self) -> &[Vec<f64>] {
        &self.posteriors
    }

    /// The number of EM iterations the fit ran.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The conditions under which this model should not be trusted as is.
    pub fn warnings(&self) -> &[ModelWarning] {
        &self.warnings
    }

    /// The digest of the vote matrix this model was fitted on.
    pub fn matrix_digest(&self) -> [u8; 32] {
        self.matrix
    }
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
        matrix: matrix.digest(),
    })
}

/// The label an item receives, or why it receives none, as [`resolve`]
/// decides it. The variants state what `resolve` guarantees of the outcomes
/// it returns; a value built by hand is only data, and no function of this
/// crate takes one ([`crate::rank_for_annotation`] reads outcomes only from a
/// [`Resolution`]).
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
    /// No class is determined and none reaches the required probability, no
    /// probabilistic function voted at all, or the classes the verifiers left
    /// carry less posterior mass than `f64::MIN_POSITIVE` (see [`resolve`]).
    /// Unknown is a valid answer.
    Unknown,
}

/// The outcome of every item of one vote matrix, each paired with the
/// posterior it was resolved from.
///
/// Only [`resolve`] makes one, from a model and the matrix it was fitted on,
/// and its fields are private: item `i`'s outcome and posterior were computed
/// together, from row `i` of that matrix and the model's posterior of item
/// `i`, and cannot be paired with another item's, reordered, truncated or
/// replaced. [`crate::rank_for_annotation`] ranks nothing else.
///
/// ```compile_fail
/// use ptr_labeling::{LabelOutcome, Resolution};
/// let forged = Resolution {
///     outcomes: vec![LabelOutcome::Unknown],
///     posteriors: vec![Some(vec![0.5, 0.5])],
///     matrix: [0; 32],
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    outcomes: Vec<LabelOutcome>,
    posteriors: Vec<Option<Vec<f64>>>,
    matrix: [u8; 32],
}

impl Resolution {
    /// The outcome of every item, in item order.
    pub fn outcomes(&self) -> &[LabelOutcome] {
        &self.outcomes
    }

    /// The outcomes alone, in item order.
    pub fn into_outcomes(self) -> Vec<LabelOutcome> {
        self.outcomes
    }

    /// The distribution over the schema's classes that item `item`'s outcome
    /// was resolved from, or `None` for an item that is `Disputed` (no class
    /// is left) or beyond the matrix.
    ///
    /// Every vetoed class has probability zero. The classes left share the
    /// model's posterior renormalized over them, as [`resolve`] computes it;
    /// where [`resolve`] computes no share (no probabilistic function voted
    /// on the item, or the classes left carry less posterior mass than
    /// `f64::MIN_POSITIVE`) they share it uniformly, since nothing the
    /// resolution holds tells them apart. A `Determined` item's one class left
    /// has probability one, and an `Estimated` item's class has exactly the
    /// probability its outcome states.
    pub fn posterior(&self, item: usize) -> Option<&[f64]> {
        self.posteriors.get(item)?.as_deref()
    }

    /// The digest of the vote matrix the items were resolved against.
    pub fn matrix_digest(&self) -> [u8; 32] {
        self.matrix
    }

    /// The number of items.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Whether there are no items. [`fit_label_model`] refuses a matrix
    /// without items, so a resolution of a fitted model never is.
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }
}

/// Resolve every item: verifier vetoes first, then the label model.
///
/// Vetoed classes get probability zero whatever the model says; the posterior
/// is renormalized over the classes that remain. Verifiers ruling out all
/// classes yield `Disputed`; leaving exactly one yields `Determined`, even
/// without probabilistic votes. If multiple classes remain and no
/// probabilistic function voted, the item is `Unknown`. Each outcome is
/// returned with the posterior it was resolved from
/// ([`Resolution::posterior`]).
///
/// The shares are computed only when the classes the verifiers left carry
/// posterior mass of at least `f64::MIN_POSITIVE`, the smallest normal `f64`.
/// The fit computes posteriors in log space but stores them as probabilities:
/// a class the model puts more than about 708 nats below the leading one is
/// stored as a subnormal, with an absolute rounding error of up to one
/// subnormal unit (`2^-1074`), or as zero. Over less mass than
/// `f64::MIN_POSITIVE` that error can be the whole share (a stored posterior
/// of `[1, 5e-324, 0]` with class 0 vetoed would estimate class 1 at
/// probability one, whatever the model's log posterior gives it), so such an
/// item is `Unknown`, as is one whose classes left carry no mass at all, even
/// when the model's exact posterior would reach the required probability.
/// Over at least `f64::MIN_POSITIVE`, subnormal rounding moves each share by
/// at most about `2.2e-16` times the number of classes left.
///
/// # Errors
/// Returns an error if `min_probability` is not finite and in `(0, 1]`, and
/// `LabelingError::MatrixMismatch` if `model` was fitted on a matrix other
/// than `matrix` (with another schema, other functions or other votes, even of
/// the same shape: its posteriors would be combined with this matrix's vetoes
/// and votes).
///
/// Before any item is resolved it also checks the model's posteriors: one per
/// item (`LabelingError::LengthMismatch` otherwise), each a probability
/// distribution over the schema's classes (`LabelingError::InvalidPosterior`
/// for the first with an entry outside `[0, 1]`, a total off one by `1e-6` or
/// more, or an entry count other than the schema's): a negative entry would
/// resolve to a "probability" above one, and an entry beyond the schema would
/// drop mass unseen. A model's posteriors are the ones [`fit_label_model`]
/// computed for the matrix its digest names, and nothing can replace them
/// ([`LabelModel`]), so these checks guard against a defect in the fit, not
/// against a caller's substitution.
pub fn resolve(
    matrix: &VoteMatrix,
    model: &LabelModel,
    min_probability: f64,
) -> Result<Resolution, LabelingError> {
    if !(min_probability.is_finite() && min_probability > 0.0 && min_probability <= 1.0) {
        return Err(LabelingError::InvalidParameter {
            field: "min_probability",
            message: "must lie in (0, 1]",
        });
    }
    if model.matrix != matrix.digest() {
        return Err(LabelingError::MatrixMismatch);
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
    let (outcomes, posteriors) = (0..matrix.items())
        .map(|item| {
            resolve_item(
                matrix.row(item),
                &model.posteriors[item],
                classes,
                min_probability,
            )
        })
        .unzip();
    Ok(Resolution {
        outcomes,
        posteriors,
        matrix: model.matrix,
    })
}

/// One item's outcome and the posterior over the classes left that it was
/// resolved from (see [`Resolution::posterior`]).
fn resolve_item(
    row: &[Vote],
    posterior: &[f64],
    classes: usize,
    min_probability: f64,
) -> (LabelOutcome, Option<Vec<f64>>) {
    let vetoed: BTreeSet<usize> = row
        .iter()
        .filter_map(|vote| match vote {
            Vote::Veto(class) => Some(*class),
            Vote::Class(_) | Vote::Abstain => None,
        })
        .collect();
    let remaining: Vec<usize> = (0..classes).filter(|c| !vetoed.contains(c)).collect();
    match remaining.as_slice() {
        [] => return (LabelOutcome::Disputed { vetoed }, None),
        [only] => {
            let certain = spread(classes, &remaining, |_| 1.0);
            return (LabelOutcome::Determined { class: *only }, Some(certain));
        }
        _ => {}
    }
    let uniform = 1.0 / remaining.len() as f64;
    let voted = row.iter().any(|vote| matches!(vote, Vote::Class(_)));
    if !voted {
        let even = spread(classes, &remaining, |_| uniform);
        return (LabelOutcome::Unknown, Some(even));
    }
    let mass: f64 = remaining.iter().map(|&c| posterior[c]).sum();
    // Below the smallest normal f64 the stored posteriors of the classes left
    // have lost their ratio to underflow (and with no mass left every share
    // would be 0 / 0): the model says nothing reliable about them, so the item
    // is Unknown.
    if mass < f64::MIN_POSITIVE {
        let even = spread(classes, &remaining, |_| uniform);
        return (LabelOutcome::Unknown, Some(even));
    }
    let left = spread(classes, &remaining, |class| posterior[class] / mass);
    let (class, probability) = remaining
        .iter()
        .map(|&c| (c, left[c]))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("at least two classes remain");
    let outcome = if probability >= min_probability {
        LabelOutcome::Estimated { class, probability }
    } else {
        LabelOutcome::Unknown
    };
    (outcome, Some(left))
}

/// A distribution over `classes` classes that gives each class in `remaining`
/// its `share` and every other class zero.
fn spread(classes: usize, remaining: &[usize], share: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut distribution = vec![0.0; classes];
    for &class in remaining {
        distribution[class] = share(class);
    }
    distribution
}

/// Tolerance on the total of a posterior, the one the statistics kernel's
/// calibration measures apply.
const DISTRIBUTION_TOLERANCE: f64 = 1e-6;

/// Refuse a posterior that is not a probability distribution: empty, with an
/// entry outside `[0, 1]` (NaN included), or not summing to one within
/// [`DISTRIBUTION_TOLERANCE`].
///
/// Every entry is held to `[0, 1]` on its own, whatever the tolerance on the
/// total: `[1.0000005, 0]` sums to one within it, but its first entry is not a
/// probability, and ranking and scoring would consume it as one.
pub(crate) fn check_posterior(item: usize, posterior: &[f64]) -> Result<(), LabelingError> {
    let total: f64 = posterior.iter().sum();
    let valid = !posterior.is_empty()
        && posterior.iter().all(|p| (0.0..=1.0).contains(p))
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

#[cfg(test)]
mod tests {
    //! Tests that set a model's posteriors directly, which only this crate
    //! can do: a caller reads them only as the fit computed them
    //! ([`LabelModel`]). Each exercises `resolve` on posteriors a fit need
    //! not produce (one missing or added, one that is not a distribution, or
    //! mass left below the smallest normal `f64` where the test chooses),
    //! pinning `resolve`'s own checks and arithmetic rather than the fit's.

    use super::*;
    use crate::acquisition::{rank_for_annotation, Acquisition};
    use crate::gold::{evaluate, EvaluationSet, GoldLabel, GoldSampling, GoldSource};
    use crate::votes::{LabelSchema, LabelingFunction};

    fn function(name: &str, kind: FunctionKind) -> LabelingFunction {
        LabelingFunction::new(name, kind)
    }

    /// `per_class` items of every one of `classes` classes on which each of
    /// `heuristics` functions votes that class and each of `verifiers` functions
    /// abstains: agreement from which the model learns the heuristics are
    /// accurate.
    fn agreeing(
        classes: usize,
        heuristics: usize,
        verifiers: usize,
        per_class: usize,
    ) -> Vec<Vec<Vote>> {
        (0..classes * per_class)
            .map(|item| {
                let mut row = vec![Vote::Class(item % classes); heuristics];
                row.extend(vec![Vote::Abstain; verifiers]);
                row
            })
            .collect()
    }

    fn two_heuristics_and_a_third(votes: Vec<Vec<Vote>>) -> VoteMatrix {
        VoteMatrix::new(
            LabelSchema::new(["a", "b"]).unwrap(),
            vec![
                function("f0", FunctionKind::Heuristic),
                function("f1", FunctionKind::Heuristic),
                function("f2", FunctionKind::Heuristic),
            ],
            votes,
        )
        .unwrap()
    }

    #[test]
    fn annotation_ranking_takes_outcomes_and_posteriors_only_from_one_resolution() {
        let (c0, c1, c2, a) = (
            Vote::Class(0),
            Vote::Class(1),
            Vote::Class(2),
            Vote::Abstain,
        );
        let mut votes = vec![
            // Uncertain: the heuristics split three ways.
            vec![c0, c1, c2, a, a, a],
            // Settled by agreement.
            vec![c1, c1, c1, a, a, a],
            // Determined by elimination.
            vec![c2, a, a, Vote::Veto(0), Vote::Veto(1), a],
            // Every class ruled out.
            vec![c0, a, a, Vote::Veto(0), Vote::Veto(1), Vote::Veto(2)],
            // Nothing probabilistic voted.
            vec![a, a, a, Vote::Veto(2), a, a],
        ];
        votes.extend(agreeing(3, 3, 3, 5));
        let matrix = VoteMatrix::new(
            LabelSchema::new(["a", "b", "c"]).unwrap(),
            vec![
                function("h1", FunctionKind::Heuristic),
                function("h2", FunctionKind::Heuristic),
                function("h3", FunctionKind::Heuristic),
                function("v1", FunctionKind::Verifier),
                function("v2", FunctionKind::Verifier),
                function("v3", FunctionKind::Verifier),
            ],
            votes,
        )
        .unwrap();
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        let resolution = resolve(&matrix, &model, 0.9).unwrap();
        assert_eq!(resolution.len(), matrix.items());
        assert_eq!(resolution.matrix_digest(), matrix.digest());
        // Ranking read outcomes and posteriors as two slices that only had to be
        // equally long: an uncertain item paired with another item's Determined
        // outcome was never proposed, and a settled one paired with a Disputed
        // outcome was proposed first. Each item's pair now comes from resolve.
        let outcomes = resolution.outcomes();
        assert_eq!(outcomes[0], LabelOutcome::Unknown);
        assert!(matches!(
            outcomes[1],
            LabelOutcome::Estimated { class: 1, .. }
        ));
        assert_eq!(outcomes[2], LabelOutcome::Determined { class: 2 });
        assert!(matches!(outcomes[3], LabelOutcome::Disputed { .. }));
        assert_eq!(outcomes[4], LabelOutcome::Unknown);
        for strategy in [Acquisition::Entropy, Acquisition::Margin] {
            let ranked = rank_for_annotation(&resolution, strategy, 99);
            // The dispute first, the determined item never, and the settled item
            // after both unresolved ones.
            assert_eq!(ranked[0], 3, "{strategy:?}");
            assert_eq!(ranked.len(), matrix.items() - 1, "{strategy:?}");
            assert!(!ranked.contains(&2), "{strategy:?}");
            let at = |item| ranked.iter().position(|ranked| *ranked == item).unwrap();
            assert!(at(1) > at(0) && at(1) > at(4), "{strategy:?} {ranked:?}");
        }
        // A model with a posterior missing or added never resolves, so no
        // shorter or longer list of either kind reaches ranking.
        let items = matrix.items();
        let fitted = model.posteriors.clone();
        model.posteriors.pop();
        assert_eq!(
            resolve(&matrix, &model, 0.9),
            Err(LabelingError::LengthMismatch {
                expected: items,
                actual: items - 1
            })
        );
        model.posteriors = fitted.clone();
        model.posteriors.push(fitted[0].clone());
        assert_eq!(
            resolve(&matrix, &model, 0.9),
            Err(LabelingError::LengthMismatch {
                expected: items,
                actual: items + 1
            })
        );
    }

    #[test]
    fn a_resolution_pairs_every_outcome_with_the_posterior_it_was_resolved_from() {
        let (c0, c1, a) = (Vote::Class(0), Vote::Class(1), Vote::Abstain);
        let mut votes = vec![
            vec![c0, c1, Vote::Veto(2), a],
            vec![c1, c1, a, a],
            vec![c0, a, Vote::Veto(0), Vote::Veto(1)],
            vec![c0, a, Vote::Veto(0), a],
            vec![a, a, Vote::Veto(0), a],
        ];
        votes.extend(agreeing(3, 2, 2, 5));
        let matrix = VoteMatrix::new(
            LabelSchema::new(["a", "b", "c"]).unwrap(),
            vec![
                function("h1", FunctionKind::Heuristic),
                function("h2", FunctionKind::Heuristic),
                function("v1", FunctionKind::Verifier),
                function("v2", FunctionKind::Verifier),
            ],
            votes,
        )
        .unwrap();
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        // Item 3's classes left carry less mass than the smallest normal f64.
        let unit = f64::from_bits(1);
        model.posteriors[3] = vec![1.0, unit, 0.0];
        let dispute = VoteMatrix::new(
            LabelSchema::new(["a", "b"]).unwrap(),
            vec![
                function("h", FunctionKind::Heuristic),
                function("v1", FunctionKind::Verifier),
                function("v2", FunctionKind::Verifier),
            ],
            vec![vec![c0, Vote::Veto(0), Vote::Veto(1)]],
        )
        .unwrap();
        let resolution = resolve(&matrix, &model, 0.5).unwrap();
        let posterior = |item: usize| resolution.posterior(item).unwrap().to_vec();

        // Vetoed classes have probability zero and the model's posterior is
        // renormalized over the rest.
        let raw = &model.posteriors[0];
        let left = posterior(0);
        assert_eq!(left[2], 0.0);
        assert_eq!(left[0], raw[0] / (raw[0] + raw[1]));
        assert_eq!(left[1], raw[1] / (raw[0] + raw[1]));
        // An estimate states exactly the probability its posterior gives it.
        match resolution.outcomes()[1] {
            LabelOutcome::Estimated { class, probability } => {
                assert_eq!(posterior(1)[class], probability)
            }
            ref other => panic!("{other:?}"),
        }
        // Determined by elimination: the one class left is certain.
        assert_eq!(
            resolution.outcomes()[2],
            LabelOutcome::Determined { class: 2 }
        );
        assert_eq!(posterior(2), vec![0.0, 0.0, 1.0]);
        // No share is computed from mass below f64::MIN_POSITIVE, nor for an
        // item no probabilistic function voted on: the classes left are uniform.
        assert!(unit < f64::MIN_POSITIVE);
        for item in [3, 4] {
            assert_eq!(resolution.outcomes()[item], LabelOutcome::Unknown, "{item}");
        }
        assert_eq!(posterior(3), vec![0.0, 0.5, 0.5]);
        assert_eq!(posterior(4), vec![0.0, 0.5, 0.5]);
        assert_eq!(resolution.posterior(matrix.items()), None);
        // A disputed item has no class left, so no posterior.
        let model = fit_label_model(&dispute, DawidSkeneParams::default()).unwrap();
        let resolution = resolve(&dispute, &model, 0.5).unwrap();
        assert!(matches!(
            resolution.outcomes()[0],
            LabelOutcome::Disputed { .. }
        ));
        assert_eq!(resolution.posterior(0), None);
    }

    #[test]
    fn a_posterior_that_is_not_a_distribution_never_reaches_annotation_ranking() {
        let matrix = two_heuristics_and_a_third(vec![vec![Vote::Class(0); 3]; 3]);
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        let confident = vec![0.99, 0.01];
        let even = vec![0.5, 0.5];
        // An empty posterior has no margin, a NaN one ranked below a confident
        // item, and one outside [0, 1] has a negative entropy. Ranking takes
        // only a resolution, and resolve refuses every one of them.
        for invalid in [
            vec![],
            vec![f64::NAN, f64::NAN],
            vec![2.0, -1.0],
            vec![0.5, 0.6],
        ] {
            model.posteriors = vec![confident.clone(), invalid.clone(), even.clone()];
            assert_eq!(
                resolve(&matrix, &model, 0.5),
                Err(LabelingError::InvalidPosterior { item: 1 }),
                "{invalid:?}"
            );
        }
    }

    #[test]
    fn resolution_refuses_a_posterior_that_is_not_a_distribution_over_the_schema() {
        let matrix = two_heuristics_and_a_third(vec![vec![Vote::Class(0); 3]]);
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        // Resolved as they stood, these gave a probability of 1.5, a class from
        // negative mass, and a class of a two-class schema while 0.8 of the mass
        // sat on a third entry.
        for posterior in [
            vec![1.5, -0.5],
            vec![-1.0, -3.0],
            vec![0.1, 0.1, 0.8],
            vec![1.0],
            vec![f64::NAN, 0.5],
            vec![0.6, 0.6],
        ] {
            model.posteriors = vec![posterior.clone()];
            assert_eq!(
                resolve(&matrix, &model, 0.5),
                Err(LabelingError::InvalidPosterior { item: 0 }),
                "{posterior:?}"
            );
        }
        model.posteriors = vec![vec![0.25, 0.75]];
        assert_eq!(
            resolve(&matrix, &model, 0.5).unwrap().into_outcomes(),
            vec![LabelOutcome::Estimated {
                class: 1,
                probability: 0.75
            }]
        );
    }

    #[test]
    fn a_model_with_no_mass_on_the_classes_left_resolves_to_unknown() {
        let schema = LabelSchema::new(["a", "b", "c"]).unwrap();
        let matrix = VoteMatrix::new(
            schema,
            vec![
                function("rule", FunctionKind::Heuristic),
                function("check", FunctionKind::Verifier),
            ],
            vec![vec![Vote::Class(0), Vote::Veto(0)]],
        )
        .unwrap();
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        // Every share of the classes left is 0 / 0, which reaches no required
        // probability however small, whatever the sign of the zeros.
        for posterior in [vec![1.0, 0.0, 0.0], vec![1.0, -0.0, 0.0]] {
            model.posteriors = vec![posterior];
            for min_probability in [f64::MIN_POSITIVE, 0.5, 1.0] {
                assert_eq!(
                    resolve(&matrix, &model, min_probability)
                        .unwrap()
                        .into_outcomes(),
                    vec![LabelOutcome::Unknown]
                );
            }
        }
    }

    #[test]
    fn a_posterior_entry_above_one_is_refused_whatever_the_tolerance_on_the_total() {
        let matrix = two_heuristics_and_a_third(vec![vec![Vote::Class(0); 3]]);
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        let mut uniform = EvaluationSet::new(GoldSampling::Uniform);
        let mut active = EvaluationSet::new(GoldSampling::Active);
        for set in [&mut uniform, &mut active] {
            set.push(GoldLabel {
                item: 0,
                class: 0,
                source: GoldSource::Oracle,
            })
            .unwrap();
        }
        // Each sums to one within the 1e-6 tolerance, and every entry is finite
        // and non-negative: they were accepted, ranked by a negative entropy and
        // scored as probabilities.
        for posterior in [vec![1.0000005, 0.0], vec![1.0 + f64::EPSILON, -0.0]] {
            model.posteriors = vec![posterior.clone()];
            assert_eq!(
                resolve(&matrix, &model, 0.5),
                Err(LabelingError::InvalidPosterior { item: 0 }),
                "{posterior:?}"
            );
            // Ranking takes only a resolution, so the refusal above keeps the
            // posterior from being ranked too.
            for set in [&uniform, &active] {
                assert_eq!(
                    evaluate(std::slice::from_ref(&posterior), set, 5),
                    Err(LabelingError::InvalidPosterior { item: 0 }),
                    "{posterior:?}"
                );
            }
        }
        // One is a probability, and so is a zero of either sign.
        for posterior in [vec![1.0, 0.0], vec![1.0, -0.0]] {
            model.posteriors = vec![posterior.clone()];
            assert_eq!(
                resolve(&matrix, &model, 0.5).unwrap().into_outcomes(),
                vec![LabelOutcome::Estimated {
                    class: 0,
                    probability: 1.0
                }],
                "{posterior:?}"
            );
            assert!(evaluate(std::slice::from_ref(&posterior), &uniform, 5).is_ok());
        }
    }

    #[test]
    fn posterior_mass_left_below_the_smallest_normal_f64_resolves_to_unknown() {
        let (c0, c1, c2, a) = (
            Vote::Class(0),
            Vote::Class(1),
            Vote::Class(2),
            Vote::Abstain,
        );
        let mut votes = vec![vec![c0, c0, a]; 10];
        votes.extend(vec![vec![c1, c1, a]; 10]);
        votes.extend(vec![vec![c2, c2, a]; 30]);
        votes.push(vec![c0, c0, Vote::Veto(0)]);
        let matrix = VoteMatrix::new(
            LabelSchema::new(["a", "b", "c"]).unwrap(),
            vec![
                function("h1", FunctionKind::Heuristic),
                function("h2", FunctionKind::Heuristic),
                function("v", FunctionKind::Verifier),
            ],
            votes,
        )
        .unwrap();
        // The smallest smoothed probability, about 7.8e-164, is a normal number,
        // so this smoothing is accepted.
        let model = fit_label_model(
            &matrix,
            DawidSkeneParams {
                smoothing: 3.98e-162,
                ..DawidSkeneParams::default()
            },
        )
        .unwrap();
        // Class 0 leads the last item by about 744 nats over class 1 and 745
        // over class 2, so the model gives class 1 about 0.75 of what the veto
        // leaves. Stored as probabilities, class 1 is one subnormal unit and
        // class 2 zero, and dividing by that residue estimated class 1 at
        // probability one.
        let posterior = &model.posteriors[50];
        assert_eq!(posterior[0], 1.0, "{posterior:?}");
        assert!(posterior[1] > 0.0, "{posterior:?}");
        assert!(
            posterior[1] + posterior[2] < f64::MIN_POSITIVE,
            "{posterior:?}"
        );
        for min_probability in [f64::MIN_POSITIVE, 0.5, 0.9, 1.0] {
            assert_eq!(
                resolve(&matrix, &model, min_probability)
                    .unwrap()
                    .outcomes()[50],
                LabelOutcome::Unknown,
                "{min_probability}"
            );
        }
        // The items no verifier touched keep their estimates.
        let outcomes = resolve(&matrix, &model, 0.9).unwrap().into_outcomes();
        for (item, class) in [(0, 0), (10, 1), (20, 2)] {
            assert!(
                matches!(outcomes[item], LabelOutcome::Estimated { class: estimated, .. } if estimated == class),
                "{item}: {:?}",
                outcomes[item]
            );
        }

        // At the boundary: two subnormals summing to exactly the smallest normal
        // f64 are resolved from their exact ratio, one unit less is not.
        let matrix = VoteMatrix::new(
            LabelSchema::new(["a", "b", "c"]).unwrap(),
            vec![
                function("rule", FunctionKind::Heuristic),
                function("check", FunctionKind::Verifier),
            ],
            vec![vec![Vote::Class(0), Vote::Veto(0)]],
        )
        .unwrap();
        let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        let quarter = f64::MIN_POSITIVE / 4.0;
        model.posteriors = vec![vec![1.0, 3.0 * quarter, quarter]];
        assert_eq!(
            resolve(&matrix, &model, 0.7).unwrap().into_outcomes(),
            vec![LabelOutcome::Estimated {
                class: 1,
                probability: 0.75
            }]
        );
        let unit = f64::from_bits(1);
        for posterior in [
            vec![1.0, 3.0 * quarter, quarter - unit],
            vec![1.0, unit, 0.0],
        ] {
            model.posteriors = vec![posterior.clone()];
            assert_eq!(
                resolve(&matrix, &model, f64::MIN_POSITIVE)
                    .unwrap()
                    .into_outcomes(),
                vec![LabelOutcome::Unknown],
                "{posterior:?}"
            );
        }
    }
}
