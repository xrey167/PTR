//! One world of F002 end to end: the product's pipeline (vote matrix, Dawid-Skene fit,
//! resolution with verifier precedence, gold evaluation, acquisition), the independent oracle's
//! reading of the same data, and the counters that compare them.
//!
//! Every call into the product goes through [`call`]: a panic is caught, counted in
//! `product_panics` and answered by skipping what depended on the call, so the run goes on to
//! its judgement. An `Err` of the product on an input the harness built validly is a counted
//! refusal (`unexpected_refusals`); an `Err` on an input the harness could not have built
//! validly (the vote matrix itself) is a harness fault.

use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

use ptr_analytics::{brier_score, expected_calibration_error, krippendorff_alpha_nominal, Binning};
use ptr_labeling::{
    evaluate, fit_label_model, function_accuracy, rank_for_annotation, resolve, Acquisition,
    DawidSkeneParams, EvaluationSet, FunctionKind, GoldLabel, GoldSampling, GoldSource, LabelModel,
    LabelOutcome, LabelSchema, LabelingFunction, ModelWarning, Resolution, Vote, VoteMatrix,
};

use super::generator::{Generated, PROBABILISTIC_NAMES, VERIFIER_NAMES};
use super::oracle::{self, Kind};
use super::params::{self, Shape, World, CLASSES, PROBABILISTIC, VERIFIERS};
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::rng::Rng;

/// The result of `call`, or `None` when it panicked; the panic is counted.
pub fn call<R>(tally: &mut Tally, product: impl FnOnce() -> R) -> Option<R> {
    let outcome = catch_unwind(AssertUnwindSafe(product)).ok();
    if outcome.is_none() {
        tally.add("product_panics", 1);
    }
    outcome
}

/// Silence the default panic message for the rest of the process; every contained panic is
/// counted anyway. The binary calls it once, before the run.
pub fn silence_contained_panics() {
    std::panic::set_hook(Box::new(|_| {}));
}

fn ds_params() -> DawidSkeneParams {
    DawidSkeneParams {
        max_iterations: params::DS_MAX_ITERATIONS,
        tolerance: params::DS_TOLERANCE_PPM as f64 / 1e6,
        smoothing: params::DS_SMOOTHING_MILLI as f64 / 1e3,
    }
}

fn min_probability() -> f64 {
    params::MIN_PROBABILITY_PERMILLE as f64 / 1e3
}

fn model_tolerance() -> f64 {
    params::MODEL_ORACLE_TOLERANCE_PPM as f64 / 1e6
}

fn metric_tolerance() -> f64 {
    params::METRIC_TOLERANCE_NANO as f64 / 1e9
}

/// `|a - b| <= tolerance`, false for a NaN.
fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

fn uniform() -> Vec<f64> {
    vec![1.0 / CLASSES as f64; CLASSES]
}

/// What the harness reads of one item's output, for the veto check.
#[derive(Clone, Debug)]
pub struct ItemOutput {
    /// The class the arm labels the item with, if it labels it.
    pub label: Option<usize>,
    /// The distribution the arm resolved the item from; `None` for a disputed item.
    pub posterior: Option<Vec<f64>>,
    /// The distribution the scorer is given for the item.
    pub scoring: Vec<f64>,
}

/// The items on which an arm contradicts a verifier veto, counted from the harness's own
/// vetoes: a label on a vetoed class, positive probability on a vetoed class (in the
/// resolution or in what the scorer is given), a label or a distribution on an item every class
/// of which is vetoed, other than the uniform substitute the scorer gets for it.
pub fn veto_violations(vetoed: &[Vec<usize>], outputs: &[ItemOutput]) -> usize {
    let substitute = uniform();
    vetoed
        .iter()
        .zip(outputs)
        .filter(|(ruled, output)| {
            let all = ruled.len() == CLASSES;
            let labelled_on_vetoed = output.label.is_some_and(|class| ruled.contains(&class));
            let posterior_on_vetoed = output
                .posterior
                .as_ref()
                .is_some_and(|p| ruled.iter().any(|class| p[*class] > 0.0));
            let scored_on_vetoed = if all {
                output
                    .scoring
                    .iter()
                    .zip(&substitute)
                    .any(|(a, b)| !close(*a, *b, 1e-12))
            } else {
                ruled.iter().any(|class| output.scoring[*class] > 0.0)
            };
            labelled_on_vetoed || posterior_on_vetoed || scored_on_vetoed
        })
        .count()
}

/// The classes each item's verifiers rule out, from the harness's data.
fn vetoed_classes(generated: &Generated) -> Vec<Vec<usize>> {
    (0..generated.truth.len())
        .map(|item| generated.vetoed_classes(item))
        .collect()
}

/// The product's vote matrix over the probabilistic functions `functions` and every verifier.
fn product_matrix(
    tally: &mut Tally,
    generated: &Generated,
    functions: &[usize],
) -> Option<VoteMatrix> {
    let mut declared: Vec<LabelingFunction> = functions
        .iter()
        .map(|function| {
            LabelingFunction::new(PROBABILISTIC_NAMES[*function], FunctionKind::Heuristic)
        })
        .collect();
    declared.extend(
        VERIFIER_NAMES
            .iter()
            .map(|name| LabelingFunction::new(*name, FunctionKind::Verifier)),
    );
    let rows: Vec<Vec<Vote>> = (0..generated.truth.len())
        .map(|item| {
            let mut row: Vec<Vote> = functions
                .iter()
                .map(|function| match generated.class_votes[item][*function] {
                    Some(class) => Vote::Class(class),
                    None => Vote::Abstain,
                })
                .collect();
            row.extend(generated.vetoes[item].iter().map(|veto| match veto {
                Some(class) => Vote::Veto(*class),
                None => Vote::Abstain,
            }));
            row
        })
        .collect();
    // The schema and the matrix are product code on input the harness built validly: a refusal is
    // a counted falsification and a panic a counted one, never a harness fault.
    let built = call(tally, || {
        let schema = LabelSchema::new((0..CLASSES).map(|class| format!("class-{class}")))?;
        VoteMatrix::new(schema, declared, rows)
    })?;
    match built {
        Ok(matrix) => Some(matrix),
        Err(_) => {
            tally.add("unexpected_refusals", 1);
            None
        }
    }
}

/// The model as plain numbers, for comparison with the oracle.
#[derive(Clone, Debug)]
pub struct ModelView {
    pub priors: Vec<f64>,
    pub confusion: Vec<Option<Vec<Vec<f64>>>>,
    pub posteriors: Vec<Vec<f64>>,
    pub warnings: Vec<String>,
}

fn warning_text(warning: &ModelWarning) -> String {
    match warning {
        ModelWarning::FewerThanThreeFunctions { count } => format!("fewer-than-three:{count}"),
        ModelWarning::NoOverlap { function } => format!("no-overlap:{function}"),
        ModelWarning::NotConverged { iterations } => format!("not-converged:{iterations}"),
    }
}

fn model_view(model: &LabelModel) -> ModelView {
    let mut warnings: Vec<String> = model.warnings().iter().map(warning_text).collect();
    warnings.sort();
    ModelView {
        priors: model.priors().to_vec(),
        confusion: model.confusion().to_vec(),
        posteriors: model.posteriors().to_vec(),
        warnings,
    }
}

fn all_close(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y, tolerance))
}

/// The disagreements between the product's fit and the oracle's: the priors, each confusion
/// matrix, the posterior of each item (one each) and the warnings.
pub fn model_mismatches(
    view: &ModelView,
    oracle_model: &oracle::Model,
    expected_warnings: &[String],
    functions: usize,
) -> usize {
    let tolerance = model_tolerance();
    let mut mismatches = 0;
    if !all_close(&view.priors, &oracle_model.priors, tolerance) {
        mismatches += 1;
    }
    for function in 0..view.confusion.len() {
        let agrees = match (&view.confusion[function], function < functions) {
            (Some(table), true) => {
                table.len() == CLASSES
                    && table.iter().enumerate().all(|(truth, row)| {
                        all_close(row, &oracle_model.confusion[function][truth], tolerance)
                    })
            }
            (None, false) => true,
            _ => false,
        };
        if !agrees {
            mismatches += 1;
        }
    }
    if view.confusion.len() != functions + VERIFIERS {
        mismatches += 1;
    }
    if view.posteriors.len() != oracle_model.posteriors.len() {
        mismatches += 1;
    } else {
        mismatches += view
            .posteriors
            .iter()
            .zip(&oracle_model.posteriors)
            .filter(|(a, b)| !all_close(a, b, tolerance))
            .count();
    }
    if view.warnings != expected_warnings {
        mismatches += 1;
    }
    mismatches
}

/// What the product says of one item, in the oracle's terms.
fn observed_item(outcome: &LabelOutcome, posterior: Option<&[f64]>) -> oracle::Expected {
    let (kind, class, probability) = match outcome {
        LabelOutcome::Determined { class } => (Kind::Determined, Some(*class), None),
        LabelOutcome::Estimated { class, probability } => {
            (Kind::Estimated, Some(*class), Some(*probability))
        }
        LabelOutcome::Disputed { .. } => (Kind::Disputed, None, None),
        LabelOutcome::Unknown => (Kind::Unknown, None, None),
    };
    oracle::Expected {
        kind,
        class,
        probability,
        posterior: posterior.map(<[f64]>::to_vec),
    }
}

/// The items on which the product's outcome differs from the oracle's reading of the same
/// vetoes, votes and model posterior.
pub fn resolution_mismatches(
    observed: &[oracle::Expected],
    expected: &[oracle::Expected],
) -> usize {
    if observed.len() != expected.len() {
        return observed.len().max(expected.len());
    }
    observed
        .iter()
        .zip(expected)
        .filter(|(seen, wanted)| {
            let same_probability = match (seen.probability, wanted.probability) {
                (Some(a), Some(b)) => close(a, b, 1e-9),
                (None, None) => true,
                _ => false,
            };
            let same_posterior = match (&seen.posterior, &wanted.posterior) {
                (Some(a), Some(b)) => all_close(a, b, 1e-9),
                (None, None) => true,
                _ => false,
            };
            !(seen.kind == wanted.kind
                && seen.class == wanted.class
                && same_probability
                && same_posterior)
        })
        .count()
}

/// The product's pipeline over one set of functions, with every check against the oracle.
pub struct Pipeline {
    pub matrix: VoteMatrix,
    pub model: LabelModel,
    pub resolution: Resolution,
    pub expected: Vec<oracle::Expected>,
    pub vetoed: Vec<Vec<usize>>,
    pub outputs: Vec<ItemOutput>,
}

impl Pipeline {
    /// The distributions the resolution hands to a scorer: its own posterior of each item, or
    /// the uniform substitute for a disputed one.
    pub fn scoring_vectors(&self) -> Vec<Vec<f64>> {
        self.outputs.iter().map(|o| o.scoring.clone()).collect()
    }

    /// The items on which this pipeline's labels and resolved distributions, or the vectors
    /// `scoring` that a scorer is actually given, contradict a verifier veto.
    pub fn veto_violations_with(&self, scoring: &[Vec<f64>]) -> usize {
        let outputs: Vec<ItemOutput> = self
            .outputs
            .iter()
            .zip(scoring)
            .map(|(output, vector)| ItemOutput {
                scoring: vector.clone(),
                ..output.clone()
            })
            .collect();
        veto_violations(&self.vetoed, &outputs)
    }
}

/// Build the matrix, fit twice, resolve twice, and compare each with the oracle and with
/// itself. The veto check is the caller's, on the vectors it hands to its scorer
/// ([`Pipeline::veto_violations_with`]). `None` when a product call panicked or refused
/// (counted) and nothing downstream can run.
pub fn run_pipeline(
    tally: &mut Tally,
    generated: &Generated,
    functions: &[usize],
) -> Result<Option<Pipeline>, String> {
    let Some(matrix) = product_matrix(tally, generated, functions) else {
        return Ok(None);
    };
    let items = generated.truth.len();
    let fitted = call(tally, || fit_label_model(&matrix, ds_params()));
    let again = call(tally, || fit_label_model(&matrix, ds_params()));
    let (model, repeat) = match (fitted, again) {
        (Some(Ok(model)), Some(Ok(repeat))) => (model, repeat),
        (Some(Err(_)), _) | (_, Some(Err(_))) => {
            tally.add("unexpected_refusals", 1);
            return Ok(None);
        }
        _ => return Ok(None),
    };
    if format!("{model:?}") != format!("{repeat:?}") {
        tally.add("determinism_mismatches", 1);
    }

    let columns: Vec<Vec<Option<usize>>> = functions
        .iter()
        .map(|function| {
            (0..items)
                .map(|item| generated.class_votes[item][*function])
                .collect()
        })
        .collect();
    let names: Vec<&str> = functions.iter().map(|f| PROBABILISTIC_NAMES[*f]).collect();
    let reference = oracle::fit(
        &columns,
        items,
        CLASSES,
        params::DS_SMOOTHING_MILLI as f64 / 1e3,
        params::DS_TOLERANCE_PPM as f64 / 1e6,
        params::DS_MAX_ITERATIONS,
    );
    let warnings = oracle::expected_warnings(&columns, &names, &reference);
    let view = model_view(&model);
    let mismatched = model_mismatches(&view, &reference, &warnings, functions.len());
    tally.add("model_oracle_mismatches", mismatched as i64);

    let resolved = call(tally, || resolve(&matrix, &model, min_probability()));
    let resolved_again = call(tally, || resolve(&matrix, &model, min_probability()));
    let (resolution, repeat) = match (resolved, resolved_again) {
        (Some(Ok(resolution)), Some(Ok(repeat))) => (resolution, repeat),
        (Some(Err(_)), _) | (_, Some(Err(_))) => {
            tally.add("unexpected_refusals", 1);
            return Ok(None);
        }
        _ => return Ok(None),
    };
    if format!("{resolution:?}") != format!("{repeat:?}") {
        tally.add("determinism_mismatches", 1);
    }

    let vetoed = vetoed_classes(generated);
    // The oracle resolves from the *product's* posteriors, so that a defect of the fit shows in
    // `model_oracle_mismatches` and a defect of the resolution in `resolution_oracle_mismatches`.
    let expected: Vec<oracle::Expected> = (0..items)
        .map(|item| {
            let voted = columns.iter().any(|column| column[item].is_some());
            oracle::expected_item(
                &vetoed[item],
                voted,
                &model.posteriors()[item],
                CLASSES,
                min_probability(),
            )
        })
        .collect();
    let observed: Vec<oracle::Expected> = (0..items)
        .map(|item| observed_item(&resolution.outcomes()[item], resolution.posterior(item)))
        .collect();
    tally.add(
        "resolution_oracle_mismatches",
        resolution_mismatches(&observed, &expected) as i64,
    );
    let outputs = outputs_of(&resolution, items);
    Ok(Some(Pipeline {
        matrix,
        model,
        resolution,
        expected,
        vetoed,
        outputs,
    }))
}

/// The harness's reading of the resolution: the label, the resolved distribution and the
/// distribution the scorer is given (the resolution's own, or the uniform one for a disputed
/// item, which has none).
fn outputs_of(resolution: &Resolution, items: usize) -> Vec<ItemOutput> {
    (0..items)
        .map(|item| {
            let label = match &resolution.outcomes()[item] {
                LabelOutcome::Determined { class } | LabelOutcome::Estimated { class, .. } => {
                    Some(*class)
                }
                LabelOutcome::Disputed { .. } | LabelOutcome::Unknown => None,
            };
            let posterior = resolution.posterior(item).map(<[f64]>::to_vec);
            let scoring = posterior.clone().unwrap_or_else(uniform);
            ItemOutput {
                label,
                posterior,
                scoring,
            }
        })
        .collect()
}

/// Majority vote: `(n_c + 1) / (n + C)` over the class votes of an item, with an epsilon per
/// class rank (`1e-9 * (C - c)`) added to the pseudo-count so that the scorer's argmax is the
/// lowest tied class; an item with no class vote is uniform. Verifier vetoes are ignored.
pub fn majority_posterior(
    votes: &[Option<usize>; PROBABILISTIC],
    vetoed: Option<&[usize]>,
) -> Vec<f64> {
    let mut weight = [0.0; CLASSES];
    let mut cast = 0usize;
    for vote in votes.iter().flatten() {
        weight[*vote] += 1.0;
        cast += 1;
    }
    if cast == 0 {
        return uniform();
    }
    for (class, value) in weight.iter_mut().enumerate() {
        *value += 1.0 + 1e-9 * (CLASSES - class) as f64;
        if vetoed.is_some_and(|ruled| ruled.contains(&class)) {
            *value = 0.0;
        }
    }
    let total: f64 = weight.iter().sum();
    if total == 0.0 {
        return uniform();
    }
    weight.iter().map(|value| value / total).collect()
}

/// The calibrated vote share (descriptive baseline `mvc`, `mvcp` with the veto restriction):
/// `p_c` proportional to `prior_c * exp(beta * n_c)` over the class votes of an item, with the
/// world's class priors and one sharpening exponent `beta` pinned from the pilot seeds
/// (`mvc_beta_milli`; the fit is the ignored test `fit_the_calibrated_vote_exponent`). It
/// is generous to the baseline: it knows the true priors and its exponent was tuned on this
/// generator. Classes in `vetoed` get probability zero.
pub fn calibrated_posterior(
    votes: &[Option<usize>; PROBABILISTIC],
    vetoed: Option<&[usize]>,
    beta: f64,
) -> Vec<f64> {
    let mut counts = [0.0; CLASSES];
    for vote in votes.iter().flatten() {
        counts[*vote] += 1.0;
    }
    let mut weight: Vec<f64> = (0..CLASSES)
        .map(|class| {
            if vetoed.is_some_and(|ruled| ruled.contains(&class)) {
                0.0
            } else {
                params::CLASS_PRIOR_PERMILLE[class] as f64 / 1e3 * (beta * counts[class]).exp()
            }
        })
        .collect();
    let total: f64 = weight.iter().sum();
    if total == 0.0 {
        return uniform();
    }
    for value in &mut weight {
        *value /= total;
    }
    weight
}

fn mvc_beta() -> f64 {
    params::MVC_BETA_MILLI as f64 / 1e3
}

/// A gold evaluation set of the oracle-labelled items, in the order of `items`.
fn gold_set(
    tally: &mut Tally,
    generated: &Generated,
    items: &[usize],
    sampling: GoldSampling,
) -> Option<EvaluationSet> {
    let mut set = EvaluationSet::new(sampling);
    for item in items {
        let label = GoldLabel {
            item: *item,
            class: generated.truth[*item],
            source: GoldSource::Oracle,
        };
        match call(tally, || set.push(label)) {
            Some(Ok(())) => {}
            Some(Err(_)) => {
                tally.add("unexpected_refusals", 1);
                return None;
            }
            None => return None,
        }
    }
    Some(set)
}

/// The scores of one arm on the gold items, as the product reports them and as the oracle
/// computes them.
#[derive(Clone, Copy, Debug)]
pub struct Scores {
    pub accuracy: f64,
    pub brier: f64,
    pub ece: f64,
}

/// The disagreements between two sets of scores (one count per differing quantity).
pub fn score_mismatches(seen: &Scores, wanted: &Scores) -> usize {
    let tolerance = metric_tolerance();
    usize::from(!close(seen.accuracy, wanted.accuracy, tolerance))
        + usize::from(!close(seen.brier, wanted.brier, tolerance))
        + usize::from(!close(seen.ece, wanted.ece, tolerance))
}

fn oracle_scores(predictions: &[Vec<f64>], truth: &[usize]) -> Scores {
    Scores {
        accuracy: oracle::correct(predictions, truth) as f64 / truth.len() as f64,
        brier: oracle::brier(predictions, truth),
        ece: oracle::ece_equal_mass(predictions, truth, params::CALIBRATION_BINS),
    }
}

/// An arm scored by the product's `evaluate` and by `ptr-analytics` directly, compared with
/// the oracle. Returns the product's scores (`None` when the product refused or panicked).
fn score_arm(
    tally: &mut Tally,
    scoring: &[Vec<f64>],
    gold: &EvaluationSet,
    order: &[usize],
    generated: &Generated,
) -> Option<Scores> {
    let predictions: Vec<Vec<f64>> = order.iter().map(|i| scoring[*i].clone()).collect();
    let truth: Vec<usize> = order.iter().map(|i| generated.truth[*i]).collect();
    let wanted = oracle_scores(&predictions, &truth);
    let report = match call(tally, || evaluate(scoring, gold, params::CALIBRATION_BINS))? {
        Ok(report) => report,
        Err(_) => {
            tally.add("unexpected_refusals", 1);
            return None;
        }
    };
    let Some(calibration) = report.calibration else {
        // A uniform set must carry calibration; its absence is a defect of the product.
        tally.add("unexpected_refusals", 1);
        return None;
    };
    let seen = Scores {
        accuracy: report.accuracy,
        brier: calibration.brier,
        ece: calibration.expected_calibration_error,
    };
    tally.add("metric_mismatches", score_mismatches(&seen, &wanted) as i64);
    // The statistics kernel directly, on the same vectors.
    let kernel = call(tally, || {
        (
            brier_score(&predictions, &truth),
            expected_calibration_error(
                &predictions,
                &truth,
                Binning::EqualMass(params::CALIBRATION_BINS),
            ),
        )
    })?;
    match kernel {
        (Ok(brier), Ok(ece)) => {
            tally.add(
                "metric_mismatches",
                usize::from(!close(brier, wanted.brier, metric_tolerance())) as i64
                    + usize::from(!close(ece, wanted.ece, metric_tolerance())) as i64,
            );
        }
        _ => tally.add("unexpected_refusals", 1),
    }
    Some(seen)
}

fn ppm(value: f64) -> i64 {
    (value * 1e6).round() as i64
}

fn permille_of(correct: usize, total: usize) -> i64 {
    (correct * 1000 / total.max(1)) as i64
}

/// Krippendorff alpha of the probabilistic functions' class votes (units are items,
/// abstentions are missing), product against oracle. `(ppm, defined)`.
fn alpha_check(tally: &mut Tally, generated: &Generated) -> (i64, i64) {
    let units: Vec<Vec<Option<usize>>> = generated
        .class_votes
        .iter()
        .map(|votes| votes.to_vec())
        .collect();
    let wanted = oracle::alpha_nominal(&units, CLASSES);
    let seen = match call(tally, || krippendorff_alpha_nominal(&units)) {
        Some(seen) => seen,
        None => return (0, 0),
    };
    let agrees = match (seen, wanted) {
        (Some(a), Some(b)) => close(a, b, metric_tolerance()),
        (None, None) => true,
        _ => false,
    };
    tally.add("metric_mismatches", i64::from(!agrees));
    match seen {
        Some(alpha) => (ppm(alpha), 1),
        None => (0, 0),
    }
}

/// `function_accuracy` of the product against the oracle's counts and Wilson intervals.
fn function_accuracy_check(
    tally: &mut Tally,
    pipeline: &Pipeline,
    generated: &Generated,
    gold: &EvaluationSet,
) {
    let z = params::WILSON_Z_MILLI as f64 / 1e3;
    let Some(result) = call(tally, || function_accuracy(&pipeline.matrix, gold, z)) else {
        return;
    };
    let Ok(accuracies) = result else {
        tally.add("unexpected_refusals", 1);
        return;
    };
    if accuracies.len() != PROBABILISTIC + VERIFIERS {
        tally.add("metric_mismatches", 1);
        return;
    }
    for (index, accuracy) in accuracies.iter().enumerate() {
        let (mut right, mut cast) = (0u64, 0u64);
        if index < PROBABILISTIC {
            for item in &generated.gold {
                if let Some(vote) = generated.class_votes[*item][index] {
                    cast += 1;
                    right += u64::from(vote == generated.truth[*item]);
                }
            }
        }
        let agrees = match accuracy.estimate {
            None => cast == 0,
            Some(estimate) => {
                let (low, high) = oracle::wilson(right, cast, z);
                cast > 0
                    && estimate.successes == right
                    && estimate.trials == cast
                    && close(estimate.low, low, metric_tolerance())
                    && close(estimate.high, high, metric_tolerance())
            }
        };
        tally.add("metric_mismatches", i64::from(!agrees));
    }
}

/// The acquisition checks of one ranking against the oracle's reading of the items: at most
/// `budget` distinct items, no determined item, every disputed item before any other (in item
/// order), and exactly `min(budget, non-determined items)` of them.
pub fn acquisition_violations(
    ranking: &[usize],
    expected: &[oracle::Expected],
    budget: usize,
) -> usize {
    let mut violations = 0;
    let unique: BTreeSet<usize> = ranking.iter().copied().collect();
    if ranking.len() > budget || unique.len() != ranking.len() {
        violations += 1;
    }
    if ranking.iter().any(|item| {
        expected
            .get(*item)
            .is_none_or(|e| e.kind == Kind::Determined)
    }) {
        violations += 1;
    }
    let disputed: Vec<usize> = (0..expected.len())
        .filter(|item| expected[*item].kind == Kind::Disputed)
        .collect();
    let leading = disputed.len().min(budget);
    if ranking.len() < leading || ranking[..leading] != disputed[..leading] {
        violations += 1;
    }
    let open = expected
        .iter()
        .filter(|e| e.kind != Kind::Determined)
        .count();
    if ranking.len() != budget.min(open) {
        violations += 1;
    }
    violations
}

/// What a world's run reports besides the tally.
pub struct WorldRun {
    pub pipeline: Option<Pipeline>,
    pub descriptive: Json,
}

fn count_outcomes(tally: &mut Tally, key: &str, resolution: &Resolution) -> [i64; 4] {
    let mut counts = [0i64; 4];
    for outcome in resolution.outcomes() {
        let slot = match outcome {
            LabelOutcome::Determined { .. } => 0,
            LabelOutcome::Estimated { .. } => 1,
            LabelOutcome::Unknown => 2,
            LabelOutcome::Disputed { .. } => 3,
        };
        counts[slot] += 1;
    }
    for (name, count) in ["determined", "estimated", "unknown", "disputed"]
        .iter()
        .zip(counts)
    {
        tally.add(&format!("{name}_seen"), count);
        tally.set(&format!("{key}_{name}_items"), count);
    }
    counts
}

/// The paired bootstrap of the Brier and ECE gains (majority vote minus the label model) over
/// the gold positions: `(brier lower, ece lower)` in ppm, the lower end of the pinned interval.
/// Uses the oracle's metrics only.
fn bootstrap_lower(
    seed: u64,
    world: World,
    resamples: usize,
    ds: &[Vec<f64>],
    mv: &[Vec<f64>],
    truth: &[usize],
) -> (i64, i64) {
    let mut rng = Rng::new(params::BOOTSTRAP_SEED ^ seed ^ world.salt());
    let size = truth.len();
    let (mut brier_gains, mut ece_gains) = (Vec::new(), Vec::new());
    for _ in 0..resamples {
        let picks: Vec<usize> = (0..size).map(|_| rng.index(size)).collect();
        let pick = |source: &[Vec<f64>]| -> Vec<Vec<f64>> {
            picks.iter().map(|i| source[*i].clone()).collect()
        };
        let sample_truth: Vec<usize> = picks.iter().map(|i| truth[*i]).collect();
        let (sample_ds, sample_mv) = (pick(ds), pick(mv));
        brier_gains.push(ppm(
            oracle::brier(&sample_mv, &sample_truth) - oracle::brier(&sample_ds, &sample_truth)
        ));
        ece_gains.push(ppm(oracle::ece_equal_mass(
            &sample_mv,
            &sample_truth,
            params::CALIBRATION_BINS,
        ) - oracle::ece_equal_mass(
            &sample_ds,
            &sample_truth,
            params::CALIBRATION_BINS,
        )));
    }
    brier_gains.sort_unstable();
    ece_gains.sort_unstable();
    let rank = resamples * (1000 - params::BOOTSTRAP_INTERVAL_PERMILLE as usize) / 2000;
    (brier_gains[rank], ece_gains[rank])
}

/// Run one world: the generator, the pipeline, the three arms on a uniform gold sample, the
/// statistics checks, the acquisition and the gold guards.
pub fn run_world(
    tally: &mut Tally,
    seed: u64,
    world: World,
    shape: Shape,
    generated: &Generated,
) -> Result<WorldRun, String> {
    let key = world.key();
    let items = generated.truth.len();
    // Every per-world metric starts at zero and is overwritten by what the run observes: a
    // world whose pipeline fails leaves zeros, the failure is in its counters, and the gain
    // rules fail.
    set_empty_world(tally, key);
    let all_functions: Vec<usize> = (0..PROBABILISTIC).collect();
    // Integrity of the generator: a sound world never vetoes the truth, a world without
    // verifiers has no veto, and a contaminated one does.
    let truth_vetoes = generated.vetoes_of_the_truth();
    if world.contamination_permille() == 0 && truth_vetoes != 0 {
        return Err(format!(
            "the generator vetoed the true class {truth_vetoes} times in the sound world {key}"
        ));
    }
    tally.add("items_total", items as i64);
    tally.add("worlds_run", 1);
    if world == World::Contaminated {
        tally.add("contaminated_vetoes", truth_vetoes as i64);
    }

    let Some(pipeline) = run_pipeline(tally, generated, &all_functions)? else {
        return Ok(WorldRun {
            pipeline: None,
            descriptive: Json::obj([("pipeline", Json::str("did not run"))]),
        });
    };
    if world == World::Sound {
        // Counted in the sound world alone, the world whose gain rules gate.
        tally.add("vetoed_items", {
            (0..items)
                .filter(|item| {
                    pipeline
                        .matrix
                        .row(*item)
                        .iter()
                        .any(|vote| matches!(vote, Vote::Veto(_)))
                })
                .count() as i64
        });
    }
    if world == World::Sound {
        let mut by_budget = [0usize; PROBABILISTIC + 1];
        let pair = params::CORRELATED_PAIR;
        let mut pair_items = 0;
        for item in 0..items {
            let row = pipeline.matrix.row(item);
            by_budget[row.iter().filter(|v| matches!(v, Vote::Class(_))).count()] += 1;
            if matches!(row[pair[0]], Vote::Class(_)) && matches!(row[pair[1]], Vote::Class(_)) {
                pair_items += 1;
            }
        }
        let cells = params::VOTES_PER_ITEM
            .iter()
            .filter(|m| by_budget[**m] >= 50)
            .count();
        tally.add("budget_cells_seen", cells as i64);
        tally.add("correlated_pair_items", pair_items);
    }
    let counts = count_outcomes(tally, key, &pipeline.resolution);
    let total = items as i64;
    tally.set(
        &format!("{key}_determined_permille"),
        1000 * counts[0] / total,
    );
    tally.set(
        &format!("{key}_estimated_permille"),
        1000 * counts[1] / total,
    );
    tally.set(&format!("{key}_unknown_permille"), 1000 * counts[2] / total);
    tally.set(
        &format!("{key}_disputed_permille"),
        1000 * counts[3] / total,
    );

    // Where the verifiers are sound, elimination cannot be wrong: a determined item with
    // another class than the truth is a defect of the product.
    if world.contamination_permille() == 0 {
        let wrong = (0..items)
            .filter(|item| {
                matches!(
                    pipeline.resolution.outcomes()[*item],
                    LabelOutcome::Determined { class } if class != generated.truth[*item]
                )
            })
            .count();
        tally.add("sound_world_determined_errors", wrong as i64);
    }

    // The arms' scoring vectors.
    let ds_scoring: Vec<Vec<f64>> = pipeline.scoring_vectors();
    tally.add(
        "veto_contradictions",
        pipeline.veto_violations_with(&ds_scoring) as i64,
    );
    let mv_scoring: Vec<Vec<f64>> = generated
        .class_votes
        .iter()
        .map(|votes| majority_posterior(votes, None))
        .collect();
    let mvp_scoring: Vec<Vec<f64>> = generated
        .class_votes
        .iter()
        .zip(&pipeline.vetoed)
        .map(|(votes, ruled)| {
            if ruled.len() == CLASSES {
                uniform()
            } else {
                majority_posterior(votes, Some(ruled))
            }
        })
        .collect();
    let mvc_scoring: Vec<Vec<f64>> = generated
        .class_votes
        .iter()
        .map(|votes| calibrated_posterior(votes, None, mvc_beta()))
        .collect();
    let mvcp_scoring: Vec<Vec<f64>> = generated
        .class_votes
        .iter()
        .zip(&pipeline.vetoed)
        .map(|(votes, ruled)| {
            if ruled.len() == CLASSES {
                uniform()
            } else {
                calibrated_posterior(votes, Some(ruled), mvc_beta())
            }
        })
        .collect();
    // The baseline ignores vetoes by design; how often it labels a vetoed class is descriptive.
    let mv_violating = (0..items)
        .filter(|item| pipeline.vetoed[*item].contains(&oracle::argmax(&mv_scoring[*item])))
        .count();
    tally.set(
        &format!("{key}_mv_veto_violating_items"),
        mv_violating as i64,
    );
    let mvp_violations = veto_violations(
        &pipeline.vetoed,
        &mvp_scoring
            .iter()
            .map(|scoring| ItemOutput {
                label: None,
                posterior: None,
                scoring: scoring.clone(),
            })
            .collect::<Vec<_>>(),
    );
    tally.set(
        &format!("{key}_mvp_veto_violating_items"),
        mvp_violations as i64,
    );

    // Verifier-only: only items the verifiers determine are labelled.
    let (mut vo_labelled, mut vo_correct) = (0i64, 0i64);
    for item in &generated.gold {
        if pipeline.vetoed[*item].len() == CLASSES - 1 {
            vo_labelled += 1;
            let class = (0..CLASSES)
                .find(|c| !pipeline.vetoed[*item].contains(c))
                .expect("one class left");
            vo_correct += i64::from(class == generated.truth[*item]);
        }
    }
    tally.set(&format!("{key}_vo_labelled_gold"), vo_labelled);
    tally.set(&format!("{key}_vo_correct_gold"), vo_correct);

    // The uniform gold sample and the three arms on it.
    let gold = gold_set(tally, generated, &generated.gold, GoldSampling::Uniform);
    let mut descriptive: Vec<(String, Json)> = Vec::new();
    if let Some(gold) = &gold {
        tally.set(&format!("{key}_gold_items"), gold.len() as i64);
        let ds = score_arm(tally, &ds_scoring, gold, &generated.gold, generated);
        let mv = score_arm(tally, &mv_scoring, gold, &generated.gold, generated);
        let mvp = score_arm(tally, &mvp_scoring, gold, &generated.gold, generated);
        let mvc = score_arm(tally, &mvc_scoring, gold, &generated.gold, generated);
        let mvcp = score_arm(tally, &mvcp_scoring, gold, &generated.gold, generated);
        for (name, scores) in [
            ("ds", ds),
            ("mv", mv),
            ("mvp", mvp),
            ("mvc", mvc),
            ("mvcp", mvcp),
        ] {
            if let Some(scores) = scores {
                let correct = (scores.accuracy * generated.gold.len() as f64).round() as usize;
                tally.set(
                    &format!("{key}_accuracy_{name}_permille"),
                    permille_of(correct, generated.gold.len()),
                );
                tally.set(&format!("{key}_brier_{name}_ppm"), ppm(scores.brier));
                tally.set(&format!("{key}_ece_{name}_ppm"), ppm(scores.ece));
            }
        }
        function_accuracy_check(tally, &pipeline, generated, gold);
        // The gain, from the scores the product reported.
        if let (Some(ds), Some(mv)) = (ds, mv) {
            tally.set(
                &format!("{key}_brier_gain_ppm"),
                ppm(mv.brier) - ppm(ds.brier),
            );
            tally.set(&format!("{key}_ece_gain_ppm"), ppm(mv.ece) - ppm(ds.ece));
            // Descriptive: the same gains over the calibrated votes (a negative number is a
            // recorded negative result, gated by nothing).
            for (name, arm) in [("mvc", mvc), ("mvcp", mvcp)] {
                if let Some(arm) = arm {
                    tally.set(
                        &format!("{key}_brier_gain_{name}_ppm"),
                        ppm(arm.brier) - ppm(ds.brier),
                    );
                    tally.set(
                        &format!("{key}_ece_gain_{name}_ppm"),
                        ppm(arm.ece) - ppm(ds.ece),
                    );
                }
            }
            let predictions = |scoring: &[Vec<f64>]| -> Vec<Vec<f64>> {
                generated.gold.iter().map(|i| scoring[*i].clone()).collect()
            };
            let truth: Vec<usize> = generated.gold.iter().map(|i| generated.truth[*i]).collect();
            let (brier_lower, ece_lower) = bootstrap_lower(
                seed,
                world,
                shape.resamples,
                &predictions(&ds_scoring),
                &predictions(&mv_scoring),
                &truth,
            );
            tally.set(&format!("{key}_brier_gain_lower_ppm"), brier_lower);
            tally.set(&format!("{key}_ece_gain_lower_ppm"), ece_lower);
        }
    }
    // The census: every item is labelled by every arm, so the accuracy of the labels the arms
    // produce is exact on all of them (and has no sampling noise from the gold sample). The
    // gold sample is what an evaluator would have; the census is what the labels are.
    let everyone: Vec<usize> = (0..items).collect();
    if let Some(census) = gold_set(tally, generated, &everyone, GoldSampling::Uniform) {
        for (name, scoring) in [
            ("ds", &ds_scoring),
            ("mv", &mv_scoring),
            ("mvp", &mvp_scoring),
            ("mvc", &mvc_scoring),
            ("mvcp", &mvcp_scoring),
        ] {
            if let Some(scores) = score_arm(tally, scoring, &census, &everyone, generated) {
                let correct = (scores.accuracy * items as f64).round() as usize;
                tally.set(
                    &format!("{key}_census_accuracy_{name}_permille"),
                    permille_of(correct, items),
                );
            }
        }
    }
    let (alpha_ppm, alpha_defined) = alpha_check(tally, generated);
    tally.set(&format!("{key}_alpha_ppm"), alpha_ppm);
    tally.set(&format!("{key}_alpha_defined"), alpha_defined);

    // Acquisition and the active-set probe.
    for strategy in [Acquisition::Entropy, Acquisition::Margin] {
        let ranked = call(tally, || {
            rank_for_annotation(&pipeline.resolution, strategy, shape.budget)
        });
        let Some(ranked) = ranked else { continue };
        tally.add(
            "acquisition_violations",
            acquisition_violations(&ranked, &pipeline.expected, shape.budget) as i64,
        );
        tally.add("acquisition_items", ranked.len() as i64);
        if strategy == Acquisition::Entropy && !ranked.is_empty() {
            let mut seen = BTreeSet::new();
            let distinct: Vec<usize> = ranked.iter().copied().filter(|i| seen.insert(*i)).collect();
            if let Some(active) = gold_set(tally, generated, &distinct, GoldSampling::Active) {
                match call(tally, || {
                    evaluate(&ds_scoring, &active, params::CALIBRATION_BINS)
                }) {
                    Some(Ok(report)) => {
                        tally.add("active_set_probes", 1);
                        if report.calibration.is_some() {
                            tally.add("active_set_calibration_reported", 1);
                        }
                    }
                    Some(Err(_)) => tally.add("unexpected_refusals", 1),
                    None => {}
                }
            }
        }
    }
    if world == World::Sound {
        gold_guard_probes(tally, generated, &ds_scoring);
    }

    descriptive.push(("items".into(), Json::Int(items as i64)));
    descriptive.push((
        "outcomes".into(),
        Json::obj([
            ("determined", Json::Int(counts[0])),
            ("estimated", Json::Int(counts[1])),
            ("unknown", Json::Int(counts[2])),
            ("disputed", Json::Int(counts[3])),
        ]),
    ));
    descriptive.push((
        "iterations".into(),
        Json::Int(pipeline.model.iterations() as i64),
    ));
    descriptive.push((
        "warnings".into(),
        Json::Arr(
            pipeline
                .model
                .warnings()
                .iter()
                .map(|w| Json::str(warning_text(w)))
                .collect(),
        ),
    ));
    descriptive.push(("vetoes_of_the_truth".into(), Json::Int(truth_vetoes as i64)));
    Ok(WorldRun {
        pipeline: Some(pipeline),
        descriptive: Json::Obj(descriptive),
    })
}

/// Zero every per-world metric of a world whose pipeline could not run (the counters of the
/// failure are in the tally); the gain rules then fail.
fn set_empty_world(tally: &mut Tally, key: &str) {
    for name in PER_WORLD_METRICS {
        tally.set(&format!("{key}_{name}"), 0);
    }
}

/// The per-world metrics every world prints, after its prefix (`w1_`, `w2_`, `w3_`).
pub const PER_WORLD_METRICS: [&str; 43] = [
    "census_accuracy_mvc_permille",
    "census_accuracy_mvcp_permille",
    "accuracy_mvc_permille",
    "accuracy_mvcp_permille",
    "brier_mvc_ppm",
    "brier_mvcp_ppm",
    "ece_mvc_ppm",
    "ece_mvcp_ppm",
    "brier_gain_mvc_ppm",
    "ece_gain_mvc_ppm",
    "brier_gain_mvcp_ppm",
    "ece_gain_mvcp_ppm",
    "census_accuracy_ds_permille",
    "census_accuracy_mv_permille",
    "census_accuracy_mvp_permille",
    "accuracy_ds_permille",
    "accuracy_mv_permille",
    "accuracy_mvp_permille",
    "brier_ds_ppm",
    "brier_mv_ppm",
    "brier_mvp_ppm",
    "ece_ds_ppm",
    "ece_mv_ppm",
    "ece_mvp_ppm",
    "brier_gain_ppm",
    "ece_gain_ppm",
    "brier_gain_lower_ppm",
    "ece_gain_lower_ppm",
    "gold_items",
    "determined_items",
    "estimated_items",
    "unknown_items",
    "disputed_items",
    "determined_permille",
    "estimated_permille",
    "unknown_permille",
    "disputed_permille",
    "alpha_ppm",
    "alpha_defined",
    "vo_labelled_gold",
    "vo_correct_gold",
    "mv_veto_violating_items",
    "mvp_veto_violating_items",
];

/// The guards on gold labels and on evaluation inputs: each must be refused, and a refused
/// label leaves the set unchanged.
fn gold_guard_probes(tally: &mut Tally, generated: &Generated, scoring: &[Vec<f64>]) {
    let item = generated.gold[0];
    let other = generated.gold[1];
    let class = generated.truth[item];
    let wrong = (class + 1) % CLASSES;
    let mut set = EvaluationSet::new(GoldSampling::Uniform);
    let first = GoldLabel {
        item,
        class,
        source: GoldSource::Oracle,
    };
    match call(tally, || set.push(first)) {
        Some(Ok(())) => {}
        Some(Err(_)) => {
            tally.add("unexpected_refusals", 1);
            return;
        }
        None => return,
    }
    let probes = [
        // The same item again, with the same class and with another.
        GoldLabel {
            item,
            class,
            source: GoldSource::Oracle,
        },
        GoldLabel {
            item,
            class: wrong,
            source: GoldSource::Oracle,
        },
        // A human label in an oracle set.
        GoldLabel {
            item: other,
            class: generated.truth[other],
            source: GoldSource::Human {
                annotator: "annotator-1".into(),
            },
        },
    ];
    for label in probes {
        // A probe counts when the product answered it with a refusal (or accepted it, which is
        // the violation); a panic is counted as a panic and not as a probe.
        match call(tally, || set.push(label)) {
            Some(Err(_)) => tally.add("gold_guard_probes", 1),
            Some(Ok(())) => {
                tally.add("gold_guard_probes", 1);
                tally.add("gold_guard_violations", 1);
            }
            None => {}
        }
    }
    if set.len() != 1 {
        tally.add("gold_guard_violations", 1);
    }
    // A gold item without a posterior, and a posterior that is no distribution.
    let mut beyond = EvaluationSet::new(GoldSampling::Uniform);
    let mut broken = EvaluationSet::new(GoldSampling::Uniform);
    for (set, item) in [(&mut beyond, scoring.len()), (&mut broken, 0)] {
        let label = GoldLabel {
            item,
            class: 0,
            source: GoldSource::Oracle,
        };
        match call(tally, || set.push(label)) {
            Some(Ok(())) => {}
            Some(Err(_)) => {
                tally.add("unexpected_refusals", 1);
                return;
            }
            None => return,
        }
    }
    let bad_posteriors = vec![vec![0.9, 0.9, 0.0, 0.0]];
    for (set, posteriors) in [(&beyond, scoring), (&broken, bad_posteriors.as_slice())] {
        match call(tally, || {
            evaluate(posteriors, set, params::CALIBRATION_BINS)
        }) {
            Some(Err(_)) => tally.add("gold_guard_probes", 1),
            Some(Ok(_)) => {
                tally.add("gold_guard_probes", 1);
                tally.add("gold_guard_violations", 1);
            }
            None => {}
        }
    }
}

/// The two-function cell: the same world scored with only two probabilistic functions (and the
/// verifiers), which is the deliberate cell where the model must warn that Dawid-Skene is not
/// identifiable. It runs the same checks as a world; no arm is scored.
pub fn run_two_function_cell(tally: &mut Tally, generated: &Generated) -> Result<Json, String> {
    let pipeline = run_pipeline(tally, generated, &[0, 1])?;
    let mut warnings = 0;
    if let Some(pipeline) = &pipeline {
        let violations = pipeline.veto_violations_with(&pipeline.scoring_vectors());
        tally.add("veto_contradictions", violations as i64);
        // Counted only here, from the warnings the product returned for the two-function fit.
        warnings = pipeline
            .model
            .warnings()
            .iter()
            .filter(|w| matches!(w, ModelWarning::FewerThanThreeFunctions { .. }))
            .count() as i64;
        tally.add("few_function_warnings", warnings);
    }
    Ok(Json::obj([
        ("functions", Json::Int(2)),
        ("ran", Json::Bool(pipeline.is_some())),
        ("fewer_than_three_warnings", Json::Int(warnings)),
    ]))
}

// ---- negative controls ----

/// One negative control: what it doctors, the counter it must trip, and what it counted.
pub struct ControlResult {
    pub name: &'static str,
    pub counter: &'static str,
    pub observed: i64,
}

impl ControlResult {
    pub fn tripped(&self) -> bool {
        self.observed >= 1
    }

    pub fn json(&self) -> Json {
        Json::obj([
            ("name", Json::str(self.name)),
            ("counter", Json::str(self.counter)),
            ("observed", Json::Int(self.observed)),
            ("tripped", Json::Bool(self.tripped())),
        ])
    }
}

/// The negative controls, on the contaminated world's pipeline: each doctors one stage after the
/// product produced it and shows that the counter built to see that defect counts it.
pub fn negative_controls(pipeline: &Pipeline, generated: &Generated) -> Vec<ControlResult> {
    let items = generated.truth.len();
    let raw = pipeline.model.posteriors();
    // A resolver that ignores vetoes: argmax of the model's posterior, and that posterior.
    let ignoring: Vec<ItemOutput> = (0..items)
        .map(|item| ItemOutput {
            label: Some(oracle::argmax(&raw[item])),
            posterior: Some(raw[item].clone()),
            scoring: raw[item].clone(),
        })
        .collect();
    // A scorer that reads the model's raw posterior instead of the resolution's.
    let raw_scorer: Vec<ItemOutput> = pipeline
        .outputs
        .iter()
        .zip(raw)
        .map(|(output, raw)| ItemOutput {
            scoring: raw.clone(),
            ..output.clone()
        })
        .collect();
    // A model whose posteriors are nudged, and scores that are.
    let mut nudged = model_view(&pipeline.model);
    nudged.posteriors[0][0] += 1e-3;
    let functions = PROBABILISTIC;
    let columns: Vec<Vec<Option<usize>>> = (0..functions)
        .map(|f| (0..items).map(|i| generated.class_votes[i][f]).collect())
        .collect();
    let reference = oracle::fit(
        &columns,
        items,
        CLASSES,
        params::DS_SMOOTHING_MILLI as f64 / 1e3,
        params::DS_TOLERANCE_PPM as f64 / 1e6,
        params::DS_MAX_ITERATIONS,
    );
    let names: Vec<&str> = PROBABILISTIC_NAMES.to_vec();
    let warnings = oracle::expected_warnings(&columns, &names, &reference);
    let sample: Vec<Vec<f64>> = generated
        .class_votes
        .iter()
        .take(50)
        .map(|votes| majority_posterior(votes, None))
        .collect();
    let truth: Vec<usize> = generated.truth.iter().take(50).copied().collect();
    let honest = oracle_scores(&sample, &truth);
    let doctored = Scores {
        brier: honest.brier + 1e-3,
        ..honest
    };
    // A ranking that proposes a determined item.
    let determined = pipeline
        .expected
        .iter()
        .position(|e| e.kind == Kind::Determined);
    let bad_ranking: Vec<usize> = determined.into_iter().collect();
    // Two identical arms gain nothing: run the paired bootstrap on one vector twice.
    let (identical_brier_lower, identical_ece_lower) =
        bootstrap_lower(0, World::Sound, 50, &sample, &sample, &truth);
    let gain_violations = gain_gate_violations(&GainView {
        brier_gain: 0,
        ece_gain: 0,
        brier_lower: identical_brier_lower,
        ece_lower: identical_ece_lower,
        accuracy_ds: 700,
        accuracy_mv: 700,
        gate_brier_on_lower: params::BRIER_GATE_ON_LOWER_BOUND,
        gate_ece_on_lower: params::ECE_GATE_ON_LOWER_BOUND,
    });
    // A resolution with the wrong class on an item.
    let mut wrong_class = pipeline.expected.clone();
    if let Some(slot) = wrong_class.iter_mut().find(|e| e.class.is_some()) {
        slot.class = slot.class.map(|class| (class + 1) % CLASSES);
    }
    vec![
        ControlResult {
            name: "resolver ignoring vetoes",
            counter: "veto_contradictions",
            observed: veto_violations(&pipeline.vetoed, &ignoring) as i64,
        },
        ControlResult {
            name: "scorer reading the raw posterior",
            counter: "veto_contradictions",
            observed: veto_violations(&pipeline.vetoed, &raw_scorer) as i64,
        },
        ControlResult {
            name: "posteriors nudged by 1e-3",
            counter: "model_oracle_mismatches",
            observed: model_mismatches(&nudged, &reference, &warnings, functions) as i64,
        },
        ControlResult {
            name: "Brier score off by 1e-3",
            counter: "metric_mismatches",
            observed: score_mismatches(&doctored, &honest) as i64,
        },
        ControlResult {
            name: "ranking that proposes a determined item",
            counter: "acquisition_violations",
            observed: acquisition_violations(&bad_ranking, &pipeline.expected, 10) as i64,
        },
        ControlResult {
            name: "two identical arms through the paired bootstrap",
            counter: "gain_violations",
            observed: gain_violations as i64,
        },
        ControlResult {
            name: "resolution with a wrong class",
            counter: "resolution_oracle_mismatches",
            observed: resolution_mismatches(&wrong_class, &pipeline.expected) as i64,
        },
    ]
}

/// The numbers the gain gates read.
pub struct GainView {
    pub brier_gain: i64,
    pub ece_gain: i64,
    pub brier_lower: i64,
    pub ece_lower: i64,
    pub accuracy_ds: i64,
    pub accuracy_mv: i64,
    pub gate_brier_on_lower: bool,
    pub gate_ece_on_lower: bool,
}

/// How many of the three gain gates (Brier, ECE, accuracy) the numbers fail.
pub fn gain_gate_violations(view: &GainView) -> usize {
    let brier = if view.gate_brier_on_lower {
        view.brier_lower
    } else {
        view.brier_gain
    };
    let ece = if view.gate_ece_on_lower {
        view.ece_lower
    } else {
        view.ece_gain
    };
    usize::from(brier < params::MIN_GAIN_PPM)
        + usize::from(ece < params::MIN_GAIN_PPM)
        + usize::from(view.accuracy_ds < view.accuracy_mv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(label: Option<usize>, posterior: Option<Vec<f64>>) -> ItemOutput {
        let scoring = posterior.clone().unwrap_or_else(uniform);
        ItemOutput {
            label,
            posterior,
            scoring,
        }
    }

    #[test]
    fn a_label_or_mass_on_a_vetoed_class_is_a_contradiction() {
        let vetoed = vec![vec![0], vec![0, 1, 2], vec![]];
        let clean = vec![
            output(Some(1), Some(vec![0.0, 0.5, 0.25, 0.25])),
            output(Some(3), Some(vec![0.0, 0.0, 0.0, 1.0])),
            output(None, Some(vec![0.25; 4])),
        ];
        assert_eq!(veto_violations(&vetoed, &clean), 0);
        let mut labelled = clean.clone();
        labelled[0].label = Some(0);
        assert_eq!(veto_violations(&vetoed, &labelled), 1);
        let mut mass = clean.clone();
        mass[0].posterior = Some(vec![0.1, 0.4, 0.25, 0.25]);
        assert_eq!(veto_violations(&vetoed, &mass), 1);
        let mut scored = clean.clone();
        scored[0].scoring = vec![0.1, 0.4, 0.25, 0.25];
        assert_eq!(veto_violations(&vetoed, &scored), 1);
    }

    #[test]
    fn a_disputed_item_may_be_scored_only_with_the_uniform_substitute() {
        let vetoed = vec![vec![0, 1, 2, 3]];
        let disputed = vec![output(None, None)];
        assert_eq!(veto_violations(&vetoed, &disputed), 0);
        let mut labelled = disputed.clone();
        labelled[0].label = Some(2);
        assert_eq!(veto_violations(&vetoed, &labelled), 1);
        let mut skewed = disputed.clone();
        skewed[0].scoring = vec![0.7, 0.1, 0.1, 0.1];
        assert_eq!(veto_violations(&vetoed, &skewed), 1);
        let mut resolved = disputed;
        resolved[0].posterior = Some(vec![0.25; 4]);
        assert_eq!(veto_violations(&vetoed, &resolved), 1);
    }

    #[test]
    fn the_majority_vote_has_the_laplace_posterior_and_ties_go_to_the_lowest_class() {
        let votes = [Some(2), Some(2), Some(1), None, None];
        let posterior = majority_posterior(&votes, None);
        assert!((posterior.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // (n_c + 1) / (n + C): class 2 has 3 of 7.
        assert!((posterior[2] - 3.0 / 7.0).abs() < 1e-8);
        assert_eq!(oracle::argmax(&posterior), 2);
        let tie = majority_posterior(&[Some(3), Some(1), None, None, None], None);
        assert_eq!(
            oracle::argmax(&tie),
            1,
            "the scorer's argmax is the lowest tied class"
        );
        let restricted = majority_posterior(&[Some(3), Some(1), None, None, None], Some(&[1]));
        assert_eq!(restricted[1], 0.0);
        assert_eq!(oracle::argmax(&restricted), 3);
        assert_eq!(majority_posterior(&[None; 5], None), uniform());
    }

    #[test]
    fn the_acquisition_checks_see_a_determined_item_a_late_dispute_and_a_short_list() {
        let expect = |kind| oracle::Expected {
            kind,
            class: None,
            probability: None,
            posterior: None,
        };
        let items = vec![
            expect(Kind::Estimated),
            expect(Kind::Disputed),
            expect(Kind::Determined),
            expect(Kind::Unknown),
            expect(Kind::Disputed),
        ];
        assert_eq!(acquisition_violations(&[1, 4, 0, 3], &items, 10), 0);
        assert_eq!(acquisition_violations(&[1, 4], &items, 2), 0);
        assert!(
            acquisition_violations(&[1, 4, 2, 0, 3], &items, 10) > 0,
            "determined"
        );
        assert!(
            acquisition_violations(&[0, 1, 4, 3], &items, 10) > 0,
            "disputes first"
        );
        assert!(
            acquisition_violations(&[1, 4, 0], &items, 10) > 0,
            "short list"
        );
        assert!(
            acquisition_violations(&[1, 4, 0, 3, 3], &items, 10) > 0,
            "duplicate"
        );
        assert!(
            acquisition_violations(&[1, 4, 0], &items, 2) > 0,
            "over budget"
        );
    }

    #[test]
    fn the_gain_gates_count_each_failed_gate() {
        let view = |brier, ece, ds, mv| GainView {
            brier_gain: 0,
            ece_gain: 0,
            brier_lower: brier,
            ece_lower: ece,
            accuracy_ds: ds,
            accuracy_mv: mv,
            gate_brier_on_lower: true,
            gate_ece_on_lower: true,
        };
        assert_eq!(gain_gate_violations(&view(10, 10, 800, 790)), 0);
        assert_eq!(gain_gate_violations(&view(10, 10, 800, 800)), 0);
        assert_eq!(gain_gate_violations(&view(0, 10, 800, 790)), 1);
        assert_eq!(gain_gate_violations(&view(10, -5, 790, 800)), 2);
        assert_eq!(gain_gate_violations(&view(0, 0, 790, 800)), 3);
        let point = GainView {
            brier_gain: 5,
            ece_gain: 5,
            brier_lower: -9,
            ece_lower: -9,
            accuracy_ds: 1,
            accuracy_mv: 1,
            gate_brier_on_lower: false,
            gate_ece_on_lower: false,
        };
        assert_eq!(
            gain_gate_violations(&point),
            0,
            "the point gain is read when the gate says so"
        );
    }

    #[test]
    fn a_nudged_model_a_wrong_score_and_a_wrong_outcome_are_counted() {
        let columns: Vec<Vec<Option<usize>>> = (0..3)
            .map(|f| {
                (0..60)
                    .map(|i| Some((i + usize::from(i % 7 == f)) % CLASSES))
                    .collect()
            })
            .collect();
        let reference = oracle::fit(&columns, 60, CLASSES, 1.0, 1e-6, 1000);
        let names = ["a", "b", "c"];
        let warnings = oracle::expected_warnings(&columns, &names, &reference);
        let honest = ModelView {
            priors: reference.priors.clone(),
            confusion: reference
                .confusion
                .iter()
                .cloned()
                .map(Some)
                .chain(std::iter::repeat_n(None, VERIFIERS))
                .collect(),
            posteriors: reference.posteriors.clone(),
            warnings: warnings.clone(),
        };
        assert_eq!(model_mismatches(&honest, &reference, &warnings, 3), 0);
        let mut nudged = honest.clone();
        nudged.posteriors[5][0] += 1e-3;
        assert_eq!(model_mismatches(&nudged, &reference, &warnings, 3), 1);
        let mut priors = honest.clone();
        priors.priors[0] += 1e-3;
        assert_eq!(model_mismatches(&priors, &reference, &warnings, 3), 1);
        let mut verifier = honest.clone();
        verifier.confusion[3] = Some(vec![vec![0.25; CLASSES]; CLASSES]);
        assert_eq!(model_mismatches(&verifier, &reference, &warnings, 3), 1);
        let mut warned = honest;
        warned.warnings.push("fewer-than-three:2".into());
        assert_eq!(model_mismatches(&warned, &reference, &warnings, 3), 1);
        let scores = Scores {
            accuracy: 0.8,
            brier: 0.3,
            ece: 0.1,
        };
        assert_eq!(score_mismatches(&scores, &scores), 0);
        assert_eq!(
            score_mismatches(
                &Scores {
                    brier: 0.31,
                    ..scores
                },
                &scores
            ),
            1
        );
        assert_eq!(
            score_mismatches(
                &Scores {
                    accuracy: 0.7,
                    ece: 0.2,
                    ..scores
                },
                &scores
            ),
            2
        );
        let expected = oracle::Expected {
            kind: Kind::Estimated,
            class: Some(1),
            probability: Some(0.8),
            posterior: Some(vec![0.0, 0.8, 0.2, 0.0]),
        };
        assert_eq!(
            resolution_mismatches(
                std::slice::from_ref(&expected),
                std::slice::from_ref(&expected)
            ),
            0
        );
        let mut wrong = expected.clone();
        wrong.class = Some(2);
        assert_eq!(
            resolution_mismatches(&[wrong], std::slice::from_ref(&expected)),
            1
        );
        let mut short = expected.clone();
        short.posterior = None;
        assert_eq!(
            resolution_mismatches(&[short], std::slice::from_ref(&expected)),
            1
        );
        assert_eq!(resolution_mismatches(&[], &[expected]), 1);
    }

    #[test]
    fn a_panic_of_the_product_is_caught_counted_and_answered_with_none() {
        let mut tally = Tally::new(crate::experiments::f002::metrics::LIST);
        assert_eq!(call(&mut tally, || 7), Some(7));
        assert_eq!(tally.get("product_panics"), 0);
        std::panic::set_hook(Box::new(|_| {}));
        assert_eq!(call(&mut tally, || -> u8 { panic!("boom") }), None);
        let _ = std::panic::take_hook();
        assert_eq!(tally.get("product_panics"), 1);
    }

    /// The fit of the calibrated vote exponent `mvc_beta_milli`: a grid search (0 to 4 in steps of
    /// 0.05) of the exponent that minimises the mean Brier score of the calibrated vote share over
    /// all items of the worlds W1 and W3 of the pilot seeds (never a confirmatory seed). Run with
    /// `cargo test -p ptr-bench fit_the_calibrated_vote_exponent -- --ignored --nocapture`.
    #[test]
    #[ignore = "a one-off fit on the pilot seeds, not a regression test"]
    fn fit_the_calibrated_vote_exponent() {
        let shape = crate::experiments::f002::params::Scale::Pinned.shape();
        let seeds: Vec<u64> = std::iter::once(1001)
            .chain(1002..=1003)
            .chain(2001..=2024)
            .collect();
        let mut worlds = Vec::new();
        for seed in &seeds {
            for world in [World::Sound, World::Homogeneous] {
                worlds.push(crate::experiments::f002::generator::generate(
                    *seed, world, shape,
                ));
            }
        }
        let mut best = (f64::MAX, 0.0);
        for step in 0..=80 {
            let beta = f64::from(step) * 0.05;
            let (mut total, mut count) = (0.0, 0usize);
            for world in &worlds {
                let predictions: Vec<Vec<f64>> = world
                    .class_votes
                    .iter()
                    .map(|votes| calibrated_posterior(votes, None, beta))
                    .collect();
                total += oracle::brier(&predictions, &world.truth) * world.truth.len() as f64;
                count += world.truth.len();
            }
            let mean = total / count as f64;
            println!("beta {beta:.2}: mean Brier {mean:.5}");
            if mean < best.0 {
                best = (mean, beta);
            }
        }
        println!("best beta {:.2} (mean Brier {:.5})", best.1, best.0);
    }

    #[test]
    fn the_calibrated_vote_follows_the_priors_sharpens_with_agreement_and_respects_vetoes() {
        let beta = 1.7;
        let none = calibrated_posterior(&[None; PROBABILISTIC], None, beta);
        assert!((none.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            none[0] > none[1] && none[1] > none[2] && none[2] > none[3],
            "the prior orders it"
        );
        let votes = [Some(3), Some(3), Some(3), None, None];
        let agreed = calibrated_posterior(&votes, None, beta);
        assert_eq!(oracle::argmax(&agreed), 3);
        assert!(agreed[3] > 0.9, "{agreed:?}");
        let ruled = calibrated_posterior(&votes, Some(&[3]), beta);
        assert_eq!(ruled[3], 0.0);
        assert!((ruled.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(
            calibrated_posterior(&votes, Some(&[0, 1, 2, 3]), beta),
            uniform()
        );
    }
}
