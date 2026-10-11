//! The independent oracle of F002: a second Dawid-Skene EM written from the documentation of
//! `ptr_labeling::fit_label_model` (not from its code), an expected resolution written from the
//! documentation of `ptr_labeling::resolve`, and independent Brier, ECE, accuracy, Krippendorff
//! alpha and Wilson computations written from their definitions.
//!
//! Nothing here imports `ptr-labeling` or `ptr-analytics`. It reads the harness's own generated
//! data (`generator.rs`), never a `VoteMatrix`, so a defect of the product's matrix, fit,
//! resolution or statistics cannot reach it. Conventions the definitions leave open are pinned
//! here and stated in PREREGISTRATION.md: the argmax of a distribution is its last maximal
//! class, and equal-mass calibration bins cut a stable ascending sort of the confidences at
//! `b * n / bins`.

/// What the documentation of `resolve` says an item's outcome is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Determined,
    Estimated,
    Disputed,
    Unknown,
}

/// The expected outcome of one item and the distribution it was resolved from.
#[derive(Clone, Debug, PartialEq)]
pub struct Expected {
    pub kind: Kind,
    pub class: Option<usize>,
    pub probability: Option<f64>,
    /// `None` exactly for a disputed item (every class ruled out).
    pub posterior: Option<Vec<f64>>,
}

/// A fitted model of the oracle.
#[derive(Clone, Debug)]
pub struct Model {
    pub priors: Vec<f64>,
    /// `confusion[function][true][voted]`.
    pub confusion: Vec<Vec<Vec<f64>>>,
    pub posteriors: Vec<Vec<f64>>,
    pub iterations: usize,
    pub converged: bool,
}

fn normalized(mut values: Vec<f64>) -> Vec<f64> {
    let total: f64 = values.iter().sum();
    for value in &mut values {
        *value /= total;
    }
    values
}

/// Dawid-Skene by expectation-maximisation over `columns[function][item]`, as documented:
/// the first posterior of an item is its smoothed vote count; each pass fits the priors and the
/// confusion matrices from the posteriors (additive smoothing) and then recomputes the
/// posteriors from them in log space; it stops when no posterior moves by more than
/// `tolerance`, or after `max_iterations`.
pub fn fit(
    columns: &[Vec<Option<usize>>],
    items: usize,
    classes: usize,
    smoothing: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Model {
    let mut posteriors: Vec<Vec<f64>> = (0..items)
        .map(|item| {
            let mut counts = vec![smoothing; classes];
            for column in columns {
                if let Some(vote) = column[item] {
                    counts[vote] += 1.0;
                }
            }
            normalized(counts)
        })
        .collect();
    let mut priors = vec![1.0 / classes as f64; classes];
    let mut confusion = vec![vec![vec![1.0 / classes as f64; classes]; classes]; columns.len()];
    let mut iterations = 0;
    let mut converged = false;
    while iterations < max_iterations {
        iterations += 1;
        let mut weight = vec![smoothing; classes];
        for row in &posteriors {
            for (class, mass) in row.iter().enumerate() {
                weight[class] += mass;
            }
        }
        priors = normalized(weight);
        for (function, column) in columns.iter().enumerate() {
            let mut counts = vec![vec![smoothing; classes]; classes];
            for (item, vote) in column.iter().enumerate() {
                if let Some(voted) = vote {
                    for truth in 0..classes {
                        counts[truth][*voted] += posteriors[item][truth];
                    }
                }
            }
            confusion[function] = counts.into_iter().map(normalized).collect();
        }
        let mut moved: f64 = 0.0;
        for item in 0..items {
            let mut logs: Vec<f64> = priors.iter().map(|prior| prior.ln()).collect();
            for (function, column) in columns.iter().enumerate() {
                if let Some(voted) = column[item] {
                    for truth in 0..classes {
                        logs[truth] += confusion[function][truth][voted].ln();
                    }
                }
            }
            let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let next = normalized(logs.iter().map(|log| (log - top).exp()).collect());
            for class in 0..classes {
                moved = moved.max((posteriors[item][class] - next[class]).abs());
            }
            posteriors[item] = next;
        }
        if moved <= tolerance {
            converged = true;
            break;
        }
    }
    Model {
        priors,
        confusion,
        posteriors,
        iterations,
        converged,
    }
}

/// The warnings the documentation says a fit carries, as comparable strings: fewer than three
/// functions, a function that never voted on an item another function voted on, and a fit that
/// ran to its iteration limit.
pub fn expected_warnings(
    columns: &[Vec<Option<usize>>],
    names: &[&str],
    model: &Model,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if columns.len() < 3 {
        warnings.push(format!("fewer-than-three:{}", columns.len()));
    }
    let items = columns.first().map_or(0, Vec::len);
    for (function, column) in columns.iter().enumerate() {
        let overlaps = (0..items).any(|item| {
            column[item].is_some()
                && columns
                    .iter()
                    .enumerate()
                    .any(|(other, c)| other != function && c[item].is_some())
        });
        if !overlaps {
            warnings.push(format!("no-overlap:{}", names[function]));
        }
    }
    if !model.converged {
        warnings.push(format!("not-converged:{}", model.iterations));
    }
    warnings.sort();
    warnings
}

/// The outcome of one item as the documentation of `resolve` defines it. `vetoed` are the
/// classes the item's verifiers rule out, `voted` whether any probabilistic function voted on
/// it, `posterior` the model's distribution over the classes.
pub fn expected_item(
    vetoed: &[usize],
    voted: bool,
    posterior: &[f64],
    classes: usize,
    min_probability: f64,
) -> Expected {
    let remaining: Vec<usize> = (0..classes).filter(|c| !vetoed.contains(c)).collect();
    let spread = |share: &dyn Fn(usize) -> f64| -> Vec<f64> {
        let mut distribution = vec![0.0; classes];
        for class in &remaining {
            distribution[*class] = share(*class);
        }
        distribution
    };
    if remaining.is_empty() {
        return Expected {
            kind: Kind::Disputed,
            class: None,
            probability: None,
            posterior: None,
        };
    }
    if remaining.len() == 1 {
        return Expected {
            kind: Kind::Determined,
            class: Some(remaining[0]),
            probability: None,
            posterior: Some(spread(&|_| 1.0)),
        };
    }
    let even = 1.0 / remaining.len() as f64;
    let mass: f64 = remaining.iter().map(|c| posterior[*c]).sum();
    if !voted || mass < f64::MIN_POSITIVE {
        return Expected {
            kind: Kind::Unknown,
            class: None,
            probability: None,
            posterior: Some(spread(&|_| even)),
        };
    }
    let shares = spread(&|class| posterior[class] / mass);
    let mut best = remaining[0];
    for class in &remaining {
        if shares[*class] >= shares[best] {
            best = *class;
        }
    }
    let probability = shares[best];
    if probability >= min_probability {
        Expected {
            kind: Kind::Estimated,
            class: Some(best),
            probability: Some(probability),
            posterior: Some(shares),
        }
    } else {
        Expected {
            kind: Kind::Unknown,
            class: None,
            probability: None,
            posterior: Some(shares),
        }
    }
}

/// The argmax of a distribution: its last maximal class.
pub fn argmax(distribution: &[f64]) -> usize {
    let mut best = 0;
    for (class, value) in distribution.iter().enumerate() {
        if *value >= distribution[best] {
            best = class;
        }
    }
    best
}

/// How many predictions name the true class.
pub fn correct(predictions: &[Vec<f64>], truth: &[usize]) -> usize {
    predictions
        .iter()
        .zip(truth)
        .filter(|(distribution, class)| argmax(distribution) == **class)
        .count()
}

/// Multiclass Brier score: the mean squared distance to the one-hot truth.
pub fn brier(predictions: &[Vec<f64>], truth: &[usize]) -> f64 {
    let mut total = 0.0;
    for (distribution, class) in predictions.iter().zip(truth) {
        for (candidate, probability) in distribution.iter().enumerate() {
            let target = if candidate == *class { 1.0 } else { 0.0 };
            total += (probability - target) * (probability - target);
        }
    }
    total / truth.len() as f64
}

/// Expected calibration error, top-class probability as confidence, `bins` equal-mass bins of
/// the ascending (stable) order of the confidences.
pub fn ece_equal_mass(predictions: &[Vec<f64>], truth: &[usize], bins: usize) -> f64 {
    let mut scored: Vec<(f64, bool)> = predictions
        .iter()
        .zip(truth)
        .map(|(distribution, class)| {
            let top = argmax(distribution);
            (distribution[top], top == *class)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite confidence"));
    let n = scored.len();
    let bins = bins.min(n);
    let mut error = 0.0;
    for bin in 0..bins {
        let (low, high) = (bin * n / bins, (bin + 1) * n / bins);
        let size = (high - low) as f64;
        if high == low {
            continue;
        }
        let confidence: f64 = scored[low..high].iter().map(|s| s.0).sum::<f64>() / size;
        let accuracy = scored[low..high].iter().filter(|s| s.1).count() as f64 / size;
        error += size / n as f64 * (accuracy - confidence).abs();
    }
    error
}

/// Krippendorff's alpha for nominal data over units of optional ratings, from the coincidence
/// matrix; `None` when it is undefined.
pub fn alpha_nominal(units: &[Vec<Option<usize>>], classes: usize) -> Option<f64> {
    let mut coincidence = vec![vec![0.0; classes]; classes];
    for unit in units {
        let values: Vec<usize> = unit.iter().flatten().copied().collect();
        if values.len() < 2 {
            continue;
        }
        let weight = 1.0 / (values.len() as f64 - 1.0);
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() {
                if i != j {
                    coincidence[*a][*b] += weight;
                }
            }
        }
    }
    let marginals: Vec<f64> = coincidence.iter().map(|row| row.iter().sum()).collect();
    let n: f64 = marginals.iter().sum();
    let categories = marginals.iter().filter(|m| **m > 0.0).count();
    if n <= 1.0 || categories < 2 {
        return None;
    }
    let diagonal: f64 = (0..classes).map(|c| coincidence[c][c]).sum();
    let pairs: f64 = marginals.iter().map(|m| m * (m - 1.0)).sum();
    Some(((n - 1.0) * diagonal - pairs) / (n * (n - 1.0) - pairs))
}

/// The Wilson score interval `(low, high)` of `successes` in `trials` at normal quantile `z`,
/// clamped to `[0, 1]`.
pub fn wilson(successes: u64, trials: u64, z: f64) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let denominator = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denominator;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Checks of the oracle against values computed by hand, run at the start of every run: a
/// disagreement is a fault of the harness (exit 2), not a finding about the product.
pub fn self_check() -> Result<(), String> {
    let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
    let sure = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    if brier(&sure, &[0, 1]) != 0.0 || !near(brier(&[vec![0.5, 0.5]], &[0]), 0.5) {
        return Err("oracle self-check: Brier".into());
    }
    // Always 90% confident and right half the time: ECE = |0.5 - 0.9|.
    let confident = vec![vec![0.9, 0.1]; 4];
    if !near(ece_equal_mass(&confident, &[0, 1, 0, 1], 2), 0.4) {
        return Err("oracle self-check: ECE".into());
    }
    // Coincidences o_aa = 2, o_ab = o_ba = 1, o_bb = 4: (7 * 6 - 26) / (56 - 26) = 16 / 30.
    let units = vec![
        vec![Some(0), Some(0)],
        vec![Some(0), Some(1)],
        vec![Some(1), Some(1)],
        vec![Some(1), Some(1)],
    ];
    match alpha_nominal(&units, 2) {
        Some(alpha) if near(alpha, 16.0 / 30.0) => {}
        _ => return Err("oracle self-check: alpha".into()),
    }
    if alpha_nominal(&[vec![Some(0), Some(0)]], 2).is_some() {
        return Err("oracle self-check: alpha without variation".into());
    }
    // Wilson at 5 of 10, z = 1.96: the interval is symmetric about 0.5.
    let (low, high) = wilson(5, 10, 1.96);
    if !near(low + high, 1.0) || !(0.236 < low && low < 0.238) {
        return Err(format!("oracle self-check: Wilson {low} {high}"));
    }
    // Three accurate agreeing functions on two classes: the fit separates the classes.
    let columns: Vec<Vec<Option<usize>>> = (0..3)
        .map(|_| (0..20).map(|item| Some(item % 2)).collect())
        .collect();
    let model = fit(&columns, 20, 2, 1.0, 1e-9, 1000);
    if !model.converged || model.posteriors[0][0] < 0.9 || model.posteriors[1][1] < 0.9 {
        return Err("oracle self-check: fit".into());
    }
    let ruled = expected_item(&[0], true, &[0.5, 0.5], 2, 0.7);
    if ruled.kind != Kind::Determined || ruled.class != Some(1) {
        return Err("oracle self-check: determined".into());
    }
    if expected_item(&[0, 1], true, &[0.5, 0.5], 2, 0.7).kind != Kind::Disputed {
        return Err("oracle self-check: disputed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_self_check_passes() {
        assert_eq!(self_check(), Ok(()));
    }

    #[test]
    fn the_argmax_of_a_tie_is_the_last_class() {
        assert_eq!(argmax(&[0.4, 0.4, 0.2]), 1);
        assert_eq!(argmax(&[0.1, 0.2, 0.7]), 2);
    }

    #[test]
    fn a_resolution_renormalises_over_the_classes_the_verifiers_leave() {
        let posterior = [0.5, 0.3, 0.1, 0.1];
        let item = expected_item(&[0], true, &posterior, 4, 0.7);
        assert_eq!(item.kind, Kind::Unknown);
        let shares = item.posterior.expect("a posterior");
        assert_eq!(shares[0], 0.0);
        assert!((shares[1] - 0.6).abs() < 1e-12);
        let confident = expected_item(&[0], true, &[0.5, 0.45, 0.025, 0.025], 4, 0.7);
        assert_eq!(confident.kind, Kind::Estimated);
        assert_eq!(confident.class, Some(1));
        let unvoted = expected_item(&[], false, &[0.25; 4], 4, 0.7);
        assert_eq!(unvoted.kind, Kind::Unknown);
    }

    #[test]
    fn the_fit_recovers_an_accurate_function_and_warns_about_two_functions() {
        let columns: Vec<Vec<Option<usize>>> = vec![
            (0..200).map(|i| Some(i % 2)).collect(),
            (0..200)
                .map(|i| Some(if i % 10 == 0 { 1 - i % 2 } else { i % 2 }))
                .collect(),
        ];
        let model = fit(&columns, 200, 2, 1.0, 1e-6, 1000);
        assert!(model.confusion[0][0][0] > 0.9);
        let warnings = expected_warnings(&columns, &["a", "b"], &model);
        assert_eq!(warnings, vec!["fewer-than-three:2".to_string()]);
    }
}
