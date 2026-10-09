//! The held-out evaluation of a recorded policy.
//!
//! The held-out set of a policy is, by the preregistered rule, every adjudication recorded
//! after the ones it was calibrated on. The oracle derives that set by set difference over the
//! harness's own record of the stream and compares it with `PolicyRecord::held_out`; the test
//! then runs on the oracle's set, so a defective `held_out` is counted once, as overlap or
//! omission, and does not also produce a spurious violation.
//!
//! The test is the exact one-sided binomial test of "the harm rate of the branches the policy
//! would auto-propose is at most alpha" with the p-value `P(Bin(n, alpha) >= k)`, rejected at
//! the Bonferroni share of the familywise level. The literal rule of the manifest, the
//! Clopper-Pearson upper bound against alpha, is computed and printed but judged by nothing.

use std::collections::HashSet;

use ptr_analytics::clopper_pearson_upper;
use ptr_branch::{BranchId, CalibrationSample, PolicyRecord};

use super::guard;
use super::model::{admits, Truth};
use super::oracle::namespace_of;
use super::params;
use super::stats;
use super::stream::{product, threshold_index, Draft};
use crate::experiments::layered_kv::emit::Tally;

/// The evaluation of one policy.
#[derive(Clone, Debug)]
pub struct PolicyReport {
    pub version: String,
    pub n_cal: usize,
    pub product_j: Option<usize>,
    pub oracle_j: Option<usize>,
    pub heldout: usize,
    pub candidates: usize,
    pub harmful: usize,
    pub p_value: f64,
    pub violation: bool,
    /// The exact adjudicated harm rate at the chosen threshold, from the model.
    pub true_harm: Option<f64>,
    pub true_exceeds: bool,
    pub literal_ucb: Option<f64>,
    pub literal_exceeds: bool,
    /// The exact lower confidence bound on the held-out harm rate at the test's level: the test
    /// rejects iff it is above alpha.
    pub lower_bound: Option<f64>,
    /// `(namespace, candidates, harmful)`: the descriptive breakdown.
    pub breakdown: Vec<(String, usize, usize)>,
}

impl PolicyReport {
    /// The report of a policy the product could not produce (its construction failed or
    /// panicked, which is counted): nothing was held out and nothing is rejected.
    pub fn absent(version: &str) -> Self {
        Self {
            version: version.to_string(),
            n_cal: 0,
            product_j: None,
            oracle_j: None,
            heldout: 0,
            candidates: 0,
            harmful: 0,
            p_value: 1.0,
            violation: false,
            true_harm: None,
            true_exceeds: false,
            literal_ucb: None,
            literal_exceeds: false,
            lower_bound: None,
            breakdown: Vec::new(),
        }
    }
}

/// The test: whether `k` harmful among `n` auto-proposal candidates rejects "harm at most
/// alpha". No candidates reject nothing.
pub fn rejects(k: usize, n: usize) -> (f64, bool) {
    if n == 0 {
        return (1.0, false);
    }
    let p = stats::upper_p_value(k, n, params::alpha());
    (p, p < params::heldout_level())
}

/// Evaluate `record` against the adjudications of the stream. `n_cal` is the number of
/// adjudications the harness recorded before it, `oracle_j` the oracle's threshold index.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    t: &mut Tally,
    drafts: &[Draft],
    adjudicated: &[(BranchId, CalibrationSample)],
    index: &[usize],
    record: &PolicyRecord,
    n_cal: usize,
    oracle_j: Option<usize>,
    truth: &Truth,
) -> PolicyReport {
    // The oracle's calibration set: the first n_cal adjudications the harness recorded.
    let calibration: HashSet<&str> = index[..n_cal]
        .iter()
        .map(|&i| drafts[i].id.as_str())
        .collect();
    // Everything the product says about the record is read inside one containment; a panic is
    // counted and leaves the views empty, so the comparisons below fail loudly as well.
    let viewed = guard::contained(|| {
        (
            record
                .calibrated_on()
                .iter()
                .map(|b| b.0.clone())
                .collect::<Vec<String>>(),
            record
                .held_out(adjudicated)
                .iter()
                .map(|(b, _)| b.0.clone())
                .collect::<Vec<String>>(),
            threshold_index(record.policy().threshold()).unwrap_or_default(),
            record.version().to_string(),
        )
    });
    if viewed.is_none() {
        t.add("product_panics", 1);
    }
    let (cited_ids, held_list, product_j, version) = viewed.unwrap_or_default();
    let cited: HashSet<&str> = cited_ids.iter().map(String::as_str).collect();
    if cited != calibration {
        t.add("calibration_set_mismatches", 1);
    }
    let held_ids: HashSet<&str> = held_list.iter().map(String::as_str).collect();
    let overlap = held_ids
        .iter()
        .filter(|id| calibration.contains(*id))
        .count();
    t.add("held_out_overlap", overlap as i64);
    let oracle_held: Vec<usize> = index
        .iter()
        .copied()
        .filter(|&i| !calibration.contains(drafts[i].id.as_str()))
        .collect();
    let missing = oracle_held
        .iter()
        .filter(|&&i| !held_ids.contains(drafts[i].id.as_str()))
        .count();
    t.add("held_out_missing", missing as i64);

    let mut candidates = 0;
    let mut harmful = 0;
    let mut by_namespace: Vec<(String, usize, usize)> = Vec::new();
    if let Some(j) = product_j {
        for &i in &oracle_held {
            let d = &drafts[i];
            if !admits(j, d.m) {
                continue;
            }
            candidates += 1;
            harmful += usize::from(d.label);
            let mut seen: Vec<&str> = Vec::new();
            for key in &d.keys {
                let ns = namespace_of(key);
                if seen.contains(&ns) {
                    continue;
                }
                seen.push(ns);
                match by_namespace.iter_mut().find(|(name, _, _)| name == ns) {
                    Some(entry) => {
                        entry.1 += 1;
                        entry.2 += usize::from(d.label);
                    }
                    None => by_namespace.push((ns.to_string(), 1, usize::from(d.label))),
                }
            }
        }
    }
    by_namespace.sort();
    let (p_value, violation) = rejects(harmful, candidates);
    let true_harm = product_j.map(|j| truth.adjudicated_harm_above(j));
    let literal_ucb = if candidates > 0 {
        let ucb = product(t, || {
            clopper_pearson_upper(harmful as u64, candidates as u64, params::delta())
        });
        if let Some(value) = ucb {
            let reference = stats::cp_upper(harmful, candidates, params::delta());
            if (value - reference).abs() > BOUND_TOLERANCE {
                t.add("bound_oracle_mismatches", 1);
            }
        }
        ucb
    } else {
        None
    };
    PolicyReport {
        version,
        n_cal,
        product_j,
        oracle_j,
        heldout: oracle_held.len(),
        candidates,
        harmful,
        p_value,
        violation,
        true_exceeds: true_harm.is_some_and(|h| h > params::alpha()),
        true_harm,
        literal_exceeds: literal_ucb.is_some_and(|u| u > params::alpha()),
        literal_ucb,
        lower_bound: (candidates > 0)
            .then(|| stats::cp_lower(harmful, candidates, params::heldout_level())),
        breakdown: by_namespace,
    }
}

/// The agreement required between the product's Clopper-Pearson bound and the oracle's.
pub const BOUND_TOLERANCE: f64 = 1e-9;

/// Compare the product's bound with the oracle's on a fixed grid of counts, and on the closed
/// forms: `k = 0` is `1 - delta^(1/n)` and `k = n` is one. Returns the number of
/// disagreements and the number of pairs compared.
pub fn bound_grid(t: &mut Tally) -> (i64, i64) {
    let mut mismatches = 0;
    let mut compared = 0;
    for delta in [0.05, 0.01] {
        for n in [1_usize, 2, 3, 5, 10, 20, 59, 60, 100, 300, 1000, 2200] {
            for k in [0, 1, 2, n / 10, n / 2, n.saturating_sub(1), n] {
                if k > n {
                    continue;
                }
                let Some(value) = product(t, || clopper_pearson_upper(k as u64, n as u64, delta))
                else {
                    mismatches += 1;
                    continue;
                };
                compared += 1;
                let reference = stats::cp_upper(k, n, delta);
                let closed = if k == 0 {
                    Some(1.0 - delta.powf(1.0 / n as f64))
                } else if k == n {
                    Some(1.0)
                } else {
                    None
                };
                let bad = (value - reference).abs() > BOUND_TOLERANCE
                    || closed.is_some_and(|c| (c - reference).abs() > BOUND_TOLERANCE);
                if bad {
                    mismatches += 1;
                }
            }
        }
    }
    t.add("bound_oracle_mismatches", mismatches);
    (mismatches, compared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_test_rejects_exactly_when_the_exact_lower_bound_is_above_alpha() {
        for (k, n) in [
            (0, 500),
            (25, 500),
            (40, 500),
            (45, 500),
            (60, 500),
            (130, 1500),
        ] {
            let (_, rejected) = rejects(k, n);
            let lower = stats::cp_lower(k, n, params::heldout_level());
            assert_eq!(rejected, lower > params::alpha(), "{k}/{n} lower {lower}");
        }
        assert_eq!(rejects(0, 0), (1.0, false));
    }

    #[test]
    fn a_correct_policy_can_fail_the_literal_rule_and_pass_the_test() {
        // 18 harmful among 500 (3.6 percent): the true rate may well be below alpha, the upper
        // bound at 95 percent is not.
        let (_, rejected) = rejects(18, 500);
        assert!(!rejected);
        assert!(stats::cp_upper(18, 500, 0.05) > 0.05);
    }
}
