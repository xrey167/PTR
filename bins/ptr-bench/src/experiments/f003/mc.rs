//! Monte-Carlo coverage of the Learn-then-Test guarantee.
//!
//! One replicate draws a fresh calibration set of eligible slice branches from the model,
//! adjudicates it with the simulated adjudicator through the product's own `triage` and
//! `adjudicate`, records a policy with `PolicyRecord::calibrate` and compares its threshold
//! with the oracle's. The guarantee is that, over the draw of the calibration set, the true
//! adjudicated harm rate of the chosen threshold exceeds alpha with probability at most
//! delta; replicates that certify nothing cannot violate it. The count of violations over all
//! replicates is held against the exact binomial critical value.
//!
//! The same samples feed the negative control: a rule that estimates without a bound
//! (`oracle::naive_threshold`) must exceed the same critical value.

use ptr_branch::{
    calibrate_threshold, AutoThreshold, BranchId, CalibrationSample, PolicyRecord, ThresholdRule,
};
use ptr_types::Probability;

use super::model::{self, Kind, Truth};
use super::oracle::{self, AdjudicationView, Suffix};
use super::params::{self, Shape};
use super::stream::{make_policy, product, threshold_index};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;

const MC_DOMAIN: u64 = 0xF003_3C00_0000_0002;

#[derive(Clone, Debug, Default)]
pub struct McOutcome {
    pub replicates: i64,
    pub certified: i64,
    pub never: i64,
    pub violations: i64,
    pub naive_certified: i64,
    pub naive_violations: i64,
    pub crc_checked: i64,
    /// Histogram of the chosen threshold index by hundreds (0..=10, `never` apart).
    pub by_hundred: [i64; 11],
    /// Per policy size (the shape's order): replicates that certified, did not, and violated.
    pub certified_by_size: Vec<i64>,
    pub never_by_size: Vec<i64>,
    pub violations_by_size: Vec<i64>,
    pub digest: u64,
    /// The digest after the first `DETERMINISM_REPLICATES` replicates, which the rerun repeats.
    pub prefix_digest: u64,
}

/// Run replicates `0..count` for `seed`. `count` is the shape's replicate count, or the prefix
/// the determinism rerun repeats.
pub fn run(
    seed: u64,
    shape: &Shape,
    count: usize,
    truth: &Truth,
    epsilon_permille: u32,
    t: &mut Tally,
) -> Result<McOutcome, String> {
    let rate = params::slice_rate();
    let Some(slice_policy) = make_policy(t, AutoThreshold::Never, rate) else {
        return Ok(McOutcome::default());
    };
    let report = Kind::Eligible.report(0);
    let rule = ThresholdRule::LearnThenTest {
        alpha: params::alpha(),
        delta: params::delta(),
    };
    let mut out = McOutcome {
        digest: 0xCBF2_9CE4_8422_2325,
        certified_by_size: vec![0; shape.policy_sizes.len()],
        never_by_size: vec![0; shape.policy_sizes.len()],
        violations_by_size: vec![0; shape.policy_sizes.len()],
        ..McOutcome::default()
    };
    for rep in 0..count {
        // The digest after the first replicates, taken before this one so that a replicate the
        // product starved of data (it folds nothing) cannot leave the prefix unset.
        if rep == params::DETERMINISM_REPLICATES {
            out.prefix_digest = out.digest;
        }
        let n = shape.policy_sizes[rep % shape.policy_sizes.len()];
        let mut rng = Rng::new(seed ^ MC_DOMAIN ^ (rep as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let mut pairs: Vec<(BranchId, CalibrationSample)> = Vec::with_capacity(n);
        let mut raw: Vec<(u32, bool)> = Vec::with_capacity(n);
        for i in 0..n {
            let ns = model::draw_namespace(&mut rng);
            let m = model::draw_score(ns, &mut rng);
            let harm = model::draw_harm(ns, m, 1, &mut rng);
            let id = format!("mc{rep}-{i}");
            let keys = [format!("{}:{i}", model::NAMESPACES[ns].name)];
            let label = oracle::adjudicate(
                seed,
                &AdjudicationView {
                    id: &id,
                    changed_keys: &keys,
                    verifier_codes: &[],
                    truth_harmful: harm,
                },
                epsilon_permille,
            );
            // A draw below the rate puts the branch in the calibration slice.
            let draw = rng.unit() * rate;
            let prob = Probability::new(model::score_f32(m)).unwrap_or_default();
            let Some(triage) = product(t, || slice_policy.triage(&report, prob, draw)) else {
                continue;
            };
            let Some(sample) = product(t, || triage.adjudicate(label)) else {
                continue;
            };
            pairs.push((BranchId(id), sample));
            raw.push((m, label));
        }
        let suffix = Suffix::new(&raw);
        let oracle_j = oracle::ltt_threshold(&suffix, params::alpha(), params::delta());
        let Some(record) = product(t, || PolicyRecord::calibrate("mc", rule, rate, &pairs)) else {
            continue;
        };
        let product_j = match threshold_index(record.policy().threshold()) {
            Ok(j) => j,
            Err(_) => {
                t.add("threshold_mismatches", 1);
                None
            }
        };
        if product_j != oracle_j {
            t.add("threshold_mismatches", 1);
        }
        out.replicates += 1;
        let size_index = rep % shape.policy_sizes.len();
        match product_j {
            Some(j) => {
                out.certified += 1;
                out.certified_by_size[size_index] += 1;
                out.by_hundred[(j / 100).min(10)] += 1;
                if truth.adjudicated_harm_above_at(j, epsilon_permille) > params::alpha() {
                    out.violations += 1;
                    out.violations_by_size[size_index] += 1;
                }
            }
            None => {
                out.never += 1;
                out.never_by_size[size_index] += 1;
            }
        }
        out.digest ^= product_j.map_or(9999, |j| j as u64) + 1;
        out.digest = out.digest.wrapping_mul(0x0000_0100_0000_01B3);
        if let Some(j) = oracle::naive_threshold(&suffix, params::alpha(), params::delta()) {
            out.naive_certified += 1;
            if truth.adjudicated_harm_above_at(j, epsilon_permille) > params::alpha() {
                out.naive_violations += 1;
            }
        }
        if rep % params::CRC_EVERY == 0 {
            let scored: Vec<CalibrationSample> = pairs.iter().map(|(_, s)| *s).collect();
            if let Some(conformal) = product(t, || calibrate_threshold(&scored, params::alpha())) {
                out.crc_checked += 1;
                if threshold_index(conformal) != Ok(oracle::crc_threshold(&suffix, params::alpha()))
                {
                    t.add("crc_mismatches", 1);
                }
            }
        }
    }
    if count <= params::DETERMINISM_REPLICATES {
        out.prefix_digest = out.digest;
    }
    Ok(out)
}
