//! Off-policy evaluation of triage policies from logs of another policy.
//!
//! A log is `ope_log_branches` branches triaged by a logging policy (threshold 0.5, the
//! stream's slice rate) with a fresh uniform draw each; a branch that is auto-proposed earns
//! `REWARD_SAFE` or `REWARD_HARM` by its true harm, everything else earns zero. Two target
//! policies the log supports (a higher threshold; a lower threshold on a higher slice rate that
//! still keeps every action possible) are estimated by IPS and by doubly robust estimation with
//! a deliberately wrong reward model and with a nearly right one. The exact value of a target
//! is the model's (`Truth::eligible_value`), so the error of each estimate is known; across the
//! logs its mean must be within `OPE_BIAS_SIGMAS` standard errors of zero. A target below the
//! logging threshold takes an action the log never took, and both estimators must refuse it.
//!
//! The control estimator averages the logged rewards without weighting: it estimates the
//! logging policy, not the target, and must fail the same test.

use ptr_branch::{
    doubly_robust, evaluate_off_policy, ArbiterError, AutoThreshold, LoggedTriage, TriageDecision,
    TriagePolicy,
};
use ptr_types::Probability;

use super::guard;
use super::model::{self, Kind, Truth};
use super::params::{self, Shape};
use super::stats;
use super::stream::{make_policy, product};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;

const OPE_DOMAIN: u64 = 0xF003_0BE0_0000_0003;

/// A target policy: grid threshold in permille and slice rate in permille.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub name: &'static str,
    pub threshold_permille: u32,
    pub rate_permille: u32,
}

pub const TARGETS: [Target; 2] = [
    Target {
        name: "higher_threshold",
        threshold_permille: 800,
        rate_permille: 150,
    },
    Target {
        name: "lower_threshold_higher_slice",
        threshold_permille: 650,
        rate_permille: 300,
    },
];

/// The target below the logging threshold, which the log cannot support.
pub const UNSUPPORTED: Target = Target {
    name: "below_logging_threshold",
    threshold_permille: 300,
    rate_permille: 150,
};

/// The estimators and the control, in the order of the error lists.
pub const ESTIMATORS: [&str; 5] = [
    "ips",
    "snips",
    "dr_wrong_model",
    "dr_near_right_model",
    "unweighted_control",
];

/// Errors of one estimator on one target, across logs.
#[derive(Clone, Debug, Default)]
pub struct ErrorList {
    pub errors: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OpeOutcome {
    pub logs: i64,
    pub positivity_probes: i64,
    pub estimates: i64,
    /// `errors[target][estimator]`.
    pub errors: Vec<Vec<ErrorList>>,
    pub truth_values: Vec<f64>,
    pub digest: u64,
}

fn policy(t: &mut Tally, target: Target) -> Option<TriagePolicy> {
    make_policy(
        t,
        AutoThreshold::AtLeast(target.threshold_permille as f32 / 1000.0),
        f64::from(target.rate_permille) / 1000.0,
    )
}

/// The exact value of `target` per branch of the stream's mixture.
pub fn truth_value(truth: &Truth, target: Target) -> f64 {
    let eligible = f64::from(params::ELIGIBLE_PERMILLE) / 1000.0;
    eligible
        * truth.eligible_value(
            target.threshold_permille as usize,
            1.0 - f64::from(target.rate_permille) / 1000.0,
        )
}

fn is_positivity(result: &Result<impl Sized, ArbiterError>) -> bool {
    matches!(result, Err(ArbiterError::PositivityViolation { .. }))
}

/// Run logs `0..count` for `seed`.
pub fn run(
    seed: u64,
    shape: &Shape,
    count: usize,
    truth: &Truth,
    t: &mut Tally,
) -> Result<OpeOutcome, String> {
    let mut out = OpeOutcome {
        logs: 0,
        positivity_probes: 0,
        estimates: 0,
        errors: vec![vec![ErrorList::default(); ESTIMATORS.len()]; TARGETS.len()],
        truth_values: TARGETS.iter().map(|x| truth_value(truth, *x)).collect(),
        digest: 0xCBF2_9CE4_8422_2325,
    };
    let logging = make_policy(
        t,
        AutoThreshold::AtLeast(params::OPE_LOGGING_THRESHOLD_PERMILLE as f32 / 1000.0),
        params::slice_rate(),
    );
    let targets: Vec<Option<TriagePolicy>> = TARGETS.iter().map(|x| policy(t, *x)).collect();
    let unsupported = policy(t, UNSUPPORTED);
    let (Some(logging), Some(unsupported)) = (logging, unsupported) else {
        return Ok(out);
    };
    let Some(targets) = targets.into_iter().collect::<Option<Vec<_>>>() else {
        return Ok(out);
    };
    for l in 0..count {
        let mut rng = Rng::new(seed ^ OPE_DOMAIN ^ (l as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let mut log: Vec<LoggedTriage> = Vec::with_capacity(shape.ope_log_branches);
        for _ in 0..shape.ope_log_branches {
            let kind = model::draw_kind(&mut rng);
            let ns = model::draw_namespace(&mut rng);
            let m = if kind == Kind::Eligible {
                model::draw_score(ns, &mut rng)
            } else {
                model::draw_adversarial_score(&mut rng)
            };
            let harm = model::draw_harm(ns, m, 1, &mut rng);
            let draw = rng.unit();
            let variant = rng.next_u64();
            let report = kind.report(variant);
            let prob = Probability::new(model::score_f32(m)).unwrap_or_default();
            let Some(outcome) = product(t, || logging.triage(&report, prob, draw)) else {
                continue;
            };
            let Some(record) = guard::contained(|| {
                let mut record = LoggedTriage::from(&outcome);
                record.reward = if outcome.decision() == TriageDecision::AutoPropose {
                    if harm {
                        params::REWARD_HARM
                    } else {
                        params::REWARD_SAFE
                    }
                } else {
                    0.0
                };
                record
            }) else {
                t.add("product_panics", 1);
                continue;
            };
            log.push(record);
        }
        out.logs += 1;
        let unweighted = log.iter().map(|r| r.reward).sum::<f64>() / log.len().max(1) as f64;
        for (ti, target) in targets.iter().enumerate() {
            let value = out.truth_values[ti];
            let wrong = |_: &LoggedTriage, action: TriageDecision| {
                if action == TriageDecision::AutoPropose {
                    0.2
                } else {
                    0.0
                }
            };
            let near = |record: &LoggedTriage, action: TriageDecision| {
                if !record.eligible || action != TriageDecision::AutoPropose {
                    return 0.0;
                }
                let cell =
                    ((f64::from(record.score) * 1000.0).floor() as usize).min(model::CELLS - 1);
                let h = truth.marginal_harm(cell);
                params::REWARD_SAFE * (1.0 - h) + params::REWARD_HARM * h
            };
            let estimate = product(t, || evaluate_off_policy(&log, target));
            let dr_wrong = product(t, || doubly_robust(&log, target, wrong));
            let dr_near = product(t, || doubly_robust(&log, target, near));
            let lists = &mut out.errors[ti];
            if let Some(e) = estimate {
                lists[0].errors.push(e.ips - value);
                lists[1].errors.push(e.snips - value);
                out.estimates += 2;
                if l < params::DETERMINISM_LOGS {
                    out.digest ^= e.ips.to_bits();
                    out.digest = out.digest.wrapping_mul(0x0000_0100_0000_01B3);
                }
            } else {
                t.add("ope_errors", 1);
            }
            for (slot, result) in [(2, dr_wrong), (3, dr_near)] {
                match result {
                    Some(v) => {
                        lists[slot].errors.push(v - value);
                        out.estimates += 1;
                        if l < params::DETERMINISM_LOGS {
                            out.digest ^= v.to_bits();
                            out.digest = out.digest.wrapping_mul(0x0000_0100_0000_01B3);
                        }
                    }
                    None => t.add("ope_errors", 1),
                }
            }
            lists[4].errors.push(unweighted - value);
        }
        // The target the log cannot support must be refused, by both estimators.
        let refused_ips = guard_result(t, || evaluate_off_policy(&log, &unsupported));
        let refused_dr = guard_result(t, || doubly_robust(&log, &unsupported, |_, _| 0.0));
        for refused in [refused_ips, refused_dr] {
            out.positivity_probes += 1;
            if refused != Some(true) {
                t.add("positivity_unrefused", 1);
            }
        }
    }
    Ok(out)
}

/// Whether a call refused with a positivity violation: `None` when it panicked.
fn guard_result<T>(t: &mut Tally, call: impl FnOnce() -> Result<T, ArbiterError>) -> Option<bool> {
    match guard::contained(call) {
        Some(result) => Some(is_positivity(&result)),
        None => {
            t.add("product_panics", 1);
            None
        }
    }
}

/// The bias test of one error list: `(mean, standard error, trips)`, where `trips` is
/// `|mean| > OPE_BIAS_SIGMAS * standard error`.
pub fn bias_test(list: &ErrorList) -> (f64, f64, bool) {
    let (mean, sd) = stats::mean_sd(&list.errors);
    let se = sd / (list.errors.len().max(1) as f64).sqrt();
    let trips = if se > 0.0 {
        mean.abs() > params::OPE_BIAS_SIGMAS * se
    } else {
        mean != 0.0
    };
    (mean, se, trips)
}
