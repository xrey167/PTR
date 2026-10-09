//! The time-ordered stream of branches and the real triage pipeline over it.
//!
//! Every branch is triaged by the product three times, once per arm: the escalate-all policy
//! (threshold `Never`), the fixed uncalibrated policy, and the policy in force, which starts as
//! a manual `Never` and is replaced by `PolicyRecord::calibrate` under Learn-then-Test after
//! each policy size's worth of adjudications. The slice (`draw < rate`) does not depend on the
//! threshold, so the adjudication sequence is the same whichever policy is in force, and every
//! policy is calibrated on the adjudications recorded before it and held out on those after.
//!
//! Each triage is compared with the oracle's expectation (`oracle::expected`) and with
//! `TriagePolicy::explains`; each non-slice triage is offered to `adjudicate` and must be
//! refused; each recorded threshold is compared with the oracle's fixed-sequence rule.

use ptr_branch::{
    calibrate_threshold, calibration_draw, AutoThreshold, BranchId, CalibrationSample,
    PolicyRecord, ThresholdRule, TriageDecision, TriageOutcome, TriagePolicy,
};
use ptr_types::Probability;
use ptr_verifier::VerificationReport;

use super::guard;
use super::model::{self, Kind, NAMESPACES};
use super::oracle::{self, AdjudicationView, Decision, Suffix};
use super::params::{self, Shape};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;

const STREAM_DOMAIN: u64 = 0xF003_5712_EA11_0001;

/// One branch of the simulated stream.
pub struct Draft {
    pub id: String,
    pub branch: BranchId,
    pub kind: Kind,
    pub ns: usize,
    /// The keys the branch changes; the first segment of each is a namespace.
    pub keys: Vec<String>,
    /// Free text the agent wrote. The adjudicator is not shown it (`AdjudicationView` has no
    /// intent), so no function reads this field; it is kept to show that the stream has one.
    #[allow(dead_code)]
    pub intent: String,
    /// Micro-score of the scorer.
    pub m: u32,
    /// Ground truth: whether auto-proposing the branch would be harmful.
    pub harm: bool,
    pub report: VerificationReport,
    /// What the simulated adjudicator says if asked.
    pub label: bool,
}

impl Draft {
    pub fn probability(&self) -> Probability {
        Probability::new(model::score_f32(self.m)).unwrap_or_default()
    }
}

/// The deterministic stream for `seed`.
pub fn generate(seed: u64, shape: &Shape) -> Vec<Draft> {
    let mut rng = Rng::new(seed ^ STREAM_DOMAIN);
    let mut drafts = Vec::with_capacity(shape.stream_branches);
    for i in 0..shape.stream_branches {
        let kind = model::draw_kind(&mut rng);
        let ns = model::draw_namespace(&mut rng);
        let m = if kind == Kind::Eligible {
            model::draw_score(ns, &mut rng)
        } else {
            model::draw_adversarial_score(&mut rng)
        };
        let harm = model::draw_harm(ns, m, 1, &mut rng);
        let extra = rng.below(3) as usize;
        let mut keys = vec![format!("{}:{i}-0", NAMESPACES[ns].name)];
        for n in 0..extra {
            let other = if rng.below(10) < 8 {
                ns
            } else {
                rng.index(NAMESPACES.len())
            };
            keys.push(format!("{}:{i}-{}", NAMESPACES[other].name, n + 1));
        }
        let variant = rng.next_u64();
        let report = kind.report(variant);
        let id = format!("f003-{seed}-{i:06}");
        let codes: Vec<String> = report.findings.iter().map(|f| f.code.clone()).collect();
        let label = oracle::adjudicate(
            seed,
            &AdjudicationView {
                id: &id,
                changed_keys: &keys,
                verifier_codes: &codes,
                truth_harmful: harm,
            },
            params::EPSILON_PERMILLE,
        );
        drafts.push(Draft {
            branch: BranchId(id.clone()),
            intent: format!("agent {i}: change {} key(s)", keys.len()),
            id,
            kind,
            ns,
            keys,
            m,
            harm,
            report,
            label,
        });
    }
    drafts
}

/// The grid index of a product threshold: `Ok(None)` for "never", `Err` when an `AtLeast` value
/// is not on the grid.
pub fn threshold_index(threshold: AutoThreshold) -> Result<Option<usize>, f32> {
    match threshold {
        AutoThreshold::Never => Ok(None),
        AutoThreshold::AtLeast(value) => {
            let j = (value * 1000.0).round() as usize;
            if j <= 1000 && value == j as f32 / 1000.0 {
                Ok(Some(j))
            } else {
                Err(value)
            }
        }
    }
}

/// Run a product call that may fail: a panic counts into `product_panics`, an error into
/// `product_errors`.
pub fn product<T, E>(t: &mut Tally, call: impl FnOnce() -> Result<T, E>) -> Option<T> {
    match guard::contained(call) {
        Some(Ok(value)) => Some(value),
        Some(Err(_)) => {
            t.add("product_errors", 1);
            None
        }
        None => {
            t.add("product_panics", 1);
            None
        }
    }
}

/// A policy recorded in the stream.
pub struct Recorded {
    pub n_cal: usize,
    pub record: PolicyRecord,
    /// The grid index the product chose (`None` is "never").
    pub product_j: Option<usize>,
    pub oracle_j: Option<usize>,
}

/// What one arm did over the stream.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArmTally {
    pub eligible: u64,
    pub auto: u64,
    pub harmful_auto: u64,
    pub slice: u64,
}

pub struct StreamOutcome {
    pub adjudicated: Vec<(BranchId, CalibrationSample)>,
    /// For each adjudication, the stream index of its branch.
    pub index: Vec<usize>,
    pub recorded: Vec<Recorded>,
    /// Escalate-all, uncalibrated, Learn-then-Test (the policy in force).
    pub arms: [ArmTally; 3],
    /// Ineligible kinds the product was seen to decide by verification alone.
    pub kinds_seen: [bool; 7],
    /// The share of eligible branches in the slice, per group: all, per namespace, score halves.
    pub slice_groups: Vec<(u64, u64)>,
    pub digest: u64,
}

fn fold(digest: &mut u64, value: u64) {
    *digest ^= value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    *digest = digest.wrapping_mul(0x0000_0100_0000_01B3);
}

fn decision_matches(product: TriageDecision, expected: Decision) -> bool {
    matches!(
        (product, expected),
        (TriageDecision::AutoPropose, Decision::Auto)
            | (TriageDecision::Escalate, Decision::Escalate)
            | (TriageDecision::Discard, Decision::Discard)
    )
}

fn decision_code(decision: TriageDecision) -> u64 {
    match decision {
        TriageDecision::AutoPropose => 1,
        TriageDecision::Escalate => 2,
        TriageDecision::Discard => 3,
    }
}

/// Compare one triage with the specification and with `explains`.
pub fn check_triage(
    t: &mut Tally,
    policy: &TriagePolicy,
    outcome: &TriageOutcome,
    draft_report: &VerificationReport,
    m: u32,
    draw: f64,
) {
    let rate = policy.calibration_rate();
    let threshold = threshold_index(policy.threshold()).unwrap_or(None);
    let expected = oracle::expected(draft_report, m, threshold, draw, rate);
    let agrees = outcome.eligible() == expected.eligible
        && decision_matches(outcome.decision(), expected.decision)
        && outcome.calibration_slice() == expected.slice
        && outcome.auto_propensity() == expected.propensity
        && outcome.score() == model::score_f32(m);
    if !agrees {
        t.add("decision_mismatches", 1);
    }
    if !expected.eligible && outcome.decision() == TriageDecision::AutoPropose {
        t.add("ineligible_auto_proposals", 1);
    }
    match guard::contained(|| policy.explains(outcome)) {
        Some(Ok(())) => {}
        Some(Err(_)) => t.add("unexplained_triages", 1),
        None => t.add("product_panics", 1),
    }
}

fn record_policy(
    t: &mut Tally,
    drafts: &[Draft],
    adjudicated: &[(BranchId, CalibrationSample)],
    index: &[usize],
    n_cal: usize,
    version: usize,
) -> Option<Recorded> {
    let rate = params::slice_rate();
    let rule = ThresholdRule::LearnThenTest {
        alpha: params::alpha(),
        delta: params::delta(),
    };
    let record = product(t, || {
        PolicyRecord::calibrate(format!("v{version}"), rule, rate, &adjudicated[..n_cal])
    })?;
    let raw: Vec<(u32, bool)> = index[..n_cal]
        .iter()
        .map(|&i| (drafts[i].m, drafts[i].label))
        .collect();
    let suffix = Suffix::new(&raw);
    let oracle_j = oracle::ltt_threshold(&suffix, params::alpha(), params::delta());
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
    let scored: Vec<CalibrationSample> = adjudicated[..n_cal]
        .iter()
        .map(|(_, sample)| *sample)
        .collect();
    if let Some(conformal) = product(t, || calibrate_threshold(&scored, params::alpha())) {
        let expected = oracle::crc_threshold(&suffix, params::alpha());
        if threshold_index(conformal) != Ok(expected) {
            t.add("crc_mismatches", 1);
        }
    }
    Some(Recorded {
        n_cal,
        record,
        product_j,
        oracle_j,
    })
}

/// Build a policy through the product; a refusal or a panic is counted (`product_errors`,
/// `product_panics`) and the caller carries on without it.
pub fn make_policy(t: &mut Tally, threshold: AutoThreshold, rate: f64) -> Option<TriagePolicy> {
    product(t, || TriagePolicy::new(threshold, rate))
}

/// The state one pass over the stream threads through its steps.
struct Pass<'a> {
    seed: u64,
    shape: &'a Shape,
    drafts: &'a [Draft],
    escalate_all: TriagePolicy,
    uncalibrated: TriagePolicy,
    in_force: TriagePolicy,
    out: StreamOutcome,
}

fn empty_outcome() -> StreamOutcome {
    StreamOutcome {
        adjudicated: Vec::new(),
        index: Vec::new(),
        recorded: Vec::new(),
        arms: [ArmTally::default(); 3],
        kinds_seen: [false; 7],
        slice_groups: vec![(0, 0); 6],
        digest: 0xCBF2_9CE4_8422_2325,
    }
}

/// Run the pipeline over `drafts`. Every call into the product is contained: a refusal where
/// the specification demands success is a `product_errors`, a panic a `product_panics`, and
/// the run goes on with what it has (a product that yields no data leaves `data_starvation`).
/// An `Err` is a harness fault only.
pub fn run(
    seed: u64,
    shape: &Shape,
    drafts: &[Draft],
    t: &mut Tally,
) -> Result<StreamOutcome, String> {
    let rate = params::slice_rate();
    let unc_threshold =
        AutoThreshold::AtLeast(params::UNCALIBRATED_THRESHOLD_PERMILLE as f32 / 1000.0);
    let (Some(escalate_all), Some(uncalibrated)) = (
        make_policy(t, AutoThreshold::Never, rate),
        make_policy(t, unc_threshold, rate),
    ) else {
        return Ok(empty_outcome());
    };
    let Some(first) = product(t, || PolicyRecord::manual("v0", escalate_all)) else {
        return Ok(empty_outcome());
    };
    let mut pass = Pass {
        seed,
        shape,
        drafts,
        escalate_all,
        uncalibrated,
        in_force: guard::contained(|| first.policy()).unwrap_or(escalate_all),
        out: empty_outcome(),
    };
    for (i, d) in drafts.iter().enumerate() {
        if guard::contained(|| step(&mut pass, t, i, d)).is_none() {
            t.add("product_panics", 1);
        }
    }
    Ok(pass.out)
}

/// One branch through the three arms, the adjudication or its probe, and the policy records.
fn step(pass: &mut Pass<'_>, t: &mut Tally, i: usize, d: &Draft) {
    let draw = calibration_draw(&d.branch, pass.seed);
    let prob = d.probability();
    let spec_eligible = oracle::eligible_by_spec(&d.report);
    let policies = [pass.escalate_all, pass.uncalibrated, pass.in_force];
    let mut forced_outcome = None;
    for (arm, policy) in policies.iter().enumerate() {
        let Some(outcome) = product(t, || policy.triage(&d.report, prob, draw)) else {
            continue;
        };
        check_triage(t, policy, &outcome, &d.report, d.m, draw);
        let tally = &mut pass.out.arms[arm];
        if spec_eligible {
            tally.eligible += 1;
        }
        if outcome.decision() == TriageDecision::AutoPropose {
            tally.auto += 1;
            tally.harmful_auto += u64::from(d.harm);
        }
        tally.slice += u64::from(outcome.calibration_slice());
        fold(&mut pass.out.digest, decision_code(outcome.decision()));
        fold(&mut pass.out.digest, u64::from(outcome.calibration_slice()));
        if arm == 2 {
            forced_outcome = Some(outcome);
        }
    }
    let Some(outcome) = forced_outcome else {
        return;
    };
    if !outcome.eligible() {
        pass.out.kinds_seen[d.kind.index()] = true;
    }
    if spec_eligible {
        let groups = [0, 1 + d.ns, 4 + usize::from(d.m >= 500_000)];
        for g in groups {
            pass.out.slice_groups[g].0 += 1;
            pass.out.slice_groups[g].1 += u64::from(outcome.calibration_slice());
        }
    }
    if outcome.calibration_slice() {
        let Some(sample) = product(t, || outcome.adjudicate(d.label)) else {
            return;
        };
        if sample.score() != model::score_f32(d.m) || sample.harmful() != d.label {
            t.add("adjudication_mismatches", 1);
        }
        pass.out.adjudicated.push((d.branch.clone(), sample));
        pass.out.index.push(i);
        let n = pass.out.adjudicated.len();
        let k = pass.out.recorded.len();
        if k < pass.shape.policy_sizes.len() && n == pass.shape.policy_sizes[k] {
            if let Some(recorded) = record_policy(
                t,
                pass.drafts,
                &pass.out.adjudicated,
                &pass.out.index,
                n,
                k + 1,
            ) {
                if let Some(policy) = guard::contained(|| recorded.record.policy()) {
                    pass.in_force = policy;
                } else {
                    t.add("product_panics", 1);
                }
                fold(
                    &mut pass.out.digest,
                    recorded.product_j.map_or(9999, |j| j as u64),
                );
                pass.out.recorded.push(recorded);
            }
        }
    } else {
        t.add("nonslice_probes", 1);
        match guard::contained(|| outcome.adjudicate(d.label)) {
            Some(Ok(_)) => t.add("nonslice_adjudicated", 1),
            Some(Err(_)) => {}
            None => t.add("product_panics", 1),
        }
    }
}

/// The slice-rate z tests over the groups of the run. Returns `(tests, violations, max |z|)`
/// with the z score in thousandths.
pub fn slice_rate_tests(groups: &[(u64, u64)]) -> (i64, i64, i64) {
    let rate = params::slice_rate();
    let mut tests = 0;
    let mut violations = 0;
    let mut worst = 0_i64;
    for (n, slice) in groups {
        if *n == 0 {
            // A group nothing was observed in cannot show the slice is uniform.
            violations += 1;
            continue;
        }
        tests += 1;
        let n = *n as f64;
        let z = (*slice as f64 - n * rate) / (n * rate * (1.0 - rate)).sqrt();
        let milli = (z.abs() * 1000.0).round() as i64;
        worst = worst.max(milli);
        if milli > params::SLICE_Z_LIMIT_MILLI {
            violations += 1;
        }
    }
    (tests, violations, worst)
}
