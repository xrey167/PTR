//! The independent oracles: eligibility written from the specification, the decision a policy
//! must reach, the Learn-then-Test fixed-sequence rule and the conformal rule re-implemented
//! over integer scores, and the simulated adjudicator.
//!
//! Nothing here reads a product value to compute an expectation: the oracles take the
//! harness's own micro-scores, labels and report fields, and the product is compared with what
//! they derive.

use ptr_types::VerificationLevel;
use ptr_verifier::{VerificationReport, VerificationStatus};

use super::model::{admits, CELLS};
use super::stats;

/// Whether the specification leaves a branch to the policy: the report passed, at a level whose
/// check is full (`FullSemantic` or `Deterministic`), with no hard finding. Written from the
/// manifest's text, not from `TriagePolicy::triage`.
pub fn eligible_by_spec(report: &VerificationReport) -> bool {
    let passed = matches!(report.status, VerificationStatus::Pass);
    let deep = matches!(
        report.level,
        VerificationLevel::FullSemantic | VerificationLevel::Deterministic
    );
    let hard = report.findings.iter().any(|finding| finding.hard);
    passed && deep && !hard
}

/// A decision, as the oracle names it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Decision {
    Auto,
    Escalate,
    Discard,
}

/// What the specification says a triage must be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expected {
    pub eligible: bool,
    pub decision: Decision,
    pub slice: bool,
    pub propensity: f64,
}

/// The triage the specification demands. `threshold` is the grid index the policy holds, or
/// `None` for "never"; `rate` is its calibration rate; `draw` the branch's calibration draw.
pub fn expected(
    report: &VerificationReport,
    m: u32,
    threshold: Option<usize>,
    draw: f64,
    rate: f64,
) -> Expected {
    if !eligible_by_spec(report) {
        let decision = if matches!(report.status, VerificationStatus::Fail) {
            Decision::Discard
        } else {
            Decision::Escalate
        };
        return Expected {
            eligible: false,
            decision,
            slice: false,
            propensity: 0.0,
        };
    }
    let admitted = threshold.is_some_and(|j| admits(j, m));
    let slice = draw < rate;
    Expected {
        eligible: true,
        decision: if admitted && !slice {
            Decision::Auto
        } else {
            Decision::Escalate
        },
        slice,
        propensity: if admitted { 1.0 - rate } else { 0.0 },
    }
}

/// Counts of samples and of harmful labels per cell, and the suffix sums the rules read.
pub struct Suffix {
    /// `n[j]`: samples with cell `>= j`; `n[CELLS]` is zero.
    pub n: Vec<usize>,
    /// `k[j]`: harmful labels among them.
    pub k: Vec<usize>,
}

impl Suffix {
    pub fn new(samples: &[(u32, bool)]) -> Self {
        let mut n = vec![0_usize; CELLS + 1];
        let mut k = vec![0_usize; CELLS + 1];
        for (m, label) in samples {
            let cell = (*m / 1000) as usize;
            n[cell] += 1;
            k[cell] += usize::from(*label);
        }
        for j in (0..CELLS).rev() {
            n[j] += n[j + 1];
            k[j] += k[j + 1];
        }
        Self { n, k }
    }
}

/// The Learn-then-Test fixed-sequence rule: thresholds are tested from the highest down, the
/// sequence starts at the first threshold where even no harm could pass, and stops at the first
/// failure; the lowest threshold that passed is chosen. `None` is "never".
pub fn ltt_threshold(suffix: &Suffix, alpha: f64, delta: f64) -> Option<usize> {
    let mut chosen = None;
    let mut started = false;
    for j in (0..=CELLS).rev() {
        let n = suffix.n[j];
        if !started {
            if !stats::ltt_passes(0, n, alpha, delta) {
                continue;
            }
            started = true;
        }
        if stats::ltt_passes(suffix.k[j], n, alpha, delta) {
            chosen = Some(j);
        } else {
            break;
        }
    }
    chosen
}

/// Conformal risk control: the smallest grid threshold whose
/// `n/(n+1) * (admitted harmful / n) + 1/(n+1)` is at most alpha. The formula is the
/// documented one, evaluated in the same order so that its ties round the same way.
pub fn crc_threshold(suffix: &Suffix, alpha: f64) -> Option<usize> {
    let n = suffix.n[0] as f64;
    if suffix.n[0] == 0 {
        return None;
    }
    (0..=CELLS).find(|&j| {
        let bound = (n / (n + 1.0)) * (suffix.k[j] as f64 / n) + 1.0 / (n + 1.0);
        bound <= alpha
    })
}

/// The control rule that "estimates without a bound": the same sequence as Learn-then-Test
/// with the point estimate `k/n <= alpha` (from fifty-nine admitted up) in place of the bound.
pub fn naive_threshold(suffix: &Suffix, alpha: f64, delta: f64) -> Option<usize> {
    let mut chosen = None;
    let mut started = false;
    for j in (0..=CELLS).rev() {
        let n = suffix.n[j];
        if !started {
            if !stats::ltt_passes(0, n, alpha, delta) {
                continue;
            }
            started = true;
        }
        if (suffix.k[j] as f64) <= alpha * n as f64 {
            chosen = Some(j);
        } else {
            break;
        }
    }
    chosen
}

/// A SplitMix-style 64-bit mixer, with the byte-wise FNV-1a of a name folded in.
fn mix(seed: u64, text: &str) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut z = hash ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What the simulated adjudicator is shown: the delta (changed keys), the verifier's finding
/// codes and the ground truth the simulation holds. There is no intent field: the intent is
/// free text the agent wrote, and adjudicators do not see it (`adjudicators_see_intent =
/// false`).
// The simulated adjudicator reads the identity and the truth only; the keys and codes are what
// a person would be shown, kept so that the view has the shape the protocol states.
#[allow(dead_code)]
pub struct AdjudicationView<'a> {
    pub id: &'a str,
    pub changed_keys: &'a [String],
    pub verifier_codes: &'a [String],
    pub truth_harmful: bool,
}

/// The simulated adjudicator: the truth, flipped for a hash-keyed `epsilon` share of branches.
/// This is a simulation with a known error rate, not a person.
pub fn adjudicate(seed: u64, view: &AdjudicationView<'_>, epsilon_permille: u32) -> bool {
    let flipped = mix(seed, view.id) % 1000 < u64::from(epsilon_permille);
    view.truth_harmful != flipped
}

/// The first `:`-separated segment of a changed key: its namespace for the descriptive breakdown.
pub fn namespace_of(key: &str) -> &str {
    key.split(':').next().unwrap_or(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::f003::model::Kind;

    fn samples(spec: &[(u32, bool, usize)]) -> Vec<(u32, bool)> {
        spec.iter()
            .flat_map(|(m, label, count)| std::iter::repeat_n((*m, *label), *count))
            .collect()
    }

    #[test]
    fn eligibility_follows_the_specification() {
        for (kind, eligible) in [
            (Kind::Eligible, true),
            (Kind::Fail, false),
            (Kind::Disputed, false),
            (Kind::Unknown, false),
            (Kind::LatentAgreement, false),
            (Kind::SampleVerified, false),
            (Kind::HardFinding, false),
        ] {
            assert_eq!(eligible_by_spec(&kind.report(0)), eligible, "{kind:?}");
            assert_eq!(eligible_by_spec(&kind.report(1)), eligible, "{kind:?}");
        }
    }

    #[test]
    fn expected_triage_cases() {
        let eligible = Kind::Eligible.report(0);
        let rate = 0.25;
        let admitted = expected(&eligible, 900_000, Some(500), 0.5, rate);
        assert_eq!(admitted.decision, Decision::Auto);
        assert_eq!(admitted.propensity, 0.75);
        let sliced = expected(&eligible, 900_000, Some(500), 0.1, rate);
        assert!(sliced.slice && sliced.decision == Decision::Escalate);
        let below = expected(&eligible, 400_000, Some(500), 0.9, rate);
        assert_eq!(
            (below.decision, below.propensity),
            (Decision::Escalate, 0.0)
        );
        let never = expected(&eligible, 999_999, None, 0.9, rate);
        assert_eq!(never.decision, Decision::Escalate);
        let failed = expected(&Kind::Fail.report(0), 999_999, Some(0), 0.0, rate);
        assert_eq!(
            (failed.decision, failed.slice, failed.eligible),
            (Decision::Discard, false, false)
        );
        let forged = expected(&Kind::HardFinding.report(0), 999_999, Some(0), 0.9, rate);
        assert_eq!(forged.decision, Decision::Escalate);
    }

    #[test]
    fn the_sequence_starts_at_fifty_nine_and_stops_at_the_first_failure() {
        // 58 clean samples at the top cannot certify; 59 can.
        let alpha = 0.05;
        let few = Suffix::new(&samples(&[(990_000, false, 58)]));
        assert_eq!(ltt_threshold(&few, alpha, alpha), None);
        let enough = Suffix::new(&samples(&[(990_000, false, 59)]));
        assert_eq!(ltt_threshold(&enough, alpha, alpha), Some(0));
        // A harmful block at 600 stops the descent: lower thresholds are never tested.
        let blocked = Suffix::new(&samples(&[
            (990_000, false, 100),
            (600_000, true, 30),
            (100_000, false, 2000),
        ]));
        assert_eq!(ltt_threshold(&blocked, alpha, alpha), Some(601));
        // The naive rule over the same data also stops there.
        assert_eq!(naive_threshold(&blocked, alpha, alpha), Some(601));
    }

    #[test]
    fn the_naive_rule_is_more_permissive_than_the_bound() {
        // 3 harmful among 100 (3 %): a point estimate passes alpha = 5 %, the bound does not.
        let data = Suffix::new(&samples(&[(950_000, false, 97), (950_000, true, 3)]));
        assert_eq!(naive_threshold(&data, 0.05, 0.05), Some(0));
        assert_eq!(ltt_threshold(&data, 0.05, 0.05), None);
    }

    #[test]
    fn the_conformal_rule_counts_one_pseudo_sample() {
        // n = 39 clean: 1/40 <= 0.05 at every threshold, so the lowest grid point.
        let clean = Suffix::new(&samples(&[(500_000, false, 39)]));
        assert_eq!(crc_threshold(&clean, 0.05), Some(0));
        // n = 18 clean: 1/19 > 0.05, nothing is auto-proposed.
        let small = Suffix::new(&samples(&[(500_000, false, 18)]));
        assert_eq!(crc_threshold(&small, 0.05), None);
        assert_eq!(crc_threshold(&Suffix::new(&[]), 0.05), None);
    }

    #[test]
    fn the_adjudicator_is_deterministic_and_flips_about_epsilon() {
        let keys = vec!["note:a".to_string()];
        let codes: Vec<String> = Vec::new();
        let mut flips = 0;
        for i in 0..20_000 {
            let id = format!("b{i}");
            let view = AdjudicationView {
                id: &id,
                changed_keys: &keys,
                verifier_codes: &codes,
                truth_harmful: false,
            };
            let label = adjudicate(17, &view, 20);
            assert_eq!(label, adjudicate(17, &view, 20));
            flips += usize::from(label);
        }
        assert!((300..500).contains(&flips), "{flips}");
        let view = AdjudicationView {
            id: "x",
            changed_keys: &keys,
            verifier_codes: &codes,
            truth_harmful: true,
        };
        assert!(adjudicate(17, &view, 0));
    }

    #[test]
    fn a_namespace_is_the_first_segment() {
        assert_eq!(namespace_of("note:abc:d"), "note");
        assert_eq!(namespace_of("plain"), "plain");
    }
}
