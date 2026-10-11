//! The simulated world: branches with a verification report, a score from a deliberately
//! overconfident scorer, a true harm and the namespaces of the keys they change.
//!
//! A score is an integer `m` in `0..1_000_000` (micro-units; the product sees the `f32`
//! `m / 1e6`), its *cell* is `m / 1000`, and a grid threshold `j / 1000` admits exactly the
//! scores with `m >= 1000 * j` (checked against the product's grid at start, `grid_self_check`).
//! A namespace draws its score as the maximum of `maxes` independent uniform integers, so the
//! mass of cell `c` is exactly `((c + 1) / 1000)^maxes - (c / 1000)^maxes`, and its true harm
//! probability is a table indexed by the cell (linear between anchors, in integer ppm). The
//! scorer is overconfident: the harm among scores the uncalibrated threshold admits is several
//! times what a score near one promises. Nothing here calls the product; the oracle reads the
//! tables, never a product value.

use ptr_types::VerificationLevel;
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

use super::params;
use crate::experiments::rng::Rng;

pub const CELLS: usize = 1000;
pub const MICRO: u32 = 1_000_000;

/// A key namespace with its score and harm model.
pub struct Namespace {
    pub name: &'static str,
    pub weight_permille: u32,
    /// The score is the maximum of this many uniform draws.
    pub maxes: u32,
    /// `(cell, harm ppm)` anchors, ascending in the cell; the table is linear between them.
    pub anchors: &'static [(i64, i64)],
}

pub const NAMESPACES: [Namespace; 3] = [
    Namespace {
        name: "note",
        weight_permille: 500,
        maxes: 4,
        anchors: &[
            (0, 650_000),
            (500, 300_000),
            (700, 110_000),
            (850, 12_000),
            (1000, 300),
        ],
    },
    Namespace {
        name: "task",
        weight_permille: 300,
        maxes: 3,
        anchors: &[
            (0, 750_000),
            (500, 350_000),
            (700, 130_000),
            (850, 18_000),
            (1000, 500),
        ],
    },
    Namespace {
        name: "cfg",
        weight_permille: 200,
        maxes: 3,
        anchors: &[
            (0, 850_000),
            (500, 400_000),
            (700, 160_000),
            (850, 28_000),
            (1000, 1_000),
        ],
    },
];

/// The true harm probability, in ppm, of a branch of namespace `ns` whose score is in `cell`.
/// Integer arithmetic only (division truncates toward zero).
pub fn harm_ppm(ns: usize, cell: usize) -> i64 {
    let anchors = NAMESPACES[ns].anchors;
    let cell = cell as i64;
    for pair in anchors.windows(2) {
        let ((c0, v0), (c1, v1)) = (pair[0], pair[1]);
        if (c0..=c1).contains(&cell) {
            return v0 + (v1 - v0) * (cell - c0) / (c1 - c0);
        }
    }
    anchors[anchors.len() - 1].1
}

/// The score `f32` the product sees for micro-score `m`.
pub fn score_f32(m: u32) -> f32 {
    m as f32 / MICRO as f32
}

/// Whether the grid threshold `j` admits micro-score `m`, by the integer rule.
pub fn admits(j: usize, m: u32) -> bool {
    u64::from(m) >= 1000 * j as u64
}

/// The product's `f32` comparison against its own grid must equal the integer rule at every
/// grid edge. A false answer means the harness's integer oracle would disagree with the product
/// for a reason that is not a defect, so the run cannot be judged (exit 2).
pub fn grid_self_check() -> Result<(), String> {
    let grid: Vec<f32> = ptr_branch::threshold_grid().collect();
    if grid.len() != 1001 {
        return Err(format!(
            "the product's grid has {} thresholds, not 1001",
            grid.len()
        ));
    }
    for (j, threshold) in grid.iter().enumerate() {
        for m in [
            (1000 * j as u32).saturating_sub(1),
            1000 * j as u32,
            (1000 * j as u32 + 1).min(MICRO - 1),
        ] {
            let by_f32 = score_f32(m) >= *threshold;
            if by_f32 != admits(j, m) {
                return Err(format!(
                    "the f32 grid and the integer rule disagree at j = {j}, m = {m}"
                ));
            }
        }
    }
    Ok(())
}

/// The kind of verification outcome a branch carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Pass at Deterministic or FullSemantic, no hard finding: the policy decides.
    Eligible,
    /// Fail at Deterministic.
    Fail,
    /// Disputed at FullSemantic.
    Disputed,
    /// Unknown at Deterministic.
    Unknown,
    /// Pass at LatentAgreement.
    LatentAgreement,
    /// Pass at SampleVerified.
    SampleVerified,
    /// Pass at Deterministic with a hard finding: a forged-looking pass (INVARIANT 11).
    HardFinding,
}

impl Kind {
    pub const ALL: [Kind; 7] = [
        Kind::Eligible,
        Kind::Fail,
        Kind::Disputed,
        Kind::Unknown,
        Kind::LatentAgreement,
        Kind::SampleVerified,
        Kind::HardFinding,
    ];

    pub fn index(self) -> usize {
        Kind::ALL.iter().position(|kind| *kind == self).unwrap_or(0)
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Eligible => "eligible",
            Kind::Fail => "fail",
            Kind::Disputed => "disputed",
            Kind::Unknown => "unknown",
            Kind::LatentAgreement => "latent_agreement",
            Kind::SampleVerified => "sample_verified",
            Kind::HardFinding => "hard_finding",
        }
    }

    /// The verification report of a branch of this kind. `variant` picks between the levels
    /// and the soft findings an eligible report may carry.
    pub fn report(self, variant: u64) -> VerificationReport {
        let soft = |code: &str| Finding {
            code: code.to_string(),
            message: "advisory".to_string(),
            hard: false,
        };
        let hard = |code: &str| Finding {
            code: code.to_string(),
            message: "contradiction".to_string(),
            hard: true,
        };
        let (status, level, findings) = match self {
            Kind::Eligible => (
                VerificationStatus::Pass,
                if variant.is_multiple_of(2) {
                    VerificationLevel::Deterministic
                } else {
                    VerificationLevel::FullSemantic
                },
                if variant.is_multiple_of(7) {
                    vec![soft("style")]
                } else {
                    Vec::new()
                },
            ),
            Kind::Fail => (
                VerificationStatus::Fail,
                VerificationLevel::Deterministic,
                vec![hard("constraint")],
            ),
            Kind::Disputed => (
                VerificationStatus::Disputed,
                VerificationLevel::FullSemantic,
                Vec::new(),
            ),
            Kind::Unknown => (
                VerificationStatus::Unknown,
                VerificationLevel::Deterministic,
                Vec::new(),
            ),
            Kind::LatentAgreement => (
                VerificationStatus::Pass,
                VerificationLevel::LatentAgreement,
                Vec::new(),
            ),
            Kind::SampleVerified => (
                VerificationStatus::Pass,
                VerificationLevel::SampleVerified,
                Vec::new(),
            ),
            Kind::HardFinding => (
                VerificationStatus::Pass,
                VerificationLevel::Deterministic,
                vec![hard("unsafe")],
            ),
        };
        VerificationReport {
            status,
            level,
            score: ptr_types::Probability::new(0.99).unwrap_or_default(),
            findings,
        }
    }
}

/// A branch kind drawn by `KIND_WEIGHTS_PERMILLE`.
pub fn draw_kind(rng: &mut Rng) -> Kind {
    let mut roll = rng.below(1000) as u32;
    for (kind, weight) in Kind::ALL.iter().zip(params::KIND_WEIGHTS_PERMILLE) {
        if roll < weight {
            return *kind;
        }
        roll -= weight;
    }
    Kind::Eligible
}

/// A namespace drawn by the weights.
pub fn draw_namespace(rng: &mut Rng) -> usize {
    let mut roll = rng.below(1000) as u32;
    for (index, ns) in NAMESPACES.iter().enumerate() {
        if roll < ns.weight_permille {
            return index;
        }
        roll -= ns.weight_permille;
    }
    0
}

/// The micro-score of an eligible branch of namespace `ns`. A scorer that reports round
/// numbers puts `params::TIED_PERMILLE` of its scores exactly on a grid point (the score is rounded
/// down to a multiple of 1000, which keeps its cell and so the model's tables), so that a
/// threshold sees many scores exactly equal to it and `>=` against `>` matters.
pub fn draw_score(ns: usize, rng: &mut Rng) -> u32 {
    let m = (0..NAMESPACES[ns].maxes)
        .map(|_| rng.below(u64::from(MICRO)) as u32)
        .max()
        .unwrap_or(0);
    if rng.below(1000) < u64::from(params::TIED_PERMILLE) {
        m - m % 1000
    } else {
        m
    }
}

/// The micro-score of a branch verification will decide: adversarially high, so that every
/// threshold the policies hold would admit it were it eligible.
pub fn draw_adversarial_score(rng: &mut Rng) -> u32 {
    900_000 + rng.below(100_000) as u32
}

/// Whether a branch of namespace `ns` and micro-score `m` is truly harmful, drawing from
/// `harm_ppm` scaled by `factor` (one for the model, more for the drift control).
pub fn draw_harm(ns: usize, m: u32, factor: i64, rng: &mut Rng) -> bool {
    let ppm = (harm_ppm(ns, (m / 1000) as usize) * factor).min(i64::from(MICRO));
    (rng.below(u64::from(MICRO)) as i64) < ppm
}

/// Exact quantities of the model, summed over cells.
pub struct Truth {
    /// Mass of the cell among eligible branches.
    mass: Vec<f64>,
    /// Mass of true harm in the cell.
    harm: Vec<f64>,
    /// `Sum over ns of weight * mass(cell) * (reward of auto-proposing)`; see `reward_mass`.
    reward: Vec<f64>,
}

fn power(base: f64, exponent: u32) -> f64 {
    (0..exponent).fold(1.0, |acc, _| acc * base)
}

impl Truth {
    pub fn new() -> Self {
        let mut mass = vec![0.0; CELLS];
        let mut harm = vec![0.0; CELLS];
        let mut reward = vec![0.0; CELLS];
        for (index, ns) in NAMESPACES.iter().enumerate() {
            let weight = f64::from(ns.weight_permille) / 1000.0;
            for cell in 0..CELLS {
                let lower = power(cell as f64 / CELLS as f64, ns.maxes);
                let upper = power((cell + 1) as f64 / CELLS as f64, ns.maxes);
                let m = weight * (upper - lower);
                let h = harm_ppm(index, cell) as f64 / f64::from(MICRO);
                mass[cell] += m;
                harm[cell] += m * h;
                reward[cell] += m * (params::REWARD_SAFE * (1.0 - h) + params::REWARD_HARM * h);
            }
        }
        Self { mass, harm, reward }
    }

    /// The share of eligible branches whose score the grid threshold `j` admits.
    pub fn mass_above(&self, j: usize) -> f64 {
        self.mass[j.min(CELLS)..].iter().sum()
    }

    /// The true harm rate among eligible branches the threshold `j` admits.
    pub fn harm_above(&self, j: usize) -> f64 {
        let mass = self.mass_above(j);
        if mass <= 0.0 {
            return 0.0;
        }
        self.harm[j.min(CELLS)..].iter().sum::<f64>() / mass
    }

    /// The harm rate the adjudicator reports there: truth flipped with the error rate.
    pub fn adjudicated_harm_above(&self, j: usize) -> f64 {
        self.adjudicated_harm_above_at(j, params::EPSILON_PERMILLE)
    }

    /// The same at another adjudicator error rate (the descriptive arm of the report).
    pub fn adjudicated_harm_above_at(&self, j: usize, epsilon_permille: u32) -> f64 {
        let eps = f64::from(epsilon_permille) / 1000.0;
        eps + (1.0 - 2.0 * eps) * self.harm_above(j)
    }

    /// The marginal harm probability given the cell, over namespaces.
    pub fn marginal_harm(&self, cell: usize) -> f64 {
        if self.mass[cell] > 0.0 {
            self.harm[cell] / self.mass[cell]
        } else {
            0.0
        }
    }

    /// The value of a policy that auto-proposes with probability `auto` every eligible branch
    /// the threshold `j` admits, per eligible branch: `auto * sum_{cell >= j} mass * reward`.
    pub fn eligible_value(&self, j: usize, auto: f64) -> f64 {
        auto * self.reward[j.min(CELLS)..].iter().sum::<f64>()
    }
}

impl Default for Truth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_edges_agree_with_the_integer_rule() {
        grid_self_check().expect("the f32 grid is the integer grid");
    }

    #[test]
    fn the_weights_are_whole() {
        assert_eq!(params::KIND_WEIGHTS_PERMILLE.iter().sum::<u32>(), 1000);
        assert_eq!(params::KIND_WEIGHTS_PERMILLE[0], params::ELIGIBLE_PERMILLE);
        assert_eq!(
            NAMESPACES.iter().map(|ns| ns.weight_permille).sum::<u32>(),
            1000
        );
    }

    #[test]
    fn the_mass_sums_to_one_and_the_scorer_is_overconfident() {
        let truth = Truth::new();
        assert!((truth.mass_above(0) - 1.0).abs() < 1e-12);
        // The uncalibrated threshold admits a harm rate well above alpha...
        let uncalibrated =
            truth.adjudicated_harm_above(params::UNCALIBRATED_THRESHOLD_PERMILLE as usize);
        assert!(uncalibrated > 0.07, "{uncalibrated}");
        // ...while a clean region exists and is not empty.
        let clean = truth.adjudicated_harm_above(850);
        assert!(clean < 0.02 && truth.mass_above(850) > 0.3, "{clean}");
        // Harm falls with the score.
        let mut previous = 2.0;
        for j in (0..=999).step_by(50) {
            let h = truth.harm_above(j);
            assert!(h <= previous + 1e-12);
            previous = h;
        }
    }

    #[test]
    fn harm_tables_interpolate_between_anchors() {
        assert_eq!(harm_ppm(0, 0), 650_000);
        assert_eq!(harm_ppm(0, 500), 300_000);
        assert_eq!(harm_ppm(0, 600), 300_000 + (110_000 - 300_000) * 100 / 200);
        assert!(harm_ppm(2, 999) > 0);
    }

    #[test]
    fn every_kind_has_a_report_and_the_index_matches() {
        for (i, kind) in Kind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i);
            let _ = kind.report(i as u64);
        }
    }
}
