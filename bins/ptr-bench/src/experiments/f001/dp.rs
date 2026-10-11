//! The exact analytic expectation of F001: for a profile, an arm and a budget, the probability
//! of success and the mean and variance of the verifier calls and of the slot draws, by dynamic
//! programming over the 2^7 defect states. It reads only the stub's parameters (the slot
//! probabilities, the repair probability and the verifier noise), never a run.
//!
//! The model of one task, which the harness' stubs implement:
//!
//! - a fresh draw has slot `s` defective with probability `1 - p_s`, independently, and costs 7
//!   slot draws;
//! - each verifier call is in a noise mode drawn independently of the candidate. A candidate
//!   with defects is always rejected; a defect-free one is accepted unless the mode is unknown,
//!   disputed, shallow or lying (probability `1 - a`, `a` = clean + forged);
//! - after a rejected candidate with defects the repair arm repairs it (each defective slot is
//!   fixed with probability `q`, at a cost of one slot draw per defective slot) unless the mode
//!   was the forged one, whose self-contradicting `Pass` is not repairable and is rewritten;
//!   after a rejected defect-free candidate it rewrites; the rewrite arm always rewrites;
//! - the loop stops at the first accepted candidate or when the call budget is spent.

use super::params::{MODE_FORGED_PERMILLE, SLOTS};
use super::stub::Noise;

const STATES: usize = 1 << SLOTS;

/// One branch after a rejected candidate: its probability, the slot draws it spends and the
/// distribution of the next candidate's defect state.
type Branch = (f64, f64, Vec<(usize, f64)>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArmKind {
    Rewrite,
    Repair,
}

/// Mean and variance of a per-task quantity.
#[derive(Clone, Copy, Debug)]
pub struct Moments {
    pub mean: f64,
    pub variance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// Probability that a task ends accepted within the budget.
    pub success: f64,
    pub calls: Moments,
    pub draws: Moments,
}

pub struct Model {
    /// The probability of a defect-free first draw per defect state.
    generate: [f64; STATES],
    q: f64,
    accept: f64,
    forged: f64,
}

impl Model {
    pub fn new(p_permille: &[u16; SLOTS], q_permille: u64, noise: &Noise) -> Self {
        let mut generate = [0.0; STATES];
        for (mask, probability) in generate.iter_mut().enumerate() {
            *probability = (0..SLOTS)
                .map(|slot| {
                    let right = f64::from(p_permille[slot]) / 1000.0;
                    if mask & (1 << slot) != 0 {
                        1.0 - right
                    } else {
                        right
                    }
                })
                .product();
        }
        let accepting = 1000 - noise.rejecting_permille();
        Self {
            generate,
            q: q_permille as f64 / 1000.0,
            accept: accepting as f64 / 1000.0,
            forged: MODE_FORGED_PERMILLE as f64 / 1000.0,
        }
    }

    /// The distribution of the defect state after repairing the slots of `mask`.
    fn repaired(&self, mask: usize) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        let mut sub = mask;
        loop {
            let kept = sub.count_ones() as i32;
            let fixed = mask.count_ones() as i32 - kept;
            out.push((sub, self.q.powi(fixed) * (1.0 - self.q).powi(kept)));
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & mask;
        }
        out
    }

    /// The branches after a rejected candidate with defect state `mask`: the probability of the
    /// branch, the slot draws it spends and the distribution of the next candidate's state.
    fn branches(&self, mask: usize, arm: ArmKind) -> Vec<Branch> {
        let fresh = || -> Vec<(usize, f64)> { self.generate.iter().copied().enumerate().collect() };
        let cost = SLOTS as f64;
        if mask == 0 {
            return vec![(1.0 - self.accept, cost, fresh())];
        }
        match arm {
            ArmKind::Rewrite => vec![(1.0, cost, fresh())],
            ArmKind::Repair => vec![
                (self.forged, cost, fresh()),
                (
                    1.0 - self.forged,
                    f64::from(mask.count_ones()),
                    self.repaired(mask),
                ),
            ],
        }
    }

    /// The cell for `arm` under a budget of `budget` verifier calls.
    pub fn cell(&self, arm: ArmKind, budget: u32) -> Cell {
        let mut success = [0.0; STATES];
        // Calls: mean and second moment of the calls from a candidate about to be verified.
        let mut calls1 = [0.0; STATES];
        let mut calls2 = [0.0; STATES];
        let mut draws1 = [0.0; STATES];
        let mut draws2 = [0.0; STATES];
        for k in 1..=budget {
            let mut next_success = [0.0; STATES];
            let mut next_calls1 = [0.0; STATES];
            let mut next_calls2 = [0.0; STATES];
            let mut next_draws1 = [0.0; STATES];
            let mut next_draws2 = [0.0; STATES];
            for mask in 0..STATES {
                let mut s = if mask == 0 { self.accept } else { 0.0 };
                // y1, y2: the first and second moment of the calls after this one.
                let (mut y1, mut y2) = (0.0, 0.0);
                let (mut d1, mut d2) = (0.0, 0.0);
                if k >= 2 {
                    for (weight, cost, next) in self.branches(mask, arm) {
                        for (state, probability) in next {
                            let w = weight * probability;
                            s += w * success[state];
                            y1 += w * calls1[state];
                            y2 += w * calls2[state];
                            d1 += w * (cost + draws1[state]);
                            d2 += w * (cost * cost + 2.0 * cost * draws1[state] + draws2[state]);
                        }
                    }
                }
                next_success[mask] = s;
                next_calls1[mask] = 1.0 + y1;
                next_calls2[mask] = 1.0 + 2.0 * y1 + y2;
                next_draws1[mask] = d1;
                next_draws2[mask] = d2;
            }
            success = next_success;
            calls1 = next_calls1;
            calls2 = next_calls2;
            draws1 = next_draws1;
            draws2 = next_draws2;
        }
        let first = SLOTS as f64;
        let mut total_success = 0.0;
        let (mut c1, mut c2, mut e1, mut e2) = (0.0, 0.0, 0.0, 0.0);
        for (mask, probability) in self.generate.iter().enumerate() {
            total_success += probability * success[mask];
            c1 += probability * calls1[mask];
            c2 += probability * calls2[mask];
            e1 += probability * draws1[mask];
            e2 += probability * draws2[mask];
        }
        // The first draw costs `first` slot draws on top of whatever follows.
        let draws_mean = first + e1;
        let draws_second = first * first + 2.0 * first * e1 + e2;
        Cell {
            success: total_success,
            calls: Moments {
                mean: c1,
                variance: (c2 - c1 * c1).max(0.0),
            },
            draws: Moments {
                mean: draws_mean,
                variance: (draws_second - draws_mean * draws_mean).max(0.0),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(p: u16, q: u64) -> Model {
        Model::new(&[p; SLOTS], q, &Noise::pinned())
    }

    #[test]
    fn the_first_call_succeeds_only_on_a_defect_free_draw_the_verifier_accepts() {
        let m = model(900, 800);
        let clean = 0.9f64.powi(7) * 0.87;
        for arm in [ArmKind::Rewrite, ArmKind::Repair] {
            let cell = m.cell(arm, 1);
            assert!((cell.success - clean).abs() < 1e-12);
            assert!((cell.calls.mean - 1.0).abs() < 1e-12);
            assert!(cell.calls.variance < 1e-12);
            assert!((cell.draws.mean - 7.0).abs() < 1e-12);
        }
    }

    #[test]
    fn rewrite_success_follows_the_geometric_law() {
        let m = model(700, 800);
        let p = 0.7f64.powi(7) * 0.87;
        for budget in [1u32, 2, 4, 8, 16] {
            let cell = m.cell(ArmKind::Rewrite, budget);
            let expected = 1.0 - (1.0 - p).powi(budget as i32);
            assert!((cell.success - expected).abs() < 1e-9, "{budget}");
            // Every rewrite costs 7 draws; the mean calls are the truncated geometric mean.
            let calls = (1.0 - (1.0 - p).powi(budget as i32)) / p;
            assert!((cell.calls.mean - calls).abs() < 1e-9, "{budget}");
            assert!((cell.draws.mean - 7.0 * calls).abs() < 1e-9, "{budget}");
        }
    }

    #[test]
    fn repair_with_a_perfect_repairer_needs_at_most_two_clean_calls() {
        // q = 1: a rejected candidate with defects is repaired to a defect-free one (unless the
        // forged mode sends it to a rewrite), so success within 2 calls is high.
        let m = model(600, 1000);
        let two = m.cell(ArmKind::Repair, 2).success;
        let rewrite_two = m.cell(ArmKind::Rewrite, 2).success;
        assert!(two > rewrite_two + 0.3);
    }

    #[test]
    fn success_grows_with_the_budget_and_repair_never_loses_to_rewrite() {
        let m = model(650, 800);
        let mut last = [0.0, 0.0];
        for budget in 1..=16u32 {
            for (i, arm) in [ArmKind::Rewrite, ArmKind::Repair].into_iter().enumerate() {
                let now = m.cell(arm, budget).success;
                assert!(now >= last[i] - 1e-12);
                last[i] = now;
            }
            assert!(last[1] >= last[0] - 1e-12, "{budget}");
        }
    }

    #[test]
    fn the_moments_agree_with_a_brute_force_enumeration_on_a_two_call_budget() {
        // One slot's worth of state is enough to enumerate by hand: with budget 2 and the
        // rewrite arm, calls is 1 w.p. a0 and 2 otherwise, a0 = P(first candidate accepted).
        let m = model(800, 800);
        let a0 = 0.8f64.powi(7) * 0.87;
        let cell = m.cell(ArmKind::Rewrite, 2);
        assert!((cell.calls.mean - (a0 + 2.0 * (1.0 - a0))).abs() < 1e-12);
        assert!((cell.calls.variance - a0 * (1.0 - a0)).abs() < 1e-12);
        assert!((cell.draws.mean - (7.0 + 7.0 * (1.0 - a0))).abs() < 1e-12);
        assert!((cell.draws.variance - 49.0 * a0 * (1.0 - a0)).abs() < 1e-9);
    }
}
