//! The harness's own record of what is live, and the ranking metrics.
//!
//! [`LiveTable`] is built from the workload's events by this file's code alone: it shares nothing
//! with `ptr-pg`, `ptr-search` or `ptr-runtime`, and it never asks the product. Every hit of every
//! list the product returns is held against it; a hit it does not call live is a stale hit. The
//! cross-check against the product's own projection (`lifecycle_projection_mismatches`) is done by
//! the runner, with the product's `is_admissible`.

use std::collections::HashMap;

use super::workload::{capsule_id, Event, EventKind};

#[derive(Debug, Default)]
pub struct LiveTable {
    /// capsule id to (live generation, revoked).
    entries: HashMap<String, (u64, bool)>,
}

impl LiveTable {
    pub fn apply(&mut self, event: &Event) {
        let capsule = capsule_id(event.cue);
        match event.kind {
            EventKind::Commit => {
                self.entries.insert(capsule, (1, false));
            }
            EventKind::Supersede { new, .. } => {
                if let Some(entry) = self.entries.get_mut(&capsule) {
                    entry.0 = new;
                }
            }
            EventKind::Revoke { generation } => {
                if let Some(entry) = self.entries.get_mut(&capsule) {
                    if entry.0 == generation {
                        entry.1 = true;
                    }
                }
            }
        }
    }

    /// Whether `generation` of `capsule` is the capsule's live generation and not revoked.
    pub fn is_live(&self, capsule: &str, generation: u64) -> bool {
        self.entries
            .get(capsule)
            .is_some_and(|(live, revoked)| *live == generation && !revoked)
    }

    /// The live (capsule, generation) pairs, in capsule order.
    pub fn live_pairs(&self) -> Vec<(String, u64)> {
        let mut pairs: Vec<(String, u64)> = self
            .entries
            .iter()
            .filter(|(_, (_, revoked))| !revoked)
            .map(|(capsule, (generation, _))| (capsule.clone(), *generation))
            .collect();
        pairs.sort();
        pairs
    }
}

/// The 1-based rank of `wanted` among the first `k` entries of `list`.
pub fn rank_within<T: PartialEq>(list: &[T], wanted: &T, k: usize) -> Option<usize> {
    list.iter()
        .take(k)
        .position(|item| item == wanted)
        .map(|at| at + 1)
}

/// `1000 * numerator / denominator`, or `empty` when there is nothing to divide.
pub fn permille(numerator: i64, denominator: i64, empty: i64) -> i64 {
    if denominator == 0 {
        empty
    } else {
        1000 * numerator / denominator
    }
}

/// Reciprocal-log discount of a relevant document at `rank` (1-based): with one relevant document
/// the ideal DCG is 1, so this is the nDCG of the list.
pub fn ndcg_gain(rank: usize) -> f64 {
    1.0 / ((rank as f64) + 1.0).log2()
}

/// The ranks recorded for a set of queries on one arm: `None` is a miss within the first `k`.
#[derive(Clone, Debug, Default)]
pub struct ArmRanks {
    pub ranks: Vec<Option<usize>>,
}

impl ArmRanks {
    pub fn push(&mut self, rank: Option<usize>) {
        self.ranks.push(rank);
    }

    pub fn hits(&self) -> i64 {
        self.ranks.iter().filter(|rank| rank.is_some()).count() as i64
    }

    pub fn recall_permille(&self) -> i64 {
        permille(self.hits(), self.ranks.len() as i64, 0)
    }

    pub fn ndcg_permille(&self) -> i64 {
        if self.ranks.is_empty() {
            return 0;
        }
        let total: f64 = self
            .ranks
            .iter()
            .flatten()
            .map(|rank| ndcg_gain(*rank))
            .sum();
        (1000.0 * total / self.ranks.len() as f64).floor() as i64
    }
}

/// The standard error of the mean paired difference of hit indicators, in permille, rounded up.
/// `a` and `b` are the two arms' ranks on the same queries.
pub fn paired_se_permille(a: &ArmRanks, b: &ArmRanks) -> i64 {
    let n = a.ranks.len().min(b.ranks.len());
    if n < 2 {
        return 0;
    }
    let differences: Vec<f64> = a
        .ranks
        .iter()
        .zip(&b.ranks)
        .map(|(x, y)| f64::from(u8::from(x.is_some())) - f64::from(u8::from(y.is_some())))
        .collect();
    let mean = differences.iter().sum::<f64>() / n as f64;
    let variance = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (1000.0 * (variance / n as f64).sqrt()).ceil() as i64
}

/// How many queries one of two arms hit and the other missed.
pub fn discordant_queries(a: &ArmRanks, b: &ArmRanks) -> i64 {
    a.ranks
        .iter()
        .zip(&b.ranks)
        .filter(|(x, y)| x.is_some() != y.is_some())
        .count() as i64
}

/// A 95 percent upper bound, in permille rounded up, on the share of discordant queries when
/// `discordant` of `n` were seen: the rule of three (3/n) when there are none, the normal
/// approximation `p + 1.96 sqrt(p (1 - p) / n)` otherwise (capped at 1000).
pub fn discordance_bound_permille(discordant: i64, n: i64) -> i64 {
    if n <= 0 {
        return 1000;
    }
    if discordant == 0 {
        return (3000 + n - 1) / n;
    }
    let p = discordant as f64 / n as f64;
    let bound = p + 1.96 * (p * (1.0 - p) / n as f64).sqrt();
    ((1000.0 * bound).ceil() as i64).min(1000)
}

/// The `fraction`-permille quantile (nearest rank) of `values`, zero for none.
pub fn quantile(values: &mut [u64], permille: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = (values.len() * permille).div_ceil(1000).saturating_sub(1);
    values[index.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::q003::workload::Event;

    fn event(kind: EventKind, cue: usize) -> Event {
        Event { kind, cue }
    }

    #[test]
    fn the_live_table_follows_commits_supersessions_and_revocations() {
        let mut table = LiveTable::default();
        table.apply(&event(EventKind::Commit, 0));
        table.apply(&event(EventKind::Commit, 1));
        assert!(table.is_live("cap-0000", 1));
        table.apply(&event(EventKind::Supersede { old: 1, new: 2 }, 0));
        assert!(!table.is_live("cap-0000", 1));
        assert!(table.is_live("cap-0000", 2));
        table.apply(&event(EventKind::Revoke { generation: 2 }, 0));
        assert!(
            !table.is_live("cap-0000", 2),
            "a revoked generation is not live"
        );
        assert!(
            !table.is_live("cap-0007", 1),
            "an unknown capsule is not live"
        );
        assert_eq!(table.live_pairs(), vec![("cap-0001".to_string(), 1)]);
    }

    #[test]
    fn ranking_metrics_are_what_their_definitions_say() {
        let list = ["a", "b", "c"];
        assert_eq!(rank_within(&list, &"c", 10), Some(3));
        assert_eq!(rank_within(&list, &"c", 2), None);
        assert_eq!(rank_within(&list, &"z", 10), None);
        let mut arm = ArmRanks::default();
        for rank in [Some(1), Some(3), None, Some(10)] {
            arm.push(rank);
        }
        assert_eq!(arm.hits(), 3);
        assert_eq!(arm.recall_permille(), 750);
        // (1 + 0.5 + 0 + 1/log2(11)) / 4
        let expected = (1.0 + 0.5 + 1.0 / 11f64.log2()) / 4.0;
        assert_eq!(arm.ndcg_permille(), (expected * 1000.0).floor() as i64);
        assert_eq!(ArmRanks::default().recall_permille(), 0);
        assert_eq!(permille(1, 0, 7), 7);
    }

    #[test]
    fn the_paired_standard_error_is_zero_for_identical_arms_and_grows_with_disagreement() {
        let mut same = ArmRanks::default();
        let mut other = ArmRanks::default();
        for index in 0..100 {
            same.push(Some(1));
            other.push(if index < 10 { None } else { Some(1) });
        }
        assert_eq!(paired_se_permille(&same, &same), 0);
        // differences: 10 ones among 100: mean 0.1, variance 0.0909, se = 0.0301 -> 31
        assert_eq!(paired_se_permille(&same, &other), 31);
        assert_eq!(paired_se_permille(&ArmRanks::default(), &same), 0);
    }

    #[test]
    fn discordance_counts_disagreements_and_bounds_their_share() {
        let mut a = ArmRanks::default();
        let mut b = ArmRanks::default();
        for (x, y) in [
            (Some(1), Some(2)),
            (Some(1), None),
            (None, Some(3)),
            (None, None),
        ] {
            a.push(x);
            b.push(y);
        }
        assert_eq!(discordant_queries(&a, &b), 2);
        assert_eq!(discordance_bound_permille(0, 180), 17);
        assert_eq!(discordance_bound_permille(0, 200), 15);
        // 18 of 200: p = 0.09, bound = 0.09 + 1.96 * 0.0202 = 0.1296
        assert_eq!(discordance_bound_permille(18, 200), 130);
        assert_eq!(discordance_bound_permille(5, 0), 1000);
        assert!(discordance_bound_permille(200, 200) <= 1000);
    }

    #[test]
    fn quantiles_take_the_nearest_rank() {
        let mut values: Vec<u64> = (1..=100).collect();
        assert_eq!(quantile(&mut values, 500), 50);
        assert_eq!(quantile(&mut values, 990), 99);
        assert_eq!(quantile(&mut [], 500), 0);
    }
}
