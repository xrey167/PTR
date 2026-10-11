//! The seeded world generator: items with a hidden true class, votes of probabilistic functions
//! with known confusion behaviour (accuracy per function, wrong votes drawn from a pinned
//! confusability table, one deliberately correlated pair), votes of verifier functions that
//! rule a class out, a vote budget per item, and a uniform gold sample.
//!
//! All draws are integer (permille thresholds on a seeded SplitMix64 stream), so a world is a
//! function of its seed and nothing else. The generated data is the harness's own: the oracle
//! (`oracle.rs`) and the majority-vote arm read it directly, and the product sees it only
//! through a `VoteMatrix` built from it (`runner.rs`).

use super::params::{self, Shape, World, CLASSES, PROBABILISTIC, VERIFIERS};
use crate::experiments::rng::Rng;

/// Names of the probabilistic functions, in vote-column order.
pub const PROBABILISTIC_NAMES: [&str; PROBABILISTIC] = ["lf_a", "lf_b", "lf_c", "lf_d", "lf_e"];
/// Names of the verifier functions, in vote-column order.
pub const VERIFIER_NAMES: [&str; VERIFIERS] = ["v_a", "v_b", "v_c", "v_d"];

/// One generated world.
#[derive(Clone, Debug, PartialEq)]
pub struct Generated {
    pub world: World,
    /// The hidden true class of every item.
    pub truth: Vec<usize>,
    /// `class_votes[item][function]`: the class a probabilistic function voted, or `None`.
    pub class_votes: Vec<[Option<usize>; PROBABILISTIC]>,
    /// `vetoes[item][verifier]`: the class a verifier ruled out, or `None`.
    pub vetoes: Vec<[Option<usize>; VERIFIERS]>,
    /// The items of the uniform gold sample, in the order they are pushed into the set.
    pub gold: Vec<usize>,
}

/// An index drawn with integer weights.
fn weighted(rng: &mut Rng, weights: &[u64]) -> usize {
    let total: u64 = weights.iter().sum();
    let mut point = rng.below(total);
    for (index, weight) in weights.iter().enumerate() {
        if point < *weight {
            return index;
        }
        point -= *weight;
    }
    weights.len() - 1
}

fn chance_permille(rng: &mut Rng, permille: u64) -> bool {
    rng.below(1000) < permille
}

/// A wrong class for `truth`, drawn from the pinned confusability row.
fn wrong_vote(rng: &mut Rng, truth: usize) -> usize {
    let row = &params::WRONG_CLASS_WEIGHTS[truth * CLASSES..(truth + 1) * CLASSES];
    weighted(rng, row)
}

/// Generate the world `world` of `seed` at `shape`.
pub fn generate(seed: u64, world: World, shape: Shape) -> Generated {
    let mut rng = Rng::new(seed ^ world.salt().wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let accuracy = world.accuracy_permille();
    let coverage = world.verifier_coverage_permille();
    let [first, second] = params::CORRELATED_PAIR;
    let mut truth = Vec::with_capacity(shape.items);
    let mut class_votes = Vec::with_capacity(shape.items);
    let mut vetoes = Vec::with_capacity(shape.items);
    for _ in 0..shape.items {
        let class = weighted(&mut rng, &params::CLASS_PRIOR_PERMILLE);
        let budget =
            params::VOTES_PER_ITEM[weighted(&mut rng, &params::VOTES_PER_ITEM_WEIGHT_PERMILLE)];
        // A seeded subset of `budget` functions: a partial Fisher-Yates shuffle.
        let mut order: Vec<usize> = (0..PROBABILISTIC).collect();
        for slot in 0..budget {
            let pick = slot + rng.index(PROBABILISTIC - slot);
            order.swap(slot, pick);
        }
        let mut selected = [false; PROBABILISTIC];
        for function in &order[..budget] {
            selected[*function] = true;
        }
        let mut votes = [None; PROBABILISTIC];
        for function in 0..PROBABILISTIC {
            if !selected[function] {
                continue;
            }
            let copies = function == second
                && selected[first]
                && chance_permille(&mut rng, world.correlated_copy_permille());
            votes[function] = if copies {
                votes[first]
            } else if chance_permille(&mut rng, accuracy[function]) {
                Some(class)
            } else {
                Some(wrong_vote(&mut rng, class))
            };
        }
        let mut vetoed = [None; VERIFIERS];
        for verifier in 0..VERIFIERS {
            if !chance_permille(&mut rng, coverage[verifier]) {
                continue;
            }
            vetoed[verifier] = if chance_permille(&mut rng, world.contamination_permille()) {
                Some(class)
            } else {
                // A uniformly drawn wrong class: skip the true one.
                let pick = rng.index(CLASSES - 1);
                Some(if pick >= class { pick + 1 } else { pick })
            };
        }
        truth.push(class);
        class_votes.push(votes);
        vetoes.push(vetoed);
    }
    // The uniform gold sample: the first `gold` items of a seeded shuffle of all items.
    let mut shuffle = rng.fork(0x601D);
    let mut items: Vec<usize> = (0..shape.items).collect();
    let gold = shape.gold.min(shape.items);
    for slot in 0..gold {
        let pick = slot + shuffle.index(shape.items - slot);
        items.swap(slot, pick);
    }
    items.truncate(gold);
    Generated {
        world,
        truth,
        class_votes,
        vetoes,
        gold: items,
    }
}

impl Generated {
    /// Items on which at least one verifier voted.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn vetoed_items(&self) -> usize {
        self.vetoes
            .iter()
            .filter(|row| row.iter().any(Option::is_some))
            .count()
    }

    /// Vetoes that rule out the item's true class: zero in a sound world.
    pub fn vetoes_of_the_truth(&self) -> usize {
        self.vetoes
            .iter()
            .zip(&self.truth)
            .map(|(row, truth)| row.iter().filter(|veto| **veto == Some(*truth)).count())
            .sum()
    }

    /// The classes the verifiers of an item rule out.
    pub fn vetoed_classes(&self, item: usize) -> Vec<usize> {
        let mut classes: Vec<usize> = self.vetoes[item].iter().flatten().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    /// How many probabilistic functions voted on an item.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn class_vote_count(&self, item: usize) -> usize {
        self.class_votes[item].iter().flatten().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::f002::params::Scale;

    #[test]
    fn a_seed_gives_the_same_world_and_another_seed_another() {
        let shape = Scale::Reduced.shape();
        let first = generate(1001, World::Sound, shape);
        assert_eq!(first, generate(1001, World::Sound, shape));
        assert_ne!(first, generate(1002, World::Sound, shape));
        assert_ne!(
            first.truth,
            generate(1001, World::Contaminated, shape).truth
        );
    }

    #[test]
    fn the_budget_votes_and_the_gold_sample_have_the_pinned_shape() {
        let shape = Scale::Reduced.shape();
        let world = generate(1001, World::Sound, shape);
        assert_eq!(world.truth.len(), shape.items);
        assert_eq!(world.gold.len(), shape.gold);
        let mut gold = world.gold.clone();
        gold.sort_unstable();
        gold.dedup();
        assert_eq!(gold.len(), shape.gold, "gold items are distinct");
        let mut seen = [0usize; PROBABILISTIC + 1];
        for item in 0..shape.items {
            seen[world.class_vote_count(item)] += 1;
        }
        assert_eq!(seen[0] + seen[1] + seen[4], 0);
        assert!(seen[2] > 0 && seen[3] > seen[2] && seen[5] > 0, "{seen:?}");
    }

    #[test]
    fn sound_worlds_never_veto_the_truth_and_the_contaminated_one_does() {
        let shape = Scale::Reduced.shape();
        assert_eq!(generate(1001, World::Sound, shape).vetoes_of_the_truth(), 0);
        assert_eq!(
            generate(1001, World::Homogeneous, shape).vetoes_of_the_truth(),
            0
        );
        assert_eq!(generate(1001, World::Homogeneous, shape).vetoed_items(), 0);
        assert!(generate(1001, World::Contaminated, shape).vetoes_of_the_truth() > 0);
    }

    #[test]
    fn the_functions_vote_with_roughly_their_pinned_accuracy_and_the_pair_is_correlated() {
        let shape = Shape {
            items: 20_000,
            gold: 10,
            budget: 10,
            resamples: 10,
        };
        let world = generate(7, World::Sound, shape);
        for function in 0..PROBABILISTIC {
            let (mut right, mut cast) = (0u64, 0u64);
            for item in 0..shape.items {
                if let Some(vote) = world.class_votes[item][function] {
                    cast += 1;
                    right += u64::from(vote == world.truth[item]);
                }
            }
            let measured = right * 1000 / cast;
            let pinned = params::ACCURACY_HETEROGENEOUS_PERMILLE[function];
            if function == params::CORRELATED_PAIR[1] {
                // It repeats a more accurate function in 60% of the cases both vote.
                assert!(measured + 30 >= pinned, "function {function}: {measured}");
            } else {
                assert!(
                    measured.abs_diff(pinned) <= 30,
                    "function {function}: {measured} vs {pinned}"
                );
            }
        }
        let [first, second] = params::CORRELATED_PAIR;
        let (mut both, mut equal) = (0u64, 0u64);
        for votes in &world.class_votes {
            if let (Some(a), Some(b)) = (votes[first], votes[second]) {
                both += 1;
                equal += u64::from(a == b);
            }
        }
        assert!(equal * 1000 / both >= 600, "{equal}/{both}");
    }

    #[test]
    fn a_verifier_vetoes_a_wrong_class_in_a_sound_world_and_class_priors_hold() {
        let shape = Shape {
            items: 20_000,
            gold: 10,
            budget: 10,
            resamples: 10,
        };
        let world = generate(7, World::Sound, shape);
        let mut counts = [0usize; CLASSES];
        for class in &world.truth {
            counts[*class] += 1;
        }
        for (class, count) in counts.iter().enumerate() {
            let measured = (*count * 1000 / shape.items) as u64;
            assert!(
                measured.abs_diff(params::CLASS_PRIOR_PERMILLE[class]) <= 25,
                "{counts:?}"
            );
        }
        assert_eq!(world.vetoes_of_the_truth(), 0);
    }
}
