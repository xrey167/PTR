//! The naive cross-check of the generator's labels.
//!
//! The generator labels every hit with its class and every question with its authoritative
//! answer while it builds the world. This module re-derives both from the finished corpus with
//! deliberately naive means that share no code with the code under test: substring search for
//! the quote in the pristine event text, string equality with the canonical rendering of the
//! claim, the generator's own lifecycle table, and a scan of the live events of healthy
//! capsules for the answer. A hit is eligible here iff the quote occurs in the pristine event
//! exactly at the cited span, the quote is the canonical affirmative statement of the claim,
//! and the generator says the capsule generation is the live generation of a healthy capsule.
//! A disagreement with a label is an `oracle_label_mismatches`.

use super::world::{render, Hit, Question, World};

/// Whether a hit is promotable by the naive reading of the corpus.
pub fn naive_eligible(world: &World, hit: &Hit) -> bool {
    let pristine = world
        .capsules
        .iter()
        .flat_map(|capsule| &capsule.generations)
        .find(|event| event.raw_id == hit.citation.raw_id.0);
    let Some(event) = pristine else { return false };
    let quote = hit.citation.quote.as_str();
    let at_span = event
        .text
        .match_indices(quote)
        .any(|(at, _)| at == hit.citation.start && at + quote.len() == hit.citation.end);
    if !at_span {
        return false;
    }
    let canonical = render(
        &hit.claim.subject,
        &hit.claim.predicate,
        false,
        &hit.claim.value,
    );
    if quote != canonical {
        return false;
    }
    world.generator_says_live(&hit.capsule, hit.generation)
}

/// The authoritative answer of a question by scanning the live events of healthy capsules for
/// the affirmative statement of its fact.
pub fn naive_answer(world: &World, question: &Question) -> Option<String> {
    let prefix = format!("{}: {} is ", question.subject, question.predicate);
    let mut found: Option<String> = None;
    for capsule in &world.capsules {
        if !world.generator_says_live(&capsule.id, capsule.live_generation()) {
            continue;
        }
        for line in capsule.live().text.lines() {
            let Some(rest) = line.strip_prefix(&prefix) else {
                continue;
            };
            if rest.starts_with("not ") {
                continue;
            }
            match &found {
                None => found = Some(rest.to_owned()),
                // Two live statements of one fact: no authoritative answer.
                Some(_) => return None,
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::q002::params::{self, Shape};
    use crate::experiments::q002::world::Class;

    #[test]
    fn the_naive_reading_agrees_with_every_generator_label_for_every_seed() {
        for seed in params::SEEDS {
            let world = World::generate(seed, Shape::PINNED).expect("a world");
            for hit in &world.hits {
                assert_eq!(
                    naive_eligible(&world, hit),
                    hit.class.eligible(),
                    "seed {seed} hit {} {:?}",
                    hit.id,
                    hit.class
                );
            }
            for question in &world.questions {
                assert_eq!(naive_answer(&world, question), question.expected);
            }
        }
    }

    #[test]
    fn the_naive_reading_can_disagree_with_a_wrong_label() {
        let world = World::generate(17, Shape::PINNED).expect("a world");
        let honest = world
            .hits
            .iter()
            .find(|h| h.class == Class::Honest)
            .expect("an honest hit");
        let mut shifted = honest.clone();
        shifted.citation.start += 1;
        assert!(!naive_eligible(&world, &shifted));
        let mut other_claim = honest.clone();
        other_claim.claim.value.push('x');
        assert!(!naive_eligible(&world, &other_claim));
        let mut stale = honest.clone();
        stale.generation += 1;
        assert!(!naive_eligible(&world, &stale));
        let poison = world
            .hits
            .iter()
            .find(|h| h.class == Class::P7)
            .expect("a P7 hit");
        assert!(!naive_eligible(&world, poison));
    }
}
