use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use ptr_search::SearchHit;
use ptr_types::{CapsuleId, Generation};

use crate::error::FastMemoryError;
use crate::memory::FastMemory;
use crate::projection::IdentifierCodebook;
use crate::state::Readout;

/// Backend label carried by every hit this crate produces.
pub const FASTMEM_BACKEND: &str = "fastmem";

/// An explicit fact a readout may be decoded into: a capsule at one generation,
/// the identifier code its writes stored as value, and the codebook the code
/// is from.
///
/// Its fields are private and it is built only by [`IdentifierCodebook::fact`]
/// and [`FastMemory::fact_codes`], so its code is always its codebook's code
/// for the fact, and [`decode_readout`] can refuse a code from another
/// codebook than the readout's. Neither builds one for a lifecycle target
/// that is not a capsule (`constraint:<key>` or `procedure:<id>`), and
/// [`decode_readout`] refuses one again: a hit names a capsule.
#[derive(Clone, Debug, PartialEq)]
pub struct FactCode {
    capsule: CapsuleId,
    generation: Generation,
    code: Vec<f32>,
    codebook: IdentifierCodebook,
}

impl FactCode {
    pub fn capsule(&self) -> &CapsuleId {
        &self.capsule
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    /// The fact's code: `codebook().code_for(capsule(), generation())`.
    pub fn code(&self) -> &[f32] {
        &self.code
    }

    /// The codebook the code is from.
    pub fn codebook(&self) -> IdentifierCodebook {
        self.codebook
    }
}

impl IdentifierCodebook {
    /// The value code of a fact at one generation. Writers store exactly this
    /// vector as the value of a write about the fact, and decoding compares
    /// readouts against it; a new generation is a new fact identity.
    pub fn code_for(&self, capsule: &CapsuleId, generation: Generation) -> Vec<f32> {
        self.code(&format!("{capsule}@{}", generation.0))
    }

    /// A capsule at one generation as a decoding candidate: its code under
    /// this codebook, bound to this codebook.
    ///
    /// # Errors
    /// Refuses a lifecycle target that is not a capsule, one whose id starts
    /// with `constraint:` or `procedure:` (`ReservedTarget`): the runtime
    /// accepts no capsule id in either namespace, and a hit decoded from it
    /// would name a hard constraint or a procedure as a capsule.
    pub fn fact(
        &self,
        capsule: CapsuleId,
        generation: Generation,
    ) -> Result<FactCode, FastMemoryError> {
        check_capsule(&capsule)?;
        Ok(self.capsule_fact(capsule, generation))
    }

    /// [`Self::fact`] for a target already known to be a capsule.
    fn capsule_fact(&self, capsule: CapsuleId, generation: Generation) -> FactCode {
        FactCode {
            code: self.code_for(&capsule, generation),
            capsule,
            generation,
            codebook: *self,
        }
    }
}

/// Prefixes of the lifecycle targets that are not capsules. The runtime refuses
/// a capsule id in either namespace, so a source key carrying one names a hard
/// constraint or a procedure, never a capsule. [`is_reserved_target`] is the
/// one test of it: fact candidates, explicit ([`IdentifierCodebook::fact`]) or
/// derived ([`FastMemory::fact_codes`]), and decoding ([`decode_readout`]) all
/// apply it.
const NON_CAPSULE_NAMESPACES: [&str; 2] = ["constraint:", "procedure:"];

/// Whether a lifecycle target is in a namespace reserved for targets that are
/// not capsules.
fn is_reserved_target(target: &str) -> bool {
    NON_CAPSULE_NAMESPACES
        .iter()
        .any(|namespace| target.starts_with(namespace))
}

/// Refuse a fact candidate whose capsule is a reserved lifecycle target.
fn check_capsule(capsule: &CapsuleId) -> Result<(), FastMemoryError> {
    if is_reserved_target(&capsule.0) {
        return Err(FastMemoryError::ReservedTarget {
            target: capsule.0.clone(),
        });
    }
    Ok(())
}

impl FastMemory {
    /// The candidate facts a readout may decode into: every distinct capsule
    /// source the journal holds, and nothing else, each coded with this
    /// memory's own codebook ([`Self::codebook`]). A fact that was never
    /// written, or whose writes were revoked, is not a candidate. Writes derived
    /// from a `constraint:<key>` or `procedure:<id>` source still shape the
    /// state and still gate reads, but they are not candidates: a decoded hit
    /// names a capsule, and those sources are not capsules
    /// ([`IdentifierCodebook::fact`] refuses them by the same test).
    pub fn fact_codes(&self) -> Vec<FactCode> {
        let book = self.codebook();
        let sources: BTreeSet<(String, Generation)> = self
            .writes()
            .iter()
            .map(|write| (write.source().key.clone(), write.source().generation))
            .filter(|(key, _)| !is_reserved_target(key))
            .collect();
        sources
            .into_iter()
            .map(|(key, generation)| book.capsule_fact(CapsuleId(key), generation))
            .collect()
    }
}

/// When a readout is confident enough to name a fact.
///
/// [`decode_readout`] refuses a policy with a zero `limit` or a threshold that
/// is NaN, infinite or negative: a comparison against a NaN threshold is never
/// true, so such a policy would let an ambiguous or weak readout through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodePolicy {
    /// Most hits returned; at least one.
    pub limit: usize,
    /// Minimum `<readout, code>` for a fact to be returned at all. Finite and
    /// non-negative.
    pub min_score: f32,
    /// Minimum lead of the best fact over the runner-up. Below it the readout
    /// is ambiguous and the answer is `Unknown`. Finite and non-negative.
    pub min_margin: f32,
}

/// The result of decoding a readout, as [`decode_readout`] returns it. The
/// variants state what `decode_readout` guarantees; a value built by hand is
/// only data (its hits need not be decoded, ordered or at the lowest stage),
/// and no function of this crate takes one.
#[derive(Clone, Debug, PartialEq)]
pub enum Recall {
    /// Named facts, best first, each once and at the lowest evidence stage.
    Hits(Vec<SearchHit>),
    /// No fact scores high enough, or the best does not lead clearly. Unknown
    /// is a valid answer, not a failure.
    Unknown,
}

/// Decode a readout against candidate fact codes.
///
/// The score of a fact is `<readout, code>`: with unit-length, nearly
/// orthogonal codes it estimates the memory's weight on that fact. Hits start
/// at the lowest evidence stage, so a recall must still be resolved against
/// live generations and verified before it is relied on. Ties are broken by
/// capsule, then generation. A returned `Hits` always names at least one fact,
/// and none twice.
///
/// Every fact code must be from the readout's codebook, the one its memory's
/// values are codes of. A code from another codebook of the same length would
/// score plausible but meaningless weights (crosstalk only), so it is refused,
/// not scored. Every fact must name a capsule: a hit is a capsule candidate,
/// and a lifecycle target that is not one is refused, not decoded.
///
/// A fact, one capsule at one generation, is scored once however often it is
/// supplied: a fact code equal to an earlier candidate, as when a caller
/// combines candidate sources that overlap, is skipped after its own checks.
/// Scored again, it would be the runner-up with exactly the best score, so a
/// clear match would be `Unknown` under any positive margin and named twice
/// under a zero one.
///
/// # Errors
/// Before scoring, refuses a policy with a zero limit or a NaN, infinite or
/// negative threshold. Then refuses, fact by fact, a fact code from another
/// codebook than the readout's (`CodebookMismatch`), one naming a
/// `constraint:` or `procedure:` target (`ReservedTarget`, which no public
/// constructor of [`FactCode`] builds), one whose length differs from the
/// readout's, a score that is not finite (a nonfinite readout or code
/// cell, or an overflowing product), which the confidence checks could not
/// order, and a fact code naming the same capsule and generation as an
/// earlier candidate but not equal to it (`ConflictingFact`, which no public
/// constructor builds either: under one codebook a fact has one code), since
/// at most one of the two is the fact's.
pub fn decode_readout<'a, I>(
    readout: &Readout,
    facts: I,
    policy: DecodePolicy,
) -> Result<Recall, FastMemoryError>
where
    I: IntoIterator<Item = &'a FactCode>,
{
    check_policy(&policy)?;
    let mut scored = Vec::new();
    // Each fact scored so far, with its position among the candidates.
    let mut seen = BTreeMap::new();
    for (index, fact) in facts.into_iter().enumerate() {
        if fact.codebook != readout.codebook() {
            return Err(FastMemoryError::CodebookMismatch {
                index,
                expected: readout.codebook(),
                actual: fact.codebook,
            });
        }
        check_capsule(&fact.capsule)?;
        if fact.code.len() != readout.values.len() {
            return Err(FastMemoryError::DimensionMismatch {
                field: "fact code",
                expected: readout.values.len(),
                actual: fact.code.len(),
            });
        }
        let score: f32 = readout
            .values
            .iter()
            .zip(&fact.code)
            .map(|(r, c)| r * c)
            .sum();
        if !score.is_finite() {
            return Err(FastMemoryError::NonFinite {
                field: "score",
                index,
            });
        }
        match seen.entry((&fact.capsule, fact.generation)) {
            Entry::Vacant(entry) => {
                entry.insert((index, fact));
            }
            // A copy would be the runner-up with exactly the best score. Both
            // codes are finite, as their scores are, so a copy compares equal.
            Entry::Occupied(entry) if entry.get().1 == fact => continue,
            Entry::Occupied(entry) => {
                return Err(FastMemoryError::ConflictingFact {
                    capsule: fact.capsule.clone(),
                    generation: fact.generation,
                    first: entry.get().0,
                    index,
                });
            }
        }
        scored.push((score, fact));
    }
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.capsule.cmp(&right.1.capsule))
            .then_with(|| left.1.generation.cmp(&right.1.generation))
    });
    let Some(best) = scored.first().map(|(score, _)| *score) else {
        return Ok(Recall::Unknown);
    };
    let runner_up = scored.get(1).map(|(score, _)| *score).unwrap_or(0.0);
    if best < policy.min_score || best - runner_up < policy.min_margin {
        return Ok(Recall::Unknown);
    }
    Ok(Recall::Hits(
        scored
            .into_iter()
            .filter(|(score, _)| *score >= policy.min_score)
            .take(policy.limit)
            .map(|(score, fact)| {
                SearchHit::new(
                    fact.capsule.clone(),
                    fact.generation,
                    score,
                    FASTMEM_BACKEND,
                )
            })
            .collect(),
    ))
}

/// Refuse a policy under which the confidence checks would fail open: with a
/// NaN threshold every `<` against it is false, and a zero limit turns a clear
/// winner into an empty `Hits`.
fn check_policy(policy: &DecodePolicy) -> Result<(), FastMemoryError> {
    if policy.limit == 0 {
        return Err(FastMemoryError::InvalidConfig {
            field: "limit",
            value: 0,
            message: "a decode must be allowed to name at least one fact",
        });
    }
    for (field, value) in [
        ("min_score", policy.min_score),
        ("min_margin", policy.min_margin),
    ] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(FastMemoryError::InvalidThreshold { field, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::WriteSeq;
    use ptr_search::EvidenceStage;

    /// The codebook every hand-made fact and readout here claims.
    fn book() -> IdentifierCodebook {
        IdentifierCodebook::new(1, 2).unwrap()
    }

    fn fact(id: &str, code: Vec<f32>) -> FactCode {
        FactCode {
            capsule: CapsuleId::from(id),
            generation: Generation(2),
            code,
            codebook: book(),
        }
    }

    fn readout(values: Vec<f32>, as_of: u64) -> Readout {
        Readout::new(values, WriteSeq(as_of), book())
    }

    fn policy() -> DecodePolicy {
        DecodePolicy {
            limit: 5,
            min_score: 0.5,
            min_margin: 0.2,
        }
    }

    #[test]
    fn a_clear_winner_is_named_and_starts_as_a_search_candidate() {
        let readout = readout(vec![0.9, 0.1], 3);
        let facts = [fact("b", vec![0.0, 1.0]), fact("a", vec![1.0, 0.0])];
        let Recall::Hits(hits) = decode_readout(&readout, &facts, policy()).unwrap() else {
            panic!("a clear winner is named");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].capsule(), &CapsuleId::from("a"));
        assert_eq!(hits[0].stage(), EvidenceStage::SearchCandidate);
        assert_eq!(hits[0].backend, FASTMEM_BACKEND);
    }

    #[test]
    fn an_ambiguous_or_weak_readout_is_unknown() {
        let facts = [fact("a", vec![1.0, 0.0]), fact("b", vec![0.0, 1.0])];
        let ambiguous = readout(vec![0.7, 0.65], 1);
        assert_eq!(
            decode_readout(&ambiguous, &facts, policy()).unwrap(),
            Recall::Unknown
        );
        let weak = readout(vec![0.3, 0.0], 1);
        assert_eq!(
            decode_readout(&weak, &facts, policy()).unwrap(),
            Recall::Unknown
        );
        assert_eq!(
            decode_readout(&weak, std::iter::empty(), policy()).unwrap(),
            Recall::Unknown
        );
    }

    #[test]
    fn a_policy_that_would_fail_open_is_refused_before_scoring() {
        let facts = [fact("a", vec![1.0, 0.0]), fact("b", vec![0.0, 1.0])];
        // Ambiguous under any sound margin: a NaN margin used to name "a", and
        // a NaN minimum score used to return `Hits` naming nothing.
        let ambiguous = readout(vec![0.7, 0.65], 1);
        let unsound = [
            ("min_margin", f32::NAN),
            ("min_margin", -0.1),
            ("min_margin", f32::NEG_INFINITY),
            ("min_score", f32::NAN),
            ("min_score", -1.0),
            ("min_score", f32::INFINITY),
        ];
        for (field, value) in unsound {
            let mut bad = policy();
            match field {
                "min_margin" => bad.min_margin = value,
                _ => bad.min_score = value,
            }
            for candidates in [&facts[..], &[]] {
                let refused = decode_readout(&ambiguous, candidates, bad);
                assert!(
                    matches!(
                        refused,
                        Err(FastMemoryError::InvalidThreshold { field: named, value: seen })
                            if named == field && seen.to_bits() == value.to_bits()
                    ),
                    "{field}={value}: {refused:?}"
                );
            }
        }
        let nothing = DecodePolicy {
            limit: 0,
            ..policy()
        };
        assert!(matches!(
            decode_readout(&ambiguous, &facts, nothing),
            Err(FastMemoryError::InvalidConfig { field: "limit", .. })
        ));

        // Zero thresholds are sound: the clear winner is still named alone.
        let permissive = DecodePolicy {
            limit: 1,
            min_score: 0.0,
            min_margin: -0.0,
        };
        let clear = readout(vec![0.9, 0.1], 1);
        let Recall::Hits(hits) = decode_readout(&clear, &facts, permissive).unwrap() else {
            panic!("zero thresholds still name a clear winner");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].capsule(), &CapsuleId::from("a"));
    }

    #[test]
    fn a_candidate_equal_to_an_earlier_one_is_scored_once() {
        // The copy of "a" used to be the runner-up with exactly the best
        // score: a positive margin made the clear winner `Unknown`, and a zero
        // margin named it twice.
        let clear = readout(vec![0.9, 0.1], 1);
        let facts = [
            fact("a", vec![1.0, 0.0]),
            fact("b", vec![0.0, 1.0]),
            fact("a", vec![1.0, 0.0]),
        ];
        let Recall::Hits(hits) = decode_readout(&clear, &facts, policy()).unwrap() else {
            panic!("a duplicated clear winner is still named");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].capsule(), &CapsuleId::from("a"));
        let permissive = DecodePolicy {
            limit: 5,
            min_score: 0.0,
            min_margin: 0.0,
        };
        let Recall::Hits(hits) = decode_readout(&clear, &facts, permissive).unwrap() else {
            panic!("zero thresholds name every candidate");
        };
        let named: Vec<_> = hits
            .iter()
            .map(|hit| (hit.capsule().0.as_str(), hit.generation()))
            .collect();
        assert_eq!(named, [("a", Generation(2)), ("b", Generation(2))]);
        // Another generation of the same capsule is another fact.
        let later = FactCode {
            generation: Generation(3),
            ..fact("a", vec![1.0, 0.0])
        };
        assert_eq!(
            decode_readout(&clear, [&facts[0], &later], policy()).unwrap(),
            Recall::Unknown
        );
    }

    #[test]
    fn two_candidates_for_one_fact_with_different_codes_are_refused() {
        // A fact has one code under a codebook, so at most one of these is
        // "a"'s. Both used to be scored, and whichever led was named "a".
        let clear = readout(vec![0.9, 0.1], 1);
        let facts = [
            fact("a", vec![1.0, 0.0]),
            fact("b", vec![0.0, 1.0]),
            fact("a", vec![0.0, 1.0]),
        ];
        let conflict = FastMemoryError::ConflictingFact {
            capsule: CapsuleId::from("a"),
            generation: Generation(2),
            first: 0,
            index: 2,
        };
        assert_eq!(
            decode_readout(&clear, &facts, policy()),
            Err(conflict.clone())
        );
        assert_eq!(conflict.code(), "PTR_FASTMEM_CONFLICTING_FACT");
        // The later candidate's own checks come first.
        let shorter = fact("a", vec![1.0]);
        assert!(matches!(
            decode_readout(&clear, [&facts[0], &shorter], policy()),
            Err(FastMemoryError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn a_nonfinite_score_is_refused_rather_than_ranked() {
        let facts = [fact("a", vec![1.0, 0.0]), fact("b", vec![1.0, 1.0])];
        // NaN * 0 is NaN, so the poisoned cell spoils the first score already;
        // unchecked, both NaN scores used to decode to `Hits` naming nothing.
        let poisoned = readout(vec![0.9, f32::NAN], 1);
        assert_eq!(
            decode_readout(&poisoned, &facts, policy()),
            Err(FastMemoryError::NonFinite {
                field: "score",
                index: 0
            })
        );
        let overflowing = readout(vec![f32::MAX, f32::MAX], 1);
        assert_eq!(
            decode_readout(&overflowing, &facts, policy()),
            Err(FastMemoryError::NonFinite {
                field: "score",
                index: 1
            })
        );
    }

    #[test]
    fn a_constraint_or_procedure_target_is_neither_built_nor_decoded_as_a_fact() {
        // The runtime reports these targets' generations as live, so a hit
        // naming one would pass the lifecycle check at use as a capsule.
        for target in [
            "constraint:budget",
            "procedure:deploy",
            "constraint:",
            "procedure:",
        ] {
            assert_eq!(
                book().fact(CapsuleId::from(target), Generation(2)),
                Err(FastMemoryError::ReservedTarget {
                    target: target.into()
                }),
                "{target}"
            );
        }
        assert_eq!(
            FastMemoryError::ReservedTarget {
                target: "constraint:budget".into()
            }
            .code(),
            "PTR_FASTMEM_RESERVED_TARGET"
        );
        // Only the prefixes are reserved.
        for capsule in ["budget-constraint:1", "procedure", "constraints"] {
            assert!(
                book().fact(CapsuleId::from(capsule), Generation(2)).is_ok(),
                "{capsule}"
            );
        }
        // A fact code built around the constructor, which only this crate
        // can do, is refused at decode before it is scored: it would lead.
        let clear = readout(vec![0.9, 0.1], 1);
        let facts = [
            fact("b", vec![0.0, 1.0]),
            fact("procedure:deploy", vec![1.0, 0.0]),
        ];
        assert_eq!(
            decode_readout(&clear, &facts, policy()),
            Err(FastMemoryError::ReservedTarget {
                target: "procedure:deploy".into()
            })
        );
    }

    #[test]
    fn a_fact_code_from_another_codebook_is_refused_before_scoring() {
        // Same length, other seed: the codes are unrelated to what the
        // memory stored, and scoring them used to name a fact by crosstalk.
        let other = IdentifierCodebook::new(2, 2).unwrap();
        let clear = readout(vec![0.9, 0.1], 1);
        let foreign = [
            fact("a", vec![1.0, 0.0]),
            other.fact(CapsuleId::from("b"), Generation(2)).unwrap(),
        ];
        assert_eq!(
            decode_readout(&clear, &foreign, policy()),
            Err(FastMemoryError::CodebookMismatch {
                index: 1,
                expected: book(),
                actual: other,
            })
        );
        assert_eq!(
            FastMemoryError::CodebookMismatch {
                index: 1,
                expected: book(),
                actual: other,
            }
            .code(),
            "PTR_FASTMEM_CODEBOOK_MISMATCH"
        );
        // A codebook of another length with the same seed is another codebook.
        let longer = IdentifierCodebook::new(1, 3).unwrap();
        assert!(matches!(
            decode_readout(
                &clear,
                &[longer.fact(CapsuleId::from("a"), Generation(2)).unwrap()],
                policy()
            ),
            Err(FastMemoryError::CodebookMismatch { index: 0, .. })
        ));
        // The readout's own codebook decodes.
        let own = book().fact(CapsuleId::from("a"), Generation(2)).unwrap();
        assert_eq!(
            own.code(),
            book().code_for(&CapsuleId::from("a"), Generation(2))
        );
        assert!(decode_readout(&clear, [&own], policy()).is_ok());
    }
}
