use std::collections::{BTreeMap, BTreeSet};

use ptr_search::{EvidencePromotionError, EvidenceStage, SearchHit};
use ptr_types::{CapsuleId, Generation, Validity};

/// A lifecycle authority as the runtime keeps one: each capsule's live
/// generation, and a tombstone per revoked generation that leaves the live
/// generation in place. It answers as `PtrRuntime::generation_validity` does.
struct Lifecycle {
    live: BTreeMap<CapsuleId, Generation>,
    revoked: BTreeSet<(CapsuleId, Generation)>,
    disputed: BTreeSet<(CapsuleId, Generation)>,
}

impl Lifecycle {
    fn new(live: &[(&str, u64)]) -> Self {
        Self {
            live: live
                .iter()
                .map(|(capsule, generation)| (CapsuleId::from(*capsule), Generation(*generation)))
                .collect(),
            revoked: BTreeSet::new(),
            disputed: BTreeSet::new(),
        }
    }

    fn live_generation(&self, capsule: &CapsuleId) -> Option<Generation> {
        self.live.get(capsule).copied()
    }

    fn generation_validity(&self, capsule: &CapsuleId, generation: Generation) -> Option<Validity> {
        let key = (capsule.clone(), generation);
        if self.revoked.contains(&key) {
            return Some(Validity::Revoked);
        }
        if self.disputed.contains(&key) {
            return Some(Validity::Disputed);
        }
        match self.live.get(capsule) {
            Some(live) if *live == generation => Some(Validity::Live),
            Some(live) if *live > generation => Some(Validity::Superseded),
            Some(_) | None => None,
        }
    }
}

fn possible(capsule: &str, generation: u64) -> SearchHit {
    let mut hit = SearchHit::new(
        CapsuleId::from(capsule),
        Generation(generation),
        0.9,
        "test",
    );
    hit.mark_possible().unwrap();
    hit
}

#[test]
fn retrieval_cannot_jump_directly_to_known() {
    let lifecycle = Lifecycle::new(&[("c1", 4)]);
    let mut hit = SearchHit::new(CapsuleId::from("c1"), Generation(3), 0.9, "test");
    assert_eq!(
        hit.promote_verified(true),
        Err(EvidencePromotionError::WrongStage)
    );
    // Observing before the hit is possible evidence asks the lifecycle nothing.
    assert_eq!(
        hit.observe(|_, _| unreachable!("asked before the stage allows"), true),
        Err(EvidencePromotionError::WrongStage)
    );

    hit.mark_possible().unwrap();
    assert_eq!(
        hit.observe(|c, g| lifecycle.generation_validity(c, g), true),
        Err(EvidencePromotionError::StaleGeneration)
    );
    assert_eq!(hit.stage(), EvidenceStage::PossibleEvidence);

    let lifecycle = Lifecycle::new(&[("c1", 3)]);
    assert_eq!(
        hit.observe(|c, g| lifecycle.generation_validity(c, g), false),
        Err(EvidencePromotionError::SourceNotResolved)
    );
    hit.observe(|c, g| lifecycle.generation_validity(c, g), true)
        .unwrap();
    assert_eq!(hit.stage(), EvidenceStage::Observed);
    hit.promote_verified(true).unwrap();
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
}

#[test]
fn a_revoked_generation_is_never_observed_although_it_is_still_the_live_one() {
    let mut lifecycle = Lifecycle::new(&[("c", 5)]);
    let mut hit = possible("c", 5);
    let capsule = CapsuleId::from("c");
    lifecycle.revoked.insert((capsule.clone(), Generation(5)));
    // The revocation left generation 5 the capsule's live one: comparing the
    // hit with it, as observe did, promoted a revoked hit to VerifiedKnown.
    assert_eq!(lifecycle.live_generation(&capsule), Some(Generation(5)));
    let mut asked = Vec::new();
    assert_eq!(
        hit.observe(
            |c, g| {
                asked.push((c.clone(), g));
                lifecycle.generation_validity(c, g)
            },
            true
        ),
        Err(EvidencePromotionError::RevokedGeneration)
    );
    // The lifecycle is asked about the hit's own capsule and generation, once.
    assert_eq!(asked, vec![(capsule, Generation(5))]);
    assert_eq!(hit.stage(), EvidenceStage::PossibleEvidence);
    assert_eq!(
        hit.promote_verified(true),
        Err(EvidencePromotionError::WrongStage)
    );
}

#[test]
fn only_a_generation_the_lifecycle_answers_live_for_is_observed() {
    let mut lifecycle = Lifecycle::new(&[("c", 5), ("live", 1)]);
    lifecycle
        .disputed
        .insert((CapsuleId::from("c"), Generation(5)));
    let cases = [
        ("c", 5, Err(EvidencePromotionError::DisputedGeneration)),
        ("c", 4, Err(EvidencePromotionError::StaleGeneration)),
        // A generation ahead of the live one, and a capsule nobody knows.
        ("c", 6, Err(EvidencePromotionError::UnknownGeneration)),
        ("gone", 1, Err(EvidencePromotionError::UnknownGeneration)),
        ("live", 1, Ok(())),
    ];
    for (capsule, generation, expected) in cases {
        let mut hit = possible(capsule, generation);
        assert_eq!(
            hit.observe(|c, g| lifecycle.generation_validity(c, g), true),
            expected,
            "{capsule}@{generation}"
        );
        let stage = if expected.is_ok() {
            EvidenceStage::Observed
        } else {
            EvidenceStage::PossibleEvidence
        };
        assert_eq!(hit.stage(), stage, "{capsule}@{generation}");
    }
}

#[test]
fn every_promotion_refusal_has_a_stable_code() {
    let codes = [
        (EvidencePromotionError::WrongStage, "PTR_SEARCH_WRONG_STAGE"),
        (
            EvidencePromotionError::StaleGeneration,
            "PTR_SEARCH_STALE_GENERATION",
        ),
        (
            EvidencePromotionError::RevokedGeneration,
            "PTR_SEARCH_REVOKED_GENERATION",
        ),
        (
            EvidencePromotionError::DisputedGeneration,
            "PTR_SEARCH_DISPUTED_GENERATION",
        ),
        (
            EvidencePromotionError::UnknownGeneration,
            "PTR_SEARCH_UNKNOWN_GENERATION",
        ),
        (
            EvidencePromotionError::SourceNotResolved,
            "PTR_SEARCH_SOURCE_NOT_RESOLVED",
        ),
        (
            EvidencePromotionError::VerificationFailed,
            "PTR_SEARCH_VERIFICATION_FAILED",
        ),
    ];
    for (error, code) in codes {
        assert_eq!(error.code(), code);
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn a_hit_is_promoted_as_the_capsule_and_generation_the_lifecycle_answered_live_for() {
    // The hit's capsule and generation were public fields: a hit observed as
    // c@5 could be renamed r@5, a generation the lifecycle calls revoked, and
    // then promoted to VerifiedKnown. They are read-only now (the
    // compile_fail examples on SearchHit), so what is promoted is what
    // observe asked about.
    let mut lifecycle = Lifecycle::new(&[("c", 5), ("r", 5)]);
    lifecycle
        .revoked
        .insert((CapsuleId::from("r"), Generation(5)));
    let mut revoked = possible("r", 5);
    assert_eq!(
        revoked.observe(|c, g| lifecycle.generation_validity(c, g), true),
        Err(EvidencePromotionError::RevokedGeneration)
    );

    let mut hit = possible("c", 5);
    let mut asked = Vec::new();
    hit.observe(
        |c, g| {
            asked.push((c.clone(), g));
            lifecycle.generation_validity(c, g)
        },
        true,
    )
    .unwrap();
    hit.promote_verified(true).unwrap();
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
    assert_eq!(asked, vec![(hit.capsule().clone(), hit.generation())]);
    assert_eq!(
        (hit.capsule(), hit.generation()),
        (&CapsuleId::from("c"), Generation(5))
    );
    assert_eq!(
        lifecycle.generation_validity(hit.capsule(), hit.generation()),
        Some(Validity::Live)
    );
}
