//! The typed promotion path: a hit becomes `VerifiedKnown` only through a resolved
//! source and a verified claim, through the public API alone.

use std::collections::BTreeMap;

use ptr_memory::{KnowledgeStore, RawEvent};
use ptr_search::{
    verify_claim, Claim, ClaimError, ClaimProof, EvidencePromotionError, EvidenceStage,
    ExactSourceResolver, RawEventSource, ResolvedSource, SearchHit, SourceCitation, SourceError,
    SourceOwners,
};
use ptr_types::{CapsuleId, EventRole, Generation, RawEventId, SessionId, Timestamp, Validity};

const BODY: &str = "ent-0001: status is active\nent-0002: tier is not gold\nent-0003: color is red";

fn store() -> KnowledgeStore {
    let mut store = KnowledgeStore::default();
    for (id, body) in [("raw-a", BODY), ("raw-b", "ent-0009: status is idle")] {
        store
            .append_raw(RawEvent::new(
                RawEventId::from(id),
                SessionId::from("s"),
                EventRole::User,
                body.as_bytes().to_vec(),
                Timestamp(1),
            ))
            .expect("a well-formed raw event");
    }
    store
}

/// Which capsule generations admitted which event: `raw-a` belongs to generations 2 and 3 of
/// capsule `c` (the later one cites the sources of the earlier), `raw-b` to generation 1 of `d`.
fn owners() -> SourceOwners {
    let mut owners = SourceOwners::default();
    for generation in [2, 3] {
        owners.record(
            RawEventId::from("raw-a"),
            CapsuleId::from("c"),
            Generation(generation),
        );
    }
    owners.record(
        RawEventId::from("raw-b"),
        CapsuleId::from("d"),
        Generation(1),
    );
    owners
}

fn line_citation(raw_id: &str, body: &str, line: &str) -> SourceCitation {
    let start = body.find(line).expect("the line is in the body");
    SourceCitation {
        raw_id: RawEventId::from(raw_id),
        start,
        end: start + line.len(),
        quote: line.to_owned(),
    }
}

/// A lifecycle authority: each capsule's live generation, with tombstones and disputes.
struct Lifecycle {
    live: BTreeMap<CapsuleId, Generation>,
    revoked: Vec<(CapsuleId, Generation)>,
    disputed: Vec<(CapsuleId, Generation)>,
}

impl Lifecycle {
    fn new(live: &[(&str, u64)]) -> Self {
        Self {
            live: live
                .iter()
                .map(|(capsule, generation)| (CapsuleId::from(*capsule), Generation(*generation)))
                .collect(),
            revoked: Vec::new(),
            disputed: Vec::new(),
        }
    }

    fn validity(&self, capsule: &CapsuleId, generation: Generation) -> Option<Validity> {
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
    hit.mark_possible().expect("a fresh hit becomes possible");
    hit
}

fn resolve(store: &KnowledgeStore, citation: &SourceCitation) -> ResolvedSource {
    ExactSourceResolver::new(store, &owners())
        .resolve(citation)
        .expect("the citation resolves")
}

fn proof_for(source: &ResolvedSource, claim: (&str, &str, &str)) -> ClaimProof {
    verify_claim(&Claim::new(claim.0, claim.1, claim.2), source).expect("the claim verifies")
}

fn observed(store: &KnowledgeStore, lifecycle: &Lifecycle, line: &str) -> SearchHit {
    let mut hit = possible("c", 2);
    hit.observe_resolved(
        resolve(store, &line_citation("raw-a", BODY, line)),
        |c, g| lifecycle.validity(c, g),
    )
    .expect("a live hit with a resolved source is observed");
    hit
}

#[test]
fn a_cited_claimed_and_live_hit_is_promoted_through_every_stage() {
    let store = store();
    let lifecycle = Lifecycle::new(&[("c", 2)]);
    let mut hit = SearchHit::new(CapsuleId::from("c"), Generation(2), 0.7, "lexical");
    assert_eq!(hit.stage(), EvidenceStage::SearchCandidate);
    hit.mark_possible().expect("candidate to possible");

    let line = "ent-0001: status is active";
    let source = resolve(&store, &line_citation("raw-a", BODY, line));
    hit.observe_resolved(source.clone(), |c, g| lifecycle.validity(c, g))
        .expect("observed");
    assert_eq!(hit.stage(), EvidenceStage::Observed);
    assert_eq!(hit.resolved_source(), Some(&source));
    assert_eq!(hit.verified_claim(), None);

    let proof = proof_for(&source, ("ent-0001", "status", "active"));
    hit.promote_verified_with(proof).expect("promoted");
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
    assert_eq!(
        hit.verified_claim(),
        Some(&Claim::new("ent-0001", "status", "active"))
    );
    // The capsule and generation the lifecycle answered for are the ones promoted.
    assert_eq!(hit.capsule(), &CapsuleId::from("c"));
    assert_eq!(hit.generation(), Generation(2));
}

#[test]
fn every_resolver_refusal_is_a_distinct_typed_error() {
    let store = store();
    let owners = owners();
    let resolver = ExactSourceResolver::new(&store, &owners);
    let line = "ent-0003: color is red";
    let good = line_citation("raw-a", BODY, line);
    assert!(resolver.resolve(&good).is_ok());

    let unknown = SourceCitation {
        raw_id: RawEventId::from("raw-zz"),
        ..good.clone()
    };
    assert_eq!(
        resolver.resolve(&unknown),
        Err(SourceError::UnknownRawEvent {
            raw_id: RawEventId::from("raw-zz")
        })
    );
    let empty = SourceCitation {
        end: good.start,
        ..good.clone()
    };
    assert!(matches!(
        resolver.resolve(&empty),
        Err(SourceError::EmptySpan { .. })
    ));
    let beyond = SourceCitation {
        end: BODY.len() + 3,
        ..good.clone()
    };
    assert!(matches!(
        resolver.resolve(&beyond),
        Err(SourceError::SpanOutOfRange { .. })
    ));
    let shifted = SourceCitation {
        start: good.start - 1,
        end: good.end - 1,
        ..good.clone()
    };
    assert!(matches!(
        resolver.resolve(&shifted),
        Err(SourceError::QuoteMismatch { .. })
    ));
    let doctored = SourceCitation {
        quote: "ent-0003: color is blue".to_owned(),
        ..good
    };
    assert!(matches!(
        resolver.resolve(&doctored),
        Err(SourceError::QuoteMismatch { .. })
    ));
}

#[test]
fn a_source_that_breaks_the_stores_digest_invariant_is_refused() {
    struct Corrupt(RawEvent);
    impl RawEventSource for Corrupt {
        fn raw_event(&self, id: &RawEventId) -> Option<&RawEvent> {
            (id == &self.0.id).then_some(&self.0)
        }
    }
    let mut event = RawEvent::new(
        RawEventId::from("raw-c"),
        SessionId::from("s"),
        EventRole::User,
        b"ent-0001: status is active".to_vec(),
        Timestamp(1),
    );
    // The bytes change, the digest does not.
    event.content[20..26].copy_from_slice(b"idle  ");
    let source = Corrupt(event);
    let line = String::from_utf8(source.0.content.clone()).expect("ascii");
    let citation = SourceCitation {
        raw_id: RawEventId::from("raw-c"),
        start: 0,
        end: line.len(),
        quote: line,
    };
    assert_eq!(
        ExactSourceResolver::new(&source, &owners()).resolve(&citation),
        Err(SourceError::DigestMismatch {
            raw_id: RawEventId::from("raw-c")
        })
    );
}

#[test]
fn every_claim_refusal_is_a_distinct_typed_error() {
    let store = store();
    let negated = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0002: tier is not gold"),
    );
    let affirmative = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0001: status is active"),
    );
    let two_lines = resolve(
        &store,
        &SourceCitation {
            raw_id: RawEventId::from("raw-a"),
            start: 0,
            end: 53,
            quote: BODY[..53].to_owned(),
        },
    );
    let cases = [
        (
            &affirmative,
            Claim::new("ent-0001", "status", "active"),
            None,
        ),
        (
            &affirmative,
            Claim::new("ent-O001", "status", "active"),
            Some(ClaimError::SubjectMismatch),
        ),
        (
            &affirmative,
            Claim::new("ent-0001", "tier", "active"),
            Some(ClaimError::PredicateMismatch),
        ),
        (
            &affirmative,
            Claim::new("ent-0001", "status", "idle"),
            Some(ClaimError::ValueMismatch),
        ),
        (
            &negated,
            Claim::new("ent-0002", "tier", "gold"),
            Some(ClaimError::NegatedStatement),
        ),
        (
            &two_lines,
            Claim::new("ent-0001", "status", "active"),
            Some(ClaimError::MalformedStatement),
        ),
        (
            &affirmative,
            Claim::new("ent-0001", "", "active"),
            Some(ClaimError::MalformedClaim),
        ),
    ];
    for (source, claim, expected) in cases {
        let outcome = verify_claim(&claim, source);
        assert_eq!(outcome.err(), expected, "{claim:?}");
    }
}

#[test]
fn a_hit_cannot_skip_a_stage_through_the_typed_path() {
    let store = store();
    let lifecycle = Lifecycle::new(&[("c", 2)]);
    let line = "ent-0001: status is active";
    let source = resolve(&store, &line_citation("raw-a", BODY, line));
    let proof = proof_for(&source, ("ent-0001", "status", "active"));

    // Straight from a retrieval candidate.
    let mut candidate = SearchHit::new(CapsuleId::from("c"), Generation(2), 0.9, "test");
    assert_eq!(
        candidate.promote_verified_with(proof.clone()),
        Err(EvidencePromotionError::WrongStage)
    );
    assert_eq!(
        candidate.observe_resolved(source.clone(), |_, _| unreachable!(
            "asked before the stage"
        )),
        Err(EvidencePromotionError::WrongStage)
    );
    assert_eq!(candidate.stage(), EvidenceStage::SearchCandidate);
    assert_eq!(candidate.resolved_source(), None);

    // From possible evidence, without observation.
    let mut hit = possible("c", 2);
    assert_eq!(
        hit.promote_verified_with(proof.clone()),
        Err(EvidencePromotionError::WrongStage)
    );
    assert_eq!(hit.stage(), EvidenceStage::PossibleEvidence);

    // Observation cannot be repeated, nor can the hit step back.
    hit.observe_resolved(source.clone(), |c, g| lifecycle.validity(c, g))
        .expect("observed");
    assert_eq!(
        hit.observe_resolved(source.clone(), |_, _| unreachable!(
            "asked in the wrong stage"
        )),
        Err(EvidencePromotionError::WrongStage)
    );
    assert_eq!(hit.mark_possible(), Err(EvidencePromotionError::WrongStage));
    hit.promote_verified_with(proof.clone()).expect("promoted");
    assert_eq!(
        hit.promote_verified_with(proof),
        Err(EvidencePromotionError::WrongStage)
    );
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
}

#[test]
fn a_lifecycle_refusal_keeps_the_hit_possible_and_keeps_no_source() {
    let store = store();
    let source = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0001: status is active"),
    );
    let mut lifecycle = Lifecycle::new(&[("c", 3), ("d", 1), ("e", 1)]);
    lifecycle
        .revoked
        .push((CapsuleId::from("d"), Generation(1)));
    lifecycle
        .disputed
        .push((CapsuleId::from("e"), Generation(1)));
    let cases = [
        ("c", 2, EvidencePromotionError::StaleGeneration),
        ("d", 1, EvidencePromotionError::RevokedGeneration),
        ("e", 1, EvidencePromotionError::DisputedGeneration),
        ("c", 4, EvidencePromotionError::UnknownGeneration),
        ("zz", 1, EvidencePromotionError::UnknownGeneration),
    ];
    for (capsule, generation, expected) in cases {
        let mut hit = possible(capsule, generation);
        assert_eq!(
            hit.observe_resolved(source.clone(), |c, g| lifecycle.validity(c, g)),
            Err(expected),
            "{capsule}@{generation}"
        );
        assert_eq!(hit.stage(), EvidenceStage::PossibleEvidence);
        assert_eq!(hit.resolved_source(), None);
        // The refused hit cannot be promoted by replaying a proof either.
        let proof = proof_for(&source, ("ent-0001", "status", "active"));
        assert_eq!(
            hit.promote_verified_with(proof),
            Err(EvidencePromotionError::WrongStage)
        );
    }
    let mut live = possible("c", 3);
    live.observe_resolved(source, |c, g| lifecycle.validity(c, g))
        .expect("the live generation is observed");
}

#[test]
fn a_proof_made_for_another_source_is_refused() {
    let store = store();
    let lifecycle = Lifecycle::new(&[("c", 2)]);
    let mut hit = observed(&store, &lifecycle, "ent-0001: status is active");

    let other_line = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0003: color is red"),
    );
    let other_event = resolve(
        &store,
        &line_citation(
            "raw-b",
            "ent-0009: status is idle",
            "ent-0009: status is idle",
        ),
    );
    for foreign in [
        proof_for(&other_line, ("ent-0003", "color", "red")),
        proof_for(&other_event, ("ent-0009", "status", "idle")),
    ] {
        assert_eq!(
            hit.promote_verified_with(foreign),
            Err(EvidencePromotionError::ClaimProofMismatch)
        );
        assert_eq!(hit.stage(), EvidenceStage::Observed);
        assert_eq!(hit.verified_claim(), None);
    }
    // The hit's own source still works afterwards.
    let own = hit
        .resolved_source()
        .expect("observed with a source")
        .clone();
    hit.promote_verified_with(proof_for(&own, ("ent-0001", "status", "active")))
        .expect("promoted with its own proof");
}

#[test]
fn a_hit_observed_through_the_boolean_path_holds_no_source_to_verify_against() {
    let store = store();
    let lifecycle = Lifecycle::new(&[("c", 2)]);
    let source = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0001: status is active"),
    );
    let proof = proof_for(&source, ("ent-0001", "status", "active"));
    let mut hit = possible("c", 2);
    hit.observe(|c, g| lifecycle.validity(c, g), true)
        .expect("trusted-caller observation");
    assert_eq!(hit.resolved_source(), None);
    assert_eq!(
        hit.promote_verified_with(proof),
        Err(EvidencePromotionError::SourceNotResolved)
    );
    assert_eq!(hit.stage(), EvidenceStage::Observed);
}

#[test]
fn the_boolean_path_stays_the_trusted_callers_and_leaves_no_evidence() {
    let lifecycle = Lifecycle::new(&[("c", 2)]);
    let mut hit = possible("c", 2);
    hit.observe(|c, g| lifecycle.validity(c, g), true)
        .expect("observed on the caller's word");
    hit.promote_verified(true)
        .expect("promoted on the caller's word");
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
    assert_eq!(hit.resolved_source(), None);
    assert_eq!(hit.verified_claim(), None);
}

#[test]
fn the_new_refusals_have_their_own_stable_codes() {
    assert_eq!(
        EvidencePromotionError::ClaimProofMismatch.code(),
        "PTR_SEARCH_CLAIM_PROOF_MISMATCH"
    );
    assert_eq!(
        EvidencePromotionError::SourceNotOwnedByHit.code(),
        "PTR_SEARCH_SOURCE_NOT_OWNED_BY_HIT"
    );
    for error in [
        EvidencePromotionError::ClaimProofMismatch,
        EvidencePromotionError::SourceNotOwnedByHit,
    ] {
        assert!(!error.to_string().is_empty());
    }
}

/// A hit that names one capsule generation must be observed with an event of that very
/// generation: an intact, exact, live-labelled citation of another capsule's event, of the
/// same capsule's other generation or of a revoked capsule's event is refused, whatever the
/// lifecycle says about the hit's own label (here it says live for every label, so only the
/// ownership can refuse).
#[test]
fn a_source_of_another_capsule_or_generation_is_not_observed() {
    let store = store();
    let live = |_: &CapsuleId, _: Generation| Some(Validity::Live);
    let a = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0001: status is active"),
    );
    let b = resolve(
        &store,
        &line_citation(
            "raw-b",
            "ent-0009: status is idle",
            "ent-0009: status is idle",
        ),
    );
    // Another capsule with a generation that matches; the right capsule with another
    // generation; neither matches.
    for (capsule, generation, source) in [
        ("d", 2, &a),
        ("d", 3, &a),
        ("c", 1, &a),
        ("c", 4, &a),
        ("c", 1, &b),
        ("c", 2, &b),
        ("d", 2, &b),
        ("zz", 1, &b),
    ] {
        let mut hit = possible(capsule, generation);
        assert_eq!(
            hit.observe_resolved(source.clone(), live),
            Err(EvidencePromotionError::SourceNotOwnedByHit),
            "{capsule}@{generation}"
        );
        assert_eq!(hit.stage(), EvidenceStage::PossibleEvidence);
        assert_eq!(hit.resolved_source(), None);
    }
    // A label that does own the event observes it, and promotes.
    for (capsule, generation, source) in [("c", 3, &a), ("c", 2, &a), ("d", 1, &b)] {
        let mut hit = possible(capsule, generation);
        hit.observe_resolved(source.clone(), live)
            .unwrap_or_else(|error| panic!("{capsule}@{generation}: {error}"));
        assert_eq!(hit.stage(), EvidenceStage::Observed);
    }
}

#[test]
fn the_lifecycle_refusal_takes_precedence_over_the_ownership_refusal() {
    let store = store();
    let mut lifecycle = Lifecycle::new(&[("c", 3), ("d", 1)]);
    lifecycle
        .revoked
        .push((CapsuleId::from("d"), Generation(1)));
    // raw-a belongs to c, not to d@1 or c@2; the lifecycle's answer still comes first.
    let a = resolve(
        &store,
        &line_citation("raw-a", BODY, "ent-0001: status is active"),
    );
    let mut revoked = possible("d", 1);
    assert_eq!(
        revoked.observe_resolved(a.clone(), |c, g| lifecycle.validity(c, g)),
        Err(EvidencePromotionError::RevokedGeneration)
    );
    let mut owner_c2 = possible("c", 2);
    assert_eq!(
        owner_c2.observe_resolved(a, |c, g| lifecycle.validity(c, g)),
        Err(EvidencePromotionError::StaleGeneration)
    );
}

#[test]
fn an_event_nothing_admitted_is_resolved_but_never_observed() {
    let store = store();
    let lifecycle = Lifecycle::new(&[("c", 2)]);
    let nobody = SourceOwners::default();
    let orphan = ExactSourceResolver::new(&store, &nobody)
        .resolve(&line_citation("raw-a", BODY, "ent-0001: status is active"))
        .expect("an intact event resolves without an owner");
    assert!(orphan.owners().is_empty());
    let mut hit = possible("c", 2);
    assert_eq!(
        hit.observe_resolved(orphan, |c, g| lifecycle.validity(c, g)),
        Err(EvidencePromotionError::SourceNotOwnedByHit)
    );
    assert_eq!(hit.stage(), EvidenceStage::PossibleEvidence);
}

#[test]
fn a_truncated_negation_does_not_verify_the_bare_value_not() {
    let store = store();
    // The span stops after `not`: the bytes are a prefix of a negated statement.
    let prefix = "ent-0002: tier is not";
    let source = resolve(&store, &line_citation("raw-a", BODY, prefix));
    for value in ["not", "gold"] {
        assert_eq!(
            verify_claim(&Claim::new("ent-0002", "tier", value), &source).err(),
            Some(ClaimError::MalformedStatement),
            "{value}"
        );
    }
}
