//! Who may see a document and when it holds, as a backend-neutral predicate, and the check a
//! hit goes through again at the moment it is relied on.
use ptr_search::{
    EvidencePromotionError, EvidenceStage, FilterError, SearchAttributes, SearchFilter, SearchHit,
    TimedPromotionError, ValidityRefusal, MAX_FILTER_PRINCIPALS, MAX_FILTER_TAGS,
    MAX_FILTER_TEXT_BYTES,
};
use ptr_types::{CapsuleId, Generation, KnowledgeValidity, PrincipalId, Timestamp, Validity};
use std::collections::BTreeSet;

fn set(members: &[&str]) -> BTreeSet<String> {
    members.iter().map(|member| (*member).to_owned()).collect()
}

fn between(from: u64, until: u64) -> KnowledgeValidity {
    KnowledgeValidity {
        valid_from: Some(Timestamp(from)),
        valid_until: Some(Timestamp(until)),
    }
}

fn restricted_to(members: &[&str]) -> SearchAttributes {
    SearchAttributes {
        principals: set(members),
        ..SearchAttributes::default()
    }
}

fn asking(principal: &str, groups: &[&str], now: u64) -> SearchFilter {
    SearchFilter::for_principal(
        PrincipalId::from(principal),
        groups.iter().map(|group| (*group).to_owned()),
        Timestamp(now),
    )
}

#[test]
fn a_filter_that_names_nobody_sees_only_project_wide_documents() {
    let anonymous = SearchFilter::audit_without_time();
    assert!(anonymous.admits(&SearchAttributes::default()));
    assert!(!anonymous.admits(&restricted_to(&["alice"])));
    assert!(
        !anonymous.admits(&restricted_to(&["*"])),
        "a literal wildcard is a name, not a way in"
    );
}

#[test]
fn another_principal_sees_nothing_of_a_restricted_document() {
    let document = restricted_to(&["alice"]);
    assert!(asking("alice", &[], 5).admits(&document));
    assert!(!asking("mallory", &[], 5).admits(&document));
    assert!(!asking("mallory", &["staff"], 5).admits(&document));
}

#[test]
fn a_group_admits_its_members_and_either_list_member_is_enough() {
    let document = restricted_to(&["alice", "staff"]);
    assert!(asking("bob", &["staff"], 5).admits(&document));
    assert!(asking("alice", &[], 5).admits(&document));
    assert!(!asking("bob", &["guests"], 5).admits(&document));
    // Project-wide documents stay visible to everyone.
    assert!(asking("bob", &[], 5).admits(&SearchAttributes::default()));
}

#[test]
fn validity_is_half_open_with_open_bounds_when_missing() {
    let document = SearchAttributes {
        validity: between(10, 20),
        ..SearchAttributes::default()
    };
    let at = |now: u64| asking("alice", &[], now).admits(&document);
    assert!(!at(9));
    assert!(at(10), "valid_from is inside");
    assert!(at(19));
    assert!(!at(20), "valid_until is outside");
    assert!(!at(u64::MAX));

    let open_ended = SearchAttributes {
        validity: KnowledgeValidity {
            valid_from: Some(Timestamp(10)),
            valid_until: None,
        },
        ..SearchAttributes::default()
    };
    assert!(!asking("a", &[], 9).admits(&open_ended));
    assert!(asking("a", &[], u64::MAX).admits(&open_ended));

    let open_started = SearchAttributes {
        validity: KnowledgeValidity {
            valid_from: None,
            valid_until: Some(Timestamp(10)),
        },
        ..SearchAttributes::default()
    };
    assert!(asking("a", &[], 0).admits(&open_started));
    assert!(!asking("a", &[], 10).admits(&open_started));
}

#[test]
fn consecutive_validities_that_meet_never_overlap_and_never_leave_a_gap() {
    let first = between(10, 20);
    let second = between(20, 30);
    for now in 10..30 {
        let both = first.is_valid_at(Timestamp(now)) && second.is_valid_at(Timestamp(now));
        let either = first.is_valid_at(Timestamp(now)) || second.is_valid_at(Timestamp(now));
        assert!(!both, "{now}");
        assert!(either, "{now}");
    }
}

#[test]
fn a_filter_without_a_now_applies_no_time_rule() {
    let expired = SearchAttributes {
        validity: between(1, 2),
        ..SearchAttributes::default()
    };
    let audit_view = SearchFilter {
        principal: Some(PrincipalId::from("auditor")),
        now: None,
        ..SearchFilter::audit_without_time()
    };
    assert!(audit_view.admits(&expired));
    assert!(!asking("auditor", &[], 5).admits(&expired));
}

#[test]
fn role_predicate_and_tags_narrow_what_is_returned() {
    let document = SearchAttributes {
        semantic_role: Some("FACT".into()),
        predicate: Some("lead_time".into()),
        tags: set(&["supplier", "eu"]),
        ..SearchAttributes::default()
    };
    let with = |edit: &dyn Fn(&mut SearchFilter)| {
        let mut filter = asking("alice", &[], 5);
        edit(&mut filter);
        filter.admits(&document)
    };
    assert!(with(&|_| {}));
    assert!(with(&|f| f.semantic_role = Some("FACT".into())));
    assert!(!with(&|f| f.semantic_role = Some("RULE".into())));
    assert!(with(&|f| f.predicate = Some("lead_time".into())));
    assert!(!with(&|f| f.predicate = Some("deadline".into())));
    assert!(
        with(&|f| f.tags_any = set(&["eu", "us"])),
        "any one tag is enough"
    );
    assert!(!with(&|f| f.tags_any = set(&["us", "asia"])));
    // A document that states no role or predicate cannot satisfy a filter that asks for one.
    let mut asks_for_a_role = asking("alice", &[], 5);
    asks_for_a_role.semantic_role = Some("FACT".into());
    assert!(!asks_for_a_role.admits(&SearchAttributes::default()));
}

#[test]
fn a_filter_and_attributes_are_validated_before_a_backend_translates_them() {
    assert_eq!(asking("alice", &["staff"], 5).validate(), Ok(()));
    assert_eq!(SearchFilter::audit_without_time().validate(), Ok(()));

    let empty_principal = SearchFilter {
        principal: Some(PrincipalId::from("")),
        ..SearchFilter::audit_without_time()
    };
    assert_eq!(
        empty_principal.validate(),
        Err(FilterError::Empty { field: "principal" })
    );

    let long = "g".repeat(MAX_FILTER_TEXT_BYTES + 1);
    let long_group = SearchFilter {
        groups: set(&[long.as_str()]),
        ..SearchFilter::audit_without_time()
    };
    assert_eq!(
        long_group.validate(),
        Err(FilterError::TooLong {
            field: "groups",
            limit: MAX_FILTER_TEXT_BYTES,
            actual: MAX_FILTER_TEXT_BYTES + 1
        })
    );

    let control = SearchFilter {
        tags_any: set(&["a\u{0}b"]),
        ..SearchFilter::audit_without_time()
    };
    assert_eq!(
        control.validate(),
        Err(FilterError::ControlCharacter { field: "tags_any" })
    );

    let many: BTreeSet<String> = (0..=MAX_FILTER_PRINCIPALS)
        .map(|n| format!("g{n}"))
        .collect();
    let crowded = SearchFilter {
        groups: many.clone(),
        ..SearchFilter::audit_without_time()
    };
    assert!(matches!(
        crowded.validate(),
        Err(FilterError::TooMany {
            field: "groups",
            ..
        })
    ));
    // The principal counts toward the audience bound too.
    let full: BTreeSet<String> = (0..MAX_FILTER_PRINCIPALS)
        .map(|n| format!("g{n}"))
        .collect();
    let overfull = SearchFilter {
        principal: Some(PrincipalId::from("alice")),
        groups: full,
        ..SearchFilter::audit_without_time()
    };
    assert!(matches!(
        overfull.validate(),
        Err(FilterError::TooMany {
            field: "audience",
            ..
        })
    ));

    let tags: BTreeSet<String> = (0..=MAX_FILTER_TAGS).map(|n| format!("t{n}")).collect();
    let tagged = SearchAttributes {
        tags,
        ..SearchAttributes::default()
    };
    assert!(matches!(
        tagged.validate(),
        Err(FilterError::TooMany { field: "tags", .. })
    ));

    let never = SearchAttributes {
        validity: between(20, 20),
        ..SearchAttributes::default()
    };
    assert_eq!(never.validate(), Err(FilterError::EmptyValidity));
    assert_eq!(SearchAttributes::default().validate(), Ok(()));
}

#[test]
fn every_filter_refusal_has_its_own_stable_code() {
    let codes: BTreeSet<&str> = [
        FilterError::Empty { field: "x" },
        FilterError::TooLong {
            field: "x",
            limit: 1,
            actual: 2,
        },
        FilterError::TooMany {
            field: "x",
            limit: 1,
            actual: 2,
        },
        FilterError::ControlCharacter { field: "x" },
        FilterError::EmptyValidity,
    ]
    .iter()
    .map(FilterError::code)
    .collect();
    assert_eq!(codes.len(), 5);
}

fn live(_: &CapsuleId, _: Generation) -> Option<Validity> {
    Some(Validity::Live)
}

fn possible() -> SearchHit {
    let mut hit = SearchHit::new(CapsuleId::from("capsule"), Generation(1), 0.5, "test");
    hit.mark_possible().unwrap();
    hit
}

#[test]
fn a_hit_is_held_to_the_authoritys_validity_when_it_is_observed() {
    let observe = |held: Option<KnowledgeValidity>, now: u64| {
        let mut hit = possible();
        let result = hit.observe_at(live, |_, _| held, Timestamp(now), true);
        (result, hit.stage())
    };
    assert_eq!(
        observe(Some(between(10, 20)), 15),
        (Ok(()), EvidenceStage::Observed)
    );
    assert_eq!(
        observe(Some(between(10, 20)), 9),
        (
            Err(TimedPromotionError::Validity(ValidityRefusal::NotYetValid)),
            EvidenceStage::PossibleEvidence
        )
    );
    assert_eq!(
        observe(Some(between(10, 20)), 20),
        (
            Err(TimedPromotionError::Validity(ValidityRefusal::Expired)),
            EvidenceStage::PossibleEvidence
        ),
        "the end is outside"
    );
    assert_eq!(
        observe(None, 15),
        (
            Err(TimedPromotionError::Validity(ValidityRefusal::Unknown)),
            EvidenceStage::PossibleEvidence
        ),
        "an authority that knows no validity is not asked to be trusted"
    );
}

#[test]
fn the_lifecycle_is_judged_before_the_time_and_the_time_before_the_source() {
    let mut hit = possible();
    let revoked = hit.observe_at(
        |_, _| Some(Validity::Revoked),
        |_, _| Some(between(10, 20)),
        Timestamp(99),
        false,
    );
    assert_eq!(
        revoked,
        Err(TimedPromotionError::Promotion(
            EvidencePromotionError::RevokedGeneration
        ))
    );

    let mut hit = possible();
    let expired = hit.observe_at(live, |_, _| Some(between(10, 20)), Timestamp(99), false);
    assert_eq!(
        expired,
        Err(TimedPromotionError::Validity(ValidityRefusal::Expired))
    );

    let mut hit = possible();
    let unresolved = hit.observe_at(live, |_, _| Some(between(10, 20)), Timestamp(15), false);
    assert_eq!(
        unresolved,
        Err(TimedPromotionError::Promotion(
            EvidencePromotionError::SourceNotResolved
        ))
    );
}

#[test]
fn a_document_that_expires_between_retrieval_and_use_is_refused_at_commit_time() {
    let mut hit = possible();
    hit.observe_at(live, |_, _| Some(between(10, 20)), Timestamp(19), true)
        .unwrap();
    // The inference ran long: by the time the hit is relied on, its validity has ended.
    assert_eq!(
        hit.promote_verified_at(true, live, |_, _| Some(between(10, 20)), Timestamp(20)),
        Err(TimedPromotionError::Validity(ValidityRefusal::Expired))
    );
    assert_eq!(
        hit.stage(),
        EvidenceStage::Observed,
        "a refusal keeps the stage"
    );
    assert_eq!(
        hit.promote_verified_at(true, live, |_, _| Some(between(10, 20)), Timestamp(19)),
        Ok(())
    );
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
}

#[test]
fn the_authority_is_asked_again_at_commit_time_not_only_at_observation() {
    let observed = || {
        let mut hit = possible();
        hit.observe_at(live, |_, _| Some(between(10, 30)), Timestamp(15), true)
            .unwrap();
        hit
    };
    // Revoked after the hit was observed.
    let mut hit = observed();
    assert_eq!(
        hit.promote_verified_at(
            true,
            |_, _| Some(Validity::Revoked),
            |_, _| Some(between(10, 30)),
            Timestamp(16)
        ),
        Err(TimedPromotionError::Promotion(
            EvidencePromotionError::RevokedGeneration
        ))
    );
    // The authority shortened the validity after the hit was observed.
    let mut hit = observed();
    assert_eq!(
        hit.promote_verified_at(true, live, |_, _| Some(between(10, 16)), Timestamp(16)),
        Err(TimedPromotionError::Validity(ValidityRefusal::Expired))
    );
    // The authority no longer holds a validity at all.
    let mut hit = observed();
    assert_eq!(
        hit.promote_verified_at(true, live, |_, _| None, Timestamp(16)),
        Err(TimedPromotionError::Validity(ValidityRefusal::Unknown))
    );
    assert_eq!(hit.stage(), EvidenceStage::Observed);
    // Unchanged authority: promoted.
    let mut hit = observed();
    assert_eq!(
        hit.promote_verified_at(true, live, |_, _| Some(between(10, 30)), Timestamp(16)),
        Ok(())
    );
}

#[test]
fn a_filter_built_for_an_instant_applies_the_time_rule() {
    let expired = SearchAttributes {
        validity: between(1, 2),
        ..SearchAttributes::default()
    };
    assert!(!SearchFilter::at(Timestamp(5)).admits(&expired));
    assert!(SearchFilter::audit_without_time().admits(&expired));
    assert!(SearchFilter::at(Timestamp(5)).admits(&SearchAttributes::default()));
}

#[test]
fn a_failed_verification_is_still_refused_after_the_time_check() {
    let mut hit = possible();
    hit.observe_at(
        live,
        |_, _| Some(KnowledgeValidity::ALWAYS),
        Timestamp(1),
        true,
    )
    .unwrap();
    assert_eq!(
        hit.promote_verified_at(
            false,
            live,
            |_, _| Some(KnowledgeValidity::ALWAYS),
            Timestamp(2)
        ),
        Err(TimedPromotionError::Promotion(
            EvidencePromotionError::VerificationFailed
        ))
    );
}

#[test]
fn the_untimed_promotion_cannot_be_used_to_skip_the_commit_time_check() {
    let mut hit = possible();
    hit.observe_at(live, |_, _| Some(between(10, 20)), Timestamp(15), true)
        .unwrap();
    assert_eq!(
        hit.promote_verified(true),
        Err(EvidencePromotionError::WrongStage)
    );
    assert_eq!(hit.stage(), EvidenceStage::Observed);
}

#[test]
fn a_hit_observed_without_a_validity_cannot_claim_a_time_check() {
    let mut hit = possible();
    hit.observe(live, true).unwrap();
    assert_eq!(
        hit.promote_verified_at(
            true,
            live,
            |_, _| Some(KnowledgeValidity::ALWAYS),
            Timestamp(1)
        ),
        Err(TimedPromotionError::Validity(ValidityRefusal::Unknown))
    );
    // The trusted-caller path is unchanged for a hit that never had a validity.
    assert_eq!(hit.promote_verified(true), Ok(()));
}

#[test]
fn the_new_refusals_have_stable_distinct_codes() {
    let codes: BTreeSet<&str> = [
        ValidityRefusal::NotYetValid,
        ValidityRefusal::Expired,
        ValidityRefusal::Unknown,
    ]
    .iter()
    .map(ValidityRefusal::code)
    .collect();
    assert_eq!(codes.len(), 3);
    assert!(codes.iter().all(|code| code.starts_with("PTR_SEARCH_")));
    assert_eq!(
        TimedPromotionError::Validity(ValidityRefusal::Expired).code(),
        "PTR_SEARCH_EXPIRED"
    );
    assert_eq!(
        TimedPromotionError::Promotion(EvidencePromotionError::WrongStage).code(),
        EvidencePromotionError::WrongStage.code()
    );
}
