mod common;

use common::estimate;
use ptr_types::{
    ConfidenceEstimate, ConfidenceTarget, ConfidenceTargetMismatch, EpistemicState, Generation,
    Probability, ProvenanceRef, SemanticRole, TypeId, TypedValue, UncertaintyKind, Validity,
};

// This fixture is deliberately not a production semantic wrapper or tensor schema.
// Step 1 specifies independent meanings without freezing Goal<T>/Claim<T> layouts.
#[derive(Clone, Debug, PartialEq)]
struct Annotation {
    role: SemanticRole,
    value_type: TypeId,
    epistemic: EpistemicState,
    uncertainty: Option<UncertaintyKind>,
    confidence: Option<ConfidenceEstimate>,
}

#[test]
fn role_epistemic_and_uncertainty_axes_are_independent() {
    let roles = [
        SemanticRole::Goal,
        SemanticRole::Constraint,
        SemanticRole::Claim,
        SemanticRole::Evidence,
        SemanticRole::Resource,
        SemanticRole::Capability,
        SemanticRole::Relation,
        SemanticRole::Procedure,
        SemanticRole::Action,
    ];
    let states = [
        EpistemicState::Unknown,
        EpistemicState::Assumed,
        EpistemicState::Hypothesis,
        EpistemicState::Observed,
        EpistemicState::Inferred,
        EpistemicState::Verified,
    ];
    let shapes = [
        UncertaintyKind::Point,
        UncertaintyKind::Interval,
        UncertaintyKind::Distribution,
    ];
    let mut cases = 0;
    for role in roles {
        for state in states {
            for shape in shapes {
                let annotation = Annotation {
                    role,
                    value_type: TypeId::from("fixture:scalar"),
                    epistemic: state,
                    uncertainty: Some(shape),
                    confidence: None,
                };
                assert_eq!(annotation.role, role);
                assert_eq!(annotation.epistemic, state);
                assert_eq!(annotation.uncertainty, Some(shape));
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 162);
}

#[test]
fn hard_constraint_role_confidence_is_not_requirement_strength() {
    let mut annotation = Annotation {
        role: SemanticRole::Constraint,
        value_type: TypeId::from("fixture:upper-bound"),
        epistemic: EpistemicState::Observed,
        uncertainty: Some(UncertaintyKind::Point),
        confidence: Some(estimate(
            ConfidenceTarget::SemanticRole(SemanticRole::Constraint),
            0.95,
        )),
    };
    annotation.confidence = Some(estimate(
        ConfidenceTarget::SemanticRole(SemanticRole::Constraint),
        0.2,
    ));
    assert_eq!(annotation.role, SemanticRole::Constraint);
    assert_eq!(annotation.epistemic, EpistemicState::Observed);
    // Whether a constraint is mandatory belongs to its separate semantic contract.
}

#[test]
fn uncertain_claim_keeps_value_type_separate_from_semantic_role() {
    let mut annotation = Annotation {
        role: SemanticRole::Claim,
        value_type: TypeId::from("fixture:time-point"),
        epistemic: EpistemicState::Hypothesis,
        uncertainty: Some(UncertaintyKind::Interval),
        confidence: Some(estimate(ConfidenceTarget::Proposition, 0.6)),
    };
    let value_type = annotation.value_type.clone();
    annotation.epistemic = EpistemicState::Observed;
    assert_eq!(annotation.value_type, value_type);
    assert_eq!(annotation.role, SemanticRole::Claim);
    assert_eq!(annotation.uncertainty, Some(UncertaintyKind::Interval));
}

#[test]
fn missing_confidence_is_not_a_half_probability() {
    let missing: Option<ConfidenceEstimate> = None;
    let half = Some(estimate(ConfidenceTarget::Proposition, 0.5));
    assert_ne!(missing, half);
}

#[test]
fn contradictory_sources_keep_their_own_evidence_and_estimates() {
    let source = |id: &str, probability| TypedValue {
        value: estimate(ConfidenceTarget::Proposition, probability),
        generation: Generation(1),
        validity: Validity::Disputed,
        provenance: vec![ProvenanceRef {
            source: id.into(),
            note: None,
        }],
    };
    let sources = [source("source:a", 0.9), source("source:b", 0.1)];
    assert_ne!(sources[0].provenance, sources[1].provenance);
    assert_ne!(sources[0].value, sources[1].value);
    assert!(sources
        .iter()
        .all(|item| item.validity == Validity::Disputed));
    // This is a representation test, not an implemented conflict resolver.
}

#[test]
fn revoked_generation_is_not_relabelled_by_full_confidence() {
    let mut previous = TypedValue {
        value: estimate(ConfidenceTarget::Proposition, 1.0),
        generation: Generation(4),
        validity: Validity::Live,
        provenance: vec![],
    };
    previous.validity = Validity::Revoked;
    let mut replacement = previous.clone();
    replacement.generation = Generation(5);
    replacement.validity = Validity::Live;
    assert_eq!(previous.validity, Validity::Revoked);
    assert_eq!(previous.value.probability().get(), 1.0);
    assert_ne!(previous.generation, replacement.generation);
    // Live-authority admission remains a runtime check, not this fixture's claim.
}

#[test]
fn role_confidence_cannot_be_read_as_proposition_confidence() {
    let role = ConfidenceTarget::SemanticRole(SemanticRole::Claim);
    let estimate = estimate(role.clone(), 1.0);
    assert_eq!(
        estimate.probability_for(&role).expect("same target").get(),
        1.0
    );
    let error = estimate
        .probability_for(&ConfidenceTarget::Proposition)
        .expect_err("a full role score is not a truth probability");
    assert_eq!(error.expected, ConfidenceTarget::Proposition);
    assert_eq!(error.actual, role);
    assert!(error.to_string().contains(ConfidenceTargetMismatch::CODE));
}

#[test]
fn different_type_alternatives_are_not_the_same_confidence_target() {
    let time = ConfidenceTarget::ValueType(TypeId::from("fixture:time-point"));
    let duration = ConfidenceTarget::ValueType(TypeId::from("fixture:duration"));
    let estimate = estimate(time.clone(), 0.9);
    assert!(estimate.probability_for(&time).is_ok());
    assert_eq!(
        estimate.probability_for(&duration),
        Err(ConfidenceTargetMismatch {
            expected: duration,
            actual: time,
        })
    );
}

#[test]
fn probability_rejects_nonfinite_and_out_of_range_inputs() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
        assert!(Probability::new(value).is_none());
    }
    for value in [0.0, 0.5, 1.0] {
        assert_eq!(Probability::new(value).expect("bounded").get(), value);
    }
}
