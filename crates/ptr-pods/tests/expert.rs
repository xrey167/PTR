use ptr_pods::{
    ExpertLevel, ExpertPodSpec, ExpertProposal, ExpertProposalError, ExpertSpecError, ExpertState,
    ExpertTarget, ProposalParts, StructuralOp,
};
use ptr_types::{Generation, PodId};

const RATIONALE: [u8; 32] = [7; 32];

fn target(name: &str, generation: u64, level: ExpertLevel, state: ExpertState) -> ExpertTarget {
    ExpertTarget {
        pod: PodId::from(name),
        generation: Generation(generation),
        level,
        state,
    }
}

fn token(layer: u16) -> ExpertLevel {
    ExpertLevel::Token { layer }
}

fn serving(name: &str) -> ExpertTarget {
    target(name, 1, token(3), ExpertState::Serving)
}

fn propose(
    op: StructuralOp,
    targets: Vec<ExpertTarget>,
    birth_level: Option<ExpertLevel>,
) -> Result<ExpertProposal, ExpertProposalError> {
    ExpertProposal::new(op, targets, birth_level, RATIONALE)
}

#[test]
fn valid_proposal_for_each_operation() {
    let frozen = target("b", 2, token(3), ExpertState::Frozen);
    let retired = target("c", 1, ExpertLevel::Segment, ExpertState::Retired);
    let cases = [
        propose(StructuralOp::Birth, vec![], Some(token(0))),
        propose(StructuralOp::Birth, vec![], Some(ExpertLevel::Segment)),
        propose(
            StructuralOp::Merge,
            vec![serving("a"), frozen.clone()],
            None,
        ),
        propose(StructuralOp::Split, vec![serving("a")], None),
        propose(StructuralOp::Freeze, vec![serving("a")], None),
        propose(StructuralOp::Reactivate, vec![retired], None),
        propose(StructuralOp::Prune, vec![frozen.clone()], None),
        propose(StructuralOp::Retire, vec![serving("a")], None),
    ];
    for case in cases {
        let proposal = case.expect("valid proposal");
        assert_eq!(proposal.rationale_digest(), &RATIONALE);
    }
}

#[test]
fn wrong_target_counts_are_rejected() {
    let count = |r: Result<ExpertProposal, ExpertProposalError>| match r {
        Err(ExpertProposalError::WrongTargetCount { found, .. }) => found,
        other => panic!("expected WrongTargetCount, got {other:?}"),
    };
    assert_eq!(
        count(propose(StructuralOp::Merge, vec![serving("a")], None)),
        1
    );
    assert_eq!(count(propose(StructuralOp::Merge, vec![], None)), 0);
    for op in [
        StructuralOp::Split,
        StructuralOp::Freeze,
        StructuralOp::Prune,
        StructuralOp::Retire,
    ] {
        assert_eq!(count(propose(op, vec![], None)), 0);
        assert_eq!(
            count(propose(op, vec![serving("a"), serving("b")], None)),
            2
        );
    }
    let retired = |n: &str| target(n, 1, token(3), ExpertState::Retired);
    assert_eq!(count(propose(StructuralOp::Reactivate, vec![], None)), 0);
    assert_eq!(
        count(propose(
            StructuralOp::Reactivate,
            vec![retired("a"), retired("b")],
            None
        )),
        2
    );
}

#[test]
fn birth_rules_are_enforced() {
    assert_eq!(
        propose(StructuralOp::Birth, vec![serving("a")], Some(token(1))),
        Err(ExpertProposalError::WrongTargetCount {
            op: StructuralOp::Birth,
            expected: "no",
            found: 1
        })
    );
    assert_eq!(
        propose(StructuralOp::Birth, vec![], None),
        Err(ExpertProposalError::BirthWithoutLevel)
    );
    for op in [
        StructuralOp::Merge,
        StructuralOp::Split,
        StructuralOp::Freeze,
        StructuralOp::Reactivate,
        StructuralOp::Prune,
        StructuralOp::Retire,
    ] {
        assert_eq!(
            propose(op, vec![serving("a"), serving("b")], Some(token(1))),
            Err(ExpertProposalError::UnexpectedBirthLevel { op })
        );
    }
}

#[test]
fn wrong_target_state_is_rejected() {
    let wrong = |r: Result<ExpertProposal, ExpertProposalError>| match r {
        Err(ExpertProposalError::WrongTargetState { state, .. }) => state,
        other => panic!("expected WrongTargetState, got {other:?}"),
    };
    let in_state = |s| target("a", 1, token(3), s);
    assert_eq!(
        wrong(propose(
            StructuralOp::Reactivate,
            vec![in_state(ExpertState::Serving)],
            None
        )),
        ExpertState::Serving
    );
    assert_eq!(
        wrong(propose(
            StructuralOp::Freeze,
            vec![in_state(ExpertState::Frozen)],
            None
        )),
        ExpertState::Frozen
    );
    for op in [StructuralOp::Split, StructuralOp::Prune] {
        for s in [ExpertState::Candidate, ExpertState::Retired] {
            assert_eq!(wrong(propose(op, vec![in_state(s)], None)), s);
        }
    }
    // A candidate that failed its gate can be retired; a retired expert cannot be retired again.
    assert!(propose(
        StructuralOp::Retire,
        vec![in_state(ExpertState::Candidate)],
        None
    )
    .is_ok());
    assert_eq!(
        wrong(propose(
            StructuralOp::Retire,
            vec![in_state(ExpertState::Retired)],
            None
        )),
        ExpertState::Retired
    );
    let candidate = target("b", 1, token(3), ExpertState::Candidate);
    assert_eq!(
        wrong(propose(
            StructuralOp::Merge,
            vec![serving("a"), candidate],
            None
        )),
        ExpertState::Candidate
    );
}

#[test]
fn merge_rejects_duplicates_and_mixed_levels() {
    assert_eq!(
        propose(StructuralOp::Merge, vec![serving("a"), serving("a")], None),
        Err(ExpertProposalError::DuplicateTarget {
            pod: PodId::from("a"),
            generation: Generation(1)
        })
    );
    // Same pod at another generation is a distinct target.
    let newer = target("a", 2, token(3), ExpertState::Serving);
    assert!(propose(StructuralOp::Merge, vec![serving("a"), newer], None).is_ok());

    let other_layer = target("b", 1, token(4), ExpertState::Serving);
    assert_eq!(
        propose(StructuralOp::Merge, vec![serving("a"), other_layer], None),
        Err(ExpertProposalError::MixedLevels)
    );
    let segment = target("b", 1, ExpertLevel::Segment, ExpertState::Serving);
    assert_eq!(
        propose(StructuralOp::Merge, vec![serving("a"), segment], None),
        Err(ExpertProposalError::MixedLevels)
    );
}

#[test]
fn malformed_targets_and_rationale_are_rejected() {
    let zero_gen = target("a", 0, token(3), ExpertState::Serving);
    assert!(matches!(
        propose(StructuralOp::Freeze, vec![zero_gen], None),
        Err(ExpertProposalError::InvalidTarget { .. })
    ));
    let blank = target(" ", 1, token(3), ExpertState::Serving);
    assert!(matches!(
        propose(StructuralOp::Freeze, vec![blank], None),
        Err(ExpertProposalError::InvalidTarget { .. })
    ));
    assert_eq!(
        ExpertProposal::new(StructuralOp::Freeze, vec![serving("a")], None, [0; 32]),
        Err(ExpertProposalError::ZeroRationaleDigest)
    );
}

#[test]
fn errors_have_readable_display() {
    let err = propose(StructuralOp::Freeze, vec![], None).unwrap_err();
    assert_eq!(
        err.to_string(),
        "freeze needs exactly one target(s), found 0"
    );
    let err: Box<dyn std::error::Error> = Box::new(ExpertProposalError::MixedLevels);
    assert!(err.to_string().contains("share one level"));
}

#[test]
fn pod_spec_validates_identity() {
    let make = |id: &str, m: [u8; 32], b: [u8; 32], g: u64| {
        ExpertPodSpec::new(PodId::from(id), m, token(2), b, Generation(g))
    };
    let spec = make("e", [1; 32], [2; 32], 1).expect("valid spec");
    assert_eq!(spec.pod(), &PodId::from("e"));
    assert_eq!(spec.manifest_digest(), &[1; 32]);
    assert_eq!(spec.base_model(), &[2; 32]);
    assert_eq!(spec.level(), token(2));
    assert_eq!(spec.generation(), Generation(1));
    assert_eq!(
        make("", [1; 32], [2; 32], 1),
        Err(ExpertSpecError::BlankPod)
    );
    assert_eq!(
        make("e", [0; 32], [2; 32], 1),
        Err(ExpertSpecError::ZeroManifestDigest)
    );
    assert_eq!(
        make("e", [1; 32], [0; 32], 1),
        Err(ExpertSpecError::ZeroBaseModel)
    );
    assert_eq!(
        make("e", [1; 32], [2; 32], 0),
        Err(ExpertSpecError::ZeroGeneration)
    );
}

/// A proposal carries no threshold, gate, verifier, score-override or policy field.
///
/// `parts()` returns every field of the proposal. The destructure below names each field and has
/// no `..`, so adding a field (for example a gate threshold) stops this test from compiling and
/// forces a reviewer to confront the change against invariants #4 and #11.
#[test]
fn proposal_has_no_override_surface() {
    let proposal = propose(StructuralOp::Merge, vec![serving("a"), serving("b")], None).unwrap();
    let ProposalParts {
        op,
        targets,
        birth_level,
        rationale_digest,
    } = proposal.parts();
    assert_eq!(op, StructuralOp::Merge);
    assert_eq!(targets.len(), 2);
    assert_eq!(birth_level, None);
    assert_eq!(rationale_digest, &RATIONALE);
    assert_eq!(targets, proposal.targets());
}
