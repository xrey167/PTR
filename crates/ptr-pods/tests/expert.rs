use ptr_pods::{
    ExpertLevel, ExpertPodSpec, ExpertProposal, ExpertProposalError, ExpertSpecError, ExpertState,
    ExpertTarget, ProposalParts, StructuralOp,
};
use ptr_types::{Digest, Generation, PodId, ProjectId};

const RATIONALE: [u8; 32] = [7; 32];
const BASE: Digest = [9; 32];

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn target(name: &str, generation: u64, level: ExpertLevel, state: ExpertState) -> ExpertTarget {
    ExpertTarget {
        project: project(),
        pod: PodId::from(name),
        generation: Generation(generation),
        level,
        base_model: BASE,
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
    // Birth names the base model of the newborn; other operations derive it from their targets.
    let birth_base = (op == StructuralOp::Birth).then_some(BASE);
    ExpertProposal::new(op, project(), targets, birth_level, birth_base, RATIONALE)
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
        propose(
            StructuralOp::Reactivate,
            vec![target("d", 1, token(3), ExpertState::Frozen)],
            None,
        ),
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
    for s in [ExpertState::Serving, ExpertState::Candidate] {
        assert_eq!(
            wrong(propose(StructuralOp::Reactivate, vec![in_state(s)], None)),
            s
        );
    }
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
fn reactivate_frozen_unfreezes_and_retired_seeds_a_new_generation() {
    let frozen = target("a", 2, token(3), ExpertState::Frozen);
    let retired = target("a", 1, token(3), ExpertState::Retired);
    let unfreeze = propose(StructuralOp::Reactivate, vec![frozen], None).unwrap();
    assert!(!unfreeze.creates_new_generation());
    let reseed = propose(StructuralOp::Reactivate, vec![retired], None).unwrap();
    assert!(reseed.creates_new_generation());
}

#[test]
fn creates_new_generation_for_every_op() {
    let created = |op, targets, level| {
        propose(op, targets, level)
            .unwrap()
            .creates_new_generation()
    };
    assert!(created(StructuralOp::Birth, vec![], Some(token(0))));
    assert!(created(
        StructuralOp::Merge,
        vec![serving("a"), serving("b")],
        None
    ));
    assert!(created(StructuralOp::Split, vec![serving("a")], None));
    assert!(!created(StructuralOp::Freeze, vec![serving("a")], None));
    assert!(!created(StructuralOp::Prune, vec![serving("a")], None));
    assert!(!created(StructuralOp::Retire, vec![serving("a")], None));
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
    // Two generations of the same logical expert are never merged together, in either order.
    let newer = target("a", 2, token(3), ExpertState::Serving);
    let multiple = Err(ExpertProposalError::MultipleGenerations {
        pod: PodId::from("a"),
        first: Generation(1),
        second: Generation(2),
    });
    assert_eq!(
        propose(StructuralOp::Merge, vec![serving("a"), newer.clone()], None),
        multiple
    );
    assert_eq!(
        propose(
            StructuralOp::Merge,
            vec![serving("b"), serving("a"), newer],
            None
        ),
        multiple
    );

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
        ExpertProposal::new(
            StructuralOp::Freeze,
            project(),
            vec![serving("a")],
            None,
            None,
            [0; 32]
        ),
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
        ExpertPodSpec::new(project(), PodId::from(id), m, token(2), b, Generation(g))
    };
    let spec = make("e", [1; 32], [2; 32], 1).expect("valid spec");
    assert_eq!(spec.project(), &project());
    assert_eq!(spec.pod(), &PodId::from("e"));
    assert_eq!(spec.manifest_digest(), &[1; 32]);
    assert_eq!(spec.base_model(), &[2; 32]);
    assert_eq!(spec.level(), token(2));
    assert_eq!(spec.generation(), Generation(1));
    assert_eq!(
        ExpertPodSpec::new(
            ProjectId::from(" "),
            PodId::from("e"),
            [1; 32],
            token(2),
            [2; 32],
            Generation(1)
        ),
        Err(ExpertSpecError::BlankProject)
    );
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
        project: parts_project,
        targets,
        birth_level,
        birth_base_model,
        rationale_digest,
    } = proposal.parts();
    assert_eq!(op, StructuralOp::Merge);
    assert_eq!(parts_project, &project());
    assert_eq!(proposal.project(), &project());
    assert_eq!(targets.len(), 2);
    assert_eq!(birth_level, None);
    assert_eq!(birth_base_model, None);
    assert_eq!(rationale_digest, &RATIONALE);
    assert_eq!(targets, proposal.targets());
}

fn in_project(mut t: ExpertTarget, project: &str) -> ExpertTarget {
    t.project = ProjectId::from(project);
    t
}

fn with_base(mut t: ExpertTarget, base: Digest) -> ExpertTarget {
    t.base_model = base;
    t
}

#[test]
fn merge_rejects_mixed_base_models() {
    let other = with_base(serving("b"), [10; 32]);
    assert_eq!(
        propose(StructuralOp::Merge, vec![serving("a"), other.clone()], None),
        Err(ExpertProposalError::MixedBaseModels)
    );
    // Order does not matter.
    assert_eq!(
        propose(StructuralOp::Merge, vec![other, serving("a")], None),
        Err(ExpertProposalError::MixedBaseModels)
    );
    // Same base model is fine; single-target ops cannot mix.
    assert!(propose(StructuralOp::Merge, vec![serving("a"), serving("b")], None).is_ok());
    let other = with_base(serving("c"), [10; 32]);
    assert!(propose(StructuralOp::Split, vec![other], None).is_ok());
    let zero = with_base(serving("c"), [0; 32]);
    assert!(matches!(
        propose(StructuralOp::Freeze, vec![zero], None),
        Err(ExpertProposalError::InvalidTarget { .. })
    ));
}

#[test]
fn targets_must_belong_to_the_proposals_project() {
    let foreign = in_project(serving("b"), "other");
    assert_eq!(
        propose(
            StructuralOp::Merge,
            vec![serving("a"), foreign.clone()],
            None
        ),
        Err(ExpertProposalError::MixedProjects {
            expected: project(),
            found: ProjectId::from("other")
        })
    );
    for op in [
        StructuralOp::Split,
        StructuralOp::Freeze,
        StructuralOp::Prune,
        StructuralOp::Retire,
    ] {
        assert!(matches!(
            propose(op, vec![foreign.clone()], None),
            Err(ExpertProposalError::MixedProjects { .. })
        ));
    }
    // A proposal for the other project accepts it.
    let proposal = ExpertProposal::new(
        StructuralOp::Freeze,
        ProjectId::from("other"),
        vec![foreign],
        None,
        None,
        RATIONALE,
    )
    .unwrap();
    assert_eq!(proposal.project(), &ProjectId::from("other"));
    // Birth names its project and no target is needed.
    let birth = ExpertProposal::new(
        StructuralOp::Birth,
        ProjectId::from("other"),
        vec![],
        Some(token(0)),
        Some(BASE),
        RATIONALE,
    )
    .unwrap();
    assert_eq!(birth.project(), &ProjectId::from("other"));
    assert_eq!(
        ExpertProposal::new(
            StructuralOp::Birth,
            ProjectId::from(""),
            vec![],
            Some(token(0)),
            Some(BASE),
            RATIONALE
        ),
        Err(ExpertProposalError::BlankProject)
    );
    let blank = in_project(serving("a"), " ");
    assert!(matches!(
        propose(StructuralOp::Freeze, vec![blank], None),
        Err(ExpertProposalError::InvalidTarget { .. })
    ));
}

#[test]
fn same_pod_id_in_different_projects_is_a_distinct_expert() {
    let a = serving("shared");
    let b = in_project(serving("shared"), "other");
    assert_ne!(a, b);
    // Not a DuplicateTarget: the project check fires first because they are different experts.
    assert!(matches!(
        propose(StructuralOp::Merge, vec![a.clone(), b], None),
        Err(ExpertProposalError::MixedProjects { .. })
    ));
    // Within one project the same (pod, generation) is still a duplicate.
    assert!(matches!(
        propose(StructuralOp::Merge, vec![a.clone(), a], None),
        Err(ExpertProposalError::DuplicateTarget { .. })
    ));
    let spec = |project: &str| {
        ExpertPodSpec::new(
            ProjectId::from(project),
            PodId::from("shared"),
            [1; 32],
            token(2),
            BASE,
            Generation(1),
        )
        .unwrap()
    };
    assert_ne!(spec("proj"), spec("other"));
    assert_eq!(spec("other").project(), &ProjectId::from("other"));
}

#[test]
fn birth_must_bind_a_base_model_and_others_must_not() {
    let birth = |base| {
        ExpertProposal::new(
            StructuralOp::Birth,
            project(),
            vec![],
            Some(token(0)),
            base,
            RATIONALE,
        )
    };
    let ok = birth(Some(BASE)).unwrap();
    assert_eq!(ok.birth_base_model(), Some(&BASE));
    assert_eq!(birth(None), Err(ExpertProposalError::BirthWithoutBaseModel));
    assert_eq!(
        birth(Some([0; 32])),
        Err(ExpertProposalError::ZeroBirthBaseModel)
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
            ExpertProposal::new(
                op,
                project(),
                vec![serving("a"), serving("b")],
                None,
                Some(BASE),
                RATIONALE
            ),
            Err(ExpertProposalError::UnexpectedBirthBaseModel { op })
        );
    }
    let freeze = propose(StructuralOp::Freeze, vec![serving("a")], None).unwrap();
    assert_eq!(freeze.birth_base_model(), None);
    let readable = ExpertProposalError::MultipleGenerations {
        pod: PodId::from("a"),
        first: Generation(1),
        second: Generation(2),
    };
    assert!(readable.to_string().contains("generations 1 and 2"));
}
