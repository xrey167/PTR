use ptr_pods::{
    ArtifactLifecycle, BaseModelDigest, BaseModelRef, ExpertIdentityError, ExpertLevel,
    ExpertPodSpec, ExpertProposal, ExpertProposalError, ExpertState, ExpertTarget,
    ExpertTargetInit, ManifestDigest, OpKind, OperandCount, ProposalParts, ProposerRef,
    RationaleDigest, RetiredReason, StructuralOp, ZeroDigest,
};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

type Proposal = Result<ExpertProposal, ExpertProposalError>;

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef {
        digest: BaseModelDigest::new([9; 32]).unwrap(),
        generation: 1,
    }
}

fn other_base() -> BaseModelRef {
    BaseModelRef {
        digest: BaseModelDigest::new([10; 32]).unwrap(),
        generation: 1,
    }
}

fn rationale() -> RationaleDigest {
    RationaleDigest::new([7; 32]).unwrap()
}

fn token(layer: u16) -> ExpertLevel {
    ExpertLevel::Token { layer, index: 0 }
}

fn init(pod: &str, generation: u64, level: ExpertLevel, state: ExpertState) -> ExpertTargetInit {
    ExpertTargetInit {
        project: project(),
        pod: PodId::from(pod),
        generation: Generation(generation),
        level,
        base_model: base(),
        manifest_digest: ManifestDigest::new([1; 32]).unwrap(),
        state,
        state_version: 1,
    }
}

fn target(pod: &str, generation: u64, level: ExpertLevel, state: ExpertState) -> ExpertTarget {
    ExpertTarget::new(init(pod, generation, level, state)).unwrap()
}

fn serving(pod: &str) -> ExpertTarget {
    target(pod, 1, token(3), ExpertState::Serving)
}

fn in_state(pod: &str, state: ExpertState) -> ExpertTarget {
    target(pod, 1, token(3), state)
}

fn spec(pod: &str, generation: u64, level: ExpertLevel) -> ExpertPodSpec {
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new([2; 32]).unwrap(),
        level,
        base(),
        Generation(generation),
    )
    .unwrap()
}

fn result(pod: &str) -> ExpertPodSpec {
    spec(pod, 1, token(3))
}

fn proposer() -> ProposerRef {
    ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap()
}

fn propose(op: StructuralOp) -> Proposal {
    ExpertProposal::new(project(), proposer(), CommitIndex(10), op, rationale())
}

fn merge(sources: Vec<ExpertTarget>) -> StructuralOp {
    StructuralOp::Merge {
        sources,
        result: result("merged"),
    }
}

fn reactivate(target: ExpertTarget, result: Option<ExpertPodSpec>) -> StructuralOp {
    StructuralOp::Reactivate { target, result }
}

const ALL_REASONS: [RetiredReason; 4] = [
    RetiredReason::Pruned,
    RetiredReason::Superseded,
    RetiredReason::GateFailed,
    RetiredReason::InputRevoked,
];

fn all_states() -> Vec<ExpertState> {
    let mut states = vec![
        ExpertState::Candidate,
        ExpertState::Gated,
        ExpertState::Serving,
        ExpertState::Frozen,
        ExpertState::Fenced,
        ExpertState::Superseded,
    ];
    states.extend(ALL_REASONS.map(ExpertState::Retired));
    states
}

#[test]
fn valid_proposal_for_each_operation() {
    let frozen = |pod: &str| in_state(pod, ExpertState::Frozen);
    let retired = target(
        "c",
        1,
        ExpertLevel::Segment,
        ExpertState::Retired(RetiredReason::GateFailed),
    );
    let cases = [
        StructuralOp::Birth {
            result: result("new"),
        },
        StructuralOp::Birth {
            result: spec("seg", 1, ExpertLevel::Segment),
        },
        merge(vec![serving("a"), serving("b")]),
        merge(vec![frozen("a"), frozen("b")]),
        StructuralOp::Split {
            source: serving("a"),
            results: vec![result("x"), result("y")],
        },
        StructuralOp::Freeze {
            target: serving("a"),
        },
        reactivate(retired, Some(spec("c", 2, ExpertLevel::Segment))),
        reactivate(frozen("d"), None),
        StructuralOp::Prune {
            targets: vec![serving("a"), serving("b")],
        },
        StructuralOp::Retire {
            target: in_state("a", ExpertState::Candidate),
            reason: RetiredReason::GateFailed,
        },
    ];
    for op in cases {
        let kind = op.kind();
        let proposal = propose(op).unwrap_or_else(|e| panic!("{kind}: {e:?}"));
        assert_eq!(proposal.rationale(), &rationale());
        assert_eq!(proposal.observed_at(), CommitIndex(10));
        assert_eq!(proposal.project(), &project());
    }
}

/// The documented (operation, state) table, written out independently of the implementation.
#[test]
fn allowed_operation_state_pairs_match_the_documented_table() {
    use ExpertState::*;
    let retired = |s: ExpertState| matches!(s, Retired(_));
    for state in all_states() {
        let expected = [
            (OpKind::Merge, matches!(state, Serving | Frozen)),
            (OpKind::Split, matches!(state, Serving | Frozen)),
            (OpKind::Freeze, matches!(state, Serving)),
            (
                OpKind::Reactivate,
                matches!(state, Frozen) || retired(state),
            ),
            (OpKind::Prune, matches!(state, Serving)),
            (
                OpKind::Retire,
                matches!(
                    state,
                    Candidate | Gated | Serving | Frozen | Fenced | Superseded
                ),
            ),
            (OpKind::Birth, false),
        ];
        for (kind, allowed) in expected {
            assert_eq!(kind.allows_state(state), allowed, "{kind} / {state:?}");
        }
    }
}

/// Validation agrees with `allows_state` for every pair, and Fenced/Superseded are valid only
/// for Retire.
#[test]
fn validation_enforces_the_table_for_every_state() {
    for state in all_states() {
        let t = || in_state("a", state);
        let ok = |op: StructuralOp| match propose(op) {
            Ok(_) => true,
            Err(ExpertProposalError::WrongTargetState { state: s, .. }) => {
                assert_eq!(s, state);
                false
            }
            Err(other) => panic!("unexpected error for {state:?}: {other:?}"),
        };
        assert_eq!(
            ok(StructuralOp::Freeze { target: t() }),
            OpKind::Freeze.allows_state(state)
        );
        assert_eq!(
            ok(StructuralOp::Split {
                source: t(),
                results: vec![result("x"), result("y")]
            }),
            OpKind::Split.allows_state(state)
        );
        assert_eq!(
            ok(StructuralOp::Prune { targets: vec![t()] }),
            OpKind::Prune.allows_state(state)
        );
        assert_eq!(
            ok(StructuralOp::Retire {
                target: t(),
                reason: RetiredReason::Pruned
            }),
            OpKind::Retire.allows_state(state)
        );
        let other = in_state("b", state);
        assert_eq!(
            ok(merge(vec![t(), other])),
            OpKind::Merge.allows_state(state)
        );
        if !matches!(state, ExpertState::Retired(_) | ExpertState::Frozen) {
            assert!(!ok(reactivate(t(), None)));
        }
        if matches!(state, ExpertState::Fenced | ExpertState::Superseded) {
            for op in [
                StructuralOp::Freeze { target: t() },
                StructuralOp::Prune { targets: vec![t()] },
                reactivate(t(), None),
            ] {
                assert!(matches!(
                    propose(op),
                    Err(ExpertProposalError::WrongTargetState { .. })
                ));
            }
        }
    }
}

#[test]
fn states_map_onto_lineage_status_and_lifecycle() {
    use ArtifactLifecycle as L;
    use ExpertState::*;
    let table: Vec<(ExpertState, &str, Vec<L>)> = vec![
        (
            Candidate,
            "candidate",
            vec![L::Candidate, L::Trained, L::Evaluated],
        ),
        (Gated, "gated", vec![L::Approved]),
        (Serving, "serving", vec![L::Active]),
        (Frozen, "gated", vec![L::Approved]),
        (Fenced, "serving", vec![L::Active]),
        (Superseded, "retired", vec![L::Superseded]),
        (Retired(RetiredReason::Pruned), "retired", vec![L::Retired]),
        (
            Retired(RetiredReason::Superseded),
            "retired",
            vec![L::Retired],
        ),
        (
            Retired(RetiredReason::GateFailed),
            "retired",
            vec![L::Retired],
        ),
        (
            Retired(RetiredReason::InputRevoked),
            "retired",
            vec![L::Revoked],
        ),
    ];
    assert_eq!(table.len(), all_states().len());
    for (state, adapter, lifecycles) in table {
        assert_eq!(state.adapter_status_name(), adapter, "{state:?}");
        assert_eq!(
            state.artifact_lifecycles(),
            lifecycles.as_slice(),
            "{state:?}"
        );
    }
}

#[test]
fn operand_counts_are_typed() {
    assert_eq!(
        propose(merge(vec![serving("a")])),
        Err(ExpertProposalError::WrongSourceCount {
            op: OpKind::Merge,
            expected: OperandCount::AtLeast(2),
            found: 1
        })
    );
    assert_eq!(
        propose(merge(vec![])),
        Err(ExpertProposalError::WrongSourceCount {
            op: OpKind::Merge,
            expected: OperandCount::AtLeast(2),
            found: 0
        })
    );
    assert_eq!(
        propose(StructuralOp::Prune { targets: vec![] }),
        Err(ExpertProposalError::WrongSourceCount {
            op: OpKind::Prune,
            expected: OperandCount::AtLeast(1),
            found: 0
        })
    );
    for results in [vec![], vec![result("x")]] {
        let found = results.len();
        assert_eq!(
            propose(StructuralOp::Split {
                source: serving("a"),
                results
            }),
            Err(ExpertProposalError::WrongResultCount {
                op: OpKind::Split,
                expected: OperandCount::AtLeast(2),
                found
            })
        );
    }
    // Single-operand shapes (birth with a target, reactivate with two targets, freeze with a
    // result) are unrepresentable: the operands are typed fields of `StructuralOp`.
    let err = propose(merge(vec![serving("a")])).unwrap_err();
    assert_eq!(err.to_string(), "merge needs at least 2 source(s), found 1");
}

#[test]
fn everything_must_belong_to_the_proposal_project() {
    let foreign = |pod: &str| {
        let mut i = init(pod, 1, token(3), ExpertState::Serving);
        i.project = ProjectId::from("other");
        ExpertTarget::new(i).unwrap()
    };
    let mixed = Err(ExpertProposalError::MixedProjects {
        expected: project(),
        found: ProjectId::from("other"),
    });
    assert_eq!(propose(merge(vec![serving("a"), foreign("b")])), mixed);
    for op in [
        StructuralOp::Freeze {
            target: foreign("a"),
        },
        StructuralOp::Prune {
            targets: vec![foreign("a")],
        },
        StructuralOp::Retire {
            target: foreign("a"),
            reason: RetiredReason::Pruned,
        },
    ] {
        assert_eq!(propose(op), mixed);
    }
    let foreign_result = ExpertPodSpec::new(
        ProjectId::from("other"),
        PodId::from("x"),
        ManifestDigest::new([2; 32]).unwrap(),
        token(3),
        base(),
        Generation(1),
    )
    .unwrap();
    assert_eq!(
        propose(StructuralOp::Birth {
            result: foreign_result
        }),
        mixed
    );
    let foreign_proposer =
        ProposerRef::new(ProjectId::from("other"), PodId::from("meta"), Generation(1)).unwrap();
    assert_eq!(
        ExpertProposal::new(
            project(),
            foreign_proposer,
            CommitIndex(1),
            StructuralOp::Freeze {
                target: serving("a")
            },
            rationale()
        ),
        Err(ExpertProposalError::ProposerProjectMismatch {
            expected: project(),
            found: ProjectId::from("other")
        })
    );
    // A proposal for the other project accepts a target from it.
    let p = ExpertProposal::new(
        ProjectId::from("other"),
        ProposerRef::new(ProjectId::from("other"), PodId::from("meta"), Generation(1)).unwrap(),
        CommitIndex(1),
        StructuralOp::Freeze {
            target: foreign("a"),
        },
        rationale(),
    )
    .unwrap();
    assert_eq!(p.project(), &ProjectId::from("other"));
}

#[test]
fn same_pod_id_in_different_projects_is_a_distinct_expert() {
    let a = serving("shared");
    let mut i = init("shared", 1, token(3), ExpertState::Serving);
    i.project = ProjectId::from("other");
    let b = ExpertTarget::new(i).unwrap();
    assert_ne!(a, b);
    assert!(matches!(
        propose(merge(vec![a.clone(), b])),
        Err(ExpertProposalError::MixedProjects { .. })
    ));
    assert!(matches!(
        propose(merge(vec![a.clone(), a])),
        Err(ExpertProposalError::DuplicateTarget { .. })
    ));
}

#[test]
fn merge_rules() {
    // Token level only.
    let seg = |pod: &str| target(pod, 1, ExpertLevel::Segment, ExpertState::Serving);
    assert_eq!(
        propose(merge(vec![seg("a"), seg("b")])),
        Err(ExpertProposalError::NotTokenLevel { op: OpKind::Merge })
    );
    // Different layers mix; different indices in one layer do not.
    let at = |pod: &str, layer: u16, index: u16| {
        target(
            pod,
            1,
            ExpertLevel::Token { layer, index },
            ExpertState::Serving,
        )
    };
    assert_eq!(
        propose(merge(vec![at("a", 3, 0), at("b", 4, 0)])),
        Err(ExpertProposalError::MixedLevels)
    );
    assert!(propose(merge(vec![at("a", 3, 0), at("b", 3, 5)])).is_ok());
    // Base models and states.
    let mut i = init("b", 1, token(3), ExpertState::Serving);
    i.base_model = other_base();
    let other_model = ExpertTarget::new(i).unwrap();
    assert_eq!(
        propose(merge(vec![serving("a"), other_model.clone()])),
        Err(ExpertProposalError::MixedBaseModels)
    );
    assert_eq!(
        propose(merge(vec![other_model, serving("a")])),
        Err(ExpertProposalError::MixedBaseModels)
    );
    let mut i = init("b", 1, token(3), ExpertState::Serving);
    i.base_model = BaseModelRef {
        generation: 2,
        ..base()
    };
    assert_eq!(
        propose(merge(vec![serving("a"), ExpertTarget::new(i).unwrap()])),
        Err(ExpertProposalError::MixedBaseModels)
    );
    assert_eq!(
        propose(merge(vec![
            serving("a"),
            in_state("b", ExpertState::Frozen)
        ])),
        Err(ExpertProposalError::MixedStates)
    );
    // Duplicates and two generations of one logical expert.
    assert_eq!(
        propose(merge(vec![serving("a"), serving("a")])),
        Err(ExpertProposalError::DuplicateTarget {
            pod: PodId::from("a"),
            generation: Generation(1)
        })
    );
    let newer = target("a", 2, token(3), ExpertState::Serving);
    let multiple = Err(ExpertProposalError::MultipleGenerations {
        pod: PodId::from("a"),
        first: Generation(1),
        second: Generation(2),
    });
    assert_eq!(propose(merge(vec![serving("a"), newer.clone()])), multiple);
    assert_eq!(
        propose(merge(vec![serving("b"), serving("a"), newer])),
        multiple
    );
}

#[test]
fn prune_is_a_batch_within_one_token_layer() {
    let at = |pod: &str, layer: u16, index: u16| {
        target(
            pod,
            1,
            ExpertLevel::Token { layer, index },
            ExpertState::Serving,
        )
    };
    let prune = |targets| propose(StructuralOp::Prune { targets });
    assert!(prune(vec![at("a", 3, 0), at("b", 3, 1), at("c", 3, 2)]).is_ok());
    assert!(prune(vec![at("a", 3, 0)]).is_ok());
    assert_eq!(
        prune(vec![at("a", 3, 0), at("b", 4, 0)]),
        Err(ExpertProposalError::MixedLevels)
    );
    assert_eq!(
        prune(vec![target(
            "a",
            1,
            ExpertLevel::Segment,
            ExpertState::Serving
        )]),
        Err(ExpertProposalError::NotTokenLevel { op: OpKind::Prune })
    );
    assert!(matches!(
        prune(vec![at("a", 3, 0), at("a", 3, 0)]),
        Err(ExpertProposalError::DuplicateTarget { .. })
    ));
    assert_eq!(
        prune(vec![at("a", 3, 0), in_state("b", ExpertState::Candidate)]),
        Err(ExpertProposalError::WrongTargetState {
            op: OpKind::Prune,
            pod: PodId::from("b"),
            state: ExpertState::Candidate
        })
    );
}

#[test]
fn split_and_prune_reject_candidate_and_retired() {
    for state in [
        ExpertState::Candidate,
        ExpertState::Retired(RetiredReason::Pruned),
    ] {
        assert!(matches!(
            propose(StructuralOp::Split {
                source: in_state("a", state),
                results: vec![result("x"), result("y")]
            }),
            Err(ExpertProposalError::WrongTargetState { .. })
        ));
        assert!(matches!(
            propose(StructuralOp::Prune {
                targets: vec![in_state("a", state)]
            }),
            Err(ExpertProposalError::WrongTargetState { .. })
        ));
    }
}

#[test]
fn reactivate_frozen_unfreezes_and_retired_seeds_a_new_generation() {
    let frozen = in_state("a", ExpertState::Frozen);
    let retired = |reason| in_state("a", ExpertState::Retired(reason));
    let new_gen = || Some(spec("a", 2, token(3)));

    let unfreeze = propose(reactivate(frozen.clone(), None)).unwrap();
    assert!(!unfreeze.creates_new_generation());
    let reseed = propose(reactivate(retired(RetiredReason::GateFailed), new_gen())).unwrap();
    assert!(reseed.creates_new_generation());

    assert_eq!(
        propose(reactivate(frozen, Some(spec("a", 2, token(3))))),
        Err(ExpertProposalError::UnexpectedResult {
            op: OpKind::Reactivate
        })
    );
    assert_eq!(
        propose(reactivate(retired(RetiredReason::Pruned), None)),
        Err(ExpertProposalError::MissingResult {
            op: OpKind::Reactivate
        })
    );
    // Revoked inputs never seed, with or without a result.
    for result in [None, new_gen()] {
        assert_eq!(
            propose(reactivate(retired(RetiredReason::InputRevoked), result)),
            Err(ExpertProposalError::SeedFromRevoked {
                pod: PodId::from("a")
            })
        );
    }
    // The result is a strictly newer generation of the same logical expert.
    assert_eq!(
        propose(reactivate(
            retired(RetiredReason::Pruned),
            Some(spec("a", 1, token(3)))
        )),
        Err(ExpertProposalError::ResultGenerationNotNewer {
            pod: PodId::from("a"),
            result: Generation(1),
            source: Generation(1)
        })
    );
    assert_eq!(
        propose(reactivate(
            retired(RetiredReason::Pruned),
            Some(spec("elsewhere", 2, token(3)))
        )),
        Err(ExpertProposalError::ResultNotSameExpert {
            expected: PodId::from("a"),
            found: PodId::from("elsewhere")
        })
    );
    // Serving and Candidate cannot be reactivated.
    for state in [ExpertState::Serving, ExpertState::Candidate] {
        assert!(matches!(
            propose(reactivate(in_state("a", state), None)),
            Err(ExpertProposalError::WrongTargetState { .. })
        ));
    }
}

#[test]
fn result_relations_are_checked() {
    // A merge or split result is a new expert, not one of the sources.
    assert_eq!(
        propose(StructuralOp::Merge {
            sources: vec![serving("a"), serving("b")],
            result: result("a")
        }),
        Err(ExpertProposalError::ResultPodIsSource {
            pod: PodId::from("a")
        })
    );
    assert_eq!(
        propose(StructuralOp::Split {
            source: serving("a"),
            results: vec![result("x"), result("a")]
        }),
        Err(ExpertProposalError::ResultPodIsSource {
            pod: PodId::from("a")
        })
    );
    assert_eq!(
        propose(StructuralOp::Split {
            source: serving("a"),
            results: vec![result("x"), result("x")]
        }),
        Err(ExpertProposalError::DuplicateResult {
            pod: PodId::from("x")
        })
    );
    // Base model and level follow the sources.
    let wrong_base = ExpertPodSpec::new(
        project(),
        PodId::from("x"),
        ManifestDigest::new([2; 32]).unwrap(),
        token(3),
        other_base(),
        Generation(1),
    )
    .unwrap();
    assert_eq!(
        propose(StructuralOp::Merge {
            sources: vec![serving("a"), serving("b")],
            result: wrong_base.clone()
        }),
        Err(ExpertProposalError::ResultBaseModelMismatch {
            pod: PodId::from("x")
        })
    );
    assert!(matches!(
        propose(StructuralOp::Split {
            source: serving("a"),
            results: vec![wrong_base, result("y")]
        }),
        Err(ExpertProposalError::ResultBaseModelMismatch { .. })
    ));
    for level in [token(4), ExpertLevel::Segment] {
        let other = spec("x", 1, level);
        assert_eq!(
            propose(StructuralOp::Merge {
                sources: vec![serving("a"), serving("b")],
                result: other.clone()
            }),
            Err(ExpertProposalError::ResultLevelMismatch {
                pod: PodId::from("x")
            })
        );
    }
    // Same layer, different expert index is compatible.
    let indexed = spec("x", 1, ExpertLevel::Token { layer: 3, index: 9 });
    assert!(propose(StructuralOp::Merge {
        sources: vec![serving("a"), serving("b")],
        result: indexed
    })
    .is_ok());
    // Birth may use any level and any base model.
    assert!(propose(StructuralOp::Birth {
        result: spec("x", 1, ExpertLevel::Segment)
    })
    .is_ok());
}

#[test]
fn ids_must_be_canonical() {
    let make_spec = |project: &str, pod: &str, g: u64| {
        ExpertPodSpec::new(
            ProjectId::from(project),
            PodId::from(pod),
            ManifestDigest::new([2; 32]).unwrap(),
            token(1),
            base(),
            Generation(g),
        )
    };
    assert!(make_spec("proj", "pod", 1).is_ok());
    assert_eq!(
        make_spec(" ", "pod", 1),
        Err(ExpertIdentityError::BlankProject)
    );
    assert_eq!(
        make_spec("", "pod", 1),
        Err(ExpertIdentityError::BlankProject)
    );
    assert_eq!(
        make_spec(" proj", "pod", 1),
        Err(ExpertIdentityError::NonCanonicalProject)
    );
    assert_eq!(
        make_spec("proj\n", "pod", 1),
        Err(ExpertIdentityError::NonCanonicalProject)
    );
    assert_eq!(make_spec("proj", "", 1), Err(ExpertIdentityError::BlankPod));
    assert_eq!(
        make_spec("proj", "pod ", 1),
        Err(ExpertIdentityError::NonCanonicalPod)
    );
    assert_eq!(
        make_spec("proj", "pod", 0),
        Err(ExpertIdentityError::ZeroGeneration)
    );

    let mut i = init("pod", 1, token(1), ExpertState::Serving);
    i.pod = PodId::from(" pod");
    assert_eq!(
        ExpertTarget::new(i),
        Err(ExpertIdentityError::NonCanonicalPod)
    );
    let mut i = init("pod", 1, token(1), ExpertState::Serving);
    i.project = ProjectId::from("p ");
    assert_eq!(
        ExpertTarget::new(i),
        Err(ExpertIdentityError::NonCanonicalProject)
    );
    assert_eq!(
        ExpertTarget::new(init("pod", 0, token(1), ExpertState::Serving)),
        Err(ExpertIdentityError::ZeroGeneration)
    );
    assert_eq!(
        ExpertTarget::new(init(" ", 1, token(1), ExpertState::Serving)),
        Err(ExpertIdentityError::BlankPod)
    );
    assert_eq!(
        ProposerRef::new(project(), PodId::from("m "), Generation(1)),
        Err(ExpertIdentityError::NonCanonicalPod)
    );
    assert_eq!(
        ProposerRef::new(ProjectId::from(""), PodId::from("m"), Generation(1)),
        Err(ExpertIdentityError::BlankProject)
    );
    assert_eq!(
        ProposerRef::new(project(), PodId::from("m"), Generation(0)),
        Err(ExpertIdentityError::ZeroGeneration)
    );
    let freeze = StructuralOp::Freeze {
        target: serving("a"),
    };
    let with_project = |p: &str| {
        let project = ProjectId::from(p);
        ExpertProposal::new(
            project.clone(),
            ProposerRef::new(ProjectId::from("proj"), PodId::from("m"), Generation(1)).unwrap(),
            CommitIndex(1),
            freeze.clone(),
            rationale(),
        )
    };
    assert_eq!(with_project(" "), Err(ExpertProposalError::BlankProject));
    assert_eq!(
        with_project(" proj"),
        Err(ExpertProposalError::NonCanonicalProject)
    );
}

#[test]
fn digests_are_non_zero_newtypes() {
    assert_eq!(ManifestDigest::new([0; 32]), Err(ZeroDigest));
    assert_eq!(BaseModelDigest::new([0; 32]), Err(ZeroDigest));
    assert_eq!(RationaleDigest::new([0; 32]), Err(ZeroDigest));
    assert_eq!(ManifestDigest::new([1; 32]).unwrap().as_bytes(), &[1; 32]);
}

#[test]
fn accessors_expose_validated_fields() {
    let mut i = init("e", 3, token(2), ExpertState::Gated);
    i.state_version = 42;
    let t = ExpertTarget::new(i).unwrap();
    assert_eq!(t.project(), &project());
    assert_eq!(t.pod(), &PodId::from("e"));
    assert_eq!(t.generation(), Generation(3));
    assert_eq!(t.level(), token(2));
    assert_eq!(t.base_model(), &base());
    assert_eq!(t.manifest_digest().as_bytes(), &[1; 32]);
    assert_eq!(t.state(), ExpertState::Gated);
    assert_eq!(t.state_version(), 42);
    let s = spec("e", 4, token(2));
    assert_eq!(s.project(), &project());
    assert_eq!(s.pod(), &PodId::from("e"));
    assert_eq!(s.manifest_digest().as_bytes(), &[2; 32]);
    assert_eq!(s.level(), token(2));
    assert_eq!(s.base_model(), &base());
    assert_eq!(s.generation(), Generation(4));
    let p = proposer();
    assert_eq!(p.project(), &project());
    assert_eq!(p.pod(), &PodId::from("meta"));
    assert_eq!(p.generation(), Generation(1));
}

#[test]
fn validation_order_is_deterministic() {
    let foreign_candidate = {
        let mut i = init("a", 1, token(3), ExpertState::Candidate);
        i.project = ProjectId::from("other");
        ExpertTarget::new(i).unwrap()
    };
    // Count (3) before project (4).
    assert!(matches!(
        propose(merge(vec![foreign_candidate.clone()])),
        Err(ExpertProposalError::WrongSourceCount { .. })
    ));
    // Project (4) before state (5).
    assert!(matches!(
        propose(StructuralOp::Freeze {
            target: foreign_candidate
        }),
        Err(ExpertProposalError::MixedProjects { .. })
    ));
    // State (5) before structure (6): a Candidate segment merge is a state error.
    let seg = |pod: &str| target(pod, 1, ExpertLevel::Segment, ExpertState::Candidate);
    assert!(matches!(
        propose(merge(vec![seg("a"), seg("b")])),
        Err(ExpertProposalError::WrongTargetState { .. })
    ));
    // Structure (6) before results (7): a segment merge with a bad result is NotTokenLevel.
    let seg = |pod: &str| target(pod, 1, ExpertLevel::Segment, ExpertState::Serving);
    assert!(matches!(
        propose(StructuralOp::Merge {
            sources: vec![seg("a"), seg("b")],
            result: result("a")
        }),
        Err(ExpertProposalError::NotTokenLevel { .. })
    ));
    // Proposer project (2) before everything about the op.
    let foreign_proposer =
        ProposerRef::new(ProjectId::from("other"), PodId::from("m"), Generation(1)).unwrap();
    assert!(matches!(
        ExpertProposal::new(
            project(),
            foreign_proposer,
            CommitIndex(1),
            merge(vec![]),
            rationale()
        ),
        Err(ExpertProposalError::ProposerProjectMismatch { .. })
    ));
}

#[test]
fn proposal_id_is_deterministic_and_covers_every_field() {
    let op = || merge(vec![serving("a"), serving("b")]);
    let id = |p: Proposal| p.unwrap().proposal_id();
    let base_id = id(propose(op()));
    assert_eq!(base_id, id(propose(op())));
    assert_ne!(base_id.as_bytes(), &[0; 32]);

    // observed_at.
    let at = |n: u64| {
        id(ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(n),
            op(),
            rationale(),
        ))
    };
    assert_eq!(at(10), base_id);
    assert_ne!(at(11), base_id);
    // rationale.
    assert_ne!(
        id(ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(10),
            op(),
            RationaleDigest::new([8; 32]).unwrap()
        )),
        base_id
    );
    // proposer.
    assert_ne!(
        id(ExpertProposal::new(
            project(),
            ProposerRef::new(project(), PodId::from("meta"), Generation(2)).unwrap(),
            CommitIndex(10),
            op(),
            rationale()
        )),
        base_id
    );
    // operands, including a state change at the same generation (state_version).
    let mut i = init("b", 1, token(3), ExpertState::Serving);
    i.state_version = 2;
    let bumped = ExpertTarget::new(i).unwrap();
    assert_ne!(id(propose(merge(vec![serving("a"), bumped]))), base_id);
    assert_ne!(
        id(propose(merge(vec![serving("b"), serving("a")]))),
        base_id
    );
    // different operation kinds with the same target differ.
    assert_ne!(
        id(propose(StructuralOp::Freeze {
            target: serving("a")
        })),
        id(propose(StructuralOp::Prune {
            targets: vec![serving("a")]
        }))
    );
    assert_ne!(
        id(propose(StructuralOp::Retire {
            target: serving("a"),
            reason: RetiredReason::Pruned
        })),
        id(propose(StructuralOp::Retire {
            target: serving("a"),
            reason: RetiredReason::GateFailed
        }))
    );
}

#[test]
fn errors_have_readable_display() {
    let err: Box<dyn std::error::Error> = Box::new(ExpertProposalError::MixedLevels);
    assert!(err.to_string().contains("share one level"));
    let err = ExpertProposalError::MultipleGenerations {
        pod: PodId::from("a"),
        first: Generation(1),
        second: Generation(2),
    };
    assert!(err.to_string().contains("generations 1 and 2"));
    assert_eq!(
        ExpertIdentityError::NonCanonicalPod.to_string(),
        "pod id has leading or trailing whitespace"
    );
    assert_eq!(OpKind::Reactivate.to_string(), "reactivate");
}

/// A proposal carries no threshold, gate, verifier, score-override or policy field.
///
/// `parts()` returns every field of the proposal. The destructure below names each field and has
/// no `..`, so adding a field (for example a gate threshold) stops this test from compiling and
/// forces a reviewer to confront the change against invariants #4 and #11.
#[test]
fn proposal_has_no_override_surface() {
    let proposal = propose(merge(vec![serving("a"), serving("b")])).unwrap();
    let ProposalParts {
        project: parts_project,
        proposer: parts_proposer,
        observed_at,
        op,
        rationale: parts_rationale,
    } = proposal.parts();
    assert_eq!(parts_project, &project());
    assert_eq!(parts_proposer, &proposer());
    assert_eq!(observed_at, CommitIndex(10));
    assert_eq!(parts_rationale, &rationale());
    assert_eq!(op, proposal.op());
    // The operation enum is matched exhaustively: a new operation or operand needs review.
    match op {
        StructuralOp::Merge { sources, result } => {
            assert_eq!(sources.len(), 2);
            assert_eq!(result.pod(), &PodId::from("merged"));
        }
        StructuralOp::Birth { .. }
        | StructuralOp::Split { .. }
        | StructuralOp::Freeze { .. }
        | StructuralOp::Reactivate { .. }
        | StructuralOp::Prune { .. }
        | StructuralOp::Retire { .. } => panic!("expected merge"),
    }
    assert_eq!(proposal.proposer(), &proposer());
}
