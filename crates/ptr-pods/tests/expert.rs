use ptr_pods::{
    ArtifactLifecycle, BaseModelDigest, BaseModelRef, ExpertIdentityError, ExpertLevel,
    ExpertOperationRequest, ExpertPodSpec, ExpertProposal, ExpertProposalError, ExpertState,
    ExpertTarget, ExpertTargetInit, ManifestDigest, OpKind, OperandCount, OperationId,
    ProposalParts, ProposerRef, RationaleDigest, RequestedOp, RetiredReason, StructuralOp,
    ZeroDigest,
};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

type Proposal = Result<ExpertProposal, ExpertProposalError>;

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(BaseModelDigest::new([9; 32]).unwrap(), 1).unwrap()
}

fn other_base() -> BaseModelRef {
    BaseModelRef::new(BaseModelDigest::new([10; 32]).unwrap(), 1).unwrap()
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

/// Layer 3 with a slot index that follows the pod name, so distinct sources never share a slot.
fn slot(pod: &str) -> ExpertLevel {
    ExpertLevel::Token {
        layer: 3,
        index: u16::from(pod.as_bytes()[0]),
    }
}

fn serving(pod: &str) -> ExpertTarget {
    target(pod, 1, slot(pod), ExpertState::Serving)
}

fn in_state(pod: &str, state: ExpertState) -> ExpertTarget {
    target(pod, 1, slot(pod), state)
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

/// A fresh generation-1 result in layer 3; the slot index follows the pod name so that results
/// of one operation get distinct slots.
fn result(pod: &str) -> ExpertPodSpec {
    let index = u16::from(pod.as_bytes()[0]);
    spec(pod, 1, ExpertLevel::Token { layer: 3, index })
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
            target: in_state("a", ExpertState::Frozen),
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

/// The documented (operation, state) table as a literal matrix, written independently of the
/// implementation. Columns: Birth, Merge, Split, Freeze, Reactivate, Prune, Retire.
#[test]
fn allowed_operation_state_pairs_match_the_documented_table() {
    use ExpertState::*;
    const OPS: [OpKind; 7] = [
        OpKind::Birth,
        OpKind::Merge,
        OpKind::Split,
        OpKind::Freeze,
        OpKind::Reactivate,
        OpKind::Prune,
        OpKind::Retire,
    ];
    let matrix: [(ExpertState, [bool; 7]); 10] = [
        //                              Birth  Merge  Split  Freeze React  Prune  Retire
        (Candidate, [false, false, false, false, false, false, false]),
        (Gated, [false, false, false, false, false, false, false]),
        (Serving, [false, true, true, true, false, true, true]),
        (Frozen, [false, true, true, false, true, false, true]),
        (Fenced, [false, false, false, false, false, false, false]),
        (
            Superseded,
            [false, false, false, false, false, false, false],
        ),
        (
            Retired(RetiredReason::Pruned),
            [false, false, false, false, false, false, false],
        ),
        (
            Retired(RetiredReason::Superseded),
            [false, false, false, false, false, false, false],
        ),
        (
            Retired(RetiredReason::GateFailed),
            [false, false, false, false, true, false, false],
        ),
        (
            Retired(RetiredReason::InputRevoked),
            [false, false, false, false, false, false, false],
        ),
    ];
    assert_eq!(matrix.len(), all_states().len());
    for state in all_states() {
        assert!(matrix.iter().any(|(s, _)| *s == state), "{state:?} missing");
    }
    for (state, row) in matrix {
        for (kind, allowed) in OPS.into_iter().zip(row) {
            assert_eq!(kind.allows_state(state), allowed, "{kind} / {state:?}");
        }
    }
}

/// Validation agrees with `allows_state` for every pair. A Fenced operand is `FencedTarget` for
/// every operation (see `fenced_operands_are_rejected_for_every_operation`).
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
            Err(ExpertProposalError::FencedTarget { .. }) => {
                assert_eq!(state, ExpertState::Fenced);
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
                reason: RetiredReason::Superseded
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
        if matches!(state, ExpertState::Superseded) {
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
        (Frozen, "serving", vec![L::Active]),
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
            reason: RetiredReason::GateFailed,
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
    i.base_model = BaseModelRef::new(*base().digest(), 2).unwrap();
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
fn reactivate_frozen_unfreezes_and_failed_gate_seeds_a_new_generation() {
    let frozen = in_state("a", ExpertState::Frozen);
    let failed = || in_state("a", ExpertState::Retired(RetiredReason::GateFailed));
    let new_gen = || Some(spec("a", 2, slot("a")));

    let unfreeze = propose(reactivate(frozen.clone(), None)).unwrap();
    assert!(!unfreeze.creates_new_generation());
    let reseed = propose(reactivate(failed(), new_gen())).unwrap();
    assert!(reseed.creates_new_generation());

    assert_eq!(
        propose(reactivate(frozen, new_gen())),
        Err(ExpertProposalError::UnexpectedResult {
            op: OpKind::Reactivate
        })
    );
    assert_eq!(
        propose(reactivate(failed(), None)),
        Err(ExpertProposalError::MissingResult {
            op: OpKind::Reactivate
        })
    );
    // Only GateFailed seeds, with or without a result.
    for reason in [
        RetiredReason::Pruned,
        RetiredReason::Superseded,
        RetiredReason::InputRevoked,
    ] {
        for result in [None, new_gen()] {
            assert_eq!(
                propose(reactivate(
                    in_state("a", ExpertState::Retired(reason)),
                    result
                )),
                Err(ExpertProposalError::SeedUnavailable {
                    pod: PodId::from("a"),
                    reason
                }),
                "{reason:?}"
            );
        }
    }
    // The result is the same expert at exactly source generation + 1.
    for wrong in [1, 3] {
        assert_eq!(
            propose(reactivate(failed(), Some(spec("a", wrong, slot("a"))))),
            Err(ExpertProposalError::ResultGenerationNotNext {
                pod: PodId::from("a"),
                result: Generation(wrong),
                expected: Some(Generation(2))
            })
        );
    }
    assert_eq!(
        propose(reactivate(failed(), Some(spec("elsewhere", 2, slot("a"))))),
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
fn reactivation_generation_overflow_is_an_error_not_a_panic() {
    let max = target(
        "a",
        u64::MAX,
        token(3),
        ExpertState::Retired(RetiredReason::GateFailed),
    );
    for generation in [u64::MAX, 1] {
        assert_eq!(
            propose(reactivate(
                max.clone(),
                Some(spec("a", generation, token(3)))
            )),
            Err(ExpertProposalError::ResultGenerationNotNext {
                pod: PodId::from("a"),
                result: Generation(generation),
                expected: None
            })
        );
    }
    let almost = target(
        "a",
        u64::MAX - 1,
        token(3),
        ExpertState::Retired(RetiredReason::GateFailed),
    );
    assert!(propose(reactivate(almost, Some(spec("a", u64::MAX, token(3))))).is_ok());
}

#[test]
fn reactivation_restores_the_exact_token_slot() {
    let slot = |index| ExpertLevel::Token { layer: 2, index };
    let failed = target(
        "a",
        1,
        slot(3),
        ExpertState::Retired(RetiredReason::GateFailed),
    );
    assert_eq!(
        propose(reactivate(failed.clone(), Some(spec("a", 2, slot(7))))),
        Err(ExpertProposalError::ResultLevelMismatch {
            pod: PodId::from("a")
        })
    );
    assert!(propose(reactivate(failed.clone(), Some(spec("a", 2, slot(3))))).is_ok());
    assert_eq!(
        propose(reactivate(failed, Some(spec("a", 2, token(4))))),
        Err(ExpertProposalError::ResultLevelMismatch {
            pod: PodId::from("a")
        })
    );
    // Merge and Split keep layer-only compatibility.
    let in_layer = |pod: &str, index| {
        target(
            pod,
            1,
            ExpertLevel::Token { layer: 2, index },
            ExpertState::Serving,
        )
    };
    assert!(propose(StructuralOp::Merge {
        sources: vec![in_layer("a", 1), in_layer("b", 2)],
        result: spec("m", 1, slot(9)),
    })
    .is_ok());
    assert!(propose(StructuralOp::Split {
        source: in_layer("a", 1),
        results: vec![spec("x", 1, slot(5)), spec("y", 1, slot(6))],
    })
    .is_ok());
}

#[test]
fn retire_accepts_only_proposable_reasons() {
    let retire = |reason| {
        propose(StructuralOp::Retire {
            target: serving("a"),
            reason,
        })
    };
    assert!(retire(RetiredReason::Superseded).is_ok());
    assert!(retire(RetiredReason::GateFailed).is_ok());
    for reason in [RetiredReason::Pruned, RetiredReason::InputRevoked] {
        assert_eq!(
            retire(reason),
            Err(ExpertProposalError::ReasonNotProposable { reason })
        );
    }
    // The state table is checked first, then the reason.
    assert!(matches!(
        propose(StructuralOp::Retire {
            target: in_state("a", ExpertState::Retired(RetiredReason::GateFailed)),
            reason: RetiredReason::Pruned
        }),
        Err(ExpertProposalError::WrongTargetState { .. })
    ));
}

#[test]
fn result_slots_are_unique_but_a_sources_slot_may_be_reused() {
    let at = |pod: &str, index| spec(pod, 1, ExpertLevel::Token { layer: 3, index });
    assert_eq!(
        propose(StructuralOp::Split {
            source: serving("a"),
            results: vec![at("x", 4), at("y", 4)]
        }),
        Err(ExpertProposalError::DuplicateResultSlot { layer: 3, index: 4 })
    );
    // `serving("a")` occupies layer 3 index 97; a result may take that slot.
    assert!(propose(StructuralOp::Split {
        source: serving("a"),
        results: vec![at("x", 97), at("y", 1)]
    })
    .is_ok());
    // Segment results have no slot.
    assert!(propose(StructuralOp::Birth {
        result: spec("s", 1, ExpertLevel::Segment)
    })
    .is_ok());
}

#[test]
fn fresh_results_must_be_generation_one() {
    let fresh = Err(ExpertProposalError::FreshResultGeneration {
        pod: PodId::from("x"),
        found: Generation(2),
    });
    assert_eq!(
        propose(StructuralOp::Birth {
            result: spec("x", 2, token(3))
        }),
        fresh
    );
    assert_eq!(
        propose(StructuralOp::Merge {
            sources: vec![serving("a"), serving("b")],
            result: spec("x", 2, token(3))
        }),
        fresh
    );
    assert_eq!(
        propose(StructuralOp::Split {
            source: serving("a"),
            results: vec![spec("x", 2, token(3)), spec("y", 1, token(3))]
        }),
        fresh
    );
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
    // Allowed punctuation.
    assert!(make_spec("proj-1.a_b:c", "Pod.0:x-y_z", 1).is_ok());
    // Interior space, NUL, zero-width space, non-ASCII, slash.
    for bad in [
        "a b",
        "a\0b",
        "\0",
        "a\u{200b}b",
        "\u{200b}",
        "caf\u{e9}",
        "a/b",
    ] {
        assert_eq!(
            make_spec(bad, "pod", 1),
            Err(ExpertIdentityError::NonCanonicalProject),
            "{bad:?}"
        );
        assert_eq!(
            make_spec("proj", bad, 1),
            Err(ExpertIdentityError::NonCanonicalPod),
            "{bad:?}"
        );
    }
    // 128 bytes pass, 129 do not.
    let max = "a".repeat(128);
    assert!(make_spec(&max, &max, 1).is_ok());
    let long = "a".repeat(129);
    assert_eq!(
        make_spec(&long, "pod", 1),
        Err(ExpertIdentityError::NonCanonicalProject)
    );
    assert_eq!(
        make_spec("proj", &long, 1),
        Err(ExpertIdentityError::NonCanonicalPod)
    );
    assert_eq!(
        ExpertTarget::new(init(&long, 1, token(1), ExpertState::Serving)),
        Err(ExpertIdentityError::NonCanonicalPod)
    );
    assert_eq!(
        ProposerRef::new(project(), PodId::from("a b"), Generation(1)),
        Err(ExpertIdentityError::NonCanonicalPod)
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
            reason: RetiredReason::Superseded
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
        "pod id is not 1-128 bytes of ASCII [A-Za-z0-9._:-]"
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

#[test]
fn retire_is_allowed_exactly_where_the_lifecycle_gate_permits_retirement() {
    use ptr_pods::LifecycleGate;
    let gate_allows_retired = |state: ExpertState| {
        state.artifact_lifecycles().iter().any(|from| {
            LifecycleGate::transition(*from, ArtifactLifecycle::Retired, true, true, true, true)
                .is_ok()
        })
    };
    // Fenced maps to Active, so the gate would allow it, but a proposal never retires a fenced
    // expert: that is the runtime revocation path.
    for state in all_states() {
        assert_eq!(
            OpKind::Retire.allows_state(state),
            gate_allows_retired(state) && state != ExpertState::Fenced,
            "{state:?}"
        );
    }
    // Candidates and gated experts are dropped by their gate, not retired by a proposal.
    for state in [
        ExpertState::Candidate,
        ExpertState::Gated,
        ExpertState::Superseded,
    ] {
        assert_eq!(
            propose(StructuralOp::Retire {
                target: in_state("a", state),
                reason: RetiredReason::GateFailed
            }),
            Err(ExpertProposalError::WrongTargetState {
                op: OpKind::Retire,
                pod: PodId::from("a"),
                state
            })
        );
    }
    for state in [ExpertState::Serving, ExpertState::Frozen] {
        assert_eq!(state.artifact_lifecycles(), &[ArtifactLifecycle::Active]);
        assert!(propose(StructuralOp::Retire {
            target: in_state("a", state),
            reason: RetiredReason::Superseded
        })
        .is_ok());
    }
}

#[test]
fn prune_requires_one_base_model() {
    let with_base = |pod: &str, base: BaseModelRef| {
        let mut i = init(pod, 1, slot(pod), ExpertState::Serving);
        i.base_model = base;
        ExpertTarget::new(i).unwrap()
    };
    let prune = |targets| propose(StructuralOp::Prune { targets });
    assert!(prune(vec![with_base("a", base()), with_base("b", base())]).is_ok());
    assert_eq!(
        prune(vec![with_base("a", base()), with_base("b", other_base())]),
        Err(ExpertProposalError::MixedBaseModels)
    );
    // Same digest, different base generation.
    let newer = BaseModelRef::new(*base().digest(), 2).unwrap();
    assert_eq!(
        prune(vec![with_base("a", base()), with_base("b", newer)]),
        Err(ExpertProposalError::MixedBaseModels)
    );
}

#[test]
fn base_model_ref_validates_generation() {
    let digest = BaseModelDigest::new([9; 32]).unwrap();
    assert_eq!(
        BaseModelRef::new(digest, 0),
        Err(ExpertIdentityError::ZeroBaseModelGeneration)
    );
    let r = BaseModelRef::new(digest, 3).unwrap();
    assert_eq!(r.digest(), &digest);
    assert_eq!(r.generation(), 3);
    assert_eq!(
        ExpertIdentityError::ZeroBaseModelGeneration.to_string(),
        "base model generation is zero"
    );
    // The base generation is part of the proposal id.
    let id = |g| {
        let mut i = init("a", 1, token(3), ExpertState::Serving);
        i.base_model = BaseModelRef::new(digest, g).unwrap();
        propose(StructuralOp::Freeze {
            target: ExpertTarget::new(i).unwrap(),
        })
        .unwrap()
        .proposal_id()
    };
    assert_eq!(id(1), id(1));
    assert_ne!(id(1), id(2));
}

#[test]
fn merge_and_prune_reject_two_sources_in_one_token_slot() {
    let at = |pod: &str, index: u16| {
        target(
            pod,
            1,
            ExpertLevel::Token { layer: 3, index },
            ExpertState::Serving,
        )
    };
    let dup = Err(ExpertProposalError::DuplicateSourceSlot { layer: 3, index: 4 });
    assert_eq!(propose(merge(vec![at("a", 4), at("b", 4)])), dup);
    assert_eq!(
        propose(StructuralOp::Prune {
            targets: vec![at("a", 4), at("b", 5), at("c", 4)]
        }),
        dup
    );
    // Different slots, or the same index in another layer (checked as MixedLevels), are fine.
    assert!(propose(merge(vec![at("a", 4), at("b", 5)])).is_ok());
    // The same pod twice stays a DuplicateTarget (checked first).
    assert!(matches!(
        propose(merge(vec![at("a", 4), at("a", 4)])),
        Err(ExpertProposalError::DuplicateTarget { .. })
    ));
    // Base model mismatch is reported before the slot clash.
    let mut i = init(
        "b",
        1,
        ExpertLevel::Token { layer: 3, index: 4 },
        ExpertState::Serving,
    );
    i.base_model = other_base();
    assert_eq!(
        propose(merge(vec![at("a", 4), ExpertTarget::new(i).unwrap()])),
        Err(ExpertProposalError::MixedBaseModels)
    );
    assert_eq!(
        ExpertProposalError::DuplicateSourceSlot { layer: 3, index: 4 }.to_string(),
        "two sources occupy token slot layer 3 index 4"
    );
}

#[test]
fn unfreezing_is_an_in_place_overlay_without_lineage_transition() {
    let frozen = in_state("a", ExpertState::Frozen);
    // Frozen reactivation carries no result, so it creates no generation...
    let unfreeze = propose(reactivate(frozen.clone(), None)).unwrap();
    assert!(!unfreeze.creates_new_generation());
    assert!(unfreeze.op().results().is_empty());
    // ...and a result is rejected.
    assert_eq!(
        propose(reactivate(frozen, Some(spec("a", 2, slot("a"))))),
        Err(ExpertProposalError::UnexpectedResult {
            op: OpKind::Reactivate
        })
    );
    // The frozen expert stays Serving/Active in lineage, so there is nothing to transition.
    assert_eq!(ExpertState::Frozen.adapter_status_name(), "serving");
    assert_eq!(
        ExpertState::Frozen.artifact_lifecycles(),
        ExpertState::Serving.artifact_lifecycles()
    );
    // A retired GateFailed expert needs a result: a new candidate generation.
    let failed = in_state("a", ExpertState::Retired(RetiredReason::GateFailed));
    assert_eq!(
        propose(reactivate(failed.clone(), None)),
        Err(ExpertProposalError::MissingResult {
            op: OpKind::Reactivate
        })
    );
    let reseed = propose(reactivate(failed, Some(spec("a", 2, slot("a"))))).unwrap();
    assert!(reseed.creates_new_generation());
    assert_eq!(reseed.op().results().len(), 1);
}

fn retire(target: ExpertTarget, reason: RetiredReason) -> StructuralOp {
    StructuralOp::Retire { target, reason }
}

#[test]
fn retire_of_a_fenced_operand_is_rejected_for_every_reason() {
    for reason in ALL_REASONS {
        let fenced = retire(in_state("a", ExpertState::Fenced), reason);
        // The fenced check precedes the reason check, so the error does not depend on the reason.
        assert_eq!(
            propose(fenced),
            Err(ExpertProposalError::FencedTarget {
                op: OpKind::Retire,
                pod: PodId::from("a")
            }),
            "{reason:?}"
        );
    }
    // A serving operand still retires with a proposable reason, and InputRevoked stays barred.
    for reason in [RetiredReason::Superseded, RetiredReason::GateFailed] {
        assert!(propose(retire(serving("a"), reason)).is_ok(), "{reason:?}");
    }
    assert_eq!(
        propose(retire(serving("a"), RetiredReason::InputRevoked)),
        Err(ExpertProposalError::ReasonNotProposable {
            reason: RetiredReason::InputRevoked
        })
    );
    assert_eq!(
        ExpertProposalError::FencedTarget {
            op: OpKind::Retire,
            pod: PodId::from("a")
        }
        .to_string(),
        "retire cannot name fenced expert a; only the runtime revocation path may retire it"
    );
}

#[test]
fn fenced_operands_are_rejected_for_every_operation() {
    let fenced = || in_state("a", ExpertState::Fenced);
    let cases = [
        (OpKind::Freeze, StructuralOp::Freeze { target: fenced() }),
        (
            OpKind::Prune,
            StructuralOp::Prune {
                targets: vec![fenced()],
            },
        ),
        (OpKind::Reactivate, reactivate(fenced(), None)),
        (
            OpKind::Split,
            StructuralOp::Split {
                source: fenced(),
                results: vec![result("x"), result("y")],
            },
        ),
        (OpKind::Merge, merge(vec![serving("b"), fenced()])),
    ];
    for (op, proposal) in cases {
        assert_eq!(
            propose(proposal),
            Err(ExpertProposalError::FencedTarget {
                op,
                pod: PodId::from("a")
            }),
            "{op}"
        );
    }
}

#[test]
fn reactivate_from_input_revoked_is_never_a_seed() {
    let revoked = in_state("a", ExpertState::Retired(RetiredReason::InputRevoked));
    for result in [None, Some(spec("a", 2, slot("a")))] {
        assert_eq!(
            propose(reactivate(revoked.clone(), result)),
            Err(ExpertProposalError::SeedUnavailable {
                pod: PodId::from("a"),
                reason: RetiredReason::InputRevoked
            })
        );
    }
}

fn proposal_with(
    op: StructuralOp,
    proposer: ProposerRef,
    observed_at: u64,
    rationale: RationaleDigest,
) -> ExpertProposal {
    ExpertProposal::new(project(), proposer, CommitIndex(observed_at), op, rationale).unwrap()
}

fn merge_with_result(result: ExpertPodSpec) -> StructuralOp {
    StructuralOp::Merge {
        sources: vec![serving("a"), serving("b")],
        result,
    }
}

fn operation_id(op: StructuralOp) -> OperationId {
    proposal_with(op, proposer(), 10, rationale()).operation_id()
}

#[test]
fn operation_id_ignores_result_fields_but_proposal_id_does_not() {
    let base_result = result("merged");
    let other_manifest = ExpertPodSpec::new(
        project(),
        PodId::from("merged"),
        ManifestDigest::new([77; 32]).unwrap(),
        base_result.level(),
        *base_result.base_model(),
        base_result.generation(),
    )
    .unwrap();
    let other_slot = spec(
        "merged",
        1,
        ExpertLevel::Token {
            layer: 3,
            index: 99,
        },
    );
    let a = proposal_with(
        merge_with_result(base_result.clone()),
        proposer(),
        10,
        rationale(),
    );
    for changed in [other_manifest, other_slot, result("renamed")] {
        let b = proposal_with(merge_with_result(changed), proposer(), 10, rationale());
        assert_eq!(a.operation_id(), b.operation_id());
        assert_ne!(a.proposal_id(), b.proposal_id());
    }
    // Same op and result: both ids are stable.
    let again = proposal_with(merge_with_result(base_result), proposer(), 10, rationale());
    assert_eq!(a.operation_id(), again.operation_id());
    assert_eq!(a.proposal_id(), again.proposal_id());
    // The two ids use distinct domains.
    assert_ne!(a.operation_id().as_bytes(), a.proposal_id().as_bytes());
}

#[test]
fn operation_id_changes_with_every_pre_training_field() {
    let base_op = || merge(vec![serving("a"), serving("b")]);
    let reference = operation_id(base_op());

    // Operands: set, order, and each observed target field.
    assert_ne!(
        reference,
        operation_id(merge(vec![serving("a"), serving("c")]))
    );
    assert_ne!(
        reference,
        operation_id(merge(vec![serving("b"), serving("a")]))
    );
    assert_ne!(
        reference,
        operation_id(merge(vec![serving("a"), serving("b"), serving("c")]))
    );
    let tweak = |f: &dyn Fn(&mut ExpertTargetInit)| {
        let mut i = init("b", 1, slot("b"), ExpertState::Serving);
        f(&mut i);
        operation_id(merge(vec![serving("a"), ExpertTarget::new(i).unwrap()]))
    };
    assert_eq!(reference, tweak(&|_| {}));
    assert_ne!(reference, tweak(&|i| i.generation = Generation(2)));
    assert_ne!(
        reference,
        tweak(&|i| i.manifest_digest = ManifestDigest::new([5; 32]).unwrap())
    );
    assert_ne!(reference, tweak(&|i| i.state_version = 2));
    // A merge needs one shared state, so the state is checked through Retire.
    assert_ne!(
        operation_id(retire(serving("a"), RetiredReason::Superseded)),
        operation_id(retire(
            in_state("a", ExpertState::Frozen),
            RetiredReason::Superseded
        ))
    );
    // Op kind: same single operand under different operations.
    let freeze = operation_id(StructuralOp::Freeze {
        target: serving("a"),
    });
    let prune = operation_id(StructuralOp::Prune {
        targets: vec![serving("a")],
    });
    let retire_super = operation_id(retire(serving("a"), RetiredReason::Superseded));
    let retire_failed = operation_id(retire(serving("a"), RetiredReason::GateFailed));
    let ids = [freeze, prune, retire_super, retire_failed];
    for (i, x) in ids.iter().enumerate() {
        for y in &ids[i + 1..] {
            assert_ne!(x, y);
        }
    }
    // Proposer, observed_at, rationale. The project is validated to equal the proposer's.
    let id =
        |proposer, at, rationale| proposal_with(base_op(), proposer, at, rationale).operation_id();
    assert_eq!(reference, id(proposer(), 10, rationale()));
    let other_proposer = ProposerRef::new(project(), PodId::from("meta2"), Generation(1)).unwrap();
    let newer_proposer = ProposerRef::new(project(), PodId::from("meta"), Generation(2)).unwrap();
    assert_ne!(reference, id(other_proposer, 10, rationale()));
    assert_ne!(reference, id(newer_proposer, 10, rationale()));
    assert_ne!(reference, id(proposer(), 11, rationale()));
    assert_ne!(
        reference,
        id(proposer(), 10, RationaleDigest::new([8; 32]).unwrap())
    );
}

#[test]
fn operation_id_changes_with_the_project() {
    let other = ProjectId::from("proj2");
    let target_in = |p: &ProjectId| {
        let mut i = init("a", 1, slot("a"), ExpertState::Serving);
        i.project = p.clone();
        ExpertTarget::new(i).unwrap()
    };
    let id = |p: &ProjectId| {
        let proposer = ProposerRef::new(p.clone(), PodId::from("meta"), Generation(1)).unwrap();
        ExpertProposal::new(
            p.clone(),
            proposer,
            CommitIndex(10),
            StructuralOp::Freeze {
                target: target_in(p),
            },
            rationale(),
        )
        .unwrap()
        .operation_id()
    };
    assert_ne!(id(&project()), id(&other));
}

type Request = Result<ExpertOperationRequest, ExpertProposalError>;

fn request(op: RequestedOp) -> Request {
    ExpertOperationRequest::new(project(), proposer(), CommitIndex(10), op, rationale())
}

#[test]
fn a_request_is_built_and_hashed_without_any_result_spec() {
    let failed = in_state("c", ExpertState::Retired(RetiredReason::GateFailed));
    let ops = [
        RequestedOp::Birth,
        RequestedOp::Merge {
            sources: vec![serving("a"), serving("b")],
        },
        RequestedOp::Split {
            source: serving("a"),
        },
        RequestedOp::Reactivate { target: failed },
    ];
    let mut ids = Vec::new();
    for op in ops {
        let req = request(op).unwrap();
        assert_eq!(req.project(), &project());
        assert_eq!(req.proposer(), &proposer());
        assert_eq!(req.observed_at(), CommitIndex(10));
        assert_eq!(req.rationale(), &rationale());
        ids.push(req.operation_id());
    }
    for (i, x) in ids.iter().enumerate() {
        for y in &ids[i + 1..] {
            assert_ne!(x, y);
        }
    }
}

#[test]
fn operation_request_id_matches_the_proposal_for_every_operation() {
    let failed = || in_state("c", ExpertState::Retired(RetiredReason::GateFailed));
    let ops = [
        StructuralOp::Birth {
            result: result("new"),
        },
        merge(vec![serving("a"), serving("b")]),
        StructuralOp::Split {
            source: serving("a"),
            results: vec![result("x"), result("y")],
        },
        StructuralOp::Freeze {
            target: serving("a"),
        },
        reactivate(in_state("a", ExpertState::Frozen), None),
        reactivate(failed(), Some(spec("c", 2, slot("c")))),
        StructuralOp::Prune {
            targets: vec![serving("a"), serving("b")],
        },
        retire(serving("a"), RetiredReason::GateFailed),
    ];
    for op in ops {
        let proposal = propose(op).unwrap();
        let req = proposal.operation_request();
        assert_eq!(req.operation_id(), proposal.operation_id());
        assert_eq!(req.op().kind(), proposal.op().kind());
        // The derived request is itself valid: rebuilding it through `new` succeeds and is equal.
        let rebuilt = ExpertOperationRequest::new(
            req.project().clone(),
            req.proposer().clone(),
            req.observed_at(),
            req.op().clone(),
            *req.rationale(),
        );
        assert_eq!(rebuilt, Ok(req));
    }
}

#[test]
fn the_request_constructor_rejects_the_same_invalid_operands_as_a_proposal() {
    let fenced = || in_state("a", ExpertState::Fenced);
    let other_project = {
        let mut i = init("a", 1, slot("a"), ExpertState::Serving);
        i.project = ProjectId::from("other");
        ExpertTarget::new(i).unwrap()
    };
    let cases: Vec<(RequestedOp, StructuralOp)> = vec![
        (
            RequestedOp::Retire {
                target: fenced(),
                reason: RetiredReason::GateFailed,
            },
            retire(fenced(), RetiredReason::GateFailed),
        ),
        (
            RequestedOp::Freeze { target: fenced() },
            StructuralOp::Freeze { target: fenced() },
        ),
        (
            RequestedOp::Freeze {
                target: in_state("a", ExpertState::Frozen),
            },
            StructuralOp::Freeze {
                target: in_state("a", ExpertState::Frozen),
            },
        ),
        (
            RequestedOp::Merge {
                sources: vec![serving("a")],
            },
            merge(vec![serving("a")]),
        ),
        (
            RequestedOp::Prune { targets: vec![] },
            StructuralOp::Prune { targets: vec![] },
        ),
        (
            RequestedOp::Freeze {
                target: other_project.clone(),
            },
            StructuralOp::Freeze {
                target: other_project,
            },
        ),
        (
            RequestedOp::Merge {
                sources: vec![serving("a"), in_state("b", ExpertState::Frozen)],
            },
            merge(vec![serving("a"), in_state("b", ExpertState::Frozen)]),
        ),
        (
            RequestedOp::Reactivate {
                target: in_state("a", ExpertState::Retired(RetiredReason::InputRevoked)),
            },
            reactivate(
                in_state("a", ExpertState::Retired(RetiredReason::InputRevoked)),
                Some(spec("a", 2, slot("a"))),
            ),
        ),
        (
            RequestedOp::Retire {
                target: serving("a"),
                reason: RetiredReason::InputRevoked,
            },
            retire(serving("a"), RetiredReason::InputRevoked),
        ),
        (
            RequestedOp::Prune {
                targets: vec![serving("a"), serving("a")],
            },
            StructuralOp::Prune {
                targets: vec![serving("a"), serving("a")],
            },
        ),
    ];
    for (requested, structural) in cases {
        let from_request = request(requested.clone());
        let from_proposal = propose(structural).map(|_| ());
        assert!(from_request.is_err(), "{requested:?}");
        assert_eq!(from_request.map(|_| ()), from_proposal, "{requested:?}");
    }
    assert_eq!(
        request(RequestedOp::Merge {
            sources: vec![serving("a"), in_state("b", ExpertState::Frozen)]
        })
        .map(|_| ()),
        Err(ExpertProposalError::MixedStates)
    );
    // Header rules: the proposer must belong to the request project.
    let foreign =
        ProposerRef::new(ProjectId::from("other"), PodId::from("meta"), Generation(1)).unwrap();
    assert!(matches!(
        ExpertOperationRequest::new(
            project(),
            foreign,
            CommitIndex(10),
            RequestedOp::Birth,
            rationale()
        ),
        Err(ExpertProposalError::ProposerProjectMismatch { .. })
    ));
}
