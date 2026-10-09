//! The L3 structural-proposal policy: each operator triggers at exactly its threshold and not
//! below, the safety rules hold (last clearer, allowed states, caps), the output is
//! deterministic and bounded, and a generated proposal gets no authority: it is admitted only on
//! a passing gate report bound to exactly that proposal.

use ptr_config::PtrConfig;
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertProposal, ExpertState,
    ExpertTarget, ExpertTargetInit, ManifestDigest, OpKind, ProposalParts, ProposerRef,
    RetiredReason,
};
use ptr_router::FindingCode;
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, unroutable_delta,
    BirthContext, BirthSite, CodeWindowStats, EnabledOps, ExpertAdmissionError,
    ExpertAdmissionGrant, ObservationLimits, PodCodeStats, RegisteredExpert,
    StructureCompletionError, StructureConfigError, StructureObservation,
    StructureObservationError, StructurePolicy, StructurePolicyConfig, StructureProposal,
    StructureRationale, StructureReport, SubspaceOverlap, Suppression, WindowObservation,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};
use std::collections::{BTreeMap, BTreeSet};

mod expert_support;

use expert_support::{binding, prime, requirement, EVAL_MANIFEST};

const LAYER: u16 = 3;

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap()
}

fn proposer() -> ProposerRef {
    ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap()
}

fn code(name: &str) -> FindingCode {
    FindingCode::from(name)
}

fn pod(name: &str) -> PodId {
    PodId::from(name)
}

fn token(index: u16) -> ExpertLevel {
    ExpertLevel::Token {
        layer: LAYER,
        index,
    }
}

fn target(name: &str, level: ExpertLevel, state: ExpertState) -> ExpertTarget {
    let mut digest = [1u8; 32];
    digest[0] = name.as_bytes()[0];
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: pod(name),
        generation: Generation(1),
        level,
        base_model: base(),
        manifest_digest: ManifestDigest::new(digest).unwrap(),
        state,
        state_version: 1,
    })
    .unwrap()
}

fn expert(name: &str, level: ExpertLevel, state: ExpertState, codes: &[&str]) -> RegisteredExpert {
    RegisteredExpert::new(
        target(name, level, state),
        codes.iter().map(|c| code(c)).collect(),
    )
}

/// A serving Token expert in `LAYER` at the slot `index`.
fn serving(name: &str, index: u16, codes: &[&str]) -> RegisteredExpert {
    expert(name, token(index), ExpertState::Serving, codes)
}

/// Thresholds with exactly representable values, so a boundary can be hit exactly.
fn cfg() -> StructurePolicyConfig {
    StructurePolicyConfig {
        min_unroutable: 4,
        merge_clear_rate_margin: 0.25,
        merge_overlap_margin: 0.25,
        merge_min_routed: 4,
        prune_usage_floor: 3,
        freeze_windows: 2,
        freeze_max_usage_variance: 1.0,
        freeze_max_clear_rate_variance: 0.015625,
        freeze_min_usage: 5,
        max_proposals_per_window: 16,
        max_proposals_per_expert: 1,
        birth_windows: 1,
        max_births_per_window: 16,
        merge_min_clear_rate: 0.25,
        prune_windows: 1,
        prune_min_age_commits: 0,
        backup_min_cleared: 1,
        backup_clear_windows: 1,
        freeze_min_clear_rate: 0.25,
        ..StructurePolicyConfig::default()
    }
}

fn only(ops: EnabledOps) -> StructurePolicyConfig {
    StructurePolicyConfig { ops, ..cfg() }
}

const NONE: EnabledOps = EnabledOps {
    birth: false,
    merge: false,
    prune: false,
    freeze: false,
    reactivate: false,
};

fn just_birth() -> StructurePolicyConfig {
    only(EnabledOps {
        birth: true,
        ..NONE
    })
}

fn just_reactivate() -> StructurePolicyConfig {
    only(EnabledOps {
        reactivate: true,
        ..NONE
    })
}

fn just_prune() -> StructurePolicyConfig {
    only(EnabledOps {
        prune: true,
        ..NONE
    })
}

fn just_merge() -> StructurePolicyConfig {
    only(EnabledOps {
        merge: true,
        ..NONE
    })
}

fn just_freeze() -> StructurePolicyConfig {
    only(EnabledOps {
        freeze: true,
        ..NONE
    })
}

fn policy(config: StructurePolicyConfig) -> StructurePolicy {
    StructurePolicy::new(config).unwrap()
}

fn window() -> WindowObservation {
    WindowObservation::default()
}

fn with_code(
    mut w: WindowObservation,
    name: &str,
    unroutable: u64,
    pods: &[(&str, u64, u64)],
) -> WindowObservation {
    w.codes.insert(
        code(name),
        CodeWindowStats {
            unroutable,
            pods: pods
                .iter()
                .map(|(p, routed, cleared)| {
                    (
                        pod(p),
                        PodCodeStats {
                            routed: *routed,
                            cleared: *cleared,
                        },
                    )
                })
                .collect(),
        },
    );
    w
}

fn with_usage(mut w: WindowObservation, name: &str, usage: u64) -> WindowObservation {
    w.usage.insert(pod(name), usage);
    w
}

fn observation(
    windows: Vec<WindowObservation>,
    experts: Vec<RegisteredExpert>,
) -> StructureObservation {
    StructureObservation {
        project: project(),
        proposer: proposer(),
        observed_at: CommitIndex(1),
        windows,
        experts,
        aimer: BTreeMap::new(),
        overlaps: Vec::new(),
        birth: Some(BirthContext {
            base_model: base(),
            site: BirthSite::Segment,
        }),
        outstanding: BTreeSet::new(),
    }
}

fn propose(config: StructurePolicyConfig, obs: &StructureObservation) -> StructureReport {
    policy(config).propose(obs).unwrap()
}

fn kinds(report: &StructureReport) -> Vec<OpKind> {
    report.proposals().iter().map(|p| p.kind()).collect()
}

/// The manifest digest a training run produced, as the host would hold it.
fn trained(seed: u8) -> ManifestDigest {
    ManifestDigest::new([seed; 32]).unwrap()
}

/// The proposal of a generated request: completed with a trained digest when it creates an
/// expert, ready as it is otherwise.
fn built(generated: &StructureProposal) -> ExpertProposal {
    if generated.needs_training() {
        generated.complete(trained(0x42)).unwrap()
    } else {
        generated.ready().unwrap()
    }
}

fn suppressed(report: &StructureReport, why: Suppression) -> u64 {
    report.suppressed().get(&why).copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------

#[test]
fn the_default_and_the_test_configuration_are_valid() {
    assert_eq!(StructurePolicyConfig::default().validate(), Ok(()));
    assert_eq!(cfg().validate(), Ok(()));
}

#[test]
fn an_invalid_configuration_is_refused() {
    let cases: Vec<(StructurePolicyConfig, StructureConfigError)> = vec![
        (only(NONE), StructureConfigError::NoOperatorEnabled),
        (
            StructurePolicyConfig {
                min_unroutable: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "min_unroutable",
            },
        ),
        (
            StructurePolicyConfig {
                max_proposals_per_window: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "max_proposals_per_window",
            },
        ),
        (
            StructurePolicyConfig {
                max_proposals_per_expert: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "max_proposals_per_expert",
            },
        ),
        (
            StructurePolicyConfig {
                limits: ObservationLimits {
                    max_experts: 0,
                    ..ObservationLimits::default()
                },
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "limits.max_experts",
            },
        ),
        (
            StructurePolicyConfig {
                merge_clear_rate_margin: f64::NAN,
                ..cfg()
            },
            StructureConfigError::BadFraction {
                field: "merge_clear_rate_margin",
            },
        ),
        (
            StructurePolicyConfig {
                merge_clear_rate_margin: 1.5,
                ..cfg()
            },
            StructureConfigError::BadFraction {
                field: "merge_clear_rate_margin",
            },
        ),
        (
            // A zero margin would merge at chance.
            StructurePolicyConfig {
                merge_overlap_margin: 0.0,
                ..cfg()
            },
            StructureConfigError::BadFraction {
                field: "merge_overlap_margin",
            },
        ),
        (
            StructurePolicyConfig {
                freeze_max_usage_variance: -1.0,
                ..cfg()
            },
            StructureConfigError::BadVariance {
                field: "freeze_max_usage_variance",
            },
        ),
        (
            StructurePolicyConfig {
                freeze_max_clear_rate_variance: f64::INFINITY,
                ..cfg()
            },
            StructureConfigError::BadVariance {
                field: "freeze_max_clear_rate_variance",
            },
        ),
        (
            StructurePolicyConfig {
                freeze_windows: 1,
                ..cfg()
            },
            StructureConfigError::FreezeNeedsTwoWindows,
        ),
        (
            StructurePolicyConfig {
                freeze_windows: 5,
                limits: ObservationLimits {
                    max_windows: 4,
                    ..ObservationLimits::default()
                },
                ..cfg()
            },
            StructureConfigError::HistoryShorterThanFreezeWindows,
        ),
        (
            StructurePolicyConfig {
                prune_usage_floor: 6,
                freeze_min_usage: 5,
                ..cfg()
            },
            StructureConfigError::FreezeFloorBelowPruneFloor,
        ),
    ];
    for (config, expected) in cases {
        assert_eq!(config.validate(), Err(expected), "{expected:?}");
        assert_eq!(StructurePolicy::new(config).err(), Some(expected));
    }
}

#[test]
fn the_configuration_digest_follows_every_threshold() {
    let a = cfg();
    assert_eq!(a.digest(), cfg().digest());
    for changed in [
        StructurePolicyConfig {
            min_unroutable: 5,
            ..cfg()
        },
        StructurePolicyConfig {
            merge_overlap_margin: 0.5,
            ..cfg()
        },
        StructurePolicyConfig {
            max_proposals_per_window: 3,
            ..cfg()
        },
        only(EnabledOps {
            birth: false,
            ..EnabledOps::ALL
        }),
    ] {
        assert_ne!(a.digest(), changed.digest());
    }
}

// ---------------------------------------------------------------------------------------------
// Birth
// ---------------------------------------------------------------------------------------------

fn birth_obs(unroutable: u64, pods: &[(&str, u64, u64)]) -> StructureObservation {
    observation(
        vec![with_code(window(), "target", unroutable, pods)],
        Vec::new(),
    )
}

#[test]
fn birth_triggers_at_min_unroutable_and_not_below() {
    for (unroutable, births) in [(0, 0), (3, 0), (4, 1), (5, 1)] {
        let report = propose(just_birth(), &birth_obs(unroutable, &[]));
        assert_eq!(
            report.count(OpKind::Birth),
            births,
            "unroutable {unroutable}"
        );
    }
    let report = propose(just_birth(), &birth_obs(4, &[]));
    let proposal = report.proposals()[0].request();
    assert_eq!(proposal.op().kind(), OpKind::Birth);
    let result = proposal.op().results()[0].clone();
    assert_eq!(result.generation(), Generation(1));
    assert_eq!(result.level(), ExpertLevel::Segment);
    assert!(matches!(
        report.proposals()[0].rationale(),
        StructureRationale::Birth {
            unroutable: 4,
            routed_uncleared: 0,
            ..
        }
    ));
}

#[test]
fn birth_also_triggers_on_a_code_routed_but_never_cleared() {
    // 4 routed to a registered Pod, none cleared: the Pod does not repair the code.
    let report = propose(just_birth(), &birth_obs(0, &[("a", 4, 0)]));
    assert_eq!(report.count(OpKind::Birth), 1);
    // One clear is enough to show some Pod repairs it.
    let report = propose(just_birth(), &birth_obs(0, &[("a", 4, 1)]));
    assert_eq!(report.count(OpKind::Birth), 0);
    // Routed but short of the count.
    let report = propose(just_birth(), &birth_obs(0, &[("a", 3, 0)]));
    assert_eq!(report.count(OpKind::Birth), 0);
    // Unroutable and uncleared reports add up.
    let report = propose(just_birth(), &birth_obs(1, &[("a", 3, 0)]));
    assert_eq!(report.count(OpKind::Birth), 1);
}

#[test]
fn birth_needs_a_context_a_free_pod_id_and_a_free_slot() {
    let mut obs = birth_obs(4, &[]);
    obs.birth = None;
    let report = propose(just_birth(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::NoBirthContext), 1);

    // A pruned repairer of the code leaves its id behind: the next birth takes the next number.
    let first = propose(just_birth(), &birth_obs(4, &[]));
    let first_pod = first.proposals()[0].request().op().results()[0]
        .pod()
        .clone();
    assert!(first_pod.0.ends_with(".1"), "{first_pod}");
    let mut obs = birth_obs(4, &[]);
    obs.experts.push(RegisteredExpert::new(
        target(
            &first_pod.0,
            token(0),
            ExpertState::Retired(RetiredReason::Pruned),
        ),
        BTreeSet::new(),
    ));
    let report = propose(just_birth(), &obs);
    let second = report.proposals()[0].request().op().results()[0]
        .pod()
        .clone();
    assert!(second.0.ends_with(".2"), "{second}");
    assert_eq!(
        second.0.trim_end_matches(".2"),
        first_pod.0.trim_end_matches(".1")
    );

    // The number it would take is taken by an expert of a gap in the history.
    let stem = first_pod.0.trim_end_matches(".1").to_owned();
    let mut obs = birth_obs(4, &[]);
    for n in [1, 3] {
        obs.experts.push(RegisteredExpert::new(
            target(
                &format!("{stem}.{n}"),
                token(n),
                ExpertState::Retired(RetiredReason::Pruned),
            ),
            BTreeSet::new(),
        ));
    }
    let report = propose(just_birth(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::ResultPodExists), 1);

    // An earlier birth of the code still being decided is in flight: no second request.
    let mut obs = birth_obs(4, &[]);
    obs.experts.push(RegisteredExpert::new(
        target(&first_pod.0, token(0), ExpertState::Candidate),
        BTreeSet::new(),
    ));
    let report = propose(just_birth(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::CoolDown), 1);

    // A request for the code the host still holds outstanding is not repeated.
    let mut obs = birth_obs(4, &[]);
    obs.outstanding.insert(code("target"));
    let report = propose(just_birth(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::CoolDown), 1);
}

#[test]
fn two_codes_the_sanitizer_would_merge_still_get_distinct_pods() {
    let w = with_code(with_code(window(), "a b", 4, &[]), "a_b", 4, &[]);
    let report = propose(just_birth(), &observation(vec![w], Vec::new()));
    let pods: BTreeSet<PodId> = report
        .proposals()
        .iter()
        .map(|p| p.request().op().results()[0].pod().clone())
        .collect();
    assert_eq!(pods.len(), 2);
    assert!(pods.iter().all(|p| p.0.len() <= 128));
}

#[test]
fn a_birth_needs_the_trigger_in_each_of_the_last_birth_windows() {
    let two = StructurePolicyConfig {
        birth_windows: 2,
        ..just_birth()
    };
    let w = |unroutable| with_code(window(), "target", unroutable, &[]);
    let births = |first, second| {
        propose(
            two.clone(),
            &observation(vec![w(first), w(second)], Vec::new()),
        )
        .count(OpKind::Birth)
    };
    assert_eq!(births(4, 4), 1);
    assert_eq!(births(3, 4), 0, "one window short of the count");
    assert_eq!(births(4, 3), 0);
    assert_eq!(births(0, 9), 0, "a burst in the latest window only");
    // One window of history is not enough for two windows.
    assert_eq!(
        propose(two, &observation(vec![w(9)], Vec::new())).count(OpKind::Birth),
        0
    );
}

#[test]
fn births_are_ranked_by_evidence_and_sub_capped_so_other_operators_keep_room() {
    let mut w = window();
    for (name, unroutable) in [("low", 4), ("high", 9), ("mid", 6)] {
        w = with_code(w, name, unroutable, &[]);
    }
    let config = StructurePolicyConfig {
        max_births_per_window: 2,
        ..just_birth()
    };
    let report = propose(config, &observation(vec![w], Vec::new()));
    let codes: Vec<&str> = report
        .proposals()
        .iter()
        .map(|p| match p.rationale() {
            StructureRationale::Birth { code, .. } => code.as_str(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(codes, vec!["high", "mid"]);
    assert_eq!(suppressed(&report, Suppression::BirthCap), 1);

    // With the whole window cap taken by junk births, a Prune would starve; the sub-cap leaves
    // room for it.
    let mut w = with_usage(window(), "dead", 0);
    for n in 0..20 {
        w = with_code(w, &format!("junk{n}"), 9, &[]);
    }
    w = with_code(w, "kept", 0, &[("live", 4, 4)]);
    let mut obs = observation(
        vec![w],
        vec![serving("dead", 0, &[]), serving("live", 1, &["kept"])],
    );
    obs.aimer = BTreeMap::from([(pod("dead"), 0.9), (pod("live"), 0.1)]);
    let config = StructurePolicyConfig {
        max_proposals_per_window: 4,
        max_births_per_window: 2,
        ..only(EnabledOps::ALL)
    };
    let report = propose(config, &obs);
    assert_eq!(report.count(OpKind::Birth), 2);
    assert_eq!(report.count(OpKind::Prune), 1);
}

#[test]
fn a_token_birth_takes_the_lowest_free_slot_and_runs_out_of_slots() {
    let mut obs = observation(
        vec![with_code(
            with_code(window(), "alpha", 4, &[]),
            "beta",
            4,
            &[],
        )],
        vec![serving("x", 0, &[]), serving("y", 2, &[])],
    );
    obs.birth = Some(BirthContext {
        base_model: base(),
        site: BirthSite::Token {
            layer: LAYER,
            slots: 4,
        },
    });
    let report = propose(just_birth(), &obs);
    let slots: Vec<ExpertLevel> = report
        .proposals()
        .iter()
        .map(|p| p.request().op().results()[0].level())
        .collect();
    assert_eq!(slots, vec![token(1), token(3)]);

    // Only slot 1 is free for two births: the second finds none.
    obs.birth = Some(BirthContext {
        base_model: base(),
        site: BirthSite::Token {
            layer: LAYER,
            slots: 3,
        },
    });
    let report = propose(just_birth(), &obs);
    assert_eq!(report.count(OpKind::Birth), 1);
    assert_eq!(suppressed(&report, Suppression::NoFreeSlot), 1);
}

#[test]
fn unroutable_delta_counts_growth_and_never_wraps() {
    let before = BTreeMap::from([(code("a"), 5), (code("b"), 9)]);
    let now = BTreeMap::from([
        (code("a"), 8),
        (code("b"), 2),
        (code("c"), 3),
        (code("d"), 0),
    ]);
    assert_eq!(
        unroutable_delta(&before, &now),
        BTreeMap::from([(code("a"), 3), (code("c"), 3)])
    );
}

// ---------------------------------------------------------------------------------------------
// Reactivate
// ---------------------------------------------------------------------------------------------

fn retired(name: &str, index: u16, reason: RetiredReason) -> RegisteredExpert {
    expert(
        name,
        token(index),
        ExpertState::Retired(reason),
        &["target"],
    )
}

#[test]
fn reactivate_replaces_a_birth_for_a_code_whose_expert_was_retired_by_the_gate() {
    let obs = |unroutable| {
        observation(
            vec![with_code(window(), "target", unroutable, &[])],
            vec![retired("old", 7, RetiredReason::GateFailed)],
        )
    };
    let all = EnabledOps::ALL;
    // At the threshold: one Reactivate and no Birth.
    let report = propose(only(all), &obs(4));
    assert_eq!(kinds(&report), vec![OpKind::Reactivate]);
    let proposal = report.proposals()[0].request();
    let result = proposal.op().results()[0].clone();
    assert_eq!(result.pod(), &pod("old"));
    assert_eq!(result.generation(), Generation(2));
    assert_eq!(result.level(), token(7));
    // Below it: nothing.
    assert!(propose(only(all), &obs(3)).proposals().is_empty());
    // Disabled: the code gets a Birth instead.
    let report = propose(
        only(EnabledOps {
            reactivate: false,
            ..all
        }),
        &obs(4),
    );
    assert_eq!(kinds(&report), vec![OpKind::Birth]);
    // Only the policy proposes; the reactivation alone is available as an ablation arm too.
    assert_eq!(
        kinds(&propose(just_reactivate(), &obs(4))),
        vec![OpKind::Reactivate]
    );
}

#[test]
fn only_a_gate_failed_expert_is_a_reactivation_seed() {
    for reason in [
        RetiredReason::Pruned,
        RetiredReason::Superseded,
        RetiredReason::InputRevoked,
    ] {
        let obs = observation(
            vec![with_code(window(), "target", 4, &[])],
            vec![retired("old", 7, reason)],
        );
        let report = propose(only(EnabledOps::ALL), &obs);
        assert_eq!(kinds(&report), vec![OpKind::Birth], "{reason:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Prune
// ---------------------------------------------------------------------------------------------

fn prune_obs(experts: Vec<RegisteredExpert>, usage: &[(&str, u64)]) -> StructureObservation {
    let mut w = window();
    for (name, used) in usage {
        w = with_usage(w, name, *used);
    }
    let mut obs = observation(vec![w], experts);
    obs.aimer = obs
        .experts
        .iter()
        .enumerate()
        .map(|(i, e)| (e.target().pod().clone(), 0.5 + i as f64 / 10.0))
        .collect();
    obs
}

#[test]
fn prune_triggers_below_the_usage_floor_and_not_at_it() {
    for (usage, prunes) in [(0, 1), (2, 1), (3, 0), (4, 0)] {
        let obs = prune_obs(vec![serving("a", 0, &[])], &[("a", usage)]);
        assert_eq!(
            propose(just_prune(), &obs).count(OpKind::Prune),
            prunes,
            "usage {usage}"
        );
    }
}

#[test]
fn prune_needs_an_aimer_score_and_a_serving_token_expert() {
    let mut obs = prune_obs(vec![serving("a", 0, &[])], &[("a", 0)]);
    obs.aimer.clear();
    let report = propose(just_prune(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::MissingAimerScore), 1);

    let segment = expert("s", ExpertLevel::Segment, ExpertState::Serving, &[]);
    let report = propose(just_prune(), &prune_obs(vec![segment], &[("s", 0)]));
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::NotTokenLevel), 1);

    let frozen = expert("f", token(0), ExpertState::Frozen, &[]);
    let report = propose(just_prune(), &prune_obs(vec![frozen], &[("f", 0)]));
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::StateNotAllowed), 1);
}

/// `obs` with the clears of `code` in its first window.
fn with_clears(
    mut obs: StructureObservation,
    name: &str,
    pods: &[(&str, u64, u64)],
) -> StructureObservation {
    obs.windows[0] = with_code(obs.windows[0].clone(), name, 0, pods);
    obs
}

#[test]
fn prune_never_removes_the_last_clearer_of_a_code() {
    // One registered clearer, unused: kept.
    let obs = prune_obs(vec![serving("a", 0, &["target"])], &[("a", 0)]);
    let report = propose(just_prune(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::LastClearer), 1);

    // Three unused clearers of one code, all demonstrably clearing it: AIMER order x, z, y; y
    // would be the last one left.
    let obs = prune_obs(
        vec![
            serving("x", 0, &["target"]),
            serving("y", 1, &["target"]),
            serving("z", 2, &["target"]),
        ],
        &[("x", 0), ("y", 0), ("z", 0)],
    );
    let mut obs = with_clears(obs, "target", &[("x", 4, 2), ("y", 4, 2), ("z", 4, 2)]);
    obs.aimer = BTreeMap::from([(pod("x"), 0.9), (pod("y"), 0.5), (pod("z"), 0.7)]);
    let report = propose(just_prune(), &obs);
    let pruned: Vec<PodId> = report
        .proposals()
        .iter()
        .map(|p| p.request().op().sources()[0].pod().clone())
        .collect();
    assert_eq!(pruned, vec![pod("x"), pod("z")]);
    assert_eq!(suppressed(&report, Suppression::LastClearer), 1);

    // A Frozen expert that clears the code is a backup that can be routed to.
    let obs = prune_obs(
        vec![
            serving("a", 0, &["target"]),
            expert("b", token(1), ExpertState::Frozen, &["target"]),
        ],
        &[("a", 0), ("b", 9)],
    );
    let obs = with_clears(obs, "target", &[("a", 4, 2), ("b", 4, 2)]);
    assert_eq!(propose(just_prune(), &obs).count(OpKind::Prune), 1);

    // A retired expert is not.
    let obs = prune_obs(
        vec![
            serving("a", 0, &["target"]),
            expert(
                "b",
                token(1),
                ExpertState::Retired(RetiredReason::GateFailed),
                &["target"],
            ),
        ],
        &[("a", 0)],
    );
    let obs = with_clears(obs, "target", &[("a", 4, 2), ("b", 4, 2)]);
    assert_eq!(propose(just_prune(), &obs).count(OpKind::Prune), 0);
}

#[test]
fn a_competitor_that_only_claims_a_code_does_not_make_its_clearer_prunable() {
    // `a` clears the code; `b` is registered for it, serving, and never clears it.
    let obs = prune_obs(
        vec![serving("a", 0, &["target"]), serving("b", 1, &["target"])],
        &[("a", 0), ("b", 9)],
    );
    let claim_only = with_clears(obs.clone(), "target", &[("a", 4, 2), ("b", 4, 0)]);
    let report = propose(just_prune(), &claim_only);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::LastClearer), 1);
    // Not routed to at all is the same.
    let unrouted = with_clears(obs.clone(), "target", &[("a", 4, 2)]);
    assert!(propose(just_prune(), &unrouted).proposals().is_empty());
    // With no statistics about the code at all nobody is demonstrated.
    assert!(propose(just_prune(), &obs).proposals().is_empty());

    // A demonstrated backup at exactly backup_min_cleared protects; one below does not.
    let two = StructurePolicyConfig {
        backup_min_cleared: 2,
        ..just_prune()
    };
    let at = with_clears(obs.clone(), "target", &[("a", 4, 2), ("b", 4, 2)]);
    assert_eq!(propose(two.clone(), &at).count(OpKind::Prune), 1);
    let below = with_clears(obs, "target", &[("a", 4, 2), ("b", 4, 1)]);
    assert_eq!(propose(two, &below).count(OpKind::Prune), 0);
}

#[test]
fn a_backup_must_have_cleared_within_the_backup_windows() {
    // The backup cleared in the older window only.
    let mut obs = prune_obs(
        vec![serving("a", 0, &["target"]), serving("b", 1, &["target"])],
        &[("a", 0), ("b", 9)],
    );
    let old = with_code(window(), "target", 0, &[("a", 4, 2), ("b", 4, 2)]);
    let recent = with_usage(with_usage(window(), "a", 0), "b", 9);
    let recent = with_code(recent, "target", 0, &[("a", 4, 2), ("b", 4, 0)]);
    obs.windows = vec![old, recent];
    let one_window = just_prune();
    assert_eq!(propose(one_window.clone(), &obs).count(OpKind::Prune), 0);
    let two_windows = StructurePolicyConfig {
        backup_clear_windows: 2,
        ..one_window
    };
    assert_eq!(propose(two_windows, &obs).count(OpKind::Prune), 1);
}

#[test]
fn prune_needs_low_usage_in_every_prune_window_and_a_mature_expert() {
    let two = StructurePolicyConfig {
        prune_windows: 2,
        ..just_prune()
    };
    let at = |usage_old, usage_new| {
        let mut obs = prune_obs(vec![serving("a", 0, &[])], &[]);
        obs.windows = vec![
            with_usage(window(), "a", usage_old),
            with_usage(window(), "a", usage_new),
        ];
        obs
    };
    assert_eq!(propose(two.clone(), &at(0, 0)).count(OpKind::Prune), 1);
    assert_eq!(
        propose(two.clone(), &at(3, 0)).count(OpKind::Prune),
        0,
        "busy before"
    );
    assert_eq!(
        propose(two.clone(), &at(0, 3)).count(OpKind::Prune),
        0,
        "busy now"
    );
    // One window of history is not two.
    let mut short = at(0, 0);
    short.windows.remove(0);
    assert_eq!(propose(two, &short).count(OpKind::Prune), 0);

    // Maturity: observed_at is 1; the expert changed at commit 0, so it is 1 commit old.
    let mature = |age: u64, known: bool| {
        let mut obs = prune_obs(vec![serving("a", 0, &[])], &[("a", 0)]);
        obs.observed_at = CommitIndex(100);
        if known {
            obs.experts[0] = obs.experts[0]
                .clone()
                .with_last_changed_at(CommitIndex(100 - age));
        }
        obs
    };
    let aged = StructurePolicyConfig {
        prune_min_age_commits: 10,
        ..just_prune()
    };
    assert_eq!(
        propose(aged.clone(), &mature(10, true)).count(OpKind::Prune),
        1
    );
    let young = propose(aged.clone(), &mature(9, true));
    assert_eq!(young.count(OpKind::Prune), 0);
    assert_eq!(suppressed(&young, Suppression::TooYoung), 1);
    // An unknown age is too young while a minimum age is required.
    assert_eq!(propose(aged, &mature(10, false)).count(OpKind::Prune), 0);
}

#[test]
fn prune_counts_a_code_the_expert_cleared_even_when_it_is_not_registered_for_it() {
    let mut obs = prune_obs(vec![serving("a", 0, &[])], &[("a", 0)]);
    obs.windows[0] = with_code(obs.windows[0].clone(), "target", 0, &[("a", 5, 5)]);
    let report = propose(just_prune(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::LastClearer), 1);
}

#[test]
fn prune_is_ranked_by_aimer_and_cut_by_the_window_cap() {
    let mut obs = prune_obs(
        vec![
            serving("a", 0, &[]),
            serving("b", 1, &[]),
            serving("c", 2, &[]),
        ],
        &[("a", 0), ("b", 0), ("c", 0)],
    );
    obs.aimer = BTreeMap::from([(pod("a"), 0.2), (pod("b"), 0.8), (pod("c"), 0.5)]);
    let capped = StructurePolicyConfig {
        max_proposals_per_window: 2,
        ..just_prune()
    };
    let report = propose(capped, &obs);
    let order: Vec<PodId> = report
        .proposals()
        .iter()
        .map(|p| p.request().op().sources()[0].pod().clone())
        .collect();
    assert_eq!(order, vec![pod("b"), pod("c")]);
    assert_eq!(suppressed(&report, Suppression::WindowCap), 1);
}

// ---------------------------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------------------------

fn merge_obs(rate_a: u64, rate_b: u64, overlap: f64, chance: f64) -> StructureObservation {
    let w = with_code(window(), "target", 0, &[("a", 4, rate_a), ("b", 4, rate_b)]);
    let mut obs = observation(
        vec![w],
        vec![serving("a", 0, &["target"]), serving("b", 1, &["target"])],
    );
    obs.overlaps = vec![SubspaceOverlap {
        first: pod("a"),
        second: pod("b"),
        overlap,
        chance,
    }];
    obs
}

#[test]
fn merge_triggers_at_both_margins_and_not_below_either() {
    // Clear rates 2/4 and 3/4 differ by exactly the margin; overlap exceeds chance by exactly
    // the margin.
    let report = propose(just_merge(), &merge_obs(2, 3, 0.5, 0.25));
    assert_eq!(kinds(&report), vec![OpKind::Merge]);
    // Rates 2/4 and 4/4 differ by more than the margin.
    assert!(propose(just_merge(), &merge_obs(2, 4, 0.5, 0.25))
        .proposals()
        .is_empty());
    // Overlap just below chance plus the margin.
    assert!(propose(just_merge(), &merge_obs(2, 3, 0.4999, 0.25))
        .proposals()
        .is_empty());
    // Overlap at chance carries no evidence.
    assert!(propose(just_merge(), &merge_obs(2, 3, 0.25, 0.25))
        .proposals()
        .is_empty());
    // No overlap measurement: no merge.
    let mut obs = merge_obs(2, 3, 0.5, 0.25);
    obs.overlaps.clear();
    assert!(propose(just_merge(), &obs).proposals().is_empty());
}

#[test]
fn a_merge_takes_the_first_sources_slot_and_is_ordered_by_pod() {
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    // The caller may name the pair either way round.
    obs.overlaps[0].first = pod("b");
    obs.overlaps[0].second = pod("a");
    let report = propose(just_merge(), &obs);
    let proposal = report.proposals()[0].request();
    let sources: Vec<PodId> = proposal
        .op()
        .sources()
        .iter()
        .map(|s| s.pod().clone())
        .collect();
    assert_eq!(sources, vec![pod("a"), pod("b")]);
    let result = proposal.op().results()[0].clone();
    assert_eq!(result.level(), token(0));
    assert_eq!(result.generation(), Generation(1));
    assert!(matches!(
        report.proposals()[0].rationale(),
        StructureRationale::Merge { overlap_excess, max_clear_rate_gap, .. }
            if *overlap_excess == 0.5 && *max_clear_rate_gap == 0.0
    ));
}

#[test]
fn merge_needs_the_same_codes_enough_samples_and_compatible_experts() {
    // Different code sets.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    obs.experts[1] = serving("b", 1, &["target", "other"]);
    let report = propose(just_merge(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::Incompatible), 1);

    // Too few routed reports for the rate to count.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    obs.windows[0] = with_code(window(), "target", 0, &[("a", 3, 2), ("b", 4, 2)]);
    assert!(propose(just_merge(), &obs).proposals().is_empty());

    // Another layer.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    obs.experts[1] = expert(
        "b",
        ExpertLevel::Token { layer: 9, index: 1 },
        ExpertState::Serving,
        &["target"],
    );
    let report = propose(just_merge(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::Incompatible), 1);

    // A Segment expert.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    obs.experts[1] = expert("b", ExpertLevel::Segment, ExpertState::Serving, &["target"]);
    let report = propose(just_merge(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::NotTokenLevel), 1);

    // Sources in different states.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    obs.experts[1] = expert("b", token(1), ExpertState::Frozen, &["target"]);
    let report = propose(just_merge(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(suppressed(&report, Suppression::StateNotAllowed), 1);

    // A fenced source.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    obs.experts[0] = expert("a", token(0), ExpertState::Fenced, &["target"]);
    obs.experts[1] = expert("b", token(1), ExpertState::Fenced, &["target"]);
    assert!(propose(just_merge(), &obs).proposals().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Freeze
// ---------------------------------------------------------------------------------------------

/// Two windows of one expert: usage and cleared out of 4 routed.
fn freeze_obs(first: (u64, u64), second: (u64, u64)) -> StructureObservation {
    let w = |(usage, cleared): (u64, u64)| {
        with_usage(
            with_code(window(), "target", 0, &[("a", 4, cleared)]),
            "a",
            usage,
        )
    };
    observation(
        vec![w(first), w(second)],
        vec![serving("a", 0, &["target"])],
    )
}

#[test]
fn freeze_triggers_at_the_variance_bounds_and_not_above() {
    // Usage 8 and 10 have variance exactly 1; clear rates 2/4 and 3/4 exactly 0.015625.
    let report = propose(just_freeze(), &freeze_obs((8, 2), (10, 3)));
    assert_eq!(kinds(&report), vec![OpKind::Freeze]);
    assert!(matches!(
        report.proposals()[0].rationale(),
        StructureRationale::Freeze { usage_variance, clear_rate_variance, windows: 2, .. }
            if *usage_variance == 1.0 && *clear_rate_variance == 0.015625
    ));
    // Usage variance just above.
    assert!(propose(just_freeze(), &freeze_obs((8, 2), (11, 3)))
        .proposals()
        .is_empty());
    // Clear-rate variance above: 2/4 and 4/4.
    assert!(propose(just_freeze(), &freeze_obs((8, 2), (10, 4)))
        .proposals()
        .is_empty());
    // The same data under tighter bounds.
    let tighter = StructurePolicyConfig {
        freeze_max_usage_variance: 0.999,
        ..just_freeze()
    };
    assert!(propose(tighter, &freeze_obs((8, 2), (10, 3)))
        .proposals()
        .is_empty());
}

#[test]
fn freeze_needs_k_windows_enough_usage_and_a_serving_expert() {
    let both = freeze_obs((8, 2), (8, 2));
    assert_eq!(propose(just_freeze(), &both).count(OpKind::Freeze), 1);

    // Only one window of history.
    let mut short = both.clone();
    short.windows.remove(0);
    assert!(propose(just_freeze(), &short).proposals().is_empty());

    // Usage below the freeze floor in one window.
    assert!(propose(just_freeze(), &freeze_obs((8, 2), (4, 2)))
        .proposals()
        .is_empty());

    // Not routed to in one window: nothing to call stable.
    let mut idle = both.clone();
    idle.windows[0].codes.clear();
    assert!(propose(just_freeze(), &idle).proposals().is_empty());

    // Already frozen: Freeze does not allow a Frozen target.
    let mut frozen = both;
    frozen.experts[0] = expert("a", token(0), ExpertState::Frozen, &["target"]);
    assert!(propose(just_freeze(), &frozen).proposals().is_empty());
}

#[test]
fn only_the_latest_k_windows_count_for_freeze() {
    let noisy = freeze_obs((1, 0), (1, 4)).windows;
    let stable = freeze_obs((8, 2), (8, 2)).windows;
    let mut obs = freeze_obs((8, 2), (8, 2));
    obs.windows = [noisy.clone(), stable.clone()].concat();
    assert_eq!(propose(just_freeze(), &obs).count(OpKind::Freeze), 1);
    obs.windows = [stable, noisy].concat();
    assert!(propose(just_freeze(), &obs).proposals().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Safety rules across operators
// ---------------------------------------------------------------------------------------------

const ALL_STATES: [ExpertState; 10] = [
    ExpertState::Candidate,
    ExpertState::Gated,
    ExpertState::Serving,
    ExpertState::Frozen,
    ExpertState::Fenced,
    ExpertState::Superseded,
    ExpertState::Retired(RetiredReason::Pruned),
    ExpertState::Retired(RetiredReason::Superseded),
    ExpertState::Retired(RetiredReason::GateFailed),
    ExpertState::Retired(RetiredReason::InputRevoked),
];

#[test]
fn no_proposal_names_a_state_its_operation_does_not_allow() {
    // One idle and one busy expert per state (three busy serving ones, so a merge pair exists),
    // every signal pointing at all of them.
    let mut experts = Vec::new();
    let mut aimer = BTreeMap::new();
    let mut w = with_code(with_code(window(), "target", 9, &[]), "fresh", 9, &[]);
    let mut previous = with_code(window(), "other", 0, &[]);
    let mut index = 0u16;
    for (i, state) in ALL_STATES.iter().enumerate() {
        let busy = if *state == ExpertState::Serving { 3 } else { 1 };
        let roster =
            std::iter::once(("idle", 0u64, 0usize)).chain((0..busy).map(|n| ("busy", 8u64, n)));
        for (kind, usage, n) in roster {
            let name = format!("{}{i}x{n}", kind.chars().next().unwrap());
            experts.push(expert(&name, token(index), *state, &["target"]));
            index += 1;
            aimer.insert(pod(&name), 0.5);
            for window in [&mut w, &mut previous] {
                window.usage.insert(pod(&name), usage);
                if usage > 0 {
                    window.codes.entry(code("target")).or_default().pods.insert(
                        pod(&name),
                        PodCodeStats {
                            routed: 4,
                            cleared: 2,
                        },
                    );
                }
            }
        }
    }
    let mut obs = observation(vec![previous, w], experts);
    obs.aimer = aimer;
    let names: Vec<PodId> = obs
        .experts
        .iter()
        .map(|e| e.target().pod().clone())
        .collect();
    for (i, a) in names.iter().enumerate() {
        for b in &names[i + 1..] {
            obs.overlaps.push(SubspaceOverlap {
                first: a.clone(),
                second: b.clone(),
                overlap: 1.0,
                chance: 0.25,
            });
        }
    }
    let config = StructurePolicyConfig {
        max_proposals_per_window: 256,
        max_proposals_per_expert: 8,
        ..cfg()
    };
    let report = propose(config, &obs);
    assert!(!report.proposals().is_empty());
    let mut seen = std::collections::HashSet::new();
    for generated in report.proposals() {
        let op = generated.request().op();
        seen.insert(op.kind());
        for source in op.sources() {
            let state = source.state();
            assert_ne!(state, ExpertState::Fenced, "{:?}", op.kind());
            assert!(
                op.kind().allows_state(state),
                "{:?} names {state:?}",
                op.kind()
            );
        }
    }
    // The signals were strong enough to exercise every implemented operator.
    for kind in [
        OpKind::Birth,
        OpKind::Reactivate,
        OpKind::Prune,
        OpKind::Merge,
        OpKind::Freeze,
    ] {
        assert!(seen.contains(&kind), "{kind:?} never proposed");
    }
    // Split and Retire are never proposed.
    assert!(!seen.contains(&OpKind::Split) && !seen.contains(&OpKind::Retire));
}

#[test]
fn the_window_and_expert_caps_bound_the_output() {
    // Five unroutable codes, a window cap of three.
    let mut w = window();
    for name in ["c1", "c2", "c3", "c4", "c5"] {
        w = with_code(w, name, 4, &[]);
    }
    let capped = StructurePolicyConfig {
        max_proposals_per_window: 3,
        ..just_birth()
    };
    let report = propose(capped, &observation(vec![w], Vec::new()));
    assert_eq!(report.proposals().len(), 3);
    assert_eq!(suppressed(&report, Suppression::WindowCap), 2);

    // A merge and a freeze both want expert a: with one proposal per expert the merge (higher
    // priority) wins; with two both are proposed.
    let mut obs = merge_obs(2, 2, 0.75, 0.25);
    let stable = with_usage(
        with_usage(
            with_code(window(), "target", 0, &[("a", 4, 2), ("b", 4, 2)]),
            "a",
            8,
        ),
        "b",
        8,
    );
    obs.windows = vec![stable.clone(), stable];
    let all = EnabledOps::ALL;
    let one = propose(only(all), &obs);
    assert_eq!(kinds(&one), vec![OpKind::Merge]);
    assert_eq!(suppressed(&one, Suppression::ExpertCap), 2);
    let two = propose(
        StructurePolicyConfig {
            max_proposals_per_expert: 2,
            ..only(all)
        },
        &obs,
    );
    assert_eq!(
        kinds(&two),
        vec![OpKind::Merge, OpKind::Freeze, OpKind::Freeze]
    );
}

#[test]
fn proposals_carry_no_gate_threshold_or_verifier_parameter() {
    let report = propose(only(EnabledOps::ALL), &birth_obs(4, &[]));
    let request = report.proposals()[0].request().clone();
    let generated = built(&report.proposals()[0]);
    // The exhaustive pattern stops compiling if a field is added to a proposal.
    let ProposalParts {
        project: p,
        proposer: who,
        observed_at,
        op,
        rationale,
    } = generated.parts();
    assert_eq!(p, &project());
    assert_eq!(who, &proposer());
    assert_eq!(observed_at, CommitIndex(1));
    assert_eq!(op.kind(), OpKind::Birth);
    assert_eq!(rationale, request.rationale());
    // The thresholds that triggered it reach the proposal only through the digest.
    let other = propose(
        StructurePolicyConfig {
            merge_overlap_margin: 0.5,
            ..only(EnabledOps::ALL)
        },
        &birth_obs(4, &[]),
    );
    assert_ne!(
        rationale,
        other.proposals()[0].request().rationale(),
        "the rationale digest follows the policy configuration"
    );
    // The proposal is rebuilt from its own parts and validates again.
    let rebuilt =
        ExpertProposal::new(p.clone(), who.clone(), observed_at, op.clone(), *rationale).unwrap();
    assert_eq!(rebuilt, generated);
}

#[test]
fn an_operation_that_creates_an_expert_is_a_request_until_training_completes_it() {
    let report = propose(
        only(EnabledOps::ALL),
        &observation(
            vec![with_code(window(), "target", 4, &[])],
            vec![retired("old", 7, RetiredReason::GateFailed)],
        ),
    );
    // Reactivate (with a result), Birth and Merge are requests with no manifest digest at all.
    let reactivate = &report.proposals()[0];
    assert!(reactivate.needs_training() && reactivate.ready().is_none());
    assert!(reactivate.request().op().results()[0].generation() == Generation(2));
    let birth = propose(just_birth(), &birth_obs(4, &[]));
    let birth = &birth.proposals()[0];
    let merge = propose(just_merge(), &merge_obs(2, 2, 0.75, 0.25));
    let merge = &merge.proposals()[0];
    for pending in [reactivate, birth, merge] {
        assert!(pending.needs_training());
        assert_eq!(pending.ready(), None);
        // Completion fixes the manifest and leaves the operation id alone; another trained
        // digest is another proposal of the same operation.
        let one = pending.complete(trained(1)).unwrap();
        let two = pending.complete(trained(2)).unwrap();
        assert_eq!(one.operation_id(), pending.request().operation_id());
        assert_eq!(one.operation_id(), two.operation_id());
        assert_ne!(one.proposal_id(), two.proposal_id());
        assert_eq!(one.rationale(), pending.request().rationale());
    }
    // An operation that creates nothing is ready and has nothing to train.
    let freeze = propose(just_freeze(), &freeze_obs((8, 2), (8, 2)));
    let freeze = &freeze.proposals()[0];
    assert!(!freeze.needs_training());
    assert!(freeze.ready().is_some());
    assert_eq!(
        freeze.complete(trained(1)).unwrap_err(),
        StructureCompletionError::NoResultToTrain
    );
}

// ---------------------------------------------------------------------------------------------
// Determinism and bounds
// ---------------------------------------------------------------------------------------------

fn busy_observation() -> StructureObservation {
    let mut w = with_code(window(), "target", 9, &[]);
    w = with_code(
        w,
        "shared",
        0,
        &[("a", 4, 2), ("b", 4, 2), ("c", 4, 1), ("d", 4, 0)],
    );
    for (name, usage) in [("a", 8), ("b", 8), ("c", 0), ("d", 1)] {
        w = with_usage(w, name, usage);
    }
    let mut obs = observation(
        vec![w.clone(), w],
        vec![
            serving("a", 0, &["shared"]),
            serving("b", 1, &["shared"]),
            serving("c", 2, &["shared"]),
            serving("d", 3, &["shared"]),
            retired("old", 4, RetiredReason::GateFailed),
        ],
    );
    obs.aimer = BTreeMap::from([(pod("c"), 0.9), (pod("d"), 0.4), (pod("a"), 0.1)]);
    obs.overlaps = vec![
        SubspaceOverlap {
            first: pod("a"),
            second: pod("b"),
            overlap: 0.75,
            chance: 0.25,
        },
        SubspaceOverlap {
            first: pod("c"),
            second: pod("d"),
            overlap: 0.5,
            chance: 0.125,
        },
    ];
    obs
}

#[test]
fn the_same_observation_gives_the_same_proposals_and_digests() {
    let obs = busy_observation();
    let first = propose(cfg(), &obs);
    assert!(first.proposals().len() >= 3);
    for _ in 0..3 {
        let again = propose(cfg(), &obs.clone());
        assert_eq!(again, first);
        for (x, y) in again.proposals().iter().zip(first.proposals()) {
            assert_eq!(built(x).proposal_id(), built(y).proposal_id());
            assert_eq!(x.request().operation_id(), y.request().operation_id());
            assert_eq!(x.request().rationale(), y.request().rationale());
        }
    }
    // A fresh policy built from an equal configuration agrees.
    assert_eq!(policy(cfg()).propose(&obs).unwrap(), first);
}

#[test]
fn the_order_of_the_input_lists_does_not_change_the_output() {
    let obs = busy_observation();
    let expected = propose(cfg(), &obs);
    let mut reversed = obs.clone();
    reversed.experts.reverse();
    reversed.overlaps.reverse();
    assert_eq!(propose(cfg(), &reversed), expected);
}

#[test]
fn a_different_observation_changes_the_digest() {
    let a = propose(just_birth(), &birth_obs(4, &[]));
    let b = propose(just_birth(), &birth_obs(5, &[]));
    assert_ne!(
        a.proposals()[0].request().rationale(),
        b.proposals()[0].request().rationale()
    );
}

#[test]
fn an_observation_over_a_limit_or_inconsistent_is_refused() {
    let limits = ObservationLimits {
        max_windows: 2,
        max_codes_per_window: 2,
        max_pods_per_code: 2,
        max_experts: 2,
        max_codes_per_expert: 2,
        max_overlap_pairs: 1,
    };
    let small = StructurePolicyConfig { limits, ..cfg() };
    let refuse = |obs: &StructureObservation| policy(small.clone()).propose(obs).unwrap_err();

    let obs = observation(vec![window(), window(), window()], Vec::new());
    assert!(matches!(
        refuse(&obs),
        StructureObservationError::TooManyWindows { found: 3, max: 2 }
    ));

    let w = ["a", "b", "c"]
        .iter()
        .fold(window(), |w, name| with_code(w, name, 1, &[]));
    assert!(matches!(
        refuse(&observation(vec![w], Vec::new())),
        StructureObservationError::TooManyCodes { .. }
    ));

    let w = with_code(window(), "a", 0, &[("x", 1, 0), ("y", 1, 0), ("z", 1, 0)]);
    assert!(matches!(
        refuse(&observation(vec![w], Vec::new())),
        StructureObservationError::TooManyPodsForCode { .. }
    ));

    let experts = vec![
        serving("a", 0, &[]),
        serving("b", 1, &[]),
        serving("c", 2, &[]),
    ];
    assert!(matches!(
        refuse(&observation(vec![window()], experts)),
        StructureObservationError::TooManyExperts { .. }
    ));

    let w = with_code(window(), "a", 0, &[("x", 1, 2)]);
    assert!(matches!(
        refuse(&observation(vec![w], Vec::new())),
        StructureObservationError::ClearedExceedsRouted { .. }
    ));

    let dup = vec![serving("a", 0, &[]), serving("a", 1, &[])];
    assert!(matches!(
        refuse(&observation(vec![window()], dup)),
        StructureObservationError::DuplicateExpert { .. }
    ));

    let mut foreign = observation(vec![window()], Vec::new());
    let other = ProjectId::from("elsewhere");
    foreign.experts.push(RegisteredExpert::new(
        ExpertTarget::new(ExpertTargetInit {
            project: other,
            pod: pod("a"),
            generation: Generation(1),
            level: token(0),
            base_model: base(),
            manifest_digest: ManifestDigest::new([2; 32]).unwrap(),
            state: ExpertState::Serving,
            state_version: 1,
        })
        .unwrap(),
        BTreeSet::new(),
    ));
    assert!(matches!(
        refuse(&foreign),
        StructureObservationError::ExpertProjectMismatch { .. }
    ));

    let mut wrong_proposer = observation(vec![window()], Vec::new());
    wrong_proposer.proposer =
        ProposerRef::new(ProjectId::from("elsewhere"), pod("meta"), Generation(1)).unwrap();
    assert_eq!(
        refuse(&wrong_proposer),
        StructureObservationError::ProposerProjectMismatch
    );

    let mut nan = observation(vec![window()], Vec::new());
    nan.aimer.insert(pod("a"), f64::NAN);
    assert!(matches!(
        refuse(&nan),
        StructureObservationError::NonFiniteSignal { field: "aimer" }
    ));

    let overlap = |first: &str, second: &str, value: f64| SubspaceOverlap {
        first: pod(first),
        second: pod(second),
        overlap: value,
        chance: 0.25,
    };
    let mut bad = observation(vec![window()], Vec::new());
    bad.overlaps = vec![overlap("a", "b", 1.5)];
    assert!(matches!(
        refuse(&bad),
        StructureObservationError::NonFiniteSignal { field: "overlap" }
    ));
    bad.overlaps = vec![overlap("a", "a", 0.5)];
    assert!(matches!(
        refuse(&bad),
        StructureObservationError::SelfOverlap { .. }
    ));
    let mut many = observation(vec![window()], Vec::new());
    many.overlaps = vec![overlap("a", "b", 0.5), overlap("b", "a", 0.5)];
    assert!(matches!(
        refuse(&many),
        StructureObservationError::TooManyOverlaps { .. }
    ));
    let roomy = StructurePolicyConfig {
        limits: ObservationLimits {
            max_overlap_pairs: 4,
            ..limits
        },
        ..cfg()
    };
    assert!(matches!(
        policy(roomy).propose(&many).unwrap_err(),
        StructureObservationError::DuplicateOverlap { .. }
    ));
}

#[test]
fn an_empty_observation_proposes_nothing() {
    let report = propose(cfg(), &observation(Vec::new(), Vec::new()));
    assert!(report.proposals().is_empty());
    assert!(report.suppressed().is_empty());
}

#[test]
fn untracked_unroutable_reports_are_reported_and_not_acted_on() {
    let mut obs = birth_obs(0, &[]);
    obs.windows[0].unroutable_untracked = 1000;
    let report = propose(cfg(), &obs);
    assert!(report.proposals().is_empty());
    assert_eq!(report.untracked_unroutable(), 1000);
}

// ---------------------------------------------------------------------------------------------
// Through the admission path
// ---------------------------------------------------------------------------------------------

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

fn gate_report(proposal: &ExpertProposal, public: PublicSuite) -> BoundGateReport {
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    gate()
        .evaluate(&AdapterId::from("candidate"), &matrix, public)
        .unwrap()
        .bind(subject)
}

fn passing() -> PublicSuite {
    PublicSuite {
        serving: 0.8,
        candidate: 0.8,
    }
}

fn failing() -> PublicSuite {
    PublicSuite {
        serving: 0.9,
        candidate: 0.1,
    }
}

fn grant() -> ExpertAdmissionGrant {
    ExpertAdmissionGrant::new(
        project(),
        proposer(),
        requirement(7, gate().config_digest().unwrap()),
        binding(),
    )
}

fn runtime() -> PtrRuntime {
    let mut rt = PtrRuntime::new(PtrConfig::default()).unwrap();
    prime(&mut rt);
    rt
}

#[test]
fn a_generated_request_is_admitted_only_when_trained_and_passed_by_the_gate() {
    let mut rt = runtime();
    let report = propose(just_birth(), &birth_obs(4, &[]));
    let pending = &report.proposals()[0];
    let born = pending.request().op().results()[0].pod().clone();
    // The generator hands out a request: there is no proposal to admit until training has
    // produced the manifest digest (admit_expert_proposal takes an ExpertProposal only).
    assert_eq!(pending.ready(), None);
    let birth = pending.complete(trained(0x42)).unwrap();

    // A failing gate report: refused, nothing appended, nothing registered.
    let events = rt.committed_events().len();
    assert_eq!(
        rt.admit_expert_proposal(&birth, &gate_report(&birth, failing()), &grant()),
        Err(RuntimeError::Expert(ExpertAdmissionError::GateFailed))
    );
    assert_eq!(rt.committed_events().len(), events);
    assert!(rt.expert_registry().get(&project(), &born).is_none());

    // A passing report for another proposal, including another training run of the same
    // request: refused.
    let retrained = pending.complete(trained(0x43)).unwrap();
    assert_eq!(retrained.operation_id(), birth.operation_id());
    assert_ne!(retrained.proposal_id(), birth.proposal_id());
    for other in [
        retrained,
        built(&propose(just_birth(), &birth_obs(5, &[("a", 1, 1)])).proposals()[0]),
    ] {
        assert_eq!(
            rt.admit_expert_proposal(&birth, &gate_report(&other, passing()), &grant()),
            Err(RuntimeError::Expert(
                ExpertAdmissionError::EvidenceProposalMismatch
            ))
        );
    }
    assert!(rt.expert_registry().get(&project(), &born).is_none());

    // A passing report bound to exactly the completed proposal: admitted.
    rt.admit_expert_proposal(&birth, &gate_report(&birth, passing()), &grant())
        .unwrap();
    let entry = rt.expert_registry().get(&project(), &born).unwrap().clone();
    assert_eq!(entry.state, ExpertState::Serving);
    assert_eq!(entry.generation, Generation(1));
    assert_eq!(entry.manifest_digest, trained(0x42));

    // The registry view feeds the next round: the new expert is stable, so the policy proposes
    // freezing it; that proposal needs no training and is decided by the gate in the same way.
    let view = RegisteredExpert::from_projection(
        rt.expert_registry(),
        &project(),
        &born,
        BTreeSet::from([code("target")]),
    )
    .unwrap();
    assert_eq!(view.target().state(), ExpertState::Serving);
    let stable = with_usage(
        with_code(window(), "target", 0, &[(born.0.as_str(), 4, 2)]),
        &born.0,
        8,
    );
    let obs = observation(vec![stable.clone(), stable], vec![view]);
    let freeze = propose(just_freeze(), &obs).proposals()[0].ready().unwrap();
    assert_eq!(freeze.op().kind(), OpKind::Freeze);
    assert_eq!(
        rt.admit_expert_proposal(&freeze, &gate_report(&freeze, failing()), &grant()),
        Err(RuntimeError::Expert(ExpertAdmissionError::GateFailed))
    );
    assert_eq!(
        rt.expert_registry().get(&project(), &born).unwrap().state,
        ExpertState::Serving
    );
    rt.admit_expert_proposal(&freeze, &gate_report(&freeze, passing()), &grant())
        .unwrap();
    assert_eq!(
        rt.expert_registry().get(&project(), &born).unwrap().state,
        ExpertState::Frozen
    );

    // The same proposal again is stale: the compare-and-set sees the new state.
    assert!(rt
        .admit_expert_proposal(&freeze, &gate_report(&freeze, passing()), &grant())
        .is_err());
}

#[test]
fn a_proposal_for_a_foreign_proposer_is_refused_whatever_the_report_says() {
    let mut rt = runtime();
    let mut obs = birth_obs(4, &[]);
    let elsewhere = ProjectId::from("elsewhere");
    obs.project = elsewhere.clone();
    obs.proposer = ProposerRef::new(elsewhere, pod("meta"), Generation(1)).unwrap();
    let report = propose(just_birth(), &obs);
    let birth = built(&report.proposals()[0]);
    assert_eq!(
        rt.admit_expert_proposal(&birth, &gate_report(&birth, passing()), &grant()),
        Err(RuntimeError::Expert(ExpertAdmissionError::ProjectMismatch))
    );
}

// ---------------------------------------------------------------------------------------------
// Review fixes: quality floors, arithmetic, provenance, calibration
// ---------------------------------------------------------------------------------------------

#[test]
fn freeze_and_merge_need_a_clear_rate_floor() {
    // Floor 0.25: 1 of 4 passes, 0 of 4 does not, in both operators.
    assert_eq!(
        propose(just_freeze(), &freeze_obs((8, 1), (8, 1))).count(OpKind::Freeze),
        1
    );
    assert!(propose(just_freeze(), &freeze_obs((8, 0), (8, 0)))
        .proposals()
        .is_empty());
    // One window below the floor is enough to refuse.
    assert!(propose(just_freeze(), &freeze_obs((8, 1), (8, 0)))
        .proposals()
        .is_empty());
    assert_eq!(
        propose(just_merge(), &merge_obs(1, 1, 0.75, 0.25)).count(OpKind::Merge),
        1
    );
    assert!(propose(just_merge(), &merge_obs(0, 0, 0.75, 0.25))
        .proposals()
        .is_empty());
    assert!(propose(just_merge(), &merge_obs(1, 0, 0.75, 0.25))
        .proposals()
        .is_empty());
}

#[test]
fn thresholds_that_are_not_exactly_representable_trigger_at_their_boundary() {
    // 0.3 - 0.1 is 0.19999999999999998 in f64: still at the 0.2 margin.
    let margin = |m| StructurePolicyConfig {
        merge_overlap_margin: m,
        ..just_merge()
    };
    let at = merge_obs(2, 2, 0.3, 0.1);
    assert_eq!(propose(margin(0.2), &at).count(OpKind::Merge), 1);
    let below = merge_obs(2, 2, 0.299, 0.1);
    assert_eq!(propose(margin(0.2), &below).count(OpKind::Merge), 0);
    // Clear rates 3/10 and 4/10 differ by 0.10000000000000003: at the 0.1 margin.
    let rates = |cleared_a, cleared_b| {
        let w = with_code(
            window(),
            "target",
            0,
            &[("a", 10, cleared_a), ("b", 10, cleared_b)],
        );
        let mut obs = merge_obs(2, 2, 0.75, 0.25);
        obs.windows = vec![w];
        obs
    };
    let gap = StructurePolicyConfig {
        merge_clear_rate_margin: 0.1,
        ..just_merge()
    };
    assert_eq!(propose(gap.clone(), &rates(3, 4)).count(OpKind::Merge), 1);
    assert_eq!(propose(gap, &rates(3, 5)).count(OpKind::Merge), 0);
}

#[test]
fn counts_are_bounded_and_sums_saturate() {
    use ptr_runtime::expert::MAX_COUNT;
    let mut huge = birth_obs(0, &[("a", MAX_COUNT, 0), ("b", MAX_COUNT, 0)]);
    // At the cap, with many pods: no overflow, and the trigger fires on the (saturated) sum.
    assert_eq!(propose(just_birth(), &huge).count(OpKind::Birth), 1);
    huge.windows[0]
        .codes
        .get_mut(&code("target"))
        .unwrap()
        .unroutable = MAX_COUNT;
    assert_eq!(propose(just_birth(), &huge).count(OpKind::Birth), 1);
    for (field, mutate) in [
        (
            "unroutable",
            Box::new(|o: &mut StructureObservation| {
                o.windows[0]
                    .codes
                    .get_mut(&code("target"))
                    .unwrap()
                    .unroutable = MAX_COUNT + 1;
            }) as Box<dyn Fn(&mut StructureObservation)>,
        ),
        (
            "routed",
            Box::new(|o: &mut StructureObservation| {
                let s = o.windows[0].codes.get_mut(&code("target")).unwrap();
                s.pods.insert(
                    pod("a"),
                    PodCodeStats {
                        routed: u64::MAX,
                        cleared: 0,
                    },
                );
            }),
        ),
        (
            "usage",
            Box::new(|o: &mut StructureObservation| {
                o.windows[0].usage.insert(pod("a"), u64::MAX);
            }),
        ),
        (
            "untracked",
            Box::new(|o: &mut StructureObservation| o.windows[0].unroutable_untracked = u64::MAX),
        ),
    ] {
        let mut obs = birth_obs(4, &[("a", 1, 0)]);
        mutate(&mut obs);
        assert!(
            matches!(
                policy(just_birth()).propose(&obs),
                Err(StructureObservationError::CountTooLarge { .. })
            ),
            "{field}"
        );
    }
}

#[test]
fn the_configuration_checks_the_new_fields() {
    let cases: Vec<(StructurePolicyConfig, StructureConfigError)> = vec![
        (
            StructurePolicyConfig {
                birth_windows: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "birth_windows",
            },
        ),
        (
            StructurePolicyConfig {
                max_births_per_window: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "max_births_per_window",
            },
        ),
        (
            StructurePolicyConfig {
                prune_windows: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "prune_windows",
            },
        ),
        (
            StructurePolicyConfig {
                backup_min_cleared: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "backup_min_cleared",
            },
        ),
        (
            StructurePolicyConfig {
                backup_clear_windows: 0,
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "backup_clear_windows",
            },
        ),
        (
            StructurePolicyConfig {
                limits: ObservationLimits {
                    max_codes_per_expert: 0,
                    ..ObservationLimits::default()
                },
                ..cfg()
            },
            StructureConfigError::ZeroField {
                field: "limits.max_codes_per_expert",
            },
        ),
        (
            StructurePolicyConfig {
                freeze_min_clear_rate: 1.5,
                ..cfg()
            },
            StructureConfigError::BadFraction {
                field: "freeze_min_clear_rate",
            },
        ),
        (
            StructurePolicyConfig {
                merge_min_clear_rate: f64::NAN,
                ..cfg()
            },
            StructureConfigError::BadFraction {
                field: "merge_min_clear_rate",
            },
        ),
        (
            StructurePolicyConfig {
                prune_windows: 5,
                limits: ObservationLimits {
                    max_windows: 4,
                    ..ObservationLimits::default()
                },
                ..cfg()
            },
            StructureConfigError::WindowsExceedHistory {
                field: "prune_windows",
            },
        ),
        (
            StructurePolicyConfig {
                birth_windows: 5,
                limits: ObservationLimits {
                    max_windows: 4,
                    ..ObservationLimits::default()
                },
                ..cfg()
            },
            StructureConfigError::WindowsExceedHistory {
                field: "birth_windows",
            },
        ),
        (
            StructurePolicyConfig {
                backup_clear_windows: 5,
                limits: ObservationLimits {
                    max_windows: 4,
                    ..ObservationLimits::default()
                },
                ..cfg()
            },
            StructureConfigError::WindowsExceedHistory {
                field: "backup_clear_windows",
            },
        ),
    ];
    for (config, expected) in cases {
        assert_eq!(config.validate(), Err(expected), "{expected:?}");
    }
    let changed = |c: StructurePolicyConfig| assert_ne!(c.digest(), cfg().digest());
    changed(StructurePolicyConfig {
        prune_min_age_commits: 3,
        ..cfg()
    });
    changed(StructurePolicyConfig {
        backup_clear_windows: 2,
        ..cfg()
    });
    changed(StructurePolicyConfig {
        merge_min_clear_rate: 0.5,
        ..cfg()
    });
}

#[test]
fn the_new_observation_errors() {
    let refuse = |obs: &StructureObservation| {
        policy(StructurePolicyConfig {
            limits: ObservationLimits {
                max_codes_per_expert: 2,
                ..ObservationLimits::default()
            },
            ..cfg()
        })
        .propose(obs)
        .unwrap_err()
    };
    let crowded = observation(vec![window()], vec![serving("a", 0, &["c1", "c2", "c3"])]);
    assert!(matches!(
        refuse(&crowded),
        StructureObservationError::TooManyCodesForExpert { .. }
    ));
    let mut stray = observation(vec![window()], vec![serving("a", 0, &[])]);
    stray.aimer.insert(pod("ghost"), 0.5);
    assert_eq!(
        refuse(&stray),
        StructureObservationError::SignalForUnknownExpert { pod: pod("ghost") }
    );
    let mut stray = observation(vec![window()], vec![serving("a", 0, &[])]);
    stray.overlaps.push(SubspaceOverlap {
        first: pod("a"),
        second: pod("ghost"),
        overlap: 0.5,
        chance: 0.25,
    });
    assert!(matches!(
        refuse(&stray),
        StructureObservationError::SignalForUnknownExpert { .. }
    ));
}

#[test]
fn the_rationale_commits_to_the_whole_observation() {
    let obs = busy_observation();
    assert_eq!(obs.digest(), busy_observation().digest());
    // Independent of the order of the lists.
    let mut shuffled = obs.clone();
    shuffled.experts.reverse();
    shuffled.overlaps.reverse();
    assert_eq!(shuffled.digest(), obs.digest());
    // Any change anywhere moves it, including one the triggering numbers do not depend on.
    let mut changed = Vec::new();
    let mut o = obs.clone();
    o.aimer.insert(pod("a"), 0.11);
    changed.push(o);
    let mut o = obs.clone();
    o.windows[0].usage.insert(pod("a"), 9);
    changed.push(o);
    let mut o = obs.clone();
    o.overlaps[1].chance = 0.2;
    changed.push(o);
    let mut o = obs.clone();
    o.observed_at = CommitIndex(2);
    changed.push(o);
    let mut o = obs.clone();
    o.outstanding.insert(code("elsewhere"));
    changed.push(o);
    let mut o = obs.clone();
    o.experts[0] = o.experts[0].clone().with_last_changed_at(CommitIndex(1));
    changed.push(o);
    let reference = propose(cfg(), &obs);
    for other in changed {
        assert_ne!(other.digest(), obs.digest());
        let report = propose(cfg(), &other);
        // The same trigger, on another observation, has another rationale digest.
        let by_kind = |r: &StructureReport, kind| {
            r.proposals()
                .iter()
                .find(|p| p.kind() == kind)
                .map(|p| *p.request().rationale())
        };
        assert_ne!(
            by_kind(&report, OpKind::Reactivate),
            by_kind(&reference, OpKind::Reactivate)
        );
    }
}

#[test]
fn a_seed_that_serves_two_triggered_codes_is_reactivated_once() {
    let mut old = retired("old", 7, RetiredReason::GateFailed);
    old = RegisteredExpert::new(
        old.target().clone(),
        BTreeSet::from([code("alpha"), code("beta")]),
    );
    let w = with_code(with_code(window(), "alpha", 4, &[]), "beta", 4, &[]);
    let report = propose(only(EnabledOps::ALL), &observation(vec![w], vec![old]));
    assert_eq!(kinds(&report), vec![OpKind::Reactivate]);
    assert_eq!(suppressed(&report, Suppression::SeedAlreadyUsed), 1);
}

#[test]
fn a_token_birth_may_reuse_a_slot_a_retired_expert_released() {
    let mut obs = birth_obs(4, &[]);
    obs.experts = vec![
        expert(
            "gone",
            token(0),
            ExpertState::Retired(RetiredReason::Pruned),
            &[],
        ),
        serving("live", 1, &[]),
    ];
    obs.birth = Some(BirthContext {
        base_model: base(),
        site: BirthSite::Token {
            layer: LAYER,
            slots: 2,
        },
    });
    let report = propose(just_birth(), &obs);
    assert_eq!(
        report.proposals()[0].request().op().results()[0].level(),
        token(0)
    );
}

#[test]
fn a_request_is_not_a_proposal_and_a_birth_result_has_no_manifest_before_training() {
    // The result of a requested Birth is an intent: project, pod, level, base model, generation.
    let report = propose(just_birth(), &birth_obs(4, &[]));
    let intent = &report.proposals()[0].request().op().results()[0];
    assert_eq!(intent.project(), &project());
    assert_eq!(intent.base_model(), &base());
    // Completing with a different trained digest never changes what the operation would create.
    let one = report.proposals()[0].complete(trained(1)).unwrap();
    let two = report.proposals()[0].complete(trained(2)).unwrap();
    assert_eq!(one.op().results()[0].pod(), two.op().results()[0].pod());
    assert_ne!(
        one.op().results()[0].manifest_digest(),
        two.op().results()[0].manifest_digest()
    );
}
