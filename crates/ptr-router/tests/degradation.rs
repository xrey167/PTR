mod common;

use common::{backend, candidate, option};
use ptr_router::{
    AvailabilityView, BackendState, ContextMode, DegradationEscalation, DegradationLevel,
    ExecutionRecipe, FallbackPlanner, PlanInputs, PodUse, ReasoningMode, RecipeError,
    RecipePlanner, RequestProfile, Resolution, RetrievalSource, SearchMode, Stage, StageBudget,
    StageEffect, StageKind, StaticAvailability, SymbolicStep,
};
use ptr_types::{CapabilityId, PodId};

fn recipe(effect: StageEffect) -> ExecutionRecipe {
    let model = option("m1", 4, 300, 0.2, &[]);
    let candidate = candidate("r", "pod-a");
    let profile = RequestProfile {
        pod_effect: effect,
        ..RequestProfile::default()
    };
    RecipePlanner::default()
        .plan(&PlanInputs {
            profile: &profile,
            reasoning: ReasoningMode::Deliberate,
            search: SearchMode::Search,
            model: Some(&model),
            pod_use: PodUse::Call,
            candidate: Some(&candidate),
        })
        .unwrap()
}

fn up() -> StaticAvailability {
    StaticAvailability::all(BackendState::Up)
}

fn resolved(plan: &ptr_router::FallbackPlan) -> (&DegradationLevel, &ExecutionRecipe) {
    match &plan.resolution {
        Resolution::Run { level, recipe } => (level, recipe),
        Resolution::Escalate { reason } => panic!("escalated: {reason:?}"),
    }
}

#[test]
fn everything_up_resolves_to_the_original_recipe() {
    let original = recipe(StageEffect::Pure);
    let plan = FallbackPlanner.plan(&original, &up());
    let (level, recipe) = resolved(&plan);
    assert_eq!(*level, DegradationLevel::Full);
    assert_eq!(recipe, &original);
    assert_eq!(plan.level(), DegradationLevel::Full);
    assert_eq!(plan.chains.len(), original.stages().len());
}

#[test]
fn a_degraded_model_runs_with_reduced_context_and_half_budget() {
    let original = recipe(StageEffect::Pure);
    let view = up().with_model(backend("m1"), BackendState::Degraded);
    let plan = FallbackPlanner.plan(&original, &view);
    let (level, recipe) = resolved(&plan);
    assert_eq!(*level, DegradationLevel::ReducedContext);
    let model = recipe
        .stages()
        .iter()
        .find_map(|s| match &s.kind {
            StageKind::ModelCall { requirement } => Some((requirement, s.budget)),
            _ => None,
        })
        .unwrap();
    assert_eq!(model.0.context, ContextMode::Reduced);
    assert_eq!(model.1, StageBudget::new(2, 150));
    // The model's chain starts with no Full alternative, then ReducedContext, then symbolic.
    let chain = &plan.chains[1].alternatives;
    assert_eq!(chain[0].level, DegradationLevel::ReducedContext);
    assert_eq!(chain.last().unwrap().level, DegradationLevel::SymbolicOnly);
}

#[test]
fn a_downed_model_falls_to_symbolic_only_for_pure_work() {
    let original = recipe(StageEffect::Pure);
    let view = up()
        .with_model(backend("m1"), BackendState::Down)
        .with_retrieval(RetrievalSource::Search, BackendState::Down);
    let plan = FallbackPlanner.plan(&original, &view);
    let (level, recipe) = resolved(&plan);
    assert_eq!(*level, DegradationLevel::SymbolicOnly);
    assert!(recipe.stages().iter().all(|s| !matches!(
        s.kind,
        StageKind::ModelCall { .. } | StageKind::Retrieve { .. }
    )));
    assert!(recipe.stages().iter().any(|s| matches!(
        s.kind,
        StageKind::Symbolic {
            step: SymbolicStep::Check
        }
    )));
    assert_eq!(recipe.stages().last().unwrap().kind, StageKind::Verify);
}

#[test]
fn a_downed_effectful_pod_escalates_instead_of_dropping_the_effect() {
    let original = recipe(StageEffect::Effectful);
    let view = up().with_pod(PodId::from("pod-a"), BackendState::Down);
    let plan = FallbackPlanner.plan(&original, &view);
    assert_eq!(plan.level(), DegradationLevel::Escalate);
    let Resolution::Escalate { reason } = &plan.resolution else {
        panic!("expected escalation");
    };
    let pod_index = original
        .stages()
        .iter()
        .position(|s| matches!(s.kind, StageKind::PodCall { .. }))
        .unwrap();
    assert_eq!(
        reason,
        &DegradationEscalation::StageUnavailable { stage: pod_index }
    );
    assert!(plan.chains[pod_index].alternatives.is_empty());
}

#[test]
fn a_degraded_effectful_pod_still_runs_and_stays_verified() {
    let original = recipe(StageEffect::Effectful);
    let view = up().with_pod(PodId::from("pod-a"), BackendState::Degraded);
    let plan = FallbackPlanner.plan(&original, &view);
    let (level, recipe) = resolved(&plan);
    assert_eq!(*level, DegradationLevel::ReducedContext);
    assert!(recipe.stages().iter().any(|s| matches!(
        &s.kind,
        StageKind::PodCall {
            effect: StageEffect::Effectful,
            class,
            ..
        } if class == &CapabilityId::from("read")
    )));
    assert_eq!(recipe.stages().last().unwrap().kind, StageKind::Verify);
}

#[test]
fn a_pure_pod_down_is_replaced_by_a_symbolic_check() {
    let original = recipe(StageEffect::Pure);
    let view = up().with_pod(PodId::from("pod-a"), BackendState::Down);
    let plan = FallbackPlanner.plan(&original, &view);
    let (level, recipe) = resolved(&plan);
    assert_eq!(*level, DegradationLevel::SymbolicOnly);
    assert!(recipe
        .stages()
        .iter()
        .all(|s| !matches!(s.kind, StageKind::PodCall { .. })));
}

#[test]
fn the_verifier_is_never_dropped_even_when_everything_else_is_fine() {
    let original = recipe(StageEffect::Pure);
    for state in [BackendState::Down, BackendState::Degraded] {
        let plan = FallbackPlanner.plan(&original, &up().with_verifier(state));
        assert_eq!(plan.level(), DegradationLevel::Escalate);
        assert_eq!(
            plan.resolution,
            Resolution::Escalate {
                reason: DegradationEscalation::VerifierUnavailable
            }
        );
        let verify = plan.chains.last().unwrap();
        assert!(verify.alternatives.is_empty());
    }
}

#[test]
fn unknown_backends_use_the_view_default_and_down_fails_closed() {
    let original = recipe(StageEffect::Pure);
    let plan = FallbackPlanner.plan(&original, &StaticAvailability::all(BackendState::Down));
    assert_eq!(
        plan.resolution,
        Resolution::Escalate {
            reason: DegradationEscalation::VerifierUnavailable
        }
    );
}

#[test]
fn only_omissions_that_leave_no_work_are_reported_as_invalid() {
    // A retrieval-only recipe with the search source down would become a bare verifier.
    let stages = vec![
        Stage {
            kind: StageKind::Retrieve {
                source: RetrievalSource::Search,
            },
            budget: StageBudget::new(1, 10),
        },
        Stage {
            kind: StageKind::Verify,
            budget: StageBudget::new(1, 10),
        },
    ];
    let original = ExecutionRecipe::new(stages).unwrap();
    let view = up().with_retrieval(RetrievalSource::Search, BackendState::Down);
    let plan = FallbackPlanner.plan(&original, &view);
    assert_eq!(
        plan.resolution,
        Resolution::Escalate {
            reason: DegradationEscalation::InvalidRecipe(RecipeError::NoWorkStage)
        }
    );
}

#[test]
fn exhaustive_outage_sweep_never_yields_an_unverified_or_unsafe_recipe() {
    const STATES: [BackendState; 3] =
        [BackendState::Up, BackendState::Degraded, BackendState::Down];
    for effect in [StageEffect::Pure, StageEffect::Effectful] {
        let original = recipe(effect);
        for model in STATES {
            for pod in STATES {
                for memory in STATES {
                    for search in STATES {
                        for verifier in STATES {
                            let view = StaticAvailability::all(BackendState::Up)
                                .with_model(backend("m1"), model)
                                .with_pod(PodId::from("pod-a"), pod)
                                .with_retrieval(RetrievalSource::Memory, memory)
                                .with_retrieval(RetrievalSource::Search, search)
                                .with_verifier(verifier);
                            let plan = FallbackPlanner.plan(&original, &view);
                            assert_eq!(plan, FallbackPlanner.plan(&original, &view));
                            match &plan.resolution {
                                Resolution::Escalate { .. } => {}
                                Resolution::Run { level, recipe } => {
                                    assert_eq!(verifier, BackendState::Up);
                                    assert_eq!(
                                        recipe.stages().last().unwrap().kind,
                                        StageKind::Verify
                                    );
                                    ExecutionRecipe::new(recipe.stages().to_vec()).unwrap();
                                    assert_ne!(*level, DegradationLevel::Escalate);
                                    let has_effect = recipe
                                        .stages()
                                        .iter()
                                        .any(|s| s.kind.effect() == StageEffect::Effectful);
                                    if let Some(at) = recipe
                                        .stages()
                                        .iter()
                                        .position(|s| s.kind.effect() == StageEffect::Effectful)
                                    {
                                        // Pre-effect rule: a verifier gates the effect.
                                        assert_eq!(recipe.stages()[at - 1].kind, StageKind::Verify);
                                    }
                                    if effect == StageEffect::Effectful {
                                        // The effect is either performed (pod not down) or the
                                        // plan escalated; it is never silently dropped.
                                        assert!(has_effect);
                                        assert_ne!(pod, BackendState::Down);
                                    }
                                    if level == &DegradationLevel::Full {
                                        assert_eq!(
                                            (model, pod, search, verifier),
                                            (
                                                BackendState::Up,
                                                BackendState::Up,
                                                BackendState::Up,
                                                BackendState::Up
                                            )
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn availability_view_is_object_safe() {
    let view: &dyn AvailabilityView = &up();
    assert_eq!(view.verifier(), BackendState::Up);
    assert_eq!(view.model(&backend("any")), BackendState::Up);
}
