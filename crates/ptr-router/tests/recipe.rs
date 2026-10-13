mod common;

use common::{candidate, option};
use ptr_router::{
    BackendCapability, ContextMode, ExecutionRecipe, PlanInputs, PodUse, ReasoningMode,
    RecipeError, RecipePlanner, RequestProfile, RetrievalSource, SearchMode, Stage, StageBudget,
    StageEffect, StageKind, SymbolicStep, MAX_STAGES,
};
use ptr_types::{CapabilityId, PodId};

fn profile(effect: StageEffect) -> RequestProfile {
    RequestProfile {
        pod_effect: effect,
        ..RequestProfile::default()
    }
}

fn kinds(recipe: &ExecutionRecipe) -> Vec<&StageKind> {
    recipe.stages().iter().map(|s| &s.kind).collect()
}

#[test]
fn direct_recipe_is_model_then_verify() {
    let model = option("m1", 4, 300, 0.2, &[BackendCapability::Reasoning]);
    let profile = profile(StageEffect::Pure);
    let recipe = RecipePlanner::default()
        .plan(&PlanInputs {
            profile: &profile,
            reasoning: ReasoningMode::Direct,
            search: SearchMode::Skip,
            model: Some(&model),
            pod_use: PodUse::Skip,
            candidate: None,
        })
        .unwrap();
    let stages = recipe.stages();
    assert_eq!(stages.len(), 2);
    let StageKind::ModelCall { requirement } = &stages[0].kind else {
        panic!("first stage must be the model call");
    };
    assert_eq!(requirement.backend.as_str(), "m1");
    assert_eq!(requirement.context, ContextMode::Full);
    assert_eq!(stages[0].budget, StageBudget::new(4, 300));
    assert_eq!(stages[1].kind, StageKind::Verify);
}

#[test]
fn full_recipe_has_the_documented_order() {
    let model = option("m1", 4, 300, 0.2, &[]);
    let candidate = candidate("r1", "zeta");
    let profile = profile(StageEffect::Effectful);
    let recipe = RecipePlanner::default()
        .plan(&PlanInputs {
            profile: &profile,
            reasoning: ReasoningMode::Deliberate,
            search: SearchMode::Search,
            model: Some(&model),
            pod_use: PodUse::Call,
            candidate: Some(&candidate),
        })
        .unwrap();
    let k = kinds(&recipe);
    assert!(matches!(
        k[0],
        StageKind::Retrieve {
            source: RetrievalSource::Search
        }
    ));
    assert!(matches!(k[1], StageKind::ModelCall { .. }));
    assert!(matches!(
        k[2],
        StageKind::Symbolic {
            step: SymbolicStep::Check
        }
    ));
    assert_eq!(
        k[3],
        &StageKind::PodCall {
            class: CapabilityId::from("read"),
            pod: PodId::from("zeta"),
            effect: StageEffect::Effectful
        }
    );
    assert_eq!(k[4], &StageKind::Verify);
    assert_eq!(k.len(), 5);
    assert!(recipe.total_latency_ms() >= 500);
}

#[test]
fn symbolic_first_puts_the_plan_before_the_model() {
    let model = option("m1", 1, 10, 0.2, &[]);
    let profile = profile(StageEffect::Pure);
    let recipe = RecipePlanner::default()
        .plan(&PlanInputs {
            profile: &profile,
            reasoning: ReasoningMode::SymbolicFirst,
            search: SearchMode::Memory,
            model: Some(&model),
            pod_use: PodUse::Skip,
            candidate: None,
        })
        .unwrap();
    let k = kinds(&recipe);
    assert!(matches!(
        k[0],
        StageKind::Retrieve {
            source: RetrievalSource::Memory
        }
    ));
    assert!(matches!(
        k[1],
        StageKind::Symbolic {
            step: SymbolicStep::Plan
        }
    ));
    assert!(matches!(k[2], StageKind::ModelCall { .. }));
}

#[test]
fn a_pod_call_without_a_checked_candidate_is_refused() {
    let profile = profile(StageEffect::Pure);
    let error = RecipePlanner::default()
        .plan(&PlanInputs {
            profile: &profile,
            reasoning: ReasoningMode::Direct,
            search: SearchMode::Skip,
            model: None,
            pod_use: PodUse::Call,
            candidate: None,
        })
        .unwrap_err();
    assert_eq!(error, RecipeError::MissingCheckedCandidate);
}

#[test]
fn an_empty_selection_still_yields_work_and_verification() {
    let profile = profile(StageEffect::Pure);
    let recipe = RecipePlanner::default()
        .plan(&PlanInputs {
            profile: &profile,
            reasoning: ReasoningMode::Direct,
            search: SearchMode::Skip,
            model: None,
            pod_use: PodUse::Skip,
            candidate: None,
        })
        .unwrap();
    assert!(matches!(
        recipe.stages()[0].kind,
        StageKind::Symbolic { .. }
    ));
    assert_eq!(recipe.stages().last().unwrap().kind, StageKind::Verify);
}

#[test]
fn every_planner_output_ends_in_verification_and_is_deterministic() {
    let model = option("m1", 2, 20, 0.2, &[]);
    let candidate = candidate("r1", "alpha");
    let planner = RecipePlanner::default();
    let mut count = 0;
    for reasoning in ReasoningMode::ALL {
        for search in SearchMode::ALL {
            for use_model in [false, true] {
                for pod_use in PodUse::ALL {
                    for effect in [StageEffect::Pure, StageEffect::Effectful] {
                        let profile = profile(effect);
                        let inputs = PlanInputs {
                            profile: &profile,
                            reasoning,
                            search,
                            model: use_model.then_some(&model),
                            pod_use,
                            candidate: Some(&candidate),
                        };
                        let first = planner.plan(&inputs).unwrap();
                        assert_eq!(first, planner.plan(&inputs).unwrap());
                        assert_eq!(first.stages().last().unwrap().kind, StageKind::Verify);
                        // The same invariant holds when re-validated from the stage list.
                        ExecutionRecipe::new(first.stages().to_vec()).unwrap();
                        count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(count, 3 * 3 * 2 * 2 * 2);
}

fn stage(kind: StageKind) -> Stage {
    Stage {
        kind,
        budget: StageBudget::new(1, 10),
    }
}

fn effectful_pod() -> StageKind {
    StageKind::PodCall {
        class: CapabilityId::from("write"),
        pod: PodId::from("w"),
        effect: StageEffect::Effectful,
    }
}

#[test]
fn recipes_without_a_verifier_after_an_effect_are_rejected() {
    assert_eq!(
        ExecutionRecipe::new(vec![stage(effectful_pod())]),
        Err(RecipeError::EffectWithoutVerifier { stage: 0 })
    );
    assert_eq!(
        ExecutionRecipe::new(vec![
            stage(effectful_pod()),
            stage(StageKind::Symbolic {
                step: SymbolicStep::Check
            }),
        ]),
        Err(RecipeError::EffectWithoutVerifier { stage: 0 })
    );
    // The verifier must come after the effect, and the first effect without one is named.
    assert_eq!(
        ExecutionRecipe::new(vec![
            stage(effectful_pod()),
            stage(StageKind::Verify),
            stage(effectful_pod()),
        ]),
        Err(RecipeError::EffectWithoutVerifier { stage: 2 })
    );
    assert!(ExecutionRecipe::new(vec![stage(effectful_pod()), stage(StageKind::Verify)]).is_ok());
}

#[test]
fn structural_errors_are_typed() {
    assert_eq!(ExecutionRecipe::new(vec![]), Err(RecipeError::Empty));
    let sym = stage(StageKind::Symbolic {
        step: SymbolicStep::Plan,
    });
    assert_eq!(
        ExecutionRecipe::new(vec![sym.clone()]),
        Err(RecipeError::TerminalStageNotVerifier)
    );
    assert_eq!(
        ExecutionRecipe::new(vec![stage(StageKind::Verify)]),
        Err(RecipeError::NoWorkStage)
    );
    assert_eq!(
        ExecutionRecipe::new(vec![sym.clone(); MAX_STAGES + 1]),
        Err(RecipeError::TooManyStages {
            count: MAX_STAGES + 1
        })
    );
    let zero = Stage {
        kind: StageKind::Verify,
        budget: StageBudget::new(1, 0),
    };
    assert_eq!(
        ExecutionRecipe::new(vec![sym, zero]),
        Err(RecipeError::ZeroLatencyBudget { stage: 1 })
    );
    assert!(!RecipeError::Empty.to_string().is_empty());
}
