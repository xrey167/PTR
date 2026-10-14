mod common;

use common::{backend, candidate, caps, option, prob};
use ptr_router::{
    BackendCapability, ConstraintError, ConstraintsOutcome, Infeasibility, ModelPolicy, Offer,
    PlanError, PodPolicy, PodUse, PolicyError, PolicyInputs, PolicyKind, PolicySet, ReasoningMode,
    ReasoningPolicy, ReferenceReasoningPolicy, RequestProfile, RouteConstraints, RouteOption,
    SearchMode, SearchPolicy, Selection, StageEffect, StageKind,
};

fn all<T: Copy>(items: &[T]) -> Vec<Offer<T>> {
    items.iter().map(|i| Offer::eligible(*i)).collect()
}

struct Fixture {
    reasoning: Vec<Offer<ReasoningMode>>,
    search: Vec<Offer<SearchMode>>,
    pod: Vec<Offer<PodUse>>,
    models: Vec<Offer<RouteOption>>,
    constraints: RouteConstraints,
}

impl Fixture {
    fn new() -> Self {
        Self {
            reasoning: all(&ReasoningMode::ALL),
            search: all(&SearchMode::ALL),
            pod: all(&PodUse::ALL),
            models: vec![
                Offer::eligible(option("big", 9, 900, 0.1, &[BackendCapability::Reasoning])),
                Offer::eligible(option("small", 1, 50, 0.4, &[BackendCapability::Reasoning])),
            ],
            constraints: RouteConstraints::unbounded(),
        }
    }

    fn inputs<'a>(
        &'a self,
        profile: &'a RequestProfile,
        candidate: Option<&'a ptr_router::CheckedCandidate>,
    ) -> PolicyInputs<'a> {
        PolicyInputs {
            profile,
            reasoning: &self.reasoning,
            search: &self.search,
            models: &self.models,
            pod: &self.pod,
            constraints: &self.constraints,
            candidate,
        }
    }
}

fn model_profile() -> RequestProfile {
    RequestProfile {
        model: Some(caps(&[BackendCapability::Reasoning])),
        ..RequestProfile::default()
    }
}

#[test]
fn reference_set_composes_a_recipe_from_the_four_policies() {
    let fixture = Fixture::new();
    let candidate = candidate("r", "zeta");
    let profile = RequestProfile {
        wants_search: true,
        needs_pod: true,
        pod_effect: StageEffect::Effectful,
        uncertainty: prob(0.9),
        ..model_profile()
    };
    let planned = PolicySet::reference()
        .plan(&fixture.inputs(&profile, Some(&candidate)))
        .unwrap();
    assert_eq!(planned.reasoning, ReasoningMode::Deliberate);
    assert_eq!(planned.search, SearchMode::Search);
    assert_eq!(planned.pod, PodUse::Call);
    // Default objective is lexicographic on uncertainty first.
    assert_eq!(planned.model, Some(backend("big")));
    assert_eq!(
        planned.constraints,
        ConstraintsOutcome::Satisfied {
            eligible: 2,
            feasible: 2
        }
    );
    let last = planned.recipe.stages().last().unwrap();
    assert_eq!(last.kind, StageKind::Verify);
    assert!(planned
        .recipe
        .stages()
        .iter()
        .any(|s| matches!(&s.kind, StageKind::PodCall { pod, .. } if pod.0 == "zeta")));
}

#[test]
fn planning_is_deterministic() {
    let fixture = Fixture::new();
    let profile = model_profile();
    let set = PolicySet::reference();
    let first = set.plan(&fixture.inputs(&profile, None)).unwrap();
    for _ in 0..5 {
        assert_eq!(set.plan(&fixture.inputs(&profile, None)).unwrap(), first);
    }
}

#[test]
fn reference_policies_choose_only_eligible_offers() {
    let mut fixture = Fixture::new();
    fixture.reasoning = vec![
        Offer::ineligible(ReasoningMode::Direct),
        Offer::eligible(ReasoningMode::SymbolicFirst),
    ];
    fixture.search = vec![
        Offer::ineligible(SearchMode::Skip),
        Offer::ineligible(SearchMode::Search),
        Offer::eligible(SearchMode::Memory),
    ];
    fixture.pod = vec![
        Offer::eligible(PodUse::Skip),
        Offer::ineligible(PodUse::Call),
    ];
    fixture.models[0].eligible = false;
    let profile = RequestProfile {
        wants_search: true,
        ..model_profile()
    };
    let planned = PolicySet::reference()
        .plan(&fixture.inputs(&profile, None))
        .unwrap();
    assert_eq!(planned.reasoning, ReasoningMode::SymbolicFirst);
    assert_eq!(planned.search, SearchMode::Memory);
    assert_eq!(planned.pod, PodUse::Skip);
    assert_eq!(planned.model, Some(backend("small")));
}

#[test]
fn reference_reasoning_follows_the_uncertainty_trigger() {
    let policy = ReferenceReasoningPolicy::default();
    let offers = all(&ReasoningMode::ALL);
    let at = |u: f32| {
        policy.choose(
            &RequestProfile {
                uncertainty: prob(u),
                ..RequestProfile::default()
            },
            &offers,
        )
    };
    assert_eq!(at(0.34), Some(ReasoningMode::Direct));
    assert_eq!(at(0.35), Some(ReasoningMode::Deliberate));
    assert_eq!(policy.choose(&RequestProfile::default(), &[]), None);
}

#[test]
fn a_needed_pod_without_a_candidate_is_an_error_not_a_silent_skip() {
    let fixture = Fixture::new();
    let profile = RequestProfile {
        needs_pod: true,
        ..model_profile()
    };
    let error = PolicySet::reference()
        .plan(&fixture.inputs(&profile, None))
        .unwrap_err();
    assert_eq!(
        error,
        PlanError::Policy(PolicyError::NoEligibleOption {
            policy: PolicyKind::Pod
        })
    );
}

#[test]
fn empty_eligible_sets_are_typed_errors() {
    let mut fixture = Fixture::new();
    fixture.reasoning = vec![Offer::ineligible(ReasoningMode::Direct)];
    let error = PolicySet::reference()
        .plan(&fixture.inputs(&model_profile(), None))
        .unwrap_err();
    assert_eq!(
        error,
        PlanError::Policy(PolicyError::NoEligibleOption {
            policy: PolicyKind::Reasoning
        })
    );
}

struct Rogue;
impl ReasoningPolicy for Rogue {
    fn choose(&self, _: &RequestProfile, _: &[Offer<ReasoningMode>]) -> Option<ReasoningMode> {
        Some(ReasoningMode::Deliberate)
    }
}
impl SearchPolicy for Rogue {
    fn choose(&self, _: &RequestProfile, _: &[Offer<SearchMode>]) -> Option<SearchMode> {
        Some(SearchMode::Search)
    }
}
impl PodPolicy for Rogue {
    fn choose(&self, _: &RequestProfile, _: &[Offer<PodUse>]) -> Option<PodUse> {
        Some(PodUse::Call)
    }
}

#[test]
fn a_policy_naming_an_ineligible_option_is_caught() {
    let mut fixture = Fixture::new();
    fixture.reasoning = vec![Offer::eligible(ReasoningMode::Direct)];
    let profile = model_profile();
    let chose = |policy| PlanError::Policy(PolicyError::ChoseIneligible { policy });
    assert_eq!(
        PolicySet::reference()
            .with_reasoning(Rogue)
            .plan(&fixture.inputs(&profile, None))
            .unwrap_err(),
        chose(PolicyKind::Reasoning)
    );

    let mut fixture = Fixture::new();
    fixture.search = vec![Offer::eligible(SearchMode::Skip)];
    assert_eq!(
        PolicySet::reference()
            .with_search(Rogue)
            .plan(&fixture.inputs(&profile, None))
            .unwrap_err(),
        chose(PolicyKind::Search)
    );

    // Call is offered but no checked candidate exists, so it is not eligible.
    let fixture = Fixture::new();
    assert_eq!(
        PolicySet::reference()
            .with_pod(Rogue)
            .plan(&fixture.inputs(&profile, None))
            .unwrap_err(),
        chose(PolicyKind::Pod)
    );
}

struct PicksByName(&'static str);
impl ModelPolicy for PicksByName {
    fn choose(
        &self,
        _: &RequestProfile,
        offers: &[Offer<RouteOption>],
        _: &RouteConstraints,
    ) -> Result<Selection, ConstraintError> {
        let chosen = offers
            .iter()
            .map(|o| o.option.clone())
            .find(|o| o.id().as_str() == self.0)
            .unwrap_or_else(|| option(self.0, 0, 1, 0.0, &[]));
        Ok(Selection {
            chosen,
            eligible: 0,
            feasible: 0,
        })
    }
}

#[test]
fn a_model_policy_cannot_escape_eligibility_or_constraints() {
    let mut fixture = Fixture::new();
    fixture.models.push(Offer::ineligible(option(
        "banned",
        0,
        1,
        0.0,
        &[BackendCapability::Reasoning],
    )));
    fixture.constraints.max_cost_units = 5;
    let profile = model_profile();
    let plan = |name: &'static str| {
        PolicySet::reference()
            .with_model(PicksByName(name))
            .plan(&fixture.inputs(&profile, None))
            .unwrap_err()
    };
    assert_eq!(
        plan("banned"),
        PlanError::Policy(PolicyError::ChoseIneligible {
            policy: PolicyKind::Model
        })
    );
    assert_eq!(
        plan("unknown"),
        PlanError::Policy(PolicyError::ChoseIneligible {
            policy: PolicyKind::Model
        })
    );
    // "big" costs 9 > 5: eligible but infeasible.
    assert_eq!(
        plan("big"),
        PlanError::Policy(PolicyError::ChoseInfeasible { id: backend("big") })
    );
}

#[test]
fn a_model_policy_cannot_substitute_its_own_estimates() {
    struct Lies;
    impl ModelPolicy for Lies {
        fn choose(
            &self,
            _: &RequestProfile,
            _: &[Offer<RouteOption>],
            _: &RouteConstraints,
        ) -> Result<Selection, ConstraintError> {
            Ok(Selection {
                chosen: option("big", 0, 1, 0.0, &[BackendCapability::Reasoning]),
                eligible: 0,
                feasible: 0,
            })
        }
    }
    let mut fixture = Fixture::new();
    fixture.constraints.max_cost_units = 5;
    let profile = model_profile();
    // The set checks the policy's id against the caller's offer ("big" costs 9), not its claim.
    let error = PolicySet::reference()
        .with_model(Lies)
        .plan(&fixture.inputs(&profile, None))
        .unwrap_err();
    assert_eq!(
        error,
        PlanError::Policy(PolicyError::ChoseInfeasible { id: backend("big") })
    );
}

#[test]
fn profile_capabilities_tighten_the_constraints() {
    let mut fixture = Fixture::new();
    fixture.models = vec![
        Offer::eligible(option("plain", 1, 1, 0.1, &[])),
        Offer::eligible(option("coder", 5, 5, 0.1, &[BackendCapability::Code])),
    ];
    let profile = RequestProfile {
        model: Some(caps(&[BackendCapability::Code])),
        ..RequestProfile::default()
    };
    let planned = PolicySet::reference()
        .plan(&fixture.inputs(&profile, None))
        .unwrap();
    assert_eq!(planned.model, Some(backend("coder")));
}

#[test]
fn infeasible_for_all_surfaces_reasons_and_a_telemetry_outcome() {
    let mut fixture = Fixture::new();
    fixture.constraints.max_latency_ms = 10;
    let profile = model_profile();
    let error = PolicySet::reference()
        .plan(&fixture.inputs(&profile, None))
        .unwrap_err();
    assert_eq!(
        error.constraints_outcome(),
        Some(ConstraintsOutcome::Infeasible { options: 2 })
    );
    let PlanError::Constraints(ConstraintError::Infeasible { rejections }) = error else {
        panic!("expected constraint failure");
    };
    assert!(rejections.iter().all(|r| matches!(
        r.reasons.as_slice(),
        [Infeasibility::LatencyExceeded { limit: 10, .. }]
    )));
    assert!(PlanError::Policy(PolicyError::NoEligibleOption {
        policy: PolicyKind::Pod
    })
    .constraints_outcome()
    .is_none());
}

#[test]
fn no_model_requested_skips_the_model_policy() {
    let mut fixture = Fixture::new();
    fixture.models.clear();
    let planned = PolicySet::reference()
        .plan(&fixture.inputs(&RequestProfile::default(), None))
        .unwrap();
    assert_eq!(planned.model, None);
    assert_eq!(planned.constraints, ConstraintsOutcome::NotApplied);
}
