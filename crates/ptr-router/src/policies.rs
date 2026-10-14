//! Separate routing policies (reasoning, model, search, Pod) and the [`PolicySet`] that composes
//! them into an [`ExecutionRecipe`].
//!
//! Every policy receives a list of [`Offer`]s the caller marked eligible or not and may only
//! return an eligible one. [`PolicySet`] does not trust that: it re-checks every answer against
//! the offers (and the model answer against the constraints), so a faulty policy yields a typed
//! [`PolicyError`] instead of a choice outside the eligible set. Policies never see gates,
//! thresholds, verifiers or the sampling key.

use crate::constraints::{
    ConstrainedSelector, ConstraintError, ConstraintsOutcome, Objective, RouteConstraints,
    Selection,
};
use crate::recipe::{
    BackendId, ExecutionRecipe, PlanInputs, RecipeError, RecipePlanner, RequestProfile,
};
use crate::{CheckedCandidate, Offer, RouteOption};
use ptr_types::Probability;
use std::fmt;

/// How the answer is reasoned out.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReasoningMode {
    /// Model answer, then verification.
    Direct,
    /// Model answer followed by a deterministic check.
    Deliberate,
    /// A deterministic planning step before the model.
    SymbolicFirst,
}

impl ReasoningMode {
    pub const ALL: [Self; 3] = [Self::Direct, Self::Deliberate, Self::SymbolicFirst];
}

/// Whether and where to retrieve before generating.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SearchMode {
    Skip,
    Memory,
    Search,
}

impl SearchMode {
    pub const ALL: [Self; 3] = [Self::Skip, Self::Memory, Self::Search];
}

/// Whether the recipe calls the checked Pod.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PodUse {
    Skip,
    Call,
}

impl PodUse {
    pub const ALL: [Self; 2] = [Self::Skip, Self::Call];
}

/// Picks a [`ReasoningMode`] among the eligible offers; `None` when none fits.
pub trait ReasoningPolicy {
    fn choose(
        &self,
        profile: &RequestProfile,
        offers: &[Offer<ReasoningMode>],
    ) -> Option<ReasoningMode>;
}

/// Picks a [`SearchMode`] among the eligible offers.
pub trait SearchPolicy {
    fn choose(&self, profile: &RequestProfile, offers: &[Offer<SearchMode>]) -> Option<SearchMode>;
}

/// Decides whether to call the checked Pod, among the eligible offers.
pub trait PodPolicy {
    fn choose(&self, profile: &RequestProfile, offers: &[Offer<PodUse>]) -> Option<PodUse>;
}

/// Picks a model option among the eligible offers that satisfy `constraints`.
pub trait ModelPolicy {
    fn choose(
        &self,
        profile: &RequestProfile,
        offers: &[Offer<RouteOption>],
        constraints: &RouteConstraints,
    ) -> Result<Selection, ConstraintError>;
}

fn first_eligible<T: Copy + PartialEq>(order: &[T], offers: &[Offer<T>]) -> Option<T> {
    order.iter().copied().find(|wanted| {
        offers
            .iter()
            .any(|offer| offer.eligible && offer.option == *wanted)
    })
}

/// Direct unless uncertainty reaches the trigger (then Deliberate) or the profile asks for a
/// symbolic start; falls through a fixed preference order when the first choice is ineligible.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceReasoningPolicy {
    pub deliberate_at: Probability,
}

impl Default for ReferenceReasoningPolicy {
    fn default() -> Self {
        Self {
            deliberate_at: Probability::new(0.35).unwrap_or_default(),
        }
    }
}

impl ReasoningPolicy for ReferenceReasoningPolicy {
    fn choose(
        &self,
        profile: &RequestProfile,
        offers: &[Offer<ReasoningMode>],
    ) -> Option<ReasoningMode> {
        let order = if profile.symbolic_first {
            [
                ReasoningMode::SymbolicFirst,
                ReasoningMode::Deliberate,
                ReasoningMode::Direct,
            ]
        } else if profile.uncertainty.get() >= self.deliberate_at.get() {
            [
                ReasoningMode::Deliberate,
                ReasoningMode::SymbolicFirst,
                ReasoningMode::Direct,
            ]
        } else {
            [
                ReasoningMode::Direct,
                ReasoningMode::Deliberate,
                ReasoningMode::SymbolicFirst,
            ]
        };
        first_eligible(&order, offers)
    }
}

/// Search when asked, else memory when asked, else nothing; the other modes follow.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceSearchPolicy;

impl SearchPolicy for ReferenceSearchPolicy {
    fn choose(&self, profile: &RequestProfile, offers: &[Offer<SearchMode>]) -> Option<SearchMode> {
        let order = if profile.wants_search {
            [SearchMode::Search, SearchMode::Memory, SearchMode::Skip]
        } else if profile.wants_retrieval {
            [SearchMode::Memory, SearchMode::Search, SearchMode::Skip]
        } else {
            [SearchMode::Skip, SearchMode::Memory, SearchMode::Search]
        };
        first_eligible(&order, offers)
    }
}

/// Calls the Pod exactly when the profile needs one. A needed Pod that is not eligible yields
/// no choice rather than a silent skip.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferencePodPolicy;

impl PodPolicy for ReferencePodPolicy {
    fn choose(&self, profile: &RequestProfile, offers: &[Offer<PodUse>]) -> Option<PodUse> {
        let wanted = if profile.needs_pod {
            PodUse::Call
        } else {
            PodUse::Skip
        };
        first_eligible(&[wanted], offers)
    }
}

/// Constrained deterministic choice by an [`Objective`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceModelPolicy {
    pub objective: Objective,
}

impl ModelPolicy for ReferenceModelPolicy {
    fn choose(
        &self,
        _profile: &RequestProfile,
        offers: &[Offer<RouteOption>],
        constraints: &RouteConstraints,
    ) -> Result<Selection, ConstraintError> {
        ConstrainedSelector::new(*constraints, self.objective).select(offers)
    }
}

/// Which policy an error concerns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyKind {
    Reasoning,
    Search,
    Model,
    Pod,
}

/// A policy produced no usable answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PolicyError {
    NoEligibleOption {
        policy: PolicyKind,
    },
    /// The policy returned an option the caller did not mark eligible.
    ChoseIneligible {
        policy: PolicyKind,
    },
    /// The model policy returned an option that violates the constraints.
    ChoseInfeasible {
        id: BackendId,
    },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoEligibleOption { policy } => write!(f, "{policy:?} policy found no option"),
            Self::ChoseIneligible { policy } => {
                write!(f, "{policy:?} policy chose an ineligible option")
            }
            Self::ChoseInfeasible { id } => write!(f, "model policy chose infeasible {id}"),
        }
    }
}

impl std::error::Error for PolicyError {}

/// Why [`PolicySet::plan`] failed.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanError {
    Policy(PolicyError),
    Constraints(ConstraintError),
    Recipe(RecipeError),
}

impl PlanError {
    /// The telemetry outcome of a constraint failure; `None` for other failures.
    pub fn constraints_outcome(&self) -> Option<ConstraintsOutcome> {
        match self {
            Self::Constraints(ConstraintError::Infeasible { rejections }) => {
                Some(ConstraintsOutcome::Infeasible {
                    options: rejections.len(),
                })
            }
            Self::Policy(_) | Self::Constraints(_) | Self::Recipe(_) => None,
        }
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(error) => write!(f, "policy: {error}"),
            Self::Constraints(error) => write!(f, "constraints: {error}"),
            Self::Recipe(error) => write!(f, "recipe: {error}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// Everything [`PolicySet::plan`] chooses among.
#[derive(Clone, Debug)]
pub struct PolicyInputs<'a> {
    pub profile: &'a RequestProfile,
    pub reasoning: &'a [Offer<ReasoningMode>],
    pub search: &'a [Offer<SearchMode>],
    pub models: &'a [Offer<RouteOption>],
    pub pod: &'a [Offer<PodUse>],
    pub constraints: &'a RouteConstraints,
    /// The checked routing result; `PodUse::Call` is offered as eligible only if it is present.
    pub candidate: Option<&'a CheckedCandidate>,
}

/// The decisions behind a recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedRoute {
    pub recipe: ExecutionRecipe,
    pub reasoning: ReasoningMode,
    pub search: SearchMode,
    pub model: Option<BackendId>,
    pub pod: PodUse,
    pub constraints: ConstraintsOutcome,
}

/// Composes the four policies and the planner.
pub struct PolicySet {
    reasoning: Box<dyn ReasoningPolicy + Send + Sync>,
    search: Box<dyn SearchPolicy + Send + Sync>,
    model: Box<dyn ModelPolicy + Send + Sync>,
    pod: Box<dyn PodPolicy + Send + Sync>,
    planner: RecipePlanner,
}

impl PolicySet {
    /// The deterministic reference policies with the default planner budgets.
    pub fn reference() -> Self {
        Self {
            reasoning: Box::new(ReferenceReasoningPolicy::default()),
            search: Box::new(ReferenceSearchPolicy),
            model: Box::new(ReferenceModelPolicy::default()),
            pod: Box::new(ReferencePodPolicy),
            planner: RecipePlanner::default(),
        }
    }

    pub fn with_reasoning(mut self, policy: impl ReasoningPolicy + Send + Sync + 'static) -> Self {
        self.reasoning = Box::new(policy);
        self
    }

    pub fn with_search(mut self, policy: impl SearchPolicy + Send + Sync + 'static) -> Self {
        self.search = Box::new(policy);
        self
    }

    pub fn with_model(mut self, policy: impl ModelPolicy + Send + Sync + 'static) -> Self {
        self.model = Box::new(policy);
        self
    }

    pub fn with_pod(mut self, policy: impl PodPolicy + Send + Sync + 'static) -> Self {
        self.pod = Box::new(policy);
        self
    }

    pub fn with_planner(mut self, planner: RecipePlanner) -> Self {
        self.planner = planner;
        self
    }

    pub fn plan(&self, inputs: &PolicyInputs<'_>) -> Result<PlannedRoute, PlanError> {
        let profile = inputs.profile;
        let reasoning = verified(
            PolicyKind::Reasoning,
            self.reasoning.choose(profile, inputs.reasoning),
            inputs.reasoning,
        )?;
        let search = verified(
            PolicyKind::Search,
            self.search.choose(profile, inputs.search),
            inputs.search,
        )?;
        let pod_offers: Vec<Offer<PodUse>> = inputs
            .pod
            .iter()
            .map(|offer| Offer {
                option: offer.option,
                eligible: offer.eligible
                    && (offer.option != PodUse::Call || inputs.candidate.is_some()),
            })
            .collect();
        let pod = verified(
            PolicyKind::Pod,
            self.pod.choose(profile, &pod_offers),
            &pod_offers,
        )?;

        let (model, outcome) = match profile.model {
            None => (None, ConstraintsOutcome::NotApplied),
            Some(required) => {
                let constraints = inputs.constraints.requiring(required);
                let selection = self
                    .model
                    .choose(profile, inputs.models, &constraints)
                    .map_err(PlanError::Constraints)?;
                let offer = inputs
                    .models
                    .iter()
                    .find(|offer| offer.option.id() == selection.chosen.id())
                    .filter(|offer| offer.eligible)
                    .ok_or(PlanError::Policy(PolicyError::ChoseIneligible {
                        policy: PolicyKind::Model,
                    }))?;
                if !constraints.violations(&offer.option).is_empty() {
                    return Err(PlanError::Policy(PolicyError::ChoseInfeasible {
                        id: offer.option.id().clone(),
                    }));
                }
                let selector = ConstrainedSelector::new(constraints, Objective::default());
                let outcome = ConstraintsOutcome::Satisfied {
                    eligible: inputs.models.iter().filter(|o| o.eligible).count(),
                    feasible: selector.feasible(inputs.models).len(),
                };
                (Some(&offer.option), outcome)
            }
        };

        let recipe = self
            .planner
            .plan(&PlanInputs {
                profile,
                reasoning,
                search,
                model,
                pod_use: pod,
                candidate: inputs.candidate,
            })
            .map_err(PlanError::Recipe)?;
        Ok(PlannedRoute {
            recipe,
            reasoning,
            search,
            model: model.map(|option| option.id().clone()),
            pod,
            constraints: outcome,
        })
    }
}

fn verified<T: Copy + PartialEq>(
    policy: PolicyKind,
    choice: Option<T>,
    offers: &[Offer<T>],
) -> Result<T, PlanError> {
    let choice = choice.ok_or(PlanError::Policy(PolicyError::NoEligibleOption { policy }))?;
    if offers
        .iter()
        .any(|offer| offer.eligible && offer.option == choice)
    {
        Ok(choice)
    } else {
        Err(PlanError::Policy(PolicyError::ChoseIneligible { policy }))
    }
}
