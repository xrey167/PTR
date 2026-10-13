//! Fallback and degradation planning against a caller-supplied view of backend availability.
//!
//! For each stage of a recipe the planner lists the alternatives that the [`AvailabilityView`]
//! allows, ordered by [`DegradationLevel`]. It then resolves the whole recipe at the least
//! degraded level at which every stage still has an alternative. The verifier stage has no
//! degraded form and is never replaced or omitted: when it is unavailable, or when an effectful
//! stage has no usable alternative, the result is [`Resolution::Escalate`], not a recipe
//! without verification.
//!
//! The planner only substitutes or omits stages of a recipe that was already built from
//! eligible choices; it never introduces a Pod, a backend or an effect the recipe did not
//! contain. Availability is the caller's view and unknown backends are the caller's to report
//! (see [`StaticAvailability`], which treats unknown ones as its default state).

use crate::recipe::{
    ContextMode, ExecutionRecipe, RecipeError, RetrievalSource, Stage, StageBudget, StageEffect,
    StageKind, SymbolicStep,
};
use crate::BackendId;
use ptr_types::PodId;
use std::collections::BTreeMap;

/// How far a plan has been reduced, from best to worst.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DegradationLevel {
    /// Every stage as planned.
    Full,
    /// Stages run with half budgets and, for model calls, a reduced context.
    ReducedContext,
    /// Model, search and pure Pod stages are replaced by deterministic symbolic checks.
    SymbolicOnly,
    /// No safe automatic plan; the request goes to a human or a higher authority.
    Escalate,
}

/// Availability of one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendState {
    Up,
    /// Usable with reduced capacity: reduced context and half budgets.
    Degraded,
    Down,
}

/// The caller's current view of what can run.
pub trait AvailabilityView {
    fn model(&self, backend: &BackendId) -> BackendState;
    fn pod(&self, pod: &PodId) -> BackendState;
    fn retrieval(&self, source: RetrievalSource) -> BackendState;
    /// The verifier must be `Up`; `Degraded` is treated like `Down` because a verifier has no
    /// weaker form.
    fn verifier(&self) -> BackendState;
}

/// A fixed availability table; backends it does not list get the default state.
#[derive(Clone, Debug)]
pub struct StaticAvailability {
    default: BackendState,
    models: BTreeMap<BackendId, BackendState>,
    pods: BTreeMap<PodId, BackendState>,
    memory: Option<BackendState>,
    search: Option<BackendState>,
    verifier: Option<BackendState>,
}

impl StaticAvailability {
    /// Everything not listed is in `default`.
    pub fn all(default: BackendState) -> Self {
        Self {
            default,
            models: BTreeMap::new(),
            pods: BTreeMap::new(),
            memory: None,
            search: None,
            verifier: None,
        }
    }

    pub fn with_model(mut self, backend: BackendId, state: BackendState) -> Self {
        self.models.insert(backend, state);
        self
    }

    pub fn with_pod(mut self, pod: PodId, state: BackendState) -> Self {
        self.pods.insert(pod, state);
        self
    }

    pub fn with_retrieval(mut self, source: RetrievalSource, state: BackendState) -> Self {
        match source {
            RetrievalSource::Memory => self.memory = Some(state),
            RetrievalSource::Search => self.search = Some(state),
        }
        self
    }

    pub fn with_verifier(mut self, state: BackendState) -> Self {
        self.verifier = Some(state);
        self
    }
}

impl AvailabilityView for StaticAvailability {
    fn model(&self, backend: &BackendId) -> BackendState {
        self.models.get(backend).copied().unwrap_or(self.default)
    }
    fn pod(&self, pod: &PodId) -> BackendState {
        self.pods.get(pod).copied().unwrap_or(self.default)
    }
    fn retrieval(&self, source: RetrievalSource) -> BackendState {
        match source {
            RetrievalSource::Memory => self.memory,
            RetrievalSource::Search => self.search,
        }
        .unwrap_or(self.default)
    }
    fn verifier(&self) -> BackendState {
        self.verifier.unwrap_or(self.default)
    }
}

/// One way to run a stage. `stage == None` means the stage is explicitly omitted at this
/// level (never produced for a verifier stage).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Alternative {
    pub level: DegradationLevel,
    pub stage: Option<Stage>,
}

/// The ordered fallback chain of one stage of the original recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageChain {
    pub index: usize,
    pub alternatives: Vec<Alternative>,
}

/// Why the plan escalates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DegradationEscalation {
    VerifierUnavailable,
    /// The stage at `stage` has no alternative at any level (for example an effectful Pod call
    /// whose Pod is down).
    StageUnavailable {
        stage: usize,
    },
    /// Substitution produced a recipe that fails validation.
    InvalidRecipe(RecipeError),
}

/// The outcome of resolving a plan against availability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Resolution {
    Run {
        level: DegradationLevel,
        recipe: ExecutionRecipe,
    },
    Escalate {
        reason: DegradationEscalation,
    },
}

/// Per-stage chains plus the resolved plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FallbackPlan {
    pub chains: Vec<StageChain>,
    pub resolution: Resolution,
}

impl FallbackPlan {
    /// The level of the resolved plan; [`DegradationLevel::Escalate`] when it escalates.
    pub fn level(&self) -> DegradationLevel {
        match &self.resolution {
            Resolution::Run { level, .. } => *level,
            Resolution::Escalate { .. } => DegradationLevel::Escalate,
        }
    }
}

/// Deterministic fallback planner.
#[derive(Clone, Copy, Debug, Default)]
pub struct FallbackPlanner;

const SYMBOLIC_FALLBACK: StageBudget = StageBudget::new(1, 100);
const RESOLVE_ORDER: [DegradationLevel; 3] = [
    DegradationLevel::Full,
    DegradationLevel::ReducedContext,
    DegradationLevel::SymbolicOnly,
];

impl FallbackPlanner {
    pub fn plan(&self, recipe: &ExecutionRecipe, view: &dyn AvailabilityView) -> FallbackPlan {
        let chains: Vec<StageChain> = recipe
            .stages()
            .iter()
            .enumerate()
            .map(|(index, stage)| StageChain {
                index,
                alternatives: alternatives(stage, view),
            })
            .collect();
        let resolution = resolve(recipe, &chains);
        FallbackPlan { chains, resolution }
    }
}

fn state_of(stage: &Stage, view: &dyn AvailabilityView) -> BackendState {
    match &stage.kind {
        StageKind::Retrieve { source } => view.retrieval(*source),
        StageKind::ModelCall { requirement } => view.model(&requirement.backend),
        StageKind::PodCall { pod, .. } => view.pod(pod),
        StageKind::Symbolic { .. } => BackendState::Up,
        StageKind::Verify => view.verifier(),
    }
}

fn reduced(stage: &Stage) -> Option<Stage> {
    let kind = match &stage.kind {
        StageKind::Retrieve { .. } | StageKind::PodCall { .. } => stage.kind.clone(),
        StageKind::ModelCall { requirement } => {
            let mut requirement = requirement.clone();
            requirement.context = ContextMode::Reduced;
            StageKind::ModelCall { requirement }
        }
        StageKind::Symbolic { .. } | StageKind::Verify => return None,
    };
    Some(Stage {
        kind,
        budget: stage.budget.halved(),
    })
}

fn symbolic_only(stage: &Stage) -> Option<Option<Stage>> {
    let check = || {
        Some(Some(Stage {
            kind: StageKind::Symbolic {
                step: SymbolicStep::Check,
            },
            budget: SYMBOLIC_FALLBACK,
        }))
    };
    match &stage.kind {
        StageKind::Retrieve { .. } => Some(None),
        StageKind::ModelCall { .. } => check(),
        StageKind::PodCall {
            effect: StageEffect::Pure,
            ..
        } => check(),
        StageKind::PodCall {
            effect: StageEffect::Effectful,
            ..
        }
        | StageKind::Symbolic { .. }
        | StageKind::Verify => None,
    }
}

fn alternatives(stage: &Stage, view: &dyn AvailabilityView) -> Vec<Alternative> {
    let state = state_of(stage, view);
    let mut chain = Vec::new();
    if state == BackendState::Up {
        chain.push(Alternative {
            level: DegradationLevel::Full,
            stage: Some(stage.clone()),
        });
    }
    if matches!(state, BackendState::Up | BackendState::Degraded) {
        if let Some(smaller) = reduced(stage) {
            chain.push(Alternative {
                level: DegradationLevel::ReducedContext,
                stage: Some(smaller),
            });
        }
    }
    if let Some(replacement) = symbolic_only(stage) {
        chain.push(Alternative {
            level: DegradationLevel::SymbolicOnly,
            stage: replacement,
        });
    }
    chain
}

fn resolve(recipe: &ExecutionRecipe, chains: &[StageChain]) -> Resolution {
    let verifier_lost = recipe
        .stages()
        .iter()
        .zip(chains)
        .any(|(stage, chain)| stage.kind == StageKind::Verify && chain.alternatives.is_empty());
    if verifier_lost {
        return Resolution::Escalate {
            reason: DegradationEscalation::VerifierUnavailable,
        };
    }
    let mut invalid = None;
    for level in RESOLVE_ORDER {
        let mut stages = Vec::new();
        let mut complete = true;
        for chain in chains {
            match chain.alternatives.iter().find(|alt| alt.level <= level) {
                Some(alt) => stages.extend(alt.stage.clone()),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        match ExecutionRecipe::new(stages) {
            Ok(recipe) => return Resolution::Run { level, recipe },
            Err(error) => invalid = Some(error),
        }
    }
    if let Some(chain) = chains.iter().find(|chain| chain.alternatives.is_empty()) {
        return Resolution::Escalate {
            reason: DegradationEscalation::StageUnavailable { stage: chain.index },
        };
    }
    let reason = match invalid {
        Some(error) => DegradationEscalation::InvalidRecipe(error),
        // Every chain is non-empty and some level resolved or recorded an error above.
        None => DegradationEscalation::StageUnavailable { stage: 0 },
    };
    Resolution::Escalate { reason }
}
