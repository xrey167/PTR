//! A typed, validated execution plan and the deterministic planner that builds it.
//!
//! An [`ExecutionRecipe`] is an ordered list of [`Stage`]s with per-stage budgets. Validation
//! requires a verifier stage after every effectful stage and a verifier stage as the terminal
//! stage, so a recipe can never end in unverified output. The planner assembles recipes from
//! choices that were already made among eligible options (see `policies`); it adds no
//! authority of its own and cannot name a Pod except through a [`CheckedCandidate`].

use crate::policies::{PodUse, ReasoningMode, SearchMode};
use crate::{CheckedCandidate, RouteOption};
use ptr_types::{CapabilityId, PodId, Probability};
use std::fmt;

/// Maximum stages in one recipe (Invariant 8).
pub const MAX_STAGES: usize = 16;
/// Maximum byte length of a [`BackendId`].
pub const MAX_BACKEND_ID_BYTES: usize = 64;

/// Identifier of a model backend or routing option: 1..=64 printable ASCII bytes without
/// whitespace, quotes or backslashes.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BackendId(String);

/// A rejected [`BackendId`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendIdError {
    Empty,
    TooLong { len: usize },
    InvalidByte { index: usize },
}

impl fmt::Display for BackendIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "backend id is empty"),
            Self::TooLong { len } => {
                write!(
                    f,
                    "backend id has {len} bytes, limit {MAX_BACKEND_ID_BYTES}"
                )
            }
            Self::InvalidByte { index } => write!(f, "backend id has an invalid byte at {index}"),
        }
    }
}

impl std::error::Error for BackendIdError {}

impl BackendId {
    pub fn new(value: &str) -> Result<Self, BackendIdError> {
        if value.is_empty() {
            return Err(BackendIdError::Empty);
        }
        if value.len() > MAX_BACKEND_ID_BYTES {
            return Err(BackendIdError::TooLong { len: value.len() });
        }
        if let Some(index) = value
            .bytes()
            .position(|b| !b.is_ascii_graphic() || b == b'"' || b == b'\\')
        {
            return Err(BackendIdError::InvalidByte { index });
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A capability a model backend may offer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BackendCapability {
    Reasoning,
    Code,
    ToolUse,
    LongContext,
    StructuredOutput,
}

impl BackendCapability {
    pub const ALL: [Self; 5] = [
        Self::Reasoning,
        Self::Code,
        Self::ToolUse,
        Self::LongContext,
        Self::StructuredOutput,
    ];

    fn bit(self) -> u8 {
        match self {
            Self::Reasoning => 1,
            Self::Code => 2,
            Self::ToolUse => 4,
            Self::LongContext => 8,
            Self::StructuredOutput => 16,
        }
    }
}

/// A set of [`BackendCapability`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn with(self, capability: BackendCapability) -> Self {
        Self(self.0 | capability.bit())
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub fn contains(self, capability: BackendCapability) -> bool {
        self.0 & capability.bit() != 0
    }

    /// The members of `self` that `have` lacks.
    pub fn missing_from(self, have: Self) -> Self {
        Self(self.0 & !have.0)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = BackendCapability> {
        BackendCapability::ALL
            .into_iter()
            .filter(move |capability| self.contains(*capability))
    }
}

/// How much context a model call gets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextMode {
    Full,
    Reduced,
}

/// What a model-call stage needs from its backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendRequirement {
    pub backend: BackendId,
    pub capabilities: CapabilitySet,
    pub context: ContextMode,
}

/// Where a retrieval stage reads from. Retrieved material is derived evidence, never truth.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RetrievalSource {
    Memory,
    Search,
}

/// A deterministic non-neural step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicStep {
    /// Decompose or constrain the task before a model sees it.
    Plan,
    /// Apply deterministic rules to what earlier stages produced.
    Check,
}

/// Whether a stage changes anything outside the recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageEffect {
    Pure,
    Effectful,
}

/// The work of one stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageKind {
    Retrieve {
        source: RetrievalSource,
    },
    ModelCall {
        requirement: BackendRequirement,
    },
    PodCall {
        class: CapabilityId,
        pod: PodId,
        effect: StageEffect,
    },
    Symbolic {
        step: SymbolicStep,
    },
    /// A deterministic verifier pass. A recipe must end in one.
    Verify,
}

impl StageKind {
    pub fn effect(&self) -> StageEffect {
        match self {
            Self::PodCall { effect, .. } => *effect,
            Self::Retrieve { .. }
            | Self::ModelCall { .. }
            | Self::Symbolic { .. }
            | Self::Verify => StageEffect::Pure,
        }
    }
}

/// Cost and latency allowance of one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageBudget {
    cost_units: u32,
    latency_ms: u32,
}

impl StageBudget {
    pub const fn new(cost_units: u32, latency_ms: u32) -> Self {
        Self {
            cost_units,
            latency_ms,
        }
    }

    pub fn cost_units(&self) -> u32 {
        self.cost_units
    }

    pub fn latency_ms(&self) -> u32 {
        self.latency_ms
    }

    /// Half the budget, never below one millisecond.
    pub(crate) fn halved(self) -> Self {
        Self {
            cost_units: self.cost_units / 2,
            latency_ms: (self.latency_ms / 2).max(1),
        }
    }
}

/// One step of a recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage {
    pub kind: StageKind,
    pub budget: StageBudget,
}

/// Why a recipe or plan was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecipeError {
    Empty,
    TooManyStages {
        count: usize,
    },
    ZeroLatencyBudget {
        stage: usize,
    },
    /// The effectful stage at `stage` has no verifier stage after it.
    EffectWithoutVerifier {
        stage: usize,
    },
    TerminalStageNotVerifier,
    /// The recipe has nothing but verification.
    NoWorkStage,
    /// A Pod call was requested without a [`CheckedCandidate`].
    MissingCheckedCandidate,
}

impl fmt::Display for RecipeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "recipe has no stages"),
            Self::TooManyStages { count } => {
                write!(f, "recipe has {count} stages, limit {MAX_STAGES}")
            }
            Self::ZeroLatencyBudget { stage } => {
                write!(f, "stage {stage} has a zero latency budget")
            }
            Self::EffectWithoutVerifier { stage } => {
                write!(
                    f,
                    "effectful stage {stage} is not followed by a verifier stage"
                )
            }
            Self::TerminalStageNotVerifier => write!(f, "the terminal stage is not a verifier"),
            Self::NoWorkStage => write!(f, "recipe contains only verification"),
            Self::MissingCheckedCandidate => {
                write!(f, "a Pod call needs a checked routing candidate")
            }
        }
    }
}

impl std::error::Error for RecipeError {}

/// An ordered, validated plan. Construct with [`ExecutionRecipe::new`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionRecipe {
    stages: Vec<Stage>,
}

impl ExecutionRecipe {
    /// Validates in this order: non-empty, at most [`MAX_STAGES`], every latency budget
    /// positive, a verifier stage after every effectful stage, a verifier as terminal stage,
    /// and at least one non-verifier stage.
    pub fn new(stages: Vec<Stage>) -> Result<Self, RecipeError> {
        if stages.is_empty() {
            return Err(RecipeError::Empty);
        }
        if stages.len() > MAX_STAGES {
            return Err(RecipeError::TooManyStages {
                count: stages.len(),
            });
        }
        if let Some(stage) = stages.iter().position(|s| s.budget.latency_ms == 0) {
            return Err(RecipeError::ZeroLatencyBudget { stage });
        }
        for (index, stage) in stages.iter().enumerate() {
            let verified_later = stages[index + 1..]
                .iter()
                .any(|later| later.kind == StageKind::Verify);
            if stage.kind.effect() == StageEffect::Effectful && !verified_later {
                return Err(RecipeError::EffectWithoutVerifier { stage: index });
            }
        }
        if stages.last().map(|s| &s.kind) != Some(&StageKind::Verify) {
            return Err(RecipeError::TerminalStageNotVerifier);
        }
        if stages.iter().all(|s| s.kind == StageKind::Verify) {
            return Err(RecipeError::NoWorkStage);
        }
        Ok(Self { stages })
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Sum of the stage cost budgets, saturating.
    pub fn total_cost_units(&self) -> u64 {
        self.stages
            .iter()
            .map(|s| u64::from(s.budget.cost_units))
            .sum()
    }

    /// Sum of the stage latency budgets (stages run in order), saturating.
    pub fn total_latency_ms(&self) -> u64 {
        self.stages
            .iter()
            .map(|s| u64::from(s.budget.latency_ms))
            .sum()
    }
}

/// What the request needs, as far as the policies and the planner care.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestProfile {
    /// Prefer reading from memory before generating.
    pub wants_retrieval: bool,
    /// Prefer a search before generating.
    pub wants_search: bool,
    /// Capabilities the model call must have; `None` means no model call is needed.
    pub model: Option<CapabilitySet>,
    /// A Pod of the checked candidate's class is required.
    pub needs_pod: bool,
    /// Whether that Pod call is effectful.
    pub pod_effect: StageEffect,
    /// Run a deterministic planning step before the model.
    pub symbolic_first: bool,
    /// Current uncertainty about the request.
    pub uncertainty: Probability,
}

impl Default for RequestProfile {
    fn default() -> Self {
        Self {
            wants_retrieval: false,
            wants_search: false,
            model: None,
            needs_pod: false,
            pod_effect: StageEffect::Pure,
            symbolic_first: false,
            uncertainty: Probability::default(),
        }
    }
}

/// Per-stage-type budgets the planner assigns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannerBudgets {
    pub retrieve: StageBudget,
    pub search: StageBudget,
    pub pod: StageBudget,
    pub symbolic: StageBudget,
    pub verify: StageBudget,
    /// Used for a model call when the chosen option carries no estimate of its own.
    pub model_floor_ms: u32,
}

impl Default for PlannerBudgets {
    fn default() -> Self {
        Self {
            retrieve: StageBudget::new(1, 200),
            search: StageBudget::new(2, 500),
            pod: StageBudget::new(5, 2_000),
            symbolic: StageBudget::new(1, 100),
            verify: StageBudget::new(1, 500),
            model_floor_ms: 1,
        }
    }
}

/// Choices already made among eligible options.
#[derive(Clone, Debug)]
pub struct PlanInputs<'a> {
    pub profile: &'a RequestProfile,
    pub reasoning: ReasoningMode,
    pub search: SearchMode,
    pub model: Option<&'a RouteOption>,
    pub pod_use: PodUse,
    /// Required when `pod_use` is [`PodUse::Call`]; the only way to name a Pod.
    pub candidate: Option<&'a CheckedCandidate>,
}

/// Builds recipes deterministically: the same inputs give the same recipe.
///
/// Order: retrieval, optional symbolic plan, model call, optional symbolic check, Pod call,
/// terminal verification. A request that selects no work at all gets one symbolic check so the
/// recipe never contains only verification.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecipePlanner {
    budgets: PlannerBudgets,
}

impl RecipePlanner {
    pub fn new(budgets: PlannerBudgets) -> Self {
        Self { budgets }
    }

    pub fn plan(&self, inputs: &PlanInputs<'_>) -> Result<ExecutionRecipe, RecipeError> {
        let budgets = &self.budgets;
        let mut stages = Vec::new();
        match inputs.search {
            SearchMode::Skip => {}
            SearchMode::Memory => stages.push(Stage {
                kind: StageKind::Retrieve {
                    source: RetrievalSource::Memory,
                },
                budget: budgets.retrieve,
            }),
            SearchMode::Search => stages.push(Stage {
                kind: StageKind::Retrieve {
                    source: RetrievalSource::Search,
                },
                budget: budgets.search,
            }),
        }
        if inputs.reasoning == ReasoningMode::SymbolicFirst {
            stages.push(symbolic(SymbolicStep::Plan, budgets));
        }
        if let Some(option) = inputs.model {
            stages.push(Stage {
                kind: StageKind::ModelCall {
                    requirement: BackendRequirement {
                        backend: option.id().clone(),
                        capabilities: option
                            .capabilities()
                            .union(inputs.profile.model.unwrap_or_default()),
                        context: ContextMode::Full,
                    },
                },
                budget: StageBudget::new(
                    option.cost_units(),
                    option.latency_ms().max(budgets.model_floor_ms.max(1)),
                ),
            });
        }
        if inputs.reasoning == ReasoningMode::Deliberate {
            stages.push(symbolic(SymbolicStep::Check, budgets));
        }
        if inputs.pod_use == PodUse::Call {
            let candidate = inputs
                .candidate
                .ok_or(RecipeError::MissingCheckedCandidate)?;
            stages.push(Stage {
                kind: StageKind::PodCall {
                    class: candidate.capability().clone(),
                    pod: candidate.pod().clone(),
                    effect: inputs.profile.pod_effect,
                },
                budget: budgets.pod,
            });
        }
        if stages.is_empty() {
            stages.push(symbolic(SymbolicStep::Check, budgets));
        }
        stages.push(Stage {
            kind: StageKind::Verify,
            budget: budgets.verify,
        });
        ExecutionRecipe::new(stages)
    }
}

fn symbolic(step: SymbolicStep, budgets: &PlannerBudgets) -> Stage {
    Stage {
        kind: StageKind::Symbolic { step },
        budget: budgets.symbolic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(kind: StageKind) -> Stage {
        Stage {
            kind,
            budget: StageBudget::new(1, 10),
        }
    }

    fn pod(effect: StageEffect) -> StageKind {
        StageKind::PodCall {
            class: CapabilityId::from("read"),
            pod: PodId::from("p"),
            effect,
        }
    }

    #[test]
    fn backend_id_rejects_empty_long_and_unsafe_bytes() {
        assert_eq!(BackendId::new(""), Err(BackendIdError::Empty));
        assert!(matches!(
            BackendId::new(&"a".repeat(65)),
            Err(BackendIdError::TooLong { len: 65 })
        ));
        assert_eq!(
            BackendId::new("a b"),
            Err(BackendIdError::InvalidByte { index: 1 })
        );
        assert!(BackendId::new("a\"b").is_err());
        assert!(BackendId::new("a\\b").is_err());
        assert!(BackendId::new(&"a".repeat(64)).is_ok());
    }

    #[test]
    fn capability_set_algebra() {
        let a = CapabilitySet::empty().with(BackendCapability::Code);
        let b = a.with(BackendCapability::ToolUse);
        assert!(b.contains(BackendCapability::Code));
        assert_eq!(
            b.missing_from(a),
            CapabilitySet::empty().with(BackendCapability::ToolUse)
        );
        assert_eq!(b.iter().count(), 2);
        assert!(CapabilitySet::empty().is_empty());
    }

    #[test]
    fn validation_orders_errors() {
        assert_eq!(ExecutionRecipe::new(vec![]), Err(RecipeError::Empty));
        let sym = || {
            stage(StageKind::Symbolic {
                step: SymbolicStep::Check,
            })
        };
        assert_eq!(
            ExecutionRecipe::new(vec![sym(); MAX_STAGES + 1]),
            Err(RecipeError::TooManyStages {
                count: MAX_STAGES + 1
            })
        );
        let mut zero = sym();
        zero.budget = StageBudget::new(0, 0);
        assert_eq!(
            ExecutionRecipe::new(vec![sym(), zero]),
            Err(RecipeError::ZeroLatencyBudget { stage: 1 })
        );
        assert_eq!(
            ExecutionRecipe::new(vec![stage(pod(StageEffect::Effectful))]),
            Err(RecipeError::EffectWithoutVerifier { stage: 0 })
        );
        assert_eq!(
            ExecutionRecipe::new(vec![sym()]),
            Err(RecipeError::TerminalStageNotVerifier)
        );
        assert_eq!(
            ExecutionRecipe::new(vec![stage(StageKind::Verify)]),
            Err(RecipeError::NoWorkStage)
        );
    }

    #[test]
    fn verifier_before_effect_does_not_count() {
        let recipe = ExecutionRecipe::new(vec![
            stage(StageKind::Verify),
            stage(pod(StageEffect::Effectful)),
        ]);
        assert_eq!(recipe, Err(RecipeError::EffectWithoutVerifier { stage: 1 }));
        let ok = ExecutionRecipe::new(vec![
            stage(StageKind::Verify),
            stage(pod(StageEffect::Effectful)),
            stage(StageKind::Verify),
        ])
        .unwrap();
        assert_eq!(ok.stages().len(), 3);
        assert_eq!(ok.total_cost_units(), 3);
        assert_eq!(ok.total_latency_ms(), 30);
    }
}
