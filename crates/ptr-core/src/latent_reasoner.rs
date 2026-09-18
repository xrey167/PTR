use crate::epistemic_workspace::EpistemicWorkspace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReasoningBudget {
    None,
    Low,
    Medium,
    High,
    Max(u32),
}

impl ReasoningBudget {
    pub fn steps(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Low => 1,
            Self::Medium => 4,
            Self::High => 8,
            Self::Max(v) => v,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LatentReasoner;

impl LatentReasoner {
    pub fn advance(&self, workspace: &mut EpistemicWorkspace, budget: ReasoningBudget) {
        workspace.step = workspace.step.saturating_add(budget.steps());
    }
}
