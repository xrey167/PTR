use crate::{KnowledgeKey, KnowledgeLifecycle, KnowledgeObject, RetentionAction};
use ptr_types::Timestamp;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenBudget(pub u32);

#[derive(Clone, Debug, PartialEq)]
pub enum ContextEntry {
    Verbatim(Vec<u8>),
    Structured(Box<KnowledgeObject>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextCandidate {
    pub object: KnowledgeObject,
    pub action: RetentionAction,
    pub verbatim: Option<Vec<u8>>,
    pub pinned: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledContext {
    pub entries: Vec<ContextEntry>,
    pub token_cost: u32,
    pub digest: [u8; 32],
    /// How many candidates the time rule left out because their validity did not hold at the
    /// instant the context was compiled for ([`compile_selected_at`]). Always zero for the
    /// audit path without a time rule ([`compile_selected`]).
    pub omitted_expired: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContextCompileError {
    BudgetExceeded,
    InvalidatedObject,
    MissingDependency,
    MissingVerbatim,
}

pub trait ContextCompiler: Send + Sync {
    fn compile(
        &self,
        objects: &[KnowledgeObject],
        budget: TokenBudget,
    ) -> Result<CompiledContext, ContextCompileError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StructuredContextCompiler;

impl ContextCompiler for StructuredContextCompiler {
    fn compile(
        &self,
        objects: &[KnowledgeObject],
        budget: TokenBudget,
    ) -> Result<CompiledContext, ContextCompileError> {
        compile_structured(objects, budget)
    }
}

/// [`compile_structured`] for the instant `now` (milliseconds since the Unix epoch): objects
/// whose validity does not hold at `now` are left out and counted in
/// [`CompiledContext::omitted_expired`]. An object that depends on an omitted one is refused
/// ([`ContextCompileError::MissingDependency`]) rather than compiled without its basis.
pub fn compile_structured_at(
    objects: &[KnowledgeObject],
    budget: TokenBudget,
    now: Timestamp,
) -> Result<CompiledContext, ContextCompileError> {
    compile_candidates(&structured_candidates(objects), budget, Some(now))
}

fn structured_candidates(objects: &[KnowledgeObject]) -> Vec<ContextCandidate> {
    objects
        .iter()
        .cloned()
        .map(|object| ContextCandidate {
            object,
            action: RetentionAction::KeepStructured,
            verbatim: None,
            pinned: false,
        })
        .collect()
}

/// Compile with **no time rule**: every object is eligible whatever its validity. This is the
/// audit and replay path; a context that feeds an answer is compiled with
/// [`compile_structured_at`].
pub fn compile_structured(
    objects: &[KnowledgeObject],
    budget: TokenBudget,
) -> Result<CompiledContext, ContextCompileError> {
    compile_selected(
        &objects
            .iter()
            .cloned()
            .map(|object| ContextCandidate {
                object,
                action: RetentionAction::KeepStructured,
                verbatim: None,
                pinned: false,
            })
            .collect::<Vec<_>>(),
        budget,
    )
}

/// [`compile_selected`] for the instant `now` (milliseconds since the Unix epoch): candidates
/// whose object validity does not hold at `now` (half-open, `valid_from <= now <
/// valid_until`) are left out and counted in [`CompiledContext::omitted_expired`].
pub fn compile_selected_at(
    candidates: &[ContextCandidate],
    budget: TokenBudget,
    now: Timestamp,
) -> Result<CompiledContext, ContextCompileError> {
    compile_candidates(candidates, budget, Some(now))
}

/// Compile with **no time rule**: the audit and replay path. See [`compile_selected_at`].
pub fn compile_selected(
    candidates: &[ContextCandidate],
    budget: TokenBudget,
) -> Result<CompiledContext, ContextCompileError> {
    compile_candidates(candidates, budget, None)
}

fn compile_candidates(
    candidates: &[ContextCandidate],
    budget: TokenBudget,
    now: Option<Timestamp>,
) -> Result<CompiledContext, ContextCompileError> {
    let mut token_cost: u32 = 0;
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    let mut omitted_expired: u32 = 0;
    let mut ordered = Vec::new();
    for candidate in candidates {
        if now.is_some_and(|now| !candidate.object.validity.is_valid_at(now)) {
            omitted_expired = omitted_expired.saturating_add(1);
        } else {
            ordered.push(candidate);
        }
    }
    ordered.sort_by_key(|candidate| {
        (
            !candidate.pinned,
            action_rank(candidate.action),
            candidate.object.id.clone(),
            candidate.object.generation,
            candidate.object.digest(),
        )
    });
    let available: std::collections::BTreeSet<KnowledgeKey> = ordered
        .iter()
        .filter(|candidate| {
            matches!(
                candidate.action,
                RetentionAction::KeepVerbatim
                    | RetentionAction::KeepStructured
                    | RetentionAction::Recall
                    | RetentionAction::Recompute
                    | RetentionAction::DemoteToPod
            ) && candidate.object.lifecycle != KnowledgeLifecycle::Invalidated
        })
        .map(|candidate| KnowledgeKey {
            logical_id: candidate.object.id.clone(),
            generation: candidate.object.generation,
        })
        .collect();
    for candidate in ordered {
        let object = &candidate.object;
        if matches!(
            candidate.action,
            RetentionAction::DropCall
                | RetentionAction::DropResult
                | RetentionAction::Archive
                | RetentionAction::Invalidate
        ) {
            continue;
        }
        if object.lifecycle == KnowledgeLifecycle::Invalidated {
            return Err(ContextCompileError::InvalidatedObject);
        }
        if object
            .dependencies
            .iter()
            .any(|dependency| !available.contains(dependency))
        {
            return Err(ContextCompileError::MissingDependency);
        }
        token_cost = token_cost.saturating_add(object.token_cost);
        if token_cost > budget.0 {
            return Err(ContextCompileError::BudgetExceeded);
        }
        bytes.extend_from_slice(&[candidate.action as u8, u8::from(candidate.pinned)]);
        match candidate.action {
            RetentionAction::KeepVerbatim => {
                let verbatim = candidate
                    .verbatim
                    .as_ref()
                    .ok_or(ContextCompileError::MissingVerbatim)?;
                bytes.extend_from_slice(&(verbatim.len() as u64).to_le_bytes());
                bytes.extend_from_slice(verbatim);
                entries.push(ContextEntry::Verbatim(verbatim.clone()));
            }
            RetentionAction::KeepStructured
            | RetentionAction::DemoteToPod
            | RetentionAction::Recall
            | RetentionAction::Recompute => {
                bytes.extend_from_slice(&object.digest());
                entries.push(ContextEntry::Structured(Box::new(object.clone())));
            }
            RetentionAction::DropCall
            | RetentionAction::DropResult
            | RetentionAction::Archive
            | RetentionAction::Invalidate => unreachable!(),
        }
        bytes.extend_from_slice(&object.token_cost.to_le_bytes());
    }
    Ok(CompiledContext {
        entries,
        token_cost,
        digest: Sha256::digest(bytes).into(),
        omitted_expired,
    })
}

fn action_rank(action: RetentionAction) -> u8 {
    match action {
        RetentionAction::KeepVerbatim => 0,
        RetentionAction::KeepStructured => 1,
        RetentionAction::Recall => 2,
        RetentionAction::Recompute => 3,
        RetentionAction::DemoteToPod => 4,
        RetentionAction::DropResult => 5,
        RetentionAction::DropCall => 6,
        RetentionAction::Archive => 7,
        RetentionAction::Invalidate => 8,
    }
}
