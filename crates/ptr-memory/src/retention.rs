use ptr_types::{
    EntityId, EventRole, Generation, KnowledgeObjectId, NamespaceId, PolicyVersion, Probability,
    ProvenanceRef, RawEventId, RetrievalKey, SessionId, Timestamp,
};
use sha2::{Digest, Sha256};

/// The action applied to an active context object. None of these actions delete
/// the immutable raw event that produced the object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionAction {
    KeepVerbatim,
    KeepStructured,
    DropResult,
    DropCall,
    DemoteToPod,
    Archive,
    Recall,
    Recompute,
    Invalidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnowledgeLifecycle {
    Hot,
    Warm,
    Cold,
    Pod,
    Archived,
    Invalidated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConflictState {
    None,
    Suspected,
    Confirmed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Freshness {
    Current,
    Aging,
    Stale,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Relation {
    pub predicate: String,
    pub target: KnowledgeObjectId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolResult {
    pub output: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawEvent {
    pub id: RawEventId,
    pub session: SessionId,
    pub role: EventRole,
    pub content: Vec<u8>,
    pub timestamp: Timestamp,
    pub digest: [u8; 32],
    pub tool_call: Option<ToolCall>,
    pub tool_result: Option<ToolResult>,
}

impl RawEvent {
    pub fn new(
        id: RawEventId,
        session: SessionId,
        role: EventRole,
        content: Vec<u8>,
        timestamp: Timestamp,
    ) -> Self {
        let digest = Self::digest_for(&content);
        Self {
            id,
            session,
            role,
            content,
            timestamp,
            digest,
            tool_call: None,
            tool_result: None,
        }
    }

    pub fn digest_for(content: &[u8]) -> [u8; 32] {
        Sha256::digest(content).into()
    }

    pub fn validate_digest(&self) -> bool {
        self.digest == Self::digest_for(&self.content)
    }

    pub fn validate_tool_pair(&self) -> bool {
        self.tool_result.is_none() || self.tool_call.is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeObject {
    pub id: KnowledgeObjectId,
    pub namespace: NamespaceId,
    pub entities: Vec<EntityId>,
    pub relations: Vec<Relation>,
    pub sources: Vec<RawEventId>,
    pub provenance: Vec<ProvenanceRef>,
    pub generation: Generation,
    pub supersedes: Option<KnowledgeKey>,
    /// Exact generation dependencies. A logical id alone is insufficient for
    /// revocation because a newer generation may coexist with an invalidated one.
    pub dependencies: Vec<KnowledgeKey>,
    pub confidence: Probability,
    pub relevance: Probability,
    pub freshness: Freshness,
    pub conflict: ConflictState,
    pub lifecycle: KnowledgeLifecycle,
    pub retrieval_keys: Vec<RetrievalKey>,
    pub token_cost: u32,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KnowledgeKey {
    pub logical_id: KnowledgeObjectId,
    pub generation: Generation,
}

impl KnowledgeObject {
    /// Digest of the complete effective semantic object in a deterministic field order.
    pub fn digest(&self) -> [u8; 32] {
        self.digest_with_lifecycle(true)
    }

    /// Immutable content digest. Lifecycle metadata is intentionally excluded;
    /// lifecycle changes are represented by the store's append-only records.
    pub fn content_digest(&self) -> [u8; 32] {
        self.digest_with_lifecycle(false)
    }

    fn digest_with_lifecycle(&self, include_lifecycle: bool) -> [u8; 32] {
        let mut canonical = String::new();
        macro_rules! field {
            ($name:expr, $value:expr) => {{
                canonical.push_str($name);
                canonical.push('=');
                canonical.push_str(&$value.to_string());
                canonical.push('\n');
            }};
        }
        field!("id", self.id.0);
        field!("namespace", self.namespace.0);
        field!("generation", self.generation.0);
        field!(
            "supersedes",
            self.supersedes
                .as_ref()
                .map(|key| format!("{}:{}", key.logical_id.0, key.generation.0))
                .unwrap_or_default()
        );
        field!("confidence", self.confidence.get());
        field!("relevance", self.relevance.get());
        field!("freshness", format!("{:?}", self.freshness));
        field!("conflict", format!("{:?}", self.conflict));
        if include_lifecycle {
            field!("lifecycle", format!("{:?}", self.lifecycle));
        }
        field!("token_cost", self.token_cost);
        for entity in &self.entities {
            field!("entity", entity.0);
        }
        for relation in &self.relations {
            field!(
                "relation",
                format!("{}:{}", relation.predicate, relation.target)
            );
        }
        for source in &self.sources {
            field!("source", source.0);
        }
        for provenance in &self.provenance {
            field!(
                "provenance",
                format!("{:?}:{:?}", provenance.source, provenance.note)
            );
        }
        for dependency in &self.dependencies {
            field!(
                "dependency",
                format!("{}:{}", dependency.logical_id.0, dependency.generation.0)
            );
        }
        for key in &self.retrieval_keys {
            field!("retrieval_key", key.0);
        }
        Sha256::digest(canonical.as_bytes()).into()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionCandidate {
    pub object_id: KnowledgeObjectId,
    pub source_events: Vec<RawEventId>,
    pub token_cost: u32,
    pub pinned: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionReason {
    Pinned,
    Classifier,
    Dependency,
    Invalidated,
    Fallback,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetentionScore {
    pub object_id: KnowledgeObjectId,
    pub keep_call: Probability,
    pub keep_result: Probability,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetentionDecision {
    pub object_id: KnowledgeObjectId,
    pub action: RetentionAction,
    pub classifier_score: Probability,
    pub reason: RetentionReason,
    pub policy_version: PolicyVersion,
    pub dependencies: Vec<KnowledgeObjectId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionState {
    pub goal: String,
    pub candidates: Vec<RetentionCandidate>,
}

pub trait RetentionModel: Send + Sync {
    fn score(&self, state: &RetentionState) -> Result<Vec<RetentionScore>, String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RuleRetentionModel;

impl RetentionModel for RuleRetentionModel {
    fn score(&self, state: &RetentionState) -> Result<Vec<RetentionScore>, String> {
        Ok(state
            .candidates
            .iter()
            .map(|candidate| RetentionScore {
                object_id: candidate.object_id.clone(),
                keep_call: Probability::new(if candidate.pinned { 1.0 } else { 0.5 }).unwrap(),
                keep_result: Probability::new(if candidate.pinned { 1.0 } else { 0.5 }).unwrap(),
            })
            .collect())
    }
}

pub struct RetentionController<M> {
    pub model: M,
    pub threshold: Probability,
    pub policy_version: PolicyVersion,
}

impl<M: RetentionModel> RetentionController<M> {
    pub fn decide_all(&self, state: &RetentionState) -> Vec<RetentionDecision> {
        let scores = score_candidates(&self.model, state).unwrap_or_else(|_| {
            state
                .candidates
                .iter()
                .map(|candidate| RetentionScore {
                    object_id: candidate.object_id.clone(),
                    keep_call: Probability::new(1.0).unwrap(),
                    keep_result: Probability::new(1.0).unwrap(),
                })
                .collect()
        });
        state
            .candidates
            .iter()
            .zip(scores)
            .map(|(candidate, score)| {
                decide(
                    candidate,
                    score,
                    self.threshold,
                    self.policy_version.clone(),
                )
                .unwrap_or_else(|_| RetentionDecision {
                    object_id: candidate.object_id.clone(),
                    action: RetentionAction::KeepVerbatim,
                    classifier_score: Probability::new(1.0).unwrap(),
                    reason: RetentionReason::Fallback,
                    policy_version: self.policy_version.clone(),
                    dependencies: Vec::new(),
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionError {
    CandidateScoreMismatch {
        expected: KnowledgeObjectId,
        actual: KnowledgeObjectId,
    },
    DuplicateCandidate(KnowledgeObjectId),
    ScoreCountMismatch {
        candidates: usize,
        scores: usize,
    },
    UnknownScore(KnowledgeObjectId),
    ModelFailure,
    InvalidRawDigest(RawEventId),
}

pub fn score_candidates(
    model: &dyn RetentionModel,
    state: &RetentionState,
) -> Result<Vec<RetentionScore>, RetentionError> {
    validate_candidates(&state.candidates)?;
    let scores = model
        .score(state)
        .map_err(|_| RetentionError::ModelFailure)?;
    if scores.len() != state.candidates.len() {
        return Err(RetentionError::ScoreCountMismatch {
            candidates: state.candidates.len(),
            scores: scores.len(),
        });
    }
    let mut scored = std::collections::BTreeSet::new();
    for score in &scores {
        if !scored.insert(score.object_id.clone())
            || !state
                .candidates
                .iter()
                .any(|candidate| candidate.object_id == score.object_id)
        {
            return Err(RetentionError::UnknownScore(score.object_id.clone()));
        }
    }
    Ok(scores)
}

fn validate_candidates(candidates: &[RetentionCandidate]) -> Result<(), RetentionError> {
    let mut ids = std::collections::BTreeSet::new();
    for candidate in candidates {
        if !ids.insert(candidate.object_id.clone()) {
            return Err(RetentionError::DuplicateCandidate(
                candidate.object_id.clone(),
            ));
        }
    }
    Ok(())
}

/// Deterministic Jev-compatible policy: result relevance wins, otherwise the
/// call itself can be retained while its result is dropped.
pub fn decide(
    candidate: &RetentionCandidate,
    score: RetentionScore,
    threshold: Probability,
    policy_version: PolicyVersion,
) -> Result<RetentionDecision, RetentionError> {
    if score.object_id != candidate.object_id {
        return Err(RetentionError::CandidateScoreMismatch {
            expected: candidate.object_id.clone(),
            actual: score.object_id,
        });
    }
    if candidate.pinned {
        return Ok(RetentionDecision {
            object_id: candidate.object_id.clone(),
            action: RetentionAction::KeepVerbatim,
            classifier_score: Probability::new(1.0).expect("constant is bounded"),
            reason: RetentionReason::Pinned,
            policy_version,
            dependencies: Vec::new(),
        });
    }
    let action = if score.keep_result.get() >= threshold.get() {
        RetentionAction::KeepVerbatim
    } else if score.keep_call.get() >= threshold.get() {
        RetentionAction::DropResult
    } else {
        RetentionAction::DropCall
    };
    Ok(RetentionDecision {
        object_id: candidate.object_id.clone(),
        action,
        classifier_score: score.keep_result,
        reason: RetentionReason::Classifier,
        policy_version,
        dependencies: Vec::new(),
    })
}
