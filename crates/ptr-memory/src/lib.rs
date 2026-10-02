mod context;
mod ingestion;
mod knowledge;
mod kv;
mod retention;

pub use context::{
    compile_selected, compile_structured, CompiledContext, ContextCandidate, ContextCompileError,
    ContextCompiler, ContextEntry, StructuredContextCompiler, TokenBudget,
};
pub use ingestion::{
    BasicKnowledgeObjectBuilder, DeterministicEntityExtractor, DeterministicTokenizer, Entity,
    EntityExtractor, IngestionError, KnowledgeObjectBuilder, NamespaceContext, NamespaceResolver,
    SemanticTokenizer, StaticNamespaceResolver, TokenizedEvent,
};
pub use knowledge::{KnowledgeStore, KnowledgeStoreError, LifecycleRecord};
pub use kv::{
    InMemoryKvStateRuntime, KvError, KvStateHandle, KvStateMetadata, KvStateRuntime, KvValidity,
};
pub use retention::{
    decide, score_candidates, ConflictState, Freshness, KnowledgeKey, KnowledgeLifecycle,
    KnowledgeObject, RawEvent, Relation, RetentionAction, RetentionCandidate, RetentionController,
    RetentionDecision, RetentionError, RetentionModel, RetentionReason, RetentionScore,
    RetentionState, RuleRetentionModel, ToolCall, ToolResult,
};

use ptr_types::{CapsuleId, Generation, ProjectId, ProvenanceRef, Revision, Validity};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryClass {
    Semantic,
    Episodic,
    Procedural,
    Epistemic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticCapsule {
    pub id: CapsuleId,
    pub project: ProjectId,
    pub generation: Generation,
    pub revision_created: Revision,
    pub validity: Validity,
    pub goals: Vec<String>,
    pub hard_constraints: Vec<String>,
    pub known: Vec<String>,
    pub hypotheses: Vec<String>,
    pub unknowns: Vec<String>,
    pub relations: Vec<String>,
    pub provenance: Vec<ProvenanceRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedProcedure {
    pub id: String,
    pub generation: Generation,
    pub task_type: String,
    pub steps: Vec<String>,
    pub replay_verified: bool,
}
