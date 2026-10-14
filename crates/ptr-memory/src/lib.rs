mod codec;
mod consolidation;
mod context;
mod durable_knowledge;
mod ingestion;
mod journal;
mod knowledge;
mod kv;
mod memory_store;
mod mirror;
mod record;
mod retention;
mod semantic;

pub use codec::CodecError;
pub use consolidation::{
    ConsolidationPolicy, Contradiction, DeferReason, PromotionContext, PromotionDecision,
    RejectReason, RetentionVerdict,
};
pub use context::{
    compile_selected, compile_selected_at, compile_structured, compile_structured_at,
    CompiledContext, ContextCandidate, ContextCompileError, ContextCompiler, ContextEntry,
    StructuredContextCompiler, TokenBudget,
};
pub use durable_knowledge::{DurableKnowledgeError, DurableKnowledgeStore, DurableOptions};
pub use ingestion::{
    BasicKnowledgeObjectBuilder, DeterministicEntityExtractor, DeterministicTokenizer, Entity,
    EntityExtractor, IngestionError, KnowledgeObjectBuilder, NamespaceContext, NamespaceResolver,
    SemanticTokenizer, StaticNamespaceResolver, TokenizedEvent,
};
pub use journal::{
    crc32, CorruptionReason, JournalError, JournalKind, JournalOptions, JournalRecovery,
    MemoryJournal,
};
pub use knowledge::{KnowledgeStore, KnowledgeStoreError, LifecycleRecord};
pub use kv::{
    InMemoryKvStateRuntime, KvError, KvStateHandle, KvStateMetadata, KvStateRuntime, KvValidity,
};
pub use memory_store::{
    AppliedRetention, DurableMemoryStore, MemoryClassName, MemoryLimits, MemoryStoreError,
    MemoryStoreOptions, NewMemoryRecord, RecordViolation, RevocationReport, TraversalLimit,
};
pub use mirror::{render_knowledge_mirror, render_memory_mirror, REDACTED};
pub use record::{MemoryRecord, MemoryRecordId, RecordState, Support, SupportKind};
pub use retention::{
    decide, score_candidates, ConflictState, Freshness, KnowledgeKey, KnowledgeLifecycle,
    KnowledgeObject, RawEvent, RawEventError, Relation, RetentionAction, RetentionCandidate,
    RetentionController, RetentionDecision, RetentionError, RetentionModel, RetentionReason,
    RetentionScore, RetentionState, RuleRetentionModel, ToolCall, ToolResult,
};
pub use semantic::{
    semantic_buckets_from_objects, semantic_objects_from_buckets, SemanticBuckets,
    SemanticConversionError, SemanticObject,
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
