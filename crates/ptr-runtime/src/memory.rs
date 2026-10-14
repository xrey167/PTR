//! Runtime-owned orchestration for the semantic memory slice.
//!
//! This module deliberately delegates durable semantic promotion to the
//! existing runtime/SemDB path. It owns only ingestion, retention selection
//! and deterministic context preparation.

use ptr_memory::RawEvent;
use ptr_memory::{
    compile_selected, compile_selected_at, compile_structured, compile_structured_at,
    BasicKnowledgeObjectBuilder, CompiledContext, ContextCandidate, DeterministicEntityExtractor,
    DeterministicTokenizer, EntityExtractor, IngestionError, KnowledgeObject,
    KnowledgeObjectBuilder, KnowledgeStore, NamespaceContext, NamespaceResolver,
    RetentionController, RetentionDecision, RetentionModel, RetentionState, SemanticTokenizer,
    StaticNamespaceResolver, TokenBudget,
};
use ptr_pods::PodManifest;
use ptr_protocol::TypedPayload;
use ptr_types::{EventRole, Generation, RequestId, Revision, SessionId, Timestamp};
use ptr_types::{KnowledgeObjectId, NamespaceId, PolicyVersion, Probability};
use ptr_verifier::Verifier;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryControllerError {
    Ingestion(IngestionError),
    Storage(String),
    MissingObject(KnowledgeObjectId),
    Context(String),
    Promotion(String),
}

pub struct ContextMemoryController<M> {
    pub store: KnowledgeStore,
    pub retention: RetentionController<M>,
}

impl<M: RetentionModel> ContextMemoryController<M> {
    pub fn new(model: M, threshold: Probability, policy_version: PolicyVersion) -> Self {
        Self {
            store: KnowledgeStore::default(),
            retention: RetentionController {
                model,
                threshold,
                policy_version,
            },
        }
    }

    pub fn ingest(
        &mut self,
        event: RawEvent,
        namespace: NamespaceId,
    ) -> Result<KnowledgeObject, MemoryControllerError> {
        self.store
            .append_raw(event.clone())
            .map_err(|error| MemoryControllerError::Storage(format!("{error:?}")))?;
        let tokenized = DeterministicTokenizer
            .tokenize(&event)
            .map_err(MemoryControllerError::Ingestion)?;
        let entities = DeterministicEntityExtractor
            .extract(&tokenized)
            .map_err(MemoryControllerError::Ingestion)?;
        let namespaces = StaticNamespaceResolver
            .resolve(
                &entities,
                &NamespaceContext {
                    namespaces: vec![namespace],
                },
            )
            .map_err(MemoryControllerError::Ingestion)?;
        let object = BasicKnowledgeObjectBuilder
            .build(&tokenized, &entities, &namespaces)
            .map_err(MemoryControllerError::Ingestion)?;
        self.store
            .admit_object(object.clone())
            .map_err(|error| MemoryControllerError::Storage(format!("{error:?}")))?;
        Ok(object)
    }

    pub fn retain(&self, state: &RetentionState) -> Vec<RetentionDecision> {
        self.retention.decide_all(state)
    }

    fn objects_of(
        &self,
        ids: &[KnowledgeObjectId],
    ) -> Result<Vec<KnowledgeObject>, MemoryControllerError> {
        ids.iter()
            .map(|id| {
                self.store
                    .object(id)
                    .ok_or_else(|| MemoryControllerError::MissingObject(id.clone()))
            })
            .collect()
    }

    /// Compile the objects `ids` name for the instant `now` (milliseconds since the Unix
    /// epoch): an object whose validity does not hold at `now` is left out and counted in
    /// [`CompiledContext::omitted_expired`]. This is the path a context that feeds an answer
    /// takes.
    pub fn compile_at(
        &self,
        ids: &[KnowledgeObjectId],
        budget: TokenBudget,
        now: ptr_types::Timestamp,
    ) -> Result<CompiledContext, MemoryControllerError> {
        compile_structured_at(&self.objects_of(ids)?, budget, now)
            .map_err(|error| MemoryControllerError::Context(format!("{error:?}")))
    }

    /// [`Self::compile_at`] for candidates that already carry a retention action.
    pub fn compile_candidates_at(
        &self,
        candidates: &[ContextCandidate],
        budget: TokenBudget,
        now: ptr_types::Timestamp,
    ) -> Result<CompiledContext, MemoryControllerError> {
        compile_selected_at(candidates, budget, now)
            .map_err(|error| MemoryControllerError::Context(format!("{error:?}")))
    }

    /// Compile with no time rule: the audit and replay path, where every object is eligible
    /// whatever its validity. Use [`Self::compile_at`] for a context that feeds an answer.
    pub fn compile(
        &self,
        ids: &[KnowledgeObjectId],
        budget: TokenBudget,
    ) -> Result<CompiledContext, MemoryControllerError> {
        compile_structured(&self.objects_of(ids)?, budget)
            .map_err(|error| MemoryControllerError::Context(format!("{error:?}")))
    }

    /// Compile candidates with no time rule: the audit and replay path. See
    /// [`Self::compile_candidates_at`].
    pub fn compile_candidates(
        &self,
        candidates: &[ContextCandidate],
        budget: TokenBudget,
    ) -> Result<CompiledContext, MemoryControllerError> {
        compile_selected(candidates, budget)
            .map_err(|error| MemoryControllerError::Context(format!("{error:?}")))
    }

    /// Promotes through the existing PTR runtime admission boundary first, then
    /// records the accepted output as a new immutable Knowledge generation.
    #[allow(clippy::too_many_arguments)]
    pub fn promote_verified_output<V>(
        &mut self,
        runtime: &mut super::PtrRuntime,
        request: &RequestId,
        manifest: &PodManifest,
        output: &TypedPayload,
        verifier: &V,
        base: &KnowledgeObject,
        session: SessionId,
        timestamp: Timestamp,
        namespace: NamespaceId,
    ) -> Result<(Revision, KnowledgeObject), MemoryControllerError>
    where
        V: Verifier<TypedPayload> + ?Sized,
    {
        let revision = runtime
            .promote_verified_pod_output(request, manifest, output, verifier)
            .map_err(|error| MemoryControllerError::Promotion(format!("{error:?}")))?;
        let event = RawEvent::new(
            ptr_types::RawEventId::from(format!("pod-output-{}", request.0).as_str()),
            session,
            EventRole::Tool,
            output.bytes.clone(),
            timestamp,
        );
        self.store
            .append_raw(event.clone())
            .map_err(|error| MemoryControllerError::Storage(format!("{error:?}")))?;
        let mut next = base.clone();
        next.namespace = namespace;
        next.generation = Generation(
            base.generation
                .0
                .checked_add(1)
                .ok_or_else(|| MemoryControllerError::Storage("generation exhausted".into()))?,
        );
        next.supersedes = Some(ptr_memory::KnowledgeKey {
            logical_id: base.id.clone(),
            generation: base.generation,
        });
        next.sources = vec![event.id];
        next.lifecycle = ptr_memory::KnowledgeLifecycle::Hot;
        self.store
            .admit_object(next.clone())
            .map_err(|error| MemoryControllerError::Storage(format!("{error:?}")))?;
        self.store
            .activate_generation(&base.id, next.generation)
            .map_err(|error| MemoryControllerError::Storage(format!("{error:?}")))?;
        Ok((revision, next))
    }
}
