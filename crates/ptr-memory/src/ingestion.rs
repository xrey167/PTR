use crate::retention::{KnowledgeObject, RawEvent};
use crate::{ConflictState, Freshness, KnowledgeLifecycle, Relation};
use ptr_types::{
    EntityId, EvidenceId, Generation, KnowledgeObjectId, NamespaceId, Probability, ProvenanceRef,
    RetrievalKey,
};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizedEvent {
    pub event: RawEvent,
    pub tokens: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entity {
    pub id: EntityId,
    pub value: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NamespaceContext {
    pub namespaces: Vec<NamespaceId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IngestionError {
    InvalidEvent,
    MissingNamespace,
    NonDeterministicInput,
}

pub trait SemanticTokenizer: Send + Sync {
    fn tokenize(&self, event: &RawEvent) -> Result<TokenizedEvent, IngestionError>;
}

pub trait EntityExtractor: Send + Sync {
    fn extract(&self, event: &TokenizedEvent) -> Result<Vec<Entity>, IngestionError>;
}

pub trait NamespaceResolver: Send + Sync {
    fn resolve(
        &self,
        entities: &[Entity],
        context: &NamespaceContext,
    ) -> Result<Vec<NamespaceId>, IngestionError>;
}

pub trait KnowledgeObjectBuilder: Send + Sync {
    fn build(
        &self,
        event: &TokenizedEvent,
        entities: &[Entity],
        namespaces: &[NamespaceId],
    ) -> Result<KnowledgeObject, IngestionError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicTokenizer;

impl SemanticTokenizer for DeterministicTokenizer {
    fn tokenize(&self, event: &RawEvent) -> Result<TokenizedEvent, IngestionError> {
        event
            .validate_digest()
            .and_then(|()| event.validate_tool_pair())
            .map_err(|_| IngestionError::InvalidEvent)?;
        let text = std::str::from_utf8(&event.content).map_err(|_| IngestionError::InvalidEvent)?;
        Ok(TokenizedEvent {
            event: event.clone(),
            tokens: text.split_whitespace().map(str::to_owned).collect(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicEntityExtractor;

impl EntityExtractor for DeterministicEntityExtractor {
    fn extract(&self, event: &TokenizedEvent) -> Result<Vec<Entity>, IngestionError> {
        let mut entities = Vec::new();
        for value in &event.tokens {
            let digest = Sha256::digest(value.as_bytes());
            let id: String = digest
                .iter()
                .take(8)
                .map(|byte| format!("{byte:02x}"))
                .collect();
            entities.push(Entity {
                id: EntityId::from(id.as_str()),
                value: value.clone(),
            });
        }
        Ok(entities)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StaticNamespaceResolver;

impl NamespaceResolver for StaticNamespaceResolver {
    fn resolve(
        &self,
        _entities: &[Entity],
        context: &NamespaceContext,
    ) -> Result<Vec<NamespaceId>, IngestionError> {
        context
            .namespaces
            .first()
            .cloned()
            .map(|namespace| vec![namespace])
            .ok_or(IngestionError::MissingNamespace)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BasicKnowledgeObjectBuilder;

impl KnowledgeObjectBuilder for BasicKnowledgeObjectBuilder {
    fn build(
        &self,
        event: &TokenizedEvent,
        entities: &[Entity],
        namespaces: &[NamespaceId],
    ) -> Result<KnowledgeObject, IngestionError> {
        let namespace = namespaces
            .first()
            .cloned()
            .ok_or(IngestionError::MissingNamespace)?;
        let confidence = Probability::new(1.0).ok_or(IngestionError::NonDeterministicInput)?;
        Ok(KnowledgeObject {
            id: KnowledgeObjectId::from(format!("raw-{}", event.event.id.0).as_str()),
            namespace,
            entities: entities.iter().map(|entity| entity.id.clone()).collect(),
            relations: Vec::<Relation>::new(),
            sources: vec![event.event.id.clone()],
            provenance: vec![ProvenanceRef {
                source: EvidenceId::from(event.event.id.0.as_str()),
                note: None,
            }],
            generation: Generation(1),
            supersedes: None,
            dependencies: Vec::new(),
            confidence,
            relevance: confidence,
            freshness: Freshness::Current,
            conflict: ConflictState::None,
            lifecycle: KnowledgeLifecycle::Hot,
            retrieval_keys: event
                .tokens
                .iter()
                .map(|token| RetrievalKey::from(token.as_str()))
                .collect(),
            token_cost: event.tokens.len() as u32,
        })
    }
}
