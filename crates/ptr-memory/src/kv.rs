use crate::context::CompiledContext;
use ptr_types::{AdapterVersion, Generation, ModelVersion, Revision, SessionId, StateId};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

pub type Digest = [u8; 32];

static NEXT_RUNTIME_OWNER: AtomicU64 = AtomicU64::new(1);

/// Opaque authority for one runtime-owned KV state. The owner identity prevents
/// a handle from one runtime from being replayed against another runtime that
/// happens to allocate the same StateId.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvStateHandle {
    state_id: StateId,
    owner: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvValidity {
    Valid,
    Invalidated,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvStateMetadata {
    pub state_id: StateId,
    pub parent_state_id: Option<StateId>,
    pub session: SessionId,
    pub model_version: ModelVersion,
    pub adapter_version: AdapterVersion,
    pub snapshot_revision: Revision,
    pub pod_generations: Vec<Generation>,
    pub dependency_digests: Vec<Digest>,
    pub context_digest: Digest,
    pub validity: KvValidity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvError {
    InvalidState,
    InvalidatedDependency,
    UnknownHandle,
    ForeignHandle,
    Backend(String),
}

pub trait KvStateRuntime: Send + Sync {
    fn continue_state(
        &mut self,
        handle: &KvStateHandle,
        input: CompiledContext,
    ) -> Result<KvStateHandle, KvError>;

    fn invalidate_dependencies(
        &mut self,
        dependencies: &[Digest],
    ) -> Result<Vec<KvStateHandle>, KvError>;

    fn recompute(&mut self, valid_context: CompiledContext) -> Result<KvStateHandle, KvError>;

    fn metadata(&self, handle: &KvStateHandle) -> Result<KvStateMetadata, KvError>;
}

pub struct InMemoryKvStateRuntime {
    owner: u64,
    session: SessionId,
    model_version: ModelVersion,
    adapter_version: AdapterVersion,
    next_state: u64,
    bindings: BTreeMap<StateId, KvStateMetadata>,
}

impl InMemoryKvStateRuntime {
    pub fn new(
        session: SessionId,
        model_version: ModelVersion,
        adapter_version: AdapterVersion,
    ) -> Self {
        Self {
            owner: NEXT_RUNTIME_OWNER.fetch_add(1, Ordering::Relaxed),
            session,
            model_version,
            adapter_version,
            next_state: 1,
            bindings: BTreeMap::new(),
        }
    }

    fn allocate_state(&mut self) -> StateId {
        let id = StateId::from(format!("kv-{}", self.next_state).as_str());
        self.next_state = self.next_state.saturating_add(1);
        id
    }

    fn handle(&self, state_id: StateId) -> KvStateHandle {
        KvStateHandle {
            state_id,
            owner: self.owner,
        }
    }

    fn resolve(&self, handle: &KvStateHandle) -> Result<&KvStateMetadata, KvError> {
        if handle.owner != self.owner {
            return Err(KvError::ForeignHandle);
        }
        self.bindings
            .get(&handle.state_id)
            .ok_or(KvError::UnknownHandle)
    }
}

impl KvStateRuntime for InMemoryKvStateRuntime {
    fn continue_state(
        &mut self,
        handle: &KvStateHandle,
        input: CompiledContext,
    ) -> Result<KvStateHandle, KvError> {
        let parent = self.resolve(handle)?.clone();
        if parent.validity != KvValidity::Valid {
            return Err(KvError::InvalidState);
        }
        let state_id = self.allocate_state();
        let mut dependencies = parent.dependency_digests;
        dependencies.push(input.digest);
        let next = KvStateMetadata {
            state_id: state_id.clone(),
            parent_state_id: Some(parent.state_id),
            session: self.session.clone(),
            model_version: self.model_version.clone(),
            adapter_version: self.adapter_version.clone(),
            snapshot_revision: parent.snapshot_revision.next(),
            pod_generations: parent.pod_generations,
            dependency_digests: dependencies,
            context_digest: input.digest,
            validity: KvValidity::Valid,
        };
        self.bindings.insert(state_id.clone(), next);
        Ok(self.handle(state_id))
    }

    fn invalidate_dependencies(
        &mut self,
        dependencies: &[Digest],
    ) -> Result<Vec<KvStateHandle>, KvError> {
        let dependencies: BTreeSet<_> = dependencies.iter().copied().collect();
        let mut affected: BTreeSet<StateId> = self
            .bindings
            .values()
            .filter(|state| {
                state.validity == KvValidity::Valid
                    && state
                        .dependency_digests
                        .iter()
                        .any(|digest| dependencies.contains(digest))
            })
            .map(|state| state.state_id.clone())
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for state in self.bindings.values() {
                if let Some(parent) = &state.parent_state_id {
                    if affected.contains(parent) && affected.insert(state.state_id.clone()) {
                        changed = true;
                    }
                }
            }
        }
        for state_id in &affected {
            if let Some(state) = self.bindings.get_mut(state_id) {
                state.validity = KvValidity::Invalidated;
            }
        }
        Ok(affected.into_iter().map(|id| self.handle(id)).collect())
    }

    fn recompute(&mut self, valid_context: CompiledContext) -> Result<KvStateHandle, KvError> {
        let state_id = self.allocate_state();
        self.bindings.insert(
            state_id.clone(),
            KvStateMetadata {
                state_id: state_id.clone(),
                parent_state_id: None,
                session: self.session.clone(),
                model_version: self.model_version.clone(),
                adapter_version: self.adapter_version.clone(),
                snapshot_revision: Revision(0),
                pod_generations: Vec::new(),
                dependency_digests: vec![valid_context.digest],
                context_digest: valid_context.digest,
                validity: KvValidity::Valid,
            },
        );
        Ok(self.handle(state_id))
    }

    fn metadata(&self, handle: &KvStateHandle) -> Result<KvStateMetadata, KvError> {
        Ok(self.resolve(handle)?.clone())
    }
}
