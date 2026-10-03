use crate::{ProtocolBinding, TypedPayload};
use ptr_types::{CapabilityId, Digest, PodIdentity, PrincipalId, Revision};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PodCacheKey {
    pub pod_identity: PodIdentity,
    pub semantic_revision: Digest,
    pub execution_manifest: Digest,
    pub capability: CapabilityId,
    pub input_digest: Digest,
    pub knowledge_revision: Revision,
    pub principal: PrincipalId,
    pub protocol: ProtocolBinding,
}

#[derive(Default)]
pub struct PodCache {
    values: BTreeMap<PodCacheKey, TypedPayload>,
}

impl PodCache {
    pub fn insert(&mut self, key: PodCacheKey, value: TypedPayload) {
        self.values.insert(key, value);
    }
    pub fn get(&self, key: &PodCacheKey) -> Option<&TypedPayload> {
        self.values.get(key)
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn invalidate<F>(&mut self, mut predicate: F) -> usize
    where
        F: FnMut(&PodCacheKey) -> bool,
    {
        let before = self.values.len();
        self.values.retain(|key, _| !predicate(key));
        before - self.values.len()
    }
}
