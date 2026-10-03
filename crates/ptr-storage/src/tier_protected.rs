use crate::{
    ChunkDescriptor, GenerationAnchor, ProtectedHandle, ProtectedRecord, ProtectedStateError,
    ProtectedStateStore, StateDomain, TierObjectManifest,
};
use ptr_types::{Digest, KeyId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtectedTierChunk {
    pub descriptor: ChunkDescriptor,
    pub handle: ProtectedHandle,
}

pub fn seal_protected_tier_chunk<S: ProtectedStateStore>(
    store: &mut S,
    manifest: &TierObjectManifest,
    chunk_index: usize,
    bytes: &[u8],
    key_id: KeyId,
    previous_anchor: Option<Digest>,
) -> Result<ProtectedTierChunk, ProtectedStateError> {
    if manifest.domain != crate::TierObjectDomain::KvSnapshot {
        return Err(ProtectedStateError::AnchorMismatch);
    }
    let descriptor = manifest
        .chunks
        .get(chunk_index)
        .cloned()
        .ok_or(ProtectedStateError::PlaintextDigestMismatch)?;
    descriptor
        .validate_bytes(bytes)
        .map_err(|_| ProtectedStateError::PlaintextDigestMismatch)?;
    let logical_id = logical_id(manifest, &descriptor);
    let handle = store.seal(ProtectedRecord {
        domain: StateDomain::KvSnapshot,
        logical_id: logical_id.clone(),
        generation: manifest.generation,
        revision: manifest.revision,
        plaintext_digest: descriptor.digest,
        payload: bytes.to_vec(),
        key_id,
        anchor: GenerationAnchor::new(logical_id, manifest.generation, previous_anchor),
    })?;
    Ok(ProtectedTierChunk { descriptor, handle })
}

pub fn open_protected_tier_chunk<S: ProtectedStateStore>(
    store: &S,
    manifest: &TierObjectManifest,
    chunk: &ProtectedTierChunk,
) -> Result<Vec<u8>, ProtectedStateError> {
    if manifest.domain != crate::TierObjectDomain::KvSnapshot {
        return Err(ProtectedStateError::AnchorMismatch);
    }
    let expected = manifest
        .chunks
        .get(chunk.descriptor.index as usize)
        .ok_or(ProtectedStateError::PlaintextDigestMismatch)?;
    if expected != &chunk.descriptor
        || chunk.handle.logical_id != logical_id(manifest, expected)
        || chunk.handle.generation != manifest.generation
        || chunk.handle.revision != manifest.revision
    {
        return Err(ProtectedStateError::AnchorMismatch);
    }
    let bytes = store.open(&chunk.handle)?;
    expected
        .validate_bytes(&bytes)
        .map_err(|_| ProtectedStateError::PlaintextDigestMismatch)?;
    Ok(bytes)
}

fn logical_id(manifest: &TierObjectManifest, chunk: &ChunkDescriptor) -> String {
    format!(
        "tier:{}:{}:chunk:{}",
        manifest.domain.code(),
        manifest.logical_id,
        chunk.index
    )
}
