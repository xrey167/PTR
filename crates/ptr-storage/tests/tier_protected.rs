use ptr_storage::{
    open_protected_tier_chunk, seal_protected_tier_chunk, InMemoryProtectedStateStore,
    PreparedTierObject, SoftwareKeyProvider, TierObjectDomain,
};
use ptr_types::{Generation, KeyId, Revision};

#[test]
fn durable_tier_chunk_is_aead_bound_to_manifest_generation_and_revision() {
    let key_id = KeyId("tier-key".into());
    let keys = SoftwareKeyProvider::default().with_key(key_id.clone(), [7; 32]);
    let mut store = InMemoryProtectedStateStore::new(keys);
    let object = PreparedTierObject::from_bytes(
        TierObjectDomain::KvSnapshot,
        "state-a",
        Generation(3),
        Revision(5),
        [9; 32],
        b"protected snapshot bytes",
        8,
    )
    .unwrap();

    let chunk = seal_protected_tier_chunk(
        &mut store,
        &object.manifest,
        0,
        &object.chunks[0],
        key_id.clone(),
        None,
    )
    .unwrap();
    assert_eq!(
        open_protected_tier_chunk(&store, &object.manifest, &chunk).unwrap(),
        object.chunks[0]
    );

    let mut stale_manifest = object.manifest.clone();
    stale_manifest.revision = Revision(4);
    assert!(open_protected_tier_chunk(&store, &stale_manifest, &chunk).is_err());

    let next = PreparedTierObject::from_bytes(
        TierObjectDomain::KvSnapshot,
        "state-a",
        Generation(4),
        Revision(6),
        [9; 32],
        b"protected snapshot bytes v2",
        8,
    )
    .unwrap();
    let next_chunk = seal_protected_tier_chunk(
        &mut store,
        &next.manifest,
        0,
        &next.chunks[0],
        key_id,
        Some(chunk.handle.anchor_digest),
    )
    .unwrap();
    assert_eq!(next_chunk.handle.generation, Generation(4));

    let rollback = PreparedTierObject::from_bytes(
        TierObjectDomain::KvSnapshot,
        "state-a",
        Generation(3),
        Revision(7),
        [9; 32],
        b"rollback",
        8,
    )
    .unwrap();
    assert!(seal_protected_tier_chunk(
        &mut store,
        &rollback.manifest,
        0,
        &rollback.chunks[0],
        KeyId("tier-key".into()),
        None,
    )
    .is_err());
}
