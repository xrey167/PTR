use ptr_runtime::ProtectedStateCoordinator;
use ptr_storage::{
    GenerationAnchor, InMemoryProtectedStateStore, ProtectedRecord, SoftwareKeyProvider,
    StateDomain,
};
use ptr_types::{Digest, Generation, KeyId, Revision};

fn digest(bytes: &[u8]) -> Digest {
    use sha2::{Digest as ShaDigest, Sha256};
    Sha256::digest(bytes).into()
}

fn record(payload: &[u8]) -> ProtectedRecord {
    let logical_id = "runtime-state".to_owned();
    let generation = Generation(1);
    let mut anchor_bytes = logical_id.as_bytes().to_vec();
    anchor_bytes.extend_from_slice(&generation.0.to_le_bytes());
    ProtectedRecord {
        domain: StateDomain::KvSnapshot,
        logical_id: logical_id.clone(),
        generation,
        revision: Revision(1),
        plaintext_digest: digest(payload),
        payload: payload.to_vec(),
        key_id: KeyId::from("runtime-key"),
        anchor: GenerationAnchor {
            logical_id,
            generation,
            previous_digest: None,
            anchor_digest: digest(&anchor_bytes),
        },
    }
}

#[test]
fn runtime_coordinator_verifies_before_recovery_uses_state() {
    let keys = SoftwareKeyProvider::default().with_key(KeyId::from("runtime-key"), [3; 32]);
    let store = InMemoryProtectedStateStore::new(keys);
    let mut coordinator = ProtectedStateCoordinator::new(store);
    let handle = coordinator.seal(record(b"kv snapshot")).unwrap();
    assert_eq!(coordinator.open(&handle).unwrap(), b"kv snapshot");
    assert!(coordinator.verify(&handle).unwrap().valid);
}

#[test]
fn coordinator_exposes_rotation_as_a_runtime_operation() {
    let keys = SoftwareKeyProvider::default().with_key(KeyId::from("runtime-key"), [3; 32]);
    let store = InMemoryProtectedStateStore::new(keys);
    let mut coordinator = ProtectedStateCoordinator::new(store);
    coordinator
        .rotate_key(KeyId::from("next-key"), [4; 32])
        .unwrap();
    let store = coordinator.into_inner();
    let _ = store;
}
