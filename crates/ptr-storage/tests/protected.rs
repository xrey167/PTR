use ptr_storage::{
    FileProtectedStateStore, GenerationAnchor, InMemoryProtectedStateStore, ProtectedRecord,
    ProtectedStateError, ProtectedStateStore, SoftwareKeyProvider, StateDomain,
};
use ptr_types::{Digest, Generation, KeyId, Revision};

fn digest(bytes: &[u8]) -> Digest {
    use sha2::{Digest as ShaDigest, Sha256};
    Sha256::digest(bytes).into()
}

fn anchor(logical_id: &str, generation: Generation, previous: Option<Digest>) -> GenerationAnchor {
    let mut value = Vec::new();
    value.extend_from_slice(logical_id.as_bytes());
    value.extend_from_slice(&generation.0.to_le_bytes());
    if let Some(previous) = previous {
        value.extend_from_slice(&previous);
    }
    GenerationAnchor {
        logical_id: logical_id.into(),
        generation,
        previous_digest: previous,
        anchor_digest: digest(&value),
    }
}

fn record(generation: Generation, payload: &[u8], previous: Option<Digest>) -> ProtectedRecord {
    ProtectedRecord {
        domain: StateDomain::Knowledge,
        logical_id: "fact".into(),
        generation,
        revision: Revision(generation.0),
        plaintext_digest: digest(payload),
        payload: payload.to_vec(),
        key_id: KeyId::from("key-1"),
        anchor: anchor("fact", generation, previous),
    }
}

#[test]
fn protected_state_round_trips_without_exposing_plaintext() {
    let keys = SoftwareKeyProvider::default().with_key(KeyId::from("key-1"), [7; 32]);
    let mut store = InMemoryProtectedStateStore::new(keys);
    let payload = b"private knowledge";
    let handle = store.seal(record(Generation(1), payload, None)).unwrap();
    assert_ne!(handle.ciphertext_digest, digest(payload));
    assert_eq!(store.open(&handle).unwrap(), payload);
    assert!(store.verify(&handle).unwrap().valid);
    assert_eq!(
        store.seal(record(Generation(1), payload, None)).unwrap(),
        handle
    );
}

#[test]
fn rollback_and_tampered_anchor_are_rejected() {
    let keys = SoftwareKeyProvider::default().with_key(KeyId::from("key-1"), [7; 32]);
    let mut store = InMemoryProtectedStateStore::new(keys);
    let first = store.seal(record(Generation(1), b"one", None)).unwrap();
    let second = store
        .seal(record(Generation(2), b"two", Some(first.anchor_digest)))
        .unwrap();
    assert_eq!(
        store.seal(record(Generation(1), b"old", None)),
        Err(ProtectedStateError::Rollback)
    );
    let mut forged = record(Generation(3), b"three", Some(second.anchor_digest));
    forged.anchor.anchor_digest = [0; 32];
    assert_eq!(store.seal(forged), Err(ProtectedStateError::AnchorMismatch));
}

#[test]
fn revoked_key_blocks_opening() {
    let key_id = KeyId::from("key-1");
    let keys = SoftwareKeyProvider::default().with_key(key_id.clone(), [7; 32]);
    let mut store = InMemoryProtectedStateStore::new(keys);
    let handle = store.seal(record(Generation(1), b"secret", None)).unwrap();
    store.revoke_key(&key_id).unwrap();
    assert_eq!(
        store.open(&handle),
        Err(ProtectedStateError::RevokedKey(key_id))
    );
}

#[test]
fn file_store_reopens_encrypted_state() {
    let path = std::env::temp_dir().join(format!("ptr-protected-{}.bin", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let key = KeyId::from("key-1");
    let keys = SoftwareKeyProvider::default().with_key(key.clone(), [7; 32]);
    let handle = {
        let mut store = FileProtectedStateStore::open(&path, keys).unwrap();
        store.seal(record(Generation(1), b"durable", None)).unwrap()
    };
    let keys = SoftwareKeyProvider::default().with_key(key, [7; 32]);
    let mut store = FileProtectedStateStore::open(&path, keys).unwrap();
    assert_eq!(store.open(&handle).unwrap(), b"durable");
    let next = store
        .seal(record(
            Generation(2),
            b"durable-v2",
            Some(handle.anchor_digest),
        ))
        .unwrap();
    drop(store);
    let keys = SoftwareKeyProvider::default().with_key(KeyId::from("key-1"), [7; 32]);
    let store = FileProtectedStateStore::open(&path, keys).unwrap();
    assert_eq!(store.open(&next).unwrap(), b"durable-v2");
    let _ = std::fs::remove_file(path);
}
