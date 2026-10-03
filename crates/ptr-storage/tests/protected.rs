use aes_gcm_siv::{
    aead::{Aead, KeyInit, Payload},
    Aes256GcmSiv, Nonce,
};
use ptr_storage::{
    FileProtectedStateStore, GenerationAnchor, InMemoryProtectedStateStore, ProtectedHandle,
    ProtectedRecord, ProtectedStateError, ProtectedStateStore, SoftwareKeyProvider, StateDomain,
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

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value);
}

fn put_text(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn legacy_v1_file(record: &ProtectedRecord, key: [u8; 32]) -> (Vec<u8>, ProtectedHandle) {
    let mut aad = b"ptr-protected-state-v1".to_vec();
    aad.extend_from_slice(b"knowledge");
    aad.extend_from_slice(&(record.logical_id.len() as u64).to_le_bytes());
    aad.extend_from_slice(record.logical_id.as_bytes());
    aad.extend_from_slice(&record.generation.0.to_le_bytes());
    aad.extend_from_slice(&record.revision.0.to_le_bytes());
    aad.extend_from_slice(record.key_id.0.as_bytes());
    aad.extend_from_slice(&record.plaintext_digest);
    let aad_digest = digest(&aad);
    let nonce: [u8; 12] = aad_digest[..12].try_into().unwrap();
    let cipher = Aes256GcmSiv::new_from_slice(&key).unwrap();
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &record.payload,
                aad: &aad,
            },
        )
        .unwrap();
    let ciphertext_digest = digest(&ciphertext);
    let handle = ProtectedHandle {
        domain: record.domain,
        logical_id: record.logical_id.clone(),
        generation: record.generation,
        revision: record.revision,
        ciphertext_digest,
        anchor_digest: record.anchor.anchor_digest,
        key_id: record.key_id.clone(),
    };

    let mut bytes = b"PTRPST01".to_vec();
    put_u32(&mut bytes, 1);
    bytes.push(1);
    put_text(&mut bytes, &handle.logical_id);
    put_u64(&mut bytes, handle.generation.0);
    put_u64(&mut bytes, handle.revision.0);
    bytes.extend_from_slice(&handle.ciphertext_digest);
    bytes.extend_from_slice(&handle.anchor_digest);
    put_text(&mut bytes, &handle.key_id.0);
    put_text(&mut bytes, &record.anchor.logical_id);
    put_u64(&mut bytes, record.anchor.generation.0);
    match record.anchor.previous_digest {
        Some(previous) => {
            bytes.push(1);
            bytes.extend_from_slice(&previous);
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&record.anchor.anchor_digest);
    put_bytes(&mut bytes, &aad);
    bytes.extend_from_slice(&nonce);
    put_bytes(&mut bytes, &ciphertext);
    bytes.extend_from_slice(&record.plaintext_digest);
    (bytes, handle)
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

#[test]
fn file_store_reads_v1_records_and_rewrites_as_v2_without_losing_them() {
    let path = std::env::temp_dir().join(format!(
        "ptr-protected-v1-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&path);
    let key_id = KeyId::from("key-1");
    let first_record = record(Generation(1), b"legacy", None);
    let (bytes, first) = legacy_v1_file(&first_record, [7; 32]);
    std::fs::write(&path, bytes).unwrap();

    let keys = SoftwareKeyProvider::default().with_key(key_id.clone(), [7; 32]);
    let second = {
        let mut store = FileProtectedStateStore::open(&path, keys).unwrap();
        assert_eq!(store.open(&first).unwrap(), b"legacy");
        store
            .seal(record(Generation(2), b"current", Some(first.anchor_digest)))
            .unwrap()
    };
    assert_eq!(&std::fs::read(&path).unwrap()[..8], b"PTRPST02");

    let keys = SoftwareKeyProvider::default().with_key(key_id, [7; 32]);
    let store = FileProtectedStateStore::open(&path, keys).unwrap();
    assert_eq!(store.open(&first).unwrap(), b"legacy");
    assert_eq!(store.open(&second).unwrap(), b"current");
    let _ = std::fs::remove_file(path);
}

#[test]
fn file_store_rejects_tampered_persisted_anchor_metadata() {
    let path = std::env::temp_dir().join(format!(
        "ptr-protected-anchor-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&path);
    let key = KeyId::from("key-1");
    let keys = SoftwareKeyProvider::default().with_key(key.clone(), [7; 32]);
    {
        let mut store = FileProtectedStateStore::open(&path, keys).unwrap();
        store.seal(record(Generation(1), b"durable", None)).unwrap();
    }

    let mut bytes = std::fs::read(&path).unwrap();
    let anchor_id = bytes
        .windows(b"fact".len())
        .enumerate()
        .filter_map(|(index, window)| (window == b"fact").then_some(index))
        .nth(1)
        .expect("serialized anchor logical id");
    bytes[anchor_id] = b'F';
    std::fs::write(&path, bytes).unwrap();

    let keys = SoftwareKeyProvider::default().with_key(key, [7; 32]);
    assert!(matches!(
        FileProtectedStateStore::open(&path, keys),
        Err(ProtectedStateError::AnchorMismatch)
    ));
    let _ = std::fs::remove_file(path);
}
