#![cfg(feature = "tier-opendal")]

#[cfg(feature = "tier-opendal-fs")]
use opendal::services::Fs;
use opendal::{services::Memory, Operator};
use ptr_storage::{ChunkDescriptor, OpenDalTierBackend, TierBackend, TierBackendId, TierError};
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
fn memory_operator_roundtrips_and_verifies_chunks() {
    let operator = Operator::new(Memory::default()).unwrap();
    let backend =
        OpenDalTierBackend::new(TierBackendId::from("opendal-memory"), operator, "ptr/tests")
            .unwrap();
    let chunk = ChunkDescriptor::for_bytes(0, 0, b"object-store-chunk");

    block_on(backend.put_chunk(chunk.clone(), b"object-store-chunk".to_vec())).unwrap();
    assert_eq!(
        block_on(backend.get_chunk(&chunk)).unwrap(),
        b"object-store-chunk"
    );
    assert!(block_on(backend.verify_chunk(&chunk)).unwrap().valid);
    block_on(backend.delete_chunk(&chunk)).unwrap();
    assert_eq!(
        block_on(backend.get_chunk(&chunk)),
        Err(TierError::UnknownChunk)
    );
}

#[test]
fn put_chunk_repairs_a_corrupt_replica_at_the_same_key() {
    let operator = Operator::new(Memory::default()).unwrap();
    let backend = OpenDalTierBackend::new(
        TierBackendId::from("opendal-memory"),
        operator.clone(),
        "ptr/tests",
    )
    .unwrap();
    let chunk = ChunkDescriptor::for_bytes(0, 0, b"object-store-chunk");
    block_on(operator.write(&backend.chunk_key(&chunk), b"damaged-replica".to_vec())).unwrap();
    assert!(!block_on(backend.verify_chunk(&chunk)).unwrap().valid);

    block_on(backend.put_chunk(chunk.clone(), b"object-store-chunk".to_vec())).unwrap();

    assert!(block_on(backend.verify_chunk(&chunk)).unwrap().valid);
    assert_eq!(
        block_on(backend.get_chunk(&chunk)).unwrap(),
        b"object-store-chunk"
    );
}

#[test]
fn repair_handles_digest_and_length_corruption_and_returns_a_stable_receipt() {
    let operator = Operator::new(Memory::default()).unwrap();
    let backend =
        OpenDalTierBackend::new(TierBackendId::from("repair"), operator.clone(), "repairs")
            .unwrap();
    let bytes = b"good".to_vec();
    let chunk = ChunkDescriptor::for_bytes(0, 0, &bytes);
    for corrupt in [b"evil".to_vec(), vec![], b"too long".to_vec()] {
        block_on(operator.write(&backend.chunk_key(&chunk), corrupt)).unwrap();
        assert_eq!(
            block_on(backend.get_chunk(&chunk)),
            Err(TierError::CorruptChunk)
        );
        let receipt = block_on(backend.put_chunk(chunk.clone(), bytes.clone())).unwrap();
        assert_eq!(receipt.backend, TierBackendId::from("repair"));
        assert_eq!(receipt.digest, chunk.digest);
        assert_eq!(receipt.length, chunk.length);
        assert_eq!(block_on(backend.get_chunk(&chunk)).unwrap(), bytes);
        assert_eq!(
            block_on(backend.put_chunk(chunk.clone(), bytes.clone())).unwrap(),
            receipt
        );
    }
}

#[test]
fn invalid_repair_bytes_leave_both_valid_and_corrupt_stored_objects_untouched() {
    let operator = Operator::new(Memory::default()).unwrap();
    let backend =
        OpenDalTierBackend::new(TierBackendId::from("repair"), operator.clone(), "repairs")
            .unwrap();
    let chunk = ChunkDescriptor::for_bytes(0, 0, b"good");
    for stored in [b"good".to_vec(), b"bad".to_vec()] {
        block_on(operator.write(&backend.chunk_key(&chunk), stored.clone())).unwrap();
        for (invalid, error) in [
            (b"evil".to_vec(), TierError::DigestMismatch),
            (vec![], TierError::LengthMismatch),
        ] {
            assert_eq!(
                block_on(backend.put_chunk(chunk.clone(), invalid)),
                Err(error)
            );
            assert_eq!(
                block_on(operator.read(&backend.chunk_key(&chunk)))
                    .unwrap()
                    .to_vec(),
                stored
            );
        }
    }
}

#[cfg(feature = "tier-opendal-fs")]
#[tokio::test]
async fn filesystem_operator_survives_backend_recreation() {
    let root = std::env::temp_dir().join(format!("ptr-opendal-fs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let operator = Operator::new(Fs::default().root(root.to_str().unwrap())).unwrap();
    let backend =
        OpenDalTierBackend::new(TierBackendId::from("opendal-fs"), operator, "ptr/tests").unwrap();
    let chunk = ChunkDescriptor::for_bytes(0, 0, b"filesystem-object");
    backend
        .put_chunk(chunk.clone(), b"filesystem-object".to_vec())
        .await
        .unwrap();
    drop(backend);

    let operator = Operator::new(Fs::default().root(root.to_str().unwrap())).unwrap();
    let reopened =
        OpenDalTierBackend::new(TierBackendId::from("opendal-fs"), operator, "ptr/tests").unwrap();
    assert_eq!(
        reopened.get_chunk(&chunk).await.unwrap(),
        b"filesystem-object"
    );
    assert!(reopened.verify_chunk(&chunk).await.unwrap().valid);
    std::fs::remove_dir_all(root).unwrap();
}
