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
