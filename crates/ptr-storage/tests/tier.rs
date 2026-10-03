use ptr_storage::{
    ChunkDescriptor, CpuTierBackend, FileTierBackend, PreparedTierObject, StorageTier, TierBackend,
    TierBackendId, TierObjectDomain, TierObjectManifest,
};
use ptr_types::{Generation, Revision};
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

fn block_on_tier<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn chunk(index: u32, offset: u64, bytes: &[u8]) -> ChunkDescriptor {
    ChunkDescriptor::for_bytes(index, offset, bytes)
}

#[test]
fn manifest_identity_is_location_independent_and_ordered() {
    let a = chunk(0, 0, b"alpha");
    let b = chunk(1, 5, b"beta");
    let manifest = TierObjectManifest::new(
        TierObjectDomain::KvSnapshot,
        "state-1",
        Generation(7),
        Revision(9),
        [3; 32],
        vec![a.clone(), b.clone()],
    )
    .unwrap();

    let same = TierObjectManifest::new(
        TierObjectDomain::KvSnapshot,
        "state-1",
        Generation(7),
        Revision(9),
        [3; 32],
        vec![a, b],
    )
    .unwrap();

    assert_eq!(manifest.root_digest, same.root_digest);
    assert!(manifest.validate().is_ok());
}

#[test]
fn manifest_rejects_gaps_and_ambiguous_chunk_order() {
    let first = chunk(0, 0, b"alpha");
    let second = chunk(2, 5, b"beta");
    assert!(TierObjectManifest::new(
        TierObjectDomain::Artifact,
        "artifact",
        Generation(1),
        Revision(1),
        [0; 32],
        vec![first, second],
    )
    .is_err());
}

#[test]
fn cpu_backend_is_immutable_and_verifies_on_read() {
    let backend = CpuTierBackend::new(TierBackendId::from("cpu-0"));
    let descriptor = chunk(0, 0, b"payload");

    let receipt =
        block_on_tier(backend.put_chunk(descriptor.clone(), b"payload".to_vec())).unwrap();
    assert_eq!(receipt.digest, descriptor.digest);
    assert_eq!(
        block_on_tier(backend.get_chunk(&descriptor)).unwrap(),
        b"payload"
    );
    assert!(
        block_on_tier(backend.verify_chunk(&descriptor))
            .unwrap()
            .valid
    );
    assert!(block_on_tier(backend.put_chunk(descriptor.clone(), b"changed".to_vec())).is_err());
}

#[test]
fn file_backend_publishes_atomically_and_reopens_by_digest() {
    let root = std::env::temp_dir().join(format!(
        "ptr-tier-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = std::fs::remove_dir_all(&root);
    let backend = FileTierBackend::new(TierBackendId::from("nvme-0"), &root).unwrap();
    let descriptor = chunk(0, 0, b"durable");

    block_on_tier(backend.put_chunk(descriptor.clone(), b"durable".to_vec())).unwrap();
    drop(backend);

    let reopened = FileTierBackend::new(TierBackendId::from("nvme-0"), &root).unwrap();
    assert_eq!(reopened.capabilities().tier, StorageTier::Nvme);
    assert_eq!(
        block_on_tier(reopened.get_chunk(&descriptor)).unwrap(),
        b"durable"
    );
    assert!(
        block_on_tier(reopened.verify_chunk(&descriptor))
            .unwrap()
            .valid
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn corrupted_file_chunk_is_never_returned_as_valid_data() {
    let root = std::env::temp_dir().join(format!("ptr-tier-corrupt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let backend = FileTierBackend::new(TierBackendId::from("nvme-0"), &root).unwrap();
    let descriptor = chunk(0, 0, b"expected");
    block_on_tier(backend.put_chunk(descriptor.clone(), b"expected".to_vec())).unwrap();
    std::fs::write(backend.chunk_path(&descriptor), b"tampered").unwrap();

    assert!(block_on_tier(backend.get_chunk(&descriptor)).is_err());
    assert!(
        !block_on_tier(backend.verify_chunk(&descriptor))
            .unwrap()
            .valid
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn prepared_object_chunks_and_reassembles_without_changing_identity() {
    let prepared = PreparedTierObject::from_bytes(
        TierObjectDomain::Artifact,
        "artifact-a",
        Generation(2),
        Revision(3),
        [8; 32],
        b"abcdefghij",
        4,
    )
    .unwrap();
    assert_eq!(prepared.chunks.len(), 3);
    assert_eq!(prepared.reassemble().unwrap(), b"abcdefghij");
    let mut damaged = prepared.clone();
    damaged.chunks[1][0] ^= 1;
    assert!(damaged.verify().is_err());
    assert_eq!(
        prepared.manifest.root_digest,
        prepared.manifest.canonical_digest()
    );
}
