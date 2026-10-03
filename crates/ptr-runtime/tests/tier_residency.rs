use ptr_runtime::{
    ActiveWriteBinding, BackendLifecycleState, BudgetedReplicaPolicy, DeterministicLruPolicy,
    LookaheadPrefetchPolicy, ReplicaState, TierPlacementPolicy, TierPlanningContext,
    TierPolicyError, TierResidencyController, TierRuntimeError, TierTransferAuthorization,
};
use ptr_storage::{
    ChunkDescriptor, ChunkReceipt, CpuTierBackend, FileTierBackend, StorageTier, TierBackend,
    TierBackendId, TierCapabilities, TierFuture, TierHealth, TierIntegrityReport, TierObjectDomain,
    TierObjectManifest,
};
use ptr_types::{Generation, Revision};
use std::future::Future;
use std::pin::pin;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Barrier,
};
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

fn manifest(bytes: &[u8]) -> TierObjectManifest {
    manifest_with_id("state-1", bytes)
}

fn manifest_with_id(logical_id: &str, bytes: &[u8]) -> TierObjectManifest {
    TierObjectManifest::new(
        TierObjectDomain::KvSnapshot,
        logical_id,
        Generation(4),
        Revision(8),
        [9; 32],
        vec![ChunkDescriptor::for_bytes(0, 0, bytes)],
    )
    .unwrap()
}

struct PausingVerifyBackend {
    inner: CpuTierBackend,
    pause: AtomicBool,
    started: Arc<Barrier>,
    resume: Arc<Barrier>,
}

impl PausingVerifyBackend {
    fn new(id: &str, started: Arc<Barrier>, resume: Arc<Barrier>) -> Self {
        Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
            pause: AtomicBool::new(false),
            started,
            resume,
        }
    }
}

impl TierBackend for PausingVerifyBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        self.inner.capabilities()
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        self.inner.put_chunk(chunk, bytes)
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        self.inner.get_chunk(chunk)
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let report = self.inner.verify_chunk(chunk).await?;
            if self.pause.load(Ordering::SeqCst) {
                self.started.wait();
                self.resume.wait();
            }
            Ok(report)
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        self.inner.delete_chunk(chunk)
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

struct PausingPutBackend {
    inner: CpuTierBackend,
    started: Arc<Barrier>,
    resume: Arc<Barrier>,
}

impl PausingPutBackend {
    fn new(id: &str, started: Arc<Barrier>, resume: Arc<Barrier>) -> Self {
        Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
            started,
            resume,
        }
    }
}

impl TierBackend for PausingPutBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        self.inner.capabilities()
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        Box::pin(async move {
            self.started.wait();
            self.resume.wait();
            self.inner.put_chunk(chunk, bytes).await
        })
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        self.inner.get_chunk(chunk)
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        self.inner.verify_chunk(chunk)
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        self.inner.delete_chunk(chunk)
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

struct CountingBackend {
    inner: CpuTierBackend,
    gets: AtomicUsize,
}

struct DelayedDeleteBackend {
    inner: CpuTierBackend,
    deletion_started: Arc<Barrier>,
}

struct PendingReadBackend {
    inner: CpuTierBackend,
}

impl PendingReadBackend {
    fn new(id: &str) -> Self {
        Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
        }
    }
}

impl TierBackend for PendingReadBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        self.inner.capabilities()
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        self.inner.put_chunk(chunk, bytes)
    }

    fn get_chunk<'a>(&'a self, _chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(std::future::pending())
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        self.inner.verify_chunk(chunk)
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        self.inner.delete_chunk(chunk)
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

impl DelayedDeleteBackend {
    fn new(id: &str, deletion_started: Arc<Barrier>) -> Self {
        Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
            deletion_started,
        }
    }
}

impl TierBackend for DelayedDeleteBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        self.inner.capabilities()
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        self.inner.put_chunk(chunk, bytes)
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        self.inner.get_chunk(chunk)
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        self.inner.verify_chunk(chunk)
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        Box::pin(async move {
            self.deletion_started.wait();
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.inner.delete_chunk(chunk).await
        })
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

impl CountingBackend {
    fn new(id: &str) -> Self {
        Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
            gets: AtomicUsize::new(0),
        }
    }
}

impl TierBackend for CountingBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        self.inner.capabilities()
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        self.inner.put_chunk(chunk, bytes)
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(async move {
            self.gets.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.inner.get_chunk(chunk).await
        })
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        self.inner.verify_chunk(chunk)
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        self.inner.delete_chunk(chunk)
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

#[test]
fn transfer_publishes_only_verified_destination_and_preserves_source() {
    let controller = TierResidencyController::default();
    let cpu = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu")));
    let root = std::env::temp_dir().join(format!("ptr-runtime-tier-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let nvme = Arc::new(FileTierBackend::new(TierBackendId::from("nvme"), &root).unwrap());
    let object = manifest(b"snapshot");

    block_on(controller.attach_backend(cpu.clone())).unwrap();
    block_on(controller.attach_backend(nvme.clone())).unwrap();
    block_on(cpu.put_chunk(object.chunks[0].clone(), b"snapshot".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), cpu.identity())).unwrap();
    block_on(controller.transfer(object.root_digest, &cpu.identity(), &nvme.identity(), None))
        .unwrap();

    let record = controller.residency(&object.root_digest).unwrap();
    assert_eq!(record.available_replicas(), 2);
    assert!(record
        .replicas
        .iter()
        .all(|replica| replica.state == ReplicaState::Available));
    assert!(block_on(cpu.verify_chunk(&object.chunks[0])).unwrap().valid);
    assert!(
        block_on(nvme.verify_chunk(&object.chunks[0]))
            .unwrap()
            .valid
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn registering_same_manifest_discovers_replica_on_second_backend() {
    let controller = TierResidencyController::default();
    let first = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-discovery-a")));
    let second = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-discovery-b")));
    let object = manifest(b"discovered-replica");
    block_on(controller.attach_backend(first.clone())).unwrap();
    block_on(controller.attach_backend(second.clone())).unwrap();
    for backend in [&first, &second] {
        block_on(backend.put_chunk(object.chunks[0].clone(), b"discovered-replica".to_vec()))
            .unwrap();
    }
    block_on(controller.register_object(object.clone(), first.identity())).unwrap();
    block_on(controller.register_object(object.clone(), second.identity())).unwrap();
    assert_eq!(
        controller
            .residency(&object.root_digest)
            .unwrap()
            .available_replicas(),
        2
    );
}

#[test]
fn pins_prevent_eviction_and_release_is_idempotent() {
    let controller = TierResidencyController::default();
    let first = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-a")));
    let second = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-b")));
    let object = manifest(b"snapshot");
    block_on(controller.attach_backend(first.clone())).unwrap();
    block_on(controller.attach_backend(second.clone())).unwrap();
    block_on(first.put_chunk(object.chunks[0].clone(), b"snapshot".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), first.identity())).unwrap();
    block_on(controller.transfer(
        object.root_digest,
        &first.identity(),
        &second.identity(),
        None,
    ))
    .unwrap();

    let pin = controller
        .pin(object.root_digest, &first.identity())
        .unwrap();
    assert_eq!(
        block_on(controller.evict(object.root_digest, &first.identity())),
        Err(TierRuntimeError::ReplicaPinned)
    );
    controller.release_pin(pin.clone()).unwrap();
    controller.release_pin(pin).unwrap();
    block_on(controller.evict(object.root_digest, &first.identity())).unwrap();
}

#[test]
fn drain_rejects_new_transfers_until_backend_is_empty() {
    let controller = TierResidencyController::default();
    let first = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-a")));
    let second = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-b")));
    let object = manifest(b"snapshot");
    block_on(controller.attach_backend(first.clone())).unwrap();
    block_on(controller.attach_backend(second.clone())).unwrap();
    block_on(first.put_chunk(object.chunks[0].clone(), b"snapshot".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), first.identity())).unwrap();

    controller.begin_detach(&second.identity()).unwrap();
    assert_eq!(
        block_on(controller.transfer(
            object.root_digest,
            &first.identity(),
            &second.identity(),
            None
        )),
        Err(TierRuntimeError::BackendDraining)
    );
    controller.finish_detach(&second.identity()).unwrap();
    assert_eq!(
        controller.backend_state(&second.identity()).unwrap(),
        BackendLifecycleState::Detached
    );
}

#[test]
fn draining_source_can_be_evacuated_before_detach() {
    let controller = TierResidencyController::default();
    let first = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-a")));
    let second = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-b")));
    let object = manifest(b"snapshot");
    block_on(controller.attach_backend(first.clone())).unwrap();
    block_on(controller.attach_backend(second.clone())).unwrap();
    block_on(first.put_chunk(object.chunks[0].clone(), b"snapshot".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), first.identity())).unwrap();

    controller.begin_detach(&first.identity()).unwrap();
    block_on(controller.transfer(
        object.root_digest,
        &first.identity(),
        &second.identity(),
        None,
    ))
    .unwrap();
    block_on(controller.evict(object.root_digest, &first.identity())).unwrap();
    controller.finish_detach(&first.identity()).unwrap();

    assert_eq!(
        controller.backend_state(&first.identity()).unwrap(),
        BackendLifecycleState::Detached
    );
    let record = controller.residency(&object.root_digest).unwrap();
    assert_eq!(record.available_replicas(), 1);
    assert_eq!(
        record
            .replicas
            .iter()
            .find(|replica| replica.backend == second.identity())
            .unwrap()
            .state,
        ReplicaState::Available
    );
}

#[test]
fn stateful_snapshot_transfer_rejects_stale_fencing() {
    let controller = TierResidencyController::default();
    let first = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-a")));
    let second = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-b")));
    let object = manifest(b"snapshot");
    block_on(controller.attach_backend(first.clone())).unwrap();
    block_on(controller.attach_backend(second.clone())).unwrap();
    block_on(first.put_chunk(object.chunks[0].clone(), b"snapshot".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), first.identity())).unwrap();
    controller
        .bind_active_write(
            object.root_digest,
            ActiveWriteBinding {
                generation: Generation(4),
                placement_epoch: 11,
                fencing_token: 99,
            },
        )
        .unwrap();

    assert_eq!(
        block_on(controller.transfer(
            object.root_digest,
            &first.identity(),
            &second.identity(),
            Some(TierTransferAuthorization {
                generation: Generation(4),
                placement_epoch: 10,
                fencing_token: 99,
            }),
        )),
        Err(TierRuntimeError::StaleWriteBinding)
    );
}

#[test]
fn deterministic_lru_uses_digest_as_stable_tie_breaker() {
    let policy = DeterministicLruPolicy;
    let a = [1; 32];
    let b = [2; 32];
    assert_eq!(policy.choose_eviction(&[(b, 7), (a, 7)]), Some(a));
}

#[test]
fn concurrent_fetches_share_one_inflight_transfer() {
    let controller = Arc::new(TierResidencyController::default());
    let source = Arc::new(CountingBackend::new("cpu-source"));
    let destination = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-destination")));
    let object = manifest(b"deduplicated-snapshot");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"deduplicated-snapshot".to_vec()))
        .unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();

    let start = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let controller = controller.clone();
        let source_id = source.identity();
        let destination_id = destination.identity();
        let start = start.clone();
        let root = object.root_digest;
        workers.push(std::thread::spawn(move || {
            start.wait();
            block_on(controller.transfer(root, &source_id, &destination_id, None)).unwrap()
        }));
    }
    start.wait();
    let first = workers.remove(0).join().unwrap();
    let second = workers.remove(0).join().unwrap();
    assert_eq!(first, second);
    assert_eq!(source.gets.load(Ordering::SeqCst), 1);
}

#[test]
fn inflight_waiter_must_pass_its_own_fencing_authorization() {
    let controller = Arc::new(TierResidencyController::default());
    let source = Arc::new(CountingBackend::new("cpu-source-auth"));
    let destination = Arc::new(CpuTierBackend::new(TierBackendId::from(
        "cpu-destination-auth",
    )));
    let object = manifest(b"authorized-snapshot");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"authorized-snapshot".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();
    controller
        .bind_active_write(
            object.root_digest,
            ActiveWriteBinding {
                generation: Generation(4),
                placement_epoch: 11,
                fencing_token: 99,
            },
        )
        .unwrap();

    let leader = {
        let controller = controller.clone();
        let source = source.identity();
        let destination = destination.identity();
        let root = object.root_digest;
        std::thread::spawn(move || {
            block_on(controller.transfer(
                root,
                &source,
                &destination,
                Some(TierTransferAuthorization {
                    generation: Generation(4),
                    placement_epoch: 11,
                    fencing_token: 99,
                }),
            ))
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert_eq!(
        block_on(controller.transfer(
            object.root_digest,
            &source.identity(),
            &destination.identity(),
            Some(TierTransferAuthorization {
                generation: Generation(4),
                placement_epoch: 10,
                fencing_token: 99,
            }),
        )),
        Err(TierRuntimeError::StaleWriteBinding)
    );
    leader.join().unwrap().unwrap();
}

#[test]
fn evicting_replica_rejects_new_pins_and_transfers() {
    let controller = Arc::new(TierResidencyController::default());
    let deletion_started = Arc::new(Barrier::new(2));
    let source = Arc::new(DelayedDeleteBackend::new(
        "cpu-evicting",
        deletion_started.clone(),
    ));
    let survivor = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-survivor")));
    let third = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-third")));
    let object = manifest(b"eviction-race");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(survivor.clone())).unwrap();
    block_on(controller.attach_backend(third.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"eviction-race".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();
    block_on(controller.transfer(
        object.root_digest,
        &source.identity(),
        &survivor.identity(),
        None,
    ))
    .unwrap();

    let evictor = {
        let controller = controller.clone();
        let source = source.identity();
        let root = object.root_digest;
        std::thread::spawn(move || block_on(controller.evict(root, &source)))
    };
    deletion_started.wait();
    assert_eq!(
        controller.pin(object.root_digest, &source.identity()),
        Err(TierRuntimeError::ReplicaUnavailable)
    );
    assert_eq!(
        block_on(controller.transfer(
            object.root_digest,
            &source.identity(),
            &third.identity(),
            None,
        )),
        Err(TierRuntimeError::ReplicaUnavailable)
    );
    evictor.join().unwrap().unwrap();
}

#[test]
fn registration_reservation_prevents_shared_chunk_eviction() {
    let controller = Arc::new(TierResidencyController::default());
    let verification_started = Arc::new(Barrier::new(2));
    let verification_resume = Arc::new(Barrier::new(2));
    let source = Arc::new(PausingVerifyBackend::new(
        "cpu-shared",
        verification_started.clone(),
        verification_resume.clone(),
    ));
    let survivor = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-survivor")));
    let first = manifest_with_id("state-a", b"shared-content");
    let second = manifest_with_id("state-b", b"shared-content");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(survivor.clone())).unwrap();
    block_on(source.put_chunk(first.chunks[0].clone(), b"shared-content".to_vec())).unwrap();
    block_on(controller.register_object(first.clone(), source.identity())).unwrap();
    block_on(controller.transfer(
        first.root_digest,
        &source.identity(),
        &survivor.identity(),
        None,
    ))
    .unwrap();

    source.pause.store(true, Ordering::SeqCst);
    let registration = {
        let controller = controller.clone();
        let source = source.identity();
        let second = second.clone();
        std::thread::spawn(move || block_on(controller.register_object(second, source)))
    };
    verification_started.wait();
    block_on(controller.evict(first.root_digest, &source.identity())).unwrap();
    verification_resume.wait();
    registration.join().unwrap().unwrap();

    assert_eq!(
        block_on(source.get_chunk(&second.chunks[0])).unwrap(),
        b"shared-content"
    );
    assert_eq!(
        controller
            .residency(&second.root_digest)
            .unwrap()
            .available_replicas(),
        1
    );
}

#[test]
fn registration_is_rejected_while_shared_chunk_deletion_is_inflight() {
    let controller = Arc::new(TierResidencyController::default());
    let deletion_started = Arc::new(Barrier::new(2));
    let source = Arc::new(DelayedDeleteBackend::new(
        "cpu-delete-race",
        deletion_started.clone(),
    ));
    let survivor = Arc::new(CpuTierBackend::new(TierBackendId::from(
        "cpu-delete-survivor",
    )));
    let first = manifest_with_id("state-a", b"shared-delete-content");
    let second = manifest_with_id("state-b", b"shared-delete-content");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(survivor.clone())).unwrap();
    block_on(source.put_chunk(first.chunks[0].clone(), b"shared-delete-content".to_vec())).unwrap();
    block_on(controller.register_object(first.clone(), source.identity())).unwrap();
    block_on(controller.transfer(
        first.root_digest,
        &source.identity(),
        &survivor.identity(),
        None,
    ))
    .unwrap();

    let evictor = {
        let controller = controller.clone();
        let source = source.identity();
        let root = first.root_digest;
        std::thread::spawn(move || block_on(controller.evict(root, &source)))
    };
    deletion_started.wait();
    assert_eq!(
        block_on(controller.register_object(second.clone(), source.identity())),
        Err(TierRuntimeError::ReplicaUnavailable)
    );
    evictor.join().unwrap().unwrap();
    assert_eq!(
        controller.residency(&second.root_digest),
        Err(TierRuntimeError::UnknownObject)
    );
}

#[test]
fn registration_cannot_publish_after_backend_starts_draining() {
    let controller = Arc::new(TierResidencyController::default());
    let verification_started = Arc::new(Barrier::new(2));
    let verification_resume = Arc::new(Barrier::new(2));
    let backend = Arc::new(PausingVerifyBackend::new(
        "cpu-draining",
        verification_started.clone(),
        verification_resume.clone(),
    ));
    let object = manifest(b"late-registration");
    block_on(controller.attach_backend(backend.clone())).unwrap();
    block_on(backend.put_chunk(object.chunks[0].clone(), b"late-registration".to_vec())).unwrap();
    backend.pause.store(true, Ordering::SeqCst);

    let registration = {
        let controller = controller.clone();
        let backend = backend.identity();
        let object = object.clone();
        std::thread::spawn(move || block_on(controller.register_object(object, backend)))
    };
    verification_started.wait();
    controller.begin_detach(&backend.identity()).unwrap();
    verification_resume.wait();
    assert_eq!(
        registration.join().unwrap(),
        Err(TierRuntimeError::BackendDraining)
    );
    assert_eq!(
        controller.residency(&object.root_digest),
        Err(TierRuntimeError::UnknownObject)
    );
    controller.finish_detach(&backend.identity()).unwrap();
}

#[test]
fn transfer_completion_does_not_reopen_a_draining_backend() {
    let controller = Arc::new(TierResidencyController::default());
    let put_started = Arc::new(Barrier::new(2));
    let put_resume = Arc::new(Barrier::new(2));
    let source = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-source")));
    let destination = Arc::new(PausingPutBackend::new(
        "cpu-draining-target",
        put_started.clone(),
        put_resume.clone(),
    ));
    let object = manifest(b"drain-race");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"drain-race".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();

    let transfer = {
        let controller = controller.clone();
        let source = source.identity();
        let destination = destination.identity();
        let root = object.root_digest;
        std::thread::spawn(move || block_on(controller.transfer(root, &source, &destination, None)))
    };
    put_started.wait();
    controller.begin_detach(&destination.identity()).unwrap();
    put_resume.wait();
    assert_eq!(
        transfer.join().unwrap(),
        Err(TierRuntimeError::BackendDraining)
    );
    let residency = controller.residency(&object.root_digest).unwrap();
    assert_eq!(
        residency
            .replicas
            .iter()
            .find(|replica| replica.backend == destination.identity())
            .unwrap()
            .state,
        ReplicaState::Corrupt
    );
    assert_eq!(
        controller.pin(object.root_digest, &destination.identity()),
        Err(TierRuntimeError::ReplicaUnavailable)
    );
}

#[test]
fn dropping_last_pin_releases_the_replica_for_eviction() {
    let controller = TierResidencyController::default();
    let source = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-source")));
    let survivor = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-survivor")));
    let object = manifest(b"raii-pin");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(survivor.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"raii-pin".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();
    block_on(controller.transfer(
        object.root_digest,
        &source.identity(),
        &survivor.identity(),
        None,
    ))
    .unwrap();

    let pin = controller
        .pin(object.root_digest, &source.identity())
        .unwrap();
    drop(pin);
    block_on(controller.evict(object.root_digest, &source.identity())).unwrap();
}

#[test]
fn dropping_transfer_future_releases_pin_budget_and_waiters() {
    let controller = TierResidencyController::with_transfer_budget(1, 1024).unwrap();
    let source = Arc::new(PendingReadBackend::new("cpu-pending"));
    let destination = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-target")));
    let object = manifest(b"cancelled-transfer");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"cancelled-transfer".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();

    let source_id = source.identity();
    let destination_id = destination.identity();
    let mut future =
        Box::pin(controller.transfer(object.root_digest, &source_id, &destination_id, None));
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);

    let residency = controller.residency(&object.root_digest).unwrap();
    assert_eq!(
        residency
            .replicas
            .iter()
            .find(|replica| replica.backend == source.identity())
            .unwrap()
            .pins,
        0
    );
    assert_eq!(
        residency
            .replicas
            .iter()
            .find(|replica| replica.backend == destination.identity())
            .unwrap()
            .state,
        ReplicaState::Corrupt
    );
}

#[test]
fn transfer_budget_rejects_unreserved_bytes_before_mutating_residency() {
    let controller = TierResidencyController::with_transfer_budget(1, 3).unwrap();
    let source = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-source")));
    let destination = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-destination")));
    let object = manifest(b"larger-than-budget");
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    block_on(source.put_chunk(object.chunks[0].clone(), b"larger-than-budget".to_vec())).unwrap();
    block_on(controller.register_object(object.clone(), source.identity())).unwrap();
    assert_eq!(
        block_on(controller.transfer(
            object.root_digest,
            &source.identity(),
            &destination.identity(),
            None,
        )),
        Err(TierRuntimeError::TransferBackpressure)
    );
    assert_eq!(
        controller
            .residency(&object.root_digest)
            .unwrap()
            .replicas
            .len(),
        1
    );
}

#[test]
fn lookahead_and_replica_policies_are_deterministic_and_bounded() {
    let context = TierPlanningContext {
        requested_tiers: vec![StorageTier::Cpu],
        lookahead: vec![[1; 32], [2; 32], [3; 32]],
        eviction_candidates: Vec::new(),
        available_replicas: 1,
    };
    let lookahead = LookaheadPrefetchPolicy { max_objects: 2 }
        .plan(&context)
        .unwrap();
    assert_eq!(lookahead.prefetch, vec![[1; 32], [2; 32]]);

    let replicas = BudgetedReplicaPolicy {
        minimum_replicas: 3,
        preferred_tiers: vec![StorageTier::Nvme, StorageTier::ObjectStore],
    }
    .plan(&context)
    .unwrap();
    assert_eq!(
        replicas.destinations,
        vec![StorageTier::Nvme, StorageTier::ObjectStore]
    );
    assert_eq!(
        BudgetedReplicaPolicy {
            minimum_replicas: 4,
            preferred_tiers: vec![StorageTier::Nvme],
        }
        .plan(&context),
        Err(TierPolicyError::ReplicaBudgetUnsatisfied)
    );
}

#[test]
fn evicting_one_object_keeps_content_addressed_chunks_shared_by_another() {
    let controller = TierResidencyController::default();
    let source = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-source")));
    let destination = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-destination")));
    let first = manifest(b"shared-chunk");
    let second = TierObjectManifest::new(
        TierObjectDomain::Artifact,
        "artifact-with-same-chunk",
        Generation(4),
        Revision(8),
        [9; 32],
        vec![ChunkDescriptor::for_bytes(0, 0, b"shared-chunk")],
    )
    .unwrap();
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    block_on(source.put_chunk(first.chunks[0].clone(), b"shared-chunk".to_vec())).unwrap();
    block_on(controller.register_object(first.clone(), source.identity())).unwrap();
    block_on(controller.register_object(second.clone(), source.identity())).unwrap();
    block_on(controller.transfer(
        first.root_digest,
        &source.identity(),
        &destination.identity(),
        None,
    ))
    .unwrap();
    block_on(controller.transfer(
        second.root_digest,
        &source.identity(),
        &destination.identity(),
        None,
    ))
    .unwrap();

    block_on(controller.evict(first.root_digest, &destination.identity())).unwrap();
    assert!(
        block_on(destination.verify_chunk(&second.chunks[0]))
            .unwrap()
            .valid
    );
    assert_eq!(
        controller
            .residency(&second.root_digest)
            .unwrap()
            .available_replicas(),
        2
    );
}
