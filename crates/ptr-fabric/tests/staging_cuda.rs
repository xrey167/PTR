//! The staging lifecycle with the CUDA backend (doc 42 rules 3a and 3b). Linux only, feature
//! `cuda-ipc-core`.
//!
//! The CPU test runs everywhere and checks that the backend refuses cleanly without a driver.
//! The GPU tests are `#[ignore]`d: they need an NVIDIA GPU and driver and are run on the owner's
//! PC per docs/GPU_HANDOVER.md. Nothing in them has run in this repository's CI or on the host
//! that wrote them.
#![cfg(all(feature = "cuda-ipc-core", target_os = "linux"))]

use ptr_fabric::{
    audit_export_padding, import_and_copy, AcquireRequest, ChargeBook, CudaBackend, CudaDriver,
    CudaError, ImportExpectation, PoolLimits, PoolRefusal, StagingPool,
};
use ptr_protocol::content_digest;

const GPU_ONLY: &str =
    "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md";

fn gpu_pool(limits: PoolLimits) -> (StagingPool<CudaBackend>, ChargeBook, CudaDriver) {
    let backend = CudaBackend::open().expect("a CUDA driver");
    let driver = CudaDriver::open().expect("a CUDA driver");
    let book = ChargeBook::new();
    (
        StagingPool::new(backend, Box::new(book.clone()), limits),
        book,
        driver,
    )
}

fn request(producer: u64, consumer: u64, len: u64) -> AcquireRequest {
    AcquireRequest {
        producer_session: producer,
        consumer_session: consumer,
        consumer_lane: 1,
        operation: 3,
        device: 0,
        len,
    }
}

fn expectation(consumer: u64, bytes: &[u8], footprint: u64) -> ImportExpectation {
    ImportExpectation {
        byte_len: bytes.len() as u64,
        content_digest: content_digest(bytes),
        consumer_session: consumer,
        operation: 3,
        source_device: 0,
        consumer_device: 0,
        locator_alloc_size: footprint,
    }
}

#[test]
fn the_backend_does_not_open_without_a_driver() {
    match CudaBackend::open() {
        Ok(_) => {}
        Err(error) => assert!(matches!(error, CudaError::NoDriver { .. }), "{error:?}"),
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn the_pool_lifecycle_runs_on_a_gpu() {
    let _ = GPU_ONLY;
    let (mut pool, book, driver) = gpu_pool(PoolLimits::small_host());
    let tensor = vec![0x37u8; 100_000];

    // A fresh VMM allocation per handle while the consumer lives; the footprint is charged as
    // allocated (granularity-rounded) to both sessions.
    let first = pool
        .acquire(&request(1, 2, tensor.len() as u64))
        .expect("first");
    let second = pool
        .acquire(&request(1, 2, tensor.len() as u64))
        .expect("second");
    assert!(!first.reused && !second.reused);
    assert!(first.footprint >= tensor.len() as u64);
    assert_eq!(book.charged(1), 2 * first.footprint);
    assert_eq!(book.charged(2), 2 * first.footprint);

    let digest = pool.fill(first.slot, &tensor).expect("fill");
    assert_eq!(digest, content_digest(&tensor));
    let fd = pool.export(first.slot, &first.record).expect("export");
    let copy = import_and_copy(
        &driver,
        fd,
        &first.record,
        &expectation(2, &tensor, first.footprint),
    )
    .expect("import and copy");
    assert_eq!(copy, tensor);

    // The consumer is reaped: the buffer is reusable, and a reused buffer is a new, zeroed
    // allocation of the same footprint, so a shorter tensor shows only zero padding.
    let report = pool.on_reaped(2);
    assert_eq!(report.became_reusable.len(), 2);
    let short = vec![0x11u8; 1000];
    let third = pool
        .acquire(&request(1, 3, short.len() as u64))
        .expect("third");
    if third.footprint == first.footprint {
        assert!(third.reused);
    }
    pool.fill(third.slot, &short)
        .expect("fill the short tensor");
    let fd = pool.export(third.slot, &third.record).expect("export");
    let audit = audit_export_padding(
        &driver,
        fd,
        &third.record,
        &expectation(3, &short, third.footprint),
    )
    .expect("audit");
    assert!(
        audit.all_zero,
        "the reused buffer exposed the previous tensor's tail"
    );

    // Freed only after both domains are reaped.
    pool.on_reaped(1);
    pool.on_reaped(3);
    assert_eq!(pool.slot_count(), 0);
    assert_eq!(book.total(), 0);
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn the_device_cap_is_a_typed_refusal_on_a_gpu() {
    let mut limits = PoolLimits::small_host();
    limits.default_device_bytes = 1; // below any granularity: nothing fits
    let (mut pool, book, _driver) = gpu_pool(limits);
    let refusal = pool
        .acquire(&request(1, 2, 4096))
        .expect_err("over the cap");
    assert!(
        matches!(refusal, PoolRefusal::DeviceCap { device: 0, .. }),
        "{refusal:?}"
    );
    assert_eq!(book.total(), 0);
}
