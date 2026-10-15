//! The staging lifecycle through the public API with the CPU backend (doc 42 rules 3a and 3b):
//! pool, export registry, footprint charging, and the order of reaps. Linux only, feature
//! `shm-backend`. The same lifecycle on a GPU is `tests/staging_cuda.rs` (ignored without one).
#![cfg(all(feature = "shm-backend", target_os = "linux"))]

use ptr_fabric::{
    AcquireRequest, ChargeBook, ExportRegistry, FakeBackend, ImportClaim, PoolLimits, PoolRefusal,
    RegistryRefusal, StagingPool,
};

const UNIT: u64 = 64 * 1024;

fn pool(
    limits: PoolLimits,
) -> (
    StagingPool<FakeBackend>,
    ptr_fabric::FakeCounters,
    ChargeBook,
) {
    let backend = FakeBackend::with_unit(UNIT);
    let counters = backend.counters();
    let book = ChargeBook::new();
    (
        StagingPool::new(backend, Box::new(book.clone()), limits),
        counters,
        book,
    )
}

fn request(producer: u64, consumer: u64, operation: u64, len: u64) -> AcquireRequest {
    AcquireRequest {
        producer_session: producer,
        consumer_session: consumer,
        consumer_lane: 1,
        operation,
        device: 0,
        len,
    }
}

#[test]
fn one_producer_serving_two_consumers_recycles_oldest_exports_first() {
    let mut limits = PoolLimits::small_host();
    limits.max_producer_bytes = 3 * UNIT;
    let (mut pool, counters, book) = pool(limits);
    let mut registry = ExportRegistry::new(16);

    // Producer 1 exports to consumers 10, 11 and 10 again; every consumer is below its cap.
    let mut leases = Vec::new();
    for (consumer, lane) in [(10u64, 1u64), (11, 2), (10, 1)] {
        let mut asked = request(1, consumer, 5, 100);
        asked.consumer_lane = lane;
        let lease = pool.acquire(&asked).expect("lease");
        registry
            .issue(lease.record, lease.slot, 1_000, 0)
            .expect("record");
        leases.push(lease);
    }
    assert_eq!(book.charged(1), 3 * UNIT);
    assert_eq!(book.charged(10), 2 * UNIT);

    // The producer is full. The refusal says how much must go, the plan says who goes.
    let mut asked = request(1, 12, 5, 100);
    asked.consumer_lane = 3;
    let PoolRefusal::ProducerQuota { needed_release, .. } = pool.acquire(&asked).expect_err("full")
    else {
        panic!("the producer quota refuses");
    };
    let plan = pool.recovery_plan(1, needed_release);
    assert_eq!(plan.consumers, vec![10]);
    assert!(plan.covers);

    // The caller recycles the producer and the listed consumer; consumer 11 is left running.
    for session in std::iter::once(plan.producer_session).chain(plan.consumers.clone()) {
        pool.on_reaped(session);
        registry.on_reaped(session);
    }
    assert_eq!(counters.live(), 1, "consumer 11's buffer waits for it");
    assert!(registry.is_empty(), "the producer's records went with it");
    assert!(book.charged(1) <= UNIT);
    pool.on_reaped(11);
    assert_eq!((counters.live(), book.total()), (0, 0));
}

#[test]
fn a_forwarded_locator_finds_no_record() {
    let (mut pool, _, _) = pool(PoolLimits::small_host());
    let mut registry = ExportRegistry::new(4);
    let lease = pool.acquire(&request(1, 2, 9, 4096)).expect("lease");
    registry
        .issue(lease.record, lease.slot, 1_000, 0)
        .expect("record");
    // A third worker presents the locator it was never given.
    let forwarded = ImportClaim {
        producer_session: 1,
        consumer_session: 3,
        operation: 9,
        device: 0,
        size: lease.footprint,
    };
    assert_eq!(
        registry.admit(&forwarded, 0),
        Err(RegistryRefusal::NoRecord)
    );
    // The recorded consumer still can: the refusal consumed nothing.
    let rightful = ImportClaim {
        consumer_session: 2,
        ..forwarded
    };
    assert_eq!(
        registry.admit(&rightful, 0).expect("admitted").slot,
        lease.slot
    );
}

#[test]
fn the_host_wide_device_cap_holds_across_pools_of_one_pool() {
    let mut limits = PoolLimits::small_host();
    limits.default_device_bytes = 2 * UNIT;
    let (mut pool, counters, _) = pool(limits);
    pool.acquire(&request(1, 2, 1, 10)).expect("one");
    pool.acquire(&request(3, 4, 1, 10)).expect("two");
    let refusal = pool
        .acquire(&request(5, 6, 1, 10))
        .expect_err("device full");
    assert!(matches!(
        refusal,
        PoolRefusal::DeviceCap {
            device: 0,
            cap,
            ..
        } if cap == 2 * UNIT
    ));
    assert_eq!(counters.allocated(), 2);
    // Reaping both sessions of a pair frees its bytes, and the cap has room again.
    pool.on_reaped(1);
    pool.on_reaped(2);
    pool.acquire(&request(5, 6, 1, 10)).expect("room again");
}

#[test]
fn a_long_then_short_reuse_never_shows_the_previous_tensor_tail() {
    let (mut pool, _, _) = pool(PoolLimits::small_host());
    let long = pool.acquire(&request(1, 2, 1, UNIT)).expect("long");
    let digest = pool
        .fill(long.slot, &vec![0x5A; UNIT as usize])
        .expect("fill");
    assert_ne!(digest, [0; 32]);
    pool.on_reaped(2);
    // Same footprint (one unit), a much shorter tensor: reused, and zeroed whole.
    let short = pool.acquire(&request(1, 3, 2, 16)).expect("short");
    assert!(short.reused);
    assert_eq!(short.footprint, UNIT);
    let fd = pool.export(short.slot, &short.record).expect("export");
    let file = std::fs::File::from(fd);
    let mut whole = vec![1u8; UNIT as usize];
    std::os::unix::fs::FileExt::read_exact_at(&file, &mut whole, 0).expect("read");
    assert!(whole.iter().all(|byte| *byte == 0));
}
