#![cfg(all(feature = "tier-pinned", target_os = "linux"))]

use ptr_storage::{
    ChunkDescriptor, PinnedHostTierBackend, StorageTier, TierBackend, TierBackendId, TierError,
    TierHealth,
};
use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex, MutexGuard};
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

/// RLIMIT_MEMLOCK is a per-process total, so tests that lock memory run one at a time.
static MLOCK_SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    MLOCK_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Soft RLIMIT_MEMLOCK in bytes; `None` means unlimited or unreadable.
fn memlock_limit() -> Option<u64> {
    let limits = std::fs::read_to_string("/proc/self/limits").ok()?;
    let line = limits
        .lines()
        .find(|line| line.starts_with("Max locked memory"))?;
    let soft = line["Max locked memory".len()..]
        .split_whitespace()
        .next()?;
    soft.parse().ok()
}

/// Returns true when the test may lock `needed` bytes; otherwise prints why it is skipped.
/// In CI (`CI` set, as on GitHub runners) a property that this host cannot exercise is a
/// failure, not a skip: the CI job exists to exercise it.
fn in_ci() -> bool {
    std::env::var_os("CI").is_some()
}

fn memlock_allows(test: &str, needed: u64) -> bool {
    match memlock_limit() {
        Some(limit) if limit < needed => {
            assert!(
                !in_ci(),
                "{test}: RLIMIT_MEMLOCK {limit} bytes < {needed} bytes needed in CI"
            );
            eprintln!("SKIP {test}: RLIMIT_MEMLOCK {limit} bytes < {needed} bytes needed");
            false
        }
        _ => true,
    }
}

fn backend(pages: usize) -> PinnedHostTierBackend {
    let probe = PinnedHostTierBackend::new(TierBackendId::from("probe"), 0).unwrap();
    PinnedHostTierBackend::new(TierBackendId::from("pinned"), pages * probe.page_size()).unwrap()
}

fn chunk(bytes: &[u8]) -> ChunkDescriptor {
    ChunkDescriptor::for_bytes(0, 0, bytes)
}

const SMALL_NEED: u64 = 4 * 4096;

#[test]
fn roundtrip_and_capabilities() {
    let _guard = serial();
    if !memlock_allows("roundtrip_and_capabilities", SMALL_NEED) {
        return;
    }
    let tier = backend(4);
    let capabilities = tier.capabilities();
    assert_eq!(capabilities.tier, StorageTier::Cpu);
    assert!(!capabilities.persistent);
    assert!(!capabilities.range_reads);
    assert!(capabilities.conditional_create);
    assert_eq!(tier.identity(), TierBackendId::from("pinned"));

    let bytes = b"pinned payload".to_vec();
    let descriptor = chunk(&bytes);
    let receipt = block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    assert_eq!(receipt.digest, descriptor.digest);
    assert_eq!(receipt.length, bytes.len() as u64);
    assert_eq!(block_on_tier(tier.get_chunk(&descriptor)).unwrap(), bytes);
    assert_eq!(block_on_tier(tier.health()).unwrap(), TierHealth::Healthy);
}

#[test]
fn identical_reput_is_ok_and_forged_bytes_are_refused() {
    let _guard = serial();
    if !memlock_allows(
        "identical_reput_is_ok_and_forged_bytes_are_refused",
        SMALL_NEED,
    ) {
        return;
    }
    let tier = backend(4);
    let bytes = b"immutable".to_vec();
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    let used = tier.used_bytes();
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    assert_eq!(
        tier.used_bytes(),
        used,
        "an identical re-put must not be charged twice"
    );

    // Same digest, same length, different bytes. Validation runs before the conflict check (as in
    // `CpuTierBackend`), so a real caller sees a digest mismatch; `ImmutableConflict` is only
    // reachable by a SHA-256 collision. The stored object must stay untouched either way.
    let mut forged = bytes.clone();
    forged[0] ^= 1;
    let error = block_on_tier(tier.put_chunk(descriptor.clone(), forged)).unwrap_err();
    assert_eq!(error, TierError::DigestMismatch);
    assert_eq!(tier.used_bytes(), used);
    assert_eq!(block_on_tier(tier.get_chunk(&descriptor)).unwrap(), bytes);
}

#[test]
fn invalid_bytes_are_refused_on_put_and_store_nothing() {
    let _guard = serial();
    let tier = backend(4);
    let descriptor = chunk(b"declared");
    let error =
        block_on_tier(tier.put_chunk(descriptor.clone(), b"different".to_vec())).unwrap_err();
    assert!(matches!(
        error,
        TierError::DigestMismatch | TierError::LengthMismatch
    ));
    assert_eq!(tier.used_bytes(), 0);
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
}

#[test]
fn unknown_chunk_is_reported_by_get_verify_and_delete_is_idempotent() {
    let _guard = serial();
    let tier = backend(4);
    let descriptor = chunk(b"never stored");
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
    assert_eq!(
        block_on_tier(tier.verify_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
    block_on_tier(tier.delete_chunk(&descriptor)).unwrap();
    assert_eq!(tier.used_bytes(), 0);
}

#[test]
fn budget_is_charged_with_the_page_rounded_footprint() {
    let _guard = serial();
    if !memlock_allows(
        "budget_is_charged_with_the_page_rounded_footprint",
        SMALL_NEED,
    ) {
        return;
    }
    let tier = backend(4);
    let page = tier.page_size();
    assert_eq!(tier.budget_bytes(), 4 * page);
    assert_eq!(tier.footprint_for(0), page);
    assert_eq!(tier.footprint_for(1), page);
    assert_eq!(tier.footprint_for(page), page);
    assert_eq!(tier.footprint_for(page + 1), 2 * page);

    let one = vec![7u8];
    block_on_tier(tier.put_chunk(chunk(&one), one.clone())).unwrap();
    assert_eq!(tier.used_bytes(), page, "1 byte is charged one page");

    let over = vec![9u8; page + 1];
    block_on_tier(tier.put_chunk(chunk(&over), over.clone())).unwrap();
    assert_eq!(
        tier.used_bytes(),
        3 * page,
        "page + 1 bytes are charged two pages"
    );
}

#[test]
fn put_over_budget_is_refused_and_leaves_budget_and_content_unchanged() {
    let _guard = serial();
    if !memlock_allows("put_over_budget_is_refused", SMALL_NEED) {
        return;
    }
    let tier = backend(2);
    let page = tier.page_size();
    let first = vec![1u8; page];
    block_on_tier(tier.put_chunk(chunk(&first), first.clone())).unwrap();

    let too_big = vec![2u8; page + 1];
    let descriptor = chunk(&too_big);
    let error = block_on_tier(tier.put_chunk(descriptor.clone(), too_big)).unwrap_err();
    assert_eq!(
        error,
        TierError::BudgetExceeded {
            requested: 2 * page as u64,
            available: page as u64,
        }
    );
    assert_eq!(tier.used_bytes(), page);
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
    assert_eq!(
        block_on_tier(tier.get_chunk(&chunk(&first))).unwrap(),
        first
    );
}

#[test]
fn zero_budget_refuses_every_put() {
    let _guard = serial();
    let tier = backend(0);
    let bytes = vec![1u8];
    let error = block_on_tier(tier.put_chunk(chunk(&bytes), bytes)).unwrap_err();
    assert!(matches!(error, TierError::BudgetExceeded { .. }));
    assert_eq!(tier.used_bytes(), 0);
}

#[test]
fn delete_releases_the_budget_and_cycles_do_not_leak() {
    let _guard = serial();
    if !memlock_allows(
        "delete_releases_the_budget_and_cycles_do_not_leak",
        SMALL_NEED,
    ) {
        return;
    }
    let tier = backend(2);
    let page = tier.page_size();
    let bytes = vec![5u8; page];
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    assert_eq!(tier.used_bytes(), page);
    block_on_tier(tier.delete_chunk(&descriptor)).unwrap();
    assert_eq!(tier.used_bytes(), 0);
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );

    for round in 0..200u32 {
        let payload = round.to_le_bytes().repeat(page / 4);
        let descriptor = chunk(&payload);
        block_on_tier(tier.put_chunk(descriptor.clone(), payload.clone())).unwrap();
        assert_eq!(block_on_tier(tier.get_chunk(&descriptor)).unwrap(), payload);
        block_on_tier(tier.delete_chunk(&descriptor)).unwrap();
    }
    assert_eq!(
        tier.used_bytes(),
        0,
        "put/delete cycles must not leak budget"
    );
    // The full budget is still usable after the cycles.
    for byte in [1u8, 2u8] {
        let payload = vec![byte; page];
        block_on_tier(tier.put_chunk(chunk(&payload), payload)).unwrap();
    }
    assert_eq!(tier.used_bytes(), 2 * page);
}

#[test]
fn verify_chunk_reports_digest_length_and_validity() {
    let _guard = serial();
    if !memlock_allows(
        "verify_chunk_reports_digest_length_and_validity",
        SMALL_NEED,
    ) {
        return;
    }
    let tier = backend(4);
    let bytes = b"verify me".to_vec();
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    let report = block_on_tier(tier.verify_chunk(&descriptor)).unwrap();
    assert!(report.valid);
    assert_eq!(report.digest, descriptor.digest);
    assert_eq!(report.length, bytes.len() as u64);

    // A descriptor with the stored digest but another declared length reports invalid.
    let mut wrong = descriptor.clone();
    wrong.length += 1;
    let report = block_on_tier(tier.verify_chunk(&wrong)).unwrap();
    assert!(!report.valid);
    assert_eq!(
        block_on_tier(tier.get_chunk(&wrong)).unwrap_err(),
        TierError::CorruptChunk
    );
}

#[test]
fn empty_chunk_costs_one_page() {
    let _guard = serial();
    if !memlock_allows("empty_chunk_costs_one_page", SMALL_NEED) {
        return;
    }
    let tier = backend(1);
    let descriptor = chunk(b"");
    block_on_tier(tier.put_chunk(descriptor.clone(), Vec::new())).unwrap();
    assert_eq!(tier.used_bytes(), tier.page_size());
    assert!(block_on_tier(tier.get_chunk(&descriptor))
        .unwrap()
        .is_empty());
}

#[test]
fn mlock_refusal_is_a_typed_error_with_no_charge() {
    let _guard = serial();
    let Some(limit) = memlock_limit() else {
        eprintln!("SKIP mlock_refusal_is_a_typed_error_with_no_charge: RLIMIT_MEMLOCK unlimited or unreadable");
        return;
    };
    let probe = backend(0);
    let page = probe.page_size();
    // One chunk that is larger than the whole limit, with a budget that would admit it.
    let length = usize::try_from(limit).unwrap() + page;
    let tier = PinnedHostTierBackend::new(TierBackendId::from("pinned"), 2 * length).unwrap();
    let bytes = vec![3u8; length];
    let descriptor = chunk(&bytes);
    match block_on_tier(tier.put_chunk(descriptor.clone(), bytes)) {
        Err(TierError::MemoryLock(message)) => {
            assert!(message.contains("mlock"), "{message}");
            assert_eq!(
                tier.used_bytes(),
                0,
                "a failed mlock must not charge the budget"
            );
            assert_eq!(
                block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
                TierError::UnknownChunk
            );
        }
        Ok(_) => {
            // CAP_IPC_LOCK (for example root in a container) bypasses RLIMIT_MEMLOCK. CI runs
            // unprivileged, so there the refusal path must be exercised.
            assert!(
                !in_ci(),
                "mlock above RLIMIT_MEMLOCK succeeded in CI: the refusal path was not exercised"
            );
            eprintln!(
                "NOTE mlock_refusal: mlock of {length} bytes succeeded above RLIMIT_MEMLOCK {limit}; \
                 the process is privileged, the refusal path was not exercised"
            );
            block_on_tier(tier.delete_chunk(&descriptor)).unwrap();
            assert_eq!(tier.used_bytes(), 0);
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}
