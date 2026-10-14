#![cfg(all(feature = "tier-direct-io", target_os = "linux"))]

use ptr_storage::{
    ChunkDescriptor, FileReadMode, FileReadStats, FileTierBackend, FileTierOptions, TierBackend,
    TierBackendId, TierError,
};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
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

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

/// Unique scratch directory under `base`, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(base: &Path, label: &str) -> Self {
        let path = base.join(format!(
            "ptr-direct-{label}-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn target_tmp() -> &'static Path {
    Path::new(env!("CARGO_TARGET_TMPDIR"))
}

fn direct(root: &Path) -> FileTierBackend {
    FileTierBackend::with_options(
        TierBackendId::from("direct"),
        root,
        FileTierOptions {
            read: FileReadMode::Direct,
        },
    )
    .unwrap()
}

fn payload(length: usize) -> Vec<u8> {
    (0..length).map(|index| (index % 251) as u8).collect()
}

fn chunk(bytes: &[u8]) -> ChunkDescriptor {
    ChunkDescriptor::for_bytes(0, 0, bytes)
}

fn total(stats: FileReadStats) -> u64 {
    stats.direct + stats.fallback
}

#[test]
fn default_constructor_and_default_options_stay_buffered() {
    let scratch = Scratch::new(target_tmp(), "default");
    let buffered = FileTierBackend::new(TierBackendId::from("buffered"), &scratch.0).unwrap();
    let bytes = payload(5000);
    let descriptor = chunk(&bytes);
    block_on_tier(buffered.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    assert_eq!(
        block_on_tier(buffered.get_chunk(&descriptor)).unwrap(),
        bytes
    );
    assert!(
        block_on_tier(buffered.verify_chunk(&descriptor))
            .unwrap()
            .valid
    );
    assert_eq!(buffered.read_stats(), FileReadStats::default());

    assert_eq!(FileTierOptions::default().read, FileReadMode::Buffered);
    let explicit = FileTierBackend::with_options(
        TierBackendId::from("explicit"),
        &scratch.0,
        FileTierOptions::default(),
    )
    .unwrap();
    assert_eq!(
        block_on_tier(explicit.get_chunk(&descriptor)).unwrap(),
        bytes
    );
    assert_eq!(explicit.read_stats(), FileReadStats::default());
}

#[test]
fn direct_mode_reads_exact_bytes_for_unaligned_lengths() {
    let scratch = Scratch::new(target_tmp(), "lengths");
    let tier = direct(&scratch.0);
    let buffered = FileTierBackend::new(TierBackendId::from("buffered"), &scratch.0).unwrap();
    let mut reads = 0;
    for length in [0usize, 1, 4095, 4096, 4097, 8192, 12_345] {
        let bytes = payload(length);
        let descriptor = chunk(&bytes);
        block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
        let got = block_on_tier(tier.get_chunk(&descriptor)).unwrap();
        assert_eq!(got.len(), length, "length {length}");
        assert_eq!(got, bytes, "length {length}");
        assert_eq!(
            block_on_tier(buffered.get_chunk(&descriptor)).unwrap(),
            got,
            "direct and buffered must agree for length {length}"
        );
        let report = block_on_tier(tier.verify_chunk(&descriptor)).unwrap();
        assert!(report.valid, "length {length}");
        assert_eq!(report.length, length as u64);
        reads += 2;
    }
    let stats = tier.read_stats();
    assert_eq!(
        total(stats),
        reads,
        "every direct-mode read is counted: {stats:?}"
    );
    eprintln!(
        "direct-io lengths test on {}: {stats:?}",
        target_tmp().display()
    );
}

#[test]
fn corrupt_file_is_detected_on_the_direct_path() {
    let scratch = Scratch::new(target_tmp(), "corrupt");
    let tier = direct(&scratch.0);
    let bytes = payload(6000);
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();

    let mut flipped = bytes.clone();
    flipped[5999] ^= 0xff;
    std::fs::write(tier.chunk_path(&descriptor), &flipped).unwrap();
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::CorruptChunk
    );
    let report = block_on_tier(tier.verify_chunk(&descriptor)).unwrap();
    assert!(!report.valid);

    // Truncated file: still refused, never returned as valid bytes.
    std::fs::write(tier.chunk_path(&descriptor), &bytes[..4097]).unwrap();
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::CorruptChunk
    );
    // Extended file: also refused.
    let mut longer = bytes.clone();
    longer.extend_from_slice(&[0; 4097]);
    std::fs::write(tier.chunk_path(&descriptor), &longer).unwrap();
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::CorruptChunk
    );
    assert_eq!(total(tier.read_stats()), 4);
}

#[test]
fn unknown_chunk_is_reported_and_is_not_counted() {
    let scratch = Scratch::new(target_tmp(), "unknown");
    let tier = direct(&scratch.0);
    let descriptor = chunk(b"missing");
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
    assert_eq!(
        block_on_tier(tier.verify_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
    assert_eq!(tier.read_stats(), FileReadStats::default());
}

#[test]
fn immutability_and_writes_are_unchanged_in_direct_mode() {
    let scratch = Scratch::new(target_tmp(), "immutable");
    let tier = direct(&scratch.0);
    let bytes = payload(100);
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    let mut forged = bytes.clone();
    forged[0] ^= 1;
    // Content addressing validates first, so forged bytes under an existing digest are refused
    // as a digest mismatch (the same order as the CPU tier); the stored object is untouched.
    assert_eq!(
        block_on_tier(tier.put_chunk(descriptor.clone(), forged)).unwrap_err(),
        TierError::DigestMismatch
    );
    assert_eq!(block_on_tier(tier.get_chunk(&descriptor)).unwrap(), bytes);
    block_on_tier(tier.delete_chunk(&descriptor)).unwrap();
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::UnknownChunk
    );
}

#[test]
fn records_which_path_ran_on_the_target_directory_filesystem() {
    let scratch = Scratch::new(target_tmp(), "which");
    let tier = direct(&scratch.0);
    let bytes = payload(10_000);
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    for _ in 0..3 {
        assert_eq!(block_on_tier(tier.get_chunk(&descriptor)).unwrap(), bytes);
    }
    let stats = tier.read_stats();
    assert_eq!(total(stats), 3);
    // Either path is acceptable; the filesystem decides. The result is recorded for E011.
    let path = if stats.direct == 3 {
        "O_DIRECT"
    } else if stats.fallback == 3 {
        "buffered fallback (filesystem refused O_DIRECT)"
    } else {
        "mixed"
    };
    eprintln!(
        "direct-io path on {}: {path} {stats:?}",
        target_tmp().display()
    );
}

#[test]
fn forced_fallback_on_shared_memory_filesystem() {
    let shm = Path::new("/dev/shm");
    if !shm.is_dir() {
        eprintln!("SKIP forced_fallback_on_shared_memory_filesystem: /dev/shm is missing");
        return;
    }
    let scratch = Scratch::new(shm, "fallback");
    let tier = direct(&scratch.0);
    let bytes = payload(4097);
    let descriptor = chunk(&bytes);
    block_on_tier(tier.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    assert_eq!(block_on_tier(tier.get_chunk(&descriptor)).unwrap(), bytes);
    assert!(block_on_tier(tier.verify_chunk(&descriptor)).unwrap().valid);
    let stats = tier.read_stats();
    assert_eq!(total(stats), 2);
    if stats.direct == 2 {
        eprintln!(
            "NOTE /dev/shm accepted O_DIRECT on this kernel; the fallback path was not forced"
        );
    } else {
        assert_eq!(
            stats.fallback, 2,
            "tmpfs refuses O_DIRECT, both reads fall back: {stats:?}"
        );
    }
    // Corruption is still detected after a fallback read.
    std::fs::write(
        tier.chunk_path(&descriptor),
        payload(4097).iter().map(|b| b ^ 1).collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        block_on_tier(tier.get_chunk(&descriptor)).unwrap_err(),
        TierError::CorruptChunk
    );
}
