//! The GPU worker profile through the public API: CUDA initialised first, then the GPU seccomp
//! filter. Linux only, features `cuda-ipc-core` and `worker-confine` (the manifest requires
//! both).
//!
//! The CPU tests run everywhere, also on a host without a driver, where they check the typed
//! "no driver" error of the pre-init. The two GPU tests are `#[ignore]`d: each re-executes this
//! test binary as a child that initialises CUDA, installs the GPU filter and then works on a
//! descriptor the parent sends over a `Channel`. They need an NVIDIA GPU and driver and are run
//! on the owner's PC per docs/GPU_HANDOVER.md section 12. Nothing in them has run on a GPU, in
//! this repository's CI or on the host that wrote them; if one fails there, `strace -f` on the
//! child names the system call the GPU filter refused.
//!
//! The filter itself (every forbidden call refused, `ioctl` allowed on an open descriptor,
//! `open` refused) is tested without a GPU in `tests/worker_confine.rs`.
#![cfg(all(
    feature = "cuda-ipc-core",
    feature = "worker-confine",
    target_os = "linux"
))]

use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::process::{Command, Stdio};

use ptr_fabric::{
    observed_profile, prepare_gpu_worker, probe_profile_calls, Channel, CudaDriver, CudaError,
    ExportRecord, ImportExpectation, ObservedProfile, SeccompProfile, StagingAllocation,
};
use ptr_protocol::content_digest;

const MODE_VAR: &str = "PTR_FABRIC_GPU_CHILD";
const SOCKET_VAR: &str = "PTR_FABRIC_GPU_FD";
const LEN: usize = 4096;

fn pattern() -> Vec<u8> {
    (0..LEN).map(|index| (index % 241) as u8 ^ 0x5a).collect()
}

// ---- CPU tests -------------------------------------------------------------------------------

#[test]
fn the_pre_init_returns_the_typed_no_driver_error_without_a_driver() {
    let before = status_line("Seccomp:");
    match CudaDriver::open() {
        Err(CudaError::NoDriver { .. }) => {
            // Any device ordinal: the driver is looked at first.
            for device in [0, 1, u32::MAX] {
                match prepare_gpu_worker(device) {
                    Err(CudaError::NoDriver { reason }) => assert!(!reason.is_empty()),
                    other => panic!("device {device}: wanted NoDriver, got {other:?}"),
                }
            }
        }
        Ok(_) => {
            // A host with a driver: an ordinal it does not have is a typed refusal, and nothing
            // is retained for it.
            assert!(matches!(
                prepare_gpu_worker(u32::MAX),
                Err(CudaError::NoSuchDevice { .. })
            ));
        }
        Err(other) => panic!("open failed with an unexpected error: {other}"),
    }
    // The pre-init installs no filter: this process is as it was.
    assert_eq!(before, status_line("Seccomp:"));
}

fn status_line(name: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .expect("status")
        .lines()
        .find(|line| line.starts_with(name))
        .unwrap_or_default()
        .to_owned()
}

// ---- GPU tests -------------------------------------------------------------------------------

/// record (36 bytes) then byte_len (8) then the digest (32), all little endian.
fn encode(record: &ExportRecord, byte_len: u64, digest: [u8; 32]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend(record.producer_session.to_le_bytes());
    body.extend(record.consumer_session.to_le_bytes());
    body.extend(record.operation.to_le_bytes());
    body.extend(record.device.to_le_bytes());
    body.extend(record.size.to_le_bytes());
    body.extend(byte_len.to_le_bytes());
    body.extend(digest);
    body
}

fn decode(body: &[u8]) -> Option<(ExportRecord, ImportExpectation)> {
    if body.len() != 36 + 8 + 32 {
        return None;
    }
    let u64_at = |at: usize| u64::from_le_bytes(body[at..at + 8].try_into().expect("8 bytes"));
    let record = ExportRecord {
        producer_session: u64_at(0),
        consumer_session: u64_at(8),
        operation: u64_at(16),
        device: u32::from_le_bytes(body[24..28].try_into().expect("4 bytes")),
        size: u64_at(28),
    };
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&body[44..76]);
    let expected = ImportExpectation {
        byte_len: u64_at(36),
        content_digest: digest,
        consumer_session: record.consumer_session,
        operation: record.operation,
        source_device: record.device,
        consumer_device: record.device,
        locator_alloc_size: record.size,
    };
    Some((record, expected))
}

fn fail(message: &str) -> ! {
    eprintln!("CHILD FAIL: {message}");
    std::process::exit(1);
}

/// The GPU worker. Does nothing when run as an ordinary test.
///
/// The order is the contract under test: the channel is opened and CUDA is initialised
/// **before** the filter; after it the child only receives, imports, copies and answers.
#[test]
fn gpu_child_helper() {
    let Ok(mode) = std::env::var(MODE_VAR) else {
        return;
    };
    let number: i32 = std::env::var(SOCKET_VAR)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or_else(|| fail("no channel descriptor"));
    // SAFETY: the parent passed this descriptor to this process and nothing else owns it.
    let channel = Channel::from_fd(unsafe { OwnedFd::from_raw_fd(number) })
        .unwrap_or_else(|error| fail(&format!("channel: {error}")));

    // Before the filter: driver, context, warm-up.
    let prepared =
        prepare_gpu_worker(0).unwrap_or_else(|error| fail(&format!("prepare_gpu_worker: {error}")));
    if let Err(error) = prepared.vmm_warm_up() {
        fail(&format!("the VMM warm-up failed: {error}"));
    }
    let worker = prepared
        .install_gpu_filter()
        .unwrap_or_else(|error| fail(&format!("install_gpu_filter: {error}")));
    if worker.profile() != SeccompProfile::Gpu || worker.device() != 0 {
        fail("the worker reports another profile or device");
    }

    // After the filter: nothing new may be opened, and the probe shows the GPU profile.
    if std::fs::File::open("/proc/self/status").is_ok() {
        fail("open succeeded under the GPU filter");
    }
    // SAFETY: `socket` takes plain integers; it must fail under the filter.
    if unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) } != -1 {
        fail("socket succeeded under the GPU filter");
    }
    let calls = probe_profile_calls();
    let observed = observed_profile(calls.iter().map(|call| (call.name, call.errno)));
    if observed != ObservedProfile::Gpu {
        fail(&format!("observed profile {observed:?}: {calls:?}"));
    }

    let mut frame = channel
        .recv()
        .unwrap_or_else(|error| fail(&format!("recv: {error}")));
    let (record, mut expected) = decode(&frame.body).unwrap_or_else(|| fail("bad body"));
    let fd = frame
        .descriptors
        .pop()
        .unwrap_or_else(|| fail("no descriptor in the frame"));
    if mode == "other-device" {
        // A device that was not prepared: a context on it would have to open its node.
        expected.consumer_device = 1;
        expected.source_device = 1;
    }
    let reply = match worker.import_and_copy(fd, &record, &expected) {
        Ok(bytes) => {
            let mut reply = b"ok".to_vec();
            reply.extend(bytes);
            reply
        }
        Err(CudaError::Unsupported { what }) => format!("refused:{what}").into_bytes(),
        Err(error) => fail(&format!("import: {error}")),
    };
    channel
        .send(&reply, &[])
        .unwrap_or_else(|error| fail(&format!("send: {error}")));
    std::process::exit(0);
}

fn run_child(mode: &str) -> Vec<u8> {
    let driver = CudaDriver::open().unwrap_or_else(|error| {
        panic!("needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md: {error}")
    });
    let bytes = pattern();
    let mut allocation = StagingAllocation::create(&driver, 0, LEN as u64).expect("create");
    let digest = allocation.fill(&bytes).expect("fill");
    assert_eq!(digest, content_digest(&bytes));
    let export = allocation
        .export(allocation.record_for(1, 2, 3))
        .expect("export");

    let (parent, child_end) = Channel::pair().expect("socket pair");
    let inheritable = rustix::io::dup(child_end.as_fd()).expect("dup");
    let mut child = Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "gpu_child_helper",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(MODE_VAR, mode)
        .env(SOCKET_VAR, inheritable.as_raw_fd().to_string())
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn the child");
    drop(inheritable);
    drop(child_end);

    parent
        .send(
            &encode(&export.record, bytes.len() as u64, digest),
            &[export.fd.as_fd()],
        )
        .expect("send the export");
    let reply = parent.recv().expect("the child's answer").body;
    let status = child.wait().expect("child exit");
    assert!(status.success(), "child status {status:?}");
    // The allocation must outlive the child's import (rule 3b): dropped only here.
    drop(allocation);
    reply
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn a_gpu_filtered_worker_imports_and_copies_a_vmm_export() {
    let reply = run_child("honest");
    assert_eq!(&reply[..2], b"ok");
    assert_eq!(&reply[2..], pattern().as_slice());
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn a_gpu_filtered_worker_refuses_a_device_it_did_not_prepare() {
    let reply = run_child("other-device");
    let text = String::from_utf8_lossy(&reply);
    assert!(
        text.starts_with("refused:") && text.contains("prepared before the filter"),
        "{text}"
    );
}
