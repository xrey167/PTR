//! A VMM export crossing a process boundary: the producer creates, fills and exports a staging
//! allocation and sends the descriptor with `SCM_RIGHTS` over the C5b `Channel` to a
//! re-executed child (this test binary), which imports it, copies the tensor, checks the digest
//! and answers over the same channel. Linux only, features `cuda-ipc` and `worker-channel`.
//!
//! The test needs an NVIDIA GPU and driver, so it is `#[ignore]`d; run it on the owner's PC per
//! docs/GPU_HANDOVER.md. The child helper does nothing when run as an ordinary test.
#![cfg(all(
    feature = "cuda-ipc-core",
    feature = "worker-channel",
    target_os = "linux"
))]

use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::process::{Command, Stdio};

use ptr_fabric::{
    import_and_copy, Channel, CudaDriver, CudaError, ExportRecord, ImportExpectation,
    StagingAllocation,
};
use ptr_protocol::content_digest;

const MODE_VAR: &str = "PTR_FABRIC_CUDA_CHILD";
const SOCKET_VAR: &str = "PTR_FABRIC_CUDA_FD";
const LEN: usize = 4096;

fn pattern() -> Vec<u8> {
    (0..LEN).map(|index| (index % 239) as u8 ^ 0xa5).collect()
}

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
    let byte_len = u64_at(36);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&body[44..76]);
    let expected = ImportExpectation {
        byte_len,
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

/// The consumer. Does nothing when run as an ordinary test.
#[test]
fn cuda_child_helper() {
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
    let driver = CudaDriver::open().unwrap_or_else(|error| fail(&format!("driver: {error}")));
    let mut frame = channel
        .recv()
        .unwrap_or_else(|error| fail(&format!("recv: {error}")));
    let (record, mut expected) = decode(&frame.body).unwrap_or_else(|| fail("bad body"));
    let fd = frame
        .descriptors
        .pop()
        .unwrap_or_else(|| fail("no descriptor in the frame"));
    if mode == "forwarded" {
        // A locator copied into another worker's handle: another session id must be refused
        // before any driver call.
        expected.consumer_session = record.consumer_session + 1;
    }
    let reply = match import_and_copy(&driver, fd, &record, &expected) {
        Ok(bytes) => {
            let mut reply = b"ok".to_vec();
            reply.extend(bytes);
            reply
        }
        Err(CudaError::RecordMismatch { field }) => format!("refused:{field}").into_bytes(),
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
            "cuda_child_helper",
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
fn an_export_crosses_a_process_boundary_and_verifies() {
    let reply = run_child("honest");
    assert_eq!(&reply[..2], b"ok");
    assert_eq!(&reply[2..], pattern().as_slice());
}

#[test]
#[ignore = "needs an NVIDIA GPU and driver; run on the owner's PC per docs/GPU_HANDOVER.md"]
fn a_forwarded_record_is_refused_in_the_other_process() {
    let reply = run_child("forwarded");
    assert_eq!(reply, b"refused:consumer_session");
}
