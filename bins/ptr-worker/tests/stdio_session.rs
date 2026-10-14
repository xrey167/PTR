//! Drives the real `ptr-worker` binary over its stdin/stdout.
//!
//! A watchdog thread kills the child after `DEADLINE`, which turns any hang into an EOF and so
//! into a test failure instead of a stuck run.

use std::io;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use ptr_protocol::control::{ControlFrame, IdempotencyKey, MAX_FRAME_BYTES};
use ptr_protocol::negotiation::{Feature, FeatureSet, ProtocolVersion, VersionOffer};
use ptr_worker::transport::StdioTransport;
use ptr_worker::wire::{WorkerCall, WorkerReply};

const DEADLINE: Duration = Duration::from_secs(30);

struct Worker {
    child: Arc<Mutex<Child>>,
    io: StdioTransport<ChildStdout, ChildStdin>,
    done: mpsc::Sender<()>,
}

fn spawn(args: &[&str]) -> Worker {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ptr-worker"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn ptr-worker");
    let stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let child = Arc::new(Mutex::new(child));
    let (done, finished) = mpsc::channel::<()>();
    let watched = Arc::clone(&child);
    thread::spawn(move || {
        if finished.recv_timeout(DEADLINE).is_err() {
            if let Ok(mut child) = watched.lock() {
                let _ = child.kill();
            }
        }
    });
    Worker {
        child,
        io: StdioTransport::new(stdout, stdin),
        done,
    }
}

impl Worker {
    fn send(&mut self, frame: &ControlFrame) {
        self.io.write_frame(frame).expect("send frame");
    }

    /// Several frames in one `write` so the worker reads them together.
    fn send_together(&mut self, frames: &[ControlFrame]) {
        let mut scratch = StdioTransport::new(io::empty(), Vec::new());
        for frame in frames {
            scratch.write_frame(frame).expect("encode");
        }
        let bytes = scratch.into_parts().1;
        self.io.write_raw(&bytes).expect("write");
    }

    fn recv(&mut self) -> ControlFrame {
        self.io
            .read_frame()
            .expect("read frame")
            .expect("worker closed stdout early")
    }

    fn recv_reply(&mut self, expect_seq: u64) -> WorkerReply {
        match self.recv() {
            ControlFrame::Reply { seq, payload } => {
                assert_eq!(seq, expect_seq);
                WorkerReply::from_payload(&payload).expect("reply payload")
            }
            other => panic!("expected Reply, got {other:?}"),
        }
    }

    /// Closes stdin, waits for exit and returns the exit code.
    fn finish(self) -> i32 {
        let Worker { child, io, done } = self;
        drop(io);
        let status = {
            let mut child = child.lock().expect("child lock");
            child.wait().expect("wait")
        };
        let _ = done.send(());
        status.code().expect("exit code (killed by watchdog?)")
    }
}

fn offer() -> VersionOffer {
    VersionOffer {
        supported: vec![ProtocolVersion::new(1, 0, 0), ProtocolVersion::new(1, 1, 0)],
        minimum: ProtocolVersion::new(1, 0, 0),
        features: FeatureSet::all(),
        required: FeatureSet::of(&[Feature::Idempotency]),
    }
}

fn hello() -> ControlFrame {
    ControlFrame::Hello {
        node_id: "test-client".into(),
        offer: offer(),
        nonce: 1,
    }
}

fn call(seq: u64, key: &str, pod: &str, input: &[u8]) -> ControlFrame {
    ControlFrame::Call {
        seq,
        idempotency_key: IdempotencyKey::new(key).expect("key"),
        payload: WorkerCall {
            pod: pod.into(),
            input: input.to_vec(),
        }
        .to_payload()
        .expect("call payload"),
    }
}

fn handshake(worker: &mut Worker) {
    worker.send(&hello());
    match worker.recv() {
        ControlFrame::Hello { node_id, .. } => assert_eq!(node_id, "ptr-worker"),
        other => panic!("expected Hello, got {other:?}"),
    }
    match worker.recv() {
        ControlFrame::Decl { capabilities } => {
            let names: Vec<_> = capabilities.iter().map(|c| c.0.as_str()).collect();
            assert_eq!(names, ["echo", "crc32", "sum", "counter"]);
        }
        other => panic!("expected Decl, got {other:?}"),
    }
    worker.send(&ControlFrame::Decl {
        capabilities: Vec::new(),
    });
}

#[test]
fn full_session_calls_duplicate_cancel_and_bye() {
    let mut w = spawn(&["serve", "--stdio"]);
    handshake(&mut w);

    w.send(&call(1, "k1", "counter", b""));
    assert_eq!(w.recv_reply(1), WorkerReply::Ok(b"1".to_vec()));

    w.send(&call(2, "k2", "sum", b"1 2 3"));
    assert_eq!(w.recv_reply(2), WorkerReply::Ok(b"6".to_vec()));

    // The same call again is answered from the cache: the counter does not move.
    w.send(&call(1, "k1", "counter", b""));
    assert_eq!(w.recv_reply(1), WorkerReply::Ok(b"1".to_vec()));
    w.send(&call(3, "k3", "counter", b""));
    assert_eq!(w.recv_reply(3), WorkerReply::Ok(b"2".to_vec()));

    // An unknown Pod is an error reply, not a session failure.
    w.send(&call(4, "k4", "nope", b""));
    assert_eq!(w.recv_reply(4), WorkerReply::Err("unknown-pod".into()));

    // Call and Cancel arrive in one read, so the call is still queued and never runs.
    w.send_together(&[
        call(5, "k5", "counter", b""),
        ControlFrame::Cancel {
            seq: 5,
            reason: "changed my mind".into(),
        },
    ]);
    match w.recv() {
        ControlFrame::Ack {
            seq,
            idempotency_key,
        } => {
            assert_eq!(seq, 5);
            assert_eq!(idempotency_key.as_str(), "k5");
        }
        other => panic!("expected Ack, got {other:?}"),
    }
    w.send(&call(6, "k6", "counter", b""));
    assert_eq!(w.recv_reply(6), WorkerReply::Ok(b"3".to_vec()));

    w.send(&call(7, "k7", "crc32", b"123456789"));
    assert_eq!(
        w.recv_reply(7),
        WorkerReply::Ok(vec![0xCB, 0xF4, 0x39, 0x26])
    );

    w.send(&ControlFrame::Bye {
        reason: "done".into(),
    });
    assert!(matches!(w.recv(), ControlFrame::Bye { .. }));
    assert_eq!(w.finish(), 0);
}

#[test]
fn stream_end_without_bye_exits_3_and_without_hello_exits_3() {
    let mut w = spawn(&["serve", "--stdio"]);
    handshake(&mut w);
    assert_eq!(w.finish(), 3);
    assert_eq!(spawn(&["serve"]).finish(), 3);
}

#[test]
fn truncated_frame_exits_1() {
    let mut w = spawn(&["serve", "--stdio"]);
    w.io.write_raw(&[0, 0, 0, 40, b'P', b'T']).expect("write");
    assert_eq!(w.finish(), 1);
}

#[test]
fn oversize_length_prefix_exits_1_without_reading_a_body() {
    let mut w = spawn(&["serve", "--stdio"]);
    let prefix = ((MAX_FRAME_BYTES + 1) as u32).to_be_bytes();
    w.io.write_raw(&prefix).expect("write");
    assert_eq!(w.finish(), 1);
}

#[test]
fn illegal_first_frame_exits_1() {
    let mut w = spawn(&["serve", "--stdio"]);
    w.send(&call(1, "k", "echo", b"x"));
    assert_eq!(w.finish(), 1);
}

fn run_plain(args: &[&str]) -> (i32, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_ptr-worker"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("run ptr-worker");
    (
        output.status.code().expect("exit code"),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

#[test]
fn plain_invocation_prints_usage_and_pods_lists_builtins() {
    let (code, out) = run_plain(&[]);
    assert_eq!(code, 0);
    assert!(out.contains("usage:") && !out.contains("scaffold"));

    let (code, out) = run_plain(&["pods"]);
    assert_eq!(code, 0);
    let names: Vec<_> = out
        .lines()
        .filter_map(|line| line.split('\t').next())
        .collect();
    assert_eq!(names, ["echo", "crc32", "sum", "counter"]);

    let (code, out) = run_plain(&["serve", "--tcp"]);
    assert_eq!(code, 2);
    assert!(out.is_empty(), "stdout is reserved for the protocol");
}
