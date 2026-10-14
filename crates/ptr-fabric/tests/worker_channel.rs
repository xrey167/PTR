//! The worker data channel through the public API: frames with descriptors over a
//! `SOCK_SEQPACKET` pair. Linux only, feature `worker-channel` (the Cargo manifest requires it).
#![cfg(all(feature = "worker-channel", target_os = "linux"))]

use std::io::IoSlice;
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::{Mutex, MutexGuard};

use ptr_fabric::{
    drain, Channel, ChannelError, DrainEnd, ShmExpectation, ShmExport, ShmMapping,
    MAX_FDS_PER_FRAME, MAX_FRAME_BODY_BYTES,
};
use ptr_protocol::content_digest;
use rustix::fs::{fcntl_setfl, OFlags};
use rustix::io::{fcntl_getfd, FdFlags};
use rustix::net::{sendmsg, SendAncillaryBuffer, SendAncillaryMessage, SendFlags};

static SERIAL: Mutex<()> = Mutex::new(());

/// The tests count this process's descriptors, so every test runs alone.
fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn open_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd").expect("proc fd").count()
}

/// `count` descriptors of distinct throwaway memfds.
fn memfds(count: usize) -> Vec<OwnedFd> {
    (0..count)
        .map(|_| {
            rustix::fs::memfd_create("ptr-channel-test", rustix::fs::MemfdFlags::CLOEXEC)
                .expect("memfd")
        })
        .collect()
}

fn borrowed(fds: &[OwnedFd]) -> Vec<BorrowedFd<'_>> {
    fds.iter().map(AsFd::as_fd).collect()
}

/// A frame sent past the `Channel` guards, as a hostile peer could.
fn send_raw(channel: &Channel, body: &[u8], fds: &[BorrowedFd<'_>]) {
    let mut space = [MaybeUninit::<u8>::uninit(); rustix::cmsg_space!(ScmRights(64))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !fds.is_empty() {
        assert!(control.push(SendAncillaryMessage::ScmRights(fds)));
    }
    let sent = sendmsg(
        channel,
        &[IoSlice::new(body)],
        &mut control,
        SendFlags::NOSIGNAL,
    )
    .expect("raw send");
    assert_eq!(sent, body.len());
}

#[test]
fn a_frame_without_descriptors_round_trips() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    left.send(b"hello worker", &[]).expect("send");
    let frame = right.recv().expect("recv");
    assert_eq!(frame.body, b"hello worker");
    assert!(frame.descriptors.is_empty());
    // And the other way, with a body of exactly the bound.
    let full = vec![7u8; MAX_FRAME_BODY_BYTES];
    right.send(&full, &[]).expect("send max");
    assert_eq!(left.recv().expect("recv max").body, full);
}

#[test]
fn a_sealed_memfd_survives_the_trip_and_maps() {
    let _serial = serial();
    let bytes: Vec<u8> = (0..10_000u32).map(|index| (index % 251) as u8).collect();
    let export = ShmExport::create(&bytes).expect("export");
    let (left, right) = Channel::pair().expect("pair");
    left.send(b"handle", &[export.fd()]).expect("send");
    let mut frame = right.recv().expect("recv");
    assert_eq!(frame.descriptors.len(), 1);
    let expected = ShmExpectation {
        byte_len: bytes.len() as u64,
        offset: 0,
        content_digest: content_digest(&bytes),
    };
    let mapping = ShmMapping::map(frame.descriptors.remove(0), &expected).expect("map");
    assert_eq!(mapping.as_bytes(), bytes.as_slice());
}

#[test]
fn received_descriptors_are_cloexec_and_the_most_allowed_arrive() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    // A memfd opened WITHOUT close-on-exec: the received copy must still have it.
    let plain: Vec<OwnedFd> = (0..MAX_FDS_PER_FRAME)
        .map(|_| {
            rustix::fs::memfd_create("ptr-channel-test", rustix::fs::MemfdFlags::empty())
                .expect("memfd")
        })
        .collect();
    assert!(!fcntl_getfd(&plain[0])
        .expect("flags")
        .contains(FdFlags::CLOEXEC));
    left.send(b"max", &borrowed(&plain)).expect("send");
    let frame = right.recv().expect("recv");
    assert_eq!(frame.descriptors.len(), MAX_FDS_PER_FRAME);
    for fd in &frame.descriptors {
        assert!(fcntl_getfd(fd).expect("flags").contains(FdFlags::CLOEXEC));
    }
}

#[test]
fn more_descriptors_than_allowed_are_refused_on_send_and_nothing_is_sent() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    let fds = memfds(MAX_FDS_PER_FRAME + 1);
    assert_eq!(
        left.send(b"x", &borrowed(&fds)).unwrap_err(),
        ChannelError::TooManyDescriptors {
            count: MAX_FDS_PER_FRAME + 1,
            max: MAX_FDS_PER_FRAME,
        }
    );
    assert_eq!(right.try_recv().unwrap_err(), ChannelError::WouldBlock);
}

#[test]
fn more_descriptors_than_allowed_are_closed_and_refused_on_receive() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    // One over the limit: seen as such, closed by the receiver.
    let fds = memfds(MAX_FDS_PER_FRAME + 1);
    send_raw(&left, b"over", &borrowed(&fds));
    let before = open_descriptors();
    assert_eq!(
        right.recv().unwrap_err(),
        ChannelError::TooManyDescriptors {
            count: MAX_FDS_PER_FRAME + 1,
            max: MAX_FDS_PER_FRAME,
        }
    );
    assert_eq!(open_descriptors(), before, "the received copies are closed");

    // Far over: the control message is truncated; what arrived is closed and the rest is
    // closed by the kernel.
    let many = memfds(40);
    send_raw(&left, b"far over", &borrowed(&many));
    let before = open_descriptors();
    assert_eq!(right.recv().unwrap_err(), ChannelError::ControlTruncated);
    assert_eq!(open_descriptors(), before);
    // The channel is still usable after a refusal.
    left.send(b"next", &[]).expect("send");
    assert_eq!(right.recv().expect("recv").body, b"next");
}

#[test]
fn an_oversize_body_is_refused_on_send_and_truncated_on_receive_with_descriptors_closed() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    let too_large = vec![1u8; MAX_FRAME_BODY_BYTES + 1];
    assert_eq!(
        left.send(&too_large, &[]).unwrap_err(),
        ChannelError::FrameTooLarge {
            len: MAX_FRAME_BODY_BYTES + 1,
            max: MAX_FRAME_BODY_BYTES,
        }
    );
    assert_eq!(left.send(&[], &[]).unwrap_err(), ChannelError::EmptyFrame);
    assert_eq!(right.try_recv().unwrap_err(), ChannelError::WouldBlock);

    // A hostile peer sends one byte more than the bound, with a descriptor.
    let fds = memfds(1);
    send_raw(&left, &too_large, &borrowed(&fds));
    let before = open_descriptors();
    assert_eq!(right.recv().unwrap_err(), ChannelError::BodyTruncated);
    assert_eq!(open_descriptors(), before);
}

#[test]
fn the_peer_closing_is_a_typed_error_on_both_directions() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    left.send(b"last", &[]).expect("send");
    drop(left);
    // What was queued is still delivered, then end of file.
    assert_eq!(right.recv().expect("queued").body, b"last");
    assert_eq!(right.recv().unwrap_err(), ChannelError::PeerClosed);
    assert_eq!(right.try_recv().unwrap_err(), ChannelError::PeerClosed);
    assert_eq!(right.send(b"x", &[]).unwrap_err(), ChannelError::PeerClosed);
}

#[test]
fn only_a_seqpacket_socket_is_accepted() {
    let _serial = serial();
    let (stream, _other) = UnixStream::pair().expect("stream pair");
    assert_eq!(
        Channel::from_fd(OwnedFd::from(stream)).unwrap_err(),
        ChannelError::NotSeqPacket
    );
    let not_a_socket = memfds(1).remove(0);
    assert_eq!(
        Channel::from_fd(not_a_socket).unwrap_err(),
        ChannelError::NotSeqPacket
    );
    let (left, right) = Channel::pair().expect("pair");
    let rewrapped = Channel::from_fd(left.into_fd()).expect("seqpacket accepted");
    rewrapped.send(b"ok", &[]).expect("send");
    assert_eq!(right.recv().expect("recv").body, b"ok");
}

#[test]
fn drain_closes_every_queued_frame_and_descriptor() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    // A pipe whose write end is queued on the channel: it reads end-of-file only once the
    // queued copy is closed too.
    let (reader, writer) = std::io::pipe().expect("pipe");
    let reader = OwnedFd::from(reader);
    let writer = OwnedFd::from(writer);
    fcntl_setfl(&reader, OFlags::NONBLOCK).expect("nonblocking");
    left.send(b"a", &[writer.as_fd()]).expect("send");
    let fds = memfds(MAX_FDS_PER_FRAME);
    left.send(b"b", &borrowed(&fds)).expect("send");
    left.send(b"c", &[]).expect("send");
    // A frame the receive path would refuse is drained as well.
    let many = memfds(40);
    send_raw(&left, b"d", &borrowed(&many));
    send_raw(
        &left,
        &vec![0u8; MAX_FRAME_BODY_BYTES + 1],
        &borrowed(&fds[..2]),
    );
    drop(writer);
    let mut one = [0u8; 1];
    assert_eq!(
        rustix::io::read(&reader, &mut one).unwrap_err(),
        rustix::io::Errno::AGAIN,
        "the queued write end is still open"
    );

    let before = open_descriptors();
    let report = drain(&right).expect("drain");
    assert_eq!(report.frames, 5);
    assert_eq!(report.end, DrainEnd::Empty);
    // 1 + 8 + 0 + what fit of 40 + 2; the kernel closes the part of the 40 that did not fit.
    assert!(report.descriptors >= 1 + MAX_FDS_PER_FRAME + 2);
    assert_eq!(
        open_descriptors(),
        before,
        "nothing the drain received stays open"
    );
    assert_eq!(
        rustix::io::read(&reader, &mut one).expect("read"),
        0,
        "the queued write end is closed: end of file"
    );
    assert_eq!(drain(&right).expect("again").frames, 0);

    // With the peer gone the drain says so.
    drop(left);
    let report = drain(&right).expect("drain closed");
    assert_eq!((report.frames, report.end), (0, DrainEnd::PeerClosed));
}

#[test]
fn drain_of_a_dead_peers_queue_ends_at_end_of_file() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    let fds = memfds(3);
    left.send(b"x", &borrowed(&fds)).expect("send");
    left.send(b"y", &borrowed(&fds[..1])).expect("send");
    drop(left);
    let before = open_descriptors();
    let report = drain(&right).expect("drain");
    assert_eq!(report.frames, 2);
    assert_eq!(report.descriptors, 4);
    assert_eq!(report.end, DrainEnd::PeerClosed);
    assert_eq!(open_descriptors(), before);
}

/// A pipe whose write end is queued on `left` after one empty datagram, plus the pipe's
/// non-blocking read end. The caller's own copy of the write end is dropped.
fn empty_datagram_then_a_queued_pipe_end(left: &Channel) -> OwnedFd {
    let (reader, writer) = std::io::pipe().expect("pipe");
    let reader = OwnedFd::from(reader);
    let writer = OwnedFd::from(writer);
    fcntl_setfl(&reader, OFlags::NONBLOCK).expect("nonblocking");
    send_raw(left, b"", &[]);
    send_raw(left, b"carries the pipe", &[writer.as_fd()]);
    drop(writer);
    let mut one = [0u8; 1];
    assert_eq!(
        rustix::io::read(&reader, &mut one).unwrap_err(),
        rustix::io::Errno::AGAIN,
        "the queued write end keeps the pipe open"
    );
    reader
}

#[test]
fn an_empty_datagram_does_not_hide_the_frames_behind_it_from_drain() {
    let _serial = serial();
    // Variant 1: the hostile peer is still connected.
    let (left, right) = Channel::pair().expect("pair");
    let reader = empty_datagram_then_a_queued_pipe_end(&left);
    let report = drain(&right).expect("drain");
    assert_eq!(report.end, DrainEnd::Empty);
    assert_eq!(
        (report.empty_frames, report.frames, report.descriptors),
        (1, 1, 1)
    );
    let mut one = [0u8; 1];
    assert_eq!(rustix::io::read(&reader, &mut one).expect("eof"), 0);

    // Variant 2: the peer is dropped after sending, so the zero-length datagram is followed
    // by end of file on the socket.
    let (left, right) = Channel::pair().expect("pair");
    let reader = empty_datagram_then_a_queued_pipe_end(&left);
    drop(left);
    let report = drain(&right).expect("drain");
    assert_eq!(report.end, DrainEnd::PeerClosed);
    assert_eq!(
        (report.empty_frames, report.frames, report.descriptors),
        (1, 1, 1)
    );
    assert_eq!(rustix::io::read(&reader, &mut one).expect("eof"), 0);
}

#[test]
fn many_empty_datagrams_in_a_row_do_not_end_the_drain_early() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    for _ in 0..200 {
        send_raw(&left, b"", &[]);
    }
    let fds = memfds(1);
    send_raw(&left, b"last", &borrowed(&fds));
    drop(left);
    let report = drain(&right).expect("drain");
    assert_eq!(
        (report.empty_frames, report.frames, report.descriptors),
        (200, 1, 1)
    );
    assert_eq!(report.end, DrainEnd::PeerClosed);
}

#[test]
fn a_flooding_live_peer_stops_the_drain_at_the_cap() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    for _ in 0..50 {
        left.send(b"x", &[]).expect("send");
    }
    let report = ptr_fabric::drain_with_limit(&right, 10).expect("drain");
    assert_eq!((report.frames, report.end), (10, DrainEnd::Capped));
    let report = drain(&right).expect("rest");
    assert_eq!((report.frames, report.end), (40, DrainEnd::Empty));
}

#[test]
fn an_empty_frame_from_a_live_peer_is_not_peer_closed() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    let fds = memfds(2);
    send_raw(&left, b"", &borrowed(&fds));
    let before = open_descriptors();
    assert_eq!(right.recv().unwrap_err(), ChannelError::EmptyFrame);
    assert_eq!(open_descriptors(), before, "its descriptors are closed");
    send_raw(&left, b"", &[]);
    assert_eq!(right.try_recv().unwrap_err(), ChannelError::EmptyFrame);
    // Once the peer is gone the same zero-length read is end of file.
    drop(left);
    assert_eq!(right.recv().unwrap_err(), ChannelError::PeerClosed);
}

#[test]
fn try_send_reports_a_full_queue_instead_of_blocking() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    let body = vec![3u8; MAX_FRAME_BODY_BYTES];
    let mut sent = 0usize;
    loop {
        match left.try_send(&body, &[]) {
            Ok(()) => sent += 1,
            Err(ChannelError::WouldBlock) => break,
            Err(other) => panic!("unexpected {other}"),
        }
        assert!(sent < 100_000, "the queue never filled");
    }
    assert!(sent > 0);
    // Reading one frame makes room again.
    assert_eq!(right.recv().expect("recv").body, body);
    left.try_send(&body, &[]).expect("room after a read");
}

/// The supervisor sent a request the worker never read, the worker sent a frame carrying a
/// pipe write end, and the worker died: the supervisor's next `recvmsg` reports `ECONNRESET`
/// once, then the queued frame, then end of file.
fn reset_then_a_queued_pipe_end() -> (Channel, OwnedFd) {
    let (supervisor, worker) = Channel::pair().expect("pair");
    let (reader, writer) = std::io::pipe().expect("pipe");
    let reader = OwnedFd::from(reader);
    let writer = OwnedFd::from(writer);
    fcntl_setfl(&reader, OFlags::NONBLOCK).expect("nonblocking");
    supervisor.send(b"req", &[]).expect("request, never read");
    worker.send(b"reply", &[writer.as_fd()]).expect("reply");
    drop(writer);
    drop(worker);
    (supervisor, reader)
}

#[test]
fn a_reset_from_a_peer_that_died_with_unread_data_does_not_end_drain() {
    let _serial = serial();
    let (supervisor, reader) = reset_then_a_queued_pipe_end();
    let report = drain(&supervisor).expect("drain");
    assert_eq!(
        (report.frames, report.descriptors, report.end),
        (1, 1, DrainEnd::PeerClosed)
    );
    let mut one = [0u8; 1];
    assert_eq!(
        rustix::io::read(&reader, &mut one).expect("eof"),
        0,
        "the queued write end was closed"
    );
}

#[test]
fn recv_reads_past_a_reset_to_the_queued_frame_and_then_reports_peer_closed() {
    let _serial = serial();
    let (supervisor, _reader) = reset_then_a_queued_pipe_end();
    let frame = supervisor.recv().expect("the queued frame");
    assert_eq!(
        (frame.body.as_slice(), frame.descriptors.len()),
        (&b"reply"[..], 1)
    );
    assert_eq!(supervisor.recv().unwrap_err(), ChannelError::PeerClosed);
}

#[test]
fn a_queue_of_many_empties_then_an_fd_frame_is_drained_when_the_peer_is_dropped() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    let (reader, writer) = std::io::pipe().expect("pipe");
    let reader = OwnedFd::from(reader);
    let writer = OwnedFd::from(writer);
    fcntl_setfl(&reader, OFlags::NONBLOCK).expect("nonblocking");
    // Many empties (the default send buffer holds about 278 before send blocks), so far more
    // than a short run-of-zeros bound, then the frame that carries the descriptor.
    let mut queued = 0usize;
    while queued < 250 {
        send_raw(&left, b"", &[]);
        queued += 1;
    }
    send_raw(&left, b"fd", &[writer.as_fd()]);
    drop(writer);
    drop(left);
    let report = drain(&right).expect("drain");
    assert_eq!(report.end, DrainEnd::PeerClosed);
    assert_eq!((report.frames, report.descriptors), (1, 1));
    assert_eq!(report.empty_frames, queued);
    let mut one = [0u8; 1];
    assert_eq!(rustix::io::read(&reader, &mut one).expect("eof"), 0);
}

/// A raw non-blocking send, so a full queue shows as `false` instead of blocking the test.
fn try_send_raw(channel: &Channel, body: &[u8], fds: &[BorrowedFd<'_>]) -> bool {
    let mut space = [MaybeUninit::<u8>::uninit(); rustix::cmsg_space!(ScmRights(64))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !fds.is_empty() {
        assert!(control.push(SendAncillaryMessage::ScmRights(fds)));
    }
    match sendmsg(
        channel,
        &[IoSlice::new(body)],
        &mut control,
        SendFlags::NOSIGNAL | SendFlags::DONTWAIT,
    ) {
        Ok(_) => true,
        Err(rustix::io::Errno::AGAIN) => false,
        Err(other) => panic!("raw send: {other}"),
    }
}

#[test]
fn a_peer_with_a_raised_send_buffer_cannot_fake_end_of_file_past_the_old_floor() {
    let _serial = serial();
    let (left, right) = Channel::pair().expect("pair");
    // The peer raises its send buffer as far as this host lets it (the kernel clamps the
    // request to `net.core.wmem_max`, which is then doubled).
    let _ = rustix::net::sockopt::set_socket_send_buffer_size(&left, 64 << 20);
    let granted = rustix::net::sockopt::socket_send_buffer_size(&left).expect("sndbuf");
    let (reader, writer) = std::io::pipe().expect("pipe");
    let reader = OwnedFd::from(reader);
    let writer = OwnedFd::from(writer);
    fcntl_setfl(&reader, OFlags::NONBLOCK).expect("nonblocking");

    let mut queued = 0usize;
    while queued < 1500 && try_send_raw(&left, b"", &[]) {
        queued += 1;
    }
    if queued < 1500 {
        // The queue is full: make room for the frame that carries the descriptor.
        assert_eq!(right.try_recv().unwrap_err(), ChannelError::EmptyFrame);
        queued -= 1;
    }
    assert!(try_send_raw(&left, b"fd", &[writer.as_fd()]));
    drop(writer);
    drop(left);
    // What this host allows: 1500 if its limits grant it, else as many as fit. The old floor
    // was 896 datagrams; the point is only demonstrated when more than that were queued.
    eprintln!("send buffer granted {granted} bytes, {queued} empties queued");

    let report = drain(&right).expect("drain");
    assert_eq!((report.frames, report.descriptors), (1, 1));
    assert_eq!(report.empty_frames, queued);
    // Either the run of zeros was proven long enough, or the drain refused to claim end of
    // file; it never reports `PeerClosed` before the fd frame was received (frames == 1).
    assert!(matches!(
        report.end,
        DrainEnd::PeerClosed | DrainEnd::Capped
    ));
    let mut one = [0u8; 1];
    assert_eq!(rustix::io::read(&reader, &mut one).expect("eof"), 0);
}
