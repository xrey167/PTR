//! The worker data channel (doc 42 stage S1): framed messages that carry file descriptors
//! with `SCM_RIGHTS` between the supervisor and a worker process.
//!
//! A frame is one opaque body (the caller encodes the PodWire or control frame) plus
//! `0..=MAX_FDS_PER_FRAME` descriptors. The channel is a connected `SOCK_SEQPACKET` pair, not
//! `SOCK_STREAM`: a sequenced-packet socket keeps message boundaries, so the descriptors of a
//! frame are attached to exactly that frame and a short or cut-off body is reported by the
//! kernel (`MSG_TRUNC`) instead of having to be reassembled from a byte stream, where
//! ancillary data would arrive with an arbitrary slice of the next frame. `std`'s
//! `UnixStream` cannot create such a socket, so the channel owns an `OwnedFd` and builds the
//! pair itself.
//!
//! Receiving is defensive because the peer is a worker that may be compromised:
//!
//! * the body is bounded by [`MAX_FRAME_BODY_BYTES`] and the receive buffer never exceeds it;
//! * descriptors arrive `CLOEXEC` (`MSG_CMSG_CLOEXEC`) and as `OwnedFd`, so every early return
//!   closes them;
//! * a truncated body (`MSG_TRUNC`) or truncated control message (`MSG_CTRUNC`) is a typed
//!   error, and every descriptor that did arrive is closed first (the kernel closes the ones
//!   that did not fit);
//! * more than [`MAX_FDS_PER_FRAME`] descriptors are closed and refused;
//! * an interrupted call (`EINTR`) is retried;
//! * the peer closing its end is [`ChannelError::PeerClosed`], which the supervisor uses as
//!   crash detection.
//!
//! An empty body is refused on send. On receive a zero-length read is end of file only if the
//! peer has hung up (`POLLHUP`); from a peer that is still connected it is the typed
//! [`ChannelError::EmptyFrame`] (a hostile worker can send empty datagrams, and they must not
//! pass for a crash), and the descriptors that came with it are closed.
//!
//! `send` blocks while the peer's queue is full. A supervisor that must not stall on a worker
//! that stopped reading polls the socket for writability and uses [`Channel::try_send`], which
//! returns [`ChannelError::WouldBlock`] instead.
//!
//! [`drain`] receives and closes everything still queued on a channel, so the supervisor can
//! empty the old endpoint of a reaped worker (doc 42 rule 3b) before it frees the buffers the
//! worker's descriptors referenced.

use std::fmt;
use std::io::{IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use rustix::event::{poll, PollFd, PollFlags, Timespec};
use rustix::io::Errno;
use rustix::net::{
    recvmsg, sendmsg, socketpair, sockopt, AddressFamily, RecvAncillaryBuffer,
    RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer, SendAncillaryMessage,
    SendFlags, SocketFlags, SocketType,
};

/// The largest body of one frame (64 KiB). The channel carries control frames; tensors travel
/// by handle, so a larger body is a protocol error, not something to buffer.
pub const MAX_FRAME_BODY_BYTES: usize = 64 * 1024;

/// The most descriptors one frame may carry. A frame holds a handful of exports; the kernel
/// limit is far higher, and a peer that sends more is refused.
pub const MAX_FDS_PER_FRAME: usize = 8;

/// Room for one descriptor more than allowed, so a frame over the limit is seen as such
/// instead of being cut off by the kernel.
const RECV_FD_CAPACITY: usize = MAX_FDS_PER_FRAME + 1;

/// Why a frame could not be sent or received. Every refusal is typed and leaves no descriptor
/// open that the caller did not receive.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChannelError {
    /// The body is larger than [`MAX_FRAME_BODY_BYTES`]; nothing was sent or allocated.
    FrameTooLarge { len: usize, max: usize },
    /// An empty body: refused on send, because a zero-length datagram cannot be told from end of
    /// file by a receiver that is not polling; received from a peer that is still connected, it
    /// is a protocol violation (every descriptor that came with it is closed).
    EmptyFrame,
    /// More than [`MAX_FDS_PER_FRAME`] descriptors (sent, or received and closed).
    TooManyDescriptors { count: usize, max: usize },
    /// The body was longer than the receive bound and was cut off (`MSG_TRUNC`); every
    /// descriptor that arrived was closed.
    BodyTruncated,
    /// The descriptors did not fit the control message (`MSG_CTRUNC`); every descriptor that
    /// arrived was closed.
    ControlTruncated,
    /// The peer closed its end, or is gone (`EPIPE`, end of file).
    PeerClosed,
    /// The one-time `ECONNRESET` a `SOCK_SEQPACKET` socket reports on receive when the peer
    /// closed while data it had not read was still in its own queue. Datagrams the peer sent
    /// earlier are still queued here and follow it. [`Channel::recv`] and
    /// [`Channel::try_recv`] read past it and never return it; [`drain`] counts it against its
    /// cap and goes on. It is its own variant so that no path can mistake it for end of file;
    /// the public calls of this module do not return it.
    ConnectionReset,
    /// Nothing is queued (receive) or the peer's queue is full (send), and the call did not
    /// block (`EAGAIN`).
    WouldBlock,
    /// The descriptor is not a connected `SOCK_SEQPACKET` socket.
    NotSeqPacket,
    /// An operating-system call failed; the text is the OS error.
    Io(String),
}

impl fmt::Display for ChannelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameTooLarge { len, max } => {
                write!(
                    formatter,
                    "frame body of {len} bytes exceeds the bound of {max}"
                )
            }
            Self::EmptyFrame => write!(formatter, "an empty frame body is not allowed"),
            Self::TooManyDescriptors { count, max } => {
                write!(formatter, "{count} descriptors exceed the bound of {max}")
            }
            Self::BodyTruncated => write!(formatter, "the frame body was truncated"),
            Self::ControlTruncated => write!(formatter, "the descriptor message was truncated"),
            Self::PeerClosed => write!(formatter, "the peer closed the channel"),
            Self::ConnectionReset => write!(formatter, "the connection was reset"),
            Self::WouldBlock => write!(formatter, "the call would block"),
            Self::NotSeqPacket => write!(formatter, "the socket is not SOCK_SEQPACKET"),
            Self::Io(message) => write!(formatter, "os error: {message}"),
        }
    }
}

impl std::error::Error for ChannelError {}

fn io_error(error: Errno) -> ChannelError {
    match error {
        Errno::PIPE => ChannelError::PeerClosed,
        Errno::CONNRESET => ChannelError::ConnectionReset,
        Errno::AGAIN => ChannelError::WouldBlock,
        other => ChannelError::Io(other.to_string()),
    }
}

/// One received frame: the body and the descriptors that came with it, all `CLOEXEC` and all
/// owned (dropping the frame closes them).
#[derive(Debug)]
pub struct Frame {
    pub body: Vec<u8>,
    pub descriptors: Vec<OwnedFd>,
}

/// What [`drain`] received and closed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrainReport {
    /// Frames with a body or a descriptor, received and discarded (a truncated one counts).
    pub frames: usize,
    /// Empty datagrams (no body, no descriptor) discarded. They are frames too, but a hostile
    /// peer can send them to look like end of file, so they never end the drain.
    pub empty_frames: usize,
    /// Descriptors closed, counting those of truncated and over-limit frames.
    pub descriptors: usize,
    /// How the drain ended.
    pub end: DrainEnd,
}

/// Why [`drain`] stopped.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DrainEnd {
    /// Nothing more is queued and the peer is still connected.
    #[default]
    Empty,
    /// The peer end is closed and the queue is empty.
    PeerClosed,
    /// The receive cap was reached with the peer still sending: the queue was not emptied.
    Capped,
}

/// One end of a framed, descriptor-carrying channel.
#[derive(Debug)]
pub struct Channel {
    fd: OwnedFd,
}

impl Channel {
    /// A connected pair (`SOCK_SEQPACKET`, `CLOEXEC`). The caller clears `CLOEXEC` on the end
    /// it hands to a child when it spawns one.
    pub fn pair() -> Result<(Self, Self), ChannelError> {
        let (left, right) = socketpair(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::CLOEXEC,
            None,
        )
        .map_err(io_error)?;
        Ok((Self { fd: left }, Self { fd: right }))
    }

    /// Wrap an already connected socket, for example the one inherited from the supervisor.
    /// Refuses a descriptor that is not a `SOCK_SEQPACKET` socket, because only that type keeps
    /// frame boundaries.
    pub fn from_fd(fd: OwnedFd) -> Result<Self, ChannelError> {
        match sockopt::socket_type(&fd) {
            Ok(SocketType::SEQPACKET) => Ok(Self { fd }),
            Ok(_) | Err(Errno::NOTSOCK) => Err(ChannelError::NotSeqPacket),
            Err(error) => Err(io_error(error)),
        }
    }

    /// Take over the descriptor an exec'd worker inherited from its supervisor (the number the
    /// supervisor put on the command line). Refuses a descriptor that is not a connected
    /// `SOCK_SEQPACKET` socket, and closes it in that case.
    ///
    /// # Safety
    ///
    /// `raw` must be an open descriptor that nothing else in this process owns or closes: the
    /// channel takes ownership and closes it on drop. A worker calls this once, at start, with
    /// the number the supervisor passed.
    pub unsafe fn from_inherited_fd(raw: std::os::fd::RawFd) -> Result<Self, ChannelError> {
        use std::os::fd::FromRawFd;
        // SAFETY: the caller guarantees `raw` is an open descriptor with no other owner.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        Self::from_fd(fd)
    }

    /// Give up the socket, for example to pass it to a child.
    pub fn into_fd(self) -> OwnedFd {
        self.fd
    }

    /// Send one frame: `body` (1 to [`MAX_FRAME_BODY_BYTES`] bytes) and up to
    /// [`MAX_FDS_PER_FRAME`] descriptors, which the kernel duplicates into the peer; the
    /// caller keeps its own. Refuses before any system call when a bound is exceeded.
    ///
    /// Blocks while the peer's receive queue is full, so a supervisor that must not stall on a
    /// worker that stopped reading uses [`try_send`](Self::try_send) (after `poll` says the
    /// socket is writable) instead.
    pub fn send(&self, body: &[u8], descriptors: &[BorrowedFd<'_>]) -> Result<(), ChannelError> {
        self.send_with(body, descriptors, SendFlags::NOSIGNAL)
    }

    /// Like [`send`](Self::send) but never blocks: a full queue is
    /// [`ChannelError::WouldBlock`] and nothing was sent.
    pub fn try_send(
        &self,
        body: &[u8],
        descriptors: &[BorrowedFd<'_>],
    ) -> Result<(), ChannelError> {
        self.send_with(body, descriptors, SendFlags::NOSIGNAL | SendFlags::DONTWAIT)
    }

    fn send_with(
        &self,
        body: &[u8],
        descriptors: &[BorrowedFd<'_>],
        flags: SendFlags,
    ) -> Result<(), ChannelError> {
        if body.is_empty() {
            return Err(ChannelError::EmptyFrame);
        }
        if body.len() > MAX_FRAME_BODY_BYTES {
            return Err(ChannelError::FrameTooLarge {
                len: body.len(),
                max: MAX_FRAME_BODY_BYTES,
            });
        }
        if descriptors.len() > MAX_FDS_PER_FRAME {
            return Err(ChannelError::TooManyDescriptors {
                count: descriptors.len(),
                max: MAX_FDS_PER_FRAME,
            });
        }
        let mut space =
            [MaybeUninit::<u8>::uninit(); rustix::cmsg_space!(ScmRights(MAX_FDS_PER_FRAME))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        if !descriptors.is_empty() {
            let message = SendAncillaryMessage::ScmRights(descriptors);
            if !control.push(message) {
                return Err(ChannelError::Io("control buffer too small".to_owned()));
            }
        }
        let iov = [IoSlice::new(body)];
        loop {
            match sendmsg(&self.fd, &iov, &mut control, flags) {
                Ok(sent) if sent == body.len() => return Ok(()),
                Ok(sent) => {
                    return Err(ChannelError::Io(format!(
                        "short send of {sent} of {} bytes",
                        body.len()
                    )))
                }
                Err(Errno::INTR) => {}
                // Sending to a peer that is gone.
                Err(Errno::CONNRESET) => return Err(ChannelError::PeerClosed),
                Err(Errno::MSGSIZE) => {
                    return Err(ChannelError::FrameTooLarge {
                        len: body.len(),
                        max: MAX_FRAME_BODY_BYTES,
                    })
                }
                Err(error) => return Err(io_error(error)),
            }
        }
    }

    /// Receive one frame, blocking until one is queued. See the module docs for what is
    /// refused and closed.
    ///
    /// Not lossless after the peer died: a zero-length datagram queued before the close reads
    /// as [`ChannelError::PeerClosed`] (the two cannot be told apart once the peer hung up),
    /// ending the reads while frames may still be queued behind it. The supervisor therefore
    /// uses `recv` for live traffic and crash detection, and [`drain`] to empty a dead peer's
    /// queue.
    pub fn recv(&self) -> Result<Frame, ChannelError> {
        let raw = receive_past_reset(&self.fd, RecvFlags::CMSG_CLOEXEC)?;
        raw.into_frame(&self.fd)
    }

    /// Wait until a frame (or the end of the peer) can be received: `true` when the socket is
    /// readable or hung up, `false` when `timeout` passed first. `None` waits without bound.
    /// A following [`try_recv`](Self::try_recv) does not block; a supervisor uses the pair to
    /// bound how long it waits for a worker.
    pub fn wait_readable(
        &self,
        timeout: Option<std::time::Duration>,
    ) -> Result<bool, ChannelError> {
        let mut fds = [PollFd::new(&self.fd, PollFlags::IN)];
        let deadline = timeout.map(|limit| std::time::Instant::now() + limit);
        loop {
            let remaining =
                deadline.map(|end| end.saturating_duration_since(std::time::Instant::now()));
            let spec = remaining.map(|left| Timespec {
                tv_sec: i64::try_from(left.as_secs()).unwrap_or(i64::MAX),
                tv_nsec: i64::from(left.subsec_nanos()),
            });
            match poll(&mut fds, spec.as_ref()) {
                Ok(0) => return Ok(false),
                Ok(_) => return Ok(true),
                Err(Errno::INTR) => {}
                Err(error) => return Err(io_error(error)),
            }
        }
    }

    /// Has the peer closed its end (`POLLHUP`)? True even while frames it sent are still
    /// queued, so it says "the peer is gone", not "nothing is left to read".
    pub fn peer_hung_up(&self) -> bool {
        peer_hung_up(&self.fd)
    }

    /// Like [`recv`](Self::recv) but returns [`ChannelError::WouldBlock`] when nothing is
    /// queued instead of waiting.
    pub fn try_recv(&self) -> Result<Frame, ChannelError> {
        let raw = receive_past_reset(&self.fd, RecvFlags::CMSG_CLOEXEC | RecvFlags::DONTWAIT)?;
        raw.into_frame(&self.fd)
    }
}

impl AsFd for Channel {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

/// What one `recvmsg` returned, before it is judged.
struct Received {
    body: Vec<u8>,
    descriptors: Vec<OwnedFd>,
    flags: ReturnFlags,
}

impl Received {
    /// Judge a received message. On every error the descriptors are dropped, which closes
    /// them.
    fn into_frame(self, fd: &OwnedFd) -> Result<Frame, ChannelError> {
        if self.flags.contains(ReturnFlags::TRUNC) {
            return Err(ChannelError::BodyTruncated);
        }
        if self.flags.contains(ReturnFlags::CTRUNC) {
            return Err(ChannelError::ControlTruncated);
        }
        if self.descriptors.len() > MAX_FDS_PER_FRAME {
            return Err(ChannelError::TooManyDescriptors {
                count: self.descriptors.len(),
                max: MAX_FDS_PER_FRAME,
            });
        }
        if self.body.is_empty() {
            // End of file, or an empty datagram: told apart by whether the peer hung up. A
            // peer that is still connected sent an empty frame, which is not end of file.
            return Err(if peer_hung_up(fd) {
                ChannelError::PeerClosed
            } else {
                ChannelError::EmptyFrame
            });
        }
        Ok(Frame {
            body: self.body,
            descriptors: self.descriptors,
        })
    }
}

/// What one `recvmsg` returned into a caller's buffer.
struct RawReceive {
    /// Bytes received (and kept in the buffer; a truncated body keeps the first part).
    len: usize,
    descriptors: Vec<OwnedFd>,
    flags: ReturnFlags,
}

/// One `recvmsg` into `buffer` with room for one descriptor more than allowed. Descriptors are
/// collected before the result is judged, so a failed call cannot strand one.
fn receive_into(
    fd: &OwnedFd,
    flags: RecvFlags,
    buffer: &mut [u8],
) -> Result<RawReceive, ChannelError> {
    let mut space = [MaybeUninit::<u8>::uninit(); rustix::cmsg_space!(ScmRights(RECV_FD_CAPACITY))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let result = loop {
        let mut iov = [IoSliceMut::new(buffer)];
        match recvmsg(fd, &mut iov, &mut control, flags) {
            Err(Errno::INTR) => {}
            other => break other,
        }
    };
    let mut descriptors = Vec::new();
    for message in control.drain() {
        if let RecvAncillaryMessage::ScmRights(received) = message {
            descriptors.extend(received);
        }
    }
    let message = result.map_err(io_error)?;
    Ok(RawReceive {
        len: message.bytes,
        descriptors,
        flags: message.flags,
    })
}

/// One `recvmsg` with a bounded, freshly allocated body buffer.
fn receive(fd: &OwnedFd, flags: RecvFlags) -> Result<Received, ChannelError> {
    let mut buffer = vec![0u8; MAX_FRAME_BODY_BYTES];
    let raw = receive_into(fd, flags, &mut buffer)?;
    buffer.truncate(raw.len);
    Ok(Received {
        body: buffer,
        descriptors: raw.descriptors,
        flags: raw.flags,
    })
}

/// `receive`, reading past the one-time `ECONNRESET` a `SOCK_SEQPACKET` socket reports when the
/// peer closed with unread data in its own queue: the datagrams the peer sent before that are
/// still queued here and follow the reset. For `recv` and `try_recv` the reset is therefore not
/// an event of its own; the caller gets those frames, then `PeerClosed`.
fn receive_past_reset(fd: &OwnedFd, flags: RecvFlags) -> Result<Received, ChannelError> {
    match receive(fd, flags) {
        Err(ChannelError::ConnectionReset) => match receive(fd, flags) {
            Err(ChannelError::ConnectionReset) => Err(ChannelError::PeerClosed),
            other => other,
        },
        other => other,
    }
}

/// Receive calls [`drain`] makes at most (frames, empty datagrams, resets and end-of-file reads
/// together) before it gives up with [`DrainEnd::Capped`]. A reaped worker's queue is bounded by
/// the kernel (by the sender's send buffer, about 768 bytes of accounting per datagram), so
/// reaching the cap means the peer is not reaped and is still sending, or the end of file could
/// not be established because `wmem_max` is raised so far that the bound exceeds the cap (see
/// [`drain`]).
pub const DRAIN_MAX_RECEIVES: usize = 1 << 16;

/// Smallest accounting of one queued datagram, in bytes. A datagram is charged to its sender's
/// send buffer at its `skb` size (the `sk_buff` header alone is over 200 bytes, plus the shared
/// info), so even an empty one is charged about 768 bytes (278 fit the default 212992); 256 is
/// a deliberately low figure, so the bound derived from it is high.
const MIN_DATAGRAM_ACCOUNTING: usize = 256;

/// Margin added to the bound on queued datagrams.
const QUEUE_BOUND_SLACK: usize = 64;

/// The most datagrams a peer can have left queued for us, or `None` where that cannot be
/// established. Queued datagrams are charged to the *peer's* send buffer. The kernel stores
/// twice the value set with `SO_SNDBUF` (`sk_sndbuf = 2 * min(value, wmem_max)`), so a peer
/// without `CAP_NET_ADMIN` (which alone allows `SO_SNDBUFFORCE`) is limited to `2 * wmem_max`
/// bytes, and our own end's `SO_SNDBUF`, which reads back already doubled, is what the default
/// gave both ends. The bound is the larger of the two over the smallest accounting of one
/// datagram (measured: about 766 bytes for an empty one, so 256 is generous), plus a margin.
///
/// `None` when either figure cannot be read, or when the bound reaches [`DRAIN_MAX_RECEIVES`]
/// (a run of zeros could then never prove end of file): the drain then reads on to the cap and
/// reports [`DrainEnd::Capped`].
///
/// Residual: a peer holding `CAP_NET_ADMIN` could exceed the bound with `SO_SNDBUFFORCE`.
/// Workers run without that capability (the supervisor drops it when it spawns them), and the
/// confined worker has no `setsockopt` at all.
///
/// `SO_MEMINFO`'s receive allocation cannot answer this: on AF_UNIX a queued datagram is
/// charged to the sender, so the receiver's own figure is zero while datagrams are queued.
fn queue_bound_from(own_sndbuf: Option<usize>, wmem_max: Option<usize>) -> Option<usize> {
    let bytes = own_sndbuf?.max(wmem_max?.saturating_mul(2));
    let bound = (bytes / MIN_DATAGRAM_ACCOUNTING).saturating_add(QUEUE_BOUND_SLACK);
    (bound < DRAIN_MAX_RECEIVES).then_some(bound)
}

fn queue_bound(fd: &OwnedFd) -> Option<usize> {
    let own = sockopt::socket_send_buffer_size(fd).ok();
    let wmem_max = std::fs::read_to_string("/proc/sys/net/core/wmem_max")
        .ok()
        .and_then(|text| text.trim().parse::<usize>().ok());
    queue_bound_from(own, wmem_max)
}

/// Like [`drain`] with a chosen cap on the receive calls.
pub fn drain_with_limit(
    channel: &Channel,
    max_receives: usize,
) -> Result<DrainReport, ChannelError> {
    let mut report = DrainReport::default();
    // Zero-length reads since the last frame, taken after the peer hung up: each is an empty
    // datagram or the end of file, and only a run longer than any queue can be is the latter.
    let mut hung_up_zeros = 0usize;
    let bound = queue_bound(&channel.fd);
    // One buffer for every receive: the bodies are discarded. A body longer than the buffer
    // comes back truncated (`MSG_TRUNC`), which still has its descriptors closed.
    let mut buffer = vec![0u8; MAX_FRAME_BODY_BYTES];
    for _ in 0..max_receives {
        match receive_into(
            &channel.fd,
            RecvFlags::CMSG_CLOEXEC | RecvFlags::DONTWAIT,
            &mut buffer,
        ) {
            Ok(raw) if raw.len == 0 && raw.descriptors.is_empty() => {
                if !peer_hung_up(&channel.fd) {
                    // The peer is live, so this is a real empty datagram.
                    report.empty_frames += 1;
                } else {
                    hung_up_zeros += 1;
                    if bound.is_some_and(|limit| hung_up_zeros > limit) {
                        // More zero reads in a row than datagrams could have been queued:
                        // end of file. (Without a bound the drain reads on to the cap.)
                        report.end = DrainEnd::PeerClosed;
                        return Ok(report);
                    }
                }
            }
            Ok(raw) => {
                // The zeros before a real frame were empty datagrams, not end of file.
                report.empty_frames += std::mem::take(&mut hung_up_zeros);
                report.frames += 1;
                report.descriptors += raw.descriptors.len();
                // Dropping `raw` closes the descriptors.
            }
            // The one-time reset of a peer that closed with unread data; what the peer sent
            // before is still queued and follows. It counts against the cap and nothing else.
            Err(ChannelError::ConnectionReset) => {}
            Err(ChannelError::WouldBlock) => {
                report.empty_frames += hung_up_zeros;
                report.end = DrainEnd::Empty;
                return Ok(report);
            }
            Err(ChannelError::PeerClosed) => {
                report.end = DrainEnd::PeerClosed;
                return Ok(report);
            }
            Err(error) => return Err(error),
        }
    }
    report.end = DrainEnd::Capped;
    Ok(report)
}

/// Receive and close every frame and descriptor queued on `channel`, without blocking, until
/// nothing is queued, the peer end is closed with the queue empty, or [`DRAIN_MAX_RECEIVES`]
/// receive calls were made. Used by the supervisor on the old endpoint of a reaped worker (doc
/// 42 rule 3b): a worker can send descriptors just before it dies, and a queued descriptor
/// keeps its memory alive outside every isolation domain until it is closed here.
///
/// Precondition: the peer process has been terminated and reaped, so the queue is finite.
/// Neither an empty datagram nor the `ECONNRESET` a closing peer leaves behind ends the drain.
/// After the peer hung up, a zero-length read is an empty datagram or end of file, and the two
/// cannot be told apart by a read. End of file is taken only after more consecutive zero reads
/// than the peer's send buffer could have queued datagrams (`queue_bound`: the larger of this
/// end's `SO_SNDBUF` and `net.core.wmem_max`, over a low per-datagram accounting). Where the
/// limit cannot be read the drain keeps reading and ends at the cap, never as `PeerClosed`.
/// Against a live, flooding peer it also stops at the cap with [`DrainEnd::Capped`], and the
/// caller must not treat the queue as emptied.
pub fn drain(channel: &Channel) -> Result<DrainReport, ChannelError> {
    drain_with_limit(channel, DRAIN_MAX_RECEIVES)
}

/// Has the peer closed its end (`POLLHUP`)? True even while datagrams it sent are still queued.
fn peer_hung_up(fd: &OwnedFd) -> bool {
    let mut fds = [PollFd::new(fd, PollFlags::empty())];
    let now = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    loop {
        match poll(&mut fds, Some(&now)) {
            Ok(_) => return fds[0].revents().contains(PollFlags::HUP),
            Err(Errno::INTR) => {}
            // Cannot tell: assume it hung up, the cautious reading for the caller.
            Err(_) => return true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_queue_bound_follows_the_send_buffer_limits() {
        // Defaults: own 212992 (already doubled), wmem_max 212992 (so 425984 for a peer).
        assert_eq!(
            queue_bound_from(Some(212_992), Some(212_992)),
            Some(425_984 / 256 + 64)
        );
        // Our own end larger than twice wmem_max (a raised default): the larger wins.
        assert_eq!(
            queue_bound_from(Some(1_000_000), Some(100_000)),
            Some(1_000_000 / 256 + 64)
        );
    }

    #[test]
    fn an_unknown_limit_or_a_bound_beyond_the_cap_gives_no_bound() {
        assert_eq!(queue_bound_from(None, Some(212_992)), None);
        assert_eq!(queue_bound_from(Some(212_992), None), None);
        assert_eq!(queue_bound_from(None, None), None);
        assert_eq!(queue_bound_from(Some(212_992), Some(usize::MAX)), None);
        assert_eq!(
            queue_bound_from(Some(DRAIN_MAX_RECEIVES * MIN_DATAGRAM_ACCOUNTING), Some(0)),
            None
        );
    }
}
