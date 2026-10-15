//! The guardian protocol: how the supervisor drives the process that holds the staging buffers
//! (doc 42 rule 3a). Feature `worker-channel`, Linux only.
//!
//! The guardian is a process of the supervisor's own making (`ptr-worker guardian
//! --channel-fd <n>`). It holds a [`StagingBackend`], no model and no worker input: the only
//! descriptor it serves is the channel to the supervisor, and a request that arrives with a
//! descriptor is refused. It answers each request with one reply, strictly in turn:
//!
//! | request | reply |
//! |---|---|
//! | `FootprintFor { device, len }` | `Footprint { bytes }` |
//! | `Allocate { device, len }` | `Allocated { id, footprint }` |
//! | `Fill { id, bytes }` | `Digest { sha256 }` |
//! | `Export { id, record }` | `Exported` with one descriptor |
//! | `Scrub { id, len }` | `Ok` |
//! | `Release { id }` | `Ok` |
//! | `Ping` | `Ok` |
//!
//! Anything wrong is `Error { reason }`; the guardian keeps serving. When the supervisor's end
//! closes, the guardian frees everything and exits: a guardian never outlives its supervisor.
//! The supervisor-side stub is [`RemoteBackend`], a [`StagingBackend`] over the channel, so the
//! [`StagingPool`](super::StagingPool) cannot tell a guardian from an in-process backend.

use std::collections::BTreeMap;
use std::fmt;
use std::os::fd::{AsFd, OwnedFd};
use std::time::Duration;

use super::backend::{BackendError, StagingBackend};
use crate::{Channel, ChannelError, ExportRecord, MAX_FRAME_BODY_BYTES};

const REQ_FOOTPRINT: u8 = 1;
const REQ_ALLOCATE: u8 = 2;
const REQ_FILL: u8 = 3;
const REQ_EXPORT: u8 = 4;
const REQ_SCRUB: u8 = 5;
const REQ_RELEASE: u8 = 6;
const REQ_PING: u8 = 7;

const REP_OK: u8 = 0x10;
const REP_FOOTPRINT: u8 = 0x11;
const REP_ALLOCATED: u8 = 0x12;
const REP_DIGEST: u8 = 0x13;
const REP_EXPORTED: u8 = 0x14;
const REP_ERROR: u8 = 0x1F;

/// The most bytes one `Fill` can carry: what is left of a frame after the request header.
pub const MAX_FILL_BYTES: usize = MAX_FRAME_BODY_BYTES - 9;

/// The bounds of a guardian.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuardianLimits {
    /// Buffers the guardian holds at once; an `Allocate` beyond it is refused.
    pub max_allocations: usize,
}

impl Default for GuardianLimits {
    fn default() -> Self {
        Self {
            max_allocations: 1024,
        }
    }
}

/// Why a guardian stopped serving.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardianExit {
    /// The supervisor closed its end.
    SupervisorClosed,
}

/// What a finished guardian run did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuardianSummary {
    pub exit: GuardianExit,
    /// Requests answered.
    pub requests: u64,
    /// Buffers freed at exit.
    pub freed_at_exit: usize,
}

/// Why the guardian loop failed (not a refused request: those are `Error` replies).
#[derive(Debug)]
pub enum GuardianError {
    Channel(ChannelError),
}

impl fmt::Display for GuardianError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Channel(error) => write!(formatter, "guardian channel: {error}"),
        }
    }
}

impl std::error::Error for GuardianError {}

impl From<ChannelError> for GuardianError {
    fn from(error: ChannelError) -> Self {
        Self::Channel(error)
    }
}

// --- encoding ---------------------------------------------------------------------------

fn put_u32(body: &mut Vec<u8>, value: u32) {
    body.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(body: &mut Vec<u8>, value: u64) {
    body.extend_from_slice(&value.to_le_bytes());
}

/// A cursor over a received body.
struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(rest: &'a [u8]) -> Self {
        Self { rest }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        if self.rest.len() < count {
            return None;
        }
        let (head, tail) = self.rest.split_at(count);
        self.rest = tail;
        Some(head)
    }

    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|bytes| bytes[0])
    }

    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u64::from_le_bytes)
    }

    fn rest(self) -> &'a [u8] {
        self.rest
    }

    fn finished(&self) -> bool {
        self.rest.is_empty()
    }
}

fn error_reply(reason: &str) -> Vec<u8> {
    let mut body = vec![REP_ERROR];
    let text = reason.as_bytes();
    let keep = text.len().min(1024);
    body.extend_from_slice(&text[..keep]);
    body
}

// --- the guardian side ------------------------------------------------------------------

/// What one request produced: the reply body and the descriptor that rides with it, if any.
struct Answer {
    body: Vec<u8>,
    descriptor: Option<OwnedFd>,
}

impl Answer {
    fn body(body: Vec<u8>) -> Self {
        Self {
            body,
            descriptor: None,
        }
    }

    fn refused(reason: &str) -> Self {
        Self::body(error_reply(reason))
    }

    fn ok() -> Self {
        Self::body(vec![REP_OK])
    }
}

fn backend_refused(error: &BackendError) -> Answer {
    Answer::refused(&error.reason)
}

fn answer<B: StagingBackend>(
    backend: &mut B,
    held: &mut BTreeMap<u64, B::Allocation>,
    next_id: &mut u64,
    limits: GuardianLimits,
    body: &[u8],
) -> Answer {
    let mut reader = Reader::new(body);
    let Some(tag) = reader.u8() else {
        return Answer::refused("empty request");
    };
    match tag {
        REQ_PING if reader.finished() => Answer::ok(),
        REQ_FOOTPRINT => {
            let (Some(device), Some(len), true) = (reader.u32(), reader.u64(), reader.finished())
            else {
                return Answer::refused("malformed FootprintFor");
            };
            match backend.footprint_for(device, len) {
                Ok(bytes) => {
                    let mut reply = vec![REP_FOOTPRINT];
                    put_u64(&mut reply, bytes);
                    Answer::body(reply)
                }
                Err(error) => backend_refused(&error),
            }
        }
        REQ_ALLOCATE => {
            let (Some(device), Some(len), true) = (reader.u32(), reader.u64(), reader.finished())
            else {
                return Answer::refused("malformed Allocate");
            };
            if held.len() >= limits.max_allocations {
                return Answer::refused("the guardian holds as many buffers as it may");
            }
            match backend.allocate(device, len) {
                Ok((allocation, footprint)) => {
                    *next_id += 1;
                    let id = *next_id;
                    held.insert(id, allocation);
                    let mut reply = vec![REP_ALLOCATED];
                    put_u64(&mut reply, id);
                    put_u64(&mut reply, footprint);
                    Answer::body(reply)
                }
                Err(error) => backend_refused(&error),
            }
        }
        REQ_FILL => {
            let Some(id) = reader.u64() else {
                return Answer::refused("malformed Fill");
            };
            let bytes = reader.rest();
            let Some(allocation) = held.get_mut(&id) else {
                return Answer::refused("unknown buffer");
            };
            match backend.fill(allocation, bytes) {
                Ok(digest) => {
                    let mut reply = vec![REP_DIGEST];
                    reply.extend_from_slice(&digest);
                    Answer::body(reply)
                }
                Err(error) => backend_refused(&error),
            }
        }
        REQ_EXPORT => {
            let (
                Some(id),
                Some(producer),
                Some(consumer),
                Some(operation),
                Some(device),
                Some(size),
                true,
            ) = (
                reader.u64(),
                reader.u64(),
                reader.u64(),
                reader.u64(),
                reader.u32(),
                reader.u64(),
                reader.finished(),
            )
            else {
                return Answer::refused("malformed Export");
            };
            let Some(allocation) = held.get(&id) else {
                return Answer::refused("unknown buffer");
            };
            let record = ExportRecord {
                producer_session: producer,
                consumer_session: consumer,
                operation,
                device,
                size,
            };
            match backend.export(allocation, &record) {
                Ok(descriptor) => Answer {
                    body: vec![REP_EXPORTED],
                    descriptor: Some(descriptor),
                },
                Err(error) => backend_refused(&error),
            }
        }
        REQ_SCRUB => {
            let (Some(id), Some(len), true) = (reader.u64(), reader.u64(), reader.finished())
            else {
                return Answer::refused("malformed Scrub");
            };
            let Some(allocation) = held.get_mut(&id) else {
                return Answer::refused("unknown buffer");
            };
            match backend.scrub(allocation, len) {
                Ok(()) => Answer::ok(),
                Err(error) => backend_refused(&error),
            }
        }
        REQ_RELEASE => {
            let (Some(id), true) = (reader.u64(), reader.finished()) else {
                return Answer::refused("malformed Release");
            };
            match held.remove(&id) {
                Some(allocation) => match backend.release(allocation) {
                    Ok(()) => Answer::ok(),
                    Err(error) => backend_refused(&error),
                },
                None => Answer::refused("unknown buffer"),
            }
        }
        _ => Answer::refused("unknown request"),
    }
}

/// Serve the supervisor on `channel` until it closes its end, then free every buffer. This is
/// the whole of the guardian: it reads requests from this one channel only, and accepts no
/// descriptor with a request.
pub fn serve_guardian<B: StagingBackend>(
    channel: &Channel,
    mut backend: B,
    limits: GuardianLimits,
) -> Result<GuardianSummary, GuardianError> {
    let mut held: BTreeMap<u64, B::Allocation> = BTreeMap::new();
    let mut next_id = 0u64;
    let mut requests = 0u64;
    loop {
        let frame = match channel.recv() {
            Ok(frame) => frame,
            Err(ChannelError::PeerClosed) => break,
            Err(error) => return Err(error.into()),
        };
        requests += 1;
        let reply = if frame.descriptors.is_empty() {
            answer(&mut backend, &mut held, &mut next_id, limits, &frame.body)
        } else {
            // Dropping the frame closes what came with it.
            Answer::refused("a request carries no descriptor")
        };
        let descriptors: Vec<_> = reply.descriptor.iter().map(AsFd::as_fd).collect();
        match channel.send(&reply.body, &descriptors) {
            Ok(()) => {}
            Err(ChannelError::PeerClosed | ChannelError::ConnectionReset) => break,
            Err(error) => return Err(error.into()),
        }
    }
    let freed_at_exit = held.len();
    for (_, allocation) in std::mem::take(&mut held) {
        let _ = backend.release(allocation);
    }
    Ok(GuardianSummary {
        exit: GuardianExit::SupervisorClosed,
        requests,
        freed_at_exit,
    })
}

// --- the supervisor side ----------------------------------------------------------------

/// A buffer held by a guardian, known to the supervisor by id. Dropping it frees nothing: the
/// guardian frees on [`RemoteBackend::release`] or when it exits.
#[derive(Debug)]
pub struct RemoteAllocation {
    id: u64,
    footprint: u64,
}

impl RemoteAllocation {
    /// The guardian's id of the buffer.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn footprint(&self) -> u64 {
        self.footprint
    }
}

/// A [`StagingBackend`] that forwards every call to a guardian over a channel. Each call waits
/// at most `timeout` for the answer; a guardian that does not answer, or whose channel is
/// closed, makes the call (and every later one) fail.
pub struct RemoteBackend {
    channel: Channel,
    timeout: Duration,
    broken: bool,
}

impl RemoteBackend {
    /// A stub over `channel`, the supervisor's end of the guardian's channel.
    pub fn new(channel: Channel, timeout: Duration) -> Self {
        Self {
            channel,
            timeout,
            broken: false,
        }
    }

    /// A call failed in a way that leaves the request/reply order unknown (no answer, closed
    /// channel): the stub refuses everything afterwards.
    pub fn is_broken(&self) -> bool {
        self.broken
    }

    /// Check that the guardian answers.
    pub fn ping(&mut self) -> Result<(), BackendError> {
        self.call(&[REQ_PING]).map(|_| ())
    }

    fn call(&mut self, request: &[u8]) -> Result<(Vec<u8>, Option<OwnedFd>), BackendError> {
        if self.broken {
            return Err(BackendError::new("the guardian channel is broken"));
        }
        if let Err(error) = self.channel.send(request, &[]) {
            self.broken = true;
            return Err(BackendError::new(format!("guardian send: {error}")));
        }
        match self.channel.wait_readable(Some(self.timeout)) {
            Ok(true) => {}
            Ok(false) => {
                self.broken = true;
                return Err(BackendError::new("the guardian did not answer in time"));
            }
            Err(error) => {
                self.broken = true;
                return Err(BackendError::new(format!("guardian wait: {error}")));
            }
        }
        let mut frame = match self.channel.recv() {
            Ok(frame) => frame,
            Err(error) => {
                self.broken = true;
                return Err(BackendError::new(format!("guardian receive: {error}")));
            }
        };
        let descriptor = frame.descriptors.pop();
        if !frame.descriptors.is_empty() {
            self.broken = true;
            return Err(BackendError::new(
                "the guardian sent more than one descriptor",
            ));
        }
        if frame.body.first() == Some(&REP_ERROR) {
            let reason = String::from_utf8_lossy(&frame.body[1..]).into_owned();
            return Err(BackendError::new(reason));
        }
        Ok((frame.body, descriptor))
    }

    fn expect(
        &mut self,
        request: &[u8],
        tag: u8,
        wants_descriptor: bool,
    ) -> Result<(Vec<u8>, Option<OwnedFd>), BackendError> {
        let (body, descriptor) = self.call(request)?;
        if body.first() != Some(&tag) || descriptor.is_some() != wants_descriptor {
            self.broken = true;
            return Err(BackendError::new("the guardian sent an unexpected reply"));
        }
        Ok((body, descriptor))
    }
}

impl fmt::Debug for RemoteBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteBackend")
            .field("broken", &self.broken)
            .finish()
    }
}

impl StagingBackend for RemoteBackend {
    type Allocation = RemoteAllocation;

    fn footprint_for(&mut self, device: u32, len: u64) -> Result<u64, BackendError> {
        let mut request = vec![REQ_FOOTPRINT];
        put_u32(&mut request, device);
        put_u64(&mut request, len);
        let (body, _) = self.expect(&request, REP_FOOTPRINT, false)?;
        Reader::new(&body[1..])
            .u64()
            .ok_or_else(|| BackendError::new("malformed footprint reply"))
    }

    fn allocate(&mut self, device: u32, len: u64) -> Result<(Self::Allocation, u64), BackendError> {
        let mut request = vec![REQ_ALLOCATE];
        put_u32(&mut request, device);
        put_u64(&mut request, len);
        let (body, _) = self.expect(&request, REP_ALLOCATED, false)?;
        let mut reader = Reader::new(&body[1..]);
        match (reader.u64(), reader.u64()) {
            (Some(id), Some(footprint)) => Ok((RemoteAllocation { id, footprint }, footprint)),
            _ => Err(BackendError::new("malformed allocate reply")),
        }
    }

    fn fill(
        &mut self,
        allocation: &mut Self::Allocation,
        bytes: &[u8],
    ) -> Result<[u8; 32], BackendError> {
        if bytes.len() > MAX_FILL_BYTES {
            return Err(BackendError::new(format!(
                "a fill carries at most {MAX_FILL_BYTES} bytes"
            )));
        }
        let mut request = vec![REQ_FILL];
        put_u64(&mut request, allocation.id);
        request.extend_from_slice(bytes);
        let (body, _) = self.expect(&request, REP_DIGEST, false)?;
        body[1..]
            .try_into()
            .map_err(|_| BackendError::new("malformed digest reply"))
    }

    fn export(
        &mut self,
        allocation: &Self::Allocation,
        record: &ExportRecord,
    ) -> Result<OwnedFd, BackendError> {
        let mut request = vec![REQ_EXPORT];
        put_u64(&mut request, allocation.id);
        put_u64(&mut request, record.producer_session);
        put_u64(&mut request, record.consumer_session);
        put_u64(&mut request, record.operation);
        put_u32(&mut request, record.device);
        put_u64(&mut request, record.size);
        let (_, descriptor) = self.expect(&request, REP_EXPORTED, true)?;
        descriptor.ok_or_else(|| BackendError::new("the export carried no descriptor"))
    }

    fn scrub(&mut self, allocation: &mut Self::Allocation, len: u64) -> Result<(), BackendError> {
        let mut request = vec![REQ_SCRUB];
        put_u64(&mut request, allocation.id);
        put_u64(&mut request, len);
        self.expect(&request, REP_OK, false).map(|_| ())
    }

    fn release(&mut self, allocation: Self::Allocation) -> Result<(), BackendError> {
        let mut request = vec![REQ_RELEASE];
        put_u64(&mut request, allocation.id);
        // An error is reported, not swallowed: the guardian did not confirm the free, so the
        // memory may still be held, and the pool keeps the charges until the guardian is gone.
        self.expect(&request, REP_OK, false).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::OwnedFd;
    use std::thread;

    use super::*;
    use crate::staging::{AcquireRequest, ChargeBook};
    use crate::staging::{FakeBackend, FakeCounters, PoolLimits, StagingPool};

    const UNIT: u64 = 4096;

    fn spawn_guardian(
        limits: GuardianLimits,
    ) -> (
        RemoteBackend,
        FakeCounters,
        thread::JoinHandle<GuardianSummary>,
    ) {
        let (ours, theirs) = Channel::pair().expect("pair");
        let backend = FakeBackend::with_unit(UNIT);
        let counters = backend.counters();
        let handle = thread::spawn(move || {
            serve_guardian(&theirs, backend, limits).expect("guardian serves")
        });
        (
            RemoteBackend::new(ours, Duration::from_secs(5)),
            counters,
            handle,
        )
    }

    fn record(footprint: u64) -> ExportRecord {
        ExportRecord {
            producer_session: 1,
            consumer_session: 2,
            operation: 3,
            device: 0,
            size: footprint,
        }
    }

    #[test]
    fn a_remote_backend_allocates_fills_exports_scrubs_and_releases() {
        let (mut remote, counters, guardian) = spawn_guardian(GuardianLimits::default());
        remote.ping().expect("ping");
        assert_eq!(remote.footprint_for(0, 100), Ok(UNIT));
        let (mut allocation, footprint) = remote.allocate(0, 100).expect("allocate");
        assert_eq!((footprint, counters.live()), (UNIT, 1));
        let digest = remote.fill(&mut allocation, &[5u8; 100]).expect("fill");
        assert_eq!(digest, ptr_protocol::content_digest(&[5u8; 100]));
        let fd: OwnedFd = remote
            .export(&allocation, &record(footprint))
            .expect("export");
        // The descriptor is the buffer itself: the first 100 bytes are the tensor.
        let file = std::fs::File::from(fd);
        let mut head = [0u8; 100];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut head, 0).expect("read");
        assert_eq!(head, [5u8; 100]);
        remote.scrub(&mut allocation, 10).expect("scrub");
        let mut head = [1u8; 100];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut head, 0).expect("read");
        assert_eq!(head, [0u8; 100]);
        remote.release(allocation).expect("release");
        assert_eq!(counters.live(), 0);
        drop(remote);
        let summary = guardian.join().expect("join");
        assert_eq!(summary.exit, GuardianExit::SupervisorClosed);
        assert_eq!(summary.freed_at_exit, 0);
    }

    #[test]
    fn the_guardian_frees_everything_when_the_supervisor_closes() {
        let (mut remote, counters, guardian) = spawn_guardian(GuardianLimits::default());
        let (_a, _) = remote.allocate(0, 1).expect("a");
        let (_b, _) = remote.allocate(0, 1).expect("b");
        assert_eq!(counters.live(), 2);
        drop(remote);
        let summary = guardian.join().expect("join");
        assert_eq!(summary.freed_at_exit, 2);
        assert_eq!(counters.live(), 0);
    }

    #[test]
    fn refusals_come_back_as_errors_and_the_guardian_keeps_serving() {
        let (mut remote, _, guardian) = spawn_guardian(GuardianLimits { max_allocations: 1 });
        assert!(remote.allocate(0, 0).is_err());
        let (mut first, footprint) = remote.allocate(0, 1).expect("first");
        let error = remote.allocate(0, 1).expect_err("limit");
        assert!(error.reason.contains("as many buffers"), "{error}");
        // A wrong record is refused by the backend; the stub is not broken by it.
        let mut wrong = record(footprint);
        wrong.size += 1;
        assert!(remote.export(&first, &wrong).is_err());
        assert!(!remote.is_broken());
        assert!(remote.fill(&mut first, &[0u8; 2]).is_err());
        let oversize = vec![0u8; MAX_FILL_BYTES + 1];
        assert!(remote.fill(&mut first, &oversize).is_err());
        remote.ping().expect("still serving");
        drop(remote);
        guardian.join().expect("join");
    }

    #[test]
    fn a_request_with_a_descriptor_is_refused() {
        let (ours, theirs) = Channel::pair().expect("pair");
        let guardian = thread::spawn(move || {
            serve_guardian(&theirs, FakeBackend::new(), GuardianLimits::default())
        });
        let (extra, _other) = Channel::pair().expect("pair");
        ours.send(&[REQ_PING], &[extra.as_fd()]).expect("send");
        let reply = ours.recv().expect("reply");
        assert_eq!(reply.body.first(), Some(&REP_ERROR));
        drop(ours);
        guardian.join().expect("join").expect("clean");
    }

    #[test]
    fn a_pool_runs_over_a_guardian_like_over_an_in_process_backend() {
        let (remote, counters, guardian) = spawn_guardian(GuardianLimits::default());
        let book = ChargeBook::new();
        let mut pool = StagingPool::new(remote, Box::new(book.clone()), PoolLimits::small_host());
        let request = AcquireRequest {
            producer_session: 1,
            consumer_session: 2,
            consumer_lane: 20,
            operation: 7,
            device: 0,
            len: 100,
        };
        let first = pool.acquire(&request).expect("first");
        let fd = pool.export(first.slot, &first.record).expect("export");
        drop(fd);
        pool.on_reaped(2);
        let mut next = request;
        next.consumer_session = 3;
        let second = pool.acquire(&next).expect("second");
        assert!(second.reused);
        assert_eq!(counters.allocated(), 1);
        assert_eq!(counters.scrubs(), 1);
        pool.on_reaped(1);
        pool.on_reaped(3);
        assert_eq!((counters.live(), book.total()), (0, 0));
        drop(pool);
        guardian.join().expect("join");
    }

    #[test]
    fn a_guardian_that_died_breaks_the_stub() {
        let (ours, theirs) = Channel::pair().expect("pair");
        drop(theirs);
        let mut remote = RemoteBackend::new(ours, Duration::from_millis(200));
        assert!(remote.ping().is_err());
        assert!(remote.is_broken());
        assert!(remote.allocate(0, 1).is_err());
    }

    #[test]
    fn malformed_requests_are_refused_and_the_guardian_keeps_serving() {
        let (ours, theirs) = Channel::pair().expect("pair");
        let guardian = thread::spawn(move || {
            serve_guardian(
                &theirs,
                FakeBackend::with_unit(UNIT),
                GuardianLimits::default(),
            )
        });
        let mut allocate = vec![REQ_ALLOCATE];
        put_u32(&mut allocate, 0);
        put_u64(&mut allocate, 10);
        let mut with_trailing = allocate.clone();
        with_trailing.push(0);
        let mut ping_trailing = vec![REQ_PING];
        ping_trailing.push(1);
        let bad: Vec<Vec<u8>> = vec![
            // Truncated at every field boundary.
            vec![REQ_ALLOCATE],
            allocate[..4].to_vec(),
            allocate[..allocate.len() - 1].to_vec(),
            vec![REQ_FILL, 1, 2],
            vec![REQ_EXPORT, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![REQ_SCRUB, 1],
            vec![REQ_RELEASE],
            vec![REQ_FOOTPRINT, 0, 0],
            // Unknown tags, a reply tag sent as a request.
            vec![0x7e],
            vec![REP_OK],
            vec![0],
            // Trailing bytes after a complete request.
            with_trailing,
            ping_trailing,
        ];
        for body in &bad {
            ours.send(body, &[]).expect("send");
            let reply = ours.recv().expect("reply");
            assert_eq!(
                reply.body.first(),
                Some(&REP_ERROR),
                "{body:?} was not refused"
            );
            assert!(reply.descriptors.is_empty());
        }
        // Still serving, and nothing was allocated by the refused ones.
        ours.send(&[REQ_PING], &[]).expect("send");
        assert_eq!(ours.recv().expect("reply").body, vec![REP_OK]);
        ours.send(&allocate, &[]).expect("send");
        assert_eq!(
            ours.recv().expect("reply").body.first(),
            Some(&REP_ALLOCATED)
        );
        drop(ours);
        let summary = guardian.join().expect("join").expect("clean");
        assert_eq!(summary.freed_at_exit, 1);
    }
}
