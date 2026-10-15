//! The staging pool (doc 42 rules 3, 3a and 3b): the supervisor's bookkeeping of the staging
//! buffers its guardian holds.
//!
//! * **Fresh while the importer lives.** Buffers form a pool per [`PairKey`] (the producer
//!   session and the consumer *lane*, the stable identity of the consumer stage). Every
//!   [`StagingPool::acquire`] gets a fresh buffer while the consumer session that imported an
//!   earlier buffer of the pair is alive, because a consumer that lies about `Consumed` may still
//!   map it. A buffer is handed out again only for the same pair, only after that consumer has
//!   been reaped ([`StagingPool::on_reaped`]), and only after the backend zeroed all of it
//!   (padding included). Sessions are never reissued, so the pair is keyed by the lane: a
//!   recycled consumer is a new session on the same lane.
//! * **Freed only after both reaps.** A buffer is released to the backend once the producer's
//!   *and* the consumer's session have been reaped since it was exported; until then it counts
//!   against both quotas. A buffer whose consumer is reaped but whose producer lives stays
//!   (reusable); a buffer whose producer is reaped waits for its consumer.
//! * **Charged as allocated.** The footprint the backend reports is charged to the producer and
//!   the consumer through a [`FootprintSink`] (no runtime dependency here) and released when the
//!   buffer is freed, or moved to the new consumer when it is reused.
//! * **Bounded.** Per handle, per pair (bytes and buffers), per producer, per consumer, per
//!   device (the host-wide VRAM cap) and in total; every refusal is a typed [`PoolRefusal`] and
//!   nothing is allocated for a refused request.
//! * **Producer quota recovery.** When a producer's quota refuses a handle,
//!   [`StagingPool::recovery_plan`] names the consumers to recycle, oldest export first, until
//!   the bytes released by recycling them and the producer cover the refused handle. The pool
//!   only plans; the caller recycles.
//!
//! The pool is single-threaded and synchronous, like the host glue that owns it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::os::fd::OwnedFd;

use ptr_protocol::MAX_HANDLE_BYTES;

use super::backend::{BackendError, StagingBackend};
use super::charges::{ChargeRefused, FootprintSink};
use crate::ExportRecord;

/// Most reaped sessions the pool remembers, to refuse an acquire for one of them. The memory
/// is bounded, and forgetting the oldest is safe only because a supervisor never reissues a
/// session id (see `SessionId`): an id that fell out of the list can still never come back as a
/// live session, so a request naming it is stale and would be refused by the host anyway.
const MAX_REMEMBERED_REAPED: usize = 1 << 16;

/// One buffer of the pool.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SlotId(pub u64);

impl fmt::Display for SlotId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "slot-{}", self.0)
    }
}

/// The pool a buffer belongs to: the producer session and the stable identity of the consumer
/// stage (`consumer_lane`), which outlives any one consumer session.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PairKey {
    pub producer_session: u64,
    pub consumer_lane: u64,
}

/// The bounds of a [`StagingPool`]. All bytes are allocated footprints.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolLimits {
    /// The longest tensor of one handle.
    pub max_handle_len: u64,
    /// Buffers in the whole pool, reusable ones included.
    pub max_slots: usize,
    /// Buffers of one pair.
    pub max_pair_slots: usize,
    /// Footprint bytes of one pair.
    pub max_pair_bytes: u64,
    /// Footprint bytes charged to one producer session.
    pub max_producer_bytes: u64,
    /// Footprint bytes charged to one consumer session.
    pub max_consumer_bytes: u64,
    /// The host-wide cap of footprint bytes per device, for a device without an override.
    pub default_device_bytes: u64,
    /// Per-device overrides of the host-wide cap.
    pub device_bytes: BTreeMap<u32, u64>,
}

impl PoolLimits {
    /// Small defaults that suit a test or a small host: 256 buffers, 64 per pair, 1 GiB per
    /// pair, 1 GiB per session, 4 GiB per device, handles up to `MAX_HANDLE_BYTES`.
    pub fn small_host() -> Self {
        Self {
            max_handle_len: MAX_HANDLE_BYTES,
            max_slots: 256,
            max_pair_slots: 64,
            max_pair_bytes: 1 << 30,
            max_producer_bytes: 1 << 30,
            max_consumer_bytes: 1 << 30,
            default_device_bytes: 4 << 30,
            device_bytes: BTreeMap::new(),
        }
    }

    fn device_cap(&self, device: u32) -> u64 {
        self.device_bytes
            .get(&device)
            .copied()
            .unwrap_or(self.default_device_bytes)
    }
}

impl Default for PoolLimits {
    fn default() -> Self {
        Self::small_host()
    }
}

/// What to acquire a staging buffer for: one handle of one operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcquireRequest {
    pub producer_session: u64,
    pub consumer_session: u64,
    /// The stable identity of the consumer stage; the pool is keyed by it.
    pub consumer_lane: u64,
    pub operation: u64,
    pub device: u32,
    /// The tensor length.
    pub len: u64,
}

/// A buffer handed out for one handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Lease {
    pub slot: SlotId,
    /// The record the supervisor issues for this export (the size is the footprint).
    pub record: ExportRecord,
    /// The allocated footprint charged to the producer and the consumer.
    pub footprint: u64,
    /// The buffer is an earlier one, zeroed after its consumer was reaped.
    pub reused: bool,
    /// The logical export time: a counter, so the order of exports is exact.
    pub sequence: u64,
}

/// Why the pool refused a request. Nothing is allocated, charged or changed when it does,
/// except where a variant says otherwise.
#[derive(Debug, Eq, PartialEq)]
pub enum PoolRefusal {
    /// An empty tensor has no buffer.
    EmptyTensor,
    /// The tensor is longer than one handle may be.
    TooLarge { len: u64, max: u64 },
    /// Producer and consumer are the same session.
    SameSession { session: u64 },
    /// A session of the request was reaped already: nothing may be exported to or from it.
    SessionReaped { session: u64 },
    /// The producer's quota would be exceeded. `needed_release` is the number of bytes that
    /// must be released first; [`StagingPool::recovery_plan`] says how.
    ProducerQuota {
        producer_session: u64,
        used: u64,
        cap: u64,
        requested: u64,
        needed_release: u64,
    },
    /// The consumer's quota would be exceeded.
    ConsumerQuota {
        consumer_session: u64,
        used: u64,
        cap: u64,
        requested: u64,
    },
    /// The pair holds too many bytes.
    PairBytes { used: u64, cap: u64, requested: u64 },
    /// The pair holds too many buffers.
    PairSlots { max: usize },
    /// The pool holds too many buffers.
    SlotLimit { max: usize },
    /// The host-wide cap of the device would be exceeded.
    DeviceCap {
        device: u32,
        used: u64,
        cap: u64,
        requested: u64,
    },
    /// The sink refused the charge.
    ChargeRefused(ChargeRefused),
    /// The buffer was filled for this lease already; a lease is filled once.
    AlreadyFilled,
    /// The backend failed.
    Backend(BackendError),
    /// The backend allocated a footprint other than the one it announced.
    FootprintDisagrees { announced: u64, allocated: u64 },
    /// No such buffer.
    UnknownSlot(SlotId),
    /// A field of the record disagrees with the buffer.
    RecordMismatch { field: &'static str },
}

impl fmt::Display for PoolRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PoolRefusal {}

impl From<BackendError> for PoolRefusal {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

/// What one [`StagingPool::on_reaped`] changed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReapReport {
    /// Buffers released to the backend: both their domains are reaped.
    pub freed: Vec<SlotId>,
    /// Buffers whose consumer was reaped while the producer lives: reusable for the pair.
    pub became_reusable: Vec<SlotId>,
    /// Footprint bytes released by the freed buffers.
    pub freed_bytes: u64,
    /// Buffers whose domains are both reaped but whose release the backend failed: their
    /// charges are kept until the backend is gone.
    pub release_failed: Vec<SlotId>,
}

/// The consumers to recycle, with their producer, so that a refused handle fits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveryPlan {
    /// The producer to recycle with them.
    pub producer_session: u64,
    /// The consumer sessions to recycle, the one with the oldest export first. Consumers that
    /// are reaped already are not listed.
    pub consumers: Vec<u64>,
    /// The footprint bytes the recycle releases: every buffer of the producer whose consumer is
    /// reaped or listed.
    pub released_bytes: u64,
    /// `released_bytes` covers what was asked.
    pub covers: bool,
}

/// A read-only view of one buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlotInfo {
    pub key: PairKey,
    pub producer_session: u64,
    pub consumer_session: u64,
    pub device: u32,
    pub len: u64,
    pub footprint: u64,
    pub sequence: u64,
    pub producer_reaped: bool,
    pub consumer_reaped: bool,
}

struct Slot<A> {
    allocation: A,
    key: PairKey,
    consumer_session: u64,
    device: u32,
    len: u64,
    footprint: u64,
    sequence: u64,
    producer_reaped: bool,
    consumer_reaped: bool,
    /// Footprint bytes the backend keeps resident beyond `footprint` (memory a retaining scrub
    /// replaced), charged with the buffer and freed with it.
    extra: u64,
    /// A scrub failed: the buffer is not handed out again, but stays charged and allocated
    /// until both domains are reaped (rule 3b).
    poisoned: bool,
    /// The lease was filled through the pool.
    filled: bool,
}

/// What a buffer was charged, kept past the buffer itself.
#[derive(Clone, Copy)]
struct Charged {
    key: PairKey,
    consumer_session: u64,
    device: u32,
    bytes: u64,
}

impl<A> Slot<A> {
    fn held(&self) -> u64 {
        add(self.footprint, self.extra)
    }

    fn charged(&self) -> Charged {
        Charged {
            key: self.key,
            consumer_session: self.consumer_session,
            device: self.device,
            bytes: self.held(),
        }
    }

    fn info(&self) -> SlotInfo {
        SlotInfo {
            key: self.key,
            producer_session: self.key.producer_session,
            consumer_session: self.consumer_session,
            device: self.device,
            len: self.len,
            footprint: self.footprint,
            sequence: self.sequence,
            producer_reaped: self.producer_reaped,
            consumer_reaped: self.consumer_reaped,
        }
    }

    fn reusable(&self) -> bool {
        self.consumer_reaped && !self.producer_reaped && !self.poisoned
    }
}

fn add(left: u64, right: u64) -> u64 {
    left.saturating_add(right)
}

/// The pool. See the module documentation.
pub struct StagingPool<B: StagingBackend> {
    backend: B,
    sink: Box<dyn FootprintSink>,
    limits: PoolLimits,
    slots: BTreeMap<SlotId, Slot<B::Allocation>>,
    next_slot: u64,
    sequence: u64,
    producer_bytes: BTreeMap<u64, u64>,
    consumer_bytes: BTreeMap<u64, u64>,
    device_bytes: BTreeMap<u32, u64>,
    pair_bytes: BTreeMap<PairKey, u64>,
    reaped: BTreeSet<u64>,
    reaped_order: VecDeque<u64>,
    /// Buffers the backend failed to release: the memory may still be held, so the charges
    /// stay until the backend is gone ([`StagingPool::abandon_all`]).
    stuck: Vec<Charged>,
}

fn decrease<K: Ord + Copy>(map: &mut BTreeMap<K, u64>, key: K, bytes: u64) {
    let left = map.get(&key).copied().unwrap_or(0).saturating_sub(bytes);
    if left == 0 {
        map.remove(&key);
    } else {
        map.insert(key, left);
    }
}

fn increase<K: Ord + Copy>(map: &mut BTreeMap<K, u64>, key: K, bytes: u64) {
    let value = add(map.get(&key).copied().unwrap_or(0), bytes);
    map.insert(key, value);
}

impl<B: StagingBackend> StagingPool<B> {
    /// A pool over `backend`, charging `sink`, bounded by `limits`.
    pub fn new(backend: B, sink: Box<dyn FootprintSink>, limits: PoolLimits) -> Self {
        Self {
            backend,
            sink,
            limits,
            slots: BTreeMap::new(),
            next_slot: 0,
            sequence: 0,
            producer_bytes: BTreeMap::new(),
            consumer_bytes: BTreeMap::new(),
            device_bytes: BTreeMap::new(),
            pair_bytes: BTreeMap::new(),
            reaped: BTreeSet::new(),
            reaped_order: VecDeque::new(),
            stuck: Vec::new(),
        }
    }

    pub fn limits(&self) -> &PoolLimits {
        &self.limits
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Buffers whose release failed and whose charges are kept until the backend is gone.
    pub fn stuck_count(&self) -> usize {
        self.stuck.len()
    }

    /// Buffers in the pool (in use, reusable and waiting for a reap).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// A view of one buffer.
    pub fn slot(&self, slot: SlotId) -> Option<SlotInfo> {
        self.slots.get(&slot).map(Slot::info)
    }

    /// All buffers, oldest export first.
    pub fn slots(&self) -> Vec<(SlotId, SlotInfo)> {
        let mut all: Vec<_> = self
            .slots
            .iter()
            .map(|(id, slot)| (*id, slot.info()))
            .collect();
        all.sort_by_key(|(_, info)| info.sequence);
        all
    }

    /// Footprint bytes charged to `session` as a producer.
    pub fn producer_bytes(&self, session: u64) -> u64 {
        self.producer_bytes.get(&session).copied().unwrap_or(0)
    }

    /// Footprint bytes charged to `session` as a consumer.
    pub fn consumer_bytes(&self, session: u64) -> u64 {
        self.consumer_bytes.get(&session).copied().unwrap_or(0)
    }

    /// Footprint bytes held on `device`.
    pub fn device_bytes(&self, device: u32) -> u64 {
        self.device_bytes.get(&device).copied().unwrap_or(0)
    }

    /// Footprint bytes held by `key`.
    pub fn pair_bytes(&self, key: PairKey) -> u64 {
        self.pair_bytes.get(&key).copied().unwrap_or(0)
    }

    /// Whether `session` was reaped (and is remembered as such).
    pub fn is_reaped(&self, session: u64) -> bool {
        self.reaped.contains(&session)
    }

    /// The sessions that a live buffer still ties to the guardian: producers and consumers of
    /// buffers they have not been reaped for.
    pub fn backed_sessions(&self) -> BTreeSet<u64> {
        let mut sessions = BTreeSet::new();
        for slot in self.slots.values() {
            if !slot.producer_reaped {
                sessions.insert(slot.key.producer_session);
            }
            if !slot.consumer_reaped {
                sessions.insert(slot.consumer_session);
            }
        }
        sessions
    }

    /// Hand out a buffer for one handle: a zeroed earlier buffer of the pair whose consumer was
    /// reaped, else a fresh one. See the module documentation for the bounds.
    pub fn acquire(&mut self, request: &AcquireRequest) -> Result<Lease, PoolRefusal> {
        if request.len == 0 {
            return Err(PoolRefusal::EmptyTensor);
        }
        if request.len > self.limits.max_handle_len {
            return Err(PoolRefusal::TooLarge {
                len: request.len,
                max: self.limits.max_handle_len,
            });
        }
        if request.producer_session == request.consumer_session {
            return Err(PoolRefusal::SameSession {
                session: request.producer_session,
            });
        }
        for session in [request.producer_session, request.consumer_session] {
            if self.reaped.contains(&session) {
                return Err(PoolRefusal::SessionReaped { session });
            }
        }
        let key = PairKey {
            producer_session: request.producer_session,
            consumer_lane: request.consumer_lane,
        };
        let footprint = self
            .backend
            .footprint_for(request.device, request.len)
            .map_err(PoolRefusal::from)?;
        if let Some(id) = self.reusable_slot(key, request.device, footprint) {
            return self.reuse(id, request);
        }
        self.check_fresh(key, request, footprint)?;
        let (allocation, allocated) = self
            .backend
            .allocate(request.device, request.len)
            .map_err(PoolRefusal::from)?;
        if allocated != footprint {
            let _ = self.backend.release(allocation);
            return Err(PoolRefusal::FootprintDisagrees {
                announced: footprint,
                allocated,
            });
        }
        if let Err(refused) = self.sink.charge(request.producer_session, footprint) {
            let _ = self.backend.release(allocation);
            return Err(PoolRefusal::ChargeRefused(refused));
        }
        if let Err(refused) = self.sink.charge(request.consumer_session, footprint) {
            self.sink.release(request.producer_session, footprint);
            let _ = self.backend.release(allocation);
            return Err(PoolRefusal::ChargeRefused(refused));
        }
        self.next_slot += 1;
        let id = SlotId(self.next_slot);
        self.sequence += 1;
        increase(
            &mut self.producer_bytes,
            request.producer_session,
            footprint,
        );
        increase(
            &mut self.consumer_bytes,
            request.consumer_session,
            footprint,
        );
        increase(&mut self.device_bytes, request.device, footprint);
        increase(&mut self.pair_bytes, key, footprint);
        self.slots.insert(
            id,
            Slot {
                allocation,
                key,
                consumer_session: request.consumer_session,
                device: request.device,
                len: request.len,
                footprint,
                sequence: self.sequence,
                producer_reaped: false,
                consumer_reaped: false,
                extra: 0,
                poisoned: false,
                filled: false,
            },
        );
        Ok(self.lease(id, request, false))
    }

    fn lease(&self, id: SlotId, request: &AcquireRequest, reused: bool) -> Lease {
        let slot = &self.slots[&id];
        Lease {
            slot: id,
            record: ExportRecord {
                producer_session: request.producer_session,
                consumer_session: request.consumer_session,
                operation: request.operation,
                device: slot.device,
                size: slot.footprint,
            },
            footprint: slot.footprint,
            reused,
            sequence: slot.sequence,
        }
    }

    /// The oldest buffer of `key` whose consumer was reaped while the producer lives, on this
    /// device and with exactly this footprint.
    fn reusable_slot(&self, key: PairKey, device: u32, footprint: u64) -> Option<SlotId> {
        self.slots
            .iter()
            .filter(|(_, slot)| {
                slot.key == key
                    && slot.reusable()
                    && slot.device == device
                    && slot.footprint == footprint
            })
            .min_by_key(|(_, slot)| slot.sequence)
            .map(|(id, _)| *id)
    }

    fn reuse(&mut self, id: SlotId, request: &AcquireRequest) -> Result<Lease, PoolRefusal> {
        let (footprint, held, producer, key, device) = {
            let slot = self.slots.get(&id).ok_or(PoolRefusal::UnknownSlot(id))?;
            (
                slot.footprint,
                slot.held(),
                slot.key.producer_session,
                slot.key,
                slot.device,
            )
        };
        // A backend that cannot zero in place keeps the memory it replaced resident, so this
        // reuse holds one more footprint until the buffer is freed: count it like an
        // allocation against every bound.
        let extra = if self.backend.scrub_retains() {
            footprint
        } else {
            0
        };
        let new_holding = add(held, extra);
        let used = self.consumer_bytes(request.consumer_session);
        if add(used, new_holding) > self.limits.max_consumer_bytes {
            return Err(PoolRefusal::ConsumerQuota {
                consumer_session: request.consumer_session,
                used,
                cap: self.limits.max_consumer_bytes,
                requested: new_holding,
            });
        }
        if extra > 0 {
            let producer_used = self.producer_bytes(producer);
            if add(producer_used, extra) > self.limits.max_producer_bytes {
                return Err(PoolRefusal::ProducerQuota {
                    producer_session: producer,
                    used: producer_used,
                    cap: self.limits.max_producer_bytes,
                    requested: extra,
                    needed_release: add(producer_used, extra)
                        .saturating_sub(self.limits.max_producer_bytes),
                });
            }
            let pair_used = self.pair_bytes(key);
            if add(pair_used, extra) > self.limits.max_pair_bytes {
                return Err(PoolRefusal::PairBytes {
                    used: pair_used,
                    cap: self.limits.max_pair_bytes,
                    requested: extra,
                });
            }
            let device_used = self.device_bytes(device);
            let device_cap = self.limits.device_cap(device);
            if add(device_used, extra) > device_cap {
                return Err(PoolRefusal::DeviceCap {
                    device,
                    used: device_used,
                    cap: device_cap,
                    requested: extra,
                });
            }
        }
        // Charge before the scrub, so that retained memory is never uncharged.
        self.sink
            .charge(request.consumer_session, new_holding)
            .map_err(PoolRefusal::ChargeRefused)?;
        if extra > 0 {
            if let Err(refused) = self.sink.charge(producer, extra) {
                self.sink.release(request.consumer_session, new_holding);
                return Err(PoolRefusal::ChargeRefused(refused));
            }
        }
        // Zero all of it before anything else can see it. A buffer that cannot be zeroed is
        // not handed out again, but it stays allocated and charged until both domains are
        // reaped (rule 3b): it is poisoned, not freed.
        let scrubbed = {
            let slot = self
                .slots
                .get_mut(&id)
                .ok_or(PoolRefusal::UnknownSlot(id))?;
            self.backend.scrub(&mut slot.allocation, request.len)
        };
        if let Err(error) = scrubbed {
            self.sink.release(request.consumer_session, new_holding);
            if extra > 0 {
                self.sink.release(producer, extra);
            }
            if let Some(slot) = self.slots.get_mut(&id) {
                slot.poisoned = true;
            }
            return Err(PoolRefusal::Backend(error));
        }
        let old_consumer;
        {
            let slot = self
                .slots
                .get_mut(&id)
                .ok_or(PoolRefusal::UnknownSlot(id))?;
            old_consumer = slot.consumer_session;
            slot.consumer_session = request.consumer_session;
            slot.consumer_reaped = false;
            slot.len = request.len;
            slot.extra = add(slot.extra, extra);
            slot.filled = false;
            self.sequence += 1;
            slot.sequence = self.sequence;
        }
        self.sink.release(old_consumer, held);
        decrease(&mut self.consumer_bytes, old_consumer, held);
        increase(
            &mut self.consumer_bytes,
            request.consumer_session,
            new_holding,
        );
        if extra > 0 {
            increase(&mut self.producer_bytes, producer, extra);
            increase(&mut self.device_bytes, device, extra);
            increase(&mut self.pair_bytes, key, extra);
        }
        Ok(self.lease(id, request, true))
    }

    fn check_fresh(
        &self,
        key: PairKey,
        request: &AcquireRequest,
        footprint: u64,
    ) -> Result<(), PoolRefusal> {
        let producer_used = self.producer_bytes(request.producer_session);
        if add(producer_used, footprint) > self.limits.max_producer_bytes {
            return Err(PoolRefusal::ProducerQuota {
                producer_session: request.producer_session,
                used: producer_used,
                cap: self.limits.max_producer_bytes,
                requested: footprint,
                needed_release: add(producer_used, footprint)
                    .saturating_sub(self.limits.max_producer_bytes),
            });
        }
        let consumer_used = self.consumer_bytes(request.consumer_session);
        if add(consumer_used, footprint) > self.limits.max_consumer_bytes {
            return Err(PoolRefusal::ConsumerQuota {
                consumer_session: request.consumer_session,
                used: consumer_used,
                cap: self.limits.max_consumer_bytes,
                requested: footprint,
            });
        }
        let pair_used = self.pair_bytes(key);
        if add(pair_used, footprint) > self.limits.max_pair_bytes {
            return Err(PoolRefusal::PairBytes {
                used: pair_used,
                cap: self.limits.max_pair_bytes,
                requested: footprint,
            });
        }
        if self.slots.values().filter(|slot| slot.key == key).count() >= self.limits.max_pair_slots
        {
            return Err(PoolRefusal::PairSlots {
                max: self.limits.max_pair_slots,
            });
        }
        if self.slots.len() >= self.limits.max_slots {
            return Err(PoolRefusal::SlotLimit {
                max: self.limits.max_slots,
            });
        }
        let device_used = self.device_bytes(request.device);
        let device_cap = self.limits.device_cap(request.device);
        if add(device_used, footprint) > device_cap {
            return Err(PoolRefusal::DeviceCap {
                device: request.device,
                used: device_used,
                cap: device_cap,
                requested: footprint,
            });
        }
        Ok(())
    }

    fn slot_for_record(
        &self,
        slot: SlotId,
        record: &ExportRecord,
    ) -> Result<&Slot<B::Allocation>, PoolRefusal> {
        let held = self
            .slots
            .get(&slot)
            .ok_or(PoolRefusal::UnknownSlot(slot))?;
        if record.producer_session != held.key.producer_session {
            return Err(PoolRefusal::RecordMismatch {
                field: "producer_session",
            });
        }
        if record.consumer_session != held.consumer_session || held.consumer_reaped {
            return Err(PoolRefusal::RecordMismatch {
                field: "consumer_session",
            });
        }
        if record.device != held.device {
            return Err(PoolRefusal::RecordMismatch { field: "device" });
        }
        if record.size != held.footprint {
            return Err(PoolRefusal::RecordMismatch { field: "size" });
        }
        Ok(held)
    }

    /// A descriptor of the leased buffer for the session the record names. The record must be
    /// the one [`acquire`](Self::acquire) minted; each call yields a new descriptor (one for the
    /// producer to write through, one for the consumer).
    pub fn export(&mut self, slot: SlotId, record: &ExportRecord) -> Result<OwnedFd, PoolRefusal> {
        self.slot_for_record(slot, record)?;
        let held = self
            .slots
            .get(&slot)
            .ok_or(PoolRefusal::UnknownSlot(slot))?;
        self.backend
            .export(&held.allocation, record)
            .map_err(PoolRefusal::from)
    }

    /// Write the tensor into the leased buffer through the backend and return its SHA-256.
    pub fn fill(&mut self, slot: SlotId, bytes: &[u8]) -> Result<[u8; 32], PoolRefusal> {
        let held = self
            .slots
            .get_mut(&slot)
            .ok_or(PoolRefusal::UnknownSlot(slot))?;
        if bytes.len() as u64 != held.len {
            return Err(PoolRefusal::RecordMismatch { field: "len" });
        }
        if held.filled {
            return Err(PoolRefusal::AlreadyFilled);
        }
        let digest = self
            .backend
            .fill(&mut held.allocation, bytes)
            .map_err(PoolRefusal::from)?;
        held.filled = true;
        Ok(digest)
    }

    /// A session was terminated and reaped (its cgroup empty, its old channel end drained):
    /// every import it held is closed. Frees the buffers both of whose domains are now reaped,
    /// and marks those whose consumer it was reusable. Unknown and repeated sessions change
    /// nothing.
    pub fn on_reaped(&mut self, session: u64) -> ReapReport {
        self.remember_reaped(session);
        let mut report = ReapReport::default();
        let affected: Vec<SlotId> = self
            .slots
            .iter()
            .filter(|(_, slot)| {
                slot.key.producer_session == session || slot.consumer_session == session
            })
            .map(|(id, _)| *id)
            .collect();
        for id in affected {
            let Some(slot) = self.slots.get_mut(&id) else {
                continue;
            };
            let was_reusable = slot.reusable();
            if slot.key.producer_session == session {
                slot.producer_reaped = true;
            }
            if slot.consumer_session == session {
                slot.consumer_reaped = true;
            }
            let both = slot.producer_reaped && slot.consumer_reaped;
            let now_reusable = slot.reusable();
            let footprint = slot.held();
            if both {
                if self.free(id) {
                    report.freed_bytes = add(report.freed_bytes, footprint);
                    report.freed.push(id);
                } else {
                    report.release_failed.push(id);
                }
            } else if !was_reusable && now_reusable {
                report.became_reusable.push(id);
            }
        }
        report
    }

    fn remember_reaped(&mut self, session: u64) {
        if self.reaped.insert(session) {
            self.reaped_order.push_back(session);
            while self.reaped_order.len() > MAX_REMEMBERED_REAPED {
                if let Some(oldest) = self.reaped_order.pop_front() {
                    self.reaped.remove(&oldest);
                }
            }
        }
    }

    /// Release a buffer to the backend and everything it was charged to. When the backend
    /// cannot release it (it is unreachable, so the memory may still be held) the charges stay,
    /// in `stuck`, until [`abandon_all`](Self::abandon_all). Returns whether it was released.
    fn free(&mut self, id: SlotId) -> bool {
        let Some(slot) = self.slots.remove(&id) else {
            return true;
        };
        let charged = slot.charged();
        match self.backend.release(slot.allocation) {
            Ok(()) => {
                self.uncharge(&charged);
                true
            }
            Err(_) => {
                self.stuck.push(charged);
                false
            }
        }
    }

    fn uncharge(&mut self, charged: &Charged) {
        self.sink
            .release(charged.key.producer_session, charged.bytes);
        self.sink.release(charged.consumer_session, charged.bytes);
        decrease(
            &mut self.producer_bytes,
            charged.key.producer_session,
            charged.bytes,
        );
        decrease(
            &mut self.consumer_bytes,
            charged.consumer_session,
            charged.bytes,
        );
        decrease(&mut self.device_bytes, charged.device, charged.bytes);
        decrease(&mut self.pair_bytes, charged.key, charged.bytes);
    }

    /// The guardian is gone, and with it every buffer it held: drop all slots without calling
    /// the backend, release every charge (those of buffers whose release failed included), and
    /// return how many buffers and bytes that was. No worker may be running that could still
    /// import one: the caller ends them first.
    pub fn abandon_all(&mut self) -> (usize, u64) {
        let slots = std::mem::take(&mut self.slots);
        let mut all: Vec<Charged> = slots.values().map(Slot::charged).collect();
        all.append(&mut self.stuck);
        let mut bytes = 0u64;
        for charged in &all {
            bytes = add(bytes, charged.bytes);
            self.uncharge(charged);
        }
        (all.len(), bytes)
    }

    /// Which consumers to recycle together with `producer_session` so that at least `needed`
    /// footprint bytes are released, oldest export first. Buffers whose consumer is reaped are
    /// released by recycling the producer alone; each listed consumer releases all its buffers of
    /// this producer. `covers` is false when even all of them do not reach `needed`. The plan
    /// may list the consumer of the very handle that was refused (its other buffers of this
    /// producer hold the quota); whether to recycle it is the caller's decision.
    pub fn recovery_plan(&self, producer_session: u64, needed: u64) -> RecoveryPlan {
        let mut own: Vec<&Slot<B::Allocation>> = self
            .slots
            .values()
            .filter(|slot| slot.key.producer_session == producer_session)
            .collect();
        own.sort_by_key(|slot| slot.sequence);
        let mut released = own
            .iter()
            .filter(|slot| slot.consumer_reaped)
            .fold(0u64, |sum, slot| add(sum, slot.held()));
        let mut consumers: Vec<u64> = Vec::new();
        for slot in &own {
            if released >= needed {
                break;
            }
            if slot.consumer_reaped || consumers.contains(&slot.consumer_session) {
                continue;
            }
            consumers.push(slot.consumer_session);
            released = own
                .iter()
                .filter(|other| other.consumer_session == slot.consumer_session)
                .fold(released, |sum, other| add(sum, other.held()));
        }
        RecoveryPlan {
            producer_session,
            consumers,
            released_bytes: released,
            covers: released >= needed,
        }
    }
}

impl<B: StagingBackend> fmt::Debug for StagingPool<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StagingPool")
            .field("slots", &self.slots.len())
            .field("device_bytes", &self.device_bytes)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staging::{ChargeBook, FakeBackend, FakeCounters};

    const UNIT: u64 = 4096;

    fn pool(limits: PoolLimits) -> (StagingPool<FakeBackend>, FakeCounters, ChargeBook) {
        let backend = FakeBackend::with_unit(UNIT);
        let counters = backend.counters();
        let book = ChargeBook::new();
        (
            StagingPool::new(backend, Box::new(book.clone()), limits),
            counters,
            book,
        )
    }

    fn request(producer: u64, consumer: u64, lane: u64, len: u64) -> AcquireRequest {
        AcquireRequest {
            producer_session: producer,
            consumer_session: consumer,
            consumer_lane: lane,
            operation: 7,
            device: 0,
            len,
        }
    }

    #[test]
    fn a_fresh_buffer_is_charged_as_allocated_to_both_sessions() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        let lease = pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        assert!(!lease.reused);
        assert_eq!(lease.footprint, UNIT);
        assert_eq!(lease.record.size, UNIT);
        assert_eq!((book.charged(1), book.charged(2)), (UNIT, UNIT));
        assert_eq!(
            (pool.producer_bytes(1), pool.consumer_bytes(2)),
            (UNIT, UNIT)
        );
        assert_eq!(pool.device_bytes(0), UNIT);
        assert_eq!(counters.live(), 1);
    }

    #[test]
    fn every_handle_gets_a_fresh_buffer_while_the_consumer_lives() {
        let (mut pool, counters, _) = pool(PoolLimits::small_host());
        let first = pool.acquire(&request(1, 2, 20, 100)).expect("first");
        let second = pool.acquire(&request(1, 2, 20, 100)).expect("second");
        assert_ne!(first.slot, second.slot);
        assert!(!second.reused);
        assert_eq!(counters.live(), 2);
    }

    #[test]
    fn a_buffer_is_reused_only_after_its_consumer_is_reaped_and_is_zeroed() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        let first = pool.acquire(&request(1, 2, 20, 100)).expect("first");
        pool.fill(first.slot, &[9u8; 100]).expect("fill");
        let report = pool.on_reaped(2);
        assert_eq!(report.became_reusable, vec![first.slot]);
        assert!(report.freed.is_empty());
        // The consumer is gone; the producer lives: the buffer stays allocated and charged.
        assert_eq!(counters.live(), 1);
        assert_eq!(book.charged(1), UNIT);
        // A new consumer session on the same lane gets the same buffer, zeroed.
        let second = pool.acquire(&request(1, 3, 20, 50)).expect("second");
        assert!(second.reused);
        assert_eq!(second.slot, first.slot);
        assert_eq!(counters.allocated(), 1);
        assert_eq!(counters.scrubs(), 1);
        assert_eq!(second.record.consumer_session, 3);
        // The charge moved from the old consumer to the new one.
        assert_eq!(
            (book.charged(1), book.charged(2), book.charged(3)),
            (UNIT, 0, UNIT)
        );
    }

    #[test]
    fn another_lane_or_producer_never_reuses_the_buffer() {
        let (mut pool, counters, _) = pool(PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 100)).expect("first");
        pool.on_reaped(2);
        let other_lane = pool.acquire(&request(1, 3, 21, 100)).expect("other lane");
        assert!(!other_lane.reused);
        let other_producer = pool
            .acquire(&request(5, 6, 20, 100))
            .expect("other producer");
        assert!(!other_producer.reused);
        assert_eq!(counters.allocated(), 3);
    }

    #[test]
    fn a_buffer_of_another_footprint_or_device_is_not_reused() {
        let (mut pool, counters, _) = pool(PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 100)).expect("first");
        pool.on_reaped(2);
        let bigger = pool.acquire(&request(1, 3, 20, UNIT + 1)).expect("bigger");
        assert!(!bigger.reused);
        let mut other_device = request(1, 4, 20, 100);
        other_device.device = 1;
        assert!(!pool.acquire(&other_device).expect("device").reused);
        assert_eq!(counters.allocated(), 3);
    }

    #[test]
    fn a_buffer_is_freed_only_after_both_domains_are_reaped() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        let lease = pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        // Producer first: the consumer may still map the buffer.
        let report = pool.on_reaped(1);
        assert!(report.freed.is_empty());
        assert_eq!(counters.live(), 1);
        assert_eq!((book.charged(1), book.charged(2)), (UNIT, UNIT));
        let report = pool.on_reaped(2);
        assert_eq!(report.freed, vec![lease.slot]);
        assert_eq!(report.freed_bytes, UNIT);
        assert_eq!(counters.live(), 0);
        assert_eq!(book.total(), 0);
        assert_eq!(pool.device_bytes(0), 0);
        assert_eq!(pool.slot_count(), 0);
    }

    #[test]
    fn a_reusable_buffer_is_freed_when_its_producer_is_reaped_later() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        pool.on_reaped(2);
        assert_eq!(counters.live(), 1);
        let report = pool.on_reaped(1);
        assert_eq!(report.freed.len(), 1);
        assert_eq!((counters.live(), book.total()), (0, 0));
    }

    #[test]
    fn reaping_twice_or_an_unknown_session_changes_nothing() {
        let (mut pool, counters, _) = pool(PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        assert_eq!(pool.on_reaped(99), ReapReport::default());
        pool.on_reaped(2);
        assert_eq!(pool.on_reaped(2), ReapReport::default());
        assert_eq!(counters.live(), 1);
    }

    #[test]
    fn a_reaped_session_gets_no_further_buffer() {
        let (mut pool, _, _) = pool(PoolLimits::small_host());
        pool.on_reaped(2);
        assert_eq!(
            pool.acquire(&request(1, 2, 20, 100)),
            Err(PoolRefusal::SessionReaped { session: 2 })
        );
        assert_eq!(
            pool.acquire(&request(2, 3, 20, 100)),
            Err(PoolRefusal::SessionReaped { session: 2 })
        );
    }

    #[test]
    fn empty_oversize_and_same_session_requests_are_refused() {
        let mut limits = PoolLimits::small_host();
        limits.max_handle_len = 1000;
        let (mut pool, counters, _) = pool(limits);
        assert_eq!(
            pool.acquire(&request(1, 2, 20, 0)),
            Err(PoolRefusal::EmptyTensor)
        );
        assert_eq!(
            pool.acquire(&request(1, 2, 20, 1001)),
            Err(PoolRefusal::TooLarge {
                len: 1001,
                max: 1000
            })
        );
        assert_eq!(
            pool.acquire(&request(1, 1, 20, 10)),
            Err(PoolRefusal::SameSession { session: 1 })
        );
        assert_eq!(counters.allocated(), 0);
    }

    #[test]
    fn the_device_cap_is_a_typed_refusal_per_device() {
        let mut limits = PoolLimits::small_host();
        limits.default_device_bytes = 2 * UNIT;
        limits.device_bytes.insert(1, 5 * UNIT);
        let (mut pool, counters, _) = pool(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("one");
        pool.acquire(&request(1, 3, 21, 10)).expect("two");
        assert_eq!(
            pool.acquire(&request(1, 4, 22, 10)),
            Err(PoolRefusal::DeviceCap {
                device: 0,
                used: 2 * UNIT,
                cap: 2 * UNIT,
                requested: UNIT
            })
        );
        assert_eq!(counters.allocated(), 2);
        // The other device has its own, larger cap.
        let mut other = request(1, 4, 22, 10);
        other.device = 1;
        pool.acquire(&other).expect("device 1");
    }

    #[test]
    fn pair_producer_consumer_and_slot_bounds_are_typed_refusals() {
        let mut limits = PoolLimits::small_host();
        limits.max_pair_slots = 2;
        let (mut pool, _, _) = pool(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("one");
        pool.acquire(&request(1, 2, 20, 10)).expect("two");
        assert_eq!(
            pool.acquire(&request(1, 2, 20, 10)),
            Err(PoolRefusal::PairSlots { max: 2 })
        );

        let mut limits = PoolLimits::small_host();
        limits.max_pair_bytes = UNIT;
        let (mut pool, _, _) = pool_with(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("one");
        assert_eq!(
            pool.acquire(&request(1, 2, 20, 10)),
            Err(PoolRefusal::PairBytes {
                used: UNIT,
                cap: UNIT,
                requested: UNIT
            })
        );

        let mut limits = PoolLimits::small_host();
        limits.max_consumer_bytes = UNIT;
        let (mut pool, _, _) = pool_with(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("one");
        assert!(matches!(
            pool.acquire(&request(5, 2, 21, 10)),
            Err(PoolRefusal::ConsumerQuota {
                consumer_session: 2,
                ..
            })
        ));

        let mut limits = PoolLimits::small_host();
        limits.max_slots = 1;
        let (mut pool, _, _) = pool_with(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("one");
        assert_eq!(
            pool.acquire(&request(3, 4, 21, 10)),
            Err(PoolRefusal::SlotLimit { max: 1 })
        );
    }

    fn pool_with(limits: PoolLimits) -> (StagingPool<FakeBackend>, FakeCounters, ChargeBook) {
        pool(limits)
    }

    #[test]
    fn a_refused_charge_rolls_everything_back() {
        let backend = FakeBackend::with_unit(UNIT);
        let counters = backend.counters();
        let book = ChargeBook::with_cap_per_session(UNIT);
        let mut pool = StagingPool::new(backend, Box::new(book.clone()), PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 10)).expect("one");
        // Producer 1 is at the book's cap: the second charge to it is refused after allocation.
        let refused = pool.acquire(&request(1, 3, 21, 10));
        assert!(matches!(refused, Err(PoolRefusal::ChargeRefused(_))));
        assert_eq!(counters.live(), 1);
        assert_eq!(
            (book.charged(1), book.charged(2), book.charged(3)),
            (UNIT, UNIT, 0)
        );
        assert_eq!(pool.slot_count(), 1);
        // A consumer refusal rolls back the producer charge made just before it.
        let mut book = ChargeBook::with_cap_per_session(UNIT);
        book.charge(9, UNIT).expect("prefill consumer 9");
        let backend = FakeBackend::with_unit(UNIT);
        let counters = backend.counters();
        let mut pool = StagingPool::new(backend, Box::new(book.clone()), PoolLimits::small_host());
        assert!(matches!(
            pool.acquire(&request(1, 9, 20, 10)),
            Err(PoolRefusal::ChargeRefused(_))
        ));
        assert_eq!(
            (counters.live(), book.charged(1), book.charged(9)),
            (0, 0, UNIT)
        );
    }

    #[test]
    fn a_backend_failure_charges_nothing() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        pool.backend_mut().fail_allocations(true);
        assert!(matches!(
            pool.acquire(&request(1, 2, 20, 10)),
            Err(PoolRefusal::Backend(_))
        ));
        assert_eq!(
            (counters.live(), book.total(), pool.slot_count()),
            (0, 0, 0)
        );
    }

    #[test]
    fn export_requires_the_record_the_pool_minted() {
        let (mut pool, counters, _) = pool(PoolLimits::small_host());
        let lease = pool.acquire(&request(1, 2, 20, 10)).expect("lease");
        let fd = pool.export(lease.slot, &lease.record).expect("export");
        drop(fd);
        assert_eq!(counters.exports(), 1);
        for (field, bad) in [
            (
                "producer_session",
                ExportRecord {
                    producer_session: 9,
                    ..lease.record
                },
            ),
            (
                "consumer_session",
                ExportRecord {
                    consumer_session: 9,
                    ..lease.record
                },
            ),
            (
                "device",
                ExportRecord {
                    device: 1,
                    ..lease.record
                },
            ),
            (
                "size",
                ExportRecord {
                    size: lease.record.size + 1,
                    ..lease.record
                },
            ),
        ] {
            assert_eq!(
                pool.export(lease.slot, &bad).map(|_| ()),
                Err(PoolRefusal::RecordMismatch { field })
            );
        }
        assert_eq!(
            pool.export(SlotId(99), &lease.record).map(|_| ()),
            Err(PoolRefusal::UnknownSlot(SlotId(99)))
        );
        // After the consumer is reaped the old record no longer exports.
        pool.on_reaped(2);
        assert!(pool.export(lease.slot, &lease.record).is_err());
    }

    #[test]
    fn recovery_lists_consumers_oldest_export_first_until_the_bytes_cover() {
        let mut limits = PoolLimits::small_host();
        limits.max_producer_bytes = 4 * UNIT;
        let (mut pool, _, _) = pool(limits);
        // Producer 1 exports to consumers 2, 3, 4 and 2 again, oldest first.
        pool.acquire(&request(1, 2, 20, 10)).expect("a");
        pool.acquire(&request(1, 3, 21, 10)).expect("b");
        pool.acquire(&request(1, 4, 22, 10)).expect("c");
        pool.acquire(&request(1, 2, 20, 10)).expect("d");
        let refusal = pool.acquire(&request(1, 5, 23, 10)).expect_err("full");
        let PoolRefusal::ProducerQuota {
            needed_release,
            used,
            cap,
            ..
        } = refusal
        else {
            panic!("expected the producer quota, got {refusal:?}");
        };
        assert_eq!((used, cap, needed_release), (4 * UNIT, 4 * UNIT, UNIT));
        // One buffer is enough: consumer 2 holds the oldest export, but recycling it releases
        // both of its buffers.
        let plan = pool.recovery_plan(1, needed_release);
        assert_eq!(plan.consumers, vec![2]);
        assert_eq!(plan.released_bytes, 2 * UNIT);
        assert!(plan.covers);
        // Three buffers needed: consumer 2 (two buffers), then 3.
        let plan = pool.recovery_plan(1, 3 * UNIT);
        assert_eq!(plan.consumers, vec![2, 3]);
        assert_eq!(plan.released_bytes, 3 * UNIT);
        // More than the producer holds: everyone, and the plan says it does not cover.
        let plan = pool.recovery_plan(1, 9 * UNIT);
        assert_eq!(plan.consumers, vec![2, 3, 4]);
        assert!(!plan.covers);
        // A producer without buffers has nothing to recycle.
        let plan = pool.recovery_plan(77, UNIT);
        assert!(plan.consumers.is_empty() && !plan.covers);
    }

    #[test]
    fn recovery_counts_buffers_of_reaped_consumers_without_listing_them() {
        let mut limits = PoolLimits::small_host();
        limits.max_producer_bytes = 2 * UNIT;
        let (mut pool, _, _) = pool(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("a");
        pool.acquire(&request(1, 3, 21, 10)).expect("b");
        pool.on_reaped(2);
        let plan = pool.recovery_plan(1, UNIT);
        assert!(plan.consumers.is_empty());
        assert!(plan.covers);
        let plan = pool.recovery_plan(1, 2 * UNIT);
        assert_eq!(plan.consumers, vec![3]);
    }

    #[test]
    fn executing_the_plan_frees_the_quota() {
        let mut limits = PoolLimits::small_host();
        limits.max_producer_bytes = 2 * UNIT;
        let (mut pool, counters, book) = pool(limits);
        pool.acquire(&request(1, 2, 20, 10)).expect("a");
        pool.acquire(&request(1, 3, 21, 10)).expect("b");
        let PoolRefusal::ProducerQuota { needed_release, .. } =
            pool.acquire(&request(1, 4, 22, 10)).expect_err("full")
        else {
            panic!("expected the producer quota");
        };
        let plan = pool.recovery_plan(1, needed_release);
        assert_eq!(plan.consumers, vec![2]);
        // The caller recycles the producer and consumer 2; the replacement producer is new.
        for session in std::iter::once(plan.producer_session).chain(plan.consumers.clone()) {
            pool.on_reaped(session);
        }
        // Consumer 3's buffer waits for it; consumer 2's is gone.
        assert_eq!(counters.live(), 1);
        assert_eq!(book.charged(1), UNIT);
        let fresh = pool.acquire(&request(10, 4, 22, 10)).expect("new producer");
        assert!(!fresh.reused);
    }

    #[test]
    fn abandoning_all_releases_every_charge_without_the_backend() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 10)).expect("a");
        pool.acquire(&request(3, 4, 21, 10)).expect("b");
        assert_eq!(pool.backed_sessions(), BTreeSet::from([1, 2, 3, 4]));
        assert_eq!(pool.abandon_all(), (2, 2 * UNIT));
        assert_eq!(
            (pool.slot_count(), book.total(), pool.device_bytes(0)),
            (0, 0, 0)
        );
        // The allocations were dropped, not released through the backend hook; the fake counts
        // both the same.
        assert_eq!(counters.live(), 0);
    }

    #[test]
    fn a_buffer_that_cannot_be_scrubbed_is_poisoned_not_freed() {
        let (mut pool, counters, book) = pool(PoolLimits::small_host());
        pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        pool.on_reaped(2);
        pool.backend_mut().fail_scrubs(true);
        assert!(matches!(
            pool.acquire(&request(1, 3, 20, 100)),
            Err(PoolRefusal::Backend(_))
        ));
        // The producer lives: the buffer stays allocated and charged (rule 3b).
        assert_eq!((counters.live(), book.charged(1)), (1, UNIT));
        assert_eq!(pool.slot_count(), 1);
        // Excluded from reuse even when scrubbing works again: a fresh buffer is made.
        pool.backend_mut().fail_scrubs(false);
        let fresh = pool.acquire(&request(1, 3, 20, 100)).expect("fresh");
        assert!(!fresh.reused);
        assert_eq!(counters.allocated(), 2);
        // It goes only when both domains are reaped.
        pool.on_reaped(1);
        assert_eq!(counters.live(), 1, "the other buffer waits for consumer 3");
        assert_eq!(book.charged(1), UNIT);
    }

    #[test]
    fn a_retaining_scrub_counts_the_old_memory_against_every_bound() {
        let mut limits = PoolLimits::small_host();
        limits.default_device_bytes = 2 * UNIT;
        let (mut pool, counters, book) = pool(limits);
        pool.backend_mut().retain_on_scrub(true);
        pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        pool.on_reaped(2);
        let reused = pool.acquire(&request(1, 3, 20, 100)).expect("reuse");
        assert!(reused.reused);
        // The old memory is still resident: two footprints are held, charged and counted.
        assert_eq!(counters.live_bytes(), 2 * UNIT);
        assert_eq!(pool.device_bytes(0), 2 * UNIT);
        assert_eq!((book.charged(1), book.charged(3)), (2 * UNIT, 2 * UNIT));
        // Another reuse would hold three: refused, nothing changes.
        pool.on_reaped(3);
        let refusal = pool
            .acquire(&request(1, 4, 20, 100))
            .expect_err("over the cap");
        assert!(
            matches!(refusal, PoolRefusal::DeviceCap { device: 0, .. }),
            "{refusal:?}"
        );
        assert_eq!(counters.live_bytes(), 2 * UNIT);
        assert_eq!(book.charged(1), 2 * UNIT);
        // Both domains reaped: everything, retired memory included, goes.
        pool.on_reaped(1);
        assert_eq!(
            (counters.live_bytes(), book.total(), pool.device_bytes(0)),
            (0, 0, 0)
        );
    }

    #[test]
    fn a_lease_is_filled_once_and_a_reuse_resets_it() {
        let (mut pool, _, _) = pool(PoolLimits::small_host());
        let lease = pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        pool.fill(lease.slot, &[1u8; 100]).expect("first fill");
        assert_eq!(
            pool.fill(lease.slot, &[2u8; 100]),
            Err(PoolRefusal::AlreadyFilled)
        );
        pool.on_reaped(2);
        let again = pool.acquire(&request(1, 3, 20, 100)).expect("reuse");
        assert!(again.reused);
        pool.fill(again.slot, &[3u8; 100])
            .expect("a reused buffer is filled again");
    }

    /// A backend whose release can be made to fail, like a guardian that stopped answering.
    struct Flaky {
        inner: FakeBackend,
        fail_release: std::rc::Rc<std::cell::Cell<bool>>,
    }

    impl StagingBackend for Flaky {
        type Allocation = <FakeBackend as StagingBackend>::Allocation;

        fn footprint_for(&mut self, device: u32, len: u64) -> Result<u64, BackendError> {
            self.inner.footprint_for(device, len)
        }

        fn allocate(
            &mut self,
            device: u32,
            len: u64,
        ) -> Result<(Self::Allocation, u64), BackendError> {
            self.inner.allocate(device, len)
        }

        fn fill(
            &mut self,
            a: &mut Self::Allocation,
            bytes: &[u8],
        ) -> Result<[u8; 32], BackendError> {
            self.inner.fill(a, bytes)
        }

        fn export(
            &mut self,
            a: &Self::Allocation,
            r: &ExportRecord,
        ) -> Result<OwnedFd, BackendError> {
            self.inner.export(a, r)
        }

        fn scrub(&mut self, a: &mut Self::Allocation, len: u64) -> Result<(), BackendError> {
            self.inner.scrub(a, len)
        }

        fn release(&mut self, a: Self::Allocation) -> Result<(), BackendError> {
            if self.fail_release.get() {
                std::mem::forget(a);
                return Err(BackendError::new("unreachable"));
            }
            self.inner.release(a)
        }
    }

    #[test]
    fn a_release_the_backend_did_not_confirm_keeps_its_charges_until_abandon() {
        let fail = std::rc::Rc::new(std::cell::Cell::new(false));
        let book = ChargeBook::new();
        let mut pool = StagingPool::new(
            Flaky {
                inner: FakeBackend::with_unit(UNIT),
                fail_release: fail.clone(),
            },
            Box::new(book.clone()),
            PoolLimits::small_host(),
        );
        pool.acquire(&request(1, 2, 20, 100)).expect("lease");
        fail.set(true);
        pool.on_reaped(1);
        let report = pool.on_reaped(2);
        assert_eq!(report.release_failed.len(), 1);
        assert!(report.freed.is_empty());
        // The memory may still be held, so nothing was recredited.
        assert_eq!(
            (book.charged(1), book.charged(2), pool.stuck_count()),
            (UNIT, UNIT, 1)
        );
        assert_eq!(pool.device_bytes(0), UNIT);
        // The guardian is gone: now they go.
        assert_eq!(pool.abandon_all(), (1, UNIT));
        assert_eq!(
            (book.total(), pool.stuck_count(), pool.device_bytes(0)),
            (0, 0, 0)
        );
    }
}
