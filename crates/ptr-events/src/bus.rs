use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::io;

use crate::codec::MAX_FIELD_BYTES;
use crate::policy::{
    slowest, Authority, BackpressurePolicy, ClassPolicies, Compaction, ConsumerLag, Limit,
    PolicyViolation, RetentionPolicy,
};

/// Largest consumer count a bus accepts (Invariant 8: bounded everything).
pub const MAX_CONSUMERS: usize = 256;
/// Default bound on one record's payload.
pub const DEFAULT_MAX_RECORD_BYTES: usize = 1 << 20;

/// Position of a record in the bus. The first record is at offset `1`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Offset(pub u64);

/// What a record is allowed to mean to a consumer.
///
/// A `Projection` record restates something the ledger committed; a consumer
/// may rebuild derived state from it but must still treat the ledger as the
/// authority. A `Telemetry` record describes execution and may never feed
/// state, memory or training labels as if it were a committed fact.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EventClass {
    Projection,
    Telemetry,
}

impl EventClass {
    /// The authority this class grants. Authority is a function of the class and nothing else.
    pub const fn authority(self) -> Authority {
        match self {
            Self::Projection => Authority::Authoritative,
            Self::Telemetry => Authority::Diagnostic,
        }
    }
}

/// A record offered to the bus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewRecord {
    pub class: EventClass,
    pub topic: String,
    /// Partitioning/deduplication key, for example a capsule id.
    pub key: String,
    pub payload: Vec<u8>,
}

impl NewRecord {
    /// Offer a record that was read from a bus again, keeping its class. A telemetry record
    /// stays telemetry.
    pub fn republish(record: &BusRecord) -> Self {
        Self {
            class: record.class,
            topic: record.topic.clone(),
            key: record.key.clone(),
            payload: record.payload.clone(),
        }
    }

    /// Re-offer a record under a (possibly different) class.
    ///
    /// # Errors
    /// [`BusError::TelemetryPromotion`] when a telemetry record is asked to become a
    /// projection: derived data never self-promotes to authority (Invariants 4 and 5). Only the
    /// reverse (an authoritative record demoted to telemetry) is allowed.
    pub fn reclassify(record: &BusRecord, to: EventClass) -> Result<Self, BusError> {
        if record.class == EventClass::Telemetry && to == EventClass::Projection {
            return Err(BusError::TelemetryPromotion {
                offset: record.offset,
            });
        }
        Ok(Self {
            class: to,
            ..Self::republish(record)
        })
    }
}

/// A record as a consumer sees it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BusRecord {
    pub offset: Offset,
    pub class: EventClass,
    /// Derived from `class`; see [`EventClass::authority`].
    pub authority: Authority,
    /// Logical tick at which the record was published (caller-supplied clock).
    pub tick: u64,
    pub topic: String,
    pub key: String,
    pub payload: Vec<u8>,
}

impl BusRecord {
    pub(crate) fn from_parts(
        offset: Offset,
        tick: u64,
        class: EventClass,
        topic: String,
        key: String,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            offset,
            class,
            authority: class.authority(),
            tick,
            topic,
            key,
            payload,
        }
    }

    /// Bytes the record occupies in memory accounting: topic, key and payload.
    pub fn size_bytes(&self) -> usize {
        self.topic.len() + self.key.len() + self.payload.len()
    }
}

/// A field with a length bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    Topic,
    Key,
    Payload,
    ConsumerName,
}

/// What a corrupt file looked like.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorruptionKind {
    ChecksumMismatch,
    BadLength,
    /// The file ends inside a frame (outside the tolerated tail of the newest segment).
    IncompleteFrame,
    BadEncoding,
    BadMagic,
    /// A record's offset is not the next expected one.
    OffsetGap {
        expected: u64,
        found: u64,
    },
    /// A segment does not start where the previous one ended.
    SegmentGap {
        expected: u64,
        found: u64,
    },
}

/// Refusals from the bus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BusError {
    /// A publish met a full bus and the class policy is to reject. `depth` is the unit-matching
    /// amount of unacknowledged data currently held; `blocked_by` names the consumer whose lag
    /// holds it. Nothing is dropped to make room.
    Full {
        limit: Limit,
        capacity: usize,
        depth: usize,
        blocked_by: Option<String>,
    },
    /// The consumer is not registered.
    UnknownConsumer { consumer: String },
    /// A consumer tried to commit, or register, past the last record. A
    /// registration offset is the consumer's commit position, so it obeys the
    /// same bound.
    CommitBeyondEnd { requested: Offset, end: Offset },
    /// The requested position is below the low-water mark: those records were compacted.
    Truncated {
        requested: Offset,
        low_water: Offset,
    },
    /// A telemetry record was asked to become an authoritative projection.
    TelemetryPromotion { offset: Offset },
    /// The policy or configuration is not allowed.
    InvalidPolicy { violation: PolicyViolation },
    /// A field exceeds its bound.
    FieldTooLong {
        field: Field,
        len: usize,
        max: usize,
    },
    /// More than [`MAX_CONSUMERS`] consumers.
    ConsumerLimit { max: usize },
    /// The logical clock moved backwards.
    ClockRegression { current: u64, requested: u64 },
    /// A persisted file failed validation (not a tolerated torn tail).
    Corrupt {
        file: String,
        position: u64,
        kind: CorruptionKind,
    },
    /// The persisted consumer offsets are past the end of the recovered log (records the
    /// consumer had acknowledged are gone, for example after an unsynced crash).
    OffsetsAheadOfLog {
        consumer: String,
        committed: Offset,
        end: Offset,
    },
    /// A failed append could not be rolled back, so the on-disk tail is unknown. Every publish
    /// fails with this until the log is reopened and recovered.
    WriterPoisoned,
    /// An operating-system I/O failure.
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
}

impl BusError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Full { .. } => "PTR_BUS_FULL",
            Self::UnknownConsumer { .. } => "PTR_BUS_UNKNOWN_CONSUMER",
            Self::CommitBeyondEnd { .. } => "PTR_BUS_COMMIT_BEYOND_END",
            Self::Truncated { .. } => "PTR_BUS_TRUNCATED",
            Self::TelemetryPromotion { .. } => "PTR_BUS_TELEMETRY_PROMOTION",
            Self::InvalidPolicy { .. } => "PTR_BUS_INVALID_POLICY",
            Self::FieldTooLong { .. } => "PTR_BUS_FIELD_TOO_LONG",
            Self::ConsumerLimit { .. } => "PTR_BUS_CONSUMER_LIMIT",
            Self::ClockRegression { .. } => "PTR_BUS_CLOCK_REGRESSION",
            Self::Corrupt { .. } => "PTR_BUS_CORRUPT",
            Self::OffsetsAheadOfLog { .. } => "PTR_BUS_OFFSETS_AHEAD_OF_LOG",
            Self::WriterPoisoned => "PTR_BUS_WRITER_POISONED",
            Self::Io { .. } => "PTR_BUS_IO",
        }
    }

    pub(crate) fn io(operation: &'static str, error: &io::Error) -> Self {
        Self::Io {
            operation,
            kind: error.kind(),
        }
    }
}

impl fmt::Display for BusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full {
                limit,
                capacity,
                depth,
                blocked_by,
            } => write!(
                formatter,
                "bus is full: {depth} of {capacity} {limit:?} unacknowledged (blocked by {blocked_by:?})"
            ),
            Self::UnknownConsumer { consumer } => {
                write!(formatter, "consumer {consumer:?} is not registered")
            }
            Self::CommitBeyondEnd { requested, end } => write!(
                formatter,
                "commit to offset {} is beyond the last offset {}",
                requested.0, end.0
            ),
            Self::Truncated {
                requested,
                low_water,
            } => write!(
                formatter,
                "offset {} is below the low-water mark {}",
                requested.0, low_water.0
            ),
            Self::TelemetryPromotion { offset } => write!(
                formatter,
                "telemetry record {} cannot become an authoritative projection",
                offset.0
            ),
            Self::InvalidPolicy { violation } => write!(formatter, "invalid policy: {violation:?}"),
            Self::FieldTooLong { field, len, max } => {
                write!(formatter, "{field:?} is {len} bytes, the limit is {max}")
            }
            Self::ConsumerLimit { max } => write!(formatter, "more than {max} consumers"),
            Self::ClockRegression { current, requested } => write!(
                formatter,
                "tick {requested} is behind the current tick {current}"
            ),
            Self::Corrupt {
                file,
                position,
                kind,
            } => write!(formatter, "{file} is corrupt at byte {position}: {kind:?}"),
            Self::OffsetsAheadOfLog {
                consumer,
                committed,
                end,
            } => write!(
                formatter,
                "consumer {consumer:?} committed {} but the log ends at {}",
                committed.0, end.0
            ),
            Self::WriterPoisoned => write!(
                formatter,
                "a failed append could not be rolled back; reopen the log to recover"
            ),
            Self::Io { operation, kind } => write!(formatter, "{operation} failed: {kind}"),
        }
    }
}

impl std::error::Error for BusError {}

/// Producer side of an event bus.
pub trait EventProducer {
    fn publish(&mut self, record: NewRecord) -> Result<Offset, BusError>;
}

/// Consumer side of an event bus with at-least-once delivery: records are
/// returned again until the consumer commits past them.
pub trait EventConsumer {
    /// Register a consumer whose commit position is `from`. A consumer registered at the
    /// low-water mark sees every retained record.
    ///
    /// # Errors
    /// `CommitBeyondEnd` if `from` is past the last offset, `Truncated` if it is below the
    /// low-water mark (records the consumer would have to see are gone).
    fn register(&mut self, consumer: &str, from: Offset) -> Result<(), BusError>;
    /// Records after the consumer's committed offset, without advancing it.
    fn poll(&self, consumer: &str, max: usize) -> Result<Vec<BusRecord>, BusError>;
    /// Acknowledge records through `upto`. Monotone: an older offset does not rewind.
    fn commit(&mut self, consumer: &str, upto: Offset) -> Result<(), BusError>;
    /// The consumer's acknowledged offset.
    fn committed(&self, consumer: &str) -> Result<Offset, BusError>;
    /// Records published but not yet acknowledged by the consumer.
    fn lag(&self, consumer: &str) -> Result<u64, BusError>;
    /// Offset of the last published record (`Offset(0)` when none was ever published).
    fn end(&self) -> Offset;
    /// Records at or below this offset were compacted away.
    fn low_water(&self) -> Offset;
    /// Replay: records after `after`, independent of any consumer.
    ///
    /// # Errors
    /// `Truncated { requested: after, low_water }` if `after < low_water`;
    /// `CommitBeyondEnd` if `after` is past the end.
    fn replay(&self, after: Offset, max: usize) -> Result<Vec<BusRecord>, BusError>;
    /// The consumer whose lag holds back reclamation (and a full publisher), if any.
    fn blocked_by(&self) -> Option<ConsumerLag>;
    /// Telemetry records this consumer never acknowledged because the bus dropped them.
    fn dropped(&self, consumer: &str) -> Result<u64, BusError>;
}

pub(crate) fn check_len(field: Field, len: usize, max: usize) -> Result<(), BusError> {
    if len > max {
        return Err(BusError::FieldTooLong { field, len, max });
    }
    Ok(())
}

pub(crate) fn validate_new(record: &NewRecord, max_payload: usize) -> Result<(), BusError> {
    check_len(Field::Topic, record.topic.len(), MAX_FIELD_BYTES)?;
    check_len(Field::Key, record.key.len(), MAX_FIELD_BYTES)?;
    check_len(Field::Payload, record.payload.len(), max_payload)
}

pub(crate) fn validate_consumer(
    committed: &BTreeMap<String, Offset>,
    consumer: &str,
) -> Result<(), BusError> {
    check_len(Field::ConsumerName, consumer.len(), MAX_FIELD_BYTES)?;
    if !committed.contains_key(consumer) && committed.len() >= MAX_CONSUMERS {
        return Err(BusError::ConsumerLimit { max: MAX_CONSUMERS });
    }
    Ok(())
}

/// Configuration of the in-memory bus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BusConfig {
    /// Bound on unacknowledged records (clamped to at least one).
    pub capacity_records: usize,
    /// Optional bound on unacknowledged bytes (topic + key + payload).
    pub capacity_bytes: Option<usize>,
    /// Bound on one record's payload.
    pub max_record_bytes: usize,
    pub policies: ClassPolicies,
    pub retention: RetentionPolicy,
}

impl BusConfig {
    pub fn new(capacity_records: usize) -> Self {
        Self {
            capacity_records: capacity_records.max(1),
            capacity_bytes: None,
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            policies: ClassPolicies::reject_all(),
            retention: RetentionPolicy::acked_only(),
        }
    }
}

/// The in-process reference bus.
///
/// Bounded: when the records no registered consumer has committed reach
/// the capacity, publishing follows the class policy. By default it refuses
/// ([`BackpressurePolicy::Reject`]), so a slow consumer slows producers
/// rather than silently losing events. An authoritative record is never
/// removed while some registered consumer has not committed it; only
/// telemetry may be evicted, and only under
/// [`BackpressurePolicy::DropOldestTelemetry`], with the drop counted per
/// consumer. Acknowledged history is kept as long as the [`RetentionPolicy`]
/// says; compaction advances the low-water mark.
#[derive(Clone, Debug)]
pub struct InMemoryBus {
    config: BusConfig,
    records: VecDeque<BusRecord>,
    next: u64,
    low_water: u64,
    committed: BTreeMap<String, Offset>,
    tick: u64,
    retained_bytes: usize,
    dropped: BTreeMap<String, u64>,
    total_dropped: u64,
}

impl InMemoryBus {
    /// Create an empty bus with no registered consumers and the default policies (reject,
    /// keep no acknowledged history). `capacity` is clamped to at least one.
    pub fn new(capacity: usize) -> Self {
        Self::with_config(BusConfig::new(capacity))
    }

    /// Create an empty bus from a configuration. The record capacity is clamped to at least one.
    ///
    /// A [`RetentionPolicy`] with no record or byte limit (age alone does not count) would keep acknowledged history
    /// forever, so this infallible constructor replaces it with
    /// [`RetentionPolicy::acked_only`]. Use [`InMemoryBus::try_with_config`] to have it refused.
    pub fn with_config(mut config: BusConfig) -> Self {
        if config.retention.is_unbounded() {
            config.retention = RetentionPolicy::acked_only();
        }
        config.capacity_records = config.capacity_records.max(1);
        Self {
            config,
            records: VecDeque::new(),
            next: 1,
            low_water: 0,
            committed: BTreeMap::new(),
            tick: 0,
            retained_bytes: 0,
            dropped: BTreeMap::new(),
            total_dropped: 0,
        }
    }

    /// Like [`InMemoryBus::with_config`], but refuses a configuration instead of adjusting it.
    ///
    /// # Errors
    /// `InvalidPolicy` with [`PolicyViolation::UnboundedRetention`] if the retention policy
    /// has no limit in any dimension.
    pub fn try_with_config(config: BusConfig) -> Result<Self, BusError> {
        if config.retention.is_unbounded() {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::UnboundedRetention,
            });
        }
        Ok(Self::with_config(config))
    }

    /// Register a consumer starting after `from`. See [`EventConsumer::register`].
    ///
    /// Registering an existing consumer replaces its offset, including when
    /// `from` moves backward, as long as `from` is not below the low-water mark.
    ///
    /// # Errors
    /// `CommitBeyondEnd` if `from` is past the last offset, `Truncated` if it is below the
    /// low-water mark, `ConsumerLimit`/`FieldTooLong` for an unbounded name or count. Nothing
    /// changes on error.
    pub fn register(&mut self, consumer: &str, from: Offset) -> Result<(), BusError> {
        let end = self.end_offset();
        if from > end {
            return Err(BusError::CommitBeyondEnd {
                requested: from,
                end,
            });
        }
        if from.0 < self.low_water {
            return Err(BusError::Truncated {
                requested: from,
                low_water: Offset(self.low_water),
            });
        }
        validate_consumer(&self.committed, consumer)?;
        self.committed.insert(consumer.to_owned(), from);
        self.dropped.entry(consumer.to_owned()).or_insert(0);
        Ok(())
    }

    /// Records currently retained (unacknowledged records plus acknowledged history).
    pub fn retained(&self) -> usize {
        self.records.len()
    }

    /// Bytes currently retained.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Unacknowledged records (those past the slowest consumer).
    pub fn depth(&self) -> usize {
        self.records.len() - self.history_len()
    }

    /// Telemetry records dropped by the bus in total.
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    /// The current logical tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Move the logical clock forward and apply age retention. The clock is caller-supplied;
    /// the bus never reads wall time.
    ///
    /// # Errors
    /// `ClockRegression` if `now` is behind the current tick.
    pub fn advance_tick(&mut self, now: u64) -> Result<Compaction, BusError> {
        if now < self.tick {
            return Err(BusError::ClockRegression {
                current: self.tick,
                requested: now,
            });
        }
        self.tick = now;
        Ok(self.compact())
    }

    /// Apply the retention policy to acknowledged history and advance the low-water mark.
    pub fn compact(&mut self) -> Compaction {
        let floor = self.floor();
        let retention = self.config.retention;
        let mut history = self.records.partition_point(|r| r.offset <= floor);
        let mut history_bytes: usize = self
            .records
            .iter()
            .take(history)
            .map(BusRecord::size_bytes)
            .sum();
        let mut removed = 0u64;
        while history > 0 {
            let Some(front) = self.records.front() else {
                break;
            };
            let expired = retention.max_records.is_some_and(|max| history > max)
                || retention.max_bytes.is_some_and(|max| history_bytes > max)
                || retention
                    .max_age_ticks
                    .is_some_and(|max| self.tick.saturating_sub(front.tick) > max);
            if !expired {
                break;
            }
            let size = front.size_bytes();
            self.records.pop_front();
            self.retained_bytes -= size;
            history -= 1;
            history_bytes -= size;
            removed += 1;
        }
        self.refresh_low_water();
        Compaction {
            removed_records: removed,
            removed_segments: 0,
            low_water: Offset(self.low_water),
        }
    }

    fn end_offset(&self) -> Offset {
        Offset(self.next - 1)
    }

    /// Everything at or below this offset is acknowledged by every registered consumer.
    fn floor(&self) -> Offset {
        self.committed
            .values()
            .min()
            .copied()
            .unwrap_or_else(|| self.end_offset())
    }

    fn history_len(&self) -> usize {
        let floor = self.floor();
        self.records.partition_point(|r| r.offset <= floor)
    }

    fn depth_bytes(&self) -> usize {
        let skip = self.history_len();
        self.records
            .iter()
            .skip(skip)
            .map(BusRecord::size_bytes)
            .sum()
    }

    fn refresh_low_water(&mut self) {
        self.low_water = self
            .records
            .front()
            .map_or(self.next - 1, |front| front.offset.0 - 1);
    }

    fn full_error(&self, limit: Limit) -> BusError {
        let (capacity, depth) = match limit {
            Limit::Records => (self.config.capacity_records, self.depth()),
            Limit::Bytes => (self.config.capacity_bytes.unwrap_or(0), self.depth_bytes()),
        };
        BusError::Full {
            limit,
            capacity,
            depth,
            blocked_by: self.blocked_by_impl().map(|lag| lag.consumer),
        }
    }

    fn check_room(&self, size: usize) -> Result<(), BusError> {
        if self.depth() >= self.config.capacity_records {
            return Err(self.full_error(Limit::Records));
        }
        if let Some(capacity) = self.config.capacity_bytes {
            if self.depth_bytes() + size > capacity {
                return Err(self.full_error(Limit::Bytes));
            }
        }
        Ok(())
    }

    /// Evict the oldest unacknowledged telemetry record. Authoritative records are skipped.
    fn evict_oldest_unacked_telemetry(&mut self) -> bool {
        let skip = self.history_len();
        let Some(index) = self
            .records
            .iter()
            .enumerate()
            .skip(skip)
            .find(|(_, r)| r.class == EventClass::Telemetry)
            .map(|(index, _)| index)
        else {
            return false;
        };
        let Some(record) = self.records.remove(index) else {
            return false;
        };
        self.retained_bytes -= record.size_bytes();
        for (consumer, committed) in &self.committed {
            if *committed < record.offset {
                *self.dropped.entry(consumer.clone()).or_insert(0) += 1;
            }
        }
        self.total_dropped += 1;
        self.refresh_low_water();
        true
    }

    fn blocked_by_impl(&self) -> Option<ConsumerLag> {
        slowest(&self.committed, self.end_offset())
    }

    fn committed_of(&self, consumer: &str) -> Result<Offset, BusError> {
        self.committed
            .get(consumer)
            .copied()
            .ok_or_else(|| BusError::UnknownConsumer {
                consumer: consumer.to_owned(),
            })
    }
}

impl EventProducer for InMemoryBus {
    /// Append a record and return its assigned offset after compacting history.
    ///
    /// When the unacknowledged records fill the capacity the class policy decides:
    /// `Reject` returns [`BusError::Full`]; `DropOldestTelemetry` (telemetry class only)
    /// evicts the oldest unacknowledged telemetry record, never an authoritative one, and
    /// rejects if there is none. A refused publish consumes no offset.
    fn publish(&mut self, record: NewRecord) -> Result<Offset, BusError> {
        validate_new(&record, self.config.max_record_bytes)?;
        self.compact();
        let size = record.topic.len() + record.key.len() + record.payload.len();
        let drops =
            self.config.policies.for_class(record.class) == BackpressurePolicy::DropOldestTelemetry;
        loop {
            match self.check_room(size) {
                Ok(()) => break,
                Err(full) => {
                    if drops && self.evict_oldest_unacked_telemetry() {
                        continue;
                    }
                    return Err(full);
                }
            }
        }
        let offset = Offset(self.next);
        self.next += 1;
        self.retained_bytes += size;
        self.records.push_back(BusRecord::from_parts(
            offset,
            self.tick,
            record.class,
            record.topic,
            record.key,
            record.payload,
        ));
        self.refresh_low_water();
        Ok(offset)
    }
}

impl EventConsumer for InMemoryBus {
    fn register(&mut self, consumer: &str, from: Offset) -> Result<(), BusError> {
        InMemoryBus::register(self, consumer, from)
    }

    /// Return up to `max` retained records after the consumer's committed
    /// offset, in offset order, without advancing it. Zero requests no records.
    /// Returns `BusError::UnknownConsumer` if the consumer is not registered.
    fn poll(&self, consumer: &str, max: usize) -> Result<Vec<BusRecord>, BusError> {
        let after = self.committed_of(consumer)?;
        Ok(self
            .records
            .iter()
            .filter(|record| record.offset > after)
            .take(max)
            .cloned()
            .collect())
    }

    /// Acknowledge records through `upto`, inclusive, and compact history. An older offset
    /// does not rewind progress. Returns `BusError::CommitBeyondEnd` for a future offset or
    /// `BusError::UnknownConsumer` for an unregistered consumer.
    fn commit(&mut self, consumer: &str, upto: Offset) -> Result<(), BusError> {
        let end = self.end_offset();
        if upto > end {
            return Err(BusError::CommitBeyondEnd {
                requested: upto,
                end,
            });
        }
        let committed =
            self.committed
                .get_mut(consumer)
                .ok_or_else(|| BusError::UnknownConsumer {
                    consumer: consumer.to_owned(),
                })?;
        // Commits are monotone: an old commit arriving late cannot rewind.
        if upto > *committed {
            *committed = upto;
        }
        self.compact();
        Ok(())
    }

    fn committed(&self, consumer: &str) -> Result<Offset, BusError> {
        self.committed_of(consumer)
    }

    fn lag(&self, consumer: &str) -> Result<u64, BusError> {
        Ok(self.end_offset().0 - self.committed_of(consumer)?.0)
    }

    fn end(&self) -> Offset {
        self.end_offset()
    }

    fn low_water(&self) -> Offset {
        Offset(self.low_water)
    }

    fn replay(&self, after: Offset, max: usize) -> Result<Vec<BusRecord>, BusError> {
        let end = self.end_offset();
        if after > end {
            return Err(BusError::CommitBeyondEnd {
                requested: after,
                end,
            });
        }
        if after.0 < self.low_water {
            return Err(BusError::Truncated {
                requested: after,
                low_water: Offset(self.low_water),
            });
        }
        Ok(self
            .records
            .iter()
            .filter(|record| record.offset > after)
            .take(max)
            .cloned()
            .collect())
    }

    fn blocked_by(&self) -> Option<ConsumerLag> {
        self.blocked_by_impl()
    }

    fn dropped(&self, consumer: &str) -> Result<u64, BusError> {
        self.committed_of(consumer)?;
        Ok(self.dropped.get(consumer).copied().unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(class: EventClass, key: &str) -> NewRecord {
        NewRecord {
            class,
            topic: "t".into(),
            key: key.into(),
            payload: key.as_bytes().to_vec(),
        }
    }

    #[test]
    fn authority_follows_the_class_and_telemetry_cannot_be_promoted() {
        assert_eq!(EventClass::Projection.authority(), Authority::Authoritative);
        assert_eq!(EventClass::Telemetry.authority(), Authority::Diagnostic);
        let mut bus = InMemoryBus::new(4);
        bus.register("c", Offset(0)).unwrap();
        bus.publish(rec(EventClass::Telemetry, "x")).unwrap();
        let seen = bus.poll("c", 1).unwrap().remove(0);
        assert_eq!(seen.authority, Authority::Diagnostic);
        assert_eq!(
            NewRecord::reclassify(&seen, EventClass::Projection),
            Err(BusError::TelemetryPromotion { offset: Offset(1) })
        );
        assert_eq!(NewRecord::republish(&seen).class, EventClass::Telemetry);
        let demoted = NewRecord::reclassify(
            &BusRecord {
                class: EventClass::Projection,
                authority: Authority::Authoritative,
                ..seen
            },
            EventClass::Telemetry,
        )
        .unwrap();
        assert_eq!(demoted.class, EventClass::Telemetry);
    }

    #[test]
    fn oversized_fields_and_consumer_count_are_bounded() {
        let mut bus = InMemoryBus::new(2);
        let mut big = rec(EventClass::Projection, "k");
        big.topic = "x".repeat(MAX_FIELD_BYTES + 1);
        assert!(matches!(
            bus.publish(big),
            Err(BusError::FieldTooLong {
                field: Field::Topic,
                ..
            })
        ));
        for index in 0..MAX_CONSUMERS {
            bus.register(&format!("c{index}"), Offset(0)).unwrap();
        }
        assert_eq!(
            bus.register("one-too-many", Offset(0)),
            Err(BusError::ConsumerLimit { max: MAX_CONSUMERS })
        );
        bus.register("c0", Offset(0)).unwrap();
    }

    #[test]
    fn eviction_keeps_low_water_consistent() {
        let mut config = BusConfig::new(2);
        config.policies = ClassPolicies::drop_oldest_telemetry();
        let mut bus = InMemoryBus::with_config(config);
        bus.register("c", Offset(0)).unwrap();
        bus.publish(rec(EventClass::Telemetry, "a")).unwrap();
        bus.publish(rec(EventClass::Projection, "b")).unwrap();
        bus.publish(rec(EventClass::Telemetry, "c")).unwrap();
        assert_eq!(bus.low_water(), Offset(1));
        assert_eq!(bus.dropped("c"), Ok(1));
        assert_eq!(bus.total_dropped(), 1);
        assert!(matches!(
            bus.publish(rec(EventClass::Projection, "d")),
            Err(BusError::Full { .. })
        ));
    }
}
