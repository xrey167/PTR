use std::collections::{BTreeMap, VecDeque};
use std::fmt;

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

/// A record offered to the bus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewRecord {
    pub class: EventClass,
    pub topic: String,
    /// Partitioning/deduplication key, for example a capsule id.
    pub key: String,
    pub payload: Vec<u8>,
}

/// A record as a consumer sees it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BusRecord {
    pub offset: Offset,
    pub class: EventClass,
    pub topic: String,
    pub key: String,
    pub payload: Vec<u8>,
}

/// Refusals from the bus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BusError {
    /// The bus holds `capacity` unconsumed records; the producer must wait.
    /// Refusing is the backpressure: nothing is dropped to make room.
    Full { capacity: usize },
    /// The consumer is not registered.
    UnknownConsumer { consumer: String },
    /// A consumer tried to commit, or register, past the last record. A
    /// registration offset is the consumer's commit position, so it obeys the
    /// same bound.
    CommitBeyondEnd { requested: Offset, end: Offset },
}

impl BusError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Full { .. } => "PTR_BUS_FULL",
            Self::UnknownConsumer { .. } => "PTR_BUS_UNKNOWN_CONSUMER",
            Self::CommitBeyondEnd { .. } => "PTR_BUS_COMMIT_BEYOND_END",
        }
    }
}

impl fmt::Display for BusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full { capacity } => write!(formatter, "bus is full at {capacity} records"),
            Self::UnknownConsumer { consumer } => {
                write!(formatter, "consumer {consumer:?} is not registered")
            }
            Self::CommitBeyondEnd { requested, end } => write!(
                formatter,
                "commit to offset {} is beyond the last offset {}",
                requested.0, end.0
            ),
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
    fn poll(&self, consumer: &str, max: usize) -> Result<Vec<BusRecord>, BusError>;
    fn commit(&mut self, consumer: &str, upto: Offset) -> Result<(), BusError>;
}

/// The in-process reference bus.
///
/// Bounded: when the records no registered consumer has committed reach
/// `capacity`, publishing refuses. Retention never discards a record some
/// registered consumer has not committed, so a slow consumer slows producers
/// rather than silently losing events.
#[derive(Clone, Debug)]
pub struct InMemoryBus {
    capacity: usize,
    records: VecDeque<BusRecord>,
    next: u64,
    committed: BTreeMap<String, Offset>,
}

impl InMemoryBus {
    /// Create an empty bus with no registered consumers. `capacity` bounds
    /// the retained record count and is clamped to at least one.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            records: VecDeque::new(),
            next: 1,
            committed: BTreeMap::new(),
        }
    }

    /// Register a consumer starting after `from`. A consumer registered at
    /// `Offset(0)` sees every retained record.
    ///
    /// Registering an existing consumer replaces its offset, including when
    /// `from` moves backward; already trimmed records cannot be recovered.
    ///
    /// # Errors
    /// Returns `BusError::CommitBeyondEnd` if `from` is past the last offset,
    /// exactly as [`EventConsumer::commit`] would, and changes nothing: a
    /// future offset would let retention trim, and the consumer skip, records
    /// it has never seen.
    pub fn register(&mut self, consumer: &str, from: Offset) -> Result<(), BusError> {
        let end = self.end();
        if from > end {
            return Err(BusError::CommitBeyondEnd {
                requested: from,
                end,
            });
        }
        self.committed.insert(consumer.to_owned(), from);
        Ok(())
    }

    /// Records currently retained.
    pub fn retained(&self) -> usize {
        self.records.len()
    }

    fn end(&self) -> Offset {
        Offset(self.next - 1)
    }

    fn trim(&mut self) {
        let floor = self
            .committed
            .values()
            .min()
            .copied()
            .unwrap_or_else(|| self.end());
        while self
            .records
            .front()
            .is_some_and(|record| record.offset <= floor)
        {
            self.records.pop_front();
        }
    }
}

impl EventProducer for InMemoryBus {
    /// Append a record and return its assigned offset after trimming records
    /// committed by all consumers. With no consumers, prior records are trimmed.
    /// Returns `BusError::Full` if the retained records still fill capacity.
    fn publish(&mut self, record: NewRecord) -> Result<Offset, BusError> {
        self.trim();
        if self.records.len() >= self.capacity {
            return Err(BusError::Full {
                capacity: self.capacity,
            });
        }
        let offset = Offset(self.next);
        self.next += 1;
        self.records.push_back(BusRecord {
            offset,
            class: record.class,
            topic: record.topic,
            key: record.key,
            payload: record.payload,
        });
        Ok(offset)
    }
}

impl EventConsumer for InMemoryBus {
    /// Return up to `max` retained records after the consumer's committed
    /// offset, in offset order, without advancing it. Zero requests no records.
    /// Returns `BusError::UnknownConsumer` if the consumer is not registered.
    fn poll(&self, consumer: &str, max: usize) -> Result<Vec<BusRecord>, BusError> {
        let after = self
            .committed
            .get(consumer)
            .ok_or_else(|| BusError::UnknownConsumer {
                consumer: consumer.to_owned(),
            })?;
        Ok(self
            .records
            .iter()
            .filter(|record| record.offset > *after)
            .take(max)
            .cloned()
            .collect())
    }

    /// Acknowledge records through `upto`, inclusive, and trim records
    /// acknowledged by every consumer. An older offset does not rewind progress.
    /// Returns `BusError::CommitBeyondEnd` for a future offset or
    /// `BusError::UnknownConsumer` for an unregistered consumer.
    fn commit(&mut self, consumer: &str, upto: Offset) -> Result<(), BusError> {
        let end = self.end();
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
        self.trim();
        Ok(())
    }
}
