//! Typed backpressure and retention policy, shared by every bus implementation.

use std::collections::BTreeMap;

use crate::bus::{BusError, EventClass, Offset};

/// Whether a record is an authoritative projection or diagnostic telemetry.
///
/// Derived from [`EventClass`] only; a record cannot carry an authority its class does not
/// grant, and the bus API has no way to turn a telemetry record into an authoritative one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Authority {
    /// Restates something the ledger committed. Never dropped or evicted while a registered
    /// consumer has not acknowledged it.
    Authoritative,
    /// Execution diagnostics. May be dropped when the telemetry policy says so.
    Diagnostic,
}

/// What a bus does when a publish of a class meets a full bus.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BackpressurePolicy {
    /// Refuse the publish with [`BusError::Full`]; nothing is dropped.
    Reject,
    /// Make room by evicting the oldest unacknowledged telemetry record, counting the drop per
    /// consumer. Valid only for the telemetry class; if no telemetry can be evicted the publish
    /// is rejected.
    DropOldestTelemetry,
}

/// Backpressure policy per event class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClassPolicies {
    projection: BackpressurePolicy,
    telemetry: BackpressurePolicy,
}

impl ClassPolicies {
    /// Build a policy pair.
    ///
    /// # Errors
    /// [`BusError::InvalidPolicy`] if the projection class is given
    /// [`BackpressurePolicy::DropOldestTelemetry`]: authoritative records are never dropped.
    pub fn new(
        projection: BackpressurePolicy,
        telemetry: BackpressurePolicy,
    ) -> Result<Self, BusError> {
        if projection == BackpressurePolicy::DropOldestTelemetry {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::ProjectionMayNotDrop,
            });
        }
        Ok(Self {
            projection,
            telemetry,
        })
    }

    /// Reject for both classes (the default).
    pub const fn reject_all() -> Self {
        Self {
            projection: BackpressurePolicy::Reject,
            telemetry: BackpressurePolicy::Reject,
        }
    }

    /// Reject authoritative publishes, drop the oldest unacknowledged telemetry on pressure.
    pub const fn drop_oldest_telemetry() -> Self {
        Self {
            projection: BackpressurePolicy::Reject,
            telemetry: BackpressurePolicy::DropOldestTelemetry,
        }
    }

    pub const fn projection(&self) -> BackpressurePolicy {
        self.projection
    }

    pub const fn telemetry(&self) -> BackpressurePolicy {
        self.telemetry
    }

    pub const fn for_class(&self, class: EventClass) -> BackpressurePolicy {
        match class {
            EventClass::Projection => self.projection,
            EventClass::Telemetry => self.telemetry,
        }
    }
}

impl Default for ClassPolicies {
    fn default() -> Self {
        Self::reject_all()
    }
}

/// Why a policy or configuration was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyViolation {
    /// The projection class was given a dropping policy.
    ProjectionMayNotDrop,
    /// The backend cannot drop individual records (the file log is append-only).
    DropUnsupportedByBackend,
    /// A bound that must be positive was zero.
    ZeroBound,
}

/// Which bound a [`BusError::Full`] refers to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Limit {
    Records,
    Bytes,
}

/// Retention of **acknowledged history**, kept so a consumer can replay or a new consumer can
/// catch up. A record becomes history once every registered consumer has acknowledged it (or
/// immediately when there are no consumers). History is compacted from the oldest end as soon as
/// any limit is exceeded. Records some consumer has not acknowledged are not history: retention
/// never removes an unacknowledged authoritative record.
///
/// `None` leaves that dimension unlimited; the total is still bounded by the publish capacity
/// plus the other limits. The default keeps no history (`max_records == Some(0)`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionPolicy {
    pub max_records: Option<usize>,
    pub max_bytes: Option<usize>,
    /// Maximum age in caller-supplied logical ticks (no wall clock is read).
    pub max_age_ticks: Option<u64>,
}

impl RetentionPolicy {
    /// Keep no acknowledged history.
    pub const fn acked_only() -> Self {
        Self {
            max_records: Some(0),
            max_bytes: None,
            max_age_ticks: None,
        }
    }

    pub const fn with_max_records(max_records: usize) -> Self {
        Self {
            max_records: Some(max_records),
            max_bytes: None,
            max_age_ticks: None,
        }
    }

    pub const fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            max_records: None,
            max_bytes: Some(max_bytes),
            max_age_ticks: None,
        }
    }

    pub const fn with_max_age_ticks(max_age_ticks: u64) -> Self {
        Self {
            max_records: None,
            max_bytes: None,
            max_age_ticks: Some(max_age_ticks),
        }
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::acked_only()
    }
}

/// How far a consumer is behind the end of the log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsumerLag {
    pub consumer: String,
    pub committed: Offset,
    pub lag: u64,
}

/// What one compaction pass removed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Compaction {
    pub removed_records: u64,
    pub removed_segments: usize,
    /// The low-water mark after the pass; records at or below it are gone.
    pub low_water: Offset,
}

/// The slowest consumer: the one whose lag blocks reclamation and, when full, the publisher.
pub(crate) fn slowest(committed: &BTreeMap<String, Offset>, end: Offset) -> Option<ConsumerLag> {
    committed
        .iter()
        .min_by_key(|(_, offset)| **offset)
        .filter(|(_, offset)| **offset < end)
        .map(|(consumer, offset)| ConsumerLag {
            consumer: consumer.clone(),
            committed: *offset,
            lag: end.0 - offset.0,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_class_cannot_be_given_a_dropping_policy() {
        assert_eq!(
            ClassPolicies::new(
                BackpressurePolicy::DropOldestTelemetry,
                BackpressurePolicy::Reject
            ),
            Err(BusError::InvalidPolicy {
                violation: PolicyViolation::ProjectionMayNotDrop
            })
        );
        let ok = ClassPolicies::new(
            BackpressurePolicy::Reject,
            BackpressurePolicy::DropOldestTelemetry,
        )
        .unwrap();
        assert_eq!(ok, ClassPolicies::drop_oldest_telemetry());
        assert_eq!(
            ok.for_class(EventClass::Projection),
            BackpressurePolicy::Reject
        );
    }

    #[test]
    fn slowest_consumer_is_the_lowest_commit_and_none_when_caught_up() {
        let mut committed = BTreeMap::new();
        committed.insert("b".to_owned(), Offset(2));
        committed.insert("a".to_owned(), Offset(5));
        committed.insert("c".to_owned(), Offset(2));
        let lag = slowest(&committed, Offset(5)).unwrap();
        assert_eq!((lag.consumer.as_str(), lag.lag), ("b", 3));
        assert_eq!(slowest(&committed, Offset(2)), None);
        assert_eq!(slowest(&BTreeMap::new(), Offset(9)), None);
    }
}
