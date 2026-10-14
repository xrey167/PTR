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
    /// The retention policy has no record or byte limit, so acknowledged history would grow
    /// without bound (an age limit alone does not bound it: records published and acknowledged
    /// at one logical tick never age).
    UnboundedRetention,
    /// A record-size bound whose frame (header, fixed fields, topic and key at their maximum,
    /// plus the payload) cannot be represented by the on-disk frame format's 32-bit length, so a
    /// record accepted under it could not be read back.
    RecordLimitTooLarge,
    /// A record, byte or backlog limit above its documented hard maximum
    /// ([`MAX_RETENTION_RECORDS`], [`MAX_RETENTION_BYTES`],
    /// [`MAX_FILE_LOG_UNACKED_RECORDS`](crate::MAX_FILE_LOG_UNACKED_RECORDS)): such a limit is
    /// nominally bounded but not in practice.
    LimitTooLarge,
}

/// Hard maximum of [`RetentionPolicy::max_records`].
pub const MAX_RETENTION_RECORDS: usize = 1 << 30;

/// Hard maximum of [`RetentionPolicy::max_bytes`] (1 TiB, or `usize::MAX` where that is smaller).
pub const MAX_RETENTION_BYTES: usize = {
    let tib = 1u64 << 40;
    if tib < usize::MAX as u64 {
        tib as usize
    } else {
        usize::MAX
    }
};

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
/// `None` leaves that dimension unlimited, but a record or byte limit must be set: age is measured
/// in caller-supplied logical ticks, so an age-only policy (or one with all three `None`,
/// [`RetentionPolicy::is_unbounded`]) would never expire history if the caller does not advance
/// the tick, or publishes many records at one tick, so
/// [`FileEventLog::open`](crate::FileEventLog::open) and
/// [`InMemoryBus::try_with_config`](crate::InMemoryBus::try_with_config) refuse it with
/// [`PolicyViolation::UnboundedRetention`] and [`InMemoryBus::with_config`](crate::InMemoryBus::with_config)
/// falls back to [`RetentionPolicy::acked_only`]. The default keeps no history
/// (`max_records == Some(0)`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionPolicy {
    pub max_records: Option<usize>,
    pub max_bytes: Option<usize>,
    /// Maximum age in caller-supplied logical ticks (no wall clock is read).
    pub max_age_ticks: Option<u64>,
}

impl RetentionPolicy {
    /// True when neither a record nor a byte limit is set, so acknowledged history is not
    /// bounded in storage. An age limit alone does not count: it depends on the caller advancing
    /// the logical tick.
    pub const fn is_unbounded(&self) -> bool {
        self.max_records.is_none() && self.max_bytes.is_none()
    }

    /// Check the policy against the bounds both bus backends require.
    ///
    /// # Errors
    /// `InvalidPolicy` with [`PolicyViolation::UnboundedRetention`] if neither a record nor a
    /// byte limit is set, or [`PolicyViolation::LimitTooLarge`] if one exceeds
    /// [`MAX_RETENTION_RECORDS`] / [`MAX_RETENTION_BYTES`].
    pub fn validate(&self) -> Result<(), BusError> {
        if self.is_unbounded() {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::UnboundedRetention,
            });
        }
        if self.max_records.is_some_and(|m| m > MAX_RETENTION_RECORDS)
            || self.max_bytes.is_some_and(|m| m > MAX_RETENTION_BYTES)
        {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::LimitTooLarge,
            });
        }
        Ok(())
    }

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

    /// Age-only retention. Alone it is **unbounded** ([`RetentionPolicy::is_unbounded`]) and is
    /// refused by the fallible constructors; combine it with a record or byte limit by setting
    /// `max_records` or `max_bytes` on the result.
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
    fn retention_beyond_the_hard_maxima_is_refused_and_the_maxima_are_accepted() {
        let too_large = Err(BusError::InvalidPolicy {
            violation: PolicyViolation::LimitTooLarge,
        });
        assert_eq!(
            RetentionPolicy::with_max_records(usize::MAX).validate(),
            too_large
        );
        assert_eq!(
            RetentionPolicy::with_max_bytes(MAX_RETENTION_BYTES.saturating_add(1)).validate(),
            if MAX_RETENTION_BYTES == usize::MAX {
                Ok(())
            } else {
                too_large
            }
        );
        assert_eq!(
            RetentionPolicy::with_max_records(MAX_RETENTION_RECORDS).validate(),
            Ok(())
        );
        assert_eq!(
            RetentionPolicy::with_max_bytes(MAX_RETENTION_BYTES).validate(),
            Ok(())
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
