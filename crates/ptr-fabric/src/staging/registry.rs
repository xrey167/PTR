//! The export registry (doc 42 rule 3a): the supervisor issues an [`ExportRecord`] for every
//! CUDA export, and admission refuses a locator that matches no live record.
//!
//! A record names the producer session, the one consumer session, the operation, the device and
//! the allocation size. The registry holds it until the one admission that consumes it (a
//! second admission of the same locator finds no record), until its expiry (the operation
//! deadline plus skew, after which it is removed deterministically), or until either named
//! session is reaped. A worker that copies a received locator into a newly signed handle for
//! another worker presents a different consumer session or operation, and finds no record.

use std::collections::BTreeMap;
use std::fmt;

use super::pool::SlotId;
use crate::ExportRecord;

/// The identity of one issued record.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExportId(pub u64);

/// What an admitted handle's locator claims, taken from the handle and from the identity the
/// supervisor knows the claimant by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImportClaim {
    /// The session that produced the handle (the one the supervisor authenticated its signature
    /// to). Two producers can export equal sizes to one consumer for one operation (fan-in);
    /// only the record of this producer matches.
    pub producer_session: u64,
    /// The session presenting the handle.
    pub consumer_session: u64,
    /// The handle's `parent_operation_id`.
    pub operation: u64,
    /// The device the locator names.
    pub device: u32,
    /// The allocation size the locator states.
    pub size: u64,
}

/// A record found for a claim, with the buffer behind it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmittedExport {
    pub id: ExportId,
    pub record: ExportRecord,
    pub slot: SlotId,
}

/// Why the registry refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistryRefusal {
    /// No live record matches the claim: never issued, issued to another consumer, another
    /// operation, device or size, already consumed, or its session reaped.
    NoRecord,
    /// A record matched but its deadline plus skew has passed; it was removed.
    Expired { id: ExportId },
    /// The registry holds as many records as it may.
    Full { max: usize },
}

impl fmt::Display for RegistryRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RegistryRefusal {}

struct Entry {
    record: ExportRecord,
    slot: SlotId,
    expires_at_ms: u64,
}

/// The live export records. Bounded; single-threaded.
pub struct ExportRegistry {
    max_records: usize,
    next: u64,
    records: BTreeMap<ExportId, Entry>,
}

impl ExportRegistry {
    /// A registry holding at most `max_records` records at once.
    pub fn new(max_records: usize) -> Self {
        Self {
            max_records,
            next: 0,
            records: BTreeMap::new(),
        }
    }

    /// Records held now.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The most records the registry holds at once.
    pub fn max_records(&self) -> usize {
        self.max_records
    }

    /// Whether `id` is still live.
    pub fn contains(&self, id: ExportId) -> bool {
        self.records.contains_key(&id)
    }

    /// Whether a record can be issued now (after sweeping the expired ones). Ask before
    /// acquiring a buffer, so that a full registry never strands a leased buffer.
    pub fn has_room(&mut self, now_ms: u64) -> bool {
        self.expire(now_ms);
        self.records.len() < self.max_records
    }

    /// Issue a record for `slot`, valid until `expires_at_ms` (the operation deadline plus the
    /// clock skew). Sweeps expired records first, so abandoned calls cannot fill the registry.
    pub fn issue(
        &mut self,
        record: ExportRecord,
        slot: SlotId,
        expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<ExportId, RegistryRefusal> {
        self.expire(now_ms);
        if self.records.len() >= self.max_records {
            return Err(RegistryRefusal::Full {
                max: self.max_records,
            });
        }
        self.next += 1;
        let id = ExportId(self.next);
        self.records.insert(
            id,
            Entry {
                record,
                slot,
                expires_at_ms,
            },
        );
        Ok(id)
    }

    /// Consume the oldest live record that matches `claim` exactly: this consumer session, this
    /// operation, this device and this size. The record is gone afterwards.
    pub fn admit(
        &mut self,
        claim: &ImportClaim,
        now_ms: u64,
    ) -> Result<AdmittedExport, RegistryRefusal> {
        let found = self
            .records
            .iter()
            .find(|(_, entry)| {
                entry.record.producer_session == claim.producer_session
                    && entry.record.consumer_session == claim.consumer_session
                    && entry.record.operation == claim.operation
                    && entry.record.device == claim.device
                    && entry.record.size == claim.size
            })
            .map(|(id, entry)| (*id, entry.expires_at_ms));
        let Some((id, expires_at_ms)) = found else {
            self.expire(now_ms);
            return Err(RegistryRefusal::NoRecord);
        };
        let entry = self.records.remove(&id).ok_or(RegistryRefusal::NoRecord)?;
        // Every other record past its deadline plus skew goes too, so that the registry does not
        // depend on `issue` being called to shed them.
        self.expire(now_ms);
        if now_ms > expires_at_ms {
            return Err(RegistryRefusal::Expired { id });
        }
        Ok(AdmittedExport {
            id,
            record: entry.record,
            slot: entry.slot,
        })
    }

    /// Remove the records whose deadline plus skew is before `now_ms`; returns their ids.
    pub fn expire(&mut self, now_ms: u64) -> Vec<ExportId> {
        let expired: Vec<ExportId> = self
            .records
            .iter()
            .filter(|(_, entry)| now_ms > entry.expires_at_ms)
            .map(|(id, _)| *id)
            .collect();
        for id in &expired {
            self.records.remove(id);
        }
        expired
    }

    /// A session was reaped: remove every record that names it as producer or consumer.
    pub fn on_reaped(&mut self, session: u64) -> Vec<ExportId> {
        let gone: Vec<ExportId> = self
            .records
            .iter()
            .filter(|(_, entry)| {
                entry.record.producer_session == session || entry.record.consumer_session == session
            })
            .map(|(id, _)| *id)
            .collect();
        for id in &gone {
            self.records.remove(id);
        }
        gone
    }

    /// Remove every record, for example when the guardian is gone.
    pub fn clear(&mut self) -> usize {
        let count = self.records.len();
        self.records.clear();
        count
    }
}

impl fmt::Debug for ExportRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExportRegistry")
            .field("records", &self.records.len())
            .field("max_records", &self.max_records)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> ExportRecord {
        ExportRecord {
            producer_session: 1,
            consumer_session: 2,
            operation: 7,
            device: 0,
            size: 4096,
        }
    }

    fn claim() -> ImportClaim {
        ImportClaim {
            producer_session: 1,
            consumer_session: 2,
            operation: 7,
            device: 0,
            size: 4096,
        }
    }

    #[test]
    fn a_matching_claim_consumes_the_record_once() {
        let mut registry = ExportRegistry::new(8);
        let id = registry.issue(record(), SlotId(1), 1000, 0).expect("issue");
        let admitted = registry.admit(&claim(), 10).expect("admit");
        assert_eq!(
            (admitted.id, admitted.slot, admitted.record),
            (id, SlotId(1), record())
        );
        assert_eq!(registry.admit(&claim(), 10), Err(RegistryRefusal::NoRecord));
        assert!(registry.is_empty());
    }

    #[test]
    fn a_locator_without_a_matching_live_record_is_refused() {
        let mut registry = ExportRegistry::new(8);
        assert_eq!(registry.admit(&claim(), 0), Err(RegistryRefusal::NoRecord));
        registry.issue(record(), SlotId(1), 1000, 0).expect("issue");
        for other in [
            ImportClaim {
                producer_session: 9,
                ..claim()
            },
            ImportClaim {
                consumer_session: 3,
                ..claim()
            },
            ImportClaim {
                operation: 8,
                ..claim()
            },
            ImportClaim {
                device: 1,
                ..claim()
            },
            ImportClaim {
                size: 8192,
                ..claim()
            },
        ] {
            assert_eq!(registry.admit(&other, 0), Err(RegistryRefusal::NoRecord));
        }
        // None of those refusals consumed the record.
        assert!(registry.admit(&claim(), 0).is_ok());
    }

    #[test]
    fn two_producers_to_one_consumer_are_told_apart() {
        let mut registry = ExportRegistry::new(8);
        let mut second = record();
        second.producer_session = 5;
        // Same consumer, operation, device and size: only the producer differs.
        let first_id = registry.issue(record(), SlotId(1), 1000, 0).expect("a");
        let second_id = registry.issue(second, SlotId(2), 1000, 0).expect("b");
        let for_second = ImportClaim {
            producer_session: 5,
            ..claim()
        };
        assert_eq!(
            registry.admit(&for_second, 0).expect("second").id,
            second_id
        );
        assert_eq!(
            registry.admit(&for_second, 0),
            Err(RegistryRefusal::NoRecord)
        );
        assert_eq!(registry.admit(&claim(), 0).expect("first").id, first_id);
    }

    #[test]
    fn admit_sweeps_other_expired_records() {
        let mut registry = ExportRegistry::new(8);
        let mut other = record();
        other.operation = 99;
        registry.issue(other, SlotId(2), 50, 0).expect("old");
        registry.issue(record(), SlotId(1), 1000, 0).expect("live");
        registry.admit(&claim(), 100).expect("admit");
        assert!(
            registry.is_empty(),
            "the expired record went with the admission"
        );
    }

    #[test]
    fn two_records_for_one_claim_are_consumed_oldest_first() {
        let mut registry = ExportRegistry::new(8);
        let first = registry.issue(record(), SlotId(1), 1000, 0).expect("first");
        let second = registry
            .issue(record(), SlotId(2), 1000, 0)
            .expect("second");
        assert_eq!(registry.admit(&claim(), 0).expect("one").id, first);
        assert_eq!(registry.admit(&claim(), 0).expect("two").id, second);
    }

    #[test]
    fn an_expired_record_is_refused_and_removed() {
        let mut registry = ExportRegistry::new(8);
        let id = registry.issue(record(), SlotId(1), 100, 0).expect("issue");
        assert_eq!(
            registry.admit(&claim(), 101),
            Err(RegistryRefusal::Expired { id })
        );
        assert!(registry.is_empty());
        let id = registry.issue(record(), SlotId(1), 100, 0).expect("issue");
        assert_eq!(registry.expire(100), Vec::<ExportId>::new());
        assert_eq!(registry.expire(101), vec![id]);
    }

    #[test]
    fn issue_sweeps_expired_records_and_is_bounded() {
        let mut registry = ExportRegistry::new(2);
        registry.issue(record(), SlotId(1), 100, 0).expect("a");
        registry.issue(record(), SlotId(2), 200, 0).expect("b");
        assert_eq!(
            registry.issue(record(), SlotId(3), 300, 50),
            Err(RegistryRefusal::Full { max: 2 })
        );
        // At 150 the first record has expired and made room.
        registry.issue(record(), SlotId(3), 300, 150).expect("c");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn reaping_either_session_removes_the_record() {
        let mut registry = ExportRegistry::new(8);
        registry.issue(record(), SlotId(1), 1000, 0).expect("a");
        let mut other = record();
        other.producer_session = 5;
        other.consumer_session = 6;
        registry.issue(other, SlotId(2), 1000, 0).expect("b");
        assert_eq!(registry.on_reaped(1).len(), 1);
        assert_eq!(registry.admit(&claim(), 0), Err(RegistryRefusal::NoRecord));
        assert_eq!(registry.on_reaped(6).len(), 1);
        assert!(registry.is_empty());
        assert_eq!(registry.on_reaped(6).len(), 0);
    }

    #[test]
    fn clear_empties_the_registry() {
        let mut registry = ExportRegistry::new(8);
        registry.issue(record(), SlotId(1), 1000, 0).expect("a");
        assert_eq!(registry.clear(), 1);
        assert!(registry.is_empty());
    }
}
