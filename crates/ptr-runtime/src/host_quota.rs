//! Host-wide byte quota for handle buffers (doc 42 rule 3b).
//!
//! [`HandlePolicy::max_inflight_bytes_per_principal`](crate::HandlePolicy) bounds one producer
//! inside **one** [`HandleTable`](crate::HandleTable), and a table serves one consumer stage. A
//! producer that serves several consumer stages (several tables) could therefore fill memory
//! although every table is below its cap, and several producers could fill one consumer. The
//! [`HostQuotaLedger`] closes that gap: one ledger per host, shared by every table through a
//! [`SharedHostQuota`], charges every admitted handle against
//!
//! - the **producer worker session** (`max_bytes_per_producer_session`),
//! - the **consumer worker session** (`max_bytes_per_consumer_session`) and
//! - the **whole host** (`max_bytes_host`).
//!
//! Charges are the **allocated footprint**, not the logical `byte_len`: the amount handed to
//! [`HostQuotaLedger::try_charge`] is rounded **up** to the policy's `charge_granularity_bytes`
//! (page size, 4096 by default; the caller supplies the granularity of the carrier, until the
//! carriers supply their own footprint). A request of 0 bytes counts as one unit, so a handle
//! can never be free. Rounding uses checked arithmetic; an overflow is refused as
//! [`QuotaRefusal::Overflow`].
//!
//! Charges are keyed by `(producer session, consumer session)`, so a verified teardown releases
//! exactly what the two reaped domains held ([`HostQuotaLedger::release_pair`], which needs a
//! [`VerifiedTeardown`]). A charge is
//! **all-or-nothing**: a refused charge changes nothing. The ledger does not decide which notice
//! frees a charge; its caller does: [`HandleTable`](crate::HandleTable) releases a charge for a
//! carrier that is not shared with the notice that frees the buffer, and for a shared carrier
//! only in `release_after_teardown`, never at `Consumed`, `finish`, `abandon` or the deadline.
//!
//! **Trust.** [`QuotaCharge`], [`HostQuotaLedger::release`] and [`HostQuotaLedger::release_pair`]
//! are trusted in-process APIs: the only caller is the supervisor/authority process (through
//! [`HandleTable`](crate::HandleTable) and the supervisor's teardown path), never a worker or a
//! peer.
//!
//! **Orphaned pairs.** A table that is dropped leaves its charges behind, and a stale pair would
//! occupy `max_tracked_sessions` forever. The supervisor must therefore list
//! [`HostQuotaLedger::tracked_pairs`] and release, with a [`VerifiedTeardown`], every pair whose
//! domains it has reaped and whose tables are gone. Releases are identity-safe: a pair release
//! starts a new generation for that pair, so a charge taken before it (still held by a live
//! table) is a counted no-op afterwards and can never debit a later charge of the same pair.
//!
//! The ledger is **in-process**: it bounds the tables of one authority process, which owns all
//! the tables of a host. Its maps are bounded by `max_tracked_sessions` each, and an entry is
//! dropped when its balance reaches zero.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::handle_admission::WorkerSession;

/// The default charge granularity: one page.
pub const DEFAULT_CHARGE_GRANULARITY_BYTES: u64 = 4096;

/// The bounds of a [`HostQuotaLedger`]. Every bound is non-zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostQuotaPolicy {
    /// Bytes charged to one producer worker session at most.
    pub max_bytes_per_producer_session: u64,
    /// Bytes charged to one consumer worker session at most.
    pub max_bytes_per_consumer_session: u64,
    /// Bytes charged across the host at most.
    pub max_bytes_host: u64,
    /// Bound of each tracked map (producer sessions, consumer sessions, session pairs). A charge
    /// that would add an entry beyond it is refused with [`QuotaRefusal::TooManySessions`].
    pub max_tracked_sessions: usize,
    /// Charges are rounded up to a multiple of this many bytes.
    pub charge_granularity_bytes: u64,
}

/// Which bound of a [`HostQuotaPolicy`] is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostQuotaPolicyViolation {
    ZeroProducerBytes,
    ZeroConsumerBytes,
    ZeroHostBytes,
    ZeroTrackedSessions,
    ZeroGranularity,
}

impl HostQuotaPolicy {
    /// A policy with the default granularity (one page).
    pub fn new(
        max_bytes_per_producer_session: u64,
        max_bytes_per_consumer_session: u64,
        max_bytes_host: u64,
        max_tracked_sessions: usize,
    ) -> Self {
        Self {
            max_bytes_per_producer_session,
            max_bytes_per_consumer_session,
            max_bytes_host,
            max_tracked_sessions,
            charge_granularity_bytes: DEFAULT_CHARGE_GRANULARITY_BYTES,
        }
    }

    /// Every bound must be non-zero.
    pub fn validate(&self) -> Result<(), HostQuotaPolicyViolation> {
        if self.max_bytes_per_producer_session == 0 {
            return Err(HostQuotaPolicyViolation::ZeroProducerBytes);
        }
        if self.max_bytes_per_consumer_session == 0 {
            return Err(HostQuotaPolicyViolation::ZeroConsumerBytes);
        }
        if self.max_bytes_host == 0 {
            return Err(HostQuotaPolicyViolation::ZeroHostBytes);
        }
        if self.max_tracked_sessions == 0 {
            return Err(HostQuotaPolicyViolation::ZeroTrackedSessions);
        }
        if self.charge_granularity_bytes == 0 {
            return Err(HostQuotaPolicyViolation::ZeroGranularity);
        }
        Ok(())
    }
}

/// Why a charge was refused. A refused charge changes nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotaRefusal {
    /// The producer session would exceed its cap.
    ProducerQuota {
        limit: u64,
        held: u64,
        requested: u64,
    },
    /// The consumer session would exceed its cap.
    ConsumerQuota {
        limit: u64,
        held: u64,
        requested: u64,
    },
    /// The host would exceed its cap.
    HostQuota {
        limit: u64,
        held: u64,
        requested: u64,
    },
    /// A new session (or session pair) would exceed `max_tracked_sessions`.
    TooManySessions { limit: usize },
    /// Rounding the request up to the granularity, or adding it to a balance, overflows `u64`.
    Overflow { requested: u64 },
}

/// A charge held against the ledger: the exact amount to hand back to
/// [`HostQuotaLedger::release`]. It is neither `Clone` nor `Copy` and its fields are private, so
/// a release consumes it and the same charge cannot be released twice through this API.
#[derive(Debug, Eq, PartialEq)]
pub struct QuotaCharge {
    producer_session: WorkerSession,
    consumer_session: WorkerSession,
    charged_bytes: u64,
    /// The generation of the pair the charge was taken in; a pair release, or the pair entry
    /// dropping to zero, makes it stale.
    generation: u64,
}

impl QuotaCharge {
    pub fn producer_session(&self) -> WorkerSession {
        self.producer_session
    }

    pub fn consumer_session(&self) -> WorkerSession {
        self.consumer_session
    }

    /// The rounded footprint that was charged.
    pub fn charged_bytes(&self) -> u64 {
        self.charged_bytes
    }
}

/// The supervisor's proof that both worker domains of a session pair were terminated and
/// reaped and the old control-socket endpoint of each was drained (doc 42 rule 3b). Construct it
/// only where that was verified; it is the precondition of [`HostQuotaLedger::release_pair`],
/// which must never run while a live table still holds claims of the pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedTeardown {
    producer_session: WorkerSession,
    consumer_session: WorkerSession,
}

impl VerifiedTeardown {
    /// The pair `(producer_session, consumer_session)` has been reaped and drained.
    pub fn reaped_and_drained(
        producer_session: WorkerSession,
        consumer_session: WorkerSession,
    ) -> Self {
        Self {
            producer_session,
            consumer_session,
        }
    }
}

/// One tracked session pair and its balance, for the supervisor's sweep of orphaned pairs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrackedPair {
    pub producer_session: WorkerSession,
    pub consumer_session: WorkerSession,
    pub bytes: u64,
}

/// A release that does not match what is charged. Nothing is changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseFault {
    /// The pair holds less than the amount: a double release or a release of an unknown charge.
    ExceedsPairBalance { held: u64, released: u64 },
}

/// The byte balances of the host, for tests and metrics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HostQuotaUsage {
    pub host_bytes: u64,
    pub producer_sessions: usize,
    pub consumer_sessions: usize,
    pub session_pairs: usize,
    /// Releases refused because they did not match a charge (should stay 0).
    pub release_faults: u64,
}

/// The host-wide ledger. See the module documentation.
#[derive(Debug)]
pub struct HostQuotaLedger {
    policy: HostQuotaPolicy,
    host_bytes: u64,
    producers: BTreeMap<WorkerSession, u64>,
    consumers: BTreeMap<WorkerSession, u64>,
    pairs: BTreeMap<(WorkerSession, WorkerSession), PairBalance>,
    /// Bumped whenever a pair entry is dropped (a pair release, or its balance reaching zero); a
    /// pair entry created later carries the new value.
    generation: u64,
    release_faults: u64,
}

#[derive(Clone, Copy, Debug)]
struct PairBalance {
    bytes: u64,
    generation: u64,
}

impl HostQuotaLedger {
    pub fn new(policy: HostQuotaPolicy) -> Result<Self, HostQuotaPolicyViolation> {
        policy.validate()?;
        Ok(Self {
            policy,
            host_bytes: 0,
            producers: BTreeMap::new(),
            consumers: BTreeMap::new(),
            pairs: BTreeMap::new(),
            generation: 0,
            release_faults: 0,
        })
    }

    pub fn policy(&self) -> &HostQuotaPolicy {
        &self.policy
    }

    /// `bytes` rounded up to the granularity; 0 counts as one unit. `None` on overflow.
    pub fn footprint_of(&self, bytes: u64) -> Option<u64> {
        let unit = self.policy.charge_granularity_bytes;
        let units = bytes.div_ceil(unit).max(1);
        units.checked_mul(unit)
    }

    /// Bytes charged to a producer session.
    pub fn producer_bytes(&self, session: WorkerSession) -> u64 {
        self.producers.get(&session).copied().unwrap_or(0)
    }

    /// Bytes charged to a consumer session.
    pub fn consumer_bytes(&self, session: WorkerSession) -> u64 {
        self.consumers.get(&session).copied().unwrap_or(0)
    }

    /// Bytes charged to one session pair.
    pub fn pair_bytes(&self, producer: WorkerSession, consumer: WorkerSession) -> u64 {
        self.pairs
            .get(&(producer, consumer))
            .map_or(0, |pair| pair.bytes)
    }

    /// Bytes charged across the host.
    pub fn host_bytes(&self) -> u64 {
        self.host_bytes
    }

    /// Every tracked session pair with its balance (at most `max_tracked_sessions`), for the
    /// supervisor to find pairs of dropped tables and release them with a [`VerifiedTeardown`].
    pub fn tracked_pairs(&self) -> Vec<TrackedPair> {
        self.pairs
            .iter()
            .map(|((producer, consumer), pair)| TrackedPair {
                producer_session: *producer,
                consumer_session: *consumer,
                bytes: pair.bytes,
            })
            .collect()
    }

    pub fn usage(&self) -> HostQuotaUsage {
        HostQuotaUsage {
            host_bytes: self.host_bytes,
            producer_sessions: self.producers.len(),
            consumer_sessions: self.consumers.len(),
            session_pairs: self.pairs.len(),
            release_faults: self.release_faults,
        }
    }

    /// Charge `bytes` (rounded up to the granularity) against the producer session, the consumer
    /// session and the host, or refuse and change nothing.
    pub fn try_charge(
        &mut self,
        producer: WorkerSession,
        consumer: WorkerSession,
        bytes: u64,
    ) -> Result<QuotaCharge, QuotaRefusal> {
        let policy = self.policy;
        let amount = self
            .footprint_of(bytes)
            .ok_or(QuotaRefusal::Overflow { requested: bytes })?;
        let held_producer = self.producer_bytes(producer);
        let held_consumer = self.consumer_bytes(consumer);
        let held_pair = self.pair_bytes(producer, consumer);
        let overflow = QuotaRefusal::Overflow { requested: amount };
        let new_producer = held_producer.checked_add(amount).ok_or(overflow)?;
        let new_consumer = held_consumer.checked_add(amount).ok_or(overflow)?;
        let new_pair = held_pair.checked_add(amount).ok_or(overflow)?;
        let new_host = self.host_bytes.checked_add(amount).ok_or(overflow)?;
        let limit = policy.max_tracked_sessions;
        if (held_producer == 0 && self.producers.len() >= limit)
            || (held_consumer == 0 && self.consumers.len() >= limit)
            || (held_pair == 0 && self.pairs.len() >= limit)
        {
            return Err(QuotaRefusal::TooManySessions { limit });
        }
        if new_producer > policy.max_bytes_per_producer_session {
            return Err(QuotaRefusal::ProducerQuota {
                limit: policy.max_bytes_per_producer_session,
                held: held_producer,
                requested: amount,
            });
        }
        if new_consumer > policy.max_bytes_per_consumer_session {
            return Err(QuotaRefusal::ConsumerQuota {
                limit: policy.max_bytes_per_consumer_session,
                held: held_consumer,
                requested: amount,
            });
        }
        if new_host > policy.max_bytes_host {
            return Err(QuotaRefusal::HostQuota {
                limit: policy.max_bytes_host,
                held: self.host_bytes,
                requested: amount,
            });
        }
        self.producers.insert(producer, new_producer);
        self.consumers.insert(consumer, new_consumer);
        let generation = self
            .pairs
            .get(&(producer, consumer))
            .map_or(self.generation, |pair| pair.generation);
        self.pairs.insert(
            (producer, consumer),
            PairBalance {
                bytes: new_pair,
                generation,
            },
        );
        self.host_bytes = new_host;
        Ok(QuotaCharge {
            producer_session: producer,
            consumer_session: consumer,
            charged_bytes: amount,
            generation,
        })
    }

    /// Hand back one charge (trusted in-process API). A release the pair does not hold, or of a
    /// charge whose pair was released since (a stale generation), is refused, counted in
    /// [`HostQuotaUsage::release_faults`] and changes no balance (never a double credit, never a
    /// debit of a later charge).
    pub fn release(&mut self, charge: QuotaCharge) -> Result<(), ReleaseFault> {
        let current = self
            .pairs
            .get(&(charge.producer_session, charge.consumer_session))
            .map(|pair| pair.generation);
        if current != Some(charge.generation) {
            self.release_faults = self.release_faults.saturating_add(1);
            return Err(ReleaseFault::ExceedsPairBalance {
                held: 0,
                released: charge.charged_bytes,
            });
        }
        self.release_amount(
            charge.producer_session,
            charge.consumer_session,
            charge.charged_bytes,
        )
    }

    /// Hand back everything the session pair holds, and return the amount (0 for a pair that
    /// holds nothing). **Supervisor only**, after both domains of the pair were reaped and the
    /// old control-socket ends drained (`proof`), and never while a live table still holds
    /// claims of the pair: the pair starts a new generation, so those claims' later releases
    /// become counted no-ops, but their buffers would be recredited while still claimed.
    pub fn release_pair(&mut self, proof: VerifiedTeardown) -> u64 {
        let (producer, consumer) = (proof.producer_session, proof.consumer_session);
        let held = self.pair_bytes(producer, consumer);
        if held == 0 {
            return 0;
        }
        match self.release_amount(producer, consumer, held) {
            Ok(()) => held,
            Err(_) => 0,
        }
    }

    fn release_amount(
        &mut self,
        producer: WorkerSession,
        consumer: WorkerSession,
        amount: u64,
    ) -> Result<(), ReleaseFault> {
        let held = self.pair_bytes(producer, consumer);
        if held < amount
            || self.host_bytes < amount
            || self.producer_bytes(producer) < amount
            || self.consumer_bytes(consumer) < amount
        {
            self.release_faults = self.release_faults.saturating_add(1);
            return Err(ReleaseFault::ExceedsPairBalance {
                held,
                released: amount,
            });
        }
        self.host_bytes -= amount;
        debit(&mut self.producers, producer, amount);
        debit(&mut self.consumers, consumer, amount);
        if let Some(pair) = self.pairs.get_mut(&(producer, consumer)) {
            pair.bytes = pair.bytes.saturating_sub(amount);
            if pair.bytes == 0 {
                self.pairs.remove(&(producer, consumer));
                // A pair entry created later must not accept a stale charge of the old one.
                self.generation = self.generation.wrapping_add(1);
            }
        }
        Ok(())
    }
}

/// Subtract `amount` (already checked to be held) and drop the entry at zero.
fn debit<K: Ord + Copy>(map: &mut BTreeMap<K, u64>, key: K, amount: u64) {
    if let Some(balance) = map.get_mut(&key) {
        *balance = balance.saturating_sub(amount);
        if *balance == 0 {
            map.remove(&key);
        }
    }
}

/// A cloneable handle on one [`HostQuotaLedger`], shared by every
/// [`HandleTable`](crate::HandleTable) of the host
/// ([`HandleTable::with_host_quota`](crate::HandleTable::with_host_quota)).
///
/// A poisoned lock is recovered: every ledger operation validates before it mutates and does not
/// panic, so the state behind a poisoned lock is still consistent.
#[derive(Clone, Debug)]
pub struct SharedHostQuota {
    inner: Arc<Mutex<HostQuotaLedger>>,
}

impl SharedHostQuota {
    pub fn new(policy: HostQuotaPolicy) -> Result<Self, HostQuotaPolicyViolation> {
        Ok(Self {
            inner: Arc::new(Mutex::new(HostQuotaLedger::new(policy)?)),
        })
    }

    fn lock(&self) -> MutexGuard<'_, HostQuotaLedger> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// See [`HostQuotaLedger::try_charge`].
    pub fn try_charge(
        &self,
        producer: WorkerSession,
        consumer: WorkerSession,
        bytes: u64,
    ) -> Result<QuotaCharge, QuotaRefusal> {
        self.lock().try_charge(producer, consumer, bytes)
    }

    /// See [`HostQuotaLedger::release`].
    pub fn release(&self, charge: QuotaCharge) -> Result<(), ReleaseFault> {
        self.lock().release(charge)
    }

    /// See [`HostQuotaLedger::release_pair`].
    pub fn release_pair(&self, proof: VerifiedTeardown) -> u64 {
        self.lock().release_pair(proof)
    }

    /// See [`HostQuotaLedger::tracked_pairs`].
    pub fn tracked_pairs(&self) -> Vec<TrackedPair> {
        self.lock().tracked_pairs()
    }

    pub fn producer_bytes(&self, session: WorkerSession) -> u64 {
        self.lock().producer_bytes(session)
    }

    pub fn consumer_bytes(&self, session: WorkerSession) -> u64 {
        self.lock().consumer_bytes(session)
    }

    pub fn pair_bytes(&self, producer: WorkerSession, consumer: WorkerSession) -> u64 {
        self.lock().pair_bytes(producer, consumer)
    }

    pub fn host_bytes(&self) -> u64 {
        self.lock().host_bytes()
    }

    pub fn usage(&self) -> HostQuotaUsage {
        self.lock().usage()
    }

    pub fn policy(&self) -> HostQuotaPolicy {
        *self.lock().policy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P1: WorkerSession = WorkerSession(1);
    const P2: WorkerSession = WorkerSession(2);
    const C1: WorkerSession = WorkerSession(11);
    const C2: WorkerSession = WorkerSession(12);

    fn policy() -> HostQuotaPolicy {
        HostQuotaPolicy::new(8192, 8192, 12288, 4)
    }

    /// What a caller that copied a charge's identity would hold (the type itself cannot be
    /// copied outside this module).
    fn duplicate_of(charge: &QuotaCharge) -> QuotaCharge {
        QuotaCharge {
            producer_session: charge.producer_session,
            consumer_session: charge.consumer_session,
            charged_bytes: charge.charged_bytes,
            generation: charge.generation,
        }
    }

    fn ledger() -> HostQuotaLedger {
        HostQuotaLedger::new(policy()).expect("valid")
    }

    #[test]
    fn every_zero_bound_is_refused() {
        let ok = policy();
        let cases = [
            (
                HostQuotaPolicy {
                    max_bytes_per_producer_session: 0,
                    ..ok
                },
                HostQuotaPolicyViolation::ZeroProducerBytes,
            ),
            (
                HostQuotaPolicy {
                    max_bytes_per_consumer_session: 0,
                    ..ok
                },
                HostQuotaPolicyViolation::ZeroConsumerBytes,
            ),
            (
                HostQuotaPolicy {
                    max_bytes_host: 0,
                    ..ok
                },
                HostQuotaPolicyViolation::ZeroHostBytes,
            ),
            (
                HostQuotaPolicy {
                    max_tracked_sessions: 0,
                    ..ok
                },
                HostQuotaPolicyViolation::ZeroTrackedSessions,
            ),
            (
                HostQuotaPolicy {
                    charge_granularity_bytes: 0,
                    ..ok
                },
                HostQuotaPolicyViolation::ZeroGranularity,
            ),
        ];
        for (policy, violation) in cases {
            assert_eq!(HostQuotaLedger::new(policy).err(), Some(violation));
        }
    }

    #[test]
    fn rounds_up_to_the_granularity_and_counts_zero_as_one_unit() {
        let mut ledger = ledger();
        assert_eq!(ledger.footprint_of(0), Some(4096));
        assert_eq!(ledger.footprint_of(1), Some(4096));
        assert_eq!(ledger.footprint_of(4096), Some(4096));
        assert_eq!(ledger.footprint_of(4097), Some(8192));
        let charge = ledger.try_charge(P1, C1, 1).expect("charged");
        assert_eq!(charge.charged_bytes(), 4096);
        assert_eq!(ledger.host_bytes(), 4096);
    }

    #[test]
    fn overflow_is_refused_and_changes_nothing() {
        let mut ledger = ledger();
        assert_eq!(
            ledger.try_charge(P1, C1, u64::MAX),
            Err(QuotaRefusal::Overflow {
                requested: u64::MAX
            })
        );
        assert_eq!(ledger.usage(), HostQuotaUsage::default());
    }

    #[test]
    fn refusals_are_typed_and_all_or_nothing() {
        let mut ledger = ledger();
        ledger.try_charge(P1, C1, 8192).expect("fills P1 and C1");
        assert!(matches!(
            ledger.try_charge(P1, C2, 1),
            Err(QuotaRefusal::ProducerQuota { .. })
        ));
        assert!(matches!(
            ledger.try_charge(P2, C1, 1),
            Err(QuotaRefusal::ConsumerQuota { .. })
        ));
        ledger.try_charge(P2, C2, 4096).expect("host at 12288");
        let before = ledger.usage();
        assert!(matches!(
            ledger.try_charge(WorkerSession(3), WorkerSession(13), 1),
            Err(QuotaRefusal::HostQuota { .. })
        ));
        assert_eq!(ledger.usage(), before);
        assert_eq!(ledger.producer_bytes(P1), 8192);
        assert_eq!(ledger.consumer_bytes(C2), 4096);
    }

    #[test]
    fn session_maps_are_bounded() {
        let mut ledger = HostQuotaLedger::new(HostQuotaPolicy::new(1 << 30, 1 << 30, 1 << 30, 2))
            .expect("valid");
        ledger.try_charge(P1, C1, 1).expect("1");
        ledger.try_charge(P2, C2, 1).expect("2");
        assert_eq!(
            ledger.try_charge(WorkerSession(3), C1, 1),
            Err(QuotaRefusal::TooManySessions { limit: 2 })
        );
        // An already tracked pair still charges.
        ledger.try_charge(P1, C1, 1).expect("tracked");
        // Releasing a pair frees its entries for new sessions.
        assert_eq!(
            ledger.release_pair(VerifiedTeardown::reaped_and_drained(P2, C2)),
            4096
        );
        ledger.try_charge(WorkerSession(3), C1, 1).expect("room");
    }

    #[test]
    fn release_returns_the_exact_charge_and_refuses_a_double_release() {
        let mut ledger = ledger();
        let charge = ledger.try_charge(P1, C1, 5000).expect("charged");
        assert_eq!(charge.charged_bytes(), 8192);
        let duplicate = duplicate_of(&charge);
        ledger.release(charge).expect("released");
        assert_eq!(ledger.usage(), HostQuotaUsage::default());
        assert!(ledger.release(duplicate).is_err());
        assert_eq!(ledger.usage().release_faults, 1);
        assert_eq!(ledger.host_bytes(), 0);
    }

    #[test]
    fn release_pair_frees_only_that_pair() {
        let mut ledger = ledger();
        ledger.try_charge(P1, C1, 4096).expect("a");
        ledger.try_charge(P1, C2, 4096).expect("b");
        assert_eq!(
            ledger.release_pair(VerifiedTeardown::reaped_and_drained(P1, C1)),
            4096
        );
        assert_eq!(
            ledger.release_pair(VerifiedTeardown::reaped_and_drained(P1, C1)),
            0
        );
        assert_eq!(ledger.producer_bytes(P1), 4096);
        assert_eq!(ledger.pair_bytes(P1, C2), 4096);
    }

    #[test]
    fn a_release_of_a_charge_before_a_pair_release_never_debits_a_later_charge() {
        let mut ledger = ledger();
        let old = ledger.try_charge(P1, C1, 1).expect("old");
        assert_eq!(
            ledger.release_pair(VerifiedTeardown::reaped_and_drained(P1, C1)),
            4096
        );
        assert_eq!(ledger.host_bytes(), 0);
        let new = ledger.try_charge(P1, C1, 1).expect("new");
        assert!(ledger.release(old).is_err());
        assert_eq!(ledger.pair_bytes(P1, C1), 4096);
        assert_eq!(ledger.host_bytes(), 4096);
        assert_eq!(ledger.usage().release_faults, 1);
        ledger.release(new).expect("the new charge releases");
        assert_eq!(ledger.usage().host_bytes, 0);
    }

    #[test]
    fn a_duplicate_release_after_the_pair_entry_dropped_debits_no_later_charge() {
        let mut ledger = ledger();
        let a = ledger.try_charge(P1, C1, 1).expect("a");
        let b = ledger.try_charge(P1, C1, 1).expect("b");
        let stale = duplicate_of(&a);
        ledger.release(a).expect("a released");
        ledger.release(b).expect("b released");
        assert!(ledger.tracked_pairs().is_empty());
        let c = ledger.try_charge(P1, C1, 1).expect("c");
        assert!(ledger.release(stale).is_err());
        assert_eq!(ledger.pair_bytes(P1, C1), 4096);
        assert_eq!(ledger.host_bytes(), 4096);
        assert_eq!(ledger.usage().release_faults, 1);
        ledger.release(c).expect("c releases");
        assert_eq!(ledger.usage().host_bytes, 0);
    }

    #[test]
    fn tracked_pairs_list_orphans_with_their_bytes_within_the_bound() {
        let mut ledger = ledger();
        ledger.try_charge(P1, C1, 1).expect("a");
        ledger.try_charge(P1, C1, 4096).expect("a2");
        ledger.try_charge(P2, C2, 1).expect("b");
        let pairs = ledger.tracked_pairs();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.len() <= ledger.policy().max_tracked_sessions);
        assert!(pairs.contains(&TrackedPair {
            producer_session: P1,
            consumer_session: C1,
            bytes: 8192
        }));
        for pair in pairs {
            ledger.release_pair(VerifiedTeardown::reaped_and_drained(
                pair.producer_session,
                pair.consumer_session,
            ));
        }
        assert!(ledger.tracked_pairs().is_empty());
        assert_eq!(ledger.usage(), HostQuotaUsage::default());
    }

    #[test]
    fn a_poisoned_lock_is_recovered() {
        let quota = SharedHostQuota::new(policy()).expect("valid");
        let clone = quota.clone();
        let _ = std::thread::spawn(move || {
            let _guard = clone.inner.lock().expect("lock");
            panic!("poison the lock");
        })
        .join();
        quota.try_charge(P1, C1, 1).expect("still usable");
        assert_eq!(quota.host_bytes(), 4096);
    }
}
