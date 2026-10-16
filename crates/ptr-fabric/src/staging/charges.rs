//! Footprint charging without a dependency on the runtime: the pool reports every allocated
//! footprint to a [`FootprintSink`] and the code that owns the quotas (the host glue, where
//! `ptr-runtime` is visible) implements it. [`ChargeBook`] is a small reference sink with
//! optional per-session caps, shared by clone so a test or a monitor can watch the same book the
//! pool charges.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// The sink refused a charge, with its own reason (a cap, a policy, a closed ledger).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChargeRefused {
    pub reason: String,
}

impl fmt::Display for ChargeRefused {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "charge refused: {}", self.reason)
    }
}

impl std::error::Error for ChargeRefused {}

/// Where the pool books allocated footprints (doc 42 rule 3b). A footprint is charged to the
/// producer and to the consumer of every buffer, as allocated, and released only when the pool
/// frees the buffer (or hands it to a new consumer). `release` must tolerate a session whose
/// account is already gone (the host may have released a recycled pair's charges itself): it
/// saturates, it never fails.
pub trait FootprintSink {
    fn charge(&mut self, session: u64, bytes: u64) -> Result<(), ChargeRefused>;
    fn release(&mut self, session: u64, bytes: u64);
}

/// A sink that books nothing and refuses nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoCharges;

impl FootprintSink for NoCharges {
    fn charge(&mut self, _session: u64, _bytes: u64) -> Result<(), ChargeRefused> {
        Ok(())
    }

    fn release(&mut self, _session: u64, _bytes: u64) {}
}

#[derive(Debug, Default)]
struct BookState {
    per_session: BTreeMap<u64, u64>,
    cap_per_session: Option<u64>,
    charges: u64,
    releases: u64,
}

/// A shared ledger of charged bytes per session. Clones share one book.
#[derive(Clone, Debug, Default)]
pub struct ChargeBook {
    state: Arc<Mutex<BookState>>,
}

impl ChargeBook {
    /// A book without a cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// A book that refuses a charge that would take a session above `cap` bytes.
    pub fn with_cap_per_session(cap: u64) -> Self {
        let book = Self::default();
        if let Ok(mut state) = book.state.lock() {
            state.cap_per_session = Some(cap);
        }
        book
    }

    /// Bytes charged to `session` now.
    pub fn charged(&self, session: u64) -> u64 {
        self.state
            .lock()
            .map(|state| state.per_session.get(&session).copied().unwrap_or(0))
            .unwrap_or(0)
    }

    /// Bytes charged over all sessions now.
    pub fn total(&self) -> u64 {
        self.state
            .lock()
            .map(|state| state.per_session.values().sum())
            .unwrap_or(0)
    }

    /// Charges accepted and releases made so far.
    pub fn counts(&self) -> (u64, u64) {
        self.state
            .lock()
            .map(|state| (state.charges, state.releases))
            .unwrap_or((0, 0))
    }
}

impl FootprintSink for ChargeBook {
    fn charge(&mut self, session: u64, bytes: u64) -> Result<(), ChargeRefused> {
        let mut state = self.state.lock().map_err(|_| ChargeRefused {
            reason: "the book is poisoned".to_owned(),
        })?;
        let used = state.per_session.get(&session).copied().unwrap_or(0);
        let total = used.checked_add(bytes).ok_or_else(|| ChargeRefused {
            reason: "charge overflow".to_owned(),
        })?;
        if let Some(cap) = state.cap_per_session {
            if total > cap {
                return Err(ChargeRefused {
                    reason: format!("session {session} would hold {total} of {cap} bytes"),
                });
            }
        }
        state.per_session.insert(session, total);
        state.charges += 1;
        Ok(())
    }

    fn release(&mut self, session: u64, bytes: u64) {
        if let Ok(mut state) = self.state.lock() {
            let left = state
                .per_session
                .get(&session)
                .copied()
                .unwrap_or(0)
                .saturating_sub(bytes);
            if left == 0 {
                state.per_session.remove(&session);
            } else {
                state.per_session.insert(session, left);
            }
            state.releases += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_book_charges_releases_and_saturates() {
        let mut book = ChargeBook::new();
        let watcher = book.clone();
        book.charge(1, 100).expect("charge");
        book.charge(1, 50).expect("charge");
        book.charge(2, 10).expect("charge");
        assert_eq!(
            (watcher.charged(1), watcher.charged(2), watcher.total()),
            (150, 10, 160)
        );
        book.release(1, 1000);
        assert_eq!(watcher.charged(1), 0);
        book.release(9, 5);
        assert_eq!(watcher.counts(), (3, 2));
    }

    #[test]
    fn a_capped_book_refuses_and_charges_nothing() {
        let mut book = ChargeBook::with_cap_per_session(100);
        book.charge(1, 60).expect("charge");
        assert!(book.charge(1, 41).is_err());
        assert_eq!(book.charged(1), 60);
        assert!(book.charge(2, u64::MAX).is_err());
        book.charge(1, 40).expect("exactly the cap");
    }
}
