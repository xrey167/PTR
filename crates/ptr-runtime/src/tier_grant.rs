//! Project-scoped, single-use grants for tier transfers (in-process; no wire format yet).
//!
//! A [`TierGrant`] authorizes one transfer of one object of one project between two named
//! backends, for one receiver session, until a deadline. The controller consumes the grant in
//! the same critical section that publishes the destination replica, and remembers its nonce
//! in a bounded tombstone table until the deadline plus the configured skew, so the grant
//! cannot be replayed. The table is bounded per receiver (entries and bytes) and in the number
//! of receivers; a full table refuses with [`TierRuntimeError::TombstoneSaturated`] and never
//! evicts a live tombstone.
//!
//! Limits of this in-process form:
//! * The presenter [`SessionId`] must come from an authenticated session. Today it is an
//!   unauthenticated in-process value, so a peer that can choose its session ids can fill
//!   `max_receivers` and make other receivers' grants fail with `TombstoneSaturated`.
//! * An object's root digest is registered under one project only, and records are not keyed
//!   by (project, digest). A refusal with `ProjectMismatch` therefore tells the caller that
//!   the digest exists under another project (or unscoped).
//! * A grant is spent by the call that satisfies it: the leader of a transfer at its publish,
//!   a call that coalesced onto an in-flight transfer when the leader's result arrives, and a
//!   call that finds the destination replica already available at that moment. A grant
//!   already held by an in-flight call is refused with `GrantReplayed`.

use crate::tier::{TierRuntimeError, TierTransferAuthorization};
use ptr_storage::TierBackendId;
use ptr_types::{Digest, ProjectId, SessionId};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Accounted size of one tombstone: the 16-byte nonce, the 8-byte expiry and map overhead.
pub const TOMBSTONE_ENTRY_BYTES: usize = 48;

/// Time source for grant deadlines, in milliseconds. The controller reads it outside its own
/// lock, so an implementation may be slow but must not call back into the controller.
pub trait TierClock: Send + Sync {
    fn now_millis(&self) -> u64;
}

/// Wall clock, milliseconds since the Unix epoch. A clock before the epoch reads as zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemTierClock;

impl TierClock for SystemTierClock {
    fn now_millis(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| {
                u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
            })
    }
}

/// A clock moved by hand, for deterministic tests and replays.
#[derive(Debug, Default)]
pub struct ManualTierClock {
    millis: AtomicU64,
}

impl ManualTierClock {
    pub fn new(start_millis: u64) -> Self {
        Self {
            millis: AtomicU64::new(start_millis),
        }
    }

    pub fn set(&self, millis: u64) {
        self.millis.store(millis, Ordering::SeqCst);
    }

    pub fn advance(&self, millis: u64) {
        let _ = self
            .millis
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |now| {
                Some(now.saturating_add(millis))
            });
    }
}

impl TierClock for ManualTierClock {
    fn now_millis(&self) -> u64 {
        self.millis.load(Ordering::SeqCst)
    }
}

/// Permission to move one object between two backends, once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierGrant {
    /// The project the object belongs to. A grant of another project is refused.
    pub project: ProjectId,
    /// Root digest of the object to move.
    pub object: Digest,
    pub source: TierBackendId,
    pub destination: TierBackendId,
    /// The active-write binding the object is under, if any; rechecked just before publish.
    pub write_authorization: Option<TierTransferAuthorization>,
    /// The session that may present this grant.
    pub receiver: SessionId,
    /// Single-use identifier; unique per receiver while its tombstone lives.
    pub nonce: u128,
    /// Last millisecond (inclusive) at which the grant is valid.
    pub deadline_millis: u64,
}

/// Bounds of the grant machinery of one controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TierGrantLimits {
    /// Extra time a tombstone outlives the grant deadline, covering clock skew between peers.
    pub skew_millis: u64,
    /// Distinct receiver sessions with at least one live tombstone.
    pub max_receivers: usize,
    /// Live tombstones per receiver.
    pub max_tombstones_per_receiver: usize,
    /// Accounted bytes of live tombstones per receiver (`TOMBSTONE_ENTRY_BYTES` each).
    pub max_tombstone_bytes_per_receiver: usize,
    /// The furthest a grant's deadline may lie beyond the time it is presented; a longer
    /// grant is `GrantInvalid`, so no tombstone can outlive this plus the skew.
    pub max_grant_lifetime_millis: u64,
}

impl Default for TierGrantLimits {
    fn default() -> Self {
        Self {
            skew_millis: 5_000,
            max_receivers: 64,
            max_tombstones_per_receiver: 1_024,
            max_tombstone_bytes_per_receiver: 1_024 * TOMBSTONE_ENTRY_BYTES,
            max_grant_lifetime_millis: 3_600_000,
        }
    }
}

impl TierGrantLimits {
    pub(crate) fn validate(&self) -> Result<(), TierRuntimeError> {
        if self.max_receivers == 0
            || self.max_tombstones_per_receiver == 0
            || self.max_tombstone_bytes_per_receiver < TOMBSTONE_ENTRY_BYTES
            || self.max_grant_lifetime_millis == 0
        {
            return Err(TierRuntimeError::GrantInvalid);
        }
        Ok(())
    }
}

/// The checks of a grant that depend on the grant alone, not on controller state.
pub(crate) fn validate_grant_shape(
    grant: &TierGrant,
    presenter: &SessionId,
    now_millis: u64,
    limits: &TierGrantLimits,
) -> Result<(), TierRuntimeError> {
    if grant.project.0.trim().is_empty()
        || grant.receiver.0.trim().is_empty()
        || grant.source == grant.destination
        || grant.nonce == 0
        || grant.deadline_millis == 0
        || grant.deadline_millis > now_millis.saturating_add(limits.max_grant_lifetime_millis)
    {
        return Err(TierRuntimeError::GrantInvalid);
    }
    if &grant.receiver != presenter {
        return Err(TierRuntimeError::ReceiverSessionMismatch);
    }
    if now_millis > grant.deadline_millis {
        return Err(TierRuntimeError::GrantExpired);
    }
    Ok(())
}

#[derive(Debug, Default)]
struct ReceiverTombstones {
    /// nonce to the millisecond after which the tombstone may be dropped
    entries: BTreeMap<u128, u64>,
}

impl ReceiverTombstones {
    fn bytes(&self) -> usize {
        self.entries.len().saturating_mul(TOMBSTONE_ENTRY_BYTES)
    }

    fn purge(&mut self, now_millis: u64) {
        self.entries.retain(|_, expiry| *expiry >= now_millis);
    }
}

/// Spent grant nonces, per receiver session.
#[derive(Debug, Default)]
pub(crate) struct TombstoneTable {
    receivers: BTreeMap<SessionId, ReceiverTombstones>,
}

impl TombstoneTable {
    /// Refuses a replayed nonce and a full table, without recording anything.
    pub(crate) fn check(
        &mut self,
        receiver: &SessionId,
        nonce: u128,
        now_millis: u64,
        limits: &TierGrantLimits,
    ) -> Result<(), TierRuntimeError> {
        if let Some(entries) = self.receivers.get_mut(receiver) {
            entries.purge(now_millis);
            if entries.entries.contains_key(&nonce) {
                return Err(TierRuntimeError::GrantReplayed);
            }
            if entries.entries.len() >= limits.max_tombstones_per_receiver
                || entries.bytes().saturating_add(TOMBSTONE_ENTRY_BYTES)
                    > limits.max_tombstone_bytes_per_receiver
            {
                return Err(TierRuntimeError::TombstoneSaturated);
            }
            return Ok(());
        }
        if self.receivers.len() >= limits.max_receivers {
            self.receivers.values_mut().for_each(|entries| {
                entries.purge(now_millis);
            });
            self.receivers
                .retain(|_, entries| !entries.entries.is_empty());
            if self.receivers.len() >= limits.max_receivers {
                return Err(TierRuntimeError::TombstoneSaturated);
            }
        }
        Ok(())
    }

    /// Spends `nonce` for `receiver` until `deadline_millis` plus the skew.
    pub(crate) fn consume(
        &mut self,
        receiver: &SessionId,
        nonce: u128,
        deadline_millis: u64,
        now_millis: u64,
        limits: &TierGrantLimits,
    ) -> Result<(), TierRuntimeError> {
        self.check(receiver, nonce, now_millis, limits)?;
        let expiry = deadline_millis.saturating_add(limits.skew_millis);
        self.receivers
            .entry(receiver.clone())
            .or_default()
            .entries
            .insert(nonce, expiry);
        Ok(())
    }

    /// Live tombstones of `receiver`, for observation.
    pub(crate) fn live(&self, receiver: &SessionId, now_millis: u64) -> usize {
        self.receivers.get(receiver).map_or(0, |entries| {
            entries
                .entries
                .values()
                .filter(|expiry| **expiry >= now_millis)
                .count()
        })
    }
}
