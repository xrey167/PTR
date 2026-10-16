use crate::tier_grant::{
    validate_grant_shape, SystemTierClock, TierClock, TierGrant, TierGrantLimits, TombstoneTable,
};
use ptr_ledger::LedgerEvent;
use ptr_storage::{
    PreparedTierObject, StorageTier, TierBackend, TierBackendId, TierCapabilities, TierError,
    TierHealth, TierObjectManifest,
};
use ptr_types::{Digest, Generation, ProjectId, Revision, SessionId};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, Weak,
};
use std::task::{Context, Poll, Waker};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendLifecycleState {
    Configured,
    HealthChecked,
    Available,
    Draining,
    Detached,
    Revoked,
}

impl BackendLifecycleState {
    pub const fn code(self) -> u8 {
        match self {
            Self::Configured => 0,
            Self::HealthChecked => 1,
            Self::Available => 2,
            Self::Draining => 3,
            Self::Detached => 4,
            Self::Revoked => 5,
        }
    }

    pub fn from_code(code: u8) -> Result<Self, TierProjectionError> {
        match code {
            0 => Ok(Self::Configured),
            1 => Ok(Self::HealthChecked),
            2 => Ok(Self::Available),
            3 => Ok(Self::Draining),
            4 => Ok(Self::Detached),
            5 => Ok(Self::Revoked),
            _ => Err(TierProjectionError::UnknownStateCode),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicaState {
    Preparing,
    Available,
    Draining,
    Corrupt,
    Revoked,
    Released,
}

impl ReplicaState {
    pub const fn code(self) -> u8 {
        match self {
            Self::Preparing => 0,
            Self::Available => 1,
            Self::Draining => 2,
            Self::Corrupt => 3,
            Self::Revoked => 4,
            Self::Released => 5,
        }
    }

    pub fn from_code(code: u8) -> Result<Self, TierProjectionError> {
        match code {
            0 => Ok(Self::Preparing),
            1 => Ok(Self::Available),
            2 => Ok(Self::Draining),
            3 => Ok(Self::Corrupt),
            4 => Ok(Self::Revoked),
            5 => Ok(Self::Released),
            _ => Err(TierProjectionError::UnknownStateCode),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TierProjectionError {
    InvalidEvent,
    UnknownStateCode,
    InvalidBackendTransition,
    InvalidReplicaTransition,
    ConflictingObject,
    UnknownBackend,
    UnknownObject,
    Manifest(TierError),
}

impl From<TierError> for TierProjectionError {
    fn from(value: TierError) -> Self {
        Self::Manifest(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JournaledBackend {
    pub tier: StorageTier,
    pub state: BackendLifecycleState,
    pub revision: Revision,
    pub event_digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JournaledReplica {
    pub tier: StorageTier,
    pub state: ReplicaState,
    pub generation: Generation,
    pub revision: Revision,
    pub event_digest: Digest,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TierJournalProjection {
    backends: BTreeMap<TierBackendId, JournaledBackend>,
    objects: BTreeMap<Digest, TierObjectManifest>,
    replicas: BTreeMap<(Digest, TierBackendId), JournaledReplica>,
    seen_events: BTreeSet<Digest>,
}

impl TierJournalProjection {
    pub fn backend(&self, id: &TierBackendId) -> Option<&JournaledBackend> {
        self.backends.get(id)
    }

    pub fn object(&self, digest: &Digest) -> Option<&TierObjectManifest> {
        self.objects.get(digest)
    }

    pub fn replica(&self, object: &Digest, backend: &TierBackendId) -> Option<&JournaledReplica> {
        self.replicas.get(&(*object, backend.clone()))
    }

    pub fn apply(&mut self, event: &LedgerEvent) -> Result<(), TierProjectionError> {
        match event {
            LedgerEvent::TierBackendLifecycle {
                backend_id,
                tier,
                state,
                revision,
                event_digest,
            } => {
                if backend_id.trim().is_empty()
                    || backend_id != backend_id.trim()
                    || revision.0 == 0
                    || *event_digest == [0; 32]
                {
                    return Err(TierProjectionError::InvalidEvent);
                }
                let id = TierBackendId(backend_id.clone());
                let next = JournaledBackend {
                    tier: StorageTier::from_code(*tier)?,
                    state: BackendLifecycleState::from_code(*state)?,
                    revision: *revision,
                    event_digest: *event_digest,
                };
                if backend_lifecycle_digest(&id, next.tier, next.state, next.revision)
                    != next.event_digest
                {
                    return Err(TierProjectionError::InvalidEvent);
                }
                if self.seen_events.contains(&next.event_digest) {
                    return Ok(());
                }
                if let Some(current) = self.backends.get(&id) {
                    if current == &next {
                        return Ok(());
                    }
                    if current.tier != next.tier
                        || next.revision.0 <= current.revision.0
                        || !valid_backend_transition(current.state, next.state)
                    {
                        return Err(TierProjectionError::InvalidBackendTransition);
                    }
                } else if next.state != BackendLifecycleState::Configured {
                    return Err(TierProjectionError::InvalidBackendTransition);
                }
                self.seen_events.insert(next.event_digest);
                self.backends.insert(id, next);
                Ok(())
            }
            LedgerEvent::TierObjectCommitted {
                root_digest,
                generation,
                revision,
                manifest,
            } => {
                let decoded = TierObjectManifest::decode_canonical(manifest)?;
                if decoded.root_digest != *root_digest
                    || decoded.generation != *generation
                    || decoded.revision != *revision
                {
                    return Err(TierProjectionError::InvalidEvent);
                }
                match self.objects.get(root_digest) {
                    Some(current) if current == &decoded => Ok(()),
                    Some(_) => Err(TierProjectionError::ConflictingObject),
                    None => {
                        self.objects.insert(*root_digest, decoded);
                        Ok(())
                    }
                }
            }
            LedgerEvent::TierReplicaLifecycle {
                root_digest,
                backend_id,
                tier,
                state,
                generation,
                revision,
                event_digest,
            } => {
                let object = self
                    .objects
                    .get(root_digest)
                    .ok_or(TierProjectionError::UnknownObject)?;
                if object.generation != *generation
                    || backend_id.trim().is_empty()
                    || backend_id != backend_id.trim()
                    || revision.0 == 0
                    || *event_digest == [0; 32]
                {
                    return Err(TierProjectionError::InvalidEvent);
                }
                let id = TierBackendId(backend_id.clone());
                let backend = self
                    .backends
                    .get(&id)
                    .ok_or(TierProjectionError::UnknownBackend)?;
                let next = JournaledReplica {
                    tier: StorageTier::from_code(*tier)?,
                    state: ReplicaState::from_code(*state)?,
                    generation: *generation,
                    revision: *revision,
                    event_digest: *event_digest,
                };
                if replica_lifecycle_digest(
                    *root_digest,
                    &id,
                    next.tier,
                    next.state,
                    next.generation,
                    next.revision,
                ) != next.event_digest
                {
                    return Err(TierProjectionError::InvalidEvent);
                }
                if self.seen_events.contains(&next.event_digest) {
                    return Ok(());
                }
                // A detached or revoked backend takes no new work, but its replicas
                // must still be retirable: marking one Revoked, Corrupt or Released
                // is how the journal stops trusting a replica on a backend that is
                // gone. Everything else (Preparing, Available, Draining) stays refused.
                let backend_gone = matches!(
                    backend.state,
                    BackendLifecycleState::Detached | BackendLifecycleState::Revoked
                );
                let retires_replica = matches!(
                    next.state,
                    ReplicaState::Revoked | ReplicaState::Corrupt | ReplicaState::Released
                );
                if backend.tier != next.tier
                    || (backend_gone && !retires_replica)
                    || (next.state == ReplicaState::Preparing
                        && backend.state != BackendLifecycleState::Available)
                {
                    return Err(TierProjectionError::InvalidEvent);
                }
                let key = (*root_digest, id);
                if let Some(current) = self.replicas.get(&key) {
                    if current == &next {
                        return Ok(());
                    }
                    if current.tier != next.tier
                        || next.revision.0 <= current.revision.0
                        || !valid_replica_transition(current.state, next.state)
                    {
                        return Err(TierProjectionError::InvalidReplicaTransition);
                    }
                } else if next.state != ReplicaState::Preparing {
                    return Err(TierProjectionError::InvalidReplicaTransition);
                }
                self.seen_events.insert(next.event_digest);
                self.replicas.insert(key, next);
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

fn valid_backend_transition(current: BackendLifecycleState, next: BackendLifecycleState) -> bool {
    matches!(
        (current, next),
        (
            BackendLifecycleState::Configured,
            BackendLifecycleState::HealthChecked
        ) | (
            BackendLifecycleState::Configured,
            BackendLifecycleState::Revoked
        ) | (
            BackendLifecycleState::HealthChecked,
            BackendLifecycleState::Available
        ) | (
            BackendLifecycleState::HealthChecked,
            BackendLifecycleState::Revoked
        ) | (
            BackendLifecycleState::Available,
            BackendLifecycleState::Draining
        ) | (
            BackendLifecycleState::Available,
            BackendLifecycleState::Revoked
        ) | (
            BackendLifecycleState::Draining,
            BackendLifecycleState::Detached
        ) | (
            BackendLifecycleState::Draining,
            BackendLifecycleState::Revoked
        )
    )
}

fn valid_replica_transition(current: ReplicaState, next: ReplicaState) -> bool {
    matches!(
        (current, next),
        (ReplicaState::Preparing, ReplicaState::Available)
            | (ReplicaState::Preparing, ReplicaState::Corrupt)
            | (ReplicaState::Preparing, ReplicaState::Revoked)
            | (ReplicaState::Available, ReplicaState::Draining)
            | (ReplicaState::Available, ReplicaState::Corrupt)
            | (ReplicaState::Available, ReplicaState::Revoked)
            | (ReplicaState::Draining, ReplicaState::Released)
            | (ReplicaState::Draining, ReplicaState::Corrupt)
            | (ReplicaState::Draining, ReplicaState::Revoked)
    )
}

pub fn backend_lifecycle_digest(
    backend: &TierBackendId,
    tier: StorageTier,
    state: BackendLifecycleState,
    revision: Revision,
) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-tier-backend-lifecycle-v1\0");
    hasher.update((backend.0.len() as u64).to_le_bytes());
    hasher.update(backend.0.as_bytes());
    hasher.update([tier.code(), state.code()]);
    hasher.update(revision.0.to_le_bytes());
    hasher.finalize().into()
}

pub fn replica_lifecycle_digest(
    object: Digest,
    backend: &TierBackendId,
    tier: StorageTier,
    state: ReplicaState,
    generation: Generation,
    revision: Revision,
) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-tier-replica-lifecycle-v1\0");
    hasher.update(object);
    hasher.update((backend.0.len() as u64).to_le_bytes());
    hasher.update(backend.0.as_bytes());
    hasher.update([tier.code(), state.code()]);
    hasher.update(generation.0.to_le_bytes());
    hasher.update(revision.0.to_le_bytes());
    hasher.finalize().into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveWriteBinding {
    pub generation: Generation,
    pub placement_epoch: u64,
    pub fencing_token: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TierTransferAuthorization {
    pub generation: Generation,
    pub placement_epoch: u64,
    pub fencing_token: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierReplica {
    pub backend: TierBackendId,
    pub tier: StorageTier,
    pub state: ReplicaState,
    pub verified_digest: Digest,
    pub pins: u64,
    pub last_access: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyRecord {
    pub object: TierObjectManifest,
    pub replicas: Vec<TierReplica>,
    pub active_write: Option<ActiveWriteBinding>,
}

impl ResidencyRecord {
    pub fn available_replicas(&self) -> usize {
        self.replicas
            .iter()
            .filter(|replica| replica.state == ReplicaState::Available)
            .count()
    }
}

struct PinLease {
    id: u64,
    owner: u64,
    object: Digest,
    backend: TierBackendId,
    state: Weak<Mutex<ControllerState>>,
    released: AtomicBool,
}

impl PinLease {
    fn release(&self) -> Result<(), TierRuntimeError> {
        if self
            .released
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(());
        }
        let Some(state) = self.state.upgrade() else {
            return Ok(());
        };
        let mut state = match state.lock() {
            Ok(state) => state,
            Err(_) => {
                self.released.store(false, Ordering::Release);
                return Err(TierRuntimeError::LockPoisoned);
            }
        };
        let Some((object, backend)) = state.pins.remove(&self.id) else {
            return Ok(());
        };
        if object != self.object || backend != self.backend {
            self.released.store(false, Ordering::Release);
            return Err(TierRuntimeError::ForeignPin);
        }
        let record = state
            .records
            .get_mut(&object)
            .ok_or(TierRuntimeError::UnknownObject)?;
        let replica = record
            .replicas
            .iter_mut()
            .find(|replica| replica.backend == backend)
            .ok_or(TierRuntimeError::UnknownReplica)?;
        replica.pins = replica.pins.saturating_sub(1);
        Ok(())
    }
}

impl Drop for PinLease {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

#[derive(Clone)]
pub struct TierPin {
    lease: Arc<PinLease>,
}

impl std::fmt::Debug for TierPin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TierPin")
            .field("id", &self.lease.id)
            .field("owner", &self.lease.owner)
            .field("object", &self.lease.object)
            .field("backend", &self.lease.backend)
            .field("released", &self.lease.released.load(Ordering::Acquire))
            .finish()
    }
}

impl PartialEq for TierPin {
    fn eq(&self, other: &Self) -> bool {
        self.lease.owner == other.lease.owner && self.lease.id == other.lease.id
    }
}

impl Eq for TierPin {}

pub struct AdmittedTierObject {
    owner: u64,
    backend: TierBackendId,
    object: PreparedTierObject,
    pin: TierPin,
}

impl AdmittedTierObject {
    pub fn backend(&self) -> &TierBackendId {
        &self.backend
    }

    pub fn object(&self) -> &PreparedTierObject {
        &self.object
    }

    pub fn root_digest(&self) -> Digest {
        self.object.manifest.root_digest
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierTransferReceipt {
    pub object: Digest,
    pub source: TierBackendId,
    pub destination: TierBackendId,
    pub chunks: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefetchOutcome {
    Ready(TierTransferReceipt),
    Skipped(TierRuntimeError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TierRuntimeError {
    LockPoisoned,
    DuplicateBackend,
    UnknownBackend,
    BackendUnavailable,
    BackendDraining,
    BackendNotDrained,
    ObjectConflict,
    UnknownObject,
    UnknownReplica,
    ReplicaUnavailable,
    ReplicaPinned,
    LastReplica,
    ForeignPin,
    StaleWriteBinding,
    CounterExhausted,
    TransferBackpressure,
    TransferCancelled,
    Storage(TierError),
    /// The write authorization held when the transfer started no longer matched the object's
    /// active write binding just before the destination replica would have become available.
    /// The destination replica was left unpublished and its chunks were removed.
    AuthorizationRevoked,
    /// The object belongs to a project, so only a grant of that project can move it.
    GrantRequired,
    /// The grant's project is not the project of the object (or the object has no project).
    ProjectMismatch,
    /// Malformed grant (empty ids, zero nonce or deadline, source equal to destination) or
    /// malformed grant limits.
    GrantInvalid,
    /// The grant is bound to another receiver session than the presenter.
    ReceiverSessionMismatch,
    /// The grant's deadline passed before it was presented or before the destination replica
    /// would have been published.
    GrantExpired,
    /// The grant's nonce was already consumed by a publish and its tombstone is still live.
    GrantReplayed,
    /// The tombstone table of the receiver (or the number of receivers) is at its bound.
    TombstoneSaturated,
    /// The transfer was refused for the contained reason and removing the unpublished
    /// destination chunks failed as well.
    CleanupFailed(Box<TierRuntimeError>),
}

impl TierRuntimeError {
    /// Refusals decided at the publish step: the copy itself succeeded, the replica is not
    /// published and its chunks are removed.
    fn refuses_publication(&self) -> bool {
        matches!(
            self,
            Self::AuthorizationRevoked
                | Self::GrantExpired
                | Self::GrantReplayed
                | Self::TombstoneSaturated
                | Self::ProjectMismatch
        )
    }
}

impl From<TierError> for TierRuntimeError {
    fn from(value: TierError) -> Self {
        Self::Storage(value)
    }
}

struct BackendEntry {
    backend: Arc<dyn TierBackend>,
    capabilities: TierCapabilities,
    state: BackendLifecycleState,
}

/// What a transfer must still hold at the moment its destination replica is published.
#[derive(Clone, Debug)]
struct PublishCheck {
    authorization: Option<TierTransferAuthorization>,
    grant: Option<TierGrant>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TransferKey {
    object: Digest,
    source: TierBackendId,
    destination: TierBackendId,
}

struct GateState {
    result: Option<Result<TierTransferReceipt, TierRuntimeError>>,
    wakers: Vec<Waker>,
}

struct TransferGate {
    state: Mutex<GateState>,
}

impl TransferGate {
    fn new() -> Self {
        Self {
            state: Mutex::new(GateState {
                result: None,
                wakers: Vec::new(),
            }),
        }
    }

    fn wait(self: &Arc<Self>) -> GateWait {
        GateWait { gate: self.clone() }
    }

    fn complete(&self, result: Result<TierTransferReceipt, TierRuntimeError>) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.result = Some(result);
        for waker in state.wakers.drain(..) {
            waker.wake();
        }
    }
}

struct GateWait {
    gate: Arc<TransferGate>,
}

impl Future for GateWait {
    type Output = Result<TierTransferReceipt, TierRuntimeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Ok(mut state) = self.gate.state.lock() else {
            return Poll::Ready(Err(TierRuntimeError::LockPoisoned));
        };
        if let Some(result) = state.result.clone() {
            return Poll::Ready(result);
        }
        if !state.wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
            state.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

struct ControllerState {
    backends: BTreeMap<TierBackendId, BackendEntry>,
    records: BTreeMap<Digest, ResidencyRecord>,
    chunk_refs: BTreeMap<(TierBackendId, Digest), u64>,
    chunk_reservations: BTreeMap<(TierBackendId, Digest), u64>,
    chunk_deletions: BTreeSet<(TierBackendId, Digest)>,
    pins: BTreeMap<u64, (Digest, TierBackendId)>,
    object_projects: BTreeMap<Digest, ProjectId>,
    tombstones: TombstoneTable,
    /// Grants (receiver, nonce) a transfer call holds right now, until it is satisfied.
    inflight_grants: BTreeSet<(SessionId, u128)>,
    inflight: BTreeMap<TransferKey, Arc<TransferGate>>,
    inflight_bytes: u64,
    next_pin: u64,
    access_clock: u64,
}

pub struct TierResidencyController {
    owner: u64,
    state: Arc<Mutex<ControllerState>>,
    max_inflight_transfers: usize,
    max_inflight_bytes: u64,
    clock: Arc<dyn TierClock>,
    grant_limits: TierGrantLimits,
}

struct TransferLeaderGuard<'a> {
    controller: &'a TierResidencyController,
    key: TransferKey,
    source: TierBackendId,
    destination: TierBackendId,
    manifest: TierObjectManifest,
    reserved_digests: Vec<(Digest, u64)>,
    gate: Arc<TransferGate>,
    publish: PublishCheck,
    active: bool,
}

/// Holds a grant's (receiver, nonce) as in flight; released when the call ends or is dropped.
struct GrantReservation<'a> {
    controller: &'a TierResidencyController,
    key: (SessionId, u128),
}

impl Drop for GrantReservation<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.controller.state.lock() {
            state.inflight_grants.remove(&self.key);
        }
    }
}

struct RegistrationReservationGuard<'a> {
    controller: &'a TierResidencyController,
    backend: TierBackendId,
    digests: Vec<(Digest, u64)>,
}

struct DeletionGuard<'a> {
    controller: &'a TierResidencyController,
    backend: TierBackendId,
    digests: Vec<Digest>,
}

impl Drop for DeletionGuard<'_> {
    fn drop(&mut self) {
        let Ok(mut state) = self.controller.state.lock() else {
            return;
        };
        for digest in &self.digests {
            state
                .chunk_deletions
                .remove(&(self.backend.clone(), *digest));
        }
    }
}

impl Drop for RegistrationReservationGuard<'_> {
    fn drop(&mut self) {
        let Ok(mut state) = self.controller.state.lock() else {
            return;
        };
        for (digest, reserved) in &self.digests {
            let key = (self.backend.clone(), *digest);
            match state.chunk_reservations.get_mut(&key) {
                Some(count) if *count > *reserved => *count -= *reserved,
                Some(_) => {
                    state.chunk_reservations.remove(&key);
                }
                None => {}
            }
        }
    }
}

impl TransferLeaderGuard<'_> {
    fn complete(
        &mut self,
        result: Result<TierTransferReceipt, TierRuntimeError>,
    ) -> Result<TierTransferReceipt, TierRuntimeError> {
        if !self.active {
            return result;
        }
        let result = self.controller.finish_transfer(
            &self.key,
            &self.source,
            &self.destination,
            &self.manifest,
            &self.reserved_digests,
            &self.publish,
            result,
        );
        self.gate.complete(result.clone());
        self.active = false;
        result
    }
}

impl Drop for TransferLeaderGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.complete(Err(TierRuntimeError::TransferCancelled));
        }
    }
}

impl Default for TierResidencyController {
    fn default() -> Self {
        static NEXT_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            owner: NEXT_OWNER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            state: Arc::new(Mutex::new(ControllerState {
                backends: BTreeMap::new(),
                records: BTreeMap::new(),
                chunk_refs: BTreeMap::new(),
                chunk_reservations: BTreeMap::new(),
                chunk_deletions: BTreeSet::new(),
                pins: BTreeMap::new(),
                object_projects: BTreeMap::new(),
                tombstones: TombstoneTable::default(),
                inflight_grants: BTreeSet::new(),
                inflight: BTreeMap::new(),
                inflight_bytes: 0,
                next_pin: 1,
                access_clock: 1,
            })),
            max_inflight_transfers: 64,
            max_inflight_bytes: u64::MAX,
            clock: Arc::new(SystemTierClock),
            grant_limits: TierGrantLimits::default(),
        }
    }
}

impl TierResidencyController {
    pub fn with_transfer_budget(
        max_inflight_transfers: usize,
        max_inflight_bytes: u64,
    ) -> Result<Self, TierRuntimeError> {
        if max_inflight_transfers == 0 || max_inflight_bytes == 0 {
            return Err(TierRuntimeError::TransferBackpressure);
        }
        Ok(Self {
            max_inflight_transfers,
            max_inflight_bytes,
            ..Self::default()
        })
    }

    /// Replaces the clock and the bounds used for grants. Refuses zero bounds.
    pub fn with_grants(
        mut self,
        clock: Arc<dyn TierClock>,
        limits: TierGrantLimits,
    ) -> Result<Self, TierRuntimeError> {
        limits.validate()?;
        self.clock = clock;
        self.grant_limits = limits;
        Ok(self)
    }

    /// The project an object was registered under, or `None` for an unscoped object.
    pub fn object_project(&self, object: &Digest) -> Result<Option<ProjectId>, TierRuntimeError> {
        let state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        if !state.records.contains_key(object) {
            return Err(TierRuntimeError::UnknownObject);
        }
        Ok(state.object_projects.get(object).cloned())
    }

    /// Live (not yet expired) grant tombstones of one receiver session.
    pub fn live_grant_tombstones(&self, receiver: &SessionId) -> Result<usize, TierRuntimeError> {
        let now = self.clock.now_millis();
        Ok(self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?
            .tombstones
            .live(receiver, now))
    }

    pub async fn attach_backend(
        &self,
        backend: Arc<dyn TierBackend>,
    ) -> Result<(), TierRuntimeError> {
        let id = backend.identity();
        let capabilities = backend.capabilities();
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            if state.backends.contains_key(&id) {
                return Err(TierRuntimeError::DuplicateBackend);
            }
            state.backends.insert(
                id.clone(),
                BackendEntry {
                    backend: backend.clone(),
                    capabilities,
                    state: BackendLifecycleState::Configured,
                },
            );
        }
        let health = backend.health().await?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        let entry = state
            .backends
            .get_mut(&id)
            .ok_or(TierRuntimeError::UnknownBackend)?;
        entry.state = BackendLifecycleState::HealthChecked;
        if health != TierHealth::Healthy {
            entry.state = BackendLifecycleState::Revoked;
            return Err(TierRuntimeError::BackendUnavailable);
        }
        entry.state = BackendLifecycleState::Available;
        Ok(())
    }

    pub fn backend_state(
        &self,
        backend: &TierBackendId,
    ) -> Result<BackendLifecycleState, TierRuntimeError> {
        self.state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?
            .backends
            .get(backend)
            .map(|entry| entry.state)
            .ok_or(TierRuntimeError::UnknownBackend)
    }

    pub fn backend_tier(&self, backend: &TierBackendId) -> Result<StorageTier, TierRuntimeError> {
        self.state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?
            .backends
            .get(backend)
            .map(|entry| entry.capabilities.tier)
            .ok_or(TierRuntimeError::UnknownBackend)
    }

    pub async fn register_object(
        &self,
        object: TierObjectManifest,
        backend: TierBackendId,
    ) -> Result<(), TierRuntimeError> {
        self.register_object_scoped(object, backend, None).await
    }

    /// Registers an object in the namespace of `project`. Only a grant of that project can
    /// move it; registering the same object under another project (or unscoped) is
    /// `ProjectMismatch`.
    pub async fn register_project_object(
        &self,
        project: ProjectId,
        object: TierObjectManifest,
        backend: TierBackendId,
    ) -> Result<(), TierRuntimeError> {
        if project.0.trim().is_empty() {
            return Err(TierRuntimeError::GrantInvalid);
        }
        self.register_object_scoped(object, backend, Some(project))
            .await
    }

    async fn register_object_scoped(
        &self,
        object: TierObjectManifest,
        backend: TierBackendId,
        project: Option<ProjectId>,
    ) -> Result<(), TierRuntimeError> {
        object.validate()?;
        let mut reservation_counts = BTreeMap::<Digest, u64>::new();
        for chunk in &object.chunks {
            let count = reservation_counts.entry(chunk.digest).or_default();
            *count = count
                .checked_add(1)
                .ok_or(TierRuntimeError::CounterExhausted)?;
        }
        let reserved_digests = reservation_counts.into_iter().collect::<Vec<_>>();
        let source = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            let entry = state
                .backends
                .get(&backend)
                .ok_or(TierRuntimeError::UnknownBackend)?;
            if entry.state != BackendLifecycleState::Available {
                return Err(TierRuntimeError::BackendUnavailable);
            }
            let source = entry.backend.clone();
            if let Some(existing) = state.records.get(&object.root_digest) {
                if existing.object != object {
                    return Err(TierRuntimeError::ObjectConflict);
                }
                if state.object_projects.get(&object.root_digest) != project.as_ref() {
                    return Err(TierRuntimeError::ProjectMismatch);
                }
                if let Some(replica) = existing
                    .replicas
                    .iter()
                    .find(|replica| replica.backend == backend)
                {
                    match replica.state {
                        ReplicaState::Available => return Ok(()),
                        ReplicaState::Preparing | ReplicaState::Draining => {
                            return Err(TierRuntimeError::ReplicaUnavailable)
                        }
                        ReplicaState::Corrupt | ReplicaState::Revoked | ReplicaState::Released => {}
                    }
                }
            }
            for (digest, count) in &reserved_digests {
                let key = (backend.clone(), *digest);
                if state.chunk_deletions.contains(&key) {
                    return Err(TierRuntimeError::ReplicaUnavailable);
                }
                let current = state.chunk_reservations.get(&key).copied().unwrap_or(0);
                current
                    .checked_add(*count)
                    .ok_or(TierRuntimeError::CounterExhausted)?;
            }
            for (digest, count) in &reserved_digests {
                let reservations = state
                    .chunk_reservations
                    .entry((backend.clone(), *digest))
                    .or_default();
                *reservations += *count;
            }
            source
        };
        let _reservation = RegistrationReservationGuard {
            controller: self,
            backend: backend.clone(),
            digests: reserved_digests.clone(),
        };
        for chunk in &object.chunks {
            let report = source.verify_chunk(chunk).await?;
            if !report.valid {
                return Err(TierRuntimeError::Storage(TierError::CorruptChunk));
            }
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        let backend_entry = state
            .backends
            .get(&backend)
            .ok_or(TierRuntimeError::UnknownBackend)?;
        if backend_entry.state != BackendLifecycleState::Available {
            return Err(TierRuntimeError::BackendDraining);
        }
        if let Some(existing) = state.records.get(&object.root_digest) {
            if existing.object != object {
                return Err(TierRuntimeError::ObjectConflict);
            }
            if state.object_projects.get(&object.root_digest) != project.as_ref() {
                return Err(TierRuntimeError::ProjectMismatch);
            }
            if let Some(replica) = existing
                .replicas
                .iter()
                .find(|replica| replica.backend == backend)
            {
                match replica.state {
                    ReplicaState::Available => return Ok(()),
                    ReplicaState::Preparing | ReplicaState::Draining => {
                        return Err(TierRuntimeError::ReplicaUnavailable)
                    }
                    ReplicaState::Corrupt | ReplicaState::Revoked | ReplicaState::Released => {}
                }
            }
            for (digest, count) in &reserved_digests {
                let current = state
                    .chunk_refs
                    .get(&(backend.clone(), *digest))
                    .copied()
                    .unwrap_or(0);
                current
                    .checked_add(*count)
                    .ok_or(TierRuntimeError::CounterExhausted)?;
            }
            state.access_clock = state.access_clock.saturating_add(1);
            let last_access = state.access_clock;
            let tier = state
                .backends
                .get(&backend)
                .ok_or(TierRuntimeError::UnknownBackend)?
                .capabilities
                .tier;
            let record = state
                .records
                .get_mut(&object.root_digest)
                .expect("record resolved above");
            if let Some(replica) = record
                .replicas
                .iter_mut()
                .find(|replica| replica.backend == backend)
            {
                replica.tier = tier;
                replica.state = ReplicaState::Available;
                replica.verified_digest = object.root_digest;
                replica.pins = 0;
                replica.last_access = last_access;
            } else {
                record.replicas.push(TierReplica {
                    backend: backend.clone(),
                    tier,
                    state: ReplicaState::Available,
                    verified_digest: object.root_digest,
                    pins: 0,
                    last_access,
                });
            }
            for (digest, count) in &reserved_digests {
                let references = state
                    .chunk_refs
                    .entry((backend.clone(), *digest))
                    .or_default();
                *references += *count;
            }
            return Ok(());
        }
        for (digest, count) in &reserved_digests {
            let current = state
                .chunk_refs
                .get(&(backend.clone(), *digest))
                .copied()
                .unwrap_or(0);
            current
                .checked_add(*count)
                .ok_or(TierRuntimeError::CounterExhausted)?;
        }
        state.access_clock = state.access_clock.saturating_add(1);
        let last_access = state.access_clock;
        let tier = state
            .backends
            .get(&backend)
            .ok_or(TierRuntimeError::UnknownBackend)?
            .capabilities
            .tier;
        if let Some(project) = project {
            state.object_projects.insert(object.root_digest, project);
        }
        state.records.insert(
            object.root_digest,
            ResidencyRecord {
                object: object.clone(),
                replicas: vec![TierReplica {
                    backend: backend.clone(),
                    tier,
                    state: ReplicaState::Available,
                    verified_digest: object.root_digest,
                    pins: 0,
                    last_access,
                }],
                active_write: None,
            },
        );
        for (digest, count) in &reserved_digests {
            let references = state
                .chunk_refs
                .entry((backend.clone(), *digest))
                .or_default();
            *references += *count;
        }
        Ok(())
    }

    pub async fn read_admitted_object(
        &self,
        object: Digest,
        backend: &TierBackendId,
    ) -> Result<AdmittedTierObject, TierRuntimeError> {
        let pin = self.pin(object, backend)?;
        let (manifest, storage) = {
            let state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            let record = state
                .records
                .get(&object)
                .ok_or(TierRuntimeError::UnknownObject)?;
            let storage = state
                .backends
                .get(backend)
                .ok_or(TierRuntimeError::UnknownBackend)?
                .backend
                .clone();
            (record.object.clone(), storage)
        };
        let mut chunks = Vec::with_capacity(manifest.chunks.len());
        for descriptor in &manifest.chunks {
            chunks.push(storage.get_chunk(descriptor).await?);
        }
        let object = PreparedTierObject { manifest, chunks };
        object.verify()?;
        Ok(AdmittedTierObject {
            owner: self.owner,
            backend: backend.clone(),
            object,
            pin,
        })
    }

    pub fn validate_admitted_object(
        &self,
        admitted: &AdmittedTierObject,
    ) -> Result<(), TierRuntimeError> {
        if admitted.owner != self.owner || admitted.pin.lease.owner != self.owner {
            return Err(TierRuntimeError::ForeignPin);
        }
        if admitted.pin.lease.released.load(Ordering::Acquire) {
            return Err(TierRuntimeError::ReplicaUnavailable);
        }
        admitted.object.verify()?;
        let state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        if !state.pins.contains_key(&admitted.pin.lease.id) {
            return Err(TierRuntimeError::ReplicaUnavailable);
        }
        let backend = state
            .backends
            .get(&admitted.backend)
            .ok_or(TierRuntimeError::UnknownBackend)?;
        if backend.state != BackendLifecycleState::Available {
            return Err(TierRuntimeError::BackendUnavailable);
        }
        let record = state
            .records
            .get(&admitted.root_digest())
            .ok_or(TierRuntimeError::UnknownObject)?;
        if record.object != admitted.object.manifest
            || !record.replicas.iter().any(|replica| {
                replica.backend == admitted.backend
                    && replica.state == ReplicaState::Available
                    && replica.verified_digest == admitted.root_digest()
            })
        {
            return Err(TierRuntimeError::ReplicaUnavailable);
        }
        Ok(())
    }

    pub fn bind_active_write(
        &self,
        object: Digest,
        binding: ActiveWriteBinding,
    ) -> Result<(), TierRuntimeError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        let record = state
            .records
            .get_mut(&object)
            .ok_or(TierRuntimeError::UnknownObject)?;
        if record.object.generation != binding.generation {
            return Err(TierRuntimeError::StaleWriteBinding);
        }
        record.active_write = Some(binding);
        Ok(())
    }

    pub fn residency(&self, object: &Digest) -> Result<ResidencyRecord, TierRuntimeError> {
        self.state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?
            .records
            .get(object)
            .cloned()
            .ok_or(TierRuntimeError::UnknownObject)
    }

    pub fn pin(
        &self,
        object: Digest,
        backend: &TierBackendId,
    ) -> Result<TierPin, TierRuntimeError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        state.next_pin = state
            .next_pin
            .checked_add(1)
            .ok_or(TierRuntimeError::CounterExhausted)?;
        let id = state.next_pin;
        state.access_clock = state.access_clock.saturating_add(1);
        let access = state.access_clock;
        let record = state
            .records
            .get_mut(&object)
            .ok_or(TierRuntimeError::UnknownObject)?;
        let replica = record
            .replicas
            .iter_mut()
            .find(|replica| &replica.backend == backend)
            .ok_or(TierRuntimeError::UnknownReplica)?;
        if replica.state != ReplicaState::Available {
            return Err(TierRuntimeError::ReplicaUnavailable);
        }
        replica.pins = replica
            .pins
            .checked_add(1)
            .ok_or(TierRuntimeError::CounterExhausted)?;
        replica.last_access = access;
        state.pins.insert(id, (object, backend.clone()));
        Ok(TierPin {
            lease: Arc::new(PinLease {
                id,
                owner: self.owner,
                object,
                backend: backend.clone(),
                state: Arc::downgrade(&self.state),
                released: AtomicBool::new(false),
            }),
        })
    }

    pub fn release_pin(&self, pin: TierPin) -> Result<(), TierRuntimeError> {
        if pin.lease.owner != self.owner {
            return Err(TierRuntimeError::ForeignPin);
        }
        pin.lease.release()
    }

    /// Moves an unscoped object. An object registered under a project is refused with
    /// `GrantRequired`; use `transfer_with_grant` for it.
    ///
    /// The authorization is checked when the transfer starts and again, in the critical
    /// section that publishes the destination replica; if it no longer matches the object's
    /// active write binding then, the replica stays unpublished (state `Revoked`), its chunks
    /// are removed and the transfer is refused with `AuthorizationRevoked`.
    pub async fn transfer(
        &self,
        object: Digest,
        source: &TierBackendId,
        destination: &TierBackendId,
        authorization: Option<TierTransferAuthorization>,
    ) -> Result<TierTransferReceipt, TierRuntimeError> {
        self.transfer_checked(
            object,
            source,
            destination,
            PublishCheck {
                authorization,
                grant: None,
            },
            None,
        )
        .await
    }

    /// Moves the object, source and destination the grant names, for the session `presenter`.
    ///
    /// Refused unless the object is registered under the grant's project, the grant is bound
    /// to `presenter`, is within its deadline and its nonce is unspent. Everything is checked
    /// again in the critical section that publishes the destination replica, and the nonce is
    /// spent there: a refusal (or any failure) before that point leaves the grant unspent, a
    /// successful publish spends it. Presenting the same grant again is `GrantReplayed`.
    ///
    /// A call that joins an in-flight transfer of the same (object, source, destination), or
    /// finds the destination replica already available, is satisfied by that replica and spends
    /// its own grant at that moment, after the same recheck; a revoked or expired grant gets the
    /// refusal, not the other call's result. A grant that another call holds in flight is
    /// `GrantReplayed`. The presenter session must come from an authenticated session (it is
    /// unauthenticated in-process today), and `ProjectMismatch` reveals that the digest exists
    /// under another project.
    pub async fn transfer_with_grant(
        &self,
        grant: &TierGrant,
        presenter: &SessionId,
    ) -> Result<TierTransferReceipt, TierRuntimeError> {
        self.transfer_checked(
            grant.object,
            &grant.source,
            &grant.destination,
            PublishCheck {
                authorization: grant.write_authorization,
                grant: Some(grant.clone()),
            },
            Some(presenter),
        )
        .await
    }

    fn validate_scope(
        state: &ControllerState,
        object: &Digest,
        grant: Option<&TierGrant>,
    ) -> Result<(), TierRuntimeError> {
        match (state.object_projects.get(object), grant) {
            (None, None) => Ok(()),
            (Some(_), None) => Err(TierRuntimeError::GrantRequired),
            (Some(project), Some(grant)) if *project == grant.project => Ok(()),
            (Some(_), Some(_)) | (None, Some(_)) => Err(TierRuntimeError::ProjectMismatch),
        }
    }

    async fn transfer_checked(
        &self,
        object: Digest,
        source: &TierBackendId,
        destination: &TierBackendId,
        publish: PublishCheck,
        presenter: Option<&SessionId>,
    ) -> Result<TierTransferReceipt, TierRuntimeError> {
        let key = TransferKey {
            object,
            source: source.clone(),
            destination: destination.clone(),
        };
        let now = self.clock.now_millis();
        let authorization = publish.authorization;
        let reservation_key = publish
            .grant
            .as_ref()
            .map(|grant| (grant.receiver.clone(), grant.nonce));
        let mut reserved = false;
        let (gate, leader, manifest, source_backend, destination_backend, reserved_digests) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            if !state.records.contains_key(&object) {
                return Err(TierRuntimeError::UnknownObject);
            }
            Self::validate_scope(&state, &object, publish.grant.as_ref())?;
            if let (Some(grant), Some(presenter)) = (publish.grant.as_ref(), presenter) {
                validate_grant_shape(grant, presenter, now, &self.grant_limits)?;
                if state
                    .inflight_grants
                    .contains(&(grant.receiver.clone(), grant.nonce))
                {
                    return Err(TierRuntimeError::GrantReplayed);
                }
                state
                    .tombstones
                    .check(&grant.receiver, grant.nonce, now, &self.grant_limits)?;
            }
            let record = state
                .records
                .get(&object)
                .ok_or(TierRuntimeError::UnknownObject)?;
            Self::validate_authorization(record, authorization)?;
            let source_replica = record
                .replicas
                .iter()
                .find(|replica| &replica.backend == source)
                .ok_or(TierRuntimeError::UnknownReplica)?;
            let source_backend_state = state
                .backends
                .get(source)
                .ok_or(TierRuntimeError::UnknownBackend)?
                .state;
            let source_readable = source_replica.state == ReplicaState::Available
                || (source_replica.state == ReplicaState::Draining
                    && source_backend_state == BackendLifecycleState::Draining);
            if !source_readable {
                return Err(TierRuntimeError::ReplicaUnavailable);
            }
            let joined = state.inflight.get(&key).cloned();
            if let Some(gate) = joined {
                if let Some(grant_key) = &reservation_key {
                    state.inflight_grants.insert(grant_key.clone());
                    reserved = true;
                }
                (gate, false, None, None, None, None)
            } else {
                let transfer_bytes = record
                    .object
                    .chunks
                    .iter()
                    .try_fold(0u64, |total, chunk| total.checked_add(chunk.length))
                    .ok_or(TierRuntimeError::CounterExhausted)?;
                if state.inflight.len() >= self.max_inflight_transfers
                    || state
                        .inflight_bytes
                        .checked_add(transfer_bytes)
                        .is_none_or(|bytes| bytes > self.max_inflight_bytes)
                {
                    return Err(TierRuntimeError::TransferBackpressure);
                }
                let source_backend = state
                    .backends
                    .get(source)
                    .ok_or(TierRuntimeError::UnknownBackend)?
                    .backend
                    .clone();
                let destination_entry = state
                    .backends
                    .get(destination)
                    .ok_or(TierRuntimeError::UnknownBackend)?;
                match destination_entry.state {
                    BackendLifecycleState::Available => {}
                    BackendLifecycleState::Draining => {
                        return Err(TierRuntimeError::BackendDraining)
                    }
                    _ => return Err(TierRuntimeError::BackendUnavailable),
                }
                let destination_backend = destination_entry.backend.clone();
                let destination_tier = destination_entry.capabilities.tier;
                let reserved_digests = record.object.chunks.iter().try_fold(
                    BTreeMap::<Digest, u64>::new(),
                    |mut counts, chunk| {
                        let count = counts.entry(chunk.digest).or_default();
                        *count = count
                            .checked_add(1)
                            .ok_or(TierRuntimeError::CounterExhausted)?;
                        Ok::<_, TierRuntimeError>(counts)
                    },
                )?;
                for (digest, count) in &reserved_digests {
                    let reservation_key = (destination.clone(), *digest);
                    if state.chunk_deletions.contains(&reservation_key) {
                        return Err(TierRuntimeError::ReplicaUnavailable);
                    }
                    state
                        .chunk_reservations
                        .get(&reservation_key)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(*count)
                        .ok_or(TierRuntimeError::CounterExhausted)?;
                }
                let already_available = state.records.get(&object).is_some_and(|record| {
                    record.replicas.iter().any(|replica| {
                        &replica.backend == destination && replica.state == ReplicaState::Available
                    })
                });
                if already_available {
                    let manifest = state
                        .records
                        .get(&object)
                        .map(|record| record.object.clone())
                        .ok_or(TierRuntimeError::UnknownObject)?;
                    // Nothing is copied or published by this call, but the grant is
                    // satisfied by the existing replica, so it is rechecked and spent here.
                    Self::recheck_and_spend(
                        &mut state,
                        &manifest,
                        &publish,
                        self.clock.now_millis(),
                        &self.grant_limits,
                    )?;
                    return Ok(TierTransferReceipt {
                        object,
                        source: source.clone(),
                        destination: destination.clone(),
                        chunks: manifest.chunks.len(),
                        bytes: manifest.chunks.iter().map(|chunk| chunk.length).sum(),
                    });
                }
                let manifest = {
                    let record = state
                        .records
                        .get_mut(&object)
                        .expect("record resolved above");
                    if !record
                        .replicas
                        .iter()
                        .any(|replica| &replica.backend == destination)
                    {
                        record.replicas.push(TierReplica {
                            backend: destination.clone(),
                            tier: destination_tier,
                            state: ReplicaState::Preparing,
                            verified_digest: [0; 32],
                            pins: 0,
                            last_access: 0,
                        });
                    }
                    let source_replica = record
                        .replicas
                        .iter_mut()
                        .find(|replica| &replica.backend == source)
                        .expect("source replica resolved above");
                    source_replica.pins = source_replica
                        .pins
                        .checked_add(1)
                        .ok_or(TierRuntimeError::CounterExhausted)?;
                    record.object.clone()
                };
                for (digest, count) in &reserved_digests {
                    let reservations = state
                        .chunk_reservations
                        .entry((destination.clone(), *digest))
                        .or_default();
                    *reservations += *count;
                }
                let gate = Arc::new(TransferGate::new());
                state.inflight.insert(key.clone(), gate.clone());
                state.inflight_bytes = state.inflight_bytes.saturating_add(transfer_bytes);
                if let Some(grant_key) = &reservation_key {
                    state.inflight_grants.insert(grant_key.clone());
                    reserved = true;
                }
                (
                    gate,
                    true,
                    Some(manifest),
                    Some(source_backend),
                    Some(destination_backend),
                    Some(reserved_digests.into_iter().collect()),
                )
            }
        };

        let _reservation = match (reserved, reservation_key) {
            (true, Some(key)) => Some(GrantReservation {
                controller: self,
                key,
            }),
            _ => None,
        };
        if !leader {
            let result = gate.wait().await?;
            if publish.grant.is_none() {
                return Ok(result);
            }
            // A follower's grant is rechecked and spent when it is satisfied, like the
            // leader's: its own authorization, scope and deadline, and its own nonce.
            let now = self.clock.now_millis();
            let mut state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            let manifest = state
                .records
                .get(&object)
                .map(|record| record.object.clone())
                .ok_or(TierRuntimeError::UnknownObject)?;
            Self::recheck_and_spend(&mut state, &manifest, &publish, now, &self.grant_limits)?;
            return Ok(result);
        }
        let manifest = manifest.expect("leader has manifest");
        let source_backend = source_backend.expect("leader has source backend");
        let destination_backend = destination_backend.expect("leader has destination backend");
        let cleanup_backend = destination_backend.clone();
        let reserved_digests = reserved_digests.expect("leader has chunk reservations");
        let mut leader_guard = TransferLeaderGuard {
            controller: self,
            key: key.clone(),
            source: source.clone(),
            destination: destination.clone(),
            manifest: manifest.clone(),
            reserved_digests,
            gate: gate.clone(),
            publish,
            active: true,
        };
        let result = self
            .copy_manifest(
                &manifest,
                source,
                destination,
                source_backend,
                destination_backend,
            )
            .await;
        match leader_guard.complete(result) {
            Err(refusal) if refusal.refuses_publication() => {
                match self
                    .discard_unpublished(&manifest, destination, cleanup_backend.as_ref())
                    .await
                {
                    Ok(()) => Err(refusal),
                    Err(_) => Err(TierRuntimeError::CleanupFailed(Box::new(refusal))),
                }
            }
            other => other,
        }
    }

    /// Removes the chunks a refused transfer copied to `destination`, except those another
    /// replica or an in-flight transfer on that backend still needs.
    async fn discard_unpublished(
        &self,
        manifest: &TierObjectManifest,
        destination: &TierBackendId,
        backend: &dyn TierBackend,
    ) -> Result<(), TierRuntimeError> {
        let chunks = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            let mut chunks = Vec::new();
            let mut seen = BTreeSet::new();
            for chunk in &manifest.chunks {
                let key = (destination.clone(), chunk.digest);
                let referenced = state.chunk_refs.get(&key).copied().unwrap_or(0) != 0;
                let reserved = state.chunk_reservations.get(&key).copied().unwrap_or(0) != 0;
                if !referenced
                    && !reserved
                    && !state.chunk_deletions.contains(&key)
                    && seen.insert(chunk.digest)
                {
                    chunks.push(chunk.clone());
                }
            }
            for chunk in &chunks {
                state
                    .chunk_deletions
                    .insert((destination.clone(), chunk.digest));
            }
            chunks
        };
        let _deletion_guard = DeletionGuard {
            controller: self,
            backend: destination.clone(),
            digests: chunks.iter().map(|chunk| chunk.digest).collect(),
        };
        for chunk in &chunks {
            match backend.delete_chunk(chunk).await {
                Ok(()) | Err(TierError::UnknownChunk) => {}
                Err(error) => return Err(TierRuntimeError::Storage(error)),
            }
        }
        Ok(())
    }

    pub async fn prefetch(
        &self,
        object: Digest,
        source: &TierBackendId,
        destination: &TierBackendId,
        authorization: Option<TierTransferAuthorization>,
    ) -> PrefetchOutcome {
        match self
            .transfer(object, source, destination, authorization)
            .await
        {
            Ok(receipt) => PrefetchOutcome::Ready(receipt),
            Err(error) => PrefetchOutcome::Skipped(error),
        }
    }

    /// Like `prefetch`, for the object, source and destination a grant names.
    pub async fn prefetch_with_grant(
        &self,
        grant: &TierGrant,
        presenter: &SessionId,
    ) -> PrefetchOutcome {
        match self.transfer_with_grant(grant, presenter).await {
            Ok(receipt) => PrefetchOutcome::Ready(receipt),
            Err(error) => PrefetchOutcome::Skipped(error),
        }
    }

    pub fn begin_detach(&self, backend: &TierBackendId) -> Result<(), TierRuntimeError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        let entry = state
            .backends
            .get_mut(backend)
            .ok_or(TierRuntimeError::UnknownBackend)?;
        if entry.state == BackendLifecycleState::Detached {
            return Ok(());
        }
        if entry.state == BackendLifecycleState::Draining {
            return Ok(());
        }
        if entry.state != BackendLifecycleState::Available {
            return Err(TierRuntimeError::BackendUnavailable);
        }
        entry.state = BackendLifecycleState::Draining;
        for record in state.records.values_mut() {
            if let Some(replica) = record
                .replicas
                .iter_mut()
                .find(|replica| &replica.backend == backend)
            {
                if replica.state == ReplicaState::Available {
                    replica.state = ReplicaState::Draining;
                }
            }
        }
        Ok(())
    }

    pub fn finish_detach(&self, backend: &TierBackendId) -> Result<(), TierRuntimeError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        let entry_state = state
            .backends
            .get(backend)
            .map(|entry| entry.state)
            .ok_or(TierRuntimeError::UnknownBackend)?;
        if entry_state == BackendLifecycleState::Detached {
            return Ok(());
        }
        if entry_state != BackendLifecycleState::Draining {
            return Err(TierRuntimeError::BackendUnavailable);
        }
        if state.inflight.keys().any(|key| &key.destination == backend)
            || state.records.values().any(|record| {
                record.replicas.iter().any(|replica| {
                    &replica.backend == backend
                        && (replica.pins != 0
                            || !matches!(
                                replica.state,
                                ReplicaState::Released | ReplicaState::Revoked
                            ))
                })
            })
        {
            return Err(TierRuntimeError::BackendNotDrained);
        }
        state
            .backends
            .get_mut(backend)
            .expect("entry resolved above")
            .state = BackendLifecycleState::Detached;
        Ok(())
    }

    pub async fn evict(
        &self,
        object: Digest,
        backend: &TierBackendId,
    ) -> Result<(), TierRuntimeError> {
        let (manifest, storage, chunks_to_delete) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| TierRuntimeError::LockPoisoned)?;
            let storage = state
                .backends
                .get(backend)
                .ok_or(TierRuntimeError::UnknownBackend)?
                .backend
                .clone();
            let record = state
                .records
                .get_mut(&object)
                .ok_or(TierRuntimeError::UnknownObject)?;
            let remaining_available = record
                .replicas
                .iter()
                .filter(|replica| {
                    &replica.backend != backend && replica.state == ReplicaState::Available
                })
                .count();
            if remaining_available == 0 {
                return Err(TierRuntimeError::LastReplica);
            }
            let replica = record
                .replicas
                .iter_mut()
                .find(|replica| &replica.backend == backend)
                .ok_or(TierRuntimeError::UnknownReplica)?;
            if replica.pins != 0 {
                return Err(TierRuntimeError::ReplicaPinned);
            }
            replica.state = ReplicaState::Draining;
            let manifest = record.object.clone();
            let chunks_to_delete = manifest
                .chunks
                .iter()
                .filter(|chunk| {
                    let referenced_once = state
                        .chunk_refs
                        .get(&(backend.clone(), chunk.digest))
                        .copied()
                        .unwrap_or(0)
                        <= 1;
                    let reserved = state
                        .chunk_reservations
                        .get(&(backend.clone(), chunk.digest))
                        .copied()
                        .unwrap_or(0)
                        != 0;
                    referenced_once && !reserved
                })
                .cloned()
                .collect::<Vec<_>>();
            for chunk in &chunks_to_delete {
                state
                    .chunk_deletions
                    .insert((backend.clone(), chunk.digest));
            }
            (manifest, storage, chunks_to_delete)
        };
        let _deletion_guard = DeletionGuard {
            controller: self,
            backend: backend.clone(),
            digests: chunks_to_delete.iter().map(|chunk| chunk.digest).collect(),
        };
        for chunk in &chunks_to_delete {
            storage.delete_chunk(chunk).await?;
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| TierRuntimeError::LockPoisoned)?;
        let record = state
            .records
            .get_mut(&object)
            .ok_or(TierRuntimeError::UnknownObject)?;
        let replica = record
            .replicas
            .iter_mut()
            .find(|replica| &replica.backend == backend)
            .ok_or(TierRuntimeError::UnknownReplica)?;
        replica.state = ReplicaState::Released;
        for chunk in &manifest.chunks {
            let key = (backend.clone(), chunk.digest);
            match state.chunk_refs.get_mut(&key) {
                Some(references) if *references > 1 => *references -= 1,
                Some(_) => {
                    state.chunk_refs.remove(&key);
                }
                None => {}
            }
        }
        Ok(())
    }

    fn validate_authorization(
        record: &ResidencyRecord,
        authorization: Option<TierTransferAuthorization>,
    ) -> Result<(), TierRuntimeError> {
        match (record.active_write, authorization) {
            (None, None) => Ok(()),
            (Some(binding), Some(authority))
                if binding.generation == authority.generation
                    && binding.placement_epoch == authority.placement_epoch
                    && binding.fencing_token == authority.fencing_token =>
            {
                Ok(())
            }
            _ => Err(TierRuntimeError::StaleWriteBinding),
        }
    }

    async fn copy_manifest(
        &self,
        manifest: &TierObjectManifest,
        source: &TierBackendId,
        destination: &TierBackendId,
        source_backend: Arc<dyn TierBackend>,
        destination_backend: Arc<dyn TierBackend>,
    ) -> Result<TierTransferReceipt, TierRuntimeError> {
        let mut bytes = 0u64;
        for chunk in &manifest.chunks {
            let payload = source_backend.get_chunk(chunk).await?;
            destination_backend
                .put_chunk(chunk.clone(), payload)
                .await?;
            let report = destination_backend.verify_chunk(chunk).await?;
            if !report.valid {
                return Err(TierRuntimeError::Storage(TierError::CorruptChunk));
            }
            bytes = bytes.saturating_add(chunk.length);
        }
        manifest.validate()?;
        Ok(TierTransferReceipt {
            object: manifest.root_digest,
            source: source.clone(),
            destination: destination.clone(),
            chunks: manifest.chunks.len(),
            bytes,
        })
    }

    /// The publish-time checks of a transfer; spends the grant when all of them pass, so a
    /// refusal never spends it. The caller holds the controller lock.
    fn recheck_and_spend(
        state: &mut ControllerState,
        manifest: &TierObjectManifest,
        publish: &PublishCheck,
        now_millis: u64,
        limits: &TierGrantLimits,
    ) -> Result<(), TierRuntimeError> {
        let record = state
            .records
            .get(&manifest.root_digest)
            .ok_or(TierRuntimeError::UnknownObject)?;
        if Self::validate_authorization(record, publish.authorization).is_err() {
            return Err(TierRuntimeError::AuthorizationRevoked);
        }
        Self::validate_scope(state, &manifest.root_digest, publish.grant.as_ref())?;
        let Some(grant) = publish.grant.as_ref() else {
            return Ok(());
        };
        if now_millis > grant.deadline_millis {
            return Err(TierRuntimeError::GrantExpired);
        }
        state.tombstones.consume(
            &grant.receiver,
            grant.nonce,
            grant.deadline_millis,
            now_millis,
            limits,
        )
    }

    fn finish_transfer(
        &self,
        key: &TransferKey,
        source: &TierBackendId,
        destination: &TierBackendId,
        manifest: &TierObjectManifest,
        reserved_digests: &[(Digest, u64)],
        publish: &PublishCheck,
        result: Result<TierTransferReceipt, TierRuntimeError>,
    ) -> Result<TierTransferReceipt, TierRuntimeError> {
        let now = self.clock.now_millis();
        let Ok(mut state) = self.state.lock() else {
            return Err(TierRuntimeError::LockPoisoned);
        };
        let destination_available = state
            .backends
            .get(destination)
            .is_some_and(|entry| entry.state == BackendLifecycleState::Available);
        let result = match result {
            Ok(_) if !destination_available => Err(TierRuntimeError::BackendDraining),
            // Everything the transfer was authorized by is checked again here, under the same
            // lock hold that publishes the replica, and the grant is spent in it.
            Ok(receipt) => {
                Self::recheck_and_spend(&mut state, manifest, publish, now, &self.grant_limits)
                    .map(|()| receipt)
            }
            other => other,
        };
        for (digest, reserved) in reserved_digests {
            let reservation_key = (destination.clone(), *digest);
            match state.chunk_reservations.get_mut(&reservation_key) {
                Some(count) if *count > *reserved => *count -= *reserved,
                Some(_) => {
                    state.chunk_reservations.remove(&reservation_key);
                }
                None => {}
            }
        }
        state.access_clock = state.access_clock.saturating_add(1);
        let access = state.access_clock;
        let mut committed_destination = false;
        if let Some(record) = state.records.get_mut(&manifest.root_digest) {
            if let Some(source_replica) = record
                .replicas
                .iter_mut()
                .find(|replica| &replica.backend == source)
            {
                source_replica.pins = source_replica.pins.saturating_sub(1);
            }
            if let Some(destination_replica) = record
                .replicas
                .iter_mut()
                .find(|replica| &replica.backend == destination)
            {
                match result {
                    Ok(_) => {
                        committed_destination =
                            destination_replica.state != ReplicaState::Available;
                        destination_replica.state = ReplicaState::Available;
                        destination_replica.verified_digest = manifest.root_digest;
                        destination_replica.last_access = access;
                    }
                    Err(ref error) if error.refuses_publication() => {
                        destination_replica.state = ReplicaState::Revoked;
                    }
                    Err(_) => destination_replica.state = ReplicaState::Corrupt,
                }
            }
        }
        if committed_destination {
            for chunk in &manifest.chunks {
                let references = state
                    .chunk_refs
                    .entry((destination.clone(), chunk.digest))
                    .or_default();
                *references = references.saturating_add(1);
            }
        }
        state.inflight.remove(key);
        state.inflight_bytes = state
            .inflight_bytes
            .saturating_sub(manifest.chunks.iter().map(|chunk| chunk.length).sum());
        result
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierPlanningContext {
    pub requested_tiers: Vec<StorageTier>,
    pub lookahead: Vec<Digest>,
    pub eviction_candidates: Vec<(Digest, u64)>,
    pub available_replicas: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TierPlan {
    pub destinations: Vec<StorageTier>,
    pub prefetch: Vec<Digest>,
    pub evict: Option<Digest>,
    pub minimum_replicas: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TierPolicyError {
    EmptyPlan,
    ReplicaBudgetUnsatisfied,
}

pub trait TierPlacementPolicy {
    fn plan(&self, input: &TierPlanningContext) -> Result<TierPlan, TierPolicyError>;

    fn choose_eviction(&self, candidates: &[(Digest, u64)]) -> Option<Digest>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicLruPolicy;

impl TierPlacementPolicy for DeterministicLruPolicy {
    fn plan(&self, input: &TierPlanningContext) -> Result<TierPlan, TierPolicyError> {
        Ok(TierPlan {
            destinations: input.requested_tiers.clone(),
            prefetch: Vec::new(),
            evict: self.choose_eviction(&input.eviction_candidates),
            minimum_replicas: 1,
        })
    }

    fn choose_eviction(&self, candidates: &[(Digest, u64)]) -> Option<Digest> {
        candidates
            .iter()
            .min_by(|(left_digest, left_access), (right_digest, right_access)| {
                left_access
                    .cmp(right_access)
                    .then_with(|| left_digest.cmp(right_digest))
            })
            .map(|(digest, _)| *digest)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExplicitTierPolicy {
    pub destinations: Vec<StorageTier>,
}

impl TierPlacementPolicy for ExplicitTierPolicy {
    fn plan(&self, _input: &TierPlanningContext) -> Result<TierPlan, TierPolicyError> {
        if self.destinations.is_empty() {
            return Err(TierPolicyError::EmptyPlan);
        }
        Ok(TierPlan {
            destinations: self.destinations.clone(),
            prefetch: Vec::new(),
            evict: None,
            minimum_replicas: 1,
        })
    }

    fn choose_eviction(&self, _candidates: &[(Digest, u64)]) -> Option<Digest> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LookaheadPrefetchPolicy {
    pub max_objects: usize,
}

impl TierPlacementPolicy for LookaheadPrefetchPolicy {
    fn plan(&self, input: &TierPlanningContext) -> Result<TierPlan, TierPolicyError> {
        if input.requested_tiers.is_empty() {
            return Err(TierPolicyError::EmptyPlan);
        }
        Ok(TierPlan {
            destinations: input.requested_tiers.clone(),
            prefetch: input
                .lookahead
                .iter()
                .copied()
                .take(self.max_objects)
                .collect(),
            evict: None,
            minimum_replicas: 1,
        })
    }

    fn choose_eviction(&self, _candidates: &[(Digest, u64)]) -> Option<Digest> {
        None
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BudgetedReplicaPolicy {
    pub minimum_replicas: usize,
    pub preferred_tiers: Vec<StorageTier>,
}

impl TierPlacementPolicy for BudgetedReplicaPolicy {
    fn plan(&self, input: &TierPlanningContext) -> Result<TierPlan, TierPolicyError> {
        if self.minimum_replicas == 0 || self.preferred_tiers.is_empty() {
            return Err(TierPolicyError::EmptyPlan);
        }
        let missing = self
            .minimum_replicas
            .saturating_sub(input.available_replicas);
        if missing > self.preferred_tiers.len() {
            return Err(TierPolicyError::ReplicaBudgetUnsatisfied);
        }
        Ok(TierPlan {
            destinations: self.preferred_tiers.iter().copied().take(missing).collect(),
            prefetch: Vec::new(),
            evict: None,
            minimum_replicas: self.minimum_replicas,
        })
    }

    fn choose_eviction(&self, _candidates: &[(Digest, u64)]) -> Option<Digest> {
        None
    }
}
