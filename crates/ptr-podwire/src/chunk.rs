//! Bounded chunking and reassembly of activation frames.
//!
//! One `ActivationFrame` can be larger than `MAX_BODY_BYTES`, so it is cut into chunks, each
//! its own `TypedPayload` of the split-pod allowlist (`ptr_protocol::activation`). A
//! [`FrameAssembler`] is the receiving side. It admits only allowlisted payloads, checks
//! every chunk against the signature of the authenticated principal and against the digests
//! the chunk carries, reassembles a frame only from chunks of one frame, and answers every
//! failure with a typed [`Rejection`]. The sender is authenticated before its content is
//! decoded: the principal must be registered, and the signature over the envelope
//! (or over a [`SplitPolicy`]) must verify, before any chunk bytes are copied; the chain the
//! envelope signs must name the one stage the principal is bound to, and must be the chain
//! inside the reassembled frame.
//!
//! **State is bounded and principal-scoped.** The first accepted chunk of a frame reserves
//! `(principal, parent_operation_id, frame_id)` in the principal's table. A reservation, a
//! completion tombstone and an abandonment tombstone each take one slot of
//! `max_frame_tombstones_per_principal`, so `incomplete frames + retained tombstones` never
//! exceeds it; when no slot is free a new frame is refused with
//! [`Rejection::FrameTableFull`] and no tombstone is ever evicted before its retention.
//! Buffered chunks are bounded per frame by the reorder window and per principal by
//! `max_chunks_in_flight`.
//!
//! **Replay.** A completed frame leaves a completion tombstone, retained until its signed
//! deadline plus the clock skew bound; an abandoned frame (explicit, mismatching, or past
//! its deadline) leaves an abandonment tombstone, retained until the abandonment time plus
//! the replay window. A chunk of a tombstoned tuple is refused whatever request id or
//! session carried it. A frame whose signed deadline has passed is refused before any tuple
//! is reserved.
//!
//! **No hidden timer.** The caller passes the time of every call; nothing here reads a
//! clock or sleeps. Expiry happens at the start of each call and in [`FrameAssembler::expire`].
//! The assembler keeps the highest time it was given and never goes back: a call with an
//! earlier time is treated as made at that highest time, so a clock that steps back cannot
//! revive a tombstone that has already been dropped.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use ptr_protocol::activation::{
    admit_view, chunk_digest, envelope_signing_bytes, frame_digest, policy_signing_bytes,
    ActivationChunk, ActivationFrame, AllowlistedView, ChunkHeader, SchemaError, SplitPolicy,
    UpstreamChain, MAX_CHUNKS_PER_FRAME,
};
use ptr_protocol::TypedPayload;
use std::collections::{BTreeMap, BTreeSet};

/// The preregistered limits of a [`FrameAssembler`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssemblyPolicy {
    /// A chunk is accepted only if its index is below the lowest missing index plus this.
    pub reorder_window: u32,
    /// Incomplete frames one principal may hold at once.
    pub max_incomplete_frames_per_peer: usize,
    /// Chunks one principal may have buffered at once, over all its incomplete frames.
    pub max_chunks_in_flight: usize,
    /// Reservations plus retained tombstones one principal may hold.
    pub max_frame_tombstones_per_principal: usize,
    /// How far ahead of now a signed deadline may lie.
    pub max_frame_deadline_lifetime_ms: u64,
    /// The clock skew the receiver tolerates in either direction.
    pub max_clock_skew_ms: u64,
    /// How long an abandonment tombstone is retained.
    pub replay_window_ms: u64,
    /// Bound on the bytes of one frame's chunks.
    pub max_frame_bytes: u64,
    /// Bound on the bytes buffered over all principals together.
    pub max_total_buffered_bytes: u64,
}

/// A policy that violates its own relations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyViolation {
    /// A bound that must be positive is zero.
    Zero(&'static str),
    /// `replay_window_ms < max_frame_deadline_lifetime_ms + 2 * max_clock_skew_ms`: a replay
    /// after the tombstone expired could still carry a live deadline.
    ReplayWindowTooShort,
    /// `max_total_buffered_bytes < max_frame_bytes`: no frame of the largest size could ever
    /// be delivered.
    TotalBelowOneFrame,
}

impl AssemblyPolicy {
    /// Check the relations the design needs.
    pub fn validate(&self) -> Result<(), PolicyViolation> {
        for (name, value) in [
            ("reorder_window", u64::from(self.reorder_window)),
            (
                "max_incomplete_frames_per_peer",
                self.max_incomplete_frames_per_peer as u64,
            ),
            ("max_chunks_in_flight", self.max_chunks_in_flight as u64),
            (
                "max_frame_tombstones_per_principal",
                self.max_frame_tombstones_per_principal as u64,
            ),
            (
                "max_frame_deadline_lifetime_ms",
                self.max_frame_deadline_lifetime_ms,
            ),
            ("replay_window_ms", self.replay_window_ms),
            ("max_frame_bytes", self.max_frame_bytes),
            ("max_total_buffered_bytes", self.max_total_buffered_bytes),
        ] {
            if value == 0 {
                return Err(PolicyViolation::Zero(name));
            }
        }
        if self.replay_window_ms
            < self
                .max_frame_deadline_lifetime_ms
                .saturating_add(self.max_clock_skew_ms.saturating_mul(2))
        {
            return Err(PolicyViolation::ReplayWindowTooShort);
        }
        if self.max_total_buffered_bytes < self.max_frame_bytes {
            return Err(PolicyViolation::TotalBelowOneFrame);
        }
        Ok(())
    }
}

/// Which header field of a later chunk disagrees with the frame's first chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MismatchField {
    ChunkTotal,
    FrameDigest,
    Deadline,
}

/// Why a payload was not taken. Every variant is a typed answer; none is a silent drop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rejection {
    /// The `type_id` is not on the split-pod allowlist.
    NotAllowlisted(String),
    /// The bytes are no instance of the schema their allowlisted `type_id` names.
    Malformed(SchemaError),
    /// No signing key is registered for the authenticated principal.
    UnknownPrincipal,
    /// The signature over the envelope does not verify under the principal's key.
    BadSignature,
    /// The chunk digest does not match the chunk's envelope and bytes.
    BadChunkDigest,
    /// The signed deadline plus the skew bound has passed.
    Expired,
    /// The chain names a stage the authenticated principal is not bound to.
    StageNotBound,
    /// Accepting the chunk would exceed the bytes buffered over all principals.
    GlobalBufferFull,
    /// The chain inside the reassembled frame is not the chain the signed envelope carried.
    /// The frame is abandoned.
    ChainFrameMismatch,
    /// The reassembled frame disagrees with the signed policy of its operation (width or
    /// dtype). The frame is abandoned.
    PolicyMismatch,
    /// The signed deadline lies further ahead than the policy allows.
    DeadlineTooFar,
    /// The frame already completed (a completion tombstone holds its tuple).
    FrameReplayed,
    /// The frame was abandoned (an abandonment tombstone holds its tuple).
    FrameAbandoned,
    /// No slot is free in the principal's table for a new frame.
    FrameTableFull,
    /// The principal already holds the maximum of incomplete frames.
    TooManyIncompleteFrames,
    /// The principal already has the maximum of chunks buffered.
    QueueOverflow,
    /// The chunk's index is outside the reorder window.
    OutOfWindow { index: u32, lowest_missing: u32 },
    /// The chunk's index was already received for this frame.
    DuplicateChunk,
    /// The chunk's upstream chain differs from the frame's first chunk. The frame is
    /// abandoned.
    ChainMismatch,
    /// A header field of the chunk differs from the frame's first chunk. The frame is
    /// abandoned.
    FrameMismatch(MismatchField),
    /// The chunks of one frame together exceed the byte bound. The frame is abandoned.
    FrameTooLarge,
    /// The digest of the complete reassembled bytes is not the digest every chunk carried.
    /// The frame is abandoned.
    FrameDigestMismatch,
    /// The reassembled bytes are no valid frame (shape, positions). The frame is abandoned.
    InvalidFrame(SchemaError),
}

impl Rejection {
    /// A short stable name, for records.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NotAllowlisted(_) => "not_allowlisted",
            Self::Malformed(_) => "malformed",
            Self::UnknownPrincipal => "unknown_principal",
            Self::BadSignature => "bad_signature",
            Self::BadChunkDigest => "bad_chunk_digest",
            Self::Expired => "expired",
            Self::StageNotBound => "stage_not_bound",
            Self::GlobalBufferFull => "global_buffer_full",
            Self::ChainFrameMismatch => "chain_frame_mismatch",
            Self::PolicyMismatch => "policy_mismatch",
            Self::DeadlineTooFar => "deadline_too_far",
            Self::FrameReplayed => "frame_replayed",
            Self::FrameAbandoned => "frame_abandoned",
            Self::FrameTableFull => "frame_table_full",
            Self::TooManyIncompleteFrames => "too_many_incomplete_frames",
            Self::QueueOverflow => "queue_overflow",
            Self::OutOfWindow { .. } => "out_of_window",
            Self::DuplicateChunk => "duplicate_chunk",
            Self::ChainMismatch => "chain_mismatch",
            Self::FrameMismatch(_) => "frame_mismatch",
            Self::FrameTooLarge => "frame_too_large",
            Self::FrameDigestMismatch => "frame_digest_mismatch",
            Self::InvalidFrame(_) => "invalid_frame",
        }
    }
}

/// A frame reassembled from all of its chunks, once.
#[derive(Clone, Debug, PartialEq)]
pub struct DeliveredFrame {
    pub principal: String,
    pub parent_operation_id: u64,
    pub frame_id: u64,
    pub frame_digest: [u8; 32],
    pub frame: ActivationFrame,
    /// The bytes the chunks carried, in order (the canonical frame bytes).
    pub bytes: Vec<u8>,
    /// How many distinct sessions contributed chunks.
    pub sessions: usize,
}

/// What an accepted payload did.
#[derive(Clone, Debug, PartialEq)]
pub enum Offered {
    /// A policy payload of the allowlist, decoded; it holds no assembly state.
    Policy(SplitPolicy),
    /// The chunk was buffered; the frame is still incomplete.
    Buffered { received: u32, total: u32 },
    /// The chunk completed its frame.
    Delivered(Box<DeliveredFrame>),
}

/// The identity of one chunk, for the release record.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ChunkId {
    pub principal: String,
    pub parent_operation_id: u64,
    pub frame_id: u64,
    pub chunk_index: u32,
}

/// What an abandonment released.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Released {
    pub chunks: Vec<ChunkId>,
    pub bytes: usize,
}

/// Why [`FrameAssembler::abandon`] did nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbandonError {
    UnknownPrincipal,
    /// No incomplete frame holds the tuple.
    NoSuchFrame,
}

struct Assembly {
    total: u32,
    frame_digest: [u8; 32],
    deadline_ms: u64,
    chain: UpstreamChain,
    chunks: BTreeMap<u32, Vec<u8>>,
    bytes: usize,
    sessions: BTreeSet<u64>,
}

#[derive(Clone, Copy)]
enum Tombstone {
    Completed { retain_until_ms: u64 },
    Abandoned { retain_until_ms: u64 },
}

impl Tombstone {
    fn retain_until(&self) -> u64 {
        match self {
            Self::Completed { retain_until_ms } | Self::Abandoned { retain_until_ms } => {
                *retain_until_ms
            }
        }
    }
}

struct PrincipalTable {
    key: VerifyingKey,
    /// The one stage identity this principal may speak for (`chain.stage.identity`).
    stage_identity: String,
    /// Verified policies by operation, kept until their signed deadline plus the skew.
    policies: BTreeMap<u64, SplitPolicy>,
    assemblies: BTreeMap<(u64, u64), Assembly>,
    tombstones: BTreeMap<(u64, u64), Tombstone>,
}

/// The receiving side of the chunk protocol.
pub struct FrameAssembler {
    policy: AssemblyPolicy,
    tables: BTreeMap<String, PrincipalTable>,
    released: Vec<ChunkId>,
    /// The highest time any call carried; time never goes back.
    clock_ms: u64,
}

impl FrameAssembler {
    pub fn new(policy: AssemblyPolicy) -> Result<Self, PolicyViolation> {
        policy.validate()?;
        Ok(Self {
            policy,
            tables: BTreeMap::new(),
            released: Vec::new(),
            clock_ms: 0,
        })
    }

    /// The call's time, clamped to the highest time seen so far.
    fn advance(&mut self, now_ms: u64) -> u64 {
        self.clock_ms = self.clock_ms.max(now_ms);
        self.clock_ms
    }

    pub fn policy(&self) -> &AssemblyPolicy {
        &self.policy
    }

    /// Register the verifying key of the upstream runtime of an authenticated principal and
    /// the one stage identity it may speak for. Both are whatever the deployment wrote down
    /// for that principal, never a value read from a payload.
    ///
    /// Registering a principal again with the same key and stage identity (a reconnect)
    /// keeps everything the table remembers: the completion and abandonment tombstones (so a
    /// captured, already delivered frame is still refused as a replay), the verified policies
    /// and the incomplete frames. If the key or the stage identity changes, what was minted
    /// under the old one is no longer trusted: the policies are dropped, the frames still
    /// being assembled are released (their chunks are reported once, through
    /// [`Self::expire`]) and their tuples tombstoned until their signed deadline plus the
    /// skew, so no chunk signed under the new key completes a frame begun under the old one
    /// and a policy signed under a compromised old key stops binding. The tombstones are
    /// kept, so a replay stays refused.
    ///
    /// Registering a principal id the assembler has not seen, with a `(key, stage_identity)`
    /// pair that an existing table holds, is the same upstream runtime behind a new endpoint
    /// identity (a process restart draws a new one): the new table inherits that table's
    /// tombstones that have not expired (bounded by `max_frame_tombstones_per_principal`), so a
    /// captured frame is still refused as a replay. Policies and incomplete frames are not
    /// inherited. Tombstones are in memory: a restarted receiver has none, and a replay is then
    /// bounded by the signed deadline plus the skew. The model is one runtime per
    /// `(key, stage_identity)`: two live principals that share both only see the tombstones of
    /// those registered before them, which is harmless for a replay guard.
    pub fn register_principal(&mut self, principal: &str, key: VerifyingKey, stage_identity: &str) {
        let skew = self.policy.max_clock_skew_ms;
        let Some(table) = self.tables.get_mut(principal) else {
            // A new principal id for a (key, stage) pair an existing table already holds is the
            // same upstream runtime behind a new endpoint identity (a process restart draws a
            // new one). It inherits the tombstones that have not expired, so a captured frame is
            // still a replay; nothing else (policies, incomplete frames) is carried over.
            let tombstones = self.inherited_tombstones(&key, stage_identity);
            self.tables.insert(
                principal.to_owned(),
                PrincipalTable {
                    key,
                    stage_identity: stage_identity.to_owned(),
                    policies: BTreeMap::new(),
                    assemblies: BTreeMap::new(),
                    tombstones,
                },
            );
            return;
        };
        let changed = table.key != key || table.stage_identity != stage_identity;
        table.key = key;
        if changed {
            table.stage_identity = stage_identity.to_owned();
            table.policies.clear();
            for (tuple, assembly) in std::mem::take(&mut table.assemblies) {
                self.released
                    .extend(assembly.chunks.keys().map(|index| ChunkId {
                        principal: principal.to_owned(),
                        parent_operation_id: tuple.0,
                        frame_id: tuple.1,
                        chunk_index: *index,
                    }));
                table.tombstones.insert(
                    tuple,
                    Tombstone::Abandoned {
                        retain_until_ms: assembly.deadline_ms.saturating_add(skew),
                    },
                );
            }
        }
    }

    /// A tombstone for `tuple` in a table of another principal id that holds this exact
    /// `(key, stage_identity)`, a completion before an abandonment (the principal's own table
    /// is checked first, on its own). Expired tombstones are already swept from every table.
    fn sibling_tombstone(
        &self,
        principal: &str,
        key: &VerifyingKey,
        stage_identity: &str,
        tuple: (u64, u64),
    ) -> Option<Tombstone> {
        let mut found = None;
        for (_, table) in self.tables.iter().filter(|(name, table)| {
            name.as_str() != principal
                && table.key == *key
                && table.stage_identity == stage_identity
        }) {
            match table.tombstones.get(&tuple) {
                Some(completed @ Tombstone::Completed { .. }) => return Some(*completed),
                Some(abandoned) => found = Some(*abandoned),
                None => {}
            }
        }
        found
    }

    /// The unexpired tombstones of every table that holds this exact `(key, stage_identity)`,
    /// merged per tuple (the longer retention wins) and cut to
    /// `max_frame_tombstones_per_principal`, keeping the longest retained when it must cut.
    /// Tombstones live in memory only: a restarted receiver has none, and a replay is then
    /// bounded by the signed deadline plus the skew alone.
    fn inherited_tombstones(
        &self,
        key: &VerifyingKey,
        stage_identity: &str,
    ) -> BTreeMap<(u64, u64), Tombstone> {
        let mut merged: BTreeMap<(u64, u64), Tombstone> = BTreeMap::new();
        for table in self
            .tables
            .values()
            .filter(|table| table.key == *key && table.stage_identity == stage_identity)
        {
            for (tuple, tombstone) in &table.tombstones {
                let retain_until_ms = tombstone.retain_until();
                if retain_until_ms <= self.clock_ms {
                    continue;
                }
                let copy = match tombstone {
                    Tombstone::Completed { .. } => Tombstone::Completed { retain_until_ms },
                    Tombstone::Abandoned { .. } => Tombstone::Abandoned { retain_until_ms },
                };
                match merged.get(tuple) {
                    Some(held) if held.retain_until() >= retain_until_ms => {}
                    _ => {
                        merged.insert(*tuple, copy);
                    }
                }
            }
        }
        let limit = self.policy.max_frame_tombstones_per_principal;
        if merged.len() > limit {
            let mut by_retention: Vec<_> = merged.into_iter().collect();
            by_retention.sort_by_key(|(tuple, tombstone)| {
                (std::cmp::Reverse(tombstone.retain_until()), *tuple)
            });
            by_retention.truncate(limit);
            merged = by_retention.into_iter().collect();
        }
        merged
    }

    /// Take one payload that arrived at time `now_ms` from `principal` over `session`.
    pub fn offer(
        &mut self,
        now_ms: u64,
        principal: &str,
        session: u64,
        payload: &TypedPayload,
    ) -> Result<Offered, Rejection> {
        let now_ms = self.advance(now_ms);
        self.sweep(now_ms);
        let policy = self.policy;
        let total_buffered = self.buffered_bytes() as u64;
        // Authenticate the sender before decoding anything it sent.
        let table = self
            .tables
            .get_mut(principal)
            .ok_or(Rejection::UnknownPrincipal)?;
        let view = match admit_view(&payload.type_id, &payload.bytes) {
            Ok(AllowlistedView::Chunk(view)) => view,
            Ok(AllowlistedView::Policy(split_policy)) => {
                return take_policy(table, &policy, now_ms, split_policy)
            }
            Err(SchemaError::NotAllowlisted(name)) => return Err(Rejection::NotAllowlisted(name)),
            Err(error) => return Err(Rejection::Malformed(error)),
        };
        let header = &view.header;
        let signature = Signature::from_bytes(&view.signature);
        if table
            .key
            .verify_strict(&envelope_signing_bytes(header), &signature)
            .is_err()
        {
            return Err(Rejection::BadSignature);
        }
        if header.upstream_chain.stage.identity != table.stage_identity {
            return Err(Rejection::StageNotBound);
        }
        let tuple = (header.parent_operation_id, header.frame_id);
        match table.tombstones.get(&tuple) {
            Some(Tombstone::Completed { .. }) => return Err(Rejection::FrameReplayed),
            Some(Tombstone::Abandoned { .. }) => return Err(Rejection::FrameAbandoned),
            None => {}
        }
        // Replay state is shared by `(key, stage_identity)`: a tombstone written under another
        // principal id of the same upstream runtime (an old endpoint identity whose frame
        // completed after this one was registered) refuses the tuple here too.
        let (key, stage) = (table.key, table.stage_identity.clone());
        if let Some(tombstone) = self.sibling_tombstone(principal, &key, &stage, tuple) {
            // The tuple is closed under a sibling identity: a partial assembly this table
            // holds for it can never complete, so its chunks and capacity are released now
            // rather than at the deadline sweep.
            if let Some(table) = self.tables.get_mut(principal) {
                self.released
                    .extend(release_assembly(table, principal, tuple));
            }
            return Err(match tombstone {
                Tombstone::Completed { .. } => Rejection::FrameReplayed,
                Tombstone::Abandoned { .. } => Rejection::FrameAbandoned,
            });
        }
        let table = self
            .tables
            .get_mut(principal)
            .ok_or(Rejection::UnknownPrincipal)?;
        if is_expired(header.signed_absolute_deadline_ms, &policy, now_ms) {
            return Err(Rejection::Expired);
        }
        if header.signed_absolute_deadline_ms
            > now_ms
                .saturating_add(policy.max_frame_deadline_lifetime_ms)
                .saturating_add(policy.max_clock_skew_ms)
        {
            return Err(Rejection::DeadlineTooFar);
        }
        if chunk_digest(header, view.bytes) != header.chunk_digest {
            return Err(Rejection::BadChunkDigest);
        }
        if total_buffered.saturating_add(view.bytes.len() as u64) > policy.max_total_buffered_bytes
        {
            return Err(Rejection::GlobalBufferFull);
        }
        match table.assemblies.get(&tuple) {
            None => {
                if table.assemblies.len() >= policy.max_incomplete_frames_per_peer {
                    return Err(Rejection::TooManyIncompleteFrames);
                }
                if table.assemblies.len() + table.tombstones.len()
                    >= policy.max_frame_tombstones_per_principal
                {
                    return Err(Rejection::FrameTableFull);
                }
                if header.chunk_index >= policy.reorder_window {
                    return Err(Rejection::OutOfWindow {
                        index: header.chunk_index,
                        lowest_missing: 0,
                    });
                }
                if in_flight(table) >= policy.max_chunks_in_flight {
                    return Err(Rejection::QueueOverflow);
                }
                if view.bytes.len() as u64 > policy.max_frame_bytes {
                    return Err(Rejection::FrameTooLarge);
                }
                table.assemblies.insert(
                    tuple,
                    Assembly {
                        total: header.chunk_total,
                        frame_digest: header.frame_digest,
                        deadline_ms: header.signed_absolute_deadline_ms,
                        chain: header.upstream_chain.clone(),
                        chunks: BTreeMap::new(),
                        bytes: 0,
                        sessions: BTreeSet::new(),
                    },
                );
            }
            Some(assembly) => {
                if assembly.chain != header.upstream_chain {
                    let released = abandon_assembly(table, principal, tuple, now_ms, &policy);
                    self.released.extend(released.chunks);
                    return Err(Rejection::ChainMismatch);
                }
                let mismatch = if assembly.total != header.chunk_total {
                    Some(MismatchField::ChunkTotal)
                } else if assembly.frame_digest != header.frame_digest {
                    Some(MismatchField::FrameDigest)
                } else if assembly.deadline_ms != header.signed_absolute_deadline_ms {
                    Some(MismatchField::Deadline)
                } else {
                    None
                };
                if let Some(field) = mismatch {
                    let released = abandon_assembly(table, principal, tuple, now_ms, &policy);
                    self.released.extend(released.chunks);
                    return Err(Rejection::FrameMismatch(field));
                }
                if assembly.chunks.contains_key(&header.chunk_index) {
                    return Err(Rejection::DuplicateChunk);
                }
                let lowest_missing = (0..).find(|index| !assembly.chunks.contains_key(index));
                let lowest_missing = lowest_missing.unwrap_or(0);
                if header.chunk_index >= lowest_missing.saturating_add(policy.reorder_window) {
                    return Err(Rejection::OutOfWindow {
                        index: header.chunk_index,
                        lowest_missing,
                    });
                }
                if in_flight(table) >= policy.max_chunks_in_flight {
                    return Err(Rejection::QueueOverflow);
                }
                if assembly.bytes as u64 + view.bytes.len() as u64 > policy.max_frame_bytes {
                    let released = abandon_assembly(table, principal, tuple, now_ms, &policy);
                    self.released.extend(released.chunks);
                    return Err(Rejection::FrameTooLarge);
                }
            }
        }
        let assembly = table
            .assemblies
            .get_mut(&tuple)
            .expect("the assembly was reserved or found above");
        assembly.bytes += view.bytes.len();
        assembly.sessions.insert(session);
        assembly
            .chunks
            .insert(header.chunk_index, view.bytes.to_vec());
        if assembly.chunks.len() < assembly.total as usize {
            return Ok(Offered::Buffered {
                received: assembly.chunks.len() as u32,
                total: assembly.total,
            });
        }
        // Every chunk is here: reassemble from the complete bytes and judge them as a whole.
        let finished = table
            .assemblies
            .remove(&tuple)
            .expect("the assembly is present");
        let mut bytes = Vec::with_capacity(finished.bytes);
        for part in finished.chunks.values() {
            bytes.extend_from_slice(part);
        }
        let digest = frame_digest(&bytes);
        let failure = if digest != finished.frame_digest {
            Some(Rejection::FrameDigestMismatch)
        } else {
            None
        };
        let decoded = match failure {
            Some(failure) => Err(failure),
            None => ActivationFrame::from_bytes(&bytes)
                .map_err(Rejection::InvalidFrame)
                .and_then(|frame| {
                    if frame.upstream_chain != finished.chain {
                        return Err(Rejection::ChainFrameMismatch);
                    }
                    match table.policies.get(&tuple.0) {
                        Some(bound) if bound.width != frame.width || bound.dtype != frame.dtype => {
                            Err(Rejection::PolicyMismatch)
                        }
                        _ => Ok(frame),
                    }
                }),
        };
        match decoded {
            Ok(frame) => {
                table.tombstones.insert(
                    tuple,
                    Tombstone::Completed {
                        retain_until_ms: finished
                            .deadline_ms
                            .saturating_add(policy.max_clock_skew_ms),
                    },
                );
                Ok(Offered::Delivered(Box::new(DeliveredFrame {
                    principal: principal.to_owned(),
                    parent_operation_id: tuple.0,
                    frame_id: tuple.1,
                    frame_digest: digest,
                    frame,
                    bytes,
                    sessions: finished.sessions.len(),
                })))
            }
            Err(rejection) => {
                for index in finished.chunks.keys() {
                    self.released.push(ChunkId {
                        principal: principal.to_owned(),
                        parent_operation_id: tuple.0,
                        frame_id: tuple.1,
                        chunk_index: *index,
                    });
                }
                table.tombstones.insert(
                    tuple,
                    Tombstone::Abandoned {
                        retain_until_ms: now_ms.saturating_add(policy.replay_window_ms),
                    },
                );
                Err(rejection)
            }
        }
    }

    /// Abandon the incomplete frame `(principal, parent_operation_id, frame_id)` at
    /// `now_ms`: its chunks are released and its tuple is tombstoned until `now_ms` plus the
    /// replay window. The released chunks are reported by the returned value and only by it:
    /// a later [`Self::expire`] does not list them again.
    pub fn abandon(
        &mut self,
        now_ms: u64,
        principal: &str,
        parent_operation_id: u64,
        frame_id: u64,
    ) -> Result<Released, AbandonError> {
        let now_ms = self.advance(now_ms);
        self.sweep(now_ms);
        let policy = self.policy;
        let table = self
            .tables
            .get_mut(principal)
            .ok_or(AbandonError::UnknownPrincipal)?;
        let tuple = (parent_operation_id, frame_id);
        if !table.assemblies.contains_key(&tuple) {
            return Err(AbandonError::NoSuchFrame);
        }
        // Reported by this return value only, never again through `expire`.
        Ok(abandon_assembly(table, principal, tuple, now_ms, &policy))
    }

    /// Abandon every incomplete frame whose signed deadline plus the skew bound has passed,
    /// and drop every tombstone whose retention has ended. Returns the chunks released since
    /// the last call that drained them, each exactly once: by expiry (here or inside an
    /// earlier [`Self::offer`] or [`Self::abandon`]), by a changed key or stage identity, or by
    /// a frame that failed to decode. Chunks released by [`Self::abandon`] itself are reported
    /// by its return value, not here.
    pub fn expire(&mut self, now_ms: u64) -> Vec<ChunkId> {
        let now_ms = self.advance(now_ms);
        self.sweep(now_ms);
        std::mem::take(&mut self.released)
    }

    /// The expiry work of [`Self::expire`] without draining the release buffer, so `offer` and
    /// `abandon` never swallow ids a caller learns only through `expire`.
    fn sweep(&mut self, now_ms: u64) {
        let policy = self.policy;
        for (principal, table) in &mut self.tables {
            let due: Vec<(u64, u64)> = table
                .assemblies
                .iter()
                .filter(|(_, assembly)| is_expired(assembly.deadline_ms, &policy, now_ms))
                .map(|(tuple, _)| *tuple)
                .collect();
            for tuple in due {
                let released = abandon_assembly(table, principal, tuple, now_ms, &policy);
                self.released.extend(released.chunks);
            }
            table
                .tombstones
                .retain(|_, tombstone| tombstone.retain_until() > now_ms);
            table.policies.retain(|_, bound| {
                bound
                    .signed_absolute_deadline_ms
                    .saturating_add(policy.max_clock_skew_ms)
                    > now_ms
            });
        }
    }

    /// Bytes buffered over all principals.
    pub fn buffered_bytes(&self) -> usize {
        self.tables
            .values()
            .flat_map(|table| table.assemblies.values())
            .map(|assembly| assembly.bytes)
            .sum()
    }

    /// Chunks buffered by `principal`.
    pub fn buffered_chunks(&self, principal: &str) -> usize {
        self.tables.get(principal).map_or(0, in_flight)
    }

    /// Incomplete frames `principal` holds.
    pub fn incomplete_frames(&self, principal: &str) -> usize {
        self.tables
            .get(principal)
            .map_or(0, |table| table.assemblies.len())
    }

    /// Tombstones (completed and abandoned) `principal` retains.
    pub fn retained_tombstones(&self, principal: &str) -> usize {
        self.tables
            .get(principal)
            .map_or(0, |table| table.tombstones.len())
    }

    /// Slots of `principal`'s table in use: incomplete frames plus retained tombstones.
    pub fn reserved_slots(&self, principal: &str) -> usize {
        self.incomplete_frames(principal) + self.retained_tombstones(principal)
    }
}

/// A signed policy: verify it, bind it to its operation until its deadline, return it. An
/// unsigned or mis-signed policy is never delivered.
fn take_policy(
    table: &mut PrincipalTable,
    policy: &AssemblyPolicy,
    now_ms: u64,
    split_policy: SplitPolicy,
) -> Result<Offered, Rejection> {
    let signature = Signature::from_bytes(&split_policy.signature);
    if table
        .key
        .verify_strict(&policy_signing_bytes(&split_policy), &signature)
        .is_err()
    {
        return Err(Rejection::BadSignature);
    }
    let deadline = split_policy.signed_absolute_deadline_ms;
    if is_expired(deadline, policy, now_ms) {
        return Err(Rejection::Expired);
    }
    if deadline
        > now_ms
            .saturating_add(policy.max_frame_deadline_lifetime_ms)
            .saturating_add(policy.max_clock_skew_ms)
    {
        return Err(Rejection::DeadlineTooFar);
    }
    match table.policies.get(&split_policy.operation_id) {
        Some(bound) if *bound != split_policy => return Err(Rejection::PolicyMismatch),
        Some(_) => {}
        None => {
            if table.policies.len() >= policy.max_frame_tombstones_per_principal {
                return Err(Rejection::FrameTableFull);
            }
            table
                .policies
                .insert(split_policy.operation_id, split_policy);
        }
    }
    Ok(Offered::Policy(split_policy))
}

/// Whether a signed deadline is dead at `now_ms`: from `deadline + skew` on. A tombstone is
/// retained while `retain_until > now`, with `retain_until >= deadline + skew`, so at every
/// tick a tuple is either tombstoned or expired and a replay never finds a gap between the two.
fn is_expired(deadline_ms: u64, policy: &AssemblyPolicy, now_ms: u64) -> bool {
    deadline_ms.saturating_add(policy.max_clock_skew_ms) <= now_ms
}

fn in_flight(table: &PrincipalTable) -> usize {
    table
        .assemblies
        .values()
        .map(|assembly| assembly.chunks.len())
        .sum()
}

/// Remove the assembly of `tuple`, release its chunks and tombstone the tuple.
fn abandon_assembly(
    table: &mut PrincipalTable,
    principal: &str,
    tuple: (u64, u64),
    now_ms: u64,
    policy: &AssemblyPolicy,
) -> Released {
    let mut released = Released::default();
    if let Some(assembly) = table.assemblies.remove(&tuple) {
        released.bytes = assembly.bytes;
        released.chunks = assembly
            .chunks
            .keys()
            .map(|index| ChunkId {
                principal: principal.to_owned(),
                parent_operation_id: tuple.0,
                frame_id: tuple.1,
                chunk_index: *index,
            })
            .collect();
    }
    table.tombstones.insert(
        tuple,
        Tombstone::Abandoned {
            retain_until_ms: now_ms.saturating_add(policy.replay_window_ms),
        },
    );
    released
}

/// Remove the assembly of `tuple` and return the ids of its chunks, writing no tombstone: the
/// tuple is already refused by a tombstone of a sibling identity.
fn release_assembly(
    table: &mut PrincipalTable,
    principal: &str,
    tuple: (u64, u64),
) -> Vec<ChunkId> {
    table
        .assemblies
        .remove(&tuple)
        .map(|assembly| {
            assembly
                .chunks
                .keys()
                .map(|index| ChunkId {
                    principal: principal.to_owned(),
                    parent_operation_id: tuple.0,
                    frame_id: tuple.1,
                    chunk_index: *index,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Where a frame is going and until when.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameMeta {
    pub frame_id: u64,
    pub parent_operation_id: u64,
    pub signed_absolute_deadline_ms: u64,
}

/// Why a frame could not be cut into chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkingError {
    /// `max_chunk_bytes` is zero.
    ZeroChunkSize,
    /// The frame needs more chunks than a frame may have.
    TooManyChunks { needed: usize },
}

/// Sign `header`'s envelope after filling its chunk digest from `header` and `bytes`.
pub fn sign_chunk(signer: &SigningKey, mut header: ChunkHeader, bytes: Vec<u8>) -> ActivationChunk {
    header.chunk_digest = chunk_digest(&header, &bytes);
    let signature = signer.sign(&envelope_signing_bytes(&header)).to_bytes();
    ActivationChunk {
        header,
        signature,
        bytes,
    }
}

/// Sign a policy: fills its signature from its other fields.
pub fn sign_policy(signer: &SigningKey, mut policy: SplitPolicy) -> SplitPolicy {
    policy.signature = signer.sign(&policy_signing_bytes(&policy)).to_bytes();
    policy
}

/// Re-sign `chunk`'s envelope as it stands (digest included): for a sender that holds the
/// key and wants an envelope that verifies whatever it says.
pub fn resign_envelope(signer: &SigningKey, chunk: &mut ActivationChunk) {
    chunk.signature = signer
        .sign(&envelope_signing_bytes(&chunk.header))
        .to_bytes();
}

/// Cut `frame` into signed chunks of at most `max_chunk_bytes` frame bytes each.
pub fn chunk_frame(
    signer: &SigningKey,
    frame: &ActivationFrame,
    meta: FrameMeta,
    max_chunk_bytes: usize,
) -> Result<Vec<ActivationChunk>, ChunkingError> {
    if max_chunk_bytes == 0 {
        return Err(ChunkingError::ZeroChunkSize);
    }
    let bytes = frame.to_bytes();
    let needed = bytes.len().div_ceil(max_chunk_bytes).max(1);
    if needed > MAX_CHUNKS_PER_FRAME as usize {
        return Err(ChunkingError::TooManyChunks { needed });
    }
    let digest = frame_digest(&bytes);
    let total = needed as u32;
    Ok(bytes
        .chunks(max_chunk_bytes)
        .enumerate()
        .map(|(index, part)| {
            sign_chunk(
                signer,
                ChunkHeader {
                    frame_id: meta.frame_id,
                    parent_operation_id: meta.parent_operation_id,
                    chunk_index: index as u32,
                    chunk_total: total,
                    signed_absolute_deadline_ms: meta.signed_absolute_deadline_ms,
                    frame_digest: digest,
                    chunk_digest: [0; 32],
                    upstream_chain: frame.upstream_chain.clone(),
                },
                part.to_vec(),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_protocol::activation::{chunk_payload, BoundaryDtype, ChainElement};

    const PRINCIPAL: &str = "upstream";

    fn policy() -> AssemblyPolicy {
        AssemblyPolicy {
            reorder_window: 4,
            max_incomplete_frames_per_peer: 8,
            max_chunks_in_flight: 32,
            max_frame_tombstones_per_principal: 64,
            max_frame_deadline_lifetime_ms: 5_000,
            max_clock_skew_ms: 1_000,
            replay_window_ms: 10_000,
            max_frame_bytes: 1 << 20,
            max_total_buffered_bytes: 4 << 20,
        }
    }

    fn element(name: &str, generation: u64) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation,
            digest: [generation as u8; 32],
            fence_epoch: 0,
        }
    }

    fn chain() -> UpstreamChain {
        UpstreamChain {
            stage: element("stage-a", 1),
            base_model: element("model", 1),
            adapters: vec![],
            experts: vec![],
        }
    }

    fn frame(tokens: u32, width: u32, fill: u8) -> ActivationFrame {
        ActivationFrame {
            tokens,
            width,
            dtype: BoundaryDtype::F32,
            positions: (0..tokens).collect(),
            upstream_chain: chain(),
            placement_epoch: 1,
            tensor: (0..tokens * width * 4)
                .map(|byte| (byte as u8).wrapping_add(fill))
                .collect(),
        }
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    fn assembler() -> FrameAssembler {
        let mut assembler = FrameAssembler::new(policy()).unwrap();
        assembler.register_principal(PRINCIPAL, key().verifying_key(), "stage-a");
        assembler
    }

    fn meta(frame_id: u64) -> FrameMeta {
        FrameMeta {
            frame_id,
            parent_operation_id: 100,
            signed_absolute_deadline_ms: 4_000,
        }
    }

    fn chunks(frame_id: u64, fill: u8) -> Vec<ActivationChunk> {
        chunk_frame(&key(), &frame(4, 8, fill), meta(frame_id), 80).unwrap()
    }

    fn offer(
        assembler: &mut FrameAssembler,
        now: u64,
        chunk: &ActivationChunk,
    ) -> Result<Offered, Rejection> {
        assembler.offer(now, PRINCIPAL, 1, &chunk_payload(chunk))
    }

    #[test]
    fn a_frame_in_order_reassembles_to_the_bytes_that_were_sent() {
        let mut assembler = assembler();
        let sent = frame(4, 8, 0);
        let parts = chunks(1, 0);
        assert!(parts.len() > 2);
        let mut delivered = None;
        for (index, chunk) in parts.iter().enumerate() {
            match offer(&mut assembler, 0, chunk).unwrap() {
                Offered::Buffered { received, total } => {
                    assert_eq!(received as usize, index + 1);
                    assert_eq!(total as usize, parts.len());
                }
                Offered::Delivered(frame) => delivered = Some(frame),
                Offered::Policy(_) => unreachable!(),
            }
        }
        let delivered = delivered.expect("the last chunk completes the frame");
        assert_eq!(delivered.frame, sent);
        assert_eq!(delivered.bytes, sent.to_bytes());
        assert_eq!(delivered.frame_digest, frame_digest(&sent.to_bytes()));
        assert_eq!(assembler.buffered_bytes(), 0);
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 1);
    }

    #[test]
    fn chunks_within_the_window_may_arrive_out_of_order() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        assert_eq!(parts.len(), 4);
        let mut order: Vec<usize> = (0..parts.len()).collect();
        order.swap(0, 1);
        order.swap(2, 3);
        let mut last = None;
        for index in order {
            last = Some(offer(&mut assembler, 0, &parts[index]).unwrap());
        }
        assert!(matches!(last, Some(Offered::Delivered(_))));
    }

    #[test]
    fn a_chunk_outside_the_window_a_duplicate_and_a_foreign_type_are_typed_rejections() {
        let mut assembler = assembler();
        let parts = chunk_frame(&key(), &frame(4, 8, 0), meta(1), 40).unwrap();
        assert!(parts.len() >= 8);
        offer(&mut assembler, 0, &parts[0]).unwrap();
        assert_eq!(
            offer(&mut assembler, 0, &parts[0]),
            Err(Rejection::DuplicateChunk)
        );
        // The lowest missing index is 1, so the window is 1..5.
        assert_eq!(
            offer(&mut assembler, 0, &parts[5]),
            Err(Rejection::OutOfWindow {
                index: 5,
                lowest_missing: 1
            })
        );
        assert!(offer(&mut assembler, 0, &parts[4]).is_ok());
        let foreign = TypedPayload {
            type_id: ptr_types::TypeId::from("ptr.kv.page"),
            bytes: vec![0; 64],
        };
        assert_eq!(
            assembler.offer(0, PRINCIPAL, 1, &foreign),
            Err(Rejection::NotAllowlisted("ptr.kv.page".to_owned()))
        );
    }

    #[test]
    fn the_signature_the_digest_and_the_principal_are_each_checked() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        let mut forged = parts[0].clone();
        forged.signature[0] ^= 1;
        assert_eq!(
            offer(&mut assembler, 0, &forged),
            Err(Rejection::BadSignature)
        );
        let mut relabelled = parts[0].clone();
        relabelled.header.frame_id += 1;
        assert_eq!(
            offer(&mut assembler, 0, &relabelled),
            Err(Rejection::BadSignature)
        );
        // A signer that signs a digest that does not match the bytes.
        let mut wrong = parts[0].clone();
        wrong.bytes[0] ^= 1;
        resign_envelope(&key(), &mut wrong);
        assert_eq!(
            offer(&mut assembler, 0, &wrong),
            Err(Rejection::BadChunkDigest)
        );
        assert_eq!(
            assembler.offer(0, "stranger", 1, &chunk_payload(&parts[0])),
            Err(Rejection::UnknownPrincipal)
        );
        assert_eq!(assembler.incomplete_frames(PRINCIPAL), 0);
    }

    #[test]
    fn a_header_that_changes_between_chunks_rejects_and_abandons_the_frame() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        offer(&mut assembler, 0, &parts[0]).unwrap();
        let mut other_chain = parts[1].clone();
        other_chain.header.upstream_chain.stage.generation = 9;
        let other_chain = sign_chunk(&key(), other_chain.header, other_chain.bytes);
        assert_eq!(
            offer(&mut assembler, 0, &other_chain),
            Err(Rejection::ChainMismatch)
        );
        assert_eq!(assembler.incomplete_frames(PRINCIPAL), 0);
        assert_eq!(assembler.buffered_bytes(), 0);
        // The frame is dead: even a valid chunk of it is refused, typed.
        assert_eq!(
            offer(&mut assembler, 0, &parts[1]),
            Err(Rejection::FrameAbandoned)
        );

        let mut assembler = self::assembler();
        offer(&mut assembler, 0, &parts[0]).unwrap();
        let mut other_digest = parts[1].clone();
        other_digest.header.frame_digest[0] ^= 1;
        let other_digest = sign_chunk(&key(), other_digest.header, other_digest.bytes);
        assert_eq!(
            offer(&mut assembler, 0, &other_digest),
            Err(Rejection::FrameMismatch(MismatchField::FrameDigest))
        );
        let mut assembler = self::assembler();
        offer(&mut assembler, 0, &parts[0]).unwrap();
        let mut other_total = parts[1].clone();
        other_total.header.chunk_total += 1;
        let other_total = sign_chunk(&key(), other_total.header, other_total.bytes);
        assert_eq!(
            offer(&mut assembler, 0, &other_total),
            Err(Rejection::FrameMismatch(MismatchField::ChunkTotal))
        );
    }

    #[test]
    fn a_consistent_but_wrong_frame_digest_is_caught_by_hashing_the_whole_payload() {
        let mut assembler = assembler();
        let sent = frame(4, 8, 0);
        let mut parts = chunk_frame(&key(), &sent, meta(1), 80).unwrap();
        let wrong = [9u8; 32];
        for chunk in &mut parts {
            chunk.header.frame_digest = wrong;
            *chunk = sign_chunk(&key(), chunk.header.clone(), chunk.bytes.clone());
        }
        let mut outcome = None;
        for chunk in &parts {
            outcome = Some(offer(&mut assembler, 0, chunk));
        }
        assert_eq!(outcome, Some(Err(Rejection::FrameDigestMismatch)));
        assert_eq!(assembler.buffered_bytes(), 0);
        assert_eq!(assembler.expire(0).len(), parts.len());
    }

    #[test]
    fn a_complete_frame_with_a_wrong_shape_or_positions_length_is_refused_whole() {
        for case in 0..2 {
            let mut assembler = assembler();
            let mut bad = frame(4, 8, 0);
            if case == 0 {
                bad.tensor.truncate(bad.tensor.len() - 4);
            } else {
                bad.positions.pop();
            }
            let parts = chunk_frame(&key(), &bad, meta(1), 80).unwrap();
            let mut outcome = None;
            for chunk in &parts {
                outcome = Some(offer(&mut assembler, 0, chunk));
            }
            assert!(
                matches!(outcome, Some(Err(Rejection::InvalidFrame(_)))),
                "case {case}: {outcome:?}"
            );
        }
    }

    #[test]
    fn a_completed_frame_cannot_be_replayed_under_any_session() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        for chunk in &parts {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        for session in [1, 2] {
            assert_eq!(
                assembler.offer(10, PRINCIPAL, session, &chunk_payload(&parts[0])),
                Err(Rejection::FrameReplayed)
            );
        }
        // After the retention (deadline + skew) the deadline itself refuses it.
        assert_eq!(
            offer(&mut assembler, 5_001, &parts[0]),
            Err(Rejection::Expired)
        );
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 0);
    }

    #[test]
    fn a_clock_that_steps_back_cannot_revive_a_dropped_tombstone() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        for chunk in &parts {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        // The tombstone (retained to deadline + skew = 5000) is dropped at 6000 ...
        assert_eq!(
            offer(&mut assembler, 6_000, &parts[0]),
            Err(Rejection::Expired)
        );
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 0);
        // ... and an earlier time afterwards is treated as 6000: still expired, not accepted.
        assert_eq!(
            offer(&mut assembler, 100, &parts[0]),
            Err(Rejection::Expired)
        );
        assert_eq!(offer(&mut assembler, 0, &parts[1]), Err(Rejection::Expired));
        assert_eq!(assembler.buffered_bytes(), 0);
    }

    #[test]
    fn a_total_below_one_frame_is_not_a_policy() {
        let mut bad = policy();
        bad.max_total_buffered_bytes = bad.max_frame_bytes - 1;
        assert_eq!(bad.validate(), Err(PolicyViolation::TotalBelowOneFrame));
        assert!(FrameAssembler::new(bad).is_err());
    }

    #[test]
    fn registering_a_principal_again_keeps_its_tombstones_and_its_frames() {
        let mut assembler = assembler();
        let delivered = chunks(1, 0);
        for chunk in &delivered {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        let partial = chunks(2, 3);
        offer(&mut assembler, 0, &partial[0]).unwrap();
        let bound = sign_policy(&key(), policy_of(100, 8));
        assembler
            .offer(0, PRINCIPAL, 1, &policy_payload(&bound))
            .unwrap();
        // A reconnect (same key, same stage identity): the table keeps everything.
        assembler.register_principal(PRINCIPAL, key().verifying_key(), "stage-a");
        assert_eq!(
            offer(&mut assembler, 10, &delivered[0]),
            Err(Rejection::FrameReplayed)
        );
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 1);
        assert_eq!(assembler.buffered_bytes(), partial[0].bytes.len());
        assert!(assembler.expire(10).is_empty());
        for chunk in &partial[1..] {
            offer(&mut assembler, 10, chunk).unwrap();
        }
    }

    #[test]
    fn a_key_rotation_drops_the_policies_and_the_frames_minted_under_the_old_key() {
        let new_key = SigningKey::from_bytes(&[8; 32]);
        let mut assembler = assembler();
        let delivered = chunks(1, 0);
        for chunk in &delivered {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        let partial = chunks(2, 3);
        offer(&mut assembler, 0, &partial[0]).unwrap();
        offer(&mut assembler, 0, &partial[1]).unwrap();
        // A policy signed under the old key binds operation 100 (width 4).
        let old_policy = sign_policy(&key(), policy_of(100, 4));
        assembler
            .offer(0, PRINCIPAL, 1, &policy_payload(&old_policy))
            .unwrap();
        assembler.register_principal(PRINCIPAL, new_key.verifying_key(), "stage-a");
        // The incomplete frame of the old key is released (once), tombstoned, not completable.
        assert_eq!(assembler.buffered_bytes(), 0);
        assert_eq!(assembler.incomplete_frames(PRINCIPAL), 0);
        let mut released = assembler.expire(1);
        released.sort();
        assert_eq!(
            released
                .iter()
                .map(|id| (id.frame_id, id.chunk_index))
                .collect::<Vec<_>>(),
            vec![(2, 0), (2, 1)]
        );
        assert!(assembler.expire(2).is_empty());
        let rest: Vec<_> = chunk_frame(&new_key, &frame(4, 8, 3), meta(2), 80).unwrap();
        for chunk in &rest {
            assert_eq!(
                offer(&mut assembler, 10, chunk),
                Err(Rejection::FrameAbandoned)
            );
        }
        // The old-key chunks no longer verify either, and the completed frame stays a replay.
        assert_eq!(
            offer(&mut assembler, 10, &partial[2]),
            Err(Rejection::BadSignature)
        );
        let again = chunk_frame(&new_key, &frame(4, 8, 0), meta(1), 80).unwrap();
        assert_eq!(
            offer(&mut assembler, 10, &again[0]),
            Err(Rejection::FrameReplayed)
        );
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 2);
        // The policy signed under the old key is gone: the same operation, re-signed by the new
        // key with another width, binds instead of being a PolicyMismatch.
        let reissued = sign_policy(&new_key, policy_of(100, 8));
        assert_eq!(
            assembler.offer(10, PRINCIPAL, 1, &policy_payload(&reissued)),
            Ok(Offered::Policy(reissued))
        );
    }

    #[test]
    fn offer_and_abandon_neither_lose_nor_repeat_released_chunk_ids() {
        // Expiry inside an offer: the ids stay in the buffer for expire().
        let mut assembler = assembler();
        let first = chunks(1, 0);
        offer(&mut assembler, 0, &first[0]).unwrap();
        offer(&mut assembler, 0, &first[1]).unwrap();
        let other = chunks(9, 5);
        assert!(offer(&mut assembler, 6_000, &other[0]).is_err());
        let released = assembler.expire(6_001);
        assert_eq!(released.len(), 2, "{released:?}");
        assert!(assembler.expire(6_002).is_empty());
        // An abandonment inside an offer (a header that changes) also stays for expire().
        let mut assembler = self::assembler();
        let parts = chunks(3, 0);
        offer(&mut assembler, 0, &parts[0]).unwrap();
        offer(&mut assembler, 0, &parts[1]).unwrap();
        let mut forged = parts[2].clone();
        forged.header.frame_digest[0] ^= 1;
        let forged = sign_chunk(&key(), forged.header, forged.bytes);
        assert!(matches!(
            offer(&mut assembler, 1, &forged),
            Err(Rejection::FrameMismatch(_))
        ));
        assert_eq!(assembler.expire(2).len(), 2);
        assert!(assembler.expire(3).is_empty());
        // abandon() reports its chunks by its return value only; a later expire() does not
        // list them again, and an abandon after an offer-time release loses nothing.
        let mut assembler = self::assembler();
        let a = chunks(4, 0);
        let b = chunks(5, 1);
        for chunk in [&a[0], &a[1], &b[0]] {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        let released = assembler.abandon(1, PRINCIPAL, 100, 4).unwrap();
        assert_eq!(released.chunks.len(), 2);
        assert!(assembler.expire(2).is_empty());
        // The other frame expires in an abandon() call: its ids are not dropped.
        assert_eq!(
            assembler.abandon(6_000, PRINCIPAL, 100, 4),
            Err(AbandonError::NoSuchFrame)
        );
        let late = assembler.expire(6_001);
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].frame_id, 5);
    }

    #[test]
    fn a_new_stage_identity_releases_the_frames_bound_to_the_old_one() {
        let mut assembler = assembler();
        let partial = chunks(2, 3);
        offer(&mut assembler, 0, &partial[0]).unwrap();
        offer(&mut assembler, 0, &partial[1]).unwrap();
        assembler.register_principal(PRINCIPAL, key().verifying_key(), "stage-b");
        assert_eq!(assembler.buffered_bytes(), 0);
        let released = assembler.expire(1);
        assert_eq!(released.len(), 2);
        assert_eq!(
            offer(&mut assembler, 10, &partial[2]),
            Err(Rejection::StageNotBound)
        );
        // The tuple stays tombstoned through its signed deadline plus the skew.
        assembler.register_principal(PRINCIPAL, key().verifying_key(), "stage-a");
        assert_eq!(
            offer(&mut assembler, 10, &partial[2]),
            Err(Rejection::FrameAbandoned)
        );
    }

    #[test]
    fn two_sessions_of_one_principal_join_one_assembly_and_one_frame_is_delivered() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        let mut delivered = 0;
        for (index, chunk) in parts.iter().enumerate() {
            let session = 1 + (index as u64 % 2);
            match assembler.offer(0, PRINCIPAL, session, &chunk_payload(chunk)) {
                Ok(Offered::Delivered(frame)) => {
                    delivered += 1;
                    assert_eq!(frame.sessions, 2);
                }
                Ok(_) => {}
                Err(error) => panic!("{error:?}"),
            }
        }
        assert_eq!(delivered, 1);
        assert_eq!(
            assembler.offer(0, PRINCIPAL, 2, &chunk_payload(&parts[0])),
            Err(Rejection::FrameReplayed)
        );
    }

    #[test]
    fn an_expired_frame_is_refused_before_a_tuple_is_reserved() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        assert_eq!(
            offer(&mut assembler, 5_000, &parts[0]),
            Err(Rejection::Expired)
        );
        assert_eq!(assembler.reserved_slots(PRINCIPAL), 0);
        // One millisecond inside the skew bound is still live (on a receiver whose clock has not
        // passed it: the assembler never goes back in time).
        let mut earlier = self::assembler();
        assert!(offer(&mut earlier, 4_999, &parts[0]).is_ok());
    }

    #[test]
    fn a_deadline_beyond_the_lifetime_is_refused() {
        let mut assembler = assembler();
        let far = chunk_frame(
            &key(),
            &frame(4, 8, 0),
            FrameMeta {
                signed_absolute_deadline_ms: 6_001,
                ..meta(1)
            },
            80,
        )
        .unwrap();
        assert_eq!(
            offer(&mut assembler, 0, &far[0]),
            Err(Rejection::DeadlineTooFar)
        );
    }

    #[test]
    fn an_abandoned_frame_releases_its_chunks_and_late_chunks_are_refused() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        offer(&mut assembler, 0, &parts[0]).unwrap();
        offer(&mut assembler, 0, &parts[1]).unwrap();
        assert!(assembler.buffered_bytes() > 0);
        let released = assembler.abandon(100, PRINCIPAL, 100, 1).unwrap();
        assert_eq!(released.chunks.len(), 2);
        assert_eq!(assembler.buffered_bytes(), 0);
        assert_eq!(assembler.buffered_chunks(PRINCIPAL), 0);
        assert_eq!(
            offer(&mut assembler, 101, &parts[2]),
            Err(Rejection::FrameAbandoned)
        );
        assert_eq!(
            assembler.abandon(101, PRINCIPAL, 100, 1),
            Err(AbandonError::NoSuchFrame)
        );
        // Retained until the abandonment time plus the replay window; after that the signed
        // deadline refuses the tuple on its own.
        assert_eq!(
            offer(&mut assembler, 100 + 9_999, &parts[2]),
            Err(Rejection::FrameAbandoned)
        );
        assert_eq!(
            offer(&mut assembler, 100 + 10_000, &parts[2]),
            Err(Rejection::Expired)
        );
    }

    #[test]
    fn a_frame_past_its_deadline_is_abandoned_by_expiry_and_its_chunks_are_released() {
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        offer(&mut assembler, 0, &parts[0]).unwrap();
        offer(&mut assembler, 0, &parts[1]).unwrap();
        let released = assembler.expire(5_001);
        assert_eq!(released.len(), 2);
        assert_eq!(assembler.buffered_bytes(), 0);
        assert_eq!(assembler.incomplete_frames(PRINCIPAL), 0);
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 1);
        assert_eq!(
            offer(&mut assembler, 5_002, &parts[2]),
            Err(Rejection::FrameAbandoned)
        );
    }

    #[test]
    fn the_table_is_bounded_and_full_refuses_new_frames_until_a_tombstone_expires() {
        let mut assembler = assembler();
        for id in 0..64u64 {
            let parts = chunks(id, 0);
            offer(&mut assembler, 0, &parts[0]).unwrap();
            assembler.abandon(0, PRINCIPAL, 100, id).unwrap();
            assert!(assembler.reserved_slots(PRINCIPAL) <= 64);
        }
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 64);
        let extra = chunks(64, 0);
        assert_eq!(
            offer(&mut assembler, 0, &extra[0]),
            Err(Rejection::FrameTableFull)
        );
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 64);
        // At 10_000 the first tombstones expire and a new frame is accepted.
        assert!(offer(&mut assembler, 9_999, &extra[0]).is_err());
        let fresh = chunk_frame(
            &key(),
            &frame(4, 8, 0),
            FrameMeta {
                signed_absolute_deadline_ms: 14_000,
                ..meta(64)
            },
            80,
        )
        .unwrap();
        assert!(offer(&mut assembler, 10_000, &fresh[0]).is_ok());
        assert!(assembler.reserved_slots(PRINCIPAL) <= 64);
    }

    #[test]
    fn the_per_peer_and_in_flight_bounds_are_enforced() {
        let mut assembler = assembler();
        for id in 0..8u64 {
            let parts = chunks(id, 0);
            offer(&mut assembler, 0, &parts[0]).unwrap();
        }
        let ninth = chunks(8, 0);
        assert_eq!(
            offer(&mut assembler, 0, &ninth[0]),
            Err(Rejection::TooManyIncompleteFrames)
        );

        let mut assembler = self::assembler();
        let long = |id: u64| {
            let big = frame(64, 16, 0);
            chunk_frame(&key(), &big, meta(id), 128).unwrap()
        };
        for id in 0..8u64 {
            let parts = long(id);
            assert!(parts.len() > 8);
            for chunk in &parts[..4] {
                offer(&mut assembler, 0, chunk).unwrap();
            }
        }
        assert_eq!(assembler.buffered_chunks(PRINCIPAL), 32);
        let more = long(0);
        assert_eq!(
            offer(&mut assembler, 0, &more[4]),
            Err(Rejection::QueueOverflow)
        );
    }

    #[test]
    fn the_chunks_of_two_frames_with_one_chain_never_mix() {
        let mut assembler = assembler();
        let first = chunks(1, 0);
        let second = chunks(2, 77);
        let mut done = Vec::new();
        for index in 0..first.len().max(second.len()) {
            for parts in [&first, &second] {
                if let Some(chunk) = parts.get(index) {
                    if let Ok(Offered::Delivered(frame)) = offer(&mut assembler, 0, chunk) {
                        done.push(frame);
                    }
                }
            }
        }
        assert_eq!(done.len(), 2);
        let by_id = |id: u64| done.iter().find(|frame| frame.frame_id == id).unwrap();
        assert_eq!(by_id(1).frame, frame(4, 8, 0));
        assert_eq!(by_id(2).frame, frame(4, 8, 77));
    }

    #[test]
    fn a_policy_payload_is_taken_without_state_and_the_replay_window_relation_is_checked() {
        use ptr_protocol::activation::{encode_policy, SPLIT_POLICY_TYPE};
        let mut assembler = assembler();
        let policy_value = sign_policy(
            &key(),
            SplitPolicy {
                operation_id: 1,
                split_layer: 2,
                width: 64,
                dtype: BoundaryDtype::F32,
                signed_absolute_deadline_ms: 4_000,
                signature: [0; 64],
            },
        );
        let payload = TypedPayload {
            type_id: ptr_types::TypeId::from(SPLIT_POLICY_TYPE),
            bytes: encode_policy(&policy_value),
        };
        assert_eq!(
            assembler.offer(0, PRINCIPAL, 1, &payload),
            Ok(Offered::Policy(policy_value))
        );
        assert_eq!(assembler.reserved_slots(PRINCIPAL), 0);
        let short = AssemblyPolicy {
            replay_window_ms: 6_999,
            ..policy()
        };
        assert_eq!(short.validate(), Err(PolicyViolation::ReplayWindowTooShort));
        let zero = AssemblyPolicy {
            reorder_window: 0,
            ..policy()
        };
        assert_eq!(
            zero.validate(),
            Err(PolicyViolation::Zero("reorder_window"))
        );
    }

    #[test]
    fn chunking_refuses_a_zero_size_and_a_frame_needing_too_many_chunks() {
        assert_eq!(
            chunk_frame(&key(), &frame(1, 1, 0), meta(1), 0),
            Err(ChunkingError::ZeroChunkSize)
        );
        assert!(matches!(
            chunk_frame(&key(), &frame(64, 64, 0), meta(1), 16),
            Err(ChunkingError::TooManyChunks { .. })
        ));
    }

    fn policy_payload(policy: &SplitPolicy) -> TypedPayload {
        use ptr_protocol::activation::{encode_policy, SPLIT_POLICY_TYPE};
        TypedPayload {
            type_id: ptr_types::TypeId::from(SPLIT_POLICY_TYPE),
            bytes: encode_policy(policy),
        }
    }

    fn policy_of(operation_id: u64, width: u32) -> SplitPolicy {
        SplitPolicy {
            operation_id,
            split_layer: 2,
            width,
            dtype: BoundaryDtype::F32,
            signed_absolute_deadline_ms: 4_000,
            signature: [0; 64],
        }
    }

    #[test]
    fn a_tombstone_is_retained_through_the_tick_at_which_the_frame_expires() {
        // Completion: retained until deadline + skew = 5000; at 5000 the tombstone is gone and
        // the frame is Expired, so no tick lets the tuple through.
        let mut assembler = assembler();
        let parts = chunks(1, 0);
        for chunk in &parts {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        assert_eq!(
            offer(&mut assembler, 4_999, &parts[0]),
            Err(Rejection::FrameReplayed)
        );
        assert_eq!(
            offer(&mut assembler, 5_000, &parts[0]),
            Err(Rejection::Expired)
        );
        // Abandonment at 100: retained until 100 + window = 10100.
        let mut assembler = assembler_with_one_chunk_of(2);
        assembler.abandon(100, PRINCIPAL, 100, 2).unwrap();
        let first = &chunks(2, 0)[0];
        assert_eq!(
            offer(&mut assembler, 10_099, first),
            Err(Rejection::FrameAbandoned)
        );
        assert_eq!(
            offer(&mut assembler, 10_100, first),
            Err(Rejection::Expired)
        );
    }

    fn assembler_with_one_chunk_of(frame_id: u64) -> FrameAssembler {
        let mut assembler = assembler();
        offer(&mut assembler, 0, &chunks(frame_id, 0)[0]).unwrap();
        assembler
    }

    #[test]
    fn a_policy_must_be_signed_by_the_principal_and_binds_its_operation() {
        let mut assembler = assembler();
        // Unsigned (zero signature) and signed by another key: refused, never delivered.
        let unsigned = policy_of(100, 8);
        assert_eq!(
            assembler.offer(0, PRINCIPAL, 1, &policy_payload(&unsigned)),
            Err(Rejection::BadSignature)
        );
        let other = SigningKey::from_bytes(&[9; 32]);
        let foreign = sign_policy(&other, policy_of(100, 8));
        assert_eq!(
            assembler.offer(0, PRINCIPAL, 1, &policy_payload(&foreign)),
            Err(Rejection::BadSignature)
        );
        // An expired policy and one with a far deadline are refused too.
        let mut old = policy_of(100, 8);
        old.signed_absolute_deadline_ms = 10;
        assert_eq!(
            assembler.offer(
                5_000,
                PRINCIPAL,
                1,
                &policy_payload(&sign_policy(&key(), old))
            ),
            Err(Rejection::Expired)
        );
        // A signed policy binds the operation: a frame of another width is refused whole (on a
        // receiver whose clock is still at 0: the assembler never goes back in time).
        let mut assembler = self::assembler();
        let good = sign_policy(&key(), policy_of(100, 4));
        assert_eq!(
            assembler.offer(0, PRINCIPAL, 1, &policy_payload(&good)),
            Ok(Offered::Policy(good))
        );
        let mut outcome = Ok(Offered::Buffered {
            received: 0,
            total: 0,
        });
        for chunk in chunks(1, 0) {
            outcome = offer(&mut assembler, 0, &chunk);
        }
        assert_eq!(outcome, Err(Rejection::PolicyMismatch));
        // The same operation with a matching width is delivered.
        let matching = sign_policy(&key(), policy_of(100, 8));
        let mut again = FrameAssembler::new(policy()).unwrap();
        again.register_principal(PRINCIPAL, key().verifying_key(), "stage-a");
        again
            .offer(0, PRINCIPAL, 1, &policy_payload(&matching))
            .unwrap();
        let mut delivered = false;
        for chunk in chunks(1, 0) {
            delivered = matches!(offer(&mut again, 0, &chunk), Ok(Offered::Delivered(_)));
        }
        assert!(delivered);
    }

    #[test]
    fn the_sender_is_authenticated_before_its_content_is_decoded() {
        use ptr_protocol::activation::ACTIVATION_CHUNK_TYPE;
        let mut assembler = assembler();
        let garbage = TypedPayload {
            type_id: ptr_types::TypeId::from(ACTIVATION_CHUNK_TYPE),
            bytes: vec![0xAB; 64],
        };
        // An unregistered principal gets UnknownPrincipal, not a verdict on its bytes.
        assert_eq!(
            assembler.offer(0, "stranger", 1, &garbage),
            Err(Rejection::UnknownPrincipal)
        );
        // A registered one is judged on its bytes.
        assert!(matches!(
            assembler.offer(0, PRINCIPAL, 1, &garbage),
            Err(Rejection::Malformed(_))
        ));
    }

    #[test]
    fn a_principal_may_only_speak_for_its_own_stage() {
        let mut assembler = assembler();
        let mut impostor = frame(4, 8, 0);
        impostor.upstream_chain.stage = element("stage-b", 1);
        let parts = chunk_frame(&key(), &impostor, meta(1), 80).unwrap();
        assert_eq!(
            offer(&mut assembler, 0, &parts[0]),
            Err(Rejection::StageNotBound)
        );
        assert_eq!(assembler.reserved_slots(PRINCIPAL), 0);
    }

    #[test]
    fn the_signed_chain_must_be_the_chain_inside_the_frame() {
        let mut assembler = assembler();
        // Header chain X (generation 1), frame bytes carrying chain Y (model generation 2).
        let mut inner = frame(4, 8, 0);
        inner.upstream_chain.base_model = element("model", 2);
        let bytes = inner.to_bytes();
        let digest = frame_digest(&bytes);
        let total = bytes.chunks(80).count() as u32;
        let mut last = None;
        for (index, part) in bytes.chunks(80).enumerate() {
            let header = ChunkHeader {
                frame_id: 1,
                parent_operation_id: 100,
                chunk_index: index as u32,
                chunk_total: total,
                signed_absolute_deadline_ms: 4_000,
                frame_digest: digest,
                chunk_digest: [0; 32],
                upstream_chain: chain(),
            };
            let chunk = sign_chunk(&key(), header, part.to_vec());
            last = Some(offer(&mut assembler, 0, &chunk));
        }
        assert_eq!(last, Some(Err(Rejection::ChainFrameMismatch)));
        assert_eq!(assembler.buffered_bytes(), 0);
        assert_eq!(
            offer(&mut assembler, 1, &chunks(1, 0)[0]),
            Err(Rejection::FrameAbandoned)
        );
    }

    #[test]
    fn buffered_bytes_are_bounded_over_all_principals_together() {
        let limits = AssemblyPolicy {
            max_total_buffered_bytes: 150,
            max_frame_bytes: 150,
            ..policy()
        };
        let mut assembler = FrameAssembler::new(limits).unwrap();
        assembler.register_principal(PRINCIPAL, key().verifying_key(), "stage-a");
        let other = SigningKey::from_bytes(&[8; 32]);
        assembler.register_principal("other", other.verifying_key(), "stage-b");
        let mine = chunks(1, 0);
        offer(&mut assembler, 0, &mine[0]).unwrap();
        let mut theirs = frame(4, 8, 0);
        theirs.upstream_chain.stage = element("stage-b", 1);
        let theirs = chunk_frame(&other, &theirs, meta(1), 80).unwrap();
        // 80 bytes are buffered; another 80 would pass 150, whichever principal sends them.
        assert_eq!(
            assembler.offer(0, "other", 1, &chunk_payload(&theirs[0])),
            Err(Rejection::GlobalBufferFull)
        );
        assert_eq!(
            offer(&mut assembler, 0, &mine[1]),
            Err(Rejection::GlobalBufferFull)
        );
        assert_eq!(assembler.buffered_bytes(), 80);
        // Releasing the buffered bytes frees the room.
        assembler.abandon(1, PRINCIPAL, 100, 1).unwrap();
        assert!(assembler
            .offer(2, "other", 1, &chunk_payload(&theirs[0]))
            .is_ok());
    }

    fn offer_as(
        assembler: &mut FrameAssembler,
        principal: &str,
        now: u64,
        chunk: &ActivationChunk,
    ) -> Result<Offered, Rejection> {
        assembler.offer(now, principal, 1, &chunk_payload(chunk))
    }

    #[test]
    fn a_new_principal_id_with_the_same_key_and_stage_inherits_the_tombstones() {
        let mut assembler = assembler();
        let delivered = chunks(1, 0);
        for chunk in &delivered {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        let abandoned = chunks(2, 1);
        offer(&mut assembler, 0, &abandoned[0]).unwrap();
        assembler.abandon(1, PRINCIPAL, 100, 2).unwrap();
        // The same runtime behind a new endpoint identity (a restart).
        assembler.register_principal("reborn", key().verifying_key(), "stage-a");
        assert_eq!(assembler.retained_tombstones("reborn"), 2);
        assert_eq!(
            offer_as(&mut assembler, "reborn", 10, &delivered[0]),
            Err(Rejection::FrameReplayed)
        );
        assert_eq!(
            offer_as(&mut assembler, "reborn", 10, &abandoned[1]),
            Err(Rejection::FrameAbandoned)
        );
        // Not carried: incomplete frames and policies, and a frame the old table never saw is
        // new to the new one.
        assert_eq!(assembler.incomplete_frames("reborn"), 0);
        let fresh = chunks(3, 2);
        assert!(offer_as(&mut assembler, "reborn", 10, &fresh[0]).is_ok());
    }

    #[test]
    fn a_frame_completed_under_the_old_identity_after_the_new_one_registered_is_a_replay() {
        let mut assembler = assembler();
        let late = chunks(1, 0);
        offer(&mut assembler, 0, &late[0]).unwrap();
        // The new endpoint identity registers while the old connection is still delivering.
        assembler.register_principal("reborn", key().verifying_key(), "stage-a");
        assert_eq!(assembler.retained_tombstones("reborn"), 0);
        for chunk in &late[1..] {
            offer(&mut assembler, 1, chunk).unwrap();
        }
        assert_eq!(assembler.retained_tombstones(PRINCIPAL), 1);
        // Nothing was copied, yet the replay under the new identity is refused.
        for chunk in &late {
            assert_eq!(
                offer_as(&mut assembler, "reborn", 2, chunk),
                Err(Rejection::FrameReplayed)
            );
        }
        // And the other way round: a frame abandoned under the new identity is refused under
        // the old one.
        let other = chunks(2, 1);
        offer_as(&mut assembler, "reborn", 3, &other[0]).unwrap();
        assembler.abandon(4, "reborn", 100, 2).unwrap();
        assert_eq!(
            offer(&mut assembler, 5, &other[1]),
            Err(Rejection::FrameAbandoned)
        );
    }

    #[test]
    fn a_sibling_tombstone_releases_the_local_partial_assembly_at_once() {
        let mut assembler = assembler();
        let frame = chunks(1, 0);
        // The old identity holds a partial assembly of the frame.
        offer(&mut assembler, 0, &frame[0]).unwrap();
        assert_eq!(assembler.incomplete_frames(PRINCIPAL), 1);
        assert!(assembler.buffered_bytes() > 0);
        // The same runtime under a new identity completes it, tombstoning the tuple there.
        assembler.register_principal("reborn", key().verifying_key(), "stage-a");
        for chunk in &frame {
            offer_as(&mut assembler, "reborn", 1, chunk).unwrap();
        }
        // The next chunk under the old identity is a replay, and its capacity is freed now.
        assert_eq!(
            offer(&mut assembler, 2, &frame[1]),
            Err(Rejection::FrameReplayed)
        );
        assert_eq!(assembler.incomplete_frames(PRINCIPAL), 0);
        assert_eq!(assembler.buffered_bytes(), 0);
        // The released chunk is reported once.
        assert_eq!(assembler.expire(3).len(), 1);
        assert!(assembler.expire(4).is_empty());
    }

    #[test]
    fn a_new_principal_with_another_key_or_another_stage_inherits_nothing() {
        let mut assembler = assembler();
        let delivered = chunks(1, 0);
        for chunk in &delivered {
            offer(&mut assembler, 0, chunk).unwrap();
        }
        let other = SigningKey::from_bytes(&[8; 32]);
        assembler.register_principal("other-key", other.verifying_key(), "stage-a");
        assembler.register_principal("other-stage", key().verifying_key(), "stage-b");
        assert_eq!(assembler.retained_tombstones("other-key"), 0);
        assert_eq!(assembler.retained_tombstones("other-stage"), 0);
        // A captured chunk is judged on its own signature and stage, not as a replay.
        assert_eq!(
            offer_as(&mut assembler, "other-key", 10, &delivered[0]),
            Err(Rejection::BadSignature)
        );
        assert_eq!(
            offer_as(&mut assembler, "other-stage", 10, &delivered[0]),
            Err(Rejection::StageNotBound)
        );
        // The original principal is unaffected.
        assert_eq!(
            offer(&mut assembler, 10, &delivered[0]),
            Err(Rejection::FrameReplayed)
        );
    }

    #[test]
    fn inherited_tombstones_expire_with_their_retention_and_stay_within_the_bound() {
        let limits = AssemblyPolicy {
            max_frame_tombstones_per_principal: 3,
            ..policy()
        };
        let mut assembler = FrameAssembler::new(limits).unwrap();
        assembler.register_principal(PRINCIPAL, key().verifying_key(), "stage-a");
        // Four deliveries into a table of three slots: the fourth is refused with no slot.
        for id in 1..=3u64 {
            for chunk in &chunks(id, 0) {
                offer(&mut assembler, 0, chunk).unwrap();
            }
        }
        assert_eq!(
            offer(&mut assembler, 0, &chunks(4, 0)[0]),
            Err(Rejection::FrameTableFull)
        );
        // A second table of the same pair already holds some of them; the merge never
        // exceeds the bound.
        assembler.register_principal("second", key().verifying_key(), "stage-a");
        assembler.register_principal("third", key().verifying_key(), "stage-a");
        assert_eq!(assembler.retained_tombstones("second"), 3);
        assert_eq!(assembler.retained_tombstones("third"), 3);
        assert!(assembler.reserved_slots("third") <= 3);
        // Past the retention (deadline 4000 plus skew 1000) nothing is inherited.
        assert!(assembler.expire(5_001).is_empty());
        assembler.register_principal("late", key().verifying_key(), "stage-a");
        assert_eq!(assembler.retained_tombstones("late"), 0);
    }
}
