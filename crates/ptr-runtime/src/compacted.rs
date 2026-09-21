//! Compacted materialized snapshots: committed state at a floor, without the
//! history that produced it.
//!
//! [`RecoverySnapshot`](crate::persistence::RecoverySnapshot) retains the complete journal by
//! construction, which makes it exact but means it can never justify discarding a
//! record. Compaction needs the opposite artifact: the state a prefix produced,
//! small enough that the prefix itself becomes redundant. That is what
//! establishes `CompactionBarrier::snapshot_covers`, and without it a host has no
//! sound basis for raising a log's floor at all.
//!
//! The snapshot deliberately does **not** embed the records above its floor. Those
//! live in the log, which is already framed, chained and anchored. Carrying a
//! second copy would create two descriptions of the same records that could
//! disagree, and the whole point of this layer is to keep exactly one authority
//! for any given fact.
//!
//! Like every other artifact here it restores committed state only: no
//! permissions, sessions, permits, registrations, external-effect outcomes or
//! neural/KV state. Those are not history and cannot be replayed.
use super::execution::{SettledOutcome, UnsettledEffect};
use super::{PtrRuntime, RuntimeError, RuntimeLedger};
use ptr_config::PtrConfig;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::{CommittedEvent, InMemoryLedger};
use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Effect, Generation, Revision};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 8] = b"PTRCS002";
const EXECUTION_MAGIC: &[u8; 8] = b"PTREX001";
const LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC001";
const HEADER: usize = 80;
const DIGEST_BYTES: usize = 32;
const MAX_SECTION_ITEMS: usize = 65_536;
const MAX_STRING_BYTES: usize = 4096;
/// Bound on one encoded section, checked before any allocation driven by a count.
pub const MAX_SECTION_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_COMPACTED_BYTES: usize = HEADER + 3 * MAX_SECTION_BYTES + DIGEST_BYTES;

/// Named compacted-snapshot failure categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactedError {
    SizeLimit,
    UnsupportedVersion,
    LengthMismatch,
    AnchorMismatch,
    StateMismatch,
    /// A record does not continue directly above the snapshot's floor.
    JournalNotAboveFloor,
    /// A section is not the single canonical encoding of its contents.
    NoncanonicalSection,
    /// A string or item count exceeds this format's bounds.
    SectionLimit,
    /// An effect or outcome tag this build does not define. Refused rather than
    /// guessed: an obligation whose kind is unknown cannot be honoured.
    UnknownTag,
}

impl CompactedError {
    /// Stable diagnostic code for this snapshot refusal.
    pub fn code(self) -> &'static str {
        match self {
            Self::SizeLimit => "PTR_COMPACTED_SIZE_LIMIT",
            Self::UnsupportedVersion => "PTR_COMPACTED_VERSION",
            Self::LengthMismatch => "PTR_COMPACTED_LENGTH",
            Self::AnchorMismatch => "PTR_COMPACTED_ANCHOR_MISMATCH",
            Self::StateMismatch => "PTR_COMPACTED_STATE_MISMATCH",
            Self::JournalNotAboveFloor => "PTR_COMPACTED_JOURNAL_NOT_ABOVE_FLOOR",
            Self::NoncanonicalSection => "PTR_COMPACTED_NONCANONICAL_SECTION",
            Self::SectionLimit => "PTR_COMPACTED_SECTION_LIMIT",
            Self::UnknownTag => "PTR_COMPACTED_UNKNOWN_TAG",
        }
    }
}

impl std::fmt::Display for CompactedError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for CompactedError {}

/// Lift a compacted-snapshot fault into the runtime error boundary.
fn invalid(kind: CompactedError) -> RuntimeError {
    RuntimeError::Compacted(kind)
}

/// Trusted identity of a compacted snapshot, retained outside the artifact.
///
/// It is ordinary data, not authentication. Keep it where the snapshot cannot
/// reach, exactly as with
/// [`ProtectedAnchor`](ptr_ledger::anchor::ProtectedAnchor); a digest read back
/// out of the file it describes proves nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactedAnchor {
    pub revision: Revision,
    /// The commit position this snapshot's state covers.
    pub floor: LogAnchor,
    pub digest: [u8; 32],
}

/// Committed lifecycle and materialized state at one commit position.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct LifecycleState {
    materialized: BTreeMap<String, String>,
    live_generations: BTreeMap<String, Generation>,
    capsule_projects: BTreeMap<String, String>,
    revoked_generations: BTreeSet<(String, Generation)>,
}

/// An immutable compacted snapshot.
#[derive(Debug)]
pub struct CompactedSnapshot {
    bytes: Vec<u8>,
    anchor: CompactedAnchor,
}

impl CompactedSnapshot {
    /// Canonical sealed bytes of this snapshot.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Trusted identity the host must retain outside the snapshot.
    pub fn anchor(&self) -> CompactedAnchor {
        self.anchor
    }

    /// The commit position a [`CompactionBarrier`](ptr_ledger::CompactionBarrier)
    /// may report as covered once this snapshot is durably retained.
    ///
    /// Reporting a position this snapshot does not cover would let a cutover
    /// discard records whose state nothing describes, so the value is taken from
    /// the snapshot rather than chosen by the caller.
    pub fn covers(&self) -> CommitIndex {
        self.anchor.floor.index
    }
}

struct Writer(Vec<u8>);

impl Writer {
    /// Append bytes while enforcing the section-size bound.
    fn raw(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        if bytes.len() > MAX_SECTION_BYTES.saturating_sub(self.0.len()) {
            return Err(invalid(CompactedError::SectionLimit));
        }
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    /// Encode a bounded collection count.
    fn count(&mut self, value: usize) -> Result<(), RuntimeError> {
        if value > MAX_SECTION_ITEMS {
            return Err(invalid(CompactedError::SectionLimit));
        }
        self.raw(&(value as u32).to_le_bytes())
    }
    /// Encode one nonempty, bounded UTF-8 string.
    fn text(&mut self, value: &str) -> Result<(), RuntimeError> {
        if value.is_empty() || value.len() > MAX_STRING_BYTES {
            return Err(invalid(CompactedError::SectionLimit));
        }
        self.raw(&(value.len() as u32).to_le_bytes())?;
        self.raw(value.as_bytes())
    }
    /// Encode one little-endian unsigned integer.
    fn number(&mut self, value: u64) -> Result<(), RuntimeError> {
        self.raw(&value.to_le_bytes())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    /// Consume exactly `length` bytes from the section.
    fn take(&mut self, length: usize) -> Result<&'a [u8], RuntimeError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(invalid(CompactedError::SectionLimit))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(invalid(CompactedError::LengthMismatch))?;
        self.offset = end;
        Ok(slice)
    }
    /// Counts bound allocation, so they are checked against the bytes that remain
    /// before anything is reserved for them.
    fn count(&mut self) -> Result<usize, RuntimeError> {
        let value = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed count")) as usize;
        if value > MAX_SECTION_ITEMS || value > self.bytes.len().saturating_sub(self.offset) {
            return Err(invalid(CompactedError::SectionLimit));
        }
        Ok(value)
    }
    /// Decode one nonempty, bounded UTF-8 string.
    fn text(&mut self) -> Result<String, RuntimeError> {
        let length = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed length")) as usize;
        if length == 0 || length > MAX_STRING_BYTES {
            return Err(invalid(CompactedError::SectionLimit));
        }
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| invalid(CompactedError::NoncanonicalSection))
    }
    /// Decode one little-endian unsigned integer.
    fn number(&mut self) -> Result<u64, RuntimeError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed number"),
        ))
    }
    /// Whether the section has no trailing bytes.
    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

impl LifecycleState {
    /// Encode lifecycle maps and sets in deterministic key order.
    fn encode(&self) -> Result<Vec<u8>, RuntimeError> {
        let mut out = Writer(Vec::new());
        out.raw(LIFECYCLE_MAGIC)?;
        out.count(self.materialized.len())?;
        for (key, value) in &self.materialized {
            out.text(key)?;
            out.text(value)?;
        }
        out.count(self.live_generations.len())?;
        for (key, generation) in &self.live_generations {
            out.text(key)?;
            out.number(generation.0)?;
        }
        out.count(self.capsule_projects.len())?;
        for (key, value) in &self.capsule_projects {
            out.text(key)?;
            out.text(value)?;
        }
        out.count(self.revoked_generations.len())?;
        for (subject, generation) in &self.revoked_generations {
            out.text(subject)?;
            out.number(generation.0)?;
        }
        Ok(out.0)
    }

    /// Decoding enforces strict ascending key order per section, so each state has
    /// exactly one encoding and a reordered or duplicated entry fails closed
    /// rather than resolving to whichever entry happens to come last.
    fn decode(bytes: &[u8]) -> Result<Self, RuntimeError> {
        if bytes.len() > MAX_SECTION_BYTES {
            return Err(invalid(CompactedError::SizeLimit));
        }
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(8)? != LIFECYCLE_MAGIC {
            return Err(invalid(CompactedError::UnsupportedVersion));
        }
        let mut state = Self::default();
        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let key = reader.text()?;
            let value = reader.text()?;
            check_ascending(&mut previous, &key)?;
            state.materialized.insert(key, value);
        }
        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let key = reader.text()?;
            let generation = Generation(reader.number()?);
            check_ascending(&mut previous, &key)?;
            state.live_generations.insert(key, generation);
        }
        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let key = reader.text()?;
            let value = reader.text()?;
            check_ascending(&mut previous, &key)?;
            state.capsule_projects.insert(key, value);
        }
        let mut previous: Option<(String, Generation)> = None;
        for _ in 0..reader.count()? {
            let subject = reader.text()?;
            let generation = Generation(reader.number()?);
            let entry = (subject, generation);
            if previous.as_ref().is_some_and(|last| *last >= entry) {
                return Err(invalid(CompactedError::NoncanonicalSection));
            }
            previous = Some(entry.clone());
            state.revoked_generations.insert(entry);
        }
        if !reader.finished() {
            return Err(invalid(CompactedError::LengthMismatch));
        }
        Ok(state)
    }
}

/// Require the next decoded key to be strictly greater than its predecessor.
fn check_ascending(previous: &mut Option<String>, key: &str) -> Result<(), RuntimeError> {
    if previous.as_deref().is_some_and(|last| last >= key) {
        return Err(invalid(CompactedError::NoncanonicalSection));
    }
    *previous = Some(key.to_owned());
    Ok(())
}

/// The execution obligations a snapshot carries across its floor.
///
/// Compaction removes the records these are derived from. Before this section
/// existed a restored runtime rebuilt them from a history that was no longer
/// there, so it came back with an empty fence and no memory of which
/// at-most-once keys had been spent — and a retry under a spent key executed the
/// effect a second time. That is the whole reason the section exists.
#[derive(Debug, Default, Eq, PartialEq)]
struct ExecutionObligations {
    unsettled: BTreeMap<CommitIndex, UnsettledEffect>,
    settled: BTreeMap<String, SettledOutcome>,
}

/// Effect tags, matching the ledger's own numbering so the two cannot drift
/// apart silently. The match is exhaustive, so a new variant breaks this build
/// rather than being encoded as something else.
fn effect_tag(effect: Effect) -> u64 {
    match effect {
        Effect::Pure => 0,
        Effect::Read => 1,
        Effect::Mutation => 2,
        Effect::External => 3,
        Effect::Irreversible => 4,
    }
}

fn effect_from_tag(tag: u64) -> Result<Effect, RuntimeError> {
    match tag {
        0 => Ok(Effect::Pure),
        1 => Ok(Effect::Read),
        2 => Ok(Effect::Mutation),
        3 => Ok(Effect::External),
        4 => Ok(Effect::Irreversible),
        _ => Err(invalid(CompactedError::UnknownTag)),
    }
}

impl ExecutionObligations {
    /// Encode both maps in deterministic key order, unsettled attempts first.
    fn encode(&self) -> Result<Vec<u8>, RuntimeError> {
        let mut out = Writer(Vec::new());
        out.raw(EXECUTION_MAGIC)?;
        out.count(self.unsettled.len())?;
        for (attempt, effect) in &self.unsettled {
            out.number(attempt.0)?;
            // The attempt index is the map key, so storing it twice would allow a
            // record that disagrees with itself. Decoding rebuilds it from the key.
            out.number(effect.attempt.0)?;
            match &effect.key {
                Some(key) => {
                    out.number(1)?;
                    out.text(key)?;
                }
                None => out.number(0)?,
            }
            out.text(&effect.target)?;
            out.text(&effect.operation)?;
            out.number(effect_tag(effect.effect))?;
        }
        out.count(self.settled.len())?;
        for (key, outcome) in &self.settled {
            out.text(key)?;
            match outcome {
                SettledOutcome::Applied { response } => {
                    out.number(0)?;
                    out.count(response.len())?;
                    out.raw(response)?;
                }
                SettledOutcome::AppliedWithoutResponse => out.number(1)?,
                SettledOutcome::NotApplied => out.number(2)?,
            }
        }
        Ok(out.0)
    }

    /// Strict ascending key order per map, so each state has exactly one
    /// encoding and a reordered or duplicated entry fails closed.
    fn decode(bytes: &[u8]) -> Result<Self, RuntimeError> {
        if bytes.len() > MAX_SECTION_BYTES {
            return Err(invalid(CompactedError::SizeLimit));
        }
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(8)? != EXECUTION_MAGIC {
            return Err(invalid(CompactedError::UnsupportedVersion));
        }
        let mut state = Self::default();

        let mut previous: Option<u64> = None;
        for _ in 0..reader.count()? {
            let attempt = reader.number()?;
            if previous.is_some_and(|last| attempt <= last) {
                return Err(invalid(CompactedError::NoncanonicalSection));
            }
            previous = Some(attempt);
            let recorded = reader.number()?;
            if recorded != attempt {
                return Err(invalid(CompactedError::NoncanonicalSection));
            }
            let key = match reader.number()? {
                0 => None,
                1 => Some(reader.text()?),
                _ => return Err(invalid(CompactedError::UnknownTag)),
            };
            let target = reader.text()?;
            let operation = reader.text()?;
            let effect = effect_from_tag(reader.number()?)?;
            state.unsettled.insert(
                CommitIndex(attempt),
                UnsettledEffect {
                    attempt: CommitIndex(attempt),
                    key,
                    target,
                    operation,
                    effect,
                },
            );
        }

        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let key = reader.text()?;
            if previous.as_ref().is_some_and(|last| &key <= last) {
                return Err(invalid(CompactedError::NoncanonicalSection));
            }
            previous = Some(key.clone());
            let outcome = match reader.number()? {
                0 => {
                    let length = reader.count()?;
                    SettledOutcome::Applied {
                        response: reader.take(length)?.to_vec(),
                    }
                }
                1 => SettledOutcome::AppliedWithoutResponse,
                2 => SettledOutcome::NotApplied,
                _ => return Err(invalid(CompactedError::UnknownTag)),
            };
            state.settled.insert(key, outcome);
        }

        if !reader.finished() {
            return Err(invalid(CompactedError::LengthMismatch));
        }
        Ok(state)
    }
}

impl PtrRuntime {
    /// Capture the lifecycle-owned portion of committed runtime state.
    fn lifecycle_state(&self) -> LifecycleState {
        LifecycleState {
            materialized: self.state.values.clone(),
            live_generations: self.live_generations.clone(),
            capsule_projects: self.capsule_projects.clone(),
            revoked_generations: self.revoked_generations.clone(),
        }
    }

    /// Export committed state at the current commit position.
    ///
    /// The floor is the runtime's own journal anchor rather than a parameter: a
    /// snapshot can only describe state this runtime actually holds, and a floor
    /// chosen by the caller would invite describing a position it never reached.
    /// Compact the log to this floor afterwards, not before — the snapshot must be
    /// durable before the records it replaces are discarded.
    pub fn export_compacted_snapshot(&self) -> Result<CompactedSnapshot, RuntimeError> {
        let floor = self.journal_anchor()?;
        if self.state.last_applied != floor.index.0 {
            return Err(invalid(CompactedError::StateMismatch));
        }
        let semantic = self
            .semdb
            .export_state()
            .encode()
            .map_err(RuntimeError::Semantic)?;
        let lifecycle = self.lifecycle_state().encode()?;
        // Compaction is what removes the records these are derived from, so they
        // travel with the snapshot or they are lost.
        let (unsettled, settled) = self.execution.retained_obligations();
        let execution = ExecutionObligations {
            unsettled: unsettled.clone(),
            settled: settled.clone(),
        }
        .encode()?;
        if semantic.len() > MAX_SECTION_BYTES
            || lifecycle.len() > MAX_SECTION_BYTES
            || execution.len() > MAX_SECTION_BYTES
        {
            return Err(invalid(CompactedError::SizeLimit));
        }
        let mut bytes = Vec::with_capacity(
            HEADER + semantic.len() + lifecycle.len() + execution.len() + DIGEST_BYTES,
        );
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.revision().0.to_le_bytes());
        bytes.extend_from_slice(&floor.index.0.to_le_bytes());
        bytes.extend_from_slice(&floor.digest);
        bytes.extend_from_slice(&(semantic.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(lifecycle.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(execution.len() as u64).to_le_bytes());
        debug_assert_eq!(bytes.len(), HEADER);
        bytes.extend_from_slice(&semantic);
        bytes.extend_from_slice(&lifecycle);
        bytes.extend_from_slice(&execution);
        let digest = integrity::sha256(&bytes);
        bytes.extend_from_slice(&digest);
        Ok(CompactedSnapshot {
            bytes,
            anchor: CompactedAnchor {
                revision: self.revision(),
                floor,
                digest,
            },
        })
    }

    /// Restore committed state from a compacted snapshot, then replay the retained
    /// journal above its floor.
    ///
    /// `above_floor` must be the records a checked log holds above `trusted.floor`
    /// — in practice
    /// [`AcknowledgedLedger::events`](ptr_ledger::AcknowledgedLedger::events) for a
    /// log whose anchor carries that same floor. Each one goes through the ordinary
    /// lifecycle and semantic validation path, so a restored runtime cannot reach a
    /// state a live run would have refused.
    pub fn restore_compacted(
        config: PtrConfig,
        bytes: &[u8],
        trusted: CompactedAnchor,
        above_floor: &[CommittedEvent],
    ) -> Result<Self, RuntimeError> {
        let (semantic, lifecycle, execution) = compacted_sections(bytes, trusted)?;
        let semantic = SemanticDelta::decode(semantic).map_err(RuntimeError::Semantic)?;
        let lifecycle = LifecycleState::decode(lifecycle)?;
        let semdb =
            SemanticHost::restore(trusted.revision, semantic).map_err(RuntimeError::Semantic)?;

        let execution = ExecutionObligations::decode(execution)?;

        let mut runtime = Self::new(config)?;
        runtime.semdb = semdb;
        // Before the fence and the at-most-once memory are reinstated, this
        // runtime would answer a spent key by executing the effect again.
        runtime
            .execution
            .restore_obligations(execution.unsettled, execution.settled);
        runtime.state = MaterializedState {
            values: lifecycle.materialized,
            last_applied: trusted.floor.index.0,
        };
        runtime.live_generations = lifecycle.live_generations;
        runtime.capsule_projects = lifecycle.capsule_projects;
        runtime.revoked_generations = lifecycle.revoked_generations;
        // The retained records keep the indices they were committed at; a ledger
        // that renumbered them from 1 would contradict the snapshot's floor.
        runtime.ledger = RuntimeLedger::Memory(InMemoryLedger::resuming_above(trusted.floor.index));

        for expected in above_floor {
            if expected.index.0 <= trusted.floor.index.0 {
                return Err(invalid(CompactedError::JournalNotAboveFloor));
            }
            runtime.validate_lifecycle_event(&expected.event)?;
            let semantic = runtime.prepare_semantic_event(&expected.event)?;
            let actual = runtime.ledger.append(expected.event.clone())?;
            if actual != expected.index {
                return Err(RuntimeError::ReplayIndexMismatch {
                    expected: expected.index,
                    actual,
                });
            }
            let committed = runtime
                .ledger
                .events()
                .last()
                .expect("append created a committed event")
                .clone();
            runtime.apply_committed(&committed, semantic)?;
        }
        Ok(runtime)
    }
}

/// Validate framing, bounds, digest and trusted identity, then hand back the two
/// sections. Nothing is decoded before the whole artifact is accounted for.
/// The three body sections of a verified snapshot: semantic, lifecycle and
/// execution, in the order they are written.
type CompactedSections<'a> = (&'a [u8], &'a [u8], &'a [u8]);

fn compacted_sections(
    bytes: &[u8],
    trusted: CompactedAnchor,
) -> Result<CompactedSections<'_>, RuntimeError> {
    if bytes.len() < HEADER + DIGEST_BYTES || bytes.len() > MAX_COMPACTED_BYTES {
        return Err(invalid(CompactedError::SizeLimit));
    }
    if &bytes[..8] != MAGIC {
        return Err(invalid(CompactedError::UnsupportedVersion));
    }
    let revision = Revision(u64::from_le_bytes(
        bytes[8..16].try_into().expect("revision"),
    ));
    let floor = LogAnchor {
        index: CommitIndex(u64::from_le_bytes(bytes[16..24].try_into().expect("index"))),
        digest: bytes[24..56].try_into().expect("floor digest"),
    };
    let semantic_len = u64::from_le_bytes(bytes[56..64].try_into().expect("semantic length"));
    let lifecycle_len = u64::from_le_bytes(bytes[64..72].try_into().expect("lifecycle length"));
    // What was a reserved field in PTRCS001 is the execution section's length in
    // PTRCS002. An older build reading a newer snapshot stops at the magic above
    // rather than here, which is why the magic moved rather than the field alone.
    let execution_len = u64::from_le_bytes(bytes[72..80].try_into().expect("execution length"));
    if semantic_len > MAX_SECTION_BYTES as u64
        || lifecycle_len > MAX_SECTION_BYTES as u64
        || execution_len > MAX_SECTION_BYTES as u64
    {
        return Err(invalid(CompactedError::SizeLimit));
    }
    let body = (bytes.len() - HEADER - DIGEST_BYTES) as u64;
    if semantic_len
        .checked_add(lifecycle_len)
        .and_then(|sum| sum.checked_add(execution_len))
        != Some(body)
    {
        return Err(invalid(CompactedError::LengthMismatch));
    }
    let end = bytes.len() - DIGEST_BYTES;
    let digest = integrity::sha256(&bytes[..end]);
    if bytes[end..] != digest
        || digest != trusted.digest
        || revision != trusted.revision
        || floor != trusted.floor
    {
        return Err(invalid(CompactedError::AnchorMismatch));
    }
    let semantic_end = HEADER + semantic_len as usize;
    let lifecycle_end = semantic_end + lifecycle_len as usize;
    Ok((
        &bytes[HEADER..semantic_end],
        &bytes[semantic_end..lifecycle_end],
        &bytes[lifecycle_end..end],
    ))
}
