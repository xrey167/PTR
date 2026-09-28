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
use super::execution::{ActionIdentity, SettledOutcome, UnsettledEffect};
use super::{PtrRuntime, RuntimeError, RuntimeLedger};
use ptr_config::PtrConfig;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::{CommittedEvent, InMemoryLedger, MAX_RETAINED_RESPONSE};
use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Effect, Generation, ProjectId, Revision};
use std::collections::{BTreeMap, BTreeSet};

/// PTRCS004 is the layout this build writes: its lifecycle section is
/// PTRLC002, the first that may carry [`ptr_state::ATTESTED_MARKER`] and the
/// merged-branch keys ([`ptr_state::MERGED_BRANCH_PREFIX`]).
///
/// The magic moved so that a build from before attributed records stops here.
/// Reading a snapshot of a history that holds attributed records, such a
/// build would ignore the marker, then append records that history's replay
/// refuses, or merge a branch the history already merged.
const MAGIC: &[u8; 8] = b"PTRCS004";
/// PTRCS003, the layout before attributed records: still read, never written,
/// and only with a PTRLC001 lifecycle section. Such a snapshot describes a
/// history of tag-8 semantic records only, so the absence of the marker it
/// implies is true, and a PTRLC001 section that holds the marker or a
/// merged-branch key is refused by version. Its execution section is
/// PTREX002, as in PTRCS004. PTRCS003 was the first layout with PTREX002: a
/// PTRCS002 or PTRCS001 snapshot is refused by version rather than read with
/// the weaker meaning its older execution section had.
const PRE_ATTESTATION_MAGIC: &[u8; 8] = b"PTRCS003";
/// PTREX002 binds every applied key to the attempt that settled it and to the
/// project, principal, revision, generation and action digest that attempt
/// recorded, and carries the same five for every unsettled attempt. PTREX001
/// carried none of them, so a key restored from it could be neither held to its
/// action nor named by its attempt, and a section in that layout is refused by
/// version.
const EXECUTION_MAGIC: &[u8; 8] = b"PTREX002";
const LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC002";
/// The lifecycle section of a PTRCS003 snapshot, read only inside one.
const PRE_ATTESTATION_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC001";

/// Which of the two readable layouts a snapshot is in, as its outer magic
/// says. The lifecycle section must be the one that layout pairs with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    /// PTRCS004 with PTRLC002.
    Attested,
    /// PTRCS003 with PTRLC001.
    PreAttestation,
}
const HEADER: usize = 80;
const DIGEST_BYTES: usize = 32;
const MAX_SECTION_ITEMS: usize = 65_536;
/// The bound on each string the lifecycle section writes, and on each
/// at-most-once key and each unsettled attempt's target and operation the
/// execution section writes: 1 to this many bytes, as in every earlier layout.
/// A key in an attempt being committed is held to it
/// ([`MAX_KEY_BYTES`](crate::execution::MAX_KEY_BYTES)). A longer key, which a
/// log written by an earlier build may hold, is replayed, and export then fails
/// with [`CompactedError::SectionLimit`] once it has settled, as it did before.
/// The project and principal an attempt recorded are written at any length, the
/// empty string included, since no build has bounded them.
///
/// That bounds each string, not the section. Every settled key, with an
/// applied one's project, principal and retained response of up to
/// [`MAX_RETAINED_RESPONSE`](ptr_ledger::MAX_RETAINED_RESPONSE), is written
/// into the one execution section, which [`MAX_SECTION_BYTES`] and
/// `MAX_SECTION_ITEMS` bound, and nothing removes a settled key. So eight keys
/// each retaining a full-size response, or more than `MAX_SECTION_ITEMS`
/// settled keys, make every later export fail with
/// [`CompactedError::SectionLimit`] although no string is out of bounds.
/// PTREX002 writes each applied key with 64 bytes more than PTREX001 did (the
/// settling attempt, revision, generation, action digest and two lengths) plus
/// its project and principal, so a history whose execution section came close
/// to the bound in the previous layout can pass it in this one.
pub(crate) const MAX_STRING_BYTES: usize = 4096;
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
    /// Encode one bounded byte string.
    ///
    /// Separate from `count`, which bounds a *collection's cardinality* at
    /// `MAX_SECTION_ITEMS`. A retained response is a payload, not a collection,
    /// and the runtime accepts one up to `MAX_RETAINED_RESPONSE` — so encoding it
    /// through `count` refused every legal response over 65,536 bytes and made
    /// compaction impossible for the rest of that runtime's life.
    fn blob(&mut self, value: &[u8]) -> Result<(), RuntimeError> {
        if value.len() > MAX_RETAINED_RESPONSE {
            return Err(invalid(CompactedError::SectionLimit));
        }
        self.raw(&(value.len() as u32).to_le_bytes())?;
        self.raw(value)
    }

    /// Encode one little-endian unsigned integer.
    fn number(&mut self, value: u64) -> Result<(), RuntimeError> {
        self.raw(&value.to_le_bytes())
    }

    /// Encode one UTF-8 string of any length the section has room for, the
    /// empty one included.
    ///
    /// For what an attempt recorded, which no build has bounded: a snapshot
    /// must not refuse a string a log it compacts could hold. The section as a
    /// whole is still bounded ([`MAX_STRING_BYTES`] says by how much more an
    /// applied key weighs here than in the previous layout).
    fn field(&mut self, value: &str) -> Result<(), RuntimeError> {
        let length =
            u32::try_from(value.len()).map_err(|_| invalid(CompactedError::SectionLimit))?;
        self.raw(&length.to_le_bytes())?;
        self.raw(value.as_bytes())
    }

    /// Encode what an attempt recorded: project, principal, revision,
    /// generation, then the fixed-width action digest.
    fn identity(&mut self, identity: &ActionIdentity) -> Result<(), RuntimeError> {
        self.field(&identity.project.0)?;
        self.field(&identity.principal)?;
        self.number(identity.revision.0)?;
        self.number(identity.generation.0)?;
        self.raw(&identity.action_digest)
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
    /// Decode one bounded byte string, with the same bound the writer used.
    fn blob(&mut self) -> Result<Vec<u8>, RuntimeError> {
        let length = u32::from_le_bytes(self.take(4)?.try_into().expect("length bytes")) as usize;
        if length > MAX_RETAINED_RESPONSE || length > self.bytes.len().saturating_sub(self.offset) {
            return Err(invalid(CompactedError::SectionLimit));
        }
        Ok(self.take(length)?.to_vec())
    }

    /// Decode one little-endian unsigned integer.
    fn number(&mut self) -> Result<u64, RuntimeError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed number"),
        ))
    }

    /// Decode one string `Writer::field` wrote. Its length is checked against
    /// the bytes that remain before anything is allocated for it.
    fn field(&mut self) -> Result<String, RuntimeError> {
        let length = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed length")) as usize;
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| invalid(CompactedError::NoncanonicalSection))
    }

    /// Decode what an attempt recorded, in the order the writer put it.
    fn identity(&mut self) -> Result<ActionIdentity, RuntimeError> {
        Ok(ActionIdentity {
            project: ProjectId(self.field()?),
            principal: self.field()?,
            revision: Revision(self.number()?),
            generation: Generation(self.number()?),
            action_digest: self.take(DIGEST_BYTES)?.try_into().expect("fixed digest"),
        })
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
    ///
    /// The section must be the one `layout` pairs with, and a PTRLC001 section
    /// must hold neither [`ptr_state::ATTESTED_MARKER`] nor a key under
    /// [`ptr_state::MERGED_BRANCH_PREFIX`]: both are refused by version.
    fn decode(bytes: &[u8], layout: Layout) -> Result<Self, RuntimeError> {
        if bytes.len() > MAX_SECTION_BYTES {
            return Err(invalid(CompactedError::SizeLimit));
        }
        let mut reader = Reader { bytes, offset: 0 };
        let magic = match layout {
            Layout::Attested => LIFECYCLE_MAGIC,
            Layout::PreAttestation => PRE_ATTESTATION_LIFECYCLE_MAGIC,
        };
        if reader.take(8)? != magic {
            return Err(invalid(CompactedError::UnsupportedVersion));
        }
        let mut state = Self::default();
        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let key = reader.text()?;
            let value = reader.text()?;
            check_ascending(&mut previous, &key)?;
            if layout == Layout::PreAttestation
                && (key == ptr_state::ATTESTED_MARKER
                    || key.starts_with(ptr_state::MERGED_BRANCH_PREFIX))
            {
                return Err(invalid(CompactedError::UnsupportedVersion));
            }
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
///
/// A spent key travels with the attempt that settled it and with what that
/// attempt recorded, because the records that held both are below the floor: a
/// restored runtime still refuses another action under the key, and still names
/// the settling attempt when it answers or refuses a retry.
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
            out.identity(&effect.identity)?;
        }
        out.count(self.settled.len())?;
        for (key, outcome) in &self.settled {
            out.text(key)?;
            // The tag comes first because it decides which fields follow: an
            // outcome that applied names its attempt and what it recorded, and
            // one that did not binds nothing, so it carries nothing.
            match outcome {
                SettledOutcome::Applied {
                    attempt,
                    identity,
                    response,
                } => {
                    out.number(0)?;
                    out.number(attempt.0)?;
                    out.identity(identity)?;
                    out.blob(response)?;
                }
                SettledOutcome::AppliedWithoutResponse { attempt, identity } => {
                    out.number(1)?;
                    out.number(attempt.0)?;
                    out.identity(identity)?;
                }
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
            let identity = reader.identity()?;
            state.unsettled.insert(
                CommitIndex(attempt),
                UnsettledEffect {
                    attempt: CommitIndex(attempt),
                    key,
                    target,
                    operation,
                    effect,
                    identity,
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
                0 => SettledOutcome::Applied {
                    attempt: CommitIndex(reader.number()?),
                    identity: reader.identity()?,
                    response: reader.blob()?,
                },
                1 => SettledOutcome::AppliedWithoutResponse {
                    attempt: CommitIndex(reader.number()?),
                    identity: reader.identity()?,
                },
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
    /// log whose anchor carries that same floor. Each one goes through the
    /// validation replay applies, the ordinary lifecycle and semantic path, so a
    /// restored runtime cannot reach a state a replay of the whole log would have
    /// refused. That is the validation a live commit applies to the record
    /// itself, except the bound on a new attempt's key, which a log written by
    /// an earlier build may exceed; the fence a live commit also checks is the
    /// runtime's state, not the record's, and replay rebuilds it.
    pub fn restore_compacted(
        config: PtrConfig,
        bytes: &[u8],
        trusted: CompactedAnchor,
        above_floor: &[CommittedEvent],
    ) -> Result<Self, RuntimeError> {
        let (layout, (semantic, lifecycle, execution)) = compacted_sections(bytes, trusted)?;
        let semantic = SemanticDelta::decode(semantic).map_err(RuntimeError::Semantic)?;
        let lifecycle = LifecycleState::decode(lifecycle, layout)?;
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
            let semantic = runtime.prepare_semantic_event(Some(expected.index), &expected.event)?;
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

/// The three body sections of a verified snapshot: semantic, lifecycle and
/// execution, in the order they are written.
type CompactedSections<'a> = (&'a [u8], &'a [u8], &'a [u8]);

/// Validate framing, bounds, digest and trusted identity, then hand back the
/// layout and the sections. Nothing is decoded before the whole artifact is
/// accounted for.
fn compacted_sections(
    bytes: &[u8],
    trusted: CompactedAnchor,
) -> Result<(Layout, CompactedSections<'_>), RuntimeError> {
    if bytes.len() < HEADER + DIGEST_BYTES || bytes.len() > MAX_COMPACTED_BYTES {
        return Err(invalid(CompactedError::SizeLimit));
    }
    let layout = match &bytes[..8] {
        magic if magic == MAGIC => Layout::Attested,
        magic if magic == PRE_ATTESTATION_MAGIC => Layout::PreAttestation,
        _ => return Err(invalid(CompactedError::UnsupportedVersion)),
    };
    let revision = Revision(u64::from_le_bytes(
        bytes[8..16].try_into().expect("revision"),
    ));
    let floor = LogAnchor {
        index: CommitIndex(u64::from_le_bytes(bytes[16..24].try_into().expect("index"))),
        digest: bytes[24..56].try_into().expect("floor digest"),
    };
    let semantic_len = u64::from_le_bytes(bytes[56..64].try_into().expect("semantic length"));
    let lifecycle_len = u64::from_le_bytes(bytes[64..72].try_into().expect("lifecycle length"));
    // What was a reserved field in PTRCS001 is the execution section's length
    // from PTRCS002 on. An older build reading a newer snapshot stops at the magic
    // above rather than here, which is why the magic moved rather than the field
    // alone.
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
        layout,
        (
            &bytes[HEADER..semantic_end],
            &bytes[semantic_end..lifecycle_end],
            &bytes[lifecycle_end..end],
        ),
    ))
}
