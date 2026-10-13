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
use super::expert::{
    fenced_at_current_epoch, AuthorityState, ExpertAdmissionEpochs, ExpertAuthorityKind,
    ExpertAuthorityStanding, ExpertDependencyIndex, ExpertEntry, ExpertInputSet,
    ExpertRegistryProjection, FenceStanding,
};
use super::{PtrRuntime, RuntimeError, RuntimeLedger};
use ptr_config::PtrConfig;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::{
    CommittedEvent, ExpertFenceCause, ExpertFenceResume, InMemoryLedger, MAX_RETAINED_RESPONSE,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertState, ManifestDigest,
    RetiredReason,
};
use ptr_semdb::{SemanticDelta, SemanticHost};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Effect, Generation, ProjectId, Revision};
use std::collections::{BTreeMap, BTreeSet};

/// PTRCS009 is written only when an evidence authority has a standing, that is, a committed
/// `ExpertAuthorityChanged` event was applied: it is PTRCS008 with lifecycle section PTRLC007.
/// PTRLC007 is PTRLC006 with the one-number flag before the fence block widened to two (whether
/// the fence block follows, whether the dependency block follows) and the authority block
/// ([`AUTHORITY_BLOCK_MAGIC`]) at the end: the standing of the gate configuration and of the
/// evaluator (epoch and identity digest, each optional) and, per expert, the epoch of each kind
/// it was admitted or last readmitted under (an expert absent from the list stands at `0` for
/// both). A snapshot of a history without such an event is written as PTRCS008 or earlier, byte
/// for byte as before, so a build from before this layout stops at the magic exactly when it
/// would otherwise restore a quarantine regime without its epochs. A PTRCS008 or older snapshot
/// is read with no standing and every admission epoch `0`, but only when its materialized state
/// holds no `expert-authority:` key (the mark an authority change projects); one that does is
/// refused by version.
///
/// Restore checks what is derivable: the standings are non-zero epochs with non-zero identities
/// and match the materialized `expert-authority:` keys, entry for entry; the admission list is in
/// strictly ascending order, names registered experts, holds no all-zero entry and no epoch above
/// the current standing of its kind.
const AUTHORITY_MAGIC: &[u8; 8] = b"PTRCS009";
/// The lifecycle section of a PTRCS009 snapshot.
const AUTHORITY_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC007";
/// Opens the authority block at the end of a PTRLC007 section.
const AUTHORITY_BLOCK_MAGIC: &[u8; 8] = b"PTRAU001";

/// PTRCS008 is written only when the expert dependency index holds anything (a version 3
/// structural record or a compound `InputInvalidated` was applied, or a source edge below one):
/// it is PTRCS007 with lifecycle section PTRLC006, which is PTRLC005 with a one-number flag
/// saying whether the fence block ([`FENCE_MAGIC`]) follows, and then the dependency block
/// ([`DEPENDENCY_MAGIC`]): the inputs recorded for each expert generation, the source edges
/// between experts, the digest registry (the digest the first committed use fixed for each
/// input generation, advanced by a supersession) and the revoked pairs and supersession floors
/// the committed invalidations set. A snapshot of a history without any of these is written as
/// PTRCS007 or earlier, byte for byte as before, so a build from before this layout stops at the
/// magic exactly when it would otherwise restore an index-bearing history with an empty index.
/// A PTRCS007 or older snapshot is read with an empty index, but only when its materialized
/// state holds no `expert-inputs:` or `expert-input:` key (the marks a version 3 record and a
/// compound invalidation project); one that does is refused by version.
///
/// Restore checks what is derivable: every list is in strictly ascending order, every expert
/// and edge endpoint names a registered pod at a generation it has reached, every recorded input
/// has a registered digest, and the materialized `expert-input:` keys agree with the revoked
/// pairs and floors, entry for entry. Each expert generation is stored as its training list, its
/// evaluation list, its base model and the canonical digest of the inputs block its record
/// committed; restore rebuilds the block from the lists and the registered digests and refuses
/// (state mismatch) one that does not hash to the stored commitment, and the commitment of the
/// latest generation of each pod must be the digest (and the counts) its `expert-inputs:` key
/// holds. An edge must start at a tracked expert. What cannot be verified: an edge dropped
/// together with nothing else (the edges have no commitment), and an older generation whose list
/// and commitment were edited consistently; the snapshot's own integrity is the only defence
/// there. PTRCS008 was introduced with the dependency index and no released build wrote it, so
/// its expert entries were changed in place to carry the role split and the commitment.
const DEPENDENCY_MAGIC: &[u8; 8] = b"PTRCS008";
/// The lifecycle section of a PTRCS008 snapshot.
const DEPENDENCY_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC006";
/// Opens the dependency block at the end of a PTRLC006 section.
const DEPENDENCY_BLOCK_MAGIC: &[u8; 8] = b"PTRDX001";

/// PTRCS007 is written only when the registry holds a fence cause or a fence memo: it is
/// PTRCS006 with lifecycle section PTRLC005, PTRLC004 followed by the fence block
/// ([`FENCE_MAGIC`]): the cause of each fenced expert's fence and the state it returns to on
/// readmission (`Serving` or `Frozen`), which decide whether and how it may be readmitted, the
/// highest epoch each expert was ever fenced at, which keeps a KV dependency recorded before a
/// fence dead after a readmission, and the highest gate epoch any readmission was admitted
/// under. PTRCS007 and PTRFC001 were introduced with the fence event and never written by a
/// released build, so this layout is changed in place rather than versioned.
///
/// The memo cannot be re-derived from a snapshot: a readmitted expert's state version looks like
/// that of an expert that went through other transitions. Restore therefore checks what is
/// derivable and fails closed on the rest: a `Fenced` expert must have a memo equal to its state
/// version, a readmitted one a memo strictly below it, a recorded readmission gate epoch needs
/// a memo, and an older snapshot holding a `Fenced` expert gets a memo of its current epoch. A
/// memo dropped for an expert that was fenced and readmitted is not detectable here; the
/// snapshot is trusted to the extent its digest and the ledger anchor are. A snapshot with
/// neither a cause, a memo nor a readmission gate epoch is written as
/// PTRCS006, byte for byte as before, so a build from before this layout stops at the magic
/// exactly when it would otherwise restore a fenced or readmitted expert without that memory. A
/// PTRCS006 or older snapshot holding a `Fenced` expert restores it with no cause (never
/// readmitted) and a memo of its current epoch.
const FENCE_CAUSE_MAGIC: &[u8; 8] = b"PTRCS007";
/// Opens the fence block at the end of a PTRLC005 section.
const FENCE_MAGIC: &[u8; 8] = b"PTRFC001";
/// The lifecycle section of a PTRCS007 snapshot.
const FENCE_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC005";
/// PTRCS006 is the layout this build writes when no fence cause is recorded: its lifecycle section is
/// PTRLC004, PTRLC003 followed by the evidence block ([`EVIDENCE_MAGIC`]): which experts rest on
/// unverified legacy evidence and the gate configuration digest first seen for each gate epoch.
///
/// The magic moved so that a build from before the block stops here: it would restore every
/// expert as verified and an empty epoch memo.
const MAGIC: &[u8; 8] = b"PTRCS006";
/// Opens the evidence block at the end of a PTRLC004 section.
const EVIDENCE_MAGIC: &[u8; 8] = b"PTREV001";
/// The lifecycle section of a PTRCS006 snapshot.
const EVIDENCE_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC004";
/// PTRCS005, the layout before the evidence block: still read, never written. It carries no
/// record of which experts were verified, so the conservative rule applies: every restored
/// expert is marked `UnverifiedLegacy` and the epoch memo starts empty. That can only
/// understate what replay knew, never claim more; a later verified record clears a mark.
/// Refusing it would make every existing snapshot unreadable.
const PRE_EVIDENCE_MAGIC: &[u8; 8] = b"PTRCS005";
/// PTRCS004, the layout before the expert registry: still read, never written.
/// Its lifecycle section is PTRLC002. It describes a history with no expert
/// structural record, so one whose materialized state holds an `expert:` key
/// is refused by version rather than restored with an empty registry.
const PRE_REGISTRY_MAGIC: &[u8; 8] = b"PTRCS004";
/// The lifecycle section of a PTRCS004 snapshot, read only inside one.
const PRE_REGISTRY_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC002";
/// Opens the expert registry block at the end of a PTRLC003 section.
const REGISTRY_MAGIC: &[u8; 8] = b"PTREG001";
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
/// PTRLC003, the lifecycle section of a PTRCS005 snapshot.
const LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC003";
/// The lifecycle section of a PTRCS003 snapshot, read only inside one.
const PRE_ATTESTATION_LIFECYCLE_MAGIC: &[u8; 8] = b"PTRLC001";

/// Which of the two readable layouts a snapshot is in, as its outer magic
/// says. The lifecycle section must be the one that layout pairs with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    /// PTRCS009 with PTRLC007.
    Authority,
    /// PTRCS008 with PTRLC006.
    Dependency,
    /// PTRCS007 with PTRLC005.
    FenceCause,
    /// PTRCS006 with PTRLC004.
    Evidence,
    /// PTRCS005 with PTRLC003.
    Registry,
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
    experts: ExpertRegistryProjection,
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
        let fence_causes: Vec<(&str, &str, FenceStanding)> =
            self.experts.fence_cause_entries().collect();
        let fenced_epochs: Vec<(&str, &str, u64)> = self.experts.fenced_epoch_entries().collect();
        let has_fence_block = !fence_causes.is_empty()
            || !fenced_epochs.is_empty()
            || self.experts.readmit_gate_epoch() != 0;
        let has_dependency_block = !self.experts.dependencies().is_empty();
        let has_authority_block = !self.experts.authority_state().is_default();
        out.raw(if has_authority_block {
            AUTHORITY_LIFECYCLE_MAGIC
        } else if has_dependency_block {
            DEPENDENCY_LIFECYCLE_MAGIC
        } else if has_fence_block {
            FENCE_LIFECYCLE_MAGIC
        } else {
            EVIDENCE_LIFECYCLE_MAGIC
        })?;
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
        out.raw(REGISTRY_MAGIC)?;
        out.count(self.experts.len())?;
        for (project, pod, entry) in self.experts.iter() {
            out.text(project)?;
            out.text(pod)?;
            out.number(entry.generation.0)?;
            out.number(entry.state_version)?;
            out.raw(entry.manifest_digest.as_bytes())?;
            out.number(state_tag(entry.state))?;
            match entry.level {
                ExpertLevel::Token { layer, index } => {
                    out.number(0)?;
                    out.number(u64::from(layer))?;
                    out.number(u64::from(index))?;
                }
                ExpertLevel::Segment => out.number(1)?,
            }
            out.text(entry.base_model.id().as_str())?;
            out.raw(entry.base_model.digest().as_bytes())?;
            out.number(entry.base_model.generation())?;
        }
        out.raw(EVIDENCE_MAGIC)?;
        let legacy: Vec<(&str, &str)> = self.experts.legacy_keys().collect();
        out.count(legacy.len())?;
        for (project, pod) in legacy {
            out.text(project)?;
            out.text(pod)?;
        }
        let memo: Vec<(u64, [u8; 32])> = self.experts.epoch_configs().collect();
        out.count(memo.len())?;
        for (epoch, digest) in memo {
            out.number(epoch)?;
            out.raw(&digest)?;
        }
        if has_authority_block {
            out.number(u64::from(has_fence_block))?;
            out.number(u64::from(has_dependency_block))?;
        } else if has_dependency_block {
            out.number(u64::from(has_fence_block))?;
        }
        if has_fence_block {
            out.raw(FENCE_MAGIC)?;
            out.count(fence_causes.len())?;
            for (project, pod, standing) in fence_causes {
                out.text(project)?;
                out.text(pod)?;
                out.number(u64::from(standing.cause.code()))?;
                out.number(u64::from(standing.resume.code()))?;
            }
            out.count(fenced_epochs.len())?;
            for (project, pod, epoch) in fenced_epochs {
                out.text(project)?;
                out.text(pod)?;
                out.number(epoch)?;
            }
            out.number(self.experts.readmit_gate_epoch())?;
        }
        if has_dependency_block {
            encode_dependency_block(&mut out, self.experts.dependencies())?;
        }
        if has_authority_block {
            encode_authority_block(&mut out, self.experts.authority_state())?;
        }
        Ok(out.0)
    }

    /// Decoding enforces strict ascending key order per section, so each state has
    /// exactly one encoding and a reordered or duplicated entry fails closed
    /// rather than resolving to whichever entry happens to come last.
    ///
    /// The section must be the one `layout` pairs with, and a PTRLC001 section
    /// must hold neither [`ptr_state::ATTESTED_MARKER`] nor a key under
    /// [`ptr_state::MERGED_BRANCH_PREFIX`]: both are refused by version. In a
    /// PTRLC002 section every key under that prefix must be exactly a
    /// [`ptr_state::merged_branch_key`] holding exactly a
    /// [`ptr_state::merged_branch_entry`], as a merge projects them, or the
    /// section is refused as noncanonical.
    fn decode(bytes: &[u8], layout: Layout) -> Result<Self, RuntimeError> {
        if bytes.len() > MAX_SECTION_BYTES {
            return Err(invalid(CompactedError::SizeLimit));
        }
        let mut reader = Reader { bytes, offset: 0 };
        let magic = match layout {
            Layout::Authority => AUTHORITY_LIFECYCLE_MAGIC,
            Layout::Dependency => DEPENDENCY_LIFECYCLE_MAGIC,
            Layout::FenceCause => FENCE_LIFECYCLE_MAGIC,
            Layout::Evidence => EVIDENCE_LIFECYCLE_MAGIC,
            Layout::Registry => LIFECYCLE_MAGIC,
            Layout::Attested => PRE_REGISTRY_LIFECYCLE_MAGIC,
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
            if key.starts_with(ptr_state::MERGED_BRANCH_PREFIX)
                && (ptr_state::merged_branch_of(&key).is_none()
                    || ptr_state::parse_merged_branch_entry(&value).is_none())
            {
                return Err(invalid(CompactedError::NoncanonicalSection));
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
        if matches!(
            layout,
            Layout::Authority
                | Layout::Dependency
                | Layout::FenceCause
                | Layout::Evidence
                | Layout::Registry
        ) {
            let entries = decode_registry(&mut reader)?;
            state.experts = match layout {
                Layout::Authority | Layout::Dependency | Layout::FenceCause | Layout::Evidence => {
                    decode_evidence(&mut reader, entries, layout)?
                }
                _ => ExpertRegistryProjection::from_entries_all_legacy(entries),
            };
        }
        if !reader.finished() {
            return Err(invalid(CompactedError::LengthMismatch));
        }
        state.check_experts(layout)?;
        Ok(state)
    }
}

/// The tag a state is written under; [`state_from_tag`] is its inverse.
fn state_tag(state: ExpertState) -> u64 {
    match state {
        ExpertState::Candidate => 0,
        ExpertState::Gated => 1,
        ExpertState::Serving => 2,
        ExpertState::Frozen => 3,
        ExpertState::Fenced => 4,
        ExpertState::Superseded => 5,
        ExpertState::Retired(RetiredReason::Pruned) => 6,
        ExpertState::Retired(RetiredReason::Superseded) => 7,
        ExpertState::Retired(RetiredReason::GateFailed) => 8,
        ExpertState::Retired(RetiredReason::InputRevoked) => 9,
    }
}

fn state_from_tag(tag: u64) -> Result<ExpertState, RuntimeError> {
    Ok(match tag {
        0 => ExpertState::Candidate,
        1 => ExpertState::Gated,
        2 => ExpertState::Serving,
        3 => ExpertState::Frozen,
        4 => ExpertState::Fenced,
        5 => ExpertState::Superseded,
        6 => ExpertState::Retired(RetiredReason::Pruned),
        7 => ExpertState::Retired(RetiredReason::Superseded),
        8 => ExpertState::Retired(RetiredReason::GateFailed),
        9 => ExpertState::Retired(RetiredReason::InputRevoked),
        _ => return Err(invalid(CompactedError::UnknownTag)),
    })
}

fn noncanonical<E>(_: E) -> RuntimeError {
    invalid(CompactedError::NoncanonicalSection)
}

/// The decoded registry entries, in key order.
type RegistryEntries = Vec<((String, String), ExpertEntry)>;

/// Read the registry block: strictly ascending `(project, pod)`, every identity through its
/// constructor, so a value no admission could have written is refused.
fn decode_registry(reader: &mut Reader<'_>) -> Result<RegistryEntries, RuntimeError> {
    if reader.take(8)? != REGISTRY_MAGIC {
        return Err(invalid(CompactedError::UnsupportedVersion));
    }
    let digest = |reader: &mut Reader<'_>| -> Result<[u8; 32], RuntimeError> {
        Ok(reader.take(32)?.try_into().expect("fixed digest"))
    };
    let mut entries = Vec::new();
    let mut previous: Option<(String, String)> = None;
    for _ in 0..reader.count()? {
        let project = reader.text()?;
        let pod = reader.text()?;
        let key = (project, pod);
        if previous.as_ref().is_some_and(|last| *last >= key) {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        let generation = Generation(reader.number()?);
        let state_version = reader.number()?;
        let manifest_digest = ManifestDigest::new(digest(reader)?).map_err(noncanonical)?;
        let state = state_from_tag(reader.number()?)?;
        let level = match reader.number()? {
            0 => {
                let layer = u16::try_from(reader.number()?).map_err(noncanonical)?;
                let index = u16::try_from(reader.number()?).map_err(noncanonical)?;
                ExpertLevel::Token { layer, index }
            }
            1 => ExpertLevel::Segment,
            _ => return Err(invalid(CompactedError::UnknownTag)),
        };
        let base_id = BaseModelId::new(reader.text()?).map_err(noncanonical)?;
        let base_digest = BaseModelDigest::new(digest(reader)?).map_err(noncanonical)?;
        let base_model =
            BaseModelRef::new(base_id, base_digest, reader.number()?).map_err(noncanonical)?;
        if generation.0 == 0 {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        previous = Some(key.clone());
        entries.push((
            key,
            ExpertEntry {
                state,
                state_version,
                generation,
                manifest_digest,
                level,
                base_model,
            },
        ));
    }
    Ok(entries)
}

/// Read the evidence block: legacy marks (strictly ascending, each naming a registered expert) and
/// the per-epoch gate configuration memo (strictly ascending epochs, no zero digest).
fn decode_evidence(
    reader: &mut Reader<'_>,
    entries: RegistryEntries,
    layout: Layout,
) -> Result<ExpertRegistryProjection, RuntimeError> {
    if reader.take(8)? != EVIDENCE_MAGIC {
        return Err(invalid(CompactedError::UnsupportedVersion));
    }
    let mut legacy = BTreeSet::new();
    let mut previous: Option<(String, String)> = None;
    for _ in 0..reader.count()? {
        let key = (reader.text()?, reader.text()?);
        if previous.as_ref().is_some_and(|last| *last >= key)
            || !entries.iter().any(|(known, _)| *known == key)
        {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        previous = Some(key.clone());
        legacy.insert(key);
    }
    let mut memo = BTreeMap::new();
    let mut previous: Option<u64> = None;
    for _ in 0..reader.count()? {
        let epoch = reader.number()?;
        let digest: [u8; 32] = reader.take(32)?.try_into().expect("fixed digest");
        if previous.is_some_and(|last| last >= epoch) || digest == [0; 32] {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        previous = Some(epoch);
        memo.insert(epoch, digest);
    }
    // PTRLC006 says whether its fence block is present; PTRLC005 always has one and PTRLC004
    // none.
    let flag = |reader: &mut Reader<'_>| -> Result<bool, RuntimeError> {
        match reader.number()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid(CompactedError::NoncanonicalSection)),
        }
    };
    let with_fence_causes = match layout {
        Layout::FenceCause => true,
        Layout::Authority | Layout::Dependency => flag(reader)?,
        _ => false,
    };
    // PTRLC007 says whether its dependency block is present; PTRLC006 always has one.
    let with_dependencies = match layout {
        Layout::Dependency => true,
        Layout::Authority => flag(reader)?,
        _ => false,
    };
    let (fence_causes, fenced_epochs, readmit_gate_epoch) = if with_fence_causes {
        decode_fence_block(reader, &entries)?
    } else if matches!(layout, Layout::Authority | Layout::Dependency) {
        // The writer sets the flag whenever a fenced expert exists, so none may be here.
        if entries
            .iter()
            .any(|(_, entry)| entry.state == ExpertState::Fenced)
        {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        (BTreeMap::new(), BTreeMap::new(), 0)
    } else {
        let memo = fenced_at_current_epoch(&entries);
        (BTreeMap::new(), memo, 0)
    };
    let deps = if with_dependencies {
        let deps = decode_dependency_block(reader, &entries)?;
        if deps.is_empty() {
            // The writer emits the block only for a non-empty index.
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        deps
    } else {
        ExpertDependencyIndex::default()
    };
    let authority = if layout == Layout::Authority {
        decode_authority_block(reader, &entries)?
    } else {
        AuthorityState::default()
    };
    Ok(ExpertRegistryProjection::from_parts(
        entries,
        legacy,
        memo,
        fence_causes,
        fenced_epochs,
        readmit_gate_epoch,
        deps,
    )
    .with_authority(authority))
}

/// Write the authority block: the standing of each kind (gate configuration first), then the
/// admission epochs per expert in ascending key order, so each state has exactly one encoding.
fn encode_authority_block(
    out: &mut Writer,
    authority: &AuthorityState,
) -> Result<(), RuntimeError> {
    out.raw(AUTHORITY_BLOCK_MAGIC)?;
    for kind in [
        ExpertAuthorityKind::GateConfig,
        ExpertAuthorityKind::Evaluator,
    ] {
        match authority.standing(kind) {
            Some(standing) => {
                out.number(1)?;
                out.number(standing.epoch())?;
                out.raw(standing.id_digest())?;
            }
            None => out.number(0)?,
        }
    }
    let admissions: Vec<_> = authority.admission_entries().collect();
    out.count(admissions.len())?;
    for (project, pod, epochs) in admissions {
        out.text(project)?;
        out.text(pod)?;
        out.number(epochs.epoch(ExpertAuthorityKind::GateConfig))?;
        out.number(epochs.epoch(ExpertAuthorityKind::Evaluator))?;
    }
    Ok(())
}

/// Read the authority block. At least one kind has a standing (the writer emits the block only
/// then), a standing is a non-zero epoch with a non-zero identity, and the admission list is
/// strictly ascending, names registered experts, holds no all-zero entry and no epoch above the
/// current standing of its kind (an admission takes the standing in force when it happened, and
/// standings only rise).
fn decode_authority_block(
    reader: &mut Reader<'_>,
    entries: &RegistryEntries,
) -> Result<AuthorityState, RuntimeError> {
    if reader.take(8)? != AUTHORITY_BLOCK_MAGIC {
        return Err(invalid(CompactedError::UnsupportedVersion));
    }
    let canonical = |ok: bool| {
        if ok {
            Ok(())
        } else {
            Err(invalid(CompactedError::NoncanonicalSection))
        }
    };
    let mut standings = [None, None];
    for slot in &mut standings {
        match reader.number()? {
            0 => {}
            1 => {
                let epoch = reader.number()?;
                let id: [u8; 32] = reader.take(DIGEST_BYTES)?.try_into().expect("fixed digest");
                canonical(epoch != 0 && id != [0; 32])?;
                *slot = Some(ExpertAuthorityStanding::new(epoch, id));
            }
            _ => return Err(invalid(CompactedError::NoncanonicalSection)),
        }
    }
    let [gate_config, evaluator] = standings;
    canonical(gate_config.is_some() || evaluator.is_some())?;
    let ceiling = |standing: Option<ExpertAuthorityStanding>| standing.map_or(0, |s| s.epoch());
    let (gate_ceiling, evaluator_ceiling) = (ceiling(gate_config), ceiling(evaluator));
    let mut admissions = BTreeMap::new();
    let mut previous: Option<(String, String)> = None;
    for _ in 0..reader.count()? {
        let key = (reader.text()?, reader.text()?);
        let gate = reader.number()?;
        let evaluator_epoch = reader.number()?;
        let epochs = ExpertAdmissionEpochs::new(gate, evaluator_epoch);
        canonical(
            previous.as_ref().is_none_or(|last| *last < key)
                && entries.iter().any(|(known, _)| *known == key)
                && !epochs.is_default()
                && gate <= gate_ceiling
                && evaluator_epoch <= evaluator_ceiling,
        )?;
        previous = Some(key.clone());
        admissions.insert(key, epochs);
    }
    Ok(AuthorityState::from_parts(
        gate_config,
        evaluator,
        admissions,
    ))
}

/// Write the dependency block: every list in the order its map iterates, which is strictly
/// ascending, so each index has exactly one encoding.
fn encode_dependency_block(
    out: &mut Writer,
    deps: &ExpertDependencyIndex,
) -> Result<(), RuntimeError> {
    out.raw(DEPENDENCY_BLOCK_MAGIC)?;
    let experts: Vec<_> = deps.expert_entries().collect();
    out.count(experts.len())?;
    for ((project, pod, generation), set) in experts {
        out.text(project)?;
        out.text(pod)?;
        out.number(generation.0)?;
        for list in [&set.training, &set.evaluation] {
            out.count(list.len())?;
            for (identity, input_generation) in list {
                out.text(identity)?;
                out.number(input_generation.0)?;
            }
        }
        out.text(&set.base.0)?;
        out.number(set.base.1 .0)?;
        out.raw(&set.commitment)?;
    }
    let edges: Vec<_> = deps.edge_entries().collect();
    out.count(edges.len())?;
    for ((source_project, source_pod, source), (result_project, result_pod, result)) in edges {
        out.text(source_project)?;
        out.text(source_pod)?;
        out.number(source.0)?;
        out.text(result_project)?;
        out.text(result_pod)?;
        out.number(result.0)?;
    }
    let digests: Vec<_> = deps.digest_entries().collect();
    out.count(digests.len())?;
    for ((identity, generation), digest) in digests {
        out.text(identity)?;
        out.number(generation.0)?;
        out.raw(digest)?;
    }
    let revoked: Vec<_> = deps.revoked_entries().collect();
    out.count(revoked.len())?;
    for (identity, generation) in revoked {
        out.text(identity)?;
        out.number(generation.0)?;
    }
    let floors: Vec<_> = deps.floor_entries().collect();
    out.count(floors.len())?;
    for (identity, floor) in floors {
        out.text(identity)?;
        out.number(floor.0)?;
    }
    Ok(())
}

/// Read the dependency block. Every list is strictly ascending; every expert and edge endpoint
/// names a registered pod at a generation the pod has reached; every input identity is at most
/// [`MAX_EXPERT_INPUT_IDENTITY_BYTES`](ptr_pods::MAX_EXPERT_INPUT_IDENTITY_BYTES) bytes with a
/// non-zero generation; no digest is zero; every recorded input has a registered digest; and no
/// expert depends on nothing (a version 3 record always carries at least its base model).
fn decode_dependency_block(
    reader: &mut Reader<'_>,
    entries: &RegistryEntries,
) -> Result<ExpertDependencyIndex, RuntimeError> {
    if reader.take(8)? != DEPENDENCY_BLOCK_MAGIC {
        return Err(invalid(CompactedError::UnsupportedVersion));
    }
    let reached: BTreeMap<(&str, &str), Generation> = entries
        .iter()
        .map(|((project, pod), entry)| ((project.as_str(), pod.as_str()), entry.generation))
        .collect();
    let canonical = |ok: bool| {
        if ok {
            Ok(())
        } else {
            Err(invalid(CompactedError::NoncanonicalSection))
        }
    };
    let expert_ref = |reader: &mut Reader<'_>| -> Result<ExpertRefKey, RuntimeError> {
        let project = reader.text()?;
        let pod = reader.text()?;
        let generation = Generation(reader.number()?);
        let known = reached
            .get(&(project.as_str(), pod.as_str()))
            .is_some_and(|current| generation.0 != 0 && generation <= *current);
        canonical(known)?;
        Ok((project, pod, generation))
    };
    let input_key = |reader: &mut Reader<'_>| -> Result<(String, Generation), RuntimeError> {
        let identity = reader.text()?;
        let generation = Generation(reader.number()?);
        canonical(
            identity.len() <= ptr_pods::MAX_EXPERT_INPUT_IDENTITY_BYTES && generation.0 != 0,
        )?;
        Ok((identity, generation))
    };
    let mut by_expert = BTreeMap::new();
    let mut previous: Option<ExpertRefKey> = None;
    for _ in 0..reader.count()? {
        let key = expert_ref(reader)?;
        canonical(previous.as_ref().is_none_or(|last| *last < key))?;
        previous = Some(key.clone());
        let mut lists: [Vec<(String, Generation)>; 2] = [Vec::new(), Vec::new()];
        for list in &mut lists {
            let count = reader.count()?;
            for _ in 0..count {
                let input = input_key(reader)?;
                canonical(list.last().is_none_or(|last| *last < input))?;
                list.push(input);
            }
        }
        let base = input_key(reader)?;
        let commitment: [u8; 32] = reader.take(DIGEST_BYTES)?.try_into().expect("fixed digest");
        canonical(commitment != [0; 32])?;
        let [training, evaluation] = lists;
        by_expert.insert(
            key,
            ExpertInputSet::new(training, evaluation, base, commitment),
        );
    }
    let mut edges = Vec::new();
    let mut previous: Option<(ExpertRefKey, ExpertRefKey)> = None;
    for _ in 0..reader.count()? {
        let edge = (expert_ref(reader)?, expert_ref(reader)?);
        canonical(edge.0 != edge.1 && previous.as_ref().is_none_or(|last| *last < edge))?;
        previous = Some(edge.clone());
        edges.push(edge);
    }
    let mut digests = BTreeMap::new();
    let mut previous: Option<(String, Generation)> = None;
    for _ in 0..reader.count()? {
        let key = input_key(reader)?;
        let digest: [u8; 32] = reader.take(DIGEST_BYTES)?.try_into().expect("fixed digest");
        canonical(digest != [0; 32] && previous.as_ref().is_none_or(|last| *last < key))?;
        previous = Some(key.clone());
        digests.insert(key, digest);
    }
    let mut revoked = BTreeSet::new();
    let mut previous: Option<(String, Generation)> = None;
    for _ in 0..reader.count()? {
        let key = input_key(reader)?;
        canonical(previous.as_ref().is_none_or(|last| *last < key))?;
        previous = Some(key.clone());
        revoked.insert(key);
    }
    let mut floors = BTreeMap::new();
    let mut previous: Option<String> = None;
    for _ in 0..reader.count()? {
        let identity = reader.text()?;
        let floor = Generation(reader.number()?);
        canonical(
            identity.len() <= ptr_pods::MAX_EXPERT_INPUT_IDENTITY_BYTES
                && floor.0 != 0
                && previous.as_ref().is_none_or(|last| *last < identity),
        )?;
        previous = Some(identity.clone());
        floors.insert(identity, floor);
    }
    // Every recorded input generation has a digest: the first committed use fixed it.
    canonical(
        by_expert
            .values()
            .flat_map(|set| set.pairs())
            .all(|input| digests.contains_key(input)),
    )?;
    // The dependency entries of every expert generation hash to the block digest the record
    // committed: the inputs block is rebuilt from the lists and the registered digests (an
    // input's digest never changes once fixed), so a dropped, added or edited entry, or a
    // moved role, no longer matches.
    for set in by_expert.values() {
        let rebuilt = |list: &[(String, Generation)]| -> Option<Vec<ptr_pods::ExpertInputRef>> {
            list.iter()
                .map(|key| {
                    let digest = digests.get(key)?;
                    ptr_pods::ExpertInputRef::new(key.0.clone(), key.1, *digest).ok()
                })
                .collect()
        };
        let block = rebuilt(&set.training)
            .zip(rebuilt(&set.evaluation))
            .and_then(|(training, evaluation)| {
                ptr_pods::ExpertInputs::new(training, evaluation).ok()
            });
        if !block.is_some_and(|block| block.canonical_digest() == set.commitment) {
            return Err(invalid(CompactedError::StateMismatch));
        }
    }
    // An edge is recorded only when its source is tracked: it has a block or is itself the
    // result of a recorded edge.
    let descendants: BTreeSet<&ExpertRefKey> = edges.iter().map(|(_, result)| result).collect();
    canonical(
        edges
            .iter()
            .all(|(source, _)| by_expert.contains_key(source) || descendants.contains(source)),
    )?;
    Ok(ExpertDependencyIndex::from_parts(
        by_expert, edges, digests, revoked, floors,
    ))
}

/// `(project, pod, generation)` as the snapshot spells an expert generation.
type ExpertRefKey = (String, String, Generation);

type FenceBlock = (
    BTreeMap<(String, String), FenceStanding>,
    BTreeMap<(String, String), u64>,
    u64,
);

/// Read the fence block of a PTRCS007 snapshot: causes, then the fence memo. Each list is in
/// strictly ascending key order and names registered experts; a cause names a `Fenced` expert; a
/// memo epoch is at least 1 and at most the expert's state version, and equals it for a `Fenced`
/// expert, which every `Fenced` expert must have. A block with neither a cause nor a memo entry
/// is not canonical: such a snapshot is written as PTRCS006.
fn decode_fence_block(
    reader: &mut Reader<'_>,
    entries: &RegistryEntries,
) -> Result<FenceBlock, RuntimeError> {
    if reader.take(8)? != FENCE_MAGIC {
        return Err(invalid(CompactedError::UnsupportedVersion));
    }
    let find = |key: &(String, String)| entries.iter().find(|(known, _)| known == key);
    let mut causes = BTreeMap::new();
    let mut previous: Option<(String, String)> = None;
    for _ in 0..reader.count()? {
        let key = (reader.text()?, reader.text()?);
        let cause = u8::try_from(reader.number()?)
            .ok()
            .and_then(ExpertFenceCause::from_code)
            .ok_or_else(|| invalid(CompactedError::UnknownTag))?;
        let resume = u8::try_from(reader.number()?)
            .ok()
            .and_then(ExpertFenceResume::from_code)
            .ok_or_else(|| invalid(CompactedError::UnknownTag))?;
        let fenced = find(&key).is_some_and(|(_, entry)| entry.state == ExpertState::Fenced);
        if previous.as_ref().is_some_and(|last| *last >= key) || !fenced {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        previous = Some(key.clone());
        causes.insert(key, FenceStanding { cause, resume });
    }
    let mut memo = BTreeMap::new();
    let mut previous: Option<(String, String)> = None;
    for _ in 0..reader.count()? {
        let key = (reader.text()?, reader.text()?);
        let epoch = reader.number()?;
        // A fenced expert stands at exactly its fence epoch. One that is not fenced was
        // readmitted, which moved its state version past the fence: the memo is strictly below.
        let in_range = find(&key).is_some_and(|(_, entry)| {
            epoch >= 1
                && if entry.state == ExpertState::Fenced {
                    epoch == entry.state_version
                } else {
                    epoch < entry.state_version
                }
        });
        if previous.as_ref().is_some_and(|last| *last >= key) || !in_range {
            return Err(invalid(CompactedError::NoncanonicalSection));
        }
        previous = Some(key.clone());
        memo.insert(key, epoch);
    }
    let covers_fenced = entries
        .iter()
        .filter(|(_, entry)| entry.state == ExpertState::Fenced)
        .all(|(key, _)| memo.contains_key(key));
    let readmit_gate_epoch = reader.number()?;
    if (causes.is_empty() && memo.is_empty() && readmit_gate_epoch == 0) || !covers_fenced {
        return Err(invalid(CompactedError::NoncanonicalSection));
    }
    // A readmission needs an earlier fence, so a recorded readmission gate epoch needs a memo.
    if readmit_gate_epoch != 0 && memo.is_empty() {
        return Err(invalid(CompactedError::NoncanonicalSection));
    }
    Ok((causes, memo, readmit_gate_epoch))
}

impl LifecycleState {
    /// The registry and the materialized `expert:` keys describe the same experts, entry for
    /// entry. A PTRCS004 section carries no registry, so it must hold no such key.
    fn check_experts(&self, layout: Layout) -> Result<(), RuntimeError> {
        // The marks a version 3 record and a compound invalidation project exist only in a
        // history whose snapshot carries the dependency block.
        let input_marks = self.materialized.keys().any(|key| {
            key.starts_with(ptr_state::EXPERT_INPUT_PREFIX)
                || key.starts_with(ptr_state::EXPERT_INPUTS_PREFIX)
        });
        if !matches!(layout, Layout::Authority | Layout::Dependency) && input_marks {
            return Err(invalid(CompactedError::UnsupportedVersion));
        }
        // The mark an authority change projects exists only in a history whose snapshot carries
        // the authority block.
        if layout != Layout::Authority
            && self
                .materialized
                .keys()
                .any(|key| key.starts_with(ptr_state::EXPERT_AUTHORITY_PREFIX))
        {
            return Err(invalid(CompactedError::UnsupportedVersion));
        }
        let keys = self
            .materialized
            .keys()
            .filter(|key| key.starts_with(ptr_state::EXPERT_PREFIX))
            .count();
        if !matches!(
            layout,
            Layout::Authority
                | Layout::Dependency
                | Layout::FenceCause
                | Layout::Evidence
                | Layout::Registry
        ) {
            return if keys == 0 {
                Ok(())
            } else {
                Err(invalid(CompactedError::UnsupportedVersion))
            };
        }
        if keys != self.experts.len() {
            return Err(invalid(CompactedError::StateMismatch));
        }
        for (project, pod, entry) in self.experts.iter() {
            let expected =
                ptr_state::expert_entry(entry.generation.0, entry.state, entry.state_version);
            if self.materialized.get(&ptr_state::expert_key(project, pod)) != Some(&expected) {
                return Err(invalid(CompactedError::StateMismatch));
            }
        }
        if matches!(layout, Layout::Authority | Layout::Dependency) {
            self.check_dependencies()?;
        }
        if layout == Layout::Authority {
            self.check_authority_marks()?;
        }
        Ok(())
    }

    /// The authority state and the materialized `expert-authority:` keys describe the same
    /// history: one key per kind that has a standing, none for a kind that has not, and each
    /// key's epoch and identity are the standing's.
    fn check_authority_marks(&self) -> Result<(), RuntimeError> {
        let mismatch = || invalid(CompactedError::StateMismatch);
        let authority = self.experts.authority_state();
        let marks = self
            .materialized
            .keys()
            .filter(|key| key.starts_with(ptr_state::EXPERT_AUTHORITY_PREFIX))
            .count();
        let mut standings = 0;
        for kind in [
            ExpertAuthorityKind::GateConfig,
            ExpertAuthorityKind::Evaluator,
        ] {
            let Some(standing) = authority.standing(kind) else {
                continue;
            };
            standings += 1;
            let value = self
                .materialized
                .get(&ptr_state::expert_authority_key(kind))
                .ok_or_else(mismatch)?;
            let id: String = standing
                .id_digest()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            if ptr_state::parse_expert_authority_entry(value) != Some((standing.epoch(), &id)) {
                return Err(mismatch());
            }
        }
        if marks != standings {
            return Err(mismatch());
        }
        Ok(())
    }

    /// The dependency index and the materialized marks describe the same history: the
    /// `expert-input:` keys name exactly the revoked pairs (a `revoked:` value) and, per
    /// identity, end at the supersession floor (a `superseded:` value, with the new generation's
    /// digest registered); each revoked pair is in the runtime's revocation set as the compound
    /// event puts it there; and every `expert-inputs:` key names an expert generation the index
    /// holds a block for, with the highest recorded generation of each pod carrying its key.
    fn check_dependencies(&self) -> Result<(), RuntimeError> {
        let mismatch = || invalid(CompactedError::StateMismatch);
        let deps = self.experts.dependencies();
        let mut revoked = BTreeSet::new();
        let mut floors: BTreeMap<String, Generation> = BTreeMap::new();
        for (key, value) in &self.materialized {
            if let Some((identity, generation)) = ptr_state::parse_expert_input_key(key) {
                let generation = Generation(generation);
                match ptr_state::parse_expert_input_entry(value) {
                    Some(ptr_state::ExpertInputEntry::Revoked) => {
                        revoked.insert((identity.to_owned(), generation));
                    }
                    Some(ptr_state::ExpertInputEntry::Superseded { new_generation }) => {
                        let new_generation = Generation(new_generation);
                        if deps.registered_digest(identity, new_generation).is_none() {
                            return Err(mismatch());
                        }
                        let floor = floors.entry(identity.to_owned()).or_insert(new_generation);
                        *floor = (*floor).max(new_generation);
                    }
                    None => return Err(mismatch()),
                }
            } else if key.starts_with(ptr_state::EXPERT_INPUT_PREFIX) {
                return Err(mismatch());
            }
        }
        let indexed_revoked: BTreeSet<(String, Generation)> = deps
            .revoked_entries()
            .map(|(identity, generation)| (identity.clone(), *generation))
            .collect();
        let indexed_floors: BTreeMap<String, Generation> = deps
            .floor_entries()
            .map(|(identity, floor)| (identity.to_owned(), floor))
            .collect();
        if revoked != indexed_revoked
            || floors != indexed_floors
            || !indexed_revoked
                .iter()
                .all(|pair| self.revoked_generations.contains(pair))
        {
            return Err(mismatch());
        }
        let mut latest: BTreeMap<(&str, &str), (Generation, &ExpertInputSet)> = BTreeMap::new();
        for ((project, pod, generation), set) in deps.expert_entries() {
            let known = latest
                .entry((project.as_str(), pod.as_str()))
                .or_insert((*generation, set));
            if *generation > known.0 {
                *known = (*generation, set);
            }
        }
        let marks = self
            .materialized
            .keys()
            .filter(|key| key.starts_with(ptr_state::EXPERT_INPUTS_PREFIX))
            .count();
        if marks != latest.len() {
            return Err(mismatch());
        }
        for ((project, pod), (generation, set)) in latest {
            let value = self
                .materialized
                .get(&ptr_state::expert_inputs_key(project, pod))
                .ok_or_else(mismatch)?;
            if ptr_state::parse_expert_inputs_generation(value) != Some(generation.0) {
                return Err(mismatch());
            }
            // The commitment the materialized value holds is the block digest the record
            // committed, and its counts are the block's: the entries the index holds for the
            // latest generation of the pod must hash to it (every generation was checked against
            // its own commitment when the block was read).
            let (training, evaluation, digest) = inputs_mark_fields(value).ok_or_else(mismatch)?;
            if training != set.training.len()
                || evaluation != set.evaluation.len()
                || digest != set.commitment
            {
                return Err(mismatch());
            }
        }
        Ok(())
    }
}

/// The training count, evaluation count and block digest of an `expert-inputs:` value
/// (`<generation>:<training>:<evaluation>:<64 hex digits>`).
fn inputs_mark_fields(value: &str) -> Option<(usize, usize, [u8; 32])> {
    let mut fields = value.split(':');
    let _generation = fields.next()?;
    let training = fields.next()?.parse().ok()?;
    let evaluation = fields.next()?.parse().ok()?;
    let hex = fields.next()?;
    if fields.next().is_some() || hex.len() != 64 {
        return None;
    }
    let mut digest = [0u8; 32];
    for (slot, pair) in digest.iter_mut().zip(hex.as_bytes().chunks(2)) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some((training, evaluation, digest))
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
            experts: self.experts.clone(),
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
        bytes.extend_from_slice(if lifecycle.starts_with(AUTHORITY_LIFECYCLE_MAGIC) {
            AUTHORITY_MAGIC
        } else if lifecycle.starts_with(DEPENDENCY_LIFECYCLE_MAGIC) {
            DEPENDENCY_MAGIC
        } else if lifecycle.starts_with(FENCE_LIFECYCLE_MAGIC) {
            FENCE_CAUSE_MAGIC
        } else {
            MAGIC
        });
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
    /// itself, with two exceptions: the bound on a new attempt's key, which a
    /// log written by an earlier build may exceed, and the refusal of every
    /// semantic record, which only the semantic write paths may write and
    /// which replay judges by the origin rules they check instead. The fence a
    /// live commit also checks is the runtime's state, not the record's, and
    /// replay rebuilds it.
    pub fn restore_compacted(
        config: PtrConfig,
        bytes: &[u8],
        trusted: CompactedAnchor,
        above_floor: &[CommittedEvent],
    ) -> Result<Self, RuntimeError> {
        Self::restore_compacted_with_attestors(
            config,
            bytes,
            trusted,
            above_floor,
            super::expert::ExpertAttestorRegistry::new(),
        )
    }

    /// [`PtrRuntime::restore_compacted`] trusting the attestor keys the host holds: a record
    /// above the floor that carries an attestation replays only under one of them (with
    /// [`PtrRuntime::restore_compacted`] the registry is empty and such a record is refused).
    pub fn restore_compacted_with_attestors(
        config: PtrConfig,
        bytes: &[u8],
        trusted: CompactedAnchor,
        above_floor: &[CommittedEvent],
        attestors: super::expert::ExpertAttestorRegistry,
    ) -> Result<Self, RuntimeError> {
        let (layout, (semantic, lifecycle, execution)) = compacted_sections(bytes, trusted)?;
        let semantic = SemanticDelta::decode(semantic).map_err(RuntimeError::Semantic)?;
        let lifecycle = LifecycleState::decode(lifecycle, layout)?;
        let semdb =
            SemanticHost::restore(trusted.revision, semantic).map_err(RuntimeError::Semantic)?;

        let execution = ExecutionObligations::decode(execution)?;

        let mut runtime = Self::new_with_attestors(config, attestors)?;
        runtime.semdb = semdb;
        // Before the fence and the at-most-once memory are reinstated, this
        // runtime would answer a spent key by executing the effect again.
        runtime
            .execution
            .restore_obligations(execution.unsettled, execution.settled);
        // The taint registry is a projection of the `kv-taint:` entries; a malformed one fails
        // closed rather than restoring a registry that forgets a taint.
        runtime.taint = super::kv_taint::TaintRegistry::from_materialized(&lifecycle.materialized)?;
        runtime.state = MaterializedState {
            values: lifecycle.materialized,
            last_applied: trusted.floor.index.0,
        };
        runtime.live_generations = lifecycle.live_generations;
        runtime.capsule_projects = lifecycle.capsule_projects;
        runtime.revoked_generations = lifecycle.revoked_generations;
        // The edit history is not in the snapshot: every key is only known to be
        // unchanged since the snapshot's revision, and a key a revocation names is dead
        // until it is edited again.
        let snapshot = runtime.semdb.snapshot();
        runtime.semantic_edits = super::kv_prefix::SemanticEdits::restored_at(
            trusted.revision,
            runtime
                .revoked_generations
                .iter()
                .filter(|(subject, _)| snapshot.value(subject).is_some())
                .map(|(subject, _)| subject.clone()),
        );
        runtime.experts = lifecycle.experts;
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
        // As after a replay: a scope the earlier incarnation left unreleased refuses work until
        // it is recovered, whatever the snapshot's floor.
        runtime.scopes.mark_recovery_required();
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
        magic if magic == AUTHORITY_MAGIC => Layout::Authority,
        magic if magic == DEPENDENCY_MAGIC => Layout::Dependency,
        magic if magic == FENCE_CAUSE_MAGIC => Layout::FenceCause,
        magic if magic == MAGIC => Layout::Evidence,
        magic if magic == PRE_EVIDENCE_MAGIC => Layout::Registry,
        magic if magic == PRE_REGISTRY_MAGIC => Layout::Attested,
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

#[cfg(test)]
mod fence_cause_tests {
    use super::*;

    fn fenced_state(causes: BTreeMap<(String, String), FenceStanding>) -> LifecycleState {
        let entry = ExpertEntry {
            state: ExpertState::Fenced,
            state_version: 2,
            generation: Generation(1),
            manifest_digest: ManifestDigest::new([1; 32]).unwrap(),
            level: ExpertLevel::Segment,
            base_model: BaseModelRef::new(
                BaseModelId::new("b").unwrap(),
                BaseModelDigest::new([2; 32]).unwrap(),
                1,
            )
            .unwrap(),
        };
        let materialized = BTreeMap::from([(
            ptr_state::expert_key("p", "a"),
            ptr_state::expert_entry(1, ExpertState::Fenced, 2),
        )]);
        LifecycleState {
            materialized,
            experts: ExpertRegistryProjection::from_parts(
                vec![(("p".into(), "a".into()), entry)],
                BTreeSet::new(),
                BTreeMap::new(),
                causes,
                BTreeMap::from([(("p".to_owned(), "a".to_owned()), 2)]),
                0,
                ExpertDependencyIndex::default(),
            ),
            ..LifecycleState::default()
        }
    }

    fn causes(cause: ExpertFenceCause) -> BTreeMap<(String, String), FenceStanding> {
        BTreeMap::from([(
            ("p".to_owned(), "a".to_owned()),
            FenceStanding {
                cause,
                resume: ExpertFenceResume::Frozen,
            },
        )])
    }

    #[test]
    fn a_fence_cause_travels_in_the_new_section_and_round_trips() {
        let state = fenced_state(causes(ExpertFenceCause::AuthorityQuarantine));
        let bytes = state.encode().unwrap();
        assert!(bytes.starts_with(FENCE_LIFECYCLE_MAGIC));
        assert_eq!(
            LifecycleState::decode(&bytes, Layout::FenceCause).unwrap(),
            state
        );
        // An older reader of the evidence layout refuses it by version.
        assert!(LifecycleState::decode(&bytes, Layout::Evidence).is_err());
    }

    #[test]
    fn an_older_section_holding_a_fenced_expert_restores_it_with_no_cause_and_a_derived_memo() {
        let state = fenced_state(causes(ExpertFenceCause::AuthorityQuarantine));
        let mut older = state.encode().unwrap();
        // Strip the fence block and name the earlier section: what PTRCS006 wrote.
        let block = older
            .windows(8)
            .rposition(|window| window == FENCE_MAGIC)
            .unwrap();
        older.truncate(block);
        older[..8].copy_from_slice(EVIDENCE_LIFECYCLE_MAGIC);
        let decoded = LifecycleState::decode(&older, Layout::Evidence).unwrap();
        // Conservative: no cause, so never readmitted; the memo is the current epoch, so no
        // dependency recorded at or below it is live.
        assert_eq!(decoded.experts.fence_cause_entries().count(), 0);
        assert_eq!(decoded.experts.highest_fenced_epoch("expert:p/a"), Some(2));
    }

    #[test]
    fn a_tampered_fence_memo_is_refused() {
        let good = fenced_state(causes(ExpertFenceCause::AuthorityQuarantine));
        let bytes = good.encode().unwrap();
        assert_eq!(
            LifecycleState::decode(&bytes, Layout::FenceCause).unwrap(),
            good
        );
        let memo_at = bytes.len() - 16;
        // A memo lowered below the fenced expert's epoch (here 1 for epoch 2) is refused.
        let mut lowered = bytes.clone();
        lowered[memo_at..memo_at + 8].copy_from_slice(&1u64.to_le_bytes());
        assert!(LifecycleState::decode(&lowered, Layout::FenceCause).is_err());
        // So is one raised above it.
        let mut raised = bytes.clone();
        raised[memo_at..memo_at + 8].copy_from_slice(&3u64.to_le_bytes());
        assert!(LifecycleState::decode(&raised, Layout::FenceCause).is_err());
        // A memo dropped for a fenced expert is refused: the fenced expert is not covered.
        let mut dropped = fenced_state(causes(ExpertFenceCause::AuthorityQuarantine));
        let entries: Vec<_> = dropped
            .experts
            .iter()
            .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
            .collect();
        dropped.experts = ExpertRegistryProjection::from_parts(
            entries,
            BTreeSet::new(),
            BTreeMap::new(),
            causes(ExpertFenceCause::AuthorityQuarantine),
            BTreeMap::new(),
            0,
            ExpertDependencyIndex::default(),
        );
        assert!(LifecycleState::decode(&dropped.encode().unwrap(), Layout::FenceCause).is_err());
        // A readmitted expert (serving at epoch 3) whose memo claims epoch 3 was never fenced
        // below its current version: refused. Epoch 2 is the readmitted shape and restores.
        let readmitted = |memo: u64, gate_epoch: u64| {
            let mut state = fenced_state(BTreeMap::new());
            let mut entries: Vec<_> = state
                .experts
                .iter()
                .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
                .collect();
            entries[0].1.state = ExpertState::Serving;
            entries[0].1.state_version = 3;
            state.materialized = BTreeMap::from([(
                ptr_state::expert_key("p", "a"),
                ptr_state::expert_entry(1, ExpertState::Serving, 3),
            )]);
            state.experts = ExpertRegistryProjection::from_parts(
                entries,
                BTreeSet::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::from([(("p".to_owned(), "a".to_owned()), memo)]),
                gate_epoch,
                ExpertDependencyIndex::default(),
            );
            state.encode().unwrap()
        };
        let restored = LifecycleState::decode(&readmitted(2, 7), Layout::FenceCause).unwrap();
        assert_eq!(restored.experts.highest_fenced_epoch("expert:p/a"), Some(2));
        assert_eq!(restored.experts.readmit_gate_epoch(), 7);
        assert!(LifecycleState::decode(&readmitted(3, 7), Layout::FenceCause).is_err());
        assert!(LifecycleState::decode(&readmitted(0, 7), Layout::FenceCause).is_err());
    }

    #[test]
    fn a_registry_that_was_never_fenced_keeps_the_earlier_section() {
        let bytes = LifecycleState::default().encode().unwrap();
        assert!(bytes.starts_with(EVIDENCE_LIFECYCLE_MAGIC));
    }

    #[test]
    fn a_non_canonical_fence_block_is_refused() {
        let good = fenced_state(causes(ExpertFenceCause::InputRevoked))
            .encode()
            .unwrap();
        // An unknown cause code is refused: magic, count, then two one-byte ids (4 + 1 each)
        // and the code.
        let block = good
            .windows(8)
            .rposition(|window| window == FENCE_MAGIC)
            .unwrap();
        let code_at = block + 8 + 4 + 5 + 5;
        assert_eq!(good[code_at], 0);
        let mut unknown = good.clone();
        unknown[code_at] = 9;
        assert!(LifecycleState::decode(&unknown, Layout::FenceCause).is_err());
        // A memo epoch of zero is refused.
        let mut zero = good.clone();
        let end = zero.len();
        zero[end - 16..end - 8].copy_from_slice(&0u64.to_le_bytes());
        assert!(LifecycleState::decode(&zero, Layout::FenceCause).is_err());
        // A fence block with neither a cause nor a memo entry is not the encoding of any state:
        // such a snapshot is PTRCS006.
        let mut empty = LifecycleState::default().encode().unwrap();
        empty[..8].copy_from_slice(FENCE_LIFECYCLE_MAGIC);
        empty.extend_from_slice(FENCE_MAGIC);
        empty.extend_from_slice(&0u32.to_le_bytes());
        empty.extend_from_slice(&0u32.to_le_bytes());
        empty.extend_from_slice(&0u64.to_le_bytes());
        assert!(LifecycleState::decode(&empty, Layout::FenceCause).is_err());
        // A fenced expert the memo does not cover is refused.
        let mut uncovered = fenced_state(causes(ExpertFenceCause::InputRevoked));
        let entries: Vec<_> = uncovered
            .experts
            .iter()
            .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
            .collect();
        uncovered.experts = ExpertRegistryProjection::from_parts(
            entries,
            BTreeSet::new(),
            BTreeMap::new(),
            causes(ExpertFenceCause::InputRevoked),
            BTreeMap::new(),
            0,
            ExpertDependencyIndex::default(),
        );
        assert!(LifecycleState::decode(&uncovered.encode().unwrap(), Layout::FenceCause).is_err());
        // A cause for an expert that is not fenced is refused.
        let mut serving = fenced_state(causes(ExpertFenceCause::InputRevoked));
        let mut entries: Vec<_> = serving
            .experts
            .iter()
            .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
            .collect();
        entries[0].1.state = ExpertState::Serving;
        serving.materialized = BTreeMap::from([(
            ptr_state::expert_key("p", "a"),
            ptr_state::expert_entry(1, ExpertState::Serving, 2),
        )]);
        serving.experts = ExpertRegistryProjection::from_parts(
            entries,
            BTreeSet::new(),
            BTreeMap::new(),
            causes(ExpertFenceCause::InputRevoked),
            BTreeMap::from([(("p".to_owned(), "a".to_owned()), 2)]),
            0,
            ExpertDependencyIndex::default(),
        );
        assert!(LifecycleState::decode(&serving.encode().unwrap(), Layout::FenceCause).is_err());
    }

    /// The block digest of the training input `data@1` (digest `[3; 32]`) of `dependency_state`.
    fn data_block_digest() -> [u8; 32] {
        ptr_pods::ExpertInputs::new(
            vec![ptr_pods::ExpertInputRef::new("data", Generation(1), [3; 32]).unwrap()],
            vec![],
        )
        .unwrap()
        .canonical_digest()
    }

    /// A fenced expert `p/a` with a recorded dependency block, a source edge from it to `p/b`,
    /// a revoked input and a supersession, as the materialized state would show them.
    fn dependency_state() -> LifecycleState {
        let mut state = fenced_state(causes(ExpertFenceCause::InputRevoked));
        let entries: Vec<_> = state
            .experts
            .iter()
            .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
            .collect();
        let mut entries = entries;
        let mut second = entries[0].1.clone();
        second.state = ExpertState::Serving;
        second.state_version = 1;
        entries.push((("p".to_owned(), "b".to_owned()), second));
        let expert = |pod: &str| ("p".to_owned(), pod.to_owned(), Generation(1));
        let input = |identity: &str, generation: u64| (identity.to_owned(), Generation(generation));
        let deps = ExpertDependencyIndex::from_parts(
            BTreeMap::from([(
                expert("a"),
                ExpertInputSet::new(
                    vec![input("data", 1)],
                    vec![],
                    input("base-model:b", 1),
                    data_block_digest(),
                ),
            )]),
            vec![(expert("a"), expert("b"))],
            BTreeMap::from([
                (input("base-model:b", 1), [2; 32]),
                (input("data", 1), [3; 32]),
                (input("data", 2), [4; 32]),
            ]),
            BTreeSet::from([input("data", 1)]),
            BTreeMap::from([("data".to_owned(), Generation(2))]),
        );
        state.experts = ExpertRegistryProjection::from_parts(
            entries,
            BTreeSet::new(),
            BTreeMap::new(),
            causes(ExpertFenceCause::InputRevoked),
            BTreeMap::from([(("p".to_owned(), "a".to_owned()), 2)]),
            0,
            deps,
        );
        state.materialized.insert(
            ptr_state::expert_key("p", "b"),
            ptr_state::expert_entry(1, ExpertState::Serving, 1),
        );
        state.materialized.insert(
            ptr_state::expert_inputs_key("p", "a"),
            ptr_state::expert_inputs_entry(1, 1, 0, &data_block_digest()),
        );
        state.materialized.insert(
            ptr_state::expert_input_key("data", 1),
            ptr_state::expert_input_entry_revoked(1, &[7; 32]),
        );
        state.materialized.insert(
            ptr_state::expert_input_key("data", 1 + 1),
            ptr_state::expert_input_entry_superseded(2, &[4; 32], 0, &[6; 32]),
        );
        state
            .revoked_generations
            .insert(("data".to_owned(), Generation(1)));
        state
    }

    #[test]
    fn the_dependency_block_round_trips_in_its_own_section() {
        let state = dependency_state();
        let bytes = state.encode().unwrap();
        assert!(bytes.starts_with(DEPENDENCY_LIFECYCLE_MAGIC));
        assert_eq!(
            LifecycleState::decode(&bytes, Layout::Dependency).unwrap(),
            state
        );
        // The encoding is canonical: decoding and encoding again gives the same bytes.
        let decoded = LifecycleState::decode(&bytes, Layout::Dependency).unwrap();
        assert_eq!(decoded.encode().unwrap(), bytes);
        // An index-bearing section is refused by an older layout, by its magic.
        for older in [Layout::FenceCause, Layout::Evidence, Layout::Registry] {
            assert!(LifecycleState::decode(&bytes, older).is_err());
        }
        // Without a fence block the flag is zero and the block still reads back.
        let mut plain = dependency_state();
        let entries: Vec<_> = plain
            .experts
            .iter()
            .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
            .collect();
        let mut entries = entries;
        entries[0].1.state = ExpertState::Serving;
        entries[0].1.state_version = 1;
        plain.materialized.insert(
            ptr_state::expert_key("p", "a"),
            ptr_state::expert_entry(1, ExpertState::Serving, 1),
        );
        let deps = plain.experts.dependencies().clone();
        plain.experts = ExpertRegistryProjection::from_parts(
            entries,
            BTreeSet::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            0,
            deps,
        );
        let bytes = plain.encode().unwrap();
        assert_eq!(
            LifecycleState::decode(&bytes, Layout::Dependency).unwrap(),
            plain
        );
    }

    #[test]
    fn an_older_layout_holding_the_marks_of_an_index_is_refused_by_version() {
        for key in [
            ptr_state::expert_inputs_key("p", "a"),
            ptr_state::expert_input_key("data", 1),
        ] {
            let mut state = fenced_state(causes(ExpertFenceCause::AuthorityQuarantine));
            state.materialized.insert(key, "x".to_owned());
            let bytes = state.encode().unwrap();
            assert!(bytes.starts_with(FENCE_LIFECYCLE_MAGIC));
            assert!(matches!(
                LifecycleState::decode(&bytes, Layout::FenceCause),
                Err(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
            ));
        }
    }

    #[test]
    fn a_dependency_block_that_disagrees_with_itself_or_the_state_is_refused() {
        let reencode = |edit: &dyn Fn(&mut LifecycleState)| {
            let mut state = dependency_state();
            edit(&mut state);
            LifecycleState::decode(&state.encode().unwrap(), Layout::Dependency)
        };
        // Materialized marks must name the index entry for entry.
        assert!(matches!(
            reencode(&|s| {
                s.materialized
                    .remove(&ptr_state::expert_input_key("data", 1));
            }),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
        assert!(matches!(
            reencode(&|s| {
                s.materialized
                    .remove(&ptr_state::expert_input_key("data", 2));
            }),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
        assert!(matches!(
            reencode(&|s| {
                s.materialized
                    .remove(&ptr_state::expert_inputs_key("p", "a"));
            }),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
        assert!(matches!(
            reencode(&|s| {
                s.materialized.insert(
                    ptr_state::expert_inputs_key("p", "a"),
                    ptr_state::expert_inputs_entry(2, 1, 0, &data_block_digest()),
                );
            }),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
        // The materialized commitment must be the block digest the index entries hash to, and
        // the counts must be the block's.
        for wrong in [
            ptr_state::expert_inputs_entry(1, 1, 0, &[9; 32]),
            ptr_state::expert_inputs_entry(1, 0, 1, &data_block_digest()),
        ] {
            assert!(matches!(
                reencode(&|s| {
                    s.materialized
                        .insert(ptr_state::expert_inputs_key("p", "a"), wrong.clone());
                }),
                Err(RuntimeError::Compacted(CompactedError::StateMismatch))
            ));
        }
        assert!(matches!(
            reencode(&|s| {
                s.materialized.insert(
                    ptr_state::EXPERT_INPUT_PREFIX.to_owned() + "garbage",
                    "x".into(),
                );
            }),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
        // The runtime's revocation set must hold what the index says is revoked.
        assert!(matches!(
            reencode(&|s| {
                s.revoked_generations.clear();
            }),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
        // A recorded input without a registered digest, an expert or edge naming a pod the
        // registry does not hold, and an expert generation it has not reached are refused.
        let with_deps = |deps: ExpertDependencyIndex| {
            reencode(&move |s| {
                let entries: Vec<_> = s
                    .experts
                    .iter()
                    .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
                    .collect();
                s.experts = ExpertRegistryProjection::from_parts(
                    entries,
                    BTreeSet::new(),
                    BTreeMap::new(),
                    causes(ExpertFenceCause::InputRevoked),
                    BTreeMap::from([(("p".to_owned(), "a".to_owned()), 2)]),
                    0,
                    deps.clone(),
                );
            })
        };
        let expert =
            |pod: &str, generation: u64| ("p".to_owned(), pod.to_owned(), Generation(generation));
        let input = |identity: &str| (identity.to_owned(), Generation(1));
        let base_key = || ("base-model:b".to_owned(), Generation(1));
        // The block digest of an expert whose training list is `training`, when the registered
        // digests cover it, so a case fails for the reason it names and not for a wrong digest.
        let commitment = |training: &[(String, Generation)],
                          digests: &[((String, Generation), [u8; 32])]| {
            let refs: Option<Vec<_>> = training
                .iter()
                .map(|key| {
                    let digest = digests.iter().find(|(k, _)| k == key)?.1;
                    ptr_pods::ExpertInputRef::new(key.0.clone(), key.1, digest).ok()
                })
                .collect();
            refs.and_then(|refs| ptr_pods::ExpertInputs::new(refs, vec![]).ok())
                .map_or([8; 32], |block| block.canonical_digest())
        };
        let index = |experts: Vec<(ExpertRefKey, Vec<(String, Generation)>)>,
                     edges: Vec<(ExpertRefKey, ExpertRefKey)>,
                     digests: Vec<((String, Generation), [u8; 32])>| {
            ExpertDependencyIndex::from_parts(
                experts
                    .into_iter()
                    .map(|(key, training)| {
                        let sum = commitment(&training, &digests);
                        (key, ExpertInputSet::new(training, vec![], base_key(), sum))
                    })
                    .collect(),
                edges,
                digests.into_iter().collect(),
                BTreeSet::new(),
                BTreeMap::new(),
            )
        };
        let digests = || vec![(input("data"), [3; 32]), (base_key(), [2; 32])];
        for bad in [
            // A recorded input without a registered digest.
            index(vec![(expert("a", 1), vec![input("data")])], vec![], vec![]),
            // A pod the registry does not hold, and a generation it has not reached.
            index(
                vec![(expert("ghost", 1), vec![input("data")])],
                vec![],
                digests(),
            ),
            index(
                vec![(expert("a", 2), vec![input("data")])],
                vec![],
                digests(),
            ),
            index(vec![], vec![(expert("a", 1), expert("ghost", 1))], vec![]),
            index(vec![], vec![(expert("a", 1), expert("a", 1))], vec![]),
            // An edge out of an expert the index does not track.
            index(vec![], vec![(expert("a", 1), expert("b", 1))], vec![]),
            // A zero digest.
            index(vec![], vec![], vec![(input("data"), [0; 32])]),
            // Dependency entries that do not hash to the block digest the record committed: an
            // input dropped, an input added and a digest edited.
            index(vec![(expert("a", 1), vec![])], vec![], digests()),
            {
                let mut digests = digests();
                digests.push((input("extra"), [5; 32]));
                // Same commitment as `data` alone, but the list now also names `extra`.
                ExpertDependencyIndex::from_parts(
                    BTreeMap::from([(
                        expert("a", 1),
                        ExpertInputSet::new(
                            vec![input("data"), input("extra")],
                            vec![],
                            base_key(),
                            data_block_digest(),
                        ),
                    )]),
                    vec![],
                    digests.into_iter().collect(),
                    BTreeSet::new(),
                    BTreeMap::new(),
                )
            },
            index(
                vec![(expert("a", 1), vec![input("data")])],
                vec![],
                vec![(input("data"), [4; 32]), (base_key(), [2; 32])],
            ),
        ] {
            assert!(with_deps(bad).is_err());
        }
        // Positive control: the same shapes with consistent digests are accepted when nothing
        // else disagrees (the expert has a block, so does its materialized mark).
        let consistent = index(
            vec![(expert("a", 1), vec![input("data")])],
            vec![(expert("a", 1), expert("b", 1))],
            digests(),
        );
        let mut state = dependency_state();
        state
            .materialized
            .retain(|key, _| !key.starts_with(ptr_state::EXPERT_INPUT_PREFIX));
        state.revoked_generations.clear();
        let entries: Vec<_> = state
            .experts
            .iter()
            .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
            .collect();
        state.experts = ExpertRegistryProjection::from_parts(
            entries,
            BTreeSet::new(),
            BTreeMap::new(),
            causes(ExpertFenceCause::InputRevoked),
            BTreeMap::from([(("p".to_owned(), "a".to_owned()), 2)]),
            0,
            consistent,
        );
        let bytes = state.encode().unwrap();
        assert_eq!(
            LifecycleState::decode(&bytes, Layout::Dependency).unwrap(),
            state
        );
    }

    #[test]
    fn a_tampered_dependency_entry_is_refused_against_the_inputs_commitment() {
        let expert = ("p".to_owned(), "a".to_owned(), Generation(1));
        let input = |identity: &str| (identity.to_owned(), Generation(1));
        let base = || input("base-model:b");
        let digests = |extra: Option<(&str, u8)>| {
            let mut digests =
                BTreeMap::from([(input("base-model:b"), [2; 32]), (input("data"), [3; 32])]);
            if let Some((identity, byte)) = extra {
                digests.insert(input(identity), [byte; 32]);
            }
            digests
        };
        let restore = |training: Vec<(String, Generation)>,
                       evaluation: Vec<(String, Generation)>,
                       digests: BTreeMap<(String, Generation), [u8; 32]>| {
            let mut state = dependency_state();
            let entries: Vec<_> = state
                .experts
                .iter()
                .map(|(p, d, e)| ((p.to_owned(), d.to_owned()), e.clone()))
                .collect();
            state.experts = ExpertRegistryProjection::from_parts(
                entries,
                BTreeSet::new(),
                BTreeMap::new(),
                causes(ExpertFenceCause::InputRevoked),
                BTreeMap::from([(("p".to_owned(), "a".to_owned()), 2)]),
                0,
                ExpertDependencyIndex::from_parts(
                    BTreeMap::from([(
                        expert.clone(),
                        ExpertInputSet::new(training, evaluation, base(), data_block_digest()),
                    )]),
                    vec![(
                        expert.clone(),
                        ("p".to_owned(), "b".to_owned(), Generation(1)),
                    )],
                    digests,
                    BTreeSet::from([input("data")]),
                    BTreeMap::from([("data".to_owned(), Generation(2))]),
                ),
            );
            LifecycleState::decode(&state.encode().unwrap(), Layout::Dependency)
        };
        let mut with_replacement = digests(None);
        with_replacement.insert(("data".to_owned(), Generation(2)), [4; 32]);
        // The untampered entry restores.
        assert!(restore(vec![input("data")], vec![], with_replacement.clone()).is_ok());
        // An input dropped from the list, one added, one moved to the other role and a digest
        // edited all fail the commitment the record made.
        let mismatch = |result: Result<LifecycleState, RuntimeError>| {
            matches!(
                result,
                Err(RuntimeError::Compacted(CompactedError::StateMismatch))
            )
        };
        assert!(mismatch(restore(vec![], vec![], with_replacement.clone())));
        let mut extra = with_replacement.clone();
        extra.insert(input("extra"), [5; 32]);
        assert!(mismatch(restore(
            vec![input("data"), input("extra")],
            vec![],
            extra
        )));
        assert!(mismatch(restore(
            vec![],
            vec![input("data")],
            with_replacement.clone()
        )));
        let mut edited = with_replacement;
        edited.insert(input("data"), [9; 32]);
        assert!(mismatch(restore(vec![input("data")], vec![], edited)));
    }

    #[test]
    fn a_truncated_padded_or_mislabelled_dependency_block_is_refused() {
        let bytes = dependency_state().encode().unwrap();
        let block = bytes
            .windows(8)
            .rposition(|window| window == DEPENDENCY_BLOCK_MAGIC)
            .unwrap();
        // Truncation at any length inside the block.
        for cut in (block..bytes.len()).step_by(7) {
            assert!(LifecycleState::decode(&bytes[..cut], Layout::Dependency).is_err());
        }
        // Trailing bytes.
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(LifecycleState::decode(&padded, Layout::Dependency).is_err());
        // A flag other than 0 or 1 is not canonical.
        let flag = bytes
            .windows(8)
            .rposition(|window| window == FENCE_MAGIC)
            .unwrap()
            - 8;
        let mut bad_flag = bytes.clone();
        bad_flag[flag] = 2;
        assert!(LifecycleState::decode(&bad_flag, Layout::Dependency).is_err());
        // An unknown block magic reads as an unsupported version.
        let mut unknown = bytes.clone();
        unknown[block + 7] = b'9';
        assert!(matches!(
            LifecycleState::decode(&unknown, Layout::Dependency),
            Err(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
        ));
        // Swapping the digests of two registry entries moves them between keys: the section
        // still parses.
        let first = [3u8; 32];
        let second = [4u8; 32];
        let position = |needle: &[u8; 32]| {
            bytes
                .windows(32)
                .position(|window| window == needle)
                .unwrap()
        };
        let (a, b) = (position(&first), position(&second));
        let mut swapped = bytes.clone();
        swapped[a..a + 32].copy_from_slice(&second);
        swapped[b..b + 32].copy_from_slice(&first);
        // Digests swapped between keys are still ascending by key, so the section parses.
        // The block digest of `p/a` hashes the digest of `data@1`, so the swap no longer matches
        // the commitment the record made.
        assert!(matches!(
            LifecycleState::decode(&swapped, Layout::Dependency),
            Err(RuntimeError::Compacted(CompactedError::StateMismatch))
        ));
    }
}

#[cfg(test)]
mod authority_tests {
    use super::*;

    fn entry(state: ExpertState, state_version: u64) -> ExpertEntry {
        ExpertEntry {
            state,
            state_version,
            generation: Generation(1),
            manifest_digest: ManifestDigest::new([1; 32]).unwrap(),
            level: ExpertLevel::Segment,
            base_model: BaseModelRef::new(
                BaseModelId::new("b").unwrap(),
                BaseModelDigest::new([2; 32]).unwrap(),
                1,
            )
            .unwrap(),
        }
    }

    fn key(pod: &str) -> (String, String) {
        ("p".to_owned(), pod.to_owned())
    }

    fn standing(epoch: u64, id: u8) -> Option<ExpertAuthorityStanding> {
        Some(ExpertAuthorityStanding::new(epoch, [id; 32]))
    }

    fn mark(kind: ExpertAuthorityKind, epoch: u64, id: u8) -> (String, String) {
        (
            ptr_state::expert_authority_key(kind),
            ptr_state::expert_authority_entry(
                epoch,
                &[id; 32],
                ptr_ledger::ExpertAuthorityClass::NonTightening,
                0,
                &[9; 32],
            ),
        )
    }

    /// Two serving experts `a` and `b`, a gate standing at epoch 3 and an evaluator standing at
    /// epoch 2, `a` admitted under (3, 2) and `b` under (2, 0), with the materialized marks.
    fn state() -> LifecycleState {
        let admissions = BTreeMap::from([
            (key("a"), ExpertAdmissionEpochs::new(3, 2)),
            (key("b"), ExpertAdmissionEpochs::new(2, 0)),
        ]);
        let mut state = authority_state(standing(3, 1), standing(2, 2), admissions);
        for (kind, epoch, id) in [
            (ExpertAuthorityKind::GateConfig, 3, 1),
            (ExpertAuthorityKind::Evaluator, 2, 2),
        ] {
            let (k, v) = mark(kind, epoch, id);
            state.materialized.insert(k, v);
        }
        state
    }

    fn authority_state(
        gate: Option<ExpertAuthorityStanding>,
        evaluator: Option<ExpertAuthorityStanding>,
        admissions: BTreeMap<(String, String), ExpertAdmissionEpochs>,
    ) -> LifecycleState {
        let entries = vec![
            (key("a"), entry(ExpertState::Serving, 1)),
            (key("b"), entry(ExpertState::Serving, 1)),
        ];
        let materialized = ["a", "b"]
            .into_iter()
            .map(|pod| {
                (
                    ptr_state::expert_key("p", pod),
                    ptr_state::expert_entry(1, ExpertState::Serving, 1),
                )
            })
            .collect();
        LifecycleState {
            materialized,
            experts: ExpertRegistryProjection::from_parts(
                entries,
                BTreeSet::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                0,
                ExpertDependencyIndex::default(),
            )
            .with_authority(AuthorityState::from_parts(gate, evaluator, admissions)),
            ..LifecycleState::default()
        }
    }

    fn with_admissions(
        admissions: BTreeMap<(String, String), ExpertAdmissionEpochs>,
    ) -> LifecycleState {
        let mut broken = state();
        broken.experts = broken.experts.with_authority(AuthorityState::from_parts(
            standing(3, 1),
            standing(2, 2),
            admissions,
        ));
        broken
    }

    #[test]
    fn the_authority_block_round_trips_in_its_own_section_and_is_canonical() {
        let state = state();
        let bytes = state.encode().unwrap();
        assert!(bytes.starts_with(AUTHORITY_LIFECYCLE_MAGIC));
        let decoded = LifecycleState::decode(&bytes, Layout::Authority).unwrap();
        assert_eq!(decoded, state);
        assert_eq!(decoded.encode().unwrap(), bytes);
        // Every earlier layout refuses it by its section magic.
        for older in [
            Layout::Dependency,
            Layout::FenceCause,
            Layout::Evidence,
            Layout::Registry,
        ] {
            assert!(LifecycleState::decode(&bytes, older).is_err());
        }
        // One standing alone is enough, and an empty admission list is canonical.
        let mut gate_only = authority_state(standing(4, 5), None, BTreeMap::new());
        let (k, v) = mark(ExpertAuthorityKind::GateConfig, 4, 5);
        gate_only.materialized.insert(k, v);
        let bytes = gate_only.encode().unwrap();
        assert_eq!(
            LifecycleState::decode(&bytes, Layout::Authority).unwrap(),
            gate_only
        );
    }

    #[test]
    fn a_history_without_a_standing_is_written_as_an_earlier_layout() {
        let plain = authority_state(None, None, BTreeMap::new());
        let bytes = plain.encode().unwrap();
        assert!(bytes.starts_with(EVIDENCE_LIFECYCLE_MAGIC));
        assert!(LifecycleState::decode(&bytes, Layout::Authority).is_err());
    }

    #[test]
    fn an_older_layout_holding_an_authority_mark_is_refused_by_version() {
        let mut older = authority_state(None, None, BTreeMap::new());
        let (k, v) = mark(ExpertAuthorityKind::GateConfig, 3, 1);
        older.materialized.insert(k, v);
        let bytes = older.encode().unwrap();
        assert!(matches!(
            LifecycleState::decode(&bytes, Layout::Evidence),
            Err(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
        ));
    }

    #[test]
    fn a_marks_disagreeing_with_the_standings_are_refused() {
        let mismatch = |state: LifecycleState| {
            let bytes = state.encode().unwrap();
            matches!(
                LifecycleState::decode(&bytes, Layout::Authority),
                Err(RuntimeError::Compacted(CompactedError::StateMismatch))
            )
        };
        // A missing mark, a stale epoch, another identity, a mark for a kind with no standing.
        let mut missing = state();
        missing
            .materialized
            .remove(&ptr_state::expert_authority_key(
                ExpertAuthorityKind::Evaluator,
            ));
        assert!(mismatch(missing));
        let mut stale = state();
        let (k, v) = mark(ExpertAuthorityKind::GateConfig, 2, 1);
        stale.materialized.insert(k, v);
        assert!(mismatch(stale));
        let mut other = state();
        let (k, v) = mark(ExpertAuthorityKind::GateConfig, 3, 7);
        other.materialized.insert(k, v);
        assert!(mismatch(other));
        let mut extra = authority_state(standing(3, 1), None, BTreeMap::new());
        for (kind, epoch, id) in [
            (ExpertAuthorityKind::GateConfig, 3, 1),
            (ExpertAuthorityKind::Evaluator, 1, 1),
        ] {
            let (k, v) = mark(kind, epoch, id);
            extra.materialized.insert(k, v);
        }
        assert!(mismatch(extra));
        let mut garbage = state();
        garbage.materialized.insert(
            ptr_state::expert_authority_key(ExpertAuthorityKind::GateConfig),
            "3:zz".to_owned(),
        );
        assert!(mismatch(garbage));
    }

    #[test]
    fn a_non_canonical_authority_block_is_refused() {
        let noncanonical = |state: LifecycleState| {
            let bytes = state.encode().unwrap();
            matches!(
                LifecycleState::decode(&bytes, Layout::Authority),
                Err(RuntimeError::Compacted(CompactedError::NoncanonicalSection))
            )
        };
        // An epoch above the standing of its kind (a standing only rises).
        assert!(noncanonical(with_admissions(BTreeMap::from([(
            key("a"),
            ExpertAdmissionEpochs::new(4, 0)
        )]))));
        assert!(noncanonical(with_admissions(BTreeMap::from([(
            key("a"),
            ExpertAdmissionEpochs::new(0, 3)
        )]))));
        // An all-zero entry (the default is not stored).
        assert!(noncanonical(with_admissions(BTreeMap::from([(
            key("a"),
            ExpertAdmissionEpochs::new(0, 0)
        )]))));
        // An admission for an expert the registry does not hold.
        assert!(noncanonical(with_admissions(BTreeMap::from([(
            key("zz"),
            ExpertAdmissionEpochs::new(1, 1)
        )]))));
        // A standing with a zero epoch or a zero identity.
        for (gate, evaluator) in [(standing(0, 1), None), (standing(3, 0), None)] {
            assert!(noncanonical(authority_state(
                gate,
                evaluator,
                BTreeMap::new()
            )));
        }
    }

    #[test]
    fn a_truncated_padded_or_reordered_authority_block_is_refused() {
        let bytes = state().encode().unwrap();
        for len in 0..bytes.len() {
            assert!(
                LifecycleState::decode(&bytes[..len], Layout::Authority).is_err(),
                "prefix {len}"
            );
        }
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(matches!(
            LifecycleState::decode(&padded, Layout::Authority),
            Err(RuntimeError::Compacted(CompactedError::LengthMismatch))
        ));
        // Unknown block magic.
        let block = bytes
            .windows(8)
            .position(|window| window == AUTHORITY_BLOCK_MAGIC)
            .unwrap();
        let mut unknown = bytes.clone();
        unknown[block + 7] = b'9';
        assert!(matches!(
            LifecycleState::decode(&unknown, Layout::Authority),
            Err(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
        ));
        // Swapping the two admission keys breaks the strict ascending order.
        let mut swapped = bytes.clone();
        let positions: Vec<usize> = (block..bytes.len() - 1)
            .filter(|at| bytes[*at] == b'a' || bytes[*at] == b'b')
            .collect();
        // The two pod names are the last single-byte texts: swap them.
        let [.., first, second] = positions[..] else {
            panic!("pod bytes");
        };
        swapped.swap(first, second);
        assert!(matches!(
            LifecycleState::decode(&swapped, Layout::Authority),
            Err(RuntimeError::Compacted(CompactedError::NoncanonicalSection))
        ));
    }
}
