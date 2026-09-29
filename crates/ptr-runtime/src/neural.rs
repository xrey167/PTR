//! Admission for opaque model, KV-cache and checkpoint state.
//!
//! Everything else in this crate reconstructs *history*: journals, semantic
//! revisions, lifecycle generations. Neural state is the opposite kind of thing.
//! It is opaque — nothing here can look inside a tensor and decide whether it is
//! still true — and it is expensive, which is exactly why a system is tempted to
//! keep it across a restart, an edit or a revocation. That temptation is the
//! resurrection global invariant 3 forbids: a revoked generation must never become
//! usable again through a cache.
//!
//! So this module does not validate neural state. It decides **admission**, from
//! facts that live entirely outside the state: which history it was computed on,
//! which semantic values it consumed, which lifecycle generations were live, and
//! which codebook assigned the integer identities inside it. A state whose
//! surroundings have moved is refused without anyone having to interpret its bytes.
//!
//! Three properties carry the design.
//!
//! **Admission is decided at every use, never remembered.** A cache that checked
//! on insertion would hand out state that was admissible an hour ago, and
//! revocation is precisely the event that arrives afterwards. [`NeuralStateCache`]
//! therefore has no way to return a payload without a runtime to check against.
//!
//! **History is identified by its anchor, not by a number.** Two different
//! histories reach `Revision(7)`, and a restart that replays a *different* journal
//! reaches the same counters. A binding carries the [`LogAnchor`] of the position
//! it was computed at, and admission requires this runtime's own history to carry
//! that exact digest at that exact index. Because the digest chains every earlier
//! record, agreement at one index is agreement about the whole prefix.
//!
//! **Failure to verify is a denial, not an error.** Every refusal — including "this
//! runtime cannot check that position" — is one [`Denial`] type, so a caller cannot
//! accidentally treat *could not check* as *not denied*.
//!
//! What this cannot do is stated in `docs/architecture/27-neural-state-admission.md`
//! and bears repeating here: the binding is only as good as the producer's declared
//! input set. An undeclared dependency is invisible to every check below.
use super::{PtrRuntime, RuntimeError, RuntimeLedger};
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_semdb::{canonical_input_bytes, SemanticValue};
use ptr_types::{
    CheckpointError, CheckpointHeader, CodeFamily, Codebook, CodebookVersion, CommitIndex,
    EvidenceId, Generation, ProvenanceRef, Revision,
};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 8] = b"PTRNEU01";
const BINDING_MAGIC: &[u8; 8] = b"PTRNB001";
const HEADER: usize = 32;
const DIGEST_BYTES: usize = 32;
const MAX_ITEMS: usize = 65_536;
const MAX_STRING_BYTES: usize = 4096;
/// Domain separator for a single semantic input's digest, so an input digest can
/// never be confused with a delta digest computed elsewhere.
const INPUT_DOMAIN: &[u8] = b"PTRNEU01-INPUT";
pub const MAX_BINDING_BYTES: usize = 8 * 1024 * 1024;
/// Bound on one opaque payload. The payload is carried in memory because this
/// layer is about identity rather than storage; streaming large checkpoints is
/// not implemented.
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_NEURAL_BYTES: usize = HEADER + MAX_BINDING_BYTES + MAX_PAYLOAD_BYTES + DIGEST_BYTES;

/// Framing, bounds and construction faults. Distinct from [`Denial`]: these say a
/// binding or artifact is malformed, not that a well-formed one is inadmissible.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NeuralError {
    SizeLimit,
    UnsupportedVersion,
    /// The artifact names a slot encoding this build has no definition for.
    ///
    /// Distinct from `UnsupportedVersion`, which is about a codebook. Nothing here
    /// trains, so this runtime has no opinion about which encoding a checkpoint
    /// *should* carry; what it can and must refuse is one it could not reproduce —
    /// the slot vectors those weights learned from are not vectors this build can
    /// compute.
    UnsupportedEncoding {
        version: ptr_types::EncodingVersion,
    },
    LengthMismatch,
    ReservedField,
    AnchorMismatch,
    NoncanonicalSection,
    SectionLimit,
    /// A declared semantic input has no committed value to bind to.
    UndeclarableInput {
        key: String,
    },
    /// A declared lifecycle target has no live generation to bind to.
    UndeclarableTarget {
        target: String,
    },
    /// The binding this runtime just built is not one it would admit.
    Unbindable(Denial),
    /// A stored model artifact's identity header is malformed, or names an
    /// assignment this build cannot reproduce.
    CheckpointHeader(CheckpointError),
    /// The artifact and the declaration disagree about which codebook the state
    /// belongs to. Not reconciled: one of the two is wrong about what every code
    /// in the payload means, and there is no way to tell which from here.
    CheckpointCodebook {
        artifact: CodebookVersion,
        declared: CodebookVersion,
    },
}

impl NeuralError {
    /// Stable diagnostic code for this framing or construction fault.
    pub fn code(&self) -> &'static str {
        match self {
            Self::SizeLimit => "PTR_NEURAL_SIZE_LIMIT",
            Self::UnsupportedVersion => "PTR_NEURAL_VERSION",
            Self::UnsupportedEncoding { .. } => "PTR_NEURAL_ENCODING",
            Self::LengthMismatch => "PTR_NEURAL_LENGTH",
            Self::ReservedField => "PTR_NEURAL_RESERVED_FIELD",
            Self::AnchorMismatch => "PTR_NEURAL_ANCHOR_MISMATCH",
            Self::NoncanonicalSection => "PTR_NEURAL_NONCANONICAL_SECTION",
            Self::SectionLimit => "PTR_NEURAL_SECTION_LIMIT",
            Self::UndeclarableInput { .. } => "PTR_NEURAL_UNDECLARABLE_INPUT",
            Self::UndeclarableTarget { .. } => "PTR_NEURAL_UNDECLARABLE_TARGET",
            Self::Unbindable(_) => "PTR_NEURAL_UNBINDABLE",
            Self::CheckpointHeader(_) => "PTR_NEURAL_CHECKPOINT_HEADER",
            Self::CheckpointCodebook { .. } => "PTR_NEURAL_CHECKPOINT_CODEBOOK",
        }
    }
}

impl std::fmt::Display for NeuralError {
    /// Render the stable fault code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for NeuralError {}

/// Why a state may not participate.
///
/// Every variant is a refusal to admit, including the ones that mean "this runtime
/// cannot verify the claim". Collapsing *unverifiable* into the same type as
/// *stale* is deliberate: the two must lead to the same behaviour, and a separate
/// error channel for one of them is how an unverifiable state gets used anyway.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Denial {
    /// Nothing is cached under that key.
    Absent { key: String },
    /// The runtime's execution outcome is ambiguous, so it cannot vouch for any
    /// position in its own history.
    Fenced,
    /// This build has no such codebook, so the codes inside the state are not
    /// interpretable at all.
    UnknownCodebookVersion { version: CodebookVersion },
    /// The version matches but the assignment behind it does not: the state was
    /// produced by a build whose tables have since been edited.
    CodebookAssignmentChanged { version: CodebookVersion },
    /// The runtime holds no chain base, so no position can be checked.
    UnanchoredHistory,
    /// The claimed position lies outside the range this runtime can check —
    /// above what it has reached, or below a floor whose records it no longer
    /// holds. One verdict, because both mean the same thing: the claim cannot be
    /// checked, so it is not admitted. The range says which side it fell off.
    UnverifiablePosition {
        bound: CommitIndex,
        from: CommitIndex,
        through: CommitIndex,
    },
    /// This runtime's history disagrees at the claimed position.
    ForeignHistory { at: CommitIndex },
    /// The recorded revision contradicts the position it claims.
    RevisionMismatch { bound: Revision, current: Revision },
    /// A bound semantic input no longer exists.
    MissingInput { key: String },
    /// A bound semantic input's committed value changed.
    EditedInput { key: String },
    /// The runtime could not compute a bound input's digest, so it cannot compare
    /// it. Unreachable for a value that was committed — the semantic layer already
    /// encoded it once — and present because denying is the only safe answer if it
    /// ever happens.
    UncheckableInput { key: String },
    /// A bound lifecycle target has no live generation.
    UnknownTarget { target: String },
    /// A bound lifecycle target moved to a different generation.
    StaleGeneration {
        target: String,
        bound: Generation,
        current: Generation,
    },
    /// A bound generation carries a revocation tombstone.
    RevokedGeneration {
        target: String,
        generation: Generation,
    },
}

impl Denial {
    /// Stable diagnostic code for this admission refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Absent { .. } => "PTR_NEURAL_ABSENT",
            Self::Fenced => "PTR_NEURAL_FENCED",
            Self::UnknownCodebookVersion { .. } => "PTR_NEURAL_UNKNOWN_CODEBOOK",
            Self::CodebookAssignmentChanged { .. } => "PTR_NEURAL_CODEBOOK_CHANGED",
            Self::UnanchoredHistory => "PTR_NEURAL_UNANCHORED_HISTORY",
            Self::UnverifiablePosition { .. } => "PTR_NEURAL_UNVERIFIABLE_POSITION",
            Self::ForeignHistory { .. } => "PTR_NEURAL_FOREIGN_HISTORY",
            Self::RevisionMismatch { .. } => "PTR_NEURAL_REVISION_MISMATCH",
            Self::MissingInput { .. } => "PTR_NEURAL_MISSING_INPUT",
            Self::EditedInput { .. } => "PTR_NEURAL_EDITED_INPUT",
            Self::UncheckableInput { .. } => "PTR_NEURAL_UNCHECKABLE_INPUT",
            Self::UnknownTarget { .. } => "PTR_NEURAL_UNKNOWN_TARGET",
            Self::StaleGeneration { .. } => "PTR_NEURAL_STALE_GENERATION",
            Self::RevokedGeneration { .. } => "PTR_NEURAL_REVOKED_GENERATION",
        }
    }
}

impl std::fmt::Display for Denial {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for Denial {}

/// Lift a neural-state fault into the runtime error boundary.
fn invalid(kind: NeuralError) -> RuntimeError {
    RuntimeError::Neural(kind)
}

/// What a producer says its state was computed from.
///
/// Only the names are the producer's to choose; every value in the resulting
/// [`StateBinding`] is taken from committed state by
/// [`PtrRuntime::bind_state`], so a producer cannot describe a position or a
/// value it did not actually read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateDeclaration {
    pub codebook: CodebookVersion,
    /// Semantic keys the state was computed from.
    pub semantic_inputs: BTreeSet<String>,
    /// Lifecycle targets — capsule ids, `constraint:<key>`, `procedure:<id>` — the
    /// state was computed under.
    pub targets: BTreeSet<String>,
    pub provenance: Vec<ProvenanceRef>,
}

impl StateDeclaration {
    /// An empty declaration against one codebook version.
    ///
    /// There is no `Default`: a declaration without a codebook version would have
    /// to invent one, and a state whose codebook is implied is the failure the
    /// codebook exists to prevent.
    pub fn at(codebook: CodebookVersion) -> Self {
        Self {
            codebook,
            semantic_inputs: BTreeSet::new(),
            targets: BTreeSet::new(),
            provenance: Vec::new(),
        }
    }

    /// Declare one semantic key read while producing the state.
    pub fn reading(mut self, key: impl Into<String>) -> Self {
        self.semantic_inputs.insert(key.into());
        self
    }

    /// Declare one lifecycle target whose generation constrained production.
    pub fn under(mut self, target: impl Into<String>) -> Self {
        self.targets.insert(target.into());
        self
    }

    /// Append one provenance entry in declaration order.
    pub fn produced_by(mut self, provenance: ProvenanceRef) -> Self {
        self.provenance.push(provenance);
        self
    }
}

/// What a state was computed from, as committed facts rather than claims about
/// itself.
///
/// This is not a credential. Its fields are public and anyone can build one; that
/// is harmless, because nothing here grants admission — [`PtrRuntime::admit`]
/// checks every field against committed state, and the only field that is hard to
/// forge is hard to forge because forging it requires the real history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateBinding {
    pub codebook: CodebookVersion,
    /// Fingerprint of `Codebook::canonical_bytes` for that version. The version
    /// alone is not enough: tables can be edited without a version bump, and a
    /// checkpoint that keeps loading under a changed assignment is exactly the
    /// silent remapping the codebook exists to prevent.
    pub codebook_fingerprint: [u8; 32],
    /// The committed position this state was computed at.
    pub journal: LogAnchor,
    /// The semantic revision at that position. Recorded for diagnosis and checked
    /// where it is checkable — see [`PtrRuntime::admission`].
    pub revision: Revision,
    /// Each declared semantic input with a digest of its exact committed value.
    pub semantic_inputs: BTreeMap<String, [u8; 32]>,
    /// Each declared lifecycle target with the generation that was live.
    pub generations: BTreeMap<String, Generation>,
    pub provenance: Vec<ProvenanceRef>,
}

/// Opaque model, KV-cache or checkpoint state with the binding it was produced
/// under.
///
/// The payload is deliberately unreachable from this type. It becomes readable
/// only through [`AdmittedState`], which only an admission decision produces.
/// [`NeuralState::seal`] does serialize the payload, because retaining a state is
/// not using it; the boundary is the shape of the API, not a sandbox.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeuralState {
    binding: StateBinding,
    payload: Vec<u8>,
}

impl NeuralState {
    /// Pair an opaque payload with the committed facts it depends on.
    pub fn new(binding: StateBinding, payload: Vec<u8>) -> Self {
        Self { binding, payload }
    }

    /// Binding that must be rechecked before the payload is read.
    pub fn binding(&self) -> &StateBinding {
        &self.binding
    }

    /// Size of the opaque payload without exposing its contents.
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }
}

/// State an admission decision has cleared for use, borrowed for as long as the
/// decision's inputs are unchanged.
///
/// It carries no lifetime relationship to the runtime on purpose: holding one
/// across a commit is a programming mistake the type system cannot catch, so
/// callers re-admit per use rather than storing this.
#[derive(Clone, Copy, Debug)]
pub struct AdmittedState<'a> {
    state: &'a NeuralState,
}

impl<'a> AdmittedState<'a> {
    /// Binding that was checked to produce this admitted handle.
    pub fn binding(&self) -> &'a StateBinding {
        &self.state.binding
    }

    /// Opaque bytes made reachable by the admission decision.
    pub fn payload(&self) -> &'a [u8] {
        &self.state.payload
    }
}

/// Trusted identity of a sealed state, retained outside the artifact.
///
/// Ordinary data, not authentication, exactly as with
/// [`CompactedAnchor`](crate::compacted::CompactedAnchor): a digest read back out
/// of the file it describes proves nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeuralAnchor {
    pub journal: LogAnchor,
    pub codebook: CodebookVersion,
    pub digest: [u8; 32],
}

/// A sealed state and the anchor a catalog has to retain for it.
#[derive(Debug)]
pub struct SealedState {
    bytes: Vec<u8>,
    anchor: NeuralAnchor,
}

impl SealedState {
    /// Canonical sealed bytes of this retained state.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Trusted identity the host must retain outside the artifact.
    pub fn anchor(&self) -> NeuralAnchor {
        self.anchor
    }
}

struct Writer(Vec<u8>);

impl Writer {
    /// Append bytes while enforcing the binding-size bound.
    fn raw(&mut self, bytes: &[u8]) -> Result<(), RuntimeError> {
        if bytes.len() > MAX_BINDING_BYTES.saturating_sub(self.0.len()) {
            return Err(invalid(NeuralError::SectionLimit));
        }
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    /// Encode a bounded collection count.
    fn count(&mut self, value: usize) -> Result<(), RuntimeError> {
        if value > MAX_ITEMS {
            return Err(invalid(NeuralError::SectionLimit));
        }
        self.raw(&(value as u32).to_le_bytes())
    }
    /// Encode one nonempty, bounded UTF-8 string.
    fn text(&mut self, value: &str) -> Result<(), RuntimeError> {
        if value.is_empty() || value.len() > MAX_STRING_BYTES {
            return Err(invalid(NeuralError::SectionLimit));
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
    /// Consume exactly `length` bytes from the binding.
    fn take(&mut self, length: usize) -> Result<&'a [u8], RuntimeError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(invalid(NeuralError::SectionLimit))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(invalid(NeuralError::LengthMismatch))?;
        self.offset = end;
        Ok(slice)
    }
    /// Counts bound allocation, so they are checked against the bytes that remain
    /// before anything is reserved for them.
    fn count(&mut self) -> Result<usize, RuntimeError> {
        let value = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed count")) as usize;
        if value > MAX_ITEMS || value > self.bytes.len().saturating_sub(self.offset) {
            return Err(invalid(NeuralError::SectionLimit));
        }
        Ok(value)
    }
    /// Decode one nonempty, bounded UTF-8 string.
    fn text(&mut self) -> Result<String, RuntimeError> {
        let length = u32::from_le_bytes(self.take(4)?.try_into().expect("fixed length")) as usize;
        if length == 0 || length > MAX_STRING_BYTES {
            return Err(invalid(NeuralError::SectionLimit));
        }
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| invalid(NeuralError::NoncanonicalSection))
    }
    /// Decode one little-endian unsigned integer.
    fn number(&mut self) -> Result<u64, RuntimeError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed number"),
        ))
    }
    /// Decode one fixed-width SHA-256 digest.
    fn digest(&mut self) -> Result<[u8; 32], RuntimeError> {
        Ok(self.take(32)?.try_into().expect("fixed digest"))
    }
    /// Decode the canonical binary presence flag.
    fn flag(&mut self) -> Result<bool, RuntimeError> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            // A presence byte with a third value is a second encoding of the same
            // meaning, which the digest would then cover two ways.
            _ => Err(invalid(NeuralError::ReservedField)),
        }
    }
    /// Whether the binding has no trailing bytes.
    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Require the next decoded key to be strictly greater than its predecessor.
fn check_ascending(previous: &mut Option<String>, key: &str) -> Result<(), RuntimeError> {
    if previous.as_deref().is_some_and(|last| last >= key) {
        return Err(invalid(NeuralError::NoncanonicalSection));
    }
    *previous = Some(key.to_owned());
    Ok(())
}

impl StateBinding {
    /// Canonical bytes: strictly ascending keys for the two maps, declaration order
    /// for provenance.
    ///
    /// The maps are sets, so a reordering is not a different value and must not be
    /// a different encoding. Provenance is a list whose order is part of what was
    /// recorded, so it is written as given.
    fn encode(&self) -> Result<Vec<u8>, RuntimeError> {
        let mut out = Writer(Vec::new());
        out.raw(BINDING_MAGIC)?;
        out.raw(&self.codebook.0.to_le_bytes())?;
        out.raw(&self.codebook_fingerprint)?;
        out.number(self.journal.index.0)?;
        out.raw(&self.journal.digest)?;
        out.number(self.revision.0)?;
        out.count(self.semantic_inputs.len())?;
        for (key, digest) in &self.semantic_inputs {
            out.text(key)?;
            out.raw(digest)?;
        }
        out.count(self.generations.len())?;
        for (target, generation) in &self.generations {
            out.text(target)?;
            out.number(generation.0)?;
        }
        out.count(self.provenance.len())?;
        for entry in &self.provenance {
            out.text(&entry.source.0)?;
            match &entry.note {
                Some(note) => {
                    out.raw(&[1])?;
                    out.text(note)?;
                }
                None => out.raw(&[0])?,
            }
        }
        Ok(out.0)
    }

    /// Decode the single canonical representation of a state binding.
    fn decode(bytes: &[u8]) -> Result<Self, RuntimeError> {
        if bytes.len() > MAX_BINDING_BYTES {
            return Err(invalid(NeuralError::SizeLimit));
        }
        let mut reader = Reader { bytes, offset: 0 };
        if reader.take(8)? != BINDING_MAGIC {
            return Err(invalid(NeuralError::UnsupportedVersion));
        }
        let codebook = CodebookVersion(u32::from_le_bytes(
            reader.take(4)?.try_into().expect("fixed version"),
        ));
        let codebook_fingerprint = reader.digest()?;
        let journal = LogAnchor {
            index: CommitIndex(reader.number()?),
            digest: reader.digest()?,
        };
        let revision = Revision(reader.number()?);

        let mut semantic_inputs = BTreeMap::new();
        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let key = reader.text()?;
            let digest = reader.digest()?;
            check_ascending(&mut previous, &key)?;
            semantic_inputs.insert(key, digest);
        }

        let mut generations = BTreeMap::new();
        let mut previous: Option<String> = None;
        for _ in 0..reader.count()? {
            let target = reader.text()?;
            let generation = Generation(reader.number()?);
            check_ascending(&mut previous, &target)?;
            generations.insert(target, generation);
        }

        let mut provenance = Vec::new();
        for _ in 0..reader.count()? {
            let source = reader.text()?;
            let note = if reader.flag()? {
                Some(reader.text()?)
            } else {
                None
            };
            provenance.push(ProvenanceRef {
                source: EvidenceId(source),
                note,
            });
        }

        if !reader.finished() {
            return Err(invalid(NeuralError::LengthMismatch));
        }
        Ok(Self {
            codebook,
            codebook_fingerprint,
            journal,
            revision,
            semantic_inputs,
            generations,
            provenance,
        })
    }
}

impl NeuralState {
    /// Serialize for retention, with the anchor a catalog must keep elsewhere.
    pub fn seal(&self) -> Result<SealedState, RuntimeError> {
        let binding = self.binding.encode()?;
        if binding.len() > MAX_BINDING_BYTES || self.payload.len() > MAX_PAYLOAD_BYTES {
            return Err(invalid(NeuralError::SizeLimit));
        }
        let mut bytes =
            Vec::with_capacity(HEADER + binding.len() + self.payload.len() + DIGEST_BYTES);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(binding.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        debug_assert_eq!(bytes.len(), HEADER);
        bytes.extend_from_slice(&binding);
        bytes.extend_from_slice(&self.payload);
        let digest = integrity::sha256(&bytes);
        bytes.extend_from_slice(&digest);
        Ok(SealedState {
            bytes,
            anchor: NeuralAnchor {
                journal: self.binding.journal,
                codebook: self.binding.codebook,
                digest,
            },
        })
    }

    /// Parse a sealed state against an independently retained anchor.
    ///
    /// Decoding is not admission. A state can be perfectly intact, match its
    /// anchor exactly, and still be inadmissible — which is the normal case after a
    /// revocation.
    pub fn open(bytes: &[u8], trusted: NeuralAnchor) -> Result<Self, RuntimeError> {
        if bytes.len() < HEADER + DIGEST_BYTES || bytes.len() > MAX_NEURAL_BYTES {
            return Err(invalid(NeuralError::SizeLimit));
        }
        if &bytes[..8] != MAGIC {
            return Err(invalid(NeuralError::UnsupportedVersion));
        }
        let binding_len = u64::from_le_bytes(bytes[8..16].try_into().expect("binding length"));
        let payload_len = u64::from_le_bytes(bytes[16..24].try_into().expect("payload length"));
        if bytes[24..32] != [0; 8] {
            return Err(invalid(NeuralError::ReservedField));
        }
        if binding_len > MAX_BINDING_BYTES as u64 || payload_len > MAX_PAYLOAD_BYTES as u64 {
            return Err(invalid(NeuralError::SizeLimit));
        }
        let body = (bytes.len() - HEADER - DIGEST_BYTES) as u64;
        if binding_len.checked_add(payload_len) != Some(body) {
            return Err(invalid(NeuralError::LengthMismatch));
        }
        let end = bytes.len() - DIGEST_BYTES;
        let digest = integrity::sha256(&bytes[..end]);
        if bytes[end..] != digest || digest != trusted.digest {
            return Err(invalid(NeuralError::AnchorMismatch));
        }
        let split = HEADER + binding_len as usize;
        let binding = StateBinding::decode(&bytes[HEADER..split])?;
        // The anchor's other two fields are redundant with the binding, and that
        // redundancy is the point: a catalog entry states what it believes it is
        // holding, so a substituted artifact is caught by the catalog and not only
        // by the digest.
        if binding.journal != trusted.journal || binding.codebook != trusted.codebook {
            return Err(invalid(NeuralError::AnchorMismatch));
        }
        Ok(Self {
            binding,
            payload: bytes[split..end].to_vec(),
        })
    }
}

/// Digest of one committed semantic value, through the semantic layer's own
/// canonical encoder.
///
/// Reusing that encoder rather than hashing a private rendering means a binding
/// and a replay cannot disagree about what a semantic value is — the same reason
/// [`crate::compacted`] reuses it for its semantic section.
fn input_digest(key: &str, value: &SemanticValue) -> Result<[u8; 32], RuntimeError> {
    let encoded = canonical_input_bytes(key, value).map_err(RuntimeError::Semantic)?;
    let mut material = Vec::with_capacity(INPUT_DOMAIN.len() + encoded.len());
    material.extend_from_slice(INPUT_DOMAIN);
    material.extend_from_slice(&encoded);
    Ok(integrity::sha256(&material))
}

impl RuntimeLedger {
    /// The chain base this ledger continues above, when it has one.
    ///
    /// An in-memory ledger resumed above a compaction floor knows the floor's
    /// index but not its digest, so it has no base to offer. Returning `None`
    /// rather than substituting the empty anchor is the difference between
    /// refusing to check and checking against the wrong chain.
    fn journal_base(&self) -> Option<LogAnchor> {
        match self {
            Self::Memory(ledger) => (ledger.base().0 == 0).then(LogAnchor::empty),
            Self::File(log) => Some(log.base()),
        }
    }
}

impl PtrRuntime {
    /// Build a binding for state computed from `declaration` at this runtime's
    /// current position.
    ///
    /// Every value comes from committed state, and the result is checked through
    /// the same admission rules a later use will apply — so there is one rule set,
    /// not a producer's copy and a consumer's copy that can drift apart.
    pub fn bind_state(&self, declaration: &StateDeclaration) -> Result<StateBinding, RuntimeError> {
        let book = Codebook::at(declaration.codebook)
            .map_err(|_| invalid(NeuralError::UnsupportedVersion))?;
        let journal = self.journal_anchor()?;
        let snapshot = self.semdb.snapshot();

        let mut semantic_inputs = BTreeMap::new();
        for key in &declaration.semantic_inputs {
            let value = snapshot
                .value(key)
                .ok_or_else(|| invalid(NeuralError::UndeclarableInput { key: key.clone() }))?;
            semantic_inputs.insert(key.clone(), input_digest(key, value)?);
        }

        let mut generations = BTreeMap::new();
        for target in &declaration.targets {
            let generation = self.live_generation(target).ok_or_else(|| {
                invalid(NeuralError::UndeclarableTarget {
                    target: target.clone(),
                })
            })?;
            generations.insert(target.clone(), generation);
        }

        let binding = StateBinding {
            codebook: declaration.codebook,
            codebook_fingerprint: integrity::sha256(&book.canonical_bytes()),
            journal,
            revision: self.revision(),
            semantic_inputs,
            generations,
            provenance: declaration.provenance.clone(),
        };
        self.admission(&binding)
            .map_err(|denial| invalid(NeuralError::Unbindable(denial)))?;
        Ok(binding)
    }

    /// Bind a stored model artifact — a real checkpoint, header and weights — to
    /// the committed facts it was produced under.
    ///
    /// The artifact carries the codebook assignment it was trained against, so
    /// this checks that identity *before* the payload is retained at all, and then
    /// binds it through [`PtrRuntime::bind_state`] so a checkpoint and any other
    /// neural state pass the same admission rules.
    ///
    /// `required` names the code families the caller needs the artifact to agree
    /// about. A model that embeds fewer families than the kernel defines is not
    /// wrong, so this layer cannot know the list: `ptr-burn-a0` names its own as
    /// `EMBEDDED_FAMILIES`. Naming none still checks the assignment itself, which
    /// is the part that decides what every code means.
    ///
    /// The payload stays opaque here. Nothing in this crate can read burn's
    /// parameter format, and nothing in this crate needs to: what a checkpoint
    /// must not do is load under an assignment it was not trained under.
    pub fn bind_checkpoint(
        &self,
        bytes: &[u8],
        declaration: &StateDeclaration,
        required: &[CodeFamily],
    ) -> Result<NeuralState, RuntimeError> {
        let (header, payload) = CheckpointHeader::read(bytes)
            .map_err(|error| invalid(NeuralError::CheckpointHeader(error)))?;
        if header.codebook != declaration.codebook {
            return Err(invalid(NeuralError::CheckpointCodebook {
                artifact: header.codebook,
                declared: declaration.codebook,
            }));
        }
        let book =
            Codebook::at(header.codebook).map_err(|_| invalid(NeuralError::UnsupportedVersion))?;
        // The encoding is checked by being *resolved*: a definition this build does
        // not have is one whose slot vectors it cannot recompute, so the weights
        // learned from inputs nothing here can produce.
        //
        // Deliberately not part of the `StateDeclaration`. A declaration says which
        // committed facts a state was computed from; the encoding is how an artifact
        // was constructed, which is what a header records — and not every neural
        // state has slot vectors at all, so a declaration naming one would attach it
        // to states it does not apply to.
        let encoding = ptr_types::SlotEncoding::at(header.encoding).ok_or_else(|| {
            invalid(NeuralError::UnsupportedEncoding {
                version: header.encoding,
            })
        })?;
        header
            .verify(&book, encoding, required)
            .map_err(|error| invalid(NeuralError::CheckpointHeader(error)))?;
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(invalid(NeuralError::SizeLimit));
        }
        let binding = self.bind_state(declaration)?;
        Ok(NeuralState::new(binding, payload.to_vec()))
    }

    /// Decide whether a binding may participate, without holding any payload.
    ///
    /// Checks run in a fixed order, and revocation comes first among the
    /// content checks. A revocation tombstone must deny a state even when
    /// everything else about it is unverifiable, so it must not sit behind a check
    /// that can fail for another reason.
    ///
    /// The recorded revision is only checked when the bound position is this
    /// runtime's current position. At an earlier position the revision that held
    /// then is not recoverable without replaying, so the anchored history and the
    /// per-input digests carry the weight; see the contract for what that leaves
    /// open.
    pub fn admission(&self, binding: &StateBinding) -> Result<(), Denial> {
        if self.execution.is_fenced() {
            return Err(Denial::Fenced);
        }

        for (target, generation) in &binding.generations {
            if self
                .revoked_generations
                .contains(&(target.clone(), *generation))
            {
                return Err(Denial::RevokedGeneration {
                    target: target.clone(),
                    generation: *generation,
                });
            }
        }

        let book = Codebook::at(binding.codebook).map_err(|_| Denial::UnknownCodebookVersion {
            version: binding.codebook,
        })?;
        if integrity::sha256(&book.canonical_bytes()) != binding.codebook_fingerprint {
            return Err(Denial::CodebookAssignmentChanged {
                version: binding.codebook,
            });
        }

        self.check_history(binding)?;

        let snapshot = self.semdb.snapshot();
        for (key, digest) in &binding.semantic_inputs {
            let value = snapshot
                .value(key)
                .ok_or_else(|| Denial::MissingInput { key: key.clone() })?;
            let current = input_digest(key, value)
                .map_err(|_| Denial::UncheckableInput { key: key.clone() })?;
            if current != *digest {
                return Err(Denial::EditedInput { key: key.clone() });
            }
        }

        for (target, generation) in &binding.generations {
            let current = self
                .live_generation(target)
                .ok_or_else(|| Denial::UnknownTarget {
                    target: target.clone(),
                })?;
            // Equality, not "at least": state computed under generation 3 describes
            // generation 3. Generation 4 is a different thing, not a newer view of
            // the same thing.
            if current != *generation {
                return Err(Denial::StaleGeneration {
                    target: target.clone(),
                    bound: *generation,
                    current,
                });
            }
        }

        Ok(())
    }

    /// Require this runtime's own history to carry the bound anchor at the bound
    /// index.
    fn check_history(&self, binding: &StateBinding) -> Result<(), Denial> {
        let base = self
            .ledger
            .journal_base()
            .ok_or(Denial::UnanchoredHistory)?;
        let events = self.ledger.events();
        let from = base.index;
        let through = CommitIndex(base.index.0.saturating_add(events.len() as u64));
        let bound = binding.journal.index;

        // One condition for both ends of the range, so the "cannot check" verdict
        // has a single construction site rather than one per direction.
        if bound < from || bound > through {
            return Err(Denial::UnverifiablePosition {
                bound,
                from,
                through,
            });
        }
        let expected = if bound == base.index {
            base
        } else {
            let offset = (bound.0 - base.index.0 - 1) as usize;
            let anchors = integrity::chain_anchors(events, base)
                .map_err(|_| Denial::ForeignHistory { at: bound })?;
            *anchors
                .get(offset)
                .ok_or(Denial::ForeignHistory { at: bound })?
        };
        if expected != binding.journal {
            return Err(Denial::ForeignHistory { at: bound });
        }
        if bound == through && binding.revision != self.revision() {
            return Err(Denial::RevisionMismatch {
                bound: binding.revision,
                current: self.revision(),
            });
        }
        Ok(())
    }

    /// Admit a state for use, yielding the only handle its payload is reachable
    /// through.
    pub fn admit<'a>(&self, state: &'a NeuralState) -> Result<AdmittedState<'a>, Denial> {
        self.admission(&state.binding)?;
        Ok(AdmittedState { state })
    }
}

/// Retained neural states keyed by whatever the host uses to find them.
///
/// The cache holds state; it does not hold admission. There is no accessor that
/// returns a payload without a runtime to decide against, because the decision has
/// to be made at the moment of use — a revocation arrives after the insert, not
/// before it.
#[derive(Clone, Debug, Default)]
pub struct NeuralStateCache {
    entries: BTreeMap<String, NeuralState>,
}

impl NeuralStateCache {
    /// Insert or replace retained state without treating it as admitted.
    pub fn insert(&mut self, key: impl Into<String>, state: NeuralState) -> Option<NeuralState> {
        self.entries.insert(key.into(), state)
    }

    /// Remove retained state without reading its payload.
    pub fn remove(&mut self, key: &str) -> Option<NeuralState> {
        self.entries.remove(key)
    }

    /// Number of retained states, admitted or denied.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no state is retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate retained keys in deterministic order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// The binding of a retained entry, for inspection. Never its payload.
    pub fn binding(&self, key: &str) -> Option<&StateBinding> {
        self.entries.get(key).map(NeuralState::binding)
    }

    /// Decide admission for one entry, now, against `runtime`.
    pub fn admit<'a>(
        &'a self,
        runtime: &PtrRuntime,
        key: &str,
    ) -> Result<AdmittedState<'a>, Denial> {
        let state = self.entries.get(key).ok_or_else(|| Denial::Absent {
            key: key.to_owned(),
        })?;
        runtime.admit(state)
    }

    /// Entries `runtime` would currently refuse.
    pub fn denied(&self, runtime: &PtrRuntime) -> Vec<(String, Denial)> {
        self.entries
            .iter()
            .filter_map(|(key, state)| {
                runtime
                    .admission(&state.binding)
                    .err()
                    .map(|denial| (key.clone(), denial))
            })
            .collect()
    }

    /// Drop entries `runtime` would currently refuse.
    ///
    /// Housekeeping, not a safety boundary: what makes the cache safe is that
    /// [`NeuralStateCache::admit`] re-decides every time. A host that never called
    /// this would still never be handed a denied payload.
    pub fn evict_denied(&mut self, runtime: &PtrRuntime) -> Vec<(String, Denial)> {
        let denied = self.denied(runtime);
        for (key, _) in &denied {
            self.entries.remove(key);
        }
        denied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> StateBinding {
        StateBinding {
            codebook: CodebookVersion::V1,
            codebook_fingerprint: [7; 32],
            journal: LogAnchor {
                index: CommitIndex(4),
                digest: [9; 32],
            },
            revision: Revision(3),
            // Equal-length keys, so the reordering test below can exchange two
            // fixed-size entries without disturbing any length field.
            semantic_inputs: BTreeMap::from([
                ("alpha".to_owned(), [1; 32]),
                ("gamma".to_owned(), [2; 32]),
            ]),
            generations: BTreeMap::from([("capsule".to_owned(), Generation(2))]),
            provenance: vec![
                ProvenanceRef {
                    source: "trainer".into(),
                    note: Some("run 4".to_owned()),
                },
                ProvenanceRef {
                    source: "trainer".into(),
                    note: None,
                },
            ],
        }
    }

    #[test]
    fn a_binding_round_trips_through_its_canonical_encoding() {
        let original = binding();
        let bytes = original.encode().unwrap();
        assert_eq!(StateBinding::decode(&bytes).unwrap(), original);
    }

    #[test]
    fn a_sealed_state_round_trips_and_carries_its_payload() {
        let state = NeuralState::new(binding(), vec![0xAB; 64]);
        let sealed = state.seal().unwrap();
        assert_eq!(sealed.anchor().journal, state.binding().journal);
        assert_eq!(sealed.anchor().codebook, CodebookVersion::V1);
        let reopened = NeuralState::open(sealed.bytes(), sealed.anchor()).unwrap();
        assert_eq!(reopened, state);
        assert_eq!(reopened.payload_len(), 64);
    }

    #[test]
    fn an_anchor_that_disagrees_with_the_binding_is_refused() {
        let state = NeuralState::new(binding(), vec![1, 2, 3]);
        let sealed = state.seal().unwrap();
        let mut wrong = sealed.anchor();
        wrong.journal.index = CommitIndex(5);
        assert_eq!(
            NeuralState::open(sealed.bytes(), wrong),
            Err(invalid(NeuralError::AnchorMismatch))
        );
        let mut wrong = sealed.anchor();
        wrong.codebook = CodebookVersion(9);
        assert_eq!(
            NeuralState::open(sealed.bytes(), wrong),
            Err(invalid(NeuralError::AnchorMismatch))
        );
    }

    #[test]
    fn a_reordered_section_is_not_a_second_encoding_of_the_same_binding() {
        let original = binding();
        let bytes = original.encode().unwrap();
        // Rebuild the inputs section with the two keys swapped. Both entries are
        // five bytes of header plus a 32-byte digest, so the swap is a fixed-size
        // block exchange.
        let header = BINDING_MAGIC.len() + 4 + 32 + 8 + 32 + 8 + 4;
        let entry = 4 + 5 + 32;
        let mut swapped = bytes.clone();
        swapped[header..header + entry].copy_from_slice(&bytes[header + entry..header + 2 * entry]);
        swapped[header + entry..header + 2 * entry].copy_from_slice(&bytes[header..header + entry]);
        assert_ne!(swapped, bytes);
        assert_eq!(
            StateBinding::decode(&swapped),
            Err(invalid(NeuralError::NoncanonicalSection))
        );
    }

    #[test]
    fn a_third_presence_value_is_refused() {
        let original = binding();
        let bytes = original.encode().unwrap();
        // The last provenance entry ends with its presence byte.
        let mut broken = bytes.clone();
        let last = broken.len() - 1;
        assert_eq!(broken[last], 0);
        broken[last] = 2;
        assert_eq!(
            StateBinding::decode(&broken),
            Err(invalid(NeuralError::ReservedField))
        );
    }

    #[test]
    fn neural_faults_and_denials_carry_stable_codes() {
        assert_eq!(NeuralError::SizeLimit.code(), "PTR_NEURAL_SIZE_LIMIT");
        assert_eq!(
            NeuralError::Unbindable(Denial::Fenced).code(),
            "PTR_NEURAL_UNBINDABLE"
        );
        assert_eq!(Denial::Fenced.code(), "PTR_NEURAL_FENCED");
        assert_eq!(
            Denial::UncheckableInput {
                key: "k".to_owned()
            }
            .code(),
            "PTR_NEURAL_UNCHECKABLE_INPUT"
        );
        assert_eq!(
            Denial::RevokedGeneration {
                target: "c".to_owned(),
                generation: Generation(1)
            }
            .code(),
            "PTR_NEURAL_REVOKED_GENERATION"
        );
        assert_eq!(
            Denial::UnverifiablePosition {
                bound: CommitIndex(9),
                from: CommitIndex(0),
                through: CommitIndex(4)
            }
            .to_string(),
            "PTR_NEURAL_UNVERIFIABLE_POSITION"
        );
    }
}
