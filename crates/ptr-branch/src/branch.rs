use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ptr_semdb::{canonical_input_bytes, SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, PrincipalId, Revision};
use sha2::{Digest, Sha256};

use crate::digest::{InputsDigest, RangeDigest, ValueDigest};
use crate::error::BranchError;
use crate::ops::{holds_member, BranchOp};

/// Key prefixes only ingress writes: raw request text and Pod outputs. A branch
/// may read them but never change them: staging refuses an operation on one,
/// and so do [`SealedBranch::from_parts`] and certification, so no sealed
/// branch however built writes one. This is `ptr_semdb::INGRESS_PREFIXES`
/// itself, so the branch checks and the runtime's cannot disagree about which
/// keys are reserved.
pub use ptr_semdb::INGRESS_PREFIXES as RESERVED_PREFIXES;

/// Domain tag of [`SealedBranch::seal_digest`].
const SEAL_DOMAIN: &[u8] = b"ptr-branch/sealed-branch/v1";

/// Identity of one speculative branch.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BranchId(pub String);

impl From<&str> for BranchId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for BranchId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// An open branch: an agent's private, speculative view over one immutable
/// semantic snapshot.
///
/// Nothing staged here is visible to anyone else or has any authority. The
/// branch declares what its conclusions depend on, the way a neural-state
/// declaration does: a digest of every value it read, a digest of every prefix
/// it scanned, and every lifecycle target it relied on at a generation.
/// Certification later decides whether those dependencies still hold.
#[derive(Clone, Debug)]
pub struct Branch {
    id: BranchId,
    author: PrincipalId,
    base: SemanticSnapshot,
    reads: BTreeMap<String, ValueDigest>,
    scans: BTreeMap<String, RangeDigest>,
    relied: BTreeMap<String, Generation>,
    ops: Vec<BranchOp>,
}

/// The parts of a sealed branch: what [`SealedBranch::into_parts`] returns
/// and [`SealedBranch::from_parts`] validates, for example when storage
/// rebuilds a branch from its rows.
///
/// Holding parts grants nothing. Only a [`SealedBranch`] can be certified or
/// stored, and one exists only once its parts pass every sealing invariant,
/// so parts edited by hand or read back from a tampered store are refused
/// rather than certified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedBranchParts {
    pub id: BranchId,
    pub author: PrincipalId,
    pub base_revision: Revision,
    /// Digest of each value the branch read, as of its base.
    pub reads: BTreeMap<String, ValueDigest>,
    /// Digest of each prefix the branch scanned, as of its base.
    pub scans: BTreeMap<String, RangeDigest>,
    /// Lifecycle targets (capsules, `constraint:<key>`, `procedure:<id>`) and
    /// the generation the branch relied on. A map holds one generation per
    /// target, so the rule [`Branch::rely_on`] enforces
    /// ([`BranchError::ConflictingReliance`]) holds by type; a store that
    /// finds two generations of one target must refuse them with that error
    /// rather than keep either.
    pub relied: BTreeMap<String, Generation>,
    /// Digest of each key an operation touches, as of the base.
    pub touched_base: BTreeMap<String, ValueDigest>,
    /// Digest of the input set each key an operation touches declared at the
    /// base, including the digest of an empty set for a key with no inputs.
    /// A merge keeps the target's dependency set for a touched key, so
    /// certification requires it to be the set the branch computed against.
    pub touched_inputs: BTreeMap<String, InputsDigest>,
    /// The staged operations in staging order. A set operation carries
    /// whether its member was in the base's set (`in_base`), which
    /// certification compares with the target's.
    pub ops: Vec<BranchOp>,
}

/// A branch that accepts no further operations. It holds digests rather than
/// the snapshot, so it can be stored and certified later against any snapshot.
///
/// # Guarantees
/// Its fields are private and it is built only by [`Branch::seal`] and
/// [`SealedBranch::from_parts`], both of which check every sealing invariant
/// ([`SealedBranch::from_parts`] lists them), so every `SealedBranch` in
/// existence could have been sealed from an open branch: none writes a
/// namespace reserved to ingress ([`RESERVED_PREFIXES`]), overwrites a key it
/// did not read, removes a key whose recorded input set is not empty, lacks
/// the base value or input set certification checks for a key it touches, or
/// records two base presences for one set member.
/// [`certify`](crate::certify) checks them once more.
///
/// No field can be reached to change a built branch:
///
/// ```compile_fail
/// use ptr_branch::{BranchOp, SealedBranch};
/// fn forge(mut branch: SealedBranch) -> SealedBranch {
///     branch.ops.push(BranchOp::Remove { key: "request:r1:raw".into() });
///     branch
/// }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedBranch {
    parts: SealedBranchParts,
}

impl SealedBranch {
    /// Build a sealed branch from its parts, for example ones storage read
    /// back, refusing any that sealing an open branch could not have
    /// produced. Checked, in this order, for each operation in turn and then
    /// for the recorded digests:
    ///
    /// - no operation's key is in a namespace reserved to ingress
    ///   ([`BranchError::ReservedNamespace`]);
    /// - every `Put` and `Remove` names a key in `reads`
    ///   ([`BranchError::UnreadTarget`]): merging an overwrite of an unread
    ///   key would erase a concurrent change unseen;
    /// - no set operation has an empty member ([`BranchError::InvalidMember`]);
    /// - every `Put` value is one the journal can encode
    ///   ([`ptr_semdb::canonical_input_bytes`] accepts it;
    ///   [`BranchError::InvalidValue`]): a value it refuses could never be
    ///   merged, and the branch would have no [`seal_digest`](Self::seal_digest);
    /// - every key an operation touches has a base value in `touched_base`
    ///   and an input set in `touched_inputs` ([`BranchError::MalformedSeal`]);
    /// - every set operation on a member records the same `in_base` as the
    ///   earlier ones on that member, since all describe one base, and none
    ///   records its member present where the key's recorded base value is
    ///   the digest of absence ([`BranchError::MalformedSeal`]);
    /// - neither `touched_base` nor `touched_inputs` has an entry for a key
    ///   no operation touches ([`BranchError::MalformedSeal`]);
    /// - a touched key the branch also read has the same digest in
    ///   `touched_base` as in `reads`, since both digest its base value
    ///   ([`BranchError::MalformedSeal`]);
    /// - last, once every rule above holds: no `Remove` names a key whose
    ///   recorded input set is not the empty set
    ///   ([`BranchError::DerivedRemoval`], naming the first such key in
    ///   operation order): the merged removal would drop the key's
    ///   dependency entry. Sealing produced such removals before it refused
    ///   them, and broke no other rule, so reporting this one only when it
    ///   is the only one broken lets a store tell such a branch, to be
    ///   re-run, from parts no sealing produced.
    ///
    /// `relied` holds one generation per target by type (see
    /// [`SealedBranchParts::relied`]). Digests are fixed-size values and are
    /// not recomputed here: that a read, scan or input set still matches is
    /// what [`certify`](crate::certify) decides against a target snapshot.
    /// Likewise an `in_base` that passes these checks is taken as declared:
    /// whether a member was in a base set that was present is hidden behind
    /// the base value's digest.
    ///
    /// # Errors
    /// The first refusal above; nothing is built.
    pub fn from_parts(parts: SealedBranchParts) -> Result<Self, BranchError> {
        check_sealed(&parts)?;
        Ok(Self { parts })
    }

    /// Give up the sealed branch for its parts. Building one again goes
    /// through [`SealedBranch::from_parts`].
    pub fn into_parts(self) -> SealedBranchParts {
        self.parts
    }

    /// Check every sealing invariant again. Every `SealedBranch` passed them
    /// when it was built and none can be changed since, so this returns
    /// `Ok(())`; certification and storage call it anyway, before anything is
    /// planned or written, so a defect in how a branch was built is refused
    /// at that boundary rather than trusted.
    ///
    /// # Errors
    /// What [`SealedBranch::from_parts`] would refuse.
    pub fn recheck(&self) -> Result<(), BranchError> {
        check_sealed(&self.parts)
    }

    pub fn id(&self) -> &BranchId {
        &self.parts.id
    }

    pub fn author(&self) -> &PrincipalId {
        &self.parts.author
    }

    pub fn base_revision(&self) -> Revision {
        self.parts.base_revision
    }

    /// Digest of each value the branch read, as of its base.
    pub fn reads(&self) -> &BTreeMap<String, ValueDigest> {
        &self.parts.reads
    }

    /// Digest of each prefix the branch scanned, as of its base.
    pub fn scans(&self) -> &BTreeMap<String, RangeDigest> {
        &self.parts.scans
    }

    /// Each lifecycle target the branch relied on and the one generation it
    /// relied on.
    pub fn relied(&self) -> &BTreeMap<String, Generation> {
        &self.parts.relied
    }

    /// Digest of each key an operation touches, as of the base: exactly the
    /// keys of [`SealedBranch::ops`].
    pub fn touched_base(&self) -> &BTreeMap<String, ValueDigest> {
        &self.parts.touched_base
    }

    /// Digest of the input set each key an operation touches declared at the
    /// base: exactly the keys of [`SealedBranch::ops`].
    pub fn touched_inputs(&self) -> &BTreeMap<String, InputsDigest> {
        &self.parts.touched_inputs
    }

    /// The staged operations, in the order they were staged.
    pub fn ops(&self) -> &[BranchOp] {
        &self.parts.ops
    }

    /// Digest of every part of the branch: what a record of its merge names
    /// as the branch that was merged, and what storage can bind a stored
    /// branch to. Two sealed branches have the same digest only if they have
    /// the same parts.
    ///
    /// SHA-256 under the domain tag `ptr-branch/sealed-branch/v1` of, in
    /// this order and each length or count a little-endian `u64`:
    ///
    /// - the id and the author, each preceded by its length, and the base
    ///   revision;
    /// - `reads`, `scans`, `relied`, `touched_base` and `touched_inputs`,
    ///   each as its entry count and then each entry in key order: the key
    ///   preceded by its length, then its 32 digest bytes or, for `relied`,
    ///   the generation;
    /// - the operations, as their count and then each in staging order: a
    ///   kind byte (1 `Put`, 2 `Remove`, 3 `Add`, 4 `SetInsert`,
    ///   5 `SetRemove`), the key preceded by its length, and then for a
    ///   `Put` the canonical bytes of the key and its value
    ///   ([`ptr_semdb::canonical_input_bytes`]) preceded by their length,
    ///   for an `Add` the amount as a little-endian `i64`, and for a set
    ///   operation the member preceded by its length and one `in_base` byte.
    ///
    /// Counting every section keeps a digest moved from one map to another,
    /// or an entry moved across an operation boundary, from hashing as the
    /// same bytes.
    ///
    /// # Errors
    /// `BranchError::InvalidValue` for a `Put` whose value the journal cannot
    /// encode, which no `SealedBranch` holds: [`SealedBranch::from_parts`]
    /// and sealing refuse it.
    pub fn seal_digest(&self) -> Result<[u8; 32], BranchError> {
        fn text(hasher: &mut Sha256, value: &str) {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        fn digests<'a>(
            hasher: &mut Sha256,
            entries: impl ExactSizeIterator<Item = (&'a String, &'a [u8; 32])>,
        ) {
            hasher.update((entries.len() as u64).to_le_bytes());
            for (key, digest) in entries {
                text(hasher, key);
                hasher.update(digest);
            }
        }

        let parts = &self.parts;
        let mut hasher = Sha256::new();
        hasher.update(SEAL_DOMAIN);
        text(&mut hasher, &parts.id.0);
        text(&mut hasher, &parts.author.0);
        hasher.update(parts.base_revision.0.to_le_bytes());
        digests(
            &mut hasher,
            parts
                .reads
                .iter()
                .map(|(key, digest)| (key, digest.as_bytes())),
        );
        digests(
            &mut hasher,
            parts
                .scans
                .iter()
                .map(|(key, digest)| (key, digest.as_bytes())),
        );
        hasher.update((parts.relied.len() as u64).to_le_bytes());
        for (target, generation) in &parts.relied {
            text(&mut hasher, target);
            hasher.update(generation.0.to_le_bytes());
        }
        digests(
            &mut hasher,
            parts
                .touched_base
                .iter()
                .map(|(key, digest)| (key, digest.as_bytes())),
        );
        digests(
            &mut hasher,
            parts
                .touched_inputs
                .iter()
                .map(|(key, digest)| (key, digest.as_bytes())),
        );
        hasher.update((parts.ops.len() as u64).to_le_bytes());
        for op in &parts.ops {
            match op {
                BranchOp::Put { key, value } => {
                    hasher.update([1u8]);
                    text(&mut hasher, key);
                    let canonical = canonical_input_bytes(key, value)
                        .map_err(|_| BranchError::InvalidValue { key: key.clone() })?;
                    hasher.update((canonical.len() as u64).to_le_bytes());
                    hasher.update(&canonical);
                }
                BranchOp::Remove { key } => {
                    hasher.update([2u8]);
                    text(&mut hasher, key);
                }
                BranchOp::Add { key, amount } => {
                    hasher.update([3u8]);
                    text(&mut hasher, key);
                    hasher.update(amount.to_le_bytes());
                }
                BranchOp::SetInsert {
                    key,
                    member,
                    in_base,
                } => {
                    hasher.update([4u8]);
                    text(&mut hasher, key);
                    text(&mut hasher, member);
                    hasher.update([u8::from(*in_base)]);
                }
                BranchOp::SetRemove {
                    key,
                    member,
                    in_base,
                } => {
                    hasher.update([5u8]);
                    text(&mut hasher, key);
                    text(&mut hasher, member);
                    hasher.update([u8::from(*in_base)]);
                }
            }
        }
        Ok(hasher.finalize().into())
    }

    /// A sealed branch built without any check, so tests can show that
    /// certification refuses one on its own.
    #[cfg(test)]
    pub(crate) fn unchecked(parts: SealedBranchParts) -> Self {
        Self { parts }
    }
}

/// Whether `key` is in a namespace only ingress writes.
pub(crate) fn is_reserved(key: &str) -> bool {
    ptr_semdb::is_ingress_key(key)
}

/// The sealing invariants [`SealedBranch::from_parts`] documents, in its
/// order.
pub(crate) fn check_sealed(parts: &SealedBranchParts) -> Result<(), BranchError> {
    let mut members: BTreeMap<(&str, &str), bool> = BTreeMap::new();
    // Reported only once every other invariant holds.
    let mut derived_removal: Option<&str> = None;
    for op in &parts.ops {
        let key = op.key();
        if is_reserved(key) {
            return Err(BranchError::ReservedNamespace {
                key: key.to_owned(),
            });
        }
        if !op.commutes() && !parts.reads.contains_key(key) {
            return Err(BranchError::UnreadTarget {
                key: key.to_owned(),
            });
        }
        if let BranchOp::SetInsert { member, .. } | BranchOp::SetRemove { member, .. } = op {
            if member.is_empty() {
                return Err(BranchError::InvalidMember {
                    key: key.to_owned(),
                });
            }
        }
        if let BranchOp::Put { value, .. } = op {
            if canonical_input_bytes(key, value).is_err() {
                return Err(BranchError::InvalidValue {
                    key: key.to_owned(),
                });
            }
        }
        if !parts.touched_base.contains_key(key) {
            return Err(BranchError::MalformedSeal {
                key: key.to_owned(),
                reason: "a key an operation touches has no recorded base value",
            });
        }
        let Some(inputs) = parts.touched_inputs.get(key) else {
            return Err(BranchError::MalformedSeal {
                key: key.to_owned(),
                reason: "a key an operation touches has no recorded input set",
            });
        };
        if matches!(op, BranchOp::Remove { .. }) && *inputs != InputsDigest::of(key, []) {
            derived_removal.get_or_insert(key);
        }
        if let Some((member, _, in_base)) = op.set_member() {
            if members
                .insert((key, member), in_base)
                .is_some_and(|earlier| earlier != in_base)
            {
                return Err(BranchError::MalformedSeal {
                    key: key.to_owned(),
                    reason: "two set operations on one member record different base presences",
                });
            }
            if in_base && parts.touched_base.get(key) == Some(&ValueDigest::absent(key)) {
                return Err(BranchError::MalformedSeal {
                    key: key.to_owned(),
                    reason: "a set operation records its member present in a base without its key",
                });
            }
        }
    }
    let operated: BTreeSet<&str> = parts.ops.iter().map(BranchOp::key).collect();
    if let Some(key) = parts
        .touched_base
        .keys()
        .chain(parts.touched_inputs.keys())
        .find(|key| !operated.contains(key.as_str()))
    {
        return Err(BranchError::MalformedSeal {
            key: key.clone(),
            reason: "a base value or input set is recorded for a key no operation touches",
        });
    }
    if let Some((key, _)) = parts
        .touched_base
        .iter()
        .find(|(key, base)| parts.reads.get(*key).is_some_and(|read| read != *base))
    {
        return Err(BranchError::MalformedSeal {
            key: key.clone(),
            reason: "the base value recorded for a touched key differs from the value read",
        });
    }
    if let Some(key) = derived_removal {
        return Err(BranchError::DerivedRemoval {
            key: key.to_owned(),
        });
    }
    Ok(())
}

impl Branch {
    /// Start an empty speculative branch over `base`, attributed to `author`.
    /// Reads, lifecycle dependencies, and operations are recorded as they occur.
    ///
    /// `author` is recorded as given: nothing here checks that it is the
    /// principal the caller's execution session admitted, so passing that
    /// one is the caller's obligation.
    pub fn open(id: BranchId, author: PrincipalId, base: SemanticSnapshot) -> Self {
        Self {
            id,
            author,
            base,
            reads: BTreeMap::new(),
            scans: BTreeMap::new(),
            relied: BTreeMap::new(),
            ops: Vec::new(),
        }
    }

    pub fn id(&self) -> &BranchId {
        &self.id
    }

    pub fn base_revision(&self) -> Revision {
        self.base.revision
    }

    /// Read a key through the branch: what merging this branch onto its
    /// unchanged base would leave at `key`. That is the base value with the
    /// branch's own operations on `key` applied, except that a key with no
    /// operation of its own is absent once the branch changes a key it is
    /// derived from, directly or through other derived keys in the base's
    /// dependency graph: the merge evicts it, as every commit evicts a
    /// derived value whose inputs change and that it does not write. A key
    /// the branch writes keeps the written value, which the merge publishes
    /// even when it equals the base value; if the branch also removes a key
    /// that one needs, the runtime refuses the merge (`MissingDependency`)
    /// rather than publish it. A key the branch changes only commutatively
    /// is never a derived key that reads as absent: staging refuses such an
    /// operation on one, whether the branch's own change evicts it or the
    /// base holds no value for it, and a change to an input of a key that
    /// has one ([`BranchError::EvictedOperand`]), so its value here is its
    /// base value with those operations applied.
    ///
    /// The first read records the digest of the base value, whatever this
    /// returns; that digest is what certification checks. Against a target
    /// that moved since the base, what the merge leaves is decided there,
    /// not here.
    pub fn read(&mut self, key: &str) -> Result<Option<SemanticValue>, BranchError> {
        self.record_read(key)?;
        self.view(key)
    }

    /// Enumerate, in key order, every key under `prefix` that holds a value
    /// in the branch's view, with that value: exactly what [`Branch::read`]
    /// returns for each key. A key the branch inserts appears, one it
    /// removes does not, one it changes shows the changed value, and a
    /// derived key the branch's own change evicts is left out, as a read of
    /// it shows it absent.
    ///
    /// The first scan of a prefix records the range digest of every key the
    /// base holds under it with its base value, whatever this returns, so a
    /// key later inserted or removed under the prefix is a conflict even
    /// though no point read covers it; that digest is what certification
    /// checks. A scan records no read of a key: a `Put` or `Remove` of a
    /// scanned key still needs [`Branch::read`] first. A scan that fails
    /// records nothing.
    ///
    /// # Errors
    /// Returns `BranchError::InvalidValue` if a base value under the prefix
    /// cannot be encoded for the range digest.
    pub fn scan_prefix(
        &mut self,
        prefix: &str,
    ) -> Result<Vec<(String, SemanticValue)>, BranchError> {
        let digest = match self.scans.get(prefix) {
            Some(digest) => *digest,
            None => RangeDigest::of(
                prefix,
                self.base
                    .keys()
                    .filter(|key| key.starts_with(prefix))
                    .filter_map(|key| Some((key, self.base.value(key)?))),
            )?,
        };
        let keys: BTreeSet<&str> = self
            .base
            .keys()
            .chain(self.ops.iter().map(BranchOp::key))
            .filter(|key| key.starts_with(prefix))
            .collect();
        let mut entries = Vec::new();
        for key in keys {
            if let Some(value) = self.view(key)? {
                entries.push((key.to_owned(), value));
            }
        }
        self.scans.entry(prefix.to_owned()).or_insert(digest);
        Ok(entries)
    }

    /// Declare that the branch's conclusions depend on `target` being live at
    /// `generation`. Certification refuses the branch once it is not.
    /// Declaring the same generation again changes nothing.
    ///
    /// # Errors
    /// Returns `BranchError::ConflictingReliance` when the branch already
    /// relied on another generation of `target`, and keeps the first: at most
    /// one generation of a target is live at a time, so conclusions resting on
    /// both could never be certified, and keeping only the later one would
    /// certify the branch although what it concluded from the earlier one is
    /// superseded.
    pub fn rely_on(&mut self, target: &str, generation: Generation) -> Result<(), BranchError> {
        match self.relied.get(target) {
            Some(&relied) if relied != generation => Err(BranchError::ConflictingReliance {
                target: target.to_owned(),
                relied,
                declared: generation,
            }),
            Some(_) => Ok(()),
            None => {
                self.relied.insert(target.to_owned(), generation);
                Ok(())
            }
        }
    }

    /// Overwrite a key the branch has read. If the key is derived, its declared
    /// inputs are read too: an upsert of a derived value is only sound while
    /// the inputs it was computed from are unchanged.
    ///
    /// A refused operation records nothing: neither the operation nor the
    /// reads of its inputs.
    ///
    /// # Errors
    /// Returns `BranchError::ReservedNamespace` for a key reserved to
    /// ingress, `BranchError::UnreadTarget` for a key the branch has not
    /// read, `BranchError::InvalidValue` for a value the journal cannot
    /// encode ([`ptr_semdb::canonical_input_bytes`] refuses it), which no
    /// sealed branch holds, and `BranchError::EvictedOperand` when the new
    /// value would change an input, direct or transitive, of a key the
    /// branch has changed only commutatively: those operations would then
    /// build on a value the merge evicts. Putting that key's recomputed value
    /// first lifts the refusal.
    pub fn put(&mut self, key: &str, value: SemanticValue) -> Result<(), BranchError> {
        self.check_writable(key)?;
        self.require_read(key)?;
        if canonical_input_bytes(key, &value).is_err() {
            return Err(BranchError::InvalidValue {
                key: key.to_owned(),
            });
        }
        let inputs = self.unread_inputs_of(key)?;
        self.record_op(BranchOp::Put {
            key: key.to_owned(),
            value,
        })?;
        self.reads.extend(inputs);
        Ok(())
    }

    /// Remove a key the branch has read. A refused removal records nothing.
    ///
    /// # Errors
    /// Returns `BranchError::ReservedNamespace` for a key reserved to
    /// ingress, `BranchError::UnreadTarget` for a key the branch has not
    /// read, and `BranchError::DerivedRemoval` for a key that is derived in
    /// the base (its input set is not empty): a merged removal drops the
    /// key's dependency entry with its value, and a branch may change values
    /// but not the dependency graph. Returns `BranchError::EvictedOperand`
    /// when the removal would change an input of a key the branch has
    /// changed only commutatively, as [`Branch::put`] does.
    pub fn remove(&mut self, key: &str) -> Result<(), BranchError> {
        self.check_writable(key)?;
        self.require_read(key)?;
        if self.base.inputs(key).next().is_some() {
            return Err(BranchError::DerivedRemoval {
                key: key.to_owned(),
            });
        }
        self.record_op(BranchOp::Remove {
            key: key.to_owned(),
        })
    }

    /// Stage a commutative operation. It is checked against the branch's own
    /// view of the key's value now, and applied to whatever the key holds at
    /// merge time later.
    ///
    /// A set operation is recorded with `in_base` set to whether its member
    /// is in the base's set at the key, replacing whatever the operation
    /// carried: certification compares it with the target's.
    ///
    /// The operation is checked before anything is recorded, so a refused one
    /// (for example an addition to a key that is not a counter) leaves no
    /// read of the key's inputs behind: such a read would be a dependency of
    /// nothing the branch does, and a later change to it would refuse the
    /// branch for no reason.
    ///
    /// # Errors
    /// Returns `BranchError::ReservedNamespace` for a key reserved to
    /// ingress and `BranchError::UnreadTarget` for a `Put` or `Remove`.
    /// Returns `BranchError::EvictedOperand` for an operation on a key the
    /// branch has not put or removed and that is derived, directly or
    /// transitively, from a key the branch changes, naming that key as
    /// `input`: [`Branch::read`] shows such a key absent, since the merge
    /// evicts it, and the operation would be merged onto the value the
    /// change invalidates. Returns it with no `input` for an operation on a
    /// key the branch has not put or removed that is derived (its input set
    /// in the base is not empty) and holds no value in the base: an earlier
    /// change to its inputs evicted it, or it was never computed, so the
    /// operation would start from zero or the empty set and the merge would
    /// publish the result as the key's derived value. Both refusals come
    /// before the value's type is checked. Returns it too when the
    /// operation would change an input of another key the branch has
    /// changed only commutatively. Otherwise returns what applying the
    /// operation to the key's value in the branch's view refuses
    /// (`NotACounter`, `NotASet`, `InvalidMember`, `CounterOverflow`,
    /// `InvalidValue`).
    pub fn stage_commutative(&mut self, op: BranchOp) -> Result<(), BranchError> {
        self.check_writable(op.key())?;
        if !op.commutes() {
            return Err(BranchError::UnreadTarget {
                key: op.key().to_owned(),
            });
        }
        let op = self.with_base_membership(op);
        if !self.overwrites(op.key()) {
            if let Some(input) = self.changed_input_of(op.key())? {
                return Err(BranchError::EvictedOperand {
                    key: op.key().to_owned(),
                    input: Some(input.to_owned()),
                });
            }
            if self.base.value(op.key()).is_none() && self.base.inputs(op.key()).next().is_some() {
                return Err(BranchError::EvictedOperand {
                    key: op.key().to_owned(),
                    input: None,
                });
            }
        }
        op.apply(self.overlay(op.key())?)?;
        let inputs = self.unread_inputs_of(op.key())?;
        self.record_op(op)?;
        self.reads.extend(inputs);
        Ok(())
    }

    /// Consume the branch and retain its operations and dependency digests
    /// for later certification, including the base value and the base input
    /// set of every touched key.
    ///
    /// The result is built by [`SealedBranch::from_parts`], so it passes the
    /// same invariants a branch rebuilt from storage must; staging already
    /// keeps every one of them, so that check refuses nothing here.
    ///
    /// # Errors
    /// Returns `BranchError::InvalidValue` if a touched base value cannot be
    /// encoded for its digest; sealing does not certify or commit the branch.
    pub fn seal(self) -> Result<SealedBranch, BranchError> {
        let mut touched_base = BTreeMap::new();
        let mut touched_inputs = BTreeMap::new();
        for op in &self.ops {
            let key = op.key();
            if !touched_base.contains_key(key) {
                touched_base.insert(key.to_owned(), ValueDigest::of(key, self.base.value(key))?);
                touched_inputs.insert(key.to_owned(), InputsDigest::of(key, self.base.inputs(key)));
            }
        }
        SealedBranch::from_parts(SealedBranchParts {
            id: self.id,
            author: self.author,
            base_revision: self.base.revision,
            reads: self.reads,
            scans: self.scans,
            relied: self.relied,
            touched_base,
            touched_inputs,
            ops: self.ops,
        })
    }

    fn record_read(&mut self, key: &str) -> Result<(), BranchError> {
        if !self.reads.contains_key(key) {
            let digest = ValueDigest::of(key, self.base.value(key))?;
            self.reads.insert(key.to_owned(), digest);
        }
        Ok(())
    }

    /// Base digests of the declared inputs of `key` the branch has not read
    /// yet, computed without recording them, so a caller records all or none.
    fn unread_inputs_of(&self, key: &str) -> Result<Vec<(String, ValueDigest)>, BranchError> {
        self.base
            .inputs(key)
            .filter(|input| !self.reads.contains_key(*input))
            .map(|input| {
                Ok((
                    input.to_owned(),
                    ValueDigest::of(input, self.base.value(input))?,
                ))
            })
            .collect()
    }

    fn check_writable(&self, key: &str) -> Result<(), BranchError> {
        if is_reserved(key) {
            return Err(BranchError::ReservedNamespace {
                key: key.to_owned(),
            });
        }
        Ok(())
    }

    fn require_read(&self, key: &str) -> Result<(), BranchError> {
        if self.reads.contains_key(key) {
            Ok(())
        } else {
            Err(BranchError::UnreadTarget {
                key: key.to_owned(),
            })
        }
    }

    /// `op` with a set operation's `in_base` taken from the base.
    fn with_base_membership(&self, op: BranchOp) -> BranchOp {
        match op {
            BranchOp::SetInsert { key, member, .. } => BranchOp::SetInsert {
                in_base: holds_member(self.base.value(&key), &member),
                key,
                member,
            },
            BranchOp::SetRemove { key, member, .. } => BranchOp::SetRemove {
                in_base: holds_member(self.base.value(&key), &member),
                key,
                member,
            },
            op @ (BranchOp::Put { .. } | BranchOp::Remove { .. } | BranchOp::Add { .. }) => op,
        }
    }

    fn operates_on(&self, key: &str) -> bool {
        self.ops.iter().any(|op| op.key() == key)
    }

    /// Whether the branch puts or removes `key`. The value of a key it
    /// operates on but does not overwrite is its base value with commutative
    /// operations applied.
    fn overwrites(&self, key: &str) -> bool {
        self.ops.iter().any(|op| op.key() == key && !op.commutes())
    }

    /// Append `op`, unless it would leave a key the branch changes only
    /// commutatively derived from a key the branch changes
    /// ([`BranchError::EvictedOperand`]); a refused operation is not kept.
    /// Every staging method records through here, so no staged branch holds
    /// such a key.
    fn record_op(&mut self, op: BranchOp) -> Result<(), BranchError> {
        self.ops.push(op);
        match self.evicted_operand() {
            Ok(None) => Ok(()),
            Ok(Some(refusal)) | Err(refusal) => {
                self.ops.pop();
                Err(refusal)
            }
        }
    }

    /// The first key, in key order, that the branch changes only
    /// commutatively and that is derived from a key the branch changes, as
    /// the refusal naming both.
    fn evicted_operand(&self) -> Result<Option<BranchError>, BranchError> {
        let operated: BTreeSet<&str> = self.ops.iter().map(BranchOp::key).collect();
        for key in operated {
            if self.overwrites(key) {
                continue;
            }
            if let Some(input) = self.changed_input_of(key)? {
                return Ok(Some(BranchError::EvictedOperand {
                    key: key.to_owned(),
                    input: Some(input.to_owned()),
                }));
            }
        }
        Ok(None)
    }

    /// A key the branch's own operations change the value of and that `key`
    /// is derived from, directly or transitively, in the base's dependency
    /// graph (which is acyclic), or `None`. A key whose operations leave its
    /// base value in place changes nothing, as a commit that writes the
    /// value a key already holds changes nothing.
    fn changed_input_of(&self, key: &str) -> Result<Option<&str>, BranchError> {
        let mut pending: Vec<&str> = self.base.inputs(key).collect();
        let mut seen = BTreeSet::new();
        while let Some(input) = pending.pop() {
            if !seen.insert(input) {
                continue;
            }
            if self.operates_on(input) && self.overlay(input)?.as_ref() != self.base.value(input) {
                return Ok(Some(input));
            }
            pending.extend(self.base.inputs(input));
        }
        Ok(None)
    }

    /// What `key` holds in the branch's view, which [`Branch::read`] and
    /// [`Branch::scan_prefix`] both return: absent for a key with no
    /// operation of its own that is derived from a key the branch changes,
    /// otherwise its [`overlay`](Self::overlay).
    fn view(&self, key: &str) -> Result<Option<SemanticValue>, BranchError> {
        if !self.operates_on(key) && self.changed_input_of(key)?.is_some() {
            return Ok(None);
        }
        self.overlay(key)
    }

    /// The base value of `key` with the branch's operations on `key` applied.
    fn overlay(&self, key: &str) -> Result<Option<SemanticValue>, BranchError> {
        self.ops
            .iter()
            .filter(|op| op.key() == key)
            .try_fold(self.base.value(key).cloned(), |value, op| op.apply(value))
    }
}
