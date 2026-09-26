use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ptr_semdb::{SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, PrincipalId, Revision};

use crate::digest::{InputsDigest, RangeDigest, ValueDigest};
use crate::error::BranchError;
use crate::ops::BranchOp;

/// Key prefixes only ingress writes: raw request text and Pod outputs. A branch
/// may read them but never change them: staging refuses an operation on one,
/// and so do [`SealedBranch::from_parts`] and certification, so no sealed
/// branch however built writes one.
pub const RESERVED_PREFIXES: [&str; 2] = ["request:", "pod-output:"];

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
/// did not read, or lacks the base value or input set certification checks
/// for a key it touches. [`certify`](crate::certify) checks them once more.
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
    /// - every key an operation touches has a base value in `touched_base`
    ///   and an input set in `touched_inputs`, and neither map has an entry
    ///   for a key no operation touches ([`BranchError::MalformedSeal`]);
    /// - a touched key the branch also read has the same digest in
    ///   `touched_base` as in `reads`, since both digest its base value
    ///   ([`BranchError::MalformedSeal`]).
    ///
    /// `relied` holds one generation per target by type (see
    /// [`SealedBranchParts::relied`]). Digests are fixed-size values and are
    /// not recomputed here: that a read, scan or input set still matches is
    /// what [`certify`](crate::certify) decides against a target snapshot.
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

    /// A sealed branch built without any check, so tests can show that
    /// certification refuses one on its own.
    #[cfg(test)]
    pub(crate) fn unchecked(parts: SealedBranchParts) -> Self {
        Self { parts }
    }
}

/// Whether `key` is in a namespace only ingress writes.
pub(crate) fn is_reserved(key: &str) -> bool {
    RESERVED_PREFIXES
        .iter()
        .any(|prefix| key.starts_with(prefix))
}

/// The sealing invariants [`SealedBranch::from_parts`] documents, in its
/// order.
pub(crate) fn check_sealed(parts: &SealedBranchParts) -> Result<(), BranchError> {
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
        if !parts.touched_base.contains_key(key) {
            return Err(BranchError::MalformedSeal {
                key: key.to_owned(),
                reason: "a key an operation touches has no recorded base value",
            });
        }
        if !parts.touched_inputs.contains_key(key) {
            return Err(BranchError::MalformedSeal {
                key: key.to_owned(),
                reason: "a key an operation touches has no recorded input set",
            });
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

    /// Read a key through the branch: the base value with this branch's own
    /// staged operations applied. The first read records the digest of the
    /// base value; that digest is what certification checks.
    pub fn read(&mut self, key: &str) -> Result<Option<SemanticValue>, BranchError> {
        self.record_read(key)?;
        self.overlay(key)
    }

    /// Enumerate every base key under `prefix` with its value. The scan is
    /// recorded as a range digest, so a key later inserted or removed under the
    /// prefix is a conflict even though no point read covers it.
    pub fn scan_prefix(
        &mut self,
        prefix: &str,
    ) -> Result<Vec<(String, SemanticValue)>, BranchError> {
        let entries: Vec<(String, SemanticValue)> = self
            .base
            .keys()
            .filter(|key| key.starts_with(prefix))
            .filter_map(|key| {
                self.base
                    .value(key)
                    .map(|value| (key.to_owned(), value.clone()))
            })
            .collect();
        if !self.scans.contains_key(prefix) {
            let digest = RangeDigest::of(prefix, entries.iter().map(|(k, v)| (k.as_str(), v)))?;
            self.scans.insert(prefix.to_owned(), digest);
        }
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
    pub fn put(&mut self, key: &str, value: SemanticValue) -> Result<(), BranchError> {
        self.check_writable(key)?;
        self.require_read(key)?;
        let inputs = self.unread_inputs_of(key)?;
        self.reads.extend(inputs);
        self.ops.push(BranchOp::Put {
            key: key.to_owned(),
            value,
        });
        Ok(())
    }

    /// Remove a key the branch has read. A refused removal records nothing.
    pub fn remove(&mut self, key: &str) -> Result<(), BranchError> {
        self.check_writable(key)?;
        self.require_read(key)?;
        let inputs = self.unread_inputs_of(key)?;
        self.reads.extend(inputs);
        self.ops.push(BranchOp::Remove {
            key: key.to_owned(),
        });
        Ok(())
    }

    /// Stage a commutative operation. It is checked against the branch's own
    /// view now, and applied to whatever the key holds at merge time later.
    ///
    /// The operation is checked before anything is recorded, so a refused one
    /// (for example an addition to a key that is not a counter) leaves no
    /// read of the key's inputs behind: such a read would be a dependency of
    /// nothing the branch does, and a later change to it would refuse the
    /// branch for no reason.
    pub fn stage_commutative(&mut self, op: BranchOp) -> Result<(), BranchError> {
        self.check_writable(op.key())?;
        if !op.commutes() {
            return Err(BranchError::UnreadTarget {
                key: op.key().to_owned(),
            });
        }
        op.apply(self.overlay(op.key())?)?;
        let inputs = self.unread_inputs_of(op.key())?;
        self.reads.extend(inputs);
        self.ops.push(op);
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

    fn overlay(&self, key: &str) -> Result<Option<SemanticValue>, BranchError> {
        self.ops
            .iter()
            .filter(|op| op.key() == key)
            .try_fold(self.base.value(key).cloned(), |value, op| op.apply(value))
    }
}
