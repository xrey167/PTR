use std::collections::BTreeMap;
use std::fmt;

use ptr_semdb::{SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, PrincipalId, Revision};

use crate::digest::{RangeDigest, ValueDigest};
use crate::error::BranchError;
use crate::ops::BranchOp;

/// Key prefixes only ingress writes: raw request text and Pod outputs. A branch
/// may read them but never change them.
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

/// A branch that accepts no further operations. It holds digests rather than
/// the snapshot, so it can be stored and certified later against any snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedBranch {
    pub id: BranchId,
    pub author: PrincipalId,
    pub base_revision: Revision,
    /// Digest of each value the branch read, as of its base.
    pub reads: BTreeMap<String, ValueDigest>,
    /// Digest of each prefix the branch scanned, as of its base.
    pub scans: BTreeMap<String, RangeDigest>,
    /// Lifecycle targets (capsules, `constraint:<key>`, `procedure:<id>`) and
    /// the generation the branch relied on.
    pub relied: BTreeMap<String, Generation>,
    /// Digest of each key an operation touches, as of the base.
    pub touched_base: BTreeMap<String, ValueDigest>,
    pub ops: Vec<BranchOp>,
}

impl Branch {
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
    pub fn rely_on(&mut self, target: &str, generation: Generation) {
        self.relied.insert(target.to_owned(), generation);
    }

    /// Overwrite a key the branch has read. If the key is derived, its declared
    /// inputs are read too: an upsert of a derived value is only sound while
    /// the inputs it was computed from are unchanged.
    pub fn put(&mut self, key: &str, value: SemanticValue) -> Result<(), BranchError> {
        self.check_writable(key)?;
        self.require_read(key)?;
        self.read_inputs_of(key)?;
        self.ops.push(BranchOp::Put {
            key: key.to_owned(),
            value,
        });
        Ok(())
    }

    /// Remove a key the branch has read.
    pub fn remove(&mut self, key: &str) -> Result<(), BranchError> {
        self.check_writable(key)?;
        self.require_read(key)?;
        self.read_inputs_of(key)?;
        self.ops.push(BranchOp::Remove {
            key: key.to_owned(),
        });
        Ok(())
    }

    /// Stage a commutative operation. It is checked against the branch's own
    /// view now, and applied to whatever the key holds at merge time later.
    pub fn stage_commutative(&mut self, op: BranchOp) -> Result<(), BranchError> {
        self.check_writable(op.key())?;
        if !op.commutes() {
            return Err(BranchError::UnreadTarget {
                key: op.key().to_owned(),
            });
        }
        self.read_inputs_of(op.key())?;
        op.apply(self.overlay(op.key())?)?;
        self.ops.push(op);
        Ok(())
    }

    pub fn seal(self) -> Result<SealedBranch, BranchError> {
        let mut touched_base = BTreeMap::new();
        for op in &self.ops {
            let key = op.key();
            if !touched_base.contains_key(key) {
                touched_base.insert(key.to_owned(), ValueDigest::of(key, self.base.value(key))?);
            }
        }
        Ok(SealedBranch {
            id: self.id,
            author: self.author,
            base_revision: self.base.revision,
            reads: self.reads,
            scans: self.scans,
            relied: self.relied,
            touched_base,
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

    fn read_inputs_of(&mut self, key: &str) -> Result<(), BranchError> {
        let inputs: Vec<String> = self.base.inputs(key).map(str::to_owned).collect();
        for input in inputs {
            self.record_read(&input)?;
        }
        Ok(())
    }

    fn check_writable(&self, key: &str) -> Result<(), BranchError> {
        if RESERVED_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
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
