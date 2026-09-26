use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use ptr_semdb::{SemanticDelta, SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, Revision, Validity};

use crate::branch::{BranchId, SealedBranch};
use crate::digest::{InputsDigest, RangeDigest, ValueDigest};
use crate::error::BranchError;
use crate::ops::BranchOp;

/// A certified branch, ready to be proposed as one ordinary semantic delta.
///
/// A plan is not a commit. It carries the revision it was certified against;
/// the runtime's `apply_verified_semantic_delta(expected, delta, required,
/// verify)` refuses it if anything has been committed since and appends it only
/// after verification of the exact state it would publish.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergePlan {
    pub branch: BranchId,
    pub expected: Revision,
    pub delta: SemanticDelta,
    /// Keys whose commutative operations were applied on top of a value that
    /// changed after the branch base.
    pub rebased: BTreeSet<String>,
    /// Digest of the branch's declared dependencies (reads, scans, relied
    /// generations) and of the input set of every key it touches. Part of
    /// [`MergePlan::digest`].
    pub dependencies: [u8; 32],
}

impl MergePlan {
    /// Whether merging would change nothing.
    pub fn is_noop(&self) -> bool {
        self.delta.upserts.is_empty() && self.delta.removals.is_empty()
    }

    /// Digest of what a person approves: the branch id (length-delimited),
    /// the expected revision, the canonical delta encoding and the dependency
    /// digest. An approval bound to this digest stands for exactly this plan
    /// of exactly this branch: a re-certification that changes the delta or
    /// the dependencies yields a different digest and voids it, and another
    /// branch proposing the same delta against the same revision with the same
    /// dependencies yields a different digest, so the approval cannot be
    /// replayed for it.
    pub fn digest(&self) -> Result<[u8; 32], BranchError> {
        let encoded = self.delta.encode().map_err(|_| BranchError::InvalidValue {
            key: "<merge delta>".into(),
        })?;
        let mut hasher = Sha256::new();
        hasher.update(b"ptr-branch/merge-plan/v2");
        hasher.update((self.branch.0.len() as u64).to_le_bytes());
        hasher.update(self.branch.0.as_bytes());
        hasher.update(self.expected.0.to_le_bytes());
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        hasher.update(self.dependencies);
        Ok(hasher.finalize().into())
    }

    /// Consume the plan into its expected revision and proposed delta.
    /// The caller must still submit these through runtime verification.
    pub fn into_parts(self) -> (Revision, SemanticDelta) {
        (self.expected, self.delta)
    }
}

/// How a branch relates to the snapshot it was certified against.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Certification {
    /// Nothing the branch depended on or touched has changed since its base.
    Clean(MergePlan),
    /// Nothing the branch depended on has changed, but keys it changes
    /// commutatively have; its operations were applied to their current
    /// values and the plan publishes the resulting absolute values.
    Rebased(MergePlan),
}

impl Certification {
    pub fn plan(&self) -> &MergePlan {
        match self {
            Self::Clean(plan) | Self::Rebased(plan) => plan,
        }
    }
}

/// Certify a sealed branch against `target`.
///
/// This is optimistic concurrency certification over the branch's declared
/// dependencies: every value it read must be unchanged (value digests), every
/// prefix it scanned must contain exactly the same keys and values (range
/// digests), every key it touches must declare the same input set it declared
/// at the base (input-set digests; a touched key with no recorded input set
/// is a conflict too), and every lifecycle target it relied on must still be
/// live at that generation according to `validity` (answered by the lifecycle
/// authority, for example `PtrRuntime::generation_validity`). The input-set
/// check matters because a merge publishes a touched key's value but keeps
/// the target's dependency set for it: a concurrent delta that rewires a
/// derived key to other inputs while recomputing it to the same value would
/// otherwise leave the branch's value standing under inputs it was never
/// computed from. Anything else is
/// refused and nothing merges; re-running the agent on the new snapshot is the
/// only repair, because only the agent knows what else its reasoning used.
/// Reads the agent did not declare are invisible here — the guarantee is as
/// complete as the declaration.
pub fn certify<V>(
    branch: &SealedBranch,
    target: &SemanticSnapshot,
    validity: V,
) -> Result<Certification, BranchError>
where
    V: Fn(&str, Generation) -> Option<Validity>,
{
    if target.revision < branch.base_revision {
        return Err(BranchError::SnapshotBehindBase {
            base: branch.base_revision,
            snapshot: target.revision,
        });
    }

    let stale: BTreeSet<String> = branch
        .relied
        .iter()
        .filter(|(target, generation)| validity(target, **generation) != Some(Validity::Live))
        .map(|(target, _)| target.clone())
        .collect();
    if !stale.is_empty() {
        return Err(BranchError::LifecycleChanged { targets: stale });
    }

    let mut conflicts = BTreeSet::new();
    for (key, digest) in &branch.reads {
        if ValueDigest::of(key, target.value(key))? != *digest {
            conflicts.insert(key.clone());
        }
    }
    for (prefix, digest) in &branch.scans {
        let now = RangeDigest::of(
            prefix,
            target
                .keys()
                .filter(|key| key.starts_with(prefix.as_str()))
                .filter_map(|key| target.value(key).map(|value| (key, value))),
        )?;
        if now != *digest {
            conflicts.insert(format!("{prefix}*"));
        }
    }
    let touched: BTreeSet<&str> = branch
        .ops
        .iter()
        .map(BranchOp::key)
        .chain(branch.touched_inputs.keys().map(String::as_str))
        .collect();
    for key in touched {
        if branch.touched_inputs.get(key) != Some(&InputsDigest::of(key, target.inputs(key))) {
            conflicts.insert(key.to_owned());
        }
    }
    if !conflicts.is_empty() {
        return Err(BranchError::Conflict { keys: conflicts });
    }

    let mut finals: BTreeMap<&str, Option<SemanticValue>> = BTreeMap::new();
    for op in &branch.ops {
        let current = match finals.remove(op.key()) {
            Some(value) => value,
            None => target.value(op.key()).cloned(),
        };
        finals.insert(op.key(), op.apply(current)?);
    }

    let mut delta = SemanticDelta::default();
    for (key, value) in finals {
        match (value, target.value(key)) {
            (Some(next), Some(current)) if &next == current => {}
            (Some(next), _) => {
                delta.upserts.insert(key.to_owned(), next);
            }
            (None, Some(_)) => {
                delta.removals.insert(key.to_owned());
            }
            (None, None) => {}
        }
    }

    let mut rebased = BTreeSet::new();
    for op in branch.ops.iter().filter(|op| op.commutes()) {
        let key = op.key();
        if let Some(base) = branch.touched_base.get(key) {
            if ValueDigest::of(key, target.value(key))? != *base {
                rebased.insert(key.to_owned());
            }
        }
    }

    let plan = MergePlan {
        branch: branch.id.clone(),
        expected: target.revision,
        delta,
        rebased,
        dependencies: dependency_digest(branch),
    };
    if plan.rebased.is_empty() {
        Ok(Certification::Clean(plan))
    } else {
        Ok(Certification::Rebased(plan))
    }
}

fn dependency_digest(branch: &SealedBranch) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-branch/dependencies/v2");
    hasher.update(branch.base_revision.0.to_le_bytes());
    for (key, digest) in &branch.reads {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.update([0xff]);
    for (prefix, digest) in &branch.scans {
        hasher.update((prefix.len() as u64).to_le_bytes());
        hasher.update(prefix.as_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.update([0xfe]);
    for (target, generation) in &branch.relied {
        hasher.update((target.len() as u64).to_le_bytes());
        hasher.update(target.as_bytes());
        hasher.update(generation.0.to_le_bytes());
    }
    hasher.update([0xfd]);
    for (key, digest) in &branch.touched_inputs {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.finalize().into()
}
