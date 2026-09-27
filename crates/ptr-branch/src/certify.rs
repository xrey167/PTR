use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use ptr_semdb::{SemanticDelta, SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, Revision, Validity};

use crate::branch::{BranchId, SealedBranch};
use crate::digest::{InputsDigest, RangeDigest, ValueDigest};
use crate::error::BranchError;
use crate::ops::holds_member;

/// A certified branch, ready to be proposed as one ordinary semantic delta.
///
/// A plan is not a commit. It carries the revision it was certified against
/// and the lifecycle generations the branch relied on, and it must be
/// committed with both: [`MergePlan::into_parts`] yields exactly the
/// `expected`, `delta` and `relied` arguments of the runtime's
/// `apply_certified_semantic_delta(expected, delta, relied, required,
/// verify)`, which refuses the plan if anything has been committed since,
/// refuses it if any relied-on generation is no longer live when it would
/// append, and appends it only after verification of the exact state it
/// would publish. Nothing enforces that path by type: the runtime does not
/// depend on this crate, so a caller that drops `relied` or calls another
/// commit path is not stopped here.
///
/// Its fields are private and only [`certify`] builds one, so nothing
/// [`MergePlan::digest`] covers can change between certification and
/// approval:
///
/// ```compile_fail
/// fn widen(mut plan: ptr_branch::MergePlan) -> ptr_branch::MergePlan {
///     plan.rebased.clear();
///     plan
/// }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergePlan {
    branch: BranchId,
    expected: Revision,
    delta: SemanticDelta,
    rebased: BTreeSet<String>,
    relied: BTreeMap<String, Generation>,
    dependencies: [u8; 32],
}

impl MergePlan {
    /// The branch this plan merges.
    pub fn branch(&self) -> &BranchId {
        &self.branch
    }

    /// The revision the plan was certified against; the runtime refuses it
    /// at any other.
    pub fn expected(&self) -> Revision {
        self.expected
    }

    /// The delta the plan would commit.
    pub fn delta(&self) -> &SemanticDelta {
        &self.delta
    }

    /// Keys whose commutative operations were applied on top of a value that
    /// changed after the branch base.
    pub fn rebased(&self) -> &BTreeSet<String> {
        &self.rebased
    }

    /// Each lifecycle target the branch relied on and the generation it
    /// relied on, all live when the plan was certified. The runtime must
    /// check them again when it commits the plan.
    pub fn relied(&self) -> &BTreeMap<String, Generation> {
        &self.relied
    }

    /// Digest of the branch's declared dependencies (base revision, reads,
    /// scans, relied generations, the input set of every key it touches and
    /// the base presence its set operations record). Part of
    /// [`MergePlan::digest`].
    pub fn dependencies(&self) -> &[u8; 32] {
        &self.dependencies
    }

    /// Whether merging would change nothing.
    pub fn is_noop(&self) -> bool {
        self.delta.upserts.is_empty() && self.delta.removals.is_empty()
    }

    /// Digest of what a person approves: the branch id, the expected
    /// revision, the canonical delta encoding, the dependency digest and the
    /// rebased keys in ascending order, under a domain tag, with every
    /// variable-length part length-delimited and the rebased keys counted.
    /// An approval bound to this digest stands for exactly this plan of
    /// exactly this branch: a re-certification that changes the delta, the
    /// dependencies or which keys were rebased yields a different digest and
    /// voids it, and another branch proposing the same delta against the
    /// same revision with the same dependencies yields a different digest, so
    /// the approval cannot be replayed for it.
    ///
    /// # Errors
    /// `BranchError::InvalidValue` (key `<merge delta>`) when the delta has
    /// no journal encoding, for example because it is longer than the
    /// journal accepts; such a plan could not be committed either.
    pub fn digest(&self) -> Result<[u8; 32], BranchError> {
        let encoded = self.delta.encode().map_err(|_| BranchError::InvalidValue {
            key: "<merge delta>".into(),
        })?;
        let mut hasher = Sha256::new();
        hasher.update(b"ptr-branch/merge-plan/v3");
        hasher.update((self.branch.0.len() as u64).to_le_bytes());
        hasher.update(self.branch.0.as_bytes());
        hasher.update(self.expected.0.to_le_bytes());
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        hasher.update(self.dependencies);
        hasher.update((self.rebased.len() as u64).to_le_bytes());
        for key in &self.rebased {
            hasher.update((key.len() as u64).to_le_bytes());
            hasher.update(key.as_bytes());
        }
        Ok(hasher.finalize().into())
    }

    /// Consume the plan into the expected revision, the delta and the relied
    /// generations: the `expected`, `delta` and `relied` arguments of the
    /// runtime's `apply_certified_semantic_delta`, which checks every relied
    /// generation again immediately before it appends. Committing the delta
    /// any other way skips that check.
    pub fn into_parts(self) -> (Revision, SemanticDelta, BTreeMap<String, Generation>) {
        (self.expected, self.delta, self.relied)
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
/// at the base (input-set digests), and every lifecycle target it relied on
/// must still be live at that generation according to `validity` (answered
/// by the lifecycle authority, for example
/// `PtrRuntime::generation_validity`). The input-set
/// check matters because a merge publishes a touched key's value but keeps
/// the target's dependency set for it (a branch cannot remove a derived key,
/// [`BranchError::DerivedRemoval`]): a concurrent delta that rewires a
/// derived key to other inputs while recomputing it to the same value would
/// otherwise leave the branch's value standing under inputs it was never
/// computed from. Anything else is
/// refused and nothing merges; re-running the agent on the new snapshot is the
/// only repair, because only the agent knows what else its reasoning used.
/// Reads the agent did not declare are invisible here — the guarantee is as
/// complete as the declaration.
///
/// Set operations are rebased member by member. The last one on a member
/// decides whether it is in the set after the merge; when that is how the
/// branch's base had it (`in_base`), the branch's operations change nothing
/// about the member in its own view, and a target that has the member the
/// other way got there through a concurrent insert or removal, which merging
/// would undo. That is a [`BranchError::Conflict`] on the set's key. When
/// the last operation moves the member away from its base presence, the
/// merge leaves it there whatever the target holds: a concurrent change of
/// the member in the same direction is kept, not duplicated. (A key the
/// branch also puts or removes was read, so a target that certifies holds
/// its base value and nothing about it is rebased.) Counter additions always
/// rebase.
///
/// The plan's delta upserts every touched key whose value changes, and also
/// a touched key whose value does not change when the plan changes a key it
/// is derived from, directly or transitively in the target's dependency
/// graph: a commit evicts a derived key whose inputs change unless it writes
/// the key, so omitting it would publish the key absent rather than as the
/// branch wrote it. If the plan removes one of those inputs, the runtime
/// refuses the delta (`MissingDependency`) rather than publishing the key
/// without it.
///
/// A touched key whose operations all commute is refused as
/// [`BranchError::EvictedOperand`] with no `input` when the target holds no
/// value for it although its input set there is not empty: a change to its
/// inputs evicted it, after the branch's base or before it, or it was never
/// computed, so its operations would start from zero or the empty set and
/// the plan would publish the result as derived from inputs it was never
/// computed from. That holds even when every value the branch read, and the
/// key's input set, are as the base had them, as after an input changed and
/// changed back, or after an input of an input changed while the input was
/// recomputed to its value. It is refused naming the input when the plan
/// changes a key it is derived from, directly or transitively in the
/// target's dependency graph: its operations were applied to the target's
/// value, which that change evicts, and publishing the result would build on
/// an invalidated value. Staging refuses an operation on a derived key its
/// base holds no value for, and one that would build on a value the
/// branch's own change evicts, in either order
/// ([`Branch::stage_commutative`](crate::Branch::stage_commutative),
/// [`Branch::put`](crate::Branch::put)), but it judges by the base, and
/// decides whether the branch changes an input by comparing its value with
/// the base's. So a branch staging accepted still meets these refusals when
/// a commit after its base evicted the key; when the target's dependency
/// graph links two keys where the base's did not; or when its commutative
/// operations leave an input as the base holds it but change the target's,
/// for example when the target holds the same count or members under
/// another payload source, which the merged value replaces with
/// [`OP_SOURCE`](crate::OP_SOURCE). A branch rebuilt from parts staging
/// would not produce meets them too. A `Put` of the value recomputed from
/// the key's inputs, staged by a branch over the target, is the way through.
///
/// The plan carries the relied-on generations, and the runtime
/// must check them again when it commits (see [`MergePlan`]); a generation
/// revoked or superseded after this call is not seen here.
///
/// What sealing guarantees is checked once more rather than trusted
/// ([`SealedBranch::recheck`], the invariants [`SealedBranch::from_parts`]
/// lists): an operation on a key reserved to ingress is refused as
/// [`BranchError::ReservedNamespace`], a `Put` or `Remove` of a key the
/// branch did not read as [`BranchError::UnreadTarget`], since merging it
/// would overwrite a concurrent change unseen, a `Remove` of a derived key as
/// [`BranchError::DerivedRemoval`], and a touched key without a
/// recorded base value (which decides `Rebased` over `Clean`) or input set
/// as [`BranchError::MalformedSeal`]; all of these before the target is
/// consulted. An input of a touched key that the branch did not read is a
/// conflict, since staging the operation reads every input of its key: the
/// input set is hidden behind its digest, so only the target, whose set
/// matches that digest, can name the inputs.
pub fn certify<V>(
    branch: &SealedBranch,
    target: &SemanticSnapshot,
    validity: V,
) -> Result<Certification, BranchError>
where
    V: Fn(&str, Generation) -> Option<Validity>,
{
    branch.recheck()?;
    if target.revision < branch.base_revision() {
        return Err(BranchError::SnapshotBehindBase {
            base: branch.base_revision(),
            snapshot: target.revision,
        });
    }

    let stale: BTreeSet<String> = branch
        .relied()
        .iter()
        .filter(|(target, generation)| validity(target, **generation) != Some(Validity::Live))
        .map(|(target, _)| target.clone())
        .collect();
    if !stale.is_empty() {
        return Err(BranchError::LifecycleChanged { targets: stale });
    }

    let mut conflicts = BTreeSet::new();
    for (key, digest) in branch.reads() {
        if ValueDigest::of(key, target.value(key))? != *digest {
            conflicts.insert(key.clone());
        }
    }
    for (prefix, digest) in branch.scans() {
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
    // The recheck above made the touched keys exactly the operated ones.
    for (key, inputs) in branch.touched_inputs() {
        if *inputs != InputsDigest::of(key, target.inputs(key)) {
            conflicts.insert(key.clone());
        } else {
            // The input set is the base's, so these are the inputs staging
            // the operation read; one the branch did not read is a
            // dependency it never declared.
            conflicts.extend(
                target
                    .inputs(key)
                    .filter(|input| !branch.reads().contains_key(*input))
                    .map(str::to_owned),
            );
        }
    }
    // The last set operation on each member, and the base presence the
    // recheck made every operation on that member agree on.
    let mut members: BTreeMap<(&str, &str), (bool, bool)> = BTreeMap::new();
    for op in branch.ops() {
        if let Some((member, after, in_base)) = op.set_member() {
            members.insert((op.key(), member), (after, in_base));
        }
    }
    for ((key, member), (after, in_base)) in members {
        if after == in_base && holds_member(target.value(key), member) != in_base {
            conflicts.insert(key.to_owned());
        }
    }
    if !conflicts.is_empty() {
        return Err(BranchError::Conflict { keys: conflicts });
    }

    // A key whose operations all commute is computed from the target's
    // value. A derived key the target holds no value for was evicted, or
    // never computed: the operations would start from absence, and the plan
    // would publish the result as derived from its inputs.
    if let Some(key) = branch.touched_inputs().keys().find(|key| {
        only_commutes(branch, key)
            && target.value(key).is_none()
            && target.inputs(key).next().is_some()
    }) {
        return Err(BranchError::EvictedOperand {
            key: key.clone(),
            input: None,
        });
    }

    let mut finals: BTreeMap<&str, Option<SemanticValue>> = BTreeMap::new();
    for op in branch.ops() {
        let current = match finals.remove(op.key()) {
            Some(value) => value,
            None => target.value(op.key()).cloned(),
        };
        finals.insert(op.key(), op.apply(current)?);
    }

    let changed: BTreeSet<&str> = finals
        .iter()
        .filter(|(key, value)| value.as_ref() != target.value(key))
        .map(|(key, _)| *key)
        .collect();
    // A key whose operations all commute was computed from the target's value;
    // if the plan changes a key it is derived from, that value is one the
    // commit evicts, and the plan would publish the operations onto it.
    for key in finals.keys() {
        if only_commutes(branch, key) {
            if let Some(input) = changed_input_of(target, key, &changed) {
                return Err(BranchError::EvictedOperand {
                    key: (*key).to_owned(),
                    input: Some(input.to_owned()),
                });
            }
        }
    }
    let mut delta = SemanticDelta::default();
    for (key, value) in finals {
        match (value, target.value(key)) {
            (Some(next), Some(current))
                if &next == current && changed_input_of(target, key, &changed).is_none() => {}
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
    for op in branch.ops().iter().filter(|op| op.commutes()) {
        let key = op.key();
        if let Some(base) = branch.touched_base().get(key) {
            if ValueDigest::of(key, target.value(key))? != *base {
                rebased.insert(key.to_owned());
            }
        }
    }

    let plan = MergePlan {
        branch: branch.id().clone(),
        expected: target.revision,
        delta,
        rebased,
        relied: branch.relied().clone(),
        dependencies: dependency_digest(branch),
    };
    if plan.rebased.is_empty() {
        Ok(Certification::Clean(plan))
    } else {
        Ok(Certification::Rebased(plan))
    }
}

/// Whether every operation of `branch` on `key` commutes: the value it
/// publishes there is computed from the target's.
fn only_commutes(branch: &SealedBranch, key: &str) -> bool {
    branch
        .ops()
        .iter()
        .all(|op| op.key() != key || op.commutes())
}

/// A key in `changed` that `key` is derived from, directly or transitively
/// in `target`'s dependency graph (which is acyclic), or `None`: a commit
/// that changes it evicts `key` unless it writes `key`.
fn changed_input_of<'a>(
    target: &'a SemanticSnapshot,
    key: &str,
    changed: &BTreeSet<&str>,
) -> Option<&'a str> {
    let mut pending: Vec<&str> = target.inputs(key).collect();
    let mut seen = BTreeSet::new();
    while let Some(input) = pending.pop() {
        if changed.contains(input) {
            return Some(input);
        }
        if seen.insert(input) {
            pending.extend(target.inputs(input));
        }
    }
    None
}

/// Every section is counted and every variable-length item length-delimited,
/// so no two declarations share an encoding.
fn dependency_digest(branch: &SealedBranch) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-branch/dependencies/v3");
    hasher.update(branch.base_revision().0.to_le_bytes());
    hasher.update((branch.reads().len() as u64).to_le_bytes());
    for (key, digest) in branch.reads() {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.update((branch.scans().len() as u64).to_le_bytes());
    for (prefix, digest) in branch.scans() {
        hasher.update((prefix.len() as u64).to_le_bytes());
        hasher.update(prefix.as_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.update((branch.relied().len() as u64).to_le_bytes());
    for (target, generation) in branch.relied() {
        hasher.update((target.len() as u64).to_le_bytes());
        hasher.update(target.as_bytes());
        hasher.update(generation.0.to_le_bytes());
    }
    hasher.update((branch.touched_inputs().len() as u64).to_le_bytes());
    for (key, digest) in branch.touched_inputs() {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update(digest.as_bytes());
    }
    // One base presence per member: the recheck refuses two.
    let members: BTreeMap<(&str, &str), bool> = branch
        .ops()
        .iter()
        .filter_map(|op| {
            op.set_member()
                .map(|(member, _, in_base)| ((op.key(), member), in_base))
        })
        .collect();
    hasher.update((members.len() as u64).to_le_bytes());
    for ((key, member), in_base) in members {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update((member.len() as u64).to_le_bytes());
        hasher.update(member.as_bytes());
        hasher.update([u8::from(in_base)]);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch::SealedBranchParts;
    use crate::ops::BranchOp;
    use ptr_semdb::SemanticHost;
    use ptr_types::PrincipalId;

    fn text(value: &str) -> SemanticValue {
        SemanticValue::Text(value.into())
    }

    fn no_lifecycle(_: &str, _: Generation) -> Option<Validity> {
        None
    }

    /// Parts for `op` with its key's current value and input set declared as
    /// read and as touched, so every digest matches `target`.
    fn matching_parts(target: &SemanticSnapshot, op: BranchOp) -> SealedBranchParts {
        let key = op.key().to_owned();
        let digest = ValueDigest::of(&key, target.value(&key)).unwrap();
        SealedBranchParts {
            id: BranchId::from("unchecked"),
            author: PrincipalId::from("agent-1"),
            base_revision: target.revision,
            reads: BTreeMap::from([(key.clone(), digest)]),
            scans: BTreeMap::new(),
            relied: BTreeMap::new(),
            touched_base: BTreeMap::from([(key.clone(), digest)]),
            touched_inputs: BTreeMap::from([(
                key.clone(),
                InputsDigest::of(&key, target.inputs(&key)),
            )]),
            ops: vec![op],
        }
    }

    fn target() -> SemanticSnapshot {
        let mut host = SemanticHost::default();
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert("request:r1:raw".into(), text("original"));
        delta.upserts.insert("k".into(), text("1"));
        host.apply_delta(delta).unwrap();
        host.snapshot()
    }

    #[test]
    fn certification_refuses_a_reserved_write_even_from_a_branch_that_skipped_the_constructor() {
        let target = target();
        let parts = matching_parts(
            &target,
            BranchOp::Put {
                key: "request:r1:raw".into(),
                value: text("rewritten"),
            },
        );
        let branch = SealedBranch::unchecked(parts);
        assert_eq!(
            certify(&branch, &target, no_lifecycle).unwrap_err(),
            BranchError::ReservedNamespace {
                key: "request:r1:raw".into()
            }
        );
        assert_eq!(
            branch.recheck(),
            certify(&branch, &target, no_lifecycle).map(drop)
        );
    }

    #[test]
    fn certification_refuses_every_other_broken_sealing_invariant_on_its_own() {
        let target = target();
        let put = || BranchOp::Put {
            key: "k".into(),
            value: text("2"),
        };
        let mut unread = matching_parts(&target, put());
        unread.reads.clear();
        let mut without_base = matching_parts(&target, put());
        without_base.touched_base.clear();
        let mut without_inputs = matching_parts(&target, put());
        without_inputs.touched_inputs.clear();
        let mut stray = matching_parts(&target, put());
        stray
            .touched_inputs
            .insert("other".into(), InputsDigest::of("other", []));
        let mut forged_base = matching_parts(&target, put());
        forged_base
            .touched_base
            .insert("k".into(), ValueDigest::of("k", None).unwrap());
        let mut derived_removal = matching_parts(&target, BranchOp::Remove { key: "k".into() });
        derived_removal
            .touched_inputs
            .insert("k".into(), InputsDigest::of("k", ["input"]));
        let set_op = |insert: bool, in_base: bool| {
            if insert {
                BranchOp::SetInsert {
                    key: "k".into(),
                    member: "m".into(),
                    in_base,
                }
            } else {
                BranchOp::SetRemove {
                    key: "k".into(),
                    member: "m".into(),
                    in_base,
                }
            }
        };
        let mut two_presences = matching_parts(&target, set_op(true, false));
        two_presences.ops.push(set_op(false, true));
        let mut present_without_key = matching_parts(&target, set_op(false, true));
        present_without_key
            .touched_base
            .insert("k".into(), ValueDigest::of("k", None).unwrap());
        present_without_key.reads.clear();
        for (parts, code) in [
            (unread, "PTR_BRANCH_UNREAD_TARGET"),
            (without_base, "PTR_BRANCH_MALFORMED_SEAL"),
            (without_inputs, "PTR_BRANCH_MALFORMED_SEAL"),
            (stray, "PTR_BRANCH_MALFORMED_SEAL"),
            (forged_base, "PTR_BRANCH_MALFORMED_SEAL"),
            (derived_removal, "PTR_BRANCH_DERIVED_REMOVAL"),
            (two_presences, "PTR_BRANCH_MALFORMED_SEAL"),
            (present_without_key, "PTR_BRANCH_MALFORMED_SEAL"),
        ] {
            let refused = SealedBranch::from_parts(parts.clone()).unwrap_err();
            assert_eq!(refused.code(), code, "{refused:?}");
            let branch = SealedBranch::unchecked(parts);
            assert_eq!(certify(&branch, &target, no_lifecycle), Err(refused));
        }
        // The same branch with every invariant intact certifies.
        let intact = SealedBranch::unchecked(matching_parts(&target, put()));
        assert!(certify(&intact, &target, no_lifecycle).is_ok());
    }
}
