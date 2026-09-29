//! The reference model: the semantics of `ptr-semdb` and of the lifecycle
//! authority, re-implemented from their documentation so that the harness has
//! an answer of its own to compare the runtime with.
//!
//! The model never calls `certify`, `ValueDigest`, `RangeDigest`,
//! `InputsDigest`, `counter_value` or `set_value`; it owns its value type and
//! writes counters and sets from their documented byte formats:
//!
//! - a counter is the payload type `ptr.counter.i64` holding the eight
//!   little-endian bytes of an `i64`;
//! - a set is the payload type `ptr.set.utf8` holding a little-endian `u32`
//!   member count, then each member as a little-endian `u32` length and its
//!   UTF-8 bytes, members sorted and distinct;
//! - a value an operation produces carries the source `ptr-branch/op`.
//!
//! A delta applies as `ptr-semdb` documents it: removals drop a key's value
//! and its dependency entry, upserts replace a value, dependency entries
//! replace a derived key's whole input set (an empty set drops it), the keys
//! affected by any change, over the old and the new graph, lose their value
//! unless the delta upserts them, and every upserted derived key needs every
//! input to hold a value afterwards. The revision moves only when values or
//! dependencies changed.

use std::collections::{BTreeMap, BTreeSet};

/// The payload type of a counter.
pub const COUNTER_TYPE: &str = "ptr.counter.i64";
/// The payload type of a set.
pub const SET_TYPE: &str = "ptr.set.utf8";
/// The source of the value an operation produces.
pub const OP_SOURCE: &str = "ptr-branch/op";

/// A value of the semantic state: text, or a typed payload with its source.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Val {
    Text(String),
    Payload {
        type_id: String,
        source: String,
        bytes: Vec<u8>,
    },
}

impl Val {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// A counter holding `count`, from `source`.
    pub fn counter(count: i64, source: &str) -> Self {
        Self::Payload {
            type_id: COUNTER_TYPE.to_string(),
            source: source.to_string(),
            bytes: count.to_le_bytes().to_vec(),
        }
    }

    /// A set of `members`, from `source`.
    pub fn set(members: &BTreeSet<String>, source: &str) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(members.len() as u32).to_le_bytes());
        for member in members {
            bytes.extend_from_slice(&(member.len() as u32).to_le_bytes());
            bytes.extend_from_slice(member.as_bytes());
        }
        Self::Payload {
            type_id: SET_TYPE.to_string(),
            source: source.to_string(),
            bytes,
        }
    }

    /// The count of a counter; `None` for anything else.
    pub fn as_counter(&self) -> Option<i64> {
        match self {
            Self::Payload { type_id, bytes, .. } if type_id == COUNTER_TYPE => {
                let bytes: [u8; 8] = bytes.as_slice().try_into().ok()?;
                Some(i64::from_le_bytes(bytes))
            }
            _ => None,
        }
    }

    /// The members of a canonical set; `None` for anything else.
    pub fn as_set(&self) -> Option<BTreeSet<String>> {
        let Self::Payload { type_id, bytes, .. } = self else {
            return None;
        };
        if type_id != SET_TYPE {
            return None;
        }
        let count = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
        let mut offset = 4;
        let mut members = Vec::new();
        for _ in 0..count {
            let length =
                u32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?) as usize;
            offset += 4;
            let member = std::str::from_utf8(bytes.get(offset..offset + length)?).ok()?;
            offset += length;
            members.push(member.to_string());
        }
        let canonical = offset == bytes.len()
            && members.windows(2).all(|pair| pair[0] < pair[1])
            && members.iter().all(|member| !member.is_empty());
        canonical.then(|| members.into_iter().collect())
    }

    /// The text of a text value.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Payload { .. } => None,
        }
    }
}

/// Why an operation cannot be applied to what a key holds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpRefusal {
    NotACounter,
    NotASet,
    EmptyMember,
    Overflow,
}

/// One staged change to a key, as a branch records it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Op {
    Put {
        key: String,
        value: Val,
    },
    Remove {
        key: String,
    },
    Add {
        key: String,
        amount: i64,
    },
    SetInsert {
        key: String,
        member: String,
        in_base: bool,
    },
    SetRemove {
        key: String,
        member: String,
        in_base: bool,
    },
}

impl Op {
    pub fn key(&self) -> &str {
        match self {
            Self::Put { key, .. }
            | Self::Remove { key }
            | Self::Add { key, .. }
            | Self::SetInsert { key, .. }
            | Self::SetRemove { key, .. } => key,
        }
    }

    /// Whether the operation is defined on the value a key holds at merge
    /// time, not on the value the branch observed.
    pub fn commutes(&self) -> bool {
        !matches!(self, Self::Put { .. } | Self::Remove { .. })
    }

    /// For a set operation: the member, whether it is in the set afterwards,
    /// and whether it was in the base's set.
    pub fn set_member(&self) -> Option<(&str, bool, bool)> {
        match self {
            Self::SetInsert {
                member, in_base, ..
            } => Some((member, true, *in_base)),
            Self::SetRemove {
                member, in_base, ..
            } => Some((member, false, *in_base)),
            _ => None,
        }
    }

    /// What `current` becomes under the operation; `None` is absent.
    pub fn apply(&self, current: Option<&Val>) -> Result<Option<Val>, OpRefusal> {
        match self {
            Self::Put { value, .. } => Ok(Some(value.clone())),
            Self::Remove { .. } => Ok(None),
            Self::Add { amount, .. } => {
                let now = match current {
                    None => 0,
                    Some(value) => value.as_counter().ok_or(OpRefusal::NotACounter)?,
                };
                let next = now.checked_add(*amount).ok_or(OpRefusal::Overflow)?;
                Ok(Some(Val::counter(next, OP_SOURCE)))
            }
            Self::SetInsert { member, .. } | Self::SetRemove { member, .. } => {
                if member.is_empty() {
                    return Err(OpRefusal::EmptyMember);
                }
                let mut set = match current {
                    None => BTreeSet::new(),
                    Some(value) => value.as_set().ok_or(OpRefusal::NotASet)?,
                };
                if matches!(self, Self::SetInsert { .. }) {
                    set.insert(member.clone());
                } else {
                    set.remove(member);
                }
                Ok(Some(Val::set(&set, OP_SOURCE)))
            }
        }
    }
}

/// One atomic update: `ptr-semdb`'s `SemanticDelta`, in the model's terms.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Delta {
    pub upserts: BTreeMap<String, Val>,
    pub removals: BTreeSet<String>,
    pub dependencies: BTreeMap<String, BTreeSet<String>>,
}

/// Why a delta is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Refusal {
    /// A key is removed and also upserted or given a dependency entry.
    Conflicting,
    Cyclic,
    MissingInput {
        derived: String,
        input: String,
    },
}

/// What applying a delta did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Applied {
    /// Whether values or dependencies changed, which moves the revision.
    pub moved: bool,
    /// Every key the change invalidates, changed keys included.
    pub affected: BTreeSet<String>,
}

/// Whether a lifecycle generation may be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Validity {
    Live,
    Superseded,
    Revoked,
}

/// The lifecycle authority: the live generation of each target and the
/// generations revoked, whichever is live.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Lifecycle {
    live: BTreeMap<String, u64>,
    revoked: BTreeSet<(String, u64)>,
}

impl Lifecycle {
    /// A capsule committed at `generation`, or superseded by it: the live one.
    pub fn set_live(&mut self, target: &str, generation: u64) {
        self.live.insert(target.to_string(), generation);
    }

    /// A generation revoked; the live generation stays where it was.
    pub fn revoke(&mut self, target: &str, generation: u64) {
        self.revoked.insert((target.to_string(), generation));
    }

    pub fn live(&self, target: &str) -> Option<u64> {
        self.live.get(target).copied()
    }

    /// `None` means the authority knows no such generation: the target is
    /// unknown or the generation is ahead of it.
    pub fn validity(&self, target: &str, generation: u64) -> Option<Validity> {
        if self.revoked.contains(&(target.to_string(), generation)) {
            return Some(Validity::Revoked);
        }
        match self.live.get(target) {
            Some(live) if *live == generation => Some(Validity::Live),
            Some(live) if *live > generation => Some(Validity::Superseded),
            _ => None,
        }
    }
}

/// The semantic state and the lifecycle, as the model tracks them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Model {
    revision: u64,
    values: BTreeMap<String, Val>,
    dependencies: BTreeMap<String, BTreeSet<String>>,
    pub lifecycle: Lifecycle,
}

impl Model {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn value(&self, key: &str) -> Option<&Val> {
        self.values.get(key)
    }

    pub fn values(&self) -> &BTreeMap<String, Val> {
        &self.values
    }

    /// The keys with a value under `prefix`, with their values, in key order.
    pub fn entries_under(&self, prefix: &str) -> Vec<(String, Val)> {
        self.values
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    /// The input set of `key`; empty for a key with none.
    pub fn inputs(&self, key: &str) -> BTreeSet<String> {
        self.dependencies.get(key).cloned().unwrap_or_default()
    }

    pub fn dependencies(&self) -> &BTreeMap<String, BTreeSet<String>> {
        &self.dependencies
    }

    /// Apply `delta`, or refuse it and change nothing.
    pub fn apply(&mut self, delta: &Delta) -> Result<Applied, Refusal> {
        if delta
            .removals
            .iter()
            .any(|key| delta.upserts.contains_key(key) || delta.dependencies.contains_key(key))
        {
            return Err(Refusal::Conflicting);
        }
        let mut values = self.values.clone();
        let mut dependencies = self.dependencies.clone();
        let mut changed: BTreeSet<String> = BTreeSet::new();
        for key in &delta.removals {
            if values.remove(key).is_some() {
                changed.insert(key.clone());
            }
            if dependencies.remove(key).is_some() {
                changed.insert(key.clone());
            }
        }
        for (key, value) in &delta.upserts {
            if values.get(key) != Some(value) {
                values.insert(key.clone(), value.clone());
                changed.insert(key.clone());
            }
        }
        for (key, inputs) in &delta.dependencies {
            let old = dependencies.get(key);
            if old != Some(inputs) && !(old.is_none() && inputs.is_empty()) {
                if inputs.is_empty() {
                    dependencies.remove(key);
                } else {
                    dependencies.insert(key.clone(), inputs.clone());
                }
                changed.insert(key.clone());
            }
        }
        if has_cycle(&dependencies) {
            return Err(Refusal::Cyclic);
        }
        let mut affected = closure(&self.dependencies, &changed);
        affected.extend(closure(&dependencies, &changed));
        for key in &affected {
            if !delta.upserts.contains_key(key) {
                values.remove(key);
            }
        }
        for key in delta.upserts.keys() {
            for input in dependencies.get(key).into_iter().flatten() {
                if !values.contains_key(input) {
                    return Err(Refusal::MissingInput {
                        derived: key.clone(),
                        input: input.clone(),
                    });
                }
            }
        }
        let moved = values != self.values || dependencies != self.dependencies;
        self.values = values;
        self.dependencies = dependencies;
        if moved {
            self.revision += 1;
        }
        Ok(Applied { moved, affected })
    }
}

/// `changed` and every key derived from one of them, directly or through
/// other derived keys, in the graph `inputs` (derived key to its inputs).
fn closure(
    inputs: &BTreeMap<String, BTreeSet<String>>,
    changed: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut derived_from: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (derived, set) in inputs {
        for input in set {
            derived_from.entry(input).or_default().push(derived);
        }
    }
    let mut out: BTreeSet<String> = BTreeSet::new();
    let mut pending: Vec<String> = changed.iter().cloned().collect();
    while let Some(key) = pending.pop() {
        if out.insert(key.clone()) {
            if let Some(children) = derived_from.get(key.as_str()) {
                pending.extend(children.iter().map(|child| (*child).to_string()));
            }
        }
    }
    out
}

/// Whether the graph has a cycle: some key that is, through its inputs, its
/// own input.
fn has_cycle(inputs: &BTreeMap<String, BTreeSet<String>>) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Open,
        Done,
    }
    fn visit<'a>(
        key: &'a str,
        inputs: &'a BTreeMap<String, BTreeSet<String>>,
        marks: &mut BTreeMap<&'a str, Mark>,
    ) -> bool {
        match marks.get(key) {
            Some(Mark::Done) => return false,
            Some(Mark::Open) => return true,
            None => {}
        }
        marks.insert(key, Mark::Open);
        for input in inputs.get(key).into_iter().flatten() {
            if visit(input, inputs, marks) {
                return true;
            }
        }
        marks.insert(key, Mark::Done);
        false
    }
    let mut marks = BTreeMap::new();
    inputs.keys().any(|key| visit(key, inputs, &mut marks))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    fn delta(
        upserts: &[(&str, &str)],
        removals: &[&str],
        dependencies: &[(&str, &[&str])],
    ) -> Delta {
        Delta {
            upserts: upserts
                .iter()
                .map(|(key, value)| ((*key).to_string(), Val::text(*value)))
                .collect(),
            removals: set(removals),
            dependencies: dependencies
                .iter()
                .map(|(key, inputs)| ((*key).to_string(), set(inputs)))
                .collect(),
        }
    }

    fn model_with_total() -> Model {
        let mut model = Model::default();
        model
            .apply(&delta(&[("a", "1"), ("b", "2")], &[], &[]))
            .expect("inputs");
        model
            .apply(&delta(&[("total", "3")], &[], &[("total", &["a", "b"])]))
            .expect("derived");
        model
    }

    #[test]
    fn counters_and_sets_are_written_in_their_documented_formats() {
        assert_eq!(
            Val::counter(-2, "src"),
            Val::Payload {
                type_id: "ptr.counter.i64".to_string(),
                source: "src".to_string(),
                bytes: vec![0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            }
        );
        assert_eq!(Val::counter(50, "s").as_counter(), Some(50));
        let members = set(&["b", "a\nc"]);
        let value = Val::set(&members, OP_SOURCE);
        let Val::Payload { type_id, bytes, .. } = &value else {
            panic!("a set is a payload");
        };
        assert_eq!(type_id, "ptr.set.utf8");
        // two members, sorted as bytes: "a\nc" (3 bytes) then "b" (1 byte)
        assert_eq!(
            bytes,
            &[2, 0, 0, 0, 3, 0, 0, 0, b'a', b'\n', b'c', 1, 0, 0, 0, b'b']
        );
        assert_eq!(value.as_set(), Some(members));
        assert_eq!(
            Val::set(&BTreeSet::new(), "s").as_set(),
            Some(BTreeSet::new())
        );
        assert_eq!(Val::text("1").as_counter(), None);
        assert_eq!(Val::counter(1, "s").as_set(), None);
        let mut broken = Val::set(&set(&["a"]), "s");
        if let Val::Payload { bytes, .. } = &mut broken {
            bytes.push(0);
        }
        assert_eq!(
            broken.as_set(),
            None,
            "trailing bytes are not a canonical set"
        );
    }

    #[test]
    fn operations_apply_to_what_a_key_holds_and_refuse_what_they_cannot() {
        let add = |amount| Op::Add {
            key: "c".into(),
            amount,
        };
        assert_eq!(add(3).apply(None), Ok(Some(Val::counter(3, OP_SOURCE))));
        assert_eq!(
            add(-4).apply(Some(&Val::counter(10, "g"))),
            Ok(Some(Val::counter(6, OP_SOURCE)))
        );
        assert_eq!(
            add(1).apply(Some(&Val::text("1"))),
            Err(OpRefusal::NotACounter)
        );
        assert_eq!(
            add(1).apply(Some(&Val::counter(i64::MAX, "g"))),
            Err(OpRefusal::Overflow)
        );
        let insert = |member: &str| Op::SetInsert {
            key: "s".into(),
            member: member.into(),
            in_base: false,
        };
        let remove = |member: &str| Op::SetRemove {
            key: "s".into(),
            member: member.into(),
            in_base: false,
        };
        let one = insert("m1").apply(None).unwrap();
        assert_eq!(one.as_ref().and_then(Val::as_set), Some(set(&["m1"])));
        let two = insert("m2").apply(one.as_ref()).unwrap();
        assert_eq!(
            remove("m1")
                .apply(two.as_ref())
                .unwrap()
                .and_then(|value| value.as_set()),
            Some(set(&["m2"]))
        );
        assert_eq!(insert("").apply(None), Err(OpRefusal::EmptyMember));
        assert_eq!(
            insert("m").apply(Some(&Val::counter(1, "g"))),
            Err(OpRefusal::NotASet)
        );
        assert_eq!(
            Op::Put {
                key: "k".into(),
                value: Val::text("v")
            }
            .apply(Some(&Val::text("w"))),
            Ok(Some(Val::text("v")))
        );
        assert_eq!(
            Op::Remove { key: "k".into() }.apply(Some(&Val::text("w"))),
            Ok(None)
        );
    }

    #[test]
    fn a_changed_input_evicts_the_derived_key_the_delta_does_not_upsert() {
        let mut model = model_with_total();
        let applied = model
            .apply(&delta(&[("a", "5")], &[], &[]))
            .expect("applies");
        assert!(applied.moved);
        assert_eq!(applied.affected, set(&["a", "total"]));
        assert_eq!(model.value("total"), None, "the derived value is evicted");
        assert_eq!(
            model.inputs("total"),
            set(&["a", "b"]),
            "its dependency entry stays"
        );
        // Upserting the derived key in the same delta keeps it.
        let mut model = model_with_total();
        model
            .apply(&delta(&[("a", "5"), ("total", "7")], &[], &[]))
            .expect("applies");
        assert_eq!(model.value("total"), Some(&Val::text("7")));
    }

    #[test]
    fn eviction_follows_the_graph_through_derived_keys_and_over_both_graphs() {
        let mut model = model_with_total();
        model
            .apply(&delta(&[("grand", "9")], &[], &[("grand", &["total"])]))
            .expect("a second level");
        model
            .apply(&delta(&[("b", "4")], &[], &[]))
            .expect("applies");
        assert_eq!(model.value("total"), None);
        assert_eq!(
            model.value("grand"),
            None,
            "a key derived from an evicted key is evicted too"
        );
        // Rewiring: the old graph's dependents are affected as well as the new one's.
        let mut model = model_with_total();
        let applied = model
            .apply(&delta(&[("total", "1")], &[], &[("total", &["a"])]))
            .expect("rewired and recomputed");
        assert!(applied.affected.contains("total"));
        assert_eq!(model.inputs("total"), set(&["a"]));
        assert_eq!(model.value("total"), Some(&Val::text("1")));
    }

    #[test]
    fn a_derived_upsert_needs_every_input_to_hold_a_value() {
        let mut model = model_with_total();
        let before = model.clone();
        let refusal = model
            .apply(&delta(&[("total", "9")], &["b"], &[]))
            .expect_err("b is gone");
        assert_eq!(
            refusal,
            Refusal::MissingInput {
                derived: "total".into(),
                input: "b".into()
            }
        );
        assert_eq!(model, before, "a refused delta changes nothing");
    }

    #[test]
    fn the_revision_moves_only_when_something_changed() {
        let mut model = model_with_total();
        let revision = model.revision();
        let applied = model
            .apply(&delta(&[("a", "1")], &[], &[]))
            .expect("a no-op");
        assert!(!applied.moved);
        assert_eq!(model.revision(), revision);
        assert!(model.apply(&delta(&[("a", "2")], &[], &[])).unwrap().moved);
        assert_eq!(model.revision(), revision + 1);
        // An empty dependency entry for a key with none changes nothing.
        let applied = model
            .apply(&delta(&[], &[], &[("a", &[])]))
            .expect("empty entry");
        assert!(!applied.moved);
    }

    #[test]
    fn a_conflicting_or_cyclic_delta_is_refused() {
        let mut model = model_with_total();
        assert_eq!(
            model.apply(&delta(&[("a", "1")], &["a"], &[])),
            Err(Refusal::Conflicting)
        );
        assert_eq!(
            model.apply(&delta(&[], &["total"], &[("total", &["a"])])),
            Err(Refusal::Conflicting)
        );
        assert_eq!(
            model.apply(&delta(&[], &[], &[("a", &["total"])])),
            Err(Refusal::Cyclic)
        );
    }

    #[test]
    fn a_removal_drops_the_value_and_the_dependency_entry() {
        let mut model = model_with_total();
        model.apply(&delta(&[], &["total"], &[])).expect("removes");
        assert_eq!(model.value("total"), None);
        assert!(model.inputs("total").is_empty());
    }

    #[test]
    fn a_generation_is_live_superseded_revoked_or_unknown() {
        let mut lifecycle = Lifecycle::default();
        assert_eq!(lifecycle.validity("policy-0", 1), None);
        lifecycle.set_live("policy-0", 1);
        assert_eq!(lifecycle.validity("policy-0", 1), Some(Validity::Live));
        assert_eq!(
            lifecycle.validity("policy-0", 2),
            None,
            "ahead of the live generation"
        );
        lifecycle.set_live("policy-0", 2);
        assert_eq!(
            lifecycle.validity("policy-0", 1),
            Some(Validity::Superseded)
        );
        lifecycle.revoke("policy-0", 2);
        assert_eq!(
            lifecycle.validity("policy-0", 2),
            Some(Validity::Revoked),
            "a revoked generation is not live although it is the live one"
        );
        assert_eq!(lifecycle.live("policy-0"), Some(2));
        lifecycle.set_live("policy-0", 3);
        assert_eq!(lifecycle.validity("policy-0", 3), Some(Validity::Live));
        assert_eq!(lifecycle.validity("policy-0", 2), Some(Validity::Revoked));
    }
}
