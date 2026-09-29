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
use std::ops::Bound;

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

    pub fn live_entries(&self) -> impl Iterator<Item = (&str, u64)> {
        self.live
            .iter()
            .map(|(target, generation)| (target.as_str(), *generation))
    }

    pub fn revoked_entries(&self) -> impl Iterator<Item = (&str, u64)> {
        self.revoked
            .iter()
            .map(|(target, generation)| (target.as_str(), *generation))
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
    /// The graph the other way round, so that what a change invalidates is
    /// found without walking every entry: each input to the derived keys
    /// that list it. Always what `dependencies` implies.
    dependents: BTreeMap<String, BTreeSet<String>>,
    pub lifecycle: Lifecycle,
}

/// What applying a delta changes, and nothing of the state it applies to:
/// each key whose value is different afterwards with the value it then has
/// (`None` for a key that no longer holds one), and each derived key whose
/// input set is different afterwards with the set it then has (empty for
/// none). Two deltas applied to one state leave the same state exactly when
/// their nets are equal.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Net {
    pub values: BTreeMap<String, Option<Val>>,
    pub dependencies: BTreeMap<String, BTreeSet<String>>,
}

impl Net {
    /// The value `key` holds after the change, given what it holds now.
    pub fn value_after(&self, key: &str, now: Option<&Val>) -> Option<Val> {
        match self.values.get(key) {
            Some(value) => value.clone(),
            None => now.cloned(),
        }
    }
}

/// What applying a delta to a state would do, worked out without doing it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub net: Net,
    /// Every key the change invalidates, changed keys included.
    pub affected: BTreeSet<String>,
}

impl Plan {
    /// Whether values or dependencies change, which moves the revision.
    pub fn moved(&self) -> bool {
        !self.net.values.is_empty() || !self.net.dependencies.is_empty()
    }
}

fn negative(value: &Val) -> bool {
    value.as_counter().is_some_and(|count| count < 0)
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

    /// The entries whose key starts with `prefix`, in key order.
    fn under<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = (&'a String, &'a Val)> {
        self.values
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .take_while(move |(key, _)| key.starts_with(prefix))
    }

    /// The keys with a value under `prefix`, with their values, in key order.
    pub fn entries_under(&self, prefix: &str) -> Vec<(String, Val)> {
        self.under(prefix)
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

    /// Whether any counter holds a negative value.
    pub fn negative_counter(&self) -> bool {
        self.negative_counter_after(&Net::default())
    }

    /// Whether any counter would hold a negative value after `net`.
    pub fn negative_counter_after(&self, net: &Net) -> bool {
        let held = self
            .under("ctr:")
            .any(|(key, value)| match net.values.get(key) {
                Some(after) => after.as_ref().is_some_and(negative),
                None => negative(value),
            });
        held || net.values.iter().any(|(key, after)| {
            key.starts_with("ctr:")
                && !self.values.contains_key(key)
                && after.as_ref().is_some_and(negative)
        })
    }

    /// Apply `delta`, or refuse it and change nothing.
    pub fn apply(&mut self, delta: &Delta) -> Result<Applied, Refusal> {
        let plan = self.plan(delta)?;
        Ok(self.commit(plan))
    }

    /// What applying `delta` would do, or why it would be refused, without
    /// changing anything.
    pub fn plan(&self, delta: &Delta) -> Result<Plan, Refusal> {
        if delta
            .removals
            .iter()
            .any(|key| delta.upserts.contains_key(key) || delta.dependencies.contains_key(key))
        {
            return Err(Refusal::Conflicting);
        }
        // The keys that change, and each derived key whose input set does.
        let mut changed: BTreeSet<&str> = BTreeSet::new();
        let mut new_inputs: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        for key in &delta.removals {
            if self.values.contains_key(key) {
                changed.insert(key);
            }
            if self.dependencies.contains_key(key) {
                changed.insert(key);
                new_inputs.insert(key, BTreeSet::new());
            }
        }
        for (key, value) in &delta.upserts {
            if self.values.get(key) != Some(value) {
                changed.insert(key);
            }
        }
        for (key, inputs) in &delta.dependencies {
            let old = self.dependencies.get(key);
            if old != Some(inputs) && !(old.is_none() && inputs.is_empty()) {
                changed.insert(key);
                new_inputs.insert(key, inputs.clone());
            }
        }
        let inputs_after = |key: &str| -> BTreeSet<String> {
            match new_inputs.get(key) {
                Some(inputs) => inputs.clone(),
                None => self.inputs(key),
            }
        };
        // A cycle in the new graph runs through an input set that changed,
        // since the old graph has none.
        for (derived, inputs) in &new_inputs {
            let mut pending: Vec<String> = inputs.iter().cloned().collect();
            let mut seen: BTreeSet<String> = BTreeSet::new();
            while let Some(key) = pending.pop() {
                if key == *derived {
                    return Err(Refusal::Cyclic);
                }
                if seen.insert(key.clone()) {
                    pending.extend(inputs_after(&key));
                }
            }
        }
        // What the change invalidates: the changed keys and everything
        // derived from them, in the old graph and in the new one.
        let mut affected: BTreeSet<String> = BTreeSet::new();
        let mut pending: Vec<String> = changed.iter().map(|key| (*key).to_string()).collect();
        while let Some(key) = pending.pop() {
            if affected.insert(key.clone()) {
                pending.extend(self.dependents.get(&key).into_iter().flatten().cloned());
            }
        }
        let mut in_new_graph: BTreeSet<String> = BTreeSet::new();
        let mut pending: Vec<String> = changed.iter().map(|key| (*key).to_string()).collect();
        while let Some(key) = pending.pop() {
            if in_new_graph.insert(key.clone()) {
                let kept = self
                    .dependents
                    .get(&key)
                    .into_iter()
                    .flatten()
                    .filter(|derived| {
                        new_inputs
                            .get(derived.as_str())
                            .is_none_or(|inputs| inputs.contains(&key))
                    })
                    .cloned();
                let gained = new_inputs
                    .iter()
                    .filter(|(_, inputs)| inputs.contains(&key))
                    .map(|(derived, _)| (*derived).to_string());
                pending.extend(kept.chain(gained));
            }
        }
        affected.extend(in_new_graph);
        // The values that differ afterwards.
        let mut values: BTreeMap<String, Option<Val>> = BTreeMap::new();
        for key in &delta.removals {
            if self.values.contains_key(key) {
                values.insert(key.clone(), None);
            }
        }
        for key in &affected {
            if !delta.upserts.contains_key(key) && self.values.contains_key(key) {
                values.insert(key.clone(), None);
            }
        }
        for (key, value) in &delta.upserts {
            if self.values.get(key) != Some(value) {
                values.insert(key.clone(), Some(value.clone()));
            }
        }
        let holds = |key: &str| match values.get(key) {
            Some(after) => after.is_some(),
            None => self.values.contains_key(key),
        };
        for key in delta.upserts.keys() {
            for input in inputs_after(key) {
                if !holds(&input) {
                    return Err(Refusal::MissingInput {
                        derived: key.clone(),
                        input,
                    });
                }
            }
        }
        let dependencies = new_inputs
            .into_iter()
            .map(|(key, inputs)| (key.to_string(), inputs))
            .collect();
        Ok(Plan {
            net: Net {
                values,
                dependencies,
            },
            affected,
        })
    }

    /// Make a plan worked out against this state the state.
    pub fn commit(&mut self, plan: Plan) -> Applied {
        let moved = plan.moved();
        for (key, value) in plan.net.values {
            match value {
                Some(value) => {
                    self.values.insert(key, value);
                }
                None => {
                    self.values.remove(&key);
                }
            }
        }
        for (derived, inputs) in plan.net.dependencies {
            for input in self.dependencies.remove(&derived).into_iter().flatten() {
                if let Some(derived_keys) = self.dependents.get_mut(&input) {
                    derived_keys.remove(&derived);
                    if derived_keys.is_empty() {
                        self.dependents.remove(&input);
                    }
                }
            }
            for input in &inputs {
                self.dependents
                    .entry(input.clone())
                    .or_default()
                    .insert(derived.clone());
            }
            if !inputs.is_empty() {
                self.dependencies.insert(derived, inputs);
            }
        }
        if moved {
            self.revision += 1;
        }
        Applied {
            moved,
            affected: plan.affected,
        }
    }
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

    /// The implementation `plan` replaced, kept to show they agree: it copies
    /// both maps and walks the whole graph for every delta.
    mod reference {
        use super::super::*;

        pub struct State {
            pub values: BTreeMap<String, Val>,
            pub dependencies: BTreeMap<String, BTreeSet<String>>,
        }

        pub fn apply(state: &mut State, delta: &Delta) -> Result<Applied, Refusal> {
            if delta
                .removals
                .iter()
                .any(|key| delta.upserts.contains_key(key) || delta.dependencies.contains_key(key))
            {
                return Err(Refusal::Conflicting);
            }
            let mut values = state.values.clone();
            let mut dependencies = state.dependencies.clone();
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
            let mut affected = closure(&state.dependencies, &changed);
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
            let moved = values != state.values || dependencies != state.dependencies;
            state.values = values;
            state.dependencies = dependencies;
            Ok(Applied { moved, affected })
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
    }

    fn random_delta(rng: &mut crate::experiments::rng::Rng) -> Delta {
        let keys: Vec<String> = (0..7).map(|index| format!("k{index}")).collect();
        let mut delta = Delta::default();
        for key in &keys {
            match rng.below(8) {
                0 | 1 => {
                    delta
                        .upserts
                        .insert(key.clone(), Val::text(format!("v{}", rng.below(3))));
                }
                2 => {
                    delta.removals.insert(key.clone());
                }
                3 | 4 => {
                    let inputs: BTreeSet<String> =
                        keys.iter().filter(|_| rng.below(3) == 0).cloned().collect();
                    delta.dependencies.insert(key.clone(), inputs);
                    if rng.below(2) == 0 {
                        delta
                            .upserts
                            .insert(key.clone(), Val::text(format!("v{}", rng.below(3))));
                    }
                }
                _ => {}
            }
        }
        delta
    }

    #[test]
    fn planning_and_committing_agree_with_the_whole_copy_implementation_on_random_deltas() {
        let mut rng = crate::experiments::rng::Rng::new(20_260_929);
        let (mut accepted, mut refused, mut moved) = (0u32, 0u32, 0u32);
        for _ in 0..1_500 {
            let mut model = Model::default();
            let mut state = reference::State {
                values: BTreeMap::new(),
                dependencies: BTreeMap::new(),
            };
            for _ in 0..30 {
                let delta = random_delta(&mut rng);
                let expected = reference::apply(&mut state, &delta);
                let planned = model.plan(&delta);
                let got = planned.map(|plan| {
                    let net_moved = plan.moved();
                    let applied = model.commit(plan);
                    assert_eq!(applied.moved, net_moved);
                    applied
                });
                assert_eq!(got, expected, "delta {delta:?}");
                assert_eq!(model.values(), &state.values, "delta {delta:?}");
                assert_eq!(model.dependencies(), &state.dependencies, "delta {delta:?}");
                let mut index: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
                for (derived, inputs) in &state.dependencies {
                    for input in inputs {
                        index
                            .entry(input.clone())
                            .or_default()
                            .insert(derived.clone());
                    }
                }
                assert_eq!(model.dependents, index, "the index follows the graph");
                match got {
                    Ok(applied) => {
                        accepted += 1;
                        moved += u32::from(applied.moved);
                    }
                    Err(_) => refused += 1,
                }
            }
        }
        // The deltas reach every branch: accepted, refused and moving.
        assert!(
            accepted > 10_000 && refused > 5_000 && moved > 5_000,
            "{accepted} {refused} {moved}"
        );
    }

    #[test]
    fn a_plan_leaves_the_model_alone_and_its_net_says_what_changed() {
        let model = model_with_total();
        let before = model.clone();
        let plan = model.plan(&delta(&[("a", "5")], &[], &[])).expect("plans");
        assert_eq!(model, before);
        assert!(plan.moved());
        assert_eq!(plan.net.values.get("a"), Some(&Some(Val::text("5"))));
        assert_eq!(
            plan.net.values.get("total"),
            Some(&None),
            "the derived value is evicted"
        );
        assert!(plan.net.dependencies.is_empty());
        // The same state through two different deltas has one net.
        let twice = model
            .plan(&delta(&[("a", "5"), ("a", "5")], &[], &[]))
            .expect("plans");
        assert_eq!(plan.net, twice.net);
    }

    #[test]
    fn a_counter_that_would_go_negative_is_seen_in_the_net_and_only_there() {
        let mut model = Model::default();
        model
            .apply(&Delta {
                upserts: [("ctr:0".to_string(), Val::counter(5, "g"))].into(),
                ..Delta::default()
            })
            .expect("applies");
        assert!(!model.negative_counter());
        let down = Delta {
            upserts: [("ctr:0".to_string(), Val::counter(-2, "g"))].into(),
            ..Delta::default()
        };
        let plan = model.plan(&down).expect("plans");
        assert!(model.negative_counter_after(&plan.net));
        assert!(!model.negative_counter(), "planning changed nothing");
        let fresh = Delta {
            upserts: [("ctr:9".to_string(), Val::counter(-1, "g"))].into(),
            ..Delta::default()
        };
        assert!(model.negative_counter_after(&model.plan(&fresh).expect("plans").net));
        let removal = Delta {
            removals: ["ctr:0".to_string()].into(),
            ..Delta::default()
        };
        assert!(!model.negative_counter_after(&model.plan(&removal).expect("plans").net));
    }
}
