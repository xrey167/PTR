//! The programs an agent runs, and the two views they run against.
//!
//! A program is one pure function over a [`View`]. The same function runs
//! against a [`BranchView`], a real `ptr_branch::Branch` over the snapshot the
//! agent opened, and against a [`RefView`], the reference model of what a
//! branch declares and shows. Both write the same [`Log`]; the harness
//! requires the logs, and the footprints made from them, to be identical.
//!
//! What a view shows follows `Branch::read` and `Branch::scan_prefix`: the
//! branch's own writes are visible, and a key with no operation of its own
//! that is derived, directly or through other derived keys of the base's
//! graph, from a key the branch changes reads as absent, since the merge
//! evicts it.

use std::collections::{BTreeMap, BTreeSet};

use ptr_branch::{Branch, BranchError, BranchId, BranchOp, SealedBranch};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticSnapshot, SemanticValue};
use ptr_types::{Generation, PrincipalId, TypeId};

use super::model::{Delta, Model, Op, Val};
use super::params;

/// The keys of the workload.
pub mod keys {
    pub fn item(group: usize, index: usize) -> String {
        format!("item:{group}:{index}")
    }
    pub fn spare(group: usize) -> String {
        format!("item:{group}:s")
    }
    pub fn items(group: usize) -> String {
        format!("item:{group}:")
    }
    pub fn extra(group: usize, task: usize) -> String {
        format!("item:{group}:x{task}")
    }
    pub fn extras(group: usize) -> String {
        format!("item:{group}:x")
    }
    pub fn total(group: usize) -> String {
        format!("total:{group}")
    }
    pub fn audit(group: usize) -> String {
        format!("audit:{group}")
    }
    pub fn counter(counter: usize) -> String {
        format!("ctr:{counter}")
    }
    pub fn set(set: usize) -> String {
        format!("set:{set}")
    }
    pub fn member(member: usize) -> String {
        format!("m{member}")
    }
    pub fn policy(policy: usize) -> String {
        format!("policy-{policy}")
    }
}

/// A prefix only ingress writes.
const RESERVED_PREFIXES: [&str; 2] = ["request:", "pod-output:"];

/// The calls a program makes. A call that fails leaves the first failure in
/// the view and returns an empty answer, so a program has no error paths of
/// its own; a refusal a view records in its log is part of what it observed.
pub trait View {
    fn read(&mut self, key: &str) -> Option<Val>;
    fn scan(&mut self, prefix: &str) -> Vec<(String, Val)>;
    /// The input set `key` declares in the base.
    fn inputs(&mut self, key: &str) -> BTreeSet<String>;
    /// The live generation of `target` when the branch was opened, if it is
    /// live and not revoked.
    fn live(&mut self, target: &str) -> Option<u64>;
    fn rely(&mut self, target: &str, generation: u64);
    fn put(&mut self, key: &str, value: Val);
    fn remove(&mut self, key: &str);
    fn add(&mut self, key: &str, amount: i64);
    fn set_insert(&mut self, key: &str, member: &str);
    fn set_remove(&mut self, key: &str, member: &str);
}

/// One call of a program and what it returned or staged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    Read {
        key: String,
        value: Option<Val>,
    },
    Scan {
        prefix: String,
        entries: Vec<(String, Val)>,
    },
    Inputs {
        key: String,
        inputs: BTreeSet<String>,
    },
    Live {
        target: String,
        generation: Option<u64>,
    },
    Rely {
        target: String,
        generation: u64,
    },
    Staged(Op),
    /// A staging call the view refused, and why.
    Refused {
        call: &'static str,
        key: String,
        why: String,
    },
}

/// Everything a view observed and staged, in order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Log {
    pub events: Vec<Event>,
}

/// The state a branch is opened on, as a footprint is made from it.
pub trait Base {
    fn value(&self, key: &str) -> Option<Val>;
    fn entries_under(&self, prefix: &str) -> Vec<(String, Val)>;
    fn inputs(&self, key: &str) -> BTreeSet<String>;
}

/// The base a [`RefView`] reads: the model at the branch's position.
pub struct ModelBase<'m>(pub &'m Model);

impl Base for ModelBase<'_> {
    fn value(&self, key: &str) -> Option<Val> {
        self.0.value(key).cloned()
    }
    fn entries_under(&self, prefix: &str) -> Vec<(String, Val)> {
        self.0.entries_under(prefix)
    }
    fn inputs(&self, key: &str) -> BTreeSet<String> {
        self.0.inputs(key)
    }
}

/// The base a [`BranchView`] reads: the runtime's snapshot.
pub struct SnapshotBase(pub SemanticSnapshot);

impl Base for SnapshotBase {
    fn value(&self, key: &str) -> Option<Val> {
        self.0.value(key).map(from_semantic)
    }
    fn entries_under(&self, prefix: &str) -> Vec<(String, Val)> {
        self.0
            .keys()
            .filter(|key| key.starts_with(prefix))
            .filter_map(|key| {
                self.0
                    .value(key)
                    .map(|value| (key.to_string(), from_semantic(value)))
            })
            .collect()
    }
    fn inputs(&self, key: &str) -> BTreeSet<String> {
        self.0.inputs(key).map(str::to_string).collect()
    }
}

/// A key with the value and input set the base held for it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Touched {
    pub base: Option<Val>,
    pub inputs: BTreeSet<String>,
}

/// What a branch declares, as the oracle judges it: the read set R with the
/// value each key had in the base, its written subset W, the scans S with the
/// entries the base held under each prefix, the commutative keys C, the
/// touched keys T = W ∪ C with their base value and input set, the relied
/// pairs L, and the staged operations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Footprint {
    pub reads: BTreeMap<String, Option<Val>>,
    pub written: BTreeSet<String>,
    pub scans: BTreeMap<String, Vec<(String, Val)>>,
    pub commutative: BTreeSet<String>,
    pub touched: BTreeMap<String, Touched>,
    pub relied: BTreeMap<String, u64>,
    pub ops: Vec<Op>,
}

/// The footprint of a log over its base. Staging a `Put` or a commutative
/// operation also reads the key's declared inputs, as `Branch::put` and
/// `Branch::stage_commutative` do.
pub fn footprint(log: &Log, base: &impl Base) -> Footprint {
    let mut footprint = Footprint::default();
    for event in &log.events {
        match event {
            Event::Read { key, .. } => {
                footprint.reads.insert(key.clone(), base.value(key));
            }
            Event::Scan { prefix, .. } => {
                footprint
                    .scans
                    .entry(prefix.clone())
                    .or_insert_with(|| base.entries_under(prefix));
            }
            Event::Rely { target, generation } => {
                footprint
                    .relied
                    .entry(target.clone())
                    .or_insert(*generation);
            }
            Event::Staged(op) => {
                let key = op.key().to_string();
                footprint
                    .touched
                    .entry(key.clone())
                    .or_insert_with(|| Touched {
                        base: base.value(&key),
                        inputs: base.inputs(&key),
                    });
                if op.commutes() {
                    footprint.commutative.insert(key.clone());
                } else {
                    footprint.written.insert(key.clone());
                }
                if !matches!(op, Op::Remove { .. }) {
                    for input in base.inputs(&key) {
                        footprint
                            .reads
                            .entry(input.clone())
                            .or_insert_with(|| base.value(&input));
                    }
                }
                footprint.ops.push(op.clone());
            }
            Event::Inputs { .. } | Event::Live { .. } | Event::Refused { .. } => {}
        }
    }
    footprint
}

/// The model's view of a branch opened at its position.
pub struct RefView<'m> {
    base: &'m Model,
    log: Log,
    reads: BTreeSet<String>,
    relied: BTreeMap<String, u64>,
    ops: Vec<Op>,
    error: Option<String>,
}

impl<'m> RefView<'m> {
    pub fn open(base: &'m Model) -> Self {
        Self {
            base,
            log: Log::default(),
            reads: BTreeSet::new(),
            relied: BTreeMap::new(),
            ops: Vec::new(),
            error: None,
        }
    }

    /// The log, the footprint made from it and the first unexpected failure.
    pub fn finish(self) -> (Log, Footprint, Option<String>) {
        let footprint = footprint(&self.log, &ModelBase(self.base));
        (self.log, footprint, self.error)
    }

    fn operates_on(&self, key: &str) -> bool {
        self.ops.iter().any(|op| op.key() == key)
    }

    fn overwrites(&self, key: &str) -> bool {
        self.ops.iter().any(|op| op.key() == key && !op.commutes())
    }

    /// The base value of `key` with the branch's operations on it applied.
    fn overlay(&mut self, key: &str) -> Option<Val> {
        let mut value = self.base.value(key).cloned();
        for op in self.ops.clone().iter().filter(|op| op.key() == key) {
            match op.apply(value.as_ref()) {
                Ok(next) => value = next,
                Err(refusal) => {
                    self.error
                        .get_or_insert_with(|| format!("{key}: {refusal:?}"));
                    return None;
                }
            }
        }
        value
    }

    /// A key the branch's operations change the value of and that `key` is
    /// derived from, directly or transitively in the base's graph.
    fn changed_input_of(&mut self, key: &str) -> Option<String> {
        let mut pending: Vec<String> = self.base.inputs(key).into_iter().collect();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while let Some(input) = pending.pop() {
            if !seen.insert(input.clone()) {
                continue;
            }
            if self.operates_on(&input) && self.overlay(&input) != self.base.value(&input).cloned()
            {
                return Some(input);
            }
            pending.extend(self.base.inputs(&input));
        }
        None
    }

    fn shown(&mut self, key: &str) -> Option<Val> {
        if !self.operates_on(key) && self.changed_input_of(key).is_some() {
            return None;
        }
        self.overlay(key)
    }

    fn refuse(&mut self, call: &'static str, key: &str, why: &str) {
        self.log.events.push(Event::Refused {
            call,
            key: key.to_string(),
            why: why.to_string(),
        });
    }

    /// Append `op` unless it leaves a key the branch changes only
    /// commutatively derived from a key the branch changes; a refused
    /// operation is not kept.
    fn record(&mut self, call: &'static str, op: Op) -> bool {
        self.ops.push(op.clone());
        let operated: BTreeSet<String> = self.ops.iter().map(|op| op.key().to_string()).collect();
        for key in operated {
            if self.overwrites(&key) {
                continue;
            }
            if self.changed_input_of(&key).is_some() {
                self.ops.pop();
                self.refuse(call, op.key(), "evicted-operand");
                return false;
            }
        }
        self.log.events.push(Event::Staged(op));
        true
    }

    fn read_inputs_of(&mut self, key: &str) {
        for input in self.base.inputs(key) {
            self.reads.insert(input);
        }
    }

    fn stage_commutative(&mut self, call: &'static str, op: Op) {
        let key = op.key().to_string();
        if RESERVED_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
            return self.refuse(call, &key, "reserved-namespace");
        }
        let op = match op {
            Op::SetInsert { key, member, .. } => {
                let in_base = holds(self.base.value(&key), &member);
                Op::SetInsert {
                    key,
                    member,
                    in_base,
                }
            }
            Op::SetRemove { key, member, .. } => {
                let in_base = holds(self.base.value(&key), &member);
                Op::SetRemove {
                    key,
                    member,
                    in_base,
                }
            }
            other => other,
        };
        if !self.overwrites(&key) {
            if self.changed_input_of(&key).is_some() {
                return self.refuse(call, &key, "evicted-operand");
            }
            if self.base.value(&key).is_none() && !self.base.inputs(&key).is_empty() {
                return self.refuse(call, &key, "evicted-operand");
            }
        }
        let current = self.overlay(&key);
        if let Err(refusal) = op.apply(current.as_ref()) {
            let why = match refusal {
                super::model::OpRefusal::NotACounter => "not-a-counter",
                super::model::OpRefusal::NotASet => "not-a-set",
                super::model::OpRefusal::EmptyMember => "invalid-member",
                super::model::OpRefusal::Overflow => "overflow",
            };
            return self.refuse(call, &key, why);
        }
        if self.record(call, op) {
            self.read_inputs_of(&key);
        }
    }
}

/// Whether the set `value` holds has `member`; false for anything else.
fn holds(value: Option<&Val>, member: &str) -> bool {
    value
        .and_then(Val::as_set)
        .is_some_and(|members| members.contains(member))
}

impl View for RefView<'_> {
    fn read(&mut self, key: &str) -> Option<Val> {
        self.reads.insert(key.to_string());
        let value = self.shown(key);
        self.log.events.push(Event::Read {
            key: key.to_string(),
            value: value.clone(),
        });
        value
    }

    fn scan(&mut self, prefix: &str) -> Vec<(String, Val)> {
        let keys: BTreeSet<String> = self
            .base
            .values()
            .keys()
            .cloned()
            .chain(self.ops.iter().map(|op| op.key().to_string()))
            .filter(|key| key.starts_with(prefix))
            .collect();
        let mut entries = Vec::new();
        for key in keys {
            if let Some(value) = self.shown(&key) {
                entries.push((key, value));
            }
        }
        self.log.events.push(Event::Scan {
            prefix: prefix.to_string(),
            entries: entries.clone(),
        });
        entries
    }

    fn inputs(&mut self, key: &str) -> BTreeSet<String> {
        let inputs = self.base.inputs(key);
        self.log.events.push(Event::Inputs {
            key: key.to_string(),
            inputs: inputs.clone(),
        });
        inputs
    }

    fn live(&mut self, target: &str) -> Option<u64> {
        let generation = self.base.lifecycle.live(target).filter(|generation| {
            self.base.lifecycle.validity(target, *generation) == Some(super::model::Validity::Live)
        });
        self.log.events.push(Event::Live {
            target: target.to_string(),
            generation,
        });
        generation
    }

    fn rely(&mut self, target: &str, generation: u64) {
        match self.relied.get(target) {
            Some(relied) if *relied != generation => {
                return self.refuse("rely", target, "conflicting-reliance");
            }
            Some(_) => {}
            None => {
                self.relied.insert(target.to_string(), generation);
            }
        }
        self.log.events.push(Event::Rely {
            target: target.to_string(),
            generation,
        });
    }

    fn put(&mut self, key: &str, value: Val) {
        if RESERVED_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
            return self.refuse("put", key, "reserved-namespace");
        }
        if !self.reads.contains(key) {
            return self.refuse("put", key, "unread-target");
        }
        let op = Op::Put {
            key: key.to_string(),
            value,
        };
        if self.record("put", op) {
            self.read_inputs_of(key);
        }
    }

    fn remove(&mut self, key: &str) {
        if RESERVED_PREFIXES
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
            return self.refuse("remove", key, "reserved-namespace");
        }
        if !self.reads.contains(key) {
            return self.refuse("remove", key, "unread-target");
        }
        if !self.base.inputs(key).is_empty() {
            return self.refuse("remove", key, "derived-removal");
        }
        self.record(
            "remove",
            Op::Remove {
                key: key.to_string(),
            },
        );
    }

    fn add(&mut self, key: &str, amount: i64) {
        self.stage_commutative(
            "add",
            Op::Add {
                key: key.to_string(),
                amount,
            },
        );
    }

    fn set_insert(&mut self, key: &str, member: &str) {
        self.stage_commutative(
            "set_insert",
            Op::SetInsert {
                key: key.to_string(),
                member: member.to_string(),
                in_base: false,
            },
        );
    }

    fn set_remove(&mut self, key: &str, member: &str) {
        self.stage_commutative(
            "set_remove",
            Op::SetRemove {
                key: key.to_string(),
                member: member.to_string(),
                in_base: false,
            },
        );
    }
}

/// A model value as the runtime's semantic value.
pub fn to_semantic(value: &Val) -> SemanticValue {
    match value {
        Val::Text(text) => SemanticValue::Text(text.clone()),
        Val::Payload {
            type_id,
            source,
            bytes,
        } => SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from(type_id.as_str()),
            source: source.clone(),
            bytes: bytes.clone(),
        }),
    }
}

/// The runtime's semantic value as a model value.
pub fn from_semantic(value: &SemanticValue) -> Val {
    match value {
        SemanticValue::Text(text) => Val::Text(text.clone()),
        SemanticValue::Payload(payload) => Val::Payload {
            type_id: payload.type_id.0.clone(),
            source: payload.source.clone(),
            bytes: payload.bytes.clone(),
        },
    }
}

/// A model delta as the runtime's.
pub fn to_semantic_delta(delta: &Delta) -> SemanticDelta {
    SemanticDelta {
        upserts: delta
            .upserts
            .iter()
            .map(|(key, value)| (key.clone(), to_semantic(value)))
            .collect(),
        removals: delta.removals.clone(),
        dependencies: delta.dependencies.clone(),
    }
}

/// The runtime's delta as a model delta.
pub fn from_semantic_delta(delta: &SemanticDelta) -> Delta {
    Delta {
        upserts: delta
            .upserts
            .iter()
            .map(|(key, value)| (key.clone(), from_semantic(value)))
            .collect(),
        removals: delta.removals.clone(),
        dependencies: delta.dependencies.clone(),
    }
}

/// The refusal a branch reports, as the kind a log names it.
fn refusal_kind(error: &BranchError) -> String {
    match error {
        BranchError::ReservedNamespace { .. } => "reserved-namespace".to_string(),
        BranchError::UnreadTarget { .. } => "unread-target".to_string(),
        BranchError::DerivedRemoval { .. } => "derived-removal".to_string(),
        BranchError::EvictedOperand { .. } => "evicted-operand".to_string(),
        BranchError::NotACounter { .. } => "not-a-counter".to_string(),
        BranchError::NotASet { .. } => "not-a-set".to_string(),
        BranchError::InvalidMember { .. } => "invalid-member".to_string(),
        BranchError::CounterOverflow { .. } => "overflow".to_string(),
        BranchError::ConflictingReliance { .. } => "conflicting-reliance".to_string(),
        other => format!("other:{other:?}"),
    }
}

/// A real branch over the snapshot an agent opened, with the lifecycle
/// answers the runtime gave then.
pub struct BranchView {
    branch: Branch,
    base: SnapshotBase,
    lifecycle: BTreeMap<String, Option<u64>>,
    log: Log,
    error: Option<String>,
}

/// What a finished [`BranchView`] leaves.
pub struct Opened {
    pub sealed: SealedBranch,
    pub log: Log,
    pub footprint: Footprint,
    pub error: Option<String>,
}

impl BranchView {
    /// `lifecycle` maps each lifecycle target a program may ask about to the
    /// generation the runtime reported live and usable at open.
    pub fn open(
        id: BranchId,
        author: PrincipalId,
        base: SemanticSnapshot,
        lifecycle: BTreeMap<String, Option<u64>>,
    ) -> Self {
        Self {
            branch: Branch::open(id, author, base.clone()),
            base: SnapshotBase(base),
            lifecycle,
            log: Log::default(),
            error: None,
        }
    }

    /// Seal the branch and return it with its log and footprint.
    pub fn finish(self) -> Result<Opened, BranchError> {
        let footprint = footprint(&self.log, &self.base);
        let sealed = self.branch.seal()?;
        Ok(Opened {
            sealed,
            log: self.log,
            footprint,
            error: self.error,
        })
    }

    fn refused(&mut self, call: &'static str, key: &str, error: &BranchError) {
        self.log.events.push(Event::Refused {
            call,
            key: key.to_string(),
            why: refusal_kind(error),
        });
    }

    fn stage(&mut self, call: &'static str, op: BranchOp, staged: Op) {
        let key = op.key().to_string();
        match self.branch.stage_commutative(op) {
            Ok(()) => {
                // The branch records whether the member was in the base.
                let staged = match staged {
                    Op::SetInsert { key, member, .. } => {
                        let in_base = holds(self.base.value(&key).as_ref(), &member);
                        Op::SetInsert {
                            key,
                            member,
                            in_base,
                        }
                    }
                    Op::SetRemove { key, member, .. } => {
                        let in_base = holds(self.base.value(&key).as_ref(), &member);
                        Op::SetRemove {
                            key,
                            member,
                            in_base,
                        }
                    }
                    other => other,
                };
                self.log.events.push(Event::Staged(staged));
            }
            Err(error) => self.refused(call, &key, &error),
        }
    }
}

impl View for BranchView {
    fn read(&mut self, key: &str) -> Option<Val> {
        let value = match self.branch.read(key) {
            Ok(value) => value.as_ref().map(from_semantic),
            Err(error) => {
                self.error
                    .get_or_insert_with(|| format!("read {key}: {error:?}"));
                None
            }
        };
        self.log.events.push(Event::Read {
            key: key.to_string(),
            value: value.clone(),
        });
        value
    }

    fn scan(&mut self, prefix: &str) -> Vec<(String, Val)> {
        let entries = match self.branch.scan_prefix(prefix) {
            Ok(entries) => entries
                .iter()
                .map(|(key, value)| (key.clone(), from_semantic(value)))
                .collect(),
            Err(error) => {
                self.error
                    .get_or_insert_with(|| format!("scan {prefix}: {error:?}"));
                Vec::new()
            }
        };
        self.log.events.push(Event::Scan {
            prefix: prefix.to_string(),
            entries: entries.clone(),
        });
        entries
    }

    fn inputs(&mut self, key: &str) -> BTreeSet<String> {
        let inputs = self.base.inputs(key);
        self.log.events.push(Event::Inputs {
            key: key.to_string(),
            inputs: inputs.clone(),
        });
        inputs
    }

    fn live(&mut self, target: &str) -> Option<u64> {
        let generation = self.lifecycle.get(target).copied().flatten();
        self.log.events.push(Event::Live {
            target: target.to_string(),
            generation,
        });
        generation
    }

    fn rely(&mut self, target: &str, generation: u64) {
        match self.branch.rely_on(target, Generation(generation)) {
            Ok(()) => self.log.events.push(Event::Rely {
                target: target.to_string(),
                generation,
            }),
            Err(error) => self.refused("rely", target, &error),
        }
    }

    fn put(&mut self, key: &str, value: Val) {
        match self.branch.put(key, to_semantic(&value)) {
            Ok(()) => self.log.events.push(Event::Staged(Op::Put {
                key: key.to_string(),
                value,
            })),
            Err(error) => self.refused("put", key, &error),
        }
    }

    fn remove(&mut self, key: &str) {
        match self.branch.remove(key) {
            Ok(()) => self.log.events.push(Event::Staged(Op::Remove {
                key: key.to_string(),
            })),
            Err(error) => self.refused("remove", key, &error),
        }
    }

    fn add(&mut self, key: &str, amount: i64) {
        self.stage(
            "add",
            BranchOp::Add {
                key: key.to_string(),
                amount,
            },
            Op::Add {
                key: key.to_string(),
                amount,
            },
        );
    }

    fn set_insert(&mut self, key: &str, member: &str) {
        self.stage(
            "set_insert",
            BranchOp::SetInsert {
                key: key.to_string(),
                member: member.to_string(),
                in_base: false,
            },
            Op::SetInsert {
                key: key.to_string(),
                member: member.to_string(),
                in_base: false,
            },
        );
    }

    fn set_remove(&mut self, key: &str, member: &str) {
        self.stage(
            "set_remove",
            BranchOp::SetRemove {
                key: key.to_string(),
                member: member.to_string(),
                in_base: false,
            },
            Op::SetRemove {
                key: key.to_string(),
                member: member.to_string(),
                in_base: false,
            },
        );
    }
}

/// What a sealed branch declares against the footprint made from the log of
/// the run that built it: the same read set, scan prefixes, relied
/// generations, operations and touched keys.
pub fn sealed_differences(sealed: &SealedBranch, footprint: &Footprint) -> Vec<String> {
    let mut differences = Vec::new();
    let read_keys: BTreeSet<&str> = sealed.reads().keys().map(String::as_str).collect();
    let declared: BTreeSet<&str> = footprint.reads.keys().map(String::as_str).collect();
    if read_keys != declared {
        differences.push(format!(
            "reads: sealed {read_keys:?}, footprint {declared:?}"
        ));
    }
    let scan_keys: BTreeSet<&str> = sealed.scans().keys().map(String::as_str).collect();
    let scanned: BTreeSet<&str> = footprint.scans.keys().map(String::as_str).collect();
    if scan_keys != scanned {
        differences.push(format!(
            "scans: sealed {scan_keys:?}, footprint {scanned:?}"
        ));
    }
    let relied: BTreeMap<&str, u64> = sealed
        .relied()
        .iter()
        .map(|(target, generation)| (target.as_str(), generation.0))
        .collect();
    let declared_relied: BTreeMap<&str, u64> = footprint
        .relied
        .iter()
        .map(|(target, generation)| (target.as_str(), *generation))
        .collect();
    if relied != declared_relied {
        differences.push(format!(
            "relied: sealed {relied:?}, footprint {declared_relied:?}"
        ));
    }
    let ops: Vec<Op> = sealed.ops().iter().map(op_from_branch).collect();
    if ops != footprint.ops {
        differences.push(format!(
            "ops: sealed {ops:?}, footprint {:?}",
            footprint.ops
        ));
    }
    let touched: BTreeSet<&str> = sealed.touched_base().keys().map(String::as_str).collect();
    let declared_touched: BTreeSet<&str> = footprint.touched.keys().map(String::as_str).collect();
    if touched != declared_touched {
        differences.push(format!(
            "touched: sealed {touched:?}, footprint {declared_touched:?}"
        ));
    }
    differences
}

fn op_from_branch(op: &BranchOp) -> Op {
    match op {
        BranchOp::Put { key, value } => Op::Put {
            key: key.clone(),
            value: from_semantic(value),
        },
        BranchOp::Remove { key } => Op::Remove { key: key.clone() },
        BranchOp::Add { key, amount } => Op::Add {
            key: key.clone(),
            amount: *amount,
        },
        BranchOp::SetInsert {
            key,
            member,
            in_base,
        } => Op::SetInsert {
            key: key.clone(),
            member: member.clone(),
            in_base: *in_base,
        },
        BranchOp::SetRemove {
            key,
            member,
            in_base,
        } => Op::SetRemove {
            key: key.clone(),
            member: member.clone(),
            in_base: *in_base,
        },
    }
}

/// One program of the workload, with its parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Program {
    Rmw {
        group: usize,
        item: usize,
        delta: i64,
    },
    WriteSkew {
        group: usize,
        first: usize,
        second: usize,
    },
    InsertCapped {
        group: usize,
        task: usize,
    },
    RemoveExtra {
        group: usize,
    },
    Audit {
        group: usize,
    },
    GroupTotal {
        group: usize,
        item: usize,
        delta: i64,
    },
    CounterAdd {
        counter: usize,
        amount: i64,
    },
    SetOp {
        set: usize,
        member: usize,
        insert: bool,
    },
    GuardedDecrement {
        counter: usize,
    },
}

fn number(value: Option<Val>) -> Option<i64> {
    value?.as_text()?.parse().ok()
}

impl Program {
    /// The program's name in the preregistered list.
    #[cfg(test)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rmw { .. } => "rmw",
            Self::WriteSkew { .. } => "write_skew",
            Self::InsertCapped { .. } => "insert_capped",
            Self::RemoveExtra { .. } => "remove_extra",
            Self::Audit { .. } => "audit",
            Self::GroupTotal { .. } => "group_total",
            Self::CounterAdd { .. } => "counter_add",
            Self::SetOp { .. } => "set_op",
            Self::GuardedDecrement { .. } => "guarded_decrement",
        }
    }

    /// Run the program against `view`. `rely` names a policy the branch also
    /// relies on, at the generation the view reports live, when it is live.
    pub fn run(&self, view: &mut impl View, rely: Option<usize>) {
        match self {
            Self::Rmw { group, item, delta } => {
                let key = keys::item(*group, *item);
                if let Some(value) = number(view.read(&key)) {
                    view.put(&key, Val::text((value + delta).to_string()));
                }
            }
            Self::WriteSkew {
                group,
                first,
                second,
            } => {
                let (a, b) = (keys::item(*group, *first), keys::item(*group, *second));
                let (va, vb) = (number(view.read(&a)), number(view.read(&b)));
                if let (Some(va), Some(vb)) = (va, vb) {
                    if va + vb >= 60 {
                        view.put(&a, Val::text((va - 1).to_string()));
                    }
                }
            }
            Self::InsertCapped { group, task } => {
                let entries = view.scan(&keys::items(*group));
                if entries.len() < params::INSERT_CAP {
                    let key = keys::extra(*group, *task);
                    view.read(&key);
                    view.put(&key, Val::text("1"));
                }
            }
            Self::RemoveExtra { group } => {
                let entries = view.scan(&keys::items(*group));
                let extras = keys::extras(*group);
                if let Some((key, _)) = entries.iter().find(|(key, _)| key.starts_with(&extras)) {
                    view.read(key);
                    view.remove(key);
                }
            }
            Self::Audit { group } => {
                let entries = view.scan(&keys::items(*group));
                let sum: i64 = entries
                    .iter()
                    .filter_map(|(_, value)| value.as_text()?.parse::<i64>().ok())
                    .sum();
                let key = keys::audit(*group);
                view.read(&key);
                view.put(&key, Val::text(format!("{},{sum}", entries.len())));
            }
            Self::GroupTotal { group, item, delta } => {
                let mut values: BTreeMap<String, i64> = BTreeMap::new();
                let mut bases: Vec<String> = (0..params::ITEMS_PER_GROUP)
                    .map(|index| keys::item(*group, index))
                    .collect();
                bases.push(keys::spare(*group));
                for key in &bases {
                    if let Some(value) = number(view.read(key)) {
                        values.insert(key.clone(), value);
                    }
                }
                let total = keys::total(*group);
                view.read(&total);
                let bumped = keys::item(*group, *item);
                if let Some(current) = values.get(&bumped).copied() {
                    view.put(&bumped, Val::text((current + delta).to_string()));
                    let inputs = view.inputs(&total);
                    let sum: i64 = inputs
                        .iter()
                        .map(|input| {
                            values.get(input).copied().unwrap_or(0)
                                + if *input == bumped { *delta } else { 0 }
                        })
                        .sum();
                    view.put(&total, Val::text(sum.to_string()));
                }
            }
            Self::CounterAdd { counter, amount } => view.add(&keys::counter(*counter), *amount),
            Self::SetOp {
                set,
                member,
                insert,
            } => {
                let (key, member) = (keys::set(*set), keys::member(*member));
                if *insert {
                    view.set_insert(&key, &member);
                } else {
                    view.set_remove(&key, &member);
                }
            }
            Self::GuardedDecrement { counter } => {
                let key = keys::counter(*counter);
                if view
                    .read(&key)
                    .and_then(|value| value.as_counter())
                    .is_some_and(|count| count >= params::GUARD_AMOUNT)
                {
                    view.add(&key, -params::GUARD_AMOUNT);
                }
            }
        }
        if let Some(policy) = rely {
            let target = keys::policy(policy);
            if let Some(generation) = view.live(&target) {
                view.rely(&target, generation);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_semdb::SemanticHost;

    /// A small workload state: one group of eight items and a spare, a total
    /// derived from the items, an audit, two counters, a set and a policy.
    fn genesis() -> Delta {
        let mut delta = Delta::default();
        for index in 0..params::ITEMS_PER_GROUP {
            delta.upserts.insert(
                keys::item(0, index),
                Val::text((10 * (index + 1)).to_string()),
            );
        }
        delta.upserts.insert(keys::spare(0), Val::text("80"));
        let inputs: BTreeSet<String> = (0..params::ITEMS_PER_GROUP)
            .map(|index| keys::item(0, index))
            .collect();
        delta.upserts.insert(keys::total(0), Val::text("360"));
        delta.dependencies.insert(keys::total(0), inputs);
        delta.upserts.insert(keys::audit(0), Val::text("9,440"));
        delta
            .upserts
            .insert(keys::counter(0), Val::counter(50, "s003/genesis"));
        delta
            .upserts
            .insert(keys::counter(1), Val::counter(5, "s003/genesis"));
        let members: BTreeSet<String> = [keys::member(1)].into();
        delta
            .upserts
            .insert(keys::set(0), Val::set(&members, "s003/genesis"));
        delta
    }

    fn model() -> Model {
        let mut model = Model::default();
        model.apply(&genesis()).expect("genesis applies");
        model.lifecycle.set_live(&keys::policy(0), 1);
        model
    }

    fn host() -> SemanticHost {
        let mut host = SemanticHost::default();
        host.apply_delta(to_semantic_delta(&genesis()))
            .expect("genesis applies");
        host
    }

    fn lifecycle() -> BTreeMap<String, Option<u64>> {
        (0..params::POLICIES)
            .map(|policy| (keys::policy(policy), (policy == 0).then_some(1)))
            .collect()
    }

    fn programs() -> Vec<(Program, Option<usize>)> {
        vec![
            (
                Program::Rmw {
                    group: 0,
                    item: 3,
                    delta: 4,
                },
                None,
            ),
            (
                Program::Rmw {
                    group: 0,
                    item: 3,
                    delta: -2,
                },
                Some(0),
            ),
            (
                Program::WriteSkew {
                    group: 0,
                    first: 4,
                    second: 5,
                },
                None,
            ),
            (
                Program::WriteSkew {
                    group: 0,
                    first: 0,
                    second: 1,
                },
                None,
            ),
            (Program::InsertCapped { group: 0, task: 7 }, None),
            (Program::RemoveExtra { group: 0 }, None),
            (Program::Audit { group: 0 }, Some(1)),
            (
                Program::GroupTotal {
                    group: 0,
                    item: 2,
                    delta: 3,
                },
                None,
            ),
            (
                Program::GroupTotal {
                    group: 0,
                    item: 7,
                    delta: -1,
                },
                Some(0),
            ),
            (
                Program::CounterAdd {
                    counter: 0,
                    amount: -7,
                },
                None,
            ),
            (
                Program::CounterAdd {
                    counter: 2,
                    amount: 3,
                },
                None,
            ),
            (
                Program::SetOp {
                    set: 0,
                    member: 1,
                    insert: true,
                },
                None,
            ),
            (
                Program::SetOp {
                    set: 0,
                    member: 2,
                    insert: true,
                },
                None,
            ),
            (
                Program::SetOp {
                    set: 0,
                    member: 1,
                    insert: false,
                },
                None,
            ),
            (Program::GuardedDecrement { counter: 0 }, None),
            (Program::GuardedDecrement { counter: 1 }, None),
        ]
    }

    fn both(program: &Program, rely: Option<usize>) -> (Opened, (Log, Footprint, Option<String>)) {
        let host = host();
        let mut branch = BranchView::open(
            BranchId::from("b"),
            PrincipalId("agent".to_string()),
            host.snapshot(),
            lifecycle(),
        );
        program.run(&mut branch, rely);
        let opened = branch.finish().expect("seals");
        let model = model();
        let mut reference = RefView::open(&model);
        program.run(&mut reference, rely);
        (opened, reference.finish())
    }

    #[test]
    fn every_program_leaves_the_same_log_and_footprint_on_a_branch_and_on_the_model() {
        for (program, rely) in programs() {
            let (opened, (log, footprint, error)) = both(&program, rely);
            assert_eq!(opened.error, None, "{program:?}");
            assert_eq!(error, None, "{program:?}");
            assert_eq!(opened.log, log, "{program:?}");
            assert_eq!(opened.footprint, footprint, "{program:?}");
            assert_eq!(
                sealed_differences(&opened.sealed, &opened.footprint),
                Vec::<String>::new(),
                "{program:?}"
            );
        }
    }

    #[test]
    fn a_program_declares_what_it_reads_scans_writes_and_relies_on() {
        let (opened, _) = both(
            &Program::GroupTotal {
                group: 0,
                item: 2,
                delta: 3,
            },
            Some(0),
        );
        let footprint = opened.footprint;
        let read: BTreeSet<&str> = footprint.reads.keys().map(String::as_str).collect();
        assert!(read.contains("total:0") && read.contains("item:0:s") && read.contains("item:0:7"));
        assert_eq!(
            footprint.written,
            ["item:0:2".to_string(), "total:0".to_string()].into()
        );
        assert!(footprint.commutative.is_empty());
        assert_eq!(footprint.relied, [("policy-0".to_string(), 1)].into());
        // The total's base input set is the eight items, not the spare.
        assert_eq!(
            footprint.touched["total:0"].inputs.len(),
            params::ITEMS_PER_GROUP
        );
        assert!(!footprint.touched["total:0"].inputs.contains("item:0:s"));
        let ops = &footprint.ops;
        assert_eq!(ops.len(), 2);
        assert_eq!(
            ops[0],
            Op::Put {
                key: "item:0:2".into(),
                value: Val::text("33")
            }
        );
        // 360 - 30 + 33: the sum of the eight inputs at base with the bump.
        assert_eq!(
            ops[1],
            Op::Put {
                key: "total:0".into(),
                value: Val::text("363")
            }
        );

        let (opened, _) = both(&Program::Audit { group: 0 }, None);
        assert_eq!(
            opened.footprint.scans.keys().collect::<Vec<_>>(),
            vec!["item:0:"]
        );
        assert_eq!(opened.footprint.scans["item:0:"].len(), 9);
        assert!(opened.footprint.reads.contains_key("audit:0"));

        let (opened, _) = both(
            &Program::CounterAdd {
                counter: 0,
                amount: -7,
            },
            None,
        );
        assert!(
            opened.footprint.reads.is_empty(),
            "a commutative operation reads nothing"
        );
        assert_eq!(opened.footprint.commutative, ["ctr:0".to_string()].into());
        assert_eq!(
            opened.footprint.touched["ctr:0"].base,
            Some(Val::counter(50, "s003/genesis"))
        );
    }

    #[test]
    fn a_view_shows_the_branchs_own_writes_and_hides_what_they_evict() {
        for reference in [true, false] {
            let model = model();
            let host = host();
            let mut branch = BranchView::open(
                BranchId::from("b"),
                PrincipalId("agent".to_string()),
                host.snapshot(),
                lifecycle(),
            );
            let mut model_view = RefView::open(&model);
            let view: &mut dyn ViewObject = if reference {
                &mut model_view
            } else {
                &mut branch
            };
            let before = view.read("total:0");
            assert_eq!(before, Some(Val::text("360")));
            view.read("item:0:3");
            view.put("item:0:3", Val::text("99"));
            assert_eq!(
                view.read("item:0:3"),
                Some(Val::text("99")),
                "own write visible"
            );
            assert_eq!(
                view.read("total:0"),
                None,
                "the derived key its change evicts reads as absent"
            );
            let entries = view.scan("item:0:");
            assert!(entries
                .iter()
                .any(|(key, value)| key == "item:0:3" && *value == Val::text("99")));
            assert_eq!(
                view.read("audit:0"),
                Some(Val::text("9,440")),
                "an unrelated key is unaffected"
            );
        }
    }

    /// Both views behind one object-safe surface.
    trait ViewObject {
        fn read(&mut self, key: &str) -> Option<Val>;
        fn scan(&mut self, prefix: &str) -> Vec<(String, Val)>;
        fn put(&mut self, key: &str, value: Val);
    }
    impl<T: View> ViewObject for T {
        fn read(&mut self, key: &str) -> Option<Val> {
            View::read(self, key)
        }
        fn scan(&mut self, prefix: &str) -> Vec<(String, Val)> {
            View::scan(self, prefix)
        }
        fn put(&mut self, key: &str, value: Val) {
            View::put(self, key, value)
        }
    }

    #[test]
    fn a_refused_staging_is_logged_alike_by_both_views() {
        fn drive(view: &mut impl View) {
            view.put("item:0:3", Val::text("1")); // never read
            view.read("total:0");
            view.remove("total:0"); // derived
            view.put("request:r1:raw", Val::text("x")); // reserved
            view.add("item:0:3", 1); // not a counter
            view.set_insert("ctr:0", "m1"); // not a set
            view.set_insert("set:0", ""); // an empty member
            view.rely("policy-0", 1);
            view.rely("policy-0", 2); // another generation of one target
        }
        let host = host();
        let mut branch = BranchView::open(
            BranchId::from("b"),
            PrincipalId("agent".to_string()),
            host.snapshot(),
            lifecycle(),
        );
        drive(&mut branch);
        let opened = branch.finish().expect("seals");
        let model = model();
        let mut reference = RefView::open(&model);
        drive(&mut reference);
        let (log, footprint, _) = reference.finish();
        assert_eq!(opened.log, log);
        let why: Vec<String> = log
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Refused { why, .. } => Some(why.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            why,
            [
                "unread-target",
                "derived-removal",
                "reserved-namespace",
                "not-a-counter",
                "not-a-set",
                "invalid-member",
                "conflicting-reliance"
            ]
        );
        assert!(footprint.ops.is_empty());
    }

    #[test]
    fn the_insert_program_stops_at_the_cap_and_the_remove_program_needs_an_extra() {
        let mut model = model();
        let mut delta = Delta::default();
        for task in 0..3 {
            delta.upserts.insert(keys::extra(0, task), Val::text("1"));
        }
        model.apply(&delta).expect("three extras");
        let mut view = RefView::open(&model);
        Program::InsertCapped { group: 0, task: 9 }.run(&mut view, None);
        let (_, footprint, _) = view.finish();
        assert!(footprint.ops.is_empty(), "twelve entries are the cap");
        let mut view = RefView::open(&model);
        Program::RemoveExtra { group: 0 }.run(&mut view, None);
        let (_, footprint, _) = view.finish();
        assert_eq!(
            footprint.ops,
            vec![Op::Remove {
                key: "item:0:x0".into()
            }],
            "the smallest extra goes"
        );
        let bare = self::model();
        let mut view = RefView::open(&bare);
        Program::RemoveExtra { group: 0 }.run(&mut view, None);
        assert!(
            view.finish().1.ops.is_empty(),
            "with no extra there is nothing to remove"
        );
    }

    #[test]
    fn the_write_skew_and_guard_conditions_decide_whether_anything_is_staged() {
        let model = model();
        let stages = |program: Program| {
            let mut view = RefView::open(&model);
            program.run(&mut view, None);
            view.finish().1.ops.len()
        };
        assert_eq!(
            stages(Program::WriteSkew {
                group: 0,
                first: 4,
                second: 5
            }),
            1,
            "50 + 60 >= 60"
        );
        assert_eq!(
            stages(Program::WriteSkew {
                group: 0,
                first: 0,
                second: 1
            }),
            0,
            "10 + 20 < 60"
        );
        assert_eq!(
            stages(Program::GuardedDecrement { counter: 0 }),
            1,
            "50 >= 10"
        );
        assert_eq!(
            stages(Program::GuardedDecrement { counter: 1 }),
            0,
            "5 < 10"
        );
    }

    #[test]
    fn the_programs_are_named_as_the_preregistration_lists_them() {
        let names: Vec<&str> = [
            Program::Rmw {
                group: 0,
                item: 0,
                delta: 1,
            },
            Program::WriteSkew {
                group: 0,
                first: 0,
                second: 1,
            },
            Program::InsertCapped { group: 0, task: 0 },
            Program::RemoveExtra { group: 0 },
            Program::Audit { group: 0 },
            Program::GroupTotal {
                group: 0,
                item: 0,
                delta: 1,
            },
            Program::CounterAdd {
                counter: 0,
                amount: 1,
            },
            Program::SetOp {
                set: 0,
                member: 0,
                insert: true,
            },
            Program::GuardedDecrement { counter: 0 },
        ]
        .iter()
        .map(Program::name)
        .collect();
        assert_eq!(names, params::PROGRAMS);
    }
}
