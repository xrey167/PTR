//! The seeded graphs and update streams of S001.
//!
//! A graph is a DAG of keys named in rank order (`n00000`, `n00001`, ...): every edge runs from a
//! lower to a higher index, so acyclicity holds by construction and the oracle can assert it. The
//! five topology classes:
//!
//! - **layered-fan-in**: a quarter of the keys are ground, the rest form eight layers; a key picks
//!   two or three inputs from a window of the layer below, so the cone of a ground key grows
//!   linearly with depth rather than covering the graph;
//! - **deep-chains**: chains of sixteen keys (depth 15), with occasional links into the
//!   neighbouring chain;
//! - **diamonds**: groups of one ground key and four diamonds (`top -> left, right -> bottom`,
//!   the bottom feeding the next top), depth 12;
//! - **hub-and-spoke**: one hub ground key that every derived key lists, plus spoke ground keys
//!   and an occasional edge to one of the last four derived keys; editing the hub invalidates
//!   every derived key;
//! - **lossy**: the layered shape with `Min` and `Parity` derivations and a small value range,
//!   so most single changes are absorbed a layer or two up.
//!
//! The update stream is a mix of operations (leaf edits, hub edits, rewiring, key removal,
//! no-op writes, mixed batches, cycle-creating attempts, MissingDependency probes). The stream
//! reads the oracle's model of the graph (never the host) to choose operations that are valid by
//! construction, apart from the two kinds that are meant to be refused.

use super::derive::Kind;
use super::oracle::{Change, Oracle};
use super::params::{self, Topology};
use crate::experiments::rng::Rng;
use std::collections::BTreeSet;

/// One key of a generated graph.
#[derive(Clone, Debug)]
pub struct Node {
    /// `None` for a ground key.
    pub kind: Option<Kind>,
    /// Indices of the inputs (all lower than this node's index); empty for a ground key.
    pub inputs: Vec<usize>,
    /// The initial value of a ground key.
    pub leaf: u64,
}

#[derive(Clone, Debug)]
pub struct Dag {
    pub topology: Topology,
    pub nodes: Vec<Node>,
    /// The designated hub for the hub-and-spoke class.
    pub hub: Option<usize>,
}

pub fn key(index: usize) -> String {
    format!("n{index:05}")
}

struct Builder {
    nodes: Vec<Node>,
    leaf_range: u64,
}

impl Builder {
    fn leaf(&mut self, rng: &mut Rng) -> usize {
        self.nodes.push(Node {
            kind: None,
            inputs: Vec::new(),
            leaf: rng.below(self.leaf_range),
        });
        self.nodes.len() - 1
    }

    fn derived(&mut self, kind: Kind, mut inputs: Vec<usize>) -> usize {
        inputs.sort_unstable();
        inputs.dedup();
        self.nodes.push(Node {
            kind: Some(kind),
            inputs,
            leaf: 0,
        });
        self.nodes.len() - 1
    }

    fn mix(&mut self, inputs: Vec<usize>) -> usize {
        let salt = self.nodes.len() as u64 + 1;
        self.derived(Kind::Mix(salt), inputs)
    }
}

/// `count` distinct picks from `from`.
fn distinct(rng: &mut Rng, from: &[usize], count: usize) -> Vec<usize> {
    let mut pool = from.to_vec();
    let mut out = Vec::new();
    while out.len() < count && !pool.is_empty() {
        out.push(pool.swap_remove(rng.index(pool.len())));
    }
    out
}

fn layered(size: usize, lossy: bool, rng: &mut Rng) -> Builder {
    let mut b = Builder {
        nodes: Vec::new(),
        leaf_range: if lossy {
            params::LOSSY_LEAF_VALUE_RANGE
        } else {
            params::LEAF_VALUE_RANGE
        },
    };
    let leaves = size / params::LEAF_FRACTION_DENOMINATOR;
    for _ in 0..leaves {
        b.leaf(rng);
    }
    let rest = size - leaves;
    let per = rest / params::LAYERED_DEPTH;
    let mut previous = (0usize, leaves);
    for layer in 0..params::LAYERED_DEPTH {
        let width = if layer + 1 == params::LAYERED_DEPTH {
            rest - per * (params::LAYERED_DEPTH - 1)
        } else {
            per
        };
        let start = b.nodes.len();
        for j in 0..width {
            let centre = j * previous.1 / width;
            let low = centre.saturating_sub(1);
            let high = (centre + 2).min(previous.1 - 1);
            let window: Vec<usize> = (low..=high).map(|p| previous.0 + p).collect();
            let fan_in = (2 + rng.index(2)).min(window.len());
            let inputs = distinct(rng, &window, fan_in);
            if lossy {
                let kind = if layer % 2 == 0 {
                    Kind::Min
                } else {
                    Kind::Parity
                };
                b.derived(kind, inputs);
            } else {
                b.mix(inputs);
            }
        }
        previous = (start, width);
    }
    b
}

fn chains(size: usize, rng: &mut Rng) -> Builder {
    let mut b = Builder {
        nodes: Vec::new(),
        leaf_range: params::LEAF_VALUE_RANGE,
    };
    let count = (size / params::CHAIN_NODES).max(1);
    let mut at: Vec<usize> = (0..count).map(|_| b.leaf(rng)).collect();
    for _ in 1..params::CHAIN_NODES {
        for chain in 0..count {
            let mut inputs = vec![at[chain]];
            if count > 1 && rng.below(1000) < params::CHAIN_CROSS_LINK_PERMILLE {
                inputs.push(at[(chain + 1) % count]);
            }
            at[chain] = b.mix(inputs);
        }
    }
    b
}

fn diamonds(size: usize, rng: &mut Rng) -> Builder {
    let mut b = Builder {
        nodes: Vec::new(),
        leaf_range: params::LEAF_VALUE_RANGE,
    };
    let group = 1 + 3 * params::DIAMONDS_PER_GROUP;
    for _ in 0..size / group {
        let mut top = b.leaf(rng);
        for _ in 0..params::DIAMONDS_PER_GROUP {
            let left = b.mix(vec![top]);
            let right = b.mix(vec![top]);
            top = b.mix(vec![left, right]);
        }
    }
    while b.nodes.len() < size {
        b.leaf(rng);
    }
    b
}

fn hub_and_spoke(size: usize, rng: &mut Rng) -> Builder {
    let mut b = Builder {
        nodes: Vec::new(),
        leaf_range: params::LEAF_VALUE_RANGE,
    };
    let hub = b.leaf(rng);
    let spokes: Vec<usize> = (0..size / params::SPOKE_FRACTION_DENOMINATOR)
        .map(|_| b.leaf(rng))
        .collect();
    let mut derived: Vec<usize> = Vec::new();
    while b.nodes.len() < size {
        let mut inputs = vec![hub, spokes[rng.index(spokes.len())]];
        if !derived.is_empty() && rng.below(1000) < params::HUB_EXTRA_DERIVED_INPUT_PERMILLE {
            let recent = &derived[derived.len().saturating_sub(4)..];
            inputs.push(recent[rng.index(recent.len())]);
        }
        derived.push(b.mix(inputs));
    }
    b
}

/// The graph of `topology` with `size` keys.
pub fn build(topology: Topology, size: usize, rng: &mut Rng) -> Dag {
    let builder = match topology {
        Topology::LayeredFanIn => layered(size, false, rng),
        Topology::DeepChains => chains(size, rng),
        Topology::Diamonds => diamonds(size, rng),
        Topology::HubAndSpoke => hub_and_spoke(size, rng),
        Topology::Lossy => layered(size, true, rng),
    };
    Dag {
        topology,
        nodes: builder.nodes,
        hub: (topology == Topology::HubAndSpoke).then_some(0),
    }
}

/// The kinds of operation of an update stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpKind {
    LeafEdit,
    HubEdit,
    Rewire,
    Remove,
    NoOp,
    Mixed,
    CycleAttempt,
    MissingDependency,
}

/// One operation: a delta in the oracle's vocabulary and how many rewires it holds.
#[derive(Clone, Debug)]
pub struct Op {
    pub kind: OpKind,
    pub change: Change,
    pub rewires: usize,
}

/// The update stream of one graph.
pub struct Stream {
    rng: Rng,
    hub: usize,
    leaf_range: u64,
    absent: u64,
}

impl Stream {
    pub fn new(rng: Rng, hub: usize, topology: Topology) -> Self {
        Self {
            rng,
            hub,
            leaf_range: if topology == Topology::Lossy {
                params::LOSSY_LEAF_VALUE_RANGE
            } else {
                params::LEAF_VALUE_RANGE
            },
            absent: 0,
        }
    }

    fn pick_kind(&mut self) -> OpKind {
        let mut roll = self.rng.below(1000);
        for (kind, share) in [
            (OpKind::LeafEdit, params::OP_LEAF_EDIT_PERMILLE),
            (OpKind::HubEdit, params::OP_HUB_EDIT_PERMILLE),
            (OpKind::Rewire, params::OP_REWIRE_PERMILLE),
            (OpKind::Remove, params::OP_REMOVE_PERMILLE),
            (OpKind::NoOp, params::OP_NO_OP_PERMILLE),
            (OpKind::Mixed, params::OP_MIXED_PERMILLE),
            (OpKind::CycleAttempt, params::OP_CYCLE_ATTEMPT_PERMILLE),
            (
                OpKind::MissingDependency,
                params::OP_MISSING_DEPENDENCY_PERMILLE,
            ),
        ] {
            if roll < share {
                return kind;
            }
            roll -= share;
        }
        OpKind::LeafEdit
    }

    /// A value for a ground key that differs from `current`.
    fn other_value(&mut self, current: Option<u64>) -> u64 {
        loop {
            let value = self.rng.below(self.leaf_range);
            if Some(value) != current {
                return value;
            }
        }
    }

    fn leaf_edit(&mut self, oracle: &Oracle, ids: &[usize]) -> Option<(String, u64)> {
        let id = *ids.get(self.rng.index(ids.len().max(1)))?;
        let value = self.other_value(oracle.leaf_value(id));
        Some((oracle.name(id).to_string(), value))
    }

    fn rewire(&mut self, oracle: &Oracle) -> Option<(String, BTreeSet<String>)> {
        let derived = oracle.derived_ids();
        let target = *derived.get(self.rng.index(derived.len().max(1)))?;
        let live: Vec<usize> = oracle
            .leaf_ids()
            .into_iter()
            .chain(derived.iter().copied())
            .filter(|id| *id < target)
            .collect();
        let old: BTreeSet<String> = oracle
            .inputs_of(target)
            .into_iter()
            .map(|id| oracle.name(id).to_string())
            .collect();
        for _ in 0..8 {
            let count = 1 + self.rng.index(params::REWIRE_MAX_INPUTS);
            let picked = distinct(&mut self.rng, &live, count);
            let new: BTreeSet<String> = picked
                .iter()
                .map(|id| oracle.name(*id).to_string())
                .collect();
            if !new.is_empty() && new != old {
                return Some((oracle.name(target).to_string(), new));
            }
        }
        None
    }

    fn removal(&mut self, oracle: &Oracle) -> Option<Change> {
        let derived = oracle.derived_ids();
        let leaves = oracle.leaf_ids();
        let in_use = oracle.input_ids_in_use();
        for _ in 0..30 {
            let victim = if self.rng.below(2) == 0 {
                let sinks: Vec<usize> = derived
                    .iter()
                    .copied()
                    .filter(|id| !in_use.contains(id))
                    .collect();
                *sinks.get(self.rng.index(sinks.len().max(1)))?
            } else {
                let all: Vec<usize> = leaves.iter().chain(&derived).copied().collect();
                all[self.rng.index(all.len())]
            };
            if victim == self.hub {
                continue;
            }
            let mut removed: BTreeSet<usize> = BTreeSet::from([victim]);
            let mut order = vec![victim];
            let mut cursor = 0;
            while cursor < order.len() {
                let gone = order[cursor];
                cursor += 1;
                for dependent in oracle.dependents_of(gone) {
                    if removed.contains(&dependent) {
                        continue;
                    }
                    let remaining = oracle
                        .inputs_of(dependent)
                        .into_iter()
                        .filter(|input| !removed.contains(input))
                        .count();
                    if remaining == 0 {
                        removed.insert(dependent);
                        order.push(dependent);
                    }
                }
            }
            if removed.contains(&self.hub) || removed.len() > params::REMOVAL_MAX_KEYS {
                continue;
            }
            let mut change = Change::default();
            for id in &removed {
                change.removals.insert(oracle.name(*id).to_string());
            }
            for id in &removed {
                for dependent in oracle.dependents_of(*id) {
                    if removed.contains(&dependent) {
                        continue;
                    }
                    let remaining: BTreeSet<String> = oracle
                        .inputs_of(dependent)
                        .into_iter()
                        .filter(|input| !removed.contains(input))
                        .map(|input| oracle.name(input).to_string())
                        .collect();
                    change
                        .inputs
                        .insert(oracle.name(dependent).to_string(), remaining);
                }
            }
            return Some(change);
        }
        None
    }

    fn no_op(&mut self, oracle: &Oracle) -> Change {
        let mut change = Change::default();
        match self.rng.index(4) {
            0 => {
                let leaves = oracle.leaf_ids();
                let id = leaves[self.rng.index(leaves.len())];
                let value = oracle.leaf_value(id).expect("a ground key holds a value");
                change.leaves.insert(oracle.name(id).to_string(), value);
            }
            1 => {
                let derived = oracle.derived_ids();
                let id = derived[self.rng.index(derived.len())];
                let same = oracle
                    .inputs_of(id)
                    .into_iter()
                    .map(|input| oracle.name(input).to_string())
                    .collect();
                change.inputs.insert(oracle.name(id).to_string(), same);
            }
            2 => {
                self.absent += 1;
                change.removals.insert(format!("gone:{}", self.absent));
            }
            _ => {}
        }
        change
    }

    fn cycle_attempt(&mut self, oracle: &Oracle) -> Change {
        let derived = oracle.derived_ids();
        let mut target = derived[self.rng.index(derived.len())];
        let mut closing = target;
        if self.rng.below(10) >= 3 {
            for _ in 0..20 {
                let candidate = derived[self.rng.index(derived.len())];
                let below: Vec<usize> = oracle
                    .descendants(candidate)
                    .into_iter()
                    .filter(|id| *id != candidate)
                    .collect();
                if !below.is_empty() {
                    target = candidate;
                    closing = below[self.rng.index(below.len())];
                    break;
                }
            }
        }
        let mut inputs: BTreeSet<String> = oracle
            .inputs_of(target)
            .into_iter()
            .map(|input| oracle.name(input).to_string())
            .collect();
        inputs.insert(oracle.name(closing).to_string());
        let mut change = Change::default();
        change
            .inputs
            .insert(oracle.name(target).to_string(), inputs);
        change
    }

    fn missing_dependency(&mut self, oracle: &Oracle) -> Change {
        let derived = oracle.derived_ids();
        let target = derived[self.rng.index(derived.len())];
        self.absent += 1;
        let mut inputs: BTreeSet<String> = oracle
            .inputs_of(target)
            .into_iter()
            .map(|input| oracle.name(input).to_string())
            .collect();
        inputs.insert(format!("absent:{}", self.absent));
        let mut change = Change::default();
        change.leaves.insert(oracle.name(target).to_string(), 7);
        change
            .inputs
            .insert(oracle.name(target).to_string(), inputs);
        change
    }

    /// The next operation, chosen from the oracle's model of the graph.
    pub fn next_op(&mut self, oracle: &Oracle) -> Op {
        let leaves = oracle.leaf_ids();
        let single = |change: Change, kind: OpKind, rewires: usize| Op {
            kind,
            change,
            rewires,
        };
        let leaf_edit = |this: &mut Self| {
            let (name, value) = this
                .leaf_edit(oracle, &leaves)
                .expect("a graph has ground keys");
            let mut change = Change::default();
            change.leaves.insert(name, value);
            single(change, OpKind::LeafEdit, 0)
        };
        match self.pick_kind() {
            OpKind::LeafEdit => leaf_edit(self),
            OpKind::HubEdit => {
                let mut change = Change::default();
                let value = self.other_value(oracle.leaf_value(self.hub));
                change
                    .leaves
                    .insert(oracle.name(self.hub).to_string(), value);
                single(change, OpKind::HubEdit, 0)
            }
            OpKind::Rewire => match self.rewire(oracle) {
                Some((name, inputs)) => {
                    let mut change = Change::default();
                    change.inputs.insert(name, inputs);
                    single(change, OpKind::Rewire, 1)
                }
                None => leaf_edit(self),
            },
            OpKind::Remove => match self.removal(oracle) {
                Some(change) => single(change, OpKind::Remove, 0),
                None => leaf_edit(self),
            },
            OpKind::NoOp => single(self.no_op(oracle), OpKind::NoOp, 0),
            OpKind::Mixed => {
                let mut change = Change::default();
                let count = 2 + self.rng.index(params::MIXED_MAX_LEAF_EDITS - 1);
                for id in distinct(&mut self.rng, &leaves, count) {
                    let value = self.other_value(oracle.leaf_value(id));
                    change.leaves.insert(oracle.name(id).to_string(), value);
                }
                let mut rewires = 0;
                if self.rng.below(2) == 0 {
                    if let Some((name, inputs)) = self.rewire(oracle) {
                        change.inputs.insert(name, inputs);
                        rewires = 1;
                    }
                }
                single(change, OpKind::Mixed, rewires)
            }
            OpKind::CycleAttempt => single(self.cycle_attempt(oracle), OpKind::CycleAttempt, 0),
            OpKind::MissingDependency => single(
                self.missing_dependency(oracle),
                OpKind::MissingDependency,
                0,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::s001::oracle::{Rule, Verdict};

    fn oracle_of(dag: &Dag) -> Oracle {
        let mut oracle = Oracle::new(0);
        for (index, node) in dag.nodes.iter().enumerate() {
            match node.kind {
                None => oracle.define_leaf(&key(index), node.leaf),
                Some(kind) => {
                    let rule = match kind {
                        Kind::Mix(salt) => Rule::Mix(salt),
                        Kind::Min => Rule::Min,
                        Kind::Parity => Rule::Parity,
                    };
                    let inputs: Vec<String> = node.inputs.iter().map(|i| key(*i)).collect();
                    let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
                    oracle.define_derived(&key(index), rule, &refs);
                }
            }
        }
        oracle
    }

    #[test]
    fn every_topology_builds_a_rank_ordered_dag_of_the_requested_size() {
        for topology in Topology::ALL {
            for size in [64, 256, 1024] {
                let dag = build(topology, size, &mut Rng::new(7));
                assert_eq!(dag.nodes.len(), size, "{topology:?} {size}");
                for (index, node) in dag.nodes.iter().enumerate() {
                    assert!(node.inputs.iter().all(|i| *i < index));
                    assert_eq!(node.kind.is_some(), !node.inputs.is_empty());
                }
                assert!(oracle_of(&dag).rank_monotone());
                assert!(dag.nodes.iter().any(|n| n.kind.is_some()), "{topology:?}");
            }
        }
    }

    #[test]
    fn the_classes_have_their_defining_shape() {
        let chain = build(Topology::DeepChains, 256, &mut Rng::new(1));
        assert!(
            oracle_of(&chain).descendants(0).len() >= 15,
            "a chain is 16 deep"
        );
        let hub = build(Topology::HubAndSpoke, 256, &mut Rng::new(1));
        let oracle = oracle_of(&hub);
        assert_eq!(hub.hub, Some(0));
        assert_eq!(oracle.descendants(0).len(), 1 + oracle.derived_ids().len());
        let lossy = build(Topology::Lossy, 256, &mut Rng::new(1));
        assert!(lossy
            .nodes
            .iter()
            .filter_map(|n| n.kind)
            .all(Kind::is_lossy));
        let plain = build(Topology::LayeredFanIn, 256, &mut Rng::new(1));
        assert!(plain
            .nodes
            .iter()
            .filter_map(|n| n.kind)
            .all(|k| !k.is_lossy()));
        let diamonds = build(Topology::Diamonds, 256, &mut Rng::new(1));
        let o = oracle_of(&diamonds);
        assert_eq!(
            o.descendants(0).len(),
            1 + 12,
            "one group: a leaf and twelve derived keys"
        );
    }

    #[test]
    fn a_seed_reproduces_the_graph_and_another_seed_changes_it() {
        let shape = |seed| {
            build(Topology::LayeredFanIn, 256, &mut Rng::new(seed))
                .nodes
                .iter()
                .map(|n| (n.inputs.clone(), n.leaf))
                .collect::<Vec<_>>()
        };
        assert_eq!(shape(5), shape(5));
        assert_ne!(shape(5), shape(6));
    }

    #[test]
    fn the_stream_is_valid_by_construction_except_the_two_refused_kinds() {
        for topology in Topology::ALL {
            let dag = build(topology, 256, &mut Rng::new(3));
            let mut oracle = oracle_of(&dag);
            let mut stream = Stream::new(Rng::new(9), dag.hub.unwrap_or(0), topology);
            let mut kinds = BTreeSet::new();
            for _ in 0..400 {
                let op = stream.next_op(&oracle);
                let plan = oracle.plan(&op.change);
                kinds.insert(format!("{:?}", op.kind));
                match op.kind {
                    OpKind::CycleAttempt => assert_eq!(plan.verdict, Verdict::Cyclic),
                    OpKind::MissingDependency => {
                        assert!(matches!(plan.verdict, Verdict::MissingInput { .. }))
                    }
                    OpKind::NoOp => {
                        assert_eq!(plan.verdict, Verdict::Valid);
                        assert!(!plan.bumps_revision, "{:?}", op.change);
                    }
                    _ => {
                        assert_eq!(
                            plan.verdict,
                            Verdict::Valid,
                            "{:?} {:?}",
                            op.kind,
                            op.change
                        );
                        oracle.commit(&op.change, &plan);
                        assert!(oracle.rank_monotone());
                        assert!(
                            oracle.fresh().iter().enumerate().all(|(id, v)| v.is_some()
                                || oracle.leaf_value(id).is_none() && !oracle.has_inputs(id)),
                            "no dangling derived key"
                        );
                    }
                }
            }
            assert_eq!(
                kinds.len(),
                8,
                "{topology:?}: every kind of operation is drawn"
            );
        }
    }
}
