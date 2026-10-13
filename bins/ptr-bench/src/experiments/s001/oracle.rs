//! The independent oracle of S001. It models what a correct invalidation does, from the delta
//! alone, and shares no code and no data structure with `ptr_semdb::DependencyGraph` (it does
//! not call it, and its graph is its own edge list):
//!
//! - an edge list of `(derived, input)` pairs over interned key ids, rebuilt into reverse
//!   adjacency lists for each query, and a breadth-first walk over them for the structural
//!   closure of a change (taken over the old edges and the new edges, as the manifest defines
//!   it);
//! - cycle detection by in-degree peeling over the edges a delta would leave behind;
//! - the fresh value of every key by one bottom-up pass from the ground (leaf) values, with its
//!   own copy of the three derivation functions (`Rule`), written independently of `derive.rs`;
//! - its own ground and revision model: which keys hold a leaf value, which are derived, whether
//!   a delta changes the state (and so must advance the revision by exactly one) or is a no-op.
//!
//! Key ids are assigned in the order keys are first named. The generator names a graph's keys in
//! rank order and only ever draws an edge from a lower to a higher id, which `rank_monotone`
//! asserts; a few passes (depth, evaluation) rely on that order.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The oracle's own copy of a derivation (see `derive::Kind`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rule {
    Mix(u64),
    Min,
    Parity,
}

impl Rule {
    fn apply(self, inputs: &[u64]) -> u64 {
        match self {
            Self::Mix(salt) => {
                let mut state = salt;
                for value in inputs {
                    state = (state.rotate_left(23) ^ *value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                }
                state ^ (state >> 32)
            }
            Self::Min => inputs.iter().fold(u64::MAX, |low, value| low.min(*value)),
            Self::Parity => {
                inputs
                    .iter()
                    .fold(0u64, |sum, value| sum.wrapping_add(*value))
                    % 2
            }
        }
    }
}

/// A delta in the oracle's vocabulary: leaf values to set, keys to remove, and complete
/// replacement input sets for derived keys.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Change {
    pub leaves: BTreeMap<String, u64>,
    pub removals: BTreeSet<String>,
    pub inputs: BTreeMap<String, BTreeSet<String>>,
}

/// What the oracle says a host must do with a delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Verdict {
    Valid,
    /// A key is both removed and written.
    Conflict,
    /// The edges the delta would leave contain a cycle.
    Cyclic,
    /// A key written with a value has an input that holds nothing afterwards.
    MissingInput {
        derived: String,
        input: String,
    },
}

/// The oracle's prediction for one delta, made before it is committed.
#[derive(Clone, Debug)]
pub struct Plan {
    pub verdict: Verdict,
    /// Keys the delta changes directly: removed while present, written with another value,
    /// or given another input set.
    pub changed: BTreeSet<String>,
    /// The structural closure: `changed` and everything that transitively depends on it, over
    /// the old edges and the new edges.
    pub closure: BTreeSet<String>,
    /// Whether the delta changes the state, so that a correct host advances its revision.
    pub bumps_revision: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Oracle {
    names: Vec<String>,
    ids: HashMap<String, usize>,
    leaf: Vec<Option<u64>>,
    rule: Vec<Option<Rule>>,
    edges: Vec<(usize, usize)>,
    revision: u64,
}

impl Oracle {
    pub fn new(revision: u64) -> Self {
        Self {
            revision,
            ..Self::default()
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn intern(&mut self, name: &str) -> usize {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = self.names.len();
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        self.leaf.push(None);
        self.rule.push(None);
        id
    }

    pub fn id(&self, name: &str) -> Option<usize> {
        self.ids.get(name).copied()
    }

    pub fn name(&self, id: usize) -> &str {
        &self.names[id]
    }

    /// Setup: a ground key.
    pub fn define_leaf(&mut self, name: &str, value: u64) {
        let id = self.intern(name);
        self.leaf[id] = Some(value);
    }

    /// Setup: a derived key over keys already defined.
    pub fn define_derived(&mut self, name: &str, rule: Rule, inputs: &[&str]) {
        let id = self.intern(name);
        self.rule[id] = Some(rule);
        for input in inputs {
            let input = self.intern(input);
            self.edges.push((id, input));
        }
    }

    pub fn has_inputs(&self, id: usize) -> bool {
        self.edges.iter().any(|(derived, _)| *derived == id)
    }

    /// The ids holding a leaf value and no input set, ascending.
    pub fn leaf_ids(&self) -> Vec<usize> {
        let derived = self.derived_set();
        (0..self.names.len())
            .filter(|id| self.leaf[*id].is_some() && !derived.contains(id))
            .collect()
    }

    /// The ids with a non-empty input set, ascending.
    pub fn derived_ids(&self) -> Vec<usize> {
        self.derived_set().into_iter().collect()
    }

    fn derived_set(&self) -> BTreeSet<usize> {
        self.edges.iter().map(|(derived, _)| *derived).collect()
    }

    pub fn leaf_value(&self, id: usize) -> Option<u64> {
        self.leaf[id]
    }

    /// The inputs of `id`, in ascending key order (the order a derivation reads them in).
    pub fn inputs_of(&self, id: usize) -> Vec<usize> {
        let mut inputs: Vec<usize> = self
            .edges
            .iter()
            .filter(|(derived, _)| *derived == id)
            .map(|(_, input)| *input)
            .collect();
        inputs.sort_by(|a, b| self.names[*a].cmp(&self.names[*b]));
        inputs
    }

    /// The inputs of every key, each in ascending key order (one pass over the edges).
    pub fn inputs_table(&self) -> Vec<Vec<usize>> {
        let mut table = vec![Vec::new(); self.names.len()];
        for (derived, input) in &self.edges {
            table[*derived].push(*input);
        }
        for list in &mut table {
            list.sort_by(|a, b| self.names[*a].cmp(&self.names[*b]));
        }
        table
    }

    /// The ids some key lists as an input.
    pub fn input_ids_in_use(&self) -> BTreeSet<usize> {
        self.edges.iter().map(|(_, input)| *input).collect()
    }

    /// How many keys have been named (the length of `fresh()`).
    pub fn name_count(&self) -> usize {
        self.names.len()
    }

    /// The keys that list `id` as an input.
    pub fn dependents_of(&self, id: usize) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .edges
            .iter()
            .filter(|(_, input)| *input == id)
            .map(|(derived, _)| *derived)
            .collect();
        out.sort_unstable();
        out
    }

    /// `id` and everything that transitively depends on it, over the current edges.
    pub fn descendants(&self, id: usize) -> BTreeSet<usize> {
        let reverse = reverse_adjacency(&self.edges, self.names.len());
        walk(&reverse, [id])
    }

    /// The ground key with the largest cone (ties: the lowest id): the stream's hub when the
    /// topology has no designated one.
    pub fn largest_cone_leaf(&self) -> Option<usize> {
        let reverse = reverse_adjacency(&self.edges, self.names.len());
        let mut best: Option<(usize, usize)> = None;
        for id in self.leaf_ids() {
            let size = walk(&reverse, [id]).len();
            if best.is_none_or(|(_, most)| size > most) {
                best = Some((id, size));
            }
        }
        best.map(|(id, _)| id)
    }

    /// Setup only: adopt the host's revision after the initial deltas.
    pub fn sync_revision(&mut self, revision: u64) {
        self.revision = revision;
    }

    /// Whether every edge runs from a lower id to a higher one (the generator's invariant).
    pub fn rank_monotone(&self) -> bool {
        self.edges.iter().all(|(derived, input)| input < derived)
    }

    /// Predict a delta without committing it.
    pub fn plan(&mut self, change: &Change) -> Plan {
        let removed: BTreeSet<usize> = change.removals.iter().map(|k| self.intern(k)).collect();
        let written: BTreeMap<usize, u64> = change
            .leaves
            .iter()
            .map(|(k, v)| (self.intern(k), *v))
            .collect();
        let replaced: BTreeMap<usize, BTreeSet<usize>> = change
            .inputs
            .iter()
            .map(|(k, set)| {
                let derived = self.intern(k);
                let set = set.iter().map(|input| self.intern(input)).collect();
                (derived, set)
            })
            .collect();
        let conflict = removed
            .iter()
            .any(|id| written.contains_key(id) || replaced.contains_key(id));

        let mut changed: BTreeSet<usize> = BTreeSet::new();
        for id in &removed {
            if self.leaf[*id].is_some() || self.has_inputs(*id) {
                changed.insert(*id);
            }
        }
        for (id, value) in &written {
            if self.leaf[*id] != Some(*value) {
                changed.insert(*id);
            }
        }
        for (id, new) in &replaced {
            let old: BTreeSet<usize> = self.inputs_of(*id).into_iter().collect();
            if old != *new {
                changed.insert(*id);
            }
        }

        let mut next: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|(derived, _)| !removed.contains(derived) && !replaced.contains_key(derived))
            .copied()
            .collect();
        for (derived, set) in &replaced {
            next.extend(set.iter().map(|input| (*derived, *input)));
        }

        let verdict = if conflict {
            Verdict::Conflict
        } else if has_cycle(&next, self.names.len()) {
            Verdict::Cyclic
        } else {
            self.missing_input(&written, &removed, &next).map_or(
                Verdict::Valid,
                |(derived, input)| Verdict::MissingInput {
                    derived: self.names[derived].clone(),
                    input: self.names[input].clone(),
                },
            )
        };

        let reverse_old = reverse_adjacency(&self.edges, self.names.len());
        let reverse_new = reverse_adjacency(&next, self.names.len());
        let mut closure = walk(&reverse_old, changed.iter().copied());
        closure.extend(walk(&reverse_new, changed.iter().copied()));
        Plan {
            bumps_revision: verdict == Verdict::Valid && !changed.is_empty(),
            verdict,
            changed: changed.iter().map(|id| self.names[*id].clone()).collect(),
            closure: closure.iter().map(|id| self.names[*id].clone()).collect(),
        }
    }

    /// The first written key (in key order) with an input that holds nothing afterwards.
    fn missing_input(
        &self,
        written: &BTreeMap<usize, u64>,
        removed: &BTreeSet<usize>,
        next: &[(usize, usize)],
    ) -> Option<(usize, usize)> {
        let derived_next: BTreeSet<usize> = next.iter().map(|(derived, _)| *derived).collect();
        let holds = |id: usize| {
            !removed.contains(&id)
                && (written.contains_key(&id)
                    || self.leaf[id].is_some()
                    || derived_next.contains(&id))
        };
        let mut keys: Vec<usize> = written.keys().copied().collect();
        keys.sort_by(|a, b| self.names[*a].cmp(&self.names[*b]));
        for key in keys {
            let mut inputs: Vec<usize> = next
                .iter()
                .filter(|(derived, _)| *derived == key)
                .map(|(_, input)| *input)
                .collect();
            inputs.sort_by(|a, b| self.names[*a].cmp(&self.names[*b]));
            if let Some(input) = inputs.into_iter().find(|input| !holds(*input)) {
                return Some((key, input));
            }
        }
        None
    }

    /// Commit a delta the plan called valid.
    pub fn commit(&mut self, change: &Change, plan: &Plan) {
        assert_eq!(
            plan.verdict,
            Verdict::Valid,
            "only a valid delta is committed"
        );
        for key in &change.removals {
            let id = self.intern(key);
            self.leaf[id] = None;
            self.rule[id] = None;
            self.edges.retain(|(derived, _)| *derived != id);
        }
        for (key, value) in &change.leaves {
            let id = self.intern(key);
            self.leaf[id] = Some(*value);
        }
        for (key, set) in &change.inputs {
            let derived = self.intern(key);
            self.edges.retain(|(from, _)| *from != derived);
            for input in set {
                let input = self.intern(input);
                self.edges.push((derived, input));
            }
        }
        if plan.bumps_revision {
            self.revision += 1;
        }
    }

    /// The revision a correct host holds once the derived keys a phase-one delta evicted have
    /// been written back (phase two changes the state iff any derived key was evicted).
    pub fn note_phase_two(&mut self, evicted_derived: bool) {
        if evicted_derived {
            self.revision += 1;
        }
    }

    /// The fresh value of every key from the ground values, bottom-up; `None` for a key that
    /// holds nothing, or a derived key with an input that holds nothing.
    pub fn fresh(&self) -> Vec<Option<u64>> {
        let count = self.names.len();
        let mut inputs: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut waiting = vec![0usize; count];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); count];
        for (derived, input) in &self.edges {
            inputs[*derived].push(*input);
            waiting[*derived] += 1;
            dependents[*input].push(*derived);
        }
        for list in &mut inputs {
            list.sort_by(|a, b| self.names[*a].cmp(&self.names[*b]));
        }
        let mut value: Vec<Option<u64>> = vec![None; count];
        let mut ready: Vec<usize> = Vec::new();
        for id in 0..count {
            if waiting[id] == 0 {
                value[id] = self.leaf[id];
                ready.push(id);
            }
        }
        while let Some(id) = ready.pop() {
            for dependent in std::mem::take(&mut dependents[id]) {
                waiting[dependent] -= 1;
                if waiting[dependent] == 0 {
                    let read: Option<Vec<u64>> =
                        inputs[dependent].iter().map(|i| value[*i]).collect();
                    value[dependent] = match (read, self.rule[dependent]) {
                        (Some(read), Some(rule)) => Some(rule.apply(&read)),
                        _ => None,
                    };
                    ready.push(dependent);
                }
            }
        }
        value
    }

    /// The longest path, in edges over the current edges, from a changed key to a key of the
    /// closure.
    pub fn closure_depth(&self, plan: &Plan) -> usize {
        let mut depth: BTreeMap<usize, usize> = BTreeMap::new();
        let mut longest = 0;
        let in_closure: BTreeSet<usize> = plan.closure.iter().filter_map(|k| self.id(k)).collect();
        for id in &in_closure {
            let here = self
                .inputs_of(*id)
                .into_iter()
                .filter_map(|input| depth.get(&input).map(|d| d + 1))
                .max()
                .unwrap_or(0);
            depth.insert(*id, here);
            longest = longest.max(here);
        }
        longest
    }
}

/// `reverse[input]` lists the keys that list `input`.
fn reverse_adjacency(edges: &[(usize, usize)], count: usize) -> Vec<Vec<usize>> {
    let mut reverse = vec![Vec::new(); count];
    for (derived, input) in edges {
        reverse[*input].push(*derived);
    }
    reverse
}

/// The ids reachable from `start` (included) over `reverse`.
fn walk(reverse: &[Vec<usize>], start: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut queue: std::collections::VecDeque<usize> = start.into_iter().collect();
    while let Some(id) = queue.pop_front() {
        if seen.insert(id) {
            queue.extend(reverse[id].iter().copied());
        }
    }
    seen
}

/// Whether the graph holds a cycle, by peeling keys whose inputs are all peeled.
fn has_cycle(edges: &[(usize, usize)], count: usize) -> bool {
    let mut waiting = vec![0usize; count];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (derived, input) in edges {
        waiting[*derived] += 1;
        dependents[*input].push(*derived);
    }
    let mut ready: Vec<usize> = (0..count).filter(|id| waiting[*id] == 0).collect();
    let mut peeled = 0;
    while let Some(id) = ready.pop() {
        peeled += 1;
        for dependent in &dependents[id] {
            waiting[*dependent] -= 1;
            if waiting[*dependent] == 0 {
                ready.push(*dependent);
            }
        }
    }
    peeled != count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a, b leaves; c = Min(a, b); d = Mix(c); e = Parity(c, b); f = Mix(a) (separate cone).
    fn sample() -> Oracle {
        let mut oracle = Oracle::new(0);
        oracle.define_leaf("a", 5);
        oracle.define_leaf("b", 9);
        oracle.define_derived("c", Rule::Min, &["a", "b"]);
        oracle.define_derived("d", Rule::Mix(3), &["c"]);
        oracle.define_derived("e", Rule::Parity, &["c", "b"]);
        oracle.define_derived("f", Rule::Mix(4), &["a"]);
        oracle
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    fn leaf_change(key: &str, value: u64) -> Change {
        Change {
            leaves: [(key.to_string(), value)].into(),
            ..Change::default()
        }
    }

    #[test]
    fn fresh_values_follow_the_rules_bottom_up() {
        let oracle = sample();
        let fresh = oracle.fresh();
        let id = |name: &str| oracle.id(name).unwrap();
        assert_eq!(fresh[id("c")], Some(5));
        assert_eq!(fresh[id("d")], Some(Rule::Mix(3).apply(&[5])));
        assert_eq!(fresh[id("e")], Some((5 + 9) % 2));
        assert!(oracle.rank_monotone());
    }

    #[test]
    fn the_closure_is_transitive_includes_the_change_and_stops_at_the_cone() {
        let mut oracle = sample();
        let plan = oracle.plan(&leaf_change("b", 1));
        assert_eq!(plan.verdict, Verdict::Valid);
        assert_eq!(plan.changed, set(&["b"]));
        assert_eq!(plan.closure, set(&["b", "c", "d", "e"]));
        assert!(plan.bumps_revision);
        let plan = oracle.plan(&leaf_change("a", 1));
        assert_eq!(plan.closure, set(&["a", "c", "d", "e", "f"]));
    }

    #[test]
    fn the_closure_covers_the_old_edges_and_the_new_edges() {
        let mut oracle = sample();
        // Rewire e from (c, b) to (a): e is changed; nothing depends on e.
        let change = Change {
            inputs: [("e".to_string(), set(&["a"]))].into(),
            ..Change::default()
        };
        let plan = oracle.plan(&change);
        assert_eq!(plan.verdict, Verdict::Valid);
        assert_eq!(plan.changed, set(&["e"]));
        assert_eq!(plan.closure, set(&["e"]));
        // Removing c while d still lists it: c is changed and d follows over the old edges.
        let change = Change {
            removals: set(&["c"]),
            ..Change::default()
        };
        let plan = oracle.plan(&change);
        assert_eq!(plan.closure, set(&["c", "d", "e"]));
    }

    #[test]
    fn no_op_deltas_change_nothing_and_do_not_bump() {
        let mut oracle = sample();
        let plan = oracle.plan(&leaf_change("a", 5));
        assert!(plan.changed.is_empty() && plan.closure.is_empty() && !plan.bumps_revision);
        let same = Change {
            inputs: [("c".to_string(), set(&["a", "b"]))].into(),
            ..Change::default()
        };
        assert!(!oracle.plan(&same).bumps_revision);
        let absent = Change {
            removals: set(&["nothing-here"]),
            ..Change::default()
        };
        assert!(!oracle.plan(&absent).bumps_revision);
        assert!(!oracle.plan(&Change::default()).bumps_revision);
    }

    #[test]
    fn cycles_self_loops_conflicts_and_missing_inputs_are_told_apart() {
        let mut oracle = sample();
        let cycle = Change {
            inputs: [("a".to_string(), set(&["d"]))].into(),
            ..Change::default()
        };
        assert_eq!(oracle.plan(&cycle).verdict, Verdict::Cyclic);
        let self_loop = Change {
            inputs: [("d".to_string(), set(&["c", "d"]))].into(),
            ..Change::default()
        };
        assert_eq!(oracle.plan(&self_loop).verdict, Verdict::Cyclic);
        let conflict = Change {
            removals: set(&["a"]),
            leaves: [("a".to_string(), 1)].into(),
            ..Change::default()
        };
        assert_eq!(oracle.plan(&conflict).verdict, Verdict::Conflict);
        let missing = Change {
            leaves: [("f".to_string(), 1)].into(),
            inputs: [("f".to_string(), set(&["ghost"]))].into(),
            ..Change::default()
        };
        assert_eq!(
            oracle.plan(&missing).verdict,
            Verdict::MissingInput {
                derived: "f".into(),
                input: "ghost".into()
            }
        );
        // A plan commits nothing.
        assert!(oracle.rank_monotone());
        assert_eq!(oracle.revision(), 0);
    }

    #[test]
    fn a_committed_delta_changes_the_model_and_the_revision_by_one() {
        let mut oracle = sample();
        let before = oracle.fresh();
        let change = leaf_change("b", 8);
        let plan = oracle.plan(&change);
        oracle.commit(&change, &plan);
        assert_eq!(oracle.revision(), 1);
        let after = oracle.fresh();
        let id = |name: &str| oracle.id(name).unwrap();
        assert_ne!(before[id("e")], after[id("e")]);
        assert_eq!(
            before[id("d")],
            after[id("d")],
            "Min absorbs a larger input"
        );
        assert_eq!(before[id("f")], after[id("f")]);
        let depth_plan = oracle.plan(&leaf_change("a", 77));
        assert_eq!(
            oracle.closure_depth(&depth_plan),
            2,
            "a, then c, then d and e"
        );
    }

    #[test]
    fn removing_a_key_drops_it_and_leaves_dangling_dependents_without_a_value() {
        let mut oracle = sample();
        let change = Change {
            removals: set(&["b"]),
            ..Change::default()
        };
        let plan = oracle.plan(&change);
        oracle.commit(&change, &plan);
        let fresh = oracle.fresh();
        assert_eq!(fresh[oracle.id("b").unwrap()], None);
        assert_eq!(
            fresh[oracle.id("c").unwrap()],
            None,
            "c still lists b, which is gone"
        );
        assert_eq!(fresh[oracle.id("a").unwrap()], Some(5));
    }
}
