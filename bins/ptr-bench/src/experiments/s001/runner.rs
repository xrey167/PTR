//! The arms of S001 and the scoring of each delta.
//!
//! For each generated graph a real `SemanticHost` (arm A) and the oracle receive the same stream
//! of deltas. For each delta that is meant to be accepted:
//!
//! 1. the oracle predicts the verdict, the structural closure and whether the revision advances;
//! 2. arm A **phase 1**: `prepare_delta`; the harness reads `affected()` and the prepared view,
//!    counts the closure and revision mismatches and the stale survivors (a derived value still
//!    present whose fresh oracle value differs), evaluates the harness derivation **only** for
//!    the derived keys marked affected, in dependency order over the host's own input sets, and
//!    applies the delta; **phase 2**: the recomputed values go back in one delta, which makes
//!    SemDB check every explicit upsert against its inputs (`MissingDependency`);
//! 3. arm G recomputes **every** derived key after the delta (reading the graph from the oracle's
//!    ground truth, not from SemDB), is held equal to the oracle's independent fresh values
//!    (disagreement is a harness failure), and is the reference the host's final state is
//!    compared with, key by key.
//!
//! A delta meant to be refused (a cycle attempt, a write whose input holds nothing) is judged
//! only on the host's verdict.
//!
//! The same code runs the negative controls: a [`Mode`] other than `Mechanism` doctors the set
//! of keys arm A treats as affected, after the host has produced it.

use super::derive::{self, Kind};
use super::generator::{self, Dag, Op, OpKind, Stream};
use super::oracle::{Change, Oracle, Rule, Verdict};
use super::params::{self, Topology};
use crate::experiments::rng::Rng;
use ptr_semdb::{SemanticDelta, SemanticError, SemanticHost, SemanticValue};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// Whose affected set arm A uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// The set `SemanticHost::prepare_delta` reports.
    Mechanism,
    /// Negative control: the same set with one key whose value really changed dropped (once
    /// per graph).
    DropOne,
    /// Negative control: every key is affected.
    Everything,
}

/// The counters of a pass; every field is an integer sum over its deltas.
#[derive(Clone, Debug, Default)]
pub struct Counters {
    pub deltas: i64,
    pub affected_total: i64,
    pub structural_closure_total: i64,
    pub value_changed_total: i64,
    pub affected_in_closure: i64,
    pub value_changed_in_affected: i64,
    pub recomputed_total: i64,
    pub recomputed_changed: i64,
    pub unnecessary_invalidations: i64,
    pub missed_invalidations: i64,
    pub closure_mismatches: i64,
    pub semantic_divergences: i64,
    pub noop_revision_bumps: i64,
    pub revision_mismatches: i64,
    pub unexpected_refusals: i64,
    pub cycle_attempts: i64,
    pub cycle_refusals: i64,
    pub missing_dependency_probes: i64,
    pub missing_dependency_refusals: i64,
    pub max_closure_depth: i64,
    pub rewires: i64,
    pub removals: i64,
    pub noop_writes: i64,
    pub hub_deltas: i64,
    pub lossy_class_deltas: i64,
    pub local_class_deltas: i64,
    pub evals_global: i64,
    pub evals_incremental: i64,
    pub local_incremental: i64,
    pub local_global: i64,
    pub hub_incremental: i64,
    pub hub_global: i64,
    // Internal counters, never judged.
    pub stale_survivors: i64,
    pub unsatisfied_recomputes: i64,
    pub phase_two_refusals: i64,
    pub value_changed_deltas: i64,
    pub changed_keys: i64,
    pub lossy_evaluations: i64,
}

/// The per-graph descriptive counters.
#[derive(Clone, Debug, Default)]
pub struct ClassStats {
    pub deltas: i64,
    pub evals_global: i64,
    pub evals_incremental: i64,
    pub local_class_deltas: i64,
    pub local_incremental: i64,
    pub local_global: i64,
    pub closure_derived_total: i64,
    pub derived_keys: i64,
}

/// Wall-clock nanoseconds, never judged.
#[derive(Clone, Debug, Default)]
pub struct SizeTiming {
    pub deltas: i64,
    pub phase_one_ns: u64,
    pub phase_two_ns: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Timing {
    pub by_size: BTreeMap<usize, SizeTiming>,
    pub recompute_incremental_ns: u64,
    pub recompute_global_ns: u64,
}

/// What a pass over a set of graphs produced.
#[derive(Clone, Debug, Default)]
pub struct Pass {
    pub counters: Counters,
    pub classes: BTreeMap<String, ClassStats>,
    pub topologies: BTreeSet<Topology>,
    pub timing: Timing,
}

fn to_semantic(change: &Change) -> SemanticDelta {
    SemanticDelta {
        upserts: change
            .leaves
            .iter()
            .map(|(key, value)| (key.clone(), SemanticValue::Text(derive::text(*value))))
            .collect(),
        removals: change.removals.clone(),
        dependencies: change.inputs.clone(),
    }
}

fn rule_of(kind: Kind) -> Rule {
    match kind {
        Kind::Mix(salt) => Rule::Mix(salt),
        Kind::Min => Rule::Min,
        Kind::Parity => Rule::Parity,
    }
}

fn count(items: impl Iterator<Item = bool>) -> i64 {
    items.filter(|hit| *hit).count() as i64
}

/// Load a graph into a fresh host (in chunks, below the delta limits) and into the oracle.
fn setup(
    dag: &Dag,
    host: &mut SemanticHost,
    oracle: &mut Oracle,
) -> Result<BTreeMap<String, Kind>, String> {
    let mut values: Vec<u64> = vec![0; dag.nodes.len()];
    let mut kinds = BTreeMap::new();
    for chunk_start in (0..dag.nodes.len()).step_by(params::SETUP_CHUNK_KEYS) {
        let end = (chunk_start + params::SETUP_CHUNK_KEYS).min(dag.nodes.len());
        let mut delta = SemanticDelta::default();
        for index in chunk_start..end {
            let node = &dag.nodes[index];
            let name = generator::key(index);
            match node.kind {
                None => {
                    values[index] = node.leaf;
                    oracle.define_leaf(&name, node.leaf);
                }
                Some(kind) => {
                    let inputs: Vec<u64> = node.inputs.iter().map(|i| values[*i]).collect();
                    values[index] = derive::eval(kind, &inputs);
                    let names: Vec<String> =
                        node.inputs.iter().map(|i| generator::key(*i)).collect();
                    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                    oracle.define_derived(&name, rule_of(kind), &refs);
                    delta
                        .dependencies
                        .insert(name.clone(), names.into_iter().collect());
                    kinds.insert(name.clone(), kind);
                }
            }
            delta
                .upserts
                .insert(name, SemanticValue::Text(derive::text(values[index])));
        }
        host.apply_delta(delta)
            .map_err(|error| format!("the setup of {:?} was refused: {error}", dag.topology))?;
    }
    oracle.sync_revision(host.revision().0);
    let fresh = oracle.fresh();
    for (index, value) in values.iter().enumerate() {
        if fresh[index] != Some(*value) {
            return Err(format!(
                "the setup of {:?} disagrees with the oracle at {}",
                dag.topology,
                generator::key(index)
            ));
        }
    }
    if !oracle.rank_monotone() {
        return Err("a generated graph has an edge that does not run to a higher index".into());
    }
    Ok(kinds)
}

struct GraphRun<'a> {
    topology: Topology,
    size: usize,
    mode: Mode,
    host: SemanticHost,
    oracle: Oracle,
    kinds: BTreeMap<String, Kind>,
    fresh: Vec<Option<u64>>,
    dropped: bool,
    pass: &'a mut Pass,
}

impl GraphRun<'_> {
    fn class(&self) -> String {
        format!("{}/{}", self.topology.name(), self.size)
    }

    /// A delta that the host must refuse: `CyclicDependency`, or `MissingDependency` naming the
    /// pair the oracle names.
    fn refusal(&mut self, op: &Op) -> Result<(), String> {
        let plan = self.oracle.plan(&op.change);
        let before = self.host.revision();
        let outcome = self.host.prepare_delta(to_semantic(&op.change));
        if self.host.revision() != before {
            return Err("prepare_delta changed the host's revision".into());
        }
        let counters = &mut self.pass.counters;
        match (op.kind, &plan.verdict) {
            (OpKind::CycleAttempt, Verdict::Cyclic) => {
                counters.cycle_attempts += 1;
                if matches!(outcome, Err(SemanticError::CyclicDependency)) {
                    counters.cycle_refusals += 1;
                }
                Ok(())
            }
            (OpKind::MissingDependency, Verdict::MissingInput { derived, input }) => {
                counters.missing_dependency_probes += 1;
                if matches!(
                    &outcome,
                    Err(SemanticError::MissingDependency { derived: d, input: i }) if d == derived && i == input
                ) {
                    counters.missing_dependency_refusals += 1;
                }
                Ok(())
            }
            (kind, verdict) => Err(format!(
                "the generator's {kind:?} is {verdict:?} for the oracle: {:?}",
                op.change
            )),
        }
    }

    /// A delta that the host must accept.
    fn accepted(&mut self, op: &Op) -> Result<(), String> {
        let plan = self.oracle.plan(&op.change);
        if plan.verdict != Verdict::Valid {
            return Err(format!(
                "the generator's {:?} is {:?} for the oracle: {:?}",
                op.kind, plan.verdict, op.change
            ));
        }
        let revision_before = self.host.revision().0;
        if self.oracle.revision() != revision_before {
            return Err(format!(
                "the oracle's revision {} and the host's {revision_before} drifted apart",
                self.oracle.revision()
            ));
        }
        let started = Instant::now();
        let prepared = match self.host.prepare_delta(to_semantic(&op.change)) {
            Ok(prepared) => prepared,
            Err(_) => {
                // The host refuses a delta the oracle accepts: the oracle does not commit it,
                // so the two stay in step.
                self.pass.counters.unexpected_refusals += 1;
                return Ok(());
            }
        };
        let prepare_ns = started.elapsed().as_nanos() as u64;

        // The oracle's side of the delta.
        self.oracle.commit(&op.change, &plan);
        let fresh_after = self.oracle.fresh();
        let derived_ids = self.oracle.derived_ids();
        let derived: BTreeSet<String> = derived_ids
            .iter()
            .map(|id| self.oracle.name(*id).to_string())
            .collect();
        let mut value_changed: BTreeSet<String> = BTreeSet::new();
        for id in &derived_ids {
            let after = fresh_after[*id].ok_or_else(|| {
                format!(
                    "derived key {} has no fresh value after {:?}",
                    self.oracle.name(*id),
                    op.kind
                )
            })?;
            if let Some(Some(before)) = self.fresh.get(*id) {
                if *before != after {
                    value_changed.insert(self.oracle.name(*id).to_string());
                }
            }
        }
        let closure_derived = plan
            .closure
            .iter()
            .filter(|key| derived.contains(*key))
            .count() as i64;
        let expected_phase_one = revision_before + u64::from(plan.bumps_revision);
        let expected_phase_two = expected_phase_one + u64::from(closure_derived > 0);

        // The set of keys arm A treats as affected.
        let mut used: BTreeSet<String> = prepared.affected().clone();
        match self.mode {
            Mode::Mechanism => {}
            Mode::DropOne => {
                if !self.dropped {
                    let victim = used
                        .iter()
                        .find(|key| derived.contains(*key) && value_changed.contains(*key))
                        .cloned();
                    if let Some(victim) = victim {
                        used.remove(&victim);
                        self.dropped = true;
                    }
                }
            }
            Mode::Everything => {
                used.extend(derived.iter().cloned());
                used.extend(
                    self.oracle
                        .leaf_ids()
                        .into_iter()
                        .map(|id| self.oracle.name(id).to_string()),
                );
            }
        }
        let recomputed: BTreeSet<&String> =
            used.iter().filter(|key| derived.contains(*key)).collect();

        // Arm A, phase 1: read the prepared state, check it, recompute the affected keys.
        let view = prepared.view();
        let survivors = stale_survivors(&self.oracle, &derived_ids, &fresh_after, |name| {
            view.value(name)
        });
        let order = dependency_order(&recomputed, |name| {
            view.inputs(name).map(str::to_string).collect()
        });
        let recompute_started = Instant::now();
        let mut local: BTreeMap<&str, u64> = BTreeMap::new();
        let mut upserts: BTreeMap<String, SemanticValue> = BTreeMap::new();
        let (mut evaluated, mut unsatisfied, mut lossy) = (0i64, 0i64, 0i64);
        for name in &order {
            let Some(kind) = self.kinds.get(*name).copied() else {
                unsatisfied += 1;
                continue;
            };
            let mut values: Vec<u64> = Vec::new();
            let mut complete = true;
            for input in view.inputs(name) {
                let stored = match view.value(input) {
                    Some(SemanticValue::Text(text)) => derive::parse(text),
                    _ => None,
                };
                match local.get(input).copied().or(stored) {
                    Some(value) => values.push(value),
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete || values.is_empty() {
                unsatisfied += 1;
                continue;
            }
            let value = derive::eval(kind, &values);
            local.insert(*name, value);
            lossy += i64::from(kind.is_lossy());
            upserts.insert(
                (*name).to_string(),
                SemanticValue::Text(derive::text(value)),
            );
            evaluated += 1;
        }
        let recompute_ns = recompute_started.elapsed().as_nanos() as u64;
        drop(local);

        let apply_started = Instant::now();
        let (revision_one, _) = self
            .host
            .apply_prepared(prepared)
            .map_err(|error| format!("apply_prepared refused its own preparation: {error}"))?;
        let phase_one_ns = prepare_ns + apply_started.elapsed().as_nanos() as u64;

        // Arm A, phase 2: the recomputed values go back as explicit upserts.
        let phase_two_started = Instant::now();
        let mut phase_two_refused = 0;
        if !upserts.is_empty() {
            let delta = SemanticDelta {
                upserts,
                ..SemanticDelta::default()
            };
            if self.host.apply_delta(delta).is_err() {
                phase_two_refused = 1;
            }
        }
        let phase_two_ns = phase_two_started.elapsed().as_nanos() as u64;
        let revision_two = self.host.revision().0;

        // Arm G: every derived key recomputed, from the oracle's graph.
        let global_started = Instant::now();
        let table = self.oracle.inputs_table();
        let mut global: Vec<Option<u64>> = vec![None; self.oracle.name_count()];
        for id in self.oracle.leaf_ids() {
            global[id] = self.oracle.leaf_value(id);
        }
        for id in &derived_ids {
            let kind = self
                .kinds
                .get(self.oracle.name(*id))
                .copied()
                .ok_or_else(|| format!("no derivation for {}", self.oracle.name(*id)))?;
            let inputs: Vec<u64> = table[*id]
                .iter()
                .map(|input| global[*input])
                .collect::<Option<Vec<u64>>>()
                .ok_or_else(|| {
                    format!(
                        "global recompute of {} reads a missing input",
                        self.oracle.name(*id)
                    )
                })?;
            let value = derive::eval(kind, &inputs);
            global[*id] = Some(value);
        }
        let global_ns = global_started.elapsed().as_nanos() as u64;
        for id in &derived_ids {
            if global[*id] != fresh_after[*id] {
                return Err(format!(
                    "the global recompute and the oracle disagree at {}: {:?} and {:?}",
                    self.oracle.name(*id),
                    global[*id],
                    fresh_after[*id]
                ));
            }
        }

        // The host's final state against arm G, key by key.
        let snapshot = self.host.snapshot();
        let (mut divergences, mut present) = (0i64, 0i64);
        for (id, expected) in global.iter().enumerate() {
            if let Some(expected) = expected {
                match snapshot.value(self.oracle.name(id)) {
                    Some(SemanticValue::Text(text)) if *text == derive::text(*expected) => {
                        present += 1
                    }
                    Some(_) => {
                        present += 1;
                        divergences += 1;
                    }
                    None => divergences += 1,
                }
            }
        }
        let extra = snapshot.keys().count() as i64 - present;
        divergences += extra;

        // Scoring.
        let missed: BTreeSet<&String> = value_changed
            .iter()
            .filter(|key| !used.contains(*key))
            .chain(survivors.iter())
            .collect();
        let unnecessary = count(recomputed.iter().map(|key| !value_changed.contains(*key)));
        let evals_global = derived_ids.len() as i64;
        let depth = self.oracle.closure_depth(&plan) as i64;
        let local_class = closure_derived >= 1
            && closure_derived as usize * 1000
                <= params::LOCAL_CLASS_MAX_CLOSURE_PERMILLE * derived_ids.len();
        let hub_class = op.kind == OpKind::HubEdit
            && self.topology == Topology::HubAndSpoke
            && closure_derived >= 1;

        let counters = &mut self.pass.counters;
        counters.deltas += 1;
        counters.affected_total += used.len() as i64;
        counters.structural_closure_total += plan.closure.len() as i64;
        counters.value_changed_total += value_changed.len() as i64;
        counters.affected_in_closure += count(used.iter().map(|key| plan.closure.contains(key)));
        counters.value_changed_in_affected +=
            count(value_changed.iter().map(|key| used.contains(key)));
        counters.recomputed_total += recomputed.len() as i64;
        counters.recomputed_changed +=
            count(recomputed.iter().map(|key| value_changed.contains(*key)));
        counters.unnecessary_invalidations += unnecessary;
        counters.missed_invalidations += missed.len() as i64;
        counters.stale_survivors += survivors.len() as i64;
        counters.changed_keys += plan.changed.len() as i64;
        counters.lossy_evaluations += lossy;
        counters.closure_mismatches += used.symmetric_difference(&plan.closure).count() as i64;
        counters.semantic_divergences += divergences;
        counters.noop_revision_bumps +=
            i64::from(!plan.bumps_revision && revision_one.0 != revision_before);
        counters.revision_mismatches += i64::from(revision_one.0 != expected_phase_one)
            + i64::from(revision_two != expected_phase_two);
        counters.unexpected_refusals += phase_two_refused;
        counters.phase_two_refusals += phase_two_refused;
        counters.unsatisfied_recomputes += unsatisfied;
        counters.max_closure_depth = counters.max_closure_depth.max(depth);
        counters.rewires += op.rewires as i64;
        counters.removals += if op.kind == OpKind::Remove {
            op.change.removals.len() as i64
        } else {
            0
        };
        counters.noop_writes += i64::from(op.kind == OpKind::NoOp);
        counters.value_changed_deltas += i64::from(!value_changed.is_empty());
        counters.lossy_class_deltas += i64::from(self.topology == Topology::Lossy);
        counters.evals_global += evals_global;
        counters.evals_incremental += evaluated;
        if local_class {
            counters.local_class_deltas += 1;
            counters.local_incremental += evaluated;
            counters.local_global += evals_global;
        }
        if hub_class {
            counters.hub_deltas += 1;
            counters.hub_incremental += evaluated;
            counters.hub_global += evals_global;
        }
        let class = self.class();
        let stats = self.pass.classes.entry(class).or_default();
        stats.deltas += 1;
        stats.evals_global += evals_global;
        stats.evals_incremental += evaluated;
        stats.closure_derived_total += closure_derived;
        stats.derived_keys = evals_global;
        if local_class {
            stats.local_class_deltas += 1;
            stats.local_incremental += evaluated;
            stats.local_global += evals_global;
        }
        let timing = &mut self.pass.timing;
        let size = timing.by_size.entry(self.size).or_default();
        size.deltas += 1;
        size.phase_one_ns += phase_one_ns;
        size.phase_two_ns += phase_two_ns;
        timing.recompute_incremental_ns += recompute_ns;
        timing.recompute_global_ns += global_ns;

        self.oracle.note_phase_two(closure_derived > 0);
        self.oracle.sync_revision(self.host.revision().0);
        self.fresh = fresh_after;
        if !self.oracle.rank_monotone() {
            return Err("an edge no longer runs to a higher index".into());
        }
        Ok(())
    }
}

/// The derived keys whose value is still stored although it differs from the oracle's fresh
/// value (a missed invalidation that the affected set alone does not show).
fn stale_survivors<'a>(
    oracle: &Oracle,
    derived_ids: &[usize],
    fresh: &[Option<u64>],
    stored: impl Fn(&str) -> Option<&'a SemanticValue>,
) -> BTreeSet<String> {
    let mut survivors = BTreeSet::new();
    for id in derived_ids {
        let name = oracle.name(*id);
        if let Some(value) = stored(name) {
            let same = matches!((value, fresh[*id]), (SemanticValue::Text(text), Some(fresh)) if *text == derive::text(fresh));
            if !same {
                survivors.insert(name.to_string());
            }
        }
    }
    survivors
}

/// `keys` in an order in which every key follows its inputs that are also in `keys`
/// (peeling by in-degree over the host's own input sets).
fn dependency_order<'a>(
    keys: &BTreeSet<&'a String>,
    inputs_of: impl Fn(&str) -> Vec<String>,
) -> Vec<&'a str> {
    let mut waiting: BTreeMap<&str, usize> = BTreeMap::new();
    let mut dependents: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for key in keys.iter().copied() {
        let inside: Vec<String> = inputs_of(key)
            .into_iter()
            .filter(|i| keys.contains(i))
            .collect();
        waiting.insert(key.as_str(), inside.len());
        for input in inside {
            dependents.entry(input).or_default().push(key.as_str());
        }
    }
    let mut ready: Vec<&str> = waiting
        .iter()
        .filter(|(_, n)| **n == 0)
        .map(|(k, _)| *k)
        .collect();
    ready.reverse();
    let mut order = Vec::with_capacity(keys.len());
    while let Some(key) = ready.pop() {
        order.push(key);
        for dependent in dependents.remove(key).unwrap_or_default() {
            let slot = waiting.get_mut(dependent).expect("a dependent is waiting");
            *slot -= 1;
            if *slot == 0 {
                ready.push(dependent);
            }
        }
    }
    order
}

/// One graph: build it, load it, and run its stream.
fn run_graph(
    seed: u64,
    label: u64,
    topology: Topology,
    size: usize,
    operations: usize,
    mode: Mode,
    pass: &mut Pass,
) -> Result<(), String> {
    let mut rng = Rng::new(seed).fork(label);
    let dag = generator::build(topology, size, &mut rng.fork(1));
    let mut host = SemanticHost::default();
    let mut oracle = Oracle::new(0);
    let kinds = setup(&dag, &mut host, &mut oracle)?;
    let hub = dag
        .hub
        .or_else(|| oracle.largest_cone_leaf())
        .ok_or_else(|| "a graph has no ground key".to_string())?;
    let mut stream = Stream::new(rng.fork(2), hub, topology);
    let fresh = oracle.fresh();
    pass.topologies.insert(topology);
    let mut run = GraphRun {
        topology,
        size,
        mode,
        host,
        oracle,
        kinds,
        fresh,
        dropped: false,
        pass,
    };
    for _ in 0..operations {
        let op = stream.next_op(&run.oracle);
        match op.kind {
            OpKind::CycleAttempt | OpKind::MissingDependency => run.refusal(&op)?,
            _ => run.accepted(&op)?,
        }
    }
    Ok(())
}

/// The graphs of a pass: (topology, size, operations).
pub type Plan = Vec<(Topology, usize, usize)>;

/// The graphs of a shape, every topology at every size.
pub fn plan_of(shape: &params::Shape) -> Plan {
    let mut plan = Vec::new();
    for topology in Topology::ALL {
        for (size, operations) in shape.sizes.iter().zip(&shape.deltas) {
            plan.push((topology, *size, *operations));
        }
    }
    plan
}

/// The graph of the negative controls.
pub fn control_plan() -> Plan {
    vec![(
        Topology::Diamonds,
        params::CONTROL_SIZE,
        params::CONTROL_DELTAS,
    )]
}

/// Run every graph of `plan` for `seed` in `mode`. An `Err` is a harness failure.
pub fn run_pass(seed: u64, plan: &Plan, mode: Mode) -> Result<Pass, String> {
    let mut pass = Pass::default();
    for (index, (topology, size, operations)) in plan.iter().enumerate() {
        run_graph(
            seed,
            index as u64 + 1,
            *topology,
            *size,
            *operations,
            mode,
            &mut pass,
        )?;
    }
    Ok(pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(seed: u64, mode: Mode, plan: &Plan) -> Pass {
        run_pass(seed, plan, mode).expect("a pass")
    }

    #[test]
    fn a_stored_value_that_differs_from_the_fresh_one_is_a_stale_survivor() {
        let mut oracle = Oracle::new(0);
        oracle.define_leaf("a", 5);
        oracle.define_derived("b", Rule::Mix(1), &["a"]);
        oracle.define_derived("c", Rule::Min, &["a", "b"]);
        let fresh = oracle.fresh();
        let (b, c) = (oracle.id("b").unwrap(), oracle.id("c").unwrap());
        let right_b = SemanticValue::Text(derive::text(fresh[b].unwrap()));
        let right_c = SemanticValue::Text(derive::text(fresh[c].unwrap()));
        let stale_c = SemanticValue::Text("0".into());
        let payload = SemanticValue::Payload(ptr_semdb::SemanticPayload {
            type_id: ptr_types::TypeId::from("t"),
            source: "s".into(),
            bytes: vec![],
        });
        let ids = [b, c];
        let none = |_: &str| None;
        assert!(
            stale_survivors(&oracle, &ids, &fresh, none).is_empty(),
            "evicted values are fine"
        );
        let held = |name: &str| match name {
            "b" => Some(&right_b),
            "c" => Some(&right_c),
            _ => None,
        };
        assert!(stale_survivors(&oracle, &ids, &fresh, held).is_empty());
        let stale = |name: &str| match name {
            "b" => Some(&right_b),
            "c" => Some(&stale_c),
            _ => None,
        };
        assert_eq!(
            stale_survivors(&oracle, &ids, &fresh, stale),
            BTreeSet::from(["c".to_string()])
        );
        let wrong_kind = |name: &str| (name == "b").then_some(&payload);
        assert_eq!(
            stale_survivors(&oracle, &ids, &fresh, wrong_kind),
            BTreeSet::from(["b".to_string()])
        );
    }

    #[test]
    fn the_dependency_order_puts_every_key_after_its_inputs() {
        let names: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let keys: BTreeSet<&String> = names.iter().collect();
        let inputs = |name: &str| -> Vec<String> {
            match name {
                "a" => vec!["d".into(), "x".into()],
                "b" => vec!["a".into()],
                "c" => vec!["a".into(), "b".into()],
                _ => vec![],
            }
        };
        let order = dependency_order(&keys, inputs);
        let at = |name: &str| order.iter().position(|k| *k == name).unwrap();
        assert_eq!(order.len(), 4);
        assert!(at("d") < at("a") && at("a") < at("b") && at("b") < at("c"));
    }

    #[test]
    fn the_mechanism_matches_the_oracle_on_every_topology_with_exactly_its_closure() {
        let plan: Plan = Topology::ALL.iter().map(|t| (*t, 96, 120)).collect();
        let result = pass(5, Mode::Mechanism, &plan);
        let c = &result.counters;
        assert_eq!(result.topologies.len(), 5);
        assert_eq!(
            (
                c.missed_invalidations,
                c.closure_mismatches,
                c.semantic_divergences
            ),
            (0, 0, 0)
        );
        assert_eq!(
            (
                c.noop_revision_bumps,
                c.revision_mismatches,
                c.unexpected_refusals
            ),
            (0, 0, 0)
        );
        assert_eq!(c.cycle_attempts, c.cycle_refusals);
        assert_eq!(c.missing_dependency_probes, c.missing_dependency_refusals);
        assert_eq!(c.affected_total, c.structural_closure_total);
        assert!(c.cycle_attempts > 0 && c.missing_dependency_probes > 0 && c.deltas > 300);
        assert_eq!(c.stale_survivors, 0);
        assert!(c.evals_incremental < c.evals_global);
    }

    #[test]
    fn dropping_one_changed_key_is_counted_as_a_miss_and_leaves_the_host_behind_arm_g() {
        let plan = vec![(Topology::DeepChains, 128, 80)];
        let result = pass(5, Mode::DropOne, &plan);
        assert!(result.counters.missed_invalidations >= 1);
        assert!(result.counters.closure_mismatches >= 1);
        assert!(result.counters.semantic_divergences >= 1);
    }

    #[test]
    fn marking_everything_affected_costs_the_whole_graph_and_misses_nothing() {
        let plan = vec![(Topology::Diamonds, 128, 80)];
        let result = pass(5, Mode::Everything, &plan);
        let c = &result.counters;
        assert_eq!(c.missed_invalidations, 0);
        assert_eq!(c.semantic_divergences, 0);
        assert!(c.closure_mismatches > 0);
        assert!(c.local_class_deltas > 0);
        assert_eq!(
            c.local_incremental, c.local_global,
            "every derived key is evaluated"
        );
    }

    #[test]
    fn the_local_class_excludes_big_cones_no_ops_and_refusals() {
        // In the hub-and-spoke graph a hub edit reaches every derived key: never local.
        let plan = vec![(Topology::HubAndSpoke, 128, 150)];
        let result = pass(7, Mode::Mechanism, &plan);
        let c = &result.counters;
        assert!(c.hub_deltas > 0);
        assert!(
            c.hub_incremental * 1000 >= 800 * c.hub_global,
            "the hub's closure is nearly the whole derived set (a rewire may drop the hub as an input)"
        );
        assert!(c.local_class_deltas + c.hub_deltas + c.noop_writes <= c.deltas);
        assert!(c.local_global > 0 && c.local_incremental * 1000 <= 50 * c.local_global);
    }
}
