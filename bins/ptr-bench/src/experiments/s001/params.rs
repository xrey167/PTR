//! The preregistered parameters of S001, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. S001 is `planned`: nothing here is
//! frozen.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/semdb/S001-invalidation/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/semdb/S001-invalidation/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "invalidation";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it from
/// the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str = "cargo run --release --locked -p ptr-bench -- invalidation <seed>";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
pub const SIZES: [usize; 3] = [256, 1024, 4096];
pub const DELTAS_PER_SIZE: [usize; 3] = [150, 150, 100];
pub const LAYERED_DEPTH: usize = 8;
pub const LEAF_FRACTION_DENOMINATOR: usize = 4;
pub const CHAIN_NODES: usize = 16;
pub const CHAIN_CROSS_LINK_PERMILLE: u64 = 200;
pub const DIAMONDS_PER_GROUP: usize = 4;
pub const SPOKE_FRACTION_DENOMINATOR: usize = 8;
pub const HUB_EXTRA_DERIVED_INPUT_PERMILLE: u64 = 250;
pub const LEAF_VALUE_RANGE: u64 = 1_000_000;
pub const LOSSY_LEAF_VALUE_RANGE: u64 = 16;
pub const SETUP_CHUNK_KEYS: usize = 1024;
pub const OP_LEAF_EDIT_PERMILLE: u64 = 440;
pub const OP_HUB_EDIT_PERMILLE: u64 = 80;
pub const OP_REWIRE_PERMILLE: u64 = 110;
pub const OP_REMOVE_PERMILLE: u64 = 50;
pub const OP_NO_OP_PERMILLE: u64 = 80;
pub const OP_MIXED_PERMILLE: u64 = 120;
pub const OP_CYCLE_ATTEMPT_PERMILLE: u64 = 70;
pub const OP_MISSING_DEPENDENCY_PERMILLE: u64 = 50;
pub const MIXED_MAX_LEAF_EDITS: usize = 4;
pub const REWIRE_MAX_INPUTS: usize = 3;
pub const REMOVAL_MAX_KEYS: usize = 4;
pub const LOCAL_CLASS_MAX_CLOSURE_PERMILLE: usize = 50;
pub const CONTROL_SIZE: usize = 1024;
pub const CONTROL_DELTAS: usize = 80;
pub const MIN_DELTAS: i64 = 1000;
pub const MIN_TOPOLOGY_CLASSES: i64 = 5;
pub const MIN_MAX_CLOSURE_DEPTH: i64 = 4;
pub const MIN_REWIRES: i64 = 1;
pub const MIN_REMOVALS: i64 = 1;
pub const MIN_NOOP_WRITES: i64 = 1;
pub const MIN_CYCLE_ATTEMPTS: i64 = 1;
pub const MIN_MISSING_DEPENDENCY_PROBES: i64 = 1;
pub const MIN_HUB_DELTAS: i64 = 1;
pub const MIN_LOSSY_CLASS_DELTAS: i64 = 1;
pub const MIN_LOCAL_CLASS_DELTAS: i64 = 100;
pub const MIN_VALUE_CHANGED_TOTAL: i64 = 1;
pub const MIN_UNNECESSARY_INVALIDATIONS: i64 = 1;
pub const MIN_EVALS_GLOBAL: i64 = 1;
pub const MAX_MISSED_INVALIDATIONS: i64 = 0;
pub const MAX_CLOSURE_MISMATCHES: i64 = 0;
pub const MAX_DEPENDENCY_MISMATCHES: i64 = 0;
pub const MAX_SEMANTIC_DIVERGENCES: i64 = 0;
pub const MAX_NOOP_REVISION_BUMPS: i64 = 0;
pub const MAX_REVISION_MISMATCHES: i64 = 0;
pub const MAX_UNEXPECTED_REFUSALS: i64 = 0;
pub const MAX_LOCALITY_RATIO_PERMILLE_LOCAL: i64 = 250;

/// The five topology classes, in the order of `config.toml`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Topology {
    LayeredFanIn,
    DeepChains,
    Diamonds,
    HubAndSpoke,
    Lossy,
}

impl Topology {
    pub const ALL: [Topology; 5] = [
        Self::LayeredFanIn,
        Self::DeepChains,
        Self::Diamonds,
        Self::HubAndSpoke,
        Self::Lossy,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::LayeredFanIn => "layered-fan-in",
            Self::DeepChains => "deep-chains",
            Self::Diamonds => "diamonds",
            Self::HubAndSpoke => "hub-and-spoke",
            Self::Lossy => "lossy",
        }
    }
}

/// What a run covers: the graph sizes and the operations of each stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shape {
    pub sizes: Vec<usize>,
    pub deltas: Vec<usize>,
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: small graphs, the same topology classes and operations.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
    /// The pinned graph sizes with this many operations per graph (the mutation checker's
    /// second argument); no evidence.
    Custom(usize),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
            Self::Custom(_) => "custom",
        }
    }

    pub fn shape(self) -> Shape {
        match self {
            Self::Pinned => Shape {
                sizes: SIZES.to_vec(),
                deltas: DELTAS_PER_SIZE.to_vec(),
            },
            Self::Reduced => Shape {
                sizes: vec![64, 128],
                deltas: vec![140, 140],
            },
            Self::Custom(deltas) => Shape {
                sizes: SIZES.to_vec(),
                deltas: vec![deltas; SIZES.len()],
            },
        }
    }
}

/// One value of the compiled table, comparable with a value of `config.toml`.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Leaf {
    Int(i64),
    Bool(bool),
    Text(String),
    Ints(Vec<i64>),
    Texts(Vec<String>),
}

/// The compiled table, one entry per key of `[pinned_thresholds]`.
#[cfg(test)]
pub fn compiled() -> Vec<(&'static str, Leaf)> {
    fn int(value: impl TryInto<i64>) -> Leaf {
        Leaf::Int(value.try_into().ok().expect("fits"))
    }
    fn ints<T: Copy + TryInto<i64>>(values: &[T]) -> Leaf {
        Leaf::Ints(
            values
                .iter()
                .map(|value| (*value).try_into().ok().expect("fits"))
                .collect(),
        )
    }
    vec![
        ("seed_aggregation", Leaf::Text(SEED_AGGREGATION.into())),
        ("seeds", ints(&SEEDS)),
        ("sizes", ints(&SIZES)),
        ("deltas_per_size", ints(&DELTAS_PER_SIZE)),
        (
            "topology_classes",
            Leaf::Texts(Topology::ALL.iter().map(|t| t.name().to_string()).collect()),
        ),
        ("layered_depth", int(LAYERED_DEPTH)),
        ("leaf_fraction_denominator", int(LEAF_FRACTION_DENOMINATOR)),
        ("chain_nodes", int(CHAIN_NODES)),
        ("chain_cross_link_permille", int(CHAIN_CROSS_LINK_PERMILLE)),
        ("diamonds_per_group", int(DIAMONDS_PER_GROUP)),
        ("spoke_fraction_denominator", int(SPOKE_FRACTION_DENOMINATOR)),
        (
            "hub_extra_derived_input_permille",
            int(HUB_EXTRA_DERIVED_INPUT_PERMILLE),
        ),
        ("leaf_value_range", int(LEAF_VALUE_RANGE)),
        ("lossy_leaf_value_range", int(LOSSY_LEAF_VALUE_RANGE)),
        ("setup_chunk_keys", int(SETUP_CHUNK_KEYS)),
        ("op_leaf_edit_permille", int(OP_LEAF_EDIT_PERMILLE)),
        ("op_hub_edit_permille", int(OP_HUB_EDIT_PERMILLE)),
        ("op_rewire_permille", int(OP_REWIRE_PERMILLE)),
        ("op_remove_permille", int(OP_REMOVE_PERMILLE)),
        ("op_no_op_permille", int(OP_NO_OP_PERMILLE)),
        ("op_mixed_permille", int(OP_MIXED_PERMILLE)),
        ("op_cycle_attempt_permille", int(OP_CYCLE_ATTEMPT_PERMILLE)),
        (
            "op_missing_dependency_permille",
            int(OP_MISSING_DEPENDENCY_PERMILLE),
        ),
        ("mixed_max_leaf_edits", int(MIXED_MAX_LEAF_EDITS)),
        ("rewire_max_inputs", int(REWIRE_MAX_INPUTS)),
        ("removal_max_keys", int(REMOVAL_MAX_KEYS)),
        (
            "local_class_max_closure_permille",
            int(LOCAL_CLASS_MAX_CLOSURE_PERMILLE),
        ),
        ("control_topology", Leaf::Text("diamonds".into())),
        ("control_size", int(CONTROL_SIZE)),
        ("control_deltas", int(CONTROL_DELTAS)),
        ("min_deltas", int(MIN_DELTAS)),
        ("min_topology_classes", int(MIN_TOPOLOGY_CLASSES)),
        ("min_max_closure_depth", int(MIN_MAX_CLOSURE_DEPTH)),
        ("min_rewires", int(MIN_REWIRES)),
        ("min_removals", int(MIN_REMOVALS)),
        ("min_noop_writes", int(MIN_NOOP_WRITES)),
        ("min_cycle_attempts", int(MIN_CYCLE_ATTEMPTS)),
        (
            "min_missing_dependency_probes",
            int(MIN_MISSING_DEPENDENCY_PROBES),
        ),
        ("min_hub_deltas", int(MIN_HUB_DELTAS)),
        ("min_lossy_class_deltas", int(MIN_LOSSY_CLASS_DELTAS)),
        ("min_local_class_deltas", int(MIN_LOCAL_CLASS_DELTAS)),
        ("min_value_changed_total", int(MIN_VALUE_CHANGED_TOTAL)),
        (
            "min_unnecessary_invalidations",
            int(MIN_UNNECESSARY_INVALIDATIONS),
        ),
        ("min_evals_global", int(MIN_EVALS_GLOBAL)),
        ("max_missed_invalidations", int(MAX_MISSED_INVALIDATIONS)),
        ("max_closure_mismatches", int(MAX_CLOSURE_MISMATCHES)),
        ("max_dependency_mismatches", int(MAX_DEPENDENCY_MISMATCHES)),
        ("max_semantic_divergences", int(MAX_SEMANTIC_DIVERGENCES)),
        ("max_noop_revision_bumps", int(MAX_NOOP_REVISION_BUMPS)),
        ("max_revision_mismatches", int(MAX_REVISION_MISMATCHES)),
        ("max_unexpected_refusals", int(MAX_UNEXPECTED_REFUSALS)),
        (
            "max_locality_ratio_permille_local",
            int(MAX_LOCALITY_RATIO_PERMILLE_LOCAL),
        ),
        ("hub_class_gated", Leaf::Bool(false)),
        ("unnecessary_invalidations_gated", Leaf::Bool(false)),
        ("timing_gated", Leaf::Bool(false)),
        (
            "owner_decision_2026_10_09_locality",
            Leaf::Text("locality falsifier locality_ratio_permille_local <= 250 (local-change class, blast radius <= 5% of derived keys); hub class ungated control; wall time descriptive only; unnecessary invalidations descriptive only (no early cutoff in the design)".into()),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn leaf_of(value: &toml::Value) -> Leaf {
        match value {
            toml::Value::Integer(number) => Leaf::Int(*number),
            toml::Value::Boolean(flag) => Leaf::Bool(*flag),
            toml::Value::String(text) => Leaf::Text(text.clone()),
            toml::Value::Array(items) if items.iter().all(toml::Value::is_integer) => Leaf::Ints(
                items
                    .iter()
                    .map(|item| item.as_integer().expect("integer"))
                    .collect(),
            ),
            toml::Value::Array(items) => Leaf::Texts(
                items
                    .iter()
                    .map(|item| item.as_str().expect("string").to_string())
                    .collect(),
            ),
            other => panic!("unexpected value {other:?}"),
        }
    }

    /// The keys on which the compiled constants and `config` disagree, one reason each.
    fn drift(config: &str) -> Vec<String> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let file: BTreeMap<String, Leaf> = parsed["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]")
            .iter()
            .map(|(key, value)| (key.clone(), leaf_of(value)))
            .collect();
        let compiled: BTreeMap<String, Leaf> = compiled()
            .into_iter()
            .map(|(key, leaf)| (key.to_string(), leaf))
            .collect();
        let mut problems = Vec::new();
        for (key, value) in &file {
            match compiled.get(key) {
                None => problems.push(format!("{key}: no constant")),
                Some(constant) if constant != value => problems.push(format!("{key}: drifted")),
                Some(_) => {}
            }
        }
        for key in compiled.keys().filter(|key| !file.contains_key(*key)) {
            problems.push(format!("{key}: no key in the file"));
        }
        problems
    }

    #[test]
    fn the_compiled_constants_are_the_pinned_thresholds_of_config_toml() {
        assert_eq!(drift(CONFIG_TOML), Vec::<String>::new());
    }

    #[test]
    fn a_changed_added_or_removed_key_of_config_toml_is_drift() {
        let changed = CONFIG_TOML.replace("layered_depth = 8\n", "layered_depth = 9\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["layered_depth: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("layered_depth = 8\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["layered_depth: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_operation_mix_adds_up_and_the_manifest_seeds_are_the_constants() {
        assert_eq!(
            OP_LEAF_EDIT_PERMILLE
                + OP_HUB_EDIT_PERMILLE
                + OP_REWIRE_PERMILLE
                + OP_REMOVE_PERMILLE
                + OP_NO_OP_PERMILLE
                + OP_MIXED_PERMILLE
                + OP_CYCLE_ATTEMPT_PERMILLE
                + OP_MISSING_DEPENDENCY_PERMILLE,
            1000
        );
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|seed| seed.as_integer().expect("integer") as u64)
            .collect();
        assert_eq!(seeds, SEEDS);
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_shape() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Pinned.shape().sizes, SIZES);
        assert_ne!(Scale::Reduced.shape(), Scale::Pinned.shape());
        assert_ne!(Scale::Custom(5).shape(), Scale::Pinned.shape());
        assert_eq!(Scale::Custom(5).name(), "custom");
        assert_eq!(Scale::Pinned.shape().deltas.len(), SIZES.len());
    }
}
