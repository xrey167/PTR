//! Experiment R006, the one-layer positional (shifted) reuse oracle: `ptr-bench
//! positional-reuse-oracle <seed>`. **The harness is not built yet.**
//!
//! This is the experiment-side skeleton: the preregistered parameters ([`params`]), the metric
//! list with the coverage and falsifier rules ([`metrics`]), the one-layer model pins
//! ([`pins`]) and the stubs of the cases and the matrix ([`cases`], [`matrix`]). The code under
//! test, a position-independent secondary index and a positional session in `ptr-runtime`,
//! does not exist in this tree, so running the subcommand fails with a typed "not built yet"
//! error and exit status 2. It prints no JSON and no metric: a stub result could be mistaken
//! for evidence.
//!
//! When the cases exist, one run is one seed's evidence, judged mechanically by the
//! experiment's `aggregate.py` under `criteria.toml`, in the manner of R007: one JSON object
//! on stdout, exit status 1 when a coverage prerequisite is unmet or a falsifier is tripped,
//! and 2 when the harness itself cannot run a case. R006 is `planned`; nothing here is frozen.

// The cases that fill and read most of these items are not built yet, so the parameters, the
// rule lists and the pins are read by the tests and by `criteria.toml` only; remove this when
// the harness exists.
#![allow(dead_code)]

pub mod cases;
pub mod matrix;
pub mod metrics;
pub mod params;
pub mod pins;

/// Run the experiment's harness for `seed`. Not built yet: prints the reason on stderr and
/// exits with status 2 (the harness cannot run a case).
pub fn run(seed: u64) {
    let seeds = params::declared_seeds();
    if !seeds.contains(&seed) {
        eprintln!("seed {seed} is not one of the manifest's seeds {seeds:?}; this is a dry run");
    }
    match cases::run_all(seed) {
        Ok(()) => {}
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAFT: &str = include_str!(
        "../../../../../experiments/runtime/R006-positional-reuse-oracle/DRAFT_PREREGISTRATION.md"
    );

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "positional-reuse-oracle");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(DRAFT.contains(params::ENTRYPOINT));
        assert!(params::ENTRYPOINT.contains(params::BENCHMARK));
    }

    #[test]
    fn the_draft_quotes_the_owner_decisions_verbatim_and_pins_the_selection_digest() {
        for decision in [
            params::OWNER_DECISION_2026_10_08_1_GRANULARITY,
            params::OWNER_DECISION_2026_10_08_2_MATRIX,
            params::OWNER_DECISION_2026_10_08_3_REFUSALS,
            params::OWNER_DECISION_2026_10_08_4_GO,
            params::OWNER_DECISION_2026_10_08_5_STALENESS,
            params::OWNER_DECISION_2026_10_08_6_MULTISET,
            params::OWNER_DECISION_2026_10_08_7_REUSE_UNIT,
            params::OWNER_DECISION_2026_10_08_8_DECLARED_AND_SNAPSHOT,
            params::OWNER_DECISION_2026_10_08_9_CONFIG,
        ] {
            assert!(DRAFT.contains(decision), "{decision}");
        }
        assert!(DRAFT.contains(params::SELECTION_DIGEST));
        assert!(DRAFT.contains(params::TINY_MODEL_CONFIG_DIGEST));
        assert!(DRAFT.contains("not frozen"));
    }

    #[test]
    fn the_draft_maps_every_metric_to_its_cases() {
        for name in metrics::METRIC_NAMES {
            assert!(DRAFT.contains(&format!("`{name}`")), "{name}");
        }
        for (case, _) in metrics::CASE_MAP {
            assert!(DRAFT.contains(&format!("| {case} ")), "{case}");
        }
    }

    #[test]
    fn the_harness_refuses_to_run_with_a_typed_error_and_prints_nothing_a_rule_could_read() {
        assert!(cases::run_all(17).is_err());
        assert!(matrix::run_all(17).is_err());
    }
}
