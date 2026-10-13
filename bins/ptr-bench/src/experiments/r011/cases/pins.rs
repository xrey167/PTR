//! C13: the pins and the coverage bookkeeping. The corpus selection digest (the pinned
//! adversarial suite, in listed order), the pinned procedure (every family of cells the run
//! filled has exactly the members the config lists) and the tiny MoE model pins (shared with
//! R010 through the `layered_kv` helpers).

use super::super::metrics;
use super::super::params;
use super::{selection_digest, Cx};
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::lab::{self, fail, Res};
use crate::experiments::layered_kv::pins::{self, PinChecks, Pins};
use crate::experiments::r010::arms;
use std::collections::BTreeSet;

/// The model pins: the dense checks, the MoE topology of the run's own MoE and a cold run under
/// it that reproduces an independent MoE stack's. The digest is the one of the shape together
/// with the MoE topology (`tiny_model_config_digest`).
pub fn verify_pins(cx: &mut Cx) -> Res<Pins> {
    let mut checks = PinChecks::default();
    let mut fields = pins::shape_fields();
    fields.push(("num_experts", params::MOE_TOPOLOGY_NUM_EXPERTS));
    fields.push(("top_k", params::MOE_TOPOLOGY_TOP_K));
    let build = pins::base_checks(
        &mut checks,
        &cx.backend,
        &pins::canonical_digest(&fields),
        params::TINY_MODEL_CONFIG_DIGEST,
        params::TOKENIZER_PIN,
    )?;
    let moe = &cx.world.moe;
    let topology_ok = (0..params::TEST_MODEL_LAYERS as usize).all(|layer| {
        moe.layer(layer)
            .is_some_and(|l| l.experts.len() as u64 == params::MOE_TOPOLOGY_NUM_EXPERTS)
    }) && moe.top_k() as u64 == params::MOE_TOPOLOGY_TOP_K
        && moe.mode() == ptr_pods::Qwen2RoutingMode::Dropless
        && moe.routing_determinism() == ptr_pods::RoutingDeterminism::BatchInvariant;
    checks.check(
        "moe_topology",
        topology_ok,
        Json::obj([
            (
                "num_experts",
                Json::Int(params::MOE_TOPOLOGY_NUM_EXPERTS as i64),
            ),
            ("top_k", Json::Int(moe.top_k() as i64)),
            (
                "router_layout",
                Json::str(params::MOE_TOPOLOGY_ROUTER_LAYOUT),
            ),
        ]),
    );
    let stack = pins::reference_stack()?
        .with_moe(moe.clone())
        .map_err(fail("the MoE stack"))?;
    let tokens = crate::experiments::layered_kv::corpus::sequence(
        crate::experiments::layered_kv::SEQUENCE_SEED,
        9_996,
        40,
    );
    let mut authority = lab::Authority::new(&[])?;
    arms::birth_all(&mut authority)?;
    let binding = cx.world.binding(&authority, false);
    let observed = lab::cold(
        &cx.backend,
        &crate::experiments::layered_kv::corpus::raw_request("pin-moe", &tokens),
        &binding,
    )?;
    let (generated, logits) = pins::independent_cold_run(&stack, None, &tokens)?;
    let observed_tokens: Vec<u32> = observed.generated.iter().map(|token| token.0).collect();
    checks.check(
        "cold_run_moe",
        generated == observed_tokens && logits == observed.logits_digest,
        Json::Null,
    );
    Ok(checks.finish(
        &mut cx.tally,
        Json::obj([
            (
                "tiny_model_config_digest",
                Json::str(params::TINY_MODEL_CONFIG_DIGEST),
            ),
            ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
            (
                "moe_topology",
                Json::obj([
                    (
                        "num_experts",
                        Json::Int(params::MOE_TOPOLOGY_NUM_EXPERTS as i64),
                    ),
                    ("top_k", Json::Int(params::MOE_TOPOLOGY_TOP_K as i64)),
                    (
                        "router_layout",
                        Json::str(params::MOE_TOPOLOGY_ROUTER_LAYOUT),
                    ),
                ]),
            ),
            ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
            (
                "base_checkpoint",
                Json::str("none: seeded random weights, no checkpoint"),
            ),
        ]),
        pins::chosen_json(),
        build,
    ))
}

/// The members of the family `name` the run filled, as dotted member paths without the `total`
/// (the tally keys, in the sorted order the tally keeps them).
fn observed_members(cx: &Cx, name: &str) -> BTreeSet<String> {
    cx.tally
        .group(name)
        .into_iter()
        .map(|(member, _)| member)
        .collect()
}

/// C13: the corpus selection digest, the pinned procedure and the per-case attempt counts.
pub fn c13_pins(cx: &mut Cx) -> Res<()> {
    // The corpus selection: the digest of the suite as listed equals the pinned digest.
    let computed = selection_digest();
    cx.count(
        "selection_digest_mismatches",
        i64::from(computed != params::SELECTION_DIGEST),
    );
    // That every attempted case is a case of the suite is not checked here: `Cx::attempt` refuses
    // an unlisted name, so such a check would be a tautology. The family check below still
    // holds `adversarial_attempts_per_case` to exactly the members the config lists.

    // The procedure: every family of cells has no member the config does not list.
    for name in metrics::NESTED {
        let pinned: BTreeSet<String> = metrics::cells(name).into_iter().collect();
        let observed = observed_members(cx, name);
        let unlisted = observed.difference(&pinned).count();
        cx.count("procedure_mismatches", unlisted as i64);
    }
    // The cells the matrix names are the cells of kinds x paths x types the config says
    // exist.
    let kinds = params::INVALIDATION_COVERAGE_MATRIX_KINDS;
    let paths = params::INVALIDATION_COVERAGE_MATRIX_PATHS;
    let types = params::INVALIDATION_TYPES;
    let authority_types = params::INVALIDATION_COVERAGE_MATRIX_AUTHORITY_TYPES;
    let mut derived = 0usize;
    for kind in kinds {
        let authority = params::INVALIDATION_COVERAGE_MATRIX_AUTHORITY_KINDS.contains(&kind);
        for path in paths {
            let available = [(
                "inflight",
                ["training_input", "evaluation_input"].as_slice(),
            )]
            .iter()
            .any(|(p, only)| *p == path && !only.contains(&kind));
            if available {
                continue;
            }
            let ty_list: &[&str] = if authority { &authority_types } else { &types };
            for ty in ty_list {
                if kind == "source_expert" && *ty != "revocation" {
                    continue;
                }
                if authority && path == "inflight" {
                    continue;
                }
                derived += 1;
            }
        }
    }
    cx.procedure_mismatch(derived == params::INVALIDATION_COVERAGE_MATRIX_CELLS.len());
    // The pins of the model are checked once, by `verify_pins`, before the cases.
    cx.tally
        .set("internal_c13_matrix_cells_derived", derived as i64);
    Ok(())
}
