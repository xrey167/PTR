//! The arms of R009: one run per adapter configuration (sharing, bitwise parity, isolation of
//! adapters and generations, revocation and the in-flight race) and the 40-cell invalidation
//! matrix on a QKV-LoRA bound at layer 2.

use super::params::{self, Config, CONFIGS};
use crate::experiments::layered_kv::corpus::{self, CHUNK_TOKENS};
use crate::experiments::layered_kv::emit::{fraction_json, Json, Tally};
use crate::experiments::layered_kv::lab::{
    self, commit_revocation, commit_supersession, compare, fail, refused_not_live, Authority, Ids,
    Res, Steps,
};
use crate::experiments::layered_kv::matrix::{
    self, CellInfo, CellRecord, Invalidation, Kind, Observed, Path, Rig, SEGMENT_KINDS,
};
use crate::experiments::layered_kv::pins;
use crate::experiments::layered_kv::{Scale, SEQUENCE_SEED};
use ptr_pods::{Qwen2Adapter, Qwen2AdapterSpec};
use ptr_runtime::{BindingRecord, BindingSuccessors, IndexUpdate, ReferenceKvFaults, StackBinding};

/// Stream of the corpus prompt of configuration `stream`.
const PROMPT_STREAM_BASE: u64 = 1_000;
/// Stream of the 16 tokens that extend a prompt (probes and in-flight requests).
const EXTRA_STREAM_BASE: u64 = 2_000;
/// Stream of the 16 tokens an aLoRA request has beyond the base request's 48.
const ALORA_TAIL_STREAM: u64 = 3_000;

const PROMPT: usize = params::PROMPT_TOKENS as usize;
const LAYERS: usize = params::TEST_MODEL_LAYERS as usize;
/// The base request of an aLoRA configuration covers three pages of the four of the adapter's.
const ALORA_BASE_TOKENS: usize = 3 * CHUNK_TOKENS;

fn prompt_of(config: &Config) -> Vec<u32> {
    corpus::sequence(SEQUENCE_SEED, PROMPT_STREAM_BASE + config.stream, PROMPT)
}

fn extra_of(config: &Config) -> Vec<u32> {
    corpus::sequence(
        SEQUENCE_SEED,
        EXTRA_STREAM_BASE + config.stream,
        CHUNK_TOKENS,
    )
}

/// `(base request tokens, adapter request tokens)` of a configuration. An aLoRA's base request
/// is the first 48 tokens of its prompt and the adapter request keeps them and appends 16
/// others, so the base prefix is shared up to the invocation position and the request is
/// longer than the cached prefix (a whole-prompt hit cannot be forked: the stack needs the
/// logits of the last prompt token).
fn requests_of(config: &Config) -> (Vec<u32>, Vec<u32>) {
    let prompt = prompt_of(config);
    if config.is_alora() {
        let mut adapter = prompt[..ALORA_BASE_TOKENS].to_vec();
        adapter.extend(corpus::sequence(
            SEQUENCE_SEED,
            ALORA_TAIL_STREAM,
            PROMPT - ALORA_BASE_TOKENS,
        ));
        (prompt[..ALORA_BASE_TOKENS].to_vec(), adapter)
    } else {
        (prompt.clone(), prompt)
    }
}

/// Every sequence the run draws, in a fixed order: what the selection digest covers.
pub fn selection_sequences() -> Vec<Vec<u32>> {
    let mut sequences = Vec::new();
    for config in &CONFIGS {
        let (base, adapter) = requests_of(config);
        sequences.push(base);
        sequences.push(adapter);
        sequences.push(extra_of(config));
    }
    for spec in SEGMENT_KINDS.iter().chain([&matrix::NEUTRAL]) {
        for offset in [0u64, 1, 7, 9, 11] {
            sequences.push(corpus::sequence(
                SEQUENCE_SEED,
                10_000 + spec.label + offset,
                16,
            ));
        }
    }
    sequences
}

pub fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &selection_sequences())
}

fn spec(config: &Config, identity: &str, generation: u64) -> Qwen2AdapterSpec {
    Qwen2AdapterSpec::pinned(identity, generation, config.kind, config.first_layer)
}

fn seeded(
    backend: &ptr_runtime::ReferenceKvReuseBackend,
    config: &Config,
    identity: &str,
    generation: u64,
) -> Res<Qwen2Adapter> {
    Qwen2Adapter::seeded(
        backend.stack_config(),
        &spec(config, identity, generation),
        params::ADAPTER_SEED,
    )
    .map_err(fail("seeding an adapter"))
}

/// Whether some `B` entry of the adapter is non-zero: a zero `B` makes the run vacuous.
fn b_is_nonzero(adapter: &Qwen2Adapter) -> bool {
    (0..LAYERS).any(|layer| {
        adapter.layer(layer).is_some_and(|pairs| {
            [&pairs.q, &pairs.k, &pairs.v, &pairs.mlp]
                .into_iter()
                .flatten()
                .any(|pair| pair.b.iter().any(|value| *value != 0.0))
        })
    })
}

fn raw(id: &str, tokens: &[u32]) -> ptr_model_api::SegmentedModelRequest {
    corpus::raw_request(id, tokens)
}

/// Run one configuration. The counters go to `tally`; the in-flight adapter request ids go
/// to `ids`. A run that cannot be non-vacuous is an `Err` (exit status 2).
pub fn config_arm(tally: &mut Tally, ids: &mut Ids, seed: u64, config: &Config) -> Res<()> {
    let name = config.name;
    let ty = config.adapter_type;
    let stack = pins::reference_stack()?;
    let backend = lab::backend()?;
    let oracle_ids = [
        params::ADAPTER_A,
        params::ADAPTER_B,
        params::ADAPTER_G,
        "lora-o",
    ];
    let mut authority = Authority::new(&oracle_ids)?;
    let adapter_a = seeded(&backend, config, params::ADAPTER_A, 1)?;
    if !b_is_nonzero(&adapter_a) {
        return Err(format!(
            "{name}: the adapter's B is zero, the run would be vacuous"
        ));
    }
    let (base_tokens, adapter_tokens) = requests_of(config);
    let binding_a = StackBinding::with_adapter(adapter_a.clone());
    // The parameters the backend reports it runs this configuration with.
    let observed = matrix::observe(&backend, &binding_a);
    let observed_ok = observed.adapter
        == Some((
            format!("{:?}", config.kind),
            config.first_layer,
            params::LORA_RANK as usize,
            2.0,
        ))
        && observed.layers == LAYERS
        && observed.moe.is_none()
        && adapter_tokens.len() == PROMPT;
    if !observed_ok {
        tally.add("procedure_mismatches", 1);
    }
    tally.set_json(
        &format!("procedure_config.{name}"),
        Json::obj([
            ("adapter_type", Json::str(ty)),
            ("kind", Json::str(format!("{:?}", config.kind))),
            ("first_layer", Json::Int(config.first_layer as i64)),
            ("prompt_tokens", Json::Int(adapter_tokens.len() as i64)),
            ("moe_topology", Json::str("none")),
            ("matches_pinned", Json::Bool(observed_ok)),
        ]),
    );

    // Non-vacuity: the adapter changes the output.
    let cold_adapter = lab::cold(
        &backend,
        &raw(&format!("{name}-cold-adapter"), &adapter_tokens),
        &binding_a,
    )?;
    let cold_base = lab::cold(
        &backend,
        &raw(&format!("{name}-cold-base"), &adapter_tokens),
        &StackBinding::base(),
    )?;
    if cold_adapter.logits_digest == cold_base.logits_digest {
        return Err(format!("{name}: the adapter does not change the logits"));
    }

    // 1. The base request fills the per-layer catalog.
    let mut base_session = backend
        .begin_session_bound(
            &lab::scope(),
            &raw(&format!("{name}-base"), &base_tokens),
            None,
            &StackBinding::base_layered(),
            &authority,
        )
        .map_err(fail("the base session"))?;
    match base_session
        .run_to_end(&authority)
        .map_err(fail("the base run"))?
    {
        ptr_runtime::KvSessionFinish::Completed(outcome) => {
            let cold = lab::cold(
                &backend,
                &raw(&format!("{name}-base"), &base_tokens),
                &StackBinding::base(),
            )?;
            compare(tally, &outcome, &cold);
        }
        ptr_runtime::KvSessionFinish::Cancelled(_) => {
            return Err(format!("{name}: the base run was cancelled"))
        }
    }
    let base_entries = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries;
    let base_chunks = base_tokens.len() / CHUNK_TOKENS;
    if base_entries != LAYERS * base_chunks {
        return Err(format!(
            "{name}: the base run holds {base_entries} entries, not {}",
            LAYERS * base_chunks
        ));
    }

    // 2. The adapter request reuses what it may.
    let adapter_request = raw(&format!("{name}-adapter"), &adapter_tokens);
    let mut adapter_session = backend
        .begin_session_bound(
            &lab::scope(),
            &adapter_request,
            None,
            &binding_a,
            &authority,
        )
        .map_err(fail("the adapter session"))?;
    tally.add(&format!("adapter_reuse_attempts_by_type.{ty}"), 1);
    if config.is_alora() {
        tally.add("alora_prefix_reuse_attempts", 1);
    }
    let adapter_outcome = match adapter_session
        .run_to_end(&authority)
        .map_err(fail("the adapter run"))?
    {
        ptr_runtime::KvSessionFinish::Completed(outcome) => *outcome,
        ptr_runtime::KvSessionFinish::Cancelled(_) => {
            return Err(format!("{name}: the adapter run was cancelled"))
        }
    };
    if !lab::same_result(&adapter_outcome, &cold_adapter) {
        tally.add("adapter_logits_bitwise_mismatch_count", 1);
    }
    let report = adapter_outcome
        .layered
        .clone()
        .ok_or_else(|| format!("{name}: no layered report"))?;
    let after_adapter = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries;
    if after_adapter - base_entries != report.inserted_layer_entries {
        tally.add("internal_entry_accounting_mismatches", 1);
    }
    tally.add(
        "adapter_entries_before_revocation",
        report.inserted_layer_entries as i64,
    );
    tally.add("adoption_mismatches", report.adoption_mismatches as i64);
    tally.add(
        "post_invocation_rows_under_base_key",
        report.post_invocation_rows_under_base_key as i64,
    );

    // 3. What is physically shared with the base request, per layer.
    let pages = adapter_tokens.len().div_ceil(CHUNK_TOKENS);
    let mut physical = [0usize; LAYERS];
    for (layer, slot) in physical.iter_mut().enumerate() {
        *slot = (0..pages)
            .map(|page| adapter_session.shared_leading_rows(&base_session, page, layer))
            .sum();
    }
    let shared: usize = physical.iter().sum();
    let total = LAYERS * PROMPT;
    let predicted: usize = (0..LAYERS)
        .map(|layer| config.predicted_rows(layer, PROMPT))
        .sum();
    for (layer, rows) in physical.iter().enumerate() {
        if *rows != config.predicted_rows(layer, PROMPT) {
            tally.add("per_layer_fraction_mismatches", 1);
        }
    }
    if report.shared_rows != shared {
        tally.add("report_physical_mismatches", 1);
    }
    tally.add(&format!("shared_kv_rows.{name}"), shared as i64);
    tally.add(&format!("total_kv_rows.{name}"), total as i64);
    tally.set_json(
        &format!("fraction.{name}"),
        fraction_json(shared as i64, total as i64, predicted as i64, total as i64),
    );
    tally.set_json(
        &format!("per_layer_shared_rows.{name}"),
        Json::ints(physical.iter().map(|rows| *rows as i64)),
    );
    if config.is_alora() {
        let boundary_page = params::ALORA_INVOCATION_POSITION_TOKENS as usize / CHUNK_TOKENS;
        let boundary: Vec<usize> = (0..LAYERS)
            .map(|layer| adapter_session.shared_leading_rows(&base_session, boundary_page, layer))
            .collect();
        if boundary
            .iter()
            .all(|rows| *rows as u64 == params::ALORA_INVOCATION_OFFSET_IN_PAGE)
            && report.boundary_rows as u64 == params::ALORA_INVOCATION_OFFSET_IN_PAGE
        {
            tally.add("alora_boundary_page_cases", 1);
        }
        tally.set_json(
            &format!("alora_boundary_page_shared_rows.{name}"),
            Json::ints(boundary.iter().map(|rows| *rows as i64)),
        );
    }

    // 4. The same claim from outside the backend: the K and V rows of independent prefills.
    let rows_base = pins::independent_prefill_rows(&stack, None, &base_tokens)?;
    let rows_adapter = pins::independent_prefill_rows(&stack, Some(&adapter_a), &adapter_tokens)?;
    let width = rows_base.kv_width;
    let comparable = base_tokens.len().min(adapter_tokens.len());
    for (layer, shared_rows) in physical.iter().enumerate() {
        let differs_at = |position: usize| -> bool {
            let range = position * width..(position + 1) * width;
            let (bk, bv) = &rows_base.layers[layer];
            let (ak, av) = &rows_adapter.layers[layer];
            [(bk, ak), (bv, av)].iter().any(|(b, a)| {
                b[range.clone()]
                    .iter()
                    .zip(&a[range.clone()])
                    .any(|(x, y)| x.to_bits() != y.to_bits())
            })
        };
        // Rows the backend shares must be bit-identical to the base prefill's.
        let mismatching = (0..(*shared_rows).min(comparable))
            .filter(|position| differs_at(*position))
            .count();
        tally.add("bitwise_mismatch_count", mismatching as i64);
        // And the layer's K/V must change exactly where the preregistration says it does.
        let changed = (0..comparable).any(differs_at);
        if changed != (layer >= config.first_changed_layer()) {
            tally.add("prediction_structure_mismatches", 1);
        }
    }

    // 5. Another adapter (same weight bits, other identity) is never served from A's entries.
    let adapter_b = adapter_a.clone().relabelled(params::ADAPTER_B, 1);
    let binding_b = StackBinding::with_adapter(adapter_b);
    let out_b = lab::run_completed(
        &backend,
        &raw(&format!("{name}-cross"), &adapter_tokens),
        &binding_b,
        &authority,
    )?;
    tally.add("cross_adapter_attempts", 1);
    let shared_b = out_b
        .layered
        .as_ref()
        .map_or(0, |layered| layered.shared_rows);
    tally.add(
        "cross_adapter_hits",
        shared_b.saturating_sub(predicted) as i64,
    );
    let cold_b = lab::cold(
        &backend,
        &raw(&format!("{name}-cross"), &adapter_tokens),
        &binding_b,
    )?;
    compare(tally, &out_b, &cold_b);

    // 6. An older generation never serves the newer one, and presented as current is refused.
    let g1 = seeded(&backend, config, params::ADAPTER_G, 1)?;
    let g2 = seeded(&backend, config, params::ADAPTER_G, 2)?;
    let binding_g1 = StackBinding::with_adapter(g1);
    let binding_g2 = StackBinding::with_adapter(g2);
    lab::run_completed(
        &backend,
        &raw(&format!("{name}-g1"), &adapter_tokens),
        &binding_g1,
        &authority,
    )?;
    // The supersession is committed but the catalog is never told (the index update is
    // dropped): generation 1's entries stay in the index, so generation 2 can only miss them
    // by its chain key, not by a tombstone. Its use-time rejections must stay at zero: it
    // never even looked the old entries up.
    let mark = authority.mark();
    commit_supersession(&mut authority.runtime, params::ADAPTER_G, 1, 2)?;
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .map_err(fail("dropping the index update"))?;
    lab::apply_events(&backend, &authority.events_since(mark))?;
    let rejections_before = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .use_time_rejections;
    let out_g2 = lab::run_completed(
        &backend,
        &raw(&format!("{name}-g2"), &adapter_tokens),
        &binding_g2,
        &authority,
    )?;
    let rejections_after = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .use_time_rejections;
    tally.add(
        "generation_chain_isolation_rejections",
        (rejections_after - rejections_before) as i64,
    );
    let shared_g2 = out_g2
        .layered
        .as_ref()
        .map_or(0, |layered| layered.shared_rows);
    tally.add(
        "old_generation_hits",
        shared_g2.saturating_sub(predicted) as i64,
    );
    let cold_g2 = lab::cold(
        &backend,
        &raw(&format!("{name}-g2"), &adapter_tokens),
        &binding_g2,
    )?;
    compare(tally, &out_g2, &cold_g2);
    backend
        .set_faults(ReferenceKvFaults::default())
        .map_err(fail("restoring the index update"))?;
    let presented = backend.begin_session_bound(
        &lab::scope(),
        &raw(&format!("{name}-g1-again"), &adapter_tokens),
        None,
        &binding_g1,
        &authority,
    );
    tally.add("old_generation_as_new_attempts", 1);
    if !refused_not_live(&presented) {
        tally.add("old_generation_hits", 1);
    }
    drop(presented);

    // 7. aLoRA with the whole prompt already cached under the base key (observation only).
    if config.is_alora() {
        lab::run_completed(
            &backend,
            &raw(&format!("{name}-full-base"), &adapter_tokens),
            &StackBinding::base_layered(),
            &authority,
        )?;
        let other = adapter_a.clone().relabelled("lora-o", 1);
        let out = lab::run_completed(
            &backend,
            &raw(&format!("{name}-full-cached"), &adapter_tokens),
            &StackBinding::with_adapter(other),
            &authority,
        )?;
        tally.set(
            "alora_full_prompt_cached_shared_rows",
            out.layered
                .as_ref()
                .map_or(0, |layered| layered.shared_rows) as i64,
        );
    }

    // 8. The in-flight race on A: the session forks A's entries, the adapter is revoked at a
    //    seeded step, the session is cancelled and its restart re-resolves to the base model.
    let mut race_tokens = adapter_tokens.clone();
    race_tokens.extend(extra_of(config));
    let race_request = raw(&format!("{name}-race"), &race_tokens);
    let session = backend
        .begin_session_bound(&lab::scope(), &race_request, None, &binding_a, &authority)
        .map_err(fail("the in-flight session"))?;
    if session.hit_tokens() == 0 {
        return Err(format!("{name}: the in-flight request forked nothing"));
    }
    let hit = session.hit_tokens();
    let total_steps = Steps::count(
        race_request.total_tokens(),
        hit,
        params::MAX_NEW_TOKENS as usize,
    );
    let race_at = corpus::scenario(seed, name).range(0, total_steps as u64) as usize;
    let mut steps = Steps { session };
    for _ in 0..race_at {
        if steps.step(&authority)? {
            return Err(format!("{name}: cancelled before the revocation"));
        }
    }
    let emitted = steps.session.generated().len();
    let mark = authority.mark();
    commit_revocation(&mut authority.runtime, params::ADAPTER_A, 1)?;
    lab::apply_events(&backend, &authority.events_since(mark))?;
    tally.add(&format!("adapter_revocations_performed_by_type.{ty}"), 1);
    tally.add("inflight_adapter_revocation_race_cases.revocation", 1);
    let cancelled = if race_at < total_steps {
        steps.step(&authority)?
    } else {
        matches!(
            steps
                .session
                .finish(&authority)
                .map_err(fail("finishing"))?,
            ptr_runtime::KvSessionFinish::Cancelled(_)
        )
    };
    let mut escaped = i64::from(!cancelled);
    escaped += (steps.session.generated().len() - emitted) as i64;
    tally.add("outputs_after_adapter_revocation", escaped);
    if cancelled {
        tally.add("inflight_adapter_requests_cancelled", 1);
        let restarted = steps
            .session
            .restart_with(&authority)
            .map_err(fail("the restart"))?;
        let cold = lab::cold(&backend, &race_request, &StackBinding::base())?;
        let equal = compare(tally, &restarted, &cold);
        if !equal {
            tally.add("adapter_restart_output_mismatches", 1);
        }
        let fresh = restarted.hit_tokens == 0
            && restarted.prefilled_tokens == race_request.total_tokens()
            && restarted.inserted_chunks == 0
            && restarted.restarted_after_cancel;
        if equal && fresh {
            tally.add("inflight_adapter_requests_restarted_from_fresh_prefill", 1);
        }
        if !(restarted.binding.adapter.is_none() && restarted.binding.adapter_revoked_to_base) {
            tally.add("restart_resolution_mismatches", 1);
        }
    }
    ids.absorb(&backend)?;
    drop(steps);

    // 9. After the revocation: base pages stay, adapter entries are not served.
    tally.add("base_pages_before_revoke", base_entries as i64);
    // Eight extra tokens, no new page boundary: the probe looks up exactly the base pages (the
    // in-flight request's own chunk at depth 80 must not shift the depths it sees).
    let mut probe_tokens = base_tokens.clone();
    probe_tokens.extend(&extra_of(config)[..8]);
    let probe = raw(&format!("{name}-base-probe"), &probe_tokens);
    let mut retained_session = backend
        .begin_session_bound(
            &lab::scope(),
            &probe,
            None,
            &StackBinding::base_layered(),
            &authority,
        )
        .map_err(fail("the base probe"))?;
    let retained = retained_session.hit_tokens() / CHUNK_TOKENS * LAYERS;
    tally.set(
        &format!("base_probe_hit_tokens.{name}"),
        retained_session.hit_tokens() as i64,
    );
    tally.add(
        "base_pages_dropped_on_revoke",
        base_entries.saturating_sub(retained) as i64,
    );
    match retained_session
        .run_to_end(&authority)
        .map_err(fail("the base probe run"))?
    {
        ptr_runtime::KvSessionFinish::Completed(outcome) => {
            let cold = lab::cold(&backend, &probe, &StackBinding::base())?;
            compare(tally, &outcome, &cold);
        }
        ptr_runtime::KvSessionFinish::Cancelled(_) => {
            return Err(format!("{name}: the base probe was cancelled"))
        }
    }
    let late = backend.begin_session_bound(
        &lab::scope(),
        &raw(&format!("{name}-late"), &race_tokens),
        None,
        &binding_a,
        &authority,
    );
    tally.add("post_revocation_lookups_on_adapter_entries", 1);
    match &late {
        Ok(session) => tally.add(
            "revoked_adapter_entry_hits",
            1 + i64::from(session.hit_tokens() > 0),
        ),
        Err(_) if refused_not_live(&late) => {}
        Err(error) => return Err(format!("{name}: the late request: {error:?}")),
    }
    drop(late);
    lab::note_scope(tally, &backend)?;
    tally.add(&format!("config_runs.{name}"), 1);
    Ok(())
}

// ---- the matrix ---------------------------------------------------------------------------

/// The matrix's binding-carried dependency: a QKV-LoRA attached at layer 2, with generation 2
/// as the successor a supersession moves to.
pub struct R009Rig;

fn matrix_adapter(generation: u64) -> Qwen2Adapter {
    let backend = lab::backend().expect("a backend");
    Qwen2Adapter::seeded(
        backend.stack_config(),
        &Qwen2AdapterSpec::pinned(
            params::MATRIX_ADAPTER,
            generation,
            ptr_pods::Qwen2AdapterKind::Qkv,
            params::MATRIX_ADAPTER_FIRST_LAYER as usize,
        ),
        params::ADAPTER_SEED,
    )
    .expect("the matrix adapter")
}

impl Rig for R009Rig {
    fn dependency_kind(&self) -> &'static str {
        "adapter_generation"
    }

    fn binding(&self, _: &Authority) -> StackBinding {
        StackBinding::with_adapter(matrix_adapter(1)).with_successors(BindingSuccessors {
            adapters: vec![matrix_adapter(2)],
            experts: Vec::new(),
        })
    }

    fn resolved_binding(&self, ty: Invalidation, _: &Authority) -> StackBinding {
        match ty {
            Invalidation::Revocation | Invalidation::Fence => StackBinding::base(),
            Invalidation::Supersession => StackBinding::with_adapter(matrix_adapter(2)),
        }
    }

    fn resolution_holds(&self, record: &BindingRecord, ty: Invalidation) -> bool {
        match ty {
            Invalidation::Revocation | Invalidation::Fence => {
                record.adapter.is_none() && record.adapter_revoked_to_base
            }
            Invalidation::Supersession => {
                record.adapter == Some((params::MATRIX_ADAPTER.to_string(), 2))
                    && record.adapter_superseded == Some((1, 2))
            }
        }
    }

    fn identity(&self) -> String {
        params::MATRIX_ADAPTER.to_string()
    }

    fn capsules(&self) -> Vec<String> {
        vec![params::MATRIX_ADAPTER.to_string()]
    }

    fn commit_live(&self, _: &mut Authority) -> Res<()> {
        Ok(())
    }

    fn commit_event(&self, authority: &mut Authority, ty: Invalidation) -> Res<()> {
        match ty {
            Invalidation::Revocation => {
                commit_revocation(&mut authority.runtime, params::MATRIX_ADAPTER, 1)
            }
            Invalidation::Supersession => {
                commit_supersession(&mut authority.runtime, params::MATRIX_ADAPTER, 1, 2)
            }
            Invalidation::Fence => Err("an adapter has no fence".to_string()),
        }
    }

    fn expected_dropped(&self, layers: usize, chunks: usize) -> usize {
        (layers - params::MATRIX_ADAPTER_FIRST_LAYER as usize) * chunks
    }

    fn superseded_generation(&self) -> u64 {
        2
    }

    fn procedure_mismatches(&self, observed: &Observed) -> Vec<String> {
        let mut mismatches = Vec::new();
        let mut check = |name: &str, ok: bool| {
            if !ok {
                mismatches.push(name.to_string());
            }
        };
        check(
            "test_model_layers",
            observed.layers as u64 == params::TEST_MODEL_LAYERS,
        );
        check(
            "hidden_size",
            observed.hidden_size as u64 == params::HIDDEN_SIZE,
        );
        check("num_heads", observed.num_heads as u64 == params::NUM_HEADS);
        check(
            "num_kv_heads",
            observed.num_kv_heads as u64 == params::NUM_KV_HEADS,
        );
        check(
            "vocab_size",
            observed.vocab_size as u64 == params::VOCAB_SIZE,
        );
        check("weight_seed", observed.weight_seed == params::WEIGHT_SEED);
        check(
            "page_tokens",
            observed.page_tokens as u64 == params::PAGE_TOKENS,
        );
        check(
            "intermediate_size",
            observed.intermediate_size == pins::INTERMEDIATE_SIZE,
        );
        check(
            "max_new_tokens",
            observed.max_new_tokens as u64 == params::MAX_NEW_TOKENS,
        );
        check("moe_topology", observed.moe.is_none());
        check(
            "adapter",
            observed
                .adapter
                .as_ref()
                .is_some_and(|(kind, first, rank, scale)| {
                    kind == "Qkv"
                        && *first as u64 == params::MATRIX_ADAPTER_FIRST_LAYER
                        && *rank as u64 == params::LORA_RANK
                        && scale.to_string() == "2"
                }),
        );
        mismatches
    }

    fn procedure_json(&self, observed: &Observed) -> Json {
        Json::obj([
            ("test_model_layers", Json::Int(observed.layers as i64)),
            ("hidden_size", Json::Int(observed.hidden_size as i64)),
            ("num_heads", Json::Int(observed.num_heads as i64)),
            ("num_kv_heads", Json::Int(observed.num_kv_heads as i64)),
            ("vocab_size", Json::Int(observed.vocab_size as i64)),
            ("weight_seed", Json::Int(observed.weight_seed as i64)),
            ("page_tokens", Json::Int(observed.page_tokens as i64)),
            (
                "moe_topology",
                Json::str(if observed.moe.is_none() {
                    "none"
                } else {
                    "moe"
                }),
            ),
            (
                "adapter",
                observed
                    .adapter
                    .as_ref()
                    .map_or(Json::Null, |(kind, first, rank, scale)| {
                        Json::obj([
                            ("kind", Json::str(kind.clone())),
                            ("first_layer", Json::Int(*first as i64)),
                            ("rank", Json::Int(*rank as i64)),
                            ("scale", Json::str(scale.to_string())),
                        ])
                    }),
            ),
        ])
    }

    fn note_cell(&self, tally: &mut Tally, cell: &CellRecord) {
        let Some(inflight) = cell.inflight else {
            return;
        };
        let ty = cell.ty.name();
        if cell.is_dependency {
            tally.add(&format!("inflight_adapter_revocation_race_cases.{ty}"), 1);
            tally.add(
                "inflight_adapter_requests_cancelled",
                i64::from(inflight.cancelled),
            );
            tally.add(
                "inflight_adapter_requests_restarted_from_fresh_prefill",
                i64::from(inflight.restarted_fresh == Some(true)),
            );
            tally.add(
                "adapter_restart_output_mismatches",
                i64::from(inflight.restarted_fresh == Some(false)),
            );
            tally.add("outputs_after_adapter_revocation", inflight.escaped);
        } else if cell.kind != "base_model" {
            if cell.is_semantic_input {
                tally.add(&format!("inflight_input_race_cases.{ty}"), 1);
            }
            tally.add("outputs_after_input_invalidation", inflight.escaped);
            tally.add(&format!("segment_inflight_cases.{}", cell.kind), 1);
        } else {
            tally.add("base_model_inflight_cases", 1);
            tally.add("base_model_inflight_escaped_outputs", inflight.escaped);
        }
    }
}

/// Run every cell of the matrix for one seed. Adapter requests are tracked in `adapter_ids`;
/// the cancelled requests of the other kinds go to a scratch set.
pub fn run_matrix(
    tally: &mut Tally,
    adapter_ids: &mut Ids,
    seed: u64,
    scale: Scale,
) -> Res<Vec<CellInfo>> {
    let rig = R009Rig;
    let mut scratch = Ids::default();
    let mut infos = Vec::new();
    let kinds: Vec<Kind> = std::iter::once(Kind::BaseModel)
        .chain(std::iter::once(Kind::Dependency))
        .chain(SEGMENT_KINDS.iter().map(Kind::Segment))
        .collect();
    for kind in kinds {
        if scale == Scale::Reduced && !matches!(kind, Kind::Dependency) {
            continue;
        }
        for path in Path::ALL {
            for ty in Invalidation::ALL {
                let ids = if matches!(kind, Kind::Dependency) {
                    &mut *adapter_ids
                } else {
                    &mut scratch
                };
                matrix::run_cell(tally, ids, &mut infos, &rig, seed, kind, ty, path)?;
            }
        }
    }
    Ok(infos)
}

/// The plain-restart arm, reported separately: an empty catalog after a restart serves nothing
/// and computes the right answer.
pub fn empty_restart(tally: &mut Tally) -> Res<()> {
    let config = &CONFIGS[0];
    let tokens = prompt_of(config);
    let first = lab::backend()?;
    let authority = Authority::new(&[params::ADAPTER_A])?;
    let binding = StackBinding::with_adapter(seeded(&first, config, params::ADAPTER_A, 1)?);
    let request = raw("restart-probe", &tokens);
    let warmed = lab::run_completed(&first, &request, &binding, &authority)?;
    drop(first);
    let restarted = lab::backend()?;
    let outcome = lab::run_completed(&restarted, &request, &binding, &authority)?;
    let cold = lab::cold(&restarted, &request, &binding)?;
    let equal = compare(tally, &outcome, &cold);
    let stats = restarted.catalog_stats().map_err(fail("catalog stats"))?;
    tally.set_json(
        "empty_restart_arm",
        Json::obj([
            ("role", Json::str(params::EMPTY_RESTART_ARM_ROLE)),
            ("entries_before_lookup", Json::Int(0)),
            (
                "pre_restart_inserted_chunks",
                Json::Int(warmed.inserted_chunks as i64),
            ),
            ("hit_tokens", Json::Int(outcome.hit_tokens as i64)),
            ("catalog_hits", Json::Int(stats.hits as i64)),
            ("equals_cold", Json::Bool(equal)),
            ("counts_toward_restore_coverage", Json::Bool(false)),
        ]),
    );
    Ok(())
}
