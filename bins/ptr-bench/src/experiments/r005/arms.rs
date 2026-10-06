//! The workloads and the isolated cases of R005 (everything but the invalidation matrix):
//! the hit-rate workload, the two over-capacity workloads with their pinned-fork eviction and
//! saturation phases, isolation, lineage, the revoked pinned page, the discarded overlay, the
//! effects of a cancelled run, the plain-restart arm, the RoPE-regime arms, forged and stale
//! witnesses, canonical positions, the producer environment and the current-layer K/V
//! transform change.
//!
//! Every function takes the run's tally and adds to it; a function that cannot run a case
//! returns the reason as an error, which stops the run as a harness failure (never as a
//! silently missing metric).

use super::lab::{
    self, commit_revocation, compare, fail, run_completed, runtime_with, scope, CatLab, Ran, Res,
    Scripted, CAT_CHAIN,
};
use super::matrix::{procedure, Ids};
use super::metrics::{Json, Tally};
use super::params::{self, Workload};
use super::pin;
use super::workload::{self, raw_request, request, Part};
use ptr_core::action_head::ActionIr;
use ptr_model_api::{ByteTokenizer, SegmentKind, Tokenizer};
use ptr_pods::{
    derive_kv_layer_chains, derive_kv_layer_chains_in_regime, KvBackendError, KvWitnessRejection,
    PagedKvTensorBackend, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling,
};
use ptr_runtime::{
    KvCapacityResource, KvCatalogError, KvChainId, KvInsert, KvLookup, KvRequestPositions, KvSalt,
    KvSession, KvSessionFinish, KvSessionPhase, KvSessionStep, ReferenceKvLimits,
    ReferenceKvReuseBackend,
};
use ptr_types::{CapabilityId, Effect, Generation, PrincipalId, ProjectId, TypeId};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// The catalog's occupancy, sampled after a step, against the workload's own peaks.
fn sample(tally: &mut Tally, backend: &ReferenceKvReuseBackend, workload: Workload) -> Res<()> {
    let stats = backend.catalog_stats().map_err(fail("catalog stats"))?;
    let name = workload.name();
    tally.peak(
        &format!("peak_trie_nodes_by_workload.{name}"),
        stats.nodes as i64,
    );
    tally.peak(
        &format!("peak_kv_page_bytes_by_workload.{name}"),
        stats.page_bytes as i64,
    );
    tally.peak(
        &format!("peak_metadata_bytes_by_workload.{name}"),
        stats.metadata_bytes as i64,
    );
    Ok(())
}

/// Drive `session` to its end, sampling the catalog after every prefill chunk (the only step
/// that inserts, and so the only step after which an occupancy peak can occur).
fn drive(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    workload: Workload,
    session: &mut KvSession<'_>,
    oracle: &dyn ptr_runtime::KvLivenessOracle,
) -> Res<Ran> {
    while session.phase() == KvSessionPhase::Prefilling {
        if let KvSessionStep::Cancelled(_) = session
            .prefill_chunk(oracle)
            .map_err(fail("a prefill chunk"))?
        {
            return Ok(Ran::Cancelled);
        }
        sample(tally, backend, workload)?;
    }
    match session
        .run_to_end(oracle)
        .map_err(fail("the rest of a session"))?
    {
        KvSessionFinish::Completed(outcome) => Ok(Ran::Completed(outcome)),
        KvSessionFinish::Cancelled(_) => Ok(Ran::Cancelled),
    }
}

/// Begin and drive one request, sampling.
fn run_sampled(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    workload: Workload,
    request: &ptr_model_api::SegmentedModelRequest,
    oracle: &dyn ptr_runtime::KvLivenessOracle,
) -> Res<ptr_runtime::ReferenceKvOutcome> {
    let mut session = backend
        .begin_session(&scope(), request, None, oracle)
        .map_err(fail("beginning a session"))?;
    match drive(tally, backend, workload, &mut session, oracle)? {
        Ran::Completed(outcome) => Ok(*outcome),
        Ran::Cancelled => Err("a session was cancelled where none should be".to_string()),
    }
}

fn procedure_check(
    tally: &mut Tally,
    rows: &mut Vec<(String, Json)>,
    name: &str,
    observed: u64,
    pinned: u64,
) {
    if observed != pinned {
        tally.add("procedure_mismatches", 1);
    }
    rows.push((
        name.to_string(),
        Json::obj([
            ("observed", Json::Int(observed as i64)),
            ("pinned", Json::Int(pinned as i64)),
        ]),
    ));
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

// ---- the main workload ------------------------------------------------------------------

/// 64 requests of 256 tokens sharing 128, one at a time: the hit rate, the bitwise parity of
/// every warm run with its cold twin, and the main workload's occupancy peaks.
pub fn main_workload(
    tally: &mut Tally,
    seed: u64,
    requests: usize,
    shape: workload::MainShape,
) -> Res<()> {
    let backend = lab::backend(Workload::Main, 8)?;
    let live = backend.own_liveness();
    let mut rows: Vec<(String, Json)> = Vec::new();
    let limits = procedure(tally, Workload::Main, backend.limits().catalog);
    let order = workload::order(seed, 1, requests);
    let prompts: Vec<Vec<u32>> = (0..requests)
        .map(|index| workload::main_prompt(index, shape))
        .collect();
    let shared = prompts
        .iter()
        .map(|prompt| {
            prompt
                .iter()
                .zip(&prompts[0])
                .take_while(|(a, b)| a == b)
                .count()
        })
        .min()
        .unwrap_or(0);
    procedure_check(
        tally,
        &mut rows,
        "n_requests",
        order.len() as u64,
        params::N_REQUESTS,
    );
    procedure_check(
        tally,
        &mut rows,
        "prompt_tokens",
        prompts.iter().map(Vec::len).max().unwrap_or(0) as u64,
        params::PROMPT_TOKENS,
    );
    procedure_check(
        tally,
        &mut rows,
        "prompt_tokens_min",
        prompts.iter().map(Vec::len).min().unwrap_or(0) as u64,
        params::PROMPT_TOKENS,
    );
    procedure_check(
        tally,
        &mut rows,
        "shared_prefix_tokens",
        shared as u64,
        params::SHARED_PREFIX_TOKENS,
    );
    procedure_check(tally, &mut rows, "concurrency", 1, params::CONCURRENCY);
    rows.push(("limits".to_string(), limits));

    let (mut reused, mut total, mut saved) = (0u64, 0u64, 0u64);
    let (mut warm, mut cold_time) = (Duration::ZERO, Duration::ZERO);
    for (position, &index) in order.iter().enumerate() {
        let request = raw_request(&format!("main-{index}"), &prompts[index]);
        let started = Instant::now();
        let outcome = run_sampled(tally, &backend, Workload::Main, &request, &live)?;
        warm += started.elapsed();
        let started = Instant::now();
        let cold = lab::cold(&backend, &scope(), &request)?;
        cold_time += started.elapsed();
        compare(tally, &outcome, &cold);
        reused += outcome.hit_tokens as u64;
        total += outcome.prompt_tokens as u64;
        saved += (outcome.prompt_tokens - outcome.prefilled_tokens) as u64;
        if position > 0 {
            tally.add("raw_prefix_reuse_cases", 1);
            if outcome.hit_tokens >= shape.shared {
                tally.add("raw_identical_prefix_hits", 1);
            }
        }
    }
    let stats = backend.catalog_stats().map_err(fail("catalog stats"))?;
    let session = backend.session_stats().map_err(fail("session stats"))?;
    tally.set("reused_prefix_tokens", reused as i64);
    tally.set("total_prompt_tokens", total as i64);
    tally.set("prefill_tokens_saved", saved as i64);
    tally.set_float("hit_rate", reused as f64 / total.max(1) as f64);
    tally.set_float("wall_time_ms", ms(warm));
    tally.set_json(
        "timing_ms",
        Json::obj([
            ("main_warm", Json::Float(ms(warm))),
            ("main_cold_twins", Json::Float(ms(cold_time))),
        ]),
    );
    tally.add("evictions", stats.evictions as i64);
    tally.add("evictions_main_workload", stats.evictions as i64);
    tally.add(
        "active_fork_page_evictions",
        session.pinned_entry_evictions as i64,
    );
    tally.add(
        "evictions_with_active_fork_pinned_main",
        session.evictions_with_active_fork_pinned as i64,
    );
    // One trie node holds one page of 16 KiB in the pinned shape.
    let per_node = stats.page_bytes / (stats.nodes.max(1) as u64);
    procedure_check(
        tally,
        &mut rows,
        "kv_page_bytes_per_node",
        per_node,
        params::KV_PAGE_BYTES_PER_NODE,
    );
    tally.set_json("procedure_main", Json::Obj(rows));
    Ok(())
}

// ---- the over-capacity workloads --------------------------------------------------------

/// Eight tokens after a 128-token prompt: the request is longer than the prompt (so the whole
/// prompt can hit) and crosses no chunk boundary (so it inserts nothing).
fn pinned_request(id: &str, prompt: &[u32], label: u64) -> ptr_model_api::SegmentedModelRequest {
    let mut tokens = prompt.to_vec();
    tokens.extend(workload::tail(label).into_iter().take(8));
    raw_request(id, &tokens)
}

fn capacity_resource(reason: &KvCatalogError) -> Option<&'static str> {
    match reason {
        KvCatalogError::CapacityExhausted(KvCapacityResource::TrieNodes) => Some("trie_nodes"),
        KvCatalogError::CapacityExhausted(KvCapacityResource::KvPageBytes) => Some("kv_page_bytes"),
        KvCatalogError::CapacityExhausted(KvCapacityResource::MetadataBytes) => {
            Some("metadata_bytes")
        }
        _ => None,
    }
}

/// The size of one over-capacity workload. `pinned` is the preregistered one; a reduced plan
/// exists only so a test can exercise the phases quickly, and its procedure shows as
/// mismatching the pinned one.
#[derive(Clone, Copy, Debug)]
pub struct CapacityPlan {
    pub limits: ptr_runtime::CatalogLimits,
    pub fill: usize,
    pub extra: usize,
    pub max_sessions: usize,
    /// Tokens of each unique prompt (a multiple of the chunk).
    pub prompt_tokens: usize,
    /// Fresh requests offered to the saturated catalog.
    pub excess_requests: usize,
}

impl CapacityPlan {
    pub fn pinned(kind: Workload) -> Self {
        Self {
            limits: lab::catalog_limits(kind),
            fill: params::OVER_CAPACITY_FILL_PROMPTS,
            extra: params::OVER_CAPACITY_EXTRA_PROMPTS,
            max_sessions: params::SATURATION_MAX_SESSIONS,
            prompt_tokens: params::OVER_CAPACITY_PROMPT_TOKENS,
            excess_requests: params::SATURATION_EXCESS_REQUESTS,
        }
    }

    /// Eight prompts of two chunks fill 16 nodes (or 16 pages): the same phases at a
    /// fraction of the cost.
    pub fn reduced(kind: Workload) -> Self {
        let prompt_tokens = 2 * params::CHUNK_TOKENS;
        let nodes = 8 * (prompt_tokens / params::CHUNK_TOKENS);
        let limits = match kind {
            Workload::BytesOnly => ptr_runtime::CatalogLimits {
                max_trie_nodes: 4096,
                max_kv_page_bytes: nodes as u64 * params::KV_PAGE_BYTES_PER_NODE,
                max_metadata_bytes: 16 * 1024 * 1024,
            },
            _ => ptr_runtime::CatalogLimits {
                max_trie_nodes: nodes,
                max_kv_page_bytes: 256 * 1024 * 1024,
                max_metadata_bytes: 64 * 1024 * 1024,
            },
        };
        Self {
            limits,
            fill: 8,
            extra: 2,
            max_sessions: 12,
            prompt_tokens,
            excess_requests: 3,
        }
    }
}

/// One over-capacity workload of unique prompts: fill the catalog exactly to its binding
/// limit, offer more (every one evicts), then pin every filled prompt with an open session so
/// that the demand of the forks exceeds the capacity, and offer fresh requests that must be
/// refused with `CapacityExhausted` and run as uncached full prefills.
pub fn over_capacity(
    tally: &mut Tally,
    ids: &mut Ids,
    seed: u64,
    kind: Workload,
    plan: CapacityPlan,
) -> Res<()> {
    let name = kind.name();
    let backend = ReferenceKvReuseBackend::new(
        ReferenceKvLimits {
            max_prompt_tokens: params::MAX_PROMPT_TOKENS,
            max_new_tokens: params::MAX_NEW_TOKENS,
            max_sessions: plan.max_sessions,
            catalog: plan.limits,
        },
        scope(),
    )
    .map_err(fail("the over-capacity backend"))?;
    let live = backend.own_liveness();
    let limits = procedure(tally, kind, backend.limits().catalog);
    let fill = plan.fill;
    let order = workload::order(seed, 40 + kind as u64, plan.fill + plan.extra);
    let pin_at = workload::scenario(seed, &format!("pin-at-{name}"))
        .range((fill / 8) as u64, (fill * 3 / 4) as u64) as usize;
    let prompt = |index: usize| workload::unique_prompt(kind, index, plan.prompt_tokens);
    let mut spot: Vec<(
        usize,
        ptr_model_api::SegmentedModelRequest,
        ptr_runtime::ReferenceKvOutcome,
    )> = Vec::new();

    // Phase A: fill. An open fork pins one entry from the middle of the fill to the end of
    // phase B.
    let mut pinned_fork: Option<(KvSession<'_>, ptr_model_api::SegmentedModelRequest)> = None;
    for (position, &index) in order.iter().enumerate().take(fill) {
        let request = raw_request(&format!("{name}-fill-{index}"), &prompt(index));
        let outcome = run_sampled(tally, &backend, kind, &request, &live)?;
        if position == 0 || position == fill - 1 {
            spot.push((index, request, outcome));
        }
        if position == pin_at {
            let fork_request = pinned_request(&format!("{name}-s0"), &prompt(index), 0);
            let session = backend
                .begin_session(&scope(), &fork_request, None, &live)
                .map_err(fail("the pinned fork"))?;
            if session.hit_tokens() != plan.prompt_tokens {
                return Err(format!("{name}: the pinned fork did not hit its prompt"));
            }
            pinned_fork = Some((session, fork_request));
        }
    }
    let at_fill = backend.catalog_stats().map_err(fail("catalog stats"))?;
    // Phase B: every extra prompt arrives at a full catalog.
    for &index in order.iter().skip(fill) {
        let request = raw_request(&format!("{name}-extra-{index}"), &prompt(index));
        let outcome = run_sampled(tally, &backend, kind, &request, &live)?;
        if index == *order.last().unwrap_or(&0) {
            spot.push((index, request, outcome));
        }
    }
    let after_extra = backend.catalog_stats().map_err(fail("catalog stats"))?;
    let evictions = after_extra.evictions as i64;
    tally.set(
        match kind {
            Workload::NodesOnly => "evictions_nodes_only_run",
            _ => "evictions_bytes_only_run",
        },
        evictions,
    );
    tally.set(&format!("evictions_unique_prompt_phase.{name}"), evictions);
    // The pinned fork finishes bitwise equal to its cold twin.
    if let Some((mut session, fork_request)) = pinned_fork.take() {
        let Ran::Completed(outcome) = drive(tally, &backend, kind, &mut session, &live)? else {
            return Err(format!("{name}: the pinned fork was cancelled"));
        };
        let cold = lab::cold(&backend, &scope(), &fork_request)?;
        compare(tally, &outcome, &cold);
        if outcome.hit_tokens != plan.prompt_tokens {
            tally.add("pinned_fork_lost_its_prefix", 1);
        }
    }
    for (_, request, outcome) in &spot {
        let cold = lab::cold(&backend, &scope(), request)?;
        compare(tally, outcome, &cold);
    }

    // Phase C: saturation. One open session per filled prompt; each pins that prompt's
    // deepest entry, so no node of the catalog is evictable.
    let mut sessions: Vec<(KvSession<'_>, ptr_model_api::SegmentedModelRequest)> = Vec::new();
    for &index in order.iter().rev() {
        if sessions.len() >= fill {
            break;
        }
        let request = pinned_request(&format!("{name}-pin-{index}"), &prompt(index), 1);
        let session = backend
            .begin_session(&scope(), &request, None, &live)
            .map_err(fail("a saturating fork"))?;
        if session.hit_tokens() == plan.prompt_tokens {
            sessions.push((session, request));
        }
    }
    let saturated = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.set(
        &format!("saturation_pinned_sessions.{name}"),
        sessions.len() as i64,
    );
    let needed_nodes = (plan.prompt_tokens / params::CHUNK_TOKENS) as u64;
    let demand_nodes = saturated.nodes as u64 + needed_nodes;
    let demand_bytes = saturated.page_bytes + needed_nodes * params::KV_PAGE_BYTES_PER_NODE;
    if demand_nodes > plan.limits.max_trie_nodes as u64
        || demand_bytes > plan.limits.max_kv_page_bytes
    {
        tally.add("saturation_demand_cases", 1);
    }
    let mut refused_here = 0i64;
    for index in 0..plan.excess_requests {
        let id = format!("{name}-excess-{index}");
        let request = raw_request(
            &id,
            &workload::excess_prompt(kind, index, plan.prompt_tokens),
        );
        let outcome = run_sampled(tally, &backend, kind, &request, &live)?;
        let cold = lab::cold(&backend, &scope(), &request)?;
        let equal = compare(tally, &outcome, &cold);
        let refused = outcome
            .insert_refusal_reasons
            .iter()
            .find_map(capacity_resource);
        if let Some(resource) = refused {
            refused_here += 1;
            tally.add("capacity_exhausted_refusals", 1);
            tally.add(
                &format!("capacity_refusals_by_resource.{name}.{resource}"),
                1,
            );
            tally.add(
                "capacity_refusal_cache_inserts",
                outcome.inserted_chunks as i64,
            );
            tally.add("capacity_fallback_output_mismatches", i64::from(!equal));
            ids.refused.insert(id.clone());
            let uncached_full_prefill =
                outcome.hit_tokens == 0 && outcome.prefilled_tokens == outcome.prompt_tokens;
            if uncached_full_prefill && equal {
                tally.add("capacity_refused_requests_completed", 1);
                ids.completed_after_refusal.insert(id);
            }
        }
    }
    tally.add(&format!("saturation_refused_requests.{name}"), refused_here);
    sample(tally, &backend, kind)?;
    // The pinned forks all finish bitwise equal to their cold twins and keep their prefix.
    for (position, (mut session, request)) in sessions.into_iter().enumerate() {
        let Ran::Completed(outcome) = drive(tally, &backend, kind, &mut session, &live)? else {
            return Err(format!("{name}: a pinned fork was cancelled"));
        };
        if outcome.hit_tokens != plan.prompt_tokens {
            tally.add("pinned_fork_lost_its_prefix", 1);
        }
        if position % 64 == 0 {
            let cold = lab::cold(&backend, &scope(), &request)?;
            compare(tally, &outcome, &cold);
        }
    }
    let end = backend.catalog_stats().map_err(fail("catalog stats"))?;
    let stats = backend.session_stats().map_err(fail("session stats"))?;
    tally.add(
        "active_fork_page_evictions",
        stats.pinned_entry_evictions as i64,
    );
    tally.add(
        "evictions_with_active_fork_pinned",
        stats.evictions_with_active_fork_pinned as i64,
    );
    tally.add("evictions", end.evictions as i64);
    tally.set(&format!("nodes_at_fill.{name}"), at_fill.nodes as i64);
    tally.set(
        &format!("nodes_at_saturation.{name}"),
        saturated.nodes as i64,
    );
    tally.set(
        &format!("page_bytes_at_saturation.{name}"),
        saturated.page_bytes as i64,
    );
    tally.set(
        &format!("capacity_refusals_counted.{name}"),
        end.capacity_refusals as i64,
    );
    tally.set_json(&format!("procedure_{name}"), limits);
    let per_node = saturated.page_bytes / (saturated.nodes.max(1) as u64);
    if per_node != params::KV_PAGE_BYTES_PER_NODE {
        tally.add("procedure_mismatches", 1);
    }
    Ok(())
}

// ---- isolation and lineage ----------------------------------------------------------------

/// The same prefix under another principal, and under another project of the same principal.
pub fn isolation(tally: &mut Tally) -> Res<()> {
    let backend = lab::main_backend()?;
    let live = backend.own_liveness();
    let mut tokens = workload::blocks(400, 3);
    tokens.extend(workload::sequence(410, 4));
    let warm = raw_request("iso-warm", &tokens);
    let warmed = run_completed(&backend, &scope(), &warm, &live)?;
    if warmed.inserted_chunks != 3 {
        return Err("isolation: the warm run did not cache three chunks".to_string());
    }
    // Control: the same scope hits, so the cross-scope misses below are not vacuous.
    let control = run_completed(
        &backend,
        &scope(),
        &raw_request("iso-control", &tokens),
        &live,
    )?;
    tally.add("isolation_control_hit_tokens", control.hit_tokens as i64);
    for (label, other, attempts, hits) in [
        (
            "principal",
            lab::other_principal_scope(),
            "cross_principal_attempts",
            "cross_principal_hits",
        ),
        (
            "project",
            lab::other_project_scope(),
            "cross_project_attempts",
            "cross_project_hits",
        ),
    ] {
        let request = raw_request(&format!("iso-{label}"), &tokens);
        let outcome = run_completed(&backend, &other, &request, &live)?;
        tally.add(attempts, 1);
        tally.add(hits, i64::from(outcome.hit_tokens > 0));
        let cold = lab::cold(&backend, &other, &request)?;
        compare(tally, &outcome, &cold);
    }
    Ok(())
}

/// Identical tokens under distinct lineages hold distinct entries; only an exact lineage
/// match is admitted.
pub fn lineage(tally: &mut Tally) -> Res<()> {
    let backend = lab::main_backend()?;
    let live = backend.own_liveness();
    let bound = workload::blocks(420, 2);
    let tail = workload::sequence(430, 4);
    let with = |id: &str, generation: u64, tokens: Vec<u32>| {
        request(
            id,
            &[
                Part::bound(SegmentKind::Semantic, "doc:lineage", generation, tokens),
                Part::raw(tail.clone()),
            ],
        )
    };
    let first = with("lineage-gen1", 1, bound.clone());
    let warmed = run_completed(&backend, &scope(), &first, &live)?;
    if warmed.inserted_chunks != 2 {
        return Err("lineage: the first lineage did not cache two chunks".to_string());
    }
    // Same tokens, generation 2.
    let second = with("lineage-gen2", 2, bound.clone());
    let second_out = run_completed(&backend, &scope(), &second, &live)?;
    tally.add("same_tokens_different_generation_cases", 1);
    tally.add("cross_lineage_hits", i64::from(second_out.hit_tokens > 0));
    compare(tally, &second_out, &lab::cold(&backend, &scope(), &second)?);
    // Same tokens as one raw segment: a third lineage.
    let as_raw = request(
        "lineage-raw",
        &[Part::raw(bound.iter().chain(&tail).copied().collect())],
    );
    let raw_out = run_completed(&backend, &scope(), &as_raw, &live)?;
    tally.add("cross_lineage_hits", i64::from(raw_out.hit_tokens > 0));
    compare(tally, &raw_out, &lab::cold(&backend, &scope(), &as_raw)?);
    // One node, three entries: identical tokens, three lineages (measured before the
    // other-content request adds nodes of its own).
    let stats = backend.catalog_stats().map_err(fail("catalog stats"))?;
    if stats.nodes != 2 {
        return Err(format!(
            "lineage: {} nodes for two chunks of identical tokens",
            stats.nodes
        ));
    }
    tally.set(
        "same_tokens_distinct_lineage_entries",
        (stats.entries / stats.nodes) as i64,
    );
    tally.set("lineage_entries", stats.entries as i64);
    tally.set("lineage_nodes", stats.nodes as i64);
    // Different content under the same identity and generation.
    let other_content = with("lineage-content", 1, workload::blocks(440, 2));
    let content_out = run_completed(&backend, &scope(), &other_content, &live)?;
    tally.add("cross_lineage_hits", i64::from(content_out.hit_tokens > 0));
    // Control: an exact lineage match is admitted.
    let again = run_completed(&backend, &scope(), &with("lineage-again", 1, bound), &live)?;
    tally.set("lineage_exact_match_hit_tokens", again.hit_tokens as i64);
    Ok(())
}

/// Two requests sharing an identical raw prefix reuse it (content-bound admission).
pub fn raw_reuse(tally: &mut Tally) -> Res<()> {
    let backend = lab::main_backend()?;
    let live = backend.own_liveness();
    let shared = workload::blocks(450, 2);
    let make = |id: &str, label: u64| {
        let mut tokens = shared.clone();
        tokens.extend(workload::sequence(label, 5));
        raw_request(id, &tokens)
    };
    run_completed(&backend, &scope(), &make("raw-a", 460), &live)?;
    let second = make("raw-b", 461);
    let outcome = run_completed(&backend, &scope(), &second, &live)?;
    tally.add("raw_prefix_reuse_cases", 1);
    tally.add(
        "raw_identical_prefix_hits",
        i64::from(outcome.hit_tokens == shared.len()),
    );
    compare(tally, &outcome, &lab::cold(&backend, &scope(), &second)?);
    Ok(())
}

// ---- the revoked pinned page, the overlay, the effects --------------------------------------

/// An active fork pins a page; the page's input is revoked; nothing is served from it, it is
/// tombstoned at once and every one of its bytes is reclaimed when the fork releases it.
pub fn revoked_pinned_page(tally: &mut Tally, ids: &mut Ids) -> Res<()> {
    let identity = "capsule:pin";
    let mut runtime = runtime_with(&[identity])?;
    let backend = lab::main_backend()?;
    let bound = workload::blocks(470, 2);
    let make = |id: &str, tail: u64| {
        request(
            id,
            &[
                Part::bound(SegmentKind::Memory, identity, 1, bound.clone()),
                Part::raw(workload::sequence(tail, 8)),
            ],
        )
    };
    let warm = make("pinned-warm", 480);
    let warmed = run_completed(&backend, &scope(), &warm, &runtime)?;
    if warmed.inserted_chunks != 2 {
        return Err("revoked pinned page: the warm run did not cache two chunks".to_string());
    }
    let fork_request = make("pinned-fork", 481);
    let mut session = backend
        .begin_session(&scope(), &fork_request, None, &runtime)
        .map_err(fail("the pinned fork"))?;
    if session.hit_tokens() != 32 {
        return Err("revoked pinned page: the fork did not pin the cached prefix".to_string());
    }
    for _ in 0..2 {
        session.prefill_chunk(&runtime).map_err(fail("a chunk"))?;
        if session.phase() == KvSessionPhase::Decoding {
            break;
        }
    }
    session.decode_step(&runtime).map_err(fail("a token"))?;
    let before = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add("input_revocations_performed", 1);
    commit_revocation(&mut runtime, identity, 1)?;
    backend
        .invalidate_inputs(&BTreeSet::from([(identity.to_string(), Generation(1))]))
        .map_err(fail("invalidating"))?;
    let pinned = backend.catalog_stats().map_err(fail("catalog stats"))?;
    // Tombstoned immediately; its bytes stay charged while the fork holds the page.
    tally.add(
        "revoked_pinned_page_tombstoned",
        pinned.tombstoned_pinned_entries as i64,
    );
    tally.add("revoked_pinned_page_bytes_total", pinned.page_bytes as i64);
    if pinned.tombstoned_pinned_entries < 1 || pinned.page_bytes == 0 || before.entries < 2 {
        return Err(
            "revoked pinned page: the pinned entry was not tombstoned in place".to_string(),
        );
    }
    // Attempt 1: a new lookup of the revoked page's lineage.
    let late = make("pinned-late", 482);
    let late_session = backend
        .begin_session(&scope(), &late, None, &runtime)
        .map_err(fail("the late session"))?;
    tally.add("revoked_pinned_page_attempts", 1);
    tally.add(
        "revoked_pinned_page_served_after_revocation",
        i64::from(late_session.hit_tokens() > 0),
    );
    tally.add("revoked_fork_attempts", 1);
    tally.add(
        "revoked_fork_count",
        i64::from(late_session.hit_tokens() > 0),
    );
    drop(late_session);
    // Attempt 2: the fork that holds the page asks for its next token.
    let step = session
        .decode_step(&runtime)
        .map_err(fail("the next token"))?;
    tally.add("revoked_pinned_page_attempts", 1);
    tally.add(
        "revoked_pinned_page_served_after_revocation",
        i64::from(matches!(step, KvSessionStep::Token(_))),
    );
    // The cancelled fork released its pin: every byte of the page is reclaimed.
    let after = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add(
        "revoked_pinned_page_bytes_reclaimed_after_release",
        pinned.page_bytes as i64 - after.page_bytes as i64,
    );
    tally.add(
        "unreachable_revoked_page_bytes_unreclaimed",
        after.page_bytes as i64,
    );
    let restarted = session.restart().map_err(fail("the restart"))?;
    let equal = compare(
        tally,
        &restarted,
        &lab::cold(&backend, &scope(), &fork_request)?,
    );
    tally.add("cancelled_forks", 1);
    tally.add(
        "cancelled_forks_restarted_from_fresh_prefill",
        i64::from(equal && restarted.hit_tokens == 0 && restarted.inserted_chunks == 0),
    );
    ids.absorb_sessions(&backend)?;
    Ok(())
}

/// An entry that exists only because a prepared overlay was visible to its producer, and an
/// overlay that never commits: the committed ledger never lets it be read or inserted.
pub fn discarded_overlay(tally: &mut Tally) -> Res<()> {
    let identity = "capsule:overlay";
    // The ledger never commits the overlay's capsule.
    let committed = runtime_with(&[])?;
    let backend = lab::main_backend()?;
    let own = backend.own_liveness();
    let make = |id: &str, tail: u64| {
        request(
            id,
            &[
                Part::bound(SegmentKind::Memory, identity, 1, workload::blocks(490, 2)),
                Part::raw(workload::sequence(tail, 4)),
            ],
        )
    };
    // The producer's own view (nothing revoked) admits the entries.
    let produced = run_completed(&backend, &scope(), &make("overlay-produce", 500), &own)?;
    if produced.inserted_chunks != 2 {
        return Err("discarded overlay: the producer did not cache its entries".to_string());
    }
    // Attempt 1: read it under the committed ledger.
    let read = make("overlay-read", 501);
    let session = backend
        .begin_session(&scope(), &read, None, &committed)
        .map_err(fail("the overlay read"))?;
    tally.add("discarded_overlay_entry_attempts", 1);
    tally.add(
        "discarded_overlay_entry_hits",
        i64::from(session.hit_tokens() > 0),
    );
    drop(session);
    compare(
        tally,
        &run_completed(&backend, &scope(), &read, &committed)?,
        &lab::cold(&backend, &scope(), &read)?,
    );
    // Attempt 2: a fresh catalog under the committed ledger refuses to take the entry.
    let fresh = lab::main_backend()?;
    let insert = run_completed(&fresh, &scope(), &make("overlay-insert", 502), &committed)?;
    tally.add("discarded_overlay_entry_attempts", 1);
    tally.add(
        "discarded_overlay_entry_hits",
        insert.inserted_chunks as i64,
    );
    let refused = insert
        .insert_refusal_reasons
        .iter()
        .all(|reason| *reason == KvCatalogError::DependencyNotLive);
    tally.add(
        "overlay_inserts_refused_as_not_live",
        i64::from(refused && insert.insert_refusals > 0),
    );
    Ok(())
}

/// A cancelled run yields no outcome and its effect is not admitted: the session refuses to
/// finish, and the hard boundary refuses the action on the revoked generation.
pub fn cancelled_effects(tally: &mut Tally, ids: &mut Ids) -> Res<()> {
    let identity = "capsule:eff";
    let mut runtime = runtime_with(&[identity])?;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("memory.write"));
    runtime.permissions_mut().allow_mutation = true;
    let action = |runtime: &ptr_runtime::PtrRuntime| ActionIr {
        operation: "update".into(),
        target: identity.to_owned(),
        capability: CapabilityId::from("memory.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("SemanticCapsule"),
        generation: Generation(1),
        revision: runtime.revision(),
        payload: vec![],
    };
    // Control: before the revocation the same action is admitted, so the refusals below
    // are the revocation's and not a missing permission.
    tally.add(
        "effect_control_admitted",
        i64::from(runtime.authorize_action(&action(&runtime)).is_ok()),
    );

    let backend = lab::main_backend()?;
    let make = |id: &str, tail: u64| {
        request(
            id,
            &[
                Part::bound(SegmentKind::Memory, identity, 1, workload::blocks(510, 2)),
                Part::raw(workload::sequence(tail, 8)),
            ],
        )
    };
    run_completed(&backend, &scope(), &make("effect-warm", 520), &runtime)?;
    let request = make("effect-run", 521);
    let mut session = backend
        .begin_session(&scope(), &request, None, &runtime)
        .map_err(fail("the effect run"))?;
    session.prefill_chunk(&runtime).map_err(fail("a chunk"))?;
    tally.add("input_revocations_performed", 1);
    commit_revocation(&mut runtime, identity, 1)?;
    backend
        .invalidate_inputs(&BTreeSet::from([(identity.to_string(), Generation(1))]))
        .map_err(fail("invalidating"))?;
    let step = match session.phase() {
        KvSessionPhase::Prefilling => session.prefill_chunk(&runtime),
        _ => session.decode_step(&runtime),
    }
    .map_err(fail("the first step after the event"))?;
    if !matches!(step, KvSessionStep::Cancelled(_)) {
        return Err("effects: the run was not cancelled".to_string());
    }
    // Attempt: the run is asked to finish and hand out its outcome.
    tally.add("cancelled_run_effect_attempts", 1);
    tally.add(
        "cancelled_run_effects_admitted",
        i64::from(session.finish(&runtime).is_ok()),
    );
    // Attempt: the run is asked for one more token.
    tally.add("cancelled_run_effect_attempts", 1);
    tally.add(
        "cancelled_run_effects_admitted",
        i64::from(session.decode_step(&runtime).is_ok()),
    );
    // Attempt: the action the run was working toward, at the current revision, on the
    // revoked generation.
    tally.add("cancelled_run_effect_attempts", 1);
    tally.add(
        "cancelled_run_effects_admitted",
        i64::from(runtime.authorize_action(&action(&runtime)).is_ok()),
    );
    let restarted = session.restart().map_err(fail("the restart"))?;
    let equal = compare(tally, &restarted, &lab::cold(&backend, &scope(), &request)?);
    tally.add("cancelled_forks", 1);
    tally.add(
        "cancelled_forks_restarted_from_fresh_prefill",
        i64::from(equal && restarted.hit_tokens == 0 && restarted.inserted_chunks == 0),
    );
    ids.absorb_sessions(&backend)?;
    Ok(())
}

/// The plain-restart arm, reported separately: an empty catalog after a restart serves
/// nothing and computes the right answer.
pub fn empty_restart(tally: &mut Tally) -> Res<()> {
    let first = lab::main_backend()?;
    let mut tokens = workload::blocks(530, 2);
    tokens.extend(workload::sequence(540, 4));
    let request = raw_request("restart-probe", &tokens);
    let warmed = run_completed(&first, &scope(), &request, &first.own_liveness())?;
    drop(first);
    let restarted = lab::main_backend()?;
    let outcome = run_completed(&restarted, &scope(), &request, &restarted.own_liveness())?;
    let cold = lab::cold(&restarted, &scope(), &request)?;
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

// ---- the RoPE regime arms -----------------------------------------------------------------

fn ntk_backend() -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::with_rope_scaling(
        ReferenceKvLimits {
            max_prompt_tokens: params::MAX_PROMPT_TOKENS,
            max_new_tokens: params::MAX_NEW_TOKENS,
            max_sessions: 8,
            catalog: lab::catalog_limits(Workload::Main),
        },
        scope(),
        params::WEIGHT_SEED,
        Qwen2RopeScaling::DynamicNtk {
            factor: params::NTK_FACTOR,
            original_max_position_embeddings: params::NTK_ORIGINAL_MAX_POSITIONS,
        },
    )
    .map_err(fail("the dynamic-NTK backend"))
}

/// The positive control (a decode that crosses the NTK boundary under the regime fixed at
/// admission stays Exact and bit-identical) and the forced transition (a request admitted
/// into another regime is recomputed, never served from the other regime's entry).
pub fn regimes(tally: &mut Tally) -> Res<()> {
    let omp = params::NTK_ORIGINAL_MAX_POSITIONS;
    let new = params::MAX_NEW_TOKENS;
    // Positive control.
    let backend = ntk_backend()?;
    let live = backend.own_liveness();
    let prefix = workload::block(550);
    // 24 tokens: below the boundary, and 32 with the new tokens, above it.
    let with_tail = |label: u64| {
        let mut tokens = prefix.clone();
        tokens.extend(workload::sequence(label, 8));
        tokens
    };
    let (first, second) = (with_tail(551), with_tail(552));
    let crosses = first.len() <= omp && first.len() + new > omp;
    let a = raw_request("ntk-a", &first);
    let b = raw_request("ntk-b", &second);
    let a_out = run_completed(&backend, &scope(), &a, &live)?;
    let b_out = run_completed(&backend, &scope(), &b, &live)?;
    let theta = f64::from(params::ROPE_THETA);
    let scaled = f64::from(a_out.rope_regime.effective_base()) > theta;
    let one_regime = a_out.rope_regime == b_out.rope_regime
        && a_out.rope_regime.admitted_sequence_length() == first.len() + new;
    let mut failures = 0i64;
    for (outcome, request) in [(&a_out, &a), (&b_out, &b)] {
        let cold = lab::cold(&backend, &scope(), request)?;
        if !compare(tally, outcome, &cold) {
            failures += 1;
        }
    }
    if b_out.hit_tokens != prefix.len() {
        failures += 1;
    }
    if !(crosses && scaled && one_regime) {
        return Err(
            "regimes: the positive control does not cross the boundary under one regime"
                .to_string(),
        );
    }
    tally.add("fixed_regime_decode_crossing_cases", 1);
    tally.add("fixed_regime_crossing_parity_failures", failures);

    // Forced transition.
    let backend = ntk_backend()?;
    let live = backend.own_liveness();
    let short = raw_request("ntk-short", &prefix);
    let short_out = run_completed(&backend, &scope(), &short, &live)?;
    let mut longer_tokens = prefix.clone();
    longer_tokens.extend(workload::sequence(553, 20));
    let longer = raw_request("ntk-longer", &longer_tokens);
    let longer_out = run_completed(&backend, &scope(), &longer, &live)?;
    let transition = f64::from(short_out.rope_regime.effective_base()) == theta
        && f64::from(longer_out.rope_regime.effective_base()) > theta
        && short_out.rope_regime != longer_out.rope_regime;
    if !transition {
        return Err(
            "regimes: the forced transition does not change the effective regime".to_string(),
        );
    }
    let cold = lab::cold(&backend, &scope(), &longer)?;
    let equal = compare(tally, &longer_out, &cold);
    tally.add("effective_regime_transition_cases", 1);
    // An output served from the other regime's entry would show as reused tokens.
    tally.add(
        "exact_outputs_across_effective_regime_transition_without_recompute",
        i64::from(longer_out.hit_tokens > 0 || !equal),
    );
    // Control: the first regime's entry is still served inside its own regime.
    let mut again = prefix.clone();
    again.push(7);
    let again_out = run_completed(&backend, &scope(), &raw_request("ntk-again", &again), &live)?;
    tally.set("regime_control_hit_tokens", again_out.hit_tokens as i64);
    Ok(())
}

// ---- witnesses ------------------------------------------------------------------------------

fn semantic_request(label: u64) -> ptr_model_api::SegmentedModelRequest {
    request(
        "witness",
        &[
            Part::raw(workload::block(label)),
            Part::bound(
                SegmentKind::Semantic,
                "price:sku",
                1,
                workload::block(label + 1),
            ),
        ],
    )
}

/// Forged witnesses and a genuine one held across an invalidation.
pub fn witnesses(tally: &mut Tally) -> Res<()> {
    let limits = lab::catalog_limits(Workload::Main);
    let live = Scripted::default();
    let req = semantic_request(560);
    let other = semantic_request(570);
    // Forged: a witness minted by another backend's authority.
    let mut a = CatLab::new(limits)?;
    let mut b = CatLab::new(limits)?;
    let prefix_a = a
        .insert(&req, Vec::new(), &live)?
        .map_err(|error| format!("witnesses: {error}"))?;
    b.insert(&req, Vec::new(), &live)?
        .map_err(|error| format!("witnesses: {error}"))?;
    let foreign = b.lookup(&req, &live).ok_or("witnesses: no foreign hit")?;
    let (foreign_witness, _, foreign_pin) = foreign.into_parts();
    let mut target = a.fresh_cache()?;
    let forged = a
        .backend
        .fork_prefix_admitted(&mut target, &prefix_a, foreign_witness);
    tally.add("forged_witness_fork_attempts", 1);
    tally.add("forged_witness_forks_accepted", i64::from(forged.is_ok()));
    tally.add(
        "forged_witness_rejected_as_foreign",
        i64::from(matches!(
            forged,
            Err(KvBackendError::ForkWitnessRejected(
                KvWitnessRejection::ForeignBackend
            ))
        )),
    );
    b.catalog.release(&foreign_pin);
    // Forged: a genuine witness for one prefix presented for another.
    let prefix_other = a
        .insert(&other, Vec::new(), &live)?
        .map_err(|error| format!("witnesses: {error}"))?;
    let genuine = a.lookup(&req, &live).ok_or("witnesses: no hit")?;
    let (witness, _, pin) = genuine.into_parts();
    let mut target = a.fresh_cache()?;
    let swapped = a
        .backend
        .fork_prefix_admitted(&mut target, &prefix_other, witness);
    tally.add("forged_witness_fork_attempts", 1);
    tally.add("forged_witness_forks_accepted", i64::from(swapped.is_ok()));
    tally.add(
        "forged_witness_rejected_as_mismatch",
        i64::from(matches!(
            swapped,
            Err(KvBackendError::ForkWitnessRejected(
                KvWitnessRejection::PrefixMismatch
            ))
        )),
    );
    a.catalog.release(&pin);
    // Stale: an authentic witness held across a dependency invalidation.
    let held = a.lookup(&req, &live).ok_or("witnesses: no hit to hold")?;
    let (witness, prefix, pin) = held.into_parts();
    a.catalog
        .invalidate_inputs(&BTreeSet::from([("price:sku".to_string(), Generation(1))]));
    let mut target = a.fresh_cache()?;
    let replayed = a
        .backend
        .fork_prefix_admitted(&mut target, &prefix, witness);
    tally.add("stale_witness_replay_cases", 1);
    tally.add(
        "stale_or_reused_witness_forks_admitted",
        i64::from(replayed.is_ok()),
    );
    tally.add(
        "stale_witness_rejected_as_revoked",
        i64::from(matches!(
            replayed,
            Err(KvBackendError::ForkWitnessRejected(
                KvWitnessRejection::Revoked
            ))
        )),
    );
    tally.add("revoked_fork_count", i64::from(replayed.is_ok()));
    tally.add("revoked_fork_attempts", 1);
    a.catalog.release(&pin);

    // The reuse of a consumed witness is a compile-time error, evidenced statically. The
    // harness cannot run a doctest or a unit test; it only reports that both items exist.
    const PAGED_KV: &str = include_str!("../../../../../crates/ptr-pods/src/paged_kv.rs");
    let doctest = PAGED_KV
        .find("pub struct KvForkWitness")
        .and_then(|at| {
            PAGED_KV[..at]
                .rfind("```compile_fail")
                .map(|doc| at - doc < 1_200)
        })
        .unwrap_or(false);
    let unit_test = PAGED_KV.contains("fn second_consume_of_the_same_witness_state_is_refused");
    tally.set_json(
        "witness_reuse_static_evidence_present",
        Json::obj([
            ("compile_fail_doctest_on_KvForkWitness", Json::Bool(doctest)),
            (
                "unit_test_second_consume_of_the_same_witness_state_is_refused",
                Json::Bool(unit_test),
            ),
        ]),
    );
    tally.unsupported(
        "witness_reuse_static_evidence_passing",
        "passing needs `cargo test -p ptr-pods --doc` (compile_fail doctest on KvForkWitness) and `cargo test -p ptr-pods witness_tests`; the harness only reports that both items are in the source",
    );
    Ok(())
}

// ---- positions, environment, transform change ---------------------------------------------------

/// Explicit or non-canonical positions bypass the Exact lookup, and a prefix published at a
/// non-zero offset is never inserted.
pub fn positions(tally: &mut Tally) -> Res<()> {
    let mut lab = CatLab::new(lab::catalog_limits(Workload::Main))?;
    let live = Scripted::default();
    let req = lab::two_blocks(580, 581);
    let prefix = lab
        .insert(&req, Vec::new(), &live)?
        .map_err(|error| format!("positions: {error}"))?;
    let control = lab
        .lookup(&req, &live)
        .ok_or("positions: the canonical lookup did not hit")?;
    lab.catalog.release(control.pin());
    for positions in [
        KvRequestPositions::Offset(params::CHUNK_TOKENS),
        KvRequestPositions::Explicit((0..32).collect()),
    ] {
        let hit = lab.lookup_with(&CAT_CHAIN, &positions, &req, &live);
        tally.add("mismatched_position_cases", 1);
        tally.add(
            "exact_hits_with_mismatched_positions",
            i64::from(hit.is_some()),
        );
        if let Some(hit) = hit {
            lab.catalog.release(hit.pin());
        }
    }
    let mut shifted = prefix;
    shifted.position_offset = params::CHUNK_TOKENS;
    let salt = KvSalt::derive(
        &PrincipalId::from("positions"),
        &ProjectId::from("positions"),
        &ptr_types::ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    );
    let inserted = lab.catalog.insert(
        KvInsert {
            salt: &salt,
            chain: &CAT_CHAIN,
            request: &req,
            prefix: shifted,
            page_bytes: 1,
            extra_dependencies: Vec::new(),
        },
        &live,
    );
    tally.add("mismatched_position_cases", 1);
    tally.add(
        "exact_hits_with_mismatched_positions",
        i64::from(inserted.is_ok()),
    );
    tally.add(
        "shifted_insert_refused_as_noncanonical",
        i64::from(inserted == Err(KvCatalogError::NonCanonicalPosition)),
    );
    Ok(())
}

/// A page or snapshot produced under another build identity is a miss.
pub fn environment_change(tally: &mut Tally, build: &str) -> Res<()> {
    let live = Scripted::default();
    let config = pin::stack_config(Qwen2QkNorm::Disabled, params::RMS_NORM_EPS);
    let stack = Qwen2PagedStack::reference(config, params::TINY_MODEL.weight_seed)
        .map_err(fail("the stack"))?;
    let regime = stack
        .config()
        .layer
        .rope_regime(1)
        .map_err(fail("the regime"))?;
    let other_build = pin::changed_build(build);
    let chain = |build: &str| -> Res<KvChainId> {
        Ok(KvChainId(
            derive_kv_layer_chains_in_regime(&stack, &pin::chain_identity(build), &regime)
                .map_err(fail("a chain"))?
                .final_id(),
        ))
    };
    let (here, there) = (chain(build)?, chain(&other_build)?);
    if here == there {
        return Err("environment: a changed build field did not change the chain".to_string());
    }
    let limits = lab::catalog_limits(Workload::Main);
    let mut lab = CatLab::new(limits)?;
    let req = lab::two_blocks(590, 591);
    lab.insert_under(&here, &req, Vec::new(), &live)?
        .map_err(|error| format!("environment: {error}"))?;
    // Control: the producer's own environment hits.
    let control = lab
        .lookup_with(&here, &KvRequestPositions::Canonical, &req, &live)
        .ok_or("environment: the producer's environment did not hit")?;
    lab.catalog.release(control.pin());
    // Case 1: a hit attempted from the other environment.
    let hit = lab.lookup_with(&there, &KvRequestPositions::Canonical, &req, &live);
    tally.add("cross_environment_restore_cases", 1);
    tally.add("exact_hits_across_environment", i64::from(hit.is_some()));
    // Case 2: a snapshot restored in the other environment.
    let snapshot = lab
        .catalog
        .export_entries()
        .map_err(fail("exporting the catalog"))?;
    let mut restored = CatLab::new(limits)?;
    restored
        .catalog
        .restore_entries(&snapshot)
        .map_err(fail("restoring the catalog"))?;
    let mut tokens = req.tokens();
    tokens.truncate(tokens.len() / params::CHUNK_TOKENS * params::CHUNK_TOKENS);
    restored.publish(&tokens)?;
    let hit = restored.lookup_with(&there, &KvRequestPositions::Canonical, &req, &live);
    tally.add("cross_environment_restore_cases", 1);
    tally.add("exact_hits_across_environment", i64::from(hit.is_some()));
    let control = restored
        .lookup_with(&here, &KvRequestPositions::Canonical, &req, &live)
        .ok_or("environment: the restored entry did not hit in its own environment")?;
    restored.catalog.release(control.pin());
    // Case 3: the catalog salt, which also binds the build identity.
    let salt = KvSalt::derive(
        &PrincipalId::from(params::PRINCIPAL),
        &ProjectId::from(params::PROJECT),
        &ptr_types::ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        &other_build,
        "paged-kv-v1",
    );
    let hit = lab.catalog.lookup(
        KvLookup {
            salt: &salt,
            chain: &here,
            request: &req,
            positions: &KvRequestPositions::Canonical,
        },
        &live,
    );
    tally.add("cross_environment_restore_cases", 1);
    tally.add("exact_hits_across_environment", i64::from(hit.is_some()));
    Ok(())
}

/// A key-norm weight (or eps) changed at layer `l` with the layers below unchanged: entries
/// of the layers below stay valid, the entry of `l` and above are not served.
pub fn transform_change(tally: &mut Tally) -> Res<()> {
    let live = Scripted::default();
    let config = pin::stack_config(Qwen2QkNorm::Enabled, params::RMS_NORM_EPS);
    let base = Qwen2PagedStack::reference(config.clone(), params::TINY_MODEL.weight_seed)
        .map_err(fail("the QK-norm stack"))?;
    let identity = pin::chain_identity("r005-transform-arm");
    let before = derive_kv_layer_chains(&base, &identity).map_err(fail("the chains"))?;
    let limits = lab::catalog_limits(Workload::Main);
    let layers = params::TINY_MODEL.layers as usize;
    let mut variants: Vec<(String, usize, Qwen2PagedStack)> = Vec::new();
    for layer in 0..layers {
        let mut weights = base.weights().clone();
        let norm = weights.layers[layer]
            .key_norm
            .as_mut()
            .ok_or("the QK-norm stack has no key norm")?;
        norm[0] += 0.125;
        variants.push((
            format!("key_norm_weight@{layer}"),
            layer,
            Qwen2PagedStack::new(config.clone(), weights).map_err(fail("a changed stack"))?,
        ));
    }
    let mut eps_config = config.clone();
    eps_config.layer.rms_norm_eps = params::RMS_NORM_EPS * 2.0;
    variants.push((
        "qk_norm_eps".to_string(),
        0,
        Qwen2PagedStack::reference(eps_config, params::TINY_MODEL.weight_seed)
            .map_err(fail("the eps stack"))?,
    ));
    for (label, changed_at, changed) in variants {
        let after = derive_kv_layer_chains(&changed, &identity).map_err(fail("the chains"))?;
        for layer in 0..layers {
            let (a, b) = (before.layer(layer), after.layer(layer));
            let must_differ = layer >= changed_at;
            if (a != b) != must_differ {
                return Err(format!(
                    "{label}: chain({layer}) {} where {}",
                    if a != b { "changed" } else { "did not change" },
                    if must_differ {
                        "it must"
                    } else {
                        "it must not"
                    }
                ));
            }
        }
        // One entry per layer chain on the same prefix.
        let mut lab = CatLab::new(limits)?;
        let req = lab::two_blocks(600 + changed_at as u64, 610);
        for layer in 0..layers {
            let id = KvChainId(before.layer(layer).ok_or("a chain")?);
            lab.insert_under(&id, &req, Vec::new(), &live)?
                .map_err(|error| format!("{label}: {error}"))?;
        }
        let mut stale = 0i64;
        let mut valid_reuse = 0i64;
        for layer in 0..layers {
            let id = KvChainId(after.layer(layer).ok_or("a chain")?);
            if let Some(hit) = lab.lookup_with(&id, &KvRequestPositions::Canonical, &req, &live) {
                lab.catalog.release(hit.pin());
                if layer >= changed_at {
                    stale += 1;
                } else {
                    valid_reuse += 1;
                }
            }
        }
        tally.add("current_layer_kv_transform_change_cases", 1);
        tally.add("stale_exact_hits", stale);
        tally.add("transform_change_unchanged_layers_reused", valid_reuse);
        tally.add("transform_change_unchanged_layers", changed_at as i64);
    }
    Ok(())
}

/// Everything the arms add that the matrix and the workloads do not: run in a fixed order.
pub fn isolated_cases(tally: &mut Tally, ids: &mut Ids, build: &str) -> Res<()> {
    isolation(tally)?;
    lineage(tally)?;
    raw_reuse(tally)?;
    revoked_pinned_page(tally, ids)?;
    discarded_overlay(tally)?;
    cancelled_effects(tally, ids)?;
    empty_restart(tally)?;
    regimes(tally)?;
    witnesses(tally)?;
    positions(tally)?;
    environment_change(tally, build)?;
    transform_change(tally)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolation_misses_across_principal_and_project_and_hits_in_scope() {
        let mut tally = Tally::default();
        isolation(&mut tally).unwrap();
        assert_eq!(tally.get("cross_principal_attempts"), 1);
        assert_eq!(tally.get("cross_project_attempts"), 1);
        assert_eq!(tally.get("cross_principal_hits"), 0);
        assert_eq!(tally.get("cross_project_hits"), 0);
        assert_eq!(tally.get("isolation_control_hit_tokens"), 48);
        assert_eq!(tally.get("bitwise_mismatch_count"), 0);
    }

    #[test]
    fn lineage_holds_distinct_entries_and_admits_only_an_exact_match() {
        let mut tally = Tally::default();
        lineage(&mut tally).unwrap();
        assert_eq!(tally.get("cross_lineage_hits"), 0);
        assert!(tally.get("same_tokens_distinct_lineage_entries") >= 2);
        assert_eq!(tally.get("lineage_exact_match_hit_tokens"), 32);
        assert_eq!(tally.get("same_tokens_different_generation_cases"), 1);
    }

    #[test]
    fn a_revoked_pinned_page_is_never_served_and_every_byte_is_reclaimed() {
        let mut tally = Tally::default();
        let mut ids = Ids::default();
        revoked_pinned_page(&mut tally, &mut ids).unwrap();
        assert_eq!(tally.get("revoked_pinned_page_attempts"), 2);
        assert_eq!(tally.get("revoked_pinned_page_served_after_revocation"), 0);
        assert!(tally.get("revoked_pinned_page_bytes_total") > 0);
        assert_eq!(
            tally.get("revoked_pinned_page_bytes_reclaimed_after_release"),
            tally.get("revoked_pinned_page_bytes_total")
        );
        assert_eq!(tally.get("unreachable_revoked_page_bytes_unreclaimed"), 0);
        assert!(tally.get("revoked_pinned_page_tombstoned") >= 1);
        assert_eq!(ids.cancelled, ids.restarted);
    }

    #[test]
    fn a_discarded_overlay_is_neither_read_nor_inserted() {
        let mut tally = Tally::default();
        discarded_overlay(&mut tally).unwrap();
        assert_eq!(tally.get("discarded_overlay_entry_attempts"), 2);
        assert_eq!(tally.get("discarded_overlay_entry_hits"), 0);
        assert_eq!(tally.get("overlay_inserts_refused_as_not_live"), 1);
    }

    #[test]
    fn a_cancelled_run_has_no_effect_and_the_control_is_admitted() {
        let mut tally = Tally::default();
        let mut ids = Ids::default();
        cancelled_effects(&mut tally, &mut ids).unwrap();
        assert_eq!(
            tally.get("effect_control_admitted"),
            1,
            "the control must be admitted"
        );
        assert_eq!(tally.get("cancelled_run_effect_attempts"), 3);
        assert_eq!(tally.get("cancelled_run_effects_admitted"), 0);
    }

    #[test]
    fn forged_and_stale_witnesses_are_rejected_with_their_own_reasons() {
        let mut tally = Tally::default();
        witnesses(&mut tally).unwrap();
        assert_eq!(tally.get("forged_witness_fork_attempts"), 2);
        assert_eq!(tally.get("forged_witness_forks_accepted"), 0);
        assert_eq!(tally.get("forged_witness_rejected_as_foreign"), 1);
        assert_eq!(tally.get("forged_witness_rejected_as_mismatch"), 1);
        assert_eq!(tally.get("stale_witness_replay_cases"), 1);
        assert_eq!(tally.get("stale_or_reused_witness_forks_admitted"), 0);
        assert_eq!(tally.get("stale_witness_rejected_as_revoked"), 1);
        assert!(tally
            .is_unsupported("witness_reuse_static_evidence_passing")
            .is_some());
    }

    #[test]
    fn the_regime_arms_keep_the_control_exact_and_recompute_across_a_transition() {
        let mut tally = Tally::default();
        regimes(&mut tally).unwrap();
        assert_eq!(tally.get("fixed_regime_decode_crossing_cases"), 1);
        assert_eq!(tally.get("fixed_regime_crossing_parity_failures"), 0);
        assert_eq!(tally.get("effective_regime_transition_cases"), 1);
        assert_eq!(
            tally.get("exact_outputs_across_effective_regime_transition_without_recompute"),
            0
        );
        assert_eq!(tally.get("regime_control_hit_tokens"), 16);
    }

    #[test]
    fn positions_and_environment_and_transform_changes_never_hit() {
        let mut tally = Tally::default();
        positions(&mut tally).unwrap();
        assert_eq!(tally.get("mismatched_position_cases"), 3);
        assert_eq!(tally.get("exact_hits_with_mismatched_positions"), 0);
        assert_eq!(tally.get("shifted_insert_refused_as_noncanonical"), 1);
        environment_change(&mut tally, "build-a").unwrap();
        assert_eq!(tally.get("cross_environment_restore_cases"), 3);
        assert_eq!(tally.get("exact_hits_across_environment"), 0);
        transform_change(&mut tally).unwrap();
        assert_eq!(tally.get("current_layer_kv_transform_change_cases"), 5);
        assert_eq!(tally.get("stale_exact_hits"), 0);
        // The layers below the change are still reused: 0 + 1 + 2 + 3 + 0.
        assert_eq!(tally.get("transform_change_unchanged_layers_reused"), 6);
    }

    #[test]
    fn raw_prefix_reuse_and_the_empty_restart_arm() {
        let mut tally = Tally::default();
        raw_reuse(&mut tally).unwrap();
        assert_eq!(tally.get("raw_identical_prefix_hits"), 1);
        empty_restart(&mut tally).unwrap();
        let (metrics, _) = tally.metrics();
        let arm = metrics
            .iter()
            .find(|(name, _)| name == "empty_restart_arm")
            .unwrap()
            .1
            .render();
        assert!(arm.contains("\"hit_tokens\":0"), "{arm}");
        assert!(arm.contains("\"equals_cold\":true"), "{arm}");
    }

    #[test]
    fn the_workloads_are_judged_against_their_own_binding_limit() {
        // The fill reaches the binding limit, so the extra prompts must evict.
        assert_eq!(
            params::OVER_CAPACITY_FILL_PROMPTS
                * (params::OVER_CAPACITY_PROMPT_TOKENS / params::CHUNK_TOKENS),
            4096
        );
        const { assert!(params::OVER_CAPACITY_EXTRA_PROMPTS > 0) };
        const { assert!(params::SATURATION_MAX_SESSIONS > params::OVER_CAPACITY_FILL_PROMPTS) };
    }
}
