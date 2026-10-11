//! Experiment E007, split-pod KV (the CPU part), as a harness: `ptr-bench split-pod-kv
//! <seed>` runs every case once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at another scale (`split-pod-kv <sequences> <seed>`) is no
//! evidence. The code under test is the CPU reference path: the stage execution of the tiny
//! seeded Qwen2 stack, the activation-frame schemas and chunk assembler of the PodWire crates
//! over the real request frame codec (in process, no network), and the downstream stage cache
//! revalidated against a `PtrRuntime` ledger and a placement controller.
//!
//! The exit status. Precedence, highest first (validity first):
//!
//! 1. 2 when a validity counter fired (a pin, the corpus selection, the procedure or the late
//!    replay window differs from the frozen run), whatever else fired: the run did not follow
//!    the preregistered procedure, so its product observations say nothing about the product.
//!    The aggregator calls such a seed INVALID.
//! 2. With every validity counter at 0: 1 when a falsifier is tripped by an *observed*
//!    product violation, even if a case could not run or a control did not trip in the same
//!    run (a control that runs the product with its protection removed cannot tell it from a
//!    product whose protection a defect has already removed; the record is printed). The
//!    aggregator lets a tripped falsifier take precedence over a dead control (the precedent
//!    of L003 and L004).
//! 3. 2 when the harness is otherwise at fault: a case that cannot run, a control that does
//!    not trip, a fixture that does not build. A product regression that kills a control
//!    without firing any falsifier ends here too: it is a gap of the plan that surfaces as
//!    INVALID, not as a pass.
//! 4. 1 when a coverage prerequisite is unmet (a vacuous run), 0 when every rule holds.

pub mod bitwise;
pub mod chunks;
pub mod controls;
pub mod large;
pub mod link;
pub mod metrics;
pub mod params;
pub mod rig;
pub mod smuggle;
pub mod stale;
pub mod wire;

use super::layered_kv::corpus;
use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::layered_kv::lab::{fail, Res};
use super::layered_kv::pins;
use bitwise::Frames;
use link::Link;
use rig::Rig;
use std::time::Instant;

/// Counters whose natural value is zero (violations, mismatches): a run in which none happens
/// still prints a zero.
const ZERO_BASED: [&str; 38] = [
    "bitwise_mismatch_count",
    "valid_frame_bitwise_mismatches",
    "large_frame_reassembled_bytes_mismatch_count",
    "large_frame_digest_mismatches",
    "wire_kv_handle_count",
    "wire_kv_page_bytes",
    "kv_pages_installed_from_wire",
    "non_allowlisted_payloads_admitted",
    "smuggling_probes_admitted",
    "activation_control_kv_installs",
    "positions_length_mismatches_admitted",
    "unsigned_chunks_admitted",
    "chain_mismatch_stores",
    "cross_operation_splices_admitted",
    "mislabelled_frame_digests_admitted",
    "cross_frame_mixing_count",
    "late_chunks_after_abandon_admitted",
    "tombstone_evictions_before_retention",
    "tombstone_bound_violations",
    "replayed_frames_admitted",
    "cross_session_replays_admitted",
    "duplicate_activation_deliveries",
    "expired_frames_admitted",
    "abandoned_frame_replays_admitted",
    "relabeled_chunks_admitted",
    "unbound_stage_frames_admitted",
    "global_cap_breaches",
    "non_exact_frames_admitted",
    "non_finite_frames_admitted",
    "chain_frame_mismatches_delivered",
    "boundary_tick_replays_admitted",
    "cache_slots_held_by_dead_entries",
    "attempts_without_typed_rejection",
    "verdict_mismatches",
    "stale_upstream_kv_served",
    "outputs_after_upstream_invalidation",
    "stale_placement_frames_admitted",
    "buffered_bytes_after_release",
];

/// Counters that are attempts or cases but may legitimately be reached by no event in a
/// reduced run; seeded at zero so the metric prints.
const COUNTED: [&str; 11] = [
    "activation_frames_crossed",
    "valid_frames_sent",
    "frames_with_consistent_whole_frame_digest",
    "downstream_reuse_cases",
    "large_frame_cases",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
    "buffered_bytes_before_abandon",
    "abandoned_chunks_sent",
    "abandoned_chunks_released",
];

/// How big the run is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    Custom(u64),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Custom(_) => "custom",
        }
    }

    pub fn sequences(self) -> u64 {
        match self {
            Self::Pinned => params::SEQUENCES,
            Self::Custom(count) => count,
        }
    }
}

/// The seeds the manifest declares, read from its `seeds = [...]` line.
fn declared_seeds() -> Vec<u64> {
    params::EXPERIMENT_TOML
        .lines()
        .find(|line| line.trim_start().starts_with("seeds"))
        .and_then(|line| line.split_once('['))
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| {
            inner
                .split(',')
                .filter_map(|seed| seed.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn prompts() -> Vec<Vec<u32>> {
    (0..params::SEQUENCES).map(bitwise::prompt).collect()
}

fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &prompts())
}

/// The pins of the model, checked from the stack the cases run.
fn verify_pins(tally: &mut Tally, rig: &Rig) -> Json {
    let config = rig.stack.config();
    let layer = &config.layer;
    let mut mismatches = 0;
    let mut check = |ok: bool| mismatches += i64::from(!ok);
    check(config.num_layers as u64 == params::TEST_MODEL_LAYERS);
    check(layer.hidden_size as u64 == params::HIDDEN_SIZE);
    check(layer.num_attention_heads as u64 == params::NUM_HEADS);
    check(layer.num_key_value_heads as u64 == params::NUM_KV_HEADS);
    check((layer.hidden_size / layer.num_attention_heads) as u64 == params::HEAD_DIM);
    check(config.vocab_size as u64 == params::VOCAB_SIZE);
    check(layer.intermediate_size as u64 == params::INTERMEDIATE_SIZE);
    check(format!("{:?}", layer.rope_theta) == params::ROPE_THETA);
    check(format!("{:?}", config.key_encoding) == params::KEY_ENCODING);
    check(params::SPLIT_LAYER > 0 && params::SPLIT_LAYER < config.num_layers as u64);
    let digest = pins::canonical_digest(&pins::shape_fields());
    check(digest == params::TINY_MODEL_CONFIG_DIGEST);
    check(rig.stack.moe().is_none());
    tally.add("model_pin_mismatches", mismatches);
    Json::obj([
        ("tiny_model_config_digest", Json::str(digest)),
        ("pinned_digest", Json::str(params::TINY_MODEL_CONFIG_DIGEST)),
        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
        ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
        ("weights", Json::str(params::WEIGHTS)),
        ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
        ("boundary_dtype", Json::str(params::BOUNDARY_DTYPE)),
        (
            "base_checkpoint",
            Json::str("none: seeded random weights, no checkpoint"),
        ),
        ("mismatches", Json::Int(mismatches)),
    ])
}

fn selection(tally: &mut Tally) -> Json {
    let computed = selection_digest();
    let matches = computed == params::SELECTION_DIGEST;
    tally.add("selection_digest_mismatches", i64::from(!matches));
    Json::obj([
        ("corpus", Json::str(params::CORPUS)),
        ("rule", Json::str(params::CORPUS_SELECTION_RULE)),
        (
            "generator_seed",
            Json::Int(params::CORPUS_GENERATOR_SEED as i64),
        ),
        ("domain", Json::str(params::SELECTION_DOMAIN)),
        ("computed", Json::str(computed)),
        ("pinned", Json::str(params::SELECTION_DIGEST)),
        ("matches", Json::Bool(matches)),
        ("sequences", Json::Int(params::SEQUENCES as i64)),
    ])
}

/// The procedure parameters as the run observed them; a difference from the pinned value is
/// a procedure mismatch.
fn procedure(tally: &mut Tally, seed: u64, scale: Scale, link: &Link) -> Json {
    let policy = link.assembler.policy();
    let observed = [
        (policy.reorder_window as u64, params::REORDER_WINDOW),
        (
            policy.max_incomplete_frames_per_peer as u64,
            params::MAX_INCOMPLETE_FRAMES_PER_PEER,
        ),
        (
            policy.max_chunks_in_flight as u64,
            params::MAX_CHUNKS_IN_FLIGHT,
        ),
        (
            policy.max_frame_tombstones_per_principal as u64,
            params::MAX_FRAME_TOMBSTONES_PER_PRINCIPAL,
        ),
        (
            policy.max_frame_deadline_lifetime_ms,
            params::MAX_FRAME_DEADLINE_LIFETIME_MS,
        ),
        (policy.max_clock_skew_ms, params::MAX_CLOCK_SKEW_MS),
        (policy.replay_window_ms, params::REPLAY_WINDOW_MS),
        (policy.max_frame_bytes, params::MAX_ASSEMBLED_FRAME_BYTES),
        (
            policy.max_total_buffered_bytes,
            params::MAX_TOTAL_BUFFERED_BYTES,
        ),
        (ptr_podwire::MAX_BODY_BYTES as u64, params::MAX_BODY_BYTES),
    ];
    let mismatches = observed
        .iter()
        .filter(|(seen, pinned)| seen != pinned)
        .count() as i64;
    tally.add("procedure_mismatches", mismatches);
    let seeds = declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        ("seeds", Json::ints(seeds.iter().map(|seed| *seed as i64))),
        ("sequences", Json::Int(scale.sequences() as i64)),
        (
            "test_model_layers",
            Json::Int(params::TEST_MODEL_LAYERS as i64),
        ),
        ("split_layer", Json::Int(params::SPLIT_LAYER as i64)),
        (
            "layers_above_split",
            Json::Int(params::TEST_MODEL_LAYERS as i64 - params::SPLIT_LAYER as i64),
        ),
        ("hidden_size", Json::Int(params::HIDDEN_SIZE as i64)),
        ("vocab_size", Json::Int(params::VOCAB_SIZE as i64)),
        ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
        ("decode_steps", Json::Int(params::DECODE_STEPS as i64)),
        (
            "large_frame_chunks",
            Json::Int(params::LARGE_FRAME_CHUNKS as i64),
        ),
        ("reorder_window", Json::Int(policy.reorder_window as i64)),
        (
            "frame_deadline_ms",
            Json::Int(params::FRAME_DEADLINE_MS as i64),
        ),
        (
            "max_incomplete_frames_per_peer",
            Json::Int(policy.max_incomplete_frames_per_peer as i64),
        ),
        (
            "max_chunks_in_flight",
            Json::Int(policy.max_chunks_in_flight as i64),
        ),
        (
            "max_frame_tombstones_per_principal",
            Json::Int(policy.max_frame_tombstones_per_principal as i64),
        ),
        (
            "replay_window_ms",
            Json::Int(policy.replay_window_ms as i64),
        ),
        (
            "max_clock_skew_ms",
            Json::Int(policy.max_clock_skew_ms as i64),
        ),
        (
            "max_frame_deadline_lifetime_ms",
            Json::Int(policy.max_frame_deadline_lifetime_ms as i64),
        ),
        ("boundary_dtype", Json::str(params::BOUNDARY_DTYPE)),
        ("transport", Json::str(params::TRANSPORT)),
        (
            "adversarial_attempt_kinds",
            Json::strs(&params::ADVERSARIAL_ATTEMPT_KINDS),
        ),
        ("valid_frame_cases", Json::strs(&params::VALID_FRAME_CASES)),
        (
            "invalidation_coverage_matrix_cells",
            Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_CELLS),
        ),
    ])
}

/// Whether an observed product violation fired. Not the pins, the selection, the procedure or
/// the late window (the harness's own) and not the sent-versus-observed comparisons, which an
/// aborted case leaves unobserved.
fn observed_product_fault(get: &dyn Fn(&str) -> i64) -> bool {
    metrics::product_counters().iter().any(|name| get(name) > 0)
}

/// Whether a validity counter fired: the run did not follow the preregistered procedure.
fn validity_fired(get: &dyn Fn(&str) -> i64) -> bool {
    metrics::HARNESS_COUNTERS.iter().any(|name| get(name) > 0)
}

/// Whether a product violation is the finding of this run: observed, and no validity counter
/// fired (validity first: an invalid run says nothing about the product).
fn product_finding(get: &dyn Fn(&str) -> i64) -> bool {
    !validity_fired(get) && observed_product_fault(get)
}

/// Whether the cases and controls leave a run that can be judged. `Err` is the harness's
/// fault (exit 2); `Ok` lets the record be printed, with exit 1 if a falsifier fired.
///
/// A case that could not run, or a control that did not trip, while an observed product
/// violation fired is the product's fault: a control that runs the product with its
/// protection removed cannot tell it from a product whose protection a defect has already
/// removed, so the violation is the finding and takes precedence.
fn run_standing(
    get: &dyn Fn(&str) -> i64,
    control_trips: &dyn Fn(&str) -> i64,
    failures: &[String],
) -> Result<(), String> {
    if !failures.is_empty() {
        if !product_finding(get) {
            return Err(format!("a case could not run: {}", failures.join("; ")));
        }
        return Ok(());
    }
    for control in metrics::CONTROLS {
        if control_trips(control) < 1 && !product_finding(get) {
            return Err(format!(
                "the control {control} did not trip: its probe proves nothing"
            ));
        }
    }
    Ok(())
}

pub struct Report {
    pub line: String,
    #[cfg_attr(not(test), allow(dead_code))]
    pub json: Json,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    /// The rules that say the record describes the frozen run; a failing one is the harness's
    /// fault.
    pub validity: Vec<Check>,
    pub gaps: Gaps,
}

impl Report {
    pub fn coverage_unmet(&self) -> usize {
        count(&self.coverage, Status::Fail)
    }

    pub fn falsifiers_tripped(&self) -> usize {
        count(&self.falsifiers, Status::Fail)
    }

    pub fn validity_violated(&self) -> usize {
        count(&self.validity, Status::Fail)
    }

    pub fn clean(&self) -> bool {
        self.coverage_unmet() == 0
            && self.falsifiers_tripped() == 0
            && self.validity_violated() == 0
            && self.gaps.is_empty()
    }

    /// The exit status of the printed record: 2 when a validity counter is violated (whatever
    /// else fired: the seed is INVALID), else 1 when a falsifier is tripped, 1 when only a
    /// coverage prerequisite is unmet or a metric has a gap, else 0.
    pub fn exit_status(&self) -> i32 {
        if self.validity_violated() > 0 {
            2
        } else if self.falsifiers_tripped() > 0 {
            1
        } else if self.clean() {
            0
        } else {
            1
        }
    }
}

/// Run every case for `seed`.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut tally = Tally::new(metrics::LIST);
    for name in ZERO_BASED.iter().chain(&COUNTED) {
        tally.add(name, 0);
    }
    tally.seed_group("smuggling_probe_attempts", &params::SMUGGLING_PROBE_KINDS);
    tally.seed_group("stale_upstream_attempts", &params::INVALIDATION_TYPES);
    tally.seed_group(
        "inflight_split_request_race_cases",
        &params::INVALIDATION_TYPES,
    );
    tally.seed_group(
        "invalidation_matrix_cases",
        &params::INVALIDATION_COVERAGE_MATRIX_CELLS,
    );
    tally.seed_group(
        "invalidation_matrix_stale_hits",
        &params::INVALIDATION_COVERAGE_MATRIX_CELLS,
    );
    tally.seed_group("control_trips", &metrics::CONTROLS);
    for name in [
        "stale_upstream_entries_before_revocation",
        "revoke_messages_dropped_or_delayed",
        "stale_worker_frame_race_cases",
        "tombstones_retained_at_saturation",
        "abandoned_frames_in_saturation_case",
        "frames_accepted_after_tombstone_expiry",
    ] {
        tally.add(name, 0);
    }
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut lap = Instant::now();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };

    let rig = Rig::new()?;
    let mut link = Link::new(&rig, seed, 1)?;
    let pins_json = verify_pins(&mut tally, &rig);
    let selection_json = selection(&mut tally);
    let procedure_json = procedure(&mut tally, seed, scale, &link);
    phase("pins_and_selection", &mut lap);

    let mut frames = Frames::default();
    let mut delays: Vec<i64> = Vec::new();
    // A case that cannot run is a failure of the run. When a falsifier counter has fired as
    // well, the product is at fault and the record is still printed (status 1); otherwise it
    // is the harness's (status 2).
    let mut failures: Vec<String> = Vec::new();
    macro_rules! case {
        ($name:expr, $body:expr) => {
            if let Err(error) = $body {
                failures.push(format!("{}: {error}", $name));
            }
        };
    }

    case!(
        "bitwise",
        bitwise::run(&mut tally, &rig, &mut link, scale.sequences(), &mut frames)
    );
    phase("bitwise", &mut lap);
    for index in 0..3.min(scale.sequences().max(1)) {
        case!(
            "downstream_reuse",
            bitwise::reuse(&mut tally, &rig, &mut link, index)
        );
    }
    phase("downstream_reuse", &mut lap);

    let mut inventory = None;
    let mut probe_page: Vec<u8> = Vec::new();
    match smuggle::pods(&rig) {
        Ok(mut pods) => {
            let start = link.wire.records.len();
            case!(
                "smuggling",
                smuggle::run(&mut tally, &rig, &mut link, &mut pods)
            );
            match smuggle::inventory_of(&pods.stage_a, &pods.stage_b) {
                Ok(held) => {
                    let (bytes, handles) = wire::scan(&link.wire.records[start..], &held);
                    tally.add("wire_kv_page_bytes", bytes);
                    tally.add("wire_kv_handle_count", handles);
                    probe_page = held.pages().first().cloned().unwrap_or_default();
                    inventory = Some(held);
                }
                Err(error) => failures.push(format!("smuggling inventory: {error}")),
            }
        }
        Err(error) => failures.push(format!("smuggling pods: {error}")),
    }
    phase("smuggling", &mut lap);

    let start = link.wire.records.len();
    case!(
        "large_frame",
        large::run(&mut tally, &rig, &mut link, seed, &mut frames)
    );
    if let Some(held) = &inventory {
        let (bytes, handles) = wire::scan(&link.wire.records[start..], held);
        tally.add("wire_kv_page_bytes", bytes);
        tally.add("wire_kv_handle_count", handles);
    }
    phase("large_frame", &mut lap);

    case!("attacks", chunks::attacks(&mut tally, &rig, seed));
    case!("bounds", chunks::bounds(&mut tally, &rig, seed));
    case!(
        "replays",
        chunks::replays(&mut tally, &rig, seed, &mut frames)
    );
    case!(
        "abandonment",
        chunks::abandonment(&mut tally, &rig, seed, &mut frames, &mut delays)
    );
    case!("saturation", chunks::saturation(&mut tally, &rig, seed));
    case!("shared_slots", chunks::shared_slots(&mut tally, &rig, seed));
    case!(
        "interleaving",
        chunks::interleaving(&mut tally, &rig, seed, &mut frames)
    );
    case!("chains", chains(&mut tally, &rig, seed));
    phase("chunk_protocol", &mut lap);

    case!("stale_upstream", stale::run(&mut tally, seed));
    phase("stale_upstream", &mut lap);

    case!(
        "control: naive cache",
        controls::naive_cache(&mut tally, seed)
    );
    case!(
        "controls: assemblers",
        (|| -> Res<()> {
            let mut l = Link::new(&rig, seed, 950)?;
            let a = {
                let frame = controls::frame(&rig, 1);
                let (_, chunks) =
                    l.chunks(&rig, &frame, params::PROBE_CHUNK_BYTES as usize, 100)?;
                (frame, chunks)
            };
            let b = {
                let frame = controls::frame(&rig, 2);
                let (_, chunks) =
                    l.chunks(&rig, &frame, params::PROBE_CHUNK_BYTES as usize, 100)?;
                (frame, chunks)
            };
            controls::chain_keyed_assembler(&mut tally, &a, &b);
            controls::tombstone_free(&mut tally, &rig, &a.1)?;
            controls::denylist(&mut tally, &probe_page);
            Ok(())
        })()
    );
    run_standing(
        &|name| tally.get(name),
        &|control| {
            tally
                .group("control_trips")
                .into_iter()
                .find(|(name, _)| name == control)
                .map_or(0, |(_, value)| value)
        },
        &failures,
    )?;
    tally.set("case_failures", failures.len() as i64);
    phase("controls", &mut lap);

    // The sets and digests.
    tally.set_json(
        "valid_sent_frame_id_set_digest",
        Json::str(bitwise::digest(&frames.ids)),
    );
    tally.set_json(
        "consistent_digest_frame_id_set_digest",
        Json::str(bitwise::digest(&frames.consistent_ids)),
    );
    let in_range = delays
        .iter()
        .filter(|delay| {
            **delay >= params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[0] as i64
                && **delay < params::LATE_WINDOW_REPLAY_DELAY_RANGE_MS[1] as i64
        })
        .count() as i64;
    tally.set_json(
        "late_window_replay_delays_ms",
        Json::ints(delays.iter().copied()),
    );
    tally.unsupported(
        "restore_path",
        "no KV is restored on a Pod, and the downstream cache has no persistence (no snapshot, tier write or restore): the restore path of the invalidation coverage matrix does not apply",
    );
    tally.unsupported(
        "network_transport",
        "frames cross the real PodWire request frame codec in process; no socket, no iroh and no second host is involved (E008)",
    );
    let unsupported_names = tally.unsupported_names();
    let (printed, gaps) = tally.metrics();
    let get = |name: &str| -> i64 {
        printed
            .iter()
            .find(|(key, _)| key == name)
            .and_then(|(_, value)| match value {
                Json::Int(number) => Some(*number),
                _ => None,
            })
            .unwrap_or(0)
    };
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("E007")),
        (
            "status_note".to_string(),
            Json::str("one seed of the E007 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
        (
            "elapsed_ms".to_string(),
            Json::Float(started.elapsed().as_secs_f64() * 1000.0),
        ),
    ];
    let mirrors = vec![
        ("late_window_delays_in_range", in_range),
        (
            "late_window_delays_out_of_range",
            delays.len() as i64 - in_range,
        ),
        (
            "abandoned_chunks_unreleased",
            (get("abandoned_chunks_sent") - get("abandoned_chunks_released")).max(0),
        ),
        (
            "frames_without_consistent_digest",
            (get("valid_frames_sent") - get("frames_with_consistent_whole_frame_digest")).abs(),
        ),
    ];
    pairs.extend(printed.clone());
    for (name, value) in mirrors {
        pairs.push((name.to_string(), Json::Int(value)));
    }
    let string_of = |name: &str| -> String {
        printed
            .iter()
            .find(|(key, _)| key == name)
            .and_then(|(_, value)| match value {
                Json::Str(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    };
    pairs.push((
        "abandoned_id_set_mismatch".to_string(),
        Json::Int(i64::from(
            string_of("abandoned_chunk_id_set_digest") != string_of("released_chunk_id_set_digest"),
        )),
    ));
    pairs.push((
        "valid_id_set_mismatch".to_string(),
        Json::Int(i64::from(
            string_of("valid_sent_frame_id_set_digest")
                != string_of("consistent_digest_frame_id_set_digest"),
        )),
    ));
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        let hits = tally
            .group("invalidation_matrix_stale_hits")
            .into_iter()
            .find(|(name, _)| name == cell)
            .map_or(0, |(_, value)| value);
        pairs.push((metrics::cell_counter(cell), Json::Int(hits)));
    }
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([("phases", Json::Obj(phases))]),
    ));
    pairs.push(("procedure".to_string(), procedure_json));
    pairs.push(("model_pins".to_string(), pins_json));
    pairs.push(("selection".to_string(), selection_json));
    pairs.push(("wire_capture".to_string(), link.wire.json()));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "wire_kv_page_bytes",
                Json::str(params::WIRE_KV_PAGE_BYTES_DEFINITION),
            ),
            (
                "bitwise_mismatch_count",
                Json::str("logits of every prompt row and decode step, and the K and V bytes of every layer on each pod, of the whole stack on one pod against (a) the two stages in process through the boundary cast and (b) the two stages with every frame crossing the wire; plus the continuation from a downstream cache hit against the recomputation"),
            ),
            (
                "typed_rejection",
                Json::str("an adversarial attempt must end in a Rejection (or ChainFault) of the variant a hand-written table pins; any other outcome is attempts_without_typed_rejection or verdict_mismatches"),
            ),
            (
                "frame_rate",
                Json::str("a completed frame holds one of the principal's 64 table slots until its signed deadline plus the skew bound; the cases sign deadlines 100 ms ahead and send a frame every 25 ms, so at most 45 slots are in use"),
            ),
        ]),
    ));
    pairs.push(("internal".to_string(), tally.internal()));
    let root = Json::Obj(pairs.clone());
    let coverage = judge(&root, &unsupported_names, &metrics::coverage());
    let falsifiers = judge(&root, &unsupported_names, &metrics::falsifiers());
    let validity = judge(&root, &unsupported_names, &metrics::validity());
    let tripped = count(&falsifiers, Status::Fail);
    pairs.insert(6, ("hard_failures".to_string(), Json::Int(tripped as i64)));
    pairs.push((
        "coverage".to_string(),
        Json::Arr(coverage.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "falsifiers".to_string(),
        Json::Arr(falsifiers.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "validity".to_string(),
        Json::Arr(validity.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "summary".to_string(),
        Json::obj([
            (
                "coverage_unmet",
                Json::Int(count(&coverage, Status::Fail) as i64),
            ),
            (
                "coverage_unsupported",
                Json::Int(count(&coverage, Status::Unsupported) as i64),
            ),
            ("falsifiers_tripped", Json::Int(tripped as i64)),
            (
                "validity_violated",
                Json::Int(count(&validity, Status::Fail) as i64),
            ),
            (
                "falsifiers_unsupported",
                Json::Int(count(&falsifiers, Status::Unsupported) as i64),
            ),
        ]),
    ));
    pairs.push((
        "harness_gaps".to_string(),
        Json::Arr(
            gaps.iter()
                .map(|(metric, reason)| {
                    Json::obj([
                        ("metric", Json::str(metric.clone())),
                        ("reason", Json::str(reason.clone())),
                    ])
                })
                .collect(),
        ),
    ));
    let json = Json::Obj(pairs);
    Ok(Report {
        line: json.render(),
        json,
        coverage,
        falsifiers,
        validity,
        gaps,
    })
}

/// The store-level half of `chain_mismatch`: frames of different chains cannot back one
/// entry.
fn chains(tally: &mut Tally, _rig: &Rig, seed: u64) -> Res<()> {
    use ptr_runtime::{admit_frame, SplitStageCache, StoreError};
    let mut rig = Rig::new()?;
    let mut link = Link::new(&rig, seed, 960)?;
    let first = rig.frame(&vec![0.5; params::HIDDEN_SIZE as usize], 1, 0);
    let first = admit_frame(&first, ptr_model_api::KvReuseMode::Exact, &rig.live())
        .map_err(fail("admitting a frame"))?;
    rig.replace_stage()?;
    let second = rig.frame(&vec![0.25; params::HIDDEN_SIZE as usize], 1, 1);
    let second = admit_frame(&second, ptr_model_api::KvReuseMode::Exact, &rig.live())
        .map_err(fail("admitting a frame"))?;
    let _ = &mut link;
    let mut cache: SplitStageCache<u8> = SplitStageCache::new(2);
    tally.add("chain_mismatch_attempts", 1);
    match cache.store(&[&first, &second], &rig.own_chain(), 0, &rig.live()) {
        Err(StoreError::ChainMismatch) => {}
        Err(_) => tally.add("verdict_mismatches", 1),
        Ok(_) => tally.add("chain_mismatch_stores", 1),
    }
    Ok(())
}

fn print_and_exit(report: &Report) {
    println!("{}", report.line);
    for check in report
        .coverage
        .iter()
        .chain(&report.falsifiers)
        .chain(&report.validity)
    {
        if check.status == Status::Fail {
            eprintln!("{} ({})", check.rule, check.observed.render());
        }
    }
    for (metric, reason) in &report.gaps {
        eprintln!("harness gap {metric}: {reason}");
    }
    match report.exit_status() {
        0 => {}
        status => std::process::exit(status),
    }
}

/// The preflight: the pins and the controls on a small run, no case of the experiment. Exit
/// status 2 on any failure.
fn preflight() -> Result<Json, String> {
    let rig = Rig::new()?;
    let mut tally = Tally::new(metrics::LIST);
    let pins_json = verify_pins(&mut tally, &rig);
    if tally.get("model_pin_mismatches") != 0 {
        return Err("a model pin differs from the configuration".to_string());
    }
    let selection_json = selection(&mut tally);
    if tally.get("selection_digest_mismatches") != 0 {
        return Err("the corpus selection digest differs from the pinned digest".to_string());
    }
    let mut link = Link::new(&rig, 0, 1)?;
    let mut frames = Frames::default();
    bitwise::run(&mut tally, &rig, &mut link, 1, &mut frames)?;
    if tally.get("bitwise_mismatch_count") != 0 {
        return Err("the split run differs from the single-pod run".to_string());
    }
    controls::naive_cache(&mut tally, 0)?;
    if tally.get("control_trips.naive_downstream_cache_serves_stale") < 1 {
        return Err("the naive downstream cache control did not trip".to_string());
    }
    Ok(Json::obj([
        ("preflight", Json::str("ok")),
        ("benchmark", Json::str(params::BENCHMARK)),
        ("model_pins", pins_json),
        ("selection", selection_json),
        (
            "frames_crossed",
            Json::Int(tally.get("activation_frames_crossed")),
        ),
    ]))
}

/// `split-pod-kv <seed>` | `split-pod-kv <sequences> <seed>` | `split-pod-kv --preflight`.
pub fn run(args: &[String]) {
    if args.first().map(String::as_str) == Some("--preflight") && args.len() == 1 {
        match preflight() {
            Ok(json) => println!("{}", json.render()),
            Err(reason) => {
                eprintln!("{}: preflight: {reason}", params::BENCHMARK);
                std::process::exit(2);
            }
        }
        return;
    }
    let parsed: Result<(Scale, u64), String> = match args {
        [seed] => seed
            .parse::<u64>()
            .map(|seed| (Scale::Pinned, seed))
            .map_err(|_| format!("seed must be a non-negative integer, got {seed:?}")),
        [sequences, seed] => sequences
            .parse::<u64>()
            .ok()
            .filter(|count| *count >= 1)
            .zip(seed.parse::<u64>().ok())
            .map(|(count, seed)| (Scale::Custom(count), seed))
            .ok_or_else(|| format!("usage: split-pod-kv <sequences >= 1> <seed>, got {args:?}")),
        _ => Err("usage: split-pod-kv <seed> | <sequences> <seed> | --preflight".to_string()),
    };
    let (scale, seed) = match parsed {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("ptr-bench: {message}");
            std::process::exit(2);
        }
    };
    let seeds = declared_seeds();
    if !seeds.contains(&seed) {
        eprintln!("seed {seed} is not one of the manifest's seeds {seeds:?}; this is a dry run");
    }
    match execute(seed, scale) {
        Ok(report) => print_and_exit(&report),
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const PREREGISTRATION: &str =
        include_str!("../../../../../experiments/system/E007-split-pod-kv/PREREGISTRATION.md");
    const CRITERIA: &str =
        include_str!("../../../../../experiments/system/E007-split-pod-kv/criteria.toml");
    const MUTATIONS: &str =
        include_str!("../../../../../experiments/system/E007-split-pod-kv/tests/mutations.toml");

    /// One reduced run (a single sequence), shared by the tests that read its output.
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(17, Scale::Custom(1)).expect("a reduced run"))
    }

    fn rules(table: &toml::Table, key: &str) -> Vec<String> {
        table[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is an array"))
            .iter()
            .map(|rule| rule.as_str().expect("a rule is a string").to_string())
            .collect()
    }

    /// `print_rules` writes the two lists in the shape of `criteria.toml`'s arrays.
    #[test]
    #[ignore = "prints the rule lists for criteria.toml"]
    fn print_rules() {
        for (name, list) in [
            ("coverage", metrics::coverage()),
            ("falsifiers", metrics::falsifiers()),
            ("validity", metrics::validity()),
        ] {
            println!("{name} = [");
            for rule in list {
                println!("  \"{rule}\",");
            }
            println!("]");
        }
    }

    #[test]
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
        assert_eq!(declared_seeds(), params::SEEDS.to_vec());
    }

    #[test]
    fn the_pinned_selection_digest_is_what_the_corpus_generator_produces() {
        assert_eq!(selection_digest(), params::SELECTION_DIGEST);
        assert!(PREREGISTRATION.contains(params::SELECTION_DIGEST));
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(rules(&table, "validity"), metrics::validity());
        assert_eq!(table["experiment"].as_str(), Some("E007"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_a_path_the_harness_prints() {
        let report = reduced_report();
        for rule in metrics::coverage()
            .iter()
            .chain(&metrics::falsifiers())
            .chain(&metrics::validity())
        {
            let tokens: Vec<&str> = rule.split(' ').collect();
            assert_eq!(tokens.len(), 3, "{rule}");
            for token in [tokens[0], tokens[2]] {
                let literal = token
                    .strip_prefix('-')
                    .unwrap_or(token)
                    .bytes()
                    .all(|byte| byte.is_ascii_digit());
                if !literal {
                    assert!(
                        emit::resolve(&report.json, token).is_some(),
                        "{rule}: {token} does not resolve in the harness JSON"
                    );
                }
            }
        }
    }

    /// The value at `path` replaced.
    fn with(root: &Json, path: &str, value: Json) -> Json {
        fn go(at: &Json, parts: &[&str], value: &Json) -> Json {
            match (at, parts.split_first()) {
                (_, None) => value.clone(),
                (Json::Obj(pairs), Some((first, rest))) => Json::Obj(
                    pairs
                        .iter()
                        .map(|(key, inner)| {
                            if key == first {
                                (key.clone(), go(inner, rest, value))
                            } else {
                                (key.clone(), inner.clone())
                            }
                        })
                        .collect(),
                ),
                _ => at.clone(),
            }
        }
        go(root, &path.split('.').collect::<Vec<_>>(), &value)
    }

    #[test]
    fn every_rule_can_fail_when_its_counter_is_set_to_a_violation() {
        let report = reduced_report();
        for rule in metrics::coverage()
            .iter()
            .chain(&metrics::falsifiers())
            .chain(&metrics::validity())
        {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let (left, op, right) = (tokens[0], tokens[1], tokens[2]);
            let violating = match emit::resolve(&report.json, left) {
                Some(Json::Str(_)) => with(&report.json, left, Json::str("sha256:different")),
                _ => {
                    let bound = right.parse::<i64>().ok().or_else(|| {
                        match emit::resolve(&report.json, right) {
                            Some(Json::Int(number)) => Some(*number),
                            _ => None,
                        }
                    });
                    let bound = bound.expect("a bound");
                    let value = if op == ">=" { bound - 1 } else { bound + 1 };
                    with(&report.json, left, Json::Int(value))
                }
            };
            assert_eq!(
                emit::evaluate(&violating, rule),
                Ok(false),
                "{rule} did not fail when violated"
            );
        }
    }

    #[test]
    fn the_reduced_run_trips_no_falsifier_and_no_control_is_dead() {
        let report = reduced_report();
        assert_eq!(report.falsifiers_tripped(), 0, "{:?}", report.falsifiers);
        assert_eq!(report.validity_violated(), 0, "{:?}", report.validity);
        // A reduced run is vacuous at the pinned scale (status 1) but never the harness's fault.
        assert_ne!(report.exit_status(), 2);
        assert!(report.gaps.is_empty(), "{:?}", report.gaps);
        for control in metrics::CONTROLS {
            let path = format!("control_trips.{control}");
            let Some(Json::Int(trips)) = emit::resolve(&report.json, &path) else {
                panic!("{path} is not printed");
            };
            assert!(*trips >= 1, "{control}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_record_apart_from_timing() {
        fn stable(json: &Json) -> String {
            let Json::Obj(pairs) = json else { panic!() };
            Json::Obj(
                pairs
                    .iter()
                    .filter(|(key, _)| key != "elapsed_ms" && key != "timing_ms")
                    .cloned()
                    .collect(),
            )
            .render()
        }
        let first = stable(&reduced_report().json);
        let second = execute(17, Scale::Custom(1)).expect("a second run");
        assert_eq!(first, stable(&second.json));
    }

    #[test]
    fn a_hard_counter_of_the_mutation_plan_is_a_top_level_integer_of_the_record() {
        let report = reduced_report();
        let plan: toml::Table = MUTATIONS.parse().expect("mutations.toml is TOML");
        let hard: Vec<String> = plan["hard_counters"]
            .as_array()
            .expect("hard_counters")
            .iter()
            .map(|name| name.as_str().expect("a name").to_string())
            .collect();
        assert_eq!(hard, metrics::hard_counters());
        for name in &hard {
            assert!(
                matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                "{name} is no top-level integer"
            );
        }
        // Every hard counter is expected by at least one mutation.
        let mutations = plan["mutation"].as_array().expect("mutations");
        for name in &hard {
            assert!(
                mutations.iter().any(|mutation| mutation["expect"]
                    .as_array()
                    .is_some_and(|expect| expect.iter().any(|e| e.as_str() == Some(name)))),
                "no mutation expects {name}"
            );
        }
    }

    #[test]
    fn a_control_that_does_not_trip_ends_the_run() {
        // The denylist control trips on a non-empty payload and never on an empty one.
        let mut tally = Tally::new(metrics::LIST);
        controls::denylist(&mut tally, &[]);
        assert_eq!(
            tally.get("control_trips.denylist_admits_mislabelled_payload"),
            0
        );
        controls::denylist(&mut tally, &[1]);
        assert_eq!(
            tally.get("control_trips.denylist_admits_mislabelled_payload"),
            1
        );
    }

    fn counters<'a>(fired: &'a [&'a str]) -> impl Fn(&str) -> i64 + 'a {
        move |name| i64::from(fired.contains(&name))
    }

    const ALL_TRIP: fn(&str) -> i64 = |_| 1;
    const NONE_TRIP: fn(&str) -> i64 = |_| 0;

    #[test]
    fn a_dead_control_or_a_failed_case_with_no_finding_is_the_harness_fault() {
        let quiet = counters(&[]);
        let failed = vec!["bitwise: boom".to_string()];
        assert!(run_standing(&quiet, &ALL_TRIP, &failed).is_err());
        assert!(run_standing(&quiet, &NONE_TRIP, &[]).is_err());
        assert!(run_standing(&quiet, &ALL_TRIP, &[]).is_ok());
    }

    #[test]
    fn an_observed_violation_takes_precedence_over_a_dead_control_or_a_failed_case() {
        // The defect_found branch: the record is printed (exit 1) instead of ending in 2.
        let fired = counters(&["replayed_frames_admitted"]);
        let failed = vec!["replays: boom".to_string()];
        assert!(run_standing(&fired, &NONE_TRIP, &[]).is_ok());
        assert!(run_standing(&fired, &ALL_TRIP, &failed).is_ok());
        assert!(observed_product_fault(&fired));
    }

    #[test]
    fn a_validity_counter_takes_precedence_over_an_observed_violation() {
        // Validity first: with a validity counter fired, the observed violation is no
        // finding; a dead control or a failed case ends in 2 (no record) as for the harness.
        for name in metrics::HARNESS_COUNTERS {
            let names = [name, "replayed_frames_admitted"];
            let fired = counters(&names);
            assert!(validity_fired(&fired), "{name}");
            assert!(!product_finding(&fired), "{name}");
            let failed = vec!["replays: boom".to_string()];
            assert!(run_standing(&fired, &NONE_TRIP, &[]).is_err(), "{name}");
            assert!(run_standing(&fired, &ALL_TRIP, &failed).is_err(), "{name}");
            // Every control tripped, every case ran: the record is printed and the exit
            // status of the report (2) decides.
            assert!(run_standing(&fired, &ALL_TRIP, &[]).is_ok(), "{name}");
        }
        let only_product = counters(&["replayed_frames_admitted"]);
        assert!(product_finding(&only_product));
    }

    #[test]
    fn the_harness_own_counters_and_the_aborted_comparisons_are_no_product_finding() {
        for name in metrics::HARNESS_COUNTERS
            .iter()
            .chain(&metrics::SENT_VERSUS_OBSERVED)
        {
            let fired = counters(std::slice::from_ref(name));
            assert!(!observed_product_fault(&fired), "{name}");
            let failed = vec!["a case: boom".to_string()];
            assert!(run_standing(&fired, &ALL_TRIP, &failed).is_err(), "{name}");
            assert!(run_standing(&fired, &NONE_TRIP, &[]).is_err(), "{name}");
        }
        // Every other hard counter is a product finding.
        let product = metrics::product_counters();
        assert_eq!(
            product.len() + metrics::HARNESS_COUNTERS.len() + metrics::SENT_VERSUS_OBSERVED.len(),
            metrics::hard_counters().len()
        );
    }

    fn check(rule: &str, status: Status) -> Check {
        Check {
            rule: rule.to_string(),
            observed: Json::Null,
            status,
        }
    }

    fn report_with(falsifier: Status, validity: Status, coverage: Status) -> Report {
        Report {
            line: String::new(),
            json: Json::Null,
            coverage: vec![check("c", coverage)],
            falsifiers: vec![check("f", falsifier)],
            validity: vec![check("v", validity)],
            gaps: Vec::new(),
        }
    }

    #[test]
    fn the_exit_status_puts_validity_before_a_falsifier_before_a_vacuous_run() {
        use Status::{Fail, Pass};
        assert_eq!(report_with(Pass, Pass, Pass).exit_status(), 0);
        assert_eq!(report_with(Fail, Pass, Pass).exit_status(), 1);
        // Validity first: a validity failure is INVALID whatever else fired.
        assert_eq!(report_with(Fail, Fail, Fail).exit_status(), 2);
        assert_eq!(report_with(Fail, Fail, Pass).exit_status(), 2);
        assert_eq!(report_with(Pass, Fail, Pass).exit_status(), 2);
        assert_eq!(report_with(Pass, Fail, Fail).exit_status(), 2);
        assert_eq!(report_with(Pass, Pass, Fail).exit_status(), 1);
    }
}
