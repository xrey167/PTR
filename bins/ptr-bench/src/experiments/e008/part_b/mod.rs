//! Experiment E008, Part B (CPU and loopback network): the chunked activation frames of E007
//! travel between a `PodHost` and several `PodClient` endpoints over loopback iroh (ptr-net,
//! podwire-backend), one chunk per request, and the receiver judges them. `ptr-bench
//! split-pod-loopback <seed>` runs every case once for one seed and prints exactly one JSON
//! object on stdout; it is compiled only with the `podwire-experiments` feature.
//!
//! What is under test, as the manifest's "Part B" says it:
//!
//! * the frames of E007, including the 16 MiB frame as five chunks of at most 3.5 MiB, arrive
//!   bit-identically over the real transport (`loopback_frames_delivered`,
//!   `loopback_bitwise_mismatch_count`);
//! * every typed rejection of E007 is unchanged when the chunk crossed a socket;
//! * the receiver admits only the allowlisted schemas on the real endpoint
//!   (`loopback_non_allowlisted_admitted`): by the host's access decision for a strict peer and by
//!   the assembler's own allowlist for a peer whose scope is deliberately too wide;
//! * a replay on a new connection, after a re-registration and under a new endpoint identity is refused with `FrameReplayed` (`loopback_replays_admitted`);
//! * the chunk signature is bound to the principal the connection authenticated, not to a key
//!   the harness registers (`loopback_unauthenticated_principal_admitted`): the assembler takes
//!   every payload under the peer id that `PodHost::serve_once` reports for the connection.
//!
//! What "the receiver" is here: the host's access policy plus a `FrameAssembler` that this harness
//! feeds under the authenticated peer id. No product code wires `FrameAssembler` into the
//! `PodHost` serve path yet; the harness does it from outside (the intake Pod hands the payload
//! over, the harness offers it under the principal `serve_once` reports).
//!
//! "Reconnect", three distinct things, each tested. A network reconnect: every request opens a new
//! authenticated connection, so the replays travel on connections that did not carry the original.
//! A re-registration of the same principal: the deployment calls `register_principal` again with
//! the same key and stage (the assembler's side only, not a network event). A new endpoint
//! identity with the same key and stage: a restarted runtime draws a new iroh key, and the
//! assembler hands the new principal id the unexpired tombstones; that case is gated like the
//! others and its admissions count toward `loopback_replays_admitted`. Tombstones are in memory:
//! a restarted receiver has none, and a replay is then bounded by the signed deadline plus skew,
//! which this harness does not cover.
//!
//! The assembler's clock is virtual (PodWire v1 carries no deadline), so no verdict depends on
//! how long a connection took; only the bounded timeout of one exchange is wall time. The exit
//! status: 2 when the harness cannot run a case, 1 when a falsifier is tripped or a coverage
//! prerequisite is unmet, 0 otherwise.
//!
//! This harness uses the v1 frame types of `ptr-protocol::activation`. The places that would
//! switch to v2 are marked `V2:` in `cases.rs` (the frame constructors) and `net.rs` (the scope of
//! the strict peer, which allowlists the chunk and policy type ids).

mod cases;
mod net;

use crate::experiments::e007::params;
use crate::experiments::e007::rig::Rig;
use crate::experiments::layered_kv::emit::Json;
pub use crate::experiments::layered_kv::lab::Res;
use cases::{Peers, H};
use net::{other_key, Net, Peer, Standing};
use std::collections::BTreeMap;
use std::time::Instant;

/// The seeds the manifest declares, read from its `seeds = [...]` line.
const EXPERIMENT_TOML: &str =
    include_str!("../../../../../../experiments/system/E008-split-pod-extensions/experiment.toml");

/// The name of the subcommand.
pub const BENCHMARK: &str = "split-pod-loopback";

/// The five metrics of the manifest this harness reports.
pub const METRICS: [&str; 5] = [
    "loopback_frames_delivered",
    "loopback_bitwise_mismatch_count",
    "loopback_non_allowlisted_admitted",
    "loopback_replays_admitted",
    "loopback_unauthenticated_principal_admitted",
];

/// The typed rejections of E007 (and the one Part B adds for an unregistered principal) that must
/// each be observed at least once on the wire.
const REJECTION_KINDS: [&str; 21] = [
    "bad_signature",
    "chain_mismatch",
    "duplicate_chunk",
    "out_of_window",
    "bad_chunk_digest",
    "frame_mismatch",
    "frame_digest_mismatch",
    "expired",
    "frame_replayed",
    "stage_not_bound",
    "chain_frame_mismatch",
    "policy_mismatch",
    "global_buffer_full",
    "too_many_incomplete_frames",
    "queue_overflow",
    "frame_abandoned",
    "frame_table_full",
    "not_allowlisted",
    "malformed",
    "invalid_frame",
    "unknown_principal",
];

/// Every attempt or case counter: at least 1, or the run is vacuous.
const COVERAGE: [&str; 38] = [
    "frames_delivered",
    "valid_frames_sent",
    "chunks_sent",
    "requests",
    "corpus_sequences",
    "large_frame_cases",
    "large_frame_chunks_sent",
    "unchunked_large_frame_attempts",
    "non_allowlisted_host_attempts",
    "non_allowlisted_assembler_attempts",
    "mislabelled_kv_payload_attempts",
    "kv_sized_allowlisted_attempts",
    "replay_cases",
    "replay_attempts",
    "reregistration_replay_cases",
    "new_identity_replay_cases",
    "new_identity_replay_attempts",
    "replay_control_attempts",
    "boundary_tick_replay_attempts",
    "relabeled_envelope_replay_cases",
    "expired_deadline_frame_attempts",
    "late_chunk_after_abandon_attempts",
    "post_abandonment_replay_cases",
    "overcapacity_frame_attempts",
    "abandoned_chunks_sent",
    "cross_connection_assembly_cases",
    "principal_probe_attempts",
    "principal_attempts.unadmitted_peer",
    "principal_attempts.unregistered_principal",
    "principal_attempts.captured_chunk_on_other_connection",
    "principal_attempts.foreign_key_on_bound_connection",
    "principal_attempts.stage_of_another_principal",
    "principal_positive_admitted",
    "rejection_cases",
    "abandoned_frame_cases",
    "tombstone_saturation_cases",
    "interleaved_frame_pairs",
    "interleaved_chunk_switches",
];

/// The controls: each must produce the effect it exists to show.
const CONTROLS: [&str; 4] = [
    "control_trips.unchunked_large_frame_refused_by_podwire",
    "control_trips.wide_scope_reaches_pod",
    "control_trips.claimed_principal_admits_captured_chunk",
    "control_trips.replay_admitted_without_tombstones",
];

/// Counters whose value must be 0 (a violation anywhere is a product finding).
const ZERO: [&str; 27] = [
    "bitwise_mismatch_count",
    "non_allowlisted_admitted",
    "replays_admitted",
    "boundary_tick_replays_admitted",
    "relabeled_chunks_admitted",
    "abandoned_frame_replays_admitted",
    "unauthenticated_principal_admitted",
    "unsigned_chunks_admitted",
    "verdict_mismatches",
    "attempts_without_typed_rejection",
    "valid_frames_not_delivered",
    "split_output_mismatch_count",
    "duplicate_activation_deliveries",
    "cross_operation_splices_admitted",
    "mislabelled_frame_digests_admitted",
    "expired_frames_admitted",
    "unbound_stage_frames_admitted",
    "chain_frame_mismatches_delivered",
    "late_chunks_after_abandon_admitted",
    "tombstone_evictions_before_retention",
    "tombstone_bound_violations",
    "global_cap_breaches",
    "cross_frame_mixing_count",
    "buffered_bytes_after_release",
    "abandon_release_mismatches",
    "mailbox_payload_mismatches",
    "ack_digest_mismatches",
];

/// The counters of a run, by name.
#[derive(Default)]
pub struct Counters(BTreeMap<String, i64>);

impl Counters {
    pub fn add(&mut self, name: &str, amount: i64) {
        *self.0.entry(name.to_string()).or_insert(0) += amount;
    }

    pub fn get(&self, name: &str) -> i64 {
        self.0.get(name).copied().unwrap_or(0)
    }

    fn group(&self, prefix: &str) -> Vec<(String, i64)> {
        let prefix = format!("{prefix}.");
        self.0
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|rest| (rest.to_string(), *value))
            })
            .collect()
    }
}

/// What a run produced.
pub struct Report {
    /// The one JSON object of the run.
    pub json: Json,
    /// The five manifest metrics, in manifest order, for a caller that embeds them.
    #[cfg_attr(not(test), allow(dead_code))]
    pub metrics: Vec<(String, i64)>,
    /// Coverage prerequisites that were not met (names).
    pub coverage_unmet: Vec<String>,
    /// Counters of a falsifier that fired (names).
    pub falsifiers_tripped: Vec<String>,
    /// Controls that did not trip (names).
    pub controls_dead: Vec<String>,
}

impl Report {
    /// AGENTS.md: 1 is a product finding, 2 a harness failure. 2 when a coverage prerequisite
    /// is unmet or a control did not trip (a vacuous or broken run, whatever else fired), else
    /// 1 when a falsifier fired, else 0.
    pub fn exit_status(&self) -> i32 {
        exit_code(
            &self.coverage_unmet,
            &self.controls_dead,
            &self.falsifiers_tripped,
        )
    }
}

/// The one classification of a run, shared by the process status and the printed record.
fn exit_code(coverage_unmet: &[String], controls_dead: &[String], falsifiers: &[String]) -> i32 {
    if !coverage_unmet.is_empty() || !controls_dead.is_empty() {
        2
    } else {
        i32::from(!falsifiers.is_empty())
    }
}

fn declared_seeds() -> Vec<u64> {
    EXPERIMENT_TOML
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

fn int(value: i64) -> Json {
    Json::Int(value)
}

fn object(pairs: Vec<(String, i64)>) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k, int(v))).collect())
}

/// Run every case for `seed` over `sequences` sequences of the E007 corpus.
pub fn execute(seed: u64, sequences: u64) -> Result<Report, String> {
    let started = Instant::now();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| format!("the runtime: {e}"))?;
    let rig = Rig::new()?;

    // The endpoints. Their ids are drawn at random by iroh and never printed.
    let peers = {
        let a = net::bind_peer(&rt, rig.signer.clone(), params::STAGE_IDENTITY)?;
        let lax = net::bind_peer(&rt, other_key(21), "stage-lax")?;
        let captor = net::bind_peer(&rt, other_key(11), params::DOWNSTREAM_STAGE_IDENTITY)?;
        let plain = net::bind_peer(&rt, other_key(31), "stage-plain")?;
        let stranger = net::bind_peer(&rt, other_key(41), "stage-stranger")?;
        let reborn = net::bind_peer(&rt, rig.signer.clone(), params::STAGE_IDENTITY)?;
        Peers {
            a,
            lax,
            captor,
            plain,
            stranger,
            reborn,
        }
    };
    let standing: [(&Peer, Standing); 6] = [
        (&peers.a, Standing::Strict),
        (&peers.lax, Standing::Lax),
        (&peers.captor, Standing::Strict),
        (&peers.plain, Standing::Strict),
        (&peers.stranger, Standing::Stranger),
        (&peers.reborn, Standing::Strict),
    ];
    let mut net = Net::new(rt, &standing)?;
    let mut counters = Counters::default();
    for name in ZERO {
        counters.add(name, 0);
    }
    {
        let mut h = H::new(&mut net, &peers, &rig, &mut counters, seed);
        cases::corpus(&mut h, sequences)?;
        cases::large(&mut h)?;
        cases::attacks(&mut h)?;
        cases::bounds(&mut h)?;
        cases::replays(&mut h)?;
        cases::abandonment(&mut h)?;
        cases::saturation(&mut h)?;
        cases::interleaving(&mut h)?;
        cases::allowlist(&mut h)?;
        cases::principals(&mut h)?;
    }
    net.close(&[
        &peers.a,
        &peers.lax,
        &peers.captor,
        &peers.plain,
        &peers.stranger,
        &peers.reborn,
    ]);
    let elapsed = started.elapsed();
    Ok(report(
        &counters,
        seed,
        sequences,
        elapsed.as_millis() as i64,
    ))
}

fn report(c: &Counters, seed: u64, sequences: u64, elapsed_ms: i64) -> Report {
    // The five metrics. The replay metric counts every chunk of an already completed or
    // abandoned frame that the receiver took (a replay, a relabelled copy of a completed frame,
    // a replay at the boundary tick, a replay of an abandoned frame); the principal metric counts
    // every chunk taken from a connection that is not the registered runtime's, and every chunk
    // whose signature is not the registered runtime's.
    let metrics = vec![
        (METRICS[0].to_string(), c.get("frames_delivered")),
        (METRICS[1].to_string(), c.get("bitwise_mismatch_count")),
        (METRICS[2].to_string(), c.get("non_allowlisted_admitted")),
        (
            METRICS[3].to_string(),
            c.get("replays_admitted")
                + c.get("boundary_tick_replays_admitted")
                + c.get("relabeled_chunks_admitted")
                + c.get("abandoned_frame_replays_admitted"),
        ),
        (
            METRICS[4].to_string(),
            c.get("unauthenticated_principal_admitted") + c.get("unsigned_chunks_admitted"),
        ),
    ];
    let metric_zero = |name: &str| {
        metrics
            .iter()
            .find(|(metric, _)| metric == name)
            .is_some_and(|(_, value)| *value == 0)
    };

    let mut coverage_unmet = Vec::new();
    let mut coverage = Vec::new();
    for name in COVERAGE {
        let value = c.get(name);
        coverage.push((name.to_string(), value));
        if value < 1 {
            coverage_unmet.push(name.to_string());
        }
    }
    let mut kinds = Vec::new();
    for kind in REJECTION_KINDS {
        let value = c.get(&format!("rejection_kinds.{kind}"));
        kinds.push((kind.to_string(), value));
        if value < 1 {
            coverage_unmet.push(format!("rejection_kinds.{kind}"));
        }
    }
    if c.get("large_frame_chunks_sent") != params::LARGE_FRAME_CHUNKS as i64 {
        coverage_unmet.push("large_frame_chunks_sent == 5".to_string());
    }
    let mut controls = Vec::new();
    let mut controls_dead = Vec::new();
    for name in CONTROLS {
        let value = c.get(name);
        controls.push((name.to_string(), value));
        if value < 1 {
            controls_dead.push(name.to_string());
        }
    }
    let mut falsifiers_tripped = Vec::new();
    for name in &METRICS[1..] {
        if !metric_zero(name) {
            falsifiers_tripped.push((*name).to_string());
        }
    }
    let mut falsifier_counters = Vec::new();
    for name in ZERO {
        let value = c.get(name);
        falsifier_counters.push((name.to_string(), value));
        if value > 0 {
            falsifiers_tripped.push(name.to_string());
        }
    }
    falsifiers_tripped.sort();
    falsifiers_tripped.dedup();

    let mut fields: Vec<(String, Json)> = vec![
        ("experiment".into(), Json::str("E008")),
        ("part".into(), Json::str("B")),
        ("benchmark".into(), Json::str(BENCHMARK)),
        ("seed".into(), int(seed as i64)),
        (
            "seed_is_declared".into(),
            Json::Bool(declared_seeds().contains(&seed)),
        ),
        ("sequences".into(), int(sequences as i64)),
        (
            "transport".into(),
            Json::str(
                "loopback iroh (ptr-net, podwire-backend): a PodHost and PodClient endpoints on \
                 127.0.0.1, ALPN ptr-podwire/1, PodRequest v1, one chunk per request, one \
                 authenticated connection per request, no relay",
            ),
        ),
    ];
    for (name, value) in &metrics {
        fields.push((name.clone(), int(*value)));
    }
    fields.push((
        "metric_composition".into(),
        Json::obj([
            (
                "loopback_replays_admitted",
                Json::strs(&[
                    "replays_admitted (every replayed chunk of a completed frame taken, on a \
                     fresh connection, after a re-registration of the same principal, and under \
                     a new principal id with the same key and stage: a restarted runtime)",
                    "boundary_tick_replays_admitted",
                    "relabeled_chunks_admitted",
                    "abandoned_frame_replays_admitted",
                ]),
            ),
            (
                "loopback_unauthenticated_principal_admitted",
                Json::strs(&[
                    "unauthenticated_principal_admitted",
                    "unsigned_chunks_admitted",
                ]),
            ),
        ]),
    ));
    fields.push(("coverage".into(), object(coverage)));
    fields.push(("coverage_unmet".into(), Json::strs(&coverage_unmet)));
    fields.push(("rejection_kinds".into(), object(kinds)));
    fields.push(("controls".into(), object(controls)));
    fields.push(("controls_dead".into(), Json::strs(&controls_dead)));
    fields.push(("falsifier_counters".into(), object(falsifier_counters)));
    fields.push(("falsifiers_tripped".into(), Json::strs(&falsifiers_tripped)));
    fields.push((
        "principal_binding".into(),
        Json::obj([
            (
                "assembler_principal",
                Json::str(
                    "the peer id PodHost::serve_once reports for the connection (the iroh \
                     endpoint id authenticated by the handshake); never a payload field, never \
                     a constant of the harness",
                ),
            ),
            ("probes", object(c.group("principal_attempts"))),
            (
                "unauthenticated_principal_admitted",
                int(c.get("unauthenticated_principal_admitted")),
            ),
            (
                "bound_connection_admitted",
                int(c.get("principal_positive_admitted")),
            ),
            (
                "claimed_principal_control_trips",
                int(c.get("control_trips.claimed_principal_admits_captured_chunk")),
            ),
        ]),
    ));
    fields.push((
        "traffic".into(),
        Json::obj([
            ("requests", int(c.get("requests"))),
            ("chunks_sent", int(c.get("chunks_sent"))),
            ("valid_frames_sent", int(c.get("valid_frames_sent"))),
            ("codec_refusals", int(c.get("codec_refusals"))),
            ("large_frame_bytes", int(params::LARGE_FRAME_BYTES as i64)),
            ("max_chunk_bytes", int(params::MAX_CHUNK_BYTES as i64)),
            ("large_frame_chunks", int(c.get("large_frame_chunks_sent"))),
        ]),
    ));
    // Transport timing: reported, never a verdict.
    fields.push((
        "timing".into(),
        Json::obj([("elapsed_ms", int(elapsed_ms))]),
    ));
    let exit = exit_code(&coverage_unmet, &controls_dead, &falsifiers_tripped);
    fields.push(("exit_status".into(), int(i64::from(exit))));
    Report {
        json: Json::Obj(fields),
        metrics,
        coverage_unmet,
        falsifiers_tripped,
        controls_dead,
    }
}

/// `split-pod-loopback <seed>` | `split-pod-loopback <sequences> <seed>`.
pub fn run(args: &[String]) {
    let parsed: Result<(u64, u64), String> = match args {
        [seed] => seed
            .parse::<u64>()
            .map(|seed| (params::SEQUENCES, seed))
            .map_err(|_| format!("seed must be a non-negative integer, got {seed:?}")),
        [sequences, seed] => sequences
            .parse::<u64>()
            .ok()
            .filter(|count| *count >= 1)
            .zip(seed.parse::<u64>().ok())
            .ok_or_else(|| format!("usage: {BENCHMARK} <sequences >= 1> <seed>, got {args:?}")),
        _ => Err(format!("usage: {BENCHMARK} <seed> | <sequences> <seed>")),
    };
    let (sequences, seed) = match parsed {
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
    match execute(seed, sequences) {
        Ok(report) => {
            println!("{}", report.json.render());
            std::process::exit(report.exit_status());
        }
        Err(reason) => {
            eprintln!("{BENCHMARK}: {reason}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reduced_run_over_the_loopback_wire_is_clean_and_deterministic() {
        let first = execute(17, 1).expect("a reduced run");
        assert_eq!(first.falsifiers_tripped, Vec::<String>::new());
        assert_eq!(first.coverage_unmet, Vec::<String>::new());
        assert_eq!(first.controls_dead, Vec::<String>::new());
        assert_eq!(first.exit_status(), 0);
        let [delivered, mismatches, non_allowlisted, replays, unauthenticated] = &first.metrics[..]
        else {
            panic!("five metrics");
        };
        assert!(delivered.1 >= 1);
        assert_eq!(
            (
                mismatches.1,
                non_allowlisted.1,
                replays.1,
                unauthenticated.1
            ),
            (0, 0, 0, 0)
        );
        // Everything but the wall time repeats.
        let second = execute(17, 1).expect("a second reduced run");
        let strip = |report: &Report| {
            let rendered = report.json.render();
            let (before, after) = rendered.split_once("\"timing\"").expect("a timing member");
            let (_, rest) = after.split_once('}').expect("a closed object");
            format!("{before}{rest}")
        };
        assert_eq!(strip(&first), strip(&second));
    }

    #[test]
    fn a_harness_failure_exits_2_and_only_a_falsifier_alone_exits_1() {
        let some = vec!["x".to_string()];
        let none: Vec<String> = Vec::new();
        assert_eq!(exit_code(&none, &none, &none), 0);
        assert_eq!(exit_code(&none, &none, &some), 1);
        assert_eq!(exit_code(&some, &none, &none), 2);
        assert_eq!(exit_code(&none, &some, &none), 2);
        assert_eq!(exit_code(&some, &some, &some), 2);
    }

    #[test]
    fn the_declared_seeds_are_the_manifests() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }
}
