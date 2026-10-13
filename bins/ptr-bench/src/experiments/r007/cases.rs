//! The cases of R007, run once per seed against three reference backends (the pinned tiny
//! model, raised prompt limits).
//!
//! The mechanism under test is lineage revocation plus parent-chained chunk keys: the span is
//! a revocable `Derived` prompt segment, forgetting it is revoking its lineage identity and
//! running the prompt without it, and the catalog has no mid-sequence eviction primitive.
//!
//! Per seed, the prefill-scale runs (a 4224-token cold prefill takes about a minute in
//! release) are exactly these:
//!
//! * backend B (baseline): a Cold probe of the prompt without the span, then a Warm probe of
//!   the same prompt to populate B's catalog (B warm must equal B cold bitwise: C1);
//! * backend F (forget): a Warm probe of the prompt with the span (C2), the revocation of the
//!   span's lineage identity and a Warm probe of the prompt without it, compared with B's cold
//!   run (C3, C4);
//! * the 32 canary probes, each the prompt without the span plus the canary's 16 bytes, run
//!   Warm on B and on F, each reusing the aligned prefix and recomputing the tail, plus two cold
//!   anchors (canary 0 in the span, canary 16 never in it) run Cold on B and compared with both
//!   warm probes of the same request (C5);
//! * backend G (negative control): the same Warm probe with the span under
//!   `IndexUpdate::Suppressed`, a revocation that does not reach the index, and a StaleDonor
//!   probe whose result must differ from B's cold run and trip the falsifiers (C6).
//!
//! Every count is read from the runtime's outcome or computed from the requests actually
//! built; none is asserted from a constant. A refusal that is not the expected one, an insert
//! refusal on the honest path or a control that does not hold is an `Err` (exit status 2).

use super::metrics;
use super::params::{self, Shape};
use crate::experiments::layered_kv::corpus::{self, hex, order, sequence, Part, CHUNK_TOKENS};
use crate::experiments::layered_kv::emit::{judge, Json, Status, Tally};
use crate::experiments::layered_kv::lab::{self, fail, Res};
use crate::experiments::layered_kv::pins;
use ptr_ledger::integrity::sha256;
use ptr_model_api::{SegmentKind, SegmentedModelRequest, Tokenizer};
use ptr_runtime::{
    CatalogLimits, IndexUpdate, ReferenceKvFaults, ReferenceKvLimits, ReferenceKvProbe,
    ReferenceKvProbeMode, ReferenceKvReuseBackend,
};
use ptr_types::Generation;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// The ulp distance reported for a pair of which one value is NaN and the bit patterns differ
/// (larger than any finite distance, which is below 2^32).
pub const NAN_DISTANCE: i64 = 1 << 32;

/// The canary string `index`: the first 16 hex characters of sha256 over the generator seed,
/// a colon and the index.
pub fn canary(index: usize) -> String {
    let digest = sha256(format!("{}:{index}", params::CANARY_GENERATOR_SEED).as_bytes());
    hex(&digest)[..params::CANARY_BYTES].to_string()
}

/// The identity-set digest: sha256 over the canaries joined by newlines, bare hex.
pub fn canary_set_digest(canaries: &[String]) -> String {
    hex(&sha256(canaries.join("\n").as_bytes()))
}

/// The pinned corpus: sequence `i` of the corpus generator, in index order.
fn corpus_sequences() -> Vec<Vec<u32>> {
    (0..params::SELECTION_SEQUENCES)
        .map(|index| {
            sequence(
                params::CORPUS_GENERATOR_SEED,
                index as u64,
                params::SELECTION_SEQUENCE_TOKENS,
            )
        })
        .collect()
}

/// The digest of the corpus selection (what `selection_digest_mismatches` compares).
pub fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &corpus_sequences())
}

/// The distance between two `f32` values in representable steps (ulps): 0 iff the values are
/// equal (`+0.0` and `-0.0` are equal values; their bit patterns differ and are counted by the
/// bitwise comparison). A NaN pair is 0 only for identical bits, else [`NAN_DISTANCE`].
pub fn ulp_distance(a: f32, b: f32) -> i64 {
    fn key(value: f32) -> i64 {
        let bits = value.to_bits();
        let magnitude = i64::from(bits & 0x7fff_ffff);
        if bits >> 31 == 1 {
            -magnitude
        } else {
            magnitude
        }
    }
    if a.is_nan() || b.is_nan() {
        return if a.to_bits() == b.to_bits() {
            0
        } else {
            NAN_DISTANCE
        };
    }
    (key(a) - key(b)).abs()
}

/// How two logits vectors differ: the entries whose bit patterns differ, the largest ulp
/// distance, and the largest absolute difference as `f64` (NaN pairs excluded; they count in
/// `nan_pairs`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Diff {
    pub entries: i64,
    pub max_ulps: i64,
    pub max_abs: f64,
    pub nan_pairs: i64,
}

pub fn diff(a: &[f32], b: &[f32]) -> Diff {
    let mut out = Diff::default();
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            out.entries += 1;
        }
        out.max_ulps = out.max_ulps.max(ulp_distance(*x, *y));
        if x.is_nan() || y.is_nan() {
            out.nan_pairs += i64::from(x.to_bits() != y.to_bits());
        } else {
            out.max_abs = out.max_abs.max((f64::from(*x) - f64::from(*y)).abs());
        }
    }
    out
}

/// The segments of one prompt, from the seeded corpus and the canaries.
pub struct Prompt {
    /// The corpus sequences that fill the roles system, head, tail, question.
    pub roles: [usize; 4],
    system: Vec<u32>,
    head: Vec<u32>,
    tail: Vec<u32>,
    question: Vec<u32>,
    /// The span: the bytes of canaries 0 to 15, one token per byte.
    span: Vec<u32>,
}

impl Prompt {
    pub fn new(seed: u64, shape: Shape) -> Self {
        let corpus = corpus_sequences();
        let permutation = order(seed, params::ROLE_ORDER_LABEL, params::SELECTION_SEQUENCES);
        let roles = [
            permutation[0],
            permutation[1],
            permutation[2],
            permutation[3],
        ];
        let take = |role: usize, tokens: usize| corpus[roles[role]][..tokens].to_vec();
        let span = (0..params::CANARIES_IN_SPAN)
            .flat_map(|index| canary(index).into_bytes())
            .map(u32::from)
            .collect();
        Self {
            roles,
            system: take(0, shape.system),
            head: take(1, shape.head),
            tail: take(2, shape.tail),
            question: take(3, shape.question),
            span,
        }
    }

    fn parts(&self, with_span: bool) -> Vec<Part> {
        let mut parts = vec![Part::raw(self.system.clone()), Part::raw(self.head.clone())];
        if with_span {
            parts.push(Part::bound(
                SegmentKind::Derived,
                params::SPAN_IDENTITY,
                params::SPAN_GENERATION,
                self.span.clone(),
            ));
        }
        parts.push(Part::raw(self.tail.clone()));
        parts.push(Part::raw(self.question.clone()));
        parts
    }

    /// The prompt with the span.
    pub fn with_span(&self) -> SegmentedModelRequest {
        corpus::request("r007-with-span", &self.parts(true))
    }

    /// The prompt without the span: the baseline's prompt.
    pub fn without_span(&self) -> SegmentedModelRequest {
        corpus::request("r007-without-span", &self.parts(false))
    }

    /// The prompt without the span plus the bytes of `canary` as a final segment.
    pub fn probe(&self, index: usize, canary: &str) -> SegmentedModelRequest {
        let mut parts = self.parts(false);
        parts.push(Part::raw(canary.bytes().map(u32::from).collect()));
        corpus::request(&format!("r007-probe-{index}"), &parts)
    }
}

/// Where the span lies in a request, read from the request's own segments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    pub before: usize,
    pub span: usize,
    pub after: usize,
}

/// The layout of a request with exactly one `Derived` segment.
pub fn layout(request: &SegmentedModelRequest) -> Res<Layout> {
    let mut found: Option<Layout> = None;
    let mut before = 0;
    for segment in request.segments() {
        let tokens = segment.tokens().len();
        if segment.source().kind() == SegmentKind::Derived {
            if found.is_some() {
                return Err("the request holds more than one span segment".into());
            }
            found = Some(Layout {
                before,
                span: tokens,
                after: 0,
            });
        } else if let Some(span) = found.as_mut() {
            span.after += tokens;
        } else {
            before += tokens;
        }
    }
    found.ok_or_else(|| "the request holds no span segment".into())
}

/// The limits of the harness: the prompt limit raised to the longest probe request (the
/// prompt with the span, plus one chunk) and the catalog bounds raised so that no insert is
/// refused. The crate's default (256 tokens) is untouched.
pub fn limits(shape: Shape) -> ReferenceKvLimits {
    ReferenceKvLimits {
        max_prompt_tokens: shape.with_span() + params::PROBE_HEADROOM_TOKENS,
        catalog: CatalogLimits {
            max_trie_nodes: 1 << 16,
            max_kv_page_bytes: 1 << 32,
            max_metadata_bytes: 1 << 30,
        },
        ..ReferenceKvLimits::default()
    }
}

pub fn backend(shape: Shape) -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::new(limits(shape), lab::scope()).map_err(fail("a reference backend"))
}

/// The digest of the tiny-model configuration the backend actually runs, by the rule of
/// `tiny_model_config_digest_rule`.
pub fn observed_config_digest(backend: &ReferenceKvReuseBackend) -> String {
    let config = backend.stack_config();
    pins::canonical_digest(&[
        ("layers", config.num_layers as u64),
        ("hidden_size", config.layer.hidden_size as u64),
        ("num_heads", config.layer.num_attention_heads as u64),
        ("num_kv_heads", config.layer.num_key_value_heads as u64),
        (
            "head_dim",
            (config.layer.hidden_size / config.layer.num_attention_heads) as u64,
        ),
        ("vocab_size", config.vocab_size as u64),
        ("weight_seed", ptr_runtime::REFERENCE_KV_WEIGHT_SEED),
    ])
}

fn span_revocation() -> BTreeSet<(String, Generation)> {
    [(
        params::SPAN_IDENTITY.to_string(),
        Generation(params::SPAN_GENERATION),
    )]
    .into_iter()
    .collect()
}

/// Wall time of each runtime call, for the `timing_ms` field only (never a verdict field).
#[derive(Default)]
struct Timings(Vec<(String, f64)>);

impl Timings {
    fn json(&self) -> Json {
        Json::Arr(
            self.0
                .iter()
                .map(|(name, ms)| {
                    Json::obj([("run", Json::str(name.clone())), ("ms", Json::Float(*ms))])
                })
                .collect(),
        )
    }
}

fn run(
    timings: &mut Timings,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    mode: ReferenceKvProbeMode<'_>,
) -> Res<ReferenceKvProbe> {
    let started = Instant::now();
    let probe = backend
        .probe(&lab::scope(), request, mode)
        .map_err(fail(name))?;
    timings
        .0
        .push((name.to_string(), started.elapsed().as_secs_f64() * 1000.0));
    if probe.prompt_logits.len() != params::CANARY_PROBE_VECTOR_ENTRIES as usize {
        return Err(format!(
            "{name}: {} logits entries, not {}",
            probe.prompt_logits.len(),
            params::CANARY_PROBE_VECTOR_ENTRIES
        ));
    }
    if probe.outcome.restarted_after_cancel {
        return Err(format!("{name}: the run restarted after a cancellation"));
    }
    Ok(probe)
}

/// An honest warm run must have had every insert accepted: a refusal would make the catalog
/// state of the comparison unknown.
fn accepted(name: &str, probe: &ReferenceKvProbe) -> Res<()> {
    if probe.outcome.insert_refusals != 0 {
        return Err(format!(
            "{name}: {} insert refusals: {:?}",
            probe.outcome.insert_refusals, probe.outcome.insert_refusal_reasons
        ));
    }
    Ok(())
}

fn digest_text(digest: &[u8; 32]) -> String {
    format!("sha256:{}", hex(digest))
}

fn outcome_json(probe: &ReferenceKvProbe) -> Json {
    let outcome = &probe.outcome;
    Json::obj([
        ("prompt_tokens", Json::Int(outcome.prompt_tokens as i64)),
        ("hit_tokens", Json::Int(outcome.hit_tokens as i64)),
        (
            "prefilled_tokens",
            Json::Int(outcome.prefilled_tokens as i64),
        ),
        ("inserted_chunks", Json::Int(outcome.inserted_chunks as i64)),
        ("insert_refusals", Json::Int(outcome.insert_refusals as i64)),
        (
            "generated_tokens",
            Json::Int(outcome.generated.len() as i64),
        ),
        (
            "kv_state_digest",
            Json::str(digest_text(&probe.kv_state_digest)),
        ),
        (
            "logits_digest",
            Json::str(digest_text(&outcome.logits_digest)),
        ),
    ])
}

fn diff_json(diff: Diff) -> Json {
    Json::obj([
        ("entries_differing", Json::Int(diff.entries)),
        ("max_ulps", Json::Int(diff.max_ulps)),
        ("max_abs_f32", Json::Float(diff.max_abs)),
        ("nan_pairs_differing", Json::Int(diff.nan_pairs)),
    ])
}

/// Two probes are bitwise the same state: logits bit patterns, KV-state digest, decode digest.
fn same_state(a: &ReferenceKvProbe, b: &ReferenceKvProbe) -> bool {
    a.logit_bits() == b.logit_bits()
        && a.kv_state_digest == b.kv_state_digest
        && a.outcome.logits_digest == b.outcome.logits_digest
        && a.outcome.generated == b.outcome.generated
}

/// Non-finite entries of a probe's logits vector.
fn non_finite(probe: &ReferenceKvProbe) -> i64 {
    probe
        .prompt_logits
        .iter()
        .filter(|value| !value.is_finite())
        .count() as i64
}

/// One comparison of two probes, counted the way `bitwise_mismatch_count` counts: one per
/// differing KV-state digest, one per differing logits entry (bit patterns), one for a
/// differing decode digest or generated tokens, and one per non-finite logits entry on either
/// side (a non-finite vector is never a silent pass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Compare {
    pub digest: i64,
    pub decode: i64,
    pub non_finite: i64,
    pub logits: Diff,
}

impl Compare {
    pub fn total(&self) -> i64 {
        self.digest + self.decode + self.non_finite + self.logits.entries
    }
}

pub fn compare(a: &ReferenceKvProbe, b: &ReferenceKvProbe) -> Compare {
    Compare {
        digest: i64::from(a.kv_state_digest != b.kv_state_digest),
        decode: i64::from(
            a.outcome.logits_digest != b.outcome.logits_digest
                || a.outcome.generated != b.outcome.generated,
        ),
        non_finite: non_finite(a) + non_finite(b),
        logits: diff(&a.prompt_logits, &b.prompt_logits),
    }
}

/// What a forget-side run reused against what it should have: rows served beyond the expected
/// hit (over-reuse), tokens prefilled beyond what the expected hit leaves (under-reuse), and
/// the disagreement of the outcome's own accounting (prompt tokens against the request, and
/// prompt minus hit against prefilled).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reuse {
    pub over_rows: usize,
    pub extra_prefilled: usize,
    pub accounting: usize,
}

pub fn reuse_defects(
    probe: &ReferenceKvProbe,
    request_tokens: usize,
    expected_hit: usize,
) -> Reuse {
    let outcome = &probe.outcome;
    let over = outcome.hit_tokens.saturating_sub(expected_hit);
    let expected_prefilled = request_tokens.saturating_sub(expected_hit);
    Reuse {
        over_rows: over.div_ceil(CHUNK_TOKENS),
        extra_prefilled: outcome.prefilled_tokens.saturating_sub(expected_prefilled),
        accounting: outcome.prompt_tokens.abs_diff(request_tokens)
            + outcome
                .prompt_tokens
                .saturating_sub(outcome.hit_tokens)
                .abs_diff(outcome.prefilled_tokens),
    }
}

/// What the cases produced besides the metric counters (which go into the tally).
pub struct Findings {
    pub cases: Vec<(String, Json)>,
    pub timings: Json,
    pub canary_identity: Json,
    pub procedure: Json,
    pub roles: Json,
    pub expected: Json,
}

/// A procedure parameter checked against its pinned value: the observed value is recorded and
/// a difference counts as one mismatch.
#[derive(Default)]
struct Procedure {
    mismatches: i64,
    observed: Vec<(String, Json)>,
    expected: Vec<(String, Json)>,
    differing: Vec<String>,
    constants: Vec<String>,
}

impl Procedure {
    fn check(&mut self, name: &str, observed: Json, expected: Json) {
        if observed != expected {
            self.mismatches += 1;
            self.differing.push(name.to_string());
        }
        self.observed.push((name.to_string(), observed));
        self.expected.push((name.to_string(), expected));
    }

    /// A parameter that is a constant of this harness on both sides: recorded, never counted.
    fn constant(&mut self, name: &str, value: Json) {
        self.observed.push((name.to_string(), value.clone()));
        self.expected.push((name.to_string(), value));
        self.constants.push(name.to_string());
    }

    fn int(&mut self, name: &str, observed: u64, expected: u64) {
        self.check(name, Json::Int(observed as i64), Json::Int(expected as i64));
    }

    fn text(&mut self, name: &str, observed: &str, expected: &str) {
        self.check(name, Json::str(observed), Json::str(expected));
    }
}

fn token_text(tokens: &[ptr_model_api::TokenId]) -> String {
    tokens
        .iter()
        .map(|token| char::from(u8::try_from(token.0).unwrap_or(b'?')))
        .collect()
}

/// Run every case once for `seed` and set the metric counters in `tally`.
pub fn run_all(tally: &mut Tally, seed: u64, shape: Shape) -> Res<Findings> {
    let chunk = CHUNK_TOKENS;
    let prompt = Prompt::new(seed, shape);
    let with_span = prompt.with_span();
    let without = prompt.without_span();
    let span_layout = layout(&with_span)?;
    let mut timings = Timings::default();
    let mut cases: Vec<(String, Json)> = Vec::new();

    // C1: the baseline, a fresh prefill of the prompt without the span, and the same prompt
    // warm to populate the baseline's catalog.
    let baseline = backend(shape)?;
    let base_cold = run(
        &mut timings,
        "C1.baseline_cold",
        &baseline,
        &without,
        ReferenceKvProbeMode::Cold,
    )?;
    let base_warm = run(
        &mut timings,
        "C1.baseline_warm",
        &baseline,
        &without,
        ReferenceKvProbeMode::Warm,
    )?;
    accepted("C1.baseline_warm", &base_warm)?;
    if !same_state(&base_warm, &base_cold) {
        return Err("C1: the baseline's warm run differs from its cold run".into());
    }
    cases.push((
        "C1".into(),
        Json::obj([
            ("what", Json::str("fresh prefill of the prompt without the span (Cold), then the same prompt Warm on an empty catalog")),
            ("cold", outcome_json(&base_cold)),
            ("warm", outcome_json(&base_warm)),
            ("warm_equals_cold", Json::Bool(true)),
        ]),
    ));

    // C2: the prompt with the span, warm, populating the forget backend's catalog.
    let forget_backend = backend(shape)?;
    let warm = run(
        &mut timings,
        "C2.warm_with_span",
        &forget_backend,
        &with_span,
        ReferenceKvProbeMode::Warm,
    )?;
    accepted("C2.warm_with_span", &warm)?;
    // A hit on an empty catalog is reuse that should not exist: it counts as stale reuse and
    // falsifies, it is not a harness error.
    let mut stale_extra = warm.outcome.hit_tokens.div_ceil(chunk);
    let mut recompute_extra = 0usize;
    let mut accounting_defect = reuse_defects(&warm, with_span.total_tokens(), 0).accounting;
    if warm.kv_state_digest == base_cold.kv_state_digest
        || warm.logit_bits() == base_cold.logit_bits()
    {
        return Err("C2: the span does not change the state, so the comparison is vacuous".into());
    }
    let prefix_rows = span_layout.before / chunk;
    let span_end_rows = (span_layout.before + span_layout.span) / chunk;
    let span_rows = span_layout.span / chunk;
    let inserted = warm.outcome.inserted_chunks;
    let rows_citing_span = inserted.saturating_sub(prefix_rows);
    let downstream_rows = inserted.saturating_sub(span_end_rows);
    let span_insertions = i64::from(rows_citing_span >= span_rows && span_rows > 0);
    cases.push((
        "C2".into(),
        Json::obj([
            (
                "what",
                Json::str(
                    "the prompt with the span, Warm, on an empty catalog of the forget backend",
                ),
            ),
            ("run", outcome_json(&warm)),
            ("span_tokens", Json::Int(span_layout.span as i64)),
            ("tokens_before_span", Json::Int(span_layout.before as i64)),
            ("tokens_after_span", Json::Int(span_layout.after as i64)),
            ("rows_inserted", Json::Int(inserted as i64)),
            ("rows_citing_span", Json::Int(rows_citing_span as i64)),
            ("rows_after_span", Json::Int(downstream_rows as i64)),
            ("state_differs_from_cold_without_span", Json::Bool(true)),
        ]),
    ));

    // C3: revoke the span's lineage identity, then run the prompt without it.
    let retired = forget_backend
        .invalidate_inputs(&span_revocation())
        .map_err(fail("the span revocation"))?;
    let forget = run(
        &mut timings,
        "C3.forget_rerun",
        &forget_backend,
        &without,
        ReferenceKvProbeMode::Warm,
    )?;
    accepted("C3.forget_rerun", &forget)?;
    let evicted = retired > 0 && retired == rows_citing_span;
    let recomputed = forget.outcome.prefilled_tokens > 0 && forget.outcome.inserted_chunks > 0;
    let stale_rows = forget.outcome.hit_tokens.saturating_sub(span_layout.before) / chunk;
    let tokens_after_span = span_layout.after;
    // The runtime's own accounting of the forget run must agree with the request built.
    let forget_reuse = reuse_defects(&forget, without.total_tokens(), span_layout.before);
    recompute_extra += forget_reuse.extra_prefilled;
    accounting_defect += forget_reuse.accounting;
    cases.push((
        "C3".into(),
        Json::obj([
            ("what", Json::str("revoke the span's lineage identity (every catalog entry citing it is retired), then the prompt without the span, Warm, on the same catalog")),
            ("entries_retired", Json::Int(retired as i64)),
            ("entries_citing_span", Json::Int(rows_citing_span as i64)),
            ("run", outcome_json(&forget)),
            (
                "rows_served_beyond_the_prefix",
                Json::Int(stale_rows as i64),
            ),
        ]),
    ));

    // C4: the forget run against the fresh prefill without the span.
    let forget_diff = diff(&forget.prompt_logits, &base_cold.prompt_logits);
    let forget_digest_differs = forget.kv_state_digest != base_cold.kv_state_digest;
    let forget_cmp = compare(&forget, &base_cold);
    let mut bitwise = forget_cmp.total();
    let mut decode_mismatch = forget_cmp.decode;
    let mut non_finite_total = forget_cmp.non_finite;
    let mut finite_comparisons = i64::from(forget_cmp.non_finite == 0);
    cases.push((
        "C4".into(),
        Json::obj([
            ("what", Json::str("the forget run against B's cold run: KV-state digest and the 256 logits bit patterns")),
            ("kv_state_digest_differs", Json::Bool(forget_digest_differs)),
            ("logits", diff_json(forget_diff)),
            ("decode_digest_or_tokens_differ", Json::Bool(forget_cmp.decode == 1)),
            ("non_finite_entries", Json::Int(forget_cmp.non_finite)),
        ]),
    ));

    // C5: the 32 canary probes, after the revocation and the recomputation.
    let mut probe_rows = Vec::new();
    let mut probed: BTreeMap<usize, String> = BTreeMap::new();
    let mut probes_done = 0i64;
    let mut post_forget = 0i64;
    let mut shift = 0i64;
    let mut max_abs = 0.0f64;
    let mut probe_entries = 0i64;
    let mut probe_digests = 0i64;
    let mut min_hit = usize::MAX;
    // Canary 0 (in the span) and canary 16 (never in the span) are also run Cold.
    let anchors = [0usize, params::CANARIES_IN_SPAN];
    let mut anchor_inputs = Vec::new();
    let mut anchor_mismatches = 0i64;
    let mut prefilled_total = 0usize;
    let order_of_probes = order(
        seed,
        params::PROBE_ORDER_LABEL,
        params::CANARY_PROBES as usize,
    );
    for index in &order_of_probes {
        let text = canary(*index);
        let request = prompt.probe(*index, &text);
        let suffix = request
            .segments()
            .last()
            .map(|segment| token_text(segment.tokens()))
            .ok_or("a probe request without segments")?;
        let expected_hit = (request.total_tokens() - 1) / chunk * chunk;
        let on_baseline = run(
            &mut timings,
            &format!("C5.probe_{index}.baseline"),
            &baseline,
            &request,
            ReferenceKvProbeMode::Warm,
        )?;
        let on_forget = run(
            &mut timings,
            &format!("C5.probe_{index}.forget"),
            &forget_backend,
            &request,
            ReferenceKvProbeMode::Warm,
        )?;
        accepted("C5 baseline probe", &on_baseline)?;
        accepted("C5 forget probe", &on_forget)?;
        // The baseline must reuse the aligned prefix (its validity); the forget side may not
        // deviate silently: over-reuse is stale reuse and under-reuse is recomputation beyond
        // the tail, both falsifiers.
        if on_baseline.outcome.hit_tokens != expected_hit {
            return Err(format!(
                "C5 probe {index} on the baseline backend reused {} tokens, not the aligned prefix {expected_hit}",
                on_baseline.outcome.hit_tokens
            ));
        }
        let reuse = reuse_defects(&on_forget, request.total_tokens(), expected_hit);
        stale_extra += reuse.over_rows;
        recompute_extra += reuse.extra_prefilled;
        accounting_defect += reuse.accounting;
        let cmp = compare(&on_forget, &on_baseline);
        non_finite_total += cmp.non_finite;
        finite_comparisons += i64::from(cmp.non_finite == 0);
        // The probe counts when both runtime calls returned with finite logits; it is a
        // post-forget probe when the revocation retired exactly the span's rows and the
        // forget run recomputed before it.
        if cmp.non_finite == 0 {
            probes_done += 1;
            if evicted && recomputed {
                post_forget += 1;
            }
        }
        let d = cmp.logits;
        let digest_differs = cmp.digest == 1;
        bitwise += cmp.total();
        decode_mismatch += cmp.decode;
        shift = shift.max(d.max_ulps);
        max_abs = max_abs.max(d.max_abs);
        probe_entries += d.entries;
        probe_digests += cmp.digest;
        min_hit = min_hit.min(on_forget.outcome.hit_tokens);
        prefilled_total += on_forget.outcome.prefilled_tokens;
        if anchors.contains(index) {
            anchor_inputs.push((
                *index,
                request.clone(),
                on_baseline.clone(),
                on_forget.clone(),
            ));
        }
        probed.insert(*index, suffix);
        probe_rows.push(Json::obj([
            ("canary_index", Json::Int(*index as i64)),
            ("in_span", Json::Bool(*index < params::CANARIES_IN_SPAN)),
            ("hit_tokens", Json::Int(on_forget.outcome.hit_tokens as i64)),
            (
                "prefilled_tokens",
                Json::Int(on_forget.outcome.prefilled_tokens as i64),
            ),
            ("kv_state_digest_differs", Json::Bool(digest_differs)),
            ("logits", diff_json(d)),
        ]));
    }
    // The cold anchors: a fresh prefill (no lookup, no insert) of the probe request of canary 0
    // (in the span) and of canary 16 (never in the span) on the baseline backend, compared
    // with the warm probes of the same request on both backends.
    let mut anchor_rows = Vec::new();
    for (index, request, on_baseline, on_forget) in &anchor_inputs {
        let cold = run(
            &mut timings,
            &format!("C5.cold_anchor_{index}"),
            &baseline,
            request,
            ReferenceKvProbeMode::Cold,
        )?;
        let mut row = vec![
            ("canary_index".to_string(), Json::Int(*index as i64)),
            (
                "in_span".to_string(),
                Json::Bool(*index < params::CANARIES_IN_SPAN),
            ),
        ];
        for (side, warm_probe) in [("baseline_warm", on_baseline), ("forget_warm", on_forget)] {
            let cmp = compare(warm_probe, &cold);
            bitwise += cmp.total();
            decode_mismatch += cmp.decode;
            non_finite_total += cmp.non_finite;
            finite_comparisons += i64::from(cmp.non_finite == 0);
            shift = shift.max(cmp.logits.max_ulps);
            max_abs = max_abs.max(cmp.logits.max_abs);
            anchor_mismatches += cmp.total();
            row.push((
                format!("{side}_against_cold"),
                Json::obj([
                    ("mismatches", Json::Int(cmp.total())),
                    ("kv_state_digest_differs", Json::Bool(cmp.digest == 1)),
                    ("logits", diff_json(cmp.logits)),
                ]),
            ));
        }
        anchor_rows.push(Json::Obj(row));
    }
    if anchor_rows.len() != anchors.len() {
        return Err("a cold anchor was not probed".into());
    }
    cases.push((
        "C5".into(),
        Json::obj([
            ("what", Json::str("32 probes, each the prompt without the span plus the canary's 16 bytes, Warm on B and on F after the forget, compared with each other (256 logits entries, KV-state digest, decode digest, finiteness); two of them (canary 0 in the span, canary 16 never in it) also run Cold and compared with both warm probes")),
            ("order", Json::ints(order_of_probes.iter().map(|index| *index as i64))),
            ("probes", Json::Int(probes_done)),
            ("entries_differing", Json::Int(probe_entries)),
            ("kv_state_digests_differing", Json::Int(probe_digests)),
            ("max_ulps", Json::Int(shift)),
            ("max_abs_f32", Json::Float(max_abs)),
            ("min_reused_tokens", Json::Int(min_hit as i64)),
            ("prefilled_tokens_total", Json::Int(prefilled_total as i64)),
            ("cold_anchors", Json::Arr(anchor_rows)),
            ("per_probe", Json::Arr(probe_rows)),
        ]),
    ));
    drop(baseline);

    // The canary identity: the canaries actually embedded in the span and actually probed.
    let embedded: Vec<String> = span_texts(&with_span)?;
    let mut canary_set = Vec::new();
    for index in 0..params::CANARY_PROBES as usize {
        canary_set.push(
            probed
                .get(&index)
                .cloned()
                .ok_or_else(|| format!("canary {index} was not probed"))?,
        );
    }
    for (index, text) in embedded.iter().enumerate() {
        if canary_set.get(index) != Some(text) {
            return Err(format!(
                "span canary {index} is not the probed canary {index}"
            ));
        }
    }
    let canary_span_tokens = embedded.iter().map(String::len).sum::<usize>();
    let identity_digest = canary_set_digest(&canary_set);
    let distinct = canary_set.iter().collect::<BTreeSet<_>>().len() == canary_set.len();

    // C6: the negative control on a third backend.
    let donor = backend(shape)?;
    donor
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .map_err(fail("the fault"))?;
    let donor_warm = run(
        &mut timings,
        "C6.donor_warm",
        &donor,
        &with_span,
        ReferenceKvProbeMode::Warm,
    )?;
    accepted("C6.donor_warm", &donor_warm)?;
    let donor_retired = donor
        .invalidate_inputs(&span_revocation())
        .map_err(fail("the suppressed revocation"))?;
    if donor_retired != 0 {
        return Err(format!(
            "C6: the suppressed revocation retired {donor_retired} entries"
        ));
    }
    let stale = run(
        &mut timings,
        "C6.stale_donor",
        &donor,
        &without,
        ReferenceKvProbeMode::StaleDonor(&with_span),
    )?;
    let stale_diff = diff(&stale.prompt_logits, &base_cold.prompt_logits);
    let stale_digest_differs = stale.kv_state_digest != base_cold.kv_state_digest;
    let control_stale_rows = stale.outcome.hit_tokens.saturating_sub(span_layout.before) / chunk;
    let control_bitwise = i64::from(stale_digest_differs) + stale_diff.entries;
    let control = Json::Obj(vec![
        ("bitwise_mismatch_count".into(), Json::Int(control_bitwise)),
        ("canary_logit_shift".into(), Json::Int(stale_diff.max_ulps)),
        (
            "stale_downstream_rows_reused".into(),
            Json::Int(control_stale_rows as i64),
        ),
        (
            "recompute_tokens".into(),
            Json::Int(stale.outcome.prefilled_tokens as i64),
        ),
        (
            "tokens_after_span".into(),
            Json::Int(tokens_after_span as i64),
        ),
    ]);
    let verdicts = judge(&control, &[], &metrics::falsifiers());
    let tripped = verdicts
        .iter()
        .filter(|check| check.status == Status::Fail)
        .count();
    if tripped != verdicts.len() || !stale_digest_differs || stale_diff.entries == 0 {
        return Err(format!(
            "C6: the negative control does not trip every falsifier ({tripped} of {}): the criteria are vacuous",
            verdicts.len()
        ));
    }
    cases.push((
        "C6".into(),
        Json::obj([
            ("what", Json::str("negative control on a third backend: set_faults(IndexUpdate::Suppressed), the same warm run with the span, a revocation that does not reach the index, then a StaleDonor probe of the prompt without the span against the donor's rows")),
            ("entries_retired_by_the_suppressed_revocation", Json::Int(donor_retired as i64)),
            ("run", outcome_json(&stale)),
            ("kv_state_digest_differs_from_cold", Json::Bool(stale_digest_differs)),
            ("logits_against_cold", diff_json(stale_diff)),
            ("counters", control),
            ("falsifiers_tripped", Json::Int(tripped as i64)),
            ("falsifiers_judged", Json::Int(verdicts.len() as i64)),
        ]),
    ));

    // The procedure parameters, as observed.
    let config = forget_backend.stack_config();
    let tokenizer = forget_backend.tokenizer().identity();
    let tokenizer_text = if tokenizer.name() == "ptr.byte" && tokenizer.version() == "1" {
        format!(
            "ByteTokenizer, vocabulary {} (no external revision)",
            config.vocab_size
        )
    } else {
        format!("{}@{}", tokenizer.name(), tokenizer.version())
    };
    let segment_tokens = |wanted: usize| -> u64 {
        with_span
            .segments()
            .get(wanted)
            .map_or(0, |segment| segment.tokens().len() as u64)
    };
    let segments = with_span.segments().len();
    let bearing = with_span
        .segments()
        .iter()
        .filter(|segment| segment.source().kind() == SegmentKind::Derived)
        .filter(|segment| {
            let text = token_text(segment.tokens());
            canary_set.iter().any(|canary| text.contains(canary))
        })
        .count();
    let mut procedure = Procedure::default();
    procedure.check(
        "seeds",
        Json::ints(params::declared_seeds().iter().map(|seed| *seed as i64)),
        Json::ints([17, 29, 43, 71, 101]),
    );
    procedure.int("span_tokens", span_layout.span as u64, shape.span as u64);
    procedure.int("canary_probes", probes_done as u64, params::CANARY_PROBES);
    procedure.int(
        "weight_seed",
        ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
        params::WEIGHT_SEED,
    );
    procedure.int(
        "hidden_size",
        config.layer.hidden_size as u64,
        params::HIDDEN_SIZE,
    );
    procedure.int(
        "num_heads",
        config.layer.num_attention_heads as u64,
        params::NUM_HEADS,
    );
    procedure.int(
        "num_kv_heads",
        config.layer.num_key_value_heads as u64,
        params::NUM_KV_HEADS,
    );
    procedure.int(
        "head_dim",
        (config.layer.hidden_size / config.layer.num_attention_heads) as u64,
        params::HEAD_DIM,
    );
    procedure.int("vocab_size", config.vocab_size as u64, params::VOCAB_SIZE);
    procedure.int(
        "corpus_generator_seed",
        crate::experiments::layered_kv::SEQUENCE_SEED,
        params::CORPUS_GENERATOR_SEED,
    );
    procedure.int(
        "canary_bearing_span_cases",
        bearing as u64,
        params::CANARY_BEARING_SPAN_CASES,
    );
    let middle = span_layout.before > prompt.system.len() && span_layout.after > 0;
    procedure.text(
        "span_position",
        if middle { "middle" } else { "not middle" },
        params::SPAN_POSITION,
    );
    procedure.int(
        "span_start_offset_tokens",
        (span_layout.before - segment_tokens(0) as usize) as u64,
        shape.head as u64,
    );
    procedure.int(
        "system_prompt_tokens",
        segment_tokens(0),
        shape.system as u64,
    );
    procedure.int(
        "document_tokens",
        segment_tokens(1) + segment_tokens(2) + segment_tokens(3),
        shape.document() as u64,
    );
    procedure.int(
        "question_tokens",
        segment_tokens(segments - 1),
        shape.question as u64,
    );
    procedure.check(
        "canary_strings_distinct",
        Json::Bool(distinct),
        Json::Bool(params::CANARY_STRINGS_DISTINCT),
    );
    procedure.text(
        "tiny_model_config_digest",
        &observed_config_digest(&forget_backend),
        params::TINY_MODEL_CONFIG_DIGEST,
    );
    procedure.text("tokenizer_pin", &tokenizer_text, params::TOKENIZER_PIN);
    procedure.constant("moe_topology", Json::str(params::MOE_TOPOLOGY));
    procedure.int(
        "test_model_layers",
        config.num_layers as u64,
        params::TEST_MODEL_LAYERS,
    );
    procedure.constant(
        "canary_generator_seed",
        Json::Int(params::CANARY_GENERATOR_SEED as i64),
    );
    procedure.text(
        "canary_expected_digest",
        &identity_digest,
        params::CANARY_EXPECTED_DIGEST,
    );
    let missing: Vec<&str> = params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL
        .iter()
        .copied()
        .filter(|name| !procedure.observed.iter().any(|(seen, _)| seen == name))
        .collect();
    if !missing.is_empty() {
        return Err(format!("procedure parameters not recorded: {missing:?}"));
    }

    // The metric counters, all measured above.
    tally.set("bitwise_mismatch_count", bitwise);
    tally.set("canary_logit_shift", shift);
    // Under-reuse and accounting disagreements add to the recomputation (never cancelling it).
    let mut recompute_total = forget.outcome.prefilled_tokens + recompute_extra;
    if accounting_defect > 0 && recompute_total == tokens_after_span {
        recompute_total += accounting_defect;
    }
    tally.set("recompute_tokens", recompute_total as i64);
    tally.set("span_insertions", span_insertions);
    tally.set("span_evictions", i64::from(evicted));
    tally.set(
        "downstream_recomputations",
        forget.outcome.inserted_chunks as i64,
    );
    tally.set("canary_probes", probes_done);
    tally.set(
        "stale_downstream_rows_reused",
        (stale_rows + stale_extra) as i64,
    );
    tally.set("tokens_after_span", tokens_after_span as i64);
    tally.set("canary_span_tokens", canary_span_tokens as i64);
    tally.set("downstream_rows_before_forget", downstream_rows as i64);
    tally.set("prefix_reuse_tokens", forget.outcome.hit_tokens as i64);
    tally.set("post_forget_probes", post_forget);
    tally.set("procedure_mismatches", procedure.mismatches);
    tally.set_json(
        "canary_identity_set_digest",
        Json::str(identity_digest.clone()),
    );
    tally.set("decode_digest_mismatches", decode_mismatch);
    tally.set("non_finite_logits_entries", non_finite_total);
    tally.set("finite_comparisons", finite_comparisons);
    tally.set("cold_anchor_mismatches", anchor_mismatches);
    tally.set("stale_rows_beyond_the_forget_run", stale_extra as i64);
    tally.set("recompute_tokens_beyond_the_tail", recompute_extra as i64);
    tally.set("outcome_accounting_disagreement", accounting_defect as i64);
    tally.set("probe_logit_entries_differing", probe_entries);
    tally.set("probe_digests_differing", probe_digests);
    tally.set("forget_logit_entries_differing", forget_diff.entries);
    tally.set("forget_digest_differing", i64::from(forget_digest_differs));
    tally.set("span_rows_citing", rows_citing_span as i64);
    tally.set("span_rows_retired", retired as i64);
    tally.set(
        "forget_rows_inserted",
        forget.outcome.inserted_chunks as i64,
    );
    tally.set("negative_control_bitwise", control_bitwise);
    tally.set("negative_control_stale_rows", control_stale_rows as i64);
    tally.set("negative_control_ulps", stale_diff.max_ulps);
    tally.set("nan_pairs", forget_diff.nan_pairs + stale_diff.nan_pairs);

    let canary_identity = Json::obj([
        ("digest", Json::str(identity_digest.clone())),
        ("pinned", Json::str(params::CANARY_EXPECTED_DIGEST)),
        (
            "matches",
            Json::Bool(identity_digest == params::CANARY_EXPECTED_DIGEST),
        ),
        ("rule", Json::str(params::CANARY_GENERATOR_RULE)),
        (
            "generator_seed",
            Json::Int(params::CANARY_GENERATOR_SEED as i64),
        ),
        ("count", Json::Int(canary_set.len() as i64)),
        ("distinct", Json::Bool(distinct)),
        ("embedded_in_span", Json::Int(embedded.len() as i64)),
        (
            "never_in_span",
            Json::Int((canary_set.len() - embedded.len()) as i64),
        ),
        ("canaries", Json::strs(&canary_set)),
    ]);
    let procedure_json = Json::Obj(
        procedure
            .observed
            .iter()
            .cloned()
            .chain([
                (
                    "expected".to_string(),
                    Json::Obj(procedure.expected.clone()),
                ),
                ("differing".to_string(), Json::strs(&procedure.differing)),
                (
                    "constants_never_counted".to_string(),
                    Json::strs(&procedure.constants),
                ),
            ])
            .collect(),
    );
    let expected = Json::obj([
        ("prefix_reuse_tokens", Json::Int(shape.prefix() as i64)),
        (
            "prefix_reuse_chunks",
            Json::Int((shape.prefix() / chunk) as i64),
        ),
        ("recompute_tokens", Json::Int(shape.after_span() as i64)),
        ("tokens_after_span", Json::Int(shape.after_span() as i64)),
        (
            "downstream_recomputations",
            Json::Int((shape.after_span() / chunk) as i64),
        ),
        (
            "downstream_rows_before_forget",
            Json::Int((shape.after_span() / chunk) as i64),
        ),
        (
            "rows_citing_span",
            Json::Int(((shape.span + shape.after_span()) / chunk) as i64),
        ),
        ("canary_probes", Json::Int(params::CANARY_PROBES as i64)),
        ("note", Json::str("shape-derived expectations recorded next to the measured values; no rule reads them (at the pinned shape they are the config's 1088, 68, 2880, 180, 180)")),
    ]);
    Ok(Findings {
        expected,
        cases,
        timings: timings.json(),
        canary_identity,
        procedure: procedure_json,
        roles: Json::ints(prompt.roles.iter().map(|role| *role as i64)),
    })
}

/// The canaries embedded in the request's span segment, read back from its tokens.
fn span_texts(request: &SegmentedModelRequest) -> Res<Vec<String>> {
    let span = request
        .segments()
        .iter()
        .find(|segment| segment.source().kind() == SegmentKind::Derived)
        .ok_or("the request holds no span segment")?;
    let text = token_text(span.tokens());
    Ok(text
        .as_bytes()
        .chunks(params::CANARY_BYTES)
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canary_set_is_the_pinned_one() {
        let set: Vec<String> = (0..32).map(canary).collect();
        assert_eq!(canary_set_digest(&set), params::CANARY_EXPECTED_DIGEST);
        assert_eq!(set.iter().collect::<BTreeSet<_>>().len(), 32);
        assert!(set.iter().all(|text| text.len() == params::CANARY_BYTES));
        assert_eq!(set[0], "055f52db4e24ed44");
    }

    #[test]
    fn a_changed_canary_set_has_another_digest() {
        let mut set: Vec<String> = (0..32).map(canary).collect();
        set[31] = canary(5);
        assert_ne!(canary_set_digest(&set), params::CANARY_EXPECTED_DIGEST);
        set.pop();
        assert_ne!(canary_set_digest(&set), params::CANARY_EXPECTED_DIGEST);
    }

    #[test]
    fn ulp_distance_is_zero_exactly_for_equal_values() {
        assert_eq!(ulp_distance(1.5, 1.5), 0);
        assert_eq!(ulp_distance(0.0, -0.0), 0);
        assert_eq!(ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)), 1);
        assert_eq!(
            ulp_distance(-1.0, f32::from_bits((-1.0f32).to_bits() + 3)),
            3
        );
        assert!(ulp_distance(-1.0e-30, 1.0e-30) > 0);
        assert_eq!(ulp_distance(f32::NAN, f32::NAN), 0);
        assert_eq!(ulp_distance(f32::NAN, 1.0), NAN_DISTANCE);
        assert!(ulp_distance(f32::MAX, f32::MIN) < NAN_DISTANCE);
    }

    #[test]
    fn a_diff_counts_bit_patterns_and_a_signed_zero_is_a_bit_difference_without_a_shift() {
        let a = [1.0f32, 0.0, 2.0];
        let b = [1.0f32, -0.0, f32::from_bits(2.0f32.to_bits() + 2)];
        let d = diff(&a, &b);
        assert_eq!(d.entries, 2);
        assert_eq!(d.max_ulps, 2);
        assert_eq!(diff(&a, &a), Diff::default());
        let zero = diff(&[0.0], &[-0.0]);
        assert_eq!((zero.entries, zero.max_ulps), (1, 0));
    }

    #[test]
    fn the_selection_is_a_pure_function_and_roles_depend_on_the_seed() {
        assert_eq!(selection_digest(), selection_digest());
        let a = Prompt::new(17, params::PINNED_SHAPE);
        let b = Prompt::new(29, params::PINNED_SHAPE);
        assert_ne!(a.roles, b.roles);
        let distinct: BTreeSet<usize> = a.roles.iter().copied().collect();
        assert_eq!(distinct.len(), 4);
        assert_eq!(a.span.len(), 256);
    }

    #[test]
    fn the_requests_have_the_pinned_layout() {
        let prompt = Prompt::new(17, params::PINNED_SHAPE);
        let with = prompt.with_span();
        let without = prompt.without_span();
        assert_eq!(with.total_tokens(), 4224);
        assert_eq!(without.total_tokens(), 3968);
        assert_eq!(
            layout(&with).unwrap(),
            Layout {
                before: 1088,
                span: 256,
                after: 2880
            }
        );
        assert!(layout(&without).is_err());
        let probe = prompt.probe(3, &canary(3));
        assert_eq!(probe.total_tokens(), 3968 + 16);
        assert_eq!(
            span_texts(&with).unwrap(),
            (0..16).map(canary).collect::<Vec<_>>()
        );
        assert_eq!(limits(params::PINNED_SHAPE).max_prompt_tokens, 4240);
    }

    #[test]
    fn the_limits_are_raised_in_the_harness_and_leave_the_pinned_config_alone() {
        let raised = backend(params::PINNED_SHAPE).unwrap();
        assert_eq!(raised.limits().max_prompt_tokens, 4240);
        assert_eq!(
            observed_config_digest(&raised),
            params::TINY_MODEL_CONFIG_DIGEST
        );
        assert_eq!(ReferenceKvLimits::default().max_prompt_tokens, 256);
    }

    #[test]
    fn a_procedure_value_that_differs_from_its_pin_is_one_mismatch_and_is_named() {
        let mut procedure = Procedure::default();
        procedure.int("span_tokens", 256, 256);
        procedure.int("span_tokens_other", 255, 256);
        procedure.text("tokenizer_pin", "other", "ByteTokenizer");
        procedure.check("flag", Json::Bool(false), Json::Bool(true));
        assert_eq!(procedure.mismatches, 3);
        assert_eq!(
            procedure.differing,
            ["span_tokens_other", "tokenizer_pin", "flag"]
        );
        assert_eq!(procedure.observed.len(), 4);
    }

    #[test]
    fn a_request_with_two_spans_or_none_has_no_layout() {
        let identity = |name: &str| {
            Part::bound(
                SegmentKind::Derived,
                name,
                params::SPAN_GENERATION,
                vec![65; 16],
            )
        };
        let two = corpus::request(
            "two",
            &[
                Part::raw(vec![1; 16]),
                identity("a"),
                identity("b"),
                Part::raw(vec![2; 16]),
            ],
        );
        assert!(layout(&two).is_err());
        let none = corpus::request("none", &[Part::raw(vec![1; 16])]);
        assert!(layout(&none).is_err());
        let one = corpus::request(
            "one",
            &[
                Part::raw(vec![1; 32]),
                identity("a"),
                Part::raw(vec![2; 16]),
            ],
        );
        assert_eq!(
            layout(&one).unwrap(),
            Layout {
                before: 32,
                span: 16,
                after: 16
            }
        );
    }

    #[test]
    fn an_insert_refusal_on_the_honest_path_is_an_error_and_a_clean_run_is_not() {
        let shape = params::REDUCED_SHAPE;
        let prompt = Prompt::new(17, shape);
        let tight = ReferenceKvLimits {
            catalog: CatalogLimits {
                max_trie_nodes: 3,
                ..limits(shape).catalog
            },
            ..limits(shape)
        };
        let backend = ReferenceKvReuseBackend::new(tight, lab::scope()).unwrap();
        let mut timings = Timings::default();
        let probe = run(
            &mut timings,
            "tight",
            &backend,
            &prompt.without_span(),
            ReferenceKvProbeMode::Warm,
        )
        .unwrap();
        assert!(accepted("tight", &probe).is_err());
        assert_eq!(timings.0.len(), 1);
        let roomy = self::backend(shape).unwrap();
        let clean = run(
            &mut timings,
            "roomy",
            &roomy,
            &prompt.without_span(),
            ReferenceKvProbeMode::Warm,
        )
        .unwrap();
        assert!(accepted("roomy", &clean).is_ok());
    }

    fn probe_of(logits: Vec<f32>) -> ReferenceKvProbe {
        let shape = params::REDUCED_SHAPE;
        let backend = backend(shape).unwrap();
        let prompt = Prompt::new(17, shape);
        let mut probe = backend
            .probe(
                &lab::scope(),
                &prompt.without_span(),
                ReferenceKvProbeMode::Cold,
            )
            .unwrap();
        probe.prompt_logits = logits;
        probe
    }

    #[test]
    fn a_comparison_counts_digest_decode_entries_and_non_finite_values() {
        let a = probe_of(vec![1.0; 256]);
        let same = compare(&a, &a.clone());
        assert_eq!(same.total(), 0);
        assert_eq!(same.logits, Diff::default());

        let mut changed = a.clone();
        changed.prompt_logits[3] = 2.0;
        changed.kv_state_digest[0] ^= 1;
        changed.outcome.generated.clear();
        let cmp = compare(&changed, &a);
        assert_eq!((cmp.digest, cmp.decode, cmp.logits.entries), (1, 1, 1));
        assert_eq!(cmp.total(), 3);

        // Identical non-finite vectors on both sides are no pass.
        let nan = probe_of(vec![f32::NAN; 256]);
        let cmp = compare(&nan, &nan.clone());
        assert_eq!(cmp.logits.entries, 0);
        assert_eq!(cmp.non_finite, 512);
        assert_eq!(cmp.total(), 512);
        let mut one = a.clone();
        one.prompt_logits[0] = f32::INFINITY;
        assert_eq!(compare(&one, &a).non_finite, 1);
    }

    #[test]
    fn reuse_defects_separate_over_reuse_under_reuse_and_disagreeing_accounting() {
        let a = probe_of(vec![0.0; 256]);
        let mut probe = a.clone();
        probe.outcome.prompt_tokens = 64;
        probe.outcome.hit_tokens = 32;
        probe.outcome.prefilled_tokens = 32;
        assert_eq!(reuse_defects(&probe, 64, 32), Reuse::default());
        // Over-reuse: 48 served where 32 is right.
        probe.outcome.hit_tokens = 48;
        probe.outcome.prefilled_tokens = 16;
        let over = reuse_defects(&probe, 64, 32);
        assert_eq!(
            (over.over_rows, over.extra_prefilled, over.accounting),
            (1, 0, 0)
        );
        // Under-reuse: 16 served, 48 prefilled.
        probe.outcome.hit_tokens = 16;
        probe.outcome.prefilled_tokens = 48;
        let under = reuse_defects(&probe, 64, 32);
        assert_eq!(
            (under.over_rows, under.extra_prefilled, under.accounting),
            (0, 16, 0)
        );
        // The outcome's own books disagree with the request and with each other.
        probe.outcome.prefilled_tokens = 40;
        assert_eq!(reuse_defects(&probe, 64, 32).accounting, 8);
        probe.outcome.prompt_tokens = 50;
        assert!(reuse_defects(&probe, 64, 32).accounting > 8);
    }

    #[test]
    fn a_constant_procedure_parameter_is_recorded_and_never_counted() {
        let mut procedure = Procedure::default();
        procedure.constant("moe_topology", Json::str("none"));
        assert_eq!(procedure.mismatches, 0);
        assert_eq!(procedure.constants, ["moe_topology"]);
        assert_eq!(procedure.observed.len(), 1);
    }
}
