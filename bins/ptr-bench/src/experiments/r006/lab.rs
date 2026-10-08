//! What the cases and the matrix of R006 share: the scale and shape of a run, the corpus
//! roles, the backends, the requests, the bitwise comparisons, the independent cold anchors and
//! the structural measurements (what a lookup returned or admitted, what a session emitted or
//! inserted after an event).
//!
//! Staleness is structural here (owner decision 5): in a one-layer model a stale page is
//! bit-identical to a fresh one, so no function in this module decides staleness from an
//! output difference. [`reachable`] counts the candidates a lookup returned, [`admitted`] the
//! spans it admitted.

use super::params;
use super::pins;
use crate::experiments::layered_kv::corpus::{self, hex, order, sequence, Part};
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::layered_kv::lab::{self as shared, fail, Res};
use ptr_model_api::{KvReuseMode, KvReuseTaint, SegmentKind, SegmentedModelRequest};
use ptr_pods::{KvKeyEncoding, Qwen2PagedStack, Qwen2RopeScaling};
use ptr_runtime::{
    CatalogLimits, KvCancelReason, KvLivenessOracle, PositionalControl, PositionalOptions,
    PositionalReport, ReferenceKvLimits, ReferenceKvProbe, ReferenceKvProbeMode,
    ReferenceKvReuseBackend, ReferenceKvStackShape, SecondaryConfig, SecondaryIndex,
    REFERENCE_KV_WEIGHT_SEED,
};
use std::collections::BTreeSet;
use std::time::Instant;

/// Why a case could not be completed, and who is to blame.
///
/// * [`Failure::Invalid`] is a failure of the harness itself (a refusal of the wrong type, a
///   cold baseline that differs from the independent stack, an accounting that does not add up,
///   a leaked pin, a negative control that does not trip): the run is INVALID (exit status 2)
///   and says nothing about the hypothesis.
/// * [`Failure::Control`] is a failed positive control (a live control that reuses less than
///   the whole segment, an origin the index did not take, a control that cannot show its
///   premise): the runtime does not do the positional reuse the experiment measures. The case
///   is recorded as failed, its post-event assertions are skipped and its case counters are
///   not incremented, so a coverage rule fails and the seed FALSIFIES (vacuous); it is never
///   a pass and never an exit status 2.
#[derive(Debug)]
pub enum Failure {
    Invalid(String),
    Control(String),
}

impl Failure {
    pub fn message(&self) -> &str {
        match self {
            Self::Invalid(message) | Self::Control(message) => message,
        }
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

/// A result that may be a failed control.
pub type Run<T> = Result<T, Failure>;

/// A failed positive control.
pub fn control<T>(message: impl Into<String>) -> Run<T> {
    Err(Failure::Control(message.into()))
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the smoke test only: the same cases and cells with shorter shifts.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
        }
    }

    pub fn shape(self) -> Shape {
        match self {
            Self::Pinned => PINNED_SHAPE,
            Self::Reduced => REDUCED_SHAPE,
        }
    }
}

/// The sizes of a run. The member names of the per-shift metrics are always the pinned ones
/// (`shift_4096`); the reduced shape only shortens the shifts the smoke test really runs, and
/// the run says so (`scale`, `procedure.shape_is_pinned`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    /// The shift of the filler for each of `config.toml`'s `position_shift_tokens`, by member.
    pub shifts: [u64; 5],
    pub inserted: [u64; 2],
    /// Tokens of the reused segment `C`.
    pub segment: usize,
    pub tail: usize,
    /// Tokens of the segment `A` before the inserted one in the non-aligned cases.
    pub origin_prefix: usize,
    pub max_prompt_tokens: usize,
}

pub const PINNED_SHAPE: Shape = Shape {
    shifts: params::POSITION_SHIFT_TOKENS,
    inserted: params::NONALIGNED_INSERTION_TOKENS,
    segment: params::REUSED_SEGMENT_TOKENS as usize,
    tail: params::TAIL_TOKENS as usize,
    origin_prefix: params::NONALIGNED_ORIGIN_PREFIX_TOKENS as usize,
    max_prompt_tokens: params::HARNESS_MAX_PROMPT_TOKENS as usize,
};

pub const REDUCED_SHAPE: Shape = Shape {
    shifts: [1, 7, 20, 33, 129],
    inserted: params::NONALIGNED_INSERTION_TOKENS,
    segment: params::REUSED_SEGMENT_TOKENS as usize,
    tail: params::TAIL_TOKENS as usize,
    origin_prefix: params::NONALIGNED_ORIGIN_PREFIX_TOKENS as usize,
    max_prompt_tokens: 256,
};

impl Shape {
    /// The longest prompt of the shift cases.
    pub fn longest_prompt(&self) -> usize {
        self.shifts.iter().copied().max().unwrap_or(0) as usize + self.segment + self.tail
    }

    /// `harness_limits_rule`: the prompt limit covers the longest prompt and the new tokens.
    pub fn limits_hold(&self) -> bool {
        self.max_prompt_tokens >= self.longest_prompt() + params::HARNESS_MAX_NEW_TOKENS as usize
    }
}

/// The limits of the harness: the prompt limit raised for the longest shift (the crate's
/// default stays 256) and the catalog bounds raised so that no primary insert is refused.
pub fn limits(shape: Shape) -> ReferenceKvLimits {
    ReferenceKvLimits {
        max_prompt_tokens: shape.max_prompt_tokens,
        catalog: CatalogLimits {
            max_trie_nodes: 1 << 16,
            max_kv_page_bytes: 1 << 32,
            max_metadata_bytes: 1 << 30,
        },
        ..ReferenceKvLimits::default()
    }
}

#[cfg(test)]
thread_local! {
    /// A test-only replacement of the index's limits, to show a runtime whose index takes
    /// nothing is a falsified seed and not an exit status 2.
    pub(crate) static LIMITS_OVERRIDE: std::cell::Cell<Option<ptr_runtime::SecondaryLimits>> =
        const { std::cell::Cell::new(None) };
}

pub(crate) fn secondary_config() -> SecondaryConfig {
    #[allow(unused_mut)]
    let mut config = SecondaryConfig {
        window_tokens: params::ROLLING_WINDOW_TOKENS as usize,
        ..SecondaryConfig::default()
    };
    #[cfg(test)]
    if let Some(limits) = LIMITS_OVERRIDE.with(std::cell::Cell::get) {
        config.limits = limits;
    }
    config
}

/// Test-only: while alive, the secondary index of every backend the harness builds on this
/// thread refuses every segment longer than the rolling window (a runtime whose index takes
/// nothing), restored on drop.
#[cfg(test)]
pub(crate) struct TestLimits;

#[cfg(test)]
impl TestLimits {
    pub(crate) fn refusing_every_insert() -> Self {
        LIMITS_OVERRIDE.with(|limits| {
            limits.set(Some(ptr_runtime::SecondaryLimits {
                max_tokens_per_run: params::ROLLING_WINDOW_TOKENS as usize,
                ..ptr_runtime::SecondaryLimits::default()
            }));
        });
        Self
    }
}

#[cfg(test)]
impl Drop for TestLimits {
    fn drop(&mut self) {
        LIMITS_OVERRIDE.with(|limits| limits.set(None));
    }
}

/// The rolling window the index runs with, read back from an index built from the
/// harness's configuration.
pub fn observed_window() -> Res<u64> {
    SecondaryIndex::new(secondary_config())
        .map(|index| index.window_tokens() as u64)
        .map_err(fail("an index at the pinned window"))
}

/// A one-layer `PreRope` backend of the pinned weights with positional reuse opted in (the
/// secondary index at the pinned window), under `control`.
pub fn positional_backend(
    shape: Shape,
    control: PositionalControl,
) -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::with_layers(
        limits(shape),
        shared::scope(),
        REFERENCE_KV_WEIGHT_SEED,
        params::TEST_MODEL_LAYERS as usize,
    )
    .map_err(fail("a one-layer backend"))?
    .with_positional_reuse(PositionalOptions {
        secondary: secondary_config(),
        control,
    })
    .map_err(fail("opting into positional reuse"))
}

/// The same one-layer backend without positional reuse: the primary catalog only.
pub fn primary_only_backend(shape: Shape) -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::with_layers(
        limits(shape),
        shared::scope(),
        REFERENCE_KV_WEIGHT_SEED,
        params::TEST_MODEL_LAYERS as usize,
    )
    .map_err(fail("a primary-only one-layer backend"))
}

/// The negative control of the key encoding: a one-layer `PostRope` stack that is allowed to
/// reuse positionally, so its keys, rotated at the position they were computed at, are wrong
/// at the new one.
pub fn post_rope_backend(shape: Shape) -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::with_stack_shape(
        limits(shape),
        shared::scope(),
        REFERENCE_KV_WEIGHT_SEED,
        Qwen2RopeScaling::None,
        ReferenceKvStackShape {
            layers: params::TEST_MODEL_LAYERS as usize,
            key_encoding: KvKeyEncoding::PostRope,
        },
    )
    .map_err(fail("a PostRope one-layer backend"))?
    .with_positional_reuse(PositionalOptions {
        secondary: secondary_config(),
        control: PositionalControl::PermitPostRopeKeys,
    })
    .map_err(fail("opting the PostRope control into positional reuse"))
}

// ---- the corpus and its roles ----------------------------------------------------------------

/// The role a corpus sequence plays in a seed's run.
#[derive(Clone, Copy, Debug)]
pub enum Role {
    Filler = 0,
    Segment = 1,
    Tail = 2,
    Inserted = 3,
    Upstream = 4,
    Ancestor = 5,
    PrefixA = 6,
    Spare = 7,
}

/// The pinned corpus (sequence `i` of the corpus generator) and the seed's assignment of
/// sequences to roles (`order(seed, 60, 8)`).
pub struct Material {
    sequences: Vec<Vec<u32>>,
    roles: Vec<usize>,
}

impl Material {
    pub fn new(seed: u64) -> Self {
        Self {
            sequences: (0..params::SELECTION_SEQUENCES as u64)
                .map(|index| {
                    sequence(
                        params::CORPUS_GENERATOR_SEED,
                        index,
                        params::SELECTION_SEQUENCE_TOKENS,
                    )
                })
                .collect(),
            roles: order(seed, params::ROLE_ORDER_LABEL, params::SELECTION_SEQUENCES),
        }
    }

    /// The digest of the sequences this run actually draws from.
    pub fn selection_digest(&self) -> String {
        corpus::selection_digest(params::SELECTION_DOMAIN, &self.sequences)
    }

    pub fn roles(&self) -> Vec<i64> {
        self.roles.iter().map(|role| *role as i64).collect()
    }

    /// `len` tokens of the sequence of `role` from `offset`.
    pub fn take(&self, role: Role, offset: usize, len: usize) -> Vec<u32> {
        let sequence = &self.sequences[self.roles[role as usize]];
        sequence[offset..offset + len].to_vec()
    }

    /// The reused segment `C`.
    pub fn segment(&self, shape: Shape) -> Vec<u32> {
        self.take(Role::Segment, 0, shape.segment)
    }

    pub fn tail(&self, shape: Shape) -> Vec<u32> {
        self.take(Role::Tail, 0, shape.tail)
    }

    /// The shift filler of `len` tokens (a `Raw` segment inserted upstream of `C`).
    pub fn filler(&self, len: usize) -> Vec<u32> {
        self.take(Role::Filler, 0, len)
    }

    /// The `Raw` segment inserted between `A` and `C`.
    pub fn inserted(&self, len: usize) -> Vec<u32> {
        self.take(Role::Inserted, 0, len)
    }

    pub fn prefix_a(&self, shape: Shape) -> Vec<u32> {
        self.take(Role::PrefixA, 0, shape.origin_prefix)
    }

    /// The upstream segment `U` (and the matrix's tested segment `X`).
    pub fn upstream(&self) -> Vec<u32> {
        self.take(Role::Upstream, 0, UPSTREAM_TOKENS)
    }

    /// The `Raw` ancestor of the raw-ancestor case.
    pub fn ancestor(&self) -> Vec<u32> {
        self.take(Role::Ancestor, 0, ANCESTOR_TOKENS)
    }

    /// A filler of `len` tokens from a spare slot; distinct slots share no token window, so a
    /// filler never matches another request's filler or any role's segment.
    pub fn spare(&self, slot: usize, len: usize) -> Vec<u32> {
        self.take(Role::Spare, slot * SPARE_SLOT_TOKENS, len)
    }
}

pub const UPSTREAM_TOKENS: usize = 20;
pub const ANCESTOR_TOKENS: usize = 20;
/// Tokens between two spare slots.
pub const SPARE_SLOT_TOKENS: usize = 128;
/// Tokens of the fillers the cases put in front of a segment they test.
pub const SHORT_FILLER_TOKENS: usize = 24;

/// The lifecycle identity of the reused segment `C` (a memory input, live in every cell).
pub const C_IDENTITY: &str = "mem:r006-c";

/// The lifecycle identity of the bystander of C3 and C4 (a memory input that is never
/// invalidated).
pub const BYSTANDER_IDENTITY: &str = "mem:r006-bystander";

pub fn c_part(material: &Material, shape: Shape) -> Part {
    Part::bound(SegmentKind::Memory, C_IDENTITY, 1, material.segment(shape))
}

pub fn tail_part(material: &Material, shape: Shape) -> Part {
    Part::raw(material.tail(shape))
}

/// The upstream segment `U` of the revocation, supersession and refusal cases: a memory input.
pub fn u_part(material: &Material, generation: u64) -> Part {
    Part::bound(
        SegmentKind::Memory,
        params::UPSTREAM_IDENTITY,
        generation,
        material.upstream(),
    )
}

/// The tokens of a request as `u32`.
pub fn tokens_of(request: &SegmentedModelRequest) -> Vec<u32> {
    request.tokens().iter().map(|token| token.0).collect()
}

/// The tokens of a live-control request `[filler][parts...][tail]` a positional lookup can
/// serve: every segment but the leading filler (new content) and the last, whose last token
/// the stack must compute itself.
pub fn reusable_tokens(request: &SegmentedModelRequest) -> usize {
    let segments = request.segments();
    segments[1..segments.len().saturating_sub(1)]
        .iter()
        .map(|segment| segment.tokens().len())
        .sum()
}

/// The offset of the first token of segment `index`.
pub fn start_of(request: &SegmentedModelRequest, index: usize) -> usize {
    request.segments()[..index]
        .iter()
        .map(|segment| segment.tokens().len())
        .sum()
}

// ---- wall time -----------------------------------------------------------------------------

/// Wall time of each runtime call, for the `timing_ms` field only (never a verdict field).
#[derive(Default)]
pub struct Timings(pub Vec<(String, f64)>);

impl Timings {
    pub fn json(&self) -> Json {
        Json::Arr(
            self.0
                .iter()
                .map(|(name, ms)| {
                    Json::obj([("run", Json::str(name.clone())), ("ms", Json::Float(*ms))])
                })
                .collect(),
        )
    }

    pub fn last(&self) -> f64 {
        self.0.last().map_or(0.0, |(_, ms)| *ms)
    }
}

// ---- runs ----------------------------------------------------------------------------------

/// The logits vector length every probe must return.
const LOGITS_ENTRIES: usize = params::VOCAB_SIZE as usize;

fn checked(name: &str, probe: ReferenceKvProbe) -> Res<ReferenceKvProbe> {
    if probe.prompt_logits.len() != LOGITS_ENTRIES {
        return Err(format!(
            "{name}: {} logits entries, not {LOGITS_ENTRIES}",
            probe.prompt_logits.len()
        ));
    }
    Ok(probe)
}

/// A positional run of `request` to its end (restarting cold if it is cancelled, which an
/// uneventful case never is). Required to have made no restart: the cases that expect one use
/// sessions by hand.
pub fn run_positional(
    timings: &mut Timings,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    oracle: &dyn KvLivenessOracle,
) -> Res<ReferenceKvProbe> {
    let started = Instant::now();
    let probe = backend
        .probe_positional(&shared::scope(), request, oracle)
        .map_err(fail(name))?;
    timings
        .0
        .push((name.to_string(), started.elapsed().as_secs_f64() * 1000.0));
    if probe.outcome.restarted_after_cancel {
        return Err(format!("{name}: the run restarted after a cancellation"));
    }
    let probe = checked(name, probe)?;
    accounted(name, &probe)?;
    Ok(probe)
}

/// A cold run of `request` (no lookup, no insert, no secondary index).
pub fn run_cold(
    timings: &mut Timings,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
) -> Res<ReferenceKvProbe> {
    let started = Instant::now();
    let probe = backend
        .probe(&shared::scope(), request, ReferenceKvProbeMode::Cold)
        .map_err(fail(name))?;
    timings
        .0
        .push((name.to_string(), started.elapsed().as_secs_f64() * 1000.0));
    checked(name, probe)
}

/// A warm run on a backend without positional reuse (the primary catalog's lookup and insert).
pub fn run_warm(
    timings: &mut Timings,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
) -> Res<ReferenceKvProbe> {
    let started = Instant::now();
    let probe = backend
        .probe(&shared::scope(), request, ReferenceKvProbeMode::Warm)
        .map_err(fail(name))?;
    timings
        .0
        .push((name.to_string(), started.elapsed().as_secs_f64() * 1000.0));
    checked(name, probe)
}

/// The positional report of a run, which every positional run has.
pub fn report(probe: &ReferenceKvProbe) -> Res<&PositionalReport> {
    probe
        .outcome
        .positional
        .as_ref()
        .ok_or_else(|| "a positional run without a positional report".to_string())
}

/// The runtime's own accounting must add up for a run that was not restarted: the prompt is
/// the primary hit plus the repacked tokens plus the prefilled ones. A disagreement means the
/// counters the metrics read cannot be trusted, so the case cannot be run.
fn accounted(name: &str, probe: &ReferenceKvProbe) -> Res<()> {
    let outcome = &probe.outcome;
    let reused = report(probe)?.reused_tokens;
    if outcome.prompt_tokens != outcome.hit_tokens + reused + outcome.prefilled_tokens {
        return Err(format!(
            "{name}: the outcome's accounting disagrees: {} prompt tokens, {} primary hits, {} repacked, {} prefilled",
            outcome.prompt_tokens, outcome.hit_tokens, reused, outcome.prefilled_tokens
        ));
    }
    Ok(())
}

/// The result is labelled Exact and carries no taint (`KvReuseReport`). A run without a
/// `kv_report` is not a tainted result: it is a result the harness cannot read, so the run is
/// INVALID (exit status 2) rather than a runtime falsification.
pub fn exact_and_untainted(probe: &ReferenceKvProbe) -> Res<(bool, bool)> {
    let report = probe.outcome.run.kv_report.as_ref().ok_or_else(|| {
        "a positional run without a kv_report: its taint cannot be read".to_string()
    })?;
    Ok((
        report.mode() == KvReuseMode::Exact,
        report.taint() == KvReuseTaint::Exact,
    ))
}

// ---- structural measurements ------------------------------------------------------------------

/// Spans a positional lookup admitted in a case that must admit nothing: the spliced spans plus
/// the segments whose only admitted run covered the last prompt token. The runtime never
/// splices the last token (the stack needs its logits), so such an admission is counted in
/// `last_token_skips` and not in `admitted_splices`; a runtime that stops enforcing a check for
/// the final segment would otherwise be invisible. In every negative case the tail run is
/// refused, so a correct runtime gives 0. Not for the positive cases (C1, C2, the reorder
/// control), where `admitted_splices` alone is the measure.
pub fn admitted(report: &PositionalReport) -> i64 {
    report.admitted_splices as i64 + report.last_token_skips as i64
}

/// Candidates a lookup returned in a case that must reach nothing, whether it admitted them
/// (spliced or skipped as the last segment) or refused them with a typed reason
/// (`reachability_definition`). After an applied invalidation none may happen.
pub fn reachable(report: &PositionalReport) -> i64 {
    admitted(report) + report.refusals.total() as i64
}

// ---- bitwise comparisons ----------------------------------------------------------------------

/// One comparison of a shifted result with its baseline, counted the way
/// `bitwise_mismatch_count` counts: one per differing KV-state digest, one per differing logits
/// entry (`f32` bit patterns), one per differing generated token, one for a differing decode
/// logits digest and one per non-finite logits entry on either side (a non-finite vector is
/// never a silent pass).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Compare {
    pub digest: i64,
    pub logits: i64,
    pub generated: i64,
    pub decode: i64,
    pub non_finite: i64,
}

impl Compare {
    pub fn total(&self) -> i64 {
        self.digest + self.logits + self.generated + self.decode + self.non_finite
    }
}

fn differing<T: PartialEq>(a: &[T], b: &[T]) -> i64 {
    a.iter().zip(b).filter(|(x, y)| x != y).count() as i64 + a.len().abs_diff(b.len()) as i64
}

pub fn compare(a: &ReferenceKvProbe, b: &ReferenceKvProbe) -> Compare {
    let non_finite = |probe: &ReferenceKvProbe| -> i64 {
        probe
            .prompt_logits
            .iter()
            .filter(|value| !value.is_finite())
            .count() as i64
    };
    Compare {
        digest: i64::from(a.kv_state_digest != b.kv_state_digest),
        logits: differing(&a.logit_bits(), &b.logit_bits()),
        generated: differing(&a.outcome.generated, &b.outcome.generated),
        decode: i64::from(a.outcome.logits_digest != b.outcome.logits_digest),
        non_finite: non_finite(a) + non_finite(b),
    }
}

/// The independent anchor for the cold run of a request: the generated tokens and the logits
/// digest of the independent one-layer stack (nothing of the runtime under test but the
/// pods stack they share).
pub struct Anchor {
    stack: Qwen2PagedStack,
    max_prompt_tokens: usize,
}

impl Anchor {
    pub fn new(max_prompt_tokens: usize) -> Res<Self> {
        Ok(Self {
            stack: pins::sized_stack(max_prompt_tokens)?,
            max_prompt_tokens,
        })
    }

    /// The anchor of `request`.
    pub fn of(&self, request: &SegmentedModelRequest) -> Res<(Vec<u32>, [u8; 32])> {
        pins::independent_cold_run_sized(&self.stack, &tokens_of(request), self.max_prompt_tokens)
    }
}

/// Mismatches of `probe` against an independent anchor: differing generated tokens plus one
/// for a differing decode logits digest.
pub fn anchor_mismatches(probe: &ReferenceKvProbe, anchor: &(Vec<u32>, [u8; 32])) -> i64 {
    let generated: Vec<u32> = probe
        .outcome
        .generated
        .iter()
        .map(|token| token.0)
        .collect();
    differing(&generated, &anchor.0) + i64::from(probe.outcome.logits_digest != anchor.1)
}

// ---- what escaped an event -----------------------------------------------------------------

/// What an in-flight race let through after the event step (`escaped_output_definition`).
#[derive(Clone, Copy, Debug)]
pub struct Escaped {
    /// Tokens emitted and spans repacked after the event, plus one if the session was not
    /// cancelled by the first step after it.
    pub outputs: i64,
    /// Secondary-index runs and catalog entries inserted from the event step on.
    pub inserts: i64,
    pub cancelled: bool,
    pub reason: Option<KvCancelReason>,
    /// The restart after a cancellation was a fresh prefill equal to a cold run and to the
    /// independent stack (`None` when the session was not cancelled).
    pub restart_fresh: Option<bool>,
    pub mismatches: i64,
}

// ---- the case-id sets ---------------------------------------------------------------------

/// The case ids of the identity-set check: the cases whose dependency was invalidated, and the
/// cases whose request then completed by a verified fresh prefill. They are filled where each
/// fact is observed, by different code, and compared as sets at the end.
#[derive(Default)]
pub struct CaseIds {
    pub invalidated: BTreeSet<String>,
    pub completed: BTreeSet<String>,
}

impl CaseIds {
    pub fn invalidated_digest(&self) -> String {
        shared::id_set_digest(self.invalidated.iter().map(String::as_str))
    }

    pub fn completed_digest(&self) -> String {
        shared::id_set_digest(self.completed.iter().map(String::as_str))
    }
}

// ---- the fresh prefill after an event ----------------------------------------------------------

/// What a request run after an invalidation was: a fresh prefill iff nothing was reused.
#[derive(Clone, Copy, Debug)]
pub struct Fresh {
    pub completed: bool,
    pub mismatches: i64,
}

/// Check the post-event run `probe` of `request` on `backend`: it must be a fresh prefill (no
/// primary hit, nothing repacked, the whole prompt prefilled, no restart), and its output must
/// equal a cold run on the backend and the independent anchor, bit for bit.
pub fn fresh_check(
    timings: &mut Timings,
    anchor: &Anchor,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    probe: &ReferenceKvProbe,
) -> Res<Fresh> {
    let baseline = run_cold(timings, &format!("{name}.cold"), backend, request)?;
    let anchored = anchor.of(request)?;
    let outcome = &probe.outcome;
    let reused = report(probe)?.reused_tokens;
    let completed = reused == 0
        && outcome.hit_tokens == 0
        && outcome.prefilled_tokens == outcome.prompt_tokens
        && outcome.prompt_tokens == request.total_tokens()
        && !outcome.restarted_after_cancel;
    Ok(Fresh {
        completed,
        mismatches: compare(probe, &baseline).total() + anchor_mismatches(probe, &anchored),
    })
}

// ---- procedure parameters --------------------------------------------------------------------

/// A procedure parameter checked against its pinned value: the observed value is recorded and
/// a difference counts as one mismatch.
#[derive(Default)]
pub struct Procedure {
    pub mismatches: i64,
    pub observed: Vec<(String, Json)>,
    pub expected: Vec<(String, Json)>,
    pub differing: Vec<String>,
    pub constants: Vec<String>,
}

impl Procedure {
    pub fn check(&mut self, name: &str, observed: Json, expected: Json) {
        if observed != expected {
            self.mismatches += 1;
            self.differing.push(name.to_string());
        }
        self.observed.push((name.to_string(), observed));
        self.expected.push((name.to_string(), expected));
    }

    /// A parameter that is a constant of this harness on both sides: recorded, never counted.
    pub fn constant(&mut self, name: &str, value: Json) {
        self.observed.push((name.to_string(), value.clone()));
        self.expected.push((name.to_string(), value));
        self.constants.push(name.to_string());
    }

    pub fn int(&mut self, name: &str, observed: u64, expected: u64) {
        self.check(name, Json::Int(observed as i64), Json::Int(expected as i64));
    }

    pub fn text(&mut self, name: &str, observed: &str, expected: &str) {
        self.check(name, Json::str(observed), Json::str(expected));
    }

    pub fn ints(&mut self, name: &str, observed: &[u64], expected: &[u64]) {
        self.check(
            name,
            Json::ints(observed.iter().map(|value| *value as i64)),
            Json::ints(expected.iter().map(|value| *value as i64)),
        );
    }

    pub fn strs(&mut self, name: &str, observed: &[String], expected: &[&str]) {
        self.check(name, Json::strs(observed), Json::strs(expected));
    }

    pub fn has(&self, name: &str) -> bool {
        self.observed.iter().any(|(seen, _)| seen == name)
    }

    pub fn json(&self) -> Json {
        Json::Obj(
            self.observed
                .iter()
                .cloned()
                .chain([
                    ("expected".to_string(), Json::Obj(self.expected.clone())),
                    ("differing".to_string(), Json::strs(&self.differing)),
                    (
                        "constants_never_counted".to_string(),
                        Json::strs(&self.constants),
                    ),
                ])
                .collect(),
        )
    }
}

/// What the cases observed of the run, for the procedure parameters.
#[derive(Default)]
pub struct Observed {
    pub shifts: Vec<u64>,
    pub inserted: Vec<u64>,
    pub segment_tokens: Option<u64>,
    pub tail_tokens: Option<u64>,
    pub origin_prefix_tokens: Option<u64>,
    pub declared_kinds: BTreeSet<&'static str>,
    pub production_modes: BTreeSet<&'static str>,
}

/// Hex of a digest as the result prints it.
pub fn digest_text(digest: &[u8; 32]) -> String {
    format!("sha256:{}", hex(digest))
}

/// The shared state of one run: the counters, the case-id sets, the timings and what was
/// observed.
pub struct Lab<'a> {
    pub tally: &'a mut Tally,
    pub shape: Shape,
    pub material: &'a Material,
    pub anchor: &'a Anchor,
    pub timings: Timings,
    pub ids: CaseIds,
    pub observed: Observed,
    /// The per-case records printed under `cases`.
    pub cases: Vec<(String, Json)>,
    /// The cases and cells whose positive control failed: `(name, why)`.
    pub failed_controls: Vec<(String, String)>,
}

impl Lab<'_> {
    /// Record a failed positive control (see [`Failure::Control`]): an internal counter the
    /// coverage rule `internal.live_control_failures <= 0` reads. The caller skips the
    /// case's post-event assertions and does not increment its case counters.
    pub fn control_failed(&mut self, name: &str, why: &str) {
        self.tally.add("live_control_failures", 1);
        self.failed_controls
            .push((name.to_string(), why.to_string()));
    }

    /// The outcome of case `name`: a failed control is recorded and the run goes on, a harness
    /// failure ends the run (exit status 2).
    pub fn settle(&mut self, name: &str, outcome: Run<()>) -> Res<()> {
        match outcome {
            Ok(()) => Ok(()),
            Err(Failure::Control(why)) => {
                self.control_failed(name, &why);
                Ok(())
            }
            Err(Failure::Invalid(why)) => Err(why),
        }
    }

    /// The failed controls as printed under `cases.control_failures`.
    pub fn failed_controls_json(&self) -> Json {
        Json::Arr(
            self.failed_controls
                .iter()
                .map(|(name, why)| {
                    Json::obj([
                        ("case", Json::str(name.clone())),
                        ("why", Json::str(why.clone())),
                    ])
                })
                .collect(),
        )
    }

    /// Count the mismatches of a fresh prefill after an invalidation.
    pub fn note_fresh(&mut self, case: &str, fresh: Fresh) {
        self.tally
            .add("fresh_prefill_output_mismatches", fresh.mismatches);
        if fresh.completed {
            self.ids.completed.insert(case.to_string());
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::experiments::layered_kv::emit::MetricList;
    use crate::experiments::r006::metrics;

    /// Run `body` on a lab at the reduced shape (seed 17) with its own tally.
    pub fn with_lab<T>(body: impl FnOnce(&mut Lab<'_>) -> T) -> T {
        let shape = REDUCED_SHAPE;
        let mut tally = Tally::new(metrics::LIST);
        let material = Material::new(17);
        let anchor = Anchor::new(shape.max_prompt_tokens).expect("an anchor");
        let mut lab = Lab {
            tally: &mut tally,
            shape,
            material: &material,
            anchor: &anchor,
            timings: Timings::default(),
            ids: CaseIds::default(),
            observed: Observed::default(),
            cases: Vec::new(),
            failed_controls: Vec::new(),
        };
        body(&mut lab)
    }

    pub const EMPTY_LIST: MetricList = MetricList {
        names: &[],
        grouped: &[],
    };
}

#[cfg(test)]
mod tests {
    use super::test_support::with_lab;
    use super::*;

    fn raw_request(lab: &Lab<'_>, id: &str, filler: usize) -> SegmentedModelRequest {
        corpus::request(
            id,
            &[
                Part::raw(lab.material.filler(filler)),
                c_part(lab.material, lab.shape),
                tail_part(lab.material, lab.shape),
            ],
        )
    }

    #[test]
    fn the_selection_digest_of_the_sequences_in_use_is_the_pinned_one() {
        let material = Material::new(17);
        assert_eq!(material.selection_digest(), params::SELECTION_DIGEST);
        assert_eq!(params::selection_digest(), params::SELECTION_DIGEST);
        // The seed assigns the roles; the corpus does not depend on it.
        let other = Material::new(29);
        assert_eq!(other.selection_digest(), material.selection_digest());
        assert_ne!(material.roles(), other.roles());
        let mut roles = material.roles();
        roles.sort_unstable();
        assert_eq!(roles, (0..8).collect::<Vec<i64>>());
    }

    #[test]
    fn spare_fillers_share_no_token_window_with_each_other() {
        let material = Material::new(17);
        let window = params::ROLLING_WINDOW_TOKENS as usize;
        let windows = |tokens: Vec<u32>| -> BTreeSet<Vec<u32>> {
            tokens.windows(window).map(<[u32]>::to_vec).collect()
        };
        let slots: Vec<BTreeSet<Vec<u32>>> = (0..18)
            .map(|slot| windows(material.spare(slot, SHORT_FILLER_TOKENS)))
            .collect();
        for (a, left) in slots.iter().enumerate() {
            for right in &slots[a + 1..] {
                assert!(left.is_disjoint(right));
            }
        }
        // And none shares one with the segment, the tail, the upstream or the ancestor.
        let shape = PINNED_SHAPE;
        let roles = windows(material.segment(shape))
            .into_iter()
            .chain(windows(material.tail(shape)))
            .chain(windows(material.upstream()))
            .chain(windows(material.ancestor()))
            .chain(windows(material.prefix_a(shape)))
            .collect::<BTreeSet<_>>();
        for slot in &slots {
            assert!(slot.is_disjoint(&roles));
        }
    }

    #[test]
    fn a_comparison_counts_digest_logits_tokens_decode_and_non_finite_entries() {
        with_lab(|lab| {
            let backend = positional_backend(lab.shape, PositionalControl::None).unwrap();
            let a = raw_request(lab, "a", 3);
            let b = raw_request(lab, "b", 5);
            let mut timings = Timings::default();
            let one = run_cold(&mut timings, "a", &backend, &a).unwrap();
            let again = run_cold(&mut timings, "a", &backend, &a).unwrap();
            let other = run_cold(&mut timings, "b", &backend, &b).unwrap();
            assert_eq!(compare(&one, &again).total(), 0);
            let cmp = compare(&one, &other);
            assert_eq!(cmp.digest, 1);
            assert!(cmp.logits > 0 && cmp.total() > cmp.logits);
            let mut broken = one.clone();
            broken.prompt_logits[3] = f32::NAN;
            assert_eq!(compare(&broken, &one).non_finite, 1);
            assert!(
                compare(&broken, &one).logits >= 1,
                "a NaN differs in its bits"
            );
        });
    }

    #[test]
    fn the_independent_anchor_reproduces_the_backends_cold_run_and_flags_a_difference() {
        with_lab(|lab| {
            let backend = positional_backend(lab.shape, PositionalControl::None).unwrap();
            let request = raw_request(lab, "anchor", 9);
            let mut timings = Timings::default();
            let cold = run_cold(&mut timings, "cold", &backend, &request).unwrap();
            let anchored = lab.anchor.of(&request).unwrap();
            assert_eq!(anchor_mismatches(&cold, &anchored), 0);
            let mut wrong = anchored.clone();
            wrong.0[0] ^= 1;
            wrong.1[0] ^= 1;
            assert_eq!(anchor_mismatches(&cold, &wrong), 2);
        });
    }

    #[test]
    fn the_case_id_sets_are_compared_as_sets_not_as_counts() {
        let mut ids = CaseIds::default();
        ids.invalidated.extend(["a".to_string(), "b".to_string()]);
        ids.completed.extend(["a".to_string(), "c".to_string()]);
        assert_ne!(
            ids.invalidated_digest(),
            ids.completed_digest(),
            "same size, other members"
        );
        ids.completed = ids.invalidated.clone();
        assert_eq!(ids.invalidated_digest(), ids.completed_digest());
    }

    #[test]
    fn a_procedure_value_that_differs_from_its_pin_is_one_mismatch_and_is_named() {
        let mut procedure = Procedure::default();
        procedure.int("a", 1, 1);
        procedure.int("b", 2, 1);
        procedure.ints("c", &[1, 2], &[1, 3]);
        procedure.strs("d", &["x".to_string()], &["x"]);
        procedure.constant("e", Json::str("k"));
        assert_eq!(procedure.mismatches, 2);
        assert_eq!(procedure.differing, ["b", "c"]);
        assert!(procedure.has("e") && !procedure.has("f"));
        assert_eq!(procedure.constants, ["e"]);
    }

    #[test]
    fn reachable_counts_what_a_lookup_returned_and_admitted_what_it_admitted() {
        let mut positional = PositionalReport {
            admitted_splices: 1,
            ..PositionalReport::default()
        };
        positional.refusals.original_upstream_dead = 2;
        assert_eq!(reachable(&positional), 3);
        assert_eq!(admitted(&positional), 1);
        positional.admitted_splices = 0;
        positional.refusals.original_upstream_dead = 0;
        assert_eq!(reachable(&positional), 0);
    }

    #[test]
    fn a_last_segment_skip_is_an_admission_and_a_reachable_candidate_in_a_negative_case() {
        let mut positional = PositionalReport {
            last_token_skips: 1,
            ..PositionalReport::default()
        };
        assert_eq!(admitted(&positional), 1);
        assert_eq!(reachable(&positional), 1);
        positional.refusals.original_upstream_omitted = 2;
        assert_eq!(reachable(&positional), 3);
        assert_eq!(positional.admitted_splices, 0, "and not a spliced span");
    }

    #[test]
    fn the_reusable_tokens_of_a_live_control_request_skip_the_filler_and_the_last_segment() {
        let request = corpus::request(
            "r",
            &[
                Part::raw(vec![1; 24]),
                Part::raw(vec![2; 20]),
                Part::raw(vec![3; 48]),
                Part::raw(vec![4; 16]),
            ],
        );
        assert_eq!(reusable_tokens(&request), 68);
    }

    #[test]
    fn a_run_without_a_kv_report_cannot_be_read() {
        with_lab(|lab| {
            let backend = positional_backend(lab.shape, PositionalControl::None).unwrap();
            let request = raw_request(lab, "r", 3);
            let mut timings = Timings::default();
            let mut probe = run_cold(&mut timings, "r", &backend, &request).unwrap();
            assert!(exact_and_untainted(&probe).is_ok());
            probe.outcome.run.kv_report = None;
            let error = exact_and_untainted(&probe).unwrap_err();
            assert!(error.contains("kv_report"), "{error}");
        });
    }

    #[test]
    fn a_failed_control_is_recorded_and_a_harness_failure_ends_the_run() {
        with_lab(|lab| {
            assert!(lab.settle("C1", Ok(())).is_ok());
            assert_eq!(lab.tally.get("live_control_failures"), 0);
            assert!(lab
                .settle("C2", control::<()>("the live control reused nothing"))
                .is_ok());
            assert_eq!(lab.tally.get("live_control_failures"), 1);
            assert_eq!(lab.failed_controls[0].0, "C2");
            let ended = lab.settle("C3", Err(Failure::Invalid("wrong type".into())));
            assert_eq!(ended, Err("wrong type".to_string()));
            assert_eq!(lab.tally.get("live_control_failures"), 1);
            let json = lab.failed_controls_json().render();
            assert!(json.contains("the live control reused nothing"), "{json}");
        });
    }

    #[test]
    fn the_shape_limits_follow_the_harness_limits_rule() {
        assert!(PINNED_SHAPE.limits_hold());
        assert_eq!(PINNED_SHAPE.max_prompt_tokens, 4352);
        let tight = Shape {
            max_prompt_tokens: PINNED_SHAPE.longest_prompt() + 8,
            ..PINNED_SHAPE
        };
        assert!(tight.limits_hold());
        assert!(!Shape {
            max_prompt_tokens: tight.max_prompt_tokens - 1,
            ..tight
        }
        .limits_hold());
    }
}
