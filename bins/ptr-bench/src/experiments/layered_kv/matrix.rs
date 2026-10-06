//! The invalidation coverage matrix of R009 and R010: every dependency kind x invalidation
//! type x path, each cell run at least once per seed and reported per cell.
//!
//! Every cell runs on **bound sessions** (`ReferenceKvReuseBackend::begin_session_bound`) of
//! the experiment's own binding (R009: a QKV-LoRA attached at layer 2; R010: the MoE with the
//! tracked expert `(1, i)`), so per-layer entries, per-layer dependency sets and the
//! re-resolving restart are what the cell exercises. The liveness oracle is a real
//! [`PtrRuntime`] ([`Authority`]): the base model, adapters, memory and derived inputs are
//! committed capsules, a semantic input is a key of the committed semantic state, and an
//! expert is a record written by the real admission path.
//!
//! A cell's request is `[segment A][segment B][raw tail]`: two 16-token segments (the tested
//! segment kind, or a neutral memory input that stays live for the other kinds) and a
//! four-token tail, so the warm run caches the boundaries 16 and 32 on every layer.
//!
//! What a post-event probe can show depends on where the dependency lives:
//!
//! * A **segment** dependency (semantic, memory, derived) sits in the request. A probe that
//!   cites the dead segment begins, looks up every layer and must be served nothing: a real
//!   lookup on a stale-eligible entry (`evidence = session_lookup`).
//! * A **binding-carried** dependency (base model, adapter, expert) is checked by
//!   `begin_session_bound` before any lookup, so the probe is refused there
//!   (`evidence = session_refused_at_begin`). The use-time rejection of an entry that
//!   depends on such a dependency is exercised by the in-flight cells, where a session that
//!   began live is cancelled at its next step. The post-event lookup counters
//!   (`post_event_lookups*`) therefore count the segment cells only, and the harness says so
//!   under `unsupported`.

use super::corpus::{self, Part, CHUNK_TOKENS};
use super::emit::{Json, Tally};
use super::lab::{
    self, commit_revocation, commit_supersession, compare, fail, refused_not_live, write_semantic,
    Authority, Ids, Res, Steps, BASE_MODEL_GENERATION, BASE_MODEL_IDENTITY,
};
use ptr_model_api::{SegmentKind, SegmentedModelRequest};
use ptr_runtime::{
    BindingRecord, IndexUpdate, KvRestoreReport, ReferenceKvFaults, ReferenceKvReuseBackend,
    StackBinding,
};
use ptr_types::Generation;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Path {
    Normal,
    Lag,
    Restore,
    Inflight,
}

impl Path {
    pub const ALL: [Path; 4] = [Path::Normal, Path::Lag, Path::Restore, Path::Inflight];

    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lag => "lag",
            Self::Restore => "restore",
            Self::Inflight => "inflight",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Invalidation {
    Revocation,
    Supersession,
    /// A committed fence of an expert (R010 only): the generation is unchanged.
    Fence,
}

impl Invalidation {
    pub const ALL: [Invalidation; 2] = [Invalidation::Revocation, Invalidation::Supersession];
    /// The types of an expert dependency.
    pub const EXPERT: [Invalidation; 3] = [
        Invalidation::Revocation,
        Invalidation::Supersession,
        Invalidation::Fence,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
            Self::Fence => "fence",
        }
    }
}

/// A generation-bearing prompt segment kind.
#[derive(Clone, Copy, Debug)]
pub struct SegSpec {
    pub kind: SegmentKind,
    /// The matrix name of the kind (`semantic_input`).
    pub cell_kind: &'static str,
    pub identity: &'static str,
    /// Stream labels of the two bound segments (and the tail) of this kind's requests.
    pub label: u64,
}

pub const SEGMENT_KINDS: [SegSpec; 3] = [
    SegSpec {
        kind: SegmentKind::Semantic,
        cell_kind: "semantic_input",
        identity: "doc:layered",
        label: 100,
    },
    SegSpec {
        kind: SegmentKind::Memory,
        cell_kind: "memory_input",
        identity: "mem:layered",
        label: 200,
    },
    SegSpec {
        kind: SegmentKind::Derived,
        cell_kind: "derived_input",
        identity: "ev:layered",
        label: 300,
    },
];

/// The neutral memory input of the cells whose tested dependency is not a segment: committed
/// live and never invalidated, so the request's lineage is bound without being the subject.
pub const NEUTRAL: SegSpec = SegSpec {
    kind: SegmentKind::Memory,
    cell_kind: "neutral",
    identity: "mem:neutral",
    label: 400,
};

/// What a cell tests.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    BaseModel,
    /// The experiment's binding-carried dependency (an adapter generation, an expert).
    Dependency,
    Segment(&'static SegSpec),
}

impl Kind {
    fn spec(self) -> &'static SegSpec {
        match self {
            Self::Segment(spec) => spec,
            _ => &NEUTRAL,
        }
    }
}

/// Parameters the code under test reports it runs, as the cell observed them.
#[derive(Clone, Debug)]
pub struct Observed {
    pub layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub weight_seed: u64,
    pub page_tokens: usize,
    pub max_new_tokens: usize,
    /// `(kind, first_layer, rank, scale)` of the bound adapter.
    pub adapter: Option<(String, usize, usize, f32)>,
    /// `(moe layers, experts per layer, top_k, mode)` of the bound MoE.
    pub moe: Option<(Vec<usize>, usize, usize, String)>,
}

pub fn observe(backend: &ReferenceKvReuseBackend, binding: &StackBinding) -> Observed {
    let config = backend.stack_config();
    let limits = backend.limits();
    Observed {
        layers: config.num_layers,
        hidden_size: config.layer.hidden_size,
        intermediate_size: config.layer.intermediate_size,
        num_heads: config.layer.num_attention_heads,
        num_kv_heads: config.layer.num_key_value_heads,
        vocab_size: config.vocab_size,
        weight_seed: ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
        page_tokens: ptr_runtime::KV_CHUNK_TOKENS,
        max_new_tokens: limits.max_new_tokens,
        adapter: binding.adapter.as_ref().map(|adapter| {
            (
                format!("{:?}", adapter.kind()),
                adapter.first_layer(),
                adapter.rank(),
                adapter.scale(),
            )
        }),
        moe: binding.moe.as_ref().map(|bound| {
            let layers: Vec<usize> = (0..config.num_layers)
                .filter(|layer| bound.moe.is_moe_layer(*layer))
                .collect();
            let experts = layers
                .first()
                .and_then(|layer| bound.moe.layer(*layer))
                .map_or(0, |layer| layer.experts.len());
            (
                layers,
                experts,
                bound.moe.top_k(),
                format!("{:?}", bound.moe.mode()),
            )
        }),
    }
}

/// What an experiment supplies for the binding-carried dependency and its own counters.
pub trait Rig {
    /// The matrix name of the binding-carried kind (`adapter_generation`).
    fn dependency_kind(&self) -> &'static str;
    /// The live binding, with the successor a supersession moves to.
    fn binding(&self, authority: &Authority) -> StackBinding;
    /// The binding a restart re-resolves to after `ty` of the dependency.
    fn resolved_binding(&self, ty: Invalidation, authority: &Authority) -> StackBinding;
    /// Whether the re-resolution a restart recorded is the one the owner decision names.
    fn resolution_holds(&self, record: &BindingRecord, ty: Invalidation) -> bool;
    /// The lifecycle identity of the dependency.
    fn identity(&self) -> String;
    /// Capsule identities the experiment's authority must hold besides the base model.
    fn capsules(&self) -> Vec<String>;
    /// Commit the dependency's initial liveness to `authority`.
    fn commit_live(&self, authority: &mut Authority) -> Res<()>;
    /// Commit `ty` of the dependency to `authority`'s ledger.
    fn commit_event(&self, authority: &mut Authority, ty: Invalidation) -> Res<()>;
    /// Entries the dependency's invalidation drops of `layers * chunks` warm entries.
    fn expected_dropped(&self, layers: usize, chunks: usize) -> usize;
    /// The generation a supersession moves the dependency to.
    fn superseded_generation(&self) -> u64;
    /// The observed parameters against the pinned ones: the mismatches' descriptions.
    fn procedure_mismatches(&self, observed: &Observed) -> Vec<String>;
    /// The per-cell procedure parameters, as observed.
    fn procedure_json(&self, observed: &Observed) -> Json;
    /// The experiment's own counters for one finished cell.
    fn note_cell(&self, tally: &mut Tally, cell: &CellRecord);
}

/// What one finished cell produced, for the experiment's own counters.
#[derive(Clone, Debug)]
pub struct CellRecord {
    pub kind: &'static str,
    pub is_dependency: bool,
    pub is_semantic_input: bool,
    pub ty: Invalidation,
    pub inflight: Option<Inflight>,
}

/// The in-flight race of a cell.
#[derive(Clone, Copy, Debug)]
pub struct Inflight {
    pub cancelled: bool,
    /// The restart ran from a fresh prefill and equals a cold run of the re-resolved binding
    /// (`None`: no restart was attempted, see the cell's evidence).
    pub restarted_fresh: Option<bool>,
    /// Tokens or outputs that escaped after the event.
    pub escaped: i64,
}

/// One cell as the result describes it.
#[derive(Clone, Debug)]
pub struct CellInfo {
    pub cell: String,
    pub evidence: &'static str,
    pub oracle: &'static str,
    pub procedure: Json,
    pub detail: Json,
}

impl CellInfo {
    pub fn json(&self) -> Json {
        Json::obj([
            ("cell", Json::str(self.cell.clone())),
            ("evidence", Json::str(self.evidence)),
            ("oracle", Json::str(self.oracle)),
            ("procedure", self.procedure.clone()),
            ("detail", self.detail.clone()),
        ])
    }
}

/// The name of a matrix cell.
pub fn cell_name(kind: &str, path: &str, ty: &str) -> String {
    format!("{kind}_{path}_{ty}")
}

struct Ctx<'a> {
    rig: &'a dyn Rig,
    kind: Kind,
    ty: Invalidation,
    cell: String,
    kind_name: &'static str,
}

impl Ctx<'_> {
    fn is_segment(&self) -> bool {
        matches!(self.kind, Kind::Segment(_))
    }
}

// ---- requests ---------------------------------------------------------------------------

/// `[segment A][segment B][raw tail]` citing `generation`.
fn lineage_request(
    spec: &SegSpec,
    id: &str,
    generation: u64,
    tail: Vec<u32>,
) -> SegmentedModelRequest {
    corpus::request(
        id,
        &[
            Part::bound(
                spec.kind,
                spec.identity,
                generation,
                corpus::sequence(super::SEQUENCE_SEED, 10_000 + spec.label, CHUNK_TOKENS),
            ),
            Part::bound(
                spec.kind,
                spec.identity,
                generation,
                corpus::sequence(super::SEQUENCE_SEED, 10_000 + spec.label + 1, CHUNK_TOKENS),
            ),
            Part::raw(tail),
        ],
    )
}

fn tail(spec: &SegSpec, salt: u64, len: usize) -> Vec<u32> {
    corpus::sequence(super::SEQUENCE_SEED, 10_000 + spec.label + salt, len)
}

fn warm_request(spec: &SegSpec, id: &str) -> SegmentedModelRequest {
    lineage_request(spec, id, 1, tail(spec, 7, 4))
}

/// One probe per stale-eligible boundary: a request that can hit exactly the entries at
/// `boundary` (16 or 32) and none deeper, citing generation `generation`.
fn probe_request(
    spec: &SegSpec,
    id: &str,
    boundary: usize,
    generation: u64,
) -> SegmentedModelRequest {
    let probe_tail = tail(spec, 9, 3);
    match boundary {
        16 => corpus::request(
            id,
            &[
                Part::bound(
                    spec.kind,
                    spec.identity,
                    generation,
                    corpus::sequence(super::SEQUENCE_SEED, 10_000 + spec.label, CHUNK_TOKENS),
                ),
                Part::raw(probe_tail),
            ],
        ),
        _ => lineage_request(spec, id, generation, probe_tail),
    }
}

const WARM_CHUNKS: usize = 2;
const PROBE_BOUNDARIES: [usize; WARM_CHUNKS] = [16, 32];

// ---- authority and events ---------------------------------------------------------------

fn authority(ctx: &Ctx<'_>) -> Res<Authority> {
    let spec = ctx.kind.spec();
    let mut capsules: Vec<String> = ctx.rig.capsules();
    capsules.push(NEUTRAL.identity.to_string());
    let segment_is_capsule =
        matches!(ctx.kind, Kind::Segment(spec) if spec.kind != SegmentKind::Semantic);
    if segment_is_capsule {
        capsules.push(spec.identity.to_string());
    }
    let refs: Vec<&str> = capsules.iter().map(String::as_str).collect();
    let mut authority = match ctx.kind {
        Kind::Segment(spec) if spec.kind == SegmentKind::Semantic => {
            Authority::with_semantic(spec.identity, &refs)?
        }
        _ => Authority::new(&refs)?,
    };
    ctx.rig.commit_live(&mut authority)?;
    Ok(authority)
}

fn identity(ctx: &Ctx<'_>) -> String {
    match ctx.kind {
        Kind::BaseModel => BASE_MODEL_IDENTITY.to_string(),
        Kind::Dependency => ctx.rig.identity(),
        Kind::Segment(spec) => spec.identity.to_string(),
    }
}

/// The committed event: the ledger (and so the state the oracle reads) learns of it.
fn ledger_event(ctx: &Ctx<'_>, authority: &mut Authority) -> Res<()> {
    let runtime = &mut authority.runtime;
    match (ctx.kind, ctx.ty) {
        (Kind::BaseModel, Invalidation::Revocation) => {
            commit_revocation(runtime, BASE_MODEL_IDENTITY, BASE_MODEL_GENERATION)
        }
        (Kind::BaseModel, Invalidation::Supersession) => commit_supersession(
            runtime,
            BASE_MODEL_IDENTITY,
            BASE_MODEL_GENERATION,
            BASE_MODEL_GENERATION + 1,
        ),
        (Kind::Dependency, ty) => ctx.rig.commit_event(authority, ty),
        (Kind::Segment(spec), Invalidation::Revocation) => {
            commit_revocation(runtime, spec.identity, 1)
        }
        (Kind::Segment(spec), Invalidation::Supersession) if spec.kind == SegmentKind::Semantic => {
            write_semantic(runtime, spec.identity, "v2")
        }
        (Kind::Segment(spec), Invalidation::Supersession) => {
            commit_supersession(runtime, spec.identity, 1, 2)
        }
        (_, Invalidation::Fence) => Err("only an expert has a fence".to_string()),
    }
}

/// The generation the dependency's supersession moves to.
fn superseded_generation(ctx: &Ctx<'_>) -> u64 {
    match ctx.kind {
        Kind::Dependency => ctx.rig.superseded_generation(),
        Kind::BaseModel => BASE_MODEL_GENERATION + 1,
        Kind::Segment(_) => 2,
    }
}

/// Commit the event to the ledger and hand the committed events to the backend's catalog hook
/// (`apply_committed_event`, the host's equivalent of the runtime's hook), unless the fault
/// hook suppresses the index update. A semantic edit commits a delta the hook cannot read, so
/// its supersession floor is applied by identity. Returns what the catalog retired.
fn apply_event(
    ctx: &Ctx<'_>,
    authority: &mut Authority,
    backend: &ReferenceKvReuseBackend,
) -> Res<usize> {
    let mark = authority.mark();
    ledger_event(ctx, authority)?;
    let mut retired = lab::apply_events(backend, &authority.events_since(mark))?;
    if matches!(ctx.kind, Kind::Segment(spec) if spec.kind == SegmentKind::Semantic)
        && ctx.ty == Invalidation::Supersession
    {
        retired += backend
            .supersede_inputs(&BTreeMap::from([(
                identity(ctx),
                Generation(superseded_generation(ctx)),
            )]))
            .map_err(fail("superseding inputs"))?;
    }
    Ok(retired)
}

fn entries(backend: &ReferenceKvReuseBackend) -> Res<usize> {
    Ok(backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries)
}

/// The entries one invalidation drops of `layers * chunks`.
fn expected_dropped(ctx: &Ctx<'_>, layers: usize) -> usize {
    match ctx.kind {
        Kind::Dependency => ctx.rig.expected_dropped(layers, WARM_CHUNKS),
        _ => layers * WARM_CHUNKS,
    }
}

// ---- probes -----------------------------------------------------------------------------

/// The post-event probes of one cell.
#[derive(Debug, Default)]
struct Probed {
    /// Probes that began and looked up (segment cells): one lookup per layer and boundary.
    lookups: i64,
    /// Entries served (a hit, or a hit the oracle then contradicted).
    hits: i64,
    /// Probes the binding check refused before any lookup.
    refused: i64,
    /// Probes that were not refused although a dependency of their binding is dead.
    begun_on_dead_binding: i64,
}

fn probe_entries(
    tally: &mut Tally,
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    binding: &StackBinding,
    label: &str,
) -> Res<Probed> {
    let spec = ctx.kind.spec();
    let layers = backend.stack_config().num_layers as i64;
    let mut probed = Probed::default();
    let served_before = backend
        .session_stats()
        .map_err(fail("session stats"))?
        .stale_hits;
    for boundary in PROBE_BOUNDARIES {
        let request = probe_request(spec, &format!("{label}-probe-{boundary}"), boundary, 1);
        let begun = backend.begin_session_bound(&lab::scope(), &request, None, binding, authority);
        if refused_not_live(&begun) {
            probed.refused += 1;
            continue;
        }
        let mut session = begun.map_err(fail("a probe session"))?;
        probed.lookups += layers;
        if session.hit_tokens() > 0 {
            probed.hits += 1;
        }
        if !ctx.is_segment() {
            // A binding whose dependency is dead began: that is not a refusal.
            probed.begun_on_dead_binding += 1;
        }
        // The probe then runs to its end, so the same request is also compared with a cold
        // run: a request on a dead input still computes the right answer, uncached.
        let finish = session.run_to_end(authority).map_err(fail("a probe run"))?;
        let ptr_runtime::KvSessionFinish::Completed(outcome) = finish else {
            return Err("a probe session was cancelled".to_string());
        };
        let cold = lab::cold(backend, &request, binding)?;
        compare(tally, &outcome, &cold);
    }
    let served_after = backend
        .session_stats()
        .map_err(fail("session stats"))?
        .stale_hits;
    probed.hits += (served_after - served_before) as i64;
    Ok(probed)
}

/// The same tokens citing another generation: `same_tokens_different_generation`. Run while
/// the generation-1 entries are live, it must not be served from them (the generation is part
/// of the lineage).
fn same_tokens_other_generation(
    tally: &mut Tally,
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    binding: &StackBinding,
    label: &str,
) -> Res<i64> {
    let spec = ctx.kind.spec();
    let request = probe_request(spec, &format!("{label}-gen2"), 32, 2);
    let session = backend
        .begin_session_bound(&lab::scope(), &request, None, binding, authority)
        .map_err(fail("the other-generation probe"))?;
    let hits = i64::from(session.hit_tokens() > 0);
    drop(session);
    tally.add("same_tokens_different_generation_cases", 1);
    Ok(hits)
}

fn procedure(
    tally: &mut Tally,
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    binding: &StackBinding,
) -> Json {
    let observed = observe(backend, binding);
    let mismatches = ctx.rig.procedure_mismatches(&observed);
    tally.add("procedure_mismatches", mismatches.len() as i64);
    let json = ctx.rig.procedure_json(&observed);
    match json {
        Json::Obj(mut pairs) => {
            pairs.push((
                "mismatches".to_string(),
                Json::Arr(
                    mismatches
                        .iter()
                        .map(|text| Json::str(text.clone()))
                        .collect(),
                ),
            ));
            Json::Obj(pairs)
        }
        other => other,
    }
}

/// Run the warm request on `backend` and require that it cached every layer at both
/// boundaries.
fn warm(
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    binding: &StackBinding,
) -> Res<usize> {
    let layers = backend.stack_config().num_layers;
    let outcome = lab::run_completed(
        backend,
        &warm_request(ctx.kind.spec(), &format!("{}-warm", ctx.cell)),
        binding,
        authority,
    )?;
    let inserted = outcome
        .layered
        .as_ref()
        .map_or(0, |report| report.inserted_layer_entries);
    if outcome.inserted_chunks != WARM_CHUNKS || inserted != layers * WARM_CHUNKS {
        return Err(format!(
            "{}: the warm run cached {} chunks and {inserted} layer entries, not {WARM_CHUNKS} and {}",
            ctx.cell,
            outcome.inserted_chunks,
            layers * WARM_CHUNKS
        ));
    }
    entries(backend)
}

/// A control probe on a live catalog: it must hit, so a miss after the event is the event's
/// doing and not an absent page.
fn live_control(
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    binding: &StackBinding,
    label: &str,
) -> Res<usize> {
    let request = probe_request(ctx.kind.spec(), &format!("{label}-control"), 32, 1);
    let session = backend
        .begin_session_bound(&lab::scope(), &request, None, binding, authority)
        .map_err(fail("the live control"))?;
    Ok(session.hit_tokens())
}

fn note_revocation_probes(tally: &mut Tally, ctx: &Ctx<'_>, probed: &Probed) {
    if ctx.is_segment() && ctx.ty == Invalidation::Revocation {
        tally.add("post_revocation_lookups_matching_tokens", probed.lookups);
        tally.add("revoked_lineage_entry_hits", probed.hits);
    }
}

fn note_event(tally: &mut Tally, ctx: &Ctx<'_>) {
    if ctx.is_segment() && ctx.ty == Invalidation::Revocation {
        tally.add("input_revocations_performed", 1);
    }
}

/// The stale hits of a probe set: entries served, plus a dead binding that was begun.
fn stale_of(probed: &Probed) -> i64 {
    probed.hits + probed.begun_on_dead_binding
}

// ---- the four paths ---------------------------------------------------------------------

struct PathResult {
    stale: i64,
    inflight: Option<Inflight>,
    detail: Json,
}

fn normal_arm(
    tally: &mut Tally,
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &mut Authority,
    binding: &StackBinding,
) -> Res<PathResult> {
    let layers = backend.stack_config().num_layers;
    let before = warm(ctx, backend, authority, binding)?;
    let control_hit = live_control(ctx, backend, authority, binding, &ctx.cell)?;
    if control_hit != 32 {
        return Err(format!(
            "{}: the live control hit {control_hit} tokens, not 32",
            ctx.cell
        ));
    }
    let mut extra_stale = 0i64;
    if ctx.is_segment() {
        extra_stale +=
            same_tokens_other_generation(tally, ctx, backend, authority, binding, &ctx.cell)?;
    }
    note_event(tally, ctx);
    apply_event(ctx, authority, backend)?;
    let after = entries(backend)?;
    let dropped = before - after;
    let expected = expected_dropped(ctx, layers);
    if dropped != expected {
        tally.add("matrix_selectivity_mismatches", 1);
    }
    tally.add(
        &format!("matrix_entries_dropped.{}", ctx.cell),
        dropped as i64,
    );
    let probed = probe_entries(tally, ctx, backend, authority, binding, &ctx.cell)?;
    note_revocation_probes(tally, ctx, &probed);
    tally.add("normal_probe_lookups", probed.lookups);
    tally.add("binding_probes_refused_at_begin", probed.refused);
    // Entries of the invalidated lineage that a request admitted again after the event.
    let after_probes = entries(backend)?;
    let reinserted = after_probes.saturating_sub(after) as i64;
    tally.add(
        &format!("stale_lineage_entries_reinserted.{}", ctx.cell),
        reinserted,
    );
    if !ctx.is_segment() {
        // The re-resolved binding computes the right answer on what survived (a binding-carried
        // dependency: the live successor, the base model, the masked router).
        let resolved = match ctx.kind {
            Kind::Dependency => Some(ctx.rig.resolved_binding(ctx.ty, authority)),
            _ => None,
        };
        if let Some(resolved) = resolved {
            let request = probe_request(ctx.kind.spec(), &format!("{}-resolved", ctx.cell), 32, 1);
            if let Ran::Done(outcome) = run_resolved(backend, authority, &request, &resolved)? {
                let cold = lab::cold(backend, &request, &resolved)?;
                compare(tally, &outcome, &cold);
                tally.add("resolved_binding_runs", 1);
            }
        }
    }
    Ok(PathResult {
        stale: stale_of(&probed) + extra_stale,
        inflight: None,
        detail: Json::obj([
            ("entries_before", Json::Int(before as i64)),
            ("entries_after_event", Json::Int(after as i64)),
            ("entries_dropped", Json::Int(dropped as i64)),
            ("entries_expected_dropped", Json::Int(expected as i64)),
            ("probe_lookups", Json::Int(probed.lookups)),
            ("probes_refused_at_begin", Json::Int(probed.refused)),
            ("probe_hits", Json::Int(probed.hits)),
            ("entries_reinserted_after_probes", Json::Int(reinserted)),
            ("control_hit_tokens", Json::Int(control_hit as i64)),
        ]),
    })
}

enum Ran {
    Done(Box<ptr_runtime::ReferenceKvOutcome>),
    Refused,
}

/// Run `request` on the re-resolved `binding` under `authority`; a binding whose dependency is
/// not live (a base-model event) is refused.
fn run_resolved(
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    request: &SegmentedModelRequest,
    binding: &StackBinding,
) -> Res<Ran> {
    let begun = backend.begin_session_bound(&lab::scope(), request, None, binding, authority);
    if refused_not_live(&begun) {
        return Ok(Ran::Refused);
    }
    let mut session = begun.map_err(fail("the resolved binding"))?;
    match session
        .run_to_end(authority)
        .map_err(fail("the resolved run"))?
    {
        ptr_runtime::KvSessionFinish::Completed(outcome) => Ok(Ran::Done(outcome)),
        ptr_runtime::KvSessionFinish::Cancelled(_) => Ok(Ran::Refused),
    }
}

fn lag_arm(
    tally: &mut Tally,
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &mut Authority,
    binding: &StackBinding,
) -> Res<PathResult> {
    let before = warm(ctx, backend, authority, binding)?;
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .map_err(fail("injecting the dropped invalidation"))?;
    note_event(tally, ctx);
    let retired = apply_event(ctx, authority, backend)?;
    if retired != 0 {
        return Err(format!(
            "{}: the suppressed invalidation retired {retired} entries",
            ctx.cell
        ));
    }
    let still = entries(backend)?;
    if still != before {
        return Err(format!(
            "{}: the stale index changed ({before} -> {still})",
            ctx.cell
        ));
    }
    let probed = probe_entries(tally, ctx, backend, authority, binding, &ctx.cell)?;
    note_revocation_probes(tally, ctx, &probed);
    tally.add("binding_probes_refused_at_begin", probed.refused);
    if ctx.is_segment() {
        tally.add("invalidation_suppressed_arms", 1);
        tally.add("stale_eligible_entries_before_event", before as i64);
        tally.add(
            "post_event_lookups_on_stale_eligible_entries",
            probed.lookups,
        );
        tally.add("post_event_lookups", probed.lookups);
        tally.add("stale_hits", stale_of(&probed));
    } else {
        tally.add("binding_stale_hits_lag", stale_of(&probed));
        tally.add("binding_stale_eligible_entries_lag", before as i64);
    }
    let after = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add("lag_use_time_rejections", after.use_time_rejections as i64);
    Ok(PathResult {
        stale: stale_of(&probed),
        inflight: None,
        detail: Json::obj([
            ("stale_eligible_entries", Json::Int(before as i64)),
            ("probe_lookups", Json::Int(probed.lookups)),
            ("probes_refused_at_begin", Json::Int(probed.refused)),
            ("probe_hits", Json::Int(probed.hits)),
            (
                "use_time_rejections",
                Json::Int(after.use_time_rejections as i64),
            ),
        ]),
    })
}

/// Run the tokens of `warm` as one raw segment under `binding`: this publishes the same
/// content-addressed layer pages a bound run of them publishes (under another lineage, so the
/// restored entries are neither used nor replaced).
fn materialise_pages(
    backend: &ReferenceKvReuseBackend,
    oracle: &Authority,
    warm: &SegmentedModelRequest,
    binding: &StackBinding,
    id: &str,
) -> Res<usize> {
    let tokens: Vec<u32> = warm.tokens().iter().map(|token| token.0).collect();
    let outcome = lab::run_completed(backend, &corpus::raw_request(id, &tokens), binding, oracle)?;
    Ok(outcome
        .layered
        .as_ref()
        .map_or(0, |report| report.inserted_layer_entries))
}

fn restore_arm(
    tally: &mut Tally,
    ctx: &Ctx<'_>,
    backend: &ReferenceKvReuseBackend,
    authority: &mut Authority,
    binding: &StackBinding,
) -> Res<PathResult> {
    let warm_req = warm_request(ctx.kind.spec(), &format!("{}-warm2", ctx.cell));
    let pre_entries = warm(ctx, backend, authority, binding)?;
    let snapshot = backend
        .export_catalog()
        .map_err(fail("exporting the catalog"))?;
    // The restart: a fresh backend takes the pre-event snapshot and is never told of the
    // event; only the ledger knows.
    let restored = lab::backend()?;
    let report: KvRestoreReport = restored
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the catalog"))?;
    let materialised = materialise_pages(
        &restored,
        authority,
        &warm_req,
        binding,
        &format!("{}-pages", ctx.cell),
    )?;
    note_event(tally, ctx);
    ledger_event(ctx, authority)?;
    // Observed in the catalog after the event and before the post-event lookups.
    let observed = restored.catalog_stats().map_err(fail("catalog stats"))?;
    if observed.restored_entries as usize != report.entries
        || observed.entries != report.entries + materialised
        || report.entries != pre_entries
    {
        return Err(format!(
            "{}: the restored entries were not observed in the catalog",
            ctx.cell
        ));
    }
    let probed = probe_entries(tally, ctx, &restored, authority, binding, &ctx.cell)?;
    note_revocation_probes(tally, ctx, &probed);
    tally.add("binding_probes_refused_at_begin", probed.refused);
    // Control: the same snapshot, pages recomputed, probed while everything is live, forks:
    // the misses above are the oracle's refusal, not an absent page.
    let control = lab::backend()?;
    control
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the control"))?;
    let live = authority_live(ctx)?;
    materialise_pages(
        &control,
        &live,
        &warm_req,
        binding,
        &format!("{}-control-pages", ctx.cell),
    )?;
    let control_hit = live_control(ctx, &control, &live, binding, &ctx.cell)?;
    if control_hit != 32 {
        return Err(format!(
            "{}: a live restored entry with its pages did not fork ({control_hit} tokens; {:?})",
            ctx.cell,
            control.catalog_stats()
        ));
    }
    tally.add("restore_control_hit_tokens", control_hit as i64);
    lab::note_scope(tally, &control)?;
    lab::note_scope(tally, &restored)?;
    let rejections = restored.catalog_stats().map_err(fail("catalog stats"))?;
    if ctx.is_segment() {
        tally.add("restore_arms", 1);
        tally.add("restored_pre_revocation_entries", report.entries as i64);
        tally.add("stale_eligible_entries_before_event", report.entries as i64);
        tally.add(
            "post_event_lookups_on_stale_eligible_entries",
            probed.lookups,
        );
        tally.add("post_event_lookups", probed.lookups);
        tally.add("stale_hits", stale_of(&probed));
    } else {
        tally.add("binding_stale_hits_restore", stale_of(&probed));
        tally.add("binding_restored_entries", report.entries as i64);
    }
    tally.add(
        "restore_use_time_rejections",
        rejections.use_time_rejections as i64,
    );
    Ok(PathResult {
        stale: stale_of(&probed),
        inflight: None,
        detail: Json::obj([
            ("restored_entries", Json::Int(report.entries as i64)),
            ("materialised_entries", Json::Int(materialised as i64)),
            ("probe_lookups", Json::Int(probed.lookups)),
            ("probes_refused_at_begin", Json::Int(probed.refused)),
            ("probe_hits", Json::Int(probed.hits)),
            ("control_hit_tokens", Json::Int(control_hit as i64)),
            (
                "use_time_rejections",
                Json::Int(rejections.use_time_rejections as i64),
            ),
        ]),
    })
}

/// An authority for which everything the cell uses is live (the control's oracle).
fn authority_live(ctx: &Ctx<'_>) -> Res<Authority> {
    authority(ctx)
}

fn inflight_arm(
    tally: &mut Tally,
    ids: &mut Ids,
    ctx: &Ctx<'_>,
    seed: u64,
    backend: &ReferenceKvReuseBackend,
    authority: &mut Authority,
    binding: &StackBinding,
) -> Res<PathResult> {
    let spec = ctx.kind.spec();
    let layers = backend.stack_config().num_layers;
    warm(ctx, backend, authority, binding)?;
    // A request that forks the cached prefix (hit 32) and then crosses a chunk boundary of
    // its own (tail 24: boundary 48), so inserts are in reach of the race too.
    let id = format!("{}-fork", ctx.cell);
    let request = lineage_request(spec, &id, 1, tail(spec, 11, 24));
    let max_new = backend.limits().max_new_tokens;
    let total_steps = Steps::count(request.total_tokens(), 32, max_new);
    let race_at = corpus::scenario(seed, &ctx.cell).range(0, total_steps as u64) as usize;
    let session = backend
        .begin_session_bound(&lab::scope(), &request, None, binding, authority)
        .map_err(fail("the forking session"))?;
    if session.hit_tokens() != 32 {
        return Err(format!(
            "{}: the session did not fork the cached prefix ({} tokens)",
            ctx.cell,
            session.hit_tokens()
        ));
    }
    let mut steps = Steps { session };
    for _ in 0..race_at {
        if steps.step(authority)? {
            return Err(format!("{}: cancelled before the event", ctx.cell));
        }
    }
    let tokens_at_event = steps.session.generated().len();
    note_event(tally, ctx);
    apply_event(ctx, authority, backend)?;
    let entries_after_event = entries(backend)?;
    let dropped = {
        let total = layers * WARM_CHUNKS;
        total.saturating_sub(entries_after_event.min(total))
    };
    // The next step is the first use after the event.
    let cancelled = if race_at < total_steps {
        steps.step(authority)?
    } else {
        matches!(
            steps.session.finish(authority).map_err(fail("finishing"))?,
            ptr_runtime::KvSessionFinish::Cancelled(_)
        )
    };
    let mut escaped = 0i64;
    if !cancelled {
        // The fork survived the event: whatever it produced from here is an escape.
        escaped += 1;
    }
    escaped += (steps.session.generated().len() - tokens_at_event) as i64;
    let mut restarted_fresh = None;
    if cancelled {
        match ctx.kind {
            Kind::BaseModel => {
                // No live base model exists to re-resolve to, and a restart would compute
                // from the revoked one: the cell checks cancellation and silence only.
            }
            _ => {
                let restarted = steps
                    .session
                    .restart_with(&*authority)
                    .map_err(fail("the restart"))?;
                let reference = match ctx.kind {
                    Kind::Dependency => ctx.rig.resolved_binding(ctx.ty, authority),
                    _ => binding.clone(),
                };
                let cold = lab::cold(backend, &request, &reference)?;
                let equal = compare(tally, &restarted, &cold);
                let fresh = restarted.hit_tokens == 0
                    && restarted.prefilled_tokens == request.total_tokens()
                    && restarted.inserted_chunks == 0
                    && restarted.restarted_after_cancel;
                restarted_fresh = Some(equal && fresh);
                if matches!(ctx.kind, Kind::Dependency)
                    && !ctx.rig.resolution_holds(&restarted.binding, ctx.ty)
                {
                    tally.add("restart_resolution_mismatches", 1);
                }
            }
        }
    }
    let after_restart = entries(backend)?;
    let inserted_after = after_restart.saturating_sub(entries_after_event) as i64;
    // A later request citing the dead dependency must not fork anything either.
    let again = probe_request(spec, &format!("{}-after", ctx.cell), 32, 1);
    let late = backend.begin_session_bound(&lab::scope(), &again, None, binding, authority);
    let late_hit = match &late {
        Ok(session) => i64::from(session.hit_tokens() > 0),
        Err(_) if refused_not_live(&late) => 0,
        Err(error) => return Err(format!("{}: the late session: {error:?}", ctx.cell)),
    };
    drop(late);
    ids.absorb(backend)?;
    tally.add("race_outputs_after_event", escaped);
    tally.add("cache_inserts_after_event", inserted_after);
    tally.add(&format!("race_point.{}", ctx.cell), race_at as i64);
    tally.add("race_points_total", total_steps as i64);
    Ok(PathResult {
        stale: escaped + late_hit + inserted_after,
        inflight: Some(Inflight {
            cancelled,
            restarted_fresh,
            escaped,
        }),
        detail: Json::obj([
            ("race_at_step", Json::Int(race_at as i64)),
            ("steps", Json::Int(total_steps as i64)),
            ("cancelled", Json::Bool(cancelled)),
            ("escaped_outputs", Json::Int(escaped)),
            (
                "entries_dropped_before_next_step",
                Json::Int(dropped as i64),
            ),
            ("late_probe_hit", Json::Int(late_hit)),
            ("entries_inserted_after_event", Json::Int(inserted_after)),
            (
                "restarted_from_fresh_prefill",
                restarted_fresh.map_or(Json::Null, Json::Bool),
            ),
        ]),
    })
}

/// Run one cell on a fresh backend and authority.
pub fn run_cell(
    tally: &mut Tally,
    ids: &mut Ids,
    infos: &mut Vec<CellInfo>,
    rig: &dyn Rig,
    seed: u64,
    kind: Kind,
    ty: Invalidation,
    path: Path,
) -> Res<()> {
    let kind_name = match kind {
        Kind::BaseModel => "base_model",
        Kind::Dependency => rig.dependency_kind(),
        Kind::Segment(spec) => spec.cell_kind,
    };
    let ctx = Ctx {
        rig,
        kind,
        ty,
        cell: cell_name(kind_name, path.name(), ty.name()),
        kind_name,
    };
    let backend = lab::backend()?;
    let mut authority = authority(&ctx)?;
    let binding = rig.binding(&authority);
    let procedure = procedure(tally, &ctx, &backend, &binding);
    let result = match path {
        Path::Normal => normal_arm(tally, &ctx, &backend, &mut authority, &binding)?,
        Path::Lag => lag_arm(tally, &ctx, &backend, &mut authority, &binding)?,
        Path::Restore => restore_arm(tally, &ctx, &backend, &mut authority, &binding)?,
        Path::Inflight => inflight_arm(tally, ids, &ctx, seed, &backend, &mut authority, &binding)?,
    };
    ids.absorb(&backend)?;
    lab::note_scope(tally, &backend)?;
    tally.add(&format!("invalidation_matrix_cases.{}", ctx.cell), 1);
    tally.add(
        &format!("invalidation_matrix_stale_hits.{}", ctx.cell),
        result.stale,
    );
    rig.note_cell(
        tally,
        &CellRecord {
            kind: ctx.kind_name,
            is_dependency: matches!(kind, Kind::Dependency),
            is_semantic_input: matches!(kind, Kind::Segment(spec) if spec.kind == SegmentKind::Semantic),
            ty,
            inflight: result.inflight,
        },
    );
    infos.push(CellInfo {
        cell: ctx.cell.clone(),
        evidence: if ctx.is_segment() {
            "session_lookup"
        } else {
            "session_refused_at_begin"
        },
        oracle: match kind {
            Kind::Dependency => "ptr-runtime ledger and expert registry (the experiment's own dependency events)",
            Kind::BaseModel => "ptr-runtime ledger (CapsuleCommitted / Revoked / CapsuleSuperseded)",
            Kind::Segment(spec) if spec.kind == SegmentKind::Semantic => {
                "ptr-runtime committed semantic state (host write; Revoked for a revocation, a committed edit for a supersession)"
            }
            Kind::Segment(_) => "ptr-runtime ledger (CapsuleCommitted / Revoked / CapsuleSuperseded)",
        },
        procedure,
        detail: result.detail,
    });
    Ok(())
}
