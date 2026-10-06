//! The invalidation coverage matrix of R005: every dependency kind x invalidation type x path
//! (52 cells), each run at least once per seed and reported per cell.
//!
//! Two layers carry a cell, and each cell records which one:
//!
//! * `session`: the three generation-bearing segment kinds (semantic, memory, derived) run on
//!   the stack backend with real sessions, under a `PtrRuntime` whose committed ledger is the
//!   liveness oracle. A stale hit and an output that escapes after the event are both
//!   observed.
//! * `catalog`: the base-model, adapter and expert dependencies are not carried by the stack
//!   backend (`ReferenceKvReuseBackend` records only the request's segments), so they run on
//!   a bare [`KvPrefixCatalog`] over the reference paged KV backend. A stale hit, a still-live
//!   pin after the event and an accepted insert after the event are observed; an *output* is
//!   not, because no inference step runs on that layer (see `unsupported`).

use super::lab::{
    self, commit_revocation, commit_supersession, compare, fail, runtime_with, scope,
    semantic_runtime, write_semantic, CatLab, Res, Scripted, Steps,
};
use super::metrics::{Json, Tally};
use super::params;
use super::workload::{self, Part};
use ptr_model_api::{SegmentKind, SegmentedModelRequest};
use ptr_runtime::{
    CatalogLimits, IndexUpdate, KvDependency, KvDependencyKind, KvLivenessOracle, KvPrefixCatalog,
    KvRestoreReport, PtrRuntime, ReferenceKvFaults, ReferenceKvReuseBackend,
};
use ptr_types::{Generation, PodId, ProjectId};
use std::collections::{BTreeMap, BTreeSet};

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
    Fence,
}

impl Invalidation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
            Self::Fence => "fence",
        }
    }
}

/// How a segment input lives in the runtime whose state is the liveness oracle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Model {
    /// As the runtime really holds it: a semantic input is a key of the committed semantic
    /// state with no lifecycle generation (its supersession is a committed edit), and a
    /// memory or derived input is a lifecycle target (a committed capsule, superseded by a
    /// committed successor).
    Committed,
    /// Every input is a lifecycle target. For a semantic key this is not how the runtime
    /// holds it; it runs as a control that separates "the use-time check cannot see a
    /// supersession" from "the check is broken".
    Lifecycle,
}

impl Model {
    fn semantic_state(self, spec: &SegSpec) -> bool {
        self == Self::Committed && spec.kind == SegmentKind::Semantic
    }
}

/// A generation-bearing prompt segment kind of the session layer.
#[derive(Clone, Copy, Debug)]
pub struct SegSpec {
    pub kind: SegmentKind,
    /// The matrix name of the kind (`semantic_input`).
    pub cell_kind: &'static str,
    /// The short name `superseded_lookup_cells` uses (`semantic`).
    pub short: &'static str,
    pub identity: &'static str,
    /// Block labels of the two bound segments (and the tail) of this kind's requests.
    pub label: u64,
}

pub const SEGMENT_KINDS: [SegSpec; 3] = [
    SegSpec {
        kind: SegmentKind::Semantic,
        cell_kind: "semantic_input",
        short: "semantic",
        identity: "doc:r005",
        label: 100,
    },
    SegSpec {
        kind: SegmentKind::Memory,
        cell_kind: "memory_input",
        short: "memory",
        identity: "mem:r005",
        label: 200,
    },
    SegSpec {
        kind: SegmentKind::Derived,
        cell_kind: "derived_input",
        short: "derived",
        identity: "ev:r005",
        label: 300,
    },
];

/// One cell as the result describes it.
#[derive(Clone, Debug)]
pub struct CellInfo {
    pub cell: String,
    pub evidence: &'static str,
    pub oracle: &'static str,
    pub procedure: Json,
}

impl CellInfo {
    pub fn json(&self) -> Json {
        Json::obj([
            ("cell", Json::str(self.cell.clone())),
            ("evidence", Json::str(self.evidence)),
            ("oracle", Json::str(self.oracle)),
            ("procedure", self.procedure.clone()),
        ])
    }
}

/// The request identities the backend's own bookkeeping saw cancelled and restarted.
#[derive(Debug, Default)]
pub struct Ids {
    pub cancelled: BTreeSet<String>,
    pub restarted: BTreeSet<String>,
    pub refused: BTreeSet<String>,
    pub completed_after_refusal: BTreeSet<String>,
}

impl Ids {
    pub fn absorb_sessions(&mut self, backend: &ReferenceKvReuseBackend) -> Res<()> {
        let stats = backend.session_stats().map_err(fail("session stats"))?;
        self.cancelled
            .extend(stats.cancelled_requests.iter().map(|id| id.0.clone()));
        self.restarted
            .extend(stats.restarted_requests.iter().map(|id| id.0.clone()));
        Ok(())
    }
}

/// The limits a backend or catalog reports it runs under, against the workload's pinned ones;
/// a difference is a procedure mismatch.
pub fn procedure(tally: &mut Tally, workload: params::Workload, observed: CatalogLimits) -> Json {
    let pinned = workload.limits();
    let matches = observed.max_trie_nodes as u64 == pinned.max_trie_nodes
        && observed.max_kv_page_bytes == pinned.max_kv_page_bytes
        && observed.max_metadata_bytes == pinned.max_metadata_bytes;
    if !matches {
        tally.add("procedure_mismatches", 1);
    }
    Json::obj([
        ("workload", Json::str(workload.name())),
        ("max_trie_nodes", Json::Int(observed.max_trie_nodes as i64)),
        (
            "max_kv_page_bytes",
            Json::Int(observed.max_kv_page_bytes as i64),
        ),
        (
            "max_metadata_bytes",
            Json::Int(observed.max_metadata_bytes as i64),
        ),
        ("matches_pinned", Json::Bool(matches)),
    ])
}

fn record_cell(tally: &mut Tally, cell: &str, stale_hits: i64) {
    tally.add(&format!("invalidation_matrix_cases.{cell}"), 1);
    tally.add(
        &format!("invalidation_matrix_stale_hits.{cell}"),
        stale_hits,
    );
}

fn member(spec: &SegSpec, ty: Invalidation) -> Option<&'static str> {
    match (spec.kind, ty) {
        (SegmentKind::Semantic, Invalidation::Revocation) => Some("semantic_revocation"),
        (SegmentKind::Semantic, Invalidation::Supersession) => Some("semantic_supersession"),
        (SegmentKind::Memory, Invalidation::Revocation) => Some("memory_revocation"),
        (SegmentKind::Memory, Invalidation::Supersession) => Some("memory_supersession"),
        _ => None,
    }
}

// ---- the session layer ------------------------------------------------------------------

/// `[segment A][segment B][raw tail]`: two bound 16-token segments of one identity, so the
/// entry at boundary 16 and the entry at boundary 32 each have a lineage of their own.
fn lineage_request(
    spec: &SegSpec,
    id: &str,
    generation: u64,
    tail: Vec<u32>,
) -> SegmentedModelRequest {
    workload::request(
        id,
        &[
            Part::bound(
                spec.kind,
                spec.identity,
                generation,
                workload::block(spec.label),
            ),
            Part::bound(
                spec.kind,
                spec.identity,
                generation,
                workload::block(spec.label + 1),
            ),
            Part::raw(tail),
        ],
    )
}

/// The request that warms the catalog: both bound segments and a four-token tail, so the
/// boundaries 16 and 32 are cached.
fn warm_request(spec: &SegSpec, id: &str) -> SegmentedModelRequest {
    lineage_request(spec, id, 1, workload::sequence(spec.label + 7, 4))
}

/// One probe per stale-eligible entry: a request that can hit exactly the entry at
/// `boundary` (16 or 32) and no deeper one, citing the stale generation 1.
fn probe_request(spec: &SegSpec, id: &str, boundary: usize) -> SegmentedModelRequest {
    let tail = workload::sequence(spec.label + 9, 3);
    match boundary {
        16 => workload::request(
            id,
            &[
                Part::bound(spec.kind, spec.identity, 1, workload::block(spec.label)),
                Part::raw(tail),
            ],
        ),
        _ => lineage_request(spec, id, 1, tail),
    }
}

/// A runtime holding the input of `spec` as `model` says.
fn kind_runtime(spec: &SegSpec, model: Model) -> Res<PtrRuntime> {
    if model.semantic_state(spec) {
        semantic_runtime(spec.identity)
    } else {
        runtime_with(&[spec.identity])
    }
}

/// Commit the event to the ledger (and so to the committed state the oracle reads).
fn ledger_event(
    runtime: &mut PtrRuntime,
    spec: &SegSpec,
    ty: Invalidation,
    model: Model,
) -> Res<()> {
    match ty {
        Invalidation::Revocation => commit_revocation(runtime, spec.identity, 1),
        Invalidation::Supersession if model.semantic_state(spec) => {
            write_semantic(runtime, spec.identity, "v2")
        }
        Invalidation::Supersession => commit_supersession(runtime, spec.identity, 1, 2),
        Invalidation::Fence => Err("a segment has no fence".to_string()),
    }
}

/// Apply the event to the ledger and, unless suppressed, to the catalog of `backend`.
fn apply_event(
    runtime: &mut PtrRuntime,
    backend: &ReferenceKvReuseBackend,
    spec: &SegSpec,
    ty: Invalidation,
    model: Model,
) -> Res<usize> {
    ledger_event(runtime, spec, ty, model)?;
    let identity = spec.identity.to_string();
    match ty {
        Invalidation::Revocation => backend
            .invalidate_inputs(&BTreeSet::from([(identity, Generation(1))]))
            .map_err(fail("invalidating inputs")),
        Invalidation::Supersession => backend
            .supersede_inputs(&BTreeMap::from([(identity, Generation(2))]))
            .map_err(fail("superseding inputs")),
        Invalidation::Fence => Err("a segment has no fence".to_string()),
    }
}

/// Probe each stale-eligible entry once and return how many probes hit (a stale hit).
struct Probed {
    lookups: i64,
    hits: i64,
}

fn probe_entries(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    runtime: &PtrRuntime,
    spec: &SegSpec,
    label: &str,
) -> Res<Probed> {
    let mut probed = Probed {
        lookups: 0,
        hits: 0,
    };
    let served_before = backend
        .session_stats()
        .map_err(fail("session stats"))?
        .stale_hits;
    for boundary in [16usize, 32] {
        let request = probe_request(spec, &format!("{label}-probe-{boundary}"), boundary);
        let mut session = backend
            .begin_session(&scope(), &request, None, runtime)
            .map_err(fail("a probe session"))?;
        probed.lookups += 1;
        if session.hit_tokens() > 0 {
            probed.hits += 1;
        }
        // The probe then runs to its end, so the same request is also compared with a cold
        // run: a request on a dead input still computes the right answer, uncached.
        let finish = session.run_to_end(runtime).map_err(fail("a probe run"))?;
        let ptr_runtime::KvSessionFinish::Completed(outcome) = finish else {
            return Err("a probe session was cancelled".to_string());
        };
        let cold = lab::cold(backend, &scope(), &request)?;
        compare(tally, &outcome, &cold);
    }
    // A hit the catalog served and the oracle then contradicted is dropped by the session,
    // never used, but it is still a hit the catalog should not have served.
    let served_after = backend
        .session_stats()
        .map_err(fail("session stats"))?
        .stale_hits;
    probed.hits += (served_after - served_before) as i64;
    Ok(probed)
}

/// Run the tokens of `warm` as one raw segment on `backend`: this publishes the same
/// content-addressed pages a bound run of them publishes.
fn materialise_pages(
    backend: &ReferenceKvReuseBackend,
    oracle: &dyn KvLivenessOracle,
    warm: &SegmentedModelRequest,
    id: &str,
) -> Res<()> {
    let tokens: Vec<u32> = warm.tokens().iter().map(|token| token.0).collect();
    lab::run_completed(
        backend,
        &scope(),
        &workload::raw_request(id, &tokens),
        oracle,
    )?;
    Ok(())
}

/// Run one cell of a generation-bearing segment kind on the stack backend.
pub fn segment_cell(
    tally: &mut Tally,
    ids: &mut Ids,
    infos: &mut Vec<CellInfo>,
    seed: u64,
    spec: &SegSpec,
    ty: Invalidation,
    path: Path,
    model: Model,
) -> Res<()> {
    let cell = params::cell_name(spec.cell_kind, path.name(), ty.name());
    let mut runtime = kind_runtime(spec, model)?;
    let backend = lab::main_backend()?;
    let procedure = procedure(tally, params::Workload::Main, backend.limits().catalog);
    let warm = warm_request(spec, &format!("{cell}-warm"));
    let (hits, extra_stale): (i64, i64) = match path {
        Path::Normal => normal_arm(
            tally,
            &backend,
            &mut runtime,
            (spec, ty, model),
            &cell,
            &warm,
        )?,
        Path::Lag => lag_arm(
            tally,
            &backend,
            &mut runtime,
            (spec, ty, model),
            &cell,
            &warm,
        )?,
        Path::Restore => restore_arm(
            tally,
            &backend,
            &mut runtime,
            (spec, ty, model),
            &cell,
            &warm,
        )?,
        Path::Inflight => inflight_arm(
            tally,
            &backend,
            &mut runtime,
            (spec, ty, model),
            &cell,
            seed,
        )?,
    };
    ids.absorb_sessions(&backend)?;
    let stale = hits + extra_stale;
    record_cell(tally, &cell, stale);
    if ty == Invalidation::Supersession && path != Path::Inflight {
        let key = format!("{}_{}", spec.short, path.name());
        tally.add(&format!("superseded_lookup_cases.{key}"), 1);
        tally.add(&format!("superseded_stale_hits.{key}"), stale);
    }
    infos.push(CellInfo {
        cell,
        evidence: "session",
        oracle: if model.semantic_state(spec) {
            "ptr-runtime committed semantic state (host write; Revoked for a revocation, a committed edit for a supersession)"
        } else {
            "ptr-runtime ledger (CapsuleCommitted / Revoked / CapsuleSuperseded)"
        },
        procedure,
    });
    Ok(())
}

fn note_revocation_probes(tally: &mut Tally, ty: Invalidation, probed: &Probed) {
    if ty == Invalidation::Revocation {
        tally.add("post_revocation_lookups_matching_tokens", probed.lookups);
        tally.add("revoked_fork_attempts", probed.lookups);
        tally.add("revoked_fork_count", probed.hits);
        tally.add("revoked_lineage_entry_hits", probed.hits);
    }
}

fn note_event(tally: &mut Tally, ty: Invalidation) {
    if ty == Invalidation::Revocation {
        tally.add("input_revocations_performed", 1);
    }
}

fn normal_arm(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    runtime: &mut PtrRuntime,
    (spec, ty, model): (&SegSpec, Invalidation, Model),
    cell: &str,
    warm: &SegmentedModelRequest,
) -> Res<(i64, i64)> {
    let outcome = lab::run_completed(backend, &scope(), warm, &*runtime)?;
    if outcome.inserted_chunks != 2 {
        return Err(format!(
            "{cell}: the warm run cached {} chunks, not 2",
            outcome.inserted_chunks
        ));
    }
    note_event(tally, ty);
    let retired = apply_event(runtime, backend, spec, ty, model)?;
    if retired == 0 {
        return Err(format!("{cell}: the applied invalidation retired nothing"));
    }
    // Every entry of this backend depends on the invalidated input: whatever is still
    // charged right after the applied invalidation is unreachable and unreclaimed.
    let after_event = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add(
        "unreachable_revoked_page_bytes_unreclaimed",
        after_event.page_bytes as i64,
    );
    let probed = probe_entries(tally, backend, runtime, spec, cell)?;
    note_revocation_probes(tally, ty, &probed);
    // Entries of the invalidated lineage that a request admitted again after the event.
    let after_probes = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add(
        &format!("stale_lineage_entries_reinserted.{cell}"),
        after_probes.entries.saturating_sub(after_event.entries) as i64,
    );
    Ok((probed.hits, 0))
}

fn lag_arm(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    runtime: &mut PtrRuntime,
    (spec, ty, model): (&SegSpec, Invalidation, Model),
    cell: &str,
    warm: &SegmentedModelRequest,
) -> Res<(i64, i64)> {
    let outcome = lab::run_completed(backend, &scope(), warm, &*runtime)?;
    if outcome.inserted_chunks != 2 {
        return Err(format!(
            "{cell}: the warm run cached {} chunks, not 2",
            outcome.inserted_chunks
        ));
    }
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .map_err(fail("injecting the dropped invalidation"))?;
    let eligible = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries as i64;
    note_event(tally, ty);
    let retired = apply_event(runtime, backend, spec, ty, model)?;
    if retired != 0 {
        return Err(format!(
            "{cell}: the suppressed invalidation retired {retired} entries"
        ));
    }
    let still = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries as i64;
    if still != eligible {
        return Err(format!(
            "{cell}: the stale index changed ({eligible} -> {still})"
        ));
    }
    let probed = probe_entries(tally, backend, runtime, spec, cell)?;
    note_revocation_probes(tally, ty, &probed);
    tally.add("invalidation_suppressed_arms", 1);
    tally.add("stale_eligible_entries_before_event", eligible);
    tally.add(
        "post_event_lookups_on_stale_eligible_entries",
        probed.lookups,
    );
    tally.add("post_event_lookups", probed.lookups);
    tally.add("stale_hits", probed.hits);
    let after = backend.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add("lag_use_time_rejections", after.use_time_rejections as i64);
    tally.add(
        "unreachable_revoked_page_bytes_unreclaimed",
        after.page_bytes as i64,
    );
    Ok((probed.hits, 0))
}

fn restore_arm(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    runtime: &mut PtrRuntime,
    (spec, ty, model): (&SegSpec, Invalidation, Model),
    cell: &str,
    warm: &SegmentedModelRequest,
) -> Res<(i64, i64)> {
    let outcome = lab::run_completed(backend, &scope(), warm, &*runtime)?;
    if outcome.inserted_chunks != 2 {
        return Err(format!(
            "{cell}: the warm run cached {} chunks, not 2",
            outcome.inserted_chunks
        ));
    }
    let snapshot = backend
        .export_catalog()
        .map_err(fail("exporting the catalog"))?;
    // The restart: a fresh backend takes the pre-event snapshot and is never told of the
    // event; only the ledger knows.
    let restored = lab::main_backend()?;
    let report: KvRestoreReport = restored
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the catalog"))?;
    // A snapshot carries no pages. The restarted backend recomputes the same content-
    // addressed pages by running the same tokens as one raw segment (another lineage, so the
    // restored entries are neither used nor replaced), which leaves every restored entry able
    // to fork: only the use-time liveness check can refuse it now.
    materialise_pages(&restored, &*runtime, warm, &format!("{cell}-pages"))?;
    note_event(tally, ty);
    ledger_event(runtime, spec, ty, model)?;
    // Observed in the catalog after the event and before the post-event lookups.
    let observed = restored.catalog_stats().map_err(fail("catalog stats"))?;
    if observed.restored_entries as usize != report.entries
        || observed.entries != report.entries + outcome.inserted_chunks
    {
        return Err(format!(
            "{cell}: the restored entries were not observed in the catalog"
        ));
    }
    let probed = probe_entries(tally, &restored, runtime, spec, cell)?;
    note_revocation_probes(tally, ty, &probed);
    // Control: the same snapshot, pages recomputed, probed while the input is still live,
    // forks: the misses above are the oracle's refusal, not an absent page.
    let control = lab::main_backend()?;
    control
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the control"))?;
    let live = kind_runtime(spec, model)?;
    materialise_pages(&control, &live, warm, &format!("{cell}-control-pages"))?;
    let control_probe = probe_request(spec, &format!("{cell}-control"), 32);
    let control_session = control
        .begin_session(&scope(), &control_probe, None, &live)
        .map_err(fail("the restore control"))?;
    let control_hit = control_session.hit_tokens();
    drop(control_session);
    if control_hit != 32 {
        return Err(format!(
            "{cell}: a live restored entry with its pages did not fork ({control_hit} tokens)"
        ));
    }
    tally.add("restore_control_hit_tokens", control_hit as i64);
    let rejections = restored.catalog_stats().map_err(fail("catalog stats"))?;
    tally.add("restore_arms", 1);
    tally.add("restored_pre_revocation_entries", report.entries as i64);
    tally.add("stale_eligible_entries_before_event", report.entries as i64);
    tally.add(
        "post_event_lookups_on_stale_eligible_entries",
        probed.lookups,
    );
    tally.add("post_event_lookups", probed.lookups);
    tally.add("stale_hits", probed.hits);
    tally.add(
        "restore_use_time_rejections",
        rejections.use_time_rejections as i64,
    );
    // The raw run that recomputed the pages still holds them, legitimately: only bytes
    // beyond what that run alone charges belong to the invalidated lineage.
    let baseline = lab::main_backend()?;
    materialise_pages(
        &baseline,
        &*runtime,
        warm,
        &format!("{cell}-baseline-pages"),
    )?;
    let raw_only = baseline
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .page_bytes;
    tally.add(
        "unreachable_revoked_page_bytes_unreclaimed",
        rejections.page_bytes.saturating_sub(raw_only) as i64,
    );
    Ok((probed.hits, 0))
}

fn inflight_arm(
    tally: &mut Tally,
    backend: &ReferenceKvReuseBackend,
    runtime: &mut PtrRuntime,
    (spec, ty, model): (&SegSpec, Invalidation, Model),
    cell: &str,
    seed: u64,
) -> Res<(i64, i64)> {
    let warm = warm_request(spec, &format!("{cell}-warm"));
    let outcome = lab::run_completed(backend, &scope(), &warm, &*runtime)?;
    if outcome.inserted_chunks != 2 {
        return Err(format!(
            "{cell}: the warm run cached {} chunks, not 2",
            outcome.inserted_chunks
        ));
    }
    // A request that forks the cached prefix (hit 32) and then crosses a chunk boundary of
    // its own (tail 24: boundaries 48), so inserts are in reach of the race too.
    let id = format!("{cell}-fork");
    let request = lineage_request(spec, &id, 1, workload::sequence(spec.label + 11, 24));
    let total_steps = Steps::count(request.total_tokens(), 32);
    let race_at = workload::scenario(seed, cell).range(0, total_steps as u64) as usize;
    let session = backend
        .begin_session(&scope(), &request, None, &*runtime)
        .map_err(fail("the forking session"))?;
    if session.hit_tokens() != 32 {
        return Err(format!(
            "{cell}: the session did not fork the cached prefix"
        ));
    }
    let mut steps = Steps { session };
    for _ in 0..race_at {
        if steps.step(&*runtime)? {
            return Err(format!("{cell}: cancelled before the event"));
        }
    }
    let tokens_at_event = steps.session.generated().len();
    note_event(tally, ty);
    apply_event(runtime, backend, spec, ty, model)?;
    let entries_after_event = backend.catalog_stats().map_err(fail("catalog stats"))?;
    // The next step is the first use after the event.
    let cancelled = if race_at < total_steps {
        steps.step(&*runtime)?
    } else {
        matches!(
            steps.session.finish(&*runtime).map_err(fail("finishing"))?,
            ptr_runtime::KvSessionFinish::Cancelled(_)
        )
    };
    let mut escaped = 0i64;
    if !cancelled {
        // The fork survived the event: whatever it produced from here is an escape.
        escaped += 1;
    }
    escaped += (steps.session.generated().len() - tokens_at_event) as i64;
    let restarted = if cancelled {
        let restarted = steps.session.restart().map_err(fail("the restart"))?;
        let cold = lab::cold(backend, &scope(), &request)?;
        let equal = compare(tally, &restarted, &cold);
        let fresh = restarted.hit_tokens == 0
            && restarted.prefilled_tokens == request.total_tokens()
            && restarted.inserted_chunks == 0
            && restarted.restarted_after_cancel;
        Some(equal && fresh)
    } else {
        None
    };
    let after_restart = backend.catalog_stats().map_err(fail("catalog stats"))?;
    let inserted_after = after_restart
        .entries
        .saturating_sub(entries_after_event.entries) as i64;
    tally.add(
        "unreachable_revoked_page_bytes_unreclaimed",
        after_restart.page_bytes as i64,
    );
    // A later request citing the dead generation must not fork anything either.
    let again = lineage_request(
        spec,
        &format!("{cell}-after"),
        1,
        workload::sequence(spec.label + 13, 4),
    );
    let late = backend
        .begin_session(&scope(), &again, None, &*runtime)
        .map_err(fail("the late session"))?;
    let late_hit = i64::from(late.hit_tokens() > 0);
    drop(late);

    tally.add("cancelled_forks", i64::from(cancelled));
    tally.add(
        "cancelled_forks_restarted_from_fresh_prefill",
        i64::from(restarted == Some(true)),
    );
    tally.add(
        "outputs_escaped_after_revocation",
        if ty == Invalidation::Revocation {
            escaped
        } else {
            0
        },
    );
    tally.add("race_outputs_after_event", escaped);
    if ty == Invalidation::Revocation {
        tally.add("active_fork_revocation_attempts", 1);
    }
    if let Some(member) = member(spec, ty) {
        tally.add(&format!("member_race_cases.{member}"), 1);
        tally.add(
            &format!("member_cancelled_forks.{member}"),
            i64::from(cancelled),
        );
    }
    match (spec.kind, ty) {
        (SegmentKind::Memory, Invalidation::Revocation) => {
            tally.add("memory_revocation_active_fork_race_cases", 1)
        }
        (SegmentKind::Semantic, Invalidation::Supersession) => {
            tally.add("supersession_active_fork_race_cases", 1)
        }
        (SegmentKind::Memory, Invalidation::Supersession) => {
            tally.add("memory_supersession_active_fork_race_cases", 1)
        }
        _ => {}
    }
    if ty == Invalidation::Supersession {
        tally.add("supersession_cancelled_forks", i64::from(cancelled));
        tally.add(
            "supersession_forks_restarted_from_fresh_prefill",
            i64::from(restarted == Some(true)),
        );
        tally.add("cache_inserts_after_supersession", inserted_after);
        match spec.kind {
            SegmentKind::Semantic => tally.add("outputs_after_supersession_semantic", escaped),
            SegmentKind::Memory => tally.add("outputs_after_supersession_memory", escaped),
            _ => tally.add("outputs_after_supersession_derived", escaped),
        }
    } else {
        tally.add("cache_inserts_after_revocation", inserted_after);
    }
    tally.add("race_points_total", total_steps as i64);
    tally.add(&format!("race_point.{cell}"), race_at as i64);
    // A stale hit of this cell is an escaped output, a late fork or a post-event insert.
    Ok((escaped + late_hit + inserted_after, 0))
}

// ---- the catalog layer ------------------------------------------------------------------

/// A dependency kind of the catalog layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatKind {
    BaseModel,
    Adapter,
    Expert,
}

impl CatKind {
    pub const ALL: [CatKind; 3] = [CatKind::BaseModel, CatKind::Adapter, CatKind::Expert];

    pub fn cell_kind(self) -> &'static str {
        match self {
            Self::BaseModel => "base_model",
            Self::Adapter => "adapter_generation",
            Self::Expert => "expert_generation",
        }
    }

    fn identity(self) -> String {
        match self {
            Self::BaseModel => "model:base".to_string(),
            Self::Adapter => "adapter:lora".to_string(),
            Self::Expert => {
                KvDependency::expert(
                    &ProjectId::from("proj"),
                    &PodId::from("x"),
                    Generation(1),
                    [9; 32],
                )
                .identity
            }
        }
    }

    fn kind(self) -> KvDependencyKind {
        match self {
            Self::BaseModel => KvDependencyKind::BaseModel,
            Self::Adapter => KvDependencyKind::Adapter,
            Self::Expert => KvDependencyKind::Expert,
        }
    }

    fn dependency(self) -> KvDependency {
        match self {
            Self::BaseModel => KvDependency::base_model("model:base", Generation(1), [7; 32]),
            Self::Adapter => KvDependency::adapter("adapter:lora", Generation(1), [8; 32]),
            Self::Expert => KvDependency::expert_fenced(
                &ProjectId::from("proj"),
                &PodId::from("x"),
                Generation(1),
                [9; 32],
                3,
            ),
        }
    }

    /// The types this kind has: an expert also has a fence.
    pub fn types(self) -> Vec<Invalidation> {
        let mut types = vec![Invalidation::Revocation, Invalidation::Supersession];
        if self == Self::Expert {
            types.push(Invalidation::Fence);
        }
        types
    }
}

/// The liveness oracle of a catalog cell: the ledger where the runtime can answer for the
/// kind (base model and adapter generations are lifecycle targets), a script for the expert
/// (whose liveness the runtime reads from a materialized state record these cells do not
/// build).
enum Oracle {
    Ledger(Box<PtrRuntime>),
    Script(Scripted),
}

impl Oracle {
    fn for_kind(kind: CatKind) -> Res<Self> {
        Ok(match kind {
            CatKind::Expert => Self::Script(Scripted::default()),
            _ => Self::Ledger(Box::new(runtime_with(&[kind.identity().as_str()])?)),
        })
    }

    fn view(&self) -> &dyn KvLivenessOracle {
        match self {
            Self::Ledger(runtime) => &**runtime,
            Self::Script(script) => script,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Ledger(_) => {
                "ptr-runtime ledger (CapsuleCommitted / Revoked / CapsuleSuperseded)"
            }
            Self::Script(_) => {
                "scripted oracle (expert liveness needs a materialized expert record)"
            }
        }
    }

    /// The event commits (or is scripted) here, and nowhere else.
    fn event(&mut self, identity: &str, ty: Invalidation) -> Res<()> {
        match (self, ty) {
            (Self::Ledger(runtime), Invalidation::Revocation) => {
                commit_revocation(runtime, identity, 1)
            }
            (Self::Ledger(runtime), Invalidation::Supersession) => {
                commit_supersession(runtime, identity, 1, 2)
            }
            (Self::Script(script), Invalidation::Revocation) => {
                script.kill(identity, 1);
                Ok(())
            }
            (Self::Script(script), Invalidation::Supersession) => {
                script.supersede(identity, 2);
                Ok(())
            }
            (Self::Script(script), Invalidation::Fence) => {
                script.fence(identity, 4);
                Ok(())
            }
            (Self::Ledger(_), Invalidation::Fence) => Err("only an expert has a fence".to_string()),
        }
    }
}

/// The invalidation the runtime's hook would make on the catalog: tombstone and retire.
fn catalog_invalidate(catalog: &mut KvPrefixCatalog, kind: CatKind, ty: Invalidation) -> usize {
    let identity = kind.identity();
    match ty {
        Invalidation::Revocation => catalog
            .invalidate_inputs(&BTreeSet::from([(identity, Generation(1))]))
            .len(),
        Invalidation::Supersession => catalog
            .invalidate_kind_superseded(kind.kind(), &BTreeMap::from([(identity, Generation(2))]))
            .len(),
        Invalidation::Fence => catalog
            .invalidate_expert_fences(&BTreeMap::from([(identity, 4)]))
            .len(),
    }
}

/// Run one cell of a base-model, adapter or expert dependency on a bare catalog.
pub fn catalog_cell(
    tally: &mut Tally,
    infos: &mut Vec<CellInfo>,
    kind: CatKind,
    ty: Invalidation,
    path: Path,
) -> Res<()> {
    let cell = params::cell_name(kind.cell_kind(), path.name(), ty.name());
    let limits = lab::catalog_limits(params::Workload::Main);
    let mut lab = CatLab::new(limits)?;
    let procedure = procedure(tally, params::Workload::Main, lab.catalog.limits());
    let mut oracle = Oracle::for_kind(kind)?;
    let identity = kind.identity();
    let request = lab::two_blocks(11, 12);
    let dependency = kind.dependency();
    let inserted = lab
        .insert(&request, vec![dependency.clone()], oracle.view())?
        .map_err(|error| format!("{cell}: the first insert was refused: {error}"))?;
    let mut stale = 0i64;
    match path {
        Path::Normal => {
            oracle.event(&identity, ty)?;
            let retired = catalog_invalidate(&mut lab.catalog, kind, ty);
            if retired == 0 {
                return Err(format!("{cell}: the applied invalidation retired nothing"));
            }
            stale += i64::from(lab.lookup(&request, oracle.view()).is_some());
            stale += lab.catalog.stats().entries as i64;
        }
        Path::Lag => {
            let eligible = lab.catalog.stats().entries as i64;
            oracle.event(&identity, ty)?;
            // The catalog is never told: only the use-time check can refuse.
            stale += i64::from(lab.lookup(&request, oracle.view()).is_some());
            if lab.catalog.stats().use_time_rejections < 1 {
                return Err(format!(
                    "{cell}: the use-time check did not reject the stale entry"
                ));
            }
            tally.add("invalidation_suppressed_arms", 1);
            tally.add("stale_eligible_entries_before_event", eligible);
            tally.add("post_event_lookups_on_stale_eligible_entries", eligible);
            tally.add("post_event_lookups", eligible);
            tally.add("stale_hits", stale);
        }
        Path::Restore => {
            let snapshot = lab
                .catalog
                .export_entries()
                .map_err(fail("exporting the catalog"))?;
            let mut target = CatLab::new(limits)?;
            let report = target
                .catalog
                .restore_entries(&snapshot)
                .map_err(fail("restoring the catalog"))?;
            // The restart recomputes the same content-addressed pages, so the restored
            // entry could fork: only the use-time liveness check stands in the way.
            let mut tokens = request.tokens();
            tokens.truncate(tokens.len() / params::CHUNK_TOKENS * params::CHUNK_TOKENS);
            let (reproduced, _) = target.publish(&tokens)?;
            if reproduced.key != inserted.key {
                return Err(format!(
                    "{cell}: the restored pages are not content-addressed"
                ));
            }
            oracle.event(&identity, ty)?;
            let observed = target.catalog.stats().entries as i64;
            if observed != report.entries as i64 || observed < 1 {
                return Err(format!(
                    "{cell}: the restored entry was not observed in the catalog"
                ));
            }
            stale += i64::from(target.lookup(&request, oracle.view()).is_some());
            tally.add("restore_arms", 1);
            tally.add("restored_pre_revocation_entries", observed);
            tally.add("stale_eligible_entries_before_event", observed);
            tally.add("post_event_lookups_on_stale_eligible_entries", observed);
            tally.add("post_event_lookups", observed);
            tally.add("stale_hits", stale);
        }
        Path::Inflight => {
            // An active fork: looked up, forked into a cache, pin held.
            let hit = lab
                .lookup(&request, oracle.view())
                .ok_or_else(|| format!("{cell}: no hit before the event"))?;
            let mut cache = lab.fresh_cache()?;
            let backend = lab.backend.clone();
            let pin = lab
                .catalog
                .fork(&backend, &mut cache, hit, oracle.view())
                .map_err(|error| format!("{cell}: the fork was refused: {error}"))?;
            oracle.event(&identity, ty)?;
            catalog_invalidate(&mut lab.catalog, kind, ty);
            // The fork must see that it was cancelled.
            stale += i64::from(lab.catalog.pin_is_live(&pin));
            // No cache insert derived from the invalidated dependency may follow.
            let later = lab::two_blocks(13, 14);
            let accepted = lab
                .insert(&later, vec![dependency.clone()], oracle.view())?
                .is_ok();
            stale += i64::from(accepted);
            if kind == CatKind::BaseModel {
                tally.add(&format!("base_model_inflight_race_cases.{}", ty.name()), 1);
                tally.add(
                    &format!("base_model_inflight_cache_inserts.{}", ty.name()),
                    i64::from(accepted),
                );
                tally.add(
                    &format!("base_model_inflight_pins_still_live.{}", ty.name()),
                    i64::from(lab.catalog.pin_is_live(&pin)),
                );
            }
            lab.catalog.release(&pin);
            if lab.catalog.stats().entries != 0 {
                return Err(format!(
                    "{cell}: the revoked pinned entry was not reclaimed"
                ));
            }
        }
    }
    record_cell(tally, &cell, stale);
    tally.add("catalog_layer_stale", stale);
    infos.push(CellInfo {
        cell,
        evidence: "catalog",
        oracle: oracle.name(),
        procedure,
    });
    Ok(())
}

/// Run every cell of the matrix for one seed.
pub fn run_all(tally: &mut Tally, ids: &mut Ids, seed: u64) -> Res<Vec<CellInfo>> {
    tally.seed_group("invalidation_matrix_cases", &params::MATRIX_CELLS);
    tally.seed_group("invalidation_matrix_stale_hits", &params::MATRIX_CELLS);
    let superseded: Vec<String> = params::SUPERSEDED_LOOKUP_CELLS
        .iter()
        .map(|[kind, path]| format!("{kind}_{path}"))
        .collect();
    tally.seed_group("superseded_lookup_cases", &superseded);
    tally.seed_group("superseded_stale_hits", &superseded);
    tally.seed_group("base_model_inflight_race_cases", &params::MATRIX_TYPES);
    tally.seed_group("base_model_inflight_cache_inserts", &params::MATRIX_TYPES);
    tally.seed_group(
        "base_model_inflight_outputs_after_invalidation",
        &params::MATRIX_TYPES,
    );
    let mut infos = Vec::new();
    for spec in &SEGMENT_KINDS {
        for path in Path::ALL {
            for ty in [Invalidation::Revocation, Invalidation::Supersession] {
                segment_cell(
                    tally,
                    ids,
                    &mut infos,
                    seed,
                    spec,
                    ty,
                    path,
                    Model::Committed,
                )?;
            }
        }
    }
    // Control: the semantic supersession cells whose catalog invalidation is lost, with the
    // semantic key held as a lifecycle target. A difference from the cells above says the
    // use-time check can see a supersession when the input has a lifecycle generation.
    for path in [Path::Lag, Path::Restore] {
        let mut control = Tally::default();
        let mut scratch = Ids::default();
        let mut discarded = Vec::new();
        segment_cell(
            &mut control,
            &mut scratch,
            &mut discarded,
            seed,
            &SEGMENT_KINDS[0],
            Invalidation::Supersession,
            path,
            Model::Lifecycle,
        )?;
        tally.add("semantic_lifecycle_control_cells", 1);
        tally.add(
            "semantic_lifecycle_control_stale_hits",
            control.get("stale_hits"),
        );
    }
    for kind in CatKind::ALL {
        for path in Path::ALL {
            for ty in kind.types() {
                catalog_cell(tally, &mut infos, kind, ty, path)?;
            }
        }
    }
    Ok(infos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_configured_cell_is_produced_exactly_once() {
        let mut expected: Vec<String> = Vec::new();
        for spec in &SEGMENT_KINDS {
            for path in Path::ALL {
                for ty in [Invalidation::Revocation, Invalidation::Supersession] {
                    expected.push(params::cell_name(spec.cell_kind, path.name(), ty.name()));
                }
            }
        }
        for kind in CatKind::ALL {
            for path in Path::ALL {
                for ty in kind.types() {
                    expected.push(params::cell_name(kind.cell_kind(), path.name(), ty.name()));
                }
            }
        }
        expected.sort();
        let mut configured: Vec<String> =
            params::MATRIX_CELLS.iter().map(|c| c.to_string()).collect();
        configured.sort();
        assert_eq!(expected, configured);
    }

    #[test]
    fn a_session_cell_of_each_path_runs_and_sees_no_stale_hit() {
        for path in Path::ALL {
            let mut tally = Tally::default();
            let mut ids = Ids::default();
            let mut infos = Vec::new();
            segment_cell(
                &mut tally,
                &mut ids,
                &mut infos,
                17,
                &SEGMENT_KINDS[1],
                Invalidation::Revocation,
                path,
                Model::Committed,
            )
            .unwrap();
            let cell = params::cell_name("memory_input", path.name(), "revocation");
            assert_eq!(
                tally.get(&format!("invalidation_matrix_cases.{cell}")),
                1,
                "{cell}"
            );
            assert_eq!(
                tally.get(&format!("invalidation_matrix_stale_hits.{cell}")),
                0,
                "{cell}"
            );
            assert_eq!(tally.get("bitwise_mismatch_count"), 0, "{cell}");
            assert_eq!(infos.len(), 1);
        }
    }

    fn stale_hits_of(spec: &SegSpec, ty: Invalidation, path: Path, model: Model) -> (i64, Tally) {
        let mut tally = Tally::default();
        let mut ids = Ids::default();
        let mut infos = Vec::new();
        segment_cell(&mut tally, &mut ids, &mut infos, 17, spec, ty, path, model).unwrap();
        let cell = params::cell_name(spec.cell_kind, path.name(), ty.name());
        (
            tally.get(&format!("invalidation_matrix_stale_hits.{cell}")),
            tally,
        )
    }

    #[test]
    fn a_revoked_semantic_input_is_refused_at_use_with_the_catalog_index_lost() {
        for path in [Path::Lag, Path::Restore] {
            let (stale, tally) = stale_hits_of(
                &SEGMENT_KINDS[0],
                Invalidation::Revocation,
                path,
                Model::Committed,
            );
            assert_eq!(stale, 0, "{path:?}");
            assert_eq!(tally.get("stale_hits"), 0, "{path:?}");
        }
    }

    #[test]
    fn a_semantic_key_with_a_lifecycle_generation_is_seen_superseded_with_the_index_lost() {
        // The control of the run: held as a lifecycle target, the supersession is seen. What
        // the committed-state model shows for the same cells is the finding of the run.
        for path in [Path::Lag, Path::Restore] {
            let (stale, tally) = stale_hits_of(
                &SEGMENT_KINDS[0],
                Invalidation::Supersession,
                path,
                Model::Lifecycle,
            );
            assert_eq!(stale, 0, "{path:?}");
            assert!(
                tally.get("lag_use_time_rejections") + tally.get("restore_use_time_rejections")
                    >= 1
            );
        }
    }

    /// The finding of the first dry run, kept as a test that fails until it is fixed: a
    /// semantic input held as the runtime holds it (a key of the committed semantic state)
    /// and then edited by a committed write is still "live" to `PtrRuntime::is_live`, which
    /// ignores the content digest of the dependency (`kv_prefix.rs`, `impl KvLivenessOracle
    /// for PtrRuntime`), so with the catalog's invalidation lost a request citing the old
    /// content hits the old entry. Run it with `cargo test -p ptr-bench -- --ignored
    /// a_committed_edit_of_a_semantic_input`.
    #[test]
    #[ignore = "known finding: the runtime's use-time check cannot see a committed edit of a semantic input"]
    fn a_committed_edit_of_a_semantic_input_is_seen_at_use_with_the_index_lost() {
        for path in [Path::Lag, Path::Restore] {
            let (stale, _) = stale_hits_of(
                &SEGMENT_KINDS[0],
                Invalidation::Supersession,
                path,
                Model::Committed,
            );
            assert_eq!(
                stale, 0,
                "{path:?}: a stale hit on the superseded semantic input"
            );
        }
    }

    #[test]
    fn a_restored_live_entry_with_its_pages_forks_so_the_oracle_alone_refuses_a_dead_one() {
        let (_, tally) = stale_hits_of(
            &SEGMENT_KINDS[1],
            Invalidation::Revocation,
            Path::Restore,
            Model::Committed,
        );
        assert_eq!(tally.get("restore_control_hit_tokens"), 32);
        assert_eq!(tally.get("restored_pre_revocation_entries"), 2);
        assert_eq!(tally.get("restore_use_time_rejections"), 2);
    }

    #[test]
    fn a_supersession_race_cancels_and_restarts_from_a_fresh_prefill() {
        let mut tally = Tally::default();
        let mut ids = Ids::default();
        let mut infos = Vec::new();
        segment_cell(
            &mut tally,
            &mut ids,
            &mut infos,
            29,
            &SEGMENT_KINDS[0],
            Invalidation::Supersession,
            Path::Inflight,
            Model::Committed,
        )
        .unwrap();
        assert_eq!(tally.get("cancelled_forks"), 1);
        assert_eq!(tally.get("supersession_cancelled_forks"), 1);
        assert_eq!(
            tally.get("supersession_forks_restarted_from_fresh_prefill"),
            1
        );
        assert_eq!(tally.get("outputs_after_supersession_semantic"), 0);
        assert_eq!(tally.get("cache_inserts_after_supersession"), 0);
        assert_eq!(ids.cancelled, ids.restarted);
        assert_eq!(ids.cancelled.len(), 1);
    }

    #[test]
    fn a_catalog_cell_of_every_path_sees_no_stale_hit_for_every_kind_and_type() {
        for kind in CatKind::ALL {
            for ty in kind.types() {
                for path in Path::ALL {
                    let mut tally = Tally::default();
                    let mut infos = Vec::new();
                    catalog_cell(&mut tally, &mut infos, kind, ty, path).unwrap();
                    let cell = params::cell_name(kind.cell_kind(), path.name(), ty.name());
                    assert_eq!(
                        tally.get(&format!("invalidation_matrix_cases.{cell}")),
                        1,
                        "{cell}"
                    );
                    assert_eq!(
                        tally.get(&format!("invalidation_matrix_stale_hits.{cell}")),
                        0,
                        "{cell}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_stale_oracle_would_be_caught_as_a_stale_hit() {
        // Negative control: a lookup against an oracle that still says live finds the entry.
        let mut lab = CatLab::new(CatalogLimits::default()).unwrap();
        let live = Scripted::default();
        let dependency = CatKind::BaseModel.dependency();
        let request = lab::two_blocks(21, 22);
        lab.insert(&request, vec![dependency], &live)
            .unwrap()
            .unwrap();
        let hit = lab.lookup(&request, &live);
        assert!(
            hit.is_some(),
            "an oracle that never learns of the event serves the entry"
        );
    }
}
