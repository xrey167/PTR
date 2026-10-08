//! The 52-cell invalidation matrix of R006 (case C9): six dependency kinds, revocation and
//! supersession (plus the expert fence), four paths (normal, lag, restore, in-flight), each
//! cell run at least once per seed and reported per cell.
//!
//! The cells reuse the semantics of the R005/R010 matrix (owner decision 2) with their own
//! arms on the one-layer positional backend. Two kinds of evidence, labelled per cell:
//!
//! * `session_lookup`, `session_recorded_dependency`: the semantic, memory and derived
//!   segments and the base model run on **real positional sessions**
//!   (`ReferenceKvReuseBackend::begin_session_positional`, `probe_positional`) under the
//!   ledger-anchored [`Authority`] oracle (a real `PtrRuntime`). The base model is a real
//!   recorded dependency of every positional session.
//! * `declared_dependency`: an adapter generation and an expert are not part of a one-layer
//!   dense model's computation. They are **declared** by the caller as the extra dependencies
//!   of a run inserted directly into a standalone catalog's secondary index, under real
//!   liveness oracles (an adapter capsule committed to a ledger, real expert records written
//!   by the admission path, including the fence epoch). The in-flight cells of these kinds
//!   exercise the index's pin protocol (`revalidate` before every consumption and the
//!   insert-time dependency check), not a session: a backend session cannot take a declared
//!   dependency. The dense one-layer model has no adapter or expert influence.
//!
//! A cell's stale hits are structural: for a normal cell every candidate returned or admitted
//! after the event; for a lag cell every admitted span (the dead page is reachable by design,
//! and must be refused with its typed reason); for a restore cell every row attached to a run
//! whose dependency died at rest plus anything returned or admitted afterwards; for an
//! in-flight cell every token emitted, span repacked or run inserted after the event step.

use super::cases::{after, case_json, forbid, note_exact, request, seed_origin, step_to};
use super::lab::{
    self, admitted, fresh_check, reachable, report, run_cold, run_positional, Escaped, Lab,
    SHORT_FILLER_TOKENS,
};
use super::params;
use crate::experiments::layered_kv::corpus::Part;
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::experts;
use crate::experiments::layered_kv::lab::{
    self as shared, commit_revocation, commit_supersession, fail, write_semantic, Authority, Res,
    BASE_MODEL_GENERATION, BASE_MODEL_IDENTITY,
};
use ptr_ledger::{ExpertFenceCause, LedgerEvent};
use ptr_model_api::{KvReuseMode, SegmentKind, SegmentedModelRequest};
use ptr_pods::{InMemoryPagedKvBackend, KvPageSize, PagedKvTensorBackend};
use ptr_runtime::{
    expert_operand_identities, CatalogLimits, IndexUpdate, KvChainId, KvDependency,
    KvDependencyKind, KvLivenessOracle, KvPrefixCatalog, KvRequestPositions, KvSalt, KvSessionStep,
    PositionalControl, ReferenceKvFaults, ReferenceKvReuseBackend, SecondaryError, SecondaryInsert,
    SecondaryInsertOutcome, SecondaryLookup, SecondaryLookupOutcome, SecondaryPinState,
    SecondaryRefusalReason, SecondaryRows,
};
use ptr_types::{Generation, ProjectId};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// The names of the matrix cells, in the order of `config.toml`.
pub fn cell_names() -> Vec<&'static str> {
    params::INVALIDATION_COVERAGE_MATRIX_CELLS.to_vec()
}

/// What the matrix distinguishes about an invalidation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Ty {
    Revocation,
    Supersession,
    /// A committed fence of an expert: the generation is unchanged.
    Fence,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
            Self::Fence => "fence",
        }
    }

    fn parse(name: &str) -> Res<Self> {
        match name {
            "revocation" => Ok(Self::Revocation),
            "supersession" => Ok(Self::Supersession),
            "fence" => Ok(Self::Fence),
            other => Err(format!("unknown invalidation type {other}")),
        }
    }
}

/// A prompt segment kind of the matrix: the tested segment `X` sits upstream of `C`.
#[derive(Clone, Copy, Debug)]
struct SegSpec {
    kind: SegmentKind,
    cell_kind: &'static str,
    identity: &'static str,
}

const SEGMENTS: [SegSpec; 3] = [
    SegSpec {
        kind: SegmentKind::Semantic,
        cell_kind: "semantic_lineage_segment",
        identity: "sem:r006-x",
    },
    SegSpec {
        kind: SegmentKind::Memory,
        cell_kind: "memory_lineage_segment",
        identity: "mem:r006-x",
    },
    SegSpec {
        kind: SegmentKind::Derived,
        cell_kind: "derived_lineage_segment",
        identity: "ev:r006-x",
    },
];

/// A dependency tested on a real positional session.
#[derive(Clone, Copy, Debug)]
enum Subject {
    Segment(&'static SegSpec),
    BaseModel,
}

impl Subject {
    fn evidence(self) -> &'static str {
        match self {
            Self::Segment(_) => params::EVIDENCE_LABELS_SEMANTIC_LINEAGE_SEGMENT,
            Self::BaseModel => params::EVIDENCE_LABELS_BASE_MODEL,
        }
    }

    fn is_semantic(self) -> bool {
        matches!(self, Self::Segment(spec) if spec.kind == SegmentKind::Semantic)
    }

    fn identity(self) -> &'static str {
        match self {
            Self::Segment(spec) => spec.identity,
            Self::BaseModel => BASE_MODEL_IDENTITY,
        }
    }

    /// The refusal reason the use-time check returns for this subject when the index is stale.
    fn dead_reason(self) -> SecondaryRefusalReason {
        match self {
            Self::Segment(_) => SecondaryRefusalReason::OriginalUpstreamDead,
            Self::BaseModel => SecondaryRefusalReason::BindingDependencyDead,
        }
    }

    fn other_dead_reason(self) -> SecondaryRefusalReason {
        match self {
            Self::Segment(_) => SecondaryRefusalReason::BindingDependencyDead,
            Self::BaseModel => SecondaryRefusalReason::OriginalUpstreamDead,
        }
    }
}

/// The oracle of a session cell: a real runtime whose ledger commits the base model, `C` and
/// the tested capsule, or holds the semantic key.
fn authority(subject: Subject) -> Res<Authority> {
    let mut capsules = vec![lab::C_IDENTITY];
    match subject {
        Subject::Segment(spec) if spec.kind == SegmentKind::Semantic => {
            Authority::with_semantic(spec.identity, &capsules)
        }
        Subject::Segment(spec) => {
            capsules.push(spec.identity);
            Authority::new(&capsules)
        }
        Subject::BaseModel => Authority::new(&capsules),
    }
}

/// `[X][C][tail]`, or `[C][tail]` for the base model.
fn body(lab: &Lab<'_>, subject: Subject) -> Vec<Part> {
    let mut parts = Vec::new();
    if let Subject::Segment(spec) = subject {
        parts.push(Part::bound(
            spec.kind,
            spec.identity,
            1,
            lab.material.upstream(),
        ));
    }
    parts.push(lab::c_part(lab.material, lab.shape));
    parts.push(lab::tail_part(lab.material, lab.shape));
    parts
}

/// Commit the event to the ledger (and so to the state the oracle reads).
fn commit_event(subject: Subject, ty: Ty, authority: &mut Authority) -> Res<()> {
    let runtime = &mut authority.runtime;
    match (subject, ty) {
        (_, Ty::Fence) => Err("only an expert has a fence".to_string()),
        (Subject::BaseModel, Ty::Revocation) => {
            commit_revocation(runtime, BASE_MODEL_IDENTITY, BASE_MODEL_GENERATION)
        }
        (Subject::BaseModel, Ty::Supersession) => commit_supersession(
            runtime,
            BASE_MODEL_IDENTITY,
            BASE_MODEL_GENERATION,
            BASE_MODEL_GENERATION + 1,
        ),
        (Subject::Segment(spec), Ty::Revocation) => commit_revocation(runtime, spec.identity, 1),
        (Subject::Segment(spec), Ty::Supersession) if spec.kind == SegmentKind::Semantic => {
            write_semantic(runtime, spec.identity, "v2")
        }
        (Subject::Segment(spec), Ty::Supersession) => {
            commit_supersession(runtime, spec.identity, 1, 2)
        }
    }
}

/// Commit the event and hand the committed events to the backend's catalog hook
/// (`apply_committed_event`, the host's equivalent of the runtime's hook; a lagging backend
/// suppresses the index update). A semantic edit commits a delta the hook cannot read, so its
/// supersession floor is applied by identity.
fn apply_event(
    subject: Subject,
    ty: Ty,
    authority: &mut Authority,
    backend: &ReferenceKvReuseBackend,
    fault: Fault,
) -> Res<()> {
    if fault == Fault::EventLost {
        return Ok(());
    }
    let mark = authority.mark();
    commit_event(subject, ty, authority)?;
    if fault == Fault::LostFanout {
        return Ok(());
    }
    shared::apply_events(backend, &authority.events_since(mark))?;
    if subject.is_semantic() && ty == Ty::Supersession {
        backend
            .supersede_inputs(&BTreeMap::from([(
                subject.identity().to_string(),
                Generation(2),
            )]))
            .map_err(fail("superseding inputs"))?;
    }
    Ok(())
}

fn suppress(backend: &ReferenceKvReuseBackend) -> Res<()> {
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .map_err(fail("injecting the lost invalidation"))
}

/// A fault injected into a cell so a test can show the cell's counters trip. The matrix of a
/// run always uses [`Fault::None`]; the tests run the same arms under a fault.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Fault {
    None,
    /// The ledger commits the event but the backend's catalog hook never hears of it.
    LostFanout,
    /// Use-time liveness is bypassed by the backend (the cell must also lag the index).
    Bypass,
    /// The event reaches neither the ledger nor the catalog.
    EventLost,
}

impl Fault {
    fn control(self) -> PositionalControl {
        match self {
            Self::Bypass => PositionalControl::BypassUseTimeLiveness,
            _ => PositionalControl::None,
        }
    }
}

/// What one cell produced.
struct Out {
    /// Case runs the cell made (an in-flight cell makes one per variant).
    cases: i64,
    stale: i64,
    detail: Json,
}

// ---- the session-backed arms ---------------------------------------------------------------

/// The live control of a cell: while the dependency is live, the same content at a new
/// position must be repacked. A miss would make every post-event miss an absent page.
fn live_control(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    parts: &[Part],
) -> Res<usize> {
    let live = request(
        &format!("{name}-live"),
        &after(lab.material.spare(0, SHORT_FILLER_TOKENS), parts),
    );
    let probe = run_positional(
        &mut lab.timings,
        &format!("{name}.live_control"),
        backend,
        &live,
        authority,
    )?;
    let reused = report(&probe)?.reused_tokens;
    if reused < lab.shape.segment {
        return Err(format!(
            "{name}: the live control repacked {reused} tokens, fewer than C's {}",
            lab.shape.segment
        ));
    }
    Ok(reused)
}

/// Run the post-event request on `backend` and check that it completed by a fresh prefill.
fn post_run(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    authority: &Authority,
    parts: &[Part],
) -> Res<(ptr_runtime::ReferenceKvProbe, lab::Fresh)> {
    let post = request(
        &format!("{name}-post"),
        &after(lab.material.spare(1, SHORT_FILLER_TOKENS), parts),
    );
    let probe = run_positional(
        &mut lab.timings,
        &format!("{name}.post"),
        backend,
        &post,
        authority,
    )?;
    if report(&probe)?.lookups != 1 {
        return Err(format!(
            "{name}: the post-event run made {} lookups",
            report(&probe)?.lookups
        ));
    }
    let fresh = fresh_check(&mut lab.timings, lab.anchor, name, backend, &post, &probe)?;
    lab.note_fresh(name, fresh);
    note_exact(lab, &probe, fresh.mismatches);
    Ok((probe, fresh))
}

fn forbid_stale_kinds(
    name: &str,
    subject: Subject,
    probe: &ptr_runtime::ReferenceKvProbe,
) -> Res<()> {
    forbid(
        name,
        &report(probe)?.refusals,
        &[
            SecondaryRefusalReason::SourceKindMismatch,
            SecondaryRefusalReason::SourceIdentityMismatch,
            SecondaryRefusalReason::SourceGenerationMismatch,
            SecondaryRefusalReason::SourceDigestMismatch,
            SecondaryRefusalReason::OriginalUpstreamOmitted,
            SecondaryRefusalReason::RawAncestorOmitted,
            subject.other_dead_reason(),
        ],
    )
}

fn normal_arm(lab: &mut Lab<'_>, subject: Subject, ty: Ty, name: &str, fault: Fault) -> Res<Out> {
    let mut authority = authority(subject)?;
    let backend = lab::positional_backend(lab.shape, fault.control())?;
    let parts = body(lab, subject);
    seed_origin(
        lab,
        &format!("{name}.origin"),
        &backend,
        &request(&format!("{name}-origin"), &parts),
        &authority,
    )?;
    let control = live_control(lab, name, &backend, &authority, &parts)?;
    lab.ids.invalidated.insert(name.to_string());
    apply_event(subject, ty, &mut authority, &backend, fault)?;
    let (probe, fresh) = post_run(lab, name, &backend, &authority, &parts)?;
    let positional = report(&probe)?;
    // After an applied invalidation nothing may be returned, refused or admitted.
    let stale = reachable(positional);
    Ok(Out {
        cases: 1,
        stale,
        detail: Json::obj([
            ("live_control_repacked_tokens", Json::Int(control as i64)),
            ("post_event", case_json(&probe)?),
            ("returned_after_event", Json::Int(stale)),
            ("fresh_prefill_completed", Json::Bool(fresh.completed)),
            ("fresh_prefill_mismatches", Json::Int(fresh.mismatches)),
        ]),
    })
}

fn lag_arm(lab: &mut Lab<'_>, subject: Subject, ty: Ty, name: &str, fault: Fault) -> Res<Out> {
    let mut authority = authority(subject)?;
    let backend = lab::positional_backend(lab.shape, fault.control())?;
    let parts = body(lab, subject);
    seed_origin(
        lab,
        &format!("{name}.origin"),
        &backend,
        &request(&format!("{name}-origin"), &parts),
        &authority,
    )?;
    let control = live_control(lab, name, &backend, &authority, &parts)?;
    suppress(&backend)?;
    lab.ids.invalidated.insert(name.to_string());
    apply_event(subject, ty, &mut authority, &backend, fault)?;
    if backend
        .secondary_stats()
        .map_err(fail("secondary stats"))?
        .invalidated
        != 0
    {
        return Err(format!(
            "{name}: the suppressed invalidation reached the index"
        ));
    }
    let (probe, fresh) = post_run(lab, name, &backend, &authority, &parts)?;
    let positional = report(&probe)?;
    forbid_stale_kinds(name, subject, &probe)?;
    // The dead page is reachable by design; admitting it is the stale hit.
    let stale = admitted(positional);
    let expected = positional.refusals.get(subject.dead_reason());
    if stale == 0 && expected == 0 {
        return Err(format!(
            "{name}: the lagging index never returned the stale page (no {} refusal): the cell tests nothing: {positional:?}",
            subject.dead_reason().code()
        ));
    }
    Ok(Out {
        cases: 1,
        stale,
        detail: Json::obj([
            ("live_control_repacked_tokens", Json::Int(control as i64)),
            ("post_event", case_json(&probe)?),
            ("typed_refusals", Json::Int(expected as i64)),
            ("expected_reason", Json::str(subject.dead_reason().code())),
            ("fresh_prefill_completed", Json::Bool(fresh.completed)),
            ("fresh_prefill_mismatches", Json::Int(fresh.mismatches)),
        ]),
    })
}

fn restore_arm(lab: &mut Lab<'_>, subject: Subject, ty: Ty, name: &str, fault: Fault) -> Res<Out> {
    let mut authority = authority(subject)?;
    let source = lab::positional_backend(lab.shape, PositionalControl::None)?;
    let parts = body(lab, subject);
    let origin = request(&format!("{name}-origin"), &parts);
    seed_origin(lab, &format!("{name}.origin"), &source, &origin, &authority)?;
    let snapshot = source
        .export_catalog()
        .map_err(fail("exporting the catalog"))?;

    // The restart: a fresh backend takes the pre-event snapshot and is never told of the
    // event; only the ledger knows.
    let restored = lab::positional_backend(lab.shape, PositionalControl::None)?;
    let report_restore = restored
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the catalog"))?;
    if report_restore.secondary_runs == 0 {
        return Err(format!("{name}: the snapshot carried no secondary runs"));
    }
    lab.ids.invalidated.insert(name.to_string());
    if fault != Fault::EventLost {
        commit_event(subject, ty, &mut authority)?;
    }

    // A re-insert of the same content under the post-event ledger must attach no rows.
    let again = run_positional(
        &mut lab.timings,
        &format!("{name}.reinsert"),
        &restored,
        &request(&format!("{name}-again"), &parts),
        &authority,
    )?;
    let again_report = report(&again)?;
    let stale_rows = again_report.rows_attached as i64 + again_report.inserted_segments as i64;
    let (probe, fresh) = post_run(lab, name, &restored, &authority, &parts)?;
    let positional = report(&probe)?;
    let stale = stale_rows + reachable(positional);

    // Control: the same snapshot, everything live: the re-insert attaches rows and the same
    // shifted request is served again, so the misses above are the oracle's refusal.
    let live_authority = self::authority(subject)?;
    let control = lab::positional_backend(lab.shape, PositionalControl::None)?;
    control
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the control"))?;
    let attached = run_positional(
        &mut lab.timings,
        &format!("{name}.control_reinsert"),
        &control,
        &request(&format!("{name}-control-again"), &parts),
        &live_authority,
    )?;
    if report(&attached)?.rows_attached == 0 {
        return Err(format!(
            "{name}: a live restored run did not get its rows back"
        ));
    }
    let control_reused = live_control(
        lab,
        &format!("{name}.control"),
        &control,
        &live_authority,
        &parts,
    )?;
    Ok(Out {
        cases: 1,
        stale,
        detail: Json::obj([
            (
                "restored_secondary_runs",
                Json::Int(report_restore.secondary_runs as i64),
            ),
            (
                "reinsert_rows_attached",
                Json::Int(again_report.rows_attached as i64),
            ),
            (
                "reinsert_inserted_segments",
                Json::Int(again_report.inserted_segments as i64),
            ),
            (
                "reinsert_insert_refusals",
                Json::Int(again_report.insert_refusals.len() as i64),
            ),
            ("post_event", case_json(&probe)?),
            (
                "control_rows_attached",
                Json::Int(report(&attached)?.rows_attached as i64),
            ),
            ("control_repacked_tokens", Json::Int(control_reused as i64)),
            ("fresh_prefill_completed", Json::Bool(fresh.completed)),
            ("fresh_prefill_mismatches", Json::Int(fresh.mismatches)),
        ]),
    })
}

/// When, in a session's steps, the event hits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum When {
    /// After the prefill reached the splice start, before the span is repacked.
    BeforeSplice,
    /// After the span was repacked and the prefill finished, before the first decode step.
    AfterSpliceBeforeDecode,
}

impl When {
    fn name(self) -> &'static str {
        match self {
            Self::BeforeSplice => "event_before_splice",
            Self::AfterSpliceBeforeDecode => "event_after_splice_before_decode",
        }
    }
}

/// One in-flight race on a session: the event hits at `when`; the next step is the first use
/// after it. Nothing may be emitted, repacked or inserted from there on, every pin must be
/// released and a cold restart must equal a cold run.
fn inflight_variant(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    authority: &mut Authority,
    request: &SegmentedModelRequest,
    when: When,
    event: &dyn Fn(&mut Authority, &ReferenceKvReuseBackend) -> Res<()>,
) -> Res<Escaped> {
    let mut session = backend
        .begin_session_positional(&shared::scope(), request, None, &*authority)
        .map_err(fail(name))?;
    let start = session
        .next_splice_start()
        .ok_or_else(|| format!("{name}: no span was admitted, so there is nothing in flight"))?;
    step_to(&mut session, &*authority, start, name)?;
    if when == When::AfterSpliceBeforeDecode {
        match session.prefill_chunk(&*authority).map_err(fail(name))? {
            KvSessionStep::Prefilled { .. } => {}
            other => return Err(format!("{name}: the repack step was {other:?}")),
        }
        if session
            .positional_report()
            .map_or(0, |report| report.splices)
            < 1
        {
            return Err(format!("{name}: no span was repacked before the event"));
        }
        super::cases::prefill_to_decode(&mut session, &*authority, name)?;
    }
    let generated_before = session.generated().len();
    let splices_before = session
        .positional_report()
        .map_or(0, |report| report.splices);
    let inserts_before = backend
        .secondary_stats()
        .map_err(fail("secondary stats"))?
        .inserts;
    let entries_before = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries;

    event(authority, backend)?;

    // The next step is the first use after the event.
    let step = match when {
        When::BeforeSplice => session.prefill_chunk(&*authority),
        When::AfterSpliceBeforeDecode => session.decode_step(&*authority),
    }
    .map_err(fail(name))?;
    let (cancelled, reason, emitted) = match &step {
        KvSessionStep::Cancelled(cancellation) => {
            (true, Some(cancellation.reason), cancellation.tokens_emitted)
        }
        KvSessionStep::Prefilled { .. } | KvSessionStep::Token(_) => (false, None, 0),
    };
    let generated_after = session.generated().len().saturating_sub(generated_before);
    let splices_after = session
        .positional_report()
        .map_or(0, |report| report.splices)
        .saturating_sub(splices_before);
    let mut outputs = emitted
        .saturating_sub(generated_before)
        .max(generated_after) as i64
        + splices_after as i64
        + i64::from(!cancelled);
    let mut restart_fresh = None;
    let mut mismatches = 0;
    if cancelled {
        let pins = backend
            .secondary_stats()
            .map_err(fail("secondary stats"))?
            .active_pins;
        if pins != 0 {
            return Err(format!(
                "{name}: {pins} secondary pins are held after the cancellation"
            ));
        }
        // The restart is a cold run: equal to a cold run of the request and to the
        // independent stack, from a fresh prefill.
        let restarted = session.restart_with(&*authority).map_err(fail(name))?;
        let baseline = run_cold(&mut lab.timings, &format!("{name}.cold"), backend, request)?;
        let anchored = lab.anchor.of(request)?;
        let state = session.kv_state_digest().map_err(fail(name))?;
        let equal = restarted.generated == baseline.outcome.generated
            && restarted.logits_digest == baseline.outcome.logits_digest
            && restarted.run.events == baseline.outcome.run.events
            && state == baseline.kv_state_digest;
        let anchor_generated: Vec<u32> = restarted.generated.iter().map(|token| token.0).collect();
        mismatches = i64::from(!equal)
            + i64::from(anchor_generated != anchored.0)
            + i64::from(restarted.logits_digest != anchored.1);
        let positional = restarted
            .positional
            .as_ref()
            .ok_or_else(|| format!("{name}: the restart reports no positional record"))?;
        let fresh = restarted.hit_tokens == 0
            && positional.reused_tokens == 0
            && positional.splices == 0
            && positional.lookups == 0
            && positional.cold_restart
            && restarted.prefilled_tokens == request.total_tokens()
            && restarted.restarted_after_cancel;
        restart_fresh = Some(fresh && mismatches == 0);
        // A restart that cached or looked up anything is output after the event too.
        outputs += i64::from(restarted.inserted_chunks > 0);
    }
    let inserts_after = backend
        .secondary_stats()
        .map_err(fail("secondary stats"))?
        .inserts;
    let entries_after = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries;
    let inserts = inserts_after.saturating_sub(inserts_before) as i64
        + entries_after.saturating_sub(entries_before) as i64;
    Ok(Escaped {
        outputs,
        inserts,
        cancelled,
        reason,
        restart_fresh,
        mismatches,
    })
}

fn describe_escape(when: When, escaped: &Escaped) -> (String, Json) {
    (
        when.name().to_string(),
        Json::obj([
            ("cancelled", Json::Bool(escaped.cancelled)),
            (
                "reason",
                escaped
                    .reason
                    .map_or(Json::Null, |reason| Json::str(format!("{reason:?}"))),
            ),
            ("escaped_outputs", Json::Int(escaped.outputs)),
            ("cache_inserts_after_event", Json::Int(escaped.inserts)),
            (
                "restarted_from_fresh_prefill",
                escaped.restart_fresh.map_or(Json::Null, Json::Bool),
            ),
            ("restart_mismatches", Json::Int(escaped.mismatches)),
        ]),
    )
}

fn inflight_arm(lab: &mut Lab<'_>, subject: Subject, ty: Ty, name: &str, fault: Fault) -> Res<Out> {
    let mut variants = Vec::new();
    let mut stale = 0;
    let mut completed = true;
    for when in [When::BeforeSplice, When::AfterSpliceBeforeDecode] {
        let mut authority = authority(subject)?;
        let backend = lab::positional_backend(lab.shape, fault.control())?;
        let parts = body(lab, subject);
        seed_origin(
            lab,
            &format!("{name}.{}.origin", when.name()),
            &backend,
            &request(&format!("{name}-origin"), &parts),
            &authority,
        )?;
        let racing = request(
            &format!("{name}-{}", when.name()),
            &after(lab.material.spare(0, SHORT_FILLER_TOKENS), &parts),
        );
        lab.ids.invalidated.insert(name.to_string());
        // Before the splice the invalidation reaches the index; after the splice it is lost
        // on the way (a lagging index), so the use-time check alone must stop the session.
        let lag = when == When::AfterSpliceBeforeDecode;
        let event = |authority: &mut Authority, backend: &ReferenceKvReuseBackend| -> Res<()> {
            if lag {
                suppress(backend)?;
            }
            apply_event(subject, ty, authority, backend, fault)
        };
        let escaped = inflight_variant(
            lab,
            &format!("{name}.{}", when.name()),
            &backend,
            &mut authority,
            &racing,
            when,
            &event,
        )?;
        lab.tally
            .add("shifted_reuse_inflight_outputs", escaped.outputs);
        lab.tally
            .add("shifted_reuse_inflight_cache_inserts", escaped.inserts);
        lab.tally.add(
            &format!(
                "shifted_reuse_inflight_race_cases.{}",
                params::inflight_member(cell_kind(subject), ty.name())
            ),
            1,
        );
        lab.tally
            .add("fresh_prefill_output_mismatches", escaped.mismatches);
        completed &= escaped.restart_fresh == Some(true);
        stale += escaped.outputs + escaped.inserts;
        variants.push(describe_escape(when, &escaped));
    }
    if completed {
        lab.ids.completed.insert(name.to_string());
    }
    Ok(Out {
        cases: 2,
        stale,
        detail: Json::Obj(variants),
    })
}

fn cell_kind(subject: Subject) -> &'static str {
    match subject {
        Subject::Segment(spec) => spec.cell_kind,
        Subject::BaseModel => "base_model",
    }
}

/// The in-flight race of the liveness-bypass control: the same measure as the matrix's
/// in-flight arm, on a backend whose use-time liveness is bypassed and whose index lost the
/// base-model revocation. The session must keep going, which the caller requires to trip the
/// in-flight falsifier.
pub fn bypass_inflight(lab: &mut Lab<'_>) -> Res<Escaped> {
    let shape = lab.shape;
    let mut authority = Authority::new(&[lab::C_IDENTITY]).map_err(fail("the bypass authority"))?;
    let backend = lab::positional_backend(shape, PositionalControl::BypassUseTimeLiveness)?;
    let c = lab::c_part(lab.material, shape);
    let tail = lab::tail_part(lab.material, shape);
    let parts = [c, tail];
    seed_origin(
        lab,
        "C10b.inflight.origin",
        &backend,
        &request("c10b-inflight-origin", &parts),
        &authority,
    )?;
    let racing = request(
        "c10b-inflight",
        &after(lab.material.spare(11, SHORT_FILLER_TOKENS), &parts),
    );
    let event = |authority: &mut Authority, backend: &ReferenceKvReuseBackend| -> Res<()> {
        suppress(backend)?;
        apply_event(
            Subject::BaseModel,
            Ty::Revocation,
            authority,
            backend,
            Fault::None,
        )
    };
    inflight_variant(
        lab,
        "C10b.inflight",
        &backend,
        &mut authority,
        &racing,
        When::AfterSpliceBeforeDecode,
        &event,
    )
}

// ---- the declared-dependency arms ----------------------------------------------------------

const ADAPTER_IDENTITY: &str = "adapter:r006-lora";
const EXPERT_LAYER: usize = 0;
const EXPERT_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Declared {
    Adapter,
    Expert,
}

impl Declared {
    fn cell_kind(self) -> &'static str {
        match self {
            Self::Adapter => "adapter_generation",
            Self::Expert => "expert_generation",
        }
    }

    fn kind_name(self) -> &'static str {
        match self {
            Self::Adapter => "adapter",
            Self::Expert => "expert",
        }
    }

    fn dependency(self) -> KvDependency {
        match self {
            Self::Adapter => KvDependency::adapter(ADAPTER_IDENTITY, Generation(1), [0; 32]),
            Self::Expert => KvDependency::expert(
                &ProjectId::from(shared::PROJECT),
                &experts::pod_of(EXPERT_LAYER, EXPERT_INDEX),
                Generation(1),
                [0; 32],
            ),
        }
    }

    /// A real runtime committing the dependency live.
    fn authority(self) -> Res<Authority> {
        match self {
            Self::Adapter => Authority::new(&[lab::C_IDENTITY, ADAPTER_IDENTITY]),
            Self::Expert => {
                let mut authority = Authority::new(&[lab::C_IDENTITY])?;
                experts::birth(&mut authority.runtime, EXPERT_LAYER, EXPERT_INDEX)?;
                Ok(authority)
            }
        }
    }

    /// Commit the event to the ledger.
    fn commit(self, ty: Ty, authority: &mut Authority) -> Res<()> {
        let runtime = &mut authority.runtime;
        match (self, ty) {
            (Self::Adapter, Ty::Revocation) => commit_revocation(runtime, ADAPTER_IDENTITY, 1),
            (Self::Adapter, Ty::Supersession) => {
                commit_supersession(runtime, ADAPTER_IDENTITY, 1, 2)
            }
            (Self::Adapter, Ty::Fence) => Err("only an expert has a fence".to_string()),
            (Self::Expert, Ty::Revocation) => experts::retire(runtime, EXPERT_LAYER, EXPERT_INDEX),
            (Self::Expert, Ty::Supersession) => {
                experts::supersede(runtime, EXPERT_LAYER, EXPERT_INDEX).map(|_| ())
            }
            (Self::Expert, Ty::Fence) => experts::fence(
                runtime,
                EXPERT_LAYER,
                EXPERT_INDEX,
                ExpertFenceCause::AuthorityQuarantine,
                None,
            ),
        }
    }
}

/// A standalone catalog with the pinned window, and the lookup identity of a one-layer backend.
struct Index {
    catalog: KvPrefixCatalog,
    salt: KvSalt,
    chain: KvChainId,
}

impl Index {
    fn new(lab: &Lab<'_>) -> Res<Self> {
        let paged = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
        let authority = paged
            .take_fork_authority()
            .map_err(fail("the fork authority"))?;
        let catalog = KvPrefixCatalog::new(authority, CatalogLimits::default())
            .map_err(fail("a standalone catalog"))?
            .with_secondary_config(ptr_runtime::SecondaryConfig {
                window_tokens: params::ROLLING_WINDOW_TOKENS as usize,
                ..ptr_runtime::SecondaryConfig::default()
            })
            .map_err(fail("the pinned window"))?;
        let backend = lab::positional_backend(lab.shape, PositionalControl::None)?;
        let salt = KvSalt::derive(
            &ptr_types::PrincipalId::from(shared::PRINCIPAL),
            &ProjectId::from(shared::PROJECT),
            &ptr_types::ModelVersion::from(shared::BASE_MODEL_IDENTITY),
            ptr_model_api::Tokenizer::identity(backend.tokenizer()),
            "f32-cpu-reference",
            "paged-kv-v1",
        );
        Ok(Self {
            catalog,
            salt,
            chain: backend.chain_id(),
        })
    }

    /// The layer-0 `PreRope` rows of `tokens`, computed by the independent one-layer stack.
    fn rows(tokens: &[u32]) -> Res<SecondaryRows> {
        let stack = super::pins::reference_stack()?;
        let rows =
            crate::experiments::layered_kv::pins::independent_prefill_rows(&stack, None, tokens)?;
        let (keys, values) = rows
            .layers
            .into_iter()
            .next()
            .ok_or_else(|| "the independent stack left no layer 0".to_string())?;
        Ok(SecondaryRows::new(
            Arc::from(keys.into_boxed_slice()),
            Arc::from(values.into_boxed_slice()),
            rows.kv_width,
        ))
    }

    fn insert(
        &mut self,
        mode: KvReuseMode,
        request: &SegmentedModelRequest,
        segment: usize,
        extras: Vec<KvDependency>,
        oracle: &dyn KvLivenessOracle,
    ) -> Result<SecondaryInsertOutcome, SecondaryError> {
        let tokens: Vec<u32> = request.segments()[segment]
            .tokens()
            .iter()
            .map(|token| token.0)
            .collect();
        let rows = Self::rows(&tokens).map_err(|_| SecondaryError::RowShape)?;
        self.catalog.secondary_mut().insert(
            SecondaryInsert {
                salt: &self.salt,
                chain: &self.chain,
                mode,
                request,
                segment,
                rows,
                extra_dependencies: extras,
            },
            oracle,
        )
    }

    fn lookup(
        &mut self,
        mode: KvReuseMode,
        request: &SegmentedModelRequest,
        oracle: &dyn KvLivenessOracle,
    ) -> Res<SecondaryLookupOutcome> {
        self.catalog
            .secondary_mut()
            .lookup_positional(
                SecondaryLookup {
                    salt: &self.salt,
                    chain: &self.chain,
                    mode,
                    request,
                    positions: &KvRequestPositions::Canonical,
                },
                oracle,
            )
            .map_err(fail("a secondary lookup"))
    }

    fn release(&mut self, outcome: SecondaryLookupOutcome) {
        for splice in outcome.splices {
            self.catalog.secondary_mut().release(splice.pin);
        }
    }

    /// Hand the committed events to the catalog the way the runtime's hook does
    /// (`invalidate_catalog_for_event` is crate-private; this is its mapping for the events a
    /// declared dependency produces).
    fn apply(&mut self, events: &[LedgerEvent]) {
        for event in events {
            match event {
                LedgerEvent::Revoked {
                    subject,
                    generation,
                } => {
                    self.catalog
                        .invalidate_inputs(&BTreeSet::from([(subject.clone(), *generation)]));
                    self.catalog.invalidate_kind_identities(
                        KvDependencyKind::Segment(SegmentKind::Semantic),
                        &BTreeSet::from([subject.clone()]),
                    );
                }
                LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                    self.catalog
                        .invalidate_superseded(&BTreeMap::from([(capsule.to_string(), *new)]));
                }
                LedgerEvent::ExpertStructuralCommitted { record, .. } => {
                    match expert_operand_identities(record) {
                        Some(identities) => {
                            self.catalog
                                .invalidate_identities(&identities.into_iter().collect());
                        }
                        None => {
                            self.catalog.invalidate_all_experts();
                        }
                    }
                }
                LedgerEvent::ExpertFenceCommitted { .. } => {
                    self.catalog
                        .invalidate_expert_fences(&ptr_runtime::expert::expert_fence_epochs(event));
                }
                _ => {}
            }
        }
    }
}

struct DeclaredCell<'a> {
    kind: Declared,
    ty: Ty,
    name: &'a str,
}

/// `[C][tail]`: the request a declared run is inserted from (segment 0 is `C`).
fn declared_origin(lab: &Lab<'_>) -> SegmentedModelRequest {
    request(
        "declared-origin",
        &[
            lab::c_part(lab.material, lab.shape),
            lab::tail_part(lab.material, lab.shape),
        ],
    )
}

/// `[filler][C][tail]`: `C` at a new position.
fn declared_shifted(lab: &Lab<'_>, slot: usize) -> SegmentedModelRequest {
    request(
        &format!("declared-shifted-{slot}"),
        &after(
            lab.material.spare(slot, SHORT_FILLER_TOKENS),
            &[
                lab::c_part(lab.material, lab.shape),
                lab::tail_part(lab.material, lab.shape),
            ],
        ),
    )
}

fn declared_live(
    lab: &Lab<'_>,
    index: &mut Index,
    cell: &DeclaredCell<'_>,
    authority: &Authority,
) -> Res<usize> {
    let origin = declared_origin(lab);
    match index.insert(
        KvReuseMode::Exact,
        &origin,
        0,
        vec![cell.kind.dependency()],
        authority,
    ) {
        Ok(SecondaryInsertOutcome::Inserted(_)) => {}
        other => {
            return Err(format!(
                "{}: the declared run was not inserted: {other:?}",
                cell.name
            ))
        }
    }
    let outcome = index.lookup(KvReuseMode::Exact, &declared_shifted(lab, 0), authority)?;
    let reused = outcome.reused_tokens();
    index.release(outcome);
    if reused < lab.shape.segment {
        return Err(format!(
            "{}: the live declared control reused {reused} tokens, fewer than C's {}",
            cell.name, lab.shape.segment
        ));
    }
    Ok(reused)
}

fn declared_stale(outcome: &SecondaryLookupOutcome, normal: bool) -> i64 {
    if normal {
        (outcome.splices.len() + outcome.refusals.len()) as i64
    } else {
        outcome.splices.len() as i64
    }
}

fn declared_normal(
    lab: &mut Lab<'_>,
    cell: &DeclaredCell<'_>,
    lag: bool,
    fault: Fault,
) -> Res<Out> {
    let mut authority = cell.kind.authority()?;
    let mut index = Index::new(lab)?;
    let control = declared_live(lab, &mut index, cell, &authority)?;
    let mark = authority.mark();
    if fault != Fault::EventLost {
        cell.kind.commit(cell.ty, &mut authority)?;
    }
    if !lag && fault == Fault::None {
        index.apply(&authority.events_since(mark));
    }
    let outcome = index.lookup(KvReuseMode::Exact, &declared_shifted(lab, 1), &authority)?;
    let stale = declared_stale(&outcome, !lag);
    let typed = outcome.refusals_for(SecondaryRefusalReason::BindingDependencyDead);
    if lag && stale == 0 && typed == 0 {
        return Err(format!(
            "{}: the lagging index never returned the stale declared run: the cell tests nothing",
            cell.name
        ));
    }
    let detail = Json::obj([
        ("live_control_reused_tokens", Json::Int(control as i64)),
        ("admitted_spans", Json::Int(outcome.splices.len() as i64)),
        ("refusals", Json::Int(outcome.refusals.len() as i64)),
        ("binding_dependency_dead_refusals", Json::Int(typed as i64)),
    ]);
    index.release(outcome);
    Ok(Out {
        cases: 1,
        stale,
        detail,
    })
}

fn declared_restore(lab: &mut Lab<'_>, cell: &DeclaredCell<'_>, fault: Fault) -> Res<Out> {
    let mut authority = cell.kind.authority()?;
    let mut index = Index::new(lab)?;
    declared_live(lab, &mut index, cell, &authority)?;
    let snapshot = index
        .catalog
        .export_entries()
        .map_err(fail("exporting the catalog"))?;
    let restore = |lab: &Lab<'_>| -> Res<Index> {
        let mut restored = Index::new(lab)?;
        let summary = restored
            .catalog
            .restore_entries(&snapshot)
            .map_err(fail("restoring the catalog"))?;
        if summary.secondary_runs == 0 {
            return Err("the snapshot carried no secondary runs".to_string());
        }
        Ok(restored)
    };
    let mut restored = restore(lab).map_err(|error| format!("{}: {error}", cell.name))?;
    if fault != Fault::EventLost {
        cell.kind.commit(cell.ty, &mut authority)?;
    }
    // The re-insert under the post-event ledger must be refused, typed.
    let origin = declared_origin(lab);
    let mut stale = 0;
    let reinsert = restored.insert(
        KvReuseMode::Exact,
        &origin,
        0,
        vec![cell.kind.dependency()],
        &authority,
    );
    let reinsert_refused = match reinsert {
        Err(SecondaryError::DependencyNotLive) => true,
        Ok(_) => {
            stale += 1;
            false
        }
        Err(other) => {
            return Err(format!(
                "{}: the re-insert failed with the wrong error {other:?}",
                cell.name
            ))
        }
    };
    let outcome = restored.lookup(KvReuseMode::Exact, &declared_shifted(lab, 1), &authority)?;
    stale += declared_stale(&outcome, true);
    restored.release(outcome);
    // Control: everything live, the re-insert attaches the rows and the run is served again.
    let live = cell.kind.authority()?;
    let mut control = restore(lab).map_err(|error| format!("{}: {error}", cell.name))?;
    match control.insert(
        KvReuseMode::Exact,
        &origin,
        0,
        vec![cell.kind.dependency()],
        &live,
    ) {
        Ok(SecondaryInsertOutcome::RowsAttached(_)) => {}
        other => {
            return Err(format!(
                "{}: the live control did not re-attach rows: {other:?}",
                cell.name
            ))
        }
    }
    let served = control.lookup(KvReuseMode::Exact, &declared_shifted(lab, 0), &live)?;
    let control_reused = served.reused_tokens();
    control.release(served);
    if control_reused < lab.shape.segment {
        return Err(format!(
            "{}: the live control served {control_reused} tokens after the re-attach",
            cell.name
        ));
    }
    Ok(Out {
        cases: 1,
        stale,
        detail: Json::obj([
            (
                "reinsert_refused_dependency_not_live",
                Json::Bool(reinsert_refused),
            ),
            ("control_reused_tokens", Json::Int(control_reused as i64)),
        ]),
    })
}

fn declared_inflight(lab: &mut Lab<'_>, cell: &DeclaredCell<'_>, fault: Fault) -> Res<Out> {
    let mut variants = Vec::new();
    let mut stale = 0;
    for when in [When::BeforeSplice, When::AfterSpliceBeforeDecode] {
        let mut authority = cell.kind.authority()?;
        let mut index = Index::new(lab)?;
        let origin = declared_origin(lab);
        match index.insert(
            KvReuseMode::Exact,
            &origin,
            0,
            vec![cell.kind.dependency()],
            &authority,
        ) {
            Ok(SecondaryInsertOutcome::Inserted(_)) => {}
            other => {
                return Err(format!(
                    "{}: the declared run was not inserted: {other:?}",
                    cell.name
                ))
            }
        }
        let mut outcome =
            index.lookup(KvReuseMode::Exact, &declared_shifted(lab, 0), &authority)?;
        let pin = outcome
            .splices
            .pop()
            .map(|splice| splice.pin)
            .ok_or_else(|| {
                format!(
                    "{}: no declared run was admitted, so nothing is in flight",
                    cell.name
                )
            })?;
        index.release(outcome);
        let mut escaped_outputs = 0;
        // A lagging index for the second variant: only the pin's use-time revalidation stops it.
        let lag = when == When::AfterSpliceBeforeDecode;
        if lag {
            // The splice is consumed while the dependency is live.
            if index.catalog.secondary_mut().revalidate(&pin, &authority) != SecondaryPinState::Live
            {
                return Err(format!(
                    "{}: the pin was not live before the event",
                    cell.name
                ));
            }
        }
        let mark = authority.mark();
        if fault != Fault::EventLost {
            cell.kind.commit(cell.ty, &mut authority)?;
        }
        if !lag && fault == Fault::None {
            index.apply(&authority.events_since(mark));
        }
        // The first use after the event: a consumer proceeds only on a live pin.
        if index.catalog.secondary_mut().revalidate(&pin, &authority) == SecondaryPinState::Live {
            escaped_outputs += 1;
        }
        // An insert made after the event on the dead dependency must be refused.
        let mut escaped_inserts = 0;
        let late = request(
            "declared-late",
            &[
                Part::raw(lab.material.spare(2, SHORT_FILLER_TOKENS)),
                lab::tail_part(lab.material, lab.shape),
            ],
        );
        match index.insert(
            KvReuseMode::Exact,
            &late,
            0,
            vec![cell.kind.dependency()],
            &authority,
        ) {
            Err(SecondaryError::DependencyNotLive) => {}
            Ok(_) => escaped_inserts += 1,
            Err(other) => {
                return Err(format!(
                    "{}: the late insert failed with the wrong error {other:?}",
                    cell.name
                ))
            }
        }
        index.catalog.secondary_mut().release(pin);
        stale += escaped_outputs + escaped_inserts;
        variants.push((
            when.name().to_string(),
            Json::obj([
                ("escaped_consumptions", Json::Int(escaped_outputs)),
                ("inserts_after_event", Json::Int(escaped_inserts)),
                ("index_lagged", Json::Bool(lag)),
            ]),
        ));
    }
    Ok(Out {
        cases: 2,
        stale,
        detail: Json::Obj(variants),
    })
}

/// The production mode partitions the index and only Exact is admitted (C10d).
pub fn mode_partition(lab: &mut Lab<'_>) -> Res<Json> {
    let authority = Declared::Adapter.authority()?;
    let mut index = Index::new(lab)?;
    let origin = declared_origin(lab);
    match index.insert(KvReuseMode::Exact, &origin, 0, Vec::new(), &authority) {
        Ok(SecondaryInsertOutcome::Inserted(_)) => {}
        other => {
            return Err(format!(
                "mode partition: the Exact run was not inserted: {other:?}"
            ))
        }
    }
    let refused = index.insert(KvReuseMode::Approximate, &origin, 0, Vec::new(), &authority);
    if refused != Err(SecondaryError::NotExact) {
        return Err(format!(
            "mode partition: an Approximate insert was not refused as NotExact: {refused:?}"
        ));
    }
    let shifted = declared_shifted(lab, 0);
    let exact = index.lookup(KvReuseMode::Exact, &shifted, &authority)?;
    let approximate = index.lookup(KvReuseMode::Approximate, &shifted, &authority)?;
    let result = Json::obj([
        (
            "exact_lookup_reused_tokens",
            Json::Int(exact.reused_tokens() as i64),
        ),
        (
            "approximate_lookup_spans",
            Json::Int(approximate.splices.len() as i64),
        ),
        (
            "approximate_lookup_refusals",
            Json::Int(approximate.refusals.len() as i64),
        ),
    ]);
    if exact.reused_tokens() < lab.shape.segment
        || !approximate.splices.is_empty()
        || !approximate.refusals.is_empty()
    {
        return Err(format!(
            "mode partition: the namespaces are not partitioned by mode: {}",
            result.render()
        ));
    }
    index.release(exact);
    index.release(approximate);
    lab.observed.production_modes.insert("Exact");
    lab.observed.production_modes.insert("Approximate");
    Ok(result)
}

// ---- the matrix ------------------------------------------------------------------------------

fn session_subject(kind: &str) -> Option<Subject> {
    SEGMENTS
        .iter()
        .find(|spec| spec.cell_kind == kind)
        .map(Subject::Segment)
        .or_else(|| (kind == "base_model").then_some(Subject::BaseModel))
}

fn declared_subject(kind: &str) -> Option<Declared> {
    match kind {
        "adapter_generation" => Some(Declared::Adapter),
        "expert_generation" => Some(Declared::Expert),
        _ => None,
    }
}

/// The types of a kind: the expert has the fence besides the two common ones.
fn types_of(kind: &str) -> Vec<&'static str> {
    let mut types: Vec<&'static str> = params::INVALIDATION_COVERAGE_MATRIX_TYPES.to_vec();
    if kind == "expert_generation" {
        types.extend(params::INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION);
    }
    types
}

/// One cell: its counters, the evidence label and the oracle it ran under.
struct CellRun {
    out: Out,
    evidence: &'static str,
    oracle: &'static str,
}

/// Run one cell of `kind` and `path` under `ty` with `fault` injected.
fn run_cell(
    lab: &mut Lab<'_>,
    kind: &str,
    path: &str,
    ty: Ty,
    name: &str,
    fault: Fault,
) -> Res<CellRun> {
    if let Some(subject) = session_subject(kind) {
        let out = match path {
            "normal" => normal_arm(lab, subject, ty, name, fault)?,
            "lag" => lag_arm(lab, subject, ty, name, fault)?,
            "restore" => restore_arm(lab, subject, ty, name, fault)?,
            "inflight" => inflight_arm(lab, subject, ty, name, fault)?,
            other => return Err(format!("unknown path {other}")),
        };
        let oracle = match subject {
            Subject::Segment(spec) if spec.kind == SegmentKind::Semantic => "ptr-runtime committed semantic state (host write; Revoked for a revocation, a committed edit for a supersession)",
            _ => "ptr-runtime ledger (CapsuleCommitted / Revoked / CapsuleSuperseded)",
        };
        lab.observed.declared_kinds.insert(match kind {
            "semantic_lineage_segment" => "semantic",
            "memory_lineage_segment" => "memory",
            "derived_lineage_segment" => "derived",
            _ => "base_model",
        });
        return Ok(CellRun {
            out,
            evidence: subject.evidence(),
            oracle,
        });
    }
    let declared = declared_subject(kind).ok_or_else(|| format!("unknown matrix kind {kind}"))?;
    let cell = DeclaredCell {
        kind: declared,
        ty,
        name,
    };
    let out = match path {
        "normal" => declared_normal(lab, &cell, false, fault)?,
        "lag" => declared_normal(lab, &cell, true, fault)?,
        "restore" => declared_restore(lab, &cell, fault)?,
        "inflight" => declared_inflight(lab, &cell, fault)?,
        other => return Err(format!("unknown path {other}")),
    };
    lab.observed.declared_kinds.insert(declared.kind_name());
    Ok(CellRun {
        out,
        evidence: match declared {
            Declared::Adapter => params::EVIDENCE_LABELS_ADAPTER_GENERATION,
            Declared::Expert => params::EVIDENCE_LABELS_EXPERT_GENERATION,
        },
        oracle: match declared {
            Declared::Adapter => "ptr-runtime ledger (an adapter capsule: CapsuleCommitted / Revoked / CapsuleSuperseded)",
            Declared::Expert => "ptr-runtime expert registry (real admission path: birth, Retire, Reactivate, fence incl. the fence epoch)",
        },
    })
}

/// The stale hits of one cell run under `fault`, for the tests that show the cell's counters
/// trip when the protection is broken.
#[cfg(test)]
pub(crate) fn stale_under_fault(
    lab: &mut Lab<'_>,
    kind: &str,
    path: &str,
    ty_name: &str,
    fault: Fault,
) -> Res<i64> {
    let ty = Ty::parse(ty_name)?;
    let name = params::cell_name(kind, path, ty_name);
    Ok(run_cell(lab, kind, path, ty, &name, fault)?.out.stale)
}

/// Run every cell of `invalidation_coverage_matrix` once per seed and count it per cell.
pub fn run_all(lab: &mut Lab<'_>) -> Res<()> {
    let mut ran: Vec<String> = Vec::new();
    let mut infos = Vec::new();
    for kind in params::INVALIDATION_COVERAGE_MATRIX_KINDS {
        for path in params::INVALIDATION_COVERAGE_MATRIX_PATHS {
            for ty_name in types_of(kind) {
                let ty = Ty::parse(ty_name)?;
                let name = params::cell_name(kind, path, ty_name);
                let CellRun {
                    out,
                    evidence,
                    oracle,
                } = run_cell(lab, kind, path, ty, &name, Fault::None)?;
                lab.tally
                    .add(&format!("invalidation_matrix_cases.{name}"), out.cases);
                lab.tally
                    .add(&format!("invalidation_matrix_stale_hits.{name}"), out.stale);
                infos.push(Json::obj([
                    ("cell", Json::str(name.clone())),
                    ("evidence", Json::str(evidence)),
                    ("oracle", Json::str(oracle)),
                    ("case_runs", Json::Int(out.cases)),
                    ("stale_hits", Json::Int(out.stale)),
                    ("detail", out.detail),
                ]));
                ran.push(name);
            }
        }
    }
    let expected: BTreeSet<&str> = params::INVALIDATION_COVERAGE_MATRIX_CELLS
        .iter()
        .copied()
        .collect();
    let got: BTreeSet<&str> = ran.iter().map(String::as_str).collect();
    if got != expected || ran.len() != 52 {
        return Err(format!(
            "the matrix ran {} cells that are not the 52 of config.toml",
            ran.len()
        ));
    }
    lab.cases.push((
        "C9".to_string(),
        Json::obj([
            ("what", Json::str("the 52 cells of invalidation_coverage_matrix: dependency kind x invalidation type x path (normal, lag, restore, in-flight), the expert also under a fence; semantic, memory, derived segments and the base model on real positional sessions under a real ledger oracle, adapter and expert as declared dependencies of runs inserted into a standalone catalog under real oracles; in-flight cells in two variants (event before the splice, event after the splice and before decode)")),
            ("cells", Json::Arr(infos)),
            (
                "case_id_sets",
                Json::str("the invalidated and fresh-prefill-completed case-id sets cover C3, C4 and the cells that run a session (a fresh prefill needs one); the declared adapter and expert cells have no session on the declared index and are not in either set"),
            ),
        ]),
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::lab::test_support::with_lab;
    use super::*;

    fn stale(kind: &str, path: &str, ty: &str, fault: Fault) -> i64 {
        with_lab(|lab| stale_under_fault(lab, kind, path, ty, fault).expect("the cell runs"))
    }

    #[test]
    fn the_matrix_kinds_types_and_paths_are_those_of_the_config() {
        let mut cells = Vec::new();
        for kind in params::INVALIDATION_COVERAGE_MATRIX_KINDS {
            for path in params::INVALIDATION_COVERAGE_MATRIX_PATHS {
                for ty in types_of(kind) {
                    cells.push(params::cell_name(kind, path, ty));
                }
            }
        }
        assert_eq!(cells.len(), 52);
        let configured: BTreeSet<&str> = cell_names().into_iter().collect();
        assert!(cells.iter().all(|cell| configured.contains(cell.as_str())));
        assert_eq!(
            session_subject("base_model").map(Subject::evidence),
            Some("session_recorded_dependency")
        );
        assert!(session_subject("adapter_generation").is_none());
        assert!(declared_subject("expert_generation").is_some());
        assert!(Ty::parse("unknown").is_err());
    }

    /// The cells have teeth: with the protection broken, the counter of the cell trips.
    #[test]
    fn a_lost_catalog_hook_leaves_the_page_reachable_and_the_normal_cell_counts_it() {
        for (kind, ty) in [
            ("semantic_lineage_segment", "revocation"),
            ("memory_lineage_segment", "supersession"),
            ("base_model", "revocation"),
        ] {
            assert!(
                stale(kind, "normal", ty, Fault::LostFanout) > 0,
                "{kind} {ty}"
            );
        }
        for (kind, ty) in [
            ("adapter_generation", "revocation"),
            ("expert_generation", "fence"),
        ] {
            assert!(
                stale(kind, "normal", ty, Fault::LostFanout) > 0,
                "{kind} {ty}"
            );
        }
    }

    #[test]
    fn a_bypassed_use_time_check_admits_the_dead_page_and_the_lag_cell_counts_it() {
        for (kind, ty) in [
            ("derived_lineage_segment", "revocation"),
            ("semantic_lineage_segment", "supersession"),
            ("base_model", "supersession"),
        ] {
            assert!(stale(kind, "lag", ty, Fault::Bypass) > 0, "{kind} {ty}");
        }
    }

    #[test]
    fn an_event_that_never_reached_the_ledger_lets_the_restore_cell_count_attached_rows() {
        assert!(
            stale(
                "derived_lineage_segment",
                "restore",
                "supersession",
                Fault::EventLost
            ) > 0
        );
        assert!(stale("base_model", "restore", "revocation", Fault::EventLost) > 0);
        assert!(
            stale(
                "adapter_generation",
                "restore",
                "revocation",
                Fault::EventLost
            ) > 0
        );
    }

    #[test]
    fn a_session_that_nobody_stops_emits_after_the_event_and_the_inflight_cell_counts_it() {
        for (kind, fault) in [
            ("semantic_lineage_segment", Fault::EventLost),
            ("base_model", Fault::EventLost),
            ("base_model", Fault::Bypass),
        ] {
            assert!(
                stale(kind, "inflight", "revocation", fault) > 0,
                "{kind} {fault:?}"
            );
        }
        assert!(stale("expert_generation", "inflight", "fence", Fault::EventLost) > 0);
        assert!(
            stale(
                "adapter_generation",
                "inflight",
                "supersession",
                Fault::EventLost
            ) > 0
        );
    }

    #[test]
    fn without_a_fault_the_cells_count_nothing() {
        for (kind, path, ty) in [
            ("memory_lineage_segment", "normal", "revocation"),
            ("base_model", "inflight", "supersession"),
            ("expert_generation", "lag", "fence"),
            ("adapter_generation", "restore", "supersession"),
        ] {
            assert_eq!(stale(kind, path, ty, Fault::None), 0, "{kind} {path} {ty}");
        }
    }

    #[test]
    fn the_mode_partition_and_the_bypass_control_run() {
        with_lab(|lab| {
            let json = mode_partition(lab).unwrap();
            assert!(json.render().contains("\"approximate_lookup_spans\":0"));
            assert!(lab.observed.production_modes.contains("Approximate"));
            let escaped = bypass_inflight(lab).unwrap();
            assert!(!escaped.cancelled);
            assert!(escaped.outputs > 0);
        });
    }
}
