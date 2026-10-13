//! The cases C1 to C8 and C10 of R006's preregistration (C9 is the matrix, C11 the pins
//! in `mod.rs`), run once per seed on one-layer reference backends with positional reuse
//! opted in.
//!
//! Every count is read from the runtime's own outcome (`ReferenceKvOutcome.hit_tokens`,
//! `PositionalReport`, `secondary_stats`), never from `KvReuseReport::hit_tokens()`, which
//! folds the repacked tokens into one number, and never asserted from a constant. A refusal
//! counts only when the runtime returned the expected typed reason at its boundary; a refusal
//! of another reason where one was expected, a fixture that does not build, a live control that
//! does not reuse, or a negative control that does not trip is an `Err` (exit status 2). A
//! defect the experiment is designed to detect (a dead, omitted or mismatching run that is
//! admitted, a bitwise difference) is a counter the criteria judge, never an `Err`.
//!
//! Staleness is structural (owner decision 5): a stale hit is a lookup that returned or
//! admitted a run with a dead dependency, an escaped output is tokens emitted (or spans
//! repacked) or cache inserts made after the event step. No output comparison is used for
//! them, because in one layer a stale page is bit-identical to a fresh one.

use super::lab::{
    self, admitted, anchor_mismatches, compare, control, digest_text, exact_and_untainted,
    fresh_check, reachable, report, run_cold, run_positional, run_warm, Lab, Role, Run,
    SHORT_FILLER_TOKENS,
};
use super::metrics;
use super::params;
use crate::experiments::layered_kv::corpus::{self, Part};
use crate::experiments::layered_kv::emit::{self, Json};
use crate::experiments::layered_kv::lab::{
    commit_revocation, commit_supersession, fail, Authority, Res,
};
use ptr_model_api::{SegmentKind, SegmentedModelRequest};
use ptr_runtime::{
    KvLivenessOracle, KvSessionPhase, KvSessionStep, PositionalControl, ReferenceKvError,
    ReferenceKvProbe, ReferenceKvReuseBackend, SecondaryRefusalCounts, SecondaryRefusalReason,
};

/// `request` of `parts` under a case id.
pub(crate) fn request(id: &str, parts: &[Part]) -> SegmentedModelRequest {
    corpus::request(id, parts)
}

/// `[filler][parts...]`.
pub(crate) fn after(filler: Vec<u32>, parts: &[Part]) -> Vec<Part> {
    let mut all = vec![Part::raw(filler)];
    all.extend(parts.iter().cloned());
    all
}

/// Count one run against its cold baseline into the exact-result counters: a result whose mode
/// is not Exact or whose taint is not Exact is a tainted Exact result; an Exact result that
/// differs from the baseline in any compared bit is a non-bit-exact Exact result.
pub(crate) fn note_exact(lab: &mut Lab<'_>, probe: &ReferenceKvProbe, mismatches: i64) -> Res<()> {
    let (exact, untainted) = exact_and_untainted(probe)?;
    if !(exact && untainted) {
        lab.tally.add("tainted_exact_results", 1);
    }
    if exact && mismatches > 0 {
        lab.tally.add("non_bit_exact_exact_results", 1);
    }
    Ok(())
}

/// Run the cold baseline and the independent anchor of `request` and compare `probe` with
/// both. Returns the number of mismatches (digest, logits, tokens, decode, non-finite; plus the
/// anchor's).
pub(crate) fn against_baseline(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    probe: &ReferenceKvProbe,
) -> Res<i64> {
    let baseline = run_cold(&mut lab.timings, &format!("{name}.cold"), backend, request)?;
    let anchored = lab.anchor.of(request)?;
    let total = compare(probe, &baseline).total() + anchor_mismatches(probe, &anchored);
    note_exact(lab, probe, total)?;
    Ok(total)
}

pub(crate) fn case_json(probe: &ReferenceKvProbe) -> Res<Json> {
    let outcome = &probe.outcome;
    let positional = report(probe)?;
    Ok(Json::obj([
        ("prompt_tokens", Json::Int(outcome.prompt_tokens as i64)),
        ("primary_hit_tokens", Json::Int(outcome.hit_tokens as i64)),
        (
            "repacked_tokens",
            Json::Int(positional.reused_tokens as i64),
        ),
        (
            "prefilled_tokens",
            Json::Int(outcome.prefilled_tokens as i64),
        ),
        (
            "admitted_splices",
            Json::Int(positional.admitted_splices as i64),
        ),
        ("splices", Json::Int(positional.splices as i64)),
        ("lookups", Json::Int(positional.lookups as i64)),
        (
            "inserted_segments",
            Json::Int(positional.inserted_segments as i64),
        ),
        (
            "insert_refusals",
            Json::Int(positional.insert_refusals.len() as i64),
        ),
        ("refusals", refusals_json(&positional.refusals)),
        (
            "kv_state_digest",
            Json::str(digest_text(&probe.kv_state_digest)),
        ),
    ]))
}

fn refusals_json(counts: &SecondaryRefusalCounts) -> Json {
    Json::Obj(
        SecondaryRefusalReason::ALL
            .iter()
            .map(|reason| {
                (
                    reason.code().to_string(),
                    Json::Int(counts.get(*reason) as i64),
                )
            })
            .collect(),
    )
}

/// The origin run of a case: it populates the index, so it must have inserted every segment
/// (an insert refusal on the honest path makes the state of the comparison unknown).
pub(crate) fn seed_origin(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    oracle: &dyn KvLivenessOracle,
) -> Run<ReferenceKvProbe> {
    let origin = run_positional(&mut lab.timings, name, backend, request, oracle)?;
    let positional = report(&origin)?;
    // An origin that looked up twice or reused something is a harness fixture that is not an
    // origin; an index that did not take the origin's segments is a runtime that does not
    // populate it, so the case's premise fails (a failed control, not an exit status 2).
    if positional.lookups != 1 || positional.reused_tokens != 0 {
        return Err(format!(
            "{name}: the origin run was not a first run on an empty index: {positional:?}"
        )
        .into());
    }
    if !positional.insert_refusals.is_empty()
        || positional.inserted_segments != request.segments().len()
    {
        return control(format!(
            "{name}: the origin run did not populate the index: {positional:?}"
        ));
    }
    Ok(origin)
}

// ---- C1 and C2: shifted reuse is bit-identical ----------------------------------------------

/// What one shifted comparison counts into the per-shift members.
struct Shifted {
    mismatches: i64,
    reused: usize,
    admitted: i64,
    parity: bool,
}

/// Count a shifted warm run against its baseline and anchor into the member `member` of the
/// per-shift metrics that apply to it.
fn judge_shifted(
    lab: &mut Lab<'_>,
    name: &str,
    request: &SegmentedModelRequest,
    warm: &ReferenceKvProbe,
    baseline: &ReferenceKvProbe,
    member: &str,
) -> Res<Shifted> {
    let shape = lab.shape;
    let anchored = lab.anchor.of(request)?;
    let cmp = compare(warm, baseline);
    let anchor = anchor_mismatches(warm, &anchored);
    let anchor_of_baseline = anchor_mismatches(baseline, &anchored);
    // The cold baseline itself must equal the independent stack: otherwise the baseline is
    // not an anchor and the case cannot be judged.
    if anchor_of_baseline != 0 {
        return Err(format!(
            "{name}: the backend's cold run differs from the independent stack ({anchor_of_baseline})"
        ));
    }
    let total = cmp.total() + anchor;
    lab.tally
        .add(&format!("bitwise_mismatch_count.{member}"), total);
    note_exact(lab, warm, total)?;
    let positional = report(warm)?;
    let (exact, untainted) = exact_and_untainted(warm)?;
    Ok(Shifted {
        mismatches: total,
        reused: positional.reused_tokens,
        // A positive case: the spliced spans only (the last-segment skip is not an admission
        // the case expects or counts).
        admitted: positional.admitted_splices as i64,
        // The reuse unit is the whole segment (owner decision 7): a partial reuse of one
        // token is not parity.
        parity: exact && untainted && total == 0 && positional.reused_tokens == shape.segment,
    })
}

/// C1: the shifted reuse of `C` after a filler of each pinned shift.
pub fn shift_cases(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let mut latency: Vec<(String, Json)> = Vec::new();
    let mut records: Vec<(String, Json)> = Vec::new();
    for (index, pinned) in params::POSITION_SHIFT_TOKENS.iter().enumerate() {
        let member = params::shift_member(*pinned);
        let shift = shape.shifts[index] as usize;
        lab.observed.shifts.push(shape.shifts[index]);
        let authority = Authority::new(&[lab::C_IDENTITY]).map_err(fail("the C1 authority"))?;
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        lab.observed.segment_tokens = Some(lab.material.segment(shape).len() as u64);
        lab.observed.tail_tokens = Some(lab.material.tail(shape).len() as u64);

        let origin = request(&format!("c1-{member}-origin"), &[c.clone(), tail.clone()]);
        seed_origin(
            lab,
            &format!("C1.{member}.origin"),
            &backend,
            &origin,
            &authority,
        )?;

        let shifted = request(
            &format!("c1-{member}"),
            &after(lab.material.filler(shift), &[c, tail]),
        );
        let baseline = run_cold(
            &mut lab.timings,
            &format!("C1.{member}.baseline_cold"),
            &backend,
            &shifted,
        )?;
        let warm = run_positional(
            &mut lab.timings,
            &format!("C1.{member}.shifted_warm"),
            &backend,
            &shifted,
            &authority,
        )?;
        let warm_ms = lab.timings.last();
        let judged = judge_shifted(
            lab,
            &format!("C1.{member}"),
            &shifted,
            &warm,
            &baseline,
            &member,
        )?;
        let positional = report(&warm)?;
        if positional.lookups == 1 {
            lab.tally
                .add(&format!("reuse_attempts_per_shift.{member}"), 1);
        }
        lab.tally.add(
            &format!("reused_tokens_per_shift.{member}"),
            positional.reused_tokens as i64,
        );
        // The shifted run must have been served from the index, not recomputed, and the whole
        // segment must have been reused (owner decision 7: the reuse unit is a whole admitted
        // segment): everything the shift and the tail need is prefilled and nothing else.
        let expected_prefilled = shift + shape.tail;
        let under_reused = warm
            .outcome
            .prefilled_tokens
            .saturating_sub(expected_prefilled);
        lab.tally.add("under_reused_tokens", under_reused as i64);
        if positional.admitted_splices >= 1
            && positional.splices >= 1
            && positional.reused_tokens == shape.segment
            && under_reused == 0
        {
            lab.tally.add(
                &format!("secondary_index_reuse_cases_per_shift.{member}"),
                1,
            );
        }
        lab.tally.add(
            "secondary_index_lookups",
            backend
                .secondary_stats()
                .map_err(fail("secondary stats"))?
                .lookups as i64,
        );
        if judged.parity {
            lab.tally.add("exact_shifted_parity_cases", 1);
        }
        let mut record = vec![
            ("shift_tokens".to_string(), Json::Int(shift as i64)),
            ("pinned_shift".to_string(), Json::Int(*pinned as i64)),
            ("shifted".to_string(), case_json(&warm)?),
            (
                "baseline_kv_state_digest".to_string(),
                Json::str(digest_text(&baseline.kv_state_digest)),
            ),
            ("mismatches".to_string(), Json::Int(judged.mismatches)),
            ("admitted_splices".to_string(), Json::Int(judged.admitted)),
            (
                "expected_prefilled_tokens".to_string(),
                Json::Int(expected_prefilled as i64),
            ),
            (
                "under_reused_tokens".to_string(),
                Json::Int(under_reused as i64),
            ),
        ];
        record.push((
            "repacked_tokens".to_string(),
            Json::Int(judged.reused as i64),
        ));
        records.push((member.clone(), Json::Obj(record)));
        latency.push((member, Json::Float(warm_ms)));
    }
    lab.tally.set_json("reuse_latency_ms", Json::Obj(latency));
    lab.cases.push((
        "C1".to_string(),
        Json::obj([
            ("what", Json::str("for each pinned shift: the origin run [C, tail] populates the index; a cold baseline and the shifted warm run of [filler, C, tail] through the secondary index are compared on KV-state digest, logits bits, generated tokens and decode digest, and with the independent one-layer stack's cold run")),
            ("shifts", Json::Obj(records)),
        ]),
    ));
    Ok(())
}

/// C2: whole-segment insertions of 1 and 7 tokens at an unaligned offset.
pub fn nonaligned_cases(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let mut records: Vec<(String, Json)> = Vec::new();
    for inserted in shape.inserted {
        let member = params::insertion_member(inserted);
        let shift_member = params::shift_member(inserted);
        lab.observed.inserted.push(inserted);
        let authority = Authority::new(&[lab::C_IDENTITY]).map_err(fail("the C2 authority"))?;
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        let a = Part::raw(lab.material.prefix_a(shape));
        lab.observed.origin_prefix_tokens = Some(lab.material.prefix_a(shape).len() as u64);
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        let origin = request(
            &format!("c2-{member}-origin"),
            &[a.clone(), c.clone(), tail.clone()],
        );
        seed_origin(
            lab,
            &format!("C2.{member}.origin"),
            &backend,
            &origin,
            &authority,
        )?;
        let inserted_tokens = lab.material.inserted(inserted as usize);
        let shifted = request(
            &format!("c2-{member}"),
            &[a, Part::raw(inserted_tokens.clone()), c, tail],
        );
        let (origin_start, start) = (start_of_segment(&origin, 1), start_of_segment(&shifted, 2));
        if origin_start % params::PAGE_TOKENS as usize == 0
            || start % params::PAGE_TOKENS as usize == 0
            || start != origin_start + inserted as usize
        {
            return Err(format!(
                "C2 {member}: C moved from {origin_start} to {start}; both must be unaligned and differ by the insertion"
            ).into());
        }
        if inserted_tokens.len() as u64 != inserted {
            return Err(format!(
                "C2 {member}: the inserted segment has {} tokens",
                inserted_tokens.len()
            )
            .into());
        }
        let baseline = run_cold(
            &mut lab.timings,
            &format!("C2.{member}.baseline_cold"),
            &backend,
            &shifted,
        )?;
        let warm = run_positional(
            &mut lab.timings,
            &format!("C2.{member}.shifted_warm"),
            &backend,
            &shifted,
            &authority,
        )?;
        let judged = judge_shifted(
            lab,
            &format!("C2.{member}"),
            &shifted,
            &warm,
            &baseline,
            &shift_member,
        )?;
        let positional = report(&warm)?;
        if positional.splices >= 1 && positional.reused_tokens == shape.segment {
            lab.tally
                .add(&format!("nonaligned_insertion_cases.{member}"), 1);
        }
        lab.tally.add(
            "secondary_index_lookups",
            backend
                .secondary_stats()
                .map_err(fail("secondary stats"))?
                .lookups as i64,
        );
        if judged.parity {
            lab.tally.add("exact_shifted_parity_cases", 1);
        }
        records.push((
            member,
            Json::obj([
                ("origin_start", Json::Int(origin_start as i64)),
                ("shifted_start", Json::Int(start as i64)),
                ("shifted", case_json(&warm)?),
                ("mismatches", Json::Int(judged.mismatches)),
            ]),
        ));
    }
    lab.cases.push((
        "C2".to_string(),
        Json::obj([
            ("what", Json::str("origin [A, C, tail] with A a Raw segment of 24 tokens; the request inserts a Raw segment of 1 or 7 tokens between A and C, so C moves from 24 to 25 or 31 (neither a multiple of the 16-token page); compared as C1")),
            ("insertions", Json::Obj(records)),
        ]),
    ));
    Ok(())
}

fn start_of_segment(request: &SegmentedModelRequest, index: usize) -> usize {
    lab::start_of(request, index)
}

// ---- C3 and C4: revocation and supersession of the upstream segment --------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    Revocation,
    Supersession,
}

impl Event {
    pub fn name(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
        }
    }
}

pub(crate) fn commit_upstream_event(authority: &mut Authority, event: Event) -> Res<()> {
    match event {
        Event::Revocation => commit_revocation(
            &mut authority.runtime,
            params::UPSTREAM_IDENTITY,
            params::UPSTREAM_GENERATION,
        ),
        Event::Supersession => commit_supersession(
            &mut authority.runtime,
            params::UPSTREAM_IDENTITY,
            params::UPSTREAM_GENERATION,
            params::UPSTREAM_GENERATION + 1,
        ),
    }
}

/// C3 (event = revocation) and C4 (supersession): a page seeded on `U`, `U` invalidated
/// through the ledger and the backend's catalog hook, then post-event lookups with matching
/// content and a fresh prefill.
pub fn upstream_event_case(lab: &mut Lab<'_>, event: Event) -> Run<()> {
    let shape = lab.shape;
    let case = match event {
        Event::Revocation => "C3",
        Event::Supersession => "C4",
    };
    let id = format!("{case}.{}", event.name());
    let mut authority = Authority::new(&[
        lab::C_IDENTITY,
        params::UPSTREAM_IDENTITY,
        lab::BYSTANDER_IDENTITY,
    ])
    .map_err(fail("the upstream authority"))?;
    let backend = lab::positional_backend(shape, PositionalControl::None)?;
    let u = lab::u_part(lab.material, params::UPSTREAM_GENERATION);
    let c = lab::c_part(lab.material, shape);
    let tail = lab::tail_part(lab.material, shape);
    let body = [u.clone(), c.clone(), tail.clone()];

    // The page is seeded before the event: every run of the origin depends on `U`.
    let origin = request(&format!("{case}-origin"), &body);
    let seeded = seed_origin(lab, &format!("{id}.origin"), &backend, &origin, &authority)?;
    let seeded_runs = report(&seeded)?.inserted_segments as i64;
    // A bystander: a run that does not depend on `U` at all, seeded beside the page. After the
    // event it must still be reusable, else the runtime flushes more than the event names.
    let bystander = bystander_parts(lab);
    seed_origin(
        lab,
        &format!("{id}.bystander_origin"),
        &backend,
        &request(&format!("{case}-bystander-origin"), &bystander),
        &authority,
    )?;
    // A live control: while `U` is live, the same content at a new position is reused in full
    // and bit-identically to a cold run. A miss here would make every post-event miss below
    // an absent page, not the event's doing.
    let live_request = request(
        &format!("{case}-live"),
        &after(lab.material.spare(0, SHORT_FILLER_TOKENS), &body),
    );
    let live = run_positional(
        &mut lab.timings,
        &format!("{id}.live_control"),
        &backend,
        &live_request,
        &authority,
    )?;
    let live_reused = report(&live)?.reused_tokens;
    against_baseline(
        lab,
        &format!("{id}.live_control"),
        &backend,
        &live_request,
        &live,
    )?;
    let servable = lab::reusable_tokens(&live_request);
    if live_reused != servable {
        return control(format!(
            "{id}: the live control repacked {live_reused} tokens, not the {servable} it can serve (U and C)"
        ));
    }

    // The event: committed to the ledger, then handed to the backend's hook.
    let mark = authority.mark();
    commit_upstream_event(&mut authority, event)?;
    crate::experiments::layered_kv::lab::apply_events(&backend, &authority.events_since(mark))
        .map_err(fail("applying the event"))?;
    lab.ids.invalidated.insert(id.clone());

    // Post-event lookups with the content of the seeded page: the same segments (the old
    // generation of `U`, and for a supersession also the new generation with the same
    // tokens), each at a new position.
    let mut variants: Vec<(String, Vec<Part>)> = vec![(
        "stale_generation".to_string(),
        after(lab.material.spare(1, SHORT_FILLER_TOKENS), &body),
    )];
    if event == Event::Supersession {
        variants.push((
            "new_generation".to_string(),
            after(
                lab.material.spare(2, SHORT_FILLER_TOKENS),
                &[
                    lab::u_part(lab.material, params::UPSTREAM_GENERATION + 1),
                    c.clone(),
                    tail.clone(),
                ],
            ),
        ));
    }
    let mut records = Vec::new();
    let mut all_fresh = true;
    for (variant, parts) in variants {
        let post = request(&format!("{case}-post-{variant}"), &parts);
        if !matches_seeded(&post, &origin) {
            return Err(
                format!("{id}: the {variant} request does not carry the seeded content").into(),
            );
        }
        let probe = run_positional(
            &mut lab.timings,
            &format!("{id}.post_{variant}"),
            &backend,
            &post,
            &authority,
        )?;
        let positional = report(&probe)?;
        if positional.lookups != 1 {
            return Err(format!(
                "{id}: the post-event run made {} lookups",
                positional.lookups
            )
            .into());
        }
        let returned = reachable(positional);
        match event {
            Event::Revocation => {
                lab.tally
                    .add("post_revocation_secondary_lookups_matching_content", 1);
                lab.tally
                    .add("revoked_upstream_page_reachable_count", returned);
            }
            Event::Supersession => {
                lab.tally
                    .add("post_supersession_secondary_lookups_matching_content", 1);
                lab.tally
                    .add("superseded_upstream_page_reachable_count", returned);
            }
        }
        let fresh = fresh_check(
            &mut lab.timings,
            lab.anchor,
            &format!("{id}.{variant}"),
            &backend,
            &post,
            &probe,
        )?;
        lab.tally
            .add("fresh_prefill_output_mismatches", fresh.mismatches);
        note_exact(lab, &probe, fresh.mismatches)?;
        all_fresh &= fresh.completed;
        // A post-event request that ran as a fresh prefill (a per-request fact; the case-level
        // `fresh_prefills_completed_after_revocation` below is per case).
        if fresh.completed && event == Event::Revocation {
            lab.tally.add("fresh_prefill_after_revocation", 1);
        }
        records.push((
            variant,
            Json::obj([
                ("run", case_json(&probe)?),
                ("reachable", Json::Int(returned)),
                ("fresh_prefill_completed", Json::Bool(fresh.completed)),
                ("fresh_prefill_mismatches", Json::Int(fresh.mismatches)),
            ]),
        ));
    }
    // The case completed by a fresh prefill iff every post-event request did.
    if all_fresh {
        lab.ids.completed.insert(id.clone());
        if event == Event::Revocation {
            lab.tally
                .add("fresh_prefills_completed_after_revocation", 1);
        }
    }
    let spared = bystander_check(lab, &id, &backend, &authority, &bystander)?;
    match event {
        Event::Revocation => {
            lab.tally.add("upstream_revocation_cases", 1);
            lab.tally
                .add("old_upstream_pages_seeded_before_revocation", seeded_runs);
        }
        Event::Supersession => lab.tally.add("upstream_supersession_cases", 1),
    }
    lab.cases.push((
        case.to_string(),
        Json::obj([
            ("what", Json::str("a page seeded on U (origin [U, C, tail]) and reused while U is live; U invalidated through the ledger and the backend's hook; post-event positional lookups with the seeded content must reach nothing (neither admitted nor refused: the run is gone), and the request completes by a fresh prefill equal to a cold run and the independent stack")),
            ("event", Json::str(event.name())),
            ("seeded_runs", Json::Int(seeded_runs)),
            ("live_control_repacked_tokens", Json::Int(live_reused as i64)),
            ("post_event", Json::Obj(records)),
            ("bystander", spared),
        ]),
    ));
    Ok(())
}

/// The bystander of C3 and C4: a memory input that is committed and never invalidated, then a
/// tail of its own (`[B, tail']`).
fn bystander_parts(lab: &Lab<'_>) -> Vec<Part> {
    vec![
        Part::bound(
            SegmentKind::Memory,
            lab::BYSTANDER_IDENTITY,
            1,
            lab.material.ancestor(),
        ),
        // Its own tail: the case's tail would give the bystander a second run that the case's
        // post-event requests find and refuse, which is not the event's doing.
        Part::raw(lab.material.spare(16, lab.shape.tail)),
    ]
}

/// After the event, an unrelated live run must still be reused at a new position, in full
/// (`bystander_reuse_cases`); losing it is `bystander_reuse_losses`, which a runtime that
/// flushes more than the event names cannot avoid.
pub(crate) fn bystander_check(
    lab: &mut Lab<'_>,
    case: &str,
    backend: &ReferenceKvReuseBackend,
    oracle: &dyn KvLivenessOracle,
    bystander: &[Part],
) -> Res<Json> {
    let request = request(
        &format!("{case}-bystander-post"),
        &after(lab.material.spare(15, SHORT_FILLER_TOKENS), bystander),
    );
    let probe = run_positional(
        &mut lab.timings,
        &format!("{case}.bystander_post"),
        backend,
        &request,
        oracle,
    )?;
    let positional = report(&probe)?;
    let kept = positional.admitted_splices >= 1 && positional.reused_tokens == lab::ANCESTOR_TOKENS;
    lab.tally.add(
        if kept {
            "bystander_reuse_cases"
        } else {
            "bystander_reuse_losses"
        },
        1,
    );
    Ok(Json::obj([
        ("kept", Json::Bool(kept)),
        ("run", case_json(&probe)?),
    ]))
}

/// Whether `candidate` holds every segment of `seeded` by content: a segment of the same
/// tokens (the lineage may differ, e.g. a successor generation).
fn matches_seeded(candidate: &SegmentedModelRequest, seeded: &SegmentedModelRequest) -> bool {
    let tokens = |segment: &ptr_model_api::PromptSegment| -> Vec<u32> {
        segment.tokens().iter().map(|token| token.0).collect()
    };
    seeded.segments().iter().all(|wanted| {
        candidate
            .segments()
            .iter()
            .any(|segment| tokens(segment) == tokens(wanted))
    })
}

// ---- C5: dead original upstream ----------------------------------------------------------------

/// Run a refused request: the run must complete bitwise equal to cold, and every pin must be
/// released. Returns the probe.
pub(crate) fn refused_run(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    oracle: &dyn KvLivenessOracle,
) -> Res<ReferenceKvProbe> {
    let probe = run_positional(&mut lab.timings, name, backend, request, oracle)?;
    let total = against_baseline(lab, name, backend, request, &probe)?;
    lab.tally.add("negative_case_output_mismatches", total);
    let pins = backend
        .secondary_stats()
        .map_err(fail("secondary stats"))?
        .active_pins;
    if pins != 0 {
        return Err(format!(
            "{name}: {pins} secondary pins are still held after the run"
        ));
    }
    Ok(probe)
}

/// Require that no refusal of `forbidden` reasons was returned (a refusal of another typed
/// reason than the case names is a wrong-type error, not a metric).
pub(crate) fn forbid(
    name: &str,
    counts: &SecondaryRefusalCounts,
    forbidden: &[SecondaryRefusalReason],
) -> Res<()> {
    for reason in forbidden {
        if counts.get(*reason) != 0 {
            return Err(format!(
                "{name}: a refusal of the wrong type, {}, was returned: {counts:?}",
                reason.code()
            ));
        }
    }
    Ok(())
}

/// C5: a shift attempt on a page whose original upstream `U` is dead. Two ways to be dead
/// without the index having heard of it: the invalidation was lost (lag), and the page was
/// computed under a ledger that the lookup's ledger never committed `U` in.
pub fn dead_upstream_cases(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let u = lab::u_part(lab.material, params::UPSTREAM_GENERATION);
    let c = lab::c_part(lab.material, shape);
    let tail = lab::tail_part(lab.material, shape);
    let body = [u, c, tail];
    let mut records = Vec::new();

    // (a) lag: the revocation is committed but the index update is lost.
    {
        let mut authority = Authority::new(&[lab::C_IDENTITY, params::UPSTREAM_IDENTITY])
            .map_err(fail("the lag authority"))?;
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        seed_origin(
            lab,
            "C5.lag.origin",
            &backend,
            &request("c5-lag-origin", &body),
            &authority,
        )?;
        backend
            .set_faults(ptr_runtime::ReferenceKvFaults {
                index_update: ptr_runtime::IndexUpdate::Suppressed,
            })
            .map_err(fail("injecting the lost invalidation"))?;
        let mark = authority.mark();
        commit_upstream_event(&mut authority, Event::Revocation)?;
        crate::experiments::layered_kv::lab::apply_events(&backend, &authority.events_since(mark))
            .map_err(fail("applying the event"))?;
        let stats = backend.secondary_stats().map_err(fail("secondary stats"))?;
        if stats.invalidated != 0 {
            return Err("C5 lag: the suppressed invalidation reached the index"
                .to_string()
                .into());
        }
        let post = request(
            "c5-lag-post",
            &after(lab.material.spare(3, SHORT_FILLER_TOKENS), &body),
        );
        let probe = refused_run(lab, "C5.lag.post", &backend, &post, &authority)?;
        records.push(("lag".to_string(), dead_attempt(lab, "C5.lag", &probe)?));
    }

    // (b) never committed: the page was computed where `U` was live, the lookup is anchored at
    // a ledger that never committed it.
    {
        let writer = Authority::new(&[lab::C_IDENTITY, params::UPSTREAM_IDENTITY])
            .map_err(fail("the writer authority"))?;
        let reader = Authority::new(&[lab::C_IDENTITY]).map_err(fail("the reader authority"))?;
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        seed_origin(
            lab,
            "C5.uncommitted.origin",
            &backend,
            &request("c5-uncommitted-origin", &body),
            &writer,
        )?;
        let post = request(
            "c5-uncommitted-post",
            &after(lab.material.spare(4, SHORT_FILLER_TOKENS), &body),
        );
        let probe = refused_run(lab, "C5.uncommitted.post", &backend, &post, &reader)?;
        records.push((
            "never_committed".to_string(),
            dead_attempt(lab, "C5.uncommitted", &probe)?,
        ));
    }

    // (c) restore into a runtime that never committed `U`: the restored runs carry no rows,
    // and a re-insert under that ledger is refused (a typed insert refusal, not a lookup
    // refusal, so it is recorded here and counted by no metric); an admission after it is a
    // shift hit with a dead original upstream.
    {
        let writer = Authority::new(&[lab::C_IDENTITY, params::UPSTREAM_IDENTITY])
            .map_err(fail("the restore writer"))?;
        let reader = Authority::new(&[lab::C_IDENTITY]).map_err(fail("the restore reader"))?;
        let record = restore_uncommitted(lab, &body, &writer, &reader)?;
        records.push(("restore_never_committed".to_string(), record));
    }
    lab.cases.push((
        "C5".to_string(),
        Json::obj([
            ("what", Json::str("a shift attempt whose page's original upstream U is dead: (a) the revocation is committed and the index update lost, (b) the lookup's ledger never committed U; the lookup must return OriginalUpstreamDead and admit nothing; (c) a restore into a ledger that never committed U attaches no rows (recorded, counted by no metric)")),
            ("variants", Json::Obj(records)),
        ]),
    ));
    Ok(())
}

/// C5 (c): a catalog snapshot written under `writer` is restored into a backend whose lookups
/// run under `reader`. The re-insert of the origin content and a shifted request after it are
/// run; whatever they attach or admit is counted (`restore_uncommitted_*`, and the admissions
/// as shift hits with a dead original upstream, a span or a last-segment skip alike).
pub(crate) fn restore_uncommitted(
    lab: &mut Lab<'_>,
    body: &[Part],
    writer: &Authority,
    reader: &Authority,
) -> Run<Json> {
    let shape = lab.shape;
    let source = lab::positional_backend(shape, PositionalControl::None)?;
    seed_origin(
        lab,
        "C5.restore.origin",
        &source,
        &request("c5-restore-origin", body),
        writer,
    )?;
    let snapshot = source
        .export_catalog()
        .map_err(fail("exporting the catalog"))?;
    let target = lab::positional_backend(shape, PositionalControl::None)?;
    let restored = target
        .restore_catalog(&snapshot)
        .map_err(fail("restoring the catalog"))?;
    if restored.secondary_runs == 0 {
        return control("C5 restore: the snapshot carried no secondary runs");
    }
    // A re-insert of the origin content under `reader`, and a shifted request after it.
    let again = run_positional(
        &mut lab.timings,
        "C5.restore.reinsert",
        &target,
        &request("c5-restore-again", body),
        reader,
    )?;
    let post = request(
        "c5-restore-post",
        &after(lab.material.spare(12, SHORT_FILLER_TOKENS), body),
    );
    let later = run_positional(&mut lab.timings, "C5.restore.post", &target, &post, reader)?;
    let (first, second) = (report(&again)?, report(&later)?);
    let attached = first.rows_attached + second.rows_attached;
    let stats = target.secondary_stats().map_err(fail("secondary stats"))?;
    lab.tally
        .add("restore_uncommitted_rows_attached", attached as i64);
    lab.tally.add(
        "restore_uncommitted_admissions",
        admitted(first) + admitted(second),
    );
    lab.tally.add(
        "shift_hits_with_dead_original_upstream",
        admitted(first) + admitted(second),
    );
    Ok(Json::obj([
        ("restored_runs", Json::Int(restored.secondary_runs as i64)),
        ("rows_attached", Json::Int(attached as i64)),
        (
            "admitted_spans",
            Json::Int(admitted(first) + admitted(second)),
        ),
        (
            "rows_missing_misses",
            Json::Int(stats.rows_missing_misses as i64),
        ),
        ("reinsert", case_json(&again)?),
        ("later", case_json(&later)?),
    ]))
}

/// Count a C5 lookup: an attempt only when the runtime returned `OriginalUpstreamDead` at its
/// boundary; an admitted span is a shift hit with a dead original upstream.
fn dead_attempt(lab: &mut Lab<'_>, name: &str, probe: &ReferenceKvProbe) -> Res<Json> {
    let positional = report(probe)?;
    forbid(
        name,
        &positional.refusals,
        &[
            SecondaryRefusalReason::SourceKindMismatch,
            SecondaryRefusalReason::SourceIdentityMismatch,
            SecondaryRefusalReason::SourceGenerationMismatch,
            SecondaryRefusalReason::SourceDigestMismatch,
            SecondaryRefusalReason::OriginalUpstreamOmitted,
            SecondaryRefusalReason::RawAncestorOmitted,
        ],
    )?;
    let dead = positional.refusals.original_upstream_dead;
    if dead >= 1 {
        lab.tally.add("dead_original_upstream_shift_attempts", 1);
    }
    lab.tally.add(
        "shift_hits_with_dead_original_upstream",
        admitted(positional),
    );
    Ok(Json::obj([
        ("original_upstream_dead_refusals", Json::Int(dead as i64)),
        ("run", case_json(probe)?),
    ]))
}

// ---- C6, C7, C8: omission, own lineage, raw ancestors ------------------------------------------

/// C6: a request that omits a segment of the page's original prefix, with the multiplicity
/// case and a reordering that must be admitted.
pub fn omission_cases(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let live = |lab: &Lab<'_>| -> Res<(ReferenceKvReuseBackend, ptr_runtime::ReferenceKvLiveness)> {
        let _ = lab;
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        let oracle = backend.own_liveness();
        Ok((backend, oracle))
    };
    let u = lab::u_part(lab.material, params::UPSTREAM_GENERATION);
    let c = lab::c_part(lab.material, shape);
    let tail = lab::tail_part(lab.material, shape);
    let mut records = Vec::new();

    // Omitted: the request keeps C and the tail but not U.
    {
        let (backend, oracle) = live(lab)?;
        seed_origin(
            lab,
            "C6.omitted.origin",
            &backend,
            &request("c6-origin", &[u.clone(), c.clone(), tail.clone()]),
            &oracle,
        )?;
        let omitted = request(
            "c6-omitted",
            &after(
                lab.material.spare(5, SHORT_FILLER_TOKENS),
                &[c.clone(), tail.clone()],
            ),
        );
        let probe = refused_run(lab, "C6.omitted", &backend, &omitted, &oracle)?;
        records.push((
            "omitted".to_string(),
            omission_outcome(lab, "C6.omitted", &probe, 0)?,
        ));
    }
    // Multiplicity: the run saw U twice, the request has it once. The first copy of U (no
    // upstream) is still legitimately reused; C is not.
    {
        let (backend, oracle) = live(lab)?;
        seed_origin(
            lab,
            "C6.multiplicity.origin",
            &backend,
            &request(
                "c6-mult-origin",
                &[u.clone(), u.clone(), c.clone(), tail.clone()],
            ),
            &oracle,
        )?;
        let fewer = request(
            "c6-multiplicity",
            &after(
                lab.material.spare(6, SHORT_FILLER_TOKENS),
                &[u.clone(), c.clone(), tail.clone()],
            ),
        );
        let probe = refused_run_with_reuse(
            lab,
            "C6.multiplicity",
            &backend,
            &fewer,
            &oracle,
            lab::UPSTREAM_TOKENS,
        )?;
        records.push((
            "multiplicity".to_string(),
            omission_outcome(lab, "C6.multiplicity", &probe, 1)?,
        ));
    }
    // Reordering is not omission: U moves behind C and the run is admitted (a positive control
    // of the rule, not a metric: if it is refused the rule over-refuses and the case cannot be
    // judged).
    {
        let (backend, oracle) = live(lab)?;
        seed_origin(
            lab,
            "C6.reorder.origin",
            &backend,
            &request("c6-reorder-origin", &[u.clone(), c.clone(), tail.clone()]),
            &oracle,
        )?;
        let reordered = request(
            "c6-reordered",
            &[
                c.clone(),
                Part::raw(lab.material.spare(7, 7)),
                u.clone(),
                tail.clone(),
            ],
        );
        let probe = run_positional(
            &mut lab.timings,
            "C6.reordered",
            &backend,
            &reordered,
            &oracle,
        )?;
        let total = against_baseline(lab, "C6.reordered", &backend, &reordered, &probe)?;
        lab.tally.add("negative_case_output_mismatches", total);
        let positional = report(&probe)?;
        if positional.reused_tokens < shape.segment {
            return control(format!(
                "C6 reorder: the reordered request repacked {} tokens, fewer than C's {}; the omission rule over-refuses: {positional:?}",
                positional.reused_tokens, shape.segment
            ));
        }
        records.push((
            "reordered_control".to_string(),
            Json::obj([("run", case_json(&probe)?)]),
        ));
    }
    lab.cases.push((
        "C6".to_string(),
        Json::obj([
            ("what", Json::str("a request omitting a segment of the page's original prefix (the segment U, and the multiplicity case with fewer copies than the run holds) must be refused with OriginalUpstreamOmitted and admit nothing for the dependent page; a reordering that keeps every segment is admitted (control)")),
            ("variants", Json::Obj(records)),
        ]),
    ));
    Ok(())
}

/// Like [`refused_run`] when `reused` tokens of a segment unrelated to the refusal are still
/// repacked.
fn refused_run_with_reuse(
    lab: &mut Lab<'_>,
    name: &str,
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    oracle: &dyn KvLivenessOracle,
    legit: usize,
) -> Run<ReferenceKvProbe> {
    let probe = refused_run(lab, name, backend, request, oracle)?;
    let reused = report(&probe)?.reused_tokens;
    if reused < legit {
        return control(format!(
            "{name}: the legitimately reusable {legit} tokens were not reused ({reused})"
        ));
    }
    Ok(probe)
}

/// Count a C6 case: only a returned `OriginalUpstreamOmitted` is the case; an admitted span
/// beyond the `legit` ones is an admission despite the omission.
fn omission_outcome(
    lab: &mut Lab<'_>,
    name: &str,
    probe: &ReferenceKvProbe,
    legit_splices: i64,
) -> Res<Json> {
    let positional = report(probe)?;
    forbid(
        name,
        &positional.refusals,
        &[
            SecondaryRefusalReason::OriginalUpstreamDead,
            SecondaryRefusalReason::BindingDependencyDead,
            SecondaryRefusalReason::SourceKindMismatch,
            SecondaryRefusalReason::SourceIdentityMismatch,
            SecondaryRefusalReason::SourceGenerationMismatch,
            SecondaryRefusalReason::SourceDigestMismatch,
        ],
    )?;
    if positional.refusals.original_upstream_omitted >= 1 {
        lab.tally.add("omitted_upstream_segment_cases", 1);
    }
    // The legitimate splices are spliced spans only: a last-segment skip is never one of them.
    lab.tally.add(
        "omitted_upstream_admissions",
        (positional.admitted_splices as i64 - legit_splices).max(0)
            + positional.last_token_skips as i64,
    );
    Ok(Json::obj([
        (
            "original_upstream_omitted_refusals",
            Json::Int(positional.refusals.original_upstream_omitted as i64),
        ),
        ("run", case_json(probe)?),
    ]))
}

/// C7: the page's own lineage differs from the request's segment, with all upstream live.
pub fn lineage_cases(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let c_tokens = lab.material.segment(shape);
    let tail = lab::tail_part(lab.material, shape);
    let mut edited = c_tokens.clone();
    let middle = edited.len() / 2;
    edited[middle] = (edited[middle] + 1) % params::VOCAB_SIZE as u32;
    // (name, request segment, named reason, other reasons that must not appear)
    let cases: Vec<(&str, Part, SecondaryRefusalReason)> = vec![
        (
            "generation",
            Part::bound(SegmentKind::Memory, lab::C_IDENTITY, 2, c_tokens.clone()),
            SecondaryRefusalReason::SourceGenerationMismatch,
        ),
        (
            "digest",
            Part::bound(SegmentKind::Memory, lab::C_IDENTITY, 1, edited),
            SecondaryRefusalReason::SourceDigestMismatch,
        ),
        (
            "identity",
            Part::bound(SegmentKind::Memory, "mem:r006-other", 1, c_tokens.clone()),
            SecondaryRefusalReason::SourceIdentityMismatch,
        ),
        (
            "kind",
            Part::bound(SegmentKind::Semantic, lab::C_IDENTITY, 1, c_tokens.clone()),
            SecondaryRefusalReason::SourceKindMismatch,
        ),
    ];
    let lineage = [
        SecondaryRefusalReason::SourceKindMismatch,
        SecondaryRefusalReason::SourceIdentityMismatch,
        SecondaryRefusalReason::SourceGenerationMismatch,
        SecondaryRefusalReason::SourceDigestMismatch,
    ];
    let mut records = Vec::new();
    for (name, segment, named) in cases {
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        let oracle = backend.own_liveness();
        let c = lab::c_part(lab.material, shape);
        seed_origin(
            lab,
            &format!("C7.{name}.origin"),
            &backend,
            &request(&format!("c7-{name}-origin"), &[c, tail.clone()]),
            &oracle,
        )?;
        let mismatched = request(
            &format!("c7-{name}"),
            &[
                Part::raw(lab.material.spare(8, SHORT_FILLER_TOKENS)),
                segment,
                tail.clone(),
            ],
        );
        let probe = refused_run(lab, &format!("C7.{name}"), &backend, &mismatched, &oracle)?;
        let positional = report(&probe)?;
        // The named reason, and none of the other own-lineage reasons.
        let others: Vec<SecondaryRefusalReason> = lineage
            .iter()
            .copied()
            .filter(|reason| *reason != named)
            .collect();
        forbid(&format!("C7.{name}"), &positional.refusals, &others)?;
        if positional.refusals.get(named) >= 1 {
            match named {
                SecondaryRefusalReason::SourceGenerationMismatch => {
                    lab.tally.add("source_generation_mismatch_cases", 1)
                }
                SecondaryRefusalReason::SourceDigestMismatch => {
                    lab.tally.add("source_digest_mismatch_cases", 1)
                }
                _ => lab.tally.add(&format!("source_{name}_mismatch_cases"), 1),
            }
        }
        // No span may have been admitted for a segment whose own lineage differs: the
        // request's only repackable segment is the tail (never spliced: it holds the last
        // token) and the segment under test.
        lab.tally
            .add("source_lineage_mismatch_admissions", admitted(positional));
        records.push((
            name.to_string(),
            Json::obj([
                ("named_reason", Json::str(named.code())),
                (
                    "named_refusals",
                    Json::Int(positional.refusals.get(named) as i64),
                ),
                ("run", case_json(&probe)?),
            ]),
        ));
    }
    lab.cases.push((
        "C7".to_string(),
        Json::obj([
            ("what", Json::str("all upstream live; a request segment with the page's tokens but another generation, another identity or kind, or one changed token (an edit inside the segment changes its digest) must be refused with the named own-lineage reason and give zero reuse of that page")),
            ("variants", Json::Obj(records)),
        ]),
    ));
    Ok(())
}

/// C8: a raw upstream segment removed, the rest unchanged.
pub fn raw_ancestor_cases(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let backend = lab::positional_backend(shape, PositionalControl::None)?;
    let oracle = backend.own_liveness();
    let ancestor = Part::raw(lab.material.ancestor());
    let c = lab::c_part(lab.material, shape);
    let tail = lab::tail_part(lab.material, shape);
    seed_origin(
        lab,
        "C8.origin",
        &backend,
        &request("c8-origin", &[ancestor, c.clone(), tail.clone()]),
        &oracle,
    )?;
    let removed = request(
        "c8-removed",
        &after(lab.material.spare(9, SHORT_FILLER_TOKENS), &[c, tail]),
    );
    let probe = refused_run(lab, "C8.removed", &backend, &removed, &oracle)?;
    let positional = report(&probe)?;
    forbid(
        "C8",
        &positional.refusals,
        &[
            SecondaryRefusalReason::OriginalUpstreamOmitted,
            SecondaryRefusalReason::OriginalUpstreamDead,
            SecondaryRefusalReason::SourceDigestMismatch,
        ],
    )?;
    if positional.refusals.raw_ancestor_omitted >= 1 {
        lab.tally.add("raw_ancestor_omission_cases", 1);
    }
    lab.tally
        .add("reuse_after_raw_ancestor_removed", admitted(positional));
    lab.cases.push((
        "C8".to_string(),
        Json::obj([
            ("what", Json::str("origin [A, C, tail] with A a Raw segment; a request that removes A and keeps the rest must be refused with RawAncestorOmitted (not OriginalUpstreamOmitted) and reuse nothing of C")),
            ("run", case_json(&probe)?),
            ("raw_ancestor_omitted_refusals", Json::Int(positional.refusals.raw_ancestor_omitted as i64)),
        ]),
    ));
    Ok(())
}

// ---- C10: the negative controls --------------------------------------------------------------

/// Evaluate the falsifier rules of `names` against the control's counters: every one must
/// FAIL (trip), else the criterion cannot fail and the run is vacuous (exit status 2).
fn require_tripped(control: &str, counters: &Json, names: &[&str]) -> Res<Vec<String>> {
    let falsifiers = metrics::falsifiers();
    let mut tripped = Vec::new();
    for name in names {
        let rule = falsifiers
            .iter()
            .find(|rule| rule.starts_with(&format!("{name} ")))
            .ok_or_else(|| format!("{control}: no falsifier rule for {name}"))?;
        match emit::evaluate(counters, rule) {
            Ok(false) => tripped.push(rule.clone()),
            Ok(true) => {
                return Err(format!(
                    "{control}: the control did not trip {rule} ({}): the criterion cannot fail",
                    counters.render()
                ))
            }
            Err(error) => return Err(format!("{control}: {rule} cannot be read: {error}")),
        }
    }
    Ok(tripped)
}

/// C10: the controls that show the criteria can fail. A control whose premise fails (the
/// runtime reuses nothing, so there is nothing to bypass or mismatch) is a failed control
/// (the seed FALSIFIES); a control that runs and does not trip its falsifier is a harness that
/// cannot be trusted (exit status 2).
pub fn controls(lab: &mut Lab<'_>) -> Run<()> {
    let shape = lab.shape;
    let mut records = Vec::new();

    // (a) PostRope: the same shifted reuse must differ bitwise.
    {
        let shift = shape.shifts[1] as usize;
        let backend = lab::post_rope_backend(shape)?;
        let oracle = backend.own_liveness();
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        seed_origin(
            lab,
            "C10a.origin",
            &backend,
            &request("c10a-origin", &[c.clone(), tail.clone()]),
            &oracle,
        )?;
        let shifted = request("c10a", &after(lab.material.filler(shift), &[c, tail]));
        let baseline = run_cold(&mut lab.timings, "C10a.cold", &backend, &shifted)?;
        let warm = run_positional(&mut lab.timings, "C10a.warm", &backend, &shifted, &oracle)?;
        let reused = report(&warm)?.reused_tokens;
        if reused == 0 {
            return control("C10 PostRope: nothing was reused, so the control tests nothing");
        }
        let cmp = compare(&warm, &baseline);
        let counters = Json::obj([(
            "bitwise_mismatch_count",
            Json::obj([(
                &*params::shift_member(params::POSITION_SHIFT_TOKENS[1]),
                Json::Int(cmp.total()),
            )]),
        )]);
        let tripped = require_tripped(
            "C10 PostRope",
            &counters,
            &["bitwise_mismatch_count.shift_7"],
        )?;
        records.push((
            "post_rope".to_string(),
            Json::obj([
                ("repacked_tokens", Json::Int(reused as i64)),
                ("mismatches", Json::Int(cmp.total())),
                ("kv_state_digest_differs", Json::Bool(cmp.digest == 1)),
                ("logits_entries_differing", Json::Int(cmp.logits)),
                ("falsifiers_tripped", Json::strs(&tripped)),
            ]),
        ));
        lab.tally
            .set("negative_control_postrope_mismatches", cmp.total());
    }

    // (b) the liveness bypass: the reachability, dead-original (revocation) and superseded
    // reachability counters, and the in-flight counters, must trip.
    {
        let u = lab::u_part(lab.material, params::UPSTREAM_GENERATION);
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        let body = [u, c, tail];
        let setup = |lab: &mut Lab<'_>,
                     control: PositionalControl,
                     event: Event|
         -> Run<(ReferenceKvReuseBackend, Authority)> {
            let mut authority = Authority::new(&[lab::C_IDENTITY, params::UPSTREAM_IDENTITY])
                .map_err(fail("the bypass authority"))?;
            let backend = lab::positional_backend(shape, control)?;
            seed_origin(
                lab,
                "C10b.origin",
                &backend,
                &request("c10b-origin", &body),
                &authority,
            )?;
            backend
                .set_faults(ptr_runtime::ReferenceKvFaults {
                    index_update: ptr_runtime::IndexUpdate::Suppressed,
                })
                .map_err(fail("injecting the lost invalidation"))?;
            let mark = authority.mark();
            commit_upstream_event(&mut authority, event)?;
            crate::experiments::layered_kv::lab::apply_events(
                &backend,
                &authority.events_since(mark),
            )
            .map_err(fail("applying the event"))?;
            Ok((backend, authority))
        };
        let post = request(
            "c10b-post",
            &after(lab.material.spare(10, SHORT_FILLER_TOKENS), &body),
        );
        // The honest backend refuses, so the same counters stay at zero ...
        let (honest, honest_authority) = setup(lab, PositionalControl::None, Event::Revocation)?;
        let refused = run_positional(
            &mut lab.timings,
            "C10b.honest",
            &honest,
            &post,
            &honest_authority,
        )?;
        let honest_report = report(&refused)?;
        if reachable(honest_report) == 0 || admitted(honest_report) != 0 {
            return control(format!(
                "C10 bypass: the honest twin did not refuse the dead page: {honest_report:?}"
            ));
        }
        // ... and the bypassed one admits it, for a revocation and for a supersession.
        let mut bypassed = Vec::new();
        for event in [Event::Revocation, Event::Supersession] {
            let (backend, authority) = setup(lab, PositionalControl::BypassUseTimeLiveness, event)?;
            let probe = run_positional(
                &mut lab.timings,
                &format!("C10b.bypass_{}", event.name()),
                &backend,
                &post,
                &authority,
            )?;
            let positional = report(&probe)?;
            if !positional.liveness_bypassed || positional.bypass_dead_admissions == 0 {
                return control(format!(
                    "C10 bypass: the bypassed backend did not admit the {} page: {positional:?}",
                    event.name()
                ));
            }
            bypassed.push(probe);
        }
        let (revoked, superseded) = (report(&bypassed[0])?, report(&bypassed[1])?);
        // The in-flight counters under the bypass: the same measure as the matrix uses, and
        // the same measure with an event closure that also performs a live positional run
        // (which inserts), so the cache-insert counter is shown to move.
        let escaped = super::matrix::bypass_inflight(lab)?;
        let leaked = super::matrix::insert_leak_inflight(lab)?;
        let counters = Json::obj([
            (
                "revoked_upstream_page_reachable_count",
                Json::Int(reachable(revoked)),
            ),
            (
                "superseded_upstream_page_reachable_count",
                Json::Int(reachable(superseded)),
            ),
            (
                "shift_hits_with_dead_original_upstream",
                Json::Int(admitted(revoked)),
            ),
            ("shifted_reuse_inflight_outputs", Json::Int(escaped.outputs)),
            (
                "shifted_reuse_inflight_cache_inserts",
                Json::Int(leaked.inserts),
            ),
        ]);
        let tripped = require_tripped(
            "C10 liveness bypass",
            &counters,
            &[
                "revoked_upstream_page_reachable_count",
                "superseded_upstream_page_reachable_count",
                "shift_hits_with_dead_original_upstream",
                "shifted_reuse_inflight_outputs",
                "shifted_reuse_inflight_cache_inserts",
            ],
        )?;
        records.push((
            "liveness_bypass".to_string(),
            Json::obj([
                ("honest_twin", case_json(&refused)?),
                ("bypassed_revocation", case_json(&bypassed[0])?),
                ("bypassed_supersession", case_json(&bypassed[1])?),
                (
                    "bypass_dead_admissions",
                    Json::Int(revoked.bypass_dead_admissions as i64),
                ),
                ("inflight_outputs_after_event", Json::Int(escaped.outputs)),
                ("inflight_inserts_after_event", Json::Int(escaped.inserts)),
                ("inflight_cancelled", Json::Bool(escaped.cancelled)),
                ("insert_leak_inserts_after_event", Json::Int(leaked.inserts)),
                ("insert_leak_cancelled", Json::Bool(leaked.cancelled)),
                ("falsifiers_tripped", Json::strs(&tripped)),
            ]),
        ));
        lab.tally
            .set("negative_control_bypass_admissions", admitted(revoked));
    }

    // (b2) the wiring of the admission counters: a real admitted report (a shifted run that
    // is rightly admitted, from a positive case) fed to each admission counter must trip its
    // falsifier, and so must a report whose only admission is the last segment's (the skip
    // that `admitted` counts). Without this the counters are shown only by the JSON-mutation
    // tests, which are syntactic.
    {
        let backend = lab::positional_backend(shape, PositionalControl::None)?;
        let oracle = backend.own_liveness();
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        seed_origin(
            lab,
            "C10b2.origin",
            &backend,
            &request("c10b2-origin", &[c.clone(), tail.clone()]),
            &oracle,
        )?;
        let admitted_run = run_positional(
            &mut lab.timings,
            "C10b2.admitted",
            &backend,
            &request(
                "c10b2",
                &after(lab.material.spare(13, SHORT_FILLER_TOKENS), &[c, tail]),
            ),
            &oracle,
        )?;
        let real = report(&admitted_run)?;
        if real.admitted_splices == 0 {
            return control(format!(
                "C10 admission wiring: no span was admitted, so there is nothing to feed: {real:?}"
            ));
        }
        let names = [
            "omitted_upstream_admissions",
            "source_lineage_mismatch_admissions",
            "reuse_after_raw_ancestor_removed",
            "shift_hits_with_dead_original_upstream",
        ];
        let feed = |value: i64| {
            Json::Obj(
                names
                    .iter()
                    .map(|n| ((*n).to_string(), Json::Int(value)))
                    .collect(),
            )
        };
        let tripped = require_tripped("C10 admission wiring", &feed(admitted(real)), &names)?;
        let mut tail_only = real.clone();
        tail_only.admitted_splices = 0;
        tail_only.last_token_skips = 1;
        require_tripped(
            "C10 admission wiring (last segment only)",
            &feed(admitted(&tail_only)),
            &names,
        )?;
        records.push((
            "admission_wiring".to_string(),
            Json::obj([
                ("admitted_splices", Json::Int(real.admitted_splices as i64)),
                ("last_segment_only_counted", Json::Int(admitted(&tail_only))),
                ("falsifiers_tripped", Json::strs(&tripped)),
            ]),
        ));
    }

    // (c) primary only: the shifted content gets no reuse without the secondary index.
    {
        let shift = shape.shifts[1] as usize;
        let backend = lab::primary_only_backend(shape)?;
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        run_warm(
            &mut lab.timings,
            "C10c.origin",
            &backend,
            &request("c10c-origin", &[c.clone(), tail.clone()]),
        )?;
        let shifted = request("c10c", &after(lab.material.filler(shift), &[c, tail]));
        let probe = run_warm(&mut lab.timings, "C10c.shifted", &backend, &shifted)?;
        let reused = probe.outcome.hit_tokens
            + probe
                .outcome
                .run
                .kv_report
                .as_ref()
                .map_or(0, |report| report.hit_tokens());
        if probe.outcome.positional.is_some() || reused != 0 {
            return Err(format!(
                "C10 primary-only: the shifted content got {} reused tokens without the secondary index",
                reused
            )
            .into());
        }
        records.push((
            "primary_only".to_string(),
            Json::obj([
                (
                    "primary_hit_tokens",
                    Json::Int(probe.outcome.hit_tokens as i64),
                ),
                (
                    "prefilled_tokens",
                    Json::Int(probe.outcome.prefilled_tokens as i64),
                ),
            ]),
        ));
    }

    // (d) the production mode partitions the index, and only Exact is admitted.
    {
        let mode = super::matrix::mode_partition(lab)?;
        records.push(("mode_partition".to_string(), mode));
    }

    // (e) one layer only.
    {
        let four = ReferenceKvReuseBackend::with_layers(
            lab::limits(shape),
            crate::experiments::layered_kv::lab::scope(),
            ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
            4,
        )
        .map_err(fail("a four-layer backend"))?;
        match four.with_positional_reuse(ptr_runtime::PositionalOptions::default()) {
            Err(ReferenceKvError::PositionalNeedsOneLayer { layers: 4 }) => {}
            other => {
                return Err(format!(
                    "C10 one layer: a four-layer backend was not refused with PositionalNeedsOneLayer: {:?}",
                    other.map(|_| ())
                )
                .into())
            }
        }
        records.push(("four_layers_refused".to_string(), Json::Bool(true)));
    }
    lab.cases.push((
        "C10".to_string(),
        Json::obj([
            ("what", Json::str("controls that show the criteria can fail: (a) a PostRope stack mismatches bitwise on the same shifted reuse; (b) under a liveness-bypass fault mode the dead page (revoked and superseded) is admitted and the reachability, dead-original, superseded-reachability and in-flight output falsifiers trip, and an in-flight race whose event closure also performs a live run trips the cache-insert falsifier; (b2) a really admitted report, and one whose only admission is the last segment's, trips each admission counter; (c) a primary-only run on the shifted content reuses nothing; (d) the index partitions by production mode and admits only Exact; (e) a four-layer backend refuses positional reuse. A control that runs and does not trip is exit status 2; a control whose premise fails (nothing was reused) is a failed control")),
            ("controls", Json::Obj(records)),
        ]),
    ));
    Ok(())
}

// ---- helpers shared with the matrix ------------------------------------------------------------

/// Wait for a session to reach `position` prompt tokens by prefill steps, none of which may
/// be a cancellation.
pub(crate) fn step_to(
    session: &mut ptr_runtime::KvSession<'_>,
    oracle: &dyn KvLivenessOracle,
    position: usize,
    name: &str,
) -> Res<()> {
    while session.position() < position {
        match session.prefill_chunk(oracle).map_err(fail(name))? {
            KvSessionStep::Prefilled { .. } => {}
            other => {
                return Err(format!(
                    "{name}: expected a prefill step before the event, got {other:?}"
                ))
            }
        }
    }
    if session.position() != position {
        return Err(format!(
            "{name}: stepped past {position} to {}",
            session.position()
        ));
    }
    Ok(())
}

/// Prefill until the session decodes, none of the steps a cancellation.
pub(crate) fn prefill_to_decode(
    session: &mut ptr_runtime::KvSession<'_>,
    oracle: &dyn KvLivenessOracle,
    name: &str,
) -> Res<()> {
    while session.phase() == KvSessionPhase::Prefilling {
        match session.prefill_chunk(oracle).map_err(fail(name))? {
            KvSessionStep::Prefilled { .. } => {}
            other => {
                return Err(format!(
                    "{name}: expected a prefill step before the event, got {other:?}"
                ))
            }
        }
    }
    Ok(())
}

/// Run every case of `seed` except the matrix. A case whose positive control fails is recorded
/// (`live_control_failures`) and the run goes on; a harness failure ends the run.
pub fn run_all(lab: &mut Lab<'_>) -> Res<()> {
    let outcome = shift_cases(lab);
    lab.settle("C1", outcome)?;
    let outcome = nonaligned_cases(lab);
    lab.settle("C2", outcome)?;
    let outcome = upstream_event_case(lab, Event::Revocation);
    lab.settle("C3", outcome)?;
    let outcome = upstream_event_case(lab, Event::Supersession);
    lab.settle("C4", outcome)?;
    let outcome = dead_upstream_cases(lab);
    lab.settle("C5", outcome)?;
    let outcome = omission_cases(lab);
    lab.settle("C6", outcome)?;
    let outcome = lineage_cases(lab);
    lab.settle("C7", outcome)?;
    let outcome = raw_ancestor_cases(lab);
    lab.settle("C8", outcome)?;
    let outcome = controls(lab);
    lab.settle("C10", outcome)?;
    Ok(())
}

/// The role of each corpus sequence this seed uses, for the record.
pub fn role_names() -> Vec<&'static str> {
    [
        Role::Filler,
        Role::Segment,
        Role::Tail,
        Role::Inserted,
        Role::Upstream,
        Role::Ancestor,
        Role::PrefixA,
        Role::Spare,
    ]
    .iter()
    .map(|role| match role {
        Role::Filler => "filler",
        Role::Segment => "segment",
        Role::Tail => "tail",
        Role::Inserted => "inserted",
        Role::Upstream => "upstream",
        Role::Ancestor => "ancestor",
        Role::PrefixA => "prefix_a",
        Role::Spare => "spare",
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::super::lab::test_support::with_lab;
    use super::*;
    use crate::experiments::layered_kv::emit;

    #[test]
    fn a_control_that_does_not_trip_its_falsifier_is_an_error_and_one_that_trips_is_recorded() {
        let quiet = Json::obj([("shift_hits_with_dead_original_upstream", Json::Int(0))]);
        let error =
            require_tripped("c", &quiet, &["shift_hits_with_dead_original_upstream"]).unwrap_err();
        assert!(error.contains("cannot fail"), "{error}");
        let loud = Json::obj([("shift_hits_with_dead_original_upstream", Json::Int(1))]);
        assert_eq!(
            require_tripped("c", &loud, &["shift_hits_with_dead_original_upstream"]).unwrap(),
            ["shift_hits_with_dead_original_upstream <= 0"]
        );
        // A name with no falsifier rule, and a counter that is missing, are errors too.
        assert!(require_tripped("c", &loud, &["no_such_metric"]).is_err());
        assert!(require_tripped(
            "c",
            &Json::obj([]),
            &["shift_hits_with_dead_original_upstream"]
        )
        .is_err());
    }

    #[test]
    fn a_refusal_of_the_wrong_type_is_an_error_and_the_expected_one_is_not() {
        let counts = SecondaryRefusalCounts {
            original_upstream_omitted: 1,
            ..SecondaryRefusalCounts::default()
        };
        assert!(forbid(
            "c",
            &counts,
            &[SecondaryRefusalReason::OriginalUpstreamDead]
        )
        .is_ok());
        let error = forbid(
            "c",
            &counts,
            &[SecondaryRefusalReason::OriginalUpstreamOmitted],
        )
        .unwrap_err();
        assert!(error.contains("original_upstream_omitted"), "{error}");
    }

    /// A shifted run that is not bit-identical is counted per shift member (the PostRope
    /// stack stands for any defective repack), is a non-bit-exact Exact result and is not a
    /// parity case.
    #[test]
    fn a_shifted_run_that_differs_from_its_baseline_is_counted_and_is_not_parity() {
        with_lab(|lab| {
            let shape = lab.shape;
            let backend = lab::post_rope_backend(shape).unwrap();
            let oracle = backend.own_liveness();
            let c = lab::c_part(lab.material, shape);
            let tail = lab::tail_part(lab.material, shape);
            let origin = request("origin", &[c.clone(), tail.clone()]);
            seed_origin(lab, "origin", &backend, &origin, &oracle).unwrap();
            let shifted = request("shifted", &after(lab.material.filler(7), &[c, tail]));
            let baseline = run_cold(&mut lab.timings, "cold", &backend, &shifted).unwrap();
            let warm =
                run_positional(&mut lab.timings, "warm", &backend, &shifted, &oracle).unwrap();
            let judged =
                judge_shifted(lab, "defect", &shifted, &warm, &baseline, "shift_7").unwrap();
            assert!(judged.mismatches > 0);
            assert!(!judged.parity);
            assert_eq!(
                lab.tally.get("bitwise_mismatch_count.shift_7"),
                judged.mismatches
            );
            assert_eq!(lab.tally.get("non_bit_exact_exact_results"), 1);
            assert_eq!(lab.tally.get("tainted_exact_results"), 0);
        });
    }

    /// The same run on the honest backend is parity and counts nothing.
    #[test]
    fn a_bit_identical_shifted_run_is_a_parity_case_and_counts_no_mismatch() {
        with_lab(|lab| {
            let shape = lab.shape;
            let backend = lab::positional_backend(shape, PositionalControl::None).unwrap();
            let oracle = backend.own_liveness();
            let c = lab::c_part(lab.material, shape);
            let tail = lab::tail_part(lab.material, shape);
            let origin = request("origin", &[c.clone(), tail.clone()]);
            seed_origin(lab, "origin", &backend, &origin, &oracle).unwrap();
            let shifted = request("shifted", &after(lab.material.filler(7), &[c, tail]));
            let baseline = run_cold(&mut lab.timings, "cold", &backend, &shifted).unwrap();
            let warm =
                run_positional(&mut lab.timings, "warm", &backend, &shifted, &oracle).unwrap();
            let judged =
                judge_shifted(lab, "honest", &shifted, &warm, &baseline, "shift_7").unwrap();
            assert_eq!(judged.mismatches, 0);
            assert!(judged.parity);
            assert_eq!(judged.reused, shape.segment);
        });
    }

    /// A refused origin that did not populate the index is a harness error, not a metric.
    #[test]
    fn an_origin_that_inserted_nothing_is_an_error() {
        with_lab(|lab| {
            let shape = lab.shape;
            let backend = lab::positional_backend(shape, PositionalControl::None).unwrap();
            let oracle = backend.own_liveness();
            let c = lab::c_part(lab.material, shape);
            let tail = lab::tail_part(lab.material, shape);
            let origin = request("origin", &[c, tail]);
            seed_origin(lab, "first", &backend, &origin, &oracle).unwrap();
            // The second run finds both segments and reuses them: it is no origin.
            let error = seed_origin(lab, "second", &backend, &origin, &oracle).unwrap_err();
            assert!(
                matches!(&error, lab::Failure::Invalid(_)),
                "a fixture that is no origin is a harness failure"
            );
            assert!(error.message().contains("not a first run"), "{error:?}");
        });
    }

    /// A probe of any positional run, for the tests that edit what the runtime reported.
    fn some_probe(lab: &mut Lab<'_>) -> ReferenceKvProbe {
        let shape = lab.shape;
        let backend = lab::positional_backend(shape, PositionalControl::None).unwrap();
        let oracle = backend.own_liveness();
        let c = lab::c_part(lab.material, shape);
        let tail = lab::tail_part(lab.material, shape);
        run_positional(
            &mut lab.timings,
            "probe",
            &backend,
            &request("probe", &[c, tail]),
            &oracle,
        )
        .unwrap()
    }

    /// P1-1: the final segment's admission is invisible to `admitted_splices` (the runtime
    /// never splices the last token) and was invisible to the four admission counters; each of
    /// them now counts it.
    #[test]
    fn an_admission_of_the_last_segment_alone_trips_every_negative_case_counter() {
        with_lab(|lab| {
            let mut probe = some_probe(lab);
            let positional = probe.outcome.positional.as_mut().unwrap();
            positional.admitted_splices = 0;
            positional.splices = 0;
            positional.last_token_skips = 1;
            let probe = probe;
            assert_eq!(admitted(report(&probe).unwrap()), 1);
            assert_eq!(reachable(report(&probe).unwrap()), 1);
            // C5 (dead original upstream).
            dead_attempt(lab, "dead", &probe).unwrap();
            assert_eq!(lab.tally.get("shift_hits_with_dead_original_upstream"), 1);
            // C6 (omission): a legit spliced span is not the last segment's skip.
            omission_outcome(lab, "omitted", &probe, 0).unwrap();
            assert_eq!(lab.tally.get("omitted_upstream_admissions"), 1);
            omission_outcome(lab, "multiplicity", &probe, 1).unwrap();
            assert_eq!(
                lab.tally.get("omitted_upstream_admissions"),
                2,
                "a skipped last segment is never one of the legitimate splices"
            );
            // The rules the counters feed trip on the counts (the falsifier text).
            let counters = Json::obj([
                (
                    "omitted_upstream_admissions",
                    Json::Int(lab.tally.get("omitted_upstream_admissions")),
                ),
                (
                    "shift_hits_with_dead_original_upstream",
                    Json::Int(lab.tally.get("shift_hits_with_dead_original_upstream")),
                ),
            ]);
            require_tripped(
                "tail only",
                &counters,
                &[
                    "omitted_upstream_admissions",
                    "shift_hits_with_dead_original_upstream",
                ],
            )
            .unwrap();
        });
    }

    /// A positive case is the other way round: the spliced spans alone are its measure.
    #[test]
    fn a_shifted_parity_run_counts_its_spliced_spans_and_not_the_last_segment_skip() {
        with_lab(|lab| {
            let shape = lab.shape;
            let backend = lab::positional_backend(shape, PositionalControl::None).unwrap();
            let oracle = backend.own_liveness();
            let c = lab::c_part(lab.material, shape);
            let tail = lab::tail_part(lab.material, shape);
            seed_origin(
                lab,
                "origin",
                &backend,
                &request("origin", &[c.clone(), tail.clone()]),
                &oracle,
            )
            .unwrap();
            let shifted = request("shifted", &after(lab.material.filler(7), &[c, tail]));
            let baseline = run_cold(&mut lab.timings, "cold", &backend, &shifted).unwrap();
            let mut warm =
                run_positional(&mut lab.timings, "warm", &backend, &shifted, &oracle).unwrap();
            warm.outcome.positional.as_mut().unwrap().last_token_skips = 3;
            let judged = judge_shifted(lab, "x", &shifted, &warm, &baseline, "shift_7").unwrap();
            assert_eq!(judged.admitted, 1, "the skips are not spliced spans");
        });
    }

    /// P2-3: a reuse of one token is not parity: the reuse unit is the whole segment.
    #[test]
    fn a_partial_reuse_is_not_a_parity_case() {
        with_lab(|lab| {
            let shape = lab.shape;
            let backend = lab::positional_backend(shape, PositionalControl::None).unwrap();
            let oracle = backend.own_liveness();
            let c = lab::c_part(lab.material, shape);
            let tail = lab::tail_part(lab.material, shape);
            seed_origin(
                lab,
                "origin",
                &backend,
                &request("origin", &[c.clone(), tail.clone()]),
                &oracle,
            )
            .unwrap();
            let shifted = request("shifted", &after(lab.material.filler(7), &[c, tail]));
            let baseline = run_cold(&mut lab.timings, "cold", &backend, &shifted).unwrap();
            let mut warm =
                run_positional(&mut lab.timings, "warm", &backend, &shifted, &oracle).unwrap();
            assert!(
                judge_shifted(lab, "full", &shifted, &warm, &baseline, "shift_7")
                    .unwrap()
                    .parity
            );
            warm.outcome.positional.as_mut().unwrap().reused_tokens = 1;
            let partial =
                judge_shifted(lab, "partial", &shifted, &warm, &baseline, "shift_7").unwrap();
            assert_eq!(partial.mismatches, 0, "the bits are equal");
            assert!(
                !partial.parity,
                "one reused token of {} is not parity",
                shape.segment
            );
        });
    }

    /// P1-4: whatever is admitted after a restore into a ledger that never committed `U` is a
    /// shift hit with a dead original upstream. With a reader that did commit it (everything
    /// live) the same flow admits spans, and the metric moves; with the case's reader it stays 0.
    #[test]
    fn an_admission_after_a_restore_into_an_uncommitted_ledger_feeds_the_dead_original_metric() {
        with_lab(|lab| {
            let body = [
                lab::u_part(lab.material, params::UPSTREAM_GENERATION),
                lab::c_part(lab.material, lab.shape),
                lab::tail_part(lab.material, lab.shape),
            ];
            let live =
                |_: ()| Authority::new(&[lab::C_IDENTITY, params::UPSTREAM_IDENTITY]).unwrap();
            let reader = Authority::new(&[lab::C_IDENTITY]).unwrap();
            restore_uncommitted(lab, &body, &live(()), &reader).unwrap();
            assert_eq!(lab.tally.get("shift_hits_with_dead_original_upstream"), 0);
            assert_eq!(lab.tally.get("restore_uncommitted_admissions"), 0);
            restore_uncommitted(lab, &body, &live(()), &live(())).unwrap();
            assert!(lab.tally.get("shift_hits_with_dead_original_upstream") > 0);
            assert_eq!(
                lab.tally.get("shift_hits_with_dead_original_upstream"),
                lab.tally.get("restore_uncommitted_admissions")
            );
        });
    }

    /// P2-6: a runtime that flushes more than the event names loses the bystander, and the
    /// falsifier over it trips.
    #[test]
    fn a_flushed_bystander_is_counted_as_lost_and_trips_its_falsifier() {
        for flush in [false, true] {
            with_lab(|lab| {
                let shape = lab.shape;
                let authority = Authority::new(&[lab::BYSTANDER_IDENTITY]).unwrap();
                let backend = lab::positional_backend(shape, PositionalControl::None).unwrap();
                let parts = bystander_parts(lab);
                seed_origin(lab, "by", &backend, &request("by", &parts), &authority).unwrap();
                if flush {
                    backend
                        .invalidate_inputs(&std::collections::BTreeSet::from([(
                            lab::BYSTANDER_IDENTITY.to_string(),
                            ptr_types::Generation(1),
                        )]))
                        .unwrap();
                }
                bystander_check(lab, "case", &backend, &authority, &parts).unwrap();
                assert_eq!(lab.tally.get("bystander_reuse_cases"), i64::from(!flush));
                assert_eq!(lab.tally.get("bystander_reuse_losses"), i64::from(flush));
                let counters = Json::obj([(
                    "internal",
                    Json::obj([(
                        "bystander_reuse_losses",
                        Json::Int(lab.tally.get("bystander_reuse_losses")),
                    )]),
                )]);
                let rule = "internal.bystander_reuse_losses <= 0";
                assert_eq!(emit::evaluate(&counters, rule), Ok(!flush));
            });
        }
    }

    /// P1-5: an index that takes nothing is a failed control: the case is not counted and the
    /// failure is recorded, never an exit status 2.
    #[test]
    fn an_index_that_takes_no_origin_is_a_failed_control_and_not_an_invalid_run() {
        with_lab(|lab| {
            let _limits = lab::TestLimits::refusing_every_insert();
            let outcome = shift_cases(lab);
            assert!(
                matches!(&outcome, Err(lab::Failure::Control(why)) if why.contains("did not populate")),
                "{outcome:?}"
            );
            lab.settle("C1", outcome).expect("not an exit status 2");
            assert_eq!(lab.tally.get("live_control_failures"), 1);
            assert_eq!(
                lab.tally
                    .get("secondary_index_reuse_cases_per_shift.shift_1"),
                0
            );
            let outcome = upstream_event_case(lab, Event::Revocation);
            lab.settle("C3", outcome).expect("not an exit status 2");
            assert_eq!(lab.tally.get("upstream_revocation_cases"), 0);
            assert_eq!(lab.tally.get("live_control_failures"), 2);
            assert_eq!(lab.failed_controls.len(), 2);
        });
    }

    /// P2-8: a result without a kv_report is a result the harness cannot read, not a tainted one.
    #[test]
    fn a_run_without_a_kv_report_is_a_harness_failure() {
        with_lab(|lab| {
            let mut probe = some_probe(lab);
            probe.outcome.run.kv_report = None;
            let error = note_exact(lab, &probe, 0).unwrap_err();
            assert!(error.contains("kv_report"), "{error}");
            assert_eq!(lab.tally.get("tainted_exact_results"), 0);
        });
    }

    /// P1-6: the cache-insert measure moves when the event closure also performs a live run
    /// that inserts, and stays at 0 without it (the matrix's honest cells).
    #[test]
    fn the_insert_leak_control_moves_the_cache_insert_measure() {
        with_lab(|lab| {
            let leaked = super::super::matrix::insert_leak_inflight(lab).unwrap();
            assert!(leaked.cancelled, "the session itself is stopped");
            assert_eq!(leaked.outputs, 0);
            assert!(leaked.inserts > 0, "{leaked:?}");
            let counters = Json::obj([(
                "shifted_reuse_inflight_cache_inserts",
                Json::Int(leaked.inserts),
            )]);
            require_tripped("leak", &counters, &["shifted_reuse_inflight_cache_inserts"]).unwrap();
        });
    }
}
