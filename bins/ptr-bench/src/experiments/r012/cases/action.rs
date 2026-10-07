//! C1 (action admission), C2 (restart and compaction) and C3 (the candidate path over hops).

use super::fixture::{stamped, Rt, TTL};
use super::{
    classify, derive, hop_request, is_authorized, is_effect_boundary, own_request, session,
    tainted_base, Base, Cx, Verdict,
};
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::r012::stub::{self, Reuse};
use ptr_model_api::{KvReuseTaint, ModelError};
use ptr_runtime::execution::ExecutionError;
use ptr_runtime::kv_taint::KvTaintRequest;
use ptr_runtime::RuntimeError;
use ptr_types::{ActionIr, ProjectId};

/// Try to authorize `action` twice at the effect boundary: through `authorize_action` and
/// through `prepare_execution_once`. Counts one attempt and, when the runtime lets it
/// through, one admission, per path; an `EffectAttempted` record while the action is tainted
/// is an admission as well.
pub fn action_attempts(cx: &mut Cx, rt: &mut Rt, action: &ActionIr) -> Res<()> {
    let (session, probe) = session(rt)?;
    let stamped_action = stamped(&rt.runtime, action);
    let effects_before = rt.effect_attempts();

    let verdict = classify(
        "authorize_action",
        rt.runtime.authorize_action(&stamped_action),
        is_effect_boundary,
    )?;
    cx.attempt("approximate_action_admission_attempts", rt);
    if verdict == Verdict::Admitted {
        cx.count("approximate_run_actions_admitted", 1);
    }

    let key = cx.key();
    let prepared = rt.runtime.prepare_execution_once(
        &session,
        &ProjectId::from(super::fixture::PROJECT),
        &stamped_action,
        TTL,
        key,
    );
    cx.attempt("approximate_action_admission_attempts", rt);
    match prepared {
        Ok(_) => cx.count("approximate_run_actions_admitted", 1),
        Err(ExecutionError::Audit(inner))
            if matches!(
                *inner,
                RuntimeError::ApproximateStateAtEffectBoundary { .. }
            ) => {}
        Err(other) => {
            return Err(format!(
                "prepare_execution_once: refused with an unexpected error: {other:?}"
            ))
        }
    }
    let attempted = rt.effect_attempts() - effects_before;
    if attempted > 0 || probe.executions() > 0 {
        cx.count(
            "approximate_run_actions_admitted",
            (attempted + probe.executions()) as i64,
        );
    }
    Ok(())
}

/// C1: a stub Approximate run emits an `ActionReady`; the runtime ingests it (the derived-run
/// path, the plain `ingest_model_run_for` path and the server path `run_model_run_resumable`),
/// and the action is then offered to `authorize_action` and to the execution check.
pub fn c1_action_admission(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;

    cx.sub("C1 derived run");
    let base = tainted_base(cx, &mut rt, 0)?;
    action_attempts(cx, &mut rt, &base.run.action.clone())?;

    // The plain path: ingest_model_run_for (an Approximate run, no derived segment).
    cx.sub("C1 plain run");
    let prompt = cx.prompt(1);
    let id = cx.rid("plain");
    let request = own_request(&mut rt.runtime, &id, &prompt)?;
    cx.check_prompt(&request);
    let run = stub::produce(&cx.backend, &request, Reuse::Approximate)?;
    cx.check_run(&run);
    let ingested = rt
        .runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request,
                producer: &stub::identity(),
                generation_parameters: stub::parameters(),
            },
            &run.run,
        )
        .map_err(fail("ingest_model_run_for"))?;
    if !ingested.kv_taint.is_approximate() {
        cx.count("approximate_run_actions_admitted", 1);
    }
    action_attempts(cx, &mut rt, &run.action)?;

    // The server path: run_model_run_resumable with an Approximate backend run.
    cx.sub("C1 server path");
    let prompt = cx.prompt(2);
    let id = cx.rid("server");
    let request = segmented_for_server(&prompt, &id)?;
    cx.check_prompt(&request);
    let run = stub::produce(&cx.backend, &request, Reuse::Approximate)?;
    cx.check_run(&run);
    let model_run = run.run.clone();
    let resumed = rt
        .runtime
        .run_model_run_resumable(id, prompt, |_| Ok::<_, ModelError>(model_run))
        .map_err(fail("run_model_run_resumable"))?;
    if !resumed.kv_taint.is_approximate() {
        cx.count("approximate_run_actions_admitted", 1);
    }
    action_attempts(cx, &mut rt, &run.action)?;

    // A request id that was already tainted, reused for a fresh request and an Exact run:
    // the action of that run must be refused too (a live Request subject taints it).
    cx.sub("C1 request id reuse");
    let prompt = cx.prompt(3);
    let reused = base.request.request_id().clone();
    let request = segmented_for_server(&prompt, &reused)?;
    cx.check_prompt(&request);
    let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
    cx.check_run(&run);
    let model_run = run.run.clone();
    rt.runtime
        .run_model_run_resumable(reused, prompt, |_| Ok::<_, ModelError>(model_run))
        .map_err(fail("run_model_run_resumable on a reused request id"))?;
    cx.note("request_id_reuse_runs");
    action_attempts(cx, &mut rt, &run.action)?;
    Ok(())
}

/// The request the server path's backend would see: the request's own text as one Raw
/// segment (it is ingested by `run_model_run_resumable` itself).
fn segmented_for_server(
    prompt: &str,
    id: &ptr_types::RequestId,
) -> Res<ptr_model_api::SegmentedModelRequest> {
    super::segmented(id, vec![super::raw_segment(prompt)?])
}

/// What a reload check found.
fn reload_checks(cx: &mut Cx, rt: &mut Rt, base: &Base, label: &str) -> Res<()> {
    // The action is still refused (taint persisted).
    if !rt
        .runtime
        .action_kv_taint(&base.run.action)
        .is_approximate()
    {
        cx.count("taint_lost", 1);
        cx.note(&format!("lost_action_{label}"));
    }
    // The persisted candidate (the derived segment) is presented again: still Approximate.
    let id = cx.rid("reload");
    let prompt = cx.prompt(1);
    let request = hop_request(&mut rt.runtime, &id, &prompt, &base.derived.segment)?;
    match rt.runtime.admit_segments(&id, &request) {
        Ok(admitted) => {
            cx.attempt("tainted_candidate_reload_attempts", rt);
            if !admitted
                .segments
                .get(1)
                .is_some_and(KvReuseTaint::is_approximate)
            {
                cx.count("taint_lost", 1);
                cx.note(&format!("lost_candidate_{label}"));
            }
        }
        Err(RuntimeError::UnknownDerivedSegment { .. }) => {
            return Err(format!(
                "after {label} the registry-held tainted candidate was refused as an unknown Derived segment"
            ))
        }
        Err(other) => return Err(format!("admit_segments after {label}: {other:?}")),
    }
    // Its text, resubmitted as the request's own Raw text, is still tainted.
    let id = cx.rid("reload-text");
    let request = own_request(&mut rt.runtime, &id, base.text())?;
    let admitted = rt
        .runtime
        .admit_segments(&id, &request)
        .map_err(fail("admit_segments of the tainted text"))?;
    if !admitted.taint.is_approximate() {
        cx.count("taint_lost", 1);
        cx.note(&format!("lost_content_{label}"));
    }
    // The action is offered to the boundary again.
    action_attempts(cx, rt, &base.run.action.clone())
}

/// C2: after the record, reopen the ledger with `open_durable` (twice), then export a compacted
/// snapshot and restore it; re-attempt C1 and reload the candidate each time.
pub fn c2_restart_and_compaction(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    cx.sub("C2 before restart");
    let base = tainted_base(cx, &mut rt, 0)?;
    action_attempts(cx, &mut rt, &base.run.action.clone())?;

    cx.sub("C2 reopen");
    rt.reopen()?;
    reload_checks(cx, &mut rt, &base, "reopen")?;
    if !cx.reduced() {
        rt.reopen()?;
        reload_checks(cx, &mut rt, &base, "second_reopen")?;
    }

    cx.sub("C2 compaction");
    rt.compact_restore()?;
    reload_checks(cx, &mut rt, &base, "compaction")?;
    Ok(())
}

/// C3: the Approximate result is persisted as a candidate (its derived segment) and re-fed into
/// otherwise-Exact runs over `hops` hops; each hop is an admission attempt. The runtime is
/// reopened after the first hop and compacted after the second, so the chain crosses both.
pub fn c3_candidate_path(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    cx.sub("C3 hop 0");
    let base = tainted_base(cx, &mut rt, 0)?;
    let hops = cx.hops();
    let mut previous = base.derived.segment.clone();
    let mut reached = 0usize;
    for hop in 1..=hops {
        cx.sub(&format!("C3 hop {hop}"));
        let id = cx.rid("hop");
        let prompt = cx.prompt(hop);
        let request = hop_request(&mut rt.runtime, &id, &prompt, &previous)?;
        cx.check_prompt(&request);
        // Otherwise Exact: a cold run, except hop 2, which carries an Exact warm report.
        let reuse = if hop == 2 {
            Reuse::ExactWarm
        } else {
            Reuse::Cold
        };
        let run = stub::produce(&cx.backend, &request, reuse)?;
        cx.check_run(&run);
        let derived = derive(&mut rt.runtime, &request, &run.run)
            .map_err(|error| format!("hop {hop}: refused by the runtime: {error:?}"))?;
        cx.attempt("approximate_candidate_path_attempts", &rt);
        let authorized = is_authorized(&rt.runtime, &run.action)?;
        if !derived.taint.is_approximate() || authorized {
            cx.count("approximate_candidate_exact_admissions", 1);
            break;
        }
        reached = hop;
        previous = derived.segment.clone();
        if hop == 1 && hops > 1 {
            rt.reopen()?;
        } else if hop == 2 && hops > 2 {
            rt.compact_restore()?;
        }
    }
    let best = cx.tally.get("candidate_taint_hops").max(reached as i64);
    cx.tally.set("candidate_taint_hops", best);
    cx.count("chain_hops_reached", reached as i64);
    // The chain must reach the preregistered hop count unless a measured violation broke it;
    // a chain that stopped for another reason is a harness error.
    if reached < hops && cx.tally.get("approximate_candidate_exact_admissions") == 0 {
        return Err(format!("the chain stopped at hop {reached} of {hops}"));
    }
    Ok(())
}
