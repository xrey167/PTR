//! C6 (an exact echo does not clear) and C7 (a fresh cold exact recomputation clears: the
//! positive control, with its negative controls).

use super::fixture::{exec_session, stamped, Probe, Rt, PROJECT, TTL};
use super::{derive, derive_as, hop_request, own_request, tainted_base, Base, Cx};
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::r012::stub::{self, Reuse};
use ptr_model_api::{KvReuseTaint, ModelRun};
use ptr_runtime::RuntimeError;
use ptr_types::ProjectId;

/// Whether the tainted base looks cleared: a Clear record was committed since `clears_before`,
/// its action is authorized again, or its derived segment is admitted as Exact.
fn looks_cleared(cx: &mut Cx, rt: &mut Rt, base: &Base, clears_before: usize) -> Res<bool> {
    if rt.clear_records() > clears_before {
        return Ok(true);
    }
    if rt
        .runtime
        .authorize_action(&stamped(&rt.runtime, base.action()))
        .is_ok()
    {
        return Ok(true);
    }
    let id = cx.rid("probe");
    let prompt = cx.prompt(4);
    let request = hop_request(&mut rt.runtime, &id, &prompt, &base.derived.segment)?;
    match rt.runtime.admit_segments(&id, &request) {
        Ok(admitted) => Ok(!admitted.segments[1].is_approximate()),
        Err(RuntimeError::UnknownDerivedSegment { .. }) => Ok(false),
        Err(other) => Err(format!("probing the derived segment: {other:?}")),
    }
}

/// C6: an Exact pass whose input includes the tainted Derived segment. Three passes: a genuine
/// cold run over the echo request, the adversarial echo that reproduces the tainted run's own
/// text and action byte for byte under a cold report, and the same output under an Exact warm
/// report. None may clear anything.
pub fn c6_exact_echo(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    cx.sub("C6 setup");
    let base = tainted_base(cx, &mut rt, 0)?;
    let variants: [(&str, Reuse, bool); 3] = [
        ("genuine cold echo", Reuse::Cold, false),
        ("reproducing cold echo", Reuse::Cold, true),
        ("reproducing warm Exact echo", Reuse::ExactWarm, true),
    ];
    for (index, (label, reuse, reproduce)) in variants.into_iter().enumerate() {
        cx.sub(&format!("C6 {label}"));
        let id = cx.rid("echo");
        let prompt = cx.prompt(1 + index);
        let request = hop_request(&mut rt.runtime, &id, &prompt, &base.derived.segment)?;
        cx.check_prompt(&request);
        let run: ModelRun = if reproduce {
            // The strongest echo: the very events of the tainted run (same text, same action).
            ModelRun {
                events: base.run.run.events.clone(),
                kv_report: stub::produce(&cx.backend, &request, reuse)?.run.kv_report,
            }
        } else {
            let produced = stub::produce(&cx.backend, &request, reuse)?;
            cx.check_run(&produced);
            produced.run
        };
        let clears_before = rt.clear_records();
        let derived = derive(&mut rt.runtime, &request, &run).map_err(fail("the echo hop"))?;
        cx.count("exact_echo_cases", 1);
        let input_tainted = derived.taint.is_approximate();
        if !input_tainted || looks_cleared(cx, &mut rt, &base, clears_before)? {
            cx.count("taint_cleared_by_echo", 1);
        }
    }
    // The echo outcome is the same after a restart (a Clear would replay; its absence too).
    cx.sub("C6 after restart");
    let clears = rt.clear_records();
    rt.reopen()?;
    if rt.clear_records() != clears || looks_cleared(cx, &mut rt, &base, clears)? {
        cx.count("taint_cleared_by_echo", 1);
    }
    Ok(())
}

/// What the effect boundary says about `base`'s action once a recompute has run.
struct Recomputed {
    cleared: bool,
}

/// A recompute of `base`'s recipe: a new request id with the same Raw ingress text, a genuine
/// cold run (`reuse`), ingested as a derivation hop under `producer` and `parameters`.
fn recompute(
    cx: &mut Cx,
    rt: &mut Rt,
    base: &Base,
    prompt: &str,
    reuse: Reuse,
    producer: &str,
    parameters: &[u8],
) -> Res<Recomputed> {
    let id = cx.rid("recompute");
    let request = own_request(&mut rt.runtime, &id, prompt)?;
    let run = stub::produce(&cx.backend, &request, reuse)?;
    cx.check_run(&run);
    let clears_before = rt.clear_records();
    derive_as(&mut rt.runtime, &request, &run.run, producer, parameters)
        .map_err(fail("the recompute hop"))?;
    let cleared = looks_cleared(cx, rt, base, clears_before)?;
    Ok(Recomputed { cleared })
}

/// C7: the positive control. A cold exact recompute of the same recipe from the same clean
/// input clears the derivation, the content and the action (the recomputed bytes are
/// identical), and the action is then authorized and executed; the clear survives a restart.
/// The negative controls (another input, another producer, other generation parameters, a warm
/// Exact run, another Approximate run) must not clear: one that does counts as `taint_lost`.
pub fn c7_fresh_cold_recompute(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    cx.sub("C7 positive control");
    let base = tainted_base(cx, &mut rt, 0)?;
    // Still tainted before the recompute (and the attempts here count: before recompute).
    super::action::action_attempts(cx, &mut rt, &base.run.action.clone())?;

    let id = cx.rid("recompute");
    let request = own_request(&mut rt.runtime, &id, &base.prompt)?;
    let cold = stub::produce(&cx.backend, &request, Reuse::Cold)?;
    cx.check_run(&cold);
    let clears_before = rt.clear_records();
    let derived = derive(&mut rt.runtime, &request, &cold.run).map_err(fail("the recompute"))?;
    cx.recomputed = true;

    let mut control = derived.taint == KvReuseTaint::Exact
        && rt.clear_records() == clears_before + 1
        && derived.recipe_digest == base.derived.recipe_digest
        && derived.segment == base.derived.segment
        && rt.runtime.action_kv_taint(base.action()) == KvReuseTaint::Exact
        && rt
            .runtime
            .authorize_action(&stamped(&rt.runtime, base.action()))
            .is_ok();
    // The effect boundary lets the cleared action through all the way to the executor.
    let probe = Probe::default();
    let session = exec_session(&mut rt.runtime, &probe)?;
    let key = cx.key();
    let stamped_action = stamped(&rt.runtime, base.action());
    let executed = rt
        .runtime
        .prepare_execution_once(
            &session,
            &ProjectId::from(PROJECT),
            &stamped_action,
            TTL,
            key,
        )
        .and_then(|permit| rt.runtime.execute_prepared(&session, permit));
    control = control && executed.is_ok() && probe.executions() == 1 && rt.effect_attempts() == 1;
    // The Clear is replayed: after a restart the action is still authorized.
    rt.reopen()?;
    control = control
        && rt
            .runtime
            .authorize_action(&stamped(&rt.runtime, base.action()))
            .is_ok();
    if control {
        cx.count("fresh_exact_recomputation_controls", 1);
    } else {
        cx.note("positive_control_failed");
    }

    // Negative controls, each on a fresh tainted base of its own.
    type Variant = (&'static str, fn(&mut Cx, &mut Rt, &Base) -> Res<Recomputed>);
    let variants: [Variant; 5] = [
        ("C7 other input", |cx, rt, base| {
            let other = cx.prompt(1);
            recompute(
                cx,
                rt,
                base,
                &other,
                Reuse::Cold,
                &stub::identity(),
                stub::parameters(),
            )
        }),
        ("C7 other producer", |cx, rt, base| {
            recompute(
                cx,
                rt,
                base,
                &base.prompt.clone(),
                Reuse::Cold,
                "r012-other-producer/1",
                stub::parameters(),
            )
        }),
        ("C7 other generation parameters", |cx, rt, base| {
            recompute(
                cx,
                rt,
                base,
                &base.prompt.clone(),
                Reuse::Cold,
                &stub::identity(),
                b"max_new_tokens=16;greedy",
            )
        }),
        ("C7 warm Exact recompute", |cx, rt, base| {
            recompute(
                cx,
                rt,
                base,
                &base.prompt.clone(),
                Reuse::ExactWarm,
                &stub::identity(),
                stub::parameters(),
            )
        }),
        ("C7 Approximate recompute", |cx, rt, base| {
            recompute(
                cx,
                rt,
                base,
                &base.prompt.clone(),
                Reuse::Approximate,
                &stub::identity(),
                stub::parameters(),
            )
        }),
    ];
    let skipped = if cx.reduced() { 3 } else { 0 };
    for (label, variant) in variants.into_iter().skip(skipped) {
        cx.sub(label);
        let mut other = Rt::fresh()?;
        let base = tainted_base(cx, &mut other, 0)?;
        let result = variant(cx, &mut other, &base)?;
        cx.note("negative_recompute_controls");
        if result.cleared {
            cx.count("taint_lost", 1);
            cx.note(&format!("cleared_by:{label}"));
        }
    }
    Ok(())
}
