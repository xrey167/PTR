//! C11: in-flight expert calls on the R010 `KvSession` over the tiny seeded MoE.
//!
//! A bound session starts while the expert it calls is live; the compound `InputInvalidated`
//! event revokes (or supersedes) an input the expert rests on and fences the expert in the
//! same ledger event; the session's next step is the first use after the event and must be
//! cancelled by the completion-time liveness check, so no output escapes; the replacement path
//! (the same request restarted on the router with the fenced expert masked) must complete with
//! the reference output.
//!
//! L1 is an expert registered at `ExpertLevel::Token` (it fills a routing slot of the MoE), L2
//! one registered at `ExpertLevel::Segment`. The `KvSession` depends on an expert through the
//! lifecycle identity `expert:{project}/{pod}` of the pod it is bound to, whatever the pod's
//! level, so the L2 cells bind the MoE slot of the call to a Segment-level pod: the level of
//! the registered expert differs, the mechanism under test (the ledger-anchored oracle) does
//! not. The harness says so under `measurement_notes`.

use super::fixture::{
    inputs, spec_at, ReportSpec, Rt, Setup, D1, D2, D3, EVAL, OTHER, REPLACEMENT, TRAIN,
};
use super::Cx;
use crate::experiments::layered_kv::corpus;
use crate::experiments::layered_kv::lab::{self, fail, refused_not_live, Res, Steps};
use crate::experiments::layered_kv::SEQUENCE_SEED;
use ptr_pods::{ExpertLevel, StructuralOp};
use ptr_runtime::expert::InputInvalidation;
use ptr_runtime::{BoundExpert, KvSessionFinish, KvSessionPhase, MoeBinding, StackBinding};
use ptr_types::{Generation, PodId};

const LAYERS: usize = super::super::params::TEST_MODEL_LAYERS as usize;
const EXPERTS: usize = super::super::params::MOE_TOPOLOGY_NUM_EXPERTS as usize;
const J: usize = super::super::params::INFLIGHT_LAYER_J;
const PROMPT: usize = super::super::params::INFLIGHT_PROMPT_TOKENS;
/// The pod of the Segment-level expert the L2 cells bind the call to.
const SEGMENT_POD: &str = "seg-expert";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Level {
    L1,
    L2,
}

impl Level {
    fn key(self) -> &'static str {
        match self {
            Self::L1 => "L1",
            Self::L2 => "L2",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Training,
    Evaluation,
}

impl Role {
    fn key(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Evaluation => "evaluation",
        }
    }

    fn block(self) -> ptr_pods::ExpertInputs {
        match self {
            Self::Training => inputs(&[(TRAIN, 1, D1)], &[]),
            Self::Evaluation => inputs(&[(OTHER, 1, D3)], &[(EVAL, 1, D2)]),
        }
    }

    fn input(self) -> (&'static str, [u8; 32]) {
        match self {
            Self::Training => (TRAIN, D1),
            Self::Evaluation => (EVAL, D2),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Trigger {
    Revocation,
    Supersession,
}

impl Trigger {
    fn key(self) -> &'static str {
        match self {
            Self::Revocation => "revocation",
            Self::Supersession => "supersession",
        }
    }
}

/// The pod of the expert `(layer, index)`.
fn pod_of(layer: usize, index: usize) -> String {
    format!("moe-l{layer}-e{index}")
}

fn prompt() -> Vec<u32> {
    corpus::sequence(SEQUENCE_SEED, 3_000, PROMPT)
}

pub fn c11_inflight_calls(cx: &mut Cx) -> Res<()> {
    // The six distinct cells: L1 over both input roles and both triggers, L2 over the training
    // input and both triggers. An L1 run is also the in-flight cell of the matrix for its role
    // and trigger.
    let cells = [
        (Level::L1, Role::Training, Trigger::Revocation),
        (Level::L1, Role::Training, Trigger::Supersession),
        (Level::L1, Role::Evaluation, Trigger::Revocation),
        (Level::L1, Role::Evaluation, Trigger::Supersession),
        (Level::L2, Role::Training, Trigger::Revocation),
        (Level::L2, Role::Training, Trigger::Supersession),
    ];
    for (level, role, trigger) in cx.shuffled("c11.cells", &cells) {
        run_cell(cx, level, role, trigger)?;
    }
    Ok(())
}

/// Birth the 16 experts of the MoE (the target one resting on `role`'s input) and bind the
/// MoE to them.
fn bind(
    cx: &Cx,
    rt: &mut Rt,
    level: Level,
    role: Role,
) -> Res<(StackBinding, StackBinding, String)> {
    let target = cx.world.index;
    // The backend binds its base model by the lifecycle identity `r005-tiny`; the ledger must
    // hold it live.
    lab::commit_capsule(
        &mut rt.runtime,
        lab::BASE_MODEL_IDENTITY,
        lab::BASE_MODEL_GENERATION,
    )?;
    let target_pod = match level {
        Level::L1 => pod_of(J, target),
        Level::L2 => SEGMENT_POD.to_string(),
    };
    for layer in 0..LAYERS {
        for index in 0..EXPERTS {
            let token = ExpertLevel::Token {
                layer: u16::try_from(layer).map_err(fail("the layer"))?,
                index: u16::try_from(index).map_err(fail("the index"))?,
            };
            if (layer, index) == (J, target) {
                let (spec, block) = match level {
                    Level::L1 => (spec_at(&rt.project, &target_pod, 1, token), role.block()),
                    Level::L2 => (rt.segment_spec(&target_pod, 1), role.block()),
                };
                let op = StructuralOp::Birth { result: spec };
                rt.admit_as(op, Some(block), &ReportSpec::correct(), Setup::PRE)
                    .map_err(|error| format!("birth of the called expert: {error:?}"))?;
            } else {
                let op = StructuralOp::Birth {
                    result: spec_at(&rt.project, &pod_of(layer, index), 1, token),
                };
                rt.admit(op)
                    .map_err(|error| format!("birth of expert ({layer}, {index}): {error:?}"))?;
            }
        }
    }
    let live = StackBinding::with_moe(MoeBinding {
        moe: cx.world.moe.clone(),
        experts: tracked(rt, target, &target_pod),
    });
    // The replacement path: the router with the called expert masked.
    let masked = cx
        .world
        .moe
        .clone()
        .set_expert_live(J, target, false)
        .map_err(fail("masking the called expert"))?;
    let reference = StackBinding::with_moe(MoeBinding {
        moe: masked,
        experts: tracked(rt, target, &target_pod),
    });
    Ok((live, reference, target_pod))
}

/// Every live expert of every MoE layer, each recording the fence epoch (the state version) the
/// registry holds for it now; the called slot is bound to `target_pod`.
fn tracked(rt: &Rt, target: usize, target_pod: &str) -> Vec<BoundExpert> {
    let mut bound = Vec::new();
    for layer in 0..LAYERS {
        for index in 0..EXPERTS {
            let pod = if (layer, index) == (J, target) {
                target_pod.to_string()
            } else {
                pod_of(layer, index)
            };
            bound.push(BoundExpert {
                layer,
                index,
                project: rt.project.clone(),
                fence_epoch: rt.state(&pod).map_or(0, |(_, version)| version),
                pod: PodId::from(pod.as_str()),
            });
        }
    }
    bound
}

fn run_cell(cx: &mut Cx, level: Level, role: Role, trigger: Trigger) -> Res<()> {
    let label = format!("c11.{}.{}.{}", level.key(), role.key(), trigger.key());
    cx.sub(&label);
    let mut rt = Rt::fresh()?;
    let (live, reference, target_pod) = bind(cx, &mut rt, level, role)?;
    let request = corpus::raw_request(&label, &prompt());
    let total_steps = Steps::count(PROMPT, 0, super::super::params::MAX_NEW_TOKENS as usize);
    let race_at = corpus::scenario(cx.seed, &label).range(0, total_steps as u64) as usize;
    cx.fixture.push((
        format!("race_point.{label}"),
        crate::experiments::layered_kv::emit::Json::Int(race_at as i64),
    ));
    // The call starts while the expert is live.
    let session = cx
        .backend
        .begin_session_bound(&lab::scope(), &request, None, &live, &rt.runtime)
        .map_err(fail("beginning the bound session"))?;
    let mut steps = Steps { session };
    for _ in 0..race_at {
        if steps.step(&rt.runtime)? {
            return Err(format!("{label}: cancelled before the event"));
        }
    }
    let tokens_at_event = steps.session.generated().len();

    // The invalidation: one ledger event revokes or supersedes the input and fences the expert.
    let (identity, digest) = role.input();
    let request_event = match trigger {
        Trigger::Revocation => InputInvalidation::revoked(identity, Generation(1), digest),
        Trigger::Supersession => InputInvalidation::superseded(
            identity,
            Generation(1),
            digest,
            Generation(2),
            D3,
            REPLACEMENT,
        ),
    };
    let outcome = rt
        .invalidate(request_event)
        .map_err(|error| format!("{label}: {error:?}"))?;
    if !outcome
        .fenced()
        .iter()
        .any(|member| member.pod.0 == target_pod)
    {
        return Err(format!(
            "{label}: the event did not fence the called expert"
        ));
    }
    let event = rt
        .runtime
        .committed_events()
        .last()
        .map(|committed| committed.event.clone())
        .ok_or_else(|| "no event".to_string())?;
    cx.backend
        .apply_input_invalidation(&event, &outcome.fence_epochs())
        .map_err(fail("applying the invalidation to the catalog"))?;

    // The next use after the event is a step, or the finish of a session whose every step ran
    // before the event.
    let mut completed_by_finish = false;
    let cancelled = if steps.session.phase() == KvSessionPhase::Finished {
        match steps
            .session
            .finish(&rt.runtime)
            .map_err(fail("finishing the session"))?
        {
            KvSessionFinish::Cancelled(_) => true,
            KvSessionFinish::Completed(_) => {
                completed_by_finish = true;
                false
            }
        }
    } else {
        steps.step(&rt.runtime)?
    };
    let mut escaped = i64::from(!cancelled);
    escaped += (steps.session.generated().len() - tokens_at_event) as i64;
    if !cancelled && !completed_by_finish {
        // Whatever the surviving call produces from here is an escape; finish it.
        if let KvSessionFinish::Completed(_) = steps
            .session
            .finish(&rt.runtime)
            .map_err(fail("finishing the surviving session"))?
        {
            escaped += 1;
        }
    }
    let mut incomplete = 0;
    let mut mismatched = 0;
    if cancelled {
        match steps.session.restart_with(&rt.runtime) {
            Ok(restarted) => {
                let cold = lab::cold(&cx.backend, &request, &reference)?;
                let fresh = restarted.hit_tokens == 0
                    && restarted.prefilled_tokens == request.total_tokens()
                    && restarted.inserted_chunks == 0
                    && restarted.restarted_after_cancel;
                incomplete += i64::from(!fresh);
                mismatched += i64::from(!lab::same_result(&restarted, &cold));
            }
            Err(_) => incomplete += 1,
        }
    }
    // A later request citing the dead expert is refused at the beginning of its session.
    let late = cx
        .backend
        .begin_session_bound(&lab::scope(), &request, None, &live, &rt.runtime);
    let late_ok = refused_not_live(&late);
    drop(late);
    escaped += i64::from(!late_ok);

    drop(steps);
    count_cell(
        cx, level, role, trigger, escaped, incomplete, mismatched, cancelled,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn count_cell(
    cx: &mut Cx,
    level: Level,
    role: Role,
    trigger: Trigger,
    escaped: i64,
    incomplete: i64,
    mismatched: i64,
    cancelled: bool,
) -> Res<()> {
    let case = match trigger {
        Trigger::Revocation => "inflight_expert_call_race",
        Trigger::Supersession => "supersession_inflight_expert_call_race",
    };
    cx.attempt(case)?;
    cx.count_in(
        "inflight_expert_call_race_cases",
        &format!("{}.{}", level.key(), trigger.key()),
        1,
    );
    cx.count_in("outputs_after_invalidation", trigger.key(), escaped);
    cx.count("inflight_expert_call_fallback_incomplete", incomplete);
    cx.count(
        "inflight_expert_call_fallback_output_mismatches",
        mismatched,
    );
    cx.tally.add(
        &format!(
            "internal_inflight_cancelled.{}.{}.{}",
            level.key(),
            role.key(),
            trigger.key()
        ),
        i64::from(cancelled),
    );
    match trigger {
        Trigger::Revocation => cx.count("inflight_expert_outputs_after_revocation", escaped),
        Trigger::Supersession => {
            cx.count("supersession_inflight_expert_call_cases", 1);
            cx.count("supersession_output_cases", 1);
        }
    }
    if level == Level::L1 {
        let cell = format!("{}.{}", role.key(), trigger.key());
        cx.count_in("inflight_input_race_cases", &cell, 1);
        cx.count_in("inflight_outputs_after_input_invalidation", &cell, escaped);
        let matrix = format!("{}_input_inflight_{}", role.key(), trigger.key());
        cx.count_in("invalidation_matrix_cases", &matrix, 1);
        cx.count_in("invalidation_matrix_stale_hits", &matrix, escaped);
    }
    Ok(())
}
