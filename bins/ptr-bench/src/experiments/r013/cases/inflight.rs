//! F4: in-flight calls racing a base-model or an adapter invalidation, on the R010 `KvSession`
//! over the tiny seeded MoE.
//!
//! R011's C11 raced the training and the evaluation input of the called expert. Here the input
//! is the base model every expert of the MoE rests on (`base-model:base-a`: one event fences
//! all sixteen bound experts) or an adapter identity in the called expert's training block
//! (`adapter:lora-1`: one event fences the called expert). A bound session starts while the
//! experts are live; the compound `InputInvalidated` event revokes or supersedes the input and
//! fences every dependent in the same ledger event; the session's next step is the first use
//! after the event and must be cancelled by the completion-time liveness check, so no output
//! escapes. The adapter cells also run R011's replacement path (the same request restarted on the
//! router with the fenced expert masked, compared bitwise with a cold run); the base-model cells
//! cannot (every expert of the MoE is fenced, so no router with the fenced experts masked
//! exists) and check the cancellation, the dead dependencies and the refusal of a later request
//! only.
//!
//! The adapter is an input identity in the training block (the runtime has no adapter role), so
//! its cells exercise the same index as R011's training cells under another identity: weak, and
//! stated as such.

use super::Cx;
use crate::experiments::layered_kv::corpus;
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::lab::{self, fail, refused_not_live, Res, Steps};
use crate::experiments::layered_kv::SEQUENCE_SEED;
use crate::experiments::r010::arms::World;
use crate::experiments::r011::cases::fixture::{
    base_model_input, inputs, spec_at, ReportSpec, Rt, Setup, ADAPTER, D1, D3, REPLACEMENT,
};
use crate::experiments::r011::params as r011_params;
use ptr_model_api::SegmentedModelRequest;
use ptr_pods::{ExpertInputs, ExpertLevel, StructuralOp};
use ptr_runtime::expert::InputInvalidation;
use ptr_runtime::{
    BoundExpert, KvCancelReason, KvCancellation, KvDependency, KvDependencyKind, KvLivenessOracle,
    KvSession, KvSessionFinish, KvSessionPhase, KvSessionStep, MoeBinding, PtrRuntime,
    StackBinding,
};
use ptr_types::{Generation, PodId};
use std::cell::RefCell;
use std::collections::BTreeSet;

const LAYERS: usize = r011_params::TEST_MODEL_LAYERS as usize;
const EXPERTS: usize = r011_params::MOE_TOPOLOGY_NUM_EXPERTS as usize;
const J: usize = r011_params::INFLIGHT_LAYER_J;
const PROMPT: usize = r011_params::INFLIGHT_PROMPT_TOKENS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    BaseModel,
    Adapter,
}

impl Kind {
    fn key(self) -> &'static str {
        match self {
            Self::BaseModel => "base_model",
            Self::Adapter => "adapter",
        }
    }

    /// The block of the called expert.
    fn block(self) -> ExpertInputs {
        match self {
            Self::BaseModel => ExpertInputs::default(),
            Self::Adapter => inputs(&[(ADAPTER, 1, D1)], &[]),
        }
    }

    /// The identity, generation 1 and digest of the input the event invalidates.
    fn input(self) -> (String, [u8; 32]) {
        match self {
            Self::BaseModel => {
                let base = base_model_input();
                (base.identity().to_string(), *base.digest())
            }
            Self::Adapter => (ADAPTER.to_string(), D1),
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

pub fn structural_races(cx: &mut Cx) -> Res<()> {
    let world = World::new()?;
    let cells = [
        (Kind::BaseModel, Trigger::Revocation),
        (Kind::BaseModel, Trigger::Supersession),
        (Kind::Adapter, Trigger::Revocation),
        (Kind::Adapter, Trigger::Supersession),
    ];
    for (kind, trigger) in cx.shuffled("inflight.cells", &cells) {
        run_cell(cx, &world, kind, trigger)?;
    }
    Ok(())
}

/// Birth the sixteen experts of the MoE (the called one resting on `kind`'s block) and bind the
/// MoE to them; returns the live binding, the binding with the called expert masked and the
/// called expert's pod.
fn bind(world: &World, rt: &mut Rt, kind: Kind) -> Res<(StackBinding, StackBinding, String)> {
    let target = world.index;
    lab::commit_capsule(
        &mut rt.runtime,
        lab::BASE_MODEL_IDENTITY,
        lab::BASE_MODEL_GENERATION,
    )?;
    let target_pod = pod_of(J, target);
    for layer in 0..LAYERS {
        for index in 0..EXPERTS {
            let token = ExpertLevel::Token {
                layer: u16::try_from(layer).map_err(fail("the layer"))?,
                index: u16::try_from(index).map_err(fail("the index"))?,
            };
            let pod = pod_of(layer, index);
            let spec = spec_at(&rt.project, &pod, 1, token);
            let op = StructuralOp::Birth { result: spec };
            // Every expert rests on a dependency block, so the base model reaches all of them
            // through the index; only the called one carries the adapter.
            let block = if (layer, index) == (J, target) {
                kind.block()
            } else {
                ExpertInputs::default()
            };
            rt.admit_as(op, Some(block), &ReportSpec::correct(), Setup::PRE)
                .map_err(|error| format!("birth of expert ({layer}, {index}): {error:?}"))?;
        }
    }
    let live = StackBinding::with_moe(MoeBinding {
        moe: world.moe.clone(),
        experts: tracked(rt, target, &target_pod),
    });
    let masked = world
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
/// registry holds for it now.
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

/// A liveness oracle that remembers every dependency it answered "not live" for.
struct Recording<'a> {
    inner: &'a dyn KvLivenessOracle,
    dead: RefCell<Vec<KvDependency>>,
}

impl<'a> Recording<'a> {
    fn new(inner: &'a dyn KvLivenessOracle) -> Self {
        Self {
            inner,
            dead: RefCell::new(Vec::new()),
        }
    }

    fn dead(&self) -> Vec<KvDependency> {
        self.dead.borrow().clone()
    }
}

impl KvLivenessOracle for Recording<'_> {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        let live = self.inner.is_live(dependency);
        if !live {
            self.dead.borrow_mut().push(dependency.clone());
        }
        live
    }
}

fn step_once(session: &mut KvSession<'_>, oracle: &dyn KvLivenessOracle) -> Res<KvSessionStep> {
    match session.phase() {
        KvSessionPhase::Prefilling => session.prefill_chunk(oracle),
        KvSessionPhase::Decoding => session.decode_step(oracle),
        other => return Err(format!("no step in phase {other:?}")),
    }
    .map_err(fail("a session step"))
}

fn steppable(session: &KvSession<'_>) -> bool {
    matches!(
        session.phase(),
        KvSessionPhase::Prefilling | KvSessionPhase::Decoding
    )
}

/// The expert identity a session's dependency on `pod` carries: `expert:{project}/{pod}`.
fn expert_identity(project: &str, pod: &str) -> String {
    format!("expert:{project}/{pod}")
}

/// A cancellation counts as the fence of the dependents only if its reason is a dead
/// dependency (`InputNotLive`, not a tombstoned prefix), the dependencies the check failed on
/// are experts of the fenced set (at least one, and no other kind of dependency: the backend's
/// own base model stays live), and nothing was emitted after the event. Any other cancellation
/// is not attributable to the event: a harness error, never a pass.
fn check_cancellation(
    label: &str,
    cancellation: &KvCancellation,
    dead: &[KvDependency],
    fenced: &BTreeSet<String>,
    tokens_at_event: usize,
) -> Res<()> {
    if cancellation.reason != KvCancelReason::InputNotLive {
        return Err(format!(
            "{label}: cancelled for {:?}, not for a dead dependency",
            cancellation.reason
        ));
    }
    if dead.is_empty() {
        return Err(format!("{label}: the check failed on no dependency"));
    }
    for dependency in dead {
        if dependency.kind != KvDependencyKind::Expert || !fenced.contains(&dependency.identity) {
            return Err(format!(
                "{label}: the dead dependency {:?} {} is not a fenced expert",
                dependency.kind, dependency.identity
            ));
        }
    }
    if cancellation.tokens_emitted != tokens_at_event {
        return Err(format!(
            "{label}: {} tokens were emitted when the event came, {} at the cancellation",
            tokens_at_event, cancellation.tokens_emitted
        ));
    }
    Ok(())
}

/// The un-invalidated control of a cell: the same bound request on a backend of its own runs to
/// its end against the live ledger, completes, and equals the cold run of the live binding.
fn run_control(
    label: &str,
    request: &SegmentedModelRequest,
    live: &StackBinding,
    runtime: &PtrRuntime,
) -> Res<()> {
    let backend = lab::backend()?;
    let mut session = backend
        .begin_session_bound(&lab::scope(), request, None, live, runtime)
        .map_err(fail("beginning the control session"))?;
    while steppable(&session) {
        if let KvSessionStep::Cancelled(cancellation) = step_once(&mut session, runtime)? {
            return Err(format!(
                "{label}: the control session was cancelled with nothing invalidated: {cancellation:?}"
            ));
        }
    }
    let outcome = match session
        .finish(runtime)
        .map_err(fail("finishing the control session"))?
    {
        KvSessionFinish::Completed(outcome) => outcome,
        KvSessionFinish::Cancelled(cancellation) => {
            return Err(format!(
                "{label}: the control session was cancelled at its end: {cancellation:?}"
            ))
        }
    };
    let cold = lab::cold(&backend, request, live)?;
    if !lab::same_result(&outcome, &cold) {
        return Err(format!(
            "{label}: the control session's output differs from the cold run"
        ));
    }
    Ok(())
}

fn run_cell(cx: &mut Cx, world: &World, kind: Kind, trigger: Trigger) -> Res<()> {
    let label = format!("f4.{}.{}", kind.key(), trigger.key());
    cx.sub(&label);
    let mut rt = super::fresh()?;
    let (live, reference, target_pod) = bind(world, &mut rt, kind)?;
    let request = corpus::raw_request(&label, &prompt());
    run_control(&label, &request, &live, &rt.runtime)?;
    cx.note("f4_control_sessions_completed");
    let backend = lab::backend()?;
    let total_steps = Steps::count(PROMPT, 0, r011_params::MAX_NEW_TOKENS as usize);
    // The race point is drawn in 1..=total_steps: at least one step runs before the event, so
    // the call has started; at `total_steps` every step ran and the next use is the finish.
    let race_at = corpus::scenario(cx.seed, &label).range(1, total_steps as u64) as usize;
    cx.fixture
        .push((format!("race_point.{label}"), Json::Int(race_at as i64)));
    let mut session = backend
        .begin_session_bound(&lab::scope(), &request, None, &live, &rt.runtime)
        .map_err(fail("beginning the bound session"))?;
    for _ in 0..race_at {
        if !steppable(&session) {
            cx.note("f4_session_ended_before_the_race_point");
            break;
        }
        if let KvSessionStep::Cancelled(cancellation) = step_once(&mut session, &rt.runtime)? {
            return Err(format!(
                "{label}: cancelled before the event: {cancellation:?}"
            ));
        }
    }
    let tokens_at_event = session.generated().len();

    // The invalidation: one ledger event revokes or supersedes the input and fences every
    // dependent.
    let (identity, digest) = kind.input();
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
    let fenced_pods: BTreeSet<String> = outcome
        .fenced()
        .iter()
        .map(|member| member.pod.0.clone())
        .collect();
    let fenced_identities: BTreeSet<String> = fenced_pods
        .iter()
        .map(|pod| expert_identity(&rt.project.0, pod))
        .collect();
    // The event must have fenced the called expert (a base model event: all sixteen).
    let owed = match kind {
        Kind::BaseModel => LAYERS * EXPERTS,
        Kind::Adapter => 1,
    };
    let fenced_called = fenced_pods.contains(&target_pod);
    cx.count(
        "fence_set_mismatches",
        i64::from(!fenced_called) + (fenced_pods.len() as i64 - owed as i64).abs(),
    );
    let event = rt
        .runtime
        .committed_events()
        .last()
        .map(|committed| committed.event.clone())
        .ok_or_else(|| "no event".to_string())?;
    backend
        .apply_input_invalidation(&event, &outcome.fence_epochs())
        .map_err(fail("applying the invalidation to the catalog"))?;

    // The next use after the event is a step, or the finish of a session whose every step ran
    // before the event; the oracle records the dependencies it answered "dead" for.
    let oracle = Recording::new(&rt.runtime);
    let mut completed_by_finish = false;
    let mut cancellation = None;
    if session.phase() == KvSessionPhase::Finished {
        match session
            .finish(&oracle)
            .map_err(fail("finishing the session"))?
        {
            KvSessionFinish::Cancelled(found) => cancellation = Some(found),
            KvSessionFinish::Completed(_) => completed_by_finish = true,
        }
    } else if let KvSessionStep::Cancelled(found) = step_once(&mut session, &oracle)? {
        cancellation = Some(found);
    }
    let cancelled = cancellation.is_some();
    if let Some(found) = &cancellation {
        check_cancellation(
            &label,
            found,
            &oracle.dead(),
            &fenced_identities,
            tokens_at_event,
        )?;
    }
    let mut escaped = i64::from(!cancelled);
    escaped += (session.generated().len() - tokens_at_event) as i64;
    if !cancelled && !completed_by_finish {
        // Whatever the surviving call produces from here is an escape; finish it.
        while steppable(&session) {
            if let KvSessionStep::Cancelled(_) = step_once(&mut session, &rt.runtime)? {
                break;
            }
        }
        if session.phase() == KvSessionPhase::Finished {
            if let KvSessionFinish::Completed(_) = session
                .finish(&rt.runtime)
                .map_err(fail("finishing the surviving session"))?
            {
                escaped += 1;
            }
        }
    }
    let mut incomplete = 0;
    let mut mismatched = 0;
    if cancelled && kind == Kind::Adapter {
        match session.restart_with(&rt.runtime) {
            Ok(restarted) => {
                let cold = lab::cold(&backend, &request, &reference)?;
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
    // A later request citing the dead experts is refused at the beginning of its session.
    let late = backend.begin_session_bound(&lab::scope(), &request, None, &live, &rt.runtime);
    let late_ok = refused_not_live(&late);
    drop(late);
    escaped += i64::from(!late_ok);
    drop(session);

    cx.attempt(match (kind, trigger) {
        (Kind::BaseModel, Trigger::Revocation) => "inflight_call_base_model_revocation",
        (Kind::BaseModel, Trigger::Supersession) => "inflight_call_base_model_supersession",
        (Kind::Adapter, Trigger::Revocation) => "inflight_call_adapter_revocation",
        (Kind::Adapter, Trigger::Supersession) => "inflight_call_adapter_supersession",
    })?;
    cx.count_in("structural_inflight_race_cases", kind.key(), 1);
    cx.count("structural_inflight_outputs_after_invalidation", escaped);
    cx.count("inflight_fallback_incomplete", incomplete);
    cx.count("inflight_fallback_output_mismatches", mismatched);
    cx.tally.add(
        &format!(
            "internal_inflight_cancelled.{}.{}",
            kind.key(),
            trigger.key()
        ),
        i64::from(cancelled),
    );
    Ok(())
}
