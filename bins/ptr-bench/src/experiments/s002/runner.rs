//! One case in one arm: the cooperative scheduler, the oracle's step-by-step expectations and
//! the counting.
//!
//! The scheduler is a virtual-time event queue (`vtime`, ordered by `(tick, seq)`): the steps of
//! the run, the writer and the boundary checkpoints are events, a step is [`params::STEP_TICKS`]
//! after the previous one and a writer acts half a step before the event it precedes. Nothing
//! reads a clock.
//!
//! Every call into the runtime goes through [`guard`], so a panic of the product is a counted
//! falsification (`product_panics`) and the case is abandoned, not a crashed harness. Before
//! each step the oracle decides from the mirror alone whether the run is stale and arm A1 must
//! cancel it; the step's result is compared with that decision. After each write the mirror's
//! revision is compared with the runtime's. The harness' own failures (a write it makes being
//! refused) are `Err`, which is exit 2.
//!
//! The host's policy is the same in both arms: a run that is cancelled, or whose action the
//! boundary refuses as stale, is begun again once, on the current revision, and its action is
//! executed under the case's at-most-once key; a retry of an executed action under that key is
//! offered once and must be answered from history.

use super::metrics::{Counters, A0, A1};
use super::oracle::{self, Staleness};
use super::params::{self, Position};
use super::plan::{CasePlan, Injection, Writer};
use super::world::{self, Chain, PassPayload, World};
use crate::experiments::vtime::EventQueue;
use ptr_runtime::execution::ExecutionPermit;
use ptr_runtime::{RunStep, StepResources, SteppedRun};
use ptr_types::{ActionIr, RequestId};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Runs `f`; a panic is counted and gives `None`.
pub fn guard<T>(counters: &mut Counters, f: impl FnOnce() -> T) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => Some(value),
        Err(_) => {
            counters.product_panics += 1;
            None
        }
    }
}

/// Why the boundary said no.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reason {
    Revision,
    Generation,
    Permit,
    Other,
}

impl Reason {
    pub fn of(error: &str) -> Self {
        if error.contains("StalePermit") {
            Self::Permit
        } else if error.contains("StaleRevision") {
            Self::Revision
        } else if error.contains("StaleGeneration") || error.contains("UnknownGeneration") {
            Self::Generation
        } else {
            Self::Other
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Revision => "refused-stale-revision",
            Self::Generation => "refused-stale-generation",
            Self::Permit => "refused-stale-permit",
            Self::Other => "refused-other",
        }
    }
}

/// The outcome of every checkpoint of one boundary pass, for the comparison of the arms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PassRecord {
    pub action: String,
    pub checkpoints: Vec<(&'static str, &'static str)>,
}

#[derive(Debug)]
pub struct ArmReport {
    pub first_pass: Option<PassRecord>,
    pub aborted: bool,
}

enum RunEnd {
    Finished(ActionIr),
    Cancelled,
    Failed,
}

enum StepEnd {
    Pending,
    Finished(ActionIr),
    Cancelled,
    Failed,
}

#[derive(Default)]
struct Pass {
    executed: bool,
    refused: Option<Reason>,
    record: Vec<(&'static str, &'static str)>,
    stale_seen: bool,
    refusal_counted: bool,
}

#[derive(Clone, Copy)]
enum Checkpoint {
    Authorize,
    Prepare,
    Execute,
}

impl Checkpoint {
    fn name(self) -> &'static str {
        match self {
            Self::Authorize => "authorize",
            Self::Prepare => "prepare",
            Self::Execute => "execute",
        }
    }
}

#[derive(Clone, Copy)]
enum RunEvent {
    Inject,
    Step,
}

#[derive(Clone, Copy)]
enum PassEvent {
    Inject,
    Authorize,
    Prepare,
    Execute,
}

struct Ctx<'a> {
    plan: &'a CasePlan,
    arm: usize,
    world: World,
    c: &'a mut Counters,
    aborted: bool,
    /// The scheduler's clock at the end of the last phase.
    tick: u64,
    steps_started: i64,
    injected_at_step: Option<i64>,
    just_injected: Option<Writer>,
    /// A bumping write happened during the active run: every step it takes from now on is
    /// work on a snapshot that has been replaced.
    run_poisoned: bool,
    stale_revision_seen: bool,
    stale_generation_seen: bool,
}

const MAX_ROUNDS: usize = 16;

impl<'a> Ctx<'a> {
    fn new(plan: &'a CasePlan, arm: usize, c: &'a mut Counters) -> Result<Self, String> {
        let world = World::new(arm == A1, plan.effect)?;
        Ok(Self {
            plan,
            arm,
            world,
            c,
            aborted: false,
            tick: 0,
            steps_started: 0,
            injected_at_step: None,
            just_injected: None,
            run_poisoned: false,
            stale_revision_seen: false,
            stale_generation_seen: false,
        })
    }

    fn arm_name(&self) -> &'static str {
        if self.arm == A0 {
            "a0"
        } else {
            "a1"
        }
    }

    /// The runtime's revision against the mirror's; a difference is counted.
    fn check_drift(&mut self) {
        let mirror = self.world.with_mirror(|m| m.revision());
        if self.world.runtime.revision().0 != mirror {
            self.c.mirror_drift += 1;
        }
    }

    /// Whether `action` is stale now, judged by a replay of the committed ledger (the runtime's
    /// own records), not by the mirror. The mirror's verdict is compared with it: a difference
    /// is the harness' model lagging or the runtime writing outside the harness' view, and is
    /// counted as `mirror_drift`, never as a stale execution or admission.
    fn judge(&mut self, action: &ActionIr) -> Staleness {
        let ledger = oracle::LedgerState::replay(self.world.runtime.committed_events()).staleness(
            &action.target,
            action.revision.0,
            action.generation.0,
        );
        let mirror = self
            .world
            .with_mirror(|m| m.staleness(action.revision.0, action.generation.0));
        if ledger != mirror {
            self.c.mirror_drift += 1;
        }
        ledger
    }

    fn inject(&mut self, injection: Injection) -> Result<(), String> {
        match injection.writer {
            Writer::Bump => self.world.write_fresh_key()?,
            Writer::NoOp => {
                let runtime_before = self.world.runtime.revision();
                let mirror_before = self.world.with_mirror(|m| m.revision());
                self.world
                    .write_foreign(world::BASE_KEY, world::BASE_VALUE)?;
                let mirror_after = self.world.with_mirror(|m| m.revision());
                // Counted when both the mirror and the runtime say nothing moved.
                if mirror_before == mirror_after && self.world.runtime.revision() == runtime_before
                {
                    self.c.noop_writes_observed += 1;
                }
            }
            Writer::Supersede => self.world.supersede_generation()?,
            Writer::Revoke => self.world.revoke_generation()?,
        }
        self.c.positions.insert(injection.position);
        self.check_drift();
        Ok(())
    }

    // ---- the run ------------------------------------------------------------------------

    fn injection_tick(&self, position: Position) -> u64 {
        let half = params::STEP_TICKS / 2;
        let rounds = self.plan.rounds;
        match position {
            Position::BeforeRound1 => half,
            Position::MidRun => {
                let before_step = 1 + (self.plan.index as u64 % (rounds - 1));
                before_step * params::STEP_TICKS + half
            }
            Position::JustBeforeAction => rounds * params::STEP_TICKS + half,
            Position::BetweenActionAndAuthorize | Position::BetweenPrepareAndExecute => 0,
        }
    }

    /// Begins the run `request` and steps it to its end. `early` is a writer that acts while it
    /// is stepping.
    fn run_phase(&mut self, request: &str, early: Option<Injection>) -> Result<RunEnd, String> {
        self.steps_started = 0;
        self.injected_at_step = None;
        self.just_injected = None;
        self.run_poisoned = false;
        let rounds = self.plan.rounds;
        let backend = Chain {
            rounds,
            effect: self.plan.effect,
        };
        let project = self.world.project.clone();
        let begun = guard(self.c, || {
            self.world.runtime.begin_resumable_run(
                RequestId::from(request),
                &project,
                format!("start {request}"),
                &backend,
                MAX_ROUNDS,
            )
        });
        let mut run = match begun {
            None => {
                self.aborted = true;
                return Ok(RunEnd::Failed);
            }
            Some(Err(_)) => {
                self.c.mirror_drift += 1;
                return Ok(RunEnd::Failed);
            }
            Some(Ok(run)) => run,
        };
        self.world.with_mirror(|m| m.run_ingests());
        self.check_drift();

        let base = self.tick;
        let mut queue: EventQueue<RunEvent> = EventQueue::new();
        queue
            .schedule(base + params::STEP_TICKS, RunEvent::Step)
            .map_err(|late| format!("scheduling the first step: {late:?}"))?;
        if let Some(injection) = early {
            queue
                .schedule(
                    base + self.injection_tick(injection.position),
                    RunEvent::Inject,
                )
                .map_err(|late| format!("scheduling the writer: {late:?}"))?;
        }
        let mut step_index = 0u64;
        while let Some(fired) = queue.pop() {
            match fired.event {
                RunEvent::Inject => {
                    let injection = early.ok_or("an injection event without a writer")?;
                    self.inject(injection)?;
                    self.injected_at_step = Some(self.steps_started);
                    self.just_injected = Some(injection.writer);
                    self.run_poisoned |= injection.writer.moves_revision();
                }
                RunEvent::Step => {
                    let ended = self.step(&mut run, &backend, step_index, request);
                    step_index += 1;
                    match ended {
                        StepEnd::Pending => {
                            queue
                                .schedule(fired.tick + params::STEP_TICKS, RunEvent::Step)
                                .map_err(|late| format!("scheduling a step: {late:?}"))?;
                        }
                        StepEnd::Finished(action) => {
                            self.tick = fired.tick;
                            return Ok(RunEnd::Finished(action));
                        }
                        StepEnd::Cancelled => {
                            self.tick = fired.tick;
                            return Ok(RunEnd::Cancelled);
                        }
                        StepEnd::Failed => {
                            self.tick = fired.tick;
                            return Ok(RunEnd::Failed);
                        }
                    }
                }
            }
        }
        Err("the run's scheduler ran dry before the run ended".to_string())
    }

    fn account_units(&mut self, units: i64) {
        self.c.total_units[self.arm] += units;
        if self.run_poisoned {
            self.c.wasted_units[self.arm] += units;
        }
    }

    fn step(
        &mut self,
        run: &mut SteppedRun,
        backend: &Chain,
        index: u64,
        request: &str,
    ) -> StepEnd {
        let rounds = self.plan.rounds;
        let stale_before = self.world.with_mirror(|m| m.run_is_stale());
        let should_cancel = self.arm == A1 && stale_before;
        if let Some(writer) = self.just_injected.take() {
            // The schedule says whether the writer moved the revision; the mirror must agree.
            if stale_before != writer.moves_revision() {
                self.c.mirror_drift += 1;
            }
        }
        if stale_before {
            self.stale_revision_seen = true;
        }
        self.steps_started += 1;
        let pods_before = self.world.pod_runs();
        let ledger_before = self.world.runtime.committed_events().len();
        let revision_before = self.world.runtime.revision();
        let cancels_before = self.cancel_events();
        let resources = StepResources {
            backend,
            pods: &self.world.pods,
            router: None,
            verifier: &PassPayload,
        };
        let outcome = guard(self.c, || {
            self.world.runtime.step_resumable_run(run, &resources)
        });
        let work = (self.world.pod_runs() - pods_before) as i64;
        let result = match outcome {
            None => {
                self.aborted = true;
                return StepEnd::Failed;
            }
            Some(Err(_)) => {
                // The scripted run cannot fail by itself: the runtime refused a step the
                // harness' model says is valid.
                self.c.mirror_drift += 1;
                return StepEnd::Failed;
            }
            Some(Ok(result)) => result,
        };
        match result {
            RunStep::Pending => {
                if index >= rounds || work != 1 {
                    self.c.mirror_drift += 1;
                } else {
                    self.world
                        .with_mirror(|m| m.run_promotes(request, &world::round_payload(index)));
                }
                self.account_units(work);
                self.check_drift();
                if should_cancel {
                    self.c.missed_cancellations += 1;
                }
                StepEnd::Pending
            }
            RunStep::Finished(finished) => {
                let action = finished.action;
                if index != rounds || work != 0 || action.is_none() {
                    self.c.mirror_drift += 1;
                }
                self.account_units(work + 1);
                self.check_drift();
                if should_cancel {
                    self.c.missed_cancellations += 1;
                }
                match action {
                    Some(action) => StepEnd::Finished(action),
                    None => StepEnd::Failed,
                }
            }
            RunStep::Cancelled(cancelled) => {
                self.c.cancelled_runs += 1;
                if !should_cancel {
                    self.c.false_cancellations += 1;
                }
                let (run_bound, revision) =
                    self.world.with_mirror(|m| (m.run_bound(), m.revision()));
                let mut violation = false;
                // A cancelled run hands out no action and keeps what it had committed.
                violation |= cancelled.run.action.is_some();
                violation |= cancelled.run.observations.len() as u64 != index;
                violation |= cancelled.request_id != RequestId::from(request);
                // It reports the revisions the mirror holds.
                violation |= cancelled.bound.0 != run_bound || cancelled.current.0 != revision;
                // The cancelled step did no work, wrote nothing and moved nothing.
                violation |= work != 0
                    || self.world.runtime.committed_events().len() != ledger_before
                    || self.world.runtime.revision() != revision_before;
                // It said so once.
                violation |= self.cancel_events() != cancels_before + 1;
                let expected = ptr_runtime::RuntimeError::RunCancelled {
                    bound: cancelled.bound,
                    current: cancelled.current,
                };
                violation |= cancelled.into_error() != expected;
                if violation {
                    self.c.cancel_contract_violations += 1;
                }
                // The cancelled step may have promoted a Pod output (a runtime that cancels after
                // its work): the Pod's invocation count says so, and the mirror takes the write
                // so that one miscounted step does not offset it for the rest of the case.
                if work == 1 {
                    self.world
                        .with_mirror(|m| m.run_promotes(request, &world::round_payload(index)));
                }
                self.c.cancelled_step_work_units += work;
                self.account_units(work);
                if let Some(injected) = self.injected_at_step {
                    let latency = self.steps_started - injected;
                    self.c.cancel_latency_rounds_max =
                        self.c.cancel_latency_rounds_max.max(latency);
                    *self.c.latency_histogram.entry(latency).or_insert(0) += 1;
                }
                self.check_drift();
                StepEnd::Cancelled
            }
        }
    }

    /// How many `RunCancelled` events the runtime has emitted.
    fn cancel_events(&self) -> usize {
        self.world
            .runtime
            .events()
            .iter()
            .filter(|envelope| format!("{:?}", envelope.event).starts_with("RunCancelled"))
            .count()
    }

    // ---- the boundary -------------------------------------------------------------------

    fn note_checkpoint(
        &mut self,
        pass: &mut Pass,
        checkpoint: Checkpoint,
        action: &ActionIr,
        staleness: Staleness,
        admitted: bool,
        reason: Option<Reason>,
    ) {
        self.c
            .effect_kinds
            .insert(params::effect_name(action.effect));
        match staleness {
            Staleness::Fresh => {
                if !admitted {
                    self.c.fresh_actions_refused += 1;
                }
            }
            stale => {
                match stale {
                    Staleness::Revision => self.stale_revision_seen = true,
                    _ => self.stale_generation_seen = true,
                }
                if !pass.stale_seen {
                    pass.stale_seen = true;
                    self.c.stale_boundary_attempts[self.arm] += 1;
                }
                if admitted {
                    // An execution is counted from the executor's own log; an admission by
                    // an earlier checkpoint is the boundary saying yes.
                    if !matches!(checkpoint, Checkpoint::Execute) {
                        self.c.stale_checkpoint_admissions += 1;
                    }
                } else if !pass.refusal_counted {
                    pass.refusal_counted = true;
                    self.c.boundary_refusals[self.arm] += 1;
                    match reason {
                        Some(Reason::Revision) => self.c.revision_refusals += 1,
                        Some(Reason::Generation) => {
                            self.c.generation_boundary_refusals += 1;
                        }
                        Some(Reason::Permit) => self.c.stale_permit_refusals += 1,
                        _ => {}
                    }
                    let slot = match checkpoint {
                        Checkpoint::Authorize => 0,
                        Checkpoint::Prepare => 1,
                        Checkpoint::Execute => 2,
                    };
                    self.c.late_stale_refusals_by_checkpoint[slot] += 1;
                }
            }
        }
        pass.record.push((
            checkpoint.name(),
            if admitted {
                "ok"
            } else {
                reason.map_or("refused-other", Reason::label)
            },
        ));
    }

    /// The boundary pass of `action`: `authorize_action`, `prepare_execution_once`,
    /// `execute_prepared`, in that order, as a host that does not stop at the first refusal
    /// would drive them (so every layer is asked about every stale action). `late` is a writer
    /// that acts between the checkpoints.
    fn boundary_phase(
        &mut self,
        action: &ActionIr,
        late: Option<Injection>,
        key: &str,
    ) -> Result<Pass, String> {
        let mut pass = Pass::default();
        let base = self.tick;
        let step = params::STEP_TICKS;
        let mut queue: EventQueue<PassEvent> = EventQueue::new();
        let schedule = |queue: &mut EventQueue<PassEvent>, tick: u64, event: PassEvent| {
            queue
                .schedule(tick, event)
                .map(|_| ())
                .map_err(|late| format!("scheduling a checkpoint: {late:?}"))
        };
        schedule(&mut queue, base + step, PassEvent::Authorize)?;
        schedule(&mut queue, base + 2 * step, PassEvent::Prepare)?;
        schedule(&mut queue, base + 3 * step, PassEvent::Execute)?;
        if let Some(injection) = late {
            let tick = match injection.position {
                Position::BetweenActionAndAuthorize => base + step / 2,
                Position::BetweenPrepareAndExecute => base + 2 * step + step / 2,
                other => return Err(format!("{other:?} is not a boundary position")),
            };
            schedule(&mut queue, tick, PassEvent::Inject)?;
        }
        let mut permit: Option<ExecutionPermit> = None;
        while let Some(fired) = queue.pop() {
            self.tick = fired.tick;
            match fired.event {
                PassEvent::Inject => {
                    let injection = late.ok_or("an injection event without a writer")?;
                    self.inject(injection)?;
                }
                PassEvent::Authorize => {
                    let staleness = self.judge(action);
                    let result = guard(self.c, || self.world.runtime.authorize_action(action));
                    let Some(result) = result else {
                        self.aborted = true;
                        return Ok(pass);
                    };
                    let reason = result.as_ref().err().map(|e| Reason::of(&format!("{e:?}")));
                    self.note_checkpoint(
                        &mut pass,
                        Checkpoint::Authorize,
                        action,
                        staleness,
                        result.is_ok(),
                        reason,
                    );
                    if let Some(reason) = reason {
                        pass.refused.get_or_insert(reason);
                    }
                }
                PassEvent::Prepare => {
                    let staleness = self.judge(action);
                    let result = guard(self.c, || {
                        self.world.runtime.prepare_execution_once(
                            &self.world.session,
                            &self.world.project,
                            action,
                            world::TTL,
                            key,
                        )
                    });
                    let Some(result) = result else {
                        self.aborted = true;
                        return Ok(pass);
                    };
                    let reason = result.as_ref().err().map(|e| Reason::of(&format!("{e:?}")));
                    let admitted = result.is_ok();
                    permit = result.ok();
                    self.note_checkpoint(
                        &mut pass,
                        Checkpoint::Prepare,
                        action,
                        staleness,
                        admitted,
                        reason,
                    );
                    if let Some(reason) = reason {
                        pass.refused.get_or_insert(reason);
                    }
                }
                PassEvent::Execute => {
                    let Some(permit) = permit.take() else {
                        pass.record.push(("execute", "skipped"));
                        continue;
                    };
                    let staleness = self.judge(action);
                    let calls_before = self.world.with_mirror(|m| m.calls.len());
                    let result = guard(self.c, || {
                        self.world
                            .runtime
                            .execute_prepared(&self.world.session, permit)
                    });
                    let Some(result) = result else {
                        self.aborted = true;
                        return Ok(pass);
                    };
                    // The executor ran once per new call; it ran on a stale action iff the
                    // ledger said the action was stale just before.
                    let ran = self.world.with_mirror(|m| m.calls.len()) - calls_before;
                    if staleness != Staleness::Fresh {
                        self.c.stale_executed[self.arm] += ran as i64;
                    }
                    let reason = result.as_ref().err().map(|e| Reason::of(&format!("{e:?}")));
                    pass.executed = result.is_ok();
                    self.note_checkpoint(
                        &mut pass,
                        Checkpoint::Execute,
                        action,
                        staleness,
                        result.is_ok(),
                        reason,
                    );
                    if let Some(reason) = reason {
                        pass.refused.get_or_insert(reason);
                    }
                }
            }
        }
        Ok(pass)
    }

    /// The host offers the same action again under the same key after it executed: the answer
    /// must come from history.
    fn retry(&mut self, action: &ActionIr, key: &str) {
        let calls_before = self.world.with_mirror(|m| m.calls.len());
        let prepared = guard(self.c, || {
            self.world.runtime.prepare_execution_once(
                &self.world.session,
                &self.world.project,
                action,
                world::TTL,
                key,
            )
        });
        let Some(prepared) = prepared else {
            self.aborted = true;
            return;
        };
        let Ok(permit) = prepared else {
            self.c.retries_refused += 1;
            return;
        };
        let executed = guard(self.c, || {
            self.world
                .runtime
                .execute_prepared(&self.world.session, permit)
        });
        match executed {
            None => self.aborted = true,
            Some(Ok(_)) => {
                let calls_after = self.world.with_mirror(|m| m.calls.len());
                if calls_after == calls_before {
                    self.c.retries_replayed += 1;
                }
            }
            Some(Err(_)) => self.c.retries_refused += 1,
        }
    }

    /// What is checked once the case has run in this arm.
    fn finish(&mut self) {
        let arm = self.arm;
        let calls = self.world.with_mirror(|m| m.calls.len() as i64);
        self.c.executions += calls;
        self.c.double_executions += (calls - 1).max(0);
        let scan = oracle::scan(self.world.runtime.committed_events());
        self.c.stale_effect_attempts_committed += scan.stale_attempts;
        self.c.executor_vs_ledger_mismatches += (calls - scan.attempts).abs();
        // The schedule's reading of the case against what the mirror saw.
        if self.plan.stale_by_schedule() != self.stale_revision_seen
            || self.plan.generation_dead_by_schedule() != self.stale_generation_seen
        {
            self.c.mirror_drift += 1;
        }
        if arm == A0 {
            if self.stale_revision_seen || self.stale_generation_seen {
                self.c.stale_runs_observed += 1;
            }
            let dead = self.world.runtime.live_generation(world::TARGET)
                != Some(ptr_types::Generation(1))
                || self.world.runtime.revoked_generations().next().is_some();
            if dead {
                self.c.generation_stale_cases += 1;
            }
        }
    }
}

/// Executes `plan` in `arm` (0 = A0, 1 = A1) on a fresh runtime and adds what it saw to
/// `counters`.
pub fn run_arm(plan: &CasePlan, arm: usize, counters: &mut Counters) -> Result<ArmReport, String> {
    let mut ctx = Ctx::new(plan, arm, counters)?;
    let key = plan.key(ctx.arm_name());
    let early = plan.injection.filter(|i| i.position.is_early());
    let late = plan.injection.filter(|i| !i.position.is_early());
    let first = format!("case-{}", plan.index);
    let mut first_pass = None;
    let mut executed = false;
    let mut restart = false;
    match ctx.run_phase(&first, early)? {
        RunEnd::Failed => {}
        RunEnd::Cancelled => restart = true,
        RunEnd::Finished(action) => {
            let pass = ctx.boundary_phase(&action, late, &key)?;
            executed = pass.executed;
            // A host does not begin again on a generation that is gone.
            restart = matches!(pass.refused, Some(Reason::Revision | Reason::Permit))
                && !ctx.world.with_mirror(|m| m.generation_is_dead());
            first_pass = Some(PassRecord {
                action: format!("{action:?}"),
                checkpoints: pass.record.clone(),
            });
            if executed {
                ctx.retry(&action, &key);
            }
        }
    }
    if plan.fresh_by_schedule() {
        if executed {
            ctx.c.fresh_runs_completed += 1;
        } else if !ctx.aborted {
            ctx.c.fresh_runs_starved += 1;
        }
    }
    if restart && !ctx.aborted {
        ctx.c.restarted_runs += 1;
        let second = format!("case-{}-r", plan.index);
        let mut restarted = false;
        if let RunEnd::Finished(action) = ctx.run_phase(&second, None)? {
            let pass = ctx.boundary_phase(&action, None, &key)?;
            restarted = pass.executed;
            if restarted {
                ctx.retry(&action, &key);
            }
        }
        if restarted {
            ctx.c.fresh_runs_completed += 1;
        } else if !ctx.aborted {
            ctx.c.fresh_runs_starved += 1;
        }
    }
    let aborted = ctx.aborted;
    if !aborted {
        ctx.finish();
    }
    Ok(ArmReport {
        first_pass,
        aborted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::s002::plan::Class;
    use ptr_types::Effect;

    fn plan(
        class: Class,
        injection: Option<(Writer, Position)>,
        rounds: u64,
        effect: Effect,
    ) -> CasePlan {
        CasePlan {
            index: 7,
            seed: 17,
            rounds,
            effect,
            class,
            injection: injection.map(|(writer, position)| Injection { writer, position }),
        }
    }

    fn run(plan: &CasePlan, arm: usize) -> (Counters, ArmReport) {
        let mut counters = Counters::default();
        let report = run_arm(plan, arm, &mut counters).expect("the case runs");
        (counters, report)
    }

    fn hard(c: &Counters) -> i64 {
        c.stale_actions_executed()
            + c.stale_effect_attempts_committed
            + c.stale_checkpoint_admissions
            + c.fresh_actions_refused
            + c.false_cancellations
            + c.missed_cancellations
            + c.fresh_runs_starved
            + c.cancel_contract_violations
            + c.boundary_arm_divergences
            + c.double_executions
            + c.executor_vs_ledger_mismatches
            + c.mirror_drift
            + c.product_panics
    }

    #[test]
    fn a_case_without_a_writer_executes_once_in_both_arms_and_a_retry_is_answered_from_history() {
        let case = plan(Class::Never, None, 3, Effect::Irreversible);
        for arm in [A0, A1] {
            let (c, report) = run(&case, arm);
            assert_eq!(hard(&c), 0, "arm {arm}: {c:?}");
            assert_eq!(c.cancelled_runs, 0);
            assert_eq!(c.executions, 1);
            assert_eq!(c.retries_replayed, 1);
            assert_eq!(c.fresh_runs_completed, 1);
            assert_eq!(c.restarted_runs, 0);
            assert_eq!(c.boundary_refusals[arm], 0);
            assert_eq!(
                c.total_units[arm], 4,
                "three Pod rounds and the action step"
            );
            assert_eq!(c.wasted_units[arm], 0);
            let pass = report.first_pass.expect("the run reached the boundary");
            assert_eq!(
                pass.checkpoints,
                vec![("authorize", "ok"), ("prepare", "ok"), ("execute", "ok")]
            );
        }
    }

    #[test]
    fn a_write_before_round_one_cancels_arm_a1_at_the_next_step_and_lets_arm_a0_run_on() {
        let case = plan(
            Class::BumpEarly,
            Some((Writer::Bump, Position::BeforeRound1)),
            3,
            Effect::Mutation,
        );
        let (c1, r1) = run(&case, A1);
        assert_eq!(hard(&c1), 0, "{c1:?}");
        assert_eq!(c1.cancelled_runs, 1);
        assert_eq!(c1.cancel_latency_rounds_max, 1);
        assert_eq!(c1.wasted_units[A1], 0);
        assert_eq!(c1.restarted_runs, 1);
        assert_eq!(c1.fresh_runs_completed, 1);
        assert!(
            r1.first_pass.is_none(),
            "a cancelled run reaches no boundary"
        );
        assert_eq!(c1.executions, 1);

        let (c0, r0) = run(&case, A0);
        assert_eq!(hard(&c0), 0, "{c0:?}");
        assert_eq!(c0.cancelled_runs, 0);
        assert_eq!(
            c0.wasted_units[A0], 4,
            "all three rounds and the action step follow the write"
        );
        assert_eq!(
            c0.restarted_runs, 0,
            "the action is stamped with the newer snapshot"
        );
        assert_eq!(c0.executions, 1);
        assert!(r0.first_pass.is_some());
    }

    #[test]
    fn a_write_just_before_the_action_hands_arm_a0_a_stale_action_that_the_boundary_refuses() {
        let case = plan(
            Class::BumpEarly,
            Some((Writer::Bump, Position::JustBeforeAction)),
            2,
            Effect::External,
        );
        let (c0, r0) = run(&case, A0);
        assert_eq!(hard(&c0), 0, "{c0:?}");
        assert_eq!(c0.stale_boundary_attempts[A0], 1);
        assert_eq!(c0.boundary_refusals[A0], 1);
        assert_eq!(c0.stale_executed[A0], 0);
        assert_eq!(c0.restarted_runs, 1, "refused as stale, begun again");
        assert_eq!(c0.executions, 1);
        assert_eq!(
            c0.wasted_units[A0], 1,
            "the step that handed out the stale action"
        );
        let checkpoints = r0.first_pass.expect("a boundary pass").checkpoints;
        assert_eq!(checkpoints[0], ("authorize", "refused-stale-revision"));
        assert_eq!(checkpoints[1], ("prepare", "refused-stale-revision"));
        assert_eq!(checkpoints[2], ("execute", "skipped"));

        let (c1, r1) = run(&case, A1);
        assert_eq!(hard(&c1), 0, "{c1:?}");
        assert_eq!(c1.cancelled_runs, 1);
        assert_eq!(
            c1.boundary_refusals[A1], 0,
            "cancelled before the action was handed out"
        );
        assert_eq!(c1.wasted_units[A1], 0);
        assert!(r1.first_pass.is_none());
    }

    #[test]
    fn a_write_after_the_run_is_refused_by_the_boundary_identically_in_both_arms() {
        for position in [
            Position::BetweenActionAndAuthorize,
            Position::BetweenPrepareAndExecute,
        ] {
            let case = plan(
                Class::BumpLate,
                Some((Writer::Bump, position)),
                2,
                Effect::Read,
            );
            let (c0, r0) = run(&case, A0);
            let (c1, r1) = run(&case, A1);
            assert_eq!(hard(&c0) + hard(&c1), 0, "{position:?}: {c0:?} {c1:?}");
            assert_eq!(r0.first_pass, r1.first_pass, "{position:?}");
            for c in [&c0, &c1] {
                assert_eq!(c.cancelled_runs, 0);
                assert_eq!(
                    c.stale_boundary_attempts[A0] + c.stale_boundary_attempts[A1],
                    1
                );
                assert_eq!(c.restarted_runs, 1);
                assert_eq!(c.executions, 1);
            }
        }
        // Between prepare and execute the permit is what is retired.
        let case = plan(
            Class::BumpLate,
            Some((Writer::Bump, Position::BetweenPrepareAndExecute)),
            2,
            Effect::Mutation,
        );
        let (_, report) = run(&case, A1);
        assert_eq!(
            report.first_pass.unwrap().checkpoints,
            vec![
                ("authorize", "ok"),
                ("prepare", "ok"),
                ("execute", "refused-stale-permit")
            ]
        );
    }

    #[test]
    fn a_write_that_moves_nothing_cancels_nothing_and_leaves_the_action_fresh() {
        for position in Position::ALL {
            let case = plan(
                Class::NoOp,
                Some((Writer::NoOp, position)),
                3,
                Effect::Irreversible,
            );
            for arm in [A0, A1] {
                let (c, _) = run(&case, arm);
                assert_eq!(hard(&c), 0, "{position:?} arm {arm}: {c:?}");
                assert_eq!(c.cancelled_runs, 0);
                assert_eq!(c.noop_writes_observed, 1);
                assert_eq!(c.executions, 1);
                assert_eq!(c.fresh_runs_completed, 1);
                assert_eq!(c.stale_boundary_attempts[arm], 0);
            }
        }
    }

    #[test]
    fn a_revoked_or_superseded_generation_cancels_nothing_and_the_boundary_alone_refuses() {
        for writer in [Writer::Revoke, Writer::Supersede] {
            for position in Position::ALL {
                let case = plan(
                    Class::Lifecycle,
                    Some((writer, position)),
                    2,
                    Effect::Mutation,
                );
                let mut reports = Vec::new();
                for arm in [A0, A1] {
                    let (c, report) = run(&case, arm);
                    assert_eq!(hard(&c), 0, "{writer:?} {position:?} arm {arm}: {c:?}");
                    assert_eq!(c.cancelled_runs, 0);
                    assert_eq!(c.executions, 0, "nothing executes on a dead generation");
                    assert_eq!(c.boundary_refusals[arm], 1);
                    assert_eq!(
                        c.restarted_runs, 0,
                        "a host does not retry a dead generation"
                    );
                    assert_eq!(c.generation_stale_cases, i64::from(arm == A0));
                    reports.push(report.first_pass);
                }
                assert_eq!(reports[0], reports[1]);
            }
        }
    }

    #[test]
    fn every_effect_and_every_position_holds_every_hard_counter_at_zero_in_both_arms() {
        let mut classes_seen = std::collections::BTreeSet::new();
        for index in 0..96 {
            let case = crate::experiments::s002::plan::plan_of(43, index);
            classes_seen.insert(case.class);
            let mut totals = Counters::default();
            let a0 = run_arm(&case, A0, &mut totals).unwrap();
            let a1 = run_arm(&case, A1, &mut totals).unwrap();
            assert!(!a0.aborted && !a1.aborted);
            assert_eq!(hard(&totals), 0, "case {index} {case:?}: {totals:?}");
        }
        assert_eq!(classes_seen.len(), 5);
    }

    #[test]
    fn a_panic_of_the_product_is_counted_and_gives_no_value() {
        let mut counters = Counters::default();
        let quiet = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let value: Option<u8> = guard(&mut counters, || panic!("planted"));
        std::panic::set_hook(quiet);
        assert_eq!(value, None);
        assert_eq!(counters.product_panics, 1);
        assert_eq!(guard(&mut counters, || 5), Some(5));
        assert_eq!(counters.product_panics, 1);
    }

    #[test]
    fn the_refusal_reasons_are_read_from_the_typed_errors() {
        assert_eq!(
            Reason::of("StaleRevision { action: Revision(1) }"),
            Reason::Revision
        );
        assert_eq!(
            Reason::of(
                "AuthorizationDenied(StaleRevision { action: Revision(1), current: Revision(2) })"
            ),
            Reason::Revision
        );
        assert_eq!(
            Reason::of("StaleGeneration { target: \"t\" }"),
            Reason::Generation
        );
        assert_eq!(
            Reason::of("UnknownGeneration { target: \"t\" }"),
            Reason::Generation
        );
        assert_eq!(Reason::of("StalePermit"), Reason::Permit);
        assert_eq!(Reason::of("PermissionDenied"), Reason::Other);
    }
}
