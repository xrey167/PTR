//! The discrete-event simulation: one producer stream, one single-server consumer, a queue
//! under test, all in virtual ticks.
//!
//! The arrivals and the service times are drawn from the seed before anything runs, so every
//! arm sees the same schedule whatever its queue admits. Events fire in `(tick, seq)` order
//! (see `vtime`): the arrivals are scheduled first, so an arrival and a service completion at
//! one tick fire arrival first. The consumer takes a message the moment it is idle and one is
//! queued; the wait of a message is the tick it starts service minus the tick it arrived.
//!
//! Every operation on the queue under test is mirrored on a [`RefModel`] (ids only) and the
//! two answers are compared at every event, while an ownership [`Ledger`] and the shared
//! [`DropLog`] audit the payloads themselves.

use super::ledger::{Ledger, Refusal, Report, Seen};
use super::params::{self, PhaseSpec};
use super::payload::{fnv, Corpus, DropLog, Tracked, FNV_START};
use super::queue::{Offer, Queue, RefModel, Take};
use crate::experiments::rng::Rng;
use crate::experiments::vtime::EventQueue;

#[derive(Clone, Copy, Debug)]
pub struct Arrival {
    pub id: u32,
    pub tick: u64,
    pub phase: usize,
}

/// The ids `first..end` of a phase (the tests read the bounds).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct PhaseRange {
    pub name: &'static str,
    pub first: u32,
    pub end: u32,
    pub overload: bool,
}

pub struct Schedule {
    pub arrivals: Vec<Arrival>,
    /// Service time in ticks, by id.
    pub service: Vec<u64>,
    pub ranges: Vec<PhaseRange>,
    /// Close the queue at the tick of this arrival, before the arrival fires.
    pub close_before: Option<u32>,
}

impl Schedule {
    /// The schedule of `phases` for `seed`; `stream` separates the schedules of one seed.
    pub fn generate(
        seed: u64,
        stream: u64,
        phases: &[PhaseSpec],
        close_before: Option<u32>,
    ) -> Self {
        let mut root = Rng::new(seed);
        let mut stream_root = root.fork(stream);
        let mut gaps = stream_root.fork(1);
        let mut services = stream_root.fork(2);
        let mut arrivals = Vec::new();
        let mut service = Vec::new();
        let mut ranges = Vec::new();
        let mut tick = 0u64;
        for (index, phase) in phases.iter().enumerate() {
            let first = arrivals.len() as u32;
            for _ in 0..phase.count {
                tick += gaps.range(phase.gap_lo, phase.gap_hi);
                arrivals.push(Arrival {
                    id: arrivals.len() as u32,
                    tick,
                    phase: index,
                });
                service.push(services.range(params::SERVICE_MIN, params::SERVICE_MAX));
            }
            ranges.push(PhaseRange {
                name: phase.name,
                first,
                end: arrivals.len() as u32,
                overload: phase.overload(),
            });
        }
        Self {
            arrivals,
            service,
            ranges,
            close_before,
        }
    }

    pub fn messages(&self) -> usize {
        self.arrivals.len()
    }
}

/// What a simulation is told about the queue it drives.
pub struct SimCfg<'a> {
    /// The capacity the queue is meant to have (also the reference model's).
    pub capacity: usize,
    /// The analytic wait bound in ticks: `capacity * SERVICE_MAX`.
    pub wait_bound: u64,
    /// The payload bytes allowed alive between events: `(capacity + 1) * body bytes`.
    pub bytes_bound: i64,
    pub corpus: &'a Corpus,
    /// The phase after which the queue should drain and accept again.
    pub recovery_phase: Option<usize>,
}

impl<'a> SimCfg<'a> {
    pub fn new(capacity: usize, body_bytes: usize, corpus: &'a Corpus) -> Self {
        Self {
            capacity,
            wait_bound: params::wait_bound(capacity),
            bytes_bound: params::payload_bound(capacity, body_bytes),
            corpus,
            recovery_phase: Some(params::RECOVERY_PHASE),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseStat {
    pub offered: u64,
    pub admitted: u64,
    pub refused_full: u64,
    pub refused_closed: u64,
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub report: Report,
    /// Events at which the queue held, or the ledger counted, more than the capacity.
    pub capacity_violations: u64,
    pub model_divergences: u64,
    /// Full answers where the reference model had room.
    pub spurious_refusals: u64,
    /// Answers where exactly one of the queue and the model said `Closed`.
    pub closed_path_mismatches: u64,
    /// Messages that waited longer than the analytic bound.
    pub wait_bound_violations: u64,
    /// Events after which more payload bytes were alive than the bound.
    pub payload_bound_violations: u64,
    /// The largest depth sampled after an offer (before the idle consumer takes) or after a take.
    pub peak_depth: usize,
    pub peak_bytes: i64,
    /// Panics of the product contained during the run.
    pub product_panics: u64,
    /// Offers refused `Closed` while the model's queue was full: the cases that tell
    /// Closed-over-Full from Full-over-Closed.
    pub closed_over_full: u64,
    /// Messages delivered after the close: the queue drains before it reports `Closed`.
    pub drained_after_close: u64,
    pub waits: Vec<u64>,
    pub phases: Vec<PhaseStat>,
    /// The queue was empty during the recovery phase and admitted arrivals afterwards.
    pub recovered: bool,
    pub admitted_after_recovery: u64,
    pub refused_after_recovery: u64,
    pub end_tick: u64,
    /// Fingerprint of the event-by-event trace.
    pub digest: u64,
}

impl Outcome {
    pub fn max_wait(&self) -> u64 {
        self.waits.iter().copied().max().unwrap_or(0)
    }

    /// The smallest wait that at least 99% of the waits do not exceed (nearest rank).
    pub fn p99_wait(&self) -> u64 {
        percentile_99(&self.waits)
    }
}

/// Nearest-rank 99th percentile of `values` (0 for none).
pub fn percentile_99(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * 99).div_ceil(100);
    sorted[rank.max(1) - 1]
}

#[derive(Debug)]
enum Event {
    Arrive(u32),
    Done,
    Close,
}

fn seen(payload: &Tracked) -> Seen {
    Seen {
        id: payload.id(),
        digest: payload.digest(),
    }
}

fn mix(state: u64, tick: u64, code: u8, id: u32, depth: usize) -> u64 {
    let state = fnv(state, &tick.to_le_bytes());
    let state = fnv(state, &[code]);
    let state = fnv(state, &id.to_le_bytes());
    fnv(state, &(depth as u32).to_le_bytes())
}

/// Drive `queue` through `schedule` and audit it. An `Err` is a fault of the harness itself.
pub fn run(
    cfg: &SimCfg<'_>,
    schedule: &Schedule,
    queue: &mut dyn Queue,
) -> Result<Outcome, String> {
    let ids = schedule.messages();
    if ids > cfg.corpus.len() {
        return Err(format!(
            "{ids} messages but a corpus of {}",
            cfg.corpus.len()
        ));
    }
    let log = DropLog::new(ids);
    let mut ledger = Ledger::new(&cfg.corpus.digests()[..ids]);
    let mut shadow = RefModel::<u32>::new(cfg.capacity);
    let mut events = EventQueue::new();
    let past = |error| format!("event scheduled in the past: {error:?}");
    for arrival in &schedule.arrivals {
        if schedule.close_before == Some(arrival.id) {
            events.schedule(arrival.tick, Event::Close).map_err(past)?;
        }
        events
            .schedule(arrival.tick, Event::Arrive(arrival.id))
            .map_err(past)?;
    }
    let mut out = Outcome {
        phases: vec![PhaseStat::default(); schedule.ranges.len()],
        ..Outcome::default()
    };
    let mut hash = FNV_START;
    let mut in_service: Option<Tracked> = None;
    let mut phase_now = 0usize;
    let mut drained_in_recovery = false;
    let mut closed_now = false;
    let mut shadow_peak = 0usize;

    while let Some(fired) = events.pop() {
        let tick = fired.tick;
        let (code, id) = match fired.event {
            Event::Close => {
                queue.close();
                shadow.close();
                closed_now = true;
                (3u8, u32::MAX)
            }
            Event::Arrive(id) => {
                let arrival = schedule.arrivals[id as usize];
                phase_now = arrival.phase;
                let full_before = shadow.depth() >= cfg.capacity;
                let modelled = shadow.offer(id);
                let payload = cfg.corpus.make(&log, id, 0);
                ledger.offer(id, tick);
                out.phases[phase_now].offered += 1;
                let code = match queue.offer(payload) {
                    Offer::Admitted => {
                        ledger.admitted(id);
                        out.phases[phase_now].admitted += 1;
                        if drained_in_recovery {
                            out.admitted_after_recovery += 1;
                        }
                        match modelled {
                            Offer::Admitted => {}
                            Offer::Full(_) => out.model_divergences += 1,
                            Offer::Closed(_) | Offer::Panicked => {
                                out.model_divergences += 1;
                                out.closed_path_mismatches += 1;
                            }
                        }
                        0u8
                    }
                    Offer::Full(returned) => {
                        ledger.refused(id, Refusal::Full, seen(&returned));
                        out.phases[phase_now].refused_full += 1;
                        if drained_in_recovery {
                            out.refused_after_recovery += 1;
                        }
                        match modelled {
                            Offer::Full(_) => {}
                            Offer::Admitted => {
                                out.spurious_refusals += 1;
                                out.model_divergences += 1;
                            }
                            Offer::Closed(_) | Offer::Panicked => {
                                out.model_divergences += 1;
                                out.closed_path_mismatches += 1;
                            }
                        }
                        drop(returned);
                        1u8
                    }
                    Offer::Closed(returned) => {
                        ledger.refused(id, Refusal::Closed, seen(&returned));
                        out.phases[phase_now].refused_closed += 1;
                        out.closed_over_full += u64::from(full_before);
                        if !matches!(modelled, Offer::Closed(_)) {
                            out.model_divergences += 1;
                            out.closed_path_mismatches += 1;
                        }
                        drop(returned);
                        2u8
                    }
                    Offer::Panicked => {
                        // The panic is counted by the queue (`product_panics`). The value went
                        // with the unwinding; the ledger records the id as refused `Full`, the
                        // answer a stand-in would have given, so the run goes on.
                        let digest = cfg.corpus.digests()[id as usize];
                        ledger.refused(id, Refusal::Full, Seen { id, digest });
                        out.phases[phase_now].refused_full += 1;
                        if !matches!(modelled, Offer::Full(_)) {
                            out.model_divergences += 1;
                        }
                        5u8
                    }
                };
                (code, id)
            }
            Event::Done => {
                let id = match in_service.take() {
                    Some(done) => {
                        let id = done.id();
                        ledger.finished(id);
                        drop(done);
                        id
                    }
                    None => return Err("a service completion with nobody in service".into()),
                };
                (4u8, id)
            }
        };
        for dropped in log.take_recent() {
            ledger.dropped(dropped);
        }

        // Sample once between the event and the idle consumer's take, so a transient over
        // capacity inside an offer is not hidden by the take that follows it.
        let mid = queue.depth();
        out.peak_depth = out.peak_depth.max(mid);
        shadow_peak = shadow_peak.max(shadow.depth());
        if mid > cfg.capacity {
            out.capacity_violations += 1;
        }
        if mid != shadow.depth() {
            out.model_divergences += 1;
        }

        // The consumer takes the next message whenever it is idle.
        if in_service.is_none() {
            let modelled = shadow.take();
            match queue.take() {
                Take::Message(payload) => {
                    out.drained_after_close += u64::from(closed_now);
                    match modelled {
                        Take::Message(expected) if expected == payload.id() => {}
                        Take::Message(_) | Take::Empty => out.model_divergences += 1,
                        Take::Closed | Take::Panicked => {
                            out.model_divergences += 1;
                            out.closed_path_mismatches += 1;
                        }
                    }
                    let taken = payload.id();
                    if let Some(wait) = ledger.started(seen(&payload), tick) {
                        out.waits.push(wait);
                        if wait > cfg.wait_bound {
                            out.wait_bound_violations += 1;
                        }
                    }
                    let service = schedule
                        .service
                        .get(taken as usize)
                        .copied()
                        .unwrap_or(params::SERVICE_MAX);
                    events.schedule_after(service, Event::Done);
                    in_service = Some(payload);
                }
                Take::Empty | Take::Panicked => match modelled {
                    Take::Empty => {}
                    Take::Message(_) => out.model_divergences += 1,
                    Take::Closed | Take::Panicked => {
                        out.model_divergences += 1;
                        out.closed_path_mismatches += 1;
                    }
                },
                Take::Closed => match modelled {
                    Take::Closed => {}
                    Take::Message(_) | Take::Empty | Take::Panicked => {
                        out.model_divergences += 1;
                        out.closed_path_mismatches += 1;
                    }
                },
            }
            for dropped in log.take_recent() {
                ledger.dropped(dropped);
            }
        }

        // After every event: the capacity, the model, the live payloads, the bytes.
        let depth = queue.depth();
        out.peak_depth = out.peak_depth.max(depth);
        shadow_peak = shadow_peak.max(shadow.depth());
        if depth > cfg.capacity || ledger.outstanding() > cfg.capacity as u64 {
            out.capacity_violations += 1;
        }
        if depth != shadow.depth() {
            out.model_divergences += 1;
        }
        ledger.observe_live(depth as i64 + i64::from(in_service.is_some()), log.live());
        let bytes = log.live_bytes();
        out.peak_bytes = out.peak_bytes.max(bytes);
        if bytes > cfg.bytes_bound {
            out.payload_bound_violations += 1;
        }
        if cfg.recovery_phase == Some(phase_now) && depth == 0 {
            drained_in_recovery = true;
        }
        hash = mix(hash, tick, code, id, depth);
    }

    // The queue's own record of its peak must be the peak the model saw.
    if let Some(peak) = queue.peak_depth() {
        if peak != shadow_peak {
            out.model_divergences += 1;
        }
    }
    out.product_panics = queue.panics();
    out.end_tick = events.now();
    out.recovered = drained_in_recovery && out.admitted_after_recovery >= 1;
    out.digest = hash;
    drop(in_service);
    out.report = ledger.finish(&log);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::queue::{
        CloseVia, Fault, Faulty, MailboxQueue, ReferenceQueue, UnboundedQueue,
    };
    use super::*;

    fn fixture(phases: &[PhaseSpec], close_before: Option<u32>) -> (Schedule, Corpus) {
        let schedule = Schedule::generate(17, 0, phases, close_before);
        let corpus = Corpus::generate(17, schedule.messages(), 64);
        (schedule, corpus)
    }

    fn cfg<'a>(capacity: usize, corpus: &'a Corpus) -> SimCfg<'a> {
        SimCfg::new(capacity, 64, corpus)
    }

    #[test]
    fn a_schedule_is_a_function_of_the_seed_with_ordered_ticks_and_bounded_gaps() {
        let (a, _) = fixture(&params::PHASES, None);
        let (b, _) = fixture(&params::PHASES, None);
        assert_eq!(a.messages(), 2300);
        assert_eq!(a.service, b.service);
        assert!(a
            .arrivals
            .iter()
            .zip(&b.arrivals)
            .all(|(x, y)| x.tick == y.tick));
        let other = Schedule::generate(18, 0, &params::PHASES, None);
        assert_ne!(a.service, other.service);
        let other_stream = Schedule::generate(17, 1, &params::PHASES, None);
        assert_ne!(a.service, other_stream.service);
        for pair in a.arrivals.windows(2) {
            assert!(pair[1].tick > pair[0].tick);
        }
        assert!(a
            .service
            .iter()
            .all(|s| (params::SERVICE_MIN..=params::SERVICE_MAX).contains(s)));
        for (range, spec) in a.ranges.iter().zip(params::PHASES) {
            assert_eq!(range.end - range.first, spec.count);
            for arrival in &a.arrivals[range.first as usize..range.end as usize] {
                assert_eq!(a.ranges[arrival.phase].name, spec.name);
            }
        }
        // Inside the underload phase every gap exceeds the longest service.
        let under = a.ranges[params::UNDERLOAD_PHASE];
        for pair in a.arrivals[under.first as usize..under.end as usize].windows(2) {
            assert!(pair[1].tick - pair[0].tick > params::SERVICE_MAX);
        }
    }

    #[test]
    fn percentile_is_nearest_rank() {
        assert_eq!(percentile_99(&[]), 0);
        assert_eq!(percentile_99(&[5]), 5);
        let hundred: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_99(&hundred), 99);
        let two_hundred: Vec<u64> = (1..=200).collect();
        assert_eq!(percentile_99(&two_hundred), 198);
        assert_eq!(percentile_99(&[9, 1, 5]), 9);
    }

    #[test]
    fn underload_never_waits_never_refuses_and_delivers_everything() {
        let (schedule, corpus) = fixture(&params::SANITY_PHASES, None);
        for capacity in params::CAPACITIES {
            let mut queue = MailboxQueue::new(capacity, CloseVia::Owner);
            let out = run(&cfg(capacity, &corpus), &schedule, &mut queue).unwrap();
            assert_eq!(out.report.refused_full, 0);
            assert_eq!(out.report.delivered, 200);
            assert_eq!(out.max_wait(), 0);
            assert!(out.peak_depth <= 1);
            assert_eq!(out.model_divergences, 0);
            assert_eq!(out.report.ownership_violations(), 0);
        }
    }

    #[test]
    fn the_mailbox_leaves_the_reference_trace_identical_at_every_event() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        for capacity in params::CAPACITIES {
            let mut real = MailboxQueue::new(capacity, CloseVia::Owner);
            let mut reference = ReferenceQueue::new(capacity);
            let a = run(&cfg(capacity, &corpus), &schedule, &mut real).unwrap();
            let b = run(&cfg(capacity, &corpus), &schedule, &mut reference).unwrap();
            assert_eq!(a.digest, b.digest, "capacity {capacity}");
            assert_eq!(a.model_divergences, 0);
            assert_eq!(b.model_divergences, 0);
            assert_eq!(a.phases, b.phases);
            assert_eq!(a.waits, b.waits);
            assert_eq!(a.capacity_violations, 0);
            assert_eq!(a.spurious_refusals, 0);
            assert_eq!(a.wait_bound_violations, 0);
            assert_eq!(a.payload_bound_violations, 0);
            assert_eq!(a.report.ownership_violations(), 0);
            assert_eq!(a.report.silent_drops, 0);
            assert_eq!(a.report.fifo_violations, 0);
            assert!(a.peak_depth <= capacity);
            assert!(a.max_wait() <= capacity as u64 * params::SERVICE_MAX);
            assert!(
                a.report.refused_full > 0,
                "overload refuses at capacity {capacity}"
            );
            assert_eq!(a.report.delivered + a.report.refused_full, a.report.offered);
            assert!(a.recovered, "recovery at capacity {capacity}");
        }
    }

    #[test]
    fn the_same_schedule_twice_gives_the_same_digest() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        let first = run(
            &cfg(8, &corpus),
            &schedule,
            &mut MailboxQueue::new(8, CloseVia::Owner),
        )
        .unwrap();
        let second = run(
            &cfg(8, &corpus),
            &schedule,
            &mut MailboxQueue::new(8, CloseVia::Owner),
        )
        .unwrap();
        assert_eq!(first.digest, second.digest);
        assert_ne!(first.digest, 0);
    }

    #[test]
    fn the_unbounded_baseline_exceeds_the_capacity_and_the_wait_bound_through_the_same_oracle() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        let mut queue = UnboundedQueue::new();
        let out = run(&cfg(8, &corpus), &schedule, &mut queue).unwrap();
        assert!(out.peak_depth > 8);
        assert!(out.capacity_violations > 0);
        assert!(out.wait_bound_violations > 0);
        assert!(out.payload_bound_violations > 0);
        assert!(
            out.model_divergences > 0,
            "it admits where the model refuses"
        );
        // It still keeps ownership: nothing lost, nothing duplicated.
        assert_eq!(out.report.ownership_violations(), 0);
        assert_eq!(out.report.silent_drops, 0);
        assert_eq!(out.report.refused_full, 0);
        assert_eq!(out.report.delivered, 2300);
    }

    fn faulty(fault: Fault, capacity: usize, schedule: &Schedule, corpus: &Corpus) -> Outcome {
        let mut queue = Faulty::new(
            MailboxQueue::new(capacity, CloseVia::Owner),
            fault,
            capacity,
        );
        run(&cfg(capacity, corpus), schedule, &mut queue).unwrap()
    }

    #[test]
    fn drop_oldest_is_a_silent_drop_and_an_ownership_violation() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        let out = faulty(Fault::DropOldest, 8, &schedule, &corpus);
        assert!(out.report.silent_drops > 0);
        assert!(out.report.dropped_while_owned > 0);
        assert!(out.report.ownership_violations() > 0);
    }

    #[test]
    fn duplicates_mutations_and_losses_are_each_seen_by_their_own_cause() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        let dup = faulty(Fault::DuplicateDelivery, 8, &schedule, &corpus);
        assert!(dup.report.duplicate_deliveries > 0);
        let mutated = faulty(Fault::MutatePayload, 8, &schedule, &corpus);
        assert!(mutated.report.payload_mutations > 0);
        let lossy = faulty(Fault::Lossy, 8, &schedule, &corpus);
        assert!(lossy.report.silent_drops > 0 || lossy.report.dropped_while_owned > 0);
        let early = faulty(Fault::RefuseEarly, 8, &schedule, &corpus);
        assert!(early.spurious_refusals > 0);
    }

    #[test]
    fn a_lifo_queue_breaks_fifo_and_the_model() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        let out = run(&cfg(8, &corpus), &schedule, &mut ReferenceQueue::lifo(8)).unwrap();
        assert!(out.report.fifo_violations > 0);
        assert!(out.model_divergences > 0);
    }

    #[test]
    fn a_larger_queue_breaks_the_capacity_and_a_doubled_one_the_wait_bound() {
        let (schedule, corpus) = fixture(&params::PHASES, None);
        let plus_one = run(
            &cfg(8, &corpus),
            &schedule,
            &mut MailboxQueue::new(9, CloseVia::Owner),
        )
        .unwrap();
        assert!(plus_one.capacity_violations > 0);
        assert!(plus_one.model_divergences > 0);
        assert!(plus_one.payload_bound_violations > 0);
        let doubled = run(
            &cfg(8, &corpus),
            &schedule,
            &mut MailboxQueue::new(16, CloseVia::Owner),
        )
        .unwrap();
        assert!(doubled.capacity_violations > 0);
        assert!(doubled.wait_bound_violations > 0);
    }

    #[test]
    fn the_closed_path_returns_every_late_value_and_drains_the_queue() {
        let (schedule, corpus) = fixture(&params::CLOSED_PHASES, Some(params::CLOSE_BEFORE_ID));
        for (capacity, via) in [
            (1, CloseVia::Owner),
            (8, CloseVia::Sender),
            (128, CloseVia::Owner),
        ] {
            let mut cfg = cfg(capacity, &corpus);
            cfg.recovery_phase = None;
            let mut queue = MailboxQueue::new(capacity, via);
            let out = run(&cfg, &schedule, &mut queue).unwrap();
            let after = schedule.messages() as u64 - u64::from(params::CLOSE_BEFORE_ID);
            assert_eq!(out.report.refused_closed, after, "capacity {capacity}");
            assert_eq!(out.closed_path_mismatches, 0);
            assert_eq!(out.model_divergences, 0);
            assert_eq!(out.report.ownership_violations(), 0);
            assert_eq!(out.report.silent_drops, 0);
            assert_eq!(
                out.report.delivered + out.report.refused_full + out.report.refused_closed,
                out.report.offered
            );
        }
    }

    #[test]
    fn closing_beats_full_on_a_full_queue() {
        let (schedule, corpus) = fixture(&params::CLOSED_PHASES, Some(params::CLOSE_BEFORE_ID));
        let mut cfg = cfg(1, &corpus);
        cfg.recovery_phase = None;
        let out = run(&cfg, &schedule, &mut MailboxQueue::new(1, CloseVia::Owner)).unwrap();
        // The queue was full when it closed, yet every later offer is Closed, none Full.
        let after: u64 = out.phases[1].refused_closed;
        assert!(after > 0);
        assert!(
            out.report.refused_full > 0,
            "before the close it refused Full"
        );
    }

    #[test]
    fn broken_close_handling_is_a_closed_path_mismatch() {
        let (schedule, corpus) = fixture(&params::CLOSED_PHASES, Some(params::CLOSE_BEFORE_ID));
        for fault in [Fault::IgnoreClose, Fault::ClosedAsEmpty] {
            let mut cfg = cfg(8, &corpus);
            cfg.recovery_phase = None;
            let mut queue = Faulty::new(MailboxQueue::new(8, CloseVia::Owner), fault, 8);
            let out = run(&cfg, &schedule, &mut queue).unwrap();
            assert!(out.closed_path_mismatches > 0, "{fault:?}");
        }
    }

    #[test]
    fn a_corpus_shorter_than_the_schedule_is_a_harness_error() {
        let (schedule, _) = fixture(&params::SANITY_PHASES, None);
        let corpus = Corpus::generate(1, 5, 8);
        let mut queue = ReferenceQueue::new(1);
        assert!(run(&cfg(1, &corpus), &schedule, &mut queue).is_err());
    }
}
