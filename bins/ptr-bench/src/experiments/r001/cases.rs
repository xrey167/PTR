//! The cases C1 to C9 of R001, run once per seed, and the arithmetic that turns their
//! outcomes into the metrics.
//!
//! - C1 underload, C2 overload x2, C3 overload x8, C4 burst and recovery: the phases of one
//!   schedule, read per cell from arm B.
//! - C5 capacity sweep and C6 payload classes: the cells (3 capacities x 2 payload classes),
//!   each run through arms B (the real mailbox), U (an unbounded channel), R (the reference
//!   queue) and S (a bounded `sync_channel`, the independent second oracle), plus a closed-path
//!   run of B and of S.
//! - C7 capacity zero and the closed path, as units and as a scheduled close.
//! - C8 the threaded arm.
//! - C9 the negative controls and the oracle's canaries.

use super::guard::contained;
use super::ledger::{Ledger, Refusal, Report, Seen};
use super::params::{self, PayloadClass};
use super::payload::{Corpus, DropLog};
use super::queue::{
    CloseVia, Fault, Faulty, MailboxQueue, Queue, ReferenceQueue, SyncChannelQueue, UnboundedQueue,
};
use super::sim::{self, Outcome, Schedule, SimCfg};
use ptr_exec::{Mailbox, Receive, SendError};

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub capacity: usize,
    pub class: PayloadClass,
}

pub fn cells() -> Vec<Cell> {
    let mut all = Vec::new();
    for capacity in params::CAPACITIES {
        for class in params::PAYLOAD_CLASSES {
            all.push(Cell { capacity, class });
        }
    }
    all
}

/// How a cell closes its mailbox in the closed-path run: the large class through a sender, so
/// both routes are exercised.
fn close_via(cell: &Cell) -> CloseVia {
    if cell.class.name == "small" {
        CloseVia::Owner
    } else {
        CloseVia::Sender
    }
}

pub struct CellRun {
    pub cell: Cell,
    pub b: Outcome,
    pub u: Outcome,
    pub r: Outcome,
    /// The bounded `sync_channel`, the independent second oracle.
    pub s: Outcome,
    pub closed: Outcome,
    /// The second oracle on the closed-path schedule.
    pub closed_s: Outcome,
}

/// One deterministic pass over every cell.
pub struct Pass {
    pub runs: Vec<CellRun>,
    /// The fingerprint of every simulation, in a fixed order.
    pub digests: Vec<u64>,
    pub main: Schedule,
    pub closed: Schedule,
    pub corpora: Vec<Corpus>,
}

/// The schedules and corpora of one seed. Generated afresh by every pass, so that a
/// generator that is not a function of the seed shows up as a determinism mismatch.
fn inputs(seed: u64) -> (Schedule, Schedule, Schedule, Vec<Corpus>) {
    let main = Schedule::generate(seed, 0, &params::PHASES, None);
    let closed = Schedule::generate(
        seed,
        1,
        &params::CLOSED_PHASES,
        Some(params::CLOSE_BEFORE_ID),
    );
    let sanity = Schedule::generate(seed, 2, &params::SANITY_PHASES, None);
    let corpora = params::PAYLOAD_CLASSES
        .iter()
        .map(|class| Corpus::generate(seed, main.messages(), class.bytes))
        .collect();
    (main, closed, sanity, corpora)
}

fn corpus_of<'a>(corpora: &'a [Corpus], class: &PayloadClass) -> &'a Corpus {
    let at = params::PAYLOAD_CLASSES
        .iter()
        .position(|candidate| candidate == class)
        .expect("a declared payload class");
    &corpora[at]
}

/// Differences between the real mailbox's run and the reference queue's run of one schedule.
/// Both answered the same offers; any difference is a divergence of the real one.
pub fn against_reference(b: &Outcome, r: &Outcome) -> u64 {
    let mut differing = 0;
    differing += u64::from(b.digest != r.digest);
    differing += u64::from(b.waits != r.waits);
    differing += u64::from(b.phases != r.phases);
    differing += u64::from(b.peak_depth != r.peak_depth);
    differing += u64::from(b.report.admitted != r.report.admitted);
    differing += u64::from(b.report.refused_full != r.report.refused_full);
    differing += u64::from(b.report.delivered != r.report.delivered);
    differing += u64::from(b.end_tick != r.end_tick);
    differing += u64::from(b.report.refused_closed != r.report.refused_closed);
    differing += u64::from(b.drained_after_close != r.drained_after_close);
    differing += u64::from(b.closed_over_full != r.closed_over_full);
    differing
}

/// Run every cell once for `seed`. An `Err` is a fault of the harness or an oracle that is
/// inconsistent with the reference queue in underload.
pub fn pass(seed: u64) -> Result<Pass, String> {
    let (main, closed, sanity, corpora) = inputs(seed);
    let mut runs = Vec::new();
    let mut digests = Vec::new();
    for cell in cells() {
        let corpus = corpus_of(&corpora, &cell.class);
        let cfg = SimCfg::new(cell.capacity, cell.class.bytes, corpus);
        let b = sim::run(
            &cfg,
            &main,
            &mut MailboxQueue::new(cell.capacity, close_via(&cell)),
        )?;
        let u = sim::run(&cfg, &main, &mut UnboundedQueue::new())?;
        let r = sim::run(&cfg, &main, &mut ReferenceQueue::new(cell.capacity))?;
        let s = sim::run(&cfg, &main, &mut SyncChannelQueue::new(cell.capacity))?;
        let mut closed_cfg = SimCfg::new(cell.capacity, cell.class.bytes, corpus);
        closed_cfg.recovery_phase = None;
        let c = sim::run(
            &closed_cfg,
            &closed,
            &mut MailboxQueue::new(cell.capacity, close_via(&cell)),
        )?;
        let cs = sim::run(
            &closed_cfg,
            &closed,
            &mut SyncChannelQueue::new(cell.capacity),
        )?;
        digests.extend([b.digest, u.digest, r.digest, s.digest, c.digest, cs.digest]);
        runs.push(CellRun {
            cell,
            b,
            u,
            r,
            s,
            closed: c,
            closed_s: cs,
        });
    }
    // The reference queue alone, in underload, must be clean: the oracle and the reference
    // queue agree about a regime in which nothing can go wrong.
    for capacity in params::CAPACITIES {
        let corpus = &corpora[0];
        let cfg = SimCfg::new(capacity, params::PAYLOAD_CLASSES[0].bytes, corpus);
        let alone = sim::run(&cfg, &sanity, &mut ReferenceQueue::new(capacity))?;
        let sane = alone.report.refused_full == 0
            && alone.report.delivered == alone.report.offered
            && alone.max_wait() == 0
            && alone.report.ownership_violations() == 0
            && alone.report.silent_drops == 0
            && alone.model_divergences == 0
            && alone.capacity_violations == 0;
        if !sane {
            return Err(format!(
                "the oracle disagrees with the reference queue in underload at capacity {capacity}: {:?}",
                alone.report
            ));
        }
        digests.push(alone.digest);
    }
    for run in &runs {
        // The reference queue and the sync channel are the oracle's other halves: each must
        // itself be clean on the main schedule. (The sync channel's closed-path run is only
        // compared with the mailbox's: a shadow model that missed the close must show up as a
        // divergence of the mailbox, a finding, not stop the run.)
        for (name, half) in [("reference queue", &run.r), ("sync channel", &run.s)] {
            if half.report.ownership_violations() != 0
                || half.report.silent_drops != 0
                || half.capacity_violations != 0
                || half.model_divergences != 0
                || half.closed_path_mismatches != 0
                || half.wait_bound_violations != 0
            {
                return Err(format!(
                    "the {name} fails the oracle at capacity {} ({}): {:?}",
                    run.cell.capacity, run.cell.class.name, half.report
                ));
            }
        }
    }
    Ok(Pass {
        runs,
        digests,
        main,
        closed,
        corpora,
    })
}

/// How many fingerprints differ between two passes of one seed.
pub fn determinism_mismatches(first: &[u64], second: &[u64]) -> u64 {
    let common = first.iter().zip(second).filter(|(a, b)| a != b).count();
    (common + first.len().abs_diff(second.len())) as u64
}

/// One row of the controls: a broken queue and whether the oracle caught it.
pub struct Probe {
    pub name: &'static str,
    pub negative_control: bool,
    pub expected: &'static str,
    pub tripped: bool,
    pub observed: Outcome,
}

fn probe(
    name: &'static str,
    negative_control: bool,
    expected: &'static str,
    observed: Outcome,
    holds: impl Fn(&Outcome) -> bool,
) -> Probe {
    Probe {
        name,
        negative_control,
        expected,
        tripped: holds(&observed),
        observed,
    }
}

/// Run the negative controls and the canaries on the control cell.
pub fn controls(pass: &Pass) -> Result<Vec<Probe>, String> {
    let capacity = params::CONTROL_CAPACITY;
    let class = params::PAYLOAD_CLASSES[0];
    let corpus = corpus_of(&pass.corpora, &class);
    let cfg = SimCfg::new(capacity, class.bytes, corpus);
    let mut closed_cfg = SimCfg::new(capacity, class.bytes, corpus);
    closed_cfg.recovery_phase = None;
    // The broken queues wrap the reference queue, not the mailbox: a control must not depend on
    // the product it is meant to discriminate from.
    let fresh = || ReferenceQueue::new(capacity);
    let on_main = |queue: &mut dyn Queue| sim::run(&cfg, &pass.main, queue);
    let on_closed = |queue: &mut dyn Queue| sim::run(&closed_cfg, &pass.closed, queue);

    let mut rows = Vec::new();
    // Negative control 1: the unbounded channel, read from the sweep itself.
    let unbounded = pass
        .runs
        .iter()
        .find(|run| run.cell.capacity == capacity && run.cell.class == class)
        .ok_or("the control cell is not in the sweep")?;
    let sum =
        |select: fn(&Outcome) -> u64| -> u64 { pass.runs.iter().map(|run| select(&run.u)).sum() };
    let tripped = sum(|u| u.capacity_violations) > 0 && sum(|u| u.wait_bound_violations) > 0;
    rows.push(Probe {
        name: "unbounded_channel",
        negative_control: true,
        expected: "capacity_violations > 0 and wait_bound_violations > 0",
        tripped,
        observed: Outcome {
            capacity_violations: sum(|u| u.capacity_violations),
            wait_bound_violations: sum(|u| u.wait_bound_violations),
            peak_depth: unbounded.u.peak_depth,
            ..Outcome::default()
        },
    });
    // Negative control 2: a queue that drops its oldest message when full.
    rows.push(probe(
        "drop_oldest",
        true,
        "silent_drops > 0 and dropped_while_owned > 0",
        on_main(&mut Faulty::new(fresh(), Fault::DropOldest, capacity))?,
        |o| o.report.silent_drops > 0 && o.report.dropped_while_owned > 0,
    ));
    // The canaries: each breaks one thing the oracle claims to see.
    rows.push(probe(
        "duplicate_delivery",
        false,
        "duplicate_deliveries > 0",
        on_main(&mut Faulty::new(
            fresh(),
            Fault::DuplicateDelivery,
            capacity,
        ))?,
        |o| o.report.duplicate_deliveries > 0,
    ));
    rows.push(probe(
        "payload_mutation",
        false,
        "payload_mutations > 0",
        on_main(&mut Faulty::new(fresh(), Fault::MutatePayload, capacity))?,
        |o| o.report.payload_mutations > 0,
    ));
    rows.push(probe(
        "lossy_consumer_path",
        false,
        "silent_drops > 0",
        on_main(&mut Faulty::new(fresh(), Fault::Lossy, capacity))?,
        |o| o.report.silent_drops > 0,
    ));
    rows.push(probe(
        "fifo_reversal",
        false,
        "fifo_violations > 0 and model_divergences > 0",
        on_main(&mut ReferenceQueue::lifo(capacity))?,
        |o| o.report.fifo_violations > 0 && o.model_divergences > 0,
    ));
    rows.push(probe(
        "capacity_plus_one",
        false,
        "capacity_violations > 0 and model_divergences > 0",
        on_main(&mut ReferenceQueue::new(capacity + 1))?,
        |o| o.capacity_violations > 0 && o.model_divergences > 0,
    ));
    rows.push(probe(
        "capacity_doubled",
        false,
        "wait_bound_violations > 0",
        on_main(&mut ReferenceQueue::new(capacity * 2))?,
        |o| o.wait_bound_violations > 0,
    ));
    rows.push(probe(
        "early_refusal",
        false,
        "spurious_refusals > 0",
        on_main(&mut Faulty::new(fresh(), Fault::RefuseEarly, capacity))?,
        |o| o.spurious_refusals > 0,
    ));
    rows.push(probe(
        "close_ignored",
        false,
        "closed_path_mismatches > 0",
        on_closed(&mut Faulty::new(fresh(), Fault::IgnoreClose, capacity))?,
        |o| o.closed_path_mismatches > 0,
    ));
    rows.push(probe(
        "closed_reported_empty",
        false,
        "closed_path_mismatches > 0",
        on_closed(&mut Faulty::new(fresh(), Fault::ClosedAsEmpty, capacity))?,
        |o| o.closed_path_mismatches > 0,
    ));
    for row in &rows {
        if row.negative_control && !row.tripped {
            return Err(format!(
                "the negative control {} did not trip ({}): the experiment could not tell it from a sound queue",
                row.name, row.expected
            ));
        }
    }
    Ok(rows)
}

/// What the capacity-0 and closed-path unit cases found.
#[derive(Debug, Default)]
pub struct UnitOutcome {
    pub capacity_violations: u64,
    pub model_divergences: u64,
    pub closed_path_mismatches: u64,
    pub report: Report,
    pub offers: u64,
    /// Panics of the product contained during the unit.
    pub product_panics: u64,
    /// Owner-drop units run (0 or 1).
    pub owner_drop_cases: u64,
}

/// Calls into the product from the unit cases, with panics counted.
struct Guard {
    panics: std::cell::Cell<u64>,
}

impl Guard {
    fn new() -> Self {
        Self {
            panics: std::cell::Cell::new(0),
        }
    }

    fn call<R>(&self, call: impl FnOnce() -> R) -> Option<R> {
        let result = contained(call);
        if result.is_none() {
            self.panics.set(self.panics.get() + 1);
        }
        result
    }
}

/// C7: a capacity-0 mailbox admits nothing and hands every value back, through the owner and
/// through a sender; once closed it hands every value back as `Closed`.
pub fn capacity_zero(seed: u64) -> UnitOutcome {
    const OFFERS: u32 = 8;
    let corpus = Corpus::generate(seed, 2 * OFFERS as usize, 64);
    let log = DropLog::new(2 * OFFERS as usize);
    let mut ledger = Ledger::new(corpus.digests());
    let guard = Guard::new();
    let mailbox = Mailbox::bounded(0);
    let sender = mailbox.sender();
    let mut out = UnitOutcome::default();
    let offer = |ledger: &mut Ledger, out: &mut UnitOutcome, id: u32, closed: bool| {
        let payload = corpus.make(&log, id, 0);
        ledger.offer(id, 0);
        out.offers += 1;
        let wanted = if closed {
            Refusal::Closed
        } else {
            Refusal::Full
        };
        let sent = guard.call(|| {
            if id.is_multiple_of(2) {
                mailbox.try_send(payload)
            } else {
                sender.try_send(payload)
            }
        });
        let (refusal, value) = match sent {
            Some(Ok(())) => {
                ledger.admitted(id);
                out.capacity_violations += 1;
                return;
            }
            Some(Err(SendError::Full(value))) => (Refusal::Full, value),
            Some(Err(SendError::Closed(value))) => (Refusal::Closed, value),
            None => {
                // Counted as a product panic; the id is recorded as the refusal it should be.
                let digest = corpus.digests()[id as usize];
                ledger.refused(id, wanted, Seen { id, digest });
                return;
            }
        };
        if refusal != wanted {
            out.closed_path_mismatches += 1;
        }
        ledger.refused(
            id,
            refusal,
            Seen {
                id: value.id(),
                digest: value.digest(),
            },
        );
        drop(value);
    };
    for id in 0..OFFERS {
        offer(&mut ledger, &mut out, id, false);
    }
    if guard.call(|| (mailbox.depth(), mailbox.peak_depth())) != Some((0, 0)) {
        out.capacity_violations += 1;
    }
    if !matches!(guard.call(|| mailbox.try_receive()), Some(Receive::Empty)) {
        out.model_divergences += 1;
    }
    let _ = guard.call(|| mailbox.close());
    for id in OFFERS..2 * OFFERS {
        offer(&mut ledger, &mut out, id, true);
    }
    if !matches!(guard.call(|| mailbox.try_receive()), Some(Receive::Closed)) {
        out.closed_path_mismatches += 1;
    }
    for dropped in log.take_recent() {
        ledger.dropped(dropped);
    }
    out.report = ledger.finish(&log);
    out.product_panics = guard.panics.get();
    out
}

/// C7b: dropping the owner closes the mailbox for the senders that outlive it and releases the
/// payloads still queued. Two messages are queued, the owner is dropped, and a sender then
/// offers a third: it must come back `Closed` with its value, and the two queued payloads must
/// have been dropped exactly once.
pub fn owner_drop(seed: u64) -> UnitOutcome {
    let corpus = Corpus::generate(seed ^ 0x6f77_6e65, 3, 64);
    let log = DropLog::new(3);
    let mut ledger = Ledger::new(corpus.digests());
    let guard = Guard::new();
    let mut out = UnitOutcome {
        owner_drop_cases: 1,
        ..UnitOutcome::default()
    };
    let mailbox = Mailbox::bounded(2);
    let sender = mailbox.sender();
    for id in 0..2u32 {
        let payload = corpus.make(&log, id, 0);
        out.offers += 1;
        match guard.call(|| sender.try_send(payload)) {
            Some(Ok(())) => {}
            _ => out.model_divergences += 1,
        }
    }
    let _ = guard.call(move || drop(mailbox));
    if guard.call(|| sender.is_closed()) != Some(true) {
        out.closed_path_mismatches += 1;
    }
    if guard.call(|| sender.depth()) != Some(0) {
        out.model_divergences += 1;
    }
    // The queued payloads were released with the owner, each exactly once.
    if log.drops_of(0) != 1 || log.drops_of(1) != 1 || log.live() != 0 {
        out.report.drop_count_mismatches += 1;
    }
    let payload = corpus.make(&log, 2, 0);
    ledger.offer(2, 0);
    out.offers += 1;
    match guard.call(|| sender.try_send(payload)) {
        Some(Err(SendError::Closed(value))) => {
            ledger.refused(
                2,
                Refusal::Closed,
                Seen {
                    id: value.id(),
                    digest: value.digest(),
                },
            );
        }
        Some(Ok(())) => {
            ledger.admitted(2);
            out.closed_path_mismatches += 1;
        }
        Some(Err(SendError::Full(value))) => {
            ledger.refused(
                2,
                Refusal::Full,
                Seen {
                    id: value.id(),
                    digest: value.digest(),
                },
            );
            out.closed_path_mismatches += 1;
        }
        None => {
            let digest = corpus.digests()[2];
            ledger.refused(2, Refusal::Closed, Seen { id: 2, digest });
        }
    }
    for dropped in log.take_recent() {
        // Ids 0 and 1 were never offered to the ledger; only id 2 is tracked.
        if dropped == 2 {
            ledger.dropped(dropped);
        }
    }
    let mut report = ledger.finish(&log);
    // Ids 0 and 1 are outside the ledger's account: they are checked above by their drop counts.
    report.drop_count_mismatches += out.report.drop_count_mismatches;
    out.report = report;
    out.product_panics = guard.panics.get();
    out
}

/// A queue-level view used by the unit tests of the cases.
#[cfg(test)]
fn admits_nothing(queue: &mut dyn Queue, corpus: &Corpus, log: &std::sync::Arc<DropLog>) -> bool {
    (0..4).all(|id| {
        matches!(
            queue.offer(corpus.make(log, id, 0)),
            super::queue::Offer::Full(_)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sweep_is_every_capacity_with_every_class() {
        let all = cells();
        assert_eq!(all.len(), 6);
        assert_eq!(all[0].capacity, 1);
        assert_eq!(all[0].class.name, "small");
        assert_eq!(all[5].capacity, 128);
        assert_eq!(all[5].class.name, "large");
    }

    #[test]
    fn a_pass_is_clean_for_the_mailbox_and_deterministic() {
        let first = pass(17).expect("a pass");
        let second = pass(17).expect("a pass");
        assert_eq!(determinism_mismatches(&first.digests, &second.digests), 0);
        assert_eq!(first.digests.len(), 6 * 6 + 3);
        for run in &first.runs {
            let b = &run.b;
            assert_eq!(against_reference(b, &run.r), 0, "{:?}", run.cell);
            assert_eq!(
                against_reference(b, &run.s),
                0,
                "sync channel {:?}",
                run.cell
            );
            assert_eq!(
                against_reference(&run.closed, &run.closed_s),
                0,
                "{:?}",
                run.cell
            );
            assert_eq!(b.product_panics + run.closed.product_panics, 0);
            assert!(
                run.closed.closed_over_full > 0 || run.cell.capacity == 128,
                "{:?}",
                run.cell
            );
            assert!(run.closed.drained_after_close > 0, "{:?}", run.cell);
            assert_eq!(b.capacity_violations, 0);
            assert_eq!(b.model_divergences, 0);
            assert_eq!(b.report.ownership_violations(), 0);
            assert_eq!(b.wait_bound_violations, 0);
            assert_eq!(b.payload_bound_violations, 0);
            assert!(b.recovered, "{:?}", run.cell);
            assert_eq!(run.closed.closed_path_mismatches, 0);
            assert_eq!(run.closed.model_divergences, 0);
            assert!(run.closed.report.refused_closed > 0);
            assert!(run.u.peak_depth > run.cell.capacity, "{:?}", run.cell);
            assert!(run.u.capacity_violations > 0);
        }
    }

    #[test]
    fn a_different_seed_changes_the_trace() {
        let first = pass(17).unwrap();
        let other = pass(29).unwrap();
        assert!(determinism_mismatches(&first.digests, &other.digests) > 0);
        assert!(determinism_mismatches(&first.digests, &first.digests[..3]) > 0);
    }

    #[test]
    fn every_control_trips_and_every_canary_is_caught() {
        let pass = pass(17).unwrap();
        let rows = controls(&pass).expect("the negative controls trip");
        assert_eq!(rows.iter().filter(|row| row.negative_control).count(), 2);
        assert_eq!(rows.len(), 11);
        for row in &rows {
            assert!(row.tripped, "{} did not trip: {}", row.name, row.expected);
        }
    }

    #[test]
    fn a_negative_control_that_does_not_trip_is_a_harness_error() {
        // A sweep whose unbounded arm never exceeds anything cannot be told from a bounded one.
        let mut pass = pass(17).unwrap();
        for run in &mut pass.runs {
            run.u.capacity_violations = 0;
        }
        let error = controls(&pass).err().expect("an error");
        assert!(error.contains("unbounded_channel"), "{error}");
    }

    #[test]
    fn capacity_zero_refuses_everything_and_returns_every_value() {
        let out = capacity_zero(17);
        assert_eq!(out.capacity_violations, 0);
        assert_eq!(out.model_divergences, 0);
        assert_eq!(out.closed_path_mismatches, 0);
        assert_eq!(out.report.ownership_violations(), 0);
        assert_eq!(out.report.silent_drops, 0);
        assert_eq!(out.report.refused_full, 8);
        assert_eq!(out.report.refused_closed, 8);
        assert_eq!(out.offers, 16);
    }

    #[test]
    fn dropping_the_owner_closes_the_senders_and_releases_the_queue() {
        let out = owner_drop(17);
        assert_eq!(out.owner_drop_cases, 1);
        assert_eq!(out.closed_path_mismatches, 0);
        assert_eq!(out.model_divergences, 0);
        assert_eq!(out.report.ownership_violations(), 0);
        assert_eq!(out.report.silent_drops, 0);
        assert_eq!(out.report.refused_closed, 1);
        assert_eq!(out.product_panics, 0);
    }

    #[test]
    fn a_zero_capacity_reference_queue_agrees() {
        let corpus = Corpus::generate(1, 4, 8);
        let log = DropLog::new(4);
        assert!(admits_nothing(&mut ReferenceQueue::new(0), &corpus, &log));
        assert!(admits_nothing(
            &mut MailboxQueue::new(0, CloseVia::Owner),
            &corpus,
            &log
        ));
    }
}
