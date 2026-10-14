//! Arm T: real producer threads, each with its own clone of a `MailboxSender`, and the
//! mailbox's owner as the one consumer.
//!
//! Three stages, separated by a poisonable barrier so that the counts that matter are exact
//! whatever the scheduler does:
//!
//! 1. `fill_rounds` exact fills: the consumer is parked, all producers are released together
//!    (they spin on an atomic after the barrier, so that they start within nanoseconds) and
//!    each offers `fill_per_producer` messages that were built beforehand. Exactly `capacity`
//!    are admitted in total, whatever the interleaving, and the consumer then finds exactly
//!    `capacity` in the queue. Repeating the fill thousands of times is what gives a race in the
//!    capacity check (a check-then-push with the lock released between) a chance to show;
//! 2. one contended stage: the consumer drains while the producers offer concurrently, and
//!    closes the mailbox part way through (after `close_after_received` receives, or at the
//!    latest when half of the stage's offers were made). Every offer is admitted, refused `Full` or refused
//!    `Closed`, and an `Ok` to a producer that had already seen `close()` return is a
//!    violation (`closed_returned`, set with `SeqCst` after `close()` returns and read with
//!    `SeqCst` before each send);
//! 3. after the close, every offer must come back `Closed` with its value.
//!
//! The ownership ledger is checked after the threads have joined, over every producer's record
//! and the consumer's: no loss, no duplication, no mutation, and each producer's messages
//! arrive in the order it sent them. The wall-clock latency and the resident set are recorded,
//! descriptive only: no rule reads them.
//!
//! Every call into the product is contained (a panic is counted in `product_panics`, see
//! `guard`), and the whole stage runs under a watchdog: a stage that does not finish within
//! [`WATCHDOG`] is a hang of the product (`product_hangs`), the barrier is poisoned so that
//! the threads that can leave do, and the run reports it.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use ptr_exec::{Mailbox, MailboxSender, Receive, SendError};

use super::guard::contained;
use super::ledger::{Ledger, Refusal, Report, Seen};
use super::params::ThreadedShape;
use super::payload::{Corpus, DropLog, Tracked};

/// How long the threaded stage may take before it counts as a hang of the product. A normal
/// run takes a fraction of a second; the deadline only decides what a stuck run is called.
pub const WATCHDOG: Duration = Duration::from_secs(20);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Admitted,
    Full,
    Closed,
    /// The product panicked in the call; the ledger books the id as refused `Full`.
    Panicked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Fill,
    Contend,
    Closed,
}

struct Sent {
    id: u32,
    stage: Stage,
    kind: Kind,
    returned: Seen,
}

/// A barrier that can be poisoned: once poisoned, every waiter (now or later) returns `Err`.
struct Rounds {
    parties: usize,
    state: Mutex<RoundState>,
    turn: Condvar,
}

struct RoundState {
    arrived: usize,
    generation: u64,
    poisoned: bool,
}

#[derive(Debug)]
struct Poisoned;

impl Rounds {
    fn new(parties: usize) -> Self {
        Self {
            parties,
            state: Mutex::new(RoundState {
                arrived: 0,
                generation: 0,
                poisoned: false,
            }),
            turn: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RoundState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait(&self) -> Result<(), Poisoned> {
        let mut state = self.lock();
        if state.poisoned {
            return Err(Poisoned);
        }
        state.arrived += 1;
        if state.arrived == self.parties {
            state.arrived = 0;
            state.generation += 1;
            self.turn.notify_all();
            return Ok(());
        }
        let generation = state.generation;
        loop {
            state = self
                .turn
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
            if state.poisoned {
                return Err(Poisoned);
            }
            if state.generation != generation {
                return Ok(());
            }
        }
    }

    fn poison(&self) {
        self.lock().poisoned = true;
        self.turn.notify_all();
    }

    fn poisoned(&self) -> bool {
        self.lock().poisoned
    }
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub producers: u64,
    pub offered: u64,
    pub admitted: u64,
    pub refused_full: u64,
    pub refused_closed: u64,
    pub delivered: u64,
    pub report: Report,
    /// Messages of one producer received out of the order it sent them.
    pub order_violations: u64,
    /// A fill that admitted other than `capacity`, a queue that held another number than it
    /// admitted, or a depth above the capacity.
    pub capacity_violations: u64,
    /// Offers after the close not refused `Closed`, a `Closed` followed by another answer to the
    /// same producer, or a drained closed mailbox not saying so.
    pub closed_path_mismatches: u64,
    /// Offers admitted after the producer had seen `close()` return.
    pub admitted_after_close: u64,
    /// Panics of the product contained during the stage.
    pub product_panics: u64,
    /// 1 when the stage did not finish within the watchdog.
    pub product_hangs: u64,
    /// Exact fills completed.
    pub fill_rounds: u64,
    /// Outcomes observed in the contended stage.
    pub contend_admitted: u64,
    pub contend_full: u64,
    pub contend_closed: u64,
    /// Producers with at least one admitted offer.
    pub producers_with_admissions: u64,
    /// Fills in which at least two producers got a message in: the producers really met at the
    /// queue. A race in the capacity check needs this; a run in which the scheduler serialised
    /// the producers cannot show one.
    pub interleaved_fills: u64,
    pub peak_depth: usize,
    pub latencies_ns: Vec<u64>,
    pub elapsed_ms: f64,
    pub peak_rss_kib: Option<u64>,
}

impl Outcome {
    /// Ledger violations of every kind: the order check and the Ok after a returned close
    /// included.
    pub fn ledger_violations(&self) -> u64 {
        self.report.ownership_violations()
            + self.report.silent_drops
            + self.report.fifo_violations
            + self.order_violations
            + self.admitted_after_close
    }
}

/// The resident-set high-water mark of this process, from `/proc` (None elsewhere).
pub fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|number| number.parse().ok())
}

/// What one producer thread brings back.
struct Record {
    sent: Vec<Sent>,
    panics: u64,
    admitted_after_close: u64,
    closed_path_mismatches: u64,
    /// The messages this producer got admitted in each fill, in order.
    fill_admissions: Vec<u8>,
}

/// What a producer thread shares with the others and the consumer.
struct Shared {
    shape: ThreadedShape,
    corpus: Corpus,
    log: Arc<DropLog>,
    rounds: Rounds,
    base: Instant,
    /// Set (`SeqCst`) by the closer right after `close()` returned.
    closed_returned: AtomicBool,
    fill_admitted: AtomicUsize,
    /// The number of fills released so far: the consumer raises it, the producers spin on it.
    fill_go: AtomicU32,
    contend_done: AtomicUsize,
    /// Producers that reached the middle of their contended offers and wait for the close.
    contend_halfway: AtomicUsize,
    /// Set (`SeqCst`) once the closer has tried to close, whether or not the call returned.
    close_attempted: AtomicBool,
    /// Product panics contained on the consumer's side.
    consumer_panics: AtomicU64,
}

fn seen(payload: &Tracked) -> Seen {
    Seen {
        id: payload.id(),
        digest: payload.digest(),
    }
}

impl Shared {
    fn stride(&self) -> u32 {
        self.shape.fill_rounds * self.shape.fill_per_producer
            + self.shape.contend_per_producer
            + self.shape.closed_per_producer
    }

    /// One offer from a producer, with the close flag read first.
    fn offer(
        &self,
        sender: &MailboxSender<Tracked>,
        id: u32,
        stage: Stage,
        payload: Tracked,
        record: &mut Record,
        seen_closed: &mut bool,
    ) {
        let flag_was_set = self.closed_returned.load(Ordering::SeqCst);
        let (kind, returned) = match contained(|| sender.try_send(payload)) {
            Some(Ok(())) => {
                record.admitted_after_close += u64::from(flag_was_set);
                (Kind::Admitted, None)
            }
            Some(Err(SendError::Full(value))) => (Kind::Full, Some(seen(&value))),
            Some(Err(SendError::Closed(value))) => (Kind::Closed, Some(seen(&value))),
            None => {
                record.panics += 1;
                (Kind::Panicked, None)
            }
        };
        // Once a producer has been told Closed, every later answer must be Closed too.
        if *seen_closed && kind != Kind::Closed {
            record.closed_path_mismatches += 1;
        }
        *seen_closed |= kind == Kind::Closed;
        if stage == Stage::Closed && kind != Kind::Closed {
            record.closed_path_mismatches += 1;
        }
        let returned = returned.unwrap_or(Seen {
            id,
            digest: self.corpus.digests()[id as usize],
        });
        record.sent.push(Sent {
            id,
            stage,
            kind,
            returned,
        });
    }

    fn produce(&self, producer: usize, sender: MailboxSender<Tracked>) -> Result<Record, Poisoned> {
        let shape = self.shape;
        let first = producer as u32 * self.stride();
        let mut record = Record {
            sent: Vec::with_capacity(self.stride() as usize),
            panics: 0,
            admitted_after_close: 0,
            closed_path_mismatches: 0,
            fill_admissions: Vec::with_capacity(shape.fill_rounds as usize),
        };
        let mut seen_closed = false;
        // Stage 1: exact fills. The payloads are built before the release, so that the offers
        // of all producers hit the queue together.
        for round in 0..shape.fill_rounds {
            let base_id = first + round * shape.fill_per_producer;
            let batch: Vec<(u32, Tracked)> = (0..shape.fill_per_producer)
                .map(|n| (base_id + n, self.corpus.make(&self.log, base_id + n, 0)))
                .collect();
            self.rounds.wait()?;
            // All producers have arrived and hold their batch. A condvar wake-up would start
            // them microseconds apart, longer than a whole fill; they spin on `fill_go` instead,
            // so that they reach the queue together and contend for the last free slot.
            let mut spins = 0u32;
            while self.fill_go.load(Ordering::Acquire) <= round {
                spins += 1;
                if spins.is_multiple_of(4096) {
                    if self.rounds.poisoned() {
                        return Err(Poisoned);
                    }
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            }
            let before = record.sent.len();
            for (id, payload) in batch {
                self.offer(
                    &sender,
                    id,
                    Stage::Fill,
                    payload,
                    &mut record,
                    &mut seen_closed,
                );
            }
            let admitted = record.sent[before..]
                .iter()
                .filter(|sent| sent.kind == Kind::Admitted)
                .count();
            self.fill_admitted.fetch_add(admitted, Ordering::SeqCst);
            record.fill_admissions.push(admitted.min(255) as u8);
            self.rounds.wait()?;
        }
        // Stage 2: contended, the consumer drains and closes part way through.
        let contend_first = first + shape.fill_rounds * shape.fill_per_producer;
        self.rounds.wait()?;
        for n in 0..shape.contend_per_producer {
            let id = contend_first + n;
            let stamp = self.base.elapsed().as_nanos() as u64;
            let payload = self.corpus.make(&self.log, id, stamp);
            self.offer(
                &sender,
                id,
                Stage::Contend,
                payload,
                &mut record,
                &mut seen_closed,
            );
            // Each producer makes half of its contended offers before the close and the other
            // half after it, so that all three outcomes are structurally present: the first half
            // races the draining consumer, the second half follows the close.
            if n + 1 == shape.contend_per_producer / 2 {
                self.contend_halfway.fetch_add(1, Ordering::SeqCst);
                while !self.close_attempted.load(Ordering::SeqCst) {
                    if self.rounds.poisoned() {
                        return Err(Poisoned);
                    }
                    std::thread::yield_now();
                }
            }
        }
        self.contend_done.fetch_add(1, Ordering::SeqCst);
        self.rounds.wait()?;
        // Stage 3: after the close.
        let closed_first = contend_first + shape.contend_per_producer;
        self.rounds.wait()?;
        for n in 0..shape.closed_per_producer {
            let id = closed_first + n;
            let payload = self.corpus.make(&self.log, id, 0);
            self.offer(
                &sender,
                id,
                Stage::Closed,
                payload,
                &mut record,
                &mut seen_closed,
            );
        }
        self.rounds.wait()?;
        Ok(record)
    }
}

/// Run the stages for `seed` under the watchdog. An `Err` is a fault of the harness (a thread
/// that could not start or hit a harness panic), not a finding; a product that hangs is a
/// finding and comes back as an `Outcome` with `product_hangs` set.
pub fn run(seed: u64, shape: ThreadedShape) -> Result<Outcome, String> {
    run_with_deadline(seed, shape, WATCHDOG)
}

pub fn run_with_deadline(
    seed: u64,
    shape: ThreadedShape,
    deadline: Duration,
) -> Result<Outcome, String> {
    run_gated(seed, shape, deadline, None)
}

/// `run_with_deadline` with an optional start gate: the coordinator waits for one message on it
/// before it starts, and leaves without running when the gate's sender is dropped instead. A
/// test that holds the gate closed gets the timeout path whatever the scheduler does.
fn run_gated(
    seed: u64,
    shape: ThreadedShape,
    deadline: Duration,
    start_gate: Option<std::sync::mpsc::Receiver<()>>,
) -> Result<Outcome, String> {
    let started = Instant::now();
    let stride = shape.fill_rounds * shape.fill_per_producer
        + shape.contend_per_producer
        + shape.closed_per_producer;
    let total = shape.producers * stride as usize;
    let shared = Arc::new(Shared {
        shape,
        corpus: Corpus::generate(seed ^ 0x7468_7265_6164, total, 64),
        log: DropLog::new(total),
        rounds: Rounds::new(shape.producers + 1),
        base: Instant::now(),
        closed_returned: AtomicBool::new(false),
        fill_admitted: AtomicUsize::new(0),
        fill_go: AtomicU32::new(0),
        contend_done: AtomicUsize::new(0),
        contend_halfway: AtomicUsize::new(0),
        close_attempted: AtomicBool::new(false),
        consumer_panics: AtomicU64::new(0),
    });
    let (done, result) = std::sync::mpsc::channel();
    let coordinator = Arc::clone(&shared);
    std::thread::Builder::new()
        .name("r001-coordinator".to_string())
        .spawn(move || {
            if let Some(gate) = start_gate {
                if gate.recv().is_err() {
                    return;
                }
            }
            let _ = done.send(coordinate(&coordinator));
        })
        .map_err(|error| format!("could not start the coordinator thread: {error}"))?;
    match result.recv_timeout(deadline) {
        Ok(outcome) => {
            let mut outcome = outcome?;
            outcome.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            Ok(outcome)
        }
        Err(RecvTimeoutError::Timeout) => {
            // The threads that can leave do; a thread stuck inside the product is left behind.
            shared.rounds.poison();
            Ok(Outcome {
                producers: shape.producers as u64,
                product_hangs: 1,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                ..Outcome::default()
            })
        }
        Err(RecvTimeoutError::Disconnected) => {
            shared.rounds.poison();
            Err("the coordinator thread hit a harness fault".to_string())
        }
    }
}

/// The consumer side: starts the producers, runs the stages in step with them and audits.
fn coordinate(shared: &Arc<Shared>) -> Result<Outcome, String> {
    let shape = shared.shape;
    let stride = shared.stride();
    let panics = Cell::new(0u64);
    macro_rules! guard {
        ($call:expr) => {
            match contained(|| $call) {
                Some(value) => Some(value),
                None => {
                    panics.set(panics.get() + 1);
                    None
                }
            }
        };
    }
    let Some(mailbox) = guard!(Mailbox::bounded(shape.capacity)) else {
        return Ok(early(shape, panics.get()));
    };
    let Some(origin) = guard!(mailbox.sender()) else {
        return Ok(early(shape, panics.get()));
    };
    let mut handles = Vec::new();
    for producer in 0..shape.producers {
        let Some(sender) = guard!(origin.clone()) else {
            shared.rounds.poison();
            return Ok(early(shape, panics.get()));
        };
        let worker = Arc::clone(shared);
        let handle = std::thread::Builder::new()
            .name(format!("r001-producer-{producer}"))
            .spawn(move || {
                // A harness panic must not leave the others waiting at the barrier.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker.produce(producer, sender)
                }));
                match result {
                    Ok(Ok(record)) => Some(record),
                    Ok(Err(Poisoned)) | Err(_) => {
                        worker.rounds.poison();
                        None
                    }
                }
            })
            .map_err(|error| format!("could not start a producer thread: {error}"))?;
        handles.push(handle);
    }
    drop(origin);

    let mut out = Outcome {
        producers: shape.producers as u64,
        ..Outcome::default()
    };
    let aborted = || "the threaded stage was aborted (a producer thread failed)".to_string();
    let mut received: Vec<Seen> = Vec::new();
    let mut latencies: Vec<u64> = Vec::new();
    let take = |received: &mut Vec<Seen>, latencies: Option<&mut Vec<u64>>| -> Receive<()> {
        match contained(|| mailbox.try_receive()) {
            Some(Receive::Message(payload)) => {
                if let Some(latencies) = latencies {
                    let now = shared.base.elapsed().as_nanos() as u64;
                    latencies.push(now.saturating_sub(payload.stamp_ns()));
                }
                received.push(seen(&payload));
                Receive::Message(())
            }
            Some(Receive::Empty) => Receive::Empty,
            Some(Receive::Closed) => Receive::Closed,
            None => {
                shared.consumer_panics.fetch_add(1, Ordering::SeqCst);
                Receive::Empty
            }
        }
    };

    // Stage 1: exact fills, the consumer parked while the producers offer.
    for round in 0..shape.fill_rounds {
        shared.rounds.wait().map_err(|_| aborted())?;
        shared.fill_go.store(round + 1, Ordering::Release);
        shared.rounds.wait().map_err(|_| aborted())?;
        let admitted = shared.fill_admitted.swap(0, Ordering::SeqCst);
        out.capacity_violations += admitted.abs_diff(shape.capacity) as u64;
        let depth = guard!(mailbox.depth()).unwrap_or(0);
        if depth != admitted.min(shape.capacity) {
            out.capacity_violations += 1;
        }
        // The real queue, not the depth mirror: exactly `admitted` messages are recoverable.
        let mut recovered = 0usize;
        loop {
            match take(&mut received, None) {
                Receive::Message(()) => recovered += 1,
                Receive::Empty => break,
                Receive::Closed => {
                    out.closed_path_mismatches += 1;
                    break;
                }
            }
        }
        out.capacity_violations += recovered.abs_diff(admitted) as u64;
        out.fill_rounds += 1;
    }

    // Stage 2: the consumer drains while the producers offer, and closes part way through.
    shared.rounds.wait().map_err(|_| aborted())?;
    let mut closed_by_consumer = false;
    let mut received_in_stage = 0u32;
    while shared.contend_done.load(Ordering::SeqCst) < shape.producers {
        if shared.rounds.poisoned() {
            return Err(aborted());
        }
        if guard!(mailbox.depth()).is_some_and(|depth| depth > shape.capacity) {
            out.capacity_violations += 1;
        }
        // Close once enough was received, or, should the producers outrun the consumer, once
        // every producer has made half of its offers and waits for the close.
        let halfway = shared.contend_halfway.load(Ordering::SeqCst) >= shape.producers;
        if !closed_by_consumer && (received_in_stage >= shape.close_after_received || halfway) {
            closed_by_consumer = true;
            if guard!(mailbox.close()).is_some() {
                shared.closed_returned.store(true, Ordering::SeqCst);
            }
            shared.close_attempted.store(true, Ordering::SeqCst);
        }
        if matches!(
            take(&mut received, Some(&mut latencies)),
            Receive::Message(())
        ) {
            received_in_stage += 1;
        }
    }
    shared.rounds.wait().map_err(|_| aborted())?;
    if guard!(mailbox.close()).is_some() {
        shared.closed_returned.store(true, Ordering::SeqCst);
    }
    shared.close_attempted.store(true, Ordering::SeqCst);

    // Stage 3: every offer after the close comes back Closed.
    shared.rounds.wait().map_err(|_| aborted())?;
    shared.rounds.wait().map_err(|_| aborted())?;
    loop {
        match take(&mut received, Some(&mut latencies)) {
            Receive::Message(()) => {}
            Receive::Closed => break,
            Receive::Empty => {
                out.closed_path_mismatches += 1;
                break;
            }
        }
    }
    if !matches!(contained(|| mailbox.try_receive()), Some(Receive::Closed)) {
        out.closed_path_mismatches += 1;
    }

    let mut records: Vec<Record> = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(Some(record)) => records.push(record),
            Ok(None) | Err(_) => return Err(aborted()),
        }
    }
    let peak = guard!(mailbox.peak_depth()).unwrap_or(0);
    out.peak_depth = peak;
    if peak > shape.capacity {
        out.capacity_violations += (peak - shape.capacity) as u64;
    }
    let _ = guard!(drop(mailbox));

    let mut ledger = Ledger::new(shared.corpus.digests()).without_order();
    let mut with_admission = vec![false; shape.producers];
    for record in &records {
        out.admitted_after_close += record.admitted_after_close;
        out.closed_path_mismatches += record.closed_path_mismatches;
        out.product_panics += record.panics;
        for sent in &record.sent {
            ledger.offer(sent.id, 0);
            match sent.kind {
                Kind::Admitted => {
                    ledger.admitted(sent.id);
                    with_admission[(sent.id / stride) as usize] = true;
                }
                Kind::Full | Kind::Panicked => {
                    ledger.refused(sent.id, Refusal::Full, sent.returned)
                }
                Kind::Closed => ledger.refused(sent.id, Refusal::Closed, sent.returned),
            }
            if sent.stage == Stage::Contend {
                match sent.kind {
                    Kind::Admitted => out.contend_admitted += 1,
                    Kind::Full => out.contend_full += 1,
                    Kind::Closed => out.contend_closed += 1,
                    Kind::Panicked => {}
                }
            }
        }
    }
    out.interleaved_fills = (0..shape.fill_rounds as usize)
        .filter(|round| {
            records
                .iter()
                .filter(|record| record.fill_admissions.get(*round).is_some_and(|n| *n > 0))
                .count()
                >= 2
        })
        .count() as u64;
    out.product_panics += panics.get() + shared.consumer_panics.load(Ordering::SeqCst);
    out.producers_with_admissions = with_admission.iter().filter(|seen| **seen).count() as u64;
    for value in &received {
        ledger.started(*value, 0);
        ledger.finished(value.id);
    }
    // Each producer's messages must arrive in the order that producer sent them.
    for producer in 0..shape.producers as u32 {
        let lane: Vec<u32> = received
            .iter()
            .filter(|value| value.id / stride == producer)
            .map(|value| value.id)
            .collect();
        out.order_violations += lane.windows(2).filter(|pair| pair[1] <= pair[0]).count() as u64;
    }
    out.report = ledger.finish(&shared.log);
    out.offered = out.report.offered;
    out.admitted = out.report.admitted;
    out.refused_full = out.report.refused_full;
    out.refused_closed = out.report.refused_closed;
    out.delivered = out.report.delivered;
    out.latencies_ns = latencies;
    out.peak_rss_kib = peak_rss_kib();
    Ok(out)
}

/// An outcome for a product that panicked before any producer could start.
fn early(shape: ThreadedShape, panics: u64) -> Outcome {
    Outcome {
        producers: shape.producers as u64,
        product_panics: panics,
        ..Outcome::default()
    }
}

#[cfg(test)]
mod tests {
    use super::super::params;
    use super::*;

    #[test]
    fn the_threaded_arm_keeps_every_promise_with_real_contention() {
        let shape = params::THREADED_REDUCED;
        let out = run(17, shape).expect("threads run");
        assert_eq!(out.producers, 4);
        let stride = u64::from(
            shape.fill_rounds * shape.fill_per_producer
                + shape.contend_per_producer
                + shape.closed_per_producer,
        );
        assert_eq!(out.offered, shape.producers as u64 * stride);
        assert_eq!(out.ledger_violations(), 0, "{:?}", out.report);
        assert_eq!(out.capacity_violations, 0);
        assert_eq!(out.closed_path_mismatches, 0);
        assert_eq!(
            (
                out.product_panics,
                out.product_hangs,
                out.admitted_after_close
            ),
            (0, 0, 0)
        );
        assert!(out.peak_depth <= shape.capacity);
        assert_eq!(out.delivered, out.admitted);
        assert_eq!(
            out.admitted + out.refused_full + out.refused_closed,
            out.offered
        );
        // The exact fills admitted exactly the capacity each; round 3 is all Closed.
        assert_eq!(out.fill_rounds, u64::from(shape.fill_rounds));
        assert!(out.admitted >= u64::from(shape.fill_rounds) * shape.capacity as u64);
        assert!(
            out.refused_closed >= shape.producers as u64 * u64::from(shape.closed_per_producer)
        );
        // Observed, not configured: all three outcomes happened in the contended stage.
        assert!(out.contend_admitted >= 1 && out.contend_full >= 1 && out.contend_closed >= 1);
        assert!(out.producers_with_admissions >= 2);
        assert!(out.interleaved_fills <= u64::from(shape.fill_rounds));
        assert!(!out.latencies_ns.is_empty());
    }

    #[test]
    fn a_poisoned_barrier_releases_every_waiter_with_an_error() {
        let rounds = Arc::new(Rounds::new(3));
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let rounds = Arc::clone(&rounds);
                std::thread::spawn(move || rounds.wait().is_err())
            })
            .collect();
        std::thread::sleep(Duration::from_millis(50));
        rounds.poison();
        for waiter in waiters {
            assert!(waiter.join().unwrap());
        }
        assert!(rounds.wait().is_err(), "later waiters fail at once");
        // An unpoisoned barrier lets a full set through, twice.
        let rounds = Arc::new(Rounds::new(2));
        let other = Arc::clone(&rounds);
        let thread = std::thread::spawn(move || (other.wait().is_ok(), other.wait().is_ok()));
        assert!(rounds.wait().is_ok() && rounds.wait().is_ok());
        assert_eq!(thread.join().unwrap(), (true, true));
    }

    #[test]
    fn a_stage_that_misses_its_deadline_is_a_hang_not_a_harness_fault() {
        let shape = params::THREADED_REDUCED;
        // The start gate stays closed until the run has returned, so the coordinator cannot report
        // before the deadline on any scheduler or timer resolution; dropping the gate afterwards
        // lets the coordinator thread leave without running.
        let (gate, start_gate) = std::sync::mpsc::channel();
        let out =
            run_gated(17, shape, Duration::from_millis(10), Some(start_gate)).expect("an outcome");
        drop(gate);
        assert_eq!(out.product_hangs, 1);
        assert_eq!(out.offered, 0);
        assert_eq!(out.producers, 4);
    }
}
