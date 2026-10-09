//! The ownership ledger: the oracle that decides whether a queue kept its promises about
//! the messages it was given.
//!
//! Every offered id must end in exactly one terminal state: delivered to the consumer, or
//! refused and handed back to the producer (`Full` or `Closed`). The ledger holds the
//! fingerprint of every payload as it was created, so it notices a payload that arrives
//! altered, an id delivered twice, an id nobody offered, a payload dropped while a queue was
//! meant to own it, and an id that simply vanished (a silent drop). It shares no code with
//! any queue under test.

use super::payload::DropLog;

/// What the ledger knows an id to be.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fate {
    Unseen,
    /// Offered; the queue's answer has not been recorded yet.
    Offered,
    Queued,
    InService,
    Delivered,
    RefusedFull,
    RefusedClosed,
}

impl Fate {
    fn terminal(self) -> bool {
        matches!(
            self,
            Self::Delivered | Self::RefusedFull | Self::RefusedClosed
        )
    }
}

/// How a refused offer was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refusal {
    Full,
    Closed,
}

/// An observed payload: what the ledger needs to know of it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Seen {
    pub id: u32,
    pub digest: u64,
}

/// The causes of ownership violations and the totals, after a run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Report {
    pub offered: u64,
    pub admitted: u64,
    pub refused_full: u64,
    pub refused_closed: u64,
    pub delivered: u64,
    /// Offered ids with no terminal state at the end of the run.
    pub silent_drops: u64,
    /// An id delivered (started) when it was already in service or delivered.
    pub duplicate_deliveries: u64,
    /// A delivery of an id the queue was never given, or whose offer was refused.
    pub phantom_deliveries: u64,
    /// A delivered or returned payload whose fingerprint is not the one created.
    pub payload_mutations: u64,
    /// A refused offer handed back a payload other than the one offered.
    pub wrong_returns: u64,
    /// A payload dropped while a queue or the consumer was meant to own it.
    pub dropped_while_owned: u64,
    /// Live payloads differing from queue depth plus the one in service.
    pub live_mismatches: u64,
    /// A completion for an id that was not in service.
    pub spurious_completions: u64,
    /// Offered ids not dropped exactly once by the end of the run.
    pub drop_count_mismatches: u64,
    /// Deliveries out of arrival order (ids are assigned in arrival order).
    pub fifo_violations: u64,
}

impl Report {
    /// Every cause of an ownership violation, summed (the FIFO order is its own counter).
    pub fn ownership_violations(&self) -> u64 {
        self.duplicate_deliveries
            + self.phantom_deliveries
            + self.payload_mutations
            + self.wrong_returns
            + self.dropped_while_owned
            + self.live_mismatches
            + self.spurious_completions
            + self.drop_count_mismatches
    }
}

pub struct Ledger {
    expected: Vec<u64>,
    fate: Vec<Fate>,
    arrival: Vec<u64>,
    outstanding: u64,
    watermark: Option<u32>,
    check_order: bool,
    report: Report,
}

impl Ledger {
    /// A ledger for ids `0..expected.len()`, `expected[id]` being the created fingerprint.
    pub fn new(expected: &[u64]) -> Self {
        Self {
            expected: expected.to_vec(),
            fate: vec![Fate::Unseen; expected.len()],
            arrival: vec![0; expected.len()],
            outstanding: 0,
            watermark: None,
            check_order: true,
            report: Report::default(),
        }
    }

    /// Do not judge the global delivery order (several producers interleave their ids).
    pub fn without_order(mut self) -> Self {
        self.check_order = false;
        self
    }

    pub fn fate(&self, id: u32) -> Fate {
        self.fate.get(id as usize).copied().unwrap_or(Fate::Unseen)
    }

    /// Admitted and not yet taken by the consumer.
    pub fn outstanding(&self) -> u64 {
        self.outstanding
    }

    pub fn offer(&mut self, id: u32, tick: u64) {
        self.report.offered += 1;
        if let Some(slot) = self.fate.get_mut(id as usize) {
            *slot = Fate::Offered;
            self.arrival[id as usize] = tick;
        }
    }

    /// The queue accepted the offer.
    pub fn admitted(&mut self, id: u32) {
        match self.fate.get_mut(id as usize) {
            Some(slot) if *slot == Fate::Offered => {
                *slot = Fate::Queued;
                self.report.admitted += 1;
                self.outstanding += 1;
            }
            _ => self.report.phantom_deliveries += 1,
        }
    }

    /// The queue refused the offer and returned `returned`.
    pub fn refused(&mut self, id: u32, how: Refusal, returned: Seen) {
        if returned.id != id || self.expected.get(id as usize) != Some(&returned.digest) {
            self.report.wrong_returns += 1;
        }
        match self.fate.get_mut(id as usize) {
            Some(slot) if *slot == Fate::Offered => {
                *slot = match how {
                    Refusal::Full => Fate::RefusedFull,
                    Refusal::Closed => Fate::RefusedClosed,
                };
                match how {
                    Refusal::Full => self.report.refused_full += 1,
                    Refusal::Closed => self.report.refused_closed += 1,
                }
            }
            _ => self.report.phantom_deliveries += 1,
        }
    }

    /// The consumer took `seen` from the queue at `tick`; the wait in ticks, when it was a
    /// first delivery of a queued id.
    pub fn started(&mut self, seen: Seen, tick: u64) -> Option<u64> {
        let id = seen.id;
        if self.expected.get(id as usize) != Some(&seen.digest) {
            self.report.payload_mutations += 1;
        }
        if let Some(mark) = self.watermark.filter(|_| self.check_order) {
            if id < mark {
                self.report.fifo_violations += 1;
            }
        }
        self.watermark = Some(self.watermark.map_or(id, |mark| mark.max(id)));
        match self.fate(id) {
            Fate::Queued => {
                self.fate[id as usize] = Fate::InService;
                self.outstanding = self.outstanding.saturating_sub(1);
                Some(tick.saturating_sub(self.arrival[id as usize]))
            }
            Fate::InService | Fate::Delivered => {
                self.report.duplicate_deliveries += 1;
                None
            }
            Fate::Unseen | Fate::Offered | Fate::RefusedFull | Fate::RefusedClosed => {
                self.report.phantom_deliveries += 1;
                None
            }
        }
    }

    /// The consumer finished the message of `id`.
    pub fn finished(&mut self, id: u32) {
        match self.fate(id) {
            Fate::InService => {
                self.fate[id as usize] = Fate::Delivered;
                self.report.delivered += 1;
            }
            _ => self.report.spurious_completions += 1,
        }
    }

    /// A payload of `id` was dropped.
    pub fn dropped(&mut self, id: u32) {
        if matches!(
            self.fate(id),
            Fate::Offered | Fate::Queued | Fate::InService
        ) {
            self.report.dropped_while_owned += 1;
        }
    }

    /// Compare the live payload count with what the queue and the consumer should hold.
    pub fn observe_live(&mut self, expected: i64, actual: i64) {
        if expected != actual {
            self.report.live_mismatches += 1;
        }
    }

    /// Close the books: every offered id must be terminal and dropped exactly once, and
    /// nothing may be alive.
    pub fn finish(mut self, log: &DropLog) -> Report {
        for id in 0..self.fate.len() {
            let fate = self.fate[id];
            if fate == Fate::Unseen {
                continue;
            }
            if !fate.terminal() {
                self.report.silent_drops += 1;
            }
            if log.drops_of(id as u32) != 1 {
                self.report.drop_count_mismatches += 1;
            }
        }
        if log.live() != 0 {
            self.report.live_mismatches += 1;
        }
        self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests() -> Vec<u64> {
        vec![100, 101, 102, 103]
    }

    fn seen(id: u32) -> Seen {
        Seen {
            id,
            digest: 100 + u64::from(id),
        }
    }

    /// A log in which every id has been dropped exactly once.
    fn settled_log() -> std::sync::Arc<DropLog> {
        let log = DropLog::new(4);
        let corpus = super::super::payload::Corpus::generate(1, 4, 8);
        for id in 0..4 {
            drop(corpus.make(&log, id, 0));
        }
        log.take_recent();
        log
    }

    #[test]
    fn a_clean_run_has_every_id_in_one_terminal_state() {
        let mut ledger = Ledger::new(&digests());
        for id in 0..3 {
            ledger.offer(id, u64::from(id) * 10);
        }
        ledger.admitted(0);
        ledger.admitted(1);
        ledger.refused(2, Refusal::Full, seen(2));
        assert_eq!(ledger.outstanding(), 2);
        assert_eq!(ledger.started(seen(0), 5), Some(5));
        ledger.finished(0);
        assert_eq!(ledger.started(seen(1), 30), Some(20));
        ledger.finished(1);
        assert_eq!(ledger.outstanding(), 0);
        let log = settled_log();
        let report = ledger.finish(&log);
        assert_eq!(report.ownership_violations(), 0);
        assert_eq!(report.silent_drops, 0);
        assert_eq!(report.fifo_violations, 0);
        assert_eq!(
            (report.offered, report.admitted, report.delivered),
            (3, 2, 2)
        );
        assert_eq!((report.refused_full, report.refused_closed), (1, 0));
    }

    #[test]
    fn a_duplicate_delivery_is_an_ownership_violation() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.admitted(0);
        ledger.started(seen(0), 1);
        ledger.finished(0);
        assert_eq!(ledger.started(seen(0), 2), None);
        let report = ledger.finish(&settled_log());
        assert_eq!(report.duplicate_deliveries, 1);
        assert!(report.ownership_violations() >= 1);
    }

    #[test]
    fn a_duplicate_while_in_service_is_flagged_too() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.admitted(0);
        ledger.started(seen(0), 1);
        ledger.started(seen(0), 1);
        assert_eq!(ledger.finish(&settled_log()).duplicate_deliveries, 1);
    }

    #[test]
    fn an_altered_payload_is_a_mutation_whether_delivered_or_returned() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.admitted(0);
        ledger.started(Seen { id: 0, digest: 999 }, 1);
        ledger.offer(1, 1);
        ledger.refused(1, Refusal::Full, Seen { id: 1, digest: 999 });
        let report = ledger.finish(&settled_log());
        assert_eq!(report.payload_mutations, 1);
        assert_eq!(report.wrong_returns, 1);
    }

    #[test]
    fn a_refusal_that_returns_another_id_is_a_wrong_return() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(1, 0);
        ledger.refused(1, Refusal::Closed, seen(2));
        assert_eq!(ledger.finish(&settled_log()).wrong_returns, 1);
    }

    #[test]
    fn a_delivery_nobody_offered_or_that_was_refused_is_phantom() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.refused(0, Refusal::Full, seen(0));
        assert_eq!(ledger.started(seen(0), 1), None);
        assert_eq!(ledger.started(seen(3), 1), None);
        assert_eq!(ledger.finish(&settled_log()).phantom_deliveries, 2);
    }

    #[test]
    fn an_admitted_id_that_never_arrives_is_a_silent_drop() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.admitted(0);
        ledger.offer(1, 1);
        ledger.admitted(1);
        ledger.started(seen(1), 2);
        ledger.finished(1);
        let report = ledger.finish(&settled_log());
        assert_eq!(report.silent_drops, 1);
    }

    #[test]
    fn a_payload_dropped_while_queued_or_in_service_is_flagged_but_not_after_delivery() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.admitted(0);
        ledger.dropped(0);
        ledger.offer(1, 0);
        ledger.admitted(1);
        ledger.started(seen(1), 0);
        ledger.dropped(1);
        ledger.offer(2, 0);
        ledger.refused(2, Refusal::Full, seen(2));
        ledger.dropped(2);
        let report = ledger.finish(&settled_log());
        assert_eq!(report.dropped_while_owned, 2);
    }

    #[test]
    fn deliveries_out_of_arrival_order_are_fifo_violations() {
        let mut ledger = Ledger::new(&digests());
        for id in 0..3 {
            ledger.offer(id, 0);
            ledger.admitted(id);
        }
        ledger.started(seen(2), 1);
        ledger.finished(2);
        ledger.started(seen(0), 2);
        ledger.finished(0);
        ledger.started(seen(1), 3);
        ledger.finished(1);
        assert_eq!(ledger.finish(&settled_log()).fifo_violations, 2);
    }

    #[test]
    fn live_counts_completions_and_drop_counts_are_checked() {
        let mut ledger = Ledger::new(&digests());
        ledger.observe_live(3, 3);
        ledger.observe_live(3, 2);
        ledger.finished(1);
        ledger.offer(0, 0);
        let log = DropLog::new(4);
        let corpus = super::super::payload::Corpus::generate(1, 4, 8);
        let leaked = corpus.make(&log, 0, 0);
        let report = ledger.finish(&log);
        drop(leaked);
        assert_eq!(
            report.live_mismatches, 2,
            "one observed, one leak at the end"
        );
        assert_eq!(report.spurious_completions, 1);
        assert_eq!(
            report.drop_count_mismatches, 1,
            "id 0 was never dropped by then"
        );
        assert_eq!(report.silent_drops, 1, "offered and never settled");
    }

    #[test]
    fn a_double_drop_is_a_drop_count_mismatch() {
        let mut ledger = Ledger::new(&digests());
        ledger.offer(0, 0);
        ledger.refused(0, Refusal::Full, seen(0));
        let log = DropLog::new(4);
        let corpus = super::super::payload::Corpus::generate(1, 4, 8);
        let first = corpus.make(&log, 0, 0);
        let second = first.forge_copy();
        drop(first);
        drop(second);
        assert_eq!(ledger.finish(&log).drop_count_mismatches, 1);
    }
}
