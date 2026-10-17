//! The deterministic core of the adaptive batcher: a bounded queue and the rule that decides
//! which requests leave it together and when.
//!
//! Nothing here reads a clock or starts a thread. The caller passes `now` in milliseconds, so
//! the rule can be tested with a virtual clock and measured without timing noise.
//!
//! **Rules.**
//! * The queue is bounded; a push into a full queue is refused before anything grows
//!   (Invariant 8).
//! * Only requests with equal [`BatchKey`] leave together.
//! * A request whose deadline has passed, or that was cancelled, leaves the queue unserved
//!   before any model call.
//! * A group is ready when it holds `max_batch` requests, when its oldest request has waited
//!   `max_wait`, or when its tightest deadline is within `deadline_margin` of now. The ready
//!   group whose oldest request arrived first goes first, whatever its key.

use crate::batch::{BatchKey, BatchPolicy, CancelFlag};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One queued request, with the payload the caller wants back.
#[derive(Debug)]
pub(crate) struct Entry<J> {
    seq: u64,
    pub(crate) key: BatchKey,
    arrived: u64,
    deadline: Option<u64>,
    cancel: Option<CancelFlag>,
    pub(crate) payload: J,
}

impl<J> Entry<J> {
    /// Whether this request must not reach the model at `now`: it was cancelled, or its
    /// deadline has passed. Asked again right before the model call, after the gate.
    pub(crate) fn stale_at(&self, now: u64) -> Option<Dropped> {
        if self.cancel.as_ref().is_some_and(CancelFlag::is_cancelled) {
            Some(Dropped::Cancelled)
        } else if self.deadline.is_some_and(|deadline| deadline <= now) {
            Some(Dropped::DeadlinePassed)
        } else {
            None
        }
    }
}

/// Why a request left the queue without being served.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Dropped {
    DeadlinePassed,
    Cancelled,
}

/// The queue was full.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QueueFull;

/// What a call to [`BatchQueue::take_ready`] found.
#[derive(Debug)]
pub(crate) struct Ready<J> {
    pub(crate) dropped: Vec<(Dropped, Entry<J>)>,
    pub(crate) batch: Option<Vec<Entry<J>>>,
    /// In how many milliseconds the queue should be looked at again if nothing arrives sooner;
    /// `None` when it is empty.
    pub(crate) wake_after: Option<u64>,
}

/// The two readings the queue is given. `wall` is the batcher clock (Unix milliseconds for
/// the system clock): absolute deadlines are compared with it. `mono` is a monotonic reading:
/// queue age (`max_wait`) is measured on it, so a wall clock that steps cannot stretch or
/// shrink a wait. A virtual clock gives both the same value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Now {
    pub(crate) wall: u64,
    pub(crate) mono: u64,
}

impl From<u64> for Now {
    fn from(time: u64) -> Self {
        Self {
            wall: time,
            mono: time,
        }
    }
}

/// One key's requests, oldest first, and where the group is registered in the schedules.
#[derive(Debug)]
struct Group<J> {
    key: BatchKey,
    entries: BTreeMap<u64, Entry<J>>,
    /// The deadlines of the entries, counted, so the tightest is the first key.
    deadlines: BTreeMap<u64, usize>,
    registered: Registered,
}

/// The schedule entries a group currently holds, so they can be removed exactly.
#[derive(Debug, Default)]
struct Registered {
    wait: Option<(u64, u64)>,
    deadline: Option<(u64, u64)>,
    ready: Option<u64>,
}

/// The bounded queue.
///
/// Readiness is kept incrementally: every key has a group, and each group sits in ordered
/// schedules (when its oldest request has waited out, when its tightest deadline is within the
/// margin, and the set of groups that are ready now, by oldest request). A push and a
/// `take_ready` touch the groups they affect and a logarithmic number of schedule entries; they
/// never walk the whole queue, however many keys it holds.
#[derive(Debug)]
pub(crate) struct BatchQueue<J> {
    policy: BatchPolicy,
    next_seq: u64,
    len: usize,
    /// Queued requests per principal (`None` is the anonymous bucket).
    per_principal: HashMap<Option<String>, usize>,
    ids: HashMap<BatchKey, u32>,
    groups: HashMap<u32, Group<J>>,
    next_gid: u32,
    /// (arrived + max_wait of the oldest request, its seq, group): monotonic reading.
    wait_due: BTreeSet<(u64, u64, u32)>,
    /// (tightest deadline - margin, oldest seq, group): wall reading.
    deadline_due: BTreeSet<(u64, u64, u32)>,
    /// (deadline, seq, group) of every request that has one: wall reading.
    expiry: BTreeSet<(u64, u64, u32)>,
    /// (oldest seq, group) of the groups that are ready now.
    ready: BTreeSet<(u64, u32)>,
    /// Schedule operations and entry visits, for the cost test.
    ops: u64,
}

impl<J> BatchQueue<J> {
    pub(crate) fn new(policy: BatchPolicy) -> Self {
        Self {
            policy,
            next_seq: 0,
            len: 0,
            per_principal: HashMap::new(),
            ids: HashMap::new(),
            groups: HashMap::new(),
            next_gid: 0,
            wait_due: BTreeSet::new(),
            deadline_due: BTreeSet::new(),
            expiry: BTreeSet::new(),
            ready: BTreeSet::new(),
            ops: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// How many schedule operations and entry visits the queue has done.
    #[cfg(test)]
    pub(crate) fn ops(&self) -> u64 {
        self.ops
    }

    /// Remove every schedule entry of a group.
    fn unregister(&mut self, gid: u32) {
        let Some(group) = self.groups.get_mut(&gid) else {
            return;
        };
        let registered = std::mem::take(&mut group.registered);
        if let Some((due, seq)) = registered.wait {
            self.wait_due.remove(&(due, seq, gid));
            self.ops += 1;
        }
        if let Some((due, seq)) = registered.deadline {
            self.deadline_due.remove(&(due, seq, gid));
            self.ops += 1;
        }
        if let Some(seq) = registered.ready {
            self.ready.remove(&(seq, gid));
            self.ops += 1;
        }
    }

    /// Enter a group in the schedules from its current entries; an empty group is dropped.
    fn register(&mut self, gid: u32) {
        let Some(group) = self.groups.get_mut(&gid) else {
            return;
        };
        let Some((&first_seq, first)) = group.entries.iter().next() else {
            let key = group.key.clone();
            self.groups.remove(&gid);
            self.ids.remove(&key);
            self.ops += 1;
            return;
        };
        let wait = (
            first.arrived.saturating_add(self.policy.max_wait_millis),
            first_seq,
        );
        let deadline = group.deadlines.keys().next().map(|tightest| {
            (
                tightest.saturating_sub(self.policy.deadline_margin_millis),
                first_seq,
            )
        });
        group.registered.wait = Some(wait);
        group.registered.deadline = deadline;
        self.wait_due.insert((wait.0, wait.1, gid));
        if let Some((due, seq)) = deadline {
            self.deadline_due.insert((due, seq, gid));
            self.ops += 1;
        }
        if group.entries.len() >= self.policy.max_batch {
            group.registered.ready = Some(first_seq);
            self.ready.insert((first_seq, gid));
            self.ops += 1;
        }
        self.ops += 1;
    }

    /// Whether a [`BatchQueue::push`] under `key` would be refused: the queue is full, or the
    /// key's principal is at its own limit.
    pub(crate) fn is_full_for(&self, key: &BatchKey) -> bool {
        self.len >= self.policy.queue_capacity
            || self.per_principal.get(&key.principal).copied().unwrap_or(0)
                >= self.policy.max_queued_per_principal
    }

    /// Remove every request whose cancel flag is set and return it, oldest first, so the caller
    /// can answer it. A cancelled request would otherwise hold a slot until its group is served
    /// or its deadline passes, and a full queue would refuse live requests meanwhile. Linear in
    /// the queue; asked only when the queue is full.
    pub(crate) fn reclaim_cancelled(&mut self) -> Vec<Entry<J>> {
        let cancelled: Vec<(u32, u64)> = self
            .groups
            .iter()
            .flat_map(|(gid, group)| {
                group
                    .entries
                    .iter()
                    .filter(|(_, entry)| {
                        entry.cancel.as_ref().is_some_and(CancelFlag::is_cancelled)
                    })
                    .map(|(seq, _)| (*gid, *seq))
            })
            .collect();
        let mut reclaimed = Vec::with_capacity(cancelled.len());
        let mut touched: Vec<u32> = cancelled.iter().map(|(gid, _)| *gid).collect();
        touched.sort_unstable();
        touched.dedup();
        for gid in &touched {
            self.unregister(*gid);
        }
        for (gid, seq) in cancelled {
            reclaimed.extend(self.remove_entry(gid, seq));
        }
        for gid in touched {
            self.register(gid);
        }
        reclaimed.sort_by_key(|entry| entry.seq);
        reclaimed
    }

    /// Queue one request. The bound is checked before the entry is added.
    pub(crate) fn push(
        &mut self,
        key: BatchKey,
        deadline: Option<u64>,
        cancel: Option<CancelFlag>,
        payload: J,
        now: impl Into<Now>,
    ) -> Result<(), QueueFull> {
        let now = now.into();
        if self.is_full_for(&key) {
            return Err(QueueFull);
        }
        *self.per_principal.entry(key.principal.clone()).or_insert(0) += 1;
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        let gid = match self.ids.get(&key) {
            Some(gid) => *gid,
            None => {
                let gid = self.next_gid;
                self.next_gid = self.next_gid.wrapping_add(1);
                self.ids.insert(key.clone(), gid);
                self.groups.insert(
                    gid,
                    Group {
                        key: key.clone(),
                        entries: BTreeMap::new(),
                        deadlines: BTreeMap::new(),
                        registered: Registered::default(),
                    },
                );
                gid
            }
        };
        self.unregister(gid);
        if let Some(group) = self.groups.get_mut(&gid) {
            group.entries.insert(
                seq,
                Entry {
                    seq,
                    key,
                    // Queue age is measured on the monotonic reading: a wall clock that steps
                    // cannot stretch or shrink max_wait.
                    arrived: now.mono,
                    deadline,
                    cancel,
                    payload,
                },
            );
            if let Some(deadline) = deadline {
                *group.deadlines.entry(deadline).or_insert(0) += 1;
                self.expiry.insert((deadline, seq, gid));
                self.ops += 1;
            }
        }
        self.len += 1;
        self.register(gid);
        Ok(())
    }

    /// Remove every entry (the queue is being closed), oldest first.
    pub(crate) fn drain(&mut self) -> Vec<Entry<J>> {
        let mut all: Vec<Entry<J>> = self
            .groups
            .drain()
            .flat_map(|(_, group)| group.entries.into_values())
            .collect();
        all.sort_by_key(|entry| entry.seq);
        self.per_principal.clear();
        self.ids.clear();
        self.wait_due.clear();
        self.deadline_due.clear();
        self.expiry.clear();
        self.ready.clear();
        self.len = 0;
        all
    }

    /// Take one entry out of its group, keeping the deadline books.
    fn remove_entry(&mut self, gid: u32, seq: u64) -> Option<Entry<J>> {
        let group = self.groups.get_mut(&gid)?;
        let entry = group.entries.remove(&seq)?;
        if let Some(deadline) = entry.deadline {
            if let Some(count) = group.deadlines.get_mut(&deadline) {
                *count -= 1;
                if *count == 0 {
                    group.deadlines.remove(&deadline);
                }
            }
            self.expiry.remove(&(deadline, seq, gid));
            self.ops += 1;
        }
        self.len -= 1;
        if let Some(count) = self.per_principal.get_mut(&entry.key.principal) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.per_principal.remove(&entry.key.principal);
            }
        }
        self.ops += 1;
        Some(entry)
    }

    /// Drop what must not reach the model, then take the batch that is ready at `now`, if any.
    pub(crate) fn take_ready(&mut self, now: impl Into<Now>) -> Ready<J> {
        let now = now.into();
        let mut dropped = Vec::new();

        // Requests whose deadline has passed leave, tightest first.
        while let Some(&(deadline, seq, gid)) = self.expiry.iter().next() {
            if deadline > now.wall {
                break;
            }
            self.unregister(gid);
            if let Some(entry) = self.remove_entry(gid, seq) {
                dropped.push((Dropped::DeadlinePassed, entry));
            }
            self.register(gid);
        }

        // Groups that have waited out, or whose tightest deadline is within the margin, are
        // ready; each moves into the ready set once.
        while let Some(&(due, seq, gid)) = self.wait_due.iter().next() {
            if due > now.mono {
                break;
            }
            self.promote(gid, seq);
        }
        while let Some(&(due, seq, gid)) = self.deadline_due.iter().next() {
            if due > now.wall {
                break;
            }
            self.promote(gid, seq);
        }

        // The ready group whose oldest request arrived first goes first. Cancelled requests of
        // that group leave with it, unserved.
        let mut batch = None;
        while batch.is_none() {
            let Some(&(_, gid)) = self.ready.iter().next() else {
                break;
            };
            self.unregister(gid);
            let mut taken = Vec::new();
            while taken.len() < self.policy.max_batch {
                let Some(&seq) = self
                    .groups
                    .get(&gid)
                    .and_then(|group| group.entries.keys().next())
                else {
                    break;
                };
                self.ops += 1;
                let Some(entry) = self.remove_entry(gid, seq) else {
                    break;
                };
                if entry.cancel.as_ref().is_some_and(CancelFlag::is_cancelled) {
                    dropped.push((Dropped::Cancelled, entry));
                } else {
                    taken.push(entry);
                }
            }
            self.register(gid);
            if !taken.is_empty() {
                batch = Some(taken);
            }
        }

        // When to look again: after a dispatch the remainder straight away, otherwise at the
        // nearest of the three schedules, each as a distance from its own reading.
        let mut wake_after = if batch.is_some() && self.len > 0 {
            Some(0)
        } else {
            None
        };
        let mut sooner = |distance: u64| {
            wake_after = Some(wake_after.map_or(distance, |at: u64| at.min(distance)));
        };
        if let Some(&(due, _, _)) = self.wait_due.iter().next() {
            sooner(due.saturating_sub(now.mono));
        }
        if let Some(&(due, _, _)) = self.deadline_due.iter().next() {
            sooner(due.saturating_sub(now.wall));
        }
        if let Some(&(deadline, _, _)) = self.expiry.iter().next() {
            sooner(deadline.saturating_sub(now.wall));
        }
        Ready {
            dropped,
            batch,
            wake_after,
        }
    }

    /// Move a group whose schedule entry came due into the ready set.
    fn promote(&mut self, gid: u32, seq: u64) {
        self.unregister(gid);
        if let Some(group) = self.groups.get_mut(&gid) {
            group.registered.ready = Some(seq);
            self.ready.insert((seq, gid));
            self.ops += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::Generation;

    fn policy(max_batch: usize, max_wait: u64, capacity: usize) -> BatchPolicy {
        BatchPolicy {
            max_queued_per_principal: capacity,
            max_batch,
            max_wait_millis: max_wait,
            queue_capacity: capacity,
            deadline_margin_millis: 5,
        }
    }

    fn key(adapter: &str) -> BatchKey {
        BatchKey {
            project: None,
            adapter: Some(adapter.into()),
            principal: None,
            generation: Generation(1),
        }
    }

    fn ids(batch: &[Entry<u32>]) -> Vec<u32> {
        batch.iter().map(|entry| entry.payload).collect()
    }

    #[test]
    fn a_queue_of_many_distinct_keys_is_scanned_in_linear_time() {
        let capacity = 60_000;
        let mut queue = BatchQueue::new(policy(8, 1_000, capacity));
        for n in 0..capacity {
            queue
                .push(key(&format!("adapter-{n}")), None, None, n as u32, 0)
                .unwrap();
        }
        let started = std::time::Instant::now();
        // Nothing is ready yet (one request per key, waiting), then everything has waited out.
        assert!(queue.take_ready(1).batch.is_none());
        assert!(queue.take_ready(1_000).batch.is_some());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "two scans of {capacity} distinct keys took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_push_and_a_take_cost_a_logarithmic_number_of_operations_in_the_queue_size() {
        let capacity = 65_536;
        let mut queue = BatchQueue::new(policy(8, 1_000, capacity));
        for n in 0..capacity {
            // One arrival on a fresh key, then the worker looks at the queue, as it does for
            // every wake-up.
            queue
                .push(
                    key(&format!("adapter-{n}")),
                    None,
                    None,
                    n as u32,
                    n as u64 / 64,
                )
                .unwrap();
            assert!(queue.take_ready(0).batch.is_none());
        }
        // Per arrival a handful of schedule operations, not a walk of the queue: well under
        // 64 per arrival (a full scan would be tens of thousands).
        assert!(
            queue.ops() < 64 * capacity as u64,
            "{} operations for {capacity} arrivals",
            queue.ops()
        );
        // Everything has waited out: one request leaves per call. Becoming ready is paid once per
        // group, so the whole drain stays within a bounded cost per request too.
        let before = queue.ops();
        let mut served = 0;
        while queue.len() > 0 {
            assert!(queue.take_ready(10_000).batch.is_some());
            served += 1;
        }
        assert_eq!(served, capacity);
        assert!(queue.ops() - before < 64 * capacity as u64);
    }

    #[test]
    fn a_full_queue_refuses_before_it_grows() {
        let mut queue = BatchQueue::new(policy(4, 10, 2));
        assert!(queue.push(key("a"), None, None, 1u32, 0).is_ok());
        assert!(queue.push(key("a"), None, None, 2, 0).is_ok());
        assert_eq!(queue.push(key("a"), None, None, 3, 0), Err(QueueFull));
        assert_eq!(queue.len(), 2);
    }

    fn keyed(principal: Option<&str>) -> BatchKey {
        BatchKey {
            principal: principal.map(str::to_owned),
            ..key("a")
        }
    }

    #[test]
    fn a_principal_over_its_limit_is_refused_and_the_others_are_not() {
        let mut queue = BatchQueue::new(BatchPolicy {
            max_queued_per_principal: 2,
            ..policy(8, 10, 6)
        });
        assert!(queue.push(keyed(Some("a")), None, None, 1u32, 0).is_ok());
        assert!(queue.push(keyed(Some("a")), None, None, 2, 0).is_ok());
        assert_eq!(
            queue.push(keyed(Some("a")), None, None, 3, 0),
            Err(QueueFull)
        );
        assert!(queue.push(keyed(Some("b")), None, None, 4, 0).is_ok());
        // The anonymous bucket is limited too, and separate from the named ones.
        assert!(queue.push(keyed(None), None, None, 5, 0).is_ok());
        assert!(queue.push(keyed(None), None, None, 6, 0).is_ok());
        assert_eq!(queue.push(keyed(None), None, None, 7, 0), Err(QueueFull));
        assert_eq!(queue.len(), 5);
    }

    #[test]
    fn dequeueing_cancelling_and_draining_free_a_principals_quota() {
        let mut queue = BatchQueue::new(BatchPolicy {
            max_queued_per_principal: 2,
            ..policy(1, 0, 8)
        });
        let flag = CancelFlag::default();
        assert!(queue
            .push(keyed(Some("a")), None, Some(flag.clone()), 1u32, 0)
            .is_ok());
        assert!(queue.push(keyed(Some("a")), None, None, 2, 0).is_ok());
        assert!(queue.is_full_for(&keyed(Some("a"))));
        // A cancelled entry is reclaimed and its quota is free again.
        flag.cancel();
        assert_eq!(queue.reclaim_cancelled().len(), 1);
        assert!(!queue.is_full_for(&keyed(Some("a"))));
        // A dispatched entry frees its quota as well.
        assert!(queue.push(keyed(Some("a")), None, None, 3, 0).is_ok());
        assert!(queue.is_full_for(&keyed(Some("a"))));
        assert!(queue.take_ready(0).batch.is_some());
        assert!(!queue.is_full_for(&keyed(Some("a"))));
        // And a drain empties every count.
        assert!(queue.push(keyed(Some("a")), None, None, 4, 0).is_ok());
        assert!(queue.is_full_for(&keyed(Some("a"))));
        assert_eq!(queue.drain().len(), 2);
        assert!(!queue.is_full_for(&keyed(Some("a"))));
    }

    #[test]
    fn a_batch_waits_for_max_wait_unless_it_is_full() {
        let mut queue = BatchQueue::new(policy(2, 10, 8));
        queue.push(key("a"), None, None, 1u32, 0).unwrap();
        let early = queue.take_ready(3);
        assert!(early.batch.is_none());
        assert_eq!(early.wake_after, Some(7));
        queue.push(key("a"), None, None, 2, 4).unwrap();
        let full = queue.take_ready(4);
        assert_eq!(ids(&full.batch.expect("full batch")), vec![1, 2]);
    }

    #[test]
    fn the_wait_expires_for_a_partial_batch() {
        let mut queue = BatchQueue::new(policy(8, 10, 8));
        queue.push(key("a"), None, None, 1u32, 0).unwrap();
        let ready = queue.take_ready(10);
        assert_eq!(ids(&ready.batch.expect("waited out")), vec![1]);
    }

    #[test]
    fn requests_with_other_keys_never_share_a_batch() {
        let mut queue = BatchQueue::new(policy(8, 0, 8));
        queue.push(key("a"), None, None, 1u32, 0).unwrap();
        queue.push(key("b"), None, None, 2, 0).unwrap();
        queue.push(key("a"), None, None, 3, 0).unwrap();
        let first = queue.take_ready(0);
        assert_eq!(ids(&first.batch.expect("a")), vec![1, 3]);
        let second = queue.take_ready(0);
        assert_eq!(ids(&second.batch.expect("b")), vec![2]);
    }

    #[test]
    fn the_key_of_the_oldest_ready_request_is_served_first() {
        let mut queue = BatchQueue::new(policy(2, 100, 8));
        queue.push(key("slow"), None, None, 1u32, 0).unwrap();
        queue.push(key("fast"), None, None, 2, 1).unwrap();
        queue.push(key("fast"), None, None, 3, 2).unwrap();
        // "fast" is full and ready; "slow" is older but not ready.
        let ready = queue.take_ready(2);
        assert_eq!(ids(&ready.batch.expect("fast is ready")), vec![2, 3]);
        // Both full: the older one wins.
        queue.push(key("slow"), None, None, 4, 3).unwrap();
        queue.push(key("fast"), None, None, 5, 3).unwrap();
        queue.push(key("fast"), None, None, 6, 3).unwrap();
        let ready = queue.take_ready(3);
        assert_eq!(ids(&ready.batch.expect("slow is older")), vec![1, 4]);
    }

    #[test]
    fn a_tight_deadline_dispatches_early() {
        let mut queue = BatchQueue::new(policy(8, 100, 8));
        queue.push(key("a"), Some(30), None, 1u32, 0).unwrap();
        assert!(queue.take_ready(24).batch.is_none());
        // 30 - margin 5 = 25.
        assert_eq!(ids(&queue.take_ready(25).batch.expect("early")), vec![1]);
    }

    #[test]
    fn an_expired_or_cancelled_request_never_reaches_the_batch() {
        let mut queue = BatchQueue::new(policy(8, 0, 8));
        let cancel = CancelFlag::default();
        queue.push(key("a"), Some(10), None, 1u32, 0).unwrap();
        queue
            .push(key("a"), None, Some(cancel.clone()), 2, 0)
            .unwrap();
        queue.push(key("a"), None, None, 3, 0).unwrap();
        cancel.cancel();
        let ready = queue.take_ready(10);
        let dropped: Vec<(Dropped, u32)> = ready
            .dropped
            .iter()
            .map(|(why, entry)| (*why, entry.payload))
            .collect();
        assert_eq!(
            dropped,
            vec![(Dropped::DeadlinePassed, 1), (Dropped::Cancelled, 2)]
        );
        assert_eq!(ids(&ready.batch.expect("the live one")), vec![3]);
    }

    #[test]
    fn a_batch_never_exceeds_max_batch_and_the_rest_wakes_at_once() {
        let mut queue = BatchQueue::new(policy(2, 0, 8));
        for payload in 0..5u32 {
            queue.push(key("a"), None, None, payload, 0).unwrap();
        }
        let ready = queue.take_ready(0);
        assert_eq!(ids(&ready.batch.expect("batch")), vec![0, 1]);
        assert_eq!(ready.wake_after, Some(0));
        assert_eq!(queue.len(), 3);
    }
}
