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
use std::collections::{HashMap, VecDeque};

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

#[derive(Debug)]
pub(crate) struct BatchQueue<J> {
    policy: BatchPolicy,
    entries: VecDeque<Entry<J>>,
    next_seq: u64,
}

impl<J> BatchQueue<J> {
    pub(crate) fn new(policy: BatchPolicy) -> Self {
        Self {
            policy,
            entries: VecDeque::new(),
            next_seq: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
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
        if self.entries.len() >= self.policy.queue_capacity {
            return Err(QueueFull);
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        self.entries.push_back(Entry {
            seq,
            key,
            // Queue age is measured on the monotonic reading: a wall clock that steps cannot
            // stretch or shrink max_wait.
            arrived: now.mono,
            deadline,
            cancel,
            payload,
        });
        Ok(())
    }

    /// Remove every entry (the queue is being closed).
    pub(crate) fn drain(&mut self) -> Vec<Entry<J>> {
        self.entries.drain(..).collect()
    }

    /// Drop what must not reach the model, then take the batch that is ready at `now`, if any.
    pub(crate) fn take_ready(&mut self, now: impl Into<Now>) -> Ready<J> {
        let now = now.into();
        let mut dropped = Vec::new();
        let mut kept = VecDeque::with_capacity(self.entries.len());
        for entry in self.entries.drain(..) {
            if entry.cancel.as_ref().is_some_and(CancelFlag::is_cancelled) {
                dropped.push((Dropped::Cancelled, entry));
            } else if entry.deadline.is_some_and(|deadline| deadline <= now.wall) {
                dropped.push((Dropped::DeadlinePassed, entry));
            } else {
                kept.push_back(entry);
            }
        }
        self.entries = kept;

        // One pass over the queue builds a group per distinct key, in order of each key's oldest
        // request; the cost is linear in the queue however many distinct keys it holds.
        // Per group: the requests and the milliseconds until it is due. Waiting is judged on the
        // monotonic reading, deadlines on the wall one; each side is turned into a distance
        // from now, so the two never meet as absolute times.
        let mut groups: Vec<(&BatchKey, usize, u64)> = Vec::new();
        let mut index_of: HashMap<&BatchKey, usize> = HashMap::new();
        for entry in &self.entries {
            let index = *index_of.entry(&entry.key).or_insert_with(|| {
                groups.push((&entry.key, 0, u64::MAX));
                groups.len() - 1
            });
            let group = &mut groups[index];
            group.1 += 1;
            group.2 = group.2.min(
                entry
                    .arrived
                    .saturating_add(self.policy.max_wait_millis)
                    .saturating_sub(now.mono),
            );
            if let Some(deadline) = entry.deadline {
                group.2 = group.2.min(
                    deadline
                        .saturating_sub(self.policy.deadline_margin_millis)
                        .saturating_sub(now.wall),
                );
            }
        }
        let mut chosen: Option<BatchKey> = None;
        let mut wake_after: Option<u64> = None;
        for (key, count, due_in) in groups {
            if count >= self.policy.max_batch || due_in == 0 {
                // Groups are visited oldest first, so the first ready one is the one to serve.
                chosen = Some(key.clone());
                break;
            }
            wake_after = Some(wake_after.map_or(due_in, |at| at.min(due_in)));
        }

        let batch = chosen.map(|key| {
            let mut batch = Vec::new();
            let mut rest = VecDeque::with_capacity(self.entries.len());
            for entry in self.entries.drain(..) {
                if entry.key == key && batch.len() < self.policy.max_batch {
                    batch.push(entry);
                } else {
                    rest.push_back(entry);
                }
            }
            self.entries = rest;
            batch
        });
        // After a dispatch the remainder is looked at again straight away.
        let wake_after = if batch.is_some() && !self.entries.is_empty() {
            Some(0)
        } else {
            wake_after
        };
        // A deadline inside the queue is itself a time to look again, so it is dropped on time.
        let wake_after = self.entries.iter().filter_map(|entry| entry.deadline).fold(
            wake_after,
            |at, deadline| {
                let in_ms = deadline.saturating_sub(now.wall);
                Some(at.map_or(in_ms, |at| at.min(in_ms)))
            },
        );
        debug_assert!(self.entries.iter().all(|entry| entry.seq < self.next_seq));
        Ready {
            dropped,
            batch,
            wake_after,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::Generation;

    fn policy(max_batch: usize, max_wait: u64, capacity: usize) -> BatchPolicy {
        BatchPolicy {
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
    fn a_full_queue_refuses_before_it_grows() {
        let mut queue = BatchQueue::new(policy(4, 10, 2));
        assert!(queue.push(key("a"), None, None, 1u32, 0).is_ok());
        assert!(queue.push(key("a"), None, None, 2, 0).is_ok());
        assert_eq!(queue.push(key("a"), None, None, 3, 0), Err(QueueFull));
        assert_eq!(queue.len(), 2);
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
