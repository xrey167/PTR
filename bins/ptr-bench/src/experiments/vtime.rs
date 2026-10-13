//! A deterministic virtual-time event queue, shared by the discrete-event harnesses (R001 now,
//! S002 later).
//!
//! Time is an integer tick that only moves forward when an event is popped. Events are ordered
//! by `(tick, seq)`: `seq` is the order in which they were scheduled, so two events at one tick
//! fire in scheduling order and a run depends on nothing but the schedule it was given (no wall
//! clock, no hash order, no thread). An event cannot be scheduled before the current tick.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// An event that came off the queue.
#[derive(Debug, PartialEq, Eq)]
pub struct Fired<E> {
    pub tick: u64,
    pub seq: u64,
    pub event: E,
}

/// Scheduling an event at a tick the clock has already passed.
#[derive(Debug, PartialEq, Eq)]
pub struct InThePast {
    pub now: u64,
    pub tick: u64,
}

struct Entry<E> {
    tick: u64,
    seq: u64,
    event: E,
}

impl<E> PartialEq for Entry<E> {
    fn eq(&self, other: &Self) -> bool {
        (self.tick, self.seq) == (other.tick, other.seq)
    }
}
impl<E> Eq for Entry<E> {}
impl<E> PartialOrd for Entry<E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<E> Ord for Entry<E> {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.tick, self.seq).cmp(&(other.tick, other.seq))
    }
}

/// The queue: a min-heap on `(tick, seq)` and the virtual clock.
pub struct EventQueue<E> {
    heap: BinaryHeap<Reverse<Entry<E>>>,
    next_seq: u64,
    now: u64,
}

impl<E> Default for EventQueue<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> EventQueue<E> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
            now: 0,
        }
    }

    /// The tick of the event popped last (0 before the first).
    pub fn now(&self) -> u64 {
        self.now
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Schedule `event` at absolute `tick`; the returned `seq` is its tie-break rank.
    pub fn schedule(&mut self, tick: u64, event: E) -> Result<u64, InThePast> {
        if tick < self.now {
            return Err(InThePast {
                now: self.now,
                tick,
            });
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(Reverse(Entry { tick, seq, event }));
        Ok(seq)
    }

    /// Schedule `event` `delay` ticks after the current tick (never in the past).
    pub fn schedule_after(&mut self, delay: u64, event: E) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let tick = self.now.saturating_add(delay);
        self.heap.push(Reverse(Entry { tick, seq, event }));
        seq
    }

    /// The next event in `(tick, seq)` order; the clock moves to its tick.
    pub fn pop(&mut self) -> Option<Fired<E>> {
        let Reverse(entry) = self.heap.pop()?;
        self.now = entry.tick;
        Some(Fired {
            tick: entry.tick,
            seq: entry.seq,
            event: entry.event,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain<E>(queue: &mut EventQueue<E>) -> Vec<(u64, u64, E)> {
        std::iter::from_fn(|| queue.pop())
            .map(|fired| (fired.tick, fired.seq, fired.event))
            .collect()
    }

    #[test]
    fn events_fire_by_tick_and_ties_by_scheduling_order() {
        let mut queue = EventQueue::new();
        queue.schedule(5, "e").unwrap();
        queue.schedule(1, "a").unwrap();
        queue.schedule(5, "f").unwrap();
        queue.schedule(1, "b").unwrap();
        queue.schedule(3, "c").unwrap();
        let order: Vec<&str> = drain(&mut queue).into_iter().map(|e| e.2).collect();
        assert_eq!(order, ["a", "b", "c", "e", "f"]);
    }

    #[test]
    fn the_clock_only_moves_forward_and_the_past_is_refused() {
        let mut queue = EventQueue::new();
        queue.schedule(10, 1).unwrap();
        assert_eq!(queue.now(), 0);
        assert_eq!(queue.pop().map(|f| f.tick), Some(10));
        assert_eq!(queue.now(), 10);
        assert_eq!(queue.schedule(9, 2), Err(InThePast { now: 10, tick: 9 }));
        assert!(
            queue.schedule(10, 3).is_ok(),
            "the current tick is not the past"
        );
        assert!(queue.pop().is_some());
        assert!(queue.pop().is_none());
        assert_eq!(queue.now(), 10, "an empty pop leaves the clock");
    }

    #[test]
    fn schedule_after_is_relative_to_the_clock_and_ranks_after_earlier_events() {
        let mut queue = EventQueue::new();
        queue.schedule(4, "arrival").unwrap();
        queue.pop();
        let first = queue.schedule_after(3, "done");
        let second = queue.schedule(7, "arrival-same-tick").unwrap();
        assert!(first < second);
        let fired = drain(&mut queue);
        assert_eq!(fired[0].0, 7);
        assert_eq!(fired[0].2, "done");
        assert_eq!(fired[1].2, "arrival-same-tick");
    }

    #[test]
    fn the_same_schedule_pops_in_the_same_order_every_time() {
        let build = || {
            let mut queue = EventQueue::new();
            for n in 0..200u64 {
                queue.schedule((n * 7919) % 31, n).unwrap();
            }
            drain(&mut queue)
        };
        assert_eq!(build(), build());
        assert_eq!(build().len(), 200);
    }

    #[test]
    fn len_and_emptiness_follow_the_heap() {
        let mut queue = EventQueue::<u8>::default();
        assert!(queue.is_empty());
        queue.schedule(0, 1).unwrap();
        assert_eq!(queue.len(), 1);
        queue.pop();
        assert!(queue.is_empty());
    }
}
