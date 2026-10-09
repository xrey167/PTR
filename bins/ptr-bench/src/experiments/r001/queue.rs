//! The queues the simulation drives, behind one small trait.
//!
//! - [`MailboxQueue`]: arm B, the real `ptr_exec::Mailbox`, offered to through the mailbox and
//!   through a cloned sender alternately.
//! - [`UnboundedQueue`]: arm U, an unbounded `std::sync::mpsc` channel.
//! - [`SyncChannelQueue`]: arm S, a bounded `std::sync::mpsc::sync_channel`, the channel the
//!   mailbox once wrapped: an independent second oracle for Full, Closed and FIFO.
//! - [`ReferenceQueue`]: arm R, a `VecDeque` with an explicit capacity (see [`RefModel`]); it
//!   shares no code with the mailbox.
//! - [`Faulty`] and friends: deliberately broken queues for the negative controls and the
//!   canaries, which show that the oracle can tell a broken queue from a sound one.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::mpsc;

use ptr_exec::{Mailbox, MailboxSender, Receive, SendError};

use super::guard::contained;
use super::payload::Tracked;

/// The answer to an offer; a refusal hands the value back.
#[derive(Debug, PartialEq, Eq)]
pub enum Offer<T> {
    Admitted,
    Full(T),
    Closed(T),
    /// The product panicked inside the call (the value is gone with the unwinding).
    Panicked,
}

/// The answer to a non-blocking take.
#[derive(Debug, PartialEq, Eq)]
pub enum Take<T> {
    Message(T),
    Empty,
    Closed,
    /// The product panicked inside the call.
    Panicked,
}

pub trait Queue {
    fn offer(&mut self, payload: Tracked) -> Offer<Tracked>;
    fn take(&mut self) -> Take<Tracked>;
    /// The queue's own account of how many messages it holds.
    fn depth(&self) -> usize;
    fn close(&mut self);
    /// The largest depth the queue itself recorded, for the queues that record one.
    fn peak_depth(&self) -> Option<usize> {
        None
    }
    /// Panics of the product contained so far (only the real mailbox can panic).
    fn panics(&self) -> u64 {
        0
    }
}

/// The reference model: a `VecDeque` with an explicit capacity, written without the mailbox.
/// A closed model refuses with `Closed` before it looks at the capacity, and keeps its
/// queued messages for the consumer. `lifo` makes a deliberately wrong variant.
pub struct RefModel<T> {
    capacity: usize,
    closed: bool,
    lifo: bool,
    items: VecDeque<T>,
}

impl<T> RefModel<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            closed: false,
            lifo: false,
            items: VecDeque::new(),
        }
    }

    pub fn offer(&mut self, value: T) -> Offer<T> {
        if self.closed {
            Offer::Closed(value)
        } else if self.items.len() >= self.capacity {
            Offer::Full(value)
        } else {
            self.items.push_back(value);
            Offer::Admitted
        }
    }

    pub fn take(&mut self) -> Take<T> {
        let next = if self.lifo {
            self.items.pop_back()
        } else {
            self.items.pop_front()
        };
        match next {
            Some(value) => Take::Message(value),
            None if self.closed => Take::Closed,
            None => Take::Empty,
        }
    }

    pub fn depth(&self) -> usize {
        self.items.len()
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

/// How the harness closes the mailbox: through its owner or through a sender.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CloseVia {
    Owner,
    Sender,
}

/// Arm B: the real mailbox. Every call into it is contained: a panic is counted, not raised.
pub struct MailboxQueue {
    mailbox: Mailbox<Tracked>,
    sender: MailboxSender<Tracked>,
    offers: u64,
    close_via: CloseVia,
    panics: Cell<u64>,
}

impl MailboxQueue {
    pub fn new(capacity: usize, close_via: CloseVia) -> Self {
        let mailbox = Mailbox::bounded(capacity);
        let sender = mailbox.sender();
        Self {
            mailbox,
            sender,
            offers: 0,
            close_via,
            panics: Cell::new(0),
        }
    }

    fn contain<R>(&self, call: impl FnOnce() -> R) -> Option<R> {
        let result = contained(call);
        if result.is_none() {
            self.panics.set(self.panics.get() + 1);
        }
        result
    }
}

impl Queue for MailboxQueue {
    fn offer(&mut self, payload: Tracked) -> Offer<Tracked> {
        self.offers += 1;
        let through_owner = self.offers.is_multiple_of(2);
        let sent = self.contain(|| {
            if through_owner {
                self.mailbox.try_send(payload)
            } else {
                self.sender.try_send(payload)
            }
        });
        match sent {
            Some(Ok(())) => Offer::Admitted,
            Some(Err(SendError::Full(value))) => Offer::Full(value),
            Some(Err(SendError::Closed(value))) => Offer::Closed(value),
            None => Offer::Panicked,
        }
    }

    fn take(&mut self) -> Take<Tracked> {
        match self.contain(|| self.mailbox.try_receive()) {
            Some(Receive::Message(value)) => Take::Message(value),
            Some(Receive::Empty) => Take::Empty,
            Some(Receive::Closed) => Take::Closed,
            None => Take::Panicked,
        }
    }

    fn depth(&self) -> usize {
        self.contain(|| self.mailbox.depth()).unwrap_or(0)
    }

    fn close(&mut self) {
        let via = self.close_via;
        let _ = self.contain(|| match via {
            CloseVia::Owner => self.mailbox.close(),
            CloseVia::Sender => self.sender.close(),
        });
    }

    fn peak_depth(&self) -> Option<usize> {
        self.contain(|| self.mailbox.peak_depth())
    }

    fn panics(&self) -> u64 {
        self.panics.get()
    }
}

/// Arm S: a bounded `std::sync::mpsc::sync_channel`, written to the spec by wrapping a channel
/// that shares no code with the mailbox. A channel has no "refuse sends but keep the queue
/// receivable" state, so `close()` here drops the sending half (the receiver drains the buffer
/// and then reports the disconnection) and the wrapper refuses later sends as `Closed`. Its
/// depth is the harness's own count of sends minus takes.
pub struct SyncChannelQueue {
    sender: Option<mpsc::SyncSender<Tracked>>,
    receiver: mpsc::Receiver<Tracked>,
    depth: usize,
    peak: usize,
}

impl SyncChannelQueue {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        Self {
            sender: Some(sender),
            receiver,
            depth: 0,
            peak: 0,
        }
    }
}

impl Queue for SyncChannelQueue {
    fn offer(&mut self, payload: Tracked) -> Offer<Tracked> {
        let Some(sender) = &self.sender else {
            return Offer::Closed(payload);
        };
        match sender.try_send(payload) {
            Ok(()) => {
                self.depth += 1;
                self.peak = self.peak.max(self.depth);
                Offer::Admitted
            }
            Err(mpsc::TrySendError::Full(value)) => Offer::Full(value),
            Err(mpsc::TrySendError::Disconnected(value)) => Offer::Closed(value),
        }
    }

    fn take(&mut self) -> Take<Tracked> {
        match self.receiver.try_recv() {
            Ok(value) => {
                self.depth -= 1;
                Take::Message(value)
            }
            Err(mpsc::TryRecvError::Empty) => Take::Empty,
            Err(mpsc::TryRecvError::Disconnected) => Take::Closed,
        }
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn close(&mut self) {
        self.sender = None;
    }

    fn peak_depth(&self) -> Option<usize> {
        Some(self.peak)
    }
}

/// Arm U: an unbounded channel. Its depth is the harness's own count of sends minus takes.
pub struct UnboundedQueue {
    sender: Option<mpsc::Sender<Tracked>>,
    receiver: mpsc::Receiver<Tracked>,
    depth: usize,
}

impl UnboundedQueue {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender: Some(sender),
            receiver,
            depth: 0,
        }
    }
}

impl Queue for UnboundedQueue {
    fn offer(&mut self, payload: Tracked) -> Offer<Tracked> {
        match &self.sender {
            Some(sender) => match sender.send(payload) {
                Ok(()) => {
                    self.depth += 1;
                    Offer::Admitted
                }
                Err(mpsc::SendError(value)) => Offer::Closed(value),
            },
            None => Offer::Closed(payload),
        }
    }

    fn take(&mut self) -> Take<Tracked> {
        match self.receiver.try_recv() {
            Ok(value) => {
                self.depth -= 1;
                Take::Message(value)
            }
            Err(mpsc::TryRecvError::Empty) => Take::Empty,
            Err(mpsc::TryRecvError::Disconnected) => Take::Closed,
        }
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn close(&mut self) {
        self.sender = None;
    }
}

/// Arm R as a queue of payloads.
pub struct ReferenceQueue {
    model: RefModel<Tracked>,
}

impl ReferenceQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            model: RefModel::new(capacity),
        }
    }

    /// A queue that serves the newest message first (a canary).
    pub fn lifo(capacity: usize) -> Self {
        let mut model = RefModel::new(capacity);
        model.lifo = true;
        Self { model }
    }
}

impl Queue for ReferenceQueue {
    fn offer(&mut self, payload: Tracked) -> Offer<Tracked> {
        self.model.offer(payload)
    }
    fn take(&mut self) -> Take<Tracked> {
        self.model.take()
    }
    fn depth(&self) -> usize {
        self.model.depth()
    }
    fn close(&mut self) {
        self.model.close();
    }
}

/// The ways a queue is broken in the controls and canaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fault {
    /// On a full queue, drop the oldest message and admit the new one (a negative control).
    DropOldest,
    /// Deliver every fifth message a second time.
    DuplicateDelivery,
    /// Flip a byte of every seventh message's body.
    MutatePayload,
    /// Lose every sixth message between the queue and the consumer.
    Lossy,
    /// Refuse once only `nominal - 1` messages are queued.
    RefuseEarly,
    /// `close()` does nothing.
    IgnoreClose,
    /// A drained, closed queue reports `Empty` rather than `Closed`.
    ClosedAsEmpty,
}

/// A queue with one injected fault, wrapped around a sound one.
pub struct Faulty<Q: Queue> {
    inner: Q,
    fault: Fault,
    nominal_capacity: usize,
    takes: u64,
    pending_copy: Option<Tracked>,
}

impl<Q: Queue> Faulty<Q> {
    pub fn new(inner: Q, fault: Fault, nominal_capacity: usize) -> Self {
        Self {
            inner,
            fault,
            nominal_capacity,
            takes: 0,
            pending_copy: None,
        }
    }
}

impl<Q: Queue> Queue for Faulty<Q> {
    fn offer(&mut self, payload: Tracked) -> Offer<Tracked> {
        match self.fault {
            Fault::DropOldest => match self.inner.offer(payload) {
                Offer::Full(value) => {
                    // The oldest message vanishes here: nobody is told.
                    let _ = self.inner.take();
                    self.inner.offer(value)
                }
                other => other,
            },
            Fault::RefuseEarly if self.inner.depth() + 1 >= self.nominal_capacity => {
                Offer::Full(payload)
            }
            _ => self.inner.offer(payload),
        }
    }

    fn take(&mut self) -> Take<Tracked> {
        self.takes += 1;
        if let Some(copy) = self.pending_copy.take() {
            return Take::Message(copy);
        }
        match (self.fault, self.inner.take()) {
            (Fault::DuplicateDelivery, Take::Message(value)) => {
                if self.takes.is_multiple_of(5) {
                    self.pending_copy = Some(value.forge_copy());
                }
                Take::Message(value)
            }
            (Fault::MutatePayload, Take::Message(mut value)) => {
                if value.id() % 7 == 0 {
                    value.corrupt();
                }
                Take::Message(value)
            }
            (Fault::Lossy, Take::Message(value)) if self.takes.is_multiple_of(6) => {
                drop(value);
                Take::Empty
            }
            (Fault::ClosedAsEmpty, Take::Closed) => Take::Empty,
            (_, taken) => taken,
        }
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn close(&mut self) {
        if self.fault != Fault::IgnoreClose {
            self.inner.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::r001::payload::{Corpus, DropLog};

    fn payloads(n: usize) -> (std::sync::Arc<DropLog>, Corpus) {
        (DropLog::new(n), Corpus::generate(3, n, 8))
    }

    fn drain(queue: &mut impl Queue) -> Vec<u32> {
        let mut ids = Vec::new();
        while let Take::Message(t) = queue.take() {
            ids.push(t.id());
        }
        ids
    }

    #[test]
    fn the_reference_model_is_fifo_bounded_and_closes_before_it_fills() {
        let mut model = RefModel::new(2);
        assert_eq!(model.offer(1), Offer::Admitted);
        assert_eq!(model.offer(2), Offer::Admitted);
        assert_eq!(model.offer(3), Offer::Full(3));
        assert_eq!(model.depth(), 2);
        model.close();
        assert_eq!(model.offer(4), Offer::Closed(4));
        assert_eq!(model.take(), Take::Message(1));
        assert_eq!(model.take(), Take::Message(2));
        assert_eq!(model.take(), Take::Closed);
        let mut open: RefModel<u8> = RefModel::new(1);
        assert_eq!(open.take(), Take::Empty);
    }

    #[test]
    fn mailbox_and_reference_queues_agree_on_a_hand_made_sequence() {
        let (log, corpus) = payloads(8);
        let mut real = MailboxQueue::new(2, CloseVia::Owner);
        let mut reference = ReferenceQueue::new(2);
        for id in 0..4u32 {
            let a = real.offer(corpus.make(&log, id, 0));
            let b = reference.offer(corpus.make(&log, id, 0));
            assert_eq!(
                std::mem::discriminant(&a),
                std::mem::discriminant(&b),
                "offer {id}"
            );
        }
        assert_eq!(real.depth(), reference.depth());
        assert_eq!(real.peak_depth(), Some(2));
        assert_eq!(drain(&mut real), drain(&mut reference));
    }

    #[test]
    fn both_closing_routes_make_offers_closed_and_drain_then_report_closed() {
        for via in [CloseVia::Owner, CloseVia::Sender] {
            let (log, corpus) = payloads(4);
            let mut queue = MailboxQueue::new(2, via);
            assert!(matches!(
                queue.offer(corpus.make(&log, 0, 0)),
                Offer::Admitted
            ));
            queue.close();
            assert!(matches!(
                queue.offer(corpus.make(&log, 1, 0)),
                Offer::Closed(_)
            ));
            assert!(matches!(
                queue.offer(corpus.make(&log, 2, 0)),
                Offer::Closed(_)
            ));
            assert!(matches!(queue.take(), Take::Message(_)));
            assert!(matches!(queue.take(), Take::Closed));
        }
    }

    #[test]
    fn the_sync_channel_arm_agrees_with_the_spec_on_full_closed_and_fifo() {
        let (log, corpus) = payloads(8);
        let mut queue = SyncChannelQueue::new(2);
        assert!(matches!(
            queue.offer(corpus.make(&log, 0, 0)),
            Offer::Admitted
        ));
        assert!(matches!(
            queue.offer(corpus.make(&log, 1, 0)),
            Offer::Admitted
        ));
        assert!(matches!(
            queue.offer(corpus.make(&log, 2, 0)),
            Offer::Full(_)
        ));
        assert_eq!((queue.depth(), queue.peak_depth()), (2, Some(2)));
        queue.close();
        // Closed beats Full, and the buffer is still drained before the channel says Closed.
        assert!(matches!(
            queue.offer(corpus.make(&log, 3, 0)),
            Offer::Closed(_)
        ));
        assert_eq!(drain(&mut queue), [0, 1]);
        assert!(matches!(queue.take(), Take::Closed));
        assert_eq!(queue.depth(), 0);
        let mut open = SyncChannelQueue::new(1);
        assert!(matches!(open.take(), Take::Empty));
    }

    #[test]
    fn a_panicking_product_is_contained_counted_and_answered_with_panicked() {
        let (log, corpus) = payloads(4);
        // A payload whose drop panics is not a product panic; a poisoned queue is not either.
        // The product panics only if the code under test does, which a mutation plants; here the
        // counting path is exercised through `contain` directly.
        let queue = MailboxQueue::new(1, CloseVia::Owner);
        assert_eq!(queue.contain(|| 3), Some(3));
        assert_eq!(queue.contain(|| -> u8 { panic!("boom") }), None);
        assert_eq!(queue.panics(), 1);
        assert_eq!(queue.contain(|| -> u8 { panic!("boom") }), None);
        assert_eq!(queue.panics(), 2);
        drop((log, corpus));
    }

    #[test]
    fn the_unbounded_queue_admits_everything_and_counts_its_depth() {
        let (log, corpus) = payloads(100);
        let mut queue = UnboundedQueue::new();
        for id in 0..100u32 {
            assert!(matches!(
                queue.offer(corpus.make(&log, id, 0)),
                Offer::Admitted
            ));
        }
        assert_eq!(queue.depth(), 100);
        queue.close();
        assert!(matches!(
            queue.offer(corpus.make(&log, 0, 0)),
            Offer::Closed(_)
        ));
        assert_eq!(drain(&mut queue).len(), 100);
        assert!(matches!(queue.take(), Take::Closed));
        assert_eq!(queue.depth(), 0);
    }

    #[test]
    fn drop_oldest_admits_everything_and_loses_the_oldest() {
        let (log, corpus) = payloads(6);
        let mut queue = Faulty::new(MailboxQueue::new(2, CloseVia::Owner), Fault::DropOldest, 2);
        for id in 0..4u32 {
            assert!(matches!(
                queue.offer(corpus.make(&log, id, 0)),
                Offer::Admitted
            ));
        }
        assert_eq!(drain(&mut queue), [2, 3]);
        assert_eq!(log.drops_of(0), 1, "message 0 was dropped silently");
    }

    #[test]
    fn each_other_fault_misbehaves_in_its_own_way() {
        let (log, corpus) = payloads(40);
        // Duplicate: the fifth take repeats the fourth message.
        let mut queue = Faulty::new(ReferenceQueue::new(40), Fault::DuplicateDelivery, 40);
        for id in 0..8u32 {
            let _ = queue.offer(corpus.make(&log, id, 0));
        }
        let ids = drain(&mut queue);
        assert!(ids.len() > 8);
        // Mutate: message 7 arrives with another body.
        let mut queue = Faulty::new(ReferenceQueue::new(40), Fault::MutatePayload, 40);
        for id in 5..9u32 {
            let _ = queue.offer(corpus.make(&log, id, 0));
        }
        let mut altered = 0;
        while let Take::Message(t) = queue.take() {
            altered += usize::from(t.digest() != corpus.digests()[t.id() as usize]);
        }
        assert_eq!(altered, 1);
        // Lossy: a message disappears and the take reports Empty.
        let mut queue = Faulty::new(ReferenceQueue::new(40), Fault::Lossy, 40);
        for id in 0..6u32 {
            let _ = queue.offer(corpus.make(&log, id, 0));
        }
        let kinds: Vec<bool> = (0..6)
            .map(|_| matches!(queue.take(), Take::Empty))
            .collect();
        assert_eq!(kinds.iter().filter(|empty| **empty).count(), 1);
        // RefuseEarly: capacity 4 holds three.
        let mut queue = Faulty::new(ReferenceQueue::new(4), Fault::RefuseEarly, 4);
        let admitted = (0..4u32)
            .filter(|id| matches!(queue.offer(corpus.make(&log, *id, 0)), Offer::Admitted))
            .count();
        assert_eq!(admitted, 3);
        // IgnoreClose and ClosedAsEmpty.
        let mut queue = Faulty::new(ReferenceQueue::new(4), Fault::IgnoreClose, 4);
        queue.close();
        assert!(matches!(
            queue.offer(corpus.make(&log, 0, 0)),
            Offer::Admitted
        ));
        let mut queue = Faulty::new(ReferenceQueue::new(4), Fault::ClosedAsEmpty, 4);
        queue.close();
        assert!(matches!(queue.take(), Take::Empty));
        // LIFO reference.
        let mut queue = ReferenceQueue::lifo(4);
        for id in 0..3u32 {
            let _ = queue.offer(corpus.make(&log, id, 0));
        }
        assert_eq!(drain(&mut queue), [2, 1, 0]);
    }
}
