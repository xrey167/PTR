//! The bounded typed mailbox: one receiving owner, any number of cloneable senders.
//!
//! Semantics (INVARIANTS #8 bounded queues, #9 single owner):
//!
//! - At most `capacity` messages are queued at any instant. A send into a full mailbox is
//!   refused with [`SendError::Full`] and hands the value back; nothing is dropped, replaced
//!   or blocked on.
//! - [`Mailbox::close`] (or [`MailboxSender::close`], or dropping the [`Mailbox`]) makes every
//!   further send return [`SendError::Closed`] with the value. Messages queued before the
//!   close stay receivable; [`Mailbox::try_receive`] reports [`Receive::Closed`] only once the
//!   queue is empty as well.
//! - Capacity 0 is a mailbox that never holds a message: every send returns
//!   [`SendError::Full`]. (The channel this type once wrapped made it a rendezvous, which a
//!   non-blocking receiver can never complete.) `ptr-config` rejects a zero
//!   `runtime.mailbox_capacity`; the type itself does not panic on it.
//! - The queue is a `VecDeque` behind a mutex; the depth is also mirrored in an atomic written
//!   only while the lock is held, so a reader on another thread sees a depth that equals the
//!   queue length at some instant and never exceeds the capacity.
//! - The receiving end is not `Sync`: it is owned by one consumer. Senders are `Clone`, and
//!   `Send + Sync` exactly when `T: Send`, so they may be used from any number of threads.
//!
//! Relation to INVARIANTS #9 ("shared mutation is not the default execution model"): the only
//! shared mutable thing is this channel's queue, behind one mutex whose critical sections only
//! push, pop or flip a flag. The receiver stays single-owner and no message is shared: a send
//! moves the value in, a receive moves it out.

use std::cell::Cell;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A send that did not enqueue the value; the value is returned to the caller.
#[derive(PartialEq, Eq)]
pub enum SendError<T> {
    /// The mailbox holds `capacity` messages.
    Full(T),
    /// The mailbox was closed (explicitly, or its receiving owner was dropped).
    Closed(T),
}

impl<T> std::fmt::Debug for SendError<T> {
    /// The variant only: the value is neither required to be `Debug` nor printed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full(_) => f.write_str("Full(..)"),
            Self::Closed(_) => f.write_str("Closed(..)"),
        }
    }
}

impl<T> std::fmt::Display for SendError<T> {
    /// The value is not printed: the message names only the refusal.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full(_) => f.write_str("mailbox is full"),
            Self::Closed(_) => f.write_str("mailbox is closed"),
        }
    }
}

impl<T> std::error::Error for SendError<T> {}

impl<T> SendError<T> {
    /// The value that was not sent.
    pub fn into_inner(self) -> T {
        match self {
            Self::Full(value) | Self::Closed(value) => value,
        }
    }
}

/// The result of a non-blocking receive: a message, nothing yet, or nothing ever again.
#[derive(Debug, PartialEq, Eq)]
pub enum Receive<T> {
    Message(T),
    /// The queue is empty and the mailbox is still open.
    Empty,
    /// The queue is empty and the mailbox is closed: no message will ever arrive.
    Closed,
}

impl<T> Receive<T> {
    /// The message, or `None` for `Empty` and `Closed` alike.
    pub fn into_message(self) -> Option<T> {
        match self {
            Self::Message(value) => Some(value),
            Self::Empty | Self::Closed => None,
        }
    }
}

struct State<T> {
    queue: VecDeque<T>,
    closed: bool,
}

struct Shared<T> {
    capacity: usize,
    state: Mutex<State<T>>,
    /// Mirror of `queue.len()`, stored only under the lock.
    depth: AtomicUsize,
    /// The largest `depth` ever stored.
    peak: AtomicUsize,
    /// Mirror of `closed`, stored only under the lock.
    closed: AtomicBool,
}

impl<T> Shared<T> {
    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // No critical section leaves the queue half-updated, so a poisoned lock is still sound.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn debug_fmt(&self, name: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(name)
            .field("capacity", &self.capacity)
            .field("depth", &self.depth.load(Ordering::Acquire))
            .field("closed", &self.closed.load(Ordering::Acquire))
            .finish()
    }

    fn try_send(&self, value: T) -> Result<(), SendError<T>> {
        let mut state = self.lock();
        if state.closed {
            return Err(SendError::Closed(value));
        }
        if state.queue.len() >= self.capacity {
            return Err(SendError::Full(value));
        }
        state.queue.push_back(value);
        let depth = state.queue.len();
        // Peak first: a reader that loads `depth` and then `peak` never sees depth > peak.
        self.peak.fetch_max(depth, Ordering::AcqRel);
        self.depth.store(depth, Ordering::Release);
        Ok(())
    }

    fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        self.closed.store(true, Ordering::Release);
    }

    fn receive(&self) -> Receive<T> {
        let mut state = self.lock();
        match state.queue.pop_front() {
            Some(value) => {
                self.depth.store(state.queue.len(), Ordering::Release);
                Receive::Message(value)
            }
            None if state.closed => Receive::Closed,
            None => Receive::Empty,
        }
    }
}

/// The owner of a bounded mailbox and its only receiver.
///
/// `Mailbox` is `Send`, but its receiving end is single-owner, so it is not `Sync`:
///
/// ```compile_fail
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<ptr_exec::Mailbox<u8>>();
/// ```
///
/// A sender is `Clone`, and `Send + Sync` whenever `T: Send`, so any number of threads can
/// share one:
///
/// ```
/// fn needs_send_sync_clone<T: Send + Sync + Clone>() {}
/// needs_send_sync_clone::<ptr_exec::MailboxSender<u8>>();
/// fn needs_send<T: Send>() {}
/// needs_send::<ptr_exec::Mailbox<u8>>();
/// ```
pub struct Mailbox<T> {
    shared: Arc<Shared<T>>,
    /// Keeps the receiving end single-owner: `Send` but not `Sync`.
    _single_consumer: PhantomData<Cell<()>>,
}

impl<T> Mailbox<T> {
    /// A mailbox that holds at most `capacity` messages (0 holds none: see the module docs).
    pub fn bounded(capacity: usize) -> Self {
        Self {
            shared: Arc::new(Shared {
                capacity,
                state: Mutex::new(State {
                    queue: VecDeque::with_capacity(capacity.min(4096)),
                    closed: false,
                }),
                depth: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                closed: AtomicBool::new(false),
            }),
            _single_consumer: PhantomData,
        }
    }

    /// A new sending handle; any number can exist and be moved to other threads.
    pub fn sender(&self) -> MailboxSender<T> {
        MailboxSender {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Enqueue `value`, or return it inside the refusal.
    pub fn try_send(&self, value: T) -> Result<(), SendError<T>> {
        self.shared.try_send(value)
    }

    /// The next message, telling an empty open mailbox from a closed, drained one.
    pub fn try_receive(&self) -> Receive<T> {
        self.shared.receive()
    }

    /// The next message, or `None` when there is none. `None` does not say whether the mailbox
    /// is closed: use [`Mailbox::try_receive`] where that matters.
    pub fn try_recv(&self) -> Option<T> {
        self.try_receive().into_message()
    }

    /// Refuse every further send with `Closed(value)`; queued messages stay receivable.
    pub fn close(&self) {
        self.shared.close();
    }

    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire)
    }

    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    /// The number of queued messages; never above [`Mailbox::capacity`].
    pub fn depth(&self) -> usize {
        self.shared.depth.load(Ordering::Acquire)
    }

    /// The largest depth this mailbox has reached.
    pub fn peak_depth(&self) -> usize {
        self.shared.peak.load(Ordering::Acquire)
    }
}

impl<T> std::fmt::Debug for Mailbox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.shared.debug_fmt("Mailbox", f)
    }
}

impl<T> Drop for Mailbox<T> {
    fn drop(&mut self) {
        // The owner is gone: senders see `Closed`, and queued payloads are released now rather
        // than when the last sender handle goes.
        let drained = {
            let mut state = self.shared.lock();
            state.closed = true;
            self.shared.closed.store(true, Ordering::Release);
            self.shared.depth.store(0, Ordering::Release);
            std::mem::take(&mut state.queue)
        };
        drop(drained);
    }
}

/// A cloneable sending handle of one [`Mailbox`]: `Send + Sync` exactly when `T: Send`.
pub struct MailboxSender<T> {
    shared: Arc<Shared<T>>,
}

impl<T> std::fmt::Debug for MailboxSender<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.shared.debug_fmt("MailboxSender", f)
    }
}

impl<T> Clone for MailboxSender<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<T> MailboxSender<T> {
    /// Enqueue `value`, or return it inside the refusal.
    pub fn try_send(&self, value: T) -> Result<(), SendError<T>> {
        self.shared.try_send(value)
    }

    /// Close the mailbox for every sender (see [`Mailbox::close`]).
    pub fn close(&self) {
        self.shared.close();
    }

    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire)
    }

    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }

    pub fn depth(&self) -> usize {
        self.shared.depth.load(Ordering::Acquire)
    }

    pub fn peak_depth(&self) -> usize {
        self.shared.peak.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_mailbox_returns_the_value_and_keeps_its_queue() {
        let mailbox = Mailbox::bounded(2);
        mailbox.try_send(1).unwrap();
        mailbox.try_send(2).unwrap();
        assert_eq!(mailbox.try_send(3), Err(SendError::Full(3)));
        assert_eq!(mailbox.depth(), 2);
        assert_eq!(mailbox.try_recv(), Some(1));
        assert_eq!(mailbox.try_recv(), Some(2));
        assert_eq!(mailbox.try_recv(), None);
    }

    #[test]
    fn capacity_zero_refuses_every_send_and_returns_the_value() {
        let mailbox = Mailbox::bounded(0);
        assert_eq!(mailbox.try_send("a"), Err(SendError::Full("a")));
        assert_eq!(mailbox.sender().try_send("b"), Err(SendError::Full("b")));
        assert_eq!(mailbox.depth(), 0);
        assert_eq!(mailbox.peak_depth(), 0);
        assert_eq!(mailbox.try_receive(), Receive::Empty);
    }

    #[test]
    fn close_returns_the_value_and_lets_the_queue_drain_before_reporting_closed() {
        let mailbox = Mailbox::bounded(4);
        mailbox.try_send(1).unwrap();
        mailbox.close();
        assert!(mailbox.is_closed());
        assert_eq!(mailbox.try_send(2), Err(SendError::Closed(2)));
        assert_eq!(mailbox.sender().try_send(3), Err(SendError::Closed(3)));
        assert_eq!(mailbox.try_receive(), Receive::Message(1));
        assert_eq!(mailbox.try_receive(), Receive::Closed);
        assert_eq!(mailbox.try_receive(), Receive::Closed);
        // Closed is reported before Full when both hold.
        let full = Mailbox::bounded(1);
        full.try_send(1).unwrap();
        full.close();
        assert_eq!(full.try_send(2), Err(SendError::Closed(2)));
    }

    #[test]
    fn empty_and_closed_are_told_apart() {
        let mailbox: Mailbox<u8> = Mailbox::bounded(1);
        assert_eq!(mailbox.try_receive(), Receive::Empty);
        assert_eq!(mailbox.try_recv(), None);
        mailbox.sender().close();
        assert_eq!(mailbox.try_receive(), Receive::Closed);
        assert_eq!(mailbox.try_recv(), None);
    }

    #[test]
    fn dropping_the_owner_closes_the_senders_and_releases_the_queue() {
        use std::sync::atomic::AtomicUsize;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Counted;
        impl Drop for Counted {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mailbox = Mailbox::bounded(3);
        let sender = mailbox.sender();
        sender.try_send(Counted).ok().unwrap();
        sender.try_send(Counted).ok().unwrap();
        drop(mailbox);
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            2,
            "queued payloads dropped with the owner"
        );
        assert!(sender.is_closed());
        assert_eq!(sender.depth(), 0);
        match sender.try_send(Counted) {
            Err(SendError::Closed(returned)) => drop(returned),
            other => panic!("expected Closed, got {:?}", other.is_ok()),
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn depth_and_peak_follow_the_queue() {
        let mailbox = Mailbox::bounded(3);
        let sender = mailbox.sender();
        assert_eq!((mailbox.depth(), mailbox.peak_depth()), (0, 0));
        sender.try_send(1).unwrap();
        sender.try_send(2).unwrap();
        assert_eq!((mailbox.depth(), sender.depth()), (2, 2));
        mailbox.try_recv();
        assert_eq!((mailbox.depth(), mailbox.peak_depth()), (1, 2));
        sender.try_send(3).unwrap();
        sender.try_send(4).unwrap();
        assert_eq!(sender.try_send(5), Err(SendError::Full(5)));
        assert_eq!((mailbox.depth(), mailbox.peak_depth()), (3, 3));
        assert_eq!(mailbox.capacity(), 3);
        assert_eq!(sender.capacity(), 3);
    }

    #[test]
    fn into_inner_gives_the_value_back_for_both_refusals() {
        assert_eq!(SendError::Full(7).into_inner(), 7);
        assert_eq!(SendError::Closed(8).into_inner(), 8);
        assert_eq!(Receive::Message(1).into_message(), Some(1));
        assert_eq!(Receive::<u8>::Empty.into_message(), None);
        assert_eq!(Receive::<u8>::Closed.into_message(), None);
    }

    #[test]
    fn refusals_display_without_the_value_and_debug_shows_the_state() {
        let full: SendError<&str> = SendError::Full("secret");
        let closed: SendError<&str> = SendError::Closed("secret");
        assert_eq!(full.to_string(), "mailbox is full");
        assert_eq!(closed.to_string(), "mailbox is closed");
        assert_eq!(format!("{full:?} {closed:?}"), "Full(..) Closed(..)");
        let as_error: &dyn std::error::Error = &full;
        assert!(as_error.source().is_none());
        let mailbox = Mailbox::bounded(2);
        mailbox.try_send(1).unwrap();
        assert_eq!(
            format!("{mailbox:?}"),
            "Mailbox { capacity: 2, depth: 1, closed: false }"
        );
        mailbox.close();
        assert_eq!(
            format!("{:?}", mailbox.sender()),
            "MailboxSender { capacity: 2, depth: 1, closed: true }"
        );
    }

    #[test]
    fn a_poisoned_lock_changes_nothing_for_every_operation() {
        let mailbox = Mailbox::bounded(2);
        let sender = mailbox.sender();
        sender.try_send(1).unwrap();
        let shared = Arc::clone(&mailbox.shared);
        let poisoner = std::thread::spawn(move || {
            let _guard = shared.state.lock().unwrap();
            panic!("poison the mailbox lock");
        });
        assert!(poisoner.join().is_err());
        assert!(mailbox.shared.state.is_poisoned());
        sender.try_send(2).unwrap();
        assert_eq!(sender.try_send(3), Err(SendError::Full(3)));
        assert_eq!(mailbox.try_receive(), Receive::Message(1));
        sender.close();
        assert_eq!(sender.try_send(4), Err(SendError::Closed(4)));
        assert_eq!(mailbox.try_receive(), Receive::Message(2));
        assert_eq!(mailbox.try_receive(), Receive::Closed);
        drop(mailbox);
        assert_eq!(sender.try_send(5), Err(SendError::Closed(5)));
    }

    #[test]
    fn a_cloned_sender_feeds_the_same_queue_in_fifo_order() {
        let mailbox = Mailbox::bounded(4);
        let first = mailbox.sender();
        let second = first.clone();
        first.try_send(1).unwrap();
        second.try_send(2).unwrap();
        mailbox.try_send(3).unwrap();
        let drained: Vec<i32> = std::iter::from_fn(|| mailbox.try_recv()).collect();
        assert_eq!(drained, [1, 2, 3]);
    }
}
