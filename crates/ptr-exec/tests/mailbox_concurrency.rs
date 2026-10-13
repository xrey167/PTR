//! Public-API tests of the mailbox under contention: producers on several threads, one
//! consumer, a bounded queue that never loses, duplicates or over-fills.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use ptr_exec::{Mailbox, Receive, SendError};

#[test]
fn full_returns_the_value_and_closed_returns_the_value() {
    let mailbox = Mailbox::bounded(1);
    let sender = mailbox.sender();
    sender.try_send(String::from("kept")).unwrap();
    match sender.try_send(String::from("refused")) {
        Err(SendError::Full(value)) => assert_eq!(value, "refused"),
        other => panic!("expected Full, got ok={}", other.is_ok()),
    }
    mailbox.close();
    match sender.try_send(String::from("late")) {
        Err(SendError::Closed(value)) => assert_eq!(value, "late"),
        other => panic!("expected Closed, got ok={}", other.is_ok()),
    }
    assert_eq!(
        mailbox.try_receive(),
        Receive::Message(String::from("kept"))
    );
    assert_eq!(mailbox.try_receive(), Receive::Closed);
}

#[test]
fn concurrent_producers_neither_lose_nor_duplicate_and_depth_never_exceeds_capacity() {
    const PRODUCERS: usize = 4;
    const PER_PRODUCER: usize = 5_000;
    const CAPACITY: usize = 8;
    let mailbox = Mailbox::bounded(CAPACITY);
    let over_capacity = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..PRODUCERS)
        .map(|producer| {
            let sender = mailbox.sender();
            let over = Arc::clone(&over_capacity);
            thread::spawn(move || {
                let mut refused = Vec::new();
                let mut admitted = 0usize;
                for n in 0..PER_PRODUCER {
                    let id = producer * PER_PRODUCER + n;
                    match sender.try_send(id) {
                        Ok(()) => admitted += 1,
                        Err(SendError::Full(value)) => {
                            assert_eq!(value, id, "Full returns the very value sent");
                            refused.push(value);
                        }
                        Err(SendError::Closed(_)) => panic!("not closed"),
                    }
                    // Read order matters: depth first, then peak. The peak is raised before the
                    // depth is published, so a later peak read is never below an earlier depth.
                    let depth = sender.depth();
                    let peak = sender.peak_depth();
                    if depth > CAPACITY || peak < depth || peak > CAPACITY {
                        over.fetch_add(1, Ordering::SeqCst);
                    }
                }
                (admitted, refused)
            })
        })
        .collect();
    let mut received = Vec::new();
    // The consumer drains on this thread while the producers run, then finishes the queue.
    while !handles.iter().all(|handle| handle.is_finished()) {
        assert!(mailbox.depth() <= CAPACITY);
        if let Some(id) = mailbox.try_recv() {
            received.push(id);
        }
    }
    let mut admitted_total = 0;
    let mut refused_ids = Vec::new();
    for handle in handles {
        let (admitted, refused) = handle.join().expect("producer");
        admitted_total += admitted;
        refused_ids.extend(refused);
    }
    while let Some(id) = mailbox.try_recv() {
        received.push(id);
    }
    assert_eq!(over_capacity.load(Ordering::SeqCst), 0);
    assert!(mailbox.peak_depth() <= CAPACITY);
    assert_eq!(
        received.len(),
        admitted_total,
        "every admitted message is received once"
    );
    let received_set: BTreeSet<_> = received.iter().copied().collect();
    assert_eq!(received_set.len(), received.len(), "no duplicate delivery");
    let refused_set: BTreeSet<_> = refused_ids.iter().copied().collect();
    assert!(
        received_set.is_disjoint(&refused_set),
        "a refused value is never also delivered"
    );
    assert_eq!(
        received_set.len() + refused_set.len(),
        PRODUCERS * PER_PRODUCER,
        "each offered id has exactly one outcome"
    );
    // Per producer, delivered ids keep their send order (FIFO per sender).
    for producer in 0..PRODUCERS {
        let lane: Vec<usize> = received
            .iter()
            .copied()
            .filter(|id| id / PER_PRODUCER == producer)
            .collect();
        assert!(
            lane.windows(2).all(|pair| pair[0] < pair[1]),
            "producer {producer} order"
        );
    }
    assert_eq!(mailbox.depth(), 0);
}

#[test]
fn close_while_producers_run_returns_every_late_value_and_closed_is_monotone() {
    const PRODUCERS: usize = 3;
    const GUARD: usize = 50_000_000;
    let mailbox = Mailbox::bounded(4);
    let handles: Vec<_> = (0..PRODUCERS)
        .map(|producer| {
            let sender = mailbox.sender();
            thread::spawn(move || {
                // Offer until Closed is seen, then offer a few more: every later result must
                // also be Closed (a close is never undone and never skipped).
                let (mut admitted, mut full, mut closed) = (0usize, 0usize, 0usize);
                let mut seen_closed = false;
                let mut after_closed = 0usize;
                for n in 0..GUARD {
                    let id = producer * GUARD + n;
                    match sender.try_send(id) {
                        Ok(()) => {
                            assert!(!seen_closed, "Ok after this producer saw Closed");
                            admitted += 1;
                        }
                        Err(SendError::Full(value)) => {
                            assert_eq!(value, id);
                            assert!(!seen_closed, "Full after this producer saw Closed");
                            full += 1;
                        }
                        Err(SendError::Closed(value)) => {
                            assert_eq!(value, id);
                            seen_closed = true;
                            closed += 1;
                        }
                    }
                    if seen_closed {
                        after_closed += 1;
                        if after_closed >= 100 {
                            break;
                        }
                    }
                }
                assert!(
                    seen_closed,
                    "the mailbox was closed while this producer ran"
                );
                (admitted, full, closed)
            })
        })
        .collect();
    let mut received = 0usize;
    let mut closed_once = false;
    while !handles.iter().all(|handle| handle.is_finished()) {
        if received > 100 && !closed_once {
            mailbox.close();
            closed_once = true;
        }
        if mailbox.try_recv().is_some() {
            received += 1;
        }
    }
    assert!(closed_once, "the close landed while producers were running");
    let (mut admitted, mut full, mut closed) = (0, 0, 0);
    for handle in handles {
        let (a, f, c) = handle.join().expect("producer");
        admitted += a;
        full += f;
        closed += c;
    }
    assert!(
        admitted > 100 && closed >= PRODUCERS * 100,
        "close landed mid-flight"
    );
    let _ = full;
    loop {
        match mailbox.try_receive() {
            Receive::Message(_) => received += 1,
            Receive::Closed => break,
            Receive::Empty => panic!("a closed mailbox is never Empty"),
        }
    }
    assert_eq!(received, admitted);
}

#[test]
fn a_fill_by_parked_producers_leaves_exactly_capacity_messages_in_the_queue() {
    const CAPACITY: usize = 8;
    let mailbox = Mailbox::bounded(CAPACITY);
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = (0..4usize)
        .map(|producer| {
            let sender = mailbox.sender();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                (0..16usize)
                    .filter(|n| sender.try_send(producer * 16 + n).is_ok())
                    .count()
            })
        })
        .collect();
    let admitted: usize = handles
        .into_iter()
        .map(|h| h.join().expect("producer"))
        .sum();
    assert_eq!(admitted, CAPACITY);
    // The real queue, not the atomic mirror: exactly CAPACITY messages are recoverable.
    let mut recovered = 0;
    while mailbox.try_recv().is_some() {
        recovered += 1;
    }
    assert_eq!(recovered, CAPACITY);
    assert_eq!(mailbox.peak_depth(), CAPACITY);
}

#[test]
fn capacity_zero_refuses_every_send_from_every_thread() {
    let mailbox: Mailbox<usize> = Mailbox::bounded(0);
    let handles: Vec<_> = (0..4usize)
        .map(|producer| {
            let sender = mailbox.sender();
            thread::spawn(move || {
                (0..500usize).all(|n| {
                    let id = producer * 500 + n;
                    sender.try_send(id) == Err(SendError::Full(id))
                })
            })
        })
        .collect();
    for handle in handles {
        assert!(handle.join().expect("producer"));
    }
    assert_eq!((mailbox.depth(), mailbox.peak_depth()), (0, 0));
    assert_eq!(mailbox.try_receive(), Receive::Empty);
}

/// A payload whose destructor panics while armed.
struct Bomb(Arc<AtomicUsize>);

impl Drop for Bomb {
    fn drop(&mut self) {
        if self.0.load(Ordering::SeqCst) == 1 {
            panic!("payload destructor panics");
        }
    }
}

#[test]
fn a_payload_destructor_that_panics_in_the_owner_drop_does_not_break_the_senders() {
    let armed = Arc::new(AtomicUsize::new(1));
    let mailbox = Mailbox::bounded(2);
    let sender = mailbox.sender();
    let other = sender.clone();
    assert!(sender.try_send(Bomb(Arc::clone(&armed))).is_ok());
    // The owner is dropped on a thread that panics inside the queued payload's destructor.
    let outcome = thread::spawn(move || drop(mailbox)).join();
    assert!(outcome.is_err(), "the payload destructor panicked");
    armed.store(0, Ordering::SeqCst);
    assert!(sender.is_closed());
    assert_eq!(sender.depth(), 0);
    match other.try_send(Bomb(Arc::clone(&armed))) {
        Err(SendError::Closed(returned)) => drop(returned),
        Err(SendError::Full(_)) | Ok(()) => panic!("expected Closed with the value"),
    }
    // The same sender is usable from other threads afterwards, still Closed, never poisoned.
    let again = thread::spawn(move || sender.try_send(Bomb(armed)).is_err())
        .join()
        .expect("thread");
    assert!(again);
}

#[test]
fn a_sender_outlives_a_dropped_owner_and_is_refused_with_its_value() {
    let mailbox = Mailbox::bounded(2);
    let sender = mailbox.sender();
    let moved = sender.clone();
    drop(mailbox);
    let result = thread::spawn(move || moved.try_send(41u32))
        .join()
        .expect("thread");
    assert_eq!(result, Err(SendError::Closed(41)));
    assert!(sender.is_closed());
}
