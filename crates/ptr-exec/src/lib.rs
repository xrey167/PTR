//! Bounded isolate execution semantics. Tokio/Compio are transport substrates, not the semantic scheduler contract.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};

#[derive(Debug, PartialEq)]
pub enum SendError<T> { Full(T), Closed(T) }

pub struct Mailbox<T> { sender: SyncSender<T>, receiver: Receiver<T> }
impl<T> Mailbox<T> {
    pub fn bounded(capacity: usize) -> Self { let (sender, receiver) = sync_channel(capacity); Self { sender, receiver } }
    pub fn try_send(&self, value: T) -> Result<(), SendError<T>> {
        match self.sender.try_send(value) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(v)) => Err(SendError::Full(v)),
            Err(TrySendError::Disconnected(v)) => Err(SendError::Closed(v)),
        }
    }
    pub fn try_recv(&self) -> Option<T> { self.receiver.try_recv().ok() }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Effect<M> { None, Emit(M), Stop, Fault(String) }

pub trait Isolate<M> { fn handle(&mut self, message: M) -> Effect<M>; }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplayTrace { pub events: Vec<String> }
impl ReplayTrace { pub fn push(&mut self, event: impl Into<String>) { self.events.push(event.into()); } }
