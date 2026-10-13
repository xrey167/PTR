//! The isolate state-machine contract and its replay scaffolds.

#[derive(Clone, Debug, PartialEq)]
pub enum Effect<M> {
    None,
    Emit(M),
    Stop,
    Fault(String),
}

pub trait Isolate<M> {
    fn handle(&mut self, message: M) -> Effect<M>;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplayTrace {
    pub events: Vec<String>,
}
impl ReplayTrace {
    pub fn push(&mut self, event: impl Into<String>) {
        self.events.push(event.into());
    }
}
