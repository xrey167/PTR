use ptr_types::{
    Digest, Generation, PodRevisionAddress, RequestId, Revision, ScopeId, SessionId, StateId,
    TraceId, TypeId,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PodTurnKind {
    SessionCreated,
    RequestStarted,
    InputReceived,
    PartialOutput,
    ToolCall,
    Observation,
    Hypothesis,
    TurnCommitted,
    TurnInterrupted,
    TurnResumed,
    RequestCompleted,
    RequestFailed,
    RequestUncertain,
    SessionClosed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodTurnEvent {
    pub kind: PodTurnKind,
    pub sequence: u64,
    pub session_id: SessionId,
    pub scope_id: ScopeId,
    pub turn_id: u64,
    pub request_id: RequestId,
    pub pod: PodRevisionAddress,
    pub generation: Generation,
    pub manifest_digest: Digest,
    pub artifact_digest: Digest,
    pub input_type: Option<TypeId>,
    pub output_type: Option<TypeId>,
    pub state_before: Option<StateId>,
    pub state_after: Option<StateId>,
    pub revision: Revision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TurnError {
    Closed,
    InvalidSequence,
    InvalidBinding,
    /// `emit` was given a kind that only `commit_turn`, `interrupt` or `close`
    /// may produce, because each of them also moves session state.
    ReservedKind,
}

#[derive(Clone, Debug)]
pub struct DuplexSession {
    pub session_id: SessionId,
    pub trace_id: TraceId,
    pub epoch: u64,
    pub turn_id: u64,
    next_sequence: u64,
    events: Vec<PodTurnEvent>,
    closed: bool,
}

impl DuplexSession {
    pub fn new(session_id: SessionId, trace_id: TraceId) -> Result<Self, TurnError> {
        if session_id.0.is_empty() || trace_id.0.is_empty() {
            return Err(TurnError::InvalidBinding);
        }
        Ok(Self {
            session_id,
            trace_id,
            epoch: 0,
            turn_id: 0,
            next_sequence: 0,
            events: Vec::new(),
            closed: false,
        })
    }
    pub fn emit(&mut self, event: PodTurnEvent) -> Result<&PodTurnEvent, TurnError> {
        // These three kinds change session state (turn counter, epoch, closed).
        // Emitting one here would record the event without that change, so they
        // only enter the log through the methods that make it.
        if matches!(
            event.kind,
            PodTurnKind::TurnCommitted | PodTurnKind::TurnInterrupted | PodTurnKind::SessionClosed
        ) {
            return Err(TurnError::ReservedKind);
        }
        self.check(&event)?;
        Ok(self.push(event))
    }

    /// Every rule an event must meet, without changing the session. A rejected
    /// event must leave the counters exactly as they were.
    fn check(&self, event: &PodTurnEvent) -> Result<(), TurnError> {
        if self.closed {
            return Err(TurnError::Closed);
        }
        if event.session_id != self.session_id || event.sequence != self.next_sequence + 1 {
            return Err(TurnError::InvalidSequence);
        }
        event
            .pod
            .validate()
            .map_err(|_| TurnError::InvalidBinding)?;
        if event.generation != event.pod.generation
            || event.manifest_digest == [0; 32]
            || event.artifact_digest == [0; 32]
        {
            return Err(TurnError::InvalidBinding);
        }
        Ok(())
    }

    fn push(&mut self, event: PodTurnEvent) -> &PodTurnEvent {
        self.next_sequence = event.sequence;
        self.events.push(event);
        self.events.last().expect("event was just pushed")
    }

    pub fn commit_turn(&mut self, mut event: PodTurnEvent) -> Result<&PodTurnEvent, TurnError> {
        event.kind = PodTurnKind::TurnCommitted;
        event.turn_id = self.turn_id + 1;
        self.check(&event)?;
        self.turn_id += 1;
        Ok(self.push(event))
    }

    pub fn interrupt(&mut self, mut event: PodTurnEvent) -> Result<&PodTurnEvent, TurnError> {
        event.kind = PodTurnKind::TurnInterrupted;
        self.check(&event)?;
        self.epoch += 1;
        Ok(self.push(event))
    }
    pub fn resume(&self, last_sequence: u64) -> Result<Vec<PodTurnEvent>, TurnError> {
        if last_sequence > self.next_sequence {
            return Err(TurnError::InvalidSequence);
        }
        Ok(self
            .events
            .iter()
            .filter(|event| event.sequence > last_sequence)
            .cloned()
            .collect())
    }
    pub fn close(&mut self, mut event: PodTurnEvent) -> Result<PodTurnEvent, TurnError> {
        event.kind = PodTurnKind::SessionClosed;
        self.check(&event)?;
        let emitted = self.push(event).clone();
        self.closed = true;
        Ok(emitted)
    }
    pub fn events(&self) -> &[PodTurnEvent] {
        &self.events
    }
}
