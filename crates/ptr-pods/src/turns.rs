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
        self.next_sequence = event.sequence;
        self.events.push(event);
        Ok(self.events.last().expect("event was just pushed"))
    }
    pub fn commit_turn(&mut self, mut event: PodTurnEvent) -> Result<&PodTurnEvent, TurnError> {
        self.turn_id += 1;
        event.kind = PodTurnKind::TurnCommitted;
        event.turn_id = self.turn_id;
        self.emit(event)
    }
    pub fn interrupt(&mut self, mut event: PodTurnEvent) -> Result<&PodTurnEvent, TurnError> {
        self.epoch += 1;
        event.kind = PodTurnKind::TurnInterrupted;
        self.emit(event)
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
        let emitted = self.emit(event)?.clone();
        self.closed = true;
        Ok(emitted)
    }
    pub fn events(&self) -> &[PodTurnEvent] {
        &self.events
    }
}
