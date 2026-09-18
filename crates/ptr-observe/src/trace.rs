use std::collections::BTreeMap;

use ptr_types::{CommitIndex, Generation, RequestId, Revision};

use crate::{fields, TraceError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceValue {
    String(String),
    U64(u64),
    Bool(bool),
    RequestId(RequestId),
    Revision(Revision),
    Generation(Generation),
    CommitIndex(CommitIndex),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceEvent {
    pub level: TraceLevel,
    pub name: String,
    pub fields: BTreeMap<String, TraceValue>,
}

impl TraceEvent {
    pub fn new(level: TraceLevel, name: impl Into<String>) -> Self {
        Self {
            level,
            name: name.into(),
            fields: BTreeMap::new(),
        }
    }

    pub fn request(
        level: TraceLevel,
        name: impl Into<String>,
        request_id: RequestId,
        revision: Revision,
    ) -> Self {
        Self::new(level, name)
            .with_field(fields::REQUEST_ID, TraceValue::RequestId(request_id))
            .with_field(fields::REVISION, TraceValue::Revision(revision))
    }

    pub fn with_field(mut self, key: impl Into<String>, value: TraceValue) -> Self {
        self.fields.insert(key.into(), value);
        self
    }
}

pub trait TraceSink: Send + Sync {
    fn emit(&self, event: &TraceEvent) -> Result<(), TraceError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopTraceSink;

impl TraceSink for NoopTraceSink {
    fn emit(&self, _event: &TraceEvent) -> Result<(), TraceError> {
        Ok(())
    }
}
