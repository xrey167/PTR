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


#[cfg(test)]
mod tests {
    use super::*;

    mod construction {
        use super::*;

        #[test]
        fn request_event_contains_typed_request_and_revision_fields() {
            let event = TraceEvent::request(
                TraceLevel::Info,
                "request.started",
                RequestId::from("r-unit"),
                Revision(11),
            );

            assert_eq!(
                event.fields.get(fields::REQUEST_ID),
                Some(&TraceValue::RequestId(RequestId::from("r-unit")))
            );
            assert_eq!(
                event.fields.get(fields::REVISION),
                Some(&TraceValue::Revision(Revision(11)))
            );
        }

        #[test]
        fn with_field_replaces_existing_named_field() {
            let event = TraceEvent::new(TraceLevel::Debug, "replace")
                .with_field(fields::OUTCOME, TraceValue::String("first".into()))
                .with_field(fields::OUTCOME, TraceValue::String("second".into()));

            assert_eq!(
                event.fields.get(fields::OUTCOME),
                Some(&TraceValue::String("second".into()))
            );
        }
    }

    mod sink {
        use super::*;

        #[test]
        fn noop_sink_returns_ok_for_valid_event() {
            let event = TraceEvent::new(TraceLevel::Trace, "noop");
            assert_eq!(NoopTraceSink.emit(&event), Ok(()));
        }
    }
}
