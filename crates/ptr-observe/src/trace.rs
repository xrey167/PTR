use std::collections::BTreeMap;

use ptr_types::{CommitIndex, Generation, RequestId, Revision};

use crate::{fields, macros::impl_trace_value_from, TraceError};

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

impl_trace_value_from!(TraceValue, String => String);
impl_trace_value_from!(TraceValue, U64 => u64);
impl_trace_value_from!(TraceValue, Bool => bool);
impl_trace_value_from!(TraceValue, RequestId => RequestId);
impl_trace_value_from!(TraceValue, Revision => Revision);
impl_trace_value_from!(TraceValue, Generation => Generation);
impl_trace_value_from!(TraceValue, CommitIndex => CommitIndex);

impl From<&str> for TraceValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
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

    pub fn with_field(
        mut self,
        key: impl Into<String>,
        value: impl Into<TraceValue>,
    ) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    pub fn iter_fields(&self) -> impl Iterator<Item = (&str, &TraceValue)> + '_ {
        self.fields
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    pub fn fields_matching<'event, F>(
        &'event self,
        mut predicate: F,
    ) -> impl Iterator<Item = (&'event str, &'event TraceValue)> + 'event
    where
        F: FnMut(&str, &TraceValue) -> bool + 'event,
    {
        self.iter_fields().filter(move |item| {
            match *item {
                (key, value) if predicate(key, value) => true,
                _ => false,
            }
        })
    }

    pub fn try_for_each_field<E, F>(&self, mut visitor: F) -> Result<(), E>
    where
        F: FnMut(&str, &TraceValue) -> Result<(), E>,
    {
        self.iter_fields()
            .try_for_each(|(key, value)| visitor(key, value))
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


        #[test]
        fn iterator_adapters_filter_named_fields_with_closure() {
            let event = TraceEvent::new(TraceLevel::Debug, "iter")
                .with_field(fields::OUTCOME, TraceValue::String("ok".into()))
                .with_field(fields::ERROR_CODE, TraceValue::String("none".into()));

            let fields = event
                .fields_matching(|key, _| key.starts_with("error."))
                .collect::<Vec<_>>();

            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].0, fields::ERROR_CODE);
        }

        #[test]
        fn try_for_each_field_short_circuits_on_named_failure() {
            let event = TraceEvent::new(TraceLevel::Debug, "try")
                .with_field("a", TraceValue::U64(1))
                .with_field("b", TraceValue::U64(2));

            let result = event.try_for_each_field(|key, _| {
                if key == "b" {
                    Err("stop")
                } else {
                    Ok(())
                }
            });

            assert_eq!(result, Err("stop"));
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
