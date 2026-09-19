use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceError {
    SinkUnavailable { sink: String },
    Export { sink: String, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceEvent {
    pub level: TraceLevel,
    pub name: String,
    pub fields: BTreeMap<String, String>,
}

impl TraceEvent {
    pub fn new(level: TraceLevel, name: impl Into<String>) -> Self {
        Self {
            level,
            name: name.into(),
            fields: BTreeMap::new(),
        }
    }

    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
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
