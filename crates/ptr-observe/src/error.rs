use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceError {
    SinkUnavailable {
        sink: String,
    },
    RejectedField {
        field: String,
    },
    Export {
        sink: String,
        message: String,
    },
}

impl TraceError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::SinkUnavailable { .. } => "sink_unavailable",
            Self::RejectedField { .. } => "rejected_field",
            Self::Export { .. } => "export",
        }
    }
}

impl fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SinkUnavailable { sink } => write!(formatter, "trace sink unavailable: {sink}"),
            Self::RejectedField { field } => write!(formatter, "trace field rejected: {field}"),
            Self::Export { sink, message } => {
                write!(formatter, "trace export failed for {sink}: {message}")
            }
        }
    }
}

impl std::error::Error for TraceError {}
