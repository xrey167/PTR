use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceError {
    SinkUnavailable { sink: String },
    RejectedField { field: String },
    Export { sink: String, message: String },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_errors_expose_stable_named_codes() {
        let cases = [
            (
                TraceError::SinkUnavailable {
                    sink: "test".into(),
                },
                "sink_unavailable",
            ),
            (
                TraceError::RejectedField {
                    field: "secret".into(),
                },
                "rejected_field",
            ),
            (
                TraceError::Export {
                    sink: "test".into(),
                    message: "failed".into(),
                },
                "export",
            ),
        ];

        for (error, expected_code) in cases {
            assert_eq!(error.code(), expected_code);
        }
    }

    #[test]
    fn export_error_display_contains_named_sink_and_message() {
        let error = TraceError::Export {
            sink: "otel".into(),
            message: "collector unavailable".into(),
        };

        let message = error.to_string();
        assert!(message.contains("otel"));
        assert!(message.contains("collector unavailable"));
    }
}
