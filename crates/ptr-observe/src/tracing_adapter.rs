use crate::{fields, TraceError, TraceEvent, TraceLevel, TraceSink};

#[derive(Clone, Copy, Debug, Default)]
pub struct TracingSink;

impl TraceSink for TracingSink {
    fn emit(&self, event: &TraceEvent) -> Result<(), TraceError> {
        let request_id = event.fields.get(fields::REQUEST_ID);
        let revision = event.fields.get(fields::REVISION);
        let generation = event.fields.get(fields::GENERATION);
        let backend_id = event.fields.get(fields::BACKEND_ID);
        let error_code = event.fields.get(fields::ERROR_CODE);
        let expected = event.fields.get(fields::EXPECTED);
        let actual = event.fields.get(fields::ACTUAL);

        match event.level {
            TraceLevel::Error => tracing::error!(
                ptr_event = %event.name,
                request_id = ?request_id,
                revision = ?revision,
                generation = ?generation,
                backend_id = ?backend_id,
                error_code = ?error_code,
                expected = ?expected,
                actual = ?actual,
                fields = ?event.fields
            ),
            TraceLevel::Warn => tracing::warn!(
                ptr_event = %event.name,
                request_id = ?request_id,
                revision = ?revision,
                generation = ?generation,
                backend_id = ?backend_id,
                error_code = ?error_code,
                expected = ?expected,
                actual = ?actual,
                fields = ?event.fields
            ),
            TraceLevel::Info => tracing::info!(
                ptr_event = %event.name,
                request_id = ?request_id,
                revision = ?revision,
                generation = ?generation,
                backend_id = ?backend_id,
                error_code = ?error_code,
                expected = ?expected,
                actual = ?actual,
                fields = ?event.fields
            ),
            TraceLevel::Debug => tracing::debug!(
                ptr_event = %event.name,
                request_id = ?request_id,
                revision = ?revision,
                generation = ?generation,
                backend_id = ?backend_id,
                error_code = ?error_code,
                expected = ?expected,
                actual = ?actual,
                fields = ?event.fields
            ),
            TraceLevel::Trace => tracing::trace!(
                ptr_event = %event.name,
                request_id = ?request_id,
                revision = ?revision,
                generation = ?generation,
                backend_id = ?backend_id,
                error_code = ?error_code,
                expected = ?expected,
                actual = ?actual,
                fields = ?event.fields
            ),
        }
        Ok(())
    }
}
