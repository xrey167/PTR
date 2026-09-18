#![cfg(feature = "tracing-adapter")]

use ptr_observe::{TraceEvent, TraceLevel, TraceSink, TracingSink};

#[test]
fn tracing_adapter_accepts_ptr_events_without_a_subscriber() {
    let event = TraceEvent::new(TraceLevel::Info, "ptr.test");
    TracingSink
        .emit(&event)
        .expect("tracing event emission is infallible");
}
