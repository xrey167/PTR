#![cfg(feature = "tracing-adapter")]

mod common;

use ptr_observe::{TraceSink, TracingSink};

#[test]
fn tracing_adapter_accepts_ptr_events_without_a_subscriber() {
    let event = common::generation_mismatch_event();
    TracingSink
        .emit(&event)
        .expect("tracing adapter must emit a valid PTR trace event");
}
