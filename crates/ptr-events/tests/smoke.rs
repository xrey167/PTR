use ptr_events::{EventEnvelope,RuntimeEvent}; use ptr_types::RequestId;
#[test] fn event_envelope_preserves_sequence(){ let e=EventEnvelope{sequence:7,event:RuntimeEvent::RequestStarted(RequestId::from("r"))}; assert_eq!(e.sequence,7); }
