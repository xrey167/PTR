#![no_main]

use libfuzzer_sys::fuzz_target;
use prost::Message;
use ptr_protocol::generated::podwire::PodCall;
use ptr_protocol::CallFrame;

fuzz_target!(|data: &[u8]| {
    if let Ok(call) = PodCall::decode(data) {
        let _ = CallFrame::try_from(call);
    }
});
