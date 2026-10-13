use std::sync::Arc;

use ptr_observe::{
    FanoutSink, MemorySink, RedactingSink, SamplingRates, SamplingSink, TraceEvent, TraceLevel,
    TraceSink,
};

fn emit_all(sink: &dyn TraceSink) {
    for i in 0..20 {
        for level in [
            TraceLevel::Error,
            TraceLevel::Warn,
            TraceLevel::Info,
            TraceLevel::Debug,
            TraceLevel::Trace,
        ] {
            sink.emit(&TraceEvent::new(level, format!("{level:?}-{i}")))
                .expect("emit");
        }
    }
}

#[test]
fn sampling_keeps_first_of_each_n_per_level() {
    let rates = SamplingRates::KEEP_ALL
        .with_rate(TraceLevel::Info, 5)
        .with_rate(TraceLevel::Debug, 10)
        .with_rate(TraceLevel::Trace, 0);
    let sink = SamplingSink::new(MemorySink::new(1024), rates);
    emit_all(&sink);
    let names: Vec<String> = sink.inner().events().into_iter().map(|e| e.name).collect();
    let of = |prefix: &str| names.iter().filter(|n| n.starts_with(prefix)).count();
    assert_eq!(of("Error"), 20);
    assert_eq!(of("Warn"), 20);
    assert_eq!(of("Info"), 4);
    assert_eq!(of("Debug"), 2);
    assert_eq!(of("Trace"), 0);
    assert!(names.contains(&"Info-0".to_owned()) && names.contains(&"Info-15".to_owned()));
    assert_eq!(sink.sampled_out(), 16 + 18 + 20);
}

#[test]
fn sampling_is_reproducible_across_instances() {
    let make = || {
        let rates = SamplingRates::KEEP_ALL.with_rate(TraceLevel::Debug, 3);
        let sink = SamplingSink::new(MemorySink::new(1024), rates);
        emit_all(&sink);
        sink.inner().events()
    };
    assert_eq!(make(), make());
}

#[test]
fn composed_pipeline_redacts_then_samples_into_fanout() {
    let ring = Arc::new(MemorySink::new(3));
    let all = Arc::new(MemorySink::new(100));
    let fanout = FanoutSink::new()
        .with_sink(ring.clone())
        .with_sink(all.clone());
    let pipeline = RedactingSink::new(SamplingSink::new(fanout, SamplingRates::KEEP_ALL));
    for i in 0..5u64 {
        pipeline
            .emit(&TraceEvent::new(TraceLevel::Info, format!("e{i}")).with_field("token", "abc"))
            .expect("emit");
    }
    assert_eq!(all.len(), 5);
    assert_eq!(ring.len(), 3);
    assert_eq!(ring.dropped(), 2);
    assert!(!format!("{:?}", all.events()).contains("\"abc\""));
}
