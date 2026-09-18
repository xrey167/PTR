use ptr_config::PtrConfig;
use ptr_model_api::{ModelEvent, ReferenceEchoBackend};
use ptr_runtime::PtrRuntime;
use ptr_types::{RequestId, Revision};

#[test]
fn reference_model_loop_runs_against_semantic_revision() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let events = runtime
        .run_model_once(RequestId::from("r1"), "hello", &ReferenceEchoBackend)
        .unwrap();

    assert_eq!(runtime.revision(), Revision(1));
    assert_eq!(
        events,
        vec![ModelEvent::Token("hello".into()), ModelEvent::Finished]
    );
    assert!(runtime
        .events()
        .iter()
        .any(|event| matches!(&event.event, ptr_events::RuntimeEvent::RequestFinished(id) if id.to_string() == "r1")));
}
