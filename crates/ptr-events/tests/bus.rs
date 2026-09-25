use ptr_events::{
    BusError, EventClass, EventConsumer, EventProducer, InMemoryBus, NewRecord, Offset,
};

fn record(key: &str) -> NewRecord {
    NewRecord {
        class: EventClass::Projection,
        topic: "capsule.committed".into(),
        key: key.into(),
        payload: key.as_bytes().to_vec(),
    }
}

#[test]
fn an_uncommitted_record_is_delivered_again() {
    let mut bus = InMemoryBus::new(8);
    bus.register("indexer", Offset(0));
    bus.publish(record("a")).unwrap();
    bus.publish(record("b")).unwrap();
    assert_eq!(bus.poll("indexer", 10).unwrap().len(), 2);
    assert_eq!(bus.poll("indexer", 10).unwrap().len(), 2);
    bus.commit("indexer", Offset(1)).unwrap();
    let rest = bus.poll("indexer", 10).unwrap();
    assert_eq!(rest.len(), 1);
    assert_eq!(rest[0].key, "b");
}

#[test]
fn a_slow_consumer_blocks_producers_instead_of_losing_records() {
    let mut bus = InMemoryBus::new(2);
    bus.register("slow", Offset(0));
    bus.register("fast", Offset(0));
    bus.publish(record("a")).unwrap();
    bus.publish(record("b")).unwrap();
    bus.commit("fast", Offset(2)).unwrap();
    assert_eq!(
        bus.publish(record("c")).unwrap_err(),
        BusError::Full { capacity: 2 }
    );
    bus.commit("slow", Offset(1)).unwrap();
    bus.publish(record("c")).unwrap();
    assert_eq!(bus.poll("slow", 10).unwrap().len(), 2);
}

#[test]
fn commits_are_monotone_and_bounded_by_the_end() {
    let mut bus = InMemoryBus::new(4);
    bus.register("c", Offset(0));
    bus.publish(record("a")).unwrap();
    bus.publish(record("b")).unwrap();
    bus.commit("c", Offset(2)).unwrap();
    bus.commit("c", Offset(1)).unwrap();
    assert!(bus.poll("c", 10).unwrap().is_empty());
    assert!(matches!(
        bus.commit("c", Offset(5)),
        Err(BusError::CommitBeyondEnd { .. })
    ));
    assert!(matches!(
        bus.poll("nobody", 1),
        Err(BusError::UnknownConsumer { .. })
    ));
}
