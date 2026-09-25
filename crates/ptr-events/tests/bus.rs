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

#[test]
fn a_refused_publish_does_not_consume_an_offset_or_replace_the_pending_record() {
    let mut bus = InMemoryBus::new(0);
    bus.register("worker", Offset(0));
    assert_eq!(bus.publish(record("first")), Ok(Offset(1)));
    assert_eq!(
        bus.publish(record("refused")),
        Err(BusError::Full { capacity: 1 })
    );
    let pending = bus.poll("worker", 10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].key, "first");
    bus.commit("worker", Offset(1)).unwrap();
    assert_eq!(bus.retained(), 0);
    assert_eq!(bus.publish(record("retry")), Ok(Offset(2)));
}

#[test]
fn polling_is_bounded_and_preserves_record_contents_without_acknowledging() {
    let mut bus = InMemoryBus::new(3);
    bus.register("worker", Offset(0));
    let telemetry = NewRecord {
        class: EventClass::Telemetry,
        topic: "execution.failed".into(),
        key: "request:1".into(),
        payload: vec![0, 255, 1],
    };
    bus.publish(telemetry.clone()).unwrap();
    bus.publish(record("second")).unwrap();
    assert!(bus.poll("worker", 0).unwrap().is_empty());
    let first = bus.poll("worker", 1).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].offset, Offset(1));
    assert_eq!(first[0].class, telemetry.class);
    assert_eq!(first[0].topic, telemetry.topic);
    assert_eq!(first[0].key, telemetry.key);
    assert_eq!(first[0].payload, telemetry.payload);
    assert_eq!(bus.poll("worker", 1).unwrap(), first);
    assert_eq!(bus.retained(), 2);
}

#[test]
fn invalid_commits_leave_the_consumer_and_retention_unchanged() {
    let mut bus = InMemoryBus::new(2);
    bus.register("worker", Offset(0));
    bus.publish(record("a")).unwrap();
    let before = bus.poll("worker", 2).unwrap();
    assert_eq!(
        bus.commit("unknown", Offset(1)),
        Err(BusError::UnknownConsumer {
            consumer: "unknown".into(),
        })
    );
    assert_eq!(
        bus.commit("worker", Offset(2)),
        Err(BusError::CommitBeyondEnd {
            requested: Offset(2),
            end: Offset(1),
        })
    );
    assert_eq!(bus.poll("worker", 2).unwrap(), before);
    assert_eq!(bus.retained(), 1);
    bus.commit("worker", Offset(1)).unwrap();
    assert_eq!(bus.retained(), 0);
}
