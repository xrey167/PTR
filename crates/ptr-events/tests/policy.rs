mod common;

use common::{keys, projection, telemetry};
use ptr_events::{
    Authority, BackpressurePolicy, BusConfig, BusError, ClassPolicies, EventClass, EventConsumer,
    EventProducer, InMemoryBus, Limit, NewRecord, Offset, PolicyViolation, RetentionPolicy,
};

fn dropping(capacity: usize) -> InMemoryBus {
    let mut config = BusConfig::new(capacity);
    config.policies = ClassPolicies::drop_oldest_telemetry();
    InMemoryBus::with_config(config)
}

#[test]
fn records_expose_their_authority() {
    let mut bus = InMemoryBus::new(4);
    bus.register("c", Offset(0)).unwrap();
    bus.publish(projection("p")).unwrap();
    bus.publish(telemetry("t")).unwrap();
    let seen = bus.poll("c", 10).unwrap();
    assert_eq!(seen[0].authority, Authority::Authoritative);
    assert_eq!(seen[1].authority, Authority::Diagnostic);
}

#[test]
fn a_telemetry_record_cannot_be_republished_as_authoritative() {
    let mut bus = InMemoryBus::new(4);
    bus.register("c", Offset(0)).unwrap();
    bus.publish(telemetry("t")).unwrap();
    let seen = bus.poll("c", 1).unwrap().remove(0);
    assert_eq!(
        NewRecord::reclassify(&seen, EventClass::Projection),
        Err(BusError::TelemetryPromotion { offset: Offset(1) })
    );
    let again = bus.publish(NewRecord::republish(&seen)).unwrap();
    let copy = bus.poll("c", 10).unwrap().pop().unwrap();
    assert_eq!(copy.offset, again);
    assert_eq!(copy.authority, Authority::Diagnostic);
}

#[test]
fn a_projection_class_cannot_be_given_a_dropping_policy() {
    assert_eq!(
        ClassPolicies::new(
            BackpressurePolicy::DropOldestTelemetry,
            BackpressurePolicy::Reject
        ),
        Err(BusError::InvalidPolicy {
            violation: PolicyViolation::ProjectionMayNotDrop
        })
    );
}

#[test]
fn reject_reports_capacity_depth_and_the_blocking_consumer() {
    let mut bus = InMemoryBus::new(2);
    bus.register("fast", Offset(0)).unwrap();
    bus.register("slow", Offset(0)).unwrap();
    bus.publish(projection("a")).unwrap();
    bus.publish(telemetry("b")).unwrap();
    bus.commit("fast", Offset(2)).unwrap();
    assert_eq!(
        bus.publish(telemetry("c")),
        Err(BusError::Full {
            limit: Limit::Records,
            capacity: 2,
            depth: 2,
            blocked_by: Some("slow".into()),
        })
    );
    let blocker = bus.blocked_by().unwrap();
    assert_eq!((blocker.consumer.as_str(), blocker.lag), ("slow", 2));
    assert_eq!(bus.lag("fast"), Ok(0));
    assert_eq!(bus.lag("slow"), Ok(2));
    // Nothing was lost: the slow consumer still gets both records.
    assert_eq!(keys(&bus.poll("slow", 10).unwrap()), ["a", "b"]);
}

#[test]
fn drop_oldest_telemetry_counts_drops_per_consumer() {
    let mut bus = dropping(3);
    bus.register("a", Offset(0)).unwrap();
    bus.register("b", Offset(0)).unwrap();
    bus.publish(telemetry("t1")).unwrap();
    bus.publish(projection("p1")).unwrap();
    bus.publish(telemetry("t2")).unwrap();
    // b has acknowledged t1; a has not. Depth counts from the slowest consumer (a).
    bus.commit("b", Offset(1)).unwrap();
    bus.publish(projection("p2")).unwrap();
    assert_eq!(keys(&bus.poll("a", 10).unwrap()), ["p1", "t2", "p2"]);
    assert_eq!(bus.dropped("a"), Ok(1));
    assert_eq!(bus.dropped("b"), Ok(0));
    assert_eq!(bus.total_dropped(), 1);
    bus.publish(projection("p3")).unwrap();
    assert_eq!(bus.dropped("a"), Ok(2));
    assert_eq!(bus.dropped("b"), Ok(1));
    assert_eq!(keys(&bus.poll("b", 10).unwrap()), ["p1", "p2", "p3"]);
}

#[test]
fn authoritative_records_are_never_dropped_even_under_a_dropping_policy() {
    let mut bus = dropping(2);
    bus.register("c", Offset(0)).unwrap();
    bus.publish(projection("p1")).unwrap();
    bus.publish(projection("p2")).unwrap();
    // No telemetry to evict: both a telemetry and a projection publish are rejected.
    assert!(matches!(
        bus.publish(telemetry("t")),
        Err(BusError::Full { .. })
    ));
    assert!(matches!(
        bus.publish(projection("p3")),
        Err(BusError::Full { .. })
    ));
    assert_eq!(bus.total_dropped(), 0);
    assert_eq!(keys(&bus.poll("c", 10).unwrap()), ["p1", "p2"]);
    // After an acknowledgement the space is reusable.
    bus.commit("c", Offset(1)).unwrap();
    bus.publish(projection("p3")).unwrap();
}

#[test]
fn a_projection_publish_may_evict_telemetry_but_never_the_reverse() {
    let mut bus = dropping(2);
    bus.register("c", Offset(0)).unwrap();
    bus.publish(telemetry("t")).unwrap();
    bus.publish(projection("p1")).unwrap();
    bus.publish(projection("p2")).unwrap();
    assert_eq!(keys(&bus.poll("c", 10).unwrap()), ["p1", "p2"]);
    assert_eq!(bus.dropped("c"), Ok(1));
    assert_eq!(bus.low_water(), Offset(1));
}

#[test]
fn a_byte_bound_applies_backpressure_in_bytes() {
    let mut config = BusConfig::new(100);
    config.capacity_bytes = Some(40);
    let mut bus = InMemoryBus::with_config(config);
    bus.register("c", Offset(0)).unwrap();
    // topic 17 + key 1 + payload 1 = 19 bytes each.
    bus.publish(projection("a")).unwrap();
    bus.publish(projection("b")).unwrap();
    assert_eq!(
        bus.publish(projection("c")),
        Err(BusError::Full {
            limit: Limit::Bytes,
            capacity: 40,
            depth: 38,
            blocked_by: Some("c".into()),
        })
    );
}

#[test]
fn retention_by_record_count_keeps_acknowledged_history_for_replay() {
    let mut config = BusConfig::new(10);
    config.retention = RetentionPolicy::with_max_records(2);
    let mut bus = InMemoryBus::with_config(config);
    bus.register("c", Offset(0)).unwrap();
    for key in ["a", "b", "c", "d", "e"] {
        bus.publish(projection(key)).unwrap();
    }
    bus.commit("c", Offset(5)).unwrap();
    assert_eq!(bus.retained(), 2);
    assert_eq!(bus.low_water(), Offset(3));
    assert_eq!(keys(&bus.replay(Offset(3), 10).unwrap()), ["d", "e"]);
    assert_eq!(
        bus.replay(Offset(2), 10),
        Err(BusError::Truncated {
            requested: Offset(2),
            low_water: Offset(3)
        })
    );
}

#[test]
fn retention_by_bytes() {
    let mut config = BusConfig::new(10);
    config.retention = RetentionPolicy::with_max_bytes(40);
    let mut bus = InMemoryBus::with_config(config);
    bus.register("c", Offset(0)).unwrap();
    for key in ["a", "b", "c", "d"] {
        bus.publish(projection(key)).unwrap();
    }
    bus.commit("c", Offset(4)).unwrap();
    // 19 bytes per record: two fit in 40.
    assert_eq!(bus.retained(), 2);
    assert_eq!(bus.retained_bytes(), 38);
    assert_eq!(bus.low_water(), Offset(2));
}

#[test]
fn retention_by_logical_age_uses_the_caller_clock() {
    let mut config = BusConfig::new(10);
    config.retention = RetentionPolicy::with_max_age_ticks(5);
    let mut bus = InMemoryBus::with_config(config);
    bus.register("c", Offset(0)).unwrap();
    bus.publish(projection("old")).unwrap();
    bus.advance_tick(4).unwrap();
    bus.publish(projection("new")).unwrap();
    bus.commit("c", Offset(2)).unwrap();
    assert_eq!(bus.retained(), 2);
    let report = bus.advance_tick(7).unwrap();
    assert_eq!(report.removed_records, 1);
    assert_eq!(report.low_water, Offset(1));
    assert_eq!(keys(&bus.replay(Offset(1), 10).unwrap()), ["new"]);
    assert_eq!(
        bus.advance_tick(3),
        Err(BusError::ClockRegression {
            current: 7,
            requested: 3
        })
    );
}

#[test]
fn retention_never_removes_an_unacknowledged_authoritative_record() {
    let mut config = BusConfig::new(10);
    config.retention = RetentionPolicy {
        max_records: Some(0),
        max_bytes: Some(0),
        max_age_ticks: Some(0),
    };
    let mut bus = InMemoryBus::with_config(config);
    bus.register("slow", Offset(0)).unwrap();
    bus.publish(projection("a")).unwrap();
    bus.publish(projection("b")).unwrap();
    bus.advance_tick(1_000_000).unwrap();
    bus.compact();
    assert_eq!(bus.retained(), 2);
    assert_eq!(bus.low_water(), Offset(0));
    assert_eq!(keys(&bus.poll("slow", 10).unwrap()), ["a", "b"]);
    bus.commit("slow", Offset(1)).unwrap();
    assert_eq!(keys(&bus.poll("slow", 10).unwrap()), ["b"]);
    assert_eq!(bus.low_water(), Offset(1));
}

#[test]
fn registering_below_the_low_water_mark_is_a_typed_truncation() {
    let mut bus = InMemoryBus::new(4);
    bus.register("a", Offset(0)).unwrap();
    bus.publish(projection("x")).unwrap();
    bus.publish(projection("y")).unwrap();
    bus.commit("a", Offset(2)).unwrap();
    assert_eq!(bus.low_water(), Offset(2));
    assert_eq!(
        bus.register("late", Offset(0)),
        Err(BusError::Truncated {
            requested: Offset(0),
            low_water: Offset(2)
        })
    );
    bus.register("late", bus.low_water()).unwrap();
}

#[test]
fn consumers_have_independent_offsets_and_ordering_is_preserved() {
    let mut bus = InMemoryBus::new(8);
    bus.register("a", Offset(0)).unwrap();
    bus.register("b", Offset(0)).unwrap();
    for key in ["1", "2", "3", "4"] {
        bus.publish(projection(key)).unwrap();
    }
    bus.commit("a", Offset(3)).unwrap();
    bus.commit("b", Offset(1)).unwrap();
    assert_eq!(keys(&bus.poll("a", 10).unwrap()), ["4"]);
    assert_eq!(keys(&bus.poll("b", 10).unwrap()), ["2", "3", "4"]);
    assert_eq!(bus.committed("a"), Ok(Offset(3)));
    assert_eq!(bus.committed("b"), Ok(Offset(1)));
    assert_eq!(bus.blocked_by().unwrap().consumer, "b");
}
