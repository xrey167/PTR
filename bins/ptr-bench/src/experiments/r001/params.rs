//! The parameters of R001, compiled in.
//!
//! They mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`, which is read
//! at compile time; the drift test fails when a constant differs from the file or the file holds a
//! key no constant mirrors. R001 is frozen at its preregistration and has completed; the file is
//! frozen with it.

// The drift test and the echo of the procedure use items the harness itself does not read.
#![cfg_attr(not(test), allow(dead_code))]

pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R001-mailbox-backpressure/config.toml");
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/runtime/R001-mailbox-backpressure/experiment.toml");
pub const DRAFT_TEXT: &str =
    include_str!("../../../../../experiments/runtime/R001-mailbox-backpressure/PREREGISTRATION.md");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "mailbox-backpressure";
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- mailbox-backpressure <seed>";
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const DECLARED_SEEDS: [u64; 5] = [17, 29, 43, 71, 101];

/// The mailbox capacities of the sweep.
pub const CAPACITIES: [usize; 3] = [1, 8, 128];

/// A payload class: a name and the bytes of one payload body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayloadClass {
    pub name: &'static str,
    pub bytes: usize,
}
pub const PAYLOAD_CLASSES: [PayloadClass; 2] = [
    PayloadClass {
        name: "small",
        bytes: 64,
    },
    PayloadClass {
        name: "large",
        bytes: 4096,
    },
];

/// Service time of one message, in ticks, drawn uniformly from `SERVICE_MIN..=SERVICE_MAX`.
pub const SERVICE_MIN: u64 = 16;
pub const SERVICE_MAX: u64 = 32;

/// The analytic wait bound of a bounded mailbox, in ticks: an admitted message has at most
/// `capacity - 1` messages ahead of it in the mailbox and one in service with at most
/// `SERVICE_MAX` ticks left.
pub const fn wait_bound(capacity: usize) -> u64 {
    capacity as u64 * SERVICE_MAX
}

/// The payload bytes that may be alive between events: `capacity` queued and one in service.
pub const fn payload_bound(capacity: usize, body_bytes: usize) -> i64 {
    ((capacity + 1) * body_bytes) as i64
}

/// One arrival phase: `count` arrivals whose gaps are drawn uniformly from `gap_lo..=gap_hi`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseSpec {
    pub name: &'static str,
    pub count: u32,
    pub gap_lo: u64,
    pub gap_hi: u64,
}

impl PhaseSpec {
    /// Whether arrivals outpace the service on average (mean gap below mean service time),
    /// decided in integers: `gap_lo + gap_hi < SERVICE_MIN + SERVICE_MAX`.
    pub const fn overload(&self) -> bool {
        self.gap_lo + self.gap_hi < SERVICE_MIN + SERVICE_MAX
    }
}

pub const PHASES: [PhaseSpec; 5] = [
    PhaseSpec {
        name: "underload",
        count: 200,
        gap_lo: 33,
        gap_hi: 40,
    },
    PhaseSpec {
        name: "overload_x2",
        count: 400,
        gap_lo: 8,
        gap_hi: 16,
    },
    PhaseSpec {
        name: "overload_x8",
        count: 800,
        gap_lo: 2,
        gap_hi: 4,
    },
    PhaseSpec {
        name: "burst",
        count: 300,
        gap_lo: 1,
        gap_hi: 1,
    },
    PhaseSpec {
        name: "recovery",
        count: 600,
        gap_lo: 33,
        gap_hi: 40,
    },
];
pub const UNDERLOAD_PHASE: usize = 0;
pub const RECOVERY_PHASE: usize = 4;

/// The closed-path schedule: a short underload, then x8 arrivals during which the mailbox is
/// closed just before arrival number `CLOSE_BEFORE_ID`.
pub const CLOSED_PHASES: [PhaseSpec; 2] = [
    PhaseSpec {
        name: "underload",
        count: 10,
        gap_lo: 33,
        gap_hi: 40,
    },
    PhaseSpec {
        name: "overload_x8",
        count: 140,
        gap_lo: 2,
        gap_hi: 4,
    },
];
pub const CLOSE_BEFORE_ID: u32 = 80;

/// The underload-only schedule that checks the oracle against the reference queue.
pub const SANITY_PHASES: [PhaseSpec; 1] = [PHASES[UNDERLOAD_PHASE]];

/// The threaded arm: producers and per-producer message counts of its three rounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedShape {
    pub producers: usize,
    pub capacity: usize,
    /// Stage 1: the number of exact fills (the consumer is parked, so each admits exactly
    /// `capacity` messages in total); repeated so that a race in the capacity check shows.
    pub fill_rounds: u32,
    /// Stage 1: the messages each producer offers in one fill.
    pub fill_per_producer: u32,
    /// Stage 2: the consumer drains while the producers offer.
    pub contend_per_producer: u32,
    /// Stage 3: after `close()`, every offer is returned as `Closed`.
    pub closed_per_producer: u32,
    /// The consumer closes the mailbox once it has received this many messages in round 2.
    pub close_after_received: u32,
}
pub const THREADED_PINNED: ThreadedShape = ThreadedShape {
    producers: 4,
    capacity: 8,
    fill_rounds: 2_000,
    fill_per_producer: 16,
    contend_per_producer: 20_000,
    closed_per_producer: 64,
    close_after_received: 10_000,
};
/// Used by the smoke test only.
pub const THREADED_REDUCED: ThreadedShape = ThreadedShape {
    fill_rounds: 100,
    contend_per_producer: 1_000,
    close_after_received: 500,
    ..THREADED_PINNED
};

/// The cell the negative controls and canaries run on.
pub const CONTROL_CAPACITY: usize = 8;

/// Total messages of one schedule.
pub const fn message_count(phases: &[PhaseSpec]) -> u32 {
    let mut total = 0;
    let mut at = 0;
    while at < phases.len() {
        total += phases[at].count;
        at += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(path: &[&str]) -> toml::Value {
        let config: toml::Table = CONFIG_TOML.parse().expect("config.toml parses");
        let mut at = toml::Value::Table(config);
        for key in path {
            at = at
                .get(key)
                .unwrap_or_else(|| panic!("config.toml has no {path:?}"))
                .clone();
        }
        at
    }

    fn int(path: &[&str]) -> i64 {
        table(path).as_integer().expect("an integer")
    }

    #[test]
    fn the_compiled_parameters_are_the_pinned_thresholds_of_config_toml() {
        let pinned = table(&["pinned_thresholds"]);
        let capacities: Vec<usize> = pinned["capacities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_integer().unwrap() as usize)
            .collect();
        assert_eq!(capacities, CAPACITIES);
        for class in PAYLOAD_CLASSES {
            assert_eq!(
                pinned["payload_class_bytes"][class.name].as_integer(),
                Some(class.bytes as i64)
            );
        }
        assert_eq!(
            pinned["payload_class_bytes"].as_table().unwrap().len(),
            PAYLOAD_CLASSES.len()
        );
        assert_eq!(
            int(&["pinned_thresholds", "service_ticks_min"]),
            SERVICE_MIN as i64
        );
        assert_eq!(
            int(&["pinned_thresholds", "service_ticks_max"]),
            SERVICE_MAX as i64
        );
        let phases = pinned["phases"].as_array().unwrap();
        assert_eq!(phases.len(), PHASES.len());
        for (row, phase) in phases.iter().zip(PHASES) {
            assert_eq!(row["name"].as_str(), Some(phase.name));
            assert_eq!(row["count"].as_integer(), Some(i64::from(phase.count)));
            assert_eq!(row["gap_lo"].as_integer(), Some(phase.gap_lo as i64));
            assert_eq!(row["gap_hi"].as_integer(), Some(phase.gap_hi as i64));
        }
        let closed = pinned["closed_phases"].as_array().unwrap();
        assert_eq!(closed.len(), CLOSED_PHASES.len());
        for (row, phase) in closed.iter().zip(CLOSED_PHASES) {
            assert_eq!(row["name"].as_str(), Some(phase.name));
            assert_eq!(row["count"].as_integer(), Some(i64::from(phase.count)));
            assert_eq!(row["gap_lo"].as_integer(), Some(phase.gap_lo as i64));
            assert_eq!(row["gap_hi"].as_integer(), Some(phase.gap_hi as i64));
        }
        assert_eq!(
            int(&["pinned_thresholds", "close_before_id"]),
            i64::from(CLOSE_BEFORE_ID)
        );
        assert_eq!(
            int(&["pinned_thresholds", "control_capacity"]),
            CONTROL_CAPACITY as i64
        );
        let threaded = table(&["pinned_thresholds", "threaded"]);
        let shape = THREADED_PINNED;
        for (key, value) in [
            ("producers", shape.producers as i64),
            ("capacity", shape.capacity as i64),
            ("fill_rounds", i64::from(shape.fill_rounds)),
            ("fill_per_producer", i64::from(shape.fill_per_producer)),
            (
                "contend_per_producer",
                i64::from(shape.contend_per_producer),
            ),
            ("closed_per_producer", i64::from(shape.closed_per_producer)),
            (
                "close_after_received",
                i64::from(shape.close_after_received),
            ),
        ] {
            assert_eq!(threaded[key].as_integer(), Some(value), "{key}");
        }
        assert_eq!(threaded.as_table().unwrap().len(), 7);
        assert_eq!(pinned["seed_aggregation"].as_str(), Some(SEED_AGGREGATION));
        // Every key of the table is one of those checked above.
        assert_eq!(pinned.as_table().unwrap().len(), 10);
    }

    #[test]
    fn the_schedule_is_what_the_analytic_bounds_assume() {
        const { assert!(SERVICE_MIN >= 1 && SERVICE_MIN <= SERVICE_MAX) };
        // Underload: the gap exceeds the longest service, so the queue never builds.
        assert!(PHASES[UNDERLOAD_PHASE].gap_lo > SERVICE_MAX);
        assert!(!PHASES[UNDERLOAD_PHASE].overload());
        assert!(!PHASES[RECOVERY_PHASE].overload());
        for phase in &PHASES[1..4] {
            assert!(phase.overload(), "{}", phase.name);
        }
        assert!(PHASES.iter().all(|p| p.gap_lo >= 1 && p.gap_lo <= p.gap_hi));
        assert_eq!(message_count(&PHASES), 2300);
        assert!(CLOSE_BEFORE_ID < message_count(&CLOSED_PHASES));
        assert!(CLOSE_BEFORE_ID >= CLOSED_PHASES[0].count);
        assert!(CAPACITIES.contains(&CONTROL_CAPACITY));
        assert!(
            u64::from(THREADED_PINNED.fill_per_producer) * THREADED_PINNED.producers as u64
                > THREADED_PINNED.capacity as u64,
            "round 1 must offer more than the capacity"
        );
    }
}
