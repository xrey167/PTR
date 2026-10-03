//! The counters of a run, in the three groups the preregistration names, and
//! the histogram of wall time a merge takes.
//!
//! A counter is a field, so a name that is not one fails to compile, and a
//! field is in exactly one group, so a hard counter cannot be counted as
//! coverage. The groups are also listed by name for the result's JSON.

use std::time::Duration;

macro_rules! metrics {
    (
        hard { $($hard:ident),* $(,)? }
        coverage { $($coverage:ident),* $(,)? }
        descriptive { $($descriptive:ident),* $(,)? }
    ) => {
        /// Everything a run counts.
        #[derive(Clone, Debug, Default, Eq, PartialEq)]
        pub struct Metrics {
            $(pub $hard: u64,)*
            $(pub $coverage: u64,)*
            $(pub $descriptive: u64,)*
            /// Wall time inside `merge_branch`, in nanoseconds, and the calls.
            pub merge_wall_ns: u128,
            pub merge_calls: u64,
            pub merge_wall_le_10us: u64,
            pub merge_wall_le_100us: u64,
            pub merge_wall_le_1ms: u64,
            pub merge_wall_le_10ms: u64,
            pub merge_wall_gt_10ms: u64,
        }

        /// The hard counters' names, in the order the result lists them.
        pub const HARD: &[&str] = &[$(stringify!($hard)),*];
        /// The coverage counters' names.
        pub const COVERAGE: &[&str] = &[$(stringify!($coverage)),*];
        /// The descriptive counters' names.
        pub const DESCRIPTIVE: &[&str] = &[$(stringify!($descriptive)),*];

        impl Metrics {
            pub fn hard(&self) -> [(&'static str, u64); HARD.len()] {
                [$((stringify!($hard), self.$hard)),*]
            }

            pub fn coverage(&self) -> [(&'static str, u64); COVERAGE.len()] {
                [$((stringify!($coverage), self.$coverage)),*]
            }

            pub fn descriptive(&self) -> [(&'static str, u64); DESCRIPTIVE.len()] {
                [$((stringify!($descriptive), self.$descriptive)),*]
            }

            /// Add every counter of `other` to this one.
            pub fn absorb(&mut self, other: &Metrics) {
                $(self.$hard += other.$hard;)*
                $(self.$coverage += other.$coverage;)*
                $(self.$descriptive += other.$descriptive;)*
                self.merge_wall_ns += other.merge_wall_ns;
                self.merge_calls += other.merge_calls;
                self.merge_wall_le_10us += other.merge_wall_le_10us;
                self.merge_wall_le_100us += other.merge_wall_le_100us;
                self.merge_wall_le_1ms += other.merge_wall_le_1ms;
                self.merge_wall_le_10ms += other.merge_wall_le_10ms;
                self.merge_wall_gt_10ms += other.merge_wall_gt_10ms;
            }
        }
    };
}

metrics! {
    hard {
        lost_updates,
        stale_read_merges,
        undetected_phantoms,
        stale_scan_merges,
        stale_input_merges,
        stale_reliance_merges,
        lost_increments,
        serialization_divergences,
        invariant_violations,
        spurious_refusals,
        refusal_kind_mismatches,
        conflict_key_errors,
        rebase_classification_errors,
        ungated_commits,
        approval_bypasses,
        double_merges,
        bypass_commits_accepted,
        reserved_writes_accepted,
        seal_invariant_failures,
        undeclared_input_merges,
        forged_histories_accepted,
        provenance_mismatches,
        model_divergences,
        program_mirror_divergences,
        replay_divergences,
        canary_misses,
        nondeterminism,
        harness_errors,
    }
    coverage {
        merges_clean,
        merges_rebased,
        conflicts,
        lifecycle_refusals,
        escalations,
        reviewed_merges,
        no_change_merges,
        probe_verification_holds,
        hazard_write,
        hazard_read,
        hazard_scan_keys,
        hazard_scan_values,
        hazard_inputs,
        hazard_lifecycle,
        hazard_rebase,
        lww_lost_updates,
        lww_lost_increments,
        occ_undetected_phantoms,
        occ_stale_scan_commits,
        occ_stale_input_commits,
        occ_stale_reliance_commits,
        probe_p1_exercised,
        probe_p2_exercised,
        probe_p3_exercised,
        probe_p4_exercised,
        probe_p5_exercised,
        probe_p6_exercised,
        probe_p7_exercised,
        probe_p8_exercised,
        probe_p9_exercised,
        probe_p10_exercised,
        probe_p11_exercised,
        probe_p12_exercised,
        probe_p13_exercised,
        probe_p14_exercised,
        probe_p15_exercised,
        probe_p16_exercised,
        probe_p17_exercised,
        probe_p18_exercised,
        probe_p19_exercised,
        probe_p20_exercised,
        probe_p21_exercised,
        probe_p22_exercised,
        probe_p23_exercised,
        probe_p24_exercised,
        probe_p25_exercised,
        probe_p26_exercised,
        canaries_run,
        replays,
        durable_roundtrips,
        compaction_roundtrips,
    }
    descriptive {
        unnecessary_refusals,
        put_increment_conflicts,
        conflict_key_set_differs,
        wasted_ticks,
        abandoned_tasks,
        review_voids,
        verification_holds,
        occ_lost_updates,
        hazard_negative,
        hazard_set_member,
        probe_unnecessary_refusals,
        probe_put_increment_conflicts,
        probe_conflict_key_set_differs,
        probe_hazard_write,
        probe_hazard_read,
        probe_hazard_scan_keys,
        probe_hazard_scan_values,
        probe_hazard_inputs,
        probe_hazard_lifecycle,
        probe_hazard_rebase,
        probe_hazard_negative,
        probe_hazard_set_member,
        probe_merges_clean,
        probe_merges_rebased,
        probe_conflicts,
        probe_lifecycle_refusals,
        probe_escalations,
        probe_reviewed_merges,
        probe_no_change_merges,
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over `bytes`, continuing `state`; `None` starts a new digest.
pub fn fnv(state: Option<u64>, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(state.unwrap_or(FNV_OFFSET), |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        })
}

impl Metrics {
    /// Every deterministic counter, in result and digest order.
    pub fn counters(&self) -> impl Iterator<Item = (&'static str, u64)> {
        self.hard()
            .into_iter()
            .chain(self.coverage())
            .chain(self.descriptive())
    }

    /// `state` continued over every counter's name and value, and over
    /// nothing that depends on the wall clock.
    pub fn digest_into(&self, mut state: u64) -> u64 {
        for (name, count) in self.counters() {
            state = fnv(Some(state), name.as_bytes());
            state = fnv(Some(state), &count.to_le_bytes());
        }
        state
    }

    /// Add every counter of a case's probes to this one, but count the hazards
    /// their fixed scenarios ran into as `probe_hazard_*`: a probe repeats the
    /// same construction in every case, so it is coverage, and only what the
    /// workload's own merges ran into (`hazard_*`) counts as trials of
    /// certification. The refusals of a probe's world are counted apart too
    /// (`probe_*`): the shares of refusals the result reports are shares of
    /// the arms' refusals.
    pub fn absorb_probes(&mut self, probes: &Metrics) {
        let mut rest = probes.clone();
        self.probe_unnecessary_refusals += std::mem::take(&mut rest.unnecessary_refusals);
        self.probe_put_increment_conflicts += std::mem::take(&mut rest.put_increment_conflicts);
        self.probe_conflict_key_set_differs += std::mem::take(&mut rest.conflict_key_set_differs);
        self.probe_hazard_write += std::mem::take(&mut rest.hazard_write);
        self.probe_hazard_read += std::mem::take(&mut rest.hazard_read);
        self.probe_hazard_scan_keys += std::mem::take(&mut rest.hazard_scan_keys);
        self.probe_hazard_scan_values += std::mem::take(&mut rest.hazard_scan_values);
        self.probe_hazard_inputs += std::mem::take(&mut rest.hazard_inputs);
        self.probe_hazard_lifecycle += std::mem::take(&mut rest.hazard_lifecycle);
        self.probe_hazard_rebase += std::mem::take(&mut rest.hazard_rebase);
        self.probe_hazard_negative += std::mem::take(&mut rest.hazard_negative);
        self.probe_hazard_set_member += std::mem::take(&mut rest.hazard_set_member);
        // The paths the coverage gate asks the workload to reach, counted apart
        // for the same reason: a probe's fixture reaches each of them in every
        // case, so were they added to the workload's counters the gate would be
        // met by the probes alone and say nothing of the workload. The one
        // path the workload does not reach, a merge held by verification, is
        // the probes' to cover: `probe_verification_holds` is the coverage
        // counter and `verification_holds` the workload's own count.
        self.probe_merges_clean += std::mem::take(&mut rest.merges_clean);
        self.probe_merges_rebased += std::mem::take(&mut rest.merges_rebased);
        self.probe_conflicts += std::mem::take(&mut rest.conflicts);
        self.probe_lifecycle_refusals += std::mem::take(&mut rest.lifecycle_refusals);
        self.probe_verification_holds += std::mem::take(&mut rest.verification_holds);
        self.probe_escalations += std::mem::take(&mut rest.escalations);
        self.probe_reviewed_merges += std::mem::take(&mut rest.reviewed_merges);
        self.probe_no_change_merges += std::mem::take(&mut rest.no_change_merges);
        // Probe runtimes are tiny fixtures, not tick-scheduled workload merges.
        rest.merge_calls = 0;
        rest.merge_wall_ns = 0;
        rest.merge_wall_le_10us = 0;
        rest.merge_wall_le_100us = 0;
        rest.merge_wall_le_1ms = 0;
        rest.merge_wall_le_10ms = 0;
        rest.merge_wall_gt_10ms = 0;
        self.absorb(&rest);
    }

    /// The sum of the hard counters: zero for a run that supports the safety
    /// claim.
    pub fn hard_failures(&self) -> u64 {
        self.hard().iter().map(|(_, count)| count).sum()
    }

    /// Count one `merge_branch` call and the wall time it took.
    pub fn record_merge(&mut self, elapsed: Duration) {
        self.merge_calls += 1;
        self.merge_wall_ns += elapsed.as_nanos();
        let bucket = if elapsed <= Duration::from_micros(10) {
            &mut self.merge_wall_le_10us
        } else if elapsed <= Duration::from_micros(100) {
            &mut self.merge_wall_le_100us
        } else if elapsed <= Duration::from_millis(1) {
            &mut self.merge_wall_le_1ms
        } else if elapsed <= Duration::from_millis(10) {
            &mut self.merge_wall_le_10ms
        } else {
            &mut self.merge_wall_gt_10ms
        };
        *bucket += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_twenty_eight_hard_counters_and_no_name_is_in_two_groups() {
        assert_eq!(HARD.len(), 28);
        let mut names: Vec<&str> = HARD
            .iter()
            .chain(COVERAGE)
            .chain(DESCRIPTIVE)
            .copied()
            .collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
    }

    #[test]
    fn every_probe_has_its_coverage_counter() {
        for probe in 1..=26 {
            let name = format!("probe_p{probe}_exercised");
            assert!(COVERAGE.contains(&name.as_str()), "{name}");
        }
    }

    #[test]
    fn a_probes_hazard_trials_are_probe_trials_and_nothing_else_moves() {
        let probes = Metrics {
            hazard_write: 2,
            hazard_negative: 1,
            hazard_set_member: 3,
            probe_p1_exercised: 1,
            replays: 4,
            unnecessary_refusals: 5,
            put_increment_conflicts: 6,
            conflict_key_set_differs: 7,
            ..Metrics::default()
        };
        let mut total = Metrics {
            hazard_write: 10,
            ..Metrics::default()
        };
        total.absorb_probes(&probes);
        assert_eq!(
            total.hazard_write, 10,
            "a workload trial count is not the probes'"
        );
        assert_eq!(total.hazard_negative + total.hazard_set_member, 0);
        assert_eq!(total.probe_hazard_write, 2);
        assert_eq!(total.probe_hazard_negative, 1);
        assert_eq!(total.probe_hazard_set_member, 3);
        assert_eq!(total.probe_p1_exercised, 1);
        assert_eq!(total.replays, 4);
        assert_eq!(
            (
                total.unnecessary_refusals,
                total.put_increment_conflicts,
                total.conflict_key_set_differs
            ),
            (0, 0, 0),
            "the refusals of a probe's world are no share of the arms' refusals"
        );
        assert_eq!(total.probe_unnecessary_refusals, 5);
        assert_eq!(total.probe_put_increment_conflicts, 6);
        assert_eq!(total.probe_conflict_key_set_differs, 7);
    }

    #[test]
    fn a_probes_coverage_paths_are_no_coverage_of_the_workload() {
        let probes = Metrics {
            merges_clean: 1,
            merges_rebased: 2,
            conflicts: 3,
            lifecycle_refusals: 4,
            verification_holds: 5,
            escalations: 6,
            reviewed_merges: 7,
            no_change_merges: 8,
            probe_p1_exercised: 1,
            replays: 9,
            ..Metrics::default()
        };
        let mut workload = Metrics {
            conflicts: 10,
            ..Metrics::default()
        };
        workload.absorb_probes(&probes);
        let reached = |name: &str| {
            workload
                .coverage()
                .into_iter()
                .find(|(field, _)| *field == name)
                .map(|(_, count)| count)
        };
        for name in [
            "merges_clean",
            "merges_rebased",
            "lifecycle_refusals",
            "escalations",
            "reviewed_merges",
            "no_change_merges",
        ] {
            assert_eq!(reached(name), Some(0), "{name}");
        }
        assert_eq!(reached("conflicts"), Some(10), "only the workload's own");
        assert_eq!(workload.verification_holds, 0);
        assert_eq!(
            reached("probe_verification_holds"),
            Some(5),
            "the one path the probes cover for the gate"
        );
        assert_eq!(
            (
                workload.probe_merges_clean,
                workload.probe_merges_rebased,
                workload.probe_conflicts,
                workload.probe_lifecycle_refusals,
                workload.probe_verification_holds,
                workload.probe_escalations,
                workload.probe_reviewed_merges,
                workload.probe_no_change_merges,
            ),
            (1, 2, 3, 4, 5, 6, 7, 8)
        );
        // What only the probes exercise stays a coverage counter.
        assert_eq!(reached("probe_p1_exercised"), Some(1));
        assert_eq!(reached("replays"), Some(9));
    }

    #[test]
    fn probes_cannot_dilute_slow_workload_merges() {
        let mut workload = Metrics::default();
        workload.record_merge(Duration::from_millis(20));
        let mut probes = Metrics {
            conflicts: 2,
            ..Metrics::default()
        };
        for _ in 0..1000 {
            probes.record_merge(Duration::from_micros(1));
        }
        workload.absorb_probes(&probes);
        assert_eq!(workload.merge_calls, 1);
        assert_eq!(workload.merge_wall_gt_10ms, 1);
        assert_eq!(workload.merge_wall_le_10us, 0);
        assert_eq!(workload.merge_wall_ns, 20_000_000);
        assert_eq!(
            workload.conflicts, 0,
            "a probe's conflicts are not the workload's"
        );
        assert_eq!(workload.probe_conflicts, 2);
    }

    #[test]
    fn the_digest_follows_the_counters_and_ignores_the_clock() {
        let mut metrics = Metrics::default();
        let empty = metrics.digest_into(fnv(None, b"x"));
        metrics.record_merge(Duration::from_millis(3));
        assert_eq!(
            metrics.digest_into(fnv(None, b"x")),
            empty,
            "wall time is not in it"
        );
        metrics.conflicts = 1;
        assert_ne!(metrics.digest_into(fnv(None, b"x")), empty);
        assert_eq!(fnv(None, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(
            fnv(None, b"a"),
            0xaf63_dc4c_8601_ec8c,
            "the FNV-1a of one byte"
        );
    }

    #[test]
    fn hard_failures_sum_the_hard_counters_only() {
        let mut metrics = Metrics::default();
        assert_eq!(metrics.hard_failures(), 0);
        metrics.merges_clean = 9;
        metrics.unnecessary_refusals = 4;
        assert_eq!(metrics.hard_failures(), 0);
        metrics.lost_updates = 2;
        metrics.harness_errors = 1;
        assert_eq!(metrics.hard_failures(), 3);
    }

    #[test]
    fn absorbing_adds_every_counter_and_the_histogram() {
        let mut first = Metrics {
            lost_updates: 1,
            conflicts: 2,
            wasted_ticks: 3,
            ..Metrics::default()
        };
        first.record_merge(Duration::from_micros(5));
        let mut second = Metrics {
            lost_updates: 10,
            ..Metrics::default()
        };
        second.record_merge(Duration::from_millis(20));
        first.absorb(&second);
        assert_eq!(
            (first.lost_updates, first.conflicts, first.wasted_ticks),
            (11, 2, 3)
        );
        assert_eq!(first.merge_calls, 2);
        assert_eq!((first.merge_wall_le_10us, first.merge_wall_gt_10ms), (1, 1));
        assert_eq!(first.merge_wall_ns, 5_000 + 20_000_000);
    }

    #[test]
    fn a_merge_lands_in_the_smallest_bucket_that_holds_it() {
        let mut metrics = Metrics::default();
        for micros in [10, 11, 100, 101, 1_000, 1_001, 10_000, 10_001] {
            metrics.record_merge(Duration::from_micros(micros));
        }
        assert_eq!(metrics.merge_wall_le_10us, 1);
        assert_eq!(metrics.merge_wall_le_100us, 2);
        assert_eq!(metrics.merge_wall_le_1ms, 2);
        assert_eq!(metrics.merge_wall_le_10ms, 2);
        assert_eq!(metrics.merge_wall_gt_10ms, 1);
        assert_eq!(metrics.merge_calls, 8);
    }
}
