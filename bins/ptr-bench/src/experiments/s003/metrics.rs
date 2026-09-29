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
            pub fn hard(&self) -> Vec<(&'static str, u64)> {
                vec![$((stringify!($hard), self.$hard)),*]
            }

            pub fn coverage(&self) -> Vec<(&'static str, u64)> {
                vec![$((stringify!($coverage), self.$coverage)),*]
            }

            pub fn descriptive(&self) -> Vec<(&'static str, u64)> {
                vec![$((stringify!($descriptive), self.$descriptive)),*]
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
        verification_holds,
        escalations,
        reviewed_merges,
        no_change_merges,
        hazard_write,
        hazard_read,
        hazard_scan_keys,
        hazard_scan_values,
        hazard_inputs,
        hazard_lifecycle,
        hazard_rebase,
        hazard_negative,
        hazard_set_member,
        lww_lost_updates,
        lww_lost_increments,
        occ_undetected_phantoms,
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
    }
}

impl Metrics {
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
