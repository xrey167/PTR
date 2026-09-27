use ptr_analytics::{Grouping, Metric, MetricSpec, Window};

use crate::config::SchemaSet;

/// Compile a metric to one SQL query over the work schema.
///
/// The query returns rows `(grp text, numerator bigint, denominator bigint)`,
/// one per group, ordered by group. The metric's meaning is defined in
/// `ptr-analytics`; this is only its Postgres rendering, and it reads working
/// state, never the projection. Each branch is counted once, and a [`Window`]
/// bounds the one timestamp per branch that `ptr-analytics` names for the
/// metric (the record that puts the branch into the denominator), measured
/// back from the server's `now()`; the day count is a `u16`, so it is
/// rendered literally, as that many times 24 hours. A window of days runs
/// from that many times 24 hours before `now()` up to `now()` itself, both
/// included, whatever the session's time zone (a calendar day there is not
/// 24 hours long across a change of its offset): a record stamped later (a
/// stamp written explicitly, or taken from a clock that has since moved
/// back) is in no such window, only in [`Window::All`]. Zero days counts
/// nothing, so its window is empty whatever the stamps, rather than the
/// records stamped exactly at `now()`.
pub fn metric_sql(spec: MetricSpec, schemas: &SchemaSet) -> String {
    let work = schemas.work.as_str();
    let group = match spec.grouping {
        Grouping::Overall => "''::text",
        Grouping::ByPrincipal => "b.author",
        Grouping::ByPolicyVersion => "coalesce(t.policy_version, '')",
    };
    let triaged = format!("{work}.branch_triage t JOIN {work}.branch b ON b.id = t.branch");
    let (numerator, denominator, from, mut conditions, stamp) = match spec.metric {
        // Both counts are of eligible branches. Verification alone never
        // auto-proposes, but the check work migration 9 added for it is NOT
        // VALID: an ineligible auto-proposal stored before it (record_triage
        // stored any triage then) is in neither count, so the numerator
        // never counts a branch the denominator leaves out.
        Metric::AutoProposeShare => (
            "count(*) FILTER (WHERE t.eligible AND t.decision = 'auto_propose')",
            "count(*) FILTER (WHERE t.eligible)",
            triaged,
            Vec::new(),
            "t.decided_at",
        ),
        Metric::EscalationShare => (
            "count(*) FILTER (WHERE t.decision = 'escalate')",
            "count(*)",
            triaged,
            Vec::new(),
            "t.decided_at",
        ),
        // A branch can hold several outcomes (the key is (branch, outcome)),
        // so outcomes are aggregated per branch before the window applies: a
        // branch conflicted if any of its outcomes is a conflict, and entered
        // the record with its first outcome.
        Metric::ConflictRate => (
            "count(*) FILTER (WHERE o.conflicted)",
            "count(*)",
            format!(
                "(SELECT branch, bool_or(outcome = 'conflicted') AS conflicted, \
                         min(observed_at) AS first_observed \
                  FROM {work}.branch_outcome GROUP BY branch) o \
                 JOIN {work}.branch b ON b.id = o.branch \
                 LEFT JOIN {work}.branch_triage t ON t.branch = o.branch"
            ),
            Vec::new(),
            "o.first_observed",
        ),
        // A branch is adjudicated at most once (branch_outcome_one_adjudication)
        // and triaged once, so each branch is one row, windowed on its
        // adjudication.
        Metric::AdjudicatedHarmRate => (
            "count(DISTINCT o.branch) FILTER (WHERE o.outcome = 'adjudicated_harmful')",
            "count(DISTINCT o.branch)",
            format!(
                "{triaged} JOIN {work}.branch_outcome o ON o.branch = t.branch \
                 AND o.outcome IN ('adjudicated_harmful', 'adjudicated_harmless')"
            ),
            vec!["t.calibration_slice".to_owned()],
            "o.observed_at",
        ),
        // A branch is merged at most once and reverted at most once (the
        // outcome key is (branch, outcome)), so both counts are of branches.
        // A revert is a later commit undoing the merge, so it counts only at
        // a commit index after the merge's: `record_outcome` refuses any
        // other, and a row written around it (before the check existed, or
        // by another writer) is not counted as reverting a merge it
        // precedes. The window applies to the merge only: its revert counts
        // whenever it was stamped.
        Metric::RevertShare => (
            "count(r.branch)",
            "count(*)",
            format!(
                "{work}.branch_outcome o JOIN {work}.branch b ON b.id = o.branch \
                 LEFT JOIN {work}.branch_triage t ON t.branch = o.branch \
                 LEFT JOIN {work}.branch_outcome r ON r.branch = o.branch \
                 AND r.outcome = 'reverted' AND r.commit_index > o.commit_index"
            ),
            vec!["o.outcome = 'merged'".to_owned()],
            "o.observed_at",
        ),
    };
    match spec.window {
        Window::All => {}
        Window::LastDays(0) => conditions.push("false".to_owned()),
        Window::LastDays(days) => {
            // Hours, which timestamptz arithmetic takes as elapsed time; days
            // would be calendar days of the session's time zone, shorter or
            // longer than 24 hours across a change of its offset.
            let hours = u32::from(days) * 24;
            conditions.push(format!(
                "{stamp} >= now() - make_interval(hours => {hours})"
            ));
            conditions.push(format!("{stamp} <= now()"));
        }
    }
    let filter = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };
    format!(
        "SELECT {group} AS grp, {numerator}::bigint AS numerator, \
         {denominator}::bigint AS denominator FROM {from}{filter} GROUP BY 1 ORDER BY 1"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_metric_grouping_and_window_compiles_against_the_work_schema_only() {
        let schemas = SchemaSet::with_prefix("ptr").unwrap();
        for metric in Metric::ALL {
            for grouping in [
                Grouping::Overall,
                Grouping::ByPrincipal,
                Grouping::ByPolicyVersion,
            ] {
                for window in [Window::All, Window::LastDays(7), Window::LastDays(0)] {
                    let sql = metric_sql(
                        MetricSpec {
                            metric,
                            grouping,
                            window,
                        },
                        &schemas,
                    );
                    assert!(sql.contains("ptr_work."), "{sql}");
                    assert!(!sql.contains("ptr_projection."), "{sql}");
                    assert!(sql.starts_with("SELECT "));
                    assert!(sql.matches(" WHERE ").count() <= 1, "{sql}");
                    // Seven times 24 hours, not seven calendar days of
                    // the session's time zone.
                    assert_eq!(
                        sql.contains("make_interval(hours => 168)"),
                        window == Window::LastDays(7),
                        "{sql}"
                    );
                    assert!(!sql.contains("days =>"), "{sql}");
                    // A window of days ends at the query's time, and zero
                    // days is empty rather than the records stamped then.
                    assert_eq!(
                        sql.contains(" <= now()"),
                        window == Window::LastDays(7),
                        "{sql}"
                    );
                    assert_eq!(
                        sql.contains(" false GROUP BY "),
                        window == Window::LastDays(0),
                        "{sql}"
                    );
                    assert!(!sql.contains("make_interval(days => 0)"), "{sql}");
                    assert!(!sql.contains("make_interval(hours => 0)"), "{sql}");
                    assert_eq!(
                        sql.contains("r.commit_index > o.commit_index"),
                        metric == Metric::RevertShare,
                        "{sql}"
                    );
                    // The auto-propose share counts eligible branches on
                    // both sides of the fraction.
                    assert_eq!(
                        sql.contains("FILTER (WHERE t.eligible AND t.decision = 'auto_propose')"),
                        metric == Metric::AutoProposeShare,
                        "{sql}"
                    );
                }
            }
        }
        // The longest window still fits the hours make_interval takes.
        let sql = metric_sql(
            MetricSpec {
                metric: Metric::AutoProposeShare,
                grouping: Grouping::Overall,
                window: Window::LastDays(u16::MAX),
            },
            &schemas,
        );
        assert!(sql.contains("make_interval(hours => 1572840)"), "{sql}");
    }
}
