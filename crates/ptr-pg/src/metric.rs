use ptr_analytics::{Grouping, Metric, MetricSpec, Window};

use crate::config::SchemaSet;

/// Compile a metric to one SQL query over the work schema.
///
/// The query returns rows `(grp text, numerator bigint, denominator bigint)`,
/// one per group, ordered by group. The metric's meaning is defined in
/// `ptr-analytics`; this is only its Postgres rendering, and it reads working
/// state, never the projection. A [`Window`] bounds the timestamp of the
/// record that puts a branch into the denominator, measured back from the
/// server's `now()`; the day count is a `u16`, so it is rendered literally.
pub fn metric_sql(spec: MetricSpec, schemas: &SchemaSet) -> String {
    let work = schemas.work.as_str();
    let group = match spec.grouping {
        Grouping::Overall => "''::text",
        Grouping::ByPrincipal => "b.author",
        Grouping::ByPolicyVersion => "coalesce(t.policy_version, '')",
    };
    let triaged = format!("{work}.branch_triage t JOIN {work}.branch b ON b.id = t.branch");
    let (numerator, denominator, from, mut conditions, stamp) = match spec.metric {
        Metric::AutoProposeShare => (
            "count(*) FILTER (WHERE t.decision = 'auto_propose')",
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
        Metric::ConflictRate => (
            "count(DISTINCT o.branch) FILTER (WHERE o.outcome = 'conflicted')",
            "count(DISTINCT o.branch)",
            format!(
                "{work}.branch_outcome o JOIN {work}.branch b ON b.id = o.branch \
                 LEFT JOIN {work}.branch_triage t ON t.branch = o.branch"
            ),
            Vec::new(),
            "o.observed_at",
        ),
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
        Metric::RevertShare => (
            "count(r.branch)",
            "count(*)",
            format!(
                "{work}.branch_outcome o JOIN {work}.branch b ON b.id = o.branch \
                 LEFT JOIN {work}.branch_triage t ON t.branch = o.branch \
                 LEFT JOIN {work}.branch_outcome r ON r.branch = o.branch \
                 AND r.outcome = 'reverted'"
            ),
            vec!["o.outcome = 'merged'".to_owned()],
            "o.observed_at",
        ),
    };
    match spec.window {
        Window::All => {}
        Window::LastDays(days) => {
            conditions.push(format!("{stamp} >= now() - make_interval(days => {days})"));
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
                for window in [Window::All, Window::LastDays(7)] {
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
                    assert_eq!(
                        sql.contains("make_interval(days => 7)"),
                        window == Window::LastDays(7),
                        "{sql}"
                    );
                }
            }
        }
    }
}
