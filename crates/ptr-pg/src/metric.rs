use ptr_analytics::{Grouping, Metric, MetricSpec};

use crate::config::SchemaSet;

/// Compile a metric to one SQL query over the work schema.
///
/// The query returns rows `(grp text, numerator bigint, denominator bigint)`,
/// one per group, ordered by group. The metric's meaning is defined in
/// `ptr-analytics`; this is only its Postgres rendering, and it reads working
/// state, never the projection.
pub fn metric_sql(spec: MetricSpec, schemas: &SchemaSet) -> String {
    let work = schemas.work.as_str();
    let group = match spec.grouping {
        Grouping::Overall => "''::text",
        Grouping::ByPrincipal => "b.author",
        Grouping::ByPolicyVersion => "coalesce(t.policy_version, '')",
    };
    let (numerator, denominator, from) = match spec.metric {
        Metric::AutoProposeShare => (
            "count(*) FILTER (WHERE t.decision = 'auto_propose')",
            "count(*) FILTER (WHERE t.eligible)",
            format!("{work}.branch_triage t JOIN {work}.branch b ON b.id = t.branch"),
        ),
        Metric::EscalationShare => (
            "count(*) FILTER (WHERE t.decision = 'escalate')",
            "count(*)",
            format!("{work}.branch_triage t JOIN {work}.branch b ON b.id = t.branch"),
        ),
        Metric::ConflictRate => (
            "count(DISTINCT o.branch) FILTER (WHERE o.outcome = 'conflicted')",
            "count(DISTINCT o.branch)",
            format!(
                "{work}.branch_outcome o JOIN {work}.branch b ON b.id = o.branch \
                 LEFT JOIN {work}.branch_triage t ON t.branch = o.branch"
            ),
        ),
        Metric::AdjudicatedHarmRate => (
            "count(DISTINCT o.branch) FILTER (WHERE o.outcome = 'adjudicated_harmful')",
            "count(DISTINCT o.branch)",
            format!(
                "{work}.branch_triage t JOIN {work}.branch b ON b.id = t.branch \
                 JOIN {work}.branch_outcome o ON o.branch = t.branch \
                 AND o.outcome IN ('adjudicated_harmful', 'adjudicated_harmless') \
                 WHERE t.calibration_slice"
            ),
        ),
    };
    format!(
        "SELECT {group} AS grp, {numerator}::bigint AS numerator, \
         {denominator}::bigint AS denominator FROM {from} GROUP BY 1 ORDER BY 1"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_metric_and_grouping_compiles_against_the_work_schema_only() {
        let schemas = SchemaSet::with_prefix("ptr").unwrap();
        for metric in Metric::ALL {
            for grouping in [
                Grouping::Overall,
                Grouping::ByPrincipal,
                Grouping::ByPolicyVersion,
            ] {
                let sql = metric_sql(MetricSpec { metric, grouping }, &schemas);
                assert!(sql.contains("ptr_work."), "{sql}");
                assert!(!sql.contains("ptr_projection."), "{sql}");
                assert!(sql.starts_with("SELECT "));
            }
        }
    }
}
