//! Platform metrics over the work schema, compiled by [`crate::metric_sql`].

use ptr_analytics::{MetricRow, MetricSpec};

use super::{database, to_u64, PgSubstrate};
use crate::error::PgError;
use crate::metric::metric_sql;

impl PgSubstrate {
    /// Evaluate one metric: a numerator and a denominator per group, ordered by
    /// group. Intervals and their meaning come from `ptr-analytics`.
    pub async fn metric(&self, spec: MetricSpec) -> Result<Vec<MetricRow>, PgError> {
        let rows = self
            .client
            .query(&metric_sql(spec, &self.schemas), &[])
            .await
            .map_err(database)?;
        rows.into_iter()
            .map(|row| {
                Ok(MetricRow {
                    group: row.get(0),
                    numerator: to_u64(row.get(1), "metric")?,
                    denominator: to_u64(row.get(2), "metric")?,
                })
            })
            .collect()
    }
}
