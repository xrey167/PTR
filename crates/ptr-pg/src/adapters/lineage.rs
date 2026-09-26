//! Interference evidence of adapter candidates, in the work schema's lineage
//! tables. The rest of the lineage catalog has no adapter here yet.

use ptr_lineage::{AdapterId, InterferenceReport, LayerInterference};

use super::{check_text, database, PgSubstrate};
use crate::error::PgError;

impl PgSubstrate {
    /// Store the interference report measured for a candidate adapter, one row
    /// per layer, in one transaction, so a promotion or consolidation decision
    /// can later be audited against the evidence it was made on. A report is
    /// stored once and never rewritten; the adapter and every adapter a layer
    /// names as its worst overlap must already be in the catalog.
    ///
    /// An overlap or chance level outside `[0, 1]` (which
    /// `ptr_lineage::measure_interference` never produces) or a layer named
    /// twice is refused by the table before anything is committed.
    pub async fn record_interference(
        &mut self,
        adapter: &AdapterId,
        report: &InterferenceReport,
    ) -> Result<(), PgError> {
        check_text("adapter_interference.adapter", &adapter.0)?;
        for layer in &report.layers {
            check_text("adapter_interference.layer", &layer.layer)?;
            if let Some(worst) = &layer.worst {
                check_text("adapter_interference.worst", &worst.0)?;
            }
        }
        let work = self.schemas.work.clone();
        let transaction = self.read_committed().await?;
        for layer in &report.layers {
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {work}.adapter_interference \
                         (adapter, layer, output_overlap, input_overlap, output_chance, \
                          input_chance, worst) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7)"
                    ),
                    &[
                        &adapter.0,
                        &layer.layer,
                        &layer.output_overlap,
                        &layer.input_overlap,
                        &layer.output_chance,
                        &layer.input_chance,
                        &layer.worst.as_ref().map(|worst| worst.0.as_str()),
                    ],
                )
                .await
                .map_err(database)?;
        }
        transaction.commit().await.map_err(database)
    }

    /// The interference report stored for an adapter, layers in name order,
    /// or `None` when none was stored.
    pub async fn load_interference(
        &self,
        adapter: &AdapterId,
    ) -> Result<Option<InterferenceReport>, PgError> {
        check_text("adapter_interference.adapter", &adapter.0)?;
        let work = &self.schemas.work;
        let rows = self
            .client
            .query(
                &format!(
                    "SELECT layer, output_overlap, input_overlap, output_chance, input_chance, \
                            worst \
                     FROM {work}.adapter_interference WHERE adapter = $1 ORDER BY layer"
                ),
                &[&adapter.0],
            )
            .await
            .map_err(database)?;
        if rows.is_empty() {
            return Ok(None);
        }
        let layers = rows
            .into_iter()
            .map(|row| LayerInterference {
                layer: row.get(0),
                output_overlap: row.get(1),
                input_overlap: row.get(2),
                output_chance: row.get(3),
                input_chance: row.get(4),
                worst: row.get::<_, Option<String>>(5).map(AdapterId),
            })
            .collect();
        Ok(Some(InterferenceReport { layers }))
    }
}
