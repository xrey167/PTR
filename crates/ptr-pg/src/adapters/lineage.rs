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
    /// The report's header row, keyed by the adapter, is written before its
    /// layers, so any second report for the adapter, whether it repeats,
    /// overlaps or avoids the stored layers and whether or not it runs
    /// concurrently, is refused by that key (SQLSTATE 23505) instead of
    /// merging into the first; an adapter the catalog does not know is
    /// refused by the header's foreign key (23503). The header records how
    /// many layers the report has, and the database refuses, when it
    /// commits, a report whose layer rows differ from that count, so a layer
    /// appended later is refused as well. A report with no layer is refused
    /// with `PgError::InvalidInterference` before anything is written: it
    /// would record no evidence while claiming the adapter's one report.
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
        let refuse = |reason| PgError::InvalidInterference {
            adapter: adapter.0.clone(),
            reason,
        };
        if report.layers.is_empty() {
            return Err(refuse("a report needs at least one layer"));
        }
        let layer_count = i32::try_from(report.layers.len())
            .map_err(|_| refuse("the report has more layers than the table can record"))?;
        let work = self.schemas.work.clone();
        let transaction = self.read_committed().await?;
        transaction
            .execute(
                &format!(
                    "INSERT INTO {work}.adapter_interference_report (adapter, layer_count) \
                     VALUES ($1, $2)"
                ),
                &[&adapter.0, &layer_count],
            )
            .await
            .map_err(database)?;
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
    /// or `None` when none was stored. A report whose stored layers are not
    /// the number it was recorded with is refused as a corrupt row rather
    /// than returned.
    pub async fn load_interference(
        &self,
        adapter: &AdapterId,
    ) -> Result<Option<InterferenceReport>, PgError> {
        check_text("adapter_interference.adapter", &adapter.0)?;
        let work = &self.schemas.work;
        let Some(header) = self
            .client
            .query_opt(
                &format!(
                    "SELECT layer_count FROM {work}.adapter_interference_report \
                     WHERE adapter = $1"
                ),
                &[&adapter.0],
            )
            .await
            .map_err(database)?
        else {
            return Ok(None);
        };
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
        let layer_count: i32 = header.get(0);
        if usize::try_from(layer_count).ok() != Some(rows.len()) {
            return Err(PgError::CorruptRow {
                table: "adapter_interference",
                reason: format!("{} layers, recorded with {layer_count}", rows.len()),
            });
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
