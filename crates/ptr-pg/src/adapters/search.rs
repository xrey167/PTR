//! Derived search cache: documents of live capsule generations, their
//! full-text lexemes and half-precision embeddings, and hybrid retrieval that
//! returns search candidates, never evidence.

use ptr_search::{weighted_rank_fusion, FusedHit, SearchHit, WeightedList};
use ptr_types::{CapsuleId, Generation, ProjectId};
use tokio_postgres::IsolationLevel;

use super::{check_text, database, to_i64, to_u64, PgSubstrate};
use crate::config::Identifier;
use crate::error::PgError;

/// Backend name of full-text hits.
pub const LEXICAL_BACKEND: &str = "postgres-fts";
/// Backend name of embedding hits.
pub const VECTOR_BACKEND: &str = "pgvector-halfvec";

/// Largest magnitude a half-precision value holds.
const HALF_MAX: f32 = 65_504.0;
/// Largest magnitude that rounds to zero in half precision: 2^-25 ties to even,
/// which is zero.
const HALF_ZERO: f32 = 2.980_232_2e-8;
/// The `hnsw.ef_search` pgvector accepts at most.
const MAX_EF_SEARCH: u32 = 1000;

/// A registered embedding space: one model at one revision with one
/// dimension. Vectors are comparable only within a space.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmbeddingSpace {
    pub id: Identifier,
    pub model: String,
    pub revision: String,
    /// At most 4,000: the widest `halfvec` an HNSW index accepts.
    pub dims: u16,
}

/// A document for one capsule generation. Its project is taken from the
/// projection, not from the writer.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchDocument {
    pub capsule: CapsuleId,
    pub generation: Generation,
    /// Digest of the capsule content the body and embedding were computed
    /// from.
    pub content_digest: [u8; 32],
    pub body: String,
    pub embedding: Option<(Identifier, Vec<f32>)>,
}

/// One hybrid query. Either or both retrieval modes may be present.
#[derive(Clone, Debug, PartialEq)]
pub struct HybridQuery {
    pub project: Option<ProjectId>,
    pub text: Option<String>,
    pub embedding: Option<(Identifier, Vec<f32>)>,
    /// Candidates per mode before fusion.
    pub limit: u32,
    pub lexical_weight: f32,
    pub vector_weight: f32,
    /// The `k` of reciprocal rank fusion; 60 is the customary default.
    pub rank_constant: f32,
}

/// What one hybrid query returned, all from one snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResults {
    pub lexical: Vec<SearchHit>,
    pub vector: Vec<SearchHit>,
    /// Weighted reciprocal rank fusion of both lists, keyed by capsule and
    /// generation.
    pub fused: Vec<FusedHit>,
    /// The projection watermark of the snapshot the query read.
    pub watermark: u64,
}

impl PgSubstrate {
    /// Register an embedding space and build its HNSW index.
    ///
    /// Idempotent for an identical definition; a different definition under
    /// an existing id is refused. The index is partial (`WHERE space = id`) over
    /// the column cast to the space's fixed dimension, so one table holds any
    /// number of spaces and each gets an index of its own shape.
    pub async fn register_space(&self, space: &EmbeddingSpace) -> Result<(), PgError> {
        if space.dims == 0 || space.dims > 4000 {
            return Err(PgError::DimensionMismatch {
                expected: 4000,
                actual: usize::from(space.dims),
            });
        }
        let derived = &self.schemas.derived;
        check_text("embedding_space.model", &space.model)?;
        check_text("embedding_space.revision", &space.revision)?;
        let id = space.id.as_str();
        let dims = i32::from(space.dims);
        self.client
            .execute(
                &format!(
                    "INSERT INTO {derived}.embedding_space (id, model, revision, dims, metric) \
                     VALUES ($1, $2, $3, $4, 'cosine') ON CONFLICT (id) DO NOTHING"
                ),
                &[&id, &space.model, &space.revision, &dims],
            )
            .await
            .map_err(database)?;
        let row = self
            .client
            .query_one(
                &format!(
                    "SELECT model, revision, dims FROM {derived}.embedding_space WHERE id = $1"
                ),
                &[&id],
            )
            .await
            .map_err(database)?;
        let (model, revision, stored): (String, String, i32) = (row.get(0), row.get(1), row.get(2));
        if model != space.model || revision != space.revision || stored != dims {
            return Err(PgError::SpaceConflict {
                space: id.to_owned(),
            });
        }
        // `id` is an Identifier and `dims` an integer: nothing here needs quoting.
        self.client
            .batch_execute(&format!(
                "CREATE INDEX IF NOT EXISTS sd_hnsw_{id} ON {derived}.search_document \
                 USING hnsw ((embedding::halfvec({dims})) halfvec_cosine_ops) \
                 WHERE space = '{id}'"
            ))
            .await
            .map_err(database)
    }

    /// Index a document for a live capsule generation.
    ///
    /// The capsule's live-generation row is held `FOR SHARE` for the whole
    /// transaction, so the projector, which updates that row before it deletes
    /// a superseded or revoked generation's document, cannot interleave: the
    /// write either lands before the projector's delete or sees the new
    /// generation and is refused. A generation that is not live, is revoked,
    /// or is not a capsule is refused with [`PgError::NotLive`].
    pub async fn upsert_document(&mut self, document: &SearchDocument) -> Result<(), PgError> {
        let schemas = self.schemas.clone();
        let (projection, derived) = (schemas.projection.as_str(), schemas.derived.as_str());
        let capsule = document.capsule.0.as_str();
        check_text("search_document.capsule", capsule)?;
        check_text("search_document.body", &document.body)?;
        let generation = to_i64(document.generation.0, "generation")?;
        let not_live = || PgError::NotLive {
            target: capsule.to_owned(),
            generation: document.generation.0,
        };

        let transaction = self.read_committed().await?;
        let live = transaction
            .query_opt(
                &format!(
                    "SELECT generation, project FROM {projection}.live_generation \
                     WHERE target = $1 FOR SHARE"
                ),
                &[&capsule],
            )
            .await
            .map_err(database)?;
        let project: String = match live {
            Some(row) if row.get::<_, i64>(0) == generation => match row.get(1) {
                Some(project) => project,
                None => return Err(not_live()),
            },
            _ => return Err(not_live()),
        };
        // A separate statement, so it reads a snapshot taken after the lock was
        // granted and sees any tombstone committed before that.
        let row = transaction
            .query_one(
                &format!(
                    "SELECT EXISTS (SELECT 1 FROM {projection}.tombstone \
                                    WHERE subject = $1 AND generation = $2), \
                            (SELECT last_applied FROM {projection}.projection_watermark \
                             WHERE id = 1)"
                ),
                &[&capsule, &generation],
            )
            .await
            .map_err(database)?;
        if row.get::<_, bool>(0) {
            return Err(not_live());
        }
        let watermark: i64 = row.get(1);

        let (space, embedding_sql, vector) = match &document.embedding {
            Some((space, vector)) => {
                let dims = space_dims(&transaction, derived, space).await?;
                check_embedding(vector, dims)?;
                (
                    Some(space.as_str().to_owned()),
                    format!("$6::real[]::halfvec({dims})"),
                    Some(vector.clone()),
                )
            }
            None => (None, "$6::real[]::halfvec".to_owned(), None),
        };
        transaction
            .execute(
                &format!(
                    "INSERT INTO {derived}.search_document \
                     (capsule, generation, project, content_digest, body, space, embedding, \
                      indexed_at_commit) \
                     VALUES ($1, $2, $3, $4, $5, $7, {embedding_sql}, $8) \
                     ON CONFLICT (capsule, generation) DO UPDATE \
                     SET content_digest = EXCLUDED.content_digest, body = EXCLUDED.body, \
                         space = EXCLUDED.space, embedding = EXCLUDED.embedding, \
                         indexed_at_commit = EXCLUDED.indexed_at_commit, \
                         project = EXCLUDED.project"
                ),
                &[
                    &capsule,
                    &generation,
                    &project,
                    &document.content_digest.to_vec(),
                    &document.body,
                    &vector,
                    &space,
                    &watermark,
                ],
            )
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)
    }

    /// Run a hybrid query in one read-only repeatable-read snapshot.
    ///
    /// Every hit is joined to the lifecycle catalog of the same snapshot, so
    /// only generations that are live and unrevoked there are returned. They
    /// are still [`SearchHit`]s at the search-candidate stage: using one as
    /// evidence requires observing it against the lifecycle authority.
    pub async fn search(&mut self, query: &HybridQuery) -> Result<SearchResults, PgError> {
        if let Some(text) = &query.text {
            check_text("query.text", text)?;
        }
        if let Some(project) = &query.project {
            check_text("query.project", &project.0)?;
        }
        let schemas = self.schemas.clone();
        let (projection, derived) = (schemas.projection.as_str(), schemas.derived.as_str());
        let project = query.project.as_ref().map(|project| project.0.clone());
        let limit = i64::from(query.limit);

        let transaction = self
            .client
            .build_transaction()
            .isolation_level(IsolationLevel::RepeatableRead)
            .read_only(true)
            .start()
            .await
            .map_err(database)?;
        let watermark: i64 = transaction
            .query_one(
                &format!("SELECT last_applied FROM {projection}.projection_watermark WHERE id = 1"),
                &[],
            )
            .await
            .map_err(database)?
            .get(0);

        let live_filter = format!(
            "JOIN {projection}.live_generation l \
                 ON l.target = d.capsule AND l.generation = d.generation \
             WHERE NOT EXISTS (SELECT 1 FROM {projection}.tombstone t \
                               WHERE t.subject = d.capsule AND t.generation = d.generation)"
        );

        let mut lexical = Vec::new();
        if let Some(text) = &query.text {
            let rows = transaction
                .query(
                    &format!(
                        "SELECT d.capsule, d.generation, ts_rank_cd(d.lexeme, q)::real AS score \
                         FROM {derived}.search_document d \
                         CROSS JOIN plainto_tsquery('simple'::regconfig, $1) q \
                         {live_filter} \
                           AND d.lexeme @@ q AND ($2::text IS NULL OR d.project = $2) \
                         ORDER BY score DESC, d.capsule, d.generation LIMIT $3"
                    ),
                    &[text, &project, &limit],
                )
                .await
                .map_err(database)?;
            lexical = hits(rows, LEXICAL_BACKEND)?;
        }

        let mut vector = Vec::new();
        if let Some((space, embedding)) = &query.embedding {
            let dims = space_dims(&transaction, derived, space).await?;
            check_embedding(embedding, dims)?;
            // An HNSW scan returns at most `hnsw.ef_search` rows unless it may
            // iterate, with or without a filter: without both settings a query
            // for 100 hits silently stops near 40. Iterative scans (pgvector
            // 0.8, required by the capability check) keep walking the graph in
            // exact order until `limit` rows pass the filter. Both values are
            // integers or keywords computed here, never caller text.
            let ef_search = query.limit.clamp(40, MAX_EF_SEARCH);
            transaction
                .batch_execute(&format!(
                    "SET LOCAL hnsw.iterative_scan = strict_order; \
                     SET LOCAL hnsw.ef_search = {ef_search};"
                ))
                .await
                .map_err(database)?;
            let id = space.as_str();
            // The ORDER BY is the indexed expression alone, so the planner can
            // use the space's HNSW index; liveness is filtered after it.
            let rows = transaction
                .query(
                    &format!(
                        "SELECT d.capsule, d.generation, (1 - v.distance)::real AS score \
                         FROM ( \
                             SELECT capsule, generation, \
                                    embedding::halfvec({dims}) <=> $1::real[]::halfvec({dims}) \
                                        AS distance \
                             FROM {derived}.search_document \
                             WHERE space = '{id}' AND ($2::text IS NULL OR project = $2) \
                             ORDER BY embedding::halfvec({dims}) <=> $1::real[]::halfvec({dims}) \
                             LIMIT $3 \
                         ) v \
                         JOIN {derived}.search_document d \
                             ON d.capsule = v.capsule AND d.generation = v.generation \
                         {live_filter} \
                         ORDER BY v.distance, d.capsule, d.generation"
                    ),
                    &[embedding, &project, &limit],
                )
                .await
                .map_err(database)?;
            vector = hits(rows, VECTOR_BACKEND)?;
        }
        transaction.commit().await.map_err(database)?;

        let fused = weighted_rank_fusion(
            &[
                WeightedList {
                    weight: query.lexical_weight,
                    hits: &lexical,
                },
                WeightedList {
                    weight: query.vector_weight,
                    hits: &vector,
                },
            ],
            query.rank_constant,
        );
        Ok(SearchResults {
            lexical,
            vector,
            fused,
            watermark: to_u64(watermark, "projection_watermark")?,
        })
    }
}

async fn space_dims(
    transaction: &tokio_postgres::Transaction<'_>,
    derived: &str,
    space: &Identifier,
) -> Result<usize, PgError> {
    let row = transaction
        .query_opt(
            &format!("SELECT dims FROM {derived}.embedding_space WHERE id = $1"),
            &[&space.as_str()],
        )
        .await
        .map_err(database)?;
    match row {
        Some(row) => usize::try_from(row.get::<_, i32>(0)).map_err(|_| PgError::CorruptRow {
            table: "embedding_space",
            reason: "negative dimension".into(),
        }),
        None => Err(PgError::InvalidEmbedding {
            reason: "the embedding space is not registered",
        }),
    }
}

fn check_embedding(vector: &[f32], dims: usize) -> Result<(), PgError> {
    if vector.len() != dims {
        return Err(PgError::DimensionMismatch {
            expected: dims,
            actual: vector.len(),
        });
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(PgError::InvalidEmbedding {
            reason: "a value is not finite",
        });
    }
    if vector.iter().any(|value| value.abs() > HALF_MAX) {
        return Err(PgError::InvalidEmbedding {
            reason: "a value exceeds the half-precision range",
        });
    }
    // Checked after rounding to half precision, which is what is stored and
    // compared: a vector of values too small for it is a zero vector there.
    if vector.iter().all(|value| value.abs() <= HALF_ZERO) {
        return Err(PgError::InvalidEmbedding {
            reason: "the vector is zero in half precision and has no cosine distance",
        });
    }
    Ok(())
}

fn hits(rows: Vec<tokio_postgres::Row>, backend: &str) -> Result<Vec<SearchHit>, PgError> {
    rows.into_iter()
        .map(|row| {
            let capsule: String = row.get(0);
            let generation = to_u64(row.get(1), "search_document")?;
            Ok(SearchHit::new(
                CapsuleId(capsule),
                Generation(generation),
                row.get(2),
                backend,
            ))
        })
        .collect()
}
