//! Derived search cache: documents of live capsule generations, their
//! full-text lexemes and half-precision embeddings, and hybrid retrieval that
//! returns search candidates, never evidence.

use ptr_search::{
    check_rank_fusion, weighted_rank_fusion, FusedHit, FusionError, SearchHit, WeightedList,
};
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
    /// Finite and nonnegative; with `vector_weight`, summing to at most
    /// [`ptr_search::MAX_TOTAL_WEIGHT`], so no fused score overflows.
    pub lexical_weight: f32,
    /// Finite and nonnegative.
    pub vector_weight: f32,
    /// The `k` of reciprocal rank fusion, finite and nonnegative; 60 is the
    /// customary default.
    pub rank_constant: f32,
}

/// What one hybrid query returned, all from one snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResults {
    pub lexical: Vec<SearchHit>,
    pub vector: Vec<SearchHit>,
    /// Weighted reciprocal rank fusion of both lists, keyed by capsule and
    /// generation. Every fused score is finite.
    pub fused: Vec<FusedHit>,
    /// The projection watermark of the snapshot the query read.
    pub watermark: u64,
}

impl PgSubstrate {
    /// Register an embedding space and build its HNSW index, in one
    /// transaction.
    ///
    /// Idempotent for an identical definition; a different definition under
    /// an existing id is refused. The index is partial (`WHERE space = id`) over
    /// the column cast to the space's fixed dimension, so one table holds any
    /// number of spaces and each gets an index of its own shape.
    ///
    /// The catalog row, the check against a stored definition and the index
    /// (a plain `CREATE INDEX`, which runs inside a transaction and holds a
    /// `SHARE` lock on the documents table while it builds, so writes of
    /// documents wait for it) commit together or not at all: when the index
    /// cannot be built (a lock timeout, missing privileges, resources), the
    /// error is returned and a space that was not registered before stays
    /// unregistered, so no document is indexed under a space that lacks its
    /// index. Registering an identical definition again builds the index if
    /// it is missing. The index is looked up by name only: a relation already
    /// holding that name in the derived schema is taken for it.
    pub async fn register_space(&mut self, space: &EmbeddingSpace) -> Result<(), PgError> {
        if space.dims == 0 || space.dims > 4000 {
            return Err(PgError::DimensionMismatch {
                expected: 4000,
                actual: usize::from(space.dims),
            });
        }
        let derived = self.schemas.derived.clone();
        check_text("embedding_space.model", &space.model)?;
        check_text("embedding_space.revision", &space.revision)?;
        let id = space.id.as_str();
        let dims = i32::from(space.dims);
        // Dropped on any early return, which rolls the registration back.
        let transaction = self.read_committed().await?;
        transaction
            .execute(
                &format!(
                    "INSERT INTO {derived}.embedding_space (id, model, revision, dims, metric) \
                     VALUES ($1, $2, $3, $4, 'cosine') ON CONFLICT (id) DO NOTHING"
                ),
                &[&id, &space.model, &space.revision, &dims],
            )
            .await
            .map_err(database)?;
        // A separate statement, so under READ COMMITTED it sees a definition
        // another registration committed while this one's insert waited.
        let row = transaction
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
        transaction
            .batch_execute(&format!(
                "CREATE INDEX IF NOT EXISTS sd_hnsw_{id} ON {derived}.search_document \
                 USING hnsw ((embedding::halfvec({dims})) halfvec_cosine_ops) \
                 WHERE space = '{id}'"
            ))
            .await
            .map_err(database)?;
        transaction.commit().await.map_err(database)
    }

    /// Index a document for a live capsule generation.
    ///
    /// The capsule's live-generation row is held `FOR SHARE` for the whole
    /// transaction, so the projector, which updates that row before it deletes
    /// a superseded or revoked generation's document, cannot interleave: the
    /// write either lands before the projector's delete or sees the new
    /// generation and is refused. A generation that is not live, is revoked,
    /// or is not a capsule is refused with [`PgError::NotLive`].
    ///
    /// A generation has one content. Indexing it again with the stored
    /// `content_digest` replaces the body, embedding and space (a
    /// re-embedding); indexing it with another digest is refused as
    /// [`PgError::DocumentConflict`] and the stored document is kept, so a
    /// stale or erroneous indexing job cannot replace the document of a live
    /// generation. Two writers racing with different digests are ordered by
    /// the table's key: whichever commits second is refused. That the stored
    /// digest is the capsule's is the first writer's obligation; nothing here
    /// reads the capsule.
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
        // A conflicting row with another digest fails the DO UPDATE's WHERE,
        // so the statement writes nothing; the same digest is a re-embedding.
        let written = transaction
            .execute(
                &format!(
                    "INSERT INTO {derived}.search_document \
                     (capsule, generation, project, content_digest, body, space, embedding, \
                      indexed_at_commit) \
                     VALUES ($1, $2, $3, $4, $5, $7, {embedding_sql}, $8) \
                     ON CONFLICT (capsule, generation) DO UPDATE \
                     SET body = EXCLUDED.body, \
                         space = EXCLUDED.space, embedding = EXCLUDED.embedding, \
                         indexed_at_commit = EXCLUDED.indexed_at_commit, \
                         project = EXCLUDED.project \
                     WHERE {derived}.search_document.content_digest = EXCLUDED.content_digest"
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
        if written == 0 {
            return Err(PgError::DocumentConflict {
                capsule: capsule.to_owned(),
                generation: document.generation.0,
            });
        }
        transaction.commit().await.map_err(database)
    }

    /// Run a hybrid query in one read-only repeatable-read snapshot.
    ///
    /// Every hit is joined to the lifecycle catalog of the same snapshot, so
    /// only generations that are live *and* unrevoked there are returned: the
    /// rule `PtrRuntime::generation_validity` applies, under which a revoked
    /// generation is not live although it is still the capsule's live
    /// generation. They are still [`SearchHit`]s at the search-candidate stage:
    /// using one as evidence requires observing it against the lifecycle
    /// authority.
    ///
    /// Each mode applies the lifecycle rule and the project filter before its
    /// `limit`, so a stale or revoked cache row never takes the place of a
    /// live document. The lexical list is exact. The vector list is read from
    /// the space's HNSW index with an iterative strict-order scan and
    /// `hnsw.ef_search` set to the limit, at least 40 and at most 1,000: it
    /// holds approximate nearest neighbours, not necessarily the nearest, and
    /// it can hold fewer than `limit` hits although more documents match,
    /// with nothing reporting it. pgvector ends an iterative scan after
    /// `hnsw.max_scan_tuples` visited tuples (20,000 by default) or
    /// `hnsw.scan_mem_multiplier` times `work_mem`, and a selective project
    /// filter, many stale rows or a large limit can reach that bound first.
    ///
    /// # Errors
    /// Refuses, before any SQL runs, a zero `limit` and the fusion parameters
    /// [`check_rank_fusion`] refuses, each as [`PgError::OutOfRange`] naming the
    /// field: `query.rank_constant`, `query.lexical_weight`,
    /// `query.vector_weight`, or `query.weights` for weights whose total
    /// could fuse to an infinite score.
    pub async fn search(&mut self, query: &HybridQuery) -> Result<SearchResults, PgError> {
        if query.limit == 0 {
            return Err(PgError::OutOfRange {
                field: "query.limit",
            });
        }
        check_rank_fusion(
            [query.lexical_weight, query.vector_weight],
            query.rank_constant,
        )
        .map_err(|error| PgError::OutOfRange {
            field: match error {
                FusionError::InvalidRankConstant => "query.rank_constant",
                FusionError::InvalidWeight { list: 0 } => "query.lexical_weight",
                FusionError::InvalidWeight { .. } => "query.vector_weight",
                FusionError::WeightTotal
                | FusionError::DuplicateHit { .. }
                | FusionError::InvalidScore { .. } => "query.weights",
            },
        })?;
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

        // The lifecycle rule, as a condition on a document `d`: its
        // generation is the capsule's live one and is not revoked.
        let live = format!(
            "EXISTS (SELECT 1 FROM {projection}.live_generation l \
                     WHERE l.target = d.capsule AND l.generation = d.generation) \
             AND NOT EXISTS (SELECT 1 FROM {projection}.tombstone t \
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
                         WHERE {live} \
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
            // for 100 hits silently stops near 40. An iterative scan (pgvector
            // 0.8, required by the capability check) keeps reading the index
            // in exact distance order, past ef_search, until `limit` rows
            // pass every condition below, or until it has visited
            // `hnsw.max_scan_tuples` tuples (pgvector's default is 20,000)
            // or used `hnsw.scan_mem_multiplier` times `work_mem`, whichever
            // comes first. ef_search is the limit, at least 40 and at most
            // the 1,000 pgvector accepts. Both values are integers or
            // keywords computed here, never caller text.
            let ef_search = query.limit.clamp(40, MAX_EF_SEARCH);
            transaction
                .batch_execute(&format!(
                    "SET LOCAL hnsw.iterative_scan = strict_order; \
                     SET LOCAL hnsw.ef_search = {ef_search};"
                ))
                .await
                .map_err(database)?;
            let id = space.as_str();
            // The lifecycle rule is a condition of the limited scan itself,
            // so a stale or revoked row the cache still holds is skipped and
            // never takes the place of a live one within `limit`. The ORDER
            // BY is the indexed expression alone, so the planner can read
            // the space's HNSW index in order and test each row it yields
            // against the conditions below the LIMIT (nested-loop semi and
            // anti joins, on the servers this was checked against).
            let rows = transaction
                .query(
                    &format!(
                        "SELECT v.capsule, v.generation, (1 - v.distance)::real AS score \
                         FROM ( \
                             SELECT d.capsule, d.generation, \
                                    d.embedding::halfvec({dims}) \
                                        <=> $1::real[]::halfvec({dims}) AS distance \
                             FROM {derived}.search_document d \
                             WHERE d.space = '{id}' AND ($2::text IS NULL OR d.project = $2) \
                               AND {live} \
                             ORDER BY d.embedding::halfvec({dims}) \
                                          <=> $1::real[]::halfvec({dims}) \
                             LIMIT $3 \
                         ) v \
                         ORDER BY v.distance, v.capsule, v.generation"
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
        )
        // The parameters passed the same check above, and each list is keyed
        // by the table's primary key, so a refusal here is a corrupt read.
        .map_err(|error| PgError::CorruptRow {
            table: "search_document",
            reason: error.to_string(),
        })?;
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
