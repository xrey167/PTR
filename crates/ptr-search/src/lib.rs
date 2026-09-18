use ptr_types::{CapsuleId, Generation};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceStage {
    SearchCandidate,
    PossibleEvidence,
    Observed,
    VerifiedKnown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidencePromotionError {
    WrongStage,
    StaleGeneration,
    SourceNotResolved,
    VerificationFailed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub capsule: CapsuleId,
    pub generation: Generation,
    pub score: f32,
    pub backend: String,
    stage: EvidenceStage,
}

impl SearchHit {
    pub fn new(
        capsule: CapsuleId,
        generation: Generation,
        score: f32,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            capsule,
            generation,
            score,
            backend: backend.into(),
            stage: EvidenceStage::SearchCandidate,
        }
    }

    pub fn stage(&self) -> EvidenceStage {
        self.stage
    }

    pub fn mark_possible(&mut self) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::SearchCandidate {
            return Err(EvidencePromotionError::WrongStage);
        }
        self.stage = EvidenceStage::PossibleEvidence;
        Ok(())
    }

    pub fn observe(
        &mut self,
        current_generation: Generation,
        exact_source_resolved: bool,
    ) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::PossibleEvidence {
            return Err(EvidencePromotionError::WrongStage);
        }
        if self.generation != current_generation {
            return Err(EvidencePromotionError::StaleGeneration);
        }
        if !exact_source_resolved {
            return Err(EvidencePromotionError::SourceNotResolved);
        }
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    pub fn promote_verified(
        &mut self,
        verification_passed: bool,
    ) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::Observed {
            return Err(EvidencePromotionError::WrongStage);
        }
        if !verification_passed {
            return Err(EvidencePromotionError::VerificationFailed);
        }
        self.stage = EvidenceStage::VerifiedKnown;
        Ok(())
    }
}

pub trait SearchIndex {
    fn search(&self, query: &str, limit: usize) -> Vec<SearchHit>;
}

pub fn reciprocal_rank_fusion(lists: &[Vec<SearchHit>], k: f32) -> Vec<(CapsuleId, f32)> {
    let mut scores: BTreeMap<CapsuleId, f32> = BTreeMap::new();
    for list in lists {
        for (rank, hit) in list.iter().enumerate() {
            *scores.entry(hit.capsule.clone()).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<_> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

#[cfg(feature = "tantivy-backend")]
mod tantivy_backend {
    use super::{SearchHit, SearchIndex};
    use ptr_types::{CapsuleId, Generation};
    use std::sync::Mutex;
    use tantivy::{
        collector::TopDocs,
        query::QueryParser,
        schema::{Field, Schema, TantivyDocument, INDEXED, STORED, STRING, TEXT},
        Index, IndexReader, IndexWriter, ReloadPolicy, Term,
    };

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct SearchBackendError(pub String);

    pub struct TantivyLexicalIndex {
        index: Index,
        reader: IndexReader,
        writer: Mutex<IndexWriter>,
        capsule_field: Field,
        generation_field: Field,
        body_field: Field,
    }

    impl TantivyLexicalIndex {
        pub fn new_in_ram() -> Result<Self, SearchBackendError> {
            let mut schema_builder = Schema::builder();
            let capsule_field = schema_builder.add_text_field("capsule", STRING | STORED);
            let generation_field =
                schema_builder.add_u64_field("generation", INDEXED | STORED);
            let body_field = schema_builder.add_text_field("body", TEXT);
            let schema = schema_builder.build();

            let index = Index::create_in_ram(schema);
            let reader: IndexReader = index
                .reader_builder()
                .reload_policy(ReloadPolicy::Manual)
                .try_into()
                .map_err(error)?;
            let writer = index.writer(15_000_000).map_err(error)?;

            Ok(Self {
                index,
                reader,
                writer: Mutex::new(writer),
                capsule_field,
                generation_field,
                body_field,
            })
        }

        pub fn upsert(
            &self,
            capsule: CapsuleId,
            generation: Generation,
            text: &str,
        ) -> Result<(), SearchBackendError> {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| SearchBackendError("tantivy writer mutex poisoned".into()))?;
            writer.delete_term(Term::from_field_text(
                self.capsule_field,
                &capsule.to_string(),
            ));

            let mut doc = TantivyDocument::default();
            doc.add_text(self.capsule_field, capsule.to_string());
            doc.add_u64(self.generation_field, generation.0);
            doc.add_text(self.body_field, text);
            writer.add_document(doc).map_err(error)?;
            writer.commit().map_err(error)?;
            drop(writer);
            self.reader.reload().map_err(error)
        }

        pub fn delete(&self, capsule: &CapsuleId) -> Result<(), SearchBackendError> {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| SearchBackendError("tantivy writer mutex poisoned".into()))?;
            writer.delete_term(Term::from_field_text(
                self.capsule_field,
                &capsule.to_string(),
            ));
            writer.commit().map_err(error)?;
            drop(writer);
            self.reader.reload().map_err(error)
        }

        pub fn search_result(
            &self,
            query: &str,
            limit: usize,
        ) -> Result<Vec<SearchHit>, SearchBackendError> {
            if query.trim().is_empty() || limit == 0 {
                return Ok(Vec::new());
            }

            let parser = QueryParser::for_index(&self.index, vec![self.body_field]);
            let query = parser.parse_query(query).map_err(error)?;
            let searcher = self.reader.searcher();
            let docs = searcher
                .search(&query, &TopDocs::with_limit(limit))
                .map_err(error)?;

            let mut hits = Vec::with_capacity(docs.len());
            for (score, address) in docs {
                let doc = searcher
                    .doc::<TantivyDocument>(address)
                    .map_err(error)?;
                let Some(capsule) = doc
                    .get_first(self.capsule_field)
                    .and_then(|value| value.as_value().as_str())
                else {
                    continue;
                };
                let Some(generation) = doc
                    .get_first(self.generation_field)
                    .and_then(|value| value.as_value().as_u64())
                else {
                    continue;
                };

                hits.push(SearchHit::new(
                    CapsuleId(capsule.to_owned()),
                    Generation(generation),
                    score,
                    "tantivy",
                ));
            }
            Ok(hits)
        }
    }

    impl SearchIndex for TantivyLexicalIndex {
        fn search(&self, query: &str, limit: usize) -> Vec<SearchHit> {
            self.search_result(query, limit).unwrap_or_default()
        }
    }

    fn error(error: impl std::fmt::Display) -> SearchBackendError {
        SearchBackendError(error.to_string())
    }

    pub use TantivyLexicalIndex as IndexBackend;
}

#[cfg(feature = "tantivy-backend")]
pub use tantivy_backend::{IndexBackend as TantivyLexicalIndex, SearchBackendError};