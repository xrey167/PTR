//! Retrieval eligibility against a real PostgreSQL server: who may see a document, when it holds,
//! and which metadata it carries, applied before the scan's `LIMIT`.
//!
//! `PTR_PG_TEST_DSN` must name one loopback PostgreSQL 16+ server with pgvector 0.8+ (the same
//! requirement as `tests/postgres.rs`). Every test works in its own schema prefix and drops it.
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU32, Ordering};

use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_pg::{
    EmbeddingSpace, HybridQuery, Identifier, PgError, PgSubstrate, SchemaSet, SearchDocument,
    DERIVED_MIGRATIONS,
};
use ptr_search::{FilterError, SearchAttributes, SearchFilter};
use ptr_types::{
    CapsuleId, CommitIndex, Generation, KnowledgeValidity, PrincipalId, ProjectId, Timestamp,
};

static NEXT_PREFIX: AtomicU32 = AtomicU32::new(0);

fn dsn() -> String {
    match std::env::var("PTR_PG_TEST_DSN") {
        Ok(dsn) if !dsn.trim().is_empty() => dsn,
        _ => panic!(
            "PTR_PG_TEST_DSN is unset: the postgres test target needs a loopback PostgreSQL 16+ \
             server with pgvector, e.g. PTR_PG_TEST_DSN=\"host=127.0.0.1 port=5432 user=postgres\""
        ),
    }
}

async fn raw_client() -> tokio_postgres::Client {
    let config: tokio_postgres::Config = dsn().parse().expect("parse PTR_PG_TEST_DSN");
    let (client, connection) = config
        .connect(tokio_postgres::NoTls)
        .await
        .expect("connect to PTR_PG_TEST_DSN");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

async fn substrate_with(options: &str) -> PgSubstrate {
    let raw = raw_client().await;
    raw.batch_execute(
        "BEGIN; SELECT pg_advisory_xact_lock(7402301); \
         CREATE EXTENSION IF NOT EXISTS vector; COMMIT;",
    )
    .await
    .expect("pgvector must be installed or installable");
    let prefix = format!(
        "ptrf_{}_{}",
        std::process::id(),
        NEXT_PREFIX.fetch_add(1, Ordering::SeqCst)
    );
    let schemas = SchemaSet::with_prefix(&prefix).unwrap();
    let dsn = if options.is_empty() {
        dsn()
    } else {
        format!("{} options='{options}'", dsn())
    };
    raw.batch_execute(&format!(
        "DROP SCHEMA IF EXISTS {prefix}_derived CASCADE; \
         DROP SCHEMA IF EXISTS {prefix}_projection CASCADE; \
         DROP SCHEMA IF EXISTS {prefix}_work CASCADE;"
    ))
    .await
    .unwrap();
    let mut substrate = PgSubstrate::connect_with(&dsn, schemas).await.unwrap();
    substrate.migrate().await.unwrap();
    substrate
}

async fn substrate() -> PgSubstrate {
    substrate_with("").await
}

fn space() -> EmbeddingSpace {
    EmbeddingSpace {
        id: Identifier::new("toy4").unwrap(),
        model: "toy-encoder".into(),
        revision: "2026-10".into(),
        dims: 4,
    }
}

fn capsule(index: u64, id: &str) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(index),
        event: LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("atlas"),
            capsule: CapsuleId::from(id),
            generation: Generation(1),
        },
    }
}

fn set(members: &[&str]) -> BTreeSet<String> {
    members.iter().map(|member| (*member).to_owned()).collect()
}

fn between(from: u64, until: u64) -> KnowledgeValidity {
    KnowledgeValidity {
        valid_from: Some(Timestamp(from)),
        valid_until: Some(Timestamp(until)),
    }
}

fn document(name: &str, embedding: [f32; 4], attributes: SearchAttributes) -> SearchDocument {
    SearchDocument {
        capsule: CapsuleId::from(name),
        generation: Generation(1),
        content_digest: [name.len() as u8; 32],
        body: format!("retrieval {name}"),
        embedding: Some((space().id, embedding.to_vec())),
        attributes,
    }
}

async fn index(substrate: &mut PgSubstrate, documents: &[SearchDocument]) {
    substrate.register_space(&space()).await.unwrap();
    let events: Vec<CommittedEvent> = documents
        .iter()
        .enumerate()
        .map(|(position, document)| capsule(position as u64 + 1, &document.capsule.0))
        .collect();
    substrate.replay(&events).await.unwrap();
    for document in documents {
        substrate.upsert_document(document).await.unwrap();
    }
}

fn query(limit: u32, filter: SearchFilter) -> HybridQuery {
    HybridQuery {
        project: Some(ProjectId::from("atlas")),
        text: Some("retrieval".into()),
        embedding: Some((space().id, vec![1.0, 0.0, 0.0, 0.0])),
        limit,
        lexical_weight: 1.0,
        vector_weight: 1.0,
        rank_constant: 60.0,
        filter,
    }
}

fn names(hits: &[ptr_search::SearchHit]) -> BTreeSet<String> {
    hits.iter().map(|hit| hit.capsule().0.clone()).collect()
}

fn asking(principal: &str, groups: &[&str], now: u64) -> SearchFilter {
    SearchFilter::for_principal(
        PrincipalId::from(principal),
        groups.iter().map(|group| (*group).to_owned()),
        Timestamp(now),
    )
}

#[tokio::test]
async fn the_derived_catalog_ships_the_filter_migration_and_it_adds_its_columns() {
    assert_eq!(DERIVED_MIGRATIONS.len(), 2);
    assert_eq!(DERIVED_MIGRATIONS[1].name, "search_filters");
    let substrate = substrate().await;
    let raw = raw_client().await;
    let derived = substrate.schemas().derived.to_string();
    let columns: BTreeSet<String> = raw
        .query(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = $1 \
             AND table_name = 'search_document'",
            &[&derived],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| row.get(0))
        .collect();
    for column in [
        "principals",
        "valid_from",
        "valid_until",
        "semantic_role",
        "predicate",
        "tags",
    ] {
        assert!(
            columns.contains(column),
            "{column} missing from {columns:?}"
        );
    }
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn another_principal_sees_nothing_of_a_restricted_document_in_either_mode() {
    let mut substrate = substrate().await;
    index(
        &mut substrate,
        &[
            document("open", [0.0, 1.0, 0.0, 0.0], SearchAttributes::default()),
            document(
                "alices",
                [1.0, 0.0, 0.0, 0.0],
                SearchAttributes {
                    principals: set(&["alice"]),
                    ..SearchAttributes::default()
                },
            ),
            document(
                "staffs",
                [0.9, 0.1, 0.0, 0.0],
                SearchAttributes {
                    principals: set(&["staff"]),
                    ..SearchAttributes::default()
                },
            ),
        ],
    )
    .await;

    let mallory = substrate
        .search(&query(10, asking("mallory", &[], 5)))
        .await
        .unwrap();
    assert_eq!(names(&mallory.lexical), set(&["open"]));
    assert_eq!(names(&mallory.vector), set(&["open"]));
    assert_eq!(
        mallory.fused.len(),
        1,
        "a document the asker may not see is not in the fused list either"
    );

    let nobody = substrate
        .search(&query(10, SearchFilter::default()))
        .await
        .unwrap();
    assert_eq!(
        names(&nobody.vector),
        set(&["open"]),
        "no identity sees project-wide only"
    );

    let alice = substrate
        .search(&query(10, asking("alice", &[], 5)))
        .await
        .unwrap();
    assert_eq!(names(&alice.vector), set(&["open", "alices"]));

    let bob = substrate
        .search(&query(10, asking("bob", &["staff"], 5)))
        .await
        .unwrap();
    assert_eq!(names(&bob.lexical), set(&["open", "staffs"]));
    assert_eq!(names(&bob.vector), set(&["open", "staffs"]));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn an_expired_or_not_yet_valid_row_drops_out_before_the_limit() {
    let mut substrate = substrate().await;
    // The two nearest documents do not hold at 15; the two farther ones do.
    index(
        &mut substrate,
        &[
            document(
                "expired",
                [1.0, 0.0, 0.0, 0.0],
                SearchAttributes {
                    validity: between(1, 10),
                    ..SearchAttributes::default()
                },
            ),
            document(
                "future",
                [0.99, 0.01, 0.0, 0.0],
                SearchAttributes {
                    validity: between(20, 30),
                    ..SearchAttributes::default()
                },
            ),
            document("always", [0.5, 0.5, 0.0, 0.0], SearchAttributes::default()),
            document(
                "current",
                [0.4, 0.6, 0.0, 0.0],
                SearchAttributes {
                    validity: between(10, 20),
                    ..SearchAttributes::default()
                },
            ),
        ],
    )
    .await;

    let results = substrate
        .search(&query(2, asking("alice", &[], 15)))
        .await
        .unwrap();
    assert_eq!(
        names(&results.vector),
        set(&["always", "current"]),
        "the limit is filled with documents that hold, not with the nearest rows"
    );
    assert_eq!(results.vector.len(), 2);
    assert_eq!(results.lexical.len(), 2);

    // The boundaries are half-open: valid_from is inside, valid_until is outside.
    let at_start = substrate
        .search(&query(10, asking("a", &[], 10)))
        .await
        .unwrap();
    assert_eq!(names(&at_start.vector), set(&["always", "current"]));
    let at_end = substrate
        .search(&query(10, asking("a", &[], 20)))
        .await
        .unwrap();
    assert_eq!(names(&at_end.vector), set(&["always", "future"]));

    // A filter without a now applies no time rule: the audit view.
    let audit = SearchFilter {
        principal: Some(PrincipalId::from("auditor")),
        ..SearchFilter::default()
    };
    let everything = substrate.search(&query(10, audit)).await.unwrap();
    assert_eq!(everything.vector.len(), 4);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn the_filter_inside_the_hnsw_scan_still_returns_limit_hits_when_that_many_are_visible() {
    // Plans that sort or scan the table instead are priced out, so the index is read.
    let mut substrate = substrate_with("-c enable_seqscan=off -c enable_sort=off").await;
    substrate.register_space(&space()).await.unwrap();
    let raw = raw_client().await;
    let (projection, derived) = (
        substrate.schemas().projection.to_string(),
        substrate.schemas().derived.to_string(),
    );
    // 600 live documents. The 570 nearest to the query belong to somebody else or have
    // expired; the 30 farthest are visible to the asker. A filter applied after the index
    // returned its first candidates would find none of them.
    raw.batch_execute(&format!(
        "INSERT INTO {projection}.live_generation (target, generation, project, commit_index) \
             SELECT 'c' || i, 1, 'atlas', 1 FROM generate_series(1, 600) i; \
         INSERT INTO {derived}.search_document \
             (capsule, generation, project, content_digest, body, space, embedding, \
              indexed_at_commit, principals, valid_from, valid_until) \
             SELECT 'c' || i, 1, 'atlas', decode(repeat('00', 32), 'hex'), 'doc', 'toy4', \
                    ARRAY[1.0, i / 600.0, 0, 0]::real[]::halfvec(4), 0, \
                    CASE WHEN i <= 285 THEN ARRAY['someone-else'] ELSE ARRAY[]::text[] END, \
                    NULL, \
                    CASE WHEN i > 285 AND i <= 570 THEN 5 ELSE NULL END \
             FROM generate_series(1, 600) i; \
         ANALYZE {derived}.search_document;"
    ))
    .await
    .unwrap();
    let mut hybrid = query(10, asking("alice", &[], 100));
    hybrid.text = None;
    let results = substrate.search(&hybrid).await.unwrap();
    assert_eq!(results.vector.len(), 10);
    let visible: BTreeSet<String> = (571..=600).map(|i| format!("c{i}")).collect();
    assert!(
        names(&results.vector).is_subset(&visible),
        "{:?}",
        names(&results.vector)
    );
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn role_predicate_and_tags_narrow_both_modes() {
    let mut substrate = substrate().await;
    let attributes = |role: &str, predicate: &str, tags: &[&str]| SearchAttributes {
        semantic_role: Some(role.into()),
        predicate: Some(predicate.into()),
        tags: set(tags),
        ..SearchAttributes::default()
    };
    index(
        &mut substrate,
        &[
            document(
                "fact_lead",
                [1.0, 0.0, 0.0, 0.0],
                attributes("FACT", "lead_time", &["eu"]),
            ),
            document(
                "fact_dead",
                [0.9, 0.1, 0.0, 0.0],
                attributes("FACT", "deadline", &["us"]),
            ),
            document(
                "rule_lead",
                [0.8, 0.2, 0.0, 0.0],
                attributes("RULE", "lead_time", &["eu", "us"]),
            ),
            document("plain", [0.7, 0.3, 0.0, 0.0], SearchAttributes::default()),
        ],
    )
    .await;
    let with = |edit: &dyn Fn(&mut SearchFilter)| {
        let mut filter = asking("alice", &[], 5);
        edit(&mut filter);
        query(10, filter)
    };
    let facts = substrate
        .search(&with(&|f| f.semantic_role = Some("FACT".into())))
        .await
        .unwrap();
    assert_eq!(names(&facts.vector), set(&["fact_lead", "fact_dead"]));
    assert_eq!(names(&facts.lexical), set(&["fact_lead", "fact_dead"]));
    let lead = substrate
        .search(&with(&|f| f.predicate = Some("lead_time".into())))
        .await
        .unwrap();
    assert_eq!(names(&lead.vector), set(&["fact_lead", "rule_lead"]));
    let us = substrate
        .search(&with(&|f| f.tags_any = set(&["us", "asia"])))
        .await
        .unwrap();
    assert_eq!(names(&us.vector), set(&["fact_dead", "rule_lead"]));
    let narrow = substrate
        .search(&with(&|f| {
            f.semantic_role = Some("FACT".into());
            f.predicate = Some("lead_time".into());
            f.tags_any = set(&["eu"]);
        }))
        .await
        .unwrap();
    assert_eq!(names(&narrow.vector), set(&["fact_lead"]));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn the_sql_predicate_admits_exactly_the_documents_the_reference_predicate_admits() {
    let mut substrate = substrate().await;
    let mut documents = Vec::new();
    let mut attribute_sets = Vec::new();
    for (position, principals) in [vec![], vec!["alice"], vec!["staff"], vec!["alice", "staff"]]
        .into_iter()
        .enumerate()
    {
        for (offset, validity) in [
            KnowledgeValidity::ALWAYS,
            between(10, 20),
            KnowledgeValidity {
                valid_from: Some(Timestamp(15)),
                valid_until: None,
            },
            KnowledgeValidity {
                valid_from: None,
                valid_until: Some(Timestamp(15)),
            },
        ]
        .into_iter()
        .enumerate()
        {
            for role in [None, Some("FACT")] {
                let attributes = SearchAttributes {
                    principals: set(&principals),
                    validity,
                    semantic_role: role.map(str::to_owned),
                    ..SearchAttributes::default()
                };
                let name = format!(
                    "d{position}_{offset}_{}",
                    role.unwrap_or("none").to_lowercase()
                );
                documents.push(document(
                    &name,
                    [1.0, position as f32 * 0.1, offset as f32 * 0.1, 0.0],
                    attributes.clone(),
                ));
                attribute_sets.push((name, attributes));
            }
        }
    }
    index(&mut substrate, &documents).await;
    let filters = [
        SearchFilter::default(),
        asking("alice", &[], 9),
        asking("alice", &[], 10),
        asking("alice", &[], 15),
        asking("alice", &[], 20),
        asking("bob", &["staff"], 14),
        asking("bob", &["staff"], 19),
        asking("mallory", &["guests"], 15),
        SearchFilter {
            semantic_role: Some("FACT".into()),
            ..asking("alice", &["staff"], 15)
        },
        SearchFilter {
            principal: Some(PrincipalId::from("alice")),
            ..SearchFilter::default()
        },
    ];
    for filter in filters {
        let expected: BTreeSet<String> = attribute_sets
            .iter()
            .filter(|(_, attributes)| filter.admits(attributes))
            .map(|(name, _)| name.clone())
            .collect();
        let results = substrate.search(&query(100, filter.clone())).await.unwrap();
        assert_eq!(names(&results.lexical), expected, "{filter:?} lexical");
        assert_eq!(names(&results.vector), expected, "{filter:?} vector");
    }
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn re_indexing_the_same_content_replaces_the_access_list_so_a_narrowing_takes_effect() {
    let mut substrate = substrate().await;
    let wide = document("shared", [1.0, 0.0, 0.0, 0.0], SearchAttributes::default());
    index(&mut substrate, std::slice::from_ref(&wide)).await;
    let bob = asking("bob", &[], 5);
    assert_eq!(
        names(
            &substrate
                .search(&query(10, bob.clone()))
                .await
                .unwrap()
                .vector
        ),
        set(&["shared"])
    );
    let narrowed = SearchDocument {
        attributes: SearchAttributes {
            principals: set(&["alice"]),
            ..SearchAttributes::default()
        },
        ..wide
    };
    substrate.upsert_document(&narrowed).await.unwrap();
    let after = substrate.search(&query(10, bob)).await.unwrap();
    assert!(after.vector.is_empty() && after.lexical.is_empty());
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_malformed_filter_or_attribute_set_is_refused_before_any_sql_runs() {
    let mut substrate = substrate().await;
    let bad_filter = SearchFilter {
        principal: Some(PrincipalId::from("")),
        ..SearchFilter::default()
    };
    assert_eq!(
        substrate.search(&query(10, bad_filter)).await.unwrap_err(),
        PgError::InvalidSearchFilter {
            field: "query.filter",
            error: FilterError::Empty { field: "principal" }
        }
    );
    let too_late = SearchFilter {
        now: Some(Timestamp(u64::MAX)),
        ..SearchFilter::default()
    };
    assert_eq!(
        substrate.search(&query(10, too_late)).await.unwrap_err(),
        PgError::OutOfRange {
            field: "query.filter.now"
        }
    );

    substrate.register_space(&space()).await.unwrap();
    substrate.replay(&[capsule(1, "c1")]).await.unwrap();
    let never = document(
        "c1",
        [1.0, 0.0, 0.0, 0.0],
        SearchAttributes {
            validity: between(20, 20),
            ..SearchAttributes::default()
        },
    );
    assert_eq!(
        substrate.upsert_document(&never).await.unwrap_err(),
        PgError::InvalidSearchFilter {
            field: "search_document.attributes",
            error: FilterError::EmptyValidity
        }
    );
    assert_eq!(
        PgError::InvalidSearchFilter {
            field: "x",
            error: FilterError::EmptyValidity
        }
        .code(),
        "PTR_PG_INVALID_SEARCH_FILTER"
    );
    substrate.drop_all().await.unwrap();
}
