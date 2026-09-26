use ptr_fastmem::{
    decode_readout, Decay, DecodePolicy, FastMemory, FastMemoryConfig, FastMemoryError,
    IdentifierCodebook, ProjectionSpec, Query, Recall, SeededProjection, SourceRef, WriteRequest,
};
use ptr_types::{CapsuleId, Generation};

const EMBEDDING_DIM: usize = 64;
const HEADS: usize = 4;
const HEAD_DIM: usize = 16;

/// A stand-in for a frozen embedding model: a deterministic, roughly isotropic
/// vector per text.
fn embed(text: &str) -> Vec<f32> {
    let mut state = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (0..EMBEDDING_DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn config() -> FastMemoryConfig {
    FastMemoryConfig {
        heads: HEADS,
        key_dim: HEAD_DIM,
        value_dim: HEAD_DIM,
        checkpoint_interval: 8,
        max_writes: 1024,
    }
}

struct Codec {
    keys: SeededProjection,
    values: IdentifierCodebook,
}

impl Codec {
    fn new() -> Self {
        Self {
            keys: key_projection(11),
            values: IdentifierCodebook::new(29, HEADS * HEAD_DIM).unwrap(),
        }
    }

    /// A write that associates a cue with a fact: the key is the projected
    /// embedding of the cue, the value the fact's identifier code.
    fn write(&self, cue: &str, capsule: &str, generation: u64) -> WriteRequest {
        let capsule_id = CapsuleId::from(capsule);
        WriteRequest {
            source: SourceRef {
                key: capsule.to_owned(),
                generation: Generation(generation),
                input_digest: [generation as u8; 32],
            },
            key: self.keys.project(&embed(cue)).unwrap(),
            value: self.values.code_for(&capsule_id, Generation(generation)),
            beta: 1.0,
            decay: Decay::Scalar(0.999),
        }
    }

    fn query(&self, cue: &str) -> Query {
        Query::new(&config(), self.keys.project(&embed(cue)).unwrap()).unwrap()
    }

    /// A query that states the key projection it was projected by.
    fn projected_query(&self, cue: &str) -> Query {
        Query::project(&config(), &self.keys, &embed(cue)).unwrap()
    }
}

fn key_projection(seed: u64) -> SeededProjection {
    SeededProjection::new(ProjectionSpec {
        input_dim: EMBEDDING_DIM,
        heads: HEADS,
        head_dim: HEAD_DIM,
        seed,
    })
    .unwrap()
}

fn policy() -> DecodePolicy {
    DecodePolicy {
        limit: 3,
        min_score: 0.5,
        min_margin: 0.25,
    }
}

fn recall(memory: &FastMemory, codec: &Codec, cue: &str) -> Recall {
    let readout = memory.read_admitted(&codec.query(cue), |_| true).unwrap();
    decode_readout(&readout, &memory.fact_codes(), policy()).unwrap()
}

#[test]
fn a_cue_recalls_the_fact_written_under_it_as_a_named_capsule() {
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    let pairs = [
        ("favourite colour", "pref-color"),
        ("home city", "pref-city"),
        ("preferred language", "pref-lang"),
    ];
    for (cue, capsule) in pairs {
        memory.write(codec.write(cue, capsule, 1)).unwrap();
    }
    for (cue, capsule) in pairs {
        match recall(&memory, &codec, cue) {
            Recall::Hits(hits) => assert_eq!(hits[0].capsule, CapsuleId::from(capsule), "{cue}"),
            Recall::Unknown => panic!("{cue:?} must be recalled"),
        }
    }
}

#[test]
fn an_update_under_the_same_cue_recalls_the_newer_fact() {
    // Two different live facts answer the same cue; the delta rule replaces the
    // association instead of adding to it, so the newer one is recalled alone.
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    memory
        .write(codec.write("home city", "city-from-2024-profile", 1))
        .unwrap();
    memory
        .write(codec.write("home city", "city-from-2026-profile", 1))
        .unwrap();
    match recall(&memory, &codec, "home city") {
        Recall::Hits(hits) => {
            assert_eq!(hits[0].capsule, CapsuleId::from("city-from-2026-profile"));
            assert_eq!(hits.len(), 1);
        }
        Recall::Unknown => panic!("the newer fact must be recalled"),
    }
}

#[test]
fn an_unrelated_cue_is_unknown_rather_than_a_guess() {
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    memory
        .write(codec.write("favourite colour", "pref-color", 1))
        .unwrap();
    memory
        .write(codec.write("home city", "pref-city", 1))
        .unwrap();
    assert_eq!(recall(&memory, &codec, "tax identifier"), Recall::Unknown);
}

#[test]
fn a_revoked_fact_is_no_longer_a_decoding_candidate() {
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    memory
        .write(codec.write("home city", "pref-city", 1))
        .unwrap();
    memory
        .write(codec.write("favourite colour", "pref-color", 1))
        .unwrap();
    memory.revoke(|source| source.key == "pref-city");
    assert!(memory
        .fact_codes()
        .iter()
        .all(|fact| *fact.capsule() != CapsuleId::from("pref-city")));
    assert_eq!(recall(&memory, &codec, "home city"), Recall::Unknown);
}

#[test]
fn constraint_and_procedure_sources_are_never_decoded_as_capsules() {
    // Sources use the lifecycle target vocabulary, in which hard constraints
    // and procedures are targets but not capsules. Their writes still shape the
    // state and gate reads, but must never decode to a hit for a fabricated
    // capsule such as `constraint:budget`.
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    memory
        .write(codec.write("monthly budget", "constraint:budget", 1))
        .unwrap();
    memory
        .write(codec.write("release steps", "procedure:deploy", 1))
        .unwrap();
    memory
        .write(codec.write("home city", "pref-city", 1))
        .unwrap();
    let candidates: Vec<CapsuleId> = memory
        .fact_codes()
        .into_iter()
        .map(|fact| fact.capsule().clone())
        .collect();
    assert_eq!(candidates, vec![CapsuleId::from("pref-city")]);
    assert_eq!(memory.sources().len(), 3);
    assert_eq!(recall(&memory, &codec, "monthly budget"), Recall::Unknown);
    assert_eq!(recall(&memory, &codec, "release steps"), Recall::Unknown);
    match recall(&memory, &codec, "home city") {
        Recall::Hits(hits) => assert_eq!(hits[0].capsule, CapsuleId::from("pref-city")),
        Recall::Unknown => panic!("the capsule source is still recalled"),
    }
}

#[test]
fn a_readout_decodes_only_against_the_codebook_its_memory_was_written_with() {
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    memory
        .write(codec.write("home city", "pref-city", 1))
        .unwrap();
    let readout = memory
        .read_admitted(&codec.query("home city"), |_| true)
        .unwrap();
    assert_eq!(readout.codebook(), codec.values);
    // Another memory of the same shape under another seed: its codes have
    // the same length, and scoring them used to report crosstalk as weights.
    let other = IdentifierCodebook::new(30, HEADS * HEAD_DIM).unwrap();
    let mut foreign = FastMemory::new(config(), other).unwrap();
    foreign
        .write(WriteRequest {
            value: other.code_for(&CapsuleId::from("pref-city"), Generation(1)),
            ..codec.write("home city", "pref-city", 1)
        })
        .unwrap();
    assert_eq!(
        decode_readout(&readout, &foreign.fact_codes(), policy()),
        Err(FastMemoryError::CodebookMismatch {
            index: 0,
            expected: codec.values,
            actual: other,
        })
    );
    let named = other
        .fact(CapsuleId::from("pref-city"), Generation(1))
        .unwrap();
    assert!(matches!(
        decode_readout(&readout, [&named], policy()),
        Err(FastMemoryError::CodebookMismatch { .. })
    ));
    // Its own codes name the fact.
    match decode_readout(&readout, &memory.fact_codes(), policy()).unwrap() {
        Recall::Hits(hits) => assert_eq!(hits[0].capsule, CapsuleId::from("pref-city")),
        Recall::Unknown => panic!("the fact written under the cue is recalled"),
    }
    // A memory is never bound to a codebook whose codes are not values.
    let short = IdentifierCodebook::new(29, HEADS * HEAD_DIM - 1).unwrap();
    let refused = FastMemoryError::DimensionMismatch {
        field: "codebook",
        expected: HEADS * HEAD_DIM,
        actual: HEADS * HEAD_DIM - 1,
    };
    assert_eq!(FastMemory::new(config(), short).unwrap_err(), refused);
    assert_eq!(
        FastMemory::restore(config(), short, std::iter::empty()).unwrap_err(),
        refused
    );
}

#[test]
fn a_constraint_or_procedure_target_is_refused_as_an_explicit_fact_candidate() {
    // The write's value is the code of the constraint target itself, so a
    // candidate built for it scored 1.0 and decoded to a hit naming
    // `constraint:budget` as a capsule; the runtime reports that
    // constraint's generation as live, so a lifecycle check kept it.
    let codec = Codec::new();
    let mut memory = FastMemory::new(config(), codec.values).unwrap();
    memory
        .write(codec.write("monthly budget", "constraint:budget", 1))
        .unwrap();
    memory
        .write(codec.write("release steps", "procedure:deploy", 1))
        .unwrap();
    for target in ["constraint:budget", "procedure:deploy"] {
        assert_eq!(
            codec.values.fact(CapsuleId::from(target), Generation(1)),
            Err(FastMemoryError::ReservedTarget {
                target: target.into()
            })
        );
    }
    assert!(memory.fact_codes().is_empty());
    assert_eq!(recall(&memory, &codec, "monthly budget"), Recall::Unknown);
}

#[test]
fn a_memory_bound_to_a_key_projection_reads_only_queries_that_state_it() {
    let codec = Codec::new();
    let digest = codec.keys.digest();
    let mut memory = FastMemory::with_projection(config(), codec.values, digest).unwrap();
    assert_eq!(memory.projection_digest(), Some(digest));
    let mut journal = Vec::new();
    for (cue, capsule) in [
        ("home city", "pref-city"),
        ("favourite colour", "pref-color"),
    ] {
        let request = codec.write(cue, capsule, 1);
        let receipt = memory.write(request.clone()).unwrap();
        journal.push((receipt.seq, request));
    }
    let restored =
        FastMemory::restore_with_projection(config(), codec.values, digest, journal.clone())
            .unwrap();
    assert_eq!(restored.projection_digest(), Some(digest));

    // Same heads and head width, another seed: the query has the right
    // shape, and it used to be read and scored against keys it was not
    // projected like.
    let other = key_projection(12);
    let foreign = Query::project(&config(), &other, &embed("home city")).unwrap();
    assert_eq!(foreign.projection_digest(), Some(other.digest()));
    let unstated = codec.query("home city");
    assert_eq!(unstated.projection_digest(), None);
    for bound in [&memory, &restored] {
        assert_eq!(
            bound.read_admitted(&foreign, |_| true).unwrap_err(),
            FastMemoryError::ProjectionMismatch {
                expected: digest,
                actual: Some(other.digest()),
            }
        );
        assert_eq!(
            bound.read_admitted(&unstated, |_| true).unwrap_err(),
            FastMemoryError::ProjectionMismatch {
                expected: digest,
                actual: None,
            }
        );
        // A query projected by the memory's own projection is read, and so
        // is a raw one that states its digest.
        for query in [
            codec.projected_query("home city"),
            Query::with_projection(
                &config(),
                digest,
                codec.keys.project(&embed("home city")).unwrap(),
            )
            .unwrap(),
        ] {
            let readout = bound.read_admitted(&query, |_| true).unwrap();
            match decode_readout(&readout, &bound.fact_codes(), policy()).unwrap() {
                Recall::Hits(hits) => assert_eq!(hits[0].capsule, CapsuleId::from("pref-city")),
                Recall::Unknown => panic!("the fact written under the cue is recalled"),
            }
        }
    }
    assert_eq!(
        FastMemoryError::ProjectionMismatch {
            expected: digest,
            actual: None
        }
        .code(),
        "PTR_FASTMEM_PROJECTION_MISMATCH"
    );

    // A memory created or restored without a projection states none and
    // reads a query of its head shape from any projection.
    let unbound = FastMemory::restore(config(), codec.values, journal).unwrap();
    assert_eq!(unbound.projection_digest(), None);
    assert!(unbound.read_admitted(&foreign, |_| true).is_ok());
    assert!(unbound.read_admitted(&unstated, |_| true).is_ok());

    // A projection of another head shape does not project a query at all.
    let narrower = SeededProjection::new(ProjectionSpec {
        input_dim: EMBEDDING_DIM,
        heads: HEADS * 2,
        head_dim: HEAD_DIM / 2,
        seed: 11,
    })
    .unwrap();
    assert_eq!(
        Query::project(&config(), &narrower, &embed("home city")).unwrap_err(),
        FastMemoryError::DimensionMismatch {
            field: "projection_heads",
            expected: HEADS,
            actual: HEADS * 2,
        }
    );
}
