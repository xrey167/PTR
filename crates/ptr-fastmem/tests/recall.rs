use ptr_fastmem::{
    decode_readout, Decay, DecodePolicy, FastMemory, FastMemoryConfig, IdentifierCodebook,
    ProjectionSpec, Query, Recall, SeededProjection, SourceRef, WriteRequest,
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
            keys: SeededProjection::new(ProjectionSpec {
                input_dim: EMBEDDING_DIM,
                heads: HEADS,
                head_dim: HEAD_DIM,
                seed: 11,
            })
            .unwrap(),
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
    decode_readout(&readout, &memory.fact_codes(&codec.values), policy()).unwrap()
}

#[test]
fn a_cue_recalls_the_fact_written_under_it_as_a_named_capsule() {
    let codec = Codec::new();
    let mut memory = FastMemory::new(config()).unwrap();
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
    let mut memory = FastMemory::new(config()).unwrap();
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
    let mut memory = FastMemory::new(config()).unwrap();
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
    let mut memory = FastMemory::new(config()).unwrap();
    memory
        .write(codec.write("home city", "pref-city", 1))
        .unwrap();
    memory
        .write(codec.write("favourite colour", "pref-color", 1))
        .unwrap();
    memory.revoke(|source| source.key == "pref-city");
    assert!(memory
        .fact_codes(&codec.values)
        .iter()
        .all(|fact| fact.capsule != CapsuleId::from("pref-city")));
    assert_eq!(recall(&memory, &codec, "home city"), Recall::Unknown);
}

#[test]
fn constraint_and_procedure_sources_are_never_decoded_as_capsules() {
    // Sources use the lifecycle target vocabulary, in which hard constraints
    // and procedures are targets but not capsules. Their writes still shape the
    // state and gate reads, but must never decode to a hit for a fabricated
    // capsule such as `constraint:budget`.
    let codec = Codec::new();
    let mut memory = FastMemory::new(config()).unwrap();
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
        .fact_codes(&codec.values)
        .into_iter()
        .map(|fact| fact.capsule)
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
