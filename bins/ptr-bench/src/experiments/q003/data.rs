//! The frozen inputs: the text pool and the sentence vectors of `research/embeddings`.
//!
//! Every file is hashed at start-up and compared with its pin; a mismatch, a pool row whose token
//! count differs from the recount, or a vectors file of the wrong size is a harness-integrity
//! failure (exit 2). The vectors are little-endian IEEE half, one row per pool line.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ptr_ledger::integrity::sha256;

use super::bridge::parse_json;
use super::params;
use crate::experiments::layered_kv::emit::{resolve, Json};

/// The repository root, from the compile-time location of this crate.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Fact,
    Query,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub text: String,
    pub kind: Kind,
    pub cue: usize,
    pub version: u8,
    pub paraphrase: u8,
}

/// Half precision to single precision, exactly.
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0f32 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// A row-major matrix of single-precision values.
pub struct Vectors {
    pub dims: usize,
    values: Vec<f32>,
}

impl Vectors {
    pub fn row(&self, index: usize) -> &[f32] {
        &self.values[index * self.dims..(index + 1) * self.dims]
    }

    #[cfg(test)]
    pub fn rows(&self) -> usize {
        self.values.len() / self.dims
    }

    /// Decode `bytes` (little-endian half) as rows of `dims` values; every value must be finite.
    pub fn decode(bytes: &[u8], dims: usize, rows: usize) -> Result<Self, String> {
        if bytes.len() != rows * dims * 2 {
            return Err(format!(
                "{} bytes, expected {rows} rows x {dims} dims x 2",
                bytes.len()
            ));
        }
        let (pairs, _) = bytes.as_chunks::<2>();
        let values: Vec<f32> = pairs
            .iter()
            .map(|pair| half_to_f32(u16::from_le_bytes(*pair)))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("a vector holds a value that is not finite".to_string());
        }
        Ok(Self { dims, values })
    }
}

/// The pool, the MiniLM vectors and, when the strong arm needs them, the bge-m3 vectors.
pub struct Data {
    pub rows: Vec<Row>,
    pub minilm: Vectors,
    pub strong: Option<Vectors>,
    facts: HashMap<(usize, u8), usize>,
    queries: HashMap<(usize, u8), usize>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub cues: usize,
    pub pool_sha256: String,
    pub vectors_sha256: String,
    pub strong_vectors_sha256: Option<String>,
}

/// The tokens of `text` under the pool's pattern `[A-Za-z0-9]+|[^\sA-Za-z0-9]`.
pub fn count_tokens(text: &str) -> usize {
    let mut count = 0;
    let mut in_word = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !character.is_whitespace() {
                count += 1;
            }
        }
    }
    count
}

/// A digest of the files of the directory `relative` under `root` (paths and contents, sorted,
/// `__pycache__` skipped): what a baseline's implementation is, for the resume header. Not the
/// repository gate's `directory_digest` (that one reads git), only a stable fingerprint.
pub fn directory_fingerprint(root: &Path, relative: &str) -> Result<String, String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<(String, [u8; 32])>) -> Result<(), String> {
        let entries =
            std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.file_name().is_some_and(|name| name == "__pycache__") {
                continue;
            }
            if path.is_dir() {
                walk(base, &path, out)?;
            } else {
                let bytes =
                    std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
                let name = path
                    .strip_prefix(base)
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.push((name, sha256(&bytes)));
            }
        }
        Ok(())
    }
    let base = root.join(relative);
    let mut files = Vec::new();
    walk(&base, &base, &mut files)?;
    files.sort();
    let mut text = String::new();
    for (name, digest) in files {
        text.push_str(&format!("{name}\0{}\n", hex(&digest)));
    }
    Ok(hex(&sha256(text.as_bytes())))
}

/// The checkout the harness runs from: `git rev-parse HEAD`, and whether the harness, the crates or
/// the baselines have uncommitted changes (`unknown` when git cannot say).
pub fn checkout_state(root: &Path) -> (String, String) {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    let head = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = match git(&[
        "status",
        "--porcelain",
        "--",
        "bins/ptr-bench",
        "crates",
        "research/baselines",
        "research/embeddings",
    ]) {
        Some(text) if text.is_empty() => "clean".to_string(),
        Some(_) => "dirty".to_string(),
        None => "unknown".to_string(),
    };
    (head, dirty)
}

fn read_checked(root: &Path, path: &str, pinned: &str) -> Result<(Vec<u8>, String), String> {
    let bytes =
        std::fs::read(root.join(path)).map_err(|error| format!("cannot read {path}: {error}"))?;
    let digest = hex(&sha256(&bytes));
    if digest != pinned {
        return Err(format!(
            "{path} has SHA-256 {digest}, the pin is {pinned}: the frozen input changed"
        ));
    }
    Ok((bytes, digest))
}

fn int_field(row: &Json, key: &str, line: usize) -> Result<i64, String> {
    match resolve(row, key) {
        Some(Json::Int(value)) => Ok(*value),
        other => Err(format!("pool line {line}: {key} is {other:?}")),
    }
}

fn str_field<'a>(row: &'a Json, key: &str, line: usize) -> Result<&'a str, String> {
    match resolve(row, key) {
        Some(Json::Str(value)) => Ok(value),
        other => Err(format!("pool line {line}: {key} is {other:?}")),
    }
}

impl Data {
    /// Load and verify the pool and the vectors; `strong` also loads the bge-m3 vectors.
    pub fn load(root: &Path, strong: bool) -> Result<Self, String> {
        let (pool_bytes, pool_sha256) = read_checked(root, params::POOL_PATH, params::POOL_SHA256)?;
        let text =
            String::from_utf8(pool_bytes).map_err(|_| "the pool is not UTF-8".to_string())?;
        let mut rows = Vec::with_capacity(params::POOL_ROWS);
        for (line, content) in text.lines().enumerate() {
            let parsed =
                parse_json(content).map_err(|error| format!("pool line {line}: {error}"))?;
            let kind = match str_field(&parsed, "kind", line)? {
                "fact" => Kind::Fact,
                "query" => Kind::Query,
                other => {
                    return Err(format!(
                        "pool line {line}: kind {other:?} is not used by Q003"
                    ))
                }
            };
            let row = Row {
                text: str_field(&parsed, "text", line)?.to_string(),
                kind,
                cue: usize::try_from(int_field(&parsed, "cue", line)?)
                    .map_err(|_| format!("pool line {line}: negative cue"))?,
                version: u8::try_from(int_field(&parsed, "version", line)?)
                    .map_err(|_| format!("pool line {line}: version out of range"))?,
                paraphrase: u8::try_from(int_field(&parsed, "paraphrase", line)?)
                    .map_err(|_| format!("pool line {line}: paraphrase out of range"))?,
            };
            let tokens = int_field(&parsed, "tokens", line)?;
            if usize::try_from(tokens).ok() != Some(count_tokens(&row.text)) {
                return Err(format!(
                    "pool line {line}: tokens {tokens} but the recount of {:?} is {}",
                    row.text,
                    count_tokens(&row.text)
                ));
            }
            if !row.text.bytes().all(|byte| (0x20..0x7f).contains(&byte)) {
                return Err(format!("pool line {line}: the text is not printable ASCII"));
            }
            rows.push(row);
        }
        if rows.len() != params::POOL_ROWS {
            return Err(format!(
                "the pool has {} rows, expected {}",
                rows.len(),
                params::POOL_ROWS
            ));
        }
        let (vector_bytes, vectors_sha256) =
            read_checked(root, params::VECTORS_PATH, params::VECTORS_SHA256)?;
        let minilm = Vectors::decode(&vector_bytes, params::EMBEDDING_DIMENSIONS, rows.len())
            .map_err(|error| format!("{}: {error}", params::VECTORS_PATH))?;
        let (strong_vectors, strong_vectors_sha256) = if strong {
            let (bytes, digest) = read_checked(
                root,
                params::STRONG_VECTORS_PATH,
                params::STRONG_VECTORS_SHA256,
            )?;
            let vectors = Vectors::decode(&bytes, params::STRONG_DIMENSIONS, rows.len())
                .map_err(|error| format!("{}: {error}", params::STRONG_VECTORS_PATH))?;
            (Some(vectors), Some(digest))
        } else {
            (None, None)
        };
        let mut facts = HashMap::new();
        let mut queries = HashMap::new();
        let mut cues = 0;
        for (index, row) in rows.iter().enumerate() {
            cues = cues.max(row.cue + 1);
            let slot = match row.kind {
                Kind::Fact => facts.insert((row.cue, row.version), index),
                Kind::Query => queries.insert((row.cue, row.paraphrase), index),
            };
            if slot.is_some() {
                return Err(format!(
                    "pool row {index} repeats cue {} version {}",
                    row.cue, row.version
                ));
            }
        }
        for cue in 0..cues {
            for version in 1..=3u8 {
                if !facts.contains_key(&(cue, version)) {
                    return Err(format!("cue {cue} has no fact version {version}"));
                }
            }
            for paraphrase in 0..3u8 {
                if !queries.contains_key(&(cue, paraphrase)) {
                    return Err(format!("cue {cue} has no query paraphrase {paraphrase}"));
                }
            }
        }
        Ok(Self {
            rows,
            minilm,
            strong: strong_vectors,
            facts,
            queries,
            cues,
            pool_sha256,
            vectors_sha256,
            strong_vectors_sha256,
        })
    }

    /// The pool row of the fact `version` (the generation) of `cue`.
    pub fn fact(&self, cue: usize, generation: u64) -> usize {
        self.facts[&(cue, u8::try_from(generation).expect("generation 1..=3"))]
    }

    pub fn query(&self, cue: usize, paraphrase: u8) -> usize {
        self.queries[&(cue, paraphrase)]
    }

    pub fn text(&self, row: usize) -> &str {
        &self.rows[row].text
    }

    pub fn embedding(&self, row: usize) -> &[f32] {
        self.minilm.row(row)
    }

    pub fn strong_embedding(&self, row: usize) -> Option<&[f32]> {
        self.strong.as_ref().map(|vectors| vectors.row(row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_fingerprint_follows_its_files_and_skips_caches() {
        let dir = std::env::temp_dir().join(format!("q003-fp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("d/__pycache__")).unwrap();
        std::fs::write(dir.join("d/a.py"), "x").unwrap();
        let first = directory_fingerprint(&dir, "d").unwrap();
        std::fs::write(dir.join("d/__pycache__/a.pyc"), "junk").unwrap();
        assert_eq!(directory_fingerprint(&dir, "d").unwrap(), first);
        std::fs::write(dir.join("d/a.py"), "y").unwrap();
        assert_ne!(directory_fingerprint(&dir, "d").unwrap(), first);
        assert!(directory_fingerprint(&dir, "missing").is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let (head, dirty) = checkout_state(&repo_root());
        assert!(!head.is_empty() && !dirty.is_empty());
    }

    #[test]
    fn half_precision_decodes_exactly() {
        for (bits, value) in [
            (0x0000u16, 0.0f32),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x3800, 0.5),
            (0x0001, 2f32.powi(-24)),
            (0x7bff, 65504.0),
            (0x3555, 0.333_251_95),
        ] {
            assert_eq!(half_to_f32(bits), value, "{bits:#06x}");
        }
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn tokens_are_counted_as_the_pool_does() {
        assert_eq!(
            count_tokens("Which language does Marta Jessop speak at home?"),
            9
        );
        assert_eq!(count_tokens("a-b  c"), 4);
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn a_vectors_file_of_the_wrong_size_or_with_a_nonfinite_value_is_refused() {
        assert!(Vectors::decode(&[0; 7], 2, 2).is_err());
        assert!(Vectors::decode(&[0, 0x7c, 0, 0], 1, 2).is_err());
        let ok = Vectors::decode(&[0, 0x3c, 0, 0xbc], 1, 2).unwrap();
        assert_eq!(ok.row(1), [-1.0]);
        assert_eq!(ok.rows(), 2);
    }

    #[test]
    fn the_committed_inputs_match_their_pins_and_the_manifest() {
        let data = Data::load(&repo_root(), true).expect("the frozen inputs load");
        assert_eq!(data.cues, params::CAPSULES);
        assert_eq!(data.rows.len(), params::POOL_ROWS);
        let manifest: toml::Table =
            std::fs::read_to_string(repo_root().join("research/embeddings/MANIFEST.toml"))
                .unwrap()
                .parse()
                .unwrap();
        assert_eq!(
            manifest["pools"]["q003"]["sha256"].as_str(),
            Some(params::POOL_SHA256)
        );
        assert_eq!(
            manifest["vectors"]["minilm-q003"]["sha256"].as_str(),
            Some(params::VECTORS_SHA256)
        );
        assert_eq!(
            manifest["vectors"]["bge-m3-q003"]["sha256"].as_str(),
            Some(params::STRONG_VECTORS_SHA256)
        );
        let model = &manifest["models"]["minilm"];
        assert_eq!(model["id"].as_str(), Some(params::EMBEDDING_MODEL));
        assert_eq!(model["revision"].as_str(), Some(params::EMBEDDING_REVISION));
        // The vectors are unit length in half precision (to the pool tool's own tolerance).
        for row in [0, 5, 11_999] {
            let norm: f32 = data
                .minilm
                .row(row)
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() < 2e-3, "row {row}: {norm}");
        }
        assert_eq!(data.fact(0, 2), 1);
        assert_eq!(
            data.text(data.fact(0, 1)),
            "Marta Jessop speaks Polish at home."
        );
    }
}
