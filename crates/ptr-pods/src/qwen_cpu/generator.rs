//! Greedy, bounded generation over the CPU model, behind [`TextGenerator`].

use std::fmt;
use std::io::Read as _;
use std::path::Path;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::config::QwenModelConfig;
use super::model::Model;
use super::tokenizer::QwenTokenizer;
use super::weights::Weights;
use crate::{
    GenerationError, GenerationOutput, GenerationRequest, ModelIdentity, StopReason, TextGenerator,
    MAX_NEW_TOKENS_LIMIT,
};

/// Default bound on `model.safetensors` (4 GiB), checked from the file size before any read.
pub const DEFAULT_MAX_WEIGHTS_BYTES: u64 = 4 << 30;
/// Bound on `config.json` and `tokenizer_config.json`.
const MAX_SMALL_FILE_BYTES: u64 = 1 << 20;
/// Bound on `tokenizer.json` (the Qwen2.5 file is about 7 MB).
const MAX_TOKENIZER_BYTES: u64 = 64 << 20;

/// Why loading or running the CPU model failed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum QwenCpuError {
    Io(String),
    Config(String),
    Weights(String),
    Tokenizer(String),
    Input(String),
    /// The digest of a file is not the one the caller pinned.
    DigestMismatch {
        /// Which file (`weights`, `config`, `tokenizer`, `tokenizer_config`).
        file: &'static str,
        expected: String,
        actual: String,
    },
    /// A file is larger than the configured bound; nothing was read.
    TooLarge {
        file: &'static str,
        bytes: u64,
        limit: u64,
    },
    /// A logit was not finite.
    NonFinite,
}

impl fmt::Display for QwenCpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(why) => write!(f, "io: {why}"),
            Self::Config(why) => write!(f, "config: {why}"),
            Self::Weights(why) => write!(f, "weights: {why}"),
            Self::Tokenizer(why) => write!(f, "tokenizer: {why}"),
            Self::Input(why) => write!(f, "input: {why}"),
            Self::DigestMismatch {
                file,
                expected,
                actual,
            } => write!(f, "{file} sha256 {actual}, expected {expected}"),
            Self::TooLarge { file, bytes, limit } => {
                write!(f, "{file} is {bytes} bytes, the limit is {limit}")
            }
            Self::NonFinite => f.write_str("a logit was not finite"),
        }
    }
}

impl std::error::Error for QwenCpuError {}

/// How to load and run the model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QwenCpuOptions {
    /// Worker threads for the matrix products (at least 1).
    pub threads: usize,
    /// Hex SHA-256 the weights file must have; checked before parsing when set.
    pub expected_weights_sha256: Option<String>,
    /// Hex SHA-256 of `config.json`; checked before parsing when set.
    pub expected_config_sha256: Option<String>,
    /// Hex SHA-256 of `tokenizer.json`; checked before parsing when set.
    pub expected_tokenizer_sha256: Option<String>,
    /// Hex SHA-256 of `tokenizer_config.json` (not read by the generator, but pinned and
    /// recorded like the others); when set the file must exist and match.
    pub expected_tokenizer_config_sha256: Option<String>,
    /// Largest `model.safetensors` accepted, in bytes; the size is checked from the file
    /// metadata before the file is read. Peak memory while loading is about twice the file
    /// (the raw bytes and the converted matrices), so a limit of N allows a peak of about 2N.
    pub max_weights_bytes: u64,
    /// Longest prompt in tokens (prompt plus generated tokens must also fit the context).
    pub max_prompt_tokens: usize,
}

impl Default for QwenCpuOptions {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            expected_weights_sha256: None,
            expected_config_sha256: None,
            expected_tokenizer_sha256: None,
            expected_tokenizer_config_sha256: None,
            max_weights_bytes: DEFAULT_MAX_WEIGHTS_BYTES,
            max_prompt_tokens: 2048,
        }
    }
}

impl QwenCpuOptions {
    /// Pin the SHA-256 (hex) the weights file must have.
    #[must_use]
    pub fn with_expected_weights_sha256(mut self, sha256_hex: impl Into<String>) -> Self {
        self.expected_weights_sha256 = Some(sha256_hex.into());
        self
    }

    /// Pin the SHA-256 (hex) of `config.json`.
    #[must_use]
    pub fn with_expected_config_sha256(mut self, sha256_hex: impl Into<String>) -> Self {
        self.expected_config_sha256 = Some(sha256_hex.into());
        self
    }

    /// Pin the SHA-256 (hex) of `tokenizer.json`.
    #[must_use]
    pub fn with_expected_tokenizer_sha256(mut self, sha256_hex: impl Into<String>) -> Self {
        self.expected_tokenizer_sha256 = Some(sha256_hex.into());
        self
    }

    /// Pin the SHA-256 (hex) of `tokenizer_config.json`.
    #[must_use]
    pub fn with_expected_tokenizer_config_sha256(mut self, sha256_hex: impl Into<String>) -> Self {
        self.expected_tokenizer_config_sha256 = Some(sha256_hex.into());
        self
    }

    /// Refuse a weights file larger than `bytes`.
    #[must_use]
    pub fn with_max_weights_bytes(mut self, bytes: u64) -> Self {
        self.max_weights_bytes = bytes;
        self
    }

    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
}

/// Counters of one generator, for throughput measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct QwenCpuStats {
    pub requests: u64,
    pub prompt_tokens_fed: u64,
    pub prompt_tokens_reused: u64,
    pub generated_tokens: u64,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

impl QwenCpuStats {
    /// Generated tokens per second of decode time (the first token comes from prefill).
    #[must_use]
    pub fn decode_tokens_per_second(&self) -> f64 {
        let decoded = self.generated_tokens.saturating_sub(self.requests);
        let seconds = self.decode_time.as_secs_f64();
        if seconds > 0.0 {
            decoded as f64 / seconds
        } else {
            0.0
        }
    }
}

/// The SHA-256 digests (lowercase hex) of the files a [`QwenCpu`] was loaded from.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct QwenDigests {
    pub weights: String,
    pub config: String,
    pub tokenizer: String,
    /// `None` when the checkpoint has no `tokenizer_config.json`.
    pub tokenizer_config: Option<String>,
}

impl QwenDigests {
    fn identity(&self) -> ModelIdentity {
        let mut files = vec![
            ("weights".to_owned(), self.weights.clone()),
            ("config".to_owned(), self.config.clone()),
            ("tokenizer".to_owned(), self.tokenizer.clone()),
        ];
        if let Some(digest) = &self.tokenizer_config {
            files.push(("tokenizer_config".to_owned(), digest.clone()));
        }
        ModelIdentity::new(files)
    }
}

/// A Qwen2-architecture model on the CPU, with its tokenizer.
///
/// `generation_config.json` of the checkpoint (sampling, repetition penalty, ...) is **not**
/// read: generation is always greedy argmax with the stop tokens and bound of the request.
pub struct QwenCpu {
    model: Model,
    tokenizer: QwenTokenizer,
    digests: QwenDigests,
    /// The tokens whose keys and values are in the cache, in order.
    cached: Vec<u32>,
    max_prompt_tokens: usize,
    stats: QwenCpuStats,
}

impl fmt::Debug for QwenCpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenCpu")
            .field("config", self.model.config())
            .field("cached_tokens", &self.cached.len())
            .finish_non_exhaustive()
    }
}

fn io_error(path: &Path, error: &std::io::Error) -> QwenCpuError {
    QwenCpuError::Io(format!("{}: {error}", path.display()))
}

/// Lowercase hex of a SHA-256 digest.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Read `path` (at most `limit` bytes, judged from the file metadata first and again while
/// reading), hash it while reading, and compare with `expected` before returning the bytes.
fn read_hashed(
    path: &Path,
    file: &'static str,
    limit: u64,
    expected: Option<&str>,
) -> Result<(Vec<u8>, String), QwenCpuError> {
    let mut handle = std::fs::File::open(path).map_err(|e| io_error(path, &e))?;
    let size = handle.metadata().map_err(|e| io_error(path, &e))?.len();
    if size > limit {
        return Err(QwenCpuError::TooLarge {
            file,
            bytes: size,
            limit,
        });
    }
    let capacity = usize::try_from(size).map_err(|_| {
        QwenCpuError::Io(format!("{}: too large for this platform", path.display()))
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut hasher = Sha256::new();
    let mut chunk = vec![0_u8; 1 << 20];
    loop {
        let read = handle.read(&mut chunk).map_err(|e| io_error(path, &e))?;
        if read == 0 {
            break;
        }
        // The file may have grown since the metadata was read.
        if (bytes.len() + read) as u64 > limit {
            return Err(QwenCpuError::TooLarge {
                file,
                bytes: (bytes.len() + read) as u64,
                limit,
            });
        }
        hasher.update(&chunk[..read]);
        bytes.extend_from_slice(&chunk[..read]);
    }
    let actual = hex(&hasher.finalize());
    if let Some(expected) = expected {
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(QwenCpuError::DigestMismatch {
                file,
                expected: expected.to_owned(),
                actual,
            });
        }
    }
    Ok((bytes, actual))
}

impl QwenCpu {
    /// Load `config.json`, `tokenizer.json`, `tokenizer_config.json` (when present, for its
    /// digest only) and `model.safetensors` from `dir`.
    ///
    /// Every file is size-checked from its metadata before it is read, hashed while it is read,
    /// and compared with the pinned digest (when one is set) before it is parsed. Peak memory is
    /// about twice the weights file: the raw bytes and the converted matrices exist together,
    /// the raw bytes are dropped as soon as the matrices are built.
    ///
    /// # Errors
    /// [`QwenCpuError`] for unreadable or oversized files, an unsupported configuration, missing
    /// or mis-shaped tensors, or a digest that differs from the pinned one.
    pub fn load(dir: &Path, options: &QwenCpuOptions) -> Result<Self, QwenCpuError> {
        // The cheapest refusal first: the weights size, from metadata alone.
        let weights_path = dir.join("model.safetensors");
        let weights_size = std::fs::metadata(&weights_path)
            .map_err(|e| io_error(&weights_path, &e))?
            .len();
        if weights_size > options.max_weights_bytes {
            return Err(QwenCpuError::TooLarge {
                file: "weights",
                bytes: weights_size,
                limit: options.max_weights_bytes,
            });
        }
        let (config_bytes, config_digest) = read_hashed(
            &dir.join("config.json"),
            "config",
            MAX_SMALL_FILE_BYTES,
            options.expected_config_sha256.as_deref(),
        )?;
        let config_text = std::str::from_utf8(&config_bytes)
            .map_err(|_| QwenCpuError::Config("config.json is not UTF-8".into()))?;
        let config = QwenModelConfig::from_json(config_text)?;
        drop(config_bytes);

        let (tokenizer_bytes, tokenizer_digest) = read_hashed(
            &dir.join("tokenizer.json"),
            "tokenizer",
            MAX_TOKENIZER_BYTES,
            options.expected_tokenizer_sha256.as_deref(),
        )?;
        let tokenizer = QwenTokenizer::from_bytes(&tokenizer_bytes)?;
        drop(tokenizer_bytes);

        let tokenizer_config_path = dir.join("tokenizer_config.json");
        let tokenizer_config = if tokenizer_config_path.exists()
            || options.expected_tokenizer_config_sha256.is_some()
        {
            Some(
                read_hashed(
                    &tokenizer_config_path,
                    "tokenizer_config",
                    MAX_SMALL_FILE_BYTES,
                    options.expected_tokenizer_config_sha256.as_deref(),
                )?
                .1,
            )
        } else {
            None
        };

        let (bytes, weights_digest) = read_hashed(
            &weights_path,
            "weights",
            options.max_weights_bytes,
            options.expected_weights_sha256.as_deref(),
        )?;
        let weights = Weights::from_bytes(&bytes, &config)?;
        drop(bytes);
        let digests = QwenDigests {
            weights: weights_digest,
            config: config_digest,
            tokenizer: tokenizer_digest,
            tokenizer_config,
        };
        Ok(Self::from_parts(
            Model::new(config, weights, options.threads),
            tokenizer,
            digests,
            options.max_prompt_tokens,
        ))
    }

    fn from_parts(
        model: Model,
        tokenizer: QwenTokenizer,
        digests: QwenDigests,
        max_prompt_tokens: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            digests,
            cached: Vec::new(),
            max_prompt_tokens,
            stats: QwenCpuStats::default(),
        }
    }

    /// The digests of the files this model was loaded from.
    #[must_use]
    pub fn digests(&self) -> &QwenDigests {
        &self.digests
    }

    #[must_use]
    pub fn config(&self) -> &QwenModelConfig {
        self.model.config()
    }

    #[must_use]
    pub fn stats(&self) -> QwenCpuStats {
        self.stats
    }

    /// Token ids of the chat-templated prompt. The template markers are inserted by id and the
    /// texts are tokenized apart, so a marker string inside `system` or `user` is plain text.
    ///
    /// # Errors
    /// [`QwenCpuError::Tokenizer`] when encoding fails.
    pub fn prompt_ids(&self, system: &str, user: &str) -> Result<Vec<u32>, QwenCpuError> {
        self.tokenizer.chat_ids(system, user)
    }

    /// The logits of the position after `prompt_ids` (a fresh pass; used for parity checks).
    ///
    /// # Errors
    /// [`QwenCpuError`] for an empty or oversized prompt or a non-finite logit.
    pub fn next_token_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>, QwenCpuError> {
        self.prefill(prompt_ids)
    }

    /// Greedy continuation of `prompt_ids`: at most `max_new_tokens` tokens, stopping before an
    /// end-of-turn token (which is not returned).
    ///
    /// # Errors
    /// [`QwenCpuError`] for an empty or oversized prompt, a bound outside `1..=128`, or a
    /// non-finite logit.
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: u32,
    ) -> Result<(Vec<u32>, StopReason), QwenCpuError> {
        if max_new_tokens == 0 || max_new_tokens > MAX_NEW_TOKENS_LIMIT {
            return Err(QwenCpuError::Input(
                "max_new_tokens must be in 1..=128".into(),
            ));
        }
        let mut out = Vec::new();
        let started = Instant::now();
        let mut logits = self.prefill(prompt_ids)?;
        self.stats.prefill_time += started.elapsed();
        self.stats.requests += 1;
        let decode_started = Instant::now();
        let stop = loop {
            let next = argmax(&logits)?;
            if self.tokenizer.stop_ids().contains(&next) {
                break StopReason::EndOfTurn;
            }
            out.push(next);
            if out.len() >= max_new_tokens as usize {
                break StopReason::MaxTokens;
            }
            logits = self
                .model
                .forward(&[next], true)?
                .ok_or(QwenCpuError::NonFinite)?;
            self.cached.push(next);
        };
        self.stats.decode_time += decode_started.elapsed();
        self.stats.generated_tokens += out.len() as u64;
        Ok((out, stop))
    }

    /// Feed the part of `prompt_ids` the cache does not already hold; the last token is always
    /// recomputed so there are logits to return.
    fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>, QwenCpuError> {
        if prompt_ids.is_empty() {
            return Err(QwenCpuError::Input("empty prompt".into()));
        }
        if prompt_ids.len() > self.max_prompt_tokens {
            return Err(QwenCpuError::Input(format!(
                "prompt has {} tokens, the limit is {}",
                prompt_ids.len(),
                self.max_prompt_tokens
            )));
        }
        let shared = self
            .cached
            .iter()
            .zip(prompt_ids)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt_ids.len() - 1);
        self.model.reset_to(shared);
        self.cached.truncate(shared);
        self.stats.prompt_tokens_reused += shared as u64;
        self.stats.prompt_tokens_fed += (prompt_ids.len() - shared) as u64;
        let logits = match self.model.forward(&prompt_ids[shared..], true) {
            Ok(Some(logits)) => logits,
            Ok(None) => return Err(QwenCpuError::NonFinite),
            Err(error) => {
                // The cache may hold a partial pass; start the next request clean.
                self.model.reset_to(0);
                self.cached.clear();
                return Err(error);
            }
        };
        self.cached.extend_from_slice(&prompt_ids[shared..]);
        Ok(logits)
    }
}

/// Index of the largest logit; the lowest index wins a tie.
fn argmax(logits: &[f32]) -> Result<u32, QwenCpuError> {
    let mut best = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(QwenCpuError::NonFinite);
        }
        if *value > best_value {
            best = index;
            best_value = *value;
        }
    }
    u32::try_from(best).map_err(|_| QwenCpuError::NonFinite)
}

impl TextGenerator for QwenCpu {
    fn generate(
        &mut self,
        request: &GenerationRequest<'_>,
    ) -> Result<GenerationOutput, GenerationError> {
        request.validate()?;
        let to_error = |error: QwenCpuError| GenerationError::Backend(error.to_string());
        let ids = self
            .prompt_ids(request.system, request.user)
            .map_err(to_error)?;
        let (new_ids, stop) = self
            .generate_ids(&ids, request.max_new_tokens)
            .map_err(to_error)?;
        let (text, dropped) = self.tokenizer.decode(&new_ids).map_err(to_error)?;
        Ok(
            GenerationOutput::new(text, u32::try_from(new_ids.len()).unwrap_or(u32::MAX), stop)
                .with_dropped_special_tokens(dropped),
        )
    }

    fn identity(&self) -> Option<ModelIdentity> {
        Some(self.digests.identity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_the_first_maximum_and_refuses_non_finite_values() {
        assert_eq!(argmax(&[0.5, 2.0, 2.0, 1.0]).unwrap(), 1);
        assert_eq!(argmax(&[f32::NAN, 1.0]), Err(QwenCpuError::NonFinite));
        assert_eq!(argmax(&[1.0, f32::INFINITY]), Err(QwenCpuError::NonFinite));
    }
}
