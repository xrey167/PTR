//! Checkpoint save/load of the CPU reference model.
//!
//! # Format
//!
//! The file is a `ptr-types` [`CheckpointHeader`] (format 3: magic, model name,
//! codebook version and verbatim assignment, slot-encoding version, embedded
//! table sizes, and an architecture binding built from
//! [`CoreConfig::architecture_bytes`]) followed by the payload below. Every
//! integer and float is little endian.
//!
//! ```text
//! u32   parameter count N
//! N x   manifest entry: u16 name_len, name (UTF-8), u32 rows, u32 cols,
//!                       u32 CRC-32 (IEEE) of that tensor's data bytes
//! N x   tensor data:    rows * cols f32, row-major, in manifest order
//! u32   CRC-32 (IEEE) of every preceding payload byte
//! ```
//!
//! The CRC-32 values detect accidental corruption and mismatched writers. They
//! are not a cryptographic digest and give no authenticity: anyone who can edit
//! the file can recompute them.

use crate::config::{ConfigError, CoreConfig};
use crate::core_model::{CoreModel, CODEBOOK};
use crate::error::CoreError;
use crate::tensor::Matrix;
use ptr_types::{
    CheckpointError, CheckpointHeader, CodeFamily, EpistemicState, ReasoningOperator, SemanticRole,
    SlotEncoding, TableSize, UncertaintyKind,
};

/// Model name recorded in the header.
pub const MODEL_NAME: &str = "ptr-core-cpu-reference";

const REQUIRED_FAMILIES: [CodeFamily; 4] = [
    CodeFamily::SemanticRole,
    CodeFamily::EpistemicState,
    CodeFamily::UncertaintyKind,
    CodeFamily::ReasoningOperator,
];

/// Why a checkpoint was not written or not accepted.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckpointIoError {
    /// The header was malformed or disagrees with this build's codebook.
    Header(CheckpointError),
    /// The header names a different model.
    WrongModel { found: String },
    /// The architecture binding differs from the requested config.
    ArchitectureMismatch,
    /// The requested config is invalid.
    Config(ConfigError),
    /// Building the skeleton model failed.
    Model(CoreError),
    /// The payload ends inside a field.
    Truncated { field: &'static str },
    /// Bytes remain after the final checksum.
    TrailingBytes { extra: usize },
    /// The manifest lists a different number of parameters.
    ManifestCount { expected: usize, found: usize },
    /// A manifest entry has an unexpected name or is not UTF-8.
    ManifestName {
        index: usize,
        expected: &'static str,
        found: String,
    },
    /// A tensor's recorded shape differs from the model's.
    ShapeMismatch {
        name: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A tensor's data does not match its recorded checksum.
    ChecksumMismatch { name: &'static str },
    /// The whole-payload checksum does not match.
    PayloadChecksum,
    /// A tensor holds a NaN or infinity.
    NonFinite { name: &'static str },
}

impl std::fmt::Display for CheckpointIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CheckpointIoError {}

impl From<CheckpointError> for CheckpointIoError {
    fn from(e: CheckpointError) -> Self {
        Self::Header(e)
    }
}

/// CRC-32 (IEEE 802.3, reflected, polynomial `0xEDB88320`).
pub fn crc32(bytes: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0_u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    let mut crc = 0xFFFF_FFFF_u32;
    for b in bytes {
        crc = TABLE[((crc ^ u32::from(*b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn tensor_bytes(m: &Matrix) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.as_slice().len() * 4);
    for x in m.as_slice() {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn header_for(model: &CoreModel) -> CheckpointHeader {
    let tables = [
        TableSize {
            family: CodeFamily::SemanticRole,
            rows: CODEBOOK.cardinality_of::<SemanticRole>(),
        },
        TableSize {
            family: CodeFamily::EpistemicState,
            rows: CODEBOOK.cardinality_of::<EpistemicState>(),
        },
        TableSize {
            family: CodeFamily::UncertaintyKind,
            rows: CODEBOOK.cardinality_of::<UncertaintyKind>(),
        },
        TableSize {
            family: CodeFamily::ReasoningOperator,
            rows: CODEBOOK.cardinality_of::<ReasoningOperator>(),
        },
    ];
    CheckpointHeader::new(MODEL_NAME, &CODEBOOK, SlotEncoding::V1, &tables)
        .with_architecture(model.config().architecture_bytes())
}

/// Serialize every parameter of `model` (see the module documentation).
pub fn save_checkpoint(model: &CoreModel) -> Vec<u8> {
    let params = model.named_params();
    let mut payload = Vec::new();
    payload.extend_from_slice(&(params.len() as u32).to_le_bytes());
    let blobs: Vec<Vec<u8>> = params.iter().map(|(_, m)| tensor_bytes(m)).collect();
    for ((name, m), blob) in params.iter().zip(&blobs) {
        payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
        payload.extend_from_slice(name.as_bytes());
        payload.extend_from_slice(&(m.rows() as u32).to_le_bytes());
        payload.extend_from_slice(&(m.cols() as u32).to_le_bytes());
        payload.extend_from_slice(&crc32(blob).to_le_bytes());
    }
    for blob in &blobs {
        payload.extend_from_slice(blob);
    }
    let total = crc32(&payload);
    payload.extend_from_slice(&total.to_le_bytes());
    header_for(model).write(&payload)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, field: &'static str) -> Result<&'a [u8], CheckpointIoError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or(CheckpointIoError::Truncated { field })?;
        let s = self
            .bytes
            .get(self.at..end)
            .ok_or(CheckpointIoError::Truncated { field })?;
        self.at = end;
        Ok(s)
    }

    fn u16(&mut self, field: &'static str) -> Result<u16, CheckpointIoError> {
        let b = self.take(2, field)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, CheckpointIoError> {
        let b = self.take(4, field)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

struct Entry {
    name: String,
    rows: usize,
    cols: usize,
    crc: u32,
}

/// Rebuild a model from checkpoint bytes, validating everything first.
///
/// `config` is the configuration the caller intends to run; the file must have
/// been written under the same architecture. Rejections are typed: wrong magic
/// or codebook (header), another model name, another architecture, a different
/// manifest, a shape that differs, or a checksum that does not match. Loaded
/// parameters must all be finite.
pub fn load_checkpoint(bytes: &[u8], config: &CoreConfig) -> Result<CoreModel, CheckpointIoError> {
    config.validate().map_err(CheckpointIoError::Config)?;
    let (header, payload) = CheckpointHeader::read(bytes)?;
    if header.model != MODEL_NAME {
        return Err(CheckpointIoError::WrongModel {
            found: header.model,
        });
    }
    header.verify(&CODEBOOK, SlotEncoding::V1, &REQUIRED_FAMILIES)?;
    if header.architecture != config.architecture_bytes() {
        return Err(CheckpointIoError::ArchitectureMismatch);
    }
    let mut model = CoreModel::new(config.clone()).map_err(CheckpointIoError::Model)?;
    let expected: Vec<(&'static str, (usize, usize))> = model
        .named_params()
        .iter()
        .map(|(n, m)| (*n, m.shape()))
        .collect();

    let (body, tail) = payload.split_at(payload.len().saturating_sub(4));
    let mut r = Reader { bytes: body, at: 0 };
    let count = r.u32("parameter_count")? as usize;
    if count != expected.len() {
        return Err(CheckpointIoError::ManifestCount {
            expected: expected.len(),
            found: count,
        });
    }
    let mut manifest = Vec::with_capacity(count);
    for (index, (name, _)) in expected.iter().enumerate() {
        let len = usize::from(r.u16("name_len")?);
        let found = String::from_utf8_lossy(r.take(len, "name")?).into_owned();
        if found != *name {
            return Err(CheckpointIoError::ManifestName {
                index,
                expected: name,
                found,
            });
        }
        manifest.push(Entry {
            name: found,
            rows: r.u32("rows")? as usize,
            cols: r.u32("cols")? as usize,
            crc: r.u32("tensor_crc")?,
        });
    }
    for (entry, (name, shape)) in manifest.iter().zip(&expected) {
        if (entry.rows, entry.cols) != *shape {
            return Err(CheckpointIoError::ShapeMismatch {
                name,
                expected: *shape,
                found: (entry.rows, entry.cols),
            });
        }
    }
    let mut tensors: Vec<Vec<f32>> = Vec::with_capacity(count);
    for (entry, (name, shape)) in manifest.iter().zip(&expected) {
        let blob = r.take(shape.0 * shape.1 * 4, "tensor_data")?;
        if crc32(blob) != entry.crc {
            return Err(CheckpointIoError::ChecksumMismatch { name });
        }
        let data: Vec<f32> = blob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if !data.iter().all(|x| x.is_finite()) {
            return Err(CheckpointIoError::NonFinite { name });
        }
        debug_assert_eq!(entry.name, *name);
        tensors.push(data);
    }
    if r.at != body.len() {
        return Err(CheckpointIoError::TrailingBytes {
            extra: body.len() - r.at,
        });
    }
    let mut tail_reader = Reader { bytes: tail, at: 0 };
    let stored_total = tail_reader.u32("payload_crc")?;
    if crc32(body) != stored_total {
        return Err(CheckpointIoError::PayloadChecksum);
    }
    for ((_, dst), data) in model.named_params_mut().into_iter().zip(tensors) {
        dst.as_mut_slice().copy_from_slice(&data);
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CoreConfig {
        CoreConfig {
            d_model: 8,
            heads: 2,
            d_ff: 16,
            max_slots: 6,
            diff_vocab: 8,
            ..CoreConfig::default()
        }
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn roundtrip_restores_every_parameter_bit_for_bit() {
        let m = CoreModel::new(config()).unwrap();
        let bytes = save_checkpoint(&m);
        let back = load_checkpoint(&bytes, &config()).unwrap();
        assert_eq!(m, back);
        assert_eq!(save_checkpoint(&back), bytes);
    }

    #[test]
    fn header_is_the_ptr_types_header_with_architecture_binding() {
        let m = CoreModel::new(config()).unwrap();
        let bytes = save_checkpoint(&m);
        let (h, _) = CheckpointHeader::read(&bytes).unwrap();
        assert_eq!(h.model, MODEL_NAME);
        assert_eq!(h.format(), ptr_types::FORMAT_V3);
        assert_eq!(h.architecture, config().architecture_bytes());
        assert_eq!(h.table(CodeFamily::ReasoningOperator), Some(11));
        assert!(h
            .verify(&CODEBOOK, SlotEncoding::V1, &REQUIRED_FAMILIES)
            .is_ok());
    }

    #[test]
    fn rejections_are_typed() {
        let m = CoreModel::new(config()).unwrap();
        let bytes = save_checkpoint(&m);

        // Not a checkpoint / truncated header.
        assert_eq!(
            load_checkpoint(b"garbage-garbage", &config()).err(),
            Some(CheckpointIoError::Header(CheckpointError::NotACheckpoint))
        );
        assert!(matches!(
            load_checkpoint(&bytes[..20], &config()),
            Err(CheckpointIoError::Header(CheckpointError::Truncated { .. }))
        ));

        // Wrong model name.
        let (h, payload) = CheckpointHeader::read(&bytes).unwrap();
        let mut other = h.clone();
        other.model = "other-model".into();
        assert_eq!(
            load_checkpoint(&other.write(payload), &config()).err(),
            Some(CheckpointIoError::WrongModel {
                found: "other-model".into()
            })
        );

        // Another architecture (config differs).
        let mut wide = config();
        wide.d_ff = 32;
        assert_eq!(
            load_checkpoint(&bytes, &wide).err(),
            Some(CheckpointIoError::ArchitectureMismatch)
        );
        let mut bad_cfg = config();
        bad_cfg.heads = 3;
        assert!(matches!(
            load_checkpoint(&bytes, &bad_cfg),
            Err(CheckpointIoError::Config(_))
        ));

        // Header architecture forged to match the new config, payload shapes differ.
        let mut forged = h.clone();
        forged.architecture = wide.architecture_bytes();
        assert!(matches!(
            load_checkpoint(&forged.write(payload), &wide),
            Err(CheckpointIoError::ShapeMismatch { .. })
        ));

        // Codebook disagreement is caught by the ptr-types verification.
        let mut moved = h.clone();
        moved.codebook_bytes.push(0);
        assert!(matches!(
            load_checkpoint(&moved.write(payload), &config()),
            Err(CheckpointIoError::Header(
                CheckpointError::CodebookMoved { .. }
            ))
        ));

        // Manifest count and name.
        let mut p = payload.to_vec();
        p[0] ^= 1;
        assert!(matches!(
            load_checkpoint(&h.write(&p), &config()),
            Err(CheckpointIoError::ManifestCount { .. })
        ));
        let mut p = payload.to_vec();
        p[4 + 2] = b'X'; // first byte of the first parameter name
        assert!(matches!(
            load_checkpoint(&h.write(&p), &config()),
            Err(CheckpointIoError::ManifestName { index: 0, .. })
        ));
    }

    #[test]
    fn corrupted_tensor_data_trips_the_tensor_checksum() {
        let m = CoreModel::new(config()).unwrap();
        let bytes = save_checkpoint(&m);
        let (h, payload) = CheckpointHeader::read(&bytes).unwrap();
        let mut p = payload.to_vec();
        let data_start = p.len() - 4 - m.parameter_count() * 4;
        p[data_start] ^= 0x01;
        assert_eq!(
            load_checkpoint(&h.write(&p), &config()).err(),
            Some(CheckpointIoError::ChecksumMismatch { name: "embed.role" })
        );
    }

    #[test]
    fn payload_checksum_truncation_and_trailing_bytes_are_detected() {
        let m = CoreModel::new(config()).unwrap();
        let bytes = save_checkpoint(&m);
        let (h, payload) = CheckpointHeader::read(&bytes).unwrap();

        let mut p = payload.to_vec();
        let last = p.len() - 1;
        p[last] ^= 0xFF;
        assert_eq!(
            load_checkpoint(&h.write(&p), &config()).err(),
            Some(CheckpointIoError::PayloadChecksum)
        );

        // Payload cut short: the declared length is rewritten by write(), so the
        // reader sees a truncated tensor section.
        let short = &payload[..payload.len() / 2];
        assert!(matches!(
            load_checkpoint(&h.write(short), &config()),
            Err(CheckpointIoError::Truncated { .. })
        ));

        // Extra bytes inside the payload body before the final checksum.
        let mut p = payload[..payload.len() - 4].to_vec();
        p.extend_from_slice(&[1, 2, 3]);
        let total = crc32(&p);
        p.extend_from_slice(&total.to_le_bytes());
        assert!(matches!(
            load_checkpoint(&h.write(&p), &config()),
            Err(CheckpointIoError::TrailingBytes { extra: 3 })
        ));

        // Bytes appended after the payload fail in the ptr-types layer.
        let mut tail = bytes.clone();
        tail.push(0);
        assert!(matches!(
            load_checkpoint(&tail, &config()),
            Err(CheckpointIoError::Header(
                CheckpointError::TrailingBytes { .. }
            ))
        ));
    }

    #[test]
    fn non_finite_parameters_are_refused_on_load() {
        let mut m = CoreModel::new(config()).unwrap();
        m.role_emb.as_mut_slice()[0] = f32::NAN;
        assert_eq!(
            load_checkpoint(&save_checkpoint(&m), &config()).err(),
            Some(CheckpointIoError::NonFinite { name: "embed.role" })
        );
    }
}
