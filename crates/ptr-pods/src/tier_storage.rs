use crate::{
    InMemoryKvTensorBackend, KvCacheLayout, KvCacheTier, KvLayerSnapshot, KvPageBinding,
    KvPageSize, KvPageSnapshot, KvPageTable, KvPagedSnapshot, KvTensorDType, KvTensorSchema,
    KvTensorSnapshot, TensorRef,
};
use ptr_storage::{
    ChunkDescriptor, PreparedTierObject, TierError, TierObjectDomain, TierObjectManifest,
};
use ptr_types::{AdapterVersion, Digest, Generation, ModelVersion, PrincipalId, Revision};
use sha2::{Digest as ShaDigest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodTierError {
    InvalidBinding,
    InvalidSnapshot,
    Truncated,
    Overflow,
    Storage(TierError),
}

impl From<TierError> for PodTierError {
    fn from(value: TierError) -> Self {
        Self::Storage(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvTierBinding {
    pub logical_id: String,
    pub generation: Generation,
    pub revision: Revision,
    pub context_digest: Digest,
    pub placement_epoch: u64,
    pub fencing_token: u128,
}

pub struct KvTierAdapter;

impl KvTierAdapter {
    pub fn prepare(
        snapshot: &KvTensorSnapshot,
        binding: &KvTierBinding,
        chunk_size: usize,
    ) -> Result<PreparedTierObject, PodTierError> {
        validate_kv_binding(binding)?;
        if !snapshot.verify_digest() {
            return Err(PodTierError::InvalidSnapshot);
        }
        let bytes = encode_snapshot(snapshot)?;
        PreparedTierObject::from_bytes(
            TierObjectDomain::KvSnapshot,
            binding.logical_id.clone(),
            binding.generation,
            binding.revision,
            kv_schema_digest(snapshot, binding),
            &bytes,
            chunk_size,
        )
        .map_err(Into::into)
    }

    pub fn restore(
        object: &PreparedTierObject,
        binding: &KvTierBinding,
    ) -> Result<KvTensorSnapshot, PodTierError> {
        validate_kv_binding(binding)?;
        if object.manifest.domain != TierObjectDomain::KvSnapshot
            || object.manifest.logical_id != binding.logical_id
            || object.manifest.generation != binding.generation
            || object.manifest.revision != binding.revision
        {
            return Err(PodTierError::InvalidBinding);
        }
        let (snapshot, legacy_digest) = decode_snapshot(&object.reassemble()?)?;
        let schema_digest = legacy_digest.map_or_else(
            || kv_schema_digest(&snapshot, binding),
            |digest| legacy_kv_schema_digest(&snapshot, digest, binding),
        );
        if object.manifest.schema_digest != schema_digest {
            return Err(PodTierError::InvalidBinding);
        }
        Ok(snapshot)
    }
}

pub struct PagedKvTierAdapter;

impl PagedKvTierAdapter {
    pub fn prepare(
        snapshot: &KvPagedSnapshot,
        binding: &KvTierBinding,
    ) -> Result<PreparedTierObject, PodTierError> {
        validate_kv_binding(binding)?;
        if !snapshot.verify_digest() || snapshot.binding.generation != binding.generation {
            return Err(PodTierError::InvalidSnapshot);
        }
        let mut chunks = Vec::with_capacity(snapshot.pages.len() + 1);
        chunks.push(encode_paged_header(snapshot));
        for page in &snapshot.pages {
            chunks.push(encode_page(page));
        }
        let mut offset = 0u64;
        let mut descriptors = Vec::with_capacity(chunks.len());
        for (index, bytes) in chunks.iter().enumerate() {
            let descriptor = ChunkDescriptor::for_bytes(
                u32::try_from(index).map_err(|_| PodTierError::Overflow)?,
                offset,
                bytes,
            );
            offset = offset
                .checked_add(descriptor.length)
                .ok_or(PodTierError::Overflow)?;
            descriptors.push(descriptor);
        }
        let manifest = TierObjectManifest::new(
            TierObjectDomain::KvSnapshot,
            binding.logical_id.clone(),
            binding.generation,
            binding.revision,
            paged_kv_schema_digest(snapshot, binding),
            descriptors,
        )?;
        Ok(PreparedTierObject { manifest, chunks })
    }

    pub fn restore(
        object: &PreparedTierObject,
        binding: &KvTierBinding,
    ) -> Result<KvPagedSnapshot, PodTierError> {
        validate_kv_binding(binding)?;
        object.verify()?;
        if object.manifest.domain != TierObjectDomain::KvSnapshot
            || object.manifest.logical_id != binding.logical_id
            || object.manifest.generation != binding.generation
            || object.manifest.revision != binding.revision
            || object.chunks.is_empty()
        {
            return Err(PodTierError::InvalidBinding);
        }
        let (
            schema,
            page_binding,
            capacity_tokens,
            sequence_length,
            position_offset,
            page_count,
            digest,
            legacy,
        ) = decode_paged_header(&object.chunks[0])?;
        if page_count != object.chunks.len() - 1 {
            return Err(PodTierError::InvalidSnapshot);
        }
        let pages = object.chunks[1..]
            .iter()
            .map(|chunk| decode_page(chunk))
            .collect::<Result<Vec<_>, _>>()?;
        let mut snapshot = KvPagedSnapshot {
            schema,
            binding: page_binding,
            capacity_tokens,
            sequence_length,
            position_offset,
            pages,
            digest,
        };
        if legacy {
            if legacy_paged_snapshot_digest(&snapshot) != snapshot.digest
                || object.manifest.schema_digest
                    != legacy_paged_kv_schema_digest(&snapshot, binding)
            {
                return Err(PodTierError::InvalidSnapshot);
            }
            snapshot.digest = crate::paged_kv::paged_snapshot_digest(&snapshot);
            snapshot
                .validate()
                .map_err(|_| PodTierError::InvalidSnapshot)?;
        } else if !snapshot.verify_digest()
            || object.manifest.schema_digest != paged_kv_schema_digest(&snapshot, binding)
        {
            return Err(PodTierError::InvalidSnapshot);
        }
        Ok(snapshot)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWeightTierBinding {
    pub logical_id: String,
    pub generation: Generation,
    pub revision: Revision,
    pub execution_manifest: Digest,
    pub artifact_digest: Digest,
    pub model: ModelVersion,
    pub adapter: AdapterVersion,
    pub tensor_name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
}

pub struct ModelWeightTierAdapter;

impl ModelWeightTierAdapter {
    pub fn prepare(
        bytes: &[u8],
        binding: &ModelWeightTierBinding,
        chunk_size: usize,
    ) -> Result<PreparedTierObject, PodTierError> {
        if binding.logical_id.trim().is_empty()
            || binding.tensor_name.trim().is_empty()
            || binding.dtype.trim().is_empty()
            || binding.generation.0 == 0
            || binding.execution_manifest == [0; 32]
            || binding.artifact_digest == [0; 32]
            || binding.shape.is_empty()
            || binding.shape.contains(&0)
        {
            return Err(PodTierError::InvalidBinding);
        }
        PreparedTierObject::from_bytes(
            TierObjectDomain::ModelWeight,
            binding.logical_id.clone(),
            binding.generation,
            binding.revision,
            weight_schema_digest(binding),
            bytes,
            chunk_size,
        )
        .map_err(Into::into)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactTierBinding {
    pub logical_id: String,
    pub generation: Generation,
    pub revision: Revision,
    pub artifact_digest: Digest,
    pub execution_manifest: Digest,
    pub provenance_digest: Digest,
}

pub struct ArtifactTierAdapter;

impl ArtifactTierAdapter {
    pub fn prepare(
        bytes: &[u8],
        binding: &ArtifactTierBinding,
        chunk_size: usize,
    ) -> Result<PreparedTierObject, PodTierError> {
        let actual_digest: Digest = Sha256::digest(bytes).into();
        if binding.logical_id.trim().is_empty()
            || binding.generation.0 == 0
            || binding.artifact_digest == [0; 32]
            || binding.execution_manifest == [0; 32]
            || binding.provenance_digest == [0; 32]
            || actual_digest != binding.artifact_digest
        {
            return Err(PodTierError::InvalidBinding);
        }
        PreparedTierObject::from_bytes(
            TierObjectDomain::Artifact,
            binding.logical_id.clone(),
            binding.generation,
            binding.revision,
            artifact_schema_digest(binding),
            bytes,
            chunk_size,
        )
        .map_err(Into::into)
    }
}

fn validate_kv_binding(binding: &KvTierBinding) -> Result<(), PodTierError> {
    if binding.logical_id.trim().is_empty()
        || binding.logical_id != binding.logical_id.trim()
        || binding.generation.0 == 0
        || binding.context_digest == [0; 32]
        || binding.placement_epoch == 0
        || binding.fencing_token == 0
    {
        return Err(PodTierError::InvalidBinding);
    }
    Ok(())
}

fn kv_schema_digest(snapshot: &KvTensorSnapshot, binding: &KvTierBinding) -> Digest {
    kv_schema_digest_versioned(snapshot, snapshot.digest, binding, true)
}

fn legacy_kv_schema_digest(
    snapshot: &KvTensorSnapshot,
    legacy_snapshot_digest: Digest,
    binding: &KvTierBinding,
) -> Digest {
    kv_schema_digest_versioned(snapshot, legacy_snapshot_digest, binding, false)
}

fn kv_schema_digest_versioned(
    snapshot: &KvTensorSnapshot,
    snapshot_digest: Digest,
    binding: &KvTierBinding,
    bind_key_value_heads: bool,
) -> Digest {
    let mut bytes = b"ptr-kv-tier-binding-v1\0".to_vec();
    put_text(&mut bytes, &snapshot.schema.model.0);
    put_text(&mut bytes, &snapshot.schema.adapter.0);
    put_u64(&mut bytes, snapshot.schema.layer_count as u64);
    put_u64(&mut bytes, snapshot.schema.attention_heads as u64);
    if bind_key_value_heads {
        put_u64(&mut bytes, snapshot.schema.key_value_heads as u64);
    }
    put_u64(&mut bytes, snapshot.schema.head_dim as u64);
    bytes.push(dtype_code(snapshot.schema.dtype));
    bytes.extend_from_slice(&snapshot_digest);
    bytes.extend_from_slice(&binding.context_digest);
    put_u64(&mut bytes, binding.generation.0);
    put_u64(&mut bytes, binding.revision.0);
    put_u64(&mut bytes, binding.placement_epoch);
    bytes.extend_from_slice(&binding.fencing_token.to_le_bytes());
    Sha256::digest(bytes).into()
}

fn paged_kv_schema_digest(snapshot: &KvPagedSnapshot, binding: &KvTierBinding) -> Digest {
    let mut bytes = b"ptr-paged-kv-tier-binding-v1\0".to_vec();
    bytes.extend_from_slice(&snapshot.digest);
    bytes.extend_from_slice(&snapshot.binding.execution_manifest);
    bytes.extend_from_slice(&binding.context_digest);
    put_u64(&mut bytes, binding.generation.0);
    put_u64(&mut bytes, binding.revision.0);
    put_u64(&mut bytes, binding.placement_epoch);
    bytes.extend_from_slice(&binding.fencing_token.to_le_bytes());
    Sha256::digest(bytes).into()
}

fn legacy_paged_kv_schema_digest(snapshot: &KvPagedSnapshot, binding: &KvTierBinding) -> Digest {
    let mut bytes = b"ptr-paged-kv-tier-binding-v1\0".to_vec();
    bytes.extend_from_slice(&snapshot.digest);
    bytes.extend_from_slice(&snapshot.binding.execution_manifest);
    bytes.extend_from_slice(&binding.context_digest);
    put_u64(&mut bytes, binding.generation.0);
    put_u64(&mut bytes, binding.revision.0);
    put_u64(&mut bytes, binding.placement_epoch);
    bytes.extend_from_slice(&binding.fencing_token.to_le_bytes());
    Sha256::digest(bytes).into()
}

fn legacy_paged_snapshot_digest(snapshot: &KvPagedSnapshot) -> Digest {
    let mut schema = b"ptr-paged-kv-schema-v1\0".to_vec();
    put_text(&mut schema, &snapshot.schema.model.0);
    put_text(&mut schema, &snapshot.schema.adapter.0);
    put_text(&mut schema, &snapshot.schema.device.0);
    for value in [
        snapshot.schema.layer_count,
        snapshot.schema.attention_heads,
        snapshot.schema.head_dim,
        snapshot.schema.batch_size,
        snapshot.binding.page_tokens.get(),
    ] {
        put_u64(&mut schema, value as u64);
    }
    schema.push(dtype_code(snapshot.schema.dtype));
    let legacy_schema: Digest = Sha256::digest(schema).into();

    let mut bytes = b"ptr-paged-kv-snapshot-v1\0".to_vec();
    bytes.extend_from_slice(&legacy_schema);
    put_page_binding(&mut bytes, &snapshot.binding);
    put_u64(&mut bytes, snapshot.capacity_tokens as u64);
    put_u64(&mut bytes, snapshot.sequence_length as u64);
    put_u64(&mut bytes, snapshot.position_offset as u64);
    for page in &snapshot.pages {
        bytes.extend_from_slice(&page.digest);
    }
    Sha256::digest(bytes).into()
}

fn put_page_binding(bytes: &mut Vec<u8>, binding: &KvPageBinding) {
    put_text(bytes, &binding.model.0);
    put_text(bytes, &binding.adapter.0);
    bytes.extend_from_slice(&binding.generation.0.to_le_bytes());
    bytes.extend_from_slice(&binding.execution_manifest);
    put_text(bytes, &binding.principal.0);
    put_text(bytes, &binding.device.0);
    bytes.push(dtype_code(binding.dtype));
    put_u64(bytes, binding.page_tokens.get() as u64);
}

fn encode_paged_header(snapshot: &KvPagedSnapshot) -> Vec<u8> {
    let mut out = b"PTRKVP3\0".to_vec();
    put_text(&mut out, &snapshot.schema.model.0);
    put_text(&mut out, &snapshot.schema.adapter.0);
    put_text(&mut out, &snapshot.schema.device.0);
    for value in [
        snapshot.schema.layer_count,
        snapshot.schema.attention_heads,
        snapshot.schema.key_value_heads,
        snapshot.schema.head_dim,
        snapshot.schema.batch_size,
    ] {
        put_u64(&mut out, value as u64);
    }
    out.push(dtype_code(snapshot.schema.dtype));
    put_text(&mut out, &snapshot.binding.model.0);
    put_text(&mut out, &snapshot.binding.adapter.0);
    put_u64(&mut out, snapshot.binding.generation.0);
    out.extend_from_slice(&snapshot.binding.execution_manifest);
    put_text(&mut out, &snapshot.binding.principal.0);
    put_text(&mut out, &snapshot.binding.device.0);
    out.push(dtype_code(snapshot.binding.dtype));
    put_u64(&mut out, snapshot.binding.page_tokens.get() as u64);
    put_u64(&mut out, snapshot.capacity_tokens as u64);
    put_u64(&mut out, snapshot.sequence_length as u64);
    put_u64(&mut out, snapshot.position_offset as u64);
    put_u64(&mut out, snapshot.pages.len() as u64);
    out.extend_from_slice(&snapshot.digest);
    out
}

type PagedHeader = (
    KvTensorSchema,
    KvPageBinding,
    usize,
    usize,
    usize,
    usize,
    Digest,
    bool,
);

fn decode_paged_header(bytes: &[u8]) -> Result<PagedHeader, PodTierError> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.take(8)?;
    let has_key_value_heads = match magic {
        b"PTRKVP2\0" => false,
        b"PTRKVP3\0" => true,
        _ => return Err(PodTierError::InvalidSnapshot),
    };
    let model = ModelVersion(cursor.text()?);
    let adapter = AdapterVersion(cursor.text()?);
    let device = ptr_types::DeviceId(cursor.text()?);
    let layer_count = cursor.usize()?;
    let attention_heads = cursor.usize()?;
    let key_value_heads = if has_key_value_heads {
        cursor.usize()?
    } else {
        attention_heads
    };
    let schema = KvTensorSchema {
        model,
        adapter,
        device,
        layer_count,
        attention_heads,
        key_value_heads,
        head_dim: cursor.usize()?,
        batch_size: cursor.usize()?,
        dtype: dtype_from_code(cursor.u8()?)?,
    };
    let binding = KvPageBinding {
        model: ModelVersion(cursor.text()?),
        adapter: AdapterVersion(cursor.text()?),
        generation: Generation(cursor.u64()?),
        execution_manifest: cursor.digest()?,
        principal: PrincipalId(cursor.text()?),
        device: ptr_types::DeviceId(cursor.text()?),
        dtype: dtype_from_code(cursor.u8()?)?,
        page_tokens: KvPageSize::new(
            u16::try_from(cursor.usize()?).map_err(|_| PodTierError::Overflow)?,
        )
        .map_err(|_| PodTierError::InvalidSnapshot)?,
    };
    let capacity_tokens = cursor.usize()?;
    let sequence_length = cursor.usize()?;
    let position_offset = cursor.usize()?;
    let page_count = cursor.usize()?;
    let digest = cursor.digest()?;
    if !cursor.finished()
        || page_count > capacity_tokens.div_ceil(binding.page_tokens.get())
        || sequence_length > capacity_tokens
    {
        return Err(PodTierError::InvalidSnapshot);
    }
    Ok((
        schema,
        binding,
        capacity_tokens,
        sequence_length,
        position_offset,
        page_count,
        digest,
        !has_key_value_heads,
    ))
}

fn encode_page(page: &KvPageSnapshot) -> Vec<u8> {
    let mut out = b"PTRPAGE2".to_vec();
    put_u64(&mut out, page.ordinal as u64);
    put_u64(&mut out, u64::from(page.valid_tokens));
    put_u64(&mut out, page.layers.len() as u64);
    for layer in &page.layers {
        put_tensor(&mut out, &layer.keys);
        put_tensor(&mut out, &layer.values);
    }
    out.extend_from_slice(&page.digest);
    out
}

fn decode_page(bytes: &[u8]) -> Result<KvPageSnapshot, PodTierError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(8)? != b"PTRPAGE2" {
        return Err(PodTierError::InvalidSnapshot);
    }
    let ordinal = cursor.usize()?;
    let valid_tokens = u16::try_from(cursor.usize()?).map_err(|_| PodTierError::Overflow)?;
    let layer_count = cursor.usize()?;
    if valid_tokens == 0 || layer_count > 4_096 {
        return Err(PodTierError::InvalidSnapshot);
    }
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        layers.push(KvLayerSnapshot {
            keys: cursor.tensor()?,
            values: cursor.tensor()?,
        });
    }
    let digest = cursor.digest()?;
    if !cursor.finished() {
        return Err(PodTierError::InvalidSnapshot);
    }
    Ok(KvPageSnapshot {
        ordinal,
        valid_tokens,
        layers,
        digest,
    })
}

fn weight_schema_digest(binding: &ModelWeightTierBinding) -> Digest {
    let mut bytes = b"ptr-weight-tier-binding-v1\0".to_vec();
    bytes.extend_from_slice(&binding.execution_manifest);
    bytes.extend_from_slice(&binding.artifact_digest);
    put_text(&mut bytes, &binding.model.0);
    put_text(&mut bytes, &binding.adapter.0);
    put_text(&mut bytes, &binding.tensor_name);
    put_text(&mut bytes, &binding.dtype);
    put_u64(&mut bytes, binding.shape.len() as u64);
    for dimension in &binding.shape {
        put_u64(&mut bytes, *dimension);
    }
    Sha256::digest(bytes).into()
}

fn artifact_schema_digest(binding: &ArtifactTierBinding) -> Digest {
    let mut bytes = b"ptr-artifact-tier-binding-v1\0".to_vec();
    bytes.extend_from_slice(&binding.artifact_digest);
    bytes.extend_from_slice(&binding.execution_manifest);
    bytes.extend_from_slice(&binding.provenance_digest);
    Sha256::digest(bytes).into()
}

fn encode_snapshot(snapshot: &KvTensorSnapshot) -> Result<Vec<u8>, PodTierError> {
    if !snapshot.verify_digest() {
        return Err(PodTierError::InvalidSnapshot);
    }
    let mut out = b"PTRKVTR2".to_vec();
    put_text(&mut out, &snapshot.schema.model.0);
    put_text(&mut out, &snapshot.schema.adapter.0);
    put_text(&mut out, &snapshot.schema.device.0);
    for value in [
        snapshot.schema.layer_count,
        snapshot.schema.attention_heads,
        snapshot.schema.key_value_heads,
        snapshot.schema.head_dim,
        snapshot.schema.batch_size,
    ] {
        put_u64(&mut out, value as u64);
    }
    out.push(dtype_code(snapshot.schema.dtype));
    put_u64(&mut out, snapshot.layout.page_tokens as u64);
    out.push(tier_code(snapshot.layout.tier));
    out.push(dtype_code(snapshot.layout.dtype));
    put_u64(&mut out, snapshot.page_table.page_tokens as u64);
    put_u64(&mut out, snapshot.page_table.capacity_tokens as u64);
    put_usizes(&mut out, &snapshot.page_table.logical_to_physical);
    put_usizes(&mut out, &snapshot.page_table.free_physical);
    for value in [
        snapshot.capacity_tokens,
        snapshot.sequence_length,
        snapshot.position_offset,
    ] {
        put_u64(&mut out, value as u64);
    }
    put_u64(&mut out, snapshot.layers.len() as u64);
    for layer in &snapshot.layers {
        put_tensor(&mut out, &layer.keys);
        put_tensor(&mut out, &layer.values);
    }
    out.extend_from_slice(&snapshot.digest);
    Ok(out)
}

fn decode_snapshot(bytes: &[u8]) -> Result<(KvTensorSnapshot, Option<Digest>), PodTierError> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.take(8)?;
    let has_key_value_heads = match magic {
        b"PTRKVTR1" => false,
        b"PTRKVTR2" => true,
        _ => return Err(PodTierError::InvalidSnapshot),
    };
    let model = ModelVersion(cursor.text()?);
    let adapter = AdapterVersion(cursor.text()?);
    let device = ptr_types::DeviceId(cursor.text()?);
    let layer_count = cursor.usize()?;
    let attention_heads = cursor.usize()?;
    let key_value_heads = if has_key_value_heads {
        cursor.usize()?
    } else {
        attention_heads
    };
    let schema = KvTensorSchema {
        model,
        adapter,
        device,
        layer_count,
        attention_heads,
        key_value_heads,
        head_dim: cursor.usize()?,
        batch_size: cursor.usize()?,
        dtype: dtype_from_code(cursor.u8()?)?,
    };
    let layout = KvCacheLayout {
        page_tokens: cursor.usize()?,
        tier: tier_from_code(cursor.u8()?)?,
        dtype: dtype_from_code(cursor.u8()?)?,
    };
    let page_table = KvPageTable {
        page_tokens: cursor.usize()?,
        capacity_tokens: cursor.usize()?,
        logical_to_physical: cursor.usizes()?,
        free_physical: cursor.usizes()?,
    };
    let capacity_tokens = cursor.usize()?;
    let sequence_length = cursor.usize()?;
    let position_offset = cursor.usize()?;
    let layer_count = cursor.usize()?;
    if layer_count > 4_096
        || (sequence_length == 0 && layer_count != 0)
        || (sequence_length != 0 && layer_count != schema.layer_count)
    {
        return Err(PodTierError::InvalidSnapshot);
    }
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        layers.push(KvLayerSnapshot {
            keys: cursor.tensor()?,
            values: cursor.tensor()?,
        });
    }
    let digest = cursor.digest()?;
    if !cursor.finished() {
        return Err(PodTierError::InvalidSnapshot);
    }
    let mut snapshot = KvTensorSnapshot {
        schema,
        layout,
        page_table,
        capacity_tokens,
        sequence_length,
        position_offset,
        layers,
        digest,
    };
    let expected_width = snapshot
        .schema
        .key_value_heads
        .checked_mul(snapshot.schema.head_dim)
        .ok_or(PodTierError::Overflow)?;
    let tensors_valid = snapshot.layers.iter().enumerate().all(|(index, layer)| {
        [&layer.keys, &layer.values].into_iter().all(|tensor| {
            tensor.layer == index
                && tensor.shape.len() == 2
                && tensor.shape[0] == snapshot.sequence_length
                && tensor.shape[1] == expected_width
                && tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |total, value| total.checked_mul(*value))
                    == Some(tensor.values.len())
        })
    });
    if !tensors_valid || snapshot.sequence_length > snapshot.capacity_tokens {
        return Err(PodTierError::InvalidSnapshot);
    }
    if has_key_value_heads {
        if !snapshot.verify_digest() {
            return Err(PodTierError::InvalidSnapshot);
        }
        Ok((snapshot, None))
    } else {
        if InMemoryKvTensorBackend::legacy_v1_digest(&snapshot) != snapshot.digest {
            return Err(PodTierError::InvalidSnapshot);
        }
        let legacy_digest = snapshot.digest;
        snapshot.digest = InMemoryKvTensorBackend::digest(&snapshot);
        Ok((snapshot, Some(legacy_digest)))
    }
}

fn put_tensor(out: &mut Vec<u8>, tensor: &TensorRef) {
    put_u64(out, tensor.layer as u64);
    put_usizes(out, &tensor.shape);
    put_u64(out, tensor.values.len() as u64);
    for value in &tensor.values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn put_usizes(out: &mut Vec<u8>, values: &[usize]) {
    put_u64(out, values.len() as u64);
    for value in values {
        put_u64(out, *value as u64);
    }
}

fn put_text(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn dtype_code(dtype: KvTensorDType) -> u8 {
    match dtype {
        KvTensorDType::F32 => 0,
        KvTensorDType::Fp8E4M3 => 1,
        KvTensorDType::Fp8E5M2 => 2,
        KvTensorDType::NvFp4 => 3,
        KvTensorDType::Fp4MxBlock16 => 4,
    }
}

fn dtype_from_code(code: u8) -> Result<KvTensorDType, PodTierError> {
    match code {
        0 => Ok(KvTensorDType::F32),
        1 => Ok(KvTensorDType::Fp8E4M3),
        2 => Ok(KvTensorDType::Fp8E5M2),
        3 => Ok(KvTensorDType::NvFp4),
        4 => Ok(KvTensorDType::Fp4MxBlock16),
        _ => Err(PodTierError::InvalidSnapshot),
    }
}

fn tier_code(tier: KvCacheTier) -> u8 {
    match tier {
        KvCacheTier::Gpu => 0,
        KvCacheTier::Cpu => 1,
        KvCacheTier::Nvme => 2,
        KvCacheTier::ObjectStore => 3,
    }
}

fn tier_from_code(code: u8) -> Result<KvCacheTier, PodTierError> {
    match code {
        0 => Ok(KvCacheTier::Gpu),
        1 => Ok(KvCacheTier::Cpu),
        2 => Ok(KvCacheTier::Nvme),
        3 => Ok(KvCacheTier::ObjectStore),
        _ => Err(PodTierError::InvalidSnapshot),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], PodTierError> {
        let end = self.at.checked_add(len).ok_or(PodTierError::Overflow)?;
        let value = self
            .bytes
            .get(self.at..end)
            .ok_or(PodTierError::Truncated)?;
        self.at = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, PodTierError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, PodTierError> {
        self.take(4)?
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| PodTierError::Truncated)
    }

    fn u64(&mut self) -> Result<u64, PodTierError> {
        self.take(8)?
            .try_into()
            .map(u64::from_le_bytes)
            .map_err(|_| PodTierError::Truncated)
    }

    fn usize(&mut self) -> Result<usize, PodTierError> {
        usize::try_from(self.u64()?).map_err(|_| PodTierError::Overflow)
    }

    fn text(&mut self) -> Result<String, PodTierError> {
        let len = self.usize()?;
        if len > 1024 * 1024 || len > self.remaining() {
            return Err(PodTierError::InvalidSnapshot);
        }
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| PodTierError::InvalidSnapshot)
    }

    fn digest(&mut self) -> Result<Digest, PodTierError> {
        self.take(32)?
            .try_into()
            .map_err(|_| PodTierError::Truncated)
    }

    fn usizes(&mut self) -> Result<Vec<usize>, PodTierError> {
        let count = self.usize()?;
        if count > 1_048_576 || count > self.remaining() / 8 {
            return Err(PodTierError::InvalidSnapshot);
        }
        (0..count).map(|_| self.usize()).collect()
    }

    fn tensor(&mut self) -> Result<TensorRef, PodTierError> {
        let layer = self.usize()?;
        let shape = self.usizes()?;
        let count = self.usize()?;
        if count > 67_108_864 || count > self.remaining() / 4 {
            return Err(PodTierError::InvalidSnapshot);
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(f32::from_bits(self.u32()?));
        }
        Ok(TensorRef {
            layer,
            shape,
            values,
        })
    }

    fn finished(&self) -> bool {
        self.at == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }
}
