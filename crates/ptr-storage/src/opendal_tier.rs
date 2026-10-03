use crate::tier::{
    digest, ChunkDescriptor, ChunkReceipt, StorageTier, TierBackend, TierBackendId,
    TierCapabilities, TierError, TierFuture, TierHealth, TierIntegrityReport,
};
use opendal::{ErrorKind, Operator};

pub struct OpenDalTierBackend {
    id: TierBackendId,
    operator: Operator,
    prefix: String,
}

impl OpenDalTierBackend {
    pub fn new(
        id: TierBackendId,
        operator: Operator,
        prefix: impl Into<String>,
    ) -> Result<Self, TierError> {
        if id.0.trim().is_empty() || id.0 != id.0.trim() {
            return Err(TierError::InvalidBackendId);
        }
        let prefix = prefix.into().trim_matches('/').to_owned();
        if prefix.split('/').any(|part| part == ".." || part == ".") {
            return Err(TierError::Backend("invalid OpenDAL prefix".into()));
        }
        Ok(Self {
            id,
            operator,
            prefix,
        })
    }

    pub fn chunk_key(&self, chunk: &ChunkDescriptor) -> String {
        let digest = hex(&chunk.digest);
        if self.prefix.is_empty() {
            format!("chunks/{digest}")
        } else {
            format!("{}/chunks/{digest}", self.prefix)
        }
    }

    async fn read_verified(&self, chunk: &ChunkDescriptor) -> Result<Vec<u8>, TierError> {
        let bytes = self
            .operator
            .read(&self.chunk_key(chunk))
            .await
            .map_err(map_error)?
            .to_vec();
        chunk
            .validate_bytes(&bytes)
            .map_err(|_| TierError::CorruptChunk)?;
        Ok(bytes)
    }
}

impl TierBackend for OpenDalTierBackend {
    fn identity(&self) -> TierBackendId {
        self.id.clone()
    }

    fn capabilities(&self) -> TierCapabilities {
        TierCapabilities {
            tier: StorageTier::ObjectStore,
            persistent: true,
            range_reads: true,
            conditional_create: false,
        }
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        Box::pin(async move {
            chunk.validate_bytes(&bytes)?;
            let key = self.chunk_key(&chunk);
            match self.operator.read(&key).await {
                Ok(existing) => {
                    let existing = existing.to_vec();
                    if existing != bytes || chunk.validate_bytes(&existing).is_err() {
                        return Err(TierError::ImmutableConflict);
                    }
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    self.operator.write(&key, bytes).await.map_err(map_error)?;
                    self.read_verified(&chunk).await?;
                }
                Err(error) => return Err(map_error(error)),
            }
            Ok(ChunkReceipt {
                backend: self.id.clone(),
                digest: chunk.digest,
                length: chunk.length,
            })
        })
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(async move { self.read_verified(chunk).await })
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let bytes = self
                .operator
                .read(&self.chunk_key(chunk))
                .await
                .map_err(map_error)?
                .to_vec();
            Ok(TierIntegrityReport {
                digest: digest(&bytes),
                length: bytes.len() as u64,
                valid: chunk.validate_bytes(&bytes).is_ok(),
            })
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        Box::pin(async move {
            self.operator
                .delete(&self.chunk_key(chunk))
                .await
                .map_err(map_error)
        })
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        Box::pin(async move {
            self.operator
                .check()
                .await
                .map(|_| TierHealth::Healthy)
                .map_err(map_error)
        })
    }
}

fn map_error(error: opendal::Error) -> TierError {
    if error.kind() == ErrorKind::NotFound {
        TierError::UnknownChunk
    } else {
        TierError::Backend(error.to_string())
    }
}

fn hex(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}
