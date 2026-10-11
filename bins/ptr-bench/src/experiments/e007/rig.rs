//! What the cases share: the tiny stack, the two stages (each with its own cache), the
//! upstream runtime's signing key, a real `PtrRuntime` ledger as the lifecycle authority and a
//! real `PodPlacementController` as the placement authority.

use super::params;
use crate::experiments::layered_kv::lab::{self, fail, Authority, Res};
use crate::experiments::layered_kv::pins;
use ed25519_dalek::SigningKey;
use ptr_ledger::integrity::sha256;
use ptr_pods::{
    InMemoryPagedKvBackend, KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType,
    KvTensorSnapshot, PagedKvTensorBackend, Qwen2PagedStack, Qwen2RopeRegime, Qwen2StageOutput,
    StageRows, StageRun,
};
use ptr_protocol::activation::{ActivationFrame, BoundaryDtype, ChainElement, UpstreamChain};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, NodeHealth, NodeRecord, PodPlacementController,
    RuntimeStageLiveness,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, PrincipalId,
    Timestamp,
};
use std::collections::BTreeMap;
use std::ops::Range;

pub type Cache = <InMemoryPagedKvBackend as KvTensorBackend>::Cache;

/// `f32` values as little-endian bytes.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Little-endian bytes as `f32` values.
pub fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The bit patterns of a vector of floats.
pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The digest committed for `(identity, generation)`.
pub fn digest_of(identity: &str, generation: u64) -> [u8; 32] {
    sha256(format!("e007.manifest|{identity}|{generation}").as_bytes())
}

/// One pod of the pipeline: a layer span, its backend and its cache.
pub struct Stage {
    pub span: Range<usize>,
    pub backend: InMemoryPagedKvBackend,
    pub cache: Cache,
}

fn page_binding(schema: &ptr_pods::KvTensorSchema) -> Res<KvPageBinding> {
    Ok(KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [1; 32],
        principal: PrincipalId::from(lab::PRINCIPAL),
        device: schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::new(params::PAGE_TOKENS as u16).map_err(fail("the page size"))?,
    })
}

fn capacity() -> usize {
    pins::MAX_PROMPT_TOKENS + pins::MAX_NEW_TOKENS
}

impl Stage {
    /// A stage holding the layers `span`; the whole stack is `0..layers`.
    pub fn new(stack: &Qwen2PagedStack, span: Range<usize>, name: &str) -> Res<Self> {
        let backend = InMemoryPagedKvBackend::new(
            KvPageSize::new(params::PAGE_TOKENS as u16).map_err(fail("the page size"))?,
        );
        let schema = stack
            .stage_kv_schema(
                &span,
                ModelVersion::from(name),
                AdapterVersion::from("base"),
                DeviceId::from("cpu"),
            )
            .map_err(fail("the stage schema"))?;
        let binding = page_binding(&schema)?;
        let cache = backend
            .allocate_bound(schema, capacity(), binding)
            .map_err(fail("a stage cache"))?;
        Ok(Self {
            span,
            backend,
            cache,
        })
    }

    /// Run the span over `rows` residual rows.
    pub fn run(
        &mut self,
        stack: &Qwen2PagedStack,
        residual: &[f32],
        rows: usize,
        decode: bool,
        regime: &Qwen2RopeRegime,
    ) -> Res<Qwen2StageOutput> {
        stack
            .run_stage_in_regime(
                &self.backend,
                &mut self.cache,
                StageRun::new(
                    self.span.clone(),
                    residual,
                    rows,
                    if decode {
                        StageRows::DecodeStep
                    } else {
                        StageRows::Prefill
                    },
                ),
                regime,
            )
            .map_err(fail("a stage run"))
    }

    pub fn snapshot(&self) -> Res<KvTensorSnapshot> {
        self.backend
            .snapshot(&self.cache)
            .map_err(fail("a snapshot"))
    }

    /// The K and V bytes of every layer of the stage's cache: keys then values, per layer.
    pub fn kv_bytes(&self) -> Res<Vec<Vec<u8>>> {
        Ok(self
            .snapshot()?
            .layers
            .iter()
            .map(|layer| {
                let mut bytes = f32_bytes(&layer.keys.values);
                bytes.extend(f32_bytes(&layer.values.values));
                bytes
            })
            .collect())
    }

    /// A stage that continues from `snapshot` (the downstream cache's hit).
    pub fn restored(
        stack: &Qwen2PagedStack,
        span: Range<usize>,
        snapshot: KvTensorSnapshot,
    ) -> Res<Self> {
        let backend = InMemoryPagedKvBackend::new(
            KvPageSize::new(params::PAGE_TOKENS as u16).map_err(fail("the page size"))?,
        );
        let device = snapshot.schema.device.clone();
        let _ = stack;
        let cache = backend
            .restore(snapshot, &device)
            .map_err(fail("restoring a snapshot"))?;
        Ok(Self {
            span,
            backend,
            cache,
        })
    }
}

/// The world a case runs in.
pub struct Rig {
    pub stack: Qwen2PagedStack,
    pub signer: SigningKey,
    pub authority: Authority,
    pub placement: PodPlacementController,
    pub manifests: BTreeMap<(String, u64), [u8; 32]>,
    stage_generation: u64,
}

impl Rig {
    pub fn new() -> Res<Self> {
        let stack = pins::reference_stack()?;
        let authority =
            Authority::new(&[params::STAGE_IDENTITY, params::DOWNSTREAM_STAGE_IDENTITY])?;
        let mut placement = PodPlacementController::default();
        for node in ["node-1", "node-2"] {
            placement
                .register_node(NodeRecord {
                    node_id: NodeId::from(node),
                    zone: "z".into(),
                    health: NodeHealth::Healthy,
                    last_heartbeat: Timestamp(1),
                    devices: vec![DeviceRecord {
                        device_id: DeviceId::from("cpu-0"),
                        vram_bytes: 1 << 20,
                        used_vram_bytes: 0,
                        health: DeviceHealth::Healthy,
                    }],
                })
                .map_err(fail("a node"))?;
        }
        let mut rig = Self {
            stack,
            signer: SigningKey::from_bytes(&seeded_key(params::SIGNING_KEY_SEED)),
            authority,
            placement,
            manifests: BTreeMap::new(),
            stage_generation: 1,
        };
        rig.manifests.insert(
            (
                lab::BASE_MODEL_IDENTITY.to_string(),
                lab::BASE_MODEL_GENERATION,
            ),
            digest_of(lab::BASE_MODEL_IDENTITY, lab::BASE_MODEL_GENERATION),
        );
        rig.manifests.insert(
            (params::STAGE_IDENTITY.to_string(), 1),
            digest_of(params::STAGE_IDENTITY, 1),
        );
        rig.manifests.insert(
            (params::DOWNSTREAM_STAGE_IDENTITY.to_string(), 1),
            digest_of(params::DOWNSTREAM_STAGE_IDENTITY, 1),
        );
        rig.place("node-1")?;
        Ok(rig)
    }

    fn place(&mut self, node: &str) -> Res<()> {
        self.placement
            .assign(
                PodId::from(params::STAGE_IDENTITY),
                ArtifactId::from("stage-a-artifact"),
                Generation(self.stage_generation),
                NodeId::from(node),
                DeviceId::from("cpu-0"),
                1,
            )
            .map(|_| ())
            .map_err(fail("placing the stage"))
    }

    pub fn live(&self) -> RuntimeStageLiveness<'_> {
        RuntimeStageLiveness {
            runtime: &self.authority.runtime,
            placement: &self.placement,
            manifests: &self.manifests,
        }
    }

    pub fn stage_generation(&self) -> u64 {
        self.stage_generation
    }

    /// The placement epoch the upstream stage holds now.
    pub fn epoch(&self) -> u64 {
        self.placement
            .placement(&PodId::from(params::STAGE_IDENTITY))
            .map(|placement| placement.epoch.0)
            .unwrap_or(0)
    }

    fn element(identity: &str, generation: u64) -> ChainElement {
        ChainElement {
            identity: identity.to_string(),
            generation,
            digest: digest_of(identity, generation),
            fence_epoch: 0,
        }
    }

    /// The upstream chain of a frame produced now.
    pub fn chain(&self) -> UpstreamChain {
        UpstreamChain {
            stage: Self::element(params::STAGE_IDENTITY, self.stage_generation),
            base_model: Self::element(lab::BASE_MODEL_IDENTITY, lab::BASE_MODEL_GENERATION),
            adapters: Vec::new(),
            experts: Vec::new(),
        }
    }

    /// The downstream stage's own chain: it keys and guards what the downstream cache stores.
    pub fn own_chain(&self) -> UpstreamChain {
        UpstreamChain {
            stage: Self::element(params::DOWNSTREAM_STAGE_IDENTITY, 1),
            base_model: Self::element(lab::BASE_MODEL_IDENTITY, lab::BASE_MODEL_GENERATION),
            adapters: Vec::new(),
            experts: Vec::new(),
        }
    }

    /// Commit the revocation of the stage's current generation.
    pub fn revoke_stage(&mut self) -> Res<()> {
        lab::commit_revocation(
            &mut self.authority.runtime,
            params::STAGE_IDENTITY,
            self.stage_generation,
        )
    }

    /// Place the successor generation, which later frames carry (on another node, so its epoch
    /// moves).
    pub fn place_successor(&mut self) -> Res<()> {
        self.place("node-2")
    }

    /// Commit the supersession of the stage's generation by the next, without moving the
    /// placement: the worker still holds the old epoch, so only the liveness of the generation tells the old chain from the live one (a
    /// re-placement would make the placement epoch refuse the zombie as well, and the liveness
    /// check would then never be the one under test).
    pub fn supersede_stage_in_place(&mut self) -> Res<()> {
        let old = self.stage_generation;
        lab::commit_supersession(
            &mut self.authority.runtime,
            params::STAGE_IDENTITY,
            old,
            old + 1,
        )?;
        self.stage_generation = old + 1;
        self.manifests.insert(
            (params::STAGE_IDENTITY.to_string(), old + 1),
            digest_of(params::STAGE_IDENTITY, old + 1),
        );
        Ok(())
    }

    /// Re-place the stage on another node without a generation change: the old worker's
    /// epoch is no longer the stage's.
    pub fn replace_stage(&mut self) -> Res<()> {
        self.placement
            .mark_node_failed(&NodeId::from("node-1"))
            .map_err(fail("failing a node"))?;
        self.place("node-2")
    }

    /// The frame stage A produces from `residual` rows at `positions`.
    pub fn frame(&self, residual: &[f32], rows: usize, first_position: usize) -> ActivationFrame {
        ActivationFrame {
            tokens: rows as u32,
            width: params::HIDDEN_SIZE as u32,
            dtype: BoundaryDtype::F32,
            positions: (first_position..first_position + rows)
                .map(|position| position as u32)
                .collect(),
            upstream_chain: self.chain(),
            placement_epoch: self.epoch(),
            tensor: f32_bytes(residual),
        }
    }
}

/// A 32-byte signing seed from a number.
pub fn seeded_key(seed: u64) -> [u8; 32] {
    sha256(format!("e007.signing-key|{seed}").as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_runtime::StageLiveness;
    use ptr_types::Validity;

    #[test]
    fn the_rig_commits_the_stage_and_places_it_and_both_changes_move_what_the_chain_says() {
        let mut rig = Rig::new().unwrap();
        let chain = rig.chain();
        let epoch = rig.epoch();
        assert!(epoch > 0);
        let live = rig.live();
        assert_eq!(
            live.validity("stage-a", Generation(1)),
            Some(Validity::Live)
        );
        assert_eq!(live.placement_epoch("stage-a"), Some(epoch));
        assert_eq!(
            live.manifest_digest("stage-a", Generation(1)),
            Some(chain.stage.digest)
        );
        rig.replace_stage().unwrap();
        assert!(rig.epoch() > epoch);
        assert_eq!(
            rig.live().validity("stage-a", Generation(1)),
            Some(Validity::Live)
        );
        rig.supersede_stage_in_place().unwrap();
        rig.place_successor().unwrap();
        assert_eq!(rig.stage_generation(), 2);
        assert_eq!(
            rig.live().validity("stage-a", Generation(1)),
            Some(Validity::Superseded)
        );
        assert_eq!(
            rig.live().validity("stage-a", Generation(2)),
            Some(Validity::Live)
        );
        rig.revoke_stage().unwrap();
        assert_eq!(
            rig.live().validity("stage-a", Generation(2)),
            Some(Validity::Revoked)
        );
    }
}
