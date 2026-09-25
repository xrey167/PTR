use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ptr_types::ArtifactId;

use crate::error::LineageError;
use crate::forgetting::GateReport;

/// Identity of one adapter checkpoint.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdapterId(pub String);

impl From<&str> for AdapterId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for AdapterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// The exact base model an adapter modifies. An adapter is meaningless on any
/// other base, so the revision is part of the identity, not metadata.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BaseModel {
    pub id: String,
    pub revision: String,
}

impl fmt::Display for BaseModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}@{}", self.id, self.revision)
    }
}

/// Where an adapter came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Origin {
    /// Trained from the base, or continued from a parent adapter.
    Trained { parent: Option<AdapterId> },
    /// Produced by merging earlier adapters into one.
    Consolidated { from: BTreeSet<AdapterId> },
}

/// Lifecycle of an adapter. Only `Serving` adapters may be routed to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterStatus {
    /// Registered, not yet gated.
    Candidate,
    /// Passed its gate; not yet serving.
    Gated,
    Serving,
    /// Withdrawn; kept for lineage and audit, never served again.
    Retired,
}

impl AdapterStatus {
    fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Gated => "gated",
            Self::Serving => "serving",
            Self::Retired => "retired",
        }
    }
}

/// Everything needed to reproduce and audit one adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterRecord {
    pub id: AdapterId,
    /// The domain or skill the adapter specialises (for example an entity
    /// family). Agents differ by context and memory, not by weights, so an
    /// adapter belongs to a domain rather than to an agent.
    pub domain: String,
    pub base: BaseModel,
    pub origin: Origin,
    pub rank: u32,
    pub target_modules: BTreeSet<String>,
    /// The sealed weights, content-addressed through `ptr-storage`.
    pub artifact: ArtifactId,
    pub artifact_sha256: [u8; 32],
    /// Fingerprint of the exact training input, replay sample ids included.
    pub data_fingerprint: [u8; 32],
    /// Ids of every raw input the adapter was trained on. This is what makes
    /// erasure propagate: revoking an input names every adapter that has to be
    /// retired or retrained.
    pub data_manifest: BTreeSet<String>,
    pub status: AdapterStatus,
}

/// All adapters of one lineage (one base model).
#[derive(Clone, Debug)]
pub struct Lineage {
    base: BaseModel,
    adapters: BTreeMap<AdapterId, AdapterRecord>,
}

impl Lineage {
    pub fn new(base: BaseModel) -> Self {
        Self {
            base,
            adapters: BTreeMap::new(),
        }
    }

    pub fn base(&self) -> &BaseModel {
        &self.base
    }

    pub fn get(&self, id: &AdapterId) -> Option<&AdapterRecord> {
        self.adapters.get(id)
    }

    /// Register a new adapter as a candidate. Its base must be the lineage base
    /// and every adapter it names must already be registered.
    pub fn register(&mut self, mut record: AdapterRecord) -> Result<(), LineageError> {
        if record.base != self.base {
            return Err(LineageError::BaseMismatch {
                expected: self.base.to_string(),
                actual: record.base.to_string(),
            });
        }
        if self.adapters.contains_key(&record.id) {
            return Err(LineageError::DuplicateAdapter {
                id: record.id.to_string(),
            });
        }
        let named: Vec<&AdapterId> = match &record.origin {
            Origin::Trained { parent } => parent.iter().collect(),
            Origin::Consolidated { from } => from.iter().collect(),
        };
        if let Some(missing) = named.iter().find(|id| !self.adapters.contains_key(id)) {
            return Err(LineageError::UnknownAdapter {
                id: missing.to_string(),
            });
        }
        if record.rank == 0 {
            return Err(LineageError::InvalidParameter {
                field: "rank",
                message: "an adapter must have rank at least 1",
            });
        }
        record.status = AdapterStatus::Candidate;
        self.adapters.insert(record.id.clone(), record);
        Ok(())
    }

    /// Promote a candidate on a passing gate report. A failing report changes
    /// nothing.
    pub fn gate(&mut self, id: &AdapterId, report: &GateReport) -> Result<(), LineageError> {
        let status = self
            .adapters
            .get(id)
            .ok_or_else(|| LineageError::UnknownAdapter { id: id.to_string() })?
            .status;
        if status != AdapterStatus::Candidate {
            return Err(LineageError::InvalidTransition {
                id: id.to_string(),
                from: status.name(),
                to: AdapterStatus::Gated.name(),
            });
        }
        if !report.passed() {
            return Err(LineageError::GateFailed { id: id.to_string() });
        }
        self.transition(id, AdapterStatus::Candidate, AdapterStatus::Gated)
    }

    pub fn serve(&mut self, id: &AdapterId) -> Result<(), LineageError> {
        self.transition(id, AdapterStatus::Gated, AdapterStatus::Serving)
    }

    /// Retire an adapter from any status. Retirement is final.
    pub fn retire(&mut self, id: &AdapterId) -> Result<(), LineageError> {
        let record = self
            .adapters
            .get_mut(id)
            .ok_or_else(|| LineageError::UnknownAdapter { id: id.to_string() })?;
        if record.status == AdapterStatus::Retired {
            return Err(LineageError::InvalidTransition {
                id: id.to_string(),
                from: "retired",
                to: "retired",
            });
        }
        record.status = AdapterStatus::Retired;
        Ok(())
    }

    /// Number of trained ancestors above `id`, following parents. A
    /// consolidated adapter restarts the count at zero: that is what
    /// consolidation buys.
    pub fn depth(&self, id: &AdapterId) -> Result<usize, LineageError> {
        let mut depth = 0;
        let mut current = self
            .adapters
            .get(id)
            .ok_or_else(|| LineageError::UnknownAdapter { id: id.to_string() })?;
        while let Origin::Trained {
            parent: Some(parent),
        } = &current.origin
        {
            depth += 1;
            current = self
                .adapters
                .get(parent)
                .ok_or_else(|| LineageError::UnknownAdapter {
                    id: parent.to_string(),
                })?;
        }
        Ok(depth)
    }

    /// Every adapter whose weights depend on any of `revoked_inputs`: trained
    /// on one directly, continued from such an adapter, or consolidated from
    /// one. Weights cannot forget a sample; the only erasure is to retire or
    /// retrain everything this returns.
    pub fn affected_by(&self, revoked_inputs: &BTreeSet<String>) -> BTreeSet<AdapterId> {
        let mut affected: BTreeSet<AdapterId> = self
            .adapters
            .values()
            .filter(|record| !record.data_manifest.is_disjoint(revoked_inputs))
            .map(|record| record.id.clone())
            .collect();
        loop {
            let before = affected.len();
            for record in self.adapters.values() {
                let inherits = match &record.origin {
                    Origin::Trained {
                        parent: Some(parent),
                    } => affected.contains(parent),
                    Origin::Trained { parent: None } => false,
                    Origin::Consolidated { from } => !from.is_disjoint(&affected),
                };
                if inherits {
                    affected.insert(record.id.clone());
                }
            }
            if affected.len() == before {
                return affected;
            }
        }
    }

    /// Adapters currently allowed to serve.
    pub fn serving(&self) -> impl Iterator<Item = &AdapterRecord> {
        self.adapters
            .values()
            .filter(|record| record.status == AdapterStatus::Serving)
    }

    fn transition(
        &mut self,
        id: &AdapterId,
        from: AdapterStatus,
        to: AdapterStatus,
    ) -> Result<(), LineageError> {
        let record = self
            .adapters
            .get_mut(id)
            .ok_or_else(|| LineageError::UnknownAdapter { id: id.to_string() })?;
        if record.status != from {
            return Err(LineageError::InvalidTransition {
                id: id.to_string(),
                from: record.status.name(),
                to: to.name(),
            });
        }
        record.status = to;
        Ok(())
    }
}

/// When a lineage should be consolidated rather than extended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsolidationPolicy {
    /// Maximum trained depth before the next adapter must be a consolidation.
    pub max_depth: usize,
    /// Maximum per-layer subspace overlap a new adapter may have with the
    /// lineage before consolidation is due instead.
    pub max_overlap: f64,
}

impl ConsolidationPolicy {
    /// Whether extending an adapter at `depth` with a candidate whose worst
    /// overlap is `overlap` should instead trigger consolidation.
    pub fn consolidation_due(&self, depth: usize, overlap: f64) -> bool {
        depth >= self.max_depth || overlap > self.max_overlap
    }
}
