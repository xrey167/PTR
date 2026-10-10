//! Rebuilding the loop's roster from the committed expert registry.
//!
//! The ledger commits which experts exist, in which state, at which generation and with which
//! manifest digest. It does not commit finding codes. A specialist's manifest names the codes it
//! repairs (`repairs:<code>` capabilities), and the manifest digest is in the ledger, so the
//! host's manifest store plus the ledger reproduce the roster, and a manifest that was swapped
//! after the commit is detected by its digest.

use std::fmt;

use ptr_pods::PodRegistry;
use ptr_router::FindingCode;
use ptr_types::{PodId, ProjectId};

use super::specialist::repaired_codes;
use crate::expert::ExpertRegistryProjection;
use crate::vcw::{state_is_routable, Roster};

/// Why the roster could not be rebuilt.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RosterRebuildError {
    /// A routable expert has no Pod in the host's registry.
    MissingPod { pod: PodId },
    /// The registered Pod's manifest does not hash to the digest the ledger committed.
    ManifestDigestMismatch { pod: PodId },
}

impl fmt::Display for RosterRebuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPod { pod } => write!(f, "expert {} has no registered Pod", pod.0),
            Self::ManifestDigestMismatch { pod } => {
                write!(f, "the manifest of {} is not the committed one", pod.0)
            }
        }
    }
}

impl std::error::Error for RosterRebuildError {}

/// `base` plus every routable (`Serving` or `Frozen`) expert of `project` in `registry`, with the
/// codes its manifest declares. An expert that is also in `base` and whose manifest declares no
/// code keeps its base codes.
///
/// # Errors
/// [`RosterRebuildError`] when a routable expert has no Pod in `pods` or the Pod's manifest does
/// not hash to the digest in the registry.
pub fn rebuild_roster(
    base: Roster,
    registry: &ExpertRegistryProjection,
    pods: &PodRegistry,
    project: &ProjectId,
) -> Result<Roster, RosterRebuildError> {
    let mut roster = base;
    for (entry_project, name, entry) in registry.iter() {
        if entry_project != project.0 || !state_is_routable(entry.state) {
            continue;
        }
        let pod = PodId::from(name);
        let registered = pods
            .get(project, &pod)
            .ok_or_else(|| RosterRebuildError::MissingPod { pod: pod.clone() })?;
        let manifest = registered.manifest();
        if manifest.digest() != *entry.manifest_digest.as_bytes() {
            return Err(RosterRebuildError::ManifestDigestMismatch { pod });
        }
        let codes: Vec<FindingCode> = repaired_codes(manifest).into_iter().collect();
        if let Some(existing) = roster.repairers.iter_mut().find(|(p, _)| *p == pod) {
            // A Pod of the base roster whose manifest declares no `repairs:` capability keeps
            // the codes the base roster gave it; overwriting them with nothing would leave it
            // silently serving no code after a restart.
            if !codes.is_empty() {
                existing.1 = codes;
            }
        } else {
            roster.repairers.push((pod, codes));
        }
    }
    Ok(roster)
}
