//! Code routes from the verifier crate's code table.
//!
//! `ptr-verifier` states, for each hard finding code it emits, the [`RepairRole`] that should
//! repair it. This module turns an assignment of Pods to roles into the repair roster of a
//! [`FindingRegistry`] (and of the runtime's `Roster`): every Pod assigned a role is registered
//! for exactly the codes of that role. It changes neither [`classify`](crate::classify) nor any
//! routing rule: it only decides which `(code, Pod)` pairs get registered. A soft code has no
//! role and is never registered, so an `Unknown` report still goes through the fallback policy.

use std::collections::{BTreeMap, BTreeSet};

use ptr_types::PodId;
use ptr_verifier::{codes_for_role, RepairRole};

use crate::findings::{
    FindingCode, FindingRegistry, RegistryChange, RegistryError, RegistryGeneration, RegistryUpdate,
};

/// The finding codes `role` repairs, in the verifier table's order.
pub fn finding_codes_for_role(role: RepairRole) -> Vec<FindingCode> {
    codes_for_role(role)
        .into_iter()
        .map(FindingCode::new)
        .collect()
}

/// Which Pods fill which repair roles.
#[derive(Clone, Debug, Default)]
pub struct RepairRoleAssignments {
    roles: BTreeMap<RepairRole, BTreeSet<PodId>>,
}

impl RepairRoleAssignments {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assigns `pod` the repair role `role` (idempotent).
    pub fn assign(mut self, role: RepairRole, pod: PodId) -> Self {
        self.roles.entry(role).or_default().insert(pod);
        self
    }

    /// The Pods of `role`, in `PodId` order.
    pub fn pods(&self, role: RepairRole) -> Vec<PodId> {
        self.roles
            .get(&role)
            .map_or_else(Vec::new, |pods| pods.iter().cloned().collect())
    }

    /// One entry per Pod, in `PodId` order, with the codes of all its roles (deduplicated, in
    /// role then table order): the shape of the runtime's `Roster::repairers`.
    pub fn repairers(&self) -> Vec<(PodId, Vec<FindingCode>)> {
        let mut by_pod: BTreeMap<PodId, Vec<FindingCode>> = BTreeMap::new();
        for (role, pods) in &self.roles {
            for pod in pods {
                let codes = by_pod.entry(pod.clone()).or_default();
                for code in finding_codes_for_role(*role) {
                    if !codes.contains(&code) {
                        codes.push(code);
                    }
                }
            }
        }
        by_pod.into_iter().collect()
    }

    /// Registers every `(code, Pod)` pair in `registry`, one accepted update each, and returns
    /// the registry's generation afterwards. A pair already registered is skipped; any other
    /// refusal stops and is returned (earlier pairs stay registered, as each update is atomic).
    pub fn register(
        &self,
        registry: &mut FindingRegistry,
    ) -> Result<RegistryGeneration, RegistryError> {
        for (pod, codes) in self.repairers() {
            for code in codes {
                let update = RegistryUpdate {
                    expected: registry.generation(),
                    change: RegistryChange::AddRepair {
                        code,
                        pod: pod.clone(),
                    },
                };
                match registry.apply(update) {
                    Ok(_) | Err(RegistryError::AlreadyRegistered) => {}
                    Err(other) => return Err(other),
                }
            }
        }
        Ok(registry.generation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pod(name: &str) -> PodId {
        PodId::from(name)
    }

    #[test]
    fn a_role_registers_its_pod_for_exactly_the_role_codes() {
        let assignments = RepairRoleAssignments::new()
            .assign(RepairRole::Schema, pod("schema"))
            .assign(RepairRole::Format, pod("fmt"))
            .assign(RepairRole::Format, pod("schema"));
        let mut registry = FindingRegistry::new();
        assignments.register(&mut registry).unwrap();
        assert_eq!(
            registry.repair_pods(&FindingCode::new("json.type")),
            vec![pod("schema")]
        );
        assert_eq!(
            registry.repair_pods(&FindingCode::new("json.parse")),
            vec![pod("fmt"), pod("schema")]
        );
        // A soft code and a role nobody holds have no repair Pod.
        assert!(registry
            .repair_pods(&FindingCode::new("fact.unknown"))
            .is_empty());
        assert!(registry
            .repair_pods(&FindingCode::new("code.syntax"))
            .is_empty());
        assert_eq!(assignments.pods(RepairRole::Schema), vec![pod("schema")]);
    }

    #[test]
    fn repairers_have_the_roster_shape_and_no_duplicate_codes() {
        let roster = RepairRoleAssignments::new()
            .assign(RepairRole::Code, pod("b"))
            .assign(RepairRole::Resource, pod("b"))
            .assign(RepairRole::Fact, pod("a"))
            .repairers();
        assert_eq!(roster[0].0, pod("a"));
        assert_eq!(roster[0].1, vec![FindingCode::new("fact.contradicted")]);
        assert_eq!(roster[1].0, pod("b"));
        let names: Vec<&str> = roster[1].1.iter().map(FindingCode::as_str).collect();
        assert_eq!(
            names,
            [
                "code.syntax",
                "code.test_failed",
                "code.timeout",
                "code.resource"
            ]
        );
    }

    #[test]
    fn registering_twice_is_idempotent_and_capacity_is_reported() {
        let assignments = RepairRoleAssignments::new().assign(RepairRole::Value, pod("v"));
        let mut registry = FindingRegistry::new();
        let first = assignments.register(&mut registry).unwrap();
        assert_eq!(assignments.register(&mut registry).unwrap(), first);
        let mut tiny = FindingRegistry::with_capacity(1, 1);
        assert_eq!(
            assignments.register(&mut tiny),
            Err(RegistryError::CapacityExceeded)
        );
    }

    #[test]
    fn every_hard_code_of_the_table_is_routable_once_its_role_is_filled() {
        let mut assignments = RepairRoleAssignments::new();
        for role in RepairRole::ALL {
            assignments = assignments.assign(role, pod(role.as_str()));
        }
        let mut registry = FindingRegistry::new();
        assignments.register(&mut registry).unwrap();
        for spec in ptr_verifier::CODE_TABLE {
            let pods = registry.repair_pods(&FindingCode::new(spec.code));
            assert_eq!(pods.is_empty(), !spec.hard, "{}", spec.code);
        }
    }
}
