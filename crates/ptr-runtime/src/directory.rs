use ptr_types::{PodEndpoint, PodRevisionAddress, PodRoute, ResolveError, RouteConstraints};
use std::collections::BTreeMap;

/// Resolves stable logical Pod addresses to currently admitted physical
/// endpoints. Resolution is deliberately separate from identity and runtime
/// admission: a route is a candidate, never an authorization grant.
pub trait PodDirectory {
    fn resolve(
        &self,
        address: &PodRevisionAddress,
        constraints: &RouteConstraints,
    ) -> Result<PodRoute, ResolveError>;
}

#[derive(Default)]
pub struct InMemoryPodDirectory {
    endpoints: BTreeMap<PodRevisionAddress, PodEndpoint>,
}

impl InMemoryPodDirectory {
    pub fn register(&mut self, endpoint: PodEndpoint) -> Result<(), ResolveError> {
        endpoint.pod.validate()?;
        if endpoint.endpoint_epoch == 0
            || endpoint.fencing_token == 0
            || endpoint.artifact_id.0.is_empty()
            || !endpoint.artifact_active
            || !endpoint.healthy
        {
            return Err(ResolveError::StaleRoute);
        }
        if let Some(mesh) = &endpoint.mesh {
            mesh.validate()?;
            if mesh.peer_id != endpoint.peer_id {
                return Err(ResolveError::ConflictingEndpoint);
            }
        }
        if let Some(current) = self.endpoints.get(&endpoint.pod) {
            if endpoint.endpoint_epoch < current.endpoint_epoch {
                return Err(ResolveError::StaleRoute);
            }
            if endpoint.endpoint_epoch == current.endpoint_epoch {
                return if current == &endpoint {
                    Ok(())
                } else {
                    Err(ResolveError::ConflictingEndpoint)
                };
            }
        }
        self.endpoints.insert(endpoint.pod.clone(), endpoint);
        Ok(())
    }

    pub fn remove(&mut self, address: &PodRevisionAddress) -> Option<PodEndpoint> {
        self.endpoints.remove(address)
    }

    pub fn rebind(&mut self, endpoint: PodEndpoint) -> Result<(), ResolveError> {
        self.register(endpoint)
    }
}

impl PodDirectory for InMemoryPodDirectory {
    fn resolve(
        &self,
        address: &PodRevisionAddress,
        constraints: &RouteConstraints,
    ) -> Result<PodRoute, ResolveError> {
        let endpoint = self.endpoints.get(address).ok_or(ResolveError::NotFound)?;
        if !endpoint.healthy {
            return Err(ResolveError::StaleRoute);
        }
        if !endpoint.artifact_active {
            return Err(ResolveError::StaleRoute);
        }
        if let Some(artifact_id) = &constraints.artifact_id {
            if &endpoint.artifact_id != artifact_id {
                return Err(ResolveError::ConstraintMismatch { field: "artifact" });
            }
        }
        if let Some(region) = &constraints.region {
            if &endpoint.region != region {
                return Err(ResolveError::ConstraintMismatch { field: "region" });
            }
        }
        if let Some(zone) = &constraints.zone {
            if &endpoint.zone != zone {
                return Err(ResolveError::ConstraintMismatch { field: "zone" });
            }
        }
        if let Some(min_epoch) = constraints.min_epoch {
            if endpoint.endpoint_epoch < min_epoch {
                return Err(ResolveError::StaleRoute);
            }
        }
        if let Some(required) = constraints.required_vram_bytes {
            if endpoint.available_vram_bytes < required {
                return Err(ResolveError::ConstraintMismatch { field: "vram" });
            }
        }
        let capability = constraints
            .capability
            .clone()
            .ok_or(ResolveError::MissingBinding {
                field: "capability",
            })?;
        let input_type = constraints
            .input_type
            .clone()
            .ok_or(ResolveError::MissingBinding {
                field: "input_type",
            })?;
        if !endpoint.capabilities.contains(&capability) {
            return Err(ResolveError::ConstraintMismatch {
                field: "capability",
            });
        }
        if !endpoint.accepts.contains(&input_type) {
            return Err(ResolveError::ConstraintMismatch {
                field: "input_type",
            });
        }
        let output_type = constraints
            .output_type
            .clone()
            .or_else(|| endpoint.produces.first().cloned())
            .ok_or(ResolveError::MissingBinding {
                field: "output_type",
            })?;
        if !endpoint.produces.contains(&output_type) {
            return Err(ResolveError::ConstraintMismatch {
                field: "output_type",
            });
        }
        let route = PodRoute {
            source: None,
            destination: address.clone(),
            artifact_id: endpoint.artifact_id.clone(),
            endpoint: endpoint.clone(),
            capability,
            input_type,
            output_type,
            trace_id: "directory".into(),
            request_id: "directory-resolve".into(),
            deadline: ptr_types::Timestamp(0),
            hop_limit: 16,
            visited: Vec::new(),
            placement_epoch: endpoint.endpoint_epoch,
            fencing_token: endpoint.fencing_token,
        };
        route.validate()?;
        Ok(route)
    }
}
