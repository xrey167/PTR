use ptr_types::{CapabilityId, Effect, Generation, Revision};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default)]
pub struct PermissionSet {
    pub capabilities: BTreeSet<CapabilityId>,
    pub allow_mutation: bool,
    pub allow_external: bool,
    pub allow_irreversible: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionAuthorization {
    pub target: String,
    pub capability: CapabilityId,
    pub effect: Effect,
    pub action_revision: Revision,
    pub current_revision: Revision,
    pub action_generation: Generation,
    pub current_generation: Option<Generation>,
    pub generation_revoked: bool,
    pub require_capability: bool,
    pub require_current_revision: bool,
    pub require_live_generation: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthorizationDenial {
    StaleRevision {
        action: Revision,
        current: Revision,
    },
    UnknownGeneration {
        target: String,
    },
    StaleGeneration {
        target: String,
        action: Generation,
        current: Option<Generation>,
    },
    MissingCapability {
        capability: CapabilityId,
    },
    EffectNotPermitted {
        effect: Effect,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizationReceipt {
    pub target: String,
    pub revision: Revision,
    pub generation: Generation,
    pub capability: CapabilityId,
    pub effect: Effect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthorizationDecision {
    Allow(AuthorizationReceipt),
    Deny(AuthorizationDenial),
}

impl PermissionSet {
    pub fn allows(&self, capability: &CapabilityId, effect: Effect) -> bool {
        self.capabilities.contains(capability) && self.allows_effect(effect)
    }

    pub fn authorize(&self, request: ActionAuthorization) -> AuthorizationDecision {
        if request.require_current_revision && request.action_revision != request.current_revision {
            return AuthorizationDecision::Deny(AuthorizationDenial::StaleRevision {
                action: request.action_revision,
                current: request.current_revision,
            });
        }

        if request.require_live_generation {
            if request.generation_revoked
                || request
                    .current_generation
                    .is_some_and(|generation| generation != request.action_generation)
            {
                return AuthorizationDecision::Deny(AuthorizationDenial::StaleGeneration {
                    target: request.target,
                    action: request.action_generation,
                    current: request.current_generation,
                });
            }

            if request.current_generation.is_none() {
                return AuthorizationDecision::Deny(AuthorizationDenial::UnknownGeneration {
                    target: request.target,
                });
            }
        }

        if request.require_capability {
            if !self.capabilities.contains(&request.capability) {
                return AuthorizationDecision::Deny(AuthorizationDenial::MissingCapability {
                    capability: request.capability,
                });
            }

            if !self.allows_effect(request.effect) {
                return AuthorizationDecision::Deny(AuthorizationDenial::EffectNotPermitted {
                    effect: request.effect,
                });
            }
        }

        AuthorizationDecision::Allow(AuthorizationReceipt {
            target: request.target,
            revision: request.action_revision,
            generation: request.action_generation,
            capability: request.capability,
            effect: request.effect,
        })
    }

    fn allows_effect(&self, effect: Effect) -> bool {
        match effect {
            Effect::Pure | Effect::Read => true,
            Effect::Mutation => self.allow_mutation,
            Effect::External => self.allow_external,
            Effect::Irreversible => self.allow_irreversible,
        }
    }
}
