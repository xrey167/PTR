use ptr_security::{
    ActionAuthorization, AuthorizationDecision, AuthorizationDenial, PermissionSet,
};
use ptr_types::{CapabilityId, Effect, Generation, Revision};

fn request(effect: Effect, flags: u8) -> ActionAuthorization {
    ActionAuthorization {
        target: "capsule:a".into(),
        capability: CapabilityId::from("modify"),
        effect,
        action_revision: Revision(2),
        current_revision: Revision(2),
        action_generation: Generation(7),
        current_generation: Some(Generation(7)),
        generation_revoked: false,
        require_capability: flags & 1 != 0,
        require_current_revision: flags & 2 != 0,
        require_live_generation: flags & 4 != 0,
    }
}

#[test]
fn no_switch_combination_bypasses_a_side_effect_boundary() {
    for effect in [Effect::Mutation, Effect::External, Effect::Irreversible] {
        for flags in 0..8 {
            let mut permissions = PermissionSet {
                allow_mutation: true,
                allow_external: true,
                allow_irreversible: true,
                ..PermissionSet::default()
            };
            permissions
                .capabilities
                .insert(CapabilityId::from("modify"));
            let valid = request(effect, flags);
            assert!(
                matches!(
                    permissions.authorize(valid.clone()),
                    AuthorizationDecision::Allow(_)
                ),
                "valid control: {effect:?}, switches {flags}"
            );

            let mut stale = valid.clone();
            stale.action_revision = Revision(1);
            assert!(matches!(
                permissions.authorize(stale),
                AuthorizationDecision::Deny(AuthorizationDenial::StaleRevision { .. })
            ));

            let mut unknown = valid.clone();
            unknown.current_generation = None;
            assert!(matches!(
                permissions.authorize(unknown),
                AuthorizationDecision::Deny(AuthorizationDenial::UnknownGeneration { .. })
            ));

            let mut superseded = valid.clone();
            superseded.current_generation = Some(Generation(8));
            assert!(matches!(
                permissions.authorize(superseded),
                AuthorizationDecision::Deny(AuthorizationDenial::StaleGeneration { .. })
            ));

            let mut revoked = valid.clone();
            revoked.generation_revoked = true;
            assert!(matches!(
                permissions.authorize(revoked),
                AuthorizationDecision::Deny(AuthorizationDenial::StaleGeneration { .. })
            ));

            permissions.capabilities.clear();
            assert!(matches!(
                permissions.authorize(valid),
                AuthorizationDecision::Deny(AuthorizationDenial::MissingCapability { .. })
            ));
        }
    }
}

#[test]
fn pure_research_requests_keep_the_explicit_optional_policy() {
    let mut pure = request(Effect::Pure, 0);
    pure.current_generation = None;
    pure.action_revision = Revision(1);
    assert!(matches!(
        PermissionSet::default().authorize(pure),
        AuthorizationDecision::Allow(_)
    ));
}
