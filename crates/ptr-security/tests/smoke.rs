use ptr_security::{
    ActionAuthorization, AuthorizationDecision, AuthorizationDenial, PermissionSet,
};
use ptr_types::{CapabilityId, Effect, Generation, Revision};

fn request() -> ActionAuthorization {
    ActionAuthorization {
        target: "artifact:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        action_revision: Revision(2),
        current_revision: Revision(2),
        action_generation: Generation(7),
        current_generation: Some(Generation(7)),
        generation_revoked: false,
        require_capability: true,
        require_current_revision: true,
        require_live_generation: true,
    }
}

#[test]
fn missing_capability_is_typed_denial() {
    let permissions = PermissionSet::default();
    assert_eq!(
        permissions.authorize(request()),
        AuthorizationDecision::Deny(AuthorizationDenial::MissingCapability {
            capability: CapabilityId::from("file.write"),
        })
    );
}

#[test]
fn stale_revision_precedes_permission_checks() {
    let permissions = PermissionSet::default();
    let mut action = request();
    action.action_revision = Revision(1);

    assert_eq!(
        permissions.authorize(action),
        AuthorizationDecision::Deny(AuthorizationDenial::StaleRevision {
            action: Revision(1),
            current: Revision(2),
        })
    );
}

#[test]
fn revoked_generation_is_typed_denial() {
    let permissions = PermissionSet::default();
    let mut action = request();
    action.generation_revoked = true;

    assert_eq!(
        permissions.authorize(action),
        AuthorizationDecision::Deny(AuthorizationDenial::StaleGeneration {
            target: "artifact:a".into(),
            action: Generation(7),
            current: Some(Generation(7)),
        })
    );
}

#[test]
fn denied_effect_is_distinct_from_missing_capability() {
    let mut permissions = PermissionSet::default();
    permissions
        .capabilities
        .insert(CapabilityId::from("file.write"));

    assert_eq!(
        permissions.authorize(request()),
        AuthorizationDecision::Deny(AuthorizationDenial::EffectNotPermitted {
            effect: Effect::Mutation,
        })
    );
}

#[test]
fn allowed_action_returns_audit_ready_receipt() {
    let mut permissions = PermissionSet::default();
    permissions
        .capabilities
        .insert(CapabilityId::from("file.write"));
    permissions.allow_mutation = true;

    let decision = permissions.authorize(request());
    let AuthorizationDecision::Allow(receipt) = decision else {
        panic!("expected allowed authorization receipt");
    };

    assert_eq!(receipt.target, "artifact:a");
    assert_eq!(receipt.revision, Revision(2));
    assert_eq!(receipt.generation, Generation(7));
    assert_eq!(receipt.capability, CapabilityId::from("file.write"));
    assert_eq!(receipt.effect, Effect::Mutation);
}

#[test]
fn disabling_capability_membership_does_not_disable_effect_authority() {
    // Give membership so this test continues to isolate the effect gate;
    // hard_boundary.rs separately verifies that membership cannot be bypassed.
    let mut permissions = PermissionSet::default();
    permissions
        .capabilities
        .insert(CapabilityId::from("file.write"));
    let mut action = request();
    action.require_capability = false;

    assert_eq!(
        permissions.authorize(action),
        AuthorizationDecision::Deny(AuthorizationDenial::EffectNotPermitted {
            effect: Effect::Mutation,
        })
    );
}
