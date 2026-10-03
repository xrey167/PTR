use ptr_runtime::{
    DeviceHealth, DeviceRecord, NodeHealth, NodeRecord, PlacementError, PodPlacementController,
};
use ptr_types::{ArtifactId, DeviceId, Generation, NodeId, PodId, StateId, Timestamp};

fn node() -> NodeRecord {
    NodeRecord {
        node_id: NodeId::from("node-a"),
        zone: "zone-a".into(),
        devices: vec![DeviceRecord {
            device_id: DeviceId::from("cuda:0"),
            vram_bytes: 24,
            used_vram_bytes: 0,
            health: DeviceHealth::Healthy,
        }],
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
    }
}

#[test]
fn placement_epoch_fences_old_state_leases() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let placement = controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            8,
        )
        .unwrap();
    let lease = controller
        .issue_lease(StateId::from("state"), &placement.pod_id)
        .unwrap();
    controller.validate_lease(&lease).unwrap();

    controller
        .mark_node_failed(&NodeId::from("node-a"))
        .unwrap();
    assert_eq!(
        controller.validate_lease(&lease),
        Err(PlacementError::StaleLease)
    );
}

#[test]
fn placement_rejects_insufficient_vram_and_stale_heartbeat() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    assert_eq!(
        controller.assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            25
        ),
        Err(PlacementError::InsufficientVram)
    );
    assert_eq!(
        controller.heartbeat(&NodeId::from("node-a"), Timestamp(0), vec![]),
        Err(PlacementError::StaleHeartbeat(NodeId::from("node-a")))
    );
}

#[test]
fn replacing_placement_revokes_previous_write_lease() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let placement = controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let lease = controller
        .issue_lease(StateId::from("state"), &placement.pod_id)
        .unwrap();
    controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(2),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    assert_eq!(
        controller.validate_lease(&lease),
        Err(PlacementError::StaleLease)
    );
}

#[test]
fn issuing_a_second_lease_for_one_state_fences_the_previous_token() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let placement = controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let first = controller
        .issue_lease(StateId::from("state"), &placement.pod_id)
        .unwrap();
    let second = controller
        .issue_lease(StateId::from("state"), &placement.pod_id)
        .unwrap();
    assert_ne!(first.fencing_token(), second.fencing_token());
    assert_eq!(
        controller.validate_lease(&first),
        Err(PlacementError::StaleLease)
    );
    controller.validate_lease(&second).unwrap();
}

#[test]
fn revoking_a_lease_invalidates_only_that_state() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let placement = controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let revoked = controller
        .issue_lease(StateId::from("revoked"), &placement.pod_id)
        .unwrap();
    let retained = controller
        .issue_lease(StateId::from("retained"), &placement.pod_id)
        .unwrap();
    controller.revoke_lease(&StateId::from("revoked")).unwrap();
    assert_eq!(
        controller.validate_lease(&revoked),
        Err(PlacementError::StaleLease)
    );
    controller.validate_lease(&retained).unwrap();
}

#[test]
fn a_recovered_node_gets_a_new_epoch_and_generation_bound_lease() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let first = controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let old = controller
        .issue_lease(StateId::from("state"), &first.pod_id)
        .unwrap();
    controller
        .mark_node_failed(&NodeId::from("node-a"))
        .unwrap();
    controller
        .heartbeat(&NodeId::from("node-a"), Timestamp(2), node().devices)
        .unwrap();
    let recovered = controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(2),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let fresh = controller
        .issue_lease(StateId::from("state"), &recovered.pod_id)
        .unwrap();
    assert!(recovered.epoch > old.placement_epoch());
    assert_eq!(fresh.generation(), Generation(2));
    assert_ne!(fresh.fencing_token(), old.fencing_token());
    assert_eq!(
        controller.validate_lease(&old),
        Err(PlacementError::StaleLease)
    );
    controller.validate_lease(&fresh).unwrap();
}
