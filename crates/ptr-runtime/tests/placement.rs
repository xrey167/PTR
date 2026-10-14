use ptr_runtime::{
    DeviceHealth, DeviceRecord, NodeHealth, NodeRecord, PlacementError, PodPlacementController,
    ProjectPlacementRequest,
};
use ptr_types::{ArtifactId, DeviceId, Generation, NodeId, PodId, ProjectId, StateId, Timestamp};

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

fn leased(controller: &mut PodPlacementController) -> ptr_runtime::FencedStateLease {
    controller.register_node(node()).unwrap();
    controller
        .assign(
            PodId::from("pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            8,
        )
        .unwrap();
    controller
        .issue_lease(StateId::from("state"), &PodId::from("pod"))
        .unwrap()
}

fn device(health: DeviceHealth) -> DeviceRecord {
    DeviceRecord {
        device_id: DeviceId::from("cuda:0"),
        vram_bytes: 24,
        used_vram_bytes: 0,
        health,
    }
}

#[test]
fn a_heartbeat_that_drops_or_fails_the_device_revokes_its_leases() {
    for devices in [vec![], vec![device(DeviceHealth::Failed)]] {
        let mut controller = PodPlacementController::default();
        let lease = leased(&mut controller);
        controller
            .heartbeat(&NodeId::from("node-a"), Timestamp(2), devices)
            .unwrap();
        assert_eq!(
            controller.validate_lease(&lease),
            Err(PlacementError::StaleLease)
        );
    }
}

#[test]
fn a_heartbeat_that_keeps_the_device_usable_keeps_the_lease() {
    let mut controller = PodPlacementController::default();
    let lease = leased(&mut controller);
    controller
        .heartbeat(
            &NodeId::from("node-a"),
            Timestamp(2),
            vec![device(DeviceHealth::Degraded)],
        )
        .unwrap();
    controller.validate_lease(&lease).unwrap();
}

#[test]
fn re_registering_a_node_without_the_device_revokes_its_leases() {
    let mut controller = PodPlacementController::default();
    let lease = leased(&mut controller);
    let mut replacement = node();
    replacement.devices.clear();
    controller.register_node(replacement).unwrap();
    assert_eq!(
        controller.validate_lease(&lease),
        Err(PlacementError::StaleLease)
    );
}

#[test]
fn no_lease_is_issued_on_a_failed_node_or_device() {
    let mut controller = PodPlacementController::default();
    leased(&mut controller);
    controller
        .mark_node_failed(&NodeId::from("node-a"))
        .unwrap();
    assert_eq!(
        controller.issue_lease(StateId::from("state-2"), &PodId::from("pod")),
        Err(PlacementError::UnhealthyNode(NodeId::from("node-a")))
    );

    let mut controller = PodPlacementController::default();
    leased(&mut controller);
    controller
        .heartbeat(
            &NodeId::from("node-a"),
            Timestamp(2),
            vec![device(DeviceHealth::Failed)],
        )
        .unwrap();
    assert_eq!(
        controller.issue_lease(StateId::from("state-2"), &PodId::from("pod")),
        Err(PlacementError::UnhealthyDevice(DeviceId::from("cuda:0")))
    );
}

fn placed(controller: &mut PodPlacementController) {
    leased(controller);
    assert!(controller.placement(&PodId::from("pod")).is_ok());
}

#[test]
fn a_device_missing_from_a_heartbeat_revokes_its_placements_and_bumps_the_epoch() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let before = controller.epoch();
    controller
        .heartbeat(&NodeId::from("node-a"), Timestamp(2), vec![])
        .unwrap();
    assert_eq!(
        controller.placement(&PodId::from("pod")).map(|p| p.epoch),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(controller.epoch().0, before.0 + 1);
}

#[test]
fn a_device_reported_failed_revokes_its_placements_and_bumps_the_epoch() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let before = controller.epoch();
    controller
        .heartbeat(
            &NodeId::from("node-a"),
            Timestamp(2),
            vec![device(DeviceHealth::Failed)],
        )
        .unwrap();
    assert_eq!(
        controller.placement(&PodId::from("pod")).map(|p| p.epoch),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(controller.epoch().0, before.0 + 1);
}

#[test]
fn a_degraded_device_keeps_its_placements() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let before = controller.epoch();
    controller
        .heartbeat(
            &NodeId::from("node-a"),
            Timestamp(2),
            vec![device(DeviceHealth::Degraded)],
        )
        .unwrap();
    assert!(controller.placement(&PodId::from("pod")).is_ok());
    assert_eq!(controller.epoch(), before);
}

#[test]
fn an_unchanged_heartbeat_keeps_every_placement_and_bumps_nothing() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let before = controller.epoch();
    let epoch = controller.placement(&PodId::from("pod")).unwrap().epoch;
    controller
        .heartbeat(&NodeId::from("node-a"), Timestamp(2), node().devices)
        .unwrap();
    assert_eq!(
        controller.placement(&PodId::from("pod")).unwrap().epoch,
        epoch
    );
    assert_eq!(controller.epoch(), before);
}

#[test]
fn a_revoked_placement_is_not_resurrected_by_a_later_heartbeat() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    controller
        .heartbeat(&NodeId::from("node-a"), Timestamp(2), vec![])
        .unwrap();
    controller
        .heartbeat(&NodeId::from("node-a"), Timestamp(3), node().devices)
        .unwrap();
    assert!(controller.placement(&PodId::from("pod")).is_err());
    assert_eq!(
        controller.issue_lease(StateId::from("state-2"), &PodId::from("pod")),
        Err(PlacementError::NoPlacement)
    );
}

#[test]
fn marking_a_node_failed_revokes_every_placement_on_it_once() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    controller
        .assign(
            PodId::from("pod-2"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let before = controller.epoch();
    controller
        .mark_node_failed(&NodeId::from("node-a"))
        .unwrap();
    assert!(controller.placement(&PodId::from("pod")).is_err());
    assert!(controller.placement(&PodId::from("pod-2")).is_err());
    assert_eq!(controller.epoch().0, before.0 + 1);
}

#[test]
fn re_registering_a_node_without_the_device_revokes_its_placements() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let before = controller.epoch();
    let mut replacement = node();
    replacement.devices.clear();
    controller.register_node(replacement).unwrap();
    assert!(controller.placement(&PodId::from("pod")).is_err());
    assert_eq!(controller.epoch().0, before.0 + 1);
}

#[test]
fn re_registering_a_node_with_the_same_devices_is_a_replacement_and_revokes_everything_on_it() {
    // The record has no incarnation: a new node process under the same id with the same
    // healthy devices must not inherit the placements and leases of the old one.
    let mut controller = PodPlacementController::default();
    let lease = leased(&mut controller);
    let before = controller.epoch();
    controller.register_node(node()).unwrap();
    assert_eq!(
        controller.placement(&PodId::from("pod")).map(|p| p.epoch),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(
        controller.validate_lease(&lease),
        Err(PlacementError::StaleLease)
    );
    assert_eq!(controller.epoch().0, before.0 + 1);
    // Re-placement under a new epoch works on the replacement.
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
    assert!(placement.epoch.0 > before.0);
}

#[test]
fn registering_another_node_keeps_the_placements_of_the_first() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let mut other = node();
    other.node_id = NodeId::from("node-b");
    controller.register_node(other).unwrap();
    assert!(controller.placement(&PodId::from("pod")).is_ok());
}

#[test]
fn a_placement_on_another_node_survives_a_failure() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let mut other = node();
    other.node_id = NodeId::from("node-b");
    controller.register_node(other).unwrap();
    controller
        .mark_node_failed(&NodeId::from("node-b"))
        .unwrap();
    assert!(controller.placement(&PodId::from("pod")).is_ok());
}

#[test]
fn a_tombstone_is_dropped_once_the_failure_is_gone() {
    let mut controller = PodPlacementController::default();
    placed(&mut controller);
    let node_id = NodeId::from("node-a");
    let pod = PodId::from("pod");
    controller
        .heartbeat(&node_id, Timestamp(2), vec![device(DeviceHealth::Failed)])
        .unwrap();
    assert_eq!(
        controller.issue_lease(StateId::from("s2"), &pod),
        Err(PlacementError::UnhealthyDevice(DeviceId::from("cuda:0")))
    );
    controller
        .heartbeat(&node_id, Timestamp(3), vec![device(DeviceHealth::Healthy)])
        .unwrap();
    assert_eq!(
        controller.issue_lease(StateId::from("s2"), &pod),
        Err(PlacementError::NoPlacement)
    );
    controller
        .heartbeat(&node_id, Timestamp(4), vec![device(DeviceHealth::Failed)])
        .unwrap();
    assert_eq!(
        controller.issue_lease(StateId::from("s2"), &pod),
        Err(PlacementError::NoPlacement)
    );
}

#[test]
fn a_node_record_or_heartbeat_naming_a_device_twice_is_refused_whole() {
    let mut controller = PodPlacementController::default();
    let mut twice = node();
    twice.devices.push(device(DeviceHealth::Failed));
    assert_eq!(
        controller.register_node(twice),
        Err(PlacementError::DuplicateDevice(DeviceId::from("cuda:0")))
    );
    placed(&mut controller);
    let before = controller.epoch();
    // A failed and a healthy record of one device cannot leave its placement standing.
    assert_eq!(
        controller.heartbeat(
            &NodeId::from("node-a"),
            Timestamp(2),
            vec![device(DeviceHealth::Failed), device(DeviceHealth::Healthy)],
        ),
        Err(PlacementError::DuplicateDevice(DeviceId::from("cuda:0")))
    );
    // Nothing was recorded: the old record stands and the epoch did not move.
    assert!(controller.placement(&PodId::from("pod")).is_ok());
    assert_eq!(controller.epoch(), before);
}

#[test]
fn revoked_placement_tombstones_are_bounded_and_drop_the_oldest() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    for index in 0..1_030 {
        controller
            .assign(
                PodId::from(format!("pod-{index}").as_str()),
                ArtifactId::from("artifact"),
                Generation(1),
                NodeId::from("node-a"),
                DeviceId::from("cuda:0"),
                1,
            )
            .unwrap();
    }
    controller
        .mark_node_failed(&NodeId::from("node-a"))
        .unwrap();
    assert_eq!(controller.revoked_placements(), 1_024);
    // The oldest placements were dropped; the newest still name the failed node.
    assert_eq!(
        controller.issue_lease(StateId::from("s"), &PodId::from("pod-0")),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(
        controller.issue_lease(StateId::from("s"), &PodId::from("pod-1029")),
        Err(PlacementError::UnhealthyNode(NodeId::from("node-a")))
    );
}

fn assign_in(controller: &mut PodPlacementController, project: &str, pod: &str) -> u64 {
    controller
        .assign_in(ProjectPlacementRequest {
            project: ProjectId::from(project),
            pod_id: PodId::from(pod),
            artifact_id: ArtifactId::from("artifact"),
            generation: Generation(1),
            node_id: NodeId::from("node-a"),
            device_id: DeviceId::from("cuda:0"),
            required_vram: 1,
        })
        .unwrap()
        .epoch
        .0
}

#[test]
fn the_same_pod_placed_in_two_projects_is_two_independent_placements() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let (a, b) = (ProjectId::from("a"), ProjectId::from("b"));
    let pod = PodId::from("stage");
    let epoch_a = assign_in(&mut controller, "a", "stage");
    // Placing the same PodId for project b does not replace or fence project a's placement.
    let epoch_b = assign_in(&mut controller, "b", "stage");
    assert_ne!(epoch_a, epoch_b);
    assert_eq!(controller.placement_in(&a, &pod).unwrap().epoch.0, epoch_a);
    assert_eq!(controller.placement_in(&b, &pod).unwrap().epoch.0, epoch_b);
    // A project that never placed it, and the legacy namespace, see nothing.
    assert_eq!(
        controller.placement_in(&ProjectId::from("c"), &pod),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(controller.placement(&pod), Err(PlacementError::NoPlacement));
    // The legacy API places into a namespace of its own: no project can see it either.
    controller
        .assign(
            pod.clone(),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    assert!(controller.placement(&pod).is_ok());
    assert_eq!(controller.placement_in(&a, &pod).unwrap().epoch.0, epoch_a);
    // Re-placing in project a moves only project a.
    let moved_a = assign_in(&mut controller, "a", "stage");
    assert_eq!(controller.placement_in(&a, &pod).unwrap().epoch.0, moved_a);
    assert_eq!(controller.placement_in(&b, &pod).unwrap().epoch.0, epoch_b);
}

#[test]
fn leases_and_freezes_are_scoped_to_their_project() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let (a, b) = (ProjectId::from("a"), ProjectId::from("b"));
    let pod = PodId::from("stage");
    assign_in(&mut controller, "a", "stage");
    assign_in(&mut controller, "b", "stage");
    let state = StateId::from("state");
    let lease_a = controller.issue_lease_in(&a, state.clone(), &pod).unwrap();
    let lease_b = controller.issue_lease_in(&b, state.clone(), &pod).unwrap();
    assert_eq!(lease_a.project(), Some(&a));
    assert_eq!(lease_b.project(), Some(&b));
    // The same state id leased in two projects is two leases: neither overwrote the other.
    controller.validate_lease(&lease_a).unwrap();
    controller.validate_lease(&lease_b).unwrap();

    // Revoking project a's lease leaves project b's untouched.
    controller.revoke_lease_in(&a, &state).unwrap();
    assert_eq!(
        controller.validate_lease(&lease_a),
        Err(PlacementError::StaleLease)
    );
    controller.validate_lease(&lease_b).unwrap();

    // Re-placing in project a revokes project a's leases only.
    let lease_a = controller.issue_lease_in(&a, state.clone(), &pod).unwrap();
    assign_in(&mut controller, "a", "stage");
    assert_eq!(
        controller.validate_lease(&lease_a),
        Err(PlacementError::StaleLease)
    );
    controller.validate_lease(&lease_b).unwrap();

    // A freeze in project b does not freeze the same state in project a.
    let lease_a = controller.issue_lease_in(&a, state.clone(), &pod).unwrap();
    controller.freeze_state(&lease_b).unwrap();
    assert_eq!(
        controller.issue_lease_in(&b, state.clone(), &pod),
        Err(PlacementError::StateFrozen)
    );
    controller.validate_lease(&lease_a).unwrap();
    controller.unfreeze_state_in(&b, &state);
    controller.issue_lease_in(&b, state, &pod).unwrap();
}

#[test]
fn a_node_failure_fences_every_project_and_each_tombstone_names_its_own_project() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let (a, b, c) = (
        ProjectId::from("a"),
        ProjectId::from("b"),
        ProjectId::from("c"),
    );
    let pod = PodId::from("stage");
    assign_in(&mut controller, "a", "stage");
    assign_in(&mut controller, "b", "stage");
    // Project c placed nothing.
    let lease_a = controller
        .issue_lease_in(&a, StateId::from("s"), &pod)
        .unwrap();
    controller
        .mark_node_failed(&NodeId::from("node-a"))
        .unwrap();
    // Nodes are host resources: the failure fences placements and leases of every project.
    assert_eq!(
        controller.validate_lease(&lease_a),
        Err(PlacementError::StaleLease)
    );
    assert_eq!(
        controller.placement_in(&a, &pod),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(
        controller.placement_in(&b, &pod),
        Err(PlacementError::NoPlacement)
    );
    assert_eq!(controller.revoked_placements(), 2);
    let named = Err(PlacementError::UnhealthyNode(NodeId::from("node-a")));
    assert_eq!(
        controller.issue_lease_in(&a, StateId::from("s"), &pod),
        named
    );
    assert_eq!(
        controller.issue_lease_in(&b, StateId::from("s"), &pod),
        named
    );
    // Project c has no tombstone to borrow: it is told there was no placement.
    assert_eq!(
        controller.issue_lease_in(&c, StateId::from("s"), &pod),
        Err(PlacementError::NoPlacement)
    );
}

#[test]
fn revoke_lease_for_uses_the_namespace_of_the_lease_it_is_given() {
    let mut controller = PodPlacementController::default();
    controller.register_node(node()).unwrap();
    let project = ProjectId::from("a");
    let pod = PodId::from("stage");
    assign_in(&mut controller, "a", "stage");
    controller
        .assign(
            pod.clone(),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    let state = StateId::from("state");
    let project_lease = controller
        .issue_lease_in(&project, state.clone(), &pod)
        .unwrap();
    let legacy_lease = controller.issue_lease(state.clone(), &pod).unwrap();

    // Revoking the project lease by object leaves the legacy lease with the same id live.
    controller.revoke_lease_for(&project_lease).unwrap();
    assert_eq!(
        controller.validate_lease(&project_lease),
        Err(PlacementError::StaleLease)
    );
    controller.validate_lease(&legacy_lease).unwrap();
    // And the reverse.
    let project_lease = controller
        .issue_lease_in(&project, state.clone(), &pod)
        .unwrap();
    controller.revoke_lease_for(&legacy_lease).unwrap();
    controller.validate_lease(&project_lease).unwrap();
    assert_eq!(
        controller.validate_lease(&legacy_lease),
        Err(PlacementError::StaleLease)
    );
}
