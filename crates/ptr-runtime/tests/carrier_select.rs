//! Carrier selection through the public API of `ptr-runtime`: every row of the table of doc 42
//! section 2 for one host with two cards, each stated as an input and an expected carrier with
//! its reason.

use ptr_protocol::Carrier;
use ptr_runtime::{
    select_carrier, Calibration, CarrierRefusal, CarrierSelection, CarrierTable, DeviceHealth,
    DeviceRecord, EndpointCapabilities, EndpointConfinement, HostProcess, NodeHealth, NodeRecord,
    PayloadOwnership, PeerProbe, SelectionReason, TensorLocation,
};
use ptr_types::{DeviceId, NodeId, Timestamp};
use std::collections::BTreeSet;

const FULL: EndpointConfinement = EndpointConfinement::Full;
const PARTIAL: EndpointConfinement = EndpointConfinement::Partial;
const UNCONFINED: EndpointConfinement = EndpointConfinement::None;

fn node_record(devices: &[(&str, DeviceHealth)]) -> NodeRecord {
    NodeRecord {
        node_id: NodeId::from("host"),
        zone: "lan".into(),
        devices: devices
            .iter()
            .map(|(name, health)| DeviceRecord {
                device_id: DeviceId::from(*name),
                vram_bytes: 24 << 30,
                used_vram_bytes: 0,
                health: *health,
            })
            .collect(),
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
    }
}

fn card(name: &str) -> DeviceId {
    DeviceId::from(name)
}

fn gpu(name: &str) -> TensorLocation {
    TensorLocation::Device(card(name))
}

fn end(process: u64, confinement: EndpointConfinement, at: TensorLocation) -> EndpointCapabilities {
    EndpointCapabilities {
        node: NodeId::from("host"),
        process: HostProcess(process),
        confinement,
        tensor_location: at,
    }
}

fn calibration(peer: u64, pinned: u64) -> Calibration {
    Calibration {
        device_peer_ns_per_mib: peer,
        pinned_host_ns_per_mib: pinned,
        measured_at: Timestamp(9),
    }
}

/// A host with two healthy cards, with `forward` recorded for `cuda:0 -> cuda:1` (if any).
fn host(forward: Option<PeerProbe>) -> CarrierTable {
    let mut table = CarrierTable::register(&node_record(&[
        ("cuda:0", DeviceHealth::Healthy),
        ("cuda:1", DeviceHealth::Healthy),
    ]))
    .expect("two cards");
    if let Some(probe) = forward {
        table
            .record_probe(&card("cuda:0"), &card("cuda:1"), probe)
            .expect("recorded");
    }
    table
}

fn select(
    producer: &EndpointCapabilities,
    consumer: &EndpointCapabilities,
    ownership: PayloadOwnership,
    table: &CarrierTable,
) -> CarrierSelection {
    select_carrier(producer, consumer, ownership, table).expect("a carrier")
}

fn expect(carrier: Carrier, reason: SelectionReason) -> CarrierSelection {
    CarrierSelection { carrier, reason }
}

#[test]
fn another_node_is_remote_whatever_else_is_true() {
    let table = host(None);
    let mut far = end(1, FULL, TensorLocation::Host);
    far.node = NodeId::from("other-host");
    for confinement in [FULL, PARTIAL, UNCONFINED] {
        for ownership in [PayloadOwnership::MovedUnique, PayloadOwnership::Shared] {
            let near = end(1, confinement, gpu("cuda:0"));
            assert_eq!(
                select(&near, &far, ownership, &table),
                expect(Carrier::Remote, SelectionReason::CrossNode)
            );
        }
    }
}

#[test]
fn the_same_process_moves_the_object_only_under_unique_ownership() {
    let table = host(None);
    for location in [TensorLocation::Host, gpu("cuda:0")] {
        let a = end(1, FULL, location.clone());
        let b = end(1, FULL, location.clone());
        assert_eq!(
            select(&a, &b, PayloadOwnership::MovedUnique, &table),
            expect(Carrier::InProcess, SelectionReason::MovedInProcess)
        );
        // Shared storage is copied into a receiver-owned buffer: never the in-process carrier.
        assert_eq!(
            select(&a, &b, PayloadOwnership::Shared, &table),
            expect(Carrier::Bytes, SelectionReason::SharedStorageCopy)
        );
        // Confinement does not matter inside one process.
        let loose = end(1, UNCONFINED, location);
        assert_eq!(
            select(&loose, &loose, PayloadOwnership::MovedUnique, &table).carrier,
            Carrier::InProcess
        );
    }
}

#[test]
fn in_process_is_never_chosen_across_processes_even_with_unique_ownership() {
    let table = host(None);
    let a = end(1, FULL, TensorLocation::Host);
    let b = end(2, FULL, TensorLocation::Host);
    assert_ne!(
        select(&a, &b, PayloadOwnership::MovedUnique, &table).carrier,
        Carrier::InProcess
    );
}

#[test]
fn one_process_across_two_cards_still_crosses_devices() {
    // One process owning both cards cannot move the object from one card to the other: the
    // bytes cross devices, so the cross-process rows decide.
    let table = host(None);
    let a = end(1, FULL, gpu("cuda:0"));
    let b = end(1, FULL, gpu("cuda:1"));
    assert_eq!(
        select(&a, &b, PayloadOwnership::MovedUnique, &table),
        expect(Carrier::PinnedHost, SelectionReason::PeerUnprobed)
    );
}

#[test]
fn cpu_tensors_across_processes_on_one_host_use_a_sealed_region_when_both_are_confined() {
    let table = host(None);
    let a = end(1, FULL, TensorLocation::Host);
    let b = end(2, FULL, TensorLocation::Host);
    assert_eq!(
        select(&a, &b, PayloadOwnership::MovedUnique, &table),
        expect(Carrier::HostShm, SelectionReason::HostToHost)
    );
}

#[test]
fn the_same_card_in_another_process_uses_cuda_ipc() {
    let table = host(None);
    let a = end(1, FULL, gpu("cuda:1"));
    let b = end(2, FULL, gpu("cuda:1"));
    assert_eq!(
        select(&a, &b, PayloadOwnership::Shared, &table),
        expect(Carrier::CudaIpc, SelectionReason::SameGpu)
    );
}

#[test]
fn host_memory_to_a_card_and_back_goes_through_a_pinned_buffer() {
    let table = host(None);
    let cpu = end(1, FULL, TensorLocation::Host);
    let card0 = end(2, FULL, gpu("cuda:0"));
    let expected = expect(Carrier::PinnedHost, SelectionReason::HostDeviceHop);
    assert_eq!(
        select(&cpu, &card0, PayloadOwnership::MovedUnique, &table),
        expected
    );
    assert_eq!(
        select(&card0, &cpu, PayloadOwnership::MovedUnique, &table),
        expected
    );
}

#[test]
fn two_cards_use_the_pinned_host_unless_this_host_measured_the_peer_path_faster() {
    let a = end(1, FULL, gpu("cuda:0"));
    let b = end(2, FULL, gpu("cuda:1"));
    let own = PayloadOwnership::MovedUnique;
    let cases = [
        (
            None,
            expect(Carrier::PinnedHost, SelectionReason::PeerUnprobed),
        ),
        (
            Some(PeerProbe::NoPeer),
            expect(Carrier::PinnedHost, SelectionReason::PeerUnavailable),
        ),
        (
            Some(PeerProbe::Peer { calibrated: None }),
            expect(Carrier::PinnedHost, SelectionReason::PeerNotCalibrated),
        ),
        (
            Some(PeerProbe::Peer {
                calibrated: Some(calibration(900, 600)),
            }),
            expect(Carrier::PinnedHost, SelectionReason::PeerNotFaster),
        ),
        (
            Some(PeerProbe::Peer {
                calibrated: Some(calibration(600, 600)),
            }),
            expect(Carrier::PinnedHost, SelectionReason::PeerNotFaster),
        ),
        (
            Some(PeerProbe::Peer {
                calibrated: Some(calibration(300, 600)),
            }),
            expect(Carrier::DevicePeer, SelectionReason::PeerCalibratedFaster),
        ),
    ];
    for (probe, expected) in cases {
        assert_eq!(select(&a, &b, own, &host(probe)), expected, "{probe:?}");
    }
}

#[test]
fn a_peer_row_is_per_ordered_pair() {
    let faster = PeerProbe::Peer {
        calibrated: Some(calibration(300, 600)),
    };
    let table = host(Some(faster));
    let forward = (end(1, FULL, gpu("cuda:0")), end(2, FULL, gpu("cuda:1")));
    assert_eq!(
        select(
            &forward.0,
            &forward.1,
            PayloadOwnership::MovedUnique,
            &table
        )
        .carrier,
        Carrier::DevicePeer
    );
    // The reverse direction was never probed: it is not assumed from the forward one.
    assert_eq!(
        select(
            &forward.1,
            &forward.0,
            PayloadOwnership::MovedUnique,
            &table
        ),
        expect(Carrier::PinnedHost, SelectionReason::PeerUnprobed)
    );
}

#[test]
fn a_shared_carrier_needs_both_endpoints_fully_confined() {
    let faster = host(Some(PeerProbe::Peer {
        calibrated: Some(calibration(300, 600)),
    }));
    let locations = [
        (TensorLocation::Host, TensorLocation::Host),
        (gpu("cuda:1"), gpu("cuda:1")),
        (TensorLocation::Host, gpu("cuda:0")),
        (gpu("cuda:0"), gpu("cuda:1")),
    ];
    let shared = [
        Carrier::HostShm,
        Carrier::CudaIpc,
        Carrier::PinnedHost,
        Carrier::DevicePeer,
    ];
    for (from, to) in locations {
        for (first, second) in [
            (FULL, FULL),
            (FULL, PARTIAL),
            (PARTIAL, FULL),
            (PARTIAL, PARTIAL),
            (FULL, UNCONFINED),
            (UNCONFINED, FULL),
            (UNCONFINED, UNCONFINED),
        ] {
            let a = end(1, first, from.clone());
            let b = end(2, second, to.clone());
            let chosen = select(&a, &b, PayloadOwnership::MovedUnique, &faster);
            if first == FULL && second == FULL {
                assert!(shared.contains(&chosen.carrier), "{from:?} -> {to:?}");
            } else {
                assert_eq!(
                    chosen,
                    expect(Carrier::Bytes, SelectionReason::ConfinementNotFull),
                    "{from:?} -> {to:?} under {first:?}/{second:?}"
                );
            }
        }
    }
}

#[test]
fn a_failed_or_vanished_device_is_refused_not_routed_around() {
    let mut table = host(Some(PeerProbe::Peer {
        calibrated: Some(calibration(300, 600)),
    }));
    let a = end(1, FULL, gpu("cuda:0"));
    let b = end(2, FULL, gpu("cuda:1"));
    assert_eq!(
        select(&a, &b, PayloadOwnership::MovedUnique, &table).carrier,
        Carrier::DevicePeer
    );
    // The next heartbeat reports cuda:1 failed.
    table
        .reconcile(&node_record(&[
            ("cuda:0", DeviceHealth::Healthy),
            ("cuda:1", DeviceHealth::Failed),
        ]))
        .expect("reconciled");
    for (from, to) in [(&a, &b), (&b, &a), (&b, &b)] {
        assert_eq!(
            select_carrier(from, to, PayloadOwnership::MovedUnique, &table),
            Err(CarrierRefusal::DeviceUnavailable(card("cuda:1")))
        );
    }
    // A device that vanished from the record altogether is the same.
    table
        .reconcile(&node_record(&[("cuda:0", DeviceHealth::Healthy)]))
        .expect("reconciled");
    assert_eq!(
        select_carrier(&a, &b, PayloadOwnership::MovedUnique, &table),
        Err(CarrierRefusal::DeviceUnavailable(card("cuda:1")))
    );
    // The surviving card still works, and the revived one must be probed again.
    let host_side = end(3, FULL, TensorLocation::Host);
    assert_eq!(
        select(&a, &host_side, PayloadOwnership::Shared, &table).carrier,
        Carrier::PinnedHost
    );
    table
        .reconcile(&node_record(&[
            ("cuda:0", DeviceHealth::Healthy),
            ("cuda:1", DeviceHealth::Healthy),
        ]))
        .expect("reconciled");
    assert_eq!(
        select(&a, &b, PayloadOwnership::MovedUnique, &table),
        expect(Carrier::PinnedHost, SelectionReason::PeerUnprobed)
    );
}

#[test]
fn across_nodes_a_failed_local_device_is_still_remote() {
    // The table is the producer's node's; cross-node selection does not consult it, so the
    // choice is Remote even for an endpoint naming a device the table dropped. Whether the
    // remote side can serve is the placement fence's business, not the carrier's.
    let mut table = host(None);
    table
        .reconcile(&node_record(&[
            ("cuda:0", DeviceHealth::Healthy),
            ("cuda:1", DeviceHealth::Failed),
        ]))
        .expect("reconciled");
    let mut far = end(2, FULL, gpu("cuda:1"));
    far.node = NodeId::from("other-host");
    let near = end(1, FULL, gpu("cuda:1"));
    assert_eq!(
        select(&near, &far, PayloadOwnership::MovedUnique, &table),
        expect(Carrier::Remote, SelectionReason::CrossNode)
    );
}

#[test]
fn selection_is_deterministic_and_never_picks_a_tier_carrier() {
    let table = host(Some(PeerProbe::Peer {
        calibrated: Some(calibration(300, 600)),
    }));
    let locations = [TensorLocation::Host, gpu("cuda:0"), gpu("cuda:1")];
    let mut seen = BTreeSet::new();
    for from in &locations {
        for to in &locations {
            for (p, q) in [(1, 1), (1, 2)] {
                for confinement in [FULL, PARTIAL, UNCONFINED] {
                    for ownership in [PayloadOwnership::MovedUnique, PayloadOwnership::Shared] {
                        let a = end(p, confinement, from.clone());
                        let b = end(q, confinement, to.clone());
                        let first = select_carrier(&a, &b, ownership, &table).expect("selected");
                        assert_eq!(
                            Ok(first),
                            select_carrier(&a, &b, ownership, &table),
                            "same input, same answer"
                        );
                        seen.insert(first.carrier);
                    }
                }
            }
        }
    }
    assert!(!seen.contains(&Carrier::NvmeFile));
    let expected: BTreeSet<Carrier> = [
        Carrier::InProcess,
        Carrier::Bytes,
        Carrier::HostShm,
        Carrier::CudaIpc,
        Carrier::DevicePeer,
        Carrier::PinnedHost,
    ]
    .into_iter()
    .collect();
    assert_eq!(seen, expected);
}
