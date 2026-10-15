//! Carrier selection for the Pod data plane (doc 42 section 2, stage S2).
//!
//! This module only **decides** which [`Carrier`] a tensor handle should name for one producer
//! and one consumer. It moves nothing, maps nothing and holds no pointer; the carriers that do
//! are other code (`ptr-fabric`), and whether a carrier is *permitted* for a confined worker is
//! stated here as an input ([`EndpointConfinement`]) and enforced by the supervisor.
//!
//! Two pieces:
//!
//! * [`CarrierTable`], the per-host record of which ordered device pairs have peer access and
//!   what this host measured for them. Peer access is **probed** at node registration and
//!   recorded; it is never assumed, and it is never inferred from the card model (doc 42: on an
//!   RTX 4090 pair PCIe peer access is, to the best of the author's knowledge, not offered). It
//!   is a structure of its own beside [`NodeRecord`]: the node, device and placement records
//!   carry no carrier field.
//! * [`select_carrier`], a pure, deterministic function of the two [`EndpointCapabilities`], the
//!   [`PayloadOwnership`] the caller states and the table.
//!
//! The decision, in order:
//!
//! 1. Producer and consumer on **different nodes**: `Remote`.
//! 2. Same node: a device named by an endpoint must be live in the table, else the selection is
//!    refused ([`CarrierRefusal::DeviceUnavailable`]); a failed device is never a fallback.
//! 3. **Same process**, tensor at the same location, ownership moved and unique: `InProcess`.
//!    Shared storage in the same process: `Bytes` (a receiver-owned copy). Same process but
//!    another device: the cross-process rows below, because the bytes must still cross devices.
//! 4. **Other process**, and either endpoint not fully confined: `Bytes`. The shared carriers
//!    (`HostShm`, `CudaIpc`, `DevicePeer`, `PinnedHost`) are permitted only between two fully
//!    confined workers (doc 42 rule 3b).
//! 5. Both fully confined: host to host `HostShm`; same GPU `CudaIpc`; two GPUs `DevicePeer`
//!    only if the ordered pair was probed as peer-capable **and** this host's calibration shows
//!    `DevicePeer` strictly faster than `PinnedHost` for that pair, else `PinnedHost`; a hop
//!    between host memory and a GPU `PinnedHost`.
//!
//! `NvmeFile` is the tier's carrier, not a transfer between two workers, and is never selected
//! here. Nothing here reads a clock or a device. The result of a probe and of a calibration
//! comes from the caller (the node's registration path and E011's measurement), and
//! [`CarrierTable::reconcile`] drops what a heartbeat shows to be gone.

use std::collections::{BTreeMap, BTreeSet};

use ptr_protocol::Carrier;
use ptr_types::{DeviceId, NodeId, Timestamp};

use crate::placement::{DeviceHealth, NodeHealth, NodeRecord};

/// The most devices one node's carrier table covers. A host has a handful of cards; the bound
/// keeps the ordered-pair table (`n * (n - 1)` rows) small and a registration with more devices
/// is refused, not truncated.
pub const MAX_DEVICES_PER_NODE: usize = 16;

/// What this host measured for one ordered device pair: the cost of moving one MiB by each of
/// the two carriers that connect two GPUs. `measured_at` is the caller's own timestamp
/// (E011's measurement record), kept so a reader can tell how old the claim is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Calibration {
    pub device_peer_ns_per_mib: u64,
    pub pinned_host_ns_per_mib: u64,
    pub measured_at: Timestamp,
}

impl Calibration {
    /// Whether `DevicePeer` is strictly faster than `PinnedHost` on this host for the pair.
    /// Equal is not faster: the carrier with more copies and no peer dependency wins a tie.
    pub fn device_peer_is_faster(&self) -> bool {
        self.device_peer_ns_per_mib < self.pinned_host_ns_per_mib
    }
}

/// The recorded answer of the peer-access probe for one ordered device pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeerProbe {
    /// No probe was recorded (the initial state, and the state after a device is invalidated).
    Unprobed,
    /// The probe ran and peer access is not available.
    NoPeer,
    /// The probe ran and peer access is available. `calibrated` is this host's measurement for
    /// the pair, if one exists; peer access without it is not enough to choose `DevicePeer`.
    Peer { calibrated: Option<Calibration> },
}

/// Why a carrier table refused an update. Nothing is changed by a refused update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CarrierTableError {
    /// The node has more than [`MAX_DEVICES_PER_NODE`] devices.
    TooManyDevices { limit: usize },
    /// The node lists one device twice.
    DuplicateDevice(DeviceId),
    /// The probe names a device the table does not hold (never registered, failed or gone).
    UnknownDevice(DeviceId),
    /// A probe from a device to itself: the pair is ordered and distinct.
    SameDevice(DeviceId),
    /// `Unprobed` is the absence of a record, not a result; use
    /// [`CarrierTable::invalidate_device`] to forget one.
    NotAResult,
    /// A calibration with a zero cost, which no measurement produces.
    ZeroCalibration,
    /// The reconcile is for another node than the table's.
    WrongNode { expected: NodeId, found: NodeId },
}

/// The per-host table of device pairs: which pairs have peer access and what was measured.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CarrierTable {
    node: NodeId,
    /// The devices that are live (present and not failed) at the last registration or
    /// reconcile.
    devices: BTreeSet<DeviceId>,
    /// Ordered pairs (from, to) with a recorded result. A missing pair is `Unprobed`.
    pairs: BTreeMap<(DeviceId, DeviceId), PeerProbe>,
}

impl CarrierTable {
    /// The table of a node being registered: every non-failed device of the record, every pair
    /// `Unprobed`. A node with more than [`MAX_DEVICES_PER_NODE`] devices, or one that lists a
    /// device twice, is refused.
    pub fn register(node: &NodeRecord) -> Result<Self, CarrierTableError> {
        let devices = live_devices(node)?;
        Ok(Self {
            node: node.node_id.clone(),
            devices,
            pairs: BTreeMap::new(),
        })
    }

    pub fn node(&self) -> &NodeId {
        &self.node
    }

    /// Whether `device` is live in the table.
    pub fn has_device(&self, device: &DeviceId) -> bool {
        self.devices.contains(device)
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// The recorded probe of the ordered pair; `Unprobed` for a pair with no record or with a
    /// device the table does not hold.
    pub fn probe(&self, from: &DeviceId, to: &DeviceId) -> PeerProbe {
        if !self.devices.contains(from) || !self.devices.contains(to) {
            return PeerProbe::Unprobed;
        }
        self.pairs
            .get(&(from.clone(), to.clone()))
            .copied()
            .unwrap_or(PeerProbe::Unprobed)
    }

    /// Record the probe result for the ordered pair `from -> to` (replacing an earlier one).
    /// Both devices must be live in the table and distinct; `Unprobed` is not a result; a
    /// calibration must have non-zero costs. A refused record changes nothing.
    pub fn record_probe(
        &mut self,
        from: &DeviceId,
        to: &DeviceId,
        probe: PeerProbe,
    ) -> Result<(), CarrierTableError> {
        for device in [from, to] {
            if !self.devices.contains(device) {
                return Err(CarrierTableError::UnknownDevice(device.clone()));
            }
        }
        if from == to {
            return Err(CarrierTableError::SameDevice(from.clone()));
        }
        match probe {
            PeerProbe::Unprobed => return Err(CarrierTableError::NotAResult),
            PeerProbe::Peer {
                calibrated: Some(calibration),
            } if calibration.device_peer_ns_per_mib == 0
                || calibration.pinned_host_ns_per_mib == 0 =>
            {
                return Err(CarrierTableError::ZeroCalibration);
            }
            PeerProbe::NoPeer | PeerProbe::Peer { .. } => {}
        }
        self.pairs.insert((from.clone(), to.clone()), probe);
        Ok(())
    }

    /// Forget a device: it leaves the live set and every row it appears in is dropped, in both
    /// directions. A device that fails or vanishes (a placement revocation) must not keep a
    /// peer claim; if it comes back it is registered again and probed again. Returns whether the
    /// device was live.
    pub fn invalidate_device(&mut self, device: &DeviceId) -> bool {
        let was_live = self.devices.remove(device);
        self.pairs
            .retain(|(from, to), _| from != device && to != device);
        was_live
    }

    /// Bring the table in line with a node record after a heartbeat: a device that is absent
    /// from the record or reports `Failed` is invalidated; a device that is new is added with
    /// no probe (a device that was lost and is back is new). Devices that are still present keep
    /// their rows. Refused, with nothing changed, for another node's record or one that lists
    /// too many or duplicate devices.
    pub fn reconcile(&mut self, node: &NodeRecord) -> Result<(), CarrierTableError> {
        if node.node_id != self.node {
            return Err(CarrierTableError::WrongNode {
                expected: self.node.clone(),
                found: node.node_id.clone(),
            });
        }
        let now = live_devices(node)?;
        let gone: Vec<DeviceId> = self.devices.difference(&now).cloned().collect();
        for device in &gone {
            self.invalidate_device(device);
        }
        self.devices = now;
        Ok(())
    }
}

fn live_devices(node: &NodeRecord) -> Result<BTreeSet<DeviceId>, CarrierTableError> {
    // Shape errors are refused whatever the node's health.
    if node.devices.len() > MAX_DEVICES_PER_NODE {
        return Err(CarrierTableError::TooManyDevices {
            limit: MAX_DEVICES_PER_NODE,
        });
    }
    let mut seen = BTreeSet::new();
    let mut live = BTreeSet::new();
    for device in &node.devices {
        if !seen.insert(device.device_id.clone()) {
            return Err(CarrierTableError::DuplicateDevice(device.device_id.clone()));
        }
        match device.health {
            DeviceHealth::Failed => {}
            DeviceHealth::Healthy | DeviceHealth::Degraded => {
                live.insert(device.device_id.clone());
            }
        }
    }
    // A node that is not healthy has no live device: the placement controller revokes every
    // placement on a Failed node, and a Stale one is not trusted either.
    match node.health {
        NodeHealth::Healthy => Ok(live),
        NodeHealth::Stale | NodeHealth::Failed => Ok(BTreeSet::new()),
    }
}

/// The identity of an operating-system process on one host, drawn by the supervisor (the same
/// kind of token as a worker session, but about the process, not the spawn). Two endpoints are
/// "the same process" only if their node and this token are equal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HostProcess(pub u64);

/// How far an endpoint's worker is confined (doc 42 rule 3b). The shared carriers are permitted
/// only when both endpoints are `Full`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointConfinement {
    /// Namespaced, seccomp-filtered and cgroup-limited: no ambient way to reach what it was not
    /// handed.
    Full,
    /// Some of that confinement is missing (another OS, a kernel without seccomp or cgroup v2,
    /// a degraded start).
    Partial,
    /// A trusted in-tree process with no confinement.
    None,
}

/// Where the tensor is, for the producer, or where the consumer needs it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorLocation {
    /// Host (CPU) memory.
    Host,
    /// Memory of one GPU of the endpoint's node.
    Device(DeviceId),
}

/// What the selection needs to know about one end of a transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointCapabilities {
    pub node: NodeId,
    pub process: HostProcess,
    pub confinement: EndpointConfinement,
    pub tensor_location: TensorLocation,
}

/// Whether the producer hands over a tensor it will never touch again. The caller states it; it
/// is the claim behind a [`MovedTensor`](crate::MovedTensor), not a guess.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadOwnership {
    /// Unique ownership moves to the consumer (rule 3a): the producer keeps no reference.
    MovedUnique,
    /// The storage is shared (a candle tensor with shared storage, a DLPack capsule across a
    /// language boundary): the consumer needs a receiver-owned copy.
    Shared,
}

/// Why the selection chose its carrier. The reason is part of the decision record, so a test
/// or a log can say which row of the table applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionReason {
    /// Another node.
    CrossNode,
    /// Same process, same location, ownership moved.
    MovedInProcess,
    /// Same process, but the storage is shared: a receiver-owned copy.
    SharedStorageCopy,
    /// Another process, and an endpoint is not fully confined.
    ConfinementNotFull,
    /// Host memory to host memory, both fully confined.
    HostToHost,
    /// The same GPU, another process, both fully confined.
    SameGpu,
    /// Two GPUs, probed peer-capable and calibrated faster than the pinned host path.
    PeerCalibratedFaster,
    /// Two GPUs, peer-capable and calibrated, but not faster than the pinned host path.
    PeerNotFaster,
    /// Two GPUs, peer-capable, with no calibration for the pair yet.
    PeerNotCalibrated,
    /// Two GPUs, probed: no peer access.
    PeerUnavailable,
    /// Two GPUs, no probe recorded for the ordered pair.
    PeerUnprobed,
    /// Between host memory and a GPU, both fully confined.
    HostDeviceHop,
}

/// The chosen carrier and the row of the table that chose it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CarrierSelection {
    pub carrier: Carrier,
    pub reason: SelectionReason,
}

/// Why no carrier can be selected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CarrierRefusal {
    /// An endpoint's tensor location names a device that is not live in the table (never
    /// registered, failed or vanished). Selecting a carrier for it would route bytes to or from
    /// a device that is gone.
    DeviceUnavailable(DeviceId),
    /// Both endpoints are on one node, but the table is another node's.
    TableForOtherNode { expected: NodeId, found: NodeId },
}

/// Choose the carrier for one transfer from `producer` to `consumer`. Pure and deterministic;
/// see the module documentation for the order of the rules. `table` is the carrier table of the
/// node both endpoints are on; for endpoints on different nodes it is not consulted.
pub fn select_carrier(
    producer: &EndpointCapabilities,
    consumer: &EndpointCapabilities,
    ownership: PayloadOwnership,
    table: &CarrierTable,
) -> Result<CarrierSelection, CarrierRefusal> {
    if producer.node != consumer.node {
        return Ok(chosen(Carrier::Remote, SelectionReason::CrossNode));
    }
    if table.node() != &producer.node {
        return Err(CarrierRefusal::TableForOtherNode {
            expected: producer.node.clone(),
            found: table.node().clone(),
        });
    }
    for endpoint in [producer, consumer] {
        if let TensorLocation::Device(device) = &endpoint.tensor_location {
            if !table.has_device(device) {
                return Err(CarrierRefusal::DeviceUnavailable(device.clone()));
            }
        }
    }
    if producer.process == consumer.process && producer.tensor_location == consumer.tensor_location
    {
        return Ok(match ownership {
            PayloadOwnership::MovedUnique => {
                chosen(Carrier::InProcess, SelectionReason::MovedInProcess)
            }
            PayloadOwnership::Shared => chosen(Carrier::Bytes, SelectionReason::SharedStorageCopy),
        });
    }
    if !fully_confined(producer) || !fully_confined(consumer) {
        return Ok(chosen(Carrier::Bytes, SelectionReason::ConfinementNotFull));
    }
    Ok(
        match (&producer.tensor_location, &consumer.tensor_location) {
            (TensorLocation::Host, TensorLocation::Host) => {
                chosen(Carrier::HostShm, SelectionReason::HostToHost)
            }
            (TensorLocation::Host, TensorLocation::Device(_))
            | (TensorLocation::Device(_), TensorLocation::Host) => {
                chosen(Carrier::PinnedHost, SelectionReason::HostDeviceHop)
            }
            (TensorLocation::Device(from), TensorLocation::Device(to)) if from == to => {
                chosen(Carrier::CudaIpc, SelectionReason::SameGpu)
            }
            (TensorLocation::Device(from), TensorLocation::Device(to)) => {
                between_gpus(table.probe(from, to))
            }
        },
    )
}

fn between_gpus(probe: PeerProbe) -> CarrierSelection {
    match probe {
        PeerProbe::Unprobed => chosen(Carrier::PinnedHost, SelectionReason::PeerUnprobed),
        PeerProbe::NoPeer => chosen(Carrier::PinnedHost, SelectionReason::PeerUnavailable),
        PeerProbe::Peer { calibrated: None } => {
            chosen(Carrier::PinnedHost, SelectionReason::PeerNotCalibrated)
        }
        PeerProbe::Peer {
            calibrated: Some(calibration),
        } => {
            if calibration.device_peer_is_faster() {
                chosen(Carrier::DevicePeer, SelectionReason::PeerCalibratedFaster)
            } else {
                chosen(Carrier::PinnedHost, SelectionReason::PeerNotFaster)
            }
        }
    }
}

fn fully_confined(endpoint: &EndpointCapabilities) -> bool {
    match endpoint.confinement {
        EndpointConfinement::Full => true,
        EndpointConfinement::Partial | EndpointConfinement::None => false,
    }
}

fn chosen(carrier: Carrier, reason: SelectionReason) -> CarrierSelection {
    CarrierSelection { carrier, reason }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::DeviceRecord;

    fn device(name: &str, health: DeviceHealth) -> DeviceRecord {
        DeviceRecord {
            device_id: DeviceId::from(name),
            vram_bytes: 24,
            used_vram_bytes: 0,
            health,
        }
    }

    fn node(devices: &[(&str, DeviceHealth)]) -> NodeRecord {
        NodeRecord {
            node_id: NodeId::from("n"),
            zone: "z".into(),
            devices: devices.iter().map(|(n, h)| device(n, *h)).collect(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
        }
    }

    fn two_gpus() -> CarrierTable {
        CarrierTable::register(&node(&[
            ("cuda:0", DeviceHealth::Healthy),
            ("cuda:1", DeviceHealth::Healthy),
        ]))
        .unwrap()
    }

    fn calibration(peer: u64, pinned: u64) -> Calibration {
        Calibration {
            device_peer_ns_per_mib: peer,
            pinned_host_ns_per_mib: pinned,
            measured_at: Timestamp(5),
        }
    }

    fn at(
        process: u64,
        confinement: EndpointConfinement,
        location: TensorLocation,
    ) -> EndpointCapabilities {
        EndpointCapabilities {
            node: NodeId::from("n"),
            process: HostProcess(process),
            confinement,
            tensor_location: location,
        }
    }

    fn pick(
        producer: &EndpointCapabilities,
        consumer: &EndpointCapabilities,
        table: &CarrierTable,
    ) -> CarrierSelection {
        select_carrier(producer, consumer, PayloadOwnership::MovedUnique, table).unwrap()
    }

    const FULL: EndpointConfinement = EndpointConfinement::Full;

    #[test]
    fn the_table_is_bounded_and_refuses_what_it_cannot_hold() {
        let many: Vec<(String, DeviceHealth)> = (0..=MAX_DEVICES_PER_NODE)
            .map(|i| (format!("cuda:{i}"), DeviceHealth::Healthy))
            .collect();
        let refs: Vec<(&str, DeviceHealth)> = many.iter().map(|(n, h)| (n.as_str(), *h)).collect();
        assert_eq!(
            CarrierTable::register(&node(&refs)),
            Err(CarrierTableError::TooManyDevices {
                limit: MAX_DEVICES_PER_NODE
            })
        );
        assert_eq!(
            CarrierTable::register(&node(&[
                ("cuda:0", DeviceHealth::Healthy),
                ("cuda:0", DeviceHealth::Healthy)
            ])),
            Err(CarrierTableError::DuplicateDevice(DeviceId::from("cuda:0")))
        );
        let mut table = two_gpus();
        let (zero, one, ghost) = (
            DeviceId::from("cuda:0"),
            DeviceId::from("cuda:1"),
            DeviceId::from("cuda:9"),
        );
        assert_eq!(
            table.record_probe(&zero, &ghost, PeerProbe::NoPeer),
            Err(CarrierTableError::UnknownDevice(ghost.clone()))
        );
        assert_eq!(
            table.record_probe(&zero, &zero, PeerProbe::NoPeer),
            Err(CarrierTableError::SameDevice(zero.clone()))
        );
        assert_eq!(
            table.record_probe(&zero, &one, PeerProbe::Unprobed),
            Err(CarrierTableError::NotAResult)
        );
        assert_eq!(
            table.record_probe(
                &zero,
                &one,
                PeerProbe::Peer {
                    calibrated: Some(calibration(0, 5))
                }
            ),
            Err(CarrierTableError::ZeroCalibration)
        );
        assert_eq!(table.probe(&zero, &one), PeerProbe::Unprobed);
    }

    #[test]
    fn a_failed_device_is_not_live_and_a_probe_is_per_ordered_pair() {
        let mut table = CarrierTable::register(&node(&[
            ("cuda:0", DeviceHealth::Healthy),
            ("cuda:1", DeviceHealth::Degraded),
            ("cuda:2", DeviceHealth::Failed),
        ]))
        .unwrap();
        assert_eq!(table.device_count(), 2);
        assert!(!table.has_device(&DeviceId::from("cuda:2")));
        let (zero, one) = (DeviceId::from("cuda:0"), DeviceId::from("cuda:1"));
        table.record_probe(&zero, &one, PeerProbe::NoPeer).unwrap();
        assert_eq!(table.probe(&zero, &one), PeerProbe::NoPeer);
        assert_eq!(table.probe(&one, &zero), PeerProbe::Unprobed);
    }

    #[test]
    fn invalidating_a_device_drops_its_rows_in_both_directions() {
        let mut table = two_gpus();
        let (zero, one) = (DeviceId::from("cuda:0"), DeviceId::from("cuda:1"));
        let faster = PeerProbe::Peer {
            calibrated: Some(calibration(1, 2)),
        };
        table.record_probe(&zero, &one, faster).unwrap();
        table.record_probe(&one, &zero, faster).unwrap();
        assert!(table.invalidate_device(&one));
        assert!(!table.invalidate_device(&one));
        assert_eq!(table.probe(&zero, &one), PeerProbe::Unprobed);
        assert_eq!(table.probe(&one, &zero), PeerProbe::Unprobed);
        // A device that comes back is new: it is probed again before DevicePeer is chosen.
        table
            .reconcile(&node(&[
                ("cuda:0", DeviceHealth::Healthy),
                ("cuda:1", DeviceHealth::Healthy),
            ]))
            .unwrap();
        assert!(table.has_device(&one));
        assert_eq!(table.probe(&zero, &one), PeerProbe::Unprobed);
    }

    #[test]
    fn reconcile_keeps_the_rows_of_devices_that_are_still_there() {
        let mut table = CarrierTable::register(&node(&[
            ("cuda:0", DeviceHealth::Healthy),
            ("cuda:1", DeviceHealth::Healthy),
            ("cuda:2", DeviceHealth::Healthy),
        ]))
        .unwrap();
        let (zero, one, two) = (
            DeviceId::from("cuda:0"),
            DeviceId::from("cuda:1"),
            DeviceId::from("cuda:2"),
        );
        table.record_probe(&zero, &one, PeerProbe::NoPeer).unwrap();
        table.record_probe(&zero, &two, PeerProbe::NoPeer).unwrap();
        table
            .reconcile(&node(&[
                ("cuda:0", DeviceHealth::Healthy),
                ("cuda:1", DeviceHealth::Healthy),
                ("cuda:2", DeviceHealth::Failed),
            ]))
            .unwrap();
        assert_eq!(table.probe(&zero, &one), PeerProbe::NoPeer);
        assert_eq!(table.probe(&zero, &two), PeerProbe::Unprobed);
        let mut other = node(&[]);
        other.node_id = NodeId::from("other");
        assert_eq!(
            table.reconcile(&other),
            Err(CarrierTableError::WrongNode {
                expected: NodeId::from("n"),
                found: NodeId::from("other")
            })
        );
    }

    #[test]
    fn a_failed_or_stale_node_has_no_live_device() {
        for health in [NodeHealth::Failed, NodeHealth::Stale] {
            let mut record = node(&[
                ("cuda:0", DeviceHealth::Healthy),
                ("cuda:1", DeviceHealth::Healthy),
            ]);
            let mut table = CarrierTable::register(&record).unwrap();
            let (zero, one) = (DeviceId::from("cuda:0"), DeviceId::from("cuda:1"));
            table.record_probe(&zero, &one, PeerProbe::NoPeer).unwrap();
            record.health = health;
            table.reconcile(&record).unwrap();
            assert_eq!(table.device_count(), 0);
            assert_eq!(table.probe(&zero, &one), PeerProbe::Unprobed);
            assert_eq!(CarrierTable::register(&record).unwrap().device_count(), 0);
        }
    }

    #[test]
    fn calibration_equal_is_not_faster() {
        assert!(calibration(1, 2).device_peer_is_faster());
        assert!(!calibration(2, 2).device_peer_is_faster());
        assert!(!calibration(3, 2).device_peer_is_faster());
    }

    #[test]
    fn a_table_of_another_node_is_refused_on_one_node_and_ignored_across_nodes() {
        let table = two_gpus();
        let mut a = at(1, FULL, TensorLocation::Host);
        let mut b = at(2, FULL, TensorLocation::Host);
        a.node = NodeId::from("elsewhere");
        b.node = NodeId::from("elsewhere");
        assert_eq!(
            select_carrier(&a, &b, PayloadOwnership::MovedUnique, &table),
            Err(CarrierRefusal::TableForOtherNode {
                expected: NodeId::from("elsewhere"),
                found: NodeId::from("n")
            })
        );
        b.node = NodeId::from("third");
        assert_eq!(
            pick(&a, &b, &table),
            chosen(Carrier::Remote, SelectionReason::CrossNode)
        );
    }
}
