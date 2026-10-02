use ptr_pods::{DeviceLease, DeviceLeaseError, DeviceLeaseState};
use ptr_types::DeviceId;

#[test]
fn device_release_waits_for_active_tensor_users() {
    let mut lease = DeviceLease::new(DeviceId::from("cuda:0"), 7, 3, 11);
    lease.begin_tensor().unwrap();
    assert_eq!(lease.release(), Err(DeviceLeaseError::Busy));
    lease.end_tensor().unwrap();
    lease.release().unwrap();
    assert_eq!(lease.state, DeviceLeaseState::Released);
    assert_eq!(lease.release(), Ok(()));
}

#[test]
fn revoked_and_failed_leases_cannot_reacquire_tensors() {
    let mut revoked = DeviceLease::new(DeviceId::from("cuda:0"), 1, 1, 1);
    revoked.revoke();
    assert_eq!(revoked.begin_tensor(), Err(DeviceLeaseError::Revoked));
    assert_eq!(revoked.release(), Err(DeviceLeaseError::Revoked));

    let mut failed = DeviceLease::new(DeviceId::from("cuda:0"), 1, 1, 1);
    failed.abort();
    assert_eq!(failed.begin_tensor(), Err(DeviceLeaseError::Failed));
    assert_eq!(failed.release(), Err(DeviceLeaseError::Failed));
}

#[test]
fn tensor_release_is_balanced() {
    let mut lease = DeviceLease::new(DeviceId::from("cuda:0"), 1, 1, 1);
    assert_eq!(
        lease.end_tensor(),
        Err(DeviceLeaseError::InvalidTensorRelease)
    );
    lease.begin_tensor().unwrap();
    lease.end_tensor().unwrap();
    assert_eq!(
        lease.end_tensor(),
        Err(DeviceLeaseError::InvalidTensorRelease)
    );
}
