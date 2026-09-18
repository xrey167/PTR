use ptr_pods::{PodLease, Ready};
use ptr_types::PodId;
#[test]
fn revoked_lease_cannot_invoke() {
    let ready: PodLease<Ready> = PodLease::new(PodId::from("p"));
    let revoked = ready.revoke();
    assert!(!revoked.can_invoke());
}
