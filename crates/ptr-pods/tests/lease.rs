use ptr_pods::{invoke_with_lease, LeaseInvokeError, Pod, PodLease, PodManifest, Ready};
use ptr_types::PodId;

struct Echo {
    manifest: PodManifest,
}

impl Pod for Echo {
    type Input = String;
    type Output = String;
    type Error = ();

    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: String) -> Result<Self::Output, Self::Error> {
        Ok(input)
    }
}

fn pod(id: &str) -> Echo {
    Echo {
        manifest: PodManifest {
            id: PodId::from(id),
            capabilities: vec![],
            accepts: vec![],
            produces: vec![],
            effects: vec![],
            protocol_version: 1,
        },
    }
}

#[test]
fn ready_lease_invokes_matching_pod() {
    let pod = pod("echo");
    let lease: PodLease<Ready> = PodLease::new(PodId::from("echo"));
    assert_eq!(
        invoke_with_lease(&pod, &lease, "hello".into()).unwrap(),
        "hello"
    );
}

#[test]
fn ready_lease_for_different_pod_is_rejected() {
    let pod = pod("echo");
    let lease: PodLease<Ready> = PodLease::new(PodId::from("other"));
    assert_eq!(
        invoke_with_lease(&pod, &lease, "hello".into()),
        Err(LeaseInvokeError::WrongPod)
    );
}
