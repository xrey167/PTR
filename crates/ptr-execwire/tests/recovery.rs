use ptr_execwire::{
    recover_uncertain_receipt, uncertain_request, RecoveryBindingError, WireOutcome, WireReceipt,
};
use ptr_types::{ScopeId, StatefulRequestRecovery, UncertainRequest};

#[derive(Default)]
struct RecoveryProbe {
    requests: Vec<UncertainRequest>,
}

impl StatefulRequestRecovery for RecoveryProbe {
    type Error = &'static str;

    fn recover_uncertain(&mut self, request: UncertainRequest) -> Result<(), Self::Error> {
        self.requests.push(request);
        Ok(())
    }
}

fn receipt(outcome: WireOutcome) -> WireReceipt {
    WireReceipt {
        responder: "peer".to_owned(),
        request_id: 42,
        request_digest: [7; 32],
        outcome,
    }
}

fn scope() -> ScopeId {
    ScopeId("scope-recovery".to_owned())
}

#[test]
fn uncertain_receipt_maps_to_shared_recovery_contract() {
    let scope = scope();
    let request = uncertain_request(&receipt(WireOutcome::Uncertain), scope.clone()).unwrap();
    assert_eq!(
        request,
        UncertainRequest {
            request_id: 42,
            scope_id: scope
        }
    );
}

#[test]
fn uncertain_receipt_is_forwarded_without_retry_or_transport_mutation() {
    let scope = scope();
    let mut recovery = RecoveryProbe::default();
    let result = recover_uncertain_receipt(&mut recovery, &receipt(WireOutcome::Uncertain), scope);
    assert!(result.is_ok());
    assert_eq!(recovery.requests.len(), 1);
    assert_eq!(recovery.requests[0].request_id, 42);
}

#[test]
fn non_uncertain_receipts_cannot_enter_recovery() {
    let scope = scope();
    let error = uncertain_request(
        &receipt(WireOutcome::Refused {
            code: ptr_execwire::RefusalCode::Runtime,
        }),
        scope,
    )
    .unwrap_err();
    assert_eq!(error, RecoveryBindingError::NotUncertain);
}
