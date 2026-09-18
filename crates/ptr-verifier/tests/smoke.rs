use ptr_verifier::VerificationStatus;
#[test] fn unknown_is_first_class(){ assert_eq!(VerificationStatus::Unknown,VerificationStatus::Unknown); }
