use ptr_ingress::{cross_check,CrossCheckStatus};
#[test] fn clean_raw_input_is_sample_verified(){ assert_eq!(cross_check(true,0),CrossCheckStatus::SampleVerified); }
