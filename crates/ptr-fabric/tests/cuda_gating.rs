//! The `cuda-ipc` feature gate. Runs with and without the feature: the flag must say whether the
//! module is compiled, so a build that did not ask for CUDA never carries it.

#[test]
fn the_compiled_flag_follows_the_feature_and_the_target() {
    assert_eq!(
        ptr_fabric::CUDA_IPC_COMPILED,
        cfg!(all(feature = "cuda-ipc", target_os = "linux"))
    );
}
