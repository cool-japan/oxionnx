//! Tests for the non-macOS stub (`stub_impl`).

use super::*;

/// Every operation on the non-macOS stub must return UnsupportedPlatform.
#[test]
fn stub_load_returns_unsupported_platform() {
    let r = MlPackageModel::load("anywhere", MlComputeUnits::All);
    assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
}

#[test]
fn stub_load_from_bytes_returns_unsupported_platform() {
    let r = MlPackageModel::load_from_bytes(&[], MlComputeUnits::All);
    assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
}
