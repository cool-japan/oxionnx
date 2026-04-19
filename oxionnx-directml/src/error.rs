use thiserror::Error;

/// Errors specific to the DirectML execution provider.
#[derive(Debug, Error)]
pub enum DirectMLError {
    /// D3D12 device initialization failed.
    #[error("D3D12 device initialization failed: {0}")]
    DeviceInitFailed(String),
    /// A DirectML operator is not supported by this provider.
    #[error("DirectML operator not supported: {0}")]
    UnsupportedOp(String),
    /// A DirectML GPU dispatch failed.
    #[error("DirectML dispatch error: {0}")]
    DispatchFailed(String),
    /// A buffer transfer between CPU and GPU failed.
    #[error("Buffer transfer error: {0}")]
    TransferError(String),
}
