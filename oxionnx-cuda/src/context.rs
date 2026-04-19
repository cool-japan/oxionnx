//! CUDA context wrapper for oxionnx-cuda.
//!
//! [`CudaContext`] holds a CUDA device context together with a [`DnnHandle`]
//! (which itself contains a `BlasHandle`, PTX cache, and stream).  A single
//! `CudaContext` is created once at `Session` build time and shared across all
//! op dispatches within a session run.

use std::sync::Arc;

use oxicuda_dnn::handle::DnnHandle;
use oxicuda_driver::{Context, Device};

/// Encapsulates the CUDA context and DNN handle used for accelerated dispatch.
///
/// Construction is fallible: if no CUDA device is available or initialisation
/// fails, `try_new` returns `None` so the caller can fall back to CPU/wgpu.
pub struct CudaContext {
    /// The underlying CUDA driver context.  Kept alive here so the context
    /// is not dropped while the DNN handle (and kernels it compiles) are in use.
    pub(crate) context: Arc<Context>,
    /// DNN handle (owns stream, BLAS handle, PTX cache, SM version).
    pub(crate) dnn: DnnHandle,
}

impl CudaContext {
    /// Return a reference to the underlying CUDA driver context.
    pub fn driver_context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Attempt to create a `CudaContext` for device 0.
    ///
    /// Returns `None` on any CUDA error (no GPU, driver not installed, etc.)
    /// so callers can degrade gracefully.
    pub fn try_new() -> Option<Self> {
        match oxicuda_driver::init() {
            Ok(()) => {}
            Err(_) => return None,
        }

        let dev = Device::get(0).ok()?;
        let context = Arc::new(Context::new(&dev).ok()?);

        // Activate the context on the current thread.
        context.set_current().ok()?;

        let dnn = DnnHandle::new(&context).ok()?;

        Some(Self { context, dnn })
    }
}
