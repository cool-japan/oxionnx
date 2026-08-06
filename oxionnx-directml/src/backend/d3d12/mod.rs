//! Raw Direct3D 12 plumbing: device, buffers, barriers, shaders, pipeline states.
//!
//! Compiled on Windows only.  Everything in here is thin FFI glue — no shape logic,
//! no dispatch math, no sizing.  Those live in [`crate::plan`] and [`crate::layout`],
//! which are compiled and tested on every platform.  If you find yourself computing a
//! dimension in this subtree, it belongs upstairs.
//!
//! Both engines share this layer: [`hlsl_backend::HlslEngine`] uses all of it, and
//! [`crate::backend::dml`] uses [`device::D3d12Core`], [`device::DescriptorHeap`] and
//! [`buffer::GpuBuffer`].

pub(crate) mod buffer;
pub(crate) mod device;
pub(crate) mod hlsl_backend;
pub(crate) mod pso;
pub(crate) mod shader;
