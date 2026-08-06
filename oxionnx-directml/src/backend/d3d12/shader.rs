//! `D3DCompile` — HLSL source text → DXBC bytecode, at run time.
//!
//! # Why compile at run time at all
//!
//! `d3dcompiler_47.dll` (FXC) is inbox on every supported Windows, needs no SDK, no
//! build-time toolchain and no external redistributable.  That is *precisely* why the
//! HLSL engine can serve as the fallback for a missing `DirectML.dll`: a backend whose
//! job is "work when the optional runtime is absent" cannot itself depend on an
//! optional runtime.
//!
//! # A failed compile is a **loud error**, never a silent decline
//!
//! [`crate::error::DirectMLError::Declined`] means "this backend cannot express this
//! node" and the router turns it into a correct CPU fallback.  A shader that does not
//! compile is not that — it is a *bug in this crate*, and it must not be laundered into
//! a decline, or the entire GPU path would quietly disappear on every user's machine
//! while the tests stayed green.  [`ShaderBlob::compile`] therefore returns
//! [`crate::error::DirectMLError::ShaderCompile`] carrying FXC's own error blob, which
//! is the only diagnostic anybody will ever get: nothing in this repository can parse
//! the HLSL, so `D3DCompile` on the user's machine is the first and only thing that
//! ever does.

use core::slice;

use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::Fxc::{
    D3DCompile, D3DCOMPILE_ENABLE_STRICTNESS, D3DCOMPILE_OPTIMIZATION_LEVEL3,
};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::D3D12_SHADER_BYTECODE;

use crate::error::{DirectMLError, Result};
use crate::hlsl::{
    ELEMENTWISE_BINARY_HLSL, ELEMENTWISE_UNARY_HLSL, MATMUL_HLSL, REDUCE_HLSL, SOFTMAX_HLSL,
};
use crate::plan::{BinaryOp, ReduceKind, UnaryOp};

/// The shader-model target handed to `D3DCompile`, NUL-terminated for [`PCSTR`].
///
/// **`cs_5_1`, not `cs_5_0`.**  Shader Model 5.1 is the D3D12-native model: FXC emits
/// DXBC for it, every D3D12 device supports it, and its resource-binding rules are the
/// ones [`super::pso`]'s root signature is written against — `register(t0)` in the HLSL
/// means *(shader register 0, register space 0)*, which is exactly what a root SRV with
/// `ShaderRegister: 0, RegisterSpace: 0` binds.
///
/// Shader Model 6.x would mean DXIL and `dxcompiler.dll`, which is **not** inbox on
/// Windows.  Depending on it would defeat the whole purpose of this engine.
const SHADER_TARGET: &[u8] = b"cs_5_1\0";

/// `D3DCompile`'s `Flags1`.
///
/// * `ENABLE_STRICTNESS` rejects the D3D9-era legacy syntax FXC otherwise still
///   tolerates.  The kernels in [`crate::hlsl`] use none of it, so this flag can only
///   ever fire on a genuine mistake.
/// * `OPTIMIZATION_LEVEL3` — each shader is compiled once per process and then runs
///   over every matching node of every model.  There is no reason to leave optimisation
///   on the table for a one-off cost.
///
/// Deliberately **not** `D3DCOMPILE_WARNINGS_ARE_ERRORS`.  We cannot compile these
/// sources anywhere in this repository, so we cannot know whether some FXC version
/// emits a benign warning for them — and if one did, warnings-as-errors would take the
/// entire GPU backend offline on a user's machine over a cosmetic diagnostic.  Warnings
/// are surfaced (see [`ShaderBlob::compile`]) but are not fatal.
const COMPILE_FLAGS: u32 = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;

/// The compute entry points this crate compiles.
///
/// One `ShaderKind` ↔ one `(source, entry_point)` pair ↔ one PSO in
/// [`super::pso::PsoCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ShaderKind {
    /// `main` in [`crate::hlsl::MATMUL_HLSL`].
    MatMul,
    /// `main_add` in [`crate::hlsl::ELEMENTWISE_BINARY_HLSL`].
    Add,
    /// `main_sub` in [`crate::hlsl::ELEMENTWISE_BINARY_HLSL`].
    Sub,
    /// `main_mul` in [`crate::hlsl::ELEMENTWISE_BINARY_HLSL`].
    Mul,
    /// `main_div` in [`crate::hlsl::ELEMENTWISE_BINARY_HLSL`].
    Div,
    /// `main_relu` in [`crate::hlsl::ELEMENTWISE_UNARY_HLSL`].
    Relu,
    /// `main_sigmoid` in [`crate::hlsl::ELEMENTWISE_UNARY_HLSL`].
    Sigmoid,
    /// `main_tanh` in [`crate::hlsl::ELEMENTWISE_UNARY_HLSL`].
    Tanh,
    /// `main` in [`crate::hlsl::SOFTMAX_HLSL`].
    Softmax,
    /// `main_sum` in [`crate::hlsl::REDUCE_HLSL`].
    ReduceSum,
    /// `main_mean` in [`crate::hlsl::REDUCE_HLSL`].
    ReduceMean,
    /// `main_max` in [`crate::hlsl::REDUCE_HLSL`].
    ReduceMax,
    /// `main_min` in [`crate::hlsl::REDUCE_HLSL`].
    ReduceMin,
}

impl ShaderKind {
    /// The [`crate::hlsl`] constant this entry point lives in.
    pub(crate) fn source(self) -> &'static str {
        match self {
            Self::MatMul => MATMUL_HLSL,
            Self::Add | Self::Sub | Self::Mul | Self::Div => ELEMENTWISE_BINARY_HLSL,
            Self::Relu | Self::Sigmoid | Self::Tanh => ELEMENTWISE_UNARY_HLSL,
            Self::Softmax => SOFTMAX_HLSL,
            Self::ReduceSum | Self::ReduceMean | Self::ReduceMax | Self::ReduceMin => REDUCE_HLSL,
        }
    }

    /// NUL-terminated entry-point name, ready for a [`PCSTR`].
    ///
    /// The trailing `\0` is load-bearing: `PCSTR` is a C string, and FXC reads until the
    /// terminator.  [`Self::entry_point_name`] derives the Rust-side name from *this*
    /// literal rather than repeating it, so the two cannot drift apart.
    pub(crate) fn entry_point(self) -> &'static [u8] {
        match self {
            Self::MatMul => b"main\0",
            Self::Add => b"main_add\0",
            Self::Sub => b"main_sub\0",
            Self::Mul => b"main_mul\0",
            Self::Div => b"main_div\0",
            Self::Relu => b"main_relu\0",
            Self::Sigmoid => b"main_sigmoid\0",
            Self::Tanh => b"main_tanh\0",
            // Softmax's entry point is `main`, deliberately the same *name* as MatMul's —
            // they live in different sources, so the `(source, entry_point)` pair is still
            // unique (see `each_kind_maps_to_a_distinct_entry_point`).
            Self::Softmax => b"main\0",
            Self::ReduceSum => b"main_sum\0",
            Self::ReduceMean => b"main_mean\0",
            Self::ReduceMax => b"main_max\0",
            Self::ReduceMin => b"main_min\0",
        }
    }

    /// The entry-point name **without** its NUL terminator, for diagnostics.
    ///
    /// Derived from [`Self::entry_point`], so there is exactly one source of truth for
    /// the eight names.
    pub(crate) fn entry_point_name(self) -> &'static str {
        let bytes = self.entry_point();
        let without_nul = bytes.split_last().map_or(bytes, |(_, head)| head);
        // The literals in `entry_point` are ASCII, so this never fails; falling back to
        // a marker string rather than unwrapping keeps the no-unwrap policy intact and
        // keeps a diagnostic path from ever being the thing that panics.
        core::str::from_utf8(without_nul).unwrap_or("<non-utf8 entry point>")
    }

    /// A pseudo-filename for `D3DCompile`'s `pSourceName`, NUL-terminated.
    ///
    /// FXC prefixes every diagnostic with this, so `hlsl::MATMUL_HLSL(12,9): error X3004`
    /// beats `(12,9): error X3004` when the error blob is all you have.
    pub(crate) fn source_name(self) -> &'static [u8] {
        match self {
            Self::MatMul => b"hlsl::MATMUL_HLSL\0",
            Self::Add | Self::Sub | Self::Mul | Self::Div => b"hlsl::ELEMENTWISE_BINARY_HLSL\0",
            Self::Relu | Self::Sigmoid | Self::Tanh => b"hlsl::ELEMENTWISE_UNARY_HLSL\0",
            Self::Softmax => b"hlsl::SOFTMAX_HLSL\0",
            Self::ReduceSum | Self::ReduceMean | Self::ReduceMax | Self::ReduceMin => {
                b"hlsl::REDUCE_HLSL\0"
            }
        }
    }

    /// Stable tag for logs and error messages.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::MatMul => "MatMul",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Relu => "Relu",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            // These MUST equal `ReduceKind::as_str()` (and `"Softmax"`), so a
            // `ComparisonReport`'s diagnostic tag names the same op the plan and the ONNX
            // node do.  `shader_kind_tags_match_reduce_kind_tags` pins that join.
            Self::Softmax => "Softmax",
            Self::ReduceSum => "ReduceSum",
            Self::ReduceMean => "ReduceMean",
            Self::ReduceMax => "ReduceMax",
            Self::ReduceMin => "ReduceMin",
        }
    }

    /// The entry point implementing `op`.
    pub(crate) fn for_binary(op: BinaryOp) -> Self {
        match op {
            BinaryOp::Add => Self::Add,
            BinaryOp::Sub => Self::Sub,
            BinaryOp::Mul => Self::Mul,
            BinaryOp::Div => Self::Div,
        }
    }

    /// The entry point implementing `op`.
    pub(crate) fn for_unary(op: UnaryOp) -> Self {
        match op {
            UnaryOp::Relu => Self::Relu,
            UnaryOp::Sigmoid => Self::Sigmoid,
            UnaryOp::Tanh => Self::Tanh,
        }
    }

    /// The [`REDUCE_HLSL`] entry point implementing `kind`.
    ///
    /// Mirrors [`Self::for_binary`] / [`Self::for_unary`]: the one per-op difference
    /// between the four reductions is which `main_*` entry point they compile, and this is
    /// where that choice is made.
    pub(crate) fn for_reduce(kind: ReduceKind) -> Self {
        match kind {
            ReduceKind::Sum => Self::ReduceSum,
            ReduceKind::Mean => Self::ReduceMean,
            ReduceKind::Max => Self::ReduceMax,
            ReduceKind::Min => Self::ReduceMin,
        }
    }
}

/// Borrow an [`ID3DBlob`]'s payload as bytes.
///
/// Safe by construction: the returned slice's lifetime is tied to `&blob`, and an
/// `ID3DBlob` is an immutable container — D3D fills it before handing it back and never
/// reallocates it afterwards — so the pointer stays valid for exactly as long as the
/// borrow.
pub(crate) fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: `GetBufferPointer` returns the base of the blob's own heap allocation and
    // `GetBufferSize` its length in bytes; the two are consistent by the `ID3DBlob`
    // contract.  Both are plain vtable calls that mutate nothing.  We hand back a slice
    // whose lifetime is bound to `&blob`, so it cannot outlive the allocation, and a
    // null pointer or zero length yields an *empty* slice rather than a dangling one —
    // `from_raw_parts` requires a non-null, aligned pointer even for `len == 0`.
    // Alignment is trivially satisfied: `u8` has alignment 1.
    unsafe {
        let ptr: *const u8 = blob.GetBufferPointer().cast();
        let len = blob.GetBufferSize();
        if ptr.is_null() || len == 0 {
            &[]
        } else {
            slice::from_raw_parts(ptr, len)
        }
    }
}

/// Render a diagnostic blob (`D3DCompile`'s or `D3D12SerializeRootSignature`'s) as text.
///
/// FXC writes NUL-terminated ASCII, so the terminator and any trailing newline are
/// trimmed.  A missing or empty blob yields a marker string rather than `""`, because
/// "the compiler failed and said nothing" is itself worth reporting.
pub(crate) fn blob_text(blob: Option<&ID3DBlob>) -> String {
    let Some(blob) = blob else {
        return "<no diagnostic blob>".to_owned();
    };
    let bytes = blob_bytes(blob);
    if bytes.is_empty() {
        return "<empty diagnostic blob>".to_owned();
    }
    let text = String::from_utf8_lossy(bytes);
    text.trim_end_matches('\0').trim_end().to_owned()
}

/// A compiled DXBC blob.
///
/// Owns the `ID3DBlob` that `D3DCompile` produced; [`Self::bytecode`] hands out a
/// `D3D12_SHADER_BYTECODE` that *borrows* it.
pub(crate) struct ShaderBlob(ID3DBlob);

impl ShaderBlob {
    /// Compile `kind`'s entry point out of its [`crate::hlsl`] source, targeting
    /// [`SHADER_TARGET`].
    ///
    /// # Errors
    ///
    /// [`DirectMLError::ShaderCompile`], carrying FXC's error blob verbatim.  This is a
    /// **hard error, not a decline**: the HLSL in this crate is a compile-time constant,
    /// so if it does not compile, that is a bug here and it must be reported loudly
    /// rather than laundered into a silent CPU fallback.
    pub(crate) fn compile(kind: ShaderKind) -> Result<Self> {
        let source = kind.source();
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;

        // SAFETY: `D3DCompile` is a synchronous, re-entrant call into
        // `d3dcompiler_47.dll` (inbox on every supported Windows; this is a normal
        // `raw-dylib` import, and the DLL is always present, so unlike `DirectML.dll` it
        // needs no `GetProcAddress` dance).
        //
        // * `source.as_ptr()` / `source.len()` describe a `&'static str` — a live,
        //   correctly-sized read-only buffer for the whole call.  FXC only reads it.
        // * `source_name` / `entry_point` / `SHADER_TARGET` are `&'static [u8]` literals
        //   that each end in an explicit NUL, satisfying `PCSTR`'s C-string contract.
        //   `entry_point_is_nul_terminated` in this module's tests pins that invariant.
        // * `pDefines` and `pInclude` are `None`: the sources have no `#define` and no
        //   `#include`, so FXC never needs to call back into us.
        // * `code` and `errors` are out-parameters.  `D3DCompile` writes an *owned* COM
        //   pointer into each (or leaves it null), and `Option<ID3DBlob>` releases it on
        //   drop — so neither leaks, on either the success or the failure path.
        let compiled = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                PCSTR(kind.source_name().as_ptr()),
                None,
                None,
                PCSTR(kind.entry_point().as_ptr()),
                PCSTR(SHADER_TARGET.as_ptr()),
                COMPILE_FLAGS,
                0,
                &mut code,
                Some(&mut errors),
            )
        };

        if let Err(e) = compiled {
            return Err(DirectMLError::ShaderCompile(format!(
                "{} ({}, entry `{}`, target `{}`) failed: {} [{}]",
                kind.as_str(),
                nul_terminated_str(kind.source_name()),
                kind.entry_point_name(),
                nul_terminated_str(SHADER_TARGET),
                blob_text(errors.as_ref()),
                e.message(),
            )));
        }

        // A success with no code blob would mean FXC returned `S_OK` and produced
        // nothing.  That should be impossible — but `code` is an `Option`, and treating
        // `None` as "fine" here would hand a null `pShaderBytecode` to
        // `CreateComputePipelineState` and crash the driver instead of the process.
        let Some(code) = code else {
            return Err(DirectMLError::ShaderCompile(format!(
                "{} (entry `{}`): D3DCompile reported success but produced no bytecode",
                kind.as_str(),
                kind.entry_point_name(),
            )));
        };

        Ok(Self(code))
    }

    /// A `D3D12_SHADER_BYTECODE` pointing into this blob.
    ///
    /// The returned struct holds a raw pointer with **no lifetime**, so `self` must
    /// outlive every use of it.  It is only ever built inside
    /// [`super::pso::PsoCache`], where the blob is a local that outlives the
    /// `CreateComputePipelineState` call it is passed to.
    pub(crate) fn bytecode(&self) -> D3D12_SHADER_BYTECODE {
        let bytes = blob_bytes(&self.0);
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }
}

/// Strip the trailing NUL from one of this module's `&'static [u8]` literals.
///
/// Diagnostics only; a malformed literal yields a marker rather than a panic.
fn nul_terminated_str(bytes: &'static [u8]) -> &'static str {
    let without_nul = bytes.split_last().map_or(bytes, |(_, head)| head);
    core::str::from_utf8(without_nul).unwrap_or("<non-utf8>")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{ShaderKind, SHADER_TARGET};
    use crate::plan::{BinaryOp, ReduceKind, UnaryOp};

    /// Every `ShaderKind`, so no test can silently skip a variant.
    const ALL: [ShaderKind; 13] = [
        ShaderKind::MatMul,
        ShaderKind::Add,
        ShaderKind::Sub,
        ShaderKind::Mul,
        ShaderKind::Div,
        ShaderKind::Relu,
        ShaderKind::Sigmoid,
        ShaderKind::Tanh,
        ShaderKind::Softmax,
        ShaderKind::ReduceSum,
        ShaderKind::ReduceMean,
        ShaderKind::ReduceMax,
        ShaderKind::ReduceMin,
    ];

    #[test]
    fn entry_point_is_nul_terminated_and_has_no_interior_nul() {
        // `PCSTR` reads until the first NUL.  A missing terminator runs off the end of
        // the literal; an interior one truncates the name to something FXC will not find.
        for kind in ALL {
            let bytes = kind.entry_point();
            assert_eq!(
                bytes.last(),
                Some(&0),
                "{} must be NUL-terminated",
                kind.as_str()
            );
            assert!(
                !bytes[..bytes.len() - 1].contains(&0),
                "{} has an interior NUL",
                kind.as_str()
            );

            let name = kind.source_name();
            assert_eq!(
                name.last(),
                Some(&0),
                "{} source name must be NUL-terminated",
                kind.as_str()
            );
        }
        assert_eq!(
            SHADER_TARGET.last(),
            Some(&0),
            "the shader target must be NUL-terminated"
        );
    }

    #[test]
    fn every_entry_point_exists_in_the_source_it_is_compiled_from() {
        // The single most likely way to break this file: point a `ShaderKind` at the
        // wrong `hlsl` constant.  FXC would fail with "entry point not found" at run
        // time, on a user's machine, with no test here having said a word.
        for kind in ALL {
            let needle = format!("void {}(", kind.entry_point_name());
            assert!(
                kind.source().contains(&needle),
                "{}: `{}` is not defined in the source it is compiled from",
                kind.as_str(),
                kind.entry_point_name()
            );
        }
    }

    #[test]
    fn entry_point_name_is_derived_from_the_nul_terminated_literal() {
        assert_eq!(ShaderKind::MatMul.entry_point_name(), "main");
        assert_eq!(ShaderKind::Add.entry_point_name(), "main_add");
        assert_eq!(ShaderKind::Sigmoid.entry_point_name(), "main_sigmoid");
        for kind in ALL {
            assert_eq!(
                kind.entry_point_name().len() + 1,
                kind.entry_point().len(),
                "{}: name and NUL-terminated literal disagree",
                kind.as_str()
            );
        }
    }

    #[test]
    fn each_kind_maps_to_a_distinct_entry_point() {
        // The real invariant is "one ShaderKind ↔ one (source, entry_point) pair ↔ one
        // PSO": two kinds resolving to the *same* pair would silently give one op the
        // other's maths and share a cache slot.  The entry-point *name* alone is NOT
        // globally unique — `Softmax` and `MatMul` both compile an entry point literally
        // called `main`, out of different sources — so uniqueness must be asserted on the
        // pair, which is what actually keys a compiled PSO.  This is not a weakening: the
        // pair-uniqueness property is the one that a real bug (two kinds pointing at the
        // same source *and* entry point) would violate.
        let mut seen: Vec<(&[u8], &str)> = ALL
            .iter()
            .map(|k| (k.source_name(), k.entry_point_name()))
            .collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(
            before,
            seen.len(),
            "two ShaderKinds share a (source, entry_point) pair"
        );

        // And the collision the pair uniqueness *tolerates* is real and intentional:
        // `main` genuinely appears under two different kinds, which is exactly why the
        // name alone can no longer be the key.
        let mains: Vec<ShaderKind> = ALL
            .iter()
            .copied()
            .filter(|k| k.entry_point_name() == "main")
            .collect();
        assert_eq!(
            mains,
            vec![ShaderKind::MatMul, ShaderKind::Softmax],
            "exactly MatMul and Softmax share the entry-point name `main`, from distinct sources"
        );
    }

    #[test]
    fn op_enums_map_onto_the_matching_entry_points() {
        assert_eq!(ShaderKind::for_binary(BinaryOp::Add), ShaderKind::Add);
        assert_eq!(ShaderKind::for_binary(BinaryOp::Sub), ShaderKind::Sub);
        assert_eq!(ShaderKind::for_binary(BinaryOp::Mul), ShaderKind::Mul);
        assert_eq!(ShaderKind::for_binary(BinaryOp::Div), ShaderKind::Div);
        assert_eq!(ShaderKind::for_unary(UnaryOp::Relu), ShaderKind::Relu);
        assert_eq!(ShaderKind::for_unary(UnaryOp::Sigmoid), ShaderKind::Sigmoid);
        assert_eq!(ShaderKind::for_unary(UnaryOp::Tanh), ShaderKind::Tanh);

        // …and the entry point's *name* carries the op's name, which is the property
        // that would actually break if the match arms above were transposed.
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let kind = ShaderKind::for_binary(op);
            assert_eq!(kind.as_str(), op.as_str());
            assert_eq!(
                kind.entry_point_name(),
                format!("main_{}", op.as_str().to_lowercase())
            );
        }
        for op in [UnaryOp::Relu, UnaryOp::Sigmoid, UnaryOp::Tanh] {
            let kind = ShaderKind::for_unary(op);
            assert_eq!(kind.as_str(), op.as_str());
            assert_eq!(
                kind.entry_point_name(),
                format!("main_{}", op.as_str().to_lowercase())
            );
        }
    }

    #[test]
    fn reduce_kinds_map_onto_the_matching_entry_points() {
        // `for_reduce` is the single place the four reductions pick their `main_*` entry
        // point; a transposed arm would give `ReduceMax` the `min` shader, which is a
        // silently wrong answer no shape check would catch.
        assert_eq!(
            ShaderKind::for_reduce(ReduceKind::Sum),
            ShaderKind::ReduceSum
        );
        assert_eq!(
            ShaderKind::for_reduce(ReduceKind::Mean),
            ShaderKind::ReduceMean
        );
        assert_eq!(
            ShaderKind::for_reduce(ReduceKind::Max),
            ShaderKind::ReduceMax
        );
        assert_eq!(
            ShaderKind::for_reduce(ReduceKind::Min),
            ShaderKind::ReduceMin
        );
    }

    #[test]
    fn shader_kind_tags_match_reduce_kind_tags() {
        // The diagnostic tag a `ShaderKind` carries must equal the one the *plan*'s
        // `ReduceKind` carries, or a `ComparisonReport` would name a different op than the
        // node it describes.  `Softmax` has no `ReduceKind`, but its tag is pinned too.
        for kind in [
            ReduceKind::Sum,
            ReduceKind::Mean,
            ReduceKind::Max,
            ReduceKind::Min,
        ] {
            assert_eq!(
                ShaderKind::for_reduce(kind).as_str(),
                kind.as_str(),
                "the shader tag must equal the reduce-kind tag"
            );
        }
        assert_eq!(ShaderKind::Softmax.as_str(), "Softmax");
    }

    #[test]
    fn the_target_is_the_d3d12_native_shader_model() {
        // `cs_5_0` would also compile, but SM 5.1 is the model whose binding rules the
        // root signature in `super::pso` is written against.  SM 6.x would need DXC.
        assert_eq!(SHADER_TARGET, b"cs_5_1\0");
    }
}
