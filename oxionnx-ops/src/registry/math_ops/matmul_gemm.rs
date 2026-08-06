//! MatMul and Gemm operator implementations.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::math;

// ── MatMul ──────────────────────────────────────────────────────────────────

pub struct MatMulOp;
impl Operator for MatMulOp {
    fn op_type(&self) -> &str {
        "MatMul"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        Ok(vec![math::matmul(a, b)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        let slot = slots
            .first_mut()
            .ok_or_else(|| OnnxError::InvalidModel("MatMul: no output slot provided".into()))?;
        let out_shape =
            math::matmul_into(a, b, &mut slot.data).map_err(OnnxError::ShapeMismatch)?;
        slot.shape = out_shape;
        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I8,
            oxionnx_core::DType::I32,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let a = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("MatMul: missing input[0]".into()))?;
        let b = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("MatMul: missing input[1]".into()))?;

        match (&a.storage, &b.storage) {
            // ── F32: borrow both operands directly and call the shared
            // sgemm-backed kernel — no clone of either operand (in
            // particular no clone of B, which is normally the layer
            // weight; see `crate::math_typed::matmul_f32`'s doc comment). ──
            (TensorStorage::F32(a_data), TensorStorage::F32(b_data)) => {
                let (out_data, out_shape) =
                    crate::math_typed::matmul_f32(a_data, &a.shape, b_data, &b.shape)
                        .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::F32(out_data),
                    out_shape,
                )])
            }

            // ── I8 × I8 → I32 (quantized accumulate) ──
            (TensorStorage::I8(a_data), TensorStorage::I8(b_data)) => {
                let (out_data, out_shape) =
                    crate::math_typed::matmul_i8_i32(a_data, &a.shape, b_data, &b.shape)
                        .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::I32(out_data),
                    out_shape,
                )])
            }

            // ── I32 × I32 → I32 ──
            (TensorStorage::I32(a_data), TensorStorage::I32(b_data)) => {
                let (out_data, out_shape) =
                    crate::math_typed::matmul_i32(a_data, &a.shape, b_data, &b.shape)
                        .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::I32(out_data),
                    out_shape,
                )])
            }

            // ── F16 × F16 → F16 (accumulate in f32) ──
            (TensorStorage::F16(a_data), TensorStorage::F16(b_data)) => {
                let (out_data, out_shape) =
                    crate::math_typed::matmul_f16(a_data, &a.shape, b_data, &b.shape)
                        .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::F16(out_data),
                    out_shape,
                )])
            }

            // ── BF16 × BF16 → BF16 (accumulate in f32) ──
            (TensorStorage::BF16(a_data), TensorStorage::BF16(b_data)) => {
                let (out_data, out_shape) =
                    crate::math_typed::matmul_bf16(a_data, &a.shape, b_data, &b.shape)
                        .map_err(OnnxError::ShapeMismatch)?;
                Ok(vec![TypedTensor::new(
                    TensorStorage::BF16(out_data),
                    out_shape,
                )])
            }

            // ── Mixed dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}

// ── Gemm ────────────────────────────────────────────────────────────────────

pub struct GemmOp;
impl Operator for GemmOp {
    fn op_type(&self) -> &str {
        "Gemm"
    }
    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        let c = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let alpha = attrs.f("alpha", 1.0);
        let beta = attrs.f("beta", 1.0);
        let trans_a = attrs.i("transA", 0) != 0;
        let trans_b = attrs.i("transB", 0) != 0;
        Ok(vec![math::gemm(a, b, c, alpha, beta, trans_a, trans_b)?])
    }
    fn supports_output_slots(&self) -> bool {
        true
    }

    fn execute_into_slots(
        &self,
        ctx: &OpContext<'_>,
        slots: &mut [Tensor],
    ) -> Result<(), OnnxError> {
        let a = ctx.input(0)?;
        let b = ctx.input(1)?;
        let c = ctx.optional_input(2);
        let attrs = ctx.attrs();
        let alpha = attrs.f("alpha", 1.0_f32);
        let beta = attrs.f("beta", 1.0_f32);
        let trans_a = attrs.i("transA", 0) != 0;
        let trans_b = attrs.i("transB", 0) != 0;
        let slot = slots
            .first_mut()
            .ok_or_else(|| OnnxError::InvalidModel("Gemm: no output slot provided".into()))?;
        let out_shape = math::gemm_into(a, b, c, alpha, beta, trans_a, trans_b, &mut slot.data)
            .map_err(OnnxError::ShapeMismatch)?;
        slot.shape = out_shape;
        Ok(())
    }

    fn native_dtypes(&self) -> &'static [oxionnx_core::DType] {
        &[
            oxionnx_core::DType::F32,
            oxionnx_core::DType::F16,
            oxionnx_core::DType::BF16,
            oxionnx_core::DType::I8,
            oxionnx_core::DType::I32,
        ]
    }

    fn execute_typed(
        &self,
        ctx: &oxionnx_core::TypedOpContext<'_>,
    ) -> Result<Vec<oxionnx_core::TypedTensor>, oxionnx_core::OnnxError> {
        use oxionnx_core::dtype::TensorStorage;
        use oxionnx_core::{OnnxError, TypedTensor};

        let a = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("Gemm: missing input A".into()))?;
        let b = ctx
            .input(1)
            .ok_or_else(|| OnnxError::TensorNotFound("Gemm: missing input B".into()))?;
        let c_opt = ctx.input(2);

        // Gemm's A and B are strictly 2D per the ONNX spec. The optimizer's
        // fusion passes only ever emit rank-2 Gemm nodes, but a hand-authored
        // model can still supply a lower/higher-rank tensor here; the M/K/N
        // computation below indexes `a_shape[0]`/`a_shape[1]` (and the same for
        // B) unconditionally, which would otherwise panic on a rank-0 or
        // rank-1 input before any dtype dispatch even runs (including the F32
        // arm, which would otherwise reach the safe `math::gemm` path too late).
        if a.shape.len() != 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Gemm: A must be 2D, got shape {:?}",
                a.shape
            )));
        }
        if b.shape.len() != 2 {
            return Err(OnnxError::ShapeMismatch(format!(
                "Gemm: B must be 2D, got shape {:?}",
                b.shape
            )));
        }

        let alpha = ctx.attrs().f("alpha", 1.0_f32);
        let beta = ctx.attrs().f("beta", 1.0_f32);
        let trans_a = ctx.attrs().i("transA", 0) != 0;
        let trans_b = ctx.attrs().i("transB", 0) != 0;
        let gp = crate::math_typed::GemmParams {
            alpha,
            beta,
            trans_a,
            trans_b,
        };

        // Compute effective M, K, N considering transposes.
        let a_shape = &a.shape;
        let b_shape = &b.shape;
        let (m, k) = if trans_a {
            (a_shape[1], a_shape[0])
        } else {
            (a_shape[0], a_shape[1])
        };
        let n = if trans_b { b_shape[0] } else { b_shape[1] };
        let gd = crate::math_typed::GemmDims { m, n, k };
        let out_shape = vec![m, n];
        let out_len = m * n;

        match (&a.storage, &b.storage) {
            // ── F32: borrow A/B directly (no clone — B is normally the
            // layer weight) and call the shared sgemm-backed kernel. C is
            // borrowed too when it is already F32; any other dtype is
            // numerically converted (matching the pre-optimisation
            // behaviour, which supported a mixed-dtype C via
            // `to_f32_vec()`), but that conversion is at most O(M*N), never
            // the O(M*K)/O(K*N) operand clone this path exists to avoid. ──
            (TensorStorage::F32(a_data), TensorStorage::F32(b_data)) => {
                let mut out = vec![0.0f32; out_len];
                match c_opt.map(|ct| (&ct.storage, ct.shape.as_slice())) {
                    None => {
                        crate::math_typed::gemm_f32(a_data, b_data, &gd, &gp, None, &mut out);
                    }
                    Some((TensorStorage::F32(cd), cs)) => {
                        crate::math_typed::gemm_f32(
                            a_data,
                            b_data,
                            &gd,
                            &gp,
                            Some((cd.as_slice(), cs)),
                            &mut out,
                        );
                    }
                    Some((other, cs)) => {
                        let c_f32 = other.to_f32_vec();
                        crate::math_typed::gemm_f32(
                            a_data,
                            b_data,
                            &gd,
                            &gp,
                            Some((c_f32.as_slice(), cs)),
                            &mut out,
                        );
                    }
                }
                Ok(vec![TypedTensor::new(TensorStorage::F32(out), out_shape)])
            }

            // ── I8 × I8 → I32 ──
            (TensorStorage::I8(a_data), TensorStorage::I8(b_data)) => {
                let c = if let Some(ct) = c_opt {
                    if !crate::math_typed::gemm_bias_shape_supported(&ct.shape, m, n) {
                        return oxionnx_core::default_typed_via_f32(self, ctx);
                    }
                    match &ct.storage {
                        TensorStorage::I32(cd) => Some((cd.as_slice(), ct.shape.as_slice())),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let mut out = vec![0i32; out_len];
                crate::math_typed::gemm_i8_i32(a_data, b_data, &gd, &gp, c, &mut out);
                Ok(vec![TypedTensor::new(TensorStorage::I32(out), out_shape)])
            }

            // ── I32 × I32 → I32 ──
            (TensorStorage::I32(a_data), TensorStorage::I32(b_data)) => {
                let c = if let Some(ct) = c_opt {
                    if !crate::math_typed::gemm_bias_shape_supported(&ct.shape, m, n) {
                        return oxionnx_core::default_typed_via_f32(self, ctx);
                    }
                    match &ct.storage {
                        TensorStorage::I32(cd) => Some((cd.as_slice(), ct.shape.as_slice())),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let mut out = vec![0i32; out_len];
                crate::math_typed::gemm_i32(a_data, b_data, &gd, &gp, c, &mut out);
                Ok(vec![TypedTensor::new(TensorStorage::I32(out), out_shape)])
            }

            // ── F16 × F16 → F16 ──
            (TensorStorage::F16(a_data), TensorStorage::F16(b_data)) => {
                let c = if let Some(ct) = c_opt {
                    if !crate::math_typed::gemm_bias_shape_supported(&ct.shape, m, n) {
                        return oxionnx_core::default_typed_via_f32(self, ctx);
                    }
                    match &ct.storage {
                        TensorStorage::F16(cd) => Some((cd.as_slice(), ct.shape.as_slice())),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let mut out = vec![0u16; out_len];
                crate::math_typed::gemm_f16(a_data, b_data, &gd, &gp, c, &mut out);
                Ok(vec![TypedTensor::new(TensorStorage::F16(out), out_shape)])
            }

            // ── BF16 × BF16 → BF16 ──
            (TensorStorage::BF16(a_data), TensorStorage::BF16(b_data)) => {
                let c = if let Some(ct) = c_opt {
                    if !crate::math_typed::gemm_bias_shape_supported(&ct.shape, m, n) {
                        return oxionnx_core::default_typed_via_f32(self, ctx);
                    }
                    match &ct.storage {
                        TensorStorage::BF16(cd) => Some((cd.as_slice(), ct.shape.as_slice())),
                        _ => return oxionnx_core::default_typed_via_f32(self, ctx),
                    }
                } else {
                    None
                };
                let mut out = vec![0u16; out_len];
                crate::math_typed::gemm_bf16(a_data, b_data, &gd, &gp, c, &mut out);
                Ok(vec![TypedTensor::new(TensorStorage::BF16(out), out_shape)])
            }

            // ── Mixed dtypes: fall back to f32 round-trip ──
            _ => oxionnx_core::default_typed_via_f32(self, ctx),
        }
    }
}
