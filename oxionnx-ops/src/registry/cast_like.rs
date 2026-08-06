//! `CastLike` operator implementation.

use oxionnx_core::{DType, OnnxError, OpContext, Operator, Tensor, TypedOpContext, TypedTensor};

/// ONNX `CastLike` (opset 15+): cast `input` to the element type of `target_type`.
///
/// Modern PyTorch / ONNX exporters emit this constantly as a dtype-agnostic
/// `Cast` (it needs no `to` attribute, so a rewritten graph does not have to
/// re-resolve dtypes).
///
/// # Behaviour on the two execution paths
///
/// * **Typed path** ([`Operator::execute_typed`]) — exact: the target's real
///   [`DType`] is read off input 1 and the cast is performed, matching `Cast`
///   with the equivalent `to` attribute.
/// * **f32 path** ([`Operator::execute`]) — an **identity copy** of input 0.
///   Every tensor on that path is already f32 and carries no dtype tag, so
///   there is no target type to read; `Cast` only works there because its `to`
///   attribute survives in the node. Guessing a dtype from the target tensor's
///   *values* (e.g. "all lanes are integral, so it must be int64") would make
///   the result data-dependent and is deliberately not done — the value of a
///   `CastLike` between float types is the identity anyway, and this is the
///   overwhelmingly common case in exported graphs (f32 → f16 mixed precision,
///   which the typed path handles exactly).
///
/// The f32-path limitation is a property of the dtype-erased runtime, not of
/// this operator; it disappears the moment the session runs typed.
pub struct CastLikeOp;

impl Operator for CastLikeOp {
    fn op_type(&self) -> &str {
        "CastLike"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        // Input 1 is required by the schema even though the f32 path cannot
        // read a dtype from it; validating its presence keeps a malformed node
        // a typed error rather than a silent pass-through.
        let x = ctx.input(0)?;
        let _target = ctx.input(1)?;
        Ok(vec![Tensor::new(x.data.clone(), x.shape.clone())])
    }

    fn native_dtypes(&self) -> &'static [DType] {
        &[
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::Bool,
        ]
    }

    fn execute_typed(&self, ctx: &TypedOpContext<'_>) -> Result<Vec<TypedTensor>, OnnxError> {
        let input = ctx
            .input(0)
            .ok_or_else(|| OnnxError::TensorNotFound("CastLike: missing input[0]".into()))?;
        let target = ctx.input(1).ok_or_else(|| {
            OnnxError::TensorNotFound("CastLike: missing input[1] (target_type)".into())
        })?;
        Ok(vec![input.cast(target.dtype())])
    }
}
