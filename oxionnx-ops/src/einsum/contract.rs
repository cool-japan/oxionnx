//! Contraction planning and execution for [`crate::einsum`].
//!
//! Two executors live here:
//!
//! * [`execute_pairwise`] — decomposes an N-operand equation into a sequence of
//!   binary contractions (chosen greedily by intermediate size) and lowers each
//!   one to `matrixmultiply::sgemm` via a
//!   *transpose → reshape → GEMM → reshape* pipeline. This is the path that
//!   makes attention-shaped equations (`bhqd,bhkd->bhqk` and friends) run at
//!   BLAS speed instead of scalar speed.
//! * [`execute_general`] — the direct interpretation of the einsum definition:
//!   one loop over output elements, one nested loop over contracted indices.
//!   Retained as the fallback for contractions too small to repay the GEMM
//!   path's materialisation, and used in tests as an independent oracle.
//!
//! # Binary contraction shape
//!
//! Given operands `A` and `B` and the set of labels still needed downstream
//! (`keep` = output labels ∪ labels of the operands not yet contracted), every
//! label falls into exactly one bucket:
//!
//! | in `A` | in `B` | in `keep` | bucket |
//! |---|---|---|---|
//! | ✓ | ✓ | ✓ | **batch** — indexes both sides and survives |
//! | ✓ | ✓ | ✗ | **K** — summed over: the GEMM's inner dimension |
//! | ✓ | ✗ | ✓ | **M** — GEMM's row dimension |
//! | ✗ | ✓ | ✓ | **N** — GEMM's column dimension |
//! | ✓ | ✗ | ✗ | summed out of `A` before the GEMM |
//! | ✗ | ✓ | ✗ | summed out of `B` before the GEMM |
//!
//! `A` is then materialised as `[batch, M, K]` and `B` as `[batch, K, N]`, and
//! one `sgemm` per batch produces `[batch, M, N]`. Every bucket may be empty:
//! an empty K gives `k = 1`, which turns the GEMM into an outer product, and an
//! empty M or N gives `m = 1` / `n = 1`.
//!
//! # Numerics
//!
//! Lowering to GEMM re-associates the sum over the contracted axes, so results
//! can differ from the scalar path in the last few ULP — see the module docs of
//! [`crate::einsum`] for the tolerance this is asserted at. Splitting the batch
//! loop across rayon threads does **not** re-associate anything: each batch
//! element is an independent GEMM writing a disjoint output tile, so the
//! parallel result is bit-identical to the sequential one.

use super::operand::{checked_product, wide_product, Operand};
use super::parse::EinsumPlan;
use oxionnx_core::Tensor;

/// Below this many multiply-accumulates the general interpreter beats the GEMM
/// lowering, whose transposes allocate three intermediate buffers.
pub(crate) const GENERAL_PATH_FLOP_LIMIT: u128 = 4096;

/// Batched GEMMs at or above this many multiply-accumulates are split across
/// rayon threads by batch element.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_GEMM_FLOPS: u128 = 64 * 64 * 64;

/// How a planned binary contraction is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepKind {
    /// `[batch, m, k] × [batch, k, n]` through `matrixmultiply::sgemm`.
    Gemm,
    /// `m == n == 1`: a batch of dot products, where a GEMM call per batch
    /// element would be pure call overhead.
    Dot,
}

/// One planned binary contraction.
#[derive(Debug, Clone)]
pub(crate) struct Step {
    /// Operand-pool index of the left operand.
    pub lhs: usize,
    /// Operand-pool index of the right operand.
    pub rhs: usize,
    /// Labels indexing both operands and surviving the step.
    pub batch_labels: Vec<usize>,
    /// Labels unique to `lhs` that survive.
    pub m_labels: Vec<usize>,
    /// Labels shared by both operands and summed over.
    pub k_labels: Vec<usize>,
    /// Labels unique to `rhs` that survive.
    pub n_labels: Vec<usize>,
    /// Labels summed out of `lhs` before the contraction.
    pub lhs_reduce: Vec<usize>,
    /// Labels summed out of `rhs` before the contraction.
    pub rhs_reduce: Vec<usize>,
    /// Labels of the produced intermediate: `batch ++ m ++ n`.
    pub result_labels: Vec<usize>,
    /// Flattened extents of the four buckets.
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Kernel chosen for this step.
    pub kind: StepKind,
}

/// A full contraction schedule: `steps.len() == operands - 1`.
#[derive(Debug, Clone)]
pub(crate) struct ContractionPlan {
    pub steps: Vec<Step>,
}

/// Distinct labels of a subscript, in first-appearance order.
fn distinct_labels(subs: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(subs.len());
    for &label in subs {
        if !out.contains(&label) {
            out.push(label);
        }
    }
    out
}

/// Membership bitmap over `0..num_labels`.
fn mark(labels: &[usize], num_labels: usize) -> Vec<bool> {
    let mut set = vec![false; num_labels];
    for &label in labels {
        if label < num_labels {
            set[label] = true;
        }
    }
    set
}

/// Bucketing of two operands' labels; see the module docs for the table.
struct Buckets {
    batch: Vec<usize>,
    m: Vec<usize>,
    k: Vec<usize>,
    n: Vec<usize>,
    lhs_reduce: Vec<usize>,
    rhs_reduce: Vec<usize>,
}

fn bucket(lhs: &[usize], rhs: &[usize], keep: &[bool], num_labels: usize) -> Buckets {
    let in_lhs = mark(lhs, num_labels);
    let in_rhs = mark(rhs, num_labels);
    let mut buckets = Buckets {
        batch: Vec::new(),
        m: Vec::new(),
        k: Vec::new(),
        n: Vec::new(),
        lhs_reduce: Vec::new(),
        rhs_reduce: Vec::new(),
    };
    for &label in lhs {
        let kept = keep.get(label).copied().unwrap_or(false);
        if in_rhs.get(label).copied().unwrap_or(false) {
            if kept {
                buckets.batch.push(label);
            } else {
                buckets.k.push(label);
            }
        } else if kept {
            buckets.m.push(label);
        } else {
            buckets.lhs_reduce.push(label);
        }
    }
    for &label in rhs {
        if in_lhs.get(label).copied().unwrap_or(false) {
            continue;
        }
        if keep.get(label).copied().unwrap_or(false) {
            buckets.n.push(label);
        } else {
            buckets.rhs_reduce.push(label);
        }
    }
    buckets
}

/// Extents of `labels`, in order.
fn extents(labels: &[usize], label_sizes: &[usize]) -> Vec<usize> {
    labels
        .iter()
        .map(|&l| label_sizes.get(l).copied().unwrap_or(1))
        .collect()
}

/// Build the contraction schedule for `operand_labels`.
///
/// The pairing order is greedy: at each step the pair whose intermediate has
/// the fewest elements wins, with the contracted extent breaking ties. That is
/// enough to turn a chain like `ij,jk,kl->il` into two matmuls instead of one
/// combinatorial sweep, without the search cost of a full DP over orderings.
///
/// Operands are addressed by a *pool* index: inputs occupy `0..n`, and the
/// result of step `s` occupies index `n + s`. [`execute_pairwise`] keeps a
/// parallel pool so it can follow the schedule without recomputing it.
pub(crate) fn plan_contraction(
    operand_labels: &[Vec<usize>],
    output: &[usize],
    label_sizes: &[usize],
) -> Result<ContractionPlan, String> {
    let num_labels = label_sizes.len();
    let mut pool: Vec<Vec<usize>> = operand_labels.to_vec();
    let mut alive: Vec<usize> = (0..pool.len()).collect();
    let mut steps: Vec<Step> = Vec::new();

    while alive.len() > 1 {
        let mut best: Option<(u128, u128, usize, usize)> = None;
        for i in 0..alive.len() {
            for j in (i + 1)..alive.len() {
                let keep = keep_set(&pool, &alive, i, j, output, num_labels);
                let buckets = bucket(&pool[alive[i]], &pool[alive[j]], &keep, num_labels);
                let result_size = wide_product(&extents(&buckets.batch, label_sizes))
                    .saturating_mul(wide_product(&extents(&buckets.m, label_sizes)))
                    .saturating_mul(wide_product(&extents(&buckets.n, label_sizes)));
                let flops =
                    result_size.saturating_mul(wide_product(&extents(&buckets.k, label_sizes)));
                let candidate = (result_size, flops, i, j);
                let improves = match best {
                    None => true,
                    Some((best_size, best_flops, _, _)) => {
                        candidate.0 < best_size
                            || (candidate.0 == best_size && candidate.1 < best_flops)
                    }
                };
                if improves {
                    best = Some(candidate);
                }
            }
        }
        let Some((_, _, i, j)) = best else {
            break;
        };

        let keep = keep_set(&pool, &alive, i, j, output, num_labels);
        let lhs = alive[i];
        let rhs = alive[j];
        let buckets = bucket(&pool[lhs], &pool[rhs], &keep, num_labels);

        let batch = checked_product(&extents(&buckets.batch, label_sizes))?;
        let m = checked_product(&extents(&buckets.m, label_sizes))?;
        let k = checked_product(&extents(&buckets.k, label_sizes))?;
        let n = checked_product(&extents(&buckets.n, label_sizes))?;

        let mut result_labels = buckets.batch.clone();
        result_labels.extend_from_slice(&buckets.m);
        result_labels.extend_from_slice(&buckets.n);

        steps.push(Step {
            lhs,
            rhs,
            batch_labels: buckets.batch,
            m_labels: buckets.m,
            k_labels: buckets.k,
            n_labels: buckets.n,
            lhs_reduce: buckets.lhs_reduce,
            rhs_reduce: buckets.rhs_reduce,
            result_labels: result_labels.clone(),
            batch,
            m,
            k,
            n,
            kind: if m == 1 && n == 1 {
                StepKind::Dot
            } else {
                StepKind::Gemm
            },
        });

        // `j > i`, so removing `j` first keeps `i` addressable.
        alive.remove(j);
        alive.remove(i);
        alive.push(pool.len());
        pool.push(result_labels);
    }

    Ok(ContractionPlan { steps })
}

/// Labels that must survive a contraction of `alive[i]` with `alive[j]`: the
/// output labels plus everything the not-yet-contracted operands still index.
fn keep_set(
    pool: &[Vec<usize>],
    alive: &[usize],
    i: usize,
    j: usize,
    output: &[usize],
    num_labels: usize,
) -> Vec<bool> {
    let mut keep = mark(output, num_labels);
    for (pos, &idx) in alive.iter().enumerate() {
        if pos == i || pos == j {
            continue;
        }
        for &label in &pool[idx] {
            if label < num_labels {
                keep[label] = true;
            }
        }
    }
    keep
}

/// Total multiply-accumulates the general interpreter would perform.
pub(crate) fn general_path_flops(plan: &EinsumPlan) -> u128 {
    let out_shape = extents(&plan.output_subscript, &plan.label_sizes);
    let mut in_output = vec![false; plan.num_labels];
    for &label in &plan.output_subscript {
        if label < plan.num_labels {
            in_output[label] = true;
        }
    }
    let contracted: Vec<usize> = (0..plan.num_labels)
        .filter(|&l| !in_output[l])
        .map(|l| plan.label_sizes.get(l).copied().unwrap_or(1))
        .collect();
    wide_product(&out_shape).saturating_mul(wide_product(&contracted))
}

// ── Pairwise / GEMM executor ────────────────────────────────────────────────

/// Contract every operand pairwise, lowering each binary step to `sgemm`.
pub(crate) fn execute_pairwise(plan: &EinsumPlan, inputs: &[&Tensor]) -> Result<Tensor, String> {
    let mut pool: Vec<Option<Operand<'_>>> = Vec::with_capacity(inputs.len());
    for (i, subs) in plan.input_subscripts.iter().enumerate() {
        let tensor = inputs
            .get(i)
            .ok_or_else(|| format!("einsum: missing input {i}"))?;
        pool.push(Some(Operand::from_tensor(tensor, subs, &plan.label_sizes)?));
    }

    let operand_labels: Vec<Vec<usize>> = plan
        .input_subscripts
        .iter()
        .map(|subs| distinct_labels(subs))
        .collect();
    let schedule = plan_contraction(&operand_labels, &plan.output_subscript, &plan.label_sizes)?;

    for step in &schedule.steps {
        let lhs = pool
            .get_mut(step.lhs)
            .and_then(Option::take)
            .ok_or_else(|| {
                "einsum: internal: contraction step consumed a missing operand".to_string()
            })?;
        let rhs = pool
            .get_mut(step.rhs)
            .and_then(Option::take)
            .ok_or_else(|| {
                "einsum: internal: contraction step consumed a missing operand".to_string()
            })?;
        let result = contract_step(&lhs, &rhs, step, &plan.label_sizes)?;
        pool.push(Some(result));
    }

    let final_operand = pool
        .into_iter()
        .flatten()
        .next()
        .ok_or_else(|| "einsum: internal: contraction produced no result".to_string())?;
    finalize(&final_operand, &plan.output_subscript, &plan.label_sizes)
}

/// Map `labels` onto `operand`'s axis indices.
fn axes_for(operand: &Operand<'_>, labels: &[usize]) -> Result<Vec<usize>, String> {
    labels
        .iter()
        .map(|&label| {
            operand.axis_of(label).ok_or_else(|| {
                format!("einsum: internal: label {label} missing from a contraction operand")
            })
        })
        .collect()
}

/// Execute one planned binary contraction.
fn contract_step<'a>(
    lhs: &Operand<'_>,
    rhs: &Operand<'_>,
    step: &Step,
    label_sizes: &[usize],
) -> Result<Operand<'a>, String> {
    // A → [batch, M, K] contiguous, summing lhs-only dead labels on the way.
    let mut lhs_keep = axes_for(lhs, &step.batch_labels)?;
    lhs_keep.extend(axes_for(lhs, &step.m_labels)?);
    lhs_keep.extend(axes_for(lhs, &step.k_labels)?);
    let lhs_reduce = axes_for(lhs, &step.lhs_reduce)?;
    let lhs_data = lhs.materialize(&lhs_keep, &lhs_reduce)?;

    // B → [batch, K, N] contiguous.
    let mut rhs_keep = axes_for(rhs, &step.batch_labels)?;
    rhs_keep.extend(axes_for(rhs, &step.k_labels)?);
    rhs_keep.extend(axes_for(rhs, &step.n_labels)?);
    let rhs_reduce = axes_for(rhs, &step.rhs_reduce)?;
    let rhs_data = rhs.materialize(&rhs_keep, &rhs_reduce)?;

    let out = match step.kind {
        StepKind::Dot => dot_batched(&lhs_data, &rhs_data, step.batch, step.k)?,
        StepKind::Gemm => gemm_batched(&lhs_data, &rhs_data, step.batch, step.m, step.k, step.n)?,
    };

    let dims = extents(&step.result_labels, label_sizes);
    Ok(Operand::from_owned(step.result_labels.clone(), dims, out))
}

/// `out[b] = Σ_i a[b, i] · b[b, i]` — the `m == n == 1` degenerate GEMM.
fn dot_batched(a: &[f32], b: &[f32], batch: usize, k: usize) -> Result<Vec<f32>, String> {
    let needed = batch
        .checked_mul(k)
        .ok_or_else(|| "einsum: contraction size overflows usize".to_string())?;
    if a.len() < needed || b.len() < needed {
        return Err(format!(
            "einsum: internal: dot operands hold {} and {} elements, need {needed}",
            a.len(),
            b.len()
        ));
    }
    let mut out = vec![0.0f32; batch];
    for (index, slot) in out.iter_mut().enumerate() {
        let base = index * k;
        *slot = a[base..base + k]
            .iter()
            .zip(b[base..base + k].iter())
            .map(|(x, y)| x * y)
            .sum();
    }
    Ok(out)
}

/// `out[b] = a[b] (m×k) × b[b] (k×n)`, one `sgemm` per batch element.
fn gemm_batched(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let overflow = || "einsum: contraction size overflows usize".to_string();
    let a_stride = m.checked_mul(k).ok_or_else(overflow)?;
    let b_stride = k.checked_mul(n).ok_or_else(overflow)?;
    let c_stride = m.checked_mul(n).ok_or_else(overflow)?;
    let out_len = batch.checked_mul(c_stride).ok_or_else(overflow)?;
    let mut out = vec![0.0f32; out_len];
    if out_len == 0 || k == 0 {
        // `k == 0` contracts over nothing, so every output element is the empty
        // sum — the zeros already in `out`.
        return Ok(out);
    }
    let a_needed = batch.checked_mul(a_stride).ok_or_else(overflow)?;
    let b_needed = batch.checked_mul(b_stride).ok_or_else(overflow)?;
    if a.len() < a_needed || b.len() < b_needed {
        return Err(format!(
            "einsum: internal: gemm operands hold {} and {} elements, need {a_needed} and {b_needed}",
            a.len(),
            b.len()
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let flops = (batch as u128)
            .saturating_mul(m as u128)
            .saturating_mul(k as u128)
            .saturating_mul(n as u128);
        if batch > 1 && flops >= PARALLEL_GEMM_FLOPS {
            use rayon::prelude::*;
            // Each batch element is an independent GEMM over a disjoint output
            // tile, so this is bit-identical to the sequential loop below.
            out.par_chunks_mut(c_stride)
                .enumerate()
                .for_each(|(index, tile)| {
                    crate::math_typed::sgemm_strided(
                        m,
                        k,
                        n,
                        1.0,
                        &a[index * a_stride..],
                        k as isize,
                        1,
                        &b[index * b_stride..],
                        n as isize,
                        1,
                        0.0,
                        tile,
                        n as isize,
                        1,
                    );
                });
            return Ok(out);
        }
    }

    for index in 0..batch {
        let tile = &mut out[index * c_stride..(index + 1) * c_stride];
        crate::math_typed::sgemm_strided(
            m,
            k,
            n,
            1.0,
            &a[index * a_stride..],
            k as isize,
            1,
            &b[index * b_stride..],
            n as isize,
            1,
            0.0,
            tile,
            n as isize,
            1,
        );
    }
    Ok(out)
}

/// Permute (and, for single-operand equations, reduce) the last operand into
/// the output's axis order.
fn finalize(
    operand: &Operand<'_>,
    output: &[usize],
    label_sizes: &[usize],
) -> Result<Tensor, String> {
    let mut kept = vec![false; operand.labels.len()];
    let mut keep_axes = Vec::with_capacity(output.len());
    for &label in output {
        let axis = operand.axis_of(label).ok_or_else(|| {
            format!("einsum: internal: output label {label} missing from the contracted result")
        })?;
        kept[axis] = true;
        keep_axes.push(axis);
    }
    let reduce_axes: Vec<usize> = (0..operand.labels.len()).filter(|&a| !kept[a]).collect();
    let data = operand.materialize(&keep_axes, &reduce_axes)?;
    let shape = extents(output, label_sizes);
    Tensor::try_new(data, shape).map_err(|err| err.to_string())
}

// ── General executor ────────────────────────────────────────────────────────

/// Row-major strides for `dims`.
fn suffix_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1].saturating_mul(dims[axis + 1]);
    }
    strides
}

/// Largest element offset any walk over `dims`/`strides` can reach.
fn reach_of(dims: &[usize], strides: &[usize]) -> Result<usize, String> {
    let mut total = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim == 0 {
            return Ok(0);
        }
        total = total
            .checked_add((dim - 1).saturating_mul(stride))
            .ok_or_else(|| "einsum: strided offset overflows usize".to_string())?;
    }
    Ok(total)
}

/// Direct evaluation of the einsum definition over strided operands.
///
/// Kept O(output × contracted) but with the per-element allocation and the
/// per-inner-iteration stride recomputation of the original implementation
/// removed: the label-value buffer is hoisted out of both loops, the contracted
/// strides are computed once, and the part of each operand's offset that
/// depends only on the output coordinate is hoisted out of the contraction
/// loop.
pub(crate) fn execute_general(
    plan: &EinsumPlan,
    operands: &[Operand<'_>],
) -> Result<Tensor, String> {
    let out_shape = extents(&plan.output_subscript, &plan.label_sizes);
    let out_numel = checked_product(&out_shape)?;
    let mut out_data = vec![0.0f32; out_numel];
    if out_numel == 0 {
        return Tensor::try_new(out_data, out_shape).map_err(|err| err.to_string());
    }

    let mut in_output = vec![false; plan.num_labels];
    for &label in &plan.output_subscript {
        if label < plan.num_labels {
            in_output[label] = true;
        }
    }
    let contracted: Vec<usize> = (0..plan.num_labels).filter(|&l| !in_output[l]).collect();
    let contracted_sizes = extents(&contracted, &plan.label_sizes);
    let contracted_total = checked_product(&contracted_sizes)?;
    if contracted_total == 0 {
        // Contracting over an empty axis makes every output element an empty
        // sum, i.e. the zeros already in `out_data`. Returning here also keeps
        // the bounds check below from rejecting the legitimately empty operand
        // buffers that such an equation necessarily has.
        return Tensor::try_new(out_data, out_shape).map_err(|err| err.to_string());
    }
    let contracted_strides = suffix_strides(&contracted_sizes);
    let out_strides = suffix_strides(&out_shape);

    // Split each operand's axes by whether their label is fixed by the output
    // coordinate or by the contraction coordinate.
    let mut out_axes: Vec<Vec<(usize, usize)>> = Vec::with_capacity(operands.len());
    let mut con_axes: Vec<Vec<(usize, usize)>> = Vec::with_capacity(operands.len());
    for operand in operands {
        let reach = reach_of(&operand.dims, &operand.strides)?;
        if reach >= operand.data.len() {
            return Err(format!(
                "einsum: strided access reaches element {reach} of a {}-element buffer",
                operand.data.len()
            ));
        }
        let mut outs = Vec::new();
        let mut cons = Vec::new();
        for (axis, &label) in operand.labels.iter().enumerate() {
            let stride = operand.strides.get(axis).copied().unwrap_or(0);
            if in_output.get(label).copied().unwrap_or(false) {
                outs.push((label, stride));
            } else {
                cons.push((label, stride));
            }
        }
        out_axes.push(outs);
        con_axes.push(cons);
    }

    // Hoisted out of both loops (was a fresh allocation per output element).
    let mut label_values = vec![0usize; plan.num_labels];
    let mut base_offsets = vec![0usize; operands.len()];

    for (out_flat, slot) in out_data.iter_mut().enumerate() {
        let mut remaining = out_flat;
        for (axis, &label) in plan.output_subscript.iter().enumerate() {
            let stride = out_strides[axis];
            label_values[label] = remaining / stride;
            remaining %= stride;
        }
        for (index, axes) in out_axes.iter().enumerate() {
            let mut offset = 0usize;
            for &(label, stride) in axes {
                offset += label_values[label] * stride;
            }
            base_offsets[index] = offset;
        }

        let mut sum = 0.0f32;
        for contracted_flat in 0..contracted_total {
            let mut remaining = contracted_flat;
            for (axis, &label) in contracted.iter().enumerate() {
                let stride = contracted_strides[axis];
                label_values[label] = remaining / stride;
                remaining %= stride;
            }
            let mut product = 1.0f32;
            for (index, operand) in operands.iter().enumerate() {
                let mut offset = base_offsets[index];
                for &(label, stride) in &con_axes[index] {
                    offset += label_values[label] * stride;
                }
                product *= operand.data[offset];
            }
            sum += product;
        }
        *slot = sum;
    }

    Tensor::try_new(out_data, out_shape).map_err(|err| err.to_string())
}
