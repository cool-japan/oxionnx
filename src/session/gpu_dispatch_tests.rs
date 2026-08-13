#[cfg(all(test, feature = "gpu"))]
mod zzz_probe {
    #[test]
    fn zzz_probe_gpu_availability() {
        let available = crate::gpu::GpuContext::try_new().is_some();
        eprintln!("PROBE_GPU_AVAILABLE={available}");
    }
}

/// Shape/attribute gating unit tests for every decline decision fixed in
/// this file. None of these need a live GPU adapter — they exercise the
/// pure helper functions `try_gpu_dispatch`'s match arms consult *before*
/// ever touching `crate::gpu`, so they run on every CI machine regardless of
/// Metal/Vulkan/DX12 availability.
#[cfg(all(test, feature = "gpu"))]
mod gating_tests {
    use super::super::*;

    // ── keyed_operand_slot / initializer_key [r3b] ──────────────────────────

    /// The structural reason weight residency cannot promote a memory-bound
    /// node into the uncalibrated `ResidencyTier::Resident` floor: only `Conv`
    /// and `Gemm` key anything, and neither keys slot 0 — which is the input
    /// activation in both. So no op can ever have *every* operand resident, and
    /// no other arm can be credited with residency it does not use.
    #[test]
    fn only_conv_and_gemm_weight_slots_are_ever_keyed() {
        for slot in 1..=2 {
            assert!(keyed_operand_slot(&OpKind::Conv, slot));
            assert!(keyed_operand_slot(&OpKind::Gemm, slot));
        }
        for op in [OpKind::Conv, OpKind::Gemm] {
            assert!(
                !keyed_operand_slot(&op, 0),
                "{op:?} slot 0 is the activation and must never be cached",
            );
            assert!(!keyed_operand_slot(&op, 3));
        }
        // Every other arm passes no keys at all, so an initializer some
        // convolution made resident is still uploaded when these consume it.
        for op in [
            OpKind::Add,
            OpKind::Mul,
            OpKind::PRelu,
            OpKind::LayerNorm,
            OpKind::BatchNorm,
            OpKind::MatMul,
            OpKind::Pad,
            OpKind::Resize,
        ] {
            for slot in 0..=4 {
                assert!(
                    !keyed_operand_slot(&op, slot),
                    "{op:?} keys nothing, so slot {slot} must not count as resident",
                );
            }
        }
    }

    /// The residency key is an initializer name and nothing else — not an
    /// activation's, not a name a node has already produced this run.
    #[test]
    fn an_initializer_key_is_never_shadowed_by_a_graph_intermediate() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0], vec![1]));
        weights.insert("shadowed".to_string(), Tensor::new(vec![2.0], vec![1]));
        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), Tensor::new(vec![3.0], vec![1]));
        intermediates.insert("shadowed".to_string(), Tensor::new(vec![4.0], vec![1]));

        let activations = RunActivations::default();
        assert_eq!(
            initializer_key("w", &weights, &intermediates, &activations),
            Some("w")
        );
        assert_eq!(
            initializer_key("x", &weights, &intermediates, &activations),
            None
        );
        assert_eq!(
            initializer_key("shadowed", &weights, &intermediates, &activations),
            None,
            "a name a node produced resolves to the intermediate, so caching \
             the initializer under it would serve the wrong bytes",
        );
        assert_eq!(
            initializer_key("", &weights, &intermediates, &activations),
            None
        );
        assert_eq!(
            initializer_key("absent", &weights, &intermediates, &activations),
            None
        );
    }

    /// The same rule for a name whose value is on the *device*: a node output
    /// that never touched the host map still shadows an initializer, and keying
    /// it would hand the weight cache one tensor's bytes for another's.
    #[test]
    fn a_device_resident_node_output_shadows_an_initializer_too() {
        let mut weights = HashMap::new();
        weights.insert("shadowed".to_string(), Tensor::new(vec![2.0], vec![1]));
        let intermediates = HashMap::new();
        let activations = RunActivations::default();
        // With nothing resident the name is a plain initializer.
        assert_eq!(
            initializer_key("shadowed", &weights, &intermediates, &activations),
            Some("shadowed")
        );
        // `holds_node_output` is what flips it, and only for node outputs — a
        // *promoted* operand is still the initializer it always was.
        assert!(!activations.holds_node_output("shadowed"));
    }

    // ── matmul_gpu_plan [a4-11/a7-1/a7-9] ───────────────────────────────────

    #[test]
    fn matmul_plan_preserves_leading_batch_dim_of_one() {
        // The canonical transformer projection: MatMul(A[1,128,768], B[768,768]).
        // Before the fix this returned Tensor::new(result, vec![m, n]) = [128, 768]
        // unconditionally — a silent rank drop from 3 to 2.
        let plan = matmul_gpu_plan(&[1, 128, 768], &[768, 768]).expect("batch of 1 must dispatch");
        assert_eq!(plan, (128, 768, 768, vec![1, 128, 768]));
    }

    #[test]
    fn matmul_plan_plain_2d_has_no_batch_prefix() {
        let plan = matmul_gpu_plan(&[4, 8], &[8, 16]).expect("plain 2-D matmul must dispatch");
        assert_eq!(plan, (4, 8, 16, vec![4, 16]));
    }

    #[test]
    fn matmul_plan_batch_of_one_on_both_operands() {
        let plan = matmul_gpu_plan(&[1, 4, 8], &[1, 8, 16])
            .expect("batch of 1 on both sides must dispatch");
        assert_eq!(plan, (4, 8, 16, vec![1, 4, 16]));
    }

    #[test]
    fn matmul_plan_declines_real_batch_on_a() {
        // a is 2-D with batch_size==1 trivially, but a real batch on `a` alone
        // must still decline: [2,4,8] @ [8,16] is not expressible by the flat
        // gpu_matmul kernel.
        assert_eq!(matmul_gpu_plan(&[2, 4, 8], &[8, 16]), None);
    }

    #[test]
    fn matmul_plan_declines_real_batch_on_b_even_when_a_is_2d() {
        // [a7-9] a=[768,768] alone looks batch-free (an=2), but b carries a
        // real batch of 2 that the flat kernel would silently truncate to
        // the first slice if this weren't checked against the *broadcast*
        // of both operands' batch prefixes.
        assert_eq!(matmul_gpu_plan(&[768, 768], &[2, 768, 512]), None);
    }

    #[test]
    fn matmul_plan_declines_mismatched_inner_dimension() {
        assert_eq!(matmul_gpu_plan(&[4, 8], &[9, 16]), None);
    }

    #[test]
    fn matmul_plan_declines_rank_below_two() {
        assert_eq!(matmul_gpu_plan(&[8], &[8, 16]), None);
        assert_eq!(matmul_gpu_plan(&[4, 8], &[8]), None);
    }

    // ── softmax_axis_is_last_dim [a4-12/a7-0] ───────────────────────────────

    #[test]
    fn softmax_axis_negative_one_is_always_last_dim() {
        assert!(softmax_axis_is_last_dim(-1, 3));
        assert!(softmax_axis_is_last_dim(-1, 1));
    }

    #[test]
    fn softmax_axis_explicit_last_index_matches() {
        assert!(softmax_axis_is_last_dim(2, 3));
    }

    #[test]
    fn softmax_axis_one_on_rank_three_is_not_last_dim() {
        // The exact [a7-0] regression case: Softmax(axis=1) on an [8,4,1024]
        // tensor. The GPU kernel can only reduce the last (1024) axis; axis=1
        // (size 4) must decline, not silently normalize over the wrong axis.
        assert!(!softmax_axis_is_last_dim(1, 3));
    }

    #[test]
    fn softmax_axis_rank_zero_never_matches() {
        assert!(!softmax_axis_is_last_dim(-1, 0));
        assert!(!softmax_axis_is_last_dim(0, 0));
    }

    // ── normalize_reduce_axes / normalize_single_reduce_axis [a4-17/a7-7] ──

    #[test]
    fn normalize_reduce_axes_resolves_negative_axes() {
        assert_eq!(normalize_reduce_axes(&[-1], 4), Some(vec![3]));
        assert_eq!(normalize_reduce_axes(&[1, -1], 4), Some(vec![1, 3]));
    }

    #[test]
    fn normalize_reduce_axes_declines_out_of_range() {
        // `axis as usize` used to turn -1 into 18446744073709551615 and index
        // straight off the end of the shape; this must decline instead.
        assert_eq!(normalize_reduce_axes(&[5], 4), None);
        assert_eq!(normalize_reduce_axes(&[-5], 4), None);
    }

    #[test]
    fn normalize_single_reduce_axis_matches_a7_7_example() {
        // ReduceSum(axes=[-1], keepdims=0) on a [100000, 3] tensor.
        assert_eq!(normalize_single_reduce_axis(&[-1], 2), Some(1));
    }

    #[test]
    fn normalize_single_reduce_axis_declines_non_singleton_lists() {
        assert_eq!(normalize_single_reduce_axis(&[], 2), None);
        assert_eq!(normalize_single_reduce_axis(&[0, 1], 2), None);
    }

    // ── single_axis_reduce_shape [a4-17/a7-7] ───────────────────────────────

    #[test]
    fn single_axis_reduce_shape_matches_a7_7_example() {
        // ReduceSum(axes=[1], keepdims=0) on [100000, 3] must produce [100000],
        // not the keepdims=1 shape [100000, 1] the pre-fix code always emitted.
        assert_eq!(
            single_axis_reduce_shape(&[100_000, 3], 1, false),
            vec![100_000]
        );
        assert_eq!(
            single_axis_reduce_shape(&[100_000, 3], 1, true),
            vec![100_000, 1]
        );
    }

    #[test]
    fn single_axis_reduce_shape_full_reduction_is_rank0() {
        // ONNX `ReduceSum` with `keepdims=0` *removes* the reduced axes, so a
        // fully-reduced rank-1 input is a rank-0 scalar: shape `[]`, not `[1]`
        // (`np.sum(np.arange(5), axis=0, keepdims=False).shape == ()`). This must
        // match `reduce_output_shape`/`reduce_with` in
        // oxionnx-ops/src/math/reduce.rs, which the CPU fallback goes through —
        // otherwise the reported output rank would depend on whether the GPU arm
        // happened to accept the node.
        let rank0: Vec<usize> = Vec::new();
        let got = single_axis_reduce_shape(&[5], 0, false);
        assert_eq!(got, rank0);
        // The element count is unchanged: the empty shape's product is 1.
        assert_eq!(got.iter().product::<usize>(), 1);
        // `keepdims=1` is deliberately untouched by the migration.
        assert_eq!(single_axis_reduce_shape(&[5], 0, true), vec![1]);
    }

    /// The CPU kernel the GPU arm stands in for must agree, dimension for
    /// dimension, on every case `single_axis_reduce_shape` claims to handle —
    /// this is the cross-check that keeps the two from drifting apart again.
    #[test]
    fn single_axis_reduce_shape_agrees_with_the_cpu_reduce_kernel() {
        for (shape, axis) in [
            (vec![5usize], 0usize),
            (vec![4, 3], 0),
            (vec![4, 3], 1),
            (vec![2, 3, 5], 1),
            (vec![1, 1], 0),
        ] {
            for keepdims in [false, true] {
                let n: usize = shape.iter().product();
                let x = Tensor::new(vec![1.0_f32; n], shape.clone());
                let want = oxionnx_ops::math::reduce_sum(&x, &[axis as i64], keepdims)
                    .expect("cpu reduce_sum runs");
                assert_eq!(
                    single_axis_reduce_shape(&shape, axis, keepdims),
                    want.shape,
                    "shape={shape:?} axis={axis} keepdims={keepdims}"
                );
            }
        }
    }

    // ── elementwise_shapes_match [a4-18] ────────────────────────────────────

    #[test]
    fn elementwise_shapes_reject_equal_element_count_unequal_shape() {
        // [1,6] and [6,1] both have 6 elements but must broadcast to a
        // 36-element [6,6] result — the flat kernel cannot do that.
        assert!(!elementwise_shapes_match(&[1, 6], &[6, 1]));
        assert!(!elementwise_shapes_match(&[2, 3], &[3, 2]));
    }

    #[test]
    fn elementwise_shapes_accept_identical_shapes() {
        assert!(elementwise_shapes_match(&[4, 5], &[4, 5]));
        assert!(elementwise_shapes_match(&[], &[]));
    }

    // ── [r3a] helpers for the new arms ──────────────────────────────────────
    //
    // Several of these arms are unreachable on native (see
    // `MEMORY_BOUND_TRANSFER_FLOOR`), so these pure-function tests are their
    // only coverage on this target. They are not decoration.

    /// The exact bug the old `Tensor::new(result, a.shape.clone())` had once
    /// the shape-equality gate was removed: InSwapper's AdaIN affine nodes are
    /// `[1,C,1,1] op [1,C,H,W]` in one operand order and the mirror in the
    /// other, so taking `a`'s shape is wrong half the time.
    #[test]
    fn broadcast_out_shape_follows_the_larger_operand_in_either_order() {
        assert_eq!(
            broadcast_binary_out_shape(&[1, 1024, 1, 1], &[1, 1024, 32, 32]),
            Some(vec![1, 1024, 32, 32]),
        );
        assert_eq!(
            broadcast_binary_out_shape(&[1, 1024, 32, 32], &[1, 1024, 1, 1]),
            Some(vec![1, 1024, 32, 32]),
        );
        // [1,6] and [6,1] have equal element counts but broadcast to [6,6].
        assert_eq!(
            broadcast_binary_out_shape(&[1, 6], &[6, 1]),
            Some(vec![6, 6])
        );
        // Genuinely incompatible operands decline.
        assert_eq!(broadcast_binary_out_shape(&[2, 3], &[4, 5]), None);
    }

    /// ONNX `pads` is `[b0,b1,b2,b3,e0,e1,e2,e3]`. H comes from 2/6 and W from
    /// 3/7; the fixture is asymmetric so transposing that pair fails.
    #[test]
    fn pad_plan_reads_the_onnx_begin_end_layout() {
        assert_eq!(
            pad_gpu_plan(4, &[0, 0, 1, 3, 0, 0, 2, 4]),
            // (top, bottom, left, right)
            Some([1, 2, 3, 4]),
        );
        // InSwapper's actual pads.
        assert_eq!(
            pad_gpu_plan(4, &[0, 0, 3, 3, 0, 0, 3, 3]),
            Some([3, 3, 3, 3])
        );
        // A non-zero N or C pad would be silently ignored by the kernel.
        assert_eq!(pad_gpu_plan(4, &[1, 0, 1, 1, 0, 0, 1, 1]), None);
        assert_eq!(pad_gpu_plan(4, &[0, 2, 1, 1, 0, 0, 1, 1]), None);
        // Wrong rank, or a pads list that does not describe rank 4.
        assert_eq!(pad_gpu_plan(3, &[0, 0, 1, 1, 0, 0]), None);
        assert_eq!(pad_gpu_plan(4, &[0, 0, 1, 1]), None);
    }

    #[test]
    fn pad_mode_maps_only_the_two_implemented_modes() {
        assert_eq!(pad_mode_for_gpu(""), Some(crate::gpu::PadMode::Constant));
        assert_eq!(
            pad_mode_for_gpu("constant"),
            Some(crate::gpu::PadMode::Constant)
        );
        assert_eq!(
            pad_mode_for_gpu("reflect"),
            Some(crate::gpu::PadMode::Reflect)
        );
        // `edge`/`wrap` are CPU-only — no WGSL entry point exists.
        assert_eq!(pad_mode_for_gpu("edge"), None);
        assert_eq!(pad_mode_for_gpu("wrap"), None);
    }

    /// The two kernels implement one exact interpolation configuration each.
    /// Anything else produces a plausible image of the right shape with wrong
    /// values, which nothing downstream can detect — so the gate is exact.
    #[test]
    fn resize_kind_matches_only_the_two_implemented_configurations() {
        // InSwapper's two upsamples.
        assert_eq!(
            resize_kind_for_gpu("linear", "pytorch_half_pixel", "floor", 0, 0, "", &[]),
            Some(ResizeGpuKind::BilinearPytorchHalfPixel),
        );
        assert_eq!(
            resize_kind_for_gpu("nearest", "asymmetric", "round_prefer_floor", 0, 0, "", &[]),
            Some(ResizeGpuKind::NearestAsymmetric),
        );
        // SCRFD's two: nearest/asymmetric but nearest_mode="floor", which is
        // NOT what the kernel implements. Measured from the real model.
        assert_eq!(
            resize_kind_for_gpu("nearest", "asymmetric", "floor", 0, 0, "", &[]),
            None,
        );
        // A different coordinate transform is a different image.
        assert_eq!(
            resize_kind_for_gpu("linear", "half_pixel", "floor", 0, 0, "", &[]),
            None,
        );
        // Modifiers the kernels do not implement.
        assert_eq!(
            resize_kind_for_gpu("linear", "pytorch_half_pixel", "floor", 1, 0, "", &[]),
            None,
        );
        assert_eq!(
            resize_kind_for_gpu("linear", "pytorch_half_pixel", "floor", 0, 1, "", &[]),
            None,
        );
        assert_eq!(
            resize_kind_for_gpu("linear", "pytorch_half_pixel", "floor", 0, 0, "", &[2, 3]),
            None,
        );
        assert_eq!(
            resize_kind_for_gpu("cubic", "pytorch_half_pixel", "floor", 0, 0, "", &[]),
            None,
        );
    }

    /// `out = floor(in * scale)` in f32, matching `resolve_plan`
    /// (oxionnx-ops/src/resize.rs) and onnxruntime.
    #[test]
    fn resize_extent_matches_the_cpu_scale_and_size_rules() {
        // InSwapper: [1,1024,32,32] with scales [1,1,2,2].
        assert_eq!(
            resize_spatial_extent(&[1, 1024, 32, 32], Some(&[1.0, 1.0, 2.0, 2.0]), None),
            Some((64, 64)),
        );
        // A non-unit N or C scale is not something the kernel can express.
        assert_eq!(
            resize_spatial_extent(&[1, 1024, 32, 32], Some(&[1.0, 2.0, 2.0, 2.0]), None),
            None,
        );
        // Fractional scale floors, and H/W are independent.
        assert_eq!(
            resize_spatial_extent(&[1, 3, 10, 10], Some(&[1.0, 1.0, 1.5, 2.5]), None),
            Some((15, 25)),
        );
        // `sizes` form: N and C must be unchanged.
        assert_eq!(
            resize_spatial_extent(&[1, 3, 10, 10], None, Some(&[1.0, 3.0, 20.0, 30.0])),
            Some((20, 30)),
        );
        assert_eq!(
            resize_spatial_extent(&[1, 3, 10, 10], None, Some(&[1.0, 6.0, 20.0, 30.0])),
            None,
        );
        // `resolve_plan` errors on both or neither — decline so it reports.
        assert_eq!(
            resize_spatial_extent(
                &[1, 3, 10, 10],
                Some(&[1.0, 1.0, 2.0, 2.0]),
                Some(&[1.0, 3.0, 20.0, 20.0])
            ),
            None,
        );
        assert_eq!(resize_spatial_extent(&[1, 3, 10, 10], None, None), None);
        // Non-finite / non-positive scales decline rather than producing a
        // garbage extent.
        assert_eq!(
            resize_spatial_extent(&[1, 3, 10, 10], Some(&[1.0, 1.0, f32::NAN, 2.0]), None),
            None,
        );
        assert_eq!(
            resize_spatial_extent(&[1, 3, 10, 10], Some(&[1.0, 1.0, 0.0, 2.0]), None),
            None,
        );
    }

    /// `gpu_gemm_nt` reads `B` as `[N, K]` row-major and implements no other
    /// layout — every other `transA`/`transB` combination must decline rather
    /// than be silently mis-indexed.
    #[test]
    fn gemm_plan_accepts_only_trans_b() {
        assert_eq!(
            gemm_gpu_plan(&[1, 512], &[2048, 512], false, true),
            Some((1, 512, 2048)),
        );
        assert_eq!(gemm_gpu_plan(&[1, 512], &[2048, 512], false, false), None);
        assert_eq!(gemm_gpu_plan(&[1, 512], &[2048, 512], true, true), None);
        // Inner dimensions must agree.
        assert_eq!(gemm_gpu_plan(&[1, 512], &[2048, 256], false, true), None);
        // Rank-2 operands only — the arm's `vec![m, n]` output shape depends
        // on this, exactly as `matmul_gpu_plan`'s batch check does.
        assert_eq!(gemm_gpu_plan(&[1, 1, 512], &[2048, 512], false, true), None);
        assert_eq!(gemm_gpu_plan(&[1, 512], &[1, 2048, 512], false, true), None);
    }

    // ── conv_activation_is_recognized / apply_conv_activation [a7-5] ───────

    #[test]
    fn conv_activation_recognizes_only_the_fusion_pass_outputs() {
        assert!(conv_activation_is_recognized(""));
        assert!(conv_activation_is_recognized("relu"));
        assert!(conv_activation_is_recognized("clip"));
        assert!(!conv_activation_is_recognized("sigmoid"));
        assert!(!conv_activation_is_recognized("relu6"));
    }

    /// [r3a] The fused mapping must agree, value for value, with the
    /// host-side pass it replaced — otherwise switching `Conv` to
    /// `gpu_conv2d_fused_async` silently changed what the GPU arm computes.
    ///
    /// The old pass was: `"relu"` → `v.max(0.0)`, `"clip"` →
    /// `v.clamp(min,max)`, anything else → no-op. It is reproduced inline
    /// here (rather than kept as production code) and checked against
    /// `ConvActivation::apply_host`, which is the *same* function the kernel's
    /// own hybrid fallback uses and is written to match its WGSL epilogue
    /// expression-for-expression.
    #[test]
    fn conv_activation_for_gpu_matches_the_old_host_pass() {
        let sample = || vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
        for (activation, min_val, max_val) in [
            ("", f32::NEG_INFINITY, f32::INFINITY),
            ("relu", f32::NEG_INFINITY, f32::INFINITY),
            ("clip", 0.0, 1.0),
            ("clip", 0.0, 6.0),
        ] {
            let mut want = sample();
            match activation {
                "relu" => want.iter_mut().for_each(|v| *v = v.max(0.0)),
                "clip" => want.iter_mut().for_each(|v| *v = v.clamp(min_val, max_val)),
                _ => {}
            }

            let act = conv_activation_for_gpu(activation, min_val, max_val)
                .expect("a recognized activation must map");
            let mut got = sample();
            act.apply_host(&mut got);

            assert_eq!(got, want, "activation={activation:?}");
        }
    }

    /// The mapping and the recognition predicate must accept exactly the same
    /// set — a string one accepts and the other rejects is either a dropped
    /// activation or an unreachable arm.
    #[test]
    fn conv_activation_mapping_and_recognition_agree() {
        for activation in ["", "relu", "clip", "sigmoid", "relu6", "bogus"] {
            assert_eq!(
                conv_activation_for_gpu(activation, 0.0, 6.0).is_some(),
                conv_activation_is_recognized(activation),
                "activation={activation:?}"
            );
        }
    }

    // ── resolve_conv_pads_for_gpu / conv_same_pad_split [conv-pool report] ─

    #[test]
    fn conv_pads_notset_uses_explicit_pads_verbatim() {
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [1, 2, 3, 4],
            ),
            Some([1, 2, 3, 4]),
        );
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "NOTSET",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [1, 2, 3, 4],
            ),
            Some([1, 2, 3, 4]),
        );
    }

    #[test]
    fn conv_pads_valid_is_always_zero_regardless_of_explicit_pads() {
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "VALID",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [9, 9, 9, 9],
            ),
            Some([0, 0, 0, 0]),
        );
    }

    #[test]
    fn conv_pads_same_upper_matches_hand_computed_values() {
        // input 7x7, kernel 3x3, stride 2, dilation 1, SAME_UPPER:
        //   out = ceil(7/2) = 4
        //   eff_k = 3
        //   needed = (4-1)*2 + 3 - 7 = 2  → half=1, split (1,1)
        // Verified: padded = 7+1+1=9, (9-3)/2+1 = 4 == out. The explicit
        // pads attribute [9,9,9,9] must be entirely ignored — this is the
        // exact conv-pool-reported bug (auto_pad was never read at all).
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_UPPER",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [2, 2],
                [1, 1],
                [9, 9, 9, 9],
            ),
            Some([1, 1, 1, 1]),
        );
    }

    #[test]
    fn conv_pads_same_upper_vs_same_lower_split_the_odd_pixel_differently() {
        // input 8x8, kernel 3x3, stride 2, dilation 1:
        //   out = ceil(8/2) = 4, eff_k = 3
        //   needed = (4-1)*2 + 3 - 8 = 1 → half=0
        //   SAME_UPPER: (0,1) → odd pixel at the end
        //   SAME_LOWER: (1,0) → odd pixel at the beginning
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_UPPER",
                &[1, 3, 8, 8],
                &[8, 3, 3, 3],
                [2, 2],
                [1, 1],
                [0, 0, 0, 0],
            ),
            Some([0, 0, 1, 1]),
        );
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_LOWER",
                &[1, 3, 8, 8],
                &[8, 3, 3, 3],
                [2, 2],
                [1, 1],
                [0, 0, 0, 0],
            ),
            Some([1, 1, 0, 0]),
        );
    }

    #[test]
    fn conv_pads_same_upper_declines_when_shape_rank_is_wrong() {
        // NotSet/Valid need no shape info and still resolve; SAME_UPPER /
        // SAME_LOWER need H/W/kH/kW from a 4-D shape and must decline
        // (falling back to the CPU kernel) rather than guess when the model
        // is malformed.
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_UPPER",
                &[1, 3, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [0; 4]
            ),
            None,
        );
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "SAME_LOWER",
                &[1, 3, 7, 7],
                &[8, 3, 3],
                [1, 1],
                [1, 1],
                [0; 4]
            ),
            None,
        );
    }

    #[test]
    fn conv_pads_unrecognized_auto_pad_declines() {
        assert_eq!(
            resolve_conv_pads_for_gpu(
                "BOGUS",
                &[1, 3, 7, 7],
                &[8, 3, 3, 3],
                [1, 1],
                [1, 1],
                [0; 4]
            ),
            None,
        );
    }

    // ── read_positive_pair_gpu / read_pads_gpu / read_group_gpu ────────────
    // [conv-pool report follow-up] These gate every value that flows into
    // `resolve_conv_pads_for_gpu`/`conv_same_pad_split`, closing the panic
    // reported below.

    #[test]
    fn read_positive_pair_gpu_accepts_valid_values_and_defaults() {
        assert_eq!(read_positive_pair_gpu(&[2, 3], 1), Some([2, 3]));
        assert_eq!(read_positive_pair_gpu(&[], 1), Some([1, 1]));
        assert_eq!(read_positive_pair_gpu(&[5], 1), Some([5, 1]));
    }

    #[test]
    fn read_positive_pair_gpu_rejects_non_positive_entries() {
        // The exact reported repro: a malformed `dilations=[-1, 1]`.
        assert_eq!(read_positive_pair_gpu(&[-1, 1], 1), None);
        assert_eq!(read_positive_pair_gpu(&[1, 0], 1), None);
    }

    #[test]
    fn read_pads_gpu_accepts_non_negative_values_and_defaults() {
        assert_eq!(read_pads_gpu(&[1, 2, 3, 4]), Some([1, 2, 3, 4]));
        assert_eq!(read_pads_gpu(&[]), Some([0, 0, 0, 0]));
        assert_eq!(read_pads_gpu(&[0, 0, 0, 0]), Some([0, 0, 0, 0]));
    }

    #[test]
    fn read_pads_gpu_rejects_negative_entries() {
        assert_eq!(read_pads_gpu(&[-1, 0, 0, 0]), None);
    }

    #[test]
    fn read_group_gpu_accepts_positive_values() {
        assert_eq!(read_group_gpu(1), Some(1));
        assert_eq!(read_group_gpu(4), Some(4));
    }

    #[test]
    fn read_group_gpu_rejects_non_positive_values() {
        assert_eq!(read_group_gpu(0), None);
        assert_eq!(read_group_gpu(-1), None);
    }

    // ── conv_same_pad_split panic regression ────────────────────────────────

    #[test]
    fn conv_same_pad_split_saturates_instead_of_panicking_on_extreme_dilation() {
        // Regression for a real bug caught in review: a malformed
        // `dilations=[-1, 1]` attribute, after the arm's raw `as usize`
        // cast, becomes `usize::MAX`. Before this fix, `conv_same_pad_split`
        // computed `eff_k` as a `saturating_mul` result plus a *bare* `+ 1`
        // — `usize::MAX + 1` panics in debug builds ("attempt to add with
        // overflow") and silently wraps to `0` in release. Every current
        // caller now validates `dilation >= 1` first
        // (`read_positive_pair_gpu`), so this value can no longer reach
        // here in practice — this test is the defense-in-depth backstop,
        // proving the arithmetic itself is safe regardless of caller
        // discipline. The result's value is unimportant (a caller that
        // reaches this with an invalid dilation has already failed to
        // validate); what matters is that it returns instead of panicking.
        let (begin, end) = conv_same_pad_split(7, 3, 1, usize::MAX, false);
        let _ = (begin, end);
    }
}

/// End-to-end regression tests that drive `try_gpu_dispatch` itself (not the
/// extracted pure helpers) through a live `GpuContext`, so they exercise the
/// real compute-shader path — not just the CPU-side shape/attribute gating.
///
/// Every test skips (rather than fails) when no adapter is available, the
/// same convention `zzz_probe` and every test in
/// `oxionnx-gpu/src/shaders/tests.rs` already use, so this suite is a no-op
/// on headless CI and a real regression check wherever Metal/Vulkan/DX12 is
/// present (confirmed available here: `PROBE_GPU_AVAILABLE=true`).
///
/// Every tensor is sized to clear the relevant kernel's own GPU_THRESHOLD
/// (oxionnx-gpu/src/compute.rs, oxionnx-gpu/src/shaders/common.rs) so a
/// `try_gpu_dispatch(...).unwrap()` that got `Ok(None)` back would mean the
/// dispatch was wrongly declined, not "too small to bother" — the `.expect`
/// messages below record which threshold each shape was chosen to clear.
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_tests {
    use super::super::*;
    use crate::graph::Attributes;

    #[test]
    fn matmul_e2e_preserves_batch_shape_and_computes_correct_values() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // a = [1, M, K] all-ones; b = [K, N] with column `l` holding the
        // constant `l + 1` in every row. output[i, l] = sum_j a[i,j]*b[j,l]
        // = K * (l + 1), independent of `i` — checking it across every row
        // confirms the batch dim was broadcast rather than only the first
        // row being computed from misaligned memory (the exact a7-9
        // failure mode when `b` carries an unexamined batch dimension).
        //
        // Sized to clear *both* `PerDispatch` gates in `GpuTuning::gemm_admits`
        // (`b` is an intermediate, not a resident weight, so the intensity gate
        // applies): m*k*n = 101*100*2500 = 25,250,000 >= `gemm_min_mac`
        // (25,000,000), and 2mkn/(mk+kn+mn) = 50,500,000/512,600 = 98 >=
        // `gemm_min_intensity` (56). N was 1000 (10.1 M mac) when the floor was
        // the legacy flat 10 M; that no longer clears the measured floor.
        const M: usize = 101;
        const K: usize = 100;
        const N: usize = 2500;

        let a_data = vec![1.0_f32; M * K];
        let mut b_data = vec![0.0_f32; K * N];
        for j in 0..K {
            for l in 0..N {
                b_data[j * N + l] = (l + 1) as f32;
            }
        }

        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), Tensor::new(a_data, vec![1, M, K]));
        intermediates.insert("b".to_string(), Tensor::new(b_data, vec![K, N]));

        let node = Node {
            op: OpKind::MatMul,
            name: "matmul0".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("25.25M mac / intensity 98 clears gemm_min_mac + gemm_min_intensity; must be claimed");
        assert_eq!(outputs.len(), 1);
        let y = &outputs[0];

        // [a4-11/a7-1] The batch prefix must survive — this used to come
        // back as the bare 2-D [M, N] regardless of `a`'s rank.
        assert_eq!(y.shape, vec![1, M, N]);
        assert_eq!(y.data.len(), M * N);
        for l in 0..N {
            let expected = K as f32 * (l as f32 + 1.0);
            for i in 0..M {
                let got = y.data[i * N + l];
                assert!(
                    (got - expected).abs() < 1e-2,
                    "output[{i},{l}] = {got}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn conv_e2e_same_upper_pads_correctly_when_pads_attribute_is_absent() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [conv-pool report] All-ones input/weight, no bias, stride 1, no
        // dilation: with SAME_UPPER padding, an *interior* output pixel sees
        // a full c_in(16) * 3 * 3 = 144 window of ones, while the top-left
        // *corner* pixel sees only the 2x2 in-bounds portion of its window
        // (the other row/col falls in the zero pad) = 16 * 2 * 2 = 64.
        //
        // Before this fix the arm read only the (absent, so all-zero)
        // `pads` attribute and never looked at `auto_pad` — indistinguishable
        // from VALID padding, which has *no* boundary effect at all (every
        // pixel, corner included, would see the full 144). This test fails
        // against the pre-fix code on both the output shape (66x66 vs
        // VALID's 64x64) and the corner value (64 vs a wrongly-uniform 144).
        const C: usize = 16;
        const HW: usize = 66;

        let input = Tensor::new(vec![1.0_f32; C * HW * HW], vec![1, C, HW, HW]);
        let weight = Tensor::new(vec![1.0_f32; C * C * 3 * 3], vec![C, C, 3, 3]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);
        intermediates.insert("w".to_string(), weight);

        let mut attrs = Attributes::default();
        attrs
            .strings
            .insert("auto_pad".to_string(), "SAME_UPPER".to_string());
        // `pads` deliberately left unset: a real SAME_UPPER-exported model
        // never emits it, and silently reading it as all-zero is the bug.

        let node = Node {
            op: OpKind::Conv,
            name: "conv0".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("16*144*4356=10,036,224 FLOPs is above GPU_THRESHOLD (10M); must be claimed");
        let y = &outputs[0];

        // SAME_UPPER must preserve the spatial extent (ceil(66/1) = 66),
        // not VALID's shrunk (66-3)/1+1 = 64.
        assert_eq!(y.shape, vec![1, C, HW, HW]);

        let at = |co: usize, row: usize, col: usize| y.data[co * HW * HW + row * HW + col];
        // Interior pixel: full 3x3x16 window of ones.
        assert!(
            (at(0, 33, 33) - 144.0).abs() < 1e-2,
            "interior = {}",
            at(0, 33, 33)
        );
        // Top-left corner: only a 2x2x16 window is in-bounds — this is the
        // value that proves zero-padding was actually applied.
        assert!(
            (at(0, 0, 0) - 64.0).abs() < 1e-2,
            "corner = {}",
            at(0, 0, 0)
        );
        // A different output channel must agree (the weight is uniform, so
        // every channel sees the same sum) — catches a channel-stride mixup.
        assert!(
            (at(7, 33, 33) - 144.0).abs() < 1e-2,
            "channel 7 interior = {}",
            at(7, 33, 33)
        );
    }

    #[test]
    fn conv_e2e_declines_malformed_negative_dilation_instead_of_panicking() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // Regression for a real bug caught in review: `dilations=[-1, 1]`
        // is invalid per the ONNX spec (the CPU kernel's `read_positive_pair`
        // rejects it with a typed error), but the arm's raw `as usize` cast
        // used to turn `-1_i64` into `usize::MAX` and feed it straight to
        // `resolve_conv_pads_for_gpu` → `conv_same_pad_split`, whose `eff_k
        // = kernel.saturating_sub(1).saturating_mul(dilation) + 1` overflowed
        // on the bare `+ 1` — a debug-build panic (`test result: ok` below
        // *is* the regression check: pre-fix, this test never reached the
        // `assert!` at all).
        //
        // The shape is deliberately the same C=16/HW=66 fixture as
        // `conv_e2e_same_upper_pads_correctly_when_pads_attribute_is_absent`
        // — large enough to clear `gpu_conv2d`'s own GPU_THRESHOLD — so
        // `outputs.is_none()` below is *also* discriminating on its own
        // merits: with a fixture too small for GPU_THRESHOLD, declining
        // would be a foregone conclusion regardless of whether the
        // dilation was ever validated, proving nothing about this fix
        // specifically. At this size, a hypothetical partial fix that
        // merely made `conv_same_pad_split` saturate (without the
        // `read_positive_pair_gpu` validation gate added alongside it)
        // would stop panicking but would still compute nonsense pads from
        // the saturated `dilation` and *dispatch* — `outputs.is_none()`
        // would then correctly fail, catching that gap too.
        const C: usize = 16;
        const HW: usize = 66;
        let input = Tensor::new(vec![1.0_f32; C * HW * HW], vec![1, C, HW, HW]);
        let weight = Tensor::new(vec![1.0_f32; C * C * 3 * 3], vec![C, C, 3, 3]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);
        intermediates.insert("w".to_string(), weight);

        let mut attrs = Attributes::default();
        attrs
            .strings
            .insert("auto_pad".to_string(), "SAME_UPPER".to_string());
        attrs.int_lists.insert("dilations".to_string(), vec![-1, 1]);

        let node = Node {
            op: OpKind::Conv,
            name: "conv_bad_dilation".to_string(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        // Must not panic (see above), and must decline — the malformed
        // attribute means the CPU kernel is the one that should report the
        // typed error, not the GPU arm computing on saturated garbage.
        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error");
        assert!(
            outputs.is_none(),
            "a malformed negative dilation must decline to CPU, not dispatch",
        );
    }

    #[test]
    fn reduce_sum_e2e_normalizes_negative_axis_and_honours_keepdims_false() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [a4-17/a7-7] The exact reported example: ReduceSum(axes=[-1],
        // keepdims=0) on a [100000, 3] tensor. out_count = 100000 >=
        // `reduce_min_output_elements`, so this is claimed by the GPU arm.
        // Before the fix, `axes[0] as usize` on `-1_i64` wrapped to
        // `usize::MAX` instead of normalizing to `1`, and `out_shape[axis] =
        // 1` never consulted `keepdims`.
        //
        // `out_count` here is exactly `ROWS` (reducing the last of `[ROWS, 3]`
        // leaves `outer = ROWS, inner = 1`), and the measured floor is
        // 8,000,000 -- the legacy flat 50_000 this was originally written
        // against is, per `GpuTuning`'s own comment, "160x too low". The
        // negative-axis and `keepdims` handling under test lives in the GPU
        // dispatch arm itself, so the tensor has to be large enough to
        // actually reach it; that costs ~96 MB of input for one test.
        const ROWS: usize = 8_000_000;
        let data: Vec<f32> = (0..ROWS).flat_map(|_| [1.0_f32, 2.0, 3.0]).collect();
        let input = Tensor::new(data, vec![ROWS, 3]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);

        let mut attrs = Attributes::default();
        attrs.int_lists.insert("axes".to_string(), vec![-1]);
        attrs.ints.insert("keepdims".to_string(), 0);

        let node = Node {
            op: OpKind::ReduceSum,
            name: "reduce0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("out_count 8000000 clears reduce_min_output_elements; must be claimed");
        let y = &outputs[0];

        // keepdims=0 must drop the axis entirely: [100000], not the
        // pre-fix, always-emitted keepdims=1 shape [100000, 1].
        assert_eq!(y.shape, vec![ROWS]);
        assert_eq!(y.data.len(), ROWS);
        for (i, &v) in y.data.iter().enumerate() {
            assert!((v - 6.0).abs() < 1e-2, "row {i}: {v} != 6.0");
        }
    }

    #[test]
    fn softmax_e2e_axis_last_dim_dispatches_and_computes_uniform_distribution() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [a4-12/a7-0] This is the positive-path counterpart to the pure
        // `softmax_axis_*` decline tests: axis=-1 on a rank-2 tensor *is* the
        // last dim, so this must still dispatch and compute correctly through
        // the real kernel. All-zero input makes softmax exactly uniform:
        // exp(0)=1 for all 1024 entries, sum=1024.0 (exact in f32), so every
        // output is exactly 1/1024 = 2^-10 -- true per row, so the row count
        // is free to grow.
        //
        // Both softmax gates must clear: `last_dim = 1024 >=
        // softmax_min_row_len` (1000), and `ROWS*LAST = 524,288 >=
        // softmax_min_elements` (262,144). ROWS was 2 when the row-length gate
        // was the only one; `softmax_min_elements` was added because `[64,
        // 1024]` cleared that gate and still measured 1.55x slower than the CPU.
        const ROWS: usize = 512;
        const LAST: usize = 1024;
        let input = Tensor::new(vec![0.0_f32; ROWS * LAST], vec![ROWS, LAST]);

        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), input);

        let mut attrs = Attributes::default();
        attrs.ints.insert("axis".to_string(), -1);

        let node = Node {
            op: OpKind::Softmax,
            name: "softmax0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("row_len 1024 and 524288 elements clear both softmax gates; must be claimed");
        let y = &outputs[0];

        assert_eq!(y.shape, vec![ROWS, LAST]);
        let expected = 1.0_f32 / LAST as f32;
        for &v in &y.data {
            assert!((v - expected).abs() < 1e-6, "{v} != {expected}");
        }
    }

    /// [r3a] `Add` used to dispatch here whenever the shapes matched and the
    /// tensor cleared `BINARY_EW_GPU_THRESHOLD`. On **native** it must now
    /// decline at every size, and that inversion is deliberate: measured over
    /// a whole InSwapper forward, `Add` on the GPU cost 60.83 ms against the
    /// CPU kernel's 36.83 ms, and `Relu` 19.07 ms against 0.45 ms. See
    /// `gpu_residency::MEMORY_BOUND_TRANSFER_FLOOR` for why no size fixes
    /// that while every operand round-trips.
    ///
    /// The floor is now the same on wasm32 (see the constant for why), but
    /// this test stays native-only for a mechanical reason: it drives a
    /// synchronous `GpuContext::try_new`, which returns `None` in a browser by
    /// construction, so there would be nothing to assert there.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn add_declines_on_native_at_every_size_while_operands_round_trip() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        for len in [100_000usize, 4_000_000] {
            let mut intermediates = HashMap::new();
            intermediates.insert("a".to_string(), Tensor::new(vec![2.0_f32; len], vec![len]));
            intermediates.insert("b".to_string(), Tensor::new(vec![3.0_f32; len], vec![len]));

            let node = Node {
                op: OpKind::Add,
                name: "add0".to_string(),
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["y".to_string()],
                attrs: Attributes::default(),
            };

            let dispatched = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
                .expect("a decline is not an error");
            assert!(
                dispatched.is_none(),
                "Add at {len} elements must decline on native: the CPU kernel is faster",
            );
        }
    }

    /// [r3a] `Pad` is one of the three op types measured to genuinely beat its
    /// CPU kernel (0.34x over 14 InSwapper nodes), so it is *not* gated — and
    /// that makes this arm reachable and worth an end-to-end check.
    ///
    /// The shape arithmetic is the part worth pinning: the arm reads ONNX's
    /// `[b0,b1,b2,b3,e0,e1,e2,e3]` layout out of an input *tensor* and has to
    /// pick indices 2/6 for H and 3/7 for W. Transposing that pair still
    /// produces a plausible tensor of the wrong shape, so the fixture is
    /// deliberately asymmetric in H vs W.
    #[test]
    fn pad_e2e_dispatches_and_places_the_spatial_pads_correctly() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // [1, 2, 4, 6], padded by 1 top/bottom and 3 left/right.
        let (n, c, h, w) = (1usize, 2usize, 4usize, 6usize);
        let data: Vec<f32> = (0..n * c * h * w).map(|i| i as f32).collect();
        let mut intermediates = HashMap::new();
        intermediates.insert("x".to_string(), Tensor::new(data, vec![n, c, h, w]));
        intermediates.insert(
            "pads".to_string(),
            Tensor::new(vec![0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 3.0], vec![8]),
        );

        let mut attrs = Attributes::default();
        attrs
            .strings
            .insert("mode".to_string(), "reflect".to_string());
        let node = Node {
            op: OpKind::Pad,
            name: "pad0".to_string(),
            inputs: vec!["x".to_string(), "pads".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &intermediates, &gpu)
            .expect("dispatch must not error")
            .expect("Pad is ungated and rank-4 reflect is supported; must be claimed");
        let y = &outputs[0];
        // H grows by 1+1, W by 3+3 — swapping them would give [1,2,10,8].
        assert_eq!(y.shape, vec![n, c, h + 2, w + 6]);
        assert_eq!(y.data.len(), n * c * (h + 2) * (w + 6));
    }

    /// [r3a] `Gemm` is arithmetic-bound, so it is gated on FLOPs, not size.
    /// InSwapper's AdaIN heads (2.1 MFLOP) must decline; a GEMM above
    /// `GEMM_GPU_MIN_FLOPS` must still dispatch and compute correctly.
    #[test]
    fn gemm_e2e_declines_below_the_flop_gate_and_dispatches_above_it() {
        let Some(gpu) = crate::gpu::GpuContext::try_new() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        let mut attrs = Attributes::default();
        attrs.ints.insert("transB".to_string(), 1);
        let node = Node {
            op: OpKind::Gemm,
            name: "gemm0".to_string(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["y".to_string()],
            attrs,
        };

        // InSwapper's shape: [1,512] x [2048,512]^T = 2.1 MFLOP -> decline.
        let mut small = HashMap::new();
        small.insert(
            "a".to_string(),
            Tensor::new(vec![1.0_f32; 512], vec![1, 512]),
        );
        small.insert(
            "b".to_string(),
            Tensor::new(vec![1.0_f32; 2048 * 512], vec![2048, 512]),
        );
        assert!(
            try_gpu_dispatch(&node, &HashMap::new(), &small, &gpu)
                .expect("a decline is not an error")
                .is_none(),
            "a 2.1 MFLOP Gemm was 3.07x slower on the GPU and must decline",
        );

        // [64, 512] x [2048, 512]^T = 134 MFLOP -> dispatch.
        let (m, k, n) = (64usize, 512usize, 2048usize);
        let mut big = HashMap::new();
        big.insert(
            "a".to_string(),
            Tensor::new(vec![1.0_f32; m * k], vec![m, k]),
        );
        big.insert(
            "b".to_string(),
            Tensor::new(vec![2.0_f32; n * k], vec![n, k]),
        );
        let outputs = try_gpu_dispatch(&node, &HashMap::new(), &big, &gpu)
            .expect("dispatch must not error")
            .expect("134 MFLOP is above GEMM_GPU_MIN_FLOPS; must be claimed");
        let y = &outputs[0];
        assert_eq!(y.shape, vec![m, n]);
        // Every output is sum over k of 1*2, with no C operand.
        let want = 2.0 * k as f32;
        assert!(
            y.data.iter().all(|&v| (v - want).abs() < 1e-2),
            "Gemm result diverged; first = {:?}",
            y.data.first()
        );
    }
}

/// [r3b] End-to-end weight residency: a session that runs the same graph twice
/// must upload its initializers on the first run and none of them on the
/// second.
///
/// This is the measurement the residency work is judged on, taken where a
/// caller would see it — through `Session::run_gpu_async`, on a real device,
/// across two whole runs — rather than at a kernel entry point. The kernel-level
/// counterpart lives in `oxionnx-gpu/tests/r3b_weight_residency.rs`.
///
/// Skips when no adapter is available, like every other GPU test here.
#[cfg(all(test, feature = "gpu"))]
mod weight_residency_e2e {
    use super::super::*;
    use crate::execution_providers::OpPlacement;
    use crate::graph::{Attributes, Graph};
    use crate::session::gpu_residency::run_stats;
    use crate::Session;

    /// `[1,256,16,16] * [256,256,3,3]` with a bias: a 2.36 MB weight against a
    /// 0.26 MB activation, and 302 MFLOP — above the 10 MFLOP dispatch gate,
    /// and weight-dominated, which is the regime the cache exists for.
    const C: usize = 256;
    const HW: usize = 16;
    const WEIGHT_BYTES: u64 = (C * C * 3 * 3 * 4) as u64;
    const BIAS_BYTES: u64 = (C * 4) as u64;

    fn conv_graph() -> (Graph, HashMap<String, Tensor>) {
        let mut attrs = Attributes::default();
        attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
        attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
        attrs.int_lists.insert("pads".to_string(), vec![1, 1, 1, 1]);
        attrs.ints.insert("group".to_string(), 1);

        let conv = Node {
            op: OpKind::Conv,
            name: "conv0".to_string(),
            inputs: vec![
                "x".to_string(),
                "conv.weight".to_string(),
                "conv.bias".to_string(),
            ],
            outputs: vec!["y".to_string()],
            attrs,
        };
        let graph = Graph {
            nodes: vec![conv],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            ..Default::default()
        };

        // Deterministic, signed, non-monotonic: a flat fill would hide a
        // cached buffer being bound at the wrong offset.
        let fill = |len: usize, seed: u32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
                    ((x % 23) as f32) * 0.037 - 0.4
                })
                .collect()
        };
        let mut weights = HashMap::new();
        weights.insert(
            "conv.weight".to_string(),
            Tensor::new(fill(C * C * 3 * 3, 13), vec![C, C, 3, 3]),
        );
        weights.insert("conv.bias".to_string(), Tensor::new(fill(C, 29), vec![C]));
        (graph, weights)
    }

    #[test]
    fn a_second_run_of_the_same_graph_uploads_no_initializer_bytes() {
        let (graph, weights) = conv_graph();
        let mut session = Session::from_graph(graph, weights).expect("from_graph");
        if !pollster::block_on(session.enable_gpu_async()) {
            eprintln!("skip: no GPU adapter available");
            return;
        }
        // The default placement is `CpuOnly`, under which the wgpu backend is
        // never offered the node and this test would prove nothing.
        session.op_placement = OpPlacement::Auto {
            gpu_threshold_bytes: 65_536,
        };

        let mut inputs = HashMap::new();
        inputs.insert(
            "x",
            Tensor::new(
                (0..C * HW * HW)
                    .map(|i| ((i % 19) as f32) * 0.031 - 0.3)
                    .collect(),
                vec![1, C, HW, HW],
            ),
        );

        let first = pollster::block_on(session.run_gpu_async(&inputs)).expect("first run");
        let first_stats = run_stats();
        if first_stats.gpu_nodes == 0 {
            eprintln!("skip: the adapter declined the convolution");
            return;
        }

        assert_eq!(
            first_stats.weight_upload_bytes,
            WEIGHT_BYTES + BIAS_BYTES,
            "the first run uploads the weight and the bias, once each",
        );
        assert_eq!(first_stats.weight_cache_misses, 2);
        assert_eq!(first_stats.weight_cache_hits, 0);

        let second = pollster::block_on(session.run_gpu_async(&inputs)).expect("second run");
        let second_stats = run_stats();

        // THE assertion this whole change exists for.
        assert_eq!(
            second_stats.weight_upload_bytes, 0,
            "a second frame must not re-upload a single byte of invariant weight",
        );
        assert_eq!(second_stats.weight_cache_misses, 0);
        assert_eq!(
            second_stats.weight_cache_hits, 2,
            "both initializers served from the device",
        );
        assert_eq!(
            second_stats.gpu_nodes, first_stats.gpu_nodes,
            "residency must not change which nodes the GPU accepts",
        );

        let first_y = first.get("y").expect("first output");
        let second_y = second.get("y").expect("second output");
        assert_eq!(first_y.shape, vec![1, C, HW, HW]);
        // Exact: same kernel, same bytes, same order. A tolerance here would
        // hide precisely the failure mode this test is for.
        assert_eq!(
            first_y.data, second_y.data,
            "a resident weight must compute bit-identical results",
        );

        // The session's own context holds them, and holds only them.
        let ctx = session.gpu.as_ref().expect("gpu context");
        assert!(ctx.is_resident("conv.weight"));
        assert!(ctx.is_resident("conv.bias"));
        assert!(!ctx.is_resident("x"), "an activation is never resident");
        assert_eq!(ctx.resident_len(), 2);
        assert_eq!(ctx.resident_bytes(), WEIGHT_BYTES + BIAS_BYTES);
        assert!(
            !ctx.is_degraded(),
            "device degraded during the run: {:?}",
            ctx.last_error()
        );
    }
}
