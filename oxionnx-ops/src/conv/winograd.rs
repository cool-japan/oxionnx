// ═══════════════════════════════════════════════════════════════════════
// Winograd F(2,3) convolution
// ═══════════════════════════════════════════════════════════════════════

/// Winograd F(2,3) filter transform: U = G · g · G^T
///
/// Transforms a 3×3 filter into Winograd domain (4×4).
/// G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
#[inline]
pub(crate) fn winograd_filter_transform(g: &[f32]) -> [f32; 16] {
    // temp = G * g  (4×3)
    let mut temp = [0.0f32; 12];
    for j in 0..3 {
        temp[j] = g[j];
        temp[3 + j] = 0.5 * (g[j] + g[3 + j] + g[6 + j]);
        temp[6 + j] = 0.5 * (g[j] - g[3 + j] + g[6 + j]);
        temp[9 + j] = g[6 + j];
    }

    // U = temp * G^T  (4×4)
    let mut u = [0.0f32; 16];
    for i in 0..4 {
        let t0 = temp[i * 3];
        let t1 = temp[i * 3 + 1];
        let t2 = temp[i * 3 + 2];
        u[i * 4] = t0;
        u[i * 4 + 1] = 0.5 * (t0 + t1 + t2);
        u[i * 4 + 2] = 0.5 * (t0 - t1 + t2);
        u[i * 4 + 3] = t2;
    }
    u
}

/// Winograd F(2,3) input transform: V = B^T · d · B
///
/// Transforms a 4×4 input tile into Winograd domain.
/// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
#[inline]
pub(crate) fn winograd_input_transform(d: &[f32; 16]) -> [f32; 16] {
    // temp = B^T * d  (4×4)
    let mut temp = [0.0f32; 16];
    for j in 0..4 {
        temp[j] = d[j] - d[8 + j];
        temp[4 + j] = d[4 + j] + d[8 + j];
        temp[8 + j] = d[8 + j] - d[4 + j];
        temp[12 + j] = d[4 + j] - d[12 + j];
    }

    // V = temp * B  (4×4)
    let mut v = [0.0f32; 16];
    for i in 0..4 {
        let t0 = temp[i * 4];
        let t1 = temp[i * 4 + 1];
        let t2 = temp[i * 4 + 2];
        let t3 = temp[i * 4 + 3];
        v[i * 4] = t0 - t2;
        v[i * 4 + 1] = t1 + t2;
        v[i * 4 + 2] = t2 - t1;
        v[i * 4 + 3] = t1 - t3;
    }
    v
}

/// Winograd F(2,3) output transform: Y = A^T · M · A
///
/// Transforms a 4×4 Winograd-domain result back to a 2×2 output tile.
/// A^T = [[1,1,1,0],[0,1,-1,-1]]
#[inline]
pub(crate) fn winograd_output_transform(m: &[f32; 16]) -> [f32; 4] {
    // temp = A^T * M  (2×4)
    let mut temp = [0.0f32; 8];
    for j in 0..4 {
        temp[j] = m[j] + m[4 + j] + m[8 + j];
        temp[4 + j] = m[4 + j] - m[8 + j] - m[12 + j];
    }

    // Y = temp * A  (2×2)
    [
        temp[0] + temp[1] + temp[2],
        temp[1] - temp[2] - temp[3],
        temp[4] + temp[5] + temp[6],
        temp[5] - temp[6] - temp[7],
    ]
}

/// Winograd F(2,3) convolution for 3×3 kernels with stride=1, dilation=1.
///
/// Computes 2×2 output tiles from 4×4 input tiles, reducing multiplications
/// from 36 to 16 per output tile (2.25× fewer).
///
/// Only valid when: kh=kw=3, stride=1, dilation=1, group=1.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_winograd_f2x3(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    c: usize,
    ih: usize,
    iw: usize,
    oc: usize,
    pad: usize,
) -> Result<Vec<f32>, String> {
    if ih + 2 * pad < 3 || iw + 2 * pad < 3 {
        return Err("conv2d_winograd_f2x3: padded input too small for 3x3 kernel".to_string());
    }
    let oh = ih + 2 * pad - 2;
    let ow = iw + 2 * pad - 2;

    let expected_weight_len = oc * c * 9;
    if weight.len() < expected_weight_len {
        return Err(format!(
            "conv2d_winograd_f2x3: weight length {} < expected {}",
            weight.len(),
            expected_weight_len
        ));
    }

    // Number of 2×2 output tiles (ceiling division)
    let tile_h = oh.div_ceil(2);
    let tile_w = ow.div_ceil(2);

    // Pre-transform all filters: U[oc][c] each 4×4 = 16 floats
    let mut u_all = vec![0.0f32; oc * c * 16];
    for o in 0..oc {
        for i in 0..c {
            let g_start = (o * c + i) * 9;
            let u = winograd_filter_transform(&weight[g_start..g_start + 9]);
            let u_start = (o * c + i) * 16;
            u_all[u_start..u_start + 16].copy_from_slice(&u);
        }
    }

    let mut output = vec![0.0f32; n * oc * oh * ow];

    // Reusable buffer for transformed input tiles per channel
    let mut v_tiles = vec![[0.0f32; 16]; c];

    for batch in 0..n {
        for th in 0..tile_h {
            for tw in 0..tile_w {
                let oy_start = th * 2;
                let ox_start = tw * 2;
                let iy_base = oy_start as isize - pad as isize;
                let ix_base = ox_start as isize - pad as isize;

                // How many output rows/cols this tile actually produces
                let out_rows = if oy_start + 2 <= oh { 2 } else { oh - oy_start };
                let out_cols = if ox_start + 2 <= ow { 2 } else { ow - ox_start };

                // Transform input tiles for all channels
                for (ic, v_tile) in v_tiles.iter_mut().enumerate() {
                    let mut d = [0.0f32; 16];
                    let plane_off = (batch * c + ic) * ih * iw;
                    for dy in 0..4usize {
                        let iy = iy_base + dy as isize;
                        for dx in 0..4usize {
                            let ix = ix_base + dx as isize;
                            d[dy * 4 + dx] =
                                if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                    input[plane_off + iy as usize * iw + ix as usize]
                                } else {
                                    0.0
                                };
                        }
                    }
                    *v_tile = winograd_input_transform(&d);
                }

                // For each output channel, accumulate in Winograd domain then transform back
                for o in 0..oc {
                    let mut m_acc = [0.0f32; 16];
                    for (ic, v_tile) in v_tiles.iter().enumerate() {
                        let u_start = (o * c + ic) * 16;
                        let u = &u_all[u_start..u_start + 16];
                        for k in 0..16 {
                            m_acc[k] += u[k] * v_tile[k];
                        }
                    }

                    let y = winograd_output_transform(&m_acc);

                    // Write output tile (handle edge tiles that produce < 2×2)
                    let out_plane = (batch * oc + o) * oh * ow;
                    for dy in 0..out_rows {
                        for dx in 0..out_cols {
                            output[out_plane + (oy_start + dy) * ow + (ox_start + dx)] =
                                y[dy * 2 + dx];
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if let Some(b) = bias {
        for batch in 0..n {
            for (o, &bias_val) in b.iter().enumerate() {
                let plane_off = (batch * oc + o) * oh * ow;
                for j in 0..oh * ow {
                    output[plane_off + j] += bias_val;
                }
            }
        }
    }

    Ok(output)
}
