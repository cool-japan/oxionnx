//! Einsum operator: Einstein summation convention
//! Supports subscript notation like "ij,jk->ik" for matmul,
//! "ij->ji" for transpose, "ii->i" for diagonal, "bij,bjk->bik" for batch matmul.

use oxionnx_core::Tensor;
use std::collections::HashMap;

/// Parse and execute einsum equation
pub fn einsum(equation: &str, inputs: &[&Tensor]) -> Result<Tensor, String> {
    let plan = parse_equation(equation, inputs)?;
    execute(&plan, inputs)
}

struct EinsumPlan {
    input_subscripts: Vec<Vec<usize>>,
    output_subscript: Vec<usize>,
    label_sizes: Vec<usize>,
    num_labels: usize,
}

fn parse_equation(equation: &str, inputs: &[&Tensor]) -> Result<EinsumPlan, String> {
    let eq = equation.replace(' ', "");
    let (lhs, rhs) = if let Some(pos) = eq.find("->") {
        (&eq[..pos], Some(eq[pos + 2..].to_string()))
    } else {
        (eq.as_str(), None)
    };

    let input_strs: Vec<&str> = lhs.split(',').collect();
    if input_strs.len() != inputs.len() {
        return Err(format!(
            "einsum: equation has {} inputs but got {}",
            input_strs.len(),
            inputs.len()
        ));
    }

    // Map each unique character to a label index
    let mut label_map: HashMap<char, usize> = HashMap::new();
    let mut label_count = 0;

    let mut input_subscripts: Vec<Vec<usize>> = Vec::new();
    for (i, s) in input_strs.iter().enumerate() {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != inputs[i].ndim() {
            return Err(format!(
                "einsum: input {} has {} dims but subscript '{}' has {} labels",
                i,
                inputs[i].ndim(),
                s,
                chars.len()
            ));
        }
        let mut subs = Vec::new();
        for &c in &chars {
            let idx = *label_map.entry(c).or_insert_with(|| {
                let v = label_count;
                label_count += 1;
                v
            });
            subs.push(idx);
        }
        input_subscripts.push(subs);
    }

    // Determine label sizes from input shapes
    let mut label_sizes = vec![0usize; label_count];
    for (i, subs) in input_subscripts.iter().enumerate() {
        for (j, &label) in subs.iter().enumerate() {
            let dim = inputs[i].shape[j];
            if label_sizes[label] == 0 {
                label_sizes[label] = dim;
            } else if label_sizes[label] != dim {
                return Err(format!(
                    "einsum: dimension mismatch for label, expected {} got {}",
                    label_sizes[label], dim
                ));
            }
        }
    }

    // Determine output subscript
    let output_subscript = if let Some(ref rhs_str) = rhs {
        rhs_str
            .chars()
            .map(|c| {
                label_map
                    .get(&c)
                    .copied()
                    .ok_or_else(|| format!("einsum: output label '{c}' not found in inputs"))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        // Implicit: labels that appear exactly once, in alphabetical order
        let mut counts = vec![0usize; label_count];
        for subs in &input_subscripts {
            for &l in subs {
                counts[l] += 1;
            }
        }
        let mut char_label_pairs: Vec<(char, usize)> =
            label_map.iter().map(|(&c, &l)| (c, l)).collect();
        char_label_pairs.sort_by_key(|&(c, _)| c);
        char_label_pairs
            .into_iter()
            .filter(|&(_, l)| counts[l] == 1)
            .map(|(_, l)| l)
            .collect()
    };

    Ok(EinsumPlan {
        input_subscripts,
        output_subscript,
        label_sizes,
        num_labels: label_count,
    })
}

fn execute(plan: &EinsumPlan, inputs: &[&Tensor]) -> Result<Tensor, String> {
    // General algorithm: iterate over all output coordinates + all contracted indices
    // For each combination, multiply corresponding elements from all inputs and accumulate

    let out_shape: Vec<usize> = plan
        .output_subscript
        .iter()
        .map(|&l| plan.label_sizes[l])
        .collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut out_data = vec![0.0f32; out_numel];

    // Identify contracted labels (in inputs but not in output)
    let out_set: std::collections::HashSet<usize> = plan.output_subscript.iter().copied().collect();
    let contracted: Vec<usize> = (0..plan.num_labels)
        .filter(|l| !out_set.contains(l))
        .collect();

    let contracted_sizes: Vec<usize> = contracted.iter().map(|&l| plan.label_sizes[l]).collect();
    let contracted_total: usize = if contracted_sizes.is_empty() {
        1
    } else {
        contracted_sizes.iter().product()
    };

    // Pre-compute strides for each input
    let input_strides: Vec<Vec<usize>> = inputs
        .iter()
        .map(|t| {
            let ndim = t.ndim();
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * t.shape[i + 1];
            }
            strides
        })
        .collect();

    let out_strides = {
        let len = out_shape.len();
        let mut s = vec![1usize; len];
        for i in (0..len.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };

    // Iterate over output elements
    for (out_flat, out_elem) in out_data.iter_mut().enumerate().take(out_numel) {
        // Decode output flat index to label values
        let mut label_values = vec![0usize; plan.num_labels];
        let mut remaining = out_flat;
        for (i, &label) in plan.output_subscript.iter().enumerate() {
            let stride = out_strides[i];
            label_values[label] = remaining / stride;
            remaining %= stride;
        }

        // Sum over contracted indices
        let mut sum = 0.0f32;
        for c_flat in 0..contracted_total {
            // Decode contracted flat index
            let mut c_remaining = c_flat;
            for (ci, &label) in contracted.iter().enumerate() {
                let stride: usize = if ci + 1 < contracted_sizes.len() {
                    contracted_sizes[ci + 1..].iter().product()
                } else {
                    1
                };
                label_values[label] = c_remaining / stride;
                c_remaining %= stride;
            }

            // Compute product of all input elements at these label values
            let mut product = 1.0f32;
            for (inp_idx, subs) in plan.input_subscripts.iter().enumerate() {
                let mut flat = 0;
                for (dim, &label) in subs.iter().enumerate() {
                    flat += label_values[label] * input_strides[inp_idx][dim];
                }
                product *= inputs[inp_idx].data[flat];
            }
            sum += product;
        }
        *out_elem = sum;
    }

    Ok(Tensor::new(out_data, out_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_matmul() {
        // "ij,jk->ik" = matrix multiplication
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let out = einsum("ij,jk->ik", &[&a, &b]).expect("einsum matmul failed");
        assert_eq!(out.shape, vec![2, 2]);
        assert!((out.data[0] - 22.0).abs() < 1e-5); // 1*1+2*3+3*5
        assert!((out.data[1] - 28.0).abs() < 1e-5); // 1*2+2*4+3*6
    }

    #[test]
    fn test_einsum_transpose() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = einsum("ij->ji", &[&a]).expect("einsum transpose failed");
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_einsum_trace() {
        // "ii->" = trace (sum of diagonal)
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let out = einsum("ii->", &[&a]).expect("einsum trace failed");
        assert_eq!(out.shape, Vec::<usize>::new());
        assert!((out.data[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_einsum_batch_matmul() {
        // "bij,bjk->bik" = batched matmul
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![1, 2, 2]);
        let out = einsum("bij,bjk->bik", &[&a, &b]).expect("einsum batch matmul failed");
        assert_eq!(out.shape, vec![1, 2, 2]);
    }

    #[test]
    fn test_einsum_dot_product() {
        // "i,i->" = dot product
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let out = einsum("i,i->", &[&a, &b]).expect("einsum dot product failed");
        assert!((out.data[0] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_einsum_outer_product() {
        // "i,j->ij" = outer product
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3]);
        let out = einsum("i,j->ij", &[&a, &b]).expect("einsum outer product failed");
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_einsum_implicit_output() {
        // No "->" means implicit: labels appearing once, alphabetically
        // "ij,jk" should be same as "ij,jk->ik"
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let out = einsum("ij,jk", &[&a, &b]).expect("einsum implicit failed");
        assert_eq!(out.shape, vec![2, 2]);
        assert!((out.data[0] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn test_einsum_input_count_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let result = einsum("ij,jk->ik", &[&a]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_dim_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let result = einsum("ij->ji", &[&a]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_sum_reduction() {
        // "ij->" = sum all elements
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = einsum("ij->", &[&a]).expect("einsum sum failed");
        assert_eq!(out.shape, Vec::<usize>::new());
        assert!((out.data[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_einsum_diagonal() {
        // "ii->i" = extract diagonal
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = einsum("ii->i", &[&a]).expect("einsum diagonal failed");
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.data, vec![1.0, 4.0]);
    }
}
