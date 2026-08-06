//! Equation parsing for [`crate::einsum`], including full ellipsis (`...`) support.
//!
//! The parser turns an ONNX `Einsum` equation string plus the concrete input
//! shapes into an [`EinsumPlan`]: one label index per input axis, one label
//! index per output axis, and the (broadcast) extent of every label.
//!
//! # Labels
//!
//! Labels are `usize` indices rather than characters because an ellipsis
//! expands to a variable number of anonymous axes that have no character of
//! their own. Indices `0..num_ellipsis` are reserved for the ellipsis axes (in
//! left-to-right order); named ASCII-letter labels are allocated after them, in
//! first-appearance order.
//!
//! # Semantics (numpy-compatible)
//!
//! * `...` may appear at most once per operand and binds to that operand's
//!   axes that are not covered by named labels. Operands may bind a different
//!   number of ellipsis axes; they are **right-aligned** and broadcast against
//!   one another exactly like numpy's leading-dimension broadcasting.
//! * A label repeated *within one operand* selects that operand's diagonal, and
//!   all of its occurrences must have **exactly** equal extents (numpy rejects
//!   `ii` on a `(1, 3)` operand rather than broadcasting it).
//! * A label shared *across operands* broadcasts when one side's extent is `1`
//!   (verified against `numpy.einsum`, which resolves `ij,jk` with `j = 1` on
//!   the left and `j = 3` on the right to `j = 3`).
//! * Explicit output mode (`->` present): the output subscript lists the kept
//!   labels in order, may contain `...` once, and may not repeat a label. If any
//!   ellipsis axes exist the output subscript must contain `...`, matching
//!   numpy's "output has more dimensions than subscripts given" error.
//! * Implicit output mode (no `->`): the output is the ellipsis axes followed by
//!   every named label that occurs exactly once across all operands, in ASCII
//!   order (so `Z` precedes `a`, as numpy does).

use oxionnx_core::Tensor;
use std::collections::HashMap;

/// A parsed, fully resolved einsum equation.
#[derive(Debug, Clone)]
pub(crate) struct EinsumPlan {
    /// One label index per axis of each input, in that input's axis order.
    /// May contain repeats, which denote a diagonal.
    pub input_subscripts: Vec<Vec<usize>>,
    /// One label index per output axis. Never contains repeats.
    pub output_subscript: Vec<usize>,
    /// Broadcast extent of every label, indexed by label.
    pub label_sizes: Vec<usize>,
    /// Number of distinct labels (ellipsis axes included).
    pub num_labels: usize,
}

/// One token of a subscript: a named label or the ellipsis placeholder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Named(u8),
    Ellipsis,
}

/// Tokenize one subscript, rejecting anything that is not an ASCII letter or a
/// well-formed `...`.
fn tokenize(s: &str, what: &str) -> Result<Vec<Token>, String> {
    let bytes = s.as_bytes();
    let mut tokens = Vec::with_capacity(bytes.len());
    let mut seen_ellipsis = false;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' {
            if i + 2 >= bytes.len() || bytes[i + 1] != b'.' || bytes[i + 2] != b'.' {
                return Err(format!(
                    "einsum: {what} contains a '.' that is not part of an ellipsis '...'"
                ));
            }
            if seen_ellipsis {
                return Err(format!(
                    "einsum: {what} contains more than one ellipsis '...'"
                ));
            }
            seen_ellipsis = true;
            tokens.push(Token::Ellipsis);
            i += 3;
        } else if b.is_ascii_alphabetic() {
            tokens.push(Token::Named(b));
            i += 1;
        } else {
            return Err(format!(
                "einsum: invalid character '{}' in {what}; subscripts may only contain \
                 ASCII letters and '...'",
                b.escape_ascii()
            ));
        }
    }
    Ok(tokens)
}

/// Number of `Token::Named` entries in a token list.
fn named_count(tokens: &[Token]) -> usize {
    tokens
        .iter()
        .filter(|t| matches!(t, Token::Named(_)))
        .count()
}

/// Merge `dim` into `slot` under numpy's cross-operand broadcasting rule.
fn merge_label_size(slot: &mut Option<usize>, dim: usize, label: &str) -> Result<(), String> {
    match *slot {
        None => *slot = Some(dim),
        Some(current) if current == dim => {}
        Some(1) => *slot = Some(dim),
        Some(_) if dim == 1 => {}
        Some(current) => {
            return Err(format!(
                "einsum: operands could not be broadcast together: label {label} has \
                 extent {current} in one operand and {dim} in another"
            ));
        }
    }
    Ok(())
}

/// Human-readable name for a label, used in diagnostics.
fn label_name(label: usize, num_ellipsis: usize, label_chars: &[Option<u8>]) -> String {
    match label_chars.get(label).copied().flatten() {
        Some(c) => format!("'{}'", c.escape_ascii()),
        None => format!("'...' axis {} of {num_ellipsis}", label + 1),
    }
}

/// Parse `equation` against the concrete `inputs` and resolve every label extent.
///
/// # Errors
/// Returns a descriptive message for any malformed equation: wrong operand
/// count, illegal characters, a subscript whose length disagrees with its
/// operand's rank, non-broadcastable extents, a repeated or unknown output
/// label, or a missing output ellipsis. Never panics.
pub(crate) fn parse_equation(equation: &str, inputs: &[&Tensor]) -> Result<EinsumPlan, String> {
    if inputs.is_empty() {
        return Err("einsum: expected at least one input tensor".to_string());
    }

    let eq: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match eq.find("->") {
        // "->" is two ASCII bytes, so `pos + 2` is always a char boundary.
        Some(pos) => (&eq[..pos], Some(eq[pos + 2..].to_string())),
        None => (eq.as_str(), None),
    };

    let input_strs: Vec<&str> = lhs.split(',').collect();
    if input_strs.len() != inputs.len() {
        return Err(format!(
            "einsum: equation has {} inputs but got {}",
            input_strs.len(),
            inputs.len()
        ));
    }

    // ── Pass 1: tokenize and size the ellipsis ──────────────────────────────
    let mut input_tokens: Vec<Vec<Token>> = Vec::with_capacity(input_strs.len());
    let mut ellipsis_dims: Vec<usize> = vec![0; input_strs.len()];
    let mut num_ellipsis = 0usize;
    for (i, s) in input_strs.iter().enumerate() {
        let tokens = tokenize(s, &format!("subscript {i} ('{s}')"))?;
        let named = named_count(&tokens);
        let has_ellipsis = tokens.contains(&Token::Ellipsis);
        let ndim = inputs[i].ndim();
        if has_ellipsis {
            if ndim < named {
                return Err(format!(
                    "einsum: input {i} has {ndim} dims but subscript '{s}' names {named} \
                     of them, leaving no axes for the ellipsis"
                ));
            }
            ellipsis_dims[i] = ndim - named;
            num_ellipsis = num_ellipsis.max(ellipsis_dims[i]);
        } else if named != ndim {
            return Err(format!(
                "einsum: input {i} has {ndim} dims but subscript '{s}' has {named} labels"
            ));
        }
        input_tokens.push(tokens);
    }

    // ── Pass 2: allocate labels (ellipsis axes first) ───────────────────────
    let mut label_map: HashMap<u8, usize> = HashMap::new();
    let mut label_chars: Vec<Option<u8>> = vec![None; num_ellipsis];
    let mut label_count = num_ellipsis;
    let mut input_subscripts: Vec<Vec<usize>> = Vec::with_capacity(input_tokens.len());
    for (i, tokens) in input_tokens.iter().enumerate() {
        let mut subs: Vec<usize> = Vec::with_capacity(inputs[i].ndim());
        for token in tokens {
            match *token {
                // Right-align this operand's ellipsis axes inside the widest
                // ellipsis, which is exactly numpy's leading-dim broadcast.
                Token::Ellipsis => subs.extend((num_ellipsis - ellipsis_dims[i])..num_ellipsis),
                Token::Named(c) => {
                    let idx = *label_map.entry(c).or_insert_with(|| {
                        let v = label_count;
                        label_count += 1;
                        v
                    });
                    if idx == label_chars.len() {
                        label_chars.push(Some(c));
                    }
                    subs.push(idx);
                }
            }
        }
        input_subscripts.push(subs);
    }

    // ── Pass 3: resolve label extents ───────────────────────────────────────
    let mut label_sizes: Vec<Option<usize>> = vec![None; label_count];
    // Reused across operands; `None` means "this label is absent from this operand".
    let mut operand_dim: Vec<Option<usize>> = vec![None; label_count];
    for (i, subs) in input_subscripts.iter().enumerate() {
        operand_dim.iter_mut().for_each(|slot| *slot = None);
        for (axis, &label) in subs.iter().enumerate() {
            let dim = inputs[i].shape[axis];
            match operand_dim[label] {
                // A label repeated inside one operand takes its diagonal, which
                // requires exactly equal extents (numpy does not broadcast here).
                Some(prev) if prev != dim => {
                    return Err(format!(
                        "einsum: dimensions in operand {i} for collapsing index {} \
                         don't match ({prev} != {dim})",
                        label_name(label, num_ellipsis, &label_chars)
                    ));
                }
                Some(_) => {}
                None => operand_dim[label] = Some(dim),
            }
        }
        for (label, slot) in operand_dim.iter().enumerate() {
            if let Some(dim) = *slot {
                merge_label_size(
                    &mut label_sizes[label],
                    dim,
                    &label_name(label, num_ellipsis, &label_chars),
                )?;
            }
        }
    }
    // Every allocated label came from some operand's subscript, so each slot is
    // populated; `unwrap_or(1)` is a total fallback that can never be reached.
    let label_sizes: Vec<usize> = label_sizes.into_iter().map(|d| d.unwrap_or(1)).collect();

    // ── Pass 4: output subscript ────────────────────────────────────────────
    let output_subscript = match rhs {
        Some(ref rhs_str) => {
            let tokens = tokenize(rhs_str, &format!("output subscript ('{rhs_str}')"))?;
            if num_ellipsis > 0 && !tokens.contains(&Token::Ellipsis) {
                return Err(format!(
                    "einsum: output has more dimensions than subscripts given, but no \
                     '...' ellipsis provided to broadcast the extra {num_ellipsis} \
                     dimension(s)"
                ));
            }
            let mut used = vec![false; label_count];
            let mut out: Vec<usize> = Vec::with_capacity(tokens.len());
            for token in &tokens {
                match *token {
                    Token::Ellipsis => {
                        for (label, slot) in used.iter_mut().enumerate().take(num_ellipsis) {
                            *slot = true;
                            out.push(label);
                        }
                    }
                    Token::Named(c) => {
                        let label = *label_map.get(&c).ok_or_else(|| {
                            format!(
                                "einsum: output label '{}' does not appear in any input \
                                 subscript",
                                c.escape_ascii()
                            )
                        })?;
                        if used[label] {
                            return Err(format!(
                                "einsum: output subscript includes label '{}' more than once",
                                c.escape_ascii()
                            ));
                        }
                        used[label] = true;
                        out.push(label);
                    }
                }
            }
            out
        }
        None => {
            let mut counts = vec![0usize; label_count];
            for subs in &input_subscripts {
                for &label in subs {
                    counts[label] += 1;
                }
            }
            // Ellipsis axes always come first, then singly-occurring named
            // labels in ASCII order ('Z' before 'a', matching numpy).
            let mut out: Vec<usize> = (0..num_ellipsis).collect();
            let mut named: Vec<(u8, usize)> = label_map.iter().map(|(&c, &l)| (c, l)).collect();
            named.sort_unstable();
            out.extend(
                named
                    .into_iter()
                    .filter(|&(_, label)| counts[label] == 1)
                    .map(|(_, label)| label),
            );
            out
        }
    };

    Ok(EinsumPlan {
        input_subscripts,
        output_subscript,
        label_sizes,
        num_labels: label_count,
    })
}
