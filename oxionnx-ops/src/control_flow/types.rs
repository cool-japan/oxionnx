//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// ONNX Scan operator.
///
/// Iterates over a sequence dimension of the scan inputs, executing the body
/// subgraph once per element. State tensors are carried across iterations.
///
/// - Inputs 0..M-1: initial state tensors
/// - Inputs M..N-1: scan input sequences
/// - Attribute: "body" (Graph), "num_scan_inputs" (int)
/// - Attribute: "scan_input_axes" (int list, default all 0)
/// - Attribute: "scan_input_directions" (int list, default all 0=forward)
/// - Body inputs: (state_0, ..., state_M-1, scan_elem_0, ..., scan_elem_K-1)
/// - Body outputs: (state_0_out, ..., state_M-1_out, scan_out_0, ...)
/// - Final outputs: final state tensors + scan output sequences (concatenated)
pub struct ScanOp;
/// ONNX Loop operator.
///
/// Repeatedly executes a body subgraph until a condition becomes false or the
/// maximum trip count is reached.
///
/// - Input 0: max_trip_count (scalar i64-like; empty string name means infinite)
/// - Input 1: initial condition (scalar bool-like; empty string name means true)
/// - Inputs 2..N: initial values for loop-carried dependencies
/// - Attribute: "body" (Graph)
/// - Body inputs: (iteration_num, condition, ...carried_deps)
/// - Body outputs: (condition_out, ...carried_deps_out, ...scan_outputs)
/// - Final outputs: final carried deps + concatenated scan outputs
pub struct LoopOp;
/// ONNX If operator.
///
/// Conditionally executes one of two subgraphs based on a boolean condition.
/// - Input 0: condition (scalar bool-like tensor; data\[0\] != 0.0 means true)
/// - Attributes: "then_branch" (Graph), "else_branch" (Graph)
/// - Outputs: the outputs of the selected branch subgraph
pub struct IfOp;
