//! WebAssembly bindings for running OxiONNX in the browser.
//!
//! Enabled with the `wasm` feature flag.
//! Provides a JavaScript-friendly API via wasm-bindgen.
//!
//! On non-wasm32 targets, the module still compiles (for `--all-features`
//! testing) but without the `#[wasm_bindgen]` attributes, so no JS glue
//! is generated.

#[cfg(target_arch = "wasm32")]
use js_sys::Float32Array;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::collections::HashMap;

/// A loaded ONNX model session accessible from JavaScript.
///
/// On wasm32 targets, this struct is exported via wasm-bindgen.
/// On native targets it is a plain Rust struct (useful for compile-checking).
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmSession {
    inner: crate::session::Session,
}

/// Error wrapper that converts to `JsValue` on wasm32 and to `String` on native.
#[cfg(target_arch = "wasm32")]
type WasmResult<T> = Result<T, JsValue>;

#[cfg(not(target_arch = "wasm32"))]
type WasmResult<T> = Result<T, String>;

/// Convert an error into the platform-appropriate error type.
#[cfg(target_arch = "wasm32")]
fn to_err(msg: String) -> JsValue {
    JsValue::from_str(&msg)
}

#[cfg(not(target_arch = "wasm32"))]
fn to_err(msg: String) -> String {
    msg
}

/// Parse a flat shape encoding into a list of shapes.
///
/// Shapes are concatenated with `-1` as a separator.
/// For example, `[1, 3, -1, 1, 5, -1, 2, 2]` becomes `[[1,3], [1,5], [2,2]]`.
fn parse_shapes(flat: &[i32]) -> Vec<Vec<usize>> {
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for &dim in flat {
        if dim < 0 {
            shapes.push(std::mem::take(&mut current));
        } else {
            current.push(dim as usize);
        }
    }
    if !current.is_empty() {
        shapes.push(current);
    }
    shapes
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmSession {
    /// Load an ONNX model from bytes.
    ///
    /// Call this from JS with a `Uint8Array` of the `.onnx` file contents.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(model_bytes: &[u8]) -> WasmResult<WasmSession> {
        let session =
            crate::session::Session::from_bytes(model_bytes).map_err(|e| to_err(format!("{e}")))?;
        Ok(Self { inner: session })
    }

    /// Get the model's input names as a comma-separated string.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn input_names(&self) -> String {
        self.inner.input_names().join(",")
    }

    /// Get the model's output names as a comma-separated string.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn output_names(&self) -> String {
        self.inner.output_names().join(",")
    }

    /// Get the number of nodes in the model.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn node_count(&self) -> usize {
        self.inner.model_info().node_count
    }

    /// Get the number of parameters.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter))]
    pub fn parameter_count(&self) -> usize {
        self.inner.model_info().parameter_count
    }
}

// ---------- wasm32-only methods that return JS types ----------

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSession {
    /// Run inference with a single named input.
    ///
    /// Returns the first output as a `Float32Array`.
    #[wasm_bindgen]
    pub fn run_one(
        &self,
        input_name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Float32Array, JsValue> {
        let tensor = oxionnx_core::Tensor::new(data.to_vec(), shape.to_vec());
        let mut inputs = HashMap::new();
        inputs.insert(input_name, tensor);

        let outputs = self
            .inner
            .run(&inputs)
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;

        let first_output = outputs
            .values()
            .next()
            .ok_or_else(|| JsValue::from_str("No outputs produced"))?;

        let result = Float32Array::new_with_length(first_output.data.len() as u32);
        result.copy_from(&first_output.data);
        Ok(result)
    }

    /// Run inference with multiple named inputs.
    ///
    /// * `input_names` — comma-separated input names
    /// * `input_data` — concatenated f32 data for all inputs
    /// * `input_shapes_flat` — concatenated shapes (flattened, separated by -1)
    ///
    /// Returns concatenated output data as a `Float32Array`.
    #[wasm_bindgen]
    pub fn run_multi(
        &self,
        input_names: &str,
        input_data: &[f32],
        input_shapes_flat: &[i32],
    ) -> Result<Float32Array, JsValue> {
        let names: Vec<&str> = input_names.split(',').collect();
        let shapes = parse_shapes(input_shapes_flat);

        if names.len() != shapes.len() {
            return Err(JsValue::from_str(&format!(
                "Name count {} != shape count {}",
                names.len(),
                shapes.len()
            )));
        }

        // Split data by shape sizes and build tensors
        let mut inputs = HashMap::new();
        let mut data_offset = 0;
        let mut tensors = Vec::with_capacity(names.len());
        for shape in &shapes {
            let size: usize = shape.iter().product();
            if data_offset + size > input_data.len() {
                return Err(JsValue::from_str(
                    "Input data too short for declared shapes",
                ));
            }
            let tensor_data = input_data[data_offset..data_offset + size].to_vec();
            tensors.push(oxionnx_core::Tensor::new(tensor_data, shape.clone()));
            data_offset += size;
        }

        for (i, name) in names.iter().enumerate() {
            inputs.insert(*name, tensors[i].clone());
        }

        let outputs = self
            .inner
            .run(&inputs)
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;

        // Concatenate all outputs in the model's declared output order
        let mut all_data: Vec<f32> = Vec::new();
        for name in self.inner.output_names() {
            if let Some(t) = outputs.get(name.as_str()) {
                all_data.extend_from_slice(&t.data);
            }
        }

        let result = Float32Array::new_with_length(all_data.len() as u32);
        result.copy_from(&all_data);
        Ok(result)
    }
}

// ---------- native-only stubs so the module is testable without wasm ----------

#[cfg(not(target_arch = "wasm32"))]
impl WasmSession {
    /// Run inference with a single named input (native stub, returns raw Vec).
    pub fn run_one_native(
        &self,
        input_name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, String> {
        let tensor = oxionnx_core::Tensor::new(data.to_vec(), shape.to_vec());
        let mut inputs = HashMap::new();
        inputs.insert(input_name, tensor);

        let outputs = self.inner.run(&inputs).map_err(|e| format!("{e}"))?;

        let first_output = outputs
            .values()
            .next()
            .ok_or_else(|| "No outputs produced".to_string())?;

        Ok(first_output.data.clone())
    }

    /// Run inference with multiple named inputs (native stub, returns raw Vec).
    pub fn run_multi_native(
        &self,
        input_names: &str,
        input_data: &[f32],
        input_shapes_flat: &[i32],
    ) -> Result<Vec<f32>, String> {
        let names: Vec<&str> = input_names.split(',').collect();
        let shapes = parse_shapes(input_shapes_flat);

        if names.len() != shapes.len() {
            return Err(format!(
                "Name count {} != shape count {}",
                names.len(),
                shapes.len()
            ));
        }

        let mut inputs = HashMap::new();
        let mut data_offset = 0;
        let mut tensors = Vec::with_capacity(names.len());
        for shape in &shapes {
            let size: usize = shape.iter().product();
            if data_offset + size > input_data.len() {
                return Err("Input data too short for declared shapes".to_string());
            }
            let tensor_data = input_data[data_offset..data_offset + size].to_vec();
            tensors.push(oxionnx_core::Tensor::new(tensor_data, shape.clone()));
            data_offset += size;
        }

        for (i, name) in names.iter().enumerate() {
            inputs.insert(*name, tensors[i].clone());
        }

        let outputs = self.inner.run(&inputs).map_err(|e| format!("{e}"))?;

        let mut all_data: Vec<f32> = Vec::new();
        for name in self.inner.output_names() {
            if let Some(t) = outputs.get(name.as_str()) {
                all_data.extend_from_slice(&t.data);
            }
        }

        Ok(all_data)
    }
}

/// Initialize the WASM module (call once from JS).
///
/// Sets up a panic hook for better error messages in the browser console.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    // console_error_panic_hook integration could go here if the crate is added.
    // For now this is a no-op placeholder that wasm-bindgen requires for start.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shapes_basic() {
        let flat: Vec<i32> = vec![1, 3, -1, 1, 5, -1, 2, 2];
        let shapes = parse_shapes(&flat);
        assert_eq!(shapes, vec![vec![1, 3], vec![1, 5], vec![2, 2]]);
    }

    #[test]
    fn test_parse_shapes_single() {
        let flat: Vec<i32> = vec![2, 3, 4];
        let shapes = parse_shapes(&flat);
        assert_eq!(shapes, vec![vec![2, 3, 4]]);
    }

    #[test]
    fn test_parse_shapes_empty() {
        let flat: Vec<i32> = vec![];
        let shapes = parse_shapes(&flat);
        assert!(shapes.is_empty());
    }

    #[test]
    fn test_parse_shapes_leading_separator() {
        // A leading -1 produces an empty shape at position 0
        let flat: Vec<i32> = vec![-1, 3, 4];
        let shapes = parse_shapes(&flat);
        assert_eq!(shapes, vec![vec![], vec![3, 4]]);
    }

    #[test]
    fn test_parse_shapes_trailing_separator() {
        let flat: Vec<i32> = vec![1, 2, -1];
        let shapes = parse_shapes(&flat);
        // trailing -1 just finishes the current shape, nothing after
        assert_eq!(shapes, vec![vec![1, 2]]);
    }
}
