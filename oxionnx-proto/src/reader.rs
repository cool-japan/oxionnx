//! Generic ONNX weight / attribute reading helpers.
//!
//! This module provides Tier-1 (op-independent) helpers that downstream
//! consumers — converters, quantisers, inspectors — can share instead of
//! re-implementing them. It covers:
//!
//! * Small dtype-code constants ([`dtype_code`]) for the subset of ONNX
//!   element types this module can decode directly.
//! * Attribute-type constants ([`attr_type`]) and typed accessors
//!   ([`attr_int`], [`attr_float`], [`attr_string`], [`attr_ints`]) for the
//!   `AttributeProto` list of an `NodeProto`.
//! * Classification and lookup helpers for externally-stored tensors
//!   ([`is_external`], [`external_entry`]).
//! * Byte-to-`f32` conversion for the floating-point dtypes commonly found
//!   in quantisation pipelines ([`bytes_to_f32`]).
//! * An optional memory-mapped model loader (`MmapModel`, gated behind the
//!   `mmap` feature) that transparently resolves inline vs external
//!   initializer bytes via a cached `memmap2::Mmap` of each sidecar file.
//!
//! The error type used by this module is [`ReaderError`]. Downstream error
//! enums typically wrap it via `#[from]` so `?`-propagation keeps working.

use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use crate::types::{AttributeProto, TensorProto};

/// ONNX element-type codes understood by [`bytes_to_f32`].
///
/// The numeric values come from the ONNX `TensorProto::DataType` enum. Every
/// fixed-width numeric dtype is listed; codes without a constant here
/// (`STRING`, `COMPLEX64/128`, the `FLOAT8*` and 4-bit families) cannot be
/// decoded into `f32` by this crate and are reported through
/// [`ReaderError::UnsupportedDtype`].
pub mod dtype_code {
    /// Sentinel used by `TensorProto.data_type` when the field is unset.
    pub const UNDEFINED: i32 = 0;
    /// IEEE 754 `float32`.
    pub const FLOAT32: i32 = 1;
    /// Unsigned 8-bit integer.
    pub const UINT8: i32 = 2;
    /// Signed 8-bit integer.
    pub const INT8: i32 = 3;
    /// Unsigned 16-bit integer.
    pub const UINT16: i32 = 4;
    /// Signed 16-bit integer.
    pub const INT16: i32 = 5;
    /// Signed 32-bit integer.
    pub const INT32: i32 = 6;
    /// Signed 64-bit integer.
    pub const INT64: i32 = 7;
    /// UTF-8 string (not convertible to `f32`).
    pub const STRING: i32 = 8;
    /// Boolean, one byte per element in `raw_data`.
    pub const BOOL: i32 = 9;
    /// IEEE 754 `float16`.
    pub const FLOAT16: i32 = 10;
    /// IEEE 754 `float64`.
    pub const DOUBLE: i32 = 11;
    /// Unsigned 32-bit integer.
    pub const UINT32: i32 = 12;
    /// Unsigned 64-bit integer.
    pub const UINT64: i32 = 13;
    /// Brain floating-point `bfloat16`.
    pub const BFLOAT16: i32 = 16;
}

/// Size in bytes of one `raw_data` element of `data_type`.
///
/// Returns `None` for dtypes this crate cannot decode into `f32`
/// (`STRING`, complex, `FLOAT8*`, 4-bit families, and `UNDEFINED`).
pub fn dtype_size_bytes(data_type: i32) -> Option<usize> {
    match data_type {
        dtype_code::BOOL | dtype_code::INT8 | dtype_code::UINT8 => Some(1),
        dtype_code::FLOAT16 | dtype_code::BFLOAT16 | dtype_code::INT16 | dtype_code::UINT16 => {
            Some(2)
        }
        dtype_code::FLOAT32 | dtype_code::INT32 | dtype_code::UINT32 => Some(4),
        dtype_code::DOUBLE | dtype_code::INT64 | dtype_code::UINT64 => Some(8),
        _ => None,
    }
}

/// ONNX `AttributeProto::AttributeType` codes used by the typed attribute
/// accessors in this module.
///
/// The numeric values come from the ONNX spec. Only the types consumed by
/// [`attr_int`], [`attr_float`], [`attr_string`], and [`attr_ints`] are
/// listed.
pub mod attr_type {
    /// Scalar `float`.
    pub const FLOAT: i32 = 1;
    /// Scalar `int64`.
    pub const INT: i32 = 2;
    /// Scalar UTF-8 string.
    pub const STRING: i32 = 3;
    /// Repeated `int64` list.
    pub const INTS: i32 = 7;
}

/// Errors raised by the generic ONNX reader helpers.
///
/// Downstream error enums typically wrap this via `#[from]` so `?`
/// propagation continues to work unchanged in callers.
///
/// `#[non_exhaustive]`: new reader failure modes may be added as more of the
/// ONNX wire format is covered. Downstream `match`es need a wildcard arm;
/// existing variants remain constructible as before.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReaderError {
    /// I/O failure while reading a file (`.onnx` or sidecar).
    #[error("I/O error for {path:?}: {source}")]
    Io {
        /// Path that was being accessed when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The `.onnx` protobuf could not be parsed.
    #[error("failed to parse ONNX file {path:?}: {msg}")]
    Parse {
        /// Path to the ONNX file.
        path: PathBuf,
        /// Human-readable parser message.
        msg: String,
    },

    /// Required external-data metadata was missing from an initializer.
    #[error("missing external_data entry '{key}' for initializer '{tensor}'")]
    MissingExternalEntry {
        /// Tensor name in the ONNX graph.
        tensor: String,
        /// Missing key (`"location"`, `"offset"`, `"length"`).
        key: &'static str,
    },

    /// An initializer's `data_type` is not one of the supported dtypes.
    #[error("unsupported initializer dtype {dtype} for tensor '{tensor}'")]
    UnsupportedDtype {
        /// Tensor name.
        tensor: String,
        /// ONNX dtype code (1=float32, 10=float16, …).
        dtype: i32,
    },

    /// A raw byte payload is not a whole number of elements of its dtype.
    #[error(
        "tensor '{tensor}': raw data length {len} is not a multiple of \
         element size {elem_size} (dtype {dtype})"
    )]
    RaggedRawData {
        /// Tensor name.
        tensor: String,
        /// ONNX dtype code.
        dtype: i32,
        /// Byte length of the payload.
        len: usize,
        /// Size of one element in bytes.
        elem_size: usize,
    },

    /// A `TensorProto.dims` entry is negative or too large for this platform.
    #[error("tensor '{tensor}': invalid dimension {dim}")]
    InvalidDim {
        /// Tensor name.
        tensor: String,
        /// Offending dimension as declared on the wire.
        dim: i64,
    },

    /// The product of `TensorProto.dims` overflows `usize`.
    #[error("tensor '{tensor}': shape {dims:?} overflows the addressable range")]
    ShapeOverflow {
        /// Tensor name.
        tensor: String,
        /// Declared dimensions.
        dims: Vec<i64>,
    },

    /// The decoded payload does not hold exactly `product(dims)` elements.
    #[error("tensor '{tensor}': expected {expected} elements for shape {dims:?}, found {found}")]
    ElementCountMismatch {
        /// Tensor name.
        tensor: String,
        /// Declared dimensions.
        dims: Vec<i64>,
        /// Element count implied by `dims`.
        expected: usize,
        /// Element count actually decoded.
        found: usize,
    },

    /// A non-empty initializer carries no data in any field.
    #[error("tensor '{tensor}': no tensor data (expected {expected} elements)")]
    MissingTensorData {
        /// Tensor name.
        tensor: String,
        /// Element count implied by `dims`.
        expected: usize,
    },

    /// An `external_data` `location` is unusable or escapes the model directory.
    #[error("tensor '{tensor}': external data location '{location}' rejected ({reason})")]
    InvalidExternalLocation {
        /// Tensor name.
        tensor: String,
        /// Location string as it appeared in the model.
        location: String,
        /// Why the location was rejected.
        reason: &'static str,
    },

    /// A catch-all for miscellaneous reader errors.
    #[error("{0}")]
    Other(String),
}

/// Return true if the tensor is stored externally (in a sidecar file).
pub fn is_external(tensor: &TensorProto) -> bool {
    tensor.data_location == 1 || !tensor.external_data.is_empty()
}

/// Look up a single key in the `external_data` key-value list.
pub fn external_entry<'a>(tensor: &'a TensorProto, key: &str) -> Option<&'a str> {
    tensor
        .external_data
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Convert raw little-endian bytes of a supported dtype into a `Vec<f32>`.
///
/// This is the single dtype table shared by every `raw_data` / external-data
/// decoding path in the crate. Integer dtypes are widened to `f32`, `BOOL`
/// maps to `0.0` / `1.0`, and `FLOAT16` / `BFLOAT16` are converted through
/// `half`.
///
/// Returns [`ReaderError::UnsupportedDtype`] for dtypes that have no `f32`
/// representation here (`STRING`, complex, `FLOAT8*`, 4-bit families) and
/// [`ReaderError::RaggedRawData`] when `bytes` is not a whole number of
/// elements.
pub fn bytes_to_f32(
    bytes: &[u8],
    data_type: i32,
    tensor_name: &str,
) -> Result<Vec<f32>, ReaderError> {
    let elem_size = dtype_size_bytes(data_type).ok_or_else(|| ReaderError::UnsupportedDtype {
        tensor: tensor_name.to_string(),
        dtype: data_type,
    })?;
    if bytes.len() % elem_size != 0 {
        return Err(ReaderError::RaggedRawData {
            tensor: tensor_name.to_string(),
            dtype: data_type,
            len: bytes.len(),
            elem_size,
        });
    }
    let values: Vec<f32> = match data_type {
        // Hot path first: float32 weights dominate real models.
        dtype_code::FLOAT32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        dtype_code::FLOAT16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        dtype_code::BFLOAT16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        dtype_code::UINT8 => bytes.iter().map(|&b| f32::from(b)).collect(),
        dtype_code::INT8 => bytes.iter().map(|&b| f32::from(b as i8)).collect(),
        dtype_code::BOOL => bytes
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect(),
        dtype_code::UINT16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        dtype_code::INT16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])))
            .collect(),
        dtype_code::INT32 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        dtype_code::UINT32 => bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        dtype_code::INT64 => bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        dtype_code::UINT64 => bytes
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        dtype_code::DOUBLE => bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        other => {
            return Err(ReaderError::UnsupportedDtype {
                tensor: tensor_name.to_string(),
                dtype: other,
            })
        }
    };
    Ok(values)
}

/// Resolve an `external_data` `location` against the directory holding the
/// model, refusing anything that would read outside it.
///
/// The ONNX spec requires `location` to be a relative path below the model
/// directory. A model is untrusted input, so this rejects absolute paths,
/// Windows path prefixes, and `..` traversal (in both separator flavours),
/// then canonicalizes the result and verifies it is still inside the
/// canonicalized model directory — which also defeats symlink escapes.
pub fn resolve_external_path(
    base_dir: &Path,
    location: &str,
    tensor_name: &str,
) -> Result<PathBuf, ReaderError> {
    let reject = |reason: &'static str| ReaderError::InvalidExternalLocation {
        tensor: tensor_name.to_string(),
        location: location.to_string(),
        reason,
    };

    if location.is_empty() {
        return Err(reject("empty location"));
    }
    // Checked on the raw string so a Windows-style `..\\..\\x` is rejected on
    // Unix too, where it would otherwise parse as a single normal component.
    if location.starts_with('/') || location.starts_with('\\') {
        return Err(reject("absolute path"));
    }
    if location.split(['/', '\\']).any(|seg| seg == "..") {
        return Err(reject("parent-directory traversal"));
    }

    let relative = Path::new(location);
    for component in relative.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => return Err(reject("parent-directory traversal")),
            Component::RootDir | Component::Prefix(_) => return Err(reject("absolute path")),
        }
    }

    let base = if base_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base_dir
    };
    let base_canonical = base.canonicalize().map_err(|e| ReaderError::Io {
        path: base.to_path_buf(),
        source: e,
    })?;
    let joined = base_canonical.join(relative);
    let resolved = joined.canonicalize().map_err(|e| ReaderError::Io {
        path: joined.clone(),
        source: e,
    })?;
    if !resolved.starts_with(&base_canonical) {
        return Err(reject("resolves outside the model directory"));
    }
    Ok(resolved)
}

/// Read an integer attribute from the `i` field of an ONNX attribute list.
///
/// Returns `None` if `name` is absent or if the attribute exists but its
/// `attr_type` is not `INT` (2).
pub fn attr_int(attrs: &[AttributeProto], name: &'static str) -> Option<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value.attr_type {
            attr_type::INT => Some(a.value.i),
            _ => None,
        })
}

/// Read a float attribute from the `f` field of an ONNX attribute list.
///
/// Returns `None` if `name` is absent or if the attribute exists but its
/// `attr_type` is not `FLOAT` (1).
pub fn attr_float(attrs: &[AttributeProto], name: &'static str) -> Option<f32> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value.attr_type {
            attr_type::FLOAT => Some(a.value.f),
            _ => None,
        })
}

/// Read a string attribute from the `s` field of an ONNX attribute list.
///
/// Returns `None` if `name` is absent or if the attribute exists but its
/// `attr_type` is not `STRING` (3).
pub fn attr_string<'a>(attrs: &'a [AttributeProto], name: &'static str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value.attr_type {
            attr_type::STRING => Some(a.value.s.as_str()),
            _ => None,
        })
}

/// Read a repeated `int64` attribute from the `ints` field of an ONNX
/// attribute list.
///
/// Returns `None` if `name` is absent or if the attribute exists but its
/// `attr_type` is not `INTS` (7).
pub fn attr_ints<'a>(attrs: &'a [AttributeProto], name: &'static str) -> Option<&'a [i64]> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value.attr_type {
            attr_type::INTS => Some(a.value.ints.as_slice()),
            _ => None,
        })
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory-mapped reader (feature = "mmap")
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "mmap")]
mod mmap_reader {
    #![allow(unsafe_code)]
    //! Feature-gated `OnnxReader` that memory-maps external-data sidecars.

    use std::fs::File;
    use std::path::{Path, PathBuf};

    use memmap2::Mmap;

    use super::{external_entry, is_external, resolve_external_path, ReaderError};
    use crate::parser::parse_model;
    use crate::types::{ModelProto, TensorProto};

    /// An open ONNX model together with its optional memory-mapped sidecars.
    ///
    /// The `.onnx` file is parsed eagerly; sidecar files referenced through
    /// `external_data` entries are memory-mapped lazily on first use and
    /// cached for the lifetime of the reader.
    pub struct OnnxReader {
        /// Path to the `.onnx` file (retained for error messages).
        pub onnx_path: PathBuf,
        /// Directory that holds the `.onnx` file and any sidecar files.
        pub base_dir: PathBuf,
        /// Parsed protobuf structure.
        pub model: ModelProto,
        /// Lazily-populated memory-maps, keyed by sidecar path.
        sidecars: Vec<(PathBuf, Mmap)>,
    }

    impl OnnxReader {
        /// Open `onnx_path` and parse the protobuf. Does not eagerly open any
        /// sidecar — sidecars are memory-mapped on first use.
        pub fn open(onnx_path: &Path) -> Result<Self, ReaderError> {
            let bytes = std::fs::read(onnx_path).map_err(|e| ReaderError::Io {
                path: onnx_path.to_path_buf(),
                source: e,
            })?;
            let model = parse_model(&bytes).map_err(|msg| ReaderError::Parse {
                path: onnx_path.to_path_buf(),
                msg,
            })?;
            let base_dir = onnx_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            Ok(Self {
                onnx_path: onnx_path.to_path_buf(),
                base_dir,
                model,
                sidecars: Vec::new(),
            })
        }

        /// Locate an initializer by exact name.
        pub fn find_initializer(&self, name: &str) -> Option<&TensorProto> {
            self.model
                .graph
                .initializers
                .iter()
                .find(|t| t.name == name)
        }

        /// Return the raw bytes for an initializer, resolving external data as
        /// needed. The returned slice has a lifetime tied to `self` (either the
        /// initializer's `raw_data` or the memory-map of a sidecar).
        pub fn initializer_bytes<'a>(
            &'a mut self,
            tensor: &'a TensorProto,
        ) -> Result<&'a [u8], ReaderError> {
            if !is_external(tensor) {
                return Ok(tensor.raw_data.as_slice());
            }

            // Fetch "location", "offset", "length" from external_data entries.
            let location = external_entry(tensor, "location").ok_or_else(|| {
                ReaderError::MissingExternalEntry {
                    tensor: tensor.name.clone(),
                    key: "location",
                }
            })?;
            let offset: usize = external_entry(tensor, "offset")
                .and_then(|s| s.parse::<u64>().ok())
                .map(|u| u as usize)
                .unwrap_or(0);
            let length: usize = external_entry(tensor, "length")
                .and_then(|s| s.parse::<u64>().ok())
                .map(|u| u as usize)
                .unwrap_or(0);
            if length == 0 {
                // Fall back to a dtype-aware product of dims (only meaningful for
                // byte-sized dtypes like uint8/int8 used by MatMulNBits weights).
                return Err(ReaderError::MissingExternalEntry {
                    tensor: tensor.name.clone(),
                    key: "length",
                });
            }

            // Ensure the sidecar is memory-mapped. The location comes from an
            // untrusted model, so it is sandboxed inside the model directory.
            let sidecar_path = resolve_external_path(&self.base_dir, location, &tensor.name)?;
            self.ensure_sidecar_mapped(&sidecar_path)?;

            let mmap = self
                .sidecars
                .iter()
                .find(|(p, _)| p == &sidecar_path)
                .map(|(_, m)| m)
                .ok_or_else(|| {
                    ReaderError::Other(format!(
                        "internal: sidecar {} was not mapped after ensure_sidecar_mapped",
                        sidecar_path.display()
                    ))
                })?;

            let end = offset.checked_add(length).ok_or_else(|| {
                ReaderError::Other(format!(
                    "offset {offset} + length {length} overflows usize for tensor '{}'",
                    tensor.name
                ))
            })?;
            if end > mmap.len() {
                return Err(ReaderError::Other(format!(
                    "external-data range {offset}..{end} exceeds sidecar size {} for tensor '{}'",
                    mmap.len(),
                    tensor.name
                )));
            }

            Ok(&mmap[offset..end])
        }

        /// Memory-map `path` if it is not already cached.
        fn ensure_sidecar_mapped(&mut self, path: &Path) -> Result<(), ReaderError> {
            if self.sidecars.iter().any(|(p, _)| p == path) {
                return Ok(());
            }
            let file = File::open(path).map_err(|e| ReaderError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            // SAFETY: the mapped file is read-only; `Mmap` is Send+Sync and the
            // underlying storage lives for the duration of the reader.
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| ReaderError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            self.sidecars.push((path.to_path_buf(), mmap));
            Ok(())
        }
    }
}

#[cfg(feature = "mmap")]
pub use mmap_reader::OnnxReader;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AttributeValue;

    #[test]
    fn bytes_to_f32_f32_roundtrip() {
        let values: [f32; 3] = [1.0, -2.5, 0.125];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = bytes_to_f32(&raw, dtype_code::FLOAT32, "t").expect("ok");
        assert_eq!(out, values);
    }

    #[test]
    fn bytes_to_f32_f16_roundtrip() {
        let values = [half::f16::from_f32(1.0), half::f16::from_f32(-0.5)];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = bytes_to_f32(&raw, dtype_code::FLOAT16, "t").expect("ok");
        assert_eq!(out.len(), values.len());
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - -0.5).abs() < 1e-4);
    }

    #[test]
    fn bytes_to_f32_bf16_roundtrip() {
        let values = [half::bf16::from_f32(1.0), half::bf16::from_f32(-0.5)];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = bytes_to_f32(&raw, dtype_code::BFLOAT16, "t").expect("ok");
        assert_eq!(out.len(), values.len());
        assert!((out[0] - 1.0).abs() < 1e-2);
        assert!((out[1] - -0.5).abs() < 1e-2);
    }

    #[test]
    fn bytes_to_f32_unsupported_dtype_errors() {
        // int64 (7) is decodable now; STRING (8) has no f32 representation.
        let err = bytes_to_f32(&[0, 0, 0, 0], dtype_code::STRING, "t").expect_err("must err");
        match err {
            ReaderError::UnsupportedDtype { tensor, dtype } => {
                assert_eq!(tensor, "t");
                assert_eq!(dtype, dtype_code::STRING);
            }
            other => panic!("expected UnsupportedDtype, got {other:?}"),
        }
    }

    #[test]
    fn bytes_to_f32_decodes_integer_dtypes() {
        // int64 [-1, 2]
        let raw: Vec<u8> = [-1i64, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        assert_eq!(
            bytes_to_f32(&raw, dtype_code::INT64, "t").expect("ok"),
            vec![-1.0, 2.0]
        );
        // int8 0xFF => -1, uint8 0xFF => 255
        assert_eq!(
            bytes_to_f32(&[0xFF], dtype_code::INT8, "t").expect("ok"),
            vec![-1.0]
        );
        assert_eq!(
            bytes_to_f32(&[0xFF], dtype_code::UINT8, "t").expect("ok"),
            vec![255.0]
        );
        // bool: any non-zero byte is true
        assert_eq!(
            bytes_to_f32(&[0, 1, 7], dtype_code::BOOL, "t").expect("ok"),
            vec![0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn bytes_to_f32_rejects_ragged_payload() {
        // 6 bytes is not a whole number of float32 elements.
        let err = bytes_to_f32(&[0u8; 6], dtype_code::FLOAT32, "t").expect_err("must err");
        assert!(
            matches!(err, ReaderError::RaggedRawData { elem_size: 4, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn resolve_external_path_rejects_escapes() {
        let dir = std::env::temp_dir().join("oxionnx_reader_sandbox");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("w.bin"), [1u8, 2, 3]).expect("write");

        // Legitimate relative location resolves inside the model directory.
        let ok = resolve_external_path(&dir, "w.bin", "t").expect("relative location");
        assert!(ok.ends_with("w.bin"));
        assert!(resolve_external_path(&dir, "./w.bin", "t").is_ok());

        for bad in [
            "/etc/passwd",
            "../../etc/passwd",
            "..\\..\\secret",
            "",
            "\\x",
        ] {
            let err = resolve_external_path(&dir, bad, "t")
                .err()
                .unwrap_or_else(|| panic!("location {bad:?} must be rejected"));
            assert!(
                matches!(err, ReaderError::InvalidExternalLocation { .. }),
                "location {bad:?}: got {err:?}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn attr(name: &str, value: AttributeValue) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            value,
        }
    }

    #[test]
    fn attr_int_reads_int_attribute() {
        let attrs = vec![attr(
            "bits",
            AttributeValue {
                i: 2,
                attr_type: attr_type::INT,
                ..Default::default()
            },
        )];
        assert_eq!(attr_int(&attrs, "bits"), Some(2));
        assert_eq!(attr_int(&attrs, "missing"), None);
    }

    #[test]
    fn attr_int_rejects_wrong_type() {
        let attrs = vec![attr(
            "bits",
            AttributeValue {
                f: 2.0,
                attr_type: attr_type::FLOAT,
                ..Default::default()
            },
        )];
        // Wrong attr_type: accessor returns None rather than misinterpreting.
        assert_eq!(attr_int(&attrs, "bits"), None);
    }

    #[test]
    fn attr_float_reads_float_attribute() {
        let attrs = vec![attr(
            "eps",
            AttributeValue {
                f: 1e-5,
                attr_type: attr_type::FLOAT,
                ..Default::default()
            },
        )];
        assert_eq!(attr_float(&attrs, "eps"), Some(1e-5));
        assert_eq!(attr_float(&attrs, "missing"), None);
    }

    #[test]
    fn attr_string_reads_string_attribute() {
        let attrs = vec![attr(
            "mode",
            AttributeValue {
                s: "nearest".to_string(),
                attr_type: attr_type::STRING,
                ..Default::default()
            },
        )];
        assert_eq!(attr_string(&attrs, "mode"), Some("nearest"));
        assert_eq!(attr_string(&attrs, "missing"), None);
    }

    #[test]
    fn attr_ints_reads_ints_attribute() {
        let attrs = vec![attr(
            "shape",
            AttributeValue {
                ints: vec![2, 3, 4],
                attr_type: attr_type::INTS,
                ..Default::default()
            },
        )];
        assert_eq!(attr_ints(&attrs, "shape"), Some(&[2, 3, 4][..]));
        assert_eq!(attr_ints(&attrs, "missing"), None);
    }

    #[test]
    fn is_external_detects_external_data() {
        let mut t = TensorProto::default();
        assert!(!is_external(&t));

        t.data_location = 1;
        assert!(is_external(&t));

        t.data_location = 0;
        t.external_data
            .push(("location".to_string(), "data.bin".to_string()));
        assert!(is_external(&t));
    }

    #[test]
    fn external_entry_lookup() {
        let mut t = TensorProto::default();
        t.external_data
            .push(("location".to_string(), "data.bin".to_string()));
        t.external_data
            .push(("offset".to_string(), "42".to_string()));
        assert_eq!(external_entry(&t, "location"), Some("data.bin"));
        assert_eq!(external_entry(&t, "offset"), Some("42"));
        assert_eq!(external_entry(&t, "missing"), None);
    }
}
