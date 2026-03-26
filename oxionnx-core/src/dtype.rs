//! Data type system for multi-dtype tensor support.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// ONNX data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F64 => "float64",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::Bool => "bool",
        }
    }

    /// Convert from ONNX protobuf data_type integer.
    pub fn from_onnx(dt: i32) -> Option<Self> {
        match dt {
            1 => Some(Self::F32),
            2 => Some(Self::U8),
            3 => Some(Self::I8),
            4 => Some(Self::U16),
            5 => Some(Self::I16),
            6 => Some(Self::I32),
            7 => Some(Self::I64),
            9 => Some(Self::Bool),
            10 => Some(Self::F16),
            11 => Some(Self::F64),
            12 => Some(Self::U32),
            13 => Some(Self::U64),
            16 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Convert to ONNX protobuf data_type integer.
    pub fn to_onnx(&self) -> i32 {
        match self {
            Self::F32 => 1,
            Self::U8 => 2,
            Self::I8 => 3,
            Self::U16 => 4,
            Self::I16 => 5,
            Self::I32 => 6,
            Self::I64 => 7,
            Self::Bool => 9,
            Self::F16 => 10,
            Self::F64 => 11,
            Self::U32 => 12,
            Self::U64 => 13,
            Self::BF16 => 16,
        }
    }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64)
    }

    /// Whether this is a signed integer type.
    pub fn is_signed_int(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Whether this is an unsigned integer type.
    pub fn is_unsigned_int(&self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::Bool
        )
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Automatic type promotion rules following NumPy/ONNX conventions.
/// Returns the result type when combining two dtypes in an operation.
pub fn promote(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }

    // Float always wins over integer
    match (a.is_float(), b.is_float()) {
        (true, false) => return promote_float(a),
        (false, true) => return promote_float(b),
        (true, true) => return promote_float_float(a, b),
        _ => {}
    }

    // Both integers: promote to wider type
    let a_size = a.size_bytes();
    let b_size = b.size_bytes();
    if a_size >= b_size {
        a
    } else {
        b
    }
}

fn promote_float(float_type: DType) -> DType {
    // Float type wins, but upgrade to at least F32
    match float_type {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

fn promote_float_float(a: DType, b: DType) -> DType {
    // Larger float wins
    if a.size_bytes() >= b.size_bytes() {
        a
    } else {
        b
    }
}

/// Type-erased tensor storage.
#[derive(Debug, Clone)]
pub enum TensorStorage {
    F32(Vec<f32>),
    F16(Vec<u16>), // stored as raw u16 bits (half::f16)
    BF16(Vec<u16>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
}

impl TensorStorage {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Data type of this storage.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::F64(_) => DType::F64,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::Bool(_) => DType::Bool,
        }
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len() * self.dtype().size_bytes()
    }

    /// Convert to f32 values (for interop with existing Tensor).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::F32(v) => v.clone(),
            Self::F64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I16(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::I64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U8(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U16(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U32(v) => v.iter().map(|&x| x as f32).collect(),
            Self::U64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
            Self::F16(v) => v
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect(),
            Self::BF16(v) => v
                .iter()
                .map(|&bits| half::bf16::from_bits(bits).to_f32())
                .collect(),
        }
    }

    /// Create from f32 values.
    pub fn from_f32(data: Vec<f32>) -> Self {
        Self::F32(data)
    }
}

/// A multi-dtype tensor with shape and typed storage.
/// This complements the existing `Tensor` (which is f32-only) for advanced use cases.
#[derive(Debug, Clone)]
pub struct TypedTensor {
    pub storage: TensorStorage,
    pub shape: Vec<usize>,
}

impl TypedTensor {
    pub fn new(storage: TensorStorage, shape: Vec<usize>) -> Self {
        debug_assert_eq!(storage.len(), shape.iter().product::<usize>());
        Self { storage, shape }
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn numel(&self) -> usize {
        self.storage.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Convert to the existing f32-only Tensor.
    pub fn to_tensor(&self) -> crate::tensor::Tensor {
        crate::tensor::Tensor::new(self.storage.to_f32_vec(), self.shape.clone())
    }

    /// Create from an existing f32-only Tensor.
    pub fn from_tensor(t: &crate::tensor::Tensor) -> Self {
        Self::new(TensorStorage::F32(t.data.clone()), t.shape.clone())
    }

    /// Create a zeros tensor of the given dtype and shape.
    pub fn zeros(dtype: DType, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let storage = match dtype {
            DType::F32 => TensorStorage::F32(vec![0.0; n]),
            DType::F64 => TensorStorage::F64(vec![0.0; n]),
            DType::I8 => TensorStorage::I8(vec![0; n]),
            DType::I16 => TensorStorage::I16(vec![0; n]),
            DType::I32 => TensorStorage::I32(vec![0; n]),
            DType::I64 => TensorStorage::I64(vec![0; n]),
            DType::U8 => TensorStorage::U8(vec![0; n]),
            DType::U16 => TensorStorage::U16(vec![0; n]),
            DType::U32 => TensorStorage::U32(vec![0; n]),
            DType::U64 => TensorStorage::U64(vec![0; n]),
            DType::Bool => TensorStorage::Bool(vec![false; n]),
            DType::F16 => TensorStorage::F16(vec![0; n]),
            DType::BF16 => TensorStorage::BF16(vec![0; n]),
        };
        Self::new(storage, shape.to_vec())
    }

    /// Convert to f16 storage (for mixed-precision inference).
    /// Useful for storing activations in half precision to save memory.
    pub fn to_f16(&self) -> Self {
        self.cast(DType::F16)
    }

    /// Convert to f32 for computation.
    pub fn to_f32(&self) -> Self {
        self.cast(DType::F32)
    }

    /// Check if this tensor is in half precision.
    pub fn is_half(&self) -> bool {
        matches!(self.dtype(), DType::F16 | DType::BF16)
    }

    /// Cast to another dtype.
    pub fn cast(&self, target: DType) -> Self {
        if self.dtype() == target {
            return self.clone();
        }
        // Convert through f32 as intermediate (simple but lossy for I64/U64)
        let f32_data = self.storage.to_f32_vec();
        let storage = match target {
            DType::F32 => TensorStorage::F32(f32_data),
            DType::F64 => TensorStorage::F64(f32_data.iter().map(|&x| x as f64).collect()),
            DType::I32 => TensorStorage::I32(f32_data.iter().map(|&x| x as i32).collect()),
            DType::I64 => TensorStorage::I64(f32_data.iter().map(|&x| x as i64).collect()),
            DType::I8 => TensorStorage::I8(f32_data.iter().map(|&x| x as i8).collect()),
            DType::I16 => TensorStorage::I16(f32_data.iter().map(|&x| x as i16).collect()),
            DType::U8 => TensorStorage::U8(f32_data.iter().map(|&x| x as u8).collect()),
            DType::U16 => TensorStorage::U16(f32_data.iter().map(|&x| x as u16).collect()),
            DType::U32 => TensorStorage::U32(f32_data.iter().map(|&x| x as u32).collect()),
            DType::U64 => TensorStorage::U64(f32_data.iter().map(|&x| x as u64).collect()),
            DType::Bool => TensorStorage::Bool(f32_data.iter().map(|&x| x != 0.0).collect()),
            DType::F16 => TensorStorage::F16(
                f32_data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect(),
            ),
            DType::BF16 => TensorStorage::BF16(
                f32_data
                    .iter()
                    .map(|&x| half::bf16::from_f32(x).to_bits())
                    .collect(),
            ),
        };
        Self::new(storage, self.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(DType::Bool.size_bytes(), 1);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I16.size_bytes(), 2);
        assert_eq!(DType::U16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::U32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
        assert_eq!(DType::I64.size_bytes(), 8);
        assert_eq!(DType::U64.size_bytes(), 8);
    }

    #[test]
    fn test_dtype_from_onnx() {
        let all = [
            DType::F32,
            DType::U8,
            DType::I8,
            DType::U16,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::Bool,
            DType::F16,
            DType::F64,
            DType::U32,
            DType::U64,
            DType::BF16,
        ];
        for dt in &all {
            let onnx_val = dt.to_onnx();
            let roundtrip = DType::from_onnx(onnx_val);
            assert_eq!(roundtrip, Some(*dt), "roundtrip failed for {:?}", dt);
        }
        // Unknown type returns None
        assert_eq!(DType::from_onnx(999), None);
        assert_eq!(DType::from_onnx(0), None);
        assert_eq!(DType::from_onnx(8), None);
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn test_promote_same() {
        let all = [DType::F32, DType::F64, DType::I32, DType::U8, DType::Bool];
        for dt in &all {
            assert_eq!(promote(*dt, *dt), *dt);
        }
    }

    #[test]
    fn test_promote_float_int() {
        // Float wins over integer
        assert_eq!(promote(DType::F32, DType::I32), DType::F32);
        assert_eq!(promote(DType::I64, DType::F64), DType::F64);
        // F16 + int => promoted to F32
        assert_eq!(promote(DType::F16, DType::I32), DType::F32);
        assert_eq!(promote(DType::BF16, DType::U8), DType::F32);
    }

    #[test]
    fn test_promote_float_float() {
        assert_eq!(promote(DType::F32, DType::F64), DType::F64);
        assert_eq!(promote(DType::F64, DType::F32), DType::F64);
        assert_eq!(promote(DType::F16, DType::F32), DType::F32);
        assert_eq!(promote(DType::F16, DType::BF16), DType::F16); // same size, first wins
    }

    #[test]
    fn test_storage_len() {
        let s = TensorStorage::F32(vec![1.0, 2.0, 3.0]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());

        let empty = TensorStorage::I32(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let bools = TensorStorage::Bool(vec![true, false]);
        assert_eq!(bools.len(), 2);
    }

    #[test]
    fn test_storage_to_f32() {
        // F32 passthrough
        let s = TensorStorage::F32(vec![1.0, 2.5]);
        assert_eq!(s.to_f32_vec(), vec![1.0, 2.5]);

        // F64
        let s = TensorStorage::F64(vec![3.0, 4.0]);
        assert_eq!(s.to_f32_vec(), vec![3.0, 4.0]);

        // I32
        let s = TensorStorage::I32(vec![-1, 0, 42]);
        assert_eq!(s.to_f32_vec(), vec![-1.0, 0.0, 42.0]);

        // U8
        let s = TensorStorage::U8(vec![0, 128, 255]);
        assert_eq!(s.to_f32_vec(), vec![0.0, 128.0, 255.0]);

        // Bool
        let s = TensorStorage::Bool(vec![true, false, true]);
        assert_eq!(s.to_f32_vec(), vec![1.0, 0.0, 1.0]);

        // I8
        let s = TensorStorage::I8(vec![-128, 0, 127]);
        assert_eq!(s.to_f32_vec(), vec![-128.0, 0.0, 127.0]);

        // I16
        let s = TensorStorage::I16(vec![-1, 0, 1]);
        assert_eq!(s.to_f32_vec(), vec![-1.0, 0.0, 1.0]);

        // I64
        let s = TensorStorage::I64(vec![100, -200]);
        assert_eq!(s.to_f32_vec(), vec![100.0, -200.0]);

        // U16
        let s = TensorStorage::U16(vec![0, 65535]);
        assert_eq!(s.to_f32_vec(), vec![0.0, 65535.0]);

        // U32
        let s = TensorStorage::U32(vec![0, 1000]);
        assert_eq!(s.to_f32_vec(), vec![0.0, 1000.0]);

        // U64
        let s = TensorStorage::U64(vec![0, 42]);
        assert_eq!(s.to_f32_vec(), vec![0.0, 42.0]);
    }

    #[test]
    fn test_typed_tensor_zeros() {
        let all_dtypes = [
            DType::F32,
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
            DType::F16,
            DType::BF16,
        ];
        for dt in &all_dtypes {
            let t = TypedTensor::zeros(*dt, &[2, 3]);
            assert_eq!(t.dtype(), *dt);
            assert_eq!(t.numel(), 6);
            assert_eq!(t.ndim(), 2);
            assert_eq!(t.shape, vec![2, 3]);
            // All values should convert to 0.0 in f32
            let f32_vals = t.storage.to_f32_vec();
            for v in &f32_vals {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn test_typed_tensor_cast() {
        let src = TypedTensor::new(TensorStorage::I32(vec![1, 2, 3, 4]), vec![2, 2]);

        // Cast to F32
        let f32_t = src.cast(DType::F32);
        assert_eq!(f32_t.dtype(), DType::F32);
        assert_eq!(f32_t.storage.to_f32_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        // Cast to I64
        let i64_t = src.cast(DType::I64);
        assert_eq!(i64_t.dtype(), DType::I64);
        assert_eq!(i64_t.storage.to_f32_vec(), vec![1.0, 2.0, 3.0, 4.0]);

        // Cast to Bool
        let bool_t = TypedTensor::new(TensorStorage::F32(vec![0.0, 1.5, -1.0]), vec![3]);
        let b = bool_t.cast(DType::Bool);
        assert_eq!(b.dtype(), DType::Bool);
        assert_eq!(b.storage.to_f32_vec(), vec![0.0, 1.0, 1.0]);

        // Cast to same type returns clone
        let same = src.cast(DType::I32);
        assert_eq!(same.dtype(), DType::I32);
        assert_eq!(same.storage.to_f32_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_typed_tensor_roundtrip() {
        let original = crate::tensor::Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let typed = TypedTensor::from_tensor(&original);
        assert_eq!(typed.dtype(), DType::F32);
        assert_eq!(typed.shape, vec![2, 3]);

        let back = typed.to_tensor();
        assert_eq!(back.data, original.data);
        assert_eq!(back.shape, original.shape);
    }

    #[test]
    fn test_typed_tensor_f16() {
        // Create F16 storage from known values
        let f32_vals = vec![1.0f32, 0.5, -2.0, 0.0];
        let f16_bits: Vec<u16> = f32_vals
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let t = TypedTensor::new(TensorStorage::F16(f16_bits), vec![4]);
        assert_eq!(t.dtype(), DType::F16);
        assert_eq!(t.numel(), 4);

        // Convert back to f32 and check
        let result = t.storage.to_f32_vec();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - 0.5).abs() < 1e-3);
        assert!((result[2] - (-2.0)).abs() < 1e-3);
        assert!((result[3] - 0.0).abs() < 1e-3);

        // Cast from F32 to F16 and back
        let src = TypedTensor::new(TensorStorage::F32(vec![3.14, -1.0]), vec![2]);
        let as_f16 = src.cast(DType::F16);
        assert_eq!(as_f16.dtype(), DType::F16);
        let back = as_f16.cast(DType::F32);
        assert_eq!(back.dtype(), DType::F32);
        let vals = back.storage.to_f32_vec();
        assert!((vals[0] - 3.14).abs() < 0.01);
        assert!((vals[1] - (-1.0)).abs() < 1e-3);
    }
}
