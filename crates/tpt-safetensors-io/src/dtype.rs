//! Safetensors element types and their byte widths.

use serde::{Deserialize, Serialize};

/// The set of tensor element types understood by this crate.
///
/// The `Serialize`/`Deserialize` implementations map directly to the
/// upper-case strings used inside a safetensors header (e.g. `"F32"`,
/// `"BF16"`, `"I64"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[non_exhaustive]
pub enum Dtype {
    /// 16-bit floating point (`half`).
    F16,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// 16-bit brain floating point.
    BF16,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Boolean, stored as a single byte per element.
    BOOL,
    /// 8-bit floating point, `e4m3` layout (1 sign, 4 exponent, 3 mantissa).
    #[serde(rename = "F8_E4M3")]
    F8E4M3,
    /// 8-bit floating point, `e5m2` layout (1 sign, 5 exponent, 2 mantissa).
    #[serde(rename = "F8_E5M2")]
    F8E5M2,
}

impl Dtype {
    /// Number of bytes occupied by a single element of this dtype.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::I8 | Dtype::U8 | Dtype::BOOL | Dtype::F8E4M3 | Dtype::F8E5M2 => 1,
        }
    }

    /// The upper-case name of this dtype as it appears in a safetensors header.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::BF16 => "BF16",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
            Dtype::U8 => "U8",
            Dtype::U16 => "U16",
            Dtype::U32 => "U32",
            Dtype::U64 => "U64",
            Dtype::BOOL => "BOOL",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E5M2 => "F8_E5M2",
        }
    }
}
