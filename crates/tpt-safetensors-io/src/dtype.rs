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
    /// Boolean, stored as a single byte per element.
    BOOL,
}

impl Dtype {
    /// Number of bytes occupied by a single element of this dtype.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::I8 | Dtype::U8 | Dtype::BOOL => 1,
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
            Dtype::BOOL => "BOOL",
        }
    }
}
