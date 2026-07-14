//! Error type for safetensors read/write operations.

use std::fmt;

/// Errors that can occur while reading or writing safetensors files.
#[derive(Debug)]
#[non_exhaustive]
pub enum SafetensorsError {
    /// An I/O error, most commonly from opening or mapping a file.
    Io(std::io::Error),
    /// A JSON (de)serialisation error in the safetensors header.
    Json(serde_json::Error),
    /// The 8-byte header-length prefix or the header itself is malformed.
    InvalidHeader(String),
    /// A requested tensor name does not exist in the file.
    TensorNotFound(String),
    /// A tensor's supplied byte buffer does not match its declared shape/dtype.
    BadDataLength {
        /// Tensor name.
        name: String,
        /// Expected number of bytes.
        expected: usize,
        /// Actual number of bytes provided.
        actual: usize,
    },
    /// A dtype name in the header is not supported by this crate.
    UnsupportedDtype(String),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetensorsError::Io(e) => write!(f, "I/O error: {e}"),
            SafetensorsError::Json(e) => write!(f, "header JSON error: {e}"),
            SafetensorsError::InvalidHeader(m) => write!(f, "invalid safetensors header: {m}"),
            SafetensorsError::TensorNotFound(n) => write!(f, "tensor not found: {n}"),
            SafetensorsError::BadDataLength {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{name}' has {actual} bytes but its shape/dtype require {expected}"
            ),
            SafetensorsError::UnsupportedDtype(d) => write!(f, "unsupported dtype: {d}"),
        }
    }
}

impl std::error::Error for SafetensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SafetensorsError::Io(e) => Some(e),
            SafetensorsError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        SafetensorsError::Io(e)
    }
}

impl From<serde_json::Error> for SafetensorsError {
    fn from(e: serde_json::Error) -> Self {
        SafetensorsError::Json(e)
    }
}
