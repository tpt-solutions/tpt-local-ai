//! Error type for LoRA merging.

use tpt_safetensors_io::SafetensorsError;

/// Errors that can occur while merging LoRA weights.
#[derive(Debug)]
#[non_exhaustive]
pub enum MergeError {
    /// An error from the underlying safetensors reader/writer.
    Safetensors(SafetensorsError),
    /// A tensor had an unexpected shape for the merge operation.
    Shape(String),
    /// A base tensor referenced by a LoRA adapter is missing from the base file.
    MissingTensor(String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::Safetensors(e) => write!(f, "safetensors error: {e}"),
            MergeError::Shape(m) => write!(f, "shape mismatch: {m}"),
            MergeError::MissingTensor(n) => write!(f, "missing base tensor: {n}"),
        }
    }
}

impl std::error::Error for MergeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MergeError::Safetensors(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SafetensorsError> for MergeError {
    fn from(e: SafetensorsError) -> Self {
        MergeError::Safetensors(e)
    }
}
