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
    /// Only one half of a LoRA adapter pair (`A`/`B` or `down`/`up`) was found
    /// for a module, which usually means a corrupt or mis-named adapter file.
    PartialAdapterPair {
        /// The base module stem (e.g. `model.layers.0.self_attn.q_proj`).
        module: String,
        /// The adapter tensor that *was* present.
        found: String,
        /// The adapter tensor that was expected but missing.
        missing: String,
    },
    /// One or more adapter tensors in the LoRA file were never consumed by the
    /// merge, which usually indicates a naming-convention mismatch that would
    /// otherwise silently produce an unchanged copy of the base model.
    UnusedAdapterTensors(Vec<String>),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::Safetensors(e) => write!(f, "safetensors error: {e}"),
            MergeError::Shape(m) => write!(f, "shape mismatch: {m}"),
            MergeError::MissingTensor(n) => write!(f, "missing base tensor: {n}"),
            MergeError::PartialAdapterPair {
                module,
                found,
                missing,
            } => write!(
                f,
                "module '{module}' has a partial LoRA adapter: found '{found}' but '{missing}' is missing"
            ),
            MergeError::UnusedAdapterTensors(names) => write!(
                f,
                "{} adapter tensor(s) were not consumed by the merge (naming mismatch?): {}",
                names.len(),
                names.join(", ")
            ),
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
