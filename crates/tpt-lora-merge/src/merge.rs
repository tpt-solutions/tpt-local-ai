//! LoRA merging logic.

use ndarray::{Array2, ArrayView2};
use tpt_safetensors_io::{SafetensorsBuilder, SafetensorsFile};

use crate::error::MergeError;

/// Suffix marking a LoRA adapter's `A` (down-projection) matrix.
const LORA_A_SUFFIX: &str = ".lora_A.weight";
/// Suffix marking a LoRA adapter's `B` (up-projection) matrix.
const LORA_B_SUFFIX: &str = ".lora_B.weight";

/// Merges a single linear weight: `result = base + scale * (B @ A)`.
///
/// * `base` has shape `(out, in)`.
/// * `lora_a` has shape `(r, in)`.
/// * `lora_b` has shape `(out, r)`.
///
/// The `scale` argument folds in the `(alpha / r)` factor (and any additional
/// user scaling) so that higher scales blend more of the adapter in.
#[must_use]
pub fn merge_linear(
    base: ArrayView2<f32>,
    lora_a: ArrayView2<f32>,
    lora_b: ArrayView2<f32>,
    scale: f32,
) -> Array2<f32> {
    &base + &(lora_b.dot(&lora_a) * scale)
}

/// The result of a LoRA merge: a set of `(name, shape, f32 data)` tensors ready
/// to be written back to a safetensors file.
#[derive(Debug, Clone, Default)]
pub struct MergedWeights {
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl MergedWeights {
    /// Number of tensors in the merged result.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the merged result contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Serialises the merged weights into a safetensors file at `path`.
    ///
    /// # Errors
    /// Returns a [`MergeError::Safetensors`] on any I/O or serialisation
    /// failure.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<(), MergeError> {
        let mut builder = SafetensorsBuilder::new();
        builder.add_metadata("merged_by", serde_json::json!("tpt-lora-merge"));
        for (name, shape, data) in &self.tensors {
            builder.add_f32(name.clone(), shape.clone(), data.clone())?;
        }
        builder.write_to_file(path)?;
        Ok(())
    }
}

/// Merges every LoRA adapter pair found in `lora` into the matching base tensor
/// of `base`, returning the combined [`MergedWeights`].
///
/// Base tensors named `<module>.weight` are paired with
/// `<module>.lora_A.weight` / `<module>.lora_B.weight`. Tensors without a
/// matching adapter are copied through unchanged.
///
/// # Errors
/// Returns a [`MergeError`] if a referenced tensor is missing, has an
/// incompatible shape, or cannot be decoded to `f32`.
pub fn merge_lora(
    base: &SafetensorsFile,
    lora: &SafetensorsFile,
    scale: f32,
) -> Result<MergedWeights, MergeError> {
    let mut out = MergedWeights::default();

    for name in base.tensor_names() {
        let view = base
            .get_tensor(name)
            .ok_or_else(|| MergeError::MissingTensor(name.to_string()))?;
        let base_vec = view.to_f32()?;
        let shape = view.shape.clone();

        // Only 2-D weights can carry a LoRA adapter.
        if shape.len() != 2 {
            out.tensors.push((name.to_string(), shape, base_vec));
            continue;
        }
        let (out_dim, in_dim) = (shape[0], shape[1]);

        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let a_name = format!("{stem}{LORA_A_SUFFIX}");
        let b_name = format!("{stem}{LORA_B_SUFFIX}");

        let (Some(a_view), Some(b_view)) = (lora.get_tensor(&a_name), lora.get_tensor(&b_name))
        else {
            // No adapter for this weight: copy the base through.
            out.tensors.push((name.to_string(), shape, base_vec));
            continue;
        };

        let a_vec = a_view.to_f32()?;
        let b_vec = b_view.to_f32()?;
        let a_shape = a_view.shape.clone();
        let b_shape = b_view.shape.clone();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(MergeError::Shape(format!(
                "{a_name}/{b_name} must both be 2-D"
            )));
        }
        let (r, a_in) = (a_shape[0], a_shape[1]);
        let (b_out, b_r) = (b_shape[0], b_shape[1]);
        if a_in != in_dim || b_out != out_dim || r != b_r {
            return Err(MergeError::Shape(format!(
                "LoRA {name} shape mismatch: base ({out_dim},{in_dim}), A ({r},{a_in}), B ({b_out},{b_r})"
            )));
        }

        let base_arr = Array2::from_shape_vec((out_dim, in_dim), base_vec)
            .map_err(|e| MergeError::Shape(e.to_string()))?;
        let a_arr = Array2::from_shape_vec((r, in_dim), a_vec)
            .map_err(|e| MergeError::Shape(e.to_string()))?;
        let b_arr = Array2::from_shape_vec((out_dim, r), b_vec)
            .map_err(|e| MergeError::Shape(e.to_string()))?;

        let merged = merge_linear(base_arr.view(), a_arr.view(), b_arr.view(), scale);
        out.tensors
            .push((name.to_string(), shape, merged.into_raw_vec_and_offset().0));
    }

    Ok(out)
}
