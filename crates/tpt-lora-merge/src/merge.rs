//! LoRA merging logic.

use std::collections::HashSet;

use serde_json::{Map, Value};
use tpt_safetensors_io::{Dtype, SafetensorsBuilder, SafetensorsFile};

use crate::error::MergeError;

/// Suffix pairs identifying a LoRA adapter's down/up (A/B) matrices, covering
/// the common HF PEFT (`lora_A`/`lora_B`) and Kohya (`lora_down`/`lora_up`)
/// naming conventions.
const ADAPTER_SUFFIXES: &[(&str, &str)] = &[
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
];

/// A row-major 2-D matrix of `f32` values.
#[derive(Default)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "expected {}×{}={} elements, got {}",
                rows,
                cols,
                rows * cols,
                data.len()
            ));
        }
        Ok(Mat { rows, cols, data })
    }

    /// Row-major matrix multiply: `self` (m×k) @ `rhs` (k×n) → m×n.
    fn matmul(&self, rhs: &Mat) -> Mat {
        let (m, k, n) = (self.rows, self.cols, rhs.cols);
        debug_assert_eq!(k, rhs.rows, "inner dimensions must match");
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for l in 0..k {
                let a = self.data[i * k + l];
                for j in 0..n {
                    data[i * n + j] += a * rhs.data[l * n + j];
                }
            }
        }
        Mat {
            rows: m,
            cols: n,
            data,
        }
    }

    fn add_assign(&mut self, rhs: &Mat) {
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a += b;
        }
    }

    fn scale(&mut self, s: f32) {
        for v in &mut self.data {
            *v *= s;
        }
    }
}

/// Merges a single linear weight: `result = base + scale * (B @ A)`.
///
/// * `base` — flat row-major data of shape `(out_dim, in_dim)`.
/// * `lora_a` — flat row-major data of shape `(rank, in_dim)`.
/// * `lora_b` — flat row-major data of shape `(out_dim, rank)`.
/// * `out_dim`, `rank`, `in_dim` — the explicit dimensions.
///
/// Returns the merged values in the same flat row-major layout as `base`.
#[must_use]
pub fn merge_linear(
    base: &[f32],
    lora_a: &[f32],
    lora_b: &[f32],
    out_dim: usize,
    rank: usize,
    in_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let a = Mat {
        rows: rank,
        cols: in_dim,
        data: lora_a.to_vec(),
    };
    let b = Mat {
        rows: out_dim,
        cols: rank,
        data: lora_b.to_vec(),
    };
    let mut delta = b.matmul(&a);
    delta.scale(scale);
    base.iter().zip(&delta.data).map(|(x, d)| x + d).collect()
}

/// The result of a LoRA merge: a set of `(name, dtype, shape, bytes)` tensors
/// ready to be written back to a safetensors file.
///
/// Merged tensors preserve the *base* tensor's dtype (so a BF16 base yields a
/// BF16 output rather than a doubled-size F32 one). Tensors that were copied
/// through unchanged keep their original dtype and raw bytes.
#[derive(Debug, Clone, Default)]
pub struct MergedWeights {
    tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    metadata: Map<String, Value>,
    merged_modules: Vec<String>,
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

    /// The names of the base modules that actually had an adapter merged in.
    #[must_use]
    pub fn merged_modules(&self) -> &[String] {
        &self.merged_modules
    }

    /// The `__metadata__` table that will be written out (base provenance plus
    /// merge annotations).
    #[must_use]
    pub fn metadata(&self) -> &Map<String, Value> {
        &self.metadata
    }

    /// Serialises the merged weights into a safetensors file at `path`.
    ///
    /// # Errors
    /// Returns a [`MergeError::Safetensors`] on any I/O or serialisation
    /// failure.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<(), MergeError> {
        let mut builder = SafetensorsBuilder::new();
        for (key, value) in &self.metadata {
            builder.add_metadata(key.clone(), value.clone());
        }
        for (name, dtype, shape, bytes) in &self.tensors {
            builder.add_tensor(name.clone(), *dtype, shape.clone(), bytes.clone())?;
        }
        builder.write_to_file(path)?;
        Ok(())
    }
}

/// A resolved adapter contribution for one base module: the summed delta
/// `Σ alpha_scale_i * (B_i @ A_i)` (before the user scale is applied), the
/// dimensions it expects, and the tensor names it consumed.
struct ResolvedDelta {
    delta: Mat,
    out_dim: usize,
    in_dim: usize,
    consumed: Vec<String>,
}

/// Merges every LoRA adapter pair found in `lora` into the matching base tensor
/// of `base`, returning the combined [`MergedWeights`].
///
/// This is a convenience wrapper over [`merge_loras`] for the single-adapter
/// case. See that function for the full semantics.
///
/// # Errors
/// Returns a [`MergeError`] if a referenced tensor is missing, has an
/// incompatible shape, forms a partial adapter pair, or is left unconsumed.
pub fn merge_lora(
    base: &SafetensorsFile,
    lora: &SafetensorsFile,
    scale: f32,
) -> Result<MergedWeights, MergeError> {
    merge_loras(base, &[(lora, scale)])
}

/// Merges one or more LoRA adapters into `base` as a weighted sum of their
/// deltas: `result = base + Σ scale_i * (B_i @ A_i)`.
///
/// Base weights named `<module>.weight` are paired with adapter tensors using
/// either the HF PEFT (`.lora_A.weight`/`.lora_B.weight`) or Kohya
/// (`.lora_down.weight`/`.lora_up.weight`) convention; a per-module `.alpha`
/// tensor, when present, scales that module's delta by `alpha / r`.
///
/// * Non-2-D base tensors (and 2-D tensors with no adapter in any file) are
///   copied through unchanged, preserving their dtype and bytes.
/// * Merged tensors are re-encoded to the base tensor's dtype.
/// * A partial adapter pair (only one of A/B present) is an error.
/// * Any adapter tensor left unconsumed across all files is an error.
///
/// # Errors
/// Returns a [`MergeError`] on missing/mismatched/partial/unused tensors.
pub fn merge_loras(
    base: &SafetensorsFile,
    adapters: &[(&SafetensorsFile, f32)],
) -> Result<MergedWeights, MergeError> {
    let mut out = MergedWeights::default();

    // Track which adapter tensors we actually consume, per adapter file.
    let mut consumed: Vec<HashSet<String>> = adapters.iter().map(|_| HashSet::new()).collect();

    for name in base.tensor_names() {
        let view = base
            .get_tensor(name)
            .ok_or_else(|| MergeError::MissingTensor(name.to_string()))?;
        let shape = view.shape.clone();

        // Only 2-D weights can carry a LoRA adapter.
        if shape.len() != 2 {
            out.tensors
                .push((name.to_string(), view.dtype, shape, view.data.to_vec()));
            continue;
        }
        let (out_dim, in_dim) = (shape[0], shape[1]);
        let stem = name.strip_suffix(".weight").unwrap_or(name);

        // Accumulate deltas from every adapter that targets this module.
        let mut delta: Option<Mat> = None;
        for (idx, (lora, user_scale)) in adapters.iter().enumerate() {
            let Some(mut resolved) = resolve_adapter(lora, stem)? else {
                continue;
            };

            if resolved.in_dim != in_dim || resolved.out_dim != out_dim {
                return Err(MergeError::Shape(format!(
                    "LoRA {name} shape mismatch: base ({out_dim},{in_dim}), adapter ({},{})",
                    resolved.out_dim, resolved.in_dim
                )));
            }

            resolved.delta.scale(*user_scale);
            delta = Some(match delta {
                Some(mut acc) => {
                    acc.add_assign(&resolved.delta);
                    acc
                }
                None => resolved.delta,
            });
            for c in resolved.consumed {
                consumed[idx].insert(c);
            }
        }

        let Some(delta) = delta else {
            // No adapter for this weight in any file: copy through unchanged.
            out.tensors
                .push((name.to_string(), view.dtype, shape, view.data.to_vec()));
            continue;
        };

        let base_f32 = view.to_f32()?;
        let merged: Vec<f32> = base_f32
            .iter()
            .zip(&delta.data)
            .map(|(b, d)| b + d)
            .collect();

        // Re-encode to the base dtype so we don't silently double the size.
        let (dtype, bytes) = encode_to_dtype(view.dtype, &merged);
        out.tensors.push((name.to_string(), dtype, shape, bytes));
        out.merged_modules.push(name.to_string());
    }

    // Every adapter tensor must have been used; otherwise the caller likely has
    // a naming mismatch that would silently no-op the merge.
    let mut unused = Vec::new();
    for ((lora, _), used) in adapters.iter().zip(&consumed) {
        for tname in lora.tensor_names() {
            if is_adapter_tensor(tname) && !used.contains(tname) {
                unused.push(tname.to_string());
            }
        }
    }
    if !unused.is_empty() {
        unused.sort();
        return Err(MergeError::UnusedAdapterTensors(unused));
    }

    // Preserve base provenance metadata, then annotate the merge.
    out.metadata = base.metadata().clone();
    out.metadata
        .insert("merged_by".into(), Value::String("tpt-lora-merge".into()));
    out.metadata.insert(
        "merged_adapters".into(),
        Value::Number((adapters.len() as u64).into()),
    );

    Ok(out)
}

/// Returns true if `name` looks like a LoRA adapter tensor (A/B/down/up/alpha).
fn is_adapter_tensor(name: &str) -> bool {
    name.contains(".lora_A")
        || name.contains(".lora_B")
        || name.contains(".lora_down")
        || name.contains(".lora_up")
        || name.ends_with(".alpha")
        || name.ends_with(".lora_alpha")
}

/// Resolves the summed adapter delta for `stem` within a single LoRA file,
/// trying the fixed HF PEFT / Kohya suffixes as well as PEFT named-adapter
/// tensors (`.lora_A.<name>.weight`).
///
/// Returns `Ok(None)` if this file has no adapter for the module, and an error
/// if it has exactly one half of a pair or inconsistent dimensions.
fn resolve_adapter(
    lora: &SafetensorsFile,
    stem: &str,
) -> Result<Option<ResolvedDelta>, MergeError> {
    let mut acc = DeltaAcc::default();

    // Per-module alpha scale (Kohya): scale = alpha / r, applied to every pair.
    let alpha_name = format!("{stem}.alpha");
    let alpha_value = lora
        .get_tensor(&alpha_name)
        .and_then(|v| v.to_f32().ok())
        .and_then(|vals| vals.first().copied());

    // 1. Fixed-suffix conventions.
    for (a_suffix, b_suffix) in ADAPTER_SUFFIXES {
        let a_name = format!("{stem}{a_suffix}");
        let b_name = format!("{stem}{b_suffix}");
        match (lora.get_tensor(&a_name), lora.get_tensor(&b_name)) {
            (Some(_), Some(_)) => {
                acc.add_pair(lora, &a_name, &b_name, alpha_value)?;
            }
            (Some(_), None) => {
                return Err(MergeError::PartialAdapterPair {
                    module: stem.to_string(),
                    found: a_name,
                    missing: b_name,
                })
            }
            (None, Some(_)) => {
                return Err(MergeError::PartialAdapterPair {
                    module: stem.to_string(),
                    found: b_name,
                    missing: a_name,
                })
            }
            (None, None) => {}
        }
    }

    // 2. PEFT named adapters: `{stem}.lora_A.<name>.weight`.
    let a_prefix = format!("{stem}.lora_A.");
    let named: Vec<String> = lora
        .tensor_names()
        .filter(|n| {
            n.starts_with(&a_prefix)
                && n.ends_with(".weight")
                && n.len() > a_prefix.len() + ".weight".len()
        })
        .map(str::to_string)
        .collect();
    for a_name in named {
        let inner = &a_name[a_prefix.len()..a_name.len() - ".weight".len()];
        let b_name = format!("{stem}.lora_B.{inner}.weight");
        if lora.get_tensor(&b_name).is_none() {
            return Err(MergeError::PartialAdapterPair {
                module: stem.to_string(),
                found: a_name,
                missing: b_name,
            });
        }
        acc.add_pair(lora, &a_name, &b_name, alpha_value)?;
    }

    if acc.delta.is_none() {
        return Ok(None);
    }
    if alpha_value.is_some() {
        acc.consumed.push(alpha_name);
    }
    Ok(Some(ResolvedDelta {
        delta: acc.delta.expect("checked above"),
        out_dim: acc.out_dim,
        in_dim: acc.in_dim,
        consumed: acc.consumed,
    }))
}

/// Accumulates adapter pair deltas while tracking dimensions and consumed names.
#[derive(Default)]
struct DeltaAcc {
    delta: Option<Mat>,
    out_dim: usize,
    in_dim: usize,
    consumed: Vec<String>,
}


impl DeltaAcc {
    fn add_pair(
        &mut self,
        lora: &SafetensorsFile,
        a_name: &str,
        b_name: &str,
        alpha_value: Option<f32>,
    ) -> Result<(), MergeError> {
        let a_view = lora
            .get_tensor(a_name)
            .ok_or_else(|| MergeError::MissingTensor(a_name.to_string()))?;
        let b_view = lora
            .get_tensor(b_name)
            .ok_or_else(|| MergeError::MissingTensor(b_name.to_string()))?;
        let a = to_2d(&a_view.shape, a_view.to_f32()?, a_name)?;
        let b = to_2d(&b_view.shape, b_view.to_f32()?, b_name)?;

        let (r, a_in) = (a.rows, a.cols);
        let (b_out, b_r) = (b.rows, b.cols);
        if r != b_r {
            return Err(MergeError::Shape(format!(
                "{a_name}/{b_name} inner ranks disagree: A rows {r}, B cols {b_r}"
            )));
        }

        // Kohya per-layer alpha scaling (alpha / r), defaulting to 1.0.
        let scale = match alpha_value {
            Some(alpha) if r > 0 => alpha / r as f32,
            _ => 1.0,
        };
        let mut contribution = b.matmul(&a);
        contribution.scale(scale);

        match &mut self.delta {
            Some(existing) => {
                if self.out_dim != b_out || self.in_dim != a_in {
                    return Err(MergeError::Shape(format!(
                        "{a_name}/{b_name} dims ({b_out},{a_in}) disagree with earlier adapter dims ({},{})",
                        self.out_dim, self.in_dim
                    )));
                }
                existing.add_assign(&contribution);
            }
            None => {
                self.out_dim = b_out;
                self.in_dim = a_in;
                self.delta = Some(contribution);
            }
        }
        self.consumed.push(a_name.to_string());
        self.consumed.push(b_name.to_string());
        Ok(())
    }
}

fn to_2d(shape: &[usize], data: Vec<f32>, name: &str) -> Result<Mat, MergeError> {
    if shape.len() != 2 {
        return Err(MergeError::Shape(format!("{name} must be 2-D")));
    }
    Mat::from_vec(shape[0], shape[1], data).map_err(MergeError::Shape)
}

/// Encodes merged `f32` values back into the base tensor's dtype, falling back
/// to `F32` for dtypes that cannot losslessly represent the result.
fn encode_to_dtype(dtype: Dtype, values: &[f32]) -> (Dtype, Vec<u8>) {
    match dtype {
        Dtype::F16 => {
            let mut bytes = Vec::with_capacity(values.len() * 2);
            for &v in values {
                bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
            (Dtype::F16, bytes)
        }
        Dtype::BF16 => {
            let mut bytes = Vec::with_capacity(values.len() * 2);
            for &v in values {
                bytes.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
            (Dtype::BF16, bytes)
        }
        Dtype::F64 => {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for &v in values {
                bytes.extend_from_slice(&f64::from(v).to_le_bytes());
            }
            (Dtype::F64, bytes)
        }
        _ => {
            // F32 and any non-float base fall back to F32 storage.
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for &v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (Dtype::F32, bytes)
        }
    }
}

/// Converts `f32` to IEEE-754 `binary16`, rounding to nearest-even.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1f {
        // Overflow to infinity.
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        // Subnormal or zero.
        if unbiased < -10 {
            return sign;
        }
        let mant = mant | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let half = mant >> shift;
        // Round to nearest-even (all arithmetic in u32).
        let round_bit = (mant >> (shift - 1)) & 1;
        let sticky = u32::from(mant & ((1u32 << (shift - 1)) - 1) != 0);
        let rounded = half + (round_bit & (sticky | (half & 1)));
        return sign | rounded as u16;
    }
    let half_exp = (unbiased as u16) << 10;
    let half_mant = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let sticky = u32::from(mant & 0x0fff != 0);
    let base = sign | half_exp | half_mant;
    base + (round_bit & (sticky | (u32::from(half_mant) & 1))) as u16
}

/// Converts `f32` to `bfloat16`, rounding to nearest-even.
fn f32_to_bf16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}
