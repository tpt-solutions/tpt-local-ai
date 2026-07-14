//! Zero-copy, memory-mapped reader for safetensors files.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::{Map, Value};

use crate::dtype::Dtype;
use crate::error::SafetensorsError;

/// A zero-copy view over a single tensor inside a [`SafetensorsFile`].
///
/// The `data` slice is borrowed directly from the underlying memory map, so no
/// copy is performed. Its lifetime is tied to the borrow of the parent file.
pub struct TensorView<'a> {
    /// The element type of the tensor.
    pub dtype: Dtype,
    /// The tensor shape, most-significant dimension first.
    pub shape: Vec<usize>,
    /// The raw little-endian element bytes of the tensor.
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Total number of elements (`shape.iter().product()`).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Interprets the raw bytes as `f32` elements.
    ///
    /// All supported integer and floating-point dtypes are converted to `f32`
    /// (boolean elements become `0.0`/`1.0`).
    ///
    /// # Errors
    /// Returns [`SafetensorsError::UnsupportedDtype`] for `BOOL` tensors or
    /// when the byte length is not an integer multiple of the element size.
    pub fn to_f32(&self) -> Result<Vec<f32>, SafetensorsError> {
        let n = self.numel();
        if self.data.len() != self.dtype.size_bytes() * n {
            return Err(SafetensorsError::BadDataLength {
                name: "<view>".to_string(),
                expected: self.dtype.size_bytes() * n,
                actual: self.data.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            Dtype::F32 => {
                for chunk in self.data.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            Dtype::F16 => {
                for chunk in self.data.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())));
                }
            }
            Dtype::BF16 => {
                for chunk in self.data.chunks_exact(2) {
                    out.push(bf16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())));
                }
            }
            Dtype::F64 => {
                for chunk in self.data.chunks_exact(8) {
                    out.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            Dtype::I8 => {
                for &b in self.data {
                    out.push(f32::from(b as i8));
                }
            }
            Dtype::I16 => {
                for chunk in self.data.chunks_exact(2) {
                    out.push(f32::from(i16::from_le_bytes(chunk.try_into().unwrap())));
                }
            }
            Dtype::I32 => {
                for chunk in self.data.chunks_exact(4) {
                    out.push(i32::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            Dtype::I64 => {
                for chunk in self.data.chunks_exact(8) {
                    out.push((i64::from_le_bytes(chunk.try_into().unwrap())) as f32);
                }
            }
            Dtype::U8 => {
                for &b in self.data {
                    out.push(f32::from(b));
                }
            }
            Dtype::U16 => {
                for chunk in self.data.chunks_exact(2) {
                    out.push(f32::from(u16::from_le_bytes(chunk.try_into().unwrap())));
                }
            }
            Dtype::U32 => {
                for chunk in self.data.chunks_exact(4) {
                    out.push(u32::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            Dtype::U64 => {
                for chunk in self.data.chunks_exact(8) {
                    out.push(u64::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            Dtype::BOOL => {
                return Err(SafetensorsError::UnsupportedDtype(
                    "BOOL -> f32 conversion is not supported".to_string(),
                ));
            }
            Dtype::F8E4M3 | Dtype::F8E5M2 => {
                return Err(SafetensorsError::UnsupportedDtype(format!(
                    "{} -> f32 conversion is not supported",
                    self.dtype.as_str()
                )));
            }
        }
        Ok(out)
    }
}

/// A memory-mapped, read-only safetensors file.
///
/// Construct one with [`SafetensorsFile::open`] and then query tensor names
/// with [`SafetensorsFile::tensor_names`] and read payloads with
/// [`SafetensorsFile::get_tensor`]. Tensor data is borrowed from the map and is
/// therefore zero-copy.
pub struct SafetensorsFile {
    mmap: Mmap,
    data_base: usize,
    tensors: Vec<(String, TensorInfo)>,
    metadata: Map<String, Value>,
}

struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl SafetensorsFile {
    /// Opens `path`, memory-maps it, and parses the safetensors header.
    ///
    /// # Errors
    /// Returns [`SafetensorsError`] if the file cannot be opened, mapped, or if
    /// its header is malformed.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SafetensorsError> {
        let file = File::open(path)?;
        // SAFETY: `Mmap::map` is unsafe because the mapped file may be mutated
        // by another process while we hold the map, which would make the
        // `&[u8]` view observe changing bytes (undefined behavior in the
        // strictest sense). This crate is a read-only viewer over local model
        // checkpoints that the caller controls; we never write through the map
        // and every offset derived from the (untrusted) header is bounds- and
        // overflow-checked in `parse_header` before it is used to slice the
        // map. Callers must not point this at a file another process is
        // concurrently truncating or rewriting.
        let mmap = unsafe { Mmap::map(&file)? };
        let (tensors, data_base, metadata) = parse_header(&mmap)?;
        Ok(Self {
            mmap,
            data_base,
            tensors,
            metadata,
        })
    }

    /// Iterates over the names of every tensor in the file, in header order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    /// Returns a zero-copy [`TensorView`] for the named tensor, if present.
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let info = self
            .tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| i)?;
        let range = self.data_base + info.start..self.data_base + info.end;
        let data = &self.mmap[range];
        Some(TensorView {
            dtype: info.dtype,
            shape: info.shape.clone(),
            data,
        })
    }

    /// The top-level `__metadata__` table from the header, if any.
    #[must_use]
    pub fn metadata(&self) -> &Map<String, Value> {
        &self.metadata
    }

    /// Number of tensors in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

#[allow(clippy::type_complexity)]
fn parse_header(
    bytes: &[u8],
) -> Result<(Vec<(String, TensorInfo)>, usize, Map<String, Value>), SafetensorsError> {
    if bytes.len() < 8 {
        return Err(SafetensorsError::InvalidHeader(
            "file is smaller than the 8-byte length prefix".to_string(),
        ));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    // `8 + header_len` is attacker-controlled and could overflow `usize` on a
    // corrupted header, wrapping to a small value that then passes the bounds
    // check. Use checked arithmetic so a hostile length is rejected outright.
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        SafetensorsError::InvalidHeader("header length overflows usize".to_string())
    })?;
    if bytes.len() < header_end {
        return Err(SafetensorsError::InvalidHeader(
            "declared header length exceeds file size".to_string(),
        ));
    }
    let header_bytes = &bytes[8..header_end];
    let value: Value = serde_json::from_slice(header_bytes)?;
    let obj = value.as_object().ok_or_else(|| {
        SafetensorsError::InvalidHeader("header is not a JSON object".to_string())
    })?;

    let mut tensors = Vec::new();
    let mut metadata = Map::new();
    for (key, val) in obj {
        if key == "__metadata__" {
            if let Value::Object(m) = val {
                metadata = m.clone();
            }
            continue;
        }
        let t = val.as_object().ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("tensor '{key}' is not an object"))
        })?;
        let dtype_str = t.get("dtype").and_then(Value::as_str).ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("tensor '{key}' missing dtype"))
        })?;
        let dtype: Dtype = serde_json::from_value(Value::String(dtype_str.to_string()))
            .map_err(|_| SafetensorsError::UnsupportedDtype(dtype_str.to_string()))?;
        let shape_val = t.get("shape").and_then(Value::as_array).ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("tensor '{key}' missing shape"))
        })?;
        let shape: Vec<usize> = shape_val
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| {
                SafetensorsError::InvalidHeader(format!("tensor '{key}' has non-integer shape"))
            })?;
        let offsets_val = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                SafetensorsError::InvalidHeader(format!("tensor '{key}' missing data_offsets"))
            })?;
        if offsets_val.len() != 2 {
            return Err(SafetensorsError::InvalidHeader(format!(
                "tensor '{key}' data_offsets must have exactly two elements"
            )));
        }
        let start = offsets_val[0]
            .as_u64()
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("tensor '{key}' bad offset")))?
            as usize;
        let end = offsets_val[1]
            .as_u64()
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("tensor '{key}' bad offset")))?
            as usize;
        tensors.push((
            key.clone(),
            TensorInfo {
                dtype,
                shape,
                start,
                end,
            },
        ));
    }

    // Validate every tensor's offsets against the actual file length. The
    // header is attacker-controlled, so an unvalidated `start`/`end` could
    // make `get_tensor` index out of bounds (panic) or, after a `usize`
    // overflow on a huge offset, slice a wrong but in-bounds region.
    let data_base = header_end;
    for (name, info) in &tensors {
        if info.start > info.end {
            return Err(SafetensorsError::InvalidHeader(format!(
                "tensor '{name}' has start ({}) > end ({})",
                info.start, info.end
            )));
        }
        // Cross-check the declared byte span against `dtype.size_bytes() *
        // numel()`. Without this a truncated/oversized region would let callers
        // reading `TensorView::data` directly see the wrong number of bytes.
        let numel: usize = info
            .shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| acc.checked_mul(d))
            .ok_or_else(|| {
                SafetensorsError::InvalidHeader(format!(
                    "tensor '{name}' element count overflows usize"
                ))
            })?;
        let expected = info.dtype.size_bytes().checked_mul(numel).ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("tensor '{name}' byte size overflows usize"))
        })?;
        let span = info.end - info.start;
        if span != expected {
            return Err(SafetensorsError::InvalidHeader(format!(
                "tensor '{name}' declares {span} bytes but shape/dtype require {expected}"
            )));
        }
        // `data_base + info.end` must stay in bounds without wrapping.
        let abs_end = data_base.checked_add(info.end).ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("tensor '{name}' offset overflows usize"))
        })?;
        if abs_end > bytes.len() {
            return Err(SafetensorsError::InvalidHeader(format!(
                "tensor '{name}' data_offsets exceed file size"
            )));
        }
    }

    Ok((tensors, data_base, metadata))
}

/// Converts an IEEE-754 `binary16` value to `f32`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = f32::from((bits >> 15) & 1) * -2.0 + 1.0; // 1.0 or -1.0
    let exp = (bits >> 10) & 0x1f;
    let mant = f32::from(bits & 0x3ff);
    let magnitude = if exp == 0 {
        if mant == 0.0 {
            0.0
        } else {
            mant * 2f32.powi(-24)
        }
    } else if exp == 31 {
        if mant == 0.0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(i32::from(exp) - 15)
    };
    sign * magnitude
}

/// Converts a `bfloat16` value to `f32` by widening the exponent/mantissa.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
