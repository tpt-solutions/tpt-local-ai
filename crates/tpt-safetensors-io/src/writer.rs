//! Builder API for writing spec-compliant safetensors files.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{Map, Value};

use crate::dtype::Dtype;
use crate::error::SafetensorsError;
use crate::reader::SafetensorsFile;

/// A builder that accumulates tensors and serialises them into a safetensors
/// byte buffer (or file) with a correctly 8-byte-aligned header.
///
/// Tensors are written to the data section in the order they are added, and the
/// header's `data_offsets` reflect that ordering.
#[derive(Debug, Default)]
pub struct SafetensorsBuilder {
    tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    metadata: Map<String, Value>,
}

impl SafetensorsBuilder {
    /// Creates a new, empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tensor to be written, supplying raw little-endian bytes.
    ///
    /// `data` must contain exactly `dtype.size_bytes() * shape.iter().product()`
    /// bytes (little-endian element order).
    ///
    /// # Errors
    /// Returns [`SafetensorsError::BadDataLength`] when the byte length does not
    /// match the declared shape and dtype.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<(), SafetensorsError> {
        let name = name.into();
        let expected = dtype.size_bytes() * shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(SafetensorsError::BadDataLength {
                name,
                expected,
                actual: data.len(),
            });
        }
        self.tensors.push((name, dtype, shape, data));
        Ok(())
    }

    /// Adds an entry to the top-level `__metadata__` table.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: Value) -> &mut Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Loads every tensor and the `__metadata__` table from an existing
    /// safetensors file into a fresh builder (copying tensor bytes), so a
    /// caller can replace or add a subset of tensors and re-serialise.
    ///
    /// This is the entry point for "patch a checkpoint" workflows: open a file,
    /// [`SafetensorsBuilder::replace_tensor`] the weights you changed, and
    /// [`SafetensorsBuilder::write_to_file`] the result.
    ///
    /// # Errors
    /// Propagates any error from opening/parsing the source file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, SafetensorsError> {
        let file = SafetensorsFile::open(path)?;
        let mut builder = Self::new();
        let names: Vec<String> = file.tensor_names().map(str::to_string).collect();
        for name in names {
            let view = file
                .get_tensor(&name)
                .ok_or_else(|| SafetensorsError::TensorNotFound(name.clone()))?;
            builder.add_tensor(name, view.dtype, view.shape.clone(), view.data.to_vec())?;
        }
        for (k, v) in file.metadata() {
            builder.metadata.insert(k.clone(), v.clone());
        }
        Ok(builder)
    }

    /// Replaces an existing tensor (matched by name) in place, preserving its
    /// position in the file, or appends it if the name is not present.
    ///
    /// # Errors
    /// Returns [`SafetensorsError::BadDataLength`] when the byte length does not
    /// match the declared shape and dtype.
    pub fn replace_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<(), SafetensorsError> {
        let name = name.into();
        let expected = dtype.size_bytes() * shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(SafetensorsError::BadDataLength {
                name,
                expected,
                actual: data.len(),
            });
        }
        if let Some(slot) = self.tensors.iter_mut().find(|(n, ..)| *n == name) {
            *slot = (name, dtype, shape, data);
        } else {
            self.tensors.push((name, dtype, shape, data));
        }
        Ok(())
    }

    /// Convenience for registering a `Dtype::F32` tensor from native `f32`
    /// values, which are converted to little-endian bytes automatically.
    ///
    /// # Errors
    /// Returns [`SafetensorsError::BadDataLength`] when the element count does
    /// not match the declared shape.
    pub fn add_f32(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        data: impl Into<Vec<f32>>,
    ) -> Result<(), SafetensorsError> {
        let values = data.into();
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.add_tensor(name, Dtype::F32, shape, bytes)
    }

    /// Serialises all registered tensors into a safetensors byte buffer.
    ///
    /// The 8-byte length prefix and the JSON header are padded so that the
    /// start of the data section is 8-byte aligned, as required by the format.
    ///
    /// For large outputs prefer [`SafetensorsBuilder::write_to_file`] or
    /// [`SafetensorsBuilder::write_to`], which stream to the sink instead of
    /// buffering the whole file in memory.
    ///
    /// # Errors
    /// Returns [`SafetensorsError::Json`] if the header cannot be serialised.
    pub fn build(&self) -> Result<Vec<u8>, SafetensorsError> {
        let prefixed_header = self.build_prefixed_header()?;
        let data_len: usize = self.tensors.iter().map(|(_, _, _, d)| d.len()).sum();
        let mut out = Vec::with_capacity(prefixed_header.len() + data_len);
        out.extend_from_slice(&prefixed_header);
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
        }
        Ok(out)
    }

    /// Builds the 8-byte length prefix followed by the padded JSON header.
    fn build_prefixed_header(&self) -> Result<Vec<u8>, SafetensorsError> {
        let mut offset: u64 = 0;
        let mut map = Map::new();
        for (name, dtype, shape, _) in &self.tensors {
            let len = (dtype.size_bytes() * shape.iter().product::<usize>()) as u64;
            let mut obj = Map::new();
            obj.insert("dtype".into(), serde_json::to_value(dtype)?);
            obj.insert(
                "shape".into(),
                Value::Array(shape.iter().map(|s| Value::from(*s as u64)).collect()),
            );
            obj.insert(
                "data_offsets".into(),
                Value::Array(vec![Value::from(offset), Value::from(offset + len)]),
            );
            map.insert(name.clone(), Value::Object(obj));
            offset += len;
        }
        if !self.metadata.is_empty() {
            map.insert("__metadata__".into(), Value::Object(self.metadata.clone()));
        }

        let mut header = serde_json::to_vec(&Value::Object(map))?;
        let aligned = (header.len() + 7) & !7;
        header.resize(aligned, b' ');

        let n = header.len() as u64;
        let mut out = Vec::with_capacity(8 + header.len());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&header);
        Ok(out)
    }

    /// Streams the full safetensors file to `writer` without buffering the whole
    /// payload in memory. The header is emitted first, then each tensor's bytes
    /// in registration order.
    ///
    /// # Errors
    /// Propagates any I/O or serialisation error.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), SafetensorsError> {
        let prefixed_header = self.build_prefixed_header()?;
        writer.write_all(&prefixed_header)?;
        for (_, _, _, data) in &self.tensors {
            writer.write_all(data)?;
        }
        Ok(())
    }

    /// Builds the file and writes it to `path` atomically.
    ///
    /// The bytes are first streamed to a temporary file in the same directory,
    /// flushed, and then renamed over `path`. This guarantees the destination
    /// is never left half-written or truncated if the write is interrupted, and
    /// never materialises the entire file in memory.
    ///
    /// # Errors
    /// Propagates any I/O or serialisation error.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), SafetensorsError> {
        let path = path.as_ref();
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let stem = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "output.safetensors".to_string());
        let tmp = dir.join(format!(".{}.{}.tmp", stem, std::process::id()));

        {
            let file = File::create(&tmp)?;
            let mut writer = BufWriter::new(file);
            self.write_to(&mut writer)?;
            let file = writer
                .into_inner()
                .map_err(|e| SafetensorsError::Io(e.into_error()))?;
            file.sync_all().ok();
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}
