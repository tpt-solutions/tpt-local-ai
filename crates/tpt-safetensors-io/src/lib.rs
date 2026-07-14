//! Memory-mapped reader and writer for the
//! [safetensors](https://github.com/huggingface/safetensors) format.
//!
//! `tpt-safetensors-io` exposes two complementary APIs:
//!
//! * [`SafetensorsFile`] — a zero-copy, memory-mapped reader. Tensor payloads
//!   are borrowed straight out of the `memmap2` mapping, so no copies are made
//!   when you only need to *look* at the bytes.
//! * [`SafetensorsBuilder`] — a small builder that serialises a set of tensors
//!   back into a spec-compliant file, including the mandatory 8-byte header
//!   alignment.
//!
//! # Example
//!
//! ```
//! use tpt_safetensors_io::{SafetensorsBuilder, Dtype};
//!
//! // Build a 2x2 F32 matrix and serialise it to bytes.
//! let mut builder = SafetensorsBuilder::new();
//! builder
//!     .add_f32("w", vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
//!     .unwrap();
//! let bytes = builder.build().unwrap();
//! assert!(!bytes.is_empty());
//! ```
#![warn(missing_docs)]

mod dtype;
mod error;
mod reader;
mod writer;

pub use dtype::Dtype;
pub use error::SafetensorsError;
pub use reader::{SafetensorsFile, TensorView};
pub use writer::SafetensorsBuilder;
