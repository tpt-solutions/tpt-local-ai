//! CPU-based merging of LoRA (Low-Rank Adaptation) adapters into base-model
//! weights, reading and writing the safetensors format.
//!
//! The core primitive is [`merge_linear`], which computes
//! `base + scale * (B @ A)` for a single linear weight. The higher-level
//! [`merge_lora`] (single adapter) and [`merge_loras`] (weighted sum of several
//! adapters) walk a base safetensors file and fold in every matching LoRA pair,
//! producing a [`MergedWeights`] you can serialise back to disk. Merged tensors
//! preserve the base dtype, base `__metadata__` is carried through, and unused
//! or partial adapter tensors are reported as errors.
//!
//! # Naming convention
//!
//! A base weight `"<module>.weight"` of shape `(out, in)` is paired with LoRA
//! tensors using either the HF PEFT (`".lora_A.weight"`/`".lora_B.weight"`) or
//! Kohya (`".lora_down.weight"`/`".lora_up.weight"`) convention, plus optional
//! PEFT named adapters (`".lora_A.<name>.weight"`) and a per-module `".alpha"`
//! tensor. Any base tensor without a matching LoRA pair is copied through
//! unchanged.
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use tpt_lora_merge::merge_linear;
//!
//! let base = array![[1.0_f32, 1.0], [1.0, 1.0]];
//! let a = array![[1.0_f32, 0.0], [0.0, 1.0]]; // (r, in)
//! let b = array![[2.0_f32, 0.0], [0.0, 2.0]]; // (out, r)
//! let merged = merge_linear(base.view(), a.view(), b.view(), 0.5);
//! assert_eq!(merged, array![[2.0, 1.0], [1.0, 2.0]]);
//! ```
#![warn(missing_docs)]

pub mod error;
pub mod merge;

pub use error::MergeError;
pub use merge::{merge_linear, merge_lora, merge_loras, MergedWeights};
