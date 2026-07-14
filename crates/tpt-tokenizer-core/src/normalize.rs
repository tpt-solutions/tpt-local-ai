//! Unicode normalization (NFC / NFD / NFKC / NFKD) applied before tokenization.
//!
//! Correct Unicode normalization requires the full Unicode Character Database
//! (composition/decomposition mappings and combining classes), which is far too
//! large to hand-roll while staying correct. To keep the crate's default build
//! **dependency-free**, this capability is gated behind the opt-in
//! `normalization` Cargo feature, which pulls in the well-tested
//! [`unicode-normalization`] crate (itself `no_std` + `alloc`).
//!
//! Enable it with:
//!
//! ```toml
//! tpt-tokenizer-core = { version = "0.1", features = ["normalization"] }
//! ```
//!
//! [`unicode-normalization`]: https://docs.rs/unicode-normalization

use alloc::string::String;
use unicode_normalization::UnicodeNormalization;

/// A Unicode normalization form, matching the four standard forms used by
/// Hugging Face tokenizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationForm {
    /// Canonical decomposition followed by canonical composition (NFC).
    Nfc,
    /// Canonical decomposition (NFD).
    Nfd,
    /// Compatibility decomposition followed by canonical composition (NFKC).
    Nfkc,
    /// Compatibility decomposition (NFKD).
    Nfkd,
}

/// Normalize `text` to the given [`NormalizationForm`].
#[must_use]
pub fn normalize(text: &str, form: NormalizationForm) -> String {
    match form {
        NormalizationForm::Nfc => text.nfc().collect(),
        NormalizationForm::Nfd => text.nfd().collect(),
        NormalizationForm::Nfkc => text.nfkc().collect(),
        NormalizationForm::Nfkd => text.nfkd().collect(),
    }
}
