//! Error type for tokenization and tokenizer loading.

use alloc::string::String;
use core::fmt;

/// Errors that can occur while tokenizing text or loading a tokenizer.
#[derive(Debug)]
#[non_exhaustive]
pub enum TokenizerError {
    /// A sub-word produced during tokenization is not present in the
    /// vocabulary, and no `<unk>` / `[UNK]` fallback id was configured.
    UnknownToken(String),
    /// An I/O error while loading a vocabulary or merge file (std builds only).
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// A line in a vocabulary or merge file could not be parsed.
    MalformedFile(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::UnknownToken(t) => write!(f, "unknown token: {t}"),
            #[cfg(feature = "std")]
            TokenizerError::Io(e) => write!(f, "I/O error: {e}"),
            TokenizerError::MalformedFile(m) => write!(f, "malformed tokenizer file: {m}"),
        }
    }
}

impl core::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        #[cfg(feature = "std")]
        if let TokenizerError::Io(e) = self {
            return Some(e);
        }
        None
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}
