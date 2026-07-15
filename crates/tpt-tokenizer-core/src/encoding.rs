//! Batch encoding, special-token insertion, padding, and truncation.
//!
//! These conveniences layer on top of the raw [`Tokenizer::encode`] output.
//! They are scheme-agnostic (they work for both BPE and WordPiece) and live in
//! the shared [`Tokenizer`] trait as provided methods, so every tokenizer gets
//! them for free.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::TokenizerError;
use crate::tokenizer::{TokenId, Tokenizer};

/// How a sequence (or batch of sequences) should be padded to a common length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Padding {
    /// Do not pad. Sequences keep their natural length.
    #[default]
    None,
    /// Pad every sequence to a fixed length. Sequences already at least this
    /// long are left unchanged (truncation is controlled separately).
    Fixed(usize),
    /// Pad every sequence in a batch to the length of the longest sequence.
    /// Behaves like [`Padding::None`] for a single
    /// [`encode_with`](TokenizerExt::encode_with) call.
    Longest,
}

/// How an over-long sequence should be shortened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Truncation {
    /// Never truncate.
    #[default]
    None,
    /// Truncate to at most `n` ids, *including* any special tokens added by
    /// [`EncodeConfig::add_special_tokens`].
    Fixed(usize),
}

/// Configuration for [`encode_with`](TokenizerExt::encode_with) /
/// [`encode_batch`](TokenizerExt::encode_batch).
///
/// Build it fluently:
///
/// ```
/// use tpt_tokenizer_core::encoding::{EncodeConfig, Padding, Truncation};
///
/// let cfg = EncodeConfig::new()
///     .with_bos(1)
///     .with_eos(2)
///     .with_pad(0)
///     .padding(Padding::Longest)
///     .truncation(Truncation::Fixed(128));
/// ```
#[derive(Debug, Clone, Default)]
pub struct EncodeConfig {
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    pad: Option<TokenId>,
    padding: Padding,
    truncation: Truncation,
    add_special_tokens: bool,
}

impl EncodeConfig {
    /// A config that adds no special tokens and applies no padding or
    /// truncation — equivalent to a plain [`Tokenizer::encode`], but wrapped in
    /// an [`Encoding`] with an attention mask.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the beginning-of-sequence token id to prepend (enables
    /// special-token insertion).
    #[must_use]
    pub fn with_bos(mut self, id: TokenId) -> Self {
        self.bos = Some(id);
        self.add_special_tokens = true;
        self
    }

    /// Set the end-of-sequence token id to append (enables special-token
    /// insertion).
    #[must_use]
    pub fn with_eos(mut self, id: TokenId) -> Self {
        self.eos = Some(id);
        self.add_special_tokens = true;
        self
    }

    /// Set the padding token id. Required for any [`Padding`] other than
    /// [`Padding::None`].
    #[must_use]
    pub fn with_pad(mut self, id: TokenId) -> Self {
        self.pad = Some(id);
        self
    }

    /// Enable or disable BOS/EOS insertion without changing the ids.
    #[must_use]
    pub fn add_special_tokens(mut self, yes: bool) -> Self {
        self.add_special_tokens = yes;
        self
    }

    /// Set the padding strategy.
    #[must_use]
    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    /// Set the truncation strategy.
    #[must_use]
    pub fn truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = truncation;
        self
    }

    /// Number of special tokens this config would add to a sequence.
    fn num_special(&self) -> usize {
        if !self.add_special_tokens {
            return 0;
        }
        usize::from(self.bos.is_some()) + usize::from(self.eos.is_some())
    }
}

/// The result of encoding a single sequence with an [`EncodeConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    /// The token ids, including any special/padding tokens.
    pub ids: Vec<TokenId>,
    /// A mask that is `1` for real (or special) tokens and `0` for padding.
    pub attention_mask: Vec<u32>,
}

impl Encoding {
    /// Length of the (possibly padded) sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the encoding is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Assembles a single [`Encoding`] from raw ids: truncate content to leave room
/// for specials, insert BOS/EOS, then build the attention mask. Padding is *not*
/// applied here (it depends on the batch), so all mask entries are `1`.
pub(crate) fn assemble(mut ids: Vec<TokenId>, cfg: &EncodeConfig) -> Encoding {
    if let Truncation::Fixed(max) = cfg.truncation {
        let room = max.saturating_sub(cfg.num_special());
        if ids.len() > room {
            ids.truncate(room);
        }
    }

    let mut out = Vec::with_capacity(ids.len() + cfg.num_special());
    if cfg.add_special_tokens {
        if let Some(bos) = cfg.bos {
            out.push(bos);
        }
    }
    out.extend(ids);
    if cfg.add_special_tokens {
        if let Some(eos) = cfg.eos {
            out.push(eos);
        }
    }

    let attention_mask = vec![1u32; out.len()];
    Encoding {
        ids: out,
        attention_mask,
    }
}

/// Pads `enc` up to `target` ids using `pad_id`, extending the attention mask
/// with zeros. A no-op if already at least `target` long.
pub(crate) fn pad_to(enc: &mut Encoding, target: usize, pad_id: TokenId) {
    if enc.ids.len() >= target {
        return;
    }
    let deficit = target - enc.ids.len();
    enc.ids.extend(std::iter::repeat_n(pad_id, deficit));
    enc.attention_mask
        .extend(std::iter::repeat_n(0u32, deficit));
}

/// Applies the batch-level padding strategy across `encodings` in place.
///
/// # Errors
/// Returns [`TokenizerError::MalformedFile`] if padding is requested but no pad
/// token id was configured.
pub(crate) fn apply_padding(
    encodings: &mut [Encoding],
    cfg: &EncodeConfig,
) -> Result<(), TokenizerError> {
    let target = match cfg.padding {
        Padding::None => return Ok(()),
        Padding::Fixed(n) => n,
        Padding::Longest => encodings.iter().map(Encoding::len).max().unwrap_or(0),
    };
    let Some(pad_id) = cfg.pad else {
        return Err(TokenizerError::MalformedFile(alloc::string::String::from(
            "padding requested without a pad token id (call EncodeConfig::with_pad)",
        )));
    };
    for enc in encodings.iter_mut() {
        pad_to(enc, target, pad_id);
    }
    Ok(())
}

/// Provided-method extensions shared by every [`Tokenizer`].
impl<T: Tokenizer + ?Sized> TokenizerExt for T {}

/// Higher-level encoding conveniences (special tokens, padding, truncation,
/// batching) available on every [`Tokenizer`].
pub trait TokenizerExt: Tokenizer {
    /// Encode a single sequence, applying special-token insertion, truncation
    /// and (fixed) padding from `cfg`.
    ///
    /// [`Padding::Longest`] has no effect on a single sequence — use
    /// [`encode_batch`](TokenizerExt::encode_batch) for that.
    ///
    /// # Errors
    /// Propagates [`Tokenizer::encode`] errors, or returns
    /// [`TokenizerError::MalformedFile`] if fixed padding is requested without a
    /// pad token id.
    fn encode_with(&self, text: &str, cfg: &EncodeConfig) -> Result<Encoding, TokenizerError> {
        let ids = self.encode(text)?;
        let mut enc = assemble(ids, cfg);
        if let Padding::Fixed(n) = cfg.padding {
            let Some(pad_id) = cfg.pad else {
                return Err(TokenizerError::MalformedFile(alloc::string::String::from(
                    "padding requested without a pad token id (call EncodeConfig::with_pad)",
                )));
            };
            pad_to(&mut enc, n, pad_id);
        }
        Ok(enc)
    }

    /// Encode a batch of sequences, applying special tokens and truncation to
    /// each and then the batch-level [`Padding`] strategy across all of them.
    ///
    /// # Errors
    /// Propagates [`Tokenizer::encode`] errors, or returns
    /// [`TokenizerError::MalformedFile`] if padding is requested without a pad
    /// token id.
    fn encode_batch(
        &self,
        texts: &[&str],
        cfg: &EncodeConfig,
    ) -> Result<Vec<Encoding>, TokenizerError> {
        let mut encodings = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = self.encode(text)?;
            encodings.push(assemble(ids, cfg));
        }
        apply_padding(&mut encodings, cfg)?;
        Ok(encodings)
    }
}
