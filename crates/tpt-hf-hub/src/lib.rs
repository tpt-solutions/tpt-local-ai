//! Async Hugging Face Hub downloader and cache manager.
//!
//! [`HubClient`] downloads files from a Hugging Face Hub compatible server
//! into an XDG-style local cache (`~/.cache/tpt/hub` by default), with:
//!
//! - Resumable downloads via HTTP `Range` requests
//! - SHA256 verification against the Hub's linked ETag, when available
//! - Atomic writes (download to `*.tmp`, rename on success)
//! - A [`ProgressReporter`] trait so callers can plug in their own UI
//!
//! ```no_run
//! use tpt_hf_hub::{HubClient, NoopProgressReporter};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = HubClient::new()?;
//! let path = client
//!     .download_file("gpt2", "config.json", &NoopProgressReporter)
//!     .await?;
//! println!("downloaded to {}", path.display());
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod cache;
mod client;
mod error;
mod progress;

pub use cache::default_cache_dir;
pub use client::HubClient;
pub use error::HubError;
pub use progress::{NoopProgressReporter, ProgressReporter};
