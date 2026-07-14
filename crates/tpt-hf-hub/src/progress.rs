//! User-pluggable progress reporting for downloads.

/// Receives progress events for a download.
///
/// Implement this to drive your own progress bar or logging; `tpt-hf-hub`
/// bundles no UI dependency of its own.
pub trait ProgressReporter: Send + Sync {
    /// Called once before the first byte of `file` is written.
    ///
    /// `total_bytes` is `None` when the server did not report a
    /// `Content-Length`.
    fn on_start(&self, file: &str, total_bytes: Option<u64>) {
        let _ = (file, total_bytes);
    }

    /// Called after each chunk is written to disk, with the cumulative byte
    /// count downloaded so far (including any bytes resumed from a prior
    /// partial download).
    fn on_progress(&self, file: &str, downloaded_bytes: u64, total_bytes: Option<u64>) {
        let _ = (file, downloaded_bytes, total_bytes);
    }

    /// Called once `file` has been fully downloaded and verified.
    fn on_complete(&self, file: &str) {
        let _ = file;
    }
}

/// A [`ProgressReporter`] that discards all events.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopProgressReporter;

impl ProgressReporter for NoopProgressReporter {}
