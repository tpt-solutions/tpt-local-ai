//! Error types for downloading and caching files from the Hugging Face Hub.

use std::fmt;
use std::path::PathBuf;

/// Errors that can occur while resolving, downloading, or caching Hub files.
#[derive(Debug)]
#[non_exhaustive]
pub enum HubError {
    /// The local cache directory could not be determined or created.
    CacheDir(String),
    /// A network request failed.
    Http(reqwest::Error),
    /// A filesystem operation failed.
    Io {
        /// The path involved in the failed operation.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The Hub API returned a response that could not be parsed.
    Api(String),
    /// The Hub API returned a non-success status code.
    Status {
        /// The URL that was requested.
        url: String,
        /// The HTTP status code returned.
        status: u16,
    },
    /// A downloaded file's SHA256 digest did not match the expected value.
    HashMismatch {
        /// The file that failed verification.
        path: PathBuf,
        /// The digest reported by the Hub.
        expected: String,
        /// The digest computed from the downloaded bytes.
        actual: String,
    },
    /// A repository id was not in the expected `owner/name` or `name` form.
    InvalidRepoId(String),
    /// A filename or server-provided path was unsafe (absolute or containing
    /// `..` traversal segments).
    InvalidPath(String),
    /// The client is in offline mode and the requested file is not cached.
    Offline {
        /// The repository id that was requested.
        repo_id: String,
        /// The filename that was requested.
        filename: String,
    },
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HubError::CacheDir(msg) => write!(f, "could not resolve cache directory: {msg}"),
            HubError::Http(e) => write!(f, "http request failed: {e}"),
            HubError::Io { path, source } => {
                write!(f, "io error at {}: {source}", path.display())
            }
            HubError::Api(msg) => write!(f, "hub api error: {msg}"),
            HubError::Status { url, status } => {
                write!(f, "request to {url} failed with status {status}")
            }
            HubError::HashMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "sha256 mismatch for {}: expected {expected}, got {actual}",
                path.display()
            ),
            HubError::InvalidRepoId(id) => write!(f, "invalid repo id: {id}"),
            HubError::InvalidPath(p) => write!(f, "unsafe path rejected: {p}"),
            HubError::Offline { repo_id, filename } => write!(
                f,
                "offline mode: {repo_id}/{filename} is not in the local cache"
            ),
        }
    }
}

impl std::error::Error for HubError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HubError::Http(e) => Some(e),
            HubError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for HubError {
    fn from(e: reqwest::Error) -> Self {
        HubError::Http(e)
    }
}
