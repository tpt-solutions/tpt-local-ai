//! The [`HubClient`] downloader and cache manager.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::cache::{default_cache_dir, sanitize_repo_id};
use crate::error::HubError;
use crate::progress::ProgressReporter;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";

/// A single file entry in a Hub repository listing.
#[derive(Debug, Clone, serde::Deserialize)]
struct RepoInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Sibling {
    rfilename: String,
}

/// Async client for downloading and caching files from a Hugging Face Hub
/// compatible server.
pub struct HubClient {
    http: reqwest::Client,
    cache_dir: PathBuf,
    endpoint: String,
}

impl HubClient {
    /// Creates a client using the default cache directory (see
    /// [`crate::cache::default_cache_dir`]) and `https://huggingface.co`.
    pub fn new() -> Result<Self, HubError> {
        Ok(Self {
            http: reqwest::Client::new(),
            cache_dir: default_cache_dir()?,
            endpoint: DEFAULT_ENDPOINT.to_string(),
        })
    }

    /// Creates a client rooted at an explicit cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            http: reqwest::Client::new(),
            cache_dir: cache_dir.into(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
        }
    }

    /// Overrides the Hub endpoint (useful for mirrors or tests). Defaults to
    /// `https://huggingface.co`.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// The root directory this client reads from and writes to.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Downloads a single file from `repo_id` at the `main` revision,
    /// reporting progress via `progress`. Returns the local cache path.
    ///
    /// If the file is already fully cached, no network request is made and
    /// the cached path is returned immediately.
    pub async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        progress: &dyn ProgressReporter,
    ) -> Result<PathBuf, HubError> {
        self.download_file_revision(repo_id, filename, DEFAULT_REVISION, progress)
            .await
    }

    /// Like [`Self::download_file`] but for an explicit `revision` (branch,
    /// tag, or commit SHA).
    pub async fn download_file_revision(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
        progress: &dyn ProgressReporter,
    ) -> Result<PathBuf, HubError> {
        validate_repo_id(repo_id)?;

        let dest_dir = self
            .cache_dir
            .join(sanitize_repo_id(repo_id))
            .join(revision);
        let final_path = dest_dir.join(filename);

        if final_path.is_file() {
            progress.on_complete(filename);
            return Ok(final_path);
        }

        if let Some(parent) = final_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| io_err(parent, source))?;
        }

        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, repo_id, revision, filename
        );
        let tmp_path = tmp_path_for(&final_path);

        let resume_from = match tokio::fs::metadata(&tmp_path).await {
            Ok(meta) => meta.len(),
            Err(_) => 0,
        };

        let mut request = self.http.get(&url);
        if resume_from > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
        }

        let response = request.send().await?;
        if !response.status().is_success() && response.status().as_u16() != 416 {
            return Err(HubError::Status {
                url,
                status: response.status().as_u16(),
            });
        }

        let expected_sha256 = response
            .headers()
            .get("x-linked-etag")
            .or_else(|| response.headers().get(reqwest::header::ETAG))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').to_string())
            .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()));

        let content_length = response.content_length();
        let total_bytes = content_length.map(|len| len + resume_from);

        let already_complete = response.status().as_u16() == 416;

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&tmp_path)
            .await
            .map_err(|source| io_err(&tmp_path, source))?;
        if resume_from > 0 {
            file.seek(std::io::SeekFrom::End(0))
                .await
                .map_err(|source| io_err(&tmp_path, source))?;
        }

        progress.on_start(filename, total_bytes);

        if !already_complete {
            let mut downloaded = resume_from;
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk)
                    .await
                    .map_err(|source| io_err(&tmp_path, source))?;
                downloaded += chunk.len() as u64;
                progress.on_progress(filename, downloaded, total_bytes);
            }
            file.flush()
                .await
                .map_err(|source| io_err(&tmp_path, source))?;
        }
        drop(file);

        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(&tmp_path).await?;
            if actual != expected {
                return Err(HubError::HashMismatch {
                    path: tmp_path,
                    expected,
                    actual,
                });
            }
        }

        tokio::fs::rename(&tmp_path, &final_path)
            .await
            .map_err(|source| io_err(&final_path, source))?;

        progress.on_complete(filename);
        Ok(final_path)
    }

    /// Downloads every file in `repo_id` at the `main` revision into the
    /// cache and returns the snapshot directory root.
    pub async fn snapshot_download(
        &self,
        repo_id: &str,
        progress: &dyn ProgressReporter,
    ) -> Result<PathBuf, HubError> {
        self.snapshot_download_revision(repo_id, DEFAULT_REVISION, progress)
            .await
    }

    /// Like [`Self::snapshot_download`] but for an explicit `revision`.
    pub async fn snapshot_download_revision(
        &self,
        repo_id: &str,
        revision: &str,
        progress: &dyn ProgressReporter,
    ) -> Result<PathBuf, HubError> {
        validate_repo_id(repo_id)?;

        let api_url = format!(
            "{}/api/models/{}/revision/{}",
            self.endpoint, repo_id, revision
        );
        let response = self.http.get(&api_url).send().await?;
        if !response.status().is_success() {
            return Err(HubError::Status {
                url: api_url,
                status: response.status().as_u16(),
            });
        }
        let info: RepoInfo = response
            .json()
            .await
            .map_err(|e| HubError::Api(e.to_string()))?;

        for sibling in &info.siblings {
            self.download_file_revision(repo_id, &sibling.rfilename, revision, progress)
                .await?;
        }

        Ok(self
            .cache_dir
            .join(sanitize_repo_id(repo_id))
            .join(revision))
    }
}

fn validate_repo_id(repo_id: &str) -> Result<(), HubError> {
    if repo_id.is_empty() || repo_id.starts_with('/') || repo_id.ends_with('/') {
        return Err(HubError::InvalidRepoId(repo_id.to_string()));
    }
    Ok(())
}

fn tmp_path_for(final_path: &Path) -> PathBuf {
    let mut tmp = final_path.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

fn io_err(path: &Path, source: std::io::Error) -> HubError {
    HubError::Io {
        path: path.to_path_buf(),
        source,
    }
}

async fn sha256_file(path: &Path) -> Result<String, HubError> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|source| io_err(path, source))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .await
            .map_err(|source| io_err(path, source))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
    s
}
