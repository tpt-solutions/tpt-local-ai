//! The [`HubClient`] downloader and cache manager.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use futures_util::stream::{self, StreamExt};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex as AsyncMutex;

use crate::cache::{default_cache_dir, sanitize_repo_id, validate_relative_path, validate_repo_id};
use crate::error::HubError;
use crate::progress::ProgressReporter;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";
const DEFAULT_MAX_RETRIES: u32 = 3;
const DEFAULT_CONCURRENCY: usize = 4;

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

/// Returns the process-global registry of per-tmp-path locks.
///
/// Concurrent downloads of the same file share a deterministic `*.tmp` path;
/// serializing on a per-path async mutex prevents them from corrupting each
/// other's partial output.
fn tmp_locks() -> &'static Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_for(path: &Path) -> Arc<AsyncMutex<()>> {
    let mut map = tmp_locks().lock().unwrap_or_else(|e| e.into_inner());
    map.entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

/// Async client for downloading and caching files from a Hugging Face Hub
/// compatible server.
pub struct HubClient {
    http: reqwest::Client,
    cache_dir: PathBuf,
    endpoint: String,
    token: Option<String>,
    offline: bool,
    max_retries: u32,
    concurrency: usize,
}

impl HubClient {
    /// Creates a client using the default cache directory (see
    /// [`crate::default_cache_dir`]).
    ///
    /// Environment conventions honored:
    /// - `HF_ENDPOINT` overrides the default `https://huggingface.co` endpoint.
    /// - `HF_TOKEN` supplies a bearer token for gated/private models.
    /// - `HF_HUB_OFFLINE=1` enables offline (cache-only) mode.
    pub fn new() -> Result<Self, HubError> {
        Ok(Self {
            http: reqwest::Client::new(),
            cache_dir: default_cache_dir()?,
            endpoint: endpoint_from_env(),
            token: token_from_env(),
            offline: offline_from_env(),
            max_retries: DEFAULT_MAX_RETRIES,
            concurrency: DEFAULT_CONCURRENCY,
        })
    }

    /// Creates a client rooted at an explicit cache directory.
    ///
    /// Like [`Self::new`], this still reads `HF_ENDPOINT`, `HF_TOKEN`, and
    /// `HF_HUB_OFFLINE` from the environment.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            http: reqwest::Client::new(),
            cache_dir: cache_dir.into(),
            endpoint: endpoint_from_env(),
            token: token_from_env(),
            offline: offline_from_env(),
            max_retries: DEFAULT_MAX_RETRIES,
            concurrency: DEFAULT_CONCURRENCY,
        }
    }

    /// Overrides the Hub endpoint (useful for mirrors or tests). Defaults to
    /// `$HF_ENDPOINT` or `https://huggingface.co`.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets a bearer token used to authenticate requests for gated or private
    /// models. Overrides any token discovered via `HF_TOKEN`.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Enables or disables offline mode. In offline mode no network request is
    /// ever made; only already-cached files are returned.
    pub fn with_offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Sets the maximum number of retry attempts for transient network
    /// failures (default 3). A value of 0 disables retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the maximum number of files downloaded concurrently by
    /// [`Self::snapshot_download`] (default 4). Values below 1 are treated as 1.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
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
        validate_relative_path(filename)?;

        let dest_dir = self
            .cache_dir
            .join(sanitize_repo_id(repo_id))
            .join(revision);
        let final_path = dest_dir.join(filename);

        if final_path.is_file() {
            progress.on_complete(filename);
            return Ok(final_path);
        }

        if self.offline {
            return Err(HubError::Offline {
                repo_id: repo_id.to_string(),
                filename: filename.to_string(),
            });
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

        // Serialize concurrent downloads that share this deterministic tmp path.
        let lock = lock_for(&tmp_path);
        let _guard = lock.lock().await;

        // Another task may have completed the download while we waited.
        if final_path.is_file() {
            progress.on_complete(filename);
            return Ok(final_path);
        }

        let mut attempt = 0;
        loop {
            match self
                .download_once(&url, &tmp_path, &final_path, filename, progress)
                .await
            {
                Ok(path) => return Ok(path),
                Err(err) if attempt < self.max_retries && is_retryable(&err) => {
                    attempt += 1;
                    let backoff = std::time::Duration::from_millis(200 * (1u64 << (attempt - 1)));
                    tokio::time::sleep(backoff).await;
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Performs a single download attempt, resuming from any existing `*.tmp`.
    async fn download_once(
        &self,
        url: &str,
        tmp_path: &Path,
        final_path: &Path,
        filename: &str,
        progress: &dyn ProgressReporter,
    ) -> Result<PathBuf, HubError> {
        let resume_from = match tokio::fs::metadata(tmp_path).await {
            Ok(meta) => meta.len(),
            Err(_) => 0,
        };

        let mut request = self.http.get(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        if resume_from > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
        }

        let response = request.send().await?;
        let status = response.status().as_u16();
        if !response.status().is_success() && status != 416 {
            return Err(HubError::Status {
                url: url.to_string(),
                status,
            });
        }

        // If we asked for a Range but the server replied `200 OK` (rather than
        // `206 Partial Content`), it ignored the Range header and is sending the
        // *whole* body. Appending it to our partial file would corrupt the
        // output, so restart from scratch.
        let restart = resume_from > 0 && status == 200;
        let effective_resume = if restart { 0 } else { resume_from };

        let expected_sha256 = response
            .headers()
            .get("x-linked-etag")
            .or_else(|| response.headers().get(reqwest::header::ETAG))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').to_string())
            .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()));

        let content_length = response.content_length();
        let total_bytes = content_length.map(|len| len + effective_resume);

        let already_complete = status == 416;

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(restart)
            .open(tmp_path)
            .await
            .map_err(|source| io_err(tmp_path, source))?;
        if effective_resume > 0 {
            file.seek(std::io::SeekFrom::End(0))
                .await
                .map_err(|source| io_err(tmp_path, source))?;
        }

        progress.on_start(filename, total_bytes);

        if !already_complete {
            let mut downloaded = effective_resume;
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk)
                    .await
                    .map_err(|source| io_err(tmp_path, source))?;
                downloaded += chunk.len() as u64;
                progress.on_progress(filename, downloaded, total_bytes);
            }
            file.flush()
                .await
                .map_err(|source| io_err(tmp_path, source))?;
        }
        drop(file);

        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(tmp_path).await?;
            if actual != expected {
                // Remove the corrupt partial so a subsequent attempt restarts
                // cleanly instead of resuming from bad bytes.
                let _ = tokio::fs::remove_file(tmp_path).await;
                return Err(HubError::HashMismatch {
                    path: tmp_path.to_path_buf(),
                    expected,
                    actual,
                });
            }
        }

        rename_atomic(tmp_path, final_path).await?;

        progress.on_complete(filename);
        Ok(final_path.to_path_buf())
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
    ///
    /// Sibling files are downloaded concurrently, up to the limit configured
    /// via [`Self::with_concurrency`].
    pub async fn snapshot_download_revision(
        &self,
        repo_id: &str,
        revision: &str,
        progress: &dyn ProgressReporter,
    ) -> Result<PathBuf, HubError> {
        validate_repo_id(repo_id)?;

        let snapshot_dir = self
            .cache_dir
            .join(sanitize_repo_id(repo_id))
            .join(revision);

        if self.offline {
            // Offline snapshots simply return the cached directory root if it
            // exists; there is no way to enumerate siblings without the network.
            if snapshot_dir.is_dir() {
                return Ok(snapshot_dir);
            }
            return Err(HubError::Offline {
                repo_id: repo_id.to_string(),
                filename: "<snapshot>".to_string(),
            });
        }

        let api_url = format!(
            "{}/api/models/{}/revision/{}",
            self.endpoint, repo_id, revision
        );
        let mut request = self.http.get(&api_url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await?;
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

        // Validate every server-provided filename up front, before any writes.
        for sibling in &info.siblings {
            validate_relative_path(&sibling.rfilename)?;
        }

        let results = stream::iter(info.siblings.iter().map(|sibling| {
            let rfilename = sibling.rfilename.clone();
            async move {
                self.download_file_revision(repo_id, &rfilename, revision, progress)
                    .await
            }
        }))
        .buffer_unordered(self.concurrency)
        .collect::<Vec<_>>()
        .await;

        for r in results {
            r?;
        }

        Ok(snapshot_dir)
    }
}

fn endpoint_from_env() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())
}

fn token_from_env() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("HUGGING_FACE_HUB_TOKEN")
                .ok()
                .filter(|s| !s.is_empty())
        })
}

fn offline_from_env() -> bool {
    matches!(
        std::env::var("HF_HUB_OFFLINE").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes")
    )
}

fn is_retryable(err: &HubError) -> bool {
    match err {
        // Transport-level failures (connect/timeout/body) are worth retrying.
        HubError::Http(_) => true,
        // 5xx and 429 responses are transient server-side conditions.
        HubError::Status { status, .. } => *status >= 500 || *status == 429,
        _ => false,
    }
}

/// Renames `tmp` over `final_path` atomically where possible.
///
/// On Windows `rename` fails if the destination already exists; to keep
/// behavior consistent across platforms we detect that a valid destination now
/// exists (e.g. produced by a concurrent download) and treat it as success,
/// discarding our redundant tmp file.
async fn rename_atomic(tmp: &Path, final_path: &Path) -> Result<(), HubError> {
    match tokio::fs::rename(tmp, final_path).await {
        Ok(()) => Ok(()),
        Err(_) if final_path.is_file() => {
            let _ = tokio::fs::remove_file(tmp).await;
            Ok(())
        }
        Err(source) => Err(io_err(final_path, source)),
    }
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
