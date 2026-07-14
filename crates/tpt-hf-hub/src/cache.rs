//! XDG-compliant cache path resolution and path-safety helpers.

use std::path::PathBuf;

use crate::error::HubError;

/// Resolves the default cache root for `tpt-hf-hub`.
///
/// Honors `TPT_HUB_CACHE` if set, otherwise falls back to the platform cache
/// directory (`$XDG_CACHE_HOME` or equivalent) joined with `tpt/hub`, e.g.
/// `~/.cache/tpt/hub` on Linux.
pub fn default_cache_dir() -> Result<PathBuf, HubError> {
    if let Ok(dir) = std::env::var("TPT_HUB_CACHE") {
        return Ok(PathBuf::from(dir));
    }
    dirs::cache_dir()
        .map(|d| d.join("tpt").join("hub"))
        .ok_or_else(|| {
            HubError::CacheDir("unable to determine platform cache directory".to_string())
        })
}

/// Converts a Hub repo id (e.g. `org/name`) into a filesystem-safe directory
/// component.
pub fn sanitize_repo_id(repo_id: &str) -> String {
    repo_id.replace('/', "--")
}

/// Validates a repo id, rejecting anything that could escape the cache root.
///
/// Rejects empty ids, leading/trailing slashes, `..` path segments, and
/// Windows-style absolute paths (e.g. `C:\...`).
pub fn validate_repo_id(repo_id: &str) -> Result<(), HubError> {
    if repo_id.is_empty() || repo_id.starts_with('/') || repo_id.ends_with('/') {
        return Err(HubError::InvalidRepoId(repo_id.to_string()));
    }
    if has_unsafe_segment(repo_id) {
        return Err(HubError::InvalidRepoId(repo_id.to_string()));
    }
    Ok(())
}

/// Validates a server-provided (or caller-provided) relative filename before it
/// is joined into a cache directory.
///
/// Rejects absolute paths, drive-letter prefixes, and any `..` traversal
/// segment so a malicious `rfilename` cannot write outside the snapshot dir.
pub fn validate_relative_path(filename: &str) -> Result<(), HubError> {
    if filename.is_empty() {
        return Err(HubError::InvalidPath(filename.to_string()));
    }
    // Absolute (POSIX) or drive-rooted (Windows) paths are never allowed.
    if filename.starts_with('/') || filename.starts_with('\\') {
        return Err(HubError::InvalidPath(filename.to_string()));
    }
    if has_unsafe_segment(filename) {
        return Err(HubError::InvalidPath(filename.to_string()));
    }
    Ok(())
}

/// Returns true if `s` contains a `..` component, a Windows drive prefix
/// (`C:`), or a UNC-style prefix, treating both `/` and `\` as separators.
fn has_unsafe_segment(s: &str) -> bool {
    // Windows drive letter, e.g. "C:\foo" or "c:/foo".
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        return true;
    }
    s.split(['/', '\\']).any(|seg| seg == ".." || seg == ".")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate the process-global `TPT_HUB_CACHE` env var
    /// so parallel test execution cannot observe each other's mutations.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn sanitizes_repo_id() {
        assert_eq!(
            sanitize_repo_id("meta-llama/Llama-3-8B"),
            "meta-llama--Llama-3-8B"
        );
        assert_eq!(sanitize_repo_id("gpt2"), "gpt2");
    }

    #[test]
    fn respects_env_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("TPT_HUB_CACHE").ok();
        std::env::set_var("TPT_HUB_CACHE", "/tmp/custom-cache");
        assert_eq!(
            default_cache_dir().unwrap(),
            PathBuf::from("/tmp/custom-cache")
        );
        match prev {
            Some(v) => std::env::set_var("TPT_HUB_CACHE", v),
            None => std::env::remove_var("TPT_HUB_CACHE"),
        }
    }

    #[test]
    fn rejects_traversal_repo_ids() {
        assert!(validate_repo_id("../etc/passwd").is_err());
        assert!(validate_repo_id("owner/../../x").is_err());
        assert!(validate_repo_id("C:\\Windows").is_err());
        assert!(validate_repo_id("gpt2").is_ok());
        assert!(validate_repo_id("meta-llama/Llama-3-8B").is_ok());
    }

    #[test]
    fn rejects_traversal_filenames() {
        assert!(validate_relative_path("../secret").is_err());
        assert!(validate_relative_path("a/../../b").is_err());
        assert!(validate_relative_path("/etc/passwd").is_err());
        assert!(validate_relative_path("\\\\server\\share").is_err());
        assert!(validate_relative_path("C:\\x").is_err());
        assert!(validate_relative_path("config.json").is_ok());
        assert!(validate_relative_path("subdir/model.bin").is_ok());
    }
}
