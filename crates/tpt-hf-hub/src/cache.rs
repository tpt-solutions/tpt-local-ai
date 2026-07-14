//! XDG-compliant cache path resolution.

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

#[cfg(test)]
mod tests {
    use super::*;

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
        std::env::set_var("TPT_HUB_CACHE", "/tmp/custom-cache");
        assert_eq!(
            default_cache_dir().unwrap(),
            PathBuf::from("/tmp/custom-cache")
        );
        std::env::remove_var("TPT_HUB_CACHE");
    }
}
