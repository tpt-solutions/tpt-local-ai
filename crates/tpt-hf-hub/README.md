# tpt-hf-hub

[![crates.io](https://img.shields.io/crates/v/tpt-hf-hub.svg)](https://crates.io/crates/tpt-hf-hub)
[![docs.rs](https://img.shields.io/docsrs/tpt-hf-hub)](https://docs.rs/tpt-hf-hub)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Async Hugging Face Hub downloader and cache manager for Rust.

- Resumable downloads via HTTP `Range` requests
- SHA256 verification against the Hub's linked ETag, when available
- Atomic writes: downloads to `*.tmp`, renamed on success
- XDG-style cache layout (`~/.cache/tpt/hub` by default, override with `TPT_HUB_CACHE`)
- `ProgressReporter` trait so you can plug in your own progress UI — no bundled TUI dependency

## Usage

```rust,no_run
use tpt_hf_hub::{HubClient, NoopProgressReporter};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = HubClient::new()?;

// Download a single file.
let path = client
    .download_file("gpt2", "config.json", &NoopProgressReporter)
    .await?;

// Or download every file in a repo.
let snapshot_dir = client
    .snapshot_download("gpt2", &NoopProgressReporter)
    .await?;
# Ok(())
# }
```

See `examples/download_model.rs` for a runnable example with stdout progress
reporting:

```sh
cargo run -p tpt-hf-hub --example download_model -- gpt2 config.json
```

## Cache layout

Files are cached at `<cache_dir>/<owner>--<repo>/<revision>/<filename>`, e.g.
`~/.cache/tpt/hub/meta-llama--Llama-3-8B/main/config.json`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
