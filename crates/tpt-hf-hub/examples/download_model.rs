//! Downloads a single file from the Hugging Face Hub, printing progress to
//! stdout.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tpt-hf-hub --example download_model -- gpt2 config.json
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use tpt_hf_hub::{HubClient, ProgressReporter};

struct StdoutProgress {
    last_reported: AtomicU64,
}

impl ProgressReporter for StdoutProgress {
    fn on_start(&self, file: &str, total_bytes: Option<u64>) {
        match total_bytes {
            Some(total) => println!("starting {file} ({total} bytes)"),
            None => println!("starting {file} (unknown size)"),
        }
    }

    fn on_progress(&self, file: &str, downloaded_bytes: u64, total_bytes: Option<u64>) {
        let last = self.last_reported.swap(downloaded_bytes, Ordering::Relaxed);
        if downloaded_bytes.saturating_sub(last) < 1_000_000 {
            return;
        }
        match total_bytes {
            Some(total) => println!("{file}: {downloaded_bytes}/{total} bytes"),
            None => println!("{file}: {downloaded_bytes} bytes"),
        }
    }

    fn on_complete(&self, file: &str) {
        println!("finished {file}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let repo_id = args.next().unwrap_or_else(|| "gpt2".to_string());
    let filename = args.next().unwrap_or_else(|| "config.json".to_string());

    let client = HubClient::new()?;
    let progress = StdoutProgress {
        last_reported: AtomicU64::new(0),
    };

    let path = client.download_file(&repo_id, &filename, &progress).await?;
    println!("cached at {}", path.display());
    Ok(())
}
