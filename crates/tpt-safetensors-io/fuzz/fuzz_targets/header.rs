#![no_main]
//! Fuzz target for the safetensors header parser.
//!
//! The header parser is explicitly designed to handle adversarial input, so it
//! must reject malformed data with an error and never panic, overflow, or slice
//! out of bounds. Run with:
//!
//! ```sh
//! cargo +nightly fuzz run header
//! ```

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use libfuzzer_sys::fuzz_target;
use tpt_safetensors_io::SafetensorsFile;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fuzz_target!(|data: &[u8]| {
    // `open` mmaps a real file, so materialise the fuzz input to a unique temp
    // path first. Parsing must never panic regardless of the bytes.
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("tpt_st_fuzz_{}_{}.bin", std::process::id(), n));
    if let Ok(mut f) = std::fs::File::create(&path) {
        if f.write_all(data).is_ok() {
            drop(f);
            let _ = SafetensorsFile::open(&path);
        }
    }
    let _ = std::fs::remove_file(&path);
});
