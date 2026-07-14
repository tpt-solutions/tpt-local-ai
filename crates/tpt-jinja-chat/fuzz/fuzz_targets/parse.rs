#![no_main]
//! Fuzz target for the chat-template scanner/parser.
//!
//! The parser is designed to accept untrusted template source, so it must never
//! panic. Run with:
//!
//! ```sh
//! cargo +nightly fuzz run parse
//! ```

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(src) = std::str::from_utf8(data) {
        // Parsing must never panic, regardless of input; errors are fine.
        let _ = tpt_jinja_chat::ChatTemplate::parse(src);
    }
});
