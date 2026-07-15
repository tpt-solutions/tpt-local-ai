# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/tpt-solutions/tpt-local-ai/compare/tpt-lora-merge-v0.1.0...tpt-lora-merge-v0.2.0) - 2026-07-15

### Other

- Format tpt-lora-merge sources with rustfmt
- Drop ndarray and clap in favor of flat-vec matmul and manual arg parsing
- Bump MSRV to 1.88.0 and simplify bf16 test construction
- Replace deprecated Iterator::repeat(...).take(n) with repeat_n
