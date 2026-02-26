# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-26

### Added
- Streaming upload endpoint (`POST /convert-stream`) that emits NDJSON events (`document`, `block`, `summary`) without LLM escalation.

## [0.2.0] - 2026-02-26

### Added
- Optional FastAPI API server (`sr-adapt-api`) with request ID, upload-size cap, rate limiting, and API key support.
- Jobs subsystem (in-memory + SQLite backend) for async-style API operation.
- Deterministic semantic confidence scoring, wired into the escalation policy as an additional gate.

### Changed
- CI defaults to running without the native runtime (environment parity with Linux runners).
