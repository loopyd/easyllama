# Changelog

Release history pulled from GitHub releases:
https://github.com/loopyd/easyllama/releases

Format follows Keep a Changelog style where possible, based on published release notes.

## [v0.3.6] - 2026-05-04

Patch release focused on reverting the Spiritbuun proxy wrapper.

### Fixed

- Reverted the Spiritbuun FastAPI/httpx proxy wrapper and restored direct `llama-server` passthrough after proxy handling clobbered Spiritbuun tool and message results.

### Changed

- Removed proxy-only Spiritbuun request rewriting, dropped proxy-only Python dependencies (`httpx`, `h11`, `h2`), and aligned the shipped Spiritbuun config and API reference with the restored direct backend behavior.
- Kept the higher tracked Spiritbuun context limit while removing the proxy-era launcher changes.

### Validation

- Rebuilt, restarted, and warmed the `spiritbuun` runtime successfully after the revert.
- Verified `GET /health`, `GET /v1/models`, `GET /ui/`, `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/completions`, `POST /v1/responses`, `POST /v1/embeddings`, and `POST /v1/rerank` against the live runtime.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.6
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.5...v0.3.6

## [v0.3.5] - 2026-05-04

Patch release focused on Spiritbuun completion-budget forwarding.

### Fixed

- Normalized `max_completion_tokens` into upstream `max_tokens` in the Spiritbuun proxy when clients omit `max_tokens`, preventing abrupt early stops from upstream default generation limits.

### Validation

- Re-ran the Spiritbuun request sanitizer against a `max_completion_tokens`-only payload and verified it forwards `max_tokens` upstream.
- Rebuilt, restarted, and warmed the `spiritbuun` runtime successfully.
- Verified `GET /health`, `GET /v1/models`, `GET /ui/`, `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/completions`, `POST /v1/responses`, `POST /v1/embeddings`, and `POST /v1/rerank` against the live runtime.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.5
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.4...v0.3.5

## [v0.3.4] - 2026-05-04

Patch release focused on Spiritbuun request handling, context defaults, and API parity.

### Fixed

- Added a Spiritbuun FastAPI proxy that strips hidden reasoning content and thinking flags before forwarding chat-style requests upstream, preventing oversized preserved-thinking payloads from tripping context-limit failures.
- Switched the Spiritbuun proxy lifecycle from deprecated `FastAPI.on_event` hooks to a lifespan handler and corrected the catch-all route declaration so the server starts cleanly.

### Changed

- Raised the tracked Spiritbuun example context size to `262144`, enabled `--context-shift`, and stripped client-side reasoning/template params in the shipped example config.
- Added Spiritbuun runtime HTTP dependencies: `httpx`, `h11`, and `h2`.
- Documented `POST /v1/messages` support for `spiritbuun` in the API reference.

### Validation

- Rebuilt and restarted the `spiritbuun` image successfully.
- Verified `GET /health`, `GET /v1/models`, `GET /ui/`, `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/completions`, `POST /v1/responses`, `POST /v1/embeddings`, and `POST /v1/rerank` against the live runtime.
- Confirmed oversized hidden-reasoning regression cases now return `200` for both `POST /v1/chat/completions` and `POST /v1/messages`.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.4
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.3...v0.3.4

## [v0.3.2] - 2026-05-03

Patch release focused on config hygiene and provider workflow tooling.

### Changed

- Disabled `sendLoadingState` in shipped config templates so `llama-swap` loading and switching messages do not pollute client reasoning/context.
- Updated `easyllama-provider` skill to use bundled helper scripts for code validation, rebuild and warmup, and full public endpoint regression checks.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.2
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.1...v0.3.2

## [v0.2.2] - 2026-05-03

### Fixed

- Fixed Lucebox reasoning-budget handling through request middleware so hidden thinking budget applies to OpenAI-compatible chat requests, including requests that omit `max_tokens` or send `max_completion_tokens`.
- Corrected Luce finish signaling so responses that exhaust `gen_len` return `finish_reason: "length"` instead of `"stop"`.

### Changed

- Raised Lucebox example preset hidden thinking budget to improve longer preserved-thinking turns.

### Validation

- Verified capped Lucebox JSON and SSE responses report `finish_reason: "length"`.
- Verified reasoning-heavy Lucebox requests that omit `max_tokens` no longer fall back to upstream `512`-token default and can reach final content.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.2.2

## [v0.2.1] - 2026-05-03

Hotfix for Lucebox token limits during agentic coding workloads.

### Changed

- Raised Lucebox preset `dflash_max_ctx` to `131072` in tracked config template.
- Bumped package metadata to `0.2.1`.

### Why

Previous Luce preset ceiling was too low for long agentic coding sessions and tool-heavy prompts, which could force premature compaction or hard context-limit failures.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.2.1

## [v0.2.0] - 2026-05-03

### Added

- Real Python package runtime under `easyllama/`, while `./run.sh` remains single user-facing entrypoint.
- Mode-specific, BuildKit-backed builds.
- One config template per mode: `config.basic.yml.example`, `config.turboquant.yml.example`, and `config.lucebox.yml.example`.

### Changed

- Default configs validated end to end across all three modes.
- Each mode now builds its own local image tag, such as `llamacpp-local:cuda13-basic` and `llamacpp-local:cuda13-lucebox`.

### API coverage verified

- `basic`: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`, `POST /v1/embeddings`, `POST /v1/rerank`
- `turboquant`: same coverage as `basic`
- `lucebox`: same coverage as `basic`, plus `POST /v1/messages`

### Upgrade notes

- `config.yml.example` replaced by mode-specific config templates.
- `./run.sh` still works, but now dispatches into `easyllama` package.
- If Python runtime code under `easyllama/` changes, rebuild affected mode image before testing because code is baked into image.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.2.0

## [v0.1.3] - 2026-05-02

### Changed

- Added dedicated `qmd-rerank` batch and ubatch settings to `config.yml.example`.
- Kept tracked server template aligned with live reranker deployment.
- Refreshed README content structure.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.1.3

## [v0.1.2] - 2026-05-02

### Added

- `./run.sh warmup [model...]` to load models early through `llama-swap` upstream health route.
- First-class rerank support documentation for `qmd-rerank` and `/v1/rerank`.

### Changed

- Default `./run.sh build` targets `TheTom/llama-cpp-turboquant@feature/turboquant-kv-cache`.
- `config.yml.example` synced with active QMD aliases: `qmd-generate`, `qmd-embed`, and `qmd-rerank`.
- README wording cleaned up for concision.

### Included commits

- `5a6368e` Simplify README wording
- `fc3f841` Document QMD rerank config and API
- `5ecc23e` Add warmup flow and turboquant defaults

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.1.2

## [v0.1.1] - 2026-05-02

Patch release for `v0.1.0` `run.sh` regression.

### Fixed

- Restored successful `cfg()` completion when `LLAMACPP_LS_CONFIG_FILE` is unset.
- Kept config fallback and logging messages on stderr so command substitution is not corrupted.
- Preserved `config.yml` / `config.yml.example` workflow introduced in `v0.1.0`.

### Notes

Recommended upgrade for anyone using `v0.1.0`.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.1.1

## [v0.1.0] - 2026-05-01

Initial release.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.1.0
