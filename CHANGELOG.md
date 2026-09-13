# Changelog

Release history pulled from GitHub releases:
[GitHub releases](https://github.com/loopyd/easyllama/releases)

Format follows Keep a Changelog style where possible, based on published release notes.

## [Unreleased]

## [v0.6.3] - 2026-09-13

Compact GPU reranking in Qwen mode, co-resident with GPU embeddings.

### Added

- Qwen mode now exposes `qwen3-reranker` at `/v1/rerank`, using the compact BGE reranker v2 M3 Q8_0 with two native GPU slots. Existing inference images support this endpoint; no native-code rebuild is required.
- Keep full-GPU Qwen embeddings and reranking resident together in an exclusive search group with idle unloading disabled. Large Qwen chat swaps out the search group to respect shared VRAM; chat retains its 30-minute idle timer.
- Keep proxy admission rejection disabled for all three models; bound upstream concurrency and let native slots queue work. The deployed reranker artifact is revision/size/SHA-256 pinned in dotfiles.

### Upgrade notes

- Merge the Qwen example into existing local overrides; upgrades do not overwrite them. The `qwen3-reranker` identifier serves BGE weights, not a Qwen-family reranker. Embedding identity, dimensions, precision and index compatibility remain unchanged from the 0.6B deployment.
- The portable example resolves named Hugging Face files. Immutable deployments should use verified local model paths; dotfiles captures the reranker revision, size and SHA-256, authenticated proxy, and four-container systemd supervisor.
- Existing inference images were reused. This release updates the source package/profile, not the historical build identity of those containers. Chat and the search group remain mutually exclusive to preserve VRAM headroom.
- Hindsight uses its built-in Cohere-compatible HTTP adapter against authenticated local EasyLlama, with explicit user approval because 9router has no rerank route. LLMs and embeddings still use 9router; no Hindsight, plugin or 9router native code was changed.

### Validation

- All 30 existing unit tests pass, along with related Ruff, formatting, YAML/rendering and diff checks. No regression tests or native runtime dependencies were added.
- Five small semantic rerank requests passed: 1.313 seconds cold and 10–21 ms warm. Four concurrent 300-document batches completed in 2.508 seconds; all 1,200 scores were finite, every document index was returned, and each known relevant document ranked first. These synthetic cases do not establish general relevance parity with FlashRank.
- Four real loaded Hindsight searches that previously all timed out at 90 seconds completed in 4.504–10.438 seconds. A repeated burst completed in 3.987–9.302 seconds, with Hindsight CPU samples of 0.90–10.59%, versus earlier CPU-reranking saturation near 800%. These bounded observations are not a full-corpus ingestion benchmark.
- Search → local Qwen chat → search passed; 28 backend samples observed search co-residency and chat alone, with no observed overlap. Anonymous reranking returned 401, private sockets remained loopback-only, and service health checks passed. No new timeout, JSON, assertion or OOM errors appeared in the checked post-migration logs.

## [v0.6.2] - 2026-09-11

Ship the Qwen chat resource profile validated alongside CPU embeddings.

### Changed

- Set Qwen chat generation and batch threads to eight, reduce RAM prompt cache from 16,384 to 4,096 MiB, and reduce context checkpoints from 32 to eight per slot.
- Preserve chat weights, embedded MTP, GPU placement, Q8 KV precision, reasoning, and four slots sharing the 262,144-token unified context. CPU 0.6B embeddings retain eight threads and four 32,768-token slots.
- Document the tested external Compose chat limits: eight CPUs, 32 GiB RAM, and 40 GiB total RAM-plus-swap. These are deployment caps, not changes to shared resource-role defaults or other modes. Hindsight's four-way embedding batch setting remains downstream configuration.

### Upgrade notes

- Merge the updated Qwen example into existing ignored overrides and restart through the deployment's owning supervisor. Existing local configuration is not overwritten automatically.
- This profile-only adjustment needs no inference-image rebuild or vector-index migration. Authentication, model IDs, embedding space and queue admission are unchanged; preserve ongoing ingestion and existing persistent caches.

### Validation

- All 30 existing unit tests pass; the opt-in Docker integration test remains skipped. Ruff, formatting, YAML/macro validation, and diff checks pass. Existing inference images were not rebuilt or cleaned, preserving active ingestion and persistent caches.
- Earlier same-day validation of the matching running deployment passed 64 isolated embedding vectors per profile, four parallel chat calls, a 17,748-token prompt, and 16 additional vectors with simultaneous chat, recall and ingestion. No errors, OOMs or automatic restarts were observed during that bounded validation window.
- Four-worker embedding time was approximately unchanged (53.079 to 52.840 seconds). Warm long-chat time was 4.569 to 4.665 seconds; four-chat wall time increased from 1.168 to 1.530 seconds with different generated token counts. These probes do not establish a general throughput improvement or absence of slowdown.
- Recall still took 27–28 seconds normally and approximately 60 seconds under embedding stress. Database acquisition and reflection-budget warnings remained; this release does not claim to resolve recall latency or complete the ingestion backlog.

## [v0.6.1] - 2026-09-11

Ship the smaller Qwen embedder already validated in the local deployment.

### Changed

- Switched the tracked Qwen profile from Qwen3-Embedding-8B Q5_K_M to the official Qwen3-Embedding-0.6B FP16 GGUF, preserving the `qwen3-embeddings` API ID.
- Run embeddings on CPU with eight threads, four 32,768-token slots, 512-token batch/microbatch sizes, and last-token pooling. Separate non-exclusive routing groups keep GPU chat resident during embedding requests.
- Preserve Qwen chat weights/settings, authentication, container resource limits, and other modes' embedding defaults. This configuration-only hotfix requires no inference-image rebuild.

### Upgrade notes

- Merge the updated example into existing ignored Qwen configuration overrides and restart the affected stack; upgrades do not overwrite local settings.
- Embedding width changes from 4,096 to 1,024. Back up and rebuild downstream vector indexes from source documents; never mix old and new embedding spaces. EasyLlama does not reset application databases.

### Validation

- All 30 existing unit tests pass; the opt-in Docker integration test remains skipped. Ruff, formatting, YAML/macro validation, and diff checks pass.
- The existing deployment passes authenticated health, discovery, chat, completion, responses, and embedding-before/after-chat checks. Returned vectors are finite, normalized, and 1,024-dimensional; active CPU settings match the shipped profile. Existing images were reused without a clean rebuild to preserve ongoing ingestion.
- The matching local CPU profile previously processed uncached 768-token embedding probes in 4.65–4.97 seconds versus approximately 21.7 seconds for the old 8B profile. This isolated 4.4–4.7× improvement is not a full-corpus ingestion or retrieval-quality benchmark.

## [v0.6.0] - 2026-09-10

Host networking and reliable queued Qwen model switching.

### Added

- Added opt-in `docker.network_mode=host` and `EASYLLAMA_NETWORK_MODE=host`, while retaining bridge networking by default.
- Added loopback-only backend and lifecycle endpoints for host mode; the proxy uses the configured host address and port without Docker port publishing.
- Documented host-port collision constraints, reduced network isolation, and the unsupported LMCache-dependent host-mode combination.

### Fixed

- Removed Qwen's four-request admission rejection so requests waiting for chat/embedding swaps queue instead of receiving premature 429 responses. Backend inference remains limited to four parallel slots per model in the exclusive swap group.
- Preserved model images, GPU allocations, persistent caches and resource limits during host-network restarts; existing images need no rebuild for this launcher change.

### Validation

- All 30 existing unit tests pass; Ruff, formatting and diff checks pass.
- Twelve mixed chat/embedding requests passed through the local gateway, including six simultaneous embedding requests; visible chat answers and 4096-dimensional vectors were verified.
- The external Hindsight integration retained and recalled a synthetic fact in 11.5 seconds after gateway deadlines and retain-only reasoning were configured. Those application-specific settings are not EasyLlama defaults.

## [v0.5.5] - 2026-08-30

Qwen llama.cpp migration with request-aware cross-container lifecycle handling.

### Added

- Added lifecycle sidecars and sleep/wake contracts so llama-swap can unload one backend before activating the other in an exclusive chat/embedding swap group.
- Added a Qwen-only 30-minute llama-swap idle TTL; other profiles retain the disabled global default.
- Added exact chat-to-embedding-to-chat regression coverage to catch corrupt output after unload/reload switching.

### Changed

- Switched the Qwen profile from vLLM/LMCache to llama.cpp with the multilingual RVN Heretic Q4_K_M GGUF, full 262,144-token context, Q8_0 KV cache, native RAM-backed prompt caching, and its embedded MTP head at draft depth two.
- Made the Qwen3.8 template the source of reasoning behavior through `--reasoning auto` and `--reasoning-preserve`; it accepts low, medium, and xhigh effort.
- Updated the API reference, chat-template guide, provider and tuning skills, launcher metadata, and Qwen example configuration for the shipped backend.

### Fixed

- Corrected skill scripts for repository-root config resolution, llama.cpp's `--n-gpu-layers` spelling, mode-aggregated logs, and the unserved `/ui/` route.
- Kept chat output coherent after switching to embeddings and back by enforcing serialized lifecycle transitions.

### Validation

- Clean rebuild of the Qwen llama.cpp and llama-swap images; all three containers healthy.
- Full-context Pi checks passed at every supported thinking level, including a 146,404-token long-context prompt.
- Public health, model, chat, completion, responses, embedding, and post-embedding chat checks passed.
- `30 passed, 1 skipped` in the host validation suite.

## [v0.5.3] - 2026-08-28

Patch release for mode-specific containers, centralized configuration, and the Qwen runtime refresh.

### Changed

- Replaced the generic `basic` server with dedicated llama.cpp and Qwen launchers, supervised mode-specific containers, private networks, health checks, and optional host publication.
- Centralized JSON configuration for modes, credentials, Docker settings, and independent CPU/RAM/swap profiles with validated floors and host-capacity controls.
- Split llama-swap, llama.cpp, vLLM, and LMCache builds into focused images with persistent BuildKit caches and bounded build parallelism.
- Updated Qwen to vLLM 0.28 and LMCache 0.5.4, including CUDA IPC scheduling, a 16 GiB L1 cache, and a model-aligned 40,960-token embedding context.
- Updated documentation, examples, project skills, scripts, and regression coverage for the new runtime architecture.

### Fixed

- Corrected vLLM compiler selection so clean CUDA builds do not pass compound ccache commands where executable paths are required.

## [v0.5.2] - 2026-08-17

Qwen profile tuned for the RTX 5090-specific checkpoint.

### Changed

- `qwen` profile now serves `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (RTX 5090-tuned ModelOpt weights) instead of `RadixArk/Qwen3.8-27B-NVFP4`.
- KV cache switched from TurboQuant k8v4 to FP8 at the full native 262,144-token context; GPU memory utilization raised to `0.96`.
- Dropped `--enforce-eager`, added `--quantization modelopt`, `--trust-remote-code`, and `--compilation-config.max_cudagraph_capture_size=4`; scheduler concurrency raised from 4 to 16 sequences.
- Tool-call parser switched from `qwen3_coder` to `qwen3_xml`; thinking enabled via the mounted `qwen3.8.jinja` template.
- Updated `README.md`, `API.md`, and the qwen config example to match the new flags and model target.

## [v0.5.1] - 2026-08-17

Patch release fixing the Qwen chat template shipping gap.

### Fixed

- Tracked `chat_template/qwen3.8.jinja` in git; `.gitignore` previously excluded it, so fresh clones and Docker builds were missing the template the `qwen` vLLM route references at `/chat_template/qwen3.8.jinja`.
- Corrected `chat_template/README.md`, which still claimed the vLLM chat route uses the model's native template.

## [v0.5.0] - 2026-08-17

Tuning release for the Qwen profile: 256K context, TurboQuant KV cache, multi-sequence scheduling, and pinned vLLM build.

### Changed

- Renamed the `mtp` runtime profile to `qwen`, including its config files, Docker target, image tag, and auxiliary llama-server path.
- Replaced the Unsloth MTP checkpoint with raw `RadixArk/Qwen3.8-27B-NVFP4` ModelOpt weights and disabled speculative decoding.
- Raised the Qwen profile's configured context from 128K to 256K tokens and switched the KV cache from FP8 to TurboQuant (k8v4); the auxiliary embedding server now runs a matching 262,144-token context.
- Raised Qwen chat concurrency from 1 to 4: `max-num-seqs` 4 and `max-num-batched-tokens` 8192, with GPU memory utilization raised to 0.94 and vLLM wrapper sleep level 2.
- Pinned the in-tree vLLM build to tag `v0.27.1` via the `VLLM_REF` Docker build arg and added `libcublas-dev-13-0` to the build dependencies.
- Enabled FlashAttention 2 for Qwen chat and serve a custom `qwen3.8.jinja` chat template with thinking enabled by default; dropped `--language-model-only`.
- Updated current README and API references for the Qwen profile's 256K configured context and TurboQuant KV cache.

## [v0.4.0] - 2026-08-15

Feature release adding the hybrid vLLM MTP runtime and simplifying shipped model profiles.

### Added

- Added explicit runtime backend metadata; the MTP profile selects vLLM while existing profiles remain llama.cpp-based.
- Added a hybrid MTP image: Qwen3.8-27B NVFP4 chat uses vLLM MTP speculation, while auxiliary GGUF routes remain on llama.cpp.
- Added llama-swap's upstream `vllm-wrapper` sleep/wake lifecycle for fast route switching.
- Added regression coverage for backend selection, MTP configuration, Docker targets, and effective-config security.
- Added verbose child-process logging with captured stdout, stderr, and exit status.
- Added `logs --tail N`; without `--tail`, logs continue to stream live.

### Changed

- Updated llama-swap from v208 to v250.
- Migrated MTP chat from Qwen3.6 GGUF to `unsloth/Qwen3.8-27B-NVFP4`.
- Set vLLM `--max-num-seqs` to 32 for MTP chat, limiting the scheduler to 32 sequences per iteration rather than setting a token-batch size.
- Corrected the documented Turboquant default and limited documented `POST /v1/messages` support to Lucebox.
- Renamed default Docker resources to `easyllama-local` and `easyllama-server-swap` so project naming no longer implies a llama.cpp-only runtime.
- Standardized public environment overrides on the `EASYLLAMA_*` project prefix across runtime settings, logging, warmup controls, and helper scripts. The former prefix is no longer supported.
- Added safe legacy-default container migration: default-name lifecycle commands remove `llamacpp-server-swap` before starting `easyllama-server-swap`; legacy image tags are retained for manual cleanup.
- Configured MTP startup to use eager execution and skip multimodal profiling; text-chat validation uses a 155,200-token context, FP8 KV cache, and 8 GiB native CPU KV offload.

### Fixed

- Host-side warmup prefetch now uses the Hugging Face token resolved from nested `config.json` credentials, while an explicit `HF_TOKEN` environment variable still takes precedence.

### Removed

- Removed the QMD generation, embedding-alias, and reranking models from every shipped profile.
- Removed the legacy `nvfp` llama.cpp mode, its Docker target, and its configuration template. The MTP profile continues to use the Qwen3.8 NVFP4 checkpoint through vLLM.

### Validation

- Clean image rebuild and MTP server restart.
- Qwen3.8-27B-NVFP4 MTP warmup completed successfully.
- `qwen-combo` returned HTTP 200 and the expected chat response through the public endpoint.

### Links

- Release: [v0.4.0](https://github.com/loopyd/easyllama/releases/tag/v0.4.0)
- Compare: [v0.3.16...v0.4.0](https://github.com/loopyd/easyllama/compare/v0.3.16...v0.4.0)

## [v0.3.16] - 2026-07-24

Patch release for deterministic, custom warmup output.

### Changed

- **Unified warmup progress**: Hugging Face transfers now use the same `Warming model #/#` prefix as worker loading, retaining downloaded size, total size, percentage, transfer speed, and ETA while a model is fetched.
- **Clean cached-model status**: Cache hits report `cached`, elapsed time, and llama-swap state without fabricating download telemetry.
- **Quiet default CLI output**: Default logging is `INFO`; warmup suppresses Hugging Face HTTP/advisory chatter and duplicate completion records.

### Fixed

- Removed duplicate progress close messages (`complete`/`close.*`) from warmup output.
- Keep the upstream warmup trigger open in the background so its cancellation cannot kill a model that is still loading.

### Validation

- `python test_download_progress.py`
- `python test_warmup_progress.py`
- `ruff check easyllama/servers/common.py easyllama/runtime.py easyllama/logger.py test_warmup_progress.py`
- `python -m compileall -q easyllama`

### Links

- Release: [v0.3.16](https://github.com/loopyd/easyllama/releases/tag/v0.3.16)
- Compare: [v0.3.15...v0.3.16](https://github.com/loopyd/easyllama/compare/v0.3.15...v0.3.16)

## [v0.3.15] - 2026-07-24

Patch release focused on accurately reporting model-download versus server-load progress.

### Added

- **Hugging Face cache prefetch progress**: Warmup now prefetches the configured Hugging Face model files and reports actual byte progress, transfer rate, and ETA while they download.
- Added narrow assertion-based checks for download progress formatting and warmup polling behavior.

### Changed

- **Warmup status wording**: Server-side model loading now reports elapsed time, state, and the initial HTTP status without implying that llama-swap exposes a download rate or ETA.

### Fixed

- Synced `easyllama.__version__` with the packaged project version.
- Ignore local `.pi-subagents/` artifacts.

### Validation

- `python test_download_progress.py`
- `python test_warmup_progress.py`

### Links

- Release: [v0.3.15](https://github.com/loopyd/easyllama/releases/tag/v0.3.15)
- Compare: [v0.3.14...v0.3.15](https://github.com/loopyd/easyllama/compare/v0.3.14...v0.3.15)

## [v0.3.14] - 2026-07-24

Feature release for pycurl-based download progress with rate/ETA and dependency migration to pyproject.toml.

### Added

- **Download progress with rate & ETA**: Replaced `urllib` download with `pycurl` in `_download_file()`. Downloads now log byte progress, current transfer speed, and ETA every five seconds during downloads.
- **Warmup timeout on health checks**: Added `timeout` parameter to `_http_json`, `_http_response`, and `model_status`; warmup polling now times out per-request instead of hanging indefinitely.
- **Elapsed time display in warmup**: Warmup reporter now shows `{downloaded}s elapsed, ETA <= {remaining}s` in its update template for clearer progress feedback.
- **`subprocess.Popen[str]` type annotation** in `ServerBase._run_foreground` for stricter typing.

### Changed

- **Dependencies moved to pyproject.toml**: Removed `requirements.txt` and `requirements-dev.txt`. All runtime dependencies (`colorama`, `docker`, `fastapi`, `huggingface_hub`, `jinja2`, `protobuf`, `pycurl`, `sentencepiece`, `torch`, `tqdm`, `transformers`, `uvicorn`) are now declared inline under `[project.dependencies]`. Dev dependencies use `[project.optional-dependencies] dev = ["ruff"]`.
- **Install commands updated in README**: Minimal install now uses `pip install .`; editable dev install uses `pip install -e ".[dev]"`.
- **Dockerfile build stage**: Runtime stage now creates venv and installs via `pip install /app` from pyproject.toml instead of `--no-deps` from requirements.txt.
- **Docker imports in runtime.py**: Replaced lazy `import docker` with explicit top-level imports from `docker`, `docker.errors`, and `docker.types` for better type safety.
- **API.md**: Updated `mtp` model references.

### Fixed

- Fixed `TimeoutError` / `URLError` not being caught in `_http_response()`, which could cause warmup hangs when the server is unresponsive.
- Fixed `self.docker.errors.*` attribute access that would break if `docker` wasn't imported at module level.
- Fixed `Popen[bytes]`/`Popen[str]` type mismatch in `runtime.py` `_stop_proc` (already fixed in prior release; reaffirmed).

### Validation

- `ruff check easyllama/` passes with zero errors.
- `pyright easyllama/` passes for all user code.
- Ran `.venv/bin/python -m compileall easyllama` after changes.

### Links

- Release: [v0.3.14](https://github.com/loopyd/easyllama/releases/tag/v0.3.14)
- Compare: [v0.3.13...v0.3.14](https://github.com/loopyd/easyllama/compare/v0.3.13...v0.3.14)

## [v0.3.13] - 2026-07-03

Refactoring release: unified repo source env vars, llama-swap concurrency limits, and CUDA unified memory for larger models.

### Changed

- **Unified repo source env vars**: Removed mode-specific `turboquant_llama_cpp_repo`, `turboquant_llama_cpp_ref`, `spiritbuun_llama_cpp_repo`, `spiritbuun_llama_cpp_ref`, `mtp_llama_cpp_repo`, and `mtp_llama_cpp_ref` from `pyproject.toml`, `Settings` dataclass, env var resolution, and Dockerfile build args. All llama.cpp variants now share one `LLAMA_CPP_REPO`/`LLAMA_CPP_REF` pair set via `LLAMACPP_LLAMA_CPP_REPO`/`LLAMACPP_LLAMA_CPP_REF` env vars or `pyproject.toml` defaults.
- **llama-swap concurrency limits**: Added `concurrencyLimit: 4` to every model in all five runtime configs (`basic`, `turboquant`, `mtp`, `spiritbuun`, `lucebox`). llama-swap now enforces at most 4 parallel requests per upstream model via its semaphore-based throttling.
- **CUDA unified memory**: Set `ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` in the Dockerfile runtime stage, allowing oversubscribed VRAM for larger models on RTX 5090 GPUs.

### Fixed

- Fixed ruff I001 (unsorted imports) in `config.py`, `helpers.py`, and `runtime.py`.
- Fixed ruff F401 (unused imports) in `config.py` (`ProgressReporter`, `shutil_which`).
- Fixed ruff F811 (duplicate `dataclass` import) in `config.py`.
- Fixed pyright `str | object` → `int()` type errors in `config.py` `load_settings`.
- Fixed pyright `Popen[bytes]` / `Popen[str]` mismatch in `runtime.py` `_stop_proc`.
- Fixed pyright signal handler type error in `runtime.py`.

### Validation

- `ruff check easyllama/` passes with zero errors.
- `pyright easyllama/` passes for all user code; 11 pre-existing docker SDK type-stub warnings remain.
- Ran `.venv/bin/python -m compileall easyllama` after changes.

### Links

- Release: [v0.3.13](https://github.com/loopyd/easyllama/releases/tag/v0.3.13)
- Compare: [v0.3.12...v0.3.13](https://github.com/loopyd/easyllama/compare/v0.3.12...v0.3.13)

## [v0.3.12] - 2026-05-10

Patch release focused on promoting the current MTP runtime profile into the tracked template.

### Changed

- Synced `config/config.mtp.yml.example` to the current live MTP layout, including the active macro names, `Q5_K_XL` chat target, `--fit off`, `--gpu-layers 99`, and explicit chat thread and polling settings.
- Kept the tracked MTP example aligned with the currently running local configuration so fresh checkouts can reproduce the same server arguments without hand-copying local edits.

### Validation

- Validated `config/config.mtp.yml.example` with `.github/skills/easyllama-provider/scripts/validate-config-yaml.sh`.
- Ran `./run.sh --mode mtp restart && ./run.sh --mode mtp warmup qwen3-chat` against the active MTP config before release.
- Ran `.venv/bin/python -m compileall easyllama` after the version bump.

### Links

- Release: [v0.3.12](https://github.com/loopyd/easyllama/releases/tag/v0.3.12)
- Compare: [v0.3.11...v0.3.12](https://github.com/loopyd/easyllama/compare/v0.3.11...v0.3.12)

## [v0.3.11] - 2026-05-09

Patch release focused on the config-layout cleanup, MTP warmup and tuning workflows, and lower-CPU MTP chat defaults.

### Added

- Added the `.github/skills/easyllama-tune/` skill with helper scripts and prompt assets for cache-quant comparisons, warmup probing, GPU-layer searches, and deterministic chat sample diffs.

### Changed

- Moved tracked mode templates under `config/`, updated repo defaults in `pyproject.toml`, ignored active `config/*.yml` files in `.gitignore`, and refreshed README setup and file-map guidance for the new layout.
- Updated the shipped MTP example to the validated RTX 5090 chat profile: `131072` context, `62` GPU layers, `q8_0` KV cache, `8` generation threads, `16` batch threads, and chat polling disabled.

### Fixed

- Added warmup progress logging so long model loads now report elapsed time, HTTP status, and upstream state instead of appearing stalled.
- Fixed stale `easyllama-tune` internal script references and removed the broken README mode-reference link.

### Validation

- Ran `./run.sh --mode mtp restart && ./run.sh --mode mtp warmup qwen3-chat`.
- Ran authenticated short and long `POST /v1/chat/completions` probes against `qwen3-chat` after the MTP thread and polling tuning.
- Validated `config/config.*.yml.example` with `.github/skills/easyllama-provider/scripts/validate-config-yaml.sh`.
- Ran `bash -n .github/skills/easyllama-tune/scripts/*.sh` and `.venv/bin/python -m compileall easyllama`.

### Links

- Release: [v0.3.11](https://github.com/loopyd/easyllama/releases/tag/v0.3.11)
- Compare: [v0.3.10...v0.3.11](https://github.com/loopyd/easyllama/compare/v0.3.10...v0.3.11)

## [v0.3.10] - 2026-05-05

Patch release focused on the Qwen 3.6 tool-call template correction and version metadata alignment.

### Fixed

- Serialized only the function schema inside the Qwen 3.6 `<tools>` block so tool-aware clients no longer receive wrapper objects instead of callable definitions.
- Emitted tool call IDs alongside assistant `<tool_call>` blocks and echoed `message.tool_call_id` inside `<tool_response>` blocks so tool responses can be matched back to the originating call.
- Synced `easyllama.__version__` with the packaged project version.

### Validation

- Ran a narrow Python smoke test to confirm `pyproject.toml` and `easyllama.__version__` now match and that `chat_template/qwen3.6.jinja` still parses as a Jinja template after the tool-call updates.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.10
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.9...v0.3.10

## [v0.3.9] - 2026-05-05

Patch release focused on catching mode-config regressions before rebuilds and shipping the cleaned Spiritbuun example macro layout.

### Fixed

- Added a host-side mode YAML validator under `.github/skills/easyllama-provider/scripts/validate-config-yaml.sh` so provider workflows now fail fast on YAML parse errors, duplicate `macros:` keys, and unresolved `${...}` references before spending time on Docker builds.
- Wired `.github/skills/easyllama-provider/scripts/rebuild-and-warmup.sh` to validate the effective mode config before `./run.sh --mode <mode> build`, exercising the same pre-build gate used during local provider validation.

### Changed

- Canonicalized `config.spiritbuun.yml.example` to the route-based macro schema already used by the active Spiritbuun config, keeping shared sampler flags centralized while preserving conservative shipped batch and completion defaults for the example.
- Updated the `easyllama-provider` skill instructions to require mode-YAML validation ahead of rebuilds.

### Validation

- Ran `.github/skills/easyllama-provider/scripts/validate-code.sh` successfully.
- Ran `.github/skills/easyllama-provider/scripts/validate-config-yaml.sh` successfully against `config.spiritbuun.yml` and `config.spiritbuun.yml.example`.
- Rebuilt, restarted, and warmed the `spiritbuun` runtime successfully through `.github/skills/easyllama-provider/scripts/rebuild-and-warmup.sh`, confirming the new pre-build validator runs in the live workflow.
- Ran `.github/skills/easyllama-provider/scripts/test-public-endpoints.sh spiritbuun` successfully against the rebuilt runtime.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.9
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.8...v0.3.9

## [v0.3.8] - 2026-05-04

Patch release focused on the Spiritbuun agent stall regression left behind by `v0.3.7`.

### Fixed

- Scoped Spiritbuun `qwen3-chat` to `--n-predict -1` so long agentic turns no longer inherit the shared `8192` token cap that could leave multi-step chats hanging mid-run.
- Added `--cache-ram 0` to the shipped Spiritbuun DFlash chat launcher so the draft path stays on the intended cache configuration during extended agent sessions.

### Validation

- Ran `.github/skills/easyllama-provider/scripts/validate-code.sh` successfully.
- Rebuilt, restarted, and warmed the `spiritbuun` runtime successfully.
- Ran `.github/skills/easyllama-provider/scripts/test-public-endpoints.sh spiritbuun` successfully against the rebuilt runtime.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.8
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.7...v0.3.8

## [v0.3.7] - 2026-05-04

Patch release focused on Spiritbuun long-conversation continuity.

### Fixed

- Enabled `--keep -1` and `--context-shift` in the shipped Spiritbuun chat config so long multi-turn sessions can continue shifting context instead of stalling once the slot fills.

### Changed

- Kept `--ignore-eos` disabled for the default Spiritbuun chat profile after upstream review showed it is intended for infinite-text generation rather than standard chat turns.
- Wrapped the duplicate-container startup error string in `easyllama/runtime.py` so the shipped host-side Ruff gate passes cleanly.

### Validation

- Ran `.github/skills/easyllama-provider/scripts/validate-code.sh` successfully.
- Rebuilt, restarted, and warmed the `spiritbuun` runtime successfully.
- Ran `.github/skills/easyllama-provider/scripts/test-public-endpoints.sh spiritbuun` successfully against the rebuilt runtime.

### Links

- Release: https://github.com/loopyd/easyllama/releases/tag/v0.3.7
- Compare: https://github.com/loopyd/easyllama/compare/v0.3.6...v0.3.7

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
