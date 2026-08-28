# easy llama(cpp)

Run local llama.cpp and vLLM backends behind one `llama-swap` endpoint at `http://127.0.0.1:8080`.

Project goal: one host command surface, one public port, one shared model cache, multiple backend modes.

## Contents

- [easy llama(cpp)](#easy-llamacpp)
  - [Contents](#contents)
  - [At a glance](#at-a-glance)
  - [Modes](#modes)
  - [System requirements](#system-requirements)
  - [Install](#install)
  - [Quick start](#quick-start)
    - [1. Create credentials](#1-create-credentials)
    - [2. Copy mode config templates](#2-copy-mode-config-templates)
    - [3. Build, start, warm](#3-build-start-warm)
    - [4. Verify runtime](#4-verify-runtime)
  - [Common commands](#common-commands)
  - [File map](#file-map)
  - [Environment overrides](#environment-overrides)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)

## At a glance

- One entrypoint: `./run.sh`
- One API base URL: `http://127.0.0.1:8080` by default; override the validated IPv4, IPv6, or hostname with `--host`
- Persistent, gitignored host caches under `cache/`: `models/` for Hugging Face repositories, `root/` for `/root/.cache`, `pkg/` for the package manager, and `python/` for pip
- One shared mmproj asset directory: `mmproj/`
- Stable model IDs exposed through `/v1/models`
- Per-model `concurrencyLimit: 4` in llama-swap configs to cap parallel requests
- Qwen vLLM profile with Unsloth Qwen3.8 NVFP4 weights, 131,072-token context, FP8 KV cache, MTP, and thinking enabled
- LMCache 0.5.4 with a 16 GiB pinned-RAM L1 cache for Qwen prompt reuse
- Higher process limit, 32 GiB shared memory, and unlimited memlock for the vLLM runtime
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` for oversubscribed llama.cpp VRAM on RTX 5090
- Lazy downloads by default; use warmup for predictable first-request latency

## Modes

Choose a mode by backend behavior; the setup flow is the same for all five modes.

- Mode-specific defaults live in the tracked templates under `config/`.

| Mode | Best for | `qwen3-chat` backend | Default chat weights | Extra API surface |
| --- | --- | --- | --- | --- |
| `llamacpp` | Plain llama.cpp path | `easyllama server llamacpp` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` | none |
| `turboquant` | Turboquant KV-cache experiments | `easyllama server turboquant` | `unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL` | none |
| `qwen` | Qwen3.8 RTX 5090 Unsloth NVFP4 + MTP with llama.cpp auxiliary routes | `vllm` via `vllm-wrapper` | `unsloth/Qwen3.8-27B-NVFP4` | none |
| `spiritbuun` | buun-llama-cpp DFlash experiments | `easyllama server spiritbuun` | `unsloth/Qwen3.6-27B-GGUF:Q5_K_M` + `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` | none |
| `lucebox` | Luce dflash/pflash experiments | `easyllama server lucebox` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` + `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` | `POST /v1/messages` |

The `qwen` profile serves `unsloth/Qwen3.8-27B-NVFP4` through vLLM with the checkpoint's native MTP head drafting two tokens per step, as recommended by Unsloth. It uses a 131,072-token context, a fixed 5 GiB FP8 KV cache, prefix caching, text-only loading, 0.94 GPU memory utilization, and the mounted Qwen3.8 template with thinking enabled. LMCache 0.5.4 adds a 16 GiB pinned host-RAM L1 cache through `LMCacheMPConnector`; its 1,600-token chunks match vLLM's Qwen3.8 unified attention block, and separate hybrid object groups plus aligned Mamba caching preserve GDN state reuse. Four scheduler sequences match the four-request public concurrency cap, with a 2,048-token scheduler budget on a 32 GiB RTX 5090. The vLLM 0.28 runtime enables asynchronous scheduling for speculative decoding; the profile explicitly skips FP4 GEMM autotuning and disables expandable CUDA allocator segments because LMCache's CUDA IPC handles require stable physical pages. Its hybrid image keeps the embedding route on llama.cpp; llama-swap stops and reloads the Qwen worker when switching routes because LMCache's CUDA IPC connector is incompatible with vLLM's sleep-mode allocator.

## System requirements

- Linux with Bash `4.1+`
- Docker daemon running
- `docker buildx`
- NVIDIA drivers and working `nvidia-smi`
- NVIDIA container runtime in Docker
- Blackwell GPU for the Qwen profile's NVFP4 checkpoint
- Python `3.11+`
- `curl`
- `jq`

## Install

Minimal host setup for running `./run.sh` from checkout:

```bash
python3 -m venv ./.venv
.venv/bin/activate
python -m pip install .
```

Editable development install:

```bash
python3 -m venv ./.venv
.venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick start

Fastest path from fresh checkout to working local endpoint.

### 1. Create credentials

```bash
cp config.json.example config.json
```

Set:

- `credentials.hf_token` for private or rate-limited Hugging Face pulls; `HF_TOKEN` takes precedence
- `credentials.api_key` for `Authorization: Bearer ...` protection on `/v1/*` routes; `API_KEY` takes precedence

`resources` centralizes host capacity, profile floors, and role assignments. `modes` centralizes each mode's source repository and required services. The active llama-swap path is derived as `config/config.<mode>.yml`; set `llama_swap_override` only for a custom path.

### 2. Copy mode config templates

```bash
cp config/config.llamacpp.yml.example config/config.llamacpp.yml
cp config/config.turboquant.yml.example config/config.turboquant.yml
cp config/config.spiritbuun.yml.example config/config.spiritbuun.yml
cp config/config.qwen.yml.example config/config.qwen.yml
cp config/config.lucebox.yml.example config/config.lucebox.yml
```

Edit configs as needed. For more config detail, see `llama-swap` docs:
[llama-swap configuration docs](https://github.com/mostlygeek/llama-swap/blob/main/docs/configuration.md)

If `config/config.<mode>.yml` does not exist, `run.sh` falls back to the matching example file in `config/`.

### 3. Build, start, warm

```bash
./run.sh --mode <mode> build
./run.sh --mode <mode> start
./run.sh --mode <mode> warmup
```

Pass model IDs to warm only a subset:

```bash
./run.sh --mode <mode> warmup qwen3-chat qwen3-embeddings
```

With no model arguments, warmup hits every model exposed by `/v1/models`.

### 4. Verify runtime

```bash
API_KEY="$(jq -r '.credentials.api_key // empty' config.json)"
AUTH=()
if [[ -n "${API_KEY}" ]]; then
  AUTH=(-H "Authorization: Bearer ${API_KEY}")
fi

./run.sh status
curl -sS http://127.0.0.1:8080/health
curl -sS "${AUTH[@]}" http://127.0.0.1:8080/v1/models | jq -r '.data[].id'
```

## Common commands

Most-used host commands through `./run.sh`.

| Command | Action |
| --- | --- |
| `./run.sh build` | Build default `llamacpp` image |
| `./run.sh --mode <mode> build` | Build every image in the selected mode's container stack |
| `./run.sh --mode <mode> build --type {llamaswap,llamacpp,vllm,lmcache}` | Compile and build an isolated image role |
| `./run.sh start` | Start default `llamacpp` container |
| `./run.sh --mode <mode> start` | Start selected mode on `127.0.0.1` |
| `./run.sh --mode <mode> --host 0.0.0.0 start` | Publish the selected mode API on every host interface |
| `./run.sh warmup [model...]` | Preload one or more models through `llama-swap` |
| `./run.sh restart` | Replace every container and dependency network in the selected mode stack |
| `./run.sh stop` | Stop and remove every selected-mode container and its private network |
| `./run.sh logs` | Follow aggregated, container-tagged logs for the selected mode stack |
| `./run.sh logs --tail N` | Merge the last N lines per selected-mode container by Docker timestamp |
| `./run.sh status` | Show runtime status and built images |
| `./run.sh clean` | Remove the current mode stack, private network, images, and all host caches |
| `./run.sh clean --all-images` | Remove all mode images and the runtime container; empty all host caches, including models |
| `./run.sh serve` | Run `llama-swap` inside container |
| `./run.sh server ...` | Run mode-specific upstream server directly |
| `./run.sh help` | Show CLI help |

## File map

| Path | Purpose |
| --- | --- |
| `run.sh` | Host and container entrypoint |
| `config.json` | Optional local nested configuration and credentials |
| `config.json.example` | Complete tracked configuration template |
| `config/config.llamacpp.yml` | Editable config for `llamacpp` |
| `config/config.turboquant.yml` | Editable config for `turboquant` |
| `config/config.spiritbuun.yml` | Editable config for `spiritbuun` |
| `config/config.qwen.yml` | Editable config for `qwen` |
| `config/config.lucebox.yml` | Editable config for `lucebox` |
| `config/config.llamacpp.yml.example` | Tracked `llamacpp` template |
| `config/config.turboquant.yml.example` | Tracked `turboquant` template |
| `config/config.spiritbuun.yml.example` | Tracked `spiritbuun` template |
| `config/config.qwen.yml.example` | Tracked `qwen` template |
| `config/config.lucebox.yml.example` | Tracked `lucebox` template |
| `cache/models/` | Shared Hugging Face cache |
| `cache/root/` | Container `/root/.cache` |
| `cache/pkg/` | System package-manager cache |
| `cache/python/` | Python package cache |
| `mmproj/` | Shared mmproj assets |
| `chat_template/` | Mounted chat templates |
| `docker/` | One Dockerfile per composable base, Python, builder, and isolated runtime stage |
| `easyllama/` | Python package: runtime, CLI, Docker compiler/orchestration, launchers |
| `tests/unit/` | Fast isolated unit and Docker-compiler tests |
| `tests/integration/` | Docker/external-service integration tests |
| `API.md` | API reference and request examples |
| `CHANGELOG.md` | Release history |

Run all tests with `python -m pytest`, or select categories with `-m unit`, `-m docker`, or `-m integration`. Runtime configuration is a serializable nested Pydantic model grouped under `dirs`, `runtime`, `docker`, `resources`, `modes`, `locale`, `lmcache`, `credentials`, `warmup`, and `llama_swap_override`. Built-in defaults load first, existing `config.json` or `--config-file PATH` loads next, matching `EASYLLAMA_*` variables override the file, and explicit CLI options win last. Each mode's managed images share a private `easyllama-<mode>` Docker network. On `start`, local llama-swap `cmd` entries are compiled into static remote proxies while explicit backend container contracts own command, port, health endpoint, stop signal, GPU, and environment lifecycle. Only llama-swap publishes the host API port, so `./run.sh --mode <mode> start` remains the public interface.

## Environment overrides

Use the `EASYLLAMA_*` project prefix. The former environment-variable prefix is no longer supported.

| Preferred variable | Purpose |
| --- | --- |
| `EASYLLAMA_MODE` | Select `llamacpp`, `turboquant`, `qwen`, `spiritbuun`, or `lucebox` |
| `EASYLLAMA_IMAGE_NAME` | Override the default `easyllama` image repository |
| `EASYLLAMA_CONTAINER_NAME` | Override the default `easyllama-server-swap` container name |
| `EASYLLAMA_LLAMA_CPP_REPO` / `EASYLLAMA_LLAMA_CPP_REF` | Override every mode-specific llama.cpp source |
| `EASYLLAMA_LUCEBOX_HUB_REPO` / `EASYLLAMA_LUCEBOX_HUB_REF` | Override the Lucebox dflash hub source |
| `EASYLLAMA_LS_CONFIG_FILE` | Use an explicit llama-swap config file |
| `EASYLLAMA_HOST` | Change the published IPv4, IPv6, or hostname from the `127.0.0.1` default |
| `EASYLLAMA_HOST_PORT` | Change the published host port |
| `EASYLLAMA_LMCACHE_CHUNK_SIZE` | Override LMCache chunk size (`1600`) |
| `EASYLLAMA_LMCACHE_L1_SIZE_GB` | Override LMCache L1 host RAM in GiB (`16`) |
| `EASYLLAMA_ROOT` / `EASYLLAMA_MODELS_DIR` | Override project root or model cache paths |
| `HF_TOKEN` or `EASYLLAMA_HF_TOKEN` | Override the Hugging Face token |
| `API_KEY` or `EASYLLAMA_API_KEY` | Override the local API key |
| `EASYLLAMA_MMPROJ_FILE` / `EASYLLAMA_HF_MMPROJ` | Select a local, URL, or Hugging Face mmproj asset |
| `EASYLLAMA_CMAKE_CUDA_ARCHITECTURES` | Override auto-detected CUDA architecture values |
| `EASYLLAMA_AVAILABLE_CPUS` | Override host CPUs used to calculate build parallelism and container CPU limits |
| `EASYLLAMA_AVAILABLE_RAM_GIB` | Override host RAM used for weighted container limits; reserves at least 16 GiB for the host |
| `EASYLLAMA_AVAILABLE_SWAP_GIB` | Override host swap added to each container's memory-plus-swap limit |

Use `--config-file PATH` to load nested JSON away from the default `config.json`. If `config.json` contains `credentials.api_key`, `/v1/*` routes require `Authorization: Bearer <api_key>`.

### Docker image names

Images use the `easyllama` repository. Backend-specific roles keep their mode in the tag, such as `easyllama:cuda13-qwen-vllm` and `easyllama:cuda13-qwen-llamacpp`. Shared roles are mode-agnostic: `easyllama:cuda13-llamaswap` and `easyllama:cuda13-lmcache` are reused by every compatible stack.

On the first default-name `start`, `restart`, `stop`, or `clean` after upgrading, EasyLlama removes the legacy `llamacpp-server-swap` container. Custom container names are never migrated automatically.

## Troubleshooting

Fast map from symptom to likely fix.

| Problem | Likely cause | What to do |
| --- | --- | --- |
| `docker buildx` build fails fast | Buildx missing or not bootstrapped | Install Buildx, then run `docker buildx inspect --bootstrap` |
| First request is slow | Model download or first load happening lazily | Run `./run.sh warmup ...` first |
| `POST /v1/messages` fails | The route is only supported by `lucebox` | Restart with `./run.sh --mode lucebox start` |
| `/v1/models` returns `401` | API key enabled | Send `Authorization: Bearer <api_key>` |
| Config edit does nothing | Wrong mode file edited or `EASYLLAMA_LS_CONFIG_FILE` set | Check active mode and config path |
| Python change seems ignored | Running image stale | Rebuild affected mode, then restart |
| Port `8080` busy | Another process owns host port | Start with `EASYLLAMA_HOST_PORT=8090 ./run.sh start` |
| Private HF downloads fail | No usable HF token | Set `credentials.hf_token` in `config.json` or export `HF_TOKEN` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
