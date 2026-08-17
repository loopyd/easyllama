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
- One API base URL: `http://127.0.0.1:8080`
- One shared Hugging Face cache: `models/`
- One shared mmproj asset directory: `mmproj/`
- Stable model IDs exposed through `/v1/models`
- Per-model `concurrencyLimit: 4` in llama-swap configs to cap parallel requests
- Qwen vLLM profile with raw ModelOpt NVFP4 weights, 256K configured context, and TurboQuant KV cache
- Higher process limit and 8 GiB shared memory for the vLLM runtime
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` for oversubscribed llama.cpp VRAM on RTX 5090
- Lazy downloads by default; use warmup for predictable first-request latency

## Modes

Choose a mode by backend behavior; the setup flow is the same for all five modes.

- Mode-specific defaults live in the tracked templates under `config/`.

| Mode | Best for | `qwen3-chat` backend | Default chat weights | Extra API surface |
| --- | --- | --- | --- | --- |
| `basic` | Plain llama.cpp path | `llama-server-basic` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` | none |
| `turboquant` | Turboquant KV-cache experiments | `llama-server-turboquant` | `unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL` | none |
| `qwen` | Qwen3.8 NVFP4 inference with llama.cpp auxiliary routes | `vllm` via `vllm-wrapper` | `RadixArk/Qwen3.8-27B-NVFP4` | none |
| `spiritbuun` | buun-llama-cpp DFlash experiments | `easyllama server spiritbuun` | `unsloth/Qwen3.6-27B-GGUF:Q5_K_M` + `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` | none |
| `lucebox` | Luce dflash/pflash experiments | `easyllama server lucebox` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` + `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` | `POST /v1/messages` |

The `qwen` profile serves raw ModelOpt Qwen3.8-27B NVFP4 weights through vLLM without speculative decoding. On a 32 GiB Blackwell GPU, FP8 KV cache uses an 8 GiB native CPU offload buffer while vLLM dynamically allocates the GPU-resident KV cache. Its hybrid image keeps the embedding route on llama.cpp. llama-swap v250 sleeps the vLLM worker when switching routes and wakes it on demand.

`--max-num-seqs 32` lets one vLLM scheduler iteration process up to 32 sequences. It is a sequence-concurrency limit, not a 32-token batch size. llama-swap still limits each public model route to four parallel requests.

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
cp auth.json.example auth.json
```

Set:

- `hf_token` for private or rate-limited Hugging Face pulls; build, startup, and host-side warmup prefetch read it from `auth.json` unless `HF_TOKEN` is already set
- `api_key` for `Authorization: Bearer ...` protection on `/v1/*` routes

### 2. Copy mode config templates

```bash
cp config/config.basic.yml.example config/config.basic.yml
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
API_KEY="$(jq -r '.api_key // empty' auth.json)"
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
| `./run.sh build` | Build default `basic` image |
| `./run.sh --mode <mode> build` | Build selected mode image |
| `./run.sh start` | Start default `basic` container |
| `./run.sh --mode <mode> start` | Start selected mode |
| `./run.sh warmup [model...]` | Preload one or more models through `llama-swap` |
| `./run.sh restart` | Restart selected mode container |
| `./run.sh stop` | Stop and remove runtime container |
| `./run.sh logs` | Follow runtime logs |
| `./run.sh status` | Show runtime status and built images |
| `./run.sh clean` | Remove current mode image and container |
| `./run.sh clean --all-images` | Remove all mode images and runtime container |
| `./run.sh serve` | Run `llama-swap` inside container |
| `./run.sh server ...` | Run mode-specific upstream server directly |
| `./run.sh help` | Show CLI help |

## File map

| Path | Purpose |
| --- | --- |
| `run.sh` | Host and container entrypoint |
| `auth.json` | Local Hugging Face token and optional API key |
| `auth.json.example` | Credential template |
| `config/config.basic.yml` | Editable config for `basic` |
| `config/config.turboquant.yml` | Editable config for `turboquant` |
| `config/config.spiritbuun.yml` | Editable config for `spiritbuun` |
| `config/config.qwen.yml` | Editable config for `qwen` |
| `config/config.lucebox.yml` | Editable config for `lucebox` |
| `config/config.basic.yml.example` | Tracked `basic` template |
| `config/config.turboquant.yml.example` | Tracked `turboquant` template |
| `config/config.spiritbuun.yml.example` | Tracked `spiritbuun` template |
| `config/config.qwen.yml.example` | Tracked `qwen` template |
| `config/config.lucebox.yml.example` | Tracked `lucebox` template |
| `models/` | Shared Hugging Face cache |
| `mmproj/` | Shared mmproj assets |
| `chat_template/` | Mounted chat templates |
| `easyllama/` | Python package: runtime, CLI, Docker orchestration, launchers |
| `API.md` | API reference and request examples |
| `CHANGELOG.md` | Release history |

## Environment overrides

Use the `EASYLLAMA_*` project prefix. The former environment-variable prefix is no longer supported.

| Preferred variable | Purpose |
| --- | --- |
| `EASYLLAMA_MODE` | Select `basic`, `turboquant`, `qwen`, `spiritbuun`, or `lucebox` |
| `EASYLLAMA_IMAGE_NAME` | Override the default `easyllama-local` image repository |
| `EASYLLAMA_CONTAINER_NAME` | Override the default `easyllama-server-swap` container name |
| `EASYLLAMA_LLAMA_CPP_REPO` / `EASYLLAMA_LLAMA_CPP_REF` | Override llama.cpp source used by llama.cpp-backed routes |
| `EASYLLAMA_LUCEBOX_HUB_REPO` / `EASYLLAMA_LUCEBOX_HUB_REF` | Override the Lucebox dflash hub source |
| `EASYLLAMA_LS_CONFIG_FILE` | Use an explicit llama-swap config file |
| `EASYLLAMA_HOST_PORT` | Change the published host port |
| `EASYLLAMA_AUTH_FILE` | Use a different auth JSON file |
| `HF_TOKEN` or `EASYLLAMA_HF_TOKEN` | Override the Hugging Face token |
| `API_KEY` or `EASYLLAMA_API_KEY` | Override the local API key |
| `EASYLLAMA_MMPROJ_FILE` / `EASYLLAMA_HF_MMPROJ` | Select a local, URL, or Hugging Face mmproj asset |
| `EASYLLAMA_CMAKE_CUDA_ARCHITECTURES` | Override auto-detected CUDA architecture values |

If `auth.json` contains `api_key`, `/v1/*` routes require `Authorization: Bearer <api_key>`.

### Default Docker name migration

On the first default-name `start`, `restart`, `stop`, or `clean` after upgrading,
EasyLlama removes the legacy `llamacpp-server-swap` container before continuing.
Custom container names are never migrated automatically.

To reuse an existing default Qwen image without rebuilding, retag it first:

```bash
docker tag llamacpp-local:cuda13-qwen easyllama-local:cuda13-qwen
./run.sh --mode qwen restart
```

After confirming the new container is healthy, the old image tag may be removed
manually. EasyLlama does not delete legacy image tags during migration.

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
| Private HF downloads fail | No usable HF token | Set `hf_token` in `auth.json` or export `HF_TOKEN` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
