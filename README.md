# easy llama(cpp)

Run local `llama.cpp`-style backends behind one `llama-swap` endpoint at `http://127.0.0.1:8080`.

Project goal: one host command surface, one public port, one shared model cache, multiple backend modes.

## Contents

- [easy llama(cpp)](#easy-llamacpp)
  - [Contents](#contents)
  - [At glance](#at-glance)
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
  - [Reader path](#reader-path)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)

## At glance

Why reader cares:

- One entrypoint: `./run.sh`
- One API base URL: `http://127.0.0.1:8080`
- One shared Hugging Face cache: `models/`
- One shared mmproj asset directory: `mmproj/`
- Stable model IDs exposed through `/v1/models`
- Lazy downloads by default; use warmup for predictable first-request latency

## Modes

Pick mode by backend behavior, not by install flow. Setup path stays same.

- Mode-specific defaults live in the tracked templates under `config/`.

| Mode | Best for | `qwen3-chat` backend | Default chat weights | Extra API surface |
| --- | --- | --- | --- | --- |
| `basic` | Plain llama.cpp path | `llama-server-basic` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` | none |
| `turboquant` | Turboquant KV-cache experiments | `llama-server-turboquant` | `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive:Q5_K_P` | none |
| `mtp` | MTP experiments (separate llama.cpp fork/config) | `llama-server-basic` (runtime mode) | `havenoammo/Qwen3.6-27B-MTP-UD-GGUF:Qwen3.6-27B-MTP-UD-Q6_K_XL.gguf` | none |
| `spiritbuun` | buun-llama-cpp DFlash experiments | `easyllama server spiritbuun` | `unsloth/Qwen3.6-27B-GGUF:Q5_K_M` + `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` | none |
| `lucebox` | Luce dflash/pflash experiments | `easyllama server lucebox` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` + `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` | `POST /v1/messages` |

## System requirements

- Linux with Bash `4.1+`
- Docker daemon running
- `docker buildx`
- NVIDIA drivers and working `nvidia-smi`
- NVIDIA container runtime in Docker
- Python `3.11+`
- `curl`
- `jq`

## Install

Minimal host setup for running `./run.sh` from checkout:

```bash
python3 -m venv ./.venv
.venv/bin/activate
python -m pip install -r ./requirements.txt
```

Editable development install:

```bash
python3 -m venv ./.venv
.venv/bin/activate
python -m pip install -e . -r ./requirements.txt -r ./requirements-dev.txt
```

## Quick start

Fastest path from fresh checkout to working local endpoint.

### 1. Create credentials

```bash
cp auth.json.example auth.json
```

Set:

- `hf_token` for private or rate-limited Hugging Face pulls
- `api_key` for `Authorization: Bearer ...` protection on `/v1/*` routes

### 2. Copy mode config templates

```bash
cp config/config.basic.yml.example config/config.basic.yml
cp config/config.turboquant.yml.example config/config.turboquant.yml
cp config/config.spiritbuun.yml.example config/config.spiritbuun.yml
cp config/config.mtp.yml.example config/config.mtp.yml
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

Pass model IDs to warm only subset:

```bash
./run.sh --mode <mode> warmup qwen3-chat qmd-rerank
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
| `config/config.mtp.yml` | Editable config for `mtp` |
| `config/config.lucebox.yml` | Editable config for `lucebox` |
| `config/config.basic.yml.example` | Tracked `basic` template |
| `config/config.turboquant.yml.example` | Tracked `turboquant` template |
| `config/config.spiritbuun.yml.example` | Tracked `spiritbuun` template |
| `config/config.mtp.yml.example` | Tracked `mtp` template |
| `config/config.lucebox.yml.example` | Tracked `lucebox` template |
| `models/` | Shared Hugging Face cache |
| `mmproj/` | Shared mmproj assets |
| `chat_template/` | Mounted chat templates |
| `easyllama/` | Python package: runtime, CLI, Docker orchestration, launchers |
| `API.md` | API reference and request examples |
| `CHANGELOG.md` | Release history |

## Environment overrides

| Variable | Purpose |
| --- | --- |
| `LLAMACPP_MODE` | Select `basic`, `turboquant`, `mtp`, `spiritbuun`, or `lucebox` |
| `LLAMACPP_LS_CONFIG_FILE` | Use explicit config file instead of mode lookup |
| `LLAMACPP_HOST_PORT` | Change published host port |
| `LLAMACPP_AUTH_FILE` | Use different auth JSON file |
| `HF_TOKEN` or `LLAMACPP_HF_TOKEN` | Override Hugging Face token |
| `API_KEY` or `LLAMACPP_API_KEY` | Override local API key |
| `LLAMACPP_MMPROJ_FILE` | Use local mmproj path, `mmproj/...`, or URL |
| `LLAMACPP_HF_MMPROJ` | Use HF mmproj asset as `owner/repo/file.gguf` |
| `LLAMACPP_CMAKE_CUDA_ARCHITECTURES` | Override auto-detected CUDA arch build values |

Notes:

- If `auth.json` contains `api_key`, `/v1/*` routes require `Authorization: Bearer <api_key>`.
- `LLAMACPP_LS_CONFIG_FILE` overrides mode-based config selection.

## Reader path

New reader:

1. Read this file for setup and mode selection.
2. Read [API.md](API.md) for endpoint usage.
3. Read [CHANGELOG.md](CHANGELOG.md) for release history and upgrade context.

## Troubleshooting

Fast map from symptom to likely fix.

| Problem | Likely cause | What to do |
| --- | --- | --- |
| `docker buildx` build fails fast | Buildx missing or not bootstrapped | Install Buildx, then run `docker buildx inspect --bootstrap` |
| First request is slow | Model download or first load happening lazily | Run `./run.sh warmup ...` first |
| `POST /v1/messages` fails | Not running `lucebox` mode | Restart with `./run.sh --mode lucebox start` |
| `/v1/models` returns `401` | API key enabled | Send `Authorization: Bearer <api_key>` |
| Config edit does nothing | Wrong mode file edited or `LLAMACPP_LS_CONFIG_FILE` set | Check active mode and config path |
| Python change seems ignored | Running image stale | Rebuild affected mode, then restart |
| Port `8080` busy | Another process owns host port | Start with `LLAMACPP_HOST_PORT=8090 ./run.sh start` |
| Private HF downloads fail | No usable HF token | Set `hf_token` in `auth.json` or export `HF_TOKEN` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

See [LICENSE](LICENSE).
