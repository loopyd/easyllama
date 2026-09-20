# easy llama(cpp)

Run local llama.cpp and vLLM backends behind one `llama-swap` endpoint at `http://127.0.0.1:8080`.

Project goal: one host command surface, one public port, one shared model cache, multiple backend modes.

## Contents

- [easy llama(cpp)](#easy-llamacpp)
  - [Contents](#contents)
  - [At a glance](#at-a-glance)
  - [Modes](#modes)
    - [Docker networking](#docker-networking)
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

- One command: `./run.sh`
- One API URL: `http://127.0.0.1:8080`
- Shared model downloads under `cache/`
- Stable model names from `/v1/models`
- Qwen mode keeps GPU embeddings and compact reranking resident together, swapping them for large chat
- Other profiles share the GPU through exclusive model swapping
- Use warmup to avoid a slow first request

## Modes

### Docker networking

`docker.network_mode` defaults to `bridge`. Set it to `host` (or set
`EASYLLAMA_NETWORK_MODE=host`) and restart the selected stack to share the Linux host
network. The proxy binds to `runtime.host:runtime.host_port`; no ports are
published. Backend and lifecycle listeners bind only to `127.0.0.1`: each model
reserves two consecutive ports (its API port plus a backend slot at +1 for
multi-socket backends such as FreeToken's torch.distributed store), lifecycle
listeners follow the reserved band, and each profile declares a distinct
`startPort` base (Qwen 9000, GLM-5.3 Flash 9100, llama.cpp 9200, TurboQuant 9300,
SpiritBuun 9400, Lucebox 9500) so co-resident models from another profile cannot
collide. These host ports must be free; do not run multiple profiles
concurrently. Host mode runs the existing
llama-swap binary directly under Docker's init process and needs no image rebuild.
LMCache-dependent profiles reject host mode. Host mode removes network isolation
and permits host-routed egress; it does not change resource limits or model caches.

The Qwen profile uses `concurrencyLimit: 0` for chat, embeddings and reranking. This disables
llama-swap's early admission rejection, allowing requests to wait during model
switching rather than returning 429 after four waiting requests. It does not
increase inference concurrency: chat and embeddings retain `--parallel 4`, while
reranking has two native slots. GPU chat and the co-resident search models occupy
mutually exclusive groups. Bound upstream bulk concurrency
and use timeouts that cover model unloading, loading, queueing and generation.

Choose a mode by backend behavior; the setup flow is the same for all six modes.

- Mode-specific defaults live in the tracked templates under `config/`.

| Mode | Best for | `qwen3-chat` backend | Default chat weights | Extra API surface |
| --- | --- | --- | --- | --- |
| `llamacpp` | Plain llama.cpp path | `easyllama server llamacpp` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` | none |
| `turboquant` | Turboquant KV-cache experiments | `easyllama server turboquant` | `unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL` | none |
| `qwen` | Qwen3.8 RVN Heretic at its native 262K context on one RTX 5090 | `easyllama server qwen` | `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:RVN-Q4_K_M-multilingual-mtp.gguf` | `POST /v1/rerank` |
| `spiritbuun` | buun-llama-cpp DFlash experiments | `easyllama server spiritbuun` | `unsloth/Qwen3.6-27B-GGUF:Q5_K_M` + `Ardenzard/Qwen3.6-27B-DFlash-GGUF:Qwen3.6-27B-DFlash-Q5_K_M.gguf` | none |
| `lucebox` | Luce dflash/pflash experiments | `easyllama server lucebox` | `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` + `KingsonHO/Qwen3.6-27B-DFlash:model.safetensors` | `POST /v1/messages` |
| `glm5.3-flash` | GLM-5.3 Flash 320B MoE (18B active) on the FreeToken runtime | `easyllama server glm5.3-flash` | `RedHatAI/GLM-5.3-Flash-NVFP4` (NVFP4 HF checkpoint, FreeToken offload) | `POST /v1/messages` |

The `glm5.3-flash` mode runs the FreeToken runtime, not llama.cpp: FreeToken is installed from the pinned `FlashML-org/FreeToken` repository into an isolated venv (`/opt/ft-venv`) inside the dedicated `freetoken` image role, and the runtime stage merges the CUDA 13 compiler (nvcc) because FreeToken JIT-compiles its kernels on first use. GLM-5.3 Flash is a 320B-total / 18B-active MoE with hybrid linear (KDA) plus sparse (DSA) attention; only the eleven DSA layers grow KV, so the full 262,144-token context costs about 2.8 GiB of KV (bf16) and the profile pins it at `--max-seq-len-override`, `--num-tokens` and `--kv-reserve-tokens`. NVFP4 routed experts live off-VRAM: FreeToken keeps an LRU expert cache in host RAM and streams misses from the checkpoint on the host SSD (`cache/models`, ~160 GiB download on first start). The mode exposes `glm53-chat` and the 30-minute idle timer keeps the model warm.

The `qwen` mode uses llama.cpp for chat and embeddings. Chat runs the multilingual RVN Heretic Q4_K_M model text-only at 262,144 tokens with full GPU placement, Q8_0 KV, Flash Attention, native RAM-backed prompt caching, and its embedded MTP head at draft depth two. vLLM, LMCache, and chat CPU weight offload are disabled. The Qwen3.8 template preserves reasoning and accepts `low`, `medium`, and `xhigh` reasoning effort (`high` aliases `xhigh`); clients with additional level names must map them first. This profile sets llama-swap's global idle timer to 30 minutes; other profiles retain the disabled default.

Qwen chat uses eight generation/batch threads, a 4,096 MiB RAM prompt cache,
and eight context checkpoints per slot. Its four slots retain the 262,144-token
shared unified context; this is not four independently allocated contexts.
These v0.6.2 command defaults preserve weights, MTP, GPU placement, KV precision,
and reasoning. Existing ignored profiles must be updated explicitly.

The matching external Compose deployment was tested with an eight-CPU quota,
32 GiB RAM limit and 40 GiB total RAM-plus-swap limit. Those per-container caps
are deployment settings, not new EasyLlama-wide defaults: the launcher still
uses the existing host-weighted `resources.roles` configuration for other modes
and for containers sharing a role. Hindsight's embedding batch concurrency of
four is likewise a downstream application setting, not a llama-swap admission cap.

Qwen embeddings use the official `Qwen/Qwen3-Embedding-0.6B-GGUF` FP16 file
on GPU, with four threads, four parallel slots of 32,768 tokens, 512-token
batch/microbatch sizes, and last-token pooling. A non-swapping search group keeps
embeddings and BGE reranking together with idle unloading disabled; chat swaps
out both when it needs VRAM. The stable
`qwen3-embeddings` ID now returns 1,024-dimensional vectors instead of 4,096.
Rebuild downstream vector indexes from their source documents; never mix vectors
from the old and new models, even if a client requests equal dimensions.

Existing ignored `config/config.qwen.yml` overrides are not overwritten by an
upgrade. Merge the embedding command and routing groups from the updated example
before restarting. The portable example downloads the named FP16 file; deployments
requiring an immutable artifact can use `--model` with a revision/checksum-verified
local copy. No launcher or inference-binary changes are required for this hotfix.

Qwen mode also exposes `qwen3-reranker` using BGE reranker v2 M3 Q8_0 at
`POST /v1/rerank`. This is the mode's endpoint name, not a Qwen-family model.
It uses full GPU placement, two native slots sharing a 16,384-token context,
and two CPU threads. The tested deployment caps its container at 2 CPUs,
8 GiB RAM/no extra swap and 4 GiB shared memory. An external Cohere-compatible
client can target this authenticated endpoint directly without moving embedding
traffic or rebuilding vector indexes. See [API details](API.md#endpoint-matrix).

## System requirements

- Linux with Bash `4.1+`
- Docker daemon running
- `docker buildx`
- NVIDIA drivers and working `nvidia-smi`
- NVIDIA container runtime in Docker
- 32 GiB NVIDIA GPU for the Qwen profile's full 262K context
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
| `./run.sh clean` | Remove the current mode stack, private network, and images; host caches are kept |
| `./run.sh clean --all-images` | Remove all mode images and the runtime container; host caches are kept |
| `./run.sh clean --wipe-cache` | Same, plus wipe every host cache (`root`, `jit`, `pkg`, `python`, `models`) |
| `./run.sh clean --wipe-cache models` | Same, plus wipe only the listed caches (comma-separated: `root`, `jit`, `pkg`, `python`, `models`) |
| `./run.sh clean --wipe-cache jit` | Wipe only the flashinfer JIT kernel cache; vLLM recompiles its kernels on next start |
| `./run.sh serve` | Run `llama-swap` inside container |
| `./run.sh server ...` | Run mode-specific upstream server directly |
| `./run.sh help` | Show CLI help |

`clean` no longer touches host caches by default: repeat cleans while debugging keep `models`,
`pkg`, `python`, and `jit` warm, so re-downloads (model weights, package archives) and vLLM's
flashinfer kernel compilation do not delay the next start or image build. Wipe them explicitly
with `--wipe-cache [list]` when a cache is corrupt or you want a cold-cache test.

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
| Warmup fails with `upstream command exited prematurely` | Model file missing from `cache/models` (fresh or wiped cache) | Run `./run.sh warmup <model>` again: `-hf` specs and hub-cache `--model` references are now re-downloaded host-side, pinned to their snapshot commit, before the container loads the model |
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
