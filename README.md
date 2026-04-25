# easy llama(cpp)

Minimal, GPU-focused llama.cpp runner for NVIDIA systems.

## Overview

| Feature | Summary |
| --- | --- |
| Runtime | Docker + NVIDIA runtime + llama-server |
| Build source | Any llama.cpp repo/ref (upstream or fork) |
| Config model | JSON + LLAMACPP_ env overrides |
| Startup safety | Cache-type compatibility check before start |
| Target use | Local Containerized GPU inference with reproducible shell workflow |

## Requirements

| Dependency | Required | Notes |
| --- | --- | --- |
| Docker | Yes | Docker daemon must be running |
| NVIDIA container runtime | Yes | Must be available in Docker runtimes |
| NVIDIA driver + nvidia-smi | Yes | Used for GPU architecture detection |
| jq | Yes | Used for JSON config parsing |
| gh | No | Optional for GitHub workflow convenience |

## Commands

| Command | Description |
| --- | --- |
| ./run.sh build | Build local CUDA image from Dockerfile |
| ./run.sh start | Start llama-server container |
| ./run.sh stop | Stop and remove container |
| ./run.sh restart | Stop then start |
| ./run.sh status | Show container state |
| ./run.sh logs | Follow container logs |
| ./run.sh clean | Remove container and local image |
| ./run.sh serve | Run llama-server from config (container entrypoint mode) |

## Quick Start

1. Build image.

```bash
./run.sh build
```

2. Start server.

```bash
./run.sh start
```

3. Verify status and health.

```bash
./run.sh status
curl -sS http://127.0.0.1:8080/health
```

## Configuration

### Precedence

| Priority | Source |
| --- | --- |
| 1 | LLAMACPP_ environment variables |
| 2 | LLAMACPP_CONFIG_FILE path (if set) |
| 3 | config.json |
| 4 | config.json.example |
| 5 | Built-in defaults in run.sh |

### Local Config Setup

```bash
cp config.json.example config.json
```

Config keys are optional. If a key is missing (or null for optional numeric keys), run.sh falls back to built-in defaults.

Host mode detail: `./run.sh start` mounts your active config as `/app/config.json` inside the container and launches `./run.sh serve` there, so URL-based downloads (for example `model.hf_mmproj`) occur at runtime against mounted host folders.

Container path keys:
- container.models_dir: host models cache mount source
- container.chat_template_dir: host chat template mount source
- container.mmproj_dir: host mmproj mount source (download target for model.hf_mmproj)

Example env override:

```bash
LLAMACPP_HOST_PORT=8090 LLAMACPP_ENABLE_REASONING=off ./run.sh restart
```

Example fork override:

```bash
LLAMACPP_LLAMA_CPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git \
LLAMACPP_LLAMA_CPP_REF=feature/turboquant-kv-cache \
./run.sh build
```

## TurboQuant Notes

Use a turbo-capable fork if you want turbo2, turbo3, or turbo4 KV V-cache types.

```json
{
    "build": {
        "llama_cpp_repo": "https://github.com/TheTom/llama-cpp-turboquant.git",
        "llama_cpp_ref": "feature/turboquant-kv-cache"
    },
    "inference": {
        "cache_type_v": "turbo4",
        "kv_unified": true,
        "cache_idle_slots": true
    }
}
```

Then rebuild and restart:

```bash
./run.sh build
./run.sh restart
```

The script checks whether your built image supports the selected cache_type_v and fails early with a clear hint if not.

## First-Class Inference Fields

### Sampling and Repetition

| Config key | llama-server flag | Description | Default |
| --- | --- | --- | --- |
| temp | --temp | Base randomness of token selection. | 0.80 |
| dynatemp_range | --dynatemp-range | Dynamic temperature range around temp. | 0.00 |
| dynatemp_exp | --dynatemp-exp | Curvature of dynamic temperature adjustment. | 1.00 |
| top_k | --top-k | Keep only top-k candidate tokens. | 40 |
| top_p | --top-p | Keep tokens within cumulative probability p. | 0.95 |
| min_p | --min-p | Drop low-probability tokens relative to top token. | 0.05 |
| top_n_sigma | --top-n-sigma | Entropy-aware filtering by log-probability distance. | -1.00 |
| typical_p | --typical-p | Locally typical sampling threshold. | 1.00 |
| xtc_probability | --xtc-probability | Chance that XTC token cutting is applied. | 0.00 |
| xtc_threshold | --xtc-threshold | Probability threshold used by XTC cutting. | 0.10 |
| repeat_last_n | --repeat-last-n | Context window used for repetition penalties. | 64 |
| repeat_penalty | --repeat-penalty | Penalize repeated token sequences. | 1.00 |
| presence_penalty | --presence-penalty | Penalize tokens that already appeared. | 0.00 |
| frequency_penalty | --frequency-penalty | Penalize tokens by repeat frequency. | 0.00 |
| dry_multiplier | --dry-multiplier | DRY anti-repetition penalty strength. | 0.00 |
| dry_base | --dry-base | DRY penalty exponential base value. | 1.75 |
| dry_allowed_length | --dry-allowed-length | Allowed repeated length before DRY penalties grow. | 2 |
| dry_penalty_last_n | --dry-penalty-last-n | Token window scanned by DRY repetition logic. | -1 |
| sampler_seq | --sampler-seq | Simplified sampler ordering string. | edskypmxt |
| samplers | --samplers | Explicit sampler chain in order. | unset (llama default) |
| backend_sampling | --backend-sampling | Run supported samplers on accelerator backend. | false |

Note: set only one of sampler_seq or samplers.

### Runtime and Behavior

| Config key | llama-server flag | Description | Default |
| --- | --- | --- | --- |
| n_predict | --n-predict | Max generated tokens per response. | -1 |
| ctx_size | --ctx-size | Context window size (0 uses model default). | 0 |
| threads | --threads | Number of CPU threads used during generation. | unset (llama default) |
| threads_batch | --threads-batch | Number of CPU threads used during batch/prompt processing. | unset (llama default) |
| batch_size | --batch-size | Logical max token batch size. | 2048 |
| ubatch_size | --ubatch-size | Physical micro-batch size for compute. | 512 |
| parallel | --parallel | Number of server slots/parallel sequences. | -1 |
| n_cpu_moe | --n-cpu-moe | Keep first N MoE layers on CPU. | unset |
| fit | --fit | Auto-adjust unset params to fit VRAM. | on |
| flash_attn | --flash-attn | Flash attention mode on/off/auto. | auto |
| cache_type_k | --cache-type-k | KV cache precision/quant type for K. | f16 |
| cache_type_v | --cache-type-v | KV cache precision/quant type for V. | f16 |
| kv_unified | --kv-unified / --no-kv-unified | Shared unified KV cache across sequences. | true |
| cache_idle_slots | --cache-idle-slots / --no-cache-idle-slots | Save and clear idle slots on new tasks. | true |
| web_ui | --webui / --no-webui | Enable or disable built-in web UI. | true |
| jinja | --jinja / --no-jinja | Enable or disable Jinja chat templating. | true |
| mmproj_file | --mmproj | Multi-modal projector GGUF file path. Supports URL download and local/container paths. | unset |
| model.hf_mmproj | --mmproj (resolved) | Hugging Face shorthand in format owner/repo/file.gguf, resolved to a blob URL and downloaded into mmproj_dir. | unset |
| no_warmup | --no-warmup | Skip warmup pass when starting the server. | false |
| no_mmap | --no-mmap | Disable memory-mapped model loading. | false |
| poll | --poll | Polling level used to wait for work. | 50 |
| chat_template_file | --chat-template-file | Path to a Jinja chat template file. **(1)** | unset |
| chat_template_kwargs | --chat-template-kwargs | Extra JSON args for template parser. | unset |

> **(1)** **Chat templates**: Setup, path mapping, and attribution are documented in [chat_template/README.md](chat_template/README.md).

For mmproj_file, you can provide:
- a Hugging Face URL (including /blob/... links), which run.sh downloads into container.mmproj_dir and passes as a mounted container path
- a local path under container.mmproj_dir (relative or absolute)
- a direct container path

For model.hf_mmproj, provide owner/repo/file.gguf (for example HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Balanced-f16.gguf). run.sh resolves it to https://huggingface.co/<owner>/<repo>/blob/main/<file>, downloads it into container.mmproj_dir, and passes --mmproj from the mounted /mmproj path.

If both model.hf_mmproj and inference.mmproj_file are set, model.hf_mmproj takes precedence.

## Extra Server Args

Use inference.extra_server_args (or LLAMACPP_EXTRA_SERVER_ARGS) for any new or fork-only flags not yet modeled.

Accepted formats:

| Format | Example |
| --- | --- |
| Shell words string | --foo bar --baz |
| JSON array | ["--foo", "bar", "--baz"] |

## Git Hook

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

The pre-commit hook blocks common secret patterns in staged content.

## License

GPL-3.0-only. See LICENSE.
