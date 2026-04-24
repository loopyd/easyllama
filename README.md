# easyllama

Minimal, GPU-focused llama.cpp runner with Docker + CUDA 13.1 and LLAMACPP_-prefixed config overrides.

## Features

- CUDA 13.1 llama.cpp image build and server run flow
- Configurable llama.cpp source repo/ref (upstream or custom forks)
- Host GPU architecture auto-detection for CMAKE_CUDA_ARCHITECTURES
- JSON configuration with strict precedence and LLAMACPP_ env overrides
- Runtime cache-type compatibility checks before container start
- Secure public-repo defaults with tracked config template
- Pre-commit secret guard hook

## Requirements

- Docker with NVIDIA runtime enabled
- NVIDIA driver + working nvidia-smi on host
- jq
- Optional: gh (for GitHub repo creation/push workflow)

## Quick Start

1. Build image:

```bash
./run.sh build
```

2. Start server:

```bash
./run.sh start
```

3. Check status:

```bash
./run.sh status
curl -sS http://127.0.0.1:8080/health
```

## TurboQuant / turbo kernels

If you want `turbo2`, `turbo3`, or `turbo4` KV cache type kernels, use a turbo-capable llama.cpp fork in nested config:

```json
{
	"build": {
		"llama_cpp_repo": "https://github.com/TheTom/llama-cpp-turboquant.git",
		"llama_cpp_ref": "feature/turboquant-kv-cache"
	},
	"inference": {
		"cache_type_v": "turbo2",
		"kv_unified": true,
		"cache_idle_slots": true
	}
}
```

Then rebuild:

```bash
./run.sh build
./run.sh restart
```

The script validates that the built image actually supports the selected cache type and fails early with a rebuild hint if not.

## Configuration

Configuration precedence:

1. LLAMACPP_* environment variables
2. LLAMACPP_CONFIG_FILE path (if set)
3. config.json (local, gitignored)
4. config.json.example (tracked template)
5. Built-in defaults

Copy template to local config when you need local overrides:

```bash
cp config.json.example config.json
```

Config is node-based for clarity. Top-level nodes:

- `container`
- `network`
- `logging`
- `locale`
- `build`
- `model`
- `inference`
- `reasoning`

Example path keys used by the script include:

- `build.llama_cpp_repo`
- `inference.cache_type_v`
- `reasoning.enable`

Example env override:

```bash
LLAMACPP_HOST_PORT=8090 LLAMACPP_ENABLE_REASONING=off ./run.sh restart
```

Change llama.cpp fork/ref from env:

```bash
LLAMACPP_LLAMA_CPP_REPO=https://github.com/TheTom/llama-cpp-turboquant.git \
LLAMACPP_LLAMA_CPP_REF=feature/turboquant-kv-cache \
./run.sh build
```

## CLI Coverage

The script provides first-class config fields for core llama.cpp and commonly used TurboQuant options (including `cache_type_v`, `kv_unified`, and `cache_idle_slots`).

For any fork-specific or newly-added CLI args not modeled yet, use `inference.extra_server_args`.

## Extra Server Args

You can pass raw extra llama-server args via LLAMACPP_EXTRA_SERVER_ARGS or `inference.extra_server_args`.

Accepted forms:

- shell words string: --foo bar --baz
- JSON array: `["--foo","bar","--baz"]`

## Git Hooks (Secret Blocking)

Install repo hooks path:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

The pre-commit hook blocks commits when staged changes match common secret patterns.

## License

GPL-3.0-only. See LICENSE.
