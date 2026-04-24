# easyllama

Minimal, GPU-focused llama.cpp runner with Docker + CUDA 13.1 and LLAMACPP_-prefixed config overrides.

## Features

- CUDA 13.1 llama.cpp image build and server run flow
- Host GPU architecture auto-detection for CMAKE_CUDA_ARCHITECTURES
- JSON configuration with strict precedence and LLAMACPP_ env overrides
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

Example env override:

```bash
LLAMACPP_HOST_PORT=8090 LLAMACPP_ENABLE_REASONING=off ./run.sh restart
```

## Extra Server Args

You can pass extra llama-server args via LLAMACPP_EXTRA_SERVER_ARGS or config value extra_server_args.

Accepted forms:

- shell words string: --foo bar --baz
- JSON array: ["--foo","bar","--baz"]

## Git Hooks (Secret Blocking)

Install repo hooks path:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

The pre-commit hook blocks commits when staged changes match common secret patterns.

## License

GPL-3.0-only. See LICENSE.
