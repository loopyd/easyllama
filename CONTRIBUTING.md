# Contributing

Thanks for contributing to easyllama.

## Development Setup

1. Ensure requirements are installed (Docker, NVIDIA runtime, jq).
2. Use the config template:

```bash
cp config.json.example config.json
```

If testing turbo cache types, set turbo-capable fork settings in local config.json and rebuild before start.

3. Configure local hooks:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## Workflow

1. Create a branch for your changes.
2. Keep changes focused and documented.
3. Verify script syntax and startup flow before submitting:

```bash
bash -n run.sh
./run.sh build
./run.sh restart
curl -sS http://127.0.0.1:8080/health
```

## Security

- Never commit secrets.
- Keep real tokens only in local config.json or environment.
- Commit only config.json.example.

## Pull Requests

- Explain what changed and why.
- Include validation steps and outputs.
- Update docs when behavior/config changes.
