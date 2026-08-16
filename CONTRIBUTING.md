# Contributing

Keep changes focused and consistent with the shipped runtime behavior.

## Code of Conduct

This project expects respectful, constructive collaboration.

- Focus feedback on code, behavior, and reproducible issues.
- Avoid harassment, discrimination, personal attacks, and hostile language.
- Assume good intent, ask for clarification when needed, and keep review discussion technical.
- If a conversation stops being productive, pause and reset before continuing.

Thanks for helping keep the project usable and easy to maintain.

## Documentation scope

- Update `README.md` when setup, mode selection, or top-level usage changes.
- Update `API.md` when model IDs, endpoint coverage, or request examples change.
- Update the matching `config/config.<mode>.yml.example` in the same change when config shape or defaults change.
- Describe shipped behavior, not ignored local `config/config.<mode>.yml` overrides.
- Keep the five supported modes and their backends consistent across docs, config, and runtime metadata.

## Validation

Run host-side checks from the repository root inside the virtual environment:

```bash
.github/skills/easyllama-provider/scripts/validate-code.sh
.venv/bin/python test_backend_profiles.py
.venv/bin/python test_runtime_security.py
.venv/bin/python test_download_progress.py
.venv/bin/python test_warmup_progress.py
```

Validate every changed template before building:

```bash
.github/skills/easyllama-provider/scripts/validate-config-yaml.sh \
  config/config.<mode>.yml.example
```

For runtime-facing changes, rebuild and warm the affected mode, then run its public endpoint suite:

```bash
.github/skills/easyllama-provider/scripts/rebuild-and-warmup.sh <mode>
.github/skills/easyllama-provider/scripts/test-public-endpoints.sh <mode>
```

At minimum, verify `GET /health` and `GET /v1/models`. If API behavior changes, update `API.md` and exercise the affected request examples.

