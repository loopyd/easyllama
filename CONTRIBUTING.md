# Contributing

Keep changes focused and consistent with the shipped runtime behavior.

## Code of Conduct

This project expects respectful, constructive collaboration.

- Focus feedback on code, behavior, and reproducible issues.
- Avoid harassment, discrimination, personal attacks, and hostile language.
- Assume good intent, ask for clarification when needed, and keep review discussion technical.
- If a conversation stops being productive, pause and reset before continuing.

Thanks for helping keep the project usable and easy to maintain.

## Scope

- Update `README.md` when setup, mode selection, or top-level usage changes.
- Update `API.md` when model IDs, endpoint coverage, or query examples change.
- Update the matching `config.*.yml.example` file in the same change when config shape or defaults change.

## Useful Checks

Run these from the repository root, inside your virtual environment:

```bash
bash -n run.sh
.venv/bin/python -m ruff check easyllama
.venv/bin/python -m compileall easyllama
./run.sh help
```

For runtime-facing changes, rebuild the affected mode image, restart it, then verify at least:

- `GET /health`
- `GET /v1/models`

If API behavior changes, update `API.md` and re-run the relevant query examples.

