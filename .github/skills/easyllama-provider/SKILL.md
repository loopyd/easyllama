---
name: easyllama-provider
description: 'Add or refactor an easyllama provider or mode. Use when integrating a new llama.cpp fork/backend, creating or extending launchers in easyllama/servers, wiring Dockerfile targets and config templates, updating README mode docs, rebuilding the mode image, warming models, and running the full public endpoint regression suite.'
argument-hint: 'Provider or mode name, upstream repo/ref, and whether it needs a dedicated server class'
---

# easyllama Provider Integration

Create or update a provider mode for this repository without reintroducing hardcoded runtime wiring.

## When to Use

- Add a new provider mode such as a new llama.cpp fork or backend.
- Refactor provider wiring so runtime, config, and CLI behavior come from shared metadata.
- Create a new launcher under `easyllama/servers/`.
- Add a new Docker build target and runtime binary.
- Validate a provider after config, runtime, or launcher changes.

## Inputs to Gather

1. Provider or mode name.
2. Upstream repo and ref, plus the runtime binary name or target.
3. Whether the provider can reuse an existing launcher shape or needs a dedicated server class.
4. Default model IDs, Hugging Face selectors, config knobs, and any provider-specific routes.
5. Whether local live validation needs an ignored active config in addition to the tracked example file.

## Procedure

1. Choose the launcher shape.

   - If the provider behaves like an existing plain launcher, prefer extending existing server metadata instead of adding new runtime branches.
   - If the provider needs custom model resolution, middleware, request rewriting, or special launch flags, create `easyllama/servers/<provider>.py`.

2. Put runtime mode metadata on the server layer.

   - Add or extend `@server_metadata(...)` on the relevant server class.
   - Declare the mode name, Docker target, and build-source repo/ref mappings there.
   - Register any new server class in `easyllama/servers/__init__.py`.
   - Do not add new mode-specific `if` branches or hardcoded mode lists in `runtime.py`, `config.py`, or `cli.py` when registry metadata can drive the behavior.

3. Wire project defaults and config files.

   - Update `pyproject.toml` defaults for new repo/ref settings when the provider introduces new upstream sources.
   - Add `config.<provider>.yml.example` and make sure the mode can resolve its active and example config paths.
   - If live validation needs pinned local settings, copy the example to an ignored `config.<provider>.yml`, but keep the tracked example as the release artifact.

4. Add Docker build and runtime targets.

   - Add a builder stage and runtime target in `Dockerfile`.
   - Copy the required binary and any companion assets into the runtime image.
   - Expose the runtime binary under `/app/bin/llama-server-<provider>` or the provider-specific equivalent.

5. Update documentation in one pass.

   - Update `README.md` for the mode overview, default model table, commands, config file list, and endpoint matrix.
   - Keep the README release-ready: remove migration notes, stale release notes, and implementation trivia that do not help users run the mode.
   - If behavior or config shape changes, update the matching `config.*.yml.example` file in the same change.

6. Run code-level validation.

   Use the [code validation script](./scripts/validate-code.sh). It runs the repo's narrow host-side checks for `run.sh`, `easyllama`, and the CLI surface from the repository root.

   If the repo lacks dedicated unit tests for the provider path, treat diagnostics plus the runtime endpoint suite as the required gate.

7. Rebuild and warm the provider mode.

   Use the [rebuild and warmup script](./scripts/rebuild-and-warmup.sh). Pass the mode name as the first argument and optional model IDs after it. If you omit model IDs, the script warms every model currently exposed by `/v1/models`.

   Rebuild whenever Python runtime code, Docker targets, launcher code, or config-loading behavior changed, because the runtime is baked into the image.

8. Run the public endpoint regression suite.

   Use the [public endpoint regression script](./scripts/test-public-endpoints.sh). Pass the mode name as the first argument. The script covers `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`, both `POST /v1/embeddings` aliases, `POST /v1/rerank`, and `GET /ui/`. It automatically checks `POST /v1/messages` for `lucebox`, and you can force that route for another provider with `--messages`.

   The script validates minimal response shape, not just status codes: advertised model IDs, assistant content for chat-style responses, non-empty embedding vectors with matching dimensions across aliases, and one rerank result per input document.

9. If validation fails, debug locally before expanding scope.

   - Confirm the active mode and config path actually being used.
   - Rebuild again if the failing behavior depends on Python runtime code.
   - Isolate the backend command on a scratch port if the failure is provider-specific.
   - Trim inherited launch flags toward the provider's minimal working surface and re-run the same failing endpoint until the regression is explained.

10. Completion criteria.

   The provider integration is done when all of the following are true:

   - the mode builds from its Dockerfile target
   - runtime, config, and CLI behavior come from shared metadata rather than new hardcoded mode branches
   - the tracked config example and README describe the shipped behavior
   - the rebuilt mode warms successfully
   - the full public endpoint suite passes after the rebuild

## Example Prompts

- `/easyllama-provider Add a new provider mode backed by <repo>@<ref> and wire it into Docker, config, and README.`
- `/easyllama-provider Refactor this provider integration so runtime metadata lives on the server classes and then rerun the endpoint suite.`
- `/easyllama-provider Audit the new provider mode, rebuild it, warm it, and run all public endpoints for regressions.`