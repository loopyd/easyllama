---
name: easyllama-tune
description: 'Tune easyllama chat-model fit for a chosen mode: gpu layers, KV cache quantization, ctx-size, warmup 502/250 failures, and full-context ceilings.'
argument-hint: 'mode=mtp, quants=q5_1, ctx=262144, max_gpu_layers=60'
---

# Fit Tuning

Tune the first chat-model alias in a chosen easyllama mode until the active config has a verified fit boundary.

## Use When

- Find the highest stable `--gpu-layers` for the first chat alias in a mode config.
- Increase context size to the maximum the model supports.
- Lower KV cache precision from `q8_0` without jumping straight to the most aggressive mode.
- Re-check the fit boundary after changing the context size.
- Explain or reproduce warmup failures that surface as upstream `502` errors or upstream exit `250` during model startup.

## Gather

1. Target mode plus current known-good settings: GPU layers, ctx size, cache type, and fit mode.
2. Desired direction: more layers, lower KV cache precision, larger context, or some combination.
3. Bounds for layer search: the last good value and the requested high or known failing value.
4. Whether the tracked example file should be synced after a passing result.
5. Whether quality regression needs a deterministic sample capture in addition to warmup validation.

## Rules

- Treat `config/config.<mode>.yml` as the active scratchpad; do not sync `config/config.<mode>.yml.example` until a setting passes.
- Before a tuning probe that starts or restarts a mode, stop any running easyllama containers so stale GPU consumers do not fake a lower fit ceiling.
- Use the real `./run.sh --mode <mode> restart && ./run.sh --mode <mode> warmup <chat-alias>` path as the acceptance check.
- When lowering KV cache precision, prefer `q5_1` before `q5_0`.
- Do not assume `q6_*` KV modes exist; check the runtime surface first.
- After a config edit, upstream `502` or exit `250` usually means a fit boundary; inspect logs only when that is unclear.

## Procedure

1. Check supported KV cache types.
   - Run [list-supported-cache-types.sh](./scripts/list-supported-cache-types.sh) with `--mode <mode>`.
   - If the desired mode is unsupported, pick the least aggressive supported fallback.
2. Record the anchor.
   - Use [set-chat-tuning.sh](./scripts/set-chat-tuning.sh) with `--mode <mode> --show`, or read the active config.
   - Keep one known-good combination before searching upward.
3. If only KV cache changes, validate that cache type at the current good layer count first.
   - Use [set-chat-tuning.sh](./scripts/set-chat-tuning.sh) to set `--cache-type`.
   - Run [probe-chat.sh](./scripts/probe-chat.sh).
   - Search higher layers only after that passes.
4. Search the GPU-layer ceiling with a bracket.
   - If good and bad bounds are known, run [search-max-gpu-layers.sh](./scripts/search-max-gpu-layers.sh) with `--mode <mode>`.
   - If the user says "try N and dial back", treat the current pass as `--good` and the requested value as the first probe; once it fails, use it as `--bad`.
5. After every substantive change, run the real completion gate.
   - Stop running easyllama containers before restart; use the `common.sh` helper through the tuning scripts, not manually.
   - Validate YAML.
   - Restart the mode.
   - Warm only the discovered chat alias with `./run.sh --mode <mode> warmup <chat-alias>`.
   - If warmup fails with upstream `502` or `exit status 250`, treat it as a fit or startup OOM signal unless logs show another root cause.
   - Otherwise inspect logs to distinguish fit failure from transient or config parsing issues.
   - If warmup succeeds, print live args to confirm the intended settings; use [probe-chat.sh](./scripts/probe-chat.sh) to combine validation, restart, warmup, and arg checking.
   - If searching for a boundary, update good or bad and repeat until the ceiling is found or the user stops.
   - If tuning to a specific setting, stop after the first pass.
   - If that setting fails, roll back to the last known good setting and stop.
6. If cache precision changed and quality matters, capture deterministic before/after samples.
   - Use [snapshot-chat-sample.sh](./scripts/snapshot-chat-sample.sh) with the same prompt and fixed generation parameters.
   - Prefer asset-pack prompts for repeatable comparisons.
   - Summarize drift with [compare-chat-samples.sh](./scripts/compare-chat-samples.sh).
   - Compare saved outputs before recommending more aggressive cache modes.
7. After a pass, sync the example file if the repo default should change.
   - Re-run validation on the example file after syncing.

## Done When

- `config/config.<mode>.yml` contains the chosen ctx size, GPU layer count, and KV cache type.
- `./run.sh --mode <mode> restart && ./run.sh --mode <mode> warmup <chat-alias>` succeeds.
- Live `llama-server-<mode>` args match the intended ctx size, layer count, and cache types.
- If the tuned setting is meant to become the repo default, `config/config.<mode>.yml.example` is synced and validates.
- If KV cache precision changed, the user has either accepted the heuristic choice (`q5_1` before `q5_0`) or compared deterministic sample outputs.

## Scripts

- [list-supported-cache-types.sh](./scripts/list-supported-cache-types.sh): show KV cache types accepted by the current server binary.
- [set-chat-tuning.sh](./scripts/set-chat-tuning.sh): update qwen3-chat ctx size, gpu layers, fit mode, and KV cache types in a mode config.
- [probe-chat.sh](./scripts/probe-chat.sh): validate config, stop running easyllama containers, restart the selected mode, warm the first chat alias under `models:`, and print live args on success.
- [search-max-gpu-layers.sh](./scripts/search-max-gpu-layers.sh): binary-search the highest passing layer count between known good and known failing bounds for a selected mode.
- [snapshot-chat-sample.sh](./scripts/snapshot-chat-sample.sh): save a deterministic chat completion response for before-and-after cache-quant comparisons.
- [compare-chat-samples.sh](./scripts/compare-chat-samples.sh): compare two saved snapshot JSON files and summarize content and format drift.

## Assets

- [cache-quant-prompt-pack.md](./assets/cache-quant-prompt-pack.md): prompt-pack guide and examples.
- [cache-quant-long-context-recall.txt](./assets/cache-quant-long-context-recall.txt): synthetic long-context retrieval prompt.
- [cache-quant-structured-output.txt](./assets/cache-quant-structured-output.txt): strict JSON-format fidelity prompt.
- [cache-quant-constraint-following.txt](./assets/cache-quant-constraint-following.txt): tight instruction-following prompt.
- [cache-quant-style-consistency.txt](./assets/cache-quant-style-consistency.txt): style-consistency prompt for side-by-side diffs.

## Example Prompts

- `/easyllama-tune In mode <mode>, raise the chat model to full 262144 context and find the highest passing gpu-layers ceiling.`
- `/easyllama-tune In mode <mode>, lower KV cache from q8_0 without an aggressive quality tradeoff, then see whether another GPU layer fits.`
- `/easyllama-tune In mode <mode>, try 61 layers with q5_1 KV cache, then tune down to the highest stable value and sync the example file.`