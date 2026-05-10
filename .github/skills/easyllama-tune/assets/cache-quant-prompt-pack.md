# Cache-Quant Prompt Pack

Use these prompts to compare before-and-after outputs when changing KV cache precision.

## Goals

- Hold the prompt constant while changing cache type or gpu-layer count.
- Stress different failure modes: retrieval, exact formatting, instruction following, and style consistency.
- Keep prompts reusable with the deterministic sampling helper.

## Suggested Workflow

1. Capture a baseline with the current known-good config.
2. Change one variable at a time: KV cache type first, then gpu-layers.
3. Re-run the same prompt with the same seed.
4. Run `./.github/skills/easyllama-tune/scripts/compare-chat-samples.sh before.json after.json`.
5. Review the saved JSON outputs and the extracted assistant text when the summary flags drift.

## Example Commands

```bash
./.github/skills/easyllama-tune/scripts/snapshot-chat-sample.sh \
  /tmp/baseline-long-context.json \
  --prompt-file ./.github/skills/easyllama-tune/assets/cache-quant-long-context-recall.txt

./.github/skills/easyllama-tune/scripts/snapshot-chat-sample.sh \
  /tmp/variant-structured.json \
  --prompt-file ./.github/skills/easyllama-tune/assets/cache-quant-structured-output.txt

./.github/skills/easyllama-tune/scripts/compare-chat-samples.sh \
  /tmp/baseline-long-context.json \
  /tmp/variant-structured.json
```

## Prompt Files

- `cache-quant-long-context-recall.txt`: checks retrieval across distractors and ordered facts.
- `cache-quant-structured-output.txt`: checks exact JSON field fidelity and escaping.
- `cache-quant-constraint-following.txt`: checks instruction-following under multiple explicit constraints.
- `cache-quant-style-consistency.txt`: checks whether tone and wording drift between runs.