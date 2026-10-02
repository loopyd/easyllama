# Chat template support

`./run.sh start` and `./run.sh restart` mount this directory at
`/chat_template` inside the container.

Override the host directory with `EASYLLAMA_CHAT_TEMPLATE_DIR`.

## Path mapping

Set the llama-swap model command's `--chat-template-file` argument to a mounted
path such as:

```yaml
--chat-template-file /chat_template/qwen3.8.jinja
```

The `qwen` llama.cpp chat route uses that template with `--reasoning auto` and
`--reasoning-preserve`. The template accepts `low`, `medium`, and `xhigh`
reasoning effort (`xhigh` is the default and `high` is accepted as an alias).
Map any additional client-side thinking levels before sending the request.

The `bonsai-chat` route uses the same Qwen3.8 template with
`--reasoning-preserve`, `--reasoning-format deepseek`, and `--reasoning-budget 1024`.
Thinking is enabled by default, returned separately from answer content, and
preserved in history. Send `chat_template_kwargs.enable_thinking: false` to skip
thinking for an individual chat request.

Bonsai's full 262,144-token window includes the formatted prompt, thinking, and
answer. Its single queued chat slot preserves that whole window for each request.
Reserve output tokens for both thinking and answer when reasoning is enabled.
