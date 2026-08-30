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
