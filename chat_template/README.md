# Chat template support

`./run.sh start` and `./run.sh restart` mount this directory at
`/chat_template` inside the container.

Override the host directory with `EASYLLAMA_CHAT_TEMPLATE_DIR`.

## Path mapping

Set the llama-swap model command's `--chat-template-file` argument to a mounted
path such as:

```yaml
--chat-template-file /chat_template/qwen3.6.jinja
```

The MTP vLLM chat route uses the model's native template; the auxiliary
llama.cpp routes may use templates from this directory.
