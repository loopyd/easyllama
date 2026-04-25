
# Chat Template Support

By default, this directory is mounted into the container at /chat_template when you run ./run.sh start or ./run.sh restart.

The host source path is configured by container.chat_template_dir (or LLAMACPP_CHAT_TEMPLATE_DIR).

## Path Mapping

Set `inference.chat_template_file` in `config.json` using either:

- `<model_alias>.jinja` (bare filename)
- `chat_template/<model_alias>.jinja` (repo-relative path)

The launcher resolves both forms to the mounted container path and passes `--chat-template-file /chat_template/<model_alias>.jinja`.

## Example

```json
{
    ...
	"inference": {
        ...
        "chat_template_file": "qwen3.6.jinja"
        ...
	}
    ...
}
```