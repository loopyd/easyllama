
# Chat Template Support

This directory is mounted into the container at /chat_template when you run ./run.sh start or ./run.sh restart.

## Path Mapping

Set `inference.chat_template_file` in `config.json` using the repo-relative path: `chat_template/<model_alias>.jinja`

The launcher maps this folder to the container path at `/chat_template` and passes `--chat-template-file /chat_template/<model_alias>.jinja`.

## Example

```json
{
    ...
	"inference": {
        ...
		"chat_template_file": "chat_template/qwen3.6.jinja"
        ...
	}
    ...
}
```

## Credit

Qwen 3.5 and Qwen 3.6 chat template fixes:
https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates