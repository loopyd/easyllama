import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from easyllama.config import IMAGE, MODE, Config
from easyllama.helpers.builder import DockerBuilder, DockerfileCompiler
from easyllama.helpers.images import ModeImages
from easyllama.helpers.logger import ColorFormatter
from easyllama.helpers.orchestrator import ProxyConfigCompiler
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.docker]


def test_build_type_is_optional_and_host_defaults_local() -> None:
    from easyllama.cli import build_parser

    args = build_parser().parse_args(["--mode", "qwen", "build"])
    assert args.mode == "qwen" and args.type is None and args.host is None
    typed = build_parser().parse_args(
        ["--mode", "qwen", "--host", "0.0.0.0", "build", "--type", "vllm"]
    )
    assert typed.type == "vllm" and typed.host == "0.0.0.0"


def test_compiler_selects_isolated_stages() -> None:
    compiler = DockerfileCompiler(Path.cwd())
    assert compiler.target(MODE.QWEN, IMAGE.VLLM) == "runtime-vllm"
    assert compiler.stages(MODE.LLAMACPP, IMAGE.LLAMASWAP) == (
        "runtime-base",
        "runtime-python",
        "vllm-wrapper-build",
        "runtime-llamaswap",
    )
    assert compiler.stages(MODE.QWEN, IMAGE.LMCACHE) == (
        "runtime-base",
        "runtime-python",
        "runtime-lmcache",
    )
    assert "vllm-builder" not in compiler.stages(MODE.QWEN, IMAGE.LMCACHE)
    compiled = compiler.compile(MODE.LLAMACPP, IMAGE.LLAMACPP)
    content = compiled.path.read_text()
    assert compiled.target == "runtime-llamacpp"
    assert "FROM builder-base AS llamacpp-builder" in content
    assert "FROM runtime-python AS runtime-llamacpp" in content
    assert "vllm-builder" not in content


def test_mode_image_dependencies() -> None:
    assert tuple(item.image for item in ModeImages.for_mode(MODE.LLAMACPP).dependencies) == (
        IMAGE.LLAMASWAP,
        IMAGE.LLAMACPP,
    )
    assert tuple(item.image for item in ModeImages.for_mode(MODE.QWEN).dependencies) == (
        IMAGE.LLAMASWAP,
        IMAGE.VLLM,
        IMAGE.LMCACHE,
        IMAGE.LLAMACPP,
    )
    assert ModeImages.for_mode(MODE.QWEN).backend is IMAGE.VLLM
    assert (
        ModeImages.for_mode(MODE.QWEN).dependency(IMAGE.VLLM).name(MODE.QWEN)
        == "easyllama-qwen-vllm"
    )


def test_runtime_base_has_no_backend() -> None:
    runtime = Path("docker/runtime-base.Dockerfile").read_text()
    assert "vllm-builder" not in runtime
    assert "llama-server" not in runtime
    assert "vllm-wrapper-build" not in runtime
    assert "COPY easyllama/" not in runtime
    assert "pip install --no-deps /app" not in runtime
    python_runtime = Path("docker/runtime-python.Dockerfile").read_text()
    assert "COPY easyllama/ /app/easyllama/" in python_runtime


def test_proxy_config_has_explicit_container_contracts() -> None:
    plan = ProxyConfigCompiler(MODE.QWEN).compile(Path("config/config.qwen.yml.example"), "secret")
    chat = plan.config["models"]["qwen3-chat"]
    assert chat["cmd"].endswith("http://easyllama-qwen-llamacpp-qwen3-chat:9002/run")
    assert chat["cmdStop"].endswith("http://easyllama-qwen-llamacpp-qwen3-chat:9002/sleep")
    assert chat["proxy"] == "http://easyllama-qwen-llamacpp-qwen3-chat:9000"
    assert "env" not in chat and "type" not in chat
    assert plan.config["routing"]["router"]["settings"]["groups"]["gpu"] == {
        "swap": True,
        "exclusive": True,
        "members": ["qwen3-chat", "qwen3-embeddings"],
    }
    assert plan.config["apiKeys"] == ["secret"]
    assert plan.config["healthCheckTimeout"] == 1800
    assert plan.config["logLevel"] == "info"
    assert plan.config["sendLoadingState"] is False
    assert plan.config["globalTTL"] == 1800
    assert "ttl" not in chat
    assert "macros" not in plan.config
    chat, embeddings = plan.containers
    assert chat.health_path == "/v1/models" and chat.stop_signal == "SIGTERM"
    assert chat.lifecycle_port == 9002
    chat_command = " ".join(chat.command)
    assert "/app/bin/llama-server-qwen" in chat_command
    assert "RVN-Q4_K_M-multilingual-mtp.gguf" in chat_command
    assert "--ctx-size 262144" in chat_command
    assert "--cache-type-k q8_0 --cache-type-v q8_0" in chat_command
    assert "--spec-type draft-mtp --spec-draft-n-max 2" in chat_command
    assert embeddings.health_path == "/v1/models"
    assert embeddings.lifecycle_port == 9003
    assert "--host 0.0.0.0" in " ".join(embeddings.command)
    assert chat.environment == ("HF_TOKEN=${HF_TOKEN}",)
    assert embeddings.environment == ("HF_TOKEN=${HF_TOKEN}",)


def test_proxy_compiler_expands_command_macros() -> None:
    plan = ProxyConfigCompiler(MODE.LLAMACPP).compile(Path("config/config.llamacpp.yml.example"))
    command = " ".join(plan.containers[0].command)
    assert "/opt/venv/bin/easyllama server llamacpp" in command
    assert "\n" not in plan.containers[0].command[-1]
    assert "${server_bin}" not in command
    assert "${env." not in command
    assert "${EASYLLAMA_MMPROJ_ARG}" in command
    assert "${PORT}" not in command
    assert plan.config["globalTTL"] == 0


def test_hf_progress_supports_xet_postfix() -> None:
    from easyllama.helpers.hf import HfProgress

    progress = HfProgress(total=10)
    assert progress.set_postfix_str("8 files") is None
    assert progress.refresh() is None


def test_dependency_containers_override_runtime_entrypoint(monkeypatch: Any) -> None:
    from easyllama.helpers.docker import DockerRuntime

    runtime = object.__new__(DockerRuntime)
    calls: list[tuple[str, dict[str, Any]]] = []
    runtime.settings = cast(
        Any,
        SimpleNamespace(
            runtime=SimpleNamespace(mode=MODE.LLAMACPP, pids_limit=1024),
            resources=Config.load().resources,
            dirs=SimpleNamespace(root=Path.cwd()),
            locale=SimpleNamespace(timezone="UTC", lang="C.UTF-8", lc_all="C.UTF-8"),
            docker=SimpleNamespace(image_name="easyllama", image_tag="cuda13"),
            mmproj_arg=lambda _auth: "",
        ),
    )
    runtime.client = cast(
        Any,
        SimpleNamespace(
            info=lambda: {"NCPU": 32, "MemTotal": 128 * 1024**3},
            containers=SimpleNamespace(run=lambda image, **kwargs: calls.append((image, kwargs))),
        ),
    )
    contract = (
        ProxyConfigCompiler(MODE.LLAMACPP)
        .compile(Path("config/config.llamacpp.yml.example"))
        .containers[0]
    )
    runtime.build_image = cast(Any, lambda _image: None)
    monkeypatch.setattr("easyllama.helpers.docker.DockerBuilder.exists", lambda _builder: True)
    runtime._run_dependency(
        contract,
        SimpleNamespace(hf_token=None),
        SimpleNamespace(name="easyllama-llamacpp"),
        {},
    )
    assert calls[0][1]["entrypoint"] == []
    assert calls[0][1]["command"][:5] == [
        "/opt/venv/bin/easyllama",
        "lifecycle",
        "--port",
        "9002",
        "--",
    ]
    assert "ipc_mode" not in calls[0][1]
    assert calls[0][1]["healthcheck"]["test"][-1].endswith(":9002/health")


def test_wait_for_dependency_requires_healthy(monkeypatch: Any) -> None:
    from easyllama.helpers.docker import DockerRuntime

    class Container:
        name = "dependency"

        def __init__(self) -> None:
            self.attrs = {"State": {"Status": "running", "Health": {"Status": "starting"}}}

        def reload(self) -> None:
            self.attrs["State"]["Health"]["Status"] = "healthy"

    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(
        Any, SimpleNamespace(warmup=SimpleNamespace(timeout=1, poll_interval=0))
    )
    monkeypatch.setattr("easyllama.helpers.docker.time.sleep", lambda _: None)
    runtime._wait_for_dependency(Container())


def test_aggregate_tail_logs_are_tagged(caplog: Any, monkeypatch: Any) -> None:
    from easyllama.helpers.docker import DockerRuntime

    class Container:
        def __init__(self, name: str, text: str) -> None:
            self.name = name
            self.text = text

        def logs(self, **_kwargs: Any) -> bytes:
            return self.text.encode()

    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(Any, SimpleNamespace(mode=MODE.LLAMACPP))
    runtime.ensure_daemon = cast(Any, lambda: None)
    runtime.mode_containers = cast(
        Any,
        lambda: [
            Container(
                "easyllama-llamacpp-z",
                "2026-08-27T20:01:02.000000002Z z line\n",
            ),
            Container(
                "easyllama-llamacpp-a",
                "2026-08-27T20:01:02.000000001Z a line\n",
            ),
        ],
    )
    caplog.set_level(logging.INFO, logger="easyllama.helpers.docker")
    monkeypatch.setattr(caplog.handler, "formatter", ColorFormatter(use_color=False))
    assert runtime.print_logs(tail=10) == 0
    assert caplog.text.splitlines() == [
        "INFO [easyllama-llamacpp-a] a line",
        "INFO [easyllama-llamacpp-z] z line",
    ]


def test_stack_stop_removes_dependencies_then_networks() -> None:
    from easyllama.helpers.docker import DockerRuntime

    events: list[str] = []
    runtime = object.__new__(DockerRuntime)
    runtime.ensure_daemon = cast(Any, lambda: events.append("daemon"))
    runtime.remove_container = cast(Any, lambda: events.append("containers"))
    runtime.remove_networks = cast(Any, lambda: events.append("networks"))
    runtime._remove_effective_configs = cast(Any, lambda: events.append("configs"))
    assert runtime.stop_container() == 0
    assert events == ["daemon", "containers", "networks", "configs"]


def test_config_validates_nested_settings(monkeypatch: Any) -> None:
    from easyllama.config import Config

    monkeypatch.setenv("EASYLLAMA_ROOT", str(Path.cwd()))
    monkeypatch.setenv("EASYLLAMA_LMCACHE_CHUNK_SIZE", "2048")
    monkeypatch.setenv("EASYLLAMA_LMCACHE_L1_SIZE_GB", "48")
    monkeypatch.setenv("EASYLLAMA_HOST_PORT", "9090")
    settings = Config.load(host_override="0.0.0.0")
    assert str(settings.runtime.host) == "0.0.0.0"
    assert settings.lmcache.chunk_size == 2048
    assert settings.lmcache.l1_size_gb == 48
    assert settings.runtime.host_port == 9090
    assert settings.dirs.models == Path.cwd() / "cache/models"
    assert settings.modes[MODE.QWEN].repo.url.endswith("ggml-org/llama.cpp.git")
    assert Config.load(host_override="Example.COM.").runtime.host == "example.com"
    assert Config.load(host_override="2001:db8::1").listen_url() == "http://[2001:db8::1]:9090"
    with pytest.raises(SystemExit, match="invalid configuration"):
        Config.load(host_override="bad_host")


def test_listen_url_uses_configured_hostname(monkeypatch: Any) -> None:
    from easyllama.config import Config

    monkeypatch.setenv("EASYLLAMA_ROOT", str(Path.cwd()))
    settings = Config.load(host_override="api.example.test")
    assert settings.listen_url() == "http://api.example.test:8080"


def test_config_field_metadata_resolves_matching_env(monkeypatch: Any) -> None:
    from easyllama.config import Config

    monkeypatch.setenv("EASYLLAMA_ROOT", str(Path.cwd()))
    monkeypatch.setenv("EASYLLAMA_MODELS_DIR", "/tmp/easyllama-models")
    monkeypatch.setenv("EASYLLAMA_LLAMA_CPP_REF", "test-ref")
    settings = Config.load()
    assert settings.dirs.models == Path("/tmp/easyllama-models")
    assert settings.modes[MODE.QWEN].repo.ref == "test-ref"


def test_config_example_matches_model_and_documentation() -> None:
    import json

    from easyllama.config import Config

    example = json.loads(Path("config.json.example").read_text())
    settings = Config.model_validate(example, context={"root": Path.cwd()})
    assert set(example) == set(Config.model_fields)
    assert Config.model_validate_json(settings.model_dump_json()) == settings
    readme = Path("README.md").read_text()
    for key in example:
        assert key in readme
    for stale in ("`repos`", "`hardware`", "`llama_swap`"):
        assert stale not in readme


def test_project_skill_scripts_are_current() -> None:
    import subprocess

    scripts = sorted(Path(".github/skills").glob("*/scripts/*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", script], check=True)
    instruction_text = "\n".join(
        path.read_text() for path in Path(".github/skills").glob("*/SKILL.md")
    )
    assert "llama_swap.modes" not in instruction_text
    assert "settings.llama_swap" not in instruction_text


def test_config_json_round_trip_and_precedence(tmp_path: Path, monkeypatch: Any) -> None:
    from easyllama.config import Config, CredentialsConfig

    config_file = tmp_path / "custom.json"
    config_file.write_text(
        '{"runtime":{"mode":"qwen","host_port":9000},"credentials":{"api_key":"from-file"}}'
    )
    monkeypatch.setenv("EASYLLAMA_HOST_PORT", "9001")
    settings = Config.load(
        config_file=config_file,
        mode_override=MODE.LLAMACPP,
        host_override="0.0.0.0",
    )
    assert settings.runtime.mode is MODE.LLAMACPP
    assert settings.runtime.host_port == 9001
    assert str(settings.runtime.host) == "0.0.0.0"
    assert settings.credentials == CredentialsConfig(hf_token=None, api_key="from-file")
    assert Config.model_validate_json(settings.model_dump_json()) == settings
    with pytest.raises(SystemExit, match="configuration file not found"):
        Config.load(config_file=tmp_path / "missing.json")


def test_server_registry_has_mode_specific_launchers() -> None:
    from easyllama.servers import defs, make

    names = {definition.name for definition in defs()}
    assert "basic" not in names
    assert {"llamacpp", "turboquant", "qwen", "lucebox", "spiritbuun"} <= names
    assert make("llamacpp").parser().parse_args([]).bin == Path("/app/bin/llama-server-llamacpp")
    assert make("turboquant").parser().parse_args([]).bin == Path(
        "/app/bin/llama-server-turboquant"
    )
    assert make("qwen").parser().parse_args([]).bin == Path("/app/bin/llama-server-qwen")


def test_server_build_sources_use_mode_specific_nested_repos(tmp_path: Path) -> None:
    from easyllama.config import Config
    from easyllama.servers import mode_def

    config_file = tmp_path / "servers.json"
    config_file.write_text(
        '{"modes":{"llamacpp":{"repo":{"url":"https://example/llamacpp.git","ref":"custom"}}}}'
    )
    settings = Config.load(config_file=config_file)
    source = mode_def("llamacpp").build_sources[0]
    assert source.values(settings) == ("https://example/llamacpp.git", "custom")
    assert source.build_args(settings) == {
        "LLAMA_CPP_REPO": "https://example/llamacpp.git",
        "LLAMA_CPP_REF": "custom",
    }
    assert (
        mode_def("turboquant")
        .build_sources[0]
        .values(settings)[0]
        .endswith("llama-cpp-turboquant.git")
    )


def test_all_server_modes_build_metadata_from_nested_config() -> None:
    from easyllama.config import Config
    from easyllama.servers import mode_defs

    settings = Config.load()
    for mode in mode_defs():
        assert mode.docker_target
        assert mode.build_args(settings)
        assert f"mode={mode.mode}" in mode.build_summary(
            settings,
            image_name=settings.image_for_mode(mode.mode),
            target=mode.docker_target,
        )


def test_lmcache_contract_uses_nested_config(monkeypatch: Any) -> None:
    from easyllama.config import Config

    monkeypatch.setenv("EASYLLAMA_ROOT", str(Path.cwd()))
    monkeypatch.setenv("EASYLLAMA_LMCACHE_CHUNK_SIZE", "2048")
    monkeypatch.setenv("EASYLLAMA_LMCACHE_L1_SIZE_GB", "48")
    settings = Config.load(mode_override=MODE.QWEN)
    plan = ProxyConfigCompiler(MODE.QWEN, settings=settings).compile(
        Path("config/config.qwen.yml.example")
    )
    assert all(contract.image is not IMAGE.LMCACHE for contract in plan.containers)


def test_network_name_uses_mode() -> None:
    from easyllama.helpers.docker import DockerRuntime

    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(Any, SimpleNamespace(runtime=SimpleNamespace(mode=MODE.QWEN)))
    assert runtime.network_name == "easyllama-qwen"


def test_remove_networks_only_removes_selected_mode() -> None:
    from easyllama.helpers.docker import DockerRuntime

    removed: list[str] = []
    networks = [SimpleNamespace(name="easyllama-qwen", remove=lambda: removed.append("qwen"))]
    runtime = object.__new__(DockerRuntime)
    runtime.settings = cast(Any, SimpleNamespace(runtime=SimpleNamespace(mode=MODE.QWEN)))
    runtime.client = cast(
        Any,
        SimpleNamespace(networks=SimpleNamespace(list=lambda **_kwargs: networks)),
    )
    runtime.remove_networks()
    assert removed == ["qwen"]


def test_builder_command_uses_enum_references(monkeypatch: Any) -> None:
    settings = cast(
        Any,
        SimpleNamespace(
            dirs=SimpleNamespace(root=Path.cwd()),
            runtime=SimpleNamespace(mode=MODE.LLAMACPP),
            docker=SimpleNamespace(image_name="easyllama", image_tag="cuda13"),
        ),
    )
    builder = DockerBuilder(settings, SimpleNamespace(), MODE.LLAMACPP, IMAGE.LLAMACPP)
    monkeypatch.setattr("easyllama.helpers.builder.shutil.which", lambda _: "/usr/bin/docker")
    command = builder.command({"BUILD_MODE": MODE.LLAMACPP})
    assert command[0:3] == ["/usr/bin/docker", "buildx", "build"]
    assert command[command.index("--target") + 1] == "runtime-llamacpp"
    assert command[command.index("--tag") + 1] == "easyllama:cuda13-llamacpp"
    assert command[command.index("--file") + 1].endswith("llamacpp.Dockerfile")
    assert "easyllama.mode=llamacpp" in command
    assert "easyllama.type=llamacpp" in command
    qwen_settings = cast(
        Any,
        SimpleNamespace(
            dirs=SimpleNamespace(root=Path.cwd()),
            runtime=SimpleNamespace(mode=MODE.QWEN),
            docker=SimpleNamespace(image_name="easyllama", image_tag="cuda13"),
        ),
    )
    qwen = DockerBuilder(qwen_settings, SimpleNamespace(), MODE.QWEN, IMAGE.VLLM)
    assert qwen.name == "easyllama:cuda13-qwen-vllm"
    assert DockerBuilder(qwen_settings, SimpleNamespace(), MODE.QWEN, IMAGE.LLAMASWAP).name == (
        "easyllama:cuda13-llamaswap"
    )
    assert DockerBuilder(qwen_settings, SimpleNamespace(), MODE.QWEN, IMAGE.LMCACHE).name == (
        "easyllama:cuda13-lmcache"
    )
