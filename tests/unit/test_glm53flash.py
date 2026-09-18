"""Unit tests for the glm5.3-flash FreeToken mode."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from easyllama.config import CPU_WEIGHT, IMAGE, MODE, RAM_WEIGHT, SWAP_WEIGHT, Config
from easyllama.helpers.builder import DockerBuilder, DockerfileCompiler
from easyllama.helpers.images import ModeImages
from easyllama.helpers.orchestrator import ProxyConfigCompiler
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.docker]


def test_mode_is_registered_and_normalizes() -> None:
    from easyllama.helpers.common import normalize_mode
    from easyllama.servers import mode_def, mode_names

    assert MODE.GLM53FLASH.value == "glm5.3-flash"
    assert normalize_mode("GLM5.3-FLASH") == "glm5.3-flash"
    assert "glm5.3-flash" in mode_names()
    metadata = mode_def("glm5.3-flash")
    assert metadata.docker_target == "runtime-freetoken"
    assert metadata.backend == "freetoken"
    assert metadata.build_args(Config.load()) == {
        "FREETOKEN_REPO": "https://github.com/FlashML-org/FreeToken.git",
        "FREETOKEN_REF": "main",
    }


def test_compiler_uses_freetoken_stages() -> None:
    compiler = DockerfileCompiler(Path.cwd())
    assert compiler.target(MODE.GLM53FLASH, IMAGE.FREETOKEN) == "runtime-freetoken"
    assert compiler.stages(MODE.GLM53FLASH, IMAGE.FREETOKEN) == (
        "runtime-base",
        "runtime-python",
        "freetoken-builder",
        "runtime-freetoken",
    )
    assert compiler.stages(MODE.GLM53FLASH, IMAGE.LLAMASWAP) == (
        "runtime-base",
        "runtime-python",
        "vllm-wrapper-build",
        "runtime-llamaswap",
    )
    compiled = compiler.compile(MODE.GLM53FLASH, IMAGE.FREETOKEN)
    content = compiled.path.read_text()
    assert compiled.target == "runtime-freetoken"
    assert "FROM runtime-python AS freetoken-builder" in content
    assert "FROM runtime-python AS runtime-freetoken" in content
    assert "COPY --from=freetoken-builder /opt/ft-venv /opt/ft-venv" in content
    assert "llamacpp-builder" not in content
    with pytest.raises(SystemExit, match="does not use a llamacpp image"):
        compiler.stages(MODE.GLM53FLASH, IMAGE.LLAMACPP)


def test_mode_image_dependencies() -> None:
    assert tuple(item.image for item in ModeImages.for_mode(MODE.GLM53FLASH).dependencies) == (
        IMAGE.LLAMASWAP,
        IMAGE.FREETOKEN,
    )
    assert ModeImages.for_mode(MODE.GLM53FLASH).dependency(IMAGE.FREETOKEN).gpu


def test_proxy_plan_reserves_freetoken_store_port() -> None:
    plan = ProxyConfigCompiler(MODE.GLM53FLASH).compile(
        Path("config/config.glm5.3-flash.yml.example")
    )
    chat = plan.containers[0]
    assert chat.port == 9100
    assert chat.lifecycle_port == 9102
    api_ports = [c.port for c in plan.containers if c.port is not None]
    allocated = {p for c in plan.containers for p in (c.port, c.lifecycle_port) if p is not None}
    assert len(allocated) == 2 * len(plan.containers)
    # FreeToken binds server_port + 1 for its torch.distributed store; the
    # allocator must never hand that slot to any API or lifecycle listener.
    assert all(port + 1 not in allocated for port in api_ports)


def test_freetoken_image_is_shared_across_modes() -> None:
    settings = cast(
        Any,
        SimpleNamespace(
            docker=SimpleNamespace(image_name="easyllama", image_tag="cuda13"),
            runtime=SimpleNamespace(mode=MODE.GLM53FLASH),
            dirs=SimpleNamespace(root=Path.cwd()),
        ),
    )
    builder = DockerBuilder(settings, None, MODE.GLM53FLASH, IMAGE.FREETOKEN)
    assert builder.name == "easyllama:cuda13-freetoken"


def test_server_builds_ft_serve_command(tmp_path: Path) -> None:
    from easyllama.servers.glm53flash import Glm53FlashServer

    server = Glm53FlashServer()
    binary = tmp_path / "ft"
    binary.write_text("#!/bin/sh\n")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model-00001-of-00010.safetensors").write_bytes(b"\x00")
    args, extra = server.parse(
        [
            "--bin",
            str(binary),
            "-m",
            str(checkpoint),
            "--max-seq-len-override",
            "262144",
            "--num-tokens",
            "262144",
        ]
    )
    spec = server.build(args, extra)
    assert spec.cmd == [
        str(binary),
        "serve",
        "--model",
        str(checkpoint),
        "--max-seq-len-override",
        "262144",
        "--num-tokens",
        "262144",
    ]
    assert spec.data["model"] == checkpoint
    assert spec.data["layout"] == "hf"

    with pytest.raises(SystemExit, match="binary not found"):
        missing = tmp_path / "missing"
        server.build(server.parse(["--bin", str(missing), "-m", str(checkpoint)])[0], [])
    with pytest.raises(SystemExit, match="checkpoint directory not found"):
        server.build(server.parse(["--bin", str(binary), "-m", str(tmp_path / "nope")])[0], [])


def test_server_accepts_ftw_checkpoint(tmp_path: Path) -> None:
    from easyllama.servers.glm53flash import Glm53FlashServer, checkpoint_layout

    server = Glm53FlashServer()
    binary = tmp_path / "ft"
    binary.write_text("#!/bin/sh\n")
    checkpoint = tmp_path / "ckpt-ftw"
    checkpoint.mkdir()
    (checkpoint / "freetoken_weight.json").write_text("{}")
    (checkpoint / "freetoken-00000.ftw").write_bytes(b"\x00")
    assert checkpoint_layout(checkpoint) == "ftw"
    args, extra = server.parse(["--bin", str(binary), "-m", str(checkpoint)])
    spec = server.build(args, extra)
    assert spec.cmd[:3] == [str(binary), "serve", "--model"]
    assert spec.data["layout"] == "ftw"


def test_server_rejects_weightless_checkpoint(tmp_path: Path) -> None:
    from easyllama.servers.glm53flash import Glm53FlashServer

    server = Glm53FlashServer()
    binary = tmp_path / "ft"
    binary.write_text("#!/bin/sh\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(SystemExit, match="no FTW weights"):
        server.build(server.parse(["--bin", str(binary), "-m", str(empty)])[0], [])


def test_ftw_layout_is_snapshot_eligible(tmp_path: Path) -> None:
    from easyllama.helpers.hf import HF_EXTS, SNAP_PATTERNS, HuggingFace

    assert "*.ftw" in SNAP_PATTERNS
    assert "*.jinja" in SNAP_PATTERNS
    assert ".ftw" in HF_EXTS
    ftw = tmp_path / "ftw"
    ftw.mkdir()
    (ftw / "freetoken-00000.ftw").write_bytes(b"\x00")
    assert HuggingFace._snapshot_has_weights(ftw)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not HuggingFace._snapshot_has_weights(empty)


def test_orchestrator_routes_chat_to_freetoken() -> None:
    compiler = ProxyConfigCompiler(MODE.GLM53FLASH)
    assert compiler.image_for_model("glm53-chat", "ft serve") is IMAGE.FREETOKEN
    plan = compiler.compile(Path("config/config.glm5.3-flash.yml.example"))
    contract = plan.containers[0]
    assert contract.name == "easyllama-glm5.3-flash-freetoken-glm53-chat"
    assert contract.image is IMAGE.FREETOKEN
    assert contract.gpu and contract.health_path == "/v1/models"
    assert "/opt/venv/bin/easyllama server glm5.3-flash" in contract.command[-1]
    assert "--model" not in contract.command[-1]
    assert "-hf oakmindai/GLM-5.3-Flash-NVFP4-FTW" in contract.command[-1]
    assert "--max-seq-len-override 262144" in contract.command[-1]
    assert "--num-tokens" not in contract.command[-1]
    assert "--kv-reserve-tokens 65536" in contract.command[-1]
    assert "macros" not in plan.config


def test_config_loads_freetoken_role() -> None:
    settings = Config.load(mode_override="glm5.3-flash")
    assert settings.runtime.mode is MODE.GLM53FLASH
    assert settings.modes[MODE.GLM53FLASH].services == (
        IMAGE.LLAMASWAP,
        IMAGE.FREETOKEN,
    )
    profile = settings.resources.profile(IMAGE.FREETOKEN)
    assert (profile.cpu, profile.ram, profile.swap) == (
        CPU_WEIGHT.XHIGH,
        RAM_WEIGHT.XHIGH,
        SWAP_WEIGHT.MEDIUM,
    )
