from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ..logger import get_logger

LOG = get_logger(__name__)


def _format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _format_duration(seconds: float) -> str:
    return f"{max(0, round(seconds))}s"


LLAMA_DIR = Path("/opt/llama.cpp")
CONVERT = LLAMA_DIR / "convert_hf_to_gguf.py"
HF_EXTS = (
    ".gguf",
    ".json",
    ".model",
    ".safetensors",
    ".safetensors.index.json",
    ".tiktoken",
    ".txt",
)
SNAP_PATTERNS = (
    "*.gguf",
    "*.json",
    "*.model",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.txt",
    "*.tiktoken",
    "README.md",
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def hf_spec(spec: str | None, *, default: str | None = None) -> tuple[str | None, str | None]:
    if not spec:
        return None, default
    repo, sep, file = spec.partition(":")
    if not repo or "/" not in repo:
        raise SystemExit(f"invalid HF spec {spec!r}; expected repo[:file]")
    if sep and not file:
        raise SystemExit(f"invalid HF spec {spec!r}; expected repo:file")
    return repo, file or default


def hf_args(
    *,
    label: str,
    spec: str | None,
    repo: str | None,
    file: str | None,
    default: str | None = None,
) -> tuple[str | None, str | None]:
    if spec and (repo or file):
        raise SystemExit(
            f"use either {label} HF spec or split {label} HF repo/file flags, not both"
        )
    spec_repo, spec_file = hf_spec(spec, default=default)
    repo = repo or spec_repo
    file = file or spec_file or default
    if file and not repo:
        raise SystemExit(f"{label} HF file selector requires a matching {label} HF repo")
    return repo, file


def is_hf_file(value: str) -> bool:
    lowered = value.lower()
    return "/" in value or any(lowered.endswith(ext) for ext in HF_EXTS)


def hf_file(
    repo: str,
    choice: str | None,
    label: str,
    *,
    suffixes: tuple[str, ...],
    default: str | None = None,
) -> str:
    choice = choice or default
    if not choice:
        raise SystemExit(f"{label} HF selector is required for {repo}")
    if is_hf_file(choice):
        return choice

    from huggingface_hub import HfApi

    try:
        files = HfApi(token=token()).list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"failed to inspect {label} files in {repo}: {exc}") from exc

    wanted = choice.upper()
    suffixes = tuple(item.upper() for item in suffixes)
    seen: list[str] = []
    matches: list[str] = []
    for item in files:
        name = Path(item).name
        if name.startswith("mmproj-"):
            continue
        upper = name.upper()
        if not upper.endswith(suffixes):
            continue
        seen.append(item)
        for suffix in suffixes:
            if upper in {wanted, f"{wanted}{suffix}"}:
                matches.append(item)
                break
            if (
                upper.endswith(f"-{wanted}{suffix}")
                or upper.endswith(f"_{wanted}{suffix}")
                or upper.endswith(f".{wanted}{suffix}")
            ):
                matches.append(item)
                break

    if len(matches) == 1:
        return matches[0]
    if not matches:
        lines = "\n - ".join(sorted(seen))
        raise SystemExit(
            f"no {label} file matching {choice!r} found in {repo}\nAvailable files:\n - {lines}"
        )
    lines = "\n - ".join(sorted(matches))
    raise SystemExit(
        f"multiple {label} files matched {choice!r} in {repo}; "
        "use repo:file explicitly\n"
        f"Matched files:\n - {lines}"
    )


class _HFProgress:
    """Tqdm-compatible progress reporter for Hugging Face downloads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = int(kwargs.get("total") or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.desc = str(kwargs.get("desc") or "Downloading model")
        self._started = self._last_log = time.monotonic()
        self._last_n = self.n

    def __enter__(self) -> _HFProgress:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        elapsed = now - self._last_log
        if elapsed < 5:
            return
        rate = (self.n - self._last_n) / elapsed
        if self.total and rate > 0:
            LOG.info(
                "%s: %s/%s (%.1f%%, %s/s, ETA %s)",
                self.desc,
                _format_bytes(self.n),
                _format_bytes(self.total),
                self.n * 100 / self.total,
                _format_bytes(rate),
                _format_duration((self.total - self.n) / rate),
            )
        else:
            LOG.info(
                "%s: %s (%s/s, ETA unknown)",
                self.desc,
                _format_bytes(self.n),
                _format_bytes(rate),
            )
        self._last_log, self._last_n = now, self.n

    def close(self) -> None:
        elapsed = time.monotonic() - self._started
        if self.n:
            LOG.info(
                "%s complete: %s in %s",
                self.desc,
                _format_bytes(self.n),
                _format_duration(elapsed),
            )


def hf_get(repo: str, file: str, label: str, *, cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        options: dict[str, Any] = {
            "repo_id": repo,
            "filename": file,
            "token": token(),
            "cache_dir": cache_dir,
            "tqdm_class": _HFProgress,
        }
        path = hf_hub_download(**options)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"failed to download {label} from {repo}/{file}: {exc}") from exc
    return Path(path)


def hf_snap(repo: str, label: str) -> Path:
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=list(SNAP_PATTERNS),
            token=token(),
        )
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"failed to download {label} snapshot from {repo}: {exc}") from exc
    return Path(path)


def pick_asset(
    *,
    label: str,
    local: Path | None,
    repo: str | None,
    file: str | None,
    suffixes: tuple[str, ...],
    default: str | None = None,
) -> Path:
    if repo:
        return hf_get(repo, hf_file(repo, file, label, suffixes=suffixes, default=default), label)
    if local is None:
        raise SystemExit(f"{label} path is required")
    return local


def gguf_path(src: Path, *, root: Path, repo: str | None, outtype: str) -> Path:
    name = src.name.lower()
    stem = src.stem if src.is_file() else root.name
    if name.endswith(".safetensors.index.json") or stem == "model":
        stem = repo.rsplit("/", 1)[-1] if repo else root.name
    return root / f"{stem}-{outtype.upper()}.gguf"


def model_src(src: Path, *, repo: str | None, outtype: str) -> tuple[Path, Path, bool]:
    if src.is_file():
        name = src.name.lower()
        if name.endswith(".gguf"):
            return src, src.parent, False
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            return src, src.parent, True
        raise SystemExit(f"unsupported model file type at {src}")

    if not src.is_dir():
        raise SystemExit(f"model source not found at {src}")

    target = gguf_path(src, root=src, repo=repo, outtype=outtype)
    if target.is_file():
        return target, src, False

    ggufs = sorted(src.glob("*.gguf"))
    if len(ggufs) == 1:
        return ggufs[0], src, False
    if len(ggufs) > 1:
        raise SystemExit(f"multiple GGUF model files found in {src}; specify one explicitly")

    if any(src.glob("*.safetensors")) or any(src.glob("*.safetensors.index.json")):
        return src, src, True
    raise SystemExit(f"no GGUF or safetensors model source found in {src}")


def ensure_gguf(src: Path, *, label: str, repo: str | None = None, outtype: str) -> Path:
    src, root, convert = model_src(src, repo=repo, outtype=outtype)
    if not convert:
        return src
    if not CONVERT.is_file():
        raise SystemExit(f"llama.cpp converter not found at {CONVERT}")

    out = gguf_path(src, root=root, repo=repo, outtype=outtype)
    if out.is_file():
        return out

    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        tmp.unlink()

    LOG.info("Converting %s to cached GGUF: %s", label, out)
    try:
        subprocess.run(
            [
                sys.executable,
                str(CONVERT),
                str(root),
                "--outtype",
                outtype,
                "--outfile",
                str(tmp),
            ],
            check=True,
            cwd=LLAMA_DIR,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise SystemExit(f"failed to convert {label} to GGUF: {exc}") from exc

    if not tmp.is_file():
        raise SystemExit(f"converter did not produce {tmp}")
    tmp.replace(out)
    return out
