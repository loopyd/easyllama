"""Provide repository-bound Hugging Face model operations."""

from __future__ import annotations

from collections.abc import Generator
from functools import partial
import logging
import os
from pathlib import Path
import time
from typing import Any
import warnings

from .common import format_bytes, format_duration
from .logger import LOG as APP_LOG

LOG = APP_LOG.get(__name__)

HF_URL_BASE = "https://huggingface.co"
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


class HfProgress:
    """Report rate-limited Hugging Face download progress.

    Attributes:
        total: The total.
        n: The n.
        desc: The desc.
        warmup: The warmup.
    _last_log: The last log.
    _last_n: The last n."""

    def __init__(self, *args: Any, warmup: str | None = None, **kwargs: Any) -> None:
        """Initialize the instance.

        Args:
            warmup: The warmup.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments."""
        self.total = int(kwargs.get("total") or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.desc = str(kwargs.get("desc") or "Downloading model")
        self.warmup = warmup
        self._last_log = time.monotonic()
        self._last_n = self.n

    @property
    def label(self) -> str:
        """Perform the label operation.

        Returns:
            str: The label result."""
        return f"{self.warmup} — {self.desc}" if self.warmup else self.desc

    def __enter__(self) -> HfProgress:
        """Enter the context manager.

        Returns:
            HfProgress: The enter result."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager.

        Args:
            *_: Additional positional arguments."""
        self.close()

    def update(self, n: int = 1) -> None:
        """Perform the update operation.

        Args:
            n: The n."""
        self.n += n
        now = time.monotonic()
        elapsed = now - self._last_log
        transferred = self.n - self._last_n
        if elapsed < 5 or transferred <= 0:
            return
        rate = transferred / elapsed
        if self.total:
            LOG.info(
                "%s: %s/%s (%.1f%%, %s/s, ETA %s)",
                self.label,
                format_bytes(self.n),
                format_bytes(self.total),
                self.n * 100 / self.total,
                format_bytes(rate),
                format_duration((self.total - self.n) / rate),
            )
        else:
            LOG.info(
                "%s: %s (%s/s, ETA unknown)",
                self.label,
                format_bytes(self.n),
                format_bytes(rate),
            )
        self._last_log, self._last_n = now, self.n

    def close(self) -> None:
        """Perform the close operation."""
        pass


class HuggingFace:
    """Perform operations against a named Hugging Face repository.

    Attributes:
        name: The name.
        repo: The repo."""

    def __init__(self, name: str, repo: str | None = None) -> None:
        """Initialize the instance.

        Args:
            name: The name.
            repo: The repo."""
        self.name = name
        self.repo = repo

    @property
    def token(self) -> str | None:
        """Perform the token operation.

        Returns:
            str | None: The token result."""
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    @staticmethod
    def mmproj_url(spec: str) -> str:
        """Build a Hugging Face multimodal projector URL.

        Args:
            spec: The spec.

        Returns:
            str: The mmproj url result.

        Raises:
            SystemExit: If the mmproj url operation cannot be completed."""
        parts = spec.split("/", 2)
        if len(parts) != 3 or not all(parts):
            raise SystemExit(f"EASYLLAMA_HF_MMPROJ must be <owner>/<repo>/<file.gguf>; got: {spec}")
        owner, repo, filename = parts
        return f"{HF_URL_BASE}/{owner}/{repo}/blob/main/{filename}"

    @staticmethod
    def parse_spec(
        spec: str | None, *, default: str | None = None
    ) -> tuple[str | None, str | None]:
        """Parse a Hugging Face repository and optional file specification.

        Args:
            spec: The spec.
            default: The default.

        Returns:
            tuple[str | None, str | None]: The parse spec result.

        Raises:
            SystemExit: If the parse spec operation cannot be completed."""
        if not spec:
            return None, default
        repo, sep, file = spec.partition(":")
        if not repo or "/" not in repo:
            raise SystemExit(f"invalid HF spec {spec!r}; expected repo[:file]")
        if sep and not file:
            raise SystemExit(f"invalid HF spec {spec!r}; expected repo:file")
        return repo, file or default

    @classmethod
    def from_args(
        cls,
        name: str,
        *,
        spec: str | None,
        repo: str | None,
        file: str | None,
        default: str | None = None,
    ) -> tuple[HuggingFace | None, str | None]:
        """Resolve split or combined Hugging Face command-line arguments.

        Args:
            name: The name.
            spec: The spec.
            repo: The repo.
            file: The file.
            default: The default.

        Returns:
            tuple[HuggingFace | None, str | None]: The from args result.

        Raises:
            SystemExit: If the from args operation cannot be completed."""
        if spec and (repo or file):
            raise SystemExit(
                f"use either {name} HF spec or split {name} HF repo/file flags, not both"
            )
        spec_repo, spec_file = cls.parse_spec(spec, default=default)
        repo = repo or spec_repo
        file = file or spec_file or default
        if file and not repo:
            raise SystemExit(f"{name} HF file selector requires a matching {name} HF repo")
        return (cls(name, repo) if repo else None), file

    def quiet(self) -> Generator[None, None, None]:
        """Yield once while Hugging Face client logging is suppressed.

        Yields:
            Control to the caller with client loggers set to warning level.
        """
        loggers = tuple(
            APP_LOG.get(name)
            for name in ("httpx", "huggingface_hub", "huggingface_hub.utils._http")
        )
        levels = tuple(logger.level for logger in loggers)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
                for logger in loggers:
                    logger.setLevel(logging.WARNING)
                yield
        finally:
            for logger, level in zip(loggers, levels, strict=True):
                logger.setLevel(level)

    def file(
        self,
        choice: str | None,
        *,
        suffixes: tuple[str, ...],
        default: str | None = None,
    ) -> str:
        """Resolve a repository file from an exact path or selector.

        Args:
            choice: The choice.
            suffixes: The suffixes.
            default: The default.

        Returns:
            str: The file result.

        Raises:
            SystemExit: If the file operation cannot be completed."""
        repo = self._repo()
        choice = choice or default
        if not choice:
            raise SystemExit(f"{self.name} HF selector is required for {repo}")
        if "/" in choice or any(choice.lower().endswith(ext) for ext in HF_EXTS):
            return choice

        from huggingface_hub import HfApi

        quiet = self.quiet()
        next(quiet)
        try:
            files = HfApi(token=self.token).list_repo_files(repo_id=repo, repo_type="model")
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"failed to inspect {self.name} files in {repo}: {exc}") from exc
        finally:
            quiet.close()

        wanted = choice.upper()
        upper_suffixes = tuple(item.upper() for item in suffixes)
        seen: list[str] = []
        matches: list[str] = []
        for item in files:
            name = Path(item).name
            if name.startswith("mmproj-"):
                continue
            upper = name.upper()
            if not upper.endswith(upper_suffixes):
                continue
            seen.append(item)
            if any(
                upper in {wanted, f"{wanted}{suffix}"}
                or upper.endswith(
                    (f"-{wanted}{suffix}", f"_{wanted}{suffix}", f".{wanted}{suffix}")
                )
                for suffix in upper_suffixes
            ):
                matches.append(item)

        if len(matches) == 1:
            return matches[0]
        if not matches:
            lines = "\n - ".join(sorted(seen))
            raise SystemExit(
                f"no {self.name} file matching {choice!r} found in {repo}\n"
                f"Available files:\n - {lines}"
            )
        lines = "\n - ".join(sorted(matches))
        raise SystemExit(
            f"multiple {self.name} files matched {choice!r} in {repo}; use repo:file explicitly\n"
            f"Matched files:\n - {lines}"
        )

    def get(
        self,
        file: str,
        *,
        cache_dir: Path | None = None,
        warmup: str | None = None,
    ) -> Path:
        """Download or return a cached repository file.

        Args:
            file: The file.
            cache_dir: The cache dir.
            warmup: The warmup.

        Returns:
            Path: The get result.

        Raises:
            SystemExit: If the get operation cannot be completed."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        repo = self._repo()
        try:
            options: dict[str, Any] = {
                "repo_id": repo,
                "filename": file,
                "token": self.token,
                "cache_dir": cache_dir,
                "tqdm_class": partial(HfProgress, warmup=warmup),
            }
            try:
                path = hf_hub_download(**options, local_files_only=True)
            except LocalEntryNotFoundError:
                quiet = self.quiet()
                next(quiet)
                try:
                    path = hf_hub_download(**options)
                finally:
                    quiet.close()
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"failed to download {self.name} from {repo}/{file}: {exc}") from exc
        return Path(path)

    def snapshot(self) -> Path:
        """Download a filtered repository snapshot.

        Returns:
            Path: The snapshot result.

        Raises:
            SystemExit: If the snapshot operation cannot be completed."""
        from huggingface_hub import snapshot_download

        repo = self._repo()
        try:
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=list(SNAP_PATTERNS),
                token=self.token,
            )
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"failed to download {self.name} snapshot from {repo}: {exc}") from exc
        return Path(path)

    def pick(
        self,
        *,
        local: Path | None,
        file: str | None,
        suffixes: tuple[str, ...],
        default: str | None = None,
    ) -> Path:
        """Choose a repository asset or a local path.

        Args:
            local: The local.
            file: The file.
            suffixes: The suffixes.
            default: The default.

        Returns:
            Path: The pick result.

        Raises:
            SystemExit: If the pick operation cannot be completed."""
        if self.repo:
            return self.get(self.file(file, suffixes=suffixes, default=default))
        if local is None:
            raise SystemExit(f"{self.name} path is required")
        return local

    def _repo(self) -> str:
        """Perform the internal repo operation.

        Returns:
            str: The repo result.

        Raises:
            SystemExit: If the repo operation cannot be completed."""
        if not self.repo:
            raise SystemExit(f"{self.name} Hugging Face repo is required")
        return self.repo
