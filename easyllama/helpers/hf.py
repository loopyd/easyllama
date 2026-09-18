"""Provide repository-bound Hugging Face model operations."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import suppress
import fnmatch
from functools import partial
import logging
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any
import warnings

from .common import format_bytes, format_duration
from .logger import LOG as APP_LOG

LOG = APP_LOG.get(__name__)

HF_URL_BASE = "https://huggingface.co"
# Attach behavior: mirror an in-flight download owned by another process
# (for example a container backend) instead of racing a second download.
ATTACH_POLL_INTERVAL = 5.0
ATTACH_QUIET_CYCLES = 12
ATTACH_STALL_SECONDS = 120
HF_EXTS = (
    ".ftw",
    ".gguf",
    ".json",
    ".jinja",
    ".model",
    ".safetensors",
    ".safetensors.index.json",
    ".tiktoken",
    ".txt",
)
SNAP_PATTERNS = (
    "*.ftw",
    "*.gguf",
    "*.json",
    "*.jinja",
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

    Implements the progress-bar contract huggingface_hub drives across versions:
    context manager, update/close, description setters, Xet transfer hooks, and
    the class-level lock that snapshot_download shares with its worker threads.

    Attributes:
        total: The total.
        n: The n.
        desc: The desc.
        warmup: The warmup.
    _last_log: The last log.
    _last_n: The last n."""

    _lock: RLock | None = None

    @classmethod
    def get_lock(cls: type[HfProgress]) -> RLock:
        """Return the class-level lock, creating it on first use.

        Returns:
            RLock: The lock result."""
        if cls._lock is None:
            cls._lock = RLock()
        return cls._lock

    @classmethod
    def set_lock(cls: type[HfProgress], lock: RLock) -> None:
        """Share a lock with worker threads (tqdm lock API).

        Args:
            lock: The lock."""
        cls._lock = lock

    def __init__(self, *args: Any, warmup: str | None = None, **kwargs: Any) -> None:
        """Initialize the instance.

        Args:
            warmup: The warmup.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments."""
        self._iterable: Iterable[Any] | None = args[0] if args else None
        self.total = int(kwargs.get("total") or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.desc = str(kwargs.get("desc") or "Downloading model")
        self.warmup = warmup
        self._is_bytes = "unit" in kwargs
        self._rate: float | None = None
        self._last_log = time.monotonic()
        self._last_n = self.n

    @property
    def label(self) -> str:
        """Perform the label operation.

        Returns:
            str: The label result."""
        return f"{self.warmup} — {self.desc}" if self.warmup else self.desc

    @property
    def format_dict(self) -> dict[str, object]:
        """Expose the smoothed rate huggingface_hub aggregate reporting reads.

        Returns:
            dict[str, object]: The format dict result."""
        return {"rate": self._rate}

    def __iter__(self) -> Iterator[Any]:
        """Iterate a wrapped file list (tqdm concurrent-map compatibility).

        Returns:
            Iterator[Any]: The iter result.

        Raises:
            TypeError: If no iterable was provided."""
        if self._iterable is None:
            raise TypeError("HfProgress only iterates when constructed with an iterable")
        return iter(self._iterable)

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

    def update(self, n: int | float = 1) -> None:
        """Perform the update operation.

        Args:
            n: The n."""
        self.n = max(0, self.n + int(n or 0))
        now = time.monotonic()
        elapsed = now - self._last_log
        transferred = self.n - self._last_n
        if elapsed < 5 or transferred <= 0:
            return
        rate = transferred / elapsed
        self._rate = rate
        if self._is_bytes:
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
        elif self.total:
            LOG.info(
                "%s: %d/%d (%.1f%%)",
                self.label,
                self.n,
                self.total,
                self.n * 100 / self.total,
            )
        else:
            LOG.info("%s: %d", self.label, self.n)
        self._last_log, self._last_n = now, self.n

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        """Accept tqdm-compatible description updates.

        Args:
            desc: The desc.
            refresh: The refresh."""
        if desc is not None:
            self.desc = str(desc)
        if refresh:
            self.refresh()

    def set_description_str(self, desc: str | None = None, refresh: bool = True) -> None:
        """Accept tqdm raw-string description updates.

        Args:
            desc: The desc.
            refresh: The refresh."""
        self.set_description(desc, refresh=refresh)

    def update_transfer(self, n: int | float = 1) -> None:
        """Accept Xet network-transfer bytes without double-counting disk bytes.

        Args:
            n: The n."""
        del n

    def set_postfix_str(self, *_: object, **__: object) -> None:
        """Accept optional Hugging Face Xet transfer details."""

    def refresh(self, *_: object, **__: object) -> None:
        """Accept Hugging Face's final progress refresh."""

    def close(self) -> None:
        """Perform the close operation."""


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
        """Resolve the Hugging Face token: host env, then configured credentials.

        Returns:
            str | None: The token result."""
        if env_token := os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return env_token
        from ..config import Config

        try:
            return Config.load().load_auth().hf_token
        except (OSError, SystemExit, ValueError):
            return None

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
        revision: str | None = None,
    ) -> Path:
        """Download or return a cached repository file.

        Args:
            file: The file.
            cache_dir: The cache dir.
            warmup: The warmup.
            revision: The revision.

        Returns:
            Path: The get result.

        Raises:
            SystemExit: If the get operation cannot be completed."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        repo = self._repo()
        options: dict[str, Any] = {
            "repo_id": repo,
            "filename": file,
            "token": self.token,
            "cache_dir": cache_dir,
            "tqdm_class": partial(HfProgress, warmup=warmup),
        }
        if revision is not None:
            options["revision"] = revision

        def local_only() -> Path:
            return Path(hf_hub_download(**options, local_files_only=True))

        try:
            try:
                cached = local_only()
                LOG.info("using cached %s file %s", self.name, file)
                return cached
            except LocalEntryNotFoundError:
                pass
            attached = self._attach_if_in_flight(
                file, cache_dir=cache_dir, warmup=warmup, local_only=local_only
            )
            if attached is not None:
                LOG.info("using cached %s file %s", self.name, file)
                return attached
            LOG.info("downloading %s file %s", self.name, file)
            quiet = self.quiet()
            next(quiet)
            try:
                return Path(hf_hub_download(**options))
            finally:
                quiet.close()
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"failed to download {self.name} from {repo}/{file}: {exc}") from exc

    def snapshot(
        self,
        *,
        cache_dir: Path | None = None,
        warmup: str | None = None,
    ) -> Path:
        """Download a filtered repository snapshot.

        Args:
            cache_dir: The cache dir.
            warmup: The warmup.

        Returns:
            Path: The snapshot result.

        Raises:
            SystemExit: If the snapshot operation cannot be completed."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        repo = self._repo()
        options: dict[str, Any] = {
            "repo_id": repo,
            "allow_patterns": list(SNAP_PATTERNS),
            "token": self.token,
            "cache_dir": cache_dir,
            "tqdm_class": partial(HfProgress, warmup=warmup),
        }

        def local_only() -> Path:
            return Path(snapshot_download(**options, local_files_only=True))

        def local_weights() -> Path | None:
            try:
                local = local_only()
            except LocalEntryNotFoundError:
                return None
            return local if self._snapshot_has_weights(local) else None

        try:
            cached = local_weights()
            if cached is not None:
                LOG.info("using cached snapshot for %s (%s)", self.name, cached)
                return cached
            attached = self._attach_if_in_flight_snapshot(
                cache_dir=cache_dir, warmup=warmup, local_weights=local_weights
            )
            if attached is not None:
                return attached
            path = snapshot_download(**options)
            LOG.info("downloaded snapshot for %s (%s)", self.name, path)
            return Path(path)
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"failed to download {self.name} snapshot from {repo}: {exc}") from exc

    @staticmethod
    def _snapshot_has_weights(snapshot: Path) -> bool:
        """Return whether a local snapshot already holds model weights.

        Args:
            snapshot: The snapshot.

        Returns:
            bool: The snapshot has weights result."""
        return bool(
            tuple(snapshot.glob("*.gguf"))
            or tuple(snapshot.glob("*.safetensors"))
            or tuple(snapshot.glob("*.model"))
            or tuple(snapshot.glob("*.ftw"))
        )

    @staticmethod
    def _hub_cache_dir(cache_dir: Path | None) -> Path | None:
        """Resolve the concrete hub cache directory Hugging Face would use."""
        if cache_dir is not None:
            return cache_dir
        if env_cache := os.environ.get("HF_HUB_CACHE"):
            return Path(env_cache)
        hf_home = os.environ.get("HF_HOME")
        root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
        return root / "hub"

    @staticmethod
    def _repo_folder(repo: str) -> str:
        """Return the huggingface_hub cache folder name for a repository."""
        from huggingface_hub.file_download import repo_folder_name

        return repo_folder_name(repo_id=repo, repo_type="model")

    @classmethod
    def _download_lock_path(cls, hub: Path, repo: str, etag: str) -> Path:
        """Return the huggingface_hub download lock path for one file blob."""
        return hub / ".locks" / cls._repo_folder(repo) / f"{etag}.lock"

    @staticmethod
    def _lock_held(lock_path: Path) -> bool:
        """Probe non-blockingly whether another process holds a download lock.

        Mirrors huggingface_hub's WeakFileLock class selection (flock with a
        SoftFileLock fallback) so the probe agrees with the downloader."""
        from filelock import FileLock, SoftFileLock, Timeout

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        probe = FileLock(str(lock_path), timeout=0, mode=0o664)
        try:
            probe.acquire()
        except NotImplementedError:
            # Filesystem without flock support: mirror WeakFileLock's fallback.
            probe = SoftFileLock(str(lock_path), timeout=0)
            try:
                probe.acquire()
            except (NotImplementedError, Timeout, RuntimeError):
                return True
        except (Timeout, RuntimeError):
            return True
        else:
            with suppress(OSError):
                probe.release()
        return False

    @classmethod
    def _incomplete_blobs(cls, hub: Path) -> list[Path]:
        """List in-flight partial blobs in the hub cache.

        Covers both the hub-shared layout (blobs/, optionally fan-out) and the
        per-repo layout (models--<repo>/blobs/) used by huggingface_hub."""
        partials: list[Path] = []
        shared = hub / "blobs"
        if shared.is_dir():
            partials.extend(shared.rglob("*.incomplete"))
        for repo_blobs in sorted(hub.glob("models--*/blobs")):
            partials.extend(repo_blobs.glob("*.incomplete"))
        return sorted(set(partials), key=str)

    def _file_metadata(self, file: str) -> tuple[str, int] | None:
        """Return (etag, size) for one repository file, or None when unavailable."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.file_download import get_hf_file_metadata

        try:
            meta = get_hf_file_metadata(hf_hub_url(self._repo(), file), token=self.token)
        except Exception as exc:
            LOG.debug("could not fetch metadata for %s/%s: %s", self.repo, file, exc)
            return None
        if not meta.etag:
            return None
        return meta.etag, int(meta.size or 0)

    def _snapshot_file_metadata(self) -> list[tuple[str, int]] | None:
        """Return (etag, size) for every snapshot-eligible file, or None."""
        from huggingface_hub import HfApi

        try:
            entries = HfApi(token=self.token).list_repo_tree(repo_id=self._repo(), recursive=True)
        except Exception as exc:
            LOG.debug("could not list %s files: %s", self.repo, exc)
            return None
        files: list[tuple[str, int]] = []
        for entry in entries:
            if getattr(entry, "size", None) is None:
                continue
            if not any(
                fnmatch.fnmatch(Path(entry.path).name, pattern) for pattern in SNAP_PATTERNS
            ):
                continue
            etag = entry.lfs.sha256 if entry.lfs else entry.blob_id
            if etag:
                files.append((etag, int(entry.size)))
        return files or None

    def _attach_if_in_flight(
        self,
        file: str,
        *,
        cache_dir: Path | None,
        warmup: str | None,
        local_only: Callable[[], Path],
    ) -> Path | None:
        """Mirror a concurrent single-file download until it completes, if one is running.

        Returns None when nothing external is downloading this file (or when
        the metadata cannot be resolved), so the caller falls back to its own
        resumable download."""
        hub = self._hub_cache_dir(cache_dir)
        if hub is None or not self._incomplete_blobs(hub):
            return None
        meta = self._file_metadata(file)
        if meta is None:
            return None
        etag, size = meta
        lock_path = self._download_lock_path(hub, self._repo(), etag)
        if not self._lock_held(lock_path):
            return None

        def mirror() -> int:
            repo_blobs = hub / self._repo_folder(self._repo()) / "blobs"
            candidates = (
                repo_blobs / f"{etag}.incomplete",
                hub / "blobs" / f"{etag}.incomplete",
                hub / "blobs" / etag[:2] / f"{etag}.incomplete",
            )
            return max((c.stat().st_size for c in candidates if c.is_file()), default=0)

        def on_complete() -> Path | None:
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                return local_only()
            except LocalEntryNotFoundError:
                return None

        return self._attach_to_external_download(
            hub=hub,
            desc=file,
            total=size,
            initial=mirror(),
            lock_held=partial(self._lock_held, lock_path),
            mirror=mirror,
            on_complete=on_complete,
            warmup=warmup,
        )

    def _attach_if_in_flight_snapshot(
        self,
        *,
        cache_dir: Path | None,
        warmup: str | None,
        local_weights: Callable[[], Path | None],
    ) -> Path | None:
        """Mirror a concurrent snapshot download until it completes, if one is running."""
        hub = self._hub_cache_dir(cache_dir)
        if hub is None or not self._incomplete_blobs(hub):
            return None
        files = self._snapshot_file_metadata()
        if not files:
            return None
        etags = {etag for etag, _ in files}
        total = sum(size for _, size in files)
        repo = self._repo()

        def our_partials() -> list[Path]:
            return [
                path
                for path in self._incomplete_blobs(hub)
                if path.name.removesuffix(".incomplete") in etags
            ]

        def lock_held() -> bool:
            return any(
                self._lock_held(self._download_lock_path(hub, repo, etag)) for etag in sorted(etags)
            )

        def mirror() -> int:
            return sum(p.stat().st_size for p in our_partials() if p.is_file())

        if not (lock_held() or mirror() > 0):
            return None

        def on_complete() -> Path | None:
            return local_weights()

        return self._attach_to_external_download(
            hub=hub,
            desc=f"{self.name} snapshot ({len(files)} files)",
            total=total,
            initial=mirror(),
            lock_held=lock_held,
            mirror=mirror,
            on_complete=on_complete,
            warmup=warmup,
        )

    def _attach_to_external_download(
        self,
        *,
        hub: Path,
        desc: str,
        total: int,
        initial: int,
        lock_held: Callable[[], bool],
        mirror: Callable[[], int],
        on_complete: Callable[[], Path | None],
        warmup: str | None,
    ) -> Path | None:
        """Mirror progress of a download owned by another process until it settles.

        Returns the resolved local path once the external download verifiably
        completed, or None when it is no longer in flight so the caller can
        download itself (resuming any partial bytes).
        """
        LOG.info(
            "attaching to in-flight download of %s owned by another process; mirroring progress",
            desc,
        )
        reporter = HfProgress(total=total, initial=initial, desc=desc, warmup=warmup, unit="B")
        last_size = initial
        last_growth = time.monotonic()
        quiet_cycles = 0
        stall_warned = False
        while True:
            size = mirror()
            delta = max(0, size - last_size)
            active = lock_held() or delta > 0
            if delta:
                reporter.update(delta)
                last_size = size
                last_growth = time.monotonic()
                stall_warned = False
            elif (
                active
                and not stall_warned
                and time.monotonic() - last_growth >= ATTACH_STALL_SECONDS
            ):
                LOG.warning(
                    "%s: in-flight download shows no progress for %ss; still waiting",
                    reporter.label,
                    ATTACH_STALL_SECONDS,
                )
                stall_warned = True
            if not active:
                result = on_complete()
                if result is not None:
                    return result
                quiet_cycles += 1
                if quiet_cycles >= ATTACH_QUIET_CYCLES:
                    return None
            else:
                quiet_cycles = 0
            time.sleep(ATTACH_POLL_INTERVAL)

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
