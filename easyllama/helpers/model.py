"""Resolve local and Hugging Face models and convert them to cached GGUF files."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .hf import HuggingFace
from .logger import LOG as APP_LOG

LOG = APP_LOG.get(__name__)

LLAMA_DIR = Path("/opt/llama.cpp")
CONVERT = LLAMA_DIR / "convert_hf_to_gguf.py"
MODEL_SUFFIXES = (".gguf", ".safetensors", ".safetensors.index.json")


class Model:
    """Represent a named model source and its GGUF conversion settings.

    Attributes:
        name: Human-readable model role used in errors and progress output.
        source: Local model file or directory, when configured.
        hf: Bound Hugging Face repository, when configured.
        file: Hugging Face file path or selector.
        outtype: GGUF output type used for safetensors conversion.
    """

    def __init__(
        self,
        name: str,
        source: Path | None = None,
        *,
        hf: HuggingFace | None = None,
        file: str | None = None,
        outtype: str = "bf16",
    ) -> None:
        """Initialize a model source.

        Args:
            name: Human-readable model role.
            source: Optional local model file or directory.
            hf: Optional Hugging Face repository binding.
            file: Hugging Face file path or selector.
            outtype: GGUF output type for conversion.
        """
        self.name = name
        self.source = source
        self.hf = hf
        self.file = file
        self.outtype = outtype

    @classmethod
    def from_args(
        cls,
        name: str,
        *,
        local: Path | None,
        spec: str | None,
        repo: str | None,
        file: str | None,
        outtype: str = "bf16",
        default: str | None = None,
    ) -> Model:
        """Build a model from local and Hugging Face command-line arguments.

        Args:
            name: Human-readable model role.
            local: Optional local model source.
            spec: Combined Hugging Face ``repo[:file]`` specification.
            repo: Split Hugging Face repository argument.
            file: Split Hugging Face file argument.
            outtype: GGUF output type for conversion.
            default: Default Hugging Face selector or file.

        Returns:
            A configured model instance.

        Raises:
            SystemExit: If Hugging Face arguments conflict.
        """
        hf, file = HuggingFace.from_args(name, spec=spec, repo=repo, file=file, default=default)
        return cls(name, local, hf=hf, file=file, outtype=outtype)

    def resolve(self, *, required: bool = True) -> Path | None:
        """Resolve the model to a local GGUF path.

        Args:
            required: Whether a missing source is an error.

        Returns:
            The resolved GGUF path, or ``None`` for an optional missing model.

        Raises:
            SystemExit: If the source is missing, unsupported, or ambiguous.
        """
        source = self.source
        repo = self.hf.repo if self.hf else None
        if self.hf:
            file = self.hf.file(self.file, suffixes=MODEL_SUFFIXES)
            if file.endswith(".gguf"):
                return self.hf.get(file)
            source = self.hf.snapshot() / file
            if not source.exists():
                raise SystemExit(f"{self.name} source {file} not found in {repo}")
        if source is None:
            if required:
                raise SystemExit(f"{self.name} model path or HF selector is required")
            return None
        return Model(self.name, source, outtype=self.outtype).gguf(repo=repo)

    def pick(
        self,
        *,
        suffixes: tuple[str, ...],
        default: str | None = None,
    ) -> Path:
        """Select a local or Hugging Face model asset without conversion.

        Args:
            suffixes: Accepted Hugging Face file suffixes.
            default: Default Hugging Face selector or file.

        Returns:
            A local asset path.

        Raises:
            SystemExit: If no source is configured.
        """
        if self.hf:
            return self.hf.get(self.hf.file(self.file, suffixes=suffixes, default=default))
        if self.source is None:
            raise SystemExit(f"{self.name} path is required")
        return self.source

    def gguf(self, *, repo: str | None = None) -> Path:
        """Return the source as GGUF, converting safetensors when needed.

        Args:
            repo: Optional repository ID used to name generic model files.

        Returns:
            Existing or converted GGUF path.

        Raises:
            SystemExit: If conversion fails or the source is unsupported.
        """
        if self.source is None:
            raise SystemExit(f"{self.name} model path is required")
        src, root, convert = self._classify(repo=repo)
        if not convert:
            return src
        if not CONVERT.is_file():
            raise SystemExit(f"llama.cpp converter not found at {CONVERT}")

        out = self._gguf_path(src, root=root, repo=repo)
        if out.is_file():
            return out
        tmp = out.with_name(f".{out.name}.tmp")
        tmp.unlink(missing_ok=True)
        LOG.info("Converting %s to cached GGUF: %s", self.name, out)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(CONVERT),
                    str(root),
                    "--outfile",
                    str(tmp),
                    "--outtype",
                    self.outtype,
                ],
                check=True,
                cwd=LLAMA_DIR,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise SystemExit(f"failed to convert {self.name} to GGUF: {exc}") from exc
        if not tmp.is_file():
            raise SystemExit(f"converter did not create expected GGUF at {tmp}")
        tmp.replace(out)
        return out

    def _classify(self, *, repo: str | None) -> tuple[Path, Path, bool]:
        """Classify the source path and whether it needs conversion."""
        assert self.source is not None
        src = self.source
        if src.is_file():
            name = src.name.lower()
            if name.endswith(".gguf"):
                return src, src.parent, False
            if name.endswith((".safetensors", ".safetensors.index.json")):
                return src, src.parent, True
            raise SystemExit(f"unsupported model file type at {src}")
        if not src.is_dir():
            raise SystemExit(f"model source not found at {src}")

        target = self._gguf_path(src, root=src, repo=repo)
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

    def _gguf_path(self, src: Path, *, root: Path, repo: str | None) -> Path:
        """Build the cached GGUF output path for a source."""
        name = src.name.lower()
        stem = src.stem if src.is_file() else root.name
        if name.endswith(".safetensors.index.json") or stem == "model":
            stem = repo.rsplit("/", 1)[-1] if repo else root.name
        return root / f"{stem}-{self.outtype.upper()}.gguf"
