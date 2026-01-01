"""PLC_RAG package exports."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("plc_rag")
except PackageNotFoundError:  # pragma: no cover - local package only
    __version__ = "0.1.0"

__all__ = ["__version__"]
