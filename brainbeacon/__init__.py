# brainbeacon/__init__.py
from importlib.metadata import PackageNotFoundError, version

from .pipeline.cell_embedding import run_bbcellformer_pipeline

__all__ = ["run_bbcellformer_pipeline"]

try:
    __version__ = version("brainbeacon")
except PackageNotFoundError:
    __version__ = "dev"
