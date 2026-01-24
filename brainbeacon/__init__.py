from importlib.metadata import PackageNotFoundError, version

from .pipeline.cell_embedding import run_bbcellformer_pipeline
from .pipeline.cell_label_transfer import run_label_transfer_pipeline

__all__ = [
    "__version__",
    "run_bbcellformer_pipeline",
    "run_label_transfer_pipeline",
]

try:
    __version__ = version("brainbeacon")
except PackageNotFoundError:
    __version__ = "0.0.1"
