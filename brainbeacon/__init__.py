# brainbeacon/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainbeacon")
except PackageNotFoundError:
    __version__ = "dev"

try:
    from .pipeline.cell_embedding import run_bbcellformer_pipeline
    from .pipeline.cell_label_transfer import run_label_transfer_pipeline

    __all__ = [
        "run_bbcellformer_pipeline",
        "run_label_transfer_pipeline",
    ]
except Exception:
    # Allow importing brainbeacon in minimal environments (e.g. docs/CI)
    __all__ = []
