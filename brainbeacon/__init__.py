# brainbeacon/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainbeacon")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = ["__version__"]
