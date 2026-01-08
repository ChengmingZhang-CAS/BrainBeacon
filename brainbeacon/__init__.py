from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("brainbeacon")
except PackageNotFoundError:
    __version__ = "0.1.0"