"""
Lightweight package to run RBPF and baseline filters for the Ecolar study.

The raw Colab export lives in ``Raw_files`` for provenance. This package
contains a cleaned, configurable implementation that can be imported or run
from the CLI.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rbpf_lab")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
