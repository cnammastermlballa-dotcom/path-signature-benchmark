"""
Run information shared by the Chapter 5 scripts.

`log_versions(logger)` logs the Python version and the versions of the
packages that determine the results, at the start of every script, so that
each run log records the software environment of the thesis results.

Versions are read from the installed package metadata (importlib.metadata),
not by importing the packages: torch is optional (only 5_3_1 needs it) and
importing it would slow every script down. A missing package is logged as
"not installed".
"""

import logging
import platform
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["numpy", "scikit-learn", "iisignature", "torch"]


def package_versions() -> dict:
    """Python and package versions, {name: version or 'not installed'}."""
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def log_versions(logger: logging.Logger) -> None:
    """Log the Python and package versions on one line."""
    logger.info(
        "Versions: %s",
        ", ".join(f"{k} {v}" for k, v in package_versions().items()),
    )
