try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - only when not installed via setuptools-scm
    __version__ = "0.0.0"
