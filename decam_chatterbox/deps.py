"""Optional-dependency handling with actionable error messages.

decam-chatterbox degrades rather than dies when an optional package is
missing: no ``ligo.skymap`` costs the plots and the skymap decode, no
``slack_sdk`` costs the posting. That is right, but a bare ``No module named
'ligo'`` does not say *which* interpreter looked, and that is the one fact
needed to fix it -- an install landing in a different Python is by far the
most common cause.

``require`` produces the message; ``CAPABILITIES`` is the same information in
a form `decam_chatterbox.doctor` can report on before anything goes wrong.
"""

import importlib
import logging
import sys
from dataclasses import dataclass

__all__ = ["Capability", "CAPABILITIES", "require", "probe"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capability:
    """One thing decam-chatterbox can do, and what it needs to do it."""

    name: str
    modules: tuple[str, ...]
    install: str
    purpose: str
    consequence: str
    extra: str = ""


#: Everything optional, in the order a reader most likely cares about.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="almanac",
        modules=("astroplan",),
        install="astroplan",
        purpose="sun/moon times and the accessible-dark-hours map for CTIO",
        consequence="alerts cannot be processed at all",
        extra="gw",
    ),
    Capability(
        name="skymap",
        # ligo.skymap is a namespace package: the failure surfaces as
        # "No module named 'ligo'", which is why it reads so cryptically.
        modules=("ligo.skymap.io",),
        install="ligo.skymap",
        purpose="decoding the multi-order sky map embedded in each notice, and the sky projections",
        consequence="no localization, no dark-hours plot, no distance",
        extra="gw",
    ),
    Capability(
        name="gracedb",
        modules=("ligo.gracedb.rest",),
        install="ligo-gracedb",
        purpose="optionally fetching the binned chirp-mass estimate from GraceDB",
        consequence="the chirp-mass line is omitted; everything else is unaffected",
        extra="gw",
    ),
    Capability(
        name="slack",
        modules=("slack_sdk",),
        install="slack-sdk",
        purpose="posting messages and uploading plots",
        consequence="output is written locally instead of posted",
        extra="slack",
    ),
    Capability(
        name="scimma",
        modules=("hop",),
        install="hop-client",
        purpose="subscribing to igwn.gwalert on SCIMMA's Hopskotch broker, the default monitoring path",
        consequence="alerts cannot be read live; use ingest.kind 'files' or 'replay'",
        extra="scimma",
    ),
    Capability(
        name="avro",
        modules=("fastavro",),
        install="fastavro",
        purpose="reading .avro alert records (SCIMMA's on-the-wire format)",
        consequence="only .json record files can be replayed",
        extra="scimma",
    ),
)

_BY_MODULE = {c.modules[0]: c for c in CAPABILITIES}


def _hint(module: str, install: str) -> str:
    """Build the install hint, naming the interpreter that actually looked."""
    return (
        f"{sys.executable} cannot import {module}. Install it into *that* "
        f"interpreter: '{sys.executable} -m pip install {install}'. "
        "A package installed under a different python will not be found."
    )


def require(module: str, purpose: str = "", install: str = ""):
    """Import a module, or raise an ImportError that says how to fix it.

    Parameters
    ----------
    module : `str`
        Import name, e.g. ``"ligo.skymap.io"``.
    purpose : `str`
        What it is needed for. Looked up from `CAPABILITIES` when omitted.
    install : `str`
        What to pip install. Looked up from `CAPABILITIES` when omitted.

    Returns
    -------
    module : module
        The imported module.

    Raises
    ------
    ImportError
        Naming ``sys.executable``, since the usual cause is an install that
        landed in a different interpreter.
    """
    known = _BY_MODULE.get(module)
    purpose = purpose or (known.purpose if known else "")
    install = install or (known.install if known else module)

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = f" ({purpose})" if purpose else ""
        raise ImportError(f"{module} is required{what}. {_hint(module, install)} [{exc}]") from exc


def probe(capability: Capability) -> tuple[bool, str]:
    """Check whether a capability's modules are importable.

    Returns
    -------
    ok : `bool`
        True when every module imports.
    detail : `str`
        Empty when ok, otherwise the first failure.
    """
    for module in capability.modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # ImportError, but a broken install can raise anything
            return False, f"{module}: {type(exc).__name__}: {exc}"
    return True, ""
