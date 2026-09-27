"""Environment diagnosis: what works, what does not, and how to fix it.

decam-chatterbox degrades instead of dying when an optional package is
missing, which is right for a service but means a broken deployment shows up
as a scatter of warnings across a live alert. ``decam-chatterbox doctor``
collects the same information up front, in one place, naming the interpreter
each check ran in -- because a package installed into a different python is
the usual cause.
"""

import logging
import sys
from dataclasses import dataclass, field

from .config import Config
from .deps import CAPABILITIES, probe

__all__ = ["Check", "diagnose", "format_report"]

logger = logging.getLogger(__name__)

#: decam-chatterbox uses ``X | Y`` annotations evaluated at runtime, so this
#: is a hard floor rather than a style preference.
MIN_PYTHON = (3, 11)


@dataclass
class Check:
    """One diagnostic result."""

    name: str
    ok: bool
    detail: str
    #: What to do about it. Empty when ok.
    fix: str = ""
    #: False when a failure is tolerable, so the summary can rank it.
    fatal: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``ok``, ``FAIL`` or ``warn``."""
        if self.ok:
            return "ok"
        return "FAIL" if self.fatal else "warn"


def _capability_checks(config: Config) -> list[Check]:
    """Probe every optional capability in this interpreter."""
    checks = []
    for capability in CAPABILITIES:
        ok, detail = probe(capability)
        fix = ""
        notes = []
        if not ok:
            notes.append(f"Without it: {capability.consequence}.")
            fix = f"{sys.executable} -m pip install {capability.install}"
        checks.append(
            Check(
                name=f"{capability.name}: {capability.purpose}",
                ok=ok,
                detail=detail or "importable",
                fix=fix,
                fatal=capability.name in ("almanac", "skymap"),
                notes=notes,
            )
        )
    return checks


def _ctio_site_check() -> Check:
    """Can astropy resolve the CTIO site?"""
    try:
        from astropy.coordinates import EarthLocation

        location = EarthLocation.of_site("ctio")
    except Exception as exc:
        return Check(
            name="CTIO site lookup",
            ok=False,
            detail=f"astropy could not resolve site 'ctio': {exc}",
            fix="Update astropy's site registry (pip install -U astropy-iers-data), or check network access",
            fatal=True,
        )
    return Check(
        name="CTIO site lookup",
        ok=True,
        detail=f"lat {location.lat.deg:.4f}, lon {location.lon.deg:.4f}, height {location.height:.0f}",
    )


def _slack_token_check(config: Config) -> Check:
    """Is a bot token available?"""
    if config.slack_token:
        return Check(
            name="Slack token",
            ok=True,
            detail=f"{config.slack.bot_token_env} is set; posting to {config.slack.channel}",
        )
    return Check(
        name="Slack token",
        ok=False,
        detail=f"{config.slack.bot_token_env} is not set",
        fix=f"export {config.slack.bot_token_env}=xoxb-...",
        notes=["Without it: payloads and plots are written locally instead of posted."],
    )


def _hop_auth_check(config: Config) -> Check:
    """Does hop-client have credentials configured for SCIMMA?"""
    from pathlib import Path

    auth_file = Path("~/.config/hop/auth.toml").expanduser()
    if config.ingest.kind != "scimma":
        detail = f"not needed (ingest.kind={config.ingest.kind!r})"
        return Check(name="hop-client credentials", ok=True, detail=detail)
    if auth_file.is_file():
        return Check(name="hop-client credentials", ok=True, detail=str(auth_file))
    return Check(
        name="hop-client credentials",
        ok=False,
        detail=f"{auth_file} not found",
        fix="Run 'hop auth add' with your SCIMMA (hop.scimma.org) username and password",
        notes=["Without it: the scimma ingest source cannot authenticate."],
    )


def diagnose(config: Config) -> list[Check]:
    """Run every diagnostic and return the results in report order."""
    checks: list[Check] = [
        Check(
            name="interpreter",
            ok=sys.version_info[:2] >= MIN_PYTHON,
            detail=f"{sys.executable} (Python {'.'.join(map(str, sys.version_info[:3]))})",
            fix=f"decam-chatterbox needs Python >= {'.'.join(map(str, MIN_PYTHON))}",
            fatal=True,
        )
    ]
    checks += _capability_checks(config)
    checks.append(_ctio_site_check())
    checks.append(_hop_auth_check(config))
    checks.append(_slack_token_check(config))
    return checks


def format_report(checks: list[Check]) -> str:
    """Render checks as a readable report."""
    width = max(len(c.name) for c in checks) + 2
    lines = ["", "decam-chatterbox environment", "=" * 29, ""]
    for check in checks:
        lines.append(f"[{check.status:>4}] {check.name:<{width}} {check.detail}")
        for note in check.notes:
            lines.append(f"{'':>7} {'':<{width}} {note}")
        if check.fix and not check.ok:
            lines.append(f"{'':>7} {'':<{width}} fix: {check.fix}")

    failures = [c for c in checks if not c.ok and c.fatal]
    warnings = [c for c in checks if not c.ok and not c.fatal]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} blocking problem(s): " + ", ".join(c.name for c in failures))
    if warnings:
        lines.append(f"{len(warnings)} degraded capability/ies: " + ", ".join(c.name for c in warnings))
    if not failures and not warnings:
        lines.append("Everything checks out.")
    lines.append("")
    return "\n".join(lines)
