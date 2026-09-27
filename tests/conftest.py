"""Shared test fixtures.

Notices are built programmatically at a small nside, so the suite runs in
seconds and needs no network access. `scripts/fetch_samples.py` downloads the
official real-schema LVK sample notices separately, for manual/integration
checks against `decam-chatterbox replay`.
"""

import base64
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

#: Small enough that writing and reading a skymap costs nothing in a test.
TEST_NSIDE = 32


def disc_probability_map(ra: float, dec: float, radius_deg: float, nside: int = TEST_NSIDE) -> np.ndarray:
    """A normalized RING-ordered probability map, uniform inside a disc."""
    prob = np.zeros(hp.nside2npix(nside))
    disc = hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(radius_deg))
    prob[disc] = 1.0
    return prob / prob.sum()


def make_skymap_bytes(
    ra: float = 60.0,
    dec: float = -40.0,
    radius_deg: float = 10.0,
    nside: int = TEST_NSIDE,
    distmean: float = 120.0,
    diststd: float = 30.0,
) -> bytes:
    """Raw (gzipped) FITS bytes for a small disc-shaped probability skymap.

    Built with ``ligo.skymap.io.write_sky_map``, the same library
    `decam_chatterbox.astro.skymap.read_notice_skymap` reads with, so this
    exercises the real read path rather than a hand-rolled substitute.
    """
    import tempfile

    from ligo.skymap.io import write_sky_map

    prob = disc_probability_map(ra, dec, radius_deg, nside)
    with tempfile.NamedTemporaryFile(suffix=".fits.gz") as f:
        write_sky_map(f.name, prob, nest=False, distmean=distmean, diststd=diststd)
        return Path(f.name).read_bytes()


def make_notice(
    superevent_id: str = "S260814a",
    alert_type: str = "PRELIMINARY",
    group: str = "CBC",
    pipeline: str = "gstlal",
    search: str = "AllSky",
    far: float = 1e-10,
    significant: bool = True,
    instruments: list[str] | None = None,
    classification: dict[str, float] | None = None,
    properties: dict[str, float] | None = None,
    event_time: str = "2026-08-14T02:00:00.000Z",
    time_created: str = "2026-08-14T02:03:00Z",
    skymap_bytes: bytes | None = None,
    skymap_encoding: str = "base64",
    external_coinc: dict | None = None,
) -> dict:
    """Build a notice matching the IGWN alert schema.

    Parameters
    ----------
    skymap_encoding : `str`
        ``"base64"`` mimics a GCN/JSON notice (skymap as a base64 string);
        ``"raw"`` mimics a SCIMMA/Avro notice (skymap as raw bytes), matching
        the one documented difference between the two transports.
    """
    default_classification = {"BNS": 0.9, "NSBH": 0.05, "BBH": 0.03, "Terrestrial": 0.02}
    default_properties = {"HasNS": 0.95, "HasRemnant": 0.9, "HasMassGap": 0.01}
    instruments = list(instruments if instruments is not None else ["H1", "L1"])
    classification = dict(classification if classification is not None else default_classification)
    properties = dict(properties if properties is not None else default_properties)

    event = {
        "time": event_time,
        "far": far,
        "significant": significant,
        "instruments": instruments,
        "group": group,
        "pipeline": pipeline,
        "search": search,
        "classification": classification,
        "properties": properties,
        "duration": None,
        "central_frequency": None,
    }
    if skymap_bytes is not None:
        if skymap_encoding == "base64":
            event["skymap"] = base64.b64encode(skymap_bytes).decode()
        else:
            event["skymap"] = skymap_bytes

    return {
        "alert_type": alert_type,
        "time_created": time_created,
        "superevent_id": superevent_id,
        "urls": {"gracedb": f"https://gracedb.ligo.org/superevents/{superevent_id}/view/"},
        "event": event,
        "external_coinc": external_coinc,
    }


@pytest.fixture
def skymap_bytes() -> bytes:
    """A small disc-shaped probability skymap, as raw FITS bytes."""
    return make_skymap_bytes()


@pytest.fixture
def notice(skymap_bytes) -> dict:
    """A minimal valid PRELIMINARY CBC notice with a base64-encoded skymap."""
    return make_notice(skymap_bytes=skymap_bytes, skymap_encoding="base64")


@pytest.fixture
def notice_avro(skymap_bytes) -> dict:
    """The same notice, with the skymap as raw bytes (SCIMMA/Avro-style)."""
    return make_notice(skymap_bytes=skymap_bytes, skymap_encoding="raw")


@pytest.fixture
def retraction_notice() -> dict:
    return {
        "alert_type": "RETRACTION",
        "time_created": "2026-08-14T03:00:00Z",
        "superevent_id": "S260814a",
        "urls": {"gracedb": "https://gracedb.ligo.org/superevents/S260814a/view/"},
        "event": None,
        "external_coinc": None,
    }


@pytest.fixture
def mock_notice(skymap_bytes) -> dict:
    """A mock/MDC notice, as GCN and SCIMMA both send once an hour."""
    return make_notice(superevent_id="M260814a", search="MDC", skymap_bytes=skymap_bytes)


@pytest.fixture(autouse=True)
def no_slack_token(monkeypatch):
    """Make it impossible for the suite to post to a real channel."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)


@pytest.fixture
def config(tmp_path):
    """A Config pointing entirely at a temporary directory."""
    from decam_chatterbox.config import Config

    cfg = Config()
    cfg.paths.work_dir = str(tmp_path / "work")
    # Tests must never reach the network.
    cfg.enrich.gracedb = False
    return cfg
