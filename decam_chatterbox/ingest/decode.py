"""Decode an LVK alert notice into a `~decam_chatterbox.models.Trigger`.

The notice is the JSON/Avro schema described in the IGWN public alerts user
guide (https://emfollow.docs.ligo.org/userguide/content.html). It reaches
decam-chatterbox through more than one transport -- Avro from SCIMMA, JSON
from a saved file, a hand-written replay fixture -- so decoding is defensive
about types while still insisting on the fields every notice carries.

The one documented difference between transports is the ``skymap`` field
itself: SCIMMA's Avro notices carry raw FITS bytes, GCN's JSON notices carry
the same bytes base64-encoded. Both are handled here.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time

from ..astro.skymap import (
    WORKING_NSIDE,
    credible_mask,
    geometry_from_mask,
    localization_from_probability,
    read_notice_skymap,
    save_notice_skymap,
)
from ..models import ExternalCoincidence, Trigger

__all__ = ["decode_notice", "load_record_file"]

logger = logging.getLogger(__name__)

#: Fields every notice carries, retractions included.
REQUIRED_FIELDS = ("alert_type", "superevent_id", "time_created")

#: Credible level the primary localization contour and the reported area are
#: computed at, matching the level the LVK circulars quote by convention.
DEFAULT_CREDIBLE_LEVEL = 0.9


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_time(raw: Any) -> Time | None:
    """Parse an LVK timestamp, tolerating format drift.

    A failure here must not drop the notice: the rest of it is still worth
    reporting even without a usable event time.
    """
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    try:
        # Time understands ISO-8601 with or without a trailing Z, but not both
        # a Z and a space separator, so normalize first.
        normalized = text.replace("Z", "").replace(" ", "T")
        return Time(normalized, format="isot", scale="utc")
    except Exception:
        pass
    try:
        return Time(text)
    except Exception as exc:
        logger.warning("Could not parse timestamp %r: %s", raw, exc)
        return None


def _skymap_bytes(value: Any) -> bytes | None:
    """Extract raw FITS bytes from a notice's ``skymap`` field.

    Avro (SCIMMA) notices already carry raw bytes; JSON (GCN) notices carry
    the same bytes base64-encoded as a string.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return base64.b64decode(value)
    raise TypeError(f"Unexpected type for a skymap field: {type(value)}")


def _decode_external_coinc(raw: Any) -> ExternalCoincidence | None:
    if not isinstance(raw, dict):
        return None
    return ExternalCoincidence(
        gcn_notice_id=raw.get("gcn_notice_id"),
        ivorn=raw.get("ivorn"),
        observatory=raw.get("observatory"),
        search=raw.get("search"),
        time_difference_s=_float_or_none(raw.get("time_difference")),
        time_coincidence_far_hz=_float_or_none(raw.get("time_coincidence_far")),
        time_sky_position_coincidence_far_hz=_float_or_none(raw.get("time_sky_position_coincidence_far")),
        has_combined_skymap=bool(raw.get("combined_skymap")),
    )


def _raw_without_binary_payloads(record: dict[str, Any]) -> dict[str, Any]:
    """The record, minus the (possibly megabyte-scale) skymap payloads."""
    raw = {k: v for k, v in record.items() if k != "event"}
    event = record.get("event")
    if isinstance(event, dict):
        raw["event"] = {k: v for k, v in event.items() if k != "skymap"}
    ext = record.get("external_coinc")
    if isinstance(ext, dict) and "combined_skymap" in ext:
        raw["external_coinc"] = {k: v for k, v in ext.items() if k != "combined_skymap"}
    return raw


def decode_notice(
    record: dict[str, Any],
    received_at: Time | None = None,
    skymap_dir: str | Path | None = None,
    working_nside: int = WORKING_NSIDE,
) -> Trigger:
    """Turn a raw LVK notice into a `~decam_chatterbox.models.Trigger`.

    Parameters
    ----------
    record : `dict`
        Decoded notice (see module docstring).
    received_at : `~astropy.time.Time`, optional
        When decam-chatterbox received it. Defaults to now.
    skymap_dir : `str` or `pathlib.Path`, optional
        Directory to save the notice's skymap into, when it carries one, as
        ``{superevent_id}_{alert_type}.fits``. Omit to skip saving it to disk
        (it is still decoded in memory).
    working_nside : `int`
        Resolution the skymap is degraded (or upgraded) to.

    Returns
    -------
    trigger : `Trigger`

    Raises
    ------
    KeyError
        If a required field is absent.
    ValueError
        If ``superevent_id`` is null or empty.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise KeyError(f"Notice is missing required field(s) {missing}")

    superevent_id = record["superevent_id"]
    if not superevent_id:
        raise ValueError("Notice has a null or empty 'superevent_id'; cannot identify the event")

    alert_type = str(record["alert_type"]).upper()
    urls = record.get("urls") or {}
    event = record.get("event") or {}

    trigger = Trigger(
        superevent_id=str(superevent_id),
        alert_type=alert_type,
        time_created=str(record["time_created"]),
        gracedb_url=urls.get("gracedb"),
        group=event.get("group"),
        pipeline=event.get("pipeline"),
        search=event.get("search"),
        significant=event.get("significant"),
        far_hz=_float_or_none(event.get("far")),
        instruments=[str(i) for i in (event.get("instruments") or [])],
        classification={k: float(v) for k, v in (event.get("classification") or {}).items()},
        properties={k: float(v) for k, v in (event.get("properties") or {}).items()},
        duration_s=_float_or_none(event.get("duration")),
        central_frequency_hz=_float_or_none(event.get("central_frequency")),
        external_coinc=_decode_external_coinc(record.get("external_coinc")),
        event_time=_parse_time(event.get("time")),
        received_at=Time.now() if received_at is None else received_at,
        raw=_raw_without_binary_payloads(record),
    )

    skymap_bytes = _skymap_bytes(event.get("skymap")) if event else None
    if skymap_bytes:
        _attach_localization(trigger, skymap_bytes, skymap_dir, working_nside)

    logger.info(
        "Decoded %s (%s): %s%s",
        trigger.superevent_id,
        trigger.alert_type,
        f"{trigger.geometry.area_deg2:.0f} deg^2" if trigger.geometry else "no localization",
        "" if trigger.is_real else " [not a real superevent]",
    )
    return trigger


def _attach_localization(
    trigger: Trigger,
    skymap_bytes: bytes,
    skymap_dir: str | Path | None,
    working_nside: int,
) -> None:
    """Decode a notice's skymap and populate the trigger's derived fields.

    Failure here is non-fatal: everything else about the notice is already
    on `trigger`, so a broken skymap loses the localization and says why,
    rather than losing the whole alert.
    """
    import tempfile

    try:
        if skymap_dir is not None:
            name = f"{trigger.superevent_id}_{trigger.alert_type.lower()}.fits"
            path = save_notice_skymap(skymap_bytes, Path(skymap_dir).expanduser() / name)
        else:
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
                f.write(skymap_bytes)
                path = Path(f.name)
        try:
            prob, meta = read_notice_skymap(path, working_nside=working_nside)
        finally:
            if skymap_dir is None:
                path.unlink(missing_ok=True)
    except Exception as exc:
        trigger.localization_error = f"could not decode the embedded skymap: {exc}"
        logger.warning("%s (%s)", trigger.localization_error, trigger.superevent_id)
        return

    nside = working_nside
    mask = credible_mask(prob, DEFAULT_CREDIBLE_LEVEL)
    trigger.localization = localization_from_probability(
        prob,
        provenance=f"{trigger.alert_type} notice for {trigger.superevent_id}, nside={nside}",
        credible_level=DEFAULT_CREDIBLE_LEVEL,
    )
    trigger.geometry = geometry_from_mask(mask, nside, prob)
    for key, attr in (("distmean", "distance_mean_mpc"), ("diststd", "distance_std_mpc")):
        value = meta.get(key)
        if value is not None and np.isfinite(value):
            setattr(trigger, attr, float(value))


def load_record_file(path: str | Path) -> dict[str, Any]:
    """Read a single notice record from disk.

    Supports plain JSON (as saved from a GCN Kafka message, or downloaded
    from the LVK sample-notice archive) and Avro (as saved from a SCIMMA
    message, via ``fastavro``, when installed).

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        File to read.

    Returns
    -------
    record : `dict`
    """
    path = Path(path).expanduser()
    if path.suffix in (".json", ".txt"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if not data:
                raise ValueError(f"{path} contains an empty list")
            return data[0]
        return data

    if path.suffix == ".avro":
        try:
            import fastavro
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Reading .avro records requires fastavro (pip install fastavro)") from exc
        with open(path, "rb") as f:
            records = list(fastavro.reader(f))
        if not records:
            raise ValueError(f"{path} contains no Avro records")
        return records[0]

    raise ValueError(f"Unsupported record file type: {path.suffix} ({path})")
