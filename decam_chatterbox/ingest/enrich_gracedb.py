"""Best-effort enrichment of a GW trigger from GraceDB: the chirp mass.

The notice schema carries no field for chirp mass; GraceDB publishes a binned
estimate as ``mchirp_source.json`` (preliminary) or ``mchirp_source_PE.json``
(later updates) for significant CBC events. Everything else -- distance, FAR,
classification, properties, the skymap -- is already on `Trigger` from the
notice itself, so unlike chatterbox's Rubin-side enrichment (which recovers
the *entire* localization and event record from GraceDB), this is a small,
strictly optional supplement. Failure never blocks the post.
"""

import json
import logging
import re
import time

import numpy as np

from ..config import EnrichConfig
from ..models import GwEnrichment, Trigger

__all__ = ["is_superevent_id", "enrich_chirp_mass"]

logger = logging.getLogger(__name__)

#: GraceDB superevent ids: S for real candidates, M/T for mock and test.
_SUPEREVENT_RE = re.compile(r"^(?:S|M|T)\d{6}[a-z]+$")

#: Chirp-mass files to try, most refined first.
_MCHIRP_FILES = ("mchirp_source_PE.json", "mchirp_source.json")

#: The threshold quoted alongside the median: a common early cut for BBH
#: follow-up interest.
_CHIRP_MASS_THRESHOLD_MSUN = 44.0


def is_superevent_id(superevent_id: str) -> bool:
    """True if `superevent_id` looks like a GraceDB superevent id."""
    return bool(_SUPEREVENT_RE.match(superevent_id or ""))


def _client(config: EnrichConfig):
    """Construct an unauthenticated GraceDB client for public data."""
    from ..deps import require

    GraceDb = require("ligo.gracedb.rest").GraceDb
    # Public superevents need no credentials; asking for them would fail on a
    # host with no certificate. fail_if_noauth=False keeps it anonymous.
    return GraceDb(service_url=config.gracedb_service_url, fail_if_noauth=False)


def _read_chirp_mass(payload: bytes, enrichment: GwEnrichment) -> bool:
    """Summarize a ``mchirp_source*.json`` payload onto `enrichment`.

    GraceDB serves this as ``{"bin_edges": [...], "probabilities": [...]}``
    with one more edge than probability.

    Returns
    -------
    ok : `bool`
        True when a value was found.
    """
    data = json.loads(payload)
    edges = np.asarray(data["bin_edges"], dtype=float)
    probs = np.asarray(data["probabilities"], dtype=float)
    if edges.size != probs.size + 1:
        logger.debug("mchirp_source.json has an unexpected shape for %s", enrichment.superevent_id)
        return False
    total = probs.sum()
    if total <= 0:
        return False
    probs = probs / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    # searchsorted rather than interpolation: the distribution is often a
    # single populated bin, which makes the CDF flat and interpolation
    # ill-defined.
    median_bin = min(int(np.searchsorted(np.cumsum(probs), 0.5)), centers.size - 1)
    enrichment.chirp_mass_median = float(centers[median_bin])
    enrichment.chirp_mass_prob_above_44 = float(probs[centers > _CHIRP_MASS_THRESHOLD_MSUN].sum())
    return True


def enrich_chirp_mass(trigger: Trigger, config: EnrichConfig) -> GwEnrichment | None:
    """Fetch the binned chirp-mass estimate for a trigger, if available.

    Parameters
    ----------
    trigger : `Trigger`
        Trigger to enrich. Not mutated; the caller attaches the result.
    config : `EnrichConfig`
        Enrichment settings, including the time budget.

    Returns
    -------
    enrichment : `GwEnrichment` or None
        None when the source is not a superevent id, the event is a
        retraction or has no CBC classification, or enrichment is disabled.
    """
    if not config.gracedb:
        return None
    if trigger.is_retraction or trigger.is_burst:
        return None
    if not is_superevent_id(trigger.superevent_id):
        logger.debug("%s is not a superevent id; skipping GraceDB enrichment", trigger.superevent_id)
        return None

    deadline = time.monotonic() + config.timeout_s
    enrichment = GwEnrichment(superevent_id=trigger.superevent_id)

    try:
        client = _client(config)
    except Exception as exc:
        enrichment.error = f"GraceDB client unavailable: {exc}"
        logger.info("%s", enrichment.error)
        return enrichment

    for name in _MCHIRP_FILES:
        if time.monotonic() > deadline:
            enrichment.error = "ran out of time before fetching the chirp mass"
            break
        try:
            payload = client.files(trigger.superevent_id, name).read()
        except Exception:
            continue
        try:
            if _read_chirp_mass(payload, enrichment):
                return enrichment
        except Exception as exc:
            logger.debug("Could not parse %s for %s: %s", name, trigger.superevent_id, exc)

    if enrichment.error is None:
        enrichment.error = "no chirp-mass estimate published for this event"
    return enrichment
