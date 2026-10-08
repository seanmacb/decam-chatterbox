"""Normalized in-memory representation of a gravitational-wave notice.

Unlike chatterbox's Rubin-side `Trigger`, which decoded a ToO producer's
*derived* record (a binary 70% reward map, with the real probability,
distance, FAR and classification stripped out and requiring a GraceDB
round-trip to recover), decam-chatterbox decodes the LVK alert notice itself.
That notice already carries the real probability skymap and every field this
tool reports, so there is no "area vs probability" distinction to track here:
every `Localization` is a genuine probability map.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.time import Time

__all__ = [
    "Localization",
    "Geometry",
    "ExternalCoincidence",
    "GwEnrichment",
    "Trigger",
]


@dataclass
class Localization:
    """A sky localization ready for plotting and observability accounting.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Normalized probability per pixel in **RING** ordering, summing to 1.
    nside : `int`
        HEALPix resolution of `prob_map`.
    credible_level : `float`
        Credible level the primary contour is drawn at, e.g. 0.9.
    provenance : `str`
        Human-readable description of where the map came from, shown in
        Slack.
    """

    prob_map: np.ndarray
    nside: int
    credible_level: float
    provenance: str

    @property
    def npix(self) -> int:
        """Number of pixels in the map."""
        return int(self.prob_map.size)


@dataclass
class Geometry:
    """Sky geometry of a credible region. All angles are in degrees.

    ``peak_ra_deg``/``peak_dec_deg`` is the single most probable position in
    the localization (see `decam_chatterbox.astro.skymap.peak_position`), not
    the mean of the region: a banana-shaped or multi-lobed map has a mean that
    can sit on sky the localization gives almost no probability to.
    """

    area_deg2: float
    dec_min_deg: float
    dec_max_deg: float
    peak_ra_deg: float
    peak_dec_deg: float
    gal_b_abs_min_deg: float
    gal_b_abs_max_deg: float
    n_pixels: int
    pixel_area_deg2: float


@dataclass
class ExternalCoincidence:
    """A coincident sub-threshold alert from another facility (e.g. Fermi GBM).

    Populated only when the notice's ``external_coinc`` field is non-null.
    """

    gcn_notice_id: int | None = None
    ivorn: str | None = None
    observatory: str | None = None
    search: str | None = None
    time_difference_s: float | None = None
    time_coincidence_far_hz: float | None = None
    time_sky_position_coincidence_far_hz: float | None = None
    #: True when a combined GW+external skymap was attached. The combined map
    #: itself is not decoded: the GW-only localization is what the rest of
    #: this tool reports.
    has_combined_skymap: bool = False


@dataclass
class GwEnrichment:
    """Optional supplement fetched from GraceDB: the binned chirp mass.

    Everything else this tool reports -- distance, FAR, classification,
    properties, the skymap itself -- is already carried by the notice.
    Chirp mass is the one number the notice schema has no field for. This is
    best-effort and time-bounded; failure never blocks the post.
    """

    superevent_id: str
    #: Median source-frame chirp mass, from mchirp_source(_PE).json.
    chirp_mass_median: float | None = None
    #: P(chirp mass > 44 Msun), a common early cut for BBH follow-up interest.
    chirp_mass_prob_above_44: float | None = None
    #: Populated when enrichment was attempted and failed, for the Slack note.
    error: str | None = None


@dataclass
class Trigger:
    """A single LVK alert notice, decoded and (optionally) enriched."""

    # --- straight from the notice ---
    superevent_id: str
    alert_type: str  # EARLYWARNING | PRELIMINARY | INITIAL | UPDATE | RETRACTION
    time_created: str
    gracedb_url: str | None
    group: str | None = None  # CBC | Burst
    pipeline: str | None = None
    search: str | None = None
    significant: bool | None = None
    far_hz: float | None = None
    instruments: list[str] = field(default_factory=list)
    #: p(BNS), p(NSBH), p(BBH), p(Terrestrial). Empty for Burst events, which
    #: carry no classification.
    classification: dict[str, float] = field(default_factory=dict)
    #: HasNS, HasRemnant, HasMassGap, HasSSM. Empty for Burst events.
    properties: dict[str, float] = field(default_factory=dict)
    #: Burst events only; both None for CBC.
    duration_s: float | None = None
    central_frequency_hz: float | None = None
    external_coinc: ExternalCoincidence | None = None

    # --- derived from the embedded skymap, when present ---
    localization: Localization | None = None
    geometry: Geometry | None = None
    distance_mean_mpc: float | None = None
    distance_std_mpc: float | None = None
    #: Set when a skymap was present but could not be decoded, so the post can
    #: say so rather than silently omitting the localization.
    localization_error: str | None = None

    # --- timing ---
    event_time: Time | None = None
    received_at: Time | None = None

    enrichment: GwEnrichment | None = None
    #: The decoded record as received (skymap bytes excluded), for debugging.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_retraction(self) -> bool:
        return self.alert_type == "RETRACTION"

    @property
    def is_burst(self) -> bool:
        return (self.group or "").upper() == "BURST"

    @property
    def is_real(self) -> bool:
        """True for a real superevent, as opposed to a mock or test one.

        GraceDB superevent ids are prefixed ``S`` for real candidates, ``M``
        for mock/MDC events (including the sample alert GCN and SCIMMA both
        send once an hour), and ``T`` for internal test events. Checking the
        prefix is the documented way to tell them apart: the ``search``
        field is a second signal (``"MDC"``) but is absent from retractions.
        """
        return self.superevent_id[:1] == "S"

    @property
    def far_per_year(self) -> float | None:
        """False alarm rate expressed as events per year."""
        if self.far_hz is None:
            return None
        return self.far_hz * 3.15576e7

    @property
    def inverse_far_years(self) -> float | None:
        """One over the false alarm rate, in years -- the usual way to quote
        it.
        """
        per_year = self.far_per_year
        if per_year is None or per_year <= 0:
            return None
        return 1.0 / per_year

    @property
    def most_likely_class(self) -> tuple[str, float] | None:
        """Highest-probability CBC classification, or None if unavailable."""
        if not self.classification:
            return None
        name, value = max(self.classification.items(), key=lambda kv: kv[1])
        return name, float(value)

    @property
    def notice_latency_s(self) -> float | None:
        """Seconds between the physical event and this notice being created.

        This is the real LVK pipeline latency (the LVK user guide quotes
        1-10 minutes for a Preliminary notice, 4-24 hours for Initial,
        1-7 days for Update), unlike chatterbox's Rubin-side `Trigger`, which
        had to infer a proxy latency because its record never carried the
        alert's own creation time.
        """
        if self.event_time is None:
            return None
        try:
            from astropy.time import Time

            created = Time(str(self.time_created).replace("Z", ""), format="isot", scale="utc")
        except Exception:
            return None
        return created.unix - self.event_time.unix

    @property
    def age_s(self) -> float | None:
        """Seconds between the physical event and decam-chatterbox receiving
        it.
        """
        if self.event_time is None or self.received_at is None:
            return None
        return self.received_at.unix - self.event_time.unix
