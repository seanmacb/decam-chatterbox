"""When a localization is up at CTIO, not just for how long.

`decam_chatterbox.astro.darkhours` says how many hours each pixel is
accessible. An observer deciding whether to act needs the other half: *when*
-- from which time to which, when the field is at its best, and for an
extended region how much of it is up at a given moment.

The definition of "visible" is the dark-hours map's own: altitude above the
limit that the airmass limit implies, while the Sun is below its limit, and
outside the Moon avoidance radius when one is set. The altitude is computed
the same way too -- apparent sidereal time from astropy once per time step,
then `~decam_chatterbox.astro.darkhours.fast_altitude_deg` on the same time
grid -- so the times here and the hours there cannot disagree.
"""

import logging
from dataclasses import dataclass

import healpy as hp
import numpy as np
from astropy.time import Time

from ..models import Localization
from .almanac import NightEvents
from .darkhours import DarkHoursMap, fast_altitude_deg
from .skymap import localization_region

__all__ = ["Window", "FieldVisibility", "field_visibility", "MAJORITY"]

logger = logging.getLogger(__name__)

#: Share of a localization's probability that has to be up for it to count
#: as "most of it".
MAJORITY = 0.5


@dataclass
class Window:
    """A stretch of one night.

    Attributes
    ----------
    start, end : `astropy.time.Time`
        UTC bounds.
    hours : `float`
        Length.
    starts_at_dusk, ends_at_dawn : `bool`
        True when the window is bounded by the observing window rather than
        by the field rising or setting: the field was already up when the
        night began, or is still up when it ends. Both is a field that is up
        all night. An observer needs the difference, since "until 09:24"
        means something else when that is simply the end of the night.
    """

    start: Time
    end: Time
    hours: float
    starts_at_dusk: bool = False
    ends_at_dawn: bool = False

    @property
    def whole_night(self) -> bool:
        """True when the field is up for the entire observing window."""
        return self.starts_at_dusk and self.ends_at_dawn


@dataclass
class FieldVisibility:
    """When a localization can be observed from CTIO during one night.

    Every window is within the night's observing window (the Sun below
    `sun_alt_limit_deg`) and above `altitude_limit_deg`, and is None when it
    does not occur at all.

    Attributes
    ----------
    peak : `Window` or None
        When the localization's most probable position is up.
    peak_highest_time : `astropy.time.Time` or None
        When the most probable position is highest during the observing
        window.
    peak_highest_altitude_deg : `float`
        Its altitude then. NaN when there is no most probable position.
    any_part : `Window` or None
        From the first moment any part of the credible region is up to the
        last.
    most : `Window` or None
        The longest stretch with at least `MAJORITY` of the region's
        probability up.
    best_time : `astropy.time.Time` or None
        When the largest share of the region's probability is up.
    best_fraction : `float`
        That share, 0-1.
    window_hours : `float`
        Length of the observing window, the most any window can have.
    airmass_limit, altitude_limit_deg, sun_alt_limit_deg : `float`
        The limits applied, copied from the dark-hours map.
    step_minutes : `float`
        Time sampling, which is the resolution of every boundary here.
    """

    peak: Window | None
    peak_highest_time: Time | None
    peak_highest_altitude_deg: float
    any_part: Window | None
    most: Window | None
    best_time: Time | None
    best_fraction: float
    window_hours: float
    airmass_limit: float
    altitude_limit_deg: float
    sun_alt_limit_deg: float
    step_minutes: float


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """Bounds ``(first, last)`` of the longest run of True, inclusive."""
    best: tuple[int, int] | None = None
    start = None
    for i, flag in enumerate(np.append(flags, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or (i - 1 - start) > (best[1] - best[0]):
                best = (start, i - 1)
            start = None
    return best


def _window(times: np.ndarray, bounds: tuple[int, int] | None, step_days: float, end_mjd: float):
    """A `Window` from sample indices, or None; the last sample owns a step."""
    if bounds is None:
        return None
    first, last = bounds
    start = float(times[first])
    end = min(float(times[last]) + step_days, end_mjd)
    return Window(
        start=Time(start, format="mjd", scale="utc"),
        end=Time(end, format="mjd", scale="utc"),
        hours=(end - start) * 24.0,
        starts_at_dusk=first == 0,
        ends_at_dawn=float(times[last]) + step_days >= end_mjd,
    )


def field_visibility(
    events: NightEvents,
    localization: Localization,
    dark_hours: DarkHoursMap,
    peak_ra_deg: float,
    peak_dec_deg: float,
    step_minutes: float = 5.0,
) -> FieldVisibility:
    """When a localization is up at CTIO during a night.

    Parameters
    ----------
    events : `NightEvents`
        The night.
    localization : `Localization`
        Whose credible region, weighted by its probability, is the field.
    dark_hours : `DarkHoursMap`
        Supplies the airmass, Sun and Moon limits so this agrees with the map.
    peak_ra_deg, peak_dec_deg : `float`
        The most probable position, for the single-point window.
    step_minutes : `float`
        Time sampling. Every boundary is good to about this.

    Returns
    -------
    visibility : `FieldVisibility`

    Notes
    -----
    The region is the credible region, not every pixel with probability: a
    real skymap has faint probability almost everywhere, which would make
    "any part of the localization is up" true whenever any of the sky is.
    Inside the region, each pixel contributes in proportion to its
    probability.
    """
    from astropy.coordinates import angular_separation

    from .almanac import ctio_observer

    observer = ctio_observer()
    start_mjd, end_mjd = events.observing_window(dark_hours.sun_alt_limit_deg)
    step_days = step_minutes / (60.0 * 24.0)
    times = np.arange(start_mjd, end_mjd, step_days)
    if times.size == 0:
        raise ValueError(f"Observing window for the night of {events.day_obs} is empty")
    lst_deg = Time(times, format="mjd", scale="utc", location=observer.location).sidereal_time("apparent").deg
    lat_deg = float(observer.location.lat.deg)
    limit = dark_hours.altitude_limit_deg

    def moon_clear(ra, dec) -> np.ndarray:
        """True where a position is outside the Moon avoidance radius."""
        if dark_hours.moon_avoidance_deg <= 0:
            return np.ones(np.shape(ra), dtype=bool)
        separation = np.degrees(
            angular_separation(
                np.radians(ra),
                np.radians(dec),
                np.radians(events.moon_ra_deg),
                np.radians(events.moon_dec_deg),
            )
        )
        return separation > dark_hours.moon_avoidance_deg

    # --- the most probable position
    peak_window = None
    highest_time = None
    highest_altitude = float("nan")
    if np.isfinite(peak_ra_deg) and np.isfinite(peak_dec_deg):
        alt = fast_altitude_deg(peak_ra_deg, peak_dec_deg, lst_deg, lat_deg)
        up = (alt > limit) & moon_clear(peak_ra_deg, peak_dec_deg)
        peak_window = _window(times, _longest_run(up), step_days, end_mjd)
        highest = int(np.argmax(alt))
        highest_time = Time(float(times[highest]), format="mjd", scale="utc")
        highest_altitude = float(alt[highest])

    # --- the credible region, probability-weighted
    nside = dark_hours.nside
    region = localization_region(localization, nside)
    weight = np.asarray(localization.prob_map, dtype=float)
    if weight.size != hp.nside2npix(nside):
        weight = hp.ud_grade(weight, nside, order_in="RING", order_out="RING", power=-2)
    pixels = np.flatnonzero(region & (weight > 0))

    any_window = most_window = best_time = None
    best_fraction = 0.0
    if pixels.size:
        ra, dec = hp.pix2ang(nside, pixels, lonlat=True)
        alt = fast_altitude_deg(ra[:, None], dec[:, None], lst_deg[None, :], lat_deg)
        up = (alt > limit) & moon_clear(ra, dec)[:, None]
        w = weight[pixels]
        # Clipped because the weighted sum can land a rounding error above 1.
        fraction = np.clip((w[:, None] * up).sum(axis=0) / w.sum(), 0.0, 1.0)

        any_window = _window(times, _longest_run(fraction > 0), step_days, end_mjd)
        most_window = _window(times, _longest_run(fraction >= MAJORITY), step_days, end_mjd)
        if fraction.max() > 0:
            best = int(np.argmax(fraction))
            best_time = Time(float(times[best]), format="mjd", scale="utc")
            best_fraction = float(fraction[best])

    logger.debug(
        "Visibility for %s: peak %s h, any part %s h, most %s h, best %.0f%%",
        events.day_obs,
        None if peak_window is None else round(peak_window.hours, 2),
        None if any_window is None else round(any_window.hours, 2),
        None if most_window is None else round(most_window.hours, 2),
        100 * best_fraction,
    )
    return FieldVisibility(
        peak=peak_window,
        peak_highest_time=highest_time,
        peak_highest_altitude_deg=highest_altitude,
        any_part=any_window,
        most=most_window,
        best_time=best_time,
        best_fraction=best_fraction,
        window_hours=(end_mjd - start_mjd) * 24.0,
        airmass_limit=dark_hours.airmass_limit,
        altitude_limit_deg=limit,
        sun_alt_limit_deg=dark_hours.sun_alt_limit_deg,
        step_minutes=step_minutes,
    )
