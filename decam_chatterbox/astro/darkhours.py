"""Accessible dark hours per HEALPix pixel, for CTIO.

For each pixel, count the hours during the night for which DECam could
observe it: airmass below a limit (default 2, i.e. altitude above 30 degrees)
while the Sun is below an altitude limit (default -12 degrees).

The altitude of every pixel at every time step is evaluated with the
standard spherical-astronomy hour-angle formula rather than a full
`~astropy.coordinates.AltAz` transform: apparent sidereal time is computed
once per time step with astropy (so it is exact, including nutation), and
every pixel at that time step is then evaluated with plain ``numpy``
trigonometry rather than astropy's per-coordinate machinery. This ignores
refraction and diurnal aberration, both far smaller than the 5-minute time
sampling already introduces; ``tests/test_almanac.py`` checks it against a
full AltAz transform at a handful of pixel/time pairs.
"""

import logging
from dataclasses import dataclass

import healpy as hp
import numpy as np

from .almanac import NightEvents

__all__ = ["DarkHoursMap", "dark_hours_map", "airmass_to_altitude_deg", "fast_altitude_deg"]

logger = logging.getLogger(__name__)


def airmass_to_altitude_deg(airmass: float) -> float:
    """Minimum altitude in degrees corresponding to a maximum airmass.

    Uses the plane-parallel relation ``X = 1 / sin(alt)``.

    Parameters
    ----------
    airmass : `float`
        Airmass limit, must be >= 1.

    Returns
    -------
    altitude : `float`
        Altitude in degrees.
    """
    if airmass < 1.0:
        raise ValueError(f"Airmass must be >= 1, got {airmass}")
    return float(np.degrees(np.arcsin(1.0 / airmass)))


def fast_altitude_deg(
    ra_deg: np.ndarray, dec_deg: np.ndarray, lst_deg: np.ndarray, lat_deg: float
) -> np.ndarray:
    """Altitude in degrees from RA/Dec, local sidereal time and latitude.

    Parameters
    ----------
    ra_deg, dec_deg : array-like
        Coordinates in degrees, broadcastable against `lst_deg`.
    lst_deg : array-like
        Local apparent sidereal time in degrees.
    lat_deg : `float`
        Observer's geodetic latitude in degrees.

    Returns
    -------
    altitude : `numpy.ndarray`
        Degrees.
    """
    hour_angle = np.radians(np.asarray(lst_deg) - np.asarray(ra_deg))
    dec = np.radians(np.asarray(dec_deg))
    lat = np.radians(lat_deg)
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


@dataclass
class DarkHoursMap:
    """Hours of accessible dark time per pixel for one night at CTIO.

    Attributes
    ----------
    hours : `numpy.ndarray`
        Accessible hours per pixel, RING ordering.
    nside : `int`
        Map resolution.
    events : `NightEvents`
        The night this map describes.
    airmass_limit : `float`
        Airmass limit applied.
    sun_alt_limit_deg : `float`
        Sun altitude limit defining the window.
    altitude_limit_deg : `float`
        Altitude limit derived from `airmass_limit`.
    window_hours : `float`
        Length of the Sun-altitude window, i.e. the maximum any pixel can have.
    moon_avoidance_deg : `float`
        Moon avoidance radius applied, 0 when disabled.
    """

    hours: np.ndarray
    nside: int
    events: NightEvents
    airmass_limit: float
    sun_alt_limit_deg: float
    altitude_limit_deg: float
    window_hours: float
    moon_avoidance_deg: float = 0.0

    def stats_in_region(self, mask: np.ndarray) -> dict[str, float]:
        """Summarize accessible hours inside a credible region.

        Parameters
        ----------
        mask : `numpy.ndarray`
            Boolean mask in RING ordering. Resampled with
            ``healpy.ud_grade`` if its resolution differs from this map's.

        Returns
        -------
        stats : `dict`
            ``max_hours``, ``mean_hours``, ``median_hours`` and
            ``fraction_accessible`` (fraction of the region reaching airmass
            below the limit at any point in the window).
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.size != self.hours.size:
            mask = hp.ud_grade(mask.astype(float), self.nside, order_in="RING", order_out="RING") > 0
        if not mask.any():
            return {
                "max_hours": float("nan"),
                "mean_hours": float("nan"),
                "median_hours": float("nan"),
                "fraction_accessible": 0.0,
            }
        values = self.hours[mask]
        return {
            "max_hours": float(values.max()),
            "mean_hours": float(values.mean()),
            "median_hours": float(np.median(values)),
            "fraction_accessible": float(np.count_nonzero(values) / values.size),
        }

    def weighted_stats(self, prob_map: np.ndarray) -> dict[str, float]:
        """Accessible hours weighted by localization probability.

        Parameters
        ----------
        prob_map : `numpy.ndarray`
            Normalized weight per pixel in RING ordering, resampled if needed.

        Returns
        -------
        stats : `dict`
            ``weighted_mean_hours`` and ``weight_accessible``, the fraction of
            total probability lying on pixels with any accessible dark time.
        """
        prob_map = np.asarray(prob_map, dtype=float)
        if prob_map.size != self.hours.size:
            prob_map = hp.ud_grade(prob_map, self.nside, order_in="RING", order_out="RING", power=-2)
        total = prob_map.sum()
        if total <= 0:
            return {"weighted_mean_hours": float("nan"), "weight_accessible": 0.0}
        weights = prob_map / total
        return {
            "weighted_mean_hours": float(np.sum(weights * self.hours)),
            "weight_accessible": float(np.sum(weights[self.hours > 0])),
        }


def dark_hours_map(
    events: NightEvents,
    nside: int = 64,
    step_minutes: float = 5.0,
    airmass_limit: float = 2.0,
    sun_alt_limit_deg: float = -12.0,
    moon_avoidance_deg: float = 0.0,
) -> DarkHoursMap:
    """Compute accessible dark hours per pixel for one night at CTIO.

    Parameters
    ----------
    events : `NightEvents`
        Night to evaluate, from `decam_chatterbox.astro.almanac.night_events`.
    nside : `int`
        Output map resolution.
    step_minutes : `float`
        Time sampling. 5 minutes is accurate to a few minutes per pixel and
        takes under a couple of seconds at nside 128.
    airmass_limit : `float`
        Maximum airmass counted as accessible.
    sun_alt_limit_deg : `float`
        Sun altitude bounding the window; -12 or -18.
    moon_avoidance_deg : `float`
        When positive, samples closer than this to the Moon are not counted.

    Returns
    -------
    dark_hours : `DarkHoursMap`

    Notes
    -----
    Each sample contributes the step length, but the total is clipped to the
    window length: a naive count of samples over-reports by up to one step
    because both endpoints can satisfy the condition.
    """
    from .almanac import ctio_observer

    observer = ctio_observer()
    start_mjd, end_mjd = events.observing_window(sun_alt_limit_deg)
    window_hours = (end_mjd - start_mjd) * 24.0

    step_days = step_minutes / (60.0 * 24.0)
    from astropy.time import Time

    mjds = np.arange(start_mjd, end_mjd, step_days)
    if mjds.size == 0:
        raise ValueError(f"Observing window for the night of {events.day_obs} is empty")
    times = Time(mjds, format="mjd", scale="utc", location=observer.location)

    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    altitude_limit = airmass_to_altitude_deg(airmass_limit)
    lat_deg = float(observer.location.lat.deg)

    # (npix, 1) broadcast against (1, ntime) evaluates the whole grid in one
    # vectorized call; sidereal time itself is the only per-time astropy call.
    lst_deg = times.sidereal_time("apparent").deg
    alt = fast_altitude_deg(ra[:, None], dec[:, None], lst_deg[None, :], lat_deg)
    accessible = alt > altitude_limit

    if moon_avoidance_deg > 0:
        from astropy.coordinates import angular_separation

        moon_sep = np.degrees(
            angular_separation(
                np.radians(ra[:, None]),
                np.radians(dec[:, None]),
                np.radians(events.moon_ra_deg),
                np.radians(events.moon_dec_deg),
            )
        )
        # The Moon moves slowly enough over one night that a single position
        # is adequate for an avoidance annulus at this resolution.
        accessible &= moon_sep > moon_avoidance_deg

    hours = accessible.sum(axis=1) * (step_minutes / 60.0)
    np.clip(hours, 0.0, window_hours, out=hours)

    logger.debug(
        "Dark-hours map: nside=%d ntime=%d window=%.2f h max=%.2f h",
        nside,
        times.size,
        window_hours,
        hours.max(),
    )

    return DarkHoursMap(
        hours=hours,
        nside=nside,
        events=events,
        airmass_limit=airmass_limit,
        sun_alt_limit_deg=sun_alt_limit_deg,
        altitude_limit_deg=altitude_limit,
        window_hours=window_hours,
        moon_avoidance_deg=moon_avoidance_deg,
    )
