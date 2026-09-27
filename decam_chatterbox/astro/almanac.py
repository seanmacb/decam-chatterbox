"""Sun and Moon statistics for the CTIO night a notice belongs to.

Built entirely on ``astropy`` and ``astroplan`` -- unlike chatterbox's Rubin
almanac, which reads a scheduler-specific precomputed table via
``rubin_scheduler.site_models.Almanac``, this has no upstream site-model
dependency and works at any site `~astropy.coordinates.EarthLocation.of_site`
knows about. CTIO is hardcoded here since DECam is this tool's one site, but
nothing below assumes that beyond the module-level constant.
"""

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
from astropy.time import Time

__all__ = [
    "CTIO_TZ",
    "ctio_observer",
    "NightEvents",
    "night_events",
    "moon_separation_deg",
    "format_time",
]

logger = logging.getLogger(__name__)

#: CTIO and Cerro Pachon share Chile's mainland time zone.
CTIO_TZ = ZoneInfo("Chile/Continental")

_observer = None


def ctio_observer():
    """The `astroplan.Observer` for CTIO, built once and cached.

    Returns
    -------
    observer : `astroplan.Observer`
    """
    global _observer
    if _observer is None:
        from astroplan import Observer
        from astropy.coordinates import EarthLocation

        _observer = Observer(location=EarthLocation.of_site("ctio"), name="CTIO", timezone=CTIO_TZ)
    return _observer


def _time_or_none(t: Time | None, window: tuple[Time, Time] | None = None) -> Time | None:
    """Pass through a Time, or None if it is invalid or outside a window.

    astroplan always returns *some* rise/set time (the Moon rises and sets
    roughly once a day), so "does not occur tonight" has to be detected by
    checking whether the returned instant actually falls inside the night,
    rather than by a NaN the way chatterbox's Rubin almanac reports it.
    """
    if t is None or not np.isfinite(t.jd):
        return None
    if window is not None and not (window[0].jd <= t.jd <= window[1].jd):
        return None
    return t


def format_time(t: Time | None, tz: ZoneInfo | None = None) -> str:
    """Format a time for Slack, or ``"--"`` when the event does not occur."""
    if t is None:
        return "--"
    if tz is None:
        return t.utc.strftime("%Y-%m-%d %H:%M UTC")
    return t.to_datetime(timezone=tz).strftime("%H:%M")


@dataclass
class NightEvents:
    """Sun and Moon events for a single observing night at CTIO.

    All times are UTC. ``None`` means the event does not occur during the
    night, which happens routinely for moonrise and moonset.
    """

    day_obs: str
    sunset: Time | None
    sun_n12_setting: Time | None
    sun_n18_setting: Time | None
    sun_n18_rising: Time | None
    sun_n12_rising: Time | None
    sunrise: Time | None
    moonrise: Time | None
    moonset: Time | None
    moon_illumination: float
    moon_ra_deg: float
    moon_dec_deg: float

    @property
    def night_length_hours(self) -> float:
        """Hours between the -12 degree evening and morning crossings."""
        if self.sun_n12_setting is None or self.sun_n12_rising is None:
            return float("nan")
        return float((self.sun_n12_rising.mjd - self.sun_n12_setting.mjd) * 24.0)

    @property
    def dark_length_hours(self) -> float:
        """Hours between the -18 degree evening and morning crossings."""
        if self.sun_n18_setting is None or self.sun_n18_rising is None:
            return float("nan")
        return float((self.sun_n18_rising.mjd - self.sun_n18_setting.mjd) * 24.0)

    def observing_window(self, sun_alt_limit_deg: float = -12.0) -> tuple[float, float]:
        """MJD bounds of the window for a given Sun altitude limit.

        Parameters
        ----------
        sun_alt_limit_deg : `float`
            Either -12 or -18; other values raise.

        Returns
        -------
        start, end : `float`
            MJD bounds.
        """
        if sun_alt_limit_deg == -12.0:
            start, end = self.sun_n12_setting, self.sun_n12_rising
        elif sun_alt_limit_deg == -18.0:
            start, end = self.sun_n18_setting, self.sun_n18_rising
        else:
            raise ValueError(f"Unsupported Sun altitude limit {sun_alt_limit_deg}; use -12 or -18")
        if start is None or end is None:
            raise ValueError(f"No {sun_alt_limit_deg} degree crossings for the night of {self.day_obs}")
        return float(start.mjd), float(end.mjd)


def night_events(when: Time | None = None, prefer_next_night: bool = True) -> NightEvents:
    """Sun and Moon events for the CTIO observing night relevant to a time.

    Parameters
    ----------
    when : `~astropy.time.Time`, optional
        A time during (or on the day of) the night of interest. Defaults to
        now.
    prefer_next_night : `bool`
        When `when` falls after the night's -12 degree morning crossing --
        that is, the observable part of that night is already over --
        advance to the following night. This is almost always what a report
        about a fresh alert should show, since a notice arriving during local
        daytime is followed up that evening, not retroactively.

    Returns
    -------
    events : `NightEvents`
    """
    observer = ctio_observer()
    when = Time.now() if when is None else when

    sunset = observer.sun_set_time(when, which="previous")
    sun_n12_rising_probe = observer.twilight_morning_nautical(sunset, which="next")
    if prefer_next_night and when.jd > sun_n12_rising_probe.jd:
        logger.info(
            "Time %s is after this night's -12 deg morning crossing; reporting the following night",
            when.utc.iso,
        )
        sunset = observer.sun_set_time(when, which="next")

    sun_n12_setting = observer.twilight_evening_nautical(sunset, which="next")
    sun_n18_setting = observer.twilight_evening_astronomical(sunset, which="next")
    sun_n18_rising = observer.twilight_morning_astronomical(sunset, which="next")
    sun_n12_rising = observer.twilight_morning_nautical(sunset, which="next")
    sunrise = observer.sun_rise_time(sunset, which="next")

    window = (sunset, sunrise)
    moonrise = _time_or_none(observer.moon_rise_time(sunset, which="next"), window)
    moonset = _time_or_none(observer.moon_set_time(sunset, which="next"), window)

    # A single Moon position at nautical dusk, as chatterbox's Rubin almanac
    # does: the Moon moves slowly enough over one night that this is adequate
    # at the resolution the dark-hours map and the separation line need.
    from astropy.coordinates import get_body

    moon = get_body("moon", sun_n12_setting, observer.location)
    illumination = float(observer.moon_illumination(sun_n12_setting))

    return NightEvents(
        day_obs=sunset.to_datetime(timezone=CTIO_TZ).strftime("%Y-%m-%d"),
        sunset=sunset,
        sun_n12_setting=sun_n12_setting,
        sun_n18_setting=sun_n18_setting,
        sun_n18_rising=sun_n18_rising,
        sun_n12_rising=sun_n12_rising,
        sunrise=sunrise,
        moonrise=moonrise,
        moonset=moonset,
        moon_illumination=illumination,
        moon_ra_deg=float(moon.ra.deg),
        moon_dec_deg=float(moon.dec.deg),
    )


def moon_separation_deg(ra_deg: float, dec_deg: float, events: NightEvents) -> float:
    """Angular separation between a coordinate and the Moon at nautical dusk.

    Parameters
    ----------
    ra_deg, dec_deg : `float`
        Target coordinates in degrees.
    events : `NightEvents`
        Night whose Moon position should be used.

    Returns
    -------
    separation : `float`
        Degrees, or NaN if the target coordinate is not finite.
    """
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return float("nan")
    from astropy.coordinates import angular_separation

    return float(
        np.degrees(
            angular_separation(
                np.radians(ra_deg),
                np.radians(dec_deg),
                np.radians(events.moon_ra_deg),
                np.radians(events.moon_dec_deg),
            )
        )
    )
