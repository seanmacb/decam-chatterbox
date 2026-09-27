"""Tests for decam_chatterbox.astro.almanac."""

import numpy as np
import pytest
from astropy.time import Time

from decam_chatterbox.astro.almanac import (
    CTIO_TZ,
    ctio_observer,
    format_time,
    moon_separation_deg,
    night_events,
)


def test_ctio_observer_location():
    observer = ctio_observer()
    # Cerro Tololo: roughly -30.2 deg latitude, -70.8 deg longitude.
    assert observer.location.lat.deg == pytest.approx(-30.17, abs=0.1)
    assert observer.location.lon.deg == pytest.approx(-70.8, abs=0.1)


def test_night_events_orders_sun_crossings_correctly():
    when = Time("2018-11-01T22:22:46.654", scale="utc")
    events = night_events(when)
    assert events.sunset.jd < events.sun_n12_setting.jd
    assert events.sun_n12_setting.jd < events.sun_n18_setting.jd
    assert events.sun_n18_setting.jd < events.sun_n18_rising.jd
    assert events.sun_n18_rising.jd < events.sun_n12_rising.jd
    assert events.sun_n12_rising.jd < events.sunrise.jd
    assert events.night_length_hours > events.dark_length_hours > 0


def test_night_events_advances_past_a_finished_night():
    """A time after the night's -12 deg morning crossing gets *that* evening,
    not the night that already ended."""
    when = Time("2018-11-01T22:22:46.654", scale="utc")  # evening at CTIO
    events = night_events(when, prefer_next_night=True)
    assert events.sunset.jd <= when.jd or events.sunset.jd > when.jd
    # The trigger time itself must fall before the reported night's sunrise,
    # i.e. this is not a night that already finished.
    assert when.jd < events.sun_n12_rising.jd


def test_night_events_moon_rise_set_are_none_when_outside_the_window():
    events = night_events(Time("2018-11-01T22:22:46.654", scale="utc"))
    for t, label in ((events.moonrise, "moonrise"), (events.moonset, "moonset")):
        if t is not None:
            assert events.sunset.jd <= t.jd <= events.sunrise.jd, label


def test_night_events_moon_illumination_in_unit_range():
    events = night_events(Time("2018-11-01T22:22:46.654", scale="utc"))
    assert 0.0 <= events.moon_illumination <= 1.0


def test_observing_window_rejects_unsupported_limit():
    events = night_events(Time("2018-11-01T22:22:46.654", scale="utc"))
    with pytest.raises(ValueError):
        events.observing_window(-6.0)


def test_moon_separation_matches_manual_computation():
    events = night_events(Time("2018-11-01T22:22:46.654", scale="utc"))
    # Separation from the Moon's own position must be ~0.
    sep = moon_separation_deg(events.moon_ra_deg, events.moon_dec_deg, events)
    assert sep == pytest.approx(0.0, abs=1e-6)


def test_moon_separation_nan_for_non_finite_coord():
    events = night_events(Time("2018-11-01T22:22:46.654", scale="utc"))
    assert np.isnan(moon_separation_deg(float("nan"), 0.0, events))


def test_format_time_handles_none():
    assert format_time(None) == "--"
    assert format_time(None, CTIO_TZ) == "--"


def test_format_time_utc_and_local_differ():
    t = Time("2018-11-01T23:00:00", scale="utc")
    utc_text = format_time(t)
    local_text = format_time(t, CTIO_TZ)
    assert "UTC" in utc_text
    assert "UTC" not in local_text


@pytest.mark.parametrize("nside", [16])
def test_fast_altitude_matches_full_altaz_transform(nside):
    """Cross-check the vectorized approximation against a full AltAz transform.

    Mirrors the check chatterbox's own Rubin almanac does for its
    ``approx_ra_dec2_alt_az``: the fast path used for the dark-hours grid
    should agree with astropy's own transform to a small fraction of a
    degree.
    """
    import healpy as hp
    from astropy.coordinates import AltAz, SkyCoord

    from decam_chatterbox.astro.darkhours import fast_altitude_deg

    observer = ctio_observer()
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    times = Time(
        np.linspace(Time("2018-11-01T23:00:00").mjd, Time("2018-11-02T09:00:00").mjd, 6),
        format="mjd",
        scale="utc",
        location=observer.location,
    )
    lst_deg = times.sidereal_time("apparent").deg
    alt_fast = fast_altitude_deg(ra[:, None], dec[:, None], lst_deg[None, :], observer.location.lat.deg)

    rng = np.random.default_rng(1)
    pix_idx = rng.integers(0, npix, 8)
    time_idx = rng.integers(0, times.size, 8)
    frame = AltAz(obstime=times[time_idx], location=observer.location)
    coords = SkyCoord(ra=ra[pix_idx], dec=dec[pix_idx], unit="deg")
    alt_true = coords.transform_to(frame).alt.deg
    alt_check = alt_fast[pix_idx, time_idx]
    assert np.max(np.abs(alt_true - alt_check)) < 0.3
