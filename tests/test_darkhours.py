"""Tests for decam_chatterbox.astro.darkhours."""

import healpy as hp
import numpy as np
import pytest
from astropy.time import Time

from decam_chatterbox.astro.almanac import night_events
from decam_chatterbox.astro.darkhours import airmass_to_altitude_deg, dark_hours_map


@pytest.fixture(scope="module")
def events():
    return night_events(Time("2018-11-01T22:22:46.654", scale="utc"))


def test_airmass_to_altitude_known_values():
    assert airmass_to_altitude_deg(1.0) == pytest.approx(90.0)
    assert airmass_to_altitude_deg(2.0) == pytest.approx(30.0, abs=0.1)


def test_airmass_to_altitude_rejects_subunity():
    with pytest.raises(ValueError):
        airmass_to_altitude_deg(0.5)


def test_dark_hours_map_bounded_by_window(events):
    dark_hours = dark_hours_map(events, nside=16, step_minutes=10.0)
    assert (dark_hours.hours >= 0).all()
    assert (dark_hours.hours <= dark_hours.window_hours + 1e-9).all()
    assert dark_hours.hours.max() > 0  # something is accessible some of the night


def test_dark_hours_map_zenith_pixel_gets_more_hours_than_a_northern_one(events):
    """A pixel near CTIO's zenith declination should be accessible longer
    than one far to the north, which barely rises if at all."""
    nside = 32
    dark_hours = dark_hours_map(events, nside=nside, step_minutes=5.0)
    ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)

    near_zenith = np.argmin(np.abs(dec - (-30.0)))
    far_north = np.argmin(np.abs(dec - 60.0))
    assert dark_hours.hours[near_zenith] > dark_hours.hours[far_north]


def test_dark_hours_map_moon_avoidance_reduces_hours_near_the_moon(events):
    nside = 32
    no_avoidance = dark_hours_map(events, nside=nside, step_minutes=10.0, moon_avoidance_deg=0.0)
    with_avoidance = dark_hours_map(events, nside=nside, step_minutes=10.0, moon_avoidance_deg=30.0)

    moon_pix = hp.ang2pix(nside, events.moon_ra_deg, events.moon_dec_deg, lonlat=True)
    assert with_avoidance.hours[moon_pix] <= no_avoidance.hours[moon_pix]
    assert with_avoidance.hours.sum() <= no_avoidance.hours.sum()


def test_stats_in_region_all_false_mask_is_nan(events):
    dark_hours = dark_hours_map(events, nside=16, step_minutes=10.0)
    stats = dark_hours.stats_in_region(np.zeros(dark_hours.hours.size, dtype=bool))
    assert np.isnan(stats["mean_hours"])
    assert stats["fraction_accessible"] == 0.0


def test_weighted_stats_matches_manual_average(events):
    dark_hours = dark_hours_map(events, nside=16, step_minutes=10.0)
    weights = np.ones(dark_hours.hours.size)
    stats = dark_hours.weighted_stats(weights)
    assert stats["weighted_mean_hours"] == pytest.approx(dark_hours.hours.mean())
