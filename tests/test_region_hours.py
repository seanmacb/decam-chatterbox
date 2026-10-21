"""Tests for decam_chatterbox.plots.darkhours.region_hours_summary.

The hours maps here are built by hand rather than from a real night, so which
pixels are accessible is known exactly and each statistic can be checked
against an answer worked out independently.
"""

import healpy as hp
import numpy as np
import pytest
from conftest import disc_probability_map

from decam_chatterbox.astro.darkhours import DarkHoursMap
from decam_chatterbox.astro.skymap import credible_mask
from decam_chatterbox.models import Localization
from decam_chatterbox.plots.darkhours import region_hours_summary

NSIDE = 32


def _dark_hours(hours: np.ndarray) -> DarkHoursMap:
    nside = hp.npix2nside(hours.size)
    return DarkHoursMap(
        hours=np.asarray(hours, dtype=float),
        nside=nside,
        events=None,
        airmass_limit=2.0,
        sun_alt_limit_deg=-12.0,
        altitude_limit_deg=30.0,
        window_hours=10.0,
    )


def _localization(prob: np.ndarray, level: float = 0.9) -> Localization:
    prob = np.asarray(prob, dtype=float)
    return Localization(
        prob_map=prob / prob.sum(), nside=hp.npix2nside(prob.size), credible_level=level, provenance="test"
    )


def _decs(nside: int = NSIDE) -> np.ndarray:
    return hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)[1]


def _gaussian(ra: float, dec: float, sigma_deg: float, nside: int = NSIDE) -> np.ndarray:
    vectors = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    centre = hp.ang2vec(ra, dec, lonlat=True)
    distance = np.degrees(np.arccos(np.clip(centre @ vectors, -1.0, 1.0)))
    return np.exp(-0.5 * (distance / sigma_deg) ** 2)


def test_uniform_weight_in_the_region_gives_equal_probability_and_area():
    """With no density gradient, probability and area are one measurement."""
    loc = _localization(disc_probability_map(60.0, -10.0, 20.0, NSIDE))
    # The disc straddles the equator; only its southern half is accessible.
    stats = region_hours_summary(_dark_hours(np.where(_decs() < -10.0, 3.0, 0.0)), loc)
    assert 0.2 < stats["area_accessible"] < 0.8
    assert stats["probability_accessible"] == pytest.approx(stats["area_accessible"])
    assert stats["area_accessible_deg2"] == pytest.approx(
        stats["area_accessible"] * stats["region_area_deg2"]
    )


def test_an_accessible_peak_buys_more_probability_than_area():
    """The reason both numbers are quoted: they disagree on a peaked map."""
    loc = _localization(_gaussian(60.0, -30.0, 8.0))
    near_peak = hp.query_disc(NSIDE, hp.ang2vec(60.0, -30.0, lonlat=True), np.radians(6.0))
    hours = np.zeros(hp.nside2npix(NSIDE))
    hours[near_peak] = 4.0
    stats = region_hours_summary(_dark_hours(hours), loc)
    assert 0.0 < stats["area_accessible"] < stats["probability_accessible"] < 1.0


def test_a_fully_accessible_region_reads_100_percent_not_the_credible_level():
    loc = _localization(_gaussian(60.0, -30.0, 8.0))
    stats = region_hours_summary(_dark_hours(np.full(hp.nside2npix(NSIDE), 5.0)), loc)
    assert stats["probability_accessible"] == pytest.approx(1.0)
    assert stats["area_accessible"] == pytest.approx(1.0)
    assert stats["region_probability"] == pytest.approx(0.9, abs=0.01)
    assert stats["credible_level"] == 0.9


def _compact_disc_with_faint_tails() -> np.ndarray:
    """95% of the probability in a disc, 5% spread over the rest of the sky.

    Every pixel has probability, as on a real skymap, but the 90% region is
    entirely inside the disc.
    """
    disc = disc_probability_map(60.0, -30.0, 10.0, NSIDE)
    tails = (disc == 0).astype(float)
    return 0.95 * disc + 0.05 * tails / tails.sum()


def test_probability_in_the_tails_does_not_count():
    """Accessible sky outside the region holds probability, but not the
    region's."""
    prob = _compact_disc_with_faint_tails()
    loc = _localization(prob)
    hours = np.where(prob < prob.max() / 2, 6.0, 0.0)  # everywhere *but* the disc
    stats = region_hours_summary(_dark_hours(hours), loc)
    assert stats["probability_accessible"] == 0.0
    assert stats["area_accessible"] == 0.0


def test_the_region_is_the_credible_region_not_every_pixel_with_probability():
    """Regression: the region used to be ``prob_map > 0``, which on a real
    skymap is the whole sky, so "X% of the region is accessible" and the mean
    hours were whole-sky numbers."""
    prob = _compact_disc_with_faint_tails()
    assert (prob > 0).all()
    loc = _localization(prob)
    hours = np.where(prob < prob.max() / 2, 6.0, 0.0)
    stats = region_hours_summary(_dark_hours(hours), loc)
    assert stats["fraction_accessible"] == 0.0
    assert stats["max_hours"] == 0.0
    assert stats["mean_hours"] == 0.0
    assert stats["weighted_mean_hours"] == 0.0
    assert stats["region_area_deg2"] < 0.05 * 41253.0


def test_the_region_is_measured_at_the_localizations_resolution():
    """A coarse dark-hours map must not quantize the region's area."""
    nside = 128
    prob = disc_probability_map(60.0, -30.0, 3.0, nside)
    loc = _localization(prob)
    coarse = _dark_hours(np.full(hp.nside2npix(8), 2.0))
    stats = region_hours_summary(coarse, loc)
    expected = np.count_nonzero(credible_mask(loc.prob_map, 0.9)) * hp.nside2pixarea(nside, degrees=True)
    assert stats["region_area_deg2"] == pytest.approx(expected)
    assert stats["area_accessible_deg2"] == pytest.approx(expected)


def test_an_empty_region_is_nan_hours_and_nothing_accessible():
    loc = Localization(
        prob_map=np.zeros(hp.nside2npix(NSIDE)), nside=NSIDE, credible_level=0.9, provenance="empty"
    )
    stats = region_hours_summary(_dark_hours(np.full(hp.nside2npix(NSIDE), 1.0)), loc)
    assert np.isnan(stats["mean_hours"])
    assert stats["probability_accessible"] == 0.0
    assert stats["area_accessible"] == 0.0
