"""Tests for decam_chatterbox.astro.skymap."""

import healpy as hp
import numpy as np
import pytest
from conftest import disc_probability_map

from decam_chatterbox.astro.skymap import (
    contour_levels,
    credible_mask,
    geometry_from_mask,
    localization_from_probability,
    localization_region,
    read_notice_skymap,
    save_notice_skymap,
)


def test_credible_mask_contains_at_least_the_requested_level():
    prob = disc_probability_map(60.0, -40.0, 10.0, nside=32)
    mask = credible_mask(prob, 0.9)
    assert prob[mask].sum() >= 0.9
    # And it should be the *smallest* such region: removing any pixel from
    # the mask must drop it below 0.9 (true for a uniform-probability disc,
    # where every pixel inside the mask has equal weight).
    n_in = mask.sum()
    assert prob[mask].sum() / n_in * (n_in - 1) < 0.9


def test_credible_mask_of_all_zero_map_is_empty():
    prob = np.zeros(hp.nside2npix(16))
    mask = credible_mask(prob, 0.9)
    assert not mask.any()


def test_contour_levels_ascending_and_bounded():
    rng = np.random.default_rng(0)
    prob = rng.random(hp.nside2npix(32))
    prob /= prob.sum()
    levels = contour_levels(prob, levels=(0.5, 0.9))
    assert levels == sorted(levels)
    assert all(0 < level <= prob.max() for level in levels)


def test_geometry_from_mask_matches_known_disc():
    nside = 64
    prob = disc_probability_map(60.0, -40.0, 10.0, nside=nside)
    mask = prob > 0
    geo = geometry_from_mask(mask, nside)
    assert geo.centroid_ra_deg == pytest.approx(60.0, abs=1.0)
    assert geo.centroid_dec_deg == pytest.approx(-40.0, abs=1.0)
    assert geo.dec_min_deg < -40.0 < geo.dec_max_deg
    assert geo.n_pixels == int(mask.sum())
    assert geo.area_deg2 == pytest.approx(geo.n_pixels * geo.pixel_area_deg2)


def test_geometry_from_mask_empty_is_degenerate():
    geo = geometry_from_mask(np.zeros(hp.nside2npix(16), dtype=bool), 16)
    assert geo.n_pixels == 0
    assert geo.area_deg2 == 0.0
    assert np.isnan(geo.centroid_ra_deg)


def test_localization_from_probability_normalizes():
    prob = disc_probability_map(0.0, 0.0, 5.0, nside=32) * 7.0  # deliberately not normalized
    loc = localization_from_probability(prob, provenance="test", credible_level=0.9)
    assert loc.prob_map.sum() == pytest.approx(1.0)
    assert loc.nside == 32


def test_localization_from_probability_rejects_zero_map():
    with pytest.raises(ValueError):
        localization_from_probability(np.zeros(hp.nside2npix(16)), provenance="test")


def test_localization_region_matches_credible_mask():
    prob = disc_probability_map(30.0, -20.0, 8.0, nside=32)
    loc = localization_from_probability(prob, provenance="test", credible_level=0.9)
    region = localization_region(loc)
    assert np.array_equal(region, credible_mask(prob, 0.9))


def test_save_and_read_notice_skymap_round_trip(skymap_bytes, tmp_path):
    path = save_notice_skymap(skymap_bytes, tmp_path / "sub" / "map.fits")
    assert path.is_file()
    prob, meta = read_notice_skymap(path, working_nside=32)
    assert prob.sum() == pytest.approx(1.0)
    assert meta["distmean"] == pytest.approx(120.0)
    # The map is uniform within the disc, so its centroid (not argmax, which
    # is order-dependent on a flat map) should land near the disc centre
    # used in conftest.make_skymap_bytes.
    geo = geometry_from_mask(prob > 0, hp.get_nside(prob))
    assert geo.centroid_ra_deg == pytest.approx(60.0, abs=3.0)
    assert geo.centroid_dec_deg == pytest.approx(-40.0, abs=3.0)


def test_read_notice_skymap_can_upgrade_resolution(skymap_bytes, tmp_path):
    path = save_notice_skymap(skymap_bytes, tmp_path / "map.fits")
    prob, _ = read_notice_skymap(path, working_nside=128)
    assert hp.get_nside(prob) == 128
    assert prob.sum() == pytest.approx(1.0)
