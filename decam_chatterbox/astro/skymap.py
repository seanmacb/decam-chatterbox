"""Skymap handling: decoding a notice's embedded map, credible regions and
geometry.

Works throughout in **RING** ordering.

An LVK notice carries its sky map inline -- base64-encoded bytes in a JSON
notice from GCN, raw bytes in an Avro notice from SCIMMA (see
`decam_chatterbox.ingest.decode`) -- as a multi-order (NUNIQ) FITS file.
``ligo.skymap.io.read_sky_map`` reads either a multi-order or flat-resolution
map and rasterizes it to a single-resolution array, which is the same
function chatterbox uses for its own GraceDB-fetched skymaps.
"""

import logging
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from ..models import Geometry, Localization

__all__ = [
    "WORKING_NSIDE",
    "pixel_corner_decs",
    "credible_mask",
    "contour_levels",
    "localization_region",
    "peak_position",
    "geometry_from_mask",
    "localization_from_probability",
    "save_notice_skymap",
    "read_notice_skymap",
]

logger = logging.getLogger(__name__)

#: Resolution the probability map is degraded to. Full-resolution BAYESTAR
#: maps can be nside 2048 (50M pixels); this keeps plotting and the
#: dark-hours overlay fast while staying far finer than the credible-region
#: geometry needs to be exact.
WORKING_NSIDE = 512


def pixel_corner_decs(nside: int, pixels, nest: bool = False, step: int = 1) -> np.ndarray:
    """Declinations of the corners of a set of HEALPix pixels, in radians.

    Corner declinations rather than centre declinations are used so a
    region's true extent is never understated by half a pixel.

    Parameters
    ----------
    nside : `int`
        Resolution of `pixels`.
    pixels : array-like of `int`
        Pixel indices, all at the same `nside`.
    nest : `bool`
        True when `pixels` are in NESTED ordering.
    step : `int`
        Corners sampled per pixel side; 1 gives the four true corners.

    Returns
    -------
    decs : `numpy.ndarray`
        Flattened declinations in radians.
    """
    indices = np.atleast_1d(np.asarray(pixels, dtype=np.int64))
    # healpy returns (3, 4 * step) for a scalar and (n, 3, 4 * step) for an
    # array; reshaping handles both without branching.
    corners = np.reshape(hp.boundaries(nside, indices, step=step, nest=nest), (-1, 3, 4 * step))
    return np.arcsin(np.clip(corners[:, 2, :], -1.0, 1.0)).ravel()


def credible_mask(prob_map: np.ndarray, level: float = 0.9) -> np.ndarray:
    """Boolean mask of the smallest region containing a given probability.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Probability per pixel (need not be normalized).
    level : `float`
        Credible level, e.g. 0.9.

    Returns
    -------
    mask : `numpy.ndarray`
        True for pixels inside the credible region.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    total = prob_map.sum()
    if total <= 0:
        return np.zeros(prob_map.size, dtype=bool)
    order = np.argsort(prob_map)[::-1]
    cumulative = np.cumsum(prob_map[order]) / total
    # searchsorted gives the first index reaching the level; include it so the
    # region contains at least `level`, never slightly less.
    cutoff = int(np.searchsorted(cumulative, level))
    cutoff = min(cutoff, prob_map.size - 1)
    mask = np.zeros(prob_map.size, dtype=bool)
    mask[order[: cutoff + 1]] = True
    return mask


def contour_levels(prob_map: np.ndarray, levels=(0.5, 0.9)) -> list[float]:
    """Density thresholds enclosing the given credible levels.

    Suitable for ``ligo.skymap``'s ``contour_hpx``, sorted ascending as
    matplotlib requires.

    Returns
    -------
    thresholds : `list` [`float`]
        One density threshold per requested level. Levels that cannot be
        reached (for example on a degenerate map) are omitted.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    total = prob_map.sum()
    if total <= 0:
        return []
    sorted_prob = np.sort(prob_map)[::-1]
    cumulative = np.cumsum(sorted_prob) / total
    thresholds = []
    for level in levels:
        idx = int(np.searchsorted(cumulative, level))
        if idx >= sorted_prob.size:
            continue
        thresholds.append(float(sorted_prob[idx]))
    # Deduplicate: a flat map gives the same threshold for every level.
    unique = sorted(set(thresholds))
    return [t for t in unique if t > 0]


def localization_region(localization: Localization, nside: int | None = None) -> np.ndarray:
    """Boolean mask of the region a localization's primary contour encloses.

    This is the counterpart to
    `decam_chatterbox.plots.style.localization_levels`: the same region that
    gets drawn, so anything selected by it agrees with the picture.

    Parameters
    ----------
    localization : `Localization`
        Localization to reduce to a region.
    nside : `int`, optional
        Resolution for the mask. Defaults to the localization's own.

    Returns
    -------
    mask : `numpy.ndarray`
        Boolean, RING ordering.
    """
    prob = np.asarray(localization.prob_map, dtype=float)
    if nside is not None and prob.size != hp.nside2npix(nside):
        # power=-2 conserves the summed probability across the resolution
        # change.
        prob = hp.ud_grade(prob, nside, order_in="RING", order_out="RING", power=-2)
    return credible_mask(prob, localization.credible_level)


def peak_position(prob_map: np.ndarray) -> tuple[float, float]:
    """RA and Dec of the most probable pixel of a RING probability map.

    Pixels tied for the maximum (a flat map, or a native pixel coarser than
    the working resolution that was split into equal children) are resolved
    to whichever of them lies closest to the tied set's vector mean, so the
    answer is always a genuine maximum-probability pixel rather than an
    arbitrary first one in array order.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Probability per pixel, full sky, RING ordering.

    Returns
    -------
    ra_deg, dec_deg : `float`
        NaN when the map has no positive probability.
    """
    prob_map = np.asarray(prob_map, dtype=float)
    if prob_map.size == 0 or not prob_map.max() > 0:
        return float("nan"), float("nan")
    tied = np.flatnonzero(prob_map >= prob_map.max() * (1.0 - 1e-9))
    nside = hp.npix2nside(prob_map.size)
    if tied.size > 1:
        vectors = np.array(hp.pix2vec(nside, tied))
        mean = vectors.mean(axis=1)
        norm = np.linalg.norm(mean)
        if norm > 0:
            tied = tied[[int(np.argmax((mean / norm) @ vectors))]]
        else:
            tied = tied[:1]
    ra, dec = hp.pix2ang(nside, int(tied[0]), lonlat=True)
    return float(ra), float(dec)


def geometry_from_mask(mask: np.ndarray, nside: int, prob_map: np.ndarray) -> Geometry:
    """Derive sky geometry from a boolean credible-region mask in RING order.

    Declination limits use pixel *corners* rather than centres, so the
    reported range is not optimistic by half a pixel.

    Parameters
    ----------
    mask : `numpy.ndarray`
        Boolean, RING ordering, full sky.
    nside : `int`
        Map resolution.
    prob_map : `numpy.ndarray`
        Probability per pixel at `nside`, RING ordering; the marked position
        is its maximum (`peak_position`).

    Returns
    -------
    geometry : `Geometry`
    """
    mask = np.asarray(mask, dtype=bool)
    pixels = np.flatnonzero(mask)
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    if pixels.size == 0:
        logger.warning("Credible-region mask is empty; geometry will be degenerate")
        return Geometry(
            area_deg2=0.0,
            dec_min_deg=float("nan"),
            dec_max_deg=float("nan"),
            peak_ra_deg=float("nan"),
            peak_dec_deg=float("nan"),
            gal_b_abs_min_deg=float("nan"),
            gal_b_abs_max_deg=float("nan"),
            n_pixels=0,
            pixel_area_deg2=pixel_area,
        )

    ra, dec = hp.pix2ang(nside, pixels, lonlat=True)

    corner_dec = np.degrees(pixel_corner_decs(nside, pixels, nest=False))
    dec_min = float(corner_dec.min())
    dec_max = float(corner_dec.max())

    peak_ra, peak_dec = peak_position(prob_map)

    gal_b = np.abs(SkyCoord(ra=ra, dec=dec, unit="deg").galactic.b.deg)

    return Geometry(
        area_deg2=float(pixels.size * pixel_area),
        dec_min_deg=dec_min,
        dec_max_deg=dec_max,
        peak_ra_deg=peak_ra,
        peak_dec_deg=peak_dec,
        gal_b_abs_min_deg=float(gal_b.min()),
        gal_b_abs_max_deg=float(gal_b.max()),
        n_pixels=int(pixels.size),
        pixel_area_deg2=pixel_area,
    )


def localization_from_probability(
    prob_map: np.ndarray,
    provenance: str,
    credible_level: float = 0.9,
) -> Localization:
    """Build a `Localization` from a real probability map in RING ordering.

    Parameters
    ----------
    prob_map : `numpy.ndarray`
        Probability per pixel, full sky, RING ordering.
    provenance : `str`
        Where the map came from, shown in Slack.
    credible_level : `float`
        Credible level used when drawing the primary contour.

    Returns
    -------
    localization : `Localization`
    """
    prob_map = np.asarray(prob_map, dtype=float)
    total = prob_map.sum()
    if total <= 0:
        raise ValueError("Probability map sums to zero")
    return Localization(
        prob_map=prob_map / total,
        nside=hp.get_nside(prob_map),
        credible_level=credible_level,
        provenance=provenance,
    )


def save_notice_skymap(skymap_bytes: bytes, out_path: str | Path) -> Path:
    """Write a notice's decoded skymap bytes to disk as a FITS file.

    Kept on disk (under ``paths.work_dir/skymaps``, by default) rather than
    only held in memory, so a real event's map can be reopened later in DS9,
    Aladin or a notebook -- ``ligo.skymap.io.read_sky_map`` also needs a path
    rather than an in-memory buffer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(skymap_bytes)
    return out_path


def read_notice_skymap(path: str | Path, working_nside: int = WORKING_NSIDE) -> tuple[np.ndarray, dict]:
    """Read a skymap FITS file as a RING probability map at `working_nside`.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        FITS file written by `save_notice_skymap`.
    working_nside : `int`
        Resolution to degrade (or occasionally upgrade) the map to.

    Returns
    -------
    prob : `numpy.ndarray`
        Probability per pixel, RING ordering, summing to 1.
    meta : `dict`
        Header metadata, including ``distmean``/``diststd`` when the map
        carries distance information (CBC events only).
    """
    from ligo.skymap.io import read_sky_map

    # moc=False rasterizes a multi-order file to a flat map, at whatever
    # order its finest pixels use.
    prob, meta = read_sky_map(str(path), moc=False, nest=True)
    prob = np.asarray(prob, dtype=float)
    in_nside = hp.get_nside(prob)

    if in_nside != working_nside:
        # power=-2 conserves the summed probability across the resolution
        # change.
        prob = hp.ud_grade(prob, working_nside, order_in="NESTED", order_out="RING", power=-2)
    else:
        prob = hp.reorder(prob, n2r=True)

    total = prob.sum()
    if total <= 0:
        raise ValueError(f"Skymap {path} has zero total probability")
    return prob / total, dict(meta or {})
