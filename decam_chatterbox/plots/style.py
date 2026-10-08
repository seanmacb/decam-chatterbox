"""Shared plotting helpers: projection, overlays and contour levels.

Plots use ``ligo.skymap.plot``'s WCS axes so that HEALPix maps and sky
coordinates can be drawn in the same frame.
"""

import logging

import numpy as np

from ..models import Localization

__all__ = [
    "use_headless_backend",
    "PROJECTION",
    "ALLSKY_CENTER_RA_DEG",
    "ALLSKY_CENTER",
    "GALACTIC_BUFFER_DEG",
    "add_galactic_plane",
    "localization_levels",
    "localization_extent",
    "sky_projection",
    "MAX_ZOOM_RADIUS_DEG",
    "mark_coord",
]

logger = logging.getLogger(__name__)

#: Default all-sky projection.
PROJECTION = "astro degrees mollweide"

#: RA at the middle of the all-sky map. ligo.skymap's own default puts RA 180
#: there; 270 puts the seam at RA 90 instead.
ALLSKY_CENTER_RA_DEG = 270.0
ALLSKY_CENTER = f"{ALLSKY_CENTER_RA_DEG:g}d 0d"

#: Half-width of the shaded Galactic plane region, degrees.
GALACTIC_BUFFER_DEG = 10.0

_backend_set = False


def use_headless_backend() -> None:
    """Select a non-interactive matplotlib backend, once per process.

    Required because the bot renders plots from a service with no display.
    """
    global _backend_set
    if _backend_set:
        return
    import matplotlib

    matplotlib.use("Agg")
    _backend_set = True


def add_galactic_plane(
    ax, buffer_deg: float = GALACTIC_BUFFER_DEG, center_ra_deg: float = ALLSKY_CENTER_RA_DEG
) -> None:
    """Overlay the Galactic plane and a +/- buffer in Galactic latitude.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        A ``ligo.skymap`` WCS axes.
    buffer_deg : `float`
        Latitude offset for the dashed limits.
    center_ra_deg : `float`
        RA at the middle of the map. The lines are broken where they cross
        the opposite edge, which is the seam of the projection.
    """
    from astropy.coordinates import SkyCoord

    ell = np.linspace(0.0, 360.0, 721)
    specs = (
        (0.0, {"ls": "--", "color": "black", "lw": 0.8, "label": "Galactic plane"}),
        (
            buffer_deg,
            {
                "ls": "--",
                "color": "black",
                "lw": 0.6,
                "alpha": 0.5,
                "label": f"|b| = {buffer_deg:.0f} deg",
            },
        ),
        (-buffer_deg, {"ls": "--", "color": "black", "lw": 0.6, "alpha": 0.5}),
    )
    for b, kwargs in specs:
        coords = SkyCoord(l=ell, b=np.full_like(ell, b), unit="deg", frame="galactic").icrs
        ra = coords.ra.deg
        dec = coords.dec.deg
        # Break the line where it crosses the map's edge so the projection
        # does not draw a horizontal streak across the whole map. RA is
        # measured from the map centre, which is what puts that edge at
        # centre +/- 180 rather than at RA 0.
        offset = (ra - center_ra_deg + 180.0) % 360.0 - 180.0
        jumps = np.flatnonzero(np.abs(np.diff(offset)) > 180.0)
        segments = np.split(np.arange(ra.size), jumps + 1)
        for n, seg in enumerate(segments):
            if seg.size < 2:
                continue
            kw = dict(kwargs)
            if n > 0:
                kw.pop("label", None)
            ax.plot(ra[seg], dec[seg], transform=ax.get_transform("world"), **kw)


def localization_levels(localization: Localization, levels=(0.5, 0.9)) -> list[float]:
    """Contour levels that trace a localization.

    Returns
    -------
    thresholds : `list` [`float`]
        Ascending, suitable for ``contour_hpx``. Empty if nothing can be drawn.
    """
    from ..astro.skymap import contour_levels

    prob = np.asarray(localization.prob_map, dtype=float)
    positive = prob[prob > 0]
    if positive.size == 0:
        logger.warning("Localization has no non-zero pixels; no contour will be drawn")
        return []

    # A map that is flat everywhere it is non-zero has no density gradient to
    # contour: asking for the level enclosing 90% lands exactly on the
    # plateau, and matplotlib then paints the whole region rather than its
    # outline. Such a map gets a single level just below the plateau, which
    # traces the boundary instead. np.unique on a working-resolution map is
    # affordable here and unambiguous.
    if np.unique(positive).size == 1:
        return [float(positive.min()) / 2.0]

    # Drop any level at or above the peak: a contour there encloses nothing.
    peak = float(positive.max())
    thresholds = [level for level in contour_levels(prob, levels) if level < peak]
    if not thresholds:
        return [float(positive.min()) / 2.0]
    return thresholds


#: A localization whose region fits inside this radius is drawn zoomed in.
MAX_ZOOM_RADIUS_DEG = 25.0


def localization_extent(localization: Localization) -> tuple[float, float, float]:
    """Centre and angular size of a localization's region, for framing.

    Returns
    -------
    ra_deg, dec_deg : `float`
        Centre, by vector mean so a region spanning RA 0 does not average
        to RA 180.
    radius_deg : `float`
        Greatest angular distance from that centre to any region pixel.
        NaN when the region is empty.
    """
    import healpy as hp

    from ..astro.skymap import localization_region

    mask = localization_region(localization)
    pixels = np.flatnonzero(mask)
    if pixels.size == 0:
        return float("nan"), float("nan"), float("nan")

    nside = hp.get_nside(mask)
    vectors = np.array(hp.pix2vec(nside, pixels))
    mean = vectors.mean(axis=1)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return float("nan"), float("nan"), float("nan")
    mean /= norm
    ra, dec = hp.vec2ang(mean, lonlat=True)
    cosines = np.clip(vectors.T @ mean, -1.0, 1.0)
    radius = float(np.degrees(np.arccos(cosines.min())))
    return float(ra[0]), float(dec[0]), radius


def sky_projection(localization: Localization, max_zoom_radius_deg: float = MAX_ZOOM_RADIUS_DEG):
    """Projection and keyword arguments suited to a localization's size.

    A compact localization is unreadable on an all-sky map, so it gets a
    zoomed frame. A large or disjoint region gets the all-sky view, because a
    zoom would silently crop part of it.

    Returns
    -------
    projection : `str`
        Projection name for ``plt.subplot``.
    kwargs : `dict`
        Extra keyword arguments (the map centre, for the all-sky case).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ra, dec, radius = localization_extent(localization)
    if not np.isfinite(radius) or radius > max_zoom_radius_deg:
        return PROJECTION, {"center": ALLSKY_CENTER}
    return "astro zoom", {
        "center": SkyCoord(ra * u.deg, dec * u.deg),
        "radius": max(radius * 1.4, 1.0) * u.deg,
    }


def mark_coord(ax, ra_deg: float, dec_deg: float, label: str | None = None, **kwargs) -> None:
    """Mark a sky position with a reticle."""
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return
    style = {"marker": "+", "color": "black", "markersize": 9, "markeredgewidth": 1.5}
    style.update(kwargs)
    if label:
        style["label"] = label
    ax.plot(ra_deg, dec_deg, transform=ax.get_transform("world"), linestyle="none", **style)
