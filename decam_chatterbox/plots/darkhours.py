"""Localization contour over the accessible-dark-hours map, at CTIO."""

import logging
from pathlib import Path

import healpy as hp
import numpy as np

from ..astro.darkhours import DarkHoursMap
from ..astro.skymap import localization_region
from ..deps import require
from ..models import Localization
from .style import (
    ALLSKY_CENTER,
    PROJECTION,
    add_galactic_plane,
    localization_levels,
    mark_coord,
    use_headless_backend,
)

__all__ = ["plot_dark_hours", "region_hours_summary"]

logger = logging.getLogger(__name__)


def plot_dark_hours(
    dark_hours: DarkHoursMap,
    localization: Localization,
    out_path: str | Path,
    title: str | None = None,
    peak: tuple[float, float] | None = None,
    dpi: int = 130,
) -> Path:
    """Draw the dark-hours map with the localization contour on top.

    Parameters
    ----------
    dark_hours : `DarkHoursMap`
        Accessible hours per pixel for the night.
    localization : `Localization`
        Localization whose contour is overlaid.
    out_path : `str` or `pathlib.Path`
        PNG destination.
    title : `str`, optional
        Plot title. A description of the night is generated when omitted.
    peak : `tuple` [`float`, `float`], optional
        RA/Dec in degrees of the most probable position, to mark.
    dpi : `int`
        Output resolution.

    Returns
    -------
    path : `pathlib.Path`
        The file written.
    """
    use_headless_backend()
    # Registers the 'astro ... mollweide' projections. require() names the
    # interpreter, since a wrong-environment install is the usual cause.
    require("ligo.skymap.plot")
    from matplotlib import pyplot as plt

    events = dark_hours.events
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 6), dpi=dpi)
    ax = plt.axes(projection=PROJECTION, center=ALLSKY_CENTER)

    # Mask pixels that never reach the airmass limit so they read as "not
    # accessible" rather than as the bottom of the colour scale.
    hours = np.array(dark_hours.hours, dtype=float)
    display = np.where(hours > 0, hours, np.nan)

    vmax = float(np.nanmax(display)) if np.isfinite(display).any() else 1.0
    # nested=False: decam-chatterbox works in RING ordering throughout.
    img = ax.imshow_hpx(display, nested=False, cmap="viridis", vmin=0.0, vmax=vmax)

    levels = localization_levels(localization)
    if levels:
        ax.contour_hpx(
            localization.prob_map,
            nested=False,
            levels=levels,
            colors=["red"],
            linewidths=[1.6],
            zorder=5,
        )
    else:
        logger.warning("No contour drawn for %s", out_path.name)

    add_galactic_plane(ax)
    if peak is not None:
        mark_coord(ax, peak[0], peak[1], label="Most probable position", color="red")

    ax.grid(alpha=0.25)
    cbar = fig.colorbar(img, ax=ax, location="bottom", pad=0.06, shrink=0.8, aspect=40)
    cbar.set_label(
        f"Accessible dark hours at CTIO (airmass < {dark_hours.airmass_limit:g}, "
        f"Sun < {dark_hours.sun_alt_limit_deg:g} deg)"
    )

    if title is None:
        stats = region_hours_summary(dark_hours, localization)
        title = (
            f"Night of {events.day_obs}: {dark_hours.window_hours:.1f} h with Sun < "
            f"{dark_hours.sun_alt_limit_deg:g} deg\n"
            f"{localization.credible_level:.0%} region: max {stats['max_hours']:.1f} h, "
            f"mean {stats['mean_hours']:.1f} h accessible; {stats['probability_accessible']:.0%} of its "
            f"probability and {stats['area_accessible']:.0%} of its area reach airmass < "
            f"{dark_hours.airmass_limit:g}"
        )
    ax.set_title(title, fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def region_hours_summary(dark_hours: DarkHoursMap, localization: Localization) -> dict[str, float]:
    """Accessible dark time inside a localization's credible region.

    The region is `decam_chatterbox.astro.skymap.localization_region` -- the
    same one the plot's contour draws -- not every pixel with any probability:
    a real skymap has faint probability almost everywhere, which would turn
    every statistic here into a whole-sky one.

    How much of the region DECam can reach tonight is reported two ways,
    because neither implies the other: the densest part of a region can be up
    while most of its sky is not, or the reverse.

    Everything is measured at the finer of the two maps' resolutions -- in
    practice the localization's own -- by replicating the coarser map's
    pixels, so the region's area is not quantized by the dark-hours map's
    much larger pixels.

    Parameters
    ----------
    dark_hours : `DarkHoursMap`
        Accessible hours per pixel for the night.
    localization : `Localization`
        Localization whose credible region is summarized.

    Returns
    -------
    stats : `dict`
        ``max_hours``, ``mean_hours``, ``median_hours``: accessible hours over
        the region's pixels, unweighted; ``weighted_mean_hours``: the same,
        weighted by probability within the region.
        ``probability_accessible``: fraction of the region's probability on
        pixels with any accessible dark time -- of the region's own, so a
        fully accessible region reads 1 rather than the credible level.
        ``area_accessible`` (alias ``fraction_accessible``): fraction of the
        region's area with any accessible dark time, ``area_accessible_deg2``
        the same in deg^2. ``region_area_deg2``, ``region_probability`` (the
        share of the whole map's probability the region holds, just above
        ``credible_level``) and ``credible_level`` describe the region itself.
    """
    prob = np.asarray(localization.prob_map, dtype=float)
    hours = np.asarray(dark_hours.hours, dtype=float)
    nside = max(hp.npix2nside(prob.size), dark_hours.nside)
    if prob.size != hp.nside2npix(nside):
        # power=-2 conserves the summed probability across the resolution
        # change.
        prob = hp.ud_grade(prob, nside, order_in="RING", order_out="RING", power=-2)
    if hours.size != hp.nside2npix(nside):
        # Upgrading replicates each pixel's hours into its children.
        hours = hp.ud_grade(hours, nside, order_in="RING", order_out="RING")

    region = localization_region(localization, nside)
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    stats: dict[str, float] = {
        "credible_level": float(localization.credible_level),
        "region_area_deg2": float(np.count_nonzero(region) * pixel_area),
        "region_probability": float(prob[region].sum()),
    }
    if not region.any() or stats["region_probability"] <= 0:
        logger.warning("Credible region is empty; no accessible dark-time statistics")
        nan = float("nan")
        stats.update(
            max_hours=nan,
            mean_hours=nan,
            median_hours=nan,
            weighted_mean_hours=nan,
            probability_accessible=0.0,
            area_accessible=0.0,
            fraction_accessible=0.0,
            area_accessible_deg2=0.0,
        )
        return stats

    values = hours[region]
    weights = prob[region] / stats["region_probability"]
    accessible = values > 0
    area_fraction = float(np.count_nonzero(accessible) / values.size)
    stats.update(
        max_hours=float(values.max()),
        mean_hours=float(values.mean()),
        median_hours=float(np.median(values)),
        weighted_mean_hours=float(np.sum(weights * values)),
        # Clipped because the weighted sum can land a rounding error above 1.
        probability_accessible=float(min(np.sum(weights[accessible]), 1.0)),
        area_accessible=area_fraction,
        fraction_accessible=area_fraction,
        area_accessible_deg2=float(np.count_nonzero(accessible) * pixel_area),
    )
    return stats
