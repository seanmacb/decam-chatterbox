"""An animation of airmass across the sky at CTIO through night zero.

Each frame is the airmass of the whole sky at one moment of the first night,
with the localization's contour drawn over it, so an observer can watch the
field rise, cross and set against the airmass DECam can use. Airmass above a
cut-off is masked out as NaN rather than painted: the far end of a colour
scale would otherwise make a field on the horizon look like a field that is
merely a little worse.

Airmass is computed straight from altitude on a sky grid, not read from a
HEALPix map, so the zoomed panel is as smooth as the all-sky one. Both the
relation (``X = 1 / sin(altitude)``) and the altitude calculation (apparent
sidereal time from astropy, then
`~decam_chatterbox.astro.darkhours.fast_altitude_deg`) are the dark-hours
map's own, so a field that is "visible" in the post is the same field that has
a colour here.
"""

import logging
from pathlib import Path

import numpy as np
from astropy.time import Time

from ..astro.almanac import CTIO_TZ, NightEvents, format_time
from ..astro.darkhours import fast_altitude_deg
from ..deps import require
from ..models import Localization
from .style import (
    ALLSKY_CENTER,
    ALLSKY_CENTER_RA_DEG,
    PROJECTION,
    localization_levels,
    mark_coord,
    sky_projection,
    use_headless_backend,
)

__all__ = [
    "DEFAULT_MASK_AIRMASS",
    "airmass_from_altitude",
    "airmass_on_grid",
    "frame_times",
    "plot_airmass_animation",
]

logger = logging.getLogger(__name__)

#: Airmass above which a position is masked. 3.8 is an altitude of about 15
#: degrees, close to the lowest the Blanco can point.
DEFAULT_MASK_AIRMASS = 3.8


def airmass_from_altitude(altitude_deg, mask_above: float = DEFAULT_MASK_AIRMASS) -> np.ndarray:
    """Airmass from altitude, with unobservable positions set to NaN.

    Parameters
    ----------
    altitude_deg : array-like
        Altitude above the horizon, degrees.
    mask_above : `float`
        Airmass beyond which a position is masked.

    Returns
    -------
    airmass : `numpy.ndarray`
        ``1 / sin(altitude)``, NaN at or below the horizon -- where the
        relation is meaningless -- and wherever it exceeds `mask_above`.
    """
    altitude = np.asarray(altitude_deg, dtype=float)
    airmass = np.full(altitude.shape, np.nan)
    up = altitude > 0.0
    airmass[up] = 1.0 / np.sin(np.radians(altitude[up]))
    airmass[airmass > mask_above] = np.nan
    return airmass


def _lst_deg(mjd) -> np.ndarray:
    """Apparent local sidereal time at CTIO, degrees, as the dark-hours map
    computes it."""
    from ..astro.almanac import ctio_observer

    location = ctio_observer().location
    return np.atleast_1d(
        Time(mjd, format="mjd", scale="utc", location=location).sidereal_time("apparent").deg
    )


def airmass_on_grid(
    ra_deg, dec_deg, mjd: float, mask_above: float = DEFAULT_MASK_AIRMASS, lst_deg: float | None = None
) -> np.ndarray:
    """Airmass at CTIO for a set of sky positions at one time.

    Parameters
    ----------
    ra_deg, dec_deg : array-like
        Positions, degrees, any matching shape.
    mjd : `float`
        Time, UTC MJD.
    mask_above : `float`
        Airmass beyond which a position is masked to NaN.
    lst_deg : `float`, optional
        Sidereal time at `mjd`, when the caller already has it.

    Returns
    -------
    airmass : `numpy.ndarray`
        Same shape as the inputs.
    """
    from ..astro.almanac import ctio_observer

    if lst_deg is None:
        lst_deg = float(_lst_deg(mjd)[0])
    lat_deg = float(ctio_observer().location.lat.deg)
    altitude = fast_altitude_deg(
        np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float), lst_deg, lat_deg
    )
    return airmass_from_altitude(altitude, mask_above)


def frame_times(
    events: NightEvents, step_minutes: float = 10.0, sun_alt_limit_deg: float = -12.0
) -> np.ndarray:
    """Frame times across a night's observing window.

    Parameters
    ----------
    events : `NightEvents`
        The night.
    step_minutes : `float`
        Spacing between frames.
    sun_alt_limit_deg : `float`
        Sun altitude bounding the window; -12 or -18.

    Returns
    -------
    mjd : `numpy.ndarray`
        From the window's start to its end inclusive, so the last frame is
        the end of the night and not a step short of it.
    """
    start, end = events.observing_window(sun_alt_limit_deg)
    step = step_minutes / (60.0 * 24.0)
    times = np.arange(start, end, step)
    return np.append(times, end)


def _zoom_grid(ra_deg: float, dec_deg: float, radius_deg: float, n: int = 90):
    """A fine RA/Dec grid around a point, for the zoomed panel.

    RA offsets are stretched by ``1 / cos(dec)`` so the grid is square on the
    sky. Longitudes are left unwrapped (they may run below 0 or above 360) so
    a field across RA 0 stays one contiguous mesh.
    """
    half = radius_deg * 1.2
    dy = np.linspace(-half, half, n)
    dx = np.linspace(-half, half, n) / max(np.cos(np.radians(dec_deg)), 0.05)
    return np.meshgrid(ra_deg + dx, np.clip(dec_deg + dy, -90.0, 90.0))


def _cell_centres(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres of a corner grid's cells, since the mesh is drawn flat."""
    clon = 0.25 * (lon[:-1, :-1] + lon[1:, :-1] + lon[:-1, 1:] + lon[1:, 1:])
    clat = 0.25 * (lat[:-1, :-1] + lat[1:, :-1] + lat[:-1, 1:] + lat[1:, 1:])
    return clon, clat


def plot_airmass_animation(
    events: NightEvents,
    localization: Localization,
    out_path: str | Path,
    mask_above: float = DEFAULT_MASK_AIRMASS,
    step_minutes: float = 10.0,
    sun_alt_limit_deg: float = -12.0,
    peak: tuple[float, float] | None = None,
    fps: int = 8,
    dpi: int = 80,
) -> Path:
    """Animate airmass at CTIO over night zero, with the localization contour.

    Parameters
    ----------
    events : `NightEvents`
        The night to animate.
    localization : `Localization`
        Whose contour is drawn on every frame.
    out_path : `str` or `pathlib.Path`
        GIF destination. A GIF rather than a video because Slack plays it
        inline and no encoder has to be installed.
    mask_above : `float`
        Airmass beyond which the map is masked to NaN.
    step_minutes : `float`
        Time between frames.
    sun_alt_limit_deg : `float`
        Sun altitude bounding the night; -12 matches the rest of the post.
    peak : `tuple` [`float`, `float`], optional
        RA/Dec in degrees of the most probable position, to mark and to
        report the airmass of.
    fps : `int`
        Playback speed.
    dpi : `int`
        Frame resolution. Kept modest: a night is dozens of frames.

    Returns
    -------
    path : `pathlib.Path`
        The GIF written.

    Notes
    -----
    A compact localization gets a second, zoomed panel beside the all-sky
    one, because on an all-sky map a few square degrees is a few pixels. A
    large or disjoint one gets the all-sky panel alone, for the same reason
    `sky_projection` gives: a zoom would crop it.
    """
    use_headless_backend()
    require("ligo.skymap.plot")
    import matplotlib
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = frame_times(events, step_minutes, sun_alt_limit_deg)
    # One astropy call for every frame's sidereal time; everything per frame
    # after that is plain numpy.
    lst = _lst_deg(times)

    projection, zoom_kwargs = sky_projection(localization)
    zoomed = projection != PROJECTION

    # A bare `Figure` rather than `plt.figure`: this runs in a background
    # thread while the next alert's post may be drawing its own plot on the
    # main one, and pyplot's global figure registry is not thread-safe.
    # Nothing here touches it, so nothing has to be locked and the alert path
    # never waits.
    fig = Figure(figsize=(12.5 if zoomed else 9.0, 5.6), dpi=dpi)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1, 2 if zoomed else 1, width_ratios=[1.5, 1.0] if zoomed else [1.0])
    panels = [fig.add_subplot(grid[0, 0], projection=PROJECTION, center=ALLSKY_CENTER)]
    if zoomed:
        panels.append(fig.add_subplot(grid[0, 1], projection=projection, **zoom_kwargs))

    # One mesh per panel, updated in place each frame: redrawing a pcolormesh
    # per frame is what makes animations like this crawl. The all-sky grid
    # runs from one side of the projection's seam (opposite ALLSKY_CENTER) to
    # the other, just inside it: a cell with a corner exactly on the seam can
    # land on both edges at once and be drawn as a band across the whole map.
    seam = ALLSKY_CENTER_RA_DEG - 180.0
    lon, lat = np.meshgrid(np.linspace(seam + 1e-6, seam + 360.0 - 1e-6, 361), np.linspace(-90.0, 90.0, 181))
    grids = [(lon, lat)]
    if zoomed:
        radius = zoom_kwargs["radius"].to_value("deg")
        centre = zoom_kwargs["center"]
        grids.append(_zoom_grid(centre.ra.deg, centre.dec.deg, radius))
    centres = [_cell_centres(glon, glat) for glon, glat in grids]

    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("0.93")
    meshes = []
    for ax, (glon, glat) in zip(panels, grids):
        blank = np.full(glon[:-1, :-1].shape, np.nan)
        meshes.append(
            ax.pcolormesh(
                glon,
                glat,
                blank,
                transform=ax.get_transform("world"),
                cmap=cmap,
                vmin=1.0,
                vmax=mask_above,
                shading="flat",
            )
        )

    levels = localization_levels(localization)
    for ax in panels:
        if levels:
            # nested=False: decam-chatterbox works in RING ordering throughout.
            ax.contour_hpx(
                localization.prob_map,
                nested=False,
                levels=levels,
                colors=["red"],
                linewidths=[1.6],
                zorder=5,
            )
        if peak is not None:
            mark_coord(ax, peak[0], peak[1], color="red")
        ax.grid(alpha=0.25)
    panels[0].set_title("All sky", fontsize=9)
    if zoomed:
        panels[1].set_title("Localization", fontsize=9)

    cbar = fig.colorbar(meshes[0], ax=panels, location="bottom", pad=0.18, shrink=0.6, aspect=40)
    cbar.set_label(f"Airmass at CTIO (grey: below the horizon or above {mask_above:g})")
    headline = fig.suptitle("", fontsize=11)

    def peak_text(mjd: float, lst_deg: float) -> str:
        if peak is None or not (np.isfinite(peak[0]) and np.isfinite(peak[1])):
            return ""
        value = float(airmass_on_grid([peak[0]], [peak[1]], mjd, mask_above, lst_deg=lst_deg)[0])
        if not np.isfinite(value):
            return "; most probable position not observable"
        return f"; most probable position at airmass {value:.2f}"

    def draw(index: int):
        mjd = float(times[index])
        for mesh, (clon, clat) in zip(meshes, centres):
            mesh.set_array(airmass_on_grid(clon, clat, mjd, mask_above, lst_deg=lst[index]).ravel())
        when = Time(mjd, format="mjd", scale="utc")
        headline.set_text(
            f"Night of {events.day_obs}: {when.utc.strftime('%H:%M')} UTC "
            f"[{format_time(when, CTIO_TZ)} Chile]{peak_text(mjd, lst[index])}"
        )
        return meshes

    animation = FuncAnimation(fig, draw, frames=len(times), blit=False)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    logger.info("Wrote %s (%d frames)", out_path, len(times))
    return out_path
