"""Slack Block Kit message construction.

decam-chatterbox posts exactly one message per notice: everything computable
from the notice itself (and, best-effort, from a quick GraceDB chirp-mass
lookup) plus CTIO observability. There is no second stage and no follow-up
strategy -- the message answers "what is this, and can DECam even see it
tonight", and stops there.
"""

import logging
from typing import Any

import numpy as np

from ..astro.almanac import CTIO_TZ, NightEvents, format_time, moon_separation_deg
from ..astro.darkhours import DarkHoursMap
from ..config import Config
from ..links import format_link_list, gracedb_link, site_links
from ..models import Trigger

__all__ = [
    "build_trigger_blocks",
    "build_failure_blocks",
    "plain_text_summary",
    "MAX_SECTION_CHARS",
]

logger = logging.getLogger(__name__)

#: Slack rejects section text longer than this.
MAX_SECTION_CHARS = 2900

#: DECam's usable field of view is a 2.2 deg diameter circle, about 3.0 deg^2
#: (Flaugher et al. 2015, AJ 150, 150). Used only to describe how many
#: pointings a localization's size corresponds to -- not to plan any tiling.
DECAM_FOV_DEG2 = 3.0

#: A region entirely north of this declination is poorly placed for a
#: southern site at CTIO's latitude (-30.2 deg); a fast, obvious-case flag,
#: not a substitute for the dark-hours map below it.
POORLY_PLACED_DEC_DEG = 30.0

_ALERT_TYPE_EMOJI = {
    "EARLYWARNING": ":hourglass_flowing_sand:",
    "PRELIMINARY": ":large_yellow_circle:",
    "INITIAL": ":large_blue_circle:",
    "UPDATE": ":arrows_counterclockwise:",
    "RETRACTION": ":x:",
}


def _section(text: str) -> dict[str, Any]:
    """A mrkdwn section block, truncated to Slack's limit."""
    if len(text) > MAX_SECTION_CHARS:
        text = text[: MAX_SECTION_CHARS - 3] + "..."
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _context(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text[:2900]}]}


def _divider() -> dict[str, Any]:
    return {"type": "divider"}


def _fmt(value: float | None, spec: str = ".1f", suffix: str = "") -> str:
    """Format a possibly-missing number."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "unknown"
    return f"{value:{spec}}{suffix}"


def _duration(seconds: float | None) -> str:
    """Format a duration with a unit that suits its magnitude.

    Notice latency spans minutes (Preliminary) to days (Update), and a replay
    of an archived event can be months old, so a fixed unit is unreadable at
    one end or the other.
    """
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 90:
        return f"{sign}{seconds:.0f} s"
    if seconds < 5400:
        return f"{sign}{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{sign}{seconds / 3600:.1f} h"
    return f"{sign}{seconds / 86400:.1f} d"


# --------------------------------------------------------------------- header


def _plain_header(trigger: Trigger) -> str:
    """Plain-text header: Slack requires <= 150 chars and no emoji."""
    prefix = "" if trigger.is_real else "[NOT REAL] "
    kind = "Retraction" if trigger.is_retraction else (trigger.group or "GW event")
    return f"{prefix}{trigger.superevent_id}: {kind} ({trigger.alert_type})"[:150]


def _header_blocks(trigger: Trigger) -> list[dict[str, Any]]:
    emoji = _ALERT_TYPE_EMOJI.get(trigger.alert_type, ":ocean:")
    marker = "" if trigger.is_real else " :warning: *NOT A REAL SUPEREVENT* (mock/MDC or test)"
    header = f"{emoji} *Gravitational wave: {trigger.superevent_id}*{marker}"
    subtitle = f"`{trigger.alert_type}`"
    if trigger.group:
        subtitle += f" -- {trigger.group}"
        if trigger.pipeline:
            subtitle += f" / {trigger.pipeline}"
        if trigger.search:
            subtitle += f" / {trigger.search}"

    lines = [subtitle]
    instruments = ", ".join(trigger.instruments)
    if instruments:
        lines.append(f"*Instruments:* {instruments}")

    timing = []
    if trigger.event_time is not None:
        timing.append(f"Event: {format_time(trigger.event_time)}")
    if trigger.notice_latency_s is not None:
        timing.append(f"notice latency {_duration(trigger.notice_latency_s)}")
    if trigger.age_s is not None:
        timing.append(f"age {_duration(trigger.age_s)}")
    if timing:
        lines.append(" | ".join(timing))

    return [
        {"type": "header", "text": {"type": "plain_text", "text": _plain_header(trigger)}},
        _section(header + "\n" + "\n".join(lines)),
    ]


def _retraction_blocks(trigger: Trigger) -> list[dict[str, Any]]:
    lines = [
        f":x: *{trigger.superevent_id} has been retracted.*",
        "Human vetting concluded this candidate is probably not astrophysical. "
        "No further information -- FAR, localization or classification -- is included in a retraction.",
    ]
    link = gracedb_link(trigger.superevent_id, trigger.gracedb_url)
    return [_section("\n".join(lines)), _context(format_link_list([("GraceDB", link)]))]


# ------------------------------------------------------------ alert specifics


def _significance_blocks(trigger: Trigger) -> list[dict[str, Any]]:
    lines = ["*Significance*"]
    far_line = f"FAR {_fmt(trigger.far_hz, '.2e', ' Hz')}"
    if trigger.inverse_far_years is not None:
        far_line += f" (1 per {trigger.inverse_far_years:,.1f} yr)"
    if trigger.significant is not None:
        far_line += "  |  " + ("*significant*" if trigger.significant else "not significant")
    lines.append(far_line)
    blocks = [_section("\n".join(lines))]

    if trigger.is_burst:
        detail = []
        if trigger.duration_s is not None:
            detail.append(f"duration {trigger.duration_s:.3g} s")
        if trigger.central_frequency_hz is not None:
            detail.append(f"central frequency {trigger.central_frequency_hz:,.0f} Hz")
        if detail:
            blocks.append(_section("*Burst parameters*\n" + ", ".join(detail)))
        return blocks

    detail = []
    most_likely = trigger.most_likely_class
    if most_likely is not None:
        detail.append(f"most likely *{most_likely[0]}* ({most_likely[1]:.0%})")
    if trigger.classification:
        detail.append(
            "classification: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(trigger.classification.items()))
        )
    if trigger.properties:
        detail.append(
            "properties: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(trigger.properties.items()))
        )
    enrichment = trigger.enrichment
    if enrichment is not None and enrichment.chirp_mass_median is not None:
        chirp = f"median chirp mass {enrichment.chirp_mass_median:.2f} Msun"
        if enrichment.chirp_mass_prob_above_44 is not None:
            chirp += f", P(>44 Msun) {enrichment.chirp_mass_prob_above_44:.0%}"
        detail.append(chirp)
    elif enrichment is not None and enrichment.error:
        detail.append(f"_chirp mass unavailable ({enrichment.error})_")
    if detail:
        blocks.append(_section("*Classification*\n" + "\n".join(detail)))

    coinc = trigger.external_coinc
    if coinc is not None:
        parts = [f"Candidate coincidence with {coinc.observatory or 'an external facility'}"]
        if coinc.search:
            parts.append(f" ({coinc.search})")
        if coinc.time_difference_s is not None:
            parts.append(f", {coinc.time_difference_s:+.2f} s")
        blocks.append(_context("".join(parts) + ". This message reports the GW-only localization."))
    return blocks


# ---------------------------------------------------------------- localization


def _localization_blocks(trigger: Trigger) -> list[dict[str, Any]]:
    if trigger.localization_error:
        return [_context(f":warning: {trigger.localization_error}")]
    if trigger.localization is None or trigger.geometry is None:
        return [_context(":warning: This notice carries no sky map yet.")]

    geo = trigger.geometry
    loc = trigger.localization
    lines = [f"*Localization* ({loc.credible_level:.0%} credible region)"]
    lines.append(
        f"Area {_fmt(geo.area_deg2, ',.0f', ' deg^2')}  |  "
        f"centroid RA {_fmt(geo.centroid_ra_deg, '.2f')}, Dec {_fmt(geo.centroid_dec_deg, '.2f')}"
    )
    lines.append(
        f"Dec range {_fmt(geo.dec_min_deg, '.1f')} to {_fmt(geo.dec_max_deg, '.1f')} deg  |  "
        f"|b| {_fmt(geo.gal_b_abs_min_deg, '.1f')} to {_fmt(geo.gal_b_abs_max_deg, '.1f')} deg"
    )
    if trigger.distance_mean_mpc is not None:
        lines.append(
            f"Distance {trigger.distance_mean_mpc:,.0f} +/- " f"{_fmt(trigger.distance_std_mpc, ',.0f')} Mpc"
        )
    if np.isfinite(geo.area_deg2) and geo.area_deg2 > 0:
        pointings = geo.area_deg2 / DECAM_FOV_DEG2
        lines.append(
            f"_~{pointings:,.0f} DECam pointings ({DECAM_FOV_DEG2:g} deg^2 each) to tile this region._"
        )
    if np.isfinite(geo.dec_min_deg) and geo.dec_min_deg > POORLY_PLACED_DEC_DEG:
        lines.append(
            f":warning: Entire region is north of Dec +{POORLY_PLACED_DEC_DEG:g}; poorly placed for a "
            "southern site like CTIO."
        )
    return [_section("\n".join(lines))]


# ------------------------------------------------------------- observability


def _observability_blocks(
    trigger: Trigger,
    events: NightEvents,
    dark_hours: DarkHoursMap,
    dark_stats: dict[str, float] | None,
    config: Config,
) -> list[dict[str, Any]]:
    def span(start, end, hours: float, label: str) -> str:
        return (
            f"{label}: {format_time(start)} [{format_time(start, CTIO_TZ)}]"
            f"  ->  {format_time(end)} [{format_time(end, CTIO_TZ)}]"
            f"  ({hours:.2f} h)"
        )

    sun_lines = [
        f"*Night of {events.day_obs}* (times UTC, local Chile in brackets)",
        f"Sunset {format_time(events.sunset)} [{format_time(events.sunset, CTIO_TZ)}]  ->  "
        f"sunrise {format_time(events.sunrise)} [{format_time(events.sunrise, CTIO_TZ)}]",
        span(events.sun_n12_setting, events.sun_n12_rising, events.night_length_hours, "Sun < -12 deg"),
        span(events.sun_n18_setting, events.sun_n18_rising, events.dark_length_hours, "Sun < -18 deg"),
    ]

    moon_lines = [
        f"*Moon:* {events.moon_illumination:.0%} illuminated",
        f"Moonrise {format_time(events.moonrise)} [{format_time(events.moonrise, CTIO_TZ)}]  |  "
        f"moonset {format_time(events.moonset)} [{format_time(events.moonset, CTIO_TZ)}]",
    ]
    if events.moonrise is None and events.moonset is None:
        moon_lines.append("_The Moon neither rises nor sets during this night._")
    if trigger.geometry is not None:
        moon_sep = moon_separation_deg(
            trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg, events
        )
        moon_lines.append(f"Separation from localization centroid: {_fmt(moon_sep, '.1f', ' deg')}")

    dark_lines = [
        f"*Accessible dark hours at CTIO* (airmass < {dark_hours.airmass_limit:g}, i.e. altitude > "
        f"{dark_hours.altitude_limit_deg:.0f} deg, while Sun < {dark_hours.sun_alt_limit_deg:g} deg)",
        f"{dark_hours.window_hours:.2f} h available tonight",
    ]
    if dark_stats is not None:
        within = (
            f"Within the localization: max {_fmt(dark_stats.get('max_hours'), '.2f', ' h')}, "
            f"mean {_fmt(dark_stats.get('mean_hours'), '.2f', ' h')}"
        )
        weighted = dark_stats.get("weighted_mean_hours")
        if weighted is not None and np.isfinite(weighted):
            within += f", probability-weighted mean {weighted:.2f} h"
        dark_lines.append(within)
        dark_lines.append(
            f"{dark_stats.get('fraction_accessible', 0.0):.0%} of the region reaches airmass < "
            f"{dark_hours.airmass_limit:g} at some point during this night"
        )
        if dark_stats.get("fraction_accessible", 0.0) == 0.0:
            dark_lines.append(
                ":warning: No part of the localization reaches the airmass limit during this night."
            )

    links = site_links(config.links)
    blocks = [
        _section("\n".join(sun_lines)),
        _section("\n".join(moon_lines)),
        _section("\n".join(dark_lines)),
    ]
    if links:
        blocks.append(_section("*Site conditions*\n" + format_link_list(links)))
    return blocks


# -------------------------------------------------------------------- public


def build_trigger_blocks(
    trigger: Trigger,
    events: NightEvents | None,
    dark_hours: DarkHoursMap | None,
    dark_stats: dict[str, float] | None,
    config: Config,
) -> list[dict[str, Any]]:
    """Build the Block Kit payload for a notice.

    Parameters
    ----------
    trigger : `Trigger`
        The decoded (and possibly chirp-mass-enriched) trigger.
    events : `NightEvents`, optional
        Sun and Moon events for the relevant night. None for a retraction.
    dark_hours : `DarkHoursMap`, optional
        Accessible dark hours map. None for a retraction.
    dark_stats : `dict`, optional
        Output of `decam_chatterbox.plots.darkhours.region_hours_summary`,
        when a localization was available.
    config : `Config`
        Configuration, for links.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    blocks: list[dict[str, Any]] = []
    blocks += _header_blocks(trigger)
    blocks.append(_divider())

    if trigger.is_retraction:
        blocks += _retraction_blocks(trigger)
        return blocks

    blocks += _significance_blocks(trigger)
    blocks.append(_divider())
    blocks += _localization_blocks(trigger)
    if events is not None and dark_hours is not None:
        blocks.append(_divider())
        blocks += _observability_blocks(trigger, events, dark_hours, dark_stats, config)

    footer = []
    if trigger.localization is not None:
        footer.append(f"Localization source: {trigger.localization.provenance}.")
    footer.append(f"Notice created {trigger.time_created}.")
    footer.append(
        "This message reports the alert as received; it makes no observing-strategy recommendation."
    )
    link = gracedb_link(trigger.superevent_id, trigger.gracedb_url)
    footer.append(format_link_list([("GraceDB", link)]))
    blocks.append(_context(" ".join(footer)))

    # Slack caps a message at 50 blocks.
    if len(blocks) > 50:
        logger.warning("Trimming %d blocks to Slack's limit of 50", len(blocks))
        blocks = blocks[:49] + [_context("_Message truncated._")]
    return blocks


def build_failure_blocks(
    what: str,
    error: BaseException | str,
    source: str = "",
    origin: str = "",
    log_tail: str = "",
) -> list[dict[str, Any]]:
    """Blocks announcing that decam-chatterbox itself failed.

    A GW alert is time-critical, so silence is the worst outcome: an alert
    nobody hears about looks exactly like no alert at all. This says what
    broke where, in enough detail to start on it without going to the host
    first.

    Parameters
    ----------
    what : `str`
        What was being attempted, as a phrase: ``"handling an alert notice"``.
    error : `BaseException` or `str`
        The failure. An exception contributes its type as well as its message.
    source : `str`
        Event identifier, when it is known. Decoding is one of the things
        that can fail, so it often is not.
    origin : `str`
        Where the record came from -- a file path, a Kafka topic.
    log_tail : `str`
        Last lines of a relevant log, included verbatim.

    Returns
    -------
    blocks : `list` [`dict`]
    """
    if isinstance(error, BaseException):
        detail = f"{type(error).__name__}: {error}"
    else:
        detail = str(error)

    header = ":rotating_light: *decam-chatterbox failed while " + what + "*"
    lines = [header]
    if source:
        lines.append(f"Event: *{source}*")
    if origin:
        lines.append(f"Received from: `{origin}`")
    lines.append(f"```{detail[:1500]}```")
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": (f"decam-chatterbox failure: {source}" if source else "decam-chatterbox failure")[
                    :150
                ],
            },
        },
        _section("\n".join(lines)),
    ]
    if log_tail.strip():
        blocks.append(_section("*Log tail*\n```" + log_tail.strip()[-1500:] + "```"))
    blocks.append(
        _context(
            "Posted by decam-chatterbox's own error handler, so a failed alert is not silent. "
            "The service log has the full traceback."
        )
    )
    return blocks


def plain_text_summary(trigger: Trigger) -> str:
    """Short fallback text, for notifications and refused blocks."""
    prefix = "" if trigger.is_real else "[NOT REAL] "
    if trigger.is_retraction:
        return f"{prefix}{trigger.superevent_id}: RETRACTED"
    area = f"{trigger.geometry.area_deg2:,.0f} deg^2" if trigger.geometry else "no localization"
    return f"{prefix}{trigger.superevent_id} ({trigger.alert_type}): {area}"
