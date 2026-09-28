"""The decam-chatterbox service: turn an LVK alert notice into a Slack post.

Unlike chatterbox's two-stage Rubin pipeline -- an immediate post, then a
scheduler-simulation reply threaded onto it -- there is exactly one stage
here: decode the notice, compute CTIO observability, post. Chirp-mass
enrichment runs concurrently with the almanac and dark-hours work so it does
not add to the critical path.

Failures are contained. If the dark-hours plot cannot be rendered, the text
still goes out; if chirp-mass enrichment fails, the post just omits it.
"""

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.time import Time

from .astro.almanac import NightEvents, night_events
from .astro.darkhours import DarkHoursMap, dark_hours_map
from .config import Config
from .heartbeat import Heartbeat, ServiceStatus, post_shutdown
from .ingest.decode import decode_notice
from .ingest.enrich_gracedb import enrich_chirp_mass
from .models import Trigger
from .plots.darkhours import plot_dark_hours, region_hours_summary
from .priority import PriorityAssessment, assess_priority
from .slackbot.blocks import build_failure_blocks, build_trigger_blocks, plain_text_summary
from .slackbot.client import PostedMessage, SlackPoster

__all__ = ["TriggerReport", "process_notice", "post_failure", "run_service"]

logger = logging.getLogger(__name__)


@dataclass
class TriggerReport:
    """Everything computed for one notice, for inspection and tests."""

    trigger: Trigger
    events: NightEvents | None
    dark_hours: DarkHoursMap | None
    dark_stats: dict[str, float] | None
    priority: PriorityAssessment
    blocks: list[dict[str, Any]]
    text: str
    plots: list[Path] = field(default_factory=list)
    posted: PostedMessage | None = None
    #: Set only when `priority.is_high_priority` and
    #: `SlackConfig.urgent_channel` is configured.
    posted_urgent: PostedMessage | None = None
    elapsed_s: float = 0.0
    #: Non-fatal problems encountered, for reporting in the CLI.
    warnings: list[str] = field(default_factory=list)


def process_notice(
    record: dict[str, Any],
    config: Config,
    poster: SlackPoster | None = None,
    post: bool = True,
    out_dir: Path | None = None,
) -> TriggerReport:
    """Handle one alert notice end to end.

    Parameters
    ----------
    record : `dict`
        Decoded notice record.
    config : `Config`
        Configuration.
    poster : `SlackPoster`, optional
        Delivery mechanism. Built from `config` when omitted.
    post : `bool`
        Actually post. When False the blocks are built and returned only.
    out_dir : `pathlib.Path`, optional
        Where the dark-hours plot is written. Defaults to a per-source work
        directory.

    Returns
    -------
    report : `TriggerReport`
    """
    started = time.monotonic()
    warnings: list[str] = []

    skymap_dir = Path(config.paths.work_dir).expanduser() / "skymaps"
    trigger = decode_notice(record, skymap_dir=skymap_dir)
    if trigger.localization_error:
        warnings.append(trigger.localization_error)
    priority = assess_priority(trigger, config.priority)

    if out_dir is None:
        out_dir = Path(config.paths.work_dir).expanduser() / "plots" / trigger.superevent_id
    out_dir.mkdir(parents=True, exist_ok=True)

    events: NightEvents | None = None
    dark_hours: DarkHoursMap | None = None
    dark_stats: dict[str, float] | None = None
    plots: list[Path] = []

    if not trigger.is_retraction:
        # Enrichment is network-bound; the almanac and dark-hours work are
        # CPU-bound. Overlap them rather than paying for both in series.
        with ThreadPoolExecutor(max_workers=1) as pool:
            enrich_future = pool.submit(enrich_chirp_mass, trigger, config.enrich)

            when = trigger.event_time if trigger.event_time is not None else Time.now()
            try:
                events = night_events(when=when)
                dark_hours = dark_hours_map(
                    events,
                    nside=config.dark_hours.nside,
                    step_minutes=config.dark_hours.step_minutes,
                    airmass_limit=config.dark_hours.airmass_limit,
                    sun_alt_limit_deg=config.dark_hours.sun_alt_limit_deg,
                    moon_avoidance_deg=config.dark_hours.moon_avoidance_deg,
                )
            except Exception as exc:
                logger.error("CTIO observability computation failed: %s", exc)
                warnings.append(f"observability computation failed: {exc}")

            try:
                trigger.enrichment = enrich_future.result()
                if trigger.enrichment is not None and trigger.enrichment.error:
                    warnings.append(f"GraceDB enrichment: {trigger.enrichment.error}")
            except Exception as exc:
                logger.error("Enrichment raised: %s", exc)
                warnings.append(f"GraceDB enrichment raised: {exc}")

        if dark_hours is not None and trigger.localization is not None:
            dark_stats = region_hours_summary(dark_hours, trigger.localization)
            try:
                plots.append(
                    plot_dark_hours(
                        dark_hours,
                        trigger.localization,
                        out_dir / f"{trigger.superevent_id}_darkhours.png",
                        centroid=(trigger.geometry.centroid_ra_deg, trigger.geometry.centroid_dec_deg),
                    )
                )
            except Exception as exc:
                logger.error("Could not render the dark-hours plot: %s", exc)
                warnings.append(f"dark-hours plot failed: {exc}")

    blocks = build_trigger_blocks(trigger, events, dark_hours, dark_stats, config, priority=priority)
    text = plain_text_summary(trigger, priority)

    report = TriggerReport(
        trigger=trigger,
        events=events,
        dark_hours=dark_hours,
        dark_stats=dark_stats,
        priority=priority,
        blocks=blocks,
        text=text,
        plots=plots,
        warnings=warnings,
    )

    if post:
        poster = poster or SlackPoster(config)
        label = f"{trigger.superevent_id}_{trigger.alert_type.lower()}"
        try:
            report.posted = poster.post(
                blocks,
                text,
                is_test=not trigger.is_real,
                files=plots,
                label=label,
            )
        except Exception as exc:
            logger.error("Posting to Slack failed: %s", exc)
            warnings.append(f"Slack post failed: {exc}")

        # A cross-post, not a replacement: the alert already reached the main
        # channel above regardless of priority, so losing this one only loses
        # the extra visibility, not the alert itself.
        if priority.is_high_priority and config.slack.urgent_channel:
            try:
                report.posted_urgent = poster.post(
                    blocks,
                    text,
                    channel=config.slack.urgent_channel,
                    mention=True,
                    files=plots,
                    label=f"{label}_urgent",
                )
            except Exception as exc:
                logger.error("Posting the urgent cross-post failed: %s", exc)
                warnings.append(f"urgent-channel post failed: {exc}")

    report.elapsed_s = time.monotonic() - started
    logger.info(
        "%s (%s) handled in %.2f s (%d plot(s), %d warning(s))",
        trigger.superevent_id,
        trigger.alert_type,
        report.elapsed_s,
        len(plots),
        len(warnings),
    )
    return report


def post_failure(
    what: str,
    error: BaseException | str,
    poster: SlackPoster | None,
    source: str = "",
    origin: str = "",
    log_tail: str = "",
    is_test: bool = False,
) -> PostedMessage | None:
    """Tell the channel that decam-chatterbox failed. Never raises.

    The whole point is to be reached from an ``except`` block, so it cannot
    add a second failure on top of the first: everything here, including
    building the blocks, is inside its own guard.

    Parameters
    ----------
    what : `str`
        What was being attempted, as a phrase.
    error : `BaseException` or `str`
        The failure to report.
    poster : `SlackPoster` or None
        Delivery. None -- a dry run with no poster -- logs and returns.
    source : `str`
        Event identifier, when known.
    origin : `str`
        Where the record came from.
    log_tail : `str`
        Last lines of a relevant log.
    is_test : `bool`
        Route to the test channel, when one is configured.

    Returns
    -------
    posted : `PostedMessage` or None
        None when there was no poster, or when posting the failure also
        failed.
    """
    logger.error("Failure while %s (%s): %s", what, source or "no event", error, exc_info=True)
    if poster is None:
        return None
    label = f"{source or 'decam-chatterbox'}_failure"
    try:
        blocks = build_failure_blocks(what, error, source=source, origin=origin, log_tail=log_tail)
        summary = f"decam-chatterbox failed while {what}"
        if source:
            summary += f" ({source})"
        return poster.post(blocks, summary, is_test=is_test, label=label)
    except Exception as exc:
        # Nothing left to do but say so in the log: the channel is exactly
        # what is not working.
        logger.error("Could not post the failure notice either: %s", exc)
        return None


def _describe_source(source: Any, config: Config) -> str:
    """Where a source is watching, for a reader who missed the startup log."""
    describe = getattr(source, "describe", None)
    return describe() if callable(describe) else config.ingest.kind


def run_service(config: Config, paths=None) -> int:
    """Consume the configured source and post about every notice.

    Runs until interrupted (Ctrl-C or SIGTERM, which is installed to behave
    like Ctrl-C for the duration of this call) or until the source raises. A
    heartbeat posts every ``heartbeat.interval_s`` on its own thread so a hang
    or a silently-dead broker connection is visible even between alerts, and
    a deliberate stop posts its own notice rather than just going quiet.

    Parameters
    ----------
    config : `Config`
        Configuration.
    paths : sequence, optional
        Explicit record paths, forcing replay mode.

    Returns
    -------
    count : `int`
        Records handled.
    """
    from .ingest.source import make_source

    try:
        # Makes `kill` (and a systemd/Docker stop, which sends SIGTERM by
        # default) raise KeyboardInterrupt exactly as Ctrl-C does, so both
        # converge on the one shutdown path below. Only possible from the
        # main thread; harmless to skip elsewhere; Ctrl-C still works either
        # way.
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    except ValueError:
        logger.debug("Could not install a SIGTERM handler (not the main thread)")

    source = make_source(config, paths=paths)
    poster = SlackPoster(config)
    status = ServiceStatus(started_at=Time.now(), origin=_describe_source(source, config))
    heartbeat = Heartbeat(config, poster, status)
    heartbeat.start()

    handled = 0
    try:
        for record, metadata in source:
            origin = metadata.get("origin") or metadata.get("topic") or metadata.get("transport") or ""
            try:
                process_notice(record, config, poster=poster)
                handled += 1
                status.handled = handled
                status.last_handled_at = Time.now()
            except Exception as exc:
                # A dropped alert must never be silent. `record` may not have
                # decoded, so its fields are read defensively -- the
                # identifier is the one worth trying for.
                identifier = record.get("superevent_id") if isinstance(record, dict) else None
                post_failure(
                    "handling an alert notice",
                    exc,
                    poster,
                    source=str(identifier or ""),
                    origin=str(origin),
                )
            finally:
                # Acknowledge either way: a record that cannot be processed
                # will not become processable on redelivery, and blocking the
                # stream on it would stall every later alert.
                source.mark_done(metadata)
    except KeyboardInterrupt:
        post_shutdown(status, config, poster, reason="interrupted")
    except Exception as exc:
        # The stream itself failed -- the broker dropped us, credentials
        # expired. The service is about to stop consuming alerts, which is
        # precisely the thing nobody would otherwise notice.
        post_failure(
            f"monitoring for alerts ({config.ingest.kind})",
            exc,
            poster,
            origin=_describe_source(source, config),
        )
        raise
    finally:
        heartbeat.stop()
        source.close()
    return handled
