"""The decam-chatterbox service: turn an LVK alert notice into a Slack post.

Unlike chatterbox's two-stage Rubin pipeline -- an immediate post, then a
scheduler-simulation reply threaded onto it -- the alert itself is one stage
here: decode the notice, compute CTIO observability, post. Chirp-mass
enrichment runs concurrently with the almanac and dark-hours work so it does
not add to the critical path. The one thing that follows later is the
animated airmass map of night zero, which takes tens of seconds to render and
so is drawn on a background thread and posted in the alert's thread once
ready.

Failures are contained. If the dark-hours plot cannot be rendered, the text
still goes out; if chirp-mass enrichment fails, the post just omits it; if the
animation fails, the alert has already gone out without it.
"""

import logging
import re
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.time import Time

from .astro.almanac import NightEvents, night_events
from .astro.darkhours import DarkHoursMap, dark_hours_map
from .astro.visibility import FieldVisibility, field_visibility
from .config import Config
from .heartbeat import Heartbeat, ServiceStatus, post_shutdown, post_startup
from .ingest.decode import decode_notice
from .ingest.enrich_gracedb import enrich_chirp_mass
from .models import Trigger
from .plots.darkhours import plot_dark_hours, region_hours_summary
from .priority import PriorityAssessment, assess_priority
from .slackbot.blocks import build_failure_blocks, build_trigger_blocks, plain_text_summary
from .slackbot.client import PostedMessage, SlackPoster

__all__ = ["TriggerReport", "process_notice", "post_failure", "run_service", "wait_for_background"]

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
    #: When the field is up during the night; None for a retraction, a notice
    #: with no localization, or when the calculation failed.
    visibility: FieldVisibility | None = None
    plots: list[Path] = field(default_factory=list)
    posted: PostedMessage | None = None
    #: Set only when `priority.is_high_priority` and
    #: `SlackConfig.urgent_channel` is configured.
    posted_urgent: PostedMessage | None = None
    elapsed_s: float = 0.0
    #: Non-fatal problems encountered, for reporting in the CLI.
    warnings: list[str] = field(default_factory=list)
    #: The thread rendering and posting the airmass animation, and the GIF it
    #: wrote. The thread is a daemon, so it has to be joined before the
    #: process exits or the animation is never posted.
    animation_thread: threading.Thread | None = None
    animation: Path | None = None

    @property
    def has_background_work(self) -> bool:
        """True while the airmass animation is still being rendered or
        posted."""
        return self.animation_thread is not None and self.animation_thread.is_alive()

    def wait_for_background(self, timeout: float | None = None) -> bool:
        """Block until the airmass animation has been posted.

        Parameters
        ----------
        timeout : `float`, optional
            Seconds to wait. None waits indefinitely.

        Returns
        -------
        done : `bool`
            False when it is still running after `timeout`.
        """
        if self.animation_thread is None:
            return True
        self.animation_thread.join(timeout=timeout)
        return not self.animation_thread.is_alive()


def process_notice(
    record: dict[str, Any],
    config: Config,
    poster: SlackPoster | None = None,
    post: bool = True,
    out_dir: Path | None = None,
    run_animation: bool = True,
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
        Where the dark-hours plot and the airmass animation are written.
        Defaults to a per-source work directory.
    run_animation : `bool`
        Render the airmass animation of night zero and post it in the
        alert's thread, when ``airmass_animation.enabled`` and the alert is
        actually being posted. It runs on a background thread; see
        `TriggerReport.wait_for_background`.

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
    visibility: FieldVisibility | None = None
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
            # When the field is up, not just for how long. A failure here
            # costs that one section, not the alert.
            try:
                visibility = field_visibility(
                    events,
                    trigger.localization,
                    dark_hours,
                    trigger.geometry.peak_ra_deg,
                    trigger.geometry.peak_dec_deg,
                    step_minutes=config.dark_hours.step_minutes,
                )
            except Exception as exc:
                logger.error("Could not work out when the field is visible: %s", exc)
                warnings.append(f"visibility windows failed: {exc}")
            try:
                plots.append(
                    plot_dark_hours(
                        dark_hours,
                        trigger.localization,
                        out_dir / f"{trigger.superevent_id}_darkhours.png",
                        peak=(trigger.geometry.peak_ra_deg, trigger.geometry.peak_dec_deg),
                    )
                )
            except Exception as exc:
                logger.error("Could not render the dark-hours plot: %s", exc)
                warnings.append(f"dark-hours plot failed: {exc}")

    blocks = build_trigger_blocks(
        trigger, events, dark_hours, dark_stats, config, priority=priority, visibility=visibility
    )
    text = plain_text_summary(trigger, priority)

    report = TriggerReport(
        trigger=trigger,
        events=events,
        dark_hours=dark_hours,
        dark_stats=dark_stats,
        priority=priority,
        blocks=blocks,
        text=text,
        visibility=visibility,
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

        # Started only now, so it never delays the alert. Posted once, under
        # the main-channel message: the urgent cross-post is for getting
        # attention, and the thread with the detail is the original.
        if (
            run_animation
            and config.airmass_animation.enabled
            and events is not None
            and trigger.localization is not None
        ):
            _start_animation(trigger, config, events, poster, report, out_dir)

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


def _start_animation(
    trigger: Trigger,
    config: Config,
    events: NightEvents,
    poster: SlackPoster,
    report: TriggerReport,
    out_dir: Path,
) -> None:
    """Render the airmass animation off the main thread and post it in-thread.

    A night is dozens of frames and takes tens of seconds, so it cannot be
    part of the first post. It is posted as a reply to it instead -- or
    standalone, when the first post could not be threaded onto (it failed,
    or it was written offline).
    """
    from .plots.airmass_animation import plot_airmass_animation
    from .slackbot.blocks import build_airmass_animation_blocks

    settings = config.airmass_animation
    sun_alt_limit_deg = config.dark_hours.sun_alt_limit_deg
    # Two notices of the same type for one event (a pair of Preliminaries,
    # say) can arrive seconds apart, so the creation time is in the name: the
    # second render must not overwrite the first while it is being uploaded.
    created = re.sub(r"[^0-9A-Za-z]", "", str(trigger.time_created))
    label = f"{trigger.superevent_id}_{trigger.alert_type.lower()}_{created}_airmass"
    geo = trigger.geometry
    peak = (geo.peak_ra_deg, geo.peak_dec_deg) if geo is not None else None

    def render_and_post() -> None:
        # A background thread's exceptions are invisible, so everything is
        # caught and logged here: a missing animation must never take the
        # alert path down, and has no right to vanish silently either.
        try:
            path = plot_airmass_animation(
                events,
                trigger.localization,
                out_dir / f"{label}.gif",
                mask_above=settings.mask_above,
                step_minutes=settings.step_minutes,
                sun_alt_limit_deg=sun_alt_limit_deg,
                peak=peak,
                fps=settings.fps,
                dpi=settings.dpi,
            )
            report.animation = path
        except Exception as exc:
            logger.error("Could not render the airmass animation for %s: %s", trigger.superevent_id, exc)
            report.warnings.append(f"airmass animation failed: {exc}")
            return

        try:
            blocks = build_airmass_animation_blocks(trigger, events, settings.mask_above, sun_alt_limit_deg)
            text = f"Airmass at CTIO through night zero for {trigger.superevent_id}"
            if report.posted is not None and report.posted.ts is not None:
                poster.reply(report.posted, blocks, text, files=[path], label=label)
            else:
                poster.post(blocks, text, is_test=not trigger.is_real, files=[path], label=label)
        except Exception as exc:
            logger.error("Could not post the airmass animation for %s: %s", trigger.superevent_id, exc)
            report.warnings.append(f"airmass animation post failed: {exc}")

    thread = threading.Thread(target=render_and_post, name=f"airmass-{trigger.superevent_id}", daemon=True)
    thread.start()
    report.animation_thread = thread


def wait_for_background(reports: list[TriggerReport], timeout: float) -> list[TriggerReport]:
    """Give outstanding airmass animations a bounded time to finish.

    Parameters
    ----------
    reports : `list` [`TriggerReport`]
        Reports that may still have an animation rendering or posting.
    timeout : `float`
        Seconds to wait for all of them together.

    Returns
    -------
    abandoned : `list` [`TriggerReport`]
        Those still running when the time ran out.
    """
    deadline = time.monotonic() + max(timeout, 0.0)
    for report in reports:
        report.wait_for_background(timeout=max(deadline - time.monotonic(), 0.0))
    return [r for r in reports if r.has_background_work]


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


def _system_stop_signals() -> list[int]:
    """Signals the system sends to ask a service to stop, where they exist."""
    return [getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)]


def _install_stop_signal_handlers() -> dict[int, Any]:
    """Make SIGTERM and SIGHUP behave exactly like Ctrl-C.

    Both then converge on the one shutdown path in `run_service`. SIGTERM is
    what `kill`, systemd and Docker send; SIGHUP is what a closed ssh/tmux
    session or a host shutting down sends to a process it started, and its
    default action kills the process instantly with no chance to say
    anything. A SIGHUP that was deliberately ignored (``nohup``) is left
    ignored: the operator asked for the process to survive that.

    Only possible from the main thread; harmless to skip elsewhere, and Ctrl-C
    still works either way.

    Returns
    -------
    previous : `dict` [`int`, `Any`]
        Handlers that were replaced, for `_restore_signal_handlers`.
    """
    previous: dict[int, Any] = {}
    for signum in _system_stop_signals():
        try:
            before = signal.getsignal(signum)
            if signum == getattr(signal, "SIGHUP", None) and before == signal.SIG_IGN:
                continue
            signal.signal(signum, signal.default_int_handler)
            previous[signum] = before
        except ValueError:
            logger.debug("Could not install a handler for signal %s (not the main thread)", signum)
            break
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler if handler is not None else signal.SIG_DFL)
        except ValueError:
            return


def _ignore_system_stop_signals() -> None:
    for signum in _system_stop_signals():
        try:
            signal.signal(signum, signal.SIG_IGN)
        except ValueError:
            return


def run_service(config: Config, paths=None, run_animation: bool = True) -> int:
    """Consume the configured source and post about every notice.

    Runs until interrupted (Ctrl-C, or SIGTERM/SIGHUP, which are installed to
    behave like Ctrl-C for the duration of this call) or until the source
    raises. A heartbeat posts every ``heartbeat.interval_s`` on its own
    thread so a hang or a silently-dead broker connection is visible even
    between alerts, and a deliberate stop posts its own notice rather than
    just going quiet.

    Airmass animations render on their own threads and never hold up the
    next alert. On the way out, any still in progress get up to
    ``airmass_animation.shutdown_wait_s`` to finish -- after the shutdown
    notice, so a service manager's stop timeout cannot cost that instead.

    Parameters
    ----------
    config : `Config`
        Configuration.
    paths : sequence, optional
        Explicit record paths, forcing replay mode.
    run_animation : `bool`
        Render and post the airmass animation under each alert.

    Returns
    -------
    count : `int`
        Records handled.
    """
    from .ingest.source import make_source

    previous_handlers = _install_stop_signal_handlers()

    source = make_source(config, paths=paths)
    poster = SlackPoster(config)
    status = ServiceStatus(started_at=Time.now(), origin=_describe_source(source, config))
    heartbeat = Heartbeat(config, poster, status)
    heartbeat.start()
    post_startup(status, config, poster)

    handled = 0
    pending: list[TriggerReport] = []
    try:
        for record, metadata in source:
            origin = metadata.get("origin") or metadata.get("topic") or metadata.get("transport") or ""
            try:
                report = process_notice(record, config, poster=poster, run_animation=run_animation)
                pending = [r for r in pending if r.has_background_work]
                if report.has_background_work:
                    pending.append(report)
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
        # A host going down sends SIGHUP and/or SIGTERM in quick succession;
        # a second one arriving mid-post would raise out of the Slack call
        # and lose the very message this exists to send. Ctrl-C is left
        # alone, as the way to abandon a post that is hanging.
        _ignore_system_stop_signals()
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
        _finish_animations(pending, config.airmass_animation.shutdown_wait_s)
        heartbeat.stop()
        source.close()
        _restore_signal_handlers(previous_handlers)
    return handled


def _finish_animations(pending: list[TriggerReport], timeout: float) -> None:
    """Let animations still in progress finish before exiting, within reason.

    They run on daemon threads, which die with the process: without this, a
    stop seconds after an alert would silently drop its animation. Bounded,
    and a further interrupt abandons the wait rather than the rest of the
    shutdown.
    """
    pending = [r for r in pending if r.has_background_work]
    if not pending:
        return
    names = ", ".join(r.trigger.superevent_id for r in pending)
    logger.info("Waiting up to %.0f s for the airmass animation(s) of %s", timeout, names)
    try:
        abandoned = wait_for_background(pending, timeout)
    except KeyboardInterrupt:
        abandoned = [r for r in pending if r.has_background_work]
    if abandoned:
        logger.warning(
            "Abandoned the airmass animation(s) of %s, still in progress at shutdown",
            ", ".join(r.trigger.superevent_id for r in abandoned),
        )
