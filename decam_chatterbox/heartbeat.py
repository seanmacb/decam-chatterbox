"""A periodic "I'm still running" post, and a message when the service stops.

A decode or Slack failure already posts to the channel, but only because
something raised. A service that has hung, or whose ingest source has quietly
stopped delivering without erroring -- a broker that silently dropped the
connection, say -- would otherwise say nothing at all, and that silence is
indistinguishable from a quiet night with no alerts. `Heartbeat` rules that
out for the `serve` command; `post_shutdown` does the equivalent for a
deliberate stop (Ctrl-C, `kill`, a systemd restart), so the channel can tell
"stopped on purpose" from "crashed" or "hung".

Neither of these applies to `replay`: they exist for the long-running
service, not a one-shot inspection of a handful of files.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

from astropy.time import Time

from .config import Config
from .slackbot.client import SlackPoster

__all__ = ["ServiceStatus", "Heartbeat", "post_shutdown"]

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    """Shared, mutable status the heartbeat and shutdown post report on.

    `run_service` updates `handled`/`last_handled_at` from the main thread as
    records are processed; `Heartbeat` reads them from its own thread. Plain
    attribute reads and writes are enough here -- nothing in either reader or
    writer is a check-then-act sequence that would need a lock.
    """

    started_at: Time
    #: Where the ingest source is watching, e.g. an ``AlertSource.describe()``
    #: or the configured ``ingest.kind``, for the reader who was not there
    #: when the service started.
    origin: str
    handled: int = 0
    last_handled_at: Time | None = None

    @property
    def uptime_s(self) -> float:
        return (Time.now() - self.started_at).sec


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(int(seconds), 0), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _status_text(status: ServiceStatus) -> str:
    last = f", last at {status.last_handled_at.utc.iso} UTC" if status.last_handled_at else ""
    return (
        f"Up {_format_duration(status.uptime_s)}, watching {status.origin}. "
        f"{status.handled} alert(s) handled since starting{last}."
    )


def _status_blocks(header: str, status: ServiceStatus) -> list[dict[str, Any]]:
    return [{"type": "section", "text": {"type": "mrkdwn", "text": f"{header}\n{_status_text(status)}"}}]


class Heartbeat:
    """Posts `_status_blocks` on a background thread, on a fixed interval.

    Parameters
    ----------
    config : `Config`
        Configuration; read for ``heartbeat.enabled``/``interval_s`` and
        ``slack.heartbeat_channel``.
    poster : `SlackPoster`
        Delivery mechanism, shared with the rest of the service.
    status : `ServiceStatus`
        Updated by the caller as alerts are handled.
    """

    def __init__(self, config: Config, poster: SlackPoster, status: ServiceStatus) -> None:
        self.config = config
        self.poster = poster
        self.status = status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background thread, unless ``heartbeat.enabled`` is
        false.
        """
        if not self.config.heartbeat.enabled:
            logger.info("Heartbeat disabled (heartbeat.enabled is false)")
            return
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)
        self._thread.start()
        logger.info("Heartbeat posting every %.0f s", self.config.heartbeat.interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and wait briefly for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        interval = self.config.heartbeat.interval_s
        # Waiting on the Event rather than sleeping means stop() returns
        # immediately instead of blocking for up to a full interval.
        while not self._stop.wait(interval):
            try:
                self._post()
            except Exception as exc:
                # A heartbeat that dies would itself go silent, exactly the
                # failure mode it exists to catch.
                logger.error("Heartbeat post failed: %s", exc)

    def _post(self) -> None:
        blocks = _status_blocks(":heartbeat: *decam-chatterbox heartbeat*", self.status)
        text = f"decam-chatterbox heartbeat: {_status_text(self.status)}"
        self.poster.post(blocks, text, channel=self.config.slack.heartbeat_channel, label="heartbeat")


def post_shutdown(
    status: ServiceStatus,
    config: Config,
    poster: SlackPoster | None,
    reason: str = "interrupted",
) -> None:
    """Tell the channel that decam-chatterbox is stopping, on purpose.

    Never raises, for the same reason `app.post_failure` never does: it is
    called while the process is already on its way out, and a second failure
    there would only be noise.

    Parameters
    ----------
    status : `ServiceStatus`
        Status to report.
    config : `Config`
        Configuration, for ``slack.heartbeat_channel``.
    poster : `SlackPoster` or None
        Delivery. None logs and returns.
    reason : `str`
        Why the service is stopping, e.g. ``"interrupted"`` (Ctrl-C or
        SIGTERM).
    """
    logger.info("Shutting down (%s) after %d alert(s)", reason, status.handled)
    if poster is None:
        return
    try:
        blocks = _status_blocks(f":wave: *decam-chatterbox is shutting down* ({reason})", status)
        text = f"decam-chatterbox shutting down ({reason}): {_status_text(status)}"
        poster.post(blocks, text, channel=config.slack.heartbeat_channel, label="shutdown")
    except Exception as exc:
        logger.error("Could not post the shutdown notice: %s", exc)
