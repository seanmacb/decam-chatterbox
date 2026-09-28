"""Transports that deliver LVK alert notices to decam-chatterbox.

- `ScimmaAlertSource` subscribes to ``igwn.gwalert`` on SCIMMA's Hopskotch
  broker with hop-client. This is the default and the one live monitoring
  path; see the README for account and topic-subscription setup.
- `FileAlertSource` watches a directory for ``.json``/``.avro`` notice files.
- `ReplaySource` yields explicit paths once and stops.

Every source yields ``(record, metadata)`` and exposes ``mark_done`` so a
Kafka offset is only committed after a record has been fully handled.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import Config, IngestConfig

__all__ = [
    "AlertSource",
    "ScimmaAlertSource",
    "FileAlertSource",
    "ReplaySource",
    "make_source",
]

logger = logging.getLogger(__name__)


class AlertSource(ABC):
    """A stream of raw LVK notice records."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield ``(record, metadata)`` pairs."""

    def mark_done(self, metadata: dict[str, Any]) -> None:
        """Acknowledge a record. Default is a no-op."""

    def close(self) -> None:
        """Release any resources."""


class ReplaySource(AlertSource):
    """Yield records from explicit file paths, then stop.

    Parameters
    ----------
    paths : sequence of `str` or `pathlib.Path`
        Record files to read, in order.
    """

    def __init__(self, paths) -> None:
        self.paths = [Path(p).expanduser() for p in paths]

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        from .decode import load_record_file

        for path in self.paths:
            logger.info("Replaying %s", path)
            yield load_record_file(path), {"origin": str(path), "transport": "replay"}


class FileAlertSource(AlertSource):
    """Watch a directory for notice files, e.g. saved by hand for testing.

    Parameters
    ----------
    watch_dir : `str`
        Directory to poll.
    poll_interval_s : `float`
        Seconds between scans.
    once : `bool`
        Process the files already present and return, instead of polling
        forever.

    Notes
    -----
    Files are tracked by ``(path, mtime, size)`` so a rewritten file (e.g. an
    UPDATE notice saved over a PRELIMINARY one under the same name) is picked
    up again rather than being mistaken for one already handled.
    """

    def __init__(self, watch_dir: str, poll_interval_s: float = 2.0, once: bool = False) -> None:
        self.watch_dir = Path(watch_dir).expanduser()
        self.poll_interval_s = poll_interval_s
        self.once = once
        self._seen: set[tuple[str, float, int]] = set()

    def _scan(self) -> list[Path]:
        if not self.watch_dir.is_dir():
            return []
        found = []
        for path in sorted(self.watch_dir.iterdir()):
            if path.suffix not in (".json", ".avro"):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (str(path), stat.st_mtime, stat.st_size)
            if key in self._seen:
                continue
            self._seen.add(key)
            found.append(path)
        return found

    def describe(self) -> str:
        return f"{self.watch_dir} (files)"

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        from .decode import load_record_file

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Watching %s for alert notices", self.watch_dir)
        while True:
            for path in self._scan():
                try:
                    record = load_record_file(path)
                except Exception as exc:
                    logger.error("Could not read %s: %s", path, exc)
                    continue
                yield record, {"origin": str(path), "transport": "files"}
            if self.once:
                return
            time.sleep(self.poll_interval_s)


class ScimmaAlertSource(AlertSource):
    """Subscribe to ``igwn.gwalert`` on SCIMMA's Hopskotch broker.

    Parameters
    ----------
    url : `str`
        hop URL, e.g. ``kafka://kafka.scimma.org/igwn.gwalert``.
    group_id : `str`
        Consumer group, kept stable across restarts so Kafka remembers the
        offset.

    Notes
    -----
    Authentication is delegated entirely to hop-client's own credential
    lookup (``~/.config/hop/auth.toml``, set up with ``hop auth add``).
    ``ignoretest`` is left at its default (nothing is filtered at the hop
    envelope level): a mock or test *gravitational-wave* event is not the
    same thing as a hop-level test message, and is instead identified from
    the decoded record's own ``superevent_id`` prefix -- see
    `decam_chatterbox.models.Trigger.is_real`.
    """

    def __init__(self, url: str, group_id: str = "") -> None:
        if not url:
            raise ValueError("No hop URL configured. Set ingest.hop_url.")
        self.url = url
        #: Empty means "derive one from the authenticated credential" --
        #: see `_default_group_id`. Resolved lazily in `_open`, not here,
        #: since it needs to read `~/.config/hop/auth.toml`.
        self.group_id = group_id
        self._stream = None

    def _open(self):
        try:
            import hop
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("SCIMMA ingest requires hop-client (pip install hop-client)") from exc
        # Cached back onto self.group_id so describe() reflects what was
        # actually used, including after this resolves it.
        self.group_id = self.group_id or _default_group_id(self.url)
        logger.info("Opening %s as group %s", self.url, self.group_id)
        return hop.io.Stream(auth=True).open(self.url, mode="r", group_id=self.group_id)

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        self._stream = self._open()
        for message, metadata in self._stream.read(metadata=True, autocommit=False):
            for record in _unwrap(message):
                meta = {
                    "transport": "scimma",
                    "topic": getattr(metadata, "topic", None),
                    "offset": getattr(metadata, "offset", None),
                    "partition": getattr(metadata, "partition", None),
                    "_raw_metadata": metadata,
                }
                yield record, meta

    def mark_done(self, metadata: dict[str, Any]) -> None:
        """Commit the offset for a handled record."""
        raw = metadata.get("_raw_metadata")
        if self._stream is not None and raw is not None:
            self._stream.mark_done(raw)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def describe(self) -> str:
        return f"{self.url} (group {self.group_id or 'not yet resolved'})"


def _default_group_id(url: str) -> str | None:
    """A stable consumer group id derived from the authenticated credential.

    SCIMMA's Kafka ACLs authorize a consumer group only when it matches the
    credential's own username (or is prefixed with it): an arbitrary fixed
    string is rejected with ``GROUP_AUTHORIZATION_FAILED``, a Kafka-level
    authorization error distinct from a topic permission problem. hop-client
    itself only ever falls back to a *randomly suffixed* group id when none
    is given (see ``hop.io._generate_group_id``), which would satisfy the
    same ACL but changes on every restart, so Kafka can never resume from
    where a previous run left off. Deriving it once, deterministically, from
    the credential that ``hop auth add`` already stored keeps both properties.

    Returns
    -------
    group_id : `str` or None
        None when no credential can be resolved (no ``~/.config/hop/auth.toml``
        yet, or more than one credential matches), in which case the caller
        should fall back to hop-client's own random default and the operator
        should set ``ingest.hop_group_id`` explicitly once they know their
        SCIMMA username.
    """
    try:
        from adc.kafka import parse_kafka_url
        from hop.auth import load_auth, select_matching_auth

        _, broker_addresses, _ = parse_kafka_url(url)
        credential = select_matching_auth(load_auth(), broker_addresses[0])
    except Exception as exc:
        logger.warning(
            "Could not derive a stable consumer group id from ~/.config/hop/auth.toml (%s); "
            "hop-client will generate a random one instead, so a restart will not resume "
            "from where it left off. Set ingest.hop_group_id explicitly to avoid this.",
            exc,
        )
        return None
    return f"{credential.username}-decam-chatterbox"


def _unwrap(message: Any) -> list[dict[str, Any]]:
    """Extract record dicts from a hop message.

    hop wraps payloads in model objects whose ``content`` is either a single
    record or a list of them -- for ``igwn.gwalert`` specifically, the LVK
    user guide's own sample code reads ``message.content[0]``.
    """
    content = getattr(message, "content", message)
    if isinstance(content, (bytes, bytearray)):
        return [json.loads(content)]
    if isinstance(content, str):
        return [json.loads(content)]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [c for c in content if isinstance(c, dict)]
    logger.error("Cannot interpret message content of type %s", type(content))
    return []


def make_source(config: Config, paths=None) -> AlertSource:
    """Build the ingest source described by the configuration.

    Parameters
    ----------
    config : `Config`
        Loaded configuration.
    paths : sequence, optional
        Explicit record paths. Forces a `ReplaySource` regardless of
        ``ingest.kind``.

    Returns
    -------
    source : `AlertSource`
    """
    if paths:
        return ReplaySource(paths)

    ingest: IngestConfig = config.ingest
    kind = ingest.kind.lower()
    if kind == "scimma":
        return ScimmaAlertSource(url=ingest.hop_url, group_id=ingest.hop_group_id)
    if kind == "files":
        return FileAlertSource(ingest.watch_dir, ingest.poll_interval_s)
    if kind == "replay":
        raise ValueError("ingest.kind='replay' requires explicit paths")
    raise ValueError(f"Unknown ingest.kind {ingest.kind!r}; expected 'scimma', 'files' or 'replay'")
