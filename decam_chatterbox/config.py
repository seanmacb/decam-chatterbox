"""Configuration for the decam-chatterbox service.

Settings come from a YAML file, with a small number of environment variable
overrides for anything secret. Secrets are *never* stored in the YAML: the
config only names the environment variable to read them from.
"""

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "SlackConfig",
    "IngestConfig",
    "DarkHoursConfig",
    "EnrichConfig",
    "PriorityConfig",
    "HeartbeatConfig",
    "PathsConfig",
    "LinksConfig",
    "load_config",
    "DEFAULT_CONFIG_PATHS",
]

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = (
    Path("config.yaml"),
    Path("~/.config/decam-chatterbox/config.yaml").expanduser(),
)


@dataclass
class SlackConfig:
    """Slack destination and credentials.

    A bot token is required rather than an incoming webhook because incoming
    webhooks cannot upload files, and every post carries the dark-hours plot.
    """

    bot_token_env: str = "SLACK_BOT_TOKEN"
    channel: str = "#decam-gw-alerts"
    #: Channel used for alerts that are not real superevents (mock/MDC or
    #: test). Falls back to `channel` when empty, in which case they are
    #: still visibly marked in the message itself.
    test_channel: str = ""
    #: Every real and mock alert still posts to `channel` regardless of
    #: `PriorityConfig`; this is an *additional* post, so a high-priority
    #: alert is never missed for being buried in a busier channel. Empty
    #: disables cross-posting.
    urgent_channel: str = ""
    #: Where `HeartbeatConfig` posts go, and the shutdown notice. Falls back
    #: to `channel` when empty. A separate channel keeps routine "still
    #: running" noise out of the one people actually watch for alerts.
    heartbeat_channel: str = ""
    username: str = "decam-chatterbox"
    icon_emoji: str = ":ocean:"
    #: Slack user/group IDs to mention, e.g. ["!subteam^S123"]. Included only
    #: on the `urgent_channel` cross-post of a high-priority real alert -- see
    #: `PriorityConfig` -- not on every post, so the people it pages are not
    #: paged for a routine BBH detection.
    mention: list[str] = field(default_factory=list)


@dataclass
class IngestConfig:
    """Where LVK alert notices are read from.

    ``kind`` selects the transport:

    - ``scimma``: subscribe to ``igwn.gwalert`` on SCIMMA's Hopskotch broker
      with hop-client. This is the default and the one live monitoring path.
    - ``files``: watch a directory for ``.json``/``.avro`` notice files, so
      an alert saved by hand (or by
      `decam_chatterbox.ingest.decode.load_record_file` via some other
      pipeline) can be picked up without Kafka access.
    - ``replay``: read explicit paths once, then stop (used by the CLI).
    """

    kind: str = "scimma"
    #: hop URL. Authentication is delegated entirely to hop-client's own
    #: ``~/.config/hop/auth.toml``, set up with ``hop auth add`` -- see the
    #: README for the SCIMMA account and topic-subscription steps.
    hop_url: str = "kafka://kafka.scimma.org/igwn.gwalert"
    #: Consumer group. Keeping this stable across restarts is what lets Kafka
    #: remember the offset, so a restart does not re-deliver every alert
    #: since the group was first used.
    hop_group_id: str = "decam-chatterbox"
    #: Directory watched by the ``files`` source.
    watch_dir: str = "~/.decam-chatterbox/incoming"
    #: Seconds between directory scans.
    poll_interval_s: float = 2.0
    #: Process notices whose superevent_id is not a real ("S...") superevent,
    #: i.e. the hourly mock/MDC alert both GCN and SCIMMA send, or an
    #: internal ("T...") test event. When True (the default) they are still
    #: posted, routed to `SlackConfig.test_channel` and visibly marked, which
    #: is the easiest way to confirm the whole pipeline works without waiting
    #: for a real gravitational-wave candidate. Set False to drop them.
    allow_mock: bool = True


@dataclass
class DarkHoursConfig:
    """Parameters for the accessible-dark-hours map at CTIO.

    ``sun_alt_limit_deg`` of -12 and ``airmass_limit`` of 2 give hours per
    pixel with airmass < 2 while the Sun is below -12 degrees, the usual
    definition of "the night's useful dark time".
    """

    nside: int = 64
    step_minutes: float = 5.0
    airmass_limit: float = 2.0
    sun_alt_limit_deg: float = -12.0
    #: Optionally exclude pixels close to the Moon. ``0`` disables the cut.
    moon_avoidance_deg: float = 0.0


@dataclass
class EnrichConfig:
    """Optional GraceDB supplement: the binned chirp-mass estimate.

    Everything else this tool reports comes from the notice itself. This is
    best-effort and strictly time-bounded, and runs concurrently with the
    almanac and dark-hours work so it does not add to the critical path.
    """

    gracedb: bool = True
    gracedb_service_url: str = "https://gracedb.ligo.org/api/"
    timeout_s: float = 5.0


@dataclass
class PriorityConfig:
    """Thresholds for flagging a real alert as high priority.

    These never change *what* gets posted -- every real and mock alert still
    reaches `SlackConfig.channel` regardless. They control only the visible
    priority badge on the message, whether `SlackConfig.mention` is included,
    and whether the alert is also cross-posted to `SlackConfig.urgent_channel`.

    An alert is high priority when every applicable threshold below is met.
    "Applicable" matters for classification: a Burst event carries no
    BNS/NSBH/BBH/Terrestrial classification at all, so that one criterion is
    exempted for it rather than automatically failing it. A retraction, and
    anything that is not a real superevent (`Trigger.is_real`), is never
    high priority.
    """

    enabled: bool = True
    #: Maximum false-alarm rate, in Hz. 3.17e-8 Hz is "about 1 per year", the
    #: threshold chatterbox's own Rubin gold/silver GW classes use.
    max_far_hz: float = 3.17e-8
    #: Maximum 90% credible area, in deg^2.
    max_area_deg2: float = 500.0
    #: Minimum p(BNS) + p(NSBH): a merger likely to involve a neutron star,
    #: and so plausibly EM-bright. Exempted for Burst events, which publish
    #: no classification.
    min_ns_classification: float = 0.5


@dataclass
class HeartbeatConfig:
    """A periodic "I'm still running" post, for the ``serve`` command only.

    A failure only posts when something raises; a hung process, or a broker
    that has quietly stopped delivering without erroring, would otherwise say
    nothing at all -- indistinguishable from a quiet night with no alerts.
    This exists to rule that out.
    """

    enabled: bool = True
    #: Seconds between posts. 3600 is hourly.
    interval_s: float = 3600.0


@dataclass
class PathsConfig:
    """Filesystem locations for runtime state."""

    work_dir: str = "~/.decam-chatterbox/work"


@dataclass
class LinksConfig:
    """Permalinks included in every post.

    Everything here is config-driven so the set can change without touching
    code; the defaults are a starting point, not an authoritative source.
    """

    weather: str = "https://www.windy.com/-30.165/-70.815?waves,-30.165,-70.815,9"
    seeing: str = "https://noirlab.edu/science/observing-noirlab/weather-webcams/cerro-tololo"
    almanac: str = ""
    observatory_status: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level decam-chatterbox configuration."""

    #: Informational only -- CTIO is this tool's one site -- but named here
    #: rather than left implicit, since every post says where it thinks it is.
    site: str = "ctio"
    slack: SlackConfig = field(default_factory=SlackConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    dark_hours: DarkHoursConfig = field(default_factory=DarkHoursConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    links: LinksConfig = field(default_factory=LinksConfig)

    @property
    def slack_token(self) -> str | None:
        """Bot token from the environment, or None when unset.

        Returns
        -------
        token : `str` or None
            When None, callers should degrade to writing output locally rather
            than failing -- that is what makes the pipeline testable offline.
        """
        return os.environ.get(self.slack.bot_token_env) or None


def _from_mapping(cls: type, data: Any) -> Any:
    """Recursively build a nested dataclass from plain YAML mappings.

    Unknown keys are reported and ignored rather than silently dropped, since
    a typo in a config key would otherwise look like the setting had no
    effect.
    """
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        name = key.replace("-", "_")
        if name not in known:
            logger.warning("Ignoring unknown config key %r for %s", key, cls.__name__)
            continue
        kwargs[name] = _from_mapping(known[name].type, value)
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to defaults.

    Parameters
    ----------
    path : `str`, `pathlib.Path`, or None
        Explicit config file. When None, the first readable entry of
        `DEFAULT_CONFIG_PATHS` is used; if none exist, defaults are returned.

    Returns
    -------
    config : `Config`
    """
    candidates = [Path(path)] if path is not None else list(DEFAULT_CONFIG_PATHS)
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_file():
            continue
        logger.info("Loading configuration from %s", candidate)
        with open(candidate) as f:
            data = yaml.safe_load(f) or {}
        return _from_mapping(Config, data)

    if path is not None:
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.info("No config file found; using defaults")
    return Config()
