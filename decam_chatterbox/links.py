"""Permalinks included in alert posts."""

from urllib.parse import quote

from .config import LinksConfig

__all__ = ["site_links", "gracedb_link", "format_link_list"]


def site_links(config: LinksConfig) -> list[tuple[str, str]]:
    """Ordered ``(label, url)`` pairs for the site conditions section.

    Parameters
    ----------
    config : `LinksConfig`
        Link configuration.

    Returns
    -------
    links : `list` [`tuple`]
        Only entries with a non-empty URL.
    """
    candidates = [
        ("Weather at Tololo", config.weather),
        ("Seeing / webcams", config.seeing),
        ("Almanac", config.almanac),
        ("Observatory status", config.observatory_status),
    ]
    candidates.extend(config.extra.items())
    return [(label, url) for label, url in candidates if url]


def gracedb_link(superevent_id: str, gracedb_url: str | None = None) -> str:
    """GraceDB superevent page, preferring the URL a notice carried."""
    if gracedb_url:
        return gracedb_url
    return f"https://gracedb.ligo.org/superevents/{quote(superevent_id)}/view/"


def format_link_list(links: list[tuple[str, str]], separator: str = "  |  ") -> str:
    """Render links as Slack mrkdwn ``<url|label>`` entries."""
    return separator.join(f"<{url}|{label}>" for label, url in links)
