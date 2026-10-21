"""Tests for the animated airmass map of night zero."""

import json

import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from conftest import make_notice, make_skymap_bytes
from PIL import Image

from decam_chatterbox.app import process_notice, run_service
from decam_chatterbox.astro.almanac import ctio_observer, night_events
from decam_chatterbox.ingest.decode import decode_notice
from decam_chatterbox.plots.airmass_animation import (
    airmass_from_altitude,
    airmass_on_grid,
    frame_times,
    plot_airmass_animation,
)
from decam_chatterbox.slackbot.client import PostedMessage, SlackPoster


def trigger_for(ra=60.0, dec=-35.0, radius=6.0):
    return decode_notice(make_notice(skymap_bytes=make_skymap_bytes(ra, dec, radius)), working_nside=64)


# -------------------------------------------------------------------- masking


def test_airmass_is_masked_above_the_limit_and_below_the_horizon():
    """Anything past airmass 3.8 is NaN, not a colour."""
    # 15.0 deg is airmass 3.86, over the limit; 15.5 deg is 3.74, under it.
    airmass = airmass_from_altitude([90.0, 30.0, 15.5, 15.0, 5.0, 0.0, -20.0], mask_above=3.8)
    assert airmass[0] == pytest.approx(1.0)
    assert airmass[1] == pytest.approx(2.0)
    assert airmass[2] == pytest.approx(3.74, abs=0.01)
    assert np.isnan(airmass[3:]).all(), "over the limit, on the horizon and below it are all NaN"


def test_the_mask_is_configurable():
    assert np.isfinite(airmass_from_altitude([15.0], mask_above=4.0)[0])
    assert np.isnan(airmass_from_altitude([15.0], mask_above=3.0)[0])


def test_airmass_on_the_sky_agrees_with_an_independent_altitude():
    """Checked against astropy's AltAz at CTIO, independent of production."""
    when = Time("2026-08-14T05:00:00")
    ra = np.array([60.0, 150.0, 10.0, 300.0])
    dec = np.array([-35.0, -60.0, -10.0, -45.0])
    ours = airmass_on_grid(ra, dec, when.mjd, mask_above=10.0)

    frame = AltAz(obstime=when, location=ctio_observer().location)
    altaz = SkyCoord(ra, dec, unit="deg").transform_to(frame)
    reference = np.where(altaz.alt.deg > 0, 1.0 / np.sin(np.radians(altaz.alt.deg)), np.nan)
    both = np.isfinite(ours) & np.isfinite(reference) & (reference < 3.0)
    assert both.any()
    assert ours[both] == pytest.approx(reference[both], abs=0.06)


# --------------------------------------------------------------------- frames


def test_frames_span_the_whole_night_inclusive():
    events = night_events(when=Time("2026-08-14T02:00:00"))
    start, end = events.observing_window(-12.0)
    times = frame_times(events, step_minutes=10.0)
    assert times[0] == pytest.approx(start)
    assert times[-1] == pytest.approx(end), "the last frame is the end of the night, not a step short"
    assert np.all(np.diff(times) > 0)
    assert np.diff(times)[:-1] == pytest.approx(10.0 / 1440.0)


# ------------------------------------------------------------------ animation


def render(tmp_path, trigger, name="a.gif", **kwargs):
    events = night_events(when=trigger.event_time)
    settings = dict(step_minutes=120.0, dpi=40, fps=4)
    settings.update(kwargs)
    path = plot_airmass_animation(
        events,
        trigger.localization,
        tmp_path / name,
        peak=(trigger.geometry.peak_ra_deg, trigger.geometry.peak_dec_deg),
        **settings,
    )
    return events, path


def test_the_animation_is_a_looping_gif_with_a_frame_per_time(tmp_path):
    trigger = trigger_for()
    events, path = render(tmp_path, trigger)
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(frame_times(events, 120.0))
        assert gif.n_frames > 3
        assert gif.info.get("loop") == 0, "loops forever"


def test_the_frames_actually_change_as_the_sky_turns(tmp_path):
    """A static image saved as a GIF would pass every other check."""
    _, path = render(tmp_path, trigger_for())
    with Image.open(path) as gif:
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"))
        gif.seek(gif.n_frames // 2)
        middle = np.asarray(gif.convert("RGB"))
    assert (first != middle).any(axis=-1).mean() > 0.02


def test_a_compact_localization_gets_a_zoomed_panel_and_a_large_one_does_not(tmp_path):
    _, compact = render(tmp_path, trigger_for(radius=6.0), "compact.gif")
    _, large = render(tmp_path, trigger_for(radius=60.0), "large.gif")
    with Image.open(compact) as a, Image.open(large) as b:
        assert a.size[0] > b.size[0], "the zoom panel makes the figure wider"


# ------------------------------------------------------------------ in the app


@pytest.fixture
def animated_config(config):
    config.airmass_animation.enabled = True
    config.airmass_animation.step_minutes = 120.0
    config.airmass_animation.dpi = 40
    return config


def test_the_animation_is_posted_once_in_the_alerts_thread(notice, animated_config, tmp_path, monkeypatch):
    """Threaded onto the main-channel post, not the urgent cross-post."""
    animated_config.slack.urgent_channel = "#urgent"
    replies = []
    posts = []

    def fake_post(self, blocks, text, channel=None, **kwargs):
        posts.append(channel or self.channel_for(kwargs.get("is_test", False)))
        return PostedMessage(channel=posts[-1], ts=f"{len(posts)}.0")

    def fake_reply(self, parent, blocks, text, files=None, label="reply"):
        replies.append((parent, blocks, list(files or [])))
        return PostedMessage(channel=parent.channel, ts="9.0")

    monkeypatch.setattr(SlackPoster, "post", fake_post)
    monkeypatch.setattr(SlackPoster, "reply", fake_reply)
    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    report = process_notice(notice, animated_config, poster=poster, out_dir=tmp_path / "plots")
    assert report.posted_urgent is not None, "the fixture notice is high priority"
    assert report.animation_thread is not None
    assert report.wait_for_background(timeout=300)
    assert not report.has_background_work

    assert report.animation is not None and report.animation.is_file()
    assert len(posts) == 2, "the alert and its urgent cross-post; the animation is a reply"
    assert len(replies) == 1
    parent, blocks, files = replies[0]
    assert parent is report.posted and parent.channel == animated_config.slack.channel
    assert files == [report.animation]
    caption = blocks[0]["text"]["text"]
    assert "above airmass 3.8" in caption, "the mask is explained, not left as mysterious grey"
    assert report.events.day_obs in caption


def test_offline_the_animation_payload_is_written_alongside_the_post(notice, animated_config, tmp_path):
    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    report = process_notice(notice, animated_config, poster=poster, out_dir=tmp_path / "plots")
    assert report.wait_for_background(timeout=300)
    payloads = sorted((tmp_path / "posts").glob("*_airmass.json"))
    assert len(payloads) == 1
    payload = json.loads(payloads[0].read_text())
    assert payload["files"] == [str(report.animation)]
    assert payload["channel"] == animated_config.slack.channel


def test_the_animation_does_not_run_when_switched_off_or_not_posting(notice, animated_config, tmp_path):
    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    declined = process_notice(
        notice, animated_config, poster=poster, out_dir=tmp_path / "a", run_animation=False
    )
    dry = process_notice(notice, animated_config, poster=poster, post=False, out_dir=tmp_path / "b")
    animated_config.airmass_animation.enabled = False
    disabled = process_notice(notice, animated_config, poster=poster, out_dir=tmp_path / "c")
    for report in (declined, dry, disabled):
        assert report.animation_thread is None
        assert report.wait_for_background(timeout=0)


def test_a_retraction_gets_no_animation(retraction_notice, animated_config, tmp_path):
    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    report = process_notice(retraction_notice, animated_config, poster=poster, out_dir=tmp_path / "plots")
    assert report.animation_thread is None


def test_a_render_failure_costs_only_the_animation(notice, animated_config, tmp_path, monkeypatch):
    import decam_chatterbox.plots.airmass_animation as module

    def broken(*args, **kwargs):
        raise RuntimeError("no frames today")

    monkeypatch.setattr(module, "plot_airmass_animation", broken)
    poster = SlackPoster(animated_config, dry_run=True, output_dir=tmp_path / "posts")
    report = process_notice(notice, animated_config, poster=poster, out_dir=tmp_path / "plots")
    assert report.posted is not None, "the alert went out first"
    assert report.wait_for_background(timeout=60)
    assert report.animation is None
    assert any("airmass animation failed" in w for w in report.warnings)


def test_run_service_waits_for_the_animation_before_exiting(tmp_path, notice, animated_config):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(notice))
    run_service(animated_config, paths=[path])
    posts = tmp_path / "work" / "posts"
    assert list(posts.glob("*_airmass.json")), "the daemon thread was joined, not abandoned"
