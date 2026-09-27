"""Tests for decam_chatterbox.slackbot.blocks."""

from conftest import make_notice

from decam_chatterbox.astro.almanac import night_events
from decam_chatterbox.astro.darkhours import dark_hours_map
from decam_chatterbox.ingest.decode import decode_notice
from decam_chatterbox.plots.darkhours import region_hours_summary
from decam_chatterbox.slackbot.blocks import build_trigger_blocks, plain_text_summary
from decam_chatterbox.slackbot.client import render_blocks_as_text


def _text(blocks) -> str:
    return render_blocks_as_text(blocks)


def test_retraction_message_has_no_localization_or_significance_section(retraction_notice, config):
    trigger = decode_notice(retraction_notice)
    blocks = build_trigger_blocks(trigger, None, None, None, config)
    text = _text(blocks)
    assert "retracted" in text.lower()
    assert "*Significance*" not in text
    assert "*Localization*" not in text


def test_full_message_contains_the_key_facts(notice, config):
    trigger = decode_notice(notice)
    events = night_events(trigger.event_time)
    dark_hours = dark_hours_map(events, nside=16, step_minutes=10.0)
    dark_stats = region_hours_summary(dark_hours, trigger.localization)
    blocks = build_trigger_blocks(trigger, events, dark_hours, dark_stats, config)
    text = _text(blocks)

    assert trigger.superevent_id in text
    assert "PRELIMINARY" in text
    assert "BNS" in text
    assert "Localization" in text
    assert "DECam pointings" in text
    assert "Night of" in text
    assert "no observing-strategy recommendation" in text
    assert "GraceDB" in text


def test_mock_event_is_visibly_marked(mock_notice, config):
    trigger = decode_notice(mock_notice)
    blocks = build_trigger_blocks(trigger, None, None, None, config)
    text = _text(blocks)
    assert "NOT A REAL SUPEREVENT" in text


def test_burst_message_shows_duration_not_classification(config):
    notice = make_notice(group="Burst", pipeline="cWB", classification={}, properties={})
    notice["event"]["duration"] = 0.15
    notice["event"]["central_frequency"] = 200.0
    trigger = decode_notice(notice)
    blocks = build_trigger_blocks(trigger, None, None, None, config)
    text = _text(blocks)
    assert "Burst parameters" in text
    assert "duration" in text
    assert "Classification" not in text


def test_missing_localization_is_noted_not_silent(config):
    notice = make_notice(skymap_bytes=None)
    trigger = decode_notice(notice)
    blocks = build_trigger_blocks(trigger, None, None, None, config)
    text = _text(blocks)
    assert "no sky map" in text.lower()


def test_plain_text_summary_marks_retraction(retraction_notice):
    trigger = decode_notice(retraction_notice)
    assert "RETRACTED" in plain_text_summary(trigger)


def test_plain_text_summary_marks_mock(mock_notice):
    trigger = decode_notice(mock_notice)
    assert plain_text_summary(trigger).startswith("[NOT REAL]")
