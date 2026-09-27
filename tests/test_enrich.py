"""Tests for decam_chatterbox.ingest.enrich_gracedb (offline paths only)."""

from conftest import make_notice

from decam_chatterbox.config import EnrichConfig
from decam_chatterbox.ingest.decode import decode_notice
from decam_chatterbox.ingest.enrich_gracedb import enrich_chirp_mass, is_superevent_id


def test_is_superevent_id_accepts_real_mock_and_test_prefixes():
    assert is_superevent_id("S260814a")
    assert is_superevent_id("M260814ab")
    assert is_superevent_id("T260814abc")


def test_is_superevent_id_rejects_junk():
    assert not is_superevent_id("")
    assert not is_superevent_id("not-an-id")
    assert not is_superevent_id("GW150914")


def test_enrich_chirp_mass_disabled_returns_none(notice):
    trigger = decode_notice(notice)
    assert enrich_chirp_mass(trigger, EnrichConfig(gracedb=False)) is None


def test_enrich_chirp_mass_skips_retraction(retraction_notice):
    trigger = decode_notice(retraction_notice)
    assert enrich_chirp_mass(trigger, EnrichConfig(gracedb=True)) is None


def test_enrich_chirp_mass_skips_burst():
    notice = make_notice(group="Burst", classification={}, properties={})
    trigger = decode_notice(notice)
    assert enrich_chirp_mass(trigger, EnrichConfig(gracedb=True)) is None
