"""Tests for decam_chatterbox.priority."""

from conftest import make_notice

from decam_chatterbox.config import PriorityConfig
from decam_chatterbox.ingest.decode import decode_notice
from decam_chatterbox.priority import assess_priority


def test_default_notice_is_high_priority(notice):
    """conftest.make_notice's defaults (FAR 1e-10, tight disc, BNS-heavy) pass
    every threshold, so this is the baseline every other test perturbs."""
    trigger = decode_notice(notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert assessment.is_high_priority
    assert len(assessment.reasons) == 3


def test_retraction_is_never_high_priority(retraction_notice):
    trigger = decode_notice(retraction_notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert not assessment.is_high_priority
    assert "retracted" in assessment.reasons[0]


def test_mock_event_is_never_high_priority(mock_notice):
    trigger = decode_notice(mock_notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert not assessment.is_high_priority


def test_high_far_fails_the_threshold(skymap_bytes):
    notice = make_notice(far=1e-6, skymap_bytes=skymap_bytes)  # ~1 per month, not "per year"
    trigger = decode_notice(notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert not assessment.is_high_priority
    assert any("FAR" in r and ">" in r for r in assessment.reasons)


def test_large_area_fails_the_threshold(skymap_bytes):
    notice = make_notice(skymap_bytes=skymap_bytes)
    trigger = decode_notice(notice)
    config = PriorityConfig(max_area_deg2=1.0)  # far smaller than the ~280 deg^2 test disc
    assessment = assess_priority(trigger, config)
    assert not assessment.is_high_priority
    assert any("area" in r and ">" in r for r in assessment.reasons)


def test_bbh_like_classification_fails_the_threshold(skymap_bytes):
    notice = make_notice(
        classification={"BNS": 0.02, "NSBH": 0.01, "BBH": 0.95, "Terrestrial": 0.02},
        properties={},
        skymap_bytes=skymap_bytes,
    )
    trigger = decode_notice(notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert not assessment.is_high_priority
    assert any("p(BNS)+p(NSBH)" in r and "<" in r for r in assessment.reasons)


def test_burst_event_is_exempted_from_the_classification_criterion(skymap_bytes):
    notice = make_notice(
        group="Burst", pipeline="cWB", classification={}, properties={}, skymap_bytes=skymap_bytes
    )
    trigger = decode_notice(notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert assessment.is_high_priority
    assert any("n/a" in r for r in assessment.reasons)


def test_missing_localization_fails_the_area_criterion():
    notice = make_notice(skymap_bytes=None)
    trigger = decode_notice(notice)
    assessment = assess_priority(trigger, PriorityConfig())
    assert not assessment.is_high_priority
    assert "no localization" in assessment.reasons


def test_disabled_config_is_never_high_priority(notice):
    trigger = decode_notice(notice)
    assessment = assess_priority(trigger, PriorityConfig(enabled=False))
    assert not assessment.is_high_priority
