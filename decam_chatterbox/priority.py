"""High-priority triage: is this alert worth a closer look right now?

This is a triage flag, not a decision. It never changes what gets posted to
`SlackConfig.channel` -- every real and mock alert still goes there,
matching the rest of this tool's "nothing goes silent" design. It only
controls the visible badge on the message, whether `SlackConfig.mention` is
included, and whether the alert is cross-posted to
`SlackConfig.urgent_channel`. Deciding what to actually point DECam at is
left to a human, same as everything else here.
"""

import logging
from dataclasses import dataclass, field

from .config import PriorityConfig
from .models import Trigger

__all__ = ["PriorityAssessment", "assess_priority"]

logger = logging.getLogger(__name__)


@dataclass
class PriorityAssessment:
    """The outcome of checking a trigger against `PriorityConfig`.

    Attributes
    ----------
    is_high_priority : `bool`
        True when every applicable criterion was met.
    reasons : `list` [`str`]
        One line per criterion checked, each stating the measured value, the
        threshold, and whether it passed -- shown in the message so "high
        priority" is never a bare, unexplained badge.
    """

    is_high_priority: bool
    reasons: list[str] = field(default_factory=list)


def _classification_ns_probability(trigger: Trigger) -> float:
    """p(BNS) + p(NSBH): probability the merger involved a neutron star."""
    return trigger.classification.get("BNS", 0.0) + trigger.classification.get("NSBH", 0.0)


def assess_priority(trigger: Trigger, config: PriorityConfig) -> PriorityAssessment:
    """Check a trigger against the configured high-priority thresholds.

    Parameters
    ----------
    trigger : `Trigger`
        Decoded (and possibly localized) trigger.
    config : `PriorityConfig`
        Thresholds to check against.

    Returns
    -------
    assessment : `PriorityAssessment`
    """
    if trigger.is_retraction:
        return PriorityAssessment(is_high_priority=False, reasons=["retracted"])
    if not trigger.is_real:
        return PriorityAssessment(is_high_priority=False, reasons=["not a real superevent"])
    if not config.enabled:
        return PriorityAssessment(is_high_priority=False, reasons=["priority triage disabled"])

    ok = True
    reasons: list[str] = []

    if trigger.far_hz is not None:
        passed = trigger.far_hz <= config.max_far_hz
        ok = ok and passed
        cmp = "<=" if passed else ">"
        reasons.append(f"FAR {trigger.far_hz:.2e} Hz {cmp} {config.max_far_hz:.2e} Hz")
    else:
        ok = False
        reasons.append("FAR unknown")

    if trigger.geometry is not None:
        passed = trigger.geometry.area_deg2 <= config.max_area_deg2
        ok = ok and passed
        cmp = "<=" if passed else ">"
        reasons.append(
            f"90% area {trigger.geometry.area_deg2:,.0f} deg^2 {cmp} {config.max_area_deg2:,.0f} deg^2"
        )
    else:
        ok = False
        reasons.append("no localization")

    if trigger.is_burst:
        reasons.append("classification n/a (Burst event)")
    elif trigger.classification:
        ns_probability = _classification_ns_probability(trigger)
        passed = ns_probability >= config.min_ns_classification
        ok = ok and passed
        cmp = ">=" if passed else "<"
        reasons.append(f"p(BNS)+p(NSBH) {ns_probability:.0%} {cmp} {config.min_ns_classification:.0%}")
    else:
        ok = False
        reasons.append("no classification")

    return PriorityAssessment(is_high_priority=ok, reasons=reasons)
