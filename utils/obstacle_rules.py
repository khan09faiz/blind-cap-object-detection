"""
Dangerous-object definitions and quick checks.
"""

# Tier 1 — fast-moving or immediately life-threatening
_TIER1: set[str] = {"car", "bus", "truck", "motorcycle"}

# Tier 2 — dangerous but typically slower / stationary
_TIER2: set[str] = {
    "bicycle",
    "stairs",
    "pole",
    "fire_hydrant",
    "trash_can",       # COCO name
    "waste_container", # Open Images name
    "knife",
    "scissors",
}

DANGEROUS_OBJECTS: set[str] = _TIER1 | _TIER2


def is_dangerous(label: str) -> bool:
    """Return True if *label* is considered dangerous."""
    return label.lower() in DANGEROUS_OBJECTS


def get_warning_prefix(label: str) -> str:
    """Return ``"Warning"`` for dangerous objects, ``""`` otherwise."""
    return "Warning" if is_dangerous(label) else ""


def priority_score(label: str, distance: float) -> float:
    """Lower score = higher priority = spoken first.

    Scoring:
      - Tier-1 dangerous (fast vehicles):   0–9.99  (distance as tie-breaker)
      - Tier-2 dangerous (obstacles):      10–19.99
      - Safe objects:                       20–29.99

    Within each tier, closer objects (smaller distance) come before farther ones.
    """
    lbl = label.lower()
    clamped = min(distance, 9.99)   # keep within tier band
    if lbl in _TIER1:
        return 0.0 + clamped
    if lbl in _TIER2:
        return 10.0 + clamped
    return 20.0 + clamped
