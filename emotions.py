"""Choose portrait expressions without changing the recorded reaction."""

from constants import EMOTION_THRESHOLD, FURIOUS_THRESHOLD


def portrait_expression(name, intensity):
    """Reserve furious eyes for the strongest angry reactions."""
    if name == "neutral" or intensity < EMOTION_THRESHOLD:
        return None
    if name == "angry" and intensity >= FURIOUS_THRESHOLD:
        return "furious"
    return name
