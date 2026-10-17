"""Small pieces of behavior shared by resident and snake interactions."""

import math

from character_data import DIRECTIONS
from constants import POST_DIALOGUE_SECONDS


def finish_generated_dialogue(interaction, *, reading=False):
    """Apply the same completion timing to every generated world conversation."""
    interaction.dialogue_state = "ready"
    interaction.talking_remaining = POST_DIALOGUE_SECONDS
    interaction.update(0.0, reading=reading)


def dialogue_finished(interaction, dt, *, reading=False):
    """Hold a viewed exchange until Done, then apply its completion timer."""
    if reading:
        return False
    interaction.talking_remaining = max(0.0, interaction.talking_remaining - dt)
    return interaction.talking_remaining <= 0.0


def face_each_other(first, second):
    for actor, other in ((first, second), (second, first)):
        angle = math.atan2(other.x - actor.x, actor.y - other.y)
        actor.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
