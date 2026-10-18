"""Small pieces of behavior shared by resident and snake interactions."""

import math

from character_data import DIRECTIONS
from constants import INTERACTION_GAP, NPC_RADIUS, POST_DIALOGUE_SECONDS


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
        face_toward(actor, (other.x, other.y))


def face_toward(actor, point):
    angle = math.atan2(point[0] - actor.x, actor.y - point[1])
    actor.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)


def move_toward(actor, point, dt, speed):
    """Move without overshooting, facing the direction of actual movement."""
    dx, dy = point[0] - actor.x, point[1] - actor.y
    distance = math.hypot(dx, dy)
    step = speed * max(0.0, dt)
    if distance and step:
        face_toward(actor, point)
    if distance <= step:
        actor.x, actor.y = point
        return True
    actor.x += dx / distance * step
    actor.y += dy / distance * step
    return False


def company_point(npc, partner, spacing=0, *, allow_vertical=False):
    """Choose the nearest reachable meeting point with room for both sprites."""
    default_size = (NPC_RADIUS * 2, NPC_RADIUS)
    gap = sum((actor.sprite_size or default_size)[0] / 2 for actor in (npc, partner))
    gap = max(gap + INTERACTION_GAP, spacing)
    points = [(partner.x - gap, partner.y), (partner.x + gap, partner.y)]
    points = [point for point in points if npc.can_reach(point)]
    if not points and allow_vertical:
        height = max((actor.sprite_size or default_size)[1] for actor in (npc, partner)) + INTERACTION_GAP
        points = [(partner.x, partner.y - height), (partner.x, partner.y + height)]
        points = [point for point in points if npc.can_reach(point)]
    return min(points, key=lambda point: math.dist(point, (npc.x, npc.y))) if points else None


def record_generated_turns(interaction, turns, rewritten_parts=()):
    """Keep speaker roles and recorded display parts identical across encounters."""
    interaction.turns = list(turns)
    for index, turn in enumerate(interaction.turns):
        actor = interaction.initiator if turn.speaker == "initiator" else interaction.recipient
        rewritten = rewritten_parts[index] if rewritten_parts else ()
        interaction.record.add_turn(actor.name, turn.text, [part.model_dump() for part in turn.parts], rewritten)


def record_failed_dialogue(interaction, turns):
    """Keep rejected speech for inspection, outside committed conversation history."""
    interaction.record.failed_turns = [turn.model_dump() for turn in turns]
