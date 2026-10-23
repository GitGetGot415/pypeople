"""Small pieces of behavior shared by resident and snake interactions."""

import math
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field
from character_data import DIRECTIONS
from constants import (CONVERSATION_CLEARANCE, CONVERSATION_SPOT_STEP, CONVERSATION_TREE_CLEARANCE,
                      INTERACTION_GAP, NPC_RADIUS, POST_DIALOGUE_SECONDS, DECISION_SECONDS,
                      DIALOGUE_TIMEOUT_SECONDS, MAX_SIMULATION_STEP_SECONDS)


@dataclass
class DialogueProgress:
    """Logical generation and completion timing, owned by the exchange.

    Transport admission and UI reading state are deliberately outside this value.
    """
    dialogue_state: Literal["waiting", "pending", "ready", "failed"] = "waiting"
    wait_elapsed: Annotated[float, Field(ge=0, le=DIALOGUE_TIMEOUT_SECONDS + MAX_SIMULATION_STEP_SECONDS)] = 0.0
    talking_remaining: Annotated[float, Field(ge=0, le=max(POST_DIALOGUE_SECONDS, DECISION_SECONDS))] = POST_DIALOGUE_SECONDS

    def retry_after_restore(self):
        if self.dialogue_state == "pending":
            self.dialogue_state, self.wait_elapsed = "waiting", 0.0


@dataclass
class PlayerReplyProgress:
    """Actor-owned reply progress; deferred payloads belong only to Simulation.

    The request identity is transient. A durable retry description belongs to the
    physical player meeting, so restoring never resurrects a worker.
    """
    request_id: str | None = None
    closing: bool = False

    @property
    def generating(self):
        return self.request_id is not None

    def start(self, request_id):
        self.request_id = request_id

    def complete(self):
        self.request_id = None

    def release(self):
        self.request_id, self.closing = None, False


def player_request_pending(request):
    """A session association refers to the actor's authoritative generation."""
    return request["npc"].player_reply.request_id == request["id"]


def finish_generated_dialogue(interaction, *, reading=False):
    """Apply the same completion timing to every generated world conversation."""
    interaction.dialogue.dialogue_state = "ready"
    interaction.dialogue.talking_remaining = POST_DIALOGUE_SECONDS
    interaction.update(0.0, reading=reading)


def dialogue_finished(interaction, dt, *, reading=False):
    """Hold a viewed exchange until Done, then apply its completion timer."""
    if reading:
        return False
    interaction.dialogue.talking_remaining = max(0.0, interaction.dialogue.talking_remaining - dt)
    return interaction.dialogue.talking_remaining <= 0.0


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
    ux, uy = dx / distance, dy / distance
    world = getattr(actor, "world", None)
    if world is not None and hasattr(actor, "location") and actor is not world.snake:
        from constants import SOFT_AVOIDANCE_DISTANCE
        from locations import clamp_point
        neighbors = list(world.npcs)
        if world.mode == "player" and world.player.alive:
            neighbors.append(world.player)
        for other in neighbors:
            if other is actor or other.location != actor.location:
                continue
            ox, oy = other.x - actor.x, other.y - actor.y
            separation = math.hypot(ox, oy)
            if separation < SOFT_AVOIDANCE_DISTANCE and ox * ux + oy * uy >= 0:
                bias = (1 if actor.id < other.id else -1) * (1 - separation / SOFT_AVOIDANCE_DISTANCE)
                if abs(dx) >= abs(dy):
                    uy += bias
                else:
                    ux += bias
        length = math.hypot(ux, uy)
        actor.x, actor.y = clamp_point(actor.location, (actor.x + ux / length * step, actor.y + uy / length * step))
    else:
        actor.x += ux * step
        actor.y += uy * step
    return False


def company_point(npc, partner, spacing=0, *, allow_vertical=False):
    """Choose the nearest reachable meeting point with room for both sprites."""
    if partner is None or npc.location != partner.location:
        return None
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


def conversation_points(first, second, spacing=0):
    """Choose two reachable positions near the midpoint, preferring a clear area."""
    world = first.world
    midpoint = ((first.x + second.x) / 2, (first.y + second.y) / 2)
    sizes = [actor.sprite_size or (NPC_RADIUS * 2, NPC_RADIUS) for actor in (first, second)]
    horizontal_gap = max(sum(size[0] / 2 for size in sizes) + INTERACTION_GAP, spacing)
    vertical_gap = max(max(size[1] for size in sizes) + INTERACTION_GAP, spacing)
    if first.location != second.location:
        return None
    occupied = ([((world.tree.x, world.tree.y), CONVERSATION_TREE_CLEARANCE)]
                if first.location == "garden" else [])
    seen = set()
    for resident in (n for n in world.npcs if n.location == first.location):
        interaction = resident.interaction
        if interaction is None or id(interaction) in seen:
            continue
        seen.add(id(interaction))
        actors = (interaction.initiator, interaction.recipient)
        if any(actor is first or actor is second for actor in actors):
            continue
        points = [actor.destination or (actor.x, actor.y) for actor in actors]
        center = tuple((a + b) / 2 for a, b in zip(*points))
        occupied.append((center, math.dist(*points) / 2 + CONVERSATION_CLEARANCE))

    candidates, clear = [], []
    for dx, dy in ((horizontal_gap / 2, 0), (0, vertical_gap / 2)):
        for sign in (-1, 1):
            offsets = ((sign * dx, sign * dy), (-sign * dx, -sign * dy))
            bounds = []
            for actor, (ox, oy) in zip((first, second), offsets):
                left, right, top, bottom = actor.movement_bounds
                bounds.append((left - ox, right - ox, top - oy, bottom - oy))
            left, right = max(b[0] for b in bounds), min(b[1] for b in bounds)
            top, bottom = max(b[2] for b in bounds), min(b[3] for b in bounds)
            if left > right or top > bottom:
                continue
            centers = [(min(right, max(left, midpoint[0])), min(bottom, max(top, midpoint[1])))]
            centers.extend((x, y)
                           for x in [left, right, *range(math.ceil(left), math.floor(right) + 1, CONVERSATION_SPOT_STEP)]
                           for y in [top, bottom, *range(math.ceil(top), math.floor(bottom) + 1, CONVERSATION_SPOT_STEP)])
            radius = math.hypot(dx, dy)
            for center in centers:
                points = tuple((center[0] + ox, center[1] + oy) for ox, oy in offsets)
                candidates.append(points)
                if all(math.dist(center, other) >= radius + clearance for other, clearance in occupied):
                    clear.append(points)
    if not candidates:
        return None

    def travel(points):
        distances = [math.dist((actor.x, actor.y), point) for actor, point in zip((first, second), points)]
        return max(distances), sum(distances)

    return min(clear or candidates, key=travel)


def record_generated_turns(interaction, turns, rewritten_parts=()):
    """Keep speaker roles and recorded display parts identical across encounters."""
    for index, turn in enumerate(turns):
        actor = interaction.initiator if turn.speaker == "initiator" else interaction.recipient
        rewritten = rewritten_parts[index] if rewritten_parts else ()
        interaction.record.add_turn(actor.name, turn.text, [part.model_dump() for part in turn.parts], rewritten,
                                    speaker_id=actor.id)


def record_failed_dialogue(interaction, turns):
    """Keep rejected speech for inspection, outside committed conversation history."""
    interaction.record.failed_turns = [turn.model_dump() for turn in turns]
