"""Small pieces of behavior shared by resident and snake interactions."""

import math
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field
from character_data import DIRECTIONS
from constants import (CONVERSATION_CLEARANCE, CONVERSATION_SPOT_STEP, CONVERSATION_TREE_CLEARANCE,
                      INTERACTION_GAP, NPC_RADIUS, POST_DIALOGUE_SECONDS, DECISION_SECONDS,
                      DIALOGUE_TIMEOUT_SECONDS, MAX_SIMULATION_STEP_SECONDS)


@dataclass(frozen=True)
class ParticipantReservation:
    kind: str
    owner: object
    parent: object | None = None


class ParticipantReservations:
    """Transient participant ownership; progress stays in the owning domains.

    Visits and meetings may contain an exchange. Acquisition does
    not decide availability: each entry point retains its purpose-specific rules.
    Releasing a parent never silently commits or interrupts its child.
    """
    def __init__(self):
        self._actors = {}

    def get(self, actor_id, kind):
        entry = self._actors.get(actor_id, {}).get(kind)
        return entry.owner if entry else None

    def acquire(self, kind, owner, *actors):
        for actor in actors:
            if actor is None:
                continue
            entries = self._actors.setdefault(actor.id, {})
            parent_kinds = {"conversation": ("visit",), "handoff": ("player", "visit"),
                            "player": ("visit",)}.get(kind, ())
            parent = next((entries[k].owner for k in parent_kinds if k in entries), None)
            entries[kind] = ParticipantReservation(kind, owner, parent)

    def release(self, kind, owner, *actors):
        """Release only the expected owner, including late/stale cleanup paths."""
        released = False
        for actor_id in ([a.id for a in actors] if actors else list(self._actors)):
            entries = self._actors.get(actor_id, {})
            if kind in entries and entries[kind].owner is owner:
                del entries[kind]
                released = True
                if not entries:
                    del self._actors[actor_id]
        return released

    def exchange(self, actor_id):
        return self.get(actor_id, "conversation") or self.get(actor_id, "handoff")

    def held(self, actor_id, *kinds):
        return any(self.get(actor_id, kind) is not None for kind in kinds)


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


@dataclass
class PlayerRequest:
    """A live session association; only the physical meeting saves retry data."""
    id: str
    actor: object
    snapshot: dict
    message: str
    record: object
    actor_revision: int
    roster_revision: int
    prepared_opening: tuple | None = None


def player_request_pending(request):
    """A session association refers to the actor's authoritative generation."""
    return request.actor.player_reply.request_id == request.id


def suspended_activity(interaction, resident):
    return next(state for actor, state in interaction.resident_activities() if actor is resident)


def release_resident(world, kind, owner, resident, state, *cooldown_ids, preserve_facing=False):
    """Late cleanup must release the expected owner before restoring its activity."""
    if not world.reservations.release(kind, owner, resident):
        return False
    if resident.world.actor_by_id(resident.id) is not resident or resident.departure:
        return False
    resident.restore_after_interaction(state, *cooldown_ids, preserve_facing=preserve_facing)
    return True


def finish_generated_dialogue(interaction, *, reading=False):
    """Apply the same completion timing to every generated world conversation."""
    observe_conversation(interaction.initiator.world, interaction.record,
                         interaction.initiator, interaction.recipient, accepted=True)
    interaction.dialogue.dialogue_state = "ready"
    interaction.dialogue.talking_remaining = POST_DIALOGUE_SECONDS
    interaction.update(0.0, reading=reading)


def observe_conversation(world, record, first, second, *, accepted=False):
    """Collect sight while met; only accepted speech establishes an exchange."""
    from locations import observers
    from information import add_fact
    # Neither side of a closed door can see the hidden conversation partner.
    if first.location != second.location:
        return
    for actor in observers(world, first, second):
        if actor not in (first, second) and actor.id not in record.pending_witness_ids:
            record.pending_witness_ids.append(actor.id)
    fact = next((f for f in world.facts if f.kind == "conversation_seen" and f.conversation_id == record.id), None)
    if fact is None and not accepted:
        return
    if fact is None:
        text = f"{first.name} and {second.name} were talking."
        event = world.publish("conversation_seen", (first, second), text, record)
        fact = add_fact(world, "conversation_seen", first, second, event.id, record, text, [])
    witnesses = [world.participant_by_id(actor_id) for actor_id in record.pending_witness_ids]
    add_fact(world, fact.kind, first, second, fact.source_id, record, fact.summary, witnesses)
    record.pending_witness_ids.clear()


def observe_active_conversations(world):
    """Called at real area transitions as well as the normal interaction ticks."""
    from apartments import observe_visit
    observe_visit(world)
    meetings = {id(n.conversation or n.gift): n.conversation or n.gift for n in world.npcs
                if n.conversation or n.gift and n.gift.thanks}
    if world.snake.conversation:
        meetings[id(world.snake.conversation)] = world.snake.conversation
    for meeting in meetings.values():
        if meeting.active and meeting.arrived:
            observe_conversation(world, meeting.record, meeting.initiator, meeting.recipient)
    for actor in world.npcs:
        if actor.eating and not actor.resume_pending:
            collect_eating_witnesses(world, actor)
    interaction = world.player_interaction
    if interaction and interaction.kind in ("resident", "snake") and interaction.stage == "conversation" and interaction.record_id:
        npc = world.actor_by_id(interaction.npc_id)
        if npc:
            observe_conversation(world, world.record_by_id(interaction.record_id), world.player, npc)


def collect_eating_witnesses(world, actor):
    from locations import observers
    witnesses = actor.eating_witness_ids
    for observer in observers(world, actor):
        if observer.id not in witnesses:
            witnesses.append(observer.id)


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
    if hasattr(actor, "location"):
        # An unreachable destination would leave the actor walking toward it forever.
        from locations import clamp_point
        reachable = clamp_point(actor.location, point)
        if tuple(point) != reachable:
            if getattr(actor, "destination", None) is not None and tuple(actor.destination) == tuple(point):
                actor.destination = reachable
            point = reachable
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
    from locations import clamp_point
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
                # The garden's walkable area is not its bounding box: the strip above it is only the bridge.
                snapped = tuple(clamp_point(actor.location, point) for actor, point in zip((first, second), points))
                if any(math.dist(point, fixed) > 1e-6 for point, fixed in zip(points, snapped)):
                    continue
                points = snapped
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
