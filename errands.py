"""Player requests a resident cannot start yet: wait until the person is free, walk there, then act normally.

An errand never performs the action itself. Once both people are in the same place and free it queues the
ordinary pending action, so validation, conversations and handoffs stay in their owning code.
"""

import math
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import Field
from activity import Behavior
from constants import (CONVERSATION_CANCELLED, CONVERSATION_COMPLETED, CONVERSATION_FINISHED,
                       ERRAND_REPORT_CHARACTERS, ERRAND_REPORT_TURNS, ERRAND_STAND_SLACK, SOCIAL_RADIUS,
                       ROOM_ARRIVAL)


@dataclass
class Errand:
    id: Annotated[str, Field(min_length=1)]
    npc_id: str
    action: Literal["TALK_TO_NPC", "GIVE_ITEM"]
    target_id: str
    day: Annotated[int, Field(ge=1)]
    topic: str = ""
    record: str | None = None
    item: str = ""
    quantity: Annotated[int, Field(ge=0)] = 0
    report: bool = False
    status: Literal["waiting", "started", "done"] = "waiting"
    waiting_for: str = ""
    conversation: str | None = None
    summary: str = ""


def queue_errand(world, npc, intent, topic, record_id):
    """A new accepted request replaces any earlier one for this resident."""
    from locations import active_player
    cancel(world, npc.id)
    world.errands.append(Errand(
        id=uuid4().hex, npc_id=npc.id, action=intent.action, target_id=intent.target, day=world.day,
        topic=topic, record=record_id, item=intent.item, quantity=intent.quantity,
        report=intent.action == "TALK_TO_NPC" and active_player(world)))


def cancel(world, npc_id, reason=""):
    for errand in [e for e in world.errands if e.npc_id == npc_id]:
        _drop(world, errand, reason)


def release(world, actor_id, reason, *, as_target=True):
    """Drop errands that can no longer be carried out because this resident was interrupted or left."""
    for errand in [e for e in world.errands if e.npc_id == actor_id or as_target and e.target_id == actor_id]:
        _drop(world, errand, reason)


def expire(world):
    """Errands last for the day they were asked."""
    for errand in tuple(world.errands):
        _drop(world, errand, "The day ended before this request was carried out." if errand.status == "waiting" else "")


def attach(world, npc, conversation_id):
    """The pending talk has begun; remember its record so the result can be reported."""
    errand = world.errand_for(npc.id, "started")
    if errand is not None:
        errand.conversation = conversation_id


def _drop(world, errand, reason=""):
    world.errands.remove(errand)
    record = world.record_by_id(errand.record) if errand.record else None
    if reason and record is not None:
        if record.status == CONVERSATION_COMPLETED:
            record.status, record.error = CONVERSATION_CANCELLED, reason
        record.outcomes.append(reason)


def step(npc, world):
    """Advance this resident's errand. True means it holds the resident for this tick."""
    errand = world.errand_for(npc.id)
    if errand is None:
        return False
    if errand.status == "started":
        return _follow_action(world, npc, errand)
    if errand.status == "done":
        return _report(world, npc, errand)
    target = world.npc_by_id(errand.target_id)
    if target is None or not target.alive:
        _drop(world, errand, "The person is no longer here.")
        return False
    if errand.action == "GIVE_ITEM" and not _holds(world, npc, errand):
        _drop(world, errand, f"{npc.name} no longer has what was to be given.")
        return False
    if npc.behavior not in (Behavior.WANDER, Behavior.WAIT):
        return False
    if npc.location != target.location:
        return _approach(world, npc, errand, target)
    from interactions import company_point
    if not _ready(world, target) or company_point(npc, target) is None:
        _wait(world, errand, f"{target.name} is busy.")
        _stand_near(npc, target)
        return True
    from simulation import PendingAction
    from transfer_records import TransferIntent
    record = world.record_by_id(errand.record) if errand.record else None
    if errand.action == "GIVE_ITEM" and record is not None:
        record.transfer_intent = TransferIntent(giver_id=npc.id, recipient_id=errand.target_id,
                                                item=errand.item, quantity=errand.quantity)
    world.pending_actions.append(PendingAction(npc.id, errand.action, npc.command_revision, errand.target_id,
                                               command_topic(npc, target, errand.action, errand.topic),
                                               errand.record, errand.item, errand.quantity))
    errand.status = "started"
    return True


def command_topic(npc, target, action, message):
    """The player's own words when typed; a picked command has none, so say who asked for what."""
    if message:
        return message
    task = "talk with" if action == "TALK_TO_NPC" else "give something to"
    return f"{npc.name} was asked by the player to {task} {target.name}."


def _hold(npc, reason):
    """Stand still instead of wandering off while the errand waits."""
    npc.behavior, npc.destination, npc.reason = Behavior.WAIT, None, reason


def _stand_near(npc, target):
    """Wait beside the busy person, leaving room for whoever they are with, and follow if they shift."""
    from interactions import company_point
    npc.behavior, npc.reason = Behavior.WAIT, f"Waiting to talk to {target.name}"
    point = company_point(npc, target, SOCIAL_RADIUS, allow_vertical=True)
    if point is not None and math.dist(point, (npc.x, npc.y)) > ERRAND_STAND_SLACK:
        npc.destination = point


def _holds(world, npc, errand):
    if errand.item == "fruit":
        return npc.inventory.get("fruit", 0) >= errand.quantity
    return any(work.id == errand.item and work.owner_id == npc.id for work in world.creations)


def _ready(world, target):
    from player import directed_recipient_available
    from apartments import creating_kind
    return (directed_recipient_available(target) and not creating_kind(target)
            and world.action_for(target.id) is None)


def _wait(world, errand, reason):
    """Show the first reason a queued request is waiting on the record that asked for it."""
    if reason == errand.waiting_for:
        return
    errand.waiting_for = reason
    record = world.record_by_id(errand.record) if errand.record else None
    if record is not None:
        record.outcomes[0:1] = [f"Queued: {reason}"]


def _approach(world, npc, errand, target):
    """Walk to the target's place; a closed room needs a knock, and the host decides."""
    from locations import clamp_point, floor_for_room, hallway_location, door_point, start_journey
    from apartments import entry_refusal
    if target.journey:
        _hold(npc, f"Waiting to see where {target.name} goes")
        return True
    destination = target.location
    try:
        if destination.startswith("room:"):
            number = int(destination.split(":")[1])
            hallway = hallway_location(floor_for_room(number))
            if npc.location != hallway:
                start_journey(npc, hallway, door_point(number), purpose="errand")
                return True
            reason, final = entry_refusal(world, npc, target)
            if final:
                _drop(world, errand, reason)
                return False
            if reason:
                _wait(world, errand, reason)
                _hold(npc, f"Waiting outside {target.name}'s door")
                return True
            start_journey(npc, destination, ROOM_ARRIVAL, purpose="errand")
            return True
        start_journey(npc, destination, clamp_point(destination, (target.x + SOCIAL_RADIUS, target.y)), purpose="errand")
    except ValueError:
        _drop(world, errand, "There was no way to reach that person.")
        return False
    return True


def _follow_action(world, npc, errand):
    if world.action_for(npc.id) is not None or npc.conversation is not None:
        return True
    record = world.record_by_id(errand.conversation) if errand.conversation else None
    if errand.action != "TALK_TO_NPC" or record is None:
        _drop(world, errand)
        return False
    if record.status == CONVERSATION_FINISHED and errand.report:
        errand.status, errand.summary = "done", _summary(world, errand, record)
        return True
    _drop(world, errand)
    return False


def _summary(world, errand, record):
    target = world.npc_by_id(errand.target_id)
    spoken = " ".join(f"{turn.speaker}: {turn.text}" for turn in record.turns[-ERRAND_REPORT_TURNS:])
    about = f' about "{errand.topic}"' if errand.topic else ""
    return (f"You were asked to talk to {target.name}{about}, and you did. "
            f"What was said: {spoken}")[:ERRAND_REPORT_CHARACTERS]


def _report(world, npc, errand):
    """Find the player and wait to be called over; the call itself is the ordinary player call."""
    from locations import active_player, colocated, clamp_point, floor_for_room, hallway_location, door_point, start_journey
    if not active_player(world) or f"report:{errand.id}" in npc.resolved_player_calls:
        _drop(world, errand)
        return False
    if npc.behavior not in (Behavior.WANDER, Behavior.WAIT):
        return False
    player = world.player
    if colocated(npc, player) or player.journey:
        _hold(npc, "Waiting to tell the player what happened")
        return True
    destination, point = player.location, clamp_point(player.location, (player.x + SOCIAL_RADIUS, player.y))
    if destination.startswith("room:"):
        number = int(destination.split(":")[1])
        destination, point = hallway_location(floor_for_room(number)), door_point(number)
        if npc.location == destination:
            _hold(npc, "Waiting to tell the player what happened")
            return True
    try:
        start_journey(npc, destination, point, purpose="errand")
    except ValueError:
        _drop(world, errand)
        return False
    return True
