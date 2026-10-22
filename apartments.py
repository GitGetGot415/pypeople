"""Local home decisions and transitions; no transport, drawing, or remote actions."""

import math
from uuid import uuid4

from constants import (APARTMENT_APPROACH, APARTMENT_ID, BEDTIME_DELAY_RANGE, CREATION_WORK_SECONDS_RANGE,
                       CREATIVE_QUEUE_SECONDS, HOME_ARRIVAL_TOLERANCE, INDOOR_RECOVERY_PER_SECOND,
                       MINIMUM_INDOOR_SECONDS, OUTDOOR_COOLDOWN_SECONDS, REST_SECONDS_RANGE,
                       RETREAT_SOCIABILITY_WEIGHT, RETREAT_THRESHOLD_BASE, WAKE_DELAY_RANGE)
from creations import CreativeJob
from npc import Behavior


INDOOR_BEHAVIORS = (Behavior.REST, Behavior.SLEEP, Behavior.PAINT, Behavior.COMPOSE)
HOME_BEHAVIORS = (Behavior.GO_HOME, *INDOOR_BEHAVIORS)


def entrance(npc):
    from constants import GARDEN_BUILDING_POINT
    return GARDEN_BUILDING_POINT


def assign_rooms(world):
    next_room = max(1, max((p.room_number for p in world.residents), default=1)) + 1
    for npc in world.npcs:
        if npc.room_number == 0:
            npc.room_number = next_room
            next_room += 1


def available_outdoors(npc):
    from locations import available
    return available(npc)


def cancel_work(npc):
    npc.creative_job = None


def relax(npc):
    cancel_work(npc)
    npc.behavior = Behavior.REST
    npc.home_remaining = max(npc.home_remaining, npc.rng.uniform(*REST_SECONDS_RANGE))
    npc.reason = "Relaxing in my apartment"


def discard_work(npc):
    if npc.behavior in (Behavior.PAINT, Behavior.COMPOSE):
        relax(npc)
    else:
        cancel_work(npc)


def leave(npc, world):
    cancel_work(npc)
    from locations import start_journey
    from constants import GARDEN_BUILDING_ARRIVAL
    start_journey(npc, "garden", GARDEN_BUILDING_ARRIVAL, purpose="leave_home")
    npc.home_reason = None
    npc.bedtime_remaining = npc.wake_remaining = None
    npc.home_remaining = 0.0
    npc.outdoor_remaining = OUTDOOR_COOLDOWN_SECONDS
    npc.command_revision += 1
    npc.begin_activity(Behavior.WANDER, world, reason="Back from my apartment")


def sleep(npc):
    cancel_work(npc)
    npc.command_revision += 1
    npc.behavior = Behavior.SLEEP
    npc.home_reason = "night"
    npc.home_remaining = MINIMUM_INDOOR_SECONDS
    npc.wake_remaining = None
    npc.reason = "Sleeping until daytime"


def enter(npc, world):
    from locations import room_location
    from constants import ROOM_ARRIVAL
    npc.location = room_location(npc.room_number)
    npc.x, npc.y = ROOM_ARRIVAL
    npc.release_claims(world)
    npc.destination = npc.target_id = None
    npc.activity_elapsed = 0.0
    npc.command_revision += 1
    npc.home_remaining = MINIMUM_INDOOR_SECONDS
    if world.phase == "night":
        sleep(npc)
        return
    if npc.last_creative_day == world.day:
        relax(npc)
        return
    choice = npc.rng.choice((Behavior.REST, Behavior.PAINT, Behavior.COMPOSE))
    if choice == Behavior.REST:
        relax(npc)
        return
    npc.behavior = choice
    npc.last_creative_day = world.day
    npc.creative_job = CreativeJob(id=uuid4().hex, creator_id=npc.id, kind="art" if choice == Behavior.PAINT else "music",
        started_day=world.day, started_phase=world.phase, revision=npc.command_revision,
        remaining=npc.rng.uniform(*CREATION_WORK_SECONDS_RANGE), queue_remaining=CREATIVE_QUEUE_SECONDS)
    npc.reason = "Painting in my apartment" if choice == Behavior.PAINT else "Composing in my apartment"


def home_priority(npc, world, dt):
    """Run after existing interactions finish and before anyone reserves partners."""
    npc.outdoor_remaining = max(0.0, npc.outdoor_remaining - dt)
    from player import owns_resident
    if (owns_resident(world, npc) or world.tree_has_actor(npc.id) or npc.journey
            or world.visit and world.visit["host_id"] == npc.id or npc.interaction):
        return
    if npc.behavior in INDOOR_BEHAVIORS:
        if world.phase == "night" and npc.behavior != Behavior.SLEEP:
            sleep(npc)
        return
    if world.phase != "night" and npc.home_reason == "night":
        npc.home_reason = None
        npc.bedtime_remaining = None
        if npc.behavior == Behavior.GO_HOME:
            npc.finish_activity(world)
    if (npc.interaction or npc.eating or npc.selected or npc.request_pending or npc.player_call or npc.end_conversation
            or npc.owner != "autonomous" or npc.behavior == Behavior.GO_HOME
            or any(npc.id in (action.npc_id, action.target_id) for action in world.pending_actions)):
        return
    if world.phase == "night":
        if npc.bedtime_remaining is None:
            npc.home_reason = "night"
            npc.bedtime_remaining = npc.rng.uniform(*BEDTIME_DELAY_RANGE)
        npc.bedtime_remaining = max(0.0, npc.bedtime_remaining - dt)
        if npc.bedtime_remaining > 0:
            return
        reason = "night"
    else:
        if (npc.social_fatigue < RETREAT_THRESHOLD_BASE + RETREAT_SOCIABILITY_WEIGHT * npc.preferences["sociability"]
                or npc.outdoor_remaining > 0):
            return
        # Deliver an already available promised apple before an optional retreat.
        from social import pending_promises
        for promise in pending_promises(world):
            if npc.id not in (promise.promisor_id, promise.recipient_id):
                continue
            giver, receiver = world.npc_by_id(promise.promisor_id), world.npc_by_id(promise.recipient_id)
            if (giver and receiver and giver.inventory.get("fruit", 0)
                    and world.company_available(giver) and world.company_available(receiver)
                    and receiver.id not in giver.cooldowns and giver.id not in receiver.cooldowns):
                from interactions import company_point
                if company_point(giver, receiver) is not None:
                    giver.decision_remaining = 0.0
                    return
        if npc.behavior not in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY):
            return
        reason = "retreat"
    npc.command_revision += 1
    npc.begin_activity(Behavior.GO_HOME, world, APARTMENT_ID, "Going home to sleep" if reason == "night" else "Taking some time alone")
    npc.home_reason = reason
    from locations import room_location, start_journey
    from constants import ROOM_ARRIVAL
    start_journey(npc, room_location(npc.room_number), ROOM_ARRIVAL, purpose="home")


def update_home(npc, world, dt):
    if npc.behavior == Behavior.GO_HOME:
        if npc.selected or npc.request_pending:
            return
        from locations import room_location
        if npc.journey:
            return
        if npc.location == room_location(npc.room_number):
            enter(npc, world)
        return
    npc.social_fatigue = max(0.0, npc.social_fatigue - INDOOR_RECOVERY_PER_SECOND * dt)
    npc.home_remaining = max(0.0, npc.home_remaining - dt)
    if npc.behavior == Behavior.SLEEP:
        if world.phase != "day":
            return
        if npc.wake_remaining is None:
            npc.wake_remaining = npc.rng.uniform(*WAKE_DELAY_RANGE)
        npc.wake_remaining = max(0.0, npc.wake_remaining - dt)
        if npc.wake_remaining == 0:
            leave(npc, world)
        return
    job = npc.creative_job
    if job:
        if not world.is_creative_job(npc, job):
            discard_work(npc)
            return
        job.remaining = max(0.0, job.remaining - dt)
        if job.accepted is not None and job.remaining == 0:
            world.commit_creation(npc)
        return
    if npc.home_remaining == 0:
        leave(npc, world)


def phase_changed(world):
    """Keep phase-sensitive state consistent even if the player saves immediately."""
    for npc in world.npcs:
        from player import owns_resident
        if owns_resident(world, npc) or world.visit and world.visit["host_id"] == npc.id or npc.interaction:
            continue
        if npc.behavior in INDOOR_BEHAVIORS:
            if world.phase == "night" and npc.behavior != Behavior.SLEEP:
                sleep(npc)
            if world.phase != "day":
                npc.wake_remaining = None
        elif world.phase != "night" and npc.home_reason == "night":
            npc.home_reason = None
            npc.bedtime_remaining = None
            if npc.behavior == Behavior.GO_HOME:
                npc.finish_activity(world)


def invitation_choices(world, npc):
    from locations import room_location
    from constants import ROOM_ARRIVAL, STARTING_PLAYER_POSITION
    return [dict(id="garden", location="garden", point=STARTING_PLAYER_POSITION, label="the garden"),
            dict(id="tree", location="garden", point=world.tree.interaction_point, label="the tree"),
            dict(id="home", location=room_location(npc.room_number), point=ROOM_ARRIVAL, label="my apartment")]


def offer_invitation(world, npc, record, destination, speech):
    choices = {c["id"]: c for c in invitation_choices(world, npc)}
    if destination not in choices or not speech.strip() or not world.player_interaction:
        raise ValueError("That invitation has no supported destination.")
    for offer in world.player_invitations:
        if offer["status"] == "offered" and offer["speaker_id"] == npc.id:
            offer["status"] = "withdrawn"
    choice = choices[destination]
    offer = dict(id=uuid4().hex, speaker_id=npc.id, conversation_id=record.id,
        turn=len(record.turns) - 1, interaction_id=world.player_interaction["id"], destination=destination,
        location=choice["location"], point=list(choice["point"]), status="offered", speech=speech)
    world.player_invitations.append(offer)
    return offer


def accept_invitation(world, offer_id):
    from player import active_player, end_interaction
    from locations import start_journey
    offer = next((o for o in world.player_invitations if o["id"] == offer_id and o["status"] == "offered"), None)
    if not offer or not active_player(world) or not world.player_interaction or offer["interaction_id"] != world.player_interaction["id"]:
        raise ValueError("This invitation is no longer available.")
    host = world.npc_by_id(offer["speaker_id"])
    if host is None or host.busy or host.request_pending:
        raise ValueError("The host is no longer here.")
    saved = world.player_interaction["saved_state"]
    offer["status"] = "accepted"
    end_interaction(world, "Invitation accepted; travelling together.")
    host.resume_pending = False
    host.behavior, host.destination, host.target_id = Behavior.WAIT, None, None
    world.visit = dict(host_id=host.id, saved_state=saved, offer_id=offer_id,
                       location=offer["location"], stage="travel", destination=offer["destination"],
                       return_location=host.location, return_point=[host.x, host.y])
    from constants import HOST_MEETING_OFFSET
    start_journey(host, offer["location"], (offer["point"][0] + HOST_MEETING_OFFSET[0],
        offer["point"][1] + HOST_MEETING_OFFSET[1]), purpose="host_visit")
    start_journey(world.player, offer["location"], offer["point"], purpose="invitation", fast=True)


def finish_visit(world):
    visit, world.visit = world.visit, None
    if not visit:
        return
    if world.player.journey and world.player.journey["purpose"] == "invitation":
        world.player.journey = None
    host = world.npc_by_id(visit["host_id"])
    if host:
        host.journey = None
        host.restore_after_interaction(visit["saved_state"], world.player.id)
        if host.location != visit["return_location"]:
            from locations import start_journey
            start_journey(host, visit["return_location"], visit["return_point"], purpose="return_after_visit")
        # A host who came from home takes the same connections back there.
        if host.behavior in INDOOR_BEHAVIORS:
            from locations import room_location, start_journey
            from constants import ROOM_ARRIVAL
            if host.location != room_location(host.room_number):
                start_journey(host, room_location(host.room_number), ROOM_ARRIVAL, purpose="return_home")


def update_visit(world, dt):
    visit = world.visit
    if not visit:
        return
    host = world.npc_by_id(visit["host_id"])
    if not host or not world.player.alive or world.mode != "player":
        finish_visit(world)
        return
    if visit["stage"] == "travel":
        if host.journey or world.player.journey:
            return
        if host.location != visit["location"] or world.player.location != visit["location"]:
            finish_visit(world)
            return
        visit["stage"] = "hosting"
        record = next(r for r in world.history if r.id == next(o for o in world.player_invitations if o["id"] == visit["offer_id"])["conversation_id"])
        world.observed_event("visit", host, world.player,
            f"{host.name} met the player for their accepted visit {'in the apartment' if visit['location'].startswith('room:') else 'in the garden'}.", record)
        if visit["destination"] == "tree":
            finish_visit(world)
            world.join_tree(world.player, "introduction")
    elif world.player.location != visit["location"] or host.location != visit["location"]:
        finish_visit(world)
