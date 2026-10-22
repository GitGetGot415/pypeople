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
    left, right, top, bottom = npc.movement_bounds
    return (min(right, max(left, APARTMENT_APPROACH[0])), min(bottom, max(top, APARTMENT_APPROACH[1])))


def assign_rooms(world):
    next_room = max((p.room_number for p in world.residents), default=0) + 1
    for npc in world.npcs:
        if npc.room_number == 0:
            npc.room_number = next_room
            next_room += 1


def available_outdoors(npc):
    return (npc.departure is None and not npc.indoors and npc.behavior != Behavior.GO_HOME
            and npc.home_reason != "night")


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
    npc.x, npc.y = entrance(npc)
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
    if npc.indoors:
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
    npc.destination = entrance(npc)


def update_home(npc, world, dt):
    if npc.behavior == Behavior.GO_HOME:
        if npc.selected or npc.request_pending:
            return
        npc.destination = entrance(npc)  # Sprite dimensions may be prepared after restoration.
        if math.dist((npc.x, npc.y), npc.destination) <= HOME_ARRIVAL_TOLERANCE or npc._move(dt):
            npc.x, npc.y = npc.destination
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
        if npc.indoors:
            if world.phase == "night" and npc.behavior != Behavior.SLEEP:
                sleep(npc)
            if world.phase != "day":
                npc.wake_remaining = None
        elif world.phase != "night" and npc.home_reason == "night":
            npc.home_reason = None
            npc.bedtime_remaining = None
            if npc.behavior == Behavior.GO_HOME:
                npc.finish_activity(world)
