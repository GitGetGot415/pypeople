"""Local home decisions and transitions; no transport, drawing, or remote actions."""

from activity import ActivitySnapshot

from uuid import uuid4
from dataclasses import dataclass
from typing import Literal
from constants import (APARTMENT_ID, CREATION_WORK_SECONDS_RANGE,
                      CREATIVE_QUEUE_SECONDS, INDOOR_RECOVERY_PER_SECOND, MINIMUM_INDOOR_SECONDS,
                      OUTDOOR_COOLDOWN_SECONDS, REST_SECONDS_RANGE, RETREAT_SOCIABILITY_WEIGHT,
                      RETREAT_THRESHOLD_BASE, PLAYER_ROOM_NUMBER, MAX_RESIDENTS)
from creations import CreativeJob
from activity import Behavior


INDOOR_BEHAVIORS = (Behavior.REST, Behavior.PAINT, Behavior.COMPOSE)
HOME_BEHAVIORS = (Behavior.GO_HOME, *INDOOR_BEHAVIORS)


def creating_kind(npc):
    """'art' or 'music' while the resident is working on a creation, else None."""
    return {Behavior.PAINT: "art", Behavior.COMPOSE: "music"}.get(npc.behavior)


def entry_refusal(world, visitor, host):
    """Why a knock at the host's door is not answered, and whether asking again today could still help."""
    from player import directed_recipient_available
    from constants import ENTRY_REFUSAL_SCORE, ENTRY_SOCIABILITY_WEIGHT
    kind = creating_kind(host)
    if kind:
        return f"{host.name} is busy making {kind}.", False
    if not directed_recipient_available(host):
        return f"{host.name} is busy.", False
    score = host.opinion_score(visitor.id) + (host.preferences["sociability"] - 0.5) * ENTRY_SOCIABILITY_WEIGHT
    if score < ENTRY_REFUSAL_SCORE:
        return f"{host.name} would not let {visitor.name} in.", True
    return "", False


@dataclass
class Invitation:
    id: str
    speaker_id: str
    conversation_id: str
    turn: int
    interaction_id: str
    destination: Literal["garden", "tree", "home"]
    location: str
    point: list[float]
    status: Literal["offered", "accepted", "withdrawn", "declined"]
    speech: str


@dataclass
class HostedVisit:
    host_id: str
    saved_state: ActivitySnapshot
    offer_id: str
    location: str
    stage: Literal["travel", "hosting"]
    destination: str
    return_location: str
    return_point: list[float]
    observation_id: str | None = None


def entrance(npc):
    from constants import GARDEN_BUILDING_POINT
    return GARDEN_BUILDING_POINT


def assign_rooms(world):
    from locations import active_player
    occupied = {npc.room_number for npc in world.npcs if npc.room_number}
    if active_player(world):
        occupied.add(PLAYER_ROOM_NUMBER)
    for npc in world.npcs:
        if npc.room_number == 0:
            room = next((number for number in range(1, MAX_RESIDENTS + 1) if number not in occupied), None)
            if room is None:
                raise ValueError("The apartment building is full.")
            npc.room_number = room
            occupied.add(room)


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
    npc.home_remaining = 0.0
    npc.outdoor_remaining = OUTDOOR_COOLDOWN_SECONDS
    npc.command_revision += 1
    npc.begin_activity(Behavior.WANDER, world, reason="Back from my apartment")


def enter(npc, world):
    from locations import room_location
    from constants import ROOM_ARRIVAL
    npc.location = room_location(npc.room_number)
    npc.x, npc.y = ROOM_ARRIVAL
    npc.destination = npc.target_id = None
    npc.activity_elapsed = 0.0
    npc.command_revision += 1
    npc.home_remaining = MINIMUM_INDOOR_SECONDS
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
        started_day=world.day, revision=npc.command_revision,
        remaining=npc.rng.uniform(*CREATION_WORK_SECONDS_RANGE), queue_remaining=CREATIVE_QUEUE_SECONDS)
    npc.reason = "Painting in my apartment" if choice == Behavior.PAINT else "Composing in my apartment"


def home_priority(npc, world, dt):
    """Run after existing interactions finish and before anyone reserves partners."""
    npc.outdoor_remaining = max(0.0, npc.outdoor_remaining - dt)
    from player import owns_resident
    from autonomy import arrival_greeting_reserved
    if arrival_greeting_reserved(npc, world):
        return
    if (owns_resident(world, npc) or npc.journey
            or world.visit and world.visit.host_id == npc.id or npc.interaction):
        return
    if npc.behavior in INDOOR_BEHAVIORS:
        return
    if (npc.interaction or npc.eating or npc.selected or npc.interaction_blocked or npc.player_call or npc.player_reply.closing
            or npc.owner != "autonomous" or npc.behavior == Behavior.GO_HOME
            or any(npc.id in (action.npc_id, action.target_id) for action in world.pending_actions)):
        return
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
    if npc.behavior not in (Behavior.WANDER, Behavior.WAIT):
        return
    npc.command_revision += 1
    npc.begin_activity(Behavior.GO_HOME, world, APARTMENT_ID, "Taking some time alone")
    npc.home_reason = "retreat"
    from locations import room_location, start_journey
    from constants import ROOM_ARRIVAL
    start_journey(npc, room_location(npc.room_number), ROOM_ARRIVAL, purpose="home")


def update_home(npc, world, dt):
    if npc.behavior == Behavior.GO_HOME:
        if npc.selected or npc.interaction_blocked:
            return
        from locations import room_location
        if npc.journey:
            return
        if npc.location == room_location(npc.room_number):
            enter(npc, world)
        return
    npc.social_fatigue = max(0.0, npc.social_fatigue - INDOOR_RECOVERY_PER_SECOND * dt)
    npc.home_remaining = max(0.0, npc.home_remaining - dt)
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


def invitation_choices(world, npc):
    from locations import room_location
    from constants import ROOM_ARRIVAL, GARDEN_VISIT_POINT
    return [dict(id="garden", location="garden", point=GARDEN_VISIT_POINT, label="the garden"),
            dict(id="tree", location="garden", point=world.tree.interaction_point, label="the tree"),
            dict(id="home", location=room_location(npc.room_number), point=ROOM_ARRIVAL, label="my apartment")]


def pending_invitation(world):
    interaction = world.player_interaction
    return next((offer for offer in reversed(world.player_invitations) if offer.status == "offered"
                 and interaction and offer.interaction_id == interaction.id), None)


def offer_invitation(world, npc, record, destination, speech):
    choices = {c["id"]: c for c in invitation_choices(world, npc)}
    if destination not in choices or not speech.strip() or not world.player_interaction:
        raise ValueError("That invitation has no supported destination.")
    for offer in world.player_invitations:
        if offer.status == "offered" and offer.speaker_id == npc.id:
            offer.status = "withdrawn"
    choice = choices[destination]
    offer = Invitation(id=uuid4().hex, speaker_id=npc.id, conversation_id=record.id,
        turn=len(record.turns) - 1, interaction_id=world.player_interaction.id, destination=destination,
        location=choice["location"], point=list(choice["point"]), status="offered", speech=speech)
    world.player_invitations.append(offer)
    return offer


def accept_invitation(world, offer_id):
    from player import active_player, end_interaction
    from locations import start_journey
    offer = next((o for o in world.player_invitations if o.id == offer_id and o.status == "offered"), None)
    if not offer or not active_player(world) or not world.player_interaction or offer.interaction_id != world.player_interaction.id:
        raise ValueError("This invitation is no longer available.")
    host = world.npc_by_id(offer.speaker_id)
    if host is None or host.busy or host.interaction_blocked:
        raise ValueError("The host is no longer here.")
    saved = world.player_interaction.saved_state
    offer.status = "accepted"
    end_interaction(world, "Invitation accepted; travelling together.")
    host.resume_pending = False
    host.behavior, host.destination, host.target_id = Behavior.WAIT, None, None
    world.visit = HostedVisit(host_id=host.id, saved_state=saved, offer_id=offer_id,
                       location=offer.location, stage="travel", destination=offer.destination,
                       return_location=host.location, return_point=[host.x, host.y])
    from constants import HOST_MEETING_OFFSET
    # At home, the host waits farther into the room, clear of its bottom exit.
    host_offset_y = -HOST_MEETING_OFFSET[1] if offer.destination == "home" else HOST_MEETING_OFFSET[1]
    start_journey(host, offer.location, (offer.point[0] + HOST_MEETING_OFFSET[0],
        offer.point[1] + host_offset_y), purpose="host_visit")
    start_journey(world.player, offer.location, offer.point, purpose="invitation", fast=True)


def finish_visit(world):
    visit, world.visit = world.visit, None
    if not visit:
        return
    if world.player.journey and world.player.journey.purpose == "invitation":
        world.player.journey = None
    host = world.npc_by_id(visit.host_id)
    if host:
        host.journey = None
        host.restore_after_interaction(visit.saved_state, world.player.id)
        if host.location != visit.return_location:
            from locations import start_journey
            start_journey(host, visit.return_location, visit.return_point, purpose="return_after_visit")
        # A host who came from home takes the same connections back there.
        if host.behavior in INDOOR_BEHAVIORS:
            from locations import room_location, start_journey
            from constants import ROOM_ARRIVAL
            if host.location != room_location(host.room_number):
                start_journey(host, room_location(host.room_number), ROOM_ARRIVAL, purpose="return_home")


def update_visit(world, dt):
    from locations import active_player
    visit = world.visit
    if not visit:
        return
    host = world.npc_by_id(visit.host_id)
    if not host or not active_player(world):
        finish_visit(world)
        return
    if visit.stage == "travel":
        if host.journey or world.player.journey:
            return
        if host.location != visit.location or world.player.location != visit.location:
            finish_visit(world)
            return
        visit.stage = "hosting"
        record = next(r for r in world.history if r.id == next(o for o in world.player_invitations if o.id == visit.offer_id).conversation_id)
        _, fact = world.observed_event("visit", host, world.player,
            f"{host.name} and {world.player.name} met.", record)
        visit.observation_id = fact.id
        if visit.destination == "tree":
            finish_visit(world)
    elif world.player.location != visit.location or host.location != visit.location:
        finish_visit(world)
    else:
        observe_visit(world)


def observe_visit(world):
    """An entrant can notice the current meeting, without its invitation or terms."""
    from locations import observers
    from social_records import fact_by_id
    from information import add_fact
    visit = world.visit
    if not visit or visit.stage != "hosting":
        return
    host = world.npc_by_id(visit.host_id)
    if host and host.location == world.player.location == visit.location:
        fact = fact_by_id(world, visit.observation_id)
        add_fact(world, fact.kind, host, world.player, fact.source_id, world.record_by_id(fact.conversation_id),
                 fact.summary, observers(world, host, world.player))
