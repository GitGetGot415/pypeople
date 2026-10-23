"""Local home decisions and transitions; no transport, drawing, or remote actions."""

from activity import ActivitySnapshot

from uuid import uuid4
from dataclasses import dataclass
from typing import Literal
from constants import (APARTMENT_ID, BEDTIME_DELAY_RANGE, CREATION_WORK_SECONDS_RANGE,
                      CREATIVE_QUEUE_SECONDS, INDOOR_RECOVERY_PER_SECOND, MINIMUM_INDOOR_SECONDS,
                      OUTDOOR_COOLDOWN_SECONDS, REST_SECONDS_RANGE, RETREAT_SOCIABILITY_WEIGHT,
                      RETREAT_THRESHOLD_BASE, WAKE_DELAY_RANGE, ROOM_SLEEP_POSITION, SLEEP_FACING,
                      PLAYER_ROOM_NUMBER, JUDY_ROOM_NUMBER, JUDY_WAKE_DELAY)
from creations import CreativeJob
from activity import Behavior


INDOOR_BEHAVIORS = (Behavior.REST, Behavior.SLEEP, Behavior.PAINT, Behavior.COMPOSE)
HOME_BEHAVIORS = (Behavior.GO_HOME, *INDOOR_BEHAVIORS)


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
    from locations import player_has_room
    next_room = max((p.room_number for p in world.residents), default=0) + 1
    for npc in world.npcs:
        if npc.room_number == 0:
            if next_room == PLAYER_ROOM_NUMBER and player_has_room(world):
                next_room += 1
            npc.room_number = next_room
            next_room += 1


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
    from locations import start_journey, door_point
    from constants import GARDEN_BUILDING_ARRIVAL, ARRIVAL_LOCATION
    from autonomy import next_arrival_to_greet
    greeting = (npc.room_number == JUDY_ROOM_NUMBER and world.phase == "day"
                and (world.pending_arrival_days or next_arrival_to_greet(world) is not None))
    start_journey(npc, ARRIVAL_LOCATION if greeting else "garden",
                  door_point(npc.room_number) if greeting else GARDEN_BUILDING_ARRIVAL,
                  purpose="arrival_greeting" if greeting else "leave_home")
    npc.home_reason = None
    npc.bedtime_remaining = npc.wake_remaining = None
    npc.home_remaining = 0.0
    npc.outdoor_remaining = OUTDOOR_COOLDOWN_SECONDS
    npc.command_revision += 1
    npc.begin_activity(Behavior.WAIT if greeting else Behavior.WANDER, world,
                       reason="Going to welcome a new arrival" if greeting else "Back from my apartment")


def sleep(npc):
    cancel_work(npc)
    npc.command_revision += 1
    npc.behavior = Behavior.SLEEP
    npc.x, npc.y = ROOM_SLEEP_POSITION
    npc.facing = SLEEP_FACING
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
    from autonomy import awaiting_arrival_greeting
    if awaiting_arrival_greeting(npc, world):
        return
    if (owns_resident(world, npc) or world.tree_has_actor(npc.id) or npc.journey
            or world.visit and world.visit.host_id == npc.id or npc.interaction):
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
    if (npc.interaction or npc.eating or npc.selected or npc.interaction_blocked or npc.player_call or npc.player_reply.closing
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
    if npc.behavior == Behavior.SLEEP:
        if world.phase != "day":
            return
        if npc.wake_remaining is None:
            npc.wake_remaining = (JUDY_WAKE_DELAY if npc.room_number == JUDY_ROOM_NUMBER
                                  else npc.rng.uniform(*WAKE_DELAY_RANGE))
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
        if owns_resident(world, npc) or world.visit and world.visit.host_id == npc.id or npc.interaction:
            continue
        if npc.behavior in INDOOR_BEHAVIORS:
            if world.phase == "night" and npc.behavior != Behavior.SLEEP:
                sleep(npc)
            if world.phase != "day":
                npc.wake_remaining = None
            elif npc.behavior == Behavior.SLEEP and npc.room_number == JUDY_ROOM_NUMBER:
                update_home(npc, world, 0.0)
        elif world.phase != "night" and npc.home_reason == "night":
            npc.home_reason = None
            npc.bedtime_remaining = None
            if npc.behavior == Behavior.GO_HOME:
                npc.finish_activity(world)


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
    visit = world.visit
    if not visit:
        return
    host = world.npc_by_id(visit.host_id)
    if not host or not world.player.alive or world.mode != "player":
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
            world.join_tree(world.player, "introduction")
    elif world.player.location != visit.location or host.location != visit.location:
        finish_visit(world)
    else:
        observe_visit(world)


def observe_visit(world):
    """An entrant can notice the current meeting, without its invitation or terms."""
    from locations import observers
    from social import add_fact, fact_by_id
    visit = world.visit
    if not visit or visit.stage != "hosting":
        return
    host = world.npc_by_id(visit.host_id)
    if host and host.location == world.player.location == visit.location:
        fact = fact_by_id(world, visit.observation_id)
        add_fact(world, fact.kind, host, world.player, fact.source_id, world.record_by_id(fact.conversation_id),
                 fact.summary, observers(world, host, world.player))


def validate_saved_visit(visit, invitations, player, by_id, layout, interaction, entries, reserved_ids, facts):
    from locations import valid_location, validate_feet
    from activity import ActivitySnapshot
    if visit:
        host = by_id.get(visit.host_id)
        if host is None or host is player or not valid_location(layout, visit.location):
            raise ValueError("A visit needs its actual living host and destination.")
        ActivitySnapshot.validate(visit.saved_state)
        offer = next((o for o in invitations if o.id == visit.offer_id), None)
        if (not offer or offer.status != "accepted" or offer.speaker_id != host.id
                or offer.location != visit.location or offer.destination != visit.destination
                or visit.stage not in ("travel", "hosting")):
            raise ValueError("A hosted visit requires its actual accepted invitation.")
        if visit.stage == "travel" and (interaction or player.id in {e.actor_id for e in entries}):
            raise ValueError("An accepted visit's travel cannot own another player interaction.")
        if visit.stage == "travel":
            for actor, purpose in ((host, "host_visit"), (player, "invitation")):
                if actor.journey:
                    if actor.journey.destination != visit.location or actor.journey.purpose != purpose:
                        raise ValueError("Host and visitor must travel to their accepted destination.")
                elif actor.location != visit.location:
                    raise ValueError("An unfinished visit needs its actual remaining journey.")
        if visit.stage == "hosting" and (host.location != visit.location or player.location != visit.location):
            raise ValueError("A host and visitor must be present together.")
        observation = facts.get(visit.observation_id)
        if (visit.stage == "travel" and visit.observation_id is not None
                or visit.stage == "hosting" and (observation is None or observation.kind != "visit"
                    or (observation.subject_id, observation.object_id) != (host.id, player.id)
                    or observation.conversation_id != offer.conversation_id)):
            raise ValueError("A hosted meeting requires its own committed visual account.")
        if not valid_location(layout, visit.return_location):
            raise ValueError("A host's return destination must remain valid.")
        validate_feet(visit.return_location, visit.return_point)
        if host.id in {e.actor_id for e in entries} or host.id in reserved_ids:
            raise ValueError("A host must stay available for the player.")


def validate_saved_invitations(invitations, records, residents, player_id, interaction, tree_point):
    from constants import GARDEN_VISIT_POINT, ROOM_ARRIVAL
    invitation_ids = set()
    for offer in invitations:
        record = records.get(offer.conversation_id)
        speaker = next((n for n in residents if n.id == offer.speaker_id), None)
        if (offer.id in invitation_ids or not record or not speaker or offer.status not in ("offered", "accepted", "withdrawn", "declined")
                or offer.destination not in ("garden", "tree", "home")
                or not 0 <= offer.turn < len(record.turns)
                or (record.turns[offer.turn].speaker_id, record.turns[offer.turn].text) != (speaker.id, offer.speech)
                or offer.location != (f"room:{speaker.room_number}" if offer.destination == "home" else "garden")):
            raise ValueError("Invitations must retain their real speaker, speech and destination.")
        expected_point = {"garden": GARDEN_VISIT_POINT, "home": ROOM_ARRIVAL,
                          "tree": tree_point}[offer.destination]
        if (tuple(offer.point) != tuple(expected_point)
                or set(record.participant_ids) != {speaker.id, player_id}):
            raise ValueError("An invitation needs its actual meeting point and conversation participants.")
        invitation_ids.add(offer.id)
        if offer.status == "offered" and (not interaction or interaction.kind != "resident"
                or offer.interaction_id != interaction.id or offer.speaker_id != interaction.npc_id):
            raise ValueError("A live invitation must belong to the current physical conversation.")
    if sum(o.status == "offered" for o in invitations) > 1:
        raise ValueError("Only the current speaker can keep a live invitation.")


def validate_saved_home(residents, creations, player, mode, day, phase, live_ids, owner_ids, visit, interaction, conversations):
    from types import SimpleNamespace
    from locations import player_has_room
    from social import phase_index
    layout = SimpleNamespace(residents=residents, mode=mode, player=player)
    if len({n.room_number for n in residents}) != len(residents):
        raise ValueError("Apartment rooms must be unique, including departed assignments.")
    creation_ids, job_ids = set(), set()
    for work in creations:
        if (work.id in creation_ids or work.job_id in job_ids or work.creator_id not in {n.id for n in residents}
                or work.owner_id not in owner_ids or phase_index(work.day, work.phase) > phase_index(day, phase)):
            raise ValueError("Invalid creation identity, credit, or completion time.")
        creation_ids.add(work.id)
        job_ids.add(work.job_id)
    for n in residents:
        if n.last_creative_day > day or n.sprite_size is not None and any(size <= 0 for size in n.sprite_size):
            raise ValueError("Invalid resident home state.")
        if n.departure and (n.creative_job or n.behavior in HOME_BEHAVIORS):
            raise ValueError("Departed residents cannot keep active home work.")
        if n.room_number == PLAYER_ROOM_NUMBER and player_has_room(layout):
            raise ValueError(f"Room {PLAYER_ROOM_NUMBER} is reserved for the player.")
        if n.behavior == Behavior.GO_HOME and not n.journey and n.location != f"room:{n.room_number}":
            raise ValueError("A home trip needs a journey to the owned room.")
        if n.behavior in INDOOR_BEHAVIORS and n.location != f"room:{n.room_number}" and not n.journey:
            raise ValueError("Home routines require the resident's real room.")
        job = n.creative_job
        effective_behavior = n.behavior
        frozen = (visit.saved_state if visit and visit.host_id == n.id else
                  interaction.saved_state if interaction and interaction.npc_id == n.id else None)
        for conversation in conversations:
            for index, activity in zip((conversation.initiator, conversation.recipient), conversation.saved_states):
                if index == n.id:
                    frozen = activity
        if frozen:
            effective_behavior = frozen.behavior
        if (effective_behavior in INDOOR_BEHAVIORS and n.location != f"room:{n.room_number}"
                and not n.journey and not (visit and visit.host_id == n.id)):
            raise ValueError("Suspended home work must retain its actual room or accepted visit.")
        if (job is None) != (effective_behavior not in (Behavior.PAINT, Behavior.COMPOSE)):
            raise ValueError("An active hobby needs exactly one creative job.")
        if job:
            if (job.id in job_ids or job.creator_id != n.id or n.id not in live_ids
                    or n.command_revision != job.revision or job.started_day != n.last_creative_day
                    or phase_index(job.started_day, job.started_phase) > phase_index(day, phase)
                    or job.remaining > CREATION_WORK_SECONDS_RANGE[1] or job.queue_remaining > CREATIVE_QUEUE_SECONDS
                    or effective_behavior != (Behavior.PAINT if job.kind == "art" else Behavior.COMPOSE)):
                raise ValueError("Invalid active creative job or completed job reuse.")
            job_ids.add(job.id)
    return creation_ids
