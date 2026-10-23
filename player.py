"""Physical player and exclusive social ownership; possessions remain engine owned."""

from dataclasses import dataclass, field
import re
from uuid import uuid4

from constants import (PLAYER_ID, STARTING_PLAYER_FRUIT, PLAYER_CALL_LIMIT, PLAYER_CALL_COOLDOWN,
                      PLAYER_CALL_CHECK_SECONDS, PLAYER_CONTEXT_WORK_LIMIT, PLAYER_OPINION_STEP,
                      PLAYER_CALL_GIFT_GENEROSITY, PLAYER_CALL_SOCIABILITY)
from conversation_history import remember
from dialogue_text import contains_question
from constants import (GARDEN_LOCATION, ARRIVAL_LOCATION, STARTING_PLAYER_POSITION, STARTING_FACING,
                      PLAYER_DEFAULT_APPEARANCE, MOVEMENT_SPEED, PLAYER_APPROACH_MULTIPLIER,
                      PLAYER_RUN_MULTIPLIER, PLAYER_APPROACH_TIMEOUT, PLAYER_EVIDENCE_LIMIT)


@dataclass
class Player:
    sprite_size = None
    id: str = PLAYER_ID
    inventory: dict = field(default_factory=lambda: {"fruit": STARTING_PLAYER_FRUIT})
    x: float = STARTING_PLAYER_POSITION[0]
    y: float = STARTING_PLAYER_POSITION[1]
    facing: int = STARTING_FACING
    location: str = ARRIVAL_LOCATION
    alive: bool = True
    appearance: dict = field(default_factory=lambda: PLAYER_DEFAULT_APPEARANCE.copy())
    journey: dict | None = None

    @property
    def name(self):
        return "the player"

    @property
    def movement_bounds(self):
        from locations import feet_bounds
        return feet_bounds(self.location)

    def can_reach(self, point):
        from locations import clamp_point
        return tuple(point) == clamp_point(self.location, point)


def active_player(world):
    return world.mode == "player" and world.player.alive


def owns_resident(world, npc):
    return bool(world.player_interaction and world.player_interaction.get("npc_id") == npc.id)


def owns_conversation(world, npc, *, stages=("conversation",)):
    interaction = world.player_interaction if world else None
    return bool(npc and world and active_player(world) and interaction
                and interaction.get("npc_id") == npc.id and interaction["stage"] in stages)


def can_claim(world):
    return active_player(world) and not (world.player_interaction or world.player.journey
        or world.tree_has_actor(world.player.id) or world.player_menu_blocked
        or world.visit and world.visit["stage"] == "travel")


def start_approach(world, npc, *, incoming=False, gift=None, knock_room=None):
    from locations import available, colocated
    from interactions import company_point
    if not can_claim(world) or world.npc_by_id(npc.id) is not npc or not available(npc):
        raise ValueError("The player or resident is unavailable.")
    if npc.busy or npc.request_pending or world.tree_has_actor(npc.id) or npc.journey:
        raise ValueError("The resident is occupied.")
    if not colocated(world.player, npc) and knock_room is None:
        raise ValueError("You must be in the same location.")
    mover, waiting = (npc, world.player) if incoming else (world.player, npc)
    point = company_point(mover, waiting, allow_vertical=True) if knock_room is None else (world.player.x, world.player.y)
    if point is None:
        raise ValueError("There is no reachable place to meet.")
    state = npc.suspend_activity("Meeting the player")
    world.player_interaction = dict(id=uuid4().hex, kind="resident", npc_id=npc.id,
        location=world.player.location, stage="approach", incoming=incoming, mover_id=mover.id,
        point=list(point), elapsed=0.0, saved_state=state, record_id=None,
        gift=gift, knock_room=knock_room)
    if knock_room is not None:
        # Talking through a closed door does not admit either actor or an audience.
        world.player_interaction["stage"] = "conversation"
    return world.player_interaction


def end_interaction(world, reason="Conversation ended."):
    """Release once; callers cancel requests using the returned ownership ID."""
    interaction, world.player_interaction = world.player_interaction, None
    if not interaction:
        return
    world.player.journey = None
    npc = world.npc_by_id(interaction.get("npc_id"))
    record = next((r for r in world.history if r.id == interaction.get("record_id")), None)
    if record:
        for offer in record.player_promise_offers:
            if offer["status"] == "offered":
                offer["status"] = "closed"
        if record.status in ("Waiting for reply", "Reply pending when saved"):
            record.status, record.error = "Cancelled", reason
    if npc:
        if npc.gift:
            npc.gift.finish(reason)
        npc.request_pending = False
        npc.selected = False
        npc.end_conversation = False
        npc.journey = None
        clear_call(npc)
        if interaction.get("saved_state"):
            npc.restore_after_interaction(interaction["saved_state"], world.player.id, preserve_facing=True)
    for invitation in world.player_invitations:
        if invitation["status"] == "offered" and invitation["interaction_id"] == interaction["id"]:
            invitation["status"] = "withdrawn"


def update_interaction(world, dt):
    from interactions import face_each_other, move_toward
    from locations import colocated
    interaction = world.player_interaction
    if not interaction or interaction["kind"] != "resident":
        return
    npc = world.npc_by_id(interaction["npc_id"])
    if (not active_player(world) or npc is None or npc.departure
            or not interaction.get("knock_room") and not colocated(world.player, npc)):
        end_interaction(world, "The participants are no longer available together.")
        return
    if interaction["stage"] != "approach":
        return
    interaction["elapsed"] += dt
    if interaction["elapsed"] > PLAYER_APPROACH_TIMEOUT:
        world.publish("interaction_failed", (world.player, npc), "The approach could not reach a meeting point.")
        end_interaction(world, "The approach timed out.")
        return
    mover = npc if interaction["incoming"] else world.player
    if move_toward(mover, interaction["point"], dt, MOVEMENT_SPEED * PLAYER_APPROACH_MULTIPLIER):
        face_each_other(world.player, npc)
        interaction["stage"] = "prompt" if interaction["incoming"] else "conversation"


def move_player(world, dx, dy, dt, *, running=False):
    if not can_claim(world) or world.paused:
        return
    from interactions import move_toward
    from locations import clamp_point
    distance = (dx * dx + dy * dy) ** 0.5
    if distance:
        p = world.player
        speed = MOVEMENT_SPEED * (PLAYER_RUN_MULTIPLIER if running else 1)
        point = clamp_point(p.location, (p.x + dx / distance * speed * dt,
                                        p.y + dy / distance * speed * dt))
        move_toward(p, point, dt, speed)


def record_claim(world, speaker, record, text, turn, listener_id):
    """Durable original reports, independently retrievable from trimmed chat context."""
    prior = next((c for c in world.player_claims if c["conversation_id"] == record.id and c["turn"] == turn), None)
    if prior:
        if listener_id not in prior["listener_ids"]:
            prior["listener_ids"].append(listener_id)
        return prior
    claim = dict(id=uuid4().hex, speaker_id=speaker.id, conversation_id=record.id, turn=turn,
                 day=world.day, phase=world.phase, text=text, kind="report", listener_ids=[listener_id])
    world.player_claims.append(claim)
    return claim


def relevant_claims(world, listener_id, message=""):
    words = set(message.casefold().split())
    claims = [c for c in world.player_claims if listener_id in c["listener_ids"]]
    ordered = sorted(enumerate(claims), key=lambda row: (len(words & set(row[1]["text"].casefold().split())), row[0]), reverse=True)
    return [c.copy() for _, c in ordered[:PLAYER_EVIDENCE_LIMIT]]


def owned_works(world, owner_id):
    return [work for work in world.creations if work.owner_id == owner_id]


def work_choices(world, owner_id, query=""):
    works = owned_works(world, owner_id)
    if query:
        words = set(query.casefold().split())
        works.sort(key=lambda w: (w.id.casefold() in query.casefold(), len(words & set(w.title.casefold().split()))), reverse=True)
    return [{"id": w.id, "title": w.title, "kind": w.kind, "creator_id": w.creator_id}
            for w in works[:PLAYER_CONTEXT_WORK_LIMIT]]


def requested_work_id(world, owner_id, message):
    """Resolve concrete references over the whole collection before bounding model input."""
    works = owned_works(world, owner_id)
    text = message.casefold()
    matches = [w for w in works if re.search(r"(?<![\w-])" + re.escape(w.id.casefold()) + r"(?![\w-])", text)]
    if not matches:
        matches = [w for w in works if re.search(r"(?<!\w)" + re.escape(w.title.casefold()) + r"(?!\w)", text)]
    if not matches:
        kind = "art" if re.search(r"\b(?:art|painting|picture)\b", text) else "music" if re.search(r"\b(?:music|song|composition)\b", text) else None
        candidates = [w for w in works if kind is None or w.kind == kind]
        ignored = {"a", "an", "the", "my", "your", "give", "to", "me", "please", "painting", "picture", "art", "music", "song", "composition"}
        words = set(re.findall(r"\w+", text)) - ignored
        scores = [(len(words & set(re.findall(r"\w+", w.title.casefold()))), w) for w in candidates]
        best = max((score for score, _ in scores), default=0)
        matches = [w for score, w in scores if score == best] if best else candidates
    return matches[0].id if len(matches) == 1 else ""


def update_player_opinion(npc, delta, reason):
    """Keep mixed evidence, and scale a bounded change by this resident's personality."""
    entry = npc.player_opinion
    if reason in entry["reason"]:
        return
    step = round(max(-PLAYER_OPINION_STEP, min(PLAYER_OPINION_STEP, delta))
                 * (0.5 + npc.preferences["sociability"] / 2))
    entry["opinion"] = max(-100, min(100, entry["opinion"] + step))
    combined = (entry["reason"] + " " + reason).strip()
    entry["reason"] = combined if len(combined) <= 400 else entry["reason"][:180] + " ... " + reason[:215]


def player_reply_closes(closing, reply, rewritten=()):
    """A question in either presentation leaves room for a player answer."""
    return closing and not any(contains_question(text) for text in (reply, *rewritten))


def accept_player_evidence(world, npc, record, message, response, *, opening=False):
    if opening:
        return
    index = len(record.turns) - 1
    if index in record.accepted_player_turns:
        return
    record.accepted_player_turns.append(index)
    record_claim(world, world.player, record, message, index, npc.id)
    # Store the actual report as a report, never as a witnessed fact or invented summary.
    event = world.publish("player_conversation", (world.player, npc), "Accepted a player conversation turn.", record)
    remember(npc, world.player, event, "conversation", f"The player told me: {message}", 0)
    change = response.player_opinion
    if change is not None:
        # The model must point to exact evidence in the accepted message, and greetings yield no reward.
        repeated = any(r.turns[i].text.strip().casefold() == message.strip().casefold()
                       for r in world.history if npc.id in r.participant_ids
                       for i in r.accepted_player_turns if (r is not record or i != index) and i < len(r.turns))
        if not repeated and change.evidence and change.evidence.casefold() in message.casefold() and len(message.split()) > 3:
            update_player_opinion(npc, change.delta, f'Player said "{change.evidence}". {change.reason}')


def accept_player_promises(world, npc, record, response, message, *, opening=False):
    from social import Promise, phase_label
    from dialogue_rules import validate_apple_promise
    for pending in record.player_promise_offers:
        if pending["status"] != "offered":
            continue
        if pending["deadline"] < world.phase_index:
            pending["status"] = "expired"
            continue
        if not message:
            continue
        from dialogue_rules import unquoted_speech
        if re.search(r"\b(?:no thanks|no thank you|decline|don't want|do not want)\b", unquoted_speech(message), re.I):
            pending["status"] = "declined"
            continue
        try:
            validate_apple_promise(record.turns[pending["offer_turn"]].text, message,
                                   pending["deadline"], current_phase=world.phase_index)
        except ValueError:
            continue
        pending["accept_turn"] = len(record.turns) - 1
        pending["status"] = "accepted"
    offer = response.apple_promise
    if offer:
        player_giving = offer.giver_id == world.player.id
        record.player_promise_offers.append(dict(id=uuid4().hex, giver_id=offer.giver_id,
            recipient_id=npc.id if player_giving else world.player.id, deadline=offer.deadline,
            offer_turn=len(record.turns) - 1 if player_giving else len(record.turns),
            accept_turn=len(record.turns) if player_giving else None, created_phase=world.phase_index,
            status="accepted" if player_giving else "offered"))
    for pending in record.player_promise_offers:
        if pending["accept_turn"] is None or any(p.id == pending["id"] for p in world.promises):
            continue
        promise = Promise(pending["id"], pending["giver_id"], pending["recipient_id"], record.id,
                          world.phase_index, pending["deadline"])
        world.promises.append(promise)
        text = f'One apple promised by {phase_label(promise.deadline)}; delivery is still pending.'
        event = world.publish("promise_created", (world.player, npc), text, record)
        remember(npc, world.player, event, "promise_created", text, 1)
        record.outcomes.append(text)


def call_reason(world, npc):
    """Select only concrete records that this resident actually owns or knows."""
    promises = [p for p in world.promises if p.status == "pending"
                and {p.promisor_id, p.recipient_id} == {npc.id, world.player.id}]
    if promises:
        p = promises[0]
        return dict(key=f"promise:{p.id}", kind="followup", reference=p.id,
                    reason="Follow up on our accepted apple commitment.")
    if npc.preferences["generosity"] > PLAYER_CALL_GIFT_GENEROSITY and npc.player_opinion["opinion"] >= 0:
        works = owned_works(world, npc.id)
        if works:
            work = works[-1]
            return dict(key=f"gift:{work.id}", kind="gift", reference=work.id,
                        reason=f'Consider offering my {work.kind} "{work.title}" to the player. I may also keep it.')
        if npc.inventory["fruit"] > 1:
            return dict(key=f"apples:{world.day}:{world.phase}", kind="gift", reference="fruit",
                        reason="Consider offering one of my actual apples to the player.")
    concern = npc.current_concern
    if concern and concern.claim_id and concern.kind == "news":
        return dict(key=f"news:{concern.claim_id}", kind="news", reference=concern.claim_id,
                    reason="Share the known news account in my current concern, retaining its source and uncertainty.")
    memories = [m for m in npc.memories if m.partner_id == world.player.id and m.kind == "conversation"]
    if memories and npc.preferences["sociability"] > PLAYER_CALL_SOCIABILITY:
        memory = memories[-1]
        return dict(key=f"memory:{memory.id}", kind="followup", reference=memory.id,
                    reason=f"Follow up on our accepted earlier encounter: {memory.summary}")
    return None


def call_valid(world, npc, call):
    from locations import available, colocated
    if not active_player(world) or not available(npc) or not colocated(world.player, npc):
        return False
    ref = call["reference"]
    if call["kind"] == "introduction":
        from player_story import valid_introduction_call
        return valid_introduction_call(world, npc, call)
    if call["kind"] == "reply":
        return any(r.id == ref and {world.player.id, npc.id} == set(r.participant_ids)
                   and r.turns and r.turns[-1].speaker == npc.name and not r.error for r in world.history)
    if call["kind"] == "gift":
        return (npc.inventory["fruit"] > 0 if ref == "fruit"
                else any(w.id == ref and w.owner_id == npc.id for w in world.creations))
    if call["kind"] == "news":
        return any(b.claim_id == ref for b in npc.beliefs)
    return (any(m.id == ref for m in npc.memories)
            or any(p.id == ref and p.status == "pending" for p in world.promises))


def clear_call(npc):
    if npc.player_call:
        if not npc.unread_player_reply:
            npc.resolved_player_calls.append(npc.player_call["key"])
        npc.player_call = None
    npc.player_call_cooldown = PLAYER_CALL_COOLDOWN


def start_call(world, npc, reason):
    """Claim an incoming meeting before normal activities can reserve its actors."""
    start_approach(world, npc, incoming=True)
    npc.player_call = reason.copy()
    event = world.publish("player_call", (npc,), f"{npc.name} would like to talk to you.")
    npc.player_call["event_id"] = event.id


def update_calls(world, dt):
    for npc in world.npcs:
        npc.player_call_cooldown = max(0.0, npc.player_call_cooldown - dt)
        if npc.player_call and not call_valid(world, npc, npc.player_call):
            clear_call(npc)
            if owns_resident(world, npc):
                end_interaction(world, "The reason for this call is no longer available.")
    world.player_call_check -= dt
    if world.player_call_check > 0 or world.phase_pending or not can_claim(world):
        return
    world.player_call_check = PLAYER_CALL_CHECK_SECONDS
    from locations import colocated
    for npc in sorted(world.npcs, key=lambda n: n.player_opinion["opinion"] + n.preferences["sociability"], reverse=True):
        if npc.player_call or npc.player_call_cooldown or not world.company_available(npc) or not colocated(world.player, npc):
            continue
        reason = call_reason(world, npc)
        if reason and reason["key"] not in npc.resolved_player_calls:
            try:
                start_call(world, npc, reason)
            except ValueError:
                continue
            break
