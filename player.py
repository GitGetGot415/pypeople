"""Physical player and exclusive social ownership; possessions remain engine owned."""

from dataclasses import dataclass, field
import re
from uuid import uuid4

from constants import (INVENTORY_CAPACITY, PLAYER_ID, STARTING_PLAYER_FRUIT, PLAYER_CALL_LIMIT,
                       PLAYER_CALL_COOLDOWN, PLAYER_CALL_CHECK_SECONDS, PLAYER_CONTEXT_WORK_LIMIT,
                       PLAYER_OPINION_STEP, GIFT_GIVE_SECONDS, GIFT_SECONDS,
                       PLAYER_CALL_GIFT_GENEROSITY, PLAYER_CALL_SOCIABILITY)
from conversation_history import ConversationRecord, remember
from dialogue_text import contains_question
from constants import (GARDEN_LOCATION, STARTING_PLAYER_POSITION, STARTING_FACING,
                       PLAYER_DEFAULT_APPEARANCE, MOVEMENT_SPEED, PLAYER_APPROACH_MULTIPLIER,
                       PLAYER_APPROACH_TIMEOUT, PLAYER_TALK_DISTANCE, PLAYER_EVIDENCE_LIMIT)


@dataclass
class Player:
    sprite_size = None
    id: str = PLAYER_ID
    inventory: dict = field(default_factory=lambda: {"fruit": STARTING_PLAYER_FRUIT})
    inventory_capacity: int = INVENTORY_CAPACITY
    x: float = STARTING_PLAYER_POSITION[0]
    y: float = STARTING_PLAYER_POSITION[1]
    facing: int = STARTING_FACING
    location: str = GARDEN_LOCATION
    alive: bool = True
    appearance: dict = field(default_factory=lambda: PLAYER_DEFAULT_APPEARANCE.copy())
    journey: dict | None = None

    @property
    def name(self):
        return "the player"

    @property
    def inventory_space(self):
        return self.inventory_capacity - self.inventory["fruit"]

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


def move_player(world, dx, dy, dt):
    if not can_claim(world) or world.paused:
        return
    from interactions import move_toward
    from locations import clamp_point
    distance = (dx * dx + dy * dy) ** 0.5
    if distance:
        p = world.player
        point = clamp_point(p.location, (p.x + dx / distance * MOVEMENT_SPEED * dt,
                                        p.y + dy / distance * MOVEMENT_SPEED * dt))
        move_toward(p, point, dt, MOVEMENT_SPEED)


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


def validate_transfer(world, giver, recipient, item, quantity, *, interaction=None, player_authorization=None):
    """Always enforce authority, including with speech diagnostics disabled."""
    from locations import available, colocated
    if giver is None or recipient is None or giver is recipient:
        raise ValueError("Choose a giver and a different recipient.")
    if not colocated(giver, recipient):
        raise ValueError("Gifts require both people in the same location.")
    if world.player in (giver, recipient):
        resident = recipient if giver is world.player else giver
        if not active_player(world) or not owns_resident(world, resident):
            raise ValueError("Meet the resident before exchanging a gift.")
        import math
        if math.dist((giver.x, giver.y), (recipient.x, recipient.y)) > PLAYER_TALK_DISTANCE and not world.player_interaction.get("knock_room"):
            raise ValueError("Move within talking distance before giving.")
        if world.player_interaction.get("knock_room"):
            raise ValueError("Enter the room before exchanging possessions.")
    for person in (giver, recipient):
        if person is world.player:
            continue
        authorized = bool(player_authorization and world.player_interaction
            and player_authorization == world.player_interaction["id"] and owns_resident(world, giver))
        if world.player not in (giver, recipient) and owns_resident(world, person) and not authorized:
            raise ValueError("The resident is meeting the player.")
        if world.player not in (giver, recipient) and world.visit and world.visit["host_id"] == person.id:
            raise ValueError("The resident is hosting the player.")
        if (interaction is None and giver is not world.player and recipient is not world.player
                and (person.player_call or person is recipient and person.selected)):
            raise ValueError("The resident is waiting to talk to the player.")
        if world.npc_by_id(person.id) is not person or not available(person):
            raise ValueError("The resident is no longer available.")
        if person.interaction is not None and person.interaction is not interaction:
            raise ValueError("The resident is occupied in another interaction.")
    if type(quantity) is not int or quantity < 1:
        raise ValueError("Select a positive apple quantity or one work.")
    if item == "fruit":
        if giver.inventory["fruit"] < quantity:
            raise ValueError("The giver no longer has enough apples.")
        if recipient is world.player and recipient.inventory_space < quantity:
            raise ValueError("The recipient's apple storage is full.")
    else:
        work = next((w for w in world.creations if w.id == item), None)
        if quantity != 1 or work is None or work.owner_id != giver.id:
            raise ValueError("This work is no longer owned by the giver. A fresh offer is needed.")
    # Existing accepted handoffs reserve stock, while unaccepted offers reserve nothing.
    pending = {id(n.gift): n.gift for n in world.npcs if n.gift and n.gift is not interaction}
    reserved = sum(g.quantity for g in pending.values()
                   if not g.transferred and g.initiator is giver and g.item == item)
    if (item == "fruit" and giver.inventory["fruit"] - reserved < quantity
            or item != "fruit" and reserved):
        raise ValueError("That possession is already reserved for an accepted handoff.")


def commit_transfer(world, interaction):
    if interaction.transferred:
        return
    giver, recipient = interaction.initiator, interaction.recipient
    item, quantity, record = interaction.item, interaction.quantity, interaction.record
    validate_transfer(world, giver, recipient, item, quantity, interaction=interaction)
    work = next((w for w in world.creations if w.id == item), None)
    if item == "fruit":
        giver.inventory["fruit"] -= quantity
        recipient.inventory["fruit"] += quantity
        description = f"{quantity} apple" + ("s" if quantity != 1 else "")
    else:
        work.owner_id = recipient.id
        description = f'{work.kind}: {work.title} [{work.id}]'
    interaction.transferred = True
    transfer = dict(id=uuid4().hex, giver_id=giver.id, recipient_id=recipient.id,
                    item=item, quantity=quantity, day=world.day, phase=world.phase,
                    sequence=sum(len(r.transfers) for r in world.history))
    record.transfers.append(transfer)
    event = world.publish("fruit_shared" if item == "fruit" else "work_shared", (giver, recipient),
                          f"{giver.name} gave {description} to {recipient.name}.", record)
    interaction.outcome_event_id = event.id
    for actor, partner, kind, summary, weight in (
        (giver, recipient, "gave_fruit" if item == "fruit" else "gave_work", f"I gave {description} to {recipient.name}.", 2),
        (recipient, giver, "received_fruit" if item == "fruit" else "received_work", f"{giver.name} gave me {description}.", 6)):
        if actor is not world.player:
            remember(actor, partner, event, kind, summary, weight)
            if partner is world.player:
                update_player_opinion(actor, weight, summary)
    if item == "fruit":
        from social import on_transfer
        on_transfer(world, giver, recipient, quantity, event, record)
    record.outcomes[0:1] = [f"Gave {description} to {recipient.name}."]
    if giver is world.player or recipient is world.player:
        resident = recipient if giver is world.player else giver
        world.player_acknowledgments.append(dict(npc_id=resident.id, record_id=record.id, transfer_id=transfer["id"]))


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


def supplied_player_name(message, *, requested=False):
    from dialogue_rules import unquoted_speech
    speech = unquoted_speech(message).strip()
    match = re.match(r"(?:my name is|call me|you can call me|i go by|i am|i['’]m)\s+([\w'-]+(?: [\w'-]+){0,3})(?=[,.!?;:]|$)", speech, re.I)
    if match:
        candidate = match[1]
        if not re.match(r"i(?: am|['’]m)\s", speech, re.I) or candidate[0].isupper():
            return candidate
    if requested and re.fullmatch(r"[^\W\d_][\w'-]*(?: [A-Z][\w'-]*){0,3}[.!]?", speech):
        if speech.rstrip(".!").casefold() in {"no", "yes", "thanks", "sorry", "maybe", "okay", "ok", "why", "nevermind"}:
            return None
        return speech.rstrip(".!")
    return None


def accept_player_evidence(world, npc, record, message, response, *, opening=False):
    if opening:
        return
    index = len(record.turns) - 1
    if index in record.accepted_player_turns:
        return
    record.accepted_player_turns.append(index)
    record_claim(world, world.player, record, message, index, npc.id)
    from name_knowledge import asks_name
    prior = record.turns[index - 1][1] if index else ""
    supplied = supplied_player_name(message, requested=asks_name(prior))
    if supplied:
        npc.known_names[world.player.id] = dict(name=supplied, conversation_id=record.id, turn=index)
        record.name_introductions[index] = [dict(actor_id=world.player.id, name=supplied, reference="self")]
    # Store the actual report as a report, never as a witnessed fact or invented summary.
    event = world.publish("player_conversation", (world.player, npc), "Accepted a player conversation turn.", record)
    remember(npc, world.player, event, "conversation", f"The player told me: {message}", 0)
    change = response.player_opinion
    if change is not None:
        # The model must point to exact evidence in the accepted message, and greetings yield no reward.
        repeated = any(r.turns[i][1].strip().casefold() == message.strip().casefold()
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
            validate_apple_promise(record.turns[pending["offer_turn"]][1], message,
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


class PlayerHandoff:
    """A timed, co-located handoff owned by the physical player interaction."""
    def __init__(self, world, giver, recipient, item, quantity, record):
        validate_transfer(world, giver, recipient, item, quantity)
        self.world, self.initiator, self.recipient = world, giver, recipient
        self.resident = recipient if giver is world.player else giver
        self.item, self.quantity, self.record = item, quantity, record
        self.record.transfer_intent = dict(giver_id=giver.id, recipient_id=recipient.id, item=item, quantity=quantity)
        self.active, self.arrived, self.transferred = True, True, False
        self.elapsed, self.outcome_event_id = 0.0, None
        self.saved_state = self.resident.suspend_activity("Exchanging a gift with the player")
        self.resident.gift = self
        self.record.status = "Giving items"

    def update(self, dt):
        if not self.active:
            return
        from locations import colocated
        if not active_player(self.world) or not colocated(self.world.player, self.resident):
            self.finish("The resident is no longer available.")
            return
        self.elapsed += max(0.0, dt)
        if not self.transferred and self.elapsed + 1e-9 >= GIFT_GIVE_SECONDS:
            try:
                commit_transfer(self.world, self)
            except ValueError as exc:
                self.finish(str(exc))
                return
        if self.elapsed + 1e-9 >= GIFT_SECONDS:
            self.finish()

    def finish(self, error=""):
        if not self.active:
            return
        self.active = False
        self.record.status, self.record.error = ("Failed", error) if error and not self.transferred else ("Completed", "")
        if error and self.transferred:
            self.record.outcomes.append(f"The transfer remains committed. {error}")
        if self.resident.gift is self:
            self.resident.gift = None
            if self.world.npc_by_id(self.resident.id) is self.resident:
                self.resident.restore_after_interaction(self.saved_state, self.world.player.id, preserve_facing=True)
        if owns_resident(self.world, self.resident) and self.world.player_interaction["stage"] == "handoff":
            end_interaction(self.world, error or "The accepted handoff finished.")


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
        if npc.inventory["fruit"] > 1 and world.player.inventory_space:
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
    if call["kind"] == "reply":
        return any(r.id == ref and {world.player.id, npc.id} == set(r.participant_ids)
                   and r.turns and r.turns[-1][0] == npc.name and not r.error for r in world.history)
    if call["kind"] == "gift":
        return (npc.inventory["fruit"] > 0 and world.player.inventory_space > 0 if ref == "fruit"
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
                start_approach(world, npc, incoming=True)
            except ValueError:
                continue
            npc.player_call = reason.copy()
            event = world.publish("player_call", (npc,), f"{npc.name} would like to talk to you.")
            npc.player_call["event_id"] = event.id
            break
