"""Accepted possession transfers and their shared approach/handoff lifecycle."""

import re
from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from player_records import PlayerAcknowledgment
from transfer_records import TransferIntent, TransferReceipt
from dialogue import ConversationPart
from uuid import uuid4
from constants import (GIFT_GIVE_SECONDS, GIFT_SECONDS, PLAYER_TALK_DISTANCE, PLAYER_LABEL,
                       CONVERSATION_MAX_AGREEMENTS, CONVERSATION_COMPLETED, CONVERSATION_FAILED,
                       CONVERSATION_GIVING_ITEMS, CONVERSATION_THANKING, CONVERSATION_WALKING_TO_GIVE)
from conversation_history import ConversationRecord, remember
from interactions import company_point, face_each_other, release_resident
from dialogue_rules import unquoted_speech
from actor_identity import actor_id


def validate_transfer(world, giver, recipient, item, quantity, *, interaction=None, player_authorization=None):
    """Always enforce authority, including with speech diagnostics disabled."""
    from locations import available, colocated
    from player import active_player, owns_resident
    from autonomy import arrival_greeting_reserved
    if giver is None or recipient is None or giver is recipient:
        raise ValueError("Choose a giver and a different recipient.")
    if not colocated(giver, recipient):
        raise ValueError("Gifts require both people in the same location.")
    if world.player in (giver, recipient):
        resident = recipient if giver is world.player else giver
        if not active_player(world) or not owns_resident(world, resident):
            raise ValueError("Meet the resident before exchanging a gift.")
        import math
        if math.dist((giver.x, giver.y), (recipient.x, recipient.y)) > PLAYER_TALK_DISTANCE and not world.player_interaction.knock_room:
            raise ValueError("Move within talking distance before giving.")
        if world.player_interaction.knock_room:
            raise ValueError("Enter the room before exchanging possessions.")
    for person in (giver, recipient):
        if person is world.player:
            continue
        if arrival_greeting_reserved(person, world):
            raise ValueError("Judy's welcome must finish before exchanging gifts.")
        authorized = bool(player_authorization and world.player_interaction
            and player_authorization == world.player_interaction.id and owns_resident(world, giver))
        if world.player not in (giver, recipient) and owns_resident(world, person) and not authorized:
            raise ValueError("The resident is meeting the player.")
        if world.player not in (giver, recipient) and world.visit and world.visit.host_id == person.id:
            raise ValueError("The resident is hosting the player.")
        if (interaction is None and giver is not world.player and recipient is not world.player
                and (person.player_call or person is recipient and person.selected)):
            raise ValueError("The resident is waiting to talk to the player.")
        held_for_player = (world.player in (giver, recipient) and owns_resident(world, person)
                           and person.departure is None)
        if world.npc_by_id(person.id) is not person or not (available(person) or held_for_player):
            raise ValueError("The resident is no longer available.")
        if person.interaction is not None and person.interaction is not interaction:
            raise ValueError("The resident is occupied in another interaction.")
    if type(quantity) is not int or quantity < 1:
        raise ValueError("Select a positive apple quantity or one work.")
    if item == "fruit":
        if giver.inventory["fruit"] < quantity:
            raise ValueError("The giver no longer has enough apples.")
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
    from player import update_player_opinion
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
    transfer = TransferReceipt.committed(id=uuid4().hex, giver_id=giver.id, recipient_id=recipient.id,
                    item=item, quantity=quantity, day=world.day, phase=world.phase,
                    sequence=sum(len(r.transfers) for r in world.history))
    record.transfers.append(transfer)
    event = world.publish("fruit_shared" if item == "fruit" else "work_shared", (giver, recipient),
                          f"{giver.name} gave {description} to {recipient.name}.", record)
    from locations import observers
    from information import add_fact
    add_fact(world, "apple_given" if item == "fruit" else "work_given", giver, recipient,
             event.id, record, event.text, observers(world, giver, recipient))
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
        world.player_acknowledgments.append(PlayerAcknowledgment(npc_id=resident.id, record_id=record.id, transfer_id=transfer.id))


@dataclass(eq=False)
class HandoffProgress:
    """Durable domain progress; live actors and suspended activities attach separately."""
    item: Annotated[str, Field(min_length=1)]
    quantity: Annotated[int, Field(strict=True, ge=1)]
    arrived: bool
    elapsed: Annotated[float, Field(ge=0, lt=GIFT_SECONDS)]
    transferred: bool
    outcome_event_id: str | None
    thanks: Annotated[list[ConversationPart], Field(max_length=3)]
    rewritten_thanks: list[str]


class Handoff(HandoffProgress):
    """Reserve both participants through approach, giving, and thanking."""

    def __init__(self, initiator, recipient, item, quantity, record=None, spacing=0, *, saved_states=None, already_met=False, thanks=None, rewritten_thanks=(), _restored=None):
        if _restored is not None:
            self._restore(initiator, recipient, record, _restored)
            return
        self.world = initiator.world
        player_exchange = self.world.player in (initiator, recipient)
        if player_exchange:
            validate_transfer(self.world, initiator, recipient, item, quantity)
            already_met = True
        destination = (initiator.x, initiator.y) if already_met else company_point(initiator, recipient, spacing)
        if destination is None:
            raise ValueError("There isn't enough room here to give the items.")
        self.initiator, self.recipient = initiator, recipient
        self.item, self.quantity = item, quantity
        self.arrived = already_met
        self.active = True
        self.transferred = False
        self.outcome_event_id = None
        self.thanks = list(thanks) if thanks is not None else []
        self.rewritten_thanks = list(rewritten_thanks)
        self.elapsed = 0.0
        labels = tuple(PLAYER_LABEL if actor is self.world.player else actor.name for actor in (initiator, recipient))
        self.record = record or ConversationRecord(labels, "Give items", CONVERSATION_WALKING_TO_GIVE,
                                                  participant_ids=[initiator.id, recipient.id])
        if set(self.record.participant_ids) != {initiator.id, recipient.id}:
            self.record = ConversationRecord(labels, "Directed gift", CONVERSATION_WALKING_TO_GIVE,
                authorization_id=record.id, participant_ids=[initiator.id, recipient.id])
            initiator.world.history.append(self.record)
        if player_exchange:
            self.world.player_interaction.record_id = self.record.id
        self.record.transfer_intent = TransferIntent(giver_id=initiator.id, recipient_id=recipient.id, item=item, quantity=quantity)
        self.saved_states = {}
        self.record.status = CONVERSATION_GIVING_ITEMS if already_met else CONVERSATION_WALKING_TO_GIVE
        for npc in self.residents:
            self.saved_states[npc.id] = npc.suspend_activity(
                "Exchanging a gift with the player" if player_exchange else
                "Giving items" if npc is initiator else "Receiving items")
            self.world.reservations.acquire("handoff", self, npc)
        if saved_states is not None:
            self.saved_states = dict(zip((npc.id for npc in self.residents), saved_states))
        if initiator is not self.world.player:
            initiator.destination = None if already_met else destination

    @classmethod
    def restore(cls, initiator, recipient, record, *, progress, saved_states):
        return cls(initiator, recipient, progress.item, progress.quantity, record,
                   _restored=dict(progress=progress, saved_states=saved_states))

    def _restore(self, initiator, recipient, record, restored):
        self.__dict__.update(vars(restored["progress"]))
        self.initiator, self.recipient, self.record = initiator, recipient, record
        self.world, self.active, self.saved_states = initiator.world, True, restored["saved_states"]
        for npc in self.residents:
            self.world.reservations.acquire("handoff", self, npc)

    @property
    def residents(self):
        return tuple(actor for actor in (self.initiator, self.recipient) if actor is not self.world.player)

    def resident_activities(self):
        return ((npc, self.saved_states[npc.id]) for npc in self.residents)

    def activity_snapshots(self):
        return iter(self.saved_states.values())

    @property
    def player_exchange(self):
        return self.world.player in (self.initiator, self.recipient)

    def update(self, dt):
        if not self.active:
            return
        giver, recipient = self.initiator, self.recipient
        from player import active_player
        if (giver.location != recipient.location or any(self.world.npc_by_id(p.id) is not p for p in self.residents)
                or self.player_exchange and not active_player(self.world)):
            self.finish("The resident is no longer available." if self.player_exchange else "A participant is no longer outdoors.")
            return
        if not self.arrived:
            self.arrived = giver._move(max(0.0, dt))
            if self.arrived:
                giver.destination = None
                self.record.status = CONVERSATION_GIVING_ITEMS
                face_each_other(giver, recipient)
            return
        if self.thanks:
            from interactions import observe_conversation
            observe_conversation(self.world, self.record, giver, recipient)
        elapsed = self.elapsed + max(0.0, dt)
        if not self.transferred and elapsed + 1e-9 >= GIFT_GIVE_SECONDS:
            try:
                commit_transfer(giver.world, self)
            except ValueError as exc:
                self.finish(str(exc))
                return
            if self.thanks:
                self.record.add_turn(recipient.name, " ".join(part.text.strip() for part in self.thanks),
                                     [part.model_dump() for part in self.thanks], self.rewritten_thanks, speaker_id=recipient.id)
                observe_conversation(self.world, self.record, giver, recipient, accepted=True)
            if not self.player_exchange:
                self.record.outcomes[0:1] = [f"Gave {self.quantity} {self.item} to {recipient.name}."]
                self.record.status = CONVERSATION_THANKING
            elapsed = max(GIFT_GIVE_SECONDS, elapsed)
        self.elapsed = elapsed
        if self.elapsed + 1e-9 >= GIFT_SECONDS:
            self.finish()

    def finish(self, error=""):
        if not self.active:
            return
        self.record.pending_witness_ids.clear()
        self.active = False
        self.record.status = CONVERSATION_FAILED if error and not self.transferred else CONVERSATION_COMPLETED
        self.record.error = error if not self.transferred else ""
        if error and self.transferred:
            self.record.outcomes.append(f"The transfer remains committed. {error}")
        elif error and not self.player_exchange:
            self.record.outcomes[0:1] = [error]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient), error, self.record)
            self.outcome_event_id = event.id
        for npc, state in self.resident_activities():
            other = self.recipient if npc is self.initiator else self.initiator
            if not release_resident(self.world, "handoff", self, npc, state, other.id, preserve_facing=True):
                continue
            if self.record.purpose == "food_request":
                npc.pause_after_conversation(True)
        if self.player_exchange:
            from player import owns_resident, end_interaction
            resident = self.residents[0]
            if owns_resident(self.world, resident) and self.world.player_interaction.stage == "handoff":
                end_interaction(self.world, error or "The accepted handoff finished.")


def gift_possession_error(item, quantity, text, owner):
    """Equivalent possession and spoken identification checks for generated gifts."""
    if item == "fruit":
        if quantity > owner.get("inventory", {}).get("fruit", 0):
            return "stock"
        if not re.search(r"\b(?:apple|apples|fruit)\b", unquoted_speech(text).casefold()):
            return "apples"
    else:
        work = next((w for w in owner.get("owned_works", []) if w["id"] == item), None)
        if work is None or quantity != 1 or work["title"].casefold() not in text.casefold():
            return "work"
    return None


def validate_resident_gift(gift, turns, snapshot):
    if gift is None:
        return
    if snapshot.get("purpose") == "food_request" or snapshot.get("arrival_focus"):
        raise ValueError("This exchange uses its existing food or orientation rules, without a separate gift offer.")
    from dialogue_rules import unquoted_speech
    roles = {actor_id(snapshot[role]): role for role in ("initiator", "recipient")}
    if gift.giver_id not in roles or gift.recipient_id not in roles or gift.giver_id == gift.recipient_id:
        raise ValueError("A gift must stay between the actual conversation participants.")
    giver = snapshot[roles[gift.giver_id]]
    if gift.offer_turn >= len(turns) or turns[gift.offer_turn].speaker != roles[gift.giver_id]:
        raise ValueError("A gift offer needs its giver's actual speech.")
    validate_gift_offer_speech(turns[gift.offer_turn].text, gift.item, gift.quantity,
                               giver.get("inventory", {}).get("fruit", 0))
    possession_error = gift_possession_error(gift.item, gift.quantity, turns[gift.offer_turn].text, giver)
    if possession_error in ("stock", "apples"):
        raise ValueError("The giver must own and offer the actual apples.")
    if possession_error == "work":
        raise ValueError("A creative offer must identify one owned work and its spoken title.")
    if gift.accept_turn is not None:
        index = gift.accept_turn
        if index <= gift.offer_turn or index >= len(turns) or turns[index].speaker != roles[gift.recipient_id]:
            raise ValueError("The recipient must accept on their own later turn.")
        acceptance = unquoted_speech(turns[index].text).casefold()
        if not gift_acceptance_supported(acceptance):
            raise ValueError("A gift transfer needs explicit generated acceptance.")


def gift_acceptance_supported(text):
    from dialogue_rules import unquoted_speech
    speech = unquoted_speech(text).replace("\u2019", "'").casefold()
    return bool(re.search(r"\b(?:i accept|i(?:'ll| will) (?:gladly |happily )?(?:take|accept)|"
                         r"i(?: am|'m) (?:happy|glad|willing) to (?:accept|take)|yes|thank you|thanks)\b", speech)
        and not re.search(r"\b(?:no[,!;:]?\s+(?:thanks|thank you)|decline|"
                          r"(?:cannot|can't|won't|will not|don't|do not) (?:accept|take|want)|"
                          r"not (?:ready|able|in a position) to (?:accept|take|receive)|"
                          r"if|unless|maybe|perhaps|not sure)\b", speech))


def validate_gift_offer_speech(text, item, quantity, stock):
    from dialogue_rules import unquoted_speech
    speech = unquoted_speech(text).casefold()
    if (not re.search(r"\b(?:offer|would you like|want|give you|for you|have this|take this)\b", speech)
            or re.search(r"\b(?:not|never|won't|cannot|can't|don't|do not)\b[^.!?]{0,35}\b(?:offer|give|have|take|for you)\b", speech)):
        from validation_errors import DialogueValidationError, RecoveryKind
        raise DialogueValidationError("A gift offer needs supporting offer speech.", RecoveryKind.MISSING_GIFT_OFFER)
    if item == "fruit":
        counts = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        spoken = re.findall(r"\b(\d+|one|two|three|four|five)\s+(?:of my\s+)?(?:apples?|fruit)\b", speech)
        values = [int(word) if word.isdigit() else counts[word] for word in spoken]
        if values and any(value != quantity for value in values):
            raise ValueError("The spoken apple quantity differs from the offered quantity.")
        if quantity > 1 and not values and not (quantity == stock and re.search(r"\ball (?:of )?my (?:apples|fruit)\b", speech)):
            raise ValueError("Identify the offered apple quantity in speech.")


def validate_player_transfer(result, snapshot):
    """Effect authorization is independent of optional dialogue coherence checks."""
    if snapshot.get("request_kind") == "player_command" and (result.invitation_acceptance
            or result.invitation_resolution == "declined" or result.apple_promise
            and result.apple_promise.giver_id == snapshot.get("player_id")):
        raise ValueError("An action selection cannot accept a separate social offer or speak for the player.")
    if result.invitation:
        if (result.action != "SAY" or result.gift_offer or result.gift_acceptance or result.apple_promise
                or result.invitation.evidence not in result.reply
                or result.invitation.destination not in {c["id"] for c in snapshot.get("invitation_choices", [])}):
            raise ValueError("An invitation needs original speech and a supplied physical destination.")
    if result.invitation_acceptance:
        offer = snapshot.get("pending_invitation")
        if (result.action != "SAY" or not offer or offer["id"] != result.invitation_acceptance
                or not snapshot.get("_player_message")):
            raise ValueError("Travel acceptance must identify the actual preceding invitation.")
    if result.invitation_resolution:
        source = snapshot.get("_player_message", "") if result.invitation_resolution == "declined" else result.reply
        if (not snapshot.get("pending_invitation") or result.invitation or result.invitation_acceptance
                or not result.invitation_resolution_evidence or result.invitation_resolution_evidence not in source):
            raise ValueError("Invitation resolution requires the actual offer and original refusal or withdrawal.")
    if snapshot.get("request_kind") in ("player_opening", "player_gift", "player_acknowledgment") and result.action != "SAY":
        raise ValueError("An engine-initiated player turn cannot authorize a player command.")
    from dialogue_rules import unquoted_speech
    speech = unquoted_speech(result.reply).casefold()
    incoming = snapshot.get("player_interaction", {}).get("incoming_gift")
    promise = result.apple_promise
    if promise:
        if snapshot.get("pending_apple_promise") or snapshot.get("player_apple_offer_count", 0) >= CONVERSATION_MAX_AGREEMENTS:
            raise ValueError("Keep the existing apple offer unchanged; this conversation has no new promise slot.")
        from dialogue_rules import validate_apple_offer, validate_apple_promise
        if result.action != "SAY" or incoming or result.gift_offer:
            raise ValueError("A future apple promise is separate from a current gift.")
        if promise.deadline not in [p["deadline"] for p in snapshot.get("apple_promise_choices", [])]:
            raise ValueError("Use an available apple deadline.")
        now = snapshot.get("phase_index", 0)
        if promise.giver_id == snapshot.get("actor_id"):
            validate_apple_offer(result.reply, promise.deadline, current_phase=now)
        elif promise.giver_id == snapshot.get("player_id"):
            validate_apple_promise(snapshot.get("_player_message", ""), result.reply, promise.deadline, current_phase=now)
        else:
            raise ValueError("An apple promise must belong to the current participants.")
    if result.gift_acceptance:
        if not incoming or not gift_acceptance_supported(result.reply):
            raise ValueError("Gift acceptance needs a current incoming gift and explicit accepted speech.")
    offer = result.gift_offer
    if offer:
        if (result.action != "SAY" or incoming or snapshot.get("request_kind") == "player_acknowledgment"
                or offer.recipient_id != snapshot.get("player_id")):
            raise ValueError("Only a SAY reply can offer one gift to the current player.")
        validate_gift_offer_speech(result.reply, offer.item, offer.quantity, snapshot.get("inventory", {}).get("fruit", 0))
        possession_error = gift_possession_error(offer.item, offer.quantity, result.reply, snapshot)
        if possession_error == "stock":
            raise ValueError("An offer cannot spend unavailable apples.")
        if possession_error == "apples":
            raise ValueError("Identify the offered apples in speech.")
        if possession_error == "work":
            raise ValueError("Identify one actual owned work by ID and its title in the offer.")
    if result.action == "GIVE_ITEM":
        if (snapshot.get("request_intent") or result._authorized_intent).get("action") != "GIVE_ITEM":
            raise ValueError("A directed gift requires explicit player-command authorization.")
    if result.action == "SAY":
        transfer_claim = re.search(r"\b(?:i (?:have |just |already )?(?:gave|given|transferred|handed)|you now own|it is now yours|it['’]s yours now)\b", speech)
        evidence = (snapshot.get("known_player_transfers") or snapshot.get("player_interaction", {}).get("committed_transfer")
                    or any(m.get("kind") in ("gave_fruit", "gave_work") for m in snapshot.get("memories", [])))
        if transfer_claim and not evidence:
            raise ValueError("SAY cannot claim an unperformed transfer.")
        if re.search(r"\bi(?: will|['’]ll| promise to)\b[^.!?]{0,35}\b(?:paint|compose|write|make)\b[^.!?]{0,50}\b(?:for you|you a|your commission)\b", speech):
            raise ValueError("A SAY reply cannot promise an unsupported creative commission.")


def validate_transfer_history(ctx):
    state = ctx.state
    history, creations, promises, acknowledgments = ctx.history, state.creations, state.promises, state.player_acknowledgments
    owner_ids, live_ids, player_id, now, creation_ids, ids = ctx.owner_ids, ctx.live_ids, state.player.id, ctx.now, ctx.creation_ids, ctx.ids
    from conversation_history import validate_history_record
    from social_records import phase_index
    from social_saves import validate_player_promise_offers
    transfer_ids, transfer_order, player_offer_ids = set(), [], set()
    for record in history.values():
        validate_player_promise_offers(record, promises, owner_ids, player_id, now, player_offer_ids)
        validate_history_record(record, history, ids, player_id)
        for transfer in record.transfers:
            if (transfer.id in transfer_ids
                    or transfer.giver_id not in owner_ids or transfer.recipient_id not in owner_ids
                    or transfer.giver_id == transfer.recipient_id or transfer.item != "fruit" and (
                        transfer.item not in creation_ids or transfer.quantity != 1)
                    or phase_index(transfer.day, transfer.phase) > now):
                raise ValueError("Invalid committed transfer provenance.")
            transfer_ids.add(transfer.id)
            transfer_order.append(transfer)
    if sorted(t.sequence for t in transfer_order) != list(range(len(transfer_order))):
        raise ValueError("Transfer history must retain its unique commit order.")
    owners = {w.id: w.creator_id for w in creations}
    for transfer in sorted(transfer_order, key=lambda t: t.sequence):
        if transfer.item != "fruit":
            if owners[transfer.item] != transfer.giver_id:
                raise ValueError("Transfer provenance does not follow the actual previous owner.")
            owners[transfer.item] = transfer.recipient_id
    if any(owners[w.id] != w.owner_id for w in creations):
        raise ValueError("Current work ownership must match its committed transfer history.")
    for ack in acknowledgments:
        record = history.get(ack.record_id)
        if (ack.npc_id not in live_ids
                or record is None or not any(t.id == ack.transfer_id
                    and {t.giver_id, t.recipient_id} == {player_id, ack.npc_id} for t in record.transfers)):
            raise ValueError("A pending acknowledgment needs its committed player handoff.")
    if len({a.transfer_id for a in acknowledgments}) != len(acknowledgments):
        raise ValueError("Duplicate pending transfer acknowledgment.")


def transfer_matches(intent, giver_id, recipient_id, item, quantity):
    return intent is not None and (intent.giver_id, intent.recipient_id, intent.item, intent.quantity) == (
        giver_id, recipient_id, item, quantity)


def validate_saved_handoffs(ctx):
    state = ctx.state
    handoffs, history, live, live_ids, player = state.handoffs, ctx.history, ctx.live, ctx.live_ids, state.player
    ownership, mode, creation_ids, events, reserved, tree_id = ctx.physical, state.mode, ctx.creation_ids, ctx.events, ctx.reserved, state.tree.id
    from activity import validate_activity_target
    from conversation_history import source_record
    for handoff in handoffs:
        pair = {handoff.initiator, handoff.recipient}
        residents_in_pair = pair - {player.id}
        record = source_record(history, handoff.record)
        if (len(pair) != 2 or not residents_in_pair <= live_ids
                or residents_in_pair & reserved or set(record.participant_ids) != pair
                or set(handoff.saved_states) != residents_in_pair):
            raise ValueError("Invalid handoff participants, reservation, or source record.")
        if player.id in pair:
            resident_id = next(iter(residents_in_pair))
            if (not ownership or ownership.kind != "resident"
                    or ownership.npc_id != resident_id or ownership.record_id != record.id
                    or ownership.stage not in ("conversation", "handoff")
                    or mode != "player" or not player.alive):
                raise ValueError("A player handoff requires its actual exclusive physical owner.")
        locations = {live[i].location if i != player.id else player.location for i in pair}
        if len(locations) != 1 or player.id in pair and not handoff.arrived:
            raise ValueError("Handoff participants must share their physical location.")
        if (handoff.transferred != bool(handoff.outcome_event_id)
                or handoff.transferred != (handoff.elapsed >= GIFT_GIVE_SECONDS)
                or not handoff.arrived and handoff.elapsed != 0
                or handoff.item != "fruit" and (handoff.item not in creation_ids or handoff.quantity != 1)):
            raise ValueError("Invalid handoff progress or possession.")
        if not transfer_matches(record.transfer_intent, handoff.initiator, handoff.recipient, handoff.item, handoff.quantity):
            raise ValueError("A handoff needs matching accepted authorization.")
        receipts = [t for t in record.transfers
                    if transfer_matches(t, handoff.initiator, handoff.recipient, handoff.item, handoff.quantity)]
        if handoff.transferred and (not receipts or handoff.outcome_event_id not in events
                or events[handoff.outcome_event_id].kind != ("fruit_shared" if handoff.item == "fruit" else "work_shared")
                or events[handoff.outcome_event_id].history != record.id
                or set(events[handoff.outcome_event_id].participants) != pair):
            raise ValueError("A committed handoff needs its receipt and event.")
        if handoff.rewritten_thanks and (len(handoff.rewritten_thanks) != len(handoff.thanks)
                or any(not text.strip() for text in handoff.rewritten_thanks)):
            raise ValueError("Rewritten thanks must match the original parts.")
        reserved.update(residents_in_pair)
        for participant_id, activity in handoff.saved_states.items():
            validate_activity_target(activity, live[participant_id], tree_id, live_ids)
