"""Accepted possession transfers and their shared approach/handoff lifecycle."""

import re
from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from dialogue import ConversationPart
from uuid import uuid4
from constants import (GIFT_GIVE_SECONDS, GIFT_SECONDS, PLAYER_TALK_DISTANCE, PLAYER_LABEL,
                      CONVERSATION_MAX_AGREEMENTS)
from conversation_history import ConversationRecord, remember
from interactions import company_point, face_each_other
from dialogue_rules import unquoted_speech
from dialogue_context import actor_id


def validate_transfer(world, giver, recipient, item, quantity, *, interaction=None, player_authorization=None):
    """Always enforce authority, including with speech diagnostics disabled."""
    from locations import available, colocated
    from player import active_player, owns_resident
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
        authorized = bool(player_authorization and world.player_interaction
            and player_authorization == world.player_interaction.id and owns_resident(world, giver))
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
        self.record = record or ConversationRecord(labels, "Give items", "Walking to give",
                                                  participant_ids=[initiator.id, recipient.id])
        if set(self.record.participant_ids) != {initiator.id, recipient.id}:
            self.record = ConversationRecord(labels, "Directed gift", "Walking to give",
                authorization_id=record.id, participant_ids=[initiator.id, recipient.id])
            initiator.world.history.append(self.record)
        if player_exchange:
            self.world.player_interaction.record_id = self.record.id
        self.record.transfer_intent = dict(giver_id=initiator.id, recipient_id=recipient.id, item=item, quantity=quantity)
        self.saved_states = {}
        self.record.status = "Giving items" if already_met else "Walking to give"
        for npc in self.residents:
            self.saved_states[npc.id] = npc.suspend_activity(
                "Exchanging a gift with the player" if player_exchange else
                "Giving items" if npc is initiator else "Receiving items")
            npc.gift = self
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
            npc.gift = self

    @property
    def residents(self):
        return tuple(actor for actor in (self.initiator, self.recipient) if actor is not self.world.player)

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
                self.record.status = "Giving items"
                face_each_other(giver, recipient)
            return
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
            if not self.player_exchange:
                self.record.outcomes[0:1] = [f"Gave {self.quantity} {self.item} to {recipient.name}."]
                self.record.status = "Thanking"
            elapsed = max(GIFT_GIVE_SECONDS, elapsed)
        self.elapsed = elapsed
        if self.elapsed + 1e-9 >= GIFT_SECONDS:
            self.finish()

    def finish(self, error=""):
        if not self.active:
            return
        self.active = False
        self.record.status = "Failed" if error and not self.transferred else "Completed"
        self.record.error = error if not self.transferred else ""
        if error and self.transferred:
            self.record.outcomes.append(f"The transfer remains committed. {error}")
        elif error and not self.player_exchange:
            self.record.outcomes[0:1] = [error]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient), error, self.record)
            self.outcome_event_id = event.id
        for npc in self.residents:
            state = self.saved_states[npc.id]
            other = self.recipient if npc is self.initiator else self.initiator
            if npc.gift is not self:
                continue
            npc.gift = None
            if npc.world.npc_by_id(npc.id) is not npc:
                continue
            npc.restore_after_interaction(state, other.id, preserve_facing=True)
            if self.record.purpose == "food_request":
                npc.pause_after_conversation(True)
        if self.player_exchange:
            from player import owns_resident, end_interaction
            resident = self.residents[0]
            if owns_resident(self.world, resident) and self.world.player_interaction.stage == "handoff":
                end_interaction(self.world, error or "The accepted handoff finished.")


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
    speech = unquoted_speech(turns[gift.offer_turn].text).casefold()
    validate_gift_offer_speech(turns[gift.offer_turn].text, gift.item, gift.quantity,
                               giver.get("inventory", {}).get("fruit", 0))
    if gift.item == "fruit":
        if gift.quantity > giver.get("inventory", {}).get("fruit", 0) or not re.search(r"\b(?:apple|apples|fruit)\b", speech):
            raise ValueError("The giver must own and offer the actual apples.")
    else:
        work = next((w for w in giver.get("owned_works", []) if w["id"] == gift.item), None)
        if work is None or gift.quantity != 1 or work["title"].casefold() not in turns[gift.offer_turn].text.casefold():
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
    speech = unquoted_speech(text).casefold()
    return bool(re.search(r"\b(?:i accept|i['’]ll take|yes|thank you|thanks)\b", speech)
        and not re.search(r"\b(?:no thanks|decline|cannot accept|can't accept|don't want|do not want|if|unless|maybe|perhaps|not sure)\b", speech))


def validate_gift_offer_speech(text, item, quantity, stock):
    from dialogue_rules import unquoted_speech
    speech = unquoted_speech(text).casefold()
    if (not re.search(r"\b(?:offer|would you like|want|give you|for you|have this|take this)\b", speech)
            or re.search(r"\b(?:not|never|won't|cannot|can't|don't|do not)\b[^.!?]{0,35}\b(?:offer|give|have|take|for you)\b", speech)):
        raise ValueError("A gift offer needs supporting offer speech.")
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
        if promise.giver_id == snapshot.get("npc_id"):
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
        if offer.item == "fruit":
            if offer.quantity > snapshot.get("inventory", {}).get("fruit", 0):
                raise ValueError("An offer cannot spend unavailable apples.")
            if not re.search(r"\b(?:apple|apples|fruit)\b", speech):
                raise ValueError("Identify the offered apples in speech.")
        else:
            work = next((w for w in snapshot.get("owned_works", []) if w["id"] == offer.item), None)
            if work is None or offer.quantity != 1 or work["title"].casefold() not in result.reply.casefold():
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
