"""Player food requests and independent survival pursuit, grounded in original speech and receipts."""

import re
from simulation.nutrition import urgent_food, starves_next_day
from simulation.player_records import PlayerCallReason, PlayerFoodRequest
from dialogue.text import unquoted_speech


def current_request(world, npc):
    request = npc.player_food_request
    return request if request is not None and request.day == world.day else None


def resolve_food(world, npc):
    request = current_request(world, npc)
    if request and (npc.inventory.get("fruit", 0) or not urgent_food(npc)):
        request.status = "resolved"
        request.pursuing = False
    return request


def food_call_reason(world, npc):
    from simulation.autonomy import food_alternative
    from simulation.social.relationships import food_refusal, confrontation_topic, grievance_against
    from simulation.social.records import claim_by_id, fact_by_id
    grievance = grievance_against(npc, world.player)
    if grievance and fact_by_id(world, claim_by_id(world, grievance).fact_id).kind not in ("apple_refused", "harvest"):
        return None
    request = resolve_food(world, npc)
    if not urgent_food(npc) or npc.inventory.get("fruit", 0) or food_alternative(npc, world):
        if request:
            request.pursuing = False
        return None
    if request is None or request.status == "approaching":
        return PlayerCallReason(key=f"food:{world.day}", kind="food", reference=str(world.day),
            reason="Ask the player for one apple. Explain my own hunger and, if applicable, risk of starvation next day. No food has been handed over.")
    if (request.status in ("refused", "deferred", "offered") and request.refusal_claim_id
            and not request.attack_resolved and starves_next_day(npc)
            and food_refusal(npc, world.player) == request.refusal_claim_id):
        return PlayerCallReason(key=f"food_confrontation:{world.day}", kind="food_confrontation",
            reference=request.refusal_claim_id, reason=confrontation_topic(npc, request.refusal_claim_id)
            + " I still need an apple now to survive the next day; a later promise has not supplied food.")
    return None


def begin_food_call(world, npc, reason):
    request = current_request(world, npc)
    if request is None:
        request = npc.player_food_request = PlayerFoodRequest(day=world.day)
    request.pursuing = reason.kind == "food_confrontation"
    return request


def record_food_opening(world, npc, record):
    request = current_request(world, npc)
    if request is None or record.purpose != "food_request" or request.status != "approaching":
        return
    request.status = "asked"
    request.conversation_id = record.id
    event = world.publish("food_requested", (npc, world.player), f"{npc.name} asked the player for an apple.", record)
    from dialogue.history import remember
    remember(npc, world.player, event, "conversation", "I asked the player for an apple; none has been handed over.", 0)


def food_answer(message):
    """Conservative classifications: unclear, conditional, quoted and questioning replies stay unresolved."""
    text = unquoted_speech(message).strip().replace("\u2019", "'").casefold()
    if not text or "?" in text or re.search(r"\b(?:if|unless|maybe|perhaps|but)\b", text):
        return ""
    if re.search(r"\bi (?:have no|(?:don't|do not) have(?: any| an?| one)?)\s+(?:apples?|fruit)\b", text):
        return "unable"
    direct = r"(?:(?:give|spare)\s+(?:you\b|(?:an?|any|one|my|the)\s+(?:apples?|fruit)\b)|share(?:\s+(?:with you\b|(?:my|any|an?|one|the)\s+(?:apples?|fruit)\b)|[.!]|$)|help(?:\s+you\b|[.!]|$))"
    if re.search(r"\b(?:give|share|spare|help)\b", text) and not re.search(direct, text):
        return ""
    if re.search(r"\b(?:apples?|fruit)\s+(?:to|with)\s+(?!you\b|anyone\b|give\b|eat\b)\w+", text):
        return ""
    if re.fullmatch(r"no(?:[, ]+(?:thanks|thank you))?[.!]*", text) or re.search(
            r"\bi (?:won't|will not|refuse to|am not going to|don't want to|do not want to)\s+" + direct, text):
        return "refused"
    if re.search(r"\b(?:tomorrow|later|next day|not now)\b", text) and (
            re.search(r"\b(?:apples?|fruit|give|share|spare)\b", text) or
            re.fullmatch(r"(?:later|tomorrow|not now)[.!]*", text)):
        return "deferred"
    if re.fullmatch(r"(?:yes|sure|okay|ok|of course)[.!]*", text) or re.search(
            r"\bi(?: will|'ll| can)\s+" + direct, text):
        return "offered"
    return ""


def record_food_answer(world, npc, record, message):
    request = current_request(world, npc)
    if request is None or request.status in ("approaching", "resolved") or not message:
        return
    status = food_answer(message)
    if not status:
        return
    text = unquoted_speech(message).strip().casefold()
    if re.fullmatch(r"(?:no(?:[, ]+(?:thanks|thank you))?|yes|sure|okay|ok|of course|later|tomorrow|not now)[.!]*", text):
        previous = next((turn.text for turn in reversed(record.turns[:-1]) if turn.speaker_id == npc.id), "")
        if not re.search(r"\b(?:apples?|fruit)\b", previous, re.I) or not re.search(
                r"\b(?:give|share|spare|need)\b", previous, re.I):
            return
    index = len(record.turns) - 1
    if index in record.accepted_player_turns:
        return
    record.accepted_player_turns.append(index)
    from simulation.player import record_claim, update_player_opinion
    record_claim(world, world.player, record, message, index, npc.id)
    request.status, request.evidence = status, message
    if status == "unable":
        request.refusal_claim_id = None
        request.pursuing = False
    if status in ("refused", "deferred"):
        text = (f'The player declined an immediate apple for {npc.name}, saying: "{message}"'
                if status == "refused" else f'The player deferred an immediate apple for {npc.name}, saying: "{message}"')
        event = world.publish("food_refused", (npc, world.player), text, record)
        from simulation.social.relationships import food_refusal
        request.refusal_claim_id = food_refusal(npc, world.player)
        update_player_opinion(npc, -5 if status == "refused" else -1, text, scaled=False)
    else:
        text = f'The player answered my food request: "{message}". No apple has been handed over.'
        event = world.publish("player_conversation", (world.player, npc), text, record)
    from dialogue.history import remember
    remember(npc, world.player, event, "refused" if status in ("refused", "deferred") else "conversation", text, 0)
    record.outcomes.append(text)


def food_arrival(world, npc, interaction):
    """Return True when this is a food approach; resolve a desperate arrival before UI input."""
    call = npc.player_call
    record = world.record_by_id(interaction.record_id)
    if record is None:
        return False
    kind = call.kind if call else ("food_confrontation" if record.food_confrontation_day is not None else
                                  "food" if record.purpose == "food_request" and record.origin == "autonomous" else "")
    if kind not in ("food", "food_confrontation"):
        return False
    reference = call.reference if call else record.grievance_id
    from constants import CONVERSATION_COMPLETED, CONFRONTATION_OPINION
    record.status = CONVERSATION_COMPLETED
    request = resolve_food(world, npc)
    from simulation.social.relationships import food_refusal
    from simulation.autonomy import food_alternative
    from simulation.player import end_interaction
    if kind == "food":
        if request is None or request.status == "resolved":
            end_interaction(world, "Food is no longer needed from the player.")
        return True  # The first request never rolls a hunger attack.
    if request and request.attack_resolved:
        return True  # A duplicate arrival notification leaves the peaceful conversation intact.
    if (request is None or request.refusal_claim_id != reference
            or food_refusal(npc, world.player) != reference or not starves_next_day(npc)
            or npc.inventory.get("fruit", 0) or food_alternative(npc, world)):
        if request:
            request.pursuing = False
        end_interaction(world, "The immediate food confrontation is no longer needed.")
        return True
    request.attack_resolved, request.pursuing = True, False
    unavailable = world.killing_unavailable(npc, world.player)
    if npc.opinion_score(world.player.id) > CONFRONTATION_OPINION:
        record.outcomes.append("The resident was not hostile enough to attack over the unresolved food request.")
    elif unavailable:
        record.outcomes.append(f"The resident could not attack: {unavailable}")
    else:
        from simulation.social.relationships import killing_chance
        if npc.rng.random() < killing_chance(npc.opinion_score(world.player.id)):
            record.outcomes.append("The resident attacked immediately on returning over the refused food request.")
            world._finish_killing(npc, world.player, record)
        else:
            record.outcomes.append("The resident's food confrontation attack roll failed.")
    return True
