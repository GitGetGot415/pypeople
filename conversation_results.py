"""Main-thread conversation acceptance; no screens, providers or worker scheduling.

The UI supplies only whether a reply and its original offer are still open.
Outcome-specific commits remain in resident, snake and transfer code.
"""

from constants import (CONVERSATION_APPLYING_PLAYER_REQUEST, CONVERSATION_COMPLETED, CONVERSATION_FAILED,
                       CONVERSATION_REPLY_WAITING_FOR_PLAYER, CONVERSATION_WAITING_FOR_DONE, CONVERSATION_WALKING_TO_GIVE)
from transfer_records import TransferIntent

from player_records import PlayerCall, PlayerReplyCall
from dataclasses import dataclass
from apartments import offer_invitation, pending_invitation
from conversation_history import describe_opinion_change
from actor_identity import (actor_catalog, name_forms)
from information import orientation_speech
from activity import Behavior
from player import owns_conversation
from simulation import PendingAction


@dataclass
class AcceptedReply:
    error: str | None = None
    pending_offer: dict | None = None
    invitation: str = ""


def accept_generated_dialogue(conversation, result, error, failed_turns, rewritten, *, reading, dialogue_validation):
    if not conversation.accepts_response:
        return "The conversation ended before the reply arrived."
    if error:
        conversation.fail_dialogue(error, failed_turns=failed_turns)
        return error
    try:
        conversation.set_response(result, reading=reading, rewritten_parts=rewritten,
                                  dialogue_validation=dialogue_validation)
    except ValueError as exc:
        conversation.fail_dialogue(str(exc), failed_turns=result.turns)
        return str(exc)
    return None


def remember_response_opinions(world, npc, response, snapshot):
    updates = {}
    for entry in response.opinion_updates:
        actor_id = response._opinion_targets.get(entry.person)
        permitted = [actor for actor in snapshot.get("other_npcs", []) if actor.get("kind") != "player"
                     and actor["name"].casefold() == entry.person.strip().casefold()]
        if (len(permitted) != 1 or permitted[0].get("actor_id") != actor_id
                or world.actor_by_id(actor_id) is None):
            return "I couldn't identify that person to remember my opinion. Please try again."
        updates[actor_id] = {"opinion": entry.opinion, "reason": entry.reason}
    if updates:
        try:
            npc.remember_opinions(updates)
        except ValueError:
            return "I couldn't update my opinion. Please try again."
    return None


def accept_resident_reply(world, request, result, error, rewritten, *, visible, record_open):
    npc, record = request.actor, request.record
    snapshot, message = request.snapshot, request.message
    pending_offer = None
    npc.player_reply.complete()
    if not error and (not owns_conversation(world, npc) or world.player_interaction.id != snapshot.get("_ownership_id")
            or npc.busy or npc.command_revision != request.actor_revision or request.roster_revision != world.roster_revision):
        error = "That reply arrived after the NPC became busy. Please send the request again."
    if not error:
        try:
            from transfers import validate_player_transfer
            from player import call_valid
            from transfers import validate_transfer
            validate_player_transfer(result, snapshot)
            context = snapshot.get("player_interaction", {})
            if context.get("call") and not call_valid(world, npc, PlayerCall(**context["call"])):
                raise ValueError("The reason for this call is no longer available.")
            if context.get("incoming_gift"):
                if not record_open:
                    raise ValueError("The gift offer closed before the resident accepted it.")
                incoming = context["incoming_gift"]
                validate_transfer(world, world.player, npc, incoming["item"], incoming["quantity"])
            if result.gift_offer:
                offer = result.gift_offer
                validate_transfer(world, npc, world.player, offer.item, offer.quantity)
        except ValueError as exc:
            error = str(exc)
    if not error and npc.speech_only:
        if (result.action != "SAY" or result.gift_offer or result.gift_acceptance or result.apple_promise
                or result.invitation or result.invitation_acceptance or result.invitation_resolution):
            error = "The snake can speak but cannot carry out resident tasks, gifts or invitations."
    if not error:
        try:
            invitation = result.invitation
            if invitation and invitation.evidence not in result.reply:
                raise ValueError("The invitation is not supported by the original reply.")
            if result.invitation_acceptance:
                offer = pending_invitation(world)
                if not offer or offer.id != result.invitation_acceptance or not message:
                    raise ValueError("This invitation acceptance has no current offer.")
        except ValueError as exc:
            error = str(exc)
    planned_action = None
    opinion_outcomes = []
    if not error:
        from player_commands import validate_action
        from dialogue import ActionIntent
        intent = snapshot.get("request_intent") or result._authorized_intent
        command = snapshot.get("player_interaction", {}).get("command")
        if command is not None:
            intent = command
        declined = bool(intent and intent["action"] == "GIVE_ITEM" and result.action == "SAY"
                        and not result.target and not result.item and not result.quantity)
        if intent and not declined and ActionIntent(**intent).model_dump() != result.model_dump(include=set(ActionIntent.model_fields)):
            error = "The reply changed the authorized player request."
        if not error and result.action != "SAY":
            try:
                if not intent:
                    raise ValueError("This action has no interpreted player request.")
                validate_action(world, npc, result, accepting=True)
                planned_action = PendingAction(npc.id, result.action, npc.command_revision,
                    result.target or None, message or record.topic, record.id if record else None,
                    result.item, result.quantity)
            except ValueError as exc:
                error = str(exc)
    if not error:
        opinion_outcomes = [describe_opinion_change(npc.name, entry.person,
            npc.opinion_score(result._opinion_targets.get(entry.person), default=None), entry.opinion, entry.reason)
            for entry in result.opinion_updates]
        error = remember_response_opinions(world, npc, result, snapshot)
    if not error and planned_action is not None:
        if planned_action.action == "GIVE_ITEM" and record is not None:
            record.transfer_intent = TransferIntent(giver_id=npc.id, recipient_id=planned_action.target_id,
                                          item=planned_action.item, quantity=planned_action.quantity)
        world.pending_actions.append(planned_action)
    closes_dialogue = bool(not error and result.end_conversation)
    if closes_dialogue and result.action == "SAY":
        from player import player_reply_closes
        closes_dialogue = player_reply_closes(True, result.reply, rewritten[0] if rewritten else ())
    if not error and record is not None:
        record.dialogue_errors = list(dict.fromkeys([*record.dialogue_errors, *result._dialogue_errors]))
        record.outcomes.extend(opinion_outcomes)
        from player import accept_player_evidence
        from transfers import Handoff
        context = snapshot.get("player_interaction", {})
        incoming = context.get("incoming_gift")
        opening = snapshot.get("request_kind") in ("player_opening", "player_gift", "player_acknowledgment")
        accept_player_evidence(world, npc, record, message, result, opening=opening)
        from player import accept_player_promises
        accept_player_promises(world, npc, record, result, message, opening=opening)
        if incoming and result.gift_acceptance:
            Handoff(world.player, npc, incoming["item"], incoming["quantity"], record)
        if visible:
            pending_offer = ({**result.gift_offer.model_dump(), "npc_id": npc.id,
                "revision": npc.command_revision, "record_id": record.id} if result.gift_offer else None)
        elif not npc.speech_only and snapshot.get("request_kind") != "player_acknowledgment":
            npc.player_call = PlayerReplyCall(kind="reply", reference=record.id, closing=closes_dialogue,
                gift_offer=result.gift_offer)
            npc.resume_pending = True
    npc.reply = f"Request failed: {error}" if error else result.reply
    npc.rewritten_reply = rewritten[0][0] if not error and rewritten else ""
    npc.player_reply.closing = bool(closes_dialogue and npc.selected)
    if error:
        npc.set_emotion("neutral", 0)
    else:
        npc.set_emotion(result.emotion.name, result.emotion.intensity)
    if record is not None:
        if not error:
            record.name_forms = name_forms(actor_catalog(world))
            record.orientation_discussed = record.orientation_discussed or orientation_speech([npc.reply])
            record.add_turn(npc.name, npc.reply, [{
                "text": npc.reply,
                "emotion": {"name": npc.emotion, "intensity": npc.emotion_intensity},
            }], rewritten[0] if rewritten else (), speaker_id=npc.id)
            from interactions import observe_conversation
            observe_conversation(world, record, world.player, npc, accepted=True)
        record.error = error or ""
        waiting_for_done = npc.waiting_for_done
        if error:
            record.status = CONVERSATION_FAILED
            record.outcomes.insert(0, f"The request could not be completed: {error}")
        else:
            if npc.unread_player_reply:
                record.status = CONVERSATION_REPLY_WAITING_FOR_PLAYER
                outcome = "Reply ready; waiting for the player to return to the conversation."
            elif waiting_for_done:
                record.status = CONVERSATION_WAITING_FOR_DONE
                outcome = "Accepted; starts after the player closes the conversation."
            elif npc.interaction_blocked:
                record.status = CONVERSATION_APPLYING_PLAYER_REQUEST
                outcome = "Accepted; applies on the next simulation tick."
            else:
                record.status = CONVERSATION_COMPLETED
                outcome = "Replied without changing their activity."
            record.outcomes.insert(0, outcome)
    if not error and result.invitation:
        offer_invitation(world, npc, record, result.invitation.destination, result.reply)
    if not error and result.invitation_resolution:
        offer = pending_invitation(world)
        if offer:
            offer.status = result.invitation_resolution
    return AcceptedReply(error, pending_offer, result.invitation_acceptance if not error else "")


def apply_pending_action(world, action, spacing):
    meeting = None
    npc = world.npc_by_id(action.npc_id)
    if npc is None:
        return None, "", None
    npc.player_reply.complete()
    error = ""
    if (world.npc_by_id(npc.id) is not npc or npc.busy or npc.command_revision != action.revision
            or npc.behavior == Behavior.GO_HOME
            or action.record is None and npc.behavior in (Behavior.REST, Behavior.PAINT, Behavior.COMPOSE)):
        error = "That request is no longer available because the NPC's activity changed."
    if not error:
        from dialogue import ActionIntent
        from player_commands import validate_action
        try:
            validate_action(world, npc, ActionIntent(action=action.action, target=action.target_id or "",
                                                    item=action.item, quantity=action.quantity))
        except ValueError as exc:
            error = str(exc)
    if error:
        pass
    elif action.action == "TALK_TO_NPC":
        target = world.npc_by_id(action.target_id)
        if target is None:
            error = "That person is unavailable."
        elif target.interaction_blocked:
            error = "That person has a pending player request. Try again once it finishes."
        else:
            try:
                meeting = world.start_conversation(npc, target, action.topic, spacing=spacing(npc, target))
            except ValueError as exc:
                error = str(exc)
    elif action.action == "GIVE_ITEM":
        target = world.participant_by_id(action.target_id)
        try:
            record = world.record_by_id(action.record) if action.record is not None else None
            gift_spacing = spacing(npc, target) if target and target is not world.player else 0
            gift = npc.give_items(target, action.item, action.quantity, record, gift_spacing)
            if record is None:
                world.history.append(gift.record)
        except ValueError as exc:
            error = str(exc)
    else:
        try:
            npc.apply_action(action.action, world.tree)
        except ValueError as exc:
            error = str(exc)
    if action.record is not None:
        record = world.record_by_id(action.record)
        record.status = CONVERSATION_FAILED if error else (CONVERSATION_WALKING_TO_GIVE if action.action == "GIVE_ITEM" else CONVERSATION_COMPLETED)
        if error:
            record.error = error
            record.outcomes[0:1] = [error]
        else:
            outcomes = {
                "GO_TO_TREE": "Started walking to the tree.",
                "TALK_TO_NPC": f"Started a conversation with {target.name}." if action.action == "TALK_TO_NPC" else "",
                "GIVE_ITEM": f"Walking to give {action.quantity} {action.item} to {target.name}." if action.action == "GIVE_ITEM" else "",
                "EAT": "Started eating an apple.",
            }
            record.outcomes[0:1] = [outcomes[action.action]]
    if error:
        npc.player_reply.closing = False
        npc.reply = error
        npc.rewritten_reply = ""
        npc.set_emotion("neutral", 0)
    return npc, error, meeting
