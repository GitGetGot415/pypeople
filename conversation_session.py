"""Main-thread conversation orchestration independent of screens and rendering."""

from dataclasses import asdict
from interactions import player_request_pending
from uuid import uuid4
from constants import PLAYER_CONTEXT_TRANSFER_LIMIT, PLAYER_PROMISE_DEADLINE_STEPS
from conversation_history import ConversationRecord
from conversation_results import accept_generated_dialogue, accept_resident_reply, accept_snake_reply
from npc import NPC
from snake import SnakeEncounter
from social import concern_description


class ConversationSession:
    def __init__(self, world, requests, *, session_id, provider, api_key, ollama_model,
                 voice_rewrites=True, dialogue_validation=True):
        self.simulation, self.requests, self.session_id = world, requests, session_id
        self.configure(provider=provider, api_key=api_key, ollama_model=ollama_model,
                       voice_rewrites=voice_rewrites, dialogue_validation=dialogue_validation)
        self.player_requests, self.conversation_requests = {}, {}
        self.pending_offer = None
        self.events = []
        self.view_actor, self.view_open, self.viewed_conversation = None, False, None
        self.menu_open = False

    def configure(self, *, provider, api_key, ollama_model, voice_rewrites, dialogue_validation):
        self.provider, self.api_key, self.ollama_model = provider, api_key, ollama_model
        self.voice_rewrites, self.dialogue_validation = voice_rewrites, dialogue_validation

    @property
    def tree(self):
        return self.simulation.tree

    @property
    def npcs(self):
        return self.simulation.npcs

    @property
    def conversation_history(self):
        return self.simulation.history

    @property
    def record(self):
        interaction = self.simulation.player_interaction
        return self.simulation.record_by_id(interaction.record_id) if interaction else None

    def select_record(self, record):
        interaction = self.simulation.player_interaction
        if interaction is not None and record is not None:
            interaction.record_id = record.id

    def set_view(self, *, actor, reply_open, viewed_conversation=None, menu_open=False):
        self.view_actor, self.view_open = actor, reply_open
        self.viewed_conversation, self.menu_open = viewed_conversation, menu_open

    def emit(self, kind, value=None):
        self.events.append((kind, value))

    def show_reply(self, text):
        self.emit("reply", text)

    def notice(self, text):
        self.emit("message", text)

    def open_view(self, npc):
        self.view_actor, self.view_open, self.viewed_conversation = npc, True, None
        self.emit("open", npc)

    def close_view(self):
        self.view_closed(terminate=False)
        self.emit("close")

    def view_closed(self, *, terminate=True):
        if terminate:
            self.end_player_conversation()
        self.pending_offer = None
        record = self.record
        if record:
            for offer in record.player_promise_offers:
                if offer.status == "offered":
                    offer.status = "closed"
        if self.view_actor is not None:
            self.view_actor.set_selected(False)
        self.view_open = False

    def submit(self, npc, message, *, active=True):
        from player import active_player
        message = message.strip()[:300]
        if self.simulation.paused or not active_player(self.simulation) or not active:
            return
        if npc is self.simulation.snake:
            self.submit_snake(message)
            return
        if (npc is None or not self.player_owns(npc) or npc.busy or npc.player_reply.closing
                or npc.interaction_blocked or not message):
            return
        if self.provider == "gemini" and not self.api_key:
            self.show_reply("Add an API key in Settings to generate a reply.")
            return
        snapshot = self.player_snapshot(npc, message)
        snapshot["player_dialogue"] = [{"speaker": t.speaker, "text": t.text}
                                      for t in self.record.turns[-12:]] if self.record else []
        snapshot["other_npcs"] = [{**other.snapshot(self.tree, private=False),
            "available_to_talk": not other.busy and not other.interaction_blocked and not other.selected and not other.player_call}
            for other in self.npcs if other is not npc and other.location == npc.location] + [self.player_target()]
        self._submit_player_message(npc, message, snapshot)

    def player_target(self):
        player = self.simulation.player
        return dict(npc_id=player.id, npc_name="You", kind="player", character={"Name": "You"},
                    inventory=player.inventory.copy(),
                    available_to_talk=True)

    def player_owns(self, npc, *, stages=("conversation",)):
        from player import owns_conversation
        return owns_conversation(self.simulation, npc, stages=stages)

    def end_player_conversation(self, reason="Conversation ended."):
        if not self.simulation or not self.simulation.player_interaction:
            return
        from player import end_interaction
        from snake import PlayerSnakeEncounter
        interaction = self.simulation.player_interaction
        for request_id, request in tuple(self.player_requests.items()):
            if request["npc"].id == interaction.npc_id:
                if player_request_pending(request):
                    self.requests.cancel(self.session_id, request_id)
                self.player_requests.pop(request_id, None)
        npc = self.simulation.npc_by_id(interaction.npc_id)
        if reason == "Conversation ended." and npc and (npc.gift or any(a.npc_id == npc.id and a.action == "GIVE_ITEM"
                and a.target_id == self.simulation.player.id for a in self.simulation.pending_actions)):
            interaction.wait_for_handoff()
            self.pending_offer = None
            return
        encounter = self.simulation.snake.encounter
        if isinstance(encounter, PlayerSnakeEncounter):
            if encounter.outcome == "eat":
                self.simulation.player_interaction = None
            else:
                encounter.finish(error=reason if reason != "Conversation ended." else "")
        else:
            end_interaction(self.simulation, reason)
        self.pending_offer = None
        self.emit("clear_call")

    def open_player_dialogue(self, npc):
        if not self.player_owns(npc):
            return
        self.open_view(npc)
        self.emit("clear_input")
        npc.set_selected(True)
        from snake import PlayerSnakeEncounter
        if npc is self.simulation.snake:
            self.select_record(npc.encounter.record)
            npc.player_reply.closing = npc.encounter.outcome == "eat"
        else:
            interaction = self.simulation.player_interaction
            record = next((r for r in self.simulation.history if r.id == interaction.record_id), None)
            if record is None:
                from player_story import scripted_player_opening
                record = scripted_player_opening(self.simulation, npc)
            self.select_record(record)
            gift = interaction.gift
            if gift:
                snapshot = self.player_snapshot(npc)
                work = next((w for w in self.simulation.creations if w.id == gift["item"]), None)
                incoming = dict(**gift, giver_id=self.simulation.player.id, title=work.title if work else "apples",
                                creator_id=work.creator_id if work else None)
                snapshot.update(request_kind="player_gift", player_interaction={"incoming_gift": incoming})
                self._submit_player_message(npc, "", snapshot)
                interaction.gift = None
        record = self.record
        pending = self.simulation.player_interaction.pending_request
        if record and pending and not npc.generation_pending and self.player_request(npc) is None:
            record.error = "Generation was interrupted by saving. Retry this exchange."
            request_id = uuid4().hex
            snapshot = self.player_snapshot(npc, pending["message"]) if npc is not self.simulation.snake else {}
            if pending["kind"]:
                snapshot.update(request_kind=pending["kind"], player_interaction=pending["context"])
            self.player_requests[request_id] = dict(id=request_id, npc=npc, record=record,
                message=pending["message"], snapshot=snapshot, revision=npc.command_revision,
                world_revision=self.simulation.roster_revision)
        self.show_reply(npc.reply if self.record else "")
        self.emit("focus")

    def accept_player_invitation(self, offer_id):
        from apartments import accept_invitation
        for request_id, request in tuple(self.player_requests.items()):
            if player_request_pending(request):
                self.requests.cancel(self.session_id, request_id)
            self.player_requests.pop(request_id, None)
        try:
            accept_invitation(self.simulation, offer_id)
            self.close_view()
        except ValueError as exc:
            self.notice(str(exc))

    def player_snapshot(self, npc, message=""):
        from player import work_choices, requested_work_id
        snapshot = npc.snapshot(self.tree, self.simulation.player.id)
        from apartments import invitation_choices, pending_invitation
        snapshot["invitation_choices"] = invitation_choices(self.simulation, npc)
        invitation = pending_invitation(self.simulation)
        snapshot["pending_invitation"] = asdict(invitation) if invitation else None
        interaction = self.simulation.player_interaction
        snapshot["closed_door_knock"] = interaction.knock_room if interaction else None
        from player import relevant_claims
        snapshot["player_claims"] = relevant_claims(self.simulation, npc.id, message)
        snapshot.update(player_id=self.simulation.player.id, player_inventory=self.player_target(),
                        owned_works=work_choices(self.simulation, npc.id, message))
        from social import phase_label
        snapshot.update(_player_message=message, phase_index=self.simulation.phase_index,
            apple_promise_choices=[dict(deadline=self.simulation.phase_index + step,
                label=phase_label(self.simulation.phase_index + step)) for step in PLAYER_PROMISE_DEADLINE_STEPS])
        snapshot["resolved_work_id"] = requested_work_id(self.simulation, npc.id, message)
        record = self.record
        offers = record.player_promise_offers if record and npc.id in record.participant_ids else []
        snapshot["player_apple_offer_count"] = len(offers)
        snapshot["pending_apple_promise"] = next((asdict(o) for o in offers if o.status == "offered"), None)
        snapshot["known_player_transfers"] = sorted([t for r in self.conversation_history for t in r.transfers
            if {t["giver_id"], t["recipient_id"]} == {self.simulation.player.id, npc.id}], key=lambda t: t["sequence"])[-PLAYER_CONTEXT_TRANSFER_LIMIT:]
        return snapshot

    def talk_to_call(self, npc):
        from player import call_valid, clear_call
        if npc.unread_player_reply:
            self.open_unread(npc)
            return ""
        if (self.simulation.paused
                or not npc.player_call or not call_valid(self.simulation, npc, npc.player_call)
                or not self.player_owns(npc, stages=("prompt",))):
            return "This call is unavailable while you or the resident are occupied."
        call = asdict(npc.player_call)
        self.close_view()
        self.simulation.player_interaction.stage = "conversation"
        self.open_view(npc)
        npc.set_selected(True)
        from player_story import scripted_player_opening
        record = scripted_player_opening(self.simulation, npc)
        if record is not None:
            self.select_record(record)
            self.show_reply(npc.reply)
            self.emit("focus")
            return ""
        snapshot = self.player_snapshot(npc)
        from player import work_choices
        snapshot["owned_works"] = work_choices(self.simulation, npc.id, call["reference"])
        snapshot.update(request_kind="player_opening", player_interaction={"call": call})
        snapshot["other_npcs"] = [self.player_target()]
        self._submit_player_message(npc, "", snapshot)
        if npc.interaction_blocked:
            self.show_reply("")
            clear_call(npc)
            return ""
        return "Could not start this call."

    def offer_player_item(self, recipient, item, quantity):
        from player import start_approach
        if self.simulation.paused:
            raise ValueError("Resume time before giving.")
        if self.player_owns(recipient) and not recipient.interaction_blocked and not recipient.busy:
            self.simulation.player_interaction.gift = dict(item=item, quantity=quantity)
            self.open_player_dialogue(recipient)
            return
        if self.menu_open:
            raise ValueError("Close the current blocking menu before giving.")
        blocked = self.simulation.player_menu_blocked
        self.simulation.player_menu_blocked = False
        try:
            start_approach(self.simulation, recipient, gift=dict(item=item, quantity=quantity))
        finally:
            self.simulation.player_menu_blocked = blocked

    def accept_gift_offer(self):
        from transfers import Handoff, validate_transfer
        offer = self.pending_offer
        npc = self.view_actor
        if not offer or not npc or self.simulation.paused:
            return
        try:
            if (offer["npc_id"] != npc.id or offer["revision"] != npc.command_revision
                    or self.record is None or offer["record_id"] != self.record.id):
                raise ValueError("The conversation changed. A fresh offer is needed.")
            validate_transfer(self.simulation, npc, self.simulation.player, offer["item"], offer["quantity"])
            Handoff(npc, self.simulation.player, offer["item"],
                          offer["quantity"], self.record)
            self.pending_offer = None
            self.notice("Gift accepted; the handoff is in progress.")
        except ValueError as exc:
            self.pending_offer = None
            self.notice(f"Offer unavailable: {exc}")

    def decline_gift_offer(self):
        npc, offer = self.view_actor, self.pending_offer
        self.pending_offer = None
        self.notice("Gift declined; no possessions changed hands.")
        if npc and offer and not self.pending_player_request(npc):
            snapshot = self.player_snapshot(npc)
            snapshot.update(request_kind="player_acknowledgment", player_interaction={"declined_offer": offer})
            self._submit_player_message(npc, "", snapshot)

    def retry_player_dialogue(self):
        npc = self.view_actor
        request = self.player_request(npc) if npc else None
        if npc is self.simulation.snake and request and not player_request_pending(request):
            self.submit_snake(request["message"])
            return
        if (not request or player_request_pending(request) or npc.interaction_blocked or npc.busy
                or self.simulation.paused or self.record is not request["record"]):
            return
        message = request["message"]
        snapshot = self.player_snapshot(npc, message)
        old = request["snapshot"]
        if old.get("request_kind") in ("player_opening", "player_gift", "player_acknowledgment"):
            snapshot.update(request_kind=old["request_kind"], player_interaction=old["player_interaction"])
            if old["player_interaction"].get("call"):
                from player import work_choices
                snapshot["owned_works"] = work_choices(self.simulation, npc.id, old["player_interaction"]["call"]["reference"])
        else:
            snapshot["other_npcs"] = [{**other.snapshot(self.tree, private=False),
                "available_to_talk": not other.busy and not other.interaction_blocked and not other.selected and not other.player_call}
                for other in self.simulation.npcs if other is not npc and other.location == npc.location] + [self.player_target()]
        if self.record:
            snapshot["player_dialogue"] = [{"speaker": t.speaker, "text": t.text} for t in self.record.turns[-12:]]
        self._submit_player_message(npc, message, snapshot)

    def player_request(self, npc):
        """Latest direct request for this actor, retained for retry after completion."""
        return next((request for request in self.player_requests.values() if request["npc"] is npc), None)

    def pending_player_request(self, npc):
        request = self.player_request(npc)
        return request if request and player_request_pending(request) else None

    def _submit_player_message(self, npc, message, snapshot, *, purpose="chat"):
        if not self.player_owns(npc):
            return
        previous = self.player_request(npc)
        if previous and player_request_pending(previous):
            return
        kind = snapshot.get("request_kind")
        if kind in ("player_opening", "player_gift", "player_acknowledgment"):
            snapshot.pop("last_reply", None)
        request_id = uuid4().hex
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key, message, snapshot,
            request_id=request_id, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation,
            queue_if_busy=True,
        ):
            self.show_reply("Could not start the request. Please try again.")
            return
        npc.player_reply.start(request_id)
        if self.record is None:
            topic = message or ("Player gift offer" if kind == "player_gift" else
                                "Gift acknowledgment" if kind == "player_acknowledgment" else "Resident initiated conversation")
            record = ConversationRecord(("You", npc.name), topic, "Waiting for reply", purpose=purpose,
                origin="autonomous" if kind == "player_opening" else "player", participant_ids=[self.simulation.player.id, npc.id])
            record.intended_subjects = {
                "You": message or "Listening to a resident's generated opening.",
                npc.name: (snapshot["player_interaction"]["call"]["reason"] if kind == "player_opening" else
                           concern_description(npc, npc.current_concern) if isinstance(npc, NPC)
                           else "Responding to the player's message.")}
            if kind == "player_gift":
                incoming = snapshot["player_interaction"]["incoming_gift"]
                record.intended_subjects["You"] = f'Offering {incoming["quantity"]} × {incoming["title"]}.'
            self.conversation_history.append(record)
            self.select_record(record)
        record = self.record
        self.simulation.player_interaction.record_id = record.id
        self.simulation.player_interaction.pending_request = dict(message=message, kind=snapshot.get("request_kind", ""),
            context=snapshot.get("player_interaction", {}))
        snapshot["_ownership_id"] = self.simulation.player_interaction.id
        if previous:
            self.player_requests.pop(previous["id"])
        self.player_requests[request_id] = dict(id=request_id, npc=npc, snapshot=snapshot, message=message,
            record=record, revision=npc.command_revision, world_revision=self.simulation.roster_revision)
        record.status = "Waiting for reply"
        record.error = ""
        if message and not (record.turns and record.turns[-1].speech == ("You", message)
                and len(record.turns) - 1 not in record.accepted_player_turns):
            record.add_turn("You", message, speaker_id=self.simulation.player.id)
        if snapshot.get("request_kind") != "player_acknowledgment":
            self.emit("clear_input")

    def request_conversation_dialogue(self, conversation, *, start=True):
        if not conversation.accepts_response or not conversation.arrived:
            return
        if conversation.dialogue.dialogue_state != "waiting":
            return
        if self.provider == "gemini" and not self.api_key:
            conversation.fail_dialogue("Add an API key in Settings to generate this conversation.")
            return
        if (self.session_id, conversation.id) not in self.requests.logs:
            self.requests.prepare(self.session_id, self.provider, conversation.snapshot(self.tree, self.npcs),
                                  request_id=conversation.id, ollama_model=self.ollama_model)
        if self.requests.busy or not start:
            return
        snapshot = conversation.snapshot(self.tree, self.npcs)
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key,
            f"Write what {conversation.initiator.name} and {conversation.recipient.name} say directly to each other.", snapshot,
            request_id=conversation.id, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation,
        ):
            conversation.fail_dialogue("Could not start dialogue generation.")
            return
        conversation.error = ""
        conversation.dialogue.dialogue_state = "pending"
        self.conversation_requests[conversation.id] = conversation

    def submit_snake(self, message):
        from snake import PlayerSnakeEncounter
        encounter = self.simulation.snake.encounter
        if (not isinstance(encounter, PlayerSnakeEncounter) or not self.player_owns(self.simulation.snake)
                or self.simulation.snake.interaction_blocked or not message or encounter.outcome == "eat"):
            return
        record = encounter.record
        self.select_record(record)
        turn = len(record.turns) - 1 if record.turns and record.turns[-1].speech == ("You", message) and len(record.turns) - 1 not in record.accepted_player_turns else len(record.turns)
        snapshot = encounter.player_snapshot(message, turn)
        self._submit_player_message(self.simulation.snake, message, snapshot, purpose="snake_player")

    def accept_result(self, request_id, result, error, failed_turns, rewritten, *, session_id, viewed=None):
        if session_id != self.session_id:
            self.requests.finish(session_id, request_id, "Request belongs to a previous session.")
            return
        conversation = self.conversation_requests.pop(request_id, None)
        if conversation is not None:
            error = (accept_generated_dialogue(conversation, result, error, failed_turns, rewritten,
                        reading=conversation is viewed, dialogue_validation=self.dialogue_validation)
                     if conversation.id == request_id else "The conversation ended before the reply arrived.")
            self.requests.finish(self.session_id, request_id, error)
            return
        request = self.player_requests.get(request_id)
        if request is None or not player_request_pending(request):
            self.requests.finish(self.session_id, request_id, "The player request is no longer active.")
            return
        request["npc"].player_reply.complete()
        npc, record = request["npc"], request["record"]
        if self.simulation.player_interaction:
            self.simulation.player_interaction.pending_request = None
        if npc is self.simulation.snake:
            accepted = accept_snake_reply(self.simulation, request, result, error, failed_turns, rewritten)
            if self.view_actor is npc:
                self.show_reply(npc.reply)
            self.requests.finish(self.session_id, request_id, accepted.error)
            if accepted.error:
                self.end_player_conversation(accepted.error)
            return
        if npc is None or self.simulation.npc_by_id(npc.id) is not npc:
            self.requests.finish(self.session_id, request_id, "The NPC is no longer available.")
            return
        visible = (self.view_open and self.view_actor is npc
                   and self.viewed_conversation is None)
        accepted = accept_resident_reply(self.simulation, request, result, error, rewritten,
            dialogue_validation=self.dialogue_validation, visible=visible,
            record_open=self.record is record)
        if not accepted.error and record is not None and visible:
            self.select_record(record)
            self.pending_offer = accepted.pending_offer
        if self.view_actor is npc and self.viewed_conversation is None:
            self.show_reply(npc.reply)
        self.requests.finish(self.session_id, request_id, accepted.error)
        if accepted.invitation:
            self.accept_player_invitation(accepted.invitation)
        elif accepted.error:
            self.end_player_conversation(accepted.error)

    def schedule_acknowledgments(self, *, paused, simulation_active):
        for acknowledgment in list(self.simulation.player_acknowledgments) if self.simulation else []:
            if self.requests.busy or paused or not simulation_active:
                break
            npc = self.simulation.npc_by_id(acknowledgment.npc_id)
            if npc and (npc.busy or npc.interaction_blocked or npc.unread_player_reply
                        or self.view_open and self.view_actor is not npc):
                continue
            record = next(r for r in self.conversation_history if r.id == acknowledgment.record_id)
            if npc and self.player_owns(npc):
                snapshot = self.player_snapshot(npc)
                snapshot.update(request_kind="player_acknowledgment", player_interaction={
                    "committed_transfer": next(t for t in record.transfers if t["id"] == acknowledgment.transfer_id)})
                self.select_record(record)
                self._submit_player_message(npc, "", snapshot)
                if not npc.interaction_blocked:
                    continue
            self.simulation.player_acknowledgments.remove(acknowledgment)

    def open_unread(self, npc):
        from player import call_valid, clear_call, player_reply_closes
        call = npc.unread_player_reply
        if not call or not call_valid(self.simulation, npc, call):
            clear_call(npc)
            return
        record = self.simulation.record_by_id(call.reference)
        self.select_record(record)
        self.open_view(npc)
        self.emit("clear_call")
        npc.set_selected(True)
        npc.player_reply.closing = call.closing
        if not npc.interaction_blocked:
            npc.player_reply.closing = player_reply_closes(npc.player_reply.closing, npc.reply,
                (npc.rewritten_reply,) if npc.rewritten_reply else ())
        self.pending_offer = ({**call.gift_offer.model_dump(), "npc_id": npc.id,
            "revision": npc.command_revision, "record_id": record.id} if call.gift_offer else None)
        clear_call(npc)
        record.status = "Waiting for Done" if npc.waiting_for_done else "Completed"
        self.emit("clear_input")
        self.show_reply(npc.reply)
        if not npc.player_reply.closing:
            self.emit("focus")

    def cancel_request(self, request_id, reason):
        conversation = self.conversation_requests.pop(request_id, None)
        if conversation is not None:
            conversation.fail_dialogue(reason)
        elif request_id in self.player_requests:
            request = self.player_requests[request_id]
            npc, record = request["npc"], request["record"]
            npc.player_reply.complete()
            npc.reply, npc.rewritten_reply = reason, ""
            if self.view_actor is npc:
                self.show_reply(reason)
            if record is not None:
                record.status, record.error = "Failed", reason
            self.end_player_conversation(reason)

    def cancel_unowned(self):
        for request_id, request in tuple(self.player_requests.items()):
            if player_request_pending(request) and not self.player_owns(request["npc"]):
                self.requests.cancel(self.session_id, request_id)
                request["npc"].player_reply.complete()

    def update_progress(self, active_session):
        for (session_id, request_id), worker in self.requests.workers.items():
            pending = self.conversation_requests.get(request_id)
            if pending is not None and worker.control and worker.control.stage == "Waiting for provider admission":
                pending.dialogue.wait_elapsed = 0.0
            if (session_id == active_session and pending is not None
                    and getattr(pending, "request_stage", None) != worker.stage):
                pending.request_stage = worker.stage
                pending.dialogue.wait_elapsed = 0.0

    def shutdown(self):
        for request in self.player_requests.values():
            if player_request_pending(request):
                self.requests.cancel(self.session_id, request["id"])
                request["npc"].player_reply.complete()
                request["record"].status = "Ended before the reply arrived"
                request["record"].outcomes.insert(0, "Left the game before a reply was received.")
        self.end_player_conversation("Left the game before a reply was received.")
        for request_id in self.conversation_requests:
            self.requests.cancel(self.session_id, request_id)
        meetings = {id(npc.conversation): npc.conversation for npc in self.npcs if npc.conversation}
        if self.simulation.snake.encounter:
            encounter = self.simulation.snake.encounter
            meetings[id(encounter)] = encounter
        for meeting in meetings.values():
            meeting.finish(self.tree)
        self.player_requests.clear()
        self.conversation_requests.clear()
        self.pending_offer = None

    def retry_conversation_dialogue(self, conversation):
        if isinstance(conversation, SnakeEncounter):
            if self.requests.busy or conversation.active or self.simulation.npc_by_id(conversation.recipient.id) is None:
                return
            try:
                conversation.recipient.set_selected(False)
                if not self.simulation.join_tree(conversation.recipient, conversation.purpose, conversation.accusation):
                    raise ValueError("Tree service is unavailable. Join the line when ready.")
            except ValueError as exc:
                conversation.record.error = str(exc)
                return
            self.close_view()
            return
        if conversation.active or conversation.turns or self.requests.busy:
            return
        try:
            replacement = self.simulation.start_conversation(
                conversation.initiator, conversation.recipient, conversation.topic,
                origin=conversation.origin, purpose=conversation.purpose, opening_kind=conversation.opening_kind,
                focus_claim_id=conversation.focus_claim_id, goal_id=conversation.goal_id)
        except ValueError as exc:
            conversation.error = str(exc)
            return
        self.emit("view_conversation", replacement)
