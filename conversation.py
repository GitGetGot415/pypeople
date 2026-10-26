from copy import deepcopy
from typing import Annotated, Literal
from pydantic import Field
from dialogue import (AcceptedProposal, ConversationPart, DisclosureChoice, LeverageUse, ResidentGift,
                     ThirdPartyOpinion, EscalationChoice)
from social_records import ConcernKind, phase_label
from information import information_views, prepare_information
from social import (commit_agreements, commit_concern_use, commit_information, concern_description, concern_topic,
                    conversation_references, dialogue_failure_cooldown, permitted_proposals,
                    validate_live_agreements)
from interactions import (DialogueProgress, conversation_points, dialogue_finished, face_each_other,
                          finish_generated_dialogue, record_failed_dialogue, record_generated_turns,
                          release_resident)
from dataclasses import asdict, dataclass
from uuid import uuid4
from conversation_history import ConversationRecord, describe_opinion_change, remember, shared_conversation_history
from constants import (DIALOGUE_TIMEOUT_SECONDS, STARTING_DAY, CONVERSATION_CANCELLED, CONVERSATION_DIALOGUE_READY,
                       CONVERSATION_ENDED_BEFORE_DIALOGUE, CONVERSATION_ENDED_BEFORE_SHARING, CONVERSATION_FAILED,
                       CONVERSATION_FINISHED, CONVERSATION_WAITING_FOR_DIALOGUE, CONVERSATION_WALKING_TO_MEET,
                       CONVERSATION_OPINION_STEP)
from information import validate_orientation
from actor_identity import (actor_catalog, name_forms)
from transfers import Handoff
from transfer_records import TransferIntent
from npc import NPC


@dataclass(eq=False)
class ResidentProgress:
    """Durable domain progress; live actors and suspended activities attach separately."""
    accepted_gift: ResidentGift | None
    id: Annotated[str, Field(min_length=1)]
    origin: Literal['player', 'autonomous']
    purpose: Literal['chat', 'food_request', 'confrontation']
    opening_kind: ConcernKind | Literal["orientation"] | None  # Orientation is only Judy's first welcome.
    goal_id: str | None
    focus_claim_id: str | None
    food_outcome: Literal['', 'accepted', 'need', 'refused', 'unavailable']
    food_reason: str
    handoff_thanks: Annotated[list[ConversationPart], Field(max_length=3)]
    rewritten_handoff_thanks: list[str]
    outcome_event_id: str | None
    reactions: Annotated[list[tuple[str, float]], Field(max_length=2)]
    topic: str
    arrived: bool
    accepted_proposals: Annotated[list[AcceptedProposal], Field(max_length=2)]
    social_effects_applied: bool
    context_references: dict[str, Literal['active', 'completed', 'failed', 'pending', 'fulfilled', 'broken', 'cancelled']]
    accepted_disclosures: Annotated[list[DisclosureChoice], Field(max_length=4)]
    third_party_opinions: Annotated[list[ThirdPartyOpinion], Field(max_length=4)]
    leverage_uses: Annotated[list[LeverageUse], Field(max_length=2)]
    escalation: EscalationChoice | None
    accepted_revisions: dict[str, Annotated[int, Field(strict=True, ge=0)]]


class Conversation(ResidentProgress):
    """Temporarily reserve two NPCs, then restore their interrupted activities."""

    def __init__(self, initiator, recipient, topic, tree, spacing=0, *, origin="player", purpose="chat",
                 opening_kind=None, focus_claim_id=None, goal_id=None, _restored=None):
        if _restored is not None:
            self._restore(initiator, recipient, _restored)
            return
        from locations import available as available_outdoors, colocated
        from player import owns_resident
        if (initiator is recipient or initiator.busy or recipient.busy
                or owns_resident(initiator.world, initiator) or owns_resident(initiator.world, recipient)
                or initiator.world.visit and initiator.world.visit.host_id in (initiator.id, recipient.id)
                or initiator.player_call or recipient.player_call or recipient.selected or recipient.interaction_blocked
                or not available_outdoors(initiator) or not available_outdoors(recipient)
                or not colocated(initiator, recipient)):
            raise ValueError("Choose another NPC who is free to talk.")
        if purpose not in ("chat", "food_request", "confrontation"):
            raise ValueError("Resident meetings use chat or food_request.")
        if purpose == "confrontation":
            from social import grievance_against
            if (not isinstance(initiator, NPC) or not isinstance(recipient, NPC)
                    or not focus_claim_id or grievance_against(initiator, recipient) != focus_claim_id):
                raise ValueError("A confrontation requires two human residents and a specific retained, unresolved grievance.")
        if focus_claim_id and not any(b.claim_id == focus_claim_id for b in initiator.beliefs):
            raise ValueError("The initiator does not know the required account.")
        self.initiator = initiator
        self.recipient = recipient
        self.topic = topic
        self.id = uuid4().hex
        self.origin, self.purpose = origin, purpose
        self.concerns = {role: deepcopy(npc.current_concern)
                         for role, npc in (("initiator", initiator), ("recipient", recipient))}
        self.opening_kind = opening_kind
        self.focus_claim_id = focus_claim_id  # Required task account; an ordinary concern is optional.
        self.goal_id = goal_id
        self.escalation = None
        self.accepted_revisions = {}
        self.dialogue = DialogueProgress()
        self.food_outcome = ""
        self.food_reason = ""
        self.handoff_thanks = []
        self.rewritten_handoff_thanks = []
        self.outcome_event_id = None
        self.reactions = []
        self.arrived = False
        self.active = True
        self.record = ConversationRecord((initiator.name, recipient.name), topic, CONVERSATION_WALKING_TO_MEET,
                                         origin=origin, purpose=purpose, id=self.id,
                                         participant_ids=[initiator.id, recipient.id],
                                         grievance_id=focus_claim_id if purpose == "confrontation" else None)
        self.record.intended_subjects = {
            npc.name: concern_description(npc, self.concerns[role])
            for role, npc in (("initiator", initiator), ("recipient", recipient))}
        self.proposals = []
        self.accepted_proposals = []
        self.social_effects_applied = False
        self.context_references = {}
        self.name_actors = {}
        self.information_context = None
        self.accepted_disclosures = []
        self.third_party_opinions = []
        self.leverage_uses = []
        self.accepted_gift = None
        self.error = ""
        self.saved_states = []
        destinations = conversation_points(initiator, recipient, spacing)
        if destinations is None:
            raise ValueError("There isn't enough room here for a conversation.")
        for npc in (initiator, recipient):
            if purpose == "food_request":
                reason = (f"Asking {recipient.name} for an apple" if npc is initiator
                          else f"Answering {initiator.name}'s request for an apple")
            else:
                reason = "Having an automatic chat" if origin == "autonomous" else "In a player-initiated conversation"
            self.saved_states.append(npc.suspend_activity(reason))
            npc.world.reservations.acquire("conversation", self, npc)
        initiator.destination, recipient.destination = destinations

    @classmethod
    def restore(cls, initiator, recipient, record, *, progress, dialogue, saved_states,
                proposals, information_context, concerns):
        return cls(initiator, recipient, progress.topic, initiator.world.tree,
                   _restored=dict(record=record, progress=progress, dialogue=dialogue, saved_states=saved_states,
                                  proposals=proposals, information_context=information_context, concerns=concerns))

    def _restore(self, initiator, recipient, restored):
        self.__dict__.update(vars(restored["progress"]))
        self.initiator, self.recipient, self.record = initiator, recipient, restored["record"]
        self.dialogue, self.saved_states = restored["dialogue"], restored["saved_states"]
        self.proposals, self.information_context = restored["proposals"], restored["information_context"]
        self.concerns, self.name_actors, self.active = restored["concerns"], {}, True
        self.dialogue.retry_after_restore()
        initiator.world.reservations.acquire("conversation", self, initiator, recipient)

    @property
    def grievance_id(self):
        return self.record.grievance_id

    def other(self, npc):
        return self.recipient if npc is self.initiator else self.initiator

    def resident_activities(self):
        return zip((self.initiator, self.recipient), self.saved_states)

    def activity_snapshots(self):
        return iter(self.saved_states)

    @property
    def turns(self):
        return self.record.generated_turns()

    @property
    def error(self):
        return self.record.error

    @error.setter
    def error(self, message):
        self.record.error = message

    def set_response(self, response, *, reading=False, rewritten_parts=(), dialogue_validation=True):
        if not self.accepts_response:
            return
        from dialogue_validation import dialogue_validation_errors, validate_dialogue_format, validate_dialogue_speakers, validate_resident_grounding
        # Reject stale actor snapshots before displaying or committing anything.
        if getattr(self, "request_roster_revision", self.initiator.world.roster_revision) != self.initiator.world.roster_revision:
            raise ValueError("The resident roster changed while dialogue was being prepared. Retry this meeting.")
        if any(n.command_revision != self.request_actor_revisions[n.id] for n in (self.initiator, self.recipient)):
            raise ValueError("The participants changed activities before the reply arrived.")
        if self.initiator.speech_only or self.recipient.speech_only:
            if response.gift or response.accepted_proposals or response.handoff_thanks or response.escalation or response.leverage:
                raise ValueError("Snake meetings allow speech and sourced disclosures only.")
        validate_live_agreements(self, response, dialogue_validation=dialogue_validation)
        from transfers import validate_resident_gift
        validate_resident_gift(response.gift, response.turns, {
            "purpose": self.purpose, "arrival_focus": self.opening_kind == "orientation",
            "initiator": self.initiator.snapshot(self.initiator.world.tree),
            "recipient": self.recipient.snapshot(self.initiator.world.tree)})
        if response.gift and response.gift.accept_turn is not None:
            from transfers import validate_transfer
            world = self.initiator.world
            validate_transfer(world, world.npc_by_id(response.gift.giver_id), world.npc_by_id(response.gift.recipient_id),
                              response.gift.item, response.gift.quantity, interaction=self)
        if dialogue_validation:
            validate_dialogue_format(response.turns)
            validate_dialogue_speakers(response.turns, {"initiator": self.initiator.name, "recipient": self.recipient.name})
            initiator, recipient = getattr(self, "request_participants", None) or (
                self.initiator.snapshot(self.initiator.world.tree), self.recipient.snapshot(self.initiator.world.tree))
            validate_resident_grounding(response, {
                "_name_actors": self.name_actors,
                "initiator": initiator,
                "recipient": recipient,
                "_information_validation": asdict(self.information_context) if self.information_context else None})
        orientation_discussed = validate_orientation(response, {
            "arrival_focus": self.opening_kind == "orientation", "_dialogue_validation": dialogue_validation})
        if not dialogue_validation:
            self.record.dialogue_errors = list(dict.fromkeys([
                *response._dialogue_errors,
                *dialogue_validation_errors(response, self.snapshot(self.initiator.world.tree, self.initiator.world.npcs))]))
        # The exchange moves each opinion from where it stood when generation began, and by at most one step;
        # anything that changed it meanwhile, such as a harvest seen, is kept rather than overwritten.
        starts = self.information_context.opinions if self.information_context is not None else {}
        opinions = []
        for role, npc, partner, entry in (("initiator", self.initiator, self.recipient, response.initiator_opinion),
                                          ("recipient", self.recipient, self.initiator, response.recipient_opinion)):
            live = npc.opinion_score(partner.id)
            change = entry.opinion - starts.get(role, {}).get(partner.id, live)
            change = max(-CONVERSATION_OPINION_STEP, min(CONVERSATION_OPINION_STEP, change))
            opinions.append((npc, partner, entry.model_copy(update={"opinion": max(-100, min(100, live + change))})))
        from social import roll_escalation, validate_escalation
        validate_escalation(self, response, opinions[0][2].opinion)
        self.escalation = roll_escalation(self, response, opinions[0][2].opinion)
        self.accepted_revisions = ({n.id: n.command_revision for n in (self.initiator, self.recipient)}
                                   if self.escalation and self.escalation.action == "attack" else {})
        self.handoff_thanks = response.handoff_thanks
        self.accepted_gift = response.gift if response.gift and response.gift.accept_turn is not None else None
        self.rewritten_handoff_thanks = list(rewritten_parts[-1]) if rewritten_parts and response.handoff_thanks else []
        record_generated_turns(self, response.turns, rewritten_parts)
        self.record.outcomes = [
            describe_opinion_change(
                npc.name, partner.name, npc.opinion_score(partner.id, default=None), entry.opinion, entry.reason
            )
            for npc, partner, entry in opinions
        ]
        if self.escalation:
            from information import account_content, believed_harvest
            from social_records import claim_by_id, fact_by_id
            world = self.initiator.world
            claim = claim_by_id(world, self.grievance_id)
            fact = fact_by_id(world, claim.fact_id)
            # Matches the confrontation topic: the whole harvest the confronter believes, not one pickup.
            grievance = (believed_harvest(world, self.initiator.beliefs, fact) if fact.kind == "harvest" and claim.affirmed
                         else account_content(world, fact, claim.affirmed))
            self.record.outcomes.append(f"Confrontation decision: {self.escalation.action}. Grievance: {grievance}")
        self.record.status = CONVERSATION_DIALOGUE_READY
        self.record.orientation_discussed = orientation_discussed
        self.accepted_proposals = list(response.accepted_proposals)
        self.accepted_disclosures = list(response.disclosures)
        self.third_party_opinions = list(response.third_party_opinions)
        self.leverage_uses = list(response.leverage)
        if not self.social_effects_applied:
            commit_agreements(self)
            commit_information(self)
            self.record.name_forms = name_forms(self.name_actors)
            goal = self.initiator.goal
            if goal and goal.id == self.goal_id and goal.status == "active":
                goal.last_attempt_phase = self.initiator.world.phase_index
            commit_concern_use(self)
            self.social_effects_applied = True
        for npc, partner, entry in opinions:
            npc.remember_opinion(partner.id, entry.opinion, entry.reason)
        from information import caught_lies, opinion_change
        people = {"initiator": self.initiator, "recipient": self.recipient}
        for liar_role, listener_role, truth in caught_lies(self.information_context, response):
            # Being lied to about something one saw firsthand sours the view of the liar, on top of the exchange itself.
            liar, listener = people[liar_role], people[listener_role]
            if listener is self.initiator.world.snake:
                continue
            before = listener.opinion_score(liar.id)
            after = max(-100, before + opinion_change("worse"))
            reason = f"Caught {liar.name} denying what I saw myself: {truth}"[:400]
            listener.remember_opinion(liar.id, after, reason)
            self.record.outcomes.append(describe_opinion_change(listener.name, liar.name, before, after, reason))
        self.reactions = [
            (entry.reason, max(-4.0, min(4.0, entry.opinion / 20)))
            for entry in (response.initiator_opinion, response.recipient_opinion)
        ]
        if self.purpose == "food_request":
            self.record_food_request()
        finish_generated_dialogue(self, reading=reading)

    @property
    def accepts_response(self):
        world = self.initiator.world
        return (self.active and self.dialogue.dialogue_state in ("waiting", "pending")
                and self.initiator.alive and self.recipient.alive
                and world.actor_by_id(self.initiator.id) is self.initiator
                and world.actor_by_id(self.recipient.id) is self.recipient
                and self.initiator.location == self.recipient.location)

    def fail_dialogue(self, reason, *, failed_turns=()):
        if not self.accepts_response:
            return
        record_failed_dialogue(self, failed_turns)
        self.dialogue.dialogue_state = "failed"
        self.record.error = reason
        self.finish(self.initiator.world.tree)

    def prepare_food_request(self):
        from autonomy import donor_choice

        self.food_outcome, self.food_reason = donor_choice(self.recipient, self.initiator)

    def record_food_request(self):
        requester, donor = self.initiator, self.recipient
        world = requester.world
        world.publish("food_requested", (requester, donor),
                      f"{requester.name} asked {donor.name} for an apple.", self.record)
        if self.food_outcome == "accepted":
            self.record.outcomes.insert(0, "Agreed to share one apple; the handoff has not happened yet.")
            return
        deliberate = self.food_outcome in ("need", "refused")
        text = (f"{donor.name} kept their last apple because they wanted to eat it."
                if self.food_outcome == "need" else f"{donor.name} declined {requester.name}'s apple request.")
        if not deliberate:
            text = f"{donor.name} could no longer share with {requester.name}: the inventories changed."
        event = world.publish("food_refused" if deliberate else "interaction_failed", (requester, donor), text, self.record)
        self.outcome_event_id = event.id
        self.record.outcomes.insert(0, self.food_reason)
        if deliberate:
            remember(requester, donor, event, "refused", f"{donor.name} declined my apple request: {self.food_reason}",
                     -1 if self.food_outcome == "need" else -5)
            remember(donor, requester, event, "kept_fruit", f"I kept my apple when {requester.name} asked: {self.food_reason}", 0)

    def update(self, dt, *, reading=False):
        if not self.active:
            return
        if self.initiator.location != self.recipient.location:
            self.cancel("The participants are no longer in the same location.")
            return
        if self.arrived:
            from interactions import observe_conversation
            observe_conversation(self.initiator.world, self.record, self.initiator, self.recipient)
            face_each_other(self.initiator, self.recipient)
            if self.dialogue.dialogue_state in ("waiting", "pending"):
                self.dialogue.wait_elapsed += dt
                if self.dialogue.wait_elapsed >= DIALOGUE_TIMEOUT_SECONDS:
                    self.fail_dialogue("Dialogue generation timed out. No dialogue was generated.")
            elif dialogue_finished(self, dt, reading=reading):
                if self.escalation and self.escalation.action == "attack":
                    self.initiator.world.commit_killing(self)
                    if not self.active:
                        return
                if self.accepted_gift or self.purpose == "food_request" and self.food_outcome == "accepted":
                    self.handoff(self.accepted_gift)
                else:
                    self.finish(self.initiator.world.tree)
            return
        self.arrived = True
        for npc in (self.initiator, self.recipient):
            if npc.destination is not None:
                if npc._move(max(0.0, dt)):
                    npc.destination = None
                else:
                    self.arrived = False
        if self.arrived:
            self.record.status = CONVERSATION_WAITING_FOR_DIALOGUE
            from interactions import observe_conversation
            observe_conversation(self.initiator.world, self.record, self.initiator, self.recipient)
            face_each_other(self.initiator, self.recipient)
            if self.purpose == "food_request":
                self.prepare_food_request()

    def snapshot(self, tree, npcs):
        from autonomy import arrival_greeter
        if self.arrived and self.active:
            from interactions import observe_conversation
            observe_conversation(self.initiator.world, self.record, self.initiator, self.recipient)
        self.request_roster_revision = self.initiator.world.roster_revision
        self.request_actor_revisions = {n.id: n.command_revision for n in (self.initiator, self.recipient)}
        def participant(npc):
            state = npc.snapshot(tree, self.other(npc).id)
            # These replies were addressed to the player before the NPCs met.
            state.pop("last_reply", None)
            return state

        if self.dialogue.dialogue_state == "waiting":
            self.proposals = permitted_proposals(self)
            self.context_references = conversation_references(self)
            self.information_context = prepare_information(self)
            self.name_actors = actor_catalog(self.initiator.world, self.initiator.location)
        views = information_views(self.information_context, self.initiator.world)
        initiator, recipient = participant(self.initiator), participant(self.recipient)
        for role, state in (("initiator", initiator), ("recipient", recipient)):
            state["private_context"].pop("known_accounts", None)
            state["private_context"]["permitted_accounts"] = views.get(role, [])
            concern = self.concerns[role]
            state["private_context"]["concern"] = ({"kind": concern.kind, "claim_id": concern.claim_id,
                "guidance": concern_topic(self.initiator if role == "initiator" else self.recipient, concern)}
                if concern else None)
        initiator["private_context"]["discussion_focus_claim_id"] = self.focus_claim_id
        initiator["private_context"]["immediate_objective"] = self.topic
        recipient["private_context"]["immediate_objective"] = "Respond to what is actually said using only my own knowledge."
        # The live check must judge speech against the same knowledge the speakers were given.
        self.request_participants = deepcopy((initiator, recipient))
        return {
            "request_kind": "conversation",
            "_name_actors": self.name_actors,
            "interaction_id": self.id,
            "origin": self.origin,
            "purpose": self.purpose,
            "opening_kind": self.opening_kind,
            "arrival_focus": self.opening_kind == "orientation",
            "arrival_welcome": (self.opening_kind == "orientation" and self.initiator is arrival_greeter(self.initiator.world)
                                and self.recipient.arrival_day > STARTING_DAY),
            "escalation_choices": self.initiator.world.escalation_choices(self),
            "food_request": {"item": "fruit", "quantity": 1, "outcome": self.food_outcome,
                             "reason": self.food_reason, "transferred": False} if self.purpose == "food_request" else None,
            "initiator": initiator,
            "recipient": recipient,
            "shared_history": shared_conversation_history(self),
            "_information_validation": asdict(self.information_context) if self.information_context else None,
            "topic": "The initiator will introduce their own concern or account; it is not shared knowledge yet.",
            "permitted_proposals": [dict(
                asdict(p), item="fruit", quantity=1, deadline_label=phase_label(p.deadline),
                giver_name=self.initiator.world.npc_by_id(p.promisor_id).name,
                receiver_name=self.initiator.world.npc_by_id(p.recipient_id).name,
                offer_speaker="initiator" if p.promisor_id == self.initiator.id else "recipient",
                acceptance_speaker="initiator" if p.recipient_id == self.initiator.id else "recipient",
            ) for p in self.proposals],
        }

    def finish(self, tree):
        if not self.active:
            return
        self.record.pending_witness_ids.clear()
        self.active = False
        self.record.status = CONVERSATION_FINISHED
        if not self.turns:
            self.record.status = CONVERSATION_FAILED if self.error else CONVERSATION_ENDED_BEFORE_DIALOGUE
            self.record.outcomes = [self.error or "The conversation ended before any dialogue was generated."]
        if self.purpose != "food_request":
            world = self.initiator.world
            text = (f"{self.initiator.name} and {self.recipient.name} talked about {self.topic}."
                    if self.turns else f"{self.initiator.name} and {self.recipient.name}'s meeting ended without generated dialogue.")
            event = world.publish("conversation_completed" if self.turns else "interaction_failed",
                                  (self.initiator, self.recipient), text, self.record)
            self.outcome_event_id = event.id
            if self.dialogue.dialogue_state == "ready":
                for npc, reaction in zip((self.initiator, self.recipient), self.reactions):
                    reason, weight = reaction
                    remember(npc, self.other(npc), event, "conversation", f"Talked about {self.topic}. {reason}", weight)
        elif not self.turns or self.food_outcome in ("", "accepted"):
            if not self.error:
                self.record.status = CONVERSATION_ENDED_BEFORE_SHARING
                self.record.outcomes = ["The food request ended before an apple was shared."]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient),
                                                self.record.outcomes[0], self.record)
            self.outcome_event_id = event.id
        self.release_participants(self.social_effects_applied)

    def cancel(self, reason):
        """Interrupt the meeting without replaying accepted speech or effects."""
        if not self.active:
            return
        self.escalation = None
        self.record.pending_witness_ids.clear()
        self.active = False
        self.record.status, self.record.error = CONVERSATION_CANCELLED, reason
        self.record.outcomes.append(reason)
        self.release_participants(bool(self.record.turns), revise=True)

    def release_participants(self, completed, *, revise=False):
        """Restore each suspended activity only while this meeting still owns it."""
        for npc, state in self.resident_activities():
            if not release_resident(npc.world, "conversation", self, npc, state, self.other(npc).id):
                continue
            npc.pause_after_conversation(completed)
            if not completed:
                npc.cooldowns[self.other(npc).id] = dialogue_failure_cooldown(npc, self.other(npc))
            if revise:
                npc.command_revision += 1

    def handoff(self, gift=None):
        """Reverse the roles while preserving suspended activity progress."""
        if not self.active:
            return
        intent = gift or TransferIntent(self.recipient.id, self.initiator.id, "fruit", 1)
        donor = self.initiator if intent.giver_id == self.initiator.id else self.recipient
        requester = self.other(donor)
        states = self.saved_states if donor is self.initiator else self.saved_states[::-1]
        self.active = False
        donor.world.reservations.release("conversation", self, requester, donor)
        Handoff(donor, requester, intent.item, intent.quantity, self.record,
             saved_states=states, already_met=True,
             thanks=self.handoff_thanks, rewritten_thanks=self.rewritten_handoff_thanks)


def validate_saved_conversations(ctx):
    state = ctx.state
    conversations, history, facts, claims, creations, reserved, tree_id = (
        state.conversations, ctx.history, ctx.facts, ctx.claims, state.creations, ctx.reserved, state.tree.id)
    live, live_ids = {**ctx.live, state.snake.id: state.snake}, ctx.live_ids | {state.snake.id}
    from constants import CONVERSATION_MAX_TURNS
    from activity import validate_activity_target
    from information import validate_information
    from social_saves import validate_saved_concern
    from social import validate_agreements
    from types import SimpleNamespace
    for interaction in conversations:
        pair = (interaction.initiator, interaction.recipient)
        if pair[0] == pair[1] or any(i not in live or i in reserved for i in pair) or interaction.record not in history:
            raise ValueError("Invalid live interaction participants.")
        for index, activity in zip(pair, interaction.saved_states):
            if live[pair[0]].location != live[pair[1]].location:
                raise ValueError("Conversation participants must share a location.")
            validate_activity_target(activity, live[index], tree_id, live_ids)
            if activity.destination is not None:
                from locations import validate_destination
                validate_destination(activity.destination, location=live[index].location)
        record = history[interaction.record]
        if set(record.participant_ids) != set(pair):
            raise ValueError("An interaction must match its record.")
        reserved.update(pair)
        thanks = interaction.handoff_thanks
        rewritten = interaction.rewritten_handoff_thanks
        if rewritten and (len(rewritten) != len(thanks) or any(not text.strip() for text in rewritten)):
            raise ValueError("Rewritten thanks must match the original parts.")
        if set(interaction.concerns) != {"initiator", "recipient"}:
            raise ValueError("A meeting must retain both frozen concerns.")
        for role, index in zip(("initiator", "recipient"), pair):
            validate_saved_concern(interaction.concerns[role], live[index], claims, facts)
        if interaction.goal_id and (live[pair[0]].goal is None or interaction.goal_id != live[pair[0]].goal.id):
            raise ValueError("A meeting goal must belong to its initiator.")
        if interaction.focus_claim_id and not any(b.claim_id == interaction.focus_claim_id for b in live[pair[0]].beliefs):
            raise ValueError("A required meeting account must be known to its initiator.")
        if len(record.turns) > CONVERSATION_MAX_TURNS:
            raise ValueError("A resident conversation exceeds its accepted turn budget.")
        ready = interaction.dialogue_state == "ready"
        statuses = (CONVERSATION_DIALOGUE_READY,) if ready else (CONVERSATION_WALKING_TO_MEET, CONVERSATION_WAITING_FOR_DIALOGUE)
        if record.status not in statuses:
            raise ValueError("A completed, failed or cancelled record cannot restore an active conversation.")
        if record.id != interaction.id or ready != bool(record.generated_turns()) or ready != interaction.social_effects_applied:
            raise ValueError("Invalid conversation commit progress.")
        if interaction.information_context:
            context = interaction.information_context
            if context.participants != {"initiator": live[pair[0]].id, "recipient": live[pair[1]].id}:
                raise ValueError("Invalid information speakers.")
            if context.focus_claim_id != interaction.focus_claim_id:
                raise ValueError("The required meeting account must match its information snapshot.")
            if any(o.fact_id not in facts or o.quantity != facts[o.fact_id].quantity
                   or o.kind != facts[o.fact_id].kind or o.subject_id != facts[o.fact_id].subject_id for o in context.options):
                raise ValueError("Missing snapshot facts.")
        validate_agreements(interaction.accepted_proposals, interaction.proposals, record.generated_turns(),
                            live[pair[0]].id, live[pair[1]].id, current_phase=ctx.now)
        if ready:
            try:
                validate_information(SimpleNamespace(turns=record.generated_turns(), disclosures=interaction.accepted_disclosures,
                    third_party_opinions=interaction.third_party_opinions, leverage=interaction.leverage_uses),
                    interaction.information_context)
            except (KeyError, TypeError, AttributeError, IndexError) as exc:
                raise ValueError("Invalid information snapshot references.") from exc
        if interaction.accepted_gift:
            from transfers import validate_resident_gift
            validate_resident_gift(interaction.accepted_gift, record.generated_turns(), {
                "purpose": interaction.purpose, "arrival_focus": interaction.opening_kind == "orientation",
                **{
                role: {"actor_id": live[index].id, "inventory": live[index].inventory,
                       "owned_works": [{"id": w.id, "title": w.title} for w in creations
                                       if w.owner_id == live[index].id]}
                for role, index in zip(("initiator", "recipient"), pair)}})
            if interaction.accepted_gift.accept_turn is None:
                raise ValueError("A pending gift handoff needs an accepted offer.")
