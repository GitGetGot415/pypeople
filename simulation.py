"""A small world with player-controlled day phases."""


import math
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field
from uuid import uuid4
from constants import (COMPANY_SOCIAL_RECOVERY, DECISION_SECONDS, MAX_SIMULATION_STEP_SECONDS,
    PROTECTED_POPULATION, SATISFACTION_PHASE_LOSS, SOCIAL_RADIUS, STARTING_DAY, STARTING_PAUSED,
    STARTING_PHASE, COMPANY_FATIGUE_PER_SECOND, CONVERSATION_CANCELLED,
    CONVERSATION_PENDING_REPLY_STATUSES)


@dataclass
class PendingAction:
    npc_id: str
    action: Literal["GO_TO_TREE", "TALK_TO_NPC", "GIVE_ITEM", "EAT"]
    revision: Annotated[int, Field(ge=0)]
    target_id: str | None = None
    topic: str = ""
    record: str | None = None
    item: str = ""
    quantity: Annotated[int, Field(ge=0)] = 0


class Simulation:
    def action_for(self, actor_id):
        """The queue is the sole authority for accepted, deferred commands."""
        return next((action for action in self.pending_actions if action.npc_id == actor_id), None)

    def __init__(self, npcs, tree=None, *, assign_goals=True, npc_trait_mode="normal", mode="player", player_start=False, initialize_opinions=True):
        from npc import Tree
        from snake import Snake

        from interactions import ParticipantReservations
        self.reservations = ParticipantReservations()
        self.npcs = npcs
        self.npc_trait_mode = npc_trait_mode
        self.snake = Snake()
        self.snake.world = self
        self.archived = []
        self.creations = []
        from player import Player
        self.player = Player()
        self.player.world = self
        self.mode = mode
        from locations import npc_capacity
        if len(npcs) > npc_capacity(self):
            raise ValueError("Too many starting NPCs for the apartment building.")
        self.tree = tree if tree is not None else Tree(fruit=self.population)
        self.player_interaction = None
        self.player_invitations = []
        self.player_claims = []
        self.visit = None
        self.bodies = []
        self.player_menu_blocked = False
        self.player_call_check = 0.0
        self.player_acknowledgments = []
        from apartments import assign_rooms
        assign_rooms(self)
        self.arrival_error = ""
        self.pending_arrival_days = []
        self.processed_arrival_days = []
        self.introductions = []
        self.roster_revision = 0
        self.day = STARTING_DAY
        self.phase = STARTING_PHASE
        self.paused = STARTING_PAUSED
        self.phase_pending = False
        self.sleep_until_day = False
        self.pending_actions = []
        self.history = []
        self.events = []
        self.promises = []
        self.facts = []
        self.claims = []
        self.disclosures = []
        self.social_rotation = 0
        for npc in self.npcs:
            npc.world = self
        if initialize_opinions:
            for npc in self.npcs:
                npc.bind_authored_opinions()
        if player_start and mode == "player":
            from player_story import prepare_player_start
            prepare_player_start(self)
        if assign_goals:
            from purgatory import remember_description
            from social import assign_goal
            for npc in self.npcs:
                remember_description(npc)
            for npc in self.npcs:
                assign_goal(npc, self)
        from social import refresh_concern
        for npc in self.npcs:
            if npc.current_concern is None:
                refresh_concern(npc)
                npc.decision_remaining = 0.0
        if player_start and mode == "player":
            from player_story import start_player_introduction
            start_player_introduction(self)

    @property
    def player_interaction(self):
        return self.reservations.get(self.player.id, "player")

    @player_interaction.setter
    def player_interaction(self, interaction):
        previous = self.player_interaction
        if previous is not None:
            self.reservations.release("player", previous)
        if interaction is not None:
            partner = self.participant_by_id(interaction.npc_id)
            self.reservations.acquire("player", interaction, self.player, partner)

    @property
    def visit(self):
        return self.reservations.get(self.player.id, "visit")

    @visit.setter
    def visit(self, visit):
        previous = self.visit
        if previous is not None:
            self.reservations.release("visit", previous)
        if visit is not None:
            self.reservations.acquire("visit", visit, self.player, self.npc_by_id(visit.host_id))

    def record_by_id(self, record_id):
        return next((record for record in self.history if record.id == record_id), None)

    @property
    def phase_index(self):
        from social_records import phase_index
        return phase_index(self.day, self.phase)

    @property
    def last_killing_phase(self):
        from social_records import phase_index
        return max((phase_index(n.departure["day"], n.departure["phase"])
                    for n in self.archived), default=None)

    @property
    def phase_label(self):
        return f"Day {self.day} / {self.phase.title()}"

    @property
    def next_phase_label(self):
        return "Night" if self.phase == "day" else f"Day {self.day + 1}"

    @property
    def player_attention_pending(self):
        return bool(self.player_interaction and self.player_interaction.stage == "conversation")

    def next_phase(self):
        if self.paused or self.phase_pending or self.player_attention_pending:
            return
        self.phase_pending = True
        self.finish_phase_change()

    def next_day(self):
        if self.paused or self.phase_pending or self.player_attention_pending:
            return
        self.sleep_until_day = True
        self.next_phase()

    def finish_phase_change(self):
        """Commit one clock request only after existing speech and handoffs finish."""
        from social import advance_social_phase
        if (not self.phase_pending or self.paused or self.player_attention_pending or self.snake.conversation is not None
                or self.snake.interaction_blocked or self.pending_actions
                or any(npc.interaction is not None or npc.interaction_blocked
                       or npc.selected and npc.player_reply.closing for npc in self.npcs)):
            return
        self.phase_pending = False
        if self.phase == "day":
            self.phase = "night"
        else:
            self.phase = "day"
            self.day += 1
            from locations import npc_capacity
            if (len(self.npcs) < npc_capacity(self) and self.day not in self.pending_arrival_days
                    and self.day not in self.processed_arrival_days):
                self.pending_arrival_days.append(self.day)
        for npc in self.npcs:
            npc.social = max(0.0, npc.social - SATISFACTION_PHASE_LOSS)
        self.tree.fruit = self.population
        from apartments import phase_changed
        phase_changed(self)
        advance_social_phase(self)
        self.publish("phase", [], f"Time changed to {self.phase}.")
        if self.sleep_until_day and self.phase == "night":
            self.phase_pending = True
            self.finish_phase_change()
            return
        self.sleep_until_day = False
        if self.phase == "day":
            self.deliver_arrivals()

    def publish(self, kind, participants, text, record=None):
        from conversation_history import WorldEvent

        reference = record.id if record is not None else None
        event = WorldEvent(self.day, self.phase, kind, [npc.id for npc in participants], text, reference)
        self.events.append(event)
        from information import record_event_fact
        record_event_fact(self, event, record)
        del self.events[:-200]
        return event

    def start_conversation(self, initiator, recipient, topic, *, origin="player", purpose="chat", spacing=0,
                           opening_kind=None, focus_claim_id=None, goal_id=None):
        from npc import Conversation

        from locations import available as available_outdoors, colocated
        from autonomy import arrival_greeter, arrival_greeting_reserved, awaiting_arrival_greeting
        welcome = (initiator is arrival_greeter(self) and awaiting_arrival_greeting(recipient, self)
                   and opening_kind == "orientation")
        if not welcome and any(arrival_greeting_reserved(p, self) for p in (initiator, recipient)):
            raise ValueError("The new arrival must speak with Judy first.")
        if self.phase_pending and origin == "autonomous":
            raise ValueError("Waiting for the pending time change before starting another automatic conversation.")
        if not colocated(initiator, recipient) or any(self.actor_by_id(p.id) is not p or not available_outdoors(p) for p in (initiator, recipient)):
            raise ValueError("Only living residents can begin this conversation.")
        interaction = Conversation(initiator, recipient, topic, self.tree, spacing, origin=origin, purpose=purpose,
                                   opening_kind=opening_kind, focus_claim_id=focus_claim_id, goal_id=goal_id)
        self.history.append(interaction.record)
        return interaction

    def npc_by_id(self, npc_id):
        return next((npc for npc in self.npcs if npc.id == npc_id), None)

    def participant_by_id(self, actor_id):
        """Historical possession identity includes the player after death."""
        return self.player if actor_id == self.player.id else self.actor_by_id(actor_id)

    def owned_works(self, owner_id):
        from player import owned_works
        return owned_works(self, owner_id)

    def cancel_gifts(self, reason):
        for npc in self.npcs:
            conversation = npc.conversation
            if conversation and conversation.initiator is npc and conversation.accepted_gift:
                conversation.accepted_gift = None
                conversation.record.outcomes.append(f"The uncommitted gift handoff was cancelled. {reason}")
        gifts = {id(n.gift): n.gift for n in self.npcs if n.gift}
        for gift in gifts.values():
            gift.finish(reason)
        retained = []
        for action in self.pending_actions:
            if action.action != "GIVE_ITEM":
                retained.append(action)
                continue
            npc = self.npc_by_id(action.npc_id)
            if npc:
                npc.player_reply.complete()
            if action.record is not None:
                record = self.record_by_id(action.record)
                record.status, record.error = CONVERSATION_CANCELLED, reason
                record.outcomes.append(reason)
        self.pending_actions = retained

    @property
    def outdoor_npcs(self):
        return [npc for npc in self.npcs if not npc.indoors]

    def is_outdoors(self, npc):
        return ((self.npc_by_id(npc.id) is npc or npc is self.player and self.mode == "player" and npc.alive)
                and npc.location == "garden")

    def commit_creation(self, npc):
        from creations import Creation
        job = npc.creative_job
        if (not job or job.accepted is None or job.remaining > 0 or not self.is_creative_job(npc, job)):
            return
        if not any(work.job_id == job.id for work in self.creations):
            self.creations.append(Creation(id=uuid4().hex, job_id=job.id, creator_id=npc.id, owner_id=npc.id, kind=job.kind,
                day=self.day, phase=self.phase, **job.accepted))
        npc.creative_job = None
        from activity import Behavior
        npc.behavior = Behavior.REST
        npc.reason = "Finished creating; relaxing in my apartment"

    def is_creative_job(self, npc, job):
        from activity import Behavior
        behavior = npc.behavior
        if self.visit and self.visit.host_id == npc.id:
            behavior = self.visit.saved_state.behavior
        elif self.player_interaction and self.player_interaction.npc_id == npc.id:
            if self.player_interaction.saved_state:
                behavior = self.player_interaction.saved_state.behavior
        elif npc.conversation:
            from interactions import suspended_activity
            behavior = suspended_activity(npc.conversation, npc).behavior
        return (self.npc_by_id(npc.id) is npc and npc.creative_job is job
                and behavior == (Behavior.PAINT if job.kind == "art" else Behavior.COMPOSE)
                and npc.command_revision == job.revision and job.creator_id == npc.id)

    def advance(self, dt, *, active=True, action_handler=None, viewed_conversation=None):
        if not active or self.paused:
            return
        if action_handler is not None:
            for action in tuple(self.pending_actions):
                if action not in self.pending_actions:
                    continue
                npc = self.npc_by_id(action.npc_id)
                if npc is not None and (npc.unread_player_reply or npc.waiting_for_done):
                    # Keep the closing reply visible until the player dismisses it.
                    continue
                self.pending_actions.remove(action)
                action_handler(action)
        # Ignore long window stalls; there is no elapsed world time to catch up.
        self.step(min(MAX_SIMULATION_STEP_SECONDS, max(0.0, dt)), viewed_conversation=viewed_conversation)

    def company_available(self, npc, *, allow_player_call=False, allow_arrival=False):
        from activity import Behavior

        from locations import available as available_outdoors
        from autonomy import arrival_greeting_reserved
        return (self.npc_by_id(npc.id) is npc and available_outdoors(npc)
                and (allow_arrival or not arrival_greeting_reserved(npc, self))
                and not self.reservations.held(npc.id, "player", "visit", "conversation", "handoff")
                and not npc.selected and not npc.interaction_blocked
                and (allow_player_call or not npc.player_call)
                and npc.owner == "autonomous" and not npc.resume_pending
                and not any(npc.id in (action.npc_id, action.target_id) for action in self.pending_actions)
                and npc.behavior in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY))

    def step(self, dt, *, viewed_conversation=None):
        from autonomy import greet_arrivals, reconsider, social_pair
        from social import refresh_concern
        from activity import Behavior
        from interactions import observe_active_conversations

        observe_active_conversations(self)
        from player import update_interaction
        from apartments import update_visit
        from locations import update_journey
        update_interaction(self, dt)
        if self.player.journey:
            update_journey(self.player, dt)
        update_visit(self, dt)
        from constants import BODY_SECONDS
        self.tree.shake_remaining = max(0.0, self.tree.shake_remaining - dt)
        for body in self.bodies:
            body["elapsed"] += dt
        self.bodies[:] = [body for body in self.bodies if body["elapsed"] < BODY_SECONDS]
        self.deliver_arrivals()
        for npc in self.npcs:
            npc.update_cooldowns(dt)
            if npc.pending_social_opinions and npc.decision_remaining == 0:
                from social import apply_opinion_effects
                if not apply_opinion_effects(npc):
                    npc.decision_remaining = DECISION_SECONDS
        # Finish interactions (including atomic request-to-gift handoffs) before decisions.
        interactions = {id(npc.conversation or npc.gift): npc.conversation or npc.gift
                        for npc in self.npcs if npc.conversation or npc.gift}
        for interaction in interactions.values():
            if interaction is viewed_conversation:
                interaction.update(dt, reading=True)
            else:
                interaction.update(dt)
        greet_arrivals(self)
        from apartments import home_priority
        for npc in self.npcs:
            home_priority(npc, self, dt)
        self.snake.update(dt)
        # Check necessary work for every actor before an idle initiator can reserve them.
        for npc in self.npcs:
            if npc.resume_pending and npc.interaction is None and not npc.player_call:
                npc.resume_activity(self)
            reconsider(npc, self, pressing_only=True)
        due = {npc.id for npc in self.npcs if npc.decision_remaining == 0 and self.company_available(npc)}
        for npc in self.npcs:
            if npc.id in due:
                refresh_concern(npc)
        while due:
            choice = social_pair(self, due=due)
            if choice is None:
                break
            initiator, recipient, purpose, topic, reason, opening, focus, goal_id = choice
            self.start_conversation(initiator, recipient, topic, origin="autonomous", purpose=purpose,
                                    opening_kind=opening, focus_claim_id=focus, goal_id=goal_id)
            initiator.reason = reason
            recipient.reason = (f"Answering {initiator.name}'s request for an apple" if purpose == "food_request"
                                else f"Chatting with {initiator.name}")
            due.difference_update((initiator.id, recipient.id))
            self.social_rotation = (self.npcs.index(initiator) + 1) % len(self.npcs)
        for npc in self.npcs:
            reconsider(npc, self)
            npc.update(dt, self.tree)
        observe_active_conversations(self)
        # Each person benefits once, regardless of how many neighbors are present.
        socializing = []
        for npc in self.npcs:
            if not self.company_available(npc):
                continue
            if any(other is not npc and self.company_available(other)
                   and npc.location == other.location
                   and (npc.behavior == Behavior.COMPANY or other.behavior == Behavior.COMPANY)
                   and math.hypot(npc.x - other.x, npc.y - other.y) <= SOCIAL_RADIUS
                   for other in self.npcs):
                socializing.append(npc)
        for npc in socializing:
            npc.social = min(100.0, npc.social + COMPANY_SOCIAL_RECOVERY * dt)
            npc.social_fatigue = min(100.0, npc.social_fatigue + COMPANY_FATIGUE_PER_SECOND * dt)
        # Keep the pending flag through this tick's decision pass, so automatic
        # work cannot reserve a fresh partner before the requested phase applies.
        # Dawn arrivals then run before pairing on the following tick as usual.
        self.finish_phase_change()
        from player import update_calls
        update_calls(self, dt)

    @property
    def population(self):
        """Living human residents, including the active player and excluding the snake."""
        from player import active_player
        return len(self.npcs) + int(active_player(self))

    @property
    def departed_count(self):
        return len(self.archived) + int(not self.player.alive)

    @property
    def residents(self):
        return [*self.npcs, *self.archived]

    @property
    def actors(self):
        return [*self.residents, self.snake]

    def actor_by_id(self, actor_id):
        if actor_id == self.player.id and self.mode == "player" and self.player.alive:
            return self.player
        return next((actor for actor in self.actors if actor.id == actor_id), None)

    @property
    def available_fruit(self):
        return self.tree.fruit

    def harvest(self, actor, quantity=1):
        """The only stock-to-inventory commit, shared by residents and player."""
        from constants import HARVEST_DISTANCE, TREE_SHAKE_SECONDS
        from player import active_player, can_claim
        from locations import available
        human = self.npc_by_id(actor.id) is actor and not actor.departure
        player = actor is self.player and active_player(self)
        if (not (human or player) or actor.location != "garden" or actor.journey
                or math.dist((actor.x, actor.y), self.tree.interaction_point) > HARVEST_DISTANCE
                or human and (not available(actor) or actor.busy or actor.interaction_blocked or actor.player_call
                              or actor.resume_pending or actor.creative_job
                              or self.reservations.held(actor.id, "player", "visit"))
                or player and not can_claim(self)):
            raise ValueError("Harvesting requires a living, available collector at the tree.")
        if type(quantity) is not int or not 1 <= quantity <= self.tree.fruit:
            raise ValueError(f"Choose a positive whole amount up to the {self.tree.fruit} apples remaining.")
        self.tree.fruit -= quantity
        actor.inventory["fruit"] = actor.inventory.get("fruit", 0) + quantity
        self.tree.shake_remaining = TREE_SHAKE_SECONDS
        from information import add_fact, harvest_summary
        from locations import observers
        text = harvest_summary(actor.name, quantity)
        event = self.publish("harvest", (actor,), text)
        witnesses = observers(self, actor)
        witness_ids = {p.id for p in witnesses if p is not None and p is not self.player}
        # Repeat pickups seen by exactly the same witnesses are one account of the
        # whole collection, so retelling it reports the total rather than one apple.
        fact = next((f for f in self.facts if f.kind == "harvest" and f.subject_id == actor.id
                     and f.phase == self.phase_index and set(f.witness_ids) == witness_ids), None)
        if fact:
            fact.quantity += quantity
            fact.summary = harvest_summary(actor.name, fact.quantity)
        else:
            add_fact(self, "harvest", actor, None, event.id, None, text, witnesses,
                     quantity=quantity, location="garden")
        self.harvest_impressions(actor, quantity, witnesses)
        return event

    def harvest_impressions(self, actor, quantity, witnesses):
        """Each resident watching minds every apple past a fair share that they saw this collector take this phase."""
        from constants import HARVEST_FREE_APPLES
        from player import update_player_opinion
        for witness in witnesses:
            if witness is actor or self.npc_by_id(witness.id) is not witness:
                continue  # The python and the collector keep their own views.
            seen = sum(fact.quantity for fact in self.facts if fact.kind == "harvest" and fact.subject_id == actor.id
                       and fact.phase == self.phase_index and witness.id in fact.witness_ids)
            # Apples numbered past the free share each cost one more point: the 2nd costs 1, the 3rd 2, and so on.
            delta = -sum(max(0, count - HARVEST_FREE_APPLES) for count in range(seen - quantity + 1, seen + 1))
            if not delta:
                continue
            text = f"Saw {actor.name} take {seen} apples from the tree, more than a fair share."
            if actor is self.player:
                update_player_opinion(witness, delta, text, scaled=False)
                continue
            previous = witness.opinions.get(actor.id, {"opinion": 0, "reason": ""})
            earlier = previous["reason"]
            if earlier.startswith(f"Saw {actor.name} take "):
                earlier = earlier.partition(" Earlier: ")[2]  # A later pickup restates the running total.
            witness.remember_opinion(actor.id, max(-100, min(100, previous["opinion"] + delta)),
                                     (text + (f" Earlier: {earlier}" if earlier else ""))[:400])

    def killing_unavailable(self, attacker, victim):
        """Shared physical restrictions and one world-wide allowance per phase."""
        from constants import (GARDEN_LOCATION, KILLING_COOLDOWN_PHASES, KILLING_DISTANCE,
                               KILLING_INDOOR_REASON, SNAKE_NAME)
        from player import active_player
        if victim is self.snake:
            return f"{SNAKE_NAME} cannot be killed."
        if (self.npc_by_id(victim.id) is not victim or victim.departure or attacker is victim
                or not (self.npc_by_id(attacker.id) is attacker
                        or attacker is self.player and active_player(self))):
            return "Choose a living resident."
        if attacker.location != GARDEN_LOCATION or victim.location != GARDEN_LOCATION:
            return KILLING_INDOOR_REASON
        if self.population <= PROTECTED_POPULATION:
            return "The remaining population is protected."
        if (self.last_killing_phase is not None
                and self.phase_index - self.last_killing_phase < KILLING_COOLDOWN_PHASES):
            return "Someone has already been killed this phase. Wait for the next phase."
        if math.dist((attacker.x, attacker.y), (victim.x, victim.y)) > KILLING_DISTANCE:
            return "Move closer to the resident first."
        return ""

    def player_killing_unavailable(self, victim):
        from player import active_player
        if not active_player(self):
            return "The player is unavailable."
        if self.paused:
            return "Resume the simulation first."
        if (self.player.journey or self.visit and self.visit.stage == "travel"
                or self.player_interaction and (self.player_interaction.npc_id != victim.id
                                               or self.player_interaction.kind == "door")):
            return "Finish the current journey or interaction first."
        return self.killing_unavailable(self.player, victim)

    def commit_player_killing(self, victim):
        from constants import CONVERSATION_COMPLETED, PLAYER_LABEL
        from conversation_history import ConversationRecord
        reason = self.player_killing_unavailable(victim)
        if reason:
            raise ValueError(reason)
        record = ConversationRecord((PLAYER_LABEL, victim.name), f"Killed {victim.name}", CONVERSATION_COMPLETED,
                                    origin="player", purpose="killing", participant_ids=[self.player.id, victim.id])
        self.history.append(record)
        return self._finish_killing(self.player, victim, record)

    def escalation_choices(self, meeting, opinion=None):
        """opinion: the confronter's view to judge eligibility by, such as the one an exchange is about to commit."""
        from constants import CONFRONTATION_OPINION, GRIEVANCE_FACT_KINDS
        from social_records import claim_by_id, fact_by_id
        first, target = meeting.initiator, meeting.recipient
        if (meeting.purpose != "confrontation" or self.npc_by_id(first.id) is not first
                or self.npc_by_id(target.id) is not target
                or not meeting.grievance_id):
            return None
        actions = ["leave"]
        belief = next((b for b in first.beliefs if b.claim_id == meeting.grievance_id), None)
        claim = claim_by_id(self, meeting.grievance_id)
        fact = fact_by_id(self, claim.fact_id)
        # Frozen information proves the grievance was retained before this meeting.
        retained = (meeting.information_context is None or any(
            b.claim_id == meeting.grievance_id and b.stance == "believed"
            for b in meeting.information_context.beliefs["initiator"]))
        if (belief and belief.stance == "believed" and claim.affirmed and fact.subject_id == target.id
                and fact.kind in GRIEVANCE_FACT_KINDS
                and retained and (first.opinion_score(target.id) if opinion is None else opinion) <= CONFRONTATION_OPINION
                and not self.killing_unavailable(first, target)
                and (fact.kind != "killing" or belief.phase < self.phase_index)):
            actions.append("attack")
        return {"target_id": target.id, "grievance_id": meeting.grievance_id, "actions": actions}

    def commit_killing(self, meeting):
        from locations import available
        decision = meeting.escalation
        attacker, victim = meeting.initiator, meeting.recipient
        choices = self.escalation_choices(meeting)
        valid = (meeting.active and meeting.arrived and meeting.dialogue.dialogue_state == "ready"
                 and meeting.social_effects_applied and decision is not None and decision.action == "attack"
                 and choices and "attack" in choices["actions"]
                 and decision.target_id == victim.id and decision.grievance_id == meeting.grievance_id
                 and bool(decision.evidence.strip()) and any(t.speaker == "recipient" and decision.evidence in t.text for t in meeting.turns)
                 and all(self.reservations.get(n.id, "conversation") is meeting
                         and meeting.accepted_revisions.get(n.id) == n.command_revision
                         and not n.departure and available(n) and not n.interaction_blocked
                         and not n.player_call for n in (attacker, victim)))
        meeting.escalation = None
        if not valid:
            meeting.record.outcomes.append("The accepted attack was cancelled because eligibility or ownership changed.")
            return False
        return self._finish_killing(attacker, victim, meeting.record, meeting=meeting)

    def _finish_killing(self, attacker, victim, record, *, meeting=None):
        """Commit one authorised death, preserving the same witnesses and consequences for either killer."""
        from constants import (KILLING_WITNESS_PENALTY, KILLING_RELIEF_OPINION, KILLING_RELIEF_CHANCE,
                               KILLING_WITNESS_MAX_PENALTY, KILLING_WITNESS_SOCIAL_RANGE,
                               KILLING_FRIENDSHIP_DIVISOR, KILLING_RELIEF_DELTA)
        from locations import observers
        from social_records import claim_by_id
        from information import add_fact
        witnesses = observers(self, attacker, victim)
        text = f"{attacker.name} killed {victim.name}."
        event = self.publish("killing", (attacker, victim), text, record)
        fact = add_fact(self, "killing", attacker, victim, event.id, record, text, witnesses,
                        location=victim.location)
        if meeting:
            meeting.outcome_event_id = event.id
        record.outcomes.append(text)
        from conversation_history import remember
        for witness in witnesses:
            remember(witness, victim if witness is attacker else attacker, event, "killing", text, 0)
        for witness in witnesses:
            if witness in (attacker, victim):
                continue
            belief = next(b for b in witness.beliefs if claim_by_id(self, b.claim_id).fact_id == fact.id)
            belief.assessed = True  # One firsthand consequence; reports cannot repeat it.
            friend = witness.opinion_score(victim.id)
            low, high = KILLING_WITNESS_SOCIAL_RANGE
            severity = round(KILLING_WITNESS_PENALTY * (low + witness.preferences["sociability"] * (high - low)))
            delta = -min(KILLING_WITNESS_MAX_PENALTY, severity + max(0, friend) // KILLING_FRIENDSHIP_DIVISOR)
            # A hostile relationship can make relief possible, never universal.
            if friend < KILLING_RELIEF_OPINION and witness.rng.random() < KILLING_RELIEF_CHANCE:
                delta = KILLING_RELIEF_DELTA
            reason = f"Witnessed {text} My relationship with {victim.name} informed my reaction."
            if attacker is self.player:
                from player import update_player_opinion
                update_player_opinion(witness, delta, reason, scaled=False)
            else:
                witness.remember_opinion(attacker.id, max(-100, min(100, witness.opinion_score(attacker.id) + delta)), reason)
        return self.depart_resident(victim, event, record, attacker, meeting=meeting)

    def observed_event(self, kind, subject, target, text, record=None, *, witnesses=None):
        """Explicit visual event, or an explicitly selected private participant account."""
        from information import add_fact
        from conversation_history import remember
        from locations import observers
        if witnesses is None:
            witnesses = observers(self, subject, *([target] if target else []))
        witnesses = list({p.id: p for p in witnesses if p is not self.player}.values())
        event_kind = {"apple_given": "fruit_shared", "work_given": "work_shared"}.get(kind, kind)
        event = self.publish(event_kind, (subject, target) if target else (subject,), text, record)
        fact = add_fact(self, kind, subject, target, event.id, record, text, witnesses) if witnesses else None
        for actor in witnesses:
            if actor is self.snake or kind in ("apple_given", "work_given", "conversation_seen"):
                continue
            partner = target if actor is subject and target else subject
            remember(actor, partner, event, kind, text, 0)
        return event, fact

    def interrupt_resident(self, target, reason, *, mode="cancel"):
        """Coordinate outcome-specific cleanup without replaying committed effects."""
        departing = mode == "departure"
        from activity import Behavior
        from transfers import Handoff
        from player import owns_resident, end_interaction
        if mode not in ("cancel", "departure", "arrival", "approach", "action", "session"):
            raise ValueError("Unknown resident interruption reason.")
        if mode == "action":
            if self.visit and self.visit.host_id == target.id:
                from apartments import finish_visit
                finish_visit(self)
            from apartments import cancel_work
            cancel_work(target)
            target.journey = None
            return
        if mode in ("approach", "session"):
            interaction = target.conversation or target.gift
            if interaction is target.conversation and interaction is not None:
                if mode == "session":
                    interaction.finish(self.tree)
                else:
                    interaction.cancel(reason)
            elif isinstance(interaction, Handoff):
                interaction.finish(reason)
            elif interaction is not None:
                interaction.finish(error=reason if mode == "approach" else "")
            return
        if owns_resident(self, target):
            end_interaction(self, reason)
        if self.visit and self.visit.host_id == target.id:
            from apartments import finish_visit
            finish_visit(self)
        interaction = target.conversation or target.gift
        from player import clear_call
        clear_call(target)
        self.player_acknowledgments = [a for a in self.player_acknowledgments if a.npc_id != target.id]
        if isinstance(interaction, Handoff):
            interaction.finish(reason)
            interaction = None
        if interaction:
            interaction.cancel(reason)
        for record in self.history:
            if record.includes(target) and self.player.id in record.participant_ids and record.status in CONVERSATION_PENDING_REPLY_STATUSES:
                record.status, record.error = CONVERSATION_CANCELLED, reason
                record.outcomes.append(reason)
        if departing:
            target.pending_social_opinions.clear()
        if mode != "arrival":
            # An unrelated interaction can have suspended a task aimed at this resident.
            interactions = {id(p.conversation or p.gift): p.conversation or p.gift
                            for p in self.npcs if p.conversation or p.gift}
            states = [state for interaction in interactions.values()
                      for state in interaction.activity_snapshots()]
            for owner in (self.player_interaction, self.visit):
                if owner is not None and owner.saved_state is not None:
                    states.append(owner.saved_state)
            for state in states:
                state.cooldowns.pop(target.id, None)
                if state.target_id == target.id:
                    state.behavior, state.owner, state.target_id = Behavior.WAIT, "autonomous", None
                    state.destination, state.activity_elapsed, state.resume_pending = None, 0.0, False
                    state.eating_witness_ids = []
                    state.reason, state.decision_remaining = reason, 0.0
        remaining = []
        for action in self.pending_actions:
            if action.npc_id == target.id or mode != "arrival" and action.target_id == target.id:
                owner = self.npc_by_id(action.npc_id)
                if owner:
                    owner.player_reply.complete()
                    owner.command_revision += 1
                if action.record is not None:
                    record = self.record_by_id(action.record)
                    record.status, record.error = CONVERSATION_CANCELLED, reason
                    record.outcomes.append(reason)
            else:
                remaining.append(action)
        self.pending_actions[:] = remaining
        for person in self.npcs:
            if mode != "arrival" and person.target_id == target.id:
                person.finish_activity(self, failed=True)
            if mode != "arrival":
                person.cooldowns.pop(target.id, None)
            if person is target:
                person.command_revision += 1
                person.player_reply.complete()
                person.selected = False
                person.destination = person.target_id = None
                person.behavior = Behavior.WAIT
                person.owner = "autonomous"
                person.activity_elapsed = 0.0
                person.eating_witness_ids = []
                person.resume_pending = False
        if mode in ("arrival", "departure"):
            from apartments import cancel_work
            cancel_work(target)
            target.journey = None
            target.home_reason = target.bedtime_remaining = target.wake_remaining = None
        if mode == "arrival":
            target.player_reply.release()
            target.begin_activity(Behavior.WAIT, self, reason=reason)

    def depart_resident(self, target, event, record, attacker, *, meeting=None):
        """Archive a committed victim and release unfinished work through its owners."""
        from copy import deepcopy
        from social import resolve_goal
        if (self.npc_by_id(target.id) is not target or target.departure or event.kind != "killing"
                or event.history != record.id or record.purpose not in ("confrontation", "killing")
                or not any(f.kind == "killing" and f.source_id == event.id and f.subject_id == attacker.id
                           and f.object_id == target.id for f in self.facts)):
            return False
        target.departure = {"day": self.day, "phase": self.phase,
                            "cause": "killed by player" if attacker is self.player else "killed by resident",
                            "event_id": event.id, "attacker_id": attacker.id, "conversation_id": record.id}
        self.bodies.append({"appearance": deepcopy(target.character), "x": target.x, "y": target.y,
                            "location": target.location, "facing": target.facing, "elapsed": 0.0})
        # The source is accepted and complete; mark death before any release can re-enter.
        if meeting:
            meeting.finish(self.tree)
            meeting.outcome_event_id = event.id
        reason = f"Cancelled because {target.name} died; this is not a deliberate betrayal."
        for promise in self.promises:
            if promise.status == "pending" and target.id in (promise.promisor_id, promise.recipient_id):
                promise.status, promise.resolved_phase, promise.outcome = "cancelled", self.phase_index, reason
                cancellation = self.publish("promise_cancelled", (self.participant_by_id(promise.promisor_id), self.participant_by_id(promise.recipient_id)), reason,
                                     next(r for r in self.history if r.id == promise.conversation_id))
                promise.resolution_event_id = promise.consequence_event_id = cancellation.id
        for person in self.npcs:
            goal = person.goal
            if goal and goal.status == "active" and (person is target or target.id in (goal.target_id, goal.third_npc_id)):
                resolve_goal(person, self, "cancelled", reason, record)
        self.interrupt_resident(target, reason, mode="departure")
        for invitation in self.player_invitations:
            if invitation.speaker_id == target.id and invitation.status == "offered":
                invitation.status = "withdrawn"
        target.home_remaining = 0.0
        self.npcs[:] = [p for p in self.npcs if p is not target]
        self.archived.append(target)
        self.roster_revision += 1
        self.introductions[:] = [i for i in self.introductions if i != target.id]
        self.social_rotation %= max(1, len(self.npcs))
        target.departure["event_id"] = event.id
        return True

    def deliver_arrivals(self):
        from character_data import random_character
        from npc import NPC, assign_npc_names
        from purgatory import remember_description
        from social import assign_goal
        from locations import npc_capacity
        if len(self.npcs) >= npc_capacity(self):
            self.pending_arrival_days.clear()
            self.arrival_error = ""
            return None
        if not self.pending_arrival_days or self.arrival_error:
            return None
        day = self.pending_arrival_days[0]
        if day in self.processed_arrival_days:
            self.pending_arrival_days.pop(0)
            return None
        try:
            character = random_character(trait_mode=self.npc_trait_mode,
                                         used_first_names=[actor.name.split()[0] for actor in self.actors if actor.name.split()])
        except (OSError, ValueError) as exc:
            self.arrival_error = f"Could not create the next arrival: {exc}"
            return None
        from constants import ARRIVAL_LOCATION, ELEVATOR_ARRIVAL
        from activity import Behavior
        npc = NPC(character=character, arrival_day=day, location=ARRIVAL_LOCATION,
                  x=ELEVATOR_ARRIVAL[0], y=ELEVATOR_ARRIVAL[1], behavior=Behavior.WANDER)
        npc.sprite_size = self.npcs[0].sprite_size if self.npcs else (40, 70)
        assign_npc_names([npc], used_names=(actor.name for actor in self.actors))
        npc.world = self
        remember_description(npc)
        self.npcs.append(npc)
        for resident in self.npcs:
            resident.bind_authored_opinions()
        from apartments import assign_rooms
        assign_rooms(self)
        self.processed_arrival_days.append(day)
        self.pending_arrival_days.pop(0)
        self.roster_revision += 1
        assign_goal(npc, self)
        from social import refresh_concern
        refresh_concern(npc)
        npc.decision_remaining = 0.0
        self.observed_event("arrival", npc, None, f"{npc.name} arrived for dawn on day {day}.")
        from autonomy import greet_arrivals
        greet_arrivals(self)
        return npc


def validate_saved_actions(ctx):
    state = ctx.state
    npcs, player, tree, pending_actions = state.npcs, state.player, state.tree, state.pending_actions
    live_ids, reserved, history, creation_ids = ctx.live_ids, ctx.reserved, ctx.history, ctx.creation_ids
    from activity import Behavior, validate_activity_target
    from transfers import transfer_matches
    for index, npc in enumerate(npcs):
        if npc.id not in reserved:
            validate_activity_target(npc, npc, tree.id, live_ids)
        if npc.id in reserved and npc.behavior != Behavior.WAIT:
            raise ValueError("Encounter participants must wait.")
    pending_owners = set()
    for action in pending_actions:
        if action.target_id == player.id and action.action != "GIVE_ITEM":
            raise ValueError("Only a gift can use the player as a queued action recipient.")
        if action.npc_id not in live_ids or action.npc_id in pending_owners or action.target_id is not None and action.target_id not in live_ids | {player.id}:
            raise ValueError("Queued actions need living actors.")
        if action.target_id and action.action in ("TALK_TO_NPC", "GIVE_ITEM"):
            positions = {n.id: n.location for n in npcs}
            positions[player.id] = player.location
            if positions[action.npc_id] != positions[action.target_id]:
                raise ValueError("A queued interaction must share a location.")
        pending_owners.add(action.npc_id)
        if action.record is not None and action.record not in history:
            raise ValueError("Invalid queued history record.")
        if action.action == "GIVE_ITEM":
            if (action.target_id is None or action.quantity < 1
                    or action.item != "fruit" and (action.item not in creation_ids or action.quantity != 1)
                    or action.record is None or not transfer_matches(history[action.record].transfer_intent,
                        action.npc_id, action.target_id, action.item, action.quantity)):
                raise ValueError("A queued gift needs its real item and accepted matching authorization.")
