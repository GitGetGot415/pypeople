"""A small world with player-controlled day phases."""

from activity import ActivitySnapshot

import math
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field
from uuid import uuid4
from constants import (COMPANY_SOCIAL_RECOVERY, CROWD_LIMIT, DECISION_SECONDS, HEARING_RADIUS,
                       MAX_SIMULATION_STEP_SECONDS, PROTECTED_POPULATION, RECOVERY_SECONDS,
                       SATISFACTION_PHASE_LOSS, SOCIAL_RADIUS, SNAKE_NAME, STARTING_DAY, STARTING_PAUSED,
                       STARTING_PHASE, WITNESS_RADIUS, COMPANY_FATIGUE_PER_SECOND, CONVERSATION_CANCELLED,
                       CONVERSATION_EATEN_PLAYER, CONVERSATION_EATEN_RESIDENT,
                       CONVERSATION_PENDING_REPLY_STATUSES)


@dataclass
class PendingAction:
    npc_id: str
    action: Literal["WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "TALK_TO_NPC", "GIVE_ITEM", "EAT"]
    revision: Annotated[int, Field(ge=0)]
    target_id: str | None = None
    topic: str = ""
    record: str | None = None
    item: str = ""
    quantity: Annotated[int, Field(ge=0)] = 0


@dataclass
class TreeCustomer:
    actor_id: str
    purpose: Literal["bargain", "judgment", "introduction"]
    reason: str
    saved_state: ActivitySnapshot | None
    elapsed: float = 0.0


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
        self.tree = tree if tree is not None else Tree(fruit=self.population)
        self.player_interaction = None
        self.player_invitations = []
        self.player_claims = []
        self.visit = None
        self.tree_queue = []
        self.tree_current = None
        self._tree_exit_targets = {}
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
        from social import phase_index
        return phase_index(self.day, self.phase)

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

    def finish_phase_change(self):
        """Commit one clock request only after existing speech and handoffs finish."""
        from social import advance_social_phase
        if (not self.phase_pending or self.paused or self.player_attention_pending or self.snake.encounter is not None
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
            self.snake.eaten_today = 0
            if self.day not in self.pending_arrival_days and self.day not in self.processed_arrival_days:
                self.pending_arrival_days.append(self.day)
        for npc in self.npcs:
            npc.social = max(0.0, npc.social - SATISFACTION_PHASE_LOSS)
        self.tree.fruit = self.population
        from apartments import phase_changed
        phase_changed(self)
        advance_social_phase(self)
        self.publish("phase", [], f"Time changed to {self.phase}.")

    def publish(self, kind, participants, text, record=None):
        from conversation_history import WorldEvent

        reference = record.id if record is not None else None
        event = WorldEvent(self.day, self.phase, kind, [npc.id for npc in participants], text, reference)
        self.events.append(event)
        from social import record_event_fact
        record_event_fact(self, event, record)
        del self.events[:-200]
        return event

    def start_conversation(self, initiator, recipient, topic, *, origin="player", purpose="chat", spacing=0,
                           opening_kind=None, focus_claim_id=None, goal_id=None):
        from npc import Conversation

        from locations import available as available_outdoors, colocated
        if self.phase_pending and origin == "autonomous":
            raise ValueError("Waiting for the pending time change before starting another automatic conversation.")
        if not colocated(initiator, recipient) or any(self.npc_by_id(p.id) is not p or not available_outdoors(p) for p in (initiator, recipient)):
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
        from autonomy import awaiting_arrival_greeting
        return (self.npc_by_id(npc.id) is npc and available_outdoors(npc)
                and (allow_arrival or not awaiting_arrival_greeting(npc, self))
                and not self.reservations.held(npc.id, "player", "tree", "visit", "conversation", "handoff", "snake")
                and not npc.selected and not npc.interaction_blocked
                and (allow_player_call or not npc.player_call)
                and npc.owner == "autonomous" and not npc.resume_pending
                and not any(npc.id in (action.npc_id, action.target_id) for action in self.pending_actions)
                and npc.behavior in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY))

    def step(self, dt, *, viewed_conversation=None):
        from autonomy import greet_arrivals, lone_introduction, reconsider, social_pair
        from social import refresh_concern
        from activity import Behavior

        self.tree.update(dt)
        from player import update_interaction
        from apartments import update_visit
        from locations import update_journey
        update_interaction(self, dt)
        if self.player.journey:
            update_journey(self.player, dt)
        update_visit(self, dt)
        clearing_tree = self.update_tree_queue(dt)
        self.deliver_arrivals()
        encounter = self.snake.encounter
        if encounter is not None:
            self.snake.update(dt)
            encounter.update(dt, reading=encounter is viewed_conversation)
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
            if npc.id not in clearing_tree:
                home_priority(npc, self, dt)
        if encounter is None:
            # Free residents establish bedtime intent before a new snake follow-up.
            self.snake.update(dt)
        # Check necessary work for every actor before an idle initiator can reserve them.
        for npc in self.npcs:
            if npc.id in clearing_tree:
                continue
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
            if npc.id in clearing_tree:
                continue
            if npc.id in due:
                lone_introduction(npc, self)
            reconsider(npc, self)
            npc.update(dt, self.tree)
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
        """Living human residents, including the active player and excluding the guard."""
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
        encounter = self.snake.encounter
        return max(0, self.tree.fruit - int(bool(encounter and encounter.active and encounter.reserved)))

    def tree_has_actor(self, actor_id):
        return self.reservations.held(actor_id, "tree")

    def can_leave_tree(self, actor_id):
        return self.tree_has_actor(actor_id) and not (self.snake.encounter and self.snake.encounter.recipient.id == actor_id)

    def tree_can_admit(self, actor):
        from constants import TREE_NPC_ADMISSION_LIMIT
        from player import active_player, owns_resident
        if not self.is_outdoors(actor) or self.tree_has_actor(actor.id):
            return False
        if actor is self.player:
            return active_player(self) and not (self.player_interaction or actor.journey)
        return (len(self.tree_queue) + bool(self.tree_current) < TREE_NPC_ADMISSION_LIMIT
                and not owns_resident(self, actor) and not actor.busy and not actor.interaction_blocked
                and not actor.journey and not actor.player_call)

    def join_tree(self, actor, purpose="bargain", reason=""):
        if not self.tree_can_admit(actor):
            return False
        if purpose == "bargain" and self.available_fruit < 1:
            return False
        from activity import Behavior
        saved = None if actor is self.player else actor.suspend_activity("Waiting in the tree line")
        if saved:
            saved.behavior, saved.target_id, saved.destination, saved.owner = Behavior.WAIT, None, None, "autonomous"
        entry = TreeCustomer(actor_id=actor.id, purpose=purpose, reason=reason, saved_state=saved, elapsed=0.0)
        self.tree_queue.append(entry)
        self.reservations.acquire("tree", entry, actor)
        return True

    def leave_tree(self, actor_id, reason="Left the tree line."):
        entry = next((e for e in self.tree_queue if e.actor_id == actor_id), None)
        if entry:
            self.tree_queue.remove(entry)
        elif self.tree_current and self.tree_current.actor_id == actor_id:
            entry, self.tree_current = self.tree_current, None
        if not entry:
            return
        self.reservations.release("tree", entry)
        actor = self.participant_by_id(actor_id)
        if actor is None:
            return
        self.tree.release(actor)
        if actor is not self.player and entry.saved_state:
            actor.restore_after_interaction(entry.saved_state, self.tree.id, self.snake.id)
        self.publish("tree_queue", (actor,), reason)

    def update_tree_queue(self, dt):
        from constants import TREE_QUEUE_SPACING, TREE_QUEUE_FACING, MOVEMENT_SPEED, PLAYER_APPROACH_MULTIPLIER, PLAYER_APPROACH_TIMEOUT
        from interactions import move_toward
        from locations import clamp_point
        from player import owns_resident
        for entry in [*self.tree_queue, *([self.tree_current] if self.tree_current else [])]:
            actor = self.participant_by_id(entry.actor_id)
            unavailable = (actor is None or not self.is_outdoors(actor)
                or entry.purpose == "bargain"
                and self.available_fruit < 1 and not (self.snake.encounter and self.snake.encounter.recipient is actor))
            if unavailable:
                self.leave_tree(entry.actor_id, "Tree service became unavailable.")
        if self.tree_current is None and self.tree_queue and self.snake.encounter is None and not self.snake.recovery_remaining:
            self.tree_current = self.tree_queue.pop(0)
        ordered = ([self.tree_current] if self.tree_current else []) + self.tree_queue
        if not ordered:
            self._tree_exit_targets.clear()
            return set()
        points = [clamp_point("garden", (self.tree.interaction_point[0] - index * TREE_QUEUE_SPACING,
                                         self.tree.interaction_point[1])) for index in range(len(ordered))]
        customers = [self.participant_by_id(entry.actor_id) for entry in ordered]
        neighbors = self.outdoor_npcs
        if self.is_outdoors(self.player):
            neighbors.append(self.player)
        queue_positions = {entry.actor_id: index for index, entry in enumerate(ordered)}
        clearing_ids = set()
        self._tree_exit_targets = {key: point for key, point in self._tree_exit_targets.items()
                                   if any(actor.id == key for actor in neighbors)}
        # Finish a departing customer's sidestep before resuming their saved
        # activity. Passersby keep their normal movement instead of being pushed.
        for actor in neighbors:
            position = queue_positions.get(actor.id)
            eligible = (position is not None if actor is self.player else
                not (actor.busy or actor.journey or owns_resident(self, actor) or actor.player_call
                     or actor.interaction_blocked or actor.owner == "player_hold")
                and (position is not None or actor.resume_pending))
            if not eligible or position == 0:
                self._tree_exit_targets.pop(actor.id, None)
                continue
            obstructed = points if position is None else [p for ahead, p in zip(customers[:position], points[:position])
                if math.dist((ahead.x, ahead.y), p) > 1e-6]
            point = next((p for p in obstructed if math.dist((actor.x, actor.y), p) < TREE_QUEUE_SPACING - 1e-6), None)
            if point is None:
                self._tree_exit_targets.pop(actor.id, None)
                continue
            def clear(p):
                return (all(math.dist(p, slot) >= TREE_QUEUE_SPACING - 1e-6 for slot in points)
                    and all(other is actor or math.dist(p, (other.x, other.y)) >= TREE_QUEUE_SPACING - 1e-6
                            for other in neighbors))
            exit_point = self._tree_exit_targets.get(actor.id)
            if exit_point is None or not clear(exit_point):
                exits = [clamp_point("garden", (point[0] + dx, point[1] + dy))
                         for dx, dy in ((0, TREE_QUEUE_SPACING), (0, -TREE_QUEUE_SPACING), (TREE_QUEUE_SPACING, 0))]
                exit_point = next((p for p in exits if clear(p)), None)
            if exit_point is not None:
                self._tree_exit_targets[actor.id] = exit_point
                clearing_ids.add(actor.id)
                if move_toward(actor, exit_point, dt, MOVEMENT_SPEED * (PLAYER_APPROACH_MULTIPLIER if actor is self.player else 1)):
                    self._tree_exit_targets.pop(actor.id, None)
        for index, (actor, entry, point) in enumerate(zip(customers, ordered, points)):
            if actor.id in clearing_ids or self.snake.encounter and self.snake.encounter.recipient is actor:
                continue
            if math.dist((actor.x, actor.y), point) > 1e-6 and any(
                    other is not actor and math.dist(point, (other.x, other.y)) < TREE_QUEUE_SPACING - 1e-6
                    for other in neighbors):
                continue  # Keep FIFO positions while a previous occupant clears them.
            arrived = move_toward(actor, point, dt, MOVEMENT_SPEED * (PLAYER_APPROACH_MULTIPLIER if actor is self.player else 1))
            if arrived:
                actor.facing = TREE_QUEUE_FACING
            if index != 0 or self.tree_current is not entry:
                continue
            entry.elapsed += dt
            if entry.elapsed > PLAYER_APPROACH_TIMEOUT:
                self.leave_tree(actor.id, "The tree approach timed out.")
            elif arrived:
                try:
                    from snake import SnakeEncounter, PlayerSnakeEncounter
                    cls = PlayerSnakeEncounter if actor is self.player else SnakeEncounter
                    if actor is not self.player:
                        actor.resume_pending = False
                    encounter = cls(self, actor, entry.purpose, entry.reason,
                                    origin="player" if actor is self.player else "autonomous")
                    self.tree.reserved_by = actor
                except ValueError as exc:
                    self.leave_tree(actor.id, str(exc))
        return clearing_ids

    def listeners(self, actor, radius=None):
        radius = HEARING_RADIUS if radius is None else radius
        return [p for p in self.npcs if p.location == actor.location and math.dist((p.x, p.y), (actor.x, actor.y)) <= radius]

    def endorsement_audience(self, actor):
        from player import active_player
        audience = [p for p in self.listeners(actor) if p is not actor]
        if (active_player(self) and actor is not self.player and self.player.location == actor.location
                and math.dist((self.player.x, self.player.y), (actor.x, actor.y)) <= HEARING_RADIUS):
            audience.append(self.player)
        return audience

    def observed_event(self, kind, subject, target, text, record=None, *, witnesses=None):
        from social import add_fact
        from conversation_history import remember
        if witnesses is None:
            witnesses = [subject, target, *self.listeners(subject, WITNESS_RADIUS)]
            if subject.location == "garden" and math.dist((self.snake.x, self.snake.y), (subject.x, subject.y)) <= WITNESS_RADIUS:
                witnesses.append(self.snake)
        witnesses = list({p.id: p for p in witnesses if p is not self.player and p.location == subject.location}.values())
        event = self.publish(kind, (subject, target), text, record)
        fact = add_fact(self, kind, subject, target, event.id, record, text, witnesses)
        for actor in witnesses:
            if actor is self.snake:
                continue
            partner = target if actor is subject else subject
            remember(actor, partner, event, kind, text, -4 if kind == "eating" else 0)
        return event, fact

    def eating_permitted(self, target=None, *, encounter=None, reachable=False):
        if self.population <= PROTECTED_POPULATION or self.snake.recovery_remaining > 0:
            return False
        if self.snake.eaten_today >= 1 and self.population <= CROWD_LIMIT:
            return False
        if target is not None:
            if not self.is_outdoors(target):
                return False
            if reachable and math.dist((self.snake.x, self.snake.y), (target.x, target.y)) > 32:
                return False
        if encounter is not None:
            return (encounter.active and encounter is self.snake.encounter and encounter.committed
                    and encounter.outcome == "eat" and encounter.stage == "eating")
        return True

    def commit_snake_response(self, encounter, response):
        from social import learn_account, add_fact
        npc, snake = encounter.recipient, self.snake
        decision = response.decision
        if not encounter.accepts_response or encounter.committed or not encounter.arrived:
            raise ValueError("This encounter is no longer available.")
        if decision not in encounter.request_context["permitted_decisions"]:
            raise ValueError("That decision is unavailable.")
        player_encounter = npc is self.player
        answer_turn = 1 if player_encounter else response.outcome_turn - 1
        if decision == "give_apple":
            if (not encounter.reserved or self.tree.fruit < 1
                    or math.dist((npc.x, npc.y), self.tree.interaction_point) > 8
                    or math.dist((npc.x, npc.y), (snake.x, snake.y)) > HEARING_RADIUS):
                raise ValueError("An apple requires stock and the present tree customer.")
            if player_encounter:
                if not response.disclosure_evidence or response.disclosure_evidence not in encounter.request_context["_player_message"]:
                    raise ValueError("The player must actually supply information now.")
            elif not response.account_id and (not response.evidence.strip() or response.evidence not in response.turns[answer_turn].text):
                raise ValueError("An apple requires an actual disclosure or quoted present report.")
        if not player_encounter and response.account_id:
            allowed = {a["claim_id"] for a in encounter.request_context["offering_accounts"]}
            if response.account_id not in allowed or not any(b.claim_id == response.account_id for b in npc.beliefs):
                raise ValueError("The disclosed account is not owned by this resident.")
        if response.snake_account_id and (not any(b.claim_id == response.snake_account_id for b in snake.beliefs)
                or response.snake_account_id not in {a["claim_id"] for a in encounter.request_context["snake"]["known_accounts"]}):
            raise ValueError("The guard cannot disclose an unknown account.")
        if decision == "eat" and (not self.eating_permitted(npc)
                or not encounter.request_context.get("known_warning")
                or snake.offenses.get(npc.id) != encounter.request_context["known_warning"]):
            raise ValueError("Condemnation requires the existing warning and current eating permission.")
        encounter.committed, encounter.outcome, encounter.reserved = True, decision, False
        # Disclosures survive refusal of a reward. They never become verified truth.
        if not player_encounter and response.account_id:
            learn_account(self, npc, snake, response.account_id, encounter.record, answer_turn, "uncertain", private=True)
        if response.snake_account_id and not player_encounter:
            known = next(a for a in encounter.request_context["snake"]["known_accounts"] if a["claim_id"] == response.snake_account_id)
            # Preserve the reply as testimony about a report, never rewrite the original fact.
            add_fact(self, "report", npc, snake, encounter.id + ":answer", encounter.record,
                     f"{npc.name} answered about this account: {known['assertion']} "
                     f"Their response: {response.turns[answer_turn].text} This response is not independently verified.", [npc, snake])
            learn_account(self, snake, npc, response.snake_account_id, encounter.record, 0, "uncertain")
        if not player_encounter and decision == "give_apple" and not response.account_id:
            add_fact(self, "report", npc, snake, encounter.id + ":report", encounter.record,
                     f"{npc.name} reported: {response.turns[answer_turn].text} This is their account, not independently verified.", [npc, snake])
        if player_encounter and response.disclosure_evidence:
            current = encounter.request_context["current_claim"]
            add_fact(self, "report", npc, snake, current["id"], encounter.record,
                     f"{npc.name} reported: {encounter.request_context['_player_message']} "
                     + (f"In response to this sourced account: {encounter.request_context['questioned_account']['assertion']} "
                        if encounter.request_context.get("questioned_account") else "")
                     + "This is their account, not independently verified.", [snake])
        if decision == "give_apple":
            self.tree.fruit -= 1
            npc.inventory["fruit"] = npc.inventory.get("fruit", 0) + 1
            text = f"{SNAKE_NAME} gave {npc.name} one apple for information supplied in this conversation. No further payment is owed."
            event, _ = self.observed_event("bargain", snake, npc, text, encounter.record)
            encounter.outcome_event_id = event.id
        elif decision == "warn":
            text = f"{SNAKE_NAME} warned {npc.name} of being eaten: {response.accusation} Evidence from their response: {response.evidence} This is an accusation, not established guilt."
            event, fact = self.observed_event("judgment", snake, npc, text, encounter.record)
            snake.offenses[npc.id] = fact.id
            encounter.outcome_event_id = event.id
        elif decision == "eat":
            encounter.accusation = response.accusation
            event, _ = self.observed_event("judgment", snake, npc,
                f"{SNAKE_NAME} condemned {npc.name} after an earlier warning and subsequent response: {response.accusation}", encounter.record)
            encounter.outcome_event_id = event.id
        elif npc.id in snake.offenses:
            # A satisfactory response resolves the warning without a separate pardon system.
            snake.offenses.pop(npc.id)
        if not player_encounter and response.attitudes is not None:
            updated = response.attitudes.model_dump()
            updated["trust"] = max(npc.attitudes["trust"] - 6, min(npc.attitudes["trust"] + 6, updated["trust"]))
            npc.attitudes = updated
        if encounter.request_context["first_meeting"]:
            self.observed_event("snake_claim", snake, npc,
                f"{npc.name} heard {SNAKE_NAME}'s claims about authority and the garden; these remain unverified.", encounter.record)
        encounter.record.outcomes.append(f"Decision: {decision}. " + ("One apple transferred." if decision == "give_apple" else "No apple transferred."))

    def interrupt_resident(self, target, reason, *, departing=False):
        """Cancel only unfinished work; preserve every committed handoff and record."""
        from activity import Behavior
        from transfers import Handoff
        from player import owns_resident, end_interaction
        if owns_resident(self, target):
            end_interaction(self, reason)
        if self.visit and self.visit.host_id == target.id:
            from apartments import finish_visit
            finish_visit(self)
        self.leave_tree(target.id, reason)
        interaction = target.conversation or target.gift
        from player import clear_call
        from transfers import Handoff
        clear_call(target)
        self.player_acknowledgments = [a for a in self.player_acknowledgments if a.npc_id != target.id]
        if isinstance(interaction, Handoff):
            interaction.finish(reason)
            interaction = None
        if interaction:
            interaction.cancel(reason)
        encounter = target.snake_encounter
        if encounter:
            encounter.finish(error=reason)
        self.tree.release(target)
        for record in self.history:
            if record.includes(target) and self.player.id in record.participant_ids and record.status in CONVERSATION_PENDING_REPLY_STATUSES:
                record.status, record.error = CONVERSATION_CANCELLED, reason
                record.outcomes.append(reason)
        if departing:
            target.pending_social_opinions.clear()
        # An unrelated interaction can have suspended a task aimed at this resident.
        interactions = {id(p.conversation or p.gift): p.conversation or p.gift
                        for p in self.npcs if p.conversation or p.gift}
        states = [state for interaction in interactions.values()
                  for state in interaction.activity_snapshots()]
        if self.snake.encounter is not None and self.snake.encounter.saved_state is not None:
            states.extend(self.snake.encounter.activity_snapshots())
        for state in states:
            state.cooldowns.pop(target.id, None)
            if state.target_id == target.id:
                state.behavior, state.owner, state.target_id = Behavior.WAIT, "autonomous", None
                state.destination, state.activity_elapsed, state.resume_pending = None, 0.0, False
                state.reason, state.decision_remaining = reason, 0.0
        remaining = []
        for action in self.pending_actions:
            if target.id in (action.npc_id, action.target_id):
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
            if person.target_id == target.id:
                person.finish_activity(self, failed=True)
            person.cooldowns.pop(target.id, None)
            if person is target:
                person.command_revision += 1
                person.player_reply.complete()
                person.selected = False
                person.destination = person.target_id = None
                person.behavior = Behavior.WAIT
                person.owner = "autonomous"
                person.activity_elapsed = 0.0
                person.resume_pending = False

    def depart_resident(self, target, encounter):
        if target is self.player:
            return self.kill_player(encounter)
        from social import resolve_goal
        if target.departure or not self.eating_permitted(target, encounter=encounter, reachable=True):
            return False
        # Mark before cleanup so a second call cannot remove or count this actor twice.
        target.departure = {"day": self.day, "phase": self.phase, "cause": "eaten by the snake", "event_id": ""}
        reason = f"Cancelled because {target.name} was eaten; this is not a deliberate betrayal."
        # The eating encounter is the preserved source, not an abandoned bargain.
        self.reservations.release("snake", encounter, target)
        for promise in self.promises:
            if promise.status == "pending" and target.id in (promise.promisor_id, promise.recipient_id):
                promise.status, promise.resolved_phase, promise.outcome = "cancelled", self.phase_index, reason
                event = self.publish("promise_cancelled", (self.participant_by_id(promise.promisor_id), self.participant_by_id(promise.recipient_id)), reason,
                                     next(r for r in self.history if r.id == promise.conversation_id))
                promise.resolution_event_id = promise.consequence_event_id = event.id
        for person in self.npcs:
            goal = person.goal
            if goal and goal.status == "active" and (person is target or target.id in (goal.target_id, goal.third_npc_id)):
                resolve_goal(person, self, "cancelled", reason, encounter.record)
        from apartments import cancel_work
        cancel_work(target)
        target.home_reason = target.bedtime_remaining = target.wake_remaining = None
        target.home_remaining = 0.0
        self.interrupt_resident(target, reason, departing=True)
        self.npcs[:] = [p for p in self.npcs if p is not target]
        self.archived.append(target)
        self.roster_revision += 1
        self.introductions[:] = [i for i in self.introductions if i != target.id]
        self.social_rotation %= max(1, len(self.npcs))
        event, _ = self.observed_event("eating", self.snake, target,
            f"{SNAKE_NAME} ate {target.name}. {SNAKE_NAME} claims this sends a soul to hell; that destination remains unverified.", encounter.record)
        target.departure["event_id"] = event.id
        for witness in self.listeners(self.snake, 220):
            friend = witness.opinion_score(target.id)
            adjustment = -6 if friend > 0 or witness.attitudes["authority"] == "rejected" else 2 if witness.attitudes["belief"] == "believed" else -3
            witness.attitudes["trust"] = max(-100, min(100, witness.attitudes["trust"] + adjustment))
            witness.attitudes["trust_reason"] = f"Witnessed {SNAKE_NAME} eat {target.name} on day {self.day}."
        self.snake.eaten_today += 1
        self.snake.recovery_remaining = RECOVERY_SECONDS
        self.snake.offenses.pop(target.id, None)
        if self.snake.retry_request and self.snake.retry_request["resident_id"] == target.id:
            self.snake.retry_request = None
        encounter.outcome_event_id = event.id
        encounter.finish()
        encounter.record.status = CONVERSATION_EATEN_RESIDENT
        encounter.record.outcomes.append(f"{target.name} departed on day {self.day}. Their destination is unverified.")
        return True

    def kill_player(self, encounter):
        if not self.player.alive or not self.eating_permitted(self.player, encounter=encounter, reachable=True):
            return False
        from player import end_interaction
        from apartments import finish_visit
        self.player.alive = False
        self.mode = "spectator"
        reason = "The player was eaten; outstanding obligations ended with their death."
        end_interaction(self, reason)
        finish_visit(self)
        self.player.journey = None
        self.leave_tree(self.player.id, reason)
        for npc in self.npcs:
            npc.player_call = None
        for promise in self.promises:
            if promise.status == "pending" and self.player.id in (promise.promisor_id, promise.recipient_id):
                promise.status, promise.resolved_phase, promise.outcome = "cancelled", self.phase_index, reason
                event = self.publish("promise_cancelled", (self.player,), reason, encounter.record)
                promise.resolution_event_id = promise.consequence_event_id = event.id
        self.snake.offenses.pop(self.player.id, None)
        self.snake.eaten_today += 1
        self.snake.recovery_remaining = RECOVERY_SECONDS
        self.player_acknowledgments.clear()
        event, _ = self.observed_event("eating", self.snake, self.player,
            f"{SNAKE_NAME} ate the player. Their claimed destination remains unverified.", encounter.record)
        encounter.outcome_event_id = event.id
        encounter.finish()
        encounter.record.status = CONVERSATION_EATEN_PLAYER
        self.roster_revision += 1
        return True

    def deliver_arrivals(self):
        from character_data import random_character
        from npc import NPC, assign_npc_names
        from purgatory import remember_description
        from social import assign_goal
        if not self.pending_arrival_days or self.arrival_error:
            return None
        day = self.pending_arrival_days[0]
        if day in self.processed_arrival_days:
            self.pending_arrival_days.pop(0)
            return None
        try:
            character = random_character(trait_mode=self.npc_trait_mode)
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
        observers = [npc, *self.listeners(npc, 220)]
        if npc.location == self.snake.location and math.dist((npc.x, npc.y), (self.snake.x, self.snake.y)) <= 220:
            observers.append(self.snake)
        self.observed_event("arrival", npc, self.snake, f"{npc.name} arrived for dawn on day {day}, remembering a former life.", witnesses=observers)
        return npc


def validate_saved_tree_queue(entries, player, interaction, by_id, pending_actions, exclusive):
    from activity import ActivitySnapshot
    from constants import PLAYER_APPROACH_TIMEOUT
    if len({e.actor_id for e in entries}) != len(entries):
        raise ValueError("Tree customers must be unique and ordered.")
    for entry in entries:
        actor = by_id.get(entry.actor_id)
        if not actor or actor.location != "garden" or actor.journey or actor.id in exclusive or entry.purpose not in ("bargain", "judgment", "introduction"):
            raise ValueError("A tree customer must be available in the garden.")
        if not isinstance(entry.elapsed, (int, float)) or not math.isfinite(entry.elapsed) or not 0 <= entry.elapsed <= PLAYER_APPROACH_TIMEOUT:
            raise ValueError("Invalid front-customer approach progress.")
        if actor is player and interaction and interaction.kind != "snake":
            raise ValueError("A queued player cannot own another interaction.")
        if any(a.npc_id == actor.id for a in pending_actions):
            raise ValueError("A queued customer cannot have another pending action.")
        if actor is not player:
            ActivitySnapshot.validate(entry.saved_state)
        exclusive.add(actor.id)


def validate_saved_actions(npcs, player, mode, tree, pending_actions, live_ids, indoor_ids, reserved, history, creation_ids):
    from activity import Behavior, validate_activity_target
    from transfers import transfer_matches
    for index, npc in enumerate(npcs):
        if npc.id not in reserved:
            validate_activity_target(npc, npc, tree.id, live_ids)
        if npc.id in reserved and npc.behavior != Behavior.WAIT:
            raise ValueError("Encounter participants must wait.")
    for reference in (tree.reserved_by, tree.shaker):
        if reference is not None and (reference not in live_ids | ({player.id} if player.alive and mode == "player" else set()) or reference in indoor_ids):
            raise ValueError("A tree reservation requires a living owner.")
    if bool(tree.shaker) != (tree.shake_elapsed is not None):
        raise ValueError("Invalid tree shake owner.")
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
