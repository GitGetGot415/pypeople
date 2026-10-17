"""A small world with player-controlled day phases."""

from dataclasses import dataclass
import math

# Ordinary seconds keep movement and animations independent of rendering speed.
MOVEMENT_SPEED = 90.0
SHAKE_SECONDS = 1.0
WANDER_PAUSE = 0.5
DECISION_SECONDS = 4.0
DIALOGUE_WAIT_SECONDS = 60.0
FAILURE_COOLDOWN = 8.0
PARTNER_COOLDOWN = 25.0
SATISFACTION_PHASE_LOSS = 5
APPLE_SATISFACTION = 10
EAT_SECONDS = 1.0
COMPANY_SOCIAL_RECOVERY = 0.96
SOCIAL_RADIUS = 110
COMPANY_MAX_SECONDS = 100.0
SOCIAL_START = 48
SOCIAL_RECOVERED = 82
INVENTORY_CAPACITY = 3
FRUIT_PER_PHASE = 2


@dataclass
class PendingAction:
    npc_id: str
    action: str
    revision: int
    target_id: str | None = None
    topic: str = ""
    record: int | None = None
    item: str = ""
    quantity: int = 0


class Simulation:
    def __init__(self, npcs, tree=None, *, assign_goals=True):
        from npc import Tree
        from snake import Snake

        self.npcs = npcs
        self.tree = tree if tree is not None else Tree()
        self.snake = Snake()
        self.day = 1
        self.phase = "day"
        self.pending_actions = []
        self.history = []
        self.events = []
        self.promises = []
        self.facts = []
        self.claims = []
        self.disclosures = []
        self.social_opportunity = False
        self.social_rotation = 0
        for npc in self.npcs:
            npc.world = self
        if assign_goals:
            from social import assign_goal
            for npc in self.npcs:
                assign_goal(npc, self)

    @property
    def phase_index(self):
        from social import phase_index
        return phase_index(self.day, self.phase)

    @property
    def phase_label(self):
        return f"Day {self.day} / {self.phase.title()}"

    def next_phase(self):
        from social import advance_social_phase
        if self.phase == "day":
            self.phase = "evening"
        elif self.phase == "evening":
            self.phase = "night"
        else:
            self.phase = "day"
            self.day += 1
        for npc in self.npcs:
            npc.social = max(0.0, npc.social - SATISFACTION_PHASE_LOSS)
        self.tree.fruit = min(self.tree.capacity, self.tree.fruit + FRUIT_PER_PHASE)
        advance_social_phase(self)
        self.social_opportunity = True
        self.publish("phase", [], f"Time changed to {self.phase}.")

    def publish(self, kind, participants, text, record=None):
        from conversation_history import WorldEvent

        reference = next((i for i, entry in enumerate(self.history) if entry is record), None)
        event = WorldEvent(self.day, self.phase, kind, [npc.id for npc in participants], text, reference)
        self.events.append(event)
        from social import record_event_fact
        record_event_fact(self, event, record)
        del self.events[:-200]
        return event

    def start_conversation(self, initiator, recipient, topic, *, origin="player", purpose="chat", spacing=0):
        from npc import Conversation

        interaction = Conversation(initiator, recipient, topic, self.tree, spacing, origin=origin, purpose=purpose)
        self.history.append(interaction.record)
        return interaction

    def npc_by_id(self, npc_id):
        return next((npc for npc in self.npcs if npc.id == npc_id), None)

    def advance(self, dt, *, active=True, action_handler=None, viewed_conversation=None):
        if not active:
            return
        if action_handler is not None:
            actions, self.pending_actions = self.pending_actions, []
            for action in actions:
                npc = self.npc_by_id(action.npc_id)
                if npc is not None and npc.selected and npc.end_conversation:
                    # Keep the closing reply visible until the player dismisses it.
                    self.pending_actions.append(action)
                    continue
                action_handler(action)
        # Ignore long window stalls; there is no elapsed world time to catch up.
        self.step(min(0.1, max(0.0, dt)), viewed_conversation=viewed_conversation)

    def company_available(self, npc):
        from npc import Behavior

        return (npc.conversation is None and npc.gift is None and not npc.selected and not npc.request_pending
                and npc.owner == "autonomous" and not npc.resume_pending
                and not any(npc.id in (action.npc_id, action.target_id) for action in self.pending_actions)
                and npc.behavior in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY))

    def step(self, dt, *, viewed_conversation=None):
        from autonomy import reconsider, social_pair
        from npc import Behavior

        self.tree.update(dt)
        self.snake.update(dt)
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
        if self.social_opportunity:
            choice = social_pair(self)
            if choice is not None:
                initiator, recipient, purpose, topic, reason = choice
                self.start_conversation(initiator, recipient, topic, origin="autonomous", purpose=purpose)
                initiator.reason = reason
                recipient.reason = (f"Answering {initiator.name}'s request for an apple" if purpose == "food_request"
                                    else f"Chatting with {initiator.name}")
                self.social_opportunity = False
                self.social_rotation = (self.npcs.index(initiator) + 1) % len(self.npcs)
        for npc in self.npcs:
            if npc.resume_pending and npc.conversation is None and npc.gift is None:
                npc.resume_activity(self)
            reconsider(npc, self)
            npc.update(dt, self.tree)
        # Each person benefits once, regardless of how many neighbors are present.
        socializing = []
        for npc in self.npcs:
            if not self.company_available(npc):
                continue
            if any(other is not npc and self.company_available(other)
                   and (npc.behavior == Behavior.COMPANY or other.behavior == Behavior.COMPANY)
                   and math.hypot(npc.x - other.x, npc.y - other.y) <= SOCIAL_RADIUS
                   for other in self.npcs):
                socializing.append(npc)
        for npc in socializing:
            npc.social = min(100.0, npc.social + COMPANY_SOCIAL_RECOVERY * dt)
