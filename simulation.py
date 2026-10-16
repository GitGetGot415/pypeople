"""A small world with player-controlled day phases."""

from dataclasses import dataclass
import math

# Ordinary seconds keep movement and animations independent of rendering speed.
MOVEMENT_SPEED = 90.0
SHAKE_SECONDS = 1.0
WANDER_PAUSE = 0.5
DECISION_SECONDS = 4.0
FAILURE_COOLDOWN = 8.0
PARTNER_COOLDOWN = 25.0
SOCIAL_DECAY = 0.09
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
    def __init__(self, npcs, tree=None):
        from npc import Tree

        self.npcs = npcs
        self.tree = tree if tree is not None else Tree()
        self.day = 1
        self.phase = "day"
        self.pending_actions = []
        for npc in self.npcs:
            npc.world = self

    @property
    def phase_label(self):
        return f"Day {self.day} / {self.phase.title()}"

    def next_phase(self):
        if self.phase == "day":
            self.phase = "evening"
        elif self.phase == "evening":
            self.phase = "night"
        else:
            self.phase = "day"
            self.day += 1
        self.tree.fruit = min(self.tree.capacity, self.tree.fruit + FRUIT_PER_PHASE)

    def npc_by_id(self, npc_id):
        return next((npc for npc in self.npcs if npc.id == npc_id), None)

    def advance(self, dt, *, active=True, action_handler=None):
        if not active:
            return
        if action_handler is not None:
            actions, self.pending_actions = self.pending_actions, []
            for action in actions:
                action_handler(action)
        # Ignore long window stalls; there is no elapsed world time to catch up.
        self.step(min(0.1, max(0.0, dt)))

    def company_available(self, npc):
        from npc import Behavior

        return (npc.conversation is None and not npc.selected and not npc.request_pending
                and npc.owner == "autonomous"
                and npc.behavior in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY))

    def step(self, dt):
        from autonomy import reconsider
        from npc import Behavior

        self.tree.update(dt)
        for npc in self.npcs:
            npc.update_social(dt)
        for npc in self.npcs:
            if npc.resume_pending and npc.conversation is None:
                npc.resume_activity(self)
            reconsider(npc, self)
            npc.update(dt, self.tree)
            if npc.conversation is not None and npc.conversation.initiator is npc:
                npc.conversation.update(dt)
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
