"""Local movement and tree animation. No API calls or Pygame dependencies."""

from dataclasses import dataclass, field
from enum import Enum
import math
import random

WORLD_WIDTH = 960
WORLD_HEIGHT = 460
NPC_RADIUS = 18
SPEED = 90.0
TREE_INTERACTION_POINT = (660.0, 280.0)


class Behavior(str, Enum):
    WANDER = "WANDER"
    STOP = "STOP"
    GO_TO_TREE = "GO_TO_TREE"
    SHAKE_TREE = "SHAKE_TREE"
    SHAKING = "SHAKING"


@dataclass
class Tree:
    x: float = 720.0
    y: float = 230.0
    shake_elapsed: float | None = None

    @property
    def shaking(self):
        return self.shake_elapsed is not None

    @property
    def offset(self):
        if self.shake_elapsed is None:
            return 0
        return round(5 * math.sin(self.shake_elapsed * math.tau * 8))

    def start_shake(self):
        self.shake_elapsed = 0.0

    def cancel_shake(self):
        self.shake_elapsed = None

    def update(self, dt):
        if self.shake_elapsed is not None:
            self.shake_elapsed += dt
            if self.shake_elapsed >= 1.0 - 1e-9:
                self.cancel_shake()


@dataclass
class NPC:
    x: float = 240.0
    y: float = 230.0
    behavior: Behavior = Behavior.WANDER
    destination: tuple[float, float] | None = None
    pause_remaining: float = 0.0
    rng: random.Random = field(default_factory=random.Random, repr=False)

    @property
    def status(self):
        return {
            Behavior.WANDER: "Wandering",
            Behavior.STOP: "Stopped",
            Behavior.GO_TO_TREE: "Walking to tree",
            Behavior.SHAKE_TREE: "Walking to tree",
            Behavior.SHAKING: "Shaking tree",
        }[self.behavior]

    def contains(self, point):
        return math.hypot(point[0] - self.x, point[1] - self.y) <= NPC_RADIUS

    def snapshot(self, tree):
        return {
            "npc_name": "Judy",
            "position": [round(self.x, 1), round(self.y, 1)],
            "behavior": self.behavior.value,
            "only_interactable_object": "tree",
            "tree_interaction_point": list(TREE_INTERACTION_POINT),
            "tree_is_shaking": tree.shaking,
        }

    def apply_action(self, action, tree):
        # Reject before changing anything. SAY preserves even an active shake.
        if action == "SAY":
            return
        if action not in ("WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE"):
            raise ValueError("Unsupported NPC action")
        tree.cancel_shake()
        self.pause_remaining = 0.0
        if action == "WANDER":
            self.behavior = Behavior.WANDER
            self.destination = None
        elif action == "STOP":
            self.behavior = Behavior.STOP
            self.destination = None
        elif action == "GO_TO_TREE":
            self.behavior = Behavior.GO_TO_TREE
            self.destination = TREE_INTERACTION_POINT
        elif action == "SHAKE_TREE":
            self.behavior = Behavior.SHAKE_TREE
            self.destination = TREE_INTERACTION_POINT

    def _move(self, dt):
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        distance = math.hypot(dx, dy)
        step = SPEED * dt
        if distance <= step:
            self.x, self.y = self.destination
            return True
        if distance:
            self.x += dx / distance * step
            self.y += dy / distance * step
        return False

    def update(self, dt, tree):
        dt = max(0.0, dt)
        tree.update(dt)
        if self.behavior == Behavior.SHAKING:
            if not tree.shaking:
                self.behavior = Behavior.STOP
        elif self.behavior == Behavior.WANDER:
            if self.pause_remaining > 0:
                remaining = self.pause_remaining
                self.pause_remaining = max(0.0, remaining - dt)
                dt = max(0.0, dt - remaining)
            if dt > 0:
                if self.destination is None:
                    self.destination = (
                        self.rng.uniform(NPC_RADIUS, WORLD_WIDTH - NPC_RADIUS),
                        self.rng.uniform(NPC_RADIUS, WORLD_HEIGHT - NPC_RADIUS),
                    )
                if self._move(dt):
                    self.destination = None
                    self.pause_remaining = 0.5
        elif self.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE):
            if self._move(dt):
                self.destination = None
                if self.behavior == Behavior.SHAKE_TREE:
                    tree.start_shake()
                    self.behavior = Behavior.SHAKING
                else:
                    self.behavior = Behavior.STOP
        self.x = min(WORLD_WIDTH - NPC_RADIUS, max(NPC_RADIUS, self.x))
        self.y = min(WORLD_HEIGHT - NPC_RADIUS, max(NPC_RADIUS, self.y))
