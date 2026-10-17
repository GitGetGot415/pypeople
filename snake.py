"""Ambient wildlife with no NPC or player interactions."""

from dataclasses import dataclass, field
import math
import random

from npc import WORLD_HEIGHT, WORLD_WIDTH


SNAKE_MARGIN = 24
SNAKE_SPEED = 35.0
SNAKE_FRAME_SECONDS = 0.22


@dataclass
class Snake:
    x: float = 720.0
    y: float = 330.0
    destination: tuple[float, float] | None = None
    facing: int = 4
    pause_remaining: float = 0.0
    animation_elapsed: float = 0.0
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    @property
    def frame(self):
        return 1 + int(self.animation_elapsed / SNAKE_FRAME_SECONDS) % 2

    def update(self, dt):
        if dt <= 0:
            return
        remaining = self.pause_remaining
        self.pause_remaining = max(0.0, remaining - dt)
        movement_dt = max(0.0, dt - remaining)
        if movement_dt == 0:
            self.animation_elapsed = 0.0
            return
        if self.destination is None:
            self.destination = (
                self.rng.uniform(SNAKE_MARGIN, WORLD_WIDTH - SNAKE_MARGIN),
                self.rng.uniform(SNAKE_MARGIN, WORLD_HEIGHT - SNAKE_MARGIN),
            )
        dx, dy = self.destination[0] - self.x, self.destination[1] - self.y
        distance = math.hypot(dx, dy)
        if distance:
            self.facing = math.floor(math.atan2(dx, -dy) / (math.pi / 4) + 0.5) % 8
        step = SNAKE_SPEED * movement_dt
        if distance <= step:
            self.x, self.y = self.destination
            self.destination = None
            self.pause_remaining = self.rng.uniform(0.8, 2.4)
            self.animation_elapsed = 0.0
        else:
            self.x += dx / distance * step
            self.y += dy / distance * step
            self.animation_elapsed = (self.animation_elapsed + movement_dt) % (2 * SNAKE_FRAME_SECONDS)
