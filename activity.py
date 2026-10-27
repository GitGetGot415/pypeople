"""Resident activity values shared by suspension and persistence."""

import random
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Annotated, Literal
from pydantic import Field
from constants import EAT_SECONDS, STARTING_FACING, DECISION_SECONDS


class Behavior(str, Enum):
    WANDER = "WANDER"
    GO_TO_TREE = "GO_TO_TREE"
    HARVEST = "HARVEST"
    EAT = "EAT"
    WAIT = "WAIT"
    GO_HOME = "GO_HOME"
    REST = "REST"
    PAINT = "PAINT"
    COMPOSE = "COMPOSE"


@dataclass(kw_only=True)
class ActivitySnapshot:
    behavior: Behavior = Behavior.WANDER
    destination: tuple[float, float] | None = None
    pause_remaining: Annotated[float, Field(ge=0)] = 0.0
    facing: Annotated[int, Field(ge=0, le=7)] = STARTING_FACING
    owner: Literal["autonomous", "player_task"] = "autonomous"
    target_id: str | None = None
    activity_elapsed: Annotated[float, Field(ge=0)] = 0.0
    reason: str = "Exploring the world"
    decision_remaining: Annotated[float, Field(ge=0)] = field(default_factory=lambda: random.uniform(0, DECISION_SECONDS))
    cooldowns: dict[str, float] = field(default_factory=dict)
    resume_pending: bool = False
    eating_witness_ids: list[str] = field(default_factory=list)

    @classmethod
    def capture(cls, actor):
        return cls(**{field.name: deepcopy(getattr(actor, field.name)) for field in fields(cls)})

    @classmethod
    def restore(cls, actor, values):
        for field in fields(cls):
            setattr(actor, field.name, deepcopy(getattr(values, field.name)))

    def validate(self):
        if (len(set(self.eating_witness_ids)) != len(self.eating_witness_ids)
                or self.eating_witness_ids and self.behavior != Behavior.EAT):
            raise ValueError("Eating witnesses belong only to an unfinished apple consumption.")
        if any(value < 0 for value in self.cooldowns.values()):
            raise ValueError("Invalid cooldown duration.")
        if self.destination is not None:
            from locations import validate_destination
            validate_destination(self.destination, location=getattr(self, "location", None))
        player_behaviors = (Behavior.GO_TO_TREE,)
        if self.behavior != Behavior.EAT and self.owner == "player_task" and self.behavior not in player_behaviors:
            raise ValueError("Invalid player task.")
        if self.behavior == Behavior.HARVEST and self.destination is not None:
            raise ValueError("Harvesting is a local action; use GO_TO_TREE for travel.")
        if self.behavior == Behavior.EAT and (self.destination is not None or self.activity_elapsed >= EAT_SECONDS):
            raise ValueError("Invalid eating progress.")
        return self
