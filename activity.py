"""Resident activity values shared by suspension and persistence."""

from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Annotated, Literal
from pydantic import Field
from constants import APARTMENT_ID, EAT_SECONDS


class Behavior(str, Enum):
    WANDER = "WANDER"
    GO_TO_TREE = "GO_TO_TREE"
    SHAKE_TREE = "SHAKE_TREE"
    SHAKING = "SHAKING"
    HARVEST = "HARVEST"
    EAT = "EAT"
    COMPANY = "COMPANY"
    WAIT = "WAIT"
    GO_HOME = "GO_HOME"
    REST = "REST"
    SLEEP = "SLEEP"
    PAINT = "PAINT"
    COMPOSE = "COMPOSE"


@dataclass
class ActivitySnapshot:
    behavior: Behavior
    destination: tuple[float, float] | None
    pause_remaining: Annotated[float, Field(ge=0)]
    facing: Annotated[int, Field(ge=0, le=7)]
    owner: Literal["autonomous", "player_task"]
    target_id: str | None
    activity_elapsed: Annotated[float, Field(ge=0)]
    reason: str
    decision_remaining: Annotated[float, Field(ge=0)]
    cooldowns: dict[str, float]
    resume_pending: bool
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
        player_behaviors = (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING)
        if self.behavior != Behavior.EAT and (self.owner == "player_task") != (self.behavior in player_behaviors):
            raise ValueError("Invalid player task.")
        if self.behavior == Behavior.EAT and (self.destination is not None or self.activity_elapsed >= EAT_SECONDS):
            raise ValueError("Invalid eating progress.")
        return self


def validate_activity_target(activity, owner, tree_id, live_ids):
    target = activity.target_id
    if activity.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING, Behavior.HARVEST):
        if target != tree_id:
            raise ValueError("Invalid tree task target.")
    elif activity.behavior == Behavior.GO_HOME:
        if target != APARTMENT_ID:
            raise ValueError("Invalid apartment target.")
    elif activity.behavior == Behavior.COMPANY:
        if target not in live_ids or target == owner.id:
            raise ValueError("A company target must be living.")
    elif target is not None:
        raise ValueError("Unexpected action target.")
