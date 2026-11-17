from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field


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
