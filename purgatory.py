"""Small shared contracts for former lives and prepared arrivals."""

from dataclasses import dataclass, field
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Background(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    life_context: str = Field(min_length=1, max_length=600)
    memories: list[str] = Field(min_length=2, max_length=3)
    unresolved_concern: str = Field(min_length=1, max_length=400)
    private_matter: str = Field(min_length=1, max_length=400)
    prior_worldview: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def nonempty(self):
        if any(not value.strip() for value in [self.life_context, *self.memories,
               self.unresolved_concern, self.private_matter, self.prior_worldview]):
            raise ValueError("A former life needs concrete, nonempty details.")
        if len(set(self.memories)) != len(self.memories) or any(len(m) > 400 for m in self.memories):
            raise ValueError("Use two or three distinct, compact memories.")
        return self


class Attitudes(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    belief: Literal["believed", "uncertain", "disbelieved"]
    belief_reason: str = Field(min_length=1, max_length=400)
    trust: int = Field(ge=-100, le=100)
    trust_reason: str = Field(min_length=1, max_length=400)
    authority: Literal["accepted", "uncertain", "rejected"]
    authority_reason: str = Field(min_length=1, max_length=400)


class BackgroundResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actor_id: str
    background: Background
    attitudes: Attitudes


@dataclass
class Arrival:
    character: dict
    id: str = field(default_factory=lambda: uuid4().hex)
    background: dict | None = None
    attitudes: dict | None = None
    state: str = "waiting"
    error: str = ""


def background_snapshot(actor):
    return {"request_kind": "background", "actor_id": actor.id,
            "character": {k: v for k, v in actor.character.items() if k != "Opinions"}}


def install_background(npc, response):
    """Register recollections as private accounts, never as current-world people."""
    from social import add_fact

    if npc.background is not None:
        return
    npc.background = response.background.model_dump()
    npc.attitudes = response.attitudes.model_dump()
    values = {**npc.background}
    memories = values.pop("memories")
    values.update({f"memory_{i}": text for i, text in enumerate(memories)})
    for key, text in values.items():
        add_fact(npc.world, "former_life", npc, npc.world.snake,
                 f"{npc.id}:background:{key}", None,
                 f"{npc.name} remembers from their former life ({key.replace('_', ' ')}): {text}", [npc])
