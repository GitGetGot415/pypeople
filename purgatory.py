"""Snake attitudes and private knowledge from a resident's description."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Attitudes(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    belief: Literal["believed", "uncertain", "disbelieved"]
    belief_reason: str = Field(min_length=1, max_length=400)
    trust: int = Field(ge=-100, le=100)
    trust_reason: str = Field(min_length=1, max_length=400)
    authority: Literal["accepted", "uncertain", "rejected"]
    authority_reason: str = Field(min_length=1, max_length=400)


def initial_attitudes():
    return {
        "belief": "uncertain", "belief_reason": "Has not heard the snake's claims yet.",
        "trust": 0, "trust_reason": "Has not dealt with the snake yet.",
        "authority": "uncertain", "authority_reason": "Has not encountered the snake's rules yet.",
    }


def remember_description(npc):
    """The authored description is the only source of former-life knowledge."""
    from social import add_fact

    description = npc.character["Details"].strip()
    if description:
        add_fact(npc.world, "former_life", npc, npc.world.snake,
                 f"{npc.id}:description", None,
                 f"From {npc.name}'s own description: {description}", [npc])
