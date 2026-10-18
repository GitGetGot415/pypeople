"""Snake attitudes and private knowledge from a resident's description."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from constants import SNAKE_NAME


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
        "belief": "uncertain", "belief_reason": f"Has not heard {SNAKE_NAME}'s claims yet.",
        "trust": 0, "trust_reason": f"Has not dealt with {SNAKE_NAME} yet.",
        "authority": "uncertain", "authority_reason": f"Has not encountered {SNAKE_NAME}'s rules yet.",
    }


def remember_description(npc):
    """The authored description is the only source of former-life knowledge."""
    from social import add_fact
    from character_data import description_parts
    from dialogue_text import split_sentences

    _, description = description_parts(npc.character["Details"])
    for index, memory in enumerate(split_sentences(description)):
        add_fact(npc.world, "former_life", npc, npc.world.snake,
                 f"{npc.id}:description:{index}", None,
                 f"From {npc.name}'s own description: {memory}", [npc])
