"""Shared speech contracts used by generation, interactions, history and saves."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from constants import DIALOGUE_MAX_PARTS


class StrictResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    # Computed locally, never requested from the model or trusted across the queue.
    _dialogue_errors: list[str] = PrivateAttr(default_factory=list)
    _authorized_intent: dict = PrivateAttr(default_factory=dict)
    _clarification: str = PrivateAttr(default="")


class EmotionReaction(StrictResponse):
    name: Literal[
        "neutral", "angry", "confused", "disappointed", "flustered",
        "giddy", "happy", "paranoid", "sad", "shifty", "shocked",
        "suspicious", "ticked_off", "tired", "unyuu",
    ] = "neutral"
    intensity: int = Field(default=0, ge=0, le=100)


class ConversationPart(StrictResponse):
    text: str = Field(min_length=1, description="Words spoken to the other NPC, never just a speaker name or role label.")
    emotion: EmotionReaction = Field(default_factory=EmotionReaction)

    @model_validator(mode="after")
    def validate_text(self):
        if not self.text.strip():
            raise ValueError("Conversation parts must contain text.")
        return self


class NameIntroduction(StrictResponse):
    actor_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    reference: str = Field(min_length=1, description="self for a self-introduction; otherwise one of the actor's supplied observable references.")


class DialoguePartsResponse(StrictResponse):
    introductions: list[NameIntroduction] = Field(default_factory=list, max_length=3)
    parts: list[ConversationPart] = Field(min_length=1, max_length=DIALOGUE_MAX_PARTS,
        description="Consecutive sections of this one speaker's uninterrupted turn for display and emotion changes. Every part has the same speaker. Never include the listener's answer or this speaker's later turns.")


class ConversationTurn(DialoguePartsResponse):
    speaker: Literal["initiator", "recipient"]

    @property
    def text(self):
        """Keep the complete spoken turn available to conversation history."""
        return " ".join(part.text.strip() for part in self.parts)
