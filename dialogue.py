"""Shared speech contracts used by generation, interactions, history and saves."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from constants import (DIALOGUE_MAX_PARTS, CONVERSATION_MAX_TURNS, CONVERSATION_MAX_AGREEMENTS,
                       CONVERSATION_MAX_DISCLOSURES, CONVERSATION_MAX_REPUTATION_EFFECTS,
                       CONVERSATION_MAX_LEVERAGE, SNAKE_TURN_COUNT, FOOD_REQUEST_TURN_COUNT)
from dialogue_text import contains_question
from purgatory import Attitudes


class StrictResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    # Computed locally, never requested from the model or trusted across the queue.
    _dialogue_errors: list[str] = PrivateAttr(default_factory=list)
    _authorized_intent: dict = PrivateAttr(default_factory=dict)
    _opinion_targets: dict[str, str] = PrivateAttr(default_factory=dict)
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


class DialoguePartsResponse(StrictResponse):
    parts: list[ConversationPart] = Field(min_length=1, max_length=DIALOGUE_MAX_PARTS,
        description="Consecutive sections of this one speaker's uninterrupted turn for display and emotion changes. Every part has the same speaker. Never include the listener's answer or this speaker's later turns.")


class ConversationTurn(DialoguePartsResponse):
    speaker: Literal["initiator", "recipient"]

    @property
    def text(self):
        """Keep the complete spoken turn available to conversation history."""
        return " ".join(part.text.strip() for part in self.parts)


class OpinionUpdate(StrictResponse):
    person: str = Field(min_length=1)
    opinion: int = Field(ge=-100, le=100)
    reason: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_text(self):
        if not self.person.strip() or not self.reason.strip():
            raise ValueError("Opinions require a person and a reason.")
        return self


class ActionIntent(StrictResponse):
    action: Literal["WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "TALK_TO_NPC", "GIVE_ITEM", "EAT", "SAY"]
    target: str = ""
    item: str = ""
    quantity: int = Field(default=0, ge=0)


class PlayerIntent(ActionIntent):
    message_type: Literal["COMMENT", "QUESTION", "COMMAND"]


class GiftOffer(StrictResponse):
    recipient_id: str
    item: str
    quantity: int = Field(ge=1)


class ResidentGift(GiftOffer):
    giver_id: str
    offer_turn: int = Field(ge=0)
    accept_turn: int | None = Field(default=None, ge=0)


class PlayerOpinionChange(StrictResponse):
    delta: int = Field(ge=-4, le=4)
    reason: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=200)


class PlayerApplePromise(StrictResponse):
    giver_id: str
    deadline: int = Field(ge=0)


class PlayerInvitation(StrictResponse):
    destination: Literal["garden", "tree", "home"]
    evidence: str = Field(min_length=1)


class ActionResponse(ActionIntent):
    reply: str = Field(description="Your words to the player. If end_conversation is true, give only a brief closing without a question or invitation to continue.")
    end_conversation: bool = Field(default=False, description="True when you finish talking without asking a question. If your SAY reply asks the player a question, this must be false so they can answer. End when the player is done or declines further conversation.")
    opinion_updates: list[OpinionUpdate] = Field(default_factory=list, max_length=8)
    emotion: EmotionReaction = Field(default_factory=EmotionReaction)
    gift_offer: GiftOffer | None = None
    gift_acceptance: bool = False
    player_opinion: PlayerOpinionChange | None = None
    apple_promise: PlayerApplePromise | None = None
    invitation: PlayerInvitation | None = None
    invitation_acceptance: str = ""
    invitation_resolution: Literal["", "declined", "withdrawn"] = ""
    invitation_resolution_evidence: str = ""

    @model_validator(mode="after")
    def close_for_action(self):
        # Committed tasks always give the player a final line to read first.
        if self.action != "SAY":
            self.end_conversation = True
        elif contains_question(self.reply):
            # Preserve generated speech; a question leaves room for an answer.
            self.end_conversation = False
        return self


class AcceptedProposal(StrictResponse):
    proposal_id: str = Field(min_length=1)
    offer_turn: int = Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)
    accept_turn: int = Field(ge=1, le=CONVERSATION_MAX_TURNS - 1)


class DisclosureChoice(StrictResponse):
    turn: int = Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)
    claim_id: str = Field(min_length=1)
    intent: Literal["share", "challenge", "lie"]
    listener_stance: Literal["uncertain", "believed", "disbelieved"]


class ThirdPartyOpinion(StrictResponse):
    turn: int = Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)
    claim_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    delta: int = Field(ge=-6, le=6)
    reason: str = Field(min_length=1, max_length=400)


class LeverageUse(StrictResponse):
    turn: int = Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)
    claim_id: str = Field(min_length=1)
    demand: Literal["apple", "repayment", "disclosure"]


class AccountDisclosure(StrictResponse):
    claim_id: str
    intent: Literal["share", "challenge", "lie"]


class AccountReaction(StrictResponse):
    disclosure_turn: int = Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)
    stance: Literal["uncertain", "believed", "disbelieved"]


class ReputationChoice(StrictResponse):
    claim_id: str
    subject_id: str
    delta: int = Field(ge=-6, le=6)
    reason: str = Field(min_length=1, max_length=400)


class LeverageChoice(StrictResponse):
    claim_id: str
    demand: Literal["apple", "repayment", "disclosure"]


class ResidentTurnResponse(DialoguePartsResponse):
    gift_offer: GiftOffer | None = None
    accept_gift: bool = False
    offers: list[str] = Field(default_factory=list, max_length=CONVERSATION_MAX_AGREEMENTS)
    acceptances: list[str] = Field(default_factory=list, max_length=CONVERSATION_MAX_AGREEMENTS)
    disclosures: list[AccountDisclosure] = Field(default_factory=list, max_length=1)
    reactions: list[AccountReaction] = Field(default_factory=list, max_length=CONVERSATION_MAX_DISCLOSURES)
    third_party_opinions: list[ReputationChoice] = Field(default_factory=list, max_length=CONVERSATION_MAX_REPUTATION_EFFECTS)
    leverage: list[LeverageChoice] = Field(default_factory=list, max_length=CONVERSATION_MAX_LEVERAGE)


class ResidentAssessment(StrictResponse):
    opinion: OpinionUpdate
    reactions: list[AccountReaction] = Field(default_factory=list, max_length=CONVERSATION_MAX_DISCLOSURES)


class SnakeResidentResponse(DialoguePartsResponse):
    account_id: str = ""
    attitudes: Attitudes | None = None


class SnakeDecision(StrictResponse):
    interaction_id: str
    decision: Literal["talk", "give_apple", "warn", "eat"]
    account_id: str = ""
    snake_account_id: str = ""
    account_stance: Literal["uncertain", "believed", "disbelieved"] = "uncertain"
    accusation: str = ""
    evidence: str = ""
    attitudes: Attitudes | None = None


class SnakeResponse(SnakeDecision):
    turns: list[ConversationTurn] = Field(min_length=SNAKE_TURN_COUNT, max_length=CONVERSATION_MAX_TURNS)
    outcome_turn: Literal[2, 4] = 2


class SnakePlayerResponse(SnakeDecision):
    interaction_id: str = ""
    decision: Literal["talk", "give_apple", "warn", "eat"] = "talk"
    reply: str = Field(min_length=1)
    disclosure_evidence: str = ""


class ConversationResponse(StrictResponse):
    gift: ResidentGift | None = None
    turns: list[ConversationTurn] = Field(min_length=FOOD_REQUEST_TURN_COUNT, max_length=CONVERSATION_MAX_TURNS,
        description="Conversation in chronological order. Each turn is one uninterrupted stretch of speech before the partner replies.")
    initiator_opinion: OpinionUpdate
    recipient_opinion: OpinionUpdate
    food_outcome: Literal["", "accepted", "need", "refused", "unavailable"] = ""
    handoff_thanks: list[ConversationPart] = Field(default_factory=list, max_length=3)
    accepted_proposals: list[AcceptedProposal] = Field(max_length=CONVERSATION_MAX_AGREEMENTS)
    disclosures: list[DisclosureChoice] = Field(max_length=CONVERSATION_MAX_DISCLOSURES)
    third_party_opinions: list[ThirdPartyOpinion] = Field(max_length=CONVERSATION_MAX_REPUTATION_EFFECTS)
    leverage: list[LeverageUse] = Field(max_length=CONVERSATION_MAX_LEVERAGE)
    orientation_discussed: bool = Field(default=False,
        description="True only if actual speech engages with arrival, surroundings, finding help or safety. This never solves the mystery or establishes a route home.")

    @model_validator(mode="after")
    def validate_turns(self):
        for index, turn in enumerate(self.turns):
            expected = "initiator" if index % 2 == 0 else "recipient"
            if turn.speaker != expected or not turn.text.strip():
                raise ValueError("Conversation must contain alternating, nonempty turns.")
        return self


class DialogueRepairResponse(DialoguePartsResponse):
    parts: list[ConversationPart] = Field(min_length=1, max_length=1)


class VoiceRewriteResponse(StrictResponse):
    turns: list[list[str]]
