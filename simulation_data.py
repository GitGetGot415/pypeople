"""Self-contained simulation saves. No character file paths are loaded."""

from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Annotated, Literal
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from character_data import normalize_character, normalize_opinions
from constants import (
    APPLE_ADVANCE_SECONDS, CONCERN_LIFETIME_RANGE, CONVERSATION_PAUSE_RANGE, CONVERSATION_RETURN_PAUSE, FAILURE_COOLDOWN,
    CONVERSATION_MAX_TURNS, CONVERSATION_MAX_AGREEMENTS, DECISION_SECONDS, DIALOGUE_TIMEOUT_SECONDS, EAT_SECONDS, GIFT_GIVE_SECONDS, GIFT_SECONDS,
    MAX_SIMULATION_STEP_SECONDS, SHAKE_SECONDS, SNAKE_EAT_SECONDS,
    POST_DIALOGUE_SECONDS, RECOVERY_SECONDS, SNAKE_FRAME_SECONDS, SNAKE_NAME, WORLD_HEIGHT, WORLD_WIDTH,
)
from creations import Creation, CreativeJob
from apartments import HOME_BEHAVIORS, INDOOR_BEHAVIORS, entrance
from constants import (APARTMENT_ID, BEDTIME_DELAY_RANGE, CREATION_WORK_SECONDS_RANGE, CREATIVE_QUEUE_SECONDS,
                       HOME_ARRIVAL_TOLERANCE, OUTDOOR_COOLDOWN_SECONDS, REST_SECONDS_RANGE, WAKE_DELAY_RANGE)
from conversation_history import ConversationRecord, Memory, WorldEvent
from llm import (AcceptedProposal, ConversationPart, ConversationTurn, DisclosureChoice, NameIntroduction,
                 EmotionReaction, GiftOffer, LeverageUse, ThirdPartyOpinion, ResidentGift)
from npc import ACTIVITY_FIELDS, Behavior, Conversation, Gift, NPC, Tree
from simulation import PendingAction, Simulation
from player import Player, PlayerHandoff
from constants import PLAYER_ID, PLAYER_LABEL, PLAYER_CALL_COOLDOWN, PLAYER_CALL_CHECK_SECONDS, PLAYER_CALL_LIMIT
from snake import Snake, SnakeEncounter
from purgatory import Attitudes
from social import (Belief, Claim, ConcernKind, ConversationConcern, Disclosure, Fact, FactKind, Goal, GoalKind, OpinionEffect, Promise, Proposal,
                    Stance, information_from_dict, phase_index, validate_agreements, validate_information)

SAVE_DIRECTORY = Path(__file__).resolve().parent / "saves"


class SaveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ActivityState(SaveModel):
    behavior: Behavior
    destination: tuple[float, float] | None
    pause_remaining: float = Field(ge=0)
    facing: int = Field(ge=0, le=7)
    owner: Literal["autonomous", "player_task", "player_hold"]
    target_id: str | None
    activity_elapsed: float = Field(ge=0)
    reason: str
    decision_remaining: float = Field(ge=0)
    cooldowns: dict[str, float]
    resume_pending: bool

    @model_validator(mode="after")
    def validate_activity(self):
        if any(value < 0 for value in self.cooldowns.values()):
            raise ValueError("Invalid cooldown duration.")
        if self.destination is not None and not (0 <= self.destination[0] <= WORLD_WIDTH and 0 <= self.destination[1] <= WORLD_HEIGHT):
            raise ValueError("Invalid destination.")
        if (self.owner == "player_hold") != (self.behavior == Behavior.STOP):
            raise ValueError("A player hold must be stopped.")
        player_behaviors = (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING)
        if self.behavior != Behavior.EAT and (self.owner == "player_task") != (self.behavior in player_behaviors):
            raise ValueError("Invalid player task.")
        if self.behavior == Behavior.EAT and (self.destination is not None or self.activity_elapsed >= EAT_SECONDS):
            raise ValueError("Invalid eating progress.")
        return self


class PreferencesState(SaveModel):
    sociability: float = Field(ge=0, le=1)
    generosity: float = Field(ge=0, le=1)


class MemoryState(SaveModel):
    id: str = Field(min_length=1)
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    partner_id: str
    kind: Literal["gave_fruit", "received_fruit", "refused", "kept_fruit", "conversation",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "heard_claim", "claim_opinion", "leverage", "arrival", "bargain", "endorsement", "defiance", "snake_claim", "judgment", "pardon", "eating", "goal_cancelled", "gave_work", "received_work", "visit"]
    summary: str = Field(max_length=400)
    weight: float = Field(ge=-12, le=12)


class EventState(SaveModel):
    id: str = Field(min_length=1)
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    kind: Literal["phase", "food_requested", "food_refused", "fruit_shared", "conversation_completed", "interaction_failed",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "claim_assessed", "reputation_changed", "leverage_used", "arrival", "bargain", "endorsement", "defiance", "snake_claim", "judgment", "pardon", "eating", "goal_cancelled", "promise_cancelled", "work_shared", "player_call", "player_conversation", "tree_queue", "travel_failed", "visit"]
    participants: list[str]
    text: str
    history: int | None = Field(ge=0)


class GoalState(SaveModel):
    id: str = Field(min_length=1)
    kind: GoalKind
    target_id: str
    objective: str = Field(min_length=1, max_length=400)
    motive: str = Field(min_length=1, max_length=400)
    approach: str = Field(min_length=1, max_length=400)
    boundary: str = Field(min_length=1, max_length=400)
    created_phase: int = Field(ge=0)
    deadline: int = Field(ge=0)
    baseline_opinion: int = Field(ge=-100, le=100)
    status: Literal["active", "completed", "failed", "cancelled"]
    resolution_event_id: str | None
    resolved_phase: int | None = Field(ge=0)
    outcome: str
    last_attempt_phase: int | None = Field(ge=0)
    third_npc_id: str | None
    claim_id: str | None


class ProposalState(SaveModel):
    id: str = Field(min_length=1)
    promisor_id: str
    recipient_id: str
    deadline: int = Field(ge=0)


class PromiseState(ProposalState):
    conversation_id: str = Field(min_length=1)
    created_phase: int = Field(ge=0)
    item: Literal["fruit"]
    quantity: Literal[1]
    status: Literal["pending", "fulfilled", "broken", "cancelled"]
    resolution_event_id: str | None
    resolved_phase: int | None = Field(ge=0)
    outcome: str
    consequence_event_id: str | None


class OpinionEffectState(SaveModel):
    event_id: str = Field(min_length=1)
    target_id: str
    delta: int = Field(ge=-6, le=6)
    reason: str = Field(min_length=1, max_length=400)
    claim_id: str | None
    actor_id: str | None
    conversation_id: str | None
    phase: int | None = Field(ge=0)


class FactState(SaveModel):
    id: str = Field(min_length=1)
    kind: FactKind
    subject_id: str
    object_id: str
    source_id: str = Field(min_length=1)
    conversation_id: str | None
    phase: int = Field(ge=0)
    summary: str = Field(min_length=1)
    witness_ids: list[str] = Field(min_length=1)


class ClaimState(SaveModel):
    id: str = Field(min_length=1)
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    deliberate: bool
    phase: int = Field(ge=0)
    conversation_id: str | None


class BeliefState(SaveModel):
    npc_id: str
    claim_id: str
    stance: Stance
    source_id: str
    disclosure_id: str | None
    direct: bool
    phase: int = Field(ge=0)
    assessed: bool
    opinion_delta: int = Field(ge=-6, le=6)


class DisclosureState(SaveModel):
    id: str = Field(min_length=1)
    claim_id: str
    conversation_id: str
    speaker_id: str
    listener_id: str
    turn: int = Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)
    intent: Literal["share", "challenge", "lie"]
    stance: Stance
    source_disclosure_id: str | None
    phase: int = Field(ge=0)
    event_id: str = Field(min_length=1)


class ClaimOptionState(SaveModel):
    id: str = Field(min_length=1)
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    creator_id: str | None
    kind: FactKind
    assertion: str = Field(min_length=1)


class InformationState(SaveModel):
    participants: dict[str, str]
    beliefs: dict[str, list[BeliefState]]
    options: list[ClaimOptionState]
    can_lie: dict[str, bool]
    opinions: dict[str, dict[str, int]]
    allowed: dict[str, list[str]]
    focus_claim_id: str | None


class DepartureState(SaveModel):
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    cause: Literal["eaten by the snake"]
    event_id: str = Field(min_length=1)


class LearnedNameState(SaveModel):
    name: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn: int = Field(ge=0)


class PlayerOpinionState(SaveModel):
    opinion: int = Field(strict=True, ge=-100, le=100)
    reason: str = Field(max_length=400)


class PlayerCallState(SaveModel):
    key: str = Field(min_length=1)
    kind: Literal["gift", "news", "followup"]
    reference: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    event_id: str = Field(min_length=1)


class PlayerReplyCallState(SaveModel):
    kind: Literal["reply"]
    reference: str = Field(min_length=1)
    closing: bool
    gift_offer: GiftOffer | None


class TransferIntentState(SaveModel):
    giver_id: str = Field(min_length=1)
    recipient_id: str = Field(min_length=1)
    item: str = Field(min_length=1)
    quantity: int = Field(strict=True, ge=1)


class TransferState(TransferIntentState):
    id: str = Field(min_length=1)
    day: int = Field(strict=True, ge=1)
    phase: Literal["day", "evening", "night"]
    sequence: int = Field(strict=True, ge=0)


class PlayerAppleOfferState(SaveModel):
    id: str = Field(min_length=1)
    giver_id: str = Field(min_length=1)
    recipient_id: str = Field(min_length=1)
    deadline: int = Field(strict=True, ge=0)
    created_phase: int = Field(strict=True, ge=0)
    offer_turn: int = Field(strict=True, ge=0)
    accept_turn: int | None = Field(strict=True, ge=0)
    status: Literal["offered", "accepted", "declined", "expired", "closed"]


class PlayerAcknowledgmentState(SaveModel):
    npc_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    transfer_id: str = Field(min_length=1)


class ConcernState(SaveModel):
    kind: ConcernKind
    claim_id: str | None
    remaining: float = Field(ge=0, le=CONCERN_LIFETIME_RANGE[1])

    @model_validator(mode="after")
    def validate_account_kind(self):
        if (self.kind in ("recollection", "news")) != bool(self.claim_id):
            raise ValueError("Only recollection and news concerns require an account.")
        return self


class NPCState(ActivityState):
    location: str
    journey: dict | None
    id: str = Field(min_length=1)
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    reply: str
    rewritten_reply: str = ""
    emotion: EmotionReaction
    social: float = Field(ge=0, le=100)
    inventory: dict[Literal["fruit"], Annotated[int, Field(strict=True, ge=0)]]
    inventory_capacity: int | None = Field(strict=True, ge=1)
    preferences: PreferencesState
    memories: list[MemoryState] = Field(max_length=20)
    goal: GoalState | None
    pending_social_opinions: list[OpinionEffectState]
    beliefs: list[BeliefState]
    current_concern: ConcernState | None
    conversation_pause_remaining: float = Field(ge=0, le=max(CONVERSATION_PAUSE_RANGE[1], FAILURE_COOLDOWN))
    conversation_return_remaining: float = Field(ge=0, le=CONVERSATION_RETURN_PAUSE)
    known_names: dict[str, LearnedNameState] = Field(default_factory=dict)
    player_opinion: dict
    player_call: dict | None
    player_call_cooldown: float = Field(ge=0, le=PLAYER_CALL_COOLDOWN)
    resolved_player_calls: list[str]
    command_revision: int = Field(ge=0)
    attitudes: Attitudes
    arrival_day: int = Field(ge=1)
    departure: DepartureState | None
    sprite_size: tuple[int, int] | None
    room_number: int = Field(strict=True, ge=1)
    social_fatigue: float = Field(ge=0, le=100)
    home_remaining: float = Field(ge=0, le=max(REST_SECONDS_RANGE[1], CREATION_WORK_SECONDS_RANGE[1]))
    outdoor_remaining: float = Field(ge=0, le=OUTDOOR_COOLDOWN_SECONDS)
    bedtime_remaining: float | None = Field(ge=0, le=BEDTIME_DELAY_RANGE[1])
    wake_remaining: float | None = Field(ge=0, le=WAKE_DELAY_RANGE[1])
    home_reason: Literal["night", "retreat"] | None
    last_creative_day: int = Field(strict=True, ge=0)
    creative_job: CreativeJob | None

    @field_validator("character")
    @classmethod
    def validate_character(cls, value):
        return normalize_character(value)

    @field_validator("player_opinion")
    @classmethod
    def validate_player_opinion(cls, value):
        return PlayerOpinionState.model_validate(value).model_dump()

    @field_validator("player_call")
    @classmethod
    def validate_player_call(cls, value):
        if value is None:
            return None
        model = PlayerReplyCallState if value.get("kind") == "reply" else PlayerCallState
        return model.model_validate(value).model_dump()


class HistoryState(SaveModel):
    id: str = Field(min_length=1)
    participants: tuple[str, str]
    topic: str
    participant_ids: list[str]
    transfers: list[dict]
    accepted_player_turns: list[Annotated[int, Field(strict=True, ge=0)]]
    authorization_id: str | None
    transfer_intent: dict | None
    player_promise_offers: list[dict]

    @field_validator("transfers")
    @classmethod
    def validate_transfers(cls, value):
        return [TransferState.model_validate(row).model_dump() for row in value]

    @field_validator("player_promise_offers")
    @classmethod
    def validate_player_promise_offers(cls, value):
        return [PlayerAppleOfferState.model_validate(row).model_dump() for row in value]

    @field_validator("transfer_intent")
    @classmethod
    def validate_transfer_intent(cls, value):
        return TransferIntentState.model_validate(value).model_dump() if value is not None else None
    intended_subjects: dict[str, str]
    status: str
    turns: list[tuple[str, str]]
    dialogue_parts: dict[int, list[ConversationPart]] = Field(default_factory=dict)
    rewritten_parts: dict[int, list[str]] = Field(default_factory=dict)
    failed_turns: list[ConversationTurn] = Field(default_factory=list)
    name_forms: list[str] = Field(default_factory=list)
    name_introductions: dict[int, list[NameIntroduction]] = Field(default_factory=dict)
    name_witnesses: dict[int, list[str]] = Field(default_factory=dict)
    orientation_discussed: bool = False
    dialogue_errors: list[str] = Field(default_factory=list)
    outcomes: list[str]
    error: str
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "orientation", "food_request", "gossip", "bargain", "judgment", "introduction", "snake_player"]

    @model_validator(mode="after")
    def validate_rewrites(self):
        for index, texts in self.rewritten_parts.items():
            if (index < 0 or index >= len(self.turns) or self.turns[index][0] == "You"
                    or len(texts) != len(self.dialogue_parts.get(index, []))
                    or not texts or any(not text.strip() for text in texts)):
                raise ValueError("A rewrite must match the original NPC dialogue parts.")
        return self


class ConversationState(SaveModel):
    accepted_gift: ResidentGift | None
    id: str = Field(min_length=1)
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "food_request"]
    concerns: dict[Literal["initiator", "recipient"], ConcernState]
    opening_kind: ConcernKind | None
    goal_id: str | None
    focus_claim_id: str | None
    dialogue_state: Literal["waiting", "pending", "ready"]
    wait_elapsed: float = Field(ge=0, le=DIALOGUE_TIMEOUT_SECONDS + MAX_SIMULATION_STEP_SECONDS)
    talking_remaining: float = Field(ge=0, le=max(POST_DIALOGUE_SECONDS, DECISION_SECONDS))
    food_outcome: Literal["", "accepted", "need", "refused", "unavailable"]
    food_reason: str
    handoff_thanks: list[ConversationPart] = Field(max_length=3)
    rewritten_handoff_thanks: list[str] = Field(default_factory=list)
    outcome_event_id: str | None
    reactions: list[tuple[str, float]] = Field(max_length=2)
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    topic: str
    arrived: bool
    turns: list[ConversationTurn] = Field(max_length=CONVERSATION_MAX_TURNS)
    saved_states: tuple[ActivityState, ActivityState]
    pending_opinions: list[tuple[int, dict]]
    proposals: list[ProposalState] = Field(max_length=2)
    accepted_proposals: list[AcceptedProposal] = Field(max_length=2)
    social_effects_applied: bool
    context_references: dict[str, Literal["active", "completed", "failed", "pending", "fulfilled", "broken", "cancelled"]]
    information_context: InformationState | None
    accepted_disclosures: list[DisclosureChoice] = Field(max_length=4)
    third_party_opinions: list[ThirdPartyOpinion] = Field(max_length=4)
    leverage_uses: list[LeverageUse] = Field(max_length=2)

    @field_validator("pending_opinions")
    @classmethod
    def validate_opinions(cls, value):
        return [(index, normalize_opinions(opinions)) for index, opinions in value]


class GiftState(SaveModel):
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    item: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    arrived: bool
    elapsed: float = Field(ge=0, lt=GIFT_SECONDS)
    transferred: bool
    outcome_event_id: str | None
    thanks: list[ConversationPart] = Field(max_length=3)
    rewritten_thanks: list[str] = Field(default_factory=list)
    saved_states: tuple[ActivityState, ActivityState]


class TreeState(SaveModel):
    id: str = Field(min_length=1)
    x: float = Field(ge=60, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT - 50)
    fruit: int = Field(ge=0)
    shake_elapsed: float | None = Field(ge=0, le=SHAKE_SECONDS)
    shaker: str | None
    reserved_by: str | None


class SnakeRetryState(SaveModel):
    resident_id: str
    purpose: Literal["bargain", "judgment", "introduction"]
    reason: str
    origin: Literal["player", "autonomous"]
    error: str


class AppleAdvanceState(SaveModel):
    offering: Literal["secret", "endorsement"]
    account_id: str
    conversation_id: str
    remaining: float = Field(ge=0, le=APPLE_ADVANCE_SECONDS)


class SnakeState(SaveModel):
    known_names: dict[str, LearnedNameState]
    id: str = Field(min_length=1)
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    destination: tuple[float, float] | None
    facing: int = Field(ge=0, le=7)
    pause_remaining: float = Field(ge=0)
    animation_elapsed: float = Field(ge=0, lt=2 * SNAKE_FRAME_SECONDS)
    activity: Literal["guarding", "recovering", "judging", "bargaining", "introducing", "approaching condemned resident", "eating"]
    target_id: str | None
    beliefs: list[BeliefState]
    standing: dict[str, int]
    offenses: dict[str, str]
    advances: dict[str, AppleAdvanceState] = Field(default_factory=dict)
    eaten_today: int = Field(ge=0)
    recovery_remaining: float = Field(ge=0, le=RECOVERY_SECONDS)
    pardons: list[str]
    judged_phase: int = Field(ge=-1)
    decision_remaining: float = Field(ge=0)
    command_revision: int = Field(ge=0)
    reply: str
    rewritten_reply: str = ""
    retry_request: SnakeRetryState | None

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value):
        if value is not None and not (0 <= value[0] <= WORLD_WIDTH and 0 <= value[1] <= WORLD_HEIGHT):
            raise ValueError("Invalid snake destination.")
        return value


class SnakeEncounterState(SaveModel):
    id: str
    resident_id: str
    record: int = Field(ge=0)
    purpose: Literal["bargain", "judgment", "introduction"]
    origin: Literal["player", "autonomous"]
    topic: str
    accusation: str
    saved_state: ActivityState
    reserved: bool
    arrived: bool
    dialogue_state: Literal["waiting", "pending", "ready"]
    wait_elapsed: float = Field(ge=0, le=DIALOGUE_TIMEOUT_SECONDS + MAX_SIMULATION_STEP_SECONDS)
    talking_remaining: float = Field(ge=0, le=max(POST_DIALOGUE_SECONDS, DECISION_SECONDS))
    committed: bool
    outcome: Literal["", "talk", "advance_secret", "advance_endorsement", "secret", "endorsement", "refuse", "negotiate", "challenge", "spare", "eat"]
    outcome_event_id: str | None
    stage: Literal["approach", "waiting", "dialogue", "condemned_approach", "eating"]
    elapsed: float = Field(ge=0, le=SNAKE_EAT_SECONDS + MAX_SIMULATION_STEP_SECONDS)
    turns: list[ConversationTurn] = Field(max_length=CONVERSATION_MAX_TURNS)
    request_context: dict | None


class PendingActionState(SaveModel):
    npc_id: str
    action: Literal["WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "TALK_TO_NPC", "GIVE_ITEM", "EAT"]
    revision: int = Field(ge=0)
    target_id: str | None
    topic: str
    record: int | None = Field(ge=0)
    item: str = ""
    quantity: int = Field(default=0, ge=0)


class PlayerState(SaveModel):
    id: Literal[PLAYER_ID]
    inventory: dict[Literal["fruit"], Annotated[int, Field(strict=True, ge=0)]]
    inventory_capacity: int = Field(strict=True, ge=1)
    x: float
    y: float
    facing: int = Field(ge=0, le=7)
    location: str
    alive: bool
    appearance: dict
    journey: dict | None

    @model_validator(mode="after")
    def valid_apples(self):
        if set(self.inventory) != {"fruit"} or not 0 <= self.inventory["fruit"] <= self.inventory_capacity:
            raise ValueError("Invalid player apple inventory.")
        return self


class PlayerHandoffState(SaveModel):
    initiator: str
    recipient: str
    record: int = Field(ge=0)
    item: str = Field(min_length=1)
    quantity: int = Field(strict=True, ge=1)
    elapsed: float = Field(ge=0, lt=GIFT_SECONDS)
    transferred: bool
    outcome_event_id: str | None
    saved_state: ActivityState


class SimulationState(SaveModel):
    mode: Literal["player", "spectator"]
    player_interaction: dict | None
    player_invitations: list[dict]
    player_claims: list[dict]
    visit: dict | None
    tree_queue: list[dict]
    tree_current: dict | None
    player_snake_encounter: dict | None
    day: int = Field(ge=1)
    player: PlayerState
    player_handoffs: list[PlayerHandoffState]
    player_call_check: float = Field(ge=0, le=PLAYER_CALL_CHECK_SECONDS)
    player_acknowledgments: list[dict]
    npc_trait_mode: Literal["normal", "extreme", "both"] = "normal"
    phase: Literal["day", "evening", "night"]
    events: list[EventState] = Field(max_length=200)
    promises: list[PromiseState]
    facts: list[FactState]
    claims: list[ClaimState]
    disclosures: list[DisclosureState]
    social_rotation: int = Field(ge=0)
    npcs: list[NPCState] = Field(min_length=1)
    creations: list[Creation]
    tree: TreeState
    snake: SnakeState
    archived: list[NPCState]
    arrival_error: str
    pending_arrival_days: list[int]
    processed_arrival_days: list[int]
    introductions: list[str]
    roster_revision: int = Field(ge=0)
    snake_encounter: SnakeEncounterState | None
    pending_actions: list[PendingActionState]
    history: list[HistoryState]
    conversations: list[ConversationState]
    gifts: list[GiftState] = Field(default_factory=list)
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(default=1.0, ge=0.5, le=2.5)

    @field_validator("player_acknowledgments")
    @classmethod
    def validate_player_acknowledgments(cls, value):
        return [PlayerAcknowledgmentState.model_validate(row).model_dump() for row in value]

    @model_validator(mode="after")
    def validate_references(self):
        try:
            validate_physical_state(self)
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError("Invalid physical state: required ownership, location or source data is missing.") from exc
        residents = [*self.npcs, *self.archived]
        live_ids = {n.id for n in self.npcs}
        indoor_ids = {n.id for n in self.npcs if n.location != "garden"}
        if len({n.room_number for n in residents}) != len(residents):
            raise ValueError("Apartment rooms must be unique, including departed assignments.")
        creation_ids, job_ids = set(), set()
        owner_ids = {self.player.id, *[n.id for n in residents]}
        for work in self.creations:
            if (work.id in creation_ids or work.job_id in job_ids or work.creator_id not in {n.id for n in residents}
                    or work.owner_id not in owner_ids or phase_index(work.day, work.phase) > phase_index(self.day, self.phase)):
                raise ValueError("Invalid creation identity, credit, or completion time.")
            creation_ids.add(work.id)
            job_ids.add(work.job_id)
        for n in residents:
            if n.last_creative_day > self.day or n.sprite_size is not None and any(size <= 0 for size in n.sprite_size):
                raise ValueError("Invalid resident home state.")
            if n.departure and (n.creative_job or n.behavior in HOME_BEHAVIORS):
                raise ValueError("Departed residents cannot keep active home work.")
            if n.room_number == 1:
                raise ValueError("Room 1 is reserved for the player.")
            from locations import feet_bounds, valid_location
            layout = SimpleNamespace(residents=residents)
            if not valid_location(layout, n.location):
                raise ValueError("Unknown resident location.")
            left, right, top, bottom = feet_bounds(n.location)
            if not left <= n.x <= right or not top <= n.y <= bottom:
                raise ValueError("Resident feet are outside the current map.")
            if n.behavior == Behavior.GO_HOME and not n.journey and n.location != f"room:{n.room_number}":
                raise ValueError("A home trip needs a journey to the owned room.")
            if n.behavior in INDOOR_BEHAVIORS and n.location != f"room:{n.room_number}" and not n.journey:
                raise ValueError("Home routines require the resident's real room.")
            job = n.creative_job
            effective_behavior = n.behavior
            frozen = (self.visit["saved_state"] if self.visit and self.visit["host_id"] == n.id else
                      self.player_interaction.get("saved_state") if self.player_interaction and self.player_interaction.get("npc_id") == n.id else None)
            for conversation in self.conversations:
                for index, activity in zip((conversation.initiator, conversation.recipient), conversation.saved_states):
                    if index < len(self.npcs) and self.npcs[index].id == n.id:
                        frozen = activity.model_dump()
            if frozen:
                effective_behavior = frozen["behavior"]
            if (effective_behavior in INDOOR_BEHAVIORS and n.location != f"room:{n.room_number}"
                    and not n.journey and not (self.visit and self.visit["host_id"] == n.id)):
                raise ValueError("Suspended home work must retain its actual room or accepted visit.")
            if (job is None) != (effective_behavior not in (Behavior.PAINT, Behavior.COMPOSE)):
                raise ValueError("An active hobby needs exactly one creative job.")
            if job:
                if (job.id in job_ids or job.creator_id != n.id or n.id not in live_ids
                        or n.command_revision != job.revision or job.started_day != n.last_creative_day
                        or phase_index(job.started_day, job.started_phase) > phase_index(self.day, self.phase)
                        or job.remaining > CREATION_WORK_SECONDS_RANGE[1] or job.queue_remaining > CREATIVE_QUEUE_SECONDS
                        or effective_behavior != (Behavior.PAINT if job.kind == "art" else Behavior.COMPOSE)):
                    raise ValueError("Invalid active creative job or completed job reuse.")
                job_ids.add(job.id)
        actors = {n.id: (n.display_name or n.character["Name"]) for n in residents}
        actors[self.snake.id] = SNAKE_NAME
        authored_names = {actors[n.id]: n.character["Name"] for n in residents}
        authored_names[SNAKE_NAME] = SNAKE_NAME
        ids = set(actors)
        now = phase_index(self.day, self.phase)
        if len(ids) != len(residents) + 1 or self.tree.id in ids or len({v.casefold() for v in actors.values()}) != len(ids):
            raise ValueError("Actor IDs and stable display names must be unique.")
        if (self.player.id in ids or self.tree.id == self.player.id
                or PLAYER_LABEL.casefold() in {name.casefold() for name in actors.values()}):
            raise ValueError("Player identity conflicts with a physical actor.")
        actors[self.player.id] = PLAYER_LABEL
        ids.add(self.player.id)
        if self.social_rotation >= len(self.npcs):
            raise ValueError("Invalid social rotation.")
        if any(n.departure is not None for n in self.npcs) or any(n.departure is None for n in self.archived):
            raise ValueError("Live and departed rosters disagree.")
        history = {h.id: h for h in self.history}
        events = {e.id: e for e in self.events}
        facts = {f.id: f for f in self.facts}
        claims = {c.id: c for c in self.claims}
        disclosures = {d.id: d for d in self.disclosures}
        for index, source in ((history, self.history), (events, self.events), (facts, self.facts), (claims, self.claims), (disclosures, self.disclosures)):
            if len(index) != len(source):
                raise ValueError("Duplicate social record IDs.")
        if len({(f.source_id, f.kind) for f in self.facts}) != len(self.facts):
            raise ValueError("Duplicate fact sources.")
        def record_for(record_id):
            record = history.get(record_id)
            if record is None:
                raise ValueError("Missing source conversation.")
            return record
        def participant(actor_id, record):
            return actor_id in ids and actors[actor_id] in record.participants
        from name_knowledge import asks_name, introduction_supported
        def introduction_evidence(record, index, intro):
            previous = record.turns[index - 1] if index else None
            requested = bool(previous and previous[0] != record.turns[index][0]
                and asks_name(previous[1], self_name=authored_names.get(previous[0], "")))
            return introduction_supported(record.turns[index][1], intro.name, intro.reference, name_requested=requested)
        for event in self.events:
            if not set(event.participants) <= ids or phase_index(event.day, event.phase) > now:
                raise ValueError("Invalid event actors or date.")
            if event.history is not None and event.history >= len(self.history):
                raise ValueError("Invalid event history link.")
        transfer_ids, transfer_order, player_offer_ids = set(), [], set()
        for record in self.history:
            if len(record.player_promise_offers) > CONVERSATION_MAX_AGREEMENTS:
                raise ValueError("Too many player apple offers in one conversation.")
            for offer in record.player_promise_offers:
                if (set(offer) != {"id", "giver_id", "recipient_id", "deadline", "offer_turn", "accept_turn", "created_phase", "status"}
                        or offer["id"] in player_offer_ids
                        or offer["giver_id"] not in owner_ids or offer["recipient_id"] not in owner_ids
                        or offer["giver_id"] == offer["recipient_id"]
                        or self.player.id not in (offer["giver_id"], offer["recipient_id"])
                        or offer["status"] not in ("offered", "accepted", "declined", "expired", "closed")
                        or not 0 <= offer["created_phase"] <= now or offer["deadline"] < offer["created_phase"]
                        or not 0 <= offer["offer_turn"] < len(record.turns)
                        or record.turns[offer["offer_turn"]][0] != actors[offer["giver_id"]]):
                    raise ValueError("Invalid player apple offer authority or source.")
                player_offer_ids.add(offer["id"])
                from dialogue_rules import validate_apple_offer, validate_apple_promise
                validate_apple_offer(record.turns[offer["offer_turn"]][1], offer["deadline"], current_phase=offer["created_phase"])
                if offer["status"] == "accepted":
                    turn = offer["accept_turn"]
                    if (type(turn) is not int or not offer["offer_turn"] < turn < len(record.turns)
                            or record.turns[turn][0] != actors[offer["recipient_id"]]
                            or not any(p.id == offer["id"] and p.conversation_id == record.id
                                       and p.promisor_id == offer["giver_id"] and p.recipient_id == offer["recipient_id"]
                                       and p.deadline == offer["deadline"] for p in self.promises)):
                        raise ValueError("An accepted player apple promise needs its actual ordered acceptance.")
                    validate_apple_promise(record.turns[offer["offer_turn"]][1], record.turns[turn][1],
                                           offer["deadline"], current_phase=offer["created_phase"])
                elif offer["accept_turn"] is not None:
                    raise ValueError("An unaccepted apple offer cannot retain an acceptance.")
            if record.authorization_id is not None and record.authorization_id not in history:
                raise ValueError("Missing transfer authorization conversation.")
            if record.participant_ids and (len(record.participant_ids) != 2 or not set(record.participant_ids) <= ids
                    or len(set(record.participant_ids)) != 2
                    or {actors[i] for i in record.participant_ids} != set(record.participants)):
                raise ValueError("Historical participant identities do not match the speakers.")
            if (len(set(record.accepted_player_turns)) != len(record.accepted_player_turns)
                    or any(i >= len(record.turns) or record.turns[i][0] != PLAYER_LABEL for i in record.accepted_player_turns)):
                raise ValueError("Player encounter effects need unique accepted player turns.")
            for transfer in record.transfers:
                if (set(transfer) != {"id", "giver_id", "recipient_id", "item", "quantity", "day", "phase", "sequence"}
                        or not isinstance(transfer["id"], str) or transfer["id"] in transfer_ids
                        or type(transfer["sequence"]) is not int or transfer["sequence"] < 0
                        or transfer["giver_id"] not in owner_ids or transfer["recipient_id"] not in owner_ids
                        or transfer["giver_id"] == transfer["recipient_id"] or type(transfer["quantity"]) is not int
                        or transfer["quantity"] < 1 or transfer["item"] != "fruit" and (
                            transfer["item"] not in creation_ids or transfer["quantity"] != 1)
                        or phase_index(transfer["day"], transfer["phase"]) > now):
                    raise ValueError("Invalid committed transfer provenance.")
                transfer_ids.add(transfer["id"])
                transfer_order.append(transfer)
            if any(name != "You" and name not in actors.values() for name in record.participants):
                raise ValueError("Unknown historical speaker.")
            if (not set(record.intended_subjects) <= set(record.participants)
                    or any(not subject.strip() for subject in record.intended_subjects.values())):
                raise ValueError("Intended subjects must describe participants in their recorded conversation.")
            if any(speaker not in record.participants or not text.strip() for speaker, text in record.turns):
                raise ValueError("Invalid historical speech.")
            for index, parts in record.dialogue_parts.items():
                if index >= len(record.turns) or " ".join(p.text.strip() for p in parts) != record.turns[index][1]:
                    raise ValueError("Dialogue parts do not match their recorded turn.")
            for index, introductions in record.name_introductions.items():
                if not 0 <= index < len(record.turns):
                    raise ValueError("Invalid introduction turn.")
                # Diagnosed metadata is retained even when its name effect was
                # unusable. Learned-name references are checked separately below.
                if record.dialogue_errors:
                    continue
                if len({i.actor_id for i in introductions}) != len(introductions):
                    raise ValueError("Invalid introduction turn or duplicate introduced actor.")
                for intro in introductions:
                    missing_speech = not introduction_evidence(record, index, intro)
                    if (intro.actor_id not in ids or missing_speech
                            or intro.reference == "self" and record.turns[index][0] != actors[intro.actor_id]):
                        raise ValueError("Invalid introduction identity or spoken evidence.")
            for index, witnesses in record.name_witnesses.items():
                if (not 0 <= index < len(record.turns) or not set(witnesses) <= ids
                        or not all(any((f.kind == "endorsement" and index == 1 or f.kind == "snake_claim" and index == 0)
                                       and f.conversation_id == record.id
                                       and witness in f.witness_ids for f in self.facts) for witness in witnesses)):
                    raise ValueError("A witnessed introduction requires a recorded hearing event.")
        if sorted(t["sequence"] for t in transfer_order) != list(range(len(transfer_order))):
            raise ValueError("Transfer history must retain its unique commit order.")
        owners = {w.id: w.creator_id for w in self.creations}
        for transfer in sorted(transfer_order, key=lambda t: t["sequence"]):
            if transfer["item"] != "fruit":
                if owners[transfer["item"]] != transfer["giver_id"]:
                    raise ValueError("Transfer provenance does not follow the actual previous owner.")
                owners[transfer["item"]] = transfer["recipient_id"]
        if any(owners[w.id] != w.owner_id for w in self.creations):
            raise ValueError("Current work ownership must match its committed transfer history.")
        for ack in self.player_acknowledgments:
            record = history.get(ack.get("record_id"))
            if (set(ack) != {"npc_id", "record_id", "transfer_id"} or ack["npc_id"] not in live_ids
                    or record is None or not any(t["id"] == ack["transfer_id"]
                        and {t["giver_id"], t["recipient_id"]} == {self.player.id, ack["npc_id"]} for t in record.transfers)):
                raise ValueError("A pending acknowledgment needs its committed player handoff.")
        if len({a["transfer_id"] for a in self.player_acknowledgments}) != len(self.player_acknowledgments):
            raise ValueError("Duplicate pending transfer acknowledgment.")
        for fact in self.facts:
            if (fact.subject_id not in ids or fact.object_id not in ids or fact.subject_id == fact.object_id
                    or fact.phase > now or not set(fact.witness_ids) <= ids
                    or len(set(fact.witness_ids)) != len(fact.witness_ids)):
                raise ValueError("Invalid fact participants or witnesses.")
            if fact.kind in ("private_motive", "former_life"):
                if fact.witness_ids != [fact.subject_id] or fact.conversation_id is not None:
                    raise ValueError("A private recollection or motive cannot be public.")
            elif fact.conversation_id is not None:
                record_for(fact.conversation_id)
            if fact.source_id in events:
                event = events[fact.source_id]
                expected = {"apple_given": "fruit_shared", "work_given": "work_shared", "apple_refused": "food_refused", "promise_kept": "promise_fulfilled"}.get(fact.kind, fact.kind)
                if event.kind != expected or fact.phase != phase_index(event.day, event.phase):
                    raise ValueError("A fact must match its actual source event.")
        for claim in self.claims:
            fact = facts.get(claim.fact_id)
            if fact is None or claim.subject_id != fact.subject_id or claim.originator_id not in ids or not fact.phase <= claim.phase <= now:
                raise ValueError("Invalid claim references.")
            if claim.parent_claim_id is None:
                if not claim.affirmed or claim.deliberate or claim.originator_id not in fact.witness_ids:
                    raise ValueError("Firsthand accounts require an actual witnessed source.")
            else:
                parent = claims.get(claim.parent_claim_id)
                if parent is None or parent.fact_id != fact.id or not parent.affirmed or claim.affirmed or not claim.deliberate or parent.phase > claim.phase:
                    raise ValueError("A denial must trace to an existing account.")
        seen = set()
        for disclosure in self.disclosures:
            record = record_for(disclosure.conversation_id)
            if (disclosure.claim_id not in claims or disclosure.speaker_id == disclosure.listener_id
                    or not participant(disclosure.speaker_id, record) or disclosure.listener_id not in ids
                    or disclosure.turn >= len(record.turns) or record.turns[disclosure.turn][0] != actors[disclosure.speaker_id]
                    or disclosure.phase > now):
                raise ValueError("Invalid disclosure actor, speech or account.")
            if disclosure.source_disclosure_id:
                source = disclosures.get(disclosure.source_disclosure_id)
                claim = claims[disclosure.claim_id]
                if source is None or source.id not in seen or source.listener_id != disclosure.speaker_id or source.claim_id not in (claim.id, claim.parent_claim_id):
                    raise ValueError("Disclosure provenance must follow an earlier source.")
            elif disclosure.speaker_id != claims[disclosure.claim_id].originator_id:
                fact = facts[claims[disclosure.claim_id].fact_id]
                if disclosure.speaker_id not in fact.witness_ids:
                    raise ValueError("A repeated account requires its actual source.")
            seen.add(disclosure.id)
        for actor in [*residents, self.snake]:
            if len({b.claim_id for b in actor.beliefs}) != len(actor.beliefs):
                raise ValueError("Duplicate owned account.")
            for belief in actor.beliefs:
                claim = claims.get(belief.claim_id)
                if claim is None or belief.npc_id != actor.id or belief.source_id not in ids or belief.phase > now:
                    raise ValueError("Invalid owned knowledge.")
                if belief.direct:
                    if actor.id not in facts[claim.fact_id].witness_ids or belief.disclosure_id is not None:
                        raise ValueError("Firsthand knowledge requires witnessing.")
                elif belief.disclosure_id:
                    source = disclosures.get(belief.disclosure_id)
                    if source is None or source.listener_id != actor.id or source.claim_id != claim.id or source.speaker_id != belief.source_id:
                        raise ValueError("Hearsay requires its actual disclosure source.")
        goal_ids = set()
        def validate_concern(concern, owner):
            if concern is None:
                return
            if concern.claim_id:
                claim = claims.get(concern.claim_id)
                belief = next((b for b in owner.beliefs if b.claim_id == concern.claim_id), None)
                if claim is None or belief is None:
                    raise ValueError("A concern requires its owner's existing knowledge.")
                fact = facts[claim.fact_id]
                if concern.kind == "recollection" and not (belief.direct and claim.affirmed and not claim.deliberate
                        and fact.kind == "former_life" and fact.subject_id == owner.id):
                    raise ValueError("A recollection concern requires the owner's direct former-life account.")
        for npc in residents:
            if npc.id in live_ids and npc.current_concern is None:
                raise ValueError("A living resident must have a concern.")
            validate_concern(npc.current_concern, npc)
            from name_knowledge import forms
            for target_id, learned in npc.known_names.items():
                learned = learned.model_dump()
                target = next((n for n in residents if n.id == target_id), None)
                actual_name = (learned["name"] if target_id == self.player.id else
                               SNAKE_NAME if target_id == self.snake.id else target.character["Name"] if target else "")
                if (target_id == npc.id or not actual_name or set(learned) != {"name", "conversation_id", "turn"}
                        or learned["name"] not in forms(actual_name)):
                    raise ValueError("Invalid learned name or actor reference.")
                source = record_for(learned["conversation_id"])
                turn = learned["turn"]
                if (type(turn) is not int or not 0 <= turn < len(source.turns)
                        or actors[npc.id] not in source.participants and npc.id not in source.name_witnesses.get(turn, [])):
                    raise ValueError("A learned name requires accepted spoken source evidence.")
                if not any(intro.actor_id == target_id and intro.name == learned["name"]
                           and (source.dialogue_errors or introduction_evidence(source, turn, intro))
                           and (intro.reference != "self" or source.turns[turn][0] == actors[target_id])
                           and source.turns[turn][0] != actors[npc.id]
                           for intro in source.name_introductions.get(turn, [])):
                    raise ValueError("A learned name must be supported by its source speech.")
            if npc.arrival_day > self.day or any(m.partner_id not in ids or phase_index(m.day, m.phase) > now for m in npc.memories):
                raise ValueError("Invalid arrival day or personal memory.")
            if npc.departure and (phase_index(npc.departure.day, npc.departure.phase) > now or not any(f.source_id == npc.departure.event_id and f.kind == "eating" and f.object_id == npc.id for f in self.facts)):
                raise ValueError("An archived resident needs its actual departure record.")
            goal = npc.goal
            if goal:
                if goal.id in goal_ids or goal.target_id not in ids or goal.third_npc_id is not None and goal.third_npc_id not in ids or goal.created_phase > now or goal.deadline < goal.created_phase:
                    raise ValueError("Invalid goal references.")
                goal_ids.add(goal.id)
                if goal.status == "active" and (npc.id not in live_ids or goal.target_id not in live_ids or goal.third_npc_id is not None and goal.third_npc_id not in live_ids):
                    raise ValueError("An active goal can target only living residents.")
                if (goal.status == "active") != (goal.resolved_phase is None):
                    raise ValueError("Invalid goal resolution.")
                if goal.claim_id is not None and goal.claim_id not in claims:
                    raise ValueError("Unknown goal account.")
            for effect in npc.pending_social_opinions:
                if effect.target_id not in ids or effect.claim_id is not None and effect.claim_id not in claims:
                    raise ValueError("Invalid queued social effect.")
        promise_ids = set()
        for promise in self.promises:
            record = record_for(promise.conversation_id)
            if promise.id in promise_ids or promise.promisor_id == promise.recipient_id or not all(participant(i, record) for i in (promise.promisor_id, promise.recipient_id)):
                raise ValueError("Invalid promise participants.")
            promise_ids.add(promise.id)
            if promise.status == "pending" and not {promise.promisor_id, promise.recipient_id} <= live_ids | {self.player.id}:
                raise ValueError("Impossible obligations must be cancelled.")
            if promise.deadline < promise.created_phase or promise.created_phase > now or (promise.status == "pending") != (promise.resolved_phase is None):
                raise ValueError("Invalid promise resolution.")
        def activity_target(activity, owner):
            target = activity.target_id
            if activity.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING, Behavior.HARVEST):
                if target != self.tree.id:
                    raise ValueError("Invalid tree task target.")
            elif activity.behavior == Behavior.GO_HOME:
                if target != APARTMENT_ID:
                    raise ValueError("Invalid apartment target.")
            elif activity.behavior == Behavior.COMPANY:
                if target not in live_ids or target == owner.id:
                    raise ValueError("A company target must be living.")
            elif target is not None:
                raise ValueError("Unexpected action target.")
        reserved = set()
        for handoff in self.player_handoffs:
            pair = {handoff.initiator, handoff.recipient}
            resident_id = next(iter(pair - {self.player.id}), None)
            if (len(pair) != 2 or self.player.id not in pair or resident_id not in live_ids
                    or handoff.record >= len(self.history)
                    or next(n for n in self.npcs if n.id == resident_id).location != self.player.location):
                raise ValueError("Invalid player handoff participants.")
            index = next(i for i, n in enumerate(self.npcs) if n.id == resident_id)
            if index in reserved or set(self.history[handoff.record].participants) != {"You", actors[resident_id]}:
                raise ValueError("Conflicting player handoff reservation or record.")
            if (handoff.transferred != bool(handoff.outcome_event_id)
                    or handoff.transferred != (handoff.elapsed >= GIFT_GIVE_SECONDS)
                    or handoff.item != "fruit" and (handoff.item not in creation_ids or handoff.quantity != 1)):
                raise ValueError("Invalid player handoff progress or possession.")
            if self.history[handoff.record].transfer_intent != dict(giver_id=handoff.initiator,
                    recipient_id=handoff.recipient, item=handoff.item, quantity=handoff.quantity):
                raise ValueError("A player handoff needs matching accepted authorization.")
            reserved.add(index)
            activity_target(handoff.saved_state, self.npcs[index])
        for interaction in [*self.conversations, *self.gifts]:
            pair = (interaction.initiator, interaction.recipient)
            if pair[0] == pair[1] or any(i >= len(self.npcs) or i in reserved for i in pair) or interaction.record >= len(self.history):
                raise ValueError("Invalid live interaction participants.")
            for index, activity in zip(pair, interaction.saved_states):
                if self.npcs[pair[0]].location != self.npcs[pair[1]].location:
                    raise ValueError("Conversation participants must share a location.")
                activity_target(activity, self.npcs[index])
            record = self.history[interaction.record]
            if set(record.participants) != {actors[self.npcs[i].id] for i in pair}:
                raise ValueError("An interaction must match its record.")
            reserved.update(pair)
            thanks = interaction.handoff_thanks if isinstance(interaction, ConversationState) else interaction.thanks
            rewritten = (interaction.rewritten_handoff_thanks if isinstance(interaction, ConversationState)
                         else interaction.rewritten_thanks)
            if rewritten and (len(rewritten) != len(thanks) or any(not text.strip() for text in rewritten)):
                raise ValueError("Rewritten thanks must match the original parts.")
            if isinstance(interaction, ConversationState):
                if set(interaction.concerns) != {"initiator", "recipient"}:
                    raise ValueError("A meeting must retain both frozen concerns.")
                for role, index in zip(("initiator", "recipient"), pair):
                    validate_concern(interaction.concerns[role], self.npcs[index])
                if interaction.goal_id and (self.npcs[pair[0]].goal is None or interaction.goal_id != self.npcs[pair[0]].goal.id):
                    raise ValueError("A meeting goal must belong to its initiator.")
                if interaction.focus_claim_id and not any(b.claim_id == interaction.focus_claim_id for b in self.npcs[pair[0]].beliefs):
                    raise ValueError("A required meeting account must be known to its initiator.")
                ready = interaction.dialogue_state == "ready"
                if record.id != interaction.id or ready != bool(interaction.turns) or ready != interaction.social_effects_applied:
                    raise ValueError("Invalid conversation commit progress.")
                if interaction.information_context:
                    context = interaction.information_context
                    if context.participants != {"initiator": self.npcs[pair[0]].id, "recipient": self.npcs[pair[1]].id}:
                        raise ValueError("Invalid information speakers.")
                    if context.focus_claim_id != interaction.focus_claim_id:
                        raise ValueError("The required meeting account must match its information snapshot.")
                    if any(o.fact_id not in facts for o in context.options):
                        raise ValueError("Missing snapshot facts.")
                validate_agreements(interaction.accepted_proposals, [p.model_dump() for p in interaction.proposals], interaction.turns, self.npcs[pair[0]].id, self.npcs[pair[1]].id)
                if ready:
                    validate_information(SimpleNamespace(turns=interaction.turns, disclosures=interaction.accepted_disclosures,
                        third_party_opinions=interaction.third_party_opinions, leverage=interaction.leverage_uses),
                        interaction.information_context.model_dump() if interaction.information_context else None)
                if interaction.accepted_gift:
                    from llm import validate_resident_gift
                    validate_resident_gift(interaction.accepted_gift, interaction.turns, {
                        "purpose": interaction.purpose, "arrival_focus": interaction.opening_kind == "orientation",
                        **{
                        role: {"npc_id": self.npcs[index].id, "inventory": self.npcs[index].inventory,
                               "owned_works": [{"id": w.id, "title": w.title} for w in self.creations
                                               if w.owner_id == self.npcs[index].id]}
                        for role, index in zip(("initiator", "recipient"), pair)}})
                    if interaction.accepted_gift.accept_turn is None:
                        raise ValueError("A pending gift handoff needs an accepted offer.")
            elif interaction.transferred != (interaction.elapsed >= GIFT_GIVE_SECONDS) or interaction.transferred != bool(interaction.outcome_event_id):
                raise ValueError("Invalid apple handoff progress.")
            if isinstance(interaction, GiftState):
                if (interaction.item != "fruit" and (interaction.item not in creation_ids or interaction.quantity != 1)
                        or record.transfer_intent != dict(giver_id=self.npcs[pair[0]].id,
                            recipient_id=self.npcs[pair[1]].id, item=interaction.item, quantity=interaction.quantity)):
                    raise ValueError("A handoff needs a real possession and accepted matching authorization.")
        encounter = self.snake_encounter
        if encounter:
            if (encounter.resident_id not in live_ids or encounter.resident_id in indoor_ids
                    or encounter.saved_state.behavior in HOME_BEHAVIORS or encounter.record >= len(self.history)):
                raise ValueError("A snake encounter needs a living resident.")
            resident_index = next(i for i, n in enumerate(self.npcs) if n.id == encounter.resident_id)
            if resident_index in reserved or self.history[encounter.record].id != encounter.id or self.snake.target_id != encounter.resident_id:
                raise ValueError("Conflicting snake encounter reservation.")
            reserved.add(resident_index)
            if encounter.committed != bool(encounter.turns) or encounter.committed != (encounter.dialogue_state == "ready") or encounter.committed != bool(encounter.outcome):
                raise ValueError("Invalid snake encounter commit progress.")
            if encounter.reserved and (encounter.committed or encounter.purpose == "introduction" or self.tree.fruit < 1):
                raise ValueError("Invalid apple reservation.")
            if encounter.stage in ("condemned_approach", "eating") and encounter.outcome != "eat":
                raise ValueError("Eating requires a resolved condemnation.")
            activity_target(encounter.saved_state, self.npcs[resident_index])
        elif self.snake.target_id is not None and not self.player_snake_encounter:
            raise ValueError("A snake target requires an encounter.")
        if (not set(self.snake.standing) <= ids or any(not -100 <= score <= 100 for score in self.snake.standing.values())
                or not set(self.snake.pardons) <= live_ids | ({self.player.id} if self.player.alive else set()) or not set(self.snake.offenses) <= live_ids | ({self.player.id} if self.player.alive else set())
                or any(f not in facts for f in self.snake.offenses.values())):
            raise ValueError("Invalid snake standing, offenses or pardons.")
        for resident_id, advance in self.snake.advances.items():
            record = record_for(advance.conversation_id)
            if (resident_id not in live_ids | ({self.player.id} if self.player.alive else set()) or not participant(resident_id, record)
                    or not participant(self.snake.id, record)
                    or not any(f.kind == "bargain" and f.conversation_id == record.id
                               and f.subject_id == self.snake.id and f.object_id == resident_id for f in self.facts)):
                raise ValueError("An apple advance needs its living recipient and actual handoff record.")
            if advance.offering == "secret" and resident_id != self.player.id:
                resident = next(n for n in self.npcs if n.id == resident_id)
                if advance.account_id not in claims or not any(b.claim_id == advance.account_id for b in resident.beliefs):
                    raise ValueError("A promised private account must belong to the recipient.")
            elif advance.account_id:
                raise ValueError("An endorsement advance cannot select a private account.")
        if self.snake.judged_phase > now or self.snake.retry_request and self.snake.retry_request.resident_id not in live_ids:
            raise ValueError("Invalid snake judgment phase or retry target.")
        for index, npc in enumerate(self.npcs):
            if index not in reserved:
                activity_target(npc, npc)
            if index in reserved and npc.behavior != Behavior.WAIT:
                raise ValueError("Encounter participants must wait.")
        for reference in (self.tree.reserved_by, self.tree.shaker):
            if reference is not None and (reference not in live_ids | ({self.player.id} if self.player.alive and self.mode == "player" else set()) or reference in indoor_ids):
                raise ValueError("A tree reservation requires a living owner.")
        if bool(self.tree.shaker) != (self.tree.shake_elapsed is not None):
            raise ValueError("Invalid tree shake owner.")
        pending_owners = set()
        for action in self.pending_actions:
            if action.target_id == self.player.id and action.action != "GIVE_ITEM":
                raise ValueError("Only a gift can use the player as a queued action recipient.")
            if action.npc_id not in live_ids or action.npc_id in pending_owners or action.target_id is not None and action.target_id not in live_ids | {self.player.id}:
                raise ValueError("Queued actions need living actors.")
            if action.target_id and action.action in ("TALK_TO_NPC", "GIVE_ITEM"):
                positions = {n.id: n.location for n in self.npcs}
                positions[self.player.id] = self.player.location
                if positions[action.npc_id] != positions[action.target_id]:
                    raise ValueError("A queued interaction must share a location.")
            pending_owners.add(action.npc_id)
            if action.record is not None and action.record >= len(self.history):
                raise ValueError("Invalid queued history record.")
            if action.action == "GIVE_ITEM":
                if (action.target_id is None or action.quantity < 1
                        or action.item != "fruit" and (action.item not in creation_ids or action.quantity != 1)
                        or action.record is None or self.history[action.record].transfer_intent != dict(
                            giver_id=action.npc_id, recipient_id=action.target_id, item=action.item, quantity=action.quantity)):
                    raise ValueError("A queued gift needs its real item and accepted matching authorization.")
        pending, processed = self.pending_arrival_days, self.processed_arrival_days
        if pending != sorted(set(pending)) or processed != sorted(set(processed)) or set(pending) & set(processed) or set(pending) | set(processed) != set(range(2, self.day + 1)):
            raise ValueError("Every dawn must have exactly one pending or completed arrival.")
        arrivals = [n.arrival_day for n in residents if n.arrival_day >= 2]
        if sorted(arrivals) != processed:
            raise ValueError("Completed arrivals must match their preserved residents.")
        if not set(self.introductions) <= live_ids:
            raise ValueError("Invalid introduction identity.")
        reserved_ids = {self.npcs[index].id for index in reserved}
        for npc in residents:
            opinion = npc.player_opinion
            if (set(opinion) != {"opinion", "reason"} or type(opinion["opinion"]) is not int
                    or not -100 <= opinion["opinion"] <= 100 or not isinstance(opinion["reason"], str)):
                raise ValueError("Invalid player opinion.")
            call = npc.player_call
            if call and call["kind"] == "reply":
                record = record_for(call["reference"])
                actions = [a for a in self.pending_actions if a.npc_id == npc.id]
                if (npc.id not in live_ids or npc.id in reserved_ids
                        or set(record.participant_ids) != {npc.id, self.player.id} or record.error
                        or not record.turns or record.turns[-1] != (actors[npc.id], npc.reply)
                        or (actions and (not call["closing"] or any(a.record is None or self.history[a.record].id != record.id for a in actions)))):
                    raise ValueError("A waiting reply needs its resident's accepted player conversation and matching queued task.")
                offer = call["gift_offer"]
                if offer and (offer["recipient_id"] != self.player.id
                              or offer["item"] != "fruit" and offer["item"] not in creation_ids):
                    raise ValueError("A waiting reply's gift offer needs the player and a real item.")
                continue
            if call and (npc.id not in live_ids
                    or npc.behavior not in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY)
                    or npc.owner != "autonomous"
                    or npc.id in reserved_ids
                    or any(npc.id in (a.npc_id, a.target_id) for a in self.pending_actions)
                    or set(call) != {"key", "kind", "reference", "reason", "event_id"}
                    or call["kind"] not in ("gift", "news", "followup")
                    or call["key"] in npc.resolved_player_calls):
                raise ValueError("Invalid waiting player call.")
            if call:
                ref = call["reference"]
                valid = (npc.inventory["fruit"] > 0 and self.player.inventory_capacity > self.player.inventory["fruit"]
                         if call["kind"] == "gift" and ref == "fruit" else
                         any(w.id == ref and w.owner_id == npc.id for w in self.creations) if call["kind"] == "gift" else
                         any(b.claim_id == ref for b in npc.beliefs) if call["kind"] == "news" else
                         any(m.id == ref and m.partner_id == self.player.id for m in npc.memories)
                         or any(p.id == ref and p.status == "pending" and {p.promisor_id, p.recipient_id} == {npc.id, self.player.id} for p in self.promises))
                if not valid or not call["reason"].strip():
                    raise ValueError("A player call needs its resident's actual unresolved source.")
        if sum(n.player_call is not None and n.player_call["kind"] != "reply" for n in self.npcs) > PLAYER_CALL_LIMIT:
            raise ValueError("Too many simultaneous player calls.")
        return self


class SimulationSave(SaveModel):
    name: str = ""
    created_at: datetime
    updated_at: datetime
    state: SimulationState


@dataclass
class SaveEntry:
    id: str
    name: str
    updated_at: datetime | None = None
    npc_count: int = 0
    error: str = ""
    mode: str = ""


class InvalidSaveName(ValueError):
    pass


class SaveStore:
    def __init__(self, directory=SAVE_DIRECTORY):
        self.directory = Path(directory)

    def _path(self, save_id):
        if (not isinstance(save_id, str) or not save_id.strip() or save_id in (".", "..")
                or save_id != save_id.strip() or save_id.endswith(".")
                or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in save_id)):
            raise InvalidSaveName('Use a filename without <>:"/\\|?* or trailing spaces/dots.')
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", save_id.split(".")[0], re.IGNORECASE):
            raise InvalidSaveName("That filename is reserved. Choose another name.")
        path = self.directory / f"{save_id}.json"
        if path.resolve().parent != self.directory.resolve():
            raise ValueError("Invalid save location.")
        return path

    def load(self, save_id):
        save = SimulationSave.model_validate_json(self._path(save_id).read_text(encoding="utf-8"))
        save.name = save_id
        return save

    def list_saves(self):
        entries = []
        for path in self.directory.glob("*.json"):
            try:
                save = self.load(path.stem)
                entries.append(SaveEntry(path.stem, save.name, save.updated_at, len(save.state.npcs), mode=save.state.mode))
            except (OSError, ValueError):
                entries.append(SaveEntry(path.stem, path.stem, error="This save cannot be loaded. You can delete it."))
        return sorted(entries, key=lambda entry: entry.updated_at.timestamp() if entry.updated_at else 0, reverse=True)

    def next_name(self):
        used = {path.stem.casefold() for path in self.directory.glob("*.json")}
        number = 1
        while f"simulation {number}" in used:
            number += 1
        return f"Simulation {number}"

    def _require_available(self, name, *, exclude=None):
        target = self._path(name)
        if self.directory.exists():
            for path in self.directory.iterdir():
                if path.name.casefold() == target.name.casefold() and path != exclude:
                    raise FileExistsError("A save with that name already exists. Choose another name.")

    def file_identity(self, name):
        stat = self._path(name).stat()
        return stat.st_dev, stat.st_ino

    def current_name(self, name, identity):
        """Follow a file renamed outside the app without recreating its old filename."""
        try:
            if self.file_identity(name) == identity:
                return name
        except FileNotFoundError:
            pass
        for path in self.directory.glob("*.json"):
            try:
                if self.file_identity(path.stem) == identity:
                    return path.stem
            except FileNotFoundError:
                continue
        return name

    def _write(self, save_id, save, *, overwrite=True):
        path = self._path(save_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(save.model_dump_json(indent=2) + "\n", encoding="utf-8")
            if not overwrite:
                self._require_available(save_id)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, state, *, save_id=None, name=None):
        now = datetime.now(timezone.utc)
        try:
            created_at = self.load(save_id).created_at if save_id else now
        except FileNotFoundError:
            # A save removed outside the app can be recreated from the live world.
            created_at = now
        overwrite = save_id is not None
        save_id = save_id if overwrite else (name if name is not None else self.next_name())
        if not overwrite:
            self._require_available(save_id)
        save = SimulationSave(name=save_id, created_at=created_at, updated_at=now, state=state)
        self._write(save_id, save, overwrite=overwrite)
        return save_id

    def rename(self, save_id, name):
        old_path = self._path(save_id)
        new_path = self._path(name)
        self._require_available(name, exclude=old_path)
        old_path.rename(new_path)
        return name

    def delete(self, save_id):
        self._path(save_id).unlink()


PLAYER_SNAKE_FIELDS = ("id", "purpose", "origin", "topic", "accusation", "reserved", "arrived",
    "dialogue_state", "stage", "wait_elapsed", "elapsed", "talking_remaining", "committed", "outcome", "outcome_event_id", "offer")


def capture_player_snake(encounter, records):
    from snake import PlayerSnakeEncounter
    if not isinstance(encounter, PlayerSnakeEncounter):
        return None
    return {**{name: deepcopy(getattr(encounter, name)) for name in PLAYER_SNAKE_FIELDS},
            "record": records[id(encounter.record)]}


def restore_player_snake(world, saved, history):
    if saved is None:
        return
    from snake import PlayerSnakeEncounter
    encounter = PlayerSnakeEncounter.__new__(PlayerSnakeEncounter)
    for name in PLAYER_SNAKE_FIELDS:
        setattr(encounter, name, deepcopy(saved[name]))
    encounter.record = history[saved["record"]]
    encounter.world, encounter.initiator, encounter.recipient = world, world.snake, world.player
    encounter.active = True
    encounter.turns, encounter.pending_opinions = [], []
    encounter.request_context = encounter.saved_state = None
    world.snake.encounter = encounter


def validate_physical_state(state):
    """Canonical map/ownership checks supplement the existing effect provenance checks."""
    import math
    import json
    from locations import valid_location, feet_bounds, route, floor_for_room, targets
    from constants import PLAYER_APPROACH_TIMEOUT, TRAVEL_STEP_TIMEOUT, ROOM_ARRIVAL, STARTING_PLAYER_POSITION
    residents = [*state.npcs, *state.archived]
    layout = SimpleNamespace(residents=residents)
    by_id = {n.id: n for n in state.npcs}
    by_id[state.player.id] = state.player
    records = {r.id: r for r in state.history}
    live_player = state.mode == "player" and state.player.alive
    if not state.player.alive and state.mode != "spectator":
        raise ValueError("A dead player must remain in spectator mode.")
    if not live_player and (state.player_interaction or state.player.journey or state.visit or state.player_snake_encounter
            or any(n.player_call for n in state.npcs) or any(e["actor_id"] == state.player.id for e in state.tree_queue)
            or state.tree_current and state.tree_current["actor_id"] == state.player.id):
        raise ValueError("Spectator mode cannot own physical player interactions.")
    def point(location, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
            raise ValueError("Invalid feet position.")
        left, right, top, bottom = feet_bounds(location)
        if not left <= value[0] <= right or not top <= value[1] <= bottom:
            raise ValueError("Feet position is outside its map.")
    for actor in [state.player, *residents]:
        if not valid_location(layout, actor.location):
            raise ValueError("Unknown actor location.")
        point(actor.location, (actor.x, actor.y))
        journey = actor.journey
        if journey:
            if set(journey) != {"destination", "point", "purpose", "fast", "elapsed", "steps"} or not valid_location(layout, journey["destination"]):
                raise ValueError("Invalid journey identity.")
            point(journey["destination"], journey["point"])
            if (actor is state.player and journey["destination"].startswith("room:")
                    and journey["destination"] not in ("room:1", actor.location)
                    and (not state.visit or state.visit["location"] != journey["destination"])):
                raise ValueError("Travel into a resident room requires its accepted hosted visit.")
            if not 0 <= journey["elapsed"] <= TRAVEL_STEP_TIMEOUT or not journey["steps"]:
                raise ValueError("Invalid journey progress.")
            expected = route(actor.location, journey["destination"], journey["point"])
            if journey["steps"] != [{**e, "point": list(e["point"]), "arrival": list(e["arrival"]) if e["arrival"] is not None else None} for e in expected] and journey["steps"] != expected:
                raise ValueError("A journey must follow the actual doors and elevator connections.")
    interaction = state.player_interaction
    exclusive = set()
    if interaction:
        if interaction.get("kind") not in ("resident", "door", "snake") or not interaction.get("id"):
            raise ValueError("Invalid player interaction owner.")
        if interaction["location"] != state.player.location:
            raise ValueError("Player ownership must follow its current location.")
        if interaction["kind"] == "resident":
            if state.player.journey:
                raise ValueError("A player cannot travel during a resident interaction.")
            npc = by_id.get(interaction.get("npc_id"))
            if not npc or npc is state.player or npc.journey:
                raise ValueError("The player needs one present resident.")
            room = interaction.get("knock_room")
            if room:
                if npc.room_number != room or npc.location != f"room:{room}" or state.player.location != f"hallway:{floor_for_room(room)}":
                    raise ValueError("A knock must address the present owner of that door.")
            elif npc.location != state.player.location:
                raise ValueError("Player conversation participants must be together.")
            if interaction["stage"] not in ("approach", "prompt", "conversation", "handoff") or not 0 <= interaction["elapsed"] <= PLAYER_APPROACH_TIMEOUT:
                raise ValueError("Invalid player approach progress.")
            ActivityState.model_validate(interaction["saved_state"])
            point(npc.location if interaction["incoming"] else state.player.location, interaction["point"])
            if interaction["stage"] == "prompt" and (not interaction["incoming"] or not npc.player_call):
                raise ValueError("An incoming prompt needs its physical call.")
            if interaction.get("record_id") and (interaction["record_id"] not in records
                    or set(records[interaction["record_id"]].participant_ids) != {state.player.id, npc.id}):
                raise ValueError("Player conversation ownership needs its actual source record.")
            exclusive.add(npc.id)
        elif interaction["kind"] == "snake" and not state.player_snake_encounter:
            raise ValueError("Player snake ownership needs a service encounter.")
        elif interaction["kind"] == "door":
            normalize = lambda value: json.loads(json.dumps(value))
            if normalize(interaction["target"]) not in [normalize(t) for t in targets(layout, state.player.location)]:
                raise ValueError("Door travel must retain an actual target in its map.")
    entries = [*state.tree_queue, *([state.tree_current] if state.tree_current else [])]
    if len({e["actor_id"] for e in entries}) != len(entries):
        raise ValueError("Tree customers must be unique and ordered.")
    for entry in entries:
        if set(entry) != {"actor_id", "purpose", "reason", "saved_state", "elapsed"}:
            raise ValueError("Invalid tree queue entry.")
        actor = by_id.get(entry["actor_id"])
        if not actor or actor.location != "garden" or actor.journey or actor.id in exclusive or entry["purpose"] not in ("bargain", "judgment", "introduction"):
            raise ValueError("A tree customer must be available in the garden.")
        if not isinstance(entry["elapsed"], (int, float)) or not math.isfinite(entry["elapsed"]) or not 0 <= entry["elapsed"] <= PLAYER_APPROACH_TIMEOUT:
            raise ValueError("Invalid front-customer approach progress.")
        if actor is state.player and interaction and interaction["kind"] != "snake":
            raise ValueError("A queued player cannot own another interaction.")
        if any(a.npc_id == actor.id for a in state.pending_actions):
            raise ValueError("A queued customer cannot have another pending action.")
        if actor is not state.player:
            ActivityState.model_validate(entry["saved_state"])
        exclusive.add(actor.id)
    reserved_ids = {state.npcs[i].id for c in [*state.conversations, *state.gifts] for i in (c.initiator, c.recipient) if i < len(state.npcs)}
    if exclusive & reserved_ids:
        raise ValueError("An actor cannot belong to two exclusive interactions.")
    if state.visit:
        visit = state.visit
        host = by_id.get(visit.get("host_id"))
        if host is None or host is state.player or not valid_location(layout, visit["location"]):
            raise ValueError("A visit needs its actual living host and destination.")
        ActivityState.model_validate(visit["saved_state"])
        offer = next((o for o in state.player_invitations if o["id"] == visit["offer_id"]), None)
        if (not offer or offer["status"] != "accepted" or offer["speaker_id"] != host.id
                or offer["location"] != visit["location"] or offer["destination"] != visit["destination"]
                or visit["stage"] not in ("travel", "hosting")):
            raise ValueError("A hosted visit requires its actual accepted invitation.")
        if visit["stage"] == "travel" and (interaction or state.player.id in {e["actor_id"] for e in entries}):
            raise ValueError("An accepted visit's travel cannot own another player interaction.")
        if visit["stage"] == "travel":
            for actor, purpose in ((host, "host_visit"), (state.player, "invitation")):
                if actor.journey:
                    if actor.journey["destination"] != visit["location"] or actor.journey["purpose"] != purpose:
                        raise ValueError("Host and visitor must travel to their accepted destination.")
                elif actor.location != visit["location"]:
                    raise ValueError("An unfinished visit needs its actual remaining journey.")
        if visit["stage"] == "hosting" and (host.location != visit["location"] or state.player.location != visit["location"]):
            raise ValueError("A host and visitor must be present together.")
        if not valid_location(layout, visit["return_location"]):
            raise ValueError("A host's return destination must remain valid.")
        point(visit["return_location"], visit["return_point"])
        if host.id in {e["actor_id"] for e in entries} or host.id in reserved_ids:
            raise ValueError("A host must stay available for the player.")
    claim_ids = set()
    for claim in state.player_claims:
        record = records.get(claim.get("conversation_id"))
        turn = claim.get("turn")
        if (set(claim) != {"id", "speaker_id", "conversation_id", "turn", "day", "phase", "text", "kind", "listener_ids"}
                or claim["id"] in claim_ids or claim["speaker_id"] != state.player.id or claim["kind"] != "report"
                or not record or type(turn) is not int or turn not in record.accepted_player_turns
                or record.turns[turn] != ("You", claim["text"])
                or set(claim["listener_ids"]) != set(record.participant_ids) - {state.player.id}
                or phase_index(claim["day"], claim["phase"]) > phase_index(state.day, state.phase)):
            raise ValueError("Player claims require their actual accepted original speech and listener.")
        claim_ids.add(claim["id"])
    invitation_ids = set()
    for offer in state.player_invitations:
        record = records.get(offer.get("conversation_id"))
        speaker = next((n for n in residents if n.id == offer.get("speaker_id")), None)
        if (offer["id"] in invitation_ids or not record or not speaker or offer["status"] not in ("offered", "accepted", "withdrawn", "declined")
                or offer["destination"] not in ("garden", "tree", "home")
                or not 0 <= offer["turn"] < len(record.turns)
                or record.turns[offer["turn"]] != (speaker.display_name or speaker.character["Name"], offer["speech"])
                or offer["location"] != (f"room:{speaker.room_number}" if offer["destination"] == "home" else "garden")):
            raise ValueError("Invitations must retain their real speaker, speech and destination.")
        expected_point = {"garden": STARTING_PLAYER_POSITION, "home": ROOM_ARRIVAL,
                          "tree": Tree(x=state.tree.x, y=state.tree.y).interaction_point}[offer["destination"]]
        if (tuple(offer["point"]) != tuple(expected_point)
                or set(record.participant_ids) != {speaker.id, state.player.id}):
            raise ValueError("An invitation needs its actual meeting point and conversation participants.")
        invitation_ids.add(offer["id"])
        if offer["status"] == "offered" and (not interaction or interaction["kind"] != "resident"
                or offer["interaction_id"] != interaction["id"] or offer["speaker_id"] != interaction["npc_id"]):
            raise ValueError("A live invitation must belong to the current physical conversation.")
    if sum(o["status"] == "offered" for o in state.player_invitations) > 1:
        raise ValueError("Only the current speaker can keep a live invitation.")
    if state.snake_encounter and (not state.tree_current or state.tree_current["actor_id"] != state.snake_encounter.resident_id):
        raise ValueError("A resident tree encounter must own the current FIFO turn.")
    if state.tree.reserved_by is not None and (not state.tree_current or state.tree.reserved_by != state.tree_current["actor_id"]):
        raise ValueError("Physical tree ownership must match the current customer.")
    for actor_id, learned in state.snake.known_names.items():
        record = records.get(learned.conversation_id)
        if actor_id != state.player.id or not record or learned.turn not in record.accepted_player_turns or state.snake.id not in record.participant_ids:
            raise ValueError("The snake's learned player name requires accepted original speech.")
    saved = state.player_snake_encounter
    if saved:
        if set(saved) != {*PLAYER_SNAKE_FIELDS, "record"} or not state.tree_current or state.tree_current["actor_id"] != state.player.id:
            raise ValueError("Player tree service needs its current FIFO association.")
        if not 0 <= saved["record"] < len(state.history) or state.history[saved["record"]].id != saved["id"] or state.snake.target_id != state.player.id:
            raise ValueError("Player tree service must match its original encounter.")
        if saved["stage"] not in ("player", "dialogue", "condemned_approach", "eating") or saved["stage"] != "player" and (not saved["committed"] or saved["outcome"] != "eat"):
            raise ValueError("Invalid player eating progress.")
        offer = saved["offer"]
        if offer:
            record = state.history[saved["record"]]
            if (set(offer) != {"offering", "turn", "conversation_id", "speech"}
                    or offer["offering"] not in ("secret", "endorsement") or offer["conversation_id"] != record.id
                    or type(offer["turn"]) is not int or not 0 <= offer["turn"] < len(record.turns)
                    or record.turns[offer["turn"]] != (SNAKE_NAME, offer["speech"])):
                raise ValueError("A player apple offer must retain its actual original speaker and turn.")


def _capture_fields(model, source, **overrides):
    """Use the save model's fields; spell out only reference/type conversions."""
    return model(**{name: overrides[name] if name in overrides else getattr(source, name)
                    for name in model.model_fields})


def _restore_fields(target, saved, *, exclude=(), **overrides):
    for name in type(saved).model_fields:
        if name not in exclude:
            setattr(target, name, overrides[name] if name in overrides else getattr(saved, name))


def capture_resident(npc):
    from player import call_valid
    stale_call = npc.player_call and not call_valid(npc.world, npc, npc.player_call)
    return _capture_fields(
        NPCState, npc, emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
        memories=[asdict(memory) for memory in npc.memories],
        goal=asdict(npc.goal) if npc.goal else None,
        current_concern=asdict(npc.current_concern) if npc.current_concern else None,
        pending_social_opinions=[asdict(effect) for effect in npc.pending_social_opinions],
        beliefs=[asdict(belief) for belief in npc.beliefs],
        player_call=None if stale_call else deepcopy(npc.player_call),
        player_call_cooldown=PLAYER_CALL_COOLDOWN if stale_call else npc.player_call_cooldown,
        resolved_player_calls=([*npc.resolved_player_calls, npc.player_call["key"]]
            if stale_call and not npc.unread_player_reply else npc.resolved_player_calls.copy()),
    )


def capture_simulation(app):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    npc_indices = {id(npc): index for index, npc in enumerate(app.npcs)}
    records = {id(record): index for index, record in enumerate(app.conversation_history)}
    history = [asdict(record) for record in app.conversation_history]
    unread_records = {npc.unread_player_reply["reference"] for npc in app.npcs if npc.unread_player_reply}
    for record in history:
        for offer in record["player_promise_offers"]:
            if offer["status"] == "offered" and record["id"] not in unread_records:
                offer["status"] = "closed"
    for request in getattr(app, "player_requests", {}).values():
        if request["pending"] and id(request["record"]) in records:
            record = history[records[id(request["record"])]]
            record["status"] = "Reply pending when saved"
            record["outcomes"].insert(0, "The reply was not saved. Send the request again after loading.")
    npcs, conversations, gifts = [], [], []
    for npc in app.npcs:
        npcs.append(capture_resident(npc))
        gift = npc.gift
        if gift is not None and not isinstance(gift, PlayerHandoff) and gift.initiator is npc:
            gifts.append(_capture_fields(GiftState, gift,
                initiator=npc_indices[id(npc)], recipient=npc_indices[id(gift.recipient)],
                record=records[id(gift.record)],
            ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(_capture_fields(ConversationState, conversation,
            initiator=npc_indices[id(npc)], recipient=npc_indices[id(conversation.recipient)],
            record=records[id(conversation.record)],
            pending_opinions=[(npc_indices[id(person)], opinions) for person, opinions in conversation.pending_opinions],
            proposals=[asdict(p) for p in conversation.proposals],
            information_context=asdict(conversation.information_context) if conversation.information_context else None,
            concerns={role: asdict(c) if c else None for role, c in conversation.concerns.items()},
        ))
    world = app.simulation
    encounter = world.snake.encounter
    return SimulationState(
        day=world.day, npc_trait_mode=world.npc_trait_mode, phase=world.phase,
        mode=world.mode, player_interaction=deepcopy(world.player_interaction),
        player_invitations=deepcopy(world.player_invitations), player_claims=deepcopy(world.player_claims),
        visit=deepcopy(world.visit), tree_queue=deepcopy(world.tree_queue), tree_current=deepcopy(world.tree_current),
        player_snake_encounter=capture_player_snake(encounter, records),
        player=PlayerState(**asdict(world.player)), player_call_check=max(0.0, world.player_call_check),
        player_acknowledgments=deepcopy(world.player_acknowledgments),
        player_handoffs=[_capture_fields(PlayerHandoffState, npc.gift,
            initiator=npc.gift.initiator.id, recipient=npc.gift.recipient.id,
            record=records[id(npc.gift.record)]) for npc in world.npcs if isinstance(npc.gift, PlayerHandoff)],
        creations=world.creations,
        events=[asdict(event) for event in world.events],
        promises=[asdict(promise) for promise in world.promises],
        facts=[asdict(fact) for fact in world.facts], claims=[asdict(claim) for claim in world.claims],
        disclosures=[asdict(disclosure) for disclosure in world.disclosures],
        social_rotation=world.social_rotation,
        snake=_capture_fields(SnakeState, world.snake, beliefs=[asdict(b) for b in world.snake.beliefs]),
        archived=[capture_resident(npc) for npc in world.archived],
        arrival_error=world.arrival_error,
        pending_arrival_days=world.pending_arrival_days.copy(), processed_arrival_days=world.processed_arrival_days.copy(),
        introductions=world.introductions.copy(), roster_revision=world.roster_revision,
        snake_encounter=(_capture_fields(SnakeEncounterState, encounter,
                         resident_id=encounter.recipient.id, record=records[id(encounter.record)])
                         if encounter is not None and encounter.recipient is not world.player else None),
        npcs=npcs, history=history, conversations=conversations, gifts=gifts,
        tree=_capture_fields(TreeState, app.tree,
                       shaker=app.tree.shaker.id if app.tree.shaker is not None else None,
                       reserved_by=app.tree.reserved_by.id if app.tree.reserved_by is not None else None),
        pending_actions=[asdict(action) for action in world.pending_actions],
        camera_offset=tuple(app.game.camera_offset),
        camera_zoom=app.game.camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    state = SimulationState.model_validate(state.model_dump())
    all_residents = []
    for saved in [*state.npcs, *state.archived]:
        npc = NPC(
            id=saved.id, character=saved.character, display_name=saved.display_name, x=saved.x, y=saved.y,
            reply=saved.reply, emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            social=saved.social,
            sprite_size=saved.sprite_size, room_number=saved.room_number, location=saved.location, journey=deepcopy(saved.journey), social_fatigue=saved.social_fatigue,
            home_remaining=saved.home_remaining, outdoor_remaining=saved.outdoor_remaining,
            bedtime_remaining=saved.bedtime_remaining, wake_remaining=saved.wake_remaining,
            home_reason=saved.home_reason, last_creative_day=saved.last_creative_day,
            creative_job=saved.creative_job.model_copy(deep=True) if saved.creative_job else None,
            inventory=saved.inventory.copy(), inventory_capacity=saved.inventory_capacity,
            rewritten_reply=saved.rewritten_reply,
            preferences=saved.preferences.model_dump(), command_revision=saved.command_revision,
            memories=[Memory(**memory.model_dump()) for memory in saved.memories],
            goal=Goal(**saved.goal.model_dump()) if saved.goal else None,
            pending_social_opinions=[OpinionEffect(**effect.model_dump()) for effect in saved.pending_social_opinions],
            beliefs=[Belief(**belief.model_dump()) for belief in saved.beliefs],
            current_concern=ConversationConcern(**saved.current_concern.model_dump()) if saved.current_concern else None,
            conversation_pause_remaining=saved.conversation_pause_remaining,
            conversation_return_remaining=saved.conversation_return_remaining,
            known_names=deepcopy(saved.model_dump()["known_names"]),
            attitudes=saved.attitudes.model_dump(),
            arrival_day=saved.arrival_day, departure=saved.departure.model_dump() if saved.departure else None,
        )
        npc.restore_activity({name: getattr(saved, name) for name in ACTIVITY_FIELDS})
        all_residents.append(npc)
    npcs = all_residents[:len(state.npcs)]
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit,
                shake_elapsed=state.tree.shake_elapsed, shaker=by_id.get(state.tree.shaker),
                reserved_by=by_id.get(state.tree.reserved_by))
    world = Simulation(npcs, tree, assign_goals=False, npc_trait_mode=state.npc_trait_mode, mode=state.mode)
    world.archived = all_residents[len(state.npcs):]
    world.player = Player(**state.player.model_dump())
    world.player.world = world
    world.player_interaction = deepcopy(state.player_interaction)
    world.player_invitations = deepcopy(state.player_invitations)
    world.player_claims = deepcopy(state.player_claims)
    world.visit = deepcopy(state.visit)
    world.tree_queue, world.tree_current = deepcopy(state.tree_queue), deepcopy(state.tree_current)
    world.player_call_check = state.player_call_check
    world.player_acknowledgments = deepcopy(state.player_acknowledgments)
    for npc, saved in zip(all_residents, [*state.npcs, *state.archived]):
        npc.player_opinion = deepcopy(saved.player_opinion)
        npc.player_call = deepcopy(saved.player_call)
        npc.player_call_cooldown = saved.player_call_cooldown
        npc.resolved_player_calls = saved.resolved_player_calls.copy()
    world.creations = [work.model_copy(deep=True) for work in state.creations]
    for npc in world.archived:
        npc.world = world
    snake_values = state.snake.model_dump()
    snake_values["beliefs"] = [Belief(**b) for b in snake_values["beliefs"]]
    world.snake = Snake(**snake_values)
    world.snake.world = world
    world.arrival_error = state.arrival_error
    world.pending_arrival_days = state.pending_arrival_days.copy()
    world.processed_arrival_days = state.processed_arrival_days.copy()
    world.introductions = state.introductions.copy()
    world.roster_revision = state.roster_revision
    world.day, world.phase = state.day, state.phase
    world.pending_actions = [PendingAction(**action.model_dump()) for action in state.pending_actions]
    for action in world.pending_actions:
        by_id[action.npc_id].request_pending = True
    history = [ConversationRecord(**saved.model_dump()) for saved in state.history]
    world.history = history
    world.events = [WorldEvent(**event.model_dump()) for event in state.events]
    world.promises = [Promise(**promise.model_dump()) for promise in state.promises]
    world.facts = [Fact(**fact.model_dump()) for fact in state.facts]
    world.claims = [Claim(**claim.model_dump()) for claim in state.claims]
    world.disclosures = [Disclosure(**disclosure.model_dump()) for disclosure in state.disclosures]
    world.social_rotation = state.social_rotation
    for saved in state.conversations:
        # Loading bypasses __init__: it would interrupt the saved activities again.
        conversation = Conversation.__new__(Conversation)
        _restore_fields(conversation, saved,
            initiator=npcs[saved.initiator], recipient=npcs[saved.recipient], record=history[saved.record],
            proposals=[Proposal(**proposal.model_dump()) for proposal in saved.proposals],
            information_context=information_from_dict(saved.information_context.model_dump()) if saved.information_context else None,
            concerns={role: ConversationConcern(**c.model_dump()) if c else None for role, c in saved.concerns.items()},
            pending_opinions=[(npcs[index], opinions) for index, opinions in saved.pending_opinions],
            saved_states=[activity.model_dump() for activity in saved.saved_states])
        conversation.active = True
        conversation.name_actors = {}  # Pending generation takes a fresh reference snapshot after loading.
        for npc in (conversation.initiator, conversation.recipient):
            npc.conversation = conversation
        if conversation.dialogue_state == "pending":
            conversation.dialogue_state = "waiting"
            conversation.wait_elapsed = 0.0
    for saved in state.gifts:
        gift = Gift.__new__(Gift)
        _restore_fields(gift, saved,
            initiator=npcs[saved.initiator], recipient=npcs[saved.recipient], record=history[saved.record],
            saved_states=[activity.model_dump() for activity in saved.saved_states])
        gift.active = True
        for npc in (gift.initiator, gift.recipient):
            npc.gift = gift
    for saved in state.player_handoffs:
        gift = PlayerHandoff.__new__(PlayerHandoff)
        _restore_fields(gift, saved, initiator=world.participant_by_id(saved.initiator),
            recipient=world.participant_by_id(saved.recipient), record=history[saved.record],
            saved_state=saved.saved_state.model_dump())
        gift.world = world
        gift.resident = gift.recipient if gift.initiator is world.player else gift.initiator
        gift.active = gift.arrived = True
        gift.resident.gift = gift
    saved = state.snake_encounter
    if saved:
        encounter = SnakeEncounter.__new__(SnakeEncounter)
        _restore_fields(encounter, saved, exclude=("resident_id",),
                        record=history[saved.record], saved_state=saved.saved_state.model_dump())
        encounter.world = world
        encounter.initiator, encounter.recipient = world.snake, by_id[saved.resident_id]
        encounter.active = True
        encounter.pending_opinions = []
        world.snake.encounter = encounter
        encounter.recipient.snake_encounter = encounter
        if encounter.dialogue_state == "pending":
            encounter.dialogue_state = "waiting"
            encounter.wait_elapsed = 0.0
            encounter.request_context = None
    restore_player_snake(world, state.player_snake_encounter, history)
    # Claims are reacquired through the same safe path as interrupted activities.
    tree.reserved_by = None
    tree.cancel_shake()
    for npc in npcs:
        if npc.conversation is None and npc.gift is None and npc.target_id == tree.id and not world.tree_has_actor(npc.id):
            npc.resume_pending = True
    from player import call_valid, clear_call
    for npc in npcs:
        if npc.player_call and not call_valid(world, npc, npc.player_call):
            clear_call(npc)
    return world
