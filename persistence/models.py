from simulation.conversation import ResidentProgress
from simulation.npc import ResidentValues
from simulation.transfers import HandoffProgress
from simulation.transfer_records import TransferIntent, TransferReceipt
from simulation.snake import SnakeValues
from datetime import datetime
from typing import Annotated, Literal, ClassVar
from pydantic import Field, field_validator, model_validator
from persistence.records import SaveModel, saved_record, restore_record
from simulation.interactions import DialogueProgress
from persistence.characters import normalize_character, normalize_opinions
from constants import (
    CONCERN_LIFETIME_RANGE, WORLD_HEIGHT, WORLD_WIDTH, GARDEN_LOCATION, PLAYER_CALL_CHECK_SECONDS,
    TRAVEL_STEP_TIMEOUT, PLAYER_APPROACH_TIMEOUT, MAX_RESIDENTS,
)
from simulation.creations import Creation
from simulation.apartments import Invitation, HostedVisit
from simulation.locations import Journey, RouteStep
from simulation.maps import validate_destination, validate_feet
from simulation.player_records import (
    PlayerOpinion, PlayerCall, PlayerReplyCall, PlayerReport, PlayerPromiseOffer, PlayerAcknowledgment,
)
from dialogue.history import RecordedTurn, ConversationRecord, Memory, WorldEvent, validate_history_turns
from dialogue.models import ConversationTurn, EmotionReaction, EscalationChoice
from simulation.activity import ActivitySnapshot
from simulation.actions import PendingAction
from simulation.errands import Errand
from simulation.player import Player, PlayerInteraction
from simulation.social.records import (
    Belief, Claim, ConversationConcern, Disclosure, Fact, Goal, OpinionEffect, Promise, Proposal,
    ClaimOption, InformationContext,
)


class ActivityState(saved_record(ActivitySnapshot)):
    @model_validator(mode="after")
    def validate_activity(self):
        ActivitySnapshot.validate(self)
        return self


class PreferencesState(SaveModel):
    sociability: float = Field(ge=0, le=1)
    generosity: float = Field(ge=0, le=1)


MemoryState = saved_record(Memory)


EventState = saved_record(WorldEvent)


GoalState = saved_record(Goal)


ProposalState = saved_record(Proposal)


PromiseState = saved_record(Promise)


OpinionEffectState = saved_record(OpinionEffect)


FactState = saved_record(Fact)


ClaimState = saved_record(Claim)


BeliefState = saved_record(Belief)


DisclosureState = saved_record(Disclosure)


ClaimOptionState = saved_record(ClaimOption)


class InformationState(saved_record(InformationContext)):
    beliefs: dict[str, list[BeliefState]]
    options: list[ClaimOptionState]

    def restore(self):
        return restore_record(InformationContext, self,
            beliefs={role: [restore_record(Belief, belief) for belief in entries]
                     for role, entries in self.beliefs.items()},
            options=[restore_record(ClaimOption, option) for option in self.options])


class DepartureState(SaveModel):
    day: int = Field(ge=1)
    cause: Literal["killed by resident", "killed by player"]
    attacker_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)


PlayerOpinionState = saved_record(PlayerOpinion)


PlayerCallState = saved_record(PlayerCall)


PlayerReplyCallState = saved_record(PlayerReplyCall)


TransferIntentState = saved_record(TransferIntent)


TransferState = saved_record(TransferReceipt)


PlayerAppleOfferState = saved_record(PlayerPromiseOffer)


PlayerAcknowledgmentState = saved_record(PlayerAcknowledgment)


# Previously retained dictionary numbers kept their integer/float representation.
PhysicalNumber = int | float | bool


class PhysicalActivityState(ActivityState):
    destination: tuple[PhysicalNumber, PhysicalNumber] | None
    pause_remaining: PhysicalNumber = Field(ge=0)
    activity_elapsed: PhysicalNumber = Field(ge=0)
    decision_remaining: PhysicalNumber = Field(ge=0)
    cooldowns: dict[str, PhysicalNumber]


def _physical_number(value):
    # These values were checked on raw dictionaries after structural parsing.
    # Retaining the parsed record must not start accepting numeric strings.
    if not isinstance(value, (int, float)):
        raise ValueError("Invalid physical state: required ownership, location or source data is missing.")
    return value


def _physical_point(value):
    if value is not None and isinstance(value, (list, tuple)):
        for coordinate in value:
            _physical_number(coordinate)
    return value


def _physical_index(value):
    if not isinstance(value, int):
        raise ValueError("Invalid physical state: required ownership, location or source data is missing.")
    return value


class RouteStepState(saved_record(RouteStep)):
    point: tuple[PhysicalNumber, PhysicalNumber]
    arrival: tuple[PhysicalNumber, PhysicalNumber] | None
    numeric_points = field_validator("point", "arrival", mode="before")(_physical_point)


class InvitationState(saved_record(Invitation)):
    point: list[PhysicalNumber]
    numeric_point = field_validator("point", mode="before")(_physical_point)
    integer_turn = field_validator("turn", mode="before")(_physical_index)


class HostedVisitState(saved_record(HostedVisit)):
    saved_state: PhysicalActivityState
    return_point: list[PhysicalNumber]
    numeric_point = field_validator("return_point", mode="before")(_physical_point)


class PlayerInteractionState(saved_record(PlayerInteraction)):
    saved_state: PhysicalActivityState | None
    point: tuple[PhysicalNumber, PhysicalNumber] | None
    elapsed: PhysicalNumber = Field(ge=0, le=PLAYER_APPROACH_TIMEOUT)
    numeric_elapsed = field_validator("elapsed", mode="before")(_physical_number)

    @model_validator(mode="after")
    def validate_pending_command(self):
        pending = self.pending_request
        if pending and pending.get("kind") == "player_command":
            from dialogue.models import ActionIntent
            if (set(pending) != {"message", "kind", "context"} or pending["message"] != ""
                    or not isinstance(pending["context"], dict) or set(pending["context"]) != {"command"}
                    or self.kind != "resident" or self.stage != "conversation"):
                raise ValueError("A saved action request needs its actual resident meeting and structured intent.")
            command = ActionIntent.model_validate(pending["context"]["command"])
            if command.action == "SAY":
                raise ValueError("A saved control request must identify a supported action.")
        return self
    numeric_point = field_validator("point", mode="before")(_physical_point)


class PlayerReportState(saved_record(PlayerReport)):
    turn: int = Field(strict=True)
    numeric_day = field_validator("day", mode="before")(_physical_number)

    @field_validator("turn", mode="before")
    @classmethod
    def integer_turn(cls, value):
        if type(value) is not int:
            raise ValueError("Player claims require their actual accepted original speech and listener.")
        return value


class JourneyState(saved_record(Journey)):
    point: tuple[PhysicalNumber, PhysicalNumber]
    elapsed: PhysicalNumber = Field(ge=0, le=TRAVEL_STEP_TIMEOUT)
    steps: list[RouteStepState]
    numeric_elapsed = field_validator("elapsed", mode="before")(_physical_number)
    numeric_point = field_validator("point", mode="before")(_physical_point)


class ConcernState(saved_record(ConversationConcern)):
    remaining: float = Field(ge=0, le=CONCERN_LIFETIME_RANGE[1])

    @model_validator(mode="after")
    def validate_account_kind(self):
        if (self.kind in ("recollection", "news")) != bool(self.claim_id):
            raise ValueError("Only recollection and news concerns require an account.")
        return self


class NPCState(saved_record(ResidentValues, exclude={"emotion_intensity"}), ActivityState):
    # Only representation conversions differ from the durable runtime record.
    journey: JourneyState | None
    emotion: EmotionReaction
    preferences: PreferencesState
    memories: list[MemoryState] = Field(max_length=20)
    goal: GoalState | None
    pending_social_opinions: list[OpinionEffectState]
    beliefs: list[BeliefState]
    current_concern: ConcernState | None
    player_opinion: PlayerOpinionState
    player_call: Annotated[PlayerCallState | PlayerReplyCallState, Field(discriminator="kind")] | None
    departure: DepartureState | None

    @field_validator("opinions")
    @classmethod
    def validate_runtime_opinions(cls, value):
        return normalize_opinions(value)

    @field_validator("character")
    @classmethod
    def validate_character(cls, value):
        return normalize_character(value)


class RecordedTurnState(saved_record(RecordedTurn)):
    # Reuse the canonical value's derivations while requiring every saved field.
    text = RecordedTurn.text
    speech = RecordedTurn.speech


class HistoryState(saved_record(ConversationRecord)):
    turns: list[RecordedTurnState]
    failed_turns: list[ConversationTurn]
    transfers: list[TransferState]
    transfer_intent: TransferIntentState | None
    player_promise_offers: list[PlayerAppleOfferState]

    @classmethod
    def capture(cls, record):
        return cls.model_validate(record)

    def restore(self):
        source = self.model_copy(deep=True)
        record = restore_record(ConversationRecord, source)
        record.turns = [restore_record(RecordedTurn, turn) for turn in source.turns]
        record.failed_turns = [turn.model_dump() for turn in source.failed_turns]
        record.transfers = [restore_record(TransferReceipt, receipt) for receipt in source.transfers]
        record.transfer_intent = restore_record(TransferIntent, source.transfer_intent) if source.transfer_intent else None
        record.player_promise_offers = [restore_record(PlayerPromiseOffer, offer) for offer in source.player_promise_offers]
        return record

    def generated_turns(self):
        return ConversationRecord.generated_turns(self)

    @model_validator(mode="after")
    def validate_turns(self):
        validate_history_turns(self.turns)
        return self


class ActiveDialogueState(saved_record(DialogueProgress)):
    @field_validator("dialogue_state")
    @classmethod
    def active_dialogue(cls, value):
        if value == "failed":
            raise ValueError("Failed dialogue cannot remain an active interaction.")
        return value


class ConversationState(saved_record(ResidentProgress), ActiveDialogueState):
    concerns: dict[Literal["initiator", "recipient"], ConcernState | None]
    escalation: EscalationChoice | None
    initiator: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    record: str = Field(min_length=1)
    saved_states: tuple[ActivityState, ActivityState]
    proposals: list[ProposalState] = Field(max_length=2)
    information_context: InformationState | None


class HandoffState(saved_record(HandoffProgress)):
    initiator: str
    recipient: str
    record: str
    saved_states: dict[str, ActivityState]


class TreeState(SaveModel):
    id: str = Field(min_length=1)
    x: float = Field(ge=60, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT - 50)
    fruit: int = Field(strict=True, ge=0)


class SnakeState(saved_record(SnakeValues)):
    location = property(lambda self: "garden")
    journey: ClassVar[None] = None
    departure: ClassVar[None] = None
    goal: ClassVar[None] = None
    player_call: ClassVar[None] = None
    home_reason: ClassVar[None] = None
    room_number: ClassVar[int] = 0
    inventory = property(lambda self: {})
    player_opinion: PlayerOpinionState
    beliefs: list[BeliefState]
    opinions: dict[str, dict]
    memories: list[MemoryState]
    pending_social_opinions: list[OpinionEffectState]
    current_concern: ConcernState | None

    @model_validator(mode="after")
    def validate_position(self):
        from simulation.activity import Behavior
        if self.behavior not in (Behavior.WANDER, Behavior.WAIT) or self.target_id is not None:
            raise ValueError("The snake can wander or join a conversation, without human tasks.")
        ActivitySnapshot.validate(self)
        validate_feet(GARDEN_LOCATION, (self.x, self.y))
        if self.destination is not None:
            validate_destination(self.destination, location=GARDEN_LOCATION)
        return self


PendingActionState = saved_record(PendingAction)


ErrandState = saved_record(Errand)


class PlayerState(saved_record(Player)):
    journey: JourneyState | None

    @model_validator(mode="after")
    def valid_apples(self):
        if set(self.inventory) != {"fruit"}:
            raise ValueError("Invalid player apple inventory.")
        return self


class SimulationState(SaveModel):
    mode: Literal["player", "spectator"]
    player_interaction: dict | None
    player_invitations: list[InvitationState]
    player_claims: list[PlayerReportState]
    visit: HostedVisitState | None
    day: int = Field(ge=1)
    player: PlayerState
    player_call_check: float = Field(ge=0, le=PLAYER_CALL_CHECK_SECONDS)
    player_acknowledgments: list[PlayerAcknowledgmentState]
    npc_trait_mode: Literal["normal", "extreme", "both"]
    events: list[EventState] = Field(max_length=200)
    promises: list[PromiseState]
    facts: list[FactState]
    claims: list[ClaimState]
    disclosures: list[DisclosureState]
    social_rotation: int = Field(ge=0)
    npcs: list[NPCState] = Field(min_length=1, max_length=MAX_RESIDENTS)
    creations: list[Creation]
    tree: TreeState
    snake: SnakeState
    archived: list[NPCState]
    arrival_error: str
    pending_arrival_days: list[int]
    processed_arrival_days: list[int]
    introductions: list[str]
    roster_revision: int = Field(ge=0)
    pending_actions: list[PendingActionState]
    errands: list[ErrandState]
    history: list[HistoryState]
    conversations: list[ConversationState]
    handoffs: list[HandoffState]
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(ge=0.5, le=2.5)

    @model_validator(mode="after")
    def validate_references(self):
        """Every saved reference must name a record that exists, since restoration looks each one up."""
        residents = [*self.npcs, *self.archived]
        actor_ids = {npc.id for npc in residents} | {self.snake.id}
        if len(actor_ids) != len(residents) + 1 or self.player.id in actor_ids:
            raise ValueError("Actor IDs must be unique.")
        live_ids = {npc.id for npc in self.npcs} | {self.snake.id}
        owner_ids = actor_ids | {self.player.id}
        history = {record.id for record in self.history}
        facts = {fact.id for fact in self.facts}
        claims = {claim.id for claim in self.claims}
        for records, ids in ((self.history, history), (self.facts, facts), (self.claims, claims),
                             (self.disclosures, {d.id for d in self.disclosures})):
            if len(records) != len(ids):
                raise ValueError("Duplicate record IDs.")
        interaction = self.player_interaction or {}
        references = [
            *[(c.initiator in live_ids and c.recipient in live_ids and c.record in history) for c in self.conversations],
            *[({h.initiator, h.recipient} <= live_ids | {self.player.id} and h.record in history) for h in self.handoffs],
            interaction.get("npc_id") in (None, *live_ids) and interaction.get("record_id") in (None, *history),
            self.visit is None or self.visit.host_id in live_ids,
            *[offer.conversation_id in history for offer in self.player_invitations],
            *[claim.conversation_id in history for claim in self.player_claims],
            *[ack.record_id in history and ack.npc_id in live_ids for ack in self.player_acknowledgments],
            *[set(record.participant_ids) <= owner_ids for record in self.history],
            *[event.history in (None, *history) and set(event.participants) <= owner_ids for event in self.events],
            *[({f.subject_id, f.object_id, *f.witness_ids} <= owner_ids and f.conversation_id in (None, *history))
              for f in self.facts],
            *[claim.fact_id in facts for claim in self.claims],
            *[d.claim_id in claims and d.conversation_id in history for d in self.disclosures],
            *[p.conversation_id in history and {p.promisor_id, p.recipient_id} <= owner_ids for p in self.promises],
            *[work.owner_id in owner_ids for work in self.creations],
            *[a.npc_id in live_ids and a.target_id in (None, *live_ids, self.player.id) and a.record in (None, *history)
              for a in self.pending_actions],
            *[e.npc_id in live_ids and e.target_id in live_ids and e.record in (None, *history)
              and e.conversation in (None, *history) for e in self.errands],
            *[c.information_context is None or {o.fact_id for o in c.information_context.options} <= facts
              for c in self.conversations],
        ]
        for actor in [*residents, self.snake]:
            references.extend([set(actor.opinions) <= actor_ids - {actor.id},
                               all(belief.claim_id in claims for belief in actor.beliefs)])
        for npc in residents:
            references.extend([npc.goal is None or npc.goal.claim_id in (None, *claims),
                               npc.player_call is None or npc.player_call.kind != "reply" or npc.player_call.reference in history])
        if not all(references):
            raise ValueError("A saved reference names a missing record.")
        return self


class SimulationSave(SaveModel):
    name: str
    created_at: datetime
    updated_at: datetime
    state: SimulationState
