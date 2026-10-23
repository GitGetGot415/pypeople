"""Self-contained simulation saves. No character file paths are loaded."""

import re
from npc import ResidentProgress, Conversation, NPC, Tree
from transfers import HandoffProgress, Handoff
from snake import SnakeProgress, PlayerSnakeProgress, Snake, SnakeEncounter
from dataclasses import asdict, dataclass, is_dataclass
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from types import SimpleNamespace
from pydantic import Field, field_validator, model_validator
from records import SaveModel, saved_record, restore_record
from interactions import DialogueProgress
from character_data import normalize_character, normalize_opinions
from constants import (CONCERN_LIFETIME_RANGE, CONVERSATION_PAUSE_RANGE, CONVERSATION_RETURN_PAUSE,
                      FAILURE_COOLDOWN, CONVERSATION_MAX_TURNS, CONVERSATION_MAX_AGREEMENTS,
                      GIFT_GIVE_SECONDS, SHAKE_SECONDS, PLAYER_ROOM_NUMBER, PLAYER_ROOM_LOCATION,
                      RECOVERY_SECONDS, SNAKE_FRAME_SECONDS, SNAKE_NAME, WORLD_HEIGHT, WORLD_WIDTH,
                      APARTMENT_ID, BEDTIME_DELAY_RANGE, CREATION_WORK_SECONDS_RANGE, CREATIVE_QUEUE_SECONDS,
                      OUTDOOR_COOLDOWN_SECONDS, REST_SECONDS_RANGE, WAKE_DELAY_RANGE, PLAYER_ID,
                      PLAYER_LABEL, PLAYER_NAME_TAG, PLAYER_CALL_COOLDOWN, PLAYER_CALL_CHECK_SECONDS,
                      PLAYER_CALL_LIMIT)
from creations import Creation, CreativeJob
from apartments import HOME_BEHAVIORS, INDOOR_BEHAVIORS, Invitation, HostedVisit
from locations import Journey, RouteStep
from player_records import (PlayerOpinion, PlayerCall, PlayerReplyCall, PlayerReport,
                            PlayerPromiseOffer, PlayerAcknowledgment)
from conversation_history import RecordedTurn, ConversationRecord, Memory, WorldEvent
from dialogue import ConversationTurn, EmotionReaction, GiftOffer
from activity import ActivitySnapshot, Behavior
from simulation import PendingAction, Simulation, TreeCustomer
from player import Player, PlayerInteraction
from purgatory import Attitudes
from player_story import validate_starting_history, valid_introduction_call, resident_arrival_days
from social import (Belief, Claim, ConcernKind, ConversationConcern, Disclosure, Fact, Goal, OpinionEffect,
                   Promise, Proposal, ClaimOption, InformationContext, information_from_dict, phase_index, validate_agreements,
                   validate_information)


PlayerInteractionState = saved_record(PlayerInteraction)

SAVE_DIRECTORY = Path(__file__).resolve().parent / "saves"


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


class DepartureState(SaveModel):
    day: int = Field(ge=1)
    phase: Literal["day", "night"]
    cause: Literal["eaten by the snake"]
    event_id: str = Field(min_length=1)


PlayerOpinionState = saved_record(PlayerOpinion)
PlayerCallState = saved_record(PlayerCall)
PlayerReplyCallState = saved_record(PlayerReplyCall)


class TransferIntentState(SaveModel):
    giver_id: str = Field(min_length=1)
    recipient_id: str = Field(min_length=1)
    item: str = Field(min_length=1)
    quantity: int = Field(strict=True, ge=1)


class TransferState(TransferIntentState):
    id: str = Field(min_length=1)
    day: int = Field(strict=True, ge=1)
    phase: Literal["day", "night"]
    sequence: int = Field(strict=True, ge=0)


PlayerAppleOfferState = saved_record(PlayerPromiseOffer)
PlayerAcknowledgmentState = saved_record(PlayerAcknowledgment)
RouteStepState = saved_record(RouteStep)
InvitationState = saved_record(Invitation)
HostedVisitState = saved_record(HostedVisit)
TreeCustomerState = saved_record(TreeCustomer)
PlayerReportState = saved_record(PlayerReport)


class JourneyState(saved_record(Journey)):
    steps: list[RouteStepState]


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

    @field_validator("journey")
    @classmethod
    def validate_journey(cls, value):
        if value is not None:
            JourneyState.model_validate(value)
        return value
    id: str = Field(min_length=1)
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    reply: str
    rewritten_reply: str
    emotion: EmotionReaction
    social: float = Field(ge=0, le=100)
    inventory: dict[Literal["fruit"], Annotated[int, Field(strict=True, ge=0)]]
    preferences: PreferencesState
    memories: list[MemoryState] = Field(max_length=20)
    goal: GoalState | None
    opinions: dict[str, dict]
    pending_social_opinions: list[OpinionEffectState]
    beliefs: list[BeliefState]
    current_concern: ConcernState | None
    conversation_pause_remaining: float = Field(ge=0, le=max(CONVERSATION_PAUSE_RANGE[1], FAILURE_COOLDOWN))
    conversation_return_remaining: float = Field(ge=0, le=CONVERSATION_RETURN_PAUSE)
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

    @field_validator("opinions")
    @classmethod
    def validate_runtime_opinions(cls, value):
        return normalize_opinions(value)

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


class RecordedTurnState(saved_record(RecordedTurn)):
    # Reuse the canonical value's derivations while requiring every saved field.
    text = RecordedTurn.text
    speech = RecordedTurn.speech


class HistoryState(saved_record(ConversationRecord)):
    turns: list[RecordedTurnState]
    failed_turns: list[ConversationTurn]
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

    @classmethod
    def capture(cls, record):
        return cls.model_validate(asdict(record))

    def restore(self):
        data = self.model_dump()
        data["turns"] = [RecordedTurn(**turn) for turn in data["turns"]]
        data["player_promise_offers"] = [PlayerPromiseOffer(**offer) for offer in data["player_promise_offers"]]
        return ConversationRecord(**data)

    def generated_turns(self):
        return ConversationRecord.generated_turns(self)

    @model_validator(mode="after")
    def validate_turns(self):
        for turn in self.turns:
            if not turn.parts:
                raise ValueError("Historical turns need their original words.")
            for part in turn.parts:
                if set(part) - {"text", "emotion"} or not part["text"].strip():
                    raise ValueError("Invalid historical speech part.")
                if "emotion" in part:
                    EmotionReaction.model_validate(part["emotion"])
            if turn.rewritten and (len(turn.rewritten) != len(turn.parts)
                    or turn.speaker_id == PLAYER_ID or any(not text.strip() for text in turn.rewritten)):
                raise ValueError("A rewrite must match the original NPC dialogue parts.")
        return self


class ActiveDialogueState(saved_record(DialogueProgress)):
    @field_validator("dialogue_state")
    @classmethod
    def active_dialogue(cls, value):
        if value == "failed":
            raise ValueError("Failed dialogue cannot remain an active interaction.")
        return value


class ConversationState(saved_record(ResidentProgress), ActiveDialogueState):
    concerns: dict[Literal["initiator", "recipient"], ConcernState]
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


class SnakeState(SaveModel):
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
    offenses: dict[str, str]
    eaten_today: int = Field(ge=0)
    recovery_remaining: float = Field(ge=0, le=RECOVERY_SECONDS)
    decision_remaining: float = Field(ge=0)
    command_revision: int = Field(ge=0)
    reply: str
    rewritten_reply: str
    retry_request: SnakeRetryState | None

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value):
        if value is not None and not (0 <= value[0] <= WORLD_WIDTH and 0 <= value[1] <= WORLD_HEIGHT):
            raise ValueError("Invalid snake destination.")
        return value


class SnakeEncounterState(saved_record(SnakeProgress), ActiveDialogueState):
    @field_validator("stage")
    @classmethod
    def resident_stage(cls, value):
        if value == "player":
            raise ValueError("Resident encounters cannot use player tree service.")
        return value

    resident_id: str
    record: str = Field(min_length=1)
    saved_state: ActivityState
    request_context: dict | None


PendingActionState = saved_record(PendingAction)


class PlayerState(saved_record(Player)):
    journey: dict | None

    @field_validator("journey")
    @classmethod
    def validate_journey(cls, value):
        if value is not None:
            JourneyState.model_validate(value)
        return value

    @model_validator(mode="after")
    def valid_apples(self):
        if set(self.inventory) != {"fruit"}:
            raise ValueError("Invalid player apple inventory.")
        return self


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
    player_call_check: float = Field(ge=0, le=PLAYER_CALL_CHECK_SECONDS)
    player_acknowledgments: list[dict]
    npc_trait_mode: Literal["normal", "extreme", "both"]
    phase: Literal["day", "night"]
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
    handoffs: list[HandoffState]
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(ge=0.5, le=2.5)

    @field_validator("player_invitations")
    @classmethod
    def validate_invitations(cls, value):
        for row in value:
            InvitationState.model_validate(row)
        return value

    @field_validator("player_claims")
    @classmethod
    def validate_reports(cls, value):
        for row in value:
            PlayerReportState.model_validate(row)
        return value

    @field_validator("visit")
    @classmethod
    def validate_visit(cls, value):
        if value is not None:
            HostedVisitState.model_validate(value)
        return value

    @field_validator("tree_queue")
    @classmethod
    def validate_tree_queue(cls, value):
        for row in value:
            TreeCustomerState.model_validate(row)
        return value

    @field_validator("tree_current")
    @classmethod
    def validate_tree_current(cls, value):
        if value is not None:
            TreeCustomerState.model_validate(value)
        return value

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
        live = {npc.id: npc for npc in self.npcs}
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
            from locations import feet_bounds, valid_location, player_has_room
            if n.room_number == PLAYER_ROOM_NUMBER and player_has_room(self):
                raise ValueError(f"Room {PLAYER_ROOM_NUMBER} is reserved for the player.")
            layout = SimpleNamespace(residents=residents, mode=self.mode, player=self.player)
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
                    if index == n.id:
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
        ids = set(actors)
        for npc in residents:
            if any(target not in ids or target == npc.id for target in npc.opinions):
                raise ValueError("Runtime opinions require distinct resident or snake actor IDs.")
        now = phase_index(self.day, self.phase)
        if len(ids) != len(residents) + 1 or self.tree.id in ids or len({v.casefold() for v in actors.values()}) != len(ids):
            raise ValueError("Actor IDs and stable display names must be unique.")
        if (self.player.id in ids or self.tree.id == self.player.id
                or {PLAYER_LABEL.casefold(), PLAYER_NAME_TAG.casefold()} & {name.casefold() for name in actors.values()}):
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
        validate_starting_history(self, residents)
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
            return actor_id in ids and actor_id in record.participant_ids
        for event in self.events:
            if not set(event.participants) <= ids or phase_index(event.day, event.phase) > now:
                raise ValueError("Invalid event actors or date.")
            if event.history is not None and event.history not in history:
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
                        or record.turns[offer["offer_turn"]].speaker_id != offer["giver_id"]):
                    raise ValueError("Invalid player apple offer authority or source.")
                player_offer_ids.add(offer["id"])
                from dialogue_rules import validate_apple_offer, validate_apple_promise
                validate_apple_offer(record.turns[offer["offer_turn"]].text, offer["deadline"], current_phase=offer["created_phase"])
                if offer["status"] == "accepted":
                    turn = offer["accept_turn"]
                    if (type(turn) is not int or not offer["offer_turn"] < turn < len(record.turns)
                            or record.turns[turn].speaker_id != offer["recipient_id"]
                            or not any(p.id == offer["id"] and p.conversation_id == record.id
                                       and p.promisor_id == offer["giver_id"] and p.recipient_id == offer["recipient_id"]
                                       and p.deadline == offer["deadline"] for p in self.promises)):
                        raise ValueError("An accepted player apple promise needs its actual ordered acceptance.")
                    validate_apple_promise(record.turns[offer["offer_turn"]].text, record.turns[turn].text,
                                           offer["deadline"], current_phase=offer["created_phase"])
                elif offer["accept_turn"] is not None:
                    raise ValueError("An unaccepted apple offer cannot retain an acceptance.")
            if record.authorization_id is not None and record.authorization_id not in history:
                raise ValueError("Missing transfer authorization conversation.")
            if (len(record.participant_ids) != 2 or len(set(record.participants)) != 2
                    or any(not label.strip() for label in record.participants) or not set(record.participant_ids) <= ids
                    or len(set(record.participant_ids)) != 2):
                raise ValueError("Historical participant identities do not match the speakers.")
            if (len(set(record.accepted_player_turns)) != len(record.accepted_player_turns)
                    or any(i >= len(record.turns) or record.turns[i].speaker_id != self.player.id for i in record.accepted_player_turns)):
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
            if (not set(record.intended_subjects) <= set(record.participants)
                    or any(not subject.strip() for subject in record.intended_subjects.values())):
                raise ValueError("Intended subjects must describe participants in their recorded conversation.")
            if any(turn.speaker_id not in record.participant_ids
                   or turn.speaker != record.participants[record.participant_ids.index(turn.speaker_id)]
                   for turn in record.turns):
                raise ValueError("Historical speech must retain its participant identity and recorded label.")
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
                    or disclosure.turn >= len(record.turns) or record.turns[disclosure.turn].speaker_id != disclosure.speaker_id
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
        for handoff in self.handoffs:
            pair = {handoff.initiator, handoff.recipient}
            residents_in_pair = pair - {self.player.id}
            record = record_for(handoff.record)
            if (len(pair) != 2 or not residents_in_pair <= live_ids
                    or residents_in_pair & reserved or set(record.participant_ids) != pair
                    or set(handoff.saved_states) != residents_in_pair):
                raise ValueError("Invalid handoff participants, reservation, or source record.")
            if self.player.id in pair:
                ownership = self.player_interaction
                resident_id = next(iter(residents_in_pair))
                if (not ownership or ownership.get("kind") != "resident"
                        or ownership.get("npc_id") != resident_id or ownership.get("record_id") != record.id
                        or ownership.get("stage") not in ("conversation", "handoff")
                        or self.mode != "player" or not self.player.alive):
                    raise ValueError("A player handoff requires its actual exclusive physical owner.")
            locations = {live[i].location if i != self.player.id else self.player.location for i in pair}
            if len(locations) != 1 or self.player.id in pair and not handoff.arrived:
                raise ValueError("Handoff participants must share their physical location.")
            if (handoff.transferred != bool(handoff.outcome_event_id)
                    or handoff.transferred != (handoff.elapsed >= GIFT_GIVE_SECONDS)
                    or not handoff.arrived and handoff.elapsed != 0
                    or handoff.item != "fruit" and (handoff.item not in creation_ids or handoff.quantity != 1)):
                raise ValueError("Invalid handoff progress or possession.")
            if record.transfer_intent != dict(giver_id=handoff.initiator,
                    recipient_id=handoff.recipient, item=handoff.item, quantity=handoff.quantity):
                raise ValueError("A handoff needs matching accepted authorization.")
            receipts = [t for t in record.transfers if t["giver_id"] == handoff.initiator
                        and t["recipient_id"] == handoff.recipient and t["item"] == handoff.item
                        and t["quantity"] == handoff.quantity]
            if handoff.transferred and (not receipts or handoff.outcome_event_id not in events
                    or events[handoff.outcome_event_id].kind != ("fruit_shared" if handoff.item == "fruit" else "work_shared")
                    or events[handoff.outcome_event_id].history != record.id
                    or set(events[handoff.outcome_event_id].participants) != pair):
                raise ValueError("A committed handoff needs its receipt and event.")
            if handoff.rewritten_thanks and (len(handoff.rewritten_thanks) != len(handoff.thanks)
                    or any(not text.strip() for text in handoff.rewritten_thanks)):
                raise ValueError("Rewritten thanks must match the original parts.")
            reserved.update(residents_in_pair)
            for actor_id, activity in handoff.saved_states.items():
                activity_target(activity, live[actor_id])
        for interaction in self.conversations:
            pair = (interaction.initiator, interaction.recipient)
            if pair[0] == pair[1] or any(i not in live or i in reserved for i in pair) or interaction.record not in history:
                raise ValueError("Invalid live interaction participants.")
            for index, activity in zip(pair, interaction.saved_states):
                if live[pair[0]].location != live[pair[1]].location:
                    raise ValueError("Conversation participants must share a location.")
                activity_target(activity, live[index])
            record = history[interaction.record]
            if set(record.participant_ids) != set(pair):
                raise ValueError("An interaction must match its record.")
            reserved.update(pair)
            thanks = interaction.handoff_thanks
            rewritten = interaction.rewritten_handoff_thanks
            if rewritten and (len(rewritten) != len(thanks) or any(not text.strip() for text in rewritten)):
                raise ValueError("Rewritten thanks must match the original parts.")
            if set(interaction.concerns) != {"initiator", "recipient"}:
                raise ValueError("A meeting must retain both frozen concerns.")
            for role, index in zip(("initiator", "recipient"), pair):
                validate_concern(interaction.concerns[role], live[index])
            if interaction.goal_id and (live[pair[0]].goal is None or interaction.goal_id != live[pair[0]].goal.id):
                raise ValueError("A meeting goal must belong to its initiator.")
            if interaction.focus_claim_id and not any(b.claim_id == interaction.focus_claim_id for b in live[pair[0]].beliefs):
                raise ValueError("A required meeting account must be known to its initiator.")
            if len(record.turns) > CONVERSATION_MAX_TURNS:
                raise ValueError("A resident conversation exceeds its accepted turn budget.")
            ready = interaction.dialogue_state == "ready"
            if record.id != interaction.id or ready != bool(record.generated_turns()) or ready != interaction.social_effects_applied:
                raise ValueError("Invalid conversation commit progress.")
            if interaction.information_context:
                context = interaction.information_context
                if context.participants != {"initiator": live[pair[0]].id, "recipient": live[pair[1]].id}:
                    raise ValueError("Invalid information speakers.")
                if context.focus_claim_id != interaction.focus_claim_id:
                    raise ValueError("The required meeting account must match its information snapshot.")
                if any(o.fact_id not in facts for o in context.options):
                    raise ValueError("Missing snapshot facts.")
            validate_agreements(interaction.accepted_proposals, [p.model_dump() for p in interaction.proposals], record.generated_turns(), live[pair[0]].id, live[pair[1]].id)
            if ready:
                validate_information(SimpleNamespace(turns=record.generated_turns(), disclosures=interaction.accepted_disclosures,
                    third_party_opinions=interaction.third_party_opinions, leverage=interaction.leverage_uses),
                    interaction.information_context.model_dump() if interaction.information_context else None)
            if interaction.accepted_gift:
                from transfers import validate_resident_gift
                validate_resident_gift(interaction.accepted_gift, record.generated_turns(), {
                    "purpose": interaction.purpose, "arrival_focus": interaction.opening_kind == "orientation",
                    **{
                    role: {"npc_id": live[index].id, "inventory": live[index].inventory,
                           "owned_works": [{"id": w.id, "title": w.title} for w in self.creations
                                           if w.owner_id == live[index].id]}
                    for role, index in zip(("initiator", "recipient"), pair)}})
                if interaction.accepted_gift.accept_turn is None:
                    raise ValueError("A pending gift handoff needs an accepted offer.")
        encounter = self.snake_encounter
        if encounter:
            if (encounter.resident_id not in live_ids or encounter.resident_id in indoor_ids
                    or encounter.saved_state.behavior in HOME_BEHAVIORS or encounter.record not in history):
                raise ValueError("A snake encounter needs a living resident.")
            resident_index = next(i for i, n in enumerate(self.npcs) if n.id == encounter.resident_id)
            if encounter.resident_id in reserved or history[encounter.record].id != encounter.id or self.snake.target_id != encounter.resident_id:
                raise ValueError("Conflicting snake encounter reservation.")
            reserved.add(encounter.resident_id)
            if len(history[encounter.record].turns) > CONVERSATION_MAX_TURNS:
                raise ValueError("A snake conversation exceeds its accepted turn budget.")
            if encounter.committed != bool(history[encounter.record].turns) or encounter.committed != (encounter.dialogue_state == "ready") or encounter.committed != bool(encounter.outcome):
                raise ValueError("Invalid snake encounter commit progress.")
            if encounter.reserved and (encounter.committed or encounter.purpose == "introduction" or self.tree.fruit < 1):
                raise ValueError("Invalid apple reservation.")
            if encounter.stage in ("condemned_approach", "eating") and encounter.outcome != "eat":
                raise ValueError("Eating requires a resolved condemnation.")
            activity_target(encounter.saved_state, self.npcs[resident_index])
        elif self.snake.target_id is not None and not self.player_snake_encounter:
            raise ValueError("A snake target requires an encounter.")
        if (not set(self.snake.offenses) <= live_ids | ({self.player.id} if self.player.alive else set())
                or any(f not in facts for f in self.snake.offenses.values())):
            raise ValueError("Invalid snake warning sources.")
        if self.snake.retry_request and self.snake.retry_request.resident_id not in live_ids:
            raise ValueError("Invalid snake retry target.")
        for index, npc in enumerate(self.npcs):
            if npc.id not in reserved:
                activity_target(npc, npc)
            if npc.id in reserved and npc.behavior != Behavior.WAIT:
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
            if action.record is not None and action.record not in history:
                raise ValueError("Invalid queued history record.")
            if action.action == "GIVE_ITEM":
                if (action.target_id is None or action.quantity < 1
                        or action.item != "fruit" and (action.item not in creation_ids or action.quantity != 1)
                        or action.record is None or history[action.record].transfer_intent != dict(
                            giver_id=action.npc_id, recipient_id=action.target_id, item=action.item, quantity=action.quantity)):
                    raise ValueError("A queued gift needs its real item and accepted matching authorization.")
        pending, processed = self.pending_arrival_days, self.processed_arrival_days
        arrival_days = resident_arrival_days(self)
        if pending != sorted(set(pending)) or processed != sorted(set(processed)) or set(pending) & set(processed) or set(pending) | set(processed) != arrival_days:
            raise ValueError("Every dawn must have exactly one pending or completed arrival.")
        arrivals = [n.arrival_day for n in residents if n.arrival_day >= 2]
        if sorted(arrivals) != processed:
            raise ValueError("Completed arrivals must match their preserved residents.")
        if not set(self.introductions) <= live_ids:
            raise ValueError("Invalid introduction identity.")
        reserved_ids = reserved
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
                        or not record.turns or (record.turns[-1].speaker_id, record.turns[-1].text) != (npc.id, npc.reply)
                        or (actions and (not call["closing"] or any(a.record is None or history[a.record].id != record.id for a in actions)))):
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
                    or set(call) != PlayerCallState.model_fields.keys()
                    or call["kind"] not in ("gift", "news", "followup", "introduction")
                    or call["key"] in npc.resolved_player_calls):
                raise ValueError("Invalid waiting player call.")
            if call:
                ref = call["reference"]
                valid = (valid_introduction_call(self, npc, PlayerCall(**call),
                             interaction=PlayerInteraction(**self.player_interaction) if self.player_interaction else None)
                         if call["kind"] == "introduction" else
                         npc.inventory["fruit"] > 0
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
    name: str
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

    @property
    def population(self):
        return self.npc_count + int(self.mode == "player")


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


class PlayerSnakeState(saved_record(PlayerSnakeProgress), ActiveDialogueState):
    record: str = Field(min_length=1)


def capture_player_snake(encounter, records):
    from snake import PlayerSnakeEncounter
    if not isinstance(encounter, PlayerSnakeEncounter):
        return None
    return _capture_fields(PlayerSnakeState, encounter, record=records[id(encounter.record)]).model_dump()


def restore_player_snake(world, saved, history):
    if saved is None:
        return
    from snake import PlayerSnakeEncounter
    values = PlayerSnakeState.model_validate(saved)
    PlayerSnakeEncounter.restore(world, world.player, history[values.record],
        progress=restore_record(PlayerSnakeProgress, values),
        dialogue=restore_record(DialogueProgress, values),
        saved_state=None)


def validate_physical_state(state):
    """Canonical map/ownership checks supplement the existing effect provenance checks."""
    import math
    import json
    from locations import valid_location, feet_bounds, route, floor_for_room, targets
    from constants import PLAYER_APPROACH_TIMEOUT, TRAVEL_STEP_TIMEOUT, ROOM_ARRIVAL, GARDEN_VISIT_POINT
    residents = [*state.npcs, *state.archived]
    layout = SimpleNamespace(residents=residents, mode=state.mode, player=state.player)
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
            if set(journey) != JourneyState.model_fields.keys() or not valid_location(layout, journey["destination"]):
                raise ValueError("Invalid journey identity.")
            point(journey["destination"], journey["point"])
            if (actor is state.player and journey["destination"].startswith("room:")
                    and journey["destination"] not in (PLAYER_ROOM_LOCATION, actor.location)
                    and (not state.visit or state.visit["location"] != journey["destination"])):
                raise ValueError("Travel into a resident room requires its accepted hosted visit.")
            if not 0 <= journey["elapsed"] <= TRAVEL_STEP_TIMEOUT or not journey["steps"]:
                raise ValueError("Invalid journey progress.")
            expected = [asdict(step) for step in route(actor.location, journey["destination"], journey["point"])]
            if journey["steps"] != [{**e, "point": list(e["point"]), "arrival": list(e["arrival"]) if e["arrival"] is not None else None} for e in expected] and journey["steps"] != expected:
                raise ValueError("A journey must follow the actual doors and elevator connections.")
    interaction = state.player_interaction
    exclusive = set()
    if interaction:
        if interaction.get("kind") not in ("resident", "door", "snake") or not interaction.get("id"):
            raise ValueError("Invalid player interaction owner.")
        try:
            physical = PlayerInteraction(**interaction)
        except TypeError:
            raise ValueError("Invalid player interaction fields.") from None
        if not physical.capture().keys() <= interaction.keys():
            raise ValueError("Missing required physical interaction fields.")
        PlayerInteractionState.model_validate(asdict(physical))
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
        if set(entry) != TreeCustomerState.model_fields.keys():
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
    reserved_ids = {i for c in [*state.conversations, *state.handoffs] for i in (c.initiator, c.recipient) if i != state.player.id}
    player_handoff_ids = {actor_id for h in state.handoffs if state.player.id in (h.initiator, h.recipient)
                          and interaction and h.record == interaction.get("record_id")
                          for actor_id in (h.initiator, h.recipient) if actor_id == interaction.get("npc_id")}
    if exclusive & (reserved_ids - player_handoff_ids):
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
        if (set(claim) != PlayerReportState.model_fields.keys()
                or claim["id"] in claim_ids or claim["speaker_id"] != state.player.id or claim["kind"] != "report"
                or not record or type(turn) is not int or turn not in record.accepted_player_turns
                or (record.turns[turn].speaker_id, record.turns[turn].text) != (state.player.id, claim["text"])
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
                or (record.turns[offer["turn"]].speaker_id, record.turns[offer["turn"]].text) != (speaker.id, offer["speech"])
                or offer["location"] != (f"room:{speaker.room_number}" if offer["destination"] == "home" else "garden")):
            raise ValueError("Invitations must retain their real speaker, speech and destination.")
        expected_point = {"garden": GARDEN_VISIT_POINT, "home": ROOM_ARRIVAL,
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

    saved = state.player_snake_encounter
    if saved:
        PlayerSnakeState.model_validate(saved)
        if not state.tree_current or state.tree_current["actor_id"] != state.player.id:
            raise ValueError("Player tree service needs its current FIFO association.")
        if saved["record"] not in records or saved["record"] != saved["id"] or state.snake.target_id != state.player.id:
            raise ValueError("Player tree service must match its original encounter.")
        if saved["stage"] not in ("player", "dialogue", "condemned_approach", "eating") or saved["stage"] != "player" and (not saved["committed"] or saved["outcome"] != "eat"):
            raise ValueError("Invalid player eating progress.")
        if saved["question_account_id"] and not any(b.claim_id == saved["question_account_id"] for b in state.snake.beliefs):
            raise ValueError("The current guard question must reference its actual knowledge.")


def _record_value(value):
    return asdict(value) if is_dataclass(value) else value


def _restore_journey(value):
    if value is None:
        return None
    return Journey(**{**value, "steps": [RouteStep(**step) for step in value["steps"]]})


def _capture_fields(model, source, **overrides):
    """Use the save model's fields; spell out only reference/type conversions."""
    return model(**{name: overrides[name] if name in overrides else _record_value(getattr(source.dialogue if name in DialogueProgress.__dataclass_fields__ else source, name))
                    for name in model.model_fields})


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
        player_call=None if stale_call else _record_value(npc.player_call),
        player_call_cooldown=PLAYER_CALL_COOLDOWN if stale_call else npc.player_call_cooldown,
        resolved_player_calls=([*npc.resolved_player_calls, npc.player_call.key]
            if stale_call and not npc.unread_player_reply else npc.resolved_player_calls.copy()),
    )


def capture_simulation(world, *, camera_offset, camera_zoom):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    records = {id(record): record.id for record in world.history}
    history = [HistoryState.capture(record).model_dump() for record in world.history]
    unread_records = {npc.unread_player_reply.reference for npc in world.npcs if npc.unread_player_reply}
    for record in history:
        for offer in record["player_promise_offers"]:
            if offer["status"] == "offered" and record["id"] not in unread_records:
                offer["status"] = "closed"
    interaction = world.player_interaction
    if interaction and interaction.pending_request:
        record = next((row for row in history if row["id"] == interaction.record_id), None)
        if record:
            record["status"] = "Reply pending when saved"
            record["outcomes"].insert(0, "The reply was not saved. Send the request again after loading.")
    npcs, conversations, handoffs = [], [], []
    captured_handoffs = set()
    for npc in world.npcs:
        npcs.append(capture_resident(npc))
        gift = npc.gift
        if gift is not None and id(gift) not in captured_handoffs:
            captured_handoffs.add(id(gift))
            handoffs.append(_capture_fields(HandoffState, gift,
                initiator=gift.initiator.id, recipient=gift.recipient.id,
                record=records[id(gift.record)],
            ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(_capture_fields(ConversationState, conversation,
            initiator=npc.id, recipient=conversation.recipient.id,
            record=records[id(conversation.record)],
            proposals=[asdict(p) for p in conversation.proposals],
            information_context=asdict(conversation.information_context) if conversation.information_context else None,
            concerns={role: asdict(c) if c else None for role, c in conversation.concerns.items()},
        ))
    encounter = world.snake.encounter
    return SimulationState(
        day=world.day, npc_trait_mode=world.npc_trait_mode, phase=world.phase,
        mode=world.mode, player_interaction=world.player_interaction.capture() if world.player_interaction else None,
        player_invitations=[asdict(o) for o in world.player_invitations], player_claims=[asdict(c) for c in world.player_claims],
        visit=_record_value(world.visit), tree_queue=[asdict(e) for e in world.tree_queue], tree_current=_record_value(world.tree_current),
        player_snake_encounter=capture_player_snake(encounter, records),
        player=PlayerState(**asdict(world.player)), player_call_check=max(0.0, world.player_call_check),
        player_acknowledgments=[asdict(a) for a in world.player_acknowledgments],
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
        npcs=npcs, history=history, conversations=conversations, handoffs=handoffs,
        tree=_capture_fields(TreeState, world.tree,
                       shaker=world.tree.shaker.id if world.tree.shaker is not None else None,
                       reserved_by=world.tree.reserved_by.id if world.tree.reserved_by is not None else None),
        pending_actions=[asdict(action) for action in world.pending_actions],
        camera_offset=tuple(camera_offset),
        camera_zoom=camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    state = SimulationState.model_validate(state.model_dump())
    all_residents = []
    for saved in [*state.npcs, *state.archived]:
        npc = NPC(
            id=saved.id, character=saved.character, display_name=saved.display_name, x=saved.x, y=saved.y,
            reply=saved.reply, emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            social=saved.social, opinions=deepcopy(saved.opinions),
            sprite_size=saved.sprite_size, room_number=saved.room_number, location=saved.location, journey=_restore_journey(saved.journey), social_fatigue=saved.social_fatigue,
            home_remaining=saved.home_remaining, outdoor_remaining=saved.outdoor_remaining,
            bedtime_remaining=saved.bedtime_remaining, wake_remaining=saved.wake_remaining,
            home_reason=saved.home_reason, last_creative_day=saved.last_creative_day,
            creative_job=saved.creative_job.model_copy(deep=True) if saved.creative_job else None,
            inventory=saved.inventory.copy(),
            rewritten_reply=saved.rewritten_reply,
            preferences=saved.preferences.model_dump(), command_revision=saved.command_revision,
            memories=[Memory(**memory.model_dump()) for memory in saved.memories],
            goal=Goal(**saved.goal.model_dump()) if saved.goal else None,
            pending_social_opinions=[OpinionEffect(**effect.model_dump()) for effect in saved.pending_social_opinions],
            beliefs=[Belief(**belief.model_dump()) for belief in saved.beliefs],
            current_concern=ConversationConcern(**saved.current_concern.model_dump()) if saved.current_concern else None,
            conversation_pause_remaining=saved.conversation_pause_remaining,
            conversation_return_remaining=saved.conversation_return_remaining,
            attitudes=saved.attitudes.model_dump(),
            arrival_day=saved.arrival_day, departure=saved.departure.model_dump() if saved.departure else None,
        )
        npc.restore_activity(ActivitySnapshot.capture(saved))
        all_residents.append(npc)
    npcs = all_residents[:len(state.npcs)]
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit,
                shake_elapsed=state.tree.shake_elapsed, shaker=by_id.get(state.tree.shaker),
                reserved_by=by_id.get(state.tree.reserved_by))
    world = Simulation(npcs, tree, assign_goals=False, npc_trait_mode=state.npc_trait_mode, mode=state.mode, initialize_opinions=False)
    world.archived = all_residents[len(state.npcs):]
    world.player = Player(**{**state.player.model_dump(), "journey": _restore_journey(state.player.journey)})
    world.player.world = world
    world.player_invitations = [Invitation(**deepcopy(o)) for o in state.player_invitations]
    world.player_claims = [PlayerReport(**deepcopy(c)) for c in state.player_claims]
    world.visit = HostedVisit(**deepcopy(state.visit)) if state.visit else None
    world.player_interaction = PlayerInteraction(**deepcopy(state.player_interaction)) if state.player_interaction else None
    world.tree_queue = [TreeCustomer(**deepcopy(e)) for e in state.tree_queue]
    world.tree_current = TreeCustomer(**deepcopy(state.tree_current)) if state.tree_current else None
    for entry in [*world.tree_queue, *([world.tree_current] if world.tree_current else [])]:
        world.reservations.acquire("tree", entry, world.participant_by_id(entry.actor_id))
    world.player_call_check = state.player_call_check
    world.player_acknowledgments = [PlayerAcknowledgment(**a) for a in state.player_acknowledgments]
    for npc, saved in zip(all_residents, [*state.npcs, *state.archived]):
        npc.player_opinion = PlayerOpinion(**saved.player_opinion)
        npc.player_call = ((PlayerReplyCall(**{**saved.player_call, "gift_offer": GiftOffer.model_validate(saved.player_call["gift_offer"]) if saved.player_call["gift_offer"] else None})
                            if saved.player_call["kind"] == "reply" else PlayerCall(**saved.player_call)) if saved.player_call else None)
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
    history = [saved.restore() for saved in state.history]
    world.history = history
    history = {record.id: record for record in history}
    world.events = [WorldEvent(**event.model_dump()) for event in state.events]
    world.promises = [Promise(**promise.model_dump()) for promise in state.promises]
    world.facts = [Fact(**fact.model_dump()) for fact in state.facts]
    world.claims = [Claim(**claim.model_dump()) for claim in state.claims]
    world.disclosures = [Disclosure(**disclosure.model_dump()) for disclosure in state.disclosures]
    world.social_rotation = state.social_rotation
    for saved in state.conversations:
        Conversation.restore(by_id[saved.initiator], by_id[saved.recipient], history[saved.record],
            progress=restore_record(ResidentProgress, saved),
            dialogue=restore_record(DialogueProgress, saved),
            proposals=[Proposal(**proposal.model_dump()) for proposal in saved.proposals],
            information_context=information_from_dict(saved.information_context.model_dump()) if saved.information_context else None,
            concerns={role: ConversationConcern(**c.model_dump()) if c else None for role, c in saved.concerns.items()},
            saved_states=[activity.model_dump() for activity in saved.saved_states])
    actor_lookup = {actor.id: actor for actor in [*world.actors, world.player]}
    for saved in state.handoffs:
        Handoff.restore(actor_lookup[saved.initiator], actor_lookup[saved.recipient], history[saved.record],
            progress=restore_record(HandoffProgress, saved),
            saved_states={actor_id: activity.model_dump() for actor_id, activity in saved.saved_states.items()})
    saved = state.snake_encounter
    if saved:
        SnakeEncounter.restore(world, by_id[saved.resident_id], history[saved.record],
            progress=restore_record(SnakeProgress, saved),
            dialogue=restore_record(DialogueProgress, saved),
            saved_state=saved.saved_state.model_dump(), request_context=deepcopy(saved.request_context))
    restore_player_snake(world, state.player_snake_encounter, history)
    if world.player_interaction is not None:
        world.player_interaction = world.player_interaction
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
