"""Self-contained simulation saves. No character file paths are loaded."""

import re
from npc import ResidentProgress, Conversation, NPC, Tree, validate_saved_conversations
from transfers import HandoffProgress, Handoff, validate_transfer_history, validate_saved_handoffs
from transfer_records import TransferIntent, TransferReceipt
from snake import (SnakeProgress, PlayerSnakeProgress, Snake, SnakeEncounter,
                   validate_saved_snake, validate_saved_tree_service)
from dataclasses import asdict, dataclass
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
                       FAILURE_COOLDOWN, SHAKE_SECONDS, RECOVERY_SECONDS, SNAKE_FRAME_SECONDS, SNAKE_NAME,
                       WORLD_HEIGHT, WORLD_WIDTH, GARDEN_BUILDING_POINT, BEDTIME_DELAY_RANGE, CREATION_WORK_SECONDS_RANGE,
                       OUTDOOR_COOLDOWN_SECONDS, REST_SECONDS_RANGE, WAKE_DELAY_RANGE, PLAYER_LABEL,
                       PLAYER_NAME_TAG, PLAYER_CALL_COOLDOWN, PLAYER_CALL_CHECK_SECONDS, TRAVEL_STEP_TIMEOUT,
                       PLAYER_APPROACH_TIMEOUT, CONVERSATION_REPLY_PENDING_WHEN_SAVED, MAX_RESIDENTS)
from creations import Creation, CreativeJob
from apartments import (Invitation, HostedVisit,
                        validate_saved_visit, validate_saved_invitations, validate_saved_home)
from locations import Journey, RouteStep, validate_saved_locations
from player_records import (PlayerOpinion, PlayerCall, PlayerReplyCall, PlayerReport,
                            PlayerPromiseOffer, PlayerAcknowledgment, validate_player_reports)
from conversation_history import (RecordedTurn, ConversationRecord, Memory, WorldEvent,
                                  validate_history_turns, validate_history_events)
from dialogue import ConversationTurn, EmotionReaction
from activity import ActivitySnapshot
from simulation import PendingAction, Simulation, TreeCustomer, validate_saved_tree_queue, validate_saved_actions
from player import (Player, PlayerInteraction, validate_saved_calls, validate_saved_interaction,
                    validate_saved_player_mode)
from purgatory import Attitudes
from player_story import validate_starting_history, resident_arrival_days
from social import (Belief, Claim, ConversationConcern, Disclosure, Fact, Goal, OpinionEffect,
                   Promise, Proposal, ClaimOption, InformationContext, phase_index,
                   validate_social_records, validate_saved_promises)


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

    def restore(self):
        return restore_record(InformationContext, self,
            beliefs={role: [restore_record(Belief, belief) for belief in entries]
                     for role, entries in self.beliefs.items()},
            options=[restore_record(ClaimOption, option) for option in self.options])


class DepartureState(SaveModel):
    day: int = Field(ge=1)
    phase: Literal["day", "night"]
    cause: Literal["eaten by the snake"]
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


class TreeCustomerState(saved_record(TreeCustomer)):
    saved_state: PhysicalActivityState | None
    elapsed: PhysicalNumber
    numeric_elapsed = field_validator("elapsed", mode="before")(_physical_number)


class PlayerInteractionState(saved_record(PlayerInteraction)):
    saved_state: PhysicalActivityState | None
    point: tuple[PhysicalNumber, PhysicalNumber] | None
    elapsed: PhysicalNumber = Field(ge=0, le=PLAYER_APPROACH_TIMEOUT)
    numeric_elapsed = field_validator("elapsed", mode="before")(_physical_number)
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


class NPCState(ActivityState):
    location: str
    journey: JourneyState | None
    id: str = Field(min_length=1)
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=GARDEN_BUILDING_POINT[1], le=WORLD_HEIGHT)
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
    player_opinion: PlayerOpinionState
    player_call: Annotated[PlayerCallState | PlayerReplyCallState, Field(discriminator="kind")] | None
    player_call_cooldown: float = Field(ge=0, le=PLAYER_CALL_COOLDOWN)
    resolved_player_calls: list[str]
    command_revision: int = Field(ge=0)
    attitudes: Attitudes
    arrival_day: int = Field(ge=1)
    departure: DepartureState | None
    sprite_size: tuple[int, int] | None
    room_number: int = Field(strict=True, ge=1, le=MAX_RESIDENTS)
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
    y: float = Field(ge=GARDEN_BUILDING_POINT[1], le=WORLD_HEIGHT)
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
        if value is not None and not (0 <= value[0] <= WORLD_WIDTH and GARDEN_BUILDING_POINT[1] <= value[1] <= WORLD_HEIGHT):
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
    journey: JourneyState | None

    @model_validator(mode="after")
    def valid_apples(self):
        if set(self.inventory) != {"fruit"}:
            raise ValueError("Invalid player apple inventory.")
        return self


class PlayerSnakeState(saved_record(PlayerSnakeProgress), ActiveDialogueState):
    record: str = Field(min_length=1)



class SimulationState(SaveModel):
    mode: Literal["player", "spectator"]
    player_interaction: dict | None
    player_invitations: list[InvitationState]
    player_claims: list[PlayerReportState]
    visit: HostedVisitState | None
    tree_queue: list[TreeCustomerState]
    tree_current: TreeCustomerState | None
    player_snake_encounter: PlayerSnakeState | None
    day: int = Field(ge=1)
    player: PlayerState
    player_call_check: float = Field(ge=0, le=PLAYER_CALL_CHECK_SECONDS)
    player_acknowledgments: list[PlayerAcknowledgmentState]
    npc_trait_mode: Literal["normal", "extreme", "both"]
    phase: Literal["day", "night"]
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
    snake_encounter: SnakeEncounterState | None
    pending_actions: list[PendingActionState]
    history: list[HistoryState]
    conversations: list[ConversationState]
    handoffs: list[HandoffState]
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(ge=0.5, le=2.5)

    @model_validator(mode="after")
    def validate_references(self):
        from locations import npc_capacity
        if len(self.npcs) > npc_capacity(self):
            raise ValueError("The living population exceeds apartment capacity.")
        residents = [*self.npcs, *self.archived]
        live = {npc.id: npc for npc in self.npcs}
        live_ids = set(live)
        indoor_ids = {n.id for n in self.npcs if n.location != "garden"}
        history = {h.id: h for h in self.history}
        events = {e.id: e for e in self.events}
        facts = {f.id: f for f in self.facts}
        claims = {c.id: c for c in self.claims}
        disclosures = {d.id: d for d in self.disclosures}
        try:
            physical = validate_physical_state(self, residents, live, history)
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ValueError("Invalid physical state: required ownership, location or source data is missing.") from exc
        owner_ids = {self.player.id, *[n.id for n in residents]}
        creation_ids = validate_saved_home(residents, self.creations, self.player, self.mode, self.day, self.phase,
                                          live_ids, owner_ids, self.visit, physical, self.conversations)
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
        activities = [*residents, *[state for c in self.conversations for state in c.saved_states],
                      *[state for h in self.handoffs for state in h.saved_states.values()]]
        for owner in [self.snake_encounter, physical, self.visit, self.tree_current, *self.tree_queue]:
            if owner and getattr(owner, "saved_state", None):
                activities.append(owner.saved_state)
        if any(not set(state.eating_witness_ids) <= ids - {self.player.id} for state in activities):
            raise ValueError("An unfinished apple action requires known observer identities.")
        if self.social_rotation >= len(self.npcs):
            raise ValueError("Invalid social rotation.")
        if any(n.departure is not None for n in self.npcs) or any(n.departure is None for n in self.archived):
            raise ValueError("Live and departed rosters disagree.")
        validate_starting_history(self, residents)
        for index, source in ((history, self.history), (events, self.events), (facts, self.facts), (claims, self.claims), (disclosures, self.disclosures)):
            if len(index) != len(source):
                raise ValueError("Duplicate social record IDs.")
        if len({(f.source_id, f.kind) for f in self.facts}) != len(self.facts):
            raise ValueError("Duplicate fact sources.")
        validate_history_events(self.events, history, ids, now)
        validate_transfer_history(history, self.creations, self.promises, self.player_acknowledgments,
                                  owner_ids, live_ids, self.player.id, now, creation_ids, ids)
        validate_social_records(facts, claims, disclosures, events, history, residents, self.snake,
                                ids, live_ids, self.day, now)
        validate_saved_promises(self.promises, history, ids, live_ids, self.player.id, now)
        reserved = set()
        validate_saved_handoffs(self.handoffs, history, live, live_ids, self.player, physical,
                                self.mode, creation_ids, events, reserved, self.tree.id)
        validate_saved_conversations(self.conversations, history, live, live_ids, facts, claims,
                                     self.creations, reserved, self.tree.id)
        validate_saved_snake(self.snake_encounter, self.player_snake_encounter, self.snake, self.npcs,
                             self.player, self.tree, history, facts, live_ids, indoor_ids, reserved)
        validate_saved_actions(self.npcs, self.player, self.mode, self.tree, self.pending_actions,
                               live_ids, indoor_ids, reserved, history, creation_ids)
        pending, processed = self.pending_arrival_days, self.processed_arrival_days
        arrival_days = resident_arrival_days(self)
        if pending != sorted(set(pending)) or processed != sorted(set(processed)) or set(pending) & set(processed) or set(pending) | set(processed) != arrival_days:
            raise ValueError("Every dawn must have exactly one pending or completed arrival.")
        arrivals = [n.arrival_day for n in residents if n.arrival_day >= 2]
        if sorted(arrivals) != processed:
            raise ValueError("Completed arrivals must match their preserved residents.")
        if not set(self.introductions) <= live_ids:
            raise ValueError("Invalid introduction identity.")
        validate_saved_calls(residents, self.npcs, self.player, self.facts, history, self.creations,
                             self.promises, self.pending_actions, physical, live_ids, reserved, creation_ids)
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




def capture_player_snake(encounter, records):
    from snake import PlayerSnakeEncounter
    if not isinstance(encounter, PlayerSnakeEncounter):
        return None
    return _capture_fields(PlayerSnakeState, encounter, record=records[id(encounter.record)])


def restore_player_snake(world, saved, history):
    if saved is None:
        return
    from snake import PlayerSnakeEncounter
    values = saved
    PlayerSnakeEncounter.restore(world, world.player, history[values.record],
        progress=restore_record(PlayerSnakeProgress, values),
        dialogue=restore_record(DialogueProgress, values),
        saved_state=None)


def _parse_player_interaction(value):
    if value is None:
        return None
    try:
        physical = PlayerInteraction(**value)
    except TypeError:
        raise ValueError("Invalid player interaction fields.") from None
    if not physical.capture().keys() <= value.keys():
        raise ValueError("Missing required physical interaction fields.")
    validated = PlayerInteractionState.model_validate(asdict(physical))
    return _restore_saved_activity(restore_record(PlayerInteraction, validated))


def validate_physical_state(state, residents, live, records):
    """Canonical map/ownership checks supplement the existing effect provenance checks."""
    layout = SimpleNamespace(residents=residents, mode=state.mode, player=state.player)
    by_id = {**live, state.player.id: state.player}
    validate_saved_player_mode(state.mode, state.player, state.player_interaction, state.visit,
                               state.player_snake_encounter, state.npcs, state.tree_queue, state.tree_current)
    validate_saved_locations([state.player, *residents], layout, state.player, state.visit)
    interaction = _parse_player_interaction(state.player_interaction)
    exclusive = validate_saved_interaction(interaction, state.player, by_id, records, state.player_snake_encounter, layout)
    entries = [*state.tree_queue, *([state.tree_current] if state.tree_current else [])]
    validate_saved_tree_queue(entries, state.player, interaction, by_id, state.pending_actions, exclusive)
    reserved_ids = {i for c in [*state.conversations, *state.handoffs] for i in (c.initiator, c.recipient) if i != state.player.id}
    player_handoff_ids = {actor_id for h in state.handoffs if state.player.id in (h.initiator, h.recipient)
                          and interaction and h.record == interaction.record_id
                          for actor_id in (h.initiator, h.recipient) if actor_id == interaction.npc_id}
    if exclusive & (reserved_ids - player_handoff_ids):
        raise ValueError("An actor cannot belong to two exclusive interactions.")
    validate_saved_visit(state.visit, state.player_invitations, state.player, by_id, layout, interaction, entries,
                         reserved_ids, {fact.id: fact for fact in state.facts})
    validate_player_reports(state.player_claims, records, state.player.id, phase_index(state.day, state.phase))
    validate_saved_invitations(state.player_invitations, records, residents, state.player.id, interaction,
                               Tree(x=state.tree.x, y=state.tree.y).interaction_point)
    validate_saved_tree_service(state.snake_encounter, state.tree_current, state.tree.reserved_by,
                                state.player_snake_encounter, state.player.id, state.snake, records)
    return interaction


def _restore_journey(value):
    if value is None:
        return None
    return restore_record(Journey, value, steps=[restore_record(RouteStep, step) for step in value.steps])


def _restore_saved_activity(owner):
    """Convert the same nested activity value after its owner has been attached."""
    if owner.saved_state is not None:
        owner.saved_state = restore_record(ActivitySnapshot, owner.saved_state)
    return owner


def _capture_fields(model, source, **overrides):
    """Use the save model's fields; spell out only reference/type conversions."""
    return model(**{name: overrides[name] if name in overrides else getattr(source.dialogue if name in DialogueProgress.__dataclass_fields__ else source, name)
                    for name in model.model_fields})


def capture_resident(npc):
    from player import call_valid
    stale_call = npc.player_call and not call_valid(npc.world, npc, npc.player_call)
    return _capture_fields(
        NPCState, npc, emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
        player_call=None if stale_call else npc.player_call,
        player_call_cooldown=PLAYER_CALL_COOLDOWN if stale_call else npc.player_call_cooldown,
        resolved_player_calls=([*npc.resolved_player_calls, npc.player_call.key]
            if stale_call and not npc.unread_player_reply else npc.resolved_player_calls.copy()),
    )


def capture_simulation(world, *, camera_offset, camera_zoom):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    records = {id(record): record.id for record in world.history}
    history = [HistoryState.capture(record) for record in world.history]
    unread_records = {npc.unread_player_reply.reference for npc in world.npcs if npc.unread_player_reply}
    for record in history:
        for offer in record.player_promise_offers:
            if offer.status == "offered" and record.id not in unread_records:
                offer.status = "closed"
    interaction = world.player_interaction
    if interaction and interaction.pending_request:
        record = next((row for row in history if row.id == interaction.record_id), None)
        if record:
            record.status = CONVERSATION_REPLY_PENDING_WHEN_SAVED
            record.outcomes.insert(0, "The reply was not saved. Send the request again after loading.")
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
        ))
    encounter = world.snake.encounter
    return SimulationState(
        day=world.day, npc_trait_mode=world.npc_trait_mode, phase=world.phase,
        mode=world.mode, player_interaction=world.player_interaction.capture() if world.player_interaction else None,
        player_invitations=world.player_invitations, player_claims=world.player_claims,
        visit=world.visit, tree_queue=world.tree_queue, tree_current=world.tree_current,
        player_snake_encounter=capture_player_snake(encounter, records),
        player=PlayerState.model_validate(world.player), player_call_check=max(0.0, world.player_call_check),
        player_acknowledgments=world.player_acknowledgments,
        creations=world.creations,
        events=world.events,
        promises=world.promises,
        facts=world.facts, claims=world.claims,
        disclosures=world.disclosures,
        social_rotation=world.social_rotation,
        snake=_capture_fields(SnakeState, world.snake),
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
        pending_actions=world.pending_actions,
        camera_offset=tuple(camera_offset),
        camera_zoom=camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    # Revalidate the external boundary and isolate all mutable values before actor attachment.
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
            memories=[restore_record(Memory, memory) for memory in saved.memories],
            goal=restore_record(Goal, saved.goal) if saved.goal else None,
            pending_social_opinions=[restore_record(OpinionEffect, effect) for effect in saved.pending_social_opinions],
            beliefs=[restore_record(Belief, belief) for belief in saved.beliefs],
            current_concern=restore_record(ConversationConcern, saved.current_concern) if saved.current_concern else None,
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
    world.player = restore_record(Player, state.player)
    world.player.journey = _restore_journey(state.player.journey)
    world.player.world = world
    world.player_invitations = [restore_record(Invitation, o) for o in state.player_invitations]
    world.player_claims = [restore_record(PlayerReport, c) for c in state.player_claims]
    world.visit = restore_record(HostedVisit, state.visit) if state.visit else None
    if world.visit:
        _restore_saved_activity(world.visit)
    world.player_interaction = _parse_player_interaction(state.player_interaction)
    world.tree_queue = [restore_record(TreeCustomer, e) for e in state.tree_queue]
    world.tree_current = restore_record(TreeCustomer, state.tree_current) if state.tree_current else None
    for entry in [*world.tree_queue, *([world.tree_current] if world.tree_current else [])]:
        _restore_saved_activity(entry)
        world.reservations.acquire("tree", entry, world.participant_by_id(entry.actor_id))
    world.player_call_check = state.player_call_check
    world.player_acknowledgments = [restore_record(PlayerAcknowledgment, a) for a in state.player_acknowledgments]
    for npc, saved in zip(all_residents, [*state.npcs, *state.archived]):
        npc.player_opinion = restore_record(PlayerOpinion, saved.player_opinion)
        npc.player_call = (restore_record(PlayerReplyCall if saved.player_call.kind == "reply" else PlayerCall,
                                          saved.player_call) if saved.player_call else None)
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
    world.pending_actions = [restore_record(PendingAction, action) for action in state.pending_actions]
    history = [saved.restore() for saved in state.history]
    world.history = history
    history = {record.id: record for record in history}
    world.events = [restore_record(WorldEvent, event) for event in state.events]
    world.promises = [restore_record(Promise, promise) for promise in state.promises]
    world.facts = [restore_record(Fact, fact) for fact in state.facts]
    world.claims = [restore_record(Claim, claim) for claim in state.claims]
    world.disclosures = [restore_record(Disclosure, disclosure) for disclosure in state.disclosures]
    world.social_rotation = state.social_rotation
    for saved in state.conversations:
        Conversation.restore(by_id[saved.initiator], by_id[saved.recipient], history[saved.record],
            progress=restore_record(ResidentProgress, saved),
            dialogue=restore_record(DialogueProgress, saved),
            proposals=[restore_record(Proposal, proposal) for proposal in saved.proposals],
            information_context=saved.information_context.restore() if saved.information_context else None,
            concerns={role: restore_record(ConversationConcern, c) if c else None for role, c in saved.concerns.items()},
            saved_states=[restore_record(ActivitySnapshot, activity) for activity in saved.saved_states])
    actor_lookup = {actor.id: actor for actor in [*world.actors, world.player]}
    for saved in state.handoffs:
        Handoff.restore(actor_lookup[saved.initiator], actor_lookup[saved.recipient], history[saved.record],
            progress=restore_record(HandoffProgress, saved),
            saved_states={actor_id: restore_record(ActivitySnapshot, activity) for actor_id, activity in saved.saved_states.items()})
    saved = state.snake_encounter
    if saved:
        SnakeEncounter.restore(world, by_id[saved.resident_id], history[saved.record],
            progress=restore_record(SnakeProgress, saved),
            dialogue=restore_record(DialogueProgress, saved),
            saved_state=restore_record(ActivitySnapshot, saved.saved_state), request_context=deepcopy(saved.request_context))
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
