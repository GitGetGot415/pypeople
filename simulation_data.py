"""Self-contained simulation saves. No character file paths are loaded."""

import re
from npc import ResidentProgress, Conversation, NPC, ResidentValues, Tree, validate_saved_conversations
from transfers import HandoffProgress, Handoff, validate_transfer_history, validate_saved_handoffs
from transfer_records import TransferIntent, TransferReceipt
from snake import Snake, SnakeValues
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, ClassVar
from types import SimpleNamespace
from pydantic import Field, field_validator, model_validator
from records import SaveContext, SaveModel, saved_record, restore_record
from interactions import DialogueProgress
from character_data import normalize_character, normalize_opinions
from constants import (CONCERN_LIFETIME_RANGE, SNAKE_NAME, WORLD_HEIGHT, WORLD_WIDTH, GARDEN_LOCATION,
    PLAYER_LABEL, PLAYER_NAME_TAG, PLAYER_CALL_COOLDOWN, PLAYER_CALL_CHECK_SECONDS, TRAVEL_STEP_TIMEOUT,
    PLAYER_APPROACH_TIMEOUT, CONVERSATION_REPLY_PENDING_WHEN_SAVED, MAX_RESIDENTS)
from creations import Creation
from apartments import (Invitation, HostedVisit,
                        validate_saved_visit, validate_saved_invitations, validate_saved_home)
from locations import Journey, RouteStep, clamp_point, validate_saved_locations, validate_destination, validate_feet
from player_records import (PlayerOpinion, PlayerCall, PlayerReplyCall, PlayerReport,
                            PlayerPromiseOffer, PlayerAcknowledgment, validate_player_reports)
from conversation_history import (RecordedTurn, ConversationRecord, Memory, WorldEvent,
                                  validate_history_turns, validate_history_events)
from dialogue import ConversationTurn, EmotionReaction, EscalationChoice
from activity import ActivitySnapshot
from simulation import PendingAction, Simulation, validate_saved_actions
from player import (Player, PlayerInteraction, validate_saved_calls, validate_saved_interaction,
                    validate_saved_player_mode)
from player_story import validate_starting_history, resident_arrival_days
from social_records import (Belief, Claim, ConversationConcern, Disclosure, Fact, Goal, OpinionEffect,
    Promise, Proposal, ClaimOption, InformationContext, phase_index)
from social_saves import validate_social_records, validate_saved_promises


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
    cause: Literal["killed by resident"]
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
            from dialogue import ActionIntent
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
        from activity import Behavior
        if self.behavior not in (Behavior.WANDER, Behavior.WAIT) or self.target_id is not None:
            raise ValueError("The snake can wander or join a conversation, without human tasks.")
        ActivitySnapshot.validate(self)
        validate_feet(GARDEN_LOCATION, (self.x, self.y))
        if self.destination is not None:
            validate_destination(self.destination, location=GARDEN_LOCATION)
        return self


PendingActionState = saved_record(PendingAction)


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
        ctx = SaveContext(state=self, residents=residents, live=live, live_ids=live_ids,
                          history=history, events=events, facts=facts, claims=claims, disclosures=disclosures,
                          owner_ids=owner_ids, physical=physical)
        ctx.creation_ids = validate_saved_home(ctx)
        actors = {n.id: (n.display_name or n.character["Name"]) for n in residents}
        actors[self.snake.id] = SNAKE_NAME
        ids = set(actors)
        for npc in [*residents, self.snake]:
            if any(target not in ids or target == npc.id for target in npc.opinions):
                raise ValueError("Runtime opinions require distinct resident or snake actor IDs.")
        ctx.ids, ctx.now = ids, phase_index(self.day, self.phase)
        now = ctx.now
        if len(ids) != len(residents) + 1 or self.tree.id in ids or len({v.casefold() for v in actors.values()}) != len(ids):
            raise ValueError("Actor IDs and stable display names must be unique.")
        if (self.player.id in ids or self.tree.id == self.player.id
                or {PLAYER_LABEL.casefold(), PLAYER_NAME_TAG.casefold()} & {name.casefold() for name in actors.values()}):
            raise ValueError("Player identity conflicts with a physical actor.")
        actors[self.player.id] = PLAYER_LABEL
        ids.add(self.player.id)
        if physical and physical.pending_request and physical.pending_request.get("kind") == "player_command":
            from dialogue import ActionIntent
            command = ActionIntent(**physical.pending_request["context"]["command"])
            if command.action in ("TALK_TO_NPC", "GIVE_ITEM"):
                if (command.target not in owner_ids - {physical.npc_id}
                        or command.action == "TALK_TO_NPC" and command.target == self.player.id):
                    raise ValueError("A saved command must retain a known eligible recipient identity.")
            elif command.target:
                raise ValueError("Unexpected saved command recipient.")
            if command.action == "GIVE_ITEM":
                if command.quantity < 1 or command.item != "fruit" and (command.item not in ctx.creation_ids or command.quantity != 1):
                    raise ValueError("A saved gift request needs its actual quantity and possession identity.")
            elif command.item or command.quantity:
                raise ValueError("Unexpected saved command possession.")
        activities = [*residents, *[state for c in self.conversations for state in c.saved_states],
                      *[state for h in self.handoffs for state in h.saved_states.values()]]
        for owner in [physical, self.visit]:
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
        validate_transfer_history(ctx)
        validate_social_records(ctx)
        validate_saved_promises(ctx)
        validate_saved_handoffs(ctx)
        validate_saved_conversations(ctx)
        self.validate_killings(now, facts, history)
        validate_saved_actions(ctx)
        pending, processed = self.pending_arrival_days, self.processed_arrival_days
        arrival_days = resident_arrival_days(self)
        if (pending != sorted(set(pending)) or processed != sorted(set(processed)) or set(pending) & set(processed)
                or not (set(pending) | set(processed)) <= arrival_days):
            raise ValueError("Pending and completed arrivals must be unique and belong to an eligible dawn. A full building skips that dawn's arrival.")
        arrivals = [n.arrival_day for n in residents if n.arrival_day >= 2]
        if sorted(arrivals) != processed:
            raise ValueError("Completed arrivals must match their preserved residents.")
        if not set(self.introductions) <= live_ids:
            raise ValueError("Invalid introduction identity.")
        validate_saved_calls(ctx)
        return self


    def validate_killings(self, now, facts, history):
        from constants import KILLING_COOLDOWN_PHASES, PROTECTED_POPULATION
        deaths = {n.departure.event_id: n for n in self.archived}
        killings = {f.source_id: f for f in facts.values() if f.kind == "killing"}
        if set(deaths) != set(killings):
            raise ValueError("Archived deaths must have exactly one retained committed killing fact.")
        by_id = {n.id: n for n in [*self.npcs, *self.archived]}
        phases, by_killer = set(), {}
        for event_id, victim in deaths.items():
            fact = killings[event_id]
            departure = victim.departure
            record = history.get(departure.conversation_id)
            if (fact.subject_id != departure.attacker_id or fact.object_id != victim.id
                    or departure.attacker_id not in by_id or fact.subject_id == victim.id
                    or fact.phase != phase_index(departure.day, departure.phase)
                    or fact.conversation_id != departure.conversation_id or not record
                    or record.purpose != "confrontation" or record.participant_ids != [fact.subject_id, victim.id]
                    or fact.phase in phases or fact.location != victim.location):
                raise ValueError("A death must match its killer, physical event and source confrontation.")
            attacker_death = by_id[fact.subject_id].departure
            if attacker_death and phase_index(attacker_death.day, attacker_death.phase) <= fact.phase:
                raise ValueError("An archived resident cannot kill after their own death.")
            phases.add(fact.phase)
            by_killer.setdefault(fact.subject_id, []).append(fact.phase)
        for actor in by_id.values():
            commits = sorted(by_killer.get(actor.id, []))
            if any(b - a < KILLING_COOLDOWN_PHASES for a, b in zip(commits, commits[1:])):
                raise ValueError("A killer cooldown must match committed deaths.")
        if deaths and len(self.npcs) + int(self.mode == "player" and self.player.alive) < PROTECTED_POPULATION:
            raise ValueError("A killing cannot remove the protected population.")
        for meeting in self.conversations:
            grievance_id = history[meeting.record].grievance_id
            if grievance_id != (meeting.focus_claim_id if meeting.purpose == "confrontation" else None):
                raise ValueError("A confrontation must retain its sourced grievance.")
            if meeting.escalation:
                decision = meeting.escalation
                if (meeting.purpose != "confrontation" or meeting.dialogue_state != "ready"
                        or not meeting.social_effects_applied or meeting.initiator not in {n.id for n in self.npcs}
                        or meeting.recipient not in {n.id for n in self.npcs}
                        or decision.target_id != meeting.recipient or decision.grievance_id != grievance_id
                        or meeting.outcome_event_id in deaths
                        or set(meeting.accepted_revisions) != ({meeting.initiator, meeting.recipient} if decision.action == "attack" else set())
                        or not any(t.speaker_id == meeting.recipient and decision.evidence in t.text
                                   for t in history[meeting.record].turns) and decision.action == "attack"):
                    raise ValueError("A pending attack must belong to an accepted, uncommitted confrontation.")
                if decision.action == "attack":
                    from constants import CONFRONTATION_OPINION
                    context = meeting.information_context
                    claim = next(c for c in self.claims if c.id == decision.grievance_id)
                    belief = next((b for b in context.beliefs["initiator"] if b.claim_id == claim.id), None) if context else None
                    retained = facts[claim.fact_id]
                    if (not belief or belief.stance != "believed" or not claim.affirmed
                            or retained.subject_id != meeting.recipient
                            or context.opinions["initiator"].get(meeting.recipient, 0) > CONFRONTATION_OPINION
                            or by_id[meeting.initiator].opinions.get(meeting.recipient, {}).get("opinion", 0) > CONFRONTATION_OPINION
                            or retained.kind == "killing" and belief.phase >= now):
                        raise ValueError("An accepted attack needs its retained, sourced grievance and final hostility.")
        for fact in facts.values():
            if (fact.kind == "harvest") != (fact.quantity is not None):
                raise ValueError("Only harvesting records carry exact quantities.")
            if fact.kind == "harvest":
                collector = next((a for a in [*self.npcs, *self.archived, self.player] if a.id == fact.subject_id), None)
                if collector is None or fact.location != "garden":
                    raise ValueError("A harvest requires a human collector in the garden.")
                from information import harvest_summary
                name = (Player.name.fget(collector) if collector is self.player
                        else collector.display_name or collector.character["Name"])
                if fact.summary != harvest_summary(name, fact.quantity):
                    raise ValueError("A harvest summary must match its actual collector and quantity.")


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
    by_id = {**live, state.player.id: state.player, state.snake.id: state.snake}
    validate_saved_player_mode(state.mode, state.player, state.player_interaction, state.visit,
                               state.npcs)
    validate_saved_locations([state.player, *residents], layout, state.player, state.visit)
    interaction = _parse_player_interaction(state.player_interaction)
    exclusive = validate_saved_interaction(interaction, state.player, by_id, records, layout)
    entries = []
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
    for actor in (*world.npcs, world.snake):
        if actor.destination is not None:
            # A destination outside the walkable map must never block saving; the actor resumes at the nearest valid point.
            actor.destination = clamp_point(actor.location, actor.destination)
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
    for npc in [*world.npcs, world.snake]:
        if npc is not world.snake:
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
    return SimulationState(
        day=world.day, npc_trait_mode=world.npc_trait_mode, phase=world.phase,
        mode=world.mode, player_interaction=world.player_interaction.capture() if world.player_interaction else None,
        player_invitations=world.player_invitations, player_claims=world.player_claims,
        visit=world.visit,
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
        npcs=npcs, history=history, conversations=conversations, handoffs=handoffs,
        tree=_capture_fields(TreeState, world.tree),
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
        values = restore_record(ResidentValues, saved,
            emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            journey=_restore_journey(saved.journey), preferences=saved.preferences.model_dump(),
            departure=saved.departure.model_dump() if saved.departure else None,
            memories=[restore_record(Memory, memory) for memory in saved.memories],
            goal=restore_record(Goal, saved.goal) if saved.goal else None,
            pending_social_opinions=[restore_record(OpinionEffect, effect) for effect in saved.pending_social_opinions],
            beliefs=[restore_record(Belief, belief) for belief in saved.beliefs],
            current_concern=restore_record(ConversationConcern, saved.current_concern) if saved.current_concern else None,
            player_opinion=restore_record(PlayerOpinion, saved.player_opinion),
            player_call=(restore_record(PlayerReplyCall if saved.player_call.kind == "reply" else PlayerCall,
                                        saved.player_call) if saved.player_call else None))
        npc = NPC(**vars(values))
        all_residents.append(npc)
    npcs = all_residents[:len(state.npcs)]
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit)
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
    world.player_call_check = state.player_call_check
    world.player_acknowledgments = [restore_record(PlayerAcknowledgment, a) for a in state.player_acknowledgments]
    world.creations = [work.model_copy(deep=True) for work in state.creations]
    for npc in world.archived:
        npc.world = world
    world.snake = Snake(**vars(restore_record(SnakeValues, state.snake,
        beliefs=[restore_record(Belief, b) for b in state.snake.beliefs],
        memories=[restore_record(Memory, m) for m in state.snake.memories],
        pending_social_opinions=[restore_record(OpinionEffect, e) for e in state.snake.pending_social_opinions],
        current_concern=restore_record(ConversationConcern, state.snake.current_concern) if state.snake.current_concern else None,
        player_opinion=restore_record(PlayerOpinion, state.snake.player_opinion))))
    world.snake.world = world
    by_id[world.snake.id] = world.snake
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
    if world.player_interaction is not None:
        world.player_interaction = world.player_interaction
    from player import call_valid, clear_call
    for npc in npcs:
        if npc.player_call and not call_valid(world, npc, npc.player_call):
            clear_call(npc)
    return world
