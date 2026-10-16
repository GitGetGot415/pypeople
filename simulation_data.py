"""Self-contained simulation saves. No character file paths are loaded."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from character_data import normalize_character, normalize_opinions
from conversation_history import ConversationRecord, Memory, WorldEvent
from llm import ConversationTurn, EmotionReaction
from npc import ACTIVITY_FIELDS, Behavior, Conversation, Gift, NPC, Tree, WORLD_HEIGHT, WORLD_WIDTH
from simulation import EAT_SECONDS, PendingAction, SHAKE_SECONDS, Simulation

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
    kind: Literal["gave_fruit", "received_fruit", "refused", "kept_fruit", "conversation"]
    summary: str = Field(max_length=400)
    weight: float = Field(ge=-12, le=12)


class EventState(SaveModel):
    id: str = Field(min_length=1)
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    kind: Literal["phase", "food_requested", "food_refused", "fruit_shared", "conversation_completed", "interaction_failed"]
    participants: list[str] = Field(max_length=2)
    text: str
    history: int | None = Field(ge=0)


class NPCState(ActivityState):
    id: str = Field(min_length=1)
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    reply: str
    emotion: EmotionReaction
    social: float = Field(ge=0, le=100)
    inventory: dict[Literal["fruit"], int]
    inventory_capacity: int = Field(ge=1)
    preferences: PreferencesState
    memories: list[MemoryState] = Field(max_length=20)
    command_revision: int = Field(ge=0)

    @field_validator("character")
    @classmethod
    def validate_character(cls, value):
        return normalize_character(value)

    @model_validator(mode="after")
    def validate_inventory(self):
        if any(value < 0 for value in self.inventory.values()) or sum(self.inventory.values()) > self.inventory_capacity:
            raise ValueError("Invalid inventory.")
        return self


class HistoryState(SaveModel):
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[tuple[str, str]]
    outcomes: list[str]
    error: str
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "food_request"]


class ConversationState(SaveModel):
    id: str = Field(min_length=1)
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "food_request"]
    dialogue_state: Literal["waiting", "pending", "ready", "fallback"]
    wait_elapsed: float = Field(ge=0, le=12.1)
    talking_remaining: float = Field(ge=0, le=4)
    food_outcome: Literal["", "accepted", "need", "refused", "unavailable"]
    outcome_event_id: str | None
    reactions: list[tuple[str, float]] = Field(max_length=2)
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    topic: str
    arrived: bool
    turns: list[ConversationTurn] = Field(max_length=8)
    saved_states: tuple[ActivityState, ActivityState]
    pending_opinions: list[tuple[int, dict]]

    @field_validator("pending_opinions")
    @classmethod
    def validate_opinions(cls, value):
        return [(index, normalize_opinions(opinions)) for index, opinions in value]


class GiftState(SaveModel):
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    item: Literal["fruit"]
    quantity: int = Field(ge=1)
    arrived: bool
    elapsed: float = Field(ge=0, lt=2)
    transferred: bool
    outcome_event_id: str | None
    saved_states: tuple[ActivityState, ActivityState]


class TreeState(SaveModel):
    id: str = Field(min_length=1)
    x: float = Field(ge=60, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT - 50)
    fruit: int = Field(ge=0)
    capacity: int = Field(ge=1)
    shake_elapsed: float | None = Field(ge=0, le=SHAKE_SECONDS)
    shaker: str | None
    reserved_by: str | None


class PendingActionState(SaveModel):
    npc_id: str
    action: Literal["WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "TALK_TO_NPC", "GIVE_ITEM", "EAT"]
    revision: int = Field(ge=0)
    target_id: str | None
    topic: str
    record: int | None = Field(ge=0)
    item: Literal["", "fruit"] = ""
    quantity: int = Field(default=0, ge=0)


class SimulationState(SaveModel):
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    events: list[EventState] = Field(max_length=200)
    social_opportunity: bool
    social_rotation: int = Field(ge=0)
    npcs: list[NPCState] = Field(min_length=1)
    tree: TreeState
    pending_actions: list[PendingActionState]
    history: list[HistoryState]
    conversations: list[ConversationState]
    gifts: list[GiftState] = Field(default_factory=list)
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(default=1.0, ge=0.5, le=2.5)

    @model_validator(mode="after")
    def validate_references(self):
        count = len(self.npcs)
        names = [(npc.display_name or npc.character["Name"]).casefold() for npc in self.npcs]
        ids = [npc.id for npc in self.npcs]
        all_ids = ids + [self.tree.id]
        if len(set(names)) != count or len(set(all_ids)) != len(all_ids):
            raise ValueError("Names and object IDs must be unique.")
        if self.tree.fruit > self.tree.capacity:
            raise ValueError("Invalid fruit resource.")
        if self.social_rotation >= count:
            raise ValueError("Invalid social rotation.")
        event_ids = [event.id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Duplicate event IDs.")
        for event in self.events:
            if (any(participant not in ids for participant in event.participants)
                    or len(set(event.participants)) != len(event.participants)
                    or (event.history is not None and event.history >= len(self.history))):
                raise ValueError("Invalid event references.")
        for npc in self.npcs:
            if len({memory.id for memory in npc.memories}) != len(npc.memories):
                raise ValueError("Duplicate memories.")
            if any(memory.partner_id not in ids or memory.partner_id == npc.id for memory in npc.memories):
                raise ValueError("Invalid memory partner.")
        if (self.tree.shaker is None) != (self.tree.shake_elapsed is None):
            raise ValueError("A shaking tree needs a shaker and elapsed time.")
        if self.tree.reserved_by is not None and self.tree.reserved_by not in ids:
            raise ValueError("Invalid tree reservation.")
        if self.tree.shaker is not None and self.tree.shaker != self.tree.reserved_by:
            raise ValueError("The shaker must own the tree slot.")
        reserved, records = set(), set()
        if len({conversation.id for conversation in self.conversations}) != len(self.conversations):
            raise ValueError("Duplicate interaction IDs.")

        def validate_target(activity, npc):
            if any(target not in all_ids or target == npc.id for target in activity.cooldowns):
                raise ValueError("Invalid cooldown target.")
            target = activity.target_id
            if activity.behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
                if target != self.tree.id:
                    raise ValueError("Invalid tree target.")
                point = (self.tree.x - 60, self.tree.y + 50)
                if activity.destination is not None and activity.destination != point:
                    raise ValueError("Invalid tree destination.")
                if activity.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE) and activity.destination is None:
                    raise ValueError("A walking task needs a destination.")
                if activity.behavior in (Behavior.HARVEST, Behavior.SHAKING) and activity.activity_elapsed > SHAKE_SECONDS:
                    raise ValueError("Invalid shake progress.")
            elif activity.behavior == Behavior.COMPANY:
                if target not in ids or target == npc.id:
                    raise ValueError("Invalid companion.")
            elif target is not None:
                raise ValueError("Unexpected activity target.")

        for conversation in self.conversations:
            pair = (conversation.initiator, conversation.recipient)
            if pair[0] == pair[1] or any(index >= count or index in reserved for index in pair):
                raise ValueError("Invalid conversation participants.")
            if conversation.record >= len(self.history) or conversation.record in records:
                raise ValueError("Invalid conversation history reference.")
            if not conversation.arrived and self.npcs[pair[0]].destination is None:
                raise ValueError("A meeting in progress needs a destination.")
            if self.npcs[pair[1]].destination is not None:
                raise ValueError("The conversation recipient must be waiting.")
            if conversation.dialogue_state == "pending" and not conversation.arrived:
                raise ValueError("Dialogue requests start only after meeting.")
            if any(not reason.strip() or not -4 <= weight <= 4 for reason, weight in conversation.reactions):
                raise ValueError("Invalid conversation reaction.")
            if conversation.arrived and any(self.npcs[index].destination is not None for index in pair):
                raise ValueError("Arrived conversation participants must be still.")
            if (self.history[conversation.record].origin != conversation.origin
                    or self.history[conversation.record].purpose != conversation.purpose):
                raise ValueError("Conversation history purpose does not match.")
            ready = conversation.dialogue_state in ("ready", "fallback")
            if ready != bool(conversation.turns) or (ready and not conversation.arrived):
                raise ValueError("Invalid dialogue progress.")
            if conversation.purpose == "food_request":
                if (conversation.origin != "autonomous" or bool(conversation.food_outcome) != conversation.arrived
                        or conversation.dialogue_state not in ("waiting", "ready")
                        or bool(conversation.outcome_event_id) != (conversation.food_outcome in ("need", "refused", "unavailable"))):
                    raise ValueError("Invalid food request progress.")
            elif conversation.food_outcome or (conversation.dialogue_state == "ready" and len(conversation.reactions) != 2):
                raise ValueError("Invalid conversation outcome.")
            if any(index not in pair for index, _ in conversation.pending_opinions):
                raise ValueError("Invalid pending opinion participant.")
            for index, activity in zip(pair, conversation.saved_states):
                validate_target(activity, self.npcs[index])
            reserved.update(pair)
            records.add(conversation.record)
        for gift in self.gifts:
            pair = (gift.initiator, gift.recipient)
            if pair[0] == pair[1] or any(index >= count or index in reserved for index in pair):
                raise ValueError("Invalid gift participants.")
            if gift.record >= len(self.history) or gift.record in records:
                raise ValueError("Invalid gift history reference.")
            giver, recipient = (self.npcs[index] for index in pair)
            if self.history[gift.record].purpose == "food_request" and gift.quantity != 1:
                raise ValueError("A food request shares exactly one apple.")
            if gift.arrived != (giver.destination is None) or (not gift.arrived and gift.elapsed != 0):
                raise ValueError("Invalid gift progress.")
            if recipient.destination is not None:
                raise ValueError("The gift recipient must be waiting.")
            if gift.transferred != (gift.elapsed >= 1) or gift.transferred != bool(gift.outcome_event_id):
                raise ValueError("Invalid gift transfer progress.")
            for index, activity in zip(pair, gift.saved_states):
                validate_target(activity, self.npcs[index])
            reserved.update(pair)
            records.add(gift.record)
        for index, npc in enumerate(self.npcs):
            if index in reserved:
                if npc.behavior != Behavior.WAIT or npc.owner != "autonomous" or npc.target_id is not None or npc.resume_pending:
                    raise ValueError("Invalid activity during an interaction.")
                if npc.id == self.tree.reserved_by:
                    raise ValueError("Interaction participants cannot reserve the tree.")
                continue
            validate_target(npc, npc)
            if npc.behavior == Behavior.EAT and npc.inventory.get("fruit", 0) < 1:
                raise ValueError("Eating needs an apple in inventory.")
            if npc.resume_pending:
                if npc.id == self.tree.reserved_by:
                    raise ValueError("A suspended activity cannot reserve the tree.")
                continue
            if npc.behavior in (Behavior.HARVEST, Behavior.SHAKING) and self.tree.reserved_by != npc.id:
                raise ValueError("An active harvest or shake must reserve the tree.")
            if npc.behavior == Behavior.HARVEST and (self.tree.fruit == 0 or sum(npc.inventory.values()) >= npc.inventory_capacity):
                raise ValueError("An active harvest needs available fruit and inventory space.")
            if self.tree.reserved_by == npc.id:
                if npc.target_id != self.tree.id or npc.behavior not in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
                    raise ValueError("Invalid tree owner activity.")
            if self.tree.shaker == npc.id:
                if npc.destination is not None or npc.behavior not in (Behavior.HARVEST, Behavior.SHAKING):
                    raise ValueError("Invalid shaker activity.")
        pending = set()
        for action in self.pending_actions:
            if action.npc_id not in ids or action.npc_id in pending:
                raise ValueError("Invalid pending action participant.")
            if action.action in ("TALK_TO_NPC", "GIVE_ITEM"):
                if action.target_id not in ids or action.target_id == action.npc_id:
                    raise ValueError("Invalid pending recipient.")
            elif action.target_id is not None:
                raise ValueError("Unexpected pending action target.")
            if action.action == "GIVE_ITEM":
                if action.item != "fruit" or action.quantity < 1:
                    raise ValueError("Invalid pending gift.")
            elif action.item or action.quantity:
                raise ValueError("Only giving actions may include items.")
            if action.record is not None and action.record >= len(self.history):
                raise ValueError("Invalid pending action history.")
            pending.add(action.npc_id)
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
                entries.append(SaveEntry(path.stem, save.name, save.updated_at, len(save.state.npcs)))
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


def capture_simulation(app):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    npc_indices = {id(npc): index for index, npc in enumerate(app.npcs)}
    records = {id(record): index for index, record in enumerate(app.conversation_history)}
    history = [asdict(record) for record in app.conversation_history]
    if app.request_history is not None and id(app.request_history) in records:
        record = history[records[id(app.request_history)]]
        record["status"] = "Reply pending when saved"
        record["outcomes"] = ["The reply was not saved. Send the request again after loading."]
    npcs, conversations, gifts = [], [], []
    for npc in app.npcs:
        npcs.append(NPCState(
            **npc.activity_state(), id=npc.id,
            character=npc.character, display_name=npc.display_name, x=npc.x, y=npc.y,
            reply=npc.reply, emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
            social=npc.social,
            inventory=npc.inventory.copy(), inventory_capacity=npc.inventory_capacity,
            preferences=npc.preferences.copy(), command_revision=npc.command_revision,
            memories=[asdict(memory) for memory in npc.memories],
        ))
        gift = npc.gift
        if gift is not None and gift.initiator is npc:
            gifts.append(GiftState(
                initiator=npc_indices[id(npc)], recipient=npc_indices[id(gift.recipient)],
                record=records[id(gift.record)], item=gift.item, quantity=gift.quantity,
                arrived=gift.arrived, elapsed=gift.elapsed,
                transferred=gift.transferred, outcome_event_id=gift.outcome_event_id,
                saved_states=[ActivityState(**activity) for activity in gift.saved_states],
            ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(ConversationState(
            id=conversation.id, origin=conversation.origin, purpose=conversation.purpose,
            dialogue_state=conversation.dialogue_state, wait_elapsed=conversation.wait_elapsed,
            talking_remaining=conversation.talking_remaining, food_outcome=conversation.food_outcome,
            outcome_event_id=conversation.outcome_event_id, reactions=conversation.reactions,
            initiator=npc_indices[id(npc)], recipient=npc_indices[id(conversation.recipient)],
            record=records[id(conversation.record)], topic=conversation.topic,
            arrived=conversation.arrived, turns=conversation.turns,
            saved_states=[ActivityState(**activity) for activity in conversation.saved_states],
            pending_opinions=[(npc_indices[id(person)], opinions) for person, opinions in conversation.pending_opinions],
        ))
    world = app.simulation
    return SimulationState(
        day=world.day, phase=world.phase,
        events=[asdict(event) for event in world.events],
        social_opportunity=world.social_opportunity, social_rotation=world.social_rotation,
        npcs=npcs, history=history, conversations=conversations, gifts=gifts,
        tree=TreeState(id=app.tree.id, x=app.tree.x, y=app.tree.y, fruit=app.tree.fruit,
                       capacity=app.tree.capacity,
                       shake_elapsed=app.tree.shake_elapsed,
                       shaker=app.tree.shaker.id if app.tree.shaker is not None else None,
                       reserved_by=app.tree.reserved_by.id if app.tree.reserved_by is not None else None),
        pending_actions=[asdict(action) for action in world.pending_actions],
        camera_offset=tuple(app.game.camera_offset),
        camera_zoom=app.game.camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    state = SimulationState.model_validate(state.model_dump())
    npcs = []
    for saved in state.npcs:
        npc = NPC(
            id=saved.id, character=saved.character, display_name=saved.display_name, x=saved.x, y=saved.y,
            reply=saved.reply, emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            social=saved.social,
            inventory=saved.inventory.copy(), inventory_capacity=saved.inventory_capacity,
            preferences=saved.preferences.model_dump(), command_revision=saved.command_revision,
            memories=[Memory(**memory.model_dump()) for memory in saved.memories],
        )
        npc.restore_activity({name: getattr(saved, name) for name in ACTIVITY_FIELDS})
        npcs.append(npc)
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit,
                capacity=state.tree.capacity,
                shake_elapsed=state.tree.shake_elapsed, shaker=by_id.get(state.tree.shaker),
                reserved_by=by_id.get(state.tree.reserved_by))
    world = Simulation(npcs, tree)
    world.day, world.phase = state.day, state.phase
    world.pending_actions = [PendingAction(**action.model_dump()) for action in state.pending_actions]
    for action in world.pending_actions:
        by_id[action.npc_id].request_pending = True
    history = [ConversationRecord(**saved.model_dump()) for saved in state.history]
    world.history = history
    world.events = [WorldEvent(**event.model_dump()) for event in state.events]
    world.social_opportunity, world.social_rotation = state.social_opportunity, state.social_rotation
    for saved in state.conversations:
        # Loading bypasses __init__: it would interrupt the saved activities again.
        conversation = Conversation.__new__(Conversation)
        for name in ("id", "origin", "purpose", "dialogue_state", "wait_elapsed", "talking_remaining",
                     "food_outcome", "outcome_event_id", "reactions"):
            setattr(conversation, name, getattr(saved, name))
        conversation.initiator, conversation.recipient = npcs[saved.initiator], npcs[saved.recipient]
        conversation.topic, conversation.arrived, conversation.active = saved.topic, saved.arrived, True
        conversation.record = history[saved.record]
        conversation.turns = saved.turns
        conversation.pending_opinions = [(npcs[index], opinions) for index, opinions in saved.pending_opinions]
        conversation.saved_states = [activity.model_dump() for activity in saved.saved_states]
        for npc in (conversation.initiator, conversation.recipient):
            npc.conversation = conversation
        if conversation.dialogue_state == "pending":
            conversation.fallback("The outstanding dialogue request was not saved; used a local greeting.")
    for saved in state.gifts:
        gift = Gift.__new__(Gift)
        gift.initiator, gift.recipient = npcs[saved.initiator], npcs[saved.recipient]
        gift.item, gift.quantity = saved.item, saved.quantity
        gift.arrived, gift.elapsed = saved.arrived, saved.elapsed
        gift.active, gift.transferred, gift.outcome_event_id = True, saved.transferred, saved.outcome_event_id
        gift.record = history[saved.record]
        gift.saved_states = [activity.model_dump() for activity in saved.saved_states]
        for npc in (gift.initiator, gift.recipient):
            npc.gift = gift
    # Claims are reacquired through the same safe path as interrupted activities.
    tree.reserved_by = None
    tree.cancel_shake()
    for npc in npcs:
        if npc.conversation is None and npc.gift is None and npc.target_id == tree.id:
            npc.resume_pending = True
    return world, history
