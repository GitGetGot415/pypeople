"""Versioned, self-contained simulation saves. No character file paths are loaded."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from character_data import normalize_character, normalize_opinions
from conversation_history import ConversationRecord
from llm import ConversationTurn, EmotionReaction
from npc import Behavior, Conversation, NPC, Tree, WORLD_HEIGHT, WORLD_WIDTH

SAVE_DIRECTORY = Path(__file__).resolve().parent / "saves"


class SaveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ActivityState(SaveModel):
    behavior: Behavior
    destination: tuple[float, float] | None
    pause_remaining: float = Field(ge=0)
    facing: int = Field(ge=0, le=7)


class NPCState(ActivityState):
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    reply: str
    emotion: EmotionReaction

    @field_validator("character")
    @classmethod
    def validate_character(cls, value):
        return normalize_character(value)


class HistoryState(SaveModel):
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[tuple[str, str]]
    outcomes: list[str]
    error: str


class ConversationState(SaveModel):
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    topic: str
    arrived: bool
    turns: list[ConversationTurn] = Field(max_length=8)
    saved_states: tuple[ActivityState, ActivityState]
    shake_elapsed: tuple[int, float] | None
    pending_opinions: list[tuple[int, dict]]

    @field_validator("pending_opinions")
    @classmethod
    def validate_opinions(cls, value):
        return [(index, normalize_opinions(opinions)) for index, opinions in value]


class TreeState(SaveModel):
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    shake_elapsed: float | None = Field(default=None, ge=0, le=1)
    shaker: int | None = Field(default=None, ge=0)


class SimulationState(SaveModel):
    npcs: list[NPCState] = Field(min_length=1)
    tree: TreeState
    history: list[HistoryState]
    conversations: list[ConversationState]
    camera_offset: tuple[float, float]

    @model_validator(mode="after")
    def validate_references(self):
        count = len(self.npcs)
        names = [(npc.display_name or npc.character["Name"]).casefold() for npc in self.npcs]
        if len(set(names)) != count:
            raise ValueError("NPC display names must be unique.")
        if (self.tree.shaker is None) != (self.tree.shake_elapsed is None):
            raise ValueError("A shaking tree needs a shaker and elapsed time.")
        if self.tree.shaker is not None and self.tree.shaker >= count:
            raise ValueError("Invalid tree shaker.")
        reserved, records = set(), set()
        for conversation in self.conversations:
            pair = (conversation.initiator, conversation.recipient)
            if pair[0] == pair[1] or any(index >= count or index in reserved for index in pair):
                raise ValueError("Invalid conversation participants.")
            if conversation.record >= len(self.history) or conversation.record in records:
                raise ValueError("Invalid conversation history reference.")
            if not conversation.arrived and self.npcs[pair[0]].destination is None:
                raise ValueError("A meeting in progress needs a destination.")
            if any(index not in pair for index, _ in conversation.pending_opinions):
                raise ValueError("Invalid pending opinion participant.")
            if conversation.shake_elapsed is not None:
                index, elapsed = conversation.shake_elapsed
                if index not in pair or not 0 <= elapsed <= 1:
                    raise ValueError("Invalid paused tree shake.")
            reserved.update(pair)
            records.add(conversation.record)
            for activity in conversation.saved_states:
                if activity.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE) and activity.destination is None:
                    raise ValueError("An interrupted walking activity needs a destination.")
        for index, npc in enumerate(self.npcs):
            if index not in reserved and npc.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE) and npc.destination is None:
                raise ValueError("A walking NPC needs a destination.")
        return self


class SimulationSave(SaveModel):
    version: Literal[1] = 1
    # Retained for compatibility; the filename is always the displayed name.
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
    """Capture only main-thread world state; running model requests are not persisted."""
    npc_ids = {id(npc): index for index, npc in enumerate(app.npcs)}
    records = {id(record): index for index, record in enumerate(app.conversation_history)}
    history = [asdict(record) for record in app.conversation_history]
    if app.request_history is not None and id(app.request_history) in records:
        record = history[records[id(app.request_history)]]
        record["status"] = "Reply pending when saved"
        record["outcomes"] = ["The reply was not saved. Send the request again after loading."]
    npcs, conversations = [], []
    for npc in app.npcs:
        npcs.append(NPCState(
            character=npc.character, display_name=npc.display_name, x=npc.x, y=npc.y,
            behavior=npc.behavior, destination=npc.destination, pause_remaining=npc.pause_remaining,
            facing=npc.facing, reply=npc.reply,
            emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
        ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(ConversationState(
            initiator=npc_ids[id(npc)], recipient=npc_ids[id(conversation.recipient)],
            record=records[id(conversation.record)], topic=conversation.topic,
            arrived=conversation.arrived, turns=conversation.turns,
            saved_states=[ActivityState(behavior=behavior, destination=destination,
                                        pause_remaining=pause, facing=facing)
                          for behavior, destination, pause, facing in conversation.saved_states],
            shake_elapsed=(npc_ids[id(conversation.shake_elapsed[0])], conversation.shake_elapsed[1])
                          if conversation.shake_elapsed is not None else None,
            pending_opinions=[(npc_ids[id(person)], opinions) for person, opinions in conversation.pending_opinions],
        ))
    return SimulationState(
        npcs=npcs, history=history, conversations=conversations,
        tree=TreeState(x=app.tree.x, y=app.tree.y, shake_elapsed=app.tree.shake_elapsed,
                       shaker=npc_ids[id(app.tree.shaker)] if app.tree.shaker is not None else None),
        camera_offset=tuple(app.game.camera_offset),
    )


def restore_simulation(state):
    """Build a complete world before replacing the current simulation."""
    npcs = [NPC(
        character=saved.character, display_name=saved.display_name, x=saved.x, y=saved.y,
        behavior=saved.behavior, destination=saved.destination, pause_remaining=saved.pause_remaining,
        facing=saved.facing, reply=saved.reply,
        emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
    ) for saved in state.npcs]
    tree = Tree(x=state.tree.x, y=state.tree.y, shake_elapsed=state.tree.shake_elapsed,
                shaker=npcs[state.tree.shaker] if state.tree.shaker is not None else None)
    history = [ConversationRecord(**saved.model_dump()) for saved in state.history]
    for saved in state.conversations:
        # Loading bypasses __init__: it would move NPCs and interrupt their saved activities again.
        conversation = Conversation.__new__(Conversation)
        conversation.initiator, conversation.recipient = npcs[saved.initiator], npcs[saved.recipient]
        conversation.topic, conversation.arrived, conversation.active = saved.topic, saved.arrived, True
        conversation.record = history[saved.record]
        conversation.turns = saved.turns
        conversation.pending_opinions = [(npcs[index], opinions) for index, opinions in saved.pending_opinions]
        conversation.saved_states = [(activity.behavior, activity.destination, activity.pause_remaining, activity.facing)
                                     for activity in saved.saved_states]
        conversation.shake_elapsed = (npcs[saved.shake_elapsed[0]], saved.shake_elapsed[1]) if saved.shake_elapsed else None
        for npc in (conversation.initiator, conversation.recipient):
            npc.conversation = conversation
    return npcs, tree, history
