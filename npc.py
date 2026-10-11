from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import json
from pathlib import Path

from character_data import DIRECTIONS, normalize_opinions
from conversation_history import ConversationRecord, describe_opinion_change

WORLD_WIDTH = 960
WORLD_HEIGHT = 460
NPC_RADIUS = 18
SPEED = 90.0
TREE_INTERACTION_POINT = (660.0, 280.0)
EMOTION_THRESHOLD = 5


class Behavior(str, Enum):
    WANDER = "WANDER"
    STOP = "STOP"
    GO_TO_TREE = "GO_TO_TREE"
    SHAKE_TREE = "SHAKE_TREE"
    SHAKING = "SHAKING"


@dataclass
class Tree:
    x: float = 720.0
    y: float = 230.0
    shake_elapsed: float | None = None
    shaker: "NPC | None" = field(default=None, repr=False, compare=False)

    @property
    def shaking(self):
        return self.shake_elapsed is not None

    @property
    def offset(self):
        if self.shake_elapsed is None:
            return 0
        return round(5 * math.sin(self.shake_elapsed * math.tau * 8))

    def start_shake(self, npc):
        self.shake_elapsed = 0.0
        self.shaker = npc

    def cancel_shake(self):
        self.shake_elapsed = None
        self.shaker = None

    def update(self, dt):
        if self.shake_elapsed is not None:
            self.shake_elapsed += dt
            if self.shake_elapsed >= 1.0 - 1e-9:
                self.cancel_shake()


@dataclass
class NPC:
    character: dict
    x: float = 240.0
    y: float = 230.0
    behavior: Behavior = Behavior.WANDER
    destination: tuple[float, float] | None = None
    pause_remaining: float = 0.0
    rng: random.Random = field(default_factory=random.Random, repr=False)
    facing: int = 4
    sprite_size: tuple[int, int] | None = None
    selected: bool = False
    reply: str = ""
    conversation: "Conversation | None" = field(default=None, repr=False, compare=False)
    character_path: Path | None = field(default=None, repr=False, compare=False)
    display_name: str | None = None
    emotion: str = "neutral"
    emotion_intensity: int = 0

    @property
    def portrait_emotion(self):
        if self.emotion_intensity >= EMOTION_THRESHOLD:
            return self.emotion if self.emotion != "neutral" else None
        return None

    def set_emotion(self, name, intensity):
        """Keep the reaction with its reply until new dialogue replaces it."""
        self.emotion = name
        self.emotion_intensity = intensity

    @property
    def name(self):
        return self.display_name if self.display_name is not None else self.character["Name"].strip()

    @property
    def movement_bounds(self):
        width, height = self.sprite_size or (NPC_RADIUS * 2, NPC_RADIUS)
        horizontal_margin = max(NPC_RADIUS, width / 2)
        return (
            horizontal_margin, WORLD_WIDTH - horizontal_margin,
            max(NPC_RADIUS, height), WORLD_HEIGHT - NPC_RADIUS,
        )

    @property
    def wandering_paused(self):
        return self.selected and self.behavior == Behavior.WANDER

    @property
    def status(self):
        if self.conversation is not None:
            if self.conversation.arrived:
                return f"Talking with {self.conversation.other(self).name}"
            if self.conversation.initiator is self:
                return f"Walking to {self.conversation.recipient.name}"
            return f"Waiting for {self.conversation.initiator.name}"
        if self.wandering_paused:
            return "Talking"
        return {
            Behavior.WANDER: "Wandering",
            Behavior.STOP: "Stopped",
            Behavior.GO_TO_TREE: "Walking to tree",
            Behavior.SHAKE_TREE: "Walking to tree",
            Behavior.SHAKING: "Shaking tree",
        }[self.behavior]

    def contains(self, point):
        if self.sprite_size is not None:
            width, height = self.sprite_size
            return (
                self.x - width / 2 <= point[0] < self.x + width / 2
                and self.y - height <= point[1] < self.y
            )
        return math.hypot(point[0] - self.x, point[1] - self.y) <= NPC_RADIUS

    def snapshot(self, tree):
        return {
            "npc_name": self.name,
            "character": {**self.character, "Name": self.name, "Opinions": self.opinions},
            "facing": DIRECTIONS[self.facing],
            "selected": self.selected,
            "position": [round(self.x, 1), round(self.y, 1)],
            "behavior": self.behavior.value,
            "only_interactable_object": "tree",
            "tree_interaction_point": list(TREE_INTERACTION_POINT),
            "tree_is_shaking": tree.shaking,
            "last_reply": self.reply,
        }

    @property
    def opinions(self):
        # Copy nested entries so worker snapshots cannot modify live opinions.
        return {person: entry.copy() for person, entry in self.character.get("Opinions", {}).items()}

    def remember_opinions(self, updates):
        opinions = normalize_opinions({**self.opinions, **updates})
        if opinions == self.opinions:
            return
        if self.character_path is not None:
            # Preserve profile edits made on disk while the game is running.
            data = json.loads(self.character_path.read_text(encoding="utf-8"))
            data["Opinions"] = opinions
            temporary = self.character_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(self.character_path)
        self.character["Opinions"] = opinions

    def _face_when_still(self):
        at_tree = math.hypot(self.x - TREE_INTERACTION_POINT[0], self.y - TREE_INTERACTION_POINT[1]) <= 1e-6
        self.facing = 2 if at_tree or self.behavior == Behavior.SHAKING else 4

    def set_selected(self, selected):
        """Pause wandering while explicit tasks keep running during dialogue."""
        self.selected = selected
        if self.conversation is None and selected and (self.wandering_paused or self.behavior in (Behavior.STOP, Behavior.SHAKING)):
            self._face_when_still()

    def apply_action(self, action, tree):
        # Reject before changing anything. SAY preserves even an active shake.
        if action == "SAY":
            return
        if action not in ("WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE"):
            raise ValueError("Unsupported NPC action")
        if tree.shaker is self:
            tree.cancel_shake()
        self.pause_remaining = 0.0
        if action == "WANDER":
            self.behavior = Behavior.WANDER
            self.destination = None
        elif action == "STOP":
            self.behavior = Behavior.STOP
            self.destination = None
            self._face_when_still()
        elif action == "GO_TO_TREE":
            self.behavior = Behavior.GO_TO_TREE
            self.destination = TREE_INTERACTION_POINT
        elif action == "SHAKE_TREE":
            self.behavior = Behavior.SHAKE_TREE
            self.destination = TREE_INTERACTION_POINT

    def _move(self, dt):
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        distance = math.hypot(dx, dy)
        step = SPEED * dt
        if distance <= step:
            self.x, self.y = self.destination
            return True
        if distance:
            self.x += dx / distance * step
            self.y += dy / distance * step
        return False

    def update(self, dt, tree):
        if self.conversation is not None:
            return
        old_x, old_y = self.x, self.y
        left, right, top, bottom = self.movement_bounds
        dt = max(0.0, dt)
        if self.behavior == Behavior.SHAKING:
            if not tree.shaking or tree.shaker is not self:
                self.behavior = Behavior.STOP
        elif self.behavior == Behavior.WANDER and not self.wandering_paused:
            if self.pause_remaining > 0:
                remaining = self.pause_remaining
                self.pause_remaining = max(0.0, remaining - dt)
                dt = max(0.0, dt - remaining)
            if dt > 0:
                if self.destination is None:
                    self.destination = (
                        self.rng.uniform(left, right),
                        self.rng.uniform(top, bottom),
                    )
                if self._move(dt):
                    self.destination = None
                    self.pause_remaining = 0.5
        elif self.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE):
            if self._move(dt):
                self.destination = None
                if self.behavior == Behavior.SHAKE_TREE:
                    tree.start_shake(self)
                    self.behavior = Behavior.SHAKING
                else:
                    self.behavior = Behavior.STOP
        self.x = min(right, max(left, self.x))
        self.y = min(bottom, max(top, self.y))
        dx, dy = self.x - old_x, self.y - old_y
        if dx or dy:
            # Compass order: north=0, east=2, south=4, west=6. Screen y grows
            # downward, so atan2(dx, -dy) measures clockwise from north.
            angle = math.atan2(dx, -dy)
            self.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
        if (
            self.wandering_paused
            or self.behavior in (Behavior.STOP, Behavior.SHAKING)
            or (self.behavior == Behavior.WANDER and (self.destination is None or self.pause_remaining > 0))
        ):
            self._face_when_still()


def assign_npc_names(npcs):
    """Give duplicate names distinct labels without editing character profiles."""
    names = [npc.character["Name"].strip() for npc in npcs]
    counts = Counter(name.casefold() for name in names)
    # Reserve original names so a generated label cannot hide another NPC
    # whose profile already has a name such as "Alex (1)".
    used_names = set(counts)
    numbers = Counter()
    for npc, name in zip(npcs, names):
        key = name.casefold()
        npc.display_name = None
        if counts[key] < 2:
            continue
        while True:
            numbers[key] += 1
            label = f"{name} ({numbers[key]})"
            if label.casefold() not in used_names:
                break
        npc.display_name = label
        used_names.add(label.casefold())


class Conversation:
    """Temporarily reserve two NPCs, then restore their interrupted activities."""

    def __init__(self, initiator, recipient, topic, tree, spacing=0):
        if initiator is recipient or initiator.conversation or recipient.conversation:
            raise ValueError("Choose another NPC who is free to talk.")
        self.initiator = initiator
        self.recipient = recipient
        self.topic = topic
        self.arrived = False
        self.active = True
        self.record = ConversationRecord((initiator.name, recipient.name), topic, "Walking to meet")
        self.turns = []
        self.pending_opinions = []
        self.error = ""
        self.saved_states = []
        self.shake_elapsed = None
        # Stand side by side with a visible gap, using the side with room.
        gap = (initiator.sprite_size or (NPC_RADIUS * 2, NPC_RADIUS))[0] / 2
        gap += (recipient.sprite_size or (NPC_RADIUS * 2, NPC_RADIUS))[0] / 2 + 12
        gap = max(gap, spacing)
        left, right, top, bottom = initiator.movement_bounds
        candidates = [
            (recipient.x - gap, recipient.y),
            (recipient.x + gap, recipient.y),
        ]
        candidates = [point for point in candidates if left <= point[0] <= right and top <= point[1] <= bottom]
        if not candidates:
            # Unusually wide sprites can require separation along the y axis.
            height = max((npc.sprite_size or (NPC_RADIUS * 2, NPC_RADIUS))[1] for npc in (initiator, recipient)) + 12
            candidates = [(recipient.x, recipient.y - height), (recipient.x, recipient.y + height)]
            candidates = [point for point in candidates if left <= point[0] <= right and top <= point[1] <= bottom]
        if not candidates:
            raise ValueError("There isn't enough room here for a conversation.")
        for npc in (initiator, recipient):
            self.saved_states.append((npc.behavior, npc.destination, npc.pause_remaining, npc.facing))
            if tree.shaker is npc:
                self.shake_elapsed = (npc, tree.shake_elapsed)
                tree.cancel_shake()
            npc.conversation = self
        initiator.destination = min(candidates, key=lambda point: math.hypot(point[0] - initiator.x, point[1] - initiator.y))

    def other(self, npc):
        return self.recipient if npc is self.initiator else self.initiator

    @property
    def error(self):
        return self.record.error

    @error.setter
    def error(self, message):
        self.record.error = message

    def set_response(self, response):
        self.turns = response.turns
        self.record.turns = [
            (self.initiator.name if turn.speaker == "initiator" else self.recipient.name, turn.text)
            for turn in response.turns
        ]
        self.record.outcomes = [
            describe_opinion_change(
                npc.name, partner.name, npc.opinions.get(partner.name, {}).get("opinion"), entry.opinion, entry.reason
            )
            for npc, partner, entry in (
                (self.initiator, self.recipient, response.initiator_opinion),
                (self.recipient, self.initiator, response.recipient_opinion),
            )
        ]
        self.record.status = "Dialogue ready"
        self.pending_opinions = [
            (npc, {partner.name: {"opinion": entry.opinion, "reason": entry.reason}})
            for npc, partner, entry in (
                (self.initiator, self.recipient, response.initiator_opinion),
                (self.recipient, self.initiator, response.recipient_opinion),
            )
        ]
        self.remember_opinions()

    def remember_opinions(self):
        """Retry unsaved opinions without regenerating or hiding the dialogue."""
        while self.pending_opinions:
            npc, updates = self.pending_opinions[0]
            try:
                npc.remember_opinions(updates)
            except (OSError, ValueError):
                self.error = f"Could not save {npc.name}'s opinion. Click Retry to save it again."
                return
            self.pending_opinions.pop(0)
        self.error = ""

    def update(self, dt):
        if not self.active or self.arrived:
            return
        npc = self.initiator
        old_x, old_y = npc.x, npc.y
        self.arrived = npc._move(max(0.0, dt))
        if self.arrived:
            self.record.status = "Waiting for dialogue"
            npc.destination = None
            for participant in (self.initiator, self.recipient):
                other = self.other(participant)
                angle = math.atan2(other.x - participant.x, participant.y - other.y)
                participant.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
        elif npc.x != old_x or npc.y != old_y:
            angle = math.atan2(npc.x - old_x, old_y - npc.y)
            npc.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)

    def snapshot(self, tree, npcs):
        return {
            "request_kind": "conversation",
            "initiator": self.initiator.snapshot(tree),
            "recipient": self.recipient.snapshot(tree),
            "player_request": self.topic,
            "initiator_opinion_of_recipient": self.initiator.opinions.get(self.recipient.name),
            "recipient_opinion_of_initiator": self.recipient.opinions.get(self.initiator.name),
            "nearby_npcs": [npc.snapshot(tree) for npc in npcs if npc is not self.initiator and npc is not self.recipient],
        }

    def finish(self, tree):
        if not self.active:
            return
        self.active = False
        self.record.status = "Finished" if self.turns else "Ended before dialogue was generated"
        if not self.turns:
            self.record.outcomes = ["The conversation ended before anyone spoke."]
        for npc, state in zip((self.initiator, self.recipient), self.saved_states):
            npc.conversation = None
            npc.behavior, npc.destination, npc.pause_remaining, npc.facing = state
        if self.shake_elapsed is not None:
            npc, elapsed = self.shake_elapsed
            at_tree = math.hypot(npc.x - TREE_INTERACTION_POINT[0], npc.y - TREE_INTERACTION_POINT[1]) <= 1e-6
            if not tree.shaking and at_tree:
                tree.start_shake(npc)
                tree.shake_elapsed = elapsed
            else:
                # Return to the tree if the initiator walked away during the chat.
                npc.behavior = Behavior.SHAKE_TREE
                npc.destination = TREE_INTERACTION_POINT
