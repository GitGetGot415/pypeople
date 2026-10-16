from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import json
from pathlib import Path
from uuid import uuid4

from character_data import DIRECTIONS, normalize_opinions
from conversation_history import ConversationRecord, describe_opinion_change, remember, unsupported_location
from simulation import (
    APPLE_SATISFACTION, COMPANY_MAX_SECONDS, DECISION_SECONDS, EAT_SECONDS, FAILURE_COOLDOWN, INVENTORY_CAPACITY,
    MOVEMENT_SPEED, PARTNER_COOLDOWN, SHAKE_SECONDS, SOCIAL_RADIUS,
    SOCIAL_RECOVERED, WANDER_PAUSE,
)

WORLD_WIDTH = 960
WORLD_HEIGHT = 460
NPC_RADIUS = 18
EMOTION_THRESHOLD = 5

ACTIVITY_FIELDS = (
    "behavior", "destination", "pause_remaining", "facing", "owner", "target_id",
    "activity_elapsed", "reason", "decision_remaining", "cooldowns", "resume_pending",
)


class Behavior(str, Enum):
    WANDER = "WANDER"
    STOP = "STOP"
    GO_TO_TREE = "GO_TO_TREE"
    SHAKE_TREE = "SHAKE_TREE"
    SHAKING = "SHAKING"
    HARVEST = "HARVEST"
    EAT = "EAT"
    COMPANY = "COMPANY"
    WAIT = "WAIT"


@dataclass
class Tree:
    x: float = WORLD_WIDTH / 2
    y: float = WORLD_HEIGHT / 2
    shake_elapsed: float | None = None
    shaker: "NPC | None" = field(default=None, repr=False, compare=False)
    id: str = "fruit-tree"
    fruit: int = 8
    capacity: int = 8
    reserved_by: "NPC | None" = field(default=None, repr=False, compare=False)

    @property
    def interaction_point(self):
        return (self.x - 60.0, self.y + 50.0)

    def reserve(self, npc):
        if self.reserved_by is not None and self.reserved_by is not npc:
            return False
        self.reserved_by = npc
        return True

    def release(self, npc):
        if self.shaker is npc:
            self.cancel_shake()
        if self.reserved_by is npc:
            self.reserved_by = None

    @property
    def shaking(self):
        return self.shake_elapsed is not None

    @property
    def offset(self):
        if self.shake_elapsed is None:
            return 0
        return round(5 * math.sin(self.shake_elapsed * math.tau * 8))

    def start_shake(self, npc):
        if not self.reserve(npc):
            return False
        self.shake_elapsed = 0.0
        self.shaker = npc
        return True

    def cancel_shake(self):
        self.shake_elapsed = None
        self.shaker = None

    def update(self, dt):
        if self.shake_elapsed is not None:
            self.shake_elapsed += dt


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
    gift: "Gift | None" = field(default=None, repr=False, compare=False)
    character_path: Path | None = field(default=None, repr=False, compare=False)
    display_name: str | None = None
    emotion: str = "neutral"
    emotion_intensity: int = 0
    id: str = field(default_factory=lambda: uuid4().hex)
    social: float = field(default_factory=lambda: random.uniform(55, 80))
    inventory: dict = field(default_factory=lambda: {"fruit": 0})
    inventory_capacity: int = INVENTORY_CAPACITY
    preferences: dict = field(default_factory=lambda: {
        "sociability": random.uniform(0.25, 0.85),
        "generosity": random.uniform(0.25, 0.85),
    })
    memories: list = field(default_factory=list)
    owner: str = "autonomous"
    target_id: str | None = None
    activity_elapsed: float = 0.0
    reason: str = "Exploring the world"
    decision_remaining: float = field(default_factory=lambda: random.uniform(0, DECISION_SECONDS))
    cooldowns: dict = field(default_factory=dict)
    resume_pending: bool = False
    command_revision: int = 0
    request_pending: bool = False
    world: object = field(default=None, repr=False, compare=False)

    @property
    def inventory_space(self):
        return self.inventory_capacity - sum(self.inventory.values())

    def can_reach(self, point):
        left, right, top, bottom = self.movement_bounds
        return left <= point[0] <= right and top <= point[1] <= bottom

    def activity_state(self):
        state = {name: getattr(self, name) for name in ACTIVITY_FIELDS}
        state["cooldowns"] = self.cooldowns.copy()
        return state

    def restore_activity(self, state):
        for name in ACTIVITY_FIELDS:
            setattr(self, name, state[name].copy() if name == "cooldowns" else state[name])

    def release_claims(self, world):
        world.tree.release(self)

    def finish_activity(self, world, failed=False):
        target = self.target_id
        if target is not None and (failed or self.behavior == Behavior.COMPANY):
            self.cooldowns[target] = FAILURE_COOLDOWN if failed else PARTNER_COOLDOWN
        self.release_claims(world)
        self.behavior = Behavior.WAIT
        self.destination = None
        self.target_id = None
        self.activity_elapsed = 0.0
        self.resume_pending = False
        self.owner = "autonomous"
        self.decision_remaining = 0.0
        self.reason = "Target unavailable; choosing another activity" if failed else "Finished; choosing what to do next"

    def begin_activity(self, behavior, world, target=None, reason="", owner="autonomous"):
        self.release_claims(world)
        self.behavior, self.target_id = behavior, target
        self.owner, self.reason = owner, reason
        self.destination = None
        self.activity_elapsed = self.pause_remaining = 0.0
        self.resume_pending = False
        self.decision_remaining = DECISION_SECONDS
        if behavior == Behavior.EAT:
            self.facing = 4
        elif behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE):
            if target != world.tree.id or not self.can_reach(world.tree.interaction_point):
                self.finish_activity(world, failed=True)
                return
            if not world.tree.reserve(self) and behavior == Behavior.HARVEST:
                self.finish_activity(world, failed=True)
                return
            self.destination = world.tree.interaction_point
        elif behavior == Behavior.COMPANY:
            from autonomy import company_point
            partner = world.npc_by_id(target)
            self.destination = company_point(self, partner) if partner is not None else None
            if self.destination is None:
                self.finish_activity(world, failed=True)

    def resume_activity(self, world):
        """Reacquire suspended claims on a simulation tick, preserving committed progress."""
        self.resume_pending = False
        if self.behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
            if (self.target_id != world.tree.id or not self.can_reach(world.tree.interaction_point)
                    or not world.tree.reserve(self)):
                self.finish_activity(world, failed=True)
                return
            if self.behavior == Behavior.SHAKING:
                self.behavior = Behavior.SHAKE_TREE
            self.destination = world.tree.interaction_point
        elif self.behavior == Behavior.COMPANY:
            from autonomy import company_point
            partner = world.npc_by_id(self.target_id)
            if partner is None or not world.company_available(partner):
                self.finish_activity(world, failed=True)
            else:
                self.destination = company_point(self, partner)

    def update_cooldowns(self, dt):
        self.decision_remaining = max(0.0, self.decision_remaining - dt)
        self.cooldowns = {key: value - dt for key, value in self.cooldowns.items() if value > dt}
        interaction = self.conversation or self.gift
        if interaction is not None:
            index = 0 if interaction.initiator is self else 1
            suspended = interaction.saved_states[index]
            suspended["decision_remaining"] = max(0.0, suspended["decision_remaining"] - dt)
            suspended["cooldowns"] = {key: value - dt for key, value in suspended["cooldowns"].items() if value > dt}

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
        return (self.selected or self.request_pending) and self.behavior == Behavior.WANDER

    @property
    def decision_reason(self):
        if self.conversation is None and self.owner == "autonomous":
            if self.request_pending:
                return f"{self.reason}. Waiting for the player request."
            if self.selected:
                return f"{self.reason}. New decisions wait while selected."
        return self.reason

    @property
    def status(self):
        if self.gift is not None:
            if self.thanking:
                return f"Thanking {self.gift.initiator.name}"
            if self.gift.initiator is self:
                if self.gift.elapsed >= 1.0:
                    return f"Waiting for {self.gift.recipient.name}'s thanks"
                return f"Giving to {self.gift.recipient.name}" if self.gift.arrived else f"Walking to {self.gift.recipient.name}"
            return f"Receiving from {self.gift.initiator.name}"
        if self.conversation is not None:
            if self.conversation.arrived:
                return f"Talking with {self.conversation.other(self).name}"
            if self.conversation.initiator is self:
                return f"Walking to {self.conversation.recipient.name}"
            return f"Waiting for {self.conversation.initiator.name}"
        if self.owner == "player_hold":
            return "Player hold (STOP)"
        if self.resume_pending:
            return "Resuming activity on next simulation tick"
        if self.wandering_paused:
            return "Waiting for player"
        return {
            Behavior.WANDER: "Wandering",
            Behavior.STOP: "Stopped",
            Behavior.GO_TO_TREE: "Walking to tree",
            Behavior.SHAKE_TREE: "Walking to tree",
            Behavior.SHAKING: "Shaking tree",
            Behavior.HARVEST: "Walking to harvest" if self.destination else "Harvesting fruit",
            Behavior.EAT: "Eating an apple",
            Behavior.COMPANY: "Seeking company" if self.destination else "Spending time together",
            Behavior.WAIT: "Waiting",
        }[self.behavior]

    def contains(self, point):
        if self.sprite_size is not None:
            width, height = self.sprite_size
            return (
                self.x - width / 2 <= point[0] < self.x + width / 2
                and self.y - height <= point[1] < self.y
            )
        return math.hypot(point[0] - self.x, point[1] - self.y) <= NPC_RADIUS

    def snapshot(self, tree, partner_id=None):
        memories = sorted(enumerate(self.memories),
                          key=lambda entry: (entry[1].partner_id == partner_id, entry[0]), reverse=True)[:5]
        return {
            "npc_id": self.id,
            "npc_name": self.name,
            "character": {**self.character, "Name": self.name, "Opinions": self.opinions},
            "facing": DIRECTIONS[self.facing],
            "selected": self.selected,
            "position": [round(self.x, 1), round(self.y, 1)],
            "behavior": self.behavior.value,
            "activity": self.status,
            "activity_reason": self.decision_reason,
            "activity_owner": self.owner,
            "satisfaction": round(self.social, 1),
            "inventory": self.inventory.copy(),
            "inventory_capacity": self.inventory_capacity,
            "inventory_space": self.inventory_space,
            "tree_interaction_point": list(tree.interaction_point),
            "tree_fruit": tree.fruit,
            "tree_occupied": tree.reserved_by is not None,
            "day": self.world.day if self.world else 1,
            "time_of_day": self.world.phase if self.world else "day",
            "tree_is_shaking": tree.shaking,
            "last_reply": self.reply,
            "memories": [{"partner_id": memory.partner_id, "kind": memory.kind,
                          "summary": memory.summary, "day": memory.day, "phase": memory.phase}
                         for _, memory in memories],
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
        point = self.world.tree.interaction_point if self.world else None
        at_tree = point is not None and math.hypot(self.x - point[0], self.y - point[1]) <= 1e-6
        self.facing = 2 if not self.eating and (at_tree or self.behavior == Behavior.SHAKING) else 4

    def set_selected(self, selected):
        """Defer new decisions while allowing the current finite activity to finish."""
        self.selected = selected
        if self.conversation is None and self.gift is None and selected and (self.wandering_paused or self.behavior in (Behavior.STOP, Behavior.SHAKING)):
            self._face_when_still()

    def apply_action(self, action, tree):
        # Reject before changing anything. SAY preserves even an active shake.
        if action == "SAY":
            return
        if action not in ("WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "EAT"):
            raise ValueError("Unsupported NPC action")
        if self.conversation is not None or self.gift is not None or self.eating:
            raise ValueError("This NPC is already in an interaction.")
        if action == "EAT" and self.inventory.get("fruit", 0) < 1:
            raise ValueError("I don't have an apple to eat.")
        self.command_revision += 1
        if action == "WANDER":
            self.begin_activity(Behavior.WANDER, self.world, reason="Player released control; choosing activities")
            self.decision_remaining = 0
        elif action == "STOP":
            self.begin_activity(Behavior.STOP, self.world, reason="Player asked me to stay here; waiting for another task", owner="player_hold")
            self._face_when_still()
        elif action == "EAT":
            self.begin_activity(Behavior.EAT, self.world, reason="Player asked me to eat an apple", owner="player_task")
        else:
            self.begin_activity(Behavior(action), self.world, tree.id, "Player asked me to visit the tree" if action == "GO_TO_TREE" else "Player asked for a cosmetic tree shake", owner="player_task")

    def check_gift(self, recipient, item, quantity):
        if recipient is self or self.world is None or not any(person is recipient for person in self.world.npcs):
            raise ValueError("Choose another NPC in this world to receive the items.")
        if (self.conversation is not None or recipient.conversation is not None
                or self.gift is not None or recipient.gift is not None or self.eating or recipient.eating or recipient.request_pending):
            raise ValueError("That person is busy now. Please try again later.")
        if item != "fruit" or type(quantity) is not int or quantity < 1:
            raise ValueError("Give a positive number of fruit.")
        if self.inventory.get(item, 0) < quantity:
            raise ValueError(f"{self.name} only has {self.inventory.get(item, 0)} fruit to give.")
        if recipient.inventory_space < quantity:
            raise ValueError(f"{recipient.name} only has room for {recipient.inventory_space} more items.")

    @property
    def eating(self):
        return self.behavior == Behavior.EAT and self.conversation is None and self.gift is None

    @property
    def giving(self):
        return (self.gift is not None and self.gift.initiator is self
                and self.gift.arrived and self.gift.elapsed < 1.0)

    @property
    def thanking(self):
        return self.gift is not None and self.gift.recipient is self and self.gift.elapsed >= 1.0

    def give_items(self, recipient, item, quantity, record=None, spacing=0):
        """Walk over and hand the items to the recipient before transferring them."""
        self.check_gift(recipient, item, quantity)
        return Gift(self, recipient, item, quantity, record, spacing)

    def _move(self, dt):
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        distance = math.hypot(dx, dy)
        step = MOVEMENT_SPEED * dt
        if distance <= step:
            self.x, self.y = self.destination
            return True
        if distance:
            self.x += dx / distance * step
            self.y += dy / distance * step
        return False

    def update(self, dt, tree):
        if self.conversation is not None or self.gift is not None or self.resume_pending:
            return
        world = self.world
        old_x, old_y = self.x, self.y
        left, right, top, bottom = self.movement_bounds
        dt = max(0.0, dt)
        behavior = self.behavior
        if behavior == Behavior.WANDER and not self.wandering_paused:
            if any(other.behavior == Behavior.COMPANY and other.target_id == self.id
                   and world.company_available(other) and math.hypot(other.x - self.x, other.y - self.y) <= SOCIAL_RADIUS
                   for other in world.npcs):
                self._face_when_still()
                return
            remaining = self.pause_remaining
            self.pause_remaining = max(0.0, remaining - dt)
            movement_dt = max(0.0, dt - remaining)
            if movement_dt > 0:
                if self.destination is None:
                    self.destination = (self.rng.uniform(left, right), self.rng.uniform(top, bottom))
                if self._move(movement_dt):
                    self.destination = None
                    self.pause_remaining = WANDER_PAUSE
        elif behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.HARVEST, Behavior.SHAKING):
            # Player visits wait briefly for an occupied tree; autonomous harvests fail fast.
            if not tree.reserve(self):
                self.pause_remaining += dt
                if self.pause_remaining >= DECISION_SECONDS:
                    self.finish_activity(world, failed=True)
                return
            if self.destination is not None:
                if self._move(dt):
                    self.destination = None
                    if behavior == Behavior.GO_TO_TREE:
                        self.finish_activity(world)
                    else:
                        tree.start_shake(self)
                        tree.shake_elapsed = self.activity_elapsed
                        if behavior == Behavior.SHAKE_TREE:
                            self.behavior = Behavior.SHAKING
            elif behavior in (Behavior.HARVEST, Behavior.SHAKING):
                if tree.shaker is not self:
                    tree.start_shake(self)
                    tree.shake_elapsed = self.activity_elapsed
                self.activity_elapsed += dt
                if self.activity_elapsed + 1e-9 >= SHAKE_SECONDS:
                    if behavior == Behavior.HARVEST:
                        if tree.fruit <= 0 or self.inventory_space <= 0:
                            self.finish_activity(world, failed=True)
                            return
                        # Commit both sides once, on the simulation thread.
                        tree.fruit -= 1
                        self.inventory['fruit'] = self.inventory.get('fruit', 0) + 1
                    self.finish_activity(world)
        elif behavior == Behavior.COMPANY:
            from autonomy import company_point
            partner = world.npc_by_id(self.target_id)
            self.activity_elapsed += dt
            if partner is None or not world.company_available(partner):
                self.finish_activity(world, failed=True)
            elif self.social >= SOCIAL_RECOVERED or self.activity_elapsed >= COMPANY_MAX_SECONDS:
                self.finish_activity(world)
            else:
                point = company_point(self, partner)
                if point is None:
                    self.finish_activity(world, failed=True)
                elif self.destination is not None or math.hypot(self.x - partner.x, self.y - partner.y) > SOCIAL_RADIUS:
                    self.destination = point
                    if self._move(dt):
                        self.destination = None
                else:
                    angle = math.atan2(partner.x - self.x, self.y - partner.y)
                    self.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
        elif behavior == Behavior.EAT:
            self.facing = 4
            self.activity_elapsed += dt
            if self.activity_elapsed + 1e-9 >= EAT_SECONDS:
                if self.inventory.get("fruit", 0) > 0:
                    self.inventory["fruit"] -= 1
                    self.social = min(100.0, self.social + APPLE_SATISFACTION)
                    self.finish_activity(world)
                else:
                    self.finish_activity(world, failed=True)
        elif behavior == Behavior.WAIT:
            self.activity_elapsed += dt
            if self.activity_elapsed >= DECISION_SECONDS:
                self.decision_remaining = 0.0
        self.x = min(right, max(left, self.x))
        self.y = min(bottom, max(top, self.y))
        dx, dy = self.x - old_x, self.y - old_y
        if dx or dy:
            angle = math.atan2(dx, -dy)
            self.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
        elif self.behavior != Behavior.COMPANY:
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

    def __init__(self, initiator, recipient, topic, tree, spacing=0, *, origin="player", purpose="chat"):
        if (initiator is recipient or initiator.conversation or recipient.conversation
                or initiator.gift or recipient.gift or initiator.eating or recipient.eating):
            raise ValueError("Choose another NPC who is free to talk.")
        self.initiator = initiator
        self.recipient = recipient
        if unsupported_location(topic):
            topic = "The current surroundings and how things are going"
        self.topic = topic
        self.id = uuid4().hex
        self.origin, self.purpose = origin, purpose
        self.dialogue_state = "waiting"
        self.wait_elapsed = 0.0
        self.talking_remaining = 0.0
        self.food_outcome = ""
        self.outcome_event_id = None
        self.reactions = []
        self.arrived = False
        self.active = True
        self.record = ConversationRecord((initiator.name, recipient.name), topic, "Walking to meet",
                                         origin=origin, purpose=purpose)
        self.turns = []
        self.pending_opinions = []
        self.error = ""
        self.saved_states = []
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
            self.saved_states.append(npc.activity_state())
            npc.release_claims(npc.world)
            npc.command_revision += 1
            npc.behavior = Behavior.WAIT
            npc.owner = "autonomous"
            npc.destination = None
            if purpose == "food_request":
                npc.reason = (f"Asking {recipient.name} for an apple" if npc is initiator
                              else f"Answering {initiator.name}'s request for an apple")
            else:
                npc.reason = "Having an automatic chat" if origin == "autonomous" else "In a player-initiated conversation"
            npc.target_id = None
            npc.activity_elapsed = 0.0
            npc.resume_pending = False
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
        if not self.accepts_response:
            return
        self.dialogue_state = "ready"
        self.talking_remaining = 4.0
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
        self.reactions = [
            (entry.reason, max(-4.0, min(4.0, entry.opinion / 20)))
            for entry in (response.initiator_opinion, response.recipient_opinion)
        ]

    @property
    def accepts_response(self):
        return self.active and self.purpose == "chat" and self.dialogue_state in ("waiting", "pending")

    def local_dialogue(self, lines):
        from llm import ConversationPart, ConversationTurn

        self.turns = [ConversationTurn(speaker=speaker, parts=[ConversationPart(text=text)])
                      for speaker, text in lines]
        self.record.turns = [(self.initiator.name if speaker == "initiator" else self.recipient.name, text)
                             for speaker, text in lines]

    def fallback(self, reason):
        if not self.accepts_response:
            return
        self.dialogue_state = "fallback"
        self.record.error = reason
        self.record.status = "Local greeting"
        self.record.outcomes = ["Exchanged a brief greeting; no conversation reaction was generated."]
        self.local_dialogue([("initiator", f"Hello, {self.recipient.name}."),
                             ("recipient", f"Hi, {self.initiator.name}.")])
        self.talking_remaining = 2.0

    def answer_food_request(self):
        from autonomy import donor_choice

        requester, donor = self.initiator, self.recipient
        world = requester.world
        world.publish("food_requested", (requester, donor),
                      f"{requester.name} asked {donor.name} for an apple.", self.record)
        self.food_outcome, answer = donor_choice(donor, requester)
        self.local_dialogue([("initiator", "Could you share an apple with me?"), ("recipient", answer)])
        self.dialogue_state = "ready"
        self.talking_remaining = 1.5
        self.record.status = "Answering food request"
        if self.food_outcome == "accepted":
            self.record.outcomes = ["Agreed to share one apple; the handoff has not happened yet."]
            return
        deliberate = self.food_outcome in ("need", "refused")
        text = (f"{donor.name} kept their last apple because they wanted to eat it."
                if self.food_outcome == "need" else f"{donor.name} declined {requester.name}'s apple request.")
        if not deliberate:
            text = f"{donor.name} could no longer share with {requester.name}: the inventories changed."
        event = world.publish("food_refused" if deliberate else "interaction_failed", (requester, donor), text, self.record)
        self.outcome_event_id = event.id
        self.record.outcomes = [answer]
        if deliberate:
            remember(requester, donor, event, "refused", f"{donor.name} declined my apple request: {answer}",
                     -1 if self.food_outcome == "need" else -5)
            remember(donor, requester, event, "kept_fruit", f"I kept my apple when {requester.name} asked: {answer}", 0)

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

    def face_each_other(self):
        for participant in (self.initiator, self.recipient):
            other = self.other(participant)
            angle = math.atan2(other.x - participant.x, participant.y - other.y)
            participant.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)

    def update(self, dt, *, reading=False):
        if not self.active:
            return
        if self.arrived:
            self.face_each_other()
            # Reading a live exchange holds both participants, not just the selected NPC.
            # Missing dialogue still reaches its bounded timeout while the view is open.
            if reading and self.turns:
                return
            if self.dialogue_state in ("waiting", "pending"):
                self.wait_elapsed += dt
                if self.wait_elapsed >= 12.0:
                    self.fallback("Dialogue timed out; used a local greeting.")
            else:
                self.talking_remaining = max(0.0, self.talking_remaining - dt)
                if self.talking_remaining <= 0:
                    if self.purpose == "food_request" and self.food_outcome == "accepted":
                        self.handoff()
                    else:
                        self.finish(self.initiator.world.tree)
            return
        npc = self.initiator
        old_x, old_y = npc.x, npc.y
        self.arrived = npc._move(max(0.0, dt))
        if self.arrived:
            self.record.status = "Waiting for dialogue"
            npc.destination = None
            self.face_each_other()
            if self.purpose == "food_request":
                self.answer_food_request()
        elif npc.x != old_x or npc.y != old_y:
            angle = math.atan2(npc.x - old_x, old_y - npc.y)
            npc.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)

    def snapshot(self, tree, npcs):
        def participant(npc):
            involved = npc is self.initiator or npc is self.recipient
            state = npc.snapshot(tree, self.other(npc).id) if involved else npc.snapshot(tree)
            if not involved:
                state.pop("memories", None)
            # These replies were addressed to the player before the NPCs met.
            state.pop("last_reply", None)
            return state

        return {
            "request_kind": "conversation",
            "interaction_id": self.id,
            "origin": self.origin,
            "initiator": participant(self.initiator),
            "recipient": participant(self.recipient),
            "topic": self.topic,
            "initiator_opinion_of_recipient": self.initiator.opinions.get(self.recipient.name),
            "recipient_opinion_of_initiator": self.recipient.opinions.get(self.initiator.name),
            "nearby_npcs": [participant(npc) for npc in npcs if npc is not self.initiator and npc is not self.recipient],
        }

    def finish(self, tree):
        if not self.active:
            return
        self.active = False
        self.record.status = "Finished (local greeting)" if self.dialogue_state == "fallback" else "Finished"
        if not self.turns:
            self.record.status = "Ended before dialogue"
            self.record.outcomes = ["The conversation ended before anyone spoke."]
        if self.purpose == "chat":
            world = self.initiator.world
            text = (f"{self.initiator.name} and {self.recipient.name} talked about {self.topic}."
                    if self.dialogue_state == "ready" else
                    f"{self.initiator.name} and {self.recipient.name} exchanged only a brief greeting."
                    if self.turns else f"{self.initiator.name} and {self.recipient.name}'s meeting ended before dialogue.")
            event = world.publish("conversation_completed" if self.turns else "interaction_failed",
                                  (self.initiator, self.recipient), text, self.record)
            self.outcome_event_id = event.id
            if self.dialogue_state == "ready":
                for npc, reaction in zip((self.initiator, self.recipient), self.reactions):
                    reason, weight = reaction
                    remember(npc, self.other(npc), event, "conversation", f"Talked about {self.topic}. {reason}", weight)
        elif self.food_outcome in ("", "accepted"):
            self.record.status = "Ended before sharing"
            self.record.outcomes = ["The food request ended before an apple was shared."]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient),
                                                self.record.outcomes[0], self.record)
            self.outcome_event_id = event.id
        for npc, state in zip((self.initiator, self.recipient), self.saved_states):
            state["cooldowns"][self.other(npc).id] = PARTNER_COOLDOWN
            npc.conversation = None
            npc.restore_activity(state)
            # Physical resumption and claim acquisition wait for the next update.
            npc.resume_pending = True

    def handoff(self):
        """Reverse the roles without restoring activities or reacquiring tree claims."""
        requester, donor = self.initiator, self.recipient
        self.active = False
        requester.conversation = donor.conversation = None
        Gift(donor, requester, "fruit", 1, self.record,
             saved_states=[self.saved_states[1], self.saved_states[0]], already_met=True)


class Gift:
    """Reserve both participants through approach, giving, and thanking."""

    def __init__(self, initiator, recipient, item, quantity, record=None, spacing=0, *, saved_states=None, already_met=False):
        from autonomy import company_point

        destination = (initiator.x, initiator.y) if already_met else company_point(initiator, recipient, spacing)
        if destination is None:
            raise ValueError("There isn't enough room here to give the items.")
        self.initiator, self.recipient = initiator, recipient
        self.item, self.quantity = item, quantity
        self.arrived = already_met
        self.active = True
        self.transferred = False
        self.outcome_event_id = None
        self.elapsed = 0.0
        self.record = record or ConversationRecord((initiator.name, recipient.name), "Give items", "Walking to give")
        self.saved_states = saved_states if saved_states is not None else [initiator.activity_state(), recipient.activity_state()]
        self.record.status = "Giving items" if already_met else "Walking to give"
        for npc in (initiator, recipient):
            npc.release_claims(npc.world)
            npc.command_revision += 1
            npc.behavior = Behavior.WAIT
            npc.owner = "autonomous"
            npc.destination = None
            npc.target_id = None
            npc.activity_elapsed = 0.0
            npc.resume_pending = False
            npc.reason = "Giving items" if npc is initiator else "Receiving items"
            npc.gift = self
        initiator.destination = None if already_met else destination

    def update(self, dt):
        if not self.active:
            return
        giver, recipient = self.initiator, self.recipient
        if not self.arrived:
            old_x, old_y = giver.x, giver.y
            self.arrived = giver._move(max(0.0, dt))
            if giver.x != old_x or giver.y != old_y:
                angle = math.atan2(giver.x - old_x, old_y - giver.y)
                giver.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
            if self.arrived:
                giver.destination = None
                self.record.status = "Giving items"
                for npc, other in ((giver, recipient), (recipient, giver)):
                    angle = math.atan2(other.x - npc.x, npc.y - other.y)
                    npc.facing = math.floor(angle / (math.pi / 4) + 0.5) % len(DIRECTIONS)
            return
        elapsed = self.elapsed + max(0.0, dt)
        if not self.transferred and elapsed + 1e-9 >= 1.0:
            if giver.inventory.get(self.item, 0) < self.quantity or recipient.inventory_space < self.quantity:
                self.finish("The items could not be given because the inventories changed.")
                return
            giver.inventory[self.item] -= self.quantity
            recipient.inventory[self.item] = recipient.inventory.get(self.item, 0) + self.quantity
            self.transferred = True
            apples = "an apple" if self.quantity == 1 else f"{self.quantity} apples"
            event = giver.world.publish("fruit_shared", (giver, recipient),
                                        f"{giver.name} shared {apples} with {recipient.name}.", self.record)
            self.outcome_event_id = event.id
            remember(giver, recipient, event, "gave_fruit", f"I gave {recipient.name} {apples}.", 2)
            remember(recipient, giver, event, "received_fruit", f"{giver.name} shared {apples} with me.", 6)
            if self.record.purpose == "food_request":
                self.record.turns.append((recipient.name, "Thank you for the apple."))
            self.record.outcomes[0:1] = [f"Gave {self.quantity} {self.item} to {recipient.name}."]
            self.record.status = "Thanking"
            elapsed = max(1.0, elapsed)
        self.elapsed = elapsed
        if self.elapsed + 1e-9 >= 2.0:
            self.finish()

    def finish(self, error=""):
        if not self.active:
            return
        self.active = False
        self.record.status = "Failed" if error else "Completed"
        self.record.error = error
        if error:
            self.record.outcomes[0:1] = [error]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient), error, self.record)
            self.outcome_event_id = event.id
        for npc, state in zip((self.initiator, self.recipient), self.saved_states):
            other = self.recipient if npc is self.initiator else self.initiator
            state["cooldowns"][other.id] = PARTNER_COOLDOWN
            npc.gift = None
            facing = npc.facing
            npc.restore_activity(state)
            npc.facing = facing
            npc.resume_pending = True
