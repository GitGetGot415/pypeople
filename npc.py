from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import random
from uuid import uuid4

from character_data import DIRECTIONS, PROFILE_FIELDS, normalize_opinions
from creations import CreativeJob
from emotions import portrait_expression
from purgatory import initial_attitudes
from conversation_history import (ConversationRecord, describe_opinion_change, relevant_memories, remember,
                                  shared_conversation_history)
from social import (ConversationConcern, Goal, check_relationship_goals, commit_agreements, commit_concern_use, commit_information,
                    concern_description, concern_topic, conversation_references, dialogue_failure_cooldown,
                    information_views, on_transfer, permitted_proposals, phase_label, prepare_information,
                    private_context, validate_live_agreements)
from constants import (
    APPLE_SATISFACTION, COMPANY_MAX_SECONDS, DECISION_SECONDS, DEFAULT_CHARACTER_NAME, DIALOGUE_TIMEOUT_SECONDS, EAT_SECONDS,
    CONVERSATION_PAUSE_RANGE, CONVERSATION_RETURN_PAUSE, FAILURE_COOLDOWN, GIFT_GIVE_SECONDS, GIFT_SECONDS,
    MOVEMENT_SPEED, NPC_RADIUS, PARTNER_COOLDOWN,
    POST_DIALOGUE_SECONDS, SHAKE_SECONDS, SNAKE_NAME, SOCIAL_RADIUS, SOCIAL_RECOVERED, WANDER_PAUSE, WORLD_HEIGHT, WORLD_WIDTH,
    STARTING_DAY, STARTING_FACING, STARTING_NPC_FRUIT, STARTING_NPC_GENEROSITY_RANGE,
    STARTING_NPC_POSITION, STARTING_NPC_SATISFACTION_RANGE, STARTING_NPC_SOCIABILITY_RANGE,
    STARTING_TREE_POSITION, STARTING_FATIGUE_RANGE, EXCHANGE_FATIGUE,
)
from interactions import (company_point, conversation_points, dialogue_finished, face_each_other, face_toward,
                          finish_generated_dialogue, move_toward, record_failed_dialogue, record_generated_turns)
from name_knowledge import actor_catalog, commit_names, name_context, orientation_context, validate_names, validate_orientation


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
    GO_HOME = "GO_HOME"
    REST = "REST"
    SLEEP = "SLEEP"
    PAINT = "PAINT"
    COMPOSE = "COMPOSE"


@dataclass
class Tree:
    x: float = STARTING_TREE_POSITION[0]
    y: float = STARTING_TREE_POSITION[1]
    shake_elapsed: float | None = None
    shaker: "NPC | None" = field(default=None, repr=False, compare=False)
    id: str = "fruit-tree"
    fruit: int = 0
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
        if self.shaker is not None and self.shaker is not npc:
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
    x: float = STARTING_NPC_POSITION[0]
    y: float = STARTING_NPC_POSITION[1]
    behavior: Behavior = Behavior.WANDER
    destination: tuple[float, float] | None = None
    pause_remaining: float = 0.0
    rng: random.Random = field(default_factory=random.Random, repr=False)
    facing: int = STARTING_FACING
    sprite_size: tuple[int, int] | None = None
    selected: bool = False
    reply: str = ""
    rewritten_reply: str = ""
    end_conversation: bool = False
    conversation: "Conversation | None" = field(default=None, repr=False, compare=False)
    gift: "Gift | None" = field(default=None, repr=False, compare=False)
    display_name: str | None = None
    emotion: str = "neutral"
    emotion_intensity: int = 0
    id: str = field(default_factory=lambda: uuid4().hex)
    social: float = field(default_factory=lambda: random.uniform(*STARTING_NPC_SATISFACTION_RANGE))
    inventory: dict = field(default_factory=lambda: {"fruit": STARTING_NPC_FRUIT})
    preferences: dict = field(default_factory=lambda: {
        "sociability": random.uniform(*STARTING_NPC_SOCIABILITY_RANGE),
        "generosity": random.uniform(*STARTING_NPC_GENEROSITY_RANGE),
    })
    memories: list = field(default_factory=list)
    goal: Goal | None = None
    pending_social_opinions: list = field(default_factory=list)
    beliefs: list = field(default_factory=list)
    current_concern: ConversationConcern | None = None
    conversation_pause_remaining: float = 0.0
    conversation_return_remaining: float = 0.0
    known_names: dict = field(default_factory=dict)
    player_opinion: dict = field(default_factory=lambda: {"opinion": 0, "reason": ""})
    player_call: dict | None = None
    player_call_cooldown: float = 0.0
    resolved_player_calls: list = field(default_factory=list)
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
    attitudes: dict = field(default_factory=initial_attitudes)
    arrival_day: int = STARTING_DAY
    departure: dict | None = None
    snake_encounter: object = field(default=None, repr=False, compare=False)

    room_number: int = 0
    social_fatigue: float = field(default_factory=lambda: random.uniform(*STARTING_FATIGUE_RANGE))
    home_remaining: float = 0.0
    outdoor_remaining: float = 0.0
    bedtime_remaining: float | None = None
    wake_remaining: float | None = None
    home_reason: str | None = None
    last_creative_day: int = 0
    creative_job: CreativeJob | None = field(default=None, repr=False, compare=False)
    location: str = "garden"
    journey: dict | None = None

    @property
    def indoors(self):
        return self.location != "garden"

    @property
    def interaction(self):
        return self.conversation or self.gift or self.snake_encounter

    @property
    def busy(self):
        return self.interaction is not None or self.eating

    @property
    def unread_player_reply(self):
        call = self.player_call
        return call if call and call["kind"] == "reply" else None

    @property
    def player_thinking(self):
        return (self.request_pending and self.world is not None
                and not any(action.npc_id == self.id for action in self.world.pending_actions))

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

    def suspend_activity(self, reason):
        """Save progress and release physical claims before an interaction."""
        state = self.activity_state()
        self.release_claims(self.world)
        if not self.creative_job:
            self.command_revision += 1
        self.behavior, self.owner = Behavior.WAIT, "autonomous"
        self.destination = self.target_id = None
        self.activity_elapsed = 0.0
        self.resume_pending = False
        self.reason = reason
        return state

    def restore_after_interaction(self, state, *cooldown_ids, preserve_facing=False):
        facing = self.facing
        self.restore_activity(state)
        if preserve_facing:
            self.facing = facing
        self.cooldowns.update(dict.fromkeys(cooldown_ids, PARTNER_COOLDOWN))
        # Reacquire physical claims on the next simulation tick.
        self.resume_pending = True
        if self.creative_job:
            self.creative_job.revision = self.command_revision

    def pause_after_conversation(self, completed):
        self.conversation_pause_remaining = (self.rng.uniform(*CONVERSATION_PAUSE_RANGE)
                                             if completed else FAILURE_COOLDOWN)
        if completed:
            self.social_fatigue = min(100.0, self.social_fatigue + EXCHANGE_FATIGUE)
            self.conversation_return_remaining = CONVERSATION_RETURN_PAUSE

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
        if behavior != Behavior.GO_HOME:
            self.home_reason = None
            self.bedtime_remaining = None
        self.behavior, self.target_id = behavior, target
        self.owner, self.reason = owner, reason
        self.destination = None
        self.activity_elapsed = self.pause_remaining = 0.0
        self.resume_pending = False
        self.decision_remaining = DECISION_SECONDS
        if behavior == Behavior.EAT:
            self.facing = 4
        elif behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE):
            if self.location != "garden":
                from locations import start_journey
                start_journey(self, "garden", world.tree.interaction_point, purpose="tree", fast=False)
                return
            if behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE):
                if not world.join_tree(self, "bargain" if behavior == Behavior.HARVEST else "introduction"):
                    self.finish_activity(world, failed=True)
                return
            if target != world.tree.id or not self.can_reach(world.tree.interaction_point):
                self.finish_activity(world, failed=True)
                return
            self.destination = world.tree.interaction_point
        elif behavior == Behavior.COMPANY:
            partner = world.npc_by_id(target)
            self.destination = company_point(self, partner) if partner is not None else None
            if self.destination is None:
                self.finish_activity(world, failed=True)

    def resume_activity(self, world):
        """Reacquire suspended claims on a simulation tick, preserving committed progress."""
        self.resume_pending = False
        if self.behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE):
            self.begin_activity(self.behavior, world, world.tree.id, self.reason, self.owner)
        elif self.behavior in (Behavior.SHAKE_TREE, Behavior.SHAKING):
            if (self.target_id != world.tree.id or not self.can_reach(world.tree.interaction_point)
                    or self.location != "garden"):
                self.finish_activity(world, failed=True)
                return
            if self.behavior == Behavior.SHAKING:
                self.behavior = Behavior.SHAKE_TREE
            self.destination = world.tree.interaction_point
        elif self.behavior == Behavior.COMPANY:
            partner = world.npc_by_id(self.target_id)
            if partner is None or not world.company_available(partner):
                self.finish_activity(world, failed=True)
            else:
                self.destination = company_point(self, partner)

    def update_cooldowns(self, dt):
        self.decision_remaining = max(0.0, self.decision_remaining - dt)
        if self.interaction is None and not self.request_pending and not self.end_conversation:
            self.conversation_pause_remaining = max(0.0, self.conversation_pause_remaining - dt)
            self.conversation_return_remaining = max(0.0, self.conversation_return_remaining - dt)
            if self.current_concern:
                self.current_concern.remaining = max(0.0, self.current_concern.remaining - dt)
        self.cooldowns = {key: value - dt for key, value in self.cooldowns.items() if value > dt}
        interaction = self.conversation or self.gift
        if interaction is not None:
            index = 0 if interaction.initiator is self else 1
            suspended = interaction.saved_states[index]
            suspended["decision_remaining"] = max(0.0, suspended["decision_remaining"] - dt)
            suspended["cooldowns"] = {key: value - dt for key, value in suspended["cooldowns"].items() if value > dt}

    @property
    def portrait_emotion(self):
        return portrait_expression(self.emotion, self.emotion_intensity)

    def set_emotion(self, name, intensity):
        """Keep the reaction with its reply until new dialogue replaces it."""
        self.emotion = name
        self.emotion_intensity = intensity

    @property
    def name(self):
        return self.display_name if self.display_name is not None else self.character["Name"].strip()

    @property
    def movement_bounds(self):
        from locations import feet_bounds
        return feet_bounds(self.location)

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
        if self.departure:
            from dialogue_rules import named_references
            return f"Departed on day {self.departure['day']} ({named_references(self.departure['cause'])})"
        if self.journey:
            return f"Travelling to {self.journey['destination']}"
        if self.world and self.world.tree_has_actor(self.id) and self.snake_encounter is None:
            if self.world.tree_current and self.world.tree_current["actor_id"] == self.id:
                return "Approaching the front of the tree line"
            position = next(i for i, e in enumerate(self.world.tree_queue, 1) if e["actor_id"] == self.id)
            return f"Waiting in the tree line ({position})"
        if self.snake_encounter is not None:
            return f"With {SNAKE_NAME}: {self.snake_encounter.record.status}"
        if self.gift is not None:
            if self.thanking:
                return f"Thanking {self.gift.initiator.name}"
            if self.gift.initiator is self:
                if self.gift.elapsed >= GIFT_GIVE_SECONDS:
                    return f"Waiting for {self.gift.recipient.name}'s thanks"
                return f"Giving to {self.gift.recipient.name}" if self.gift.arrived else f"Walking to {self.gift.recipient.name}"
            return f"Receiving from {self.gift.initiator.name}"
        if self.conversation is not None:
            if self.conversation.arrived:
                return f"Talking with {self.conversation.other(self).name}"
            partner = self.conversation.other(self)
            return f"Walking to meet {partner.name}" if self.destination is not None else f"Waiting for {partner.name}"
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
            Behavior.HARVEST: "Walking to the guarded tree",
            Behavior.EAT: "Eating an apple",
            Behavior.COMPANY: "Seeking company" if self.destination else "Spending time together",
            Behavior.WAIT: "Waiting",
            Behavior.GO_HOME: "Walking home",
            Behavior.REST: "Relaxing at home",
            Behavior.SLEEP: "Sleeping at home",
            Behavior.PAINT: "Painting at home",
            Behavior.COMPOSE: "Composing at home",
        }[self.behavior]

    def contains(self, point):
        if self.sprite_size is not None:
            width, height = self.sprite_size
            return (
                self.x - width / 2 <= point[0] < self.x + width / 2
                and self.y - height <= point[1] < self.y
            )
        return math.hypot(point[0] - self.x, point[1] - self.y) <= NPC_RADIUS

    def snapshot(self, tree, partner_id=None, *, private=True):
        from locations import environment
        state = {
            "location": self.location,
            "environment": environment(self.location),
            "npc_id": self.id,
            "npc_name": self.name,
            "character": {key: value for key, value in self.character.items() if key != "Opinions"},
            "name_knowledge": name_context(self),
            "_name_actors": actor_catalog(self.world, self.location) if self.world else {},
            "facing": DIRECTIONS[self.facing],
            "selected": self.selected,
            "position": [round(self.x, 1), round(self.y, 1)],
            "behavior": self.behavior.value,
            "activity": self.status,
            "activity_reason": self.decision_reason,
            "activity_owner": self.owner,
            "satisfaction": round(self.social, 1),
            "inventory": self.inventory.copy(),
            "tree_interaction_point": list(tree.interaction_point),
            "tree_fruit": tree.fruit,
            "tree_available_fruit": self.world.available_fruit if self.world else tree.fruit,
            "tree_occupied": tree.reserved_by is not None,
            "day": self.world.day if self.world else 1,
            "living_count": self.world.population if self.world else 1,
            "time_of_day": self.world.phase if self.world else "day",
            "tree_is_shaking": tree.shaking,
            "last_reply": self.reply,
        }
        if private:
            from player import work_choices
            from player_story import starting_encounter, starting_dialogue_context
            state["owned_works"] = work_choices(self.world, self.id)
            state["player_relationship"] = {"actor_id": self.world.player.id,
                "learned_address": self.known_names.get(self.world.player.id), "opinion": self.player_opinion.copy()}
            state["memories"] = [{"partner_id": memory.partner_id, "kind": memory.kind,
                          "summary": memory.summary, "day": memory.day, "phase": memory.phase}
                                 for memory in relevant_memories(self, partner_id)]
            state["private_context"] = private_context(self, partner_id)
            starting_experience = starting_dialogue_context(self.world, self.id)
            if starting_experience:
                state["private_context"]["starting_experience"] = starting_experience
            state["private_context"]["opinions"] = self.opinions
            state["private_context"]["snake_attitudes"] = self.attitudes.copy()
            advance = self.world.snake.advances.get(self.id)
            state["private_context"]["snake_apple_advance"] = advance.copy() if advance else None
            encounters = [record for record in self.world.history if self.name in record.participants and record.turns]
            state["private_context"]["situation"] = {
                **orientation_context(self),
                "arrival_day": self.arrival_day,
                "transition_memory": "absent", "explanation": "unknown", "way_home": "unknown",
                "prior_conversations": len(encounters),
                "prior_snake_encounters": (sum(self.world.snake.name in record.participants for record in encounters)
                                           + int(starting_encounter(self.world, self.id) is not None)),
                "received_snake_apple": any(fact.kind == "bargain" and fact.subject_id == self.world.snake.id
                                            and fact.object_id == self.id for fact in self.world.facts),
                "known_danger": [account for account in state["private_context"]["known_accounts"]
                                 if account["kind"] in ("eating", "judgment", "defiance")][:3],
                "apple_bargain": {"stage": "apple already received; offering still owed", **advance} if advance else None,
            }
        else:
            state.pop("name_knowledge", None)
            state.pop("_name_actors", None)
            state["character"].pop("Opinions", None)
            for field in PROFILE_FIELDS:
                state["character"].pop(field, None)
            for key in ("last_reply", "activity_reason"):
                state.pop(key, None)
        state["arrival_day"] = self.arrival_day
        state["tree_guarded"] = True
        if self.indoors:
            for key in [key for key in state if key.startswith("tree_")]:
                state.pop(key)
        return state

    @property
    def opinions(self):
        # Copy nested entries so worker snapshots cannot modify live opinions.
        return {person: entry.copy() for person, entry in self.character.get("Opinions", {}).items()}

    def opinion_score(self, actor_id, default=0):
        """Resolve engine identity here; profile and prompt dictionaries retain labels."""
        target = self.world.participant_by_id(actor_id)
        return self.opinions.get(target.name, {}).get("opinion", default) if target else default

    def remember_opinion(self, actor_id, opinion, reason):
        target = self.world.participant_by_id(actor_id)
        if target is None:
            raise ValueError("The opinion target is no longer available.")
        self.remember_opinions({target.name: {"opinion": opinion, "reason": reason}})

    def remember_opinions(self, updates):
        updates = normalize_opinions(updates)
        if self.world:
            for person, entry in updates.items():
                target = next((npc for npc in self.world.npcs if npc.name == person), None)
                outcomes = [p.outcome for p in self.world.promises if p.status != "pending" and target is not None
                            and {p.promisor_id, p.recipient_id} == {self.id, target.id}]
                if outcomes and outcomes[-1] not in entry["reason"]:
                    entry["reason"] = (outcomes[-1] + " Current impression: " + entry["reason"])[:400]
        opinions = normalize_opinions({**self.opinions, **updates})
        if opinions == self.opinions:
            return
        self.character["Opinions"] = opinions
        if self.world:
            interaction = self.conversation or self.gift
            check_relationship_goals(self.world, self, interaction.record if interaction else None)

    def _face_when_still(self):
        point = self.world.tree.interaction_point if self.world else None
        at_tree = point is not None and math.hypot(self.x - point[0], self.y - point[1]) <= 1e-6
        self.facing = 2 if not self.eating and (at_tree or self.behavior == Behavior.SHAKING) else 4

    def set_selected(self, selected):
        """Defer new decisions while allowing the current finite activity to finish."""
        self.selected = selected
        if not selected:
            self.end_conversation = False
        if self.conversation is None and self.gift is None and selected and (self.wandering_paused or self.behavior in (Behavior.STOP, Behavior.SHAKING)):
            self._face_when_still()

    def apply_action(self, action, tree):
        from locations import available
        if self.world is None or self.world.npc_by_id(self.id) is not self or not available(self):
            raise ValueError("Only living residents can act in this world.")
        # Reject before changing anything. SAY preserves even an active shake.
        if action == "SAY":
            return
        if action not in ("WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "EAT"):
            raise ValueError("Unsupported NPC action")
        if self.busy:
            raise ValueError("This NPC is already in an interaction.")
        if action == "EAT" and self.inventory.get("fruit", 0) < 1:
            raise ValueError("I don't have an apple to eat.")
        if self.world.visit and self.world.visit["host_id"] == self.id:
            from apartments import finish_visit
            finish_visit(self.world)
        self.journey = None
        if self.creative_job:
            from apartments import cancel_work
            cancel_work(self)
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

    def check_gift(self, recipient, item, quantity, *, player_authorization=None):
        from player import validate_transfer
        validate_transfer(self.world, self, recipient, item, quantity, player_authorization=player_authorization)
        if recipient is self.world.player:
            return
        if (recipient is self or self.world is None or self.world.npc_by_id(self.id) is not self
                or not any(person is recipient for person in self.world.npcs)):
            raise ValueError("Choose another NPC in this world to receive the items.")
        from apartments import available_outdoors
        if not available_outdoors(self) or not available_outdoors(recipient):
            raise ValueError("Both residents must be available outdoors to exchange items.")
        if self.busy or recipient.busy or recipient.request_pending:
            raise ValueError("That person is busy now. Please try again later.")
        if item != "fruit":
            return
        if type(quantity) is not int or quantity < 1:
            raise ValueError("Give a positive number of fruit.")
        if self.inventory.get(item, 0) < quantity:
            raise ValueError(f"{self.name} only has {self.inventory.get(item, 0)} fruit to give.")

    @property
    def eating(self):
        return self.behavior == Behavior.EAT and self.conversation is None and self.gift is None

    @property
    def giving(self):
        return (self.gift is not None and self.gift.initiator is self
                and self.gift.arrived and self.gift.elapsed < GIFT_GIVE_SECONDS)

    @property
    def thanking(self):
        return self.gift is not None and self.gift.recipient is self and self.gift.elapsed >= GIFT_GIVE_SECONDS

    def give_items(self, recipient, item, quantity, record=None, spacing=0):
        """Walk over and hand the items to the recipient before transferring them."""
        self.check_gift(recipient, item, quantity)
        if recipient is self.world.player:
            from player import PlayerHandoff
            return PlayerHandoff(self.world, self, recipient, item, quantity, record)
        return Gift(self, recipient, item, quantity, record, spacing)

    def _move(self, dt):
        return move_toward(self, self.destination, dt, MOVEMENT_SPEED)

    def update(self, dt, tree):
        from player import owns_resident
        if owns_resident(self.world, self) or self.world.tree_has_actor(self.id):
            return
        if self.journey:
            from locations import update_journey
            if update_journey(self, dt) and self.behavior in (Behavior.GO_TO_TREE, Behavior.HARVEST, Behavior.SHAKE_TREE):
                self.begin_activity(self.behavior, self.world, self.world.tree.id, self.reason, self.owner)
            return
        if self.interaction is not None or self.resume_pending or self.player_call:
            return
        world = self.world
        if self.behavior in (Behavior.REST, Behavior.SLEEP, Behavior.PAINT, Behavior.COMPOSE, Behavior.GO_HOME):
            from apartments import update_home
            update_home(self, world, max(0.0, dt))
            return
        old_x, old_y = self.x, self.y
        left, right, top, bottom = self.movement_bounds
        dt = max(0.0, dt)
        behavior = self.behavior
        if self.selected and behavior == Behavior.COMPANY:
            return
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
        elif behavior in (Behavior.GO_TO_TREE, Behavior.HARVEST):
            self.begin_activity(behavior, world, tree.id, self.reason, self.owner)
        elif behavior in (Behavior.SHAKE_TREE, Behavior.SHAKING):
            if self.destination is not None:
                if self._move(dt):
                    self.destination = None
                    tree.start_shake(self)
                    tree.shake_elapsed = self.activity_elapsed
                    if behavior == Behavior.SHAKE_TREE:
                        self.behavior = Behavior.SHAKING
            elif behavior == Behavior.SHAKING:
                if tree.shaker is not self:
                    tree.start_shake(self)
                    tree.shake_elapsed = self.activity_elapsed
                self.activity_elapsed += dt
                if self.activity_elapsed + 1e-9 >= SHAKE_SECONDS:
                    self.finish_activity(world)
        elif behavior == Behavior.COMPANY:
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
                    face_toward(self, (partner.x, partner.y))
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
            face_toward(self, (self.x + dx, self.y + dy))
        elif self.behavior != Behavior.COMPANY:
            self._face_when_still()



def assign_npc_names(npcs, *, used_names=()):
    """Assign new labels without renaming any existing residents."""
    from constants import PLAYER_LABEL
    used = {name.casefold() for name in used_names} | {"python", SNAKE_NAME.casefold(), PLAYER_LABEL.casefold()}
    for npc in npcs:
        base = npc.character["Name"].strip() or DEFAULT_CHARACTER_NAME
        label, number = base, 2
        while label.casefold() in used:
            label = f"{base} ({number})"
            number += 1
        npc.display_name = label
        used.add(label.casefold())


class Conversation:
    """Temporarily reserve two NPCs, then restore their interrupted activities."""

    def __init__(self, initiator, recipient, topic, tree, spacing=0, *, origin="player", purpose="chat",
                 opening_kind=None, focus_claim_id=None, goal_id=None):
        from locations import available as available_outdoors, colocated
        from player import owns_resident
        if (initiator is recipient or initiator.busy or recipient.busy
                or owns_resident(initiator.world, initiator) or owns_resident(initiator.world, recipient)
                or initiator.world.visit and initiator.world.visit["host_id"] in (initiator.id, recipient.id)
                or initiator.player_call or recipient.player_call or recipient.selected or recipient.request_pending
                or not available_outdoors(initiator) or not available_outdoors(recipient)
                or not colocated(initiator, recipient)):
            raise ValueError("Choose another NPC who is free to talk.")
        if purpose not in ("chat", "food_request"):
            raise ValueError("Resident meetings use chat or food_request.")
        if focus_claim_id and not any(b.claim_id == focus_claim_id for b in initiator.beliefs):
            raise ValueError("The initiator does not know the required account.")
        self.initiator = initiator
        self.recipient = recipient
        self.topic = topic
        self.id = uuid4().hex
        self.origin, self.purpose = origin, purpose
        self.concerns = {role: ConversationConcern(**asdict(npc.current_concern)) if npc.current_concern else None
                         for role, npc in (("initiator", initiator), ("recipient", recipient))}
        self.opening_kind = opening_kind
        self.focus_claim_id = focus_claim_id  # Required task account; an ordinary concern is optional.
        self.goal_id = goal_id
        self.dialogue_state = "waiting"
        self.wait_elapsed = 0.0
        self.talking_remaining = POST_DIALOGUE_SECONDS
        self.food_outcome = ""
        self.food_reason = ""
        self.handoff_thanks = []
        self.rewritten_handoff_thanks = []
        self.outcome_event_id = None
        self.reactions = []
        self.arrived = False
        self.active = True
        self.record = ConversationRecord((initiator.name, recipient.name), topic, "Walking to meet",
                                         origin=origin, purpose=purpose, id=self.id,
                                         participant_ids=[initiator.id, recipient.id])
        self.record.intended_subjects = {
            npc.name: concern_description(npc, self.concerns[role])
            for role, npc in (("initiator", initiator), ("recipient", recipient))}
        self.proposals = []
        self.accepted_proposals = []
        self.social_effects_applied = False
        self.context_references = {}
        self.name_actors = {}
        self.information_context = None
        self.accepted_disclosures = []
        self.third_party_opinions = []
        self.leverage_uses = []
        self.accepted_gift = None
        self.error = ""
        self.saved_states = []
        destinations = conversation_points(initiator, recipient, spacing)
        if destinations is None:
            raise ValueError("There isn't enough room here for a conversation.")
        for npc in (initiator, recipient):
            if purpose == "food_request":
                reason = (f"Asking {recipient.name} for an apple" if npc is initiator
                          else f"Answering {initiator.name}'s request for an apple")
            else:
                reason = "Having an automatic chat" if origin == "autonomous" else "In a player-initiated conversation"
            self.saved_states.append(npc.suspend_activity(reason))
            npc.conversation = self
        initiator.destination, recipient.destination = destinations

    def other(self, npc):
        return self.recipient if npc is self.initiator else self.initiator

    @property
    def turns(self):
        return self.record.generated_turns()

    @property
    def error(self):
        return self.record.error

    @error.setter
    def error(self, message):
        self.record.error = message

    def set_response(self, response, *, reading=False, rewritten_parts=(), dialogue_validation=True):
        if not self.accepts_response:
            return
        from llm import (dialogue_validation_errors, validate_dialogue_format,
                         validate_dialogue_speakers, validate_resident_grounding)
        from name_knowledge import record_self_introductions
        # Reject stale actor snapshots before displaying or committing anything.
        if getattr(self, "request_roster_revision", self.initiator.world.roster_revision) != self.initiator.world.roster_revision:
            raise ValueError("The resident roster changed while dialogue was being prepared. Retry this meeting.")
        if any(n.command_revision != self.request_actor_revisions[n.id] for n in (self.initiator, self.recipient)):
            raise ValueError("The participants changed activities before the reply arrived.")
        validate_live_agreements(self, response, dialogue_validation=dialogue_validation)
        from llm import validate_resident_gift
        validate_resident_gift(response.gift, response.turns, {
            "purpose": self.purpose, "arrival_focus": self.opening_kind == "orientation",
            "initiator": self.initiator.snapshot(self.initiator.world.tree),
            "recipient": self.recipient.snapshot(self.initiator.world.tree)})
        if response.gift and response.gift.accept_turn is not None:
            from player import validate_transfer
            world = self.initiator.world
            validate_transfer(world, world.npc_by_id(response.gift.giver_id), world.npc_by_id(response.gift.recipient_id),
                              response.gift.item, response.gift.quantity, interaction=self)
        if dialogue_validation:
            validate_dialogue_format(response.turns)
            validate_dialogue_speakers(response.turns, {"initiator": self.initiator.name, "recipient": self.recipient.name})
            validate_resident_grounding(response, {
                "_name_actors": self.name_actors,
                "initiator": self.initiator.snapshot(self.initiator.world.tree),
                "recipient": self.recipient.snapshot(self.initiator.world.tree),
                "_information_validation": asdict(self.information_context) if self.information_context else None})
        name_snapshot = {"request_kind": "conversation", "interaction_id": self.id,
                         "purpose": self.purpose, "arrival_focus": self.opening_kind == "orientation", "_dialogue_validation": dialogue_validation,
                         "initiator": self.initiator.snapshot(self.initiator.world.tree),
                         "recipient": self.recipient.snapshot(self.initiator.world.tree),
                         "_name_actors": self.name_actors}
        record_self_introductions(response.turns, name_snapshot)
        knowledge, _, name_changes = validate_names(response.turns, name_snapshot)
        if response.handoff_thanks:
            from llm import ConversationTurn
            thanks_role = "recipient" if response.gift and response.gift.recipient_id == self.recipient.id else "initiator"
            validate_names([ConversationTurn(speaker=thanks_role, parts=response.handoff_thanks)], {
                **name_snapshot, thanks_role: {**name_snapshot[thanks_role],
                    "name_knowledge": {"known_names": knowledge[thanks_role]}}})
        orientation_discussed = validate_orientation(response, name_snapshot)
        if not dialogue_validation:
            self.record.dialogue_errors = list(dict.fromkeys([
                *response._dialogue_errors,
                *dialogue_validation_errors(response, self.snapshot(self.initiator.world.tree, self.initiator.world.npcs))]))
        self.handoff_thanks = response.handoff_thanks
        self.accepted_gift = response.gift if response.gift and response.gift.accept_turn is not None else None
        self.rewritten_handoff_thanks = list(rewritten_parts[-1]) if rewritten_parts and response.handoff_thanks else []
        record_generated_turns(self, response.turns, rewritten_parts)
        opinions = ((self.initiator, self.recipient, response.initiator_opinion),
                    (self.recipient, self.initiator, response.recipient_opinion))
        self.record.outcomes = [
            describe_opinion_change(
                npc.name, partner.name, npc.opinion_score(partner.id, default=None), entry.opinion, entry.reason
            )
            for npc, partner, entry in opinions
        ]
        self.record.status = "Dialogue ready"
        self.record.orientation_discussed = orientation_discussed
        self.accepted_proposals = list(response.accepted_proposals)
        self.accepted_disclosures = list(response.disclosures)
        self.third_party_opinions = list(response.third_party_opinions)
        self.leverage_uses = list(response.leverage)
        if not self.social_effects_applied:
            commit_agreements(self)
            commit_information(self)
            commit_names(self, name_changes)
            goal = self.initiator.goal
            if goal and goal.id == self.goal_id and goal.status == "active":
                goal.last_attempt_phase = self.initiator.world.phase_index
            commit_concern_use(self)
            self.social_effects_applied = True
        for npc, partner, entry in opinions:
            npc.remember_opinion(partner.id, entry.opinion, entry.reason)
        self.reactions = [
            (entry.reason, max(-4.0, min(4.0, entry.opinion / 20)))
            for entry in (response.initiator_opinion, response.recipient_opinion)
        ]
        if self.purpose == "food_request":
            self.record_food_request()
        finish_generated_dialogue(self, reading=reading)

    @property
    def accepts_response(self):
        world = self.initiator.world
        return (self.active and self.dialogue_state in ("waiting", "pending")
                and world.npc_by_id(self.initiator.id) is self.initiator
                and world.npc_by_id(self.recipient.id) is self.recipient
                and self.initiator.location == self.recipient.location)

    def fail_dialogue(self, reason, *, failed_turns=()):
        if not self.accepts_response:
            return
        record_failed_dialogue(self, failed_turns)
        self.dialogue_state = "failed"
        self.record.error = reason
        self.finish(self.initiator.world.tree)

    def prepare_food_request(self):
        from autonomy import donor_choice

        self.food_outcome, self.food_reason = donor_choice(self.recipient, self.initiator)

    def record_food_request(self):
        requester, donor = self.initiator, self.recipient
        world = requester.world
        world.publish("food_requested", (requester, donor),
                      f"{requester.name} asked {donor.name} for an apple.", self.record)
        if self.food_outcome == "accepted":
            self.record.outcomes.insert(0, "Agreed to share one apple; the handoff has not happened yet.")
            return
        deliberate = self.food_outcome in ("need", "refused")
        text = (f"{donor.name} kept their last apple because they wanted to eat it."
                if self.food_outcome == "need" else f"{donor.name} declined {requester.name}'s apple request.")
        if not deliberate:
            text = f"{donor.name} could no longer share with {requester.name}: the inventories changed."
        event = world.publish("food_refused" if deliberate else "interaction_failed", (requester, donor), text, self.record)
        self.outcome_event_id = event.id
        self.record.outcomes.insert(0, self.food_reason)
        if deliberate:
            remember(requester, donor, event, "refused", f"{donor.name} declined my apple request: {self.food_reason}",
                     -1 if self.food_outcome == "need" else -5)
            remember(donor, requester, event, "kept_fruit", f"I kept my apple when {requester.name} asked: {self.food_reason}", 0)

    def update(self, dt, *, reading=False):
        if not self.active:
            return
        if self.initiator.location != self.recipient.location:
            self.fail_dialogue("The participants are no longer in the same location.")
            return
        if self.arrived:
            face_each_other(self.initiator, self.recipient)
            if self.dialogue_state in ("waiting", "pending"):
                self.wait_elapsed += dt
                if self.wait_elapsed >= DIALOGUE_TIMEOUT_SECONDS:
                    self.fail_dialogue("Dialogue generation timed out. No dialogue was generated.")
            elif dialogue_finished(self, dt, reading=reading):
                if self.accepted_gift:
                    gift = self.accepted_gift
                    giver = self.initiator if gift.giver_id == self.initiator.id else self.recipient
                    recipient = self.other(giver)
                    states = self.saved_states if giver is self.initiator else self.saved_states[::-1]
                    self.active = False
                    self.initiator.conversation = self.recipient.conversation = None
                    Gift(giver, recipient, gift.item, gift.quantity, self.record, saved_states=states,
                         already_met=True, thanks=self.handoff_thanks, rewritten_thanks=self.rewritten_handoff_thanks)
                elif self.purpose == "food_request" and self.food_outcome == "accepted":
                    self.handoff()
                else:
                    self.finish(self.initiator.world.tree)
            return
        self.arrived = True
        for npc in (self.initiator, self.recipient):
            if npc.destination is not None:
                if npc._move(max(0.0, dt)):
                    npc.destination = None
                else:
                    self.arrived = False
        if self.arrived:
            self.record.status = "Waiting for dialogue"
            face_each_other(self.initiator, self.recipient)
            if self.purpose == "food_request":
                self.prepare_food_request()

    def snapshot(self, tree, npcs):
        self.request_roster_revision = self.initiator.world.roster_revision
        self.request_actor_revisions = {n.id: n.command_revision for n in (self.initiator, self.recipient)}
        def participant(npc):
            involved = npc is self.initiator or npc is self.recipient
            state = npc.snapshot(tree, self.other(npc).id) if involved else npc.snapshot(tree, private=False)
            # These replies were addressed to the player before the NPCs met.
            state.pop("last_reply", None)
            return state

        if self.dialogue_state == "waiting":
            self.proposals = permitted_proposals(self)
            self.context_references = conversation_references(self)
            self.information_context = prepare_information(self)
            self.name_actors = actor_catalog(self.initiator.world, self.initiator.location)
        views = information_views(self.information_context, self.initiator.world)
        initiator, recipient = participant(self.initiator), participant(self.recipient)
        for role, state in (("initiator", initiator), ("recipient", recipient)):
            state["private_context"].pop("known_accounts", None)
            state["private_context"]["permitted_accounts"] = views.get(role, [])
            concern = self.concerns[role]
            state["private_context"]["concern"] = ({"kind": concern.kind, "claim_id": concern.claim_id,
                "guidance": concern_topic(self.initiator if role == "initiator" else self.recipient, concern)}
                if concern else None)
        initiator["private_context"]["discussion_focus_claim_id"] = self.focus_claim_id
        initiator["private_context"]["immediate_objective"] = self.topic
        recipient["private_context"]["immediate_objective"] = "Respond to what is actually said using only my own knowledge."
        return {
            "request_kind": "conversation",
            "_name_actors": self.name_actors,
            "interaction_id": self.id,
            "origin": self.origin,
            "purpose": self.purpose,
            "opening_kind": self.opening_kind,
            "arrival_focus": self.opening_kind == "orientation",
            "food_request": {"item": "fruit", "quantity": 1, "outcome": self.food_outcome,
                             "reason": self.food_reason, "transferred": False} if self.purpose == "food_request" else None,
            "initiator": initiator,
            "recipient": recipient,
            "shared_history": shared_conversation_history(self),
            "_information_validation": asdict(self.information_context) if self.information_context else None,
            "topic": "The initiator will introduce their own concern or account; it is not shared knowledge yet.",
            "permitted_proposals": [dict(
                asdict(p), item="fruit", quantity=1, deadline_label=phase_label(p.deadline),
                giver_name=self.initiator.world.npc_by_id(p.promisor_id).name,
                receiver_name=self.initiator.world.npc_by_id(p.recipient_id).name,
                offer_speaker="initiator" if p.promisor_id == self.initiator.id else "recipient",
                acceptance_speaker="initiator" if p.recipient_id == self.initiator.id else "recipient",
            ) for p in self.proposals],
            "nearby_npcs": [participant(npc) for npc in self.initiator.world.listeners(self.initiator) if npc is not self.initiator and npc is not self.recipient],
        }

    def finish(self, tree):
        if not self.active:
            return
        self.active = False
        self.record.status = "Finished"
        if not self.turns:
            self.record.status = "Failed" if self.error else "Ended before dialogue"
            self.record.outcomes = [self.error or "The conversation ended before any dialogue was generated."]
        if self.purpose != "food_request":
            world = self.initiator.world
            text = (f"{self.initiator.name} and {self.recipient.name} talked about {self.topic}."
                    if self.turns else f"{self.initiator.name} and {self.recipient.name}'s meeting ended without generated dialogue.")
            event = world.publish("conversation_completed" if self.turns else "interaction_failed",
                                  (self.initiator, self.recipient), text, self.record)
            self.outcome_event_id = event.id
            if self.dialogue_state == "ready":
                for npc, reaction in zip((self.initiator, self.recipient), self.reactions):
                    reason, weight = reaction
                    remember(npc, self.other(npc), event, "conversation", f"Talked about {self.topic}. {reason}", weight)
        elif not self.turns or self.food_outcome in ("", "accepted"):
            if not self.error:
                self.record.status = "Ended before sharing"
                self.record.outcomes = ["The food request ended before an apple was shared."]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient),
                                                self.record.outcomes[0], self.record)
            self.outcome_event_id = event.id
        for npc, state in zip((self.initiator, self.recipient), self.saved_states):
            if npc.conversation is not self:
                continue
            npc.conversation = None
            if npc.world.npc_by_id(npc.id) is not npc:
                continue
            npc.restore_after_interaction(state, self.other(npc).id)
            npc.pause_after_conversation(self.social_effects_applied)
            if not self.social_effects_applied:
                npc.cooldowns[self.other(npc).id] = dialogue_failure_cooldown(npc, self.other(npc))

    def handoff(self):
        """Reverse the roles without restoring activities or reacquiring tree claims."""
        requester, donor = self.initiator, self.recipient
        self.active = False
        requester.conversation = donor.conversation = None
        Gift(donor, requester, "fruit", 1, self.record,
             saved_states=[self.saved_states[1], self.saved_states[0]], already_met=True,
             thanks=self.handoff_thanks, rewritten_thanks=self.rewritten_handoff_thanks)


class Gift:
    """Reserve both participants through approach, giving, and thanking."""

    def __init__(self, initiator, recipient, item, quantity, record=None, spacing=0, *, saved_states=None, already_met=False, thanks=None, rewritten_thanks=()):
        destination = (initiator.x, initiator.y) if already_met else company_point(initiator, recipient, spacing)
        if destination is None:
            raise ValueError("There isn't enough room here to give the items.")
        self.initiator, self.recipient = initiator, recipient
        self.item, self.quantity = item, quantity
        self.arrived = already_met
        self.active = True
        self.transferred = False
        self.outcome_event_id = None
        self.thanks = list(thanks) if thanks is not None else []
        self.rewritten_thanks = list(rewritten_thanks)
        self.elapsed = 0.0
        self.record = record or ConversationRecord((initiator.name, recipient.name), "Give items", "Walking to give",
                                                  participant_ids=[initiator.id, recipient.id])
        if set(self.record.participants) != {initiator.name, recipient.name}:
            self.record = ConversationRecord((initiator.name, recipient.name), "Directed gift", "Walking to give",
                authorization_id=record.id, participant_ids=[initiator.id, recipient.id])
            initiator.world.history.append(self.record)
        self.record.transfer_intent = dict(giver_id=initiator.id, recipient_id=recipient.id, item=item, quantity=quantity)
        self.saved_states = []
        self.record.status = "Giving items" if already_met else "Walking to give"
        for npc in (initiator, recipient):
            self.saved_states.append(npc.suspend_activity("Giving items" if npc is initiator else "Receiving items"))
            npc.gift = self
        if saved_states is not None:
            self.saved_states = saved_states
        initiator.destination = None if already_met else destination

    def update(self, dt):
        if not self.active:
            return
        giver, recipient = self.initiator, self.recipient
        if giver.location != recipient.location or any(giver.world.npc_by_id(p.id) is not p for p in (giver, recipient)):
            self.finish("A participant is no longer outdoors.")
            return
        if not self.arrived:
            self.arrived = giver._move(max(0.0, dt))
            if self.arrived:
                giver.destination = None
                self.record.status = "Giving items"
                face_each_other(giver, recipient)
            return
        elapsed = self.elapsed + max(0.0, dt)
        if not self.transferred and elapsed + 1e-9 >= GIFT_GIVE_SECONDS:
            from player import commit_transfer
            try:
                commit_transfer(giver.world, self)
            except ValueError as exc:
                self.finish(str(exc))
                return
            if self.thanks:
                self.record.add_turn(recipient.name, " ".join(part.text.strip() for part in self.thanks),
                                     [part.model_dump() for part in self.thanks], self.rewritten_thanks)
            self.record.outcomes[0:1] = [f"Gave {self.quantity} {self.item} to {recipient.name}."]
            self.record.status = "Thanking"
            elapsed = max(GIFT_GIVE_SECONDS, elapsed)
        self.elapsed = elapsed
        if self.elapsed + 1e-9 >= GIFT_SECONDS:
            self.finish()

    def finish(self, error=""):
        if not self.active:
            return
        self.active = False
        self.record.status = "Failed" if error and not self.transferred else "Completed"
        self.record.error = error if not self.transferred else ""
        if error and self.transferred:
            self.record.outcomes.append(f"The transfer remains committed. {error}")
        elif error:
            self.record.outcomes[0:1] = [error]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient), error, self.record)
            self.outcome_event_id = event.id
        for npc, state in zip((self.initiator, self.recipient), self.saved_states):
            other = self.recipient if npc is self.initiator else self.initiator
            if npc.gift is not self:
                continue
            npc.gift = None
            if npc.world.npc_by_id(npc.id) is not npc:
                continue
            npc.restore_after_interaction(state, other.id, preserve_facing=True)
            if self.record.purpose == "food_request":
                npc.pause_after_conversation(True)
