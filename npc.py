import math
import random
from typing import Annotated, Literal
from pydantic import Field
from social_records import ConversationConcern, Goal, OpinionEffect, Belief
from social import check_relationship_goals, private_context
from interactions import PlayerReplyProgress, company_point, face_toward, move_toward, suspended_activity
from dataclasses import asdict, dataclass, field
from activity import ActivitySnapshot, Behavior
from uuid import uuid4
from character_data import DIRECTIONS, PROFILE_FIELDS, normalize_opinions
from creations import CreativeJob
from emotions import portrait_expression
from conversation_history import Memory, relevant_memories
from constants import (MAX_RESIDENTS, PLAYER_CALL_COOLDOWN, REST_SECONDS_RANGE, CREATION_WORK_SECONDS_RANGE,
                       OUTDOOR_COOLDOWN_SECONDS, BEDTIME_DELAY_RANGE, WAKE_DELAY_RANGE, APPLE_SATISFACTION,
                       COMPANY_MAX_SECONDS, DECISION_SECONDS, DEFAULT_CHARACTER_NAME, EAT_SECONDS,
                       CONVERSATION_PAUSE_RANGE, CONVERSATION_RETURN_PAUSE, FAILURE_COOLDOWN, GIFT_GIVE_SECONDS,
                       MOVEMENT_SPEED, NPC_RADIUS, PARTNER_COOLDOWN, SNAKE_NAME, SOCIAL_RADIUS, SOCIAL_RECOVERED,
                       WANDER_PAUSE, FOOT_MARGIN, STARTING_DAY, STARTING_NPC_FRUIT, STARTING_NPC_GENEROSITY_RANGE,
                       STARTING_NPC_POSITION, STARTING_NPC_SATISFACTION_RANGE, STARTING_NPC_SOCIABILITY_RANGE,
                       STARTING_TREE_POSITION, STARTING_FATIGUE_RANGE, EXCHANGE_FATIGUE)
from information import NPC_ARRIVAL_MEMORY
from actor_identity import actor_catalog
from transfers import Handoff
from locations import Journey
from player_records import PlayerOpinion, PlayerCall, PlayerReplyCall


@dataclass
class Tree:
    x: float = STARTING_TREE_POSITION[0]
    y: float = STARTING_TREE_POSITION[1]
    id: str = "fruit-tree"
    fruit: int = 0
    shake_remaining: float = field(default=0.0, repr=False, compare=False)

    @property
    def interaction_point(self):
        return (self.x - 60.0, self.y + 50.0)



@dataclass
class ResidentValues(ActivitySnapshot):
    """The resident's durable values, inherited directly by the live actor."""
    character: dict
    x: float = STARTING_NPC_POSITION[0]
    y: float = STARTING_NPC_POSITION[1]
    sprite_size: tuple[int, int] | None = None
    reply: str = ""
    rewritten_reply: str = ""
    display_name: str | None = None
    emotion: str = "neutral"
    emotion_intensity: int = 0
    id: Annotated[str, Field(min_length=1)] = field(default_factory=lambda: uuid4().hex)
    social: Annotated[float, Field(ge=0, le=100)] = field(default_factory=lambda: random.uniform(*STARTING_NPC_SATISFACTION_RANGE))
    inventory: dict[Literal["fruit"], Annotated[int, Field(strict=True, ge=0)]] = field(default_factory=lambda: {"fruit": STARTING_NPC_FRUIT})
    preferences: dict = field(default_factory=lambda: {
        "sociability": random.uniform(*STARTING_NPC_SOCIABILITY_RANGE),
        "generosity": random.uniform(*STARTING_NPC_GENEROSITY_RANGE),
    })
    memories: Annotated[list[Memory], Field(max_length=20)] = field(default_factory=list)
    goal: Goal | None = None
    opinions: dict[str, dict] = field(default_factory=dict)
    pending_social_opinions: list[OpinionEffect] = field(default_factory=list)
    beliefs: list[Belief] = field(default_factory=list)
    current_concern: ConversationConcern | None = None
    conversation_pause_remaining: Annotated[float, Field(ge=0, le=max(CONVERSATION_PAUSE_RANGE[1], FAILURE_COOLDOWN))] = 0.0
    conversation_return_remaining: Annotated[float, Field(ge=0, le=CONVERSATION_RETURN_PAUSE)] = 0.0
    player_opinion: PlayerOpinion = field(default_factory=PlayerOpinion)
    player_call: PlayerCall | PlayerReplyCall | None = None
    player_call_cooldown: Annotated[float, Field(ge=0, le=PLAYER_CALL_COOLDOWN)] = 0.0
    resolved_player_calls: list[str] = field(default_factory=list)
    command_revision: Annotated[int, Field(ge=0)] = 0
    arrival_day: Annotated[int, Field(ge=1)] = STARTING_DAY
    departure: dict | None = None

    room_number: Annotated[int, Field(strict=True, ge=1, le=MAX_RESIDENTS)] = 0
    social_fatigue: Annotated[float, Field(ge=0, le=100)] = field(default_factory=lambda: random.uniform(*STARTING_FATIGUE_RANGE))
    home_remaining: Annotated[float, Field(ge=0, le=max(REST_SECONDS_RANGE[1], CREATION_WORK_SECONDS_RANGE[1]))] = 0.0
    outdoor_remaining: Annotated[float, Field(ge=0, le=OUTDOOR_COOLDOWN_SECONDS)] = 0.0
    bedtime_remaining: Annotated[float | None, Field(ge=0, le=BEDTIME_DELAY_RANGE[1])] = None
    wake_remaining: Annotated[float | None, Field(ge=0, le=WAKE_DELAY_RANGE[1])] = None
    home_reason: Literal["night", "retreat"] | None = None
    last_creative_day: Annotated[int, Field(strict=True, ge=0)] = 0
    creative_job: CreativeJob | None = field(default=None, repr=False, compare=False)
    location: str = "garden"
    journey: Journey | None = None


class ConversationActor:
    """Shared participant state; each actor owns its movement and capabilities."""

    speech_only = False  # True: may talk and share accounts, but cannot carry out tasks, gifts or invitations

    @property
    def alive(self):
        return self.departure is None

    def activity_state(self):
        return ActivitySnapshot.capture(self)

    def restore_activity(self, state):
        ActivitySnapshot.restore(self, state)

    def suspend_activity(self, reason):
        """Save progress and release physical claims before an interaction."""
        state = self.activity_state()
        if not self.creative_job:
            self.command_revision += 1
        self.behavior, self.owner = Behavior.WAIT, "autonomous"
        self.destination = self.target_id = None
        self.activity_elapsed = 0.0
        self.eating_witness_ids = []
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

    def update_cooldowns(self, dt):
        self.decision_remaining = max(0.0, self.decision_remaining - dt)
        if self.interaction is None and not self.interaction_blocked and not self.player_reply.closing:
            self.conversation_pause_remaining = max(0.0, self.conversation_pause_remaining - dt)
            self.conversation_return_remaining = max(0.0, self.conversation_return_remaining - dt)
            if self.current_concern:
                self.current_concern.remaining = max(0.0, self.current_concern.remaining - dt)
        self.cooldowns = {key: value - dt for key, value in self.cooldowns.items() if value > dt}
        interaction = self.conversation or self.gift
        if interaction is not None:
            suspended = suspended_activity(self.gift or interaction, self)
            suspended.decision_remaining = max(0.0, suspended.decision_remaining - dt)
            suspended.cooldowns = {key: value - dt for key, value in suspended.cooldowns.items() if value > dt}

    def can_reach(self, point):
        from locations import clamp_point
        return tuple(point) == clamp_point(self.location, point)

    def named_opinions(self):
        actors = {actor.id: actor.name for actor in self.world.actors} if self.world else {}
        bound_names = {actors[key] for key in self.opinions if key in actors}
        result = {name: entry.copy() for name, entry in self.character.get("Opinions", {}).items()
                  if name not in bound_names}
        result.update({actors[key]: entry.copy() for key, entry in self.opinions.items() if key in actors})
        return result

    def opinion_score(self, actor_id, default=0):
        if self.world and actor_id == self.world.player.id:
            return self.player_opinion.opinion
        return self.opinions.get(actor_id, {}).get("opinion", default)

    def remember_opinion(self, actor_id, opinion, reason):
        self.remember_opinions({actor_id: {"opinion": opinion, "reason": reason}})

    def remember_opinions(self, updates):
        updates = normalize_opinions(updates)
        for actor_id, entry in updates.items():
            target = self.world.participant_by_id(actor_id) if self.world else None
            if target is None or target is self or target is self.world.player:
                raise ValueError("The opinion target is no longer available.")
            outcomes = [p.outcome for p in self.world.promises if p.status != "pending"
                        and {p.promisor_id, p.recipient_id} == {self.id, target.id}]
            if outcomes and outcomes[-1] not in entry["reason"]:
                entry["reason"] = (outcomes[-1] + " Current impression: " + entry["reason"])[:400]
        if all(self.opinions.get(key) == value for key, value in updates.items()):
            return
        self.opinions.update(updates)
        interaction = self.conversation or self.gift
        check_relationship_goals(self.world, self, interaction.record if interaction else None)


@dataclass
class NPC(ResidentValues, ConversationActor):
    rng: random.Random = field(default_factory=random.Random, repr=False)
    selected: bool = False
    player_reply: PlayerReplyProgress = field(default_factory=PlayerReplyProgress, repr=False, compare=False)
    world: object = field(default=None, repr=False, compare=False)

    @property
    def indoors(self):
        return self.location != "garden"

    @property
    def conversation(self):
        return self.world.reservations.get(self.id, "conversation") if self.world else None

    @property
    def gift(self):
        return self.world.reservations.get(self.id, "handoff") if self.world else None

    @property
    def interaction(self):
        return self.world.reservations.exchange(self.id) if self.world else None

    @property
    def busy(self):
        return self.interaction is not None or self.eating

    @property
    def unread_player_reply(self):
        call = self.player_call
        return call if call and call.kind == "reply" else None

    @property
    def generation_pending(self):
        return self.player_reply.generating

    @property
    def action_pending(self):
        return bool(self.world and self.world.action_for(self.id))

    @property
    def interaction_blocked(self):
        return self.generation_pending or self.action_pending

    @property
    def waiting_for_done(self):
        return self.action_pending and self.player_reply.closing and self.selected


    def finish_activity(self, world, failed=False):
        target = self.target_id
        if target is not None and (failed or self.behavior == Behavior.COMPANY):
            self.cooldowns[target] = FAILURE_COOLDOWN if failed else PARTNER_COOLDOWN
        self.behavior = Behavior.WAIT
        self.destination = None
        self.target_id = None
        self.activity_elapsed = 0.0
        self.eating_witness_ids = []
        self.resume_pending = False
        self.owner = "autonomous"
        self.decision_remaining = 0.0
        self.reason = "Target unavailable; choosing another activity" if failed else "Finished; choosing what to do next"

    def begin_activity(self, behavior, world, target=None, reason="", owner="autonomous"):
        if behavior != Behavior.GO_HOME:
            self.home_reason = None
            self.bedtime_remaining = None
        self.behavior, self.target_id = behavior, target
        self.owner, self.reason = owner, reason
        self.destination = None
        self.activity_elapsed = self.pause_remaining = 0.0
        self.eating_witness_ids = []
        self.resume_pending = False
        self.decision_remaining = DECISION_SECONDS
        if behavior == Behavior.EAT:
            self.facing = 4
        elif behavior == Behavior.GO_TO_TREE:
            if self.location != "garden":
                from locations import start_journey
                start_journey(self, "garden", world.tree.interaction_point, purpose="tree", fast=False)
                return
            self.destination = world.tree.interaction_point
        elif behavior == Behavior.COMPANY:
            partner = world.npc_by_id(target)
            self.destination = company_point(self, partner) if partner is not None else None
            if self.destination is None:
                self.finish_activity(world, failed=True)

    def resume_activity(self, world):
        """Resume suspended activity on a simulation tick, preserving committed progress."""
        self.resume_pending = False
        if self.behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE):
            self.begin_activity(self.behavior, world, world.tree.id, self.reason, self.owner)
        elif self.behavior == Behavior.COMPANY:
            partner = world.npc_by_id(self.target_id)
            if partner is None or not world.company_available(partner):
                self.finish_activity(world, failed=True)
            else:
                self.destination = company_point(self, partner)


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
        return (self.selected or self.interaction_blocked) and self.behavior == Behavior.WANDER

    @property
    def decision_reason(self):
        if self.conversation is None and self.owner == "autonomous":
            if self.interaction_blocked:
                return f"{self.reason}. Waiting for the player request."
            if self.selected:
                return f"{self.reason}. New decisions wait while selected."
        return self.reason

    @property
    def status(self):
        if self.departure:
            from dialogue_rules import named_references
            return f"Departed on day {self.departure['day']} ({named_references(self.departure['cause'])})"
        if self.world:
            from player import owns_resident
            if owns_resident(self.world, self):
                return self.reason
        if self.journey:
            return f"Travelling to {self.journey.destination}"
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
        if self.resume_pending:
            return "Resuming activity on next simulation tick"
        if self.wandering_paused:
            return "Waiting for player"
        return {
            Behavior.WANDER: "Wandering",
            Behavior.GO_TO_TREE: "Walking to tree",
            Behavior.HARVEST: "Collecting apples at the tree",
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
            if self.behavior == Behavior.SLEEP:
                width, height = height, width
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
            "actor_id": self.id,
            "name": self.name,
            "character": {key: value for key, value in self.character.items() if key != "Opinions"},
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
            "day": self.world.day if self.world else 1,
            "living_count": self.world.population if self.world else 1,
            "time_of_day": self.world.phase if self.world else "day",
            "last_reply": self.reply,
        }
        if private:
            from area_knowledge import area_knowledge
            state["area_knowledge"] = area_knowledge(self, partner_id)
            from player import work_choices
            from player_story import starting_encounter, starting_dialogue_context
            state["owned_works"] = work_choices(self.world, self.id)
            state["player_relationship"] = {"actor_id": self.world.player.id,
                "opinion": asdict(self.player_opinion)}
            state["memories"] = [{"partner_id": memory.partner_id, "kind": memory.kind,
                          "summary": memory.summary, "day": memory.day, "phase": memory.phase}
                                 for memory in relevant_memories(self, partner_id)]
            state["private_context"] = private_context(self, partner_id)
            starting_experience = starting_dialogue_context(self.world, self.id)
            if starting_experience:
                state["private_context"]["starting_experience"] = starting_experience
            state["private_context"]["opinions"] = self.named_opinions()
            # Validation only: speech claiming a harvest needs one this resident actually made.
            state["_harvested"] = any(f.kind == "harvest" and f.subject_id == self.id for f in self.world.facts)
            encounters = [record for record in self.world.history if record.includes(self) and record.turns]
            state["private_context"]["situation"] = {
                "days_here": self.world.day - self.arrival_day,
                "arrival_day": self.arrival_day,
                "transition_memory": NPC_ARRIVAL_MEMORY, "explanation": "unknown", "way_home": "unknown",
                "prior_conversations": len(encounters),
                "prior_snake_encounters": (sum(self.world.snake.id in record.participant_ids for record in encounters)
                                           + int(starting_encounter(self.world, self.id) is not None)),
                "known_danger": [account for account in state["private_context"]["known_accounts"]
                                 if account["kind"] == "killing"][:3],
            }
        else:
            state.pop("_name_actors", None)
            state["character"].pop("Opinions", None)
            for profile_field in PROFILE_FIELDS:
                state["character"].pop(profile_field, None)
            for key in ("last_reply", "activity_reason"):
                state.pop(key, None)
            # A visible person is not an invitation to inspect their task, travel
            # destination, biography, satisfaction, or player control state.
            state = {key: state[key] for key in ("location", "actor_id", "name", "character", "position", "facing")}
            state["activity"] = "Moving" if self.journey or self.destination else "Eating an apple" if self.eating else "Present"
            return state
        state["arrival_day"] = self.arrival_day
        if self.indoors:
            for key in [key for key in state if key.startswith("tree_")]:
                state.pop(key)
        return state

    def bind_authored_opinions(self):
        """Bind exact assigned labels once; unresolved authored names stay visible."""
        if self.world is None:
            return
        unresolved = {}
        for person, entry in self.character.get("Opinions", {}).items():
            targets = [actor for actor in self.world.actors if actor.name == person and actor is not self]
            if len(targets) == 1:
                self.opinions.setdefault(targets[0].id, entry.copy())
            else:
                unresolved[person] = entry.copy()
        self.character = {**self.character, "Opinions": unresolved}


    def _face_when_still(self):
        point = self.world.tree.interaction_point if self.world else None
        at_tree = point is not None and math.hypot(self.x - point[0], self.y - point[1]) <= 1e-6
        self.facing = 2 if not self.eating and at_tree else 4

    def set_selected(self, selected):
        """Defer new decisions while allowing the current finite activity to finish."""
        self.selected = selected
        if not selected:
            self.player_reply.closing = False
        if self.conversation is None and self.gift is None and selected and self.wandering_paused:
            self._face_when_still()

    def apply_action(self, action, tree):
        from locations import available
        if self.world is None or self.world.npc_by_id(self.id) is not self or not available(self):
            raise ValueError("Only living residents can act in this world.")
        # Reject unsupported actions before changing anything.
        if action == "SAY":
            return
        if action not in ("GO_TO_TREE", "EAT"):
            raise ValueError("Unsupported NPC action")
        if self.busy:
            raise ValueError("This NPC is already in an interaction.")
        if action == "EAT" and self.inventory.get("fruit", 0) < 1:
            raise ValueError("I don't have an apple to eat.")
        self.world.interrupt_resident(self, "The player assigned a new activity.", mode="action")
        self.command_revision += 1
        if action == "EAT":
            self.begin_activity(Behavior.EAT, self.world, reason="Player asked me to eat an apple", owner="player_task")
        else:
            self.begin_activity(Behavior.GO_TO_TREE, self.world, tree.id, "Player asked me to visit the tree", owner="player_task")

    def check_gift(self, recipient, item, quantity, *, player_authorization=None):
        from transfers import validate_transfer
        validate_transfer(self.world, self, recipient, item, quantity, player_authorization=player_authorization)
        if recipient is self.world.player:
            return
        if (recipient is self or self.world is None or self.world.npc_by_id(self.id) is not self
                or not any(person is recipient for person in self.world.npcs)):
            raise ValueError("Choose another NPC in this world to receive the items.")
        from locations import available as available_outdoors
        if not available_outdoors(self) or not available_outdoors(recipient):
            raise ValueError("Both residents must be available outdoors to exchange items.")
        if self.busy or recipient.busy or recipient.interaction_blocked:
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
        return Handoff(self, recipient, item, quantity, record, spacing)

    def _move(self, dt):
        return move_toward(self, self.destination, dt, MOVEMENT_SPEED)

    def update(self, dt, tree):
        from player import owns_resident
        from autonomy import awaiting_arrival_greeting
        if (self.owner == "autonomous" and self.behavior in (Behavior.WANDER, Behavior.WAIT)
                and not self.journey and awaiting_arrival_greeting(self, self.world) and self.interaction is None):
            return
        if owns_resident(self.world, self):
            return
        if self.journey:
            from locations import update_journey
            if update_journey(self, dt) and self.behavior in (Behavior.GO_TO_TREE, Behavior.HARVEST):
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
                    from locations import clamp_point
                    self.destination = clamp_point(self.location, (self.rng.uniform(left, right),
                                                                  self.rng.uniform(max(FOOT_MARGIN, top), bottom)))
                if self._move(movement_dt):
                    self.destination = None
                    self.pause_remaining = WANDER_PAUSE
        elif behavior == Behavior.GO_TO_TREE:
            if self.destination is None:
                self.destination = tree.interaction_point
            if self._move(dt):
                self.finish_activity(world)
        elif behavior == Behavior.HARVEST:
            try:
                world.harvest(self, 1)
            except ValueError:
                self.finish_activity(world, failed=True)
                return
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
            from interactions import collect_eating_witnesses
            collect_eating_witnesses(world, self)
            self.facing = 4
            self.activity_elapsed += dt
            if self.activity_elapsed + 1e-9 >= EAT_SECONDS:
                if self.inventory.get("fruit", 0) > 0:
                    self.inventory["fruit"] -= 1
                    self.social = min(100.0, self.social + APPLE_SATISFACTION)
                    world.observed_event("apple_eaten", self, None, f"{self.name} ate an apple.",
                        witnesses=[world.participant_by_id(actor_id) for actor_id in self.eating_witness_ids])
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
    from constants import PLAYER_LABEL, PLAYER_NAME_TAG
    used = {name.casefold() for name in used_names} | {"python", SNAKE_NAME.casefold(), PLAYER_LABEL.casefold(), PLAYER_NAME_TAG.casefold()}
    for npc in npcs:
        base = npc.character["Name"].strip() or DEFAULT_CHARACTER_NAME
        label, number = base, 2
        while label.casefold() in used:
            label = f"{base} ({number})"
            number += 1
        npc.display_name = label
        used.add(label.casefold())
