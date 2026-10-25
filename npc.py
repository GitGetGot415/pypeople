import math
import random
from copy import deepcopy
from typing import Annotated, Literal
from pydantic import Field
from dialogue import (AcceptedProposal, ConversationPart, DisclosureChoice, LeverageUse, ResidentGift,
                     ThirdPartyOpinion, EscalationChoice)
from social_records import ConcernKind, ConversationConcern, Goal, OpinionEffect, Belief, phase_label
from information import information_views, prepare_information
from social import (check_relationship_goals, commit_agreements, commit_concern_use, commit_information,
    concern_description, concern_topic, conversation_references, dialogue_failure_cooldown,
    permitted_proposals, private_context, validate_live_agreements)
from interactions import (DialogueProgress, PlayerReplyProgress, company_point, conversation_points,
                         dialogue_finished, face_each_other, face_toward, finish_generated_dialogue,
                         move_toward, record_failed_dialogue, record_generated_turns,
                         release_resident, suspended_activity)
from dataclasses import asdict, dataclass, field
from activity import ActivitySnapshot, Behavior
from uuid import uuid4
from character_data import DIRECTIONS, PROFILE_FIELDS, normalize_opinions
from creations import CreativeJob
from emotions import portrait_expression
from conversation_history import (ConversationRecord, Memory, describe_opinion_change, relevant_memories, remember,
                                 shared_conversation_history)
from constants import (MAX_RESIDENTS, PLAYER_CALL_COOLDOWN, REST_SECONDS_RANGE,
    CREATION_WORK_SECONDS_RANGE, OUTDOOR_COOLDOWN_SECONDS, BEDTIME_DELAY_RANGE, WAKE_DELAY_RANGE,
    APPLE_SATISFACTION, COMPANY_MAX_SECONDS, DECISION_SECONDS, DEFAULT_CHARACTER_NAME,
    DIALOGUE_TIMEOUT_SECONDS, EAT_SECONDS, CONVERSATION_PAUSE_RANGE, CONVERSATION_RETURN_PAUSE,
    FAILURE_COOLDOWN, GIFT_GIVE_SECONDS, MOVEMENT_SPEED, NPC_RADIUS, PARTNER_COOLDOWN, SNAKE_NAME,
    SOCIAL_RADIUS, SOCIAL_RECOVERED, WANDER_PAUSE, FOOT_MARGIN, STARTING_DAY, STARTING_NPC_FRUIT,
    JUDY_ROOM_NUMBER, STARTING_NPC_GENEROSITY_RANGE, STARTING_NPC_POSITION,
    STARTING_NPC_SATISFACTION_RANGE, STARTING_NPC_SOCIABILITY_RANGE, STARTING_TREE_POSITION,
    STARTING_FATIGUE_RANGE, EXCHANGE_FATIGUE, CONVERSATION_CANCELLED, CONVERSATION_DIALOGUE_READY,
    CONVERSATION_ENDED_BEFORE_DIALOGUE, CONVERSATION_ENDED_BEFORE_SHARING, CONVERSATION_FAILED,
    CONVERSATION_FINISHED, CONVERSATION_WAITING_FOR_DIALOGUE, CONVERSATION_WALKING_TO_MEET, CONVERSATION_OPINION_STEP)
from information import NPC_ARRIVAL_MEMORY, validate_orientation
from actor_identity import (actor_catalog, name_forms)
from transfers import Handoff
from transfer_records import TransferIntent
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
    def last_killing_phase(self):
        from social_records import phase_index
        return max((phase_index(n.departure["day"], n.departure["phase"])
                    for n in self.world.archived if n.departure["attacker_id"] == self.id), default=None)

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


@dataclass(eq=False)
class ResidentProgress:
    """Durable domain progress; live actors and suspended activities attach separately."""
    accepted_gift: ResidentGift | None
    id: Annotated[str, Field(min_length=1)]
    origin: Literal['player', 'autonomous']
    purpose: Literal['chat', 'food_request', 'confrontation']
    opening_kind: ConcernKind | Literal["orientation"] | None  # Orientation is only Judy's first welcome.
    goal_id: str | None
    focus_claim_id: str | None
    food_outcome: Literal['', 'accepted', 'need', 'refused', 'unavailable']
    food_reason: str
    handoff_thanks: Annotated[list[ConversationPart], Field(max_length=3)]
    rewritten_handoff_thanks: list[str]
    outcome_event_id: str | None
    reactions: Annotated[list[tuple[str, float]], Field(max_length=2)]
    topic: str
    arrived: bool
    accepted_proposals: Annotated[list[AcceptedProposal], Field(max_length=2)]
    social_effects_applied: bool
    context_references: dict[str, Literal['active', 'completed', 'failed', 'pending', 'fulfilled', 'broken', 'cancelled']]
    accepted_disclosures: Annotated[list[DisclosureChoice], Field(max_length=4)]
    third_party_opinions: Annotated[list[ThirdPartyOpinion], Field(max_length=4)]
    leverage_uses: Annotated[list[LeverageUse], Field(max_length=2)]
    escalation: EscalationChoice | None
    accepted_revisions: dict[str, Annotated[int, Field(strict=True, ge=0)]]


class Conversation(ResidentProgress):
    """Temporarily reserve two NPCs, then restore their interrupted activities."""

    def __init__(self, initiator, recipient, topic, tree, spacing=0, *, origin="player", purpose="chat",
                 opening_kind=None, focus_claim_id=None, goal_id=None, _restored=None):
        if _restored is not None:
            self._restore(initiator, recipient, _restored)
            return
        from locations import available as available_outdoors, colocated
        from player import owns_resident
        if (initiator is recipient or initiator.busy or recipient.busy
                or owns_resident(initiator.world, initiator) or owns_resident(initiator.world, recipient)
                or initiator.world.visit and initiator.world.visit.host_id in (initiator.id, recipient.id)
                or initiator.player_call or recipient.player_call or recipient.selected or recipient.interaction_blocked
                or not available_outdoors(initiator) or not available_outdoors(recipient)
                or not colocated(initiator, recipient)):
            raise ValueError("Choose another NPC who is free to talk.")
        if purpose not in ("chat", "food_request", "confrontation"):
            raise ValueError("Resident meetings use chat or food_request.")
        if purpose == "confrontation":
            from social import grievance_against
            if (not isinstance(initiator, NPC) or not isinstance(recipient, NPC)
                    or not focus_claim_id or grievance_against(initiator, recipient) != focus_claim_id):
                raise ValueError("A confrontation requires two human residents and a specific retained, unresolved grievance.")
        if focus_claim_id and not any(b.claim_id == focus_claim_id for b in initiator.beliefs):
            raise ValueError("The initiator does not know the required account.")
        self.initiator = initiator
        self.recipient = recipient
        self.topic = topic
        self.id = uuid4().hex
        self.origin, self.purpose = origin, purpose
        self.concerns = {role: deepcopy(npc.current_concern)
                         for role, npc in (("initiator", initiator), ("recipient", recipient))}
        self.opening_kind = opening_kind
        self.focus_claim_id = focus_claim_id  # Required task account; an ordinary concern is optional.
        self.goal_id = goal_id
        self.escalation = None
        self.accepted_revisions = {}
        self.dialogue = DialogueProgress()
        self.food_outcome = ""
        self.food_reason = ""
        self.handoff_thanks = []
        self.rewritten_handoff_thanks = []
        self.outcome_event_id = None
        self.reactions = []
        self.arrived = False
        self.active = True
        self.record = ConversationRecord((initiator.name, recipient.name), topic, CONVERSATION_WALKING_TO_MEET,
                                         origin=origin, purpose=purpose, id=self.id,
                                         participant_ids=[initiator.id, recipient.id],
                                         grievance_id=focus_claim_id if purpose == "confrontation" else None)
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
            npc.world.reservations.acquire("conversation", self, npc)
        initiator.destination, recipient.destination = destinations

    @classmethod
    def restore(cls, initiator, recipient, record, *, progress, dialogue, saved_states,
                proposals, information_context, concerns):
        return cls(initiator, recipient, progress.topic, initiator.world.tree,
                   _restored=dict(record=record, progress=progress, dialogue=dialogue, saved_states=saved_states,
                                  proposals=proposals, information_context=information_context, concerns=concerns))

    def _restore(self, initiator, recipient, restored):
        self.__dict__.update(vars(restored["progress"]))
        self.initiator, self.recipient, self.record = initiator, recipient, restored["record"]
        self.dialogue, self.saved_states = restored["dialogue"], restored["saved_states"]
        self.proposals, self.information_context = restored["proposals"], restored["information_context"]
        self.concerns, self.name_actors, self.active = restored["concerns"], {}, True
        self.dialogue.retry_after_restore()
        initiator.world.reservations.acquire("conversation", self, initiator, recipient)

    @property
    def grievance_id(self):
        return self.record.grievance_id

    def other(self, npc):
        return self.recipient if npc is self.initiator else self.initiator

    def resident_activities(self):
        return zip((self.initiator, self.recipient), self.saved_states)

    def activity_snapshots(self):
        return iter(self.saved_states)

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
        from dialogue_validation import dialogue_validation_errors, validate_dialogue_format, validate_dialogue_speakers, validate_resident_grounding
        # Reject stale actor snapshots before displaying or committing anything.
        if getattr(self, "request_roster_revision", self.initiator.world.roster_revision) != self.initiator.world.roster_revision:
            raise ValueError("The resident roster changed while dialogue was being prepared. Retry this meeting.")
        if any(n.command_revision != self.request_actor_revisions[n.id] for n in (self.initiator, self.recipient)):
            raise ValueError("The participants changed activities before the reply arrived.")
        if self.initiator.speech_only or self.recipient.speech_only:
            if response.gift or response.accepted_proposals or response.handoff_thanks or response.escalation or response.leverage:
                raise ValueError("Snake meetings allow speech and sourced disclosures only.")
        validate_live_agreements(self, response, dialogue_validation=dialogue_validation)
        from transfers import validate_resident_gift
        validate_resident_gift(response.gift, response.turns, {
            "purpose": self.purpose, "arrival_focus": self.opening_kind == "orientation",
            "initiator": self.initiator.snapshot(self.initiator.world.tree),
            "recipient": self.recipient.snapshot(self.initiator.world.tree)})
        if response.gift and response.gift.accept_turn is not None:
            from transfers import validate_transfer
            world = self.initiator.world
            validate_transfer(world, world.npc_by_id(response.gift.giver_id), world.npc_by_id(response.gift.recipient_id),
                              response.gift.item, response.gift.quantity, interaction=self)
        if dialogue_validation:
            validate_dialogue_format(response.turns)
            validate_dialogue_speakers(response.turns, {"initiator": self.initiator.name, "recipient": self.recipient.name})
            initiator, recipient = getattr(self, "request_participants", None) or (
                self.initiator.snapshot(self.initiator.world.tree), self.recipient.snapshot(self.initiator.world.tree))
            validate_resident_grounding(response, {
                "_name_actors": self.name_actors,
                "initiator": initiator,
                "recipient": recipient,
                "_information_validation": asdict(self.information_context) if self.information_context else None})
        orientation_discussed = validate_orientation(response, {
            "arrival_focus": self.opening_kind == "orientation", "_dialogue_validation": dialogue_validation})
        if not dialogue_validation:
            self.record.dialogue_errors = list(dict.fromkeys([
                *response._dialogue_errors,
                *dialogue_validation_errors(response, self.snapshot(self.initiator.world.tree, self.initiator.world.npcs))]))
        # The exchange moves each opinion from where it stood when generation began, and by at most one step;
        # anything that changed it meanwhile, such as a harvest seen, is kept rather than overwritten.
        starts = self.information_context.opinions if self.information_context is not None else {}
        opinions = []
        for role, npc, partner, entry in (("initiator", self.initiator, self.recipient, response.initiator_opinion),
                                          ("recipient", self.recipient, self.initiator, response.recipient_opinion)):
            live = npc.opinion_score(partner.id)
            change = entry.opinion - starts.get(role, {}).get(partner.id, live)
            change = max(-CONVERSATION_OPINION_STEP, min(CONVERSATION_OPINION_STEP, change))
            opinions.append((npc, partner, entry.model_copy(update={"opinion": max(-100, min(100, live + change))})))
        from social import validate_escalation
        validate_escalation(self, response, opinions[0][2].opinion)
        self.escalation = response.escalation
        self.accepted_revisions = ({n.id: n.command_revision for n in (self.initiator, self.recipient)}
                                   if self.escalation and self.escalation.action == "attack" else {})
        self.handoff_thanks = response.handoff_thanks
        self.accepted_gift = response.gift if response.gift and response.gift.accept_turn is not None else None
        self.rewritten_handoff_thanks = list(rewritten_parts[-1]) if rewritten_parts and response.handoff_thanks else []
        record_generated_turns(self, response.turns, rewritten_parts)
        self.record.outcomes = [
            describe_opinion_change(
                npc.name, partner.name, npc.opinion_score(partner.id, default=None), entry.opinion, entry.reason
            )
            for npc, partner, entry in opinions
        ]
        if self.escalation:
            self.record.outcomes.append(f"Confrontation decision: {self.escalation.action}. Grievance: {self.grievance_id}.")
        self.record.status = CONVERSATION_DIALOGUE_READY
        self.record.orientation_discussed = orientation_discussed
        self.accepted_proposals = list(response.accepted_proposals)
        self.accepted_disclosures = list(response.disclosures)
        self.third_party_opinions = list(response.third_party_opinions)
        self.leverage_uses = list(response.leverage)
        if not self.social_effects_applied:
            commit_agreements(self)
            commit_information(self)
            self.record.name_forms = name_forms(self.name_actors)
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
        return (self.active and self.dialogue.dialogue_state in ("waiting", "pending")
                and not self.initiator.departure and not self.recipient.departure
                and world.actor_by_id(self.initiator.id) is self.initiator
                and world.actor_by_id(self.recipient.id) is self.recipient
                and self.initiator.location == self.recipient.location)

    def fail_dialogue(self, reason, *, failed_turns=()):
        if not self.accepts_response:
            return
        record_failed_dialogue(self, failed_turns)
        self.dialogue.dialogue_state = "failed"
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
            self.cancel("The participants are no longer in the same location.")
            return
        if self.arrived:
            from interactions import observe_conversation
            observe_conversation(self.initiator.world, self.record, self.initiator, self.recipient)
            face_each_other(self.initiator, self.recipient)
            if self.dialogue.dialogue_state in ("waiting", "pending"):
                self.dialogue.wait_elapsed += dt
                if self.dialogue.wait_elapsed >= DIALOGUE_TIMEOUT_SECONDS:
                    self.fail_dialogue("Dialogue generation timed out. No dialogue was generated.")
            elif dialogue_finished(self, dt, reading=reading):
                if self.escalation and self.escalation.action == "attack":
                    self.initiator.world.commit_killing(self)
                    if not self.active:
                        return
                if self.accepted_gift or self.purpose == "food_request" and self.food_outcome == "accepted":
                    self.handoff(self.accepted_gift)
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
            self.record.status = CONVERSATION_WAITING_FOR_DIALOGUE
            from interactions import observe_conversation
            observe_conversation(self.initiator.world, self.record, self.initiator, self.recipient)
            face_each_other(self.initiator, self.recipient)
            if self.purpose == "food_request":
                self.prepare_food_request()

    def snapshot(self, tree, npcs):
        if self.arrived and self.active:
            from interactions import observe_conversation
            observe_conversation(self.initiator.world, self.record, self.initiator, self.recipient)
        self.request_roster_revision = self.initiator.world.roster_revision
        self.request_actor_revisions = {n.id: n.command_revision for n in (self.initiator, self.recipient)}
        def participant(npc):
            state = npc.snapshot(tree, self.other(npc).id)
            # These replies were addressed to the player before the NPCs met.
            state.pop("last_reply", None)
            return state

        if self.dialogue.dialogue_state == "waiting":
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
        # The live check must judge speech against the same knowledge the speakers were given.
        self.request_participants = deepcopy((initiator, recipient))
        return {
            "request_kind": "conversation",
            "_name_actors": self.name_actors,
            "interaction_id": self.id,
            "origin": self.origin,
            "purpose": self.purpose,
            "opening_kind": self.opening_kind,
            "arrival_focus": self.opening_kind == "orientation",
            "arrival_welcome": (self.opening_kind == "orientation" and self.initiator.room_number == JUDY_ROOM_NUMBER
                                and self.recipient.arrival_day > STARTING_DAY),
            "escalation_choices": self.initiator.world.escalation_choices(self),
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
        }

    def finish(self, tree):
        if not self.active:
            return
        self.record.pending_witness_ids.clear()
        self.active = False
        self.record.status = CONVERSATION_FINISHED
        if not self.turns:
            self.record.status = CONVERSATION_FAILED if self.error else CONVERSATION_ENDED_BEFORE_DIALOGUE
            self.record.outcomes = [self.error or "The conversation ended before any dialogue was generated."]
        if self.purpose != "food_request":
            world = self.initiator.world
            text = (f"{self.initiator.name} and {self.recipient.name} talked about {self.topic}."
                    if self.turns else f"{self.initiator.name} and {self.recipient.name}'s meeting ended without generated dialogue.")
            event = world.publish("conversation_completed" if self.turns else "interaction_failed",
                                  (self.initiator, self.recipient), text, self.record)
            self.outcome_event_id = event.id
            if self.dialogue.dialogue_state == "ready":
                for npc, reaction in zip((self.initiator, self.recipient), self.reactions):
                    reason, weight = reaction
                    remember(npc, self.other(npc), event, "conversation", f"Talked about {self.topic}. {reason}", weight)
        elif not self.turns or self.food_outcome in ("", "accepted"):
            if not self.error:
                self.record.status = CONVERSATION_ENDED_BEFORE_SHARING
                self.record.outcomes = ["The food request ended before an apple was shared."]
            event = self.initiator.world.publish("interaction_failed", (self.initiator, self.recipient),
                                                self.record.outcomes[0], self.record)
            self.outcome_event_id = event.id
        self.release_participants(self.social_effects_applied)

    def cancel(self, reason):
        """Interrupt the meeting without replaying accepted speech or effects."""
        if not self.active:
            return
        self.escalation = None
        self.record.pending_witness_ids.clear()
        self.active = False
        self.record.status, self.record.error = CONVERSATION_CANCELLED, reason
        self.record.outcomes.append(reason)
        self.release_participants(bool(self.record.turns), revise=True)

    def release_participants(self, completed, *, revise=False):
        """Restore each suspended activity only while this meeting still owns it."""
        for npc, state in self.resident_activities():
            if not release_resident(npc.world, "conversation", self, npc, state, self.other(npc).id):
                continue
            npc.pause_after_conversation(completed)
            if not completed:
                npc.cooldowns[self.other(npc).id] = dialogue_failure_cooldown(npc, self.other(npc))
            if revise:
                npc.command_revision += 1

    def handoff(self, gift=None):
        """Reverse the roles while preserving suspended activity progress."""
        if not self.active:
            return
        intent = gift or TransferIntent(self.recipient.id, self.initiator.id, "fruit", 1)
        donor = self.initiator if intent.giver_id == self.initiator.id else self.recipient
        requester = self.other(donor)
        states = self.saved_states if donor is self.initiator else self.saved_states[::-1]
        self.active = False
        donor.world.reservations.release("conversation", self, requester, donor)
        Handoff(donor, requester, intent.item, intent.quantity, self.record,
             saved_states=states, already_met=True,
             thanks=self.handoff_thanks, rewritten_thanks=self.rewritten_handoff_thanks)


def validate_saved_conversations(ctx):
    state = ctx.state
    conversations, history, facts, claims, creations, reserved, tree_id = (
        state.conversations, ctx.history, ctx.facts, ctx.claims, state.creations, ctx.reserved, state.tree.id)
    live, live_ids = {**ctx.live, state.snake.id: state.snake}, ctx.live_ids | {state.snake.id}
    from constants import CONVERSATION_MAX_TURNS
    from activity import validate_activity_target
    from information import validate_information
    from social_saves import validate_saved_concern
    from social import validate_agreements
    from types import SimpleNamespace
    for interaction in conversations:
        pair = (interaction.initiator, interaction.recipient)
        if pair[0] == pair[1] or any(i not in live or i in reserved for i in pair) or interaction.record not in history:
            raise ValueError("Invalid live interaction participants.")
        for index, activity in zip(pair, interaction.saved_states):
            if live[pair[0]].location != live[pair[1]].location:
                raise ValueError("Conversation participants must share a location.")
            validate_activity_target(activity, live[index], tree_id, live_ids)
            if activity.destination is not None:
                from locations import validate_destination
                validate_destination(activity.destination, location=live[index].location)
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
            validate_saved_concern(interaction.concerns[role], live[index], claims, facts)
        if interaction.goal_id and (live[pair[0]].goal is None or interaction.goal_id != live[pair[0]].goal.id):
            raise ValueError("A meeting goal must belong to its initiator.")
        if interaction.focus_claim_id and not any(b.claim_id == interaction.focus_claim_id for b in live[pair[0]].beliefs):
            raise ValueError("A required meeting account must be known to its initiator.")
        if len(record.turns) > CONVERSATION_MAX_TURNS:
            raise ValueError("A resident conversation exceeds its accepted turn budget.")
        ready = interaction.dialogue_state == "ready"
        statuses = (CONVERSATION_DIALOGUE_READY,) if ready else (CONVERSATION_WALKING_TO_MEET, CONVERSATION_WAITING_FOR_DIALOGUE)
        if record.status not in statuses:
            raise ValueError("A completed, failed or cancelled record cannot restore an active conversation.")
        if record.id != interaction.id or ready != bool(record.generated_turns()) or ready != interaction.social_effects_applied:
            raise ValueError("Invalid conversation commit progress.")
        if interaction.information_context:
            context = interaction.information_context
            if context.participants != {"initiator": live[pair[0]].id, "recipient": live[pair[1]].id}:
                raise ValueError("Invalid information speakers.")
            if context.focus_claim_id != interaction.focus_claim_id:
                raise ValueError("The required meeting account must match its information snapshot.")
            if any(o.fact_id not in facts or o.quantity != facts[o.fact_id].quantity
                   or o.kind != facts[o.fact_id].kind or o.subject_id != facts[o.fact_id].subject_id for o in context.options):
                raise ValueError("Missing snapshot facts.")
        validate_agreements(interaction.accepted_proposals, interaction.proposals, record.generated_turns(), live[pair[0]].id, live[pair[1]].id)
        if ready:
            try:
                validate_information(SimpleNamespace(turns=record.generated_turns(), disclosures=interaction.accepted_disclosures,
                    third_party_opinions=interaction.third_party_opinions, leverage=interaction.leverage_uses),
                    interaction.information_context)
            except (KeyError, TypeError, AttributeError, IndexError) as exc:
                raise ValueError("Invalid information snapshot references.") from exc
        if interaction.accepted_gift:
            from transfers import validate_resident_gift
            validate_resident_gift(interaction.accepted_gift, record.generated_turns(), {
                "purpose": interaction.purpose, "arrival_focus": interaction.opening_kind == "orientation",
                **{
                role: {"actor_id": live[index].id, "inventory": live[index].inventory,
                       "owned_works": [{"id": w.id, "title": w.title} for w in creations
                                       if w.owner_id == live[index].id]}
                for role, index in zip(("initiator", "recipient"), pair)}})
            if interaction.accepted_gift.accept_turn is None:
                raise ValueError("A pending gift handoff needs an accepted offer.")
