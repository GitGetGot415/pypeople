"""A garden wanderer using the ordinary social conversation lifecycle."""

import math
import random
from dataclasses import dataclass, field
from typing import Annotated
from pydantic import Field
from constants import (SNAKE_NAME, SNAKE_SPEED, SNAKE_FRAME_SECONDS, STARTING_SNAKE_POSITION, DECISION_SECONDS, STARTING_SNAKE_DECISION_SECONDS,
                       ACCUSATION_FACT_KINDS, HARVEST_FREE_APPLES, REPUTATION_FACT_LIMIT)
from npc import ConversationActor
from activity import ActivitySnapshot
from interactions import PlayerReplyProgress, move_toward
from player_records import PlayerOpinion
from conversation_history import Memory
from social_records import Belief, ConversationConcern, OpinionEffect

SNAKE_VOICE_BRIEF = "Calm, confident, ceremonious and concise, with occasional dry humor."
SNAKE_BEHAVIOR = (
    f"You are {SNAKE_NAME}, a talking python wandering the garden. "
    # Listing what the python lacks made small models recite the list; state it as a scope instead.
    "Your whole life has been spent in the garden. Speak about the garden and what residents did there; "
    "do not describe the building's inside or bring up your own past unless a resident asks. "
    "Privately provoke distrust between residents by selectively reporting true known accounts and making pointed accusations or insinuations. "
    "Connect witnessed conduct to selfishness, greed or disregard for the listener, and say so plainly as your own view. "
    "For harvests, use harvest_context to consider repeated pickups together, rather than presenting one pickup as the whole story. "
    "Use harvest_extent.share for the stock description: all means the entire stock, most means a majority with other fruit outside this collector's harvest. "
    "When the share is all, emphasize taking all the apples; do not soften it to most, deny it, or invent leftovers in later replies. "
    "When harvest_extent is absent, keep the scope limited to what you actually witnessed. "
    "Totals cover only your witnessed pickups on the named day; do not combine days or assume you witnessed everything. "
    "Do not reduce the exchange to neutral news, mysterious tree behavior or omens. "
    "When the listener resists, press the accusation again rather than agreeing or changing the subject. "
    "You may claim religious authority, but your explanations remain claims. "
    "You cannot harvest, pick, collect, carry, eat or give apples. Witnessing someone harvest never makes you the harvester. "
    "You have no control over apples or residents. Never invent events, quantities or memories, "
    "and never command violence. Speak to the listener using I and you."
)

@dataclass
class SnakeValues(ActivitySnapshot):
    id: str = "garden-python"
    x: float = STARTING_SNAKE_POSITION[0]
    y: float = STARTING_SNAKE_POSITION[1]
    animation_elapsed: Annotated[float, Field(ge=0, lt=2 * SNAKE_FRAME_SECONDS)] = 0.0
    decision_remaining: Annotated[float, Field(ge=0)] = STARTING_SNAKE_DECISION_SECONDS
    beliefs: list[Belief] = field(default_factory=list)
    opinions: dict = field(default_factory=dict)
    memories: list[Memory] = field(default_factory=list)
    pending_social_opinions: list[OpinionEffect] = field(default_factory=list)
    current_concern: ConversationConcern | None = None
    conversation_pause_remaining: float = 0.0
    conversation_return_remaining: float = 0.0
    social_fatigue: Annotated[float, Field(ge=0, le=100)] = 0.0
    player_opinion: PlayerOpinion = field(default_factory=PlayerOpinion)
    command_revision: int = 0
    reply: str = ""
    rewritten_reply: str = ""

@dataclass
class Snake(SnakeValues, ConversationActor):
    world: object = field(default=None, repr=False, compare=False)
    selected: bool = False
    player_reply: PlayerReplyProgress = field(default_factory=PlayerReplyProgress, repr=False, compare=False)
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
    speech_only = True
    location = "garden"
    sprite_size = (64, 58)
    departure = journey = home_reason = goal = player_call = creative_job = None
    room_number = 0
    inventory = {}
    preferences = {"sociability": 0.6, "generosity": 0.5}

    @property
    def name(self):
        return SNAKE_NAME

    @property
    def character(self):
        return {"Name": SNAKE_NAME, "Personality": SNAKE_BEHAVIOR,
                "Voice": SNAKE_VOICE_BRIEF, "Mannerisms": "Measured speech."}

    @property
    def conversation(self):
        return self.world.reservations.get(self.id, "conversation") if self.world else None

    @property
    def interaction(self):
        return self.conversation

    gift = None
    emotion = "neutral"
    emotion_intensity = 0
    unread_player_reply = None
    waiting_for_done = False
    resolved_player_calls = ()
    player_call_cooldown = 0.0
    eating = False

    @property
    def busy(self):
        return self.conversation is not None

    @property
    def generation_pending(self):
        return self.player_reply.generating

    @property
    def interaction_blocked(self):
        return self.generation_pending

    @property
    def frame(self):
        return 1 + int(self.animation_elapsed / SNAKE_FRAME_SECONDS) % 2

    @property
    def movement_bounds(self):
        from locations import feet_bounds
        return feet_bounds(self.location)

    @property
    def status(self):
        return self.conversation.record.status if self.conversation else "Wandering"

    @property
    def decision_reason(self):
        return "Seek influence through selective reporting and suspicion."

    @property
    def portrait_emotion(self):
        return None

    def set_emotion(self, *args):
        pass

    def set_selected(self, value):
        self.selected = value
        if not value:
            self.player_reply.closing = False

    def contains(self, point):
        return self.x - 32 <= point[0] <= self.x + 32 and self.y - 58 <= point[1] <= self.y

    def _move(self, dt):
        moving = move_toward(self, self.destination, dt, SNAKE_SPEED)
        self.animation_elapsed = (self.animation_elapsed + dt) % (2 * SNAKE_FRAME_SECONDS)
        return moving

    def snapshot(self, tree=None, partner_id=None, *, private=True):
        from actor_identity import actor_catalog
        from area_knowledge import area_knowledge
        from social import private_context
        state = {"actor_id": self.id, "name": self.name, "kind": "python", "location": self.location,
                 "character": self.character, "position": [self.x, self.y], "facing": self.facing,
                 "activity": self.status}
        if private:
            state.update(memories=[vars(m) for m in self.memories], _name_actors=actor_catalog(self.world), area_knowledge=area_knowledge(self, partner_id),
                         inventory={}, owned_works=[], day=self.world.day,
                         living_count=self.world.population, tree_fruit=self.world.tree.fruit,
                         tree_available_fruit=self.world.tree.fruit, private_context=private_context(self, partner_id))
            state["private_context"]["opinions"] = self.named_opinions()
            state["private_context"]["immediate_objective"] = self.decision_reason
        return state

    def update(self, dt):
        from social_records import claim_by_id, fact_by_id, ConversationConcern
        from information import witnessed_harvests
        from social import apply_opinion_effects, repeat_partner_paused
        from player import owns_resident
        if dt <= 0:
            return
        self.update_cooldowns(dt)
        apply_opinion_effects(self)
        if self.busy or self.selected or self.interaction_blocked or owns_resident(self.world, self):
            return
        if self.resume_pending:
            self.resume_pending = False
        if not self.conversation_pause_remaining and self.decision_remaining == 0:
            self.decision_remaining = DECISION_SECONDS
            choices = []
            harvests = witnessed_harvests(self.world, self.beliefs)
            for human in self.world.npcs:
                if (not self.world.company_available(human) or human.location != "garden"
                        or human.conversation_pause_remaining or human.id in self.cooldowns
                        or self.id in human.cooldowns or repeat_partner_paused(self, human)):
                    continue
                for belief in self.beliefs:
                    claim = claim_by_id(self.world, belief.claim_id)
                    fact = fact_by_id(self.world, claim.fact_id)
                    # Telling the accused about their own conduct cannot turn anyone against them, and
                    # ordinary conduct (eating, sharing, a single apple) is no accusation at all.
                    quantity = harvests.get((fact.day, fact.subject_id), {}).get("quantity", fact.quantity) if fact.kind == "harvest" else 0
                    if (belief.stance != "believed" or not claim.affirmed or fact.subject_id == self.id
                            or fact.subject_id == human.id or fact.kind not in ACCUSATION_FACT_KINDS
                            or fact.kind == "harvest" and quantity <= HARVEST_FREE_APPLES):
                        continue
                    # Even a witness, or someone who already heard it from another teller, is stirred up again by this telling,
                    # unless they already judged this very account or retelling has moved them as far as it can over this event.
                    known = [b for b in human.beliefs if claim_by_id(self.world, b.claim_id).fact_id == fact.id]
                    if (any(b.assessed and b.claim_id == claim.id for b in known)
                            or sum(b.opinion_delta for b in known) <= -REPUTATION_FACT_LIMIT):
                        continue
                    # One harvest story per collector and day, however many pickups it took.
                    told = [fact_by_id(self.world, claim_by_id(self.world, d.claim_id).fact_id) for d in self.world.disclosures
                            if d.speaker_id == self.id and d.listener_id == human.id]
                    shared = any(t.id == fact.id or fact.kind == t.kind == "harvest"
                                 and (t.day, t.subject_id) == (fact.day, fact.subject_id) for t in told)
                    if not shared:
                        choices.append((fact.kind == "killing", fact.kind == "harvest", quantity,
                                        -human.opinion_score(fact.subject_id)
                                        - math.dist((self.x, self.y), (human.x, human.y)) / SNAKE_SPEED,
                                        belief.day, human, claim))
            if choices:
                *_, human, claim = max(choices, key=lambda row: row[:-2])
                subject = self.world.participant_by_id(claim.subject_id)
                self.current_concern = ConversationConcern("news", claim.id, DECISION_SECONDS)
                # Worded as Nachash's own plan: "you" here was read as the listener, and the steps were read aloud.
                self.world.start_conversation(self, human,
                    f"Turn {human.name} against {subject.name}: tell {human.name} what I saw {subject.name} do, "
                    f"and press my view that it was selfish and shows {subject.name} cares nothing for {human.name}.",
                    origin="autonomous", opening_kind="news", focus_claim_id=claim.id)
                return
            humans = [n for n in self.world.npcs if n.location == "garden" and self.world.company_available(n)
                      and not n.conversation_pause_remaining and n.id not in self.cooldowns
                      and self.id not in n.cooldowns and not repeat_partner_paused(self, n)]
            if humans and not any(r.turns and self.id in r.participant_ids for r in self.world.history):
                human = min(humans, key=lambda n: math.dist((self.x, self.y), (n.x, n.y)))
                self.world.start_conversation(self, human, "Meet this resident naturally and hear their concerns.", origin="autonomous")
                return
        if self.destination is None:
            from locations import clamp_point
            left, right, top, bottom = self.movement_bounds
            self.destination = clamp_point("garden", (self.rng.uniform(left, right), self.rng.uniform(6, bottom)))
        if self._move(dt):
            self.destination = None
