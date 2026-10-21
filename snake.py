"""The tree's python and its generated, locally committed encounters."""

from dataclasses import dataclass, field
import math
import random
from uuid import uuid4

from constants import (
    DIALOGUE_TIMEOUT_SECONDS, HEARING_RADIUS, POST_DIALOGUE_SECONDS,
    SNAKE_EAT_SECONDS, SNAKE_FRAME_SECONDS, SNAKE_NAME, SNAKE_SPEED,
    STARTING_FACING, STARTING_SNAKE_ACTIVITY, STARTING_SNAKE_DECISION_SECONDS, STARTING_SNAKE_POSITION,
)
from conversation_history import ConversationRecord
from interactions import (dialogue_finished, face_each_other, face_toward,
                          finish_generated_dialogue, move_toward, record_failed_dialogue, record_generated_turns)


# These are instructions to the provider, never locally supplied spoken lines.
SNAKE_DOCTRINE = (
    f"{SNAKE_NAME} guards the only apple tree and speaks as the appointed messenger of a god. "
    "Its divine authority is asserted directly and confidently in first person. Its teachings: "
    "this garden is purgatory for undecided souls; obedience may lead to heaven; "
    "being eaten sends a disobedient soul to hell. Other purgatory zones distribute "
    f"the population, explaining the few arrivals here. These are {SNAKE_NAME}'s positions, not narrator-certified truth. "
    "Its private aim is dependence, information and public support; that aim shapes offers, not spoken self-description. "
    "When stock is available, it gives real apples upfront for obligations owed later, confident it can judge and eat deliberate defaulters. "
    "It cannot offer or promise an apple from an empty tree. Earlier apple advances still carry their existing obligations. "
    "It may present a learned account as divine insight, but knows only supplied accounts with actual sources. "
    "It has no access to undisclosed secrets and cannot invent offenses."
)

SNAKE_VOICE_BRIEF = (
    "Calm, confident and ceremonious. Use measured, concise sentences, few contractions, "
    "no slang or swearing, rare hesitation and occasional dry humor. No marked accent."
)

SNAKE_BEHAVIOR = (
    f"Speak with calm, confident divine authority as {SNAKE_NAME}, appointed messenger of a god and guard of the apple tree. "
    "Keep your private pursuit of influence private. Follow the current transaction and supplied knowledge. "
    "Use a brief introduction only on first contact. Answer the resident's latest point and perform only the current task; "
    "keep earlier introductions, theology, offers, and accusations out of later replies."
)


@dataclass
class Snake:
    id: str = "garden-python"
    x: float = STARTING_SNAKE_POSITION[0]
    y: float = STARTING_SNAKE_POSITION[1]
    destination: tuple[float, float] | None = None
    facing: int = STARTING_FACING
    pause_remaining: float = 0.0
    animation_elapsed: float = 0.0
    activity: str = STARTING_SNAKE_ACTIVITY
    target_id: str | None = None
    beliefs: list = field(default_factory=list)
    standing: dict = field(default_factory=dict)
    offenses: dict = field(default_factory=dict)
    advances: dict = field(default_factory=dict)
    eaten_today: int = 0
    recovery_remaining: float = 0.0
    pardons: list = field(default_factory=list)
    judged_phase: int = -1
    decision_remaining: float = STARTING_SNAKE_DECISION_SECONDS
    encounter: object = field(default=None, repr=False, compare=False)
    world: object = field(default=None, repr=False, compare=False)
    selected: bool = False
    reply: str = ""
    rewritten_reply: str = ""
    end_conversation: bool = False
    request_pending: bool = False
    command_revision: int = 0
    retry_request: dict | None = None
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    @property
    def name(self):
        return SNAKE_NAME

    @property
    def frame(self):
        return 1 + int(self.animation_elapsed / SNAKE_FRAME_SECONDS) % 2

    @property
    def status(self):
        if self.encounter:
            return self.encounter.record.status
        if self.recovery_remaining:
            return f"Recovering ({self.recovery_remaining:.0f}s)"
        return self.activity.title()

    @property
    def decision_reason(self):
        return "Controls the tree; seeks information, support, and influence."

    @property
    def portrait_emotion(self):
        return None

    @property
    def conversation(self):
        return self.encounter

    @property
    def gift(self):
        return None

    @property
    def eating(self):
        return self.activity == "eating"

    def set_emotion(self, *args):
        pass

    def set_selected(self, value):
        self.selected = value
        if not value:
            self.end_conversation = False

    def contains(self, point):
        return self.x - 32 <= point[0] <= self.x + 32 and self.y - 58 <= point[1] <= self.y

    def snapshot(self, tree=None):
        from social import account_view, claim_by_id
        from name_knowledge import actor_catalog
        # Rehearing an event cannot replace the snake's own firsthand knowledge.
        by_fact = {}
        for belief in self.beliefs:
            claim = claim_by_id(self.world, belief.claim_id)
            previous = by_fact.get(claim.fact_id)
            if previous is None or belief.direct or not previous.direct:
                by_fact[claim.fact_id] = belief
        known = []
        for belief in list(by_fact.values())[-16:]:
            claim = claim_by_id(self.world, belief.claim_id)
            known.append(account_view(self.world, claim, belief))
        return {"actor_id": self.id, "name": self.name, "activity": self.status,
                "name_rule": "knows_resident_names", "_name_actors": actor_catalog(self.world),
                "name_knowledge": {"own_name": SNAKE_NAME, "known_names": {}},
                "known_accounts": known, "doctrine": SNAKE_DOCTRINE,
                "tree_fruit": self.world.tree.fruit, "tree_available_fruit": self.world.available_fruit,
                "day": self.world.day, "phase": self.world.phase,
                "living_count": len(self.world.npcs), "eaten_today": self.eaten_today,
                "recovery_remaining": self.recovery_remaining,
                "apple_advances": [{"resident_id": actor_id, "offering": advance["offering"],
                                    "follow_up_in": advance["remaining"]}
                                   for actor_id, advance in self.advances.items()]}

    def move_toward(self, point, dt):
        if (self.x, self.y) != point:
            face_toward(self, point)
        if move_toward(self, point, dt, SNAKE_SPEED):
            self.animation_elapsed = 0.0
            return True
        self.animation_elapsed = (self.animation_elapsed + dt) % (2 * SNAKE_FRAME_SECONDS)
        return False

    def update(self, dt):
        if dt <= 0:
            return
        self.recovery_remaining = max(0.0, self.recovery_remaining - dt)
        for advance in self.advances.values():
            advance["remaining"] = max(0.0, advance["remaining"] - dt)
        tree = self.world.tree
        if self.encounter:
            # The resident comes to the guard; the snake never follows them out.
            if self.encounter.stage not in ("condemned_approach", "eating"):
                guard = (tree.x + 60, tree.y + 70)
                self.move_toward(guard, dt)
            return
        if self.recovery_remaining:
            self.activity = "recovering"
            return
        if not (self.world.phase_pending or self.selected or self.request_pending or self.retry_request):
            from social import fact_by_id
            for npc in self.world.outdoor_npcs:
                if npc.selected:
                    continue
                advance = self.advances.get(npc.id)
                if npc.id in self.offenses and npc.id not in self.pardons and self.world.eating_permitted(npc):
                    purpose, reason = "judgment", fact_by_id(self.world, self.offenses[npc.id]).summary
                elif advance and advance["remaining"] <= 0:
                    purpose, reason = "bargain", "Collecting the offering for an apple already given upfront"
                else:
                    continue
                try:
                    SnakeEncounter(self.world, npc, purpose, reason)
                except ValueError:
                    continue  # Finish an existing activity before following up.
                return
        self.activity = "guarding"
        if self.destination is None or math.dist(self.destination, (tree.x, tree.y)) > 150:
            self.destination = (tree.x + self.rng.uniform(40, 100), tree.y + self.rng.uniform(50, 100))
        if self.move_toward(self.destination, dt):
            self.destination = None


class SnakeEncounter:
    """One resident, one snake, one reservation, one final commit."""

    def __init__(self, world, resident, purpose, reason="", *, origin="autonomous"):
        if world.phase_pending and origin == "autonomous":
            raise ValueError("Waiting for the pending time change before starting another automatic encounter.")
        if (world.snake.encounter or world.snake.request_pending or resident.snake_encounter
                or not world.is_outdoors(resident) or resident.behavior.value == "GO_HOME"
                or resident.home_reason == "night"
                or resident.busy or resident.request_pending
                or any(resident.id in (action.npc_id, action.target_id) for action in world.pending_actions)):
            raise ValueError(f"{SNAKE_NAME} or the resident is busy.")
        self.world = world
        self.initiator, self.recipient = world.snake, resident
        self.id = uuid4().hex
        self.purpose, self.origin = purpose, origin
        self.topic = reason or purpose.replace("_", " ")
        self.accusation = reason
        self.record = ConversationRecord((world.snake.name, resident.name), self.topic,
                                         "Awaiting tree visitor", origin=origin, purpose=purpose, id=self.id)
        from social import concern_description
        self.record.intended_subjects = {world.snake.name: self.topic,
                                        resident.name: concern_description(resident, resident.current_concern)}
        self.saved_state = resident.suspend_activity(f"Meeting {SNAKE_NAME}: {self.topic}")
        resident.snake_encounter = self
        world.snake.encounter = self
        world.snake.target_id = resident.id
        world.snake.activity = ("judging" if purpose == "judgment" else
                                "introducing" if purpose == "introduction" else "bargaining")
        self.reserved = (purpose == "bargain" and resident.id not in world.snake.advances
                         and world.tree.fruit > 0 and resident.inventory_space > 0)
        self.active = True
        self.arrived = False
        self.dialogue_state = "waiting"
        self.wait_elapsed = 0.0
        self.talking_remaining = POST_DIALOGUE_SECONDS
        self.turns = []
        self.pending_opinions = []
        self.committed = False
        self.outcome = ""
        self.outcome_event_id = None
        self.stage = "approach"
        self.elapsed = 0.0
        self.request_context = None
        retry = world.snake.retry_request
        if retry and retry["resident_id"] == resident.id and retry["purpose"] == purpose:
            world.snake.retry_request = None
        world.history.append(self.record)

    @property
    def error(self):
        return self.record.error

    @error.setter
    def error(self, message):
        self.record.error = message

    @property
    def accepts_response(self):
        return (self.active and not self.committed and self.dialogue_state in ("waiting", "pending")
                and self.world.is_outdoors(self.recipient)
                and self.initiator.encounter is self)

    def other(self, actor):
        return self.recipient if actor is self.initiator else self.initiator

    def snapshot(self, tree=None, npcs=None):
        from social import account_view, claim_by_id
        from name_knowledge import actor_catalog
        npc, snake, world = self.recipient, self.initiator, self.world
        advance = snake.advances.get(npc.id)
        can_give_apple = self.reserved and world.tree.fruit > 0 and npc.inventory_space > 0 and not advance
        accounts = []
        known = {(claim_by_id(world, b.claim_id).fact_id, claim_by_id(world, b.claim_id).affirmed) for b in snake.beliefs}
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            if (belief.stance != "disbelieved" and
                    (advance and claim.id == advance["account_id"]
                     or not advance and (claim.fact_id, claim.affirmed) not in known)):
                accounts.append(account_view(world, claim, belief))
        audience = [p.id for p in world.listeners(npc) if p is not npc]
        permitted = ["refuse", "challenge"] if self.purpose == "bargain" else ["spare"] if self.purpose == "judgment" else ["talk"]
        if self.purpose == "judgment" and world.eating_permitted(npc):
            permitted.append("eat")
        if advance:
            if advance["offering"] == "secret" and accounts:
                permitted.append("secret")
            elif advance["offering"] == "endorsement" and audience:
                permitted.append("endorsement")
            else:
                permitted.append("negotiate")  # An unavailable audience is not a broken promise.
        elif can_give_apple:
            if accounts:
                permitted.append("advance_secret")
            if len(world.npcs) > 1:
                permitted.append("advance_endorsement")
        if self.purpose == "bargain" and not advance and not can_give_apple:
            permitted = ["talk"]
            accounts = []
        self.request_context = {"request_kind": "snake_encounter", "interaction_id": self.id,
            "_name_actors": actor_catalog(world),
            "purpose": self.purpose, "snake": snake.snapshot(), "resident": npc.snapshot(world.tree),
            "first_meeting": not any(record is not self.record and record.turns
                                     and set(record.participants) == {snake.name, npc.name} for record in world.history),
            "accusation": self.accusation, "standing": snake.standing.get(npc.id, 0),
            "permitted_decisions": permitted, "offering_accounts": accounts,
            "can_give_apple": bool(can_give_apple), "apple_reserved": self.reserved,
            "advance": advance.copy() if advance else None,
            "audience_ids": audience, "roster_revision": world.roster_revision,
            "_name_audience": [p.snapshot(world.tree) for p in world.listeners(npc) if p is not npc],
            "_name_snake_audience": [p.snapshot(world.tree) for p in world.listeners(snake) if p is not npc],
            "resident_revision": npc.command_revision,
            "_resident_knowledge": [(b.claim_id, b.stance, b.source_id, b.disclosure_id) for b in npc.beliefs],
            "_snake_knowledge": [(b.claim_id, b.stance, b.source_id, b.disclosure_id) for b in snake.beliefs]}
        return self.request_context

    def set_response(self, response, *, reading=False, rewritten_parts=(), dialogue_validation=True):
        from llm import dialogue_validation_errors, validate_snake_response
        from name_knowledge import commit_names, orientation_speech, snake_name_witnesses, validate_names
        from social import claim_by_id
        if not self.accepts_response or not self.request_context:
            return
        validate_snake_response(response, self.request_context, dialogue_validation=dialogue_validation)
        npc, world = self.recipient, self.world
        if (self.request_context["roster_revision"] != world.roster_revision
                or self.request_context["resident_revision"] != npc.command_revision):
            raise ValueError("The residents changed while this exchange was being prepared. Retry the encounter.")
        relevant_facts = {claim_by_id(world, claim_id).fact_id
                          for claim_id in (response.account_id, response.snake_account_id) if claim_id}
        for key, actor in (("_resident_knowledge", npc), ("_snake_knowledge", self.initiator)):
            before = {b[0]: tuple(b) for b in self.request_context[key]}
            current = {b.claim_id: (b.claim_id, b.stance, b.source_id, b.disclosure_id) for b in actor.beliefs}
            # Observing an unrelated new event does not invalidate this reply.
            # Changed sources/stances or new evidence about a chosen account do.
            if (any(current.get(claim_id) != belief for claim_id, belief in before.items())
                    or any(b.claim_id not in before and claim_by_id(world, b.claim_id).fact_id in relevant_facts
                           for b in actor.beliefs)):
                raise ValueError("An account changed while this exchange was being prepared. Retry the encounter.")
        if npc.attitudes != self.request_context["resident"]["private_context"]["snake_attitudes"]:
            raise ValueError("The resident's attitudes changed during generation. Retry the encounter.")
        witnesses = [p for p in world.listeners(npc) if p is not npc and p.id in self.request_context["audience_ids"]]
        witnessed = snake_name_witnesses({**self.request_context,
            "_name_audience": [p.snapshot(world.tree) for p in witnesses],
            "_name_snake_audience": [p.snapshot(world.tree) for p in world.listeners(self.initiator) if p is not npc]},
            response.decision)
        _, _, name_changes = validate_names(response.turns, {**self.request_context,
            "_dialogue_validation": dialogue_validation,
            "resident": npc.snapshot(world.tree), "_name_witnesses": witnessed})
        if not dialogue_validation:
            self.record.dialogue_errors = list(dict.fromkeys([
                *response._dialogue_errors, *dialogue_validation_errors(response, self.request_context)]))
        # Speech describing a handoff is displayed only after the real transfer commits.
        world.commit_snake_response(self, response)
        record_generated_turns(self, response.turns, rewritten_parts)
        self.record.name_witnesses = {index: [p["npc_id"] for p in people] for index, people in witnessed.items()}
        commit_names(self, name_changes)
        self.record.orientation_discussed = orientation_speech(turn.text for turn in response.turns)
        self.stage = "dialogue"
        self.record.status = "Judgment announced" if self.purpose == "judgment" else "Dialogue ready"
        finish_generated_dialogue(self, reading=reading)

    def fail_dialogue(self, reason, *, failed_turns=()):
        if self.committed or not self.active:
            return
        record_failed_dialogue(self, failed_turns)
        self.record.error = reason
        self.dialogue_state = "failed"
        self.initiator.retry_request = {"resident_id": self.recipient.id, "purpose": self.purpose,
                                        "reason": self.accusation, "origin": self.origin, "error": reason}
        self.world.publish("interaction_failed", (self.initiator, self.recipient),
                           f"Meeting with {SNAKE_NAME} needs retry: {reason}", self.record)
        self.finish(error=reason)

    def finish(self, tree=None, *, error=""):
        if not self.active:
            return
        self.active = False
        self.reserved = False
        self.record.status = "Failed" if error else "Finished"
        self.record.error = error
        if error:
            self.record.outcomes.append(error)
        npc = self.recipient
        if npc.snake_encounter is self:
            npc.snake_encounter = None
            if self.world.is_outdoors(npc):
                npc.restore_after_interaction(self.saved_state, self.initiator.id, self.world.tree.id)
                npc.pause_after_conversation(self.committed)
        if self.initiator.encounter is self:
            self.initiator.encounter = None
            self.initiator.target_id = None
            self.initiator.destination = None
            self.initiator.activity = "recovering" if self.initiator.recovery_remaining else "guarding"

    def update(self, dt, *, reading=False):
        if not self.active:
            return
        npc, snake = self.recipient, self.initiator
        if not self.world.is_outdoors(npc):
            self.finish(error="The resident is no longer here.")
            return
        if self.stage == "waiting" and math.dist((npc.x, npc.y), self.world.tree.interaction_point) > 8:
            self.stage, self.arrived = "approach", False
        if self.stage == "approach":
            npc.destination = self.world.tree.interaction_point
            at_tree = npc._move(dt)
            if at_tree and math.dist((snake.x, snake.y), (npc.x, npc.y)) <= HEARING_RADIUS:
                npc.destination = None
                self.arrived = True
                self.stage = "waiting"
                self.record.status = "Waiting for generated dialogue"
            return
        if self.stage == "waiting":
            face_each_other(self.initiator, self.recipient)
            self.wait_elapsed += dt
            if self.wait_elapsed >= DIALOGUE_TIMEOUT_SECONDS:
                self.fail_dialogue("Dialogue generation timed out. Retry to continue.")
            return
        if self.stage == "dialogue":
            face_each_other(self.initiator, self.recipient)
            if not dialogue_finished(self, dt, reading=reading):
                return
            if self.outcome == "eat":
                self.stage = "condemned_approach"
                self.record.status = "Approaching to eat"
                snake.activity = "approaching condemned resident"
            else:
                self.finish()
        elif self.stage == "condemned_approach":
            if snake.move_toward((npc.x, npc.y), dt):
                self.stage = "eating"
                snake.activity = "eating"
                self.record.status = "Eating"
                self.elapsed = 0.0
        elif self.stage == "eating":
            self.elapsed += dt
            if self.elapsed >= SNAKE_EAT_SECONDS:
                if not self.world.depart_resident(npc, self):
                    self.finish(error="Eating permission changed; the resident was spared.")
