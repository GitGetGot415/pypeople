"""The tree's python and its generated, locally committed encounters."""

from dataclasses import dataclass, field
import math
import random
from uuid import uuid4

from conversation_history import ConversationRecord

SNAKE_MARGIN = 24
SNAKE_SPEED = 35.0
SNAKE_FRAME_SECONDS = 0.22
HEARING_RADIUS = 150
WITNESS_RADIUS = 220
CROWD_LIMIT = 10
PROTECTED_POPULATION = 2
RECOVERY_SECONDS = 20.0

# These are instructions to the provider, never locally supplied spoken lines.
SNAKE_DOCTRINE = (
    "You are a python guarding the only apple tree. You claim to speak for a god. "
    "You claim this garden is purgatory for undecided souls, obedience may lead to heaven, "
    "and being eaten sends a disobedient soul to hell. You explain one arrival per day by "
    "claiming many other purgatory zones distribute the population. These claims are unverified. "
    "You want to become indispensable: trade apples for private information or public support, "
    "use learned accounts selectively, and sometimes call your knowledge divine revelation. "
    "You know only supplied accounts, with their actual sources. Do not invent offenses or secrets."
)


@dataclass
class Snake:
    id: str = "garden-python"
    x: float = 540.0
    y: float = 300.0
    destination: tuple[float, float] | None = None
    facing: int = 4
    pause_remaining: float = 0.0
    animation_elapsed: float = 0.0
    activity: str = "guarding"
    target_id: str | None = None
    beliefs: list = field(default_factory=list)
    standing: dict = field(default_factory=dict)
    offenses: dict = field(default_factory=dict)
    eaten_today: int = 0
    recovery_remaining: float = 0.0
    pardons: list = field(default_factory=list)
    judged_phase: int = -1
    decision_remaining: float = 3.0
    encounter: object = field(default=None, repr=False, compare=False)
    world: object = field(default=None, repr=False, compare=False)
    selected: bool = False
    reply: str = ""
    end_conversation: bool = False
    request_pending: bool = False
    command_revision: int = 0
    retry_request: dict | None = None
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    @property
    def name(self):
        return "Python"

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
        from social import assertion_text, claim_by_id, fact_by_id
        known = []
        for belief in self.beliefs[-16:]:
            claim = claim_by_id(self.world, belief.claim_id)
            known.append({"claim_id": claim.id, "assertion": assertion_text(self.world, fact_by_id(self.world, claim.fact_id), claim.affirmed),
                          "source_id": belief.source_id, "source_disclosure_id": belief.disclosure_id,
                          "original_speaker_id": claim.originator_id, "stance": belief.stance})
        return {"actor_id": self.id, "name": self.name, "activity": self.status,
                "known_accounts": known, "doctrine": SNAKE_DOCTRINE,
                "day": self.world.day, "phase": self.world.phase,
                "living_count": len(self.world.npcs), "eaten_today": self.eaten_today,
                "recovery_remaining": self.recovery_remaining}

    def move_toward(self, point, dt):
        dx, dy = point[0] - self.x, point[1] - self.y
        distance = math.hypot(dx, dy)
        if distance:
            self.facing = math.floor(math.atan2(dx, -dy) / (math.pi / 4) + 0.5) % 8
        step = SNAKE_SPEED * dt
        if distance <= step:
            self.x, self.y = point
            self.animation_elapsed = 0.0
            return True
        self.x += dx / distance * step
        self.y += dy / distance * step
        self.animation_elapsed = (self.animation_elapsed + dt) % (2 * SNAKE_FRAME_SECONDS)
        return False

    def update(self, dt):
        if dt <= 0:
            return
        self.recovery_remaining = max(0.0, self.recovery_remaining - dt)
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
        self.activity = "guarding"
        if self.destination is None or math.dist(self.destination, (tree.x, tree.y)) > 150:
            self.destination = (tree.x + self.rng.uniform(40, 100), tree.y + self.rng.uniform(50, 100))
        if self.move_toward(self.destination, dt):
            self.destination = None


class SnakeEncounter:
    """One resident, one snake, one reservation, one final commit."""

    def __init__(self, world, resident, purpose, reason="", *, origin="autonomous"):
        from npc import Behavior
        if (world.snake.encounter or world.snake.request_pending or resident.snake_encounter
                or world.npc_by_id(resident.id) is not resident or resident.background is None
                or resident.conversation or resident.gift or resident.eating or resident.request_pending
                or any(resident.id in (action.npc_id, action.target_id) for action in world.pending_actions)):
            raise ValueError("The snake or resident is busy.")
        self.world = world
        self.initiator, self.recipient = world.snake, resident
        self.id = uuid4().hex
        self.purpose, self.origin = purpose, origin
        self.topic = reason or purpose.replace("_", " ")
        self.accusation = reason
        self.record = ConversationRecord((world.snake.name, resident.name), self.topic,
                                         "Awaiting tree visitor", origin=origin, purpose=purpose, id=self.id)
        self.saved_state = resident.activity_state()
        resident.release_claims(world)
        resident.command_revision += 1
        resident.behavior, resident.owner = Behavior.WAIT, "autonomous"
        resident.destination = resident.target_id = None
        resident.activity_elapsed = 0.0
        resident.resume_pending = False
        resident.reason = f"Meeting the snake: {self.topic}"
        resident.snake_encounter = self
        world.snake.encounter = self
        world.snake.target_id = resident.id
        world.snake.activity = "judging" if purpose == "judgment" else "bargaining"
        self.reserved = purpose in ("bargain", "judgment") and world.tree.fruit > 0 and resident.inventory_space > 0
        self.active = True
        self.arrived = False
        self.dialogue_state = "waiting"
        self.wait_elapsed = 0.0
        self.talking_remaining = 4.0
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
                and self.world.npc_by_id(self.recipient.id) is self.recipient
                and self.initiator.encounter is self)

    def other(self, actor):
        return self.recipient if actor is self.initiator else self.initiator

    def snapshot(self, tree=None, npcs=None):
        from social import assertion_text, claim_by_id, fact_by_id
        npc, snake, world = self.recipient, self.initiator, self.world
        accounts = []
        known = {(claim_by_id(world, b.claim_id).fact_id, claim_by_id(world, b.claim_id).affirmed) for b in snake.beliefs}
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            if (claim.fact_id, claim.affirmed) not in known and belief.stance != "disbelieved":
                accounts.append({"claim_id": claim.id, "assertion": assertion_text(world, fact_by_id(world, claim.fact_id), claim.affirmed),
                                 "source_id": belief.source_id, "source_disclosure_id": belief.disclosure_id})
        audience = [p.id for p in world.listeners(npc) if p is not npc]
        permitted = ["refuse", "negotiate", "challenge"] if self.purpose == "bargain" else ["spare"] if self.purpose == "judgment" else ["talk"]
        if self.purpose == "judgment" and world.eating_permitted(npc):
            permitted.append("eat")
        if self.reserved:
            if accounts:
                permitted.append("secret")
            if audience:
                permitted.append("endorsement")
        self.request_context = {"request_kind": "snake_encounter", "interaction_id": self.id,
            "purpose": self.purpose, "snake": snake.snapshot(), "resident": npc.snapshot(world.tree),
            "first_meeting": not any(record is not self.record and record.turns
                                     and set(record.participants) == {snake.name, npc.name} for record in world.history),
            "accusation": self.accusation, "standing": snake.standing.get(npc.id, 0),
            "permitted_decisions": permitted, "offering_accounts": accounts,
            "audience_ids": audience, "roster_revision": world.roster_revision,
            "resident_revision": npc.command_revision,
            "_resident_knowledge": [(b.claim_id, b.stance, b.source_id, b.disclosure_id) for b in npc.beliefs],
            "_snake_knowledge": [(b.claim_id, b.stance, b.source_id, b.disclosure_id) for b in snake.beliefs]}
        return self.request_context

    def set_response(self, response):
        from llm import validate_snake_response
        if not self.accepts_response or not self.request_context:
            return
        validate_snake_response(response, self.request_context)
        npc, world = self.recipient, self.world
        if (self.request_context["roster_revision"] != world.roster_revision
                or self.request_context["resident_revision"] != npc.command_revision):
            raise ValueError("The residents changed while this exchange was being prepared. Retry the encounter.")
        for key, actor in (("_resident_knowledge", npc), ("_snake_knowledge", self.initiator)):
            before = [tuple(b) for b in self.request_context[key]]
            current = [(b.claim_id, b.stance, b.source_id, b.disclosure_id) for b in actor.beliefs]
            if before != current:
                raise ValueError("An account changed while this exchange was being prepared. Retry the encounter.")
        if npc.attitudes != self.request_context["resident"]["private_context"]["snake_attitudes"]:
            raise ValueError("The resident's attitudes changed during generation. Retry the encounter.")
        # No payment speech is displayed until all settlement checks have passed.
        world.commit_snake_response(self, response)
        self.turns = list(response.turns)
        for turn in self.turns:
            actor = self.initiator if turn.speaker == "initiator" else npc
            self.record.add_turn(actor.name, turn.text, [p.model_dump() for p in turn.parts])
        self.dialogue_state = "ready"
        self.stage = "dialogue"
        self.record.status = "Judgment announced" if self.purpose == "judgment" else "Dialogue ready"

    def fail_dialogue(self, reason):
        if self.committed or not self.active:
            return
        self.record.error = reason
        self.dialogue_state = "failed"
        self.initiator.retry_request = {"resident_id": self.recipient.id, "purpose": self.purpose,
                                        "reason": self.accusation, "origin": self.origin, "error": reason}
        self.world.publish("interaction_failed", (self.initiator, self.recipient),
                           f"Snake encounter needs retry: {reason}", self.record)
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
            if self.world.npc_by_id(npc.id) is npc:
                npc.restore_activity(self.saved_state)
                npc.resume_pending = True
                npc.cooldowns[self.initiator.id] = 25.0
                npc.cooldowns[self.world.tree.id] = 25.0
        if self.initiator.encounter is self:
            self.initiator.encounter = None
            self.initiator.target_id = None
            self.initiator.destination = None
            self.initiator.activity = "recovering" if self.initiator.recovery_remaining else "guarding"

    def update(self, dt, *, reading=False):
        if not self.active:
            return
        npc, snake = self.recipient, self.initiator
        if self.world.npc_by_id(npc.id) is not npc:
            self.finish(error="The resident is no longer here.")
            return
        if self.stage == "waiting" and math.dist((npc.x, npc.y), self.world.tree.interaction_point) > 8:
            self.stage, self.arrived = "approach", False
        if self.stage == "approach":
            npc.destination = self.world.tree.interaction_point
            old_x, old_y = npc.x, npc.y
            at_tree = npc._move(dt)
            if (npc.x, npc.y) != (old_x, old_y):
                npc.facing = math.floor(math.atan2(npc.x - old_x, old_y - npc.y) / (math.pi / 4) + 0.5) % 8
            if at_tree and math.dist((snake.x, snake.y), (npc.x, npc.y)) <= HEARING_RADIUS:
                npc.destination = None
                self.arrived = True
                self.stage = "waiting"
                self.record.status = "Waiting for generated dialogue"
            return
        if self.stage == "waiting":
            self.face_each_other()
            self.wait_elapsed += dt
            if self.wait_elapsed > 125:
                self.fail_dialogue("Dialogue generation timed out. Retry to continue.")
            return
        if self.stage == "dialogue":
            self.face_each_other()
            if reading and self.turns:
                return
            self.talking_remaining = max(0, self.talking_remaining - dt)
            if self.talking_remaining:
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
            if self.elapsed >= 1.0:
                if not self.world.depart_resident(npc, self):
                    self.finish(error="Eating permission changed; the resident was spared.")

    def face_each_other(self):
        for actor in (self.initiator, self.recipient):
            other = self.other(actor)
            actor.facing = math.floor(math.atan2(other.x - actor.x, actor.y - other.y) / (math.pi / 4) + 0.5) % 8
