"""Records for social knowledge, goals and promises, dated by the daily clock."""

from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field
from constants import CONVERSATION_MAX_TURNS, PHASES, REPUTATION_FACT_LIMIT


GoalKind = Literal["obtain_apple", "help_with_apple", "repair_relationship", "sabotage_relationship"]
FactKind = Literal["apple_given", "work_given", "apple_refused", "promise_kept", "promise_broken", "private_motive", "former_life", "arrival", "departure", "apple_eaten", "conversation_seen", "harvest", "killing", "snake_claim", "visit", "report"]
Stance = Literal["uncertain", "believed", "disbelieved"]
ConcernKind = Literal["recollection", "news", "surroundings", "plans"]


@dataclass
class ConversationConcern:
    kind: ConcernKind
    claim_id: str | None
    remaining: float


def phase_index(day, phase):
    return len(PHASES) * (day - 1) + PHASES.index(phase)


def phase_label(index):
    return str(index + 1)


@dataclass
class Goal:
    id: Annotated[str, Field(min_length=1)]
    kind: GoalKind
    target_id: str
    objective: Annotated[str, Field(min_length=1, max_length=400)]
    motive: Annotated[str, Field(min_length=1, max_length=400)]
    approach: Annotated[str, Field(min_length=1, max_length=400)]
    boundary: Annotated[str, Field(min_length=1, max_length=400)]
    created_phase: Annotated[int, Field(ge=0)]
    deadline: Annotated[int, Field(ge=0)]
    baseline_opinion: Annotated[int, Field(ge=-100, le=100)] = 0
    status: Literal["active", "completed", "failed", "cancelled"] = "active"
    resolution_event_id: str | None = None
    resolved_phase: Annotated[int | None, Field(ge=0)] = None
    outcome: str = ""
    last_attempt_phase: Annotated[int | None, Field(ge=0)] = None
    third_npc_id: str | None = None
    claim_id: str | None = None


@dataclass
class Proposal:
    id: Annotated[str, Field(min_length=1)]
    promisor_id: str
    recipient_id: str
    deadline: Annotated[int, Field(ge=0)]


@dataclass
class Promise:
    id: Annotated[str, Field(min_length=1)]
    promisor_id: str
    recipient_id: str
    conversation_id: Annotated[str, Field(min_length=1)]
    created_phase: Annotated[int, Field(ge=0)]
    deadline: Annotated[int, Field(ge=0)]
    item: Literal["fruit"] = "fruit"
    quantity: Literal[1] = 1
    status: Literal["pending", "fulfilled", "broken", "cancelled"] = "pending"
    resolution_event_id: str | None = None
    resolved_phase: Annotated[int | None, Field(ge=0)] = None
    outcome: str = ""
    consequence_event_id: str | None = None


@dataclass
class OpinionEffect:
    event_id: Annotated[str, Field(min_length=1)]
    target_id: str
    delta: Annotated[int, Field(ge=-REPUTATION_FACT_LIMIT, le=REPUTATION_FACT_LIMIT)]
    reason: Annotated[str, Field(min_length=1, max_length=400)]
    claim_id: str | None = None
    actor_id: str | None = None
    conversation_id: str | None = None
    phase: Annotated[int | None, Field(ge=0)] = None


@dataclass
class Fact:
    id: Annotated[str, Field(min_length=1)]
    kind: FactKind
    subject_id: str
    object_id: str
    source_id: Annotated[str, Field(min_length=1)]
    conversation_id: str | None
    phase: Annotated[int, Field(ge=0)]
    summary: Annotated[str, Field(min_length=1)]
    witness_ids: list[str]
    quantity: Annotated[int | None, Field(strict=True, ge=1)] = None
    location: str | None = None


@dataclass
class Claim:
    id: Annotated[str, Field(min_length=1)]
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    deliberate: bool
    phase: Annotated[int, Field(ge=0)]
    conversation_id: str | None


@dataclass
class Belief:
    npc_id: str
    claim_id: str
    stance: Stance
    source_id: str
    disclosure_id: str | None
    direct: bool
    phase: Annotated[int, Field(ge=0)]
    assessed: bool = False
    opinion_delta: Annotated[int, Field(ge=-REPUTATION_FACT_LIMIT, le=REPUTATION_FACT_LIMIT)] = 0


@dataclass
class Disclosure:
    id: Annotated[str, Field(min_length=1)]
    claim_id: str
    conversation_id: str
    speaker_id: str
    listener_id: str
    turn: Annotated[int, Field(ge=0, le=CONVERSATION_MAX_TURNS - 1)]
    intent: Literal["share", "challenge", "lie"]
    stance: Stance
    source_disclosure_id: str | None
    phase: Annotated[int, Field(ge=0)]
    event_id: Annotated[str, Field(min_length=1)]


@dataclass
class ClaimOption:
    id: Annotated[str, Field(min_length=1)]
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    creator_id: str | None
    kind: FactKind
    assertion: Annotated[str, Field(min_length=1)]
    quantity: Annotated[int | None, Field(strict=True, ge=1)] = None


@dataclass
class InformationContext:
    participants: dict[str, str]
    beliefs: dict[str, list[Belief]]
    options: list[ClaimOption]
    can_lie: dict[str, bool]
    opinions: dict[str, dict[str, int]]
    allowed: dict[str, list[str]]
    focus_claim_id: str | None


def _record_index(world, name):
    # Records are only appended between trims, so the list and its length identify the index; trims clear it.
    records = getattr(world, name)
    cache = getattr(world, f"_{name}_index", None)
    if cache is None or cache[0] is not records or cache[1] != len(records):
        cache = (records, len(records), {record.id: record for record in records})
        setattr(world, f"_{name}_index", cache)
    return cache[2]


def clear_record_indexes(world):
    world._facts_index = world._claims_index = None


def fact_by_id(world, fact_id):
    return _record_index(world, "facts")[fact_id]


def claim_by_id(world, claim_id):
    return _record_index(world, "claims")[claim_id]


def owns_recollection(owner_id, belief, claim, fact):
    return (belief.direct and claim.affirmed and not claim.deliberate
            and fact.kind == "former_life" and fact.subject_id == owner_id)
