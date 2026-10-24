"""Private objectives and apple commitments, resolved only by committed world changes."""

import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import Field
from constants import (SNAKE_NAME, CONVERSATION_MAX_AGREEMENTS, CONCERN_LIFETIME_RANGE, CONCERN_WEIGHTS, CONVERSATION_MAX_TURNS, FAILURE_COOLDOWN,
                      GOAL_IMPROVEMENT, LEVERAGE_ACCOUNT_KINDS, MAX_DIALOGUE_FAILURE_COOLDOWN, PHASES,
                      CONVERSATION_MAX_DISCLOSURES, CONVERSATION_MAX_REPUTATION_EFFECTS,
                      CONVERSATION_MAX_LEVERAGE, REPUTATION_FACT_LIMIT)
from conversation_history import remember
from dialogue_text import split_sentences
from dialogue_rules import unquoted_speech, mentions_account
from types import SimpleNamespace
from actor_identity import account_references, actor_id
from dialogue_rules import named_references, validate_account_speech, validate_apple_promise


GoalKind = Literal["obtain_apple", "help_with_apple", "repair_relationship", "sabotage_relationship"]
FactKind = Literal["apple_given", "work_given", "apple_refused", "promise_kept", "promise_broken", "private_motive", "former_life", "arrival", "departure", "apple_eaten", "conversation_seen", "harvest", "killing", "snake_claim", "visit", "report"]
Stance = Literal["uncertain", "believed", "disbelieved"]
ConcernKind = Literal["recollection", "news", "surroundings", "plans"]


@dataclass
class ConversationConcern:
    kind: ConcernKind
    claim_id: str | None
    remaining: float


def valid_concern(npc, concern):
    if concern is None:
        return False
    if concern.kind not in ("recollection", "news"):
        return concern.claim_id is None and concern.kind in ("surroundings", "plans")
    belief = next((b for b in npc.beliefs if b.claim_id == concern.claim_id), None)
    claim = next((c for c in npc.world.claims if c.id == concern.claim_id), None)
    if belief is None or claim is None:
        return False
    fact = fact_by_id(npc.world, claim.fact_id)
    return concern.kind == "news" or owns_recollection(npc.id, belief, claim, fact)


def owns_recollection(owner_id, belief, claim, fact):
    return (belief.direct and claim.affirmed and not claim.deliberate
            and fact.kind == "former_life" and fact.subject_id == owner_id)


def concern_key(npc, concern):
    if concern is None:
        return None
    claim = next((c for c in npc.world.claims if c.id == concern.claim_id), None)
    return ("account", claim.fact_id) if claim else (concern.kind, None)


def concern_topic(npc, concern):
    guidance = {
        "recollection": "Talk about a memory from my former life",
        "news": "Compare views on an account I know",
        "surroundings": "Talk about life here and what is on my mind",
        "plans": "Work out what to do next",
    }
    return guidance[concern.kind]


def last_conversation_partner(npc):
    return next((next(other.id for other in npc.world.actors if other is not npc and r.includes(other))
                 for r in reversed(npc.world.history)
                 if r.includes(npc) and "You" not in r.participants and r.turns), None)


def repeat_partner_paused(npc, other):
    return any(person.conversation_return_remaining > 0 and last_conversation_partner(person) == partner.id
               for person, partner in ((npc, other), (other, npc)))


def dialogue_failure_cooldown(npc, other):
    """Repeated failed meetings back off; any accepted exchange breaks the streak."""
    delay = FAILURE_COOLDOWN
    failures = 0
    for record in reversed(npc.world.history):
        if not record.includes(npc):
            continue
        if record.turns:
            break
        if record.between(npc, other) and record.error:
            failures += 1
            if failures > 1:
                delay = min(MAX_DIALOGUE_FAILURE_COOLDOWN, delay * 2)
            if delay == MAX_DIALOGUE_FAILURE_COOLDOWN:
                break
    return delay


def concern_description(npc, concern):
    """Observer text only; inspecting a concern never selects or consumes one."""
    if concern is None:
        return "No subject selected."
    descriptions = {
        "surroundings": "Reflecting on their surroundings and present situation.",
        "plans": "Considering what to do next.",
    }
    if concern.kind == "plans":
        plans = [npc.goal.objective] if npc.goal and npc.goal.status == "active" else []
        plans.extend(f"Deliver an apple to {npc.world.participant_by_id(p.recipient_id).name}, due {phase_label(p.deadline)}"
                     for p in pending_promises(npc.world, npc.id))
        return "Plans: " + "; ".join(plans) if plans else descriptions["plans"]
    if concern.kind in descriptions:
        return descriptions[concern.kind]
    claim = next((c for c in npc.world.claims if c.id == concern.claim_id), None)
    belief = next((b for b in npc.beliefs if b.claim_id == concern.claim_id), None)
    if claim is None or belief is None:
        return "The account they wanted to discuss is no longer available."
    account = account_view(npc.world, claim, belief)
    title = "Former-life recollection" if concern.kind == "recollection" else "News or a known account"
    basis = ("Firsthand experience" if belief.direct else
             f"Heard from {account['immediate_source_name']}" if belief.source_id != npc.id else "Their own assertion")
    return (f"{title}: {account['assertion']}\n"
            f"About {account['subject_name']}. {basis}. Stance: {belief.stance}.")


def danger_topic(npc, claim_id):
    claim = claim_by_id(npc.world, claim_id)
    return "Seek safety or help about a known alarming account: " + account_content(npc.world, fact_by_id(npc.world, claim.fact_id), claim.affirmed)


def undisclosed_danger(npc):
    """A new alarming fact can interrupt routine thoughts; a repeated source cannot."""
    world = npc.world
    discussed = {claim_by_id(world, d.claim_id).fact_id for d in world.disclosures if d.speaker_id == npc.id}
    return next((b.claim_id for b in reversed(npc.beliefs) if b.stance != "disbelieved"
                 and claim_by_id(world, b.claim_id).fact_id not in discussed
                 and fact_by_id(world, claim_by_id(world, b.claim_id).fact_id).kind in ("killing",)
                 and not any(r.turns and r.topic == danger_topic(npc, b.claim_id) and npc.name in r.participants
                             for r in world.history)), None)


def refresh_concern(npc):
    """Choose locally at a safe decision boundary, never during an interaction."""
    if npc.interaction is not None or npc.departure is not None:
        return
    previous = npc.current_concern
    danger = undisclosed_danger(npc)
    # Do not repeatedly replace a thought with another witness/denial of the same fact.
    if danger and (previous is None or concern_key(npc, previous) != ("account", claim_by_id(npc.world, danger).fact_id)):
        npc.current_concern = ConversationConcern("news", danger, npc.rng.uniform(*CONCERN_LIFETIME_RANGE))
        return
    if valid_concern(npc, previous) and previous.remaining > 0:
        return
    candidates = {"surroundings": [None], "plans": [None], "recollection": [], "news": []}
    for belief in npc.beliefs:
        claim = claim_by_id(npc.world, belief.claim_id)
        fact = fact_by_id(npc.world, claim.fact_id)
        if belief.direct and claim.affirmed and not claim.deliberate and fact.kind == "former_life" and fact.subject_id == npc.id:
            candidates["recollection"].append(claim.id)
        elif fact.kind != "private_motive" or fact.subject_id != npc.id:
            candidates["news"].append(claim.id)
    retired = concern_key(npc, previous)
    candidates = {kind: [claim_id for claim_id in entries
                         if concern_key(npc, ConversationConcern(kind, claim_id, 0)) != retired]
                  for kind, entries in candidates.items()}
    kinds = [kind for kind, entries in candidates.items() if entries]
    kind = npc.rng.choices(kinds, weights=[CONCERN_WEIGHTS[k] for k in kinds])[0]
    npc.current_concern = ConversationConcern(kind, npc.rng.choice(candidates[kind]), npc.rng.uniform(*CONCERN_LIFETIME_RANGE))


def commit_concern_use(conversation):
    """Accepted base effects alone consume an account; private presence consumes nothing."""
    world = conversation.initiator.world
    options = {o.id: o for o in conversation.information_context.options} if conversation.information_context else {}
    for role, npc in (("initiator", conversation.initiator), ("recipient", conversation.recipient)):
        concern = conversation.concerns[role]
        if concern is None or npc.current_concern != concern:
            continue
        if concern.claim_id:
            fact_id = claim_by_id(world, concern.claim_id).fact_id
            used = any(conversation.turns[d.turn].speaker == role and options[d.claim_id].fact_id == fact_id
                       for d in conversation.accepted_disclosures)
        else:
            used = (role == "initiator" and conversation.opening_kind == concern.kind
                    and conversation.purpose == "chat" and conversation.origin == "autonomous")
        if used:
            npc.current_concern.remaining = 0.0


def phase_index(day, phase):
    return len(PHASES) * (day - 1) + PHASES.index(phase)


def phase_label(index):
    return f"Day {index // len(PHASES) + 1} / {PHASES[index % len(PHASES)].title()}"


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
    delta: Annotated[int, Field(ge=-6, le=6)]
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


def fact_by_id(world, fact_id):
    return next(fact for fact in world.facts if fact.id == fact_id)


def claim_by_id(world, claim_id):
    return next(claim for claim in world.claims if claim.id == claim_id)


def assertion_text(world, fact, affirmed=True):
    if affirmed:
        summary = harvest_summary(world.participant_by_id(fact.subject_id).name, fact.quantity) if fact.kind == "harvest" else fact.summary
        return f"At {phase_label(fact.phase)}: {summary}"
    subject, target = world.participant_by_id(fact.subject_id), world.participant_by_id(fact.object_id)
    action = {"apple_given": "give an apple to", "apple_refused": "refuse an apple to",
              "promise_kept": "keep the apple promise to", "promise_broken": "miss the apple promise to"}
    if fact.kind == "private_motive":
        return f"{subject.name} does not have the motive described in this account: {fact.summary}"
    if fact.kind not in action:
        return f"An account disputes: {fact.summary}"
    return f"{subject.name} did not {action[fact.kind]} {target.name} in the exchange at {phase_label(fact.phase)}."


def account_content(world, fact, affirmed=True):
    """Request prose has no record header; keep UI formatting in assertion_text()."""
    summary = (harvest_summary(world.participant_by_id(fact.subject_id).name, fact.quantity)
               if fact.kind == "harvest" else fact.summary if fact.kind == "former_life" else named_references(fact.summary))
    if fact.kind == "former_life":
        _, marker, description = summary.partition("'s own description: ")
        summary = description if marker else summary
    elif fact.kind == "private_motive":
        # The heading describes the goal; the owed private information is why
        # its owner wants it. Ownership stays in the separate account fields.
        _, marker, motive = summary.partition(": ")
        summary = motive if marker else summary
    if affirmed:
        return summary
    subject, target = world.participant_by_id(fact.subject_id), world.participant_by_id(fact.object_id)
    action = {"apple_given": "give an apple to", "apple_refused": "refuse an apple to",
              "promise_kept": "keep the apple promise to", "promise_broken": "miss the apple promise to"}
    if fact.kind == "private_motive":
        return f"{subject.name} does not have the motive described in this account: {summary}"
    if fact.kind not in action:
        return f"An account disputes: {summary}"
    return f"{subject.name} did not {action[fact.kind]} {target.name} in that exchange."


def fact_context(world, fact, affirmed=True):
    date = {"day": fact.phase // len(PHASES) + 1, "phase": PHASES[fact.phase % len(PHASES)]}
    return {"assertion": account_content(world, fact, affirmed), "kind": fact.kind,
            "recorded_at": date, "event_time": None if fact.kind == "former_life" else date,
            "time_scope": "remembered former life; event date unknown" if fact.kind == "former_life" else "life here",
            "quantity": fact.quantity, "location": fact.location}


def account_view(world, claim, belief):
    fact = fact_by_id(world, claim.fact_id)
    catalog = {actor.id: {"name": actor.name} for actor in [*world.actors, world.player]}
    account = {**fact_context(world, fact, claim.affirmed), "claim_id": claim.id,
               "subject_id": claim.subject_id, "object_id": fact.object_id,
               "original_speaker_id": claim.originator_id}
    projected = build_account_view(account, account_knowledge(belief, belief.npc_id), catalog, belief.npc_id, None)
    return {**fact_context(world, fact, claim.affirmed),
            **{key: projected[key] for key in ("claim_id", "subject_id", "subject_name", "object_id", "object_name",
                "knowledge_owner_name", "original_speaker_id", "immediate_source_id", "original_speaker_name", "immediate_source_name")},
            "source_disclosure_id": belief.disclosure_id, "stance": projected["stance"],
            "learned_at": {"day": belief.phase // len(PHASES) + 1, "phase": PHASES[belief.phase % len(PHASES)]},
            "basis": projected["basis"]}


def harvest_summary(name, quantity):
    return f"{name} harvested {quantity} {'apple' if quantity == 1 else 'apples'} from the tree."


def add_fact(world, kind, subject, target, source_id, record, summary, witnesses, *, quantity=None, location=None):
    # Callers own audience selection: direct participants can speak through a door,
    # and an observer may have left before accepted speech establishes the exchange.
    if (kind == "harvest") != (type(quantity) is int and quantity > 0) or kind != "harvest" and quantity is not None:
        raise ValueError("Only harvesting facts require an exact positive integer quantity.")
    if kind == "harvest":
        summary = harvest_summary(subject.name, quantity)
    witnesses = list({p.id: p for p in witnesses if p is not None and p is not world.player}.values())
    fact = next((fact for fact in world.facts if fact.source_id == source_id and fact.kind == kind), None)
    if fact is None:
        fact = Fact(uuid4().hex, kind, subject.id, target.id if target else subject.id, source_id,
                    record.id if record else None, world.phase_index, summary, [], quantity, location)
        world.facts.append(fact)
    for npc in witnesses:
        if npc.id in fact.witness_ids:
            continue
        fact.witness_ids.append(npc.id)
        claim = Claim(uuid4().hex, fact.id, subject.id, npc.id, True, None, False,
                      world.phase_index, record.id if record else None)
        world.claims.append(claim)
        # Promise consequences already have their own opinion adjustment.
        assessed = npc is target and kind in ("promise_kept", "promise_broken")
        delta = (4 if kind == "promise_kept" else -6) if assessed else 0
        npc.beliefs.append(Belief(npc.id, claim.id, "believed", npc.id, None, True,
                                 world.phase_index, assessed, delta))
    return fact


def record_intention(npc, world):
    goal = npc.goal
    if goal:
        add_fact(world, "private_motive", npc, world.participant_by_id(goal.target_id), goal.id, None,
                 f"{npc.name}'s motive for {goal.objective}: {goal.motive}", [npc])


def record_event_fact(world, event, record):
    """Private consequences only; physical commits explicitly record visual facts."""
    kinds = {"food_refused": "apple_refused", "promise_fulfilled": "promise_kept", "promise_broken": "promise_broken"}
    kind = kinds.get(event.kind)
    if kind is None:
        return
    first, second = [world.participant_by_id(npc_id) for npc_id in event.participants]
    subject, target = (second, first) if kind == "apple_refused" else (first, second)
    add_fact(world, kind, subject, target, event.id, record, event.text, [subject, target])


def can_deceive(npc):
    return npc is not npc.world.snake and (npc.preferences["generosity"] < 0.5 or bool(npc.goal and npc.goal.kind == "sabotage_relationship"
                                                     and npc.goal.status == "active"))


def selected_beliefs(npc, partner_id=None, *, pinned=()):
    world = npc.world
    goal = npc.goal
    order = {id(b): i for i, b in enumerate(npc.beliefs)}
    def rank(belief):
        claim = claim_by_id(world, belief.claim_id)
        fact = fact_by_id(world, claim.fact_id)
        return (claim.id in pinned or bool(npc.current_concern and npc.current_concern.claim_id == claim.id),
                bool(goal and goal.claim_id == claim.id),
                fact.kind in ("killing",),
                fact.kind in ("killing", "snake_claim", "arrival", "harvest"),
                partner_id in (fact.subject_id, fact.object_id, belief.source_id),
                bool(goal and fact.subject_id == goal.third_npc_id),
                fact.kind == "promise_broken" or fact.kind == "private_motive" and fact.subject_id != npc.id,
                fact.kind != "private_motive" or fact.subject_id != npc.id, belief.phase, order[id(belief)])
    selected = sorted(npc.beliefs, key=rank, reverse=True)[:6]
    if goal:
        motive = next((b for b in npc.beliefs if fact_by_id(world, claim_by_id(world, b.claim_id).fact_id).source_id == goal.id), None)
        if motive and motive not in selected:
            selected = [*selected[:5], motive]
    return selected


def prepare_information(conversation):
    if conversation.purpose == "food_request":
        return None
    pair = {"initiator": conversation.initiator, "recipient": conversation.recipient}
    world = conversation.initiator.world
    options, allowed = {}, {}
    for role, npc in pair.items():
        concern = conversation.concerns[role]
        selected = selected_beliefs(npc, conversation.other(npc).id,
                                   pinned=(concern.claim_id,) if concern and concern.claim_id else ())
        if role == "initiator" and conversation.focus_claim_id:
            focus = next(b for b in npc.beliefs if b.claim_id == conversation.focus_claim_id)
            selected = [focus, *[b for b in selected if b is not focus]]
        allowed[role] = [belief.claim_id for belief in selected]
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            fact = fact_by_id(world, claim.fact_id)
            options[claim.id] = ClaimOption(claim.id, fact.id, fact.subject_id, claim.originator_id,
                                           claim.affirmed, claim.parent_claim_id, None, fact.kind,
                                           account_content(world, fact, claim.affirmed), quantity=fact.quantity)
        if can_deceive(npc):
            bases = [claim_by_id(world, belief.claim_id) for belief in selected
                     if belief.stance == "believed" and claim_by_id(world, belief.claim_id).affirmed]
            for index, base in enumerate(bases[:2]):
                fact = fact_by_id(world, base.fact_id)
                option_id = f"{conversation.id}:deny_{role}_{index}"
                options[option_id] = ClaimOption(option_id, fact.id, fact.subject_id, npc.id, False,
                                                base.id, npc.id, fact.kind, account_content(world, fact, False), quantity=fact.quantity)
                allowed[role].append(option_id)
        if npc.goal and npc.goal.kind == "sabotage_relationship" and npc.goal.status == "active":
            allowed[role].sort(key=lambda claim_id: options[claim_id].creator_id is None)
    relevant_facts = {options[claim_id].fact_id for entries in allowed.values() for claim_id in entries}
    for role, npc in pair.items():
        for belief in npc.beliefs:
            if options[belief.claim_id].fact_id in relevant_facts and belief.claim_id not in allowed[role]:
                allowed[role].append(belief.claim_id)
    return InformationContext(
        {role: npc.id for role, npc in pair.items()},
        {role: deepcopy(npc.beliefs) for role, npc in pair.items()}, list(options.values()),
        {role: can_deceive(npc) for role, npc in pair.items()},
        {role: {other.id: npc.opinion_score(other.id) for other in world.actors if other is not npc}
         for role, npc in pair.items()}, allowed, conversation.focus_claim_id)


def information_views(context, world):
    """Only speaker-owned knowledge reaches prompts; engine truth stays out of them."""
    if context is None:
        return {}
    catalog = {actor.id: {"name": actor.name} for actor in [*world.actors, world.player]}
    options = {option.id: option for option in context.options}
    result = {}
    for role, npc_id in context.participants.items():
        beliefs = {belief.claim_id: belief for belief in context.beliefs[role]}
        firsthand = {options[b.claim_id].fact_id: options[b.claim_id]
                     for b in beliefs.values() if b.direct}
        views = []
        for claim_id in context.allowed[role]:
            option = options[claim_id]
            belief = beliefs.get(claim_id)
            fact = fact_by_id(world, option.fact_id)
            selected = {**fact_context(world, fact, option.affirmed),
                        "claim_id": claim_id, "subject_id": option.subject_id,
                        "object_id": fact.object_id, "original_speaker_id": option.originator_id}
            projected = build_account_view(selected, account_knowledge(belief, npc_id) if belief else None,
                                           catalog, npc_id, context.participants["recipient" if role == "initiator" else "initiator"])
            view = {**fact_context(world, fact, option.affirmed),
                    **{key: projected[key] for key in ("claim_id", "subject_id", "subject_name", "object_id", "object_name",
                        "knowledge_owner_name", "original_speaker_id", "original_speaker_name")},
                    "permitted_use": "deliberate denial; private to this speaker" if option.creator_id else "known account"}
            if belief:
                view.update(stance=projected["stance"], immediate_source_id=projected["immediate_source_id"],
                            immediate_source_name=projected["immediate_source_name"],
                            learned_at={"day": belief.phase // len(PHASES) + 1, "phase": PHASES[belief.phase % len(PHASES)]},
                            source_disclosure_id=belief.disclosure_id,
                            basis=projected["basis"],
                            already_assessed=any(b.assessed and options[b.claim_id].fact_id == option.fact_id
                                                 and options[b.claim_id].affirmed == option.affirmed
                                                 for b in context.beliefs[role]),
                            current_reputation_influence=sum(b.opinion_delta for b in context.beliefs[role]
                                                             if options[b.claim_id].fact_id == option.fact_id))
            evidence = firsthand.get(option.fact_id)
            if evidence:
                view["firsthand_evidence"] = evidence.assertion
                view["firsthand_stance"] = "believed" if evidence.affirmed == option.affirmed else "disbelieved"
                if belief and belief.direct:
                    view["permitted_use"] = "share personal experience; its truth needs no confirmation"
            views.append(view)
        result[role] = views
    return result


def related_beliefs(option, beliefs, options):
    return [b for b in beliefs if options[b.claim_id].fact_id == option.fact_id]


def reputation_assessment_available(option, related, options, *, believed, heard, participants, assessed=False):
    """Stance is actual during validation, possible before a pending reaction."""
    return (option.kind != "conversation_seen" and heard and believed and option.subject_id not in participants and not assessed
            and not any(b.direct or b.assessed and options[b.claim_id].affirmed == option.affirmed
                        for b in related))


def reputation_deltas(related):
    influence = sum(b.opinion_delta for b in related)
    return [delta for delta in range(-REPUTATION_FACT_LIMIT, REPUTATION_FACT_LIMIT + 1)
            if delta and abs(influence + delta) <= REPUTATION_FACT_LIMIT]


def resident_effect_limits():
    return dict(offers=CONVERSATION_MAX_AGREEMENTS, accepted_proposals=CONVERSATION_MAX_AGREEMENTS,
                disclosures=CONVERSATION_MAX_DISCLOSURES,
                third_party_opinions=CONVERSATION_MAX_REPUTATION_EFFECTS, leverage=CONVERSATION_MAX_LEVERAGE)


def resident_effect_capacity(draft):
    return {("acceptances" if field == "accepted_proposals" else field): max(0, limit - len(draft[field]))
            for field, limit in resident_effect_limits().items()}


def demote_contrary_beliefs(option, related, options, listener_id):
    for belief in related:
        if (options[belief.claim_id].affirmed != option.affirmed and not belief.direct
                and belief.source_id != listener_id):
            belief.stance = "uncertain"


def receive_belief(prior, *, listener_id, claim_id, stance, source_id, disclosure_id, phase, update_source=True):
    """Only hearsay may change stance; exact repeats keep assessment bookkeeping."""
    if prior is None:
        return Belief(listener_id, claim_id, stance, source_id, disclosure_id, False, phase)
    if not prior.direct:
        prior.stance = stance
        if update_source:
            prior.source_id, prior.disclosure_id = source_id, disclosure_id
    return prior


def listener_stances(option, beliefs, options, *, firsthand_source=False):
    """The same belief constraints govern a pending reaction and the final record."""
    related = related_beliefs(option, beliefs.values(), options)
    firsthand = next((b for b in related if b.direct), None)
    prior = beliefs.get(option.id)
    if firsthand:
        return ["believed" if options[firsthand.claim_id].affirmed == option.affirmed else "disbelieved"]
    if prior is not None:
        return [prior.stance]
    stances = ["uncertain", "believed", "disbelieved"]
    if (not firsthand_source and any(options[b.claim_id].affirmed == option.affirmed
                                    and b.stance != "believed" for b in related)):
        stances.remove("believed")
    return stances


def validate_information(response, context, *, names=(), participant_names=None, dialogue_validation=True,
                         actor_references=None, incomplete=False):
    """Simulate knowledge in speech order before any real state is touched."""
    if isinstance(context, dict):
        context = information_from_dict(context)
    if context is None:
        if response.disclosures or response.third_party_opinions or response.leverage:
            raise ValueError("This exchange does not permit disclosures, gossip opinions, or leverage.")
        return
    options = {option.id: option for option in context.options}
    beliefs = {role: {b.claim_id: deepcopy(b) for b in entries} for role, entries in context.beliefs.items()}
    acquired = {role: set() for role in context.participants}
    heard = {}
    seen_disclosures, seen_opinions, seen_leverage = set(), set(), set()
    for entry in (*response.disclosures, *response.third_party_opinions, *response.leverage):
        if not 0 <= entry.turn < len(response.turns):
            raise ValueError("An information effect refers to a missing dialogue turn.")
    if len({entry.turn for entry in response.disclosures}) != len(response.disclosures):
        raise ValueError("Use at most one structured disclosure per spoken turn.")
    for turn, speech in enumerate(response.turns):
        role = speech.speaker
        listener = "recipient" if role == "initiator" else "initiator"
        npc_id = context.participants[role]
        for use in [entry for entry in response.leverage if entry.turn == turn]:
            option = options.get(use.claim_id)
            if (use.claim_id not in beliefs[role] or option is None
                    or use.claim_id not in context.allowed[role] and use.claim_id not in acquired[role]
                    or use.claim_id in seen_leverage
                    or option.kind not in LEVERAGE_ACCOUNT_KINDS):
                raise ValueError(f"Leverage on turn {turn} ({role}) requires a permitted claim already known before this turn. "
                                 "An account disclosed on this same turn is too late; omit leverage for ordinary requests.")
            seen_leverage.add(use.claim_id)
        for entry in [entry for entry in response.third_party_opinions if entry.turn == turn]:
            option = options.get(entry.claim_id)
            belief = beliefs[role].get(entry.claim_id)
            if (option is None or belief is None or (role, entry.claim_id) not in heard
                    or entry.subject_id != option.subject_id or entry.subject_id in context.participants.values()
                    or heard[role, entry.claim_id] >= turn):
                raise ValueError(f"A third-party opinion on turn {turn} ({role}) must cite an account this speaker heard "
                                 "in an earlier turn, with its actual absent subject and exact disclosed claim ID.")
            related = related_beliefs(option, beliefs[role].values(), options)
            if (not reputation_assessment_available(option, related, options,
                    believed=belief.stance == "believed", heard=True, participants=context.participants.values(),
                    assessed=(role, option.fact_id) in seen_opinions)
                    or entry.delta not in reputation_deltas(related)):
                raise ValueError("A repeated or disbelieved account cannot cause another reputation penalty.")
            seen_opinions.add((role, option.fact_id))
            belief.assessed, belief.opinion_delta = True, belief.opinion_delta + entry.delta
        for entry in [entry for entry in response.disclosures if entry.turn == turn]:
            option = options.get(entry.claim_id)
            if option is None or (role, entry.claim_id) in seen_disclosures:
                raise ValueError("Unknown or repeated disclosure.")
            belief = beliefs[role].get(entry.claim_id)
            if dialogue_validation:
                validate_account_speech(speech.text, option.assertion, names=names,
                                        speaker=(participant_names or {}).get(role),
                                        listener=(participant_names or {}).get(listener), kind=option.kind, quantity=option.quantity,
                                        firsthand=bool(belief and belief.direct), actor_references=actor_references,
                                        known_events=[{"assertion": options[b.claim_id].assertion, "kind": options[b.claim_id].kind, "quantity": options[b.claim_id].quantity}
                                                      for b in beliefs[role].values() if b.direct and b.claim_id in options])
            seen_disclosures.add((role, entry.claim_id))
            fabricating = option.creator_id is not None and belief is None
            if fabricating:
                if (option.creator_id != npc_id or entry.intent != "lie" or not context.can_lie[role]
                        or option.parent_claim_id not in beliefs[role]):
                    raise ValueError("This speaker cannot fabricate that assertion.")
                beliefs[role][option.id] = Belief(npc_id, option.id, "disbelieved", npc_id, None, False, 0)
            elif (belief is None or entry.claim_id not in context.allowed[role] and entry.claim_id not in acquired[role]):
                raise ValueError(f"The {role} on turn {turn} has not learned or been permitted to disclose that claim. "
                                 "Use that speaker's permitted_accounts or an account heard in an earlier turn.")
            elif ((entry.intent == "lie" and (not context.can_lie[role] or belief.stance == "believed"))
                  or (entry.intent == "share" and belief.stance == "disbelieved")):
                raise ValueError("A deliberate lie and an honestly repeated belief must remain distinct.")
            firsthand = next((b for b in beliefs[role].values()
                              if b.direct and options[b.claim_id].fact_id == option.fact_id), None)
            if firsthand:
                agrees = options[firsthand.claim_id].affirmed == option.affirmed
                if agrees and entry.intent == "challenge":
                    raise ValueError("The speaker already knows this account firsthand. Share it as a personal experience; "
                                     "do not ask someone else to confirm whether it happened.")
                if (not agrees and entry.intent == "share") or (agrees and entry.intent == "lie"):
                    raise ValueError("The disclosure intent must respect the speaker's firsthand knowledge of this event.")
            prior = beliefs[listener].get(option.id)
            related = related_beliefs(option, beliefs[listener].values(), options)
            if incomplete and entry.listener_stance is None:
                # The assertion has been spoken, but only its listener can choose
                # a stance. Do not authorize downstream effects until they do.
                continue
            stances = listener_stances(option, beliefs[listener], options,
                                       firsthand_source=bool(belief and belief.direct))
            if entry.listener_stance not in stances:
                raise ValueError("The listener's stance must respect firsthand evidence and prior beliefs; "
                                 "repeated hearsay is not independent confirmation. "
                                 f"Permitted stances: {', '.join(stances)}.")
            if entry.listener_stance == "believed":
                demote_contrary_beliefs(option, related, options, context.participants[listener])
            beliefs[listener][option.id] = receive_belief(prior,
                listener_id=context.participants[listener], claim_id=option.id, stance=entry.listener_stance,
                source_id=npc_id, disclosure_id=None, phase=0, update_source=False)
            acquired[listener].add(option.id)
            heard[listener, option.id] = turn
    if context.focus_claim_id and not incomplete:
        focus = options[context.focus_claim_id]
        if not any(response.turns[entry.turn].speaker == "initiator" and options[entry.claim_id].fact_id == focus.fact_id
                   for entry in response.disclosures):
            raise ValueError("A gossip exchange must disclose its chosen account or a permitted denial of it.")
    return beliefs, acquired, heard


def information_from_dict(value):
    if value is None:
        return None
    return InformationContext(
        value["participants"], {role: [Belief(**b) for b in entries] for role, entries in value["beliefs"].items()},
        [ClaimOption(**option) for option in value["options"]], value["can_lie"], value["opinions"], value["allowed"], value["focus_claim_id"])


def commit_information(conversation):
    context = conversation.information_context
    if context is None:
        return
    world = conversation.initiator.world
    options = {option.id: option for option in context.options}
    people = {"initiator": conversation.initiator, "recipient": conversation.recipient}
    heard = {}
    claims = {claim.id: claim for claim in world.claims}
    for turn, speech in enumerate(conversation.turns):
        speaker = people[speech.speaker]
        listener = conversation.other(speaker)
        for use in [entry for entry in conversation.leverage_uses if entry.turn == turn]:
            text = f"{speaker.name} pressured {listener.name} for {use.demand} using a private account; cooperation is not established."
            event = world.publish("leverage_used", (speaker, listener), text, conversation.record)
            conversation.record.outcomes.append(text)
            for npc, partner in ((speaker, listener), (listener, speaker)):
                remember(npc, partner, event, "leverage", text, 0)
        for entry in [entry for entry in conversation.third_party_opinions if entry.turn == turn]:
            claim = claim_by_id(world, entry.claim_id)
            belief = next(b for b in speaker.beliefs if b.claim_id == entry.claim_id)
            source = heard[speaker.id, entry.claim_id]
            originator = world.participant_by_id(claim.originator_id)
            option = options[claim.id]
            reason = (f"Heard {source.name}'s account (original source: {originator.name}): {option.assertion} "
                      f"Personal assessment: {entry.reason}")[:400]
            event = world.publish("claim_assessed", (source, speaker),
                                  f"{speaker.name} assessed an allegation about {world.participant_by_id(entry.subject_id).name} from {source.name}.",
                                  conversation.record)
            belief.assessed = True
            belief.opinion_delta += entry.delta
            speaker.pending_social_opinions.append(OpinionEffect(event.id, entry.subject_id, entry.delta,
                                                                  reason, claim.id, source.id, conversation.record.id, world.phase_index))
        for entry in [entry for entry in conversation.accepted_disclosures if entry.turn == turn]:
            option = options[entry.claim_id]
            source_belief = next((b for b in speaker.beliefs if b.claim_id == option.id), None)
            if not any(claim.id == option.id for claim in world.claims):
                parent = next(b for b in speaker.beliefs if b.claim_id == option.parent_claim_id)
                source_belief = parent
                world.claims.append(Claim(option.id, option.fact_id, option.subject_id, speaker.id,
                                          option.affirmed, option.parent_claim_id, True,
                                          world.phase_index, conversation.record.id))
                claims[option.id] = world.claims[-1]
                speaker.beliefs.append(Belief(speaker.id, option.id, "disbelieved", speaker.id, None,
                                              False, world.phase_index))
            text = f"{speaker.name} {'questioned an account with' if entry.intent == 'challenge' else 'alleged to'} {listener.name}: {option.assertion}"
            event = world.publish("claim_shared", (speaker, listener), text, conversation.record)
            disclosure = Disclosure(f"{conversation.id}:disclosure:{turn}", option.id, conversation.record.id,
                                    speaker.id, listener.id, turn, entry.intent, entry.listener_stance,
                                    source_belief.disclosure_id, world.phase_index, event.id)
            world.disclosures.append(disclosure)
            prior = next((b for b in listener.beliefs if b.claim_id == option.id), None)
            if entry.listener_stance == "believed":
                demote_contrary_beliefs(option, related_beliefs(option, listener.beliefs, claims), claims, listener.id)
            learned = receive_belief(prior, listener_id=listener.id, claim_id=option.id,
                stance=entry.listener_stance, source_id=speaker.id, disclosure_id=disclosure.id, phase=world.phase_index)
            if prior is None:
                listener.beliefs.append(learned)
            heard[listener.id, option.id] = speaker
            originator = world.participant_by_id(option.originator_id)
            remember(speaker, listener, event, "claim_shared", f"I discussed an account with {listener.name}: {option.assertion}", 0)
            remember(listener, speaker, event, "heard_claim",
                     f"{speaker.name} passed on an account from {originator.name}: {option.assertion} My stance: {entry.listener_stance}.", 0)
            conversation.record.outcomes.append(f"Allegation: {speaker.name} -> {listener.name}; {entry.listener_stance}. {option.assertion}")
    for npc in people.values():
        apply_opinion_effects(npc)


def check_sabotage_goal(world, effect, listener, actual_delta, record):
    owner = world.participant_by_id(effect.actor_id)
    goal = owner.goal if owner else None
    if (goal and goal.status == "active" and goal.kind == "sabotage_relationship"
            and goal.target_id == listener.id and goal.third_npc_id == effect.target_id and actual_delta < 0
            and effect.phase is not None and goal.created_phase <= effect.phase <= world.phase_index <= goal.deadline
            and claim_by_id(world, goal.claim_id).fact_id == claim_by_id(world, effect.claim_id).fact_id
            and listener.opinion_score(effect.target_id) <= goal.baseline_opinion - 4):
        resolve_goal(owner, world, "completed", "A committed opinion change from this exchange weakened the relationship.", record)


def trim_information(world):
    """Bound unused accounts while retaining provenance, disputes, and reputation deduplication."""
    claims = {claim.id: claim for claim in world.claims}
    residents = world.residents
    current_goals = {npc.goal.id for npc in residents if npc.goal}
    for npc in residents:
        polarities = {}
        for belief in npc.beliefs:
            claim = claims[belief.claim_id]
            polarities.setdefault(claim.fact_id, set()).add(claim.affirmed)
        keep = {b.claim_id for b in npc.beliefs[-24:]}
        for b in npc.beliefs:
            claim = claims[b.claim_id]
            if b.direct or b.assessed or len(polarities[claim.fact_id]) > 1 or fact_by_id(world, claim.fact_id).source_id in current_goals:
                keep.add(b.claim_id)
        if npc.goal and npc.goal.claim_id:
            keep.add(npc.goal.claim_id)
        if npc.current_concern and npc.current_concern.claim_id:
            keep.add(npc.current_concern.claim_id)
        if npc.conversation:
            role = "initiator" if npc.conversation.initiator is npc else "recipient"
            concern = npc.conversation.concerns[role]
            if concern and concern.claim_id:
                keep.add(concern.claim_id)
            if npc.conversation.information_context:
                keep.update(b.claim_id for b in npc.conversation.information_context.beliefs[role])
        if npc.conversation and npc.conversation.initiator is npc and npc.conversation.focus_claim_id:
            keep.add(npc.conversation.focus_claim_id)
        npc.beliefs[:] = [b for b in npc.beliefs if b.claim_id in keep]
    claim_ids = {b.claim_id for npc in world.actors for b in npc.beliefs}
    claim_ids.update(r.grievance_id for r in world.history if r.grievance_id)
    claim_ids.update(effect.claim_id for npc in residents for effect in npc.pending_social_opinions if effect.claim_id)
    disclosure_ids = {b.disclosure_id for npc in world.actors for b in npc.beliefs if b.disclosure_id}
    disclosure_ids.update(d.id for d in world.disclosures for npc in residents
                          for effect in npc.pending_social_opinions if effect.claim_id == d.claim_id
                          and effect.conversation_id == d.conversation_id and d.listener_id == npc.id)
    disclosure_ids.update(d.id for d in world.disclosures[-100:])
    confrontations = {r.id for r in world.history if r.grievance_id and r.turns}
    disclosure_ids.update(d.id for d in world.disclosures if d.conversation_id in confrontations)
    event_ids = {event.id for event in world.events}
    disclosure_ids.update(d.id for d in world.disclosures if d.event_id in event_ids)
    fact_ids = {fact.id for fact in world.facts[-100:] if fact.source_id not in current_goals}
    active_records = {n.conversation.record.id for n in residents if n.conversation}
    active_records.update(n.gift.record.id for n in residents if n.gift)
    if world.snake.conversation:
        active_records.add(world.snake.conversation.record.id)
    if world.player_interaction and world.player_interaction.record_id:
        active_records.add(world.player_interaction.record_id)
    fact_ids.update(f.id for f in world.facts if f.kind == "conversation_seen" and f.conversation_id in active_records)
    fact_ids.update(fact.id for fact in world.facts if fact.source_id in current_goals or fact.kind == "killing")
    for npc in residents:
        if npc.conversation and npc.conversation.information_context:
            disclosure_ids.update(b.disclosure_id for entries in npc.conversation.information_context.beliefs.values()
                                  for b in entries if b.disclosure_id)
            for option in npc.conversation.information_context.options:
                fact_ids.add(option.fact_id)
                if option.id in claims:
                    claim_ids.add(option.id)
                if option.parent_claim_id:
                    claim_ids.add(option.parent_claim_id)
            disclosure_ids.update(d.id for d in world.disclosures if d.conversation_id == npc.conversation.id)
        if npc.conversation and npc.conversation.focus_claim_id:
            claim_ids.add(npc.conversation.focus_claim_id)
    by_id = {d.id: d for d in world.disclosures}
    pending = list(disclosure_ids)
    while pending:
        disclosure = by_id[pending.pop()]
        claim_ids.add(disclosure.claim_id)
        if disclosure.source_disclosure_id and disclosure.source_disclosure_id not in disclosure_ids:
            disclosure_ids.add(disclosure.source_disclosure_id)
            pending.append(disclosure.source_disclosure_id)
    claim_ids.update(claim.id for claim in world.claims[-100:])
    pending = list(claim_ids)
    while pending:
        claim = claims[pending.pop()]
        fact_ids.add(claim.fact_id)
        if claim.parent_claim_id and claim.parent_claim_id not in claim_ids:
            claim_ids.add(claim.parent_claim_id)
            pending.append(claim.parent_claim_id)
    world.disclosures[:] = [d for d in world.disclosures if d.id in disclosure_ids]
    world.claims[:] = [claim for claim in world.claims if claim.id in claim_ids]
    world.facts[:] = [fact for fact in world.facts if fact.id in fact_ids]


def pending_promises(world, promisor_id=None):
    return [promise for promise in world.promises if promise.status == "pending"
            and (promisor_id is None or promise.promisor_id == promisor_id)]


def assign_goal(npc, world):
    from autonomy import company_point

    candidates = [other for other in world.npcs if other is not npc
                  and company_point(npc, other) is not None]
    choices = []
    generous = npc.preferences["generosity"] >= 0.55
    for target in candidates:
        opinion = target.opinion_score(npc.id)
        if target.inventory.get("fruit", 0) or target.can_reach(world.tree.interaction_point):
            choices.append((1 if generous else 3, "obtain_apple", target, opinion))
        if npc.inventory.get("fruit", 0) or npc.can_reach(world.tree.interaction_point):
            choices.append((3 if generous else 1, "help_with_apple", target, opinion))
        if opinion < 0:
            choices.append((2 + npc.preferences["sociability"], "repair_relationship", target, opinion))
    choices = [(*choice, None, None) for choice in choices]
    sabotage = []
    if npc.preferences["generosity"] < 0.5:
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            fact = fact_by_id(world, claim.fact_id)
            subject = world.participant_by_id(claim.subject_id)
            if subject is npc or subject is world.snake or subject is world.player or subject.departure or belief.stance == "disbelieved":
                continue
            for target in candidates:
                if target is not subject:
                    baseline = target.opinion_score(subject.id)
                    polarity = fact.kind not in ("apple_given", "work_given", "promise_kept")
                    related = [(b, claim_by_id(world, b.claim_id)) for b in target.beliefs
                               if claim_by_id(world, b.claim_id).fact_id == fact.id]
                    blocked = any(
                        b.direct and c.affirmed != polarity or b.assessed and c.affirmed == polarity
                        or c.affirmed == polarity and b.stance != "believed"
                        and (c.id == claim.id or not belief.direct or not polarity)
                        for b, c in related)
                    if baseline > -96 and not blocked:
                        sabotage.append((2, "sabotage_relationship", target, baseline, subject, claim))
    choices.extend(npc.rng.sample(sabotage, min(3, len(sabotage))))
    if not choices:
        npc.goal = None
        return
    _, kind, target, baseline, third, claim = npc.rng.choices(choices, weights=[choice[0] for choice in choices])[0]
    descriptions = {
        "obtain_apple": (f"Obtain an apple from {target.name}",
                         "I want a sign that someone will look out for me." if generous
                         else "I want to secure an apple while keeping my own resources.",
                         "Ask honestly and offer consideration in return." if generous
                         else "Try flattery and bargaining; keep my self-interest private."),
        "help_with_apple": (f"Give {target.name} an apple",
                            "I want them to feel supported." if generous
                            else "I want my generosity to make a good impression.",
                            "Offer practical help without demanding repayment." if generous
                            else "Make the favor memorable, but accept resistance."),
        "repair_relationship": (f"Improve {target.name}'s opinion of me by {GOAL_IMPROVEMENT} points",
                                "I want to rebuild a strained connection.",
                                "Listen and acknowledge specific history." if generous
                                else "Find common ground and present my side persuasively."),
    }
    if third:
        descriptions["sabotage_relationship"] = (
            f"Lower {target.name}'s opinion of {third.name} by 4 points through a supported account",
            "I want to weaken their connection so my own influence matters more.",
            "Use an account I know, or a permitted denial of it; persuasion can fail.")
    objective, motive, approach = descriptions[kind]
    npc.goal = Goal(uuid4().hex, kind, target.id, objective, motive, approach,
                    "Respect refusal; use only supported accounts or permitted denials, without inventing other events.",
                    world.phase_index, world.phase_index + len(PHASES), baseline,
                    third_npc_id=third.id if third else None, claim_id=claim.id if claim else None)
    record_intention(npc, world)


def resolve_goal(npc, world, status, outcome, record=None):
    goal = npc.goal
    if goal is None or goal.status != "active":
        return
    target = world.participant_by_id(goal.target_id)
    goal.status, goal.outcome = status, outcome
    goal.resolved_phase = world.phase_index
    kind = "goal_completed" if status == "completed" else "goal_cancelled" if status == "cancelled" else "goal_failed"
    event = world.publish(kind,
                          (npc, target), f"{npc.name}: {goal.objective}. {outcome}", record)
    goal.resolution_event_id = event.id
    remember(npc, target, event, kind,
             f"My objective: {goal.objective}. {outcome}", 2 if status == "completed" else 0 if status == "cancelled" else -2)


def check_relationship_goals(world, opinion_owner, record=None):
    for npc in world.npcs:
        goal = npc.goal
        if (goal and goal.status == "active" and goal.kind == "repair_relationship"
                and goal.target_id == opinion_owner.id and world.phase_index <= goal.deadline
                and opinion_owner.opinion_score(npc.id)
                >= goal.baseline_opinion + GOAL_IMPROVEMENT):
            resolve_goal(npc, world, "completed", "Their opinion improved through a real interaction.", record)


def permitted_proposals(conversation):
    if conversation.purpose in ("food_request", "confrontation") or conversation.opening_kind == "orientation" or conversation.initiator is conversation.initiator.world.snake or conversation.recipient is conversation.initiator.world.snake:
        return []
    world = conversation.initiator.world
    proposals = []
    goal = conversation.initiator.goal
    goal_pair = None
    if goal and goal.status == "active" and goal.target_id == conversation.recipient.id:
        if goal.kind == "obtain_apple":
            goal_pair = (conversation.recipient.id, conversation.initiator.id)
        elif goal.kind == "help_with_apple":
            goal_pair = (conversation.initiator.id, conversation.recipient.id)
    for giver, recipient in ((conversation.initiator, conversation.recipient),
                             (conversation.recipient, conversation.initiator)):
        if goal_pair and (giver.id, recipient.id) != goal_pair:
            continue
        pending = pending_promises(world, giver.id)
        if len(pending) < 2 and not any(p.recipient_id == recipient.id for p in pending):
            role = "initiator" if giver is conversation.initiator else "recipient"
            proposals.append(Proposal(f"{conversation.id}:{role}_gives", giver.id, recipient.id, world.phase_index + 1))
    return proposals


def proposals_from_snapshot(snapshot):
    """Project proposal boundary data, leaving prompt-only labels at the boundary."""
    return [Proposal(row["id"], row["promisor_id"], row["recipient_id"], row["deadline"])
            for row in snapshot.get("permitted_proposals", [])]


def validate_agreements(agreements, proposals, turns, initiator_id, recipient_id, *, current_phase=None,
                        dialogue_validation=True):
    """Bound offers and acceptances to ordered speech by the correct participants."""
    permitted = {proposal.id: proposal for proposal in proposals}
    seen = set()
    for agreement in agreements:
        proposal = permitted.get(agreement.proposal_id)
        if proposal is None or agreement.proposal_id in seen:
            raise ValueError("The agreement refers to an unavailable or repeated proposal.")
        seen.add(agreement.proposal_id)
        if not 0 <= agreement.offer_turn < agreement.accept_turn < len(turns):
            raise ValueError(
                f"Invalid promise indices: offer_turn={agreement.offer_turn}, accept_turn={agreement.accept_turn}. "
                f"This exchange has {len(turns)} turns, indexed 0 through {len(turns) - 1}. "
                "Choose an existing giver's offer followed by the receiver's acceptance in a later turn, "
                "or use accepted_proposals=[] if no agreement was made."
            )
        speakers = {"initiator": initiator_id, "recipient": recipient_id}
        if (speakers[turns[agreement.offer_turn].speaker] != proposal.promisor_id
                or speakers[turns[agreement.accept_turn].speaker] != proposal.recipient_id):
            raise ValueError("Only the apple's giver can offer and its recipient can accept.")
        if dialogue_validation:
            validate_apple_promise(turns[agreement.offer_turn].text, turns[agreement.accept_turn].text, proposal.deadline,
                                   current_phase=current_phase)


def validate_live_agreements(conversation, response, *, dialogue_validation=True):
    world = conversation.initiator.world
    pair = (conversation.initiator, conversation.recipient)
    if any(npc.conversation is not conversation or world.actor_by_id(npc.id) is not npc or npc.departure or npc.location != pair[0].location for npc in pair):
        raise ValueError("The conversation participants are no longer available.")
    if conversation.context_references != conversation_references(conversation):
        raise ValueError("An objective or promise changed while the reply was being generated. Please try again.")
    context = conversation.information_context
    if context is not None:
        # Keep the original permitted choices. Reselecting them after a nearby
        # event rejected valid replies just because new accounts ranked higher.
        relevant_facts = {option.fact_id for option in context.options
                          if any(option.id in ids for ids in context.allowed.values())}
        for role, npc in (("initiator", pair[0]), ("recipient", pair[1])):
            before = {belief.claim_id: belief for belief in context.beliefs[role]}
            current = {belief.claim_id: belief for belief in npc.beliefs}
            changed = any(current.get(claim_id) != belief for claim_id, belief in before.items())
            related = any(b.claim_id not in before and claim_by_id(world, b.claim_id).fact_id in relevant_facts
                          for b in npc.beliefs)
            opinions = {other.id: npc.opinion_score(other.id)
                        for other in world.actors if other is not npc}
            if changed or related or opinions != context.opinions[role] or can_deceive(npc) != context.can_lie[role]:
                raise ValueError("A speaker's knowledge or opinion changed during generation. Please try again.")
    validate_information(response, context, names=[actor.name for actor in world.actors],
                         participant_names={"initiator": pair[0].name, "recipient": pair[1].name},
                         dialogue_validation=dialogue_validation, actor_references=account_references(conversation.name_actors))
    validate_agreements(response.accepted_proposals, conversation.proposals,
                        response.turns, pair[0].id, pair[1].id, current_phase=world.phase_index,
                        dialogue_validation=dialogue_validation)
    pending = [(p.promisor_id, p.recipient_id) for p in pending_promises(world)]
    for agreement in response.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        if (conversation.purpose == "food_request" or world.phase_index > proposal.deadline
                or {proposal.promisor_id, proposal.recipient_id} != {npc.id for npc in pair}
                or (proposal.promisor_id, proposal.recipient_id) in pending
                or sum(giver == proposal.promisor_id for giver, _ in pending) >= 2):
            raise ValueError("The promise's deadline or participants changed. Please try the conversation again.")
        pending.append((proposal.promisor_id, proposal.recipient_id))


def conversation_references(conversation):
    pair = (conversation.initiator, conversation.recipient)
    ids = {npc.id for npc in pair}
    # Only the objective that actually initiated this meeting can invalidate it.
    # Other personal goals and promises to third parties may progress elsewhere.
    references = {npc.goal.id: npc.goal.status for npc in pair
                  if npc.goal and npc.goal.id == conversation.goal_id}
    references.update({p.id: p.status for p in pair[0].world.promises
                       if ids == {p.promisor_id, p.recipient_id}})
    return references


def commit_agreements(conversation):
    world = conversation.initiator.world
    for agreement in conversation.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        promise = Promise(proposal.id, proposal.promisor_id, proposal.recipient_id,
                          conversation.record.id, world.phase_index, proposal.deadline)
        world.promises.append(promise)
        giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
        text = f"{giver.name} promised {recipient.name} one apple by {phase_label(promise.deadline)}."
        event = world.publish("promise_created", (giver, recipient), text, conversation.record)
        conversation.record.outcomes.append(text)
        for npc, partner in ((giver, recipient), (recipient, giver)):
            remember(npc, partner, event, "promise_created", text, 1)


def apply_opinion_effects(npc):
    while npc.pending_social_opinions:
        effect = npc.pending_social_opinions[0]
        target = npc.world.participant_by_id(effect.target_id)
        if target is npc.world.player:
            from player import update_player_opinion
            update_player_opinion(npc, effect.delta, effect.reason)
            npc.pending_social_opinions.pop(0)
            continue
        previous = npc.opinions.get(target.id, {"opinion": 0, "reason": ""})
        # Keep the specific unresolved outcome when later impressions are added.
        reason = effect.reason + (f" Earlier: {previous['reason']}" if previous["reason"] else "")
        try:
            npc.remember_opinions({target.id: {
                "opinion": max(-100, min(100, previous["opinion"] + effect.delta)), "reason": reason[:400]}})
        except ValueError:
            return False
        npc.pending_social_opinions.pop(0)
        if effect.claim_id:
            record = next(entry for entry in npc.world.history if entry.id == effect.conversation_id)
            actual_delta = npc.opinions[target.id]["opinion"] - previous["opinion"]
            text = f"{npc.name}'s opinion of {target.name} changed {actual_delta:+d} after assessing an allegation. {effect.reason}"
            event = npc.world.publish("reputation_changed", (npc, target), text, record)
            record.outcomes.append(text)
            remember(npc, target, event, "claim_opinion", effect.reason, max(-4, min(4, actual_delta)))
            check_sabotage_goal(npc.world, effect, npc, actual_delta, record)
    return True


def resolve_promise(world, promise, status, transfer=None):
    if promise.status != "pending":
        return
    giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
    record = next(entry for entry in world.history if entry.id == promise.conversation_id)
    promise.status, promise.resolved_phase = status, world.phase_index
    text = (f"{giver.name} kept the apple promise to {recipient.name} with an actual handoff."
            if status == "fulfilled" else
            f"{giver.name} missed the apple promise to {recipient.name}, due {phase_label(promise.deadline)}; the reason is unknown.")
    event = world.publish("promise_fulfilled" if status == "fulfilled" else "promise_broken",
                          (giver, recipient), text, record)
    promise.consequence_event_id = event.id
    promise.resolution_event_id = transfer.id if transfer is not None else event.id
    promise.outcome = text
    record.outcomes.append(text)
    for npc, partner in ((giver, recipient), (recipient, giver)):
        if npc is world.player:
            continue
        remember(npc, partner, event, "promise_fulfilled" if status == "fulfilled" else "promise_broken",
                 text, 4 if status == "fulfilled" else -5)
    if recipient is not world.player:
        recipient.pending_social_opinions.append(OpinionEffect(event.id, giver.id, 4 if status == "fulfilled" else -6, text))
        apply_opinion_effects(recipient)


def on_transfer(world, giver, recipient, quantity, event, record):
    # A transfer is consumed by the oldest matching commitments once only.
    matching = sorted((p for p in pending_promises(world, giver.id)
                       if p.recipient_id == recipient.id and p.created_phase <= world.phase_index <= p.deadline),
                      key=lambda p: p.created_phase)
    for promise in matching[:quantity]:
        resolve_promise(world, promise, "fulfilled", event)
    for npc, partner, kind in ((giver, recipient, "help_with_apple"), (recipient, giver, "obtain_apple")):
        if npc is world.player:
            continue
        goal = npc.goal
        if (goal and goal.status == "active" and goal.kind == kind and goal.target_id == partner.id
                and goal.created_phase <= world.phase_index <= goal.deadline):
            resolve_goal(npc, world, "completed", "The apple was actually transferred.", record)


def advance_social_phase(world):
    for promise in world.promises:
        if promise.status == "pending" and promise.deadline < world.phase_index:
            resolve_promise(world, promise, "broken")
    for npc in world.npcs:
        goal = npc.goal
        if goal and goal.status == "active" and goal.deadline < world.phase_index:
            resolve_goal(npc, world, "failed", "The deadline passed without completing the objective.")
        elif goal is None or (goal.status != "active" and goal.resolved_phase < world.phase_index):
            assign_goal(npc, world)
        apply_opinion_effects(npc)
    trim_information(world)


def private_context(npc, partner_id=None):
    world = npc.world
    promises = [p for p in world.promises if npc.id in (p.promisor_id, p.recipient_id)] if world else []
    promises.sort(key=lambda p: (p.status == "pending", partner_id in (p.promisor_id, p.recipient_id), p.created_phase), reverse=True)
    known = []
    if world:
        for belief in selected_beliefs(npc, partner_id):
            claim = claim_by_id(world, belief.claim_id)
            known.append({**account_view(world, claim, belief), "already_assessed": belief.assessed})
    goal_view = asdict(npc.goal) if npc.goal else None
    if goal_view:
        goal_view.pop("baseline_opinion")
    partner = world.participant_by_id(partner_id) if world and partner_id else None
    return {"dangerous_partner": bool(partner and regards_as_dangerous(npc, partner)),
            "owner_id": npc.id, "visibility": "Only this speaker knows their private motive and memories.",
            "concern": ({"kind": npc.current_concern.kind, "claim_id": npc.current_concern.claim_id,
                         "guidance": concern_topic(npc, npc.current_concern)} if npc.current_concern else None),
            "goal": goal_view,
            "known_accounts": known,
            "promises": [dict(asdict(p), deadline_label=phase_label(p.deadline)) for p in promises[:8]]}


def pair_objective(world, npc, other):
    pair = {npc.id, other.id}
    def discussed(topic):
        return any(r.turns and set(r.participants) == {npc.name, other.name} and r.topic == topic for r in world.history)
    promises = [p for p in world.promises if {p.promisor_id, p.recipient_id} == pair]
    pending = [p for p in promises if p.status == "pending"]
    if pending:
        promise = min(pending, key=lambda p: p.deadline)
        giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
        topic = f"Arrange {giver.name}'s promised apple for {recipient.name}, due {phase_label(promise.deadline)}"
        if not discussed(topic):
            return 3, topic
    if promises and promises[-1].status == "broken":
        topic = f"Discuss the missed apple promise: {promises[-1].outcome} Seek an explanation or practical repair"
        if not discussed(topic):
            return 2, topic
    for belief in selected_beliefs(npc, other.id):
        claim = claim_by_id(world, belief.claim_id)
        contrary = [b for b in npc.beliefs if claim_by_id(world, b.claim_id).fact_id == claim.fact_id
                    and claim_by_id(world, b.claim_id).affirmed != claim.affirmed]
        if contrary and other.id in (belief.source_id, claim.originator_id):
            topic = f"Ask {other.name} to explain conflicting accounts about {world.participant_by_id(claim.subject_id).name}"
            if not discussed(topic):
                return 2, topic
    danger = undisclosed_danger(npc)
    if danger:
        return 2, danger_topic(npc, danger)
    goal = npc.goal
    if goal and goal.status == "active" and goal.target_id == other.id:
        return 1, f"{goal.objective}, by {phase_label(goal.deadline)}"
    return 0, ""


def goal_attempt_cooling_down(npc, other, world):
    goal = npc.goal
    return bool(goal and goal.status == "active" and goal.target_id == other.id
                and goal.last_attempt_phase is not None and world.phase_index < goal.last_attempt_phase + 2)


def learn_account(world, speaker, listener, claim_id, record, turn, stance, *, private=False):
    """Disclose an existing account only to its direct recipient, with its source chain."""
    source = next(b for b in speaker.beliefs if b.claim_id == claim_id)
    claim = claim_by_id(world, claim_id)
    fact = fact_by_id(world, claim.fact_id)
    event = world.publish("claim_shared", (speaker, listener),
        f"{speaker.name} disclosed {'private information' if private else 'a sourced account'} to {listener.name}.", record)
    disclosure = Disclosure(uuid4().hex, claim_id, record.id, speaker.id, listener.id, turn,
                            "share", stance, source.disclosure_id, world.phase_index, event.id)
    world.disclosures.append(disclosure)
    existing = next((b for b in listener.beliefs if b.claim_id == claim_id), None)
    learned = receive_belief(existing, listener_id=listener.id, claim_id=claim_id, stance=stance,
                             source_id=speaker.id, disclosure_id=disclosure.id, phase=world.phase_index)
    if existing is None:
        listener.beliefs.append(learned)
    if listener is not world.snake:
        remember(listener, speaker, event, "heard_claim",
                 f"{speaker.name}'s account, originally from {world.participant_by_id(claim.originator_id).name}: {account_content(world, fact, claim.affirmed)}", 0)
    return disclosure


NPC_ARRIVAL_MEMORY = (
    "You remember waking up one day in an elevator that was going up. "
    "When its doors opened, you found yourself on the first floor of the apartment building."
)


def orientation_speech(texts):
    pattern = (r"\b(?:arriv\w*|welcome|layout|hallway|rooms?|garden exit|bridge|elevator|where (?:are|am) (?:we|i)|where (?:we|i) (?:are|am)|"
               r"why (?:we|i) (?:are|am) here|how (?:did|do) (?:we|i) get|how (?:we|i) got here|"
               r"(?:get|getting|go|going|way|back) home|"
               r"(?:remember|recall).{0,35}(?:how|why|when).{0,25}here|"
               r"this place|where we are|surroundings|bearings|safety|shelter|"
               r"(?:find|finding|seek|seeking|need) (?:some )?help|safe here)\b")
    return any(re.search(pattern, sentence, re.I)
               for text in texts for sentence in split_sentences(unquoted_speech(text)))


def validate_orientation(response, snapshot):
    # A structured claim alone is insufficient. Derive the accepted record's
    # marker from actual speech, including when the model omitted the flag.
    discussed = orientation_speech(turn.text for turn in response.turns)
    if snapshot.get("arrival_focus"):
        if any((response.accepted_proposals, response.disclosures, response.third_party_opinions, response.leverage)):
            raise ValueError("This orientation exchange does not authorize apple promises, private disclosures or leverage. Welcome the person and discuss the known layout or practical routines.")
        if snapshot.get("_dialogue_validation", True):
            if not discussed:
                raise ValueError("This is an orientation exchange. Discuss the known surroundings, practical routines or remembered elevator arrival, rather than pursuing a routine apple goal.")
    # This is derived bookkeeping, not a reason to regenerate otherwise valid speech.
    response.orientation_discussed = discussed
    return discussed


def account_knowledge(belief, speaker_id, *, creator_id=None, fallback_source=None):
    """Project only this knower's evidence; newly heard reports own no private stance."""
    return dict(immediate_source_id=belief.source_id if belief else fallback_source,
                stance=belief.stance if belief else "pending your reaction",
                basis=("firsthand" if belief and belief.direct else
                       "deliberate denial; private to this speaker" if creator_id == speaker_id else
                       "own assertion" if belief and belief.source_id == speaker_id else "hearsay"))


def build_account_view(account, knowledge, actors, speaker_id, listener_id):
    """Calculate attribution once; callers explicitly select their request fields.

    Selection and factual lookup belong to social/interaction code. This function
    receives only the selected account and the speaker's knowledge, never world
    truth or the listener's private context. IDs determine ownership; names are
    presentation. A private motive's target is not evidence of an encounter.
    """
    view = deepcopy(account)
    if knowledge is not None:
        view.update(knowledge)
    for field in ("subject", "object", "original_speaker", "immediate_source"):
        target = view.get(field + "_id")
        if field + "_id" in view:
            view[field + "_name"] = actors.get(target, {}).get("name", view.get(field + "_name", ""))
        view[field + "_relation"] = ("speaker" if target == speaker_id else
                                    "listener" if target == listener_id else "another person")
    view["knowledge_owner_name"] = actors.get(speaker_id, {}).get("name", view.get("knowledge_owner_name", ""))
    if view.get("kind") == "private_motive":
        view.pop("object_relation", None)
    return view


def resident_information(snapshot, draft, *, incomplete=True):
    return validate_information(SimpleNamespace(**draft), snapshot.get("_information_validation"),
        names=(*[p["name"] for p in snapshot.get("_name_actors", {}).values()], SNAKE_NAME),
        participant_names={role: snapshot[role]["name"] for role in ("initiator", "recipient")},
        dialogue_validation=snapshot.get("_dialogue_validation", True),
        actor_references=account_references(snapshot.get("_name_actors", {})), incomplete=incomplete)


def pending_reactions(snapshot, draft, role):
    info = information_from_dict(snapshot["_information_validation"]) if snapshot.get("_information_validation") else None
    if info is None:
        return []
    beliefs, _, _ = resident_information(snapshot, draft)
    options = {option.id: option for option in info.options}
    pending = []
    for entry in draft["disclosures"]:
        source_role = draft["turns"][entry.turn].speaker
        if source_role == role or entry.listener_stance is not None:
            continue
        option = options[entry.claim_id]
        source = beliefs[source_role].get(entry.claim_id)
        owner = build_account_view(asdict(option), None, snapshot.get("_name_actors", {}),
                                   info.participants[role], info.participants[source_role])
        pending.append(dict(disclosure_turn=entry.turn, claim_id=entry.claim_id, assertion=option.assertion,
                            subject_name=owner["subject_name"], subject_relation=owner["subject_relation"],
                            stances=listener_stances(option, beliefs[role], options,
                                                    firsthand_source=bool(source and source.direct))))
    return pending


def resident_accounts_for_turn(accounts, info, role, draft, catalog, *, concern_id=None):
    """Offer one own private fact, keeping heard accounts and garden evidence."""
    options = {option.id: option for option in info.options}
    owner = info.participants[role]
    personal = [a for a in accounts if a["subject_id"] == owner and a["kind"] in ("former_life", "private_motive")]
    if not personal:
        return accounts
    personal_facts = {options[a["claim_id"]].fact_id for a in personal}
    heard = draft["turns"][-1].text if draft["turns"] and draft["turns"][-1].speaker != role else ""
    names = [p["name"] for p in catalog.values()]
    required = info.focus_claim_id if role == "initiator" else None
    selected = next((options[a["claim_id"]].fact_id for a in personal if a["claim_id"] == required), None)
    if selected is None:
        selected = next((options[a["claim_id"]].fact_id for a in personal if mentions_account(heard, a["assertion"], names)), None)
    if selected is None:
        selected = next((options[d.claim_id].fact_id for d in reversed(draft["disclosures"])
                         if draft["turns"][d.turn].speaker == role and options[d.claim_id].fact_id in personal_facts), None)
    if selected is None:
        selected = next((options[a["claim_id"]].fact_id for a in personal if a["claim_id"] == concern_id), None)
    if selected is None:
        used = {options[d.claim_id].fact_id for d in draft["disclosures"] if draft["turns"][d.turn].speaker == role}
        available = [o for o in info.options if o.fact_id in personal_facts - used and o.creator_id is None]
        choice = next((o for o in available if o.kind == "private_motive"), next(iter(available), None))
        selected = choice.fact_id if choice else None
    return [a for a in accounts if options[a["claim_id"]].fact_id not in personal_facts
            or options[a["claim_id"]].fact_id == selected]


def resident_turn_choices(snapshot, draft, role, private, *, closing=False, assessment=False):
    """Select this actor's accounts and eligible effects from the accepted prefix."""
    index = len(draft["turns"])
    other = "recipient" if role == "initiator" else "initiator"
    person, partner = snapshot[role], snapshot[other]
    catalog = snapshot.get("_name_actors", {})
    orientation = snapshot.get("arrival_focus", False)
    pending = pending_reactions(snapshot, draft, role)
    projection = resident_information(snapshot, draft)
    info = information_from_dict(snapshot["_information_validation"]) if snapshot.get("_information_validation") else None
    accounts = []
    if info and not orientation:
        beliefs, acquired, heard = projection
        options = {option.id: option for option in info.options}
        original_views = {a["claim_id"]: a for a in private.get("permitted_accounts", [])}
        for claim_id in dict.fromkeys([*info.allowed[role], *acquired[role], *[p["claim_id"] for p in pending]]):
            option = options[claim_id]
            belief = beliefs[role].get(claim_id)
            # An account newly heard is hearsay with this speaker as its knowledge
            # owner. Never copy the discloser's private motive, stance or evidence.
            view = deepcopy(original_views.get(claim_id, {}))
            if not view:
                disclosed = next((d for d in reversed(draft["disclosures"]) if d.claim_id == claim_id), None)
                if disclosed is not None:
                    supplied = next((a for source_role in ("initiator", "recipient")
                                     for a in snapshot[source_role].get("private_context", {}).get("permitted_accounts", [])
                                     if a["claim_id"] == claim_id), {})
                    view = {key: deepcopy(supplied[key]) for key in ("object_id", "object_name", "event_time") if key in supplied}
            selected = {**view, "claim_id": claim_id, "assertion": option.assertion, "kind": option.kind, "quantity": option.quantity,
                        "subject_id": option.subject_id, "original_speaker_id": option.originator_id}
            projected = build_account_view(selected, account_knowledge(belief, actor_id(person),
                creator_id=option.creator_id, fallback_source=actor_id(partner)), catalog,
                actor_id(person), actor_id(partner))
            view.update(claim_id=claim_id, assertion=option.assertion, kind=option.kind,
                        subject_id=option.subject_id, subject_name=projected["subject_name"],
                        original_speaker_id=option.originator_id, original_speaker_name=projected["original_speaker_name"],
                        immediate_source_id=projected["immediate_source_id"], immediate_source_name=projected["immediate_source_name"],
                        knowledge_owner_name=person["name"], basis=projected["basis"], stance=projected["stance"])
            for field in ("subject", "object", "original_speaker", "immediate_source"):
                if view.get(field + "_id") and field + "_relation" in projected:
                    view[field + "_relation"] = projected[field + "_relation"]
            if option.kind == "private_motive":
                for field in ("object_id", "object_name", "object_relation"):
                    view.pop(field, None)  # A goal target is not evidence of an event with that person.
            accounts.append(view)
        accounts = resident_accounts_for_turn(accounts, info, role, draft, catalog,
                                               concern_id=(private.get("concern") or {}).get("claim_id"))
    choices = dict(offers=[], acceptances=[], disclosures=[], third_party_opinions=[], leverage=[])
    if not orientation and snapshot.get("purpose") != "food_request":
        used = {a.proposal_id for a in draft["accepted_proposals"]}
        for proposal in snapshot.get("permitted_proposals", []):
            if proposal["id"] in used:
                continue
            if (not closing and not assessment and proposal["promisor_id"] == actor_id(person)
                    and proposal["id"] not in draft["offers"] and resident_effect_capacity(draft)["offers"] > 0):
                choices["offers"].append(proposal["id"])
            if proposal["recipient_id"] == actor_id(person) and proposal["id"] in draft["offers"]:
                choices["acceptances"].append(proposal["id"])
        if info:
            beliefs, acquired, heard = projection
            seen = {(draft["turns"][d.turn].speaker, d.claim_id) for d in draft["disclosures"]}
            choices["disclosures"] = [a["claim_id"] for a in accounts if (role, a["claim_id"]) not in seen]
            # Pending reactions are chosen before this reply's effects are checked.
            heard_ids = {claim_id for (hearer, claim_id), turn in heard.items() if hearer == role and turn < index}
            heard_ids.update(p["claim_id"] for p in pending)
            pending_stances = {p["claim_id"]: p["stances"] for p in pending}
            for account in accounts:
                claim_id = account["claim_id"]
                option = options[claim_id]
                related = related_beliefs(option, beliefs[role].values(), options)
                if reputation_assessment_available(option, related, options,
                        heard=claim_id in heard_ids, participants=info.participants.values(),
                        believed=account["stance"] == "believed" or "believed" in pending_stances.get(claim_id, []),
                        assessed=any(draft["turns"][e.turn].speaker == role
                                     and options[e.claim_id].fact_id == option.fact_id for e in draft["third_party_opinions"])):
                    choices["third_party_opinions"].append(claim_id)
            choices["leverage"] = [a["claim_id"] for a in accounts if a["kind"] in LEVERAGE_ACCOUNT_KINDS
                and (a["claim_id"] in beliefs[role] or a["claim_id"] in heard_ids)
                and a["claim_id"] not in {entry.claim_id for entry in draft["leverage"]}]
    if not orientation and snapshot.get("purpose") != "food_request":
        focus = (snapshot.get("_information_validation") or {}).get("focus_claim_id")
        if role == "initiator" and index == 0 and focus:
            focus_fact = next(o.fact_id for o in info.options if o.id == focus)
            choices["disclosures"] = [claim for claim in choices["disclosures"]
                                      if next(o.fact_id for o in info.options if o.id == claim) == focus_fact]
    if any(snapshot[r].get("kind") == "python" for r in ("initiator", "recipient")):
        choices["leverage"] = []
    return info, accounts, choices, pending


def validate_player_promise_offers(record, promises, owner_ids, player_id, now, player_offer_ids):
    if len(record.player_promise_offers) > CONVERSATION_MAX_AGREEMENTS:
        raise ValueError("Too many player apple offers in one conversation.")
    for offer in record.player_promise_offers:
        if (offer.id in player_offer_ids
                or offer.giver_id not in owner_ids or offer.recipient_id not in owner_ids
                or offer.giver_id == offer.recipient_id
                or player_id not in (offer.giver_id, offer.recipient_id)
                or offer.status not in ("offered", "accepted", "declined", "expired", "closed")
                or not 0 <= offer.created_phase <= now or offer.deadline < offer.created_phase
                or not 0 <= offer.offer_turn < len(record.turns)
                or record.turns[offer.offer_turn].speaker_id != offer.giver_id):
            raise ValueError("Invalid player apple offer authority or source.")
        player_offer_ids.add(offer.id)
        from dialogue_rules import validate_apple_offer
        validate_apple_offer(record.turns[offer.offer_turn].text, offer.deadline, current_phase=offer.created_phase)
        if offer.status == "accepted":
            turn = offer.accept_turn
            if (type(turn) is not int or not offer.offer_turn < turn < len(record.turns)
                    or record.turns[turn].speaker_id != offer.recipient_id
                    or not any(p.id == offer.id and p.conversation_id == record.id
                               and p.promisor_id == offer.giver_id and p.recipient_id == offer.recipient_id
                               and p.deadline == offer.deadline for p in promises)):
                raise ValueError("An accepted player apple promise needs its actual ordered acceptance.")
            validate_apple_promise(record.turns[offer.offer_turn].text, record.turns[turn].text,
                                   offer.deadline, current_phase=offer.created_phase)
        elif offer.accept_turn is not None:
            raise ValueError("An unaccepted apple offer cannot retain an acceptance.")


def validate_saved_concern(concern, owner, claims, facts):
    if concern is None:
        return
    if concern.claim_id:
        claim = claims.get(concern.claim_id)
        belief = next((b for b in owner.beliefs if b.claim_id == concern.claim_id), None)
        if claim is None or belief is None:
            raise ValueError("A concern requires its owner's existing knowledge.")
        fact = facts[claim.fact_id]
        if concern.kind == "recollection" and not owns_recollection(owner.id, belief, claim, fact):
            raise ValueError("A recollection concern requires the owner's direct former-life account.")


def validate_social_records(facts, claims, disclosures, events, history, residents, snake, ids, live_ids, day, now):
    from conversation_history import source_record
    from constants import PLAYER_ID
    observed_conversations = set()
    for fact in facts.values():
        single_actor = fact.kind in ("arrival", "departure", "apple_eaten", "harvest")
        if (fact.subject_id not in ids or fact.object_id not in ids
                or (fact.subject_id == fact.object_id) != single_actor
                or fact.phase > now or not set(fact.witness_ids) <= ids
                or PLAYER_ID in fact.witness_ids
                or len(set(fact.witness_ids)) != len(fact.witness_ids)
                or not fact.witness_ids and fact.kind not in ("conversation_seen", "harvest")):
            raise ValueError("Invalid fact participants or witnesses.")
        if fact.kind == "conversation_seen":
            record = source_record(history, fact.conversation_id)
            if (not any(t.speaker_id != PLAYER_ID for t in record.turns)
                    or fact.conversation_id in observed_conversations
                    or set(record.participant_ids) != {fact.subject_id, fact.object_id}
                    or set(fact.witness_ids) & set(record.participant_ids)):
                raise ValueError("A conversation observation requires accepted speech and actual bystanders.")
            observed_conversations.add(fact.conversation_id)
        if fact.kind in ("private_motive", "former_life"):
            if fact.witness_ids != [fact.subject_id] or fact.conversation_id is not None:
                raise ValueError("A private recollection or motive cannot be public.")
        elif fact.conversation_id is not None:
            source_record(history, fact.conversation_id)
        if fact.source_id in events:
            event = events[fact.source_id]
            expected = {"apple_given": "fruit_shared", "work_given": "work_shared", "apple_refused": "food_refused", "promise_kept": "promise_fulfilled"}.get(fact.kind, fact.kind)
            if (event.kind != expected or fact.phase != phase_index(event.day, event.phase)
                    or set(event.participants) != {fact.subject_id, fact.object_id}
                    or event.history != fact.conversation_id
                    or fact.kind == "harvest" and event.text != fact.summary):
                raise ValueError("A fact must match its actual source event.")
    for claim in claims.values():
        fact = facts.get(claim.fact_id)
        if fact is None or claim.subject_id != fact.subject_id or claim.originator_id not in ids or not fact.phase <= claim.phase <= now:
            raise ValueError("Invalid claim references.")
        if claim.parent_claim_id is None:
            if not claim.affirmed or claim.deliberate or claim.originator_id not in fact.witness_ids:
                raise ValueError("Firsthand accounts require an actual witnessed source.")
        else:
            parent = claims.get(claim.parent_claim_id)
            if parent is None or parent.fact_id != fact.id or not parent.affirmed or claim.affirmed or not claim.deliberate or parent.phase > claim.phase:
                raise ValueError("A denial must trace to an existing account.")
    seen = set()
    for disclosure in disclosures.values():
        record = source_record(history, disclosure.conversation_id)
        if (disclosure.claim_id not in claims or disclosure.speaker_id == disclosure.listener_id
                or not {disclosure.speaker_id, disclosure.listener_id} <= ids
                or {disclosure.speaker_id, disclosure.listener_id} != set(record.participant_ids)
                or disclosure.turn >= len(record.turns) or record.turns[disclosure.turn].speaker_id != disclosure.speaker_id
                or disclosure.phase > now):
            raise ValueError("Invalid disclosure actor, speech or account.")
        if disclosure.source_disclosure_id:
            source = disclosures.get(disclosure.source_disclosure_id)
            claim = claims[disclosure.claim_id]
            if source is None or source.id not in seen or source.listener_id != disclosure.speaker_id or source.claim_id not in (claim.id, claim.parent_claim_id):
                raise ValueError("Disclosure provenance must follow an earlier source.")
        elif disclosure.speaker_id != claims[disclosure.claim_id].originator_id:
            fact = facts[claims[disclosure.claim_id].fact_id]
            if disclosure.speaker_id not in fact.witness_ids:
                raise ValueError("A repeated account requires its actual source.")
        seen.add(disclosure.id)
    for actor in [*residents, snake]:
        validate_saved_concern(actor.current_concern, actor, claims, facts)
        if any(m.partner_id not in ids or phase_index(m.day, m.phase) > now for m in actor.memories):
            raise ValueError("Invalid personal memory ownership or phase.")
        for effect in actor.pending_social_opinions:
            if effect.target_id not in ids or effect.claim_id is not None and effect.claim_id not in claims:
                raise ValueError("Invalid queued social effect.")
        if len({b.claim_id for b in actor.beliefs}) != len(actor.beliefs):
            raise ValueError("Duplicate owned account.")
        for belief in actor.beliefs:
            claim = claims.get(belief.claim_id)
            if claim is None or belief.npc_id != actor.id or belief.source_id not in ids or belief.phase > now:
                raise ValueError("Invalid owned knowledge.")
            if belief.direct:
                if actor.id not in facts[claim.fact_id].witness_ids or belief.disclosure_id is not None:
                    raise ValueError("Firsthand knowledge requires witnessing.")
            elif belief.disclosure_id:
                source = disclosures.get(belief.disclosure_id)
                if source is None or source.listener_id != actor.id or source.claim_id != claim.id or source.speaker_id != belief.source_id:
                    raise ValueError("Hearsay requires its actual disclosure source.")
    for record in history.values():
        if record.purpose == "confrontation":
            grievance = claims.get(record.grievance_id)
            if (not grievance or not grievance.affirmed or grievance.subject_id != record.participant_ids[1]
                    or not set(record.participant_ids) <= {n.id for n in residents}):
                raise ValueError("A confrontation history needs two human residents and its original sourced grievance.")
            if record.turns and not any(d.conversation_id == record.id and d.speaker_id == record.participant_ids[0]
                    and claims[d.claim_id].fact_id == grievance.fact_id for d in disclosures.values()):
                raise ValueError("An accepted confrontation must retain its actual grievance discussion and date.")
        elif record.grievance_id is not None:
            raise ValueError("Only confrontation history owns a grievance.")
    goal_ids = set()
    for npc in residents:
        if npc.id in live_ids and npc.current_concern is None:
            raise ValueError("A living resident must have a concern.")
        if npc.arrival_day > day:
            raise ValueError("Invalid arrival day or personal memory.")
        if npc.departure and (phase_index(npc.departure.day, npc.departure.phase) > now or not any(f.source_id == npc.departure.event_id and f.kind == "killing" and f.object_id == npc.id for f in facts.values())):
            raise ValueError("An archived resident needs its actual departure record.")
        goal = npc.goal
        if goal:
            if goal.id in goal_ids or goal.target_id not in ids or goal.third_npc_id is not None and goal.third_npc_id not in ids or goal.created_phase > now or goal.deadline < goal.created_phase:
                raise ValueError("Invalid goal references.")
            goal_ids.add(goal.id)
            if goal.status == "active" and (npc.id not in live_ids or goal.target_id not in live_ids or goal.third_npc_id is not None and goal.third_npc_id not in live_ids):
                raise ValueError("An active goal can target only living residents.")
            if (goal.status == "active") != (goal.resolved_phase is None):
                raise ValueError("Invalid goal resolution.")
            if goal.claim_id is not None and goal.claim_id not in claims:
                raise ValueError("Unknown goal account.")


def validate_saved_promises(promises, history, ids, live_ids, player_id, now):
    from conversation_history import source_record
    promise_ids = set()
    for promise in promises:
        record = source_record(history, promise.conversation_id)
        if promise.id in promise_ids or promise.promisor_id == promise.recipient_id or not all(i in ids and i in record.participant_ids for i in (promise.promisor_id, promise.recipient_id)):
            raise ValueError("Invalid promise participants.")
        promise_ids.add(promise.id)
        if promise.status == "pending" and not {promise.promisor_id, promise.recipient_id} <= live_ids | {player_id}:
            raise ValueError("Impossible obligations must be cancelled.")
        if promise.deadline < promise.created_phase or promise.created_phase > now or (promise.status == "pending") != (promise.resolved_phase is None):
            raise ValueError("Invalid promise resolution.")


def grievance_against(npc, target):
    """A retained believed allegation, with its original source and uncertainty intact."""
    from constants import CONFRONTATION_START_OPINION, GRIEVANCE_FACT_KINDS, CONFRONTATION_GRIEVANCE_PHASE_PAUSE
    if npc.opinion_score(target.id) > CONFRONTATION_START_OPINION:
        return None
    for belief in reversed(npc.beliefs):
        claim = claim_by_id(npc.world, belief.claim_id)
        fact = fact_by_id(npc.world, claim.fact_id)
        if (belief.stance != "believed" or not claim.affirmed or fact.subject_id != target.id
                or fact.kind not in GRIEVANCE_FACT_KINDS):
            continue
        latest = next((r for r in reversed(npc.world.history) if r.purpose == "confrontation"
                       and r.between(npc, target) and r.grievance_id
                       and claim_by_id(npc.world, r.grievance_id).fact_id == fact.id and r.turns), None)
        if latest is not None:
            discussed_phase = max((d.phase for d in npc.world.disclosures
                if d.conversation_id == latest.id and d.speaker_id == npc.id
                and claim_by_id(npc.world, d.claim_id).fact_id == fact.id), default=npc.world.phase_index)
            if npc.world.phase_index - discussed_phase < CONFRONTATION_GRIEVANCE_PHASE_PAUSE:
                continue
        return claim.id
    return None


def confrontation_topic(npc, claim_id):
    belief = next(b for b in npc.beliefs if b.claim_id == claim_id)
    account = account_view(npc.world, claim_by_id(npc.world, claim_id), belief)
    return f"Seek an explanation for this unresolved account: {account['assertion']} Source: {account['immediate_source_name']}; stance: {account['stance']}."


def regards_as_dangerous(npc, target):
    return npc.opinion_score(target.id) < 0 and any(
        b.stance == "believed" and (c := claim_by_id(npc.world, b.claim_id)).affirmed
        and (f := fact_by_id(npc.world, c.fact_id)).kind == "killing" and f.subject_id == target.id
        for b in npc.beliefs)


def validate_escalation_snapshot(decision, snapshot, turns, opinion):
    choices = snapshot.get("escalation_choices")
    if decision is None:
        if choices:
            raise ValueError("The confrontation initiator must choose a peaceful leave or an available attack.")
        return
    if not choices or snapshot.get("purpose") != "confrontation":
        raise ValueError("Only an eligible confrontation initiator may select escalation.")
    if (decision.target_id != choices["target_id"] or decision.grievance_id != choices["grievance_id"]
            or decision.action not in choices["actions"]):
        raise ValueError("Escalation must match the supplied target, grievance and actions.")
    if not any(t.speaker == "recipient" for t in turns):
        raise ValueError("The target must have a real opportunity to answer.")
    if decision.action == "attack":
        from constants import CONFRONTATION_OPINION
        if opinion > CONFRONTATION_OPINION or not decision.evidence.strip() or not any(
                decision.evidence in t.text for t in turns if t.speaker == "recipient"):
            raise ValueError("An attack requires final hostility and exact evidence from the target's accepted answer.")


def validate_escalation(conversation, response):
    validate_escalation_snapshot(response.escalation,
        {"purpose": conversation.purpose, "escalation_choices": conversation.initiator.world.escalation_choices(conversation)},
        response.turns, response.initiator_opinion.opinion)
