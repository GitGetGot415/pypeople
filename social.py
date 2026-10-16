"""Private objectives and apple commitments, resolved only by committed world changes."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Literal
from uuid import uuid4

from conversation_history import remember

GoalKind = Literal["obtain_apple", "help_with_apple", "repair_relationship", "sabotage_relationship"]
FactKind = Literal["apple_given", "apple_refused", "promise_kept", "promise_broken", "private_motive"]
Stance = Literal["uncertain", "believed", "disbelieved"]
GOAL_IMPROVEMENT = 8


def phase_index(day, phase):
    return 3 * (day - 1) + ("day", "evening", "night").index(phase)


def phase_label(index):
    return f"Day {index // 3 + 1} / {('Day', 'Evening', 'Night')[index % 3]}"


@dataclass
class Goal:
    id: str
    kind: GoalKind
    target_id: str
    objective: str
    motive: str
    approach: str
    boundary: str
    created_phase: int
    deadline: int
    baseline_opinion: int = 0
    status: Literal["active", "completed", "failed"] = "active"
    resolution_event_id: str | None = None
    resolved_phase: int | None = None
    outcome: str = ""
    last_attempt_phase: int | None = None
    third_npc_id: str | None = None
    claim_id: str | None = None


@dataclass
class Proposal:
    id: str
    promisor_id: str
    recipient_id: str
    deadline: int


@dataclass
class Promise:
    id: str
    promisor_id: str
    recipient_id: str
    conversation_id: str
    created_phase: int
    deadline: int
    item: Literal["fruit"] = "fruit"
    quantity: Literal[1] = 1
    status: Literal["pending", "fulfilled", "broken"] = "pending"
    resolution_event_id: str | None = None
    resolved_phase: int | None = None
    outcome: str = ""
    consequence_event_id: str | None = None


@dataclass
class OpinionEffect:
    event_id: str
    target_id: str
    delta: int
    reason: str
    claim_id: str | None = None
    actor_id: str | None = None
    conversation_id: str | None = None
    phase: int | None = None


@dataclass
class Fact:
    id: str
    kind: FactKind
    subject_id: str
    object_id: str
    source_id: str
    conversation_id: str | None
    phase: int
    summary: str
    witness_ids: list[str]


@dataclass
class Claim:
    id: str
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    deliberate: bool
    phase: int
    conversation_id: str | None


@dataclass
class Belief:
    npc_id: str
    claim_id: str
    stance: Stance
    source_id: str
    disclosure_id: str | None
    direct: bool
    phase: int
    assessed: bool = False
    opinion_delta: int = 0


@dataclass
class Disclosure:
    id: str
    claim_id: str
    conversation_id: str
    speaker_id: str
    listener_id: str
    turn: int
    intent: Literal["share", "challenge", "lie"]
    stance: Stance
    source_disclosure_id: str | None
    phase: int
    event_id: str


@dataclass
class ClaimOption:
    id: str
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    creator_id: str | None
    kind: FactKind
    assertion: str


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
        return f"At {phase_label(fact.phase)}: {fact.summary}"
    subject, target = world.npc_by_id(fact.subject_id), world.npc_by_id(fact.object_id)
    action = {"apple_given": "give an apple to", "apple_refused": "refuse an apple to",
              "promise_kept": "keep the apple promise to", "promise_broken": "miss the apple promise to"}
    if fact.kind == "private_motive":
        return f"{subject.name} does not have the motive described in this account: {fact.summary}"
    return f"{subject.name} did not {action[fact.kind]} {target.name} in the exchange at {phase_label(fact.phase)}."


def add_fact(world, kind, subject, target, source_id, record, summary, witnesses):
    existing = next((fact for fact in world.facts if fact.source_id == source_id and fact.kind == kind), None)
    if existing is not None:
        return existing
    fact = Fact(uuid4().hex, kind, subject.id, target.id, source_id,
                record.id if record else None, world.phase_index, summary, [npc.id for npc in witnesses])
    world.facts.append(fact)
    for npc in witnesses:
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
        add_fact(world, "private_motive", npc, world.npc_by_id(goal.target_id), goal.id, None,
                 f"{npc.name}'s motive for {goal.objective}: {goal.motive}", [npc])


def record_event_fact(world, event, record):
    kinds = {"fruit_shared": "apple_given", "food_refused": "apple_refused",
             "promise_fulfilled": "promise_kept", "promise_broken": "promise_broken"}
    kind = kinds.get(event.kind)
    if kind is None:
        return
    first, second = [world.npc_by_id(npc_id) for npc_id in event.participants]
    subject, target = (second, first) if kind == "apple_refused" else (first, second)
    add_fact(world, kind, subject, target, event.id, record, event.text, [subject, target])


def can_deceive(npc):
    return npc.preferences["generosity"] < 0.5 or bool(npc.goal and npc.goal.kind == "sabotage_relationship"
                                                     and npc.goal.status == "active")


def selected_beliefs(npc, partner_id=None):
    world = npc.world
    goal = npc.goal
    def rank(belief):
        claim = claim_by_id(world, belief.claim_id)
        fact = fact_by_id(world, claim.fact_id)
        return (bool(goal and goal.claim_id == claim.id),
                partner_id in (fact.subject_id, fact.object_id, belief.source_id),
                bool(goal and fact.subject_id == goal.third_npc_id),
                fact.kind == "promise_broken" or fact.kind == "private_motive" and fact.subject_id != npc.id,
                fact.kind != "private_motive" or fact.subject_id != npc.id, belief.phase)
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
        selected = selected_beliefs(npc, conversation.other(npc).id)
        if role == "initiator" and conversation.focus_claim_id:
            focus = next(b for b in npc.beliefs if b.claim_id == conversation.focus_claim_id)
            selected = [focus, *[b for b in selected if b is not focus]]
        allowed[role] = [belief.claim_id for belief in selected]
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            fact = fact_by_id(world, claim.fact_id)
            options[claim.id] = ClaimOption(claim.id, fact.id, fact.subject_id, claim.originator_id,
                                           claim.affirmed, claim.parent_claim_id, None, fact.kind,
                                           assertion_text(world, fact, claim.affirmed))
        if can_deceive(npc):
            bases = [claim_by_id(world, belief.claim_id) for belief in selected
                     if belief.stance == "believed" and claim_by_id(world, belief.claim_id).affirmed]
            for index, base in enumerate(bases[:2]):
                fact = fact_by_id(world, base.fact_id)
                option_id = f"{conversation.id}:deny_{role}_{index}"
                options[option_id] = ClaimOption(option_id, fact.id, fact.subject_id, npc.id, False,
                                                base.id, npc.id, fact.kind, assertion_text(world, fact, False))
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
        {role: {other.id: npc.opinions.get(other.name, {}).get("opinion", 0) for other in world.npcs if other is not npc}
         for role, npc in pair.items()}, allowed, conversation.focus_claim_id)


def information_views(context):
    """Only speaker-owned knowledge reaches prompts; engine truth stays out of them."""
    if context is None:
        return {}
    options = {option.id: option for option in context.options}
    result = {}
    for role, npc_id in context.participants.items():
        beliefs = {belief.claim_id: belief for belief in context.beliefs[role]}
        views = []
        for claim_id in context.allowed[role]:
            option = options[claim_id]
            belief = beliefs.get(claim_id)
            view = {"claim_id": claim_id, "subject_id": option.subject_id, "assertion": option.assertion,
                    "original_speaker_id": option.originator_id,
                    "permitted_use": "deliberate denial; private to this speaker" if option.creator_id else "known account"}
            if belief:
                view.update(stance=belief.stance, immediate_source_id=belief.source_id,
                            source_disclosure_id=belief.disclosure_id,
                            basis="firsthand" if belief.direct else "hearsay" if belief.source_id != npc_id else "own assertion",
                            already_assessed=any(b.assessed and options[b.claim_id].fact_id == option.fact_id
                                                 and options[b.claim_id].affirmed == option.affirmed
                                                 for b in context.beliefs[role]),
                            current_reputation_influence=sum(b.opinion_delta for b in context.beliefs[role]
                                                             if options[b.claim_id].fact_id == option.fact_id))
            views.append(view)
        result[role] = views
    return result


def validate_information(response, context):
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
                    or option.kind not in ("private_motive", "apple_given", "apple_refused", "promise_kept", "promise_broken")):
                raise ValueError("Leverage must concern a claim this speaker actually knows.")
            seen_leverage.add(use.claim_id)
        for entry in [entry for entry in response.third_party_opinions if entry.turn == turn]:
            option = options.get(entry.claim_id)
            belief = beliefs[role].get(entry.claim_id)
            if (option is None or belief is None or (role, entry.claim_id) not in heard
                    or entry.subject_id != option.subject_id or entry.subject_id in context.participants.values()
                    or heard[role, entry.claim_id] >= turn):
                raise ValueError("A third-party opinion needs a claim heard before this speaker's reaction.")
            related = [b for b in beliefs[role].values() if options[b.claim_id].fact_id == option.fact_id]
            if (belief.stance != "believed" or entry.delta == 0
                    or any(b.assessed and options[b.claim_id].affirmed == option.affirmed for b in related)
                    or abs(sum(b.opinion_delta for b in related) + entry.delta) > 6
                    or (role, option.fact_id) in seen_opinions):
                raise ValueError("A repeated or disbelieved account cannot cause another reputation penalty.")
            seen_opinions.add((role, option.fact_id))
            belief.assessed, belief.opinion_delta = True, belief.opinion_delta + entry.delta
        for entry in [entry for entry in response.disclosures if entry.turn == turn]:
            option = options.get(entry.claim_id)
            if option is None or (role, entry.claim_id) in seen_disclosures:
                raise ValueError("Unknown or repeated disclosure.")
            seen_disclosures.add((role, entry.claim_id))
            belief = beliefs[role].get(entry.claim_id)
            fabricating = option.creator_id is not None and belief is None
            if fabricating:
                if (option.creator_id != npc_id or entry.intent != "lie" or not context.can_lie[role]
                        or option.parent_claim_id not in beliefs[role]):
                    raise ValueError("This speaker cannot fabricate that assertion.")
                beliefs[role][option.id] = Belief(npc_id, option.id, "disbelieved", npc_id, None, False, 0)
            elif (belief is None or entry.claim_id not in context.allowed[role] and entry.claim_id not in acquired[role]):
                raise ValueError("The speaker has not learned or been permitted to disclose that claim.")
            elif ((entry.intent == "lie" and (not context.can_lie[role] or belief.stance == "believed"))
                  or (entry.intent == "share" and belief.stance == "disbelieved")):
                raise ValueError("A deliberate lie and an honestly repeated belief must remain distinct.")
            prior = beliefs[listener].get(option.id)
            related = [b for b in beliefs[listener].values() if options[b.claim_id].fact_id == option.fact_id]
            if any(b.direct and options[b.claim_id].affirmed != option.affirmed for b in related):
                if entry.listener_stance != "disbelieved":
                    raise ValueError("A listener's firsthand contrary evidence cannot be silently overwritten.")
            elif prior is not None and entry.listener_stance != prior.stance:
                raise ValueError("Repeating an account is not new independent confirmation.")
            elif (any(options[b.claim_id].affirmed == option.affirmed and b.stance != "believed" for b in related)
                  and entry.listener_stance == "believed" and not (belief and belief.direct)):
                raise ValueError("Hearsay repeated through another source is not independent confirmation.")
            if entry.listener_stance == "believed":
                for other in related:
                    if (options[other.claim_id].affirmed != option.affirmed and not other.direct
                            and other.source_id != context.participants[listener]):
                        other.stance = "uncertain"
            if prior is None:
                beliefs[listener][option.id] = Belief(context.participants[listener], option.id,
                                                     entry.listener_stance, npc_id, None, False, 0)
            acquired[listener].add(option.id)
            heard[listener, option.id] = turn
    if context.focus_claim_id:
        focus = options[context.focus_claim_id]
        if not any(response.turns[entry.turn].speaker == "initiator" and options[entry.claim_id].fact_id == focus.fact_id
                   for entry in response.disclosures):
            raise ValueError("A gossip exchange must disclose its chosen account or a permitted denial of it.")


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
            originator = world.npc_by_id(claim.originator_id)
            option = options[claim.id]
            reason = (f"Heard {source.name}'s account (original source: {originator.name}): {option.assertion} "
                      f"Personal assessment: {entry.reason}")[:400]
            event = world.publish("claim_assessed", (source, speaker),
                                  f"{speaker.name} assessed an allegation about {world.npc_by_id(entry.subject_id).name} from {source.name}.",
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
                for belief in listener.beliefs:
                    other = claim_by_id(world, belief.claim_id)
                    if (other.fact_id == option.fact_id and other.affirmed != option.affirmed and not belief.direct
                            and belief.source_id != listener.id):
                        belief.stance = "uncertain"
            if prior is None:
                prior = Belief(listener.id, option.id, entry.listener_stance, speaker.id,
                               disclosure.id, False, world.phase_index)
                listener.beliefs.append(prior)
            elif not prior.direct:
                prior.source_id, prior.disclosure_id = speaker.id, disclosure.id
                prior.stance = entry.listener_stance
            heard[listener.id, option.id] = speaker
            originator = world.npc_by_id(option.originator_id)
            remember(speaker, listener, event, "claim_shared", f"I discussed an account with {listener.name}: {option.assertion}", 0)
            remember(listener, speaker, event, "heard_claim",
                     f"{speaker.name} passed on an account from {originator.name}: {option.assertion} My stance: {entry.listener_stance}.", 0)
            conversation.record.outcomes.append(f"Allegation: {speaker.name} -> {listener.name}; {entry.listener_stance}. {option.assertion}")
    for npc in people.values():
        apply_opinion_effects(npc)


def check_sabotage_goal(world, effect, listener, actual_delta, record):
    owner = world.npc_by_id(effect.actor_id)
    goal = owner.goal if owner else None
    if (goal and goal.status == "active" and goal.kind == "sabotage_relationship"
            and goal.target_id == listener.id and goal.third_npc_id == effect.target_id and actual_delta < 0
            and effect.phase is not None and goal.created_phase <= effect.phase <= world.phase_index <= goal.deadline
            and claim_by_id(world, goal.claim_id).fact_id == claim_by_id(world, effect.claim_id).fact_id
            and listener.opinions.get(world.npc_by_id(effect.target_id).name, {}).get("opinion", 0) <= goal.baseline_opinion - 4):
        resolve_goal(owner, world, "completed", "A committed opinion change from this exchange weakened the relationship.", record)


def trim_information(world):
    """Bound unused accounts while retaining provenance, disputes, and reputation deduplication."""
    claims = {claim.id: claim for claim in world.claims}
    current_goals = {npc.goal.id for npc in world.npcs if npc.goal}
    for npc in world.npcs:
        polarities = {}
        for belief in npc.beliefs:
            claim = claims[belief.claim_id]
            polarities.setdefault(claim.fact_id, set()).add(claim.affirmed)
        keep = {b.claim_id for b in npc.beliefs[-24:]}
        for b in npc.beliefs:
            claim = claims[b.claim_id]
            if b.assessed or len(polarities[claim.fact_id]) > 1 or fact_by_id(world, claim.fact_id).source_id in current_goals:
                keep.add(b.claim_id)
        if npc.goal and npc.goal.claim_id:
            keep.add(npc.goal.claim_id)
        if npc.conversation and npc.conversation.initiator is npc and npc.conversation.focus_claim_id:
            keep.add(npc.conversation.focus_claim_id)
        npc.beliefs[:] = [b for b in npc.beliefs if b.claim_id in keep]
    claim_ids = {b.claim_id for npc in world.npcs for b in npc.beliefs}
    claim_ids.update(effect.claim_id for npc in world.npcs for effect in npc.pending_social_opinions if effect.claim_id)
    disclosure_ids = {b.disclosure_id for npc in world.npcs for b in npc.beliefs if b.disclosure_id}
    disclosure_ids.update(d.id for d in world.disclosures for npc in world.npcs
                          for effect in npc.pending_social_opinions if effect.claim_id == d.claim_id
                          and effect.conversation_id == d.conversation_id and d.listener_id == npc.id)
    disclosure_ids.update(d.id for d in world.disclosures[-100:])
    event_ids = {event.id for event in world.events}
    disclosure_ids.update(d.id for d in world.disclosures if d.event_id in event_ids)
    fact_ids = {fact.id for fact in world.facts[-100:] if fact.source_id not in current_goals}
    fact_ids.update(fact.id for fact in world.facts if fact.source_id in current_goals)
    for npc in world.npcs:
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
        opinion = target.opinions.get(npc.name, {}).get("opinion", 0)
        if npc.inventory_space and (target.inventory.get("fruit", 0)
                                   or target.can_reach(world.tree.interaction_point)):
            choices.append((1 if generous else 3, "obtain_apple", target, opinion))
        if target.inventory_space and (npc.inventory.get("fruit", 0)
                                      or npc.can_reach(world.tree.interaction_point)):
            choices.append((3 if generous else 1, "help_with_apple", target, opinion))
        if opinion < 0:
            choices.append((2 + npc.preferences["sociability"], "repair_relationship", target, opinion))
    choices = [(*choice, None, None) for choice in choices]
    sabotage = []
    if npc.preferences["generosity"] < 0.5:
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            fact = fact_by_id(world, claim.fact_id)
            subject = world.npc_by_id(claim.subject_id)
            if subject is npc or belief.stance == "disbelieved":
                continue
            for target in candidates:
                if target is not subject:
                    baseline = target.opinions.get(subject.name, {}).get("opinion", 0)
                    polarity = fact.kind not in ("apple_given", "promise_kept")
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
                    world.phase_index, world.phase_index + 3, baseline,
                    third_npc_id=third.id if third else None, claim_id=claim.id if claim else None)
    record_intention(npc, world)


def resolve_goal(npc, world, status, outcome, record=None):
    goal = npc.goal
    if goal is None or goal.status != "active":
        return
    target = world.npc_by_id(goal.target_id)
    goal.status, goal.outcome = status, outcome
    goal.resolved_phase = world.phase_index
    event = world.publish("goal_completed" if status == "completed" else "goal_failed",
                          (npc, target), f"{npc.name}: {goal.objective}. {outcome}", record)
    goal.resolution_event_id = event.id
    remember(npc, target, event, "goal_completed" if status == "completed" else "goal_failed",
             f"My objective: {goal.objective}. {outcome}", 2 if status == "completed" else -2)


def check_relationship_goals(world, opinion_owner, record=None):
    for npc in world.npcs:
        goal = npc.goal
        if (goal and goal.status == "active" and goal.kind == "repair_relationship"
                and goal.target_id == opinion_owner.id and world.phase_index <= goal.deadline
                and opinion_owner.opinions.get(npc.name, {}).get("opinion", 0)
                >= goal.baseline_opinion + GOAL_IMPROVEMENT):
            resolve_goal(npc, world, "completed", "Their opinion improved through a real interaction.", record)


def permitted_proposals(conversation):
    if conversation.purpose == "food_request":
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


def validate_agreements(agreements, proposals, turns, initiator_id, recipient_id):
    """Bound offers and acceptances to ordered speech by the correct participants."""
    permitted = {proposal["id"]: proposal for proposal in proposals}
    seen = set()
    for agreement in agreements:
        proposal = permitted.get(agreement.proposal_id)
        if proposal is None or agreement.proposal_id in seen:
            raise ValueError("The agreement refers to an unavailable or repeated proposal.")
        seen.add(agreement.proposal_id)
        if not 0 <= agreement.offer_turn < agreement.accept_turn < len(turns):
            raise ValueError("A promise needs an offer followed by acceptance in this exchange.")
        speakers = {"initiator": initiator_id, "recipient": recipient_id}
        if (speakers[turns[agreement.offer_turn].speaker] != proposal["promisor_id"]
                or speakers[turns[agreement.accept_turn].speaker] != proposal["recipient_id"]):
            raise ValueError("Only the apple's giver can offer and its recipient can accept.")


def validate_live_agreements(conversation, response):
    world = conversation.initiator.world
    pair = (conversation.initiator, conversation.recipient)
    if any(npc.conversation is not conversation or not any(person is npc for person in world.npcs) for npc in pair):
        raise ValueError("The conversation participants are no longer available.")
    if conversation.context_references != conversation_references(conversation):
        raise ValueError("An objective or promise changed while the reply was being generated. Please try again.")
    current_information = prepare_information(conversation)
    if ((asdict(current_information) if current_information else None)
            != (asdict(conversation.information_context) if conversation.information_context else None)):
        raise ValueError("A speaker's knowledge or opinion changed during generation. Please try again.")
    validate_information(response, conversation.information_context)
    validate_agreements(response.accepted_proposals, [asdict(p) for p in conversation.proposals],
                        response.turns, pair[0].id, pair[1].id)
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
    references = {npc.goal.id: npc.goal.status for npc in pair if npc.goal}
    references.update({p.id: p.status for p in pair[0].world.promises
                       if ids.intersection((p.promisor_id, p.recipient_id))})
    return references


def commit_agreements(conversation):
    world = conversation.initiator.world
    for agreement in conversation.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        promise = Promise(proposal.id, proposal.promisor_id, proposal.recipient_id,
                          conversation.record.id, world.phase_index, proposal.deadline)
        world.promises.append(promise)
        giver, recipient = world.npc_by_id(promise.promisor_id), world.npc_by_id(promise.recipient_id)
        text = f"{giver.name} promised {recipient.name} one apple by {phase_label(promise.deadline)}."
        event = world.publish("promise_created", (giver, recipient), text, conversation.record)
        conversation.record.outcomes.append(text)
        for npc, partner in ((giver, recipient), (recipient, giver)):
            remember(npc, partner, event, "promise_created", text, 1)


def apply_opinion_effects(npc):
    while npc.pending_social_opinions:
        effect = npc.pending_social_opinions[0]
        target = npc.world.npc_by_id(effect.target_id)
        previous = npc.opinions.get(target.name, {"opinion": 0, "reason": ""})
        # Keep the specific unresolved outcome when later impressions are added.
        reason = effect.reason + (f" Earlier: {previous['reason']}" if previous["reason"] else "")
        try:
            npc.remember_opinions({target.name: {
                "opinion": max(-100, min(100, previous["opinion"] + effect.delta)), "reason": reason[:400]}})
        except (OSError, ValueError):
            return False
        npc.pending_social_opinions.pop(0)
        if effect.claim_id:
            record = next(entry for entry in npc.world.history if entry.id == effect.conversation_id)
            actual_delta = npc.opinions[target.name]["opinion"] - previous["opinion"]
            text = f"{npc.name}'s opinion of {target.name} changed {actual_delta:+d} after assessing an allegation. {effect.reason}"
            event = npc.world.publish("reputation_changed", (npc, target), text, record)
            record.outcomes.append(text)
            remember(npc, target, event, "claim_opinion", effect.reason, max(-4, min(4, actual_delta)))
            check_sabotage_goal(npc.world, effect, npc, actual_delta, record)
    return True


def resolve_promise(world, promise, status, transfer=None):
    if promise.status != "pending":
        return
    giver, recipient = world.npc_by_id(promise.promisor_id), world.npc_by_id(promise.recipient_id)
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
        remember(npc, partner, event, "promise_fulfilled" if status == "fulfilled" else "promise_broken",
                 text, 4 if status == "fulfilled" else -5)
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
            fact = fact_by_id(world, claim.fact_id)
            known.append({"claim_id": claim.id, "subject_id": claim.subject_id,
                          "assertion": assertion_text(world, fact, claim.affirmed), "stance": belief.stance,
                          "original_speaker_id": claim.originator_id, "immediate_source_id": belief.source_id,
                          "source_disclosure_id": belief.disclosure_id, "basis": "firsthand" if belief.direct else "hearsay",
                          "already_assessed": belief.assessed})
    goal_view = asdict(npc.goal) if npc.goal else None
    if goal_view:
        goal_view.pop("baseline_opinion")
    return {"owner_id": npc.id, "visibility": "Only this speaker knows their private motive and memories.",
            "goal": goal_view,
            "known_accounts": known,
            "promises": [dict(asdict(p), deadline_label=phase_label(p.deadline)) for p in promises[:8]]}


def pair_objective(world, npc, other):
    pair = {npc.id, other.id}
    promises = [p for p in world.promises if {p.promisor_id, p.recipient_id} == pair]
    pending = [p for p in promises if p.status == "pending"]
    if pending:
        promise = min(pending, key=lambda p: p.deadline)
        giver, recipient = world.npc_by_id(promise.promisor_id), world.npc_by_id(promise.recipient_id)
        return 3, f"Arrange {giver.name}'s promised apple for {recipient.name}, due {phase_label(promise.deadline)}"
    if promises and promises[-1].status == "broken":
        return 2, f"Discuss the missed apple promise: {promises[-1].outcome} Seek an explanation or practical repair"
    for belief in selected_beliefs(npc, other.id):
        claim = claim_by_id(world, belief.claim_id)
        contrary = [b for b in npc.beliefs if claim_by_id(world, b.claim_id).fact_id == claim.fact_id
                    and claim_by_id(world, b.claim_id).affirmed != claim.affirmed]
        if contrary and other.id in (belief.source_id, claim.originator_id):
            return 2, f"Ask {other.name} to explain conflicting accounts about {world.npc_by_id(claim.subject_id).name}"
    goal = npc.goal
    if goal and goal.status == "active" and goal.target_id == other.id:
        return 1, f"{goal.objective}, by {phase_label(goal.deadline)}"
    other_known = {belief.claim_id for belief in other.beliefs}
    news = [belief for belief in selected_beliefs(npc, other.id) if belief.claim_id not in other_known
            and claim_by_id(world, belief.claim_id).subject_id not in (npc.id, other.id)]
    if news:
        subject = world.npc_by_id(claim_by_id(world, news[0].claim_id).subject_id)
        return 0.5, f"Discuss an account about {subject.name}, distinguishing hearsay from firsthand experience"
    return 0, ""


def goal_attempt_cooling_down(npc, other, world):
    goal = npc.goal
    return bool(goal and goal.status == "active" and goal.target_id == other.id
                and goal.last_attempt_phase is not None and world.phase_index < goal.last_attempt_phase + 2)


def gossip_claim(npc, partner):
    if npc.goal and npc.goal.status == "active" and npc.goal.kind == "sabotage_relationship" and npc.goal.target_id == partner.id:
        return npc.goal.claim_id
    known = {b.claim_id for b in partner.beliefs}
    return next((b.claim_id for b in selected_beliefs(npc, partner.id) if b.claim_id not in known
                 and claim_by_id(npc.world, b.claim_id).subject_id not in (npc.id, partner.id)), None)
