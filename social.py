"""Private objectives and apple commitments, resolved only by committed world changes."""

from dataclasses import asdict, dataclass
from typing import Literal
from uuid import uuid4

from conversation_history import remember

GoalKind = Literal["obtain_apple", "help_with_apple", "repair_relationship"]
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
    if not choices:
        npc.goal = None
        return
    _, kind, target, baseline = npc.rng.choices(choices, weights=[choice[0] for choice in choices])[0]
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
    objective, motive, approach = descriptions[kind]
    npc.goal = Goal(uuid4().hex, kind, target.id, objective, motive, approach,
                    "Respect a refusal; do not claim a gift or invent shared history.",
                    world.phase_index, world.phase_index + 3, baseline)


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
    validate_agreements(response.accepted_proposals, [asdict(p) for p in conversation.proposals],
                        response.turns, pair[0].id, pair[1].id)
    pending = [(p.promisor_id, p.recipient_id) for p in pending_promises(world)]
    for agreement in response.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        if (conversation.purpose != "chat" or world.phase_index > proposal.deadline
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


def private_context(npc, partner_id=None):
    world = npc.world
    promises = [p for p in world.promises if npc.id in (p.promisor_id, p.recipient_id)] if world else []
    promises.sort(key=lambda p: (p.status == "pending", partner_id in (p.promisor_id, p.recipient_id), p.created_phase), reverse=True)
    return {"owner_id": npc.id, "visibility": "Only this speaker knows their private motive and memories.",
            "goal": asdict(npc.goal) if npc.goal else None,
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
    goal = npc.goal
    if goal and goal.status == "active" and goal.target_id == other.id:
        return 1, f"{goal.objective}, by {phase_label(goal.deadline)}"
    return 0, ""


def goal_attempt_cooling_down(npc, other, world):
    goal = npc.goal
    return bool(goal and goal.status == "active" and goal.target_id == other.id
                and goal.last_attempt_phase is not None and world.phase_index < goal.last_attempt_phase + 2)
