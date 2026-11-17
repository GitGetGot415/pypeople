from dataclasses import asdict
from simulation.social.records import claim_by_id
from simulation.social.information import account_view, selected_beliefs
from simulation.social.concerns import concern_topic, danger_topic, undisclosed_danger
from simulation.social.promises import pending_promises
from simulation.social.relationships import opinion_causes, regards_as_dangerous


def private_context(npc, partner_id=None):
    world = npc.world
    promises = [p for p in world.promises if npc.id in (p.promisor_id, p.recipient_id)] if world else []
    promises.sort(key=lambda p: (p.status == "pending", partner_id in (p.promisor_id, p.recipient_id), p.created_day), reverse=True)
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
            "listener_grievances": opinion_causes(npc, partner) if partner else [],
            "owner_id": npc.id, "visibility": "Only this speaker knows their private motive and memories.",
            "concern": ({"kind": npc.current_concern.kind, "claim_id": npc.current_concern.claim_id,
                         "guidance": concern_topic(npc, npc.current_concern)} if npc.current_concern else None),
            "goal": goal_view,
            "known_accounts": known,
            "promises": [dict(asdict(p), deadline_label=str(p.deadline)) for p in promises[:8]]}


def pair_objective(world, npc, other):
    pair = {npc.id, other.id}
    def discussed(topic):
        return any(r.turns and set(r.participants) == {npc.name, other.name} and r.topic == topic for r in world.history)
    promises = [p for p in world.promises if {p.promisor_id, p.recipient_id} == pair]
    pending = [p for p in promises if p.status == "pending"]
    if pending:
        promise = min(pending, key=lambda p: p.deadline)
        giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
        topic = f"Arrange {giver.name}'s promised apple for {recipient.name}, due {promise.deadline}"
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
    # Seeking safety from the killer themself confuses who did it; their opinion of the killer may even be warm.
    if danger and claim_by_id(world, danger).subject_id != other.id:
        return 2, danger_topic(npc, danger)
    goal = npc.goal
    if goal and goal.status == "active" and goal.target_id == other.id:
        if goal.kind == "help_with_apple" and not npc.inventory.get("fruit", 0):
            # Nothing to hand over yet, so the doable step is a promise kept from a later harvest.
            if len(pending_promises(world, npc.id)) >= 2:
                return 0, ""  # No further promise is permitted until one is kept.
            return 1, f"Promise {other.name} an apple by {world.day + 1}; I carry none right now"
        return 1, f"{goal.objective}, by {goal.deadline}"
    return 0, ""
