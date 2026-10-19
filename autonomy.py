"""Small, local activity choices. Autonomous activities never call a model."""

import math
from constants import ORIENTATION_PRIORITY
from name_knowledge import orientation_context

from npc import Behavior
from conversation_history import memory_bias
from constants import APPLE_SATISFACTION, DECISION_SECONDS, MOVEMENT_SPEED, SOCIAL_START, SNAKE_NAME
from interactions import company_point
from social import (claim_by_id, fact_by_id, goal_attempt_cooling_down, pair_objective,
                    pending_promises, selected_beliefs)


ORIENTATION_TOPIC = (
    "Find my bearings: understand these surroundings, the missing transition, or where to find help and safety"
)


def target_valid(npc, world):
    if npc.behavior == Behavior.EAT:
        return npc.inventory.get("fruit", 0) > 0
    if npc.behavior == Behavior.HARVEST:
        return (npc.target_id == world.tree.id and world.tree.reserved_by is npc
                and world.available_fruit > 0 and npc.inventory_space > 0
                and npc.can_reach(world.tree.interaction_point))
    if npc.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
        return npc.target_id == world.tree.id and npc.can_reach(world.tree.interaction_point)
    if npc.behavior == Behavior.COMPANY:
        partner = world.npc_by_id(npc.target_id)
        return (partner is not None and partner is not npc and world.company_available(partner)
                and company_point(npc, partner) is not None)
    return True


def seek_orientation(npc, world):
    """Find a first conversation without spending a world-event opportunity."""
    if (not orientation_context(npc)["needs_orientation"] or npc.departure is not None or npc.interaction is not None
            or npc.resume_pending or npc.owner != "autonomous" or npc.selected or npc.request_pending
            or any(npc.id in (action.npc_id, action.target_id) for action in world.pending_actions)):
        return
    partners = [other for other in world.npcs if other is not npc and world.company_available(other)
                and other.id not in npc.cooldowns and npc.id not in other.cooldowns
                and company_point(npc, other, allow_vertical=True) is not None]
    # Keep wandering while waiting, without resetting the route each tick.
    reason = (f"Waiting to find my bearings with {SNAKE_NAME}" if len(world.npcs) == 1 else
              "Waiting to find my bearings with another resident")
    if npc.behavior != Behavior.WANDER:
        npc.begin_activity(Behavior.WANDER, world, reason=reason)
    else:
        npc.reason = reason
    if partners:
        partner = min(partners, key=lambda other: math.dist((npc.x, npc.y), (other.x, other.y)))
        world.start_conversation(npc, partner, ORIENTATION_TOPIC, origin="autonomous", purpose="orientation")
        npc.reason = f"Finding my bearings with {partner.name}"
        partner.reason = f"Helping {npc.name} find their bearings"
    elif len(world.npcs) == 1:
        snake = world.snake
        if (snake.encounter or snake.selected or snake.request_pending or snake.recovery_remaining
                or snake.id in npc.cooldowns or not npc.can_reach(world.tree.interaction_point)):
            return
        from snake import SnakeEncounter
        SnakeEncounter(world, npc, "introduction", ORIENTATION_TOPIC)


def reconsider(npc, world):
    if npc.interaction is not None or npc.resume_pending:
        return
    if not target_valid(npc, world):
        npc.finish_activity(world, failed=True)
    if npc.owner != "autonomous" or npc.selected or npc.request_pending:
        return
    if any(npc.id in (action.npc_id, action.target_id) for action in world.pending_actions):
        return
    if orientation_context(npc)["needs_orientation"]:
        return
    if npc.decision_remaining > 0:
        return
    npc.decision_remaining = DECISION_SECONDS
    if npc.behavior in (Behavior.HARVEST, Behavior.EAT):
        return

    obligations = sorted(pending_promises(world, npc.id), key=lambda promise: (promise.deadline, promise.created_phase))
    goal = npc.goal if npc.goal and npc.goal.status == "active" else None
    delivery_targets = [(promise.recipient_id, "Delivering a promised apple") for promise in obligations]
    if goal and goal.kind == "help_with_apple":
        delivery_targets.append((goal.target_id, "Giving an apple to complete my objective"))
    if npc.inventory.get("fruit", 0) > 0 and world.company_available(npc):
        for target_id, reason in delivery_targets:
            target = world.npc_by_id(target_id)
            if (target is not None and world.company_available(target) and target.inventory_space > 0
                    and target.id not in npc.cooldowns and npc.id not in target.cooldowns
                    and company_point(npc, target) is not None):
                gift = npc.give_items(target, "fruit", 1)
                gift.record.origin = "autonomous"
                world.history.append(gift.record)
                npc.reason = f"{reason} to {target.name}"
                return
    apples_needed = len({target_id for target_id, _ in delivery_targets})
    needs_apple = npc.inventory.get("fruit", 0) < min(apples_needed, npc.inventory_capacity)
    if npc.behavior == Behavior.COMPANY and not needs_apple:
        return

    def distance(point):
        return math.hypot(npc.x - point[0], npc.y - point[1]) / MOVEMENT_SPEED

    candidates = [(1 + npc.rng.random(), Behavior.WANDER, None, "Exploring the world")]
    if npc.inventory.get("fruit", 0) > apples_needed and npc.social <= 100 - APPLE_SATISFACTION:
        candidates.append((130 - npc.social, Behavior.EAT, None, "Eating an apple to improve satisfaction"))
    tree = world.tree
    if (world.available_fruit > 0 and tree.reserved_by in (None, npc) and npc.inventory_space > 0
            and tree.id not in npc.cooldowns and npc.can_reach(tree.interaction_point)):
        score = 24 - sum(npc.inventory.values()) * 4 - distance(tree.interaction_point) + (110 if needs_apple else 0)
        candidates.append((score, Behavior.HARVEST, tree.id,
                           "Seeking an apple bargain to help someone" if needs_apple else f"Seeking an apple bargain with {SNAKE_NAME}"))
    if npc.social <= SOCIAL_START:
        partners = [other for other in world.npcs if other is not npc and world.company_available(other)
                    and other.id not in npc.cooldowns and company_point(npc, other) is not None]
        if partners:
            def company_score(other):
                opinion = npc.opinions.get(other.name, {}).get("opinion", 0)
                return opinion * 0.08 + memory_bias(npc, other.id) - distance((other.x, other.y))
            partner = max(partners, key=company_score)
            score = (100 - npc.social) * (0.8 + npc.preferences["sociability"] * 0.3) + company_score(partner)
            candidates.append((score, Behavior.COMPANY, partner.id,
                               f"Wanting company; spending time with {partner.name}{memory_reason(npc, partner)}"))
    _, behavior, target, reason = max(candidates, key=lambda c: c[0] + npc.rng.uniform(0, 3))
    npc.begin_activity(behavior, world, target, reason)


def memory_reason(npc, partner):
    bias = memory_bias(npc, partner.id)
    if bias >= 1:
        if any(memory.partner_id == partner.id and memory.kind == "received_fruit" for memory in npc.memories):
            return "; remembering that they shared an apple with me"
        return "; remembering a positive experience with them"
    if bias <= -1:
        return "; cautious after an unpleasant experience with them"
    return ""


def donor_choice(donor, requester):
    fruit = donor.inventory.get("fruit", 0)
    if fruit < 1 or requester.inventory_space < 1:
        return "unavailable", "Sharing is no longer possible because the inventories changed."
    opinion = donor.opinions.get(requester.name, {}).get("opinion", 0)
    score = ((donor.social - 50) * 0.12 + (fruit - 1) * 6
             + (donor.preferences["generosity"] - 0.5) * 10
             + opinion * 0.12 + memory_bias(donor, requester.id))
    if fruit == 1 and donor.social <= SOCIAL_START:
        score -= 8
    if score >= 0:
        return "accepted", "The donor is willing to share one apple."
    if fruit == 1 and donor.social <= SOCIAL_START:
        return "need", "The donor wants to eat their last apple themselves."
    return "refused", "The donor prefers to keep their apples this time."


def social_pair(world):
    """Spend one phase opportunity; rotating ties prevents a fixed first speaker."""
    ordered = world.npcs[world.social_rotation:] + world.npcs[:world.social_rotation]
    available = [npc for npc in ordered if world.company_available(npc)]

    def partners(npc):
        return [other for other in available if other is not npc
                and other.id not in npc.cooldowns and npc.id not in other.cooldowns
                and (pair_objective(world, npc, other)[0] >= 2
                     or not (goal_attempt_cooling_down(npc, other, world)
                             or goal_attempt_cooling_down(other, npc, world)))
                and company_point(npc, other) is not None]

    def rank(npc, other):
        return (npc.opinions.get(other.name, {}).get("opinion", 0) * 0.08
                + memory_bias(npc, other.id) - math.hypot(npc.x - other.x, npc.y - other.y) / 90)

    purposeful = []
    for npc in available:
        for other in partners(npc):
            priority, objective = pair_objective(world, npc, other)
            if orientation_context(npc)["needs_orientation"] and priority < ORIENTATION_PRIORITY:
                priority = ORIENTATION_PRIORITY
                objective = ORIENTATION_TOPIC
            # A known immediate danger keeps its place above routine orientation.
            if priority < 2 and any(b.stance != "disbelieved" and
                    fact_by_id(world, claim_by_id(world, b.claim_id).fact_id).kind in ("eating", "judgment", "defiance")
                    for b in selected_beliefs(npc, other.id)):
                priority, objective = 2, "Seek safety or help after a known alarming event, using its actual source"
            if priority:
                deadline = min((p.deadline for p in pending_promises(world)
                                if {p.promisor_id, p.recipient_id} == {npc.id, other.id}), default=world.phase_index)
                purposeful.append((priority, -deadline, rank(npc, other), npc, other, objective))
    if purposeful:
        priority, _, _, npc, partner, objective = max(purposeful, key=lambda choice: choice[:3])
        purpose = ("orientation" if priority == ORIENTATION_PRIORITY else
                   "gossip" if priority == 0.5 or (priority == 1 and npc.goal and npc.goal.kind == "sabotage_relationship") else "chat")
        return npc, partner, purpose, objective, f"Meeting {partner.name}: {objective}"

    food = []
    for npc in available:
        if npc.social > SOCIAL_START or npc.inventory.get("fruit", 0) or npc.inventory_space < 1:
            continue
        donors = [other for other in partners(npc) if other.inventory.get("fruit", 0) > 0]
        if not donors:
            continue
        donor = max(donors, key=lambda other: rank(npc, other))
        tree = world.tree
        if (world.available_fruit > 0 and tree.reserved_by is None and npc.can_reach(tree.interaction_point)
                and math.dist((npc.x, npc.y), tree.interaction_point)
                < math.dist((npc.x, npc.y), (donor.x, donor.y))):
            continue
        food.append((npc, donor))
    if food:
        npc, donor = min(food, key=lambda pair: pair[0].social)
        return npc, donor, "food_request", "An apple", f"Asking {donor.name} for an apple{memory_reason(npc, donor)}"
    choices = []
    for npc in available:
        others = partners(npc)
        if others:
            partner = max(others, key=lambda other: rank(npc, other))
            score = (npc.preferences["sociability"] * 10 + (100 - npc.social) * 0.1 + rank(npc, partner))
            choices.append((score, npc, partner))
    if not choices:
        return None
    _, npc, partner = max(choices, key=lambda choice: choice[0])
    recollection = next((memory for memory in reversed(npc.memories) if memory.partner_id == partner.id), None)
    danger = next((belief for belief in selected_beliefs(npc, partner.id)
                   if belief.stance != "disbelieved" and fact_by_id(world, claim_by_id(world, belief.claim_id).fact_id).kind
                   in ("eating", "judgment", "defiance")), None)
    if danger:
        topic = f"Safety around {SNAKE_NAME} after a known alarming event or sourced warning"
    elif recollection:
        topic = f"Remembering: {recollection.summary}"
    else:
        topic = "Life here and whatever still feels unresolved, building on what we have already discussed"
    return npc, partner, "chat", topic, f"Seeking a chat with {partner.name}{memory_reason(npc, partner)}"
