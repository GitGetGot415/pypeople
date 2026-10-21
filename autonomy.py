"""Small, local activity choices. Autonomous activities never call a model."""

import math
from name_knowledge import orientation_context

from npc import Behavior
from conversation_history import memory_bias
from constants import APPLE_SATISFACTION, DECISION_SECONDS, MOVEMENT_SPEED, SOCIAL_PARTNER_RANDOMNESS, SOCIAL_START, SNAKE_NAME
from interactions import company_point
from social import (concern_topic, goal_attempt_cooling_down, last_conversation_partner, pair_objective,
                    pending_promises, repeat_partner_paused)


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


def lone_introduction(npc, world):
    if (world.phase_pending or len(world.npcs) != 1 or not world.company_available(npc) or npc.conversation_pause_remaining
            or orientation_context(npc)["orientation_discussions"]
            or any(r.purpose == "introduction" and r.turns and npc.name in r.participants for r in world.history)):
        return
    snake = world.snake
    if (snake.encounter or snake.selected or snake.request_pending or snake.recovery_remaining
            or snake.id in npc.cooldowns or not npc.can_reach(world.tree.interaction_point)):
        return
    from snake import SnakeEncounter
    SnakeEncounter(world, npc, "introduction", ORIENTATION_TOPIC)


def reconsider(npc, world, *, pressing_only=False):
    from apartments import HOME_BEHAVIORS
    if (world.phase_pending or npc.interaction is not None or npc.resume_pending or npc.behavior in HOME_BEHAVIORS
            or npc.home_reason == "night"):
        return
    if not target_valid(npc, world):
        npc.finish_activity(world, failed=True)
    if npc.owner != "autonomous" or npc.selected or npc.request_pending:
        return
    if any(npc.id in (action.npc_id, action.target_id) for action in world.pending_actions):
        return
    if npc.decision_remaining > 0:
        return
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
    tree = world.tree
    tree_available = (world.available_fruit > 0 and tree.reserved_by in (None, npc) and npc.inventory_space > 0
                      and tree.id not in npc.cooldowns and npc.can_reach(tree.interaction_point))
    if pressing_only:
        if obligations and needs_apple and tree_available:
            npc.begin_activity(Behavior.HARVEST, world, tree.id, "Seeking an apple bargain to fulfill a commitment")
        return
    npc.decision_remaining = DECISION_SECONDS
    if actionable_obligation(npc, world):
        return
    if npc.behavior == Behavior.COMPANY and not needs_apple:
        return

    def distance(point):
        return math.hypot(npc.x - point[0], npc.y - point[1]) / MOVEMENT_SPEED

    candidates = [(1 + npc.rng.random(), Behavior.WANDER, None, "Exploring the world")]
    if npc.inventory.get("fruit", 0) > apples_needed and npc.social <= 100 - APPLE_SATISFACTION:
        candidates.append((130 - npc.social, Behavior.EAT, None, "Eating an apple to improve satisfaction"))
    if tree_available:
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
    if behavior == npc.behavior == Behavior.WANDER:
        return  # Waiting for company must not reset an existing route.
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


def actionable_obligation(npc, world):
    from apartments import available_outdoors
    if not available_outdoors(npc):
        return False
    # Preserve a free receiver for a delivery that is already physically possible,
    # even when the giver's ordinary decision timer has not elapsed yet.
    if npc.inventory_space > 0:
        for promise in pending_promises(world):
            giver = world.npc_by_id(promise.promisor_id)
            if (promise.recipient_id == npc.id and giver is not None and world.company_available(giver)
                    and giver.inventory.get("fruit", 0) and giver.id not in npc.cooldowns and npc.id not in giver.cooldowns
                    and company_point(giver, npc) is not None):
                return True
    obligations = pending_promises(world, npc.id)
    if not obligations:
        return False
    if npc.inventory.get("fruit", 0):
        return any((target := world.npc_by_id(p.recipient_id)) is not None and world.company_available(target)
                   and target.inventory_space > 0 and target.id not in npc.cooldowns and npc.id not in target.cooldowns
                   and company_point(npc, target) is not None for p in obligations)
    tree = world.tree
    return (world.available_fruit > 0 and tree.reserved_by in (None, npc) and npc.inventory_space > 0
            and tree.id not in npc.cooldowns and npc.can_reach(tree.interaction_point))


def social_pair(world, *, due=None):
    """Reserve through the caller immediately; rotate initiators and vary partners."""
    if world.phase_pending:
        return None
    ordered = world.npcs[world.social_rotation:] + world.npcs[:world.social_rotation]
    available = [npc for npc in ordered if world.company_available(npc)
                 and not npc.conversation_pause_remaining and not actionable_obligation(npc, world)]

    def partners(npc):
        return [other for other in available if other is not npc
                and other.id not in npc.cooldowns and npc.id not in other.cooldowns
                and company_point(npc, other, allow_vertical=True) is not None]

    def rank(npc, other):
        return (npc.opinions.get(other.name, {}).get("opinion", 0) * 0.08
                + memory_bias(npc, other.id) - math.dist((npc.x, npc.y), (other.x, other.y)) / MOVEMENT_SPEED
                + npc.rng.uniform(0, SOCIAL_PARTNER_RANDOMNESS))

    purposeful = []
    for npc in available:
        for other in partners(npc):
            if due is not None and npc.id not in due and other.id not in due:
                continue
            priority, objective = pair_objective(world, npc, other)
            if priority == 1 and (goal_attempt_cooling_down(npc, other, world)
                                  or npc.current_concern.kind == "orientation" or repeat_partner_paused(npc, other)):
                continue
            if priority:
                purposeful.append((priority, rank(npc, other), npc, other, objective))
    if purposeful:
        priority, _, npc, partner, objective = max(purposeful, key=lambda choice: choice[:2])
        goal = npc.goal if priority == 1 else None
        focus = goal.claim_id if goal and goal.kind == "sabotage_relationship" else None
        return npc, partner, "chat", objective, f"Meeting {partner.name}: {objective}", "news" if priority == 2 else None, focus, goal.id if goal else None

    for npc in available:
        if due is not None and npc.id not in due:
            continue
        others = [other for other in partners(npc) if not repeat_partner_paused(npc, other)]
        if not others:
            continue
        # Keep the existing fixed food-request eligibility and donor decision.
        if (npc.current_concern.kind != "orientation" and npc.social <= SOCIAL_START
                and not npc.inventory.get("fruit", 0) and npc.inventory_space):
            donors = [other for other in others if other.inventory.get("fruit", 0) > 0]
            if donors:
                donor = max(donors, key=lambda other: rank(npc, other))
                tree = world.tree
                if not (world.available_fruit > 0 and tree.reserved_by is None and npc.can_reach(tree.interaction_point)
                        and math.dist((npc.x, npc.y), tree.interaction_point)
                        < math.dist((npc.x, npc.y), (donor.x, donor.y))):
                    return npc, donor, "food_request", "An apple", f"Asking {donor.name} for an apple", None, None, None
        partner = max(others, key=lambda other: (
            last_conversation_partner(npc) != other.name and last_conversation_partner(other) != npc.name,
            rank(npc, other)))
        concern = npc.current_concern
        topic = concern_topic(npc, concern)
        return npc, partner, "chat", topic, f"Seeking a chat with {partner.name}: {topic}", concern.kind, None, None
    return None
