"""Small, local activity choices. Autonomous activities never call a model."""

import math

from npc import Behavior
from conversation_history import memory_bias
from simulation import APPLE_SATISFACTION, DECISION_SECONDS, SOCIAL_START


def company_point(npc, partner, spacing=0):
    left, right, top, bottom = npc.movement_bounds
    gap = ((npc.sprite_size or (36, 18))[0] + (partner.sprite_size or (36, 18))[0]) / 2 + 12
    gap = max(gap, spacing)
    points = [(partner.x - gap, partner.y), (partner.x + gap, partner.y)]
    points = [point for point in points if left <= point[0] <= right and top <= point[1] <= bottom]
    return min(points, key=lambda p: math.hypot(p[0] - npc.x, p[1] - npc.y)) if points else None


def target_valid(npc, world):
    if npc.behavior == Behavior.EAT:
        return npc.inventory.get("fruit", 0) > 0
    if npc.behavior == Behavior.HARVEST:
        return (npc.target_id == world.tree.id and world.tree.reserved_by is npc
                and world.tree.fruit > 0 and npc.inventory_space > 0
                and npc.can_reach(world.tree.interaction_point))
    if npc.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
        return npc.target_id == world.tree.id and npc.can_reach(world.tree.interaction_point)
    if npc.behavior == Behavior.COMPANY:
        partner = world.npc_by_id(npc.target_id)
        return (partner is not None and partner is not npc and world.company_available(partner)
                and company_point(npc, partner) is not None)
    return True


def reconsider(npc, world):
    if npc.conversation is not None or npc.gift is not None or npc.resume_pending:
        return
    if not target_valid(npc, world):
        npc.finish_activity(world, failed=True)
    if npc.owner != "autonomous" or npc.selected or npc.request_pending:
        return
    if npc.decision_remaining > 0:
        return
    npc.decision_remaining = DECISION_SECONDS
    if npc.behavior in (Behavior.HARVEST, Behavior.COMPANY, Behavior.EAT):
        return

    def distance(point):
        return math.hypot(npc.x - point[0], npc.y - point[1]) / 90

    candidates = [(1 + npc.rng.random(), Behavior.WANDER, None, "Exploring the world")]
    if npc.inventory.get("fruit", 0) > 0 and npc.social <= 100 - APPLE_SATISFACTION:
        candidates.append((130 - npc.social, Behavior.EAT, None, "Eating an apple to improve satisfaction"))
    tree = world.tree
    if (tree.fruit > 0 and tree.reserved_by in (None, npc) and npc.inventory_space > 0
            and tree.id not in npc.cooldowns and npc.can_reach(tree.interaction_point)):
        score = 24 - sum(npc.inventory.values()) * 4 - distance(tree.interaction_point)
        candidates.append((score, Behavior.HARVEST, tree.id, "Collecting fruit for my inventory"))
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
                and company_point(npc, other) is not None]

    def rank(npc, other):
        return (npc.opinions.get(other.name, {}).get("opinion", 0) * 0.08
                + memory_bias(npc, other.id) - math.hypot(npc.x - other.x, npc.y - other.y) / 90)

    food = []
    for npc in available:
        if npc.social > SOCIAL_START or npc.inventory.get("fruit", 0) or npc.inventory_space < 1:
            continue
        donors = [other for other in partners(npc) if other.inventory.get("fruit", 0) > 0]
        if not donors:
            continue
        donor = max(donors, key=lambda other: rank(npc, other))
        tree = world.tree
        if (tree.fruit > 0 and tree.reserved_by is None and npc.can_reach(tree.interaction_point)
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
    topic = f"Remembering: {recollection.summary}" if recollection else f"The {world.phase} and how things are going"
    return npc, partner, "chat", topic, f"Seeking a chat with {partner.name}{memory_reason(npc, partner)}"
