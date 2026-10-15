"""Small, local activity choices. Autonomous activities never call a model."""

import math

from npc import Behavior
from simulation import DECISION_SECONDS, SOCIAL_START


def company_point(npc, partner):
    left, right, top, bottom = npc.movement_bounds
    gap = ((npc.sprite_size or (36, 18))[0] + (partner.sprite_size or (36, 18))[0]) / 2 + 12
    points = [(partner.x - gap, partner.y), (partner.x + gap, partner.y)]
    points = [point for point in points if left <= point[0] <= right and top <= point[1] <= bottom]
    return min(points, key=lambda p: math.hypot(p[0] - npc.x, p[1] - npc.y)) if points else None


def target_valid(npc, world):
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
    if npc.conversation is not None or npc.resume_pending:
        return
    if not target_valid(npc, world):
        npc.finish_activity(world, failed=True)
    if npc.owner != "autonomous" or npc.selected or npc.request_pending:
        return
    if npc.decision_remaining > 0:
        return
    npc.decision_remaining = DECISION_SECONDS
    if npc.behavior in (Behavior.HARVEST, Behavior.COMPANY):
        return

    def distance(point):
        return math.hypot(npc.x - point[0], npc.y - point[1]) / 90

    candidates = [(1 + npc.rng.random(), Behavior.WANDER, None, "Exploring the world")]
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
                return opinion * 0.08 - distance((other.x, other.y))
            partner = max(partners, key=company_score)
            score = (100 - npc.social) * (0.8 + npc.preferences["sociability"] * 0.3) + company_score(partner)
            candidates.append((score, Behavior.COMPANY, partner.id, f"Wanting company; spending time with {partner.name}"))
    _, behavior, target, reason = max(candidates, key=lambda c: c[0] + npc.rng.uniform(0, 3))
    npc.begin_activity(behavior, world, target, reason)
