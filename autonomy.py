"""Small, local activity choices. Autonomous activities never call a model."""

import math
from activity import Behavior
from conversation_history import memory_bias
from constants import (APPLE_SATISFACTION, DECISION_SECONDS, MOVEMENT_SPEED, SOCIAL_PARTNER_RANDOMNESS,
    SOCIAL_START, STARTING_DAY, JUDY_ROOM_NUMBER, SOCIAL_RADIUS)
from interactions import company_point
from social import (concern_topic, goal_attempt_cooling_down, last_conversation_partner, pair_objective,
    pending_promises, repeat_partner_paused, regards_as_dangerous)

JUDY_ARRIVAL_OPENING = (
    "Hi, I'm {name}. Arriving here can be a lot to take in. "
    "You have a room in this building; the elevator connects the floors. "
    "The garden is outside through the first-floor exit, across the bridge. "
    "There's an apple tree there, and you can pick from it when there's fruit. "
    "That's all I've found here: the building and the garden. "
    "I don't know why we're here, or how to leave. How are you holding up?"
)


def arrival_greeter(world, *, include_archived=False):
    people = [*world.npcs, *world.archived] if include_archived else world.npcs
    return next((person for person in people
                 if person.room_number == JUDY_ROOM_NUMBER and person.arrival_day == STARTING_DAY), None)


def awaiting_arrival_greeting(npc, world):
    greeter = arrival_greeter(world)
    return (npc is not world.snake and greeter is not None and npc is not greeter and npc.arrival_day > STARTING_DAY
            and not any(greeter.id in record.participant_ids and npc.id in record.participant_ids
                        and any(turn.speaker_id == greeter.id for turn in record.turns)
                        for record in world.history))


def next_arrival_to_greet(world):
    return next((person for person in sorted(world.npcs, key=lambda person: person.arrival_day)
                 if awaiting_arrival_greeting(person, world)), None)


def arrival_greeting_reserved(npc, world):
    return awaiting_arrival_greeting(npc, world) or (
        npc is arrival_greeter(world) and next_arrival_to_greet(world) is not None)


def greet_arrivals(world):
    """Reserve Judy's first conversation immediately, before any other activity."""
    if world.phase_pending:
        return
    greeter = arrival_greeter(world)
    if greeter is None:
        return
    newcomer = next_arrival_to_greet(world)
    if newcomer is None:
        return
    if greeter.conversation and greeter.conversation.other(greeter) is newcomer:
        return
    reason = f"Welcoming the new arrival {newcomer.name} takes priority."
    world.interrupt_resident(greeter, reason, mode="arrival")
    from locations import clamp_point
    greeter.location = newcomer.location
    greeter.x, greeter.y = clamp_point(newcomer.location, (newcomer.x + SOCIAL_RADIUS, newcomer.y))
    if not world.company_available(newcomer, allow_arrival=True):
        return
    try:
        meeting = world.start_conversation(greeter, newcomer,
            f"Welcome {newcomer.name} and explain the known garden, apartment rooms and routes. Do not propose exploring for discoveries.",
            origin="autonomous", opening_kind="orientation")
    except ValueError:
        return
    for person in (greeter, newcomer):
        person.x, person.y = person.destination
    meeting.update(0)
    greeter.reason = f"Welcoming {newcomer.name}"
    newcomer.reason = f"Meeting {greeter.name} after arriving"


def target_valid(npc, world):
    if npc.behavior == Behavior.EAT:
        return npc.inventory.get("fruit", 0) > 0
    if npc.behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE):
        return (npc.target_id == world.tree.id
                and (npc.behavior == Behavior.GO_TO_TREE or world.available_fruit > 0)
                and npc.can_reach(world.tree.interaction_point))
    if npc.behavior == Behavior.COMPANY:
        partner = world.npc_by_id(npc.target_id)
        return (partner is not None and partner is not npc and world.company_available(partner)
                and company_point(npc, partner) is not None)
    return True


def reconsider(npc, world, *, pressing_only=False):
    from apartments import HOME_BEHAVIORS
    from player import owns_resident
    if arrival_greeting_reserved(npc, world):
        return
    if owns_resident(world, npc) or npc.journey or world.visit and world.visit.host_id == npc.id:
        return
    if (world.phase_pending or npc.interaction is not None or npc.resume_pending or npc.player_call or npc.behavior in HOME_BEHAVIORS
            or npc.home_reason == "night"):
        return
    if not target_valid(npc, world):
        npc.finish_activity(world, failed=True)
    if npc.owner != "autonomous" or npc.selected or npc.interaction_blocked:
        return
    if any(npc.id in (action.npc_id, action.target_id) for action in world.pending_actions):
        return
    if not pressing_only and npc.location.startswith("hallway:") and npc.behavior in (Behavior.WANDER, Behavior.WAIT):
        from constants import GARDEN_BUILDING_ARRIVAL
        from locations import start_journey
        start_journey(npc, "garden", GARDEN_BUILDING_ARRIVAL, purpose="leave_hallway")
        return
    if npc.decision_remaining > 0:
        return
    if npc.behavior in (Behavior.GO_TO_TREE, Behavior.HARVEST, Behavior.EAT):
        return

    obligations = sorted(pending_promises(world, npc.id), key=lambda promise: (promise.deadline, promise.created_phase))
    goal = npc.goal if npc.goal and npc.goal.status == "active" else None
    delivery_targets = [(promise.recipient_id, "Delivering a promised apple") for promise in obligations]
    goal_target = world.npc_by_id(goal.target_id) if goal else None
    from constants import DANGEROUS_OPTIONAL_GIFT_CHANCE
    if (goal and goal.kind == "help_with_apple" and goal_target and
            (not regards_as_dangerous(npc, goal_target) or npc.rng.random() < DANGEROUS_OPTIONAL_GIFT_CHANCE)):
        delivery_targets.append((goal.target_id, "Giving an apple to complete my objective"))
    if npc.inventory.get("fruit", 0) > 0 and world.company_available(npc):
        for target_id, reason in delivery_targets:
            target = world.npc_by_id(target_id)
            if (target is not None and world.company_available(target)
                    and target.id not in npc.cooldowns and npc.id not in target.cooldowns
                    and company_point(npc, target) is not None):
                gift = npc.give_items(target, "fruit", 1)
                gift.record.origin = "autonomous"
                world.history.append(gift.record)
                npc.reason = f"{reason} to {target.name}"
                return
    apples_needed = len({target_id for target_id, _ in delivery_targets})
    needs_apple = npc.inventory.get("fruit", 0) < apples_needed
    tree = world.tree
    tree_available = (world.is_outdoors(npc) and world.available_fruit > 0
                      and tree.id not in npc.cooldowns and npc.can_reach(tree.interaction_point))
    from constants import HARVEST_DISTANCE
    collection_behavior = (Behavior.HARVEST if npc.location == "garden" and
        math.dist((npc.x, npc.y), tree.interaction_point) <= HARVEST_DISTANCE else Behavior.GO_TO_TREE)
    if pressing_only:
        if obligations and needs_apple and tree_available:
            npc.begin_activity(collection_behavior, world, tree.id, "Collecting an apple to fulfill a commitment")
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
        score = 12 + (100 - npc.social) * 0.2 + npc.preferences["generosity"] * 8 - sum(npc.inventory.values()) * 4 - distance(tree.interaction_point) + (110 if needs_apple else 0)
        candidates.append((score, collection_behavior, tree.id,
                           "Harvesting an apple to help someone" if needs_apple else "Harvesting apples from the tree"))
    if npc.social <= SOCIAL_START:
        partners = [other for other in world.npcs if other is not npc and world.company_available(other)
                    and not regards_as_dangerous(npc, other)
                    and other.id not in npc.cooldowns and company_point(npc, other) is not None]
        if partners:
            def company_score(other):
                opinion = npc.opinion_score(other.id)
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
    if fruit < 1:
        return "unavailable", "Sharing is no longer possible because the inventories changed."
    opinion = donor.opinion_score(requester.id)
    from constants import DANGEROUS_GIFT_PENALTY
    score = ((donor.social - 50) * 0.12 + (fruit - 1) * 6
             + (donor.preferences["generosity"] - 0.5) * 10
             + opinion * 0.12 + memory_bias(donor, requester.id)
             - DANGEROUS_GIFT_PENALTY * regards_as_dangerous(donor, requester))
    if fruit == 1 and donor.social <= SOCIAL_START:
        score -= 8
    if score >= 0:
        return "accepted", "The donor is willing to share one apple."
    if fruit == 1 and donor.social <= SOCIAL_START:
        return "need", "The donor wants to eat their last apple themselves."
    return "refused", "The donor prefers to keep their apples this time."


def actionable_obligation(npc, world):
    from locations import available as available_outdoors
    if not available_outdoors(npc):
        return False
    # Preserve a free receiver for a delivery that is already physically possible,
    # even when the giver's ordinary decision timer has not elapsed yet.
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
                   and target.id not in npc.cooldowns and npc.id not in target.cooldowns
                   and company_point(npc, target) is not None for p in obligations)
    tree = world.tree
    return (world.available_fruit > 0
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
        from social import regards_as_dangerous
        from constants import DANGEROUS_COMPANY_PENALTY
        return (-DANGEROUS_COMPANY_PENALTY * regards_as_dangerous(npc, other) + npc.opinion_score(other.id) * 0.08
                + memory_bias(npc, other.id) - math.dist((npc.x, npc.y), (other.x, other.y)) / MOVEMENT_SPEED
                + npc.rng.uniform(0, SOCIAL_PARTNER_RANDOMNESS))

    from social import grievance_against, confrontation_topic
    for npc in available:
        if due is not None and npc.id not in due:
            continue
        for other in partners(npc):
            if repeat_partner_paused(npc, other):
                continue
            grievance = grievance_against(npc, other)
            if grievance:
                topic = confrontation_topic(npc, grievance)
                return npc, other, "confrontation", topic, f"Demanding an explanation from {other.name}", "news", grievance, None
    purposeful = []
    for npc in available:
        for other in partners(npc):
            if due is not None and npc.id not in due and other.id not in due:
                continue
            priority, objective = pair_objective(world, npc, other)
            if priority == 1 and (goal_attempt_cooling_down(npc, other, world) or repeat_partner_paused(npc, other)):
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
        if npc.social <= SOCIAL_START and not npc.inventory.get("fruit", 0):
            donors = [other for other in others if other.inventory.get("fruit", 0) > 0]
            if donors:
                donor = max(donors, key=lambda other: rank(npc, other))
                tree = world.tree
                if not (world.available_fruit > 0 and npc.can_reach(tree.interaction_point)
                        and math.dist((npc.x, npc.y), tree.interaction_point)
                        < math.dist((npc.x, npc.y), (donor.x, donor.y))):
                    return npc, donor, "food_request", "An apple", f"Asking {donor.name} for an apple", None, None, None
        partner = max(others, key=lambda other: (
            last_conversation_partner(npc) != other.id and last_conversation_partner(other) != npc.id,
            rank(npc, other)))
        concern = npc.current_concern
        topic = concern_topic(npc, concern)
        return npc, partner, "chat", topic, f"Seeking a chat with {partner.name}: {topic}", concern.kind, None, None
    return None
