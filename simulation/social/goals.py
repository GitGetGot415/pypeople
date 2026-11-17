from uuid import uuid4
from constants import GOAL_IMPROVEMENT
from dialogue.history import remember
from simulation.social.records import Goal, fact_by_id, claim_by_id
from simulation.social.information import record_intention


def check_sabotage_goal(world, effect, listener, actual_delta, record):
    owner = world.participant_by_id(effect.actor_id)
    goal = owner.goal if owner else None
    if (goal and goal.status == "active" and goal.kind == "sabotage_relationship"
            and goal.target_id == listener.id and goal.third_npc_id == effect.target_id and actual_delta < 0
            and effect.day is not None and goal.created_day <= effect.day <= world.day <= goal.deadline
            and claim_by_id(world, goal.claim_id).fact_id == claim_by_id(world, effect.claim_id).fact_id
            and listener.opinion_score(effect.target_id) <= goal.baseline_opinion - 4):
        resolve_goal(owner, world, "completed", "A committed opinion change from this exchange weakened the relationship.", record)


def assign_goal(npc, world):
    from simulation.interactions import company_point

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
            if subject is npc or subject is world.snake or subject is world.player or not subject.alive or belief.stance == "disbelieved":
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
                         else "I would rather get an apple from someone else than give up anything of mine.",
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
                    world.day, world.day + 1, baseline,
                    third_npc_id=third.id if third else None, claim_id=claim.id if claim else None)
    record_intention(npc, world)


def resolve_goal(npc, world, status, outcome, record=None):
    goal = npc.goal
    if goal is None or goal.status != "active":
        return
    target = world.participant_by_id(goal.target_id)
    goal.status, goal.outcome = status, outcome
    goal.resolved_day = world.day
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
                and goal.target_id == opinion_owner.id and world.day <= goal.deadline
                and opinion_owner.opinion_score(npc.id)
                >= goal.baseline_opinion + GOAL_IMPROVEMENT):
            resolve_goal(npc, world, "completed", "Their opinion improved through a real interaction.", record)


def goal_attempt_cooling_down(npc, other, world):
    goal = npc.goal
    return bool(goal and goal.status == "active" and goal.target_id == other.id
                and goal.last_attempt_day is not None and world.day < goal.last_attempt_day + 2)
