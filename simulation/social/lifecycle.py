from simulation.social.information import trim_information
from simulation.social.goals import assign_goal, resolve_goal
from simulation.social.promises import pending_promises, resolve_promise
from simulation.social.relationships import apply_opinion_effects


def on_transfer(world, giver, recipient, quantity, event, record):
    # A transfer is consumed by the oldest matching commitments once only.
    matching = sorted((p for p in pending_promises(world, giver.id)
                       if p.recipient_id == recipient.id and p.created_day <= world.day <= p.deadline),
                      key=lambda p: p.created_day)
    for promise in matching[:quantity]:
        resolve_promise(world, promise, "fulfilled", event)
    for npc, partner, kind in ((giver, recipient, "help_with_apple"), (recipient, giver, "obtain_apple")):
        if npc is world.player:
            continue
        goal = npc.goal
        if (goal and goal.status == "active" and goal.kind == kind and goal.target_id == partner.id
                and goal.created_day <= world.day <= goal.deadline):
            resolve_goal(npc, world, "completed", "The apple was actually transferred.", record)


def advance_social_day(world):
    for promise in world.promises:
        if promise.status == "pending" and promise.deadline < world.day:
            resolve_promise(world, promise, "broken")
    for npc in world.npcs:
        goal = npc.goal
        if goal and goal.status == "active" and goal.deadline < world.day:
            resolve_goal(npc, world, "failed", "The deadline passed without completing the objective.")
        elif goal is None or (goal.status != "active" and goal.resolved_day < world.day):
            assign_goal(npc, world)
        apply_opinion_effects(npc)
    trim_information(world)
