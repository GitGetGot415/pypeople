from dialogue.history import remember
from dialogue.rules import validate_apple_promise
from dialogue.identity import account_references
from simulation.social.records import Proposal, Promise, OpinionEffect, claim_by_id
from simulation.social.information import can_deceive, validate_information
from simulation.social.relationships import apply_opinion_effects


def pending_promises(world, promisor_id=None):
    return [promise for promise in world.promises if promise.status == "pending"
            and (promisor_id is None or promise.promisor_id == promisor_id)]


def permitted_proposals(conversation):
    if conversation.purpose == "food_request" or conversation.opening_kind == "orientation" or conversation.initiator.speech_only or conversation.recipient.speech_only:
        return []
    world = conversation.initiator.world
    proposals = []
    goal = conversation.initiator.goal
    goal_pair = None
    if conversation.purpose == "confrontation":
        # The accused can make the grievance right only by promising the confronter an apple.
        goal_pair = (conversation.recipient.id, conversation.initiator.id)
    elif goal and goal.status == "active" and goal.target_id == conversation.recipient.id:
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
            proposals.append(Proposal(f"{conversation.id}:{role}_gives", giver.id, recipient.id, world.day + 1))
    return proposals


def proposals_from_snapshot(snapshot):
    """Project proposal boundary data, leaving prompt-only labels at the boundary."""
    return [Proposal(row["id"], row["promisor_id"], row["recipient_id"], row["deadline"])
            for row in snapshot.get("permitted_proposals", [])]


def validate_agreements(agreements, proposals, turns, initiator_id, recipient_id, *, current_day=None,
                        dialogue_validation=True):
    """Bound offers and acceptances to ordered speech by the correct participants."""
    permitted = {proposal.id: proposal for proposal in proposals}
    seen = set()
    for agreement in agreements:
        proposal = permitted.get(agreement.proposal_id)
        if proposal is None or agreement.proposal_id in seen:
            raise ValueError("The agreement refers to an unavailable or repeated proposal.")
        seen.add(agreement.proposal_id)
        if not 0 <= agreement.offer_turn < agreement.accept_turn < len(turns):
            raise ValueError(
                f"Invalid promise indices: offer_turn={agreement.offer_turn}, accept_turn={agreement.accept_turn}. "
                f"This exchange has {len(turns)} turns, indexed 0 through {len(turns) - 1}. "
                "Choose an existing giver's offer followed by the receiver's acceptance in a later turn, "
                "or use accepted_proposals=[] if no agreement was made."
            )
        speakers = {"initiator": initiator_id, "recipient": recipient_id}
        if (speakers[turns[agreement.offer_turn].speaker] != proposal.promisor_id
                or speakers[turns[agreement.accept_turn].speaker] != proposal.recipient_id):
            raise ValueError("Only the apple's giver can offer and its recipient can accept.")
        if dialogue_validation:
            validate_apple_promise(turns[agreement.offer_turn].text, turns[agreement.accept_turn].text, proposal.deadline,
                                   current_day=current_day)


def validate_live_agreements(conversation, response, *, dialogue_validation=True):
    world = conversation.initiator.world
    pair = (conversation.initiator, conversation.recipient)
    if any(npc.conversation is not conversation or world.actor_by_id(npc.id) is not npc or not npc.alive or npc.location != pair[0].location for npc in pair):
        raise ValueError("The conversation participants are no longer available.")
    if conversation.context_references != conversation_references(conversation):
        raise ValueError("An objective or promise changed while the reply was being generated.")
    context = conversation.information_context
    if context is not None:
        # Keep the original permitted choices. Reselecting them after a nearby
        # event rejected valid replies just because new accounts ranked higher.
        relevant_facts = {option.fact_id for option in context.options
                          if any(option.id in ids for ids in context.allowed.values())}
        for role, npc in (("initiator", pair[0]), ("recipient", pair[1])):
            before = {belief.claim_id: belief for belief in context.beliefs[role]}
            current = {belief.claim_id: belief for belief in npc.beliefs}
            changed = any(current.get(claim_id) != belief for claim_id, belief in before.items())
            related = any(b.claim_id not in before and claim_by_id(world, b.claim_id).fact_id in relevant_facts
                          for b in npc.beliefs)
            # Opinions may move meanwhile (a harvest seen); the result is applied as a change, so only knowledge counts.
            if changed or related or can_deceive(npc) != context.can_lie[role]:
                raise ValueError("A speaker's knowledge changed during generation.")
    validate_information(response, context, names=[actor.name for actor in world.actors],
                         participant_names={"initiator": pair[0].name, "recipient": pair[1].name},
                         dialogue_validation=dialogue_validation, actor_references=account_references(conversation.name_actors))
    validate_agreements(response.accepted_proposals, conversation.proposals,
                        response.turns, pair[0].id, pair[1].id, current_day=world.day,
                        dialogue_validation=dialogue_validation)
    pending = [(p.promisor_id, p.recipient_id) for p in pending_promises(world)]
    for agreement in response.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        if (conversation.purpose == "food_request" or world.day > proposal.deadline
                or {proposal.promisor_id, proposal.recipient_id} != {npc.id for npc in pair}
                or (proposal.promisor_id, proposal.recipient_id) in pending
                or sum(giver == proposal.promisor_id for giver, _ in pending) >= 2):
            raise ValueError("The promise's deadline or participants changed.")
        pending.append((proposal.promisor_id, proposal.recipient_id))


def conversation_references(conversation):
    pair = (conversation.initiator, conversation.recipient)
    ids = {npc.id for npc in pair}
    # Only the objective that actually initiated this meeting can invalidate it.
    # Other personal goals and promises to third parties may progress elsewhere.
    references = {npc.goal.id: npc.goal.status for npc in pair
                  if npc.goal and npc.goal.id == conversation.goal_id}
    references.update({p.id: p.status for p in pair[0].world.promises
                       if ids == {p.promisor_id, p.recipient_id}})
    return references


def commit_agreements(conversation):
    world = conversation.initiator.world
    for agreement in conversation.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        promise = Promise(proposal.id, proposal.promisor_id, proposal.recipient_id,
                          conversation.record.id, world.day, proposal.deadline)
        world.promises.append(promise)
        giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
        text = f"{giver.name} promised {recipient.name} one apple by {promise.deadline}."
        event = world.publish("promise_created", (giver, recipient), text, conversation.record)
        conversation.record.outcomes.append(text)
        for npc, partner in ((giver, recipient), (recipient, giver)):
            remember(npc, partner, event, "promise_created", text, 1)


def resolve_promise(world, promise, status, transfer=None):
    if promise.status != "pending":
        return
    giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
    record = next(entry for entry in world.history if entry.id == promise.conversation_id)
    promise.status, promise.resolved_day = status, world.day
    text = (f"{giver.name} kept the apple promise to {recipient.name} with an actual handoff."
            if status == "fulfilled" else
            f"{giver.name} missed the apple promise to {recipient.name}, due {promise.deadline}; the reason is unknown.")
    event = world.publish("promise_fulfilled" if status == "fulfilled" else "promise_broken",
                          (giver, recipient), text, record)
    promise.consequence_event_id = event.id
    promise.resolution_event_id = transfer.id if transfer is not None else event.id
    promise.outcome = text
    record.outcomes.append(text)
    for npc, partner in ((giver, recipient), (recipient, giver)):
        if npc is world.player:
            continue
        remember(npc, partner, event, "promise_fulfilled" if status == "fulfilled" else "promise_broken",
                 text, 4 if status == "fulfilled" else -5)
    if recipient is not world.player:
        recipient.pending_social_opinions.append(OpinionEffect(event.id, giver.id, 4 if status == "fulfilled" else -6, text))
        apply_opinion_effects(recipient)
