"""Private objectives and apple commitments, resolved only by committed world changes."""

from dataclasses import asdict
from uuid import uuid4
from constants import (CONCERN_LIFETIME_RANGE, CONCERN_WEIGHTS, FAILURE_COOLDOWN, GOAL_IMPROVEMENT,
    MAX_DIALOGUE_FAILURE_COOLDOWN, PHASES, ROUTINE_FACT_KINDS)
from conversation_history import remember
from dialogue_rules import validate_apple_promise
from actor_identity import account_references
from social_records import (ConversationConcern, phase_label, Goal, Proposal, Promise, OpinionEffect, Claim, Belief,
    Disclosure, fact_by_id, claim_by_id, owns_recollection)
from information import (account_content, account_view, record_intention, can_deceive, selected_beliefs, related_beliefs,
    demote_contrary_beliefs, receive_belief, validate_information, trim_information, harvest_summary)


def valid_concern(npc, concern):
    if concern is None:
        return False
    if concern.kind not in ("recollection", "news"):
        return concern.claim_id is None and concern.kind in ("surroundings", "plans")
    belief = next((b for b in npc.beliefs if b.claim_id == concern.claim_id), None)
    claim = next((c for c in npc.world.claims if c.id == concern.claim_id), None)
    if belief is None or claim is None:
        return False
    fact = fact_by_id(npc.world, claim.fact_id)
    return concern.kind == "news" or owns_recollection(npc.id, belief, claim, fact)


def concern_key(npc, concern):
    if concern is None:
        return None
    claim = next((c for c in npc.world.claims if c.id == concern.claim_id), None)
    return ("account", claim.fact_id) if claim else (concern.kind, None)


def concern_topic(npc, concern):
    guidance = {
        "recollection": "Talk about a memory from my former life",
        "news": "Compare views on an account I know",
        "surroundings": "Talk about life here and what is on my mind",
        "plans": "Work out what to do next",
    }
    return guidance[concern.kind]


def last_conversation_partner(npc):
    return next((next(other.id for other in npc.world.actors if other is not npc and r.includes(other))
                 for r in reversed(npc.world.history)
                 if r.includes(npc) and "You" not in r.participants and r.turns), None)


def repeat_partner_paused(npc, other):
    return any(person.conversation_return_remaining > 0 and last_conversation_partner(person) == partner.id
               for person, partner in ((npc, other), (other, npc)))


def dialogue_failure_cooldown(npc, other):
    """Repeated failed meetings back off; any accepted exchange breaks the streak."""
    delay = FAILURE_COOLDOWN
    failures = 0
    for record in reversed(npc.world.history):
        if not record.includes(npc):
            continue
        if record.turns:
            break
        if record.between(npc, other) and record.error:
            failures += 1
            if failures > 1:
                delay = min(MAX_DIALOGUE_FAILURE_COOLDOWN, delay * 2)
            if delay == MAX_DIALOGUE_FAILURE_COOLDOWN:
                break
    return delay


def concern_description(npc, concern):
    """Observer text only; inspecting a concern never selects or consumes one."""
    if concern is None:
        return "No subject selected."
    descriptions = {
        "surroundings": "Reflecting on their surroundings and present situation.",
        "plans": "Considering what to do next.",
    }
    if concern.kind == "plans":
        plans = [npc.goal.objective] if npc.goal and npc.goal.status == "active" else []
        plans.extend(f"Deliver an apple to {npc.world.participant_by_id(p.recipient_id).name}, due {phase_label(p.deadline)}"
                     for p in pending_promises(npc.world, npc.id))
        return "Plans: " + "; ".join(plans) if plans else descriptions["plans"]
    if concern.kind in descriptions:
        return descriptions[concern.kind]
    claim = next((c for c in npc.world.claims if c.id == concern.claim_id), None)
    belief = next((b for b in npc.beliefs if b.claim_id == concern.claim_id), None)
    if claim is None or belief is None:
        return "The account they wanted to discuss is no longer available."
    account = account_view(npc.world, claim, belief)
    title = "Former-life recollection" if concern.kind == "recollection" else "News or a known account"
    basis = ("Firsthand experience" if belief.direct else
             f"Heard from {account['immediate_source_name']}" if belief.source_id != npc.id else "Their own assertion")
    return (f"{title}: {account['assertion']}\n"
            f"About {account['subject_name']}. {basis}. Stance: {belief.stance}.")


def danger_topic(npc, claim_id):
    claim = claim_by_id(npc.world, claim_id)
    return "Seek safety or help about a known alarming account: " + account_content(npc.world, fact_by_id(npc.world, claim.fact_id), claim.affirmed)


def undisclosed_danger(npc):
    """A new alarming fact can interrupt routine thoughts; a repeated source cannot."""
    world = npc.world
    discussed = {claim_by_id(world, d.claim_id).fact_id for d in world.disclosures if d.speaker_id == npc.id}
    return next((b.claim_id for b in reversed(npc.beliefs) if b.stance != "disbelieved"
                 and claim_by_id(world, b.claim_id).fact_id not in discussed
                 and fact_by_id(world, claim_by_id(world, b.claim_id).fact_id).kind in ("killing",)
                 and not any(r.turns and r.topic == danger_topic(npc, b.claim_id) and npc.name in r.participants
                             for r in world.history)), None)


def refresh_concern(npc):
    """Choose locally at a safe decision boundary, never during an interaction."""
    if npc.interaction is not None or npc.departure is not None:
        return
    previous = npc.current_concern
    danger = undisclosed_danger(npc)
    # Do not repeatedly replace a thought with another witness/denial of the same fact.
    if danger and (previous is None or concern_key(npc, previous) != ("account", claim_by_id(npc.world, danger).fact_id)):
        npc.current_concern = ConversationConcern("news", danger, npc.rng.uniform(*CONCERN_LIFETIME_RANGE))
        return
    if valid_concern(npc, previous) and previous.remaining > 0:
        return
    candidates = {"surroundings": [None], "plans": [None], "recollection": [], "news": []}
    for belief in npc.beliefs:
        claim = claim_by_id(npc.world, belief.claim_id)
        fact = fact_by_id(npc.world, claim.fact_id)
        if belief.direct and claim.affirmed and not claim.deliberate and fact.kind == "former_life" and fact.subject_id == npc.id:
            candidates["recollection"].append(claim.id)
        elif (fact.kind != "private_motive" or fact.subject_id != npc.id) and fact.kind not in ROUTINE_FACT_KINDS:
            # Someone walking past or chatting is not worth raising as news.
            candidates["news"].append(claim.id)
    retired = concern_key(npc, previous)
    candidates = {kind: [claim_id for claim_id in entries
                         if concern_key(npc, ConversationConcern(kind, claim_id, 0)) != retired]
                  for kind, entries in candidates.items()}
    kinds = [kind for kind, entries in candidates.items() if entries]
    kind = npc.rng.choices(kinds, weights=[CONCERN_WEIGHTS[k] for k in kinds])[0]
    npc.current_concern = ConversationConcern(kind, npc.rng.choice(candidates[kind]), npc.rng.uniform(*CONCERN_LIFETIME_RANGE))


def commit_concern_use(conversation):
    """Accepted base effects alone consume an account; private presence consumes nothing."""
    world = conversation.initiator.world
    options = {o.id: o for o in conversation.information_context.options} if conversation.information_context else {}
    for role, npc in (("initiator", conversation.initiator), ("recipient", conversation.recipient)):
        concern = conversation.concerns[role]
        if concern is None or npc.current_concern != concern:
            continue
        if concern.claim_id:
            fact_id = claim_by_id(world, concern.claim_id).fact_id
            used = any(conversation.turns[d.turn].speaker == role and options[d.claim_id].fact_id == fact_id
                       for d in conversation.accepted_disclosures)
        else:
            used = (role == "initiator" and conversation.opening_kind == concern.kind
                    and conversation.purpose == "chat" and conversation.origin == "autonomous")
        if used:
            npc.current_concern.remaining = 0.0


def commit_information(conversation):
    context = conversation.information_context
    if context is None:
        return
    world = conversation.initiator.world
    options = {option.id: option for option in context.options}
    people = {"initiator": conversation.initiator, "recipient": conversation.recipient}
    heard, alleged = {}, []
    claims = {claim.id: claim for claim in world.claims}
    for turn, speech in enumerate(conversation.turns):
        speaker = people[speech.speaker]
        listener = conversation.other(speaker)
        for use in [entry for entry in conversation.leverage_uses if entry.turn == turn]:
            text = f"{speaker.name} pressured {listener.name} for {use.demand} using a private account; cooperation is not established."
            event = world.publish("leverage_used", (speaker, listener), text, conversation.record)
            conversation.record.outcomes.append(text)
            for npc, partner in ((speaker, listener), (listener, speaker)):
                remember(npc, partner, event, "leverage", text, 0)
        for entry in [entry for entry in conversation.third_party_opinions if entry.turn == turn]:
            claim = claim_by_id(world, entry.claim_id)
            belief = next(b for b in speaker.beliefs if b.claim_id == entry.claim_id)
            source = heard[speaker.id, entry.claim_id]
            originator = world.participant_by_id(claim.originator_id)
            option = options[claim.id]
            reason = (f"Heard {source.name}'s account (original source: {originator.name}): {option.assertion} "
                      f"Personal assessment: {entry.reason}")[:400]
            event = world.publish("claim_assessed", (source, speaker),
                                  f"{speaker.name} assessed an allegation about {world.participant_by_id(entry.subject_id).name} from {source.name}.",
                                  conversation.record)
            belief.assessed = True
            belief.opinion_delta += entry.delta
            speaker.pending_social_opinions.append(OpinionEffect(event.id, entry.subject_id, entry.delta,
                                                                  reason, claim.id, source.id, conversation.record.id, world.phase_index))
        for entry in [entry for entry in conversation.accepted_disclosures if entry.turn == turn]:
            option = options[entry.claim_id]
            source_belief = next((b for b in speaker.beliefs if b.claim_id == option.id), None)
            if not any(claim.id == option.id for claim in world.claims):
                parent = next(b for b in speaker.beliefs if b.claim_id == option.parent_claim_id)
                source_belief = parent
                world.claims.append(Claim(option.id, option.fact_id, option.subject_id, speaker.id,
                                          option.affirmed, option.parent_claim_id, True,
                                          world.phase_index, conversation.record.id))
                claims[option.id] = world.claims[-1]
                speaker.beliefs.append(Belief(speaker.id, option.id, "disbelieved", speaker.id, None,
                                              False, world.phase_index))
            pickups = _witnessed_pickups(world, speaker, option, claims) if entry.intent == "share" else []
            assertion = option.assertion
            if pickups:
                # Speech reports the speaker's witnessed total, so the record and the listener's knowledge do too.
                total = sum(fact_by_id(world, claim.fact_id).quantity for claim in [claims[option.id], *pickups])
                assertion = harvest_summary(world.participant_by_id(option.subject_id).name, total)
            text = f"{speaker.name} {'questioned an account with' if entry.intent == 'challenge' else 'alleged to'} {listener.name}: {assertion}"
            event = world.publish("claim_shared", (speaker, listener), text, conversation.record)
            disclosure = Disclosure(f"{conversation.id}:disclosure:{turn}", option.id, conversation.record.id,
                                    speaker.id, listener.id, turn, entry.intent, entry.listener_stance,
                                    source_belief.disclosure_id, world.phase_index, event.id)
            world.disclosures.append(disclosure)
            prior = next((b for b in listener.beliefs if b.claim_id == option.id), None)
            witnessed = any(b.direct and claims[b.claim_id].fact_id == option.fact_id for b in listener.beliefs if b.claim_id in claims)
            if entry.intent == "share":
                alleged.append((turn, listener, option))
            if entry.listener_stance == "believed":
                demote_contrary_beliefs(option, related_beliefs(option, listener.beliefs, claims), claims, listener.id)
            learned = receive_belief(prior, listener_id=listener.id, claim_id=option.id,
                stance=entry.listener_stance, source_id=speaker.id, disclosure_id=disclosure.id, phase=world.phase_index)
            if prior is None:
                listener.beliefs.append(learned)
            for claim in pickups:
                # Each pickup the telling covers is its own disclosed account, so hearsay keeps an exact source.
                told = Disclosure(f"{disclosure.id}:{claim.id}", claim.id, conversation.record.id, speaker.id, listener.id,
                                  turn, entry.intent, entry.listener_stance, None, world.phase_index, event.id)
                world.disclosures.append(told)
                prior_pickup = next((b for b in listener.beliefs if b.claim_id == claim.id), None)
                learned_pickup = receive_belief(prior_pickup, listener_id=listener.id, claim_id=claim.id, stance=entry.listener_stance,
                                                source_id=speaker.id, disclosure_id=told.id, phase=world.phase_index)
                if prior_pickup is None:
                    listener.beliefs.append(learned_pickup)
            heard[listener.id, option.id] = speaker
            originator = world.participant_by_id(option.originator_id)
            remember(speaker, listener, event, "claim_shared", f"I discussed an account with {listener.name}: {assertion}", 0)
            remember(listener, speaker, event, "heard_claim",
                     f"{speaker.name} passed on an account from {originator.name}: {assertion} My stance: {entry.listener_stance}.", 0)
            # A witness can only believe what they saw; that is not the same as being persuaded.
            stance = f"{entry.listener_stance} (saw it firsthand)" if witnessed else entry.listener_stance
            conversation.record.outcomes.append(f"Allegation: {speaker.name} -> {listener.name}; {stance}. {assertion}")
    for npc in people.values():
        apply_opinion_effects(npc)
    for turn, listener, option in alleged:
        unweighed_allegation(conversation, turn, listener, option, claims)


def unweighed_allegation(conversation, turn, listener, option, claims):
    """Show why a believed allegation left the absent subject's standing alone when the listener gave it no verdict."""
    world = listener.world
    subject = world.participant_by_id(option.subject_id)
    belief = next((b for b in listener.beliefs if b.claim_id == option.id), None)
    if (listener is world.player or subject is None or subject in (conversation.initiator, conversation.recipient)
            or belief is None or belief.stance != "believed"
            or any(conversation.turns[e.turn].speaker != conversation.turns[turn].speaker
                   and claims[e.claim_id].fact_id == option.fact_id for e in conversation.third_party_opinions)):
        return
    related = [b for b in listener.beliefs if b.claim_id in claims and claims[b.claim_id].fact_id == option.fact_id]
    if any(b.assessed for b in related):
        why = f"{listener.name} had already judged {subject.name} over this in an earlier conversation."
    elif any(b.direct for b in related) and option.kind != "harvest":
        why = f"{listener.name} saw this firsthand, so hearing it retold changed nothing."
    else:
        why = f"{listener.name} formed no new view of {subject.name} from it."
    conversation.record.outcomes.append(
        f"{listener.name}'s opinion of {subject.name} changed +0 after assessing an allegation. {why}")


def _witnessed_pickups(world, speaker, option, claims):
    """The speaker's claims about the same collector's other pickups that phase, which a firsthand telling includes."""
    fact = fact_by_id(world, option.fact_id)
    if (fact.kind != "harvest" or not option.affirmed or world.participant_by_id(fact.subject_id) is None
            or not any(b.direct and claims[b.claim_id].fact_id == fact.id for b in speaker.beliefs if b.claim_id in claims)):
        return []
    pickups = {}
    for belief in speaker.beliefs:
        claim = claims.get(belief.claim_id)
        if claim is None or not belief.direct or not claim.affirmed or claim.fact_id == fact.id:
            continue
        other = fact_by_id(world, claim.fact_id)
        if other.kind == "harvest" and other.subject_id == fact.subject_id and other.phase == fact.phase:
            pickups.setdefault(other.id, claim)
    return list(pickups.values())


def check_sabotage_goal(world, effect, listener, actual_delta, record):
    owner = world.participant_by_id(effect.actor_id)
    goal = owner.goal if owner else None
    if (goal and goal.status == "active" and goal.kind == "sabotage_relationship"
            and goal.target_id == listener.id and goal.third_npc_id == effect.target_id and actual_delta < 0
            and effect.phase is not None and goal.created_phase <= effect.phase <= world.phase_index <= goal.deadline
            and claim_by_id(world, goal.claim_id).fact_id == claim_by_id(world, effect.claim_id).fact_id
            and listener.opinion_score(effect.target_id) <= goal.baseline_opinion - 4):
        resolve_goal(owner, world, "completed", "A committed opinion change from this exchange weakened the relationship.", record)


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
            if subject is npc or subject is world.snake or subject is world.player or subject.departure or belief.stance == "disbelieved":
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
    if third:
        descriptions["sabotage_relationship"] = (
            f"Lower {target.name}'s opinion of {third.name} by 4 points through a supported account",
            "I want to weaken their connection so my own influence matters more.",
            "Use an account I know, or a permitted denial of it; persuasion can fail.")
    objective, motive, approach = descriptions[kind]
    npc.goal = Goal(uuid4().hex, kind, target.id, objective, motive, approach,
                    "Respect refusal; use only supported accounts or permitted denials, without inventing other events.",
                    world.phase_index, world.phase_index + len(PHASES), baseline,
                    third_npc_id=third.id if third else None, claim_id=claim.id if claim else None)
    record_intention(npc, world)


def resolve_goal(npc, world, status, outcome, record=None):
    goal = npc.goal
    if goal is None or goal.status != "active":
        return
    target = world.participant_by_id(goal.target_id)
    goal.status, goal.outcome = status, outcome
    goal.resolved_phase = world.phase_index
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
                and goal.target_id == opinion_owner.id and world.phase_index <= goal.deadline
                and opinion_owner.opinion_score(npc.id)
                >= goal.baseline_opinion + GOAL_IMPROVEMENT):
            resolve_goal(npc, world, "completed", "Their opinion improved through a real interaction.", record)


def permitted_proposals(conversation):
    if conversation.purpose in ("food_request", "confrontation") or conversation.opening_kind == "orientation" or conversation.initiator.speech_only or conversation.recipient.speech_only:
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


def proposals_from_snapshot(snapshot):
    """Project proposal boundary data, leaving prompt-only labels at the boundary."""
    return [Proposal(row["id"], row["promisor_id"], row["recipient_id"], row["deadline"])
            for row in snapshot.get("permitted_proposals", [])]


def validate_agreements(agreements, proposals, turns, initiator_id, recipient_id, *, current_phase=None,
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
                                   current_phase=current_phase)


def validate_live_agreements(conversation, response, *, dialogue_validation=True):
    world = conversation.initiator.world
    pair = (conversation.initiator, conversation.recipient)
    if any(npc.conversation is not conversation or world.actor_by_id(npc.id) is not npc or npc.departure or npc.location != pair[0].location for npc in pair):
        raise ValueError("The conversation participants are no longer available.")
    if conversation.context_references != conversation_references(conversation):
        raise ValueError("An objective or promise changed while the reply was being generated. Please try again.")
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
            opinions = {other.id: npc.opinion_score(other.id)
                        for other in world.actors if other is not npc}
            if changed or related or opinions != context.opinions[role] or can_deceive(npc) != context.can_lie[role]:
                raise ValueError("A speaker's knowledge or opinion changed during generation. Please try again.")
    validate_information(response, context, names=[actor.name for actor in world.actors],
                         participant_names={"initiator": pair[0].name, "recipient": pair[1].name},
                         dialogue_validation=dialogue_validation, actor_references=account_references(conversation.name_actors))
    validate_agreements(response.accepted_proposals, conversation.proposals,
                        response.turns, pair[0].id, pair[1].id, current_phase=world.phase_index,
                        dialogue_validation=dialogue_validation)
    pending = [(p.promisor_id, p.recipient_id) for p in pending_promises(world)]
    for agreement in response.accepted_proposals:
        proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
        if (conversation.purpose == "food_request" or world.phase_index > proposal.deadline
                or {proposal.promisor_id, proposal.recipient_id} != {npc.id for npc in pair}
                or (proposal.promisor_id, proposal.recipient_id) in pending
                or sum(giver == proposal.promisor_id for giver, _ in pending) >= 2):
            raise ValueError("The promise's deadline or participants changed. Please try the conversation again.")
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
                          conversation.record.id, world.phase_index, proposal.deadline)
        world.promises.append(promise)
        giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
        text = f"{giver.name} promised {recipient.name} one apple by {phase_label(promise.deadline)}."
        event = world.publish("promise_created", (giver, recipient), text, conversation.record)
        conversation.record.outcomes.append(text)
        for npc, partner in ((giver, recipient), (recipient, giver)):
            remember(npc, partner, event, "promise_created", text, 1)


def apply_opinion_effects(npc):
    while npc.pending_social_opinions:
        effect = npc.pending_social_opinions[0]
        target = npc.world.participant_by_id(effect.target_id)
        if not effect.delta and effect.claim_id:
            # A weighed account that left the view unchanged is shown, but the standing and its reason stay as they were.
            record = next(entry for entry in npc.world.history if entry.id == effect.conversation_id)
            record.outcomes.append(f"{npc.name}'s opinion of {target.name} changed +0 after assessing an allegation. {effect.reason}")
            npc.pending_social_opinions.pop(0)
            continue
        if target is npc.world.player:
            from player import update_player_opinion
            update_player_opinion(npc, effect.delta, effect.reason)
            npc.pending_social_opinions.pop(0)
            continue
        previous = npc.opinions.get(target.id, {"opinion": 0, "reason": ""})
        # Keep the specific unresolved outcome when later impressions are added.
        reason = effect.reason + (f" Earlier: {previous['reason']}" if previous["reason"] else "")
        try:
            npc.remember_opinions({target.id: {
                "opinion": max(-100, min(100, previous["opinion"] + effect.delta)), "reason": reason[:400]}})
        except ValueError:
            return False
        npc.pending_social_opinions.pop(0)
        if effect.claim_id:
            record = next(entry for entry in npc.world.history if entry.id == effect.conversation_id)
            actual_delta = npc.opinions[target.id]["opinion"] - previous["opinion"]
            text = f"{npc.name}'s opinion of {target.name} changed {actual_delta:+d} after assessing an allegation. {effect.reason}"
            event = npc.world.publish("reputation_changed", (npc, target), text, record)
            record.outcomes.append(text)
            remember(npc, target, event, "claim_opinion", effect.reason, max(-4, min(4, actual_delta)))
            check_sabotage_goal(npc.world, effect, npc, actual_delta, record)
    return True


def resolve_promise(world, promise, status, transfer=None):
    if promise.status != "pending":
        return
    giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
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
        if npc is world.player:
            continue
        remember(npc, partner, event, "promise_fulfilled" if status == "fulfilled" else "promise_broken",
                 text, 4 if status == "fulfilled" else -5)
    if recipient is not world.player:
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
        if npc is world.player:
            continue
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
    trim_information(world)


def private_context(npc, partner_id=None):
    world = npc.world
    promises = [p for p in world.promises if npc.id in (p.promisor_id, p.recipient_id)] if world else []
    promises.sort(key=lambda p: (p.status == "pending", partner_id in (p.promisor_id, p.recipient_id), p.created_phase), reverse=True)
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
            "owner_id": npc.id, "visibility": "Only this speaker knows their private motive and memories.",
            "concern": ({"kind": npc.current_concern.kind, "claim_id": npc.current_concern.claim_id,
                         "guidance": concern_topic(npc, npc.current_concern)} if npc.current_concern else None),
            "goal": goal_view,
            "known_accounts": known,
            "promises": [dict(asdict(p), deadline_label=phase_label(p.deadline)) for p in promises[:8]]}


def pair_objective(world, npc, other):
    pair = {npc.id, other.id}
    def discussed(topic):
        return any(r.turns and set(r.participants) == {npc.name, other.name} and r.topic == topic for r in world.history)
    promises = [p for p in world.promises if {p.promisor_id, p.recipient_id} == pair]
    pending = [p for p in promises if p.status == "pending"]
    if pending:
        promise = min(pending, key=lambda p: p.deadline)
        giver, recipient = world.participant_by_id(promise.promisor_id), world.participant_by_id(promise.recipient_id)
        topic = f"Arrange {giver.name}'s promised apple for {recipient.name}, due {phase_label(promise.deadline)}"
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
    if danger:
        return 2, danger_topic(npc, danger)
    goal = npc.goal
    if goal and goal.status == "active" and goal.target_id == other.id:
        return 1, f"{goal.objective}, by {phase_label(goal.deadline)}"
    return 0, ""


def goal_attempt_cooling_down(npc, other, world):
    goal = npc.goal
    return bool(goal and goal.status == "active" and goal.target_id == other.id
                and goal.last_attempt_phase is not None and world.phase_index < goal.last_attempt_phase + 2)


def grievance_against(npc, target):
    """A retained believed allegation, with its original source and uncertainty intact."""
    from constants import CONFRONTATION_START_OPINION, GRIEVANCE_FACT_KINDS, CONFRONTATION_GRIEVANCE_PHASE_PAUSE
    if npc.opinion_score(target.id) > CONFRONTATION_START_OPINION:
        return None
    for belief in reversed(npc.beliefs):
        claim = claim_by_id(npc.world, belief.claim_id)
        fact = fact_by_id(npc.world, claim.fact_id)
        if (belief.stance != "believed" or not claim.affirmed or fact.subject_id != target.id
                or fact.kind not in GRIEVANCE_FACT_KINDS):
            continue
        latest = next((r for r in reversed(npc.world.history) if r.purpose == "confrontation"
                       and r.between(npc, target) and r.grievance_id
                       and claim_by_id(npc.world, r.grievance_id).fact_id == fact.id and r.turns), None)
        if latest is not None:
            discussed_phase = max((d.phase for d in npc.world.disclosures
                if d.conversation_id == latest.id and d.speaker_id == npc.id
                and claim_by_id(npc.world, d.claim_id).fact_id == fact.id), default=npc.world.phase_index)
            if npc.world.phase_index - discussed_phase < CONFRONTATION_GRIEVANCE_PHASE_PAUSE:
                continue
        return claim.id
    return None


def confrontation_topic(npc, claim_id):
    belief = next(b for b in npc.beliefs if b.claim_id == claim_id)
    account = account_view(npc.world, claim_by_id(npc.world, claim_id), belief)
    return f"Seek an explanation for this unresolved account: {account['assertion']} Source: {account['immediate_source_name']}; stance: {account['stance']}."


def regards_as_dangerous(npc, target):
    return npc.opinion_score(target.id) < 0 and any(
        b.stance == "believed" and (c := claim_by_id(npc.world, b.claim_id)).affirmed
        and (f := fact_by_id(npc.world, c.fact_id)).kind == "killing" and f.subject_id == target.id
        for b in npc.beliefs)


def validate_escalation_snapshot(decision, snapshot, turns, opinion):
    choices = snapshot.get("escalation_choices")
    if decision is None:
        if choices:
            raise ValueError("The confrontation initiator must choose a peaceful leave or an available attack.")
        return
    if not choices or snapshot.get("purpose") != "confrontation":
        raise ValueError("Only an eligible confrontation initiator may select escalation.")
    if (decision.target_id != choices["target_id"] or decision.grievance_id != choices["grievance_id"]
            or decision.action not in choices["actions"]):
        raise ValueError("Escalation must match the supplied target, grievance and actions.")
    if not any(t.speaker == "recipient" for t in turns):
        raise ValueError("The target must have a real opportunity to answer.")
    if decision.action == "attack":
        from constants import CONFRONTATION_OPINION
        if opinion > CONFRONTATION_OPINION or not decision.evidence.strip() or not any(
                decision.evidence in t.text for t in turns if t.speaker == "recipient"):
            raise ValueError("An attack requires final hostility and exact evidence from the target's accepted answer.")


def validate_escalation(conversation, response):
    validate_escalation_snapshot(response.escalation,
        {"purpose": conversation.purpose, "escalation_choices": conversation.initiator.world.escalation_choices(conversation)},
        response.turns, response.initiator_opinion.opinion)
