from constants import FAILURE_COOLDOWN, MAX_DIALOGUE_FAILURE_COOLDOWN
from dialogue.history import remember
from simulation.social.records import fact_by_id, claim_by_id
from simulation.social.information import account_view, believed_harvest
from simulation.social.goals import check_sabotage_goal


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
            from simulation.player import update_player_opinion
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


def opinion_causes(npc, partner):
    """What this resident holds against a partner, from recorded facts rather than any written reason.

    A conversation rewrites the stored reason, but these causes stay, so speech can still show why.
    """
    from constants import HARVEST_FREE_APPLES, ACCUSATION_FACT_KINDS
    world = npc.world
    if world is None:
        return []
    harvests, causes = {}, []
    claims = {claim.id: claim for claim in world.claims}
    seen = {claims[b.claim_id].fact_id for b in npc.beliefs if b.direct and b.claim_id in claims}
    for told in world.disclosures:
        if (told.intent == "lie" and (told.speaker_id, told.listener_id) == (partner.id, npc.id) and told.claim_id in claims
                and claims[told.claim_id].fact_id in seen):
            causes.append(f"{partner.name} lied to you, denying what you saw yourself: {fact_by_id(world, claims[told.claim_id].fact_id).summary}")
    for fact in world.facts:
        if fact.kind == "harvest" and fact.subject_id == partner.id and npc.id in fact.witness_ids:
            harvests[fact.day] = harvests.get(fact.day, 0) + fact.quantity
    causes.extend(f"You saw {partner.name} take {total} apples from the tree at {day}."
                  for day, total in sorted(harvests.items(), reverse=True) if total > HARVEST_FREE_APPLES)
    causes.extend(p.outcome for p in world.promises if p.status == "broken" and p.outcome
                  and (p.promisor_id, p.recipient_id) == (partner.id, npc.id))
    for belief in sorted(reversed(npc.beliefs), key=lambda b: fact_by_id(npc.world, claim_by_id(npc.world, b.claim_id).fact_id).kind == "killing", reverse=True):
        claim = claim_by_id(world, belief.claim_id)
        fact = fact_by_id(world, claim.fact_id)
        if (belief.stance != "believed" or not claim.affirmed or fact.subject_id != partner.id
                or fact.kind not in ACCUSATION_FACT_KINDS or fact.kind == "harvest" and belief.direct):
            continue
        source = world.participant_by_id(belief.source_id)
        causes.append(f"You saw this yourself: {fact.summary}" if belief.direct else
                      f"{source.name if source else 'Someone'} told you: {fact.summary}")
    return list(dict.fromkeys(causes))[:3]


def grievance_against(npc, target):
    """A retained believed allegation, with its original source and uncertainty intact."""
    from constants import CONFRONTATION_START_OPINION, GRIEVANCE_FACT_KINDS, CONFRONTATION_GRIEVANCE_DAY_PAUSE
    if npc.opinion_score(target.id) > CONFRONTATION_START_OPINION:
        return None
    def priority(belief):
        fact = fact_by_id(npc.world, claim_by_id(npc.world, belief.claim_id).fact_id)
        return fact.kind == "killing", fact.day, belief.direct
    for belief in sorted(reversed(npc.beliefs), key=priority, reverse=True):
        claim = claim_by_id(npc.world, belief.claim_id)
        fact = fact_by_id(npc.world, claim.fact_id)
        if (belief.stance != "believed" or not claim.affirmed or fact.subject_id != target.id
                or fact.kind not in GRIEVANCE_FACT_KINDS):
            continue
        latest = next((r for r in reversed(npc.world.history) if r.purpose == "confrontation"
                       and r.between(npc, target) and r.grievance_id
                       and claim_by_id(npc.world, r.grievance_id).fact_id == fact.id and r.turns), None)
        if latest is not None and target is not npc.world.player:
            discussed_day = max((d.day for d in npc.world.disclosures
                if d.conversation_id == latest.id and d.speaker_id == npc.id
                and claim_by_id(npc.world, d.claim_id).fact_id == fact.id), default=npc.world.day)
            if npc.world.day - discussed_day < CONFRONTATION_GRIEVANCE_DAY_PAUSE:
                continue
        return claim.id
    return None


def confrontation_topic(npc, claim_id):
    belief = next(b for b in npc.beliefs if b.claim_id == claim_id)
    claim = claim_by_id(npc.world, claim_id)
    account = account_view(npc.world, claim, belief)
    fact = fact_by_id(npc.world, claim.fact_id)
    # A harvest is raised as the whole collection this resident believes, not the one pickup the grievance names.
    assertion = believed_harvest(npc.world, npc.beliefs, fact) if fact.kind == "harvest" and claim.affirmed else account["assertion"]
    source = account["immediate_source_name"]
    told = "I saw it myself." if source == npc.name else f"{source} told me about it."
    return f"Ask for an explanation of what I hold against them: {assertion} {told}"


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


def killing_chance(opinion):
    from constants import CONFRONTATION_OPINION, KILLING_CHANCE_PER_POINT
    return max(0.0, min(1.0, (CONFRONTATION_OPINION + 1 - opinion) * KILLING_CHANCE_PER_POINT))


def roll_escalation(conversation, response, opinion):
    """The confronter's leave or attack, rolled by the game rather than chosen by the model.

    opinion is the view the exchange commits. Each point it falls past CONFRONTATION_OPINION adds
    KILLING_CHANCE_PER_POINT; an accepted promise from the accused to make it right rules an attack out.
    The accused's last answer is the evidence the attack rests on."""
    decision = response.escalation
    choices = conversation.initiator.world.escalation_choices(conversation, opinion=opinion)
    if decision is None or not choices:
        return decision
    accused = conversation.recipient.id
    promised = any(p.id == a.proposal_id and p.promisor_id == accused
                   for a in response.accepted_proposals for p in conversation.proposals)
    answer = next((t.text for t in reversed(response.turns) if t.speaker == "recipient"), "")[:400]
    chance = killing_chance(opinion)
    attack = ("attack" in choices["actions"] and not promised and bool(answer.strip())
              and conversation.initiator.rng.random() < chance)
    return decision.model_copy(update={"action": "attack", "evidence": answer} if attack else {"action": "leave"})


def validate_escalation(conversation, response, opinion=None):
    """opinion: the initiator's resulting view of the target as it will be committed; defaults to the response's own."""
    validate_escalation_snapshot(response.escalation,
        {"purpose": conversation.purpose, "escalation_choices": conversation.initiator.world.escalation_choices(conversation)},
        response.turns, response.initiator_opinion.opinion if opinion is None else opinion)
