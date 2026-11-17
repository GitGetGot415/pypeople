from dialogue.history import remember
from simulation.social.records import OpinionEffect, Claim, Belief, Disclosure, fact_by_id, claim_by_id
from simulation.social.information import (
    related_beliefs, demote_contrary_beliefs, receive_belief, harvest_summary, retold_accusation,
    reputation_deltas,
)
from simulation.social.relationships import apply_opinion_effects


def commit_information(conversation):
    context = conversation.information_context
    if context is None:
        return
    world = conversation.initiator.world
    options = {option.id: option for option in context.options}
    people = {"initiator": conversation.initiator, "recipient": conversation.recipient}
    heard, told_as, alleged = {}, {}, []
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
            # Quote the account as it was told, so a harvest keeps the teller's total rather than one pickup.
            reason = (f"Heard {source.name}'s account (original source: {originator.name}): "
                      f"{told_as.get((speaker.id, entry.claim_id), option.assertion)} "
                      f"Personal assessment: {entry.reason}")[:400]
            event = world.publish("claim_assessed", (source, speaker),
                                  f"{speaker.name} assessed an allegation about {world.participant_by_id(entry.subject_id).name} from {source.name}.",
                                  conversation.record)
            belief.assessed = True
            belief.opinion_delta += entry.delta
            speaker.pending_social_opinions.append(OpinionEffect(event.id, entry.subject_id, entry.delta,
                                                                  reason, claim.id, source.id, conversation.record.id, world.day))
        for entry in [entry for entry in conversation.accepted_disclosures if entry.turn == turn]:
            option = options[entry.claim_id]
            source_belief = next((b for b in speaker.beliefs if b.claim_id == option.id), None)
            if not any(claim.id == option.id for claim in world.claims):
                parent = next(b for b in speaker.beliefs if b.claim_id == option.parent_claim_id)
                source_belief = parent
                world.claims.append(Claim(option.id, option.fact_id, option.subject_id, speaker.id,
                                          option.affirmed, option.parent_claim_id, True,
                                          world.day, conversation.record.id))
                claims[option.id] = world.claims[-1]
                speaker.beliefs.append(Belief(speaker.id, option.id, "disbelieved", speaker.id, None,
                                              False, world.day))
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
                                    source_belief.disclosure_id, world.day, event.id)
            world.disclosures.append(disclosure)
            prior = next((b for b in listener.beliefs if b.claim_id == option.id), None)
            witnessed = any(b.direct and claims[b.claim_id].fact_id == option.fact_id for b in listener.beliefs if b.claim_id in claims)
            if entry.intent == "share":
                alleged.append((turn, listener, option))
            if entry.listener_stance == "believed":
                demote_contrary_beliefs(option, related_beliefs(option, listener.beliefs, claims), claims, listener.id)
            learned = receive_belief(prior, listener_id=listener.id, claim_id=option.id,
                stance=entry.listener_stance, source_id=speaker.id, disclosure_id=disclosure.id, day=world.day)
            if prior is None:
                listener.beliefs.append(learned)
            for claim in pickups:
                # Each pickup the telling covers is its own disclosed account, so hearsay keeps an exact source.
                told = Disclosure(f"{disclosure.id}:{claim.id}", claim.id, conversation.record.id, speaker.id, listener.id,
                                  turn, entry.intent, entry.listener_stance, None, world.day, event.id)
                world.disclosures.append(told)
                prior_pickup = next((b for b in listener.beliefs if b.claim_id == claim.id), None)
                learned_pickup = receive_belief(prior_pickup, listener_id=listener.id, claim_id=claim.id, stance=entry.listener_stance,
                                                source_id=speaker.id, disclosure_id=told.id, day=world.day)
                if prior_pickup is None:
                    listener.beliefs.append(learned_pickup)
            heard[listener.id, option.id] = speaker
            told_as[listener.id, option.id] = assertion
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
    if retold_accusation(option):
        # A retold accusation worsens its subject unless that has run its course (see reputation_assessment_available).
        why = (f"{listener.name} had already judged this same account in an earlier conversation."
               if any(b.assessed and b.claim_id == option.id for b in related) else
               f"Retelling had already stirred {listener.name} up against {subject.name} as far as it can over this."
               if not any(delta < 0 for delta in reputation_deltas(related, option)) else
               f"{listener.name} formed no new view of {subject.name} from it.")
    elif any(b.assessed for b in related):
        why = f"{listener.name} had already judged {subject.name} over this in an earlier conversation."
    elif any(b.direct for b in related) and option.kind != "harvest":
        why = f"{listener.name} saw this firsthand, so hearing it retold changed nothing."
    else:
        why = f"{listener.name} formed no new view of {subject.name} from it."
    conversation.record.outcomes.append(
        f"{listener.name}'s opinion of {subject.name} changed +0 after assessing an allegation. {why}")


def _witnessed_pickups(world, speaker, option, claims):
    """The speaker's claims about the same collector's other pickups that day, which a firsthand telling includes."""
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
        if other.kind == "harvest" and other.subject_id == fact.subject_id and other.day == fact.day:
            pickups.setdefault(other.id, claim)
    return list(pickups.values())
