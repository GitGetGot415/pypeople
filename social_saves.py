"""Validation of persisted social records."""

from constants import CONVERSATION_MAX_AGREEMENTS
from dialogue_rules import validate_apple_promise
from social_records import phase_index, owns_recollection


def validate_player_promise_offers(record, promises, owner_ids, player_id, now, player_offer_ids):
    if len(record.player_promise_offers) > CONVERSATION_MAX_AGREEMENTS:
        raise ValueError("Too many player apple offers in one conversation.")
    for offer in record.player_promise_offers:
        if (offer.id in player_offer_ids
                or offer.giver_id not in owner_ids or offer.recipient_id not in owner_ids
                or offer.giver_id == offer.recipient_id
                or player_id not in (offer.giver_id, offer.recipient_id)
                or offer.status not in ("offered", "accepted", "declined", "expired", "closed")
                or not 0 <= offer.created_phase <= now or offer.deadline < offer.created_phase
                or not 0 <= offer.offer_turn < len(record.turns)
                or record.turns[offer.offer_turn].speaker_id != offer.giver_id):
            raise ValueError("Invalid player apple offer authority or source.")
        player_offer_ids.add(offer.id)
        from dialogue_rules import validate_apple_offer
        validate_apple_offer(record.turns[offer.offer_turn].text, offer.deadline, current_phase=offer.created_phase)
        if offer.status == "accepted":
            turn = offer.accept_turn
            if (type(turn) is not int or not offer.offer_turn < turn < len(record.turns)
                    or record.turns[turn].speaker_id != offer.recipient_id
                    or not any(p.id == offer.id and p.conversation_id == record.id
                               and p.promisor_id == offer.giver_id and p.recipient_id == offer.recipient_id
                               and p.deadline == offer.deadline for p in promises)):
                raise ValueError("An accepted player apple promise needs its actual ordered acceptance.")
            validate_apple_promise(record.turns[offer.offer_turn].text, record.turns[turn].text,
                                   offer.deadline, current_phase=offer.created_phase)
        elif offer.accept_turn is not None:
            raise ValueError("An unaccepted apple offer cannot retain an acceptance.")


def validate_saved_concern(concern, owner, claims, facts):
    if concern is None:
        return
    if concern.claim_id:
        claim = claims.get(concern.claim_id)
        belief = next((b for b in owner.beliefs if b.claim_id == concern.claim_id), None)
        if claim is None or belief is None:
            raise ValueError("A concern requires its owner's existing knowledge.")
        fact = facts[claim.fact_id]
        if concern.kind == "recollection" and not owns_recollection(owner.id, belief, claim, fact):
            raise ValueError("A recollection concern requires the owner's direct former-life account.")


def validate_social_records(ctx):
    state = ctx.state
    facts, claims, disclosures, events, history = ctx.facts, ctx.claims, ctx.disclosures, ctx.events, ctx.history
    residents, snake, ids, live_ids, day, now = ctx.residents, state.snake, ctx.ids, ctx.live_ids, state.day, ctx.now
    from conversation_history import source_record
    from constants import PLAYER_ID
    observed_conversations = set()
    for fact in facts.values():
        single_actor = fact.kind in ("arrival", "departure", "apple_eaten", "harvest")
        if (fact.subject_id not in ids or fact.object_id not in ids
                or (fact.subject_id == fact.object_id) != single_actor
                or fact.phase > now or not set(fact.witness_ids) <= ids
                or PLAYER_ID in fact.witness_ids
                or len(set(fact.witness_ids)) != len(fact.witness_ids)
                or not fact.witness_ids and fact.kind not in ("conversation_seen", "harvest")):
            raise ValueError("Invalid fact participants or witnesses.")
        if fact.kind == "conversation_seen":
            record = source_record(history, fact.conversation_id)
            if (not any(t.speaker_id != PLAYER_ID for t in record.turns)
                    or fact.conversation_id in observed_conversations
                    or set(record.participant_ids) != {fact.subject_id, fact.object_id}
                    or set(fact.witness_ids) & set(record.participant_ids)):
                raise ValueError("A conversation observation requires accepted speech and actual bystanders.")
            observed_conversations.add(fact.conversation_id)
        if fact.kind in ("private_motive", "former_life"):
            if fact.witness_ids != [fact.subject_id] or fact.conversation_id is not None:
                raise ValueError("A private recollection or motive cannot be public.")
        elif fact.conversation_id is not None:
            source_record(history, fact.conversation_id)
        if fact.source_id in events:
            event = events[fact.source_id]
            expected = {"apple_given": "fruit_shared", "work_given": "work_shared", "apple_refused": "food_refused", "promise_kept": "promise_fulfilled"}.get(fact.kind, fact.kind)
            if (event.kind != expected or fact.phase != phase_index(event.day, event.phase)
                    or set(event.participants) != {fact.subject_id, fact.object_id}
                    or event.history != fact.conversation_id
                    or fact.kind == "harvest" and event.text != fact.summary):
                raise ValueError("A fact must match its actual source event.")
    for claim in claims.values():
        fact = facts.get(claim.fact_id)
        if fact is None or claim.subject_id != fact.subject_id or claim.originator_id not in ids or not fact.phase <= claim.phase <= now:
            raise ValueError("Invalid claim references.")
        if claim.parent_claim_id is None:
            if not claim.affirmed or claim.deliberate or claim.originator_id not in fact.witness_ids:
                raise ValueError("Firsthand accounts require an actual witnessed source.")
        else:
            parent = claims.get(claim.parent_claim_id)
            if parent is None or parent.fact_id != fact.id or not parent.affirmed or claim.affirmed or not claim.deliberate or parent.phase > claim.phase:
                raise ValueError("A denial must trace to an existing account.")
    seen = set()
    for disclosure in disclosures.values():
        record = source_record(history, disclosure.conversation_id)
        if (disclosure.claim_id not in claims or disclosure.speaker_id == disclosure.listener_id
                or not {disclosure.speaker_id, disclosure.listener_id} <= ids
                or {disclosure.speaker_id, disclosure.listener_id} != set(record.participant_ids)
                or disclosure.turn >= len(record.turns) or record.turns[disclosure.turn].speaker_id != disclosure.speaker_id
                or disclosure.phase > now):
            raise ValueError("Invalid disclosure actor, speech or account.")
        if disclosure.source_disclosure_id:
            source = disclosures.get(disclosure.source_disclosure_id)
            claim = claims[disclosure.claim_id]
            if source is None or source.id not in seen or source.listener_id != disclosure.speaker_id or source.claim_id not in (claim.id, claim.parent_claim_id):
                raise ValueError("Disclosure provenance must follow an earlier source.")
        elif disclosure.speaker_id != claims[disclosure.claim_id].originator_id:
            fact = facts[claims[disclosure.claim_id].fact_id]
            if disclosure.speaker_id not in fact.witness_ids:
                raise ValueError("A repeated account requires its actual source.")
        seen.add(disclosure.id)
    for actor in [*residents, snake]:
        validate_saved_concern(actor.current_concern, actor, claims, facts)
        if any(m.partner_id not in ids or phase_index(m.day, m.phase) > now for m in actor.memories):
            raise ValueError("Invalid personal memory ownership or phase.")
        for effect in actor.pending_social_opinions:
            if effect.target_id not in ids or effect.claim_id is not None and effect.claim_id not in claims:
                raise ValueError("Invalid queued social effect.")
        if len({b.claim_id for b in actor.beliefs}) != len(actor.beliefs):
            raise ValueError("Duplicate owned account.")
        for belief in actor.beliefs:
            claim = claims.get(belief.claim_id)
            if claim is None or belief.npc_id != actor.id or belief.source_id not in ids or belief.phase > now:
                raise ValueError("Invalid owned knowledge.")
            if belief.direct:
                if actor.id not in facts[claim.fact_id].witness_ids or belief.disclosure_id is not None:
                    raise ValueError("Firsthand knowledge requires witnessing.")
            elif belief.disclosure_id:
                source = disclosures.get(belief.disclosure_id)
                if source is None or source.listener_id != actor.id or source.claim_id != claim.id or source.speaker_id != belief.source_id:
                    raise ValueError("Hearsay requires its actual disclosure source.")
    for record in history.values():
        if record.purpose == "confrontation":
            grievance = claims.get(record.grievance_id)
            if (not grievance or not grievance.affirmed or grievance.subject_id != record.participant_ids[1]
                    or not set(record.participant_ids) <= {n.id for n in residents}):
                raise ValueError("A confrontation history needs two human residents and its original sourced grievance.")
            if record.turns and not any(d.conversation_id == record.id and d.speaker_id == record.participant_ids[0]
                    and claims[d.claim_id].fact_id == grievance.fact_id for d in disclosures.values()):
                raise ValueError("An accepted confrontation must retain its actual grievance discussion and date.")
        elif record.grievance_id is not None:
            raise ValueError("Only confrontation history owns a grievance.")
    goal_ids = set()
    for npc in residents:
        if npc.id in live_ids and npc.current_concern is None:
            raise ValueError("A living resident must have a concern.")
        if npc.arrival_day > day:
            raise ValueError("Invalid arrival day or personal memory.")
        if npc.departure and (phase_index(npc.departure.day, npc.departure.phase) > now or not any(f.source_id == npc.departure.event_id and f.kind == "killing" and f.object_id == npc.id for f in facts.values())):
            raise ValueError("An archived resident needs its actual departure record.")
        goal = npc.goal
        if goal:
            if goal.id in goal_ids or goal.target_id not in ids or goal.third_npc_id is not None and goal.third_npc_id not in ids or goal.created_phase > now or goal.deadline < goal.created_phase:
                raise ValueError("Invalid goal references.")
            goal_ids.add(goal.id)
            if goal.status == "active" and (npc.id not in live_ids or goal.target_id not in live_ids or goal.third_npc_id is not None and goal.third_npc_id not in live_ids):
                raise ValueError("An active goal can target only living residents.")
            if (goal.status == "active") != (goal.resolved_phase is None):
                raise ValueError("Invalid goal resolution.")
            if goal.claim_id is not None and goal.claim_id not in claims:
                raise ValueError("Unknown goal account.")


def validate_saved_promises(ctx):
    promises, history, ids, live_ids, player_id, now = ctx.state.promises, ctx.history, ctx.ids, ctx.live_ids, ctx.state.player.id, ctx.now
    from conversation_history import source_record
    promise_ids = set()
    for promise in promises:
        record = source_record(history, promise.conversation_id)
        if promise.id in promise_ids or promise.promisor_id == promise.recipient_id or not all(i in ids and i in record.participant_ids for i in (promise.promisor_id, promise.recipient_id)):
            raise ValueError("Invalid promise participants.")
        promise_ids.add(promise.id)
        if promise.status == "pending" and not {promise.promisor_id, promise.recipient_id} <= live_ids | {player_id}:
            raise ValueError("Impossible obligations must be cancelled.")
        if promise.deadline < promise.created_phase or promise.created_phase > now or (promise.status == "pending") != (promise.resolved_phase is None):
            raise ValueError("Invalid promise resolution.")
