"""Facts, claims, beliefs and disclosures: what residents know and how accounts are told, heard and committed."""

import re
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4
from constants import (SNAKE_NAME, CONVERSATION_MAX_AGREEMENTS, LEVERAGE_ACCOUNT_KINDS, PHASES,
    CONVERSATION_MAX_DISCLOSURES, CONVERSATION_MAX_REPUTATION_EFFECTS, CONVERSATION_MAX_LEVERAGE,
    REPUTATION_FACT_LIMIT, OPINION_CHANGES, ROUTINE_FACT_KINDS)
from conversation_history import remember
from dialogue_text import split_sentences
from dialogue_rules import (unquoted_speech, mentions_account, named_references, validate_account_speech, harvest_extent)
from types import SimpleNamespace
from actor_identity import account_references, actor_id, speech_only
from social_records import (phase_label, Fact, Claim, Belief, Disclosure, ClaimOption, InformationContext, fact_by_id,
    claim_by_id)


def assertion_text(world, fact, affirmed=True):
    if affirmed:
        summary = harvest_summary(world.participant_by_id(fact.subject_id).name, fact.quantity) if fact.kind == "harvest" else fact.summary
        return f"At {phase_label(fact.phase)}: {summary}"
    subject, target = world.participant_by_id(fact.subject_id), world.participant_by_id(fact.object_id)
    action = {"apple_given": "give an apple to", "apple_refused": "refuse an apple to",
              "promise_kept": "keep the apple promise to", "promise_broken": "miss the apple promise to"}
    if fact.kind == "private_motive":
        return f"{subject.name} does not have the motive described in this account: {fact.summary}"
    if fact.kind not in action:
        return f"An account disputes: {fact.summary}"
    return f"{subject.name} did not {action[fact.kind]} {target.name} in the exchange at {phase_label(fact.phase)}."


def account_content(world, fact, affirmed=True):
    """Request prose has no record header; keep UI formatting in assertion_text()."""
    summary = (harvest_summary(world.participant_by_id(fact.subject_id).name, fact.quantity)
               if fact.kind == "harvest" else fact.summary if fact.kind == "former_life" else named_references(fact.summary))
    if fact.kind == "former_life":
        _, marker, description = summary.partition("'s own description: ")
        summary = description if marker else summary
    elif fact.kind == "private_motive":
        # The heading describes the goal; the owed private information is why
        # its owner wants it. Ownership stays in the separate account fields.
        _, marker, motive = summary.partition(": ")
        summary = motive if marker else summary
    if affirmed:
        return summary
    subject, target = world.participant_by_id(fact.subject_id), world.participant_by_id(fact.object_id)
    action = {"apple_given": "give an apple to", "apple_refused": "refuse an apple to",
              "promise_kept": "keep the apple promise to", "promise_broken": "miss the apple promise to"}
    if fact.kind == "private_motive":
        return f"{subject.name} does not have the motive described in this account: {summary}"
    if fact.kind not in action:
        return f"An account disputes: {summary}"
    return f"{subject.name} did not {action[fact.kind]} {target.name} in that exchange."


def fact_context(world, fact, affirmed=True):
    date = {"day": fact.phase // len(PHASES) + 1, "phase": PHASES[fact.phase % len(PHASES)]}
    return {"assertion": account_content(world, fact, affirmed), "kind": fact.kind,
            "recorded_at": date, "event_time": None if fact.kind == "former_life" else date,
            "time_scope": "remembered former life; event date unknown" if fact.kind == "former_life" else "life here",
            "quantity": fact.quantity, "location": fact.location}


def account_view(world, claim, belief):
    fact = fact_by_id(world, claim.fact_id)
    catalog = {actor.id: {"name": actor.name} for actor in [*world.actors, world.player]}
    account = {**fact_context(world, fact, claim.affirmed), "claim_id": claim.id,
               "subject_id": claim.subject_id, "object_id": fact.object_id,
               "original_speaker_id": claim.originator_id}
    projected = build_account_view(account, account_knowledge(belief, belief.npc_id), catalog, belief.npc_id, None)
    context = fact_context(world, fact, claim.affirmed)
    if fact.kind == "harvest":
        harvests = witnessed_harvests(world, world.participant_by_id(belief.npc_id).beliefs)
        context["harvest_context"] = [row for (phase, _), row in harvests.items() if phase == fact.phase]
    return harvest_account_view({**context,
            **{key: projected[key] for key in ("claim_id", "subject_id", "subject_name", "object_id", "object_name",
                "knowledge_owner_name", "original_speaker_id", "immediate_source_id", "original_speaker_name", "immediate_source_name")},
            "source_disclosure_id": belief.disclosure_id, "stance": projected["stance"],
            "learned_at": {"day": belief.phase // len(PHASES) + 1, "phase": PHASES[belief.phase % len(PHASES)]},
            "basis": projected["basis"]}, tree_stock=world.tree.fruit,
            current_time={"day": world.day, "phase": world.phase})


def harvest_summary(name, quantity):
    return f"{name} harvested {quantity} {'apple' if quantity == 1 else 'apples'} from the tree."


def witnessed_harvests(world, beliefs):
    """Derive totals from this observer's distinct witnessed events, within each phase."""
    claims = {claim.id: claim for claim in world.claims}
    facts = {fact.id: fact for fact in world.facts}
    totals, seen = {}, set()
    for belief in beliefs:
        claim = claims[belief.claim_id]
        fact = facts[claim.fact_id]
        if not belief.direct or not claim.affirmed or fact.kind != "harvest" or fact.id in seen:
            continue
        seen.add(fact.id)
        key = (fact.phase, fact.subject_id)
        row = totals.setdefault(key, dict(subject_id=fact.subject_id,
            subject_name=world.participant_by_id(fact.subject_id).name,
            event_time={"day": fact.phase // len(PHASES) + 1, "phase": PHASES[fact.phase % len(PHASES)]},
            quantity=0, pickups=0))
        row["quantity"] += fact.quantity
        row["pickups"] += 1
    unseen_phases = {fact.phase for fact in world.facts if fact.kind == "harvest" and fact.id not in seen}
    for (phase, _), row in totals.items():
        row["all_harvests_witnessed"] = phase not in unseen_phases
    return totals


def harvest_account_view(account, *, tree_stock=None, current_time=None):
    """Present the witnessed collection consistently; durable facts remain individual pickups."""
    if account.get("kind") != "harvest" or account.get("basis") != "firsthand" or account.get("stance") != "believed":
        return account
    total = next((row for row in account.get("harvest_context", [])
                  if row["subject_id"] == account["subject_id"]), None)
    if total is None:
        return account
    account = {**account, "assertion": harvest_summary(account["subject_name"], total["quantity"]),
               "quantity": total["quantity"], "pickup_count": total["pickups"],
               "quantity_scope": "total witnessed for this collector during event_time"}
    extent = harvest_extent(account, tree_stock, current_time) if current_time is not None else account.get("harvest_extent")
    if current_time is not None:
        account.pop("harvest_extent", None)
    if extent:
        account["harvest_extent"] = extent
        if extent["share"] == "all":
            account["assertion"] += " This was the entire phase's tree stock; no apples remain on the tree."
    if account.get("firsthand_stance") == "believed":
        account["firsthand_evidence"] = account["assertion"]
    return account


def add_fact(world, kind, subject, target, source_id, record, summary, witnesses, *, quantity=None, location=None):
    # Callers own audience selection: direct participants can speak through a door,
    # and an observer may have left before accepted speech establishes the exchange.
    if (kind == "harvest") != (type(quantity) is int and quantity > 0) or kind != "harvest" and quantity is not None:
        raise ValueError("Only harvesting facts require an exact positive integer quantity.")
    if kind == "harvest":
        summary = harvest_summary(subject.name, quantity)
    witnesses = list({p.id: p for p in witnesses if p is not None and p is not world.player}.values())
    fact = next((fact for fact in world.facts if fact.source_id == source_id and fact.kind == kind), None)
    if fact is None:
        fact = Fact(uuid4().hex, kind, subject.id, target.id if target else subject.id, source_id,
                    record.id if record else None, world.phase_index, summary, [], quantity, location)
        world.facts.append(fact)
    for npc in witnesses:
        if npc.id in fact.witness_ids:
            continue
        fact.witness_ids.append(npc.id)
        claim = Claim(uuid4().hex, fact.id, subject.id, npc.id, True, None, False,
                      world.phase_index, record.id if record else None)
        world.claims.append(claim)
        # Promise consequences already have their own opinion adjustment.
        assessed = npc is target and kind in ("promise_kept", "promise_broken")
        delta = (4 if kind == "promise_kept" else -6) if assessed else 0
        npc.beliefs.append(Belief(npc.id, claim.id, "believed", npc.id, None, True,
                                 world.phase_index, assessed, delta))
    return fact


def record_intention(npc, world):
    goal = npc.goal
    if goal:
        add_fact(world, "private_motive", npc, world.participant_by_id(goal.target_id), goal.id, None,
                 f"{npc.name}'s motive for {goal.objective}: {goal.motive}", [npc])


def record_event_fact(world, event, record):
    """Private consequences only; physical commits explicitly record visual facts."""
    kinds = {"food_refused": "apple_refused", "promise_fulfilled": "promise_kept", "promise_broken": "promise_broken"}
    kind = kinds.get(event.kind)
    if kind is None:
        return
    first, second = [world.participant_by_id(npc_id) for npc_id in event.participants]
    subject, target = (second, first) if kind == "apple_refused" else (first, second)
    add_fact(world, kind, subject, target, event.id, record, event.text, [subject, target])


def can_deceive(npc):
    return npc is not npc.world.snake and (npc.preferences["generosity"] < 0.5 or bool(npc.goal and npc.goal.kind == "sabotage_relationship"
                                                     and npc.goal.status == "active"))


def selected_beliefs(npc, partner_id=None, *, pinned=()):
    world = npc.world
    goal = npc.goal
    order = {id(b): i for i, b in enumerate(npc.beliefs)}
    def rank(belief):
        claim = claim_by_id(world, belief.claim_id)
        fact = fact_by_id(world, claim.fact_id)
        return (claim.id in pinned or bool(npc.current_concern and npc.current_concern.claim_id == claim.id),
                bool(goal and goal.claim_id == claim.id),
                fact.kind in ("killing",),
                fact.kind in ("killing", "snake_claim", "harvest"),
                fact.kind not in ROUTINE_FACT_KINDS,
                partner_id in (fact.subject_id, fact.object_id, belief.source_id),
                bool(goal and fact.subject_id == goal.third_npc_id),
                fact.kind == "promise_broken" or fact.kind == "private_motive" and fact.subject_id != npc.id,
                fact.kind != "private_motive" or fact.subject_id != npc.id, belief.phase, order[id(belief)])
    selected = sorted(npc.beliefs, key=rank, reverse=True)[:6]
    if goal:
        motive = next((b for b in npc.beliefs if fact_by_id(world, claim_by_id(world, b.claim_id).fact_id).source_id == goal.id), None)
        if motive and motive not in selected:
            selected = [*selected[:5], motive]
    return selected


def prepare_information(conversation):
    if conversation.purpose == "food_request":
        return None
    pair = {"initiator": conversation.initiator, "recipient": conversation.recipient}
    world = conversation.initiator.world
    options, allowed = {}, {}
    for role, npc in pair.items():
        concern = conversation.concerns[role]
        selected = selected_beliefs(npc, conversation.other(npc).id,
                                   pinned=(concern.claim_id,) if concern and concern.claim_id else ())
        if role == "initiator" and conversation.focus_claim_id:
            focus = next(b for b in npc.beliefs if b.claim_id == conversation.focus_claim_id)
            selected = [focus, *[b for b in selected if b is not focus]]
        allowed[role] = [belief.claim_id for belief in selected]
        for belief in npc.beliefs:
            claim = claim_by_id(world, belief.claim_id)
            fact = fact_by_id(world, claim.fact_id)
            options[claim.id] = ClaimOption(claim.id, fact.id, fact.subject_id, claim.originator_id,
                                           claim.affirmed, claim.parent_claim_id, None, fact.kind,
                                           account_content(world, fact, claim.affirmed), quantity=fact.quantity)
        if can_deceive(npc):
            bases = [claim_by_id(world, belief.claim_id) for belief in selected
                     if belief.stance == "believed" and claim_by_id(world, belief.claim_id).affirmed]
            for index, base in enumerate(bases[:2]):
                fact = fact_by_id(world, base.fact_id)
                option_id = f"{conversation.id}:deny_{role}_{index}"
                options[option_id] = ClaimOption(option_id, fact.id, fact.subject_id, npc.id, False,
                                                base.id, npc.id, fact.kind, account_content(world, fact, False), quantity=fact.quantity)
                allowed[role].append(option_id)
        if npc.goal and npc.goal.kind == "sabotage_relationship" and npc.goal.status == "active":
            allowed[role].sort(key=lambda claim_id: options[claim_id].creator_id is None)
    relevant_facts = {options[claim_id].fact_id for entries in allowed.values() for claim_id in entries}
    for role, npc in pair.items():
        for belief in npc.beliefs:
            if options[belief.claim_id].fact_id in relevant_facts and belief.claim_id not in allowed[role]:
                allowed[role].append(belief.claim_id)
    return InformationContext(
        {role: npc.id for role, npc in pair.items()},
        {role: deepcopy(npc.beliefs) for role, npc in pair.items()}, list(options.values()),
        {role: can_deceive(npc) for role, npc in pair.items()},
        {role: {other.id: npc.opinion_score(other.id) for other in world.actors if other is not npc}
         for role, npc in pair.items()}, allowed, conversation.focus_claim_id)


def information_views(context, world):
    """Only speaker-owned knowledge reaches prompts; engine truth stays out of them."""
    if context is None:
        return {}
    catalog = {actor.id: {"name": actor.name} for actor in [*world.actors, world.player]}
    options = {option.id: option for option in context.options}
    result = {}
    for role, npc_id in context.participants.items():
        beliefs = {belief.claim_id: belief for belief in context.beliefs[role]}
        harvests = witnessed_harvests(world, beliefs.values())
        firsthand = {options[b.claim_id].fact_id: options[b.claim_id]
                     for b in beliefs.values() if b.direct}
        views = []
        for claim_id in context.allowed[role]:
            option = options[claim_id]
            belief = beliefs.get(claim_id)
            fact = fact_by_id(world, option.fact_id)
            selected = {**fact_context(world, fact, option.affirmed),
                        "claim_id": claim_id, "subject_id": option.subject_id,
                        "object_id": fact.object_id, "original_speaker_id": option.originator_id}
            projected = build_account_view(selected, account_knowledge(belief, npc_id) if belief else None,
                                           catalog, npc_id, context.participants["recipient" if role == "initiator" else "initiator"])
            view = {**fact_context(world, fact, option.affirmed),
                    **{key: projected[key] for key in ("claim_id", "subject_id", "subject_name", "object_id", "object_name",
                        "knowledge_owner_name", "original_speaker_id", "original_speaker_name")},
                    "permitted_use": "deliberate denial; private to this speaker" if option.creator_id else "known account"}
            if fact.kind == "harvest":
                view["harvest_context"] = [row for (phase, _), row in harvests.items() if phase == fact.phase]
            if belief:
                view.update(stance=projected["stance"], immediate_source_id=projected["immediate_source_id"],
                            immediate_source_name=projected["immediate_source_name"],
                            learned_at={"day": belief.phase // len(PHASES) + 1, "phase": PHASES[belief.phase % len(PHASES)]},
                            source_disclosure_id=belief.disclosure_id,
                            basis=projected["basis"],
                            already_assessed=any(b.assessed and options[b.claim_id].fact_id == option.fact_id
                                                 and options[b.claim_id].affirmed == option.affirmed
                                                 for b in context.beliefs[role]),
                            current_reputation_influence=sum(b.opinion_delta for b in context.beliefs[role]
                                                             if options[b.claim_id].fact_id == option.fact_id))
            evidence = firsthand.get(option.fact_id)
            if evidence:
                view["firsthand_evidence"] = evidence.assertion
                view["firsthand_stance"] = "believed" if evidence.affirmed == option.affirmed else "disbelieved"
                if belief and belief.direct:
                    view["permitted_use"] = ("report the named collector's actions you witnessed; witnessing is not collecting"
                                             if fact.kind == "harvest" else "share personal experience; its truth needs no confirmation")
            views.append(harvest_account_view(view, tree_stock=world.tree.fruit,
                         current_time={"day": world.day, "phase": world.phase}))
        result[role] = views
    return result


def related_beliefs(option, beliefs, options):
    return [b for b in beliefs if options[b.claim_id].fact_id == option.fact_id]


def reputation_assessment_available(option, related, options, *, believed, heard, participants, assessed=False):
    """Stance is actual during validation, possible before a pending reaction."""
    # A witness knows who picked the fruit, but may reconsider the collector's
    # generosity when someone frames that conduct. Assessment remains once per fact.
    return (option.kind != "conversation_seen" and heard and believed and option.subject_id not in participants and not assessed
            and not any(b.direct and option.kind != "harvest"
                        or b.assessed and options[b.claim_id].affirmed == option.affirmed
                        for b in related))


def reputation_deltas(related):
    influence = sum(b.opinion_delta for b in related)
    return [delta for delta in range(-REPUTATION_FACT_LIMIT, REPUTATION_FACT_LIMIT + 1)
            if delta and abs(influence + delta) <= REPUTATION_FACT_LIMIT]


def reputation_labels(related):
    allowed = reputation_deltas(related)
    return [label for label, delta in OPINION_CHANGES.items() if delta in allowed]


def resident_effect_limits():
    return dict(offers=CONVERSATION_MAX_AGREEMENTS, accepted_proposals=CONVERSATION_MAX_AGREEMENTS,
                disclosures=CONVERSATION_MAX_DISCLOSURES,
                third_party_opinions=CONVERSATION_MAX_REPUTATION_EFFECTS, leverage=CONVERSATION_MAX_LEVERAGE)


def resident_effect_capacity(draft):
    return {("acceptances" if field == "accepted_proposals" else field): max(0, limit - len(draft[field]))
            for field, limit in resident_effect_limits().items()}


def demote_contrary_beliefs(option, related, options, listener_id):
    for belief in related:
        if (options[belief.claim_id].affirmed != option.affirmed and not belief.direct
                and belief.source_id != listener_id):
            belief.stance = "uncertain"


def receive_belief(prior, *, listener_id, claim_id, stance, source_id, disclosure_id, phase, update_source=True):
    """Only hearsay may change stance; exact repeats keep assessment bookkeeping."""
    if prior is None:
        return Belief(listener_id, claim_id, stance, source_id, disclosure_id, False, phase)
    if not prior.direct:
        prior.stance = stance
        if update_source:
            prior.source_id, prior.disclosure_id = source_id, disclosure_id
    return prior


def listener_stances(option, beliefs, options, *, firsthand_source=False):
    """The same belief constraints govern a pending reaction and the final record."""
    related = related_beliefs(option, beliefs.values(), options)
    firsthand = next((b for b in related if b.direct), None)
    prior = beliefs.get(option.id)
    if firsthand:
        return ["believed" if options[firsthand.claim_id].affirmed == option.affirmed else "disbelieved"]
    if prior is not None:
        return [prior.stance]
    stances = ["uncertain", "believed", "disbelieved"]
    if (not firsthand_source and any(options[b.claim_id].affirmed == option.affirmed
                                    and b.stance != "believed" for b in related)):
        stances.remove("believed")
    return stances


def validate_information(response, context, *, names=(), participant_names=None, dialogue_validation=True,
                         actor_references=None, incomplete=False):
    """Simulate knowledge in speech order before any real state is touched."""
    if isinstance(context, dict):
        context = information_from_dict(context)
    if context is None:
        if response.disclosures or response.third_party_opinions or response.leverage:
            raise ValueError("This exchange does not permit disclosures, gossip opinions, or leverage.")
        return
    options = {option.id: option for option in context.options}
    beliefs = {role: {b.claim_id: deepcopy(b) for b in entries} for role, entries in context.beliefs.items()}
    acquired = {role: set() for role in context.participants}
    heard = {}
    seen_disclosures, seen_opinions, seen_leverage = set(), set(), set()
    for entry in (*response.disclosures, *response.third_party_opinions, *response.leverage):
        if not 0 <= entry.turn < len(response.turns):
            raise ValueError("An information effect refers to a missing dialogue turn.")
    if len({entry.turn for entry in response.disclosures}) != len(response.disclosures):
        raise ValueError("Use at most one structured disclosure per spoken turn.")
    for turn, speech in enumerate(response.turns):
        role = speech.speaker
        listener = "recipient" if role == "initiator" else "initiator"
        npc_id = context.participants[role]
        for use in [entry for entry in response.leverage if entry.turn == turn]:
            option = options.get(use.claim_id)
            if (use.claim_id not in beliefs[role] or option is None
                    or use.claim_id not in context.allowed[role] and use.claim_id not in acquired[role]
                    or use.claim_id in seen_leverage
                    or option.kind not in LEVERAGE_ACCOUNT_KINDS):
                raise ValueError(f"Leverage on turn {turn} ({role}) requires a permitted claim already known before this turn. "
                                 "An account disclosed on this same turn is too late; omit leverage for ordinary requests.")
            seen_leverage.add(use.claim_id)
        for entry in [entry for entry in response.third_party_opinions if entry.turn == turn]:
            option = options.get(entry.claim_id)
            belief = beliefs[role].get(entry.claim_id)
            if (option is None or belief is None or (role, entry.claim_id) not in heard
                    or entry.subject_id != option.subject_id or entry.subject_id in context.participants.values()
                    or heard[role, entry.claim_id] >= turn):
                raise ValueError(f"A third-party opinion on turn {turn} ({role}) must cite an account this speaker heard "
                                 "in an earlier turn, with its actual absent subject and exact disclosed claim ID.")
            related = related_beliefs(option, beliefs[role].values(), options)
            if (not reputation_assessment_available(option, related, options,
                    believed=belief.stance == "believed", heard=True, participants=context.participants.values(),
                    assessed=(role, option.fact_id) in seen_opinions)
                    or entry.delta not in reputation_deltas(related)):
                raise ValueError("A repeated or disbelieved account cannot cause another reputation penalty.")
            seen_opinions.add((role, option.fact_id))
            belief.assessed, belief.opinion_delta = True, belief.opinion_delta + entry.delta
        for entry in [entry for entry in response.disclosures if entry.turn == turn]:
            option = options.get(entry.claim_id)
            if option is None or (role, entry.claim_id) in seen_disclosures:
                raise ValueError("Unknown or repeated disclosure.")
            belief = beliefs[role].get(entry.claim_id)
            if dialogue_validation:
                validate_account_speech(speech.text, option.assertion, names=names,
                                        speaker=(participant_names or {}).get(role),
                                        listener=(participant_names or {}).get(listener), kind=option.kind,
                                        firsthand=bool(belief and belief.direct), actor_references=actor_references,
                                        known_events=[{"assertion": options[b.claim_id].assertion, "kind": options[b.claim_id].kind}
                                                      for b in beliefs[role].values() if b.direct and b.claim_id in options])
            seen_disclosures.add((role, entry.claim_id))
            fabricating = option.creator_id is not None and belief is None
            if fabricating:
                if (option.creator_id != npc_id or entry.intent != "lie" or not context.can_lie[role]
                        or option.parent_claim_id not in beliefs[role]):
                    raise ValueError("This speaker cannot fabricate that assertion.")
                beliefs[role][option.id] = Belief(npc_id, option.id, "disbelieved", npc_id, None, False, 0)
            elif (belief is None or entry.claim_id not in context.allowed[role] and entry.claim_id not in acquired[role]):
                raise ValueError(f"The {role} on turn {turn} has not learned or been permitted to disclose that claim. "
                                 "Use that speaker's permitted_accounts or an account heard in an earlier turn.")
            elif ((entry.intent == "lie" and (not context.can_lie[role] or belief.stance == "believed"))
                  or (entry.intent == "share" and belief.stance == "disbelieved")):
                raise ValueError("A deliberate lie and an honestly repeated belief must remain distinct.")
            firsthand = next((b for b in beliefs[role].values()
                              if b.direct and options[b.claim_id].fact_id == option.fact_id), None)
            if firsthand:
                agrees = options[firsthand.claim_id].affirmed == option.affirmed
                if agrees and entry.intent == "challenge":
                    entry.intent = "share"  # Challenging an event the speaker witnessed is only a mislabel.
                if (not agrees and entry.intent == "share") or (agrees and entry.intent == "lie"):
                    raise ValueError("The disclosure intent must respect the speaker's firsthand knowledge of this event.")
            prior = beliefs[listener].get(option.id)
            related = related_beliefs(option, beliefs[listener].values(), options)
            if incomplete and entry.listener_stance is None:
                # The assertion has been spoken, but only its listener can choose
                # a stance. Do not authorize downstream effects until they do.
                continue
            stances = listener_stances(option, beliefs[listener], options,
                                       firsthand_source=bool(belief and belief.direct))
            if entry.listener_stance not in stances:
                raise ValueError("The listener's stance must respect firsthand evidence and prior beliefs; "
                                 "repeated hearsay is not independent confirmation. "
                                 f"Permitted stances: {', '.join(stances)}.")
            if entry.listener_stance == "believed":
                demote_contrary_beliefs(option, related, options, context.participants[listener])
            beliefs[listener][option.id] = receive_belief(prior,
                listener_id=context.participants[listener], claim_id=option.id, stance=entry.listener_stance,
                source_id=npc_id, disclosure_id=None, phase=0, update_source=False)
            acquired[listener].add(option.id)
            heard[listener, option.id] = turn
    if context.focus_claim_id and not incomplete:
        focus = options[context.focus_claim_id]
        if not any(response.turns[entry.turn].speaker == "initiator" and options[entry.claim_id].fact_id == focus.fact_id
                   for entry in response.disclosures):
            raise ValueError("A gossip exchange must disclose its chosen account or a permitted denial of it.")
    return beliefs, acquired, heard


def information_from_dict(value):
    if value is None:
        return None
    return InformationContext(
        value["participants"], {role: [Belief(**b) for b in entries] for role, entries in value["beliefs"].items()},
        [ClaimOption(**option) for option in value["options"]], value["can_lie"], value["opinions"], value["allowed"], value["focus_claim_id"])


def trim_information(world):
    """Bound unused accounts while retaining provenance, disputes, and reputation deduplication."""
    claims = {claim.id: claim for claim in world.claims}
    residents = world.residents
    current_goals = {npc.goal.id for npc in residents if npc.goal}
    for npc in residents:
        polarities = {}
        for belief in npc.beliefs:
            claim = claims[belief.claim_id]
            polarities.setdefault(claim.fact_id, set()).add(claim.affirmed)
        keep = {b.claim_id for b in npc.beliefs[-24:]}
        for b in npc.beliefs:
            claim = claims[b.claim_id]
            if b.direct or b.assessed or len(polarities[claim.fact_id]) > 1 or fact_by_id(world, claim.fact_id).source_id in current_goals:
                keep.add(b.claim_id)
        if npc.goal and npc.goal.claim_id:
            keep.add(npc.goal.claim_id)
        if npc.current_concern and npc.current_concern.claim_id:
            keep.add(npc.current_concern.claim_id)
        if npc.conversation:
            role = "initiator" if npc.conversation.initiator is npc else "recipient"
            concern = npc.conversation.concerns[role]
            if concern and concern.claim_id:
                keep.add(concern.claim_id)
            if npc.conversation.information_context:
                keep.update(b.claim_id for b in npc.conversation.information_context.beliefs[role])
        if npc.conversation and npc.conversation.initiator is npc and npc.conversation.focus_claim_id:
            keep.add(npc.conversation.focus_claim_id)
        npc.beliefs[:] = [b for b in npc.beliefs if b.claim_id in keep]
    claim_ids = {b.claim_id for npc in world.actors for b in npc.beliefs}
    claim_ids.update(r.grievance_id for r in world.history if r.grievance_id)
    claim_ids.update(effect.claim_id for npc in residents for effect in npc.pending_social_opinions if effect.claim_id)
    disclosure_ids = {b.disclosure_id for npc in world.actors for b in npc.beliefs if b.disclosure_id}
    disclosure_ids.update(d.id for d in world.disclosures for npc in residents
                          for effect in npc.pending_social_opinions if effect.claim_id == d.claim_id
                          and effect.conversation_id == d.conversation_id and d.listener_id == npc.id)
    disclosure_ids.update(d.id for d in world.disclosures[-100:])
    confrontations = {r.id for r in world.history if r.grievance_id and r.turns}
    disclosure_ids.update(d.id for d in world.disclosures if d.conversation_id in confrontations)
    event_ids = {event.id for event in world.events}
    disclosure_ids.update(d.id for d in world.disclosures if d.event_id in event_ids)
    fact_ids = {fact.id for fact in world.facts[-100:] if fact.source_id not in current_goals}
    active_records = {n.conversation.record.id for n in residents if n.conversation}
    active_records.update(n.gift.record.id for n in residents if n.gift)
    if world.snake.conversation:
        active_records.add(world.snake.conversation.record.id)
    if world.player_interaction and world.player_interaction.record_id:
        active_records.add(world.player_interaction.record_id)
    fact_ids.update(f.id for f in world.facts if f.kind == "conversation_seen" and f.conversation_id in active_records)
    fact_ids.update(fact.id for fact in world.facts if fact.source_id in current_goals or fact.kind == "killing")
    for npc in residents:
        if npc.conversation and npc.conversation.information_context:
            disclosure_ids.update(b.disclosure_id for entries in npc.conversation.information_context.beliefs.values()
                                  for b in entries if b.disclosure_id)
            for option in npc.conversation.information_context.options:
                fact_ids.add(option.fact_id)
                if option.id in claims:
                    claim_ids.add(option.id)
                if option.parent_claim_id:
                    claim_ids.add(option.parent_claim_id)
            disclosure_ids.update(d.id for d in world.disclosures if d.conversation_id == npc.conversation.id)
        if npc.conversation and npc.conversation.focus_claim_id:
            claim_ids.add(npc.conversation.focus_claim_id)
    by_id = {d.id: d for d in world.disclosures}
    pending = list(disclosure_ids)
    while pending:
        disclosure = by_id[pending.pop()]
        claim_ids.add(disclosure.claim_id)
        if disclosure.source_disclosure_id and disclosure.source_disclosure_id not in disclosure_ids:
            disclosure_ids.add(disclosure.source_disclosure_id)
            pending.append(disclosure.source_disclosure_id)
    claim_ids.update(claim.id for claim in world.claims[-100:])
    pending = list(claim_ids)
    while pending:
        claim = claims[pending.pop()]
        fact_ids.add(claim.fact_id)
        if claim.parent_claim_id and claim.parent_claim_id not in claim_ids:
            claim_ids.add(claim.parent_claim_id)
            pending.append(claim.parent_claim_id)
    world.disclosures[:] = [d for d in world.disclosures if d.id in disclosure_ids]
    world.claims[:] = [claim for claim in world.claims if claim.id in claim_ids]
    world.facts[:] = [fact for fact in world.facts if fact.id in fact_ids]


def learn_account(world, speaker, listener, claim_id, record, turn, stance, *, private=False):
    """Disclose an existing account only to its direct recipient, with its source chain."""
    source = next(b for b in speaker.beliefs if b.claim_id == claim_id)
    claim = claim_by_id(world, claim_id)
    fact = fact_by_id(world, claim.fact_id)
    event = world.publish("claim_shared", (speaker, listener),
        f"{speaker.name} disclosed {'private information' if private else 'a sourced account'} to {listener.name}.", record)
    disclosure = Disclosure(uuid4().hex, claim_id, record.id, speaker.id, listener.id, turn,
                            "share", stance, source.disclosure_id, world.phase_index, event.id)
    world.disclosures.append(disclosure)
    existing = next((b for b in listener.beliefs if b.claim_id == claim_id), None)
    learned = receive_belief(existing, listener_id=listener.id, claim_id=claim_id, stance=stance,
                             source_id=speaker.id, disclosure_id=disclosure.id, phase=world.phase_index)
    if existing is None:
        listener.beliefs.append(learned)
    if listener is not world.snake:
        remember(listener, speaker, event, "heard_claim",
                 f"{speaker.name}'s account, originally from {world.participant_by_id(claim.originator_id).name}: {account_content(world, fact, claim.affirmed)}", 0)
    return disclosure


NPC_ARRIVAL_MEMORY = (
    "You remember waking up one day in an elevator that was going up. "
    "When its doors opened, you found yourself on the first floor of the apartment building."
)


def orientation_speech(texts):
    pattern = (r"\b(?:arriv\w*|welcome|layout|hallway|rooms?|garden exit|bridge|elevator|where (?:are|am) (?:we|i)|where (?:we|i) (?:are|am)|"
               r"why (?:we|i) (?:are|am) here|how (?:did|do) (?:we|i) get|how (?:we|i) got here|"
               r"(?:get|getting|go|going|way|back) home|"
               r"(?:remember|recall).{0,35}(?:how|why|when).{0,25}here|"
               r"this place|where we are|surroundings|bearings|safety|shelter|"
               r"(?:find|finding|seek|seeking|need) (?:some )?help|safe here)\b")
    return any(re.search(pattern, sentence, re.I)
               for text in texts for sentence in split_sentences(unquoted_speech(text)))


def validate_orientation(response, snapshot):
    # A structured claim alone is insufficient. Derive the accepted record's
    # marker from actual speech, including when the model omitted the flag.
    discussed = orientation_speech(turn.text for turn in response.turns)
    if snapshot.get("arrival_focus"):
        if any((response.accepted_proposals, response.disclosures, response.third_party_opinions, response.leverage)):
            raise ValueError("This orientation exchange does not authorize apple promises, private disclosures or leverage. Welcome the person and discuss the known layout or practical routines.")
        if snapshot.get("_dialogue_validation", True):
            if not discussed:
                raise ValueError("This is an orientation exchange. Discuss the known surroundings, practical routines or remembered elevator arrival, rather than pursuing a routine apple goal.")
    # This is derived bookkeeping, not a reason to regenerate otherwise valid speech.
    response.orientation_discussed = discussed
    return discussed


def account_knowledge(belief, speaker_id, *, creator_id=None, fallback_source=None):
    """Project only this knower's evidence; newly heard reports own no private stance."""
    return dict(immediate_source_id=belief.source_id if belief else fallback_source,
                stance=belief.stance if belief else "pending your reaction",
                basis=("firsthand" if belief and belief.direct else
                       "deliberate denial; private to this speaker" if creator_id == speaker_id else
                       "own assertion" if belief and belief.source_id == speaker_id else "hearsay"))


def build_account_view(account, knowledge, actors, speaker_id, listener_id):
    """Calculate attribution once; callers explicitly select their request fields.

    Selection and factual lookup belong to social/interaction code. This function
    receives only the selected account and the speaker's knowledge, never world
    truth or the listener's private context. IDs determine ownership; names are
    presentation. A private motive's target is not evidence of an encounter.
    """
    view = deepcopy(account)
    if knowledge is not None:
        view.update(knowledge)
    for field in ("subject", "object", "original_speaker", "immediate_source"):
        target = view.get(field + "_id")
        if field + "_id" in view:
            view[field + "_name"] = actors.get(target, {}).get("name", view.get(field + "_name", ""))
        view[field + "_relation"] = ("speaker" if target == speaker_id else
                                    "listener" if target == listener_id else "another person")
    view["knowledge_owner_name"] = actors.get(speaker_id, {}).get("name", view.get("knowledge_owner_name", ""))
    if view.get("kind") == "private_motive":
        view.pop("object_relation", None)
    return view


def resident_information(snapshot, draft, *, incomplete=True):
    return validate_information(SimpleNamespace(**draft), snapshot.get("_information_validation"),
        names=(*[p["name"] for p in snapshot.get("_name_actors", {}).values()], SNAKE_NAME),
        participant_names={role: snapshot[role]["name"] for role in ("initiator", "recipient")},
        dialogue_validation=snapshot.get("_dialogue_validation", True),
        actor_references=account_references(snapshot.get("_name_actors", {})), incomplete=incomplete)


def pending_reactions(snapshot, draft, role):
    info = information_from_dict(snapshot["_information_validation"]) if snapshot.get("_information_validation") else None
    if info is None:
        return []
    beliefs, _, _ = resident_information(snapshot, draft)
    options = {option.id: option for option in info.options}
    pending = []
    for entry in draft["disclosures"]:
        source_role = draft["turns"][entry.turn].speaker
        if source_role == role or entry.listener_stance is not None:
            continue
        option = options[entry.claim_id]
        source = beliefs[source_role].get(entry.claim_id)
        owner = build_account_view(asdict(option), None, snapshot.get("_name_actors", {}),
                                   info.participants[role], info.participants[source_role])
        pending.append(dict(disclosure_turn=entry.turn, claim_id=entry.claim_id, assertion=option.assertion,
                            subject_name=owner["subject_name"], subject_relation=owner["subject_relation"],
                            stances=listener_stances(option, beliefs[role], options,
                                                    firsthand_source=bool(source and source.direct))))
    return pending


def resident_accounts_for_turn(accounts, info, role, draft, catalog, *, concern_id=None):
    """Offer one own private fact, keeping heard accounts and garden evidence."""
    options = {option.id: option for option in info.options}
    owner = info.participants[role]
    personal = [a for a in accounts if a["subject_id"] == owner and a["kind"] in ("former_life", "private_motive")]
    if not personal:
        return accounts
    personal_facts = {options[a["claim_id"]].fact_id for a in personal}
    heard = draft["turns"][-1].text if draft["turns"] and draft["turns"][-1].speaker != role else ""
    names = [p["name"] for p in catalog.values()]
    required = info.focus_claim_id if role == "initiator" else None
    selected = next((options[a["claim_id"]].fact_id for a in personal if a["claim_id"] == required), None)
    if selected is None:
        selected = next((options[a["claim_id"]].fact_id for a in personal if mentions_account(heard, a["assertion"], names)), None)
    if selected is None:
        selected = next((options[d.claim_id].fact_id for d in reversed(draft["disclosures"])
                         if draft["turns"][d.turn].speaker == role and options[d.claim_id].fact_id in personal_facts), None)
    if selected is None:
        selected = next((options[a["claim_id"]].fact_id for a in personal if a["claim_id"] == concern_id), None)
    if selected is None:
        used = {options[d.claim_id].fact_id for d in draft["disclosures"] if draft["turns"][d.turn].speaker == role}
        available = [o for o in info.options if o.fact_id in personal_facts - used and o.creator_id is None]
        choice = next((o for o in available if o.kind == "private_motive"), next(iter(available), None))
        selected = choice.fact_id if choice else None
    return [a for a in accounts if options[a["claim_id"]].fact_id not in personal_facts
            or options[a["claim_id"]].fact_id == selected]


def assessment_turn(draft, role):
    return next((i for i in reversed(range(len(draft["turns"]))) if draft["turns"][i].speaker == role), None)


def resident_turn_choices(snapshot, draft, role, private, *, closing=False, assessment=False):
    """Select this actor's accounts and eligible effects from the accepted prefix."""
    index = len(draft["turns"])
    other = "recipient" if role == "initiator" else "initiator"
    person, partner = snapshot[role], snapshot[other]
    catalog = snapshot.get("_name_actors", {})
    orientation = snapshot.get("arrival_focus", False)
    pending = pending_reactions(snapshot, draft, role)
    projection = resident_information(snapshot, draft)
    info = information_from_dict(snapshot["_information_validation"]) if snapshot.get("_information_validation") else None
    accounts = []
    if info and not orientation:
        beliefs, acquired, heard = projection
        options = {option.id: option for option in info.options}
        original_views = {a["claim_id"]: a for a in private.get("permitted_accounts", [])}
        for claim_id in dict.fromkeys([*info.allowed[role], *acquired[role], *[p["claim_id"] for p in pending]]):
            option = options[claim_id]
            belief = beliefs[role].get(claim_id)
            # An account newly heard is hearsay with this speaker as its knowledge
            # owner. Never copy the discloser's private motive, stance or evidence.
            view = deepcopy(original_views.get(claim_id, {}))
            if not view:
                disclosed = next((d for d in reversed(draft["disclosures"]) if d.claim_id == claim_id), None)
                if disclosed is not None:
                    supplied = next((a for source_role in ("initiator", "recipient")
                                     for a in snapshot[source_role].get("private_context", {}).get("permitted_accounts", [])
                                     if a["claim_id"] == claim_id), {})
                    view = {key: deepcopy(supplied[key]) for key in ("object_id", "object_name", "event_time") if key in supplied}
            selected = {**view, "claim_id": claim_id, "assertion": option.assertion, "kind": option.kind, "quantity": option.quantity,
                        "subject_id": option.subject_id, "original_speaker_id": option.originator_id}
            projected = build_account_view(selected, account_knowledge(belief, actor_id(person),
                creator_id=option.creator_id, fallback_source=actor_id(partner)), catalog,
                actor_id(person), actor_id(partner))
            view.update(claim_id=claim_id, assertion=option.assertion, kind=option.kind,
                        subject_id=option.subject_id, subject_name=projected["subject_name"],
                        original_speaker_id=option.originator_id, original_speaker_name=projected["original_speaker_name"],
                        immediate_source_id=projected["immediate_source_id"], immediate_source_name=projected["immediate_source_name"],
                        knowledge_owner_name=person["name"], basis=projected["basis"], stance=projected["stance"])
            for field in ("subject", "object", "original_speaker", "immediate_source"):
                if view.get(field + "_id") and field + "_relation" in projected:
                    view[field + "_relation"] = projected[field + "_relation"]
            if option.kind == "private_motive":
                for field in ("object_id", "object_name", "object_relation"):
                    view.pop(field, None)  # A goal target is not evidence of an event with that person.
            accounts.append(harvest_account_view(view))
        accounts = resident_accounts_for_turn(accounts, info, role, draft, catalog,
                                               concern_id=(private.get("concern") or {}).get("claim_id"))
    choices = dict(offers=[], acceptances=[], disclosures=[], third_party_opinions=[], leverage=[])
    if not orientation and snapshot.get("purpose") != "food_request":
        used = {a.proposal_id for a in draft["accepted_proposals"]}
        for proposal in snapshot.get("permitted_proposals", []):
            if proposal["id"] in used:
                continue
            if (not closing and not assessment and proposal["promisor_id"] == actor_id(person)
                    and proposal["id"] not in draft["offers"] and resident_effect_capacity(draft)["offers"] > 0):
                choices["offers"].append(proposal["id"])
            if proposal["recipient_id"] == actor_id(person) and proposal["id"] in draft["offers"]:
                choices["acceptances"].append(proposal["id"])
        if info:
            beliefs, acquired, heard = projection
            seen = {(draft["turns"][d.turn].speaker, d.claim_id) for d in draft["disclosures"]}
            # Telling an account back to the person it came from is an echo, not news for them.
            echoed = {options[d.claim_id].fact_id for d in draft["disclosures"] if draft["turns"][d.turn].speaker == other}
            required = (snapshot.get("_information_validation") or {}).get("focus_claim_id") if role == "initiator" and index == 0 else None
            choices["disclosures"] = [a["claim_id"] for a in accounts if (role, a["claim_id"]) not in seen
                                      and (a["claim_id"] == required or options[a["claim_id"]].fact_id not in echoed
                                           and a.get("immediate_source_id") != actor_id(partner))]
            # Pending reactions are chosen before this reply's effects are checked.
            # A closing assessment is recorded on the assessor's own last spoken turn.
            cutoff = assessment_turn(draft, role) if assessment else index
            heard_ids = {claim_id for (hearer, claim_id), turn in heard.items()
                         if hearer == role and cutoff is not None and turn < cutoff}
            if not assessment:
                heard_ids.update(p["claim_id"] for p in pending)
            pending_stances = {p["claim_id"]: p["stances"] for p in pending}
            # Hearsay is judged once, after the exchange; the python's own views of residents drive nothing.
            for account in (accounts if assessment and person.get("kind") != "python" else []):
                claim_id = account["claim_id"]
                option = options[claim_id]
                related = related_beliefs(option, beliefs[role].values(), options)
                if reputation_assessment_available(option, related, options,
                        heard=claim_id in heard_ids, participants=info.participants.values(),
                        believed=account["stance"] == "believed" or "believed" in pending_stances.get(claim_id, []),
                        assessed=any(draft["turns"][e.turn].speaker == role
                                     and options[e.claim_id].fact_id == option.fact_id for e in draft["third_party_opinions"])):
                    choices["third_party_opinions"].append(claim_id)
            # Pressure needs an account about the listener; one's own memories or motives are no leverage.
            choices["leverage"] = [a["claim_id"] for a in accounts if a["kind"] in LEVERAGE_ACCOUNT_KINDS
                and actor_id(partner) in (a["subject_id"], a.get("object_id"))
                and (a["claim_id"] in beliefs[role] or a["claim_id"] in heard_ids)
                and a["claim_id"] not in {entry.claim_id for entry in draft["leverage"]}]
    if not orientation and snapshot.get("purpose") != "food_request":
        focus = (snapshot.get("_information_validation") or {}).get("focus_claim_id")
        if role == "initiator" and index == 0 and focus:
            focus_fact = next(o.fact_id for o in info.options if o.id == focus)
            choices["disclosures"] = [claim for claim in choices["disclosures"]
                                      if next(o.fact_id for o in info.options if o.id == claim) == focus_fact]
    if any(speech_only(snapshot[r]) for r in ("initiator", "recipient")):
        choices["leverage"] = []
    return info, accounts, choices, pending
