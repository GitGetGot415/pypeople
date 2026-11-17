from constants import CONCERN_LIFETIME_RANGE, CONCERN_WEIGHTS, ROUTINE_FACT_KINDS
from simulation.social.records import ConversationConcern, fact_by_id, claim_by_id, owns_recollection
from simulation.social.information import account_content, account_view
from simulation.social.promises import pending_promises


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
        plans.extend(f"Deliver an apple to {npc.world.participant_by_id(p.recipient_id).name}, due {p.deadline}"
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
                 and fact_by_id(world, claim_by_id(world, b.claim_id).fact_id).kind in ("killing", "death")
                 and not any(r.turns and r.topic == danger_topic(npc, b.claim_id) and npc.name in r.participants
                             for r in world.history)), None)


def refresh_concern(npc):
    """Choose locally at a safe decision boundary, never during an interaction."""
    if npc.interaction is not None or not npc.alive:
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
