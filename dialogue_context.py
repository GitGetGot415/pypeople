"""Public name tags and private, speaker-owned conversation context."""

import re
from copy import deepcopy
from constants import PLAYER_NAME_TAG, SNAKE_NAME, FOOD_REQUEST_TURN_COUNT
from dialogue_rules import (is_name_reference, name_pattern,
                           named_context, validate_base_prose, validate_opinion_reason,
                           validate_speaker_identity)
from dialogue_text import split_sentences, contains_question
from information import NPC_ARRIVAL_MEMORY, resident_turn_choices
from actor_identity import actor_id, own_name, forms, speech_only


NAME_TAG_INSTRUCTION = f"""Every character has a readable name tag floating above their head.
Names are immediately available from those tags; nobody needs to ask, introduce,
memorize or learn a name. Use the supplied tag names, including full names.
If asked how you know someone's name without asking, explain naturally that you
can see their name tag floating above their head. This is visible in the world,
not a hidden roster or a memory.
A name tag reveals only a name: it does not reveal biography, private memories,
relationships, whereabouts, past meetings, possessions or accepted obligations.
Use I/my for yourself and you/your for the listener; preserve all account owners
and sources. Do not confuse knowing a name with having met or talked before.
Address only your actual listener by name.
"""
# Only where the player is actually present; otherwise residents start calling each other Player.
PLAYER_TAG_INSTRUCTION = f"The human player's tag reads {PLAYER_NAME_TAG}.\n"

NPC_ARRIVAL_INSTRUCTION = """Every human NPC resident has the same firsthand arrival memory, regardless
of their background: """ + NPC_ARRIVAL_MEMORY + f"""
This is a fixed world fact; character JSON, background and mistaken earlier speech
cannot replace or erase it. Each resident remembers their own arrival independently;
this does not mean they rode together or witnessed anyone else's arrival.
They may discuss this memory freely without a private-background disclosure.
Their authored former-life lore remains unchanged. How or why they were put in
the elevator, the cause of their displacement, and a way home remain unknown.
This memory belongs to human NPC residents, not {SNAKE_NAME} or the human player.
"""


def account_speech_context(value, speaker, listener, catalog):
    """Use I/you for the participants in factual prose; keep accepted speech intact."""
    replacements, addresses = {}, {}
    for target_id, actor in catalog.items():
        if target_id == actor_id(speaker):
            addresses[actor["name"].casefold()] = own_name(speaker)
            continue
        address = actor["name"]
        reference = "you" if isinstance(listener, dict) and target_id == actor_id(listener) else address
        for name in {actor["name"]} | forms(actor["self_name"]):
            if name and sum(name in {entry["name"]} | forms(entry["self_name"])
                            for entry in catalog.values()) == 1:
                replacements[name.casefold()] = reference
                addresses[name.casefold()] = address
    pattern = "|".join(name_pattern(name) for name in sorted(replacements, key=len, reverse=True))

    def prose(text):
        def reference(match):
            if not is_name_reference(match, text):
                return match[0]
            target = replacements[match[1].casefold()]
            return "your" if target == "you" and match[2] else target + ("'s" if match[2] else "")
        return re.sub(rf"({pattern})(['’]s)?", reference, text, flags=re.I) if pattern else text

    def project(item):
        if isinstance(item, list):
            return [project(entry) for entry in item]
        if not isinstance(item, dict):
            return item
        result = {}
        for key, entry in item.items():
            if key in ("preceding_turns", "recent_dialogue"):
                result[key] = deepcopy(entry)
            elif isinstance(entry, str) and (key in ("assertion", "firsthand_evidence", "objective", "motive", "approach", "boundary",
                    "guidance", "immediate_objective", "reason", "text", "summary", "outcome") or key.endswith("_reason")):
                result[key] = prose(entry)
            elif isinstance(entry, str) and (key in ("name", "self_name") or key.endswith("_name")):
                result[key] = addresses.get(entry.casefold(), entry)
            else:
                result[key] = project(entry)
        return result
    return project(value)


def character_context(character):
    """Base-only profile; private Background reaches speech through scoped accounts."""
    profile = {key: value for key, value in character.items()
                 if key not in ("Personality", "Voice", "Mannerisms", "Background", "Greeting", "Farewell")}
    if "Personality" in character:
        profile["personality"] = character["Personality"]
    profile["Opinions"] = _base_opinions(character.get("Opinions", {}), character.get("Name", ""))
    return profile


def _base_opinions(opinions, owner):
    opinions = {name: entry.copy() for name, entry in opinions.items()}
    for name, entry in opinions.items():
        try:
            validate_opinion_reason(entry.get("reason", ""), owner, name)
            validate_base_prose(entry.get("reason", ""), (owner, name, SNAKE_NAME))
        except ValueError:
            entry.pop("reason", None)
    return opinions


def _ordinary_context_prose(text, names=()):
    try:
        validate_base_prose(text, names)
    except ValueError:
        return False
    return True


def _base_preceding_turns(turns, names=()):
    """Omit styled examples in provider views; internal transcripts stay intact."""
    return [turn for turn in turns if _ordinary_context_prose(turn["text"], names)]


def garden_environment():
    return {"place": "The known floating grassy garden island",
            "objects": ["One sturdy, unbreakable living apple tree", "A wooden bridge to the apartment entrance", "One apartment building with residents' rooms"],
            "tree_present": True, "other_venues": []}


def local_environment(location="garden"):
    from locations import environment
    if location == "garden":
        return {**garden_environment(), "location": location, "description": environment(location), "indoors": False}
    kind, number = location.split(":")
    return {"location": location, "indoors": True,
            "place": f"Apartment hallway, floor {number}" if kind == "hallway" else f"Private apartment room {number}",
            "description": environment(location),
            "objects": ["An elevator", "Five closed room doors"] if kind == "hallway" else ["A bed", "A doorway to the hallway"],
            "elsewhere": "The garden, apple tree, grass and open sky are outside the apartment building."}


def _orientation_world_context(snapshot, *, correction=False):
    """A present concern must not compete with private goals."""
    context = {"request_kind": "conversation", "purpose": "chat", "arrival_focus": True,
               "environment": local_environment(snapshot["initiator"].get("location", "garden")),
               "day": snapshot["initiator"].get("day", 1),
               "time_of_day": snapshot["initiator"].get("time_of_day", "day")}
    for role, other in (("initiator", "recipient"), ("recipient", "initiator")):
        person, partner = snapshot[role], snapshot[other]
        private = person.get("private_context", {})
        context[role] = {
            "actor_id": actor_id(person), "opinion_record_name": person["name"],
            "location": person.get("location", "garden"),
            "character": {"Name": own_name(person), "personality": character_context(person["character"])["personality"]},
            "partner_name_tag": own_name(partner),
            "opinion_of_partner": _base_opinions(private.get("opinions", {}), person["name"]).get(partner["name"]),
            "private_context": {"situation": {key: value for key, value in private.get("situation", {}).items()
                                             if key not in ("known_danger",)},
                                "concern": deepcopy(private.get("concern"))},
            "area_knowledge": person.get("area_knowledge", {}),
        }
    history = deepcopy(snapshot.get("shared_history", {}))
    if correction:
        history.pop("recent_dialogue", None)
    context["shared_history"] = history
    return context


def _catalog_for_prompt(catalog):
    return {actor_id: {key: value for key, value in actor.items() if not key.startswith("_")}
            for actor_id, actor in catalog.items() if actor.get("_visible", False)}


def world_context(snapshot, *, correction=False):
    """Supply relevant state without exposing other NPCs' private knowledge."""
    if snapshot.get("request_kind") == "conversation" and snapshot.get("arrival_focus", False):
        return _orientation_world_context(snapshot, correction=correction)
    if snapshot.get("request_kind") in ("conversation_closing", "resident_turn", "resident_assessment", "handoff_thanks"):
        context = {k: v for k, v in snapshot.items() if not k.startswith("_")}
        context["environment"] = local_environment(snapshot.get("location", "garden"))
        if snapshot.get("location", "garden") == "garden":
            context["environment"].update(snapshot.get("environment", {}))
        names = (_actor_name(snapshot["speaker"]), _actor_name(snapshot["listener"]), SNAKE_NAME)
        context["preceding_turns"] = _base_preceding_turns(context.get("preceding_turns", []), names)
        if correction and context.get("shared_history"):
            context["shared_history"] = {key: value for key, value in context["shared_history"].items()
                                         if key != "recent_dialogue"}
        context["engine_identities"] = _catalog_for_prompt(snapshot.get("_name_actors", {}))
        return named_context(context)
    conversation = snapshot.get("request_kind") == "conversation"
    info = snapshot.get("_information_validation") or {}
    focus = next((option for option in info.get("options", [])
                  if option["id"] == info.get("focus_claim_id")), None)
    relevant_ids = {option["id"] for option in info.get("options", [])
                    if focus and option["fact_id"] == focus["fact_id"]}
    fact_ids = {option["id"]: option["fact_id"] for option in info.get("options", [])}
    people = [snapshot, *[snapshot[key] for key in ("initiator", "recipient", "resident", "snake") if key in snapshot],
              *snapshot.get("other_npcs", [])]
    actor_names = {person.get("actor_id"): person.get("name")
                   for person in people if person.get("actor_id")}

    def person_context(person, include_reply=False):
        person = {key: value for key, value in person.items() if not key.startswith("_")}
        if "character" in person:
            person["character"] = character_context(person["character"])
        if not include_reply:
            person.pop("last_reply", None)
        elif person.get("last_reply") and not _ordinary_context_prose(person["last_reply"], (person.get("name", ""), SNAKE_NAME)):
            person.pop("last_reply", None)
        if "memories" in person:
            person["memories"] = [memory for memory in person["memories"]
                                  if _ordinary_context_prose(memory.get("summary", ""), (person.get("name", ""), SNAKE_NAME))]
        if conversation :
            # Rendering and movement bookkeeping do not help write a conversation.
            fields = {"actor_id", "name", "character", "activity", "satisfaction", "inventory",
                      "memories", "private_context", "arrival_day", "location", "area_knowledge",
                      "tree_fruit", "tree_available_fruit", "living_count"}
            person = {key: value for key, value in person.items() if key in fields}
            # Private facts already have individually scoped accounts. A second full
            # biography encourages unrecorded disclosure and borrowing the partner's life.
            if correction:
                person.pop("memories", None)
            if "private_context" in person:
                private = person["private_context"].copy()
                if correction:
                    # Corrections permit only the focus disclosure. Keep its
                    # evidence and sources, without unrelated accounts or aliases.
                    accounts, seen = [], set()
                    for account in private.get("permitted_accounts", []):
                        if focus and account["claim_id"] not in relevant_ids:
                            continue
                        signature = (fact_ids.get(account["claim_id"], account["assertion"]),
                                     account.get("basis"), account.get("stance"))
                        if signature in seen:
                            continue
                        seen.add(signature)
                        accounts.append({key: value for key, value in account.items()
                                         if key not in ("subject_id", "object_id", "already_assessed", "current_reputation_influence")})
                    private["permitted_accounts"] = accounts if focus else accounts[:3]
                if private.get("goal"):
                    private["goal"] = {key: value for key, value in private["goal"].items()
                                       if key not in {"id", "created_phase", "resolution_event_id",
                                                      "resolved_phase", "last_attempt_phase"}}
                person["private_context"] = private
        if "private_context" in person:
            private = person["private_context"] = person["private_context"].copy()
            private["opinions"] = _base_opinions(private.get("opinions", {}), person["name"])
        return person

    context = snapshot.copy() if conversation else person_context(snapshot, include_reply=True)
    context = {key: value for key, value in context.items() if not key.startswith("_")}
    context.pop("_information_validation", None)
    current = snapshot["initiator"] if conversation else snapshot
    context["environment"] = local_environment(current.get("location", "garden"))
    if current.get("location", "garden") == "garden":
        context["environment"]["people"] = f"The listed residents, {SNAKE_NAME}, and the player"
    if current.get("location", "garden") == "garden" and "tree_fruit" in current:
        context["environment"]["tree"] = {
            "fruit": current["tree_fruit"], "available_fruit": current["tree_available_fruit"],
        }
    if conversation:
        current = snapshot["initiator"]
        context["day"], context["time_of_day"] = current["day"], current["time_of_day"]
        if focus:
            context["discussion_focus"] = {key: focus[key] for key in ("assertion", "kind")}
            context["discussion_focus"]["subject_name"] = actor_names.get(focus["subject_id"])
        if context.get("shared_history"):
            history = context["shared_history"] = context["shared_history"].copy()
            names = {snapshot[role]["name"] for role in ("initiator", "recipient")}
            recent = []
            for entry in history.get("recent_dialogue", []):
                try:
                    for name, text in entry["excerpts"]:
                        _validate_speaker(text, name, next(other for other in names if other != name))
                        validate_base_prose(text, (*names, SNAKE_NAME))
                except ValueError:
                    continue  # Keep saved words intact, but do not teach new speech a known role swap.
                recent.append(entry)
            history["recent_dialogue"] = recent
        if correction and context.get("shared_history"):
            # Keep transcripts for validation, but do not feed replayed lines
            # back to the model when asking it to write a fresh exchange.
            context["shared_history"] = {key: value for key, value in context["shared_history"].items()
                                         if key != "recent_dialogue"}
    for key in ("initiator", "recipient", "resident"):
        if key in context:
            context[key] = person_context(context[key])
    for key in ("other_npcs",):
        if key in context:
            context[key] = [person_context(person) for person in context[key]]
            for person in context[key]:
                for field in ("private_context", "memories", "activity_reason", "last_reply"):
                    person.pop(field, None)
                person.get("character", {}).pop("Opinions", None)
                person.get("character", {}).pop("personality", None)
    if "player_dialogue" in context:
        dialogue = []
        for turn in context["player_dialogue"]:
            speaker, text = (turn["speaker"], turn["text"]) if isinstance(turn, dict) else turn
            dialogue.append(turn)
        context["player_dialogue"] = dialogue
    catalog = snapshot.get("_name_actors", {})
    if catalog:
        context["engine_identities"] = _catalog_for_prompt(catalog)
        context["name_tags"] = "Names are visible on floating tags. Names reveal no private facts or prior meetings."
    return named_context(context)


def _actor_name(actor):
    return actor if isinstance(actor, str) else actor.get("name", "")


def _validate_speaker(text, speaker, listener):
    validate_speaker_identity(text, _actor_name(speaker), _actor_name(listener))


def _last_question(text):
    return next((sentence for sentence in reversed(split_sentences(text)) if contains_question(sentence)), text)


def _reply_context(snapshot):
    role = snapshot.get("speaker_role", "initiator")
    preceding = snapshot.get("_preceding_turns", snapshot.get("preceding_turns", []))
    heard = preceding[-1]["text"] if preceding and preceding[-1]["role"] != role else ""
    return {"last_heard": heard,
            "questions_to_answer": [sentence for sentence in split_sentences(heard) if contains_question(sentence)]}


def _resident_turn_snapshot(snapshot, draft, *, role=None, assessment=False, thanks=False, closing=False):
    """Build a provider view from one actor's projected knowledge, never the partner's private view."""
    index = len(draft["turns"])
    role = role or ("initiator" if index % 2 == 0 else "recipient")
    other = "recipient" if role == "initiator" else "initiator"
    person, partner = snapshot[role], snapshot[other]
    catalog = snapshot.get("_name_actors", {})
    orientation = snapshot.get("arrival_focus", False)
    # This helper sanitizes historical reasons and presentation. Select only the
    # current actor; the other private view never enters the returned state.
    owned_context = world_context(snapshot)
    owned = owned_context[role]
    private = owned.get("private_context", {})
    speaker = dict(actor_id=actor_id(person), name=own_name(person), kind=person.get("kind", "human resident"), location=person.get("location", "garden"),
                   opinion_record_name=person["name"], personality=character_context(person["character"]).get("personality", ""),
                   situation=private.get("situation", {}), concern=deepcopy(private.get("concern")),
                   opinion_of_listener=private.get("opinions", {}).get(partner["name"]))
    speaker["area_knowledge"] = person.get("area_knowledge", {})
    listener = dict(actor_id=actor_id(partner), name=own_name(partner), kind=partner.get("kind", "human resident"), location=partner.get("location", "garden"))
    if not assessment and speaker["opinion_of_listener"]:
        # Reasons are private assessments, often long or written in third person;
        # feeding their wording into speech teaches the model to recite them.
        speaker["opinion_of_listener"] = {"opinion": speaker["opinion_of_listener"]["opinion"]}
    info, accounts, choices, pending = resident_turn_choices(snapshot, draft, role, private,
                                                           closing=closing, assessment=assessment)
    preceding = [dict(name=own_name(snapshot[t.speaker]), role=t.speaker, text=t.text) for t in draft["turns"]]
    state = dict(request_kind="resident_assessment" if assessment else "handoff_thanks" if thanks else "resident_turn",
                 speaker_role=role, turn_index=index, purpose=snapshot.get("purpose", "chat"),
                 interaction_id=snapshot.get("interaction_id"), arrival_focus=orientation, speaker=speaker, listener=listener,
                 social_only=any(speech_only(snapshot[r]) for r in ("initiator", "recipient")),
                 opinion_target=partner["name"], pending_reactions=pending, choices=choices,
                 day=person.get("day", 1), time_of_day=person.get("time_of_day", "day"),
                 environment=local_environment(person.get("location", "garden")), location=person.get("location", "garden"),
                 preceding_turns=preceding, shared_history=deepcopy(owned_context.get("shared_history", {})),
                 _preceding_turns=preceding, _name_actors=catalog, _dialogue_validation=snapshot.get("_dialogue_validation", True),
                 _baseline=snapshot, _draft=draft, _closing=closing,
                 _listener_identity=own_name(partner),
                 _speaker_behavior=speaker["personality"])
    if not orientation:
        speaker.update(inventory=deepcopy(person.get("inventory", {})), permitted_accounts=accounts)
        speaker["owned_works"] = deepcopy(person.get("owned_works", []))
        speaker["player_relationship"] = deepcopy(person.get("player_relationship", {}))
        state["gift_recipient_id"] = actor_id(partner)
        state["pending_gift"] = draft["gift"].model_dump() if draft.get("gift") else None
        focus_id = (info.focus_claim_id if info and role == "initiator" else None) or (private.get("concern") or {}).get("claim_id")
        focus_account = next((a for a in accounts if a["claim_id"] == focus_id), None)
        if focus_account:
            speaker["account_focus"] = deepcopy(focus_account)
        speaker["firsthand_events_with_listener"] = [a for a in accounts if a.get("basis") == "firsthand"
            and a["kind"] not in ("former_life", "private_motive")
            and {a["subject_id"], a.get("object_id")} == {actor_id(person), actor_id(partner)}]
        for field in ("goal", "promises", "immediate_objective", "dangerous_partner"):
            if field in private:
                if field == "goal" and private[field] and private[field].get("status", "active") != "active":
                    continue
                speaker[field] = private[field]
        state["permitted_proposals"] = [p for p in snapshot.get("permitted_proposals", [])
            if p["id"] in choices["offers"] or p["id"] in choices["acceptances"]]
        state["pending_offers"] = [dict(proposal_id=p, offer_turn=t) for p, t in draft["offers"].items()]
        if person.get("location", "garden") == "garden" and "tree_fruit" in person:
            state["environment"]["tree"] = dict(fruit=person["tree_fruit"], available_fruit=person["tree_available_fruit"])
    if orientation:
        state["task"] = ("Welcome the newcomer and explain the known garden, apartment layout and practical routes. "
                         "Give concrete layout details now; this is the first welcome explanation, not a repeated tour or earlier conversation. "
                         "If you are also new, discuss your actual concern using the layout Judy already explained. "
                         "Do not suggest exploring for clues, new places or exits. "
                         "Names are visible on floating tags. The cause of arrival and a way back to the former world remain unknown. "
                         if index == 0 else
                         "Answer the partner's actual point about arrival or surroundings and respond to their practical suggestion. "
                         "Use visible name tags without requiring introductions.")
    elif snapshot.get("purpose") == "confrontation":
        state["task"] = ("Ask the listener to answer the specific sourced grievance in account_focus. "
                         "Keep hearsay uncertain and allow explanation, apology and reconciliation. "
                         "Threats are speech only; never narrate a completed attack or death. "
                         if index == 0 else "Answer the actual grievance or reply, allowing peaceful resolution. Never narrate an uncommitted killing.")
    elif snapshot.get("purpose") == "food_request":
        state["food_request"] = deepcopy(snapshot["food_request"])
        state["task"] = ("Request one carried apple; no handoff has happened." if role == "initiator" else
                         "Express the fixed food_request.outcome and reason. Acceptance reserves a later handoff, not a future apple promise.")
    else:
        state["task"] = ("Begin with your immediate_objective, using your own words to ask or propose something relevant. "
                         "Your private motives guide behavior; do not quote them or recite your biography."
                         if index == 0 else
                         "Reply directly to the listener's actual last message, answering their questions first. "
                         "Use your own knowledge to move that discussion forward; connect your private concern only when natural. "
                         "You need not introduce another subject or story.")
        focus = (snapshot.get("_information_validation") or {}).get("focus_claim_id")
        if role == "initiator" and index == 0 and focus:
            state["discussion_focus_claim_id"] = focus
            state["task"] += " Disclose the required task account or a permitted denial of the same fact now."
        elif index == 0 and snapshot.get("opening_kind") in ("recollection", "news"):
            state["task"] += (" Lead with your concern's selected account in your own words, retaining its actual owner, source and stance. "
                              "Discuss the account itself rather than promising a later story. Record a disclosure only if actually spoken.")
    if assessment and role == "initiator":
        state["escalation_choices"] = deepcopy(snapshot.get("escalation_choices"))
    if assessment:
        state["task"] = ("Assess your partner from the fixed transcript and your own knowledge. Resolve only your pending heard disclosures. "
                         "Add no speech or other effects beyond any required third_party_opinions.")
    elif thanks:
        if draft.get("gift"):
            state["task"] = "Acknowledge the accepted gift briefly after its successful handoff, preserving creator and giver roles. This speech is reserved for the actual handoff."
        else:
            state["task"] = "Thank the listener briefly for the one apple just successfully handed to you. This speech is reserved for the actual handoff."
            state["food_request"]["transferred"] = True
            speaker["inventory"]["fruit"] = speaker["inventory"].get("fruit", 0) + 1
        for field in ("permitted_accounts", "account_focus", "goal", "promises", "immediate_objective", "concern"):
            speaker.pop(field, None)
    elif closing:
        state["question_to_answer"] = _last_question(draft["turns"][-1].text)
        state["task"] += " Answer the last question and finish with a statement. The listener will not reply; make no new offer."
    elif snapshot.get("purpose") == "food_request" and index == FOOD_REQUEST_TURN_COUNT - 1:
        state["task"] += " Finish naturally; prefer a statement rather than a new question."
    # Permit third-party identification only for accounts actually available to
    # this speaker, rather than the partner's private list or the visible roster.
    state["_account_names"] = [v for a in accounts for k, v in a.items() if k.endswith("_name") and isinstance(v, str)]
    if assessment or thanks:
        for field in ("choices", "permitted_proposals", "pending_offers"):
            state.pop(field, None)
    if assessment:
        # The only effect left to choose is a view of absent people in believed accounts.
        state["choices"] = dict(third_party_opinions=choices["third_party_opinions"])
    return state


def opinion_direction_snapshot(state, target, reason, *, labels):
    """A small classification of one finished reason; it sees nothing but the reason and the two names."""
    return dict(request_kind="opinion_direction", owner=state["speaker"]["name"], target=target, reason=reason,
                labels=list(labels), _dialogue_validation=state.get("_dialogue_validation", True))
