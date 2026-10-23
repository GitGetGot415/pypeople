"""Public name tags and private, speaker-owned conversation context."""

import re
from copy import deepcopy
from constants import (PLAYER_NAME_TAG, SNAKE_NAME, LEVERAGE_ACCOUNT_KINDS, FOOD_REQUEST_TURN_COUNT,
                      CONVERSATION_MAX_AGREEMENTS, SNAKE_TURN_COUNT)
from dialogue_rules import (is_name_reference, name_pattern, unquoted_speech, mentions_account,
                           named_context, validate_base_prose, validate_opinion_reason,
                           validate_speaker_identity)
from dialogue_text import split_sentences, contains_question
from types import SimpleNamespace
from social import information_from_dict, listener_stances, validate_information


NAME_TAG_INSTRUCTION = f"""Every character has a readable name tag floating above their head.
Names are immediately available from those tags; nobody needs to ask, introduce,
memorize or learn a name. Use the supplied tag names, including full names.
If asked how you know someone's name without asking, explain naturally that you
can see their name tag floating above their head. This is visible in the world,
not a hidden roster or a memory. The human player's tag reads {PLAYER_NAME_TAG}.
A name tag reveals only a name: it does not reveal biography, private memories,
relationships, whereabouts, past meetings, possessions or accepted obligations.
Use I/my for yourself and you/your for the listener; preserve all account owners
and sources. Do not confuse knowing a name with having met or talked before.
"""

NPC_ARRIVAL_MEMORY = (
    "You remember waking up one day in an elevator that was going up. "
    "When its doors opened, you found yourself on the first floor of the apartment building."
)

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


def actor_id(person):
    return person.get("npc_id", person.get("actor_id"))


def own_name(person):
    return person.get("npc_name", person.get("name", person.get("character", {}).get("Name", "")))


def forms(name):
    return {name, *name.split()} if name else set()


def actor_catalog(world, location="garden"):
    people = [*world.npcs, *world.archived]
    live_ids = {p.id for p in world.npcs}
    catalog = {p.id: {"name": p.name, "self_name": p.character["Name"],
                      "_visible": p.id in live_ids and p.location == location} for p in people}
    catalog[world.snake.id] = {"name": SNAKE_NAME, "self_name": SNAKE_NAME,
                               "_visible": location == "garden", "references": ["the python guarding the apple tree"]}
    catalog[world.player.id] = {"name": PLAYER_NAME_TAG, "self_name": PLAYER_NAME_TAG, "kind": "player",
        "_visible": world.mode == "player" and world.player.alive and world.player.location == location,
        "references": ["the player"]}
    return catalog


def name_forms(catalog):
    return sorted({form for actor in catalog.values() for name in (actor["name"], actor["self_name"])
                   for form in forms(name)}, key=lambda s: (-len(s), s))


def account_references(catalog):
    return {actor["name"]: actor.get("references", []) for actor in catalog.values()}


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


def orientation_context(npc):
    world = npc.world
    from constants import ARRIVAL_DISCUSSION_THRESHOLD
    discussed = {record.id for record in world.history if record.includes(npc)
                 and record.orientation_discussed and record.turns
                 and orientation_speech(turn.text for turn in record.turns)}
    return {"days_here": world.day - npc.arrival_day,
            "orientation_discussions": len(discussed),
            "needs_orientation": len(discussed) < ARRIVAL_DISCUSSION_THRESHOLD,
            "adjustment": "finding bearings" if not discussed else "developing routines; questions remain unanswered"}


def orientation_speech(texts):
    pattern = (r"\b(?:arriv\w*|elevator|where (?:are|am) (?:we|i)|where (?:we|i) (?:are|am)|"
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
            raise ValueError("This orientation exchange does not authorize apple promises, private disclosures or leverage. Meet the person and find your bearings.")
        if snapshot.get("_dialogue_validation", True):
            if not discussed:
                raise ValueError("This is an orientation exchange. Discuss the unfamiliar surroundings or remembered elevator arrival, rather than pursuing a routine apple goal.")
    # This is derived bookkeeping, not a reason to regenerate otherwise valid speech.
    response.orientation_discussed = discussed
    return discussed


def character_context(character):
    """Base-only profile; private Background reaches speech through scoped accounts."""
    profile = {key: value for key, value in character.items()
                 if key not in ("Personality", "Voice", "Mannerisms", "Background")}
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
    return {"place": "One unfamiliar grassy area",
            "objects": ["One living apple tree", "One apartment building with residents' rooms"],
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
    """A present concern must not compete with private goals or a menu of bargains."""
    context = {"request_kind": "conversation", "purpose": "chat", "arrival_focus": True,
               "environment": local_environment(snapshot["initiator"].get("location", "garden")),
               "day": snapshot["initiator"].get("day", 1),
               "time_of_day": snapshot["initiator"].get("time_of_day", "day")}
    for role, other in (("initiator", "recipient"), ("recipient", "initiator")):
        person, partner = snapshot[role], snapshot[other]
        private = person.get("private_context", {})
        context[role] = {
            "npc_id": actor_id(person), "opinion_record_name": person["npc_name"],
            "location": person.get("location", "garden"),
            "character": {"Name": own_name(person), "personality": character_context(person["character"])["personality"]},
            "partner_name_tag": own_name(partner),
            "opinion_of_partner": _base_opinions(private.get("opinions", {}), person["npc_name"]).get(partner["npc_name"]),
            "private_context": {"situation": {key: value for key, value in private.get("situation", {}).items()
                                             if key not in ("known_danger", "apple_bargain")},
                                "concern": deepcopy(private.get("concern"))},
        }
    history = deepcopy(snapshot.get("shared_history", {}))
    if correction:
        history.pop("recent_dialogue", None)
    context["shared_history"] = history
    return context


def _catalog_for_prompt(catalog):
    return {actor_id: {key: value for key, value in actor.items() if not key.startswith("_")}
            for actor_id, actor in catalog.items()}


def world_context(snapshot, *, correction=False):
    """Supply relevant state without exposing other NPCs' private knowledge."""
    if snapshot.get("request_kind") == "conversation" and snapshot.get("arrival_focus", False):
        return _orientation_world_context(snapshot, correction=correction)
    if snapshot.get("request_kind") in ("snake_turn", "conversation_closing", "resident_turn", "resident_assessment", "handoff_thanks"):
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
              *snapshot.get("nearby_npcs", []), *snapshot.get("other_npcs", [])]
    actor_names = {person.get("npc_id", person.get("actor_id")): person.get("npc_name", person.get("name"))
                   for person in people if person.get("npc_id", person.get("actor_id"))}

    def person_context(person, include_reply=False):
        person = {key: value for key, value in person.items() if not key.startswith("_")}
        if "character" in person:
            person["character"] = character_context(person["character"])
        if not include_reply:
            person.pop("last_reply", None)
        elif person.get("last_reply") and not _ordinary_context_prose(person["last_reply"], (person.get("npc_name", ""), SNAKE_NAME)):
            person.pop("last_reply", None)
        if "memories" in person:
            person["memories"] = [memory for memory in person["memories"]
                                  if _ordinary_context_prose(memory.get("summary", ""), (person.get("npc_name", ""), SNAKE_NAME))]
        if conversation or snapshot.get("request_kind") in ("snake_encounter", "snake_decision"):
            # Rendering and movement bookkeeping do not help write a conversation.
            fields = {"npc_id", "npc_name", "character", "activity", "satisfaction", "inventory",
                      "memories", "private_context", "arrival_day", "location",
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
            private["opinions"] = _base_opinions(private.get("opinions", {}), person["npc_name"])
            if "snake_attitudes" in private:
                private["snake_attitudes"] = {key: value for key, value in private["snake_attitudes"].items()
                    if not key.endswith("_reason") or _ordinary_context_prose(value, (person["npc_name"], SNAKE_NAME))}
            private["promises"] = [
                {**promise, "giver_name": actor_names.get(promise["promisor_id"], promise["promisor_id"]),
                 "receiver_name": actor_names.get(promise["recipient_id"], promise["recipient_id"])}
                for promise in private.get("promises", [])]
        return person

    snake_request = snapshot.get("request_kind") in ("snake_encounter", "snake_decision", "snake_player")
    context = snapshot.copy() if conversation or snake_request else person_context(snapshot, include_reply=True)
    context = {key: value for key, value in context.items() if not key.startswith("_")}
    context.pop("_information_validation", None)
    current = snapshot["initiator"] if conversation else snapshot["snake"] if snake_request else snapshot
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
        if "tree" in context["environment"]:
            context["environment"]["tree"]["shaking"] = current.get("tree_is_shaking", False)
        if focus:
            context["discussion_focus"] = {key: focus[key] for key in ("assertion", "kind")}
            context["discussion_focus"]["subject_name"] = actor_names.get(focus["subject_id"])
            if "offering remained unpaid" in focus["assertion"].casefold() and focus["affirmed"]:
                context["discussion_focus"].update(what_was_received="one apple upfront",
                                                   what_was_refused=refused_offering(focus["assertion"]),
                                                   remaining_obligation="unpaid")
        if context.get("shared_history"):
            history = context["shared_history"] = context["shared_history"].copy()
            names = {snapshot[role]["npc_name"] for role in ("initiator", "recipient")}
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
    if snake_request:
        context["snake"] = {key: value for key, value in context["snake"].items()
                            if not key.startswith("_") and key != "doctrine"}
        context["transaction"] = snapshot["transaction"] if "transaction" in snapshot else {
            "apple_giver": snapshot["snake"]["name"], "apple_receiver": snapshot["resident"]["npc_name"],
            "can_give_apple": snapshot["can_give_apple"], "immediate_exchange": True}
        if snapshot.get("request_kind") == "snake_decision":
            # The guard assesses accepted speech, never the resident's private biography.
            context.pop("resident", None)
            context.pop("offering_accounts", None)
    for key in ("other_npcs", "nearby_npcs"):
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
    return actor if isinstance(actor, str) else actor.get("name", actor.get("npc_name", ""))


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


def _resident_information(snapshot, draft, *, incomplete=True):
    return validate_information(SimpleNamespace(**draft), snapshot.get("_information_validation"),
        names=(*[p["name"] for p in snapshot.get("_name_actors", {}).values()], SNAKE_NAME),
        participant_names={role: snapshot[role]["npc_name"] for role in ("initiator", "recipient")},
        dialogue_validation=snapshot.get("_dialogue_validation", True),
        actor_references=account_references(snapshot.get("_name_actors", {})), incomplete=incomplete)


def _pending_reactions(snapshot, draft, role):
    info = information_from_dict(snapshot["_information_validation"]) if snapshot.get("_information_validation") else None
    if info is None:
        return []
    beliefs, _, _ = _resident_information(snapshot, draft)
    options = {option.id: option for option in info.options}
    pending = []
    for entry in draft["disclosures"]:
        source_role = draft["turns"][entry.turn].speaker
        if source_role == role or entry.listener_stance is not None:
            continue
        option = options[entry.claim_id]
        source = beliefs[source_role].get(entry.claim_id)
        pending.append(dict(disclosure_turn=entry.turn, claim_id=entry.claim_id, assertion=option.assertion,
                            stances=listener_stances(option, beliefs[role], options,
                                                    firsthand_source=bool(source and source.direct))))
    return pending


def _resident_accounts_for_turn(accounts, info, role, draft, catalog, *, concern_id=None):
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
    speaker = dict(npc_id=actor_id(person), name=own_name(person), kind="human resident", location=person.get("location", "garden"),
                   opinion_record_name=person["npc_name"], personality=character_context(person["character"]).get("personality", ""),
                   situation=private.get("situation", {}), concern=deepcopy(private.get("concern")),
                   opinion_of_listener=private.get("opinions", {}).get(partner["npc_name"]))
    listener = dict(npc_id=actor_id(partner), name=own_name(partner), kind="human resident", location=partner.get("location", "garden"))
    if not assessment and speaker["opinion_of_listener"]:
        # Reasons are private assessments, often long or written in third person;
        # feeding their wording into speech teaches the model to recite them.
        speaker["opinion_of_listener"] = {"opinion": speaker["opinion_of_listener"]["opinion"]}
    pending = _pending_reactions(snapshot, draft, role)
    projection = _resident_information(snapshot, draft)
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
            selected = {**view, "claim_id": claim_id, "assertion": option.assertion, "kind": option.kind,
                        "subject_id": option.subject_id, "original_speaker_id": option.originator_id}
            projected = build_account_view(selected, account_knowledge(belief, actor_id(person),
                creator_id=option.creator_id, fallback_source=actor_id(partner)), catalog,
                actor_id(person), actor_id(partner))
            view.update(claim_id=claim_id, assertion=option.assertion, kind=option.kind,
                        subject_id=option.subject_id, subject_name=projected["subject_name"],
                        original_speaker_id=option.originator_id, original_speaker_name=projected["original_speaker_name"],
                        immediate_source_id=projected["immediate_source_id"], immediate_source_name=projected["immediate_source_name"],
                        knowledge_owner_name=person["npc_name"], basis=projected["basis"], stance=projected["stance"])
            for field in ("subject", "object", "original_speaker", "immediate_source"):
                if view.get(field + "_id") and field + "_relation" in projected:
                    view[field + "_relation"] = projected[field + "_relation"]
            if option.kind == "private_motive":
                for field in ("object_id", "object_name", "object_relation"):
                    view.pop(field, None)  # A goal target is not evidence of an event with that person.
            accounts.append(view)
        accounts = _resident_accounts_for_turn(accounts, info, role, draft, catalog,
                                               concern_id=(private.get("concern") or {}).get("claim_id"))
    choices = dict(offers=[], acceptances=[], disclosures=[], third_party_opinions=[], leverage=[])
    if not orientation and snapshot.get("purpose") != "food_request":
        used = {a.proposal_id for a in draft["accepted_proposals"]}
        for proposal in snapshot.get("permitted_proposals", []):
            if proposal["id"] in used:
                continue
            if (not closing and not assessment and proposal["promisor_id"] == actor_id(person)
                    and proposal["id"] not in draft["offers"] and len(draft["offers"]) < CONVERSATION_MAX_AGREEMENTS):
                choices["offers"].append(proposal["id"])
            if proposal["recipient_id"] == actor_id(person) and proposal["id"] in draft["offers"]:
                choices["acceptances"].append(proposal["id"])
        if info:
            beliefs, acquired, heard = projection
            seen = {(draft["turns"][d.turn].speaker, d.claim_id) for d in draft["disclosures"]}
            choices["disclosures"] = [a["claim_id"] for a in accounts if (role, a["claim_id"]) not in seen]
            # Pending reactions are chosen before this reply's effects are checked.
            heard_ids = {claim_id for (hearer, claim_id), turn in heard.items() if hearer == role and turn < index}
            heard_ids.update(p["claim_id"] for p in pending)
            pending_stances = {p["claim_id"]: p["stances"] for p in pending}
            for account in accounts:
                claim_id = account["claim_id"]
                option = options[claim_id]
                related = [b for b in beliefs[role].values() if options[b.claim_id].fact_id == option.fact_id]
                if (claim_id in heard_ids and option.subject_id not in info.participants.values()
                        and (account["stance"] == "believed" or "believed" in pending_stances.get(claim_id, []))
                        and not any(b.direct or b.assessed and options[b.claim_id].affirmed == option.affirmed for b in related)
                        and not any(draft["turns"][e.turn].speaker == role
                                    and options[e.claim_id].fact_id == option.fact_id for e in draft["third_party_opinions"])):
                    choices["third_party_opinions"].append(claim_id)
            choices["leverage"] = [a["claim_id"] for a in accounts if a["kind"] in LEVERAGE_ACCOUNT_KINDS
                and (a["claim_id"] in beliefs[role] or a["claim_id"] in heard_ids)
                and a["claim_id"] not in {entry.claim_id for entry in draft["leverage"]}]
    preceding = [dict(name=own_name(snapshot[t.speaker]), role=t.speaker, text=t.text) for t in draft["turns"]]
    state = dict(request_kind="resident_assessment" if assessment else "handoff_thanks" if thanks else "resident_turn",
                 speaker_role=role, turn_index=index, purpose=snapshot.get("purpose", "chat"),
                 interaction_id=snapshot.get("interaction_id"), arrival_focus=orientation, speaker=speaker, listener=listener,
                 opinion_target=partner["npc_name"], pending_reactions=pending, choices=choices,
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
        for field in ("goal", "promises", "snake_attitudes", "immediate_objective"):
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
        state["task"] = ("Find your bearings: raise an arrival or surroundings concern. "
                         "Suggest or respond to one practical first step without inventing discoveries. "
                         "Names are already visible on floating name tags. The cause of arrival and way home remain unknown. "
                         if index == 0 else
                         "Answer the partner's actual point about arrival or surroundings and respond to their practical suggestion. "
                         "Use visible name tags without requiring introductions.")
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
            focus_fact = next(o.fact_id for o in info.options if o.id == focus)
            choices["disclosures"] = [claim for claim in choices["disclosures"]
                                      if next(o.fact_id for o in info.options if o.id == claim) == focus_fact]
            state["task"] += " Disclose the required task account or a permitted denial of the same fact now."
        elif index == 0 and snapshot.get("opening_kind") in ("recollection", "news"):
            state["task"] += (" Lead with your concern's selected account in your own words, retaining its actual owner, source and stance. "
                              "Discuss the account itself rather than promising a later story. Record a disclosure only if actually spoken.")
    if assessment:
        state["task"] = "Assess your partner from the fixed transcript and your own knowledge. Resolve only your pending heard disclosures. Add no speech or other effects."
    elif thanks:
        if draft.get("gift"):
            state["task"] = "Acknowledge the accepted gift briefly after its successful handoff, preserving creator and giver roles. This speech is reserved for the actual handoff."
        else:
            state["task"] = "Thank the listener briefly for the one apple just successfully handed to you. This speech is reserved for the actual handoff."
            state["food_request"]["transferred"] = True
            speaker["inventory"]["fruit"] = speaker["inventory"].get("fruit", 0) + 1
        for field in ("permitted_accounts", "account_focus", "goal", "promises", "snake_attitudes", "immediate_objective", "concern"):
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
    return state


def _snake_turn_snapshot(snapshot, decision, turns, *, outcome_turn=2):
    """One speaker's knowledge; reward selection happens only after the resident speaks."""
    snake, resident = snapshot["snake"], snapshot["resident"]
    index = len(turns)
    role = "initiator" if index % 2 == 0 else "recipient"
    guard = dict(actor_id=snake["actor_id"], name=snake["name"], kind="python")
    human = dict(actor_id=resident["npc_id"], name=resident["npc_name"], kind="human resident")
    speaker, listener = (guard.copy(), human) if role == "initiator" else (human.copy(), guard)
    accounts = snake["known_accounts"] if role == "initiator" else snapshot["offering_accounts"]
    if role == "initiator" and index >= outcome_turn and decision.account_id:
        disclosed = next((a for a in snapshot["offering_accounts"] if a["claim_id"] == decision.account_id), None)
        if disclosed:
            disclosed = {**disclosed, "basis": "hearsay", "immediate_source_id": human["actor_id"],
                         "immediate_source_name": human["name"], "knowledge_owner_name": guard["name"], "stance": "uncertain"}
            accounts = [*accounts, disclosed]
    def speech_account(account):
        projected = build_account_view(account, None, snapshot.get("_name_actors", {}),
                                       speaker["actor_id"], listener["actor_id"])
        view = {key: projected[key] for key in account if key in (
            "claim_id", "assertion", "kind", "subject_id", "subject_name", "object_id", "object_name",
            "original_speaker_name", "immediate_source_name", "basis", "stance", "event_time")}
        for field in ("subject", "object", "original_speaker", "immediate_source"):
            if field + "_relation" in projected:
                view[field + "_relation"] = projected[field + "_relation"]
        if account["kind"] == "private_motive":
            view.pop("object_name", None)
        return view

    pair = {guard["actor_id"], human["actor_id"]}
    firsthand = [a for a in accounts if a.get("basis") == "firsthand"
        and a.get("kind") not in ("former_life", "private_motive")
        and {a.get("subject_id"), a.get("object_id")} == pair]
    speaker["firsthand_events_with_listener"] = [speech_account(a) for a in firsthand]
    voluntary_reply = role == "recipient" and (index == 1 or (index == 3
        and (index < outcome_turn or decision.decision == "talk" and not decision.account_id)
        and snapshot["can_give_apple"] and snapshot["purpose"] != "introduction"))
    if voluntary_reply:
        permitted = accounts if snapshot["purpose"] != "introduction" else []
        speaker["permitted_accounts"] = [speech_account(a) for a in permitted]
        # One private menu is enough. Unselected biographies are not a second speech task.
        speaker["known_accounts"] = [speech_account(a) for a in firsthand]
    else:
        public = accounts if role == "initiator" else [a for a in accounts
            if a["kind"] not in ("former_life", "private_motive") or a["claim_id"] == decision.account_id]
        speaker["known_accounts"] = [speech_account(a) for a in public]
    if role == "recipient" and index >= 1 and decision.snake_account_id:
        heard = next((a for a in snake["known_accounts"] if a["claim_id"] == decision.snake_account_id), None)
        if heard:
            speaker["heard_accounts"] = [speech_account({**heard, "basis": "hearsay",
                "immediate_source_id": guard["actor_id"], "immediate_source_name": guard["name"]})]
    speaker["first_meeting"] = snapshot["first_meeting"]
    if role == "recipient":
        private = resident["private_context"]
        speaker.update(situation=private.get("situation", {}), inventory=resident["inventory"],
                       view_of_listener=private["snake_attitudes"])
    from snake import SNAKE_BEHAVIOR
    direction = SNAKE_BEHAVIOR if role == "initiator" else character_context(resident["character"])["personality"]
    selected = decision.snake_account_id if role == "initiator" else (decision.account_id if index == outcome_turn - 1 else "")
    account = next((a.copy() for a in accounts if a["claim_id"] == selected), None)
    account = speech_account(account) if account else None
    if index == 0:
        task = "Ask what the visitor wishes to discuss. Do not invent their thoughts or a witnessed arrival."
        if snapshot["can_give_apple"] and snapshot["purpose"] != "introduction":
            task = ("Explicitly offer one apple in exchange for useful information now, then ask one concrete question "
                    "about the visitor's own interests, experiences or news. Include the apple offer in your spoken words. "
                    "An offer is not a handoff or a reward already earned. Do not invent a witnessed arrival.")
        if decision.snake_account_id:
            task = ("Discuss only account_to_share in this opening. Tell its concrete content first, preserving "
                    "who made the claim and who heard it; then ask the visitor for their own view or relevant knowledge. "
                    "Do not substitute a general greeting, stock report, or question about what to discuss. "
                    "Your own past claims remain yours; a person who heard you is not their author. "
                    "State your divine role confidently when relevant, without treating your own authority as hearsay.")
            if snapshot["can_give_apple"] and snapshot["purpose"] != "introduction":
                task += " Explicitly offer one apple for useful relevant information in their answer; include the offer in your spoken words."
        if snapshot.get("known_warning"):
            task += " Ask the visitor to answer this unresolved warning before judging: " + snapshot["warning_record"]
    elif voluntary_reply:
        task = ("Answer the guard's actual words using your own personality and knowledge. "
                "Choose whether to share an account from speaker.permitted_accounts together with your speech. "
                "You may instead decline, ask a question, or express uncertainty with empty account_id. "
                "Do not invent a memory or promise later payment. The listener is the tree guard claiming divine authority; "
                "do not adopt its role, its first-person statements or its memories as your own.")
    elif index in (2, outcome_turn):
        outcome = decision.decision if index >= outcome_turn else "talk"
        task = {"talk": "Respond to the visitor's actual point and explain what information would interest you. Give no apple now.",
                "give_apple": "Hand over one apple now for the information just heard; no further payment is owed.",
                "warn": "State the sourced accusation and explicitly warn of being eaten; invite an answer.",
                "eat": "Announce condemnation for the supplied accusation after the earlier warning and response."}[outcome]
        if (index == 2 and outcome == "talk" and not decision.account_id and snapshot["can_give_apple"]
                and snapshot["purpose"] != "introduction"):
            task += (" Ask one specific follow-up inviting information about their own interests, experiences or relevant news. "
                     "Make clear that useful information now can earn one apple. A refusal is safe; do not demand a secret or a future service.")
        if decision.accusation:
            task += " Accusation: " + decision.accusation
    else:
        task = "Answer the latest point naturally, preserving the recorded outcome; no additional transfer or punishment."
    if snapshot["purpose"] == "introduction":
        task += " Help the visitor find their bearings. Do not initiate an apple bargain or demand private information."
    preceding = [dict(name=snake["name"] if t.speaker == "initiator" else resident["npc_name"],
                      role=t.speaker, text=t.text) for t in turns]
    speaker["earlier_speech"] = [t["text"] for t in preceding if t["role"] == role]
    transaction = dict(apple_giver=snake["name"], apple_receiver=resident["npc_name"],
                       immediate_exchange=True, can_give_apple=snapshot["can_give_apple"],
                       selected_outcome=decision.decision if index >= outcome_turn else "talk", outcome_turn=outcome_turn)
    environment = world_context(snapshot)["environment"]
    if index > outcome_turn and decision.decision == "give_apple":
        transaction.update(can_give_apple=False, apple_already_received=True)
        if role == "recipient":
            speaker["inventory"] = {**resident["inventory"], "fruit": resident["inventory"].get("fruit", 0) + 1}
        environment["tree"] = {**environment["tree"], "fruit": environment["tree"]["fruit"] - 1}
    state = dict(request_kind="snake_turn", turn_index=index, speaker_role=role,
        interaction_id=snapshot["interaction_id"], _name_actors=snapshot.get("_name_actors", {}),
        _dialogue_validation=snapshot.get("_dialogue_validation", True), speaker=speaker, listener=listener,
        day=snake["day"], time_of_day=snake["phase"], environment=environment,
        transaction=transaction, task=task, accusation=snapshot["accusation"], audience_ids=[],
        account_to_share=account if index == 0 or index == outcome_turn - 1 else None, preceding_turns=preceding[-1:],
        _preceding_turns=preceding, _speaker_behavior=direction)
    if index >= SNAKE_TURN_COUNT and turns:
        state["question_to_answer"] = _last_question(turns[-1].text)
    return state
