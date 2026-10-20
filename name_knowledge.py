"""Small, directional name knowledge; engine identities never grant familiarity."""

from copy import deepcopy
import re

from constants import SNAKE_NAME
from dialogue_rules import is_name_reference, name_pattern, unquoted_speech
from dialogue_text import split_sentences


NAME_INSTRUCTION = """Engine names, actor IDs, roster labels, goals, opinions, memories,
accounts, promises and correction metadata identify actors for bookkeeping only.
They never teach a resident a name. name_knowledge is the authority for spoken
names: use only your own authored name and learned forms, without surnames or
numbered disambiguation suffixes you have not learned. Use you or a visible
description for a stranger. In an orientation exchange, ask what to call an
unfamiliar partner and answer with your own name; do not ask again after learning it.
Other concerns may take precedence outside these introductory conversations.
The engine records your own spoken self-introduction and exact name form; do not
add self-introduction bookkeeping. Where the schema includes introductions, use
it only to identify another person you already know, with their supplied actor_id,
exact learned name and observable reference. Direct replies to the player can
state your own name without an NPC learning effect.
For another person you must already know that form, and explicitly identify them
using one of their supplied observable references in the spoken introduction. A mere
mention, quotation or name in a record does not identify a visible stranger.
Only the listener learns after that turn; earlier speech cannot use that knowledge.
The snake explicitly knows residents' authored names in advance as a character
rule. Residents still learn the snake's name through a supported introduction.
"""


def actor_id(person):
    return person.get("npc_id", person.get("actor_id"))


def own_name(person):
    return person.get("name_knowledge", {}).get("own_name", person.get("character", {}).get("Name", person.get("name", person.get("npc_name", ""))))


def forms(name):
    """A learned full form permits its first name, never the reverse."""
    return {name, *name.split()} if name else set()


def actor_catalog(world):
    people = [*world.npcs, *world.archived]
    catalog = {p.id: {"name": p.name, "self_name": p.character["Name"]} for p in people}
    catalog[world.snake.id] = {"name": SNAKE_NAME, "self_name": SNAKE_NAME}
    # Only unique, observable descriptions can bind a third-party introduction.
    # Ties deliberately provide no reference; position alone never teaches a name.
    selectors = (("the person furthest to the left", lambda p: p.x),
                 ("the person furthest to the right", lambda p: -p.x),
                 ("the person closest to the apple tree",
                  lambda p: (p.x - world.tree.x) ** 2 + (p.y - world.tree.y) ** 2))
    for label, distance in selectors:
        ordered = sorted(world.npcs, key=distance)
        if ordered and (len(ordered) == 1 or distance(ordered[0]) < distance(ordered[1])):
            catalog[ordered[0].id].setdefault("references", []).append(label)
    catalog[world.snake.id]["references"] = ["the python guarding the apple tree"]
    return catalog


def name_context(npc):
    return {"own_name": npc.character["Name"], "known_names": deepcopy(npc.known_names)}


def name_forms(catalog):
    return sorted({form for person in catalog.values()
                   for name in (person["name"], person["self_name"])
                   for form in forms(name)}, key=lambda s: (-len(s), s))


def account_references(catalog):
    return {actor["name"]: actor.get("references", []) for actor in catalog.values()}


def account_speech_context(value, speaker, listener, catalog):
    """Keep factual account references speakable without granting name knowledge."""
    allowed = permitted_forms(speaker, catalog, speaker.get("name_knowledge", {}).get("known_names", {}))
    replacements = {}
    for target_id, actor in catalog.items():
        reference = ("you" if isinstance(listener, dict) and target_id == actor_id(listener) else
                     next(iter(actor.get("references", [])), None) if actor["self_name"] not in allowed else None)
        if target_id == actor_id(speaker) or reference is None:
            continue
        for name in {actor["name"], actor["self_name"]}:
            if name and sum(name in (entry["name"], entry["self_name"]) for entry in catalog.values()) == 1:
                replacements[name.casefold()] = reference
    pattern = "|".join(name_pattern(name) for name in sorted(replacements, key=len, reverse=True))

    def project(item):
        if isinstance(item, list):
            return [project(entry) for entry in item]
        if not isinstance(item, dict):
            return item
        result = {key: project(entry) for key, entry in item.items()}
        if pattern and isinstance(item.get("assertion"), str) and item.get("kind") not in ("former_life", "private_motive"):
            def reference(match):
                target = replacements[match[1].casefold()]
                return "your" if target == "you" and match[2] else target + ("'s" if match[2] else "")
            result["assertion"] = re.sub(rf"({pattern})(['’]s)?", reference, item["assertion"], flags=re.I)
        return result
    return project(value)


def _participants(snapshot):
    if snapshot.get("request_kind") in ("conversation",):
        return {role: snapshot[role] for role in ("initiator", "recipient")}
    if snapshot.get("request_kind") in ("snake_encounter", "snake_decision"):
        return {"initiator": snapshot["snake"], "recipient": snapshot["resident"]}
    if "speaker" in snapshot:
        role = snapshot.get("speaker_role", "initiator")
        return {role: snapshot["speaker"]}
    return {"initiator": snapshot}


def permitted_forms(person, catalog, knowledge):
    permitted = forms(own_name(person))
    if person.get("kind") == "python" or person.get("name_rule") == "knows_resident_names":
        for actor in catalog.values():
            permitted.update(forms(actor["self_name"]))
    for entry in knowledge.values():
        permitted.update(forms(entry["name"]))
    return permitted


def listener_address(person, listener, knowledge):
    if person.get("kind") == "python" or person.get("name_rule") == "knows_resident_names":
        return own_name(listener)
    learned = knowledge.get(actor_id(listener))
    if learned:
        return learned["name"]
    return "the python guarding the apple tree" if listener.get("name_rule") == "knows_resident_names" else "the other resident"


def validate_name_use(text, permitted, catalog):
    """Reject identifiable unknown references, leaving ambiguous ordinary words alone."""
    speech = unquoted_speech(text)
    candidates = name_forms(catalog)
    if not candidates:
        return
    pattern = "|".join(name_pattern(name) for name in candidates)
    for match in re.finditer(pattern, speech, re.I):
        used = match.group()
        if used.casefold() in {name.casefold() for name in permitted}:
            continue
        # Ordinary words and ambiguous shared first names are insufficient to
        # identify an actor. They can never create knowledge without an effect.
        if " " not in used and sum(p["self_name"].split()[0].casefold() == used.casefold()
                                  for p in catalog.values()) > 1:
            continue
        if not is_name_reference(match, speech):
            continue
        # Shared first names do not establish *which* actor bears the name.
        # If the form is permitted for one actor, it was already allowed above.
        raise ValueError(f"The speaker has not learned the name form {used!r}. Use you or an observable description; metadata does not teach names.")


def introduction_supported(text, name, reference, *, name_requested=False):
    speech = unquoted_speech(text)
    escaped = name_pattern(name)
    if reference == "self" and name_requested:
        first = next(iter(split_sentences(speech)), "").strip()
        if re.fullmatch(escaped + r"[.!]?", first, re.I):
            return True  # A direct, exact name answer is identifying speech too.
    if reference == "self":
        pattern = (r"\b(?:my name is|my name['’]s|name['’]s|i am|i['’]m|call me|you can call me|i go by|i['’]m called)\s+"
                   + escaped + r"|\bi,\s*" + escaped + r"\s*,|(?:^|[.!?]\s*|,\s*)"
                   + escaped + r"\s+here(?=\s*[,!.]|$)|\bmy name\b[^.!?;]{0,45}\bit['’]s\s+" + escaped)
    else:
        ref = re.escape(reference)
        pattern = (r"\b" + ref + r"\s+(?:is|is called|goes by|is named)\s+" + escaped
                   + "|" + escaped + r"\s+is\s+" + ref + r"\b")
    for match in re.finditer(pattern, speech, re.I):
        prefix = re.split(r"[.!?;]", speech[:match.start()])[-1]
        if (re.search(r"\b(?:if|suppose|pretend|imagine|whether|you think)\b", prefix, re.I)
                or re.match(r"\s*\?", speech[match.end():])):
            continue
        return True
    return False


def _name_requested(turns, index, snapshot, people):
    turn = turns[index]
    if index:
        previous = turns[index - 1]
        return previous.speaker != turn.speaker and asks_name(previous.text, self_name=own_name(people[previous.speaker]))
    preceding = snapshot.get("_preceding_turns", snapshot.get("preceding_turns", []))
    if not preceding or preceding[-1]["role"] == turn.speaker:
        return False
    listener = snapshot.get("listener", {})
    name = snapshot.get("_listener_identity", own_name(listener) if isinstance(listener, dict) else listener)
    return asks_name(preceding[-1]["text"], self_name=name)


def record_self_introductions(turns, snapshot):
    """Derive unambiguous self effects from generated words, never invent speech."""
    from llm import NameIntroduction
    people = _participants(snapshot)
    for index, turn in enumerate(turns):
        person = people[turn.speaker]
        if not actor_id(person) or any(intro.reference == "self" for intro in turn.introductions):
            continue
        introduced = next((name for name in sorted(forms(own_name(person)), key=len, reverse=True)
                           if introduction_supported(turn.text, name, "self",
                                                     name_requested=_name_requested(turns, index, snapshot, people))), None)
        if introduced:
            turn.introductions.append(NameIntroduction(actor_id=actor_id(person), name=introduced, reference="self"))


def validate_names(turns, snapshot, *, check_use=True, require_effects=True):
    """Return temporary knowledge and per-turn permissions; never mutate live state."""
    people = _participants(snapshot)
    check_dialogue = snapshot.get("_dialogue_validation", True)
    witnessed = snapshot.get("_name_witnesses", {})
    for hearers in witnessed.values():
        for person in hearers:
            if not any(actor_id(known) == actor_id(person) for known in people.values()):
                people["witness:" + actor_id(person)] = person
    catalog = snapshot.get("_name_actors", {})
    knowledge = {role: deepcopy(person.get("name_knowledge", {}).get("known_names", {}))
                 for role, person in people.items()}
    permissions, changes = [], []
    for index, turn in enumerate(turns):
        person = people[turn.speaker]
        requested = _name_requested(turns, index, snapshot, people)
        permitted = permitted_forms(person, catalog, knowledge[turn.speaker])
        permissions.append(sorted(permitted))
        if check_use and check_dialogue:
            validate_name_use(turn.text, permitted, catalog)
        seen = set()
        for intro in turn.introductions:
            target = catalog.get(intro.actor_id)
            error = ""
            if target is None or intro.actor_id in seen:
                error = "An introduction must identify one supplied actor exactly once per turn."
            elif intro.name not in forms(target["self_name"]):
                error = "An introduction must use the actor's authored name form, without a hidden surname or suffix."
            elif intro.reference == "self" and intro.actor_id != actor_id(person):
                error = "A self-introduction must identify the current speaker."
            elif intro.reference != "self" and (intro.reference not in target.get("references", [])
                    or check_dialogue and intro.name.casefold() not in {s.casefold() for s in permitted}):
                error = "A third-party introduction needs an informed speaker and the actor's supplied observable reference."
            if error:
                if check_dialogue:
                    raise ValueError(error)
                # Retain unchecked speech/metadata, but an unresolvable name effect
                # cannot teach a phantom actor or overwrite another actor's name.
                continue
            seen.add(intro.actor_id)
            if check_dialogue and not introduction_supported(turn.text, intro.name, intro.reference, name_requested=requested):
                raise ValueError("The introduction effect is unsupported by actual unquoted identifying speech.")
            if (check_dialogue and intro.name != target["self_name"]
                    and introduction_supported(turn.text, target["self_name"], intro.reference, name_requested=requested)):
                raise ValueError("Record the exact name form actually introduced, including a spoken surname.")
            for role, hearer in people.items():
                if role == turn.speaker or actor_id(hearer) == intro.actor_id:
                    continue
                if role.startswith("witness:") and actor_id(hearer) not in {
                        actor_id(w) for w in witnessed.get(index, [])}:
                    continue
                current = knowledge[role].get(intro.actor_id)
                if current and intro.name in forms(current["name"]):
                    continue
                entry = {"name": intro.name, "conversation_id": snapshot.get("interaction_id"), "turn": index}
                knowledge[role][intro.actor_id] = entry
                changes.append((actor_id(hearer), intro.actor_id, entry))
        if check_dialogue and check_use and require_effects and catalog:
            spoken_intros = [name for name in sorted(forms(own_name(person)), key=len, reverse=True)
                             if introduction_supported(turn.text, name, "self", name_requested=requested)]
            if spoken_intros and not any(intro.actor_id == actor_id(person) and intro.name == spoken_intros[0]
                                        and intro.reference == "self" for intro in turn.introductions):
                raise ValueError("A spoken self-introduction needs its matching structured introduction effect.")
    return knowledge, permissions, changes


def commit_names(interaction, changes):
    """Called once, only after live response validation and other effect checks."""
    for hearer_id, target_id, entry in changes:
        hearer = interaction.initiator.world.actor_by_id(hearer_id)
        if hasattr(hearer, "known_names"):
            hearer.known_names[target_id] = entry.copy()
    interaction.record.name_forms = name_forms(actor_catalog(interaction.initiator.world))


def snake_name_witnesses(snapshot, decision):
    """Only the opening claim and a fulfilled public endorsement record hearing."""
    heard = {}
    if snapshot.get("first_meeting"):
        heard[0] = snapshot.get("_name_snake_audience", [])
    if decision == "endorsement":
        heard[1] = snapshot.get("_name_audience", [])
    return heard


def orientation_context(npc):
    world = npc.world
    from constants import ARRIVAL_DISCUSSION_THRESHOLD
    discussed = {record.id for record in world.history if npc.name in record.participants
                 and record.orientation_discussed and record.turns
                 and orientation_speech(text for _, text in record.turns)}
    return {"days_here": world.day - npc.arrival_day,
            "orientation_discussions": len(discussed),
            "needs_orientation": len(discussed) < ARRIVAL_DISCUSSION_THRESHOLD,
            "adjustment": "finding bearings" if not discussed else "developing routines; questions remain unanswered"}


def orientation_speech(texts):
    pattern = (r"\b(?:arriv\w*|where (?:are|am) (?:we|i)|where (?:we|i) (?:are|am)|"
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
                raise ValueError("This is an orientation exchange. Discuss the unfamiliar surroundings or missing arrival memory, rather than pursuing a routine apple goal.")
            knowledge, _, _ = validate_names(response.turns, snapshot)
            for role, other in (("initiator", "recipient"), ("recipient", "initiator")):
                other_id = actor_id(snapshot[other])
                if other_id not in knowledge[role] and not any(
                        turn.speaker == role and asks_name(turn.text) for turn in response.turns):
                    raise ValueError(f"The {role} still does not know the partner's name. Ask what to call them and give them a chance to answer; introduce only names actually spoken.")
    # This is derived bookkeeping, not a reason to regenerate otherwise valid speech.
    response.orientation_discussed = discussed
    return discussed


def asks_name(text, *, self_name=None):
    speech = unquoted_speech(text)
    if re.search(r"\b(?:your (?:full |first |last |family |preferred )?name|call you(?:rself)?|address you|who are you)\b[^.!?]*\?", speech, re.I):
        return True
    reciprocal = re.search(r"\b(?:and you|(?:and )?yours)\s*\?", speech, re.I)
    preceding = next(iter(reversed(split_sentences(speech[:reciprocal.start()]))), "") if reciprocal else ""
    return bool(reciprocal and (self_name is None or any(
        introduction_supported(preceding, name, "self", name_requested=True) for name in forms(self_name))))


def declines_name(text):
    """An explicit refusal/unavailability answers a name question without teaching a name."""
    speech = unquoted_speech(text).replace("\u2019", "'")
    return bool(re.search(
        r"\b(?:rather|prefer|want|wish)\s+(?:to\s+)?(?:keep (?:(?:my |the )?name|mine) (?:private|to myself)|not (?:to )?(?:say|share|tell|give|answer))\b|"
        r"\b(?:won't|will not|don't want to|do not want to|not ready to)\s+(?:share|tell|give|say)\b[^.!?]{0,35}\b(?:name|call)\b|"
        r"\b(?:my name is private|keep my name (?:private|to myself)|decline to (?:give|share|answer)|"
        r"(?:can't|cannot|don't|do not) remember my name)\b", speech, re.I))
