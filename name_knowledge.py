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
Where the schema includes introductions, record actor_id, name (the exact form
spoken), and reference for each actual spoken introduction in that turn. Direct
replies to the player can state your own name without an NPC learning effect.
For yourself reference is self and
speech must actually introduce you (for example using my name is or call me).
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


def introduction_supported(text, name, reference):
    speech = unquoted_speech(text)
    escaped = name_pattern(name)
    if reference == "self":
        pattern = (r"\b(?:my name is|my name['’]s|name['’]s|i am|i['’]m|call me|you can call me|i go by|i['’]m called)\s+"
                   + escaped + r"|\bi,\s*" + escaped + r"\s*,")
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


def validate_names(turns, snapshot, *, check_use=True, require_effects=True):
    """Return temporary knowledge and per-turn permissions; never mutate live state."""
    people = _participants(snapshot)
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
        permitted = permitted_forms(person, catalog, knowledge[turn.speaker])
        permissions.append(sorted(permitted))
        if check_use:
            validate_name_use(turn.text, permitted, catalog)
        seen = set()
        for intro in turn.introductions:
            target = catalog.get(intro.actor_id)
            if target is None or intro.actor_id in seen:
                raise ValueError("An introduction must identify one supplied actor exactly once per turn.")
            seen.add(intro.actor_id)
            if intro.name not in forms(target["self_name"]):
                raise ValueError("An introduction must use the actor's authored name form, without a hidden surname or suffix.")
            if intro.reference == "self":
                if intro.actor_id != actor_id(person):
                    raise ValueError("A self-introduction must identify the current speaker.")
            elif (intro.reference not in target.get("references", [])
                  or intro.name.casefold() not in {s.casefold() for s in permitted}):
                raise ValueError("A third-party introduction needs an informed speaker and the actor's supplied observable reference.")
            if not introduction_supported(turn.text, intro.name, intro.reference):
                raise ValueError("The introduction effect is unsupported by actual unquoted identifying speech.")
            if (intro.name != target["self_name"]
                    and introduction_supported(turn.text, target["self_name"], intro.reference)):
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
        if check_use and require_effects and catalog:
            spoken_intros = [name for name in sorted(forms(own_name(person)), key=len, reverse=True)
                             if introduction_supported(turn.text, name, "self")]
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
    discussed = [record for record in world.history if npc.name in record.participants
                 and record.orientation_discussed and record.turns]
    return {"days_here": world.day - npc.arrival_day,
            "orientation_discussions": len(discussed),
            "needs_orientation": not discussed,
            "adjustment": "finding bearings" if not discussed else "developing routines; questions remain unanswered"}


def orientation_speech(texts):
    pattern = (r"\b(?:arriv\w*|where (?:are|am) (?:we|i)|how (?:did|do) (?:we|i) get|"
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
    if snapshot.get("purpose") == "orientation":
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
    if response.orientation_discussed and not discussed:
        raise ValueError("orientation_discussed requires actual speech about arrival, surroundings, help or safety.")
    return discussed


def asks_name(text):
    speech = unquoted_speech(text)
    return bool(re.search(r"\b(?:your name|call you(?:rself)?|address you|who are you)\b[^.!?]*\?|"
                          r"\b(?:and you|(?:and )?yours)\s*\?", speech, re.I))
