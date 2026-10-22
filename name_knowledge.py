"""Small, directional name knowledge; engine identities never grant familiarity."""

from copy import deepcopy
import re

from validation_errors import DialogueValidationError, RecoveryKind
from constants import SNAKE_NAME
from dialogue_rules import is_name_reference, name_pattern, unquoted_speech, validate_speaker_identity
from dialogue_text import split_sentences


NAME_INSTRUCTION = """Engine names, actor IDs, roster labels, goals, opinions, memories,
accounts, promises and correction metadata identify actors for bookkeeping only.
They never teach a resident a name. name_knowledge is the authority for spoken
names: use only your own authored name and learned forms, without surnames or
numbered disambiguation suffixes you have not learned. Use you or a visible
description for a stranger. In an orientation exchange, ask what to call an
unfamiliar partner and answer with your own name; do not ask again after learning it.
Other concerns may take precedence outside these introductory conversations.
Hearing someone merely mention a name does not identify that actor to you.
To ask who was meant, put the mentioned name in an explicit quotation; otherwise
use a supplied observable reference. In a reply to the player, an identity
clarification question may repeat the exact name they mentioned without quotes.
Do not adopt the name as a known identity or infer its bearer from the roster.
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
rule. Residents learn the snake's name through a supported introduction, or the
authored prior-day meeting supplied in a new player game's starting knowledge.
The player's personal name is unknown until they introduce themselves to you.
Use your learned player address when known; otherwise use you or ask their name.
The interface labels You and Player never establish a personal name.
"""


def actor_id(person):
    return person.get("npc_id", person.get("actor_id"))


def own_name(person):
    return person.get("name_knowledge", {}).get("own_name", person.get("character", {}).get("Name", person.get("name", person.get("npc_name", ""))))


def forms(name):
    """A learned full form permits its first name, never the reverse."""
    return {name, *name.split()} if name else set()


def actor_catalog(world, location="garden"):
    people = [*world.npcs, *world.archived]
    visible = [p for p in world.npcs if p.location == location]
    visible_ids = {p.id for p in visible}
    catalog = {p.id: {"name": p.name, "self_name": p.character["Name"], "_visible": p.id in visible_ids} for p in people}
    catalog[world.snake.id] = {"name": SNAKE_NAME, "self_name": SNAKE_NAME, "_visible": location == "garden"}
    # Only unique, observable descriptions can bind a third-party introduction.
    # Ties deliberately provide no reference; position alone never teaches a name.
    selectors = (("the person furthest to the left", lambda p: p.x),
                 ("the person furthest to the right", lambda p: -p.x),
                 ("the person closest to the apple tree",
                  lambda p: (p.x - world.tree.x) ** 2 + (p.y - world.tree.y) ** 2))
    for label, distance in (selectors if location == "garden" else selectors[:2]):
        ordered = sorted(visible, key=distance)
        if ordered and (len(ordered) == 1 or distance(ordered[0]) < distance(ordered[1])):
            catalog[ordered[0].id].setdefault("references", []).append(label)
    # A larger roster has many people who are neither an extreme nor nearest
    # the tree. Give those actors a unique visible reference too; their names
    # cannot be spoken merely because an account or goal mentions them.
    for coordinate, direction in (("x", "left"), ("y", "top")):
        ordered = sorted(visible, key=lambda p: getattr(p, coordinate))
        for rank, person in enumerate(ordered, 1):
            if catalog[person.id].get("references") or sum(
                    getattr(p, coordinate) == getattr(person, coordinate) for p in ordered) != 1:
                continue
            suffix = "th" if 10 <= rank % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
            catalog[person.id]["references"] = [f"the {rank}{suffix} person from the {direction}"]
    catalog[world.snake.id]["references"] = ["the python guarding the apple tree"]
    player_visible = world.mode == "player" and world.player.alive and world.player.location == location
    catalog[world.player.id] = {"_visible": player_visible, "name": "Player", "self_name": "Player", "kind": "player",
                               "references": ["the player"] if player_visible else [],
                               "_aliases": list({p.known_names[world.player.id]["name"] for p in [*people, world.snake] if world.player.id in p.known_names})}
    return catalog


def name_context(npc):
    return {"own_name": npc.character["Name"], "known_names": deepcopy(npc.known_names)}


def name_forms(catalog):
    return sorted({form for person in catalog.values()
                   for name in (person.get("_aliases", ()) if person.get("kind") == "player" else (person["name"], person["self_name"]))
                   for form in forms(name)}, key=lambda s: (-len(s), s))


def account_references(catalog):
    return {actor["name"]: actor.get("references", []) for actor in catalog.values()}


def account_speech_context(value, speaker, listener, catalog):
    """Project writing context into known references; leave accepted speech intact."""
    knowledge = speaker.get("name_knowledge", {}).get("known_names", {})
    allowed = permitted_forms(speaker, catalog, knowledge)
    replacements, addresses = {}, {}
    for target_id, actor in catalog.items():
        if target_id == actor_id(speaker):
            addresses[actor["name"].casefold()] = own_name(speaker)
            continue
        learned = knowledge.get(target_id, {}).get("name")
        address = (actor["self_name"] if actor["self_name"] in allowed else learned or
                   next(iter(actor.get("references", [])), "another resident"))
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
            if key in ("name_knowledge", "preceding_turns", "recent_dialogue"):
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


def mentioned_names(text, catalog):
    """Return only name forms actually heard, without binding them to actors."""
    candidates = name_forms(catalog)
    if not candidates:
        return set()
    pattern = "|".join(name_pattern(name) for name in candidates)
    return {match.group() for match in re.finditer(pattern, text, re.I)
            if is_name_reference(match, text)}


def player_heard_names(snapshot):
    texts = [snapshot.get("_player_message", "")]
    for turn in snapshot.get("player_dialogue", []):
        speaker, text = (turn["speaker"], turn["text"]) if isinstance(turn, dict) else turn
        if speaker == "You":
            texts.append(text)
    return mentioned_names("\n".join(texts), snapshot.get("_name_actors", {}))


def unidentified_player_names(snapshot):
    catalog = snapshot.get("_name_actors", {})
    allowed = permitted_forms(snapshot, catalog, snapshot.get("name_knowledge", {}).get("known_names", {}))
    allowed = {name.casefold() for name in allowed}
    return {name for name in mentioned_names(snapshot.get("_player_message", ""), catalog)
            if name.casefold() not in allowed}


def _identity_question(speech, match):
    """A heard label can be questioned without claiming to know its bearer."""
    start = max((speech.rfind(mark, 0, match.start()) for mark in ".!?\uFF1F\u061F"), default=-1) + 1
    end = re.search(r"[.!?\uFF1F\u061F]", speech[match.end():])
    if end is None:
        return False
    sentence = speech[start:match.end() + end.end()].strip()
    name = re.escape(match.group())
    return bool(re.fullmatch(
        rf"(?:who(?: is| was|['’]s) {name}|who do you mean by {name}|which {name} do you mean|"
        rf"(?:can|could|would) you (?:please )?tell me (?:more about )?who {name} (?:is|was)"
        rf"(?: or what (?:they|he|she)(?:['’]s| is) like)?|{name})\s*[?\uFF1F\u061F]",
        sentence, re.I))


def _unknown_name_statement(speech, match):
    """Explicit lack of recognition can repeat a heard label without identifying it."""
    start = max((speech.rfind(mark, 0, match.start()) for mark in ".!?\uFF1F\u061F"), default=-1) + 1
    end = re.search(r"[.!?\uFF1F\u061F]", speech[match.end():])
    sentence = speech[start:match.end() + end.end() if end else len(speech)].strip()
    sentence = re.sub(r"^i(?:['’]m| am) (?:a bit )?(?:confused|unsure)(?: though)?,\s*", "", sentence, flags=re.I)
    name = re.escape(match.group())
    return bool(re.fullmatch(
        rf"(?:i(?:['’]ve| have) never heard of {name}(?: before)?|"
        rf"i (?:don['’]t|do not) know (?:who {name} is|anyone (?:called|named) {name}))\s*[.!?]?",
        sentence, re.I))


def validate_name_use(text, permitted, catalog, *, heard=()):
    """Reject identifiable unknown references, leaving ambiguous ordinary words alone."""
    speech = unquoted_speech(text)
    candidates = name_forms(catalog)
    if not candidates:
        return
    heard = {name.casefold() for name in heard}
    pattern = "|".join(name_pattern(name) for name in candidates)
    for match in re.finditer(pattern, speech, re.I):
        used = match.group()
        if used.casefold() in {name.casefold() for name in permitted}:
            continue
        if used.casefold() in heard and (_identity_question(speech, match) or _unknown_name_statement(speech, match)):
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
        raise DialogueValidationError(f"The speaker has not learned the name form {used!r}. Use you or an observable description; metadata does not teach names.", RecoveryKind.UNKNOWN_NAME)


def validate_player_reply_names(text, snapshot, *, person=None):
    """Use request-time hearing/knowledge with the current catalog at acceptance."""
    person = snapshot if person is None else person
    catalog = person.get("_name_actors", {})
    known = snapshot.get("name_knowledge", {}).get("known_names", {})
    address = known.get(snapshot.get("player_id"), {}).get("name")
    if snapshot.get("player_id"):
        validate_speaker_identity(text, own_name(snapshot), address or "You")
    if (player_asks_own_name(snapshot.get("_player_message", ""))
            and not (address and any(re.search(name_pattern(name), text, re.I) for name in forms(address)))
            and any(introduction_supported(text, name, "self") for name in forms(own_name(snapshot)))):
        raise ValueError("The player asked their own name, not yours. Use the name they actually introduced, or admit you do not know it and ask what to call them; do not substitute your own introduction.")
    validate_name_use(text, permitted_forms(person, catalog, known), catalog,
                      heard=player_heard_names(snapshot))


def player_asks_own_name(message):
    speech = unquoted_speech(message).replace("\u2019", "'")
    return bool(re.search(r"\b(?:what(?:'s| is| was) my (?:full |first |preferred )?name|"
                          r"(?:do you (?:know|remember)|can you (?:read|see)) my name|"
                          r"what am i called|who am i)\b", speech, re.I))


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
                   + escaped + r"\s+here(?=\s*[,!.]|$)|\bmy (?:own )?name\b[^.!?;]{0,45}\b(?:it is|it['’]s)\s+" + escaped)
        pattern += r"|(?:^|[.!?]\s*)as\s+" + escaped + r",\s*i\b"
        pattern += (r"|\bi(?:['’]d| would)? (?:prefer|like) to be called\s+" + escaped
                    + r"|\byou (?:can|may) call me\s+" + escaped)
        if name_requested or asks_name(speech):
            pattern += r"|\bmine(?: is|['’]s)\s+" + escaped
        if name_requested:
            pattern += r"|^\s*(?:it is|it['’]s)\s+" + escaped + r"(?=\s*(?:[,!.;]|$))"
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
            elif intro.name not in forms(target["self_name"]) and not (target.get("kind") == "player"
                    and intro.name in forms(knowledge[turn.speaker].get(intro.actor_id, {}).get("name", ""))):
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
                 and orientation_speech(turn.text for turn in record.turns)}
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
    if re.search(r"\b(?:your (?:full |first |last |family |preferred )?name|call you(?:rself)?|address you|who are you|"
                 r"(?:can|could|would|will) you (?:please )?introduce yourself|"
                 r"(?:what|how) (?:would|do) you (?:like|prefer) to be called|what do you go by)\b[^.!?]*\?", speech, re.I):
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
