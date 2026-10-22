"""Public name tags and private, speaker-owned conversation context."""

from copy import deepcopy
import re

from constants import PLAYER_NAME_TAG, SNAKE_NAME
from dialogue_rules import is_name_reference, name_pattern, unquoted_speech
from dialogue_text import split_sentences


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
    # This is derived bookkeeping, not a reason to regenerate otherwise valid speech.
    response.orientation_discussed = discussed
    return discussed
