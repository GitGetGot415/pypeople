"""Player action identities and availability, shared by both input routes."""

import re
from actor_identity import speech_only
from dialogue import ActionIntent


COMMAND_LABELS = {
    "GO_TO_TREE": "Go to the tree", "TALK_TO_NPC": "Talk to someone",
    "GIVE_ITEM": "Give an item", "EAT": "Eat an apple",
}


class Unavailable(ValueError):
    """The request cannot start now but may later: the person is busy or somewhere else. It is queued, not refused."""


def normalize_action(intent, snapshot, *, verbal=False):
    """Resolve a verbal label or structured ID once, before generating a reply."""
    if speech_only(snapshot):
        return ActionIntent(action="SAY")
    result = ActionIntent.model_validate(intent.model_dump(include=set(ActionIntent.model_fields)) if isinstance(intent, ActionIntent) else intent)
    if result.action not in ("TALK_TO_NPC", "GIVE_ITEM"):
        result.target = ""
    else:
        candidates = [*snapshot.get("other_npcs", []), *snapshot.get("remote_npcs", [])]
        exact = [p for p in candidates if p["actor_id"] == result.target]
        matches = exact or [p for p in candidates if p["name"].casefold() == result.target.strip().casefold()]
        if len(matches) != 1:
            raise ValueError("Choose one specific recipient; that reference is missing or ambiguous.")
        target = matches[0]
        if verbal and target.get("kind") != "player":
            text = snapshot.get("_player_message", "").casefold()
            base = re.sub(r"\s+\(\d+\)$", "", target["name"]).casefold()
            first = base.split()[0]
            references = [p for p in candidates if p.get("kind") != "player"
                          and p["name"].split()[0].casefold() == first]
            numbered = [p for p in references if re.sub(r"\s+\(\d+\)$", "", p["name"]).casefold() == base]
            specific = [p for p in references if p["name"].casefold() in text]
            exact_numbered = [p for p in specific if re.search(r"\(\d+\)$", p["name"])]
            if (len(numbered) > 1 and base in text and not exact_numbered
                    or len(references) > 1 and re.search(r"\b" + re.escape(first) + r"\b", text) and not specific):
                raise ValueError("That name identifies more than one resident. Ask which person is intended.")
        if result.action == "TALK_TO_NPC" and target.get("kind") == "player":
            return ActionIntent(action="SAY")
        result.target = target["actor_id"]
    if result.action != "GIVE_ITEM":
        result.item, result.quantity = "", 0
    else:
        works = {w["id"] for w in snapshot.get("owned_works", [])}
        if result.quantity < 1 or result.item != "fruit" and (result.item not in works or result.quantity != 1):
            raise ValueError("Choose carried apples or one specific owned work.")
    return result


def validate_action(world, npc, intent, *, accepting=False):
    """Check current world authority before acceptance and again before execution."""
    from locations import available, colocated
    from player import directed_recipient_available
    from autonomy import arrival_greeting_reserved
    if (world.npc_by_id(npc.id) is not npc or not available(npc) or npc.busy
            or arrival_greeting_reserved(npc, world)):
        raise ValueError("The resident is currently unavailable.")
    if intent.action == "EAT" and npc.inventory.get("fruit", 0) < 1:
        raise ValueError("There are no carried apples to eat.")
    if intent.action in ("TALK_TO_NPC", "GIVE_ITEM"):
        target = world.participant_by_id(intent.target)
        if target is None or target is npc:
            raise ValueError("Choose another person.")
        away = not colocated(npc, target)
        if target is world.player:
            if intent.action != "GIVE_ITEM":
                raise ValueError("Choose another resident to talk to.")
            if away:
                raise ValueError("Choose another person in the same location.")
        busy = target is not world.player and (not available(target) or not directed_recipient_available(target)
                                               or arrival_greeting_reserved(target, world))
        if intent.action == "GIVE_ITEM" and (away or busy):
            # A request that has to wait still needs something to give, so it cannot be queued for nothing.
            if intent.item == "fruit":
                holds = type(intent.quantity) is int and npc.inventory.get("fruit", 0) >= intent.quantity >= 1
            else:
                holds = intent.quantity == 1 and any(w.id == intent.item and w.owner_id == npc.id for w in world.creations)
            if not holds:
                raise ValueError(f"{npc.name} does not have that to give.")
        if away:
            raise Unavailable(f"{target.name} is not here; {npc.name} will go find them.")
        if busy:
            raise Unavailable(f"{target.name} is busy; {npc.name} will wait until they are free.")
        if intent.action == "GIVE_ITEM":
            npc.check_gift(target, intent.item, intent.quantity,
                player_authorization=world.player_interaction.id if accepting and world.player_interaction else None)


def action_description(intent, snapshot):
    label = COMMAND_LABELS[intent.action]
    target = next((p["name"] for p in [*snapshot.get("other_npcs", []), *snapshot.get("remote_npcs", [])]
                   if p["actor_id"] == intent.target), "")
    if intent.action == "TALK_TO_NPC":
        return f"{label}: {target}"
    if intent.action == "GIVE_ITEM":
        title = next((w["title"] for w in snapshot.get("owned_works", []) if w["id"] == intent.item), "apples")
        return f"{label}: {intent.quantity} × {title} to {target}"
    return label


def command_speech(intent, snapshot):
    """What picking an action amounts to saying, shown as the player's own line in the conversation."""
    if intent.action == "GO_TO_TREE":
        return "Go to the tree."
    if intent.action == "EAT":
        return "Eat an apple."
    target = next((p for p in [*snapshot.get("other_npcs", []), *snapshot.get("remote_npcs", [])]
                   if p["actor_id"] == intent.target), None)
    name = "me" if target is None or target.get("kind") == "player" else target["name"]
    if intent.action == "TALK_TO_NPC":
        return f"Go talk to {name}."
    title = next((w["title"] for w in snapshot.get("owned_works", []) if w["id"] == intent.item), None)
    item = (f'your work "{title}"' if title else "an apple" if intent.quantity == 1
            else f"{intent.quantity} apples")
    return f"Give {name} {item}."


def command_options(world, npc, action=None, target=None):
    """Return labelled requests and live reasons for the compact picker."""
    if action is None:
        options = []
        for value, label in COMMAND_LABELS.items():
            if value in ("TALK_TO_NPC", "GIVE_ITEM"):
                choices = command_options(world, npc, value)
                error = "" if any(not reason for _, _, reason in choices) else "No eligible recipients or items."
            else:
                try:
                    validate_action(world, npc, ActionIntent(action=value), accepting=True)
                    error = ""
                except ValueError as exc:
                    error = str(exc)
            options.append((label, dict(action=value), error))
        return options
    if target is None:
        people = [p for p in world.npcs if p is not npc]
        if action == "GIVE_ITEM":
            people.append(world.player)
        options = []
        for person in people:
            intent = ActionIntent(action=action, target=person.id)
            try:
                if action == "GIVE_ITEM":
                    choices = command_options(world, npc, action, person.id)
                    if not any(not reason for _, _, reason in choices):
                        continue
                else:
                    validate_action(world, npc, intent, accepting=True)
            except Unavailable:
                pass  # Busy or elsewhere: the request is queued.
            except ValueError:
                continue
            options.append(("You" if person is world.player else person.name, intent.model_dump(), ""))
        return options
    possessions = [("Carried apples", "fruit")]
    works = world.owned_works(npc.id)
    possessions.extend((work.title + (f" ({index + 1})" if sum(w.title == work.title for w in works) > 1 else ""), work.id)
                       for index, work in enumerate(works))
    options = []
    for label, item in possessions:
        intent = ActionIntent(action=action, target=target, item=item, quantity=1)
        try:
            validate_action(world, npc, intent, accepting=True)
        except Unavailable:
            pass
        except ValueError:
            continue
        options.append((label, intent.model_dump(), ""))
    return options
