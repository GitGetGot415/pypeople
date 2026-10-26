"""Actor identities and visible name tags, independent of prompt construction."""

from constants import PLAYER_NAME_TAG, SNAKE_NAME
from locations import active_player


def actor_id(person):
    return person.get("actor_id")


def own_name(person):
    return person.get("name", "")


def speech_only(person):
    """The snake's snapshots are marked kind "python": it speaks and shares accounts but never acts."""
    return person.get("kind") == "python"


def forms(name):
    return {name, *name.split()} if name else set()


def actor_catalog(world, location="garden"):
    people = [*world.npcs, *world.archived]
    live_ids = {p.id for p in world.npcs}
    catalog = {p.id: {"name": p.name, "self_name": p.character["Name"],
                      "_visible": p.id in live_ids and p.location == location} for p in people}
    catalog[world.snake.id] = {"name": SNAKE_NAME, "self_name": SNAKE_NAME,
                               "_visible": location == "garden", "references": ["the python wandering the garden"]}
    catalog[world.player.id] = {"name": PLAYER_NAME_TAG, "self_name": PLAYER_NAME_TAG, "kind": "player",
        "_visible": active_player(world) and world.player.location == location,
        "references": ["the player"]}
    return catalog


def name_forms(catalog):
    return sorted({form for actor in catalog.values() for name in (actor["name"], actor["self_name"])
                   for form in forms(name)}, key=lambda s: (-len(s), s))


def account_references(catalog):
    return {actor["name"]: actor.get("references", []) for actor in catalog.values()}


