from constants import (
    GARDEN_LOCATION, GARDEN_FLOOR_SIZE, GARDEN_FLOOR_OFFSET, HALLWAY_SIZE, ROOM_SIZE, FOOT_MARGIN,
    INDOOR_FLOOR_TOP, ROOMS_PER_FLOOR, HALLWAY_DOOR_ORIGINS, HALLWAY_ELEVATOR_ORIGIN,
    APARTMENT_OVERLAY_SIZE, APARTMENT_OVERLAY_SCALE, HALLWAY_ARRIVAL_Y, ROOM_ARRIVAL, ELEVATOR_ARRIVAL,
    HALLWAY_EXIT, ROOM_EXIT, HALLWAY_ENTRY, HALLWAY_EXIT_RECT, ROOM_EXIT_RECT, GARDEN_BUILDING_POINT,
    GARDEN_BUILDING_ARRIVAL, GARDEN_BRIDGE_RECT, PLAYER_ROOM_NUMBER, APARTMENT_FLOORS, MAX_RESIDENTS,
)


def room_location(number):
    return f"room:{number}"


def hallway_location(floor):
    return f"hallway:{floor}"


def active_player(world):
    return world.mode == "player" and world.player.alive


def floor_for_room(number):
    return (number - 1) // ROOMS_PER_FLOOR + 1


def allocated_floors(world):
    return range(1, APARTMENT_FLOORS + 1)


def npc_capacity(world):
    return MAX_RESIDENTS - int(active_player(world))


def valid_location(world, location):
    if location == GARDEN_LOCATION:
        return True
    try:
        kind, number = location.split(":")
        number = int(number)
    except (ValueError, AttributeError):
        return False
    return (kind == "hallway" and number in allocated_floors(world)
            or kind == "room" and (number in {n.room_number for n in world.residents}
                                  or number == PLAYER_ROOM_NUMBER and active_player(world)))


def map_size(location):
    if location == GARDEN_LOCATION:
        return GARDEN_FLOOR_SIZE
    width, height = HALLWAY_SIZE if location.startswith("hallway:") else ROOM_SIZE
    exit_rect = HALLWAY_EXIT_RECT if location.startswith("hallway:") else ROOM_EXIT_RECT
    return width, max(height, exit_rect[1] + exit_rect[3])


def feet_bounds(location):
    # Exit strips attach beyond the floor; feet stay on the floor and touch
    # the strip at its edge rather than walking into the surrounding black.
    if location == GARDEN_LOCATION:
        left, top = GARDEN_FLOOR_OFFSET
        width, height = GARDEN_FLOOR_SIZE
        return left + FOOT_MARGIN, left + width - FOOT_MARGIN, GARDEN_BUILDING_POINT[1], top + height - FOOT_MARGIN
    width, height = HALLWAY_SIZE if location.startswith("hallway:") else ROOM_SIZE
    return FOOT_MARGIN, width - FOOT_MARGIN, INDOOR_FLOOR_TOP + FOOT_MARGIN, height - FOOT_MARGIN


def clamp_point(location, point):
    left, right, top, bottom = feet_bounds(location)
    if location == GARDEN_LOCATION:
        x, y, width, height = GARDEN_BRIDGE_RECT
        garden = (min(right, max(left, point[0])), min(bottom, max(FOOT_MARGIN, point[1])))
        bridge = (min(x + width - FOOT_MARGIN, max(x + FOOT_MARGIN, point[0])),
                  # Join the garden's inset walking area without extending the image.
                  min(max(y + height, FOOT_MARGIN), max(top, point[1])))
        return min((garden, bridge), key=lambda p: (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)
    return min(right, max(left, point[0])), min(bottom, max(top, point[1]))


def colocated(first, second):
    return first is not None and second is not None and first.location == second.location


def observers(world, *actors):
    """Living knowledge owners who can see every supplied actor in their actual area."""
    if not actors or any(not colocated(actors[0], actor) for actor in actors):
        return []
    return [actor for actor in [*world.npcs, world.snake]
            if actor.alive and colocated(actor, actors[0])]


def door_point(number):
    x, _ = HALLWAY_DOOR_ORIGINS[(number - 1) % ROOMS_PER_FLOOR]
    return x + APARTMENT_OVERLAY_SIZE[0] / 2, HALLWAY_ARRIVAL_Y


def overlay_rect(origin):
    """Rendered overlay bounds: horizontally centered and resting on the floor."""
    width, height = APARTMENT_OVERLAY_SIZE
    rendered_width = max(1, round(width * APARTMENT_OVERLAY_SCALE))
    rendered_height = max(1, round(height * APARTMENT_OVERLAY_SCALE))
    x, y = origin
    return (x + (width - rendered_width) // 2, y + height - rendered_height,
            rendered_width, rendered_height)


def targets(world, location):
    """UI targets include their floor standing point and destination identity."""
    if location == GARDEN_LOCATION:
        from constants import APARTMENT_DOOR_RECT
        return [dict(kind="building", rect=APARTMENT_DOOR_RECT, point=GARDEN_BUILDING_POINT,
                     destination=hallway_location(1))]
    kind, number = location.split(":")
    number = int(number)
    if kind == "room":
        return [dict(kind="exit_room", rect=ROOM_EXIT_RECT,
                     point=ROOM_EXIT, destination=hallway_location(floor_for_room(number)), room=number)]
    result = []
    for slot, (x, y) in enumerate(HALLWAY_DOOR_ORIGINS):
        room = (number - 1) * ROOMS_PER_FLOOR + slot + 1
        result.append(dict(kind="room", rect=overlay_rect((x, y)), point=door_point(room),
                           destination=room_location(room), room=room))
    result.append(dict(kind="elevator", rect=overlay_rect(HALLWAY_ELEVATOR_ORIGIN),
                       point=ELEVATOR_ARRIVAL, destination=hallway_location(2 if number == 1 else 1)))
    if number == 1:
        result.append(dict(kind="garden", rect=HALLWAY_EXIT_RECT,
                           point=HALLWAY_EXIT, destination=GARDEN_LOCATION))
    return result


def doorway_arrival(target):
    destination = target["destination"]
    return (ROOM_ARRIVAL if destination.startswith("room:") else
            ELEVATOR_ARRIVAL if target["kind"] == "elevator" else
            GARDEN_BUILDING_ARRIVAL if destination == GARDEN_LOCATION else
            door_point(target["room"]) if target["kind"] == "exit_room" else HALLWAY_ENTRY)


def environment(location):
    if location == GARDEN_LOCATION:
        return "An open grassy garden with its finite apple tree and a narrow wooden bridge across a gap to the apartment building entrance."
    if location.startswith("hallway:"):
        floor = location.split(":")[1]
        return (f"Inside the apartment building's hallway on floor {floor}, with five closed room doors and an elevator. "
                "The garden and its apple tree are outside the building; there is no grass or open sky here. "
                "People in this hallway can see visible activity throughout it; speech is private to its participants. "
                "The garden exit is on the first floor; other floors reach it by elevator.")
    number = location.split(":")[1]
    return (f"Inside narrow private apartment room {number}, with a bed at the top of the floor and an entrance and walk-back exit at the bottom. "
            "The exit leads to the hallway. The garden and apple tree are outside the building, with no grass or open sky here. "
            "People in this room can see its visible activity; speech is private to its participants.")


def validate_feet(location, value):
    import math
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        raise ValueError("Invalid feet position.")
    if tuple(value) != clamp_point(location, value):
        raise ValueError("Feet position is outside its map.")


def validate_destination(value, *, location=None):
    import math
    # Suspended activities have no location of their own. Their destination
    # must still belong to one of the game's physical maps.
    maps = (location,) if location is not None else (GARDEN_LOCATION, "hallway:1", "room:1")
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or not all(isinstance(coordinate, (int, float)) and math.isfinite(coordinate) for coordinate in value)
            or not any(tuple(value) == clamp_point(area, value) for area in maps)):
        raise ValueError(f"Invalid destination {value!r} for {location or 'the available maps'}.")
