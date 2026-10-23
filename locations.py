"""The garden and rectangular apartment maps; no scene or pathfinding framework."""

from constants import (GARDEN_LOCATION, WORLD_WIDTH, WORLD_HEIGHT, HALLWAY_SIZE, ROOM_SIZE, FOOT_MARGIN,
                      INDOOR_FLOOR_TOP, ROOMS_PER_FLOOR, HALLWAY_DOOR_ORIGINS, HALLWAY_ELEVATOR_ORIGIN,
                      APARTMENT_OVERLAY_SIZE, ROOM_DOOR_ORIGIN, HALLWAY_ARRIVAL_Y, ROOM_ARRIVAL,
                      ELEVATOR_ARRIVAL, HALLWAY_EXIT, GARDEN_BUILDING_POINT, GARDEN_BUILDING_ARRIVAL,
                      MOVEMENT_SPEED, TRAVEL_STEP_TIMEOUT, PLAYER_APPROACH_MULTIPLIER, PLAYER_ROOM_NUMBER)
from interactions import move_toward


def room_location(number):
    return f"room:{number}"


def hallway_location(floor):
    return f"hallway:{floor}"


def player_has_room(world):
    return world.mode == "player" or not world.player.alive


def floor_for_room(number):
    return (number - 1) // ROOMS_PER_FLOOR + 1


def allocated_floors(world):
    return range(1, floor_for_room(max((n.room_number for n in world.residents), default=1)) + 1)


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
                                  or number == PLAYER_ROOM_NUMBER and player_has_room(world)))


def map_size(location):
    return (WORLD_WIDTH, WORLD_HEIGHT) if location == GARDEN_LOCATION else HALLWAY_SIZE if location.startswith("hallway:") else ROOM_SIZE


def feet_bounds(location):
    width, height = map_size(location)
    return FOOT_MARGIN, width - FOOT_MARGIN, FOOT_MARGIN if location == GARDEN_LOCATION else INDOOR_FLOOR_TOP + FOOT_MARGIN, height - FOOT_MARGIN


def clamp_point(location, point):
    left, right, top, bottom = feet_bounds(location)
    return min(right, max(left, point[0])), min(bottom, max(top, point[1]))


def colocated(first, second):
    return first is not None and second is not None and first.location == second.location


def available(npc):
    from activity import Behavior
    return (npc.departure is None and npc.behavior not in (Behavior.SLEEP, Behavior.GO_HOME)
            and not npc.journey and npc.home_reason != "night")


def door_point(number):
    x, _ = HALLWAY_DOOR_ORIGINS[(number - 1) % ROOMS_PER_FLOOR]
    return x + APARTMENT_OVERLAY_SIZE[0] / 2, HALLWAY_ARRIVAL_Y


def targets(world, location):
    """UI targets include their floor standing point and destination identity."""
    if location == GARDEN_LOCATION:
        from constants import APARTMENT_RECT
        return [dict(kind="building", rect=APARTMENT_RECT, point=GARDEN_BUILDING_POINT,
                     destination=hallway_location(1))]
    kind, number = location.split(":")
    number = int(number)
    if kind == "room":
        return [dict(kind="exit_room", rect=(*ROOM_DOOR_ORIGIN, *APARTMENT_OVERLAY_SIZE),
                     point=ROOM_ARRIVAL, destination=hallway_location(floor_for_room(number)), room=number)]
    result = []
    for slot, (x, y) in enumerate(HALLWAY_DOOR_ORIGINS):
        room = (number - 1) * ROOMS_PER_FLOOR + slot + 1
        result.append(dict(kind="room", rect=(x, y, *APARTMENT_OVERLAY_SIZE), point=door_point(room),
                           destination=room_location(room), room=room))
    result.append(dict(kind="elevator", rect=(*HALLWAY_ELEVATOR_ORIGIN, *APARTMENT_OVERLAY_SIZE),
                       point=ELEVATOR_ARRIVAL))
    if number == 1:
        result.append(dict(kind="garden", rect=(0, INDOOR_FLOOR_TOP, FOOT_MARGIN * 3, HALLWAY_SIZE[1] - INDOOR_FLOOR_TOP),
                           point=HALLWAY_EXIT, destination=GARDEN_LOCATION))
    return result


def route(source, destination, point):
    """Use the same door and elevator connections for residents and player."""
    steps = []
    current = source
    if current.startswith("room:") and current != destination:
        room = int(current.split(":")[1])
        current = hallway_location(floor_for_room(room))
        steps.append(dict(point=ROOM_ARRIVAL, destination=current, arrival=door_point(room)))
    target_hall = (hallway_location(floor_for_room(int(destination.split(":")[1])))
                   if destination.startswith("room:") else destination)
    if current == GARDEN_LOCATION and target_hall != GARDEN_LOCATION:
        current = hallway_location(1)
        steps.append(dict(point=GARDEN_BUILDING_POINT, destination=current, arrival=HALLWAY_EXIT))
    if current.startswith("hallway:") and target_hall == GARDEN_LOCATION:
        if current != hallway_location(1):
            steps.append(dict(point=ELEVATOR_ARRIVAL, destination=hallway_location(1), arrival=ELEVATOR_ARRIVAL))
        current = GARDEN_LOCATION
        steps.append(dict(point=HALLWAY_EXIT, destination=current, arrival=GARDEN_BUILDING_ARRIVAL))
    elif current.startswith("hallway:") and current != target_hall:
        steps.append(dict(point=ELEVATOR_ARRIVAL, destination=target_hall, arrival=ELEVATOR_ARRIVAL))
        current = target_hall
    if destination.startswith("room:") and current != destination:
        room = int(destination.split(":")[1])
        steps.append(dict(point=door_point(room), destination=destination, arrival=ROOM_ARRIVAL))
    steps.append(dict(point=clamp_point(destination, point), destination=destination, arrival=None))
    return steps


def start_journey(actor, destination, point, *, purpose="travel", fast=False):
    if not valid_location(actor.world, destination):
        raise ValueError("That destination is unavailable.")
    actor.destination = None
    actor.journey = dict(destination=destination, point=tuple(point), purpose=purpose, fast=fast,
                         elapsed=0.0, steps=route(actor.location, destination, point))


def update_journey(actor, dt):
    journey = actor.journey
    if journey is None:
        return False
    journey["elapsed"] += dt
    if journey["elapsed"] > TRAVEL_STEP_TIMEOUT:
        actor.journey = None
        actor.world.publish("travel_failed", (actor,), "The journey could not reach its destination.")
        if actor is actor.world.player:
            from player import end_interaction
            end_interaction(actor.world, "The journey could not reach its destination.")
        return False
    step = journey["steps"][0]
    speed = MOVEMENT_SPEED * (PLAYER_APPROACH_MULTIPLIER if journey["fast"] else 1)
    if move_toward(actor, clamp_point(actor.location, step["point"]), dt, speed):
        if step["arrival"] is not None:
            actor.location = step["destination"]
            actor.x, actor.y = clamp_point(actor.location, step["arrival"])
        journey["steps"].pop(0)
        journey["elapsed"] = 0.0
        if not journey["steps"]:
            actor.journey = None
            return True
    return False


def environment(location):
    if location == GARDEN_LOCATION:
        return "An open grassy garden with its guarded apple tree and apartment building entrance."
    if location.startswith("hallway:"):
        floor = location.split(":")[1]
        return (f"Inside the apartment building's hallway on floor {floor}, with five closed room doors and an elevator. "
                "The garden and its apple tree are outside the building; there is no grass or open sky here. "
                "Only people in this hallway can hear. The garden exit is on the first floor; other floors reach it by elevator.")
    number = location.split(":")[1]
    return (f"Inside narrow private apartment room {number}, with a doorway at its north end and a bed at its south end. "
            "The doorway leads to the hallway. The garden and apple tree are outside the building, with no grass or open sky here. "
            "Only people in this room can hear.")
