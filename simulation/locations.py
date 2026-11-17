from constants import (
    GARDEN_LOCATION, FOOT_MARGIN, ROOM_ARRIVAL, ELEVATOR_ARRIVAL, HALLWAY_EXIT, ROOM_EXIT,
    HALLWAY_ENTRY, GARDEN_BUILDING_POINT, GARDEN_BUILDING_ARRIVAL, GARDEN_BRIDGE_RECT,
    GARDEN_BRIDGE_APPROACH, MOVEMENT_SPEED, TRAVEL_STEP_TIMEOUT, PLAYER_APPROACH_MULTIPLIER,
    PLAYER_RUN_MULTIPLIER,
)
from simulation.interactions import move_toward
from dataclasses import dataclass, field
from typing import Annotated
from pydantic import Field
from simulation.maps import clamp_point, door_point, floor_for_room, hallway_location, valid_location


@dataclass
class RouteStep:
    point: tuple[float, float]
    destination: str
    arrival: tuple[float, float] | None


@dataclass
class Journey:
    destination: str
    point: tuple[float, float]
    purpose: str
    fast: bool = False
    elapsed: Annotated[float, Field(ge=0, le=TRAVEL_STEP_TIMEOUT)] = 0.0
    steps: list[RouteStep] = field(default_factory=list)


def available(npc):
    from simulation.activity import Behavior
    return (npc.alive and npc.behavior != Behavior.GO_HOME
            and not npc.journey)


def route(source, destination, point):
    """Use the same door and elevator connections for residents and player."""
    steps = []
    current = source
    if current.startswith("room:") and current != destination:
        room = int(current.split(":")[1])
        current = hallway_location(floor_for_room(room))
        steps.append(RouteStep(point=ROOM_EXIT, destination=current, arrival=door_point(room)))
    target_hall = (hallway_location(floor_for_room(int(destination.split(":")[1])))
                   if destination.startswith("room:") else destination)
    if current == GARDEN_LOCATION and target_hall != GARDEN_LOCATION:
        current = hallway_location(1)
        steps.append(RouteStep(point=GARDEN_BUILDING_POINT, destination=current, arrival=HALLWAY_ENTRY))
    if current.startswith("hallway:") and target_hall == GARDEN_LOCATION:
        if current != hallway_location(1):
            steps.append(RouteStep(point=ELEVATOR_ARRIVAL, destination=hallway_location(1), arrival=ELEVATOR_ARRIVAL))
        current = GARDEN_LOCATION
        steps.append(RouteStep(point=HALLWAY_EXIT, destination=current, arrival=GARDEN_BUILDING_ARRIVAL))
    elif current.startswith("hallway:") and current != target_hall:
        steps.append(RouteStep(point=ELEVATOR_ARRIVAL, destination=target_hall, arrival=ELEVATOR_ARRIVAL))
        current = target_hall
    if destination.startswith("room:") and current != destination:
        room = int(destination.split(":")[1])
        steps.append(RouteStep(point=door_point(room), destination=destination, arrival=ROOM_ARRIVAL))
    steps.append(RouteStep(point=clamp_point(destination, point), destination=destination, arrival=None))
    return steps


def start_journey(actor, destination, point, *, purpose="travel", fast=False):
    if not valid_location(actor.world, destination):
        raise ValueError("That destination is unavailable.")
    actor.destination = None
    actor.journey = Journey(destination=destination, point=tuple(point), purpose=purpose, fast=fast,
                         elapsed=0.0, steps=route(actor.location, destination, point))


def update_journey(actor, dt):
    journey = actor.journey
    if journey is None:
        return False
    journey.elapsed += dt
    if journey.elapsed > TRAVEL_STEP_TIMEOUT:
        actor.journey = None
        actor.world.publish("travel_failed", (actor,), "The journey could not reach its destination.")
        if actor is actor.world.player:
            from simulation.player import end_interaction
            end_interaction(actor.world, "The journey could not reach its destination.")
        return False
    step = journey.steps[0]
    speed = MOVEMENT_SPEED * (PLAYER_APPROACH_MULTIPLIER if journey.fast else 1)
    if actor is actor.world.player and journey.purpose == "walk" and actor.world.player_running:
        speed = MOVEMENT_SPEED * PLAYER_RUN_MULTIPLIER
    destination = clamp_point(actor.location, step.point)
    waypoint = destination
    if actor.location == GARDEN_LOCATION:
        x, _, width, _ = GARDEN_BRIDGE_RECT
        if destination[1] < FOOT_MARGIN and not x + FOOT_MARGIN <= actor.x <= x + width - FOOT_MARGIN:
            waypoint = GARDEN_BRIDGE_APPROACH
        elif actor.y < FOOT_MARGIN and destination[1] >= FOOT_MARGIN:
            waypoint = GARDEN_BRIDGE_APPROACH
    arrived = move_toward(actor, waypoint, dt, speed) and waypoint == destination
    if actor is actor.world.player and journey.purpose == "walk":
        from simulation.player import enter_touching_doorway
        if enter_touching_doorway(actor.world):
            return False
    if arrived:
        if step.arrival is not None:
            world = actor.world
            world.observed_event("departure", actor, None, f"{actor.name} left the area.")
            actor.location = step.destination
            actor.x, actor.y = clamp_point(actor.location, step.arrival)
            world.observed_event("arrival", actor, None, f"{actor.name} arrived in the area.")
            from simulation.interactions import observe_active_conversations
            observe_active_conversations(world)
        journey.steps.pop(0)
        journey.elapsed = 0.0
        if not journey.steps:
            actor.journey = None
            return True
    return False
