import math
from constants import (
    APARTMENT_FLOORS, ROOMS_PER_FLOOR, STARTING_DAY, ROOM_BED_RECT, ROOM_EXIT, ELEVATOR_ARRIVAL,
    HALLWAY_EXIT, GARDEN_BUILDING_POINT, GARDEN_BRIDGE_RECT, APARTMENT_RECT, TREE_BASE_OFFSET,
    SPATIAL_NEAR_DISTANCE,
)


AREA_KNOWLEDGE_INSTRUCTION = """The supplied area_knowledge describes the finished, known layout.
Residents already know the garden, apartment hallways, rooms and routes between them.
Do not suggest exploring, searching the area, finding the right door, hidden clues,
new places or an undiscovered exit. Walking for company or using a known route is fine.
Why anyone was brought here and how to return to their former world remain unknown;
that does not make the local layout or the garden exit unknown. Answer local navigation
questions with the known route. Rooms contain only the supplied fixtures; do not invent
shops, corridors, furniture, secret passages or scenery beyond this layout.
The elevator is the only route between floors; there are no stairs.
current_location identifies where you actually stand: hallway:N is a hallway,
room:N is a private room, and garden is outdoors. Other layout entries describe
other places; do not describe their fixtures as being in your current location.
Layout information is public knowledge and requires no private-memory disclosure,
account_id or credit to Judy or any other guide. Familiarity is background context,
not a required script or attribution. Discuss it only when relevant.
Use only your CURRENT familiarity source, never the conditional rules for a later
phase. With source judy, you have not yet confirmed the layout yourself; do not say
you already walked around, or narrate a time shift that has not happened. Retain
your remembered elevator arrival; uncertainty about why you are here does not erase it.
With awaiting_welcome, nobody has explained the layout to you yet. Listen to the
current welcome; do not invent an earlier conversation or tour.
With unguided_arrival, this is your first spoken turn after arriving without a welcome.
React to the unexplained arrival in your own personality: confused, angry, skeptical
or quietly uncertain. Answer the listener's actual words; routine goals can wait.
With general_layout, you have the basic layout mapped out and can use known routes.
You need not announce this or repeat your arrival confusion in every conversation.
Neither unguided_arrival nor general_layout supplies a guide, a tour, an earlier
welcome, or a firsthand walk through every location. Do not invent these experiences.
Knowing the layout does not explain why you are here or how to return home.
With own_observation or confirmed_by_walking, you have already walked through both
the garden and building. Do not claim you have never been outside or only know the
hallway. The garden is outside the building, connected by the bridge, not a room
on the first floor. No further exploration is needed to learn its layout.
Answer layout-source questions directly, in first person about the speaker. Do not
tell the listener that they were the one who toured the place, or recite the elevator
arrival instead of explaining where the speaker's layout knowledge came from.
For directions, use spatial_view: map directions are fixed, body directions depend on
facing. My left refers to from_speaker; your left refers to from_listener. Facing each
other gives different left/right. Use compass directions if a body direction is unclear.
Never guess directions from an earlier reply or a listener's mistaken wording.
"""


def known_layout():
    return dict(
        garden="A rectangular floating grassy island with one living apple tree that is sturdy and unbreakable. A narrow wooden bridge at its northwestern edge crosses the open gap to the apartment entrance. The island has an exposed earthen underside, and the apartment has a separate underside with visible pipes. Open sky surrounds them; no other accessible land or places are visible.",
        apartment=f"A cream-colored building with muted green trim, rows of windows and a brown entrance door. It has {APARTMENT_FLOORS} floors, with {ROOMS_PER_FLOOR} rooms per floor. The elevator serves every floor, even when rooms are vacant. There are no stairs.",
        hallway=f"A long, narrow hallway with a tan floor, cream wall and muted green trim. The elevator is at the western end; {ROOMS_PER_FLOOR} brown room doors line the north wall. Only floor 1 has the garden exit, at the southwestern end beside the elevator.",
        room="A narrow private room with a tan floor, cream wall and muted green trim. A wood-framed bed with a pale pillow and muted green blanket is against the north wall. One doorway at the south end leads back to that floor's hallway. No other furnishings or exits are present.",
        apple_access="Living residents and the player can harvest available tree apples locally without permission, payment or quotas. Stock is finite and replenishes at the start of each new day. Harvesting briefly shakes the tree. Visiting is travel only; speech never transfers apples. Carried apples can be given to others even when the tree is empty.",
        routes="From a room, use the south doorway to its hallway. On floors above 1, take the elevator to floor 1. Use the garden exit at the southwestern end of floor 1, then cross the wooden bridge to the grass. This exits the building, not this world.")


def unguided_familiarity(*, first_turn):
    return dict(source="unguided_arrival" if first_turn else "general_layout",
                explanation=("I remember my elevator arrival, but nobody welcomed me. I am still making sense of being here."
                             if first_turn else
                             "I have the general layout mapped out. This does not mean someone welcomed me or that I toured every location."))


def familiarity(npc):
    from simulation.autonomy import arrival_greeter, awaiting_arrival_greeting
    world = npc.world
    greeter = arrival_greeter(world, include_archived=True)
    if npc is greeter:
        return dict(source="own_observation", explanation="I already walked around the garden, building and rooms, and this is all I could see.")
    arrival = next((f.day for f in world.facts if f.kind == "arrival" and f.subject_id == npc.id
                    and "arrived for dawn" in f.summary),
                   npc.arrival_day)
    # Starting residents share the initial setup; later arrivals need an actual welcome.
    welcomed = greeter is not None and (npc.arrival_day == STARTING_DAY or any(
        record.between(greeter, npc) and any(turn.speaker_id == greeter.id for turn in record.turns)
        for record in world.history))
    if welcomed:
        if world.day > arrival:
            return dict(source="confirmed_by_walking", guide_name=greeter.name,
                        explanation=f"{greeter.name} told me about the layout, and since then I confirmed it by walking around myself. This is all I could see.")
        return dict(source="judy", guide_name=greeter.name,
                    explanation=f"{greeter.name} told me what the garden, building and rooms look like and where their exits are.")
    if greeter in world.npcs and awaiting_arrival_greeting(npc, world):
        return dict(source="awaiting_welcome", explanation="I remember arriving in the elevator. Nobody has explained the layout to me yet.")
    # The immediate question mark on a player approach is a listening cue, not an arrival reaction.
    has_spoken = any(turn.speaker_id == npc.id and turn.text.strip() != "?"
                     for record in world.history for turn in record.turns)
    return unguided_familiarity(first_turn=not has_spoken)


def relative_position(actor, point):
    dx, dy = point[0] - actor.x, point[1] - actor.y
    if dx == 0 and dy == 0:
        return dict(map_direction="here", body_directions=["here"])
    direction = round(math.atan2(dx, -dy) / (math.pi / 4)) % 8
    body = (direction - actor.facing) % 8
    return dict(map_direction=("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")[direction],
                nearby=math.hypot(dx, dy) <= SPATIAL_NEAR_DISTANCE,
                body_directions=(("ahead",), ("ahead", "right"), ("right",), ("behind", "right"),
                                 ("behind",), ("behind", "left"), ("left",), ("ahead", "left"))[body])


def area_knowledge(actor, listener=None):
    from simulation.maps import active_player, targets, door_point
    world = actor.world
    if isinstance(listener, str):
        listener = world.participant_by_id(listener)
    landmarks = []
    if actor.location == "garden":
        landmarks = [("apple tree", (world.tree.x, world.tree.y + TREE_BASE_OFFSET)),
                     ("apartment building", (APARTMENT_RECT[0] + APARTMENT_RECT[2] / 2,
                                             APARTMENT_RECT[1] + APARTMENT_RECT[3] / 2)),
                     ("apartment entrance", GARDEN_BUILDING_POINT),
                     ("wooden bridge", (GARDEN_BRIDGE_RECT[0] + GARDEN_BRIDGE_RECT[2] / 2,
                                        GARDEN_BRIDGE_RECT[1] + GARDEN_BRIDGE_RECT[3] / 2))]
    elif actor.location.startswith("hallway:"):
        landmarks = [("elevator", ELEVATOR_ARRIVAL)]
        if actor.location == "hallway:1":
            landmarks.append(("garden exit", HALLWAY_EXIT))
        landmarks.extend((f"room {t['room']} door", door_point(t["room"]))
                         for t in targets(world, actor.location) if t["kind"] == "room")
    else:
        landmarks = [("bed", (ROOM_BED_RECT[0] + ROOM_BED_RECT[2] / 2, ROOM_BED_RECT[1] + ROOM_BED_RECT[3] / 2)),
                     ("room exit", ROOM_EXIT)]
    visible = []
    people = [*world.npcs, world.snake]
    if active_player(world):
        people.append(world.player)
    for other in people:
        if other is not actor and other.location == actor.location:
            visible.append((other.name, (other.x, other.y)))
    spatial = []
    for name, point in [*landmarks, *visible]:
        row = dict(name=name, from_speaker=relative_position(actor, point))
        if listener is not None and listener.location == actor.location:
            row["from_listener"] = relative_position(listener, point)
        spatial.append(row)
    layout = known_layout()
    if getattr(actor, "speech_only", False):
        # The python has only ever lived in the garden: it sees the building from outside and has no resident familiarity.
        layout = {key: layout[key] for key in ("garden", "apple_access")}
        layout["apartment"] = ("A cream-colored building with muted green trim, rows of windows and a brown entrance door, "
                               "across the wooden bridge. What lies inside is unknown to you.")
    result = dict(layout=layout, current_location=actor.location,
                  facing=("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")[actor.facing],
                  spatial_view=spatial)
    if listener is not None:
        result["listener_name"] = listener.name
    if not getattr(actor, "speech_only", False):
        result["familiarity"] = familiarity(actor)
    return result
