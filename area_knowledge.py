"""The finished public layout and each speaker's current physical perspective."""

import math
import re

from constants import (APARTMENT_FLOORS, ROOMS_PER_FLOOR, JUDY_ROOM_NUMBER, STARTING_PHASE,
                       ROOM_BED_RECT, ROOM_EXIT, ELEVATOR_ARRIVAL, HALLWAY_EXIT,
                       GARDEN_BUILDING_POINT, GARDEN_BRIDGE_RECT, APARTMENT_RECT, TREE_BASE_OFFSET,
                       SPATIAL_NEAR_DISTANCE)


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
If asked how you know the layout, use familiarity.explanation: Judy has already walked
around; other residents first know from Judy, and after a time shift have confirmed it
by walking around themselves and seeing nothing else. This public layout knowledge
requires no private-memory disclosure or account_id. Do not recite it unless relevant.
Use only your CURRENT familiarity source, never the conditional rules for a later
phase. With source judy, you have not yet confirmed the layout yourself; do not say
you already walked around, or narrate a time shift that has not happened. Retain
your remembered elevator arrival; uncertainty about why you are here does not erase it.
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
        garden="A rectangular floating grassy island with one living apple tree. A narrow wooden bridge at its northwestern edge crosses the open gap to the apartment entrance. The island has an exposed earthen underside, and the apartment has a separate underside with visible pipes. Open sky surrounds them; no other accessible land or places are visible.",
        apartment=f"A cream-colored building with muted green trim, rows of windows and a brown entrance door. It has {APARTMENT_FLOORS} floors, with {ROOMS_PER_FLOOR} rooms per floor. The elevator serves every floor, even when rooms are vacant. There are no stairs.",
        hallway=f"A long, narrow hallway with a tan floor, cream wall and muted green trim. The elevator is at the western end; {ROOMS_PER_FLOOR} brown room doors line the north wall. Only floor 1 has the garden exit, at the southwestern end beside the elevator.",
        room="A narrow private room with a tan floor, cream wall and muted green trim. A wood-framed bed with a pale pillow and muted green blanket is against the north wall. One doorway at the south end leads back to that floor's hallway. No other furnishings or exits are present.",
        routes="From a room, use the south doorway to its hallway. On floors above 1, take the elevator to floor 1. Use the garden exit at the southwestern end of floor 1, then cross the wooden bridge to the grass. This exits the building, not this world.")


def familiarity(npc):
    from social import phase_index
    world = npc.world
    greeter = next((p for p in [*world.npcs, *world.archived] if p.room_number == JUDY_ROOM_NUMBER), None)
    name = greeter.name if greeter else "Judy Judge"
    if npc is greeter:
        return dict(source="own_observation", explanation="I already walked around the garden, building and rooms, and this is all I could see.")
    arrival = next((f.phase for f in world.facts if f.kind == "arrival" and f.subject_id == npc.id
                    and "arrived for dawn" in f.summary),
                   phase_index(npc.arrival_day, STARTING_PHASE))
    if world.phase_index > arrival:
        return dict(source="confirmed_by_walking", guide_name=name,
                    explanation=f"{name} told me about the layout, and since then I confirmed it by walking around myself. This is all I could see.")
    return dict(source="judy", guide_name=name,
                explanation=f"{name} already told me what the garden, building and rooms look like and where their exits are.")


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
    from locations import targets, door_point
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
    if world.mode == "player" and world.player.alive:
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
    result = dict(layout=known_layout(), current_location=actor.location,
                  facing=("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")[actor.facing],
                  spatial_view=spatial)
    if listener is not None:
        result["listener_name"] = listener.name
    if actor is not world.snake:
        result["familiarity"] = familiarity(actor)
    return result


def validate_area_speech(text, knowledge, *, question=""):
    """Check clear navigation claims; leave ambiguous phrasing to the model."""
    from dialogue_rules import unquoted_speech
    from dialogue_text import split_sentences
    source = knowledge.get("familiarity", {})
    layout_question = bool(re.search(r"\b(?:how|who|where)\b.*\b(?:know|learn|told|confirm|sure)\b", question, re.I)
        and re.search(r"\b(?:layout|rooms?|hallways?|building|doors?|exits?|look|surroundings|other places)\b", question, re.I))
    if source and layout_question:
        speech = unquoted_speech(text)
        if re.search(r"\byou (?:already )?know (?:the|this|what|how|because)\b|\bthat['’]s how you know\b", speech, re.I):
            raise ValueError("The question asks how the speaker knows the layout. Answer about I/my knowledge, not how the listener knows it.")
        if source["source"] == "judy":
            guide = re.escape(source["guide_name"].split()[0])
            to_guide = (knowledge.get("listener_name") == source["guide_name"]
                and re.search(r"\byou (?:already )?(?:told|explained|described|showed)\b", speech, re.I))
            told = re.search(rf"\b{guide}\b[^.!?]{{0,60}}\b(?:told|explained|described)\b|"
                             rf"\bfrom {guide}\b|\b{guide}['’]s (?:explanation|description)\b", speech, re.I)
            if not to_guide and not told:
                raise ValueError("Answer in first person about the speaker's knowledge: credit Judy's explanation of the layout, rather than merely mentioning Judy or describing her own walk.")
        elif source["source"] in ("own_observation", "confirmed_by_walking"):
            observed = re.search(r"\bi(?:['’]ve| have| did)? (?:(?:already|personally|also) )*(?:walked|looked|checked|confirmed|saw|seen)\b", speech, re.I)
            complete = re.search(r"\b(?:(?:all|everything) (?:i|we)(?:['’]ve| have| could| can)? (?:see|saw|seen|found)|"
                                 r"nothing else|no other|(?:that['’]s|that is|this is) (?:all|everything)|"
                                 r"(?:haven['’]t|have not|didn['’]t|did not) (?:see|seen|find|found) anything else)\b", speech, re.I)
            if not observed or not complete:
                raise ValueError("Use I/my for the speaker's own observation. Explain the speaker's own walk and that no other places were visible. Do not attribute that walk to the listener or answer how the listener knows.")
    for sentence in split_sentences(unquoted_speech(text)):
        if (not sentence.rstrip().endswith("?") and re.search(r"\b(?:stairs|staircase|stairwell)\b", sentence, re.I)
                and not re.search(r"\b(?:no|not|never|aren['’]t|isn['’]t|if|imagine|wish|remember|used to|former life)\b", sentence, re.I)):
            raise ValueError("There are no stairs in this building. Use the elevator for travel between floors and the floor-one garden exit to go outside.")
        proposal = re.search(r"\b(?:let['’]s|let us|we (?:can|could|should|might)|we['’]ll|(?:shall|should|could) we|why don['’]t we|what if we|would you like to|you['’]d like to|would like to|looking forward to|trying to find|(?:happy|eager|ready|excited) to|i (?:will|want to|suggest|plan to|intend to)|i['’]ll|i['’]d (?:like|love) to)\b", sentence, re.I)
        exploring = re.search(r"\b(?:explore|exploring|search|investigate|look around|scout|discover|uncover)\b", sentence, re.I)
        physical = exploring and (exploring[0].casefold() in ("look around", "scout") or re.search(
            r"\b(?:area|garden|building|place|surroundings|room|hallway|exit|clues?|hidden|discoveries|new things|nooks?|crann(?:y|ies))\b", sentence, re.I))
        if exploring and not physical and knowledge:
            physical = re.search(r"\bexplore (?:when|once|whenever|later|sometime)\b", sentence, re.I)
        if proposal and physical and not re.search(r"\b(?:not|don't|no need|already|former life|past life|used to)\b", sentence, re.I):
            raise ValueError("The layout is already known. Do not propose exploring or searching for new places, exits or clues. Use the known routes or discuss the actual concern.")
        if not knowledge or sentence.rstrip().endswith("?"):
            continue
        if (source.get("source") in ("own_observation", "confirmed_by_walking")
                and re.search(r"\bi (?:have not|haven['’]t|never)(?: (?:yet|ever|tried))? (?:been|gone|walked|going) (?:outside|out there|into the garden)\b", sentence, re.I)):
            raise ValueError("Your layout knowledge includes walking through the garden as well as the building. Do not claim you have never been outside or have yet to visit the garden.")
        if (knowledge.get("current_location", "").startswith("hallway:")
                and re.search(r"\b(?:the room we(?: are|['’]re) in|we(?: are|['’]re) (?:in|inside) (?:a |the |this )?(?:private |apartment )?room|this (?:private )?room (?:has|is|contains))\b", sentence, re.I)
                and not re.search(r"\b(?:if|imagine|remember|used to|earlier|thought|mistaken)\b", sentence, re.I)):
            raise ValueError("You are in the hallway, not a private room. Describe room furnishings as being inside the separate rooms, not where you currently stand.")
        if (knowledge.get("familiarity", {}).get("source") == "judy"
                and (layout_question or re.search(r"\b(?:layout|garden|building|hallways?|rooms?|this information)\b", sentence, re.I))
                and re.search(r"\bi(?: already| have| personally)? (?:walked around|confirmed|checked)|\bmy firsthand experience\b|"
                              r"\b(?:i|we)(?:['’]ve| have)?(?: already| personally)? (?:walked|explored|toured|confirmed)\b|"
                              r"\bi (?:remember|recall) (?:walking|looking) around\b|"
                              r"\bi(?:['’]ve| have)(?: been)? (?:walking|looking) around\b", sentence, re.I)
                and not re.search(r"\b(?:former life|past life|used to|not|never)\b", sentence, re.I)):
            raise ValueError("You have not been here through a time shift yet. Judy supplied this layout knowledge; do not claim you already confirmed it by walking around or by firsthand experience.")
        for landmark in knowledge.get("spatial_view", []):
            aliases = {"garden exit": r"(?:garden exit|exit door)", "room exit": r"(?:room exit|exit door)"}
            target = aliases.get(landmark["name"], re.escape(landmark["name"]))
            match = re.search(rf"\b{target}\b[^.!?;]{{0,80}}?\b(?:to |on )?(my|your) (left|right)|"
                              rf"\b{target}\b[^.!?;]{{0,80}}?\b(left|right) of (me|you)\b", sentence, re.I)
            if match:
                if re.search(r"\b(?:not|isn't|never)\b", match[0], re.I):
                    continue
                owner = (match[1] or match[4]).casefold()
                direction = (match[2] or match[3]).casefold()
                perspective = landmark.get("from_speaker" if owner in ("my", "me") else "from_listener")
                if perspective and direction not in perspective["body_directions"]:
                    raise ValueError("That left/right direction conflicts with the current positions and facing. Use the landmark's supplied from_speaker or from_listener direction, or its compass direction.")
            longitudinal = re.search(rf"\b{target}\b[^.!?;]{{0,80}}?\b(behind|in front of|ahead of) (me|you)\b", sentence, re.I)
            if longitudinal and not re.search(r"\b(?:not|isn't|never)\b", longitudinal[0], re.I):
                perspective = landmark.get("from_speaker" if longitudinal[2].casefold() == "me" else "from_listener")
                direction = "behind" if longitudinal[1].casefold() == "behind" else "ahead"
                if perspective and direction not in perspective["body_directions"]:
                    raise ValueError("That ahead/behind direction conflicts with the participants' current positions and facing. Use the supplied perspective for the person you are describing, or compass directions.")
