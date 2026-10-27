"""Authored events for new Player games; ordinary simulation dialogue stays generated."""

from player_records import PlayerCallReason
from constants import SNAKE_NAME, STARTING_DAY, CONVERSATION_COMPLETED


PLAYER_START_ENCOUNTER = "player-start-encounter"
INTRODUCTION_REASON = (
    "Start our first conversation: greet the player and discuss arriving here. "
    "We have not met before; their floating tag already shows their name."
)
JUDY_OPENING = (
    "Oh! Hi, I'm {name}. Sorry, I wasn't expecting anyone else to come up in the elevator. "
    "I arrived here yesterday, and you're the first other person I've seen here, aside from the snake. "
    "It called this place purgatory. I don't know what to make of that, but after a whole day "
    "on my own, I started wondering whether it was just for me. My own private purgatory. "
    "Though I suppose all those empty rooms should have been a clue that someone else might turn up. "
    "I feel a little silly now. Mostly relieved, honestly. It's good to have someone else here. "
    "I've looked through every nook and cranny in this world, and aside from this apartment building and the garden outside... "
    "There genuinely appears to be nothing else. This is it. "
    "Anyways, how are you holding up?"
)


def starting_encounter(world, resident_id=None):
    """The authored first-day meeting, retained as ordinary firsthand knowledge."""
    return next((fact for fact in world.facts if fact.source_id == PLAYER_START_ENCOUNTER
                 and fact.kind == "snake_claim"
                 and (resident_id is None or fact.object_id == resident_id)), None)


def starting_dialogue_context(world, resident_id):
    """Keep the authored conversation clear when answering questions about the snake."""
    if starting_encounter(world, resident_id) is None:
        return ""
    return (
        f"The snake the player is referring to IS {SNAKE_NAME}, the python wandering the garden. "
        "There is only one snake here, not a separate snake and a human resident. "
        f"Before this player arrived, you had a spoken conversation with {SNAKE_NAME}. "
        "You could read its floating name tag and heard its claimed divine role. You already know it speaks. At that meeting you "
        "had not yet seen another human here. Recall that conversation "
        "when asked about the snake, rather than describing an unfamiliar animal you have only watched. "
        "You took no apple in that meeting. You may still doubt its divine claims."
    )


def prepare_player_start(world):
    """Set the player's arrival day and give Judy her prior encounter."""
    from information import add_fact

    resident = world.npcs[0]
    add_fact(world, "snake_claim", world.snake, resident, PLAYER_START_ENCOUNTER, None,
             f"{resident.name} arrived in the garden and met {SNAKE_NAME}, the python wandering the garden. "
             "No apple was taken. "
             f"{SNAKE_NAME} described itself as a divine messenger and called this place purgatory; "
             f"{resident.name} heard these claims without knowing whether they are true. "
             f"At that time, {resident.name} had seen no other human here.",
             [resident, world.snake])
    world.day = STARTING_DAY + len(world.npcs)
    from constants import ARRIVAL_LOCATION
    from locations import door_point
    resident.location = ARRIVAL_LOCATION
    resident.x, resident.y = door_point(resident.room_number)


def start_player_introduction(world):
    from player import start_call

    judy = world.npcs[0]
    start_call(world, judy, PlayerCallReason(key=f"introduction:{judy.id}", kind="introduction",
                               reference=world.player.id, reason=INTRODUCTION_REASON))


def valid_introduction_call(world, npc, call, *, interaction):
    return (call.reference == world.player.id and call.key == f"introduction:{npc.id}"
            and starting_encounter(world, npc.id) is not None
            and bool(interaction and interaction.kind == "resident"
                     and interaction.npc_id == npc.id and interaction.incoming))


def starting_player_opening(world, npc):
    """The one-time story opening is prepared before Judy approaches the player."""
    if (starting_encounter(world, npc.id) is not None
            and not any(previous.between(world.player, npc)
                        and any(turn.speaker_id == npc.id for turn in previous.turns)
                        for previous in world.history)):
        # Include archived residents: the choice describes who was present at
        # the start, even if someone leaves before the introduction happens.
        starting_npcs = [person for person in (*world.npcs, *world.archived)
                         if person.arrival_day == STARTING_DAY]
        if len(starting_npcs) == 1:
            opening = JUDY_OPENING
        else:
            from autonomy import JUDY_ARRIVAL_OPENING
            opening = JUDY_ARRIVAL_OPENING
        return opening.format(name=npc.character["Name"])
    return ""


def player_question_opening(world, npc):
    """A player-initiated meeting starts with an immediate, neutral question mark."""
    from conversation_history import ConversationRecord
    from actor_identity import (actor_catalog, name_forms)
    from player import active_player, clear_call, owns_resident

    if not active_player(world) or not owns_resident(world, npc):
        return None
    text = "?"
    record = world.record_by_id(world.player_interaction.record_id)
    if record is None:
        record = ConversationRecord(("You", npc.name), "Conversation with the player", CONVERSATION_COMPLETED,
                                    origin="autonomous" if world.player_interaction.incoming else "player",
                                    participant_ids=[world.player.id, npc.id])
        record.intended_subjects = {"You": "Talking to this resident.", npc.name: "Listen to the player."}
        record.name_forms = name_forms(actor_catalog(world))
        world.history.append(record)
    record.status = CONVERSATION_COMPLETED
    record.error = ""
    record.add_turn(npc.name, text, [{"text": text, "emotion": {"name": "neutral", "intensity": 0}}], speaker_id=npc.id)
    world.player_interaction.record_id = record.id
    world.player_interaction.pending_request = None
    from interactions import observe_conversation
    observe_conversation(world, record, world.player, npc, accepted=True)
    npc.reply, npc.rewritten_reply, npc.player_reply.closing = text, "", False
    npc.set_emotion("neutral", 0)
    if not npc.unread_player_reply:
        clear_call(npc)
    return record
