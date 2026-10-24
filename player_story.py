"""Authored events for new Player games; ordinary simulation dialogue stays generated."""

from player_records import PlayerCallReason
from constants import (SNAKE_NAME, STARTING_DAY, STARTING_PHASE, STARTING_PLAYER_DAY, CONVERSATION_COMPLETED)
from character_data import DEFAULT_DIALOGUE


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
    "How are you holding up?"
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
        f"The snake the player is referring to IS {SNAKE_NAME}, the python guarding the apple tree. "
        "There is only one snake here, not a separate snake and a human tree guard. "
        f"Before this player arrived, you had a spoken conversation with {SNAKE_NAME}. "
        "You could read its floating name tag and heard it explain its apple "
        "information exchanges and claimed divine role. You already know it speaks. At that meeting you "
        "had not yet seen another human here. Recall that conversation "
        "when asked about the snake, rather than describing an unfamiliar animal you have only watched. "
        "You took no apple in that meeting. You may still doubt its divine claims."
    )


def prepare_player_start(world):
    """Give Judy her prior day here before assigning current-day goals."""
    from social import add_fact

    resident = world.npcs[0]
    add_fact(world, "snake_claim", world.snake, resident, PLAYER_START_ENCOUNTER, None,
             f"{resident.name} arrived in the garden and met {SNAKE_NAME}, the python guarding the apple tree. "
             f"{SNAKE_NAME} explained that it controls access to the apples: "
             "when fruit is available, it exchanges an apple immediately for information it values. "
             "No apple was taken and no information was traded. "
             f"{SNAKE_NAME} described itself as a divine messenger and called this place purgatory; "
             f"{resident.name} heard these claims without knowing whether they are true. "
             f"At that time, {resident.name} had seen no other human here.",
             [resident, world.snake])
    resident.attitudes.update(
        belief_reason=f"Has heard {SNAKE_NAME}'s claims but does not know whether they are true.",
        trust_reason=f"Has met {SNAKE_NAME} and heard its apple terms, but has not made a bargain.",
        authority_reason=f"Knows {SNAKE_NAME} guards the apples, but is unsure of its claimed divine authority.")
    world.day = STARTING_PLAYER_DAY
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


def scripted_player_phrase(world, npc, *, closing=False):
    """Record an authored player greeting or farewell without a provider call."""
    from conversation_history import ConversationRecord
    from actor_identity import (actor_catalog, name_forms)
    from player import active_player, clear_call, owns_resident

    if not active_player(world) or not owns_resident(world, npc):
        return None
    field = "Farewell" if closing else "Greeting"
    text = npc.character.get(field, "").strip() or DEFAULT_DIALOGUE[field]
    introduction = (not closing and starting_encounter(world, npc.id) is not None
                    and not any(previous.between(world.player, npc)
                                and any(turn.speaker_id == npc.id for turn in previous.turns)
                                for previous in world.history))
    if introduction:
        text = JUDY_OPENING.format(name=npc.character["Name"])
    record = world.record_by_id(world.player_interaction.record_id)
    if record is None:
        record = ConversationRecord(("You", npc.name), "Conversation with the player", CONVERSATION_COMPLETED,
                                    origin="autonomous" if world.player_interaction.incoming else "player",
                                    participant_ids=[world.player.id, npc.id])
        record.intended_subjects = {"You": "Talking to this resident.", npc.name: "Greet the player."}
        record.name_forms = name_forms(actor_catalog(world))
        world.history.append(record)
    if introduction:
        from social import orientation_speech
        record.topic = "First meeting with the player"
        record.intended_subjects = {"You": "Listening to Judy's introduction.", npc.name: INTRODUCTION_REASON}
        record.orientation_discussed = orientation_speech([text])
    record.status = CONVERSATION_COMPLETED
    record.error = ""
    record.add_turn(npc.name, text, [{"text": text, "emotion": {"name": "neutral", "intensity": 0}}], speaker_id=npc.id)
    world.player_interaction.record_id = record.id
    world.player_interaction.pending_request = None
    from interactions import observe_conversation
    observe_conversation(world, record, world.player, npc, accepted=True)
    npc.reply, npc.rewritten_reply, npc.player_reply.closing = text, "", closing
    npc.set_emotion("neutral", 0)
    if not npc.unread_player_reply:
        clear_call(npc)
    return record


def validate_starting_history(world, residents):
    from social import phase_index

    encounter = starting_encounter(world)
    if encounter is not None:
        resident = next((n for n in residents if n.id == encounter.object_id), None)
        if (resident is None or resident.arrival_day != STARTING_DAY or world.day < STARTING_PLAYER_DAY
                or encounter.subject_id != world.snake.id
                or encounter.phase != phase_index(STARTING_DAY, STARTING_PHASE)
                or encounter.conversation_id is not None
                or encounter.witness_ids != [resident.id, world.snake.id]):
            raise ValueError("Invalid prior-day starting encounter.")


def resident_arrival_days(world):
    days = set(range(2, world.day + 1))
    if starting_encounter(world) is not None:
        # This dawn brought the player; the next resident arrives the following day.
        days.discard(STARTING_PLAYER_DAY)
    return days
