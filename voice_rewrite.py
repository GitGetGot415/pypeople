"""Optional voice-rewrite pass: instruction, brief construction, parsing and diagnostic-only validation."""

import json
import logging

from actor_identity import actor_id, name_forms, own_name
from conversation_history import rewrite_validation_error, usable_rewrite
from dialogue import ActionResponse, ConversationResponse, VoiceRewriteResponse
from model_debug import record_output

logger = logging.getLogger(__name__)

# Free-form delivery guidance shared by all model-written voice rewrites.
DELIVERY_INSTRUCTION = """Interpret the COMPLETE Voice and Mannerisms as natural-language delivery requests.
Honor frequency, scope, negation and conditions, rather than latching onto a style
keyword. Occasional changes affect scattered words; keep the surrounding speech
normal. An always-on request can affect the whole utterance. Follow explicit
frequency limits in Voice even when Mannerisms repeats the style more generally.
Apply conditional delivery only when the supplied original and emotion support
that condition; do not invent an event or emotion to trigger it.

Unusual spelling, invented speech patterns, mixed alphabets, codes and symbolic
delivery are allowed. Interpret unfamiliar styles from the author's description;
there is no fixed list of supported voices. Preserve the supplied meaning and
protected names. Choose which words to transform according to the brief, without
explaining the transformation or adding translations unless the brief asks for them.
When a language is requested, express the entire utterance in that language;
do not leave ordinary source-language words untranslated or switch to a third
language. Keep protected names in their original spelling, even inside another
script. Emoji or symbol delivery uses actual Unicode characters, not colon-wrapped
shortcodes, English descriptions of symbols, or placeholder labels.
When asked to replace words, substitute the chosen words IN PLACE and leave the
other words in their original order. The replacement must represent that actual
source word. Never prepend arbitrary code-like syllables or decorative symbols,
and never keep both a coded word and its plain duplicate unless explicitly asked.
Transform content only once. Repetition is delivery only when requested, and must
stay finite. Stop each slot when its original meaning has been expressed; close
every JSON string and return all requested slots, even for symbolic speech.
"""


class VoiceRewriteValidationError(ValueError):
    """An invalid but displayable rewrite, retained alongside its diagnosis."""

    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


VOICE_REWRITE_INSTRUCTION = """Rewrite ONE speaker's supplied utterances in their Voice and Mannerisms.
Each part_N is an independent slot containing the current speaker's original.
Optional speaker_name and listener_name identify only names already in that original.
I/my belongs to the speaker; you/your belongs to the listener. Never answer as the
listener or borrow a different slot's words. Return only the flat part_N JSON strings.
An original statement stays a statement and an original question stays a question.
A questioning mannerism changes delivery; it cannot make known facts uncertain.

Preserve meaning first: facts, stance, refusals and objectives; keep
uncertainty, negation, agreements, quantities and deadlines, sources and roles.
Never copy another slot's speech. Do not invent events, explanations, gestures as stage
directions, private memories, thanks, answers or additional questions.

Preserve every proper name EXACTLY. Names are protected text, not style.
Keep names_to_preserve EXACTLY, in the same slot.
These names are literal protected spans: do not translate, transliterate, respell,
or replace them with emoji. All surrounding text follows the requested delivery.
Internal role descriptions are context, never names or greetings to add to speech.
Do not add a name absent from that original, even the speaker's or listener's.
Keep every original person's name and exact form intact.
When Voice asks for third-person self-reference or titles, apply it only where it
can preserve the original names and identity. These preservation rules take priority.

Change phrasing and rhythm only where the brief calls for it. A warm, concise voice
can stay close to an already natural original. Keep short replies short. Do not pad
speech or add a tic to every sentence. Convey mannerisms through spoken delivery,
without narration, stage directions, wrappers, labels, outer quotes or field readouts.
For emoji-only delivery, use actual Unicode emoji rather than colon-wrapped names
or shortcode descriptions. Keep symbolic clauses varied and finite. Finish each
slot once its meaning is conveyed; do not loop on symbols, modifiers or selectors.
Use the supplied emotion subtly. Keep each slot's complete original meaning.
Emotion changes tone only: anger is not a refusal. Never turn can into cannot,
agreement into disagreement, or a positive statement into its negation.
""" + "\n" + DELIVERY_INSTRUCTION


def voice_rewrite_snapshot(response, snapshot):
    """Copy only speech, explicit presentation fields and recorded delivery metadata."""

    def resident_brief(person):
        return {"name": own_name(person),
                "Voice": person["character"]["Voice"],
                "Mannerisms": person["character"]["Mannerisms"]}

    if isinstance(response, ConversationResponse):
        speakers = {role: resident_brief(snapshot[role]) for role in ("initiator", "recipient")}
        turns = [{"speaker": turn.speaker, "parts": [part.model_dump() for part in turn.parts]}
                 for turn in response.turns]
        thanks_role = ("recipient" if isinstance(response, ConversationResponse) and response.gift
                       and response.gift.recipient_id == actor_id(snapshot.get("recipient", {})) else "initiator")
        if isinstance(response, ConversationResponse) and response.handoff_thanks:
            turns.append({"speaker": thanks_role, "parts": [p.model_dump() for p in response.handoff_thanks]})
    elif isinstance(response, ActionResponse) and response.reply.strip():
        speaker = resident_brief(snapshot)
        speakers = {"speaker": speaker}
        emotion = response.emotion
        turns = [{"speaker": "speaker", "parts": [{"text": response.reply, "emotion": emotion.model_dump()}]}]
        person = snapshot
    else:
        return None
    for role, person in speakers.items():
        person["listener_name"] = next((other["name"] for key, other in speakers.items() if key != role), "You")
    catalog = snapshot.get("_name_actors", {})
    for turn in turns:
        turn["name_forms"] = name_forms(catalog)
    return {"request_kind": "voice_rewrite", "speakers": speakers, "turns": turns}


def voice_rewrite_batches(state):
    """Each provider call sees one speaker's brief and only that speaker's originals."""
    for role, person in state["speakers"].items():
        indices = [index for index, turn in enumerate(state["turns"]) if turn["speaker"] == role]
        if indices:
            yield indices, {"request_kind": "voice_rewrite", "speakers": {role: person},
                            "turns": [state["turns"][index] for index in indices]}


def validate_voice_rewrite(raw, turns, *, speakers=None):
    """Check display association and identity; simulation effects still use originals."""
    result = VoiceRewriteResponse.model_validate_json(raw)
    if len(result.turns) != len(turns) or any(not usable_rewrite(original["parts"], rewritten)
                                           for original, rewritten in zip(turns, result.turns)):
        raise ValueError("A rewrite needs exactly one nonempty text string per original part.")
    for original, rewritten in zip(turns, result.turns):
        speaker = speakers[original["speaker"]] if speakers is not None else {}
        error = rewrite_validation_error(original["parts"], rewritten, speaker=speaker.get("name", ""),
                                         listener=speaker.get("listener_name", ""), names=original.get("name_forms", ()),
                                         mannerisms=speaker.get("Mannerisms", ""))
        if error:
            raise VoiceRewriteValidationError(error, result)
    return result


def retain_voice_rewrite(raw, turns, *, speakers=None):
    """Keep a displayable failed rewrite and report the failed validation."""
    try:
        return validate_voice_rewrite(raw, turns, speakers=speakers)
    except VoiceRewriteValidationError as exc:
        message = f"Voice rewrite failed validation; retaining generated text: {exc}"
        logger.warning(message)
        record_output("validation", message)
        return exc.response


def voice_rewrite_parts(turns):
    parts = {}
    for turn in turns:
        for part in turn["parts"]:
            parts[f"part_{len(parts)}"] = {"speaker": turn["speaker"], "original": part["text"], "emotion": part["emotion"]}
            parts[f"part_{len(parts) - 1}"].update({key: turn.get(key, []) for key in ("name_forms",)})
            if "listener_name" in turn:
                parts[f"part_{len(parts) - 1}"]["listener_name"] = turn["listener_name"]
    return parts


def parse_voice_rewrite(raw, turns, *, speakers=None):
    """The provider fills flat named slots; Python reconstructs turn association."""
    def unique_parts(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("The rewrite repeats a part key.")
            data[key] = value
        return data
    data = json.loads(raw, object_pairs_hook=unique_parts)
    expected = voice_rewrite_parts(turns)
    if not isinstance(data, dict) or data.keys() != expected.keys():
        raise ValueError("The rewrite must supply exactly the requested part keys.")
    offset, rewritten = 0, []
    for turn in turns:
        rewritten.append([data[f"part_{index}"] for index in range(offset, offset + len(turn["parts"]))])
        offset += len(turn["parts"])
    return retain_voice_rewrite(json.dumps({"turns": rewritten}), turns, speakers=speakers)
