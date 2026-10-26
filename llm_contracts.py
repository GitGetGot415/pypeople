"""Instruction, JSON schema and message assembly for every model request."""

import json
import re
from copy import deepcopy
from dataclasses import dataclass

import dialogue_context
import dialogue_rules
from actor_identity import actor_id, forms, own_name, speech_only
from area_knowledge import AREA_KNOWLEDGE_INSTRUCTION
from constants import (CONVERSATION_MAX_REPUTATION_EFFECTS, DIALOGUE_CLOSING_WORD_LIMIT, DIALOGUE_CORRECTION_TARGET_CHARACTERS,
    DIALOGUE_PART_TARGET_CHARACTERS, DIALOGUE_PARTS_PER_TURN, PLAYER_NAME_TAG, PLAYER_REPLY_CONTEXT_TURN_LIMIT,
    SNAKE_NAME)
from dialogue import (ActionResponse, ConversationResponse, DialoguePartsResponse, DialogueRepairResponse,
    EmotionReaction, OpinionDirection, PlayerIntent, ResidentAssessment, ResidentTurnResponse, VoiceRewriteResponse)
from dialogue_context import NAME_TAG_INSTRUCTION, NPC_ARRIVAL_INSTRUCTION, PLAYER_TAG_INSTRUCTION, account_speech_context
from dialogue_rules import is_name_reference, name_pattern, named_references
from llm_prompts import (CONVERSATION_HISTORY_INSTRUCTION, DIALOGUE_VARIETY_INSTRUCTION, EMOTION_INSTRUCTION,
    GARDEN_INSTRUCTION, INTENT_INSTRUCTION, ORIENTATION_SPEAKING_BRIEF, PLAYER_POSSESSIONS_INSTRUCTION,
    SPEAKING_BRIEF, WORLD_INSTRUCTION, PLAYER_OPENING_INSTRUCTION, build_system_instruction)
from information import information_from_dict, resident_effect_capacity
from voice_rewrite import VOICE_REWRITE_INSTRUCTION, voice_rewrite_parts


def structured_schema(model):
    """Inline model definitions for the two providers' existing JSON schema path."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    def expand(value):
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return expand(definitions[value["$ref"].split("/")[-1]])
        return {key: expand(item) for key, item in value.items()
                if key not in ("default", "additionalProperties") and (key != "title" or isinstance(item, dict))}
    return expand(schema)


EMOTION_SCHEMA = structured_schema(EmotionReaction)
EMOTION_SCHEMA["required"] = list(EMOTION_SCHEMA["properties"])


# Response models own field types and limits; request-specific restrictions
# and required fields are applied below.
RESPONSE_SCHEMA = structured_schema(ActionResponse)
# Keep the generation order: choose the action/closing state before writing speech.
RESPONSE_SCHEMA["properties"] = {
    name: RESPONSE_SCHEMA["properties"][name]
    for name in ("action", "end_conversation", "gift_offer", "gift_acceptance", "player_opinion", "apple_promise", "invitation", "invitation_acceptance", "invitation_resolution", "invitation_resolution_evidence", "reply", "target", "item", "quantity", "emotion", "opinion_updates")
}
RESPONSE_SCHEMA["required"] = ["action", "reply", "end_conversation", "target", "item", "quantity", "opinion_updates", "emotion"]
RESPONSE_SCHEMA["properties"]["emotion"] = EMOTION_SCHEMA

INTENT_SCHEMA = structured_schema(PlayerIntent)
INTENT_SCHEMA["properties"] = {"message_type": INTENT_SCHEMA["properties"]["message_type"], **INTENT_SCHEMA["properties"]}
INTENT_SCHEMA["required"] = list(INTENT_SCHEMA["properties"])
INTENT_SCHEMA["properties"]["item"] = RESPONSE_SCHEMA["properties"]["item"]


def intent_snapshot(snapshot):
    """Route from current intent, with only the latest utterance for short answers."""
    def identity(person):
        return {
            "actor_id": person["actor_id"],
            "name": person["name"],
            "kind": person.get("kind", "resident"),
            "character": {"Name": person["character"]["Name"]},
            "inventory": person.get("inventory", {}).copy(),
        }

    previous = ""
    if dialogue_rules.short_player_answer(snapshot.get("_player_message", "")):
        previous = snapshot.get("last_reply", "")
        for turn in reversed(snapshot.get("player_dialogue", [])):
            speaker, text = (turn["speaker"], turn["text"]) if isinstance(turn, dict) else turn
            if speaker == snapshot["name"]:
                previous = text
                break
    return {
        "request_kind": "intent",
        **identity(snapshot),
        "owned_works": snapshot.get("owned_works", []),
        "player_inventory": snapshot.get("player_inventory", {}),
        "_player_message": snapshot.get("_player_message", ""),
        "answering_npc_utterance": previous,
        "resolved_work_id": snapshot.get("resolved_work_id"),
        "other_npcs": [
            {**identity(other), "available_to_talk": other.get("available_to_talk", True)}
            for other in snapshot.get("other_npcs", [])
        ],
    }


def _dialogue_parts_schema(*, correction=False):
    """Share part counts and soft size guidance without clipping sentences."""
    parts = structured_schema(DialoguePartsResponse)["properties"]["parts"]
    parts["items"]["properties"]["emotion"] = EMOTION_SCHEMA
    parts["items"]["required"] = ["text", "emotion"]
    parts["maxItems"] = 1 if correction else DIALOGUE_PARTS_PER_TURN
    target = DIALOGUE_CORRECTION_TARGET_CHARACTERS if correction else DIALOGUE_PART_TARGET_CHARACTERS
    parts["items"]["properties"]["text"]["description"] += (
        f" Prefer about {target} characters, but finish each sentence even if it is longer. "
        "Never stop mid-sentence to meet a size target. End with sentence punctuation.")
    return parts


def _available_speech_fields(schema):
    """Do not ask the model to fill bookkeeping that this stage cannot use."""
    properties = schema["properties"]
    for field, definition in list(properties.items()):
        if definition.get("type") == "array" and definition.get("maxItems") == 0:
            del properties[field]
    schema["required"] = [field for field in schema.get("required", []) if field in properties]
    return schema


def _possession_offer_schema(person, recipient_id):
    stock = person.get("inventory", {}).get("fruit", 0)
    items = [("fruit", stock)] if stock else []
    items.extend((w["id"], 1) for w in person.get("owned_works", []))
    choices = [{"type": "object", "properties": {
        "recipient_id": {"type": "string", "enum": [recipient_id]},
        "item": {"type": "string", "enum": [item]},
        "quantity": {"type": "integer", "minimum": 1, "maximum": amount}},
        "required": ["recipient_id", "item", "quantity"], "additionalProperties": False}
        for item, amount in items]
    return {"anyOf": [{"type": "null"}, *choices]} if choices else {"type": "null"}


def _resident_stage_model(kind):
    return {"resident_turn": ResidentTurnResponse, "resident_assessment": ResidentAssessment,
            "handoff_thanks": DialoguePartsResponse}.get(kind)


def _reputation_items(snapshot, item):
    """One schema branch per eligible heard account, fixing its absent subject."""
    options = {o.id: o for o in information_from_dict(snapshot["_baseline"]["_information_validation"]).options}
    branches = []
    for claim_id in snapshot["choices"]["third_party_opinions"]:
        branch = deepcopy(item)
        branch["properties"]["claim_id"] = {"type": "string", "enum": [claim_id]}
        branch["properties"]["subject_id"] = {"type": "string", "enum": [options[claim_id].subject_id]}
        branches.append(branch)
    return branches[0] if len(branches) == 1 else {"anyOf": branches}


def _resident_contract(snapshot, *, correction=False):
    assessment = snapshot["request_kind"] == "resident_assessment"
    thanks = snapshot["request_kind"] == "handoff_thanks"
    orientation = snapshot.get("arrival_focus", False)
    model = response_model(snapshot)
    schema = structured_schema(model)
    properties = schema["properties"]
    if not assessment:
        properties["parts"] = _dialogue_parts_schema(correction=correction)
    if not thanks:
        reactions = []
        for entry in snapshot["pending_reactions"]:
            reactions.append({"type": "object", "properties": {
                "disclosure_turn": {"type": "integer", "enum": [entry["disclosure_turn"]]},
                "stance": {"type": "string", "enum": entry["stances"]}}, "required": ["disclosure_turn", "stance"]})
        properties["reactions"].update(minItems=len(reactions), maxItems=len(reactions))
        if reactions:
            properties["reactions"]["items"] = reactions[0] if len(reactions) == 1 else {"anyOf": reactions}
    if assessment:
        # A confrontation's leave or attack is rolled by the game from the committed view, never chosen here.
        properties["escalation"] = {"type": "null"}
        properties["opinion"]["properties"]["person"] = {"type": "string", "enum": [snapshot["opinion_target"]]}
        gossip = snapshot["choices"]["third_party_opinions"]
        if gossip:
            # Believed hearsay about an absent person must leave an impression;
            # small models otherwise never volunteer this optional effect.
            facts = {o.id: o.fact_id for o in information_from_dict(snapshot["_baseline"]["_information_validation"]).options}
            count = min(len({facts[c] for c in gossip}), CONVERSATION_MAX_REPUTATION_EFFECTS)
            properties["third_party_opinions"].update(minItems=count, maxItems=count,
                                                      items=_reputation_items(snapshot, properties["third_party_opinions"]["items"]))
        else:
            properties["third_party_opinions"] = {"type": "array", "maxItems": 0}
    elif not thanks:
        choices, draft = snapshot["choices"], snapshot["_draft"]
        properties["gift_offer"] = (_possession_offer_schema(snapshot["speaker"], snapshot.get("gift_recipient_id", ""))
                                    if not draft.get("gift") else {"type": "null"})
        pending = draft.get("gift")
        if not pending or pending.recipient_id != actor_id(snapshot["speaker"]) or pending.accept_turn is not None:
            properties.pop("accept_gift", None)
        # A confrontation is made right with a promised apple (offers), never an unasked gift.
        if (snapshot.get("arrival_focus") or snapshot.get("purpose") in ("food_request", "confrontation")
                or snapshot.get("social_only")):
            properties.pop("gift_offer", None)
            properties.pop("accept_gift", None)
        for field, limit in resident_effect_capacity(draft).items():
            if field == "third_party_opinions":
                continue  # Hearsay is judged only in the closing assessment.
            if field == "disclosures":
                limit = min(1, limit)
            properties[field]["maxItems"] = max(0, limit) if choices[field] else 0
            if field in ("offers", "acceptances"):
                properties[field]["maxItems"] = min(properties[field]["maxItems"], len(choices[field]))
            if choices[field]:
                item = properties[field]["items"]
                if field in ("offers", "acceptances"):
                    item["enum"] = choices[field]
                else:
                    item["properties"]["claim_id"] = {"type": "string", "enum": choices[field]}
        if snapshot.get("_disclosure_repair"):
            properties["disclosures"]["items"]["properties"]["claim_id"]["enum"] = [snapshot["_disclosure_repair"]]
            properties["disclosures"]["minItems"] = 1
        elif bool(snapshot.get("discussion_focus_claim_id")) and snapshot["turn_index"] == 0:
            properties["disclosures"]["minItems"] = 1
    schema["required"] = (["opinion"] if assessment else ["parts"])
    if assessment and snapshot["choices"]["third_party_opinions"]:
        schema["required"].append("third_party_opinions")
    if snapshot.get("pending_reactions"):
        schema["required"].append("reactions")
    if bool(snapshot.get("discussion_focus_claim_id")) and snapshot["turn_index"] == 0 and not assessment:
        schema["required"].append("disclosures")
    elif not assessment and not thanks and properties["disclosures"].get("maxItems", 0):
        schema["required"].append("disclosures")  # [] remains valid when no account is spoken.
    _available_speech_fields(schema)
    if snapshot.get("_compact_rewrite"):
        instruction = (f"You are {snapshot['speaker']['name']}, replying only to {snapshot['listener']['name']}. "
            "Write your own answer to the last listener message, never a copy or continuation of their words. "
            "Their I/my belongs to them; your I/my belongs to you. Acknowledge their point briefly, then add your own response. "
            "Use only your supplied knowledge. Keep the owners and sources of memories, claims and obligations unchanged. "
            "Do not invent events, read planning notes aloud, or rewrite accepted speech. "
            "Return parts with one short complete sentence ending in punctuation and your own emotion. "
            "Finish that sentence before starting another; a soft size target must never clip a sentence. "
            "Use only available choices for effects: disclose a claim only when actually telling its content, "
            "omit disclosures for ordinary reactions, and resolve each pending reaction using its allowed stance. "
            "An offer needs your spoken apple commitment and deadline; acceptance needs your spoken agreement. "
            "Keep all accepted effects fixed.\nPersonality: " + snapshot["speaker"].get("personality", "")
            + "\nCurrent task: " + snapshot["task"])
        if snapshot.get("discussion_focus_claim_id") and snapshot["turn_index"] == 0 and not assessment:
            instruction += (" Your sentence must state the concrete event of the required account in your own words: "
                            "who did what, and how many apples if it is a harvest.")
        # The short prompt otherwise loses the metadata a told account needs; name it outright.
        required = snapshot.get("_disclosure_repair") or (
            snapshot.get("discussion_focus_claim_id") if snapshot["turn_index"] == 0 and not assessment else None)
        if required:
            instruction += (" Include " + json.dumps({"disclosures": [{"claim_id": required, "intent": "share"}]})
                            + " alongside parts.")
        if snapshot.get("_closing"):
            instruction += f" Answer question_to_answer in at most {DIALOGUE_CLOSING_WORD_LIMIT} words and end without a question."
        return instruction, schema, model
    instruction = (f"You are {snapshot['speaker']['name']}, speaking directly to {snapshot['listener']['name']}.\n"
        + (ORIENTATION_SPEAKING_BRIEF if orientation else SPEAKING_BRIEF)
        + "\n" + EMOTION_INSTRUCTION + "\n"
        "Use only your own supplied context and accepted preceding speech. The listener's private views are unknown. "
        "Speaker and listener names are visible on their floating tags. Give only your own reply, never the listener's answer. "
        "Use one or two short complete sentences with your emotion, ending in sentence punctuation. "
        "Sources, subjects, and objects of accounts remain distinct. Firsthand means you witnessed the event, not that you performed its action. "
        "Use subject_relation: speaker means I, listener means you, another person means subject_name. "
        "A heard account is hearsay, never your own remembered life. Denials, uncertainty and deliberate lies remain distinct. "
        "Hearing a divine claim is different from believing its theology. "
        "Respond to what the partner actually said; do not invent transfers, events, discoveries or shared experiences.\n"
        + CONVERSATION_HISTORY_INSTRUCTION + "\n" + DIALOGUE_VARIETY_INSTRUCTION + "\n"
        + snapshot["speaker"].get("personality", "") + "\nCurrent task: " + snapshot["task"])
    if (not assessment and not thanks and not orientation and snapshot.get("purpose") not in ("food_request", "confrontation")
            and not snapshot.get("social_only")):
        instruction += (" You may offer one actual possession using gift_offer with recipient_id for this listener, item fruit or an exact owned_works ID, and quantity. "
            "Say clearly what you offer; name the work title if creative. Only one gift offer per conversation. "
            "Use accept_gift true only when you explicitly accept an earlier pending_gift directed to you. "
            "You may decline without penalty. Gifts occur after this conversation, never claim delivery already happened. "
            "Collections are unlimited and separate from apple capacity. You know your own works; do not infer the listener's private collection.")
    feeling = (snapshot["speaker"].get("opinion_of_listener") or {}).get("feeling")
    if not thanks and feeling and snapshot["speaker"].get("kind") != "python":
        instruction += (" opinion_of_listener.feeling is how you feel about the listener; "
                        + ("weigh this exchange against it and against anything because lists that you already held against them."
                           if assessment else
                           "let it set your tone. neutral: as your personality suggests. warm, friendly or close: friendly and at ease. "
                           "cold: reserved and guarded, with no warm praise. hostile: openly wary and distrustful, with no warmth, "
                           "praise or thanks beyond bare courtesy. very hostile: curt, cold and openly accusing. "
                           "because lists what you hold against them; you may raise it in your own words. Never state the score."))
    if snapshot["speaker"].get("dangerous_partner"):
        instruction += (" Your own believed account and current relationship make this listener seem dangerous. "
                        "You are usually reluctant to seek optional company, give unsolicited gifts or invite them home. "
                        "An explanation or peaceful confrontation is possible; honor accepted promises and real handoffs.")
    if snapshot.get("social_only"):
        instruction += (" The python cannot harvest or possess apples. Never attribute a resident's harvest to the python, "
                        "whether you are speaking as the python or addressing it. Witnessing is not collecting. "
                        "This meeting permits conversation and sourced reports; do not bargain, demand payment, offer gifts or promise apples.")
    if assessment:
        instruction += ("\nThe transcript is fixed. Write a first-person reason in plain prose, in your own words, saying how "
                        "opinion_target's words or conduct in this exchange made you feel about them: better, worse or no different, and why. "
                        "It is not a summary of the conversation or of your own personality, interests, confusion, or behavior. "
                        "Use I/my for yourself; do not describe yourself by name. Resolve pending_reactions using only their allowed stances. "
                        "Add no dialogue, introductions or other effects, and never revise an accepted stance.")
        if "third_party_opinions" in properties:
            instruction += (" You believed an account about an absent person in this exchange. For each one, add a third_party_opinions "
                            "entry whose reason says, in your own words, whether what you heard makes you think better, worse or no "
                            "differently of that absent person, and why. believed_accounts lists each account, who told it to you, and "
                            "whether you saw it happen yourself. You accept that each one happened; never deny or doubt it. "
                            "In that reason, you/your means the person who told you. When an entry has your_view_of_them worse "
                            "or far worse, what you heard makes you think that much worse of that person, even if you saw it "
                            "yourself: hearing it told again stirs you up against them. Say why in your own words.")
        settled = snapshot.get("confrontation_verdict")
        if settled:
            accused = settled["accused"]
            instruction += (f" {accused} promised to make this right, so your view of them is no different for now. "
                            if settled["promised"] else
                            f" {accused} made no promise to make this right; an apology or explanation alone does not repair it, "
                            "so you think " + ("far " if settled["change"] == "terrible" else "") + "worse of them. ") + "Say why in your own words."
        verdict = snapshot.get("snake_verdict")
        if verdict:
            accused = verdict["accused"]
            instruction += {
                "better": f" {snapshot['opinion_target']} came to think worse of {accused} after your accusation, so you think better of them.",
                "worse": f" {snapshot['opinion_target']} did not accept your accusation against {accused}, so you think worse of them.",
                "terrible": f" {snapshot['opinion_target']} did not accept your accusation against {accused}, so you think far worse of them.",
                "unchanged": f" {snapshot['opinion_target']} heard your accusation against {accused} without changing their view, "
                             "so your view of them is no different.",
            }[verdict["change"]] + (f" Say why in your own words. You remain certain {accused} acted selfishly: never doubt, "
                                    f"excuse or reconsider {accused}.")
    elif not thanks:
        instruction += ("\nReply to the listener first. Personality is private behavioral guidance, never text to recite "
                        "or attribute to the listener. Do not read objectives or account menus aloud.")
        if "disclosures" in properties:
            instruction += (" Tell at most one selected account, with its actual owner and source, and record only its claim and intent. "
                            "Do not recite other private details; the listener chooses their own stance later. "
                            + ("Regenerate the account you chose to tell, with its matching disclosure entry." if snapshot.get("_disclosure_repair") else
                               "Disclose the required task account now." if bool(snapshot.get("discussion_focus_claim_id")) and snapshot["turn_index"] == 0
                               else "Return disclosures: [] if you tell no account. If you tell a supplied memory or account, include its claim_id and intent in disclosures on this same reply."))
            focus = snapshot["speaker"].get("account_focus")
            if focus and focus["claim_id"] in snapshot["choices"]["disclosures"]:
                instruction += (" Your selected account is account_focus. If you tell its content, include "
                                + json.dumps({"disclosures": [{"claim_id": focus["claim_id"], "intent": "share"}]})
                                + " alongside parts. Preserve subject_relation and subject_name: firsthand knowledge does not make the speaker the actor. "
                                "For harvest accounts, assertion and quantity give the witnessed collector total for event_time; pickup_count counts the separate pickups. "
                                "harvest_context gives the other witnessed collectors' totals in that same phase. Do not shrink a repeated harvest to one pickup. "
                                "Use those totals and the current tree stock without inventing unobserved pickups or combining phases. "
                                "Read the current remaining stock from environment.tree.fruit, never from population minus one pickup. "
                                "harvest_extent, when supplied, establishes the collector's share of the phase's full stock. "
                                "If its share is all, the collector took every apple; do not change that to most, deny it, or suggest some remained in a follow-up. "
                                "Do not use current stock as the stock of a different phase. "
                                "An empty tree alone does not prove a particular resident took all its fruit.")
            elif focus and any(d.claim_id == focus["claim_id"] for d in snapshot["_draft"]["disclosures"]):
                instruction += (" account_focus has already been shared in this exchange. Respond to the listener's reaction "
                                "without another disclosure entry for it; an ordinary follow-up uses disclosures: [].")
        if "offers" in properties or "acceptances" in properties:
            instruction += (" Only a giver can offer an available future apple proposal, explicitly naming its apple and deadline. "
                            "Only its receiver can accept an earlier offer in spoken words. A request is not an offer; acceptance is not delivery. "
                            "Omit offers/acceptances when your speech makes no such commitment.")
        if "reactions" in properties:
            instruction += " Resolve your pending heard accounts using only their permitted stances before choosing this reply's effects."
        if "leverage" in properties:
            instruction += " Leverage is explicit pressure using an already known account; ordinary requests need no leverage."
        if any(field in properties for field in ("offers", "acceptances", "disclosures", "reactions", "third_party_opinions", "leverage")):
            instruction += " All metadata belongs to this speaker and this turn; Python assigns indices. Do not choose the listener's effects."
    if snapshot.get("arrival_focus", False):
        instruction += ("\nDiscuss arrival, surroundings or seeking help and safety. Defer routine apple goals. "
            "Names are visible on floating tags; start with the actual arrival concern. "
            "The cause of arrival and way home are unknown. "
            "No apple promises, accounts or leverage are available.")
    if snapshot.get("_closing"):
        instruction += f"\nAnswer question_to_answer in at most {DIALOGUE_CLOSING_WORD_LIMIT} words; finish without any question."
    return instruction, schema, model


def creative_contract(kind):
    from creations import response_model
    from constants import (ART_COLOR_NAMES, ART_PALETTE, ART_SIZE, CREATION_TITLE_LIMIT, MUSIC_MAX_NOTES,
                           MUSIC_MAX_NOTE_STEPS, MUSIC_MAX_TRACKS, MUSIC_PITCH_RANGE, MUSIC_STEPS, MUSIC_TEMPO_RANGE)
    instruction = (f"Create a private work inspired by the supplied author's own profile and the garden. "
                   f"Return only title (1-{CREATION_TITLE_LIMIT} characters) and payload. "
                   "Do not include dialogue, descriptions, facts about other residents, authorship IDs or dates. ")
    if kind == "art":
        palette = ", ".join(f"{i}={name} #{r:02X}{g:02X}{b:02X}"
                            for i, (name, (r, g, b)) in enumerate(zip(ART_COLOR_NAMES, ART_PALETTE)))
        instruction += (f"Paint a recognizable tiny pixel picture. payload is {{pixels: rows}} with exactly "
                        f"{ART_SIZE} rows of exactly {ART_SIZE} integer indices each. All pixels are opaque. "
                        f"Use only this palette: {palette}.")
    else:
        instruction += (f"Compose {MUSIC_STEPS} steps: four bars, four beats per bar, four steps per beat. "
                        f"payload has integer tempo {MUSIC_TEMPO_RANGE[0]}-{MUSIC_TEMPO_RANGE[1]} BPM and "
                        f"1-{MUSIC_MAX_TRACKS} tracks. Each track has waveform square or triangle and notes. "
                        f"Use 1-{MUSIC_MAX_NOTES} notes TOTAL. Each note has integer MIDI pitch "
                        f"{MUSIC_PITCH_RANGE[0]}-{MUSIC_PITCH_RANGE[1]}, only natural notes C D E F G A B "
                        f"(pitch modulo 12 in 0,2,4,5,7,9,11), start 0-{MUSIC_STEPS - 1}, duration "
                        f"1-{MUSIC_MAX_NOTE_STEPS} steps. Every note ends by {MUSIC_STEPS}; no overlapping notes "
                        "within a track. Use a short motif, repetition, and rests. A compact valid example: "
                        '{"title":"Garden Bells","payload":{"tempo":120,"tracks":[{"waveform":"triangle",'
                        '"notes":[{"pitch":60,"start":0,"duration":4},{"pitch":64,"start":8,"duration":4},'
                        '{"pitch":60,"start":32,"duration":4}]}]}}')
    model = response_model(kind)
    return instruction, structured_schema(model), model


def response_model(snapshot):
    """Select the trusted contract without constructing prompts or dynamic schemas."""
    kind = snapshot.get("request_kind")
    if kind in ("create_art", "create_music"):
        from creations import response_model as creation_model
        return creation_model(kind.removeprefix("create_"))
    resident = _resident_stage_model(kind)
    if resident is not None:
        return resident
    return {"conversation": ConversationResponse,
            "conversation_closing": DialogueRepairResponse, "opinion_direction": OpinionDirection,
            "intent": PlayerIntent, "voice_rewrite": VoiceRewriteResponse}.get(kind, ActionResponse)


def _player_command_brief(intent):
    """The same action meaning accompanies either player input route and retries."""
    if not intent or intent["action"] == "SAY":
        return ""
    brief = ("Briefly acknowledge the current action request in your own voice, before carrying it out. "
             "Do not answer an earlier conversation question or claim the task already happened. ")
    return brief


def response_contract(snapshot, *, correction=False):
    kind = snapshot.get("request_kind")
    if kind in ("create_art", "create_music"):
        return creative_contract(kind.removeprefix("create_"))
    if _resident_stage_model(kind) is not None:
        return _resident_contract(snapshot, correction=correction)
    model = response_model(snapshot)
    if kind == "conversation":
        # Assembly/queue contract only. Resident provider calls use individual stages.
        return WORLD_INSTRUCTION, structured_schema(model), model
    if kind == "conversation_closing":
        listener = snapshot["listener"]
        listener_name = listener["name"] if isinstance(listener, dict) else listener
        instruction = (
            f"You are {snapshot['speaker']['name']}, speaking directly to {listener_name}.\n"
            + SPEAKING_BRIEF
            + "Answer question_to_answer directly using your own "
            "views and what has already been said, then bring the exchange to a natural close. "
            "If you do not know the answer, acknowledge that honestly. Do not ask another question. "
            f"Use one or two short declarative sentences, at most {DIALOGUE_CLOSING_WORD_LIMIT} words total, and your own emotion. "
            "State your answer or intended plan directly. A suggestion must be a statement, not a question "
            "asking for approval, agreement, or another choice. "
            "This is your final message; the listener will not reply. "
            "Focus on the question; omit unrelated updates. Do not recap the conversation, "
            "speak for the listener, disclose a new account, make a new agreement, or change the "
            "settled outcome. Do not claim a new item transfer or invent events or private knowledge. "
            "Existing agreements name their giver and receiver; preserve who owes what to whom. "
            "Situation and task fields are context, never words to quote. Return only parts.\n"
            + snapshot.get("_speaker_behavior", snapshot["speaker"].get("personality", ""))
            + "\n" + EMOTION_INSTRUCTION

            + "\nFinal task: briefly answer the listener's last question using only already stated facts "
            "and your own views. End without asking any question."
        )
        schema = structured_schema(model)
        schema["properties"]["parts"]["items"]["properties"]["text"]["description"] = (
            "The speaker's direct answer in complete statements. State an answer, intention, or honest uncertainty. "
            "Finish the exchange without asking a question or requesting approval.")
        return instruction, _available_speech_fields(schema), model
    if snapshot.get("request_kind") == "intent":
        targets = [other["actor_id"] for other in snapshot.get("other_npcs", []) if other.get("available_to_talk", True)]
        schema = {**INTENT_SCHEMA, "properties": {**INTENT_SCHEMA["properties"],
            "item": {"type": "string", "enum": ["", "fruit", *[w["id"] for w in snapshot.get("owned_works", [])]]},
            "target": {"type": "string", "enum": ["", *targets]}}}
        return INTENT_INSTRUCTION + PLAYER_POSSESSIONS_INSTRUCTION, schema, model
    if speech_only(snapshot):
        snapshot["request_intent"] = dict(action="SAY", target="", item="", quantity=0)
    opinion_schema = RESPONSE_SCHEMA["properties"]["opinion_updates"]
    people = [other["name"] for other in snapshot.get("other_npcs", []) if other.get("kind") != "player"]
    schema = {**RESPONSE_SCHEMA, "properties": {
        **RESPONSE_SCHEMA["properties"],
        "opinion_updates": {**opinion_schema, "maxItems": 8 if people else 0, "items": {
            **opinion_schema["items"], "properties": {
                **opinion_schema["items"]["properties"],
                "person": {"type": "string", "enum": people or [""]},
            },
        }},
    }}
    instruction = build_system_instruction(snapshot["character"],
        conversation_only=snapshot.get("request_intent", {}).get("action") == "SAY")
    if snapshot.get("request_kind") == "player_opening":
        instruction = PLAYER_OPENING_INSTRUCTION + "\n" + EMOTION_INSTRUCTION
    instruction += ("\nPhysical invitations: invitation selects only a supplied invitation_choices destination, with exact evidence from your reply. "
        "This creates an offer, not travel. invitation_acceptance identifies an existing pending_invitation only when the current player "
        "message accepts that actual offer in context, including short answers. Questions, refusals and quotations are not acceptance. "
        "Use invitation_resolution declined for the player's actual refusal, or withdrawn when your original reply withdraws the offer; "
        "quote exact invitation_resolution_evidence from that player message or your withdrawal respectively. Never resolve an absent offer. "
        "A closed-door knock requests entry: offer home only when physically present and available. Otherwise use null and empty acceptance ID.")
    schema["properties"]["gift_offer"] = _possession_offer_schema(snapshot, snapshot.get("player_id", ""))
    if snapshot.get("request_kind") in ("player_gift", "player_acknowledgment"):
        schema["properties"]["gift_offer"] = {"type": "null"}
    if not snapshot.get("player_interaction", {}).get("incoming_gift"):
        schema["properties"]["gift_acceptance"] = {"type": "boolean", "enum": [False]}
    if speech_only(snapshot):
        for field in ("gift_offer", "apple_promise", "invitation"):
            schema["properties"][field] = {"type": "null"}
        schema["properties"]["gift_acceptance"] = {"type": "boolean", "enum": [False]}
        instruction += " You are a python: speak only. You cannot harvest, give possessions, make apple promises, invite anyone home, or perform resident tasks."
    instruction += PLAYER_POSSESSIONS_INSTRUCTION
    if snapshot.get("request_kind") == "player_command":
        instruction += (" The player deliberately selected the structured player_interaction.command. "
                        "It is a trusted action request, not a spoken utterance. Respond naturally to that request "
                        "using request_intent directly; do not invent or quote player speech. "
                        "Keep IDs in metadata only; use names or you in speech. This selection does not accept "
                        "an invitation, apple promise, or unsolicited gift offer.")
    if snapshot.get("request_kind") in ("player_opening", "player_gift", "player_acknowledgment"):
        instruction += " This is an engine-initiated turn, not a player command. Use action SAY, empty target/item, quantity zero. Generate speech from player_interaction, without inventing player speech."
        snapshot = {**snapshot, "request_intent": dict(action="SAY", target="", item="", quantity=0)}
    if snapshot.get("request_kind") == "player_opening":
        instruction += (" You have reached the player. Express your actual "
                        "player_interaction.call reason immediately, without asking permission to talk. "
                        "The player has said nothing yet. Leave room for their response: end_conversation false.")
        schema["properties"]["end_conversation"] = {"type": "boolean", "enum": [False]}
    if snapshot.get("request_intent"):
        intent = snapshot["request_intent"]
        instruction += "\n" + _player_command_brief(intent)
        giving = intent["action"] == "GIVE_ITEM"
        schema = {**schema, "properties": {
            **schema["properties"],
            "action": {"type": "string", "enum": [intent["action"], "SAY"] if giving else [intent["action"]]},
            "target": {"type": "string", "enum": [intent["target"], ""] if giving else [intent["target"]]},
            "item": {"type": "string", "enum": [intent["item"], ""] if giving else [intent["item"]]},
            "quantity": {"type": "integer", "enum": [intent["quantity"], 0] if giving else [intent["quantity"]]},
        }}
        if giving:
            instruction += (" A requested gift is not yet a commitment. If you decline, lack the item, "
                            "or need an answer before agreeing, use SAY with empty target/item and quantity 0. "
                            "A question must use SAY. Use GIVE_ITEM only when agreeing to perform the exact "
                            "requested handoff; give a brief acknowledgment with no question. "
                            "Never substitute a different recipient, item or quantity.")
    starting_experience = snapshot.get("private_context", {}).get("starting_experience")
    if starting_experience:
        instruction += "\nYour established personal experience: " + starting_experience
        instruction += (" When asked about this encounter, if earlier dialogue denied it, correct that mistake. "
                        "The player's uncertainty does not erase what you personally heard. "
                        "Messages labeled Engine context are recorded background, not player speech. "
                        "Answer the actual final player message. Use this experience only when relevant; "
                        "it is not an instruction to discuss the snake in every reply.")
    elif snapshot.get("private_context", {}).get("situation", {}).get("prior_snake_encounters", 0):
        instruction += (f"\nEstablished personal experience: you have already met {SNAKE_NAME}. "
                        "When asked about that encounter, use your firsthand known_accounts to recall what you heard, including its actual claims. "
                        "You may doubt its divine claims without denying the meeting or conversation. "
                        "When that encounter is relevant, if earlier dialogue wrongly denied it, acknowledge the mistake and answer "
                        "from recorded experience. Answer the current message rather than changing the topic to this encounter.")
    return instruction, schema, ActionResponse


def _tree_stock_instruction(context):
    tree = context.get("environment", {}).get("tree")
    if tree is None:
        return ""
    instruction = (f"\nKnown public tree stock: {tree['fruit']} apples total, "
                   f"{tree['available_fruit']} available. All participants already know this state. "
                   "Use this to constrain actions; discuss stock only when it answers the listener's concern. ")
    if tree["fruit"] == 0:
        instruction += (
            "The living apple tree remains standing and visible; only its apple count is zero. "
            "It has not vanished, died, burned or been removed. Answers about obtaining its apples must acknowledge that none can be "
            "obtained now, rather than treating the stock as unknown or needing inspection. "
            "Do not claim a new tree apple has been obtained. Harvesting needs positive stock and local collection. "
            "Carried apples and actually recorded earlier obligations are separate from the tree's missing fruit."
        )
    return instruction


def _request_context(message, snapshot, correction):
    """Build the same instructions and world payload for both providers."""
    if snapshot.get("request_kind") in ("create_art", "create_music"):
        instruction, schema, _ = response_contract(snapshot)
        content = {"author": snapshot["author"], "theme": "A quiet garden with an apple tree",
                   "correction": correction}
        return instruction, schema, content
    if snapshot.get("request_kind") == "conversation":
        raise ValueError("Autonomous conversations must use the worker's sequential turn loop.")
    if snapshot.get("request_kind") == "opinion_direction":
        owner, target = snapshot["owner"], snapshot["target"]
        instruction = (f"{owner} wrote the reason below about {target}. Decide only how it changes {owner}'s opinion of {target}. "
            f"Choose a worse change when the reason describes {target} as selfish, dishonest, unkind, manipulative or untrustworthy, "
            f"or says {owner} feels wary, hurt, disappointed or uneasy about them. "
            + (f"Choose a terrible change, instead of worse, only when the reason describes something so grave that worse is not enough, "
               f"such as betrayal, cruelty, threats or violence by {target}. " if "terrible" in snapshot["labels"] else "")
            + f"Choose a better change when it describes {target} "
            f"as kind, honest, generous, helpful or pleasant, or says {owner} feels appreciative, comfortable or close to them. "
            f"Choose unchanged when the reason is neutral, only restates events without judging them, says {owner} feels no "
            f"different, or judges someone other than {target}. Return only the change JSON."
            + ("\n" + correction if correction else ""))
        schema = {"type": "object", "properties": {"change": {"type": "string", "enum": snapshot["labels"]}}, "required": ["change"]}
        return instruction, schema, {"owner": owner, "target": target, "reason": snapshot["reason"]}
    if snapshot.get("request_kind") == "voice_rewrite":
        speakers = snapshot["speakers"]
        parts = {key: {
            "speaker_name": speakers[part["speaker"]]["name"],
            "listener_name": part.get("listener_name", speakers[part["speaker"]]["listener_name"]),
            "Voice": speakers[part["speaker"]]["Voice"],
            "Mannerisms": speakers[part["speaker"]]["Mannerisms"],
            "emotion": part["emotion"], "original": part["original"],
            "speaker_role": "The original's I/my; retain this speaker in every clause.",
            "listener_role": "The original's you/your; never substitute the speaker's name.",
        } for key, part in voice_rewrite_parts(snapshot["turns"]).items()}
        # Repeat the exact references locally so small models need not resolve
        # names across other speakers' very similar lines.
        names = sorted({form for person in speakers.values()
                        for name in (person["name"], person["listener_name"]) if name != "You"
                        for form in (name, name.split()[0])}, key=lambda name: (-len(name), name))
        names = sorted(set(names).union(*(part["name_forms"] for part in voice_rewrite_parts(snapshot["turns"]).values())),
                       key=lambda name: (-len(name), name))
        for part in parts.values():
            part["names_to_preserve"] = list(dict.fromkeys(match.group() for match in re.finditer(
                "|".join(name_pattern(name) for name in names), part["original"], re.I)
                if is_name_reference(match, part["original"])))
            # Names absent from this original are unnecessary style context.
            # Listing them as labels or names to avoid primed small models to add them.
            for field in ("speaker_name", "listener_name"):
                label = part[field]
                spoken = next((name for name in part["names_to_preserve"] if name in forms(label)), None)
                if spoken:
                    part[field] = spoken
                else:
                    part.pop(field)
        schema = {"type": "object", "properties": {key: {
            "type": "string", "minLength": 1,
            "description": f"Direct utterance by {part.get('speaker_name', 'the current speaker')} to {part.get('listener_name', 'the current listener')}. No speaker label, narration or outer quotes. Keep original names unchanged."
        } for key, part in parts.items()}, "required": list(parts)}
        return (VOICE_REWRITE_INSTRUCTION + ("\n" + correction if correction else ""), schema,
                {"parts": parts})
    instruction, schema, _ = response_contract(snapshot, correction=bool(correction))
    context = dialogue_context.world_context(snapshot, correction=bool(correction))
    opening = snapshot.get("request_kind") == "player_opening"
    if opening:
        context = {key: value for key, value in context.items() if key in (
            "name", "actor_id", "player_relationship", "player_interaction", "opening_account",
            "earlier_player_conversations", "inventory", "owned_works")}
        context["personality"] = dialogue_context.character_context(snapshot["character"]).get("personality", "")
        context["environment"] = dialogue_context.local_environment(snapshot.get("location", "garden"))["description"]
    if snapshot.get("_compact_rewrite") and snapshot.get("request_kind") != "resident_turn":
        instruction = (
            f"You are {dialogue_context._actor_name(snapshot['speaker'])}, speaking only to {dialogue_context._actor_name(snapshot['listener'])}. "
            "Write fresh speech for this one step. The structured outcome and account selection are fixed. "
            "Use only supplied facts and your own knowledge. Follow the current task; "
            "previous events are not a new offer or transfer. Respond to the listener's latest point, "
            "without repeating the opening, introduction, accusation, or earlier wording. "
            "If account_to_share is present, express its meaning in natural speech with its true subject, "
            "owner, source and stance; never read JSON, keys, narrator labels or instructions aloud. "
            "Do not invent an agreement, action or private memory. Keep I/my with the speaker and you/your "
            "with the listener. Write one or two short complete sentences ending in punctuation. Return only parts.\n"
            + snapshot.get("_speaker_behavior", snapshot["speaker"].get("personality", ""))
            + "\nCurrent task: " + snapshot.get("task", "Answer question_to_answer and finish without another question.")
            + "\n" + EMOTION_INSTRUCTION)
        if snapshot["request_kind"] == "conversation_closing":
            instruction += f"\nAnswer question_to_answer in at most {DIALOGUE_CLOSING_WORD_LIMIT} words. End on a statement, not a question or an invitation."
        elif "word_limit" in snapshot:
            instruction += f"\nAim for at most {snapshot['word_limit']} words, completing the sentence."
    focused_account = snapshot.get("account_to_share") or snapshot.get("speaker", {}).get("account_focus")
    tree_relevant = not (focused_account and focused_account.get("kind") == "killing")
    if snapshot.get("player_id"):
        tree_relevant = bool(re.search(r"\b(?:apples?|fruit|tree|harvest\w*)\b", message, re.I))
    stock_instruction = ("" if opening or not tree_relevant or snapshot.get("request_kind") == "conversation_closing"
                         or snapshot.get("_closing") else _tree_stock_instruction(context))
    instruction += stock_instruction
    if snapshot.get("request_kind") == "conversation_closing":
        instruction += (
            f"\nWrite only {dialogue_context._actor_name(snapshot['speaker'])}'s words to {dialogue_context._actor_name(snapshot['listener'])}. "
            "Keep that identity in every part. I/my refers to the speaker; you/your refers to the listener. "
            "Do not address yourself by name, answer your own question as the other person, or adopt the listener's "
            "memories, authority, debts, or task. Explicit quotations and correct self-introductions are allowed.")
    instruction += ("\nUse ordinary capitalization and plain prose for base speech and opinion reasons. "
                    "Express the supplied behavioral personality; presentation is applied in a separate optional pass. "
                    "Do not imitate casing, accents or verbal tics from previous speech.")
    if NAME_TAG_INSTRUCTION not in instruction:
        instruction += "\n" + NAME_TAG_INSTRUCTION
    if PLAYER_TAG_INSTRUCTION not in instruction and any(
            actor.get("kind") == "player" and actor.get("_visible") for actor in snapshot.get("_name_actors", {}).values()):
        instruction += PLAYER_TAG_INSTRUCTION
    if not opening and NPC_ARRIVAL_INSTRUCTION not in instruction:
        instruction += "\n" + NPC_ARRIVAL_INSTRUCTION
    if not opening:
        instruction += "\n" + AREA_KNOWLEDGE_INSTRUCTION
    if snapshot.get("_name_actors") and not opening:
        context["engine_identities"] = dialogue_context._catalog_for_prompt(snapshot["_name_actors"])
    if snapshot.get("request_kind") in ("resident_turn", "conversation_closing"):
        context["reply_context"] = dialogue_context._reply_context(snapshot)
        instruction += ("\nConversation messages contain actual accepted speech: assistant/model messages are your earlier "
                        "words; user messages after the context are the listener's words. Write your reply to the last "
                        "listener message. Answer their questions before moving the topic forward; do not summarize "
                        "what they said or substitute your profile for an answer. If asked how you know a name, explain the floating name tag. "
                        "Their I/my refers to the listener; their you/your refers to you. Your first sentence should "
                        "address their specific point from your own perspective. Keep emotion labels and internal "
                        "resource terms in metadata.")
    instruction = named_references(instruction)
    if snapshot.get("request_kind") in ("resident_turn", "conversation_closing", "handoff_thanks"):
        context = account_speech_context(context, snapshot["speaker"], snapshot["listener"], snapshot.get("_name_actors", {}))
        # Opinion targets are needed only by the separate assessment stage.
        context.pop("opinion_target", None)
        context["speaker"].pop("opinion_record_name", None)
    if snapshot.get("player_id") and snapshot.get("request_kind") != "intent":
        speaker = snapshot
        context["spoken_identity"] = {
            "speaker_actor_id": actor_id(speaker), "speaker_self_name": own_name(speaker),
            "listener": {"actor_id": snapshot["player_id"], "name_tag": PLAYER_NAME_TAG},
        }
        instruction += (
            f"\nWrite only {own_name(speaker)}'s next reply to the human player, in first person. "
            "In YOUR reply, I/my is the speaker and you/your is the player. "
            "In the incoming player message, I/me/my means the human player; you/your means you, the NPC. "
            "The player's floating name tag reads Player. Their preferred nickname may also appear in their actual speech; "
            "do not confuse your own name with theirs. "
            "Your profile, private_context, memories, opinions, goals belong to YOU. "
            "They are not the player's biography or actions. Attribute every account to its actual actors. "
            "Knowing a tag name proves no past meeting, apple transfer or obligation. "
            "Only actual player speech and recorded player transfers support such claims. "
            "An engine-initiated opening has no player utterance to thank, summarize or answer. "
            "Earlier accepted replies have already been spoken; do not copy their prose into a new reply. "
            "When the player answers your question, acknowledge what their answer changes instead of asking it again."
        )
    location = snapshot.get("location", snapshot.get("initiator", {}).get("location", "garden"))
    if location == "garden" and not opening and tree_relevant:
        context.setdefault("environment", dialogue_context.local_environment(location))
        instruction += "\n" + GARDEN_INSTRUCTION
    elif location == "garden" and not opening:
        instruction += "\nCurrent surroundings: " + dialogue_context.local_environment(location)["description"]
    elif not opening:
        context["environment"] = dialogue_context.local_environment(location)
        instruction += ("\nYour current surroundings are " + context["environment"]["place"] + ". "
                        "Use environment.description and its visible objects when describing where you are. "
                        "You are inside the apartment building. The garden, grass, apple tree and open sky are outside, "
                        "and are not visible from here. Distinguish memories of being outside and plans to go outside "
                        "from your present surroundings. Earlier dialogue cannot change your current location.")
    payload = {"world_state": context, "player_message": message}
    if correction:
        if snapshot.get("request_kind") in ("resident_turn", "conversation_closing", "handoff_thanks"):
            feedback = account_speech_context({"assertion": correction}, snapshot["speaker"], snapshot["listener"],
                                               snapshot.get("_name_actors", {}))["assertion"]
            instruction += "\nCorrection for the current reply: " + named_references(feedback)
        else:
            payload["correction_request"] = named_references(correction)
    if stock_instruction:
        payload["resource_constraints"] = stock_instruction.strip()
    return instruction, schema, payload


def _conversation_messages(payload, snapshot):
    """Keep private context separate from chronological, speaker-relative speech."""
    if snapshot.get("player_id") and snapshot.get("request_kind") != "intent":
        context = payload["world_state"]
        context.pop("player_dialogue", None)
        context.pop("last_reply", None)
        message = payload.pop("player_message", "")
        starting_experience = snapshot.get("private_context", {}).get("starting_experience")
        correction = payload.pop("correction_request", None) if starting_experience else None
        messages = [{"role": "user", "content": payload}]
        for turn in snapshot.get("player_dialogue", []):
            speaker, text = (turn["speaker"], turn["text"]) if isinstance(turn, dict) else turn
            if speaker == "You":
                messages.append({"role": "user", "content": text})
            elif speaker == own_name(snapshot) or speaker == snapshot.get("name"):
                messages.append({"role": "assistant", "content": {"reply": text}})
        if starting_experience:
            reminder = {"context": "Engine context, not player speech. Recorded experience overrides mistaken earlier replies. "
                                   "This is background knowledge; answer the current player message, using this only if relevant.",
                        "recorded_personal_experience": starting_experience}
            if correction:
                reminder["correction_request"] = correction
            messages.append({"role": "user", "content": reminder})
        if message:
            content = message
            if snapshot.get("request_intent", {}).get("action") == "SAY":
                # Keep the current task beside the utterance. Long world rules,
                # old farewells and the starting memory can otherwise dominate it.
                content = {"player_message": message,
                           "reply_task": "Write your reply to this player_message. Your first sentence must address "
                                         "what the player asks or says, using earlier_player_conversations for follow-ups. "
                                         "If this message answers your last question, react to the answer; do not repeat that question or your previous response. "
                                         "If they refer to 'the question', answer previous_player_message explicitly; "
                                         "an apology or general confusion alone does not answer it. "
                                         "If you do not know the answer, say what you do not know directly. "
                                         "Unknown does not mean impossible. Do not invent an answer, an escape restriction "
                                         "or past investigations. Prefer one or two short sentences; when the answer is "
                                         "unknown, say so without padding it with speculative explanations or memories. "
                                         "For questions, do not begin with a farewell, an unrelated "
                                         "memory, or your own plans; answer even if you choose end_conversation true. "
                                         "For a goodbye or declined offer of more conversation, just close. Return the required reply JSON."}
                if snapshot.get("conversation_account"):
                    content["ongoing_accusation"] = snapshot["conversation_account"]
                    content["opinion_of_player"] = snapshot.get("player_relationship", {}).get("opinion")
                    content["reply_task"] += (" Respond to these actual words while retaining the ongoing accusation. "
                        "A joke, denial, boast or topic change does not erase what you know. "
                        "Firsthand means you witnessed the event; hearsay stays attributed to its source. "
                        "Do not claim you forgot the event, never heard of the victim, or just learned it from the player. "
                        "Do not change the killer or claim that a witness committed the killing.")
                recent = [turn for exchange in snapshot.get("earlier_player_conversations", [])
                          for turn in exchange["turns"]] + snapshot.get("player_dialogue", [])
                if recent:
                    # Follow-ups must not require finding one short question
                    # inside the much larger private knowledge payload.
                    content["recent_player_dialogue"] = recent[-PLAYER_REPLY_CONTEXT_TURN_LIMIT:]
                    content["previous_player_message"] = next((turn["text"] for turn in reversed(recent)
                        if turn["speaker"] == "You" and turn["text"] != message), "")
                situation = snapshot.get("private_context", {}).get("situation", {})
                if situation:
                    content["known_situation"] = {key: situation[key] for key in
                                                  ("explanation", "way_home") if key in situation}
            if snapshot.get("request_intent", {}).get("action") not in (None, "SAY"):
                content = {"player_message": message, "request_intent": snapshot["request_intent"],
                           "reply_task": _player_command_brief(snapshot["request_intent"])}
            messages.append({"role": "user", "content": content})
        else:
            call = snapshot.get("player_interaction", {}).get("call")
            if snapshot.get("request_kind") == "player_opening" and call:
                messages.append({"role": "user", "content": {
                    "opening_reason": call["reason"],
                    "opening_account": snapshot.get("opening_account"),
                    "opinion_of_player": snapshot.get("player_relationship", {}).get("opinion"),
                    "reply_task": "You have reached the player. Start the conversation about opening_reason now. "
                                  "Express opening_reason directly in your own voice, rather than a generic greeting "
                                  "or a request for permission to talk. If considering a gift, offer it or explain "
                                  "your decision to keep it. The player has not said anything. "
                                  "Use SAY and end_conversation false. Return the required reply JSON."}})
            elif snapshot.get("request_kind") == "player_command":
                messages.append({"role": "user", "content": {
                    "player_action_request": snapshot["player_interaction"]["command"],
                    "reply_task": "The player selected this action, without speaking a new line. "
                                  "Respond to the structured request using request_intent. A gift may be declined. "
                                  "Acknowledge an accepted task before it starts; do not claim it already happened. "
                                  "This selection does not accept a separate social offer. Return the required reply JSON. "
                                  + _player_command_brief(snapshot["request_intent"])}})
            else:
                messages.append({"role": "user", "content": "Generate your own opening or acknowledgment from player_interaction. The player has not spoken a new line."})
        return messages
    if snapshot.get("request_kind") not in ("resident_turn", "conversation_closing"):
        return [{"role": "user", "content": payload}]
    payload["world_state"].pop("preceding_turns", None)
    payload["world_state"].get("reply_context", {}).pop("last_heard", None)
    # Accepted current speech is authoritative, including diagnosed speech when
    # validation is off. Historical-example filtering must not drop a live reply.
    preceding = snapshot.get("preceding_turns", [])
    messages = [{"role": "user", "content": payload}]
    for turn in preceding:
        role = "assistant" if turn["role"] == snapshot["speaker_role"] else "user"
        # Own past replies use the same speech-object shape as new output. The
        # listener speaks in plain words; neither message carries private effects.
        content = ({"parts": [{"text": turn["text"]}]}
                   if role == "assistant" else turn["text"])
        messages.append({"role": role, "content": content})
    if snapshot.get("_compact_rewrite"):
        reminder = (f"Writing request: now generate only {own_name(snapshot['speaker'])}'s next reply. "
                    "Answer the listener above from your own perspective in one or two fresh complete sentences. "
                    "Do not copy their speech, adopt their I/my, or recite planning notes. Use only your permitted names "
                    "and effects; quote an unfamiliar name only when asking who was mentioned. Return the required JSON.")
        if snapshot.get("_closing") or snapshot.get("request_kind") == "conversation_closing":
            reminder += " This is the final answer. End with a statement and ask no question."
        messages.append({"role": "user", "content": reminder})
    return messages


@dataclass(frozen=True)
class PreparedRequest:
    system_instruction: str
    messages: list[dict]
    response_schema: dict
    response_model: type


def prepare_request(message, snapshot, *, correction=""):
    # Context builders and chronological projection may remove fields. Never let
    # those operations alter a caller's snapshot or an accepted stage prefix.
    snapshot = deepcopy(snapshot)
    instruction, schema, context = _request_context(message, snapshot, correction)
    return PreparedRequest(instruction, _conversation_messages(context, snapshot), schema,
                           response_model(snapshot))
