"""Provider calls, response validation and the bounded correction loop."""

import json
import logging
import re
import validation_errors
from copy import deepcopy
from time import monotonic

from pydantic import ValidationError

import dialogue_context
import dialogue_validation
from actor_identity import account_references, actor_id, name_forms, speech_only
from constants import (CONVERSATION_MAX_TURNS, DIALOGUE_REPEAT_PHRASE_WORDS, DIALOGUE_STAGE_ATTEMPTS,
    DIALOGUE_TIMEOUT_SECONDS, FOOD_REQUEST_TURN_COUNT, MODEL_DEFAULT_CONTEXT_TOKENS, MODEL_DIALOGUE_CONTEXT_TOKENS,
    OPINION_CHANGES, RESIDENT_TURN_COUNT, VOICE_REWRITE_TEMPERATURE)
from dialogue import ActionResponse, ConversationResponse, ConversationTurn, DialoguePartsResponse, DisclosureChoice
from dialogue_rules import copied_share, mentions_private_recollection, states_no_change, validate_account_speech
from dialogue_text import contains_question
from generation_errors import InvalidResponseError, RequestError, StreamRepetitionError
from llm_contracts import intent_snapshot, prepare_request, response_model
from model_debug import admitted_provider, check_cancelled, provider_wait_seconds, record_output
from model_transport import TransportSettings, provider_errors, request_gemini, request_ollama
from ollama_manager import DEFAULT_OLLAMA_MODEL, mark_ollama_ready
from information import resident_information
from validation_errors import RecoveryKind
from voice_rewrite import (parse_voice_rewrite, retain_voice_rewrite, voice_rewrite_batches,
    voice_rewrite_snapshot)

logger = logging.getLogger(__name__)


def request_voice_rewrite(provider, api_key, response, snapshot, *, deadline,
                          ollama_model=DEFAULT_OLLAMA_MODEL, progress=None):
    """Interpret every full voice brief through the same bounded optional model pass."""
    rewritten = []
    wait_before = provider_wait_seconds()

    def available_rewrites():
        if not any(parts is not None for parts in rewritten):
            return []
        return [parts if parts is not None else [part["text"] for part in turn["parts"]]
                for parts, turn in zip(rewritten, state["turns"])]

    try:
        state = voice_rewrite_snapshot(response, snapshot)
        if state is None:
            return []
        rewritten = [None] * len(state["turns"])
        for indices, batch in voice_rewrite_batches(state):
            remaining = deadline + provider_wait_seconds() - wait_before - monotonic()
            if remaining <= 0:
                logger.warning("Voice rewrite deadline expired; using originals for unavailable voices.")
                return available_rewrites()
            batch["_timeout"] = remaining
            name = next(iter(batch["speakers"].values()))["name"]
            if progress is not None:
                progress(f"Rewriting {name}'s voice")
            if provider not in ("gemini", "ollama"):
                return available_rewrites()

            def generate(correction=""):
                return _provider_request(provider, api_key, "Rewrite the supplied speech.", batch, ollama_model,
                                         correction=correction)
            try:
                try:
                    result = generate()
                except StreamRepetitionError as exc:
                    # Only an interrupted generation gets a retry. Completed rewrite
                    # content/identity diagnostics remain nonblocking and never retry.
                    remaining = deadline + provider_wait_seconds() - wait_before - monotonic()
                    if remaining <= 0:
                        raise
                    batch["_timeout"] = remaining
                    if progress is not None:
                        progress(f"Correcting looping voice generation for {name}")
                    result = generate(correction=str(exc))
            except Exception as exc:
                logger.warning("Voice rewrite unavailable for %s; using that speaker's originals: %s", name, exc)
                record_output("validation", f"Voice rewrite unavailable for {name}; using that speaker's originals: {exc}")
                continue
            if monotonic() > deadline + provider_wait_seconds() - wait_before:
                logger.warning("Voice rewrite arrived after its deadline; using originals for unavailable voices.")
                return available_rewrites()
            checked = retain_voice_rewrite(result.model_dump_json(), batch["turns"], speakers=batch["speakers"])
            for index, parts in zip(indices, checked.turns):
                rewritten[index] = parts
        return available_rewrites()
    except Exception as exc:
        # Presentation failure must never discard an accepted original or its effects.
        logger.warning("Voice rewrite unavailable; using originals for unavailable voices: %s", exc)
        record_output("validation", f"Voice rewrite unavailable; using originals for unavailable voices: {exc}")
        return available_rewrites()


def validate_response(raw, snapshot):
    if snapshot.get("request_kind") in ("create_art", "create_music"):
        from creations import validate_creation
        try:
            return validate_creation(snapshot["request_kind"].removeprefix("create_"), raw)
        except (ValidationError, ValueError, TypeError) as exc:
            reason = str(exc.errors(include_input=False, include_url=False)) if isinstance(exc, ValidationError) else str(exc)
            raise InvalidResponseError("Invalid creation data: " + reason) from None
    if snapshot.get("request_kind") == "voice_rewrite":
        return parse_voice_rewrite(raw, snapshot["turns"], speakers=snapshot["speakers"])
    model = response_model(snapshot)
    try:
        result = dialogue_validation._validate_response(raw, snapshot, model)
        if not snapshot.get("_dialogue_validation", True):
            result._dialogue_errors = dialogue_validation.dialogue_validation_errors(result, snapshot)
            for error in result._dialogue_errors:
                record_output("validation", f"Base dialogue failed validation; retaining dialogue and effects: {error}")
        if isinstance(result, ActionResponse):
            for entry in result.opinion_updates:
                matches = [actor for actor in snapshot.get("other_npcs", [])
                           if actor.get("kind") != "player" and actor["name"].casefold() == entry.person.strip().casefold()]
                if len(matches) != 1 or not matches[0].get("actor_id"):
                    raise ValueError("Opinion targets must identify one permitted resident.")
                result._opinion_targets[entry.person] = matches[0]["actor_id"]
        return result
    except validation_errors.DialogueTurnError as exc:
        raise InvalidResponseError(str(exc), kind=getattr(exc, "kind", RecoveryKind.GENERAL), response=model.model_validate_json(raw),
                                   repeated_turns=exc.turn_indices if isinstance(exc, validation_errors.DialogueRepetitionError) else (),
                                   repair_turns=exc.turn_indices,
                                   failed_turns=_generated_dialogue(raw, snapshot)) from None
    except ValidationError as exc:
        issue = exc.errors(include_input=False, include_url=False)[0]
        field = issue["loc"][0] if issue["loc"] and issue["loc"][0] in model.model_fields else "response"
        raise InvalidResponseError(
            f"Invalid {field}: use the required JSON fields, value limits, and alternating speaker order.",
            failed_turns=_generated_dialogue(raw, snapshot),
        ) from None
    except ValueError as exc:
        # These validators supply application-authored reasons, never provider text.
        raise InvalidResponseError(str(exc), kind=getattr(exc, "kind", RecoveryKind.GENERAL), response=model.model_validate_json(raw),
                                   failed_turns=_generated_dialogue(raw, snapshot)) from None


def _generated_dialogue(raw, snapshot):
    """Retain only readable speech fields from a rejected response, never its effects."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    values = data.get("turns", [])
    if snapshot.get("request_kind") in ("conversation_closing", "resident_turn", "handoff_thanks") and "parts" in data:
        values = [{"speaker": snapshot.get("speaker_role", "initiator"), "parts": data["parts"],
}]
    if not isinstance(values, list):
        return []
    turns = []
    for value in values:
        try:
            turns.append(ConversationTurn.model_validate(value))
        except (ValidationError, ValueError, TypeError):
            break
    return turns


def _deduplicate_dialogue_parts(turn):
    """Keep generated words while dropping exact repeated parts or long sentences."""
    parts, seen_parts, seen_sentences = [], set(), set()
    for part in turn.parts:
        signature = tuple(dialogue_validation._speech_words(part.text))
        if signature in seen_parts:
            continue
        seen_parts.add(signature)
        sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", part.text.strip()):
            signature = tuple(dialogue_validation._speech_words(sentence))
            if len(signature) >= DIALOGUE_REPEAT_PHRASE_WORDS:
                if signature in seen_sentences:
                    continue
                seen_sentences.add(signature)
            sentences.append(sentence)
        if sentences:
            parts.append(part.model_copy(update={"text": " ".join(sentences)}))
    return parts


def _resident_draft():
    """Private worker state. Pending stances are never a saved/provider contract."""
    return dict(turns=[], offers={}, gift=None, accepted_proposals=[], disclosures=[],
                third_party_opinions=[], leverage=[], opinions={}, assessments={})


def _assemble_resident(snapshot, draft, thanks):
    resident_information(snapshot, draft, incomplete=False)
    return ConversationResponse(turns=draft["turns"],
        gift=draft.get("gift"), escalation=draft.get("escalation"),
        initiator_opinion=draft["opinions"]["initiator"], recipient_opinion=draft["opinions"]["recipient"],
        accepted_proposals=draft["accepted_proposals"],
        disclosures=[DisclosureChoice(**vars(d)) for d in draft["disclosures"]],
        third_party_opinions=draft["third_party_opinions"], leverage=draft["leverage"],
        food_outcome=snapshot["food_request"]["outcome"] if snapshot.get("purpose") == "food_request" else "",
        handoff_thanks=thanks)


def _speech_retry_snapshot(snapshot):
    """Rewrite only this speech step, retaining its chosen action, sources and full validation context."""
    state = deepcopy(snapshot)
    state["_compact_rewrite"] = state["request_kind"] != "handoff_thanks"
    if "choices" in state:
        state["choices"] = {**state["choices"], "offers": [], "acceptances": []}
    if state["request_kind"] == "resident_turn":
        return state  # Keep the conversation messages and unanswered questions on every attempt.
    state["preceding_turns"] = state["preceding_turns"][-1:]
    if state["request_kind"] == "conversation_closing" or state.get("_closing"):
        # Retain the listener's actual words as well as the separate question.
        state["task"] = ("State your intended next step based on the existing agreement. "
                         if state["speaker"].get("existing_agreements") else
                         "State your answer to question_to_answer using settled facts and your own views. ")
        state["task"] += "This is your final reply; the listener will not respond."
    return state


def transport_settings(snapshot, *, correction=""):
    options = {
        "num_ctx": MODEL_DIALOGUE_CONTEXT_TOKENS if snapshot.get("player_id") or snapshot.get("request_kind") in (
            "resident_turn", "resident_assessment", "handoff_thanks") else MODEL_DEFAULT_CONTEXT_TOKENS,
        "num_predict": -1,
    }
    if snapshot.get("request_kind") == "voice_rewrite":
        options["temperature"] = VOICE_REWRITE_TEMPERATURE
    elif snapshot.get("request_kind") in ("resident_turn", "resident_assessment", "handoff_thanks", "conversation_closing"):
        # Greedy decoding encouraged small models to replay the supplied history.
        # Corrections need fresh wording too; intent routing stays deterministic.
        options["temperature"] = 0.7
    elif correction or snapshot.get("request_kind") in ("intent", "opinion_direction"):
        options["temperature"] = 0
    return TransportSettings(
        min(DIALOGUE_TIMEOUT_SECONDS, snapshot.get("_timeout", DIALOGUE_TIMEOUT_SECONDS)),
        VOICE_REWRITE_TEMPERATURE if snapshot.get("request_kind") == "voice_rewrite" else
        0 if snapshot.get("request_kind") == "intent" else None,
        options)


@admitted_provider
def _provider_request(provider, api_key, message, snapshot, ollama_model, *, correction=""):
    if provider not in ("gemini", "ollama"):
        raise RequestError("Choose an AI provider in Settings.")
    prepared = prepare_request(message, snapshot, correction=correction)
    settings = transport_settings(snapshot, correction=correction)
    record_output("call", snapshot.get("request_kind", "reply"))
    if correction:
        record_output("validation", correction)
    with provider_errors(provider, ollama_model):
        raw = (request_gemini(api_key, prepared, settings) if provider == "gemini" else
               request_ollama(prepared, settings, ollama_model))
        result = validate_response(raw, snapshot)
        if provider == "ollama":
            mark_ollama_ready()
        return result


def _stage_correction(state, exc):
    """Construct correction text without changing accepted speech or recovery choices."""
    kind = state.get("request_kind")
    # No accepted prefix, decision, stance or offer can be rewritten.
    # Resident errors can originate in a full-prefix check. Only send
    # current-speaker errors; those checks never quote private accounts.
    correction = (f"The previous response was rejected: {exc} Return fresh complete JSON for only this stage. "
                  "Correct the error using your supplied choices and own knowledge. "
                  "Keep accepted preceding speech and effects unchanged. Write shorter complete sentences. "
                  "Do not read internal records aloud or write the listener's answer.")
    if kind == "opinion_direction":
        return f"The previous choice was rejected: {exc} Read the reason again and return only the change JSON."
    if kind == "resident_assessment":
        owner, target = state["speaker"]["name"], state["opinion_target"]
        correction = (f"You are {owner}. Assess only {target}, your conversation partner. "
            f"Write one short first-person reason beginning with I about how {target}'s words affected your view. "
            f"Do not describe {owner}'s personality or write {target}'s opinion of you. "
            "Use the fixed transcript as evidence; keep the target and pending reaction choices unchanged. "
            f"The previous assessment failed: {exc}. Return only the corrected assessment JSON.")
    if exc.repeated_turns and kind in ("resident_turn", "conversation_closing"):
        earlier = [t["text"] for t in state.get("_preceding_turns", []) if t["role"] == state["speaker_role"]]
        correction += ("\nYour own accepted earlier speech, supplied only to avoid repeating it: "
            + json.dumps(earlier, ensure_ascii=False)
            + " Respond to the listener's latest point with a new answer, decision or specific follow-up. "
            "Do not repeat your earlier question, even in a shortened form.")
    if exc.repeated_turns and isinstance(exc.response, ActionResponse):
        correction += ("\nYour already spoken replies, supplied only to avoid copying them: "
            + json.dumps(dialogue_validation.player_reply_history(state), ensure_ascii=False)
            + "\nRespond to the current player message: " + json.dumps(state.get("_player_message", ""), ensure_ascii=False)
            + ". React to its new information and use fresh wording. Keep the action, recipient, possession, "
              "quantity and other authorized effects consistent with the request. Do not rewrite accepted history.")
    if (isinstance(exc.response, ActionResponse) and exc.kind == RecoveryKind.KNOWN_ENCOUNTER
            and state.get("private_context", {}).get("starting_experience")):
        correction = ("Your reply denied an established conversation. Regenerate this reply from these facts: "
                      + state["private_context"]["starting_experience"]
                      + " Correct any earlier mistaken denial and answer the player's latest question. "
                      "Write your own speech, not these instructions. Keep accepted history and effects unchanged.")
    return correction


def _resident_disclosure_choices(state, speech):
    """Retain only resident account choices supported by this rejected reply."""
    disclosure_choices = []
    for account in state["speaker"].get("permitted_accounts", []):
        if account["claim_id"] not in state["choices"]["disclosures"]:
            continue
        try:
            validate_account_speech(speech, account["assertion"],
                names=name_forms(state.get("_name_actors", {})), speaker=state["speaker"]["name"],
                listener=state.get("_listener_identity"), kind=account["kind"],
                firsthand=account.get("basis") == "firsthand",
                actor_references=account_references(state.get("_name_actors", {})))
        except ValueError:
            continue
        disclosure_choices.append(account["claim_id"])
    return disclosure_choices


def _corrected_request(state, request, report):
    kind = state.get("request_kind")
    activity = (f"turn {state['turn_index'] + 1} for {state['speaker']['name']}" if kind == "resident_turn" else
                f"assessment for {state['speaker']['name']}" if kind == "resident_assessment" else
                f"opinion direction for {state['owner']}" if kind == "opinion_direction" else
                {"conversation_closing": "closing answer",
                 "handoff_thanks": "reserved handoff thanks", "intent": "player intent"}.get(kind, "reply"))
    correction = ""
    corrections = []
    last_error = None
    disclosure_repair = None
    disclosure_choices = None
    effect_repairs = {}
    for attempt in range(DIALOGUE_STAGE_ATTEMPTS):
        report(("Generating " if attempt == 0 else "Correcting " if attempt == 1 else "Recovery: rewriting ") + activity)
        retry_state = (_speech_retry_snapshot(state) if attempt == DIALOGUE_STAGE_ATTEMPTS - 1
                       and "speaker_role" in state and issubclass(response_model(state), DialoguePartsResponse) else state)
        if disclosure_repair:
            account = next(a for a in state["speaker"]["permitted_accounts"] if a["claim_id"] == disclosure_repair)
            retry_state = {**retry_state, "_disclosure_repair": disclosure_repair,
                           "speaker": {**retry_state["speaker"], "account_focus": account}}
        if disclosure_choices is not None:
            retry_state = {**retry_state, "choices": {**retry_state["choices"], "disclosures": disclosure_choices}}
        if effect_repairs:
            retry_state = {**retry_state, "choices": {**retry_state["choices"], **effect_repairs}}
        try:
            result = request(retry_state, correction)
            if kind == "intent":
                # Transport validation reduces PlayerIntent to ActionIntent;
                # that accepted value intentionally has no message_type field.
                return result
            # A compact retry retains the complete prefix and original rules.
            return validate_response(result.model_dump_json(), state)
        except InvalidResponseError as exc:
            last_error = exc
            # A chosen, concrete own memory needs speech and metadata together.
            # Constrain only its regenerated reply; infer no live disclosure.
            if kind == "resident_turn" and exc.kind == RecoveryKind.MISSING_DISCLOSURE:
                speech = " ".join(p.text for p in getattr(exc.response, "parts", []))
                memories = [account for account in state["speaker"].get("permitted_accounts", [])
                            if account["kind"] == "former_life" and account["subject_id"] == actor_id(state["speaker"])
                            and account.get("basis") == "firsthand" and account["claim_id"] in state["choices"]["disclosures"]
                            and mentions_private_recollection(speech, account["assertion"], ())]
                if len(memories) == 1:
                    disclosure_repair = memories[0]["claim_id"]
                    disclosure_choices = None
            if exc.repeated_turns and isinstance(exc.response, DialoguePartsResponse):
                candidate = exc.response.model_copy(deep=True)
                candidate.parts = _deduplicate_dialogue_parts(candidate)
                try:
                    return validate_response(candidate.model_dump_json(), state)
                except InvalidResponseError:
                    pass
            correction = _stage_correction(state, exc)
            if kind == "resident_turn" and exc.kind == RecoveryKind.MISSING_PROMISE:
                effect_repairs["offers"] = []
                correction = ("Your speech made no supported apple commitment. Regenerate this reply without offers; "
                              "the schema now provides no new offer for this stage. Respond naturally using your own knowledge. "
                              "Keep earlier accepted speech, agreements and other permitted effects unchanged.")
            if kind == "resident_turn" and exc.kind == RecoveryKind.MISSING_ACCEPTANCE:
                effect_repairs["acceptances"] = []
                correction = ("Your speech did not accept an apple commitment. Regenerate this reply without acceptances; "
                              "the schema now provides none for this stage. Keep earlier accepted speech and effects unchanged.")
            if (kind == "resident_turn" and exc.kind == RecoveryKind.UNMATCHED_ACCOUNT
                    and not state.get("discussion_focus_claim_id")):
                disclosure_repair = None  # An optional memory may be withdrawn on an unaccepted retry.
                speech = " ".join(p.text for p in getattr(exc.response, "parts", []))
                disclosure_choices = _resident_disclosure_choices(state, speech)
                correction = ("Your disclosure metadata did not match what you said. Regenerate only this reply. "
                              "For an ordinary reaction with no account, omit disclosures and introduce no new story. "
                              "If your reply actually reports an available account, use only its matching claim ID. "
                              "Keep accepted speech and the listener's reaction unchanged; never invent events to justify a tag.")
            if correction not in corrections:
                corrections.append(correction)
            correction = "\n".join(corrections)
    raise RequestError(f"The model's reply failed validation after bounded correction: {last_error}")

def _exchange_turns(snapshot, draft, request_turn, retain, *, budget, limit, continuation=None):
    """Advance only accepted prefixes; family callbacks own stages and effects."""
    for index in range(limit):
        check_cancelled()
        forced = continuation is not None and continuation(index)
        if index >= budget and not forced and (not snapshot.get("_dialogue_validation", True)
                or not contains_question(draft["turns"][-1].text)):
            break
        request_turn(index, index == limit - 1)
        check_cancelled()
        retain(draft["turns"], snapshot)


def _complete_exchange(result, snapshot, report):
    check_cancelled()
    report("Validating completed conversation")
    return validate_response(result.model_dump_json(), snapshot)


def _resident_exchange(snapshot, request, retain, report):
    draft = _resident_draft()
    budget = FOOD_REQUEST_TURN_COUNT if snapshot.get("purpose") == "food_request" else RESIDENT_TURN_COUNT
    limit = FOOD_REQUEST_TURN_COUNT + 1 if snapshot.get("purpose") == "food_request" else CONVERSATION_MAX_TURNS
    if snapshot.get("arrival_welcome"):
        limit = budget
    def turn(index, closing):
        state = dialogue_context._resident_turn_snapshot(snapshot, draft, closing=closing)
        speech = request(state)
        accepted = dialogue_validation._apply_resident_candidate(snapshot, draft, speech, state)
        draft.update(accepted)

    _exchange_turns(snapshot, draft, turn, retain, budget=budget, limit=limit)
    def direction(state, role, target, reason, labels, you=""):
        # A finished reason is classified on its own; small models ignore a label chosen alongside long prose.
        if len(labels) == 1:
            return labels[0]  # Already settled, as for a believed accusation from the python.
        if states_no_change(reason):
            return "unchanged"
        try:
            return request(dialogue_context.opinion_direction_snapshot(
                state, target, reason, labels=labels, people=dialogue_validation._reason_people(snapshot, role), you=you)).change
        except RequestError:
            return None  # No direction could be found; recorded as unchanged.

    # The python's view of its partner follows the partner's verdict on its accusation, so it is assessed last.
    roles = ("recipient", "initiator") if snapshot["initiator"].get("kind") == "python" else ("initiator", "recipient")
    for role in roles:
        state = dialogue_context._resident_turn_snapshot(snapshot, draft, role=role, assessment=True)
        try:
            assessment = request(state)
        except RequestError:
            # A verdict whose written reason never passed its checks must not discard the whole accepted exchange.
            assessment = dialogue_validation.fallback_assessment(snapshot, state)
            draft = dialogue_validation._apply_resident_candidate({**snapshot, "_dialogue_validation": False}, draft, assessment, state)
            verdict = state.get("snake_verdict")
            settled = [dialogue_validation.gossip_labels(snapshot, draft, role, entry.claim_id) for entry in assessment.third_party_opinions]
            draft = dialogue_validation.apply_opinion_directions(snapshot, draft, state, verdict["change"] if verdict else "unchanged",
                                                                 [labels[0] if len(labels) == 1 else "unchanged" for labels in settled])
            continue
        draft = dialogue_validation._apply_resident_candidate(snapshot, draft, assessment, state)
        stored = (state["speaker"].get("opinion_of_listener") or {}).get("reason", "")
        verdict = state.get("snake_verdict")
        # Restating the view already held is no new reason to move it again.
        change = (verdict["change"] if verdict else
                  "unchanged" if stored and copied_share(assessment.opinion.reason, [stored]) >= 0.8 else
                  direction(state, role, state["opinion_target"], assessment.opinion.reason, list(OPINION_CHANGES)) or "unchanged")
        partner = snapshot["recipient" if role == "initiator" else "initiator"]["name"]
        gossip = [direction(state, role, snapshot["_name_actors"][entry.subject_id]["name"], entry.reason,
                            dialogue_validation.gossip_labels(snapshot, draft, role, entry.claim_id), you=partner)
                  for entry in assessment.third_party_opinions]
        draft = dialogue_validation.apply_opinion_directions(snapshot, draft, state, change, gossip)
    thanks = []
    if snapshot.get("purpose") == "food_request" and snapshot["food_request"]["outcome"] == "accepted":
        state = dialogue_context._resident_turn_snapshot(snapshot, draft, role="initiator", thanks=True)
        thanks = request(state).parts
    elif draft.get("gift") and draft["gift"].accept_turn is not None:
        role = "recipient" if draft["gift"].recipient_id == actor_id(snapshot["recipient"]) else "initiator"
        state = dialogue_context._resident_turn_snapshot(snapshot, draft, role=role, thanks=True)
        state["task"] = "Generate a brief acknowledgment reserved for AFTER the accepted possession is actually handed over. Preserve giver, recipient and creator roles."
        thanks = request(state).parts
    result = _assemble_resident(snapshot, draft, thanks)
    return _complete_exchange(result, snapshot, report)


def _individual_reply(snapshot, request):
    kind = snapshot.get("request_kind")
    if speech_only(snapshot):
        snapshot["request_intent"] = dict(action="SAY", target="", item="", quantity=0)
        return request(snapshot)
    if response_model(snapshot) is ActionResponse and kind not in ("player_opening", "player_gift", "player_acknowledgment", "player_command"):
        intent = request(intent_snapshot(snapshot))
        snapshot["request_intent"] = intent.model_dump()
        if intent._clarification:
            snapshot["intent_clarification"] = intent._clarification
    if kind == "player_command":
        from player_commands import normalize_action
        intent = normalize_action(snapshot["player_interaction"]["command"], snapshot)
        snapshot["request_intent"] = intent.model_dump()
    return request(snapshot)


def request_response(provider, api_key, message, snapshot, *, ollama_model=DEFAULT_OLLAMA_MODEL, progress=None):
    """One worker owns every sequential stage; drafts have no live effects."""
    if snapshot.get("request_kind") in ("create_art", "create_music"):
        from constants import CREATIVE_ATTEMPTS, CREATIVE_REQUEST_SECONDS
        deadline = monotonic() + CREATIVE_REQUEST_SECONDS
        correction = ""
        for attempt in range(CREATIVE_ATTEMPTS):
            check_cancelled()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RequestError("Creation generation timed out.")
            if progress:
                progress("Creating" if attempt == 0 else "Correcting creation data")
            state = {**snapshot, "_timeout": remaining}
            try:
                result = _provider_request(provider, api_key, message, state, ollama_model, correction=correction)
                check_cancelled()
                if monotonic() > deadline:
                    raise RequestError("Creation generation timed out.")
                return result
            except InvalidResponseError as exc:
                correction = f"The creation was invalid: {exc}. Return a complete corrected work within the fixed rules."
        raise RequestError("The model could not produce a valid creation after one correction.")
    generated = []

    def report(stage):
        check_cancelled()
        if progress is not None:
            progress(stage)

    def retain(turns, state):
        if not turns:
            return
        if "speaker_role" in state:
            index = state.get("turn_index", 0)
            generated[index:index + 1] = turns
        else:
            generated[:] = turns

    def request(state, correction=""):
        check_cancelled()
        prompt = ("Perform only the current speaker's task in world_state." if "speaker_role" in state else
                  message)
        try:
            result = _provider_request(provider, api_key, prompt, state, ollama_model, correction=correction)
            check_cancelled()
        except InvalidResponseError as exc:
            retain(exc.failed_turns, state)
            raise
        if isinstance(result, DialoguePartsResponse):
            retain([ConversationTurn(speaker=state["speaker_role"], parts=result.parts)], state)
        return result

    def validated_request(state):
        return _corrected_request(state, request, report)

    try:
        kind = snapshot.get("request_kind")
        if kind == "conversation":
            return _resident_exchange(snapshot, validated_request, retain, report)

        return _individual_reply(snapshot, validated_request)
    except (RequestError, ValueError) as exc:
        if not isinstance(exc, RequestError):
            exc = RequestError(f"The completed conversation failed validation: {exc}")
        exc.failed_turns = list(generated)
        raise exc from None
