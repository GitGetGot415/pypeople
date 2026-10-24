"""Provider-independent speech, captured choices and accepted-prefix validation."""

import json
import re
import unicodedata
from copy import deepcopy
from types import SimpleNamespace
from validation_errors import (DialogueValidationError, RecoveryKind, DialogueRepetitionError,
                              DialogueSpeakerError)
from dialogue import (AcceptedProposal, ActionIntent, ActionResponse, ConversationPart, ConversationResponse,
                     ConversationTurn, DialoguePartsResponse, LeverageUse, PlayerIntent, ResidentAssessment,
                     ResidentGift, ResidentTurnResponse, SnakeDecision, SnakePlayerResponse,
                     SnakeResidentResponse, SnakeResponse, ThirdPartyOpinion)
from constants import (RESIDENT_TURN_COUNT, CONVERSATION_MAX_TURNS, DIALOGUE_REPEAT_PHRASE_WORDS,
                      DIALOGUE_REPEAT_SENTENCE_WORDS, DIALOGUE_REPEAT_TURN_WORDS,
                      PLAYER_LABEL,
                      PLAYER_NAME_TAG, SNAKE_NAME)
from dialogue_text import (contains_question, contains_serialized_dialogue, contains_speaker_label,
                          speech_is_complete, split_sentences)
from dialogue_rules import (is_name_reference, mentions_account, mentions_private_recollection, name_pattern,
                           unquoted_speech, validate_account_speech, validate_apple_roles,
                           validate_continuity, validate_base_prose, validate_claim_source, validate_english,
                           validate_opinion_reason, validate_environment_speech, validate_speaker_identity,
                           _personal_apple_request, _short_player_answer, validate_snake_perspective)
from social import phase_index, validate_agreements, resident_information, resident_effect_limits, proposals_from_snapshot
from actor_identity import (account_references, name_forms, actor_id, forms, own_name)
from social import (orientation_speech, validate_orientation)
from dialogue_context import (_actor_name, _snake_turn_snapshot, _validate_speaker)
from transfers import validate_player_transfer, validate_resident_gift


def snake_disclosure_accounts(snapshot):
    return snapshot.get("snake", {}).get("known_accounts", [])


def validate_snake_decision(result, snapshot):
    if ((result.interaction_id != snapshot.get("interaction_id", "") and not (isinstance(result, SnakePlayerResponse) and not result.interaction_id))
            or result.decision not in snapshot.get("permitted_decisions", ["talk"])):
        raise ValueError("Use this encounter's identity and a locally permitted decision.")
    if result.decision == "give_apple" and not snapshot["can_give_apple"]:
        raise ValueError("No apple is available for this exchange.")
    accounts = {a["claim_id"] for a in snapshot.get("offering_accounts", [])}
    if snapshot.get("request_kind") == "snake_decision" and result.account_id != (snapshot.get("disclosed_account") or {}).get("claim_id", ""):
        raise ValueError("Select only the account actually disclosed in this exchange.")
    if result.account_id and result.account_id not in accounts:
        raise ValueError("The resident can disclose only an account they actually know.")
    known = {a["claim_id"] for a in snake_disclosure_accounts(snapshot)}
    if result.snake_account_id and result.snake_account_id not in known:
        raise ValueError("The guard can discuss only a supplied known account.")
    if result.decision in ("warn", "eat"):
        speech = snapshot.get("_player_message", "") or " ".join(
            t["text"] for t in snapshot.get("heard_turns", []) if t["role"] == "recipient")
        if not result.accusation.strip() or not result.evidence.strip() or result.evidence not in speech:
            raise ValueError("An accusation requires an exact quote from this person's actual response.")
    if result.decision == "eat" and not snapshot.get("known_warning"):
        raise ValueError("Condemnation requires an earlier warning and a subsequent response.")
    if result.decision == "give_apple" and snapshot.get("request_kind") != "snake_player" and not result.account_id:
        speech = next((t["text"] for t in reversed(snapshot.get("heard_turns", [])) if t["role"] == "recipient"), "")
        if (not result.evidence.strip() or result.evidence not in speech
                or contains_question(result.evidence)
                or (_short_player_answer(result.evidence) and not snapshot.get("question_account_id"))):
            raise ValueError("An apple for a new present report requires an exact quote from the resident's accepted answer.")


def validate_snake_response(result, snapshot, *, dialogue_validation=None):
    if dialogue_validation is None:
        dialogue_validation = snapshot.get("_dialogue_validation", True)
    if result.outcome_turn >= len(result.turns):
        raise ValueError("The guard must perform the outcome after the resident's actual answer.")
    if result.decision == "give_apple" and (result.outcome_turn + 1 >= len(result.turns)
            or result.turns[result.outcome_turn + 1].speaker != "recipient"):
        raise ValueError("Let the resident accept or decline the offered apple before ending the exchange.")
    validate_snake_decision(result, {**snapshot, "heard_turns": [dict(role=t.speaker, text=t.text) for t in result.turns[:result.outcome_turn]]})
    if dialogue_validation:
        _validate_base_presentation(result, snapshot)
        validate_dialogue_format(result.turns)
        validate_dialogue_completion(result.turns)
        if len(result.turns) == CONVERSATION_MAX_TURNS and contains_question(result.turns[-1].text):
            raise ValueError("At the turn limit, finish with an answer or closing rather than a new question.")
    for i, turn in enumerate(result.turns):
        if turn.speaker != ("initiator" if i % 2 == 0 else "recipient"):
            raise ValueError("Alternate snake and resident speech, beginning with the snake.")
        if dialogue_validation:
            validate_snake_perspective(turn.text, snake_role="speaker" if turn.speaker == "initiator" else "listener")
            state = _snake_turn_snapshot(snapshot, result, result.turns[:i], outcome_turn=result.outcome_turn)
            if turn.speaker == "recipient" and "permitted_accounts" in state["speaker"]:
                _validate_response(json.dumps({"parts": [p.model_dump() for p in turn.parts],
                    "account_id": result.account_id if i == result.outcome_turn - 1 else ""}), state, SnakeResidentResponse)
            else:
                validate_speech_grounding(turn.text, state)
    if dialogue_validation:
        validate_dialogue_speakers(result.turns, {
            "initiator": snapshot["snake"]["name"], "recipient": snapshot["resident"]["npc_name"]})
        validate_dialogue_repetition(result.turns, actor_references=_repetition_references(snapshot))


def validate_dialogue_speakers(turns, names, *, start_index=0):
    errors, indices = [], []
    for index, turn in enumerate(turns, start_index):
        try:
            _validate_speaker(turn.text, names[turn.speaker], names["recipient" if turn.speaker == "initiator" else "initiator"])
        except ValueError as exc:
            errors.append(str(exc))
            indices.append(index)
    if errors:
        raise DialogueSpeakerError(" ".join(dict.fromkeys(errors)), indices)


def _speech_words(text):
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'").casefold()
    return re.findall(r"\w+(?:'\w+)*", text)


def _speech_phrases(text):
    # Match copied speech despite punctuation, casing, or display part breaks.
    words = _speech_words(text)
    size = DIALOGUE_REPEAT_PHRASE_WORDS
    return (tuple(words[index:index + size]) for index in range(len(words) - size + 1))


def _resident_repetition_facts(snapshot, response):
    """Stage and complete-response checks exempt the same disclosed facts."""
    disclosed = {entry.claim_id for entry in response.disclosures}
    facts = [option["assertion"] for option in (snapshot.get("_information_validation") or {}).get("options", [])
             if option["id"] in disclosed]
    facts.extend(event["text"] for event in (snapshot.get("shared_history") or {}).get("confirmed_events", []))
    return facts


def _repetition_references(snapshot):
    return {actor_id: [actor["name"], *forms(actor["self_name"]), *actor.get("references", [])]
            for actor_id, actor in snapshot.get("_name_actors", {}).items()}


def validate_dialogue_repetition(turns, recent_dialogue=(), *, factual_context=(), actor_references=None, participant_ids=None):
    """Reject copied speech before dialogue or its social effects are committed."""
    previous = {phrase for entry in recent_dialogue for _, text in entry["excerpts"]
                for phrase in _speech_phrases(text)}
    factual = {phrase for text in factual_context for phrase in _speech_phrases(text)}
    current, signatures, repeated, sentences_by_speaker, own_turns = {}, [], set(), set(), set()
    echoed_sentences = set()
    factual_sentences = {tuple(_speech_words(sentence)) for text in factual_context for sentence in split_sentences(text)}
    aliases = {}
    for actor, names in (actor_references or {}).items():
        for name in names:
            if name:
                name = name.casefold()
                marker = "actor:" + actor
                aliases[name] = marker if name not in aliases or aliases[name] == marker else None
    aliases = {name: marker for name, marker in aliases.items() if marker is not None}
    reference_pattern = "|".join(name_pattern(name) for name in sorted(aliases, key=len, reverse=True))

    def echo_words(text):
        # A name substitution does not make a copied reply a new response.
        if reference_pattern:
            text = re.sub(reference_pattern,
                lambda match: "person" if is_name_reference(match, text) else match[0], text, flags=re.I)
        return tuple(_speech_words(unquoted_speech(text)))

    def fact_tokens(text, role=None):
        text = unicodedata.normalize("NFKC", text).replace("\u2019", "'").casefold()
        references = [(m.start(), m.end(), aliases[m[0]]) for m in re.finditer(reference_pattern, text, re.I)
                      if is_name_reference(m, text)] if reference_pattern else []
        # Collapsing actor references keeps even a window beginning in
        # the middle of a long positional description aligned with its fact.
        tokens = []
        for match in re.finditer(r"\w+(?:'\w+)*", text):
            reference = next((entry for entry in references if entry[0] <= match.start() and match.end() <= entry[1]), None)
            word = match[0]
            if reference:
                word = reference[2] if match.start() == reference[0] else None
            elif word in {"i", "me", "my", "you", "your"}:
                owner = role if word in {"i", "me", "my"} else "recipient" if role == "initiator" else "initiator"
                known_id = (participant_ids or {}).get(owner) if role else None
                word = "actor:" + known_id if known_id else "actor"
            elif word in {"he", "him", "his", "she", "her", "they", "them", "their", "it", "its"}:
                word = "actor"
            elif word in {"and", "the", "a", "an"}:
                word = None
            tokens.append(word)
        return tokens

    fact_sequences = [tuple(word for word in fact_tokens(text) if word) for text in factual_context]
    fact_sentence_sequences = [tuple(word for word in fact_tokens(sentence) if word)
                               for text in factual_context for sentence in split_sentences(text)]

    def same_fact_words(words, sequence):
        return bool(words) and len(words) == len(sequence) and all(
            left == right or left == "actor" and right.startswith("actor:") or right == "actor" and left.startswith("actor:")
            for left, right in zip(words, sequence))

    def matches_fact(words):
        return bool(words) and any(same_fact_words(words, sequence[offset:offset + len(words)])
                                  for sequence in fact_sequences for offset in range(len(sequence) - len(words) + 1))

    copied = False
    for index, turn in enumerate(turns):
        # Joining parts catches loops split across multiple display pages too.
        phrases = list(_speech_phrases(turn.text))
        normalized = fact_tokens(turn.text, turn.speaker) if factual_context else []
        signature = tuple(_speech_words(turn.text))
        if (len(phrases) != len(set(phrases))
                or len(signature) >= DIALOGUE_REPEAT_PHRASE_WORDS and signature in signatures):
            repeated.add(index)
        if (len(signature) >= DIALOGUE_REPEAT_TURN_WORDS and (turn.speaker, signature) in own_turns
                and signature not in factual_sentences):
            repeated.add(index)
        own_turns.add((turn.speaker, signature))
        current_sentences = set()
        for sentence in split_sentences(turn.text):
            words = tuple(_speech_words(unquoted_speech(sentence)))
            factual_sentence = words in factual_sentences or bool(factual_context and any(
                same_fact_words(tuple(word for word in fact_tokens(sentence, turn.speaker) if word), sequence)
                for sequence in fact_sentence_sequences))
            key = (turn.speaker, words)
            echo = echo_words(sentence)
            if ((len(words) >= DIALOGUE_REPEAT_SENTENCE_WORDS and key in sentences_by_speaker
                    or len(words) >= DIALOGUE_REPEAT_TURN_WORDS and words in current_sentences)
                    and not factual_sentence):
                repeated.add(index)
            if len(words) >= DIALOGUE_REPEAT_SENTENCE_WORDS and echo in echoed_sentences and not factual_sentence:
                repeated.add(index)
            sentences_by_speaker.add(key)
            echoed_sentences.add(echo)
            current_sentences.add(words)
        for offset, phrase in enumerate(phrases):
            # Two speakers can refer to the same recorded fact while answering
            # each other. Repeating their prose or cycling through it is a loop.
            is_fact = phrase in factual or matches_fact(tuple(word for word in normalized[offset:offset + DIALOGUE_REPEAT_PHRASE_WORDS] if word))
            if phrase in current and (not is_fact or current[phrase] >= 2):
                repeated.add(index)
            if phrase in previous and not is_fact:
                repeated.add(index)
                copied = True
            current[phrase] = current.get(phrase, 0) + 1
        signatures.append(signature)
    if sum(map(len, signatures)) >= 2 * DIALOGUE_REPEAT_PHRASE_WORDS and any(
            tuple(signatures) == tuple(tuple(_speech_words(text)) for _, text in entry["excerpts"])
            for entry in recent_dialogue):
        repeated.update(range(len(turns)))
        copied = True
    if repeated:
        reason = ("Dialogue copies a substantial passage from a previous conversation." if copied else
                  "Dialogue repeats a substantial passage within this exchange.")
        indices = sorted(repeated)
        raise DialogueRepetitionError(
            f"{reason} Rewrite repeated turns {', '.join(map(str, indices))} (zero-based) with distinct replies.", indices,
        )


def player_reply_history(snapshot):
    """Only this speaker's recorded originals, never rewrites or player speech."""
    person = snapshot.get("snake", snapshot)
    speaker = own_name(person)
    history = [turn for exchange in snapshot.get("earlier_player_conversations", []) for turn in exchange["turns"]]
    history.extend(snapshot.get("player_dialogue", []))
    replies = []
    for turn in history:
        who, text = (turn["speaker"], turn["text"]) if isinstance(turn, dict) else turn
        if who == speaker:
            replies.append(text)
    return replies


def validate_player_reply_repetition(text, snapshot):
    """Check the new reply against accepted speech without revalidating old turns."""
    message = unquoted_speech(snapshot.get("_player_message", ""))
    if re.match(r"\s*(?:(?:can|could|would|will) you (?:please )?|please )?"
                r"(?:repeat\b|say (?:that|it) again\b|quote yourself\b|what did you (?:just )?say\b)", message, re.I):
        return
    speech = unquoted_speech(text).strip()
    if not speech:
        return
    person = snapshot.get("snake", snapshot)
    accounts = person.get("private_context", {}).get("known_accounts", person.get("known_accounts", []))
    facts = [account["assertion"] for account in accounts if account.get("assertion")]
    facts.extend(person.get("area_knowledge", {}).get("layout", {}).values())
    recent = [{"excerpts": [(own_name(person), unquoted_speech(reply))]} for reply in player_reply_history(snapshot)]
    try:
        validate_dialogue_repetition([ConversationTurn(speaker="initiator", parts=[ConversationPart(text=speech)])],
            recent, factual_context=facts, actor_references=_repetition_references(snapshot),
            participant_ids={"initiator": actor_id(person), "recipient": snapshot.get("player_id", "")})
    except DialogueRepetitionError:
        raise DialogueRepetitionError(
            "The new reply repeats a substantial passage. Respond to the player's latest message with fresh wording; "
            "do not repeat a question they just answered. Rewrite only the new reply, keeping accepted speech unchanged.", (0,)) from None


def dialogue_validation_errors(response, snapshot):
    """Diagnose accepted originals without another provider call or any effects."""
    try:
        _validate_response(response.model_dump_json(), {**snapshot, "_dialogue_validation": True}, type(response))
    except ValueError as exc:
        return [str(exc)]
    return []


def validate_dialogue_format(turns):
    for turn in turns:
        if contains_serialized_dialogue(turn.text):
            raise ValueError("The speech reads internal dialogue or account JSON aloud. Return spoken words without serialized records.")
        if contains_speaker_label(turn.text):
            raise ValueError("Return only directly spoken words, without narrator or speaker-name prefixes. The interface supplies the speaker label.")
        speech = unquoted_speech(turn.text)
        if re.search(r"\b(?:account_to_share|question_to_answer|preceding_turns|zero-based|new[_ ]apple[_ ]count)\b|\bunreserved\s*[:=]|"
                     r"\bconcrete event from (?:my|the|a supplied) (?:account|assertion)\b|"
                     r"\b(?:retain|store) this knowledge as\b|\bowner and source of this recollection\b|"
                     r"\b(?:this|the) (?:account|recollection) (?:is |was )?(?:owned by|sourced from)\b|"
                     r"\b(?:clos(?:e|ing)|end(?:ing)?|finish(?:ing)?) (?:this|the|our) (?:conversation|exchange) "
                     r"without (?:any )?(?:further |another |more )?questions\b", speech, re.I):
            raise ValueError("Keep internal writing instructions and resource field labels out of dialogue. Answer the listener and close naturally in character.")
        fields = re.findall(r'''["'](assertion|subject_relation|object_relation|knowledge_owner_name|original_speaker_relation|immediate_source_relation)["']\s*:''',
                            turn.text.replace('\\"', '"'))
        if len(set(fields)) >= 2:
            raise ValueError("The speech reads internal account JSON aloud. Tell the account's meaning in natural words, "
                             "preserving its subject, source and stance; never speak field names, braces or serialized records.")


def validate_speech_grounding(text, snapshot):
    """Check this speech before a partner sees it or its effects are committed."""
    speaker, listener = snapshot["speaker"], snapshot["listener"]
    from area_knowledge import validate_area_speech
    preceding = snapshot.get("preceding_turns", [])
    question = snapshot.get("_player_message", preceding[-1]["text"] if preceding else "")
    validate_area_speech(text, speaker.get("area_knowledge", {}), question=question)
    listener_name = snapshot.get("_listener_identity", _actor_name(listener))
    names = [_actor_name(speaker), listener_name, SNAKE_NAME, *snapshot.get("known_actor_names", [])]
    validate_english(text, names)
    situation = speaker.get("situation", {})
    firsthand = [*speaker.get("firsthand_events_with_listener", []),
                 *speaker.get("permitted_accounts", []), *speaker.get("known_accounts", [])]
    knows_claims = bool(situation.get("prior_snake_encounters", 0)) or any(
        a.get("kind") == "snake_claim" and a.get("basis") == "firsthand" for a in firsthand)
    validate_continuity(text, day=snapshot.get("day"), situation=situation, knows_claims=knows_claims,
                        listener_is_guard=listener_name == SNAKE_NAME)
    if knows_claims:
        validate_claim_source(text, names=names, speaker=_actor_name(speaker), listener=listener_name,
                              actor_references=account_references(snapshot.get("_name_actors", {})))
    transaction = snapshot.get("transaction")
    if transaction:
        validate_apple_roles(text, giver=transaction["apple_giver"], receiver=transaction["apple_receiver"],
                             speaker=_actor_name(speaker), apple_received=transaction.get("apple_already_received", False))
    account = snapshot.get("account_to_share")
    if account:
        account_names = [value for key, value in account.items() if key.endswith("_name") and value]
        validate_account_speech(text, account["assertion"], names=(*names, *account_names),
                                speaker=_actor_name(speaker), listener=listener_name, kind=account.get("kind"),
                                firsthand=account.get("basis") == "firsthand",
                                actor_references=account_references(snapshot.get("_name_actors", {})),
                                known_events=[event for event in firsthand if event.get("basis") == "firsthand"])
    if snapshot.get("request_kind") != "snake_turn":
        return
    for report in ([account] if account else []) + speaker.get("heard_accounts", []):
        validate_snake_account_roles(text, report, speaker, listener)
    if snapshot["speaker_role"] == "initiator":
        validate_immediate_snake_speech(text, transaction["selected_outcome"]
            if snapshot["turn_index"] == transaction.get("outcome_turn", 2) else "talk", listener=listener_name)
        for heard in speaker.get("known_accounts", []):
            if (heard.get("kind") == "former_life" and heard.get("subject_id") != speaker["actor_id"]
                    and mentions_private_recollection(text, heard["assertion"], names)
                    and re.search(r"\bi (?:remember|recall|worked|kept|helped|planted|repaired|used to)\b", unquoted_speech(text), re.I)):
                raise ValueError("The apple giver must not adopt the resident's recollection as its own.")


def validate_snake_account_roles(text, report, speaker, listener):
    if report.get("kind") == "snake_claim":
        hearer = report.get("object_name", "")
        if report.get("subject_id") != report.get("object_id") and hearer:
            reference = name_pattern(hearer)
            if re.search(rf"\b(?:{reference})(?:['\u2019]s claims?\b| (?:claims?|claimed|made claims?|asserts?|asserted)\b)", unquoted_speech(text), re.I):
                raise ValueError("The account's listener heard the guard's claims; do not make that listener the claimant.")
        claimant = report.get("subject_id")
        if claimant == speaker.get("actor_id") and re.search(
                r"\byour claims? about (?:authority|the garden)\b|\byour claim to divine authority\b", unquoted_speech(text), re.I):
            raise ValueError("You made these claims yourself. Use my claims, not your claims about the listener.")
        if claimant == listener.get("actor_id") and re.search(
                r"\bmy claims? about (?:authority|the garden)\b|\bmy claim to divine authority\b", unquoted_speech(text), re.I):
            raise ValueError("The listener made these claims. Address the guard's claims as your claims; do not adopt them as my claims.")


def validate_immediate_snake_speech(text, outcome, *, listener=""):
    speech = unquoted_speech(text).replace("\u2019", "'")
    if re.search(r"\b(?:you owe me|offering (?:is |still )?owed|pay me later|endorsement owed)\b", speech, re.I):
        raise ValueError("This is an immediate exchange; no offering or debt remains.")
    # Direct address can interrupt a handoff: "Here, Judy Judge, is one apple."
    for name in sorted({listener, listener.split()[0] if listener else ""}, key=len, reverse=True):
        if name:
            speech = re.sub(rf",\s*{name_pattern(name)}\s*,", " ", speech, flags=re.I)
    apple = r"(?:your|an?|one|the|this|that)(?:\s+one)?\s+apple"
    handoff_pattern = (rf"\b(?:here(?: is|'s)\s+{apple}|take\s+{apple}|"
        rf"i\s+(?:give|am giving|hand|am handing)\s+you\s+{apple}|"
        rf"i(?: am|'m)?\s+(?:hand(?:ing)? over|giving)\s+{apple}(?=\s*(?:now\b|[.!;,]|$))|"
        rf"i(?: am|'m)\s+(?:giving|handing)(?: over)?\s+{apple}\s+to you|"
        rf"i'm\s+(?:giving|handing)\s+you\s+{apple}|i hand(?: over)?\s+{apple}\s+to you)\b")
    handoff = False
    for sentence in split_sentences(speech):
        # A question, future offer or refusal does not perform a present handoff.
        if contains_question(sentence):
            continue
        for clause in sentence.split(";"):
            if re.search(r"\b(?:if|once|tomorrow|later)\b", clause, re.I):
                continue
            for match in re.finditer(handoff_pattern, clause, re.I):
                if not re.search(r"\b(?:not|never|don't|won't|cannot|can't|would|could|will|i'll|refuse to|decline to)\s+"
                                 r"(?:\w+\s+){0,2}$", clause[:match.start()], re.I):
                    handoff = True
    if outcome != "give_apple" and handoff:
        raise ValueError("No apple is transferred by this reply.")
    if outcome == "give_apple" and not handoff:
        raise ValueError("Explicitly hand over the apple now in the generated speech.")
    if outcome == "warn" and not re.search(r"\b(?:warn|eat|eaten|devour|condemn)\b", speech, re.I):
        raise ValueError("Make the warning and physical danger explicit and allow an answer.")
    if outcome == "eat" and not re.search(r"\b(?:eat|eaten|devour|condemn)\b", speech, re.I):
        raise ValueError("Announce the selected condemnation.")


def validate_resident_grounding(result, snapshot):
    _validate_base_presentation(result, snapshot)
    info = snapshot.get("_information_validation") or {}
    options = {a["id"]: a for a in info.get("options", [])}
    names = [snapshot[role]["npc_name"] for role in ("initiator", "recipient")]
    names.extend(actor["name"] for actor in snapshot.get("_name_actors", {}).values())
    names.append(SNAKE_NAME)
    for role in ("initiator", "recipient"):
        private = snapshot[role].get("private_context", {})
        names.extend(value for account in private.get("permitted_accounts", private.get("known_accounts", []))
                     for key, value in account.items() if key.endswith("_name") and isinstance(value, str) and value)
    known = {role: {options[b["claim_id"]]["fact_id"] for b in info.get("beliefs", {}).get(role, [])
                     if b["claim_id"] in options} for role in ("initiator", "recipient")}
    for index, turn in enumerate(result.turns):
        speaker = snapshot[turn.speaker]
        validate_environment_speech(turn.text, location=speaker.get("location", "garden"))
        other_role = "recipient" if turn.speaker == "initiator" else "initiator"
        private = speaker.get("private_context", {})
        validate_speech_grounding(turn.text, {
            "_name_actors": snapshot.get("_name_actors", {}),
            "speaker": {"name": speaker["npc_name"], "situation": private.get("situation", {}),
                        "firsthand_events_with_listener": private.get("permitted_accounts", private.get("known_accounts", []))},
            "listener": snapshot["recipient" if turn.speaker == "initiator" else "initiator"]["npc_name"],
            "day": speaker.get("day"), "known_actor_names": names})
        # General beliefs can also be formed in the current discussion. Match
        # concrete recollections separately, including sentences in older accounts.
        private_accounts = [{**a, "assertion": sentence} for a in options.values() if a["kind"] == "former_life"
                            for sentence in split_sentences(a["assertion"])
                            if not re.search(r"\b(?:believe[sd]?|beliefs|like[sd]? to think)\b", sentence, re.I)
                            and mentions_private_recollection(turn.text, sentence, names)]
        own_memories = [a for a in private_accounts if a["subject_id"] == speaker["npc_id"]]
        disclosed = {options[d.claim_id]["fact_id"] for d in result.disclosures if d.turn == index and d.claim_id in options}
        for account in private_accounts:
            own_support = any(mentions_account(a["assertion"], account["assertion"], names) for a in own_memories)
            if account["fact_id"] not in known[turn.speaker] and not own_support:
                raise ValueError("The speaker uses another person's private account before learning it. Keep each biography and motive with its owner.")
            if account["subject_id"] != speaker["npc_id"] and not own_support and re.search(
                    r"\bi\s+(?:once\s+)?(?:worked|kept|helped|was an?|used to|remember working|recall working)\b", unquoted_speech(turn.text), re.I):
                raise ValueError("The speaker adopts another person's remembered life as their own. Respond to that person's account without changing its owner.")
            if (account["subject_id"] == speaker["npc_id"] and account["fact_id"] not in known[other_role]
                    and account["fact_id"] not in disclosed):
                raise DialogueValidationError("A new private recollection is spoken without its disclosure entry. Record the actual account on this turn or omit its private details.", RecoveryKind.MISSING_DISCLOSURE)
        known[other_role].update(disclosed)
    for role in ("initiator", "recipient"):
        owner, entry = snapshot[role], getattr(result, role + "_opinion", None)
        if entry is None:  # Incomplete prefixes have no final assessments yet.
            continue
        validate_english(entry.reason, (owner["npc_name"], entry.person))
        validate_opinion_reason(entry.reason, owner["npc_name"], entry.person)
    for entry in getattr(result, "third_party_opinions", []):
        owner = snapshot[result.turns[entry.turn].speaker]["npc_name"]
        target = snapshot.get("_name_actors", {}).get(entry.subject_id, {}).get("name", "")
        validate_english(entry.reason, (owner, target))
        validate_opinion_reason(entry.reason, owner, target)


def validate_dialogue_completion(turns, *, start_index=0):
    incomplete = [index for index, turn in enumerate(turns, start_index) if not speech_is_complete(turn.text)]
    if incomplete:
        raise ValueError(
            f"Speech ends without finishing a sentence in turn(s) {', '.join(map(str, incomplete))}. "
            "Finish the current speaker's sentence and end with sentence punctuation before the partner replies. "
            "Rewrite a shorter complete sentence if needed; never clip speech to meet a size target.")


def _validate_base_presentation(result, snapshot):
    names = {SNAKE_NAME}
    people = [snapshot, *[snapshot[key] for key in ("initiator", "recipient", "resident", "snake", "speaker", "listener")
                         if isinstance(snapshot.get(key), dict)],
              *snapshot.get("other_npcs", [])]
    for person in people:
        name = person.get("npc_name", person.get("name", person.get("character", {}).get("Name", "")))
        if name:
            names.update((name, name.split()[0]))
    texts = []
    if isinstance(result, (ConversationResponse, SnakeResponse)):
        texts.extend(turn.text for turn in result.turns)
    elif isinstance(result, DialoguePartsResponse):
        texts.append(" ".join(part.text for part in result.parts))
    elif isinstance(result, (ActionResponse, SnakePlayerResponse)):
        texts.append(result.reply)
    if isinstance(result, ConversationResponse):
        texts.extend((result.initiator_opinion.reason, result.recipient_opinion.reason))
        texts.extend(entry.reason for entry in result.third_party_opinions)
        texts.extend(part.text for part in result.handoff_thanks)
    elif isinstance(result, ResidentTurnResponse):
        texts.extend(entry.reason for entry in result.third_party_opinions)
    elif isinstance(result, ResidentAssessment):
        texts.append(result.opinion.reason)
    elif isinstance(result, ActionResponse):
        texts.extend(entry.reason for entry in result.opinion_updates)
    if isinstance(result, (SnakeDecision, SnakeResidentResponse)) and result.attitudes is not None:
        texts.extend(value for key, value in result.attitudes.model_dump().items() if key.endswith("_reason"))
    for text in texts:
        validate_base_prose(text, names)
        validate_environment_speech(text, location=snapshot.get("location", snapshot.get("initiator", {}).get("location", "garden")))


def _validate_response(raw, snapshot, model):
    result = model.model_validate_json(raw)
    check_dialogue = snapshot.get("_dialogue_validation", True)
    if check_dialogue:
        _validate_base_presentation(result, snapshot)
    if isinstance(result, ResidentAssessment):
        _apply_resident_candidate(snapshot["_baseline"], snapshot["_draft"], result, snapshot)
        return result
    if isinstance(result, SnakeResidentResponse):
        accounts = snapshot["speaker"]["permitted_accounts"]
        if result.account_id and result.account_id not in {a["claim_id"] for a in accounts}:
            raise ValueError("Select only an account this resident actually knows, or leave account_id empty.")
        snapshot = {**snapshot, "account_to_share": next((a for a in accounts if a["claim_id"] == result.account_id), None)}
        if check_dialogue:
            text = " ".join(p.text for p in result.parts)
            for account in accounts:
                if (account["kind"] in ("former_life", "private_motive") and account["claim_id"] != result.account_id
                        and mentions_private_recollection(text, account["assertion"], name_forms(snapshot.get("_name_actors", {})))):
                    raise DialogueValidationError("The spoken recollection needs its actual account_id, not an unrelated selection or an empty field.", RecoveryKind.MISSING_DISCLOSURE)
    if isinstance(result, DialoguePartsResponse):
        current = ConversationTurn(speaker=snapshot.get("speaker_role", "initiator"), parts=result.parts)
        if not check_dialogue:
            if isinstance(result, ResidentTurnResponse):
                _apply_resident_candidate(snapshot["_baseline"], snapshot["_draft"], result, snapshot)
            return result
        validate_dialogue_format([current])
        validate_speech_grounding(current.text, snapshot)
        if snapshot.get("request_kind") == "handoff_thanks":
            draft = deepcopy(snapshot["_draft"])
            draft["turns"].append(current)
            if not draft.get("gift"):
                validate_resident_grounding(SimpleNamespace(**draft), snapshot["_baseline"])
        validate_dialogue_completion([current], start_index=snapshot.get("turn_index", 0))
        names = {current.speaker: _actor_name(snapshot["speaker"]),
                 "recipient" if current.speaker == "initiator" else "initiator": snapshot.get("_listener_identity", _actor_name(snapshot["listener"]))}
        validate_dialogue_speakers([current], names, start_index=snapshot.get("turn_index", 0))
        for role in ("speaker", "listener"):
            actor = snapshot.get(role)
            if isinstance(actor, dict) and actor.get("kind") == "python":
                validate_snake_perspective(current.text, snake_role=role)
        if (snapshot.get("request_kind") == "conversation_closing" or snapshot.get("_closing")) and any(
                contains_question(part.text) for part in result.parts):
            raise ValueError("Answer the last question and close without asking another question.")
        if isinstance(result, ResidentTurnResponse):
            projected = _apply_resident_candidate(snapshot["_baseline"], snapshot["_draft"], result, snapshot)
            facts = _resident_repetition_facts(snapshot["_baseline"], SimpleNamespace(**projected))
        else:
            account = snapshot.get("account_to_share")
            facts = [account["assertion"]] if account else snapshot.get("_factual_context", [])
        if snapshot.get("request_kind") in ("snake_turn", "conversation_closing", "resident_turn"):
            preceding = [ConversationTurn(speaker=turn["role"], parts=[ConversationPart(text=turn["text"])])
                         for turn in snapshot.get("_preceding_turns", snapshot["preceding_turns"])]
            validate_dialogue_repetition([*preceding, current], snapshot.get("shared_history", {}).get("recent_dialogue", []),
                                        factual_context=facts, actor_references=_repetition_references(snapshot),
                                        participant_ids={role: actor_id(snapshot[who]) for role, who in
                                            ((current.speaker, "speaker"), ("recipient" if current.speaker == "initiator" else "initiator", "listener"))
                                            if isinstance(snapshot.get(who), dict)})
        return result
    if isinstance(result, SnakeResponse):
        validate_snake_response(result, snapshot)
        return result
    if isinstance(result, SnakePlayerResponse):
        from snake import validate_player_snake_decision
        if "permitted_decisions" in snapshot and "decision" not in result.model_fields_set:
            raise ValueError("Return the immediate decision and generated reply together.")
        validate_player_snake_decision(result, snapshot)
        if check_dialogue:
            validate_player_reply_repetition(result.reply, snapshot)
            validate_english(result.reply, (SNAKE_NAME,))
            from area_knowledge import validate_area_speech
            validate_area_speech(result.reply, snapshot.get("snake", {}).get("area_knowledge", {}))
            validate_dialogue_format([ConversationTurn(speaker="initiator", parts=[ConversationPart(text=result.reply)])])
            validate_snake_perspective(result.reply, snake_role="speaker")
            _validate_speaker(result.reply, SNAKE_NAME, PLAYER_LABEL)
            validate_immediate_snake_speech(result.reply, result.decision, listener=PLAYER_LABEL)
            if result.snake_account_id:
                account = next(a for a in snake_disclosure_accounts(snapshot) if a["claim_id"] == result.snake_account_id)
                validate_account_speech(result.reply, account["assertion"], speaker=SNAKE_NAME, listener=PLAYER_LABEL,
                                        kind=account["kind"], names=name_forms(snapshot.get("_name_actors", {})))
                validate_snake_account_roles(result.reply, account, snapshot["snake"], {"actor_id": snapshot["player_id"]})
        return result
    if isinstance(result, SnakeDecision):
        validate_snake_decision(result, snapshot)
        return result
    if isinstance(result, PlayerIntent):
        # Observations and questions cannot issue commands, even if the model
        # also supplied an action or irrelevant recipient/item fields.
        result = (ActionIntent(action="SAY") if result.message_type != "COMMAND" or _short_player_answer(snapshot.get("_player_message", "")) else
                  ActionIntent.model_validate(result.model_dump(exclude={"message_type"})))
        # A first-person apple request has no third-party recipient. Resolve its
        # pronoun before the reply is generated or any gift is authorized.
        if result.action == "GIVE_ITEM" and _personal_apple_request(snapshot.get("_player_message", "")):
            result.target = PLAYER_LABEL
        if result.action == "TALK_TO_NPC" and result.target.strip().casefold() == PLAYER_LABEL.casefold():
            # The player is already here talking. No new trip or task is needed.
            result = ActionIntent(action="SAY")
        if (result.action == "GIVE_ITEM" and result.item != "fruit" and snapshot.get("resolved_work_id") is not None
                and result.item != snapshot["resolved_work_id"]):
            result = ActionIntent(action="SAY")
            result._clarification = "The work reference is ambiguous. Ask which exact owned work is intended."
        from player_commands import normalize_action
        try:
            normalized = normalize_action(result, snapshot, verbal=True)
            normalized._clarification = result._clarification
            result = normalized
        except ValueError as exc:
            result = ActionIntent(action="SAY")
            result._clarification = str(exc) + " Ask for clarification without promising an action."
    if isinstance(result, ActionIntent):
        intent = snapshot.get("request_intent")
        if isinstance(result, ActionResponse) and intent:
            result._authorized_intent = intent.copy()
        declined_gift = (isinstance(result, ActionResponse) and intent and intent["action"] == "GIVE_ITEM"
                         and result.action == "SAY" and not result.target and not result.item and result.quantity == 0)
        if intent and not declined_gift and (result.action != intent["action"] or result.target.strip().casefold() != intent["target"].casefold()
                       or result.item != intent["item"] or result.quantity != intent["quantity"]):
            raise ValueError("The action, recipient, item, or quantity differs from the current request.")
        if isinstance(result, ActionResponse) and not result.reply.strip():
            raise ValueError("Empty reply")
        if isinstance(result, ActionResponse):
            validate_player_transfer(result, snapshot)
            if check_dialogue:
                validate_speaker_identity(result.reply, own_name(snapshot), PLAYER_NAME_TAG)
        if isinstance(result, ActionResponse) and check_dialogue:
            validate_player_reply_repetition(result.reply, snapshot)
            validate_speech_grounding(result.reply, {
                "_name_actors": snapshot.get("_name_actors", {}),
                "_player_message": snapshot.get("_player_message", ""),
                "speaker": {"name": snapshot.get("npc_name", ""),
                            "area_knowledge": snapshot.get("area_knowledge", {}),
                            "situation": snapshot.get("private_context", {}).get("situation", {})},
                "listener": "You", "day": snapshot.get("day"),
            })
            validate_dialogue_format([ConversationTurn(speaker="initiator", parts=[ConversationPart(text=result.reply)])])
            for entry in result.opinion_updates:
                validate_opinion_reason(entry.reason, snapshot.get("npc_name", snapshot.get("character", {}).get("Name", "")), entry.person)
        if (check_dialogue and isinstance(result, ActionResponse) and result.end_conversation
                and contains_question(result.reply)):
            raise ValueError(
                "A closing reply cannot ask a question. For SAY, either use end_conversation false so the player "
                "can answer, or keep it true and finish without questions. Other actions must keep "
                "end_conversation true and acknowledge the task without questions."
            )
        from player_commands import normalize_action
        normalized = normalize_action(result, snapshot)
        if (result.target, result.item, result.quantity) != (normalized.target, normalized.item, normalized.quantity):
            raise ValueError("Use the normalized recipient identity and possession from the request.")
    elif isinstance(result, ConversationResponse):
        validate_resident_gift(result.gift, result.turns, snapshot)
        resident_information(snapshot, vars(result), incomplete=False)
        validate_agreements(result.accepted_proposals, proposals_from_snapshot(snapshot),
                            result.turns, snapshot["initiator"]["npc_id"], snapshot["recipient"]["npc_id"],
                            current_phase=phase_index(snapshot["initiator"].get("day", 1),
                                                      snapshot["initiator"].get("time_of_day", "day")),
                            dialogue_validation=check_dialogue)
        if check_dialogue:
            validate_dialogue_format(result.turns)
            validate_dialogue_completion(result.turns)
            validate_resident_grounding(result, snapshot)
        history = snapshot.get("shared_history") or {}
        if snapshot.get("purpose") == "food_request":
            outcome = snapshot["food_request"]["outcome"]
            if (result.accepted_proposals or len(result.turns) not in (2, 3) or result.food_outcome != outcome
                    or bool(result.handoff_thanks) != (outcome == "accepted")):
                raise ValueError("Dialogue must respect the food decision and reserve thanks for a successful handoff.")
            if check_dialogue and len(result.turns) == 3 and (not contains_question(result.turns[-2].text)
                                          or contains_question(result.turns[-1].text)):
                raise ValueError("A third food-request turn may only answer a closing question.")
            if check_dialogue and result.handoff_thanks:
                thanks = ConversationTurn(speaker="initiator", parts=result.handoff_thanks)
                validate_dialogue_format([thanks])
                validate_dialogue_completion([thanks])
                validate_english(thanks.text, people)
                validate_resident_grounding(SimpleNamespace(turns=[*result.turns, thanks], disclosures=result.disclosures), snapshot)
                _validate_speaker(" ".join(part.text for part in result.handoff_thanks),
                                  snapshot["initiator"]["npc_name"], snapshot["recipient"]["npc_name"])
        elif len(result.turns) < RESIDENT_TURN_COUNT or result.food_outcome or (result.handoff_thanks and not (result.gift and result.gift.accept_turn is not None)):
            raise ValueError("Chats need at least four turns and cannot include a food outcome or handoff thanks.")
        if check_dialogue and len(result.turns) == CONVERSATION_MAX_TURNS and contains_question(result.turns[-1].text):
            raise ValueError("At the turn limit, finish with an answer or closing rather than a new question.")
        for entry, partner in (
            (result.initiator_opinion, snapshot["recipient"]),
            (result.recipient_opinion, snapshot["initiator"]),
        ):
            if entry.person.strip().casefold() != partner["npc_name"].casefold():
                raise ValueError("Conversation opinions must refer to the other participant.")
        if check_dialogue:
            validate_dialogue_speakers(result.turns, {role: snapshot[role]["npc_name"] for role in ("initiator", "recipient")})
            validate_dialogue_repetition(result.turns, history.get("recent_dialogue", []),
                                        factual_context=_resident_repetition_facts(snapshot, result),
                                        actor_references=_repetition_references(snapshot),
                                        participant_ids={role: actor_id(snapshot[role]) for role in ("initiator", "recipient")})
        validate_orientation(result, snapshot)
    return result


def _apply_resident_candidate(snapshot, draft, candidate, state):
    """Return a validated copy; a rejected attempt cannot alter accepted state."""
    projected = deepcopy(draft)
    role, index = state["speaker_role"], state["turn_index"]
    pending = {entry["disclosure_turn"]: entry for entry in state["pending_reactions"]}
    if len({r.disclosure_turn for r in candidate.reactions}) != len(candidate.reactions):
        raise ValueError("React once to each pending disclosure.")
    if {r.disclosure_turn for r in candidate.reactions} != set(pending):
        raise ValueError("Supply your own reaction to every pending heard disclosure, and no others.")
    for reaction in candidate.reactions:
        if reaction.stance not in pending[reaction.disclosure_turn]["stances"]:
            raise ValueError("Your stance must respect your firsthand evidence and prior beliefs.")
        entry = next(d for d in projected["disclosures"] if d.turn == reaction.disclosure_turn)
        entry.listener_stance = reaction.stance
    if isinstance(candidate, ResidentAssessment):
        other = "recipient" if role == "initiator" else "initiator"
        if candidate.opinion.person != snapshot[other]["npc_name"]:
            raise ValueError("Assess only your conversation partner using the supplied opinion record name.")
        if snapshot.get("_dialogue_validation", True):
            validate_english(candidate.opinion.reason, (own_name(snapshot[role]), own_name(snapshot[other])))
            validate_base_prose(candidate.opinion.reason, (own_name(snapshot[role]), own_name(snapshot[other])))
            validate_opinion_reason(candidate.opinion.reason, snapshot[role]["npc_name"], snapshot[other]["npc_name"])
        projected["opinions"][role] = candidate.opinion
        resident_information(snapshot, projected)
        return projected

    turn = ConversationTurn(speaker=role, parts=candidate.parts)
    projected["turns"].append(turn)
    if candidate.gift_offer:
        if projected["gift"] is not None or snapshot.get("arrival_focus") or snapshot.get("purpose") == "food_request":
            raise ValueError("Only one gift offer is available in an ordinary conversation.")
        projected["gift"] = ResidentGift(**candidate.gift_offer.model_dump(), giver_id=actor_id(snapshot[role]), offer_turn=index)
    if candidate.accept_gift:
        gift = projected["gift"]
        if gift is None or gift.recipient_id != actor_id(snapshot[role]) or gift.accept_turn is not None:
            raise ValueError("Accept only an earlier pending gift addressed to you.")
        gift.accept_turn = index
    validate_resident_gift(projected["gift"], projected["turns"], snapshot)
    choices = state["choices"]
    for field in ("offers", "acceptances"):
        entries = getattr(candidate, field)
        if len(set(entries)) != len(entries) or any(entry not in choices[field] for entry in entries):
            raise ValueError("Choose only available proposals for your own offer or an earlier giver's offer.")
    for proposal_id in candidate.offers:
        if proposal_id in projected["offers"]:
            raise ValueError("This proposal has already been offered.")
        if snapshot.get("_dialogue_validation", True):
            # Validate the giver's authorization now; acceptance must still be
            # genuinely spoken on a later turn before an agreement is assembled.
            from dialogue_rules import validate_apple_offer
            proposal = next(p for p in snapshot["permitted_proposals"] if p["id"] == proposal_id)
            validate_apple_offer(turn.text, proposal["deadline"], current_phase=phase_index(
                snapshot[role].get("day", 1), snapshot[role].get("time_of_day", "day")))
        projected["offers"][proposal_id] = index
    for proposal_id in candidate.acceptances:
        projected["accepted_proposals"].append(AcceptedProposal(
            proposal_id=proposal_id, offer_turn=projected["offers"][proposal_id], accept_turn=index))
    for entry in candidate.disclosures:
        if entry.claim_id not in choices["disclosures"]:
            raise ValueError("Disclose only your own permitted or previously heard accounts.")
        projected["disclosures"].append(SimpleNamespace(turn=index, claim_id=entry.claim_id,
                                                        intent=entry.intent, listener_stance=None))
    if bool((snapshot.get("_information_validation") or {}).get("focus_claim_id")) and index == 0:
        info = snapshot["_information_validation"]
        options = {o["id"]: o for o in info["options"]}
        focus_fact = options[info["focus_claim_id"]]["fact_id"]
        if not any(options[d.claim_id]["fact_id"] == focus_fact for d in candidate.disclosures):
            raise ValueError("Disclose the required task account or a permitted denial of that same fact.")
    for entry in candidate.third_party_opinions:
        if entry.claim_id not in choices["third_party_opinions"]:
            raise ValueError("Assess only an eligible account you heard earlier.")
        projected["third_party_opinions"].append(ThirdPartyOpinion(turn=index, **entry.model_dump()))
        if snapshot.get("_dialogue_validation", True):
            validate_base_prose(entry.reason, [p["name"] for p in snapshot.get("_name_actors", {}).values()])
    for entry in candidate.leverage:
        if entry.claim_id not in choices["leverage"]:
            raise ValueError("Leverage requires a permitted account known before this turn.")
        projected["leverage"].append(LeverageUse(turn=index, **entry.model_dump()))
    limits = resident_effect_limits()
    for field in ("accepted_proposals", "disclosures", "third_party_opinions", "leverage"):
        limit = limits[field]
        if len(projected[field]) > limit:
            raise ValueError(f"The complete conversation exceeds its {field} limit.")
    if len(projected["offers"]) > limits["offers"]:
        raise ValueError("The complete conversation exceeds its offer limit.")
    resident_information(snapshot, projected)
    validate_agreements(projected["accepted_proposals"], proposals_from_snapshot(snapshot), projected["turns"],
                        snapshot["initiator"]["npc_id"], snapshot["recipient"]["npc_id"],
                        current_phase=phase_index(snapshot[role].get("day", 1), snapshot[role].get("time_of_day", "day")),
                        dialogue_validation=snapshot.get("_dialogue_validation", True))
    if snapshot.get("_dialogue_validation", True):
        validate_resident_grounding(SimpleNamespace(**projected), snapshot)
        if snapshot.get("arrival_focus", False) and index == 0:
            if not orientation_speech([turn.text]):
                raise ValueError("Open this orientation exchange with an arrival or surroundings concern; routine apple goals are deferred.")
            if snapshot.get("arrival_welcome") and re.search(
                    r"\bi(?:['’]ve| have)? already (?:told|explained|described|shown)\b",
                    unquoted_speech(turn.text), re.I):
                raise ValueError("This is Judy's first conversation with this newcomer. Explain the layout now; do not claim you already told them in an earlier conversation.")
            if snapshot.get("arrival_welcome") and not (
                    (re.search(r"\bbed\b", turn.text, re.I)
                     and re.search(r"\b(?:doorway|door|exit)\b", turn.text, re.I)
                     and re.search(r"\b(?:south|hallway)\b", turn.text, re.I))
                    or (re.search(r"\belevator\b", turn.text, re.I)
                        and re.search(r"\bgarden exit\b", turn.text, re.I)
                        and re.search(r"\b(?:first floor|floor (?:1|one))\b", turn.text, re.I))
                    or (re.search(r"\bbridge\b", turn.text, re.I)
                        and re.search(r"\bgarden\b", turn.text, re.I)
                        and re.search(r"\b(?:apartment entrance|building entrance)\b", turn.text, re.I))):
                raise ValueError("Give concrete layout information in this welcome: describe the room's bed and exit, or explain the elevator and floor-one garden exit. A greeting, saying you walked around, or offering to explain later is insufficient.")
    return projected
