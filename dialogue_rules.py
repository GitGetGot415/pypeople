"""Conservative checks connecting newly generated speech to conversation effects."""

import re
import unicodedata

from constants import SNAKE_NAME
from dialogue_text import split_sentences


def unquoted_speech(text):
    return re.sub(r'''"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)'[^'\n]+'(?!\w)''', " ", text)


def named_references(text):
    """Present narrator aliases consistently; never apply this to stored dialogue."""
    return re.sub(r"\bthe\s+(?:snake|python)\b", lambda _: SNAKE_NAME, text, flags=re.I)


def named_context(value):
    if isinstance(value, str):
        return named_references(value)
    if isinstance(value, list):
        return [named_context(item) for item in value]
    if isinstance(value, dict):
        return {key: item if key in ("preceding_turns", "recent_dialogue", "player_dialogue", "former_life", "Details")
                or key == "assertion" and value.get("kind") == "former_life" else named_context(item)
                for key, item in value.items()}
    return value


def name_pattern(name):
    return rf"\b{re.escape(name)}\b"


def validate_english(text, names=()):
    text = unquoted_speech(text)
    for name in names:
        text = re.sub(name_pattern(name), "", text, flags=re.I)
    letters = [unicodedata.name(c, "") for c in text if c.isalpha()]
    foreign = sum("LATIN" not in letter for letter in letters)
    if foreign >= 4 and foreign > len(letters) / 4:
        raise ValueError("Write the current reply in English, preserving the speaker's voice and the supplied facts.")


def validate_opinion_reason(reason, owner):
    # Self-focused explanations from old model output must not seed new opinions.
    if owner and re.search(name_pattern(owner) + r"['’]s\b", unquoted_speech(reason), re.I):
        raise ValueError("An opinion reason must explain the owner's view of their partner, not criticize the owner by name.")


_CONTENT_NOISE = set("""a an the and or but so if while to of in on at by for with from as into
    it its is are was were be been being do does did have has had that this these those
    i me my we us our you your he him his she her they them their one some any all no not
    own already now still just about after before then later here there what who how why
    would could should will can may might must very more much also only out up
    account accounts claim claims claimed heard hear told tell said say share sharing
    private firsthand sourced source information recollection story experience supplied
    remains remain unverified earlier current previous garden recorded agreed offering
    former life remember recall""".split())
_CONCEPTS = (
    {"authority", "control", "controls", "rule", "rules", "govern", "governs", "divine", "god", "messenger"},
    {"apple", "apples", "fruit"}, {"refuse", "refused", "refusal", "decline", "declined", "reject", "rejected"},
    {"give", "gave", "given", "hand", "handed", "receive", "received", "accept", "accepted"},
    {"promise", "promised", "agreement", "bargain"}, {"pardon", "pardoned", "spare", "spared", "mercy"},
)


def _content_words(text, names):
    for name in names:
        for part in name.split():
            text = re.sub(name_pattern(part), " ", text, flags=re.I)
    words = set(re.findall(r"[a-z]+", text.casefold())) - _CONTENT_NOISE
    for group in _CONCEPTS:
        if words & group:
            words.difference_update(group)
            words.add(min(group))
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words}


def mentions_account(text, assertion, names):
    expected = _content_words(assertion, names)
    return bool(expected) and len(expected & _content_words(text, names)) >= min(2, len(expected))


def validate_account_speech(text, assertion, *, names=(), speaker=None, listener=None, kind=None):
    """Require content evidence, rather than an ID or a promise to tell it later.

    This is a conservative vocabulary check, not a proof of semantic equivalence.
    """
    sentences = [s for s in split_sentences(text) if not re.search(
        r"\b(?:will|would|could|might|going to|want to|like to)\s+(?:\w+\s+){0,3}(?:share[d]?|tell|told|disclose[d]?)\b", s, re.I)]
    if kind == "former_life":
        references = split_sentences(assertion)
        source_words = _content_words(assertion, names)
        usable = []
        for sentence in sentences:
            past_event = re.search(r"\bi\s+(?:remember|recall|worked|kept|helped|saw|received|returned|organized|lost|saved)\b", sentence, re.I)
            simple_reaction = re.search(r"\bi\s+(?:remember|recall)\s+(?:it|that|this|those times)\s+(?:well|fondly|clearly)\b", sentence, re.I)
            if past_event and not simple_reaction and not source_words & _content_words(sentence, names):
                raise ValueError("The recollection invents an additional remembered event outside the selected account. Tell only its supplied memory and your present reaction.")
            reference = max(references, key=lambda s: len(_content_words(s, names) & _content_words(sentence, names)), default="")
            denies_event = re.search(r"\b(?:have not|haven['’]t|had not|hadn['’]t|did not|didn['’]t|never)\s+"
                                     r"(?:\w+\s+){0,2}(?:received|seen|kept|worked|met|gave|helped|returned)\b", sentence, re.I)
            if denies_event and not re.search(r"\b(?:not|never|couldn['’]t|didn['’]t|cannot|can['’]t)\b", reference, re.I):
                continue
            if re.search(r"\b(?:await|awaiting|wait for|waiting for|will receive|yet to receive)\b", sentence, re.I):
                continue
            usable.append(sentence)
        sentences = usable
    expected = _content_words(assertion, names)
    actual = _content_words(" ".join(sentences), names)
    missing_actor = speaker is not None and any(
        name not in (speaker, listener) and re.search(name_pattern(name), assertion, re.I)
        and not re.search(name_pattern(name.split()[0]), " ".join(sentences), re.I) for name in names if name)
    if missing_actor or not expected or len(expected & actual) < min(2, len(expected)):
        raise ValueError("The selected account is not actually told in this turn. Describe a concrete event or memory "
                         "from its assertion in your own words; discussing trust or promising to tell it later is insufficient.")


def validate_apple_promise(offer, acceptance, deadline, *, current_phase=None):
    offer, acceptance = unquoted_speech(offer), unquoted_speech(acceptance)
    giving = re.search(r"\bi\s+(?:(?:will|shall|can|could|promise to|intend to|plan to)\s+)?"
                      r"(?:give|bring|deliver|share|save|offer)\b|\bi['’]ll\s+(?:give|bring|deliver|share|save)\b|"
                      r"\bi(?:['’]ll|\s+(?:will|can))?\s+get you (?:an?|one) (?:apple|fruit)\b", offer, re.I)
    negative = r"\b(?:cannot|can['’]t|won['’]t|wouldn['’]t|will not|do not|don['’]t|decline|refuse|maybe|perhaps|might)\b"
    if not giving or not re.search(r"\b(?:apple|fruit)\b", offer, re.I) or re.search(negative, offer, re.I):
        raise ValueError("A recorded apple promise needs the giver's explicit spoken commitment to give an apple, not a request or unrelated discussion.")
    phase = ("day", "evening", "night")[deadline % 3]
    phase_words = {"day": r"day|daytime|morning", "evening": r"evening", "night": r"night|tonight"}[phase]
    day = re.search(r"\bday\s+(\d+)\b", offer, re.I)
    phase_text = re.sub(r"\bday\s+\d+\b", "", offer, flags=re.I)
    if (not re.search(rf"\b(?:{phase_words})\b", phase_text, re.I)
            or day and int(day[1]) != deadline // 3 + 1
            or current_phase is not None and re.search(r"\btomorrow\b", offer, re.I)
            and deadline // 3 != current_phase // 3 + 1):
        raise ValueError("State the apple promise's actual deadline in the offer; do not invent a different day or phase.")
    if ("?" in acceptance or re.search(negative, acceptance, re.I)
            or not re.search(r"\b(?:yes|agreed|deal|that works|i accept|i agree|thank you|thanks|i['’]ll take)\b", acceptance, re.I)):
        raise ValueError("The apple receiver must explicitly accept the spoken promise before it can be recorded.")


def validate_continuity(text, *, day, situation, knows_claims=False):
    speech = unquoted_speech(text)
    if day == 1 and re.search(r"\b(?:yesterday|(?:a few|several|\d+) days ago|last (?:night|week))\b", speech, re.I) and re.search(
            rf"{name_pattern(SNAKE_NAME)}|\b(?:garden|authority|claims)\b", speech, re.I):
        raise ValueError("This is day one. Refer to earlier today for garden events, preserving unknown former-life dates.")
    if situation.get("prior_snake_encounters", 0) and re.search(
            rf"\bi\s+(?:have not|haven['’]t|never)\s+(?:yet\s+)?(?:met|encountered|spoken to|dealt with)\s+"
            rf"(?:{re.escape(SNAKE_NAME)}|it|him|her|you)\b", speech, re.I):
        raise ValueError("The speaker has already met the named guard. Doubting authority does not erase that encounter.")
    if knows_claims and re.search(rf"{name_pattern(SNAKE_NAME)}|\bclaims\b", speech, re.I) and re.search(
            r"\bi['’]?(?:ve| have)?\s+(?:heard nothing|haven['’]t heard anything|have not heard anything)\b", speech, re.I):
        raise ValueError("The speaker has firsthand knowledge of the claims being made; preserve that experience while allowing skepticism.")
    if "prior_snake_encounters" in situation and not situation["prior_snake_encounters"] and not knows_claims:
        if re.search(rf"\bi\s+(?:have\s+)?(?:met|heard|talked to|spoken to|spoken with)\s+{re.escape(SNAKE_NAME)}\b", speech, re.I):
            raise ValueError("No earlier encounter or firsthand hearing is supplied for this speaker. Describe the present uncertainty without inventing a meeting.")


def validate_apple_roles(text, *, giver, receiver, speaker, owed_offering=False):
    speech = unquoted_speech(text)
    if speaker == receiver and re.search(
            r"\b(?:apple|fruit)\s+i\s+(?:offered|gave|handed|provided)\s+you\b|\bi\s+(?:gave|handed)\s+you\s+(?:an?|the|one)\s+(?:apple|fruit)\b|"
            r"(?:^|[.!?;,]\s+)(?:please\s+)?(?:take|accept)\s+(?:the|this|an?|one)\s+(?:apple|fruit)\b|"
            r"\bi(?:['’]ll|\s+(?:will|can|shall))\s+(?:give|hand|offer)\s+you\s+(?:an?|the|one|this)\s+(?:apple|fruit)\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}. Keep the original giver and receiver in their actual roles.")
    if speaker == giver and re.search(r"\bapple\s+you\s+(?:agreed|promised)\s+to\s+(?:give|provide)\b", speech, re.I):
        raise ValueError(f"{giver} provides the apple; {receiver} provides the offering, not another apple.")
    account = r"(?:private\s+)?(?:recollection|secret|information|endorsement|story|offering)"
    for sentence in split_sentences(speech):
        if re.search(r"\bnot\s+(?:owed|due)\s+to\s+me\b", sentence, re.I):
            continue
        reversed_price = (speaker == receiver and re.search(
            rf"\b{account}\b[^.!?]{{0,45}}\b(?:owed|due)\s+to\s+me\b|\byou\s+owe\s+me\s+(?:an?\s+|the\s+)?{account}\b", sentence, re.I)
            or speaker == giver and re.search(rf"\bi\s+owe\s+you\s+(?:an?\s+|the\s+)?{account}\b", sentence, re.I))
        if reversed_price or re.search(r"\bwe owe\s+(?:to\s+)?(?:each other|one another)\b", sentence, re.I):
            raise ValueError(f"{receiver} owes the recollection or endorsement to {giver}. This is not a mutual debt or information owed to the resident.")
    if owed_offering:
        for sentence in split_sentences(speech):
            if re.search(r"\b(?:not|never|don't|do not)\s+(?:\w+\s+){0,2}(?:owe|need to return)\b", sentence, re.I):
                continue
            if re.search(r"\bowe\s+(?:you\s+|me\s+)?(?:an?|the|one)\s+(?:apple|fruit)\b|"
                         r"\b(?:apple|fruit)\s+(?:you\s+(?:(?:already|previously)\s+)?received\s+)?(?:is|was|remains)\s+(?:(?:still|now)\s+)?owed\b|"
                         r"\b(?:i am|you are)\s+entitled to the apple\b|"
                         r"\b(?:obligation|obliged|required|must)\b[^.!?]{0,35}\breturn\s+(?:it|that|(?:the|an?)\s+(?:apple|fruit))\b", sentence, re.I):
                raise ValueError("The obligation is information or public support for an apple already received, not an apple repayment or return.")
