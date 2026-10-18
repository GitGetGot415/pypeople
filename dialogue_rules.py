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
    if owner and re.search(name_pattern(owner) + r"(?:['’]s\b|\s+(?:kept|seemed|was|appeared|dismissed|ignored|refused)\b)",
                           unquoted_speech(reason), re.I):
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
_APPLE_REFUSAL = (r"\b(?:refused|declined|rejected)\s+(?:to accept\s+)?(?:the|an?|that)\s+(?:upfront\s+)?apple\b|"
                  r"\brefusal (?:to accept|of) (?:the|an?|that) apple\b")


def _content_words(text, names):
    for name in names:
        for part in name.split():
            text = re.sub(name_pattern(part), " ", text, flags=re.I)
    # Single-letter fragments of contractions and possessives (he's, dog's)
    # are not evidence that a private memory was mentioned.
    words = {word for word in re.findall(r"[a-z]+", text.casefold()) if len(word) > 1} - _CONTENT_NOISE
    for group in _CONCEPTS:
        if words & group:
            words.difference_update(group)
            words.add(min(group))
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words}


def mentions_account(text, assertion, names):
    expected = _content_words(assertion, names)
    return bool(expected) and len(expected & _content_words(text, names)) >= min(2, len(expected))


def _asserts(pattern, text):
    """Match an assertion without treating an explicit nearby negation as assent."""
    for match in re.finditer(pattern, text, re.I):
        if not re.search(r"\b(?:not|never|no|didn['’]t|don['’]t|won['’]t|haven['’]t)\s+"
                         r"(?:(?:a|the)\s+)?"
                         r"(?:(?:ever|already|yet|actually|really)\s+){0,2}$", text[:match.start()], re.I):
            return True
    return False


def validate_account_speech(text, assertion, *, names=(), speaker=None, listener=None, kind=None):
    """Require content evidence, rather than an ID or a promise to tell it later.

    This is a conservative vocabulary check, not a proof of semantic equivalence.
    """
    speech = unquoted_speech(text)
    if "offering remained unpaid" in assertion.casefold() and not assertion.startswith("An account disputes:"):
        if (_asserts(_APPLE_REFUSAL, speech)
                or not re.search(r"\b(?:offering|information|recollection|memory|endorsement|support)\b", text, re.I)):
            raise ValueError("This account concerns an unpaid offering after an apple was received, not refusal of the apple. Preserve what was refused.")
        subject = assertion.partition(" refused the agreed ")[0]
        if " refused the agreed " in assertion and speaker is not None and subject not in (speaker, listener):
            actor = rf"(?:{name_pattern(subject)}|{name_pattern(subject.split()[0])})"
            if not re.search(actor + r"\s*,?\s+(?:(?:she|he|they|has|had|have|did|not|never|already|also|then|still|to)\s+){0,4}"
                             r"(?:refuse[d]?|decline[d]?|reject(?:ed)?|receive[d]?|accept(?:ed)?|took|got|take|said|told|won['’]t|wouldn['’]t)\b",
                             text, re.I):
                raise ValueError("Keep the selected account's subject with its action. Describe that person's receipt or refusal; mentioning their name elsewhere does not let the reporter adopt the event as I/my.")
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
    receipt = situation.get("received_snake_apple")
    if receipt is False and re.search(
            rf"\bi(?:['’]ve|\s+have)?\s+(?:already\s+)?(?:received|accepted|got)\s+(?:(?:an?|one|the) apple|one|fruit)\s+"
            rf"(?:(?:upfront|earlier)\s+)?from\s+{name_pattern(SNAKE_NAME)}|"
            rf"{name_pattern(SNAKE_NAME)}\s+(?:already\s+)?(?:gave|handed)\s+me\s+(?:an? apple|one|fruit)\b", speech, re.I):
        raise ValueError("The speaker has not received an apple from the named guard. Do not adopt another resident's apple transaction.")
    if receipt and re.search(r"\bi heard (?:he|she|they) (?:gave|handed) me (?:an?|one) apple\b", speech, re.I):
        raise ValueError("Receiving the apple is the speaker's own experience, not hearsay. Preserve the actual receipt and distinguish it from refusing the later offering.")
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


def validate_apple_roles(text, *, giver, receiver, speaker, owed_offering=False,
                         apple_received=False, offering_pending=False):
    speech = unquoted_speech(text)
    if speaker == receiver and re.search(
            r"\b(?:apple|fruit)\s+i\s+(?:offered|gave|handed|provided)\s+you\b|\bi\s+(?:gave|handed)\s+you\s+(?:an?|the|one)\s+(?:apple|fruit)\b|"
            r"(?:^|[.!?;,]\s+)(?:please\s+)?(?:take|accept)\s+(?:the|this|an?|one)\s+(?:apple|fruit)\b|"
            r"\bi(?:['’]ll|\s+(?:will|can|shall))\s+(?:give|hand|offer)\s+you\s+(?:an?|the|one|this)\s+(?:apple|fruit)\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}. Keep the original giver and receiver in their actual roles.")
    if speaker == giver and re.search(r"\bapple\s+you\s+(?:agreed|promised)\s+to\s+(?:give|provide)\b", speech, re.I):
        raise ValueError(f"{giver} provides the apple; {receiver} provides the offering, not another apple.")
    if speaker == giver and re.search(r"\bi\s+(?:accept|receive|take)\s+(?:the|an?|your|this)\s+apple\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple; only {receiver} accepts it. Preserve the settled handoff.")
    account = r"(?:private\s+)?(?:recollection|secret|information|endorsement|story|offering)"
    for sentence in split_sentences(speech):
        if re.search(r"\bnot\s+(?:owed|due)\s+to\s+(?:me|you)\b", sentence, re.I):
            continue
        reversed_price = (speaker == receiver and re.search(
            rf"\b{account}\b[^.!?]{{0,45}}\b(?:owed|due)\s+to\s+me\b|\byou\s+owe\s+me\s+(?:an?\s+|the\s+)?{account}\b", sentence, re.I)
            or speaker == giver and re.search(
                rf"\bi\s+owe\s+you\s+(?:an?\s+|the\s+)?{account}\b|"
                rf"\b{account}\b[^.!?]{{0,45}}\b(?:owed|due)\s+to\s+you\b|"
                rf"\byou(?:['’]re| are)\s+owed\s+(?:an?\s+|the\s+)?{account}\b|"
                rf"\bi have (?:an?|my) {account} to share with you\b|"
                rf"\bi\s+(?:will|shall|promise to)\s+(?:offer|share|provide|tell)\s+"
                rf"(?:with\s+)?you\s+(?:an?|my|the)\s+{account}\b", sentence, re.I)
            or speaker == receiver and re.search(
                r"\byour promise to share your (?:secrets|recollection)\b|"
                r"\b(?:hearing|hear|receive) your private recollection\b", sentence, re.I))
        if reversed_price or re.search(r"\bwe owe\s+(?:to\s+)?(?:each other|one another)\b", sentence, re.I):
            raise ValueError(f"{receiver} owes the recollection or endorsement to {giver}. This is not a mutual debt or information owed to the resident.")
    if apple_received:
        debtor = "i" if speaker == receiver else "you"
        if re.search(rf"\b{debtor}\s+(?:(?:simply|actually|really)\s+)?"
                     r"(?:have yet to|have not(?: yet)?|haven['’]t(?: yet)?|did not|didn['’]t|never)\s+"
                     r"(?:been\s+)?(?:receive[d]?|get|got|taken?|given|offered)\s+(?:(?:an?|the|one|any|your|this)\s+)?(?:apple|fruit)\b", speech, re.I):
            raise ValueError("The apple was already received. Doubting the bargain does not undo the recorded handoff.")
        if _asserts(_APPLE_REFUSAL, speech):
            raise ValueError("The resident received the apple. Any later refusal concerns the offering, not the original apple.")
    if offering_pending:
        debtor = "i" if speaker == receiver else "you"
        if re.search(rf"\b{debtor}\s+(?:(?:have|had)\s+)?(?:already\s+)?(?:provided|shared|told|fulfilled)\s+"
                     rf"(?:you\s+)?(?:the|my|your|that|a)\s+(?:agreed\s+)?{account}\b", speech, re.I):
            raise ValueError("The offering is still unpaid. Do not claim it has already been provided or settled.")
    if owed_offering:
        for sentence in split_sentences(speech):
            if re.search(r"\b(?:not|never|don't|do not)\s+(?:\w+\s+){0,2}(?:owe|need to return)\b", sentence, re.I):
                continue
            if re.search(r"\bowe\s+(?:you\s+|me\s+)?(?:an?|the|one)\s+(?:apple|fruit)\b|"
                         r"\b(?:apple|fruit)\s+(?:you\s+(?:(?:already|previously)\s+)?received\s+)?(?:is|was|remains)\s+(?:(?:still|now)\s+)?owed\b|"
                         r"\b(?:i am|you are)\s+entitled to the apple\b|"
                         r"\b(?:obligation|obliged|required|must)\b[^.!?]{0,35}\breturn\s+(?:it|that|(?:the|an?)\s+(?:apple|fruit))\b", sentence, re.I):
                raise ValueError("The obligation is information or public support for an apple already received, not an apple repayment or return.")
