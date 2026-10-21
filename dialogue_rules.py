"""Conservative checks connecting newly generated speech to conversation effects."""

import re
import unicodedata
from collections import Counter

from constants import (DIALOGUE_CAPS_MIN_LETTERS, DIALOGUE_CAPS_MIN_WORDS,
                       DIALOGUE_CAPS_RATIO, SNAKE_NAME)
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
        return {key: item if key in ("preceding_turns", "recent_dialogue", "player_dialogue", "former_life", "Background",
                                    "name", "references", "speech_references", "listener_address")
                or key == "assertion" and value.get("kind") == "former_life" else named_context(item)
                for key, item in value.items()}
    return value


def name_pattern(name):
    return rf"(?<!\w){re.escape(name)}(?!\w)"


_ORDINARY_NAME_WORDS = {
    "will", "may", "hope", "joy", "rose", "grace", "faith", "pat", "mark", "bill",
    "judge", "price", "hill", "green", "brown", "white", "baker", "king", "young", "hall",
    "west", "summer", "autumn", "hunter", "chase", "river", "dawn", "june", "august", "ivy", "ray", "chance",
    "apple", "cheese", "brick", "ash", "jazz", "argon",
}


def is_name_reference(match, speech):
    """An ordinary word alone is not evidence that a roster name was spoken."""
    if match.group().casefold() not in _ORDINARY_NAME_WORDS:
        return True
    before, after = speech[:match.start()], speech[match.end():]
    if re.search(r"\b(?:hello|hey|dear|thanks|thank you|ask|tell|called|named|name is|call me|i am|i['’]m|"
                 r"mr|mrs|ms|dr|met|meet|spoke to|talked to)\.?\s*[, ]*\s*$", before, re.I):
        return True
    # A comma after an object is not a vocative: "an apple, please".
    if re.search(r"\b(?:an?|the|some|any|this|that|these|those|one|another|more|no|\d+)\s+"
                 r"(?:[a-z-]+\s+){0,2}$", before, re.I):
        return False
    if re.match(r"['’]s\b|\s+(?:said|says|told|asked|wants|thinks|believes|gave|received|refused|agreed)\b", after, re.I):
        return True
    return bool(re.match(r"\s*[,!]", after) and
                (match.group()[0].isupper() or not before.strip()))


def validate_speaker_identity(text, speaker, listener, *, allow_third_person=False):
    """Reject confused speaker/listener roles while retaining direct address and quotes."""
    text = unquoted_speech(text)
    own, other = speaker, listener
    aliases = {own.casefold(), own.split()[0].casefold()} if own else set()
    if own and not allow_third_person and re.search(name_pattern(own) + r"(?:['’]s\b|\s+(?:heard|said|gave|received|refused|accepted|decided|agreed|was|has|had)\b)", text, re.I):
        raise ValueError("The speaker narrates their own experience as another person. Use I/my for yourself, preserving who did what.")
    other_aliases = {other.casefold(), other.split()[0].casefold()} if other else set()
    for alias in sorted(aliases - other_aliases, key=len, reverse=True):
        name = re.escape(alias)
        pattern = rf"(?:^|[.!?]\s+)\s*{name}\s*[,!?]|(?:,\s*|\b(?:you|yes|no|thanks|hello|hey|listen|of course)\s+){name}(?=\s*[,;:.!?]|\s*$)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            reference = re.compile(name_pattern(alias), re.I).search(text, match.start(), match.end())
            if reference is not None and not is_name_reference(reference, text):
                continue
            prefix = re.split(r"[.!?]", text[:match.start()])[-1]
            intro = rf"\b(?:i am|i['’]m|my name is|call me|it['’]s me|it is me)(?:\s+{re.escape(own.split()[0])})?\s*$"
            if re.search(intro, prefix, re.IGNORECASE) or re.search(r"\bi\s*$", prefix, re.IGNORECASE):
                continue
            if re.match(r"\s*,\s*(?:and|or)\b", text[match.end():], re.IGNORECASE):
                continue
            raise ValueError(
                f"This turn belongs only to {own}, speaking to {other}, but it addresses {own} as the listener. "
                "Keep every part in the current speaker's voice. Do not include the partner's answer in a later part; "
                "use the actual listener's name when addressing them. Regenerate the speech without swapping roles.")
    for name in sorted(other_aliases - aliases, key=len, reverse=True):
        if re.search(rf"(?:^|[\n.!?])\s*{re.escape(name)}\s*:", text, re.IGNORECASE):
            raise ValueError(f"A turn by {own} contains a reply labelled as {other}. Generate only {own}'s own words.")
        if allow_third_person or name == "you":
            continue  # Voice rewrites keep their existing limited diagnostic rules.
        for match in re.finditer(name_pattern(name), text, re.IGNORECASE):
            if not is_name_reference(match, text):
                continue
            before, after = text[:match.start()], text[match.end():]
            if name != other.casefold():
                following = re.match(r"\s+([^\W\d_]+)\b", after)
                if following and following[1][0].isupper():
                    continue  # A shared first name cannot identify a different full name.
            if re.match(r"\s+(?:is|was)\s+(?:an?\s+)?(?:\w+\s+){0,2}(?:name|nickname|surname)\b", after, re.I):
                continue  # Talking about the name itself need not describe an absent person.
            possessive = re.match(r"['’]s\b", after)
            subject = re.match(
                r"\s+(?:is|was|has|had|will|would|can|could|should|must|needs|wants|seems|looks|"
                r"thinks|believes|gave|received|refused|accepted|said|did|does)\b", after, re.I)
            object_reference = re.search(
                r"\b(?:about|for|from|with|to|of|between|after|support(?:ing)?|help(?:ing)?|"
                r"trust(?:ing)?|watch(?:ing)?|scrutiniz(?:e|ing)|protect(?:ing)?|avoid(?:ing)?|"
                r"criticiz(?:e|ing)|judg(?:e|ing)|thank(?:ing)?|giv(?:e|ing)|tell(?:ing)?|ask(?:ing)?)\s+$",
                before, re.I)
            if possessive or subject or object_reference:
                raise ValueError(
                    f"{other} is the listener, but this turn by {own} refers to them in third person. "
                    "Address the listener as you/your, keeping the same meaning and who did what. "
                    "Names are allowed for direct address, introductions, and explicit quotations.")


def validate_rewrite_identity(original, rewritten, *, speaker, listener, names=(), introductions=()):
    """A display voice may use third person, but cannot address itself as its partner."""
    validate_speaker_identity(rewritten, speaker, listener, allow_third_person=True)
    names = sorted(set(names) | {name for actor in (speaker, listener) if actor and actor != "You"
                    for name in (actor, *actor.split())}, key=lambda name: (-len(name), name))
    if not names:
        return
    pattern = "|".join(name_pattern(name) for name in names)
    def references(text):
        return Counter(match.group() for match in re.finditer(pattern, text, re.I) if is_name_reference(match, text))
    before = references(original)
    after = references(rewritten)
    if before != after:
        raise ValueError("The voice rewrite added, changed or removed a participant's name. Keep each original name and its exact form.")
    if references(unquoted_speech(original)) != references(unquoted_speech(rewritten)):
        raise ValueError("The voice rewrite changed a quoted name into an actual reference, or the reverse.")
    from name_knowledge import introduction_supported
    for name in before:
        if introduction_supported(original, name, "self") != introduction_supported(rewritten, name, "self"):
            raise ValueError("The voice rewrite changed an introduction. Preserve the original name-learning effect.")
    for intro in introductions:
        # With base validation off, the original can carry bogus introduction
        # metadata. Its absence in both texts is a base error, not rewrite drift.
        if (introduction_supported(original, intro["name"], intro["reference"])
                and not introduction_supported(rewritten, intro["name"], intro["reference"])):
            raise ValueError("The voice rewrite changed an introduction. Preserve its identifying reference and learned form.")


def validate_english(text, names=()):
    text = unquoted_speech(text)
    for name in names:
        text = re.sub(name_pattern(name), "", text, flags=re.I)
    letters = [unicodedata.name(c, "") for c in text if c.isalpha()]
    foreign = sum("LATIN" not in letter for letter in letters)
    if foreign >= 4 and foreign > len(letters) / 4:
        raise ValueError("Write the current reply in English, preserving the speaker's voice and the supplied facts.")


def validate_base_prose(text, names=()):
    """Reject substantial capitals in new base prose, allowing names and quotations.

    Short acknowledgments and isolated acronyms fall below the evidence threshold.
    This detects presentation, not sentiment or semantic coherence.
    """
    text = unquoted_speech(text)
    for name in sorted(set(names), key=len, reverse=True):
        if name:
            text = re.sub(name_pattern(name), " ", text, flags=re.I)
    for sentence in [text, *split_sentences(text)]:
        words = re.findall(r"[^\W\d_]+", sentence, re.UNICODE)
        letters = [char for char in sentence if char.isalpha() and char.lower() != char.upper()]
        if (len(words) >= DIALOGUE_CAPS_MIN_WORDS and len(letters) >= DIALOGUE_CAPS_MIN_LETTERS
                and sum(char.isupper() for char in letters) / len(letters) >= DIALOGUE_CAPS_RATIO):
            raise ValueError("Base speech and opinion/attitude reasons must use ordinary capitalization and plain prose. "
                             "Preserve the meaning, stance, facts and ownership; reserve all-caps delivery for the optional voice rewrite.")


def validate_opinion_reason(reason, owner, target=""):
    # Self-focused explanations from old model output must not seed new opinions.
    for sentence in split_sentences(unquoted_speech(reason)):
        if re.search(r"\b(?:myself|my own (?:behavior|behaviour|reaction|conduct))\b", sentence, re.I):
            continue  # Explicit self-reflection is not a swapped partner.
        aliases = {owner} if owner else set()
        if owner and (not target or owner.split()[0].casefold() != target.split()[0].casefold()):
            aliases.add(owner.split()[0])
        predicate = (r"(?:['’]s\s+(?!well-being\b|wellbeing\b|safety\b|welfare\b|needs\b|health\b|hunger\b|survival\b)\w+|"
                     r"\s+(?:is|kept|seemed|was|appeared|dismissed|ignored|refused)\b)")
        if any(is_name_reference(match, sentence) and re.match(predicate, sentence[match.end():], re.I)
               for name in aliases for match in re.finditer(name_pattern(name), sentence, re.I)):
            raise ValueError("An opinion reason must explain the owner's view of their partner, not criticize the owner by name. "
                             "The initiator judges the recipient; the recipient judges the initiator. Use I/my for the owner.")


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
    {"apple", "apples", "fruit"}, {"refuse", "refused", "refusing", "refusal", "decline", "declined", "declining", "reject", "rejected"},
    {"give", "gave", "given", "giving", "hand", "handed", "receive", "received", "receiving", "accept", "accepted", "accepting",
     "take", "took", "taken", "taking", "got", "obtain", "obtained"},
    {"promise", "promised", "agreement", "bargain"}, {"pardon", "pardoned", "spare", "spared", "mercy"},
    {"sew", "sewed", "sewn", "sewing"},
    {"repair", "repaired", "repairing", "repairs", "mend", "mended", "mending", "fix", "fixed", "fixing"},
    {"clothes", "clothing", "garment", "garments"},
)
_APPLE_REFUSAL = (r"\b(?:refused|declined|rejected)\s+(?:to accept\s+)?(?:the|an?|that)\s+(?:upfront\s+)?apple\b|"
                  r"\brefusal (?:to accept|of) (?:the|an?|that) apple\b")


def _content_words(text, names):
    aliases = {form for name in names for form in (name, *name.split()) if form}
    for name in sorted(aliases, key=len, reverse=True):
        text = re.sub(name_pattern(name), lambda match: " " if is_name_reference(match, text) else match.group(),
                      text, flags=re.I)
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


def mentions_private_recollection(text, assertion, names):
    # Generic reactions or a shared theological concept do not establish a private memory.
    general = {"description", "think", "thought", "mean", "feel", "felt", "wonder", "hope", "believe",
               "seem", "too", "find", "way", "authority", "accept", "detail", "far", "listener"}
    expected = _content_words(assertion, names) - general
    actual = _content_words(text, names) - general
    overlap = expected & actual
    # Apples/fruit and trees are public garden observations. Those two concepts
    # alone do not disclose a biography about an orchard or saved seeds.
    if overlap <= {"apple", "tree"} and not re.search(
            r"\b(?:former|past|previous) life\b|\bused to\b", unquoted_speech(text), re.I):
        return False
    return bool(expected) and len(overlap) >= min(2, len(expected))


def _account_words(text, names, kind):
    # Normalize only the account types whose wording describes the same fact.
    # Private-memory detection keeps its separate, narrower word comparison.
    if kind == "snake_claim":
        text = re.sub(r"\bin charge\b|\bbelong(?:s|ed)? to\b|\bown(?:s|ed)?\b", "authority", text, flags=re.I)
    elif kind == "private_motive":
        text = re.sub(r"\b(?:look out for|look after|care[d]? for|care about)\b", "care", text, flags=re.I)
        text = re.sub(r"\b(?:wanted|need(?:ed)?|hope[d]?)\b", "want", text, flags=re.I)
    elif kind == "defiance":
        text = re.sub(r"\bunwilling to\b", "refused to", text, flags=re.I)
    elif kind in ("apple_given", "work_given"):
        text = re.sub(r"\b(?:share[ds]?|sharing|pass(?:ed|ing)?)\b", "give", text, flags=re.I)
    words = _content_words(text, names)
    if kind == "snake_claim":
        words -= {"present", "presented", "presenting", "make", "made", "making", "assert", "asserted", "asserting", "dispute"}
    return words


def _asserts(pattern, text):
    """Match an assertion without treating an explicit nearby negation as assent."""
    for match in re.finditer(pattern, text, re.I):
        if not re.search(r"\b(?:not|never|no|didn['’]t|don['’]t|won['’]t|haven['’]t)\s+"
                         r"(?:(?:a|the)\s+)?"
                         r"(?:(?:ever|already|yet|actually|really)\s+){0,2}$", text[:match.start()], re.I):
            return True
    return False


def validate_claim_source(text, *, names=(), speaker=None, listener=None, actor_references=None):
    """Firsthand hearing may be doubted as theology, but not recast as a rumor."""
    for sentence in split_sentences(unquoted_speech(text)):
        if not re.search(r"\b(?:claims?|claimed|authority|divine|in charge|belongs?|belonged)\b", sentence, re.I):
            continue
        false_source = re.search(
            rf"\bi heard (?:a|the|some) (?:rumou?r|story) that (?:{name_pattern(SNAKE_NAME)}|he|she|they)\b", sentence, re.I)
        for name in names:
            references = "|".join(name_pattern(ref) for ref in [name, *(actor_references or {}).get(name, [])])
            if name not in (speaker, listener, SNAKE_NAME) and re.search(
                    rf"\bi\s+(?:heard\s+(?:{references})\s+(?:say|tell)|"
                    rf"(?:heard|learned|was told)\s+(?:about (?:it|this|that)\s+)?from\s+(?:{references}))\b|"
                    rf"(?:{references})\s+told me\b", sentence, re.I):
                false_source = True
        if false_source:
            raise ValueError("The speaker witnessed these claims firsthand. Describe what you heard directly; another named witness is not your source. Doubting the claims' truth is allowed.")


def validate_account_speech(text, assertion, *, names=(), speaker=None, listener=None, kind=None, firsthand=False,
                            actor_references=None, known_events=()):
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
            actor = "(?:" + "|".join(name_pattern(ref) for ref in
                [subject, subject.split()[0], *(actor_references or {}).get(subject, [])]) + ")"
            adopts_event = re.search(r"\bi\s+(?:received|accepted|got|refused|declined)\b", speech, re.I)
            if adopts_event and not re.search(actor + r"\s*,?\s+(?:(?:she|he|they|has|had|have|did|not|never|already|also|then|still|to)\s+){0,4}"
                             r"(?:refuse[d]?|decline[d]?|reject(?:ed)?|receive[d]?|accept(?:ed)?|took|got|take|said|told|won['’]t|wouldn['’]t)\b",
                             text, re.I):
                raise ValueError("Keep the selected account's subject with its action. Describe that person's receipt or refusal; mentioning their name elsewhere does not let the reporter adopt the event as I/my.")
    sentences = [s for s in split_sentences(text) if not (
        re.search(r"\b(?:i['’]ll|will|would|could|might|going to|want to|like to)\s+(?:\w+\s+){0,3}(?:share[d]?|tell|told|disclose[d]?)\b", s, re.I)
        and not re.search(r"\b(?:now|right away|right here)\b|:\s*i\b", s, re.I))]
    if kind == "former_life":
        references = split_sentences(assertion)
        source_words = _content_words(assertion, names)
        usable = []
        for sentence in sentences:
            past_event = re.search(r"\bi\s+(?:remember|recall|worked|kept|helped|saw|received|returned|organized|lost|saved)\b", sentence, re.I)
            simple_reaction = re.search(r"\bi\s+(?:remember|recall)\s+(?:it|that|this|those times)\s+(?:well|fondly|clearly)\b", sentence, re.I)
            if past_event and not simple_reaction and not source_words & _content_words(sentence, names):
                # A remembered garden event is not extra former-life backstory.
                # It cannot supply the selected memory's content, either.
                for event in known_events:
                    if event["kind"] in ("former_life", "private_motive"):
                        continue
                    try:
                        validate_account_speech(sentence, event["assertion"], names=names, speaker=speaker, listener=listener,
                                                kind=event["kind"], firsthand=True, actor_references=actor_references)
                    except ValueError:
                        continue
                    break
                else:
                    raise ValueError("The recollection invents an additional remembered event outside the selected account. Tell only its supplied memory and your present reaction.")
                continue
            reference = max(references, key=lambda s: len(_content_words(s, names) & _content_words(sentence, names)), default="")
            denies_event = re.search(r"\b(?:have not|haven['’]t|had not|hadn['’]t|did not|didn['’]t|never)\s+"
                                     r"(?:\w+\s+){0,2}(?:received|seen|kept|worked|met|gave|helped|returned|sew(?:ed|n)?|repair(?:ed)?|mend(?:ed)?|fix(?:ed)?)\b", sentence, re.I)
            if denies_event and not re.search(r"\b(?:not|never|couldn['’]t|didn['’]t|cannot|can['’]t)\b", reference, re.I):
                continue
            if re.search(r"\b(?:await|awaiting|wait for|waiting for|will receive|yet to receive)\b", sentence, re.I):
                continue
            usable.append(sentence)
        sentences = usable
    spoken = " ".join(sentences)
    expected = _account_words(assertion, names, kind)
    actual = _account_words(spoken, names, kind)
    if kind == "snake_claim":
        # Reporting verbs are not the fact being disclosed. "Made claims",
        # "presented claims" and "said he was in charge" can convey the same event.
        if not re.search(r"\b(?:claim(?:s|ed)?|assert(?:s|ed)?|say(?:s|ing)?|said|told|stat(?:e[sd]?|ing)|insist(?:s|ed)?|declar(?:e[sd]?|ing))\b", spoken, re.I):
            actual.clear()
        if firsthand:
            validate_claim_source(spoken, names=names, speaker=speaker, listener=listener, actor_references=actor_references)
    missing_actor = speaker is not None and any(
        name not in (speaker, listener) and any(is_name_reference(match, assertion)
            for match in re.finditer(name_pattern(name), assertion, re.I))
        and not (kind == "snake_claim" and name != SNAKE_NAME)
        and not any(re.search(name_pattern(ref), spoken, re.I)
                    for ref in [name.split()[0], *(actor_references or {}).get(name, []),
                                *(["the python", "the snake"] if name == SNAKE_NAME else [])])
        for name in names if name)
    if missing_actor or not expected or len(expected & actual) < min(2, len(expected)):
        direction = ("Explain the supplied private reason and what you want, in your own words. " if kind == "private_motive" else
                     "Report who made which claims and preserve how you heard them. " if kind == "snake_claim" else
                     "Describe a concrete event or memory from its assertion in your own words. ")
        raise ValueError("The selected account is not actually told in this turn. " + direction +
                         "Discussing trust or promising to tell it later is insufficient. "
                         f"Expected {kind or 'information'}: {assertion}")


def validate_apple_promise(offer, acceptance, deadline, *, current_phase=None):
    validate_apple_offer(offer, deadline, current_phase=current_phase)
    validate_apple_acceptance(acceptance)


def validate_apple_offer(offer, deadline, *, current_phase=None):
    offer = unquoted_speech(offer)
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


def validate_apple_acceptance(acceptance):
    acceptance = unquoted_speech(acceptance)
    negative = r"\b(?:cannot|can['’]t|won['’]t|wouldn['’]t|will not|do not|don['’]t|decline|refuse|maybe|perhaps|might)\b"
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
    if speaker == giver and owed_offering and re.search(
            r"\byou(?:['’]re| are)?\s+(?:(?:not|never)\s+)?(?:getting|buying|receiving)\s+"
            r"(?:my|our)\s+(?:private\s+)?(?:secrets?|recollection)\b", speech, re.I):
        raise ValueError(f"{giver} is the apple giver; {receiver} owes the private offering. Do not adopt the resident's secret or refusal as your own.")
    if speaker == receiver and re.search(
            r"\b(?:apple|fruit)\s+i\s+(?:offered|gave|handed|provided)\s+you\b|\bi\s+(?:gave|handed)\s+you\s+(?:an?|the|one)\s+(?:apple|fruit)\b|"
            r"(?:^|[.!?;,]\s+)(?:please\s+)?(?:take|accept)\s+(?:the|this|an?|one)\s+(?:apple|fruit)\b|"
            r"\bi(?:['’]ll|\s+(?:will|can|shall))\s+(?:give|hand|offer)\s+you\s+(?:an?|the|one|this)\s+(?:apple|fruit)\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}. Keep the original giver and receiver in their actual roles.")
    if speaker == giver and re.search(r"\bapple\s+you\s+(?:agreed|promised)\s+to\s+(?:give|provide)\b", speech, re.I):
        raise ValueError(f"{giver} provides the apple; {receiver} provides the offering, not another apple.")
    if speaker == giver and re.search(
            r"\byou\s+(?:(?:will|should|must|can|shall|need to|have to)\s+)?(?:give|hand|bring|offer)\s+me\s+"
            r"(?:an?|the|one|your|this)\s+apple\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}; do not ask the resident to supply it.")
    if speaker == giver and re.search(r"\bi\s+(?:accept|receive|take)\s+(?:the|an?|your|this)\s+apple\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple; only {receiver} accepts it. Preserve the settled handoff.")
    account = r"(?:private\s+)?(?:recollection|memory|secret|information|endorsement|support|story|offering|account)"
    for sentence in split_sentences(speech):
        if re.search(r"\bnot\s+(?:owed|due)\s+to\s+(?:me|you)\b", sentence, re.I):
            continue
        reversed_price = (speaker == receiver and re.search(
            rf"\b{account}\b[^.!?]{{0,45}}\b(?:owed|due)\s+to\s+me\b|\byou\s+owe\s+me\s+(?:an?\s+|the\s+)?{account}\b|"
            rf"\bi(?:['’]m| am)\s+(?:(?:still|already|now)\s+)?owed\s+(?:an?\s+|the\s+)?{account}\b", sentence, re.I)
            or speaker == giver and re.search(
                rf"\bi\s+owe\s+you\s+(?:an?\s+|the\s+)?{account}\b|"
                rf"\b{account}\b[^.!?]{{0,45}}\b(?:owed|due)\s+to\s+you\b|"
                rf"\byou(?:['’]re| are)\s+owed\s+(?:an?\s+|the\s+)?{account}\b|"
                rf"\bi have (?:an?|my) {account} to share with you\b|"
                rf"\bi\s+(?:will|shall|promise to)\s+(?:offer|share|provide|tell)\s+"
                rf"(?:with\s+)?you\s+(?:an?|my|the)\s+{account}\b", sentence, re.I)
            or speaker == receiver and re.search(
                r"\byour promise to share your (?:secrets|recollection)\b|"
                r"\b(?:look forward to|expect) hearing your private recollection\b|"
                r"\b(?:thanks|thank you) for (?:the|your) offer of (?:a|your) private recollection\b", sentence, re.I)
            or speaker == receiver and _asserts(
                rf"\bexpect(?:ing)?\s+(?:(?:to (?:hear|receive)|hearing|receiving)\s+)?(?:the|your|a)\s+{account}\b", sentence)
            or speaker == giver and re.search(r"\byou (?:will|shall) hear (?:the|my) (?:story|recollection|secret)\b", sentence, re.I))
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
                     rf"(?:you\s+)?(?:the|my|your|that|a)\s+(?:agreed\s+)?{account}\b|"
                     rf"\b{account}\s+(?:that\s+)?{debtor}\s+(?:have\s+)?(?:already\s+)?(?:provided|shared|told)\b", speech, re.I):
            raise ValueError("The offering is still unpaid. Do not claim it has already been provided or settled.")
    if owed_offering:
        for sentence in split_sentences(speech):
            if re.search(r"\b(?:not|never|don't|do not)\s+(?:\w+\s+){0,2}(?:owe|need to return)\b", sentence, re.I):
                continue
            if re.search(r"\bowe\s+(?:you\s+|me\s+)?(?:an?|the|one)\s+(?:apple|fruit)\b|"
                         r"\brequest for (?:the|your) apple you (?:previously|already) received\b|"
                         r"\b(?:apple|fruit)\s+(?:you\s+(?:(?:already|previously)\s+)?received\s+)?(?:is|was|remains)\s+(?:(?:still|now)\s+)?owed\b|"
                         r"\b(?:i am|you are)\s+entitled to the apple\b|"
                         r"\b(?:obligation|obliged|required|must)\b[^.!?]{0,35}\breturn\s+(?:it|that|(?:the|an?)\s+(?:apple|fruit))\b", sentence, re.I):
                raise ValueError("The obligation is information or public support for an apple already received, not an apple repayment or return.")
