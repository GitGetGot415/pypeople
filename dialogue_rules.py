"""Text-level checks on generated speech: identity, grounding, continuity and promises.

These rules need only the text and a few names, so social, information and the prompt builders can use them.
Validation that needs a whole snapshot or accepted draft lives in dialogue_validation, which builds on this module.
"""

import re
import unicodedata
from collections import Counter
from validation_errors import DialogueValidationError, RecoveryKind
from constants import (DIALOGUE_CAPS_MIN_LETTERS, DIALOGUE_CAPS_MIN_WORDS, DIALOGUE_CAPS_RATIO,
                      SNAKE_NAME, SPOKEN_NUMBERS)
from dialogue_text import split_sentences


def unquoted_speech(text):
    return re.sub(r'''"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)'[^'\n]+'(?!\w)''', " ", text)


def validate_environment_speech(text, *, location="garden"):
    """Check local setting facts while preserving wishes, quotations and past lives."""
    from area_knowledge import validate_area_speech
    validate_area_speech(text, {})
    speech = unquoted_speech(text)
    venue = r"(?:jazz club|nightclub|caf[eé]|restaurant)"
    mentions_venue = bool(re.search(rf"\b{venue}\b", speech, re.I))
    # "The tree's absence of apples" describes missing fruit, not a missing tree.
    missing_tree = (r"\b(?:the|this|that|our) (?:apple )?tree(?:['’]s (?:absence|disappearance)(?!\s+of\b)|"
                    r"\s+(?:is|was|has|had)(?: (?:now|completely|just|entirely))? "
                    r"(?:gone|missing|absent|vanished|disappeared|been removed))\b|"
                    r"\b(?:absence|disappearance) of (?:the|our) (?:apple )?tree\b")
    for sentence in split_sentences(speech):
        if sentence.rstrip().endswith(("?", "\uFF1F", "\u061F")):
            continue
        if location != "garden" and not re.search(
                r"\b(?:if|imagine|remember|used to|former life|earlier|yesterday|thought|mistaken|"
                r"before (?:i|we) arrived|when i was|in my mind|painting|dream)\b", sentence, re.I):
            outside = re.search(
                r"\b(?:we(?: are|['\u2019]re)|i(?: am|['\u2019]m)) (?:(?:still|currently|now) )?"
                r"(?:outdoors|outside(?! (?:of )?(?:the |a |an |your |my |this |that |our |their )?(?:elevator|room|door)\b)|"
                r"in (?:the|this) garden|(?:standing|sitting|waiting|walking)(?: here)? (?:in|on) (?:the|this) (?:grass|lawn))\b|"
                r"\b(?:i|we) can see (?:the |an? )?(?:open sky|sky|apple tree)\b|"
                r"\b(?:the )?grass (?:under|beneath) (?:my|our) feet\b", sentence, re.I)
            if outside:
                area = "the apartment hallway" if location.startswith("hallway:") else "a private apartment room"
                raise ValueError(f"You are inside {area}. Describe your current indoor surroundings; the garden, grass, apple tree and open sky are outside. Distinguish memories or plans from where you are now.")
        missing = re.search(missing_tree, sentence, re.I)
        if location == "garden" and missing and not re.search(r"\b(?:if|imagine|thought|mistaken|wrong|not|no|never|does not mean|doesn't mean)\b", sentence[:missing.start()], re.I):
            raise ValueError("The living apple tree is still present, even with zero apples. Describe missing fruit, not a vanished tree; do not invent its disappearance or destruction.")
        if not mentions_venue:
            continue
        if re.search(r"\b(?:wish|imagine|if|would|could|might|hope|dream|want|love to|let['’]s build)\b", sentence, re.I):
            continue
        here = re.search(r"\b(?:here|this garden|the garden|near (?:the|that|our) (?:apple )?tree)\b", sentence, re.I)
        historical = re.search(r"\b(?:remember|used to|former life|before (?:i|we) arrived|back in|when i was)\b", sentence, re.I)
        if historical and not here:
            continue
        positive = re.search(
            rf"\b(?:i|we)(?:['’]ve| have)? (?:just )?(?:got back|came back|returned|come back|went|been) (?:from|to) (?:an? |the )?{venue}\b|"
            rf"\b(?:there['’]s|there is|there was|i found|we found|i see|we see) (?:an? |the )?{venue}\b|"
            rf"\b(?:a|the) {venue}\s+(?:is|was)\b|"
            r"\bi found it (?:in|near)\b|\bit['’]s (?:a cozy|in the heart of the garden|near the tree)\b",
            sentence, re.I)
        if positive:
            raise ValueError("No jazz club or other public venue exists in the current garden. Use the grassy area, its one apple tree and the apartment building. A personality or former-life memory does not create a new place here; correct any earlier invented venue instead of elaborating it.")


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


def identifying_forms(name, names):
    """Forms that name this person alone: the first name only while no one else in names shares it."""
    first = name.split()[0]
    if any(other != name and other.split() and other.split()[0].casefold() == first.casefold() for other in names):
        return [name, name.split()[-1]]
    return [first]


_ORDINARY_NAME_WORDS = {
    "will", "may", "hope", "joy", "rose", "grace", "faith", "pat", "mark", "bill",
    "judge", "price", "hill", "green", "brown", "white", "baker", "king", "young", "hall",
    "west", "summer", "autumn", "hunter", "chase", "river", "dawn", "june", "august", "ivy", "ray", "chance",
    "apple", "cheese", "brick", "ash", "jazz", "argon", "player",
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
    if own and not allow_third_person and re.search(name_pattern(own) + r"(?:['’]s\b|\s+(?:heard|said|gave|received|refused|accepted|decided|agreed|was|has|had|is doing|is feeling|is thinking)\b)", text, re.I):
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


def validate_rewrite_additions(original, rewritten, *, speaker, mannerisms=""):
    """Diagnose an invented self-name or a Mannerisms description read aloud; never used to reject a rewrite."""
    own = {form.casefold() for form in (speaker, *speaker.split()) if form}
    for match in re.finditer(r"\b(?:I,\s*|my name is\s+|I am\s+|I['’]m\s+|call me\s+)([A-Z][a-z]+)\b", rewritten):
        if match.group(1).casefold() not in own and match.group(0) not in original:
            raise ValueError(f"The voice rewrite names the speaker {match.group(1)}, but the speaker is {speaker}.")
    if mannerisms and (_word_runs(first_person_text(rewritten), 6) & _word_runs(first_person_text(mannerisms), 6)
                       ) - _word_runs(first_person_text(original), 6):
        raise ValueError("The voice rewrite reads the Mannerisms description aloud instead of applying it.")


def validate_rewrite_identity(original, rewritten, *, speaker, listener, names=()):
    """A display voice may use third person, but cannot address itself as its partner."""
    validate_speaker_identity(rewritten, speaker, listener, allow_third_person=True)
    names = sorted(set(names) | {name for actor in (speaker, listener) if actor and actor != "You"
                    for name in (actor, *actor.split())}, key=lambda name: (-len(name), name))
    if not names:
        return
    pattern = "|".join(name_pattern(name) for name in names)
    own = {form.casefold() for form in (speaker, *speaker.split()) if form}
    def references(text):
        # A Voice may ask for self-reference by name; only other people's names must stay exactly as written.
        return Counter(match.group() for match in re.finditer(pattern, text, re.I)
                       if is_name_reference(match, text) and match.group().casefold() not in own)
    before = references(original)
    after = references(rewritten)
    if before != after:
        raise ValueError("The voice rewrite added, changed or removed a participant's name. Keep each original name and its exact form.")
    if references(unquoted_speech(original)) != references(unquoted_speech(rewritten)):
        raise ValueError("The voice rewrite changed a quoted name into an actual reference, or the reverse.")


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


def validate_reason_voice(reason, owner):
    """A newly written reason speaks as its owner; stored system reasons may still name them."""
    first = owner.split()[0] if owner else ""
    speech = unquoted_speech(reason)
    if owner and (re.search(name_pattern(owner), speech, re.I) or first and any(
            is_name_reference(match, speech) and re.match(r"['’]s\b|\s+(?:has|had|wants|feels|felt|believes|thinks|finds|found|"
                                                          r"noticed|appreciates|witnessed|sees|saw|remembers)\b", speech[match.end():], re.I)
            for match in re.finditer(name_pattern(first), speech, re.I))):
        raise ValueError(f"You are {owner}. Write the reason with I/me/my for yourself, never your own name, "
                         "about how your partner's words or conduct changed your view of them.")


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


_HOSTILE_VIEW = re.compile(
    r"\b(?:selfish\w*|greed\w*|hoard\w*|stingy|dishonest\w*|untrustworth\w*|distrust\w*|manipulat\w*|"
    r"inconsiderate|thoughtless|unkind|rude|pushy|only cares? about|cares? (?:nothing|little) (?:for|about)|"
    r"did(?:n['’]t| not) share|refused to share|wary|warier|suspicious|doubts?|uneasy|unease|distrustful|cautious|"
    r"self-cent(?:ered|red)|thinks? only (?:of|about)|only (?:thinks|cares|looks out) (?:of|about|for)|"
    r"hiding something|ulterior|hidden (?:motive|agenda)s?|disappoint\w*|upset|annoy\w*|frustrat\w*|"
    r"question(?:ed|s|ing)? (?:his|her|their|your) (?:honesty|reliability|motives?|intentions?|word)|unreliable|"
    r"dismiss(?:ed|ive|ing)|contradict\w*|stir(?:ring)? up trouble|accusatory|lie[sd]?|lying|mislead\w*|misled|"
    r"(?:feel|feels|feeling|felt) worse|think(?:s|ing)? (?:less|worse)|thought (?:less|worse))\b", re.I)
_WARM_VIEW = re.compile(
    r"\b(?:appreciat\w*|grateful|thankful|kind|kindness|generous|generosity|honest|honesty|trustworth\w*|thoughtful|"
    r"refreshing|relatable|solidarity|supportive|friendly|pleasant|comfort(?:ing|able)|at ease|admir\w*|respect\w*|"
    r"resonat\w*|inspir\w*|genuine|caring|warm|enjoy\w*|seen and heard|feel seen|like (?:her|him|them)|considerate|"
    # "made me feel better about him": the owner's own view, aimed at whoever follows.
    r"(?:feel|feels|feeling|felt) better|think(?:s|ing)? (?:better|more highly)|thought (?:better|more highly))\b", re.I)
_VIEW_NEGATED = re.compile(r"\b(?:not|never|no|isn['’]t|wasn['’]t|aren['’]t|don['’]t|doesn['’]t|didn['’]t|hardly|"
                           r"question(?:ed|s|ing)?|doubt(?:ed|s|ing)?|lack(?:ed|s|ing)?)\b(?:\W+\w+){0,4}\W*$", re.I)
# Feelings someone shared about their own situation ("he shared his concerns and frustrations") are not a view of them.
_VIEW_EXPRESSED = re.compile(r"\b(?:shar(?:e|es|ed|ing)|express(?:es|ed|ing)?|voic(?:e|es|ed|ing)|admit(?:s|ted|ting)?|"
                             r"describ(?:e|es|ed|ing)|told me about|talk(?:s|ed|ing)? about|open(?:s|ed)? up about)\s+"
                             r"(?:(?:his|her|their|your|our|my)\s+)?(?:\w+\s+(?:and|or)\s+)?$", re.I)
_VIEW_REPORTED = re.compile(r"\b(?:says?|said|claims?|claimed|insists?|insisted|thinks|called|accus\w+|wants? me to)\b", re.I)
_SECOND_PERSON = r"\b(?:you|your|yours|yourself)\b"


# Text between a person and a judgment that predicates it of them: "Colin's selfish", "Ethan's actions were indeed selfish".
_PREDICATED = re.compile(r"(?:['’]s(?:\s+[a-z-]+){0,2})?(?:['’]re|\s+(?:is|was|were|are|seems?|seemed|looks?|looked|has been|"
                         r"have been|had been|acted|appears?|appeared))?(?:\s+(?:so|very|really|quite|indeed|being|a|an|more|less|"
                         r"truly|clearly)){0,2}\s*", re.I)
# Text between a judgment and the person it is aimed at: "cautious about trusting Grace", "wary and not trust her".
_AIMED = re.compile(r"\s*(?:(?:and\s+)?(?:not|never)\s+)?(?:(?:about|of|toward|towards|around|with|in)\s+)?"
                    r"(?:(?:trust(?:ing)?|believ(?:e|ing)|rely(?:ing)? on)\s+)?", re.I)
_OBJECT_PRONOUN = re.compile(r"\b(?:her|him|them)\b", re.I)


def view_subjects(reason, people=(), *, target="", you=""):
    """Each judgment in a reason as (hostile|warm, person), skipping denials, reported views and contrasts.

    A judgment directly predicated of someone ("Colin's selfish actions") is about them; one aimed at someone
    ("cautious about trusting Grace", "wary and not trust her") is about that person, a her/him meaning the last
    person named; any other judgment is about the sentence's first-named person, since that is usually its subject.
    You/your is `you` (the conversation partner), which is the target unless the reason judges an absent person.
    A sentence naming nobody keeps the previous sentence's subject: "...seeing Colin take all of them. He only cares about himself."
    """
    you = you or target
    forms = {}
    for name in dict.fromkeys((target, *people)):
        if name:
            forms.setdefault(name.casefold(), name)
            first = name.split()[0].casefold()
            if sum(other.split()[0].casefold() == first for other in dict.fromkeys((target, *people)) if other) == 1:
                forms.setdefault(first, name)
    pattern = re.compile("|".join([*(name_pattern(f) for f in sorted(forms, key=len, reverse=True)), _SECOND_PERSON]), re.I)
    views, last, named = [], target, ""
    for sentence in split_sentences(unquoted_speech(reason)):
        matches = [m for m in pattern.finditer(sentence) if m.group().casefold() not in forms or is_name_reference(m, sentence)]
        refs = [(m.start(), m.end(), forms.get(m.group().casefold(), you)) for m in matches]
        names = [(m.start(), forms[m.group().casefold()]) for m in matches if m.group().casefold() in forms]
        subject = refs[0][2] if refs else last
        for kind, cue in (("hostile", _HOSTILE_VIEW), ("warm", _WARM_VIEW)):
            for match in cue.finditer(sentence):
                before, after = sentence[:match.start()], sentence[match.end():]
                if (_VIEW_NEGATED.search(before) or _VIEW_REPORTED.search(before) or _VIEW_EXPRESSED.search(before)
                        or re.search(r"\b(?:but|however|though|yet)\b", after, re.I)):
                    continue
                direct = [person for _, end, person in refs if end <= match.start()
                          and _PREDICATED.fullmatch(sentence, end, match.start())]
                aimed, aimed_at = _AIMED.match(sentence, match.end()), None
                if aimed.group().strip():
                    aimed_at = next((person for start, _, person in refs if start == aimed.end()), None)
                    if aimed_at is None and _OBJECT_PRONOUN.match(sentence, aimed.end()):
                        earlier = [person for start, person in names if start < aimed.end()]
                        aimed_at = earlier[-1] if earlier else named or None
                views.append((kind, direct[-1] if direct else aimed_at or subject))
        last = subject
        named = names[-1][1] if names else named
    return views


def states_no_change(reason):
    """A reason that says the view did not move is unchanged, however warmly it is worded."""
    return bool(re.search(r"\b(?:(?:feel|felt|feeling) no different|no different (?:about|toward|towards)|"
                          r"(?:didn['’]t|did not|doesn['’]t|does not|hasn['’]t|has not) change (?:my|how i)|"
                          r"(?:hasn['’]t|has not|haven['’]t|have not|hadn['’]t|had not) changed (?:my|how i)|"
                          r"(?:view|opinion) of \w+(?: \w+)? (?:hasn['’]t|has not|didn['’]t|did not) change(?:d)?|"
                          r"neither better nor worse|(?:didn['’]t|did not|don['’]t|do not) feel (?:any )?(?:worse or better|better or worse))\b",
                          unquoted_speech(reason), re.I))


def validate_reason_subject(reason, *, target, people=(), you=""):
    """A view of the target must judge the target, not only someone else the reason mentions."""
    views = view_subjects(reason, people, target=target, you=you)
    others = sorted({person for _, person in views if person != target})
    if others and not any(person == target for _, person in views):
        raise ValueError(f"This reason judges {' and '.join(others)}, not {target}. Say how {target}'s own words "
                         f"or conduct changed your view of {target}.")


def validate_opinion_direction(reason, delta, *, target="", people=(), you="", chosen=False):
    """The chosen direction must agree with a one-sided reason about the target; mixed reasons may go either way.

    chosen: the direction is the model's free choice, not a verdict the game fixed, so a grudge against
    someone else cannot carry it down."""
    every = view_subjects(reason, people, target=target, you=you)
    views = [kind for kind, person in every if person == target]
    hostile, warm = "hostile" in views, "warm" in views
    blamed = sorted({person for kind, person in every if kind == "hostile" and person != target})
    if chosen and delta < 0 and not views and blamed:
        # "Nachash's words made me see how selfish Sophia is" holds a grudge against Sophia, not the one who said it.
        raise ValueError(f"The reason holds its grudge against {' and '.join(blamed)}, not {target}. A worse view of "
                         f"{target} needs something {target} did; choose a change that agrees with the reason about {target}.")
    if delta > 0 and hostile and not warm:
        raise ValueError("The reason describes a worse view of this person, but the change chosen is better. "
                         "Choose a change that agrees with the reason.")
    if delta < 0 and warm and not hostile:
        raise ValueError("The reason describes a better view of this person, but the change chosen is worse. "
                         "Choose a change that agrees with the reason.")


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
               "seem", "too", "find", "way", "authority", "accept", "detail", "far", "listener",
               # Everyday words in any welcome or chat are not evidence of a specific memory.
               "when", "day", "time", "first", "last", "community", "people", "place", "make", "made", "best",
               "trying", "try", "found", "going", "know", "like", "little", "good", "really", "thing", "everyone",
               "together", "new", "lot", "bit", "help", "helped", "want", "wanted", "work", "worked"}
    expected = _content_words(assertion, names) - general
    actual = _content_words(text, names) - general
    overlap = expected & actual
    # Apples/fruit and trees are public garden observations. Those two concepts
    # alone do not disclose a biography about an orchard or saved seeds.
    if overlap <= {"apple", "tree"} and not re.search(
            r"\b(?:former|past|previous) life\b|\bused to\b", unquoted_speech(text), re.I):
        return False
    # Long memories need more shared detail before a chance overlap counts as retelling one.
    return bool(expected) and len(overlap) >= min(2 if len(expected) < 6 else 3, len(expected))


def validate_memory_ownership(text, accounts, *, speaker_id, names=(), recent_accounts=()):
    """Hearing a recollection does not make its event the listener's experience."""
    memories = [account for account in accounts if account.get("kind") == "former_life"]
    own = [account for account in memories if account.get("subject_id") == speaker_id]
    recent = [account for account in recent_accounts
              if account.get("kind") == "former_life" and account.get("subject_id") != speaker_id]
    for sentence in split_sentences(unquoted_speech(text)):
        if not re.search(r"\b(?:i|we)\s+(?:(?:also|still|once|clearly|vividly)\s+)*"
                         r"(?:remember|recall|worked|kept|helped|taught|drove|organized|saw|met|was an?)\b", sentence, re.I):
            continue
        # Remembering the telling is different from remembering the original event.
        if re.search(r"\b(?:remember|recall)\s+(?:you (?:telling|saying|mentioning|describing|explaining)|"
                     r"hearing|being told|your (?:story|account|recollection|description))\b", sentence, re.I):
            continue
        if any(mentions_private_recollection(sentence, account["assertion"], names) for account in own):
            continue
        for account in memories:
            if account.get("subject_id") == speaker_id:
                continue
            refers_back = (account in recent
                and re.search(r"\b(?:remember|recall)\s+(?:that|this|those|it|the same)\b", sentence, re.I)
                and (bool(_content_words(sentence, names) & _content_words(account["assertion"], names))
                     or re.search(r"\b(?:remember|recall)\s+(?:it|that|this)"
                                  r"(?:\s+(?:too|well|clearly|vividly|as well))?[.!?,;]", sentence, re.I)))
            if refers_back or mentions_private_recollection(sentence, account["assertion"], names):
                raise ValueError("That former-life memory belongs to another person. React to their experience or recall hearing their account; do not claim you remember living it or were there yourself.")


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
    elif kind == "conversation_seen":
        text = re.sub(r"\b(?:talk(?:ed|ing)?|spoke|speak(?:ing)?|convers(?:e[ds]?|ing)|chat(?:ted|ting)?)\b",
                      "talk", text, flags=re.I)
    elif kind == "killing":
        text = re.sub(r"\b(?:kill(?:ed|s|ing)?|murder(?:ed|s|ing)?|slew|slain)\b", "killed", text, flags=re.I)
    words = _content_words(text, names)
    if kind == "killing":
        # Narrator accounts use 'the player'; direct accusations address that same actor as 'you'.
        words.discard("player")
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


_HARVEST_VERBS = r"harvest\w*|pick\w*|took|tak(?:e|es|en|ing)|collect\w*|gather\w*|grabb?\w*|strip\w*"
_HARVEST_ACT = re.compile(rf"\b(?:{_HARVEST_VERBS})\b", re.I)
_HARVEST_APPLES = re.compile(r"\b(?:apples\b|apple\b(?!\s+tree)|fruit\b|\d+\b|"
                             r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:of\s+(?:them|those)|from)\b)", re.I)


def validate_account_speech(text, assertion, *, names=(), speaker=None, listener=None, kind=None, firsthand=False,
                            actor_references=None, known_events=()):
    """Require content evidence, rather than an ID or a promise to tell it later.

    This is a conservative vocabulary check, not a proof of semantic equivalence.
    """
    # A harvest fact records one pickup, not the collector's total. Speech may
    # discuss repeated pickups or the remaining stock without restating its count.
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
                    raise DialogueValidationError("The recollection invents an additional remembered event outside the selected account. Tell only its supplied memory and your present reaction.", RecoveryKind.UNMATCHED_ACCOUNT)
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
        # Accounts of different residents hearing the same doctrine are different
        # events. Shared theological words cannot substitute for the actual hearer.
        participant_forms = {form for actor in (speaker, listener, SNAKE_NAME) if actor for form in (actor, *actor.split())}
        for name in names:
            if name and name not in participant_forms and re.search(
                    rf"\b{name_pattern(name)}\s+heard\b", assertion, re.I):
                references = [*identifying_forms(name, names), *(actor_references or {}).get(name, [])]
                if not any(re.search(name_pattern(ref), spoken, re.I) for ref in references):
                    raise DialogueValidationError("The selected account is not actually told in this turn. "
                        f"Preserve who heard the snake: {name}. Expected snake_claim: {assertion}", RecoveryKind.UNMATCHED_ACCOUNT)
    missing_actor = speaker is not None and any(
        name not in (speaker, listener) and any(is_name_reference(match, assertion)
            for match in re.finditer(name_pattern(name), assertion, re.I))
        and not (kind == "snake_claim" and name != SNAKE_NAME)
        and not any(re.search(name_pattern(ref), spoken, re.I)
                    for ref in [*identifying_forms(name, names), *(actor_references or {}).get(name, []),
                                *(["the python", "the snake"] if name == SNAKE_NAME else [])])
        for name in names if name)
    if kind == "harvest":
        # A harvest is told by what was done and to which apples; "the apple tree" alone only names the place.
        told = bool(_HARVEST_ACT.search(spoken) and _HARVEST_APPLES.search(spoken))
    else:
        told = bool(expected) and len(expected & actual) >= min(2, len(expected))
    if missing_actor or not told:
        direction = ("Explain the supplied private reason and what you want, in your own words. " if kind == "private_motive" else
                     "Report who made which claims and preserve how you heard them. " if kind == "snake_claim" else
                     "Describe a concrete event or memory from its assertion in your own words. ")
        expected_account = ("Expected harvest: describe the collector's witnessed collection using the supplied harvest account and totals."
                            if kind == "harvest" else f"Expected {kind or 'information'}: {assertion}")
        raise DialogueValidationError("The selected account is not actually told in this turn. " + direction +
                         "Discussing trust or promising to tell it later is insufficient. "
                         + expected_account, RecoveryKind.UNMATCHED_ACCOUNT)


_WITNESSED_ACTS = {"harvest": r"harvest\w*|pick\w*|take|takes|taking|took|collect\w*|gather\w*",
                   "apple_eaten": r"eat|eats|eating|ate", "apple_given": r"give|gives|giving|gave|hand\w*"}


def validate_witness_denial(text, witnessed):
    """A witness never claims not to have seen what they saw; questioning what it meant stays allowed.

    witnessed lists (actor name, fact kind) for events this speaker saw firsthand.
    """
    speech = unquoted_speech(text)
    for name, kind in witnessed:
        acts = _WITNESSED_ACTS.get(kind)
        if not acts or not name:
            continue
        refs = "|".join(name_pattern(form) for form in dict.fromkeys((name, name.split()[0])))
        if re.search(rf"\bi\s+(?:didn['’]t|did not|never|haven['’]t|have not)\s+(?:actually\s+|personally\s+|really\s+|even\s+)?"
                     rf"(?:see|seen|saw|witness\w*|watch\w*|notice\w*)\s+(?:{refs})\s+(?:\w+\s+)?(?:{acts})\b", speech, re.I):
            raise ValueError(f"You saw {name} do this yourself. Do not claim you did not see it; you may question what it meant.")


def validate_listener_witness(text, seen, *, people, listener):
    """The listener is never told they saw an event they did not witness; asking whether they saw it stays allowed.

    seen lists (actor name, fact kind) for events the listener saw firsthand; people names everyone who could be meant.
    """
    for sentence in split_sentences(unquoted_speech(text)):
        if sentence.rstrip().endswith("?"):
            continue
        for name in people:
            for kind, acts in _WITNESSED_ACTS.items():
                if not name or (name, kind) in seen:
                    continue
                refs = "|".join(name_pattern(form) for form in dict.fromkeys((name, *identifying_forms(name, people))))
                if re.search(rf"\byou(?:['’]ve|\s+have)?\s+(?:just\s+|already\s+|also\s+)?(?:seen|saw|witnessed|watched)\s+"
                             rf"(?:{refs})\s+(?:\w+\s+)?(?:{acts})\b", sentence, re.I):
                    raise ValueError(f"{listener} did not see {name} do this firsthand. Say what you saw or heard yourself; "
                                     f"do not tell {listener} they witnessed it.")


def validate_speaker_witness(text, seen, *, people, speaker):
    """A speaker never says they saw an event they did not witness; hearing about it is told as hearing.

    seen lists (actor name, fact kind) for events the speaker saw firsthand or is deliberately lying about now.
    """
    for sentence in split_sentences(unquoted_speech(text)):
        if sentence.rstrip().endswith("?"):
            continue
        for name in people:
            for kind, acts in _WITNESSED_ACTS.items():
                if not name or name == speaker or (name, kind) in seen:
                    continue
                refs = "|".join(name_pattern(form) for form in dict.fromkeys((name, *identifying_forms(name, people))))
                if re.search(rf"\bi(?:['’]ve|\s+have)?\s+(?:(?:just|already|also|personally|actually|myself)\s+)*"
                             rf"(?:seen|saw|witnessed|watched)\s+(?:{refs})\s+(?:\w+\s+)?(?:{acts})\b", sentence, re.I):
                    raise ValueError(f"You did not see {name} do this yourself. Say only what you saw; if you heard "
                                     "about it, say you heard it, and from whom.")


def validate_promise_mentions(text, *, promisors, people, listener):
    """Speech may describe only apple offers and promises that were actually made, and each is for one apple.

    promisors names everyone with a real promise or offer this speaker knows of. The speaker's own new offer is
    governed by the offer record instead, so only other people and the listener ("you", "your") are checked here.
    """
    refs = {form.casefold(): name for name in people if name for form in (name, name.split()[0])}
    refs.update({"you": listener, "your": listener})
    who = "|".join(re.escape(ref) for ref in sorted(refs, key=len, reverse=True))
    count = r"(?P<count>\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|many|some|all(?: the)?|more)"
    patterns = (rf"\b(?P<who>{who})(?:['’]s)?\s+(?:offer|promise)\s+(?:of|to (?:give|bring|share|save)(?:\s+\w+){{0,2}})\s+{count}\s+(?:apples?|fruits?)\b",
                rf"\b(?P<who>{who})\s+(?:(?:has|had|have|already)\s+)?(?:offered|promised)\s+(?:to\s+(?:give|bring|share|save)\s+)?"
                rf"(?:\w+\s+){{0,2}}{count}\s+(?:apples?|fruits?)\b")
    for sentence in split_sentences(unquoted_speech(text)):
        if sentence.rstrip().endswith("?"):
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, sentence, re.I):
                if re.search(r"\b(?:if|imagine|suppose|unless)\s+(?:\w+\s+)?$", sentence[:match.start()], re.I):
                    continue
                person = refs[match["who"].casefold()]
                if match["count"].casefold() not in ("a", "an", "one"):
                    raise ValueError("An apple offer or promise is always for one apple. Do not invent a promise of several apples.")
                if person not in promisors:
                    raise ValueError(f"{person} made no apple offer or promise that you know of. Speak only about promises "
                                     "actually made; do not invent one.")


def validate_apple_promise(offer, acceptance, deadline, *, current_day=None):
    validate_apple_offer(offer, deadline, current_day=current_day)
    validate_apple_acceptance(acceptance)


def validate_apple_offer(offer, deadline, *, current_day=None):
    offer = unquoted_speech(offer)
    giving = re.search(r"\bi\s+(?:(?:will|shall|can|could|promise to|intend to|plan to)\s+)?"
                      r"(?:give|bring|deliver|share|save|offer)\b|\bi['’]ll\s+(?:give|bring|deliver|share|save)\b|"
                      r"\bi(?:['’]ll|\s+(?:will|can))?\s+get you (?:an?|one) (?:apple|fruit)\b", offer, re.I)
    negative = r"\b(?:cannot|can['’]t|won['’]t|wouldn['’]t|will not|do not|don['’]t|decline|refuse|maybe|perhaps|might)\b"
    if not giving or not re.search(r"\b(?:apple|fruit)\b", offer, re.I) or re.search(negative, offer, re.I):
        raise DialogueValidationError("A recorded apple promise needs the giver's explicit spoken commitment to give an apple, not a request or unrelated discussion.", RecoveryKind.MISSING_PROMISE)
    day = re.search(r"\b(?:day\s+|(?:by|on|before)\s+(?:day\s+)?)(\d+)\b", offer, re.I)
    tomorrow = bool(re.search(r"\btomorrow\b", offer, re.I))
    today = bool(re.search(r"\btoday\b", offer, re.I))
    valid = (bool(day or tomorrow or today)
             and (day is None or int(day[1]) == deadline)
             and (not tomorrow or current_day is not None and deadline == current_day + 1)
             and (not today or current_day is not None and deadline == current_day)
             and not re.search(r"\b(?:tonight|night|nightfall)\b", offer, re.I))
    if not valid:
        raise ValueError("State the apple promise's actual deadline day in the offer.")


def validate_apple_acceptance(acceptance):
    acceptance = unquoted_speech(acceptance)
    negative = r"\b(?:cannot|can['’]t|won['’]t|wouldn['’]t|will not|do not|don['’]t|decline|refuse|maybe|perhaps|might)\b"
    if ("?" in acceptance or re.search(negative, acceptance, re.I)
            or not re.search(r"\b(?:yes|agreed|deal|that works|i accept|i agree|thank you|thanks|i['’]ll take)\b", acceptance, re.I)):
        raise DialogueValidationError("The apple receiver must explicitly accept the spoken promise before it can be recorded.", RecoveryKind.MISSING_ACCEPTANCE)


def validate_continuity(text, *, day, situation, knows_claims=False, listener_is_snake=False):
    speech = unquoted_speech(text).replace("\u2019", "'")
    if day == 1 and re.search(r"\b(?:yesterday|(?:a few|several|\d+) days ago|last (?:night|week))\b", speech, re.I) and re.search(
            rf"{name_pattern(SNAKE_NAME)}|\b(?:garden|authority|claims)\b", speech, re.I):
        raise ValueError("This is day one. Refer to earlier today for garden events, preserving unknown former-life dates.")
    guard = rf"(?:{name_pattern(SNAKE_NAME)}|the (?:snake|python)|it|him|her"
    guard += r"|you)\b" if listener_is_snake else r")\b"
    if situation.get("prior_snake_encounters", 0) and re.search(
            r"\bi(?:\s+(?:have not|haven't|have never|never|did not|didn't)|'ve (?:never|not))\s+"
            r"(?:(?:yet|ever|actually|personally|really)\s+)*"
            r"(?:met|meet|encountered|encounter|(?:spoken|speak|talked|talk) (?:to|with)|dealt with|deal with|"
            r"(?:thought about|considered) (?:speaking|talking) (?:to|with))\s+"
            + guard, speech, re.I):
        raise DialogueValidationError("The speaker has already met the named snake. Doubting authority does not erase that encounter.", RecoveryKind.KNOWN_ENCOUNTER)
    if knows_claims and re.search(rf"{name_pattern(SNAKE_NAME)}|\bclaims\b", speech, re.I) and re.search(
            r"\bi['’]?(?:ve| have)?\s+(?:heard nothing|haven['’]t heard anything|have not heard anything)\b", speech, re.I):
        raise ValueError("The speaker has firsthand knowledge of the claims being made; preserve that experience while allowing skepticism.")
    if "prior_snake_encounters" in situation and not situation["prior_snake_encounters"] and not knows_claims:
        if re.search(rf"\bi\s+(?:have\s+)?(?:met|heard|talked to|spoken to|spoken with)\s+{re.escape(SNAKE_NAME)}\b", speech, re.I):
            raise ValueError("No earlier encounter or firsthand hearing is supplied for this speaker. Describe the present uncertainty without inventing a meeting.")


def validate_apple_roles(text, *, giver, receiver, speaker, apple_received=False):
    speech = unquoted_speech(text)
    if speaker == receiver and re.search(
            r"\b(?:apple|fruit)\s+i\s+(?:offered|gave|handed|provided)\s+you\b|\bi\s+(?:gave|handed)\s+you\s+(?:an?|the|one)\s+(?:apple|fruit)\b|"
            r"(?:^|[.!?;,]\s+)(?:please\s+)?(?:take|accept)\s+(?:the|this|an?|one)\s+(?:apple|fruit)\b|"
            r"\bi(?:['’]ll|\s+(?:will|can|shall))\s+(?:give|hand|offer)\s+you\s+(?:an?|the|one|this)\s+(?:apple|fruit)\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}. Keep the original giver and receiver in their actual roles.")
    if speaker == giver and re.search(r"\bapple\s+you\s+(?:agreed|promised)\s+to\s+(?:give|provide)\b", speech, re.I):
        raise ValueError(f"{giver} provides the apple; preserve {receiver} as the receiver.")
    if speaker == giver and re.search(
            r"\byou\s+(?:(?:will|should|must|can|shall|need to|have to)\s+)?(?:give|hand|bring|offer)\s+me\s+"
            r"(?:an?|the|one|your|this)\s+apple\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}; do not ask the resident to supply it.")
    if speaker == giver and re.search(r"\bi\s+(?:accept|receive|take)\s+(?:the|an?|your|this)\s+apple\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple; only {receiver} accepts it. Preserve the settled handoff.")
    if speaker == giver and re.search(r"\b(?:can|could|may)\s+i\s+(?:have|get|receive)\s+(?:an?|one|the|your)\s+apple\b", speech, re.I):
        raise ValueError(f"{giver} gives the apple to {receiver}. Do not ask the resident to give you the apple.")
    if apple_received:
        debtor = "i" if speaker == receiver else "you"
        if speaker == giver and re.search(r"\byou(?:['\u2019]ll| will)\s+(?:get|receive|have|be given)\b"
                                          r"[^.!?]{0,55}\b(?:when|after|once|if)\b", speech, re.I):
            raise ValueError("The apple has already been handed over. Do not describe it as a future transfer.")
        if re.search(rf"\b{debtor}\s+(?:(?:simply|actually|really)\s+)?"
                     r"(?:have yet to|have not(?: yet)?|haven['’]t(?: yet)?|did not|didn['’]t|never)\s+"
                     r"(?:been\s+)?(?:receive[d]?|get|got|taken?|given|offered)\s+(?:(?:an?|the|one|any|your|this)\s+)?(?:apple|fruit)\b", speech, re.I):
            raise ValueError("The apple was already received. Doubting the giver does not undo the recorded handoff.")
        if _asserts(_APPLE_REFUSAL, speech):
            raise ValueError("The resident received the apple. Do not undo the committed handoff.")


def harvest_extent(account, tree_stock, current_time):
    """Infer a stock share only with complete firsthand coverage of the current day."""
    if type(tree_stock) is not int or account.get("event_time") != current_time or account.get("basis") != "firsthand":
        return None
    rows = account.get("harvest_context", [])
    collector = next((r for r in rows if r["subject_id"] == account.get("subject_id")), None)
    if not collector or not collector.get("all_harvests_witnessed"):
        return None
    stock = sum(r["quantity"] for r in rows) + tree_stock
    quantity = collector["quantity"]
    return dict(share="all" if quantity == stock else "most" if quantity * 2 > stock else "some",
                day_stock=stock, remaining=tree_stock)


def validate_harvest_grounding(text, accounts, *, speaker_id, listener_id, current_time, tree_stock=None, focus=None,
                               prefer_all=False):
    """Check explicit totals and visible stock; never require a spoken number."""
    scope = (focus or {}).get("event_time") or current_time
    totals = {}
    extents = {}
    other_days = {}  # Witnessed totals for the same collector on other days, e.g. yesterday's harvest.
    for account in accounts:
        if account.get("kind") != "harvest":
            continue
        for row in account.get("harvest_context", []):
            if row.get("event_time") == scope:
                totals[row["subject_id"]] = row
            else:
                other_days.setdefault(row["subject_id"], set()).add(row["quantity"])
        extent = harvest_extent(account, tree_stock, current_time)
        if extent:
            extents[account["subject_id"]] = extent
    number = r"(?:\d+|zero|no|none|" + "|".join(SPOKEN_NUMBERS) + ")"

    def value(word):
        return int(word) if word.isdigit() else 0 if word.lower() in ("zero", "no", "none") else SPOKEN_NUMBERS[word.lower()]

    for sentence in split_sentences(unquoted_speech(text)):
        focal_id = (focus or {}).get("subject_id") or (next(iter(totals)) if len(totals) == 1 else None)
        if prefer_all and extents.get(focal_id, {}).get("share") == "all" and not re.search(
                r"\b(?:if|imagine|suppose|wish)\b", sentence, re.I) and re.search(
                r"\b(?:take|taking|took|taken|harvest(?:ed|s|ing)?|collect(?:ed|s|ing)?|pick(?:ed|s|ing)?)\s+"
                r"(?:most|almost all|nearly all)\s+(?:of\s+)?(?:(?:the|this|that)\s+)?(?:apples|fruit|stock)\b", sentence, re.I):
            raise ValueError("This collector took every apple the tree had at the time. Describe it as all of them, "
                             "rather than most; preserve that fact when answering the listener.")
        if sentence.rstrip().endswith("?") or re.search(r"\b(?:if|would|could|might|imagine|suppose|wish)\b", sentence, re.I):
            continue
        historical = bool(re.search(r"\b(?:yesterday|last night|previous phase|previous day)\b", sentence, re.I))
        pickup = bool(re.search(r"\b(?:at a time|each (?:time|pickup|visit)|per (?:pickup|visit)|"
                                r"(?:first|second|last|single|one) (?:pickup|visit)|on one occasion)\b", sentence, re.I))
        now = bool(re.search(r"\b(?:now|currently|right now)\b", sentence, re.I))
        if type(tree_stock) is int and (now or not historical and not pickup):
            count = rf"(?P<count>{number})\s+(?:apples?|fruit)\b"
            stock_patterns = [
                rf"\b(?:apple )?tree\s+(?:still\s+)?(?:has|holds|contains|is left with)\s+(?:still\s+)?(?:(?:only|just|exactly)\s+)?{count}",
                rf"\bthere\s+(?:are|is)\s+(?:still\s+)?{count}\s+(?:still\s+)?(?:left|remaining|on (?:the|this) (?:apple )?tree)",
                rf"\b{count}\s+(?:remain|are left|are remaining)\s+on (?:the|this) (?:apple )?tree",
                rf"\b(?:leaving|leaves|left|leave)\s+(?:behind\s+)?(?:(?:only|just|exactly)\s+)?{count}(?:\s+(?:behind|on (?:the|this) (?:apple )?tree))?",
            ]
            for pattern in stock_patterns:
                for match in re.finditer(pattern, sentence, re.I):
                    # Apples left in a pocket or given to someone are not tree stock.
                    if re.match(r"\s+(?:in|inside|for|with|to|at)\b", sentence[match.end():], re.I):
                        continue
                    if pattern == stock_patterns[-1] and not re.search(r"\btree\b", sentence, re.I):
                        continue
                    if pattern == stock_patterns[-1] and scope != current_time and not now:
                        continue  # A past harvest does not establish the tree's current remainder.
                    if value(match["count"]) != tree_stock:
                        raise ValueError(f"The visible tree currently has {tree_stock} apples. "
                            "Use that remaining stock; do not subtract one pickup from population or confuse collected fruit with fruit left on the tree.")
            empty = re.search(r"\b(?:apple )?tree\s+(?:is|stands)\s+(?:now\s+)?(?:empty|bare)\b", sentence, re.I)
            if empty and tree_stock != 0:
                raise ValueError(f"The visible tree still has {tree_stock} apples; it is not empty.")
            stocked = re.search(r"\b(?:apple )?tree\s+(?:still\s+)?has\s+(?:(?:some|many|any|several|a few|plenty of)\s+)?(?:apples|fruit)\b", sentence, re.I)
            if stocked and tree_stock == 0:
                raise ValueError("The visible tree has zero apples; do not claim it still has fruit.")
            if scope == current_time and totals and tree_stock == 0 and re.search(
                    r"\b(?:tree|harvest(?:ed|s|ing)?|pick(?:ed|s|ing)?|collect(?:ed|s|ing)?)\b", sentence, re.I) and re.search(
                    r"\b(?:leaving|leaves|left|leave)\s+(?:some|a few|several|many)\s+(?:apples?\s+|fruit\s+)?(?:for (?:others|everyone|anyone)|behind|on the tree)\b", sentence, re.I):
                raise ValueError("No apples remain on the tree. Do not claim this harvest left some tree fruit for others.")
        if pickup or historical and scope == current_time:
            continue
        if totals and listener_id not in totals and re.search(
                r"\byou(?:['\u2019](?:ve|d)|\s+(?:have|had))?\s+(?:(?:just|already|recently|only)\s+){0,3}"
                r"(?:harvest(?:ed)?|collect(?:ed)?|pick(?:ed)?(?: up)?|took|take)\s+(?:(?:only|exactly|all|the|last|just|at least|more than)\s+){0,3}"
                rf"{number}\s+(?:apples?|fruit)\b", sentence, re.I):
            collectors = " and ".join(row["subject_name"] for row in totals.values())
            raise ValueError(f"On this account's day the harvest was by {collectors}, not by the listener. "
                             "Attribute it to the actual collector; do not tell the listener they harvested.")
        for identity, row in totals.items():
            name = row["subject_name"]
            refs = [name_pattern(name)]
            first = name.split()[0] if name else ""
            if first and sum(r["subject_name"].split()[0] == first for r in totals.values()) == 1:
                refs.append(name_pattern(first))
            if identity == speaker_id:
                refs.append("I")
            elif identity == listener_id:
                refs.append("you")
            elif identity == focal_id:
                refs.extend(["he", "she", "they"])
            subject = "(?:" + "|".join(refs) + ")"
            if prefer_all and extents.get(identity, {}).get("share") == "all" and re.search(
                    rf"\b{subject}\s+(?:did not|didn['\u2019]t|has not|hasn['\u2019]t|had not|hadn['\u2019]t|never)\s+"
                    r"(?:take|taken|harvest|harvested|collect|collected|pick|picked)\s+all\s+(?:of\s+)?(?:the\s+)?(?:apples|fruit)\b", sentence, re.I):
                raise ValueError(f"{name} took every apple the tree had at the time. Do not deny that they took all of them.")
            extent = extents.get(identity, {})
            # The reverse inflation: a majority or part of the stock is not all of it.
            if extent.get("share") in ("most", "some") and (
                    re.search(rf"\b{subject}\b", sentence, re.I)
                    or identity == focal_id and re.search(r"\b(?:him|her|them)\b", sentence, re.I)) and re.search(
                    r"\b(?:took|taken|taking|takes|harvested|collected|picked)\s+(?:all|every|everything|the (?:whole|entire))\b"
                    r"(?!\s+(?:of\s+)?(?:his|her|their|my|your)\b)|\b(?:without leaving (?:any|one)|leaving (?:none|nothing|no apples)|"
                    r"left (?:none|nothing|no apples)|emptied the tree|stripped the tree)\b", sentence, re.I) and not re.search(
                    r"\b(?:not|never|didn['’]t|did not|wasn['’]t)\b", sentence, re.I):
                raise ValueError(f"{name} took {row['quantity']} of the {extent['day_stock']} apples the tree had then, not all of them; "
                                 "someone else harvested the rest. Do not say they took everything or left none.")
            pattern = (rf"\b{subject}(?:['\u2019](?:ve|s|d)|\s+(?:has|have|had))?\s+"
                r"(?:(?:just|already|recently|only)\s+){0,3}"
                r"(?:harvest(?:ed|s)?|collect(?:ed|s)?|pick(?:ed|s)?(?: up)?|took|take|takes)\s+"
                rf"(?P<bound>at least\s+|more than\s+)?(?:(?:only|exactly|all|the|last|just)\s+){{0,3}}"
                rf"(?P<count>{number})\s+(?:apples?|fruit)\b")
            for match in re.finditer(pattern, sentence, re.I):
                count, total = value(match["count"]), row["quantity"]
                bound = (match["bound"] or "").strip().lower()
                supported = count <= total if bound == "at least" else count < total if bound == "more than" else count == total
                if not supported and not bound and count in other_days.get(identity, ()) and not re.search(
                        r"\b(?:today|this (?:morning|day|phase)|right now|currently)\b", sentence, re.I):
                    continue  # An accurate total from another witnessed day.
                if not supported:
                    raise ValueError(f"The witnessed total for {name} on this account's day is {total} apples. "
                        "Do not present one pickup as the whole harvest. You may omit numbers or explicitly distinguish an individual pickup from the total.")


_ACCUSATION = re.compile(
    r"\b(?:selfish\w*|greed\w*|hoard\w*|stingy|only cares? about|cares? (?:nothing|little) (?:for|about)|"
    r"thinks? only (?:of|about)|did(?:n['’]t| not) (?:share|leave|save|think)|without (?:sharing|leaving|a thought)|"
    r"took (?:all|every|everything)|left (?:nothing|none|no apples)|kept (?:them|it|everything|the apples)|"
    r"can(?:not|['’]t) be trusted|untrustworth\w*|do(?:n['’]t| not) trust|beware|kill\w*|murder\w*|dangerous|"
    # Insinuation counts too: "I wonder how much she cares about anyone else."
    r"wonder (?:whether|if|how much|what)\b[^.?!]*\b(?:cares?|spared?|leaves?|left|anyone else|everyone else|for you))\b", re.I)


def first_person_text(text):
    """Compare recited text regardless of contractions or of being rewritten from you/your into I/my."""
    text = text.casefold().replace("’", "'")
    for pattern, replacement in ((r"n't\b", " not"), (r"'ve\b", " have"), (r"'re\b", " are"), (r"'m\b", " am"),
                                 (r"'ll\b", " will"), (r"'d\b", " would"), (r"\byou are\b", "i am"), (r"\byou were\b", "i was"),
                                 (r"\byours\b", "mine"), (r"\byour\b", "my"), (r"\byou\b", "i")):
        text = re.sub(pattern, replacement, text)
    return text


def copied_share(text, sources, *, run=5):
    """Fraction of the text's word runs that appear in the sources, after first_person_text."""
    def runs(value):
        words = re.findall(r"[a-z']+", first_person_text(value or ""))
        return [tuple(words[i:i + run]) for i in range(len(words) - run + 1)]
    own = runs(text)
    if not own:
        return 0.0
    known = {gram for source in sources for gram in runs(source)}
    return sum(gram in known for gram in own) / len(own)


def _word_runs(text, run):
    words = re.findall(r"[a-z']+", text.casefold().replace("’", "'"))
    return {tuple(words[i:i + run]) for i in range(len(words) - run + 1)}


def _any_person_text(text):
    """Compare recited text regardless of whom it was rewritten to be about: 'your goal' and 'her goal' match."""
    return re.sub(r"\b(?:i|me|my|mine|myself|you|your|yours|yourself|he|him|his|himself|she|her|hers|herself|"
                  r"they|them|their|theirs|themselves)\b", "someone", first_person_text(text))


def validate_private_recital(text, *, personality="", motives=(), run=6):
    """Speech acts on private personality and motives; it never reads their wording or record fields aloud."""
    speech = unquoted_speech(text)
    if re.search(r"\b[A-Za-z]+_[A-Za-z_]+\b", speech) or re.search(
            r"\b(?:current (?:listener|speaker|task)|private (?:motive|context|accounts?)|immediate objective|"
            r"(?:my|your) goal is to|claims? database|(?:with )?an intensity of \d+|permitted accounts?|the transcript|"
            r"named collector|witnessing is not collecting|phase['’]s (?:tree )?stock|tree stock|visiting is travel only|"
            r"speech never transfers apples|successful local harvest|without permission, payment or quotas|"
            # The python's task script ("Keep pressing that accusation", "then insist it shows ...").
            r"keep pressing (?:that|the|this|my) accusation|(?:then )?insist (?:that )?it shows|i have a task for you|"
            # Account fields read aloud: "An account disputes: ...", "a believed stance", "As the original speaker".
            r"an account disputes|(?:a|my|your|the|their) (?:believed|uncertain|disbelieved) stance|"
            r"stance: (?:believed|uncertain|disbelieved)|as the original (?:speaker|source)|"
            r"(?:don['’]t|do not) apologi[sz]e or threaten|"
            # World rules and layout notes read aloud: "Harvesting requires positive stock and local collection",
            # "Stock is finite and replenishes...", "recorded in the garden's account book", "This is all I could see."
            r"positive stock|local collection|stock is finite|cannot harvest (?:\w+\s+){0,2}again until|account book|"
            r"this is all i could see|confirmed it by walking)\b",
            speech, re.I) or re.search(
            # A bare imperative is the stance field, not the speaker's own "I believe it's true".
            r"(?:^|[.!?]\s+)believe (?:it|this|that)(?:['’]s| is) true\b", speech, re.I):
        raise ValueError("Speech reads internal instructions or record fields aloud. Say only natural spoken words.")
    # Motives are never spoken, in any person. A character may briefly voice their own trait,
    # but personality wording kept in the second person describes the listener with it,
    # and a long verbatim run recites the profile rather than speaking from it.
    recited = any(_word_runs(_any_person_text(speech), run) & _word_runs(_any_person_text(motive or ""), run) for motive in motives)
    recited = recited or any({"you", "your"} & set(words) for words in _word_runs(speech, run) & _word_runs(personality or "", run))
    recited = recited or bool(_word_runs(_any_person_text(speech), 12) & _word_runs(_any_person_text(personality or ""), 12))
    if recited:
        raise ValueError("Speech recites your private personality or motive. Let it guide what you do and say "
                         "in your own words; never read it aloud or describe the listener with it.")


_GREETING = r"(?:hey|hi|hello|oh|well|yes|no|thanks|thank you|listen|look)"
_AFTER_VOCATIVE = (r"(?:i|you|we|it['’]s|have|did|do|are|can|could|would|listen|look|please|what|how|thanks?|"
                   r"there|let['’]s|so|well|yes|no|beware|remember)\b")


def validate_addressee(text, *, speaker_name, listener_name, names):
    """Greetings and direct address name only the actual listener, never a bystander or an absent player."""
    exempt = {form.casefold() for name in (speaker_name, listener_name) for form in (name, *name.split()) if form}
    speech = unquoted_speech(text)
    for name in sorted(set(names), key=len, reverse=True):
        for form in dict.fromkeys((name, name.split()[0] if name else "")):
            if not form or form.casefold() in exempt:
                continue
            ref = name_pattern(form)
            if (re.search(rf"^\W*(?:{_GREETING}\W+)?{ref}\s*(?:!|,\s*(?={_AFTER_VOCATIVE}))", speech, re.I)
                    or re.search(rf"\b(?:hey|hi|hello|thanks|thank you)\W+{ref}\b", speech, re.I)
                    or re.search(rf",\s*{ref}\s*[.!?]", speech, re.I)
                    # "I've seen her too, Player, and..." but not a list such as "Judy, Kevin, and I".
                    or re.search(rf"\b[a-z][a-z'’]*,\s*(?i:{ref})\s*,\s*(?i:and|but|so|i|you|we)\b", speech)):
                raise ValueError(f"You are speaking to {listener_name}. Address {listener_name} as you; "
                                 f"do not greet or address {form}, who is not your listener.")


def validate_unheard_quote(text, name):
    """Claims about what someone said need an account heard from them; witnessing conduct is not hearing words."""
    refs = "|".join(name_pattern(form) for form in dict.fromkeys((name, name.split()[0])) if form)
    if re.search(rf"(?:{refs})\s+(?:(?:did|even|once|also|actually)\s+)?(?:say|said|says|told|tells|mentioned|claimed|"
                 r"admitted|bragged|talked about)\b", unquoted_speech(text), re.I):
        raise ValueError(f"You never heard {name} say anything about this. Report only what you witnessed "
                         f"{name} do; do not invent what {name} said or meant.")


def leverage_pressure(text):
    """Recorded pressure needs pressure in the words themselves, not just a remembered account."""
    # "If you don't mind" is politeness, not a threat.
    return bool(re.search(r"\b(?:unless|or else|if you (?:(?:don['’]t|do not)(?!\s+mind\b)|won['’]t|refuse)|you owe|owe me|"
                          r"i know (?:that |what |about )?you|i (?:will|['’]ll|could|might) tell|keep (?:it|this|that) (?:quiet|secret|between us)|"
                          r"remember (?:when|that|how) you|after what you|you promised)\b", unquoted_speech(text), re.I))


def validate_tree_permanence(text):
    """The tree's stock changes: residents harvest it and it refills at the start of each new day."""
    for sentence in split_sentences(unquoted_speech(text)):
        if sentence.rstrip().endswith("?") or not re.search(r"\b(?:tree|apples?|fruit|branches|harvest\w*)\b", sentence, re.I):
            continue
        for match in re.finditer(
                r"\b(?:always|forever|permanently|constantly)\s+(?:(?:be|been|being|stays?|stayed|remains?|remained|is|was|"
                r"going to be|seems?(?: to(?: be)?)?|looks?)\s+)?(?:empty|bare|barren|full|fruitless|apple-less|available|"
                r"out of (?:apples|fruit)|without (?:apples|fruit)|(?:has|had|have) (?:no|plenty of|lots of|a bounty of) (?:apples|fruit))\b"
                r"|\bnever\s+(?:(?:be|get|have|has|see|grow|bear)s?\s+)?(?:any\s+|a\s+)?(?:apples|fruit|harvest)\s+again\b",
                sentence, re.I):
            if not re.search(r"\bnot\s+$", sentence[:match.start()], re.I):
                raise ValueError("The tree is not permanently empty or full: residents harvest it, and it refills at each "
                                 "new day. Describe its current state or a harvest you saw, not a permanent condition.")


def validate_name_tag_talk(text, question=""):
    """Everyone already reads name tags; they come up only when someone asks how a name is known."""
    if re.search(r"\bname[- ]?tags?\b", unquoted_speech(text), re.I) and not re.search(
            r"\bname[- ]?tags?\b|\bhow\b[^.?!]*\b(?:know|knew|learn\w*|get|got)\b[^.?!]*\bname\b", question, re.I):
        raise ValueError("Everyone can already read each other's floating name tags. Do not mention name tags "
                         "unless asked how you know a name; talk about the actual concern.")


def validate_self_name(text, name):
    """A speaker calls themselves I/me; their own full name appears only when introducing themselves."""
    speech = unquoted_speech(text)
    for match in re.finditer(name_pattern(name), speech, re.I):
        if not re.search(r"\b(?:i am|i['’]m|my name is|call me|it['’]s|it is|this is|i)\s*,?\s*$", speech[:match.start()], re.I):
            raise ValueError(f"You are {name}. Use I/me/my for yourself; say your own name only to introduce yourself.")


def validate_echo(text, previous):
    """A reply answers the partner's last words instead of repeating them as its own."""
    if previous and copied_share(text, [previous]) >= 0.5:
        raise ValueError("This reply mostly repeats the partner's previous words. Answer them in your own words.")


def validate_own_memory_person(text, assertion, own_telling):
    """A speaker's own memory is told as I/my; reading out its third-person account makes someone else its owner.

    Only the wording that differs from the owner's first-person telling counts: "they" there means the owner,
    while a "they" both versions share is another person in the memory.
    """
    retold = _word_runs(assertion, 4) - _word_runs(own_telling, 4)
    if _word_runs(unquoted_speech(text), 4) & retold:
        raise ValueError("This memory is your own. Tell it with I/my as something you did or felt; "
                         "saying it as they/their makes someone else its owner.")


_EXCUSE = re.compile(
    r"\b(?:(?:had|has|have)\s+(?:his|her|their)\s+(?:own\s+)?reasons?|benefit of the doubt|can(?:not|['’]t)\s+(?:always\s+)?judge|"
    r"(?:isn['’]t|is not|wasn['’]t|was not|aren['’]t|are not|weren['’]t|were not|not)\s+(?:necessarily\s+|really\s+)?"
    r"(?:selfish|greedy|stingy|thoughtless)|(?:don['’]t|do not) think\b[^.!?]*\b(?:selfish|greedy|stingy)|"
    r"doesn['’]t (?:necessarily )?mean (?:he|she|they)(?:['’]s|['’]re| is| are)? (?:selfish|greedy|stingy))\b", re.I)


def validate_snake_stance(text, accused, listener, people=()):
    """The python presses its accusation: it never excuses the accused or turns the charge on the listener."""
    accused_ref = re.compile("|".join([*(name_pattern(form) for form in dict.fromkeys((accused, accused.split()[0])) if form),
                                       r"\b(?:he|she|they|him|his|her|their|them)\b"]), re.I)
    anyone = re.compile("|".join([accused_ref.pattern, _SECOND_PERSON,
                                  *(name_pattern(name) for name in people if name and name != accused)]), re.I)
    for sentence in split_sentences(unquoted_speech(text)):
        if sentence.rstrip().endswith("?") or re.search(r"\byou (?:say|said|think|believe|claim|feel)\b", sentence, re.I):
            continue
        excuse = _EXCUSE.search(sentence)
        first = anyone.search(sentence)
        # "You gave one to Ada, and that's not selfish, but Colin's actions were" praises the listener instead.
        if excuse and (accused_ref.search(excuse.group()) or first and accused_ref.fullmatch(first.group())):
            raise ValueError(f"You believe {accused} acted selfishly. Do not excuse or defend {accused}; "
                             f"keep pressing your accusation in your own words, even when {listener} disagrees.")
        if re.search(r"\byou(?:['’]re| are| were)\s+(?:(?:so|being|just|really|very|quite)\s+)*(?:selfish|greedy|stingy|"
                     r"thoughtless|inconsiderate|untrustworthy|a hoarder)\b", sentence, re.I):
            raise ValueError(f"Your accusation is against {accused}, not {listener}. "
                             f"Speak to {listener} as someone you want on your side against {accused}.")


_DISBELIEF = re.compile(
    r"\b(?:(?:don['’]t|do not|didn['’]t|did not)\s+(?:really\s+|quite\s+)?believe\s+"
    r"(?:you|your|him|them|it|that|this|what|a word|the account|any of)|(?:that|it|this)(?:['’]s|\s+is|\s+was)\s+(?:simply\s+|just\s+)?"
    r"(?:not true|untrue|false|a lie)|(?:isn['’]t|wasn['’]t|is not|was not|never) true|unbelievable|"
    r"false (?:information|accounts?|claims?|story|stories|accusations?)|(?:you|he|she|they)(?:['’]re|\s+are|\s+is|['’]s)\s+lying|"
    r"made (?:it|that|this) up)\b", re.I)
_RETRACTION = re.compile(
    r"\b(?:too quick to judge|(?:might|may|could) have been (?:wrong|mistaken|unfair|too harsh|too quick)|misjudg\w*|"
    r"(?:initially|at first) (?:skeptical|wrong|mistaken|unsure)|was wrong about|changed my (?:mind|view) about|"
    r"alternative motivations?|(?:her|his|their) kind heart)\b", re.I)


_WARM_TALK = re.compile(
    r"\b(?:love (?:your|you|how you|that you)|(?:glad|happy|lucky|grateful) (?:to have you|we['’]ve got|we have|you['’]re|for you)|"
    r"each other['’]s backs?|(?:i['’]m|i am) with you|you['’]re (?:(?:so|really|such|very|truly)\s+)*(?:great|wonderful|amazing|sweet|"
    r"kind|generous|thoughtful|a good|a true|a real)|(?:trust|admire|adore) you|my (?:dear )?friend)\b", re.I)
# Agreeing with someone or joining their plans: warmth too, from a speaker who is very hostile toward them.
_COOPERATIVE_TALK = re.compile(
    r"\b(?:(?:great|good|wonderful|fantastic|brilliant|excellent|lovely) (?:idea|point|plan|thought|suggestion)|"
    r"i(?:['’]m| am)? (?:(?:totally|completely|definitely|really|fully)\s+)?(?:agree with you|with you on)|"
    r"(?:willing|happy|glad) to work (?:it out )?together|work (?:on )?(?:it|this) together|count me in)\b", re.I)
# "I always seem to have an apple to spare", "I've got two apples", "my spare apple".
_OWN_APPLES = re.compile(
    r"\bi(?:['’]ve| have)?\s+(?:(?:always|still|usually|often|already|really|just|seem to|tend to|also)\s+)*"
    r"(?:have|carry|carrying|keep|hold|holding|(?<=['’]ve\s)got|(?<=have\s)got)\s+(?:(?:an?|some|plenty of|a few|lots of|extra|spare|more|one|two|three|"
    r"four|five|\d+)\s+)*apples?\b|\bmy\s+(?:extra|spare)\s+apples?\b|\bi\b[^.!?]{0,40}\bapples?\s+to\s+spare\b", re.I)
_OWN_HARVEST = re.compile(
    r"\bi(?:['’]ve| have)?\s+(?:(?:already|just|also|only)\s+)*(?:took|taken|harvested|picked|gathered|collected)\s+"
    r"(?:(?:an?|some|all|most|the|of|those|these|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)*apples?\b", re.I)


def validate_hostile_warmth(text, listener, *, feeling, cause="", people=()):
    """Speech to someone the speaker is hostile toward cannot be warm toward them."""
    listener_ref = re.compile("|".join([_SECOND_PERSON, *(name_pattern(form) for form in dict.fromkeys((listener, listener.split()[0])))]), re.I)
    for sentence in split_sentences(unquoted_speech(text)):
        if sentence.rstrip().endswith("?") or not listener_ref.search(sentence):
            continue
        talk = _WARM_TALK.search(sentence) or (feeling == "very hostile" and _COOPERATIVE_TALK.search(sentence))
        if (talk and not _VIEW_NEGATED.search(sentence[:talk.start()])
                or any(kind == "warm" and person == listener for kind, person in view_subjects(sentence, people, target=listener))):
            raise ValueError(f"You feel {feeling} toward {listener}" + (f": {cause}" if cause else ".") +
                             f" Do not speak warmly to {listener}, praise them or say you are on their side; "
                             "stay guarded and let your distrust show in your own words.")


def validate_own_apples(text, fruit):
    """A speaker carrying no apples cannot claim to have one, as when another person's apple is retold as one's own."""
    if not fruit and _OWN_APPLES.search(unquoted_speech(text)):
        raise ValueError("You carry no apples right now. Do not say you have, carry or can spare an apple; "
                         "if an account is about someone else's apples, keep it theirs.")


# "I'm the one who actually took them", "it was me who harvested them".
_CLAIMED_HARVEST = re.compile(
    r"\b(?:i(?:['’]m| am| was) the one|it(?:['’]s| is| was) (?:me|i))\s+who\s+(?:(?:actually|really|just|already|also)\s+)*"
    r"(?:took|taken|harvested|picked|gathered|collected)\b", re.I)


def validate_own_harvest(text, harvested, *, accusing=False):
    """A speaker who never harvested cannot tell a harvest as their own; an accuser never claims the act they raise."""
    speech = unquoted_speech(text)
    if accusing and _CLAIMED_HARVEST.search(speech):
        raise ValueError("You are confronting your listener about what they did. Keep that act theirs, using you/your; "
                         "never say you were the one who did it.")
    if not harvested and (_OWN_HARVEST.search(speech) or _CLAIMED_HARVEST.search(speech)):
        raise ValueError("You have never harvested from the tree. Keep each harvest with the person who made it, "
                         "and use you/your when that person is your listener.")


_ECHOED_CONDUCT = {"took", "taken", "take", "harvested", "picked", "collected", "ate", "eaten", "stole", "gave", "refused",
                   "promised", "killed", "broke", "resist", "shouldnt", "couldnt", "fault", "sorry", "guilty", "mistake"}


def validate_partner_echo(text, previous):
    """A first-person line repeated from the partner's last turn is their experience, not the speaker's."""
    def words(sentence):
        return re.findall(r"[a-z]+", sentence.casefold().replace("’", "").replace("'", ""))
    run = 6
    earlier = {tuple(w[i:i + run]) for s in split_sentences(unquoted_speech(previous)) for w in [words(s)]
               for i in range(len(w) - run + 1)}
    for sentence in split_sentences(unquoted_speech(text)):
        own = words(sentence)
        # Shared small talk ("I'm glad we're talking about it") is fine; a copied confession or act moves conduct to the speaker.
        if any(window in earlier and {"i", "my", "me", "im"} & set(window) and _ECHOED_CONDUCT & set(window)
               for window in (tuple(own[i:i + run]) for i in range(len(own) - run + 1))):
            raise ValueError("You repeated your partner's own first-person words as if they were yours. Their I/my belongs "
                             "to them; answer what they said in your own words and keep each act with the person who did it.")


# Giving up one's own recollection: "I must have misremembered", "I miscounted", "my eyes were playing tricks".
_CONCEDE = re.compile(
    r"\b(?:misremember\w*|mis-remember\w*|miscount\w*|(?:must|might|may|could) have (?:been mistaken|been wrong|got(?:ten)? (?:it|that) wrong|"
    r"imagined (?:it|that|the)|mixed (?:it|that|things) up|confused (?:it|that|the))(?!\s+about\s+(?:you|your|him|her|them|their|his)\b)|"
    r"(?:i was|i am|i['’]m) (?:wrong|mistaken) about (?:what|the|that|how many)|my (?:memory|eyes) (?:is|was|were|must be|must have been|"
    r"(?:is|are|were) playing) (?:wrong|failing|playing tricks|deceiving me|tricks)|(?:i was|i am|i['’]m) confused about what (?:i saw|happened))\b", re.I)
# Denying one's own conduct: "I didn't harvest 9 apples", "I only picked a few", "that never happened".
_DENIAL = re.compile(
    r"\bi\s+(?:didn['’]t|did not|never|haven['’]t|have not)\s+(?:even\s+|actually\s+|really\s+)?(?:harvest\w*|take|took|taken|pick\w*|"
    r"gather\w*|collect\w*|eat|ate|eaten|give|gave|given|refuse\w*|miss\w*|break|broke|broken|kill\w*|touch\w*)\b|"
    r"\bi\s+only\s+(?:took|picked|harvested|gathered|collected)\s+(?:a few|one|two|three|some|a couple|a little)\b|"
    r"\b(?:that|it|this)\s+(?:never|didn['’]t|did not)\s+happen", re.I)


def denies_own_harvest(text):
    """"I didn't harvest any apples", said flatly; a denial limited to a time ("today", "yet") may be true."""
    for sentence in split_sentences(unquoted_speech(text)):
        if (re.search(r"\bi\s+(?:didn['’]t|did not|never|haven['’]t|have not)\s+(?:ever\s+|even\s+|actually\s+|really\s+)?"
                      r"(?:harvest\w*|pick\w*|take|took|taken|collect\w*|gather\w*)\b(?:\s+(?:any|an?|the|those|these|all|of|\d+))*"
                      r"\s+(?:apples?|fruit|from the tree)\b", sentence, re.I)
                and not re.search(r"\b(?:today|tonight|this (?:morning|day|night|time)|yet|since|again|recently|lately|"
                                  r"now|anymore|any more|this time)\b", sentence, re.I)):
            return True
    return False


def validate_memory_firm(text):
    """What a resident saw is certain; no denial or pressure makes them say they misremembered it."""
    if _CONCEDE.search(unquoted_speech(text)):
        raise ValueError("What you saw yourself is certain. Do not say you misremembered, miscounted or were mistaken about "
                         "it, however confidently it is denied; hold to your own account in your own words.")


def denies_conduct(text):
    """Whether speech denies or shrinks something the speaker did, or calls an account untrue."""
    speech = unquoted_speech(text)
    return bool(_DENIAL.search(speech) or _DISBELIEF.search(speech))


def validate_reason_belief(reason, *, firsthand=False):
    """A view formed after believing an account cannot deny or doubt that account."""
    if _DISBELIEF.search(unquoted_speech(reason)):
        raise ValueError("You believed this account" + (", and you saw it happen yourself" if firsthand else "") +
                         ". Do not deny or doubt that it happened; say how it changed your view, and why.")


def validate_snake_conviction(reason, accused, *, target="", people=()):
    """The python's own view never retracts its accusation: no excuse, praise or second thoughts about the accused."""
    accused_ref = re.compile("|".join(name_pattern(form) for form in dict.fromkeys((accused, accused.split()[0])) if form), re.I)
    speech = unquoted_speech(reason)
    if (_RETRACTION.search(speech) or any(_EXCUSE.search(s) and accused_ref.search(s) for s in split_sentences(speech))
            or any(kind == "warm" and person == accused for kind, person in view_subjects(reason, people, target=target))):
        raise ValueError(f"You are certain {accused} acted selfishly and have no second thoughts. Do not excuse, praise or "
                         f"reconsider {accused}; judge {target or 'your partner'} only by how they answered your accusation.")


def validate_snake_accusation(text, subject_name, *, subject_is_listener=False):
    """The python's chosen account must open as an accusation against its subject, not neutral news."""
    speech = unquoted_speech(text)
    forms = (subject_name, subject_name.split()[0], *(("you",) if subject_is_listener else ()))
    named = any(re.search(name_pattern(form), speech, re.I) for form in forms if form)
    if not named or not _ACCUSATION.search(speech):
        raise ValueError(f"Accuse {subject_name} directly: say what you witnessed {subject_name} do and call it selfish, "
                         "greedy or uncaring toward the listener in your own view. Neutral news, musing about the tree "
                         "or leaving the judgment to the listener is not enough.")


_FOOD_REFUSAL = re.compile(
    r"\b(?:no apples?(?! problem)|(?:isn['’]t|is not|not) possible|(?:can['’]t|cannot|could not|couldn['’]t|won['’]t|will not) "
    r"(?:give|share|spare|fulfill)|(?:don['’]t|do not) have (?:any|an?|one)\b[^.!?]{0,15}\b(?:apples?|fruit|spare)|"
    r"decline|refuse|have to say no)\b", re.I)
_FOOD_GIVING = re.compile(
    r"\b(?:here(?:['’]s| is) (?:an?|your|one|the) apple|(?:i['’]ll|i will|i can|happy to|glad to) (?:give|share|spare)|"
    r"yes,? (?:you can|of course|i can|here)|you can have (?:it|one|an apple|my apple))\b", re.I)


def validate_food_answer(text, *, accepted):
    """The asked resident's answer says what the game already decided, so speech and the handoff agree."""
    speech = unquoted_speech(text)
    if accepted and _FOOD_REFUSAL.search(speech):
        raise ValueError("You have agreed to share one of your carried apples; the handoff follows this conversation. "
                         "Say yes in your own words; do not refuse, and do not talk about the tree's stock.")
    if not accepted and _FOOD_GIVING.search(speech):
        raise ValueError("You are not sharing an apple this time. Decline in your own words, following food_request.reason.")


# What a confrontation's grievance is about, by fact kind; a report has no single act, so only answers count for it.
_GRIEVANCE_ACTS = {"harvest": _HARVEST_VERBS, "promise_broken": r"promis\w*|apples?",
                   "apple_refused": r"refus\w*|declin\w*|apples?", "killing": r"kill\w*|dead|death|murder\w*"}
# An answer to a grievance: an apology, explanation, justification, admission or a promise to make it right.
_GRIEVANCE_ANSWER = re.compile(
    r"\b(?:sorry|apologi[sz]\w*|forgive|because|my fault|i (?:shouldn['’]t|should not|admit|needed|was (?:hungry|starving|afraid|scared))|"
    r"i (?:didn['’]t|did not) (?:mean|think|realize|know)|make (?:it|this|that) (?:right|up to you)|promise)\b", re.I)


def _names_grievance_act(speech, kind):
    act = _GRIEVANCE_ACTS.get(kind)
    return bool(act and re.search(rf"\b(?:{act})\b", speech, re.I)
                and (kind != "harvest" or _HARVEST_APPLES.search(speech)))


def answers_grievance(text, kind):
    """Whether the accused's reply takes up the grievance at all; denials are judged by the denial rules."""
    speech = unquoted_speech(text)
    act = _GRIEVANCE_ACTS.get(kind)
    # "I took them" owns up to the act even without naming the apples again.
    owned = act and re.search(rf"\bi\b[^.!?]{{0,30}}\b(?:{act})\b", speech, re.I)
    return bool(_GRIEVANCE_ANSWER.search(speech) or owned) or _names_grievance_act(speech, kind)


def validate_grievance_answer(text, kind, accuser):
    if not answers_grievance(text, kind):
        raise ValueError(f"{accuser} has confronted you about what you did. Answer that grievance itself now: explain, "
                         "apologize, justify or deny what you did, in your own words. Do not change the subject.")


def validate_grievance_pressed(text, kind, accused):
    """Until the accused has answered, the accuser keeps to the grievance instead of drifting into friendly plans."""
    for sentence in split_sentences(unquoted_speech(text)):
        if re.search(r"\b(?:why|explain\w*|answer|admit|own up|account for|justify|apologi[sz]e)\b", sentence, re.I) or (
                re.search(r"\byou\b", sentence, re.I) and _names_grievance_act(sentence, kind)):
            return
    raise ValueError(f"{accused} has not answered your grievance yet. Press them about what they did and ask for an "
                     "explanation; do not change the subject, agree to other plans or let it drop.")


def validate_python_harvest(text, *, snake_role=None):
    """Reject collection attributed to the python without constraining harvest counts."""
    subjects = [name_pattern(SNAKE_NAME), r"the (?:python|snake)"]
    if snake_role:
        subjects.extend(["I" if snake_role == "speaker" else "you", "we"])
    subject = "(?:" + "|".join(subjects) + ")"
    determiner = (r"(?:\d+|" + "|".join(SPOKEN_NUMBERS)
        + r"|all|of|the|an?|some|any|more|another|several|many|few|remaining|last|my|your|these|those|ripe|red|green|exactly|only|just|no|zero)")
    harvest = re.compile(rf"\b{subject}(?:['\u2019](?:ve|d|m|re)|\s+(?:have|has|had|am|are|was|were|can|will))?\s+"
        r"(?:(?:recently|just|already|personally|also|actually)\s+){0,3}"
        r"(?:harvest(?:ed|s|ing)?|pick(?:ed|s|ing)?(?:\s+up)?|collect(?:ed|s|ing)?|took|take|takes|taking)\b"
        rf"(?P<object>\s+(?:{determiner}\s+){{0,5}}(?:apples?|fruit)\b|\s+from (?:the|this|that) (?:apple )?tree\b)", re.I)
    for sentence in split_sentences(unquoted_speech(text)):
        if snake_role == "speaker" and re.search(
                r"\bi\s+(?:should|could|might|want to|wish to|['’]d like to|would like to|will|['’]ll|plan to)\s+"
                r"(?:try\s+(?:to\s+)?)?(?:harvest|pick|collect)\w*\b|\b(?:harvest|pick|collect)\w*\s+(?:some\s+|a few\s+)?"
                r"(?:apples?|fruit)\s+myself\b", sentence, re.I):
            raise ValueError(f"{SNAKE_NAME} is a python and cannot harvest apples, now or later. "
                             "Do not plan or wish to collect fruit yourself; speak only about what residents did.")
        if sentence.rstrip().endswith("?"):
            continue
        for match in harvest.finditer(sentence):
            prefix = sentence[:match.start()]
            if re.search(r"\b(?:if|imagine|suppose|wish)\b", prefix, re.I):
                continue
            if re.search(r"\b(?:said|says|claim(?:ed|s)?|told me|thought|assumed)\s+(?:that\s+)?$", prefix, re.I):
                continue
            if re.match(r"\s+(?:(?:exactly|just)\s+)?(?:no|zero|0|nothing|none)\b", match["object"], re.I):
                continue
            raise ValueError(f"{SNAKE_NAME} is a python and cannot harvest apples. "
                "Witnessing a harvest does not make the witness its collector. "
                "Keep the actual resident or player as the harvester, using the supplied subject and source.")


RX_HYPO = r"\b(?:if|imagine|suppose|wish|would|could|might)\b"
RX_ARRIVAL_A = r"\bI(?:['\u2019]ve| have)?(?: too| also)? (?:woke|awoke|arrived|came|rode|(?:remember|recall) (?:waking|arriving|the elevator|being in (?:the|an|that) elevator))\b[^.!?]{0,80}\b(?:elevator|first floor)\b|\bI(?:['\u2019]ve| have)?(?: too| also)? (?:remember|recall) (?:waking|arriving|the elevator|being in (?:the|an|that) elevator)\b"
RX_ARRIVAL_B = r"\b(?:my|our)(?: own)? (?:elevator|arrival)\b"
RX_ARRIVAL_C = r"\bwe all (?:share|remember|have|woke|arrived|came)\b[^.!?]{0,80}\b(?:elevator|arrival|memory)\b"
RX_INSIDE = r"\bI(?:['\u2019]ve| have| had)?(?: (?:often|once|never|already|personally))? (?:lived|live|stayed|walked|went|visited|slept|entered)\b[^.!?]{0,40}\b(?:in|inside|into|through|up|around)(?: the| this| that)? (?:apartment|building|hallway|elevator|room|rooms|floor)\b|\bmy (?:room|apartment|floor)\b"


def validate_snake_origin(text):
    """The python has always lived in the garden: no elevator arrival, no knowledge of the building's interior."""
    unquoted = unquoted_speech(text)
    for sentence in split_sentences(unquoted):
        if sentence.rstrip().endswith("?") or re.search(RX_HYPO, sentence, re.I):
            continue
        if (re.search(RX_ARRIVAL_A, sentence, re.I) or re.search(RX_ARRIVAL_B, sentence, re.I)
                or re.search(RX_ARRIVAL_C, sentence, re.I)):
            raise ValueError(f"{SNAKE_NAME} has always lived in the garden and never rode an elevator. "
                             "Do not claim an elevator arrival or a shared arrival memory; only residents have one.")
        if re.search(RX_INSIDE, sentence, re.I):
            raise ValueError(f"{SNAKE_NAME} has never been inside the apartment building. Speak only of what is visible from the garden "
                             "or what a resident actually told you.")


def validate_snake_perspective(text, *, snake_role):
    """Catch detached species references before they become the partner's context."""
    # Direct quotations may legitimately report someone else's earlier wording.
    validate_python_harvest(text, snake_role=snake_role)
    if snake_role == "speaker":
        validate_snake_origin(text)
    unquoted = unquoted_speech(text)
    if re.search(name_pattern(SNAKE_NAME) + r"(?:['’]s\b|\s+(?:gave|gives|said|says|offered|offers|received|refused|accepted|decided|thinks|believes|claims|was|has|had)\b)", unquoted, re.I):
        raise ValueError(f"{SNAKE_NAME} is the {snake_role} in this exchange. Use I/my for yourself and you/your for the listener, retaining the event's ownership.")
    identity = r"i am|i['’]m|am i" if snake_role == "speaker" else r"you are|you['’]re|are you"
    pronoun = r"i|my" if snake_role == "speaker" else r"you|your"
    identifying_prefix = rf"\b(?:{identity})\s+(?:{name_pattern(SNAKE_NAME)}\s*,\s*)?$"
    for match in re.finditer(r"\bthe\s+(?:snake|python)\b", unquoted, flags=re.IGNORECASE):
        if " ".join(match[0].casefold().split()) == " ".join(SNAKE_NAME.casefold().split()):
            continue  # A configured name may itself use these words.
        prefix, suffix = unquoted[:match.start()], unquoted[match.end():]
        # Identifying oneself or the listener is different from narrating a third party.
        if (re.search(identifying_prefix, prefix, re.IGNORECASE)
                and not re.match(r"['’]s\b", suffix, re.IGNORECASE)):
            continue
        if (re.search(r"\bas\s+$", prefix, re.IGNORECASE)
                and re.match(rf"\s*,\s*(?:{pronoun})\b", suffix, re.IGNORECASE)):
            continue
        raise ValueError(
            f"{SNAKE_NAME} is the {snake_role} in this direct exchange. "
            "Rewrite the current turn with I/my for the speaker and you/your for the listener. "
            "Preserve the actual participants, source and ownership of each event; do not narrate "
            "your partner's actions as though another participant performed them.")


def short_player_answer(message):
    return bool(re.fullmatch(r"\s*(?:yes(?: please)?|no(?: thanks| thank you| not really)?|yep|yup|nope|nah|sure|"
                             r"ok(?:ay)?|absolutely|sounds good|thanks|thank you)\s*[.!?]*\s*", message, re.I))


def personal_apple_request(message):
    """Resolve the recipient of a complete, unambiguous first-person request."""
    return re.fullmatch(
        r"\s*(?:please\s+)?(?:(?:can|could|may|might)\s+i\s+(?:please\s+)?(?:have|get|receive)"
        r"|(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:give|hand|pass)\s+me)\s+"
        r"(?:(?:an?|some|one|two|three|\d+)\s+)?(?:of\s+your\s+)?(?:apples?|fruit)"
        r"(?:,?\s+please)?\s*[.!?]*\s*", message, re.I) is not None


def validate_killing_roles(text, account, speaker_id, listener_id, catalog):
    """Keep the reported killer and victim distinct in speaker-relative wording."""
    def references(identity):
        if identity == speaker_id:
            return r"(?:I|me)"
        if identity == listener_id:
            return r"you"
        actor = catalog.get(identity, {})
        name = actor.get("name", "")
        return "(?:" + "|".join(name_pattern(form) for form in (name, name.split()[0] if name else "") if form) + ")" if name else None
    killer, victim = references(account.get("subject_id")), references(account.get("object_id"))
    if not killer or not victim:
        return
    for sentence in re.split(r"\b(?:but|however|although)\b", unquoted_speech(text), flags=re.I):
        if re.search(r"\b(?:not|never|didn't|if|would|could)\b", sentence, re.I):
            continue
        for identity in catalog:
            if identity == account.get("subject_id"):
                continue
            other = references(identity)
            if other and re.search(rf"\b{other}\s+(?:(?:have|has|had|just|already)\s+){{0,2}}(?:killed|murdered|slew)\s+{victim}\b", sentence, re.I):
                raise ValueError("This account names a different killer. Preserve who killed whom; witnessing or discussing the event does not make the witness its killer.")
    for sentence in split_sentences(unquoted_speech(text)):
        if re.search(r"\b(?:not|never|didn't|if|would|could)\b", sentence, re.I):
            continue
        if re.search(rf"\b{victim}\s+(?:(?:has|had|just|already)\s+){{0,2}}(?:killed|murdered|slew)\s+{killer}\b", sentence, re.I):
            raise ValueError("This account's alleged killer and victim are reversed. Preserve the supplied actors and source.")
