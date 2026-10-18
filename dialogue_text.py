"""Sentence boundaries shared by generated speech checks and dialogue display."""

import json
import re


_SENTENCE_END = re.compile(r'''[.!?…。！？]+["'’”»\)\]\}]*(?=\s|$)''')
_CLOSING_MARKS = "\"'’”»)]}"
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "jr.", "sr.", "st.", "e.g.", "i.e."}


def contains_serialized_dialogue(text):
    """Recognize response envelopes masquerading as spoken text, even if truncated."""
    unescaped = text.replace('\\"', '"')
    if re.search(r'''[{,]\s*["']?(?:text|reply|parts|turns|speaker|emotion|name|intensity|assertion|subject_relation|object_relation|knowledge_owner_name|original_speaker_relation|immediate_source_relation)["']?\s*:''',
                 unescaped, re.I) or re.search(r"```(?:json)?\s*[\[{]", unescaped, re.I):
        return True
    try:
        return isinstance(json.loads(text), (dict, list))
    except (ValueError, TypeError):
        return False


def speech_is_complete(text):
    """Detect an unfinished ending without guessing or adding missing speech."""
    text = text.strip().rstrip(_CLOSING_MARKS)
    return bool(text) and text[-1] in ".!?…。！？"


def split_sentences(text):
    """Keep sentence punctuation and quoted endings attached to their speech."""
    text = " ".join(text.split())
    sentences, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        token = text[:match.end()].rstrip(_CLOSING_MARKS).rsplit(" ", 1)[-1]
        if match.end() < len(text) and (
                token.casefold() in _ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", token)):
            continue
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences
