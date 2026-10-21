"""Free-form delivery guidance shared by all model-written voice rewrites."""


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
