"""Transient panel presentation; recorded speech belongs to conversation history."""

from dataclasses import dataclass, field
from .text import paginate_turn


@dataclass
class DialogueView:
    open: bool = False
    actor: object = None
    conversation: object = None
    record: object = None
    status: str = ""
    pages: list = field(default_factory=list)
    index: int = 0
    scroll: int = 0
    scroll_limit: int = 0
    version: object = None
    latest: object = None
    picker: dict | None = None

    def reset(self):
        self.__dict__.update(vars(type(self)()))

    def show(self, actor, conversation=None):
        if self.actor is not actor or self.conversation is not conversation:
            self.reset()
        self.open, self.actor, self.conversation = True, actor, conversation

    @property
    def final_page(self):
        return self.index >= max(0, len(self.pages) - 1)

    def move(self, delta):
        self.index = max(0, min(max(0, len(self.pages) - 1), self.index + delta))
        self.scroll = 0

    def refresh(self, record, font, width, *, original, condensed=False, player_reply=False):
        self.record = record
        failed = bool(record and not record.turns and record.failed_turns)
        source = record.failed_turns if failed else record.turns if record else []
        version = (id(record), failed, tuple(id(turn) for turn in source),
                   tuple(tuple(turn.rewritten) for turn in source) if not failed else (), original, condensed, width,
                   self.status if player_reply else "")
        if version == self.version:
            return False
        was_final = self.final_page
        previous = self.pages[self.index][0] if self.pages else None
        offset = sum(page[0] is previous for page in self.pages[:self.index])
        turns = record.display_turns(failed=failed) if record else []
        latest = next((turn for turn in reversed(turns) if turn.speaker_id == self.actor.id), None) if player_reply else None
        new_reply = player_reply and latest is not None and latest is not self.latest
        pages = [(turn, emotion, text) for turn in turns
                 for emotion, text in paginate_turn(turn, font, width, original=original, condensed=condensed)]
        if player_reply and self.status:
            # An unrecorded notice has a page, never a fabricated speech turn.
            pages.append((None, None, self.status))
        anchor = latest if new_reply else previous
        if failed and previous is not None:
            old_turns = list(dict.fromkeys(id(page[0]) for page in self.pages))
            ordinal = old_turns.index(id(previous))
            anchor = turns[ordinal] if ordinal < len(turns) else None
        positions = [i for i, page in enumerate(pages) if page[0] is anchor]
        appended_input = (player_reply and not new_reply and was_final and self.version is not None
                          and len(source) > len(self.version[2]))
        new_status = player_reply and self.status and (self.version is None or self.status != self.version[-1])
        if new_status or appended_input:
            self.index = max(0, len(pages) - 1)
        elif positions:
            self.index = positions[min(0 if new_reply else offset, len(positions) - 1)]
        elif player_reply and was_final and self.version is not None:
            self.index = max(0, len(pages) - 1)
        else:
            self.index = min(self.index, max(0, len(pages) - 1))
        self.pages, self.version, self.latest = pages, version, latest
        self.scroll = 0
        return new_reply
