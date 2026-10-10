"""Session records of dialogue and its outcomes."""

from dataclasses import dataclass, field


@dataclass
class ConversationRecord:
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[tuple[str, str]] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    error: str = ""


def describe_opinion_change(person, target, previous, opinion, reason):
    before = "Not recorded" if previous is None else f"{previous:+d}"
    change = "" if previous is None else f" ({opinion - previous:+d})"
    return f"{person}'s opinion of {target}: {before} -> {opinion:+d}{change}\nReason: {reason}"
