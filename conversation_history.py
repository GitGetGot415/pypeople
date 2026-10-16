"""Session records of dialogue and its outcomes."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ConversationRecord:
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[tuple[str, str]] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    error: str = ""
    origin: str = "player"
    purpose: str = "chat"


@dataclass
class Memory:
    id: str
    day: int
    phase: str
    partner_id: str
    kind: str
    summary: str
    weight: float


@dataclass
class WorldEvent:
    day: int
    phase: str
    kind: str
    participants: list[str]
    text: str
    history: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex)


def memory_bias(npc, partner_id):
    """Recent personal experiences contribute at most twelve decision points."""
    memories = [memory for memory in npc.memories if memory.partner_id == partner_id]
    return max(-12.0, min(12.0, sum(
        memory.weight * 0.8 ** age for age, memory in enumerate(reversed(memories))
    )))


def remember(npc, partner, event, kind, summary, weight):
    if any(memory.id == event.id for memory in npc.memories):
        return
    npc.memories.append(Memory(event.id, event.day, event.phase, partner.id, kind, summary[:400], weight))
    del npc.memories[:-20]


def describe_opinion_change(person, target, previous, opinion, reason):
    before = "Not recorded" if previous is None else f"{previous:+d}"
    change = "" if previous is None else f" ({opinion - previous:+d})"
    return f"{person}'s opinion of {target}: {before} -> {opinion:+d}{change}\nReason: {reason}"
