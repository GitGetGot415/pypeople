"""Session records of dialogue and its outcomes."""

from dataclasses import dataclass, field
import re
from uuid import uuid4


# These destinations do not exist in this simulation. Reject invented outings
# before they become dialogue or context for a later conversation.
UNSUPPORTED_LOCATIONS = re.compile(
    r"\b(?:markets?|shops?|buildings?|towns?|cities|city|villages?|roads?|streets?|caf[eé]s?|restaurants?)\b",
    re.IGNORECASE,
)


def unsupported_location(text):
    return UNSUPPORTED_LOCATIONS.search(text) is not None


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
    id: str = field(default_factory=lambda: uuid4().hex)


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


def relevant_memories(npc, partner_id=None):
    """Keep a small, deduplicated selection with obligations ahead of routine chat."""
    goal_target = npc.goal.target_id if npc.goal and npc.goal.status == "active" else None
    promises = [p for p in npc.world.promises if npc.id in (p.promisor_id, p.recipient_id)
                and p.status in ("pending", "broken")] if npc.world else []
    partners = {p.recipient_id if p.promisor_id == npc.id else p.promisor_id for p in promises}
    ranked = sorted(enumerate(npc.memories), key=lambda entry: (
        entry[1].partner_id in partners and entry[1].kind in ("promise_created", "promise_broken"),
        entry[1].kind in ("heard_claim", "claim_opinion", "leverage"),
        entry[1].partner_id == partner_id, entry[1].partner_id == goal_target,
        abs(entry[1].weight), entry[0]), reverse=True)
    seen, selected = set(), []
    for _, memory in ranked:
        if memory.id not in seen:
            selected.append(memory)
            seen.add(memory.id)
        if len(selected) == 5:
            break
    return selected


def describe_opinion_change(person, target, previous, opinion, reason):
    before = "Not recorded" if previous is None else f"{previous:+d}"
    change = "" if previous is None else f" ({opinion - previous:+d})"
    return f"{person}'s opinion of {target}: {before} -> {opinion:+d}{change}\nReason: {reason}"
