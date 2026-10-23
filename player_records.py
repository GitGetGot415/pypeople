"""Player conversation records, distinct from generated proposals and committed promises."""

from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import Field
from dialogue import GiftOffer


@dataclass
class PlayerOpinion:
    opinion: Annotated[int, Field(strict=True, ge=-100, le=100)] = 0
    reason: Annotated[str, Field(max_length=400)] = ""


@dataclass
class PlayerCallReason:
    key: Annotated[str, Field(min_length=1)]
    kind: Literal["gift", "news", "followup", "introduction"]
    reference: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]


@dataclass
class PlayerCall(PlayerCallReason):
    event_id: Annotated[str, Field(min_length=1)]


@dataclass
class PlayerReplyCall:
    kind: Literal["reply"]
    reference: Annotated[str, Field(min_length=1)]
    closing: bool
    gift_offer: GiftOffer | None


@dataclass
class PlayerReport:
    id: str
    speaker_id: str
    conversation_id: str
    turn: int
    day: int
    phase: Literal["day", "night"]
    text: str
    kind: Literal["report"]
    listener_ids: list[str]


@dataclass
class PlayerPromiseOffer:
    id: Annotated[str, Field(min_length=1)]
    giver_id: Annotated[str, Field(min_length=1)]
    recipient_id: Annotated[str, Field(min_length=1)]
    deadline: Annotated[int, Field(strict=True, ge=0)]
    created_phase: Annotated[int, Field(strict=True, ge=0)]
    offer_turn: Annotated[int, Field(strict=True, ge=0)]
    accept_turn: Annotated[int | None, Field(strict=True, ge=0)]
    status: Literal["offered", "accepted", "declined", "expired", "closed"]


@dataclass
class PlayerAcknowledgment:
    npc_id: Annotated[str, Field(min_length=1)]
    record_id: Annotated[str, Field(min_length=1)]
    transfer_id: Annotated[str, Field(min_length=1)]


def validate_player_reports(reports, records, player_id, now):
    from social import phase_index
    claim_ids = set()
    for claim in reports:
        record = records.get(claim.conversation_id)
        turn = claim.turn
        if (claim.id in claim_ids or claim.speaker_id != player_id or claim.kind != "report"
                or not record or type(turn) is not int or turn not in record.accepted_player_turns
                or (record.turns[turn].speaker_id, record.turns[turn].text) != (player_id, claim.text)
                or set(claim.listener_ids) != set(record.participant_ids) - {player_id}
                or phase_index(claim.day, claim.phase) > now):
            raise ValueError("Player claims require their actual accepted original speech and listener.")
        claim_ids.add(claim.id)
