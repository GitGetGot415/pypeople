"""Accepted transfer authorization and committed receipts, distinct from offers."""

from dataclasses import asdict, dataclass
from typing import Annotated, Literal
from pydantic import Field


@dataclass
class TransferIntent:
    giver_id: Annotated[str, Field(min_length=1)]
    recipient_id: Annotated[str, Field(min_length=1)]
    item: Annotated[str, Field(min_length=1)]
    quantity: Annotated[int, Field(strict=True, ge=1)]


@dataclass
class TransferReceipt(TransferIntent):
    id: Annotated[str, Field(min_length=1)]
    day: Annotated[int, Field(strict=True, ge=1)]
    phase: Literal["day"]
    sequence: Annotated[int, Field(strict=True, ge=0)]

    @classmethod
    def committed(cls, **values):
        receipt = cls(**values)
        # Fresh receipts historically used commit keyword order; restored receipts
        # used save-field order. Keep that boundary detail without another field.
        receipt.__dict__ = values.copy()
        return receipt


def transfer_payload(receipt):
    """Keep the original receipt field order at the provider boundary."""
    values = asdict(receipt)
    return {name: values[name] for name in vars(receipt) if name in values}
