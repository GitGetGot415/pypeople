"""Validation for persisted value records, derived from their dataclass fields."""

from dataclasses import fields
from typing import get_type_hints

from pydantic import BaseModel, ConfigDict, create_model


class SaveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def saved_record(record):
    """Keep every save field required, including fields with runtime defaults."""
    annotations = get_type_hints(record, include_extras=True)
    return create_model(record.__name__ + "State", __base__=SaveModel,
                        **{field.name: (annotations[field.name], ...) for field in fields(record)})
