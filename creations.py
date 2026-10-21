"""Fixed, data-only creative contracts shared by providers and saves."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from constants import (ART_PALETTE, ART_SIZE, CREATION_TITLE_LIMIT, MUSIC_MAX_NOTES,
                       MUSIC_MAX_NOTE_STEPS, MUSIC_MAX_TRACKS, MUSIC_PITCH_CLASSES,
                       MUSIC_PITCH_RANGE, MUSIC_STEPS, MUSIC_TEMPO_RANGE)


class CreationData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


PaletteIndex = Annotated[int, Field(strict=True, ge=0, le=len(ART_PALETTE) - 1)]
PixelRow = Annotated[list[PaletteIndex], Field(min_length=ART_SIZE, max_length=ART_SIZE)]


class ArtPayload(CreationData):
    pixels: list[PixelRow] = Field(min_length=ART_SIZE, max_length=ART_SIZE)


class Note(CreationData):
    pitch: int = Field(strict=True, ge=MUSIC_PITCH_RANGE[0], le=MUSIC_PITCH_RANGE[1])
    start: int = Field(strict=True, ge=0, le=MUSIC_STEPS - 1)
    duration: int = Field(strict=True, ge=1, le=MUSIC_MAX_NOTE_STEPS)

    @model_validator(mode="after")
    def musical_limits(self):
        if self.pitch % 12 not in MUSIC_PITCH_CLASSES or self.start + self.duration > MUSIC_STEPS:
            raise ValueError("Use natural notes ending within the 64-step score.")
        return self


class Track(CreationData):
    waveform: Literal["square", "triangle"]
    notes: list[Note] = Field(max_length=MUSIC_MAX_NOTES)

    @model_validator(mode="after")
    def monophonic(self):
        self.notes.sort(key=lambda note: note.start)
        if any(a.start + a.duration > b.start for a, b in zip(self.notes, self.notes[1:])):
            raise ValueError("Notes in a track cannot overlap.")
        return self


class MusicPayload(CreationData):
    tempo: int = Field(strict=True, ge=MUSIC_TEMPO_RANGE[0], le=MUSIC_TEMPO_RANGE[1])
    tracks: list[Track] = Field(min_length=1, max_length=MUSIC_MAX_TRACKS)

    @model_validator(mode="after")
    def note_count(self):
        if not 1 <= sum(len(track.notes) for track in self.tracks) <= MUSIC_MAX_NOTES:
            raise ValueError("A song needs 1-64 notes total.")
        return self


class TitledData(CreationData):
    title: str = Field(min_length=1, max_length=CREATION_TITLE_LIMIT, pattern=r"\S")

    @field_validator("title")
    @classmethod
    def printable_title(cls, value):
        if not all(character.isprintable() for character in value):
            raise ValueError("A creation title must be printable on one line.")
        return value


class ArtResponse(TitledData):
    payload: ArtPayload


class MusicResponse(TitledData):
    payload: MusicPayload


def response_model(kind):
    if kind == "art":
        return ArtResponse
    if kind == "music":
        return MusicResponse
    raise ValueError("Unknown creation kind.")


def validate_creation(kind, value):
    model = response_model(kind)
    return model.model_validate_json(value) if isinstance(value, str) else model.model_validate(value)


class CreativeJob(CreationData):
    id: str = Field(min_length=1)
    creator_id: str = Field(min_length=1)
    kind: Literal["art", "music"]
    started_day: int = Field(strict=True, ge=1)
    started_phase: Literal["day", "evening"]
    revision: int = Field(strict=True, ge=0)
    remaining: float = Field(ge=0)
    queue_remaining: float = Field(ge=0)
    accepted: dict | None = None

    @model_validator(mode="after")
    def accepted_data(self):
        if self.accepted is not None:
            self.accepted = validate_creation(self.kind, self.accepted).model_dump()
        return self


class Creation(TitledData):
    id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    creator_id: str = Field(min_length=1)
    kind: Literal["art", "music"]
    day: int = Field(strict=True, ge=1)
    phase: Literal["day", "evening", "night"]
    payload: dict

    @model_validator(mode="after")
    def valid_payload(self):
        result = validate_creation(self.kind, {"title": self.title, "payload": self.payload})
        self.payload = result.payload.model_dump()
        return self
