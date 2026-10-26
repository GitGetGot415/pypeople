"""Stable recovery reasons and the validation and generation failures built on them."""

from enum import Enum


class RecoveryKind(str, Enum):
    GENERAL = "general"
    MISSING_DISCLOSURE = "missing_disclosure"
    KNOWN_ENCOUNTER = "known_encounter"
    MISSING_PROMISE = "missing_promise"
    MISSING_ACCEPTANCE = "missing_acceptance"
    UNMATCHED_ACCOUNT = "unmatched_account"
    MISSING_GIFT_OFFER = "missing_gift_offer"


class DialogueValidationError(ValueError):
    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


class DialogueTurnError(ValueError):
    def __init__(self, message, turn_indices):
        super().__init__(message)
        self.turn_indices = tuple(turn_indices)


class DialogueRepetitionError(DialogueTurnError):
    pass


class DialogueSpeakerError(DialogueTurnError):
    pass


class RequestError(Exception):
    """An application-authored failure reason and optional generated draft for inspection."""

    def __init__(self, message, *, failed_turns=()):
        super().__init__(message)
        self.failed_turns = list(failed_turns)


class InvalidResponseError(ValueError):
    """An application-authored validation reason that can be sent back for correction."""

    def __init__(self, message, *, kind=RecoveryKind.GENERAL, response=None, repeated_turns=(), repair_turns=(), failed_turns=()):
        super().__init__(message)
        self.kind = kind
        self.response = response
        self.repeated_turns = tuple(repeated_turns)
        self.repair_turns = tuple(repair_turns or repeated_turns)
        self.failed_turns = list(failed_turns)


class StreamRepetitionError(InvalidResponseError):
    """A still-generating response loop, before usable JSON or speech exists."""
