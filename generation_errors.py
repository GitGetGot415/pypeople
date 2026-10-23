"""Generation failures shared by transport, orchestration and result consumers."""

from validation_errors import RecoveryKind


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


