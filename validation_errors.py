"""Stable recovery reasons, independent of the wording shown to the player."""

from enum import Enum


class RecoveryKind(str, Enum):
    GENERAL = "general"
    MISSING_DISCLOSURE = "missing_disclosure"
    KNOWN_ENCOUNTER = "known_encounter"
    UNKNOWN_NAME = "unknown_name"
    NAME_QUESTION = "name_question"
    MISSING_PROMISE = "missing_promise"
    MISSING_ACCEPTANCE = "missing_acceptance"
    UNMATCHED_ACCOUNT = "unmatched_account"


class DialogueValidationError(ValueError):
    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind
