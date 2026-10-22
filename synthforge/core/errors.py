"""Typed, user-facing errors. Every error carries a stable code and an actionable hint."""

from __future__ import annotations


class SynthForgeError(Exception):
    """Base class for all expected SynthForge failures."""

    code = "SYNTHFORGE_ERROR"

    def __init__(self, message: str, *, hint: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


class IngestError(SynthForgeError):
    code = "INGEST_ERROR"


class UnsupportedFormatError(IngestError):
    code = "UNSUPPORTED_FORMAT"


class SchemaError(SynthForgeError):
    code = "SCHEMA_ERROR"


class ConstraintError(SynthForgeError):
    code = "CONSTRAINT_ERROR"


class ModelError(SynthForgeError):
    code = "MODEL_ERROR"


class PrivacyError(SynthForgeError):
    code = "PRIVACY_ERROR"


class BudgetExceededError(PrivacyError):
    code = "BUDGET_EXCEEDED"


class JobError(SynthForgeError):
    code = "JOB_ERROR"


class CancelledError(JobError):
    code = "JOB_CANCELLED"


class RegistryError(SynthForgeError):
    code = "REGISTRY_ERROR"


class AuthError(SynthForgeError):
    code = "AUTH_ERROR"


class PermissionDeniedError(SynthForgeError):
    code = "PERMISSION_DENIED"


class NotFoundError(SynthForgeError):
    code = "NOT_FOUND"


class ReproductionError(SynthForgeError):
    code = "REPRODUCTION_ERROR"
