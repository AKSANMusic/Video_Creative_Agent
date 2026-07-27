"""Typed exceptions for the AI Creative Engine pipeline.

All errors raised by internal modules derive from :class:`CreativeEngineError`
so callers can catch the whole family with a single ``except`` if needed.
"""


class CreativeEngineError(Exception):
    """Base class for all AI Creative Engine errors."""


class ConfigError(CreativeEngineError):
    """Raised when required configuration (e.g. API token) is missing."""


class APIError(CreativeEngineError):
    """Raised when an upstream inference API call fails permanently."""


class ParseError(CreativeEngineError):
    """Raised when an API response cannot be parsed into the expected shape."""


class CacheError(CreativeEngineError):
    """Raised when the local SQLite cache cannot be read or written."""


class EmbeddingError(CreativeEngineError):
    """Raised when local embedding computation fails."""
