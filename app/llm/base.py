"""Common types for every model adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(Exception):
    """Base class. Anything raised by an adapter derives from this."""


class ProviderError(LLMError):
    """HTTP / transport / auth / rate-limit failure."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RefusedError(LLMError):
    """The provider declined the request (safety stop)."""


class TruncatedError(LLMError):
    """Output hit the token limit before the JSON finished."""


class InvalidOutputError(LLMError):
    """The model answered, but not with usable JSON / the expected structure."""


class CostCapError(LLMError):
    """Today's spend reached the daily cap in Settings."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0  # billed output, thinking included
    thinking_tokens: int = 0  # of which thinking, when the provider says (0 when it does not)


@dataclass
class LLMResult:
    data: dict
    usage: Usage = field(default_factory=Usage)
    served_model: str = ""  # the model name the API says answered, not just the one we asked for
    raw_text: str = ""


class LLMClient(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        max_tokens: int,
        context: dict | None = None,  # only the offline demo provider reads this
    ) -> LLMResult:
        """Return one JSON object that follows `schema`."""
