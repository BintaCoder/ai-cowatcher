"""Token usage from LiteLLM completion responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def empty(cls) -> TokenUsage:
        return cls()

    def merge(self, other: TokenUsage | None) -> TokenUsage:
        if other is None:
            return self
        return TokenUsage(
            prompt_tokens=_sum_optional(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_sum_optional(self.completion_tokens, other.completion_tokens),
            total_tokens=_sum_optional(self.total_tokens, other.total_tokens),
        )


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def usage_from_litellm_response(response: object) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
