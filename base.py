"""Provider adapter.

Sponsor challenge tracks are announced mid-build and projects are judged
inside the sponsor's category. Everything that talks to a model goes through
this interface so that swapping provider is one new file, not a refactor.

A provider must support tool calling. That is the only hard requirement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None
    stop_reason: Optional[str] = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Provider(Protocol):
    """Minimal surface the agent depends on."""

    name: str

    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...

    def tool_result_message(self, call: ToolCall, result: Any) -> Dict[str, Any]:
        """Format a tool result in whatever shape this provider expects."""
        ...

    def assistant_message(self, response: ModelResponse) -> Dict[str, Any]:
        """Format the assistant turn for appending to history."""
        ...


def get_provider(name: Optional[str] = None) -> Provider:
    """Resolve a provider by name, defaulting to the IC_PROVIDER env var."""
    name = (name or os.environ.get("IC_PROVIDER") or "anthropic").lower()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name in ("openai", "groq", "cerebras", "openrouter"):
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "echo":
        from .echo_provider import EchoProvider
        return EchoProvider()
    raise ValueError(f"unknown provider '{name}'")
