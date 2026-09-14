"""Anthropic implementation. Mirror this file to add a sponsor provider."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import ModelResponse, ToolCall

DEFAULT_MODEL = os.environ.get("IC_MODEL", "claude-sonnet-4-6")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        import anthropic  # imported lazily so the sim runs without the SDK
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def complete(self, system, messages, tools=None, max_tokens=1500,
                 temperature=0.0) -> ModelResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name,
                                      arguments=dict(block.input or {})))
        return ModelResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            raw=resp,
            stop_reason=resp.stop_reason,
        )

    def assistant_message(self, response: ModelResponse) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for c in response.tool_calls:
            content.append({"type": "tool_use", "id": c.id,
                            "name": c.name, "input": c.arguments})
        return {"role": "assistant", "content": content}

    def tool_result_message(self, call: ToolCall, result: Any) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result, default=str),
            }],
        }