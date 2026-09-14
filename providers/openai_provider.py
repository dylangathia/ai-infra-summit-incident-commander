"""OpenAI-compatible implementation. Mirror the Anthropic adapter."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import ModelResponse, ToolCall

DEFAULT_MODEL = os.environ.get("IC_MODEL", "gpt-4o-mini")


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        from openai import OpenAI  # imported lazily so the sim runs without the SDK

        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, system, messages, tools=None, max_tokens=1500,
                 temperature=0.0) -> ModelResponse:
        chat_messages: List[Dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object"}),
                    },
                }
                for tool in tools
            ]

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        text = msg.content or ""
        calls: List[ToolCall] = []
        for call in msg.tool_calls or []:
            raw_args = call.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {"raw": raw_args}
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=parsed if isinstance(parsed, dict) else {"value": parsed},
                )
            )

        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        return ModelResponse(
            text=text,
            tool_calls=calls,
            raw=resp,
            stop_reason=finish_reason,
        )

    def assistant_message(self, response: ModelResponse) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": "assistant", "content": response.text or ""}
        if response.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": json.dumps(c.arguments, default=str),
                    },
                }
                for c in response.tool_calls
            ]
        return out

    def tool_result_message(self, call: ToolCall, result: Any) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, default=str),
        }
