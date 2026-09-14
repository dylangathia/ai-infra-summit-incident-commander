"""OpenAI-compatible provider.

Not a copy of the Anthropic one. Four things genuinely differ: tool schema
shape, where the system prompt goes, how tool arguments arrive (a JSON
*string*, not a dict), and how tool results are returned.

Because the base URL is configurable, this one file also covers Groq,
Cerebras, Together and OpenRouter — all of which speak the OpenAI wire
format and several of which have free tiers:

    IC_PROVIDER=openai \\
    OPENAI_BASE_URL=https://api.groq.com/openai/v1 \\
    OPENAI_API_KEY=gsk_... \\
    IC_MODEL=llama-3.3-70b-versatile \\
    python3 diagnose.py openai
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from .base import ModelResponse, ToolCall

DEFAULT_MODEL = os.environ.get("IC_MODEL", "gpt-4o-mini")


def to_openai_tools(tools: List[dict]) -> List[dict]:
    """Anthropic-shaped schemas are canonical in this repo; translate here."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema")
                              or {"type": "object", "properties": {}},
            },
        })
    return out


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        from openai import OpenAI  # lazy: the sim runs without the SDK
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

    def complete(self, system, messages, tools=None, max_tokens=1500,
                 temperature=0.0) -> ModelResponse:
        payload: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        payload.extend(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        resp = self._with_retry(kwargs)
        msg = resp.choices[0].message

        calls: List[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                # arguments arrive as a JSON string, not a dict
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ModelResponse(
            text=(msg.content or "").strip(),
            tool_calls=calls,
            raw=resp,
            stop_reason=resp.choices[0].finish_reason,
        )

    def _with_retry(self, kwargs: Dict[str, Any], attempts: int = 6):
        """Free tiers are token-per-minute limited. A 429 is pacing, not failure.

        Groq reports how long to wait in the error text; honour it when present
        and fall back to exponential backoff with jitter otherwise.
        """
        delay = 1.0
        for i in range(attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                text = str(exc)
                # Server-side generation parse failures are stochastic, not a
                # bad request. Retrying with a nudged temperature breaks the
                # sampling path that produced the unparseable output.
                if "output_parse_failed" in text or "Parsing failed" in text:
                    if i == attempts - 1:
                        raise
                    kwargs = dict(kwargs)
                    kwargs["temperature"] = min(
                        1.0, float(kwargs.get("temperature") or 0.0) + 0.3)
                    time.sleep(1.0 + random.uniform(0, 0.5))
                    continue
                if "rate_limit" not in text and "429" not in text:
                    raise
                if i == attempts - 1:
                    raise
                # A daily cap is not pacing — waiting will not clear it, and
                # retrying just burns the little quota that remains.
                if "per day" in text or "TPD" in text:
                    raise RuntimeError(
                        "daily token quota exhausted for this model. Switch "
                        "IC_MODEL (limits are per-model) or wait for reset. "
                        f"Provider said: {text[:160]}")
                hinted = re.search(r"try again in (?:(\d+)m)?([\d.]+)(m?s)", text)
                if hinted:
                    mins = float(hinted.group(1) or 0)
                    val = float(hinted.group(2))
                    secs = val / 1000 if hinted.group(3) == "ms" else val
                    wait = mins * 60 + secs + 0.5
                    if wait > 90:
                        raise RuntimeError(
                            f"rate limit needs a {wait:.0f}s wait — not retrying. "
                            f"Provider said: {text[:160]}")
                else:
                    wait = delay + random.uniform(0, 0.5)
                    delay = min(delay * 2, 30.0)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def assistant_message(self, response: ModelResponse) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant",
                               "content": response.text or None}
        if response.tool_calls:
            msg["tool_calls"] = [{
                "id": c.id,
                "type": "function",
                "function": {"name": c.name,
                             "arguments": json.dumps(c.arguments)},
            } for c in response.tool_calls]
        return msg

    def tool_result_message(self, call: ToolCall, result: Any) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, default=str),
        }