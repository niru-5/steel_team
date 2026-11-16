"""Claude client for WRC's agent loop (hackathon DM port; NOT upstream WRC).

WRC ships an ``anthropic`` profile (agent/llm.py::AnthropicClient), but it sends a
``temperature`` and the current Claude models reject sampling parameters with a 400.
This client implements the same ``LLMClient.chat()`` contract against the current API:

  * model from $WRC_CLAUDE_MODEL (default claude-opus-5-5; e.g. claude-fable-5-2,
    claude-sonnet-5-5, claude-haiku-5-5)
  * thinking is left at the model default (adaptive, always on for Opus 5.5 / Fable);
    depth is set with output_config.effort from $WRC_CLAUDE_EFFORT (default medium)
  * no temperature / top_p / top_k, no forced tool_choice (both are 400s on these models)
  * one tool call per turn (WRC executes only the first one anyway)
  * server-side refusal fallback ("default" routing) on the models that support it;
    set WRC_CLAUDE_FALLBACKS=0 to turn it off
  * thinking blocks are never replayed: WRC rebuilds the history from its own
    text + tool-call records, which the API accepts (dropping all thinking blocks keeps
    the history valid under preserved thinking)

Credentials: the SDK reads ANTHROPIC_API_KEY (run_dm.py loads wrc/.env) or an
``ant auth login`` profile. The key is never logged.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

from wrc_demo.agent.llm import LLMClient, LLMResponse, ToolCall, _b64

_FALLBACK_MODELS = ("claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
                    "claude-sonnet-5-5")
_MIN_MAX_TOKENS = 8192     # thinking counts toward max_tokens; WRC asks for as little as 128


class ClaudeLLM(LLMClient):
    supports_vision = True

    def __init__(self, model: str | None = None, effort: str | None = None, system_suffix: str = ""):
        import anthropic

        self.system_suffix = system_suffix.strip()   # robot-specific notes added after WRC's own system prompt

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("WRC_CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = effort or os.environ.get("WRC_CLAUDE_EFFORT", "medium")
        self.use_fallbacks = (os.environ.get("WRC_CLAUDE_FALLBACKS", "1") != "0"
                              and self.model.startswith(_FALLBACK_MODELS))
        self.calls: list[dict] = []          # per-call latency + token usage (for the report)
        self.on_call = None                  # optional callback(record) after every API response

    # ── WRC message list -> Messages API ─────────────────────────────────
    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", "call_0"),
                    "content": m["content"],
                }]})
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = []
                if (m.get("content") or "").strip():
                    blocks.append({"type": "text", "text": m["content"]})
                for i, tc in enumerate(m["tool_calls"]):
                    blocks.append({"type": "tool_use", "id": tc.id or f"call_{i}",
                                   "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks})
                continue
            text = m["content"] if (m.get("content") or "").strip() else "(no text)"
            content: list[dict] = [{"type": "text", "text": text}]
            for jpeg in m.get("images") or []:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": _b64(jpeg)}})
            out.append({"role": m["role"], "content": content})
        return out

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        anthropic = self._anthropic
        if self.system_suffix:
            system = f"{system}\n\n{self.system_suffix}"
        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            messages=self._convert(messages),
            max_tokens=max(int(max_tokens), _MIN_MAX_TOKENS),
            output_config={"effort": self.effort},
        )
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                                "input_schema": t.get("parameters", {"type": "object"})} for t in tools]
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        t0 = time.monotonic()
        try:
            if self.use_fallbacks:
                resp = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise RuntimeError("Claude rejected the API key (ANTHROPIC_API_KEY in wrc/.env or the "
                               "environment). Check or rotate the key.") from e
        except anthropic.PermissionDeniedError as e:
            raise RuntimeError(f"Claude API: this key may not use {self.model}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise RuntimeError(f"Claude API: unknown model {self.model!r} (set WRC_CLAUDE_MODEL)") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError("Claude API rate limit hit; wait a moment and retry") from e
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"Claude API rejected the request: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError("Could not reach the Claude API (network?)") from e

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input), id=block.id))
        usage = resp.usage
        rec = {"s": round(time.monotonic() - t0, 2), "in": usage.input_tokens, "out": usage.output_tokens,
               "model": resp.model, "stop": resp.stop_reason, "tool": calls[0].name if calls else None,
               "args": calls[0].arguments if calls else {},
               # the API's own id for this request: look it up in the Anthropic Console to confirm the call
               "request_id": getattr(resp, "_request_id", None)}
        self.calls.append(rec)
        print(f"[claude] {rec['s']}s in={rec['in']} out={rec['out']} stop={rec['stop']} tool={rec['tool']} "
              f"request={rec['request_id']}", file=sys.stderr, flush=True)
        if self.on_call is not None:
            try:
                self.on_call(rec)
            except Exception:  # noqa: BLE001 -- a display hook must never break the robot loop
                pass
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise RuntimeError(f"Claude declined this request (category: {getattr(details, 'category', None)})")
        return LLMResponse(text="\n".join(text_parts), tool_calls=calls)
