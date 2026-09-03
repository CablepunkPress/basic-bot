"""Local inference provider.

Talks to llama-server's OpenAI-compatible /v1/chat/completions endpoint.
One server, one model — the model is chosen at construction time and is
the only entry get_models() returns.

Model metadata and sampling parameters come from the hardware profile,
injected at construction by the factory. This module never imports
from profile.py or config.py for model-specific values.

stdlib urllib only — no SDK for localhost HTTP.
"""

import json
import logging
import re
import urllib.error
import urllib.request

from basic_bot.providers.protocol import ChatResponse, ModelInfo, ToolCall

logger = logging.getLogger(__name__)

_SEQ_ANNOTATION = re.compile(r'<!--\s*seq:\d+\s*-->')

DEFAULT_MAX_TOKENS = 4096
REQUEST_TIMEOUT = 600


class LocalProvider:
    """InferenceProvider implementation for a local llama-server."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_info: ModelInfo | None = None,
        sampling: dict | None = None,
    ):
        self._model_id = model_id
        self._endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self._max_tokens = max_tokens
        self._model_info = model_info
        self._sampling = sampling or {}
        logger.info(
            "Local provider configured: %s → %s", model_id, self._endpoint,
        )

    # --- Protocol: model catalog ---

    def get_models(self) -> dict[str, ModelInfo]:
        if self._model_info:
            return {self._model_id: self._model_info}
        return {}

    def get_default_model(self) -> str:
        return self._model_id

    def get_fallback_model(self) -> str:
        return self._model_id

    # --- Protocol: chat ---

    def chat(
        self,
        *,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        model_id: str | None = None,
        effort: str | None = None,
        thinking: bool = False,
        sampling: dict | None = None,
    ) -> ChatResponse:
        payload: dict = {
            "model": self._model_id,
            "messages": self._build_messages(system, messages),
            "max_tokens": self._max_tokens,
        }

        if tools:
            payload["tools"] = [self._translate_tool(t) for t in tools]

        # Thinking toggle — Qwen-specific
        if self._model_info and self._model_info.thinking_type == "qwen":
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}

        # Sampling — caller override (summary) or injected defaults (chat)
        if sampling:
            payload.update(sampling)
        elif self._sampling:
            mode = "thinking" if thinking else "non_thinking"
            payload.update(self._sampling.get(mode, {}))

        data = self._post(payload)
        return self._parse_response(data)

    # --- Outbound translation ---

    def _build_messages(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]

        for m in messages:
            role = m["role"]

            if role == "tool_result":
                for r in m["results"]:
                    out.append({
                        "role": "tool",
                        "tool_call_id": r["tool_call_id"],
                        "content": r["content"],
                    })

            elif role == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m["content"] or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            },
                        }
                        for tc in m["tool_calls"]
                    ],
                })

            else:
                out.append({"role": role, "content": m["content"]})

        return out

    @staticmethod
    def _translate_tool(tool: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "input_schema",
                    {"type": "object", "properties": {}},
                ),
            },
        }

    # --- Inbound translation ---

    def _parse_response(self, data: dict) -> ChatResponse:
        choice = data["choices"][0]
        message = choice["message"]

        text = message.get("content") or ""
        text = _SEQ_ANNOTATION.sub("", text).strip()

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc["function"]
            try:
                tool_input = json.loads(fn["arguments"]) if fn["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning(
                    "Model produced unparseable tool arguments for %s: %r",
                    fn["name"], fn["arguments"],
                )
                tool_input = {}
            tool_calls.append(
                ToolCall(name=fn["name"], input=tool_input, id=tc["id"])
            )

        model_used = data.get("model", self._model_id)
        thinking = bool(message.get("reasoning_content"))

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            logger.warning(
                "Response truncated — hit max_tokens (%d)", self._max_tokens,
            )

        return ChatResponse(
            text=text,
            model_used=model_used,
            thinking=thinking,
            tool_calls=tool_calls,
        )

    # --- HTTP ---

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Local inference server unreachable at {self._endpoint} — "
                f"is llama-server running? ({e})"
            ) from e
