"""Local inference provider.

Talks to llama-server's OpenAI-compatible /v1/chat/completions endpoint.
Each instance wraps a single model on a single endpoint. The factory
builds one per available model; the router or runtime holds them.

Model metadata, reasoning controls, and sampling parameters come from
the hardware profile, injected at construction by the factory. This
module never imports from profile.py for model-specific values.

Reasoning and effort reach the model through chat_template_kwargs.
The profile's [reasoning] table names the keys each model's template
expects, so a new model with a new dialect needs a profile entry,
not code.

stdlib urllib only — no SDK for localhost HTTP.
"""

import json
import logging
import re
import urllib.error
import urllib.request

import basic_bot.config as config
from basic_bot.providers.protocol import (
    REASONING_OPTIONAL,
    ChatResponse,
    ModelInfo,
    ToolCall,
)

logger = logging.getLogger(__name__)

_SEQ_ANNOTATION = re.compile(r'<!--\s*seq:\d+\s*-->')


class LocalProvider:
    """InferenceProvider implementation for a local llama-server."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        max_tokens: int,
        model_info: ModelInfo,
        reasoning: dict | None = None,
        sampling: dict | None = None,
    ):
        self._model_id = model_id
        self._endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self._max_tokens = max_tokens
        self._model_info = model_info
        self._sampling = sampling or {}

        # Template keys from the profile's [reasoning] table
        reasoning = reasoning or {}
        self._switch_key = reasoning.get("switch_key")
        self._effort_key = reasoning.get("effort_key")

        logger.info(
            "Local provider configured: %s → %s", model_id, self._endpoint,
        )

    # --- Protocol: model catalog ---

    def get_models(self) -> dict[str, ModelInfo]:
        return {self._model_id: self._model_info}

    def get_default_model(self) -> str:
        return self._model_id

    def get_fallback_model(self) -> str:
        return self._model_id

    # --- Protocol: chat ---
    # --- Method for inference models (chat AND summary) ---

    def chat(
        self,
        *,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        model_id: str | None = None,
        effort: str | None = None,
        thinking: bool | None = None,
        sampling: dict | None = None,
    ) -> ChatResponse:
        reasoning_on, level = self._model_info.resolve(thinking, effort)

        payload: dict = {
            "model": self._model_id,
            "messages": self._build_messages(system, messages),
            "max_tokens": self._max_tokens,
        }

        if tools:
            payload["tools"] = [self._translate_tool(t) for t in tools]

        # Reasoning and effort, in this model's template vocabulary
        template_kwargs: dict = {}
        if self._model_info.reasoning == REASONING_OPTIONAL and self._switch_key:
            template_kwargs[self._switch_key] = reasoning_on
        if level and self._effort_key:
            template_kwargs[self._effort_key] = level
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs

        # Sampling — caller override (summary) or profile defaults (chat)
        if sampling:
            payload.update(sampling)
        else:
            payload.update(self._sampling_for(reasoning_on))

        data = self._post(payload)
        return self._parse_response(data)

    def _sampling_for(self, reasoning_on: bool) -> dict:
        """Profile sampling for the current reasoning state.

        A model with separate settings per mode has thinking and
        non_thinking subtables. A model with one set of settings has
        a flat table, used as it is.
        """
        if "thinking" in self._sampling or "non_thinking" in self._sampling:
            mode = "thinking" if reasoning_on else "non_thinking"
            return self._sampling.get(mode, {})
        return self._sampling

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

        reasoning = message.get("reasoning_content")
        thinking = bool(reasoning)

        if reasoning and config.LOG_REASONING:
            logger.info("Reasoning: %s", reasoning)

        # Token usage
        usage = data.get("usage", {})
        if usage:
            logger.info(
                "Tokens — prompt: %d, completion: %d, total: %d",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

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
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise RuntimeError(
                    f"Local inference server rejected request (HTTP 400) — "
                    f"context may exceed model's max context length"
                ) from e
            raise RuntimeError(
                f"Local inference server error at {self._endpoint} — "
                f"HTTP {e.code}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Local inference server unreachable at {self._endpoint} — "
                f"is llama-server running? ({e})"
            ) from e
