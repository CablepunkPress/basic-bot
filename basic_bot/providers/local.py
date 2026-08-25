"""Local inference provider.

Talks to llama-server's OpenAI-compatible /v1/chat/completions endpoint.
One server, one model — the model is chosen at construction time and is
the only entry get_models() returns. The full catalog of models the
ecosystem knows how to run lives in MODEL_CATALOG; infra/ uses it to
download and launch, this provider uses it to describe what's loaded.

Translation between the engine's internal message format and the OpenAI
chat format happens entirely inside this class, mirroring how
ClaudeProvider owns the Anthropic translation. The engine never sees
OpenAI message shapes.

stdlib urllib only, matching LocalEmbedder — no SDK for localhost HTTP.
"""

import json
import logging
import re
import urllib.error
import urllib.request

from basic_bot.config import INFERENCE_URL
from basic_bot.providers.protocol import ChatResponse, ModelInfo, ToolCall

logger = logging.getLogger(__name__)

_SEQ_ANNOTATION = re.compile(r'<!--\s*seq:\d+\s*-->')

DEFAULT_MAX_TOKENS = 4096
REQUEST_TIMEOUT = 300  # local generation on constrained hardware is slow

# Every model the local stack knows how to serve. infra/ reads this to
# download GGUFs and launch llama-server with --alias set to the key.
# The provider serves whichever one the server was launched with.
MODEL_CATALOG: dict[str, ModelInfo] = {
    "qwen3-8b": ModelInfo(
        id="qwen3-8b",
        display_name="Qwen3 8B",
        provider="Alibaba",
        family="Qwen",
        host="local",
        rank=1,
        effort_levels=None,
        thinking_type="qwen",
    ),
    "qwen3.6-35b-a3b": ModelInfo(
        id="qwen3.6-35b-a3b",
        display_name="Qwen3.6 35B A3B",
        provider="Alibaba",
        family="Qwen",
        host="local",
        rank=2,
        effort_levels=None,
        thinking_type="qwen",
    ),
}


class LocalProvider:
    """InferenceProvider implementation for a local llama-server."""

    def __init__(
        self,
        model_id: str,
        base_url: str = INFERENCE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if model_id not in MODEL_CATALOG:
            raise ValueError(
                f"Unknown local model {model_id!r}. "
                f"Known models: {sorted(MODEL_CATALOG)}"
            )
        self._model_id = model_id
        self._endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self._max_tokens = max_tokens
        logger.info(
            "Local provider ready: %s at %s", model_id, self._endpoint,
        )

    # --- Protocol: model catalog ---

    def get_models(self) -> dict[str, ModelInfo]:
        """Only the loaded model. The server runs one model; offering
        others in the UI would offer models that can't respond."""
        return {self._model_id: MODEL_CATALOG[self._model_id]}

    def get_default_model(self) -> str:
        return self._model_id

    def get_fallback_model(self) -> str:
        """Same as default. Fallback exists for API-world concerns
        (silent substitution, cost tiers); locally there is one model
        and it either answers or it doesn't."""
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
    ) -> ChatResponse:
        payload: dict = {
            "model": self._model_id,
            "messages": self._build_messages(system, messages),
            "max_tokens": self._max_tokens,
        }

        if tools:
            payload["tools"] = [self._translate_tool(t) for t in tools]

        model_info = MODEL_CATALOG[self._model_id]
        if model_info.thinking_type == "qwen":
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}

        data = self._post(payload)
        return self._parse_response(data)

    # --- Outbound translation ---

    def _build_messages(self, system: str, messages: list[dict]) -> list[dict]:
        """Engine format → OpenAI format."""
        out: list[dict] = [{"role": "system", "content": system}]

        for m in messages:
            role = m["role"]

            if role == "tool_result":
                # Engine batches results in one message; OpenAI wants
                # one tool message per result.
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
        """Anthropic tool schema → OpenAI function schema."""
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

        # llama-server echoes its --alias here. infra/ launches with
        # --alias set to the catalog id, so a mismatch in chat.py means
        # the server is running a different model than the provider
        # believes — a real misconfiguration worth surfacing.
        model_used = data.get("model", self._model_id)

        thinking = bool(message.get("reasoning_content"))

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
