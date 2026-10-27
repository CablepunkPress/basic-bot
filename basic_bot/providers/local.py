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

The [reasoning] table can also set a budget: the most tokens the
model may spend reasoning on one request, for every effort level or
for each. llama-server counts them, and when the budget runs out it
writes BUDGET_MESSAGE into the reasoning and closes it, so the model
answers instead of reasoning on. It's a guard against reasoning that
never ends, not a way to make answers shorter: max_tokens still caps
the whole reply. llama-server applies it only when it can find the
template's start and end of reasoning. It finds them itself for most
templates; for one it can't, such as Muse's, the table names them
(budget_start_tag and budget_end_tag) and they're sent with the
request. Each model's budget is checked once by hand (models.md).

The [reasoning] table can also give a prefill: the opening words of
the model's reasoning, written for it at the start of each turn. A
model that decides for itself whether a question deserves reasoning,
as Muse does, skips it on turns it judges easy, and without reasoning
it never considers its tools. With a prefill, llama-server opens the
model's reasoning with these words (an assistant message carrying
reasoning_content, continued with continue_final_message "reasoning"),
so every turn begins by reasoning. The words should end in an open
question, so the model has something to work through rather than a
finished thought to close. It goes at the very end of the prompt, so
the cache isn't touched. config.FORCE_REASONING turns it off, for an
agent or for comparison.

warm() reads a prompt into the server's cache without generating
(max_tokens 0, which llama-server documents as evaluating the prompt
into the cache). It builds the prompt exactly as chat() does, through
_payload(), so the cache it prepares is the start of the next request.
The prompt ends with the last reply, rendered as a finished message,
which needs llama-server started with --no-prefill-assistant (see
basic_bot.infrastructure.server); otherwise the server would leave the
reply open, to be continued, and the next request wouldn't match.

stdlib urllib only — no SDK for localhost HTTP.
"""

import json
import logging
import urllib.error
import urllib.request

import basic_bot.config as config
from basic_bot.annotations import strip as strip_annotations
from basic_bot.providers.protocol import (
    REASONING_OPTIONAL,
    ChatResponse,
    ModelInfo,
    ToolCall,
)

logger = logging.getLogger(__name__)

# Written into the model's reasoning, in its own voice, when the
# reasoning budget runs out. llama-server adds the end-of-reasoning
# tag after it, so the next thing the model writes is its answer.
BUDGET_MESSAGE = (
    "\n\nI've thought about this long enough. I'll answer now, "
    "and if I'm not sure, I'll say so plainly and ask.\n"
)


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
        self._budget = reasoning.get("budget")
        self._budget_tags = {
            "reasoning_budget_start_tag": reasoning.get("budget_start_tag"),
            "reasoning_budget_end_tag": reasoning.get("budget_end_tag"),
        }
        self._prefill = reasoning.get("prefill")

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

        payload = self._payload(system, messages, tools, reasoning_on, level)
        payload["max_tokens"] = self._max_tokens

        # Open the model's reasoning for it, at the start of a turn only:
        # after a tool result it's already in the middle of its work.
        # "reasoning_content" is the name llama-server accepts; v0.7.0
        # rejects anything else, where v0.5.0 quietly ignored it.
        if (
            self._prefill and reasoning_on and config.FORCE_REASONING
            and messages and messages[-1]["role"] == "user"
        ):
            payload["messages"].append({
                "role": "assistant",
                "content": "",
                "reasoning_content": self._prefill,
            })
            payload["continue_final_message"] = "reasoning_content"
            payload["add_generation_prompt"] = False

        # Reasoning budget, a guard against reasoning that never ends
        budget = self._budget_for(reasoning_on, level)
        if budget is not None:
            payload["reasoning_budget_tokens"] = budget
            payload["reasoning_budget_message"] = BUDGET_MESSAGE
            payload.update({k: v for k, v in self._budget_tags.items() if v})

        # Sampling — caller override (summary) or profile defaults (chat)
        if sampling:
            payload.update(sampling)
        else:
            payload.update(self._sampling_for(reasoning_on))

        data = self._post(payload)
        return self._parse_response(data, budget)

    def warm(
        self,
        *,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        model_id: str | None = None,
        effort: str | None = None,
        thinking: bool | None = None,
    ) -> None:
        """Read this prompt into the server's cache, generating nothing."""
        reasoning_on, level = self._model_info.resolve(thinking, effort)
        payload = self._payload(system, messages, tools, reasoning_on, level)
        payload["max_tokens"] = 0
        payload["add_generation_prompt"] = False
        self._log_timing(self._post(payload), "Warmed")

    def _payload(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        reasoning_on: bool,
        level: str | None,
    ) -> dict:
        """Everything in a request that shapes the prompt.

        chat() and warm() both start from this, so the prompt warm()
        reads is the one chat() will send.
        """
        payload: dict = {
            "model": self._model_id,
            "messages": self._build_messages(system, messages),
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

        return payload

    def _budget_for(self, reasoning_on: bool, level: str | None) -> int | None:
        """The reasoning budget for this request, or None for no limit.

        The profile gives one number for every level, or a table by
        effort level. A level missing from the table has no limit.
        """
        if not reasoning_on or self._budget is None:
            return None
        if isinstance(self._budget, dict):
            return self._budget.get(level) if level else None
        return self._budget

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
                call: dict = {
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
                }
                # The reasoning that led to the calls, earlier in this
                # turn. A template that preserves reasoning puts it back
                # in the prompt, so the model keeps its plan and the
                # server's cache still matches; one that doesn't drops it.
                if m.get("reasoning"):
                    call["reasoning_content"] = m["reasoning"]
                out.append(call)

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

    def _parse_response(self, data: dict, budget: int | None = None) -> ChatResponse:
        choice = data["choices"][0]
        message = choice["message"]

        text = message.get("content") or ""
        text = strip_annotations(text).strip()

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

        # The budget message in the reasoning means llama-server cut it
        # short. Logged, so budgets can be tuned against real turns.
        if reasoning and budget is not None and BUDGET_MESSAGE.strip() in reasoning:
            logger.warning(
                "Reasoning budget reached (%d tokens); the model was told to answer",
                budget,
            )

        self._log_timing(data)

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
            reasoning=reasoning or "",
        )

    def _log_timing(self, data: dict, what: str = "Timing") -> None:
        """One line per request: what the server reused, read, and wrote.

        llama-server returns its timings with every reply. Logging them
        here puts them in the agent's own log, beside the step they
        belong to, where a server restart can't erase them.
        """
        t = data.get("timings") or {}
        if not t:
            usage = data.get("usage") or {}
            logger.info(
                "%s (%s): prompt %d, completion %d",
                what, self._model_info.display_name,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            return

        def rate(n: float, ms: float) -> str:
            return f"{n / (ms / 1000):.0f}/s" if ms else "-"

        cached = t.get("cache_n", 0)
        read, read_ms = t.get("prompt_n", 0), t.get("prompt_ms", 0.0)
        made, made_ms = t.get("predicted_n", 0), t.get("predicted_ms", 0.0)
        line = (
            f"{what} ({self._model_info.display_name}): "
            f"{cached:,} cached, {read:,} read in {read_ms / 1000:.1f} s "
            f"({rate(read, read_ms)}), {made:,} generated in "
            f"{made_ms / 1000:.1f} s ({rate(made, made_ms)})"
        )
        drafted = t.get("draft_n", 0)
        if drafted:
            line += f", drafter {t.get('draft_n_accepted', 0) / drafted:.2f}"
        logger.info(line)

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
