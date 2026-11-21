"""Chat orchestration.

Provider-agnostic conversation loop: loads memory, builds the prompt,
calls the provider, handles tool execution, and returns the result.
The provider handles all API translation internally.

The prompt is ordered from most stable to most volatile so the
inference server's prompt cache can reuse as much as possible. The
system prompt (persona, instructions, context, tools, summary)
changes only at startup or at a fold. The transcript only grows.

Nothing that changes from turn to turn is written into the prompt.
The current model and settings reach the model only through the
provider: as API parameters, or through the model's own chat
template. The reply's metadata records them for the user. The model
asks for the current time with the date_and_time tool.

Each turn keeps a record of its tool calls, saved with the reply (see
basic_bot.store): the tool, its input, how it went, and the sources
its result named. A tool names its sources with a "sources" list at
the top of its result. The record keeps them, and they're taken out of
the result before the model reads it, since the model already sees
the same addresses in the result itself. On later turns the model
sees the record only through recall_message and search_archive, which
show the tools each reply used.

The reply's thinking means the model reasoned at any step of the
turn, not only the last. A turn that reasons, calls a tool, and then
answers straight from the result still reasoned.
"""

import json
import logging
import re
import threading
from collections.abc import Callable

import basic_bot.config as config
from basic_bot.annotations import strip_comments
from basic_bot.memory import load_history, load_window
from basic_bot.providers.protocol import (
    ChatResponse,
    InferenceProvider,
    ModelInfo,
    RequestCancelled,
)
from basic_bot.runtime import BotRuntime
from basic_bot.shutdown import shutting_down

logger = logging.getLogger(__name__)

# Rounds past MAX_TOOL_ROUNDS in which calls are refused before the
# turn is ended for the model
REFUSED_ROUNDS = 2

# The parts of a source the record keeps
SOURCE_FIELDS = ("title", "url", "section")

REMINDER = "\n\nI planned to call {tool} and haven't yet, so I'll call it now.\n"


class TurnInterrupted(Exception):
    """The user interrupted the turn before it was saved."""


def _check(stop: threading.Event | None) -> None:
    """Raise TurnInterrupted if the user has interrupted the turn."""
    if stop is not None and stop.is_set():
        raise TurnInterrupted


CUT_OFF_NOTE = "\n\n[This reply ran past its length limit and was cut short here.]"


def _execute_tool(
    registry: dict, name: str, context: dict, tool_input: dict,
) -> str:
    """Dispatch a tool call and return the result as a JSON string.

    The handler's context is the turn's context plus its own group's
    settings, so a tool sees its group's settings and no other's.
    """
    entry = registry.get(name)
    if not entry:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    tool_context = {**context, "settings": entry.get("settings", {})}
    try:
        result = entry["handler"](tool_context, **tool_input)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        logger.exception("Tool '%s' failed", name)
        return json.dumps({"error": str(e)})


def _outcome(result) -> str:
    """How a call went, from its parsed result.

    "error" if the result carries one, "nothing found" if its results
    are an empty list, and "ok" otherwise.
    """
    if isinstance(result, dict):
        if result.get("error"):
            return "error"
        if "results" in result and result["results"] == []:
            return "nothing found"
    return "ok"


def _sources(value) -> list[dict]:
    """The sources a result named, keeping only their text fields."""
    if not isinstance(value, list):
        return []
    kept = []
    for source in value:
        if not isinstance(source, dict):
            continue
        entry = {
            k: source[k] for k in SOURCE_FIELDS
            if isinstance(source.get(k), str) and source[k]
        }
        if entry.get("url"):
            kept.append(entry)
    return kept


def _record_call(name: str, tool_input: dict, content: str) -> tuple[dict, str]:
    """A call's record, and its result as the model will read it.

    The result's "sources" go into the record and out of the result.
    A result without sources reaches the model exactly as the tool
    wrote it.
    """
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        parsed = None

    record = {"tool": name, "input": tool_input, "outcome": _outcome(parsed)}
    if isinstance(parsed, dict) and "sources" in parsed:
        record["sources"] = _sources(parsed.pop("sources"))
        content = json.dumps(parsed)
    else:
        record["sources"] = []
    return record, content


def _faked_call(text: str, names: list[str]) -> str | None:
    """Return a tool the reply writes out as a call, such as search_archive(...)."""
    for name in names:
        if re.search(rf"\b{re.escape(name)}\s*\(", text):
            return name
    return None


def _planned_tool(response: ChatResponse, names: list[str]) -> str | None:
    """Return the tool a reply without a call should have called, if any.

    The reply is read as the model wrote it, before its comments are
    removed. A tool the reply writes out as a call, even inside a
    made-up annotation, counts first. Otherwise it is
    the tool the reasoning named last, unless the reply names it too.
    """
    faked = _faked_call(response.text, names)
    if faked:
        return faked
    found = [
        (response.reasoning.rfind(name), name)
        for name in names if name in response.reasoning
    ]
    if not found:
        return None
    tool = max(found)[1]
    return None if tool in response.text else tool


def _shorten(text: str) -> str:
    """Return a reply cut off at max_tokens, kept short, with a note saying so."""
    limit = config.CUT_OFF_REPLY_CHARS
    if len(text) > limit:
        head = text[:limit]
        space = max(head.rfind("\n"), head.rfind(" "))
        text = head[:space] if space > 0 else head
    return text.rstrip() + CUT_OFF_NOTE


# ---------------------------------------------------------------------------
# System prompt — stable between startup and fold
# ---------------------------------------------------------------------------

def _build_tools_section(tool_registry: dict) -> str:
    """List the available tools by name. Fixed for the session."""
    if not tool_registry:
        return ""
    names = ", ".join(sorted(tool_registry.keys()))
    return f"# TOOLS\n\nYou have {len(tool_registry)} tools: {names}."


def _build_memory_section(summary: str, summarized_through: int) -> str:
    """Present the rolling summary. Changes only at a fold."""
    if summary and summarized_through:
        return (
            "# MEMORY\n\n"
            f"Running summary of messages 1 through {summarized_through}:\n\n"
            f"{summary}"
        )
    return "# MEMORY\n\nNo earlier messages have been summarized yet."


def build_system_prompt(
    runtime: BotRuntime,
    summary: str = "",
    summarized_through: int = 0,
) -> str:
    """Build the system prompt: stable prefix, tools, and memory."""
    parts = [runtime.persona]

    tools_section = _build_tools_section(runtime.tool_registry)
    if tools_section:
        parts.append(tools_section)

    parts.append(_build_memory_section(summary, summarized_through))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def _prepare_messages(window: list[dict]) -> list[dict]:
    """Copy the window for the API.

    A fresh copy is built per attempt so a fallback never inherits a
    failed attempt's tool calls, which the tool loop appends.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in window
    ]


def _lightest_settings(model_info: ModelInfo) -> tuple[bool, str | None]:
    """Reasoning off and the lowest effort level, as far as the model allows."""
    lowest = model_info.effort_levels[0] if model_info.effort_levels else None
    return model_info.resolve(False, lowest)


# ---------------------------------------------------------------------------
# Chat orchestration
# ---------------------------------------------------------------------------

def chat_with_model(
    runtime: BotRuntime,
    user_id: str,
    user_message: str,
    model_id: str | None = None,
    effort: str | None = None,
    thinking: bool | None = None,
    *,
    stop: threading.Event | None = None,
    narrate: Callable[[str], None] | None = None,
) -> dict:
    """Chat using conversation memory and tools.

    effort and thinking of None mean "use the model's default."

    stop is set when the user interrupts the turn. The turn then ends
    with TurnInterrupted, and no fallback is tried.

    narrate is called with each tool's narration as the tool starts.

    Returns a result dict with the reply text and metadata reflecting
    what was actually used (not what was requested, where detectable).
    """
    provider: InferenceProvider = runtime.chat_provider

    if model_id is None:
        model_id = provider.get_default_model()

    models = provider.get_models()
    if model_id not in models:
        logger.warning("Unknown model '%s', using default", model_id)
        model_id = provider.get_default_model()

    model_info = models[model_id]
    fallback_id = provider.get_fallback_model()
    fallback_info = models[fallback_id]

    reasoning_on, level = model_info.resolve(thinking, effort)

    logger.info("Processing chat for user %s", user_id)

    window, summary, position = load_window(
        runtime.store, user_id, user_message,
    )
    summarized_through = position["summarized_through"]
    logger.info(
        "Context: %d prior messages, summary=%s, boundary=seq %s",
        len(window) - 1, bool(summary), summarized_through,
    )

    # One system prompt for every attempt — it does not depend on the model
    system_prompt = build_system_prompt(runtime, summary, summarized_through)

    tool_schemas = [
        entry["schema"] for entry in runtime.tool_registry.values()
    ]
    context = {
        "agent_id": runtime.agent_id,
        "user_id": user_id,
        "store": runtime.store,
        "embedder": runtime.embedder,
    }

    messages = _prepare_messages(window)

    logger.info(
        "Sending to %s (effort=%s, reasoning=%s) — %d messages, %d tools",
        model_info.display_name, level, reasoning_on,
        len(messages), len(tool_schemas),
    )

    try:
        response, calls, reasoned = _chat_loop(
            provider, messages, system_prompt,
            tool_schemas, context, runtime.tool_registry,
            model_id, level, reasoning_on, stop, narrate,
        )

        if response.model_used != model_id:
            logger.warning(
                "Model mismatch: requested %s, got %s",
                model_id, response.model_used,
            )

        used_info = models.get(response.model_used, model_info)

        logger.info(
            "Response from %s (%d chars, thinking=%s, %d tool calls)",
            used_info.display_name, len(response.text), reasoned, len(calls),
        )

        return {
            "reply": response.text,
            "model_used": response.model_used,
            "display_name": used_info.display_name,
            "effort": level,
            "thinking": reasoned,
            "fallback": False,
            "model_mismatch": response.model_used != model_id,
            "tools": calls,
        }

    except TurnInterrupted:
        raise

    except Exception:
        # A request cut off by shutdown isn't a failure to recover from:
        # the servers are being stopped on purpose, and retrying would
        # only be declined. The session explains it to the user.
        if shutting_down():
            raise
        if stop is not None and stop.is_set():
            raise TurnInterrupted from None

        # Retry on the fallback model at its lightest settings. If that
        # is exactly what just failed, retrying would change nothing,
        # so the error goes straight to the session, which logs it.
        fb_reasoning, fb_level = _lightest_settings(fallback_info)
        if model_id == fallback_id and (fb_reasoning, fb_level) == (reasoning_on, level):
            raise

        # The retry happens, so this error is logged here, with its
        # traceback. If the retry also fails, the session logs that one.
        if model_id == fallback_id:
            logger.exception(
                "Error with %s; retrying at its lightest settings",
                model_info.display_name,
            )
        else:
            logger.exception(
                "Error with %s, falling back to %s",
                model_info.display_name, fallback_info.display_name,
            )

        try:
            response, calls, reasoned = _chat_loop(
                provider, _prepare_messages(window), system_prompt,
                tool_schemas, context, runtime.tool_registry,
                fallback_id, fb_level, fb_reasoning, stop, narrate,
            )
        except TurnInterrupted:
            raise
        except Exception:
            logger.error("Fallback to %s also failed", fallback_info.display_name)
            raise

        used_info = models.get(response.model_used, fallback_info)

        logger.info(
            "Fallback to %s succeeded (%d chars, thinking=%s, %d tool calls)",
            used_info.display_name, len(response.text), reasoned, len(calls),
        )

        return {
            "reply": response.text,
            "model_used": response.model_used,
            "display_name": used_info.display_name,
            "effort": fb_level,
            "thinking": reasoned,
            "fallback": True,
            "model_mismatch": response.model_used != fallback_id,
            "tools": calls,
        }


def warm_request(
    runtime: BotRuntime,
    user_id: str,
    model_id: str,
    effort: str | None = None,
    thinking: bool | None = None,
) -> dict:
    """The arguments for warming the cache for the next turn.

    The prompt is built as chat_with_model() will build it for the
    next message, minus that message: the same system prompt, tools,
    and settings, and the stored conversation ending with the last
    reply. Reading it into the cache now means the next request only
    adds the new message, and nothing has to be reread.

    The session calls this while it holds the store, and sends the
    request in the background (see basic_bot.session).
    """
    provider = runtime.chat_provider
    info = provider.get_models()[model_id]
    reasoning_on, level = info.resolve(thinking, effort)

    window, summary, state = load_history(runtime.store, user_id)
    return {
        "messages": _prepare_messages(window),
        "system": build_system_prompt(
            runtime, summary, state["summarized_through"],
        ),
        "tools": [e["schema"] for e in runtime.tool_registry.values()] or None,
        "model_id": model_id,
        "effort": level,
        "thinking": reasoning_on,
    }


def _chat_loop(
    provider: InferenceProvider,
    messages: list[dict],
    system: str,
    tool_schemas: list[dict],
    context: dict,
    tool_registry: dict,
    model_id: str,
    effort: str | None,
    thinking: bool,
    stop: threading.Event | None = None,
    narrate: Callable[[str], None] | None = None,
) -> tuple[ChatResponse, list[dict], bool]:
    """Call the provider in a loop, executing tools until no more are requested.

    Returns (response, calls, reasoned): the final response, the
    record of every call the model made, in order, refused ones too,
    and whether the model reasoned at any step.

    A guard against chains that never end: after config.MAX_TOOL_ROUNDS
    rounds of tool calls, further calls aren't run, and each gets a
    result telling the model to answer with what it has. The tools stay
    in the request, since removing them would change the start of the
    prompt and the server would reread all of it. If the model still
    won't answer after a few more rounds, the loop ends with a reply
    that says so.
    """
    rounds = 0
    calls: list[dict] = []
    reasoned = False
    reminded = False
    prefill = None
    while True:
        _check(stop)
        try:
            response = provider.chat(
                messages=messages,
                system=system,
                tools=tool_schemas or None,
                model_id=model_id,
                effort=effort,
                thinking=thinking,
                prefill=prefill,
            )
        except RequestCancelled:
            raise TurnInterrupted from None
        _check(stop)
        prefill = None
        reasoned = reasoned or response.thinking

        tool = None
        if not response.tool_calls and not reminded and rounds == 0:
            tool = _planned_tool(response, list(tool_registry))
        if tool:
            logger.warning(
                "The reply needed %s but made no call; asking again",
                tool,
            )
            reminded = True
            prefill = response.reasoning.rstrip() + REMINDER.format(tool=tool)
            continue
        response.text = strip_comments(response.text).strip()

        if not response.tool_calls:
            if reminded and rounds == 0:
                logger.warning("Still no tool call after the reminder")
            if response.cut_off:
                logger.warning("The reply ran past max_tokens; saving it shortened")
                response.text = _shorten(response.text)
            return response, calls, reasoned

        rounds += 1
        limit = config.MAX_TOOL_ROUNDS
        if rounds > limit + REFUSED_ROUNDS:
            logger.warning(
                "Tool calls still coming after %d rounds; ending the turn", rounds - 1,
            )
            ended = ChatResponse(
                text=(
                    f"I stopped after {limit} rounds of looking things up "
                    f"without settling on an answer. Could you narrow the "
                    f"question, or tell me what you already know?"
                ),
                model_used=response.model_used,
                thinking=response.thinking,
            )
            return ended, calls, reasoned

        # Append the assistant's tool calls, with the reasoning that led
        # to them, so the model keeps its plan for the rest of the turn.
        # These messages live only for this turn; the turn saves only
        # the final reply.
        messages.append({
            "role": "assistant",
            "content": response.text,
            "tool_calls": response.tool_calls,
            "reasoning": response.reasoning,
        })

        # Execute tools and append results, or past the limit, refuse them
        results = []
        if rounds > limit:
            if rounds == limit + 1:
                logger.warning("Tool round limit reached (%d); asking for an answer", limit)
            refusal = json.dumps({
                "error": f"You've reached this turn's limit of {limit} rounds "
                         f"of tool calls. Don't call more tools: answer now "
                         f"with what you've found, and say what you couldn't "
                         f"confirm.",
            })
            for tc in response.tool_calls:
                calls.append({
                    "tool": tc.name, "input": tc.input,
                    "outcome": "refused", "sources": [],
                })
                results.append({"tool_call_id": tc.id, "content": refusal})
        else:
            for tc in response.tool_calls:
                logger.info("Executing tool: %s", tc.name)
                if narrate is not None and tc.name in tool_registry:
                    narrate(tool_registry[tc.name]["narration"])
                result = _execute_tool(tool_registry, tc.name, context, tc.input)
                record, result = _record_call(tc.name, tc.input, result)
                calls.append(record)
                results.append({"tool_call_id": tc.id, "content": result})
                _check(stop)

        messages.append({"role": "tool_result", "results": results})
