"""The agent loop: drives a tool-using model through one scenario and records every step."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from openai import AsyncOpenAI

from harness.config import Settings, get_settings
from harness.schemas import ScenarioIn


@dataclass
class StepRecord:
    """One recorded event of a run: a model turn, a tool call or the final answer."""

    kind: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    tokens: int = 0


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class AgentRunner:
    """Runs one scenario against the configured model and returns the recorded steps.

    The runner is deliberately small: it sends the scenario task with the scenario's
    tool definitions, executes the tool calls the model asks for through the handlers
    passed in, and stops when the model answers without a tool call or when the step
    budget runs out. Every turn is timed and its token usage is kept so the checks and
    the judge can look at the whole trajectory, not only the last message.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
        tool_handlers: dict[str, ToolHandler] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.tool_handlers = tool_handlers or {}

    async def run(self, scenario: ScenarioIn) -> list[StepRecord]:
        """Execute the scenario and return the steps in the order they happened."""
        steps: list[StepRecord] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": scenario.system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.task},
        ]
        tools = [tool.as_openai_tool() for tool in scenario.tools]

        for index in range(self.settings.max_steps):
            started = time.perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=self.settings.agent_model,
                    messages=messages,
                    tools=tools or None,
                    temperature=0,
                )
            except Exception as exc:  # network, quota, bad request
                steps.append(
                    StepRecord(
                        kind="error",
                        name="model_call",
                        payload={"message": str(exc)},
                        latency_ms=_elapsed_ms(started),
                    )
                )
                return steps

            latency = _elapsed_ms(started)
            choice = response.choices[0]
            message = choice.message
            tokens = response.usage.total_tokens if response.usage else 0
            tool_calls = message.tool_calls or []

            steps.append(
                StepRecord(
                    kind="model",
                    name=f"turn_{index}",
                    payload={
                        "content": message.content or "",
                        "tool_calls": [call.function.name for call in tool_calls],
                    },
                    latency_ms=latency,
                    tokens=tokens,
                )
            )

            if not tool_calls:
                steps.append(
                    StepRecord(
                        kind="answer",
                        name="final_answer",
                        payload={"content": message.content or ""},
                        latency_ms=0,
                        tokens=0,
                    )
                )
                return steps

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                arguments = _parse_arguments(call.function.arguments)
                result, tool_latency = await self._call_tool(call.function.name, arguments)
                steps.append(
                    StepRecord(
                        kind="tool",
                        name=call.function.name,
                        payload={"arguments": arguments, "result": result},
                        latency_ms=tool_latency,
                        tokens=0,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, default=str)[:4000],
                    }
                )

        steps.append(
            StepRecord(
                kind="error",
                name="step_budget_exhausted",
                payload={"max_steps": self.settings.max_steps},
            )
        )
        return steps

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[Any, int]:
        """Run one tool handler and return its result with the time it took."""
        handler = self.tool_handlers.get(name)
        started = time.perf_counter()
        if handler is None:
            return {"error": f"unknown tool: {name}"}, _elapsed_ms(started)
        try:
            result = await handler(arguments)
        except Exception as exc:
            return {"error": str(exc)}, _elapsed_ms(started)
        return result, _elapsed_ms(started)


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse the JSON arguments of a tool call, tolerating an empty or broken payload."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Use the provided tools when they help you answer. "
    "Call one tool at a time, read its result, and only then decide the next step. "
    "When you have enough information, answer in plain text without calling a tool."
)
