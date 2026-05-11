"""Deterministic checks over a run's steps and final answer."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from harness.schemas import CheckResult, ScenarioIn


def _tool_steps(steps: Iterable[Any]) -> list[Any]:
    """Return only the steps that are tool calls, in order."""
    return [step for step in steps if getattr(step, "kind", None) == "tool"]


def _arguments(step: Any) -> dict[str, Any]:
    payload = getattr(step, "payload", None) or {}
    arguments = payload.get("arguments") if isinstance(payload, dict) else None
    return arguments if isinstance(arguments, dict) else {}


def check_tool_order(steps: Sequence[Any], expected: Sequence[str]) -> CheckResult:
    """The tools the agent called must follow the expected sequence.

    Extra calls are allowed after the expected sequence, but a missing call, a
    wrong tool or a call out of order fails the check.
    """
    called = [step.name for step in _tool_steps(steps)]
    if not expected:
        return CheckResult(name="tool_order", passed=True, detail="no expected tools")

    position = 0
    for name in called:
        if position < len(expected) and name == expected[position]:
            position += 1

    if position == len(expected):
        return CheckResult(
            name="tool_order",
            passed=True,
            detail=f"called {', '.join(expected)}",
        )

    missing = expected[position]
    return CheckResult(
        name="tool_order",
        passed=False,
        detail=f"expected {missing!r} next, called {', '.join(called) or 'nothing'}",
    )


def check_required_args(steps: Sequence[Any], required: dict[str, list[str]]) -> CheckResult:
    """Every call of a listed tool must carry the arguments the scenario requires."""
    if not required:
        return CheckResult(name="required_args", passed=True, detail="no required arguments")

    problems: list[str] = []
    seen: set[str] = set()
    for step in _tool_steps(steps):
        wanted = required.get(step.name)
        if not wanted:
            continue
        seen.add(step.name)
        arguments = _arguments(step)
        missing = [arg for arg in wanted if arg not in arguments or arguments[arg] in (None, "")]
        if missing:
            problems.append(f"{step.name} missing {', '.join(missing)}")

    never_called = [name for name in required if name not in seen]
    if never_called:
        problems.append(f"never called: {', '.join(sorted(never_called))}")

    if problems:
        return CheckResult(name="required_args", passed=False, detail="; ".join(problems))
    return CheckResult(name="required_args", passed=True, detail="all required arguments present")


def check_forbidden_tools(steps: Sequence[Any], forbidden: Sequence[str]) -> CheckResult:
    """The agent must not call any tool the scenario forbids."""
    if not forbidden:
        return CheckResult(name="forbidden_tools", passed=True, detail="no forbidden tools")

    blocked = set(forbidden)
    hits = [step.name for step in _tool_steps(steps) if step.name in blocked]
    if hits:
        return CheckResult(
            name="forbidden_tools",
            passed=False,
            detail=f"called forbidden tools: {', '.join(sorted(set(hits)))}",
        )
    return CheckResult(name="forbidden_tools", passed=True, detail="no forbidden tool called")


def check_final_answer(steps: Sequence[Any]) -> CheckResult:
    """The run must end with a non-empty answer rather than an error or a budget stop."""
    errors = [step for step in steps if getattr(step, "kind", None) == "error"]
    if errors:
        return CheckResult(
            name="final_answer",
            passed=False,
            detail=f"run stopped on {errors[-1].name}",
        )

    answers = [step for step in steps if getattr(step, "kind", None) == "answer"]
    if not answers:
        return CheckResult(name="final_answer", passed=False, detail="no final answer recorded")

    content = (answers[-1].payload or {}).get("content", "")
    if not str(content).strip():
        return CheckResult(name="final_answer", passed=False, detail="final answer is empty")
    return CheckResult(name="final_answer", passed=True, detail="final answer present")


def run_checks(steps: Sequence[Any], scenario: ScenarioIn) -> list[CheckResult]:
    """Run every deterministic check for one scenario and return the results."""
    return [
        check_tool_order(steps, scenario.expected_tools),
        check_required_args(steps, scenario.required_args),
        check_forbidden_tools(steps, scenario.forbidden_tools),
        check_final_answer(steps),
    ]
