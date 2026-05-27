"""Loads scenario files from a directory and validates them."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ScenarioIn, ToolSpec

SUFFIXES = (".json", ".yaml", ".yml")


class ScenarioError(ValueError):
    """Raised when a scenario file cannot be read or does not make sense."""


def _read_raw(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ScenarioError(f"{path.name}: YAML scenarios need PyYAML installed") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ScenarioError(f"{path.name}: top level must be an object")
    return data


def load_scenario(path: Path) -> ScenarioIn:
    """Read one scenario file and return it validated."""
    path = Path(path)
    if not path.is_file():
        raise ScenarioError(f"{path} is not a file")
    try:
        raw = _read_raw(path)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{path.name}: invalid JSON at line {exc.lineno}") from exc

    tools = raw.get("tools", [])
    if isinstance(tools, dict):
        tools = [{"name": name, **(spec or {})} for name, spec in tools.items()]
    raw["tools"] = [ToolSpec.model_validate(tool) for tool in tools]

    try:
        scenario = ScenarioIn.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ScenarioError(f"{path.name}: {exc}") from exc

    validate_scenario(scenario)
    return scenario


def load_directory(directory: Path) -> list[ScenarioIn]:
    """Load every scenario file in a directory, sorted by file name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ScenarioError(f"{directory} is not a directory")
    scenarios: list[ScenarioIn] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUFFIXES and path.is_file():
            scenarios.append(load_scenario(path))
    return scenarios


def validate_scenario(s: ScenarioIn) -> None:
    """Check the cross-field rules a single scenario must satisfy."""
    tool_names = {tool.name for tool in s.tools}
    if not tool_names and s.expected_tools:
        raise ScenarioError(f"{s.name}: expected_tools given but the scenario declares no tools")

    unknown = [name for name in s.expected_tools if name not in tool_names]
    if unknown:
        raise ScenarioError(f"{s.name}: expected tools not declared: {', '.join(unknown)}")

    for tool_name, args in s.required_args.items():
        if tool_name not in tool_names:
            raise ScenarioError(f"{s.name}: required_args names unknown tool {tool_name!r}")
        if not args:
            raise ScenarioError(f"{s.name}: required_args for {tool_name!r} is empty")

    overlap = set(s.forbidden_tools) & set(s.expected_tools)
    if overlap:
        raise ScenarioError(f"{s.name}: tools both expected and forbidden: {', '.join(sorted(overlap))}")

    if not s.rubric.strip():
        raise ScenarioError(f"{s.name}: rubric is empty, the judge would have nothing to score against")
