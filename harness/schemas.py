"""Pydantic request and response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolSpec(BaseModel):
    """A tool the agent may call, in the shape the model expects."""

    name: str = Field(min_length=1, max_length=64)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ScenarioIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    task: str = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    required_args: dict[str, list[str]] = Field(default_factory=dict)
    forbidden_tools: list[str] = Field(default_factory=list)
    rubric: str = ""

    @field_validator("name")
    @classmethod
    def name_is_slug(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("scenario name must not be blank")
        return cleaned


class ScenarioOut(ScenarioIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class StepOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    index: int
    kind: str
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int
    tokens: int


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasoning: str = ""
    missing: list[str] = Field(default_factory=list)


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scenario_id: int
    model: str
    status: Literal["pending", "running", "passed", "failed", "error"]
    score: float | None = None
    passed: bool | None = None
    final_answer: str | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class RunReport(RunOut):
    checks: list[CheckResult] = Field(default_factory=list)
    judge: JudgeVerdict | None = None
    steps: list[StepOut] = Field(default_factory=list)


class CompareOut(BaseModel):
    run_a: int
    run_b: int
    score_a: float | None = None
    score_b: float | None = None
    delta: float | None = None
    regressed_checks: list[str] = Field(default_factory=list)
    fixed_checks: list[str] = Field(default_factory=list)
