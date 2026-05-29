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

