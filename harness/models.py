"""SQLAlchemy tables for scenarios, runs and steps."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (UniqueConstraint("name", name="uq_scenarios_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    tools_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    expected_tools_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    required_args_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    forbidden_tools_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    rubric: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="scenario", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Scenario id={self.id} name={self.name!r}>"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    checks_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    verdict_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    scenario: Mapped[Scenario] = relationship(back_populates="runs")
    steps: Mapped[list["Step"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Step.index"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Run id={self.id} scenario_id={self.scenario_id} status={self.status!r}>"


class Step(Base):
    __tablename__ = "steps"
    __table_args__ = (UniqueConstraint("run_id", "index", name="uq_steps_run_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    run: Mapped[Run] = relationship(back_populates="steps")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Step run_id={self.run_id} index={self.index} kind={self.kind!r}>"
