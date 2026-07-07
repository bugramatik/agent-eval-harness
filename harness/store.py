"""Database access for scenarios, runs and steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from harness.models import Run, Scenario, Step
from harness.schemas import (
    CheckResult,
    CompareOut,
    JudgeVerdict,
    RunOut,
    RunReport,
    ScenarioIn,
    ScenarioOut,
    StepOut,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Store:
    """Thin persistence layer over the scenario, run and step tables."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    # Scenarios ---------------------------------------------------------

    def save_scenario(self, scenario: ScenarioIn) -> ScenarioOut:
        with self.session_factory() as session:
            row = session.scalar(select(Scenario).where(Scenario.name == scenario.name))
            if row is None:
                row = Scenario(name=scenario.name)
                session.add(row)
            row.task = scenario.task
            row.tools_json = _dumps([tool.model_dump() for tool in scenario.tools])
            row.expected_tools_json = _dumps(scenario.expected_tools)
            row.rubric = scenario.rubric
            session.commit()
            session.refresh(row)
            return self._scenario_out(row)

    def list_scenarios(self) -> list[ScenarioOut]:
        with self.session_factory() as session:
            rows = session.scalars(select(Scenario).order_by(Scenario.name)).all()
            return [self._scenario_out(row) for row in rows]

    def get_scenario(self, scenario_id: int) -> ScenarioOut | None:
        with self.session_factory() as session:
            row = session.get(Scenario, scenario_id)
            return self._scenario_out(row) if row else None

    # Runs --------------------------------------------------------------

    def create_run(self, scenario_id: int, model: str) -> RunOut:
        with self.session_factory() as session:
            scenario = session.get(Scenario, scenario_id)
            if scenario is None:
                raise LookupError(f"scenario {scenario_id} does not exist")
            run = Run(
                scenario_id=scenario_id,
                model=model,
                status="running",
                started_at=_utcnow(),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return self._run_out(run, steps=[])

    def add_steps(self, run_id: int, steps: Sequence[Any]) -> None:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise LookupError(f"run {run_id} does not exist")
            for index, record in enumerate(steps):
                session.add(
                    Step(
                        run_id=run_id,
                        index=index,
                        kind=record.kind,
                        name=record.name,
                        payload_json=_dumps(record.payload),
                        latency_ms=record.latency_ms,
                        tokens=record.tokens,
                    )
                )
            session.commit()

    def finish_run(
        self,
        run_id: int,
        checks: Sequence[CheckResult],
        verdict: JudgeVerdict | None,
        final_answer: str,
    ) -> RunOut:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise LookupError(f"run {run_id} does not exist")
            checks_passed = all(check.passed for check in checks)
            judge_score = verdict.score if verdict else 0.0
            run.status = "passed" if checks_passed and judge_score >= 0.6 else "failed"
            run.passed = checks_passed and judge_score >= 0.6
            run.score = round(judge_score, 4)
            run.finished_at = _utcnow()
            run.checks_json = _dumps([check.model_dump() for check in checks])
            run.verdict_json = _dumps(verdict.model_dump() if verdict else None)
            run.final_answer = final_answer
            session.commit()
            session.refresh(run)
            steps = session.scalars(
                select(Step).where(Step.run_id == run_id).order_by(Step.index)
            ).all()
            return self._run_out(run, steps=list(steps))

    def get_run(self, run_id: int) -> RunReport | None:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                return None
            steps = session.scalars(
                select(Step).where(Step.run_id == run_id).order_by(Step.index)
            ).all()
            return self._report(run, list(steps))

    def list_runs(self, limit: int = 50) -> list[RunOut]:
        with self.session_factory() as session:
            runs = session.scalars(
                select(Run).order_by(Run.started_at.desc()).limit(limit)
            ).all()
            return [self._run_out(run, steps=[]) for run in runs]

    def compare(self, run_a: int, run_b: int) -> CompareOut:
        report_a = self.get_run(run_a)
        report_b = self.get_run(run_b)
        if report_a is None or report_b is None:
            raise LookupError("both runs must exist to compare them")

        checks_a = {check.name: check.passed for check in report_a.checks}
        checks_b = {check.name: check.passed for check in report_b.checks}
        names = sorted(set(checks_a) | set(checks_b))
        regressions = [
            name for name in names if checks_a.get(name, True) and not checks_b.get(name, True)
        ]
        improvements = [
            name for name in names if not checks_a.get(name, True) and checks_b.get(name, True)
        ]
        return CompareOut(
            run_a=report_a.run,
            run_b=report_b.run,
            score_delta=round(report_b.run.score - report_a.run.score, 4),
            regressions=regressions,
            improvements=improvements,
            checks_a=report_a.checks,
            checks_b=report_b.checks,
        )

    # Mapping -----------------------------------------------------------

    def _scenario_out(self, row: Scenario) -> ScenarioOut:
        return ScenarioOut(
            id=row.id,
            name=row.name,
            task=row.task,
            tools=_loads(row.tools_json, []),
            expected_tools=_loads(row.expected_tools_json, []),
            rubric=row.rubric,
        )

    def _step_out(self, row: Step) -> StepOut:
        return StepOut(
            index=row.index,
            kind=row.kind,
            name=row.name,
            payload=_loads(row.payload_json, {}),
            latency_ms=row.latency_ms,
            tokens=row.tokens,
        )

    def _run_out(self, row: Run, steps: list[Step]) -> RunOut:
        return RunOut(
            id=row.id,
            scenario_id=row.scenario_id,
            model=row.model,
            status=row.status,
            score=row.score,
            passed=bool(row.passed),
            started_at=row.started_at,
            finished_at=row.finished_at,
            steps=[self._step_out(step) for step in steps],
        )

    def _report(self, row: Run, steps: list[Step]) -> RunReport:
        checks = [CheckResult(**item) for item in _loads(row.checks_json, [])]
        verdict_raw = _loads(row.verdict_json, None)
        verdict = JudgeVerdict(**verdict_raw) if verdict_raw else None
        return RunReport(
            run=self._run_out(row, steps),
            checks=checks,
            verdict=verdict,
            final_answer=row.final_answer or "",
        )
