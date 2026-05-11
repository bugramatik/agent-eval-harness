"""FastAPI app with routes to start runs, read results and compare runs."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from harness import __version__
from harness.agent import AgentRunner
from harness.checks import run_checks
from harness.config import Settings, get_settings
from harness.judge import Judge
from harness.models import Base
from harness.scenarios import ScenarioError, load_directory
from harness.schemas import CompareOut, RunOut, RunReport, ScenarioIn, ScenarioOut
from harness.store import Store


def _build_store(settings: Settings) -> Store:
    engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    Base.metadata.create_all(engine)
    return Store(sessionmaker(bind=engine, expire_on_commit=False, class_=Session))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.store = _build_store(settings)
    yield


app = FastAPI(title="Agent Eval Harness", version=__version__, lifespan=lifespan)


def get_store() -> Store:
    return app.state.store


def get_config() -> Settings:
    return app.state.settings


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/scenarios", response_model=ScenarioOut, status_code=201)
def create_scenario(scenario: ScenarioIn, store: Store = Depends(get_store)) -> ScenarioOut:
    return store.save_scenario(scenario)


@app.get("/scenarios", response_model=list[ScenarioOut])
def list_scenarios(store: Store = Depends(get_store)) -> list[ScenarioOut]:
    return store.list_scenarios()


@app.post("/scenarios/import", response_model=list[ScenarioOut])
def import_scenarios(
    directory: str | None = Query(default=None),
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_config),
) -> list[ScenarioOut]:
    target = Path(directory or settings.scenario_dir)
    try:
        scenarios = load_directory(target)
    except ScenarioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [store.save_scenario(scenario) for scenario in scenarios]


@app.post("/runs/{scenario_id}", response_model=RunReport)
async def start_run(
    scenario_id: int,
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_config),
) -> RunReport:
    scenario = store.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"scenario {scenario_id} not found")

    run = store.create_run(scenario_id, settings.agent_model)
    runner = AgentRunner(settings=settings)
    steps = await runner.run(scenario)
    store.add_steps(run.id, steps)

    checks = run_checks(steps, scenario)
    answers = [step for step in steps if step.kind == "answer"]
    final_answer = (answers[-1].payload or {}).get("content", "") if answers else ""

    verdict = None
    if settings.has_api_key and final_answer:
        verdict = await Judge(settings=settings).score(scenario, final_answer)

    store.finish_run(run.id, checks, verdict, final_answer)
    report = store.get_run(run.id)
    if report is None:  # pragma: no cover - the run was just written
        raise HTTPException(status_code=500, detail="run disappeared after scoring")
    return report


@app.get("/runs", response_model=list[RunOut])
def list_runs(limit: int = Query(default=50, ge=1, le=500), store: Store = Depends(get_store)) -> list[RunOut]:
    return store.list_runs(limit=limit)


@app.get("/runs/{run_id}", response_model=RunReport)
def get_run(run_id: int, store: Store = Depends(get_store)) -> RunReport:
    report = store.get_run(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return report


@app.get("/compare", response_model=CompareOut)
def compare(
    a: int = Query(ge=1),
    b: int = Query(ge=1),
    store: Store = Depends(get_store),
) -> CompareOut:
    try:
        return store.compare(a, b)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/scenarios/{scenario_id}/export")
def export_scenario(scenario_id: int, store: Store = Depends(get_store)) -> dict:
    scenario = store.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"scenario {scenario_id} not found")
    return json.loads(scenario.model_dump_json())
