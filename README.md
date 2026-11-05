# Agent Eval Harness

Replay tool-using LLM agents against recorded scenarios and score every run with checks and a judge.

## Why

Changing a prompt or swapping a model in an assistant that calls tools is easy; knowing whether the change helped is not. A single edit can fix one flow and quietly break three others, and the usual answer is a throwaway script that replays a few conversations and reads the answers by eye. The harness keeps those scenarios in files, records every step of every run, and scores each run the same way twice, so a prompt rewrite becomes a comparison instead of a guess. It is for people who ship agents and need a regression signal before the change reaches anyone.

## How it works

A scenario file names the task, the tools the agent may call, the tool sequence it should follow, the arguments those calls must carry, the tools it must avoid and a rubric for the final answer. The API stores the scenario, starts a run, and the runner drives the model through the tool loop, recording each model turn, tool call and final answer with its latency and token count. Deterministic checks read the recorded steps, the judge scores the final answer against the rubric, and the run is stored with both results so two runs can be compared side by side.

```mermaid
flowchart LR
    sc["Scenario files"] --> api["FastAPI service"]
    api --> store[("Postgres")]
    api --> runner["Agent runner"]
    runner --> model["Agent model"]
    runner --> tools["Tool handlers"]
    runner --> steps["Recorded steps"]
    steps --> checks["Deterministic checks"]
    steps --> judge["LLM judge"]
    checks --> report["Run report"]
    judge --> report
    report --> store
    store --> ui["Dashboard"]
```

## Key decisions

- Scenarios are files on disk, loaded and validated before they reach the database, so a broken scenario fails at load time with a message naming the field rather than halfway through a paid run.
- The checks are deterministic and separate from the judge: tool order, required arguments, forbidden tools and a non-empty final answer are decided by code, and only the answer quality goes to a model. The cost is that a rubric cannot express everything a reviewer would notice.
- Every model turn and tool call is stored as its own step with latency and tokens, which makes the trajectory inspectable and the comparison meaningful, at the price of a larger table than a single row per run.
- The judge is asked for a JSON object and its reply is parsed defensively, with a failed call or unparseable text recorded as a zero score instead of an exception, so one bad judge response does not lose the run.

## Tech stack

- Python 3.11 and FastAPI with uvicorn
- Pydantic and pydantic-settings for schemas and configuration
- SQLAlchemy 2.0 with psycopg against PostgreSQL 16
- OpenAI Python client for the agent model and the judge
- pytest and pytest-asyncio for tests, ruff for linting
- Docker and docker-compose for the local stack, Kubernetes manifests for deployment

## Quick start

Prerequisites: Python 3.11 or newer, Docker with the compose plugin, and an OpenAI API key.

```bash
docker compose up --build
```

The API listens on port 8000 and Postgres on 5432. To run the service outside Docker:

```bash
pip install -e ".[dev]"
uvicorn harness.api:app --host 0.0.0.0 --port 8000
```

| Variable | What it sets | Default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy connection string for Postgres | `postgresql+psycopg://harness:harness@localhost:5432/harness` |
| `OPENAI_API_KEY` | Key used by the agent runner and the judge | empty |
| `AGENT_MODEL` | Model the agent runs on | `gpt-4o-mini` |
| `JUDGE_MODEL` | Model that scores final answers | `gpt-4o` |
| `MAX_STEPS` | Model turns allowed per run before the budget stops it | `8` |
| `REQUEST_TIMEOUT` | Timeout in seconds for model calls | `60.0` |
| `SCENARIO_DIR` | Directory the service reads scenario files from | `scenarios` |

## Usage

Start a run for a stored scenario and read the report back:

```bash
curl -s -X POST http://localhost:8000/runs/1
```

```json
{
  "id": 41,
  "scenario_id": 1,
  "model": "gpt-4o-mini",
  "status": "passed",
  "score": 0.85,
  "passed": true,
  "final_answer": "The invoice INV-2041 is overdue by 12 days; a reminder was scheduled for tomorrow.",
  "started_at": "2025-11-04T09:12:31Z",
  "finished_at": "2025-11-04T09:12:33Z"
}
```

Compare two runs to see which checks moved:

```bash
curl -s "http://localhost:8000/compare?a=40&b=41"
```

## Project structure

```text
harness/        agent loop, checks, judge, store, API
k8s/            Deployment and Service for the API
ui/index.html   single-page dashboard
Dockerfile      image that runs the API with uvicorn
docker-compose.yml  API and Postgres for local work
```

## Limitations

- Tool handlers are passed in by the caller; the service does not ship a registry of real tools, so scenarios run against the handlers a test or script supplies.
- Runs execute inside the request that starts them, so a long scenario holds a worker for its whole duration.
- The judge is a single model call with no calibration set, and its score is only as stable as the rubric it is given.
- Scenario files in YAML need PyYAML installed, which is not a declared dependency.

## License

MIT. The project builds on no other application; it uses the OpenAI Python client and FastAPI as libraries.
