"""LLM judge that scores the final answer against the scenario rubric."""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from harness.config import Settings, get_settings
from harness.schemas import JudgeVerdict, ScenarioIn

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of assistant answers. You are given the task, the "
    "rubric and the answer. Score the answer from 0 to 1 against the rubric only. "
    "Reply with a single JSON object with the keys score (number), passed (boolean), "
    "reasoning (string, at most 60 words) and missing (array of short strings naming "
    "what the answer left out). Do not add any other text."
)

PASS_THRESHOLD = 0.6


class Judge:
    """Scores one final answer with the judge model and returns a structured verdict."""

    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or AsyncOpenAI(api_key=self.settings.openai_api_key)

    async def score(self, scenario: ScenarioIn, final_answer: str) -> JudgeVerdict:
        """Ask the judge model for a verdict and parse it into a JudgeVerdict."""
        user_prompt = (
            f"Task:\n{scenario.task}\n\n"
            f"Rubric:\n{scenario.rubric}\n\n"
            f"Answer:\n{final_answer.strip() or '(empty)'}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            return JudgeVerdict(
                score=0.0,
                passed=False,
                reasoning=f"judge call failed: {exc}",
                missing=[],
            )

        raw = response.choices[0].message.content or "{}"
        return self._parse(raw)

    def _parse(self, raw: str) -> JudgeVerdict:
        """Turn the judge's JSON into a verdict, falling back to a failed verdict."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return JudgeVerdict(
                score=0.0,
                passed=False,
                reasoning="judge returned text that is not JSON",
                missing=[],
            )

        if not isinstance(data, dict):
            return JudgeVerdict(score=0.0, passed=False, reasoning="judge returned no object")

        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = min(max(score, 0.0), 1.0)

        missing = data.get("missing") or []
        if not isinstance(missing, list):
            missing = [str(missing)]

        passed = bool(data.get("passed", score >= PASS_THRESHOLD))
        return JudgeVerdict(
            score=score,
            passed=passed and score >= PASS_THRESHOLD,
            reasoning=str(data.get("reasoning", ""))[:600],
            missing=[str(item)[:120] for item in missing][:10],
        )
