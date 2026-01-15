from typing import Any
import json
import numpy as np
from pathlib import Path
import math

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class Agent:
    required_roles = ["purple"]
    required_config_keys = []

    def __init__(self):
        self.messenger = Messenger()
        self.benchmark = self.load_benchmark()
        self.weights = {
            "core": 1.0,
            "edge": 0.75,
            "noisy": 0.5,
        }

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing = set(self.required_roles) - set(request.participants.keys())
        if missing:
            return False, f"Missing roles: {missing}"
        return True, "ok"

    def load_benchmark(self) -> dict:
        path = Path("data/reflena_benchmark.json")
        if not path.exists():
            raise RuntimeError("Benchmark file data/reflena_benchmark.json not found")
        with path.open() as f:
            return json.load(f)

    def execute_candidate(self, code: str, problem: dict) -> tuple[float, float]:
        env = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
            "math": math,
        }

        exec(code, env, env)

        fn_name = problem["function_name"]
        fn = env.get(fn_name)
        if not callable(fn):
            raise RuntimeError(f"{fn_name} not defined")

        score = 0.0
        total = 0.0
        tol = problem["tolerance"]

        for case in problem["cases"]:
            w = self.weights.get(case["type"], 1.0)
            total += w
            try:
                result = fn(**case["input"])
                expected = case["output"]
                if isinstance(expected, list):
                    ok = np.allclose(result, expected, atol=tol, rtol=0)
                elif isinstance(expected, bool):
                    ok = result is expected
                else:
                    ok = abs(result - expected) <= tol
                if ok:
                    score += w
            except Exception:
                pass

        return score, total

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
        except ValidationError:
            await updater.reject(
                new_agent_text_message("Invalid request format (expected EvalRequest)")
            )
            return

        ok, msg = self.validate_request(request)
        if not ok:
            await updater.reject(new_agent_text_message(msg))
            return

        problems = self.benchmark["problems"]
        purple_url = str(request.participants["purple"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Running Reflena scientific benchmark ({len(problems)} problems)..."
            ),
        )

        total_score = 0.0
        total_possible = 0.0
        details = []

        for idx, problem in enumerate(problems, start=1):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{idx}/{len(problems)}] Evaluating {problem['problem']}"
                ),
            )

            prompt = f"""
You are implementing a scientific computing function.

Problem description:
{problem['description']}

Function signature:
def {problem['function_name']}({problem['signature']}):

Constraints:
- Do not import libraries
- Do not print anything
- Return ONLY valid Python code
- Numerical stability required

Return ONLY the function implementation.
"""

            try:
                code = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=purple_url,
                    new_conversation=True,
                )
                score, possible = self.execute_candidate(code, problem)
            except Exception:
                score = 0.0
                possible = sum(self.weights.get(c["type"], 1.0) for c in problem["cases"])

            total_score += score
            total_possible += possible

            details.append({
                "problem": problem["problem"],
                "score": score,
                "total": possible,
            })

        accuracy = total_score / total_possible if total_possible > 0 else 0.0

        await updater.add_artifact(
            name="Result",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "benchmark": self.benchmark["benchmark"],
                            "num_problems": len(details),
                            "score": total_score,
                            "total": total_possible,
                            "accuracy": round(accuracy * 100, 2),
                            "details": details,
                        }
                    )
                )
            ],
        )

        await updater.complete()
