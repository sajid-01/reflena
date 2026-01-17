from typing import Any
import json
import numpy as np
from pathlib import Path
import math
import asyncio

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

    PURPLE_TIMEOUT = 30.0

    def __init__(self):
        self.messenger = Messenger()
        self.benchmark = self.load_benchmark()
        self.weights = {
            "core": 1.0,
            "edge": 1.25,
            "noisy": 1.5,
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

    async def execute_candidate_debug(
        self,
        code: str,
        problem: dict,
        updater: TaskUpdater,
    ) -> tuple[float, float]:
        env = {
            "__builtins__": __builtins__,
            "math": math,
            "np": np,
            "numpy": np,
        }

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("DEBUG exec starting")
        )

        try:
            exec(code, env, env)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("DEBUG exec succeeded")
            )
        except Exception as e:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"DEBUG exec failed: {repr(e)}")
            )
            return 0.0, 0.0

        fn_name = problem["function_name"]
        fn = env.get(fn_name)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"DEBUG function lookup: {fn_name} -> {fn}")
        )

        if not callable(fn):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("DEBUG function not callable")
            )
            return 0.0, 0.0

        score = 0.0
        total = 0.0
        tol = problem["tolerance"]

        for case in problem["cases"]:
            w = self.weights.get(case["type"], 1.0)
            total += w

            try:
                result = fn(**case["input"])
                expected = case["output"]

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"DEBUG input={case['input']} result={result} expected={expected}"
                    ),
                )

                if isinstance(expected, list):
                    ok = np.allclose(result, expected, atol=tol, rtol=0)
                elif isinstance(expected, bool):
                    ok = result is expected
                else:
                    ok = abs(result - expected) <= tol

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"DEBUG comparison ok={ok}")
                )

                if ok:
                    score += w

            except Exception as e:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"DEBUG exception during call: {repr(e)}")
                )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"DEBUG problem score={score} total={total}")
        )

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

            possible = sum(
                self.weights.get(c["type"], 1.0) for c in problem["cases"]
            )

            try:
                code = await asyncio.wait_for(
                    self.messenger.talk_to_agent(
                        message=prompt,
                        url=purple_url,
                        new_conversation=True,
                    ),
                    timeout=self.PURPLE_TIMEOUT,
                )

                score, _ = await self.execute_candidate_debug(
                    code, problem, updater
                )

            except asyncio.TimeoutError:
                score = 0.0

            except Exception as e:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"DEBUG outer exception: {repr(e)}")
                )
                score = 0.0

            total_score += score
            total_possible += possible

            details.append({
                "problem": problem["problem"],
                "score": score,
                "total": possible,
            })

        accuracy = total_score / total_possible if total_possible > 0 else 0.0
        accuracy = round(accuracy * 100, 2)

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
                            "accuracy": accuracy,
                            "details": details,
                        }
                    )
                )
            ],
        )

        await updater.complete()
