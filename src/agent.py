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


    def execute_candidate(self, code: str, problem: dict) -> tuple[int, int]:
        env = {
            "__builtins__": {},
            "np": np,
            "numpy": np,
            "math": math,
        }

        # Execute candidate code
        exec(code, env, env)

        fn_name = problem["function_name"]
        if fn_name not in env or not callable(env[fn_name]):
            raise RuntimeError(f"{fn_name} not defined")

        fn = env[fn_name]

        passed = 0
        total = len(problem["inputs"])
        tol = problem["tolerance"]

        for inp, expected in zip(problem["inputs"], problem["outputs"]):
            try:
                result = fn(**inp)

                if isinstance(expected, list):
                    ok = np.allclose(result, expected, atol=tol)
                elif isinstance(expected, bool):
                    ok = result is expected
                else:
                    ok = abs(result - expected) <= tol

                if ok:
                    passed += 1
            except Exception:
                pass

        return passed, total
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        # Parse request
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
                f"Running Reflena scientific benchmark "
                f"({len(problems)} problems)..."
            ),
        )

        total_passed = 0
        total_tests = 0
        details = []

        for idx, problem in enumerate(problems, start=1):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{idx}/{len(problems)}] "
                    f"Evaluating {problem['problem']}"
                ),
            )

            prompt = f"""
You are implementing a scientific computing function.

Problem:
{problem['problem']}

Function signature:
def {problem['function_name']}( ... ):

Constraints:
- Do not import any libraries
- Do not print anything
- Return ONLY valid Python code
- The function must be numerically correct and stable

Return ONLY the Python function implementation.
"""

            try:
                code = await self.messenger.talk_to_agent(
                    message=prompt,
                    url=purple_url,
                    new_conversation=True,
                )

                passed, total = self.execute_candidate(code, problem)
            except Exception:
                passed = 0
                total = len(problem["inputs"])

            total_passed += passed
            total_tests += total

            details.append({
                "problem": problem["problem"],
                "passed": passed,
                "total": total,
            })

        accuracy = total_passed / total_tests if total_tests > 0 else 0.0

        await updater.add_artifact(
            name="Result",
            parts=[
                Part(
                    root=DataPart(
                        data={
                            "benchmark": self.benchmark["benchmark"],
                            "num_problems": len(details),
                            "passed": total_passed,
                            "total": total_tests,
                            "accuracy": accuracy,
                            "details": details,
                        }
                    )
                )
            ],
        )

        await updater.complete()
